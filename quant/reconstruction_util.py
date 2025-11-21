import torch
from collections import defaultdict
from enum import Enum
from quant.quant_layer import QMODE, QuantLayer
from quant.quant_utils import lp_loss
from quant.quantizer import AdaRoundQuantizer
from models.quant_ldm_blocks import BaseQuantBlock
from typing import Union
import logging
import pdb
logger = logging.getLogger(__name__)

RLOSS = Enum('RLOSS', ('RELAXATION', 'MSE', 'FISHER_DIAG', 'FISHER_FULL', 'NONE'))
print_freq = 2000

class LossFunc:
    def __init__(self,
                 o: Union[QuantLayer, BaseQuantBlock],
                 round_loss: RLOSS = RLOSS.RELAXATION,
                 w: float = 1.,
                 rec_loss: RLOSS = RLOSS.MSE,
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 r: float = 0,
                 momentum: float = 0.95,
                 loss_weight_type='focal',
                 ) -> None:
        self.o = o
        self.round_loss = round_loss
        self.w = w
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p 
        self.temp_decay = LinearTempDecay(t_max = max_count, 
                                          rel_start_decay = warmup + (1 - warmup) * decay_start,
                                          start_b = b_range[0], 
                                          end_b = b_range[1])
        self.count = 0
        self.loss_func = LossFuncTimeWeighted(r=r, rec_loss=rec_loss, p=p, momentum=momentum, 
                                              loss_weight_type=loss_weight_type)
        self.loss_func.count = 1e-1

    def __call__(self, 
                 pred: torch.Tensor, 
                 tgt: torch.Tensor, 
                 t: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        # rec_loss = self.loss_func(pred, tgt, t)
        rec_loss = lp_loss(pred, tgt, p=self.p)

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == RLOSS.NONE:
            b = round_loss = 0
        elif self.round_loss == RLOSS.RELAXATION:
            if isinstance(self.o, QuantLayer):
                round_vals: torch.Tensor = self.o.wqtizer.get_soft_tgt()
                round_loss = self.w * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            else:
                self.o: BaseQuantBlock
                round_loss = 0
                for _, module in self.o.named_modules():
                    if isinstance(module, QuantLayer):
                        if not module.ignore_recon:
                            if getattr(module, 'split', 0) == 0 or QMODE.QDIFF.value not in module.aq_mode:
                                round_vals: torch.Tensor = module.wqtizer.get_soft_tgt()
                                round_loss += self.w * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
                            else:
                                round_vals: torch.Tensor = module.wqtizer.get_soft_tgt()
                                round_vals1: torch.Tensor = module.wqtizer1.get_soft_tgt()
                                round_loss += self.w * ((1 - ((round_vals - .5).abs() * 2).pow(b)).sum() * module.split \
                                    + (1 - ((round_vals1 - .5).abs() * 2).pow(b)).sum() * (module.w.shape[1] - module.split)) / module.w.shape[1]
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % print_freq == 0 or self.count == 1:
            logger.info('Total loss:\t{:.4f} (rec:{:.4f}, rec org: {:.4f}, round:{:.4f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss),  float(lp_loss(pred, tgt, p=self.p)), float(round_loss ** (1/(b+1e-5))), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, 
                 t_max: int, 
                 rel_start_decay: float = 0.2, 
                 start_b: int = 10, 
                 end_b: int = 2
                 ) -> None:
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t) -> float:
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class LossFuncTimeWeighted:
    def __init__(self,
                 r: float = 0,
                 rec_loss: RLOSS = RLOSS.MSE,
                 p: float = 2.,
                 momentum=0.95,
                 loss_weight_type='focal',
                 **kwargs) -> None:
        self.r = r
        self.rec_loss = rec_loss
        self.p = p 
        self.momentum = momentum
        self.loss_weight_type = loss_weight_type

        self.count = 0
        if self.loss_weight_type == 'focal':
            self.weight_dict = torch.zeros(20)
        elif self.loss_weight_type == 'linear':
            self.weight_dict = 1 + (1 - torch.tensor(range(0, 20)) / 20) * self.r
        else:
            NotImplementedError
        
    def __call__(self, 
                 pred: torch.Tensor, 
                 tgt: torch.Tensor, 
                 t: torch.Tensor) -> torch.Tensor:
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        self.weight_dict = self.weight_dict.to(pred.device)
        rec_loss = (pred - tgt).abs().pow(self.p).sum(1)
        rec_loss = rec_loss.mean(dim=[*range(1, len(rec_loss.shape))])  # Shape: [B]

        # time to index
        t_idx = (19 - (t - 1) // 50).to(int)  # Shape: [B]

        with torch.no_grad():
            # Update weight dict
            if self.loss_weight_type == 'focal':
                accumulated_updates = torch.scatter_add(torch.zeros_like(self.weight_dict), 0, t_idx, rec_loss.detach())
                self.weight_dict = self.weight_dict * self.momentum + accumulated_updates * (1 - self.momentum)

            # Get timestep-wise weight for loss
            weight = self.weight_dict[t_idx]    # Shape: [B]
            
            if self.loss_weight_type == 'focal':
                weight = (1 - weight / self.weight_dict.sum()) ** self.r

        total_loss = (weight * rec_loss).mean()
        if self.count % print_freq == 0 or self.count == 1:
            logger.info('Total loss:\t{:.4f} (rec:{:.4f}) \tcount={}'.format(
                float(total_loss), float(rec_loss.mean()), self.count))
            
        return total_loss