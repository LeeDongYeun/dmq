from typing import List, Dict
import torch.nn as nn
import torch

from ldm.modules.diffusionmodules.util import timestep_embedding
from ldm.modules.diffusionmodules.openaimodel import (UNetModel, TimestepEmbedSequential, 
                                                      Upsample, Downsample, ResBlock, 
                                                      AttentionBlock, QKMatMul,SMVMatMul,)
from ldm.modules.attention import GEGLU, FeedForward, CrossAttention, BasicTransformerBlock, SpatialTransformer

from models.quant_ldm_blocks import (QuantTimestepEmbedSequential, BaseQuantBlock, 
                                     QuantUpsample, QuantDownsample, QuantResBlock,
                                     QuantAttentionBlock, QuantQKMatMul, QuantSMVMatMul, 
                                     QuantGEGLU, QuantFeedForward, QuantCrossAttention,
                                     QuantBasicTransformerBlock, QuantSpatialTransformer)
from quant.quant_layer import QMODE, QuantLayer
from quant.utils import get_op_by_name, get_op_name, set_op_by_name

class QuantModel(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 wq_params: dict = {},
                 aq_params: dict = {},
                 cali: bool = True,
                 use_scale: bool = False,
                 use_split: bool = False,
                 bound_range: float = 1.,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.model = model
        self.wq_params = wq_params
        self.aq_params = aq_params
        self.use_scale = use_scale
        self.use_split = use_split
        self.softmax_a_bit = kwargs.get("softmax_a_bit", 8)
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        
        self.quant_module(
            self.model, 
            wq_params, aq_params, 
            aq_mode=kwargs.get("aq_mode", [QMODE.NORMAL.value]), 
            use_scale=use_scale,
            bound_range=bound_range
        )
        self.quant_block(aq_params)
        self.disable_out_quantization()

    def quant_module(self,
                     module: nn.Module,
                     wq_params: dict = {},
                     aq_params: dict = {},
                     aq_mode: List[int] = [QMODE.NORMAL.value],
                     use_scale=False,
                     bound_range=1.0,
                     ) -> None:
        for name, child in module.named_children():
            full_name = get_op_name(self.model, child)
            if (name == 'time_embed' or  name == 'emb_layers'):
                continue
            
            cur_aq_params = aq_params
            if aq_params['symmetric'] and ('in_layers' in full_name or 'out_layers' in full_name):
                cur_aq_params = aq_params.copy()
                cur_aq_params['symmetric'] = False
                cur_aq_params['after_silu'] = True

            if isinstance(child, tuple(QuantLayer.QMAP.keys())):
                setattr(module, name, QuantLayer(child, wq_params, cur_aq_params, aq_mode=aq_mode, 
                                                 use_scale=use_scale, bound_range=bound_range))
            else:
                self.quant_module(child, wq_params, aq_params, aq_mode=aq_mode, 
                                  use_scale=use_scale, bound_range=bound_range)

    def quant_block(self, aq_params):
        for name, module in self.model.named_modules():
            # LDM
            if isinstance(module, TimestepEmbedSequential):
                qmodule = QuantTimestepEmbedSequential(*module.children())
            elif isinstance(module, Upsample):
                qmodule = QuantUpsample(module, aq_params)
            elif isinstance(module, Downsample):
                qmodule = QuantDownsample(module, aq_params)
            elif isinstance(module, ResBlock):
                qmodule = QuantResBlock(module, aq_params)
            elif isinstance(module, AttentionBlock):
                qmodule = QuantAttentionBlock(module, aq_params)
            elif isinstance(module, QKMatMul):
                qmodule = QuantQKMatMul(aq_params)
            elif isinstance(module, SMVMatMul):
                qmodule = QuantSMVMatMul(aq_params, softmax_a_bit=self.softmax_a_bit)
            elif isinstance(module, GEGLU):
                qmodule = QuantGEGLU(module, aq_params)
            elif isinstance(module, FeedForward):
                qmodule = QuantFeedForward(module, aq_params)
            elif isinstance(module, CrossAttention):
                qmodule = QuantCrossAttention(module, aq_params, softmax_a_bit=self.softmax_a_bit)
            elif isinstance(module, BasicTransformerBlock):
                qmodule = QuantBasicTransformerBlock(module, aq_params, softmax_a_bit=self.softmax_a_bit)
            elif isinstance(module, SpatialTransformer):
                qmodule = QuantSpatialTransformer(module, aq_params)
            else:
                continue
            set_op_by_name(self.model, name, qmodule)

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.model.modules():
            if isinstance(m, (BaseQuantBlock, QuantLayer)):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)
    
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.model.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        emb_outs = []
        t_emb = timestep_embedding(timesteps, self.model.model_channels, repeat_only=False)
        emb = self.model.time_embed(t_emb)

        if self.model.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.model.label_emb(y)

        h = x.type(self.model.dtype)
        emb_out = None
        for module in self.model.input_blocks:
            h, emb_out = module(h, emb, context, timesteps, emb_out)
            hs.append(h)
            if emb_out is not None:
                emb_outs.append(emb_out)
        h, emb_out = self.model.middle_block(h, emb, context, timesteps, emb_out)
        for module in self.model.output_blocks:
            split = h.shape[1] if self.use_split else 0
            h = torch.cat([h, hs.pop()], dim=1)
            if len(emb_outs) == 0:
                emb_out = None
            else:
                emb_out = torch.cat([emb_out, emb_outs.pop()], dim=1)
            h, emb_out = module(h, emb, context, t=timesteps, prev_emb_out=emb_out, split=split)
        h = h.type(x.dtype)
        if self.model.predict_codebook_ids:
            return self.model.id_predictor(h)
        else:
            return self.model.out(h)

    def disable_out_quantization(self) -> None:
        modules = []
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                modules.append(m)
        modules: List[QuantLayer]
        # disable the last layer and the first layer
        modules[0].use_wq = False
        modules[0].ignore_recon = True
        modules[0].disable_aq = True

        # modules[2].use_wq = False
        # modules[2].ignore_recon = True
        # modules[2].disable_aq = True

        modules[-1].use_wq = False
        modules[-1].ignore_recon = True
        modules[-1].disable_aq = True

        if isinstance(self.model, UNetModel):
            # Disable activation quantization of attention
            for name, module in self.model.named_modules():
                if isinstance(module, (QuantQKMatMul, QuantSMVMatMul, QuantCrossAttention)):
                    module.disable_aq = True
        
        # pdb.set_trace()

    def set_grad_ckpt(self, grad_ckpt: bool) -> None:
        for _, module in self.model.named_modules():
            if isinstance(module, (QuantBasicTransformerBlock, BasicTransformerBlock)):
                module.checkpoint = grad_ckpt

    def set_running_stat(self,
                         running_stat: bool = False
                         ) -> None:
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                m.attn1.aqtizer_q.running_stat = running_stat
                m.attn1.aqtizer_k.running_stat = running_stat
                m.attn1.aqtizer_v.running_stat = running_stat
                m.attn1.aqtizer_w.running_stat = running_stat
                m.attn2.aqtizer_q.running_stat = running_stat
                m.attn2.aqtizer_k.running_stat = running_stat
                m.attn2.aqtizer_v.running_stat = running_stat
                m.attn2.aqtizer_w.running_stat = running_stat
            elif isinstance(m, QuantQKMatMul):
                m.aqtizer_q.running_stat = running_stat
                m.aqtizer_k.running_stat = running_stat
            elif isinstance(m, QuantSMVMatMul):
                m.aqtizer_v.running_stat = running_stat
                m.aqtizer_w.running_stat = running_stat
            elif isinstance(m, QuantLayer):
                m.set_running_stat(running_stat)