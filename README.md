# DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization</sub>

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2507.12933-b31b1b)](https://arxiv.org/abs/2507.12933) -->
<!-- [![GitHub Stars](https://img.shields.io/github/stars/ModelTC/TFMQ-DM.svg?style=social&label=Star&maxAge=60)](https://github.com/ModelTC/TFMQ-DM) -->

![Teaser image](./assets/stable_diffusion_grid.png)

**DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization**<br>
[Dongyeun Lee](https://leedongyeun.github.io/), [Jiwan Hur](https://jiwanhur.github.io/), [Hyounguk Shon](https://hushon.github.io/), [Jae Young Lee](https://scholar.google.com/citations?user=f6VnfxcAAAAJ&hl=en), [Junmo Kim](http://siit.kaist.ac.kr/)<br>
https://arxiv.org/abs/2507.12933

>**Abstract**: 
Diffusion models have achieved remarkable success in image generation but come with significant computational costs, posing challenges for deployment in resource-constrained environments. Recent post-training quantization (PTQ) methods have attempted to mitigate this issue by focusing on the iterative nature of diffusion models. However, these approaches often overlook outliers, leading to degraded performance at low bit-widths. In this paper, we propose a DMQ which combines Learned Equivalent Scaling (LES) and channel-wise Power-of-Two Scaling (PTS) to effectively address these challenges. Learned Equivalent Scaling optimizes channel-wise scaling factors to redistribute quantization difficulty between weights and activations, reducing overall quantization error. Recognizing that early denoising steps, despite having small quantization errors, crucially impact the final output due to error accumulation, we incorporate an adaptive timestep weighting scheme to prioritize these critical steps during learning. Furthermore, identifying that layers such as skip connections exhibit high inter-channel variance, we introduce channel-wise Power-of-Two Scaling for activations. To ensure robust selection of PTS factors even with small calibration set, we introduce a voting algorithm that enhances reliability. Extensive experiments demonstrate that our method significantly outperforms existing works, especially at low bit-widths such as W4A6 (4-bit weight, 6-bit activation) and W4A8, maintaining high image generation quality and model stability.


## Requirements
To set up the environment for running the code from our paper, you have two options.

#### 1. Using Docker (Recommended)
```bash
bash scripts/build_docker.sh
```
This image is based on an NVIDIA PyTorch container and includes all the necessary dependencies.

#### 2. Manual Python Environment Setup
If you prefer not to use Docker, you can set up the environment manually using the requirements.txt. We developed our code using PyTorch version 2.3.0. Please note that pytorch is not included in the requirements.txt file, so you will need to install it separately first.

```bash
pip install torch==2.3.0
pip install -r requirements.txt
```

After setting up the environment, run the following commands.
```bash
cd ./stable-diffusion
pip install -e .
```

## Original Models
Before quantization, you need to download the pre-trained weights. More detailed information can be found on [stable-diffusion](https://github.com/CompVis/stable-diffusion).
#### LDM

```bash
cd ./stable-diffusion/
sh ./scripts/download_first_stages.sh
sh ./scripts/download_models.sh
mkdir -p models/ldm/cin256-v2/
wget -O models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
cd ../
```

#### Stable Diffusion

```bash
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
mkdir -p ./stable-diffusion/models/ldm/stable-diffusion-v1/
mv sd-v1-4.ckpt ./stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt
```

## Quantization
We have provided shell scripts to run the quantization experiments for each dataset. These scripts are located in the `scripts/` directory and follow the naming convention `run_<dataset>.sh`.

To run the experiment for a specific dataset, simply execute the corresponding script from the root directory of the repository.

For example, to run the experiment for the **LSUN-Bedrooms** dataset:

```bash
# Execute the provided shell script
sh scripts/run_beds.sh
```

The shell script above is a simplified way to run the core Python command. For a better understanding of the various options and arguments, here is the full Python command used for the LSUN-Bedrooms experiment:

```bash
# LSUN-Bedrooms
python run_quant.py \
	--task ldm \
	--seed 123 \
	--logdir <PATH/TO/SAVE/LOG> \
	--resume ./stable-diffusion/models/ldm/lsun_beds256/model.ckpt \
	--cali_data_path <PATH/TO/SAVE/CALIBRATION/DATA> \
	--ptq \
	--run_quant \
	--use_aq \
	--w_bit 4 \
	--a_bit <6 OR 8> \
	--dynamic \
	--use_scale \
	--use_split \
	--iters_scale 6000 \
	--layerwise_recon \
	--loss_weight_type focal \
	--r 20 \
	--ratio_threshold 0.85 \
	--ptf_layers "skip_connection"
```
**Note**: Please replace the placeholder values like `<PATH/TO/SAVE/LOG>` and `<PATH/TO/SAVE/CALIBRATION/DATA>` with your desired file paths. For the a_bit argument, you can choose to set the value to either **6** or **8**.

## Inference

After the quantization, you can run the inference with the provided shell scripts, which are located in the `scripts/` directory. Each script is set up to run on a specific dataset and follows the naming convention `sample_<dataset>.sh`.

For example, to generate samples from the **LSUN-Bedrooms** dataset, simply run the following script:
```bash
# Execute the provided shell script
sh scripts/sample_beds.sh
```

For a more detailed look at the parameters, here is the full Python command executed by the shell script. This command uses the quantized model you have already trained to generate new samples.
```bash
# LSUN-Bedrooms
python sample_ldm.py \
	--task ldm \
	--seed 123 \
	--logdir <PATH/TO/SAVE/RESULT> \
	--resume ./stable-diffusion/models/ldm/lsun_beds256/model.ckpt \
	--load_quant <PATH/TO/LOAD/QUANTIZED/MODEL> \
	--ptq \
	--use_aq \
	--w_bit 4 \
	--a_bit <6 OR 8> \
	--use_scale \
	--use_split \
	--n_samples 50000 \
	--custom_steps 20 \
	--batch_size 32
```
**Note**: Be sure to replace the placeholders, such as `<PATH/TO/SAVE/RESULT>` and `<PATH/TO/LOAD/QUANTIZED/MODEL>`, with your actual file paths. The a_bit argument can be set to either **6** or **8**, depending on your quantization configuration.

## Citation
```
@inproceedings{lee2025dmq,
  title={DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization},
  author={Lee, Dongyeun and Hur, Jiwan and Shon, Hyounguk and Lee, Jae Young and Kim, Junmo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={18510--18520},
  year={2025}
}
```

## Acknowledgments

Our code is directly built upon the [TFMQ-DM](https://github.com/ModelTC/TFMQ-DM) project. We also gratefully acknowledge [stable-diffusion](https://github.com/CompVis/stable-diffusion) and [Q-Diffusion](https://github.com/Xiuyu-Li/q-diffusion), which provided the foundational code that TFMQ-DM was developed from.