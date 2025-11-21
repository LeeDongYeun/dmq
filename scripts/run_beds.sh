#!/bin/bash

WQ_BIT=4
AQ_BIT=8
ITERS_SCALE=6000
R=20

SEED=123
TASK=ldm
DATASET=lsun_beds256

LOGDIR="./results/W${WQ_BIT}A${AQ_BIT}/"
RESUME="./stable-diffusion/models/ldm/${DATASET}/model.ckpt"
# CALI_DATA_PATH="./cali_data/${DATASET}/cali_data.npz"
CALI_DATA_PATH="../../jupyter/cali_data/${DATASET}/cali_data.npz"

python run_quant.py \
    --task $TASK \
    --seed $SEED \
    --logdir $LOGDIR \
    --resume $RESUME \
    --cali_data_path $CALI_DATA_PATH \
    --ptq \
    --run_quant \
	--use_aq \
    --w_bit $WQ_BIT \
    --a_bit $AQ_BIT \
    --dynamic \
    --use_scale \
    --use_split \
    --iters_scale $ITERS_SCALE \
    --layerwise_recon \
    --loss_weight_type focal \
    --r $R \
    --ratio_threshold 0.85 \
    --ptf_layers "skip_connection" #\
    # --load_quant '../results/ours/W4A8/lsun_beds256/2024_1019_0250_33_ScaleTWFocalR20/quant_sd.pth'