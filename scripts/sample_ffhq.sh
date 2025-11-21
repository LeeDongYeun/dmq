#!/bin/bash

N_SAMPLES=50000
CUSTOM_STEPS=20

WQ_BIT=4
AQ_BIT=8

SEED=123
TASK=ldm
DATASET=ffhq256

LOGDIR="./generated_samples/W${WQ_BIT}A${AQ_BIT}/"
RESUME="./stable-diffusion/models/ldm/${DATASET}/model.ckpt"
QUANT_CKPT=".results/W${WQ_BIT}A${AQ_BIT}/${DATASET}/2025_0515_1131_25/quant_sd.pth" # Your model ckpt path

python sample_ldm.py \
    --task $TASK \
    --seed $SEED \
  	--logdir $LOGDIR \
    --resume $RESUME \
  	--load_quant $QUANT_CKPT \
    --ptq \
	--use_aq \
    --w_bit $WQ_BIT \
  	--a_bit $AQ_BIT \
	--use_scale \
	--use_split \
	--n_samples $N_SAMPLES \
	--custom_steps $CUSTOM_STEPS \
	--batch_size 32