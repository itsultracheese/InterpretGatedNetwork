#!/bin/bash

python run.py \
    --model $1 \
    --dnn_type FCN \
    --dataset $2 \
    --train_epochs 300 \
    --batch_size 32 \
    --lr 5e-3 \
    --dropout 0. \
    --num_shapelet $3 \
    --lambda_div $4 \
    --lambda_reg $5 \
    --epsilon $6 \
    --beta_schedule $7 \
    --seed $8 \
    --gating_value $9 \
    --amp