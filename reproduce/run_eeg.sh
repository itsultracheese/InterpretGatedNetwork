#!/bin/bash

# Model Configs, default settings, change as needed
# MODEL=InterpGN
MODEL=SBM
DNN_TYPE=FCN
NUM_SHAPELET=10
LAMBDA_DIV=0.1
LAMBDA_REG=0.1
EPS=1
BETA_SCHEDULE=constant
GATING_VALUE=1
DATA_ROOT=/media/ssd-3t/amiftakhova/bimai


EEG_DATASETS=(
    "FaceDetection"
    "FingerMovements"
    "SelfRegulationSCP1"
    "SelfRegulationSCP2"
    "MotorImagery"
)
# SEEDS=(8237 2023 8237 2023 8237)
SEEDS=(0)

for dataset in ${EEG_DATASETS[@]}; do
    for seed in ${SEEDS[@]}; do
        python run.py \
            --model $MODEL \
            --dnn_type $DNN_TYPE \
            --data_root $DATA_ROOT \
            --dataset $dataset \
            --train_epochs 500 \
            --batch_size 32 \
            --lr 5e-3 \
            --dropout 0. \
            --num_shapelet $NUM_SHAPELET \
            --lambda_div $LAMBDA_DIV \
            --lambda_reg $LAMBDA_REG \
            --epsilon $EPS \
            --beta_schedule $BETA_SCHEDULE \
            --seed $seed \
            --gating_value $GATING_VALUE \
            --amp
    done
done
