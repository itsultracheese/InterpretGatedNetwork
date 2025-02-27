#!/bin/bash

# Model Configs
MODEL=InterpGN
DNN_TYPE=FCN
NUM_SHAPELET=10
LAMBDA_DIV=0.1
LAMBDA_REG=0.1
EPS=1
BETA_SCHEDULE=constant
GATING_VALUE=1


UEA_DATASETS=(
    # "ArticularyWordRecognition"
    # "AtrialFibrillation"
    "BasicMotions"
    # "CharacterTrajectories"
    # "LSST"
    # "ERing"
    # "Epilepsy"
    # "EthanolConcentration" 
    # "FaceDetection"
    # "FingerMovements"
    # "Handwriting"
    # "Heartbeat"
    # "InsectWingbeat"
    # "JapaneseVowels"
    # "Libras"
    # "NATOPS"
    # "PenDigits"
    # "RacketSports"
    # "SpokenArabicDigits" 
    # "UWaveGestureLibrary"
    # "Cricket" 
    # "PhonemeSpectra" 
    # "HandMovementDirection"
    # "SelfRegulationSCP1"
    # "SelfRegulationSCP2"
    # "StandWalkJump"
    # # Datasets that MAY cause high memory usage (many variates)
    # "PEMS-SF"
    # "DuckDuckGeese"
    # # Datasets with VERY LONG length and WILL cause high memory usage
    # "MotorImagery"
    # "EigenWorms"
)


# SEEDS=(0 42 1234 8237 2023)

SEEDS=(0)

for dataset in ${UEA_DATASETS[@]}; do
    for seed in ${SEEDS[@]}; do
        python run.py \
            --model $MODEL \
            --dnn_type $DNN_TYPE \
            --dataset $dataset \
            --train_epochs 300 \
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
