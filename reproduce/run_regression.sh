#!/bin/bash

# Model Configs, default settings, change as needed
MODEL=InterpGN
DNN_TYPE=FCN


DATASETS=(
    "BIDMC32HR"
    "BeijingPM10Quality"
    "BeijingPM25Quality"
    "BenzeneConcentration"
    "Covid3Month"
    "FloodModeling1"
    "HouseholdPowerConsumption1"
    "IEEEPPG"
    "LiveFuelMoistureContent"
    "NewsHeadlineSentiment"
)


SEEDS=(0 42 1234 8237 2023)

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        python run.py \
            --model $MODEL \
            --dnn_type $DNN_TYPE \
            --seed $seed \
            --dataset $dataset \
            --task_name regression \
            --data Monash \
            --data_root ./data/Monash_UEA_UCR_Regression_Archive
    done
done