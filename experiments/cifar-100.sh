#!/bin/bash

# experiment settings
DATASET=CIFAR100
CONFIG=configs/config.yaml

# hard coded inputs
GPUID='0'

REPEAT=1
OVERWRITE=0

# hyperparameter arrays
LR=0.004
SCHEDULE=30
EMA_COEFF=0.7
SEED_LIST=(1 2 3)


# Create log directory
LOG_DIR="logs/${DATASET}"
mkdir -p $LOG_DIR

for seed in "${SEED_LIST[@]}"
    do
        # save directory
        OUTDIR="./checkpoints/${DATASET}/seed${seed}"
        mkdir -p $OUTDIR

        # Create unique log file name
        LOG_FILE="${LOG_DIR}/${DATASET}/seed${seed}.log"

        echo "Starting experiment with seed=$seed"
        
        python -u run.py \
            --config $CONFIG \
            --dataset $DATASET \
            --gpuid $GPUID \
            --repeat $REPEAT \
            --overwrite $OVERWRITE \
            --learner_type prompt \
            --learner_name APT_Learner \
            --prompt_param "100" "0.01" \
            --lr $LR \
            --seed $seed \
            --ema_coeff $EMA_COEFF \
            --schedule $SCHEDULE \
            --log_dir ${OUTDIR} 2>&1 | tee "$LOG_FILE"

        
        
        # Check if process completed successfully
        if [ $? -eq 0 ]; then
            echo "Experiment with seed $seed completed successfully"
        else
            echo "Experiment with seed $seed failed. Check $LOG_FILE for details."
        fi

        
        echo "----------------------------------------"
        sleep 5  # Short delay to ensure logs are flushed before next experiment starts
        
        
    done

echo "All experiments completed!"
