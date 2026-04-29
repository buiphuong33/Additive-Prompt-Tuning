#!/bin/bash

# experiment settings
DATASET=cifar-100
N_CLASS=100

# Chỉnh GPUID về 0 (Kaggle dùng GPU 0)
GPUID='0'
CONFIG=configs/cifar-100_prompt.yaml
REPEAT=1
OVERWRITE=0

# hyperparameter arrays
LR=0.004
SCHEDULE=30
EMA_COEFF=0.7
SEED_LIST=(1)

# Additional loss weights
ORTHOGONAL_WEIGHT=0.1
CONTRASTIVE_WEIGHT=1.0
TEMPERATURE=0.1
PROMPT_TOP_K=3

DELAY_BETWEEN_EXPERIMENTS=10

# SỬA LỖI TẠO THƯ MỤC: Tạo cả thư mục con theo DATASET
LOG_DIR="logs/${DATASET}"
mkdir -p "$LOG_DIR"

for seed in "${SEED_LIST[@]}"
do
    OUTDIR="./checkpoints/${DATASET}/seed${seed}"
    mkdir -p "$OUTDIR"

    LOG_FILE="${LOG_DIR}/seed${seed}.log"

    echo "Starting experiment with seed=$seed"
    
    # Bỏ nohup và & để chạy tuần tự trên Kaggle cho dễ theo dõi log
    python -u run.py \
        --config $CONFIG \
        --gpuid $GPUID \
        --repeat $REPEAT \
        --overwrite $OVERWRITE \
        --learner_type prompt \
        --learner_name APT_Learner \
        --prompt_param 0.01 \
        --lr $LR \
        --seed $seed \
        --ema_coeff $EMA_COEFF \
        --schedule $SCHEDULE \
        --orthogonal_weight $ORTHOGONAL_WEIGHT \
        --contrastive_weight $CONTRASTIVE_WEIGHT \
        --temperature $TEMPERATURE \
        --prompt_top_k $PROMPT_TOP_K \
        --log_dir ${OUTDIR} 2>&1 | tee "$LOG_FILE"

    if [ $? -eq 0 ]; then
        echo "Experiment completed successfully"
    else
        echo "Experiment failed"
    fi

    rm -rf ${OUTDIR}/models
    echo "----------------------------------------"
    echo "Waiting for $DELAY_BETWEEN_EXPERIMENTS seconds..."
    sleep $DELAY_BETWEEN_EXPERIMENTS
done

echo "All experiments completed!"
exit 0