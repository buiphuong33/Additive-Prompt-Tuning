#!/bin/bash

# experiment settings
DATASET=ImageNet_R
N_CLASS=200

# Chỉnh GPUID về 0 (Kaggle dùng GPU 0)
GPUID='0'
CONFIG=configs/imnet-r_prompt.yaml
REPEAT=1
OVERWRITE=0

# hyperparameter arrays
LR=0.003
SCHEDULE=30
EMA_COEFF=0.8
SEED_LIST=(1)

# Cực kỳ quan trọng: Phải có 3 thông số này để APT hoạt động đúng
# "Số_token Dropout Hidden_dim"
PROMPT_P="10 0.1 768"

# Set delay between experiments
DELAY_BETWEEN_EXPERIMENTS=10

# Sửa lỗi tạo thư mục log: tạo folder dataset trước
LOG_DIR="logs/${DATASET}"
mkdir -p "$LOG_DIR"

for seed in "${SEED_LIST[@]}"
do
    # save directory
    OUTDIR="./checkpoints/${DATASET}/seed${seed}"
    mkdir -p "$OUTDIR"

    # Đường dẫn file log
    LOG_FILE="${LOG_DIR}/seed${seed}.log"

    echo "Starting ImageNet-R experiment with seed=$seed"
    
    # BỎ nohup và &, dùng tee để xem log trực tiếp trên Kaggle
    python -u run.py \
        --config $CONFIG \
        --gpuid $GPUID \
        --repeat $REPEAT \
        --overwrite $OVERWRITE \
        --learner_type prompt \
        --learner_name APT_Learner \
        --prompt_param $PROMPT_P \
        --lr $LR \
        --seed $seed \
        --ema_coeff $EMA_COEFF \
        --schedule $SCHEDULE \
        --log_dir ${OUTDIR} 2>&1 | tee "$LOG_FILE"

    # Kiểm tra trạng thái kết thúc
    if [ $? -eq 0 ]; then
        echo "Experiment completed successfully"
    else
        echo "Experiment failed"
    fi

    # Xóa models để tiết kiệm bộ nhớ Kaggle (chỉ giữ log và kết quả)
    rm -rf ${OUTDIR}/models
    
    echo "----------------------------------------"
    echo "Waiting for $DELAY_BETWEEN_EXPERIMENTS seconds..."
    sleep $DELAY_BETWEEN_EXPERIMENTS
done

echo "All ImageNet-R experiments completed!"
exit 0