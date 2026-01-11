#!/bin/bash
#SBATCH --job-name=pooling_all
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/%A_%a.log
#SBATCH --array=0-19   # 集群限制：最多 0-19，一共 20 个 task

# ---- 1. 加载必要模块 ----
source ~/.bashrc
module load CUDA/12.8.0

# ---- 2. 激活虚拟环境 ----
source ~/PY_envs/deepconf_py312/bin/activate

# ---- 3. 进入项目目录 ----
cd ~/Projects/Method/deepconf

nvidia-smi

# ---- 4. 配置数据集 & 任务分配 ----
DATASETS=("aime_2024" "aime_2025" "brumo_2025" "hmmt_2025")
NUM_DATASETS=${#DATASETS[@]}        # 4
QUESTIONS_PER_DATASET=30            # 题号 0-29
TOTAL_JOBS=$(( NUM_DATASETS * QUESTIONS_PER_DATASET ))   # 120

ARRAY_TASKS=20                      # array 0-19 一共 20 个
JOBS_PER_TASK=$(( (TOTAL_JOBS + ARRAY_TASKS - 1) / ARRAY_TASKS ))  # 向上取整 = 6

TASK_ID=${SLURM_ARRAY_TASK_ID}
START_INDEX=$(( TASK_ID * JOBS_PER_TASK ))
END_INDEX=$(( START_INDEX + JOBS_PER_TASK - 1 ))

# 防止最后一个 task 越界
if [ "$END_INDEX" -ge "$TOTAL_JOBS" ]; then
    END_INDEX=$(( TOTAL_JOBS - 1 ))
fi

echo "This Slurm array task (ID=${TASK_ID}) will process global job indices ${START_INDEX}..${END_INDEX}"

# ---- 5. 依次跑分配给当前 array-task 的多个 (dataset, qid) ----
for GLOBAL_ID in $(seq "$START_INDEX" "$END_INDEX"); do
    DATASET_INDEX=$(( GLOBAL_ID / QUESTIONS_PER_DATASET ))
    QID=$(( GLOBAL_ID % QUESTIONS_PER_DATASET ))
    DATASET_NAME=${DATASETS[$DATASET_INDEX]}

    OUT_DIR="data/processed/${DATASET_NAME}/pool_information_v5"
    OUT_PATH="${OUT_DIR}/${DATASET_NAME}_${QID}_deepconflow_self_check.jsonl"

    mkdir -p "$OUT_DIR"

    if [ -f "$OUT_PATH" ]; then
        echo "✅ Skip: dataset=${DATASET_NAME}, QID=${QID}, file already exists at ${OUT_PATH}"
        continue
    fi

    echo "🚀 Running: dataset=${DATASET_NAME}, QID=${QID}"
    python examples/online_RL/approach_exploration/pool_information/pool_information.py \
        --question_id "$QID" \
        --output_dir "$OUT_DIR" \
        --data_name "$DATASET_NAME"

    echo "✅ Finished: dataset=${DATASET_NAME}, QID=${QID}"
done

echo "🎉 Slurm array task ${TASK_ID} done."
