#!/bin/bash
#SBATCH --job-name=data_creation_array
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=19:00:00
#SBATCH --output=logs/%A_%a.log
#SBATCH --array=0

# ---- 1. 加载必要模块 ----
module load CUDA/12.8.0

# ---- 2. 激活虚拟环境 ----
source ~/PY_envs/deepconf_py312/bin/activate

# ---- 3. 进入项目目录 ----
cd ~/Projects/Method/deepconf

nvidia-smi
echo "Starting SLURM array job ${SLURM_ARRAY_JOB_ID} with task ID ${SLURM_ARRAY_TASK_ID}"
date

# ======================
# 🔹 Trace 分批生成逻辑
# ======================
DATASET_NAME="hmmt_2025"
DATASET="/home/yz54720/Projects/Method/deepconf/data/raw/${DATASET_NAME}.jsonl"
QID=${SLURM_ARRAY_TASK_ID}
OUTDIR="/home/yz54720/Projects/Method/deepconf/data/processed/${DATASET_NAME}/traces"
mkdir -p ${OUTDIR}

TOTAL_TRACE=256
BATCH_SIZE=64  # 每批生成64条 trace
NUM_BATCHES=$((TOTAL_TRACE / BATCH_SIZE))

for ((i=0; i<NUM_BATCHES; i++)); do
    OUTFILE="${OUTDIR}/${DATASET_NAME}_${QID}_part${i}.jsonl"

    # 若文件已存在则跳过（支持断点续跑）
    if [ -f "${OUTFILE}" ]; then
        echo "[INFO] Skipping existing ${OUTFILE}"
        continue
    fi

    echo "[$(date)] Generating batch $i (QID=${QID})"

    # 添加错误处理
    if python examples/online_RL/data_creation.py \
        --dataset ${DATASET} \
        --qid ${QID} \
        --budget ${BATCH_SIZE} \
        --output_path ${OUTFILE}; then
        echo "[$(date)] Successfully finished batch $i"
    else
        echo "[ERROR] Failed to generate batch $i for QID=${QID}"
        exit 1
    fi
done

# ---- 5. 合并分批结果 ----
FINAL_FILE="${OUTDIR}/${DATASET_NAME}_${QID}_full.jsonl"
echo "[INFO] Merging all parts for QID=${QID}"

# 更安全的合并方式，只合并当前QID的文件
if ls ${OUTDIR}/${DATASET_NAME}_${QID}_part*.jsonl 1> /dev/null 2>&1; then
    for f in ${OUTDIR}/${DATASET_NAME}_${QID}_part*.jsonl; do
        cat "$f" >> ${FINAL_FILE}
        echo >> ${FINAL_FILE}
    done
    echo "[SUCCESS] Saved full trace set to ${FINAL_FILE}"
    
    # 清理临时文件
    echo "[INFO] Cleaning up temporary part files..."
    rm ${OUTDIR}/${DATASET_NAME}_${QID}_part*.jsonl
    echo "[INFO] Cleanup completed"
else
    echo "[WARNING] No part files found for QID=${QID}"
fi

date
echo "Finished SLURM task ID ${SLURM_ARRAY_TASK_ID}"