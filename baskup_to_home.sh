#!/bin/bash
# ============================================================
# 在 /scratch 中保留未压缩数据
# 在 /home 中保存压缩包（删除 scratch 中的压缩副本）
# ============================================================

# 路径设置
HOME_PROCESSED="/home/yz54720/Projects/Method/deepconf/data/processed"
SCRATCH_PROCESSED="/scratch/yz54720/deepconf_processed"

# 要处理的子目录
DIRS=("aime_2024" "aime_2025" "brumo_2025" "hhmt_2025")

# 压缩设置
COMPRESSOR="zstd"
EXT="tar.zst"

echo "🚀 开始同步与压缩任务"
date
echo "-------------------------------------------"

# 1️⃣ 确保 scratch 存在
mkdir -p "$SCRATCH_PROCESSED"

# 2️⃣ 遍历各子目录
for d in "${DIRS[@]}"; do
    SRC="${HOME_PROCESSED}/${d}"
    SCRATCH_DST="${SCRATCH_PROCESSED}/${d}"
    HOME_OUT="${HOME_PROCESSED}/${d}.${EXT}"
    SCRATCH_TMP="${SCRATCH_PROCESSED}/${d}.${EXT}"

    if [ -d "$SRC" ]; then
        echo "📂 处理目录: ${d}"

        # Step 1: 同步到 /scratch（仅更新）
        echo "➡️ 同步 ${SRC} → ${SCRATCH_PROCESSED}"
        rsync -ah --progress "$SRC" "$SCRATCH_PROCESSED/"

        # Step 2: 在 /scratch 压缩
        echo "🗜️ 压缩 ${SCRATCH_DST} → ${SCRATCH_TMP}"
        tar --use-compress-program="${COMPRESSOR}" -cf "${SCRATCH_TMP}" -C "${SCRATCH_PROCESSED}" "${d}"

        # Step 3: 拷贝压缩包回 home
        echo "⬆️ 拷贝压缩包到 ${HOME_PROCESSED}"
        mv "${SCRATCH_TMP}" "${HOME_OUT}"

        # Step 4: 删除 /scratch 中的压缩文件（保留原目录）
        echo "🧹 删除 /scratch 压缩文件 ${SCRATCH_TMP}"
        rm -f "${SCRATCH_TMP}"

        echo "✅ 完成 ${d}"
    else
        echo "⚠️ 未找到目录: ${SRC}"
    fi
    echo "-------------------------------------------"
done

echo "🎉 全部完成"
date
