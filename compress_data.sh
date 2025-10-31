#!/bin/bash

MAX_JOBS=4
current_jobs=0

for dataset in generated/*/; do
    dataset=${dataset%/}
    # 获取目录名作为压缩文件名
    tar_name="${dataset}.tar.gz"

    # 压缩当前数据集到tar.gz（后台运行）
    echo "Compressing ${dataset} to ${tar_name}..."
    tar -czf "${tar_name}" -C "generated" "$(basename "$dataset")" &

    # 增加当前作业计数
    ((current_jobs++))

    # 如果达到最大并发数，等待任一后台作业完成
    if (( current_jobs >= MAX_JOBS )); then
        wait -n
        ((current_jobs--))
    fi
done

# 等待所有剩余后台作业完成
wait

echo "所有数据集压缩完成"