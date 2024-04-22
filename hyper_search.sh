#!/bin/bash
check_gpu_availability() {
    while true; do
        for gpu_id in {0..6}; do
            gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits --id=$gpu_id | awk '{print $1}')
            memory_util=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$gpu_id | awk '{print $1}')
            memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=$gpu_id | awk '{print $1}')
            memory_ratio=$(echo "scale=2; $memory_util / $memory_total * 100" | bc)

            if [[ $gpu_util -lt 60 && $memory_ratio < 60 ]]; then
                echo $gpu_id
                return
            fi
        done
        sleep 300
    done
}

var=0

for experiment_count in {1..10}; do
    for VAR1 in 0.0001 0.00005 0.00001; do
        for VAR2 in 0.2 0.5; do
            for VAR3 in 5 6 7 8 9 10; do
                for VAR4 in 32 64 128 ; do
                    for VAR5 in 0.05; do
                        for VAR6 in 0.001; do
                            for VAR7 in 0.001; do
                                gpu=$(check_gpu_availability)
                                CUDA_VISIBLE_DEVICES=$gpu python -W ignore run.py \
                                    --lr $VAR1 \
                                    --dropout $VAR2 \
                                    --G $VAR3 \
                                    --batch_size $VAR4 \
                                    --gamma $VAR5 \
                                    --tau $VAR6 \
                                    --weight_decay $VAR7 \
                                    --wandb_project_name "2024_CIKM_DeepTrader_IXIC_results" \
                                    --wandb_group_name "preliminary_hyper_search" \
                                    --wandb_session_name "setting_${var}_exp${experiment_count}" &
                                var=$((var + 1))
                                sleep 10
                            done
                        done
                    done
                done
            done
        done
    done
done
