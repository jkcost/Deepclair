var=0 

for VAR1 in 0.1 0.01 0.001 0.0001
do
	for VAR2 in 0.0 0.1 0.2 0.5 0.8
	do
		for VAR3 in 5
		do
			for VAR4 in 32 64 128 256 512
			do
				for VAR5 in 0.1 0.001 0.0001
				do
					for VAR6 in 0.01 0.05 0.1 0.5
					do
						for VAR7 in 0.0 0.000001 0.0001
						do
							gpu=$((var % 8))
		
							CUDA_VISIBLE_DEVICES=$gpu python -W ignore run.py \
							                                --lr $VAR1 \
							                                --dropout $VAR2 \
							                                --window_len $VAR3 \
							                                --batch_size $VAR4 \
							                                --gamma $VAR5 \
							                                --tau $VAR6 \
							                                --weight_decay $VAR7 \
							                                --wandb_project_name "2024_CIKM_DeepClair" \
							                                --wandb_group_name "preliminary_hyper_search" \
							                                --wandb_session_name "setting_"$var &
							var=$((var + 1))
							sleep 10
						done
					done
				done
			done
		done
	done
done

