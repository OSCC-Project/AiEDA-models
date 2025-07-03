python ./tuner/tuner_train.py \
         --multiobj 1 \
         --cfgSearchFile test/ariane133_nangate45_51/configspace.json \
         --n_workers 1 \
         --n_iterations 20 \
         --min_points_in_model 10 \
         --log_dir test/ariane133_nangate45_51/mobohb_log/NV_NVDLA_partition_c \
         --run_args aux_input=test/ariane133_nangate45_51/NV_NVDLA_partition_c.aux

python ./tuner/tuner_train.py \
        --multiobj 1 \
        --log_dir test/ariane133_nangate45_51/mobohb_log/NV_ariane133_partition_c \
        --worker \
        --worker_id 1 \
        --run_args aux_input=test/ariane133_nangate45_51/NV_ariane133_partition_c.aux gpu=1  base_ppa=test/ariane133_nangate45_51/ariane133_ppa.json  reuse_params=\"\" \ 
        --density_ratio 0 \
        --congestion_ratio 0 \