FREE_PORT=$(python ./tuner/find_free_port.py)
echo "Found free port: $FREE_PORT"

/home/zhaoxueyan/anaconda3/envs/iEDA-DSE/bin/python ./tuner/tuner_train.py \
         --multiobj 1 \
         --cfgSearchFile test/XS_TOP_TSMC28/configspace.json \
         --n_workers 4 \
         --n_iterations 20 \
         --min_points_in_model 10 \
         --log_dir test/XS_TOP_TSMC28/mobohb_log/XS_TOP \
         --nameserver_port $FREE_PORT \
         --run_args aux_input=test/XS_TOP_TSMC28/XS_TOP.aux &

/home/zhaoxueyan/anaconda3/envs/iEDA-DSE/bin/python ./tuner/tuner_train.py \
        --multiobj 1 \
        --log_dir test/XS_TOP_TSMC28/mobohb_log/XS_TOP \
        --worker \
        --worker_id 1 \
        --nameserver_port $FREE_PORT \
        --run_args aux_input=test/XS_TOP_TSMC28/XS_TOP.aux gpu=1  base_ppa=test/XS_TOP_TSMC28/XS_TOP_ppa.json  reuse_params=\"\" --density_ratio 0 \
        --congestion_ratio 0 &

/home/zhaoxueyan/anaconda3/envs/iEDA-DSE/bin/python ./tuner/tuner_train.py \
        --multiobj 1 \
        --log_dir test/XS_TOP_TSMC28/mobohb_log/XS_TOP \
        --worker \
        --worker_id 2 \
        --nameserver_port $FREE_PORT \
        --run_args aux_input=test/XS_TOP_TSMC28/XS_TOP.aux gpu=1  base_ppa=test/XS_TOP_TSMC28/XS_TOP_ppa.json  reuse_params=\"\" --density_ratio 0 \
        --congestion_ratio 0 &

/home/zhaoxueyan/anaconda3/envs/iEDA-DSE/bin/python ./tuner/tuner_train.py \
        --multiobj 1 \
        --log_dir test/XS_TOP_TSMC28/mobohb_log/XS_TOP \
        --worker \
        --worker_id 3 \
        --nameserver_port $FREE_PORT \
        --run_args aux_input=test/XS_TOP_TSMC28/XS_TOP.aux gpu=1  base_ppa=test/XS_TOP_TSMC28/XS_TOP_ppa.json  reuse_params=\"\" --density_ratio 0 \
        --congestion_ratio 0 &

/home/zhaoxueyan/anaconda3/envs/iEDA-DSE/bin/python ./tuner/tuner_train.py \
        --multiobj 1 \
        --log_dir test/XS_TOP_TSMC28/mobohb_log/XS_TOP \
        --worker \
        --worker_id 4 \
        --nameserver_port $FREE_PORT \
        --run_args aux_input=test/XS_TOP_TSMC28/XS_TOP.aux gpu=1  base_ppa=test/XS_TOP_TSMC28/XS_TOP_ppa.json  reuse_params=\"\" --density_ratio 0 \
        --congestion_ratio 0 &