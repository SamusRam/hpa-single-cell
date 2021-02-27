#!/bin/bash
nohup bash orchestration_scripts/data_prep.sh &

nohup bash orchestration_scripts/train_init.sh &

bash orchestration_scripts/graph_denoising_init.sh