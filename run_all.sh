#!/bin/bash
nohup bash orchestration_scripts/data_prep.sh &

bash orchestration_scripts/graph_denoising_init.sh