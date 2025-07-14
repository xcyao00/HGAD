#!/bin/bash

# Default run with command line arguments
# echo "Running with default command line arguments..."
# python main.py --config configs/mvtecad.json --seed 123

# Run with config file
# echo "Running with config file..."
# python main.py --config configs/default_config.json

# Run with config file and override some parameters
# echo "Running with config file and parameter override..."
# python main.py --config configs/default_config.json --seed 42 --batch_size 16

# Fast test run
# echo "Running fast test..."
# python main.py --config configs/fast_test_config.json

python continual_main.py \
    --dataset mvtec \
    --img_size 1024 \
    --batch_size 8 \
    --meta_epochs 5 \
    --sub_epochs 2 \
    --num_tasks 5 \
    --classes_per_task 3