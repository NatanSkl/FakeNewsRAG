#!/bin/bash
# Experiment Runner Commands
# 
# Part 1: Run both default and optimized with prompt_type=0 and fake_reliable
cd /home/student/FakeNewsRAG && /StudentData/rag2/bin/python evaluate/experiment_runner.py /StudentData/preprocessed/val_sampled.csv --retrieval-configs default optimized --prompt-types 0 --naming-conventions fake_reliable;

# Part 2: Run optimized with prompt_types 0,1,2 and both naming conventions with limit=5
cd /home/student/FakeNewsRAG && /StudentData/rag2/bin/python evaluate/experiment_runner.py /StudentData/preprocessed/val_sampled.csv --retrieval-configs optimized --prompt-types 0 1 2 --naming-conventions fake_reliable type1_type2 --limit 5;

# Part 3: Kill tmux server
tmux kill-server

