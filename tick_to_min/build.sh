#!/bin/bash
# ==========================================
# High-Frequency Futures Feature Engine Build Script
# ==========================================

# 1. 编译 C++ 零拷贝加速引擎
echo "Compiling C++ Tick Engine..."
cd ./tick_to_min
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) tick_engine.cpp -o tick_engine$(python3-config --extension-suffix)
echo "C++ Engine Compiled Successfully."

# 2. 启动分钟频切片生产任务 (Tick to Min)
# echo "Starting Tick-to-Min Pipeline..."
# nohup python ./tick_to_min/run.py > run_tick.log 2>&1 &

# 3. 启动截面因子与特征图谱生产任务 (Factor Generation)
# echo "Starting Factor Generation Pipeline..."
# nohup python ./features/feature_calculation.py > run_factor.log 2>&1 &

echo "Done."