#!/usr/bin/env python3
"""
CoLaKG CPU优化监控脚本
用于监控和显示CPU利用率优化效果
"""

import psutil
import time
import os
import subprocess
import torch
import multiprocessing
from datetime import datetime

def get_system_info():
    """获取系统信息"""
    print("=" * 60)
    print("CoLaKG CPU优化配置信息")
    print("=" * 60)
    print(f"逻辑CPU核心数: {multiprocessing.cpu_count()}")
    print(f"PyTorch线程数: {torch.get_num_threads()}")
    print(f"PyTorch交互线程数: {torch.get_num_interop_threads()}")
    print(f"内存总量: {psutil.virtual_memory().total // (1024**3)} GB")
    print(f"可用内存: {psutil.virtual_memory().available // (1024**3)} GB")
    print("=" * 60)

def monitor_cpu_usage(duration=30):
    """监控CPU利用率"""
    print(f"\n开始监控CPU利用率 ({duration}秒)...")
    print("时间\t\tCPU%\t内存%\t线程数")
    print("-" * 50)
    
    start_time = time.time()
    cpu_percentages = []
    
    while time.time() - start_time < duration:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        thread_count = len(psutil.Process().threads())
        
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"{current_time}\t\t{cpu_percent:5.1f}%\t{memory_percent:5.1f}%\t{thread_count:3d}")
        
        cpu_percentages.append(cpu_percent)
    
    avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
    max_cpu = max(cpu_percentages)
    min_cpu = min(cpu_percentages)
    
    print("-" * 50)
    print(f"CPU利用率统计:")
    print(f"平均值: {avg_cpu:.1f}%")
    print(f"最大值: {max_cpu:.1f}%")
    print(f"最小值: {min_cpu:.1f}%")
    print("=" * 60)

def check_optimization_settings():
    """检查优化设置"""
    print("\n优化配置检查:")
    
    # 检查环境变量
    omp_threads = os.environ.get('OMP_NUM_THREADS', '未设置')
    mkl_threads = os.environ.get('MKL_NUM_THREADS', '未设置')
    
    print(f"OMP_NUM_THREADS: {omp_threads}")
    print(f"MKL_NUM_THREADS: {mkl_threads}")
    
    # 建议配置
    recommended_cores = max(4, multiprocessing.cpu_count() - 4)
    print(f"\n推荐配置:")
    print(f"使用CPU核心数: {recommended_cores}")
    print(f"保留系统核心: 4")
    print("=" * 60)

def main():
    """主函数"""
    get_system_info()
    check_optimization_settings()
    
    print("\n优化建议:")
    print("1. 已在world.py中设置CORES = cpu_count() - 4")
    print("2. 已设置环境变量OMP_NUM_THREADS和MKL_NUM_THREADS")
    print("3. 已在Procedure.py中设置torch.set_num_threads")
    print("4. 测试阶段已强制启用多核处理")
    print("\n现在可以运行训练脚本查看CPU利用率提升效果！")

if __name__ == "__main__":
    main()