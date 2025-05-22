#!/usr/bin/env python3
"""
交互式任务执行脚本

此脚本提供一个用户友好的界面，用于执行Neural-Component-Analysis项目中的各种任务。
它允许用户选择要运行的脚本，设置相关参数，并显示执行结果。

用法：
  python scripts/run_interactive.py
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def clear_screen():
    """清除终端屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    """打印标题头"""
    clear_screen()
    print("="*80)
    print(f"{title:^80}")
    print("="*80)
    print()

def print_menu(options):
    """打印菜单选项"""
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")
    print("0. 返回上一级菜单")
    print()

def get_user_choice(max_choice):
    """获取用户选择"""
    while True:
        try:
            choice = int(input("请输入您的选择 [0-{}]: ".format(max_choice)))
            if 0 <= choice <= max_choice:
                return choice
            print(f"错误: 请输入0到{max_choice}之间的数字")
        except ValueError:
            print("错误: 请输入一个有效的数字")

def run_command(command):
    """运行命令并实时显示输出"""
    print("\n" + "-"*40)
    print(f"执行命令: {command}")
    print("-"*40 + "\n")
    
    try:
        # 使用subprocess运行命令，并捕获输出
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时显示输出
        for line in process.stdout:
            print(line, end='')
        
        # 等待进程完成
        process.wait()
        
        print("\n" + "-"*40)
        if process.returncode == 0:
            print("命令成功执行！")
        else:
            print(f"命令执行失败，返回代码: {process.returncode}")
        print("-"*40)
        
        return process.returncode
    except Exception as e:
        print(f"执行命令时出错: {str(e)}")
        return 1

def check_for_results():
    """检查并显示结果文件信息"""
    result_dirs = [
        "results/plots",
        "results/models"
    ]
    
    print("\n生成的结果文件:")
    
    # 检查plots目录中的新图像
    plots_dir = os.path.join(project_root, "results/plots")
    if os.path.exists(plots_dir):
        plot_files = sorted(Path(plots_dir).glob("*.png"), key=os.path.getmtime, reverse=True)
        if plot_files:
            print("\n最近生成的图像:")
            for i, plot_file in enumerate(plot_files[:5]):  # 只显示最近的5个文件
                mod_time = os.path.getmtime(plot_file)
                print(f"  - {os.path.basename(plot_file)} (修改时间: {mod_time})")
    
    # 检查模型目录
    models_dir = os.path.join(project_root, "results/models")
    if os.path.exists(models_dir):
        # 统计每个数据集的模型数量
        secom_models = len(list(Path(models_dir).glob("**/secom/**/*.pth")))
        te_models = len(list(Path(models_dir).glob("**/te/**/*.pth")))
        
        if secom_models > 0 or te_models > 0:
            print("\n保存的模型:")
            if secom_models > 0:
                print(f"  - SECOM数据集: {secom_models}个模型")
            if te_models > 0:
                print(f"  - TE数据集: {te_models}个模型")

def run_secom_comparison():
    """运行SECOM数据集比较脚本"""
    print_header("SECOM数据集比较")
    
    # 参数选项
    print("参数选项:")
    print("1. 包含改进型Transformer模型 (--include_improved)")
    print("2. 包含Transformer增强型两阶段检测器 (--include_transformer)")
    print("3. 跳过PCA基线方法 (--skip_pca)")
    print("4. 使用默认参数运行")
    print()
    
    choice = get_user_choice(4)
    
    if choice == 0:
        return
    
    # 构建命令
    command = "PYTHONPATH=$PYTHONPATH:. python scripts/secom_comparison_detection.py"
    
    if choice == 1:
        command += " --include_improved"
    elif choice == 2:
        command += " --include_transformer"
    elif choice == 3:
        command += " --skip_pca"
    
    # 运行命令
    run_command(command)
    
    # 检查结果
    check_for_results()
    
    input("\n按Enter键继续...")

def run_te_comparison():
    """运行TE数据集比较脚本"""
    print_header("TE数据集比较")
    
    # 参数选项
    print("参数选项:")
    print("1. 选择特定故障类别")
    print("2. 跳过改进型Transformer模型 (--skip_improved)")
    print("3. 使用默认参数运行（分析所有故障类别）")
    print()
    
    choice = get_user_choice(3)
    
    if choice == 0:
        return
    
    # 构建命令
    command = "PYTHONPATH=$PYTHONPATH:. python scripts/te_comparison_detection.py"
    
    if choice == 1:
        # 获取用户想要分析的故障类别
        categories = input("请输入要分析的故障类别（用空格分隔，例如 '1 2 3'）: ")
        if categories.strip():
            command += f" --categories {categories}"
    elif choice == 2:
        command += " --skip_improved"
    
    # 运行命令
    run_command(command)
    
    # 检查结果
    check_for_results()
    
    input("\n按Enter键继续...")

def run_secom_fault_detection():
    """运行SECOM故障检测脚本"""
    print_header("SECOM故障检测")
    
    # 参数选项（这个脚本没有命令行参数，直接运行）
    print("该脚本将运行多种故障检测方法并比较它们的性能:")
    print("- 增强型Transformer检测")
    print("- 改进型Transformer检测")
    print("- 组合故障检测（动态阈值）")
    print("- 特征选择模型")
    print("- 超敏感集成检测器")
    print("- 极端异常检测器")
    print("- 超极端异常检测器")
    print("- 平衡两阶段检测器")
    print()
    
    print("1. 运行脚本")
    print()
    
    choice = get_user_choice(1)
    
    if choice == 0:
        return
    
    # 运行命令
    command = "PYTHONPATH=$PYTHONPATH:. python scripts/run_secom_fault_detection.py"
    run_command(command)
    
    # 检查结果
    check_for_results()
    
    input("\n按Enter键继续...")

def run_t2_spe_comparison():
    """运行T²和SPE比较脚本"""
    print_header("T²和SPE性能比较")
    
    # 参数选项（这个脚本没有命令行参数，直接运行）
    print("该脚本将对平衡两阶段检测器的T²和SPE指标进行可视化和比较")
    print()
    
    print("1. 运行脚本")
    print()
    
    choice = get_user_choice(1)
    
    if choice == 0:
        return
    
    # 运行命令
    command = "PYTHONPATH=$PYTHONPATH:. python scripts/compare_t2_spe.py"
    run_command(command)
    
    # 检查结果
    check_for_results()
    
    input("\n按Enter键继续...")

def run_transformer_comparison():
    """运行Transformer比较脚本"""
    print_header("Transformer模型比较")
    
    # 参数选项
    print("参数选项:")
    print("1. 选择数据集 (SECOM, TE, 或两者)")
    print("2. 跳过基本Transformer模型 (--skip_basic)")
    print("3. 使用默认参数运行（分析两个数据集）")
    print()
    
    choice = get_user_choice(3)
    
    if choice == 0:
        return
    
    # 构建命令
    command = "PYTHONPATH=$PYTHONPATH:. python scripts/transformer_comparison_detection.py"
    
    if choice == 1:
        # 获取用户想要分析的数据集
        print("\n选择数据集:")
        print("1. SECOM")
        print("2. TE")
        print("3. 两者")
        
        dataset_choice = get_user_choice(3)
        if dataset_choice == 1:
            command += " --dataset secom"
        elif dataset_choice == 2:
            command += " --dataset te"
        elif dataset_choice == 3:
            command += " --dataset both"
        else:
            return
    elif choice == 2:
        command += " --skip_basic"
    
    # 运行命令
    run_command(command)
    
    # 检查结果
    check_for_results()
    
    input("\n按Enter键继续...")

def run_custom_script():
    """运行用户指定的自定义脚本"""
    print_header("自定义脚本执行")
    
    # 列出scripts目录中的所有Python脚本，排除辅助模块和初始化文件
    scripts_dir = os.path.join(project_root, "scripts")
    excluded_files = ["__init__.py", "utils.py", "run_interactive.py"]
    script_files = [f for f in os.listdir(scripts_dir) 
                   if f.endswith(".py") 
                   and f not in excluded_files
                   and not f.startswith("_")]  # 排除以下划线开头的文件
    
    # 按照文件名排序
    script_files.sort()
    
    if not script_files:
        print("未找到可执行的Python脚本！")
        input("\n按Enter键继续...")
        return
    
    print("可用脚本:")
    for idx, script in enumerate(script_files, 1):
        print(f"{idx}. {script}")
    print()
    
    choice = get_user_choice(len(script_files))
    
    if choice == 0:
        return
    
    selected_script = script_files[choice-1]
    
    # 获取用户指定的参数
    args = input(f"\n请输入要传递给 {selected_script} 的参数（可选）: ")
    
    # 构建命令
    command = f"PYTHONPATH=$PYTHONPATH:. python scripts/{selected_script} {args}"
    
    # 运行命令
    run_command(command)
    
    # 检查结果
    check_for_results()
    
    input("\n按Enter键继续...")

def run_comparative_analysis():
    """运行综合比较分析"""
    print_header("综合比较分析")
    
    # 参数选项
    print("参数选项:")
    print("1. 选择数据集 (SECOM, TE, 或两者)")
    print("2. 跳过改进型Transformer模型 (--skip_improved_transformer)")
    print("3. 包含Transformer增强型两阶段检测器 (--include_transformer)")
    print("4. 使用默认参数运行（分析两个数据集）")
    print()
    
    choice = get_user_choice(4)
    
    if choice == 0:
        return
    
    # 构建命令
    command = "PYTHONPATH=$PYTHONPATH:. python scripts/run_comparison.py"
    
    if choice == 1:
        # 获取用户想要分析的数据集
        print("\n选择数据集:")
        print("1. SECOM")
        print("2. TE")
        print("3. 两者")
        print("4. Transformer比较")
        
        dataset_choice = get_user_choice(4)
        if dataset_choice == 1:
            command += " --dataset secom"
        elif dataset_choice == 2:
            command += " --dataset te"
        elif dataset_choice == 3:
            command += " --dataset both"
        elif dataset_choice == 4:
            command += " --dataset transformer"
        else:
            return
    elif choice == 2:
        command += " --skip_improved_transformer"
    elif choice == 3:
        command += " --include_transformer"
    
    # 运行命令
    run_command(command)
    
    # 检查结果
    check_for_results()
    
    input("\n按Enter键继续...")

def main_menu():
    """显示主菜单"""
    while True:
        print_header("Neural-Component-Analysis 交互式任务执行")
        
        options = [
            "SECOM数据集比较",
            "TE数据集比较",
            "SECOM故障检测",
            "T²和SPE性能比较",
            "Transformer模型比较",
            "综合比较分析",
            "自定义脚本执行",
            "退出"
        ]
        
        print_menu(options)
        choice = get_user_choice(len(options))
        
        if choice == 1:
            run_secom_comparison()
        elif choice == 2:
            run_te_comparison()
        elif choice == 3:
            run_secom_fault_detection()
        elif choice == 4:
            run_t2_spe_comparison()
        elif choice == 5:
            run_transformer_comparison()
        elif choice == 6:
            run_comparative_analysis()
        elif choice == 7:
            run_custom_script()
        elif choice == 8 or choice == 0:
            print("\n感谢使用，再见！")
            sys.exit(0)

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        sys.exit(1) 