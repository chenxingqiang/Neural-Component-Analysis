"""
Neural Component Analysis 主入口点
提供命令行接口用于运行各种故障检测任务
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Neural Component Analysis - 工业过程故障检测')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # SECOM检测命令
    secom_parser = subparsers.add_parser('secom', help='在SECOM数据集上运行故障检测')
    secom_parser.add_argument('--model', choices=['enhanced', 'improved-t2', 'two-stage'], 
                             default='enhanced', help='要使用的模型类型')
    secom_parser.add_argument('--output', help='输出结果的文件路径')
    
    # TE检测命令
    te_parser = subparsers.add_parser('te', help='在TE数据集上运行故障检测')
    te_parser.add_argument('--model', choices=['enhanced', 'improved-t2', 'two-stage'], 
                          default='enhanced', help='要使用的模型类型')
    te_parser.add_argument('--fault-type', type=int, default=1, help='要检测的故障类型')
    te_parser.add_argument('--output', help='输出结果的文件路径')
    
    # 比较命令
    compare_parser = subparsers.add_parser('compare', help='比较不同检测方法的性能')
    compare_parser.add_argument('--dataset', choices=['secom', 'te'], required=True, 
                             help='要使用的数据集')
    compare_parser.add_argument('--metrics', choices=['auc', 'f1', 'all'], default='all',
                              help='要比较的指标')
    
    args = parser.parse_args()
    
    if args.command == 'secom':
        # 导入并运行SECOM检测
        from scripts.run_secom_fault_detection import main as run_secom
        run_secom(model_type=args.model, output_file=args.output)
        
    elif args.command == 'te':
        # 导入并运行TE检测
        from scripts.te_comparison_detection import main as run_te
        run_te(model_type=args.model, fault_type=args.fault_type, output_file=args.output)
        
    elif args.command == 'compare':
        # 导入并运行比较脚本
        if args.dataset == 'secom':
            from scripts.secom_comparison_detection import main as compare_secom
            compare_secom(metrics=args.metrics)
        else:
            from scripts.te_comparison_detection import compare_methods as compare_te
            compare_te(metrics=args.metrics)
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 