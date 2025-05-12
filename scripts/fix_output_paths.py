#!/usr/bin/env python3
"""
自动修复结果保存路径的工具脚本。
将脚本中的输出文件保存位置从根目录移动到results目录下。
"""

import os
import re
import sys
from pathlib import Path

# 需要替换的模式和其对应的新路径
FILE_PATTERNS = {
    # 模型文件
    r'(["\'])(\w+)(_model\w*\.pth)(["\'])': r'\1results/models/\2\3\4',
    r'(["\'])(\w+)(\.pth)(["\'])': r'\1results/models/\2\3\4',
    
    # 图表文件
    r'(["\'])(\w+)(\.png)(["\'])': r'\1results/plots/\2\3\4',
    
    # 日志和CSV文件
    r'(["\'])(\w+)(\.csv)(["\'])': r'\1results/logs/\2\3\4',
    r'(["\'])(\w+)(\.log)(["\'])': r'\1results/logs/\2\3\4',
    r'(["\'])(\w+)(\.json)(["\'])': r'\1results/logs/\2\3\4',
}

# 更明确的文件路径替换
EXPLICIT_PATHS = {
    # 模型文件路径
    'secom_enhanced_transformer.pth': 'results/models/secom_enhanced_transformer.pth',
    'secom_improved_transformer.pth': 'results/models/secom_improved_transformer.pth',
    'enhanced_transformer_autoencoder.pth': 'results/models/enhanced_transformer_autoencoder.pth',
    'improved_transformer_t2.pth': 'results/models/improved_transformer_t2.pth',
    'transformer_refiner.pth': 'results/models/transformer_refiner.pth',
    'secom_selected_50_features.pth': 'results/models/secom_selected_50_features.pth',
    
    # 图表文件路径
    'secom_combined_detection.png': 'results/plots/secom_combined_detection.png',
    'secom_enhanced_transformer_fault_detection.png': 'results/plots/secom_enhanced_transformer_fault_detection.png',
    'secom_enhanced_transformer_training.png': 'results/plots/secom_enhanced_transformer_training.png',
    'secom_feature_importance.png': 'results/plots/secom_feature_importance.png',
    'secom_improved_transformer_spe.png': 'results/plots/secom_improved_transformer_spe.png',
    'secom_methods_comparison.png': 'results/plots/secom_methods_comparison.png',
    'secom_selected_50_features_spe.png': 'results/plots/secom_selected_50_features_spe.png',
    'secom_selected_50_features_training.png': 'results/plots/secom_selected_50_features_training.png',
    'secom_extreme_anomaly_detector.png': 'results/plots/secom_extreme_anomaly_detector.png',
    'secom_ultra_extreme_detection.png': 'results/plots/secom_ultra_extreme_detection.png',
    'secom_ultra_sensitive_ensemble.png': 'results/plots/secom_ultra_sensitive_ensemble.png',
    'secom_pca_comparison.png': 'results/plots/secom_pca_comparison.png',
    'secom_balanced_two_stage_detection.png': 'results/plots/secom_balanced_two_stage_detection.png',
    'transformer_enhanced_two_stage.png': 'results/plots/transformer_enhanced_two_stage.png',
    'fault_contribution_plot.png': 'results/plots/fault_contribution_plot.png',
    'improved_transformer_t2_metrics.png': 'results/plots/improved_transformer_t2_metrics.png',
    'improved_transformer_combined_metrics.png': 'results/plots/improved_transformer_combined_metrics.png',
}

def fix_file_paths(file_path):
    """修复单个文件中的输出路径"""
    print(f"处理文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    
    # 先替换明确的路径
    for old, new in EXPLICIT_PATHS.items():
        if f'"{old}"' in content or f"'{old}'" in content:
            content = content.replace(f'"{old}"', f'"{new}"')
            content = content.replace(f"'{old}'", f"'{new}'")
            modified = True
            print(f"  - 替换路径: {old} -> {new}")
    
    # 再使用正则表达式进行模式替换
    for pattern, replacement in FILE_PATTERNS.items():
        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            content = new_content
            modified = True
            print(f"  - 使用模式替换了 {count} 处路径")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ 已更新文件")
    else:
        print(f"  • 无需修改")

def fix_scripts_dir(scripts_dir):
    """修复scripts目录中所有Python文件的路径"""
    script_files = Path(scripts_dir).glob('*.py')
    for script_file in script_files:
        # 排除当前脚本和导入修复脚本
        if script_file.name not in ['fix_output_paths.py', 'fix_imports.py']:
            fix_file_paths(script_file)

def fix_examples_dir(examples_dir):
    """修复examples目录中所有Python文件的路径"""
    example_files = Path(examples_dir).glob('*.py')
    for example_file in example_files:
        fix_file_paths(example_file)

def fix_src_dir(src_dir):
    """修复src目录中所有Python文件的路径"""
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                fix_file_paths(file_path)

def main():
    """主函数"""
    # 获取项目根目录
    root_dir = Path(__file__).parent.parent
    
    # 确保结果目录存在
    for dir_path in ['results/models', 'results/plots', 'results/logs']:
        os.makedirs(os.path.join(root_dir, dir_path), exist_ok=True)
    
    # 修复scripts目录
    scripts_dir = root_dir / 'scripts'
    print(f"\n修复scripts目录下的输出路径:")
    fix_scripts_dir(scripts_dir)
    
    # 修复examples目录
    examples_dir = root_dir / 'examples'
    print(f"\n修复examples目录下的输出路径:")
    fix_examples_dir(examples_dir)
    
    # 修复src目录
    src_dir = root_dir / 'src'
    print(f"\n修复src目录下的输出路径:")
    fix_src_dir(src_dir)
    
    print("\n输出路径修复完成!")

if __name__ == '__main__':
    main() 