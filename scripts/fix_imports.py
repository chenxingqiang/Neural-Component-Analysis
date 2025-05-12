#!/usr/bin/env python3
"""
自动修复导入路径的工具脚本。
将脚本中的本地导入转换为从src模块导入。
"""

import os
import re
import sys
from pathlib import Path

# 需要替换的导入映射
IMPORT_MAPPING = {
    'from enhanced_transformer_autoencoder import': 'from src.models.enhanced_transformer_autoencoder import',
    'from enhanced_transformer_detection import': 'from src.detectors.enhanced_transformer_detection import',
    'from improved_transformer_t2 import': 'from src.models.improved_transformer_t2 import',
    'from spe_fault_detector import': 'from src.detectors.spe_fault_detector import',
    'from transformer_enhanced_two_stage import': 'from src.models.transformer_enhanced_two_stage import',
    'from fault_detector_factory import': 'from src.detectors.fault_detector_factory import',
    'from process_secom_data import': 'from src.utils.process_secom_data import',
}

# 当脚本导入其他脚本时的映射
SCRIPT_IMPORT_MAPPING = {
    'from run_secom_fault_detection import': 'from scripts.run_secom_fault_detection import',
    'from compare_t2_spe import': 'from scripts.compare_t2_spe import',
    'from secom_comparison_detection import': 'from scripts.secom_comparison_detection import',
    'from te_comparison_detection import': 'from scripts.te_comparison_detection import',
    'from secom_comparison_detection_pca import': 'from scripts.secom_comparison_detection_pca import',
}

def fix_file_imports(file_path):
    """修复单个文件中的导入语句"""
    print(f"处理文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    for old, new in {**IMPORT_MAPPING, **SCRIPT_IMPORT_MAPPING}.items():
        if old in content:
            content = content.replace(old, new)
            modified = True
            print(f"  - 替换: {old} -> {new}")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ 已更新文件")
    else:
        print(f"  • 无需修改")

def fix_scripts_dir(scripts_dir):
    """修复scripts目录中所有Python文件的导入"""
    script_files = Path(scripts_dir).glob('*.py')
    for script_file in script_files:
        if script_file.name != 'fix_imports.py':  # 跳过自身
            fix_file_imports(script_file)

def fix_examples_dir(examples_dir):
    """修复examples目录中所有Python文件的导入"""
    example_files = Path(examples_dir).glob('*.py')
    for example_file in example_files:
        fix_file_imports(example_file)

def main():
    """主函数"""
    # 获取项目根目录
    root_dir = Path(__file__).parent.parent
    
    # 修复scripts目录
    scripts_dir = root_dir / 'scripts'
    print(f"\n修复scripts目录下的导入:")
    fix_scripts_dir(scripts_dir)
    
    # 修复examples目录
    examples_dir = root_dir / 'examples'
    print(f"\n修复examples目录下的导入:")
    fix_examples_dir(examples_dir)
    
    print("\n导入修复完成!")

if __name__ == '__main__':
    main() 