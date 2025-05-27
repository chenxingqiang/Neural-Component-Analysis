#!/usr/bin/env python3
"""
修复现有绘图脚本的字体导入问题
"""

import os
import re

def fix_script_imports(script_path):
    """修复单个脚本的导入"""
    if not os.path.exists(script_path):
        print(f"文件不存在: {script_path}")
        return False
    
    print(f"修复脚本: {script_path}")
    
    # 读取文件内容
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经包含字体配置
    if 'setup_chinese_fonts' in content:
        print(f"  - {script_path} 已包含字体配置")
        return True
    
    # 查找导入部分的结束位置
    lines = content.split('\n')
    import_end_idx = 0
    
    for i, line in enumerate(lines):
        if line.strip().startswith(('import ', 'from ')) or line.strip() == '':
            import_end_idx = i
        elif line.strip() and not line.strip().startswith('#'):
            break
    
    # 在导入部分添加字体配置
    lines.insert(import_end_idx + 1, 'from scripts.utils import setup_chinese_fonts')
    lines.insert(import_end_idx + 2, '')
    
    # 查找类的__init__方法或main函数
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # 在类的__init__方法中添加字体配置
        if 'def __init__(self):' in line:
            indent = len(line) - len(line.lstrip())
            new_lines.append(' ' * (indent + 4) + 'setup_chinese_fonts()  # 配置中文字体')
        
        # 在main函数开始添加字体配置
        elif line.strip() == 'def main():':
            new_lines.append('    setup_chinese_fonts()  # 配置中文字体')
    
    # 写回文件
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    print(f"  ✓ 已修复 {script_path}")
    return True

def main():
    """主函数"""
    print("="*60)
    print("修复绘图脚本字体导入")
    print("="*60)
    
    # 需要修复的脚本列表
    scripts_to_fix = [
        'secom_advanced_analysis.py',
        'secom_deep_research_analysis.py', 
        'secom_comprehensive_research.py',
        'secom_comparison_detection.py',
        'secom_plot_optimization.py'
    ]
    
    fixed_count = 0
    for script in scripts_to_fix:
        if fix_script_imports(script):
            fixed_count += 1
    
    print(f"\n修复完成: {fixed_count}/{len(scripts_to_fix)} 个脚本")
    print("\n建议:")
    print("1. 重新运行绘图脚本测试中文字体显示")
    print("2. 如果仍有问题，请检查系统中文字体安装")

if __name__ == "__main__":
    main() 