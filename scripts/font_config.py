#!/usr/bin/env python3
"""
matplotlib中文字体配置脚本
解决绘图中中文字体显示问题

作者: Neural Component Analysis Team
日期: 2024
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
import warnings

def setup_chinese_fonts():
    """
    配置matplotlib中文字体显示
    """
    print("正在配置matplotlib中文字体...")
    
    # 获取系统类型
    system = platform.system()
    
    # 根据系统选择合适的中文字体
    chinese_fonts = []
    
    if system == "Darwin":  # macOS
        chinese_fonts = [
            'PingFang SC',      # 苹方
            'Hiragino Sans GB', # 冬青黑体
            'STHeiti',          # 华文黑体
            'SimHei',           # 黑体
            'Arial Unicode MS'  # Arial Unicode MS
        ]
    elif system == "Windows":  # Windows
        chinese_fonts = [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',           # 黑体
            'SimSun',           # 宋体
            'KaiTi',            # 楷体
            'FangSong'          # 仿宋
        ]
    else:  # Linux
        chinese_fonts = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Noto Sans CJK SC',     # 思源黑体
            'Source Han Sans CN',   # 思源黑体
            'DejaVu Sans'           # DejaVu Sans
        ]
    
    # 查找可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = None
    
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            print(f"找到可用中文字体: {font}")
            break
    
    if selected_font is None:
        print("警告: 未找到合适的中文字体，将使用默认字体")
        print("可用字体列表:")
        chinese_available = [f for f in available_fonts if any(keyword in f.lower() for keyword in ['chinese', 'cjk', 'han', 'hei', 'song', 'kai'])]
        for font in chinese_available[:10]:  # 只显示前10个
            print(f"  - {font}")
        
        # 尝试使用第一个包含中文关键词的字体
        if chinese_available:
            selected_font = chinese_available[0]
            print(f"使用字体: {selected_font}")
        else:
            selected_font = 'DejaVu Sans'
            print(f"使用默认字体: {selected_font}")
    
    # 配置matplotlib
    plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 设置其他字体参数
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'font.family': 'sans-serif'
    })
    
    print(f"matplotlib中文字体配置完成: {selected_font}")
    return selected_font

def test_chinese_display():
    """
    测试中文字体显示效果
    """
    import numpy as np
    
    print("测试中文字体显示...")
    
    # 创建测试图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # 绘制图形
    ax.plot(x, y1, label='正弦函数', linewidth=2)
    ax.plot(x, y2, label='余弦函数', linewidth=2)
    
    # 设置中文标签
    ax.set_title('中文字体显示测试', fontsize=16, fontweight='bold')
    ax.set_xlabel('横坐标 (时间)', fontsize=12)
    ax.set_ylabel('纵坐标 (幅值)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加中文注释
    ax.annotate('最大值点', xy=(np.pi/2, 1), xytext=(2, 1.2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    
    # 保存测试图
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.utils import save_plot
    save_plot("chinese_font_test.png")
    plt.close()
    
    print("中文字体测试完成，已保存为 chinese_font_test.png")

def get_system_fonts():
    """
    获取系统可用字体列表
    """
    print("获取系统字体列表...")
    
    # 获取所有字体
    fonts = [f.name for f in fm.fontManager.ttflist]
    fonts = sorted(list(set(fonts)))  # 去重并排序
    
    # 筛选可能的中文字体
    chinese_keywords = ['chinese', 'cjk', 'han', 'hei', 'song', 'kai', 'ping', 'fang', 'yahei', 'simhei', 'simsun']
    chinese_fonts = []
    
    for font in fonts:
        if any(keyword in font.lower() for keyword in chinese_keywords):
            chinese_fonts.append(font)
    
    print(f"系统总字体数: {len(fonts)}")
    print(f"可能的中文字体数: {len(chinese_fonts)}")
    
    if chinese_fonts:
        print("\n可能的中文字体:")
        for font in chinese_fonts:
            print(f"  - {font}")
    
    return fonts, chinese_fonts

def fix_existing_plots():
    """
    修复现有绘图脚本的字体问题
    """
    print("正在修复现有绘图脚本的字体配置...")
    
    # 需要修复的脚本列表
    scripts_to_fix = [
        'scripts/secom_advanced_analysis.py',
        'scripts/secom_deep_research_analysis.py',
        'scripts/secom_comprehensive_research.py',
        'scripts/secom_comparison_detection.py'
    ]
    
    for script_path in scripts_to_fix:
        if os.path.exists(script_path):
            print(f"修复脚本: {script_path}")
            
            # 读取文件内容
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否已经包含字体配置
            if 'from scripts.font_config import setup_chinese_fonts' not in content:
                # 在import部分添加字体配置导入
                import_lines = []
                other_lines = []
                in_imports = True
                
                for line in content.split('\n'):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        import_lines.append(line)
                    elif line.strip() == '' and in_imports:
                        import_lines.append(line)
                    else:
                        if in_imports and line.strip() != '':
                            # 添加字体配置导入
                            import_lines.append('from scripts.font_config import setup_chinese_fonts')
                            import_lines.append('')
                            in_imports = False
                        other_lines.append(line)
                
                # 重新组合内容
                new_content = '\n'.join(import_lines + other_lines)
                
                # 在主函数或类初始化中添加字体配置调用
                if 'def __init__(self):' in new_content:
                    new_content = new_content.replace(
                        'def __init__(self):',
                        'def __init__(self):\n        setup_chinese_fonts()  # 配置中文字体'
                    )
                elif 'def main():' in new_content:
                    new_content = new_content.replace(
                        'def main():',
                        'def main():\n    setup_chinese_fonts()  # 配置中文字体'
                    )
                
                # 写回文件
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"  ✓ 已添加字体配置到 {script_path}")
            else:
                print(f"  - {script_path} 已包含字体配置")
        else:
            print(f"  ✗ 文件不存在: {script_path}")

def main():
    """
    主函数：配置字体并测试
    """
    print("="*60)
    print("matplotlib中文字体配置工具")
    print("="*60)
    
    # 1. 获取系统字体信息
    fonts, chinese_fonts = get_system_fonts()
    
    # 2. 配置中文字体
    selected_font = setup_chinese_fonts()
    
    # 3. 测试字体显示
    test_chinese_display()
    
    # 4. 修复现有脚本
    fix_existing_plots()
    
    print("\n" + "="*60)
    print("字体配置完成！")
    print(f"选用字体: {selected_font}")
    print("建议:")
    print("1. 重新运行绘图脚本以应用新的字体配置")
    print("2. 查看 chinese_font_test.png 确认中文显示效果")
    print("3. 如果仍有问题，请安装系统中文字体包")
    print("="*60)

if __name__ == "__main__":
    main() 