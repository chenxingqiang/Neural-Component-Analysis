#!/usr/bin/env python3
"""
测试中文字体显示
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import save_plot

def test_chinese_font():
    """测试中文字体显示"""
    print("测试中文字体显示...")
    
    # 创建测试数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制数据
    ax.plot(x, y1, 'b-', linewidth=2, label='正弦函数 sin(x)')
    ax.plot(x, y2, 'r-', linewidth=2, label='余弦函数 cos(x)')
    
    # 设置中文标题和标签
    ax.set_title('SECOM故障检测系统 - 中文字体测试', fontsize=16, fontweight='bold')
    ax.set_xlabel('时间 (秒)', fontsize=14)
    ax.set_ylabel('幅值', fontsize=14)
    
    # 添加图例
    ax.legend(fontsize=12)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加中文注释
    ax.annotate('最大值点', xy=(np.pi/2, 1), xytext=(2, 1.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, color='red', fontweight='bold')
    
    ax.annotate('最小值点', xy=(3*np.pi/2, -1), xytext=(5, -1.2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=12, color='blue', fontweight='bold')
    
    # 添加文本框
    textstr = '''测试内容:
• 中文标题显示
• 中文标签显示  
• 中文图例显示
• 中文注释显示
• 负号显示: -1.0'''
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    save_plot("chinese_font_test_simple.png")
    plt.close()
    
    print("✓ 中文字体测试完成，图像已保存")

if __name__ == "__main__":
    test_chinese_font() 