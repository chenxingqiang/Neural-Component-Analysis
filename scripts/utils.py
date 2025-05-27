import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

def setup_chinese_fonts():
    """
    配置matplotlib中文字体显示
    """
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
            break
    
    if selected_font is None:
        # 尝试使用包含中文关键词的字体
        chinese_available = [f for f in available_fonts if any(keyword in f.lower() for keyword in ['chinese', 'cjk', 'han', 'hei', 'song', 'kai'])]
        if chinese_available:
            selected_font = chinese_available[0]
        else:
            selected_font = 'DejaVu Sans'
    
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
    
    return selected_font

# 自动配置中文字体
try:
    setup_chinese_fonts()
except:
    pass  # 如果配置失败，使用默认字体

def save_plot(filename, prefix="results/plots"):
    """
    Save plot to specified directory with proper path handling
    
    Parameters:
    -----------
    filename : str
        Name of the file to save
    prefix : str
        Directory prefix to save to
    """
    # Create plots directory if it doesn't exist
    os.makedirs(prefix, exist_ok=True)
    
    # Ensure filename doesn't have the prefix already
    if filename.startswith(prefix):
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        # Construct full path
        full_path = os.path.join(prefix, os.path.basename(filename))
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved as {os.path.join(prefix, os.path.basename(filename))}")
