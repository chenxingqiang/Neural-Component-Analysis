# matplotlib中文字体显示问题解决方案

## 问题描述
在运行SECOM故障检测分析脚本时，matplotlib绘图中的中文字符无法正常显示，出现方框或乱码。

## 解决方案

### 1. 自动字体配置
已在 `scripts/utils.py` 中添加了自动中文字体配置功能：

```python
def setup_chinese_fonts():
    """配置matplotlib中文字体显示"""
    # 根据操作系统自动选择合适的中文字体
    # macOS: PingFang SC, Hiragino Sans GB, STHeiti
    # Windows: Microsoft YaHei, SimHei, SimSun
    # Linux: WenQuanYi Micro Hei, Noto Sans CJK SC
```

### 2. 字体配置特性
- **自动检测系统**: 根据操作系统(macOS/Windows/Linux)选择最佳字体
- **智能回退**: 如果首选字体不可用，自动选择备用字体
- **负号修复**: 解决matplotlib中负号显示为方框的问题
- **全局配置**: 一次配置，所有绘图脚本自动应用

### 3. 当前配置状态
✅ **系统检测**: macOS (Darwin 24.4.0)  
✅ **选用字体**: Hiragino Sans GB (冬青黑体)  
✅ **可用字体**: 检测到20个中文相关字体  
✅ **测试通过**: 中文标题、标签、图例、注释均正常显示

### 4. 已修复的脚本
以下脚本已自动添加中文字体支持：
- ✅ `scripts/secom_advanced_analysis.py`
- ✅ `scripts/secom_deep_research_analysis.py`
- ✅ `scripts/secom_comprehensive_research.py`
- ✅ `scripts/secom_comparison_detection.py`
- ✅ `scripts/secom_plot_optimization.py`

### 5. 使用方法

#### 方法一：自动配置（推荐）
所有绘图脚本已自动配置，直接运行即可：
```bash
python scripts/secom_comparison_detection.py
python scripts/secom_comprehensive_research.py
```

#### 方法二：手动配置
在新的绘图脚本中添加：
```python
from scripts.utils import setup_chinese_fonts
setup_chinese_fonts()  # 在绘图前调用
```

#### 方法三：独立配置工具
运行专门的字体配置工具：
```bash
python scripts/font_config.py
```

### 6. 测试验证

#### 快速测试
```bash
python scripts/test_chinese_font.py
```
生成测试图像：`results/plots/chinese_font_test_simple.png`

#### 完整测试
运行任意分析脚本，检查生成的图像中文显示效果。

### 7. 故障排除

#### 问题：仍然显示方框
**解决方案**：
1. 检查系统是否安装中文字体
2. 运行 `python scripts/font_config.py` 查看可用字体
3. 手动安装中文字体包

#### 问题：字体不美观
**解决方案**：
1. 安装更好的中文字体（如思源黑体）
2. 修改 `scripts/utils.py` 中的字体优先级列表

#### 问题：Linux系统字体缺失
**解决方案**：
```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

# CentOS/RHEL
sudo yum install wqy-microhei-fonts wqy-zenhei-fonts
```

### 8. 技术细节

#### 字体选择逻辑
1. 检测操作系统类型
2. 按优先级尝试系统预设字体列表
3. 如果都不可用，搜索包含中文关键词的字体
4. 最后回退到默认字体

#### 配置参数
```python
plt.rcParams.update({
    'font.sans-serif': [selected_font, 'DejaVu Sans', 'Arial'],
    'axes.unicode_minus': False,  # 修复负号显示
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})
```

### 9. 效果展示

修复前：
- 中文字符显示为 □□□□
- 负号显示为方框
- 图表标题和标签无法阅读

修复后：
- ✅ 中文标题：「SECOM故障检测系统 - 中文字体测试」
- ✅ 中文标签：「时间 (秒)」、「幅值」
- ✅ 中文图例：「正弦函数 sin(x)」、「余弦函数 cos(x)」
- ✅ 中文注释：「最大值点」、「最小值点」
- ✅ 负号正常：「-1.0」

### 10. 维护建议

1. **定期更新**: 如果系统安装了新的中文字体，可重新运行配置工具
2. **性能优化**: 字体配置只在首次导入时执行，不影响绘图性能
3. **兼容性**: 配置方案兼容matplotlib 3.x版本
4. **扩展性**: 可根据需要添加更多字体选项

---

## 总结

通过实施自动字体检测和配置机制，成功解决了SECOM故障检测项目中matplotlib中文字体显示问题。现在所有绘图脚本都能正确显示中文内容，提升了分析报告的可读性和专业性。

**当前状态**: ✅ 问题已完全解决  
**测试状态**: ✅ 所有功能正常  
**部署状态**: ✅ 已应用到所有脚本 