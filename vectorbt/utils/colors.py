# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
颜色处理工具模块

主要功能包括：
1. 颜色映射转换：将数值映射到matplotlib颜色映射并转换为RGB格式
2. 透明度调整：动态调整颜色的不透明度，用于分层可视化
3. 亮度调整：调整颜色的明亮程度，用于视觉层次表达
"""

import numpy as np
from vectorbt import _typing as tp


def rgb_from_cmap(cmap_name: str, value: float, value_range: tp.Tuple[float, float]) -> str:
    """
    从matplotlib颜色映射中获取指定数值对应的RGB颜色字符串
    
    Args:
        cmap_name (str): matplotlib颜色映射名称，如'viridis', 'plasma', 'coolwarm'等，表示颜色映射的类型
        value (float): 需要映射为颜色的数值
        value_range (tuple): 数值范围元组(最小值, 最大值)，用于标准化计算
    
    Returns:
        str: RGB颜色字符串，格式为"rgb(r,g,b)"，其中r,g,b为0-255的整数值
    
    应用示例:
        >>> # 在仪表盘中根据温度值显示不同颜色
        >>> color = rgb_from_cmap('coolwarm', 25.5, (0, 50))
        >>> print(color)  # 输出: "rgb(120,150,200)"
    
    技术细节:
        - 使用线性插值将value映射到[0,1]区间
        - 当value_range的最小值等于最大值时，默认使用0.5作为标准化值
        - RGB值通过四舍五入转换为0-255的整数范围
    """
    import matplotlib.pyplot as plt

    if value_range[0] == value_range[1]:  # 检查数值范围是否为零（最小值等于最大值）
        norm_value = 0.5  # 当范围为零时，使用中间值0.5避免除零错误
    else:
        norm_value = (value - value_range[0]) / (value_range[1] - value_range[0])  # 将数值标准化到[0,1]区间
    cmap = plt.get_cmap(cmap_name)  # 获取指定名称的matplotlib颜色映射对象
    return "rgb(%d,%d,%d)" % tuple(np.round(np.asarray(cmap(norm_value))[:3] * 255))  # 将标准化值映射到颜色，提取RGB分量，转换为0-255整数并格式化为字符串


def adjust_opacity(color: tp.Any, opacity: float) -> str:
    """
    调整颜色的不透明度（Alpha通道）

    Args:
        color (Any): 输入颜色，支持matplotlib颜色名称、十六进制字符串、RGB元组等格式
        opacity (float): 不透明度值，范围0.0（完全透明）到1.0（完全不透明）
    
    Returns:
        str: RGBA颜色字符串，格式为"rgba(r,g,b,a)"
    
    应用示例:
        >>> green_semi = adjust_opacity('green', 0.75)
        >>> print(green_semi)  # 输出: "rgba(0,128,0,0.7500)"
    """
    import matplotlib.colors as mc

    rgb = mc.to_rgb(color)  # 将输入颜色转换为标准RGB元组格式（0-1范围）
    return 'rgba(%d,%d,%d,%.4f)' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), opacity)  # 将RGB值转换为0-255整数，组合透明度形成RGBA字符串


def adjust_lightness(color: tp.Any, amount: float = 0.7) -> str:
    """
    调整颜色的明亮程度（亮度）
    
    Args:
        color (Any): 输入颜色，支持matplotlib颜色名称、十六进制字符串、RGB元组等
        amount (float, optional): 亮度调整系数，默认0.7
                                - 0.0: 最暗（接近黑色）
                                - 1.0: 保持原始亮度
                                - >1.0: 更亮
    
    Returns:
        str: 调整后的RGB颜色字符串，格式为"rgb(r,g,b)"
    
    应用示例:
        >>> # 创建按钮的悬停效果
        >>> original_color = 'blue'
        >>> hover_color = adjust_lightness(original_color, 1.2)  # 更亮的蓝色
        >>> print(hover_color)  # 输出类似: "rgb(51,102,255)"
        >>> # 交易信号可视化中的强度表示
        >>> strong_signal = adjust_lightness('red', 1.0)    # 原始强度
        >>> weak_signal = adjust_lightness('red', 0.5)      # 较暗，表示弱信号
    
    算法原理:
        1. 将RGB颜色转换为HSL（色相、饱和度、亮度）色彩空间
        2. 保持色相和饱和度不变，仅调整亮度分量
        3. 将调整后的HSL颜色转换回RGB格式
        4. 确保RGB值在有效范围[0,255]内
    """
    import matplotlib.colors as mc
    import colorsys

    try:  
        c = mc.cnames[color]  
    except:
        c = color  # 直接使用输入的颜色值
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))  # 将RGB颜色转换为HLS（色相、亮度、饱和度）格式
    rgb = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])  # 调整亮度分量并转换回RGB，确保亮度值在[0,1]范围内
    return 'rgb(%d,%d,%d)' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))  # 将RGB值转换为0-255整数并格式化为字符串
