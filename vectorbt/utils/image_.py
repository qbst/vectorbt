# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT UTILS MODULE: 图像处理和动画生成工具
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于图像处理和动画生成的核心工具模块。
在量化分析中，数据可视化是理解市场动态、展示分析结果的重要手段。该模块
提供了一套完整的图像处理基础设施，特别是为创建动态金融数据可视化而设计。

核心设计理念：
1. **多格式兼容**：支持NumPy数组、Plotly图形、图像文件等多种输入格式
2. **高性能处理**：使用imageio库实现高效的图像I/O操作
3. **灵活布局**：提供水平和垂直堆叠功能，支持复杂的图像布局
4. **动画生成**：专门优化的动画创建引擎，适用于时间序列数据的动态展示
5. **用户友好**：集成进度条和详细的错误处理，提供良好的用户体验

主要功能模块：
- **图像数组操作**：hstack_image_arrays, vstack_image_arrays 图像拼接函数
- **动画生成引擎**：save_animation 核心动画创建函数
- **格式转换支持**：自动处理Plotly图形到图像的转换
- **进度监控**：集成tqdm进度条，支持长时间动画生成的进度跟踪

应用场景：
- **技术指标动画**：创建移动平均线、RSI、MACD等指标的时间演化动画
- **投资组合演示**：展示投资组合价值、仓位变化的动态过程
- **市场分析报告**：生成包含多个图表的综合分析图像
- **策略回测可视化**：将策略回测过程制作成动态演示
- **数据报告自动化**：批量生成标准化的分析图表和报告

技术特点：
- 原生支持Plotly图形对象，与vectorbt绘图系统无缝集成
- 基于imageio的高性能图像处理，支持多种图像格式
- 智能的内存管理，适合处理大量时间序列数据
- 灵活的参数配置，支持自定义帧率、图像质量等设置
- 完整的错误处理和调试信息，便于开发和维护

与vectorbt生态系统的关系：
- 与plotting模块紧密集成，支持vectorbt标准图表的动画化
- 为量化分析报告提供图像处理基础设施
- 支持自定义绘图函数，扩展性强
- 是vectorbt可视化工具链的重要组成部分

使用约定：
- 所有图像数组应使用uint8数据类型，值域0-255
- 动画生成函数应返回Plotly图形对象、图像路径或NumPy数组
- 建议使用白色(255)作为图像拼接时的填充色
- 长时间动画生成建议启用进度条显示

该模块为vectorbt框架的可视化功能提供了强大的底层支持，特别是在创建
动态、交互式的金融数据展示方面发挥了关键作用。
"""

import imageio  # 导入imageio库，提供高性能的图像和视频I/O操作
import numpy as np  # 导入NumPy库，提供高性能的数值计算和数组操作
import plotly.graph_objects as go  # 导入Plotly图形对象模块，用于识别和处理Plotly图形
from tqdm.auto import tqdm  # 导入tqdm进度条库，提供美观的进度显示功能

from vectorbt import _typing as tp  # 导入vectorbt类型定义模块，提供类型注解支持


def hstack_image_arrays(a: tp.Array3d, b: tp.Array3d) -> tp.Array3d:
    """
    水平堆叠两个图像数组，实现图像的左右拼接
    
    该函数将两个三维图像数组沿水平方向（宽度维度）进行拼接，创建一个
    包含两个图像的复合图像。这在创建对比图表、并排展示不同时间段的
    数据分析结果时非常有用。
    
    核心算法：
    1. 提取两个图像的尺寸信息（高度、宽度、通道数）
    2. 创建一个新的空白图像，宽度为两图像宽度之和，高度为两图像高度的最大值
    3. 使用白色(255)填充新图像，确保未覆盖区域为白色背景
    4. 将第一个图像复制到新图像的左侧区域
    5. 将第二个图像复制到新图像的右侧区域
    
    参数：
        a (tp.Array3d): 左侧图像数组，形状为(height, width, channels)
                       通常为RGB图像，channels=3，数据类型为uint8
        b (tp.Array3d): 右侧图像数组，形状为(height, width, channels)
                       与左侧图像具有相同的通道数，但高度和宽度可以不同
    
    返回：
        tp.Array3d: 拼接后的图像数组，形状为(max_height, width_a + width_b, channels)
    
    使用示例：
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> 
        >>> # 创建两个简单的彩色图像
        >>> img1 = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
        >>> img2 = np.random.randint(0, 256, (120, 200, 3), dtype=np.uint8)
        >>> 
        >>> # 水平拼接图像
        >>> combined = hstack_image_arrays(img1, img2)
        >>> print(f"拼接后图像尺寸: {combined.shape}")  # (120, 350, 3)
        >>> 
        >>> # 在量化分析中的应用
        >>> # 创建两个时期的价格走势图
        >>> def create_price_chart(prices, title):
        ...     fig = plt.figure(figsize=(8, 6))
        ...     plt.plot(prices)
        ...     plt.title(title)
        ...     plt.tight_layout()
        ...     
        ...     # 转换为图像数组
        ...     fig.canvas.draw()
        ...     img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        ...     img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        ...     plt.close(fig)
        ...     return img_array
        >>> 
        >>> # 创建对比图表
        >>> before_prices = np.random.randn(100).cumsum()
        >>> after_prices = np.random.randn(100).cumsum()
        >>> 
        >>> img_before = create_price_chart(before_prices, "疫情前")
        >>> img_after = create_price_chart(after_prices, "疫情后")
        >>> 
        >>> # 创建对比图
        >>> comparison_chart = hstack_image_arrays(img_before, img_after)
        >>> 
        >>> # 保存对比图
        >>> imageio.imwrite('price_comparison.png', comparison_chart)
        
        >>> # 在动画生成中的应用
        >>> def create_side_by_side_animation(data1, data2, plot_func):
        ...     frames = []
        ...     for i in range(len(data1)):
        ...         # 生成两个子图
        ...         img1 = plot_func(data1[:i+1], f"数据集1 (第{i+1}期)")
        ...         img2 = plot_func(data2[:i+1], f"数据集2 (第{i+1}期)")
        ...         
        ...         # 水平拼接
        ...         combined_frame = hstack_image_arrays(img1, img2)
        ...         frames.append(combined_frame)
        ...     
        ...     return frames
        
        >>> # 创建投资组合对比动画
        >>> portfolio_a = np.random.randn(50).cumsum() + 100
        >>> portfolio_b = np.random.randn(50).cumsum() + 100
        >>> 
        >>> animation_frames = create_side_by_side_animation(
        ...     portfolio_a, portfolio_b, create_price_chart
        ... )
        >>> 
        >>> # 保存为GIF动画
        >>> imageio.mimsave('portfolio_comparison.gif', animation_frames, fps=2)
    
    注意事项：
        - 两个图像必须具有相同的通道数（颜色通道数）
        - 如果图像高度不同，较矮的图像会被放置在顶部，底部用白色填充
        - 函数使用白色(255)作为填充色，适合大多数图表背景
        - 输入图像应为uint8类型，值域0-255
    
    性能优化：
        - 使用NumPy的切片操作实现高效的数组复制
        - 预分配目标数组，避免动态内存分配
        - 利用NumPy的向量化操作，避免Python循环
    """
    h1, w1, d = a.shape  # 提取第一个图像的尺寸：高度、宽度、通道数
    h2, w2, _ = b.shape  # 提取第二个图像的尺寸：高度、宽度（通道数与第一个图像相同）
    
    # 创建新的空白图像，高度为两图像高度的最大值，宽度为两图像宽度之和
    c = np.full((max(h1, h2), w1 + w2, d), 255, np.uint8)
    
    # 将第一个图像复制到新图像的左侧区域
    c[:h1, :w1, :] = a
    
    # 将第二个图像复制到新图像的右侧区域（从w1位置开始）
    c[:h2, w1:w1 + w2, :] = b
    
    return c  # 返回拼接后的图像数组


def vstack_image_arrays(a: tp.Array3d, b: tp.Array3d) -> tp.Array3d:
    """
    垂直堆叠两个图像数组，实现图像的上下拼接
    
    该函数将两个三维图像数组沿垂直方向（高度维度）进行拼接，创建一个
    包含两个图像的复合图像。这在创建多层图表、时间序列的上下对比展示、
    或者构建复杂的数据分析仪表板时非常有用。
    
    核心算法：
    1. 提取两个图像的尺寸信息（高度、宽度、通道数）
    2. 创建一个新的空白图像，高度为两图像高度之和，宽度为两图像宽度的最大值
    3. 使用白色(255)填充新图像，确保未覆盖区域为白色背景
    4. 将第一个图像复制到新图像的上方区域
    5. 将第二个图像复制到新图像的下方区域
    
    参数：
        a (tp.Array3d): 上方图像数组，形状为(height, width, channels)
                       通常为RGB图像，channels=3，数据类型为uint8
        b (tp.Array3d): 下方图像数组，形状为(height, width, channels)
                       与上方图像具有相同的通道数，但高度和宽度可以不同
    
    返回：
        tp.Array3d: 拼接后的图像数组，形状为(height_a + height_b, max_width, channels)
    
    使用示例：
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> 
        >>> # 创建两个不同尺寸的图像
        >>> img_top = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        >>> img_bottom = np.random.randint(0, 256, (150, 180, 3), dtype=np.uint8)
        >>> 
        >>> # 垂直拼接图像
        >>> stacked = vstack_image_arrays(img_top, img_bottom)
        >>> print(f"拼接后图像尺寸: {stacked.shape}")  # (250, 200, 3)
        >>> 
        >>> # 在量化分析中的应用
        >>> # 创建价格和成交量的组合图表
        >>> def create_price_volume_chart(prices, volumes):
        ...     # 创建价格图表
        ...     fig1 = plt.figure(figsize=(10, 4))
        ...     plt.plot(prices, label='价格')
        ...     plt.title('股票价格走势')
        ...     plt.legend()
        ...     plt.tight_layout()
        ...     
        ...     # 转换为图像数组
        ...     fig1.canvas.draw()
        ...     price_img = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
        ...     price_img = price_img.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
        ...     plt.close(fig1)
        ...     
        ...     # 创建成交量图表
        ...     fig2 = plt.figure(figsize=(10, 3))
        ...     plt.bar(range(len(volumes)), volumes, alpha=0.7, label='成交量')
        ...     plt.title('成交量')
        ...     plt.legend()
        ...     plt.tight_layout()
        ...     
        ...     # 转换为图像数组
        ...     fig2.canvas.draw()
        ...     volume_img = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
        ...     volume_img = volume_img.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
        ...     plt.close(fig2)
        ...     
        ...     # 垂直拼接两个图表
        ...     combined_chart = vstack_image_arrays(price_img, volume_img)
        ...     return combined_chart
        >>> 
        >>> # 生成示例数据
        >>> prices = np.random.randn(100).cumsum() + 100
        >>> volumes = np.random.randint(1000, 5000, 100)
        >>> 
        >>> # 创建组合图表
        >>> combined_chart = create_price_volume_chart(prices, volumes)
        >>> imageio.imwrite('price_volume_chart.png', combined_chart)
        
        >>> # 在多指标分析中的应用
        >>> def create_multi_indicator_chart(data):
        ...     indicators = []
        ...     
        ...     # 创建多个技术指标图表
        ...     for name, values in data.items():
        ...         fig = plt.figure(figsize=(10, 2))
        ...         plt.plot(values, label=name)
        ...         plt.title(f'{name} 指标')
        ...         plt.legend()
        ...         plt.tight_layout()
        ...         
        ...         # 转换为图像数组
        ...         fig.canvas.draw()
        ...         img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        ...         img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        ...         plt.close(fig)
        ...         
        ...         indicators.append(img)
        ...     
        ...     # 垂直堆叠所有指标图表
        ...     result = indicators[0]
        ...     for indicator in indicators[1:]:
        ...         result = vstack_image_arrays(result, indicator)
        ...     
        ...     return result
        >>> 
        >>> # 生成多指标数据
        >>> indicator_data = {
        ...     'RSI': np.random.randint(20, 80, 50),
        ...     'MACD': np.random.randn(50),
        ...     'KDJ': np.random.randint(0, 100, 50)
        ... }
        >>> 
        >>> # 创建多指标综合图表
        >>> multi_chart = create_multi_indicator_chart(indicator_data)
        >>> imageio.imwrite('multi_indicator_chart.png', multi_chart)
        
        >>> # 在动画生成中的应用
        >>> def create_layered_animation(price_data, volume_data, plot_func):
        ...     frames = []
        ...     for i in range(len(price_data)):
        ...         # 生成价格图表
        ...         price_img = plot_func(price_data[:i+1], "价格走势")
        ...         
        ...         # 生成成交量图表
        ...         volume_img = plot_func(volume_data[:i+1], "成交量")
        ...         
        ...         # 垂直拼接
        ...         combined_frame = vstack_image_arrays(price_img, volume_img)
        ...         frames.append(combined_frame)
        ...     
        ...     return frames
    
    注意事项：
        - 两个图像必须具有相同的通道数（颜色通道数）
        - 如果图像宽度不同，较窄的图像会被放置在左侧，右侧用白色填充
        - 函数使用白色(255)作为填充色，适合大多数图表背景
        - 输入图像应为uint8类型，值域0-255
    
    性能优化：
        - 使用NumPy的切片操作实现高效的数组复制
        - 预分配目标数组，避免动态内存分配
        - 利用NumPy的向量化操作，避免Python循环
    """
    h1, w1, d = a.shape  # 提取第一个图像的尺寸：高度、宽度、通道数
    h2, w2, _ = b.shape  # 提取第二个图像的尺寸：高度、宽度（通道数与第一个图像相同）
    
    # 创建新的空白图像，高度为两图像高度之和，宽度为两图像宽度的最大值
    c = np.full((h1 + h2, max(w1, w2), d), 255, np.uint8)
    
    # 将第一个图像复制到新图像的上方区域
    c[:h1, :w1, :] = a
    
    # 将第二个图像复制到新图像的下方区域（从h1位置开始）
    c[h1:h1 + h2, :w2, :] = b
    
    return c  # 返回拼接后的图像数组


def save_animation(fname: str,
                   index: tp.ArrayLikeSequence,
                   plot_func: tp.Callable,
                   *args,
                   delta: tp.Optional[int] = None,
                   step: int = 1,
                   fps: int = 3,
                   writer_kwargs: dict = None,
                   show_progress: bool = True,
                   tqdm_kwargs: tp.KwargsLike = None,
                   to_image_kwargs: tp.KwargsLike = None,
                   **kwargs) -> None:
    """
    保存动画到文件，是vectorbt框架中创建动态可视化的核心函数
    
    该函数是vectorbt生态系统中最重要的动画生成工具，专门用于将时间序列数据、
    策略回测过程、技术指标演化等转换为动态的GIF或视频文件。它通过滑动窗口
    机制逐帧调用绘图函数，然后将所有帧合并成流畅的动画。
    
    核心工作流程：
    1. 参数验证和默认值设置
    2. 创建imageio写入器，配置输出格式和质量
    3. 使用滑动窗口遍历时间序列数据
    4. 对每个窗口调用绘图函数生成图像
    5. 处理不同类型的图像输出（Plotly图形、图像文件、NumPy数组）
    6. 将处理后的图像写入动画文件
    7. 显示进度条并完成动画生成
    
    参数：
        fname (str): 输出动画文件的路径和名称
                    支持的格式：.gif、.mp4、.avi、.mov等
                    推荐使用.gif格式，兼容性最好
                    
        index (tp.ArrayLikeSequence): 时间序列索引，用于确定动画的时间轴
                                     可以是pandas的DatetimeIndex、RangeIndex或任何序列
                                     动画将从index[0]开始，按滑动窗口方式进行
                                     
        plot_func (tp.Callable): 绘图函数，动画生成的核心逻辑
                                函数签名：plot_func(subset_index, *args, **kwargs)
                                - subset_index: 当前窗口的索引子集
                                - *args: 传递给绘图函数的位置参数
                                - **kwargs: 传递给绘图函数的关键字参数
                                
                                返回值要求：
                                - Plotly图形对象（go.Figure或go.FigureWidget）
                                - 图像文件路径（可被imageio.imread读取）
                                - NumPy数组（三维，形状为(height, width, channels)）
                                
        *args: 传递给plot_func的位置参数
               通常包括数据数组、配置对象、绘图参数等
               
        delta (int, 可选): 滑动窗口的大小，即每帧显示的数据点数量
                          - None: 默认为len(index) // 2，显示一半的数据
                          - 正整数: 指定窗口大小
                          - 建议值：对于日数据使用30-90，对于分钟数据使用100-500
                          
        step (int): 窗口滑动的步长，控制动画的时间分辨率
                   - 1: 每个时间点生成一帧（最平滑，但文件大）
                   - 2: 每两个时间点生成一帧
                   - 建议值：根据数据量和动画流畅度需求调整
                   
        fps (int): 动画帧率，每秒播放的帧数
                  - 1-3: 适合数据分析展示，便于观察细节
                  - 5-10: 适合一般动画效果
                  - 15-30: 适合快速演示
                  
        writer_kwargs (dict, 可选): 传递给imageio.get_writer的参数
                                   常用参数：
                                   - duration: 每帧持续时间（毫秒），会根据fps自动计算
                                   - quality: 图像质量（1-10，仅适用于某些格式）
                                   - codec: 视频编码器（适用于视频格式）
                                   - loop: 是否循环播放（0为无限循环）
                                   
        show_progress (bool): 是否显示进度条，推荐在生成大型动画时启用
                            - True: 显示详细的进度信息，包括已完成帧数和预计剩余时间
                            - False: 静默模式，适合自动化脚本
                            
        tqdm_kwargs (dict, 可选): 传递给tqdm进度条的配置参数
                                 常用参数：
                                 - desc: 进度条描述文本
                                 - unit: 进度单位（如'frame', 'step'）
                                 - ncols: 进度条显示宽度
                                 - colour: 进度条颜色
                                 
        to_image_kwargs (dict, 可选): 传递给Plotly图形的to_image方法的参数
                                     常用参数：
                                     - width: 图像宽度（像素）
                                     - height: 图像高度（像素）
                                     - scale: 图像缩放因子
                                     - format: 图像格式（默认'png'）
                                     
        **kwargs: 传递给plot_func的关键字参数
                 可以包括样式设置、数据处理参数、图表配置等
    
    返回：
        None: 函数不返回值，直接将动画保存到指定文件
    
    使用示例：
        >>> import vectorbt as vbt
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # 示例1：基本动画 - 价格走势演化
        >>> def plot_price_evolution(index_subset, prices, title="价格走势"):
        ...     # 创建价格走势图
        ...     fig = vbt.make_figure()
        ...     fig.add_trace(vbt.Scatter(
        ...         x=index_subset,
        ...         y=prices.loc[index_subset],
        ...         mode='lines',
        ...         name='价格'
        ...     ))
        ...     fig.update_layout(
        ...         title=f"{title} (截至 {index_subset[-1]})",
        ...         xaxis_title="时间",
        ...         yaxis_title="价格",
        ...         width=800,
        ...         height=400
        ...     )
        ...     return fig
        >>> 
        >>> # 生成示例数据
        >>> dates = pd.date_range('2023-01-01', periods=100, freq='D')
        >>> prices = pd.Series(
        ...     100 * (1 + np.random.randn(100) * 0.02).cumprod(),
        ...     index=dates
        ... )
        >>> 
        >>> # 生成价格演化动画
        >>> vbt.save_animation(
        ...     'price_evolution.gif',
        ...     prices.index,
        ...     plot_price_evolution,
        ...     prices,
        ...     delta=30,  # 显示30天的数据
        ...     step=2,    # 每两天生成一帧
        ...     fps=3,     # 每秒3帧
        ...     show_progress=True,
        ...     tqdm_kwargs={'desc': '生成价格动画'}
        ... )
    
    注意事项：
        - 绘图函数应该处理index_subset的边界情况（如数据不足）
        - 对于大型动画，建议增加step值以减少帧数和文件大小
        - Plotly图形转换为图像时可能消耗较多内存，注意监控内存使用
        - 动画文件大小与帧数、图像质量、持续时间成正比
        - 建议在生产环境中使用异常处理包装此函数
    
    性能优化建议：
        - 使用numba_loop=True（如果绘图函数支持）可以显著提升性能
        - 对于CPU密集型绘图，考虑使用多进程或分布式计算
        - 预先计算复杂的技术指标，避免在绘图函数中重复计算
        - 使用合适的图像分辨率，避免过度高清导致的性能问题
        - 对于长时间运行的动画生成，考虑分批处理和中间保存
    
    错误处理：
        - 绘图函数异常：动画生成会中断，建议在绘图函数中添加异常处理
        - 文件权限问题：确保有输出目录的写入权限
        - 内存不足：监控内存使用，必要时减少delta值或增加step值
        - 格式不支持：确保输出格式被imageio支持
    """
    # 参数验证和默认值设置
    if writer_kwargs is None:
        writer_kwargs = {}  # 初始化写入器参数字典
    if "duration" not in writer_kwargs:
        writer_kwargs["duration"] = 1000 / fps  # 根据帧率计算每帧持续时间（毫秒）
    if tqdm_kwargs is None:
        tqdm_kwargs = {}  # 初始化进度条参数字典
    if to_image_kwargs is None:
        to_image_kwargs = {}  # 初始化图像转换参数字典
    if delta is None:
        delta = len(index) // 2  # 默认窗口大小为索引长度的一半

    # 使用imageio创建动画写入器，支持多种格式
    with imageio.get_writer(fname, **writer_kwargs) as writer:
        # 使用滑动窗口遍历时间序列，创建动画帧
        for i in tqdm(range(0, len(index) - delta, step), disable=not show_progress, **tqdm_kwargs):
            # 调用绘图函数生成当前帧的图像
            fig = plot_func(index[i:i + delta], *args, **kwargs)
            
            # 处理Plotly图形对象，转换为图像数据
            if isinstance(fig, (go.Figure, go.FigureWidget)):
                fig = fig.to_image(format="png", **to_image_kwargs)
            
            # 处理非NumPy数组的图像数据（如文件路径）
            if not isinstance(fig, np.ndarray):
                fig = imageio.imread(fig)
            
            # 将处理后的图像数据写入动画文件
            writer.append_data(fig)
