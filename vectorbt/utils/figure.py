# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
图形对象构建与显示工具模块。通过扩展plotly的原生Figure类，提供统一的图形创建、配置和显示接口。
"""

from plotly.graph_objects import Figure as _Figure, FigureWidget as _FigureWidget
from plotly.subplots import make_subplots as _make_subplots

from vectorbt import _typing as tp
from vectorbt.utils.config import merge_dicts


def get_domain(ref: str, fig: tp.BaseFigure) -> tp.Tuple[int, int]:
    """
    获取图形中指定坐标轴的显示域范围
    
    此函数用于提取plotly图形中特定坐标轴的domain属性，domain定义了坐标轴
    在图形画布中的相对位置和大小。
    
    Args:
        ref (str): 坐标轴引用标识符，如'x', 'y', 'x2', 'y3'等
                  'x'表示主X轴，'x2'表示第二个X轴，以此类推
        fig (BaseFigure): plotly图形对象，包含布局信息的完整图表
    
    Returns:
        Tuple[int, int]: 坐标轴域的范围元组(起始位置, 结束位置)
                        值为0-1之间的浮点数，表示在画布中的相对位置
                        (0, 1)表示占据整个画布宽度或高度
    
    应用示例:
        >>> import plotly.graph_objects as go
        >>> fig = go.Figure()
        >>> fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        >>> x_domain = get_domain('x', fig)
        >>> print(x_domain)  # 输出: (0, 1)
        
        >>> # 在多子图场景中获取特定子图的坐标域
        >>> from plotly.subplots import make_subplots
        >>> fig = make_subplots(rows=2, cols=1)
        >>> y2_domain = get_domain('y2', fig)  # 获取第二个Y轴的域范围
    
    技术细节:
        - domain属性定义坐标轴在标准化坐标系统中的位置
        - 默认情况下，单个图表的domain为(0, 1)，即占据全部空间
        - 多子图情况下，每个子图有独立的domain范围
    """
    axis = ref[0] + 'axis' + ref[1:]  # 根据坐标轴引用构建完整的轴名称，如'x'变为'xaxis'，'x2'变为'xaxis2'
    if axis in fig.layout:  # 检查图形布局中是否存在指定的坐标轴配置
        if 'domain' in fig.layout[axis]:  # 检查该坐标轴是否定义了domain属性
            if fig.layout[axis]['domain'] is not None:  # 确保domain属性不为空值
                return fig.layout[axis]['domain']  # 返回该坐标轴的实际domain范围
    return 0, 1  # 如果未找到domain配置，返回默认的全画布范围(0, 1)


class FigureMixin:
    """
    图形显示功能混入类
    
    提供了图形对象的多种显示方法，支持不同的渲染格式和输出方式。
    这个混入类被Figure和FigureWidget共同继承，确保两种图形对象都具有
    一致的显示接口和功能。
    
    应用场景：
    >>> fig = make_figure()  # 创建图形对象
    >>> fig.show_png()      # 以PNG格式显示，适合静态报告
    >>> fig.show_svg()      # 以SVG格式显示，适合高质量打印
    >>> fig.show()          # 默认格式显示，通常为交互式HTML
    """
    
    def show(self, *args, **kwargs) -> None:
        """
        显示图形的抽象方法
        
        这是一个抽象方法，需要在具体的图形类中实现。定义了显示图形的基本接口，
        子类必须根据自己的特性来实现具体的显示逻辑。
        """
        raise NotImplementedError

    def show_png(self, **kwargs) -> None:
        """
        以PNG格式显示图形
        
        PNG格式适用于需要静态图像的场景，如报告生成、文档嵌入等。
        相比交互式格式，PNG具有更好的兼容性和更小的文件大小。
        
        Args:
            **kwargs: 传递给show方法的其他参数
        """
        self.show(renderer="png", **kwargs)  # 调用show方法并指定PNG渲染器

    def show_svg(self, **kwargs) -> None:
        """
        以SVG格式显示图形
        
        SVG是矢量图形格式，具有无损缩放特性，适合高质量打印和专业出版。
        在需要精确图形质量或大尺寸输出的场景中非常有用。
        
        Args:
            **kwargs: 传递给show方法的其他参数
        """
        self.show(renderer="svg", **kwargs)  # 调用show方法并指定SVG渲染器


class Figure(_Figure, FigureMixin):
    """
    vectorbt增强版静态图形类。专门用于创建静态图形。
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        初始化vectorbt增强版Figure对象
        
        在plotly原生Figure基础上，自动集成vectorbt的全局配置系统，
        
        Args:
            *args: 传递给plotly.Figure的位置参数
            **kwargs: 传递给plotly.Figure的关键字参数，其中layout会被特殊处理
        
        配置合并策略：
        1. 获取vectorbt全局plotting配置
        2. 提取用户传入的layout配置
        3. 使用merge_dicts智能合并配置，用户配置优先
        4. 应用合并后的配置到图形对象
        """
        from vectorbt._settings import settings  # 导入vectorbt全局设置模块
        plotting_cfg = settings['plotting']  # 获取绘图相关的配置字典

        layout = kwargs.pop('layout', {})  # 从参数中提取layout配置，如果没有则使用空字典
        super().__init__(*args, **kwargs)  # 调用plotly.Figure的初始化方法，传入剩余参数
        self.update_layout(**merge_dicts(plotting_cfg['layout'], layout))  # 合并vectorbt默认布局配置和用户自定义配置并应用

    def show(self, *args, **kwargs) -> None:
        """
        显示图形，集成vectorbt显示配置。重写plotly原生的show方法，自动应用vectorbt的显示配置。
        
        Args:
            *args: 传递给plotly.Figure.show的位置参数
            **kwargs: 传递给plotly.Figure.show的关键字参数
        
        配置优先级：
        1. 用户传入的kwargs（最高优先级）
        2. vectorbt全局show_kwargs配置
        3. 从图形布局中提取的尺寸参数
        """
        from vectorbt._settings import settings  # 导入vectorbt全局设置模块
        plotting_cfg = settings['plotting']  # 获取绘图相关的配置字典

        fig_kwargs = dict(width=self.layout.width, height=self.layout.height)  # 从当前图形布局中提取宽度和高度参数
        show_kwargs = merge_dicts(fig_kwargs, plotting_cfg['show_kwargs'], kwargs)  # 按优先级合并显示参数：图形尺寸 < 全局配置 < 用户参数
        _Figure.show(self, *args, **show_kwargs)  # 调用plotly原生Figure的show方法，使用合并后的参数


class FigureWidget(_FigureWidget, FigureMixin):
    """
    vectorbt增强版交互式图形组件类。扩展了plotly原生的FigureWidget类，专门用于创建交互式图形组件。
    
    使用示例：
        >>> # 创建实时价格监控图表
        >>> widget = FigureWidget()
        >>> widget.add_trace(go.Scatter(x=[], y=[], mode='lines'))
        >>> # 支持动态更新：widget.data[0].x += [new_time]
        
        >>> # 创建交互式技术指标分析工具
        >>> widget = FigureWidget(layout=dict(title="Interactive Analysis"))
        >>> # 用户可以缩放、选择时间范围等
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        初始化vectorbt增强版FigureWidget对象
        
        在plotly原生FigureWidget基础上，自动集成vectorbt的配置系统，
        创建具有一致样式和行为的交互式图形组件。
        
        Args:
            *args: 传递给plotly.FigureWidget的位置参数
            **kwargs: 传递给plotly.FigureWidget的关键字参数，layout会被特殊处理
        
        初始化流程：
        1. 导入vectorbt全局配置
        2. 提取用户layout配置
        3. 调用父类构造函数
        4. 智能合并并应用配置
        """
        from vectorbt._settings import settings  # 导入vectorbt全局设置模块
        plotting_cfg = settings['plotting']  # 获取绘图相关的配置字典

        layout = kwargs.pop('layout', {})  # 从参数中提取layout配置，如果没有则使用空字典
        super().__init__(*args, **kwargs)  # 调用plotly.FigureWidget的初始化方法
        self.update_layout(**merge_dicts(plotting_cfg['layout'], layout))  # 合并vectorbt默认布局配置和用户自定义配置并应用

    def show(self, *args, **kwargs) -> None:
        """
        显示交互式图形组件
        
        重写plotly原生的show方法，为FigureWidget提供与vectorbt配置系统
        集成的显示功能。
        
        Args:
            *args: 传递给显示方法的位置参数
            **kwargs: 传递给显示方法的关键字参数
        
        显示特性：
        - 自动应用vectorbt显示配置
        - 支持多种渲染格式
        - 保持交互性功能完整
        - 与Jupyter环境深度集成
        """
        from vectorbt._settings import settings  # 导入vectorbt全局设置模块
        plotting_cfg = settings['plotting']  # 获取绘图相关的配置字典

        fig_kwargs = dict(width=self.layout.width, height=self.layout.height)  # 从当前组件布局中提取宽度和高度参数
        show_kwargs = merge_dicts(fig_kwargs, plotting_cfg['show_kwargs'], kwargs)  # 按优先级合并显示参数
        _Figure.show(self, *args, **show_kwargs)  # 调用plotly基类的show方法，使用合并后的参数


def make_figure(*args, **kwargs) -> tp.BaseFigure:
    """
    vectorbt图形创建的统一入口。
    根据全局配置settings['plotting']['use_widgets']自动选择创建Figure（静态图形）或FigureWidget（交互式组件）。
    
    Args:
        *args: 传递给图形构造函数的位置参数
        **kwargs: 传递给图形构造函数的关键字参数
    
    Returns:
        BaseFigure: 根据配置返回Figure或FigureWidget实例
    
    应用示例：
        >>> # 在技术指标计算中创建图形
        >>> def plot_ma(data):
        ...     fig = make_figure()  # 自动选择图形类型
        ...     fig.add_trace(go.Scatter(x=data.index, y=data.values))
        ...     return fig
        配置管理：
            >>> # 在vectorbt设置中控制图形类型
            >>> vbt.settings.plotting['use_widgets'] = True   # 使用交互式组件
            >>> vbt.settings.plotting['use_widgets'] = False  # 使用静态图形
            >>> 
            >>> # 所有通过make_figure创建的图形都会自动遵循此配置
    """
    from vectorbt._settings import settings  
    plotting_cfg = settings['plotting']  

    if plotting_cfg['use_widgets']:  
        return FigureWidget(*args, **kwargs)  
    return Figure(*args, **kwargs)


def make_subplots(*args, **kwargs) -> tp.BaseFigure:
    """
    子图创建工厂函数。
    
    基于plotly的make_subplots功能，创建具有多个子图的复合图形。
    这个函数自动将plotly原生的子图结构包装为vectorbt的图形对象，
    
    Args:
        *args: 传递给plotly.subplots.make_subplots的位置参数
        **kwargs: 传递给plotly.subplots.make_subplots的关键字参数
    
    Returns:
        BaseFigure: 包含多个子图的vectorbt图形对象
    
    应用示例：
        >>> # 创建技术分析复合图表
        >>> fig = make_subplots(
        ...     rows=3, cols=1,
        ...     subplot_titles=['Price', 'Volume', 'RSI'],
        ...     vertical_spacing=0.05
        ... )
        >>> # 添加价格数据到第一个子图
        >>> fig.add_trace(go.Scatter(x=dates, y=prices), row=1, col=1)
        >>> # 添加成交量到第二个子图
        >>> fig.add_trace(go.Bar(x=dates, y=volumes), row=2, col=1)
        >>> # 添加RSI到第三个子图
        >>> fig.add_trace(go.Scatter(x=dates, y=rsi), row=3, col=1)
    
    技术实现：
        函数内部调用流程：
        1. plotly.make_subplots创建原生子图结构
        2. make_figure将其包装为vectorbt图形对象
        3. 自动应用vectorbt配置和样式
        4. 返回完整的可用图形对象
    """
    return make_figure(_make_subplots(*args, **kwargs))  # 使用plotly创建子图结构，然后通过make_figure包装为vectorbt图形对象
