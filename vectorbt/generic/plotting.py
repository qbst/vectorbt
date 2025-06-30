# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
基础绘图功能模块。提供了一套完整的基础绘图工具类。

主要功能组件：
- TraceUpdater：图表轨迹更新的基础类，提供统一的数据更新接口
- Gauge：仪表盘图表，用于显示关键指标的当前值和历史变化
- Bar：柱状图，适合类别数据的对比展示
- Scatter：散点图，用于探索变量间的相关关系
- Histogram：直方图，用于数据分布分析
- Box：箱线图，用于统计分布和异常值检测
- Heatmap：热力图，用于多维数据的相关性可视化
- Volume：3D体积图，用于三维数据的立体展示
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

from vectorbt import _typing as tp
from vectorbt.base import reshape_fns
from vectorbt.utils import checks
from vectorbt.utils.array_ import renormalize
from vectorbt.utils.colors import rgb_from_cmap
from vectorbt.utils.config import Configured, resolve_dict
from vectorbt.utils.figure import make_figure


def clean_labels(labels: tp.ArrayLikeSequence) -> tp.ArrayLikeSequence:
    """
    清理和标准化图表标签
    
    此函数解决了plotly不支持pandas MultiIndex等复杂索引结构的问题，
    将各种复杂的标签格式转换为plotly可接受的简单格式。
    
    Args:
        labels: 需要清理的标签序列，可以是pandas Index、MultiIndex或普通数组
    
    Returns:
        清理后的标签序列，保证与plotly兼容
    
    应用示例：
        >>> import pandas as pd
        >>> # 处理MultiIndex情况
        >>> multi_idx = pd.MultiIndex.from_tuples([('AAPL', '2023'), ('GOOGL', '2023')])
        >>> clean_labels(multi_idx)
        ['(AAPL, 2023)', '(GOOGL, 2023)']
        
        >>> # 处理包含元组的普通列表
        >>> tuple_labels = [('股票', 'AAPL'), ('债券', 'TLT')]
        >>> clean_labels(tuple_labels)
        ['(股票, AAPL)', '(债券, TLT)']
    
    技术细节：
        - MultiIndex会被转换为flat_index格式
        - 元组标签会被转换为字符串表示
        - 保持其他格式的标签不变
    """
    if isinstance(labels, pd.MultiIndex):
        labels = labels.to_flat_index()  # 将MultiIndex转换为平坦索引格式
    if len(labels) > 0 and isinstance(labels[0], tuple):  # 检查标签是否包含元组
        labels = list(map(str, labels))  # 将所有元组转换为字符串格式
    return labels


class TraceUpdater:
    """
    图表轨迹更新器基类。所有可视化组件的基础类，提供了统一的图表更新接口和管理机制。
    
    示例：
        >>> # TraceUpdater通常不直接使用，而是通过子类
        >>> bar_chart = Bar(data=[[1, 2], [3, 4]], trace_names=['A', 'B'])
        >>> # 更新数据
        >>> bar_chart.update([[2, 3], [4, 5]])
        >>> # 获取图形对象
        >>> fig = bar_chart.fig
    """
    
    def __init__(self, fig: tp.BaseFigure, traces: tp.Tuple[BaseTraceType, ...]) -> None:
        """
        初始化轨迹更新器
        
        Args:
            fig: plotly图形对象，承载所有可视化内容的容器
            traces: 需要管理的轨迹对象元组，每个轨迹代表图表中的一个数据系列
        """
        self._fig = fig  # 保存图形对象的引用
        self._traces = traces  # 保存轨迹对象元组的引用

    @property
    def fig(self) -> tp.BaseFigure:
        """
        获取图形对象
        """
        return self._fig  # 返回图形对象

    @property
    def traces(self) -> tp.Tuple[BaseTraceType, ...]:
        """
        获取轨迹对象
        """
        return self._traces  # 返回轨迹对象元组

    def update(self, *args, **kwargs) -> None:
        """
        更新轨迹数据的抽象方法。需要在具体的图表类中实现。
        
        Args:
            *args: 位置参数，通常包含新的数据
            **kwargs: 关键字参数，用于额外的更新选项
        """
        raise NotImplementedError


class Gauge(Configured, TraceUpdater):
    """
    仪表盘图表类
    
    专门用于创建仪表盘样式的图表，特别适合展示关键绩效指标(KPI)、
    实时监控数据和单一数值的状态展示。该类提供了丰富的自定义选项，
    包括颜色映射、数值范围设置和实时更新功能。
    
    使用示例：
        >>> # 创建简单的仪表盘
        >>> gauge = Gauge(value=0.75, value_range=(0, 1), label='投资组合收益率')
        >>> gauge.fig.show()
        
        >>> # 创建带颜色映射的风险仪表盘
        >>> risk_gauge = Gauge(
        ...     value=25,
        ...     value_range=(0, 100),
        ...     label='风险评分',
        ...     cmap_name='RdYlGn_r'  # 反向红黄绿色映射
        ... )
        
        >>> # 实时更新数据
        >>> risk_gauge.update(35)  # 更新为新的风险值
    """
    
    def __init__(self,
                 value: tp.Optional[float] = None,
                 label: tp.Optional[str] = None,
                 value_range: tp.Optional[tp.Tuple[float, float]] = None,
                 cmap_name: str = 'Spectral',
                 trace_kwargs: tp.KwargsLike = None,
                 add_trace_kwargs: tp.KwargsLike = None,
                 fig: tp.Optional[tp.BaseFigure] = None,
                 **layout_kwargs) -> None:
        """
        创建仪表盘图表
        
        Args:
            value: 要显示的数值，如果为None则需要后续通过update方法设置
            label: 仪表盘的标题标签，显示在图表顶部
            value_range: 数值显示范围的元组(最小值, 最大值)，用于刻度设置
            cmap_name: matplotlib兼容的颜色映射名称，用于根据数值变化颜色
                      常用选项：'Spectral'(光谱色)、'RdYlGn'(红黄绿)、'viridis'等
            trace_kwargs: 传递给plotly.Indicator的关键字参数，用于自定义样式
            add_trace_kwargs: 传递给add_trace方法的参数，用于图层管理
            fig: 可选的现有图形对象，如果不提供则自动创建
            **layout_kwargs: 布局相关的关键字参数，用于自定义图表外观
        """
        Configured.__init__(  # 初始化配置管理功能
            self,
            value=value,
            label=label,
            value_range=value_range,
            cmap_name=cmap_name,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs
        )

        from vectorbt._settings import settings  # 导入vectorbt全局设置
        layout_cfg = settings['plotting']['layout']  # 获取布局配置

        if trace_kwargs is None:  # 如果没有提供轨迹参数
            trace_kwargs = {}  # 使用空字典作为默认值
        if add_trace_kwargs is None:  # 如果没有提供添加轨迹参数
            add_trace_kwargs = {}  # 使用空字典作为默认值

        if fig is None:  # 如果没有提供现有图形对象
            fig = make_figure()  # 创建新的图形对象
            if 'width' in layout_cfg:  # 如果配置中指定了宽度
                # 计算合适的宽度和高度比例
                fig.update_layout(
                    width=layout_cfg['width'] * 0.7,  # 宽度为默认值的70%
                    height=layout_cfg['width'] * 0.5,  # 高度为宽度的一半，保持美观比例
                    margin=dict(t=80)  # 顶部留出空间显示标题
                )
        fig.update_layout(**layout_kwargs)  # 应用用户自定义的布局参数

        indicator = go.Indicator(  # 创建plotly指示器对象
            domain=dict(x=[0, 1], y=[0, 1]),  # 设置显示域占据整个图表区域
            mode="gauge+number+delta",  # 设置显示模式：表盘+数字+变化量
            title=dict(text=label)  # 设置标题文本
        )
        indicator.update(**trace_kwargs)  # 应用用户自定义的轨迹参数
        fig.add_trace(indicator, **add_trace_kwargs)  # 将指示器添加到图形中

        TraceUpdater.__init__(self, fig, (fig.data[-1],))  # 初始化轨迹更新器，使用最后添加的轨迹
        self._value_range = value_range  # 保存数值范围配置
        self._cmap_name = cmap_name  # 保存颜色映射名称

        if value is not None:  # 如果提供了初始值
            self.update(value)  # 立即更新显示

    @property
    def value_range(self) -> tp.Tuple[float, float]:
        return self._value_range  # 返回当前的数值范围

    @property
    def cmap_name(self) -> str:
        return self._cmap_name  # 返回颜色映射名称

    def update(self, value: float) -> None:
        if self.value_range is None:  # 如果还没有设置数值范围
            self._value_range = value, value  # 将范围设置为当前值
        else:  # 如果已有范围
            # 扩展范围以包含新值，确保所有数据都能正确显示
            self._value_range = min(self.value_range[0], value), max(self.value_range[1], value)

        with self.fig.batch_update():  # 使用批量更新提高性能
            if self.value_range is not None:  # 如果有有效的数值范围
                self.traces[0].gauge.axis.range = self.value_range  # 更新表盘刻度范围
                if self.cmap_name is not None:  # 如果指定了颜色映射
                    # 根据当前值和范围计算对应的颜色
                    self.traces[0].gauge.bar.color = rgb_from_cmap(self.cmap_name, value, self.value_range)
            self.traces[0].delta.reference = self.traces[0].value  # 设置变化量的参考值为当前值
            self.traces[0].value = value  # 更新显示的数值


class Bar(Configured, TraceUpdater):
    """
    柱状图类

    使用示例：
        >>> # 创建简单的月度收益柱状图
        >>> monthly_returns = [[0.05, 0.03], [0.02, 0.07], [-0.01, 0.04]]
        >>> bar_chart = Bar(
        ...     data=monthly_returns,
        ...     trace_names=['策略A', '策略B'],
        ...     x_labels=['1月', '2月', '3月']
        ... )
        >>> bar_chart.fig.show()
    """
    
    def __init__(self,
                 data: tp.Optional[tp.ArrayLike] = None,
                 trace_names: tp.TraceNames = None,
                 x_labels: tp.Optional[tp.Labels] = None,
                 trace_kwargs: tp.KwargsLikeSequence = None,
                 add_trace_kwargs: tp.KwargsLike = None,
                 fig: tp.Optional[tp.BaseFigure] = None,
                 **layout_kwargs) -> None:
        """
        创建柱状图
        
        Args:
            data: 图表数据，可以是任何能转换为NumPy数组的格式
                 数据形状必须为(x_labels, trace_names)
                 支持列表、数组、DataFrame等多种格式
            trace_names: 数据系列名称，对应pandas的列名概念
                        可以是字符串、字符串列表，用于图例显示
            x_labels: X轴标签，对应pandas的索引概念
                     用于标识每个柱状组的类别或时间点
            trace_kwargs: 传递给plotly.Bar的关键字参数
                         可以是单个字典或字典列表，支持每个系列独立配置
                         常用参数：color、opacity、marker等
            add_trace_kwargs: 传递给add_trace方法的参数，用于图层控制
            fig: 可选的现有图形对象，支持在现有图表上添加柱状图
            **layout_kwargs: 布局参数，用于自定义图表外观
                            常用参数：title、width、height等
        """
        Configured.__init__(  # 初始化配置管理功能
            self,
            data=data,
            trace_names=trace_names,
            x_labels=x_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs
        )

        if trace_kwargs is None:  # 如果没有提供轨迹配置参数
            trace_kwargs = {}  # 使用空字典作为默认值
        if add_trace_kwargs is None:  # 如果没有提供添加轨迹参数
            add_trace_kwargs = {}  # 使用空字典作为默认值
        if data is not None:  # 如果提供了数据
            data = reshape_fns.to_2d_array(data)  # 将数据转换为2D数组格式
            if trace_names is not None:  # 如果同时提供了系列名称
                checks.assert_shape_equal(data, trace_names, (1, 0))  # 验证数据形状与系列名称数量是否匹配
        else:  # 如果没有提供数据
            if trace_names is None:  # 但也没有提供系列名称
                raise ValueError("At least data or trace_names must be passed")  # 抛出错误，至少需要提供其中一个
        if trace_names is None:  # 如果没有提供系列名称
            trace_names = [None] * data.shape[1]  # 创建与数据列数相等的None列表
        if isinstance(trace_names, str):  # 如果系列名称是单个字符串
            trace_names = [trace_names]  # 将其转换为列表格式
        if x_labels is not None:  # 如果提供了X轴标签
            x_labels = clean_labels(x_labels)  # 清理标签格式，处理MultiIndex等复杂情况

        if fig is None:  # 如果没有提供现有图形对象
            fig = make_figure()  # 创建新的图形对象
        fig.update_layout(**layout_kwargs)  # 应用用户提供的布局配置

        for i, trace_name in enumerate(trace_names):  # 遍历每个数据系列
            _trace_kwargs = resolve_dict(trace_kwargs, i=i)  # 解析当前系列的配置参数
            trace_name = _trace_kwargs.pop('name', trace_name)  # 从配置中提取名称，如果没有则使用默认名称
            if trace_name is not None:  # 如果有有效的系列名称
                trace_name = str(trace_name)  # 确保名称为字符串格式
            bar = go.Bar(  # 创建plotly柱状图对象
                x=x_labels,  # 设置X轴标签
                name=trace_name,  # 设置系列名称
                showlegend=trace_name is not None  # 根据是否有名称决定是否显示图例
            )
            bar.update(**_trace_kwargs)  # 应用当前系列的自定义配置
            fig.add_trace(bar, **add_trace_kwargs)  # 将柱状图添加到图形中

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names):])  # 初始化轨迹更新器，管理新添加的轨迹

        if data is not None:  # 如果提供了初始数据
            self.update(data)  # 立即更新图表显示

    def update(self, data: tp.ArrayLike) -> None:
        """
        更新柱状图数据
        """
        data = reshape_fns.to_2d_array(data)  # 将输入数据转换为标准的2D数组格式
        with self.fig.batch_update():  # 使用批量更新上下文管理器
            for i, bar in enumerate(self.traces):  # 遍历所有柱状图轨迹
                bar.y = data[:, i]  # 更新第i个轨迹的Y轴数据
                if bar.marker.colorscale is not None:  # 如果配置了颜色映射
                    bar.marker.color = data[:, i]  # 根据数据值更新颜色


class Scatter(Configured, TraceUpdater):
    """
    散点图类
    """
    
    def __init__(self,
                 data: tp.Optional[tp.ArrayLike] = None,
                 trace_names: tp.TraceNames = None,
                 x_labels: tp.Optional[tp.Labels] = None,
                 trace_kwargs: tp.KwargsLikeSequence = None,
                 add_trace_kwargs: tp.KwargsLike = None,
                 fig: tp.Optional[tp.BaseFigure] = None,
                 **layout_kwargs) -> None:
        """
        创建散点图
        
        Args:
            data: 散点图数据，可以是任何能转换为NumPy数组的格式
                 数据形状必须为(x_labels, trace_names)
                 每个数据点代表一个散点的坐标值
            trace_names: 数据系列名称，用于区分不同的散点组
                        可以是字符串或字符串列表，用于图例显示
            x_labels: X轴标签，定义散点图的X轴坐标系
                     通常表示自变量或分类标识
            trace_kwargs: 传递给plotly.Scatter的关键字参数
                         可以是单个字典或字典列表，支持每个系列独立配置
                         常用参数：mode、marker、line等
            add_trace_kwargs: 传递给add_trace方法的参数，用于图层管理
            fig: 可选的现有图形对象，支持在现有图表上添加散点
            **layout_kwargs: 布局参数，用于自定义图表外观
                            常用参数：title、xaxis_title、yaxis_title等
        
        配置选项详解：
            - mode参数控制显示模式：'markers'(仅标记)、'lines'(仅线条)、'markers+lines'(标记+线条)
            - marker参数控制标记样式：size、color、symbol、opacity等
            - line参数控制线条样式：width、color、dash等
            - 支持每个系列独立的视觉配置
        """
        Configured.__init__(  # 初始化配置管理功能
            self,
            data=data,
            trace_names=trace_names,
            x_labels=x_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs
        )

        if trace_kwargs is None:  # 如果没有提供轨迹配置参数
            trace_kwargs = {}  # 使用空字典作为默认值
        if add_trace_kwargs is None:  # 如果没有提供添加轨迹参数
            add_trace_kwargs = {}  # 使用空字典作为默认值
        if data is not None:  # 如果提供了数据
            data = reshape_fns.to_2d_array(data)  # 将数据转换为2D数组格式
            if trace_names is not None:  # 如果同时提供了系列名称
                checks.assert_shape_equal(data, trace_names, (1, 0))  # 验证数据形状与系列名称数量是否匹配
        else:  # 如果没有提供数据
            if trace_names is None:  # 但也没有提供系列名称
                raise ValueError("At least data or trace_names must be passed")  # 抛出错误，至少需要提供其中一个
        if trace_names is None:  # 如果没有提供系列名称
            trace_names = [None] * data.shape[1]  # 创建与数据列数相等的None列表
        if isinstance(trace_names, str):  # 如果系列名称是单个字符串
            trace_names = [trace_names]  # 将其转换为列表格式
        if x_labels is not None:  # 如果提供了X轴标签
            x_labels = clean_labels(x_labels)  # 清理标签格式，处理MultiIndex等复杂情况

        if fig is None:  # 如果没有提供现有图形对象
            fig = make_figure()  # 创建新的图形对象
        fig.update_layout(**layout_kwargs)  # 应用用户提供的布局配置

        for i, trace_name in enumerate(trace_names):  # 遍历每个数据系列
            _trace_kwargs = resolve_dict(trace_kwargs, i=i)  # 解析当前系列的配置参数
            trace_name = _trace_kwargs.pop('name', trace_name)  # 从配置中提取名称，如果没有则使用默认名称
            if trace_name is not None:  # 如果有有效的系列名称
                trace_name = str(trace_name)  # 确保名称为字符串格式
            scatter = go.Scatter(  # 创建plotly散点图对象
                x=x_labels,  # 设置X轴数据（标签或坐标）
                name=trace_name,  # 设置系列名称
                showlegend=trace_name is not None  # 根据是否有名称决定是否显示图例
            )
            scatter.update(**_trace_kwargs)  # 应用当前系列的自定义配置
            fig.add_trace(scatter, **add_trace_kwargs)  # 将散点图添加到图形中

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names):])  # 初始化轨迹更新器，管理新添加的轨迹

        if data is not None:  # 如果提供了初始数据
            self.update(data)  # 立即更新图表显示

    def update(self, data: tp.ArrayLike) -> None:
        """
        更新散点图数据
        
        这是散点图的核心数据更新方法，支持动态刷新图表内容。
        更新过程会保持所有样式配置，只改变数据点的Y轴坐标值。
        """
        data = reshape_fns.to_2d_array(data)  # 将输入数据转换为标准的2D数组格式

        with self.fig.batch_update():  # 使用批量更新上下文管理器
            for i, trace in enumerate(self.traces):  # 遍历所有散点图轨迹
                trace.y = data[:, i]  # 更新第i个轨迹的Y轴数据，X轴保持不变


class Histogram(Configured, TraceUpdater):
    """
    直方图类
    """
    
    def __init__(self,
                 data: tp.Optional[tp.ArrayLike] = None,
                 trace_names: tp.TraceNames = None,
                 horizontal: bool = False,
                 remove_nan: bool = True,
                 from_quantile: tp.Optional[float] = None,
                 to_quantile: tp.Optional[float] = None,
                 trace_kwargs: tp.KwargsLikeSequence = None,
                 add_trace_kwargs: tp.KwargsLike = None,
                 fig: tp.Optional[tp.BaseFigure] = None,
                 **layout_kwargs) -> None:
        """
        创建直方图
        
        构建功能完整的直方图表，支持多系列数据分布展示和高级数据过滤功能。
        该方法会自动处理数据预处理、分箱计算和图表配置。
        
        Args:
            data: 直方图数据，可以是任何能转换为NumPy数组的格式
                 数据形状为(任意长度, trace_names)
                 每列代表一个数据系列的观测值
            trace_names: 数据系列名称，用于区分不同的分布
                        可以是字符串或字符串列表，用于图例显示
            horizontal: 是否水平显示直方图
                       True: 柱子水平排列，适合类别名称较长的情况
                       False: 柱子垂直排列，传统直方图显示方式
            remove_nan: 是否移除NaN值
                       True: 自动过滤掉无效数据点
                       False: 保留所有数据，可能影响统计准确性
            from_quantile: 数据裁剪的起始分位数
                          范围[0, 1]，用于过滤极端低值
                          例如0.05表示过滤掉最低5%的数据
            to_quantile: 数据裁剪的结束分位数
                        范围[0, 1]，用于过滤极端高值
                        例如0.95表示过滤掉最高5%的数据
            trace_kwargs: 传递给plotly.Histogram的关键字参数
                         可以是单个字典或字典列表，支持每个系列独立配置
                         常用参数：nbinsx、opacity、marker等
            add_trace_kwargs: 传递给add_trace方法的参数，用于图层管理
            fig: 可选的现有图形对象，支持在现有图表上添加直方图
            **layout_kwargs: 布局参数，用于自定义图表外观
                            常用参数：title、xaxis_title、yaxis_title等
        
        数据处理流程：
            1. 数据格式验证和2D数组转换
            2. NaN值处理和数据清理
            3. 基于分位数的数据裁剪
            4. 分箱参数计算和优化
            5. 图形对象创建和配置
            6. 多系列叠加显示设置
        
        配置选项详解：
            - barmode='overlay': 设置多系列直方图叠加显示
            - opacity: 自动调整透明度，多系列时为0.75，单系列时为1.0
            - 分位数裁剪有助于排除异常值，提高分布分析的准确性
            - 水平/垂直模式适应不同的数据展示需求
        """
        Configured.__init__(  # 初始化配置管理功能
            self,
            data=data,
            trace_names=trace_names,
            horizontal=horizontal,
            remove_nan=remove_nan,
            from_quantile=from_quantile,
            to_quantile=to_quantile,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs
        )

        if trace_kwargs is None:  # 如果没有提供轨迹配置参数
            trace_kwargs = {}  # 使用空字典作为默认值
        if add_trace_kwargs is None:  # 如果没有提供添加轨迹参数
            add_trace_kwargs = {}  # 使用空字典作为默认值
        if data is not None:  # 如果提供了数据
            data = reshape_fns.to_2d_array(data)  # 将数据转换为2D数组格式
            if trace_names is not None:  # 如果同时提供了系列名称
                checks.assert_shape_equal(data, trace_names, (1, 0))  # 验证数据形状与系列名称数量是否匹配
        else:  # 如果没有提供数据
            if trace_names is None:  # 但也没有提供系列名称
                raise ValueError("At least data or trace_names must be passed")  # 抛出错误，至少需要提供其中一个
        if trace_names is None:  # 如果没有提供系列名称
            trace_names = [None] * data.shape[1]  # 创建与数据列数相等的None列表
        if isinstance(trace_names, str):  # 如果系列名称是单个字符串
            trace_names = [trace_names]  # 将其转换为列表格式

        if fig is None:  # 如果没有提供现有图形对象
            fig = make_figure()  # 创建新的图形对象
            fig.update_layout(barmode='overlay')  # 设置柱状图叠加模式，允许多个直方图重叠显示
        fig.update_layout(**layout_kwargs)  # 应用用户提供的布局配置

        for i, trace_name in enumerate(trace_names):  # 遍历每个数据系列
            _trace_kwargs = resolve_dict(trace_kwargs, i=i)  # 解析当前系列的配置参数
            trace_name = _trace_kwargs.pop('name', trace_name)  # 从配置中提取名称，如果没有则使用默认名称
            if trace_name is not None:  # 如果有有效的系列名称
                trace_name = str(trace_name)  # 确保名称为字符串格式
            hist = go.Histogram(  # 创建plotly直方图对象
                opacity=0.75 if len(trace_names) > 1 else 1,  # 多系列时使用半透明，单系列时完全不透明
                name=trace_name,  # 设置系列名称
                showlegend=trace_name is not None  # 根据是否有名称决定是否显示图例
            )
            hist.update(**_trace_kwargs)  # 应用当前系列的自定义配置
            fig.add_trace(hist, **add_trace_kwargs)  # 将直方图添加到图形中

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names):])  # 初始化轨迹更新器，管理新添加的轨迹
        self._horizontal = horizontal  # 保存水平显示设置
        self._remove_nan = remove_nan  # 保存NaN值移除设置
        self._from_quantile = from_quantile  # 保存起始分位数设置
        self._to_quantile = to_quantile  # 保存结束分位数设置

        if data is not None:  # 如果提供了初始数据
            self.update(data)  # 立即更新图表显示

    @property
    def horizontal(self):
        return self._horizontal  # 返回水平显示设置

    @property
    def remove_nan(self):
        return self._remove_nan  # 返回NaN值移除设置

    @property
    def from_quantile(self):
        return self._from_quantile  # 返回起始分位数设置

    @property
    def to_quantile(self):
        return self._to_quantile  # 返回结束分位数设置

    def update(self, data: tp.ArrayLike) -> None:
        """
        更新直方图数据
        
        这是直方图的核心数据更新方法，支持动态刷新分布展示。
        更新过程包括数据预处理、分位数裁剪、NaN值过滤和图表刷新。
        
        Args:
            data: 新的直方图数据，必须与初始化时的数据格式一致
                 形状应为(任意长度, trace_names)
                 每列代表一个数据系列的观测值
        
        更新机制：
            1. 数据格式验证和2D数组转换
            2. 逐系列应用数据预处理规则
            3. NaN值过滤（如果启用）
            4. 基于分位数的数据裁剪（如果配置）
            5. 根据显示方向设置X/Y轴数据
            6. 批量更新所有轨迹
        
        数据预处理详解：
            - remove_nan=True: 自动过滤无效数值
            - from_quantile: 过滤低于指定分位数的值
            - to_quantile: 过滤高于指定分位数的值
            - 分位数裁剪有助于专注于数据的主要分布区间
        
        性能优化：
            使用batch_update上下文管理器确保所有系列在一个批次中更新，
            避免多次重绘，提供流畅的更新体验。
        """
        data = reshape_fns.to_2d_array(data)  # 将输入数据转换为标准的2D数组格式

        with self.fig.batch_update():  # 使用批量更新上下文管理器
            for i, trace in enumerate(self.traces):  # 遍历所有直方图轨迹
                d = data[:, i]  # 获取第i个系列的数据
                if self.remove_nan:  # 如果启用了NaN值移除
                    d = d[~np.isnan(d)]  # 过滤掉所有NaN值
                mask = np.full(d.shape, True)  # 创建全为True的掩码数组
                if self.from_quantile is not None:  # 如果设置了起始分位数
                    mask &= d >= np.quantile(d, self.from_quantile)  # 应用下界过滤
                if self.to_quantile is not None:  # 如果设置了结束分位数
                    mask &= d <= np.quantile(d, self.to_quantile)  # 应用上界过滤
                d = d[mask]  # 应用掩码，获取过滤后的数据
                if self.horizontal:  # 如果是水平显示模式
                    trace.x = None  # 清空X轴数据
                    trace.y = d  # 将数据设置为Y轴
                else:  # 如果是垂直显示模式（默认）
                    trace.x = d  # 将数据设置为X轴
                    trace.y = None  # 清空Y轴数据


class Box(Configured, TraceUpdater):
    """
    箱线图类
    """
    
    def __init__(self,
                 data: tp.Optional[tp.ArrayLike] = None,
                 trace_names: tp.TraceNames = None,
                 horizontal: bool = False,
                 remove_nan: bool = True,
                 from_quantile: tp.Optional[float] = None,
                 to_quantile: tp.Optional[float] = None,
                 trace_kwargs: tp.KwargsLikeSequence = None,
                 add_trace_kwargs: tp.KwargsLike = None,
                 fig: tp.Optional[tp.BaseFigure] = None,
                 **layout_kwargs) -> None:
        """
        创建箱线图
        
        构建功能完整的箱线图表，支持多系列数据统计分布展示和高级数据预处理功能。
        该方法会自动计算各种统计指标并进行可视化配置。
        
        Args:
            data: 箱线图数据，可以是任何能转换为NumPy数组的格式
                 数据形状为(任意长度, trace_names)
                 每列代表一个数据系列的所有观测值
            trace_names: 数据系列名称，用于区分不同的数据组
                        可以是字符串或字符串列表，用于图例和X轴标签显示
            horizontal: 是否水平显示箱线图
                       True: 箱子水平排列，适合系列名称较长的情况
                       False: 箱子垂直排列，传统箱线图显示方式
            remove_nan: 是否移除NaN值
                       True: 自动过滤掉无效数据点，提高统计准确性
                       False: 保留所有数据，可能影响统计计算
            from_quantile: 数据裁剪的起始分位数
                          范围[0, 1]，用于过滤极端低值
                          例如0.05表示过滤掉最低5%的数据
            to_quantile: 数据裁剪的结束分位数
                        范围[0, 1]，用于过滤极端高值
                        例如0.95表示过滤掉最高5%的数据
            trace_kwargs: 传递给plotly.Box的关键字参数
                         可以是单个字典或字典列表，支持每个系列独立配置
                         常用参数：boxpoints、jitter、fillcolor等
            add_trace_kwargs: 传递给add_trace方法的参数，用于图层管理
            fig: 可选的现有图形对象，支持在现有图表上添加箱线图
            **layout_kwargs: 布局参数，用于自定义图表外观
                            常用参数：title、xaxis_title、yaxis_title等
        
        数据处理流程：
            1. 数据格式验证和2D数组转换
            2. NaN值处理和数据清理
            3. 基于分位数的数据裁剪
            4. 统计指标计算（中位数、四分位数等）
            5. 异常值识别和标记
            6. 图形对象创建和配置
        
        统计计算说明：
            箱线图自动计算以下统计指标：
            - Q1（第一四分位数）：25%分位数
            - Q2（中位数）：50%分位数
            - Q3（第三四分位数）：75%分位数
            - IQR（四分位距）：Q3 - Q1
            - 异常值：超出Q1-1.5*IQR或Q3+1.5*IQR的值
        """
        Configured.__init__(  # 初始化配置管理功能
            self,
            data=data,
            trace_names=trace_names,
            horizontal=horizontal,
            remove_nan=remove_nan,
            from_quantile=from_quantile,
            to_quantile=to_quantile,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs
        )

        if trace_kwargs is None:  # 如果没有提供轨迹配置参数
            trace_kwargs = {}  # 使用空字典作为默认值
        if add_trace_kwargs is None:  # 如果没有提供添加轨迹参数
            add_trace_kwargs = {}  # 使用空字典作为默认值
        if data is not None:  # 如果提供了数据
            data = reshape_fns.to_2d_array(data)  # 将数据转换为2D数组格式
            if trace_names is not None:  # 如果同时提供了系列名称
                checks.assert_shape_equal(data, trace_names, (1, 0))  # 验证数据形状与系列名称数量是否匹配
        else:  # 如果没有提供数据
            if trace_names is None:  # 但也没有提供系列名称
                raise ValueError("At least data or trace_names must be passed")  # 抛出错误，至少需要提供其中一个
        if trace_names is None:  # 如果没有提供系列名称
            trace_names = [None] * data.shape[1]  # 创建与数据列数相等的None列表
        if isinstance(trace_names, str):  # 如果系列名称是单个字符串
            trace_names = [trace_names]  # 将其转换为列表格式

        if fig is None:  # 如果没有提供现有图形对象
            fig = make_figure()  # 创建新的图形对象
        fig.update_layout(**layout_kwargs)  # 应用用户提供的布局配置

        for i, trace_name in enumerate(trace_names):  # 遍历每个数据系列
            _trace_kwargs = resolve_dict(trace_kwargs, i=i)  # 解析当前系列的配置参数
            trace_name = _trace_kwargs.pop('name', trace_name)  # 从配置中提取名称，如果没有则使用默认名称
            if trace_name is not None:  # 如果有有效的系列名称
                trace_name = str(trace_name)  # 确保名称为字符串格式
            box = go.Box(  # 创建plotly箱线图对象
                name=trace_name,  # 设置系列名称
                showlegend=trace_name is not None  # 根据是否有名称决定是否显示图例
            )
            box.update(**_trace_kwargs)  # 应用当前系列的自定义配置
            fig.add_trace(box, **add_trace_kwargs)  # 将箱线图添加到图形中

        TraceUpdater.__init__(self, fig, fig.data[-len(trace_names):])  # 初始化轨迹更新器，管理新添加的轨迹
        self._horizontal = horizontal  # 保存水平显示设置
        self._remove_nan = remove_nan  # 保存NaN值移除设置
        self._from_quantile = from_quantile  # 保存起始分位数设置
        self._to_quantile = to_quantile  # 保存结束分位数设置

        if data is not None:  # 如果提供了初始数据
            self.update(data)  # 立即更新图表显示

    @property
    def horizontal(self):
        return self._horizontal  # 返回水平显示设置

    @property
    def remove_nan(self):
        """
        获取NaN值移除设置
        
        Returns:
            bool: 是否移除NaN值
        
        用途说明：
            移除NaN值有助于提高统计分析的准确性，确保
            分位数计算和异常值检测的可靠性。
        """
        return self._remove_nan  # 返回NaN值移除设置

    @property
    def from_quantile(self):
        """
        获取起始分位数设置
        
        Returns:
            float or None: 数据裁剪的起始分位数
        
        用途说明：
            起始分位数用于过滤极端低值，有助于专注于
            数据的主要分布区间，提高箱线图的可读性。
        """
        return self._from_quantile  # 返回起始分位数设置

    @property
    def to_quantile(self):
        """
        获取结束分位数设置
        
        Returns:
            float or None: 数据裁剪的结束分位数
        
        用途说明：
            结束分位数用于过滤极端高值，与起始分位数配合
            使用可以有效控制箱线图的显示范围。
        """
        return self._to_quantile  # 返回结束分位数设置

    def update(self, data: tp.ArrayLike) -> None:
        """
        更新箱线图数据
        
        这是箱线图的核心数据更新方法，支持动态刷新统计分布展示。
        更新过程包括数据预处理、统计指标重新计算和图表刷新。
        
        Args:
            data: 新的箱线图数据，必须与初始化时的数据格式一致
                 形状应为(任意长度, trace_names)
                 每列代表一个数据系列的所有观测值
        
        更新机制：
            1. 数据格式验证和2D数组转换
            2. 逐系列应用数据预处理规则
            3. NaN值过滤（如果启用）
            4. 基于分位数的数据裁剪（如果配置）
            5. 重新计算所有统计指标
            6. 根据显示方向设置数据轴
            7. 批量更新所有轨迹
        
        统计更新详解：
            每次更新都会重新计算：
            - 中位数和四分位数
            - 异常值识别和标记
            - 箱体和须的位置
            - 数据点的分布展示
        
        数据预处理详解：
            - remove_nan=True: 自动过滤无效数值
            - from_quantile: 过滤低于指定分位数的值
            - to_quantile: 过滤高于指定分位数的值
            - 裁剪后的数据用于统计计算，保证结果的稳健性
        """
        data = reshape_fns.to_2d_array(data)  # 将输入数据转换为标准的2D数组格式

        with self.fig.batch_update():  # 使用批量更新上下文管理器
            for i, trace in enumerate(self.traces):  # 遍历所有箱线图轨迹
                d = data[:, i]  # 获取第i个系列的数据
                if self.remove_nan:  # 如果启用了NaN值移除
                    d = d[~np.isnan(d)]  # 过滤掉所有NaN值
                mask = np.full(d.shape, True)  # 创建全为True的掩码数组
                if self.from_quantile is not None:  # 如果设置了起始分位数
                    mask &= d >= np.quantile(d, self.from_quantile)  # 应用下界过滤
                if self.to_quantile is not None:  # 如果设置了结束分位数
                    mask &= d <= np.quantile(d, self.to_quantile)  # 应用上界过滤
                d = d[mask]  # 应用掩码，获取过滤后的数据
                if self.horizontal:  # 如果是水平显示模式
                    trace.x = d  # 将数据设置为X轴，箱子水平排列
                    trace.y = None  # 清空Y轴数据
                else:  # 如果是垂直显示模式（默认）
                    trace.x = None  # 清空X轴数据
                    trace.y = d  # 将数据设置为Y轴，箱子垂直排列


class Heatmap(Configured, TraceUpdater):
    """
    热力图类
    """
    
    def __init__(self,
                 data: tp.Optional[tp.ArrayLike] = None,
                 x_labels: tp.Optional[tp.Labels] = None,
                 y_labels: tp.Optional[tp.Labels] = None,
                 is_x_category: bool = False,
                 is_y_category: bool = False,
                 trace_kwargs: tp.KwargsLike = None,
                 add_trace_kwargs: tp.KwargsLike = None,
                 fig: tp.Optional[tp.BaseFigure] = None,
                 **layout_kwargs) -> None:
        """
        创建热力图
        
        构建功能完整的热力图表，支持二维数据的颜色映射展示和灵活的轴配置。
        该方法会自动处理数据格式转换、轴标签设置和颜色方案配置。
        
        Args:
            data: 热力图数据，可以是任何能转换为NumPy数组的格式
                 数据形状必须为(y_labels, x_labels)，即行对应Y轴，列对应X轴
                 数据值将通过颜色深浅进行视觉编码
            x_labels: X轴标签，对应pandas DataFrame的列名
                     用于标识热力图的列（水平方向）
                     可以是字符串列表或数组
            y_labels: Y轴标签，对应pandas DataFrame的索引
                     用于标识热力图的行（垂直方向）
                     可以是字符串列表或数组
            is_x_category: X轴是否为分类轴
                          True: 将X轴设置为分类轴，适合离散标签
                          False: 将X轴设置为数值轴，适合连续数据
            is_y_category: Y轴是否为分类轴
                          True: 将Y轴设置为分类轴，适合离散标签
                          False: 将Y轴设置为数值轴，适合连续数据
            trace_kwargs: 传递给plotly.Heatmap的关键字参数
                         用于自定义热力图的外观和行为
                         常用参数：colorscale、zmin、zmax、hoverongaps等
            add_trace_kwargs: 传递给add_trace方法的参数，用于图层管理
            fig: 可选的现有图形对象，支持在现有图表上添加热力图
            **layout_kwargs: 布局参数，用于自定义图表外观
                            常用参数：title、width、height等
        
        数据处理流程：
            1. 数据格式验证和2D数组转换
            2. 轴标签清理和格式化处理
            3. 图表尺寸智能计算和优化
            4. 颜色方案配置和映射设置
            5. 轴类型设置（分类/数值）
            6. 热力图对象创建和配置
        
        尺寸计算逻辑：
            系统会根据数据维度自动计算最优的图表尺寸：
            - 考虑X轴和Y轴的标签数量比例
            - 在预设的尺寸范围内进行调整
            - 为颜色条预留150像素的额外空间
            - 限制最大高度以保持良好的显示效果
        
        轴类型配置：
            - 分类轴：适合离散的类别标签，如资产名称、时间段等
            - 数值轴：适合连续的数值标签，如价格、收益率等
            - 混合使用：X轴和Y轴可以独立设置为不同类型
        
        颜色方案配置：
            - 默认使用'Plasma'颜色方案，适合大部分数据类型
            - 支持所有matplotlib和plotly内置颜色方案
            - 自动设置hoverongaps=False，避免缺失数据的悬停问题
        """
        Configured.__init__(  # 初始化配置管理功能
            self,
            data=data,
            x_labels=x_labels,
            y_labels=y_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs
        )

        from vectorbt._settings import settings  # 导入vectorbt全局设置
        layout_cfg = settings['plotting']['layout']  # 获取布局配置

        if trace_kwargs is None:  # 如果没有提供轨迹配置参数
            trace_kwargs = {}  # 使用空字典作为默认值
        if add_trace_kwargs is None:  # 如果没有提供添加轨迹参数
            add_trace_kwargs = {}  # 使用空字典作为默认值
        if data is not None:  # 如果提供了数据
            data = reshape_fns.to_2d_array(data)  # 将数据转换为2D数组格式
            if x_labels is not None:  # 如果提供了X轴标签
                checks.assert_shape_equal(data, x_labels, (1, 0))  # 验证数据列数与X轴标签数量是否匹配
            if y_labels is not None:  # 如果提供了Y轴标签
                checks.assert_shape_equal(data, y_labels, (0, 0))  # 验证数据行数与Y轴标签数量是否匹配
        else:  # 如果没有提供数据
            if x_labels is None or y_labels is None:  # 但缺少轴标签
                raise ValueError("At least data, or x_labels and y_labels must be passed")  # 抛出错误
        if x_labels is not None:  # 如果提供了X轴标签
            x_labels = clean_labels(x_labels)  # 清理X轴标签格式
        if y_labels is not None:  # 如果提供了Y轴标签
            y_labels = clean_labels(y_labels)  # 清理Y轴标签格式

        if fig is None:  # 如果没有提供现有图形对象
            fig = make_figure()  # 创建新的图形对象
            if 'width' in layout_cfg:  # 如果配置中指定了宽度
                # 智能计算图表尺寸
                max_width = layout_cfg['width']  # 获取最大宽度
                if data is not None:  # 如果有数据
                    x_len = data.shape[1]  # X轴元素数量
                    y_len = data.shape[0]  # Y轴元素数量
                else:  # 如果没有数据但有标签
                    x_len = len(x_labels)  # X轴标签数量
                    y_len = len(y_labels)  # Y轴标签数量
                
                # 根据轴长度比例计算宽度
                width = math.ceil(renormalize(
                    x_len / (x_len + y_len),  # 计算X轴在总长度中的比例
                    (0, 1),  # 输入范围
                    (0.3 * max_width, max_width)  # 输出范围：30%到100%的最大宽度
                ))
                width = min(width + 150, max_width)  # 为颜色条预留150像素空间，但不超过最大宽度
                
                # 根据轴长度比例计算高度
                height = math.ceil(renormalize(
                    y_len / (x_len + y_len),  # 计算Y轴在总长度中的比例
                    (0, 1),  # 输入范围
                    (0.3 * max_width, max_width)  # 输出范围：30%到100%的最大宽度
                ))
                height = min(height, max_width * 0.7)  # 限制高度不超过最大宽度的70%
                
                fig.update_layout(
                    width=width,  # 设置计算得出的宽度
                    height=height  # 设置计算得出的高度
                )

        heatmap = go.Heatmap(  # 创建plotly热力图对象
            hoverongaps=False,  # 禁用缺失数据的悬停效果
            colorscale='Plasma',  # 设置默认颜色方案为Plasma
            x=x_labels,  # 设置X轴标签
            y=y_labels  # 设置Y轴标签
        )
        heatmap.update(**trace_kwargs)  # 应用用户自定义的轨迹配置
        fig.add_trace(heatmap, **add_trace_kwargs)  # 将热力图添加到图形中

        # 配置轴类型（分类或数值）
        axis_kwargs = dict()  # 创建轴配置字典
        if is_x_category:  # 如果X轴是分类轴
            if fig.data[-1]['xaxis'] is not None:  # 如果热力图有指定的X轴
                axis_kwargs['xaxis' + fig.data[-1]['xaxis'][1:]] = dict(type='category')  # 设置对应X轴为分类型
            else:  # 如果使用默认X轴
                axis_kwargs['xaxis'] = dict(type='category')  # 设置主X轴为分类型
        if is_y_category:  # 如果Y轴是分类轴
            if fig.data[-1]['yaxis'] is not None:  # 如果热力图有指定的Y轴
                axis_kwargs['yaxis' + fig.data[-1]['yaxis'][1:]] = dict(type='category')  # 设置对应Y轴为分类型
            else:  # 如果使用默认Y轴
                axis_kwargs['yaxis'] = dict(type='category')  # 设置主Y轴为分类型
        fig.update_layout(**axis_kwargs)  # 应用轴类型配置
        fig.update_layout(**layout_kwargs)  # 应用用户自定义的布局配置

        TraceUpdater.__init__(self, fig, (fig.data[-1],))  # 初始化轨迹更新器，管理新添加的热力图轨迹

        if data is not None:  # 如果提供了初始数据
            self.update(data)  # 立即更新图表显示

    def update(self, data: tp.ArrayLike) -> None:
        """
        更新热力图数据
        """
        data = reshape_fns.to_2d_array(data)  # 将输入数据转换为标准的2D数组格式

        with self.fig.batch_update():  # 使用批量更新上下文管理器
            self.traces[0].z = data  # 更新热力图的Z值数据，即颜色映射的数据源


class Volume(Configured, TraceUpdater):
    """
    3D体积图类
    """
    
    def __init__(self,
                 data: tp.Optional[tp.ArrayLike] = None,
                 x_labels: tp.Optional[tp.Labels] = None,
                 y_labels: tp.Optional[tp.Labels] = None,
                 z_labels: tp.Optional[tp.Labels] = None,
                 trace_kwargs: tp.KwargsLike = None,
                 add_trace_kwargs: tp.KwargsLike = None,
                 scene_name: str = 'scene',
                 fig: tp.Optional[tp.BaseFigure] = None,
                 **layout_kwargs) -> None:
        """
        创建3D体积图
        
        构建功能完整的三维体积图表，支持复杂多维数据的立体可视化展示。
        该方法会自动处理3D数据格式转换、轴标签设置和场景配置。
        
        Args:
            data: 3D体积图数据，必须是3维数组
                 数据形状为(x_len, y_len, z_len)
                 数据值将通过颜色和透明度进行三维编码
            x_labels: X轴标签，定义3D空间的X坐标系
                     可以是数值或分类标签的数组
            y_labels: Y轴标签，定义3D空间的Y坐标系
                     可以是数值或分类标签的数组
            z_labels: Z轴标签，定义3D空间的Z坐标系
                     可以是数值或分类标签的数组
            trace_kwargs: 传递给plotly.Volume的关键字参数
                         用于自定义3D体积图的外观和行为
                         常用参数：opacity、surface_count、colorscale等
            add_trace_kwargs: 传递给add_trace方法的参数，用于图层管理
            scene_name: 3D场景的引用名称，用于多场景管理
                       默认为'scene'，可设置为'scene2'、'scene3'等
            fig: 可选的现有图形对象，支持在现有3D图表上添加体积图
            **layout_kwargs: 布局参数，用于自定义图表外观
                            常用参数：title、scene配置等
        """
        Configured.__init__(  # 初始化配置管理功能
            self,
            data=data,
            x_labels=x_labels,
            y_labels=y_labels,
            z_labels=z_labels,
            trace_kwargs=trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            scene_name=scene_name,
            fig=fig,
            **layout_kwargs
        )

        from vectorbt._settings import settings  # 导入vectorbt全局设置
        layout_cfg = settings['plotting']['layout']  # 获取布局配置

        if trace_kwargs is None:  # 如果没有提供轨迹配置参数
            trace_kwargs = {}  # 使用空字典作为默认值
        if add_trace_kwargs is None:  # 如果没有提供添加轨迹参数
            add_trace_kwargs = {}  # 使用空字典作为默认值
        if data is not None:  # 如果提供了数据
            checks.assert_ndim(data, 3)  # 验证数据是3维数组
            data = np.asarray(data)  # 转换为NumPy数组
            x_len, y_len, z_len = data.shape  # 获取三个维度的长度
            if x_labels is not None:  # 如果提供了X轴标签
                checks.assert_shape_equal(data, x_labels, (0, 0))  # 验证X轴标签数量与数据X维度匹配
            if y_labels is not None:  # 如果提供了Y轴标签
                checks.assert_shape_equal(data, y_labels, (1, 0))  # 验证Y轴标签数量与数据Y维度匹配
            if z_labels is not None:  # 如果提供了Z轴标签
                checks.assert_shape_equal(data, z_labels, (2, 0))  # 验证Z轴标签数量与数据Z维度匹配
        else:  # 如果没有提供数据
            if x_labels is None or y_labels is None or z_labels is None:  # 但缺少轴标签
                raise ValueError("At least data, or x_labels, y_labels and z_labels must be passed")  # 抛出错误
            x_len = len(x_labels)  # 从标签获取X维度长度
            y_len = len(y_labels)  # 从标签获取Y维度长度
            z_len = len(z_labels)  # 从标签获取Z维度长度
        
        # 处理轴标签，如果没有提供则使用默认数值
        if x_labels is None:  # 如果没有X轴标签
            x_labels = np.arange(x_len)  # 使用0到x_len-1的数值序列
        else:  # 如果有X轴标签
            x_labels = clean_labels(x_labels)  # 清理标签格式
        if y_labels is None:  # 如果没有Y轴标签
            y_labels = np.arange(y_len)  # 使用0到y_len-1的数值序列
        else:  # 如果有Y轴标签
            y_labels = clean_labels(y_labels)  # 清理标签格式
        if z_labels is None:  # 如果没有Z轴标签
            z_labels = np.arange(z_len)  # 使用0到z_len-1的数值序列
        else:  # 如果有Z轴标签
            z_labels = clean_labels(z_labels)  # 清理标签格式
        
        # 转换标签为NumPy数组，便于后续处理
        x_labels = np.asarray(x_labels)
        y_labels = np.asarray(y_labels)
        z_labels = np.asarray(z_labels)

        if fig is None:  # 如果没有提供现有图形对象
            fig = make_figure()  # 创建新的图形对象
            if 'width' in layout_cfg:  # 如果配置中指定了宽度
                # 为3D图表计算合适的尺寸
                fig.update_layout(
                    width=layout_cfg['width'],  # 设置宽度
                    height=0.7 * layout_cfg['width']  # 设置高度为宽度的70%，保持良好的3D视觉比例
                )

        # 处理非数值数据类型的轴标签
        # plotly.Volume不支持非数值标签，需要转换并配置刻度显示
        more_layout = dict()  # 创建额外布局配置字典
        if not np.issubdtype(x_labels.dtype, np.number):  # 如果X轴标签不是数值类型
            x_ticktext = x_labels  # 保存原始文本标签
            x_labels = np.arange(x_len)  # 转换为数值索引
            more_layout[scene_name] = dict(  # 配置3D场景的X轴
                xaxis=dict(
                    ticktext=x_ticktext,  # 设置显示的文本
                    tickvals=x_labels,    # 设置对应的数值
                    tickmode='array'      # 使用数组模式显示刻度
                )
            )
        if not np.issubdtype(y_labels.dtype, np.number):  # 如果Y轴标签不是数值类型
            y_ticktext = y_labels  # 保存原始文本标签
            y_labels = np.arange(y_len)  # 转换为数值索引
            more_layout[scene_name] = dict(  # 配置3D场景的Y轴
                yaxis=dict(
                    ticktext=y_ticktext,  # 设置显示的文本
                    tickvals=y_labels,    # 设置对应的数值
                    tickmode='array'      # 使用数组模式显示刻度
                )
            )
        if not np.issubdtype(z_labels.dtype, np.number):  # 如果Z轴标签不是数值类型
            z_ticktext = z_labels  # 保存原始文本标签
            z_labels = np.arange(z_len)  # 转换为数值索引
            more_layout[scene_name] = dict(  # 配置3D场景的Z轴
                zaxis=dict(
                    ticktext=z_ticktext,  # 设置显示的文本
                    tickvals=z_labels,    # 设置对应的数值
                    tickmode='array'      # 使用数组模式显示刻度
                )
            )
        fig.update_layout(**more_layout)  # 应用轴标签配置
        fig.update_layout(**layout_kwargs)  # 应用用户自定义的布局配置

        # 创建3D坐标网格
        # 数组长度必须与展平的数据数组长度相同
        x = np.repeat(x_labels, len(y_labels) * len(z_labels))  # X坐标：每个X值重复y_len*z_len次
        y = np.tile(np.repeat(y_labels, len(z_labels)), len(x_labels))  # Y坐标：Y值重复z_len次，然后整体重复x_len次
        z = np.tile(z_labels, len(x_labels) * len(y_labels))  # Z坐标：Z值重复x_len*y_len次

        volume = go.Volume(  # 创建plotly 3D体积图对象
            x=x,  # 设置X坐标数组
            y=y,  # 设置Y坐标数组
            z=z,  # 设置Z坐标数组
            opacity=0.2,  # 设置透明度为0.2，提供良好的3D透视效果
            surface_count=15,  # 设置表面数量为15，在性能和视觉效果间平衡
            colorscale='Plasma'  # 设置颜色方案为Plasma，提供清晰的颜色层次
        )
        volume.update(**trace_kwargs)  # 应用用户自定义的轨迹配置
        fig.add_trace(volume, **add_trace_kwargs)  # 将体积图添加到图形中

        TraceUpdater.__init__(self, fig, (fig.data[-1],))  # 初始化轨迹更新器，管理新添加的体积图轨迹

        if data is not None:  # 如果提供了初始数据
            self.update(data)  # 立即更新图表显示

    def update(self, data: tp.ArrayLike) -> None:
        """
        更新3D体积图数据
        """
        data = np.asarray(data).flatten()  # 将输入数据转换为NumPy数组并展平为一维

        with self.fig.batch_update():  # 使用批量更新上下文管理器
            self.traces[0].value = data  # 更新体积图的值数据，即3D密度分布的数据源
