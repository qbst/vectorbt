# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
技术指标基础模块 - 基于IndicatorFactory构建的技术分析指标
================================================================================

文件设计逻辑：
本文件实现了量化交易中常用的技术指标，通过vectorbt的IndicatorFactory工厂模式统一构建。
主要包含8个核心技术指标：MA（移动平均线）、MSTD（移动标准差）、BBANDS（布林带）、
RSI（相对强弱指数）、STOCH（随机振荡器）、MACD（平滑异同移动平均线）、
ATR（真实波幅）、OBV（能量潮）。

核心特性：
1. 高效的矢量化计算，支持2维数组批量处理
2. 内置缓存机制，提高大矩阵计算效率
3. 统一的可视化接口，提供专业图表绘制功能
4. 相比TA-Lib，针对多列数据优化性能

使用场景：
- 股票、期货、数字货币等金融数据的技术分析
- 量化交易策略开发中的信号生成
- 风险管理和市场趋势分析
- 高频交易中的实时指标计算

架构设计：
每个指标通过IndicatorFactory创建基础类，然后通过私有类扩展绘图功能，
最终将扩展功能注入到基础类中，实现功能的模块化和可扩展性。
================================================================================

使用vectorbt.indicators.factory.IndicatorFactory构建的技术指标。

您可以通过vbt.*或vbt.indicators.*访问所有指标。

示例代码：
```pycon
>>> import pandas as pd
>>> import vectorbt as vbt

>>> # 创建移动平均线指标
>>> vbt.MA.run(pd.Series([1, 2, 3]), [2, 3]).ma
ma_window     2     3
ma_ewm    False False
0           NaN   NaN
1           1.5   NaN
2           2.5   2.0
```

相比TA-Lib的优势：
这些指标主要在2维数组上工作并利用缓存机制，使得它们在处理大量列的矩阵时更加高效。
它们还具有绘图方法，便于数据可视化分析。

运行以下示例：

```pycon
>>> import vectorbt as vbt
>>> from datetime import datetime

>>> start = '2019-03-01 UTC'  # 加密货币使用UTC时间
>>> end = '2019-09-01 UTC'
>>> cols = ['Open', 'High', 'Low', 'Close', 'Volume']  # OHLCV数据列名
>>> ohlcv = vbt.YFData.download("BTC-USD", start=start, end=end).get(cols)
>>> ohlcv
                                   Open          High          Low  \\
Date
2019-03-01 00:00:00+00:00   3853.757080   3907.795410  3851.692383
2019-03-02 00:00:00+00:00   3855.318115   3874.607422  3832.127930
2019-03-03 00:00:00+00:00   3862.266113   3875.483643  3836.905762
...                                 ...           ...          ...
2019-08-30 00:00:00+00:00   9514.844727   9656.124023  9428.302734
2019-08-31 00:00:00+00:00   9597.539062   9673.220703  9531.799805
2019-09-01 00:00:00+00:00   9630.592773   9796.755859  9582.944336

                                 Close       Volume
Date
2019-03-01 00:00:00+00:00  3859.583740   7661247975
2019-03-02 00:00:00+00:00  3864.415039   7578786076
2019-03-03 00:00:00+00:00  3847.175781   7253558152
...                                ...          ...
2019-08-30 00:00:00+00:00  9598.173828  13595263986
2019-08-31 00:00:00+00:00  9630.664062  11454806419
2019-09-01 00:00:00+00:00  9757.970703  11445355859

[185 rows x 5 columns]

>>> ohlcv.vbt.ohlcv.plot()
```
![](/assets/images/basic_price.svg)"""

# 导入必要的库
import numpy as np  # 导入NumPy库，用于数值计算
import plotly.graph_objects as go  # 导入Plotly图形对象，用于创建交互式图表

# 导入vectorbt相关模块
from vectorbt import _typing as tp  # 导入类型提示模块
from vectorbt.generic import nb as generic_nb  # 导入通用NumPy函数
from vectorbt.indicators import nb  # 导入指标相关的NumPy函数
from vectorbt.indicators.factory import IndicatorFactory  # 导入指标工厂类
from vectorbt.utils.colors import adjust_opacity  # 导入颜色透明度调整函数
from vectorbt.utils.config import merge_dicts  # 导入字典合并函数
from vectorbt.utils.figure import make_figure  # 导入图形创建函数

# ############# MA（移动平均线）############# #

# 创建移动平均线指标工厂
MA = IndicatorFactory(
    class_name='MA',  # 指标类名
    module_name=__name__,  # 当前模块名
    short_name='ma',  # 指标简称
    input_names=['close'],  # 输入参数名称列表（收盘价）
    param_names=['window', 'ewm'],  # 参数名称列表（窗口期、是否指数加权）
    output_names=['ma']  # 输出参数名称列表（移动平均值）
).from_apply_func(
    nb.ma_apply_nb,  # 应用函数（NumPy实现）
    cache_func=nb.ma_cache_nb,  # 缓存函数
    kwargs_to_args=['adjust'],  # 将关键字参数转换为位置参数
    ewm=False,  # 默认不使用指数加权移动平均
    adjust=False  # 默认不调整偏差
)


class _MA(MA):
    """
    移动平均线（Moving Average, MA）技术指标类
    
    移动平均线是技术分析中广泛使用的指标，通过过滤随机短期价格波动的"噪音"
    来平滑价格走势。它是趋势跟踪指标的基础，帮助识别价格趋势方向。
    
    计算方法：
    - 简单移动平均线（SMA）：窗口期内价格的算术平均值
    - 指数加权移动平均线（EMA）：对近期价格给予更高权重
    
    应用场景：
    - 趋势识别：价格在MA上方为上升趋势，下方为下降趋势
    - 支撑阻力：MA线常作为动态支撑或阻力位
    - 交易信号：价格突破MA线可产生买卖信号
    - 多重MA组合：不同周期MA的交叉产生金叉死叉信号
    
    使用示例：
    ```python
    >>> import vectorbt as vbt
    >>> import pandas as pd
    >>> 
    >>> # 创建示例价格数据
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    >>> 
    >>> # 计算5日简单移动平均线
    >>> sma5 = vbt.MA.run(prices, window=5, ewm=False)
    >>> print(sma5.ma)
    >>> 
    >>> # 计算5日指数加权移动平均线
    >>> ema5 = vbt.MA.run(prices, window=5, ewm=True)
    >>> print(ema5.ma)
    ```
    
    参考资料：
    参见[移动平均线](https://www.investopedia.com/terms/m/movingaverage.asp)
    """

    def plot(self,
             column: tp.Optional[tp.Label] = None,  # 要绘制的列名
             plot_close: bool = True,  # 是否绘制收盘价
             close_trace_kwargs: tp.KwargsLike = None,  # 收盘价线条样式参数
             ma_trace_kwargs: tp.KwargsLike = None,  # MA线条样式参数
             add_trace_kwargs: tp.KwargsLike = None,  # 添加线条的参数
             fig: tp.Optional[tp.BaseFigure] = None,  # 现有图形对象
             **layout_kwargs) -> tp.BaseFigure:  # 布局参数
        """
        绘制移动平均线与收盘价的对比图表
        
        该方法创建一个专业的技术分析图表，显示原始价格数据与移动平均线的关系，
        帮助分析师识别趋势和潜在的交易机会。
        
        参数说明：
            column (str, optional): 要绘制的数据列名称。如果数据包含多列，需要指定列名。
                                  默认为None，表示使用所有列或单列数据。
            plot_close (bool): 是否在图表中绘制原始收盘价数据。默认为True。
                             设置为False可以只显示移动平均线。
            close_trace_kwargs (dict, optional): 收盘价线条的样式参数，传递给plotly.graph_objects.Scatter。
                                               可设置颜色、线宽、透明度等属性。
            ma_trace_kwargs (dict, optional): 移动平均线的样式参数，传递给plotly.graph_objects.Scatter。
                                            可自定义MA线的外观。
            add_trace_kwargs (dict, optional): 添加线条时的通用参数，如图例组、可见性等。
            fig (Figure or FigureWidget, optional): 现有的Plotly图形对象。如果提供，
                                                   将在此图形上添加新的线条。
            **layout_kwargs: 图表布局的关键字参数，如标题、坐标轴标签、图例位置等。
        
        返回值：
            tp.BaseFigure: 包含移动平均线和收盘价数据的Plotly图形对象。
        
        使用示例：
        ```python
        >>> import vectorbt as vbt
        >>> import pandas as pd
        >>> 
        >>> # 下载股票数据
        >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
        >>> close_prices = data.get("Close")
        >>> 
        >>> # 计算20日移动平均线
        >>> ma20 = vbt.MA.run(close_prices, window=20)
        >>> 
        >>> # 绘制图表
        >>> fig = ma20.plot(
        ...     title="苹果股票价格与20日移动平均线",
        ...     ma_trace_kwargs={"line": {"color": "red", "width": 2}},
        ...     close_trace_kwargs={"line": {"color": "blue", "width": 1}}
        ... )
        >>> fig.show()
        ```
        
        图表特点：
        - 蓝色线条表示原始收盘价数据
        - 红色线条表示移动平均线
        - 交互式图表，支持缩放和平移
        - 图例显示各线条的标识
        """
        # 导入设置配置
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']  # 获取绘图配置

        # 选择指定列的数据
        self_col = self.select_one(column=column)

        # 创建图形对象（如果未提供）
        if fig is None:
            fig = make_figure()  # 创建新的图形对象
        fig.update_layout(**layout_kwargs)  # 应用布局参数

        # 设置默认的线条样式参数
        if close_trace_kwargs is None:
            close_trace_kwargs = {}  # 初始化收盘价线条参数
        if ma_trace_kwargs is None:
            ma_trace_kwargs = {}  # 初始化MA线条参数
        
        # 合并默认样式配置
        close_trace_kwargs = merge_dicts(dict(
            name='Close',  # 线条名称
            line=dict(
                color=plotting_cfg['color_schema']['blue']  # 设置为蓝色
            )
        ), close_trace_kwargs)
        ma_trace_kwargs = merge_dicts(dict(
            name='MA'  # 移动平均线名称
        ), ma_trace_kwargs)

        # 绘制收盘价数据
        if plot_close:
            fig = self_col.close.vbt.plot(
                trace_kwargs=close_trace_kwargs,  # 线条样式参数
                add_trace_kwargs=add_trace_kwargs,  # 添加线条参数
                fig=fig)  # 目标图形对象
        
        # 绘制移动平均线
        fig = self_col.ma.vbt.plot(
            trace_kwargs=ma_trace_kwargs,  # 线条样式参数
            add_trace_kwargs=add_trace_kwargs,  # 添加线条参数
            fig=fig)  # 目标图形对象

        return fig  # 返回完整的图形对象


# 将私有类的文档字符串和方法注入到公共类中
setattr(MA, '__doc__', _MA.__doc__)  # 设置类文档字符串
setattr(MA, 'plot', _MA.plot)  # 设置plot方法

# ############# MSTD（移动标准差）############# #

# 创建移动标准差指标工厂
MSTD = IndicatorFactory(
    class_name='MSTD',  # 指标类名
    module_name=__name__,  # 当前模块名
    short_name='mstd',  # 指标简称
    input_names=['close'],  # 输入参数名称列表（收盘价）
    param_names=['window', 'ewm'],  # 参数名称列表（窗口期、是否指数加权）
    output_names=['mstd']  # 输出参数名称列表（移动标准差）
).from_apply_func(
    nb.mstd_apply_nb,  # 应用函数（NumPy实现）
    cache_func=nb.mstd_cache_nb,  # 缓存函数
    kwargs_to_args=['adjust', 'ddof'],  # 将关键字参数转换为位置参数
    ewm=False,  # 默认不使用指数加权移动平均
    adjust=False,  # 默认不调整偏差
    ddof=0  # 自由度增量，默认为0（总体标准差）
)


class _MSTD(MSTD):
    """
    移动标准差（Moving Standard Deviation, MSTD）技术指标类
    
    移动标准差是衡量资产近期价格变动幅度的指标，用于预测价格未来的波动性。
    它反映了价格相对于移动平均线的离散程度，是波动率分析的重要工具。
    
    计算原理：
    - 计算窗口期内价格数据的标准差
    - 标准差越大，表示价格波动越剧烈
    - 标准差越小，表示价格相对稳定
    
    应用场景：
    - 波动率分析：评估市场或个股的波动程度
    - 风险管理：高标准差预示着高风险
    - 交易时机：低波动后常出现高波动的突破
    - 止损设置：基于标准差设定动态止损位
    - 布林带构建：MSTD是布林带的重要组成部分
    
    使用示例：
    ```python
    >>> import vectorbt as vbt
    >>> import pandas as pd
    >>> 
    >>> # 创建示例价格数据
    >>> prices = pd.Series([100, 102, 98, 103, 97, 105, 94, 108])
    >>> 
    >>> # 计算5日移动标准差
    >>> mstd5 = vbt.MSTD.run(prices, window=5)
    >>> print(mstd5.mstd)
    >>> 
    >>> # 计算指数加权移动标准差
    >>> ewm_mstd = vbt.MSTD.run(prices, window=5, ewm=True)
    >>> print(ewm_mstd.mstd)
    ```
    
    技术要点：
    - 高MSTD值表示价格波动剧烈，市场不稳定
    - 低MSTD值表示价格相对稳定，可能酝酿变盘
    - 常与布林带结合使用，形成价格通道
    """

    def plot(self,
             column: tp.Optional[tp.Label] = None,  # 要绘制的列名
             mstd_trace_kwargs: tp.KwargsLike = None,  # MSTD线条样式参数
             add_trace_kwargs: tp.KwargsLike = None,  # 添加线条的参数
             fig: tp.Optional[tp.BaseFigure] = None,  # 现有图形对象
             **layout_kwargs) -> tp.BaseFigure:  # 布局参数
        """
        绘制移动标准差指标图表
        
        该方法创建移动标准差的时间序列图表，帮助分析师识别价格波动的变化趋势，
        评估市场的波动性状况。
        
        参数说明：
            column (str, optional): 要绘制的数据列名称。多列数据时需指定。
            mstd_trace_kwargs (dict, optional): MSTD线条的样式参数，传递给plotly.graph_objects.Scatter。
                                              可设置颜色、线宽、样式等属性。
            add_trace_kwargs (dict, optional): 添加线条时的通用参数。
            fig (Figure or FigureWidget, optional): 现有的Plotly图形对象。
            **layout_kwargs: 图表布局的关键字参数。
        
        返回值：
            tp.BaseFigure: 包含移动标准差数据的Plotly图形对象。
        
        使用示例：
        ```python
        >>> import vectorbt as vbt
        >>> 
        >>> # 下载数据并计算移动标准差
        >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
        >>> close_prices = data.get("Close")
        >>> mstd = vbt.MSTD.run(close_prices, window=20)
        >>> 
        >>> # 绘制图表
        >>> fig = mstd.plot(
        ...     title="苹果股票20日移动标准差",
        ...     mstd_trace_kwargs={"line": {"color": "orange", "width": 2}}
        ... )
        >>> fig.show()
        ```
        """
        # 选择指定列的数据
        self_col = self.select_one(column=column)

        # 创建图形对象（如果未提供）
        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # 设置默认的线条样式参数
        if mstd_trace_kwargs is None:
            mstd_trace_kwargs = {}
        mstd_trace_kwargs = merge_dicts(dict(
            name='MSTD'  # 线条名称
        ), mstd_trace_kwargs)

        # 绘制移动标准差
        fig = self_col.mstd.vbt.plot(
            trace_kwargs=mstd_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig)

        return fig


# 将私有类的文档字符串和方法注入到公共类中
setattr(MSTD, '__doc__', _MSTD.__doc__)
setattr(MSTD, 'plot', _MSTD.plot)

# ############# BBANDS（布林带）############# #

# 创建布林带指标工厂
BBANDS = IndicatorFactory(
    class_name='BBANDS',  # 指标类名
    module_name=__name__,  # 当前模块名
    short_name='bb',  # 指标简称
    input_names=['close'],  # 输入参数名称列表（收盘价）
    param_names=['window', 'ewm', 'alpha'],  # 参数名称列表（窗口期、是否指数加权、标准差倍数）
    output_names=['middle', 'upper', 'lower'],  # 输出参数名称列表（中轨、上轨、下轨）
    custom_output_props=dict(
        # 自定义输出属性：%B指标（价格在布林带中的相对位置）
        percent_b=lambda self: self.wrapper.wrap(
            (self.close.values - self.lower.values) / (self.upper.values - self.lower.values)),
        # 自定义输出属性：带宽指标（布林带宽度相对于中轨的比例）
        bandwidth=lambda self: self.wrapper.wrap(
            (self.upper.values - self.lower.values) / self.middle.values)
    )
).from_apply_func(
    nb.bb_apply_nb,  # 应用函数（NumPy实现）
    cache_func=nb.bb_cache_nb,  # 缓存函数
    kwargs_to_args=['adjust', 'ddof'],  # 将关键字参数转换为位置参数
    window=20,  # 默认窗口期为20
    ewm=False,  # 默认不使用指数加权移动平均
    alpha=2,  # 默认标准差倍数为2
    adjust=False,  # 默认不调整偏差
    ddof=0  # 自由度增量，默认为0
)


class _BBANDS(BBANDS):
    """
    布林带（Bollinger Bands, BBANDS）技术指标类
    
    布林带是一个由简单移动平均线（中轨）和两条标准差线（上轨和下轨）组成的技术分析工具。
    上下轨分别位于移动平均线的正负两个标准差位置，形成一个动态的价格通道。
    
    构成要素：
    - 中轨（Middle Band）：n日简单移动平均线
    - 上轨（Upper Band）：中轨 + (k × n日标准差)
    - 下轨（Lower Band）：中轨 - (k × n日标准差)
    
    默认参数：
    - 窗口期：20日
    - 标准差倍数：2倍
    
    核心概念：
    - %B指标：价格在布林带中的相对位置，范围0-1
    - 带宽指标：上下轨之间的距离相对于中轨的比例
    
    应用场景：
    - 趋势识别：价格突破上轨看涨，跌破下轨看跌
    - 超买超卖：价格接近上轨可能超买，接近下轨可能超卖
    - 波动性分析：带宽收窄预示低波动，扩张表示高波动
    - 支撑阻力：上下轨常作为动态支撑阻力位
    
    使用示例：
    ```python
    >>> import vectorbt as vbt
    >>> import pandas as pd
    >>> 
    >>> # 下载股票数据
    >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
    >>> close_prices = data.get("Close")
    >>> 
    >>> # 计算布林带（20日，2倍标准差）
    >>> bbands = vbt.BBANDS.run(close_prices, window=20, alpha=2)
    >>> 
    >>> # 查看布林带数据
    >>> print(bbands.upper)  # 上轨
    >>> print(bbands.middle)  # 中轨
    >>> print(bbands.lower)  # 下轨
    >>> 
    >>> # 查看自定义指标
    >>> print(bbands.percent_b)  # %B指标
    >>> print(bbands.bandwidth)  # 带宽指标
    ```
    
    参考资料：
    参见[布林带](https://www.investopedia.com/terms/b/bollingerbands.asp)
    """

    def plot(self,
             column: tp.Optional[tp.Label] = None,  # 要绘制的列名
             plot_close: bool = True,  # 是否绘制收盘价
             close_trace_kwargs: tp.KwargsLike = None,  # 收盘价线条样式参数
             middle_trace_kwargs: tp.KwargsLike = None,  # 中轨线条样式参数
             upper_trace_kwargs: tp.KwargsLike = None,  # 上轨线条样式参数
             lower_trace_kwargs: tp.KwargsLike = None,  # 下轨线条样式参数
             add_trace_kwargs: tp.KwargsLike = None,  # 添加线条的参数
             fig: tp.Optional[tp.BaseFigure] = None,  # 现有图形对象
             **layout_kwargs) -> tp.BaseFigure:  # 布局参数
        """
        绘制布林带指标与收盘价的综合图表
        
        该方法创建一个包含布林带三条线（上轨、中轨、下轨）和收盘价的图表，
        上下轨之间填充半透明区域，形成价格通道效果。
        
        参数说明：
            column (str, optional): 要绘制的数据列名称。
            plot_close (bool): 是否绘制收盘价线条。默认为True。
            close_trace_kwargs (dict, optional): 收盘价线条的样式参数。
            middle_trace_kwargs (dict, optional): 中轨线条的样式参数。
            upper_trace_kwargs (dict, optional): 上轨线条的样式参数。
            lower_trace_kwargs (dict, optional): 下轨线条的样式参数。
            add_trace_kwargs (dict, optional): 添加线条时的通用参数。
            fig (Figure or FigureWidget, optional): 现有的Plotly图形对象。
            **layout_kwargs: 图表布局的关键字参数。
        
        返回值：
            tp.BaseFigure: 包含布林带和收盘价数据的Plotly图形对象。
        
        使用示例：
        ```python
        >>> import vectorbt as vbt
        >>> 
        >>> # 下载数据并计算布林带
        >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
        >>> close_prices = data.get("Close")
        >>> bbands = vbt.BBANDS.run(close_prices)
        >>> 
        >>> # 绘制图表
        >>> fig = bbands.plot(
        ...     title="苹果股票布林带分析",
        ...     close_trace_kwargs={"line": {"color": "black", "width": 2}},
        ...     middle_trace_kwargs={"line": {"color": "blue", "width": 1}},
        ...     upper_trace_kwargs={"line": {"color": "red", "width": 1}},
        ...     lower_trace_kwargs={"line": {"color": "green", "width": 1}}
        ... )
        >>> fig.show()
        ```
        """
        # 导入设置配置
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        # 选择指定列的数据
        self_col = self.select_one(column=column)

        # 创建图形对象（如果未提供）
        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # 设置默认的线条样式参数
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if middle_trace_kwargs is None:
            middle_trace_kwargs = {}
        if upper_trace_kwargs is None:
            upper_trace_kwargs = {}
        if lower_trace_kwargs is None:
            lower_trace_kwargs = {}
        
        # 配置下轨样式
        lower_trace_kwargs = merge_dicts(dict(
            name='Lower Band',  # 下轨名称
            line=dict(
                color=adjust_opacity(plotting_cfg['color_schema']['gray'], 0.75)  # 灰色，75%透明度
            ),
        ), lower_trace_kwargs)
        
        # 配置上轨样式（包含填充效果）
        upper_trace_kwargs = merge_dicts(dict(
            name='Upper Band',  # 上轨名称
            line=dict(
                color=adjust_opacity(plotting_cfg['color_schema']['gray'], 0.75)  # 灰色，75%透明度
            ),
            fill='tonexty',  # 填充到下一条线
            fillcolor='rgba(128, 128, 128, 0.2)'  # 填充颜色：半透明灰色
        ), upper_trace_kwargs)
        
        # 配置中轨样式
        middle_trace_kwargs = merge_dicts(dict(
            name='Middle Band'  # 中轨名称
        ), middle_trace_kwargs)
        
        # 配置收盘价样式
        close_trace_kwargs = merge_dicts(dict(
            name='Close',  # 收盘价名称
            line=dict(color=plotting_cfg['color_schema']['blue'])  # 蓝色
        ), close_trace_kwargs)

        # 按顺序绘制各条线（下轨 -> 上轨 -> 中轨 -> 收盘价）
        fig = self_col.lower.vbt.plot(
            trace_kwargs=lower_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)
        fig = self_col.upper.vbt.plot(
            trace_kwargs=upper_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)
        fig = self_col.middle.vbt.plot(
            trace_kwargs=middle_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)
        if plot_close:
            fig = self_col.close.vbt.plot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs, fig=fig)

        return fig


# 将私有类的文档字符串和方法注入到公共类中
setattr(BBANDS, '__doc__', _BBANDS.__doc__)
setattr(BBANDS, 'plot', _BBANDS.plot)

# ############# RSI（相对强弱指数）############# #

# 创建相对强弱指数指标工厂
RSI = IndicatorFactory(
    class_name='RSI',  # 指标类名
    module_name=__name__,  # 当前模块名
    short_name='rsi',  # 指标简称
    input_names=['close'],  # 输入参数名称列表（收盘价）
    param_names=['window', 'ewm'],  # 参数名称列表（窗口期、是否指数加权）
    output_names=['rsi']  # 输出参数名称列表（RSI值）
).from_apply_func(
    nb.rsi_apply_nb,  # 应用函数（NumPy实现）
    cache_func=nb.rsi_cache_nb,  # 缓存函数
    kwargs_to_args=['adjust'],  # 将关键字参数转换为位置参数
    window=14,  # 默认窗口期为14
    ewm=False,  # 默认不使用指数加权移动平均
    adjust=False  # 默认不调整偏差
)


class _RSI(RSI):
    """
    相对强弱指数（Relative Strength Index, RSI）技术指标类
    
    RSI是一个动量振荡器，通过比较指定时间段内的收益和损失幅度来衡量价格变动的速度和变化。
    它主要用于识别资产交易中的超买或超卖条件。
    
    计算原理：
    1. 计算每日价格变化
    2. 分别计算上涨日和下跌日的平均收益/损失
    3. 计算相对强度RS = 平均收益/平均损失
    4. 计算RSI = 100 - (100 / (1 + RS))
    
    取值范围：
    - RSI值在0-100之间
    - 70以上通常被认为是超买区域
    - 30以下通常被认为是超卖区域
    - 50为中性水平
    
    应用场景：
    - 超买超卖识别：RSI > 70超买，RSI < 30超卖
    - 背离分析：价格与RSI走势相反可能预示反转
    - 中线交易：RSI穿越50线可作为买卖信号
    - 趋势确认：RSI持续高位或低位确认趋势强度
    
    使用示例：
    ```python
    >>> import vectorbt as vbt
    >>> import pandas as pd
    >>> 
    >>> # 下载股票数据
    >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
    >>> close_prices = data.get("Close")
    >>> 
    >>> # 计算14日RSI
    >>> rsi = vbt.RSI.run(close_prices, window=14)
    >>> print(rsi.rsi)
    >>> 
    >>> # 识别超买超卖
    >>> overbought = rsi.rsi > 70
    >>> oversold = rsi.rsi < 30
    >>> print(f"超买次数: {overbought.sum()}")
    >>> print(f"超卖次数: {oversold.sum()}")
    ```
    
    参考资料：
    参见[相对强弱指数](https://www.investopedia.com/terms/r/rsi.asp)
    """

    def plot(self,
             column: tp.Optional[tp.Label] = None,  # 要绘制的列名
             levels: tp.Tuple[float, float] = (30, 70),  # 超买超卖水平线
             rsi_trace_kwargs: tp.KwargsLike = None,  # RSI线条样式参数
             add_trace_kwargs: tp.KwargsLike = None,  # 添加线条的参数
             xref: str = 'x',  # X轴参考
             yref: str = 'y',  # Y轴参考
             fig: tp.Optional[tp.BaseFigure] = None,  # 现有图形对象
             **layout_kwargs) -> tp.BaseFigure:  # 布局参数
        """
        绘制RSI指标图表
        
        该方法创建RSI指标的时间序列图表，包含超买超卖区域的标识，
        帮助识别价格的极端水平和潜在的反转信号。
        
        参数说明：
            column (str, optional): 要绘制的数据列名称。
            levels (tuple): 超买超卖水平线，默认为(30, 70)。
                          第一个值为超卖线，第二个值为超买线。
            rsi_trace_kwargs (dict, optional): RSI线条的样式参数。
            add_trace_kwargs (dict, optional): 添加线条时的通用参数。
            xref (str): X轴坐标参考，默认为'x'。
            yref (str): Y轴坐标参考，默认为'y'。
            fig (Figure or FigureWidget, optional): 现有的Plotly图形对象。
            **layout_kwargs: 图表布局的关键字参数。
        
        返回值：
            tp.BaseFigure: 包含RSI指标的Plotly图形对象。
        
        使用示例：
        ```python
        >>> import vectorbt as vbt
        >>> 
        >>> # 下载数据并计算RSI
        >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
        >>> close_prices = data.get("Close")
        >>> rsi = vbt.RSI.run(close_prices)
        >>> 
        >>> # 绘制图表
        >>> fig = rsi.plot(
        ...     title="苹果股票RSI指标",
        ...     levels=(20, 80),  # 自定义超买超卖水平
        ...     rsi_trace_kwargs={"line": {"color": "purple", "width": 2}}
        ... )
        >>> fig.show()
        ```
        """
        # 选择指定列的数据
        self_col = self.select_one(column=column)

        # 创建图形对象（如果未提供）
        if fig is None:
            fig = make_figure()
        
        # 设置默认布局（Y轴范围为-5到105，留出边距）
        default_layout = dict()
        default_layout['yaxis' + yref[1:]] = dict(range=[-5, 105])
        fig.update_layout(**default_layout)
        fig.update_layout(**layout_kwargs)

        # 设置默认的线条样式参数
        if rsi_trace_kwargs is None:
            rsi_trace_kwargs = {}
        rsi_trace_kwargs = merge_dicts(dict(
            name='RSI'  # RSI线条名称
        ), rsi_trace_kwargs)

        # 绘制RSI线条
        fig = self_col.rsi.vbt.plot(
            trace_kwargs=rsi_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)

        # 添加超买超卖区域填充
        fig.add_shape(
            type="rect",  # 矩形形状
            xref=xref,  # X轴参考
            yref=yref,  # Y轴参考
            x0=self_col.rsi.index[0],  # 起始X坐标
            y0=levels[0],  # 起始Y坐标（超卖线）
            x1=self_col.rsi.index[-1],  # 结束X坐标
            y1=levels[1],  # 结束Y坐标（超买线）
            fillcolor="purple",  # 填充颜色
            opacity=0.2,  # 透明度
            layer="below",  # 放置在底层
            line_width=0,  # 无边框
        )

        return fig


# 将私有类的文档字符串和方法注入到公共类中
setattr(RSI, '__doc__', _RSI.__doc__)
setattr(RSI, 'plot', _RSI.plot)

# ############# STOCH（随机振荡器）############# #

# 创建随机振荡器指标工厂
STOCH = IndicatorFactory(
    class_name='STOCH',  # 指标类名
    module_name=__name__,  # 当前模块名
    short_name='stoch',  # 指标简称
    input_names=['high', 'low', 'close'],  # 输入参数名称列表（最高价、最低价、收盘价）
    param_names=['k_window', 'd_window', 'd_ewm'],  # 参数名称列表（K值窗口、D值窗口、D值是否指数加权）
    output_names=['percent_k', 'percent_d']  # 输出参数名称列表（%K值、%D值）
).from_apply_func(
    nb.stoch_apply_nb,  # 应用函数（NumPy实现）
    cache_func=nb.stoch_cache_nb,  # 缓存函数
    kwargs_to_args=['adjust'],  # 将关键字参数转换为位置参数
    k_window=14,  # 默认K值窗口期为14
    d_window=3,  # 默认D值窗口期为3
    d_ewm=False,  # 默认D值不使用指数加权移动平均
    adjust=False  # 默认不调整偏差
)


class _STOCH(STOCH):
    """
    随机振荡器（Stochastic Oscillator, STOCH）技术指标类
    
    随机振荡器是一个动量指标，将特定收盘价与一定时间段内的价格区间进行比较。
    它用于生成超买和超卖交易信号，使用0-100的有界值范围。
    
    计算公式：
    - %K = (收盘价 - 最低价) / (最高价 - 最低价) × 100
    - %D = %K的n日移动平均值
    
    组成要素：
    - %K线（快速随机指标）：反映当前价格在近期价格区间中的相对位置
    - %D线（慢速随机指标）：%K线的平滑版本，减少假信号
    
    默认参数：
    - K值窗口期：14日
    - D值窗口期：3日
    
    应用场景：
    - 超买超卖识别：%K或%D > 80超买，< 20超卖
    - 金叉死叉：%K线上穿%D线为买入信号，下穿为卖出信号
    - 背离分析：价格与指标走势相反可能预示反转
    - 区间震荡：在震荡市中效果较好
    
    使用示例：
    ```python
    >>> import vectorbt as vbt
    >>> import pandas as pd
    >>> 
    >>> # 下载股票数据
    >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
    >>> ohlc = data.get(["High", "Low", "Close"])
    >>> 
    >>> # 计算随机振荡器
    >>> stoch = vbt.STOCH.run(ohlc['High'], ohlc['Low'], ohlc['Close'])
    >>> print(stoch.percent_k)  # %K值
    >>> print(stoch.percent_d)  # %D值
    >>> 
    >>> # 识别交易信号
    >>> golden_cross = (stoch.percent_k > stoch.percent_d) & (stoch.percent_k.shift(1) <= stoch.percent_d.shift(1))
    >>> death_cross = (stoch.percent_k < stoch.percent_d) & (stoch.percent_k.shift(1) >= stoch.percent_d.shift(1))
    ```
    
    参考资料：
    参见[随机振荡器](https://www.investopedia.com/terms/s/stochasticoscillator.asp)
    """

    def plot(self,
             column: tp.Optional[tp.Label] = None,  # 要绘制的列名
             levels: tp.Tuple[float, float] = (30, 70),  # 超买超卖水平线
             percent_k_trace_kwargs: tp.KwargsLike = None,  # %K线条样式参数
             percent_d_trace_kwargs: tp.KwargsLike = None,  # %D线条样式参数
             shape_kwargs: tp.KwargsLike = None,  # 形状样式参数
             add_trace_kwargs: tp.KwargsLike = None,  # 添加线条的参数
             xref: str = 'x',  # X轴参考
             yref: str = 'y',  # Y轴参考
             fig: tp.Optional[tp.BaseFigure] = None,  # 现有图形对象
             **layout_kwargs) -> tp.BaseFigure:  # 布局参数
        """
        绘制随机振荡器指标图表
        
        该方法创建包含%K线和%D线的随机振荡器图表，并标识超买超卖区域，
        帮助识别动量变化和潜在的交易机会。
        
        参数说明：
            column (str, optional): 要绘制的数据列名称。
            levels (tuple): 超买超卖水平线，默认为(30, 70)。
            percent_k_trace_kwargs (dict, optional): %K线的样式参数。
            percent_d_trace_kwargs (dict, optional): %D线的样式参数。
            shape_kwargs (dict, optional): 超买超卖区域形状的样式参数。
            add_trace_kwargs (dict, optional): 添加线条时的通用参数。
            xref (str): X轴坐标参考，默认为'x'。
            yref (str): Y轴坐标参考，默认为'y'。
            fig (Figure or FigureWidget, optional): 现有的Plotly图形对象。
            **layout_kwargs: 图表布局的关键字参数。
        
        返回值：
            tp.BaseFigure: 包含随机振荡器指标的Plotly图形对象。
        
        使用示例：
        ```python
        >>> import vectorbt as vbt
        >>> 
        >>> # 下载数据并计算随机振荡器
        >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
        >>> ohlc = data.get(["High", "Low", "Close"])
        >>> stoch = vbt.STOCH.run(ohlc['High'], ohlc['Low'], ohlc['Close'])
        >>> 
        >>> # 绘制图表
        >>> fig = stoch.plot(
        ...     title="苹果股票随机振荡器",
        ...     levels=(20, 80),  # 自定义超买超卖水平
        ...     percent_k_trace_kwargs={"line": {"color": "blue", "width": 2}},
        ...     percent_d_trace_kwargs={"line": {"color": "red", "width": 2}}
        ... )
        >>> fig.show()
        ```
        """
        # 选择指定列的数据
        self_col = self.select_one(column=column)

        # 创建图形对象（如果未提供）
        if fig is None:
            fig = make_figure()
        
        # 设置默认布局（Y轴范围为-5到105）
        default_layout = dict()
        default_layout['yaxis' + yref[1:]] = dict(range=[-5, 105])
        fig.update_layout(**default_layout)
        fig.update_layout(**layout_kwargs)

        # 设置默认的线条样式参数
        if percent_k_trace_kwargs is None:
            percent_k_trace_kwargs = {}
        if percent_d_trace_kwargs is None:
            percent_d_trace_kwargs = {}
        if shape_kwargs is None:
            shape_kwargs = {}
        
        # 配置%K线样式
        percent_k_trace_kwargs = merge_dicts(dict(
            name='%K'  # %K线名称
        ), percent_k_trace_kwargs)
        
        # 配置%D线样式
        percent_d_trace_kwargs = merge_dicts(dict(
            name='%D'  # %D线名称
        ), percent_d_trace_kwargs)

        # 绘制%K线和%D线
        fig = self_col.percent_k.vbt.plot(
            trace_kwargs=percent_k_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)
        fig = self_col.percent_d.vbt.plot(
            trace_kwargs=percent_d_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)

        # 添加超买超卖区域填充
        shape_kwargs = merge_dicts(dict(
            type="rect",  # 矩形形状
            xref=xref,  # X轴参考
            yref=yref,  # Y轴参考
            x0=self_col.percent_k.index[0],  # 起始X坐标
            y0=levels[0],  # 起始Y坐标（超卖线）
            x1=self_col.percent_k.index[-1],  # 结束X坐标
            y1=levels[1],  # 结束Y坐标（超买线）
            fillcolor="purple",  # 填充颜色
            opacity=0.2,  # 透明度
            layer="below",  # 放置在底层
            line_width=0,  # 无边框
        ), shape_kwargs)
        fig.add_shape(**shape_kwargs)

        return fig


# 将私有类的文档字符串和方法注入到公共类中
setattr(STOCH, '__doc__', _STOCH.__doc__)
setattr(STOCH, 'plot', _STOCH.plot)

# ############# MACD（平滑异同移动平均线）############# #

# 创建MACD指标工厂
MACD = IndicatorFactory(
    class_name='MACD',  # 指标类名
    module_name=__name__,  # 当前模块名
    short_name='macd',  # 指标简称
    input_names=['close'],  # 输入参数名称列表（收盘价）
    param_names=['fast_window', 'slow_window', 'signal_window', 'macd_ewm', 'signal_ewm'],  # 参数名称列表
    output_names=['macd', 'signal'],  # 输出参数名称列表（MACD线、信号线）
    custom_output_props=dict(
        # 自定义输出属性：MACD柱状图（MACD线与信号线的差值）
        hist=lambda self: self.wrapper.wrap(self.macd.values - self.signal.values),
    )
).from_apply_func(
    nb.macd_apply_nb,  # 应用函数（NumPy实现）
    cache_func=nb.macd_cache_nb,  # 缓存函数
    kwargs_to_args=['adjust'],  # 将关键字参数转换为位置参数
    fast_window=12,  # 默认快速移动平均窗口期为12
    slow_window=26,  # 默认慢速移动平均窗口期为26
    signal_window=9,  # 默认信号线窗口期为9
    macd_ewm=False,  # 默认MACD不使用指数加权移动平均
    signal_ewm=False,  # 默认信号线不使用指数加权移动平均
    adjust=False  # 默认不调整偏差
)


class _MACD(MACD):
    """
    平滑异同移动平均线（Moving Average Convergence Divergence, MACD）技术指标类
    
    MACD是一个趋势跟踪动量指标，显示两条价格移动平均线之间的关系。
    它是技术分析中最受欢迎和广泛使用的指标之一。
    
    组成要素：
    - MACD线：快速移动平均线（12日）- 慢速移动平均线（26日）
    - 信号线：MACD线的移动平均线（通常为9日）
    - 柱状图：MACD线与信号线的差值
    
    默认参数：
    - 快速移动平均：12日
    - 慢速移动平均：26日
    - 信号线：9日
    
    应用场景：
    - 趋势跟踪：MACD线上穿信号线为买入信号，下穿为卖出信号
    - 动量分析：柱状图反映动量变化，正值表示上涨动量，负值表示下跌动量
    - 背离分析：价格与MACD走势相反可能预示趋势反转
    - 零轴交叉：MACD线穿越零轴表示趋势可能发生改变
    
    使用示例：
    ```python
    >>> import vectorbt as vbt
    >>> import pandas as pd
    >>> 
    >>> # 下载股票数据
    >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
    >>> close_prices = data.get("Close")
    >>> 
    >>> # 计算MACD指标
    >>> macd = vbt.MACD.run(close_prices)
    >>> print(macd.macd)    # MACD线
    >>> print(macd.signal)  # 信号线
    >>> print(macd.hist)    # 柱状图
    >>> 
    >>> # 识别交易信号
    >>> golden_cross = (macd.macd > macd.signal) & (macd.macd.shift(1) <= macd.signal.shift(1))
    >>> death_cross = (macd.macd < macd.signal) & (macd.macd.shift(1) >= macd.signal.shift(1))
    ```
    
    参考资料：
    参见[MACD](https://www.investopedia.com/terms/m/macd.asp)
    """

    def plot(self,
             column: tp.Optional[tp.Label] = None,  # 要绘制的列名
             macd_trace_kwargs: tp.KwargsLike = None,  # MACD线条样式参数
             signal_trace_kwargs: tp.KwargsLike = None,  # 信号线条样式参数
             hist_trace_kwargs: tp.KwargsLike = None,  # 柱状图样式参数
             add_trace_kwargs: tp.KwargsLike = None,  # 添加线条的参数
             fig: tp.Optional[tp.BaseFigure] = None,  # 现有图形对象
             **layout_kwargs) -> tp.BaseFigure:  # 布局参数
        """
        绘制MACD指标综合图表
        
        该方法创建包含MACD线、信号线和柱状图的综合图表，柱状图使用颜色编码
        来表示动量的强弱和方向变化。
        
        参数说明：
            column (str, optional): 要绘制的数据列名称。
            macd_trace_kwargs (dict, optional): MACD线的样式参数。
            signal_trace_kwargs (dict, optional): 信号线的样式参数。
            hist_trace_kwargs (dict, optional): 柱状图的样式参数。
            add_trace_kwargs (dict, optional): 添加线条时的通用参数。
            fig (Figure or FigureWidget, optional): 现有的Plotly图形对象。
            **layout_kwargs: 图表布局的关键字参数。
        
        返回值：
            tp.BaseFigure: 包含MACD指标的Plotly图形对象。
        
        柱状图颜色编码：
        - 深绿色：正值且上升（强势上涨动量）
        - 浅绿色：正值且下降（弱势上涨动量）
        - 深红色：负值且下降（强势下跌动量）
        - 浅红色：负值且上升（弱势下跌动量）
        
        使用示例：
        ```python
        >>> import vectorbt as vbt
        >>> 
        >>> # 下载数据并计算MACD
        >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
        >>> close_prices = data.get("Close")
        >>> macd = vbt.MACD.run(close_prices)
        >>> 
        >>> # 绘制图表
        >>> fig = macd.plot(
        ...     title="苹果股票MACD指标",
        ...     macd_trace_kwargs={"line": {"color": "blue", "width": 2}},
        ...     signal_trace_kwargs={"line": {"color": "red", "width": 2}}
        ... )
        >>> fig.show()
        ```
        """
        # 选择指定列的数据
        self_col = self.select_one(column=column)

        # 创建图形对象（如果未提供）
        if fig is None:
            fig = make_figure()
            fig.update_layout(bargap=0)  # 设置柱状图间距为0
        fig.update_layout(**layout_kwargs)

        # 设置默认的线条样式参数
        if macd_trace_kwargs is None:
            macd_trace_kwargs = {}
        if signal_trace_kwargs is None:
            signal_trace_kwargs = {}
        if hist_trace_kwargs is None:
            hist_trace_kwargs = {}
        
        # 配置MACD线样式
        macd_trace_kwargs = merge_dicts(dict(
            name='MACD'  # MACD线名称
        ), macd_trace_kwargs)
        
        # 配置信号线样式
        signal_trace_kwargs = merge_dicts(dict(
            name='Signal'  # 信号线名称
        ), signal_trace_kwargs)
        
        # 配置柱状图样式
        hist_trace_kwargs = merge_dicts(dict(name='Histogram'), hist_trace_kwargs)

        # 绘制MACD线和信号线
        fig = self_col.macd.vbt.plot(
            trace_kwargs=macd_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)
        fig = self_col.signal.vbt.plot(
            trace_kwargs=signal_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)

        # 绘制柱状图（带颜色编码）
        hist = self_col.hist.values  # 获取柱状图数值
        hist_diff = generic_nb.diff_1d_nb(hist)  # 计算柱状图的变化
        
        # 创建颜色数组
        marker_colors = np.full(hist.shape, adjust_opacity('silver', 0.75), dtype=object)
        
        # 根据柱状图数值和变化方向设置颜色
        marker_colors[(hist > 0) & (hist_diff > 0)] = adjust_opacity('green', 0.75)  # 正值上升：深绿
        marker_colors[(hist > 0) & (hist_diff <= 0)] = adjust_opacity('lightgreen', 0.75)  # 正值下降：浅绿
        marker_colors[(hist < 0) & (hist_diff < 0)] = adjust_opacity('red', 0.75)  # 负值下降：深红
        marker_colors[(hist < 0) & (hist_diff >= 0)] = adjust_opacity('lightcoral', 0.75)  # 负值上升：浅红

        # 创建柱状图对象
        hist_bar = go.Bar(
            x=self_col.hist.index,  # X轴数据（时间）
            y=self_col.hist.values,  # Y轴数据（柱状图数值）
            marker_color=marker_colors,  # 柱状图颜色
            marker_line_width=0  # 无边框
        )
        hist_bar.update(**hist_trace_kwargs)  # 应用自定义样式
        
        # 添加柱状图到图表
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        fig.add_trace(hist_bar, **add_trace_kwargs)

        return fig


# 将私有类的文档字符串和方法注入到公共类中
setattr(MACD, '__doc__', _MACD.__doc__)
setattr(MACD, 'plot', _MACD.plot)

# ############# ATR（真实波幅）############# #

# 创建真实波幅指标工厂
ATR = IndicatorFactory(
    class_name='ATR',  # 指标类名
    module_name=__name__,  # 当前模块名
    short_name='atr',  # 指标简称
    input_names=['high', 'low', 'close'],  # 输入参数名称列表（最高价、最低价、收盘价）
    param_names=['window', 'ewm'],  # 参数名称列表（窗口期、是否指数加权）
    output_names=['tr', 'atr']  # 输出参数名称列表（真实波幅、平均真实波幅）
).from_apply_func(
    nb.atr_apply_nb,  # 应用函数（NumPy实现）
    cache_func=nb.atr_cache_nb,  # 缓存函数
    kwargs_to_args=['adjust'],  # 将关键字参数转换为位置参数
    window=14,  # 默认窗口期为14
    ewm=True,  # 默认使用指数加权移动平均
    adjust=False  # 默认不调整偏差
)


class _ATR(ATR):
    """
    平均真实波幅（Average True Range, ATR）技术指标类
    
    ATR指标提供了价格波动程度的指示。强烈的价格变动（无论方向）通常伴随着大的波幅，
    即大的真实波幅。它主要用于衡量市场波动性，而不是价格方向。
    
    计算方法：
    1. 真实波幅TR = max(最高价-最低价, |最高价-前收盘价|, |最低价-前收盘价|)
    2. 平均真实波幅ATR = TR的n日移动平均
    
    默认参数：
    - 窗口期：14日
    - 使用指数加权移动平均
    
    应用场景：
    - 波动性分析：ATR值越高，波动性越大
    - 风险管理：根据ATR设置止损位
    - 仓位管理：根据波动性调整仓位大小
    - 市场状态识别：低ATR可能预示突破，高ATR表示高波动
    - 交易系统：作为其他指标的输入参数
    
    使用示例：
    ```python
    >>> import vectorbt as vbt
    >>> import pandas as pd
    >>> 
    >>> # 下载股票数据
    >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
    >>> ohlc = data.get(["High", "Low", "Close"])
    >>> 
    >>> # 计算ATR指标
    >>> atr = vbt.ATR.run(ohlc['High'], ohlc['Low'], ohlc['Close'])
    >>> print(atr.tr)   # 真实波幅
    >>> print(atr.atr)  # 平均真实波幅
    >>> 
    >>> # 基于ATR设置止损
    >>> stop_loss_pct = atr.atr / ohlc['Close'] * 2  # 2倍ATR作为止损
    >>> print(stop_loss_pct)
    ```
    
    注意事项：
    与Wilder的原始计算方法相比，本实现使用简单移动平均和指数移动平均。
    
    参考资料：
    参见[平均真实波幅](https://www.investopedia.com/terms/a/atr.asp)
    """

    def plot(self,
             column: tp.Optional[tp.Label] = None,  # 要绘制的列名
             tr_trace_kwargs: tp.KwargsLike = None,  # TR线条样式参数
             atr_trace_kwargs: tp.KwargsLike = None,  # ATR线条样式参数
             add_trace_kwargs: tp.KwargsLike = None,  # 添加线条的参数
             fig: tp.Optional[tp.BaseFigure] = None,  # 现有图形对象
             **layout_kwargs) -> tp.BaseFigure:  # 布局参数
        """
        绘制ATR指标图表
        
        该方法创建包含真实波幅和平均真实波幅的图表，帮助分析价格波动性的变化。
        
        参数说明：
            column (str, optional): 要绘制的数据列名称。
            tr_trace_kwargs (dict, optional): TR线条的样式参数。
            atr_trace_kwargs (dict, optional): ATR线条的样式参数。
            add_trace_kwargs (dict, optional): 添加线条时的通用参数。
            fig (Figure or FigureWidget, optional): 现有的Plotly图形对象。
            **layout_kwargs: 图表布局的关键字参数。
        
        返回值：
            tp.BaseFigure: 包含ATR指标的Plotly图形对象。
        
        使用示例：
        ```python
        >>> import vectorbt as vbt
        >>> 
        >>> # 下载数据并计算ATR
        >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
        >>> ohlc = data.get(["High", "Low", "Close"])
        >>> atr = vbt.ATR.run(ohlc['High'], ohlc['Low'], ohlc['Close'], window=10)
        >>> 
        >>> # 绘制图表
        >>> fig = atr.plot(
        ...     title="苹果股票ATR指标",
        ...     tr_trace_kwargs={"line": {"color": "lightblue", "width": 1}},
        ...     atr_trace_kwargs={"line": {"color": "blue", "width": 2}}
        ... )
        >>> fig.show()
        ```
        """
        # 选择指定列的数据
        self_col = self.select_one(column=column)

        # 创建图形对象（如果未提供）
        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # 设置默认的线条样式参数
        if tr_trace_kwargs is None:
            tr_trace_kwargs = {}
        if atr_trace_kwargs is None:
            atr_trace_kwargs = {}
        
        # 配置TR线样式
        tr_trace_kwargs = merge_dicts(dict(
            name='TR'  # 真实波幅线名称
        ), tr_trace_kwargs)
        
        # 配置ATR线样式
        atr_trace_kwargs = merge_dicts(dict(
            name='ATR'  # 平均真实波幅线名称
        ), atr_trace_kwargs)

        # 绘制TR线和ATR线
        fig = self_col.tr.vbt.plot(
            trace_kwargs=tr_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)
        fig = self_col.atr.vbt.plot(
            trace_kwargs=atr_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)

        return fig


# 将私有类的文档字符串和方法注入到公共类中
setattr(ATR, '__doc__', _ATR.__doc__)
setattr(ATR, 'plot', _ATR.plot)

# ############# OBV（能量潮）############# #

# 创建能量潮指标工厂
OBV = IndicatorFactory(
    class_name='OBV',  # 指标类名
    module_name=__name__,  # 当前模块名
    short_name='obv',  # 指标简称
    input_names=['close', 'volume'],  # 输入参数名称列表（收盘价、成交量）
    param_names=[],  # 参数名称列表（无额外参数）
    output_names=['obv'],  # 输出参数名称列表（OBV值）
).from_custom_func(nb.obv_custom_nb)  # 使用自定义函数


class _OBV(OBV):
    """
    能量潮（On-Balance Volume, OBV）技术指标类
    
    OBV是一个动量指标，通过将价格和成交量相关联来分析股票市场。
    它基于累积总成交量，是最早的价量指标之一。
    
    计算原理：
    - 如果收盘价高于前一日收盘价，则当日成交量为正
    - 如果收盘价低于前一日收盘价，则当日成交量为负
    - 如果收盘价等于前一日收盘价，则当日成交量为零
    - OBV = 前一日OBV + 当日成交量（带符号）
    
    核心思想：
    成交量先于价格变动，OBV的变化可以预测价格走势的变化。
    
    应用场景：
    - 趋势确认：OBV与价格同向变动确认趋势
    - 背离分析：OBV与价格反向变动可能预示反转
    - 突破确认：价格突破配合OBV突破更可靠
    - 资金流向：OBV上升表示资金流入，下降表示资金流出
    
    使用示例：
    ```python
    >>> import vectorbt as vbt
    >>> import pandas as pd
    >>> 
    >>> # 下载股票数据
    >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
    >>> close_prices = data.get("Close")
    >>> volume = data.get("Volume")
    >>> 
    >>> # 计算OBV指标
    >>> obv = vbt.OBV.run(close_prices, volume)
    >>> print(obv.obv)
    >>> 
    >>> # 分析资金流向
    >>> obv_change = obv.obv.diff()
    >>> money_flow_in = obv_change > 0
    >>> money_flow_out = obv_change < 0
    >>> print(f"资金流入天数: {money_flow_in.sum()}")
    >>> print(f"资金流出天数: {money_flow_out.sum()}")
    ```
    
    参考资料：
    参见[能量潮](https://www.investopedia.com/terms/o/onbalancevolume.asp)
    """

    def plot(self,
             column: tp.Optional[tp.Label] = None,  # 要绘制的列名
             obv_trace_kwargs: tp.KwargsLike = None,  # OBV线条样式参数
             add_trace_kwargs: tp.KwargsLike = None,  # 添加线条的参数
             fig: tp.Optional[tp.BaseFigure] = None,  # 现有图形对象
             **layout_kwargs) -> tp.BaseFigure:  # 布局参数
        """
        绘制OBV指标图表
        
        该方法创建能量潮指标的时间序列图表，显示累积成交量的变化趋势，
        帮助分析资金流向和价格趋势的关系。
        
        参数说明：
            column (str, optional): 要绘制的数据列名称。
            obv_trace_kwargs (dict, optional): OBV线条的样式参数。
            add_trace_kwargs (dict, optional): 添加线条时的通用参数。
            fig (Figure or FigureWidget, optional): 现有的Plotly图形对象。
            **layout_kwargs: 图表布局的关键字参数。
        
        返回值：
            tp.BaseFigure: 包含OBV指标的Plotly图形对象。
        
        使用示例：
        ```python
        >>> import vectorbt as vbt
        >>> 
        >>> # 下载数据并计算OBV
        >>> data = vbt.YFData.download("AAPL", start="2023-01-01", end="2023-12-31")
        >>> close_prices = data.get("Close")
        >>> volume = data.get("Volume")
        >>> obv = vbt.OBV.run(close_prices, volume)
        >>> 
        >>> # 绘制图表
        >>> fig = obv.plot(
        ...     title="苹果股票OBV指标",
        ...     obv_trace_kwargs={"line": {"color": "green", "width": 2}}
        ... )
        >>> fig.show()
        ```
        """
        # 选择指定列的数据
        self_col = self.select_one(column=column)

        # 创建图形对象（如果未提供）
        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # 设置默认的线条样式参数
        if obv_trace_kwargs is None:
            obv_trace_kwargs = {}
        obv_trace_kwargs = merge_dicts(dict(
            name='OBV'  # OBV线条名称
        ), obv_trace_kwargs)

        # 绘制OBV线条
        fig = self_col.obv.vbt.plot(
            trace_kwargs=obv_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)

        return fig


# 将私有类的文档字符串和方法注入到公共类中
setattr(OBV, '__doc__', _OBV.__doc__)
setattr(OBV, 'plot', _OBV.plot)
