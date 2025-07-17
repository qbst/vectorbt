# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT OHLCV_ACCESSORS MODULE: OHLCV金融数据访问器模块
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于处理OHLCV（开盘价Open、最高价High、最低价Low、
收盘价Close、成交量Volume）金融数据的核心访问器模块。OHLCV数据是量化交易中最基础、
最重要的数据类型，几乎所有的技术分析、策略回测和风险管理都建立在这类数据之上。

核心设计理念：
1. **标准化数据接口**：提供统一的OHLCV数据访问接口，屏蔽不同数据源的列名差异，
   支持灵活的列名映射配置，确保代码的可移植性和兼容性。

2. **专业金融分析**：内置丰富的OHLCV专用统计指标（如价格区间、成交量分布等），
   为金融数据分析提供了开箱即用的专业工具。

3. **高质量可视化**：提供专业的K线图（蜡烛图）和OHLC柱状图绘制功能，
   支持成交量叠加显示，满足金融数据可视化的专业要求。

4. **配置驱动架构**：通过配置系统支持灵活的自定义，包括列名映射、
   统计指标配置、绘图主题等，适应不同的业务需求。

主要功能模块：
- **列名智能映射**：自动识别和映射不同格式的OHLCV数据列名
- **数据属性访问**：提供open、high、low、close、volume等便捷属性访问
- **统计指标计算**：计算价格区间、成交量统计等专业金融指标
- **专业图表绘制**：支持K线图、OHLC图、成交量图等多种金融图表
- **配置管理系统**：支持全局和局部的配置定制

技术架构特点：
- 继承自GenericDFAccessor，获得通用数据处理能力
- 使用装饰器注册为DataFrame的vbt访问器
- 集成vectorbt的配置系统和绘图框架
- 支持灵活的列名映射和数据验证

应用场景：
- **技术分析**：为移动平均线、RSI、MACD等技术指标提供数据基础
- **策略回测**：为交易策略提供标准化的价格和成交量数据
- **风险分析**：计算价格波动率、成交量异常等风险指标
- **数据可视化**：生成专业的K线图、价格走势图等金融图表
- **多数据源整合**：统一处理来自不同交易所或数据供应商的OHLCV数据

使用示例：
```python
import pandas as pd
import vectorbt as vbt

# 创建OHLCV数据
df = pd.DataFrame({
    'open': [100, 105, 102, 108, 106],
    'high': [110, 108, 107, 112, 110],
    'low': [98, 102, 100, 105, 104],
    'close': [105, 104, 106, 109, 107],
    'volume': [1000000, 1200000, 800000, 1500000, 1100000]
})

# 使用OHLCV访问器
ohlcv = df.vbt.ohlcv

# 获取各种数据
print("开盘价:", ohlcv.open)
print("最高价:", ohlcv.high)
print("收盘价:", ohlcv.close)
print("成交量:", ohlcv.volume)

# 计算统计指标
stats = ohlcv.stats()
print("统计指标:", stats)

# 绘制K线图
fig = ohlcv.plot(plot_type='candlestick', show_volume=True)
fig.show()

# 自定义列名映射
custom_df = pd.DataFrame({
    'o': [100, 105, 102],
    'h': [110, 108, 107],
    'l': [98, 102, 100],
    'c': [105, 104, 106],
    'v': [1000000, 1200000, 800000]
})

custom_ohlcv = custom_df.vbt.ohlcv(column_names={
    'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'
})
```

与vectorbt生态系统的关系：
- 为Portfolio、Strategy等高级模块提供数据输入
- 与Indicators模块协作进行技术指标计算
- 集成vectorbt的绘图系统，支持主题和样式定制
- 使用vectorbt的配置系统，支持全局参数设置

该模块是vectorbt框架中金融数据处理的基础组件，为量化交易分析提供了
专业、高效、易用的OHLCV数据处理能力。

自定义OHLC(V)数据的pandas访问器。

方法可以通过以下方式访问：

* `OHLCVDFAccessor` -> `pd.DataFrame.vbt.ohlc.*`
* `OHLCVDFAccessor` -> `pd.DataFrame.vbt.ohlcv.*`

访问器继承自`vectorbt.generic.accessors`。

!!! note
    访问器不使用缓存机制。

## 列名

默认情况下，vectorbt会搜索名为'open'、'high'、'low'、'close'和'volume'的列
（不区分大小写）。您可以通过`vectorbt._settings.settings`中的`ohlcv.column_names`
或直接向访问器提供`column_names`来更改命名。

```pycon
>>> import pandas as pd
>>> import vectorbt as vbt

>>> df = pd.DataFrame({
...     'my_open1': [2, 3, 4, 3.5, 2.5],
...     'my_high2': [3, 4, 4.5, 4, 3],
...     'my_low3': [1.5, 2.5, 3.5, 2.5, 1.5],
...     'my_close4': [2.5, 3.5, 4, 3, 2],
...     'my_volume5': [10, 11, 10, 9, 10]
... })

>>> # vectorbt无法找到列
>>> df.vbt.ohlcv.get_column('open')
None

>>> my_column_names = dict(
...     open='my_open1',
...     high='my_high2',
...     low='my_low3',
...     close='my_close4',
...     volume='my_volume5',
... )
>>> ohlcv_acc = df.vbt.ohlcv(freq='d', column_names=my_column_names)
>>> ohlcv_acc.get_column('open')
0    2.0
1    3.0
2    4.0
3    3.5
4    2.5
Name: my_open1, dtype: float64
```

## 统计指标

!!! hint
    参见`vectorbt.generic.stats_builder.StatsBuilderMixin.stats`和`OHLCVDFAccessor.metrics`。

```pycon
>>> ohlcv_acc.stats()
Start                           0
End                             4
Period            5 days 00:00:00
First Price                   2.0
Lowest Price                  1.5
Highest Price                 4.5
Last Price                    2.0
First Volume                   10
Lowest Volume                   9
Highest Volume                 11
Last Volume                    10
Name: agg_func_mean, dtype: object
```

## 绘图

!!! hint
    参见`vectorbt.generic.plots_builder.PlotsBuilderMixin.plots`和`OHLCVDFAccessor.subplots`。

`OHLCVDFAccessor`类有一个基于`OHLCVDFAccessor.plot`的单一子图（不包含成交量）：

```pycon
>>> ohlcv_acc.plots(settings=dict(plot_type='candlestick'))
```

![](/assets/images/ohlcv_plots.svg)
"""

import numpy as np  # 导入NumPy，用于数值计算和数组操作
import pandas as pd  # 导入Pandas，用于数据结构和数据分析
import plotly.graph_objects as go  # 导入Plotly图形对象，用于创建交互式图表

from vectorbt import _typing as tp  # 导入vectorbt类型定义模块
from vectorbt.generic import nb  # 导入通用Numba编译函数模块
from vectorbt.generic.accessors import GenericAccessor, GenericDFAccessor  # 导入通用访问器类
from vectorbt.root_accessors import register_dataframe_vbt_accessor  # 导入DataFrame访问器注册函数
from vectorbt.utils.config import merge_dicts, Config  # 导入配置管理工具
from vectorbt.utils.figure import make_figure, make_subplots  # 导入图表创建工具

__pdoc__ = {}  # 初始化文档字典，用于控制文档生成


@register_dataframe_vbt_accessor('ohlc')  # 注册为DataFrame的'ohlc'访问器
@register_dataframe_vbt_accessor('ohlcv')  # 注册为DataFrame的'ohlcv'访问器（支持两种名称）
class OHLCVDFAccessor(GenericDFAccessor):  # pragma: no cover
    """
    OHLCV数据访问器类 - 专门处理金融时间序列数据的高级访问器
    
    该类是vectorbt框架中专门用于处理OHLCV（开盘价、最高价、最低价、收盘价、成交量）
    金融数据的核心访问器。它提供了完整的OHLCV数据处理、分析和可视化功能，
    是量化交易分析的重要工具。
    
    核心功能：
    - **智能列名映射**：自动识别和映射不同格式的OHLCV数据列名
    - **数据属性访问**：提供便捷的open、high、low、close、volume属性访问
    - **专业统计指标**：计算价格区间、成交量分布等金融专用指标
    - **专业图表绘制**：支持K线图、OHLC图、成交量图等多种金融图表
    - **配置管理**：支持灵活的列名映射和绘图配置
    
    技术特点：
    - 继承自GenericDFAccessor，获得通用数据处理能力
    - 使用装饰器模式注册为DataFrame的vbt访问器
    - 集成vectorbt的配置系统和绘图框架
    - 支持不区分大小写的列名匹配
    - 提供丰富的统计指标和可视化选项
    
    访问方式：
    - 通过`pd.DataFrame.vbt.ohlcv`访问
    - 通过`pd.DataFrame.vbt.ohlc`访问（别名）
    
    列名映射：
    默认搜索以下列名（不区分大小写）：
    - 'open': 开盘价
    - 'high': 最高价  
    - 'low': 最低价
    - 'close': 收盘价
    - 'volume': 成交量
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 标准OHLCV数据
    df = pd.DataFrame({
        'open': [100, 105, 102, 108, 106],
        'high': [110, 108, 107, 112, 110], 
        'low': [98, 102, 100, 105, 104],
        'close': [105, 104, 106, 109, 107],
        'volume': [1000000, 1200000, 800000, 1500000, 1100000]
    })
    
    # 创建OHLCV访问器
    ohlcv = df.vbt.ohlcv
    
    # 访问数据属性
    print("开盘价:", ohlcv.open)
    print("最高价:", ohlcv.high) 
    print("收盘价:", ohlcv.close)
    print("OHLC数据:", ohlcv.ohlc)
    
    # 计算统计指标
    stats = ohlcv.stats()
    print("统计指标:", stats)
    
    # 绘制K线图
    fig = ohlcv.plot(plot_type='candlestick', show_volume=True)
    fig.show()
    
    # 自定义列名映射
    custom_df = pd.DataFrame({
        'o': [100, 105, 102],
        'h': [110, 108, 107],
        'l': [98, 102, 100], 
        'c': [105, 104, 106],
        'v': [1000000, 1200000, 800000]
    })
    
    # 使用自定义列名
    custom_ohlcv = custom_df.vbt.ohlcv(column_names={
        'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'
    })
    
    # 绘制自定义数据的图表
    fig = custom_ohlcv.plot(plot_type='ohlc')
    fig.show()
    ```
    
    高级应用：
    ```python
    # 技术分析应用
    sma_20 = ohlcv.close.rolling(20).mean()  # 20日移动平均
    volatility = ohlcv.close.pct_change().std()  # 价格波动率
    
    # 成交量分析
    avg_volume = ohlcv.volume.mean()  # 平均成交量
    volume_ratio = ohlcv.volume / avg_volume  # 成交量比率
    
    # 价格分析
    price_range = ohlcv.high - ohlcv.low  # 价格区间
    body_size = abs(ohlcv.close - ohlcv.open)  # K线实体大小
    
    # 组合分析
    high_volume_days = ohlcv.volume > ohlcv.volume.quantile(0.8)
    price_on_high_volume = ohlcv.close[high_volume_days]
    ```
    
    配置选项：
    - column_names: 自定义列名映射
    - freq: 时间频率设置
    - plot_type: 默认绘图类型（'candlestick'或'ohlc'）
    - show_volume: 是否显示成交量
    
    注意事项：
    - 访问器不使用缓存，每次调用都会重新计算
    - 列名匹配不区分大小写
    - 如果找不到对应列，相关属性将返回None
    - 支持时间序列索引，自动处理时间相关的统计指标
    
    仅适用于DataFrame数据。
    
    可通过`pd.DataFrame.vbt.ohlcv`访问。
    """

    def __init__(self, obj: tp.Frame, column_names: tp.KwargsLike = None, **kwargs) -> None:
        """
        初始化OHLCV访问器实例
        
        该方法创建一个OHLCV访问器实例，设置列名映射和其他配置参数。
        
        参数：
            obj: pandas DataFrame对象，包含OHLCV数据
            column_names: 自定义列名映射字典，用于指定OHLCV列的实际名称
            **kwargs: 传递给父类的其他配置参数
            
        初始化流程：
        1. 存储自定义列名映射配置
        2. 调用父类GenericDFAccessor的初始化方法
        3. 设置访问器的基本配置和元数据
        
        列名映射格式：
        ```python
        column_names = {
            'open': 'actual_open_column_name',
            'high': 'actual_high_column_name', 
            'low': 'actual_low_column_name',
            'close': 'actual_close_column_name',
            'volume': 'actual_volume_column_name'
        }
        ```
        
        使用示例：
        ```python
        # 标准列名（自动识别）
        df = pd.DataFrame({
            'open': [100, 105], 'high': [110, 108], 
            'low': [98, 102], 'close': [105, 104], 'volume': [1000, 1200]
        })
        ohlcv1 = df.vbt.ohlcv  # 自动识别标准列名
        
        # 自定义列名映射
        custom_df = pd.DataFrame({
            'o': [100, 105], 'h': [110, 108], 
            'l': [98, 102], 'c': [105, 104], 'v': [1000, 1200]
        })
        ohlcv2 = custom_df.vbt.ohlcv(column_names={
            'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'
        })
        
        # 部分列名映射（只映射部分列）
        partial_df = pd.DataFrame({
            'open_price': [100, 105], 'high_price': [110, 108],
            'low': [98, 102], 'close': [105, 104]  # low和close使用标准名称
        })
        ohlcv3 = partial_df.vbt.ohlcv(column_names={
            'open': 'open_price', 'high': 'high_price'
        })
        ```
        """
        self._column_names = column_names  # 存储自定义列名映射配置

        GenericDFAccessor.__init__(self, obj, column_names=column_names, **kwargs)  # 调用父类初始化方法

    @property
    def column_names(self) -> tp.Kwargs:
        """
        获取列名映射配置
        
        该属性返回当前访问器使用的列名映射配置，合并了全局设置和实例级别的自定义配置。
        
        返回：
            dict: 包含OHLCV列名映射的字典
            
        配置合并逻辑：
        1. 从vectorbt全局设置中获取默认列名配置
        2. 如果实例创建时提供了自定义列名，将其合并到默认配置中
        3. 实例级别的配置优先级高于全局配置
        
        默认列名配置：
        ```python
        {
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        ```
        
        使用示例：
        ```python
        # 查看当前列名映射
        ohlcv = df.vbt.ohlcv
        print("当前列名映射:", ohlcv.column_names)
        
        # 自定义列名映射
        custom_ohlcv = df.vbt.ohlcv(column_names={'open': 'o', 'close': 'c'})
        print("自定义列名映射:", custom_ohlcv.column_names)
        # 输出: {'open': 'o', 'high': 'high', 'low': 'low', 'close': 'c', 'volume': 'volume'}
        ```
        """
        from vectorbt._settings import settings  # 导入vectorbt全局设置
        ohlcv_cfg = settings['ohlcv']  # 获取OHLCV相关配置

        return merge_dicts(ohlcv_cfg['column_names'], self._column_names)  # 合并全局配置和实例配置

    def get_column(self, col_name: str) -> tp.Optional[tp.Series]:
        """
        根据列名获取对应的数据列
        
        该方法根据指定的列名（如'open'、'high'等）从DataFrame中获取对应的数据列。
        支持不区分大小写的列名匹配，并使用列名映射配置进行转换。
        
        参数：
            col_name: 要获取的列名，如'open'、'high'、'low'、'close'、'volume'
            
        返回：
            Optional[Series]: 如果找到对应列则返回Series，否则返回None
            
        查找逻辑：
        1. 获取DataFrame的所有列名并转换为小写
        2. 从列名映射配置中获取目标列名并转换为小写
        3. 在DataFrame的列名中查找匹配项
        4. 如果找到匹配项，返回对应的Series，否则返回None
        
        使用示例：
        ```python
        # 标准列名
        df = pd.DataFrame({
            'Open': [100, 105, 102],  # 大写O
            'HIGH': [110, 108, 107],  # 全大写
            'low': [98, 102, 100],    # 小写
            'Close': [105, 104, 106], # 首字母大写
            'Volume': [1000, 1200, 800]
        })
        
        ohlcv = df.vbt.ohlcv
        
        # 获取各列数据（不区分大小写）
        open_data = ohlcv.get_column('open')    # 匹配'Open'列
        high_data = ohlcv.get_column('high')    # 匹配'HIGH'列
        low_data = ohlcv.get_column('low')      # 匹配'low'列
        close_data = ohlcv.get_column('close')  # 匹配'Close'列
        volume_data = ohlcv.get_column('volume') # 匹配'Volume'列
        
        # 获取不存在的列
        missing_data = ohlcv.get_column('missing')  # 返回None
        
        # 自定义列名映射
        custom_df = pd.DataFrame({
            'o': [100, 105, 102],
            'h': [110, 108, 107], 
            'l': [98, 102, 100],
            'c': [105, 104, 106],
            'v': [1000, 1200, 800]
        })
        
        custom_ohlcv = custom_df.vbt.ohlcv(column_names={
            'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'
        })
        
        # 通过映射获取数据
        open_data = custom_ohlcv.get_column('open')  # 实际获取'o'列
        ```
        """
        df_column_names = self.obj.columns.str.lower().tolist()  # 获取DataFrame列名并转换为小写列表
        col_name = self.column_names[col_name].lower()  # 获取映射后的列名并转换为小写
        if col_name not in df_column_names:  # 如果列名不存在于DataFrame中
            return None  # 返回None表示未找到
        return self.obj.iloc[:, df_column_names.index(col_name)]  # 根据列名索引返回对应的Series

    @property
    def open(self) -> tp.Optional[tp.Series]:
        """
        获取开盘价数据列
        
        该属性返回包含开盘价数据的pandas Series。开盘价是指每个交易周期开始时的价格，
        是技术分析中的重要价格指标之一。
        
        返回：
            Optional[Series]: 包含开盘价数据的Series，如果找不到开盘价列则返回None
            
        数据特点：
        - 代表每个时间周期的第一个交易价格
        - 用于计算价格缺口、支撑阻力位等技术指标
        - 与收盘价比较可以判断价格趋势方向
        
        使用示例：
        ```python
        # 获取开盘价数据
        ohlcv = df.vbt.ohlcv
        open_prices = ohlcv.open
        
        # 开盘价的常用分析
        # 1. 价格缺口分析
        gap = open_prices - ohlcv.close.shift(1)  # 开盘价与前一周期收盘价的差异
        up_gap = gap > 0  # 向上缺口
        down_gap = gap < 0  # 向下缺口
        
        # 2. 开盘价与收盘价比较
        bullish_candle = ohlcv.close > open_prices  # 收盘价高于开盘价（看涨）
        bearish_candle = ohlcv.close < open_prices  # 收盘价低于开盘价（看跌）
        
        # 3. 开盘价统计分析
        avg_open = open_prices.mean()  # 平均开盘价
        open_volatility = open_prices.std()  # 开盘价波动率
        
        # 4. 开盘价趋势分析
        open_trend = open_prices.rolling(5).mean()  # 5期开盘价移动平均
        open_change = open_prices.pct_change()  # 开盘价变化率
        
        print(f"开盘价数据: {open_prices}")
        print(f"平均开盘价: {avg_open}")
        print(f"开盘价波动率: {open_volatility}")
        ```
        """
        return self.get_column('open')  # 调用get_column方法获取开盘价列

    @property
    def high(self) -> tp.Optional[tp.Series]:
        """
        获取最高价数据列
        
        该属性返回包含最高价数据的pandas Series。最高价是指每个交易周期内达到的最高价格，
        是衡量价格波动上限和市场强度的重要指标。
        
        返回：
            Optional[Series]: 包含最高价数据的Series，如果找不到最高价列则返回None
            
        数据特点：
        - 代表每个时间周期内的价格上限
        - 用于计算支撑阻力位、价格通道等技术指标
        - 与最低价结合可以计算价格区间和波动率
        
        使用示例：
        ```python
        # 获取最高价数据
        ohlcv = df.vbt.ohlcv
        high_prices = ohlcv.high
        
        # 最高价的常用分析
        # 1. 价格区间分析
        price_range = high_prices - ohlcv.low  # 价格区间（最高价-最低价）
        avg_range = price_range.mean()  # 平均价格区间
        
        # 2. 突破分析
        prev_high = high_prices.shift(1)  # 前一周期最高价
        breakout = high_prices > prev_high.rolling(20).max()  # 突破20期最高价
        
        # 3. 价格位置分析
        close_to_high = (ohlcv.close - ohlcv.low) / (high_prices - ohlcv.low)  # 收盘价在区间中的位置
        
        # 4. 最高价统计分析
        max_high = high_prices.max()  # 历史最高价
        high_percentile = high_prices.quantile(0.9)  # 90%分位数
        
        # 5. 最高价趋势分析
        high_ma = high_prices.rolling(10).mean()  # 10期最高价移动平均
        high_expansion = high_prices > high_ma * 1.02  # 价格扩张信号
        
        print(f"最高价数据: {high_prices}")
        print(f"平均价格区间: {avg_range}")
        print(f"历史最高价: {max_high}")
        ```
        """
        return self.get_column('high')  # 调用get_column方法获取最高价列

    @property
    def low(self) -> tp.Optional[tp.Series]:
        """
        获取最低价数据列
        
        该属性返回包含最低价数据的pandas Series。最低价是指每个交易周期内达到的最低价格，
        是衡量价格波动下限和市场支撑的重要指标。
        
        返回：
            Optional[Series]: 包含最低价数据的Series，如果找不到最低价列则返回None
            
        数据特点：
        - 代表每个时间周期内的价格下限
        - 用于计算支撑位、回撤幅度等技术指标
        - 与最高价结合可以计算价格波动范围
        
        使用示例：
        ```python
        # 获取最低价数据
        ohlcv = df.vbt.ohlcv
        low_prices = ohlcv.low
        
        # 最低价的常用分析
        # 1. 支撑位分析
        support_level = low_prices.rolling(20).min()  # 20期最低价作为支撑位
        near_support = (ohlcv.close - support_level) / support_level < 0.02  # 接近支撑位
        
        # 2. 回撤分析
        rolling_max = ohlcv.close.rolling(20).max()  # 20期最高收盘价
        drawdown = (low_prices - rolling_max) / rolling_max  # 最大回撤
        
        # 3. 价格位置分析
        close_to_low = (ohlcv.close - low_prices) / (ohlcv.high - low_prices)  # 收盘价相对位置
        
        # 4. 最低价统计分析
        min_low = low_prices.min()  # 历史最低价
        low_percentile = low_prices.quantile(0.1)  # 10%分位数
        
        # 5. 最低价趋势分析
        low_ma = low_prices.rolling(10).mean()  # 10期最低价移动平均
        low_support = low_prices > low_ma * 0.98  # 价格支撑信号
        
        # 6. 价格反弹分析
        prev_low = low_prices.shift(1)  # 前一周期最低价
        bounce = (ohlcv.close - low_prices) / (ohlcv.high - low_prices) > 0.8  # 强反弹信号
        
        print(f"最低价数据: {low_prices}")
        print(f"历史最低价: {min_low}")
        print(f"平均回撤: {drawdown.mean()}")
        ```
        """
        return self.get_column('low')  # 调用get_column方法获取最低价列

    @property
    def close(self) -> tp.Optional[tp.Series]:
        """
        获取收盘价数据列
        
        该属性返回包含收盘价数据的pandas Series。收盘价是指每个交易周期结束时的价格，
        是技术分析中最重要的价格指标，大多数技术指标都基于收盘价计算。
        
        返回：
            Optional[Series]: 包含收盘价数据的Series，如果找不到收盘价列则返回None
            
        数据特点：
        - 代表每个时间周期的最终成交价格
        - 反映市场对当前价值的最终共识
        - 用于计算绝大部分技术指标（MA、RSI、MACD等）
        - 是价格趋势分析的核心数据
        
        使用示例：
        ```python
        # 获取收盘价数据
        ohlcv = df.vbt.ohlcv
        close_prices = ohlcv.close
        
        # 收盘价的常用分析
        # 1. 基本统计分析
        mean_price = close_prices.mean()  # 平均收盘价
        price_std = close_prices.std()  # 价格标准差
        price_range = close_prices.max() - close_prices.min()  # 价格区间
        
        # 2. 收益率分析
        returns = close_prices.pct_change()  # 收益率
        daily_returns = returns.dropna()  # 去除空值
        cumulative_returns = (1 + returns).cumprod()  # 累积收益
        
        # 3. 技术指标计算
        sma_10 = close_prices.rolling(10).mean()  # 10日简单移动平均
        sma_20 = close_prices.rolling(20).mean()  # 20日简单移动平均
        ema_12 = close_prices.ewm(span=12).mean()  # 12日指数移动平均
        
        # 4. 趋势分析
        price_trend = close_prices > sma_20  # 价格趋势（高于20日均线为上涨）
        golden_cross = (sma_10 > sma_20) & (sma_10.shift(1) <= sma_20.shift(1))  # 黄金交叉
        
        # 5. 支撑阻力分析
        resistance = close_prices.rolling(20).max()  # 20期阻力位
        support = close_prices.rolling(20).min()  # 20期支撑位
        
        # 6. 波动率分析
        volatility = returns.rolling(20).std() * np.sqrt(252)  # 年化波动率
        
        print(f"收盘价数据: {close_prices}")
        print(f"平均收盘价: {mean_price}")
        print(f"价格波动率: {volatility.iloc[-1]}")
        print(f"最新收益率: {returns.iloc[-1]}")
        ```
        """
        return self.get_column('close')  # 调用get_column方法获取收盘价列

    @property
    def ohlc(self) -> tp.Optional[tp.Frame]:
        """
        获取OHLC数据组合
        
        该属性返回包含开盘价、最高价、最低价和收盘价的DataFrame。
        这是一个组合属性，将四个关键价格指标合并在一起，便于进行综合分析。
        
        返回：
            Optional[DataFrame]: 包含OHLC数据的DataFrame，如果没有找到任何价格列则返回None
            
        数据结构：
        返回的DataFrame包含以下列（如果存在）：
        - open: 开盘价列
        - high: 最高价列  
        - low: 最低价列
        - close: 收盘价列
        
        组合逻辑：
        1. 检查每个价格列是否存在
        2. 将存在的价格列按顺序合并
        3. 如果没有任何价格列，返回None
        
        使用示例：
        ```python
        # 获取OHLC数据
        ohlcv = df.vbt.ohlcv
        ohlc_data = ohlcv.ohlc
        
        # OHLC数据的常用分析
        # 1. K线形态分析
        body_size = abs(ohlc_data['close'] - ohlc_data['open'])  # K线实体大小
        upper_shadow = ohlc_data['high'] - ohlc_data[['open', 'close']].max(axis=1)  # 上影线
        lower_shadow = ohlc_data[['open', 'close']].min(axis=1) - ohlc_data['low']  # 下影线
        
        # 2. 价格区间分析
        price_range = ohlc_data['high'] - ohlc_data['low']  # 价格区间
        true_range = pd.concat([
            ohlc_data['high'] - ohlc_data['low'],
            abs(ohlc_data['high'] - ohlc_data['close'].shift(1)),
            abs(ohlc_data['low'] - ohlc_data['close'].shift(1))
        ], axis=1).max(axis=1)  # 真实区间
        
        # 3. 收盘价位置分析
        close_position = (ohlc_data['close'] - ohlc_data['low']) / (ohlc_data['high'] - ohlc_data['low'])
        
        # 4. K线类型识别
        bullish = ohlc_data['close'] > ohlc_data['open']  # 阳线
        bearish = ohlc_data['close'] < ohlc_data['open']  # 阴线
        doji = abs(ohlc_data['close'] - ohlc_data['open']) < price_range * 0.1  # 十字线
        
        # 5. 价格突破分析
        prev_high = ohlc_data['high'].shift(1)  # 前一周期最高价
        prev_low = ohlc_data['low'].shift(1)  # 前一周期最低价
        breakout_up = ohlc_data['close'] > prev_high  # 向上突破
        breakout_down = ohlc_data['close'] < prev_low  # 向下突破
        
        # 6. 技术指标计算
        # 威廉指标
        williams_r = (ohlc_data['high'].rolling(14).max() - ohlc_data['close']) / (
            ohlc_data['high'].rolling(14).max() - ohlc_data['low'].rolling(14).min()
        ) * -100
        
        print(f"OHLC数据形状: {ohlc_data.shape}")
        print(f"平均价格区间: {price_range.mean()}")
        print(f"阳线比例: {bullish.mean():.2%}")
        ```
        """
        to_concat = []  # 初始化要合并的列列表
        if self.open is not None:  # 如果开盘价列存在
            to_concat.append(self.open)  # 添加开盘价列
        if self.high is not None:  # 如果最高价列存在
            to_concat.append(self.high)  # 添加最高价列
        if self.low is not None:  # 如果最低价列存在
            to_concat.append(self.low)  # 添加最低价列
        if self.close is not None:  # 如果收盘价列存在
            to_concat.append(self.close)  # 添加收盘价列
        if len(to_concat) == 0:  # 如果没有任何价格列
            return None  # 返回None
        return pd.concat(to_concat, axis=1)  # 按列合并所有价格数据

    @property
    def volume(self) -> tp.Optional[tp.Series]:
        """
        获取成交量数据列
        
        该属性返回包含成交量数据的pandas Series。成交量是指每个交易周期内的交易数量，
        是衡量市场活跃度和价格变动可信度的重要指标。
        
        返回：
            Optional[Series]: 包含成交量数据的Series，如果找不到成交量列则返回None
            
        数据特点：
        - 代表每个时间周期内的交易活跃程度
        - 用于验证价格变动的可信度（放量上涨/下跌）
        - 是资金流入流出的重要指标
        - 用于计算量价关系和资金流向指标
        
        使用示例：
        ```python
        # 获取成交量数据
        ohlcv = df.vbt.ohlcv
        volume_data = ohlcv.volume
        
        # 成交量的常用分析
        # 1. 基本统计分析
        avg_volume = volume_data.mean()  # 平均成交量
        volume_std = volume_data.std()  # 成交量标准差
        max_volume = volume_data.max()  # 最大成交量
        
        # 2. 成交量相对分析
        volume_ratio = volume_data / avg_volume  # 成交量比率
        high_volume = volume_data > avg_volume * 1.5  # 放量标准
        low_volume = volume_data < avg_volume * 0.5  # 缩量标准
        
        # 3. 成交量趋势分析
        volume_ma = volume_data.rolling(10).mean()  # 10期成交量移动平均
        volume_increasing = volume_data > volume_ma  # 成交量上升
        
        # 4. 量价关系分析
        price_change = ohlcv.close.pct_change()  # 价格变化率
        volume_change = volume_data.pct_change()  # 成交量变化率
        
        # 量价配合分析
        bullish_volume = (price_change > 0) & (volume_data > avg_volume)  # 放量上涨
        bearish_volume = (price_change < 0) & (volume_data > avg_volume)  # 放量下跌
        
        # 5. 资金流向分析
        # 简单资金流向指标
        typical_price = (ohlcv.high + ohlcv.low + ohlcv.close) / 3  # 典型价格
        money_flow = typical_price * volume_data  # 资金流
        
        # 6. 成交量指标
        # 成交量加权平均价格（VWAP）
        vwap = (volume_data * ohlcv.close).rolling(20).sum() / volume_data.rolling(20).sum()
        
        # 7. 异常成交量检测
        volume_zscore = (volume_data - volume_data.rolling(20).mean()) / volume_data.rolling(20).std()
        abnormal_volume = abs(volume_zscore) > 2  # 异常成交量
        
        print(f"成交量数据: {volume_data}")
        print(f"平均成交量: {avg_volume:,.0f}")
        print(f"最大成交量: {max_volume:,.0f}")
        print(f"放量天数比例: {high_volume.mean():.2%}")
        ```
        """
        return self.get_column('volume')  # 调用get_column方法获取成交量列

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """
        获取统计指标的默认配置
        
        该属性返回OHLCV访问器统计指标计算的默认配置，合并了父类的通用配置
        和OHLCV特定的配置设置。
        
        返回：
            dict: 包含统计指标默认配置的字典
            
        配置合并逻辑：
        1. 获取父类GenericAccessor的默认统计配置
        2. 从vectorbt全局设置中获取OHLCV特定的统计配置
        3. 合并两个配置，OHLCV特定配置优先级更高
        
        配置内容：
        - 统计指标的计算方法
        - 聚合函数的设置
        - 显示格式和标题
        - 标签和分组信息
        
        使用示例：
        ```python
        # 查看默认统计配置
        ohlcv = df.vbt.ohlcv
        defaults = ohlcv.stats_defaults
        print("统计指标默认配置:", defaults)
        
        # 使用默认配置计算统计指标
        stats = ohlcv.stats()
        print("统计指标:", stats)
        
        # 自定义统计配置
        custom_stats = ohlcv.stats(
            settings=dict(
                tags=['ohlcv', 'custom'],  # 只计算特定标签的指标
                agg_func='median'  # 使用中位数聚合
            )
        )
        ```
        """
        from vectorbt._settings import settings  # 导入vectorbt全局设置
        ohlcv_stats_cfg = settings['ohlcv']['stats']  # 获取OHLCV统计配置

        return merge_dicts(  # 合并配置字典
            GenericAccessor.stats_defaults.__get__(self),  # 获取父类默认配置
            ohlcv_stats_cfg  # 合并OHLCV特定配置
        )

    _metrics: tp.ClassVar[Config] = Config(  # 定义统计指标配置类变量
        dict(
            start=dict(  # 起始时间指标
                title='Start',  # 指标标题
                calc_func=lambda self: self.wrapper.index[0],  # 计算函数：获取索引第一个元素
                agg_func=None,  # 聚合函数：不进行聚合
                tags='wrapper'  # 标签：属于wrapper类别
            ),
            end=dict(  # 结束时间指标
                title='End',  # 指标标题
                calc_func=lambda self: self.wrapper.index[-1],  # 计算函数：获取索引最后一个元素
                agg_func=None,  # 聚合函数：不进行聚合
                tags='wrapper'  # 标签：属于wrapper类别
            ),
            period=dict(  # 时间周期指标
                title='Period',  # 指标标题
                calc_func=lambda self: len(self.wrapper.index),  # 计算函数：获取索引长度
                apply_to_timedelta=True,  # 应用到时间差：将结果转换为时间差格式
                agg_func=None,  # 聚合函数：不进行聚合
                tags='wrapper'  # 标签：属于wrapper类别
            ),
            first_price=dict(  # 首个价格指标
                title='First Price',  # 指标标题
                calc_func=lambda ohlc: nb.bfill_1d_nb(ohlc.values.flatten())[0],  # 计算函数：获取首个非空价格
                resolve_ohlc=True,  # 解析OHLC：自动传入OHLC数据
                tags=['ohlcv', 'ohlc']  # 标签：属于ohlcv和ohlc类别
            ),
            lowest_price=dict(  # 最低价格指标
                title='Lowest Price',  # 指标标题
                calc_func=lambda ohlc: ohlc.values.min(),  # 计算函数：获取最小值
                resolve_ohlc=True,  # 解析OHLC：自动传入OHLC数据
                tags=['ohlcv', 'ohlc']  # 标签：属于ohlcv和ohlc类别
            ),
            highest_price=dict(  # 最高价格指标
                title='Highest Price',  # 指标标题
                calc_func=lambda ohlc: ohlc.values.max(),  # 计算函数：获取最大值
                resolve_ohlc=True,  # 解析OHLC：自动传入OHLC数据
                tags=['ohlcv', 'ohlc']  # 标签：属于ohlcv和ohlc类别
            ),
            last_price=dict(  # 最后价格指标
                title='Last Price',  # 指标标题
                calc_func=lambda ohlc: nb.ffill_1d_nb(ohlc.values.flatten())[-1],  # 计算函数：获取最后非空价格
                resolve_ohlc=True,  # 解析OHLC：自动传入OHLC数据
                tags=['ohlcv', 'ohlc']  # 标签：属于ohlcv和ohlc类别
            ),
            first_volume=dict(  # 首个成交量指标
                title='First Volume',  # 指标标题
                calc_func=lambda volume: nb.bfill_1d_nb(volume.values)[0],  # 计算函数：获取首个非空成交量
                resolve_volume=True,  # 解析成交量：自动传入成交量数据
                tags=['ohlcv', 'volume']  # 标签：属于ohlcv和volume类别
            ),
            lowest_volume=dict(  # 最低成交量指标
                title='Lowest Volume',  # 指标标题
                calc_func=lambda volume: volume.values.min(),  # 计算函数：获取最小值
                resolve_volume=True,  # 解析成交量：自动传入成交量数据
                tags=['ohlcv', 'volume']  # 标签：属于ohlcv和volume类别
            ),
            highest_volume=dict(  # 最高成交量指标
                title='Highest Volume',  # 指标标题
                calc_func=lambda volume: volume.values.max(),  # 计算函数：获取最大值
                resolve_volume=True,  # 解析成交量：自动传入成交量数据
                tags=['ohlcv', 'volume']  # 标签：属于ohlcv和volume类别
            ),
            last_volume=dict(  # 最后成交量指标
                title='Last Volume',  # 指标标题
                calc_func=lambda volume: nb.ffill_1d_nb(volume.values)[-1],  # 计算函数：获取最后非空成交量
                resolve_volume=True,  # 解析成交量：自动传入成交量数据
                tags=['ohlcv', 'volume']  # 标签：属于ohlcv和volume类别
            ),
        ),
        copy_kwargs=dict(copy_mode='deep')  # 配置复制参数：深度复制
    )

    @property
    def metrics(self) -> Config:
        """
        获取统计指标配置
        
        该属性返回OHLCV访问器的统计指标配置对象，包含了所有可用的统计指标定义。
        
        返回：
            Config: 包含统计指标配置的Config对象
            
        指标分类：
        1. **时间相关指标**：
           - start: 数据起始时间
           - end: 数据结束时间  
           - period: 数据时间周期
           
        2. **价格相关指标**：
           - first_price: 首个价格
           - lowest_price: 最低价格
           - highest_price: 最高价格
           - last_price: 最后价格
           
        3. **成交量相关指标**：
           - first_volume: 首个成交量
           - lowest_volume: 最低成交量
           - highest_volume: 最高成交量
           - last_volume: 最后成交量
           
        使用示例：
        ```python
        # 查看所有可用指标
        ohlcv = df.vbt.ohlcv
        metrics_config = ohlcv.metrics
        print("可用指标:", list(metrics_config.keys()))
        
        # 查看特定指标配置
        price_config = metrics_config['first_price']
        print("首个价格指标配置:", price_config)
        
        # 计算特定指标
        stats = ohlcv.stats(metrics=['first_price', 'last_price', 'highest_price'])
        print("价格指标:", stats)
        
        # 按标签筛选指标
        price_stats = ohlcv.stats(tags=['ohlc'])  # 只计算价格相关指标
        volume_stats = ohlcv.stats(tags=['volume'])  # 只计算成交量相关指标
        ```
        """
        return self._metrics  # 返回统计指标配置

    # ############# Plotting ############# #

    def plot(self,
             plot_type: tp.Union[None, str, tp.BaseTraceType] = None,  # 绘图类型：None、字符串或Plotly轨迹对象
             show_volume: tp.Optional[bool] = None,  # 是否显示成交量
             ohlc_kwargs: tp.KwargsLike = None,  # OHLC绘图参数
             volume_kwargs: tp.KwargsLike = None,  # 成交量绘图参数
             ohlc_add_trace_kwargs: tp.KwargsLike = None,  # OHLC轨迹添加参数
             volume_add_trace_kwargs: tp.KwargsLike = None,  # 成交量轨迹添加参数
             fig: tp.Optional[tp.BaseFigure] = None,  # 图形对象
             **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover  # 布局参数
        """
        绘制OHLCV数据图表
        
        该方法创建专业的OHLCV数据可视化图表，支持K线图、OHLC柱状图和成交量图的组合显示。
        这是金融数据分析中最常用的图表类型，能够直观地展示价格变动和成交量信息。
        
        参数：
            plot_type: 绘图类型选择
                - None: 使用默认类型（从配置获取）
                - 'OHLC': OHLC柱状图，显示开高低收四个价格
                - 'Candlestick': K线图（蜡烛图），更直观的价格表示
                - BaseTraceType: 自定义Plotly轨迹对象
                
            show_volume: 是否显示成交量
                - None: 自动决定（如果有成交量数据则显示）
                - True: 强制显示成交量子图
                - False: 不显示成交量
                
            ohlc_kwargs: OHLC/K线图的样式参数
                - 传递给Plotly的OHLC或Candlestick对象
                - 如颜色、线宽、透明度等
                
            volume_kwargs: 成交量柱状图的样式参数
                - 传递给Plotly的Bar对象
                - 如颜色、透明度、边框等
                
            ohlc_add_trace_kwargs: OHLC轨迹添加参数
                - 传递给fig.add_trace()的参数
                - 如子图位置、图例设置等
                
            volume_add_trace_kwargs: 成交量轨迹添加参数
                - 传递给fig.add_trace()的参数
                - 如子图位置、图例设置等
                
            fig: 现有图形对象
                - None: 创建新图形
                - Figure: 在现有图形上添加轨迹
                
            **layout_kwargs: 布局参数
                - 传递给fig.update_layout()的参数
                - 如标题、坐标轴设置、图例等
                
        返回：
            BaseFigure: 完成的Plotly图形对象
            
        图表特点：
        - 专业的金融图表样式
        - 支持价格和成交量的分层显示
        - 自动应用涨跌颜色主题
        - 支持时间序列缩放和交互
        - 可自定义所有视觉元素
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 基本K线图
        fig = ohlcv.plot()
        fig.show()
        
        # 指定图表类型
        fig = ohlcv.plot(plot_type='candlestick')  # K线图
        fig = ohlcv.plot(plot_type='ohlc')  # OHLC柱状图
        
        # 显示成交量
        fig = ohlcv.plot(show_volume=True)
        fig.show()
        
        # 自定义样式
        fig = ohlcv.plot(
            plot_type='candlestick',
            show_volume=True,
            ohlc_kwargs=dict(
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            volume_kwargs=dict(
                opacity=0.7,
                marker_line_width=1
            ),
            title='股票价格走势图',
            height=600
        )
        
        # 在现有图形上添加
        fig = make_subplots(rows=2, cols=1)
        ohlcv.plot(fig=fig, ohlc_add_trace_kwargs=dict(row=1, col=1))
        
        # 多资产对比
        fig = make_subplots(rows=2, cols=2)
        stock1.vbt.ohlcv.plot(fig=fig, ohlc_add_trace_kwargs=dict(row=1, col=1))
        stock2.vbt.ohlcv.plot(fig=fig, ohlc_add_trace_kwargs=dict(row=1, col=2))
        ```
        
        高级用法：
        ```python
        # 技术指标叠加
        fig = ohlcv.plot(plot_type='candlestick')
        
        # 添加移动平均线
        sma_20 = ohlcv.close.rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=sma_20.index, y=sma_20,
            name='SMA20', line=dict(color='blue')
        ))
        
        # 添加交易信号
        buy_signals = sma_20 > sma_20.shift(1)
        fig.add_trace(go.Scatter(
            x=buy_signals[buy_signals].index,
            y=ohlcv.close[buy_signals],
            mode='markers',
            marker=dict(color='green', size=8, symbol='triangle-up'),
            name='买入信号'
        ))
        
        fig.show()
        ```
        
        注意事项：
        - 确保数据包含必要的OHLC列
        - 成交量显示需要volume列存在
        - 图表会自动应用vectorbt的颜色主题
        - 支持时间序列索引的自动格式化
        
        使用方法：
        ```python
            >>> import vectorbt as vbt

            >>> vbt.YFData.download("BTC-USD").get().vbt.ohlcv.plot()
            ```

            ![](/assets/images/ohlcv_plot.svg)
        """
        from vectorbt._settings import settings  # 导入vectorbt全局设置
        plotting_cfg = settings['plotting']  # 获取绘图配置
        ohlcv_cfg = settings['ohlcv']  # 获取OHLCV配置

        if ohlc_kwargs is None:  # 如果OHLC参数为空
            ohlc_kwargs = {}  # 初始化为空字典
        if volume_kwargs is None:  # 如果成交量参数为空
            volume_kwargs = {}  # 初始化为空字典
        if ohlc_add_trace_kwargs is None:  # 如果OHLC轨迹参数为空
            ohlc_add_trace_kwargs = {}  # 初始化为空字典
        if volume_add_trace_kwargs is None:  # 如果成交量轨迹参数为空
            volume_add_trace_kwargs = {}  # 初始化为空字典
        if show_volume is None:  # 如果未指定是否显示成交量
            show_volume = self.volume is not None  # 根据是否有成交量数据自动决定
        if show_volume:  # 如果要显示成交量
            ohlc_add_trace_kwargs = merge_dicts(dict(row=1, col=1), ohlc_add_trace_kwargs)  # 设置OHLC在第一行
            volume_add_trace_kwargs = merge_dicts(dict(row=2, col=1), volume_add_trace_kwargs)  # 设置成交量在第二行

        # 设置图形对象
        if fig is None:  # 如果没有提供图形对象
            if show_volume:  # 如果要显示成交量
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0, row_heights=[0.7, 0.3])  # 创建双行子图
            else:  # 如果不显示成交量
                fig = make_figure()  # 创建单一图形
            fig.update_layout(  # 更新布局设置
                showlegend=True,  # 显示图例
                xaxis=dict(  # X轴设置
                    rangeslider_visible=False,  # 不显示范围滑块
                    showgrid=True  # 显示网格
                ),
                yaxis=dict(  # Y轴设置
                    showgrid=True  # 显示网格
                )
            )
            if show_volume:  # 如果显示成交量
                fig.update_layout(  # 更新布局设置
                    xaxis2=dict(  # 第二个X轴设置
                        showgrid=True  # 显示网格
                    ),
                    yaxis2=dict(  # 第二个Y轴设置
                        showgrid=True  # 显示网格
                    ),
                    bargap=0  # 柱状图间隙为0
                )
        fig.update_layout(**layout_kwargs)  # 应用用户提供的布局参数
        if plot_type is None:  # 如果没有指定绘图类型
            plot_type = ohlcv_cfg['plot_type']  # 从配置获取默认类型
        if isinstance(plot_type, str):  # 如果绘图类型是字符串
            if plot_type.lower() == 'ohlc':  # 如果是OHLC类型
                plot_type = 'OHLC'  # 标准化名称
                plot_obj = go.Ohlc  # 使用Plotly的OHLC对象
            elif plot_type.lower() == 'candlestick':  # 如果是K线图类型
                plot_type = 'Candlestick'  # 标准化名称
                plot_obj = go.Candlestick  # 使用Plotly的K线图对象
            else:  # 如果是其他类型
                raise ValueError("Plot type can be either 'OHLC' or 'Candlestick'")  # 抛出错误
        else:  # 如果绘图类型是对象
            plot_obj = plot_type  # 直接使用提供的对象
        ohlc = plot_obj(  # 创建OHLC/K线图轨迹
            x=self.wrapper.index,  # X轴数据：时间索引
            open=self.open,  # 开盘价数据
            high=self.high,  # 最高价数据
            low=self.low,  # 最低价数据
            close=self.close,  # 收盘价数据
            name=plot_type,  # 轨迹名称
            increasing=dict(  # 上涨时的样式
                line=dict(  # 线条样式
                    color=plotting_cfg['color_schema']['increasing']  # 上涨颜色
                )
            ),
            decreasing=dict(  # 下跌时的样式
                line=dict(  # 线条样式
                    color=plotting_cfg['color_schema']['decreasing']  # 下跌颜色
                )
            )
        )
        ohlc.update(**ohlc_kwargs)  # 应用用户提供的OHLC参数
        fig.add_trace(ohlc, **ohlc_add_trace_kwargs)  # 添加OHLC轨迹到图形

        if show_volume:  # 如果要显示成交量
            marker_colors = np.empty(self.volume.shape, dtype=object)  # 创建颜色数组
            marker_colors[(self.close.values - self.open.values) > 0] = plotting_cfg['color_schema']['increasing']  # 上涨时的颜色
            marker_colors[(self.close.values - self.open.values) == 0] = plotting_cfg['color_schema']['gray']  # 平盘时的颜色
            marker_colors[(self.close.values - self.open.values) < 0] = plotting_cfg['color_schema']['decreasing']  # 下跌时的颜色
            volume_bar = go.Bar(  # 创建成交量柱状图
                x=self.wrapper.index,  # X轴数据：时间索引
                y=self.volume,  # Y轴数据：成交量
                marker=dict(  # 标记样式
                    color=marker_colors,  # 颜色数组
                    line_width=0  # 线宽为0
                ),
                opacity=0.5,  # 透明度
                name='Volume'  # 轨迹名称
            )
            volume_bar.update(**volume_kwargs)  # 应用用户提供的成交量参数
            fig.add_trace(volume_bar, **volume_add_trace_kwargs)  # 添加成交量轨迹到图形

        return fig  # 返回完成的图形对象

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """
        获取绘图的默认配置
        
        该属性返回OHLCV访问器绘图功能的默认配置，合并了父类的通用绘图配置
        和OHLCV特定的绘图配置。
        
        返回：
            dict: 包含绘图默认配置的字典
            
        配置合并逻辑：
        1. 获取父类GenericAccessor的默认绘图配置
        2. 从vectorbt全局设置中获取OHLCV特定的绘图配置
        3. 合并两个配置，OHLCV特定配置优先级更高
        
        配置内容：
        - 默认图表类型设置
        - 颜色主题配置
        - 子图布局参数
        - 坐标轴设置
        - 图例和标题设置
        
        使用示例：
        ```python
        # 查看默认绘图配置
        ohlcv = df.vbt.ohlcv
        defaults = ohlcv.plots_defaults
        print("绘图默认配置:", defaults)
        
        # 使用默认配置绘图
        fig = ohlcv.plots()
        fig.show()
        
        # 自定义绘图配置
        custom_fig = ohlcv.plots(
            settings=dict(
                plot_type='candlestick',
                show_volume=True,
                height=600
            )
        )
        ```
        """
        from vectorbt._settings import settings  # 导入vectorbt全局设置
        ohlcv_plots_cfg = settings['ohlcv']['plots']  # 获取OHLCV绘图配置

        return merge_dicts(  # 合并配置字典
            GenericAccessor.plots_defaults.__get__(self),  # 获取父类默认配置
            ohlcv_plots_cfg  # 合并OHLCV特定配置
        )

    _subplots: tp.ClassVar[Config] = Config(  # 定义子图配置类变量
        dict(
            plot=dict(  # 主绘图配置
                title='OHLC',  # 子图标题
                xaxis_kwargs=dict(  # X轴参数
                    showgrid=True,  # 显示网格
                    rangeslider_visible=False  # 不显示范围滑块
                ),
                yaxis_kwargs=dict(  # Y轴参数
                    showgrid=True  # 显示网格
                ),
                check_is_not_grouped=True,  # 检查是否未分组
                plot_func='plot',  # 绘图函数名称
                show_volume=False,  # 不显示成交量
                tags='ohlcv'  # 标签
            )
        ),
        copy_kwargs=dict(copy_mode='deep')  # 配置复制参数：深度复制
    )

    @property
    def subplots(self) -> Config:
        """
        获取子图配置
        
        该属性返回OHLCV访问器的子图配置对象，定义了如何创建和组织多个子图。
        
        返回：
            Config: 包含子图配置的Config对象
            
        子图配置：
        - plot: 主OHLC绘图配置
          - title: 子图标题
          - xaxis_kwargs: X轴配置参数
          - yaxis_kwargs: Y轴配置参数
          - plot_func: 使用的绘图函数
          - show_volume: 是否显示成交量
          - tags: 标签分类
        
        使用示例：
        ```python
        # 查看子图配置
        ohlcv = df.vbt.ohlcv
        subplots_config = ohlcv.subplots
        print("子图配置:", subplots_config)
        
        # 使用子图配置创建图表
        fig = ohlcv.plots()
        fig.show()
        
        # 自定义子图配置
        custom_fig = ohlcv.plots(
            subplots='plot',  # 使用特定子图配置
            settings=dict(
                plot=dict(
                    title='自定义OHLC图表',
                    show_volume=True
                )
            )
        )
        ```
        """
        return self._subplots  # 返回子图配置


OHLCVDFAccessor.override_metrics_doc(__pdoc__)  # 重写指标文档
OHLCVDFAccessor.override_subplots_doc(__pdoc__)  # 重写子图文档
