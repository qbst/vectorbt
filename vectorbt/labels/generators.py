# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT LABELS MODULE: 机器学习标签生成器 (Label Generators)
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于机器学习标签生成的核心模块。该模块提供了一套完整的
标签生成器，用于创建监督学习模型的目标变量（labels），特别适用于金融时间序列预测任务。

核心设计理念：
1. **前瞻性分析工具**：所有指标都使用未来数据，专门用于机器学习训练而非实时交易
2. **工厂模式架构**：使用IndicatorFactory统一构建各种标签生成器，确保API一致性
3. **高性能计算**：底层使用Numba JIT编译的函数，实现接近C语言的执行速度
4. **模块化设计**：每个标签生成器都是独立的类，便于组合和扩展

⚠️ 重要警告：
本模块中的所有指标都包含前瞻性偏差（Look-ahead Bias），这些指标使用了未来的数据信息：
- 仅适用于离线训练机器学习模型
- 绝对不能用于实时交易系统
- 用于回测时需要确保没有引入未来信息泄露

标签生成器分类：
【前瞻性指标 (Look-ahead Indicators)】
- FMEAN: 未来均值指标，计算未来时间窗口内的平均价格
- FSTD: 未来标准差指标，计算未来时间窗口内的价格波动率
- FMIN: 未来最小值指标，计算未来时间窗口内的最低价格
- FMAX: 未来最大值指标，计算未来时间窗口内的最高价格

【标签生成器 (Label Generators)】
- FIXLB: 固定时间点标签生成器，计算固定时间后的价格变化
- MEANLB: 均值标签生成器，计算与未来均值的价格变化
- LEXLB: 局部极值标签生成器，识别价格的局部极值点
- TRENDLB: 趋势标签生成器，生成多种模式的趋势标签
- BOLB: 突破标签生成器，检测价格突破事件

应用场景：
- 股价预测模型：预测股价上涨/下跌的概率
- 趋势识别系统：识别价格趋势的方向和强度
- 波动率预测：预测未来一段时间的价格波动率
- 极值检测：识别价格的转折点和突破点
- 量化选股模型：基于未来表现对股票进行排序

技术特点：
- 使用vectorbt.indicators.factory.IndicatorFactory构建
- 集成绘图功能，支持标签的可视化分析
- 支持多种参数配置，适应不同的市场环境
- 与vectorbt生态系统无缝集成

使用示例：
```python
import pandas as pd
import numpy as np
import vectorbt as vbt

# 创建示例价格数据
np.random.seed(42)
prices = pd.Series(np.random.randn(100).cumsum() + 100, 
                  index=pd.date_range('2023-01-01', periods=100))

# 1. 生成未来5天的平均价格标签
future_mean = vbt.FMEAN.run(prices, window=5)
print("未来5天平均价格:", future_mean.fmean.tail())

# 2. 生成固定3天后的价格变化标签
fixed_labels = vbt.FIXLB.run(prices, n=3)
print("3天后价格变化:", fixed_labels.labels.tail())

# 3. 生成趋势标签
trend_labels = vbt.TRENDLB.run(prices, pos_th=0.05, neg_th=0.05)
print("趋势标签:", trend_labels.labels.tail())

# 4. 绘制标签热力图
fixed_labels.plot(title="价格变化标签热力图")
```

与传统技术分析的区别：
- 传统技术分析：基于历史数据，用于实时交易决策
- 标签生成器：基于未来数据，用于训练预测模型

该模块是vectorbt框架中连接传统技术分析与现代机器学习的重要桥梁，为量化交易中的
预测模型训练提供了工业级的标签生成解决方案。
================================================================================

基于vectorbt.labels.nb中Numba编译函数的基础指标和标签生成器。

您可以通过vbt.*或vbt.labels.*访问所有指标。
"""

# 导入必要的模块
from vectorbt import _typing as tp  # 导入vectorbt的类型定义模块
from vectorbt.indicators.configs import flex_elem_param_config  # 导入灵活元素参数配置
from vectorbt.indicators.factory import IndicatorFactory  # 导入指标工厂类
from vectorbt.labels import nb  # 导入标签模块的Numba编译函数
from vectorbt.labels.enums import TrendMode  # 导入趋势模式枚举

# ############# 前瞻性指标 (Look-ahead Indicators) ############# #

# 未来均值指标 (Future Mean Indicator)
FMEAN = IndicatorFactory(
    class_name='FMEAN',  # 指标类名
    module_name=__name__,  # 所属模块名
    input_names=['close'],  # 输入参数名：收盘价
    param_names=['window', 'ewm'],  # 参数名：窗口大小、是否使用指数加权移动平均
    output_names=['fmean']  # 输出名：未来均值
).from_apply_func(
    nb.future_mean_apply_nb,  # 底层Numba编译函数
    kwargs_to_args=['wait', 'adjust'],  # 关键字参数转换为位置参数
    ewm=False,  # 默认不使用指数加权移动平均
    wait=1,  # 默认等待1个时间点（避免使用当前值）
    adjust=False  # 默认不使用权重调整
)

FMEAN.__doc__ = """
未来均值指标 - 基于未来数据的移动平均线

该指标计算每个时间点未来指定窗口内的平均价格，用于生成机器学习模型的回归标签。
⚠️ 包含前瞻性偏差，仅用于模型训练，不能用于实时交易。

核心功能：
- 计算未来时间窗口内的平均价格
- 支持简单移动平均和指数加权移动平均
- 提供价格目标预测的基准值
- 生成连续型回归标签

参数说明：
- window (int): 未来统计窗口大小
- ewm (bool): 是否使用指数加权移动平均
- wait (int): 等待期数，默认为1
- adjust (bool): 指数加权的权重调整参数

使用示例：
```python
import pandas as pd
import vectorbt as vbt

# 创建示例价格数据
prices = pd.Series([100, 105, 98, 110, 95, 108, 103])

# 计算未来3天的简单移动平均
future_ma = vbt.FMEAN.run(prices, window=3, ewm=False)
print("未来3天平均价格:", future_ma.fmean)

# 计算未来5天的指数加权移动平均
future_ewma = vbt.FMEAN.run(prices, window=5, ewm=True)
print("未来5天指数加权平均:", future_ewma.fmean)

# 绘制价格与未来均值的对比图
future_ma.plot(title="价格与未来均值对比")
```

应用场景：
- 股价预测模型的回归标签
- 趋势强度评估
- 价格目标设置
- 均值回归策略的基准计算
"""

# 未来标准差指标 (Future Standard Deviation Indicator)
FSTD = IndicatorFactory(
    class_name='FSTD',  # 指标类名
    module_name=__name__,  # 所属模块名
    input_names=['close'],  # 输入参数名：收盘价
    param_names=['window', 'ewm'],  # 参数名：窗口大小、是否使用指数加权
    output_names=['fstd']  # 输出名：未来标准差
).from_apply_func(
    nb.future_std_apply_nb,  # 底层Numba编译函数
    kwargs_to_args=['wait', 'adjust', 'ddof'],  # 关键字参数转换为位置参数
    ewm=False,  # 默认不使用指数加权
    wait=1,  # 默认等待1个时间点
    adjust=False,  # 默认不使用权重调整
    ddof=0  # 默认使用总体标准差公式
)

FSTD.__doc__ = """
未来标准差指标 - 基于未来数据的波动率指标

该指标计算每个时间点未来指定窗口内的价格标准差，用于度量未来价格的波动性。
⚠️ 包含前瞻性偏差，仅用于模型训练，不能用于实时交易。

核心功能：
- 计算未来时间窗口内的价格波动率
- 支持简单标准差和指数加权标准差
- 提供风险预测的基准值
- 生成波动率预测标签

参数说明：
- window (int): 未来统计窗口大小
- ewm (bool): 是否使用指数加权标准差
- wait (int): 等待期数，默认为1
- adjust (bool): 指数加权的权重调整参数
- ddof (int): 自由度增量，0为总体标准差，1为样本标准差

使用示例：
```python
import pandas as pd
import vectorbt as vbt

# 创建示例价格数据
prices = pd.Series([100, 105, 98, 110, 95, 108, 103])

# 计算未来5天的标准差
future_std = vbt.FSTD.run(prices, window=5, ewm=False)
print("未来5天波动率:", future_std.fstd)

# 计算未来3天的指数加权标准差
future_ewm_std = vbt.FSTD.run(prices, window=3, ewm=True)
print("未来3天指数加权波动率:", future_ewm_std.fstd)

# 绘制波动率预测图
future_std.plot(title="未来波动率预测")
```

应用场景：
- 波动率预测模型
- 风险评估标签生成
- 期权定价模型的波动率输入
- 动态风险管理系统
"""

# 未来最小值指标 (Future Minimum Indicator)
FMIN = IndicatorFactory(
    class_name='FMIN',  # 指标类名
    module_name=__name__,  # 所属模块名
    input_names=['close'],  # 输入参数名：收盘价
    param_names=['window'],  # 参数名：窗口大小
    output_names=['fmin']  # 输出名：未来最小值
).from_apply_func(
    nb.future_min_apply_nb,  # 底层Numba编译函数
    kwargs_to_args=['wait'],  # 关键字参数转换为位置参数
    wait=1  # 默认等待1个时间点
)

FMIN.__doc__ = """
未来最小值指标 - 基于未来数据的支撑位指标

该指标计算每个时间点未来指定窗口内的最低价格，用于生成支撑位预测标签。
⚠️ 包含前瞻性偏差，仅用于模型训练，不能用于实时交易。

核心功能：
- 计算未来时间窗口内的最低价格
- 提供支撑位预测的基准值
- 生成价格底部预测标签
- 用于风险评估和止损位设置

参数说明：
- window (int): 未来统计窗口大小
- wait (int): 等待期数，默认为1

使用示例：
```python
import pandas as pd
import vectorbt as vbt

# 创建示例价格数据
prices = pd.Series([100, 105, 98, 110, 95, 108, 103])

# 计算未来5天的最低价格
future_min = vbt.FMIN.run(prices, window=5)
print("未来5天最低价:", future_min.fmin)

# 计算未来3天的最低价格
future_min_3d = vbt.FMIN.run(prices, window=3)
print("未来3天最低价:", future_min_3d.fmin)

# 绘制价格与未来最低价的对比图
future_min.plot(title="价格与未来支撑位对比")
```

应用场景：
- 支撑位预测模型
- 止损位设置参考
- 风险管理系统
- 价格底部预测
"""

# 未来最大值指标 (Future Maximum Indicator)
FMAX = IndicatorFactory(
    class_name='FMAX',  # 指标类名
    module_name=__name__,  # 所属模块名
    input_names=['close'],  # 输入参数名：收盘价
    param_names=['window'],  # 参数名：窗口大小
    output_names=['fmax']  # 输出名：未来最大值
).from_apply_func(
    nb.future_max_apply_nb,  # 底层Numba编译函数
    kwargs_to_args=['wait'],  # 关键字参数转换为位置参数
    wait=1  # 默认等待1个时间点
)

FMAX.__doc__ = """
未来最大值指标 - 基于未来数据的阻力位指标

该指标计算每个时间点未来指定窗口内的最高价格，用于生成阻力位预测标签。
⚠️ 包含前瞻性偏差，仅用于模型训练，不能用于实时交易。

核心功能：
- 计算未来时间窗口内的最高价格
- 提供阻力位预测的基准值
- 生成价格顶部预测标签
- 用于收益目标设置和止盈位参考

参数说明：
- window (int): 未来统计窗口大小
- wait (int): 等待期数，默认为1

使用示例：
```python
import pandas as pd
import vectorbt as vbt

# 创建示例价格数据
prices = pd.Series([100, 105, 98, 110, 95, 108, 103])

# 计算未来5天的最高价格
future_max = vbt.FMAX.run(prices, window=5)
print("未来5天最高价:", future_max.fmax)

# 计算未来3天的最高价格
future_max_3d = vbt.FMAX.run(prices, window=3)
print("未来3天最高价:", future_max_3d.fmax)

# 绘制价格与未来最高价的对比图
future_max.plot(title="价格与未来阻力位对比")
```

应用场景：
- 阻力位预测模型
- 收益目标设置
- 止盈位参考
- 价格顶部预测
"""


# ############# 标签生成器 (Label Generators) ############# #


def _plot(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:  # pragma: no cover
    """
    绘制标签热力图的通用绘图函数
    
    该函数为所有标签生成器提供统一的绘图接口，在价格线图上叠加标签的热力图显示。
    
    参数说明：
        column: 可选的列标签，用于选择特定的数据列
        **kwargs: 传递给overlay_with_heatmap方法的其他参数
    
    返回值：
        Plotly图表对象，包含价格线图和标签热力图
    
    使用场景：
        - 可视化标签的分布和模式
        - 分析标签与价格的关系
        - 调试和验证标签生成器的输出
    
    示例：
        ```python
        # 生成标签后绘制热力图
        labels = vbt.FIXLB.run(prices, n=5)
        fig = labels.plot(title="5天价格变化标签")
        fig.show()
        ```
    """
    # 选择指定的列或使用默认列
    self_col = self.select_one(column=column, group_by=False)

    # 在价格线图上叠加标签热力图
    # close.rename('close'): 将价格数据重命名为'close'
    # labels.rename('labels'): 将标签数据重命名为'labels'
    # overlay_with_heatmap: 在线图上叠加热力图
    return self_col.close.rename('close').vbt.overlay_with_heatmap(self_col.labels.rename('labels'), **kwargs)


# 固定时间点标签生成器 (Fixed Time Point Label Generator)
FIXLB = IndicatorFactory(
    class_name='FIXLB',  # 指标类名
    module_name=__name__,  # 所属模块名
    input_names=['close'],  # 输入参数名：收盘价
    param_names=['n'],  # 参数名：固定时间步数
    output_names=['labels']  # 输出名：标签
).from_apply_func(
    nb.fixed_labels_apply_nb  # 底层Numba编译函数
)


class _FIXLB(FIXLB):
    """
    固定时间点标签生成器
    
    该生成器计算当前价格与未来固定时间点价格的百分比变化，是最简单直接的标签生成方法。
    广泛用于股价涨跌预测模型，提供明确的时间边界和计算逻辑。
    
    核心功能：
    - 计算固定时间后的价格变化百分比
    - 生成回归模型的目标变量
    - 提供直观的收益率预测标签
    - 支持多种时间周期的标签生成
    
    计算公式：
    labels = (price_t+n - price_t) / price_t
    
    其中：
    - price_t: 当前时间点的价格
    - price_t+n: 未来第n个时间点的价格
    - n: 固定的时间步数
    
    标签含义：
    - 正值: 价格上涨的百分比
    - 负值: 价格下跌的百分比
    - 0: 价格无变化
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建示例价格数据
    prices = pd.Series([100, 105, 98, 110, 95, 108, 103])
    
    # 生成3天后的价格变化标签
    labels_3d = vbt.FIXLB.run(prices, n=3)
    print("3天后价格变化:", labels_3d.labels)
    
    # 生成5天后的价格变化标签
    labels_5d = vbt.FIXLB.run(prices, n=5)
    print("5天后价格变化:", labels_5d.labels)
    
    # 绘制标签热力图
    labels_3d.plot(title="3天价格变化标签")
    
    # 分析标签分布
    print("标签统计:", labels_3d.labels.describe())
    ```
    
    应用场景：
    - 股价涨跌预测模型的目标变量
    - 短期收益率预测
    - 分类模型的回归标签生成
    - 策略收益率计算
    - 基准收益率对比
    
    优势：
    - 计算简单直观
    - 标签含义明确
    - 时间边界清晰
    - 适用于各种机器学习模型
    
    注意事项：
    - 包含前瞻性偏差，仅用于模型训练
    - 末尾n个时间点的标签为NaN
    - 需要根据预测周期选择合适的n值
    """

    plot = _plot  # 继承通用绘图函数


# 将文档字符串和绘图方法添加到FIXLB类
setattr(FIXLB, '__doc__', _FIXLB.__doc__)
setattr(FIXLB, 'plot', _FIXLB.plot)

# 均值标签生成器 (Mean Label Generator)
MEANLB = IndicatorFactory(
    class_name='MEANLB',  # 指标类名
    module_name=__name__,  # 所属模块名
    input_names=['close'],  # 输入参数名：收盘价
    param_names=['window', 'ewm'],  # 参数名：窗口大小、是否使用指数加权
    output_names=['labels']  # 输出名：标签
).from_apply_func(
    nb.mean_labels_apply_nb,  # 底层Numba编译函数
    kwargs_to_args=['wait', 'adjust'],  # 关键字参数转换为位置参数
    ewm=False,  # 默认不使用指数加权
    wait=1,  # 默认等待1个时间点
    adjust=False  # 默认不使用权重调整
)


class _MEANLB(MEANLB):
    """
    均值标签生成器
    
    该生成器计算当前价格与未来一段时间内平均价格的百分比变化，相比固定时间点标签
    更加平滑和稳定，有助于减少标签的噪声。
    
    核心功能：
    - 计算与未来均值的价格变化百分比
    - 生成平滑的回归标签
    - 减少极端值的影响
    - 提供稳定的训练目标
    
    计算公式：
    labels = (future_mean - current_price) / current_price
    
    其中：
    - current_price: 当前时间点的价格
    - future_mean: 未来window个时间点的平均价格
    - window: 未来统计窗口大小
    
    标签含义：
    - 正值: 未来平均价格高于当前价格
    - 负值: 未来平均价格低于当前价格
    - 0: 当前价格等于未来平均价格
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建示例价格数据
    prices = pd.Series([100, 105, 98, 110, 95, 108, 103])
    
    # 生成未来5天均值标签
    labels_mean = vbt.MEANLB.run(prices, window=5, ewm=False)
    print("未来5天均值标签:", labels_mean.labels)
    
    # 生成未来3天指数加权均值标签
    labels_ewm = vbt.MEANLB.run(prices, window=3, ewm=True)
    print("未来3天指数加权均值标签:", labels_ewm.labels)
    
    # 绘制标签热力图
    labels_mean.plot(title="未来均值标签")
    
    # 对比固定时间点标签和均值标签
    fixed_labels = vbt.FIXLB.run(prices, n=5)
    print("固定标签标准差:", fixed_labels.labels.std())
    print("均值标签标准差:", labels_mean.labels.std())
    ```
    
    应用场景：
    - 平滑的价格趋势预测标签
    - 减少噪声的回归目标
    - 中长期趋势方向预测
    - 均值回归策略的基准
    
    优势：
    - 相比固定时间点标签更加平滑
    - 减少了极端值的影响
    - 提供更稳定的训练目标
    - 适用于趋势跟踪策略
    
    参数说明：
    - window: 未来统计窗口大小
    - ewm: 是否使用指数加权移动平均
    - wait: 等待期数，避免使用当前时间点
    - adjust: 指数加权的权重调整参数
    """

    plot = _plot  # 继承通用绘图函数


# 将文档字符串和绘图方法添加到MEANLB类
setattr(MEANLB, '__doc__', _MEANLB.__doc__)
setattr(MEANLB, 'plot', _MEANLB.plot)

# 局部极值标签生成器 (Local Extrema Label Generator)
LEXLB = IndicatorFactory(
    class_name='LEXLB',  # 指标类名
    module_name=__name__,  # 所属模块名
    input_names=['close'],  # 输入参数名：收盘价
    param_names=['pos_th', 'neg_th'],  # 参数名：正向阈值、负向阈值
    output_names=['labels']  # 输出名：标签
).from_apply_func(
    nb.local_extrema_apply_nb,  # 底层Numba编译函数
    param_settings=dict(
        pos_th=flex_elem_param_config,  # 正向阈值使用灵活元素参数配置
        neg_th=flex_elem_param_config   # 负向阈值使用灵活元素参数配置
    ),
    pass_flex_2d=True  # 传递2维灵活索引参数
)


class _LEXLB(LEXLB):
    """
    局部极值标签生成器
    
    该生成器识别价格序列中的局部极值点（波峰和波谷），基于动态阈值检测算法，
    只有当价格变化超过设定的阈值时才认为是有效的极值点。
    
    核心功能：
    - 识别价格序列中的局部极值点
    - 基于动态阈值过滤噪声
    - 生成离散的极值标签
    - 提供趋势转折点检测
    
    算法逻辑：
    1. 从价格序列开始遍历，寻找第一个满足阈值条件的转折点
    2. 一旦确定了方向（上涨或下跌），就持续更新当前的极值点
    3. 当价格反向变化超过阈值时，确认前一个极值点并改变方向
    4. 重复此过程直到序列结束
    
    标签含义：
    - 1: 波峰（局部最高点）
    - -1: 波谷（局部最低点）
    - 0: 非极值点
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建示例价格数据
    prices = pd.Series([100, 105, 98, 110, 95, 108, 103, 95, 112, 98])
    
    # 生成局部极值标签（5%阈值）
    extrema_labels = vbt.LEXLB.run(prices, pos_th=0.05, neg_th=0.05)
    print("局部极值标签:", extrema_labels.labels)
    
    # 生成局部极值标签（3%阈值）
    extrema_labels_3pct = vbt.LEXLB.run(prices, pos_th=0.03, neg_th=0.03)
    print("局部极值标签(3%):", extrema_labels_3pct.labels)
    
    # 绘制标签热力图
    extrema_labels.plot(title="局部极值标签")
    
    # 找到所有波峰和波谷
    peaks = extrema_labels.labels[extrema_labels.labels == 1]
    troughs = extrema_labels.labels[extrema_labels.labels == -1]
    print(f"波峰数量: {len(peaks)}, 波谷数量: {len(troughs)}")
    ```
    
    应用场景：
    - 技术分析中的波峰波谷识别
    - 趋势转折点检测
    - 支撑阻力位自动识别
    - 波动交易策略的信号生成
    - 市场结构分析
    
    优势：
    - 动态阈值检测，避免噪声干扰
    - 实时更新极值点位置
    - 双向检测上涨和下跌极值
    - 适用于各种市场条件
    
    参数说明：
    - pos_th: 正向（上涨）阈值，范围(0, 1]
    - neg_th: 负向（下跌）阈值，范围(0, 1]
    - 支持标量、1维数组或2维数组格式
    
    注意事项：
    - 阈值设置需要平衡敏感性和稳定性
    - 过小的阈值可能产生过多的虚假信号
    - 过大的阈值可能遗漏重要的转折点
    - 该算法具有一定的滞后性
    """

    plot = _plot  # 继承通用绘图函数


# 将文档字符串和绘图方法添加到LEXLB类
setattr(LEXLB, '__doc__', _LEXLB.__doc__)
setattr(LEXLB, 'plot', _LEXLB.plot)

# 趋势标签生成器 (Trend Label Generator)
TRENDLB = IndicatorFactory(
    class_name='TRENDLB',  # 指标类名
    module_name=__name__,  # 所属模块名
    input_names=['close'],  # 输入参数名：收盘价
    param_names=['pos_th', 'neg_th', 'mode'],  # 参数名：正向阈值、负向阈值、模式
    output_names=['labels']  # 输出名：标签
).from_apply_func(
    nb.trend_labels_apply_nb,  # 底层Numba编译函数
    param_settings=dict(
        pos_th=flex_elem_param_config,  # 正向阈值使用灵活元素参数配置
        neg_th=flex_elem_param_config,  # 负向阈值使用灵活元素参数配置
        mode=dict(dtype=TrendMode)  # 模式参数使用TrendMode枚举类型
    ),
    pass_flex_2d=True,  # 传递2维灵活索引参数
    mode=TrendMode.Binary  # 默认使用二进制模式
)


class _TRENDLB(TRENDLB):
    """
    趋势标签生成器
    
    该生成器是最强大和灵活的标签生成器，支持多种趋势标签生成模式。它首先识别
    局部极值点，然后根据指定的模式生成相应的趋势标签。
    
    核心功能：
    - 支持多种趋势标签生成模式
    - 基于局部极值点进行标签生成
    - 提供二进制和连续型标签
    - 支持百分比变化标签
    
    支持的模式：
    1. Binary (二进制模式): 
       - 0: 下跌趋势（从波峰到波谷）
       - 1: 上涨趋势（从波谷到波峰）
    
    2. BinaryCont (连续模式):
       - 0-1连续值，表示在趋势区间内的相对位置
       - 0: 接近最高点（将会下跌）
       - 1: 接近最低点（将会上涨）
    
    3. BinaryContSat (饱和连续模式):
       - 带饱和处理的连续标签
       - 当价格变化超过阈值时，标签饱和为0或1
    
    4. PctChange (百分比变化模式):
       - 计算到下一个极值点的百分比变化
       - 使用当前价格作为分母
    
    5. PctChangeNorm (标准化百分比变化模式):
       - 计算到下一个极值点的标准化百分比变化
       - 使用未来价格作为分母，提供更好的对称性
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建示例价格数据
    prices = pd.Series([100, 105, 98, 110, 95, 108, 103, 95, 112, 98])
    
    # 生成二进制趋势标签
    binary_labels = vbt.TRENDLB.run(
        prices, 
        pos_th=0.05, 
        neg_th=0.05, 
        mode=vbt.labels.enums.TrendMode.Binary
    )
    print("二进制趋势标签:", binary_labels.labels)
    
    # 生成连续趋势标签
    cont_labels = vbt.TRENDLB.run(
        prices, 
        pos_th=0.05, 
        neg_th=0.05, 
        mode=vbt.labels.enums.TrendMode.BinaryCont
    )
    print("连续趋势标签:", cont_labels.labels)
    
    # 生成百分比变化标签
    pct_labels = vbt.TRENDLB.run(
        prices, 
        pos_th=0.05, 
        neg_th=0.05, 
        mode=vbt.labels.enums.TrendMode.PctChange
    )
    print("百分比变化标签:", pct_labels.labels)
    
    # 绘制不同模式的标签对比
    binary_labels.plot(title="二进制趋势标签")
    cont_labels.plot(title="连续趋势标签")
    pct_labels.plot(title="百分比变化标签")
    ```
    
    应用场景：
    - 多种机器学习模型的标签生成
    - 趋势跟踪策略的信号生成
    - 分类和回归任务的目标变量
    - 复杂的市场状态分析
    
    优势：
    - 支持多种标签生成模式
    - 统一的接口，便于使用
    - 基于局部极值点，标签含义明确
    - 适用于各种预测任务
    
    参数说明：
    - pos_th: 正向阈值，用于识别波峰
    - neg_th: 负向阈值，用于识别波谷
    - mode: 趋势模式，决定标签的生成方式
    """

    plot = _plot  # 继承通用绘图函数


# 将文档字符串和绘图方法添加到TRENDLB类
setattr(TRENDLB, '__doc__', _TRENDLB.__doc__)
setattr(TRENDLB, 'plot', _TRENDLB.plot)

# 突破标签生成器 (Breakout Label Generator)
BOLB = IndicatorFactory(
    class_name='BOLB',  # 指标类名（Breakout Label）
    module_name=__name__,  # 所属模块名
    input_names=['close'],  # 输入参数名：收盘价
    param_names=['window', 'pos_th', 'neg_th'],  # 参数名：窗口大小、正向阈值、负向阈值
    output_names=['labels']  # 输出名：标签
).from_apply_func(
    nb.breakout_labels_nb,  # 底层Numba编译函数
    param_settings=dict(
        pos_th=flex_elem_param_config,  # 正向阈值使用灵活元素参数配置
        neg_th=flex_elem_param_config   # 负向阈值使用灵活元素参数配置
    ),
    pass_flex_2d=True,  # 传递2维灵活索引参数
    kwargs_to_args=['wait'],  # 关键字参数转换为位置参数
    pos_th=0.,  # 默认正向阈值为0
    neg_th=0.,  # 默认负向阈值为0
    wait=1  # 默认等待1个时间点
)


class _BOLB(BOLB):
    """
    突破标签生成器
    
    该生成器检测在指定的未来时间窗口内，价格是否突破了预设的阈值。这是一种
    常用的技术分析方法，用于识别价格的显著变化和动量信号。
    
    核心功能：
    - 检测未来时间窗口内的价格突破
    - 基于动态阈值的突破识别
    - 生成动量交易信号标签
    - 支持双向突破检测
    
    突破检测机制：
    - 正向突破：价格上涨超过pos_th阈值
    - 负向突破：价格下跌超过neg_th阈值
    - 首次突破原则：一旦检测到突破，立即标记并停止搜索
    - 无突破：在窗口内没有超过阈值的价格变化
    
    标签含义：
    - 1: 正向突破（上涨突破）
    - -1: 负向突破（下跌突破）
    - 0: 无突破
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建示例价格数据
    prices = pd.Series([100, 105, 98, 110, 95, 108, 103, 95, 112, 98])
    
    # 生成突破标签（未来5天窗口，5%阈值）
    breakout_labels = vbt.BOLB.run(
        prices, 
        window=5, 
        pos_th=0.05, 
        neg_th=0.05
    )
    print("突破标签:", breakout_labels.labels)
    
    # 生成突破标签（未来3天窗口，3%阈值）
    breakout_labels_3d = vbt.BOLB.run(
        prices, 
        window=3, 
        pos_th=0.03, 
        neg_th=0.03
    )
    print("突破标签(3天):", breakout_labels_3d.labels)
    
    # 绘制突破标签热力图
    breakout_labels.plot(title="价格突破标签")
    
    # 统计突破情况
    positive_breakouts = (breakout_labels.labels == 1).sum()
    negative_breakouts = (breakout_labels.labels == -1).sum()
    no_breakouts = (breakout_labels.labels == 0).sum()
    
    print(f"正向突破: {positive_breakouts}")
    print(f"负向突破: {negative_breakouts}")
    print(f"无突破: {no_breakouts}")
    ```
    
    应用场景：
    - 技术分析中的突破信号生成
    - 动量策略的信号标签
    - 短期价格波动预测
    - 止损止盈点的动态设置
    - 波动率突破策略
    
    优势：
    - 基于未来信息的突破检测
    - 首次突破原则避免重复计算
    - 灵活的阈值设置
    - 适用于各种时间框架
    
    参数说明：
    - window: 未来观察窗口大小（时间点数量）
    - pos_th: 正向突破阈值，范围[0, 1]
    - neg_th: 负向突破阈值，范围[0, 1]
    - wait: 等待期数，避免使用当前时间点
    
    注意事项：
    - 包含前瞻性偏差，仅用于模型训练
    - 返回的标签数组末尾部分可能没有足够的未来数据
    - 阈值设置需要根据市场特点进行调整
    - 首次突破原则确保每个时间点只有一个突破标签
    """

    plot = _plot  # 继承通用绘图函数


# 将文档字符串和绘图方法添加到BOLB类
setattr(BOLB, '__doc__', _BOLB.__doc__)
setattr(BOLB, 'plot', _BOLB.plot)
