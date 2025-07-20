# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
VectorBT收益率数据访问器模块

该模块是VectorBT量化分析框架中收益率分析的核心接口层，提供了对pandas Series和DataFrame
收益率数据的高级分析功能。通过自定义的pandas访问器模式，用户可以直接在收益率序列上
调用各种专业的金融风险和绩效度量方法。

设计逻辑与架构：
    本模块采用访问器模式(Accessor Pattern)设计，通过pandas的扩展机制，为收益率数据
    添加专门的分析方法。主要包含以下几个层次：
    
    1. **基础访问器层**：ReturnsAccessor作为基础类，提供核心的收益率计算功能
    2. **类型专用层**：ReturnsSRAccessor和ReturnsDFAccessor分别处理Series和DataFrame
    3. **功能模块层**：集成统计度量、风险分析、绩效评估、图表绘制等功能模块
    4. **配置管理层**：提供默认参数管理和用户自定义配置支持

核心功能模块：
    - **收益率转换**：价格序列到收益率序列的转换和处理
    - **基础统计**：累计收益、年化收益、波动率等基础指标计算
    - **风险度量**：VaR、CVaR、最大回撤、下行风险等风险指标
    - **绩效评估**：夏普比率、索提诺比率、卡尔玛比率等绩效指标
    - **相对分析**：Alpha、Beta、信息比率、捕获比率等相对绩效指标
    - **滚动分析**：所有指标的时变滚动窗口计算版本
    - **统计报告**：自动化的统计报告生成和指标汇总
    - **可视化**：专业的金融图表绘制功能

技术特点：
    - **高性能计算**：底层使用Numba JIT编译的nb模块进行数值计算
    - **广播机制**：支持不同维度数据的自动广播和对齐
    - **缓存优化**：继承自GenericAccessor的智能缓存机制
    - **配置灵活**：支持全局默认配置和实例级别的参数覆盖
    - **类型安全**：完整的类型注释支持，提高代码可维护性

使用模式：
    ```python
    # 方法1：从价格序列创建收益率访问器
    price = pd.Series([1.1, 1.2, 1.3, 1.2, 1.1])
    ret_acc = pd.Series.vbt.returns.from_value(price, freq='d')
    
    # 方法2：从收益率序列创建访问器
    returns = price.pct_change()
    ret_acc = returns.vbt.returns(freq='d')
    
    # 方法3：使用通用转换方法
    returns = price.vbt.to_returns()
    ret_acc = returns.vbt.returns(freq='d')
    ```

应用场景：
    - **量化投资研究**：策略收益率分析和绩效评估
    - **风险管理**：投资组合风险度量和监控
    - **基金管理**：基金产品绩效分析和基准比较
    - **学术研究**：金融时间序列分析和实证研究
    - **监管报告**：符合监管要求的风险和绩效报告生成

集成生态：
    - 与vectorbt.generic.accessors无缝集成，继承通用数组操作功能
    - 与vectorbt.returns.nb模块深度集成，提供高性能数值计算
    - 与vectorbt.generic.drawdowns集成，提供专业的回撤分析
    - 支持QuantStats适配器，兼容主流量化分析工具

注意事项：
    - 输入数据必须已经是收益率序列，如需从价格转换请使用from_value方法
    - 访问器不使用缓存机制，每次调用都会重新计算
    - 分组功能仅在支持group_by参数的方法中可用
    - 某些指标需要设置基准收益率才能正常计算

Custom pandas accessors for returns data.

Methods can be accessed as follows:

* `ReturnsSRAccessor` -> `pd.Series.vbt.returns.*`
* `ReturnsDFAccessor` -> `pd.DataFrame.vbt.returns.*`

!!! note
    The underlying Series/DataFrame must already be a return series.
    To convert price to returns, use `ReturnsAccessor.from_value`.

    Grouping is only supported by the methods that accept the `group_by` argument.

    Accessors do not utilize caching.

There are three options to compute returns and get the accessor:

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbt as vbt

>>> price = pd.Series([1.1, 1.2, 1.3, 1.2, 1.1])

>>> # 1. pd.Series.pct_change
>>> rets = price.pct_change()
>>> ret_acc = rets.vbt.returns(freq='d')

>>> # 2. vectorbt.generic.accessors.GenericAccessor.to_returns
>>> rets = price.vbt.to_returns()
>>> ret_acc = rets.vbt.returns(freq='d')

>>> # 3. vectorbt.returns.accessors.ReturnsAccessor.from_value
>>> ret_acc = pd.Series.vbt.returns.from_value(price, freq='d')

>>> # vectorbt.returns.accessors.ReturnsAccessor.total
>>> ret_acc.total()
0.0
```

The accessors extend `vectorbt.generic.accessors`.

```pycon
>>> # inherited from GenericAccessor
>>> ret_acc.max()
0.09090909090909083
```

## Defaults

`vectorbt.returns.accessors.ReturnsAccessor` accepts `defaults` dictionary where you can pass
defaults for arguments used throughout the accessor, such as

* `start_value`: The starting value.
* `window`: Window length.
* `minp`: Minimum number of observations in a window required to have a value.
* `ddof`: Delta Degrees of Freedom.
* `risk_free`: Constant risk-free return throughout the period.
* `levy_alpha`: Scaling relation (Levy stability exponent).
* `required_return`: Minimum acceptance return of the investor.
* `cutoff`: Decimal representing the percentage cutoff for the bottom percentile of returns.

## Stats

!!! hint
    See `vectorbt.generic.stats_builder.StatsBuilderMixin.stats` and `ReturnsAccessor.metrics`.

```pycon
>>> ret_acc.stats()
UserWarning: Metric 'benchmark_return' requires benchmark_rets to be set
UserWarning: Metric 'alpha' requires benchmark_rets to be set
UserWarning: Metric 'beta' requires benchmark_rets to be set

Start                                      0
End                                        4
Duration                     5 days 00:00:00
Total Return [%]                           0
Annualized Return [%]                      0
Annualized Volatility [%]            184.643
Sharpe Ratio                        0.691185
Calmar Ratio                               0
Max Drawdown [%]                     15.3846
Omega Ratio                          1.08727
Sortino Ratio                        1.17805
Skew                              0.00151002
Kurtosis                            -5.94737
Tail Ratio                           1.08985
Common Sense Ratio                   1.08985
Value at Risk                     -0.0823718
dtype: object
```

The missing `benchmark_rets` can be either passed to the contrustor of the accessor
or as a setting to `ReturnsAccessor.stats`:

```pycon
>>> benchmark = pd.Series([1.05, 1.1, 1.15, 1.1, 1.05])
>>> benchmark_rets = benchmark.vbt.to_returns()

>>> ret_acc.stats(settings=dict(benchmark_rets=benchmark_rets))
Start                                      0
End                                        4
Duration                     5 days 00:00:00
Total Return [%]                           0
Benchmark Return [%]                       0
Annualized Return [%]                      0
Annualized Volatility [%]            184.643
Sharpe Ratio                        0.691185
Calmar Ratio                               0
Max Drawdown [%]                     15.3846
Omega Ratio                          1.08727
Sortino Ratio                        1.17805
Skew                              0.00151002
Kurtosis                            -5.94737
Tail Ratio                           1.08985
Common Sense Ratio                   1.08985
Value at Risk                     -0.0823718
Alpha                                0.78789
Beta                                 1.83864
dtype: object
```

!!! note
    `ReturnsAccessor.stats` does not support grouping.

## Plots

!!! hint
    See `vectorbt.generic.plots_builder.PlotsBuilderMixin.plots` and `ReturnsAccessor.subplots`.

This class inherits subplots from `vectorbt.generic.accessors.GenericAccessor`.
"""

import warnings  # 导入警告模块，用于发出运行时警告信息

import numpy as np  # 导入NumPy库，提供高性能数值计算和数组操作功能
import pandas as pd  # 导入Pandas库，提供数据结构和数据分析工具
from scipy.stats import skew, kurtosis  # 从SciPy统计模块导入偏度和峰度计算函数

from vectorbt import _typing as tp  # 导入vectorbt类型注释模块，提供类型提示支持
from vectorbt.base.array_wrapper import ArrayWrapper, Wrapping  # 导入数组包装器和包装基类
from vectorbt.base.reshape_fns import to_1d_array, to_2d_array, broadcast, broadcast_to  # 导入数组重塑和广播函数
from vectorbt.generic.accessors import (  # 导入通用访问器类
    GenericAccessor,     # 通用访问器基类
    GenericSRAccessor,   # Series专用通用访问器
    GenericDFAccessor    # DataFrame专用通用访问器
)
from vectorbt.generic.drawdowns import Drawdowns  # 导入回撤分析类
from vectorbt.returns import nb, metrics  # 导入收益率相关的Numba编译函数和统计指标模块
from vectorbt.root_accessors import register_dataframe_vbt_accessor, register_series_vbt_accessor  # 导入访问器注册装饰器
from vectorbt.utils import checks  # 导入数据验证工具模块
from vectorbt.utils.config import merge_dicts, Config  # 导入配置管理工具
from vectorbt.utils.datetime_ import freq_to_timedelta, DatetimeIndexes  # 导入日期时间处理工具
from vectorbt.utils.figure import make_figure, get_domain  # 导入图形绘制工具

__pdoc__ = {}  # 初始化文档生成器的配置字典

# 定义ReturnsAccessor类型变量，用于类型注释中的泛型支持
ReturnsAccessorT = tp.TypeVar("ReturnsAccessorT", bound="ReturnsAccessor")


class ReturnsAccessor(GenericAccessor):
    """
    收益率数据访问器基类
    
    该类是VectorBT收益率分析功能的核心访问器，为pandas Series和DataFrame提供专业的
    金融收益率分析方法。通过继承GenericAccessor，获得了基础的数组操作功能，并在此
    基础上扩展了专门针对收益率数据的分析功能。
    
    设计特点：
        - **多态支持**：同时支持Series和DataFrame类型的收益率数据
        - **基准对比**：内置基准收益率支持，便于相对绩效分析
        - **年化处理**：自动处理不同频率数据的年化计算
        - **配置灵活**：支持全局和实例级别的参数配置
        - **高性能**：底层使用Numba JIT编译的数值计算函数
    
    核心功能分类：
        1. **基础收益率指标**：
           - 累计收益率、总收益率、年化收益率
           - 年化波动率、滚动窗口版本
        
        2. **风险度量指标**：
           - 最大回撤、回撤序列分析
           - VaR、CVaR风险价值计算
           - 下行风险、尾部风险分析
        
        3. **风险调整绩效指标**：
           - 夏普比率、索提诺比率、卡尔玛比率
           - 欧米伽比率、信息比率
           - 紧缩夏普比率（统计显著性检验）
        
        4. **相对绩效指标**：
           - Alpha、Beta系数（CAPM模型）
           - 上行/下行捕获比率
           - 各种捕获比率的滚动版本
        
        5. **分布特征指标**：
           - 偏度、峰度
           - 尾部比率、常识比率
        
        6. **时间序列分析**：
           - 日收益率、年收益率转换
           - 滚动窗口分析
           - 重采样分析
    
    访问方式：
        - Series: `pd.Series.vbt.returns.*`
        - DataFrame: `pd.DataFrame.vbt.returns.*`
    
    参数说明：
        obj (pd.Series or pd.DataFrame): 
            表示收益率的Pandas对象，必须已经是收益率序列
        benchmark_rets (array_like, optional): 
            基准收益率数据，用于相对绩效分析
            - 支持Series、DataFrame或数组类型
            - 会自动广播到与obj相同的维度
        year_freq (FrequencyLike, optional): 
            年化频率，用于年化计算
            - 例如：'252D'表示252个交易日为一年
            - 默认从全局设置中获取
        defaults (dict, optional): 
            默认参数字典，覆盖全局默认设置
            - 可设置风险免费利率、窗口大小等参数
        **kwargs: 
            传递给GenericAccessor的其他关键字参数
    
    使用示例：
        >>> # 基础使用
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # 创建示例收益率数据
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
        >>> ret_acc = returns.vbt.returns(freq='D')
        >>> 
        >>> # 计算基础指标
        >>> total_return = ret_acc.total()
        >>> sharpe_ratio = ret_acc.sharpe_ratio()
        >>> max_drawdown = ret_acc.max_drawdown()
        >>> 
        >>> # 与基准比较
        >>> benchmark = pd.Series([0.008, -0.015, 0.012, -0.008, 0.003])
        >>> ret_acc_with_bench = returns.vbt.returns(
        ...     benchmark_rets=benchmark, freq='D'
        ... )
        >>> alpha = ret_acc_with_bench.alpha()
        >>> beta = ret_acc_with_bench.beta()
        
        >>> # 批量分析（DataFrame）
        >>> returns_df = pd.DataFrame({
        ...     'Strategy_A': [0.01, -0.02, 0.015, -0.01, 0.005],
        ...     'Strategy_B': [0.008, -0.018, 0.012, -0.012, 0.008]
        ... })
        >>> ret_acc_df = returns_df.vbt.returns(freq='D')
        >>> sharpe_ratios = ret_acc_df.sharpe_ratio()  # 返回Series，每个策略一个值
    
    注意事项：
        - 输入数据必须已经是收益率，不是价格数据
        - 如需从价格转换，请使用from_value类方法
        - 某些方法需要设置基准收益率才能正常工作
        - 访问器不使用缓存，每次调用都会重新计算
        - 滚动窗口分析需要足够的历史数据
    """

    def __init__(self,
                 obj: tp.SeriesFrame,
                 benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                 year_freq: tp.Optional[tp.FrequencyLike] = None,
                 defaults: tp.KwargsLike = None,
                 **kwargs) -> None:
        """
        初始化收益率访问器实例
        
        该初始化方法设置收益率分析所需的基础参数，包括基准收益率、年化频率和
        默认配置等。通过调用父类GenericAccessor的初始化方法，继承了基础的
        数组操作功能。
        
        初始化流程：
            1. 调用父类GenericAccessor的初始化方法
            2. 处理和广播基准收益率数据
            3. 设置年化频率和默认参数
            4. 存储实例属性供后续方法使用
        
        参数处理：
            - benchmark_rets会被广播到与obj相同的维度
            - year_freq用于所有涉及年化的计算（收益率、波动率等）
            - defaults会与全局设置合并，实例设置优先级更高
        
        参数说明：
            obj (tp.SeriesFrame): 收益率数据，Series或DataFrame
            benchmark_rets (tp.Optional[tp.ArrayLike]): 基准收益率
                - None: 不使用基准，相关方法会报错或跳过
                - Series/DataFrame: 会自动广播匹配obj的维度
            year_freq (tp.Optional[tp.FrequencyLike]): 年化频率
                - None: 从全局设置获取，如'252D'表示252个交易日
                - 支持pandas频率字符串格式
            defaults (tp.KwargsLike): 默认参数覆盖
                - 可设置risk_free、window、ddof等参数
            **kwargs: 传递给GenericAccessor的其他参数
        
        异常处理：
            - 如果benchmark_rets维度不匹配，会尝试自动广播
            - 如果广播失败，会在后续使用时抛出异常
        """
        # 调用父类GenericAccessor的初始化方法，传递所有参数
        # 这确保了基础的数组包装和操作功能正常工作
        GenericAccessor.__init__(
            self,
            obj,
            benchmark_rets=benchmark_rets,  # 基准收益率传递给父类
            year_freq=year_freq,           # 年化频率传递给父类
            defaults=defaults,             # 默认参数传递给父类
            **kwargs
        )

        # 处理基准收益率：如果提供了基准数据，则广播到与obj相同的维度
        if benchmark_rets is not None:
            benchmark_rets = broadcast_to(benchmark_rets, obj)
        self._benchmark_rets = benchmark_rets  # 存储基准收益率，供后续方法使用
        
        # 存储年化频率，用于所有涉及年化的计算（收益率、波动率、比率等）
        self._year_freq = year_freq
        
        # 存储默认参数配置，会与全局设置合并使用
        self._defaults = defaults

    @property
    def sr_accessor_cls(self) -> tp.Type["ReturnsSRAccessor"]:
        """
        Series专用访问器类属性
        
        返回用于处理pandas Series的专用访问器类ReturnsSRAccessor。
        这是多态设计的一部分，允许基类根据数据类型返回相应的专用访问器。
        
        返回值：
            tp.Type["ReturnsSRAccessor"]: ReturnsSRAccessor类型
        
        设计目的：
            - 支持多态：基类可以根据数据类型选择合适的子类
            - 类型安全：确保Series数据使用Series专用的访问器
            - 扩展性：便于后续添加Series特有的功能
        """
        return ReturnsSRAccessor

    @property
    def df_accessor_cls(self) -> tp.Type["ReturnsDFAccessor"]:
        """
        DataFrame专用访问器类属性
        
        返回用于处理pandas DataFrame的专用访问器类ReturnsDFAccessor。
        这是多态设计的一部分，允许基类根据数据类型返回相应的专用访问器。
        
        返回值：
            tp.Type["ReturnsDFAccessor"]: ReturnsDFAccessor类型
        
        设计目的：
            - 支持多态：基类可以根据数据类型选择合适的子类
            - 类型安全：确保DataFrame数据使用DataFrame专用的访问器
            - 扩展性：便于后续添加DataFrame特有的功能（如列间分析）
        """
        return ReturnsDFAccessor

    def indexing_func(self: ReturnsAccessorT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> ReturnsAccessorT:
        """
        执行索引操作的函数
        
        该方法实现了对ReturnsAccessor的索引操作，包括行索引和列索引。
        当用户对收益率数据进行切片或选择时，该方法确保返回的仍然是一个
        正确配置的ReturnsAccessor实例，并保持基准收益率等属性的一致性。
        
        索引处理流程：
            1. 通过wrapper获取索引操作的元数据（行列索引）
            2. 对主要收益率数据应用索引操作
            3. 对基准收益率数据应用相同的索引操作
            4. 根据结果数据类型选择合适的访问器类
            5. 创建并返回新的访问器实例
        
        参数说明：
            pd_indexing_func (tp.PandasIndexingFunc): pandas索引函数
                - 如.loc、.iloc、[]等索引操作
            **kwargs: 传递给索引函数的其他参数
        
        返回值：
            ReturnsAccessorT: 索引后的新访问器实例
                - 如果结果是Series，返回ReturnsSRAccessor
                - 如果结果是DataFrame，返回ReturnsDFAccessor
        
        使用示例：
            >>> # 时间切片
            >>> ret_acc['2023-01-01':'2023-12-31']  # 选择特定时间范围
            >>> 
            >>> # 列选择（DataFrame）
            >>> ret_acc[['Strategy_A', 'Strategy_B']]  # 选择特定策略
            >>> 
            >>> # 行列组合选择
            >>> ret_acc.loc['2023-01-01':'2023-06-30', 'Strategy_A']
        
        注意事项：
            - 索引操作会创建新的访问器实例，不会修改原实例
            - 基准收益率会应用相同的索引操作以保持对齐
            - 包装器(wrapper)会相应更新以反映新的数据形状
        """
        # 获取索引操作的元数据：新的包装器、行索引、列索引等
        new_wrapper, idx_idxs, _, col_idxs = self.wrapper.indexing_func_meta(pd_indexing_func, **kwargs)
        
        # 对主要收益率数据应用索引操作
        # 先转换为2D数组，应用行列索引，然后用新包装器包装
        new_obj = new_wrapper.wrap(self.to_2d_array()[idx_idxs, :][:, col_idxs], group_by=False)
        
        # 对基准收益率应用相同的索引操作（如果存在基准数据）
        if self.benchmark_rets is not None:
            new_benchmark_rets = new_wrapper.wrap(
                to_2d_array(self.benchmark_rets)[idx_idxs, :][:, col_idxs],
                group_by=False
            )
        else:
            new_benchmark_rets = None
        
        # 根据索引结果的数据类型选择合适的访问器类
        if checks.is_series(new_obj):
            # 如果结果是Series，返回Series专用访问器
            return self.replace(
                cls_=self.sr_accessor_cls,      # 使用Series访问器类
                obj=new_obj,                   # 索引后的数据
                benchmark_rets=new_benchmark_rets,  # 索引后的基准数据
                wrapper=new_wrapper            # 新的包装器
            )
        
        # 如果结果是DataFrame，返回DataFrame专用访问器
        return self.replace(
            cls_=self.df_accessor_cls,         # 使用DataFrame访问器类
            obj=new_obj,                      # 索引后的数据
            benchmark_rets=new_benchmark_rets, # 索引后的基准数据
            wrapper=new_wrapper               # 新的包装器
        )

    @classmethod
    def from_value(cls: tp.Type[ReturnsAccessorT],
                   value: tp.SeriesFrame,
                   init_value: tp.MaybeSeries = np.nan,
                   broadcast_kwargs: tp.KwargsLike = None,
                   wrap_kwargs: tp.KwargsLike = None,
                   **kwargs) -> ReturnsAccessorT:
        """
        从价格数据创建收益率访问器的类方法
        
        这是一个重要的工厂方法，用于将价格时间序列转换为收益率时间序列，
        并创建相应的ReturnsAccessor实例。该方法是价格数据到收益率分析的
        入口点，自动处理收益率计算和访问器创建。
        
        转换原理：
            收益率计算公式：returns[t] = (value[t] - value[t-1]) / value[t-1]
            其中value[t-1]可以是前一期价格或指定的初始值
        
        处理流程：
            1. 验证和转换输入数据格式
            2. 广播初始值到合适的维度
            3. 使用Numba编译函数计算收益率
            4. 包装结果并创建访问器实例
        
        参数说明：
            value (tp.SeriesFrame): 价格数据
                - Series: 单一资产的价格序列
                - DataFrame: 多个资产的价格矩阵
                - 必须是数值型数据
            init_value (tp.MaybeSeries): 初始价格值
                - np.nan: 使用前一期价格计算收益率（默认）
                - 标量: 所有资产使用相同初始值
                - Series: 每个资产使用不同初始值
            broadcast_kwargs (tp.KwargsLike): 广播参数
                - 控制初始值的广播行为
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 控制结果数据的包装选项
            **kwargs: 传递给访问器构造函数的其他参数
        
        返回值：
            ReturnsAccessorT: 新创建的收益率访问器实例
        
        使用示例：
            >>> # 示例1：从单一资产价格创建
            >>> import pandas as pd
            >>> import numpy as np
            >>> 
            >>> prices = pd.Series([100, 105, 102, 108, 110], 
            ...                    index=pd.date_range('2023-01-01', periods=5))
            >>> ret_acc = pd.Series.vbt.returns.from_value(prices, freq='D')
            >>> print(ret_acc.obj)  # 查看计算的收益率
            
            >>> # 示例2：从多资产价格创建
            >>> prices_df = pd.DataFrame({
            ...     'Stock_A': [100, 105, 102, 108, 110],
            ...     'Stock_B': [50, 52, 49, 51, 53]
            ... }, index=pd.date_range('2023-01-01', periods=5))
            >>> ret_acc = pd.DataFrame.vbt.returns.from_value(prices_df, freq='D')
            >>> 
            >>> # 示例3：指定初始值
            >>> ret_acc = pd.Series.vbt.returns.from_value(
            ...     prices, init_value=95, freq='D'  # 相对于95的初始价格计算
            ... )
        
        应用场景：
            - **策略回测**：将策略净值序列转换为收益率进行分析
            - **资产分析**：将股票、基金等价格数据转换为收益率
            - **组合分析**：将投资组合净值转换为收益率序列
            - **基准构建**：将基准指数价格转换为基准收益率
        
        注意事项：
            - 第一个收益率通常为NaN（除非指定了init_value）
            - 确保价格数据不包含零值或负值（会导致异常收益率）
            - 对于价格跳跃或停牌数据，可能需要预处理
            - init_value的选择会影响第一个收益率的计算
        """
        # 设置默认参数
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}
        
        # 确保输入数据是数组格式
        if not checks.is_any_array(value):
            value = np.asarray(value)
        
        # 转换为2D数组以便统一处理
        value_2d = to_2d_array(value)
        
        # 广播初始值到合适的维度（每列一个初始值）
        init_value = broadcast(init_value, to_shape=value_2d.shape[1], **broadcast_kwargs)

        # 使用Numba编译函数计算收益率
        # nb.returns_nb实现高效的收益率计算
        returns = nb.returns_nb(value_2d, init_value)
        
        # 使用ArrayWrapper包装计算结果，保持原始数据的索引和列信息
        returns = ArrayWrapper.from_obj(value).wrap(returns, **wrap_kwargs)
        
        # 创建并返回收益率访问器实例
        return cls(returns, **kwargs)

    @property
    def benchmark_rets(self) -> tp.Optional[tp.SeriesFrame]:
        """
        基准收益率属性
        
        获取用于相对绩效分析的基准收益率数据。基准收益率是投资绩效评估中的
        重要参考标准，用于计算Alpha、Beta、信息比率、捕获比率等相对绩效指标。
        
        返回值：
            tp.Optional[tp.SeriesFrame]: 基准收益率数据
                - None: 未设置基准收益率，相关相对绩效方法将无法使用
                - Series: 单一基准的收益率序列
                - DataFrame: 多基准或与主数据维度匹配的基准矩阵
        
        基准选择原则：
            - **市场基准**：如沪深300、标普500等市场指数
            - **行业基准**：相应行业或板块的代表性指数
            - **策略基准**：特定投资策略的代表性基准
            - **无风险基准**：国债收益率等无风险资产收益率
        
        使用场景：
            - Alpha和Beta计算（CAPM模型）
            - 信息比率计算（超额收益的一致性）
            - 上行/下行捕获比率（牛熊市表现分析）
            - 基准相对绩效统计报告
        
        注意事项：
            - 基准数据必须与主收益率数据在时间上对齐
            - 基准的选择应该与投资策略的风格相匹配
            - 某些方法在基准为None时会发出警告或跳过计算
        """
        return self._benchmark_rets

    @property
    def year_freq(self) -> tp.Optional[pd.Timedelta]:
        """
        年化频率属性
        
        获取用于年化计算的时间频率。该属性定义了一年包含多少个观测期，
        是所有年化指标计算的基础，如年化收益率、年化波动率、夏普比率等。
        
        返回值：
            tp.Optional[pd.Timedelta]: 年化频率时间差
                - None: 未设置年化频率，年化相关方法将无法使用
                - Timedelta: 表示一年的时间长度
        
        常见年化频率：
            - **日频数据**: 252个交易日 ('252D')
            - **周频数据**: 52周 ('52W')  
            - **月频数据**: 12个月 ('12M')
            - **季频数据**: 4个季度 ('4Q')
        
        获取逻辑：
            1. 如果实例初始化时指定了year_freq，使用指定值
            2. 否则从全局设置中获取默认年化频率
            3. 使用freq_to_timedelta转换为Timedelta对象
        
        年化计算公式：
            - 年化收益率 = (1 + 平均收益率)^年化因子 - 1
            - 年化波动率 = 收益率标准差 × sqrt(年化因子)
            - 年化因子 = year_freq / data_freq
        
        使用示例：
            >>> ret_acc.year_freq  # 查看当前年化频率
            >>> # 输出: Timedelta('252 days 00:00:00')
        
        注意事项：
            - 年化频率的选择应该与数据的实际交易频率匹配
            - 不同市场的交易日数可能不同（如中国A股约250天）
            - 加密货币等7×24交易的资产可能需要365天年化
        """
        # 如果实例初始化时未指定年化频率，从全局设置获取
        if self._year_freq is None:
            from vectorbt._settings import settings
            returns_cfg = settings['returns']

            year_freq = returns_cfg['year_freq']
            if year_freq is None:
                return None
            return freq_to_timedelta(year_freq)
        
        # 返回实例指定的年化频率（转换为Timedelta格式）
        return freq_to_timedelta(self._year_freq)

    @property
    def ann_factor(self) -> float:
        """
        年化因子属性
        
        计算用于年化转换的乘数因子。年化因子是年化频率与数据频率的比值，
        用于将周期性指标（如收益率、波动率）转换为年化指标。
        
        计算公式：
            年化因子 = year_freq / wrapper.freq
            
        例如：
            - 日频数据，年化频率252天：年化因子 = 252
            - 月频数据，年化频率12月：年化因子 = 12
            - 周频数据，年化频率52周：年化因子 = 52
        
        返回值：
            float: 年化因子数值
        
        应用场景：
            - 年化收益率计算：(1 + mean_return)^ann_factor - 1
            - 年化波动率计算：std_return × sqrt(ann_factor)
            - 夏普比率等风险调整收益指标的年化
        
        异常处理：
            - 如果wrapper.freq为None，抛出ValueError
            - 如果year_freq为None，抛出ValueError
        
        使用示例：
            >>> ret_acc.ann_factor
            >>> # 对于日频数据：252.0
            >>> # 对于月频数据：12.0
        
        注意事项：
            - 确保数据的频率信息正确设置
            - 年化因子直接影响所有年化指标的计算结果
            - 不同市场和资产类型可能需要不同的年化因子
        """
        # 检查数据频率是否已设置
        if self.wrapper.freq is None:
            raise ValueError("Index frequency is None. "
                             "Pass it as `freq` or define it globally under `settings.array_wrapper`.")
        
        # 检查年化频率是否已设置
        if self.year_freq is None:
            raise ValueError("Year frequency is None. "
                             "Pass `year_freq` or define it globally under `settings.returns`.")
        
        # 计算并返回年化因子：年化频率除以数据频率
        return self.year_freq / self.wrapper.freq

    @property
    def defaults(self) -> tp.Kwargs:
        """
        默认参数配置属性
        
        获取访问器的默认参数配置，这些参数在各种计算方法中作为默认值使用。
        该属性将全局默认设置与实例特定设置进行合并，实例设置具有更高优先级。
        
        合并逻辑：
            1. 从全局设置获取returns.defaults配置
            2. 与实例初始化时传入的defaults参数合并
            3. 实例参数覆盖全局参数
        
        返回值：
            tp.Kwargs: 合并后的默认参数字典
        
        常见默认参数：
            - start_value: 起始值（默认1.0）
            - window: 滚动窗口大小（默认252）
            - minp: 最小观测值数量（默认1）
            - ddof: 自由度增量（默认1）
            - risk_free: 无风险利率（默认0.0）
            - levy_alpha: Levy稳定性指数（默认2.0）
            - required_return: 最低要求收益率（默认0.0）
            - cutoff: VaR计算的分位数（默认0.05）
        
        参数用途：
            - risk_free: 夏普比率、Alpha等指标计算
            - window: 所有滚动窗口分析的默认窗口大小
            - ddof: 标准差、方差等统计量的自由度调整
            - cutoff: VaR、CVaR等风险指标的置信水平
        
        使用示例：
            >>> # 查看当前默认参数
            >>> ret_acc.defaults
            >>> # 输出: {'start_value': 1.0, 'window': 252, 'risk_free': 0.0, ...}
            >>> 
            >>> # 创建带自定义默认参数的访问器
            >>> custom_defaults = {'risk_free': 0.02, 'window': 63}
            >>> ret_acc = returns.vbt.returns(defaults=custom_defaults)
        
        配置层次结构：
            1. 全局设置 (settings['returns']['defaults'])
            2. 实例设置 (传入__init__的defaults参数)
            3. 方法参数 (调用具体方法时传入的参数)
        
        注意事项：
            - 参数修改不会影响已创建的访问器实例
            - 建议在创建访问器时就设置好默认参数
            - 某些参数的不当设置可能影响计算结果的合理性
        """
        from vectorbt._settings import settings
        returns_defaults_cfg = settings['returns']['defaults']

        # 使用merge_dicts合并全局默认设置和实例特定设置
        # 实例设置（self._defaults）会覆盖全局设置
        return merge_dicts(
            returns_defaults_cfg,  # 全局默认设置作为基础
            self._defaults         # 实例特定设置覆盖全局设置
        )

    def daily(self, **kwargs) -> tp.SeriesFrame:
        """
        转换为日收益率
        
        将收益率数据转换为日频率。如果原数据已经是日频率，则直接返回原数据；
        否则使用重采样方法将数据聚合为日频率，每日收益率通过该日内的总收益率计算。
        
        转换原理：
            对于非日频数据，通过重采样聚合计算每日的总收益率：
            日收益率 = (1 + r1) × (1 + r2) × ... × (1 + rn) - 1
            其中r1, r2, ..., rn是当日内的各期收益率
        
        参数说明：
            **kwargs: 传递给重采样函数的其他参数
        
        返回值：
            tp.SeriesFrame: 日频率的收益率数据
                - 如果原数据已是日频率，返回原数据
                - 否则返回重采样后的日收益率数据
        
        使用示例：
            >>> # 小时数据转日数据
            >>> hourly_returns = pd.Series([0.001, 0.002, -0.001, ...], 
            ...                           freq='H')
            >>> daily_returns = hourly_returns.vbt.returns.daily()
            >>> 
            >>> # 分钟数据转日数据
            >>> minute_returns = pd.DataFrame({...}, freq='T')
            >>> daily_returns = minute_returns.vbt.returns.daily()
        
        应用场景：
            - **高频数据聚合**：将分钟、小时数据聚合为日数据
            - **跨周期分析**：统一不同频率数据到日频率进行比较
            - **报告标准化**：将分析结果标准化为日频率报告
            - **存储优化**：减少高频数据的存储和计算负担
        
        技术实现：
            - 使用resample_apply方法进行重采样
            - 底层调用nb.total_return_apply_nb进行高效计算
            - 自动处理时区和交易日历问题
        
        注意事项：
            - 要求索引必须是DatetimeIndex类型
            - 重采样可能会丢失日内的波动信息
            - 对于24小时交易的资产，需要注意日期边界的定义
            - 聚合后的数据点数会减少
        """
        # 验证索引类型：必须是日期时间索引才能进行日频率转换
        checks.assert_instance_of(self.wrapper.index, DatetimeIndexes)

        # 如果当前数据已经是日频率，直接返回原数据
        if self.wrapper.freq == pd.Timedelta('1D'):
            return self.obj
        
        # 使用重采样方法将数据聚合为日频率
        # '1D'表示1天的频率，nb.total_return_apply_nb计算总收益率
        return self.resample_apply('1D', nb.total_return_apply_nb, **kwargs)

    def annual(self, **kwargs) -> tp.SeriesFrame:
        """
        转换为年收益率
        
        将收益率数据转换为年频率。如果原数据已经是年频率，则直接返回原数据；
        否则使用重采样方法将数据聚合为年频率，每年收益率通过该年内的总收益率计算。
        
        转换原理：
            对于非年频数据，通过重采样聚合计算每年的总收益率：
            年收益率 = (1 + r1) × (1 + r2) × ... × (1 + rn) - 1
            其中r1, r2, ..., rn是当年内的各期收益率
        
        参数说明：
            **kwargs: 传递给重采样函数的其他参数
        
        返回值：
            tp.SeriesFrame: 年频率的收益率数据
                - 如果原数据已是年频率，返回原数据
                - 否则返回重采样后的年收益率数据
        
        使用示例：
            >>> # 日数据转年数据
            >>> daily_returns = pd.Series([...], freq='D')
            >>> annual_returns = daily_returns.vbt.returns.annual()
            >>> 
            >>> # 月数据转年数据  
            >>> monthly_returns = pd.DataFrame({...}, freq='M')
            >>> annual_returns = monthly_returns.vbt.returns.annual()
            >>> 
            >>> # 查看各年收益率
            >>> print(annual_returns)
            >>> # 2020    0.156
            >>> # 2021   -0.023
            >>> # 2022    0.089
        
        应用场景：
            - **长期绩效分析**：分析各年度的投资表现
            - **年度报告**：生成年度投资绩效报告
            - **长期趋势分析**：观察多年期的收益率趋势
            - **基准比较**：与年化基准进行长期比较
            - **税务规划**：按年度计算投资收益用于税务申报
        
        年度边界处理：
            - 默认按自然年（1月1日-12月31日）划分
            - 可以通过kwargs调整年度起始月份
            - 自动处理跨年度的数据连续性
        
        技术实现：
            - 使用self.year_freq确定年化频率
            - 调用resample_apply进行重采样聚合
            - 底层使用nb.total_return_apply_nb高效计算
        
        注意事项：
            - 要求索引必须是DatetimeIndex类型
            - 年化频率必须已正确设置
            - 不完整年份的数据仍会被计算（可能需要特别注意）
            - 聚合后会丢失年内的波动和时序信息
        """
        # 验证索引类型：必须是日期时间索引才能进行年频率转换
        checks.assert_instance_of(self.obj.index, DatetimeIndexes)

        # 如果当前数据已经是年频率，直接返回原数据
        if self.wrapper.freq == self.year_freq:
            return self.obj
        
        # 使用重采样方法将数据聚合为年频率
        # self.year_freq提供年化频率，nb.total_return_apply_nb计算总收益率
        return self.resample_apply(self.year_freq, nb.total_return_apply_nb, **kwargs)

    def cumulative(self,
                   start_value: tp.Optional[float] = None,
                   wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        计算累计收益率序列
        
        将收益率时间序列转换为累计收益率序列，显示投资从起始点到各个时间点的
        累计表现。累计收益率是投资分析中最直观的绩效展示方式，常用于绘制
        净值曲线和进行可视化分析。
        
        计算公式：
            累计收益率[t] = (1 + start_value) × ∏(1 + returns[i]) - 1
            其中i从0到t，∏表示连乘
            
        等价于：
            cum_ret[0] = start_value + returns[0]
            cum_ret[t] = (1 + cum_ret[t-1]) × (1 + returns[t]) - 1
        
        参数说明：
            start_value (tp.Optional[float]): 起始累计收益率值
                - None: 使用默认配置中的start_value
                - 0.0: 从0开始累计，表示初始投资为基准
                - 其他值: 自定义起始累计收益率
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 控制返回结果的包装选项
        
        返回值：
            tp.SeriesFrame: 累计收益率序列
                - 与输入数据相同的形状和索引
                - 显示从起始点到各时点的累计表现
        
        使用示例：
            >>> # 基础累计收益率计算
            >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
            >>> cum_returns = returns.vbt.returns.cumulative()
            >>> print(cum_returns)
            >>> # 输出: [0.01, -0.0098, 0.0053, -0.0047, -0.0000]
            
            >>> # 指定起始值
            >>> cum_returns = returns.vbt.returns.cumulative(start_value=0.1)
            >>> # 从10%的初始收益率开始累计
            
            >>> # 多策略累计收益率
            >>> returns_df = pd.DataFrame({
            ...     'Strategy_A': [0.01, -0.02, 0.015],
            ...     'Strategy_B': [0.005, 0.01, -0.008]
            ... })
            >>> cum_returns_df = returns_df.vbt.returns.cumulative()
        
        应用场景：
            - **净值曲线绘制**：生成投资策略的净值走势图
            - **绩效可视化**：直观展示投资表现的时间演变
            - **回撤分析**：作为回撤计算的基础数据
            - **比较分析**：多个策略或资产的累计表现对比
            - **风险监控**：实时监控投资组合的累计损益
        
        与其他指标的关系：
            - 总收益率 = 最后一期的累计收益率
            - 最大回撤 = 基于累计收益率计算的最大下跌幅度
            - 年化收益率 = 基于累计收益率计算的年化表现
        
        技术实现：
            - 使用nb.cum_returns_nb进行高效计算
            - 支持Series和DataFrame的批量处理
            - 自动处理NaN值和边界情况
        
        注意事项：
            - 累计收益率序列对极端收益率敏感
            - 负的累计收益率表示相对起始点的损失
            - start_value的选择会影响整个序列的基准线
            - 用于绘图时通常加1转换为净值序列
        
        See `vectorbt.returns.nb.cum_returns_nb`.
        """
        # 获取起始值：如果未指定则使用默认配置
        if start_value is None:
            start_value = self.defaults['start_value']
        
        # 调用Numba编译函数计算累计收益率
        # 转换为2D数组进行统一处理，支持Series和DataFrame
        cumulative = nb.cum_returns_nb(self.to_2d_array(), start_value)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 使用wrapper包装结果，保持原始索引和列信息
        return self.wrapper.wrap(cumulative, group_by=False, **wrap_kwargs)

    def total(self, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算总收益率
        
        计算整个投资期间的总收益率，即从第一个观测点到最后一个观测点的
        累计收益率。总收益率是衡量投资整体表现的最基本指标，常用于
        绩效评估和策略比较。
        
        计算公式：
            总收益率 = ∏(1 + returns[i]) - 1
            其中i从0到n-1，∏表示连乘
            
        等价于：
            总收益率 = 最后一期的累计收益率（起始值为0）
        
        参数说明：
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 控制返回结果的包装选项
                - 默认设置name_or_index为'total_return'
        
        返回值：
            tp.MaybeSeries: 总收益率
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个总收益率
        
        使用示例：
            >>> # 单一策略总收益率
            >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
            >>> total_ret = returns.vbt.returns.total()
            >>> print(f"总收益率: {total_ret:.2%}")
            >>> # 输出: 总收益率: -0.00%
            
            >>> # 多策略总收益率比较
            >>> returns_df = pd.DataFrame({
            ...     'Strategy_A': [0.01, -0.02, 0.015, -0.01, 0.005],
            ...     'Strategy_B': [0.005, 0.01, -0.008, 0.012, -0.003]
            ... })
            >>> total_rets = returns_df.vbt.returns.total()
            >>> print(total_rets)
            >>> # Strategy_A   -0.000047
            >>> # Strategy_B    0.015549
            
            >>> # 转换为百分比显示
            >>> print((total_rets * 100).round(2))
        
        应用场景：
            - **绩效评估**：评估投资策略的整体表现
            - **策略排名**：对多个策略按总收益率排序
            - **基准比较**：与基准指数的总收益率对比
            - **投资报告**：作为投资报告的核心指标
            - **风险调整**：作为风险调整收益率的分子
        
        与其他指标的关系：
            - 年化收益率 = (1 + 总收益率)^(年化因子/期间数) - 1
            - 夏普比率 = (年化收益率 - 无风险利率) / 年化波动率
            - 累计收益率 = 包含总收益率的时间序列
        
        解读指南：
            - 正值：投资获得正收益
            - 负值：投资发生亏损
            - 接近0：投资基本持平
            - 绝对值大小：反映收益或损失的幅度
        
        技术实现：
            - 使用nb.cum_returns_final_nb计算最终累计收益率
            - 起始值固定为0.0，表示从无收益开始累计
            - 支持Series和DataFrame的批量处理
        
        注意事项：
            - 总收益率不考虑时间因素，长期和短期投资不可直接比较
            - 对于需要考虑时间价值的比较，应使用年化收益率
            - 极端收益率会显著影响总收益率的计算结果
            - 负收益率的连乘可能导致意外的结果
        
        See `vectorbt.returns.nb.cum_returns_final_nb`.
        """
        # 调用Numba编译函数计算最终累计收益率（即总收益率）
        # 起始值设为0.0，表示从零收益开始累计
        result = nb.cum_returns_final_nb(self.to_2d_array(), 0.)
        
        # 设置包装参数，默认名称为'total_return'
        wrap_kwargs = merge_dicts(dict(name_or_index='total_return'), wrap_kwargs)
        
        # 包装结果为降维输出（标量或Series）
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_total(self,
                      window: tp.Optional[int] = None,
                      minp: tp.Optional[int] = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口总收益率计算
        
        计算滚动窗口内的总收益率，提供时变的收益率分析。对于每个时间点，
        计算该点之前window长度时间窗口内的总收益率，用于分析收益率的
        时间变化特征和趋势。
        
        计算原理：
            对于每个时间点t，计算窗口[t-window+1, t]内的总收益率：
            滚动总收益率[t] = ∏(1 + returns[i]) - 1
            其中i从(t-window+1)到t
        
        参数说明：
            window (tp.Optional[int]): 滚动窗口大小
                - None: 使用默认配置中的window值
                - 整数: 指定窗口长度（时间点数）
            minp (tp.Optional[int]): 最小观测值数量
                - None: 使用默认配置中的minp值
                - 整数: 窗口内最少需要的有效观测值
            wrap_kwargs (tp.KwargsLike): 包装参数
        
        返回值：
            tp.SeriesFrame: 滚动总收益率序列
                - 前window-1个值为NaN（数据不足）
                - 后续每个值为对应窗口的总收益率
        
        使用示例：
            >>> # 计算21天滚动总收益率
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=100))
            >>> rolling_total_ret = returns.vbt.returns.rolling_total(window=21)
            >>> 
            >>> # 可视化滚动总收益率趋势
            >>> import matplotlib.pyplot as plt
            >>> rolling_total_ret.plot(title='21天滚动总收益率')
            >>> plt.show()
            
            >>> # 多策略滚动总收益率比较
            >>> returns_df = pd.DataFrame({...})
            >>> rolling_total_df = returns_df.vbt.returns.rolling_total(window=63)
            >>> # 每列显示一个策略的滚动总收益率
        
        应用场景：
            - **趋势分析**：观察收益率在不同时期的变化趋势
            - **周期性分析**：识别收益率的周期性模式
            - **动态监控**：实时监控近期投资表现
            - **择时参考**：为投资择时提供参考信号
            - **风险预警**：识别收益率恶化的早期信号
        
        窗口大小选择：
            - **短期窗口**（5-21天）：捕捉短期趋势变化
            - **中期窗口**（21-63天）：平衡趋势性和敏感性
            - **长期窗口**（63-252天）：观察长期趋势
            - **年化窗口**（252天）：年化滚动收益率
        
        技术实现：
            - 使用nb.rolling_cum_returns_final_nb进行高效计算
            - 支持最小观测值要求，提高结果可靠性
            - 自动处理边界情况和NaN值
        
        分析技巧：
            - 结合不同窗口大小进行多时间尺度分析
            - 观察滚动收益率的趋势转折点
            - 与基准的滚动收益率进行比较分析
            - 结合波动率指标进行风险调整分析
        
        注意事项：
            - 窗口大小影响结果的平滑度和敏感性
            - 较小窗口更敏感但噪声更大
            - 较大窗口更平滑但滞后性更强
            - minp参数影响有效计算的开始时间
        
        Rolling version of `ReturnsAccessor.total`.
        """
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 调用Numba编译函数计算滚动总收益率
        # 起始值设为0.0，表示每个窗口都从零收益开始累计
        result = nb.rolling_cum_returns_final_nb(self.to_2d_array(), window, minp, 0.)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def annualized(self, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算年化收益率
        
        将总收益率转换为年化收益率，消除投资期间长短的影响，使不同期间的
        投资表现具有可比性。年化收益率是投资分析中最重要的绩效指标之一，
        广泛用于投资决策和绩效评估。
        
        计算公式：
            年化收益率 = (1 + 总收益率)^(年化因子 / 观测期数) - 1
            
        其中：
            - 年化因子 = year_freq / data_freq
            - 观测期数 = 数据的时间点数量
        
        参数说明：
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'annualized_return'
        
        返回值：
            tp.MaybeSeries: 年化收益率
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个年化收益率
        
        使用示例：
            >>> # 单一策略年化收益率
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=252))
            >>> ann_ret = returns.vbt.returns.annualized()
            >>> print(f"年化收益率: {ann_ret:.2%}")
            >>> # 输出: 年化收益率: 8.56%
            
            >>> # 多策略年化收益率比较
            >>> returns_df = pd.DataFrame({...})
            >>> ann_rets = returns_df.vbt.returns.annualized()
            >>> print(ann_rets.sort_values(ascending=False))
            >>> # 按年化收益率降序排列
            
            >>> # 不同投资期间的年化收益率
            >>> short_term = returns['2023-01':'2023-06'].vbt.returns.annualized()
            >>> long_term = returns.vbt.returns.annualized()
            >>> print(f"上半年年化: {short_term:.2%}, 全年年化: {long_term:.2%}")
        
        应用场景：
            - **绩效比较**：比较不同期间长度的投资表现
            - **基准对比**：与年化基准收益率进行比较
            - **投资决策**：评估投资机会的吸引力
            - **风险调整**：作为夏普比率等指标的分子
            - **目标设定**：设定投资收益目标
        
        解读指南：
            - **优秀表现** (>15%)：显著超越市场平均水平
            - **良好表现** (8%-15%)：超越大多数基准指数
            - **平均表现** (3%-8%)：与市场平均水平相当
            - **需要改进** (<3%)：低于无风险收益率
            - **负收益** (<0%)：投资发生亏损
        
        与其他指标的关系：
            - 夏普比率 = (年化收益率 - 无风险利率) / 年化波动率
            - 年化超额收益率 = 年化收益率 - 基准年化收益率
            - 信息比率 = 年化超额收益率 / 跟踪误差
        
        技术实现：
            - 使用nb.annualized_return_nb进行高效计算
            - 自动获取年化因子进行转换
            - 支持不同频率数据的统一年化处理
        
        注意事项：
            - 需要正确设置数据频率和年化频率
            - 短期数据的年化结果可能不稳定
            - 负收益率的年化可能产生复数结果（实际返回NaN）
            - 年化收益率假设复利增长，可能高估实际表现
        
        See `vectorbt.returns.nb.annualized_return_nb`.
        """
        # 调用Numba编译函数计算年化收益率
        # 使用ann_factor进行年化转换
        result = nb.annualized_return_nb(self.to_2d_array(), self.ann_factor)
        
        # 设置包装参数，默认名称为'annualized_return'
        wrap_kwargs = merge_dicts(dict(name_or_index='annualized_return'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_annualized(self,
                           window: tp.Optional[int] = None,
                           minp: tp.Optional[int] = None,
                           wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口年化收益率计算
        
        计算滚动窗口内的年化收益率，提供时变的年化收益率分析。对于每个时间点，
        计算该点之前window长度时间窗口内的年化收益率，用于动态监控投资表现
        的时间变化和趋势分析。
        
        计算原理：
            对于每个时间点t，计算窗口[t-window+1, t]内的年化收益率：
            滚动年化收益率[t] = (1 + 窗口总收益率)^(年化因子/窗口大小) - 1
        
        参数说明：
            window (tp.Optional[int]): 滚动窗口大小
                - None: 使用默认配置中的window值
                - 整数: 指定窗口长度（时间点数）
                - 建议至少21个观测值以获得稳定结果
            minp (tp.Optional[int]): 最小观测值数量
                - None: 使用默认配置中的minp值
                - 整数: 窗口内最少需要的有效观测值
            wrap_kwargs (tp.KwargsLike): 包装参数
        
        返回值：
            tp.SeriesFrame: 滚动年化收益率序列
                - 前window-1个值为NaN（数据不足）
                - 后续每个值为对应窗口的年化收益率
        
        使用示例：
            >>> # 计算63天滚动年化收益率
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=252))
            >>> rolling_ann = returns.vbt.returns.rolling_annualized(window=63)
            >>> 
            >>> # 可视化年化收益率趋势
            >>> import matplotlib.pyplot as plt
            >>> fig, ax = plt.subplots(figsize=(12, 6))
            >>> rolling_ann.plot(ax=ax, title='63天滚动年化收益率')
            >>> ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            >>> plt.ylabel('年化收益率')
            >>> plt.show()
            
            >>> # 多策略滚动年化收益率比较
            >>> returns_df = pd.DataFrame({...})
            >>> rolling_ann_df = returns_df.vbt.returns.rolling_annualized(window=126)
            >>> # 绘制多条年化收益率曲线
            >>> rolling_ann_df.plot(title='各策略滚动年化收益率对比')
        
        应用场景：
            - **动态绩效监控**：实时监控投资策略的年化表现
            - **趋势识别**：识别年化收益率的上升或下降趋势
            - **择时决策**：基于年化收益率变化进行投资时点选择
            - **风险预警**：当年化收益率持续下降时发出预警
            - **策略调整**：根据滚动年化收益率调整投资策略
        
        窗口大小建议：
            - **短期监控**（21-42天）：快速响应收益率变化
            - **中期分析**（63-126天）：平衡敏感性和稳定性
            - **长期趋势**（126-252天）：观察长期收益率趋势
            - **年度滚动**（252天）：滚动年度年化收益率
        
        分析技巧：
            - 观察滚动年化收益率的趋势转折点
            - 结合市场事件分析收益率变化原因
            - 与基准的滚动年化收益率进行对比
            - 设置年化收益率目标线进行监控
        
        技术实现：
            - 使用nb.rolling_annualized_return_nb进行高效计算
            - 自动处理年化转换和边界情况
            - 支持最小观测值要求，提高结果可靠性
        
        投资决策应用：
            - **加仓信号**：滚动年化收益率持续上升
            - **减仓信号**：滚动年化收益率持续下降
            - **策略切换**：不同策略滚动年化收益率交叉
            - **风险控制**：年化收益率低于预设阈值
        
        注意事项：
            - 窗口大小影响结果的平滑度和时效性
            - 短期窗口的年化结果波动较大
            - 需要足够的历史数据才能开始计算
            - 市场异常期间的结果需要谨慎解读
        
        Rolling version of `ReturnsAccessor.annualized`.
        """
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 调用Numba编译函数计算滚动年化收益率
        result = nb.rolling_annualized_return_nb(self.to_2d_array(), window, minp, self.ann_factor)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def annualized_volatility(self,
                              levy_alpha: tp.Optional[float] = None,
                              ddof: tp.Optional[int] = None,
                              wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算年化波动率
        
        年化波动率是衡量投资收益不确定性的核心风险指标，表示收益率的标准差
        经过年化处理后的结果。它是风险调整收益指标（如夏普比率）的重要组成部分，
        也是投资组合优化和风险管理的基础指标。
        
        计算公式：
            标准年化波动率：σ_ann = σ × √(年化因子)
            广义年化波动率：σ_ann = σ × (年化因子)^(1/levy_alpha)
            
        其中：
            - σ: 收益率的标准差
            - 年化因子: year_freq / data_freq
            - levy_alpha: Levy稳定性指数，控制波动率的年化方式
        
        参数说明：
            levy_alpha (tp.Optional[float]): Levy稳定性指数
                - None: 使用默认配置中的levy_alpha值
                - 2.0: 标准正态分布假设（默认值）
                - 其他值: 适用于非正态分布的收益率数据
            ddof (tp.Optional[int]): 自由度增量
                - None: 使用默认配置中的ddof值
                - 1: 样本标准差（默认值，无偏估计）
                - 0: 总体标准差（有偏估计）
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'annualized_volatility'
        
        返回值：
            tp.MaybeSeries: 年化波动率
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个年化波动率
        
        使用示例：
            >>> # 基础年化波动率计算
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=252))
            >>> ann_vol = returns.vbt.returns.annualized_volatility()
            >>> print(f"年化波动率: {ann_vol:.2%}")
            >>> # 输出: 年化波动率: 15.23%
            
            >>> # 多策略波动率比较
            >>> returns_df = pd.DataFrame({...})
            >>> ann_vols = returns_df.vbt.returns.annualized_volatility()
            >>> print(ann_vols.sort_values())
            >>> # 按波动率升序排列，波动率低的策略更稳定
            
            >>> # 自定义Levy指数（适用于非正态分布）
            >>> ann_vol_levy = returns.vbt.returns.annualized_volatility(levy_alpha=1.8)
            >>> # 适用于尖峰厚尾分布的收益率数据
        
        应用场景：
            - **风险评估**：量化投资策略的风险水平
            - **夏普比率计算**：作为分母计算风险调整收益
            - **投资组合优化**：作为风险约束或目标函数
            - **风险预算**：分配各资产的风险贡献度
            - **监管合规**：满足监管对风险指标的要求
        
        波动率解读：
            - **低波动率** (<10%)：保守型投资，风险较低
            - **中等波动率** (10%-20%)：平衡型投资，适中风险
            - **高波动率** (20%-30%)：成长型投资，风险较高
            - **极高波动率** (>30%)：激进型投资，风险极高
        
        Levy稳定性指数说明：
            - **α = 2.0**：正态分布，标准的波动率年化
            - **1.0 < α < 2.0**：尖峰厚尾分布，适用于大多数金融数据
            - **α = 1.0**：柯西分布，极端厚尾情况
            - **0 < α < 1.0**：极端非正态分布，罕见情况
        
        与其他指标的关系：
            - 夏普比率 = (年化收益率 - 无风险利率) / 年化波动率
            - VaR估算 ≈ 均值 - Z分位数 × 年化波动率 / √年化因子
            - 波动率聚类：高波动率期间往往聚集出现
        
        技术实现：
            - 使用nb.annualized_volatility_nb进行高效计算
            - 支持广义的Levy稳定分布框架
            - 自动处理年化转换和统计参数
        
        注意事项：
            - 波动率假设收益率的独立同分布
            - 对于结构性变化的数据，整体波动率可能不准确
            - levy_alpha的选择应基于收益率分布的实际特征
            - 样本容量过小时波动率估计不稳定
        
        See `vectorbt.returns.nb.annualized_volatility_nb`.
        """
        # 获取Levy稳定性指数：如果未指定则使用默认配置
        if levy_alpha is None:
            levy_alpha = self.defaults['levy_alpha']
        
        # 获取自由度增量：如果未指定则使用默认配置
        if ddof is None:
            ddof = self.defaults['ddof']
        
        # 调用Numba编译函数计算年化波动率
        result = nb.annualized_volatility_nb(self.to_2d_array(), self.ann_factor, levy_alpha, ddof)
        
        # 设置包装参数，默认名称为'annualized_volatility'
        wrap_kwargs = merge_dicts(dict(name_or_index='annualized_volatility'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_annualized_volatility(self,
                                      window: tp.Optional[int] = None,
                                      minp: tp.Optional[int] = None,
                                      levy_alpha: tp.Optional[float] = None,
                                      ddof: tp.Optional[int] = None,
                                      wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口年化波动率计算
        
        计算滚动窗口内的年化波动率，提供时变的风险分析能力。通过观察波动率
        在时间序列上的变化，可以识别风险的聚集性、周期性和结构性变化，
        为动态风险管理和投资决策提供重要依据。
        
        计算原理：
            对于每个时间点t，计算窗口[t-window+1, t]内的年化波动率：
            滚动年化波动率[t] = std(returns[i]) × (年化因子)^(1/levy_alpha)
            其中i从(t-window+1)到t
        
        参数说明：
            window (tp.Optional[int]): 滚动窗口大小
                - None: 使用默认配置中的window值
                - 整数: 指定窗口长度（时间点数）
                - 建议至少30个观测值以获得稳定的波动率估计
            minp (tp.Optional[int]): 最小观测值数量
                - None: 使用默认配置中的minp值
                - 整数: 窗口内最少需要的有效观测值
            levy_alpha (tp.Optional[float]): Levy稳定性指数
                - None: 使用默认配置中的levy_alpha值
                - 2.0: 标准正态分布假设（默认值）
            ddof (tp.Optional[int]): 自由度增量
                - None: 使用默认配置中的ddof值
                - 1: 样本标准差（默认值）
            wrap_kwargs (tp.KwargsLike): 包装参数
        
        返回值：
            tp.SeriesFrame: 滚动年化波动率序列
                - 前window-1个值为NaN（数据不足）
                - 后续每个值为对应窗口的年化波动率
        
        使用示例：
            >>> # 计算30天滚动年化波动率
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=252))
            >>> rolling_vol = returns.vbt.returns.rolling_annualized_volatility(window=30)
            >>> 
            >>> # 可视化波动率聚集性
            >>> import matplotlib.pyplot as plt
            >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            >>> returns.plot(ax=ax1, title='收益率序列')
            >>> rolling_vol.plot(ax=ax2, title='30天滚动年化波动率', color='red')
            >>> plt.tight_layout()
            >>> plt.show()
            
            >>> # 多策略波动率风险监控
            >>> returns_df = pd.DataFrame({...})
            >>> rolling_vol_df = returns_df.vbt.returns.rolling_annualized_volatility(window=60)
            >>> # 识别各策略的风险变化模式
            
            >>> # 波动率突破检测
            >>> vol_threshold = rolling_vol.quantile(0.8)  # 80%分位数作为阈值
            >>> high_vol_periods = rolling_vol > vol_threshold
            >>> print(f"高波动率期间占比: {high_vol_periods.mean():.1%}")
        
        应用场景：
            - **动态风险监控**：实时监控投资组合的风险水平变化
            - **波动率择时**：基于波动率变化进行投资时点选择
            - **风险预警系统**：当波动率异常升高时触发预警
            - **对冲策略**：根据波动率水平调整对冲比例
            - **期权定价**：为期权定价模型提供时变波动率输入
        
        波动率模式识别：
            - **波动率聚集**：高波动率期间聚集，低波动率期间也聚集
            - **波动率均值回归**：极端波动率会向长期均值回归
            - **杠杆效应**：负收益后波动率往往上升
            - **周末效应**：周一的波动率通常较高
        
        窗口大小选择策略：
            - **短期窗口**（10-30天）：快速捕捉波动率变化
            - **中期窗口**（30-90天）：平衡敏感性和稳定性
            - **长期窗口**（90-252天）：识别波动率的长期趋势
            - **多窗口分析**：结合不同窗口进行多时间尺度分析
        
        风险管理应用：
            - **风险预算调整**：根据波动率变化调整仓位
            - **止损设置**：基于当前波动率设置动态止损点
            - **资产配置**：在高波动率期间降低风险资产比例
            - **流动性管理**：波动率上升时提高现金比例
        
        技术分析指标：
            - **波动率突破**：波动率突破历史分位数
            - **波动率背离**：价格与波动率的背离信号
            - **波动率支撑阻力**：波动率的关键技术位
            - **波动率趋势线**：波动率的趋势方向判断
        
        注意事项：
            - 窗口大小影响波动率估计的平滑度和敏感性
            - 市场极端事件会导致波动率出现异常值
            - 波动率的预测能力有限，主要用于当前风险评估
            - 不同资产的波动率特征可能差异很大
        
        Rolling version of `ReturnsAccessor.annualized_volatility`.
        """
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 获取Levy稳定性指数：如果未指定则使用默认配置
        if levy_alpha is None:
            levy_alpha = self.defaults['levy_alpha']
        
        # 获取自由度增量：如果未指定则使用默认配置
        if ddof is None:
            ddof = self.defaults['ddof']
        
        # 调用Numba编译函数计算滚动年化波动率
        result = nb.rolling_annualized_volatility_nb(
            self.to_2d_array(), window, minp, self.ann_factor, levy_alpha, ddof)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def calmar_ratio(self, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算卡尔玛比率（Calmar Ratio）
        
        卡尔玛比率是一种风险调整收益指标，衡量单位最大回撤风险下的年化收益率。
        它特别适用于评估绝对收益策略和对冲基金的表现，因为这些策略通常
        更关注控制回撤风险而非相对波动率。
        
        计算公式：
            卡尔玛比率 = 年化收益率 / |最大回撤|
            
        其中：
            - 年化收益率：投资策略的年化表现
            - 最大回撤：历史上最大的峰谷损失幅度（取绝对值）
        
        参数说明：
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'calmar_ratio'
        
        返回值：
            tp.MaybeSeries: 卡尔玛比率
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个卡尔玛比率
        
        使用示例：
            >>> # 基础卡尔玛比率计算
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=252))
            >>> calmar = returns.vbt.returns.calmar_ratio()
            >>> print(f"卡尔玛比率: {calmar:.2f}")
            >>> # 输出: 卡尔玛比率: 1.85
            
            >>> # 多策略卡尔玛比率比较
            >>> returns_df = pd.DataFrame({
            ...     'Strategy_A': [...],
            ...     'Strategy_B': [...],
            ...     'Strategy_C': [...]
            ... })
            >>> calmar_ratios = returns_df.vbt.returns.calmar_ratio()
            >>> print(calmar_ratios.sort_values(ascending=False))
            >>> # 按卡尔玛比率降序排列，评估风险调整后表现
            
            >>> # 与夏普比率对比分析
            >>> sharpe_ratios = returns_df.vbt.returns.sharpe_ratio()
            >>> comparison = pd.DataFrame({
            ...     'Calmar': calmar_ratios,
            ...     'Sharpe': sharpe_ratios
            ... })
            >>> print(comparison)
        
        应用场景：
            - **对冲基金评估**：评估绝对收益策略的风险调整表现
            - **CTA策略分析**：商品交易顾问策略的专业评估指标
            - **回撤风险控制**：重点关注最大损失的投资策略评估
            - **资产管理**：私募基金和专户产品的绩效评价
            - **风险预算**：基于回撤风险的资产配置决策
        
        指标解读：
            - **优秀表现** (>3.0)：年化收益是最大回撤的3倍以上
            - **良好表现** (1.5-3.0)：具有较好的风险调整收益
            - **一般表现** (0.5-1.5)：风险收益比相对平衡
            - **需要改进** (<0.5)：回撤风险过大或收益不足
            - **负值**：年化收益为负，投资策略存在问题
        
        与其他指标的比较：
            - **vs夏普比率**：卡尔玛关注回撤，夏普关注波动率
            - **vs索提诺比率**：卡尔玛用最大回撤，索提诺用下行偏差
            - **vs信息比率**：卡尔玛是绝对指标，信息比率是相对指标
        
        优势特点：
            - **直观性强**：最大回撤比波动率更直观易懂
            - **实用性高**：符合投资者对损失的关注重点
            - **稳健性好**：不受收益率分布形状影响
            - **适用性广**：特别适合评估绝对收益策略
        
        技术实现：
            - 使用nb.calmar_ratio_nb进行高效计算
            - 自动处理最大回撤为零的边界情况
            - 集成年化收益率和最大回撤的计算
        
        注意事项：
            - 当最大回撤为0时，比率为无穷大（实际返回NaN）
            - 历史最大回撤不能预测未来回撤风险
            - 短期数据计算的卡尔玛比率可能不稳定
            - 需要足够长的历史数据才能获得可靠结果
        
        投资决策指导：
            - 卡尔玛比率高的策略适合风险厌恶投资者
            - 可以作为投资组合构建的重要筛选指标
            - 结合其他指标进行综合评估更为可靠
        
        See `vectorbt.returns.nb.calmar_ratio_nb`.
        """
        # 调用Numba编译函数计算卡尔玛比率
        result = nb.calmar_ratio_nb(self.to_2d_array(), self.ann_factor)
        
        # 设置包装参数，默认名称为'calmar_ratio'
        wrap_kwargs = merge_dicts(dict(name_or_index='calmar_ratio'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_calmar_ratio(self,
                             window: tp.Optional[int] = None,
                             minp: tp.Optional[int] = None,
                             wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口卡尔玛比率计算
        
        计算滚动窗口内的卡尔玛比率，提供时变的风险调整绩效分析。通过观察
        卡尔玛比率在时间序列上的变化，可以识别策略绩效的稳定性和风险控制
        能力的时间变化特征，为动态绩效评估和策略调整提供依据。
        
        计算原理：
            对于每个时间点t，计算窗口[t-window+1, t]内的卡尔玛比率：
            滚动卡尔玛比率[t] = 窗口年化收益率 / |窗口最大回撤|
        
        参数说明：
            window (tp.Optional[int]): 滚动窗口大小
                - None: 使用默认配置中的window值
                - 整数: 指定窗口长度（时间点数）
                - 建议至少60个观测值以获得稳定的回撤估计
            minp (tp.Optional[int]): 最小观测值数量
                - None: 使用默认配置中的minp值
                - 整数: 窗口内最少需要的有效观测值
            wrap_kwargs (tp.KwargsLike): 包装参数
        
        返回值：
            tp.SeriesFrame: 滚动卡尔玛比率序列
                - 前window-1个值为NaN（数据不足）
                - 后续每个值为对应窗口的卡尔玛比率
        
        使用示例：
            >>> # 计算126天滚动卡尔玛比率
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=500))
            >>> rolling_calmar = returns.vbt.returns.rolling_calmar_ratio(window=126)
            >>> 
            >>> # 可视化卡尔玛比率时间变化
            >>> import matplotlib.pyplot as plt
            >>> fig, ax = plt.subplots(figsize=(12, 6))
            >>> rolling_calmar.plot(ax=ax, title='126天滚动卡尔玛比率')
            >>> ax.axhline(y=1.0, color='red', linestyle='--', label='基准线')
            >>> ax.legend()
            >>> plt.ylabel('卡尔玛比率')
            >>> plt.show()
            
            >>> # 多策略滚动卡尔玛比率监控
            >>> returns_df = pd.DataFrame({...})
            >>> rolling_calmar_df = returns_df.vbt.returns.rolling_calmar_ratio(window=90)
            >>> # 识别各策略风险调整表现的时间变化
            
            >>> # 卡尔玛比率恶化检测
            >>> calmar_decline = rolling_calmar.pct_change(periods=21)  # 21天变化率
            >>> significant_decline = calmar_decline < -0.3  # 30%以上下降
            >>> print(f"显著恶化期间数: {significant_decline.sum()}")
        
        应用场景：
            - **动态绩效监控**：监控策略风险调整表现的时间变化
            - **策略优化时点**：识别需要调整策略参数的时间点
            - **风险管理**：当卡尔玛比率持续下降时及时采取措施
            - **投资决策**：基于滚动卡尔玛比率进行投资时点选择
            - **基金评估**：评估基金经理在不同市场环境下的表现
        
        分析模式：
            - **趋势分析**：观察卡尔玛比率的上升或下降趋势
            - **均值回归**：极端卡尔玛比率的回归特征
            - **周期性模式**：识别季节性或周期性变化
            - **结构性变化**：检测策略绩效的结构性改变
        
        窗口大小选择：
            - **短期窗口**（30-60天）：快速响应绩效变化
            - **中期窗口**（60-126天）：平衡敏感性和稳定性
            - **长期窗口**（126-252天）：观察长期绩效趋势
            - **多窗口分析**：结合不同窗口进行综合分析
        
        预警信号设置：
            - **绩效恶化**：滚动卡尔玛比率持续下降
            - **风险失控**：卡尔玛比率低于历史分位数
            - **策略失效**：比率长期低于基准阈值
            - **机会识别**：比率从低位开始回升
        
        技术实现：
            - 使用nb.rolling_calmar_ratio_nb进行高效计算
            - 自动处理窗口内最大回撤的计算
            - 支持最小观测值要求，提高结果可靠性
        
        注意事项：
            - 窗口大小影响回撤估计的准确性和时效性
            - 短期窗口可能无法捕捉真正的最大回撤
            - 市场趋势变化可能导致比率出现异常值
            - 需要结合绝对水平和相对变化进行分析
        
        Rolling version of `ReturnsAccessor.calmar_ratio`.
        """
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 调用Numba编译函数计算滚动卡尔玛比率
        result = nb.rolling_calmar_ratio_nb(self.to_2d_array(), window, minp, self.ann_factor)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def omega_ratio(self,
                    risk_free: tp.Optional[float] = None,
                    required_return: tp.Optional[float] = None,
                    wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算欧米伽比率（Omega Ratio）
        
        欧米伽比率是一种全面的风险调整绩效指标，它考虑了收益分布的所有矩，
        不仅仅是均值和方差。该比率衡量了超过阈值收益的概率加权收益与
        低于阈值收益的概率加权损失之比，提供了更全面的风险收益评估。
        
        计算公式：
            欧米伽比率 = ∫[r≥τ] (1-F(r))dr / ∫[r<τ] F(r)dr
            
        其中：
            - F(r): 收益率的累积分布函数
            - τ: 阈值收益率（通常为要求收益率或无风险利率）
            - 分子：高于阈值的期望超额收益
            - 分母：低于阈值的期望缺口损失
        
        参数说明：
            risk_free (tp.Optional[float]): 无风险收益率
                - None: 使用默认配置中的risk_free值
                - 浮点数: 指定无风险利率（年化）
            required_return (tp.Optional[float]): 要求收益率
                - None: 使用默认配置中的required_return值
                - 浮点数: 投资者的最低要求收益率
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'omega_ratio'
        
        返回值：
            tp.MaybeSeries: 欧米伽比率
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个欧米伽比率
        
        使用示例：
            >>> # 基础欧米伽比率计算
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=252))
            >>> omega = returns.vbt.returns.omega_ratio(risk_free=0.03)
            >>> print(f"欧米伽比率: {omega:.2f}")
            >>> # 输出: 欧米伽比率: 1.25
            
            >>> # 不同阈值下的欧米伽比率
            >>> omega_0 = returns.vbt.returns.omega_ratio(required_return=0.0)
            >>> omega_5 = returns.vbt.returns.omega_ratio(required_return=0.05)
            >>> omega_10 = returns.vbt.returns.omega_ratio(required_return=0.10)
            >>> print(f"0%阈值: {omega_0:.2f}, 5%阈值: {omega_5:.2f}, 10%阈值: {omega_10:.2f}")
            
            >>> # 多策略欧米伽比率比较
            >>> returns_df = pd.DataFrame({...})
            >>> omega_ratios = returns_df.vbt.returns.omega_ratio(risk_free=0.03)
            >>> print(omega_ratios.sort_values(ascending=False))
        
        应用场景：
            - **全面绩效评估**：考虑收益分布所有特征的综合评估
            - **非正态分布分析**：适用于收益率非正态分布的策略
            - **下行风险评估**：更准确地衡量下行风险暴露
            - **投资组合优化**：作为优化目标函数的候选指标
            - **另类投资评估**：对冲基金、私募股权等复杂策略评估
        
        指标解读：
            - **优秀表现** (>1.5)：盈利概率和幅度明显超过损失
            - **良好表现** (1.2-1.5)：具有较好的风险调整收益
            - **一般表现** (1.0-1.2)：盈亏基本平衡
            - **需要改进** (0.8-1.0)：损失概率或幅度偏高
            - **表现不佳** (<0.8)：系统性产生损失
        
        与其他指标的比较：
            - **vs夏普比率**：欧米伽考虑分布形状，夏普假设正态分布
            - **vs索提诺比率**：欧米伽更全面，索提诺只看下行偏差
            - **vs卡尔玛比率**：欧米伽看全分布，卡尔玛只看最大回撤
        
        优势特点：
            - **分布无关**：不假设收益率分布形状
            - **信息丰富**：利用收益率分布的全部信息
            - **直观理解**：盈利与亏损的直接比较
            - **阈值灵活**：可以根据投资目标调整阈值
        
        阈值选择策略：
            - **无风险利率**：最常用的基准，评估超额收益
            - **通胀率**：保值增值的基本要求
            - **基准收益率**：与特定基准的比较
            - **目标收益率**：投资者的具体收益目标
        
        技术实现：
            - 使用nb.omega_ratio_nb进行高效计算
            - 基于经验分布函数计算积分
            - 自动处理边界情况和数值稳定性
        
        注意事项：
            - 阈值的选择显著影响比率的数值
            - 样本量小时估计可能不稳定
            - 极端值对结果有较大影响
            - 需要足够的历史数据才能准确估计分布
        
        See `vectorbt.returns.nb.omega_ratio_nb`.
        """
        # 获取无风险收益率：如果未指定则使用默认配置
        if risk_free is None:
            risk_free = self.defaults['risk_free']
        
        # 获取要求收益率：如果未指定则使用默认配置
        if required_return is None:
            required_return = self.defaults['required_return']
        
        # 调用Numba编译函数计算欧米伽比率
        result = nb.omega_ratio_nb(self.to_2d_array(), self.ann_factor, risk_free, required_return)
        
        # 设置包装参数，默认名称为'omega_ratio'
        wrap_kwargs = merge_dicts(dict(name_or_index='omega_ratio'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_omega_ratio(self,
                            window: tp.Optional[int] = None,
                            minp: tp.Optional[int] = None,
                            risk_free: tp.Optional[float] = None,
                            required_return: tp.Optional[float] = None,
                            wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口欧米伽比率计算
        
        计算滚动窗口内的欧米伽比率，提供时变的全面风险调整绩效分析。
        通过观察欧米伽比率的时间变化，可以识别策略在不同市场环境下
        的表现差异和风险收益特征的演变。
        
        Rolling version of `ReturnsAccessor.omega_ratio`.
        """
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 获取无风险收益率：如果未指定则使用默认配置
        if risk_free is None:
            risk_free = self.defaults['risk_free']
        
        # 获取要求收益率：如果未指定则使用默认配置
        if required_return is None:
            required_return = self.defaults['required_return']
        
        # 调用Numba编译函数计算滚动欧米伽比率
        result = nb.rolling_omega_ratio_nb(
            self.to_2d_array(), window, minp, self.ann_factor, risk_free, required_return)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def sharpe_ratio(self,
                     risk_free: tp.Optional[float] = None,
                     ddof: tp.Optional[int] = None,
                     wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算夏普比率（Sharpe Ratio）
        
        夏普比率是最著名和最广泛使用的风险调整收益指标，衡量单位风险
        （标准差）下的超额收益。它是投资绩效评估的黄金标准，广泛应用于
        投资决策、基金评估和投资组合优化。
        
        计算公式：
            夏普比率 = (年化收益率 - 无风险利率) / 年化波动率
            
        其中：
            - 年化收益率：投资策略的年化表现
            - 无风险利率：无风险资产的年化收益率
            - 年化波动率：收益率的年化标准差
        
        参数说明：
            risk_free (tp.Optional[float]): 无风险收益率
                - None: 使用默认配置中的risk_free值
                - 浮点数: 指定无风险利率（年化）
                - 通常使用国债收益率作为代理
            ddof (tp.Optional[int]): 自由度增量
                - None: 使用默认配置中的ddof值
                - 1: 样本标准差（默认值，无偏估计）
                - 0: 总体标准差（有偏估计）
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'sharpe_ratio'
        
        返回值：
            tp.MaybeSeries: 夏普比率
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个夏普比率
        
        使用示例：
            >>> # 基础夏普比率计算
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=252))
            >>> sharpe = returns.vbt.returns.sharpe_ratio(risk_free=0.03)
            >>> print(f"夏普比率: {sharpe:.2f}")
            >>> # 输出: 夏普比率: 1.45
            
            >>> # 多策略夏普比率比较
            >>> returns_df = pd.DataFrame({
            ...     'Growth_Strategy': [...],
            ...     'Value_Strategy': [...],
            ...     'Momentum_Strategy': [...]
            ... })
            >>> sharpe_ratios = returns_df.vbt.returns.sharpe_ratio(risk_free=0.03)
            >>> print(sharpe_ratios.sort_values(ascending=False))
            >>> # Growth_Strategy      1.85
            >>> # Momentum_Strategy    1.42
            >>> # Value_Strategy       1.18
            
            >>> # 不同无风险利率下的敏感性分析
            >>> sharpe_2 = returns.vbt.returns.sharpe_ratio(risk_free=0.02)
            >>> sharpe_3 = returns.vbt.returns.sharpe_ratio(risk_free=0.03)
            >>> sharpe_4 = returns.vbt.returns.sharpe_ratio(risk_free=0.04)
            >>> print(f"2%: {sharpe_2:.2f}, 3%: {sharpe_3:.2f}, 4%: {sharpe_4:.2f}")
        
        应用场景：
            - **投资绩效评估**：评估基金、策略或投资组合的表现
            - **资产选择**：比较不同投资选择的风险调整收益
            - **投资组合优化**：最大化夏普比率的投资组合构建
            - **基金经理评估**：评估基金经理的投资技能
            - **风险预算**：基于夏普比率的风险资本分配
        
        指标解读：
            - **卓越表现** (>2.0)：世界级的投资表现
            - **优秀表现** (1.5-2.0)：非常好的风险调整收益
            - **良好表现** (1.0-1.5)：具有投资价值的策略
            - **一般表现** (0.5-1.0)：收益与风险基本匹配
            - **需要改进** (0-0.5)：风险相对收益过高
            - **表现不佳** (<0)：收益低于无风险收益率
        
        理论基础：
            - 基于现代投资组合理论（MPT）
            - 假设收益率服从正态分布
            - 反映投资者的效用函数（均值-方差偏好）
            - 与资本资产定价模型（CAPM）密切相关
        
        与其他指标的关系：
            - **信息比率** = 夏普比率（当基准为无风险资产时）
            - **M²指标** = 夏普比率 × 基准波动率 + 无风险利率
            - **特雷诺比率** = 超额收益 / Beta系数
        
        优势特点：
            - **简单直观**：易于理解和计算
            - **广泛接受**：行业标准的绩效评估指标
            - **理论支撑**：有坚实的金融理论基础
            - **可比性强**：不同资产和策略间可直接比较
        
        局限性：
            - **分布假设**：假设收益率正态分布（实际常不满足）
            - **对称性**：将上行和下行波动同等对待
            - **线性关系**：假设风险收益呈线性关系
            - **静态性**：不能反映时变的风险收益特征
        
        改进版本：
            - **修正夏普比率**：考虑收益率分布的偏度和峰度
            - **下行夏普比率**：只考虑下行波动率
            - **条件夏普比率**：基于极值理论的修正
        
        技术实现：
            - 使用nb.sharpe_ratio_nb进行高效计算
            - 自动处理年化转换和统计参数
            - 支持不同频率数据的统一计算
        
        最佳实践：
            - 使用至少1年以上的数据计算
            - 选择合适的无风险利率基准
            - 结合其他指标进行综合评估
            - 注意市场环境对夏普比率的影响
        
        注意事项：
            - 短期数据的夏普比率波动较大
            - 极端市场条件下指标可能失真
            - 不同市场环境的可比性需要注意
            - 波动率为零时比率为无穷大（实际返回NaN）
        
        See `vectorbt.returns.nb.sharpe_ratio_nb`.
        """
        # 获取无风险收益率：如果未指定则使用默认配置
        if risk_free is None:
            risk_free = self.defaults['risk_free']
        
        # 获取自由度增量：如果未指定则使用默认配置
        if ddof is None:
            ddof = self.defaults['ddof']
        
        # 调用Numba编译函数计算夏普比率
        result = nb.sharpe_ratio_nb(self.to_2d_array(), self.ann_factor, risk_free, ddof)
        
        # 设置包装参数，默认名称为'sharpe_ratio'
        wrap_kwargs = merge_dicts(dict(name_or_index='sharpe_ratio'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_sharpe_ratio(self,
                             window: tp.Optional[int] = None,
                             minp: tp.Optional[int] = None,
                             risk_free: tp.Optional[float] = None,
                             ddof: tp.Optional[int] = None,
                             wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口夏普比率计算
        
        计算滚动窗口内的夏普比率，提供时变的风险调整绩效分析。滚动夏普比率
        是动态绩效监控和投资决策的重要工具，能够识别策略在不同市场环境下
        的表现变化和风险调整能力的时间特征。
        
        计算原理：
            对于每个时间点t，计算窗口[t-window+1, t]内的夏普比率：
            滚动夏普比率[t] = (窗口年化收益率 - 无风险利率) / 窗口年化波动率
        
        参数说明：
            window (tp.Optional[int]): 滚动窗口大小
                - None: 使用默认配置中的window值
                - 整数: 指定窗口长度（时间点数）
                - 建议至少30个观测值以获得稳定的夏普比率
            minp (tp.Optional[int]): 最小观测值数量
                - None: 使用默认配置中的minp值
                - 整数: 窗口内最少需要的有效观测值
            risk_free (tp.Optional[float]): 无风险收益率
                - None: 使用默认配置中的risk_free值
                - 浮点数: 指定无风险利率（年化）
            ddof (tp.Optional[int]): 自由度增量
                - None: 使用默认配置中的ddof值
                - 1: 样本标准差（默认值）
            wrap_kwargs (tp.KwargsLike): 包装参数
        
        返回值：
            tp.SeriesFrame: 滚动夏普比率序列
                - 前window-1个值为NaN（数据不足）
                - 后续每个值为对应窗口的夏普比率
        
        使用示例：
            >>> # 计算63天滚动夏普比率
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=500))
            >>> rolling_sharpe = returns.vbt.returns.rolling_sharpe_ratio(
            ...     window=63, risk_free=0.03
            ... )
            >>> 
            >>> # 可视化夏普比率时间变化
            >>> import matplotlib.pyplot as plt
            >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            >>> 
            >>> # 上图：累计收益率
            >>> cum_returns = returns.vbt.returns.cumulative()
            >>> cum_returns.plot(ax=ax1, title='累计收益率')
            >>> 
            >>> # 下图：滚动夏普比率
            >>> rolling_sharpe.plot(ax=ax2, title='63天滚动夏普比率', color='red')
            >>> ax2.axhline(y=1.0, color='black', linestyle='--', label='优秀阈值')
            >>> ax2.axhline(y=0.5, color='gray', linestyle='--', label='一般阈值')
            >>> ax2.legend()
            >>> plt.tight_layout()
            >>> plt.show()
            
            >>> # 多策略滚动夏普比率比较
            >>> returns_df = pd.DataFrame({...})
            >>> rolling_sharpe_df = returns_df.vbt.returns.rolling_sharpe_ratio(window=126)
            >>> # 绘制多策略夏普比率对比
            >>> rolling_sharpe_df.plot(title='各策略滚动夏普比率', figsize=(12, 6))
            
            >>> # 夏普比率稳定性分析
            >>> sharpe_std = rolling_sharpe.rolling(window=63).std()
            >>> print(f"夏普比率波动率: {sharpe_std.mean():.3f}")
            >>> stable_periods = sharpe_std < sharpe_std.quantile(0.2)
            >>> print(f"稳定期间占比: {stable_periods.mean():.1%}")
        
        应用场景：
            - **动态绩效监控**：实时监控投资策略的风险调整表现
            - **策略择时**：基于滚动夏普比率进行投资时点选择
            - **风险管理**：当夏普比率恶化时及时调整策略
            - **基金评估**：评估基金经理的持续表现能力
            - **组合再平衡**：根据夏普比率变化调整资产配置
        
        分析维度：
            - **趋势分析**：识别夏普比率的改善或恶化趋势
            - **稳定性评估**：评估策略绩效的稳定性
            - **周期性模式**：识别季节性或周期性变化
            - **极值分析**：识别异常表现的时间段
        
        窗口大小策略：
            - **短期窗口**（21-42天）：快速响应绩效变化
            - **中期窗口**（63-126天）：平衡响应速度和稳定性
            - **长期窗口**（126-252天）：观察长期绩效趋势
            - **多窗口监控**：同时使用多个窗口进行综合分析
        
        投资决策信号：
            - **加仓信号**：滚动夏普比率持续改善且超过阈值
            - **减仓信号**：滚动夏普比率持续恶化
            - **策略切换**：不同策略夏普比率的交叉信号
            - **风险控制**：夏普比率低于预设底线时的保护措施
        
        市场环境适应性：
            - **牛市**：关注夏普比率的绝对水平
            - **熊市**：更关注相对表现和风险控制
            - **震荡市**：重视夏普比率的稳定性
            - **极端市场**：结合其他指标综合判断
        
        技术实现：
            - 使用nb.rolling_sharpe_ratio_nb进行高效计算
            - 自动处理窗口内的年化转换
            - 支持最小观测值要求，提高结果可靠性
        
        质量控制：
            - **异常值检测**：识别和处理极端夏普比率值
            - **稳定性检验**：评估不同窗口大小的结果一致性
            - **敏感性分析**：测试无风险利率变化的影响
        
        注意事项：
            - 窗口大小显著影响夏普比率的平滑度和敏感性
            - 市场转折期间可能出现滞后或误导信号
            - 需要结合市场环境和策略特征进行解读
            - 短期异常表现可能被过度放大或平滑
        
        Rolling version of `ReturnsAccessor.sharpe_ratio`.
        """
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 获取无风险收益率：如果未指定则使用默认配置
        if risk_free is None:
            risk_free = self.defaults['risk_free']
        
        # 获取自由度增量：如果未指定则使用默认配置
        if ddof is None:
            ddof = self.defaults['ddof']
        
        # 调用Numba编译函数计算滚动夏普比率
        result = nb.rolling_sharpe_ratio_nb(self.to_2d_array(), window, minp, self.ann_factor, risk_free, ddof)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def deflated_sharpe_ratio(self,
                              risk_free: tp.Optional[float] = None,
                              ddof: tp.Optional[int] = None,
                              var_sharpe: tp.Optional[float] = None,
                              nb_trials: tp.Optional[int] = None,
                              bias: bool = True,
                              wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算紧缩夏普比率（Deflated Sharpe Ratio, DSR）
        
        紧缩夏普比率是一种校正多重假设检验偏差的统计指标，用于评估投资策略
        夏普比率的真实统计显著性。该指标通过考虑策略选择偏差、回测过拟合和
        收益分布的非正态性，提供更加严格和可靠的策略绩效评估。
        
        核心思想：
            传统的夏普比率在多策略测试环境中容易产生选择偏差，即研究者倾向于
            选择表现最好的策略，而忽略了这种选择过程本身带来的统计偏差。DSR
            通过引入统计显著性检验，将策略的夏普比率转换为其统计显著性的概率。
        
        理论基础：
            DSR基于假设检验理论，原假设H0为策略无真实超额收益能力，备择假设H1为
            策略具有真实的投资技能。DSR值表示在给定证据下，拒绝原假设的置信度，
            即策略具有真实技能的概率。
        
        参数说明：
            risk_free (tp.Optional[float]): 无风险收益率
                - None: 使用默认配置中的risk_free值
                - 浮点数: 指定无风险利率（年化）
            ddof (tp.Optional[int]): 自由度增量
                - None: 使用默认配置中的ddof值
                - 1: 样本标准差（默认值，无偏估计）
            var_sharpe (tp.Optional[float]): 夏普比率的方差
                - None: 基于所有列计算夏普比率方差
                - 浮点数: 指定夏普比率的方差估计
            nb_trials (tp.Optional[int]): 试验次数（测试的策略数量）
                - None: 使用数据的列数作为试验次数
                - 整数: 指定实际测试的策略总数
            bias (bool): 偏度和峰度计算是否使用偏估计
                - True: 使用有偏估计（默认值）
                - False: 使用无偏估计
            wrap_kwargs (tp.KwargsLike): 包装参数
        
        返回值：
            tp.MaybeSeries: 紧缩夏普比率
                - 取值范围在[0, 1]之间
                - 接近1表示策略具有高度统计显著性
                - 接近0.5表示策略与随机策略无显著差异
                - 低于0.5表示策略表现可能不如随机策略
        
        使用示例：
            >>> # 基础紧缩夏普比率计算
            >>> returns_df = pd.DataFrame({
            ...     'Strategy_A': [...],
            ...     'Strategy_B': [...],  
            ...     'Strategy_C': [...]
            ... })
            >>> dsr = returns_df.vbt.returns.deflated_sharpe_ratio(
            ...     risk_free=0.03, nb_trials=100
            ... )
            >>> print(dsr)
            >>> # Strategy_A    0.892
            >>> # Strategy_B    0.654
            >>> # Strategy_C    0.234
            
            >>> # 结果解读
            >>> for strategy, dsr_val in dsr.items():
            ...     if dsr_val > 0.95:
            ...         print(f"{strategy}: 高度统计显著 (DSR={dsr_val:.3f})")
            ...     elif dsr_val > 0.8:
            ...         print(f"{strategy}: 统计显著 (DSR={dsr_val:.3f})")
            ...     elif dsr_val > 0.5:
            ...         print(f"{strategy}: 可能有一定技能 (DSR={dsr_val:.3f})")
            ...     else:
            ...         print(f"{strategy}: 缺乏统计显著性 (DSR={dsr_val:.3f})")
            
            >>> # 与传统夏普比率对比
            >>> sharpe = returns_df.vbt.returns.sharpe_ratio(risk_free=0.03)
            >>> comparison = pd.DataFrame({
            ...     'Sharpe_Ratio': sharpe,
            ...     'Deflated_Sharpe': dsr,
            ...     'Significant': dsr > 0.8
            ... })
            >>> print(comparison)
        
        应用场景：
            - **量化基金管理**：评估基金经理策略的真实有效性
            - **投资研究**：验证新开发策略的统计显著性
            - **风险管理**：识别过拟合的策略，避免投资风险
            - **监管合规**：满足监管机构对策略有效性的统计验证要求
            - **学术研究**：为金融学术研究提供严格的统计推断工具
        
        结果解释指南：
            - **DSR > 0.95**: 策略具有高度统计显著性，强烈推荐采用
            - **0.8 < DSR ≤ 0.95**: 策略具有统计显著性，可以考虑采用
            - **0.6 < DSR ≤ 0.8**: 策略可能具有一定技能，需要进一步验证
            - **0.4 < DSR ≤ 0.6**: 策略与随机策略差异不大，需要谨慎
            - **DSR ≤ 0.4**: 策略缺乏统计显著性，不建议采用
        
        技术实现：
            - 使用metrics.deflated_sharpe_ratio进行计算
            - 自动处理收益率分布的偏度和峰度
            - 考虑多重假设检验的影响
        
        注意事项：
            - 需要准确估计试验次数以获得正确的DSR值
            - 收益率分布特征对DSR计算结果有重要影响
            - DSR较低不一定意味着策略无价值，可能需要更长的验证期
        
        Deflated Sharpe Ratio (DSR).

        Expresses the chance that the advertised strategy has a positive Sharpe ratio.

        If `var_sharpe` is None, is calculated based on all columns.
        If `nb_trials` is None, is set to the number of columns.
        """
        # 获取无风险收益率：如果未指定则使用默认配置
        if risk_free is None:
            risk_free = self.defaults['risk_free']
        
        # 获取自由度增量：如果未指定则使用默认配置
        if ddof is None:
            ddof = self.defaults['ddof']
        
        # 首先计算传统夏普比率
        sharpe_ratio = to_1d_array(self.sharpe_ratio(risk_free=risk_free))
        
        # 计算夏普比率的方差：如果未指定则基于所有列计算
        if var_sharpe is None:
            var_sharpe = np.var(sharpe_ratio, ddof=ddof)
        
        # 设置试验次数：如果未指定则使用数据的列数
        if nb_trials is None:
            nb_trials = self.wrapper.shape_2d[1]
        
        # 获取收益率数据并处理NaN值
        returns = to_2d_array(self.obj)
        nanmask = np.isnan(returns)
        if nanmask.any():
            returns = returns.copy()
            returns[nanmask] = 0.  # 将NaN值替换为0
        
        # 调用metrics模块计算紧缩夏普比率
        result = metrics.deflated_sharpe_ratio(
            est_sharpe=sharpe_ratio / np.sqrt(self.ann_factor),  # 调整为期间夏普比率
            var_sharpe=var_sharpe / self.ann_factor,             # 调整方差的年化影响
            nb_trials=nb_trials,                                 # 试验次数
            backtest_horizon=self.wrapper.shape_2d[0],          # 回测时间长度
            skew=skew(returns, axis=0, bias=bias),               # 收益率分布的偏度
            kurtosis=kurtosis(returns, axis=0, bias=bias)        # 收益率分布的峰度
        )
        
        # 设置包装参数，默认名称为'deflated_sharpe_ratio'
        wrap_kwargs = merge_dicts(dict(name_or_index='deflated_sharpe_ratio'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def downside_risk(self,
                      required_return: tp.Optional[float] = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算下行风险（Downside Risk）
        
        下行风险是一种专门衡量投资损失风险的指标，只考虑低于目标收益率的
        负面偏差。相比传统波动率将上行和下行波动同等对待，下行风险更符合
        投资者的风险感知，因为投资者通常只将低于预期的收益视为风险。
        
        计算公式：
            下行风险 = √[E(max(目标收益率 - 收益率, 0)²)] × √年化因子
            
        其中：
            - 只计算低于目标收益率的收益率偏差
            - 高于目标收益率的收益率不视为风险
            - 结果经过年化处理便于比较
        
        参数说明：
            required_return (tp.Optional[float]): 目标收益率（年化）
                - None: 使用默认配置中的required_return值
                - 0.0: 以零收益率为目标（常用设置）
                - 浮点数: 投资者的最低要求收益率
                - 可以设置为无风险利率或基准收益率
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'downside_risk'
        
        返回值：
            tp.MaybeSeries: 年化下行风险
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个下行风险
                - 数值越小表示下行风险越低
        
        使用示例：
            >>> # 基础下行风险计算
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=252))
            >>> downside = returns.vbt.returns.downside_risk(required_return=0.0)
            >>> print(f"下行风险: {downside:.2%}")
            >>> # 输出: 下行风险: 12.34%
            
            >>> # 不同目标收益率下的下行风险比较
            >>> dr_0 = returns.vbt.returns.downside_risk(required_return=0.0)    # 0%目标
            >>> dr_3 = returns.vbt.returns.downside_risk(required_return=0.03)   # 3%目标
            >>> dr_5 = returns.vbt.returns.downside_risk(required_return=0.05)   # 5%目标
            >>> print(f"0%目标: {dr_0:.2%}, 3%目标: {dr_3:.2%}, 5%目标: {dr_5:.2%}")
            
            >>> # 多策略下行风险分析
            >>> returns_df = pd.DataFrame({...})
            >>> downside_risks = returns_df.vbt.returns.downside_risk(required_return=0.03)
            >>> print(downside_risks.sort_values())  # 按下行风险升序排列
        
        应用场景：
            - **索提诺比率计算**：作为索提诺比率的分母
            - **风险预算**：基于下行风险的投资组合构建
            - **资产配置**：评估不同资产的下行风险暴露
            - **风险控制**：设置基于下行风险的止损策略
            - **绩效评估**：更准确地评估策略的风险特征
        
        与传统波动率的比较：
            - **风险感知**：下行风险更符合投资者的损失厌恶心理
            - **风险度量**：传统波动率惩罚上行波动，下行风险只关注损失
            - **实用性**：下行风险在熊市中更能反映真实风险
            - **适用性**：特别适合评估绝对收益策略和保本策略
        
        目标收益率设置策略：
            - **零收益率**（0%）：最基本的保本要求
            - **无风险利率**：超越无风险收益的基本要求
            - **通胀率**：维持购买力的基本要求
            - **基准收益率**：相对基准的下行风险评估
            - **个人目标**：基于投资者具体收益目标
        
        解读指南：
            - **低下行风险** (<5%)：策略下行保护能力强
            - **中等下行风险** (5%-15%)：可接受的下行风险水平
            - **高下行风险** (15%-25%)：需要关注的风险水平
            - **极高下行风险** (>25%)：需要谨慎对待的高风险策略
        
        技术实现：
            - 使用nb.downside_risk_nb进行高效计算
            - 自动处理年化转换
            - 只计算低于目标收益率的偏差
        
        数学性质：
            - 下行风险总是小于或等于标准差
            - 当收益率分布对称时，下行风险约等于标准差的1/√2倍
            - 当收益率分布左偏时，下行风险接近标准差
        
        注意事项：
            - 目标收益率的选择显著影响下行风险的计算结果
            - 样本期间的选择可能影响下行风险的代表性
            - 需要足够的历史数据才能获得稳定的估计
            - 下行风险假设历史模式在未来仍然有效
        
        See `vectorbt.returns.nb.downside_risk_nb`.
        """
        # 获取目标收益率：如果未指定则使用默认配置
        if required_return is None:
            required_return = self.defaults['required_return']
        
        # 调用Numba编译函数计算下行风险
        result = nb.downside_risk_nb(self.to_2d_array(), self.ann_factor, required_return)
        
        # 设置包装参数，默认名称为'downside_risk'
        wrap_kwargs = merge_dicts(dict(name_or_index='downside_risk'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_downside_risk(self,
                              window: tp.Optional[int] = None,
                              minp: tp.Optional[int] = None,
                              required_return: tp.Optional[float] = None,
                              wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口下行风险计算
        
        计算滚动窗口内的下行风险，提供时变的下行风险分析。通过观察下行风险
        在时间序列上的变化，可以识别策略下行保护能力的时间特征和市场环境
        适应性，为动态风险管理提供重要依据。
        
        计算原理：
            对于每个时间点t，计算窗口[t-window+1, t]内的下行风险：
            滚动下行风险[t] = √[窗口内低于目标收益率的偏差平方均值] × √年化因子
        
        参数说明：
            window (tp.Optional[int]): 滚动窗口大小
                - None: 使用默认配置中的window值
                - 整数: 指定窗口长度（时间点数）
                - 建议至少30个观测值以获得稳定的下行风险估计
            minp (tp.Optional[int]): 最小观测值数量
                - None: 使用默认配置中的minp值
                - 整数: 窗口内最少需要的有效观测值
            required_return (tp.Optional[float]): 目标收益率
                - None: 使用默认配置中的required_return值
                - 浮点数: 投资者的最低要求收益率（年化）
            wrap_kwargs (tp.KwargsLike): 包装参数
        
        返回值：
            tp.SeriesFrame: 滚动下行风险序列
                - 前window-1个值为NaN（数据不足）
                - 后续每个值为对应窗口的下行风险
        
        使用示例：
            >>> # 计算60天滚动下行风险
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=500))
            >>> rolling_dr = returns.vbt.returns.rolling_downside_risk(
            ...     window=60, required_return=0.0
            ... )
            >>> 
            >>> # 可视化下行风险时间变化
            >>> import matplotlib.pyplot as plt
            >>> fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            >>> 
            >>> # 上图：收益率序列
            >>> returns.plot(ax=ax1, title='日收益率', alpha=0.7)
            >>> ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            >>> 
            >>> # 中图：累计收益率
            >>> cum_returns = returns.vbt.returns.cumulative()
            >>> cum_returns.plot(ax=ax2, title='累计收益率')
            >>> 
            >>> # 下图：滚动下行风险
            >>> rolling_dr.plot(ax=ax3, title='60天滚动下行风险', color='red')
            >>> ax3.set_ylabel('下行风险')
            >>> plt.tight_layout()
            >>> plt.show()
            
            >>> # 下行风险异常检测
            >>> dr_threshold = rolling_dr.quantile(0.8)  # 80%分位数作为阈值
            >>> high_risk_periods = rolling_dr > dr_threshold
            >>> print(f"高下行风险期间占比: {high_risk_periods.mean():.1%}")
        
        应用场景：
            - **动态风险监控**：实时监控投资策略的下行风险变化
            - **风险预警系统**：当下行风险异常升高时触发预警
            - **投资择时**：基于下行风险水平进行投资时点选择
            - **对冲决策**：根据下行风险变化调整对冲策略
            - **资产配置**：在高下行风险期间调整资产配置
        
        市场环境分析：
            - **牛市**：下行风险通常较低且稳定
            - **熊市**：下行风险显著上升
            - **震荡市**：下行风险波动性较大
            - **危机期间**：下行风险急剧上升
        
        风险管理应用：
            - **仓位管理**：根据下行风险调整仓位大小
            - **止损设置**：基于滚动下行风险设置动态止损
            - **对冲比例**：根据下行风险变化调整对冲比例
            - **现金配置**：高下行风险期间提高现金比例
        
        技术实现：
            - 使用nb.rolling_downside_risk_nb进行高效计算
            - 自动处理窗口内的下行偏差计算
            - 支持最小观测值要求，提高结果可靠性
        
        注意事项：
            - 窗口大小影响下行风险估计的平滑度和敏感性
            - 市场极端事件可能导致下行风险出现异常值
            - 需要结合市场环境和策略特征进行解读
            - 目标收益率的选择影响下行风险的绝对水平
        
        Rolling version of `ReturnsAccessor.downside_risk`.
        """
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 获取目标收益率：如果未指定则使用默认配置
        if required_return is None:
            required_return = self.defaults['required_return']
        
        # 调用Numba编译函数计算滚动下行风险
        result = nb.rolling_downside_risk_nb(self.to_2d_array(), window, minp, self.ann_factor, required_return)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def sortino_ratio(self,
                      required_return: tp.Optional[float] = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算索提诺比率（Sortino Ratio）
        
        索提诺比率是一种改进的夏普比率，用下行风险代替标准差作为风险度量。
        这种修改更好地反映了投资者的真实风险偏好，因为投资者通常只将
        低于预期的收益视为风险，而不会将超额收益视为风险。
        
        计算公式：
            索提诺比率 = (年化收益率 - 目标收益率) / 下行风险
            
        其中：
            - 年化收益率：投资策略的年化表现
            - 目标收益率：投资者的最低要求收益率
            - 下行风险：只考虑低于目标收益率的波动率
        
        参数说明：
            required_return (tp.Optional[float]): 目标收益率（年化）
                - None: 使用默认配置中的required_return值
                - 0.0: 以零收益率为目标（常用设置）
                - 浮点数: 投资者的最低要求收益率
                - 建议设置为无风险利率或个人投资目标
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'sortino_ratio'
        
        返回值：
            tp.MaybeSeries: 索提诺比率
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个索提诺比率
                - 数值越高表示单位下行风险的超额收益越高
        
        使用示例：
            >>> # 基础索提诺比率计算
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=252))
            >>> sortino = returns.vbt.returns.sortino_ratio(required_return=0.0)
            >>> print(f"索提诺比率: {sortino:.2f}")
            >>> # 输出: 索提诺比率: 1.85
            
            >>> # 与夏普比率的对比分析
            >>> sharpe = returns.vbt.returns.sharpe_ratio(risk_free=0.03)
            >>> sortino_3 = returns.vbt.returns.sortino_ratio(required_return=0.03)
            >>> print(f"夏普比率: {sharpe:.2f}")
            >>> print(f"索提诺比率: {sortino_3:.2f}")
            >>> print(f"索提诺/夏普比率: {sortino_3/sharpe:.2f}")
            
            >>> # 多策略索提诺比率比较
            >>> returns_df = pd.DataFrame({
            ...     'Conservative': [...],  # 保守策略
            ...     'Balanced': [...],      # 平衡策略
            ...     'Aggressive': [...]     # 激进策略
            ... })
            >>> sortino_ratios = returns_df.vbt.returns.sortino_ratio(required_return=0.0)
            >>> sharpe_ratios = returns_df.vbt.returns.sharpe_ratio(risk_free=0.03)
            >>> 
            >>> comparison = pd.DataFrame({
            ...     'Sortino': sortino_ratios,
            ...     'Sharpe': sharpe_ratios,
            ...     'Sortino_Premium': sortino_ratios - sharpe_ratios
            ... })
            >>> print(comparison.sort_values('Sortino', ascending=False))
        
        应用场景：
            - **风险厌恶投资者**：更适合评估保守型投资策略
            - **绝对收益策略**：评估对冲基金和绝对收益产品
            - **退休规划**：评估养老金投资组合的风险调整收益
            - **资产配置**：构建下行风险可控的投资组合
            - **基金选择**：筛选下行保护能力强的基金产品
        
        指标解读：
            - **卓越表现** (>2.5)：优秀的下行风险控制能力
            - **优秀表现** (1.8-2.5)：良好的风险调整收益
            - **良好表现** (1.2-1.8)：可接受的下行风险管理
            - **一般表现** (0.8-1.2)：下行风险控制一般
            - **需要改进** (0.5-0.8)：下行风险偏高
            - **表现不佳** (<0.5)：下行风险控制能力差
        
        与夏普比率的比较：
            - **风险定义**：索提诺只考虑下行风险，夏普考虑全部波动
            - **投资理念**：索提诺更符合损失厌恶心理
            - **数值关系**：索提诺比率通常高于夏普比率
            - **适用场景**：索提诺更适合评估绝对收益策略
        
        索提诺比率溢价：
            索提诺比率通常高于夏普比率，差额称为"索提诺溢价"：
            - 溢价大：说明策略具有良好的下行保护能力
            - 溢价小：说明策略的上下行波动相对对称
            - 负溢价：说明策略可能存在左尾风险
        
        优势特点：
            - **风险感知更准确**：符合投资者的损失厌恶心理
            - **下行保护评估**：重点评估策略的下行保护能力
            - **绝对收益友好**：更适合评估绝对收益策略
            - **直观理解**：易于向投资者解释的风险调整指标
        
        技术实现：
            - 使用nb.sortino_ratio_nb进行高效计算
            - 集成年化收益率和下行风险的计算
            - 自动处理目标收益率的设置
        
        注意事项：
            - 目标收益率的选择显著影响索提诺比率的数值
            - 当下行风险为零时，比率为无穷大（实际返回NaN）
            - 需要足够的历史数据才能获得稳定的下行风险估计
            - 极端市场条件下指标可能出现异常值
        
        投资决策指导：
            - 高索提诺比率的策略适合风险厌恶投资者
            - 可以作为构建保守型投资组合的重要筛选指标
            - 结合夏普比率使用可以更全面地评估策略特征
        
        See `vectorbt.returns.nb.sortino_ratio_nb`.
        """
        # 获取目标收益率：如果未指定则使用默认配置
        if required_return is None:
            required_return = self.defaults['required_return']
        
        # 调用Numba编译函数计算索提诺比率
        result = nb.sortino_ratio_nb(self.to_2d_array(), self.ann_factor, required_return)
        
        # 设置包装参数，默认名称为'sortino_ratio'
        wrap_kwargs = merge_dicts(dict(name_or_index='sortino_ratio'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_sortino_ratio(self,
                              window: tp.Optional[int] = None,
                              minp: tp.Optional[int] = None,
                              required_return: tp.Optional[float] = None,
                              wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口索提诺比率计算
        
        计算滚动窗口内的索提诺比率，提供时变的下行风险调整绩效分析。
        通过观察索提诺比率的时间变化，可以评估策略在不同市场环境下的
        下行保护能力和风险调整表现的稳定性。
        
        计算原理：
            对于每个时间点t，计算窗口[t-window+1, t]内的索提诺比率：
            滚动索提诺比率[t] = (窗口年化收益率 - 目标收益率) / 窗口下行风险
        
        应用场景：
            - **动态绩效监控**：监控策略下行风险调整表现的时间变化
            - **市场环境适应性**：评估策略在不同市场条件下的表现
            - **风险管理决策**：基于滚动索提诺比率调整风险管理策略
        
        Rolling version of `ReturnsAccessor.sortino_ratio`.
        """
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 获取目标收益率：如果未指定则使用默认配置
        if required_return is None:
            required_return = self.defaults['required_return']
        
        # 调用Numba编译函数计算滚动索提诺比率
        result = nb.rolling_sortino_ratio_nb(self.to_2d_array(), window, minp, self.ann_factor, required_return)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def information_ratio(self,
                          benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                          ddof: tp.Optional[int] = None,
                          wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算信息比率（Information Ratio）
        
        信息比率是衡量主动投资管理技能的核心指标，表示单位跟踪误差下的
        超额收益。它是夏普比率在相对绩效评估中的应用，专门用于评估
        投资组合相对于基准的风险调整超额收益。
        
        计算公式：
            信息比率 = 超额收益的均值 / 超额收益的标准差
            
        其中：
            - 超额收益 = 投资组合收益率 - 基准收益率
            - 跟踪误差 = 超额收益的标准差
            - 信息比率 = 年化超额收益 / 年化跟踪误差
        
        参数说明：
            benchmark_rets (tp.Optional[tp.ArrayLike]): 基准收益率
                - None: 使用实例初始化时设置的benchmark_rets
                - ArrayLike: 指定基准收益率数据
                - 必须与投资组合收益率在时间上对齐
            ddof (tp.Optional[int]): 自由度增量
                - None: 使用默认配置中的ddof值
                - 1: 样本标准差（默认值，无偏估计）
                - 0: 总体标准差（有偏估计）
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'information_ratio'
        
        返回值：
            tp.MaybeSeries: 信息比率
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个信息比率
                - 数值越高表示主动管理技能越强
        
        使用示例：
            >>> # 基础信息比率计算
            >>> portfolio_returns = pd.Series([...])
            >>> benchmark_returns = pd.Series([...])  # 如沪深300指数收益率
            >>> info_ratio = portfolio_returns.vbt.returns.information_ratio(
            ...     benchmark_rets=benchmark_returns
            ... )
            >>> print(f"信息比率: {info_ratio:.2f}")
            >>> # 输出: 信息比率: 0.85
            
            >>> # 多基金信息比率比较
            >>> funds_returns = pd.DataFrame({
            ...     'Fund_A': [...],
            ...     'Fund_B': [...],
            ...     'Fund_C': [...]
            ... })
            >>> info_ratios = funds_returns.vbt.returns.information_ratio(
            ...     benchmark_rets=benchmark_returns
            ... )
            >>> print(info_ratios.sort_values(ascending=False))
            >>> # Fund_B    1.25
            >>> # Fund_A    0.85
            >>> # Fund_C    0.42
            
            >>> # 信息比率与跟踪误差分析
            >>> excess_returns = portfolio_returns - benchmark_returns
            >>> tracking_error = excess_returns.std() * np.sqrt(252)  # 年化跟踪误差
            >>> excess_return_ann = excess_returns.mean() * 252        # 年化超额收益
            >>> manual_ir = excess_return_ann / tracking_error
            >>> print(f"手动计算: {manual_ir:.2f}, 函数计算: {info_ratio:.2f}")
        
        应用场景：
            - **基金经理评估**：评估主动管理基金经理的投资技能
            - **基金选择**：筛选具有持续超额收益能力的基金
            - **绩效归因**：区分Beta收益和Alpha收益的贡献
            - **风险预算**：基于信息比率进行主动风险配置
            - **投资组合构建**：选择高信息比率的子策略
        
        指标解读：
            - **卓越表现** (>1.0)：优秀的主动管理技能，持续超越基准
            - **良好表现** (0.5-1.0)：具有一定的主动管理能力
            - **一般表现** (0.2-0.5)：主动管理效果有限
            - **需要改进** (0-0.2)：主动管理价值不明显
            - **表现不佳** (<0)：系统性跑输基准，可能存在问题
        
        理论基础：
            - **主动管理基本法则**：信息比率 = 信息系数 × √策略广度
            - **信息系数**：预测能力的相关系数
            - **策略广度**：独立投资决策的数量
            - **最优化理论**：信息比率是主动组合优化的目标函数
        
        与其他指标的关系：
            - **夏普比率**：当基准为无风险资产时，信息比率等于夏普比率
            - **跟踪误差**：信息比率的分母，衡量相对风险
            - **Alpha**：信息比率的分子，衡量超额收益
            - **Beta**：系统性风险暴露，与信息比率互补
        
        主动管理价值评估：
            - **信息比率 > 0.5**：主动管理费用可能合理
            - **信息比率 < 0.2**：建议考虑被动投资
            - **负信息比率**：主动管理可能损害价值
        
        技术实现：
            - 使用nb.information_ratio_nb进行高效计算
            - 自动处理基准收益率的广播和对齐
            - 支持多资产批量计算
        
        数据要求：
            - 投资组合和基准收益率必须时间对齐
            - 建议至少1年以上的历史数据
            - 收益率频率应该一致（日、周、月等）
        
        注意事项：
            - 基准的选择对信息比率有重要影响
            - 短期数据计算的信息比率波动较大
            - 需要考虑基准的代表性和可投资性
            - 极端市场条件下指标可能失真
        
        投资决策应用：
            - 基金筛选：选择信息比率持续为正的基金
            - 费用评估：评估主动管理费用的合理性
            - 风险预算：为高信息比率策略分配更多风险预算
            - 组合构建：结合多个高信息比率的子策略
        
        See `vectorbt.returns.nb.information_ratio_nb`.
        """
        # 获取基准收益率：如果未指定则使用实例设置的基准
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_rets
        
        # 获取自由度增量：如果未指定则使用默认配置
        if ddof is None:
            ddof = self.defaults['ddof']
        
        # 将基准收益率广播到与主收益率相同的维度
        benchmark_rets = broadcast_to(to_2d_array(benchmark_rets), to_2d_array(self.obj))
        
        # 调用Numba编译函数计算信息比率
        result = nb.information_ratio_nb(self.to_2d_array(), benchmark_rets, ddof)
        
        # 设置包装参数，默认名称为'information_ratio'
        wrap_kwargs = merge_dicts(dict(name_or_index='information_ratio'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_information_ratio(self,
                                  benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                                  window: tp.Optional[int] = None,
                                  minp: tp.Optional[int] = None,
                                  ddof: tp.Optional[int] = None,
                                  wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口信息比率计算
        
        计算滚动窗口内的信息比率，提供时变的主动管理技能分析。通过观察
        信息比率在时间序列上的变化，可以评估基金经理或投资策略在不同
        市场环境下的相对表现稳定性和主动管理能力的时间特征。
        
        计算原理：
            对于每个时间点t，计算窗口[t-window+1, t]内的信息比率：
            滚动信息比率[t] = 窗口超额收益均值 / 窗口超额收益标准差
        
        参数说明：
            benchmark_rets (tp.Optional[tp.ArrayLike]): 基准收益率
                - None: 使用实例初始化时设置的benchmark_rets
                - ArrayLike: 指定基准收益率数据
            window (tp.Optional[int]): 滚动窗口大小
                - None: 使用默认配置中的window值
                - 整数: 指定窗口长度（时间点数）
                - 建议至少60个观测值以获得稳定的信息比率
            minp (tp.Optional[int]): 最小观测值数量
                - None: 使用默认配置中的minp值
                - 整数: 窗口内最少需要的有效观测值
            ddof (tp.Optional[int]): 自由度增量
                - None: 使用默认配置中的ddof值
                - 1: 样本标准差（默认值）
            wrap_kwargs (tp.KwargsLike): 包装参数
        
        返回值：
            tp.SeriesFrame: 滚动信息比率序列
                - 前window-1个值为NaN（数据不足）
                - 后续每个值为对应窗口的信息比率
        
        使用示例：
            >>> # 计算252天滚动信息比率
            >>> portfolio_returns = pd.Series([...], index=pd.date_range('2020-01-01', periods=1000))
            >>> benchmark_returns = pd.Series([...], index=pd.date_range('2020-01-01', periods=1000))
            >>> rolling_ir = portfolio_returns.vbt.returns.rolling_information_ratio(
            ...     benchmark_rets=benchmark_returns, window=252
            ... )
            >>> 
            >>> # 可视化信息比率时间变化
            >>> import matplotlib.pyplot as plt
            >>> fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            >>> 
            >>> # 上图：累计收益率对比
            >>> portfolio_cum = portfolio_returns.vbt.returns.cumulative()
            >>> benchmark_cum = benchmark_returns.vbt.returns.cumulative()
            >>> portfolio_cum.plot(ax=ax1, label='投资组合', linewidth=2)
            >>> benchmark_cum.plot(ax=ax1, label='基准', linewidth=2, alpha=0.7)
            >>> ax1.legend()
            >>> ax1.set_title('累计收益率对比')
            >>> 
            >>> # 中图：超额收益
            >>> excess_returns = portfolio_returns - benchmark_returns
            >>> excess_cum = excess_returns.vbt.returns.cumulative()
            >>> excess_cum.plot(ax=ax2, color='green', linewidth=2)
            >>> ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            >>> ax2.set_title('累计超额收益')
            >>> 
            >>> # 下图：滚动信息比率
            >>> rolling_ir.plot(ax=ax3, color='red', linewidth=2)
            >>> ax3.axhline(y=0.5, color='orange', linestyle='--', label='良好阈值')
            >>> ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            >>> ax3.legend()
            >>> ax3.set_title('252天滚动信息比率')
            >>> plt.tight_layout()
            >>> plt.show()
            
            >>> # 信息比率稳定性分析
            >>> ir_positive_ratio = (rolling_ir > 0).mean()
            >>> ir_good_ratio = (rolling_ir > 0.5).mean()
            >>> print(f"信息比率为正的时间占比: {ir_positive_ratio:.1%}")
            >>> print(f"信息比率良好的时间占比: {ir_good_ratio:.1%}")
        
        应用场景：
            - **基金经理监控**：实时监控基金经理的主动管理能力
            - **投资决策**：基于滚动信息比率进行基金选择和替换
            - **风险管理**：当信息比率持续恶化时调整投资策略
            - **绩效评估**：评估不同市场环境下的相对表现
            - **费用合理性**：评估主动管理费用在不同时期的合理性
        
        市场环境分析：
            - **牛市**：主动管理难度增加，信息比率可能下降
            - **熊市**：优秀的主动管理能力更容易体现
            - **震荡市**：信息比率波动性较大
            - **危机期间**：考验真正的主动管理技能
        
        投资决策信号：
            - **持续为正**：继续持有或增加配置
            - **转为负值**：考虑减少配置或更换基金
            - **波动加剧**：可能需要调整投资策略
            - **趋势改善**：可能是增加配置的机会
        
        技术实现：
            - 使用nb.rolling_information_ratio_nb进行高效计算
            - 自动处理窗口内基准收益率的对齐
            - 支持最小观测值要求，提高结果可靠性
        
        质量控制：
            - **数据对齐检查**：确保投资组合和基准数据时间对齐
            - **异常值处理**：识别和处理极端信息比率值
            - **窗口大小优化**：根据策略特征选择合适的窗口大小
        
        注意事项：
            - 窗口大小影响信息比率的平滑度和敏感性
            - 基准选择的一致性对时间序列分析很重要
            - 市场结构变化可能影响信息比率的可比性
            - 需要结合绝对收益表现进行综合评估
        
        Rolling version of `ReturnsAccessor.information_ratio`.
        """
        # 获取基准收益率：如果未指定则使用实例设置的基准
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_rets
        
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 获取自由度增量：如果未指定则使用默认配置
        if ddof is None:
            ddof = self.defaults['ddof']
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 将基准收益率广播到与主收益率相同的维度
        benchmark_rets = broadcast_to(to_2d_array(benchmark_rets), to_2d_array(self.obj))
        
        # 调用Numba编译函数计算滚动信息比率
        result = nb.rolling_information_ratio_nb(self.to_2d_array(), window, minp, benchmark_rets, ddof)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def beta(self,
             benchmark_rets: tp.Optional[tp.ArrayLike] = None,
             wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算Beta系数（Beta Coefficient）
        
        Beta系数是资本资产定价模型（CAPM）的核心参数，衡量投资组合相对于
        市场基准的系统性风险暴露程度。Beta系数反映了投资组合收益率对
        市场收益率变化的敏感性，是投资风险管理和资产配置的重要指标。
        
        计算公式：
            Beta = Cov(投资组合收益率, 基准收益率) / Var(基准收益率)
            
        等价于：
            Beta = 投资组合与基准的相关系数 × (投资组合标准差 / 基准标准差)
        
        参数说明：
            benchmark_rets (tp.Optional[tp.ArrayLike]): 基准收益率
                - None: 使用实例初始化时设置的benchmark_rets
                - ArrayLike: 指定基准收益率数据（通常是市场指数）
                - 必须与投资组合收益率在时间上对齐
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'beta'
        
        返回值：
            tp.MaybeSeries: Beta系数
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个Beta系数
        
        Beta系数解读：
            - **Beta = 1.0**: 与市场风险完全一致
            - **Beta > 1.0**: 高Beta资产，风险高于市场，收益波动更大
            - **0 < Beta < 1.0**: 低Beta资产，风险低于市场，防御性强
            - **Beta = 0**: 与市场无相关性，独立于市场波动
            - **Beta < 0**: 与市场负相关，对冲属性
        
        使用示例：
            >>> # 基础Beta系数计算
            >>> stock_returns = pd.Series([...])
            >>> market_returns = pd.Series([...])  # 如沪深300指数收益率
            >>> beta = stock_returns.vbt.returns.beta(benchmark_rets=market_returns)
            >>> print(f"Beta系数: {beta:.2f}")
            >>> # 输出: Beta系数: 1.25
            
            >>> # 多股票Beta系数分析
            >>> stocks_returns = pd.DataFrame({
            ...     'Tech_Stock': [...],      # 科技股
            ...     'Utility_Stock': [...],   # 公用事业股
            ...     'Finance_Stock': [...]    # 金融股
            ... })
            >>> betas = stocks_returns.vbt.returns.beta(benchmark_rets=market_returns)
            >>> print(betas.sort_values(ascending=False))
            >>> # Tech_Stock      1.45  (高Beta，高风险)
            >>> # Finance_Stock   1.15  (略高于市场)
            >>> # Utility_Stock   0.65  (低Beta，防御性)
            
            >>> # Beta系数风险分类
            >>> for stock, beta_val in betas.items():
            ...     if beta_val > 1.2:
            ...         risk_type = "高风险成长型"
            ...     elif beta_val > 0.8:
            ...         risk_type = "市场风险型"
            ...     else:
            ...         risk_type = "防御型"
            ...     print(f"{stock}: {beta_val:.2f} ({risk_type})")
        
        应用场景：
            - **投资组合构建**：根据Beta调整资产配置比例
            - **风险管理**：评估系统性风险暴露
            - **CAPM模型**：计算预期收益率和资本成本
            - **对冲策略**：构建Beta中性的投资组合
            - **绩效归因**：区分Alpha和Beta对收益的贡献
        
        投资策略应用：
            - **高Beta策略**：牛市中配置高Beta资产获取超额收益
            - **低Beta策略**：熊市中配置低Beta资产降低损失
            - **Beta轮动**：根据市场环境调整Beta暴露
            - **Smart Beta**：基于Beta等因子的指数化投资
        
        理论基础：
            - **CAPM模型**：E(R) = Rf + Beta × (E(Rm) - Rf)
            - **系统性风险**：Beta衡量不可分散的市场风险
            - **证券市场线**：Beta是风险溢价的斜率
        
        技术实现：
            - 使用nb.beta_nb进行高效计算
            - 基于协方差和方差的回归计算
            - 自动处理基准收益率的广播和对齐
        
        注意事项：
            - Beta系数假设线性关系，实际可能存在非线性
            - 估计期间的选择影响Beta的稳定性
            - 基准选择应该具有代表性和可投资性
            - Beta系数在不同市场环境下可能发生变化
        
        See `vectorbt.returns.nb.beta_nb`.
        """
        # 获取基准收益率：如果未指定则使用实例设置的基准
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_rets
        
        # 将基准收益率广播到与主收益率相同的维度
        benchmark_rets = broadcast_to(to_2d_array(benchmark_rets), to_2d_array(self.obj))
        
        # 调用Numba编译函数计算Beta系数
        result = nb.beta_nb(self.to_2d_array(), benchmark_rets)
        
        # 设置包装参数，默认名称为'beta'
        wrap_kwargs = merge_dicts(dict(name_or_index='beta'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_beta(self,
                     benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                     window: tp.Optional[int] = None,
                     minp: tp.Optional[int] = None,
                     wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口Beta系数计算
        
        计算滚动窗口内的Beta系数，提供时变的系统性风险暴露分析。通过观察
        Beta系数在时间序列上的变化，可以识别投资组合风险特征的时间变化，
        为动态风险管理和资产配置调整提供重要依据。
        
        计算原理：
            对于每个时间点t，计算窗口[t-window+1, t]内的Beta系数：
            滚动Beta[t] = Cov(窗口内收益率, 窗口内基准收益率) / Var(窗口内基准收益率)
        
        应用场景：
            - **动态风险管理**：监控投资组合系统性风险的时间变化
            - **Beta择时**：根据Beta变化调整市场暴露程度
            - **风险归因**：分析不同时期的风险来源变化
            - **对冲决策**：基于滚动Beta调整对冲比例
        
        Rolling version of `ReturnsAccessor.beta`.
        """
        # 获取基准收益率：如果未指定则使用实例设置的基准
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_rets
        
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 将基准收益率广播到与主收益率相同的维度
        benchmark_rets = broadcast_to(to_2d_array(benchmark_rets), to_2d_array(self.obj))
        
        # 调用Numba编译函数计算滚动Beta系数
        result = nb.rolling_beta_nb(self.to_2d_array(), window, minp, benchmark_rets)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def alpha(self,
              benchmark_rets: tp.Optional[tp.ArrayLike] = None,
              risk_free: tp.Optional[float] = None,
              wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算Alpha系数（Alpha Coefficient）
        
        Alpha系数是衡量投资组合超额收益的核心指标，表示在调整系统性风险（Beta）
        后，投资组合相对于基准的超额表现。Alpha是CAPM模型的重要组成部分，
        代表投资经理的主动管理价值和选股择时能力。
        
        计算公式：
            Alpha = 投资组合年化收益率 - [无风险利率 + Beta × (基准年化收益率 - 无风险利率)]
            
        CAPM模型表示：
            E(Rp) = Rf + Beta × (E(Rm) - Rf) + Alpha
            
        其中：
            - E(Rp): 投资组合预期收益率
            - Rf: 无风险利率
            - E(Rm): 市场预期收益率
            - Alpha: 超额收益（Jensen's Alpha）
        
        参数说明：
            benchmark_rets (tp.Optional[tp.ArrayLike]): 基准收益率
                - None: 使用实例初始化时设置的benchmark_rets
                - ArrayLike: 指定基准收益率数据（通常是市场指数）
                - 必须与投资组合收益率在时间上对齐
            risk_free (tp.Optional[float]): 无风险收益率
                - None: 使用默认配置中的risk_free值
                - 浮点数: 指定无风险利率（年化）
                - 通常使用国债收益率作为代理
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'alpha'
        
        返回值：
            tp.MaybeSeries: Alpha系数（年化）
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个Alpha系数
        
        Alpha系数解读：
            - **Alpha > 0**: 正Alpha，投资组合跑赢风险调整基准
            - **Alpha = 0**: 投资组合表现符合CAPM预期
            - **Alpha < 0**: 负Alpha，投资组合跑输风险调整基准
        
        使用示例：
            >>> # 基础Alpha系数计算
            >>> portfolio_returns = pd.Series([...])
            >>> market_returns = pd.Series([...])  # 如沪深300指数收益率
            >>> alpha = portfolio_returns.vbt.returns.alpha(
            ...     benchmark_rets=market_returns, risk_free=0.03
            ... )
            >>> print(f"Alpha系数: {alpha:.2%}")
            >>> # 输出: Alpha系数: 2.50%
            
            >>> # 多基金Alpha分析
            >>> funds_returns = pd.DataFrame({
            ...     'Active_Fund_A': [...],
            ...     'Active_Fund_B': [...],
            ...     'Index_Fund': [...]
            ... })
            >>> alphas = funds_returns.vbt.returns.alpha(
            ...     benchmark_rets=market_returns, risk_free=0.03
            ... )
            >>> print(alphas.sort_values(ascending=False))
            >>> # Active_Fund_A    0.0350  (3.5% Alpha)
            >>> # Active_Fund_B    0.0180  (1.8% Alpha)
            >>> # Index_Fund      -0.0020  (-0.2% Alpha)
            
            >>> # Alpha与费用比较分析
            >>> management_fees = pd.Series({
            ...     'Active_Fund_A': 0.015,  # 1.5% 管理费
            ...     'Active_Fund_B': 0.012,  # 1.2% 管理费
            ...     'Index_Fund': 0.003     # 0.3% 管理费
            ... })
            >>> net_alpha = alphas - management_fees
            >>> comparison = pd.DataFrame({
            ...     'Gross_Alpha': alphas,
            ...     'Management_Fee': management_fees,
            ...     'Net_Alpha': net_alpha,
            ...     'Value_Added': net_alpha > 0
            ... })
            >>> print(comparison)
        
        应用场景：
            - **基金经理评估**：评估主动管理基金经理的投资技能
            - **投资决策**：选择具有持续正Alpha的投资产品
            - **绩效归因**：区分Beta收益和Alpha收益的贡献
            - **费用合理性评估**：评估主动管理费用是否合理
            - **投资组合构建**：寻找和配置高Alpha的投资机会
        
        理论意义：
            - **市场有效性检验**：持续正Alpha挑战市场有效性假设
            - **主动管理价值**：Alpha衡量主动管理相对被动投资的价值
            - **技能vs运气**：统计显著的Alpha表明真实的投资技能
            - **风险调整收益**：Alpha提供风险调整后的超额收益度量
        
        实际应用考虑：
            - **Alpha的持续性**：历史Alpha不保证未来表现
            - **统计显著性**：需要足够样本量和时间跨度
            - **基准适当性**：基准选择影响Alpha的计算和解释
            - **费用影响**：需要区分总Alpha和净Alpha
        
        投资决策指导：
            - **正Alpha且显著**：值得投资和持有的标的
            - **Alpha不显著**：考虑被动投资策略
            - **负Alpha持续**：需要重新评估投资策略
            - **Alpha衰减**：可能需要调整或替换投资标的
        
        技术实现：
            - 使用nb.alpha_nb进行高效计算
            - 集成Beta计算和CAPM模型
            - 自动处理年化转换和风险调整
        
        注意事项：
            - Alpha计算依赖于Beta的准确估计
            - 基准选择对Alpha计算结果有重要影响
            - 需要足够长的历史数据才能获得可靠估计
            - 市场环境变化可能影响Alpha的稳定性
        
        See `vectorbt.returns.nb.alpha_nb`.
        """
        # 获取基准收益率：如果未指定则使用实例设置的基准
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_rets
        
        # 获取无风险收益率：如果未指定则使用默认配置
        if risk_free is None:
            risk_free = self.defaults['risk_free']
        
        # 将基准收益率广播到与主收益率相同的维度
        benchmark_rets = broadcast_to(to_2d_array(benchmark_rets), to_2d_array(self.obj))
        
        # 调用Numba编译函数计算Alpha系数
        result = nb.alpha_nb(self.to_2d_array(), benchmark_rets, self.ann_factor, risk_free)
        
        # 设置包装参数，默认名称为'alpha'
        wrap_kwargs = merge_dicts(dict(name_or_index='alpha'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_alpha(self,
                      benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                      window: tp.Optional[int] = None,
                      minp: tp.Optional[int] = None,
                      risk_free: tp.Optional[float] = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口Alpha系数计算
        
        计算滚动窗口内的Alpha系数，提供时变的超额收益分析。通过观察
        Alpha系数在时间序列上的变化，可以评估投资经理或策略在不同
        市场环境下的主动管理技能和超额收益创造能力的时间特征。
        
        计算原理：
            对于每个时间点t，计算窗口[t-window+1, t]内的Alpha系数：
            滚动Alpha[t] = 窗口年化收益率 - [无风险利率 + 窗口Beta × (基准年化收益率 - 无风险利率)]
        
        应用场景：
            - **基金经理监控**：实时监控基金经理的主动管理能力变化
            - **投资决策**：基于滚动Alpha进行投资标的选择和替换
            - **绩效评估**：评估不同市场环境下的超额收益创造能力
            - **风险管理**：当Alpha持续为负时及时调整投资策略
        
        Rolling version of `ReturnsAccessor.alpha`.
        """
        # 获取基准收益率：如果未指定则使用实例设置的基准
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_rets
        
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 获取无风险收益率：如果未指定则使用默认配置
        if risk_free is None:
            risk_free = self.defaults['risk_free']
        
        # 将基准收益率广播到与主收益率相同的维度
        benchmark_rets = broadcast_to(to_2d_array(benchmark_rets), to_2d_array(self.obj))
        
        # 调用Numba编译函数计算滚动Alpha系数
        result = nb.rolling_alpha_nb(self.to_2d_array(), window, minp, benchmark_rets, self.ann_factor, risk_free)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def tail_ratio(self, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """See `vectorbt.returns.nb.tail_ratio_nb`."""
        result = nb.tail_ratio_nb(self.to_2d_array())
        wrap_kwargs = merge_dicts(dict(name_or_index='tail_ratio'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_tail_ratio(self,
                           window: tp.Optional[int] = None,
                           minp: tp.Optional[int] = None,
                           wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.tail_ratio`."""
        if window is None:
            window = self.defaults['window']
        if minp is None:
            minp = self.defaults['minp']
        result = nb.rolling_tail_ratio_nb(self.to_2d_array(), window, minp)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def common_sense_ratio(self, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Common Sense Ratio."""
        result = to_1d_array(self.tail_ratio()) * (1 + to_1d_array(self.annualized()))
        wrap_kwargs = merge_dicts(dict(name_or_index='common_sense_ratio'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_common_sense_ratio(self,
                                   window: tp.Optional[int] = None,
                                   minp: tp.Optional[int] = None,
                                   wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.common_sense_ratio`."""
        if window is None:
            window = self.defaults['window']
        if minp is None:
            minp = self.defaults['minp']
        rolling_tail_ratio = to_2d_array(self.rolling_tail_ratio(window, minp=minp))
        rolling_annualized = to_2d_array(self.rolling_annualized(window, minp=minp))
        result = rolling_tail_ratio * (1 + rolling_annualized)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def value_at_risk(self,
                      cutoff: tp.Optional[float] = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算风险价值（Value at Risk, VaR）
        
        VaR是量化投资风险的核心指标，表示在给定置信水平下，投资组合在
        特定时间内可能遭受的最大损失。VaR广泛应用于风险管理、监管合规
        和投资决策，是现代风险管理体系的基石。
        
        计算方法：
            基于历史收益率分布的经验分位数方法：
            VaR(α) = -Percentile(收益率分布, α)
            
        其中α为置信水平（如5%），VaR值为正数表示损失金额。
        
        参数说明：
            cutoff (tp.Optional[float]): 置信水平（尾部概率）
                - None: 使用默认配置中的cutoff值
                - 0.05: 95%置信水平（默认值，5%尾部风险）
                - 0.01: 99%置信水平（1%尾部风险）
                - 0.10: 90%置信水平（10%尾部风险）
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'value_at_risk'
        
        返回值：
            tp.MaybeSeries: VaR值（正数表示潜在损失）
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个VaR值
        
        使用示例：
            >>> # 基础VaR计算
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=252))
            >>> var_5 = returns.vbt.returns.value_at_risk(cutoff=0.05)
            >>> print(f"95%置信水平VaR: {var_5:.2%}")
            >>> # 输出: 95%置信水平VaR: 3.45%
            
            >>> # 不同置信水平的VaR比较
            >>> var_1 = returns.vbt.returns.value_at_risk(cutoff=0.01)   # 99%置信
            >>> var_5 = returns.vbt.returns.value_at_risk(cutoff=0.05)   # 95%置信
            >>> var_10 = returns.vbt.returns.value_at_risk(cutoff=0.10)  # 90%置信
            >>> print(f"99%VaR: {var_1:.2%}, 95%VaR: {var_5:.2%}, 90%VaR: {var_10:.2%}")
            
            >>> # 多资产VaR分析
            >>> portfolio_returns = pd.DataFrame({
            ...     'Stock_Portfolio': [...],
            ...     'Bond_Portfolio': [...],
            ...     'Mixed_Portfolio': [...]
            ... })
            >>> vars_5 = portfolio_returns.vbt.returns.value_at_risk(cutoff=0.05)
            >>> print(vars_5.sort_values(ascending=False))
            >>> # Stock_Portfolio    0.0456  (高风险)
            >>> # Mixed_Portfolio    0.0234  (中等风险)
            >>> # Bond_Portfolio     0.0123  (低风险)
        
        应用场景：
            - **风险限额管理**：设置投资组合的最大风险暴露限制
            - **监管合规**：满足巴塞尔协议等监管要求
            - **资本充足率**：确定风险资本缓冲需求
            - **投资决策**：评估不同投资选择的风险水平
            - **风险报告**：向投资者和监管机构报告风险状况
        
        VaR解读指南：
            - **低VaR** (<1%)：低风险投资，如债券或货币基金
            - **中等VaR** (1%-3%)：平衡型投资，如混合基金
            - **高VaR** (3%-5%)：高风险投资，如股票基金
            - **极高VaR** (>5%)：极高风险投资，如杠杆产品
        
        优势特点：
            - **直观易懂**：以货币或百分比形式表达损失
            - **监管认可**：被广泛用于金融监管
            - **风险聚合**：可以计算投资组合整体VaR
            - **决策支持**：为风险管理决策提供量化依据
        
        局限性：
            - **尾部风险**：VaR不能描述超过置信水平的极端损失
            - **分布假设**：基于历史分布，可能不适用于未来
            - **相关性变化**：危机期间相关性上升，VaR可能低估风险
            - **模型风险**：计算方法选择影响结果准确性
        
        技术实现：
            - 使用nb.value_at_risk_nb进行高效计算
            - 基于经验分布的分位数方法
            - 自动处理数据排序和分位数计算
        
        风险管理应用：
            - **止损设置**：基于VaR设置止损点
            - **仓位控制**：根据VaR调整仓位大小
            - **资产配置**：在VaR约束下优化资产配置
            - **压力测试**：结合情景分析进行压力测试
        
        注意事项：
            - VaR假设历史收益率分布代表未来风险
            - 样本期间的选择影响VaR的准确性
            - 需要结合CVaR等指标进行全面风险评估
            - 极端市场条件下VaR可能严重低估风险
        
        See `vectorbt.returns.nb.value_at_risk_nb`.
        """
        # 获取置信水平：如果未指定则使用默认配置
        if cutoff is None:
            cutoff = self.defaults['cutoff']
        
        # 调用Numba编译函数计算VaR
        result = nb.value_at_risk_nb(self.to_2d_array(), cutoff)
        
        # 设置包装参数，默认名称为'value_at_risk'
        wrap_kwargs = merge_dicts(dict(name_or_index='value_at_risk'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_value_at_risk(self,
                              window: tp.Optional[int] = None,
                              minp: tp.Optional[int] = None,
                              cutoff: tp.Optional[float] = None,
                              wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.value_at_risk`."""
        if window is None:
            window = self.defaults['window']
        if minp is None:
            minp = self.defaults['minp']
        if cutoff is None:
            cutoff = self.defaults['cutoff']
        result = nb.rolling_value_at_risk_nb(self.to_2d_array(), window, minp, cutoff)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def cond_value_at_risk(self,
                           cutoff: tp.Optional[float] = None,
                           wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算条件风险价值（Conditional Value at Risk, CVaR）
        
        CVaR，也称为期望短缺（Expected Shortfall, ES），是VaR的重要补充指标，
        衡量超过VaR阈值的条件期望损失。CVaR提供了比VaR更全面的尾部风险信息，
        特别关注极端损失情况，是现代风险管理的重要工具。
        
        计算公式：
            CVaR(α) = E[损失 | 损失 > VaR(α)]
            
        即：CVaR是超过VaR阈值的所有损失的平均值。
        
        参数说明：
            cutoff (tp.Optional[float]): 置信水平（尾部概率）
                - None: 使用默认配置中的cutoff值
                - 0.05: 95%置信水平（默认值，5%尾部风险）
                - 0.01: 99%置信水平（1%尾部风险）
                - 0.10: 90%置信水平（10%尾部风险）
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'cond_value_at_risk'
        
        返回值：
            tp.MaybeSeries: CVaR值（正数表示条件期望损失）
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个CVaR值
                - CVaR值总是大于或等于对应的VaR值
        
        使用示例：
            >>> # 基础CVaR计算
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=252))
            >>> cvar_5 = returns.vbt.returns.cond_value_at_risk(cutoff=0.05)
            >>> var_5 = returns.vbt.returns.value_at_risk(cutoff=0.05)
            >>> print(f"95%置信水平 - VaR: {var_5:.2%}, CVaR: {cvar_5:.2%}")
            >>> # 输出: 95%置信水平 - VaR: 3.45%, CVaR: 4.82%
            
            >>> # VaR与CVaR的比较分析
            >>> risk_metrics = pd.DataFrame({
            ...     'VaR_95': returns.vbt.returns.value_at_risk(cutoff=0.05),
            ...     'CVaR_95': returns.vbt.returns.cond_value_at_risk(cutoff=0.05),
            ...     'VaR_99': returns.vbt.returns.value_at_risk(cutoff=0.01),
            ...     'CVaR_99': returns.vbt.returns.cond_value_at_risk(cutoff=0.01)
            ... }, index=['Risk_Metrics'])
            >>> risk_metrics['CVaR/VaR_95'] = risk_metrics['CVaR_95'] / risk_metrics['VaR_95']
            >>> risk_metrics['CVaR/VaR_99'] = risk_metrics['CVaR_99'] / risk_metrics['VaR_99']
            >>> print(risk_metrics)
            
            >>> # 多资产尾部风险比较
            >>> portfolio_returns = pd.DataFrame({
            ...     'Growth_Fund': [...],
            ...     'Value_Fund': [...],
            ...     'Bond_Fund': [...]
            ... })
            >>> cvars = portfolio_returns.vbt.returns.cond_value_at_risk(cutoff=0.05)
            >>> vars = portfolio_returns.vbt.returns.value_at_risk(cutoff=0.05)
            >>> tail_risk_comparison = pd.DataFrame({
            ...     'VaR': vars,
            ...     'CVaR': cvars,
            ...     'Tail_Risk_Premium': cvars - vars,
            ...     'CVaR_VaR_Ratio': cvars / vars
            ... })
            >>> print(tail_risk_comparison.sort_values('CVaR', ascending=False))
        
        应用场景：
            - **尾部风险管理**：评估和控制极端损失风险
            - **资本配置**：基于CVaR进行风险资本分配
            - **投资组合优化**：最小化CVaR的投资组合构建
            - **监管合规**：满足更严格的风险管理要求
            - **压力测试**：评估极端市场条件下的损失
        
        CVaR相对VaR的优势：
            - **尾部敏感**：CVaR考虑所有超过VaR的损失，不仅仅是分位点
            - **风险一致**：CVaR满足风险度量的一致性公理
            - **优化友好**：CVaR是凸函数，便于优化求解
            - **极端关注**：更好地反映极端市场条件下的风险
        
        CVaR/VaR比率解读：
            - **比率接近1**：收益率分布尾部较薄，极端风险有限
            - **比率>1.2**：存在显著的尾部风险，需要额外关注
            - **比率>1.5**：尾部风险很高，可能存在厚尾分布
            - **比率>2.0**：极端尾部风险，需要特别的风险管理措施
        
        技术实现：
            - 使用nb.cond_value_at_risk_nb进行高效计算
            - 基于经验分布的条件期望计算
            - 自动处理尾部数据的筛选和平均
        
        风险管理策略：
            - **CVaR限制**：设置投资组合的CVaR上限
            - **尾部对冲**：针对CVaR风险进行专门对冲
            - **动态调整**：根据CVaR变化调整投资策略
            - **情景规划**：基于CVaR进行极端情景规划
        
        投资决策指导：
            - **低CVaR**：适合风险厌恶投资者
            - **高CVaR/VaR比率**：需要谨慎评估尾部风险
            - **CVaR趋势**：关注CVaR的时间变化趋势
            - **相对比较**：比较不同投资选择的CVaR水平
        
        注意事项：
            - CVaR基于历史数据，可能不反映未来风险
            - 样本量不足时CVaR估计可能不稳定
            - 需要结合其他风险指标进行综合评估
            - 极端市场条件下历史CVaR可能失效
        
        See `vectorbt.returns.nb.cond_value_at_risk_nb`.
        """
        # 获取置信水平：如果未指定则使用默认配置
        if cutoff is None:
            cutoff = self.defaults['cutoff']
        
        # 调用Numba编译函数计算CVaR
        result = nb.cond_value_at_risk_nb(self.to_2d_array(), cutoff)
        
        # 设置包装参数，默认名称为'cond_value_at_risk'
        wrap_kwargs = merge_dicts(dict(name_or_index='cond_value_at_risk'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_cond_value_at_risk(self,
                                   window: tp.Optional[int] = None,
                                   minp: tp.Optional[int] = None,
                                   cutoff: tp.Optional[float] = None,
                                   wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口条件风险价值计算
        
        计算滚动窗口内的条件风险价值（CVaR），提供时变的尾部风险分析。
        通过观察CVaR在时间序列上的变化，可以动态监控投资组合的极端损失
        风险，特别是在市场波动和危机期间的尾部风险暴露。
        
        计算原理：
            对于每个时间点t，计算窗口[t-window+1, t]内的CVaR：
            滚动CVaR[t] = E[损失 | 损失 > 窗口VaR[t]]
        
        参数说明：
            window (tp.Optional[int]): 滚动窗口大小
                - None: 使用默认配置中的window值
                - 整数: 指定窗口长度（时间点数）
                - 建议至少100个观测值以获得稳定的CVaR估计
            minp (tp.Optional[int]): 最小观测值数量
                - None: 使用默认配置中的minp值
                - 整数: 窗口内最少需要的有效观测值
            cutoff (tp.Optional[float]): 置信水平（尾部概率）
                - None: 使用默认配置中的cutoff值
                - 0.05: 95%置信水平（默认值）
                - 0.01: 99%置信水平（更严格的尾部风险）
            wrap_kwargs (tp.KwargsLike): 包装参数
        
        返回值：
            tp.SeriesFrame: 滚动CVaR序列
                - 前window-1个值为NaN（数据不足）
                - 后续每个值为对应窗口的CVaR
        
        应用场景：
            - **尾部风险监控**：实时监控极端市场条件下的潜在损失
            - **危机预警**：识别尾部风险急剧上升的时期
            - **风险管理**：基于CVaR变化调整风险管理策略
            - **资本配置**：动态调整风险资本缓冲
            - **压力测试**：评估不同时期的极端风险承受能力
        
        Rolling version of `ReturnsAccessor.cond_value_at_risk`.
        """
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 获取置信水平：如果未指定则使用默认配置
        if cutoff is None:
            cutoff = self.defaults['cutoff']
        
        # 调用Numba编译函数计算滚动CVaR
        result = nb.rolling_cond_value_at_risk_nb(self.to_2d_array(), window, minp, cutoff)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def capture(self,
                benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算捕获比率（Capture Ratio）
        
        捕获比率衡量投资组合相对于基准的收益捕获能力，表示投资组合收益率
        与基准收益率的比值。该指标反映了投资策略在整个市场周期中相对于
        基准的表现效率，是评估主动管理效果的重要工具。
        
        计算公式：
            捕获比率 = 投资组合年化收益率 / 基准年化收益率
        
        参数说明：
            benchmark_rets (tp.Optional[tp.ArrayLike]): 基准收益率
                - None: 使用实例初始化时设置的benchmark_rets
                - ArrayLike: 指定基准收益率数据
                - 必须与投资组合收益率在时间上对齐
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'capture'
        
        返回值：
            tp.MaybeSeries: 捕获比率
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个捕获比率
        
        捕获比率解读：
            - **比率 > 1.0**: 投资组合跑赢基准，表现优于市场
            - **比率 = 1.0**: 投资组合与基准表现一致
            - **比率 < 1.0**: 投资组合跑输基准，表现逊于市场
            - **比率 < 0**: 投资组合与基准方向相反（对冲特征）
        
        使用示例：
            >>> # 基础捕获比率计算
            >>> portfolio_returns = pd.Series([...])
            >>> market_returns = pd.Series([...])  # 如沪深300指数收益率
            >>> capture_ratio = portfolio_returns.vbt.returns.capture(
            ...     benchmark_rets=market_returns
            ... )
            >>> print(f"捕获比率: {capture_ratio:.2f}")
            >>> # 输出: 捕获比率: 1.15 (超越基准15%)
            
            >>> # 多策略捕获比率比较
            >>> strategies_returns = pd.DataFrame({
            ...     'Growth_Strategy': [...],      # 成长策略
            ...     'Value_Strategy': [...],       # 价值策略
            ...     'Momentum_Strategy': [...],    # 动量策略
            ...     'Mean_Reversion': [...]        # 均值回归策略
            ... })
            >>> capture_ratios = strategies_returns.vbt.returns.capture(
            ...     benchmark_rets=market_returns
            ... )
            >>> print(capture_ratios.sort_values(ascending=False))
            >>> # Momentum_Strategy     1.28  (最佳表现)
            >>> # Growth_Strategy      1.15  (良好表现)
            >>> # Value_Strategy       0.92  (略逊基准)
            >>> # Mean_Reversion       0.88  (表现较差)
        
        应用场景：
            - **策略评估**：评估投资策略的整体表现效率
            - **基金选择**：比较不同基金的市场捕获能力
            - **绩效归因**：分析超额收益的来源和稳定性
            - **投资决策**：选择具有持续捕获优势的投资标的
            - **组合构建**：配置高捕获比率的策略组合
        
        与其他指标的关系：
            - **上行捕获比率**：牛市中的捕获能力
            - **下行捕获比率**：熊市中的风险控制能力
            - **信息比率**：风险调整后的相对表现
            - **Alpha系数**：绝对超额收益能力
        
        投资策略分类：
            - **进攻型策略** (捕获比率 > 1.2)：追求超额收益，风险较高
            - **平衡型策略** (捕获比率 0.9-1.2)：追求稳健表现
            - **防御型策略** (捕获比率 0.7-0.9)：注重风险控制
            - **保守型策略** (捕获比率 < 0.7)：追求资本保值
        
        技术实现：
            - 使用nb.capture_nb进行高效计算
            - 基于年化收益率的比值计算
            - 自动处理基准收益率的广播和对齐
        
        注意事项：
            - 捕获比率不考虑风险因素，需要结合风险指标分析
            - 基准的选择对捕获比率有重要影响
            - 负收益环境下捕获比率的解释需要谨慎
            - 应该结合上行和下行捕获比率进行综合评估
        
        See `vectorbt.returns.nb.capture_nb`.
        """
        # 获取基准收益率：如果未指定则使用实例设置的基准
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_rets
        
        # 将基准收益率广播到与主收益率相同的维度
        benchmark_rets = broadcast_to(to_2d_array(benchmark_rets), to_2d_array(self.obj))
        
        # 调用Numba编译函数计算捕获比率
        result = nb.capture_nb(self.to_2d_array(), benchmark_rets, self.ann_factor)
        
        # 设置包装参数，默认名称为'capture'
        wrap_kwargs = merge_dicts(dict(name_or_index='capture'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_capture(self,
                        benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                        window: tp.Optional[int] = None,
                        minp: tp.Optional[int] = None,
                        wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口捕获比率计算
        
        计算滚动窗口内的捕获比率，提供时变的相对表现分析。通过观察
        捕获比率在时间序列上的变化，可以评估投资策略在不同市场环境
        下相对于基准的表现稳定性和一致性。
        
        计算原理：
            对于每个时间点t，计算窗口[t-window+1, t]内的捕获比率：
            滚动捕获比率[t] = 窗口投资组合年化收益率 / 窗口基准年化收益率
        
        应用场景：
            - **表现稳定性分析**：评估策略相对表现的时间稳定性
            - **市场适应性**：分析策略在不同市场环境下的适应能力
            - **动态调整**：基于滚动捕获比率进行策略动态调整
            - **风险监控**：监控相对表现的变化趋势
        
        Rolling version of `ReturnsAccessor.capture`.
        """
        # 获取基准收益率：如果未指定则使用实例设置的基准
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_rets
        
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 将基准收益率广播到与主收益率相同的维度
        benchmark_rets = broadcast_to(to_2d_array(benchmark_rets), to_2d_array(self.obj))
        
        # 调用Numba编译函数计算滚动捕获比率
        result = nb.rolling_capture_nb(self.to_2d_array(), window, minp, benchmark_rets, self.ann_factor)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def up_capture(self,
                   benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                   wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算上行捕获比率（Up Capture Ratio）
        
        上行捕获比率衡量投资组合在市场上涨期间相对于基准的收益捕获能力。
        该指标专门分析牛市或上涨行情中的表现，反映投资策略在有利市场
        环境下的收益获取效率，是评估进攻性投资能力的重要指标。
        
        计算公式：
            上行捕获比率 = 基准上涨期间投资组合年化收益率 / 基准上涨期间基准年化收益率
            
        其中只考虑基准收益率为正的时间段。
        
        参数说明：
            benchmark_rets (tp.Optional[tp.ArrayLike]): 基准收益率
                - None: 使用实例初始化时设置的benchmark_rets
                - ArrayLike: 指定基准收益率数据
                - 必须与投资组合收益率在时间上对齐
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'up_capture'
        
        返回值：
            tp.MaybeSeries: 上行捕获比率
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个上行捕获比率
        
        上行捕获比率解读：
            - **比率 > 1.0**: 牛市中表现优于基准，进攻能力强
            - **比率 = 1.0**: 牛市中与基准表现一致
            - **比率 < 1.0**: 牛市中表现逊于基准，进攻能力弱
            - **比率越高**: 牛市收益捕获能力越强
        
        使用示例：
            >>> # 基础上行捕获比率计算
            >>> portfolio_returns = pd.Series([...])
            >>> market_returns = pd.Series([...])  # 如沪深300指数收益率
            >>> up_capture = portfolio_returns.vbt.returns.up_capture(
            ...     benchmark_rets=market_returns
            ... )
            >>> print(f"上行捕获比率: {up_capture:.2f}")
            >>> # 输出: 上行捕获比率: 1.25 (牛市超越基准25%)
            
            >>> # 上行与下行捕获比率对比分析
            >>> up_capture = portfolio_returns.vbt.returns.up_capture(
            ...     benchmark_rets=market_returns
            ... )
            >>> down_capture = portfolio_returns.vbt.returns.down_capture(
            ...     benchmark_rets=market_returns
            ... )
            >>> 
            >>> capture_analysis = pd.DataFrame({
            ...     'Up_Capture': up_capture,
            ...     'Down_Capture': down_capture,
            ...     'Capture_Ratio': up_capture / down_capture,  # 上行下行比
            ...     'Strategy_Type': '进攻型' if up_capture > 1.1 else '平衡型'
            ... })
            >>> print(capture_analysis)
        
        应用场景：
            - **牛市策略评估**：评估策略在牛市中的收益获取能力
            - **进攻性分析**：分析投资策略的进攻性特征
            - **策略分类**：区分进攻型、平衡型和防御型策略
            - **市场择时**：评估在不同市场环境下的表现特征
            - **组合构建**：根据市场预期配置不同捕获特征的策略
        
        策略特征分析：
            - **高上行捕获** (>1.2): 典型的成长型或动量策略
            - **中等上行捕获** (0.9-1.2): 平衡型策略
            - **低上行捕获** (<0.9): 防御型或价值策略
            - **极高上行捕获** (>1.5): 高Beta或杠杆策略
        
        与下行捕获比率的组合分析：
            - **高上行，低下行**: 理想的进攻防守兼备
            - **高上行，高下行**: 高Beta特征，风险较大
            - **低上行，低下行**: 防御型特征，收益有限
            - **低上行，高下行**: 不理想的风险收益特征
        
        投资决策指导：
            - **牛市配置**: 选择高上行捕获比率的策略
            - **熊市规避**: 关注下行捕获比率，控制风险
            - **策略轮动**: 根据市场环境在不同捕获特征间轮动
            - **风险管理**: 平衡上行收益和下行风险
        
        技术实现：
            - 使用nb.up_capture_nb进行高效计算
            - 仅考虑基准收益率为正的时间段
            - 基于条件年化收益率比值计算
        
        注意事项：
            - 上行捕获比率需要与下行捕获比率结合分析
            - 基准上涨期间的定义影响计算结果
            - 样本期间的选择对结果有重要影响
            - 应该结合整体风险水平进行评估
        
        See `vectorbt.returns.nb.up_capture_nb`.
        """
        # 获取基准收益率：如果未指定则使用实例设置的基准
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_rets
        
        # 将基准收益率广播到与主收益率相同的维度
        benchmark_rets = broadcast_to(to_2d_array(benchmark_rets), to_2d_array(self.obj))
        
        # 调用Numba编译函数计算上行捕获比率
        result = nb.up_capture_nb(self.to_2d_array(), benchmark_rets, self.ann_factor)
        
        # 设置包装参数，默认名称为'up_capture'
        wrap_kwargs = merge_dicts(dict(name_or_index='up_capture'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_up_capture(self,
                           benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                           window: tp.Optional[int] = None,
                           minp: tp.Optional[int] = None,
                           wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口上行捕获比率计算
        
        计算滚动窗口内的上行捕获比率，提供时变的牛市表现分析。
        通过观察上行捕获比率的时间变化，可以评估投资策略在不同
        时期牛市环境下的收益获取能力变化。
        
        应用场景：
            - **动态进攻性分析**：监控策略进攻能力的时间变化
            - **牛市适应性**：评估在不同牛市环境下的表现
            - **策略调整**：基于上行捕获变化调整投资策略
            - **市场择时**：识别最佳的进攻时机
        
        Rolling version of `ReturnsAccessor.up_capture`.
        """
        # 获取基准收益率：如果未指定则使用实例设置的基准
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_rets
        
        # 获取窗口大小和最小观测值数量
        if window is None:
            window = self.defaults['window']
        if minp is None:
            minp = self.defaults['minp']
        
        # 广播基准收益率并计算滚动上行捕获比率
        benchmark_rets = broadcast_to(to_2d_array(benchmark_rets), to_2d_array(self.obj))
        result = nb.rolling_up_capture_nb(self.to_2d_array(), window, minp, benchmark_rets, self.ann_factor)
        
        # 包装结果
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def down_capture(self,
                     benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                     wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算下行捕获比率（Down Capture Ratio）
        
        下行捕获比率衡量投资组合在市场下跌期间相对于基准的损失控制能力。
        该指标专门分析熊市或下跌行情中的表现，反映投资策略在不利市场
        环境下的风险控制效率，是评估防御性投资能力的重要指标。
        
        计算公式：
            下行捕获比率 = 基准下跌期间投资组合年化收益率 / 基准下跌期间基准年化收益率
            
        其中只考虑基准收益率为负的时间段。
        
        下行捕获比率解读：
            - **比率 < 1.0**: 熊市中损失小于基准，防御能力强
            - **比率 = 1.0**: 熊市中与基准表现一致
            - **比率 > 1.0**: 熊市中损失大于基准，防御能力弱
            - **比率越低**: 熊市风险控制能力越强
        
        应用场景：
            - **熊市策略评估**：评估策略在熊市中的风险控制能力
            - **防御性分析**：分析投资策略的防御性特征
            - **风险管理**：评估下行风险控制效果
            - **策略分类**：区分进攻型、平衡型和防御型策略
        
        See `vectorbt.returns.nb.down_capture_nb`.
        """
        # 获取基准收益率并广播到相同维度
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_rets
        benchmark_rets = broadcast_to(to_2d_array(benchmark_rets), to_2d_array(self.obj))
        
        # 计算下行捕获比率
        result = nb.down_capture_nb(self.to_2d_array(), benchmark_rets, self.ann_factor)
        wrap_kwargs = merge_dicts(dict(name_or_index='down_capture'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_down_capture(self,
                             benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                             window: tp.Optional[int] = None,
                             minp: tp.Optional[int] = None,
                             wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口下行捕获比率计算
        
        计算滚动窗口内的下行捕获比率，提供时变的熊市表现分析。
        通过观察下行捕获比率的时间变化，可以评估投资策略在不同
        时期熊市环境下的风险控制能力变化。
        
        应用场景：
            - **动态防御性分析**：监控策略防御能力的时间变化
            - **熊市适应性**：评估在不同熊市环境下的风险控制
            - **风险管理**：基于下行捕获变化调整风险管理策略
            - **市场择时**：识别最佳的防御时机
        
        Rolling version of `ReturnsAccessor.down_capture`.
        """
        # 获取基准收益率和参数设置
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_rets
        if window is None:
            window = self.defaults['window']
        if minp is None:
            minp = self.defaults['minp']
        
        # 广播基准收益率并计算滚动下行捕获比率
        benchmark_rets = broadcast_to(to_2d_array(benchmark_rets), to_2d_array(self.obj))
        result = nb.rolling_down_capture_nb(self.to_2d_array(), window, minp, benchmark_rets, self.ann_factor)
        
        # 包装结果
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def drawdown(self, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        计算回撤序列（Drawdown Series）
        
        回撤是衡量投资组合从历史峰值下跌程度的重要风险指标。回撤序列显示
        了投资组合在每个时间点相对于历史最高点的下跌幅度，是评估投资
        风险和制定风险管理策略的基础数据。
        
        计算公式：
            回撤[t] = (累计收益率[t] - 历史最大累计收益率[0:t]) / (1 + 历史最大累计收益率[0:t])
            
        其中历史最大累计收益率是从起始点到当前时点的最高累计收益率。
        
        参数说明：
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 控制返回结果的包装选项
        
        返回值：
            tp.SeriesFrame: 回撤序列
                - 与输入数据相同的形状和索引
                - 负值表示相对峰值的下跌幅度
                - 零值表示创出新高
        
        使用示例：
            >>> # 基础回撤序列计算
            >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
            >>> drawdowns = returns.vbt.returns.drawdown()
            >>> print(drawdowns)
            >>> # 显示每个时点的回撤情况
            
            >>> # 回撤可视化分析
            >>> import matplotlib.pyplot as plt
            >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            >>> 
            >>> # 上图：累计收益率
            >>> cum_returns = returns.vbt.returns.cumulative()
            >>> cum_returns.plot(ax=ax1, title='累计收益率')
            >>> ax1.fill_between(cum_returns.index, cum_returns, alpha=0.3)
            >>> 
            >>> # 下图：回撤序列
            >>> drawdowns.plot(ax=ax2, title='回撤序列', color='red')
            >>> ax2.fill_between(drawdowns.index, drawdowns, alpha=0.3, color='red')
            >>> ax2.set_ylabel('回撤幅度')
            >>> plt.tight_layout()
            >>> plt.show()
        
        应用场景：
            - **风险评估**：评估投资策略的回撤风险特征
            - **最大回撤计算**：作为计算最大回撤的基础数据
            - **回撤分析**：分析回撤的频率、深度和持续时间
            - **风险管理**：设置基于回撤的止损和风控规则
            - **策略优化**：优化策略参数以控制回撤风险
        
        回撤特征分析：
            - **回撤深度**：回撤的最大幅度
            - **回撤持续时间**：从开始回撤到恢复的时间长度
            - **回撤频率**：回撤发生的频繁程度
            - **恢复时间**：从回撤最低点恢复到峰值的时间
        
        技术实现：
            - 使用nb.drawdown_nb进行高效计算
            - 基于累计收益率和滚动最大值计算
            - 自动处理时间序列的峰值追踪
        
        与其他指标的关系：
            - 最大回撤 = min(回撤序列)
            - 卡尔玛比率 = 年化收益率 / |最大回撤|
            - 回撤时间 = 基于回撤序列的持续时间统计
        
        Relative decline from a peak.
        """
        # 调用Numba编译函数计算回撤序列
        result = nb.drawdown_nb(self.to_2d_array())
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    def max_drawdown(self, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算最大回撤（Maximum Drawdown, MDD）
        
        最大回撤是投资风险评估的核心指标之一，表示投资组合从历史最高点
        到随后最低点的最大下跌幅度。MDD是衡量投资策略极端风险的重要指标，
        广泛用于风险管理、绩效评估和投资决策。
        
        计算公式：
            最大回撤 = max(历史峰值 - 当前值) / 历史峰值
            
        等价于：
            最大回撤 = min(回撤序列)（绝对值）
        
        参数说明：
            wrap_kwargs (tp.KwargsLike): 包装参数
                - 默认设置name_or_index为'max_drawdown'
        
        返回值：
            tp.MaybeSeries: 最大回撤值
                - Series: 单一资产返回标量值
                - DataFrame: 多资产返回Series，每列一个最大回撤
                - 正数表示最大下跌幅度
        
        使用示例：
            >>> # 基础最大回撤计算
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=252))
            >>> max_dd = returns.vbt.returns.max_drawdown()
            >>> print(f"最大回撤: {max_dd:.2%}")
            >>> # 输出: 最大回撤: 15.67%
            
            >>> # 多策略最大回撤比较
            >>> strategies_returns = pd.DataFrame({
            ...     'Conservative': [...],    # 保守策略
            ...     'Balanced': [...],        # 平衡策略  
            ...     'Aggressive': [...],      # 激进策略
            ...     'Market_Index': [...]     # 市场指数
            ... })
            >>> max_drawdowns = strategies_returns.vbt.returns.max_drawdown()
            >>> print(max_drawdowns.sort_values())
            >>> # Conservative    0.0823  (8.23% 最大回撤)
            >>> # Market_Index    0.1245  (12.45% 最大回撤)
            >>> # Balanced        0.1456  (14.56% 最大回撤)
            >>> # Aggressive      0.2134  (21.34% 最大回撤)
            
            >>> # 风险调整收益分析
            >>> annual_returns = strategies_returns.vbt.returns.annualized()
            >>> risk_adjusted = pd.DataFrame({
            ...     'Annual_Return': annual_returns,
            ...     'Max_Drawdown': max_drawdowns,
            ...     'Calmar_Ratio': annual_returns / max_drawdowns,
            ...     'Risk_Category': pd.cut(max_drawdowns, 
            ...                           bins=[0, 0.1, 0.2, 0.3, 1.0],
            ...                           labels=['低风险', '中风险', '高风险', '极高风险'])
            ... })
            >>> print(risk_adjusted.sort_values('Calmar_Ratio', ascending=False))
        
        应用场景：
            - **风险评估**：评估投资策略的极端风险水平
            - **卡尔玛比率**：计算风险调整收益指标的分母
            - **风险限制**：设置投资组合的最大回撤限制
            - **策略比较**：比较不同投资策略的风险特征
            - **投资决策**：选择最大回撤可接受的投资方案
        
        最大回撤解读：
            - **优秀控制** (<5%)：出色的风险控制能力
            - **良好控制** (5%-10%)：较好的风险管理水平
            - **可接受** (10%-20%)：一般的风险控制水平
            - **需要关注** (20%-30%)：风险控制能力有待提高
            - **高风险** (>30%)：存在较大的极端风险
        
        投资策略分类：
            - **保守型策略** (MDD < 10%)：适合风险厌恶投资者
            - **平衡型策略** (MDD 10%-20%)：适合中等风险承受能力投资者
            - **成长型策略** (MDD 20%-35%)：适合风险偏好投资者
            - **激进型策略** (MDD > 35%)：适合高风险承受能力投资者
        
        与其他指标的关系：
            - 卡尔玛比率 = 年化收益率 / 最大回撤
            - 回撤恢复时间 = 从最大回撤点恢复到峰值的时间
            - 下行波动率 vs 最大回撤：不同的风险度量视角
        
        技术实现：
            - 使用nb.max_drawdown_nb进行高效计算
            - 基于累计收益率序列的滚动最大值计算
            - 自动处理时间序列的峰谷识别
        
        风险管理应用：
            - **止损设置**：基于最大回撤设置止损点
            - **仓位控制**：根据最大回撤调整仓位大小
            - **策略筛选**：筛选最大回撤在可接受范围内的策略
            - **风险预算**：基于最大回撤进行风险资本分配
        
        注意事项：
            - 最大回撤是基于历史数据的后视指标
            - 未来的最大回撤可能超过历史最大回撤
            - 需要结合回撤持续时间进行综合评估
            - 不同市场环境下最大回撤的参考价值不同
        
        See `vectorbt.returns.nb.max_drawdown_nb`.

        Yields the same result as `max_drawdown` of `ReturnsAccessor.drawdowns`.
        """
        # 调用Numba编译函数计算最大回撤
        result = nb.max_drawdown_nb(self.to_2d_array())
        
        # 设置包装参数，默认名称为'max_drawdown'
        wrap_kwargs = merge_dicts(dict(name_or_index='max_drawdown'), wrap_kwargs)
        
        # 包装结果为降维输出
        return self.wrapper.wrap_reduced(result, group_by=False, **wrap_kwargs)

    def rolling_max_drawdown(self,
                             window: tp.Optional[int] = None,
                             minp: tp.Optional[int] = None,
                             wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口最大回撤计算
        
        计算滚动窗口内的最大回撤，提供时变的极端风险分析。通过观察最大回撤
        在时间序列上的变化，可以动态监控投资组合的极端风险暴露，为风险管理
        和投资决策提供重要的时间维度信息。
        
        计算原理：
            对于每个时间点t，计算窗口[t-window+1, t]内的最大回撤：
            滚动最大回撤[t] = max(窗口内历史峰值 - 当前值) / 窗口内历史峰值
        
        参数说明：
            window (tp.Optional[int]): 滚动窗口大小
                - None: 使用默认配置中的window值
                - 整数: 指定窗口长度（时间点数）
                - 建议至少60个观测值以获得稳定的最大回撤
            minp (tp.Optional[int]): 最小观测值数量
                - None: 使用默认配置中的minp值
                - 整数: 窗口内最少需要的有效观测值
            wrap_kwargs (tp.KwargsLike): 包装参数
        
        返回值：
            tp.SeriesFrame: 滚动最大回撤序列
                - 前window-1个值为NaN（数据不足）
                - 后续每个值为对应窗口的最大回撤
        
        使用示例：
            >>> # 计算252天滚动最大回撤
            >>> returns = pd.Series([...], index=pd.date_range('2023-01-01', periods=1000))
            >>> rolling_mdd = returns.vbt.returns.rolling_max_drawdown(window=252)
            >>> 
            >>> # 可视化最大回撤时间变化
            >>> import matplotlib.pyplot as plt
            >>> fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            >>> 
            >>> # 上图：累计收益率和回撤
            >>> cum_returns = returns.vbt.returns.cumulative()
            >>> drawdowns = returns.vbt.returns.drawdown()
            >>> cum_returns.plot(ax=ax1, label='累计收益率', linewidth=2)
            >>> ax1_twin = ax1.twinx()
            >>> drawdowns.plot(ax=ax1_twin, color='red', alpha=0.7, label='回撤')
            >>> ax1.legend(loc='upper left')
            >>> ax1_twin.legend(loc='upper right')
            >>> ax1.set_title('累计收益率与回撤')
            >>> 
            >>> # 下图：滚动最大回撤
            >>> rolling_mdd.plot(ax=ax2, color='darkred', linewidth=2)
            >>> ax2.axhline(y=0.1, color='orange', linestyle='--', label='10%风险线')
            >>> ax2.axhline(y=0.2, color='red', linestyle='--', label='20%风险线')
            >>> ax2.legend()
            >>> ax2.set_title('252天滚动最大回撤')
            >>> ax2.set_ylabel('最大回撤')
            >>> plt.tight_layout()
            >>> plt.show()
        
        应用场景：
            - **动态风险监控**：实时监控投资组合的极端风险变化
            - **风险预警**：当滚动最大回撤超过阈值时发出预警
            - **策略调整**：基于最大回撤趋势调整投资策略
            - **风险归因**：分析不同时期最大回撤的驱动因素
            - **压力测试**：评估策略在不同市场环境下的表现
        
        风险管理信号：
            - **回撤扩大**：滚动最大回撤持续增加，风险上升
            - **回撤稳定**：滚动最大回撤保持稳定，风险可控
            - **回撤收窄**：滚动最大回撤下降，风险改善
            - **突破阈值**：超过预设风险限制，需要采取行动
        
        投资决策应用：
            - **仓位管理**：根据滚动最大回撤调整仓位大小
            - **止损策略**：设置基于滚动最大回撤的动态止损
            - **策略切换**：当最大回撤恶化时切换到防御性策略
            - **风险预算**：基于滚动最大回撤分配风险预算
        
        技术实现：
            - 使用nb.rolling_max_drawdown_nb进行高效计算
            - 基于滚动窗口内的累计收益率峰谷分析
            - 支持最小观测值要求，提高结果可靠性
        
        Rolling version of `ReturnsAccessor.max_drawdown`.
        """
        # 获取窗口大小：如果未指定则使用默认配置
        if window is None:
            window = self.defaults['window']
        
        # 获取最小观测值数量：如果未指定则使用默认配置
        if minp is None:
            minp = self.defaults['minp']
        
        # 调用Numba编译函数计算滚动最大回撤
        result = nb.rolling_max_drawdown_nb(self.to_2d_array(), window, minp)
        
        # 设置包装参数的默认值
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        
        # 包装结果，保持原始形状和索引
        return self.wrapper.wrap(result, group_by=False, **wrap_kwargs)

    @property
    def drawdowns(self) -> Drawdowns:
        """`ReturnsAccessor.get_drawdowns` with default arguments."""
        return self.get_drawdowns()

    def get_drawdowns(self, wrapper_kwargs: tp.KwargsLike = None, **kwargs) -> Drawdowns:
        """Generate drawdown records of cumulative returns.

        See `vectorbt.generic.drawdowns.Drawdowns`."""
        wrapper_kwargs = merge_dicts(self.wrapper.config, wrapper_kwargs)
        return Drawdowns.from_ts(self.cumulative(start_value=1.), wrapper_kwargs=wrapper_kwargs, **kwargs)

    @property
    def qs(self):
        """Quantstats adapter."""
        from vectorbt.returns.qs_adapter import QSAdapter

        return QSAdapter(self)

    # ############# Resolution ############# #

    def resolve_self(self: ReturnsAccessorT,
                     cond_kwargs: tp.KwargsLike = None,
                     custom_arg_names: tp.Optional[tp.Set[str]] = None,
                     impacts_caching: bool = True,
                     silence_warnings: bool = False) -> ReturnsAccessorT:
        """Resolve self.

        See `vectorbt.base.array_wrapper.Wrapping.resolve_self`.

        Creates a copy of this instance `year_freq` is different in `cond_kwargs`."""
        if cond_kwargs is None:
            cond_kwargs = {}
        if custom_arg_names is None:
            custom_arg_names = set()

        reself = Wrapping.resolve_self(
            self,
            cond_kwargs=cond_kwargs,
            custom_arg_names=custom_arg_names,
            impacts_caching=impacts_caching,
            silence_warnings=silence_warnings
        )
        if 'year_freq' in cond_kwargs:
            self_copy = reself.replace(year_freq=cond_kwargs['year_freq'])

            if self_copy.year_freq != reself.year_freq:
                if not silence_warnings:
                    warnings.warn(f"Changing the year frequency will create a copy of this object. "
                                  f"Consider setting it upon object creation to re-use existing cache.", stacklevel=2)
                for alias in reself.self_aliases:
                    if alias not in custom_arg_names:
                        cond_kwargs[alias] = self_copy
                cond_kwargs['year_freq'] = self_copy.year_freq
                if impacts_caching:
                    cond_kwargs['use_caching'] = False
                return self_copy
        return reself

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `ReturnsAccessor.stats`.

        Merges `vectorbt.generic.accessors.GenericAccessor.stats_defaults`,
        defaults from `ReturnsAccessor.defaults` (acting as `settings`), and
        `returns.stats` from `vectorbt._settings.settings`"""
        from vectorbt._settings import settings
        returns_stats_cfg = settings['returns']['stats']

        return merge_dicts(
            GenericAccessor.stats_defaults.__get__(self),
            dict(settings=self.defaults),
            dict(settings=dict(year_freq=self.year_freq)),
            returns_stats_cfg
        )

    _metrics: tp.ClassVar[Config] = Config(
        dict(
            start=dict(
                title='Start',
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                check_is_not_grouped=False,
                tags='wrapper'
            ),
            end=dict(
                title='End',
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                check_is_not_grouped=False,
                tags='wrapper'
            ),
            period=dict(
                title='Period',
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                check_is_not_grouped=False,
                tags='wrapper'
            ),
            total_return=dict(
                title='Total Return [%]',
                calc_func='total',
                post_calc_func=lambda self, out, settings: out * 100,
                tags='returns'
            ),
            benchmark_return=dict(
                title='Benchmark Return [%]',
                calc_func='benchmark_rets.vbt.returns.total',
                post_calc_func=lambda self, out, settings: out * 100,
                check_has_benchmark_rets=True,
                tags='returns'
            ),
            ann_return=dict(
                title='Annualized Return [%]',
                calc_func='annualized',
                post_calc_func=lambda self, out, settings: out * 100,
                check_has_freq=True,
                check_has_year_freq=True,
                tags='returns'
            ),
            ann_volatility=dict(
                title='Annualized Volatility [%]',
                calc_func='annualized_volatility',
                post_calc_func=lambda self, out, settings: out * 100,
                check_has_freq=True,
                check_has_year_freq=True,
                tags='returns'
            ),
            max_dd=dict(
                title='Max Drawdown [%]',
                calc_func='drawdowns.max_drawdown',
                post_calc_func=lambda self, out, settings: -out * 100,
                tags=['returns', 'drawdowns']
            ),
            max_dd_duration=dict(
                title='Max Drawdown Duration',
                calc_func='drawdowns.max_duration',
                fill_wrap_kwargs=True,
                tags=['returns', 'drawdowns', 'duration']
            ),
            sharpe_ratio=dict(
                title='Sharpe Ratio',
                calc_func='sharpe_ratio',
                check_has_freq=True,
                check_has_year_freq=True,
                tags='returns'
            ),
            calmar_ratio=dict(
                title='Calmar Ratio',
                calc_func='calmar_ratio',
                check_has_freq=True,
                check_has_year_freq=True,
                tags='returns'
            ),
            omega_ratio=dict(
                title='Omega Ratio',
                calc_func='omega_ratio',
                check_has_freq=True,
                check_has_year_freq=True,
                tags='returns'
            ),
            sortino_ratio=dict(
                title='Sortino Ratio',
                calc_func='sortino_ratio',
                check_has_freq=True,
                check_has_year_freq=True,
                tags='returns'
            ),
            skew=dict(
                title='Skew',
                calc_func='obj.skew',
                tags='returns'
            ),
            kurtosis=dict(
                title='Kurtosis',
                calc_func='obj.kurtosis',
                tags='returns'
            ),
            tail_ratio=dict(
                title='Tail Ratio',
                calc_func='tail_ratio',
                tags='returns'
            ),
            common_sense_ratio=dict(
                title='Common Sense Ratio',
                calc_func='common_sense_ratio',
                check_has_freq=True,
                check_has_year_freq=True,
                tags='returns'
            ),
            value_at_risk=dict(
                title='Value at Risk',
                calc_func='value_at_risk',
                tags='returns'
            ),
            alpha=dict(
                title='Alpha',
                calc_func='alpha',
                check_has_freq=True,
                check_has_year_freq=True,
                check_has_benchmark_rets=True,
                tags='returns'
            ),
            beta=dict(
                title='Beta',
                calc_func='beta',
                check_has_benchmark_rets=True,
                tags='returns'
            )
        ),
        copy_kwargs=dict(copy_mode='deep')
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `ReturnsAccessor.plots`.

        Merges `vectorbt.generic.accessors.GenericAccessor.plots_defaults`,
        defaults from `ReturnsAccessor.defaults` (acting as `settings`), and
        `returns.plots` from `vectorbt._settings.settings`"""
        from vectorbt._settings import settings
        returns_plots_cfg = settings['returns']['plots']

        return merge_dicts(
            GenericAccessor.plots_defaults.__get__(self),
            dict(settings=self.defaults),
            dict(settings=dict(year_freq=self.year_freq)),
            returns_plots_cfg
        )

    @property
    def subplots(self) -> Config:
        return self._subplots


ReturnsAccessor.override_metrics_doc(__pdoc__)
ReturnsAccessor.override_subplots_doc(__pdoc__)


@register_series_vbt_accessor('returns')
class ReturnsSRAccessor(ReturnsAccessor, GenericSRAccessor):
    """
    Series专用收益率访问器
    
    该类是专门为pandas Series设计的收益率分析访问器，继承了ReturnsAccessor
    的所有功能，并针对单一时间序列数据进行了优化。通过pandas的访问器机制，
    可以直接在Series对象上使用.vbt.returns访问各种收益率分析功能。
    
    设计特点：
        - **单序列优化**：专门针对单一收益率时间序列优化
        - **完整功能**：继承ReturnsAccessor的所有分析方法
        - **易于使用**：通过.vbt.returns直接访问
        - **高性能**：针对Series数据结构优化的计算性能
    
    主要功能：
        - 基础收益率指标：总收益率、年化收益率、累计收益率
        - 风险度量：波动率、VaR、CVaR、最大回撤
        - 风险调整收益：夏普比率、索提诺比率、卡尔玛比率
        - 相对绩效：Alpha、Beta、信息比率（需要基准）
        - 滚动分析：所有指标的时变版本
        - 可视化：专门的累计收益率绘图方法
    
    Accessor on top of return series. For Series only.

    Accessible through `pd.Series.vbt.returns`.
    """

    def __init__(self,
                 obj: tp.Series,
                 benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                 year_freq: tp.Optional[tp.FrequencyLike] = None,
                 defaults: tp.KwargsLike = None,
                 **kwargs) -> None:
        """
        初始化Series收益率访问器
        
        参数说明：
            obj (tp.Series): 收益率数据Series
            benchmark_rets (tp.Optional[tp.ArrayLike]): 基准收益率
            year_freq (tp.Optional[tp.FrequencyLike]): 年化频率
            defaults (tp.KwargsLike): 默认参数配置
            **kwargs: 传递给父类的其他参数
        """
        # 初始化GenericSRAccessor，提供基础Series访问器功能
        GenericSRAccessor.__init__(self, obj, **kwargs)
        
        # 初始化ReturnsAccessor，提供收益率分析功能
        ReturnsAccessor.__init__(
            self,
            obj,
            benchmark_rets=benchmark_rets,
            year_freq=year_freq,
            defaults=defaults,
            **kwargs
        )

    def plot_cumulative(self,
                        benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                        start_value: float = 1,
                        fill_to_benchmark: bool = False,
                        main_kwargs: tp.KwargsLike = None,
                        benchmark_kwargs: tp.KwargsLike = None,
                        hline_shape_kwargs: tp.KwargsLike = None,
                        add_trace_kwargs: tp.KwargsLike = None,
                        xref: str = 'x',
                        yref: str = 'y',
                        fig: tp.Optional[tp.BaseFigure] = None,
                        **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot cumulative returns.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against.
                Will broadcast per element.
            start_value (float): The starting returns.
            fill_to_benchmark (bool): Whether to fill between main and benchmark, or between main and `start_value`.
            main_kwargs (dict): Keyword arguments passed to `vectorbt.generic.accessors.GenericSRAccessor.plot` for main.
            benchmark_kwargs (dict): Keyword arguments passed to `vectorbt.generic.accessors.GenericSRAccessor.plot` for benchmark.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for `start_value` line.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import pandas as pd
            >>> import numpy as np

            >>> np.random.seed(0)
            >>> rets = pd.Series(np.random.uniform(-0.05, 0.05, size=100))
            >>> benchmark_rets = pd.Series(np.random.uniform(-0.05, 0.05, size=100))
            >>> rets.vbt.returns.plot_cumulative(benchmark_rets=benchmark_rets)
            ```

            ![](/assets/images/plot_cumulative.svg)
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(xref, fig)
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_rets
        fill_to_benchmark = fill_to_benchmark and benchmark_rets is not None

        if benchmark_rets is not None:
            # Plot benchmark
            benchmark_rets = broadcast_to(benchmark_rets, self.obj)
            if benchmark_kwargs is None:
                benchmark_kwargs = {}
            benchmark_kwargs = merge_dicts(dict(
                trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg['color_schema']['gray']
                    ),
                    name='Benchmark'
                )
            ), benchmark_kwargs)
            benchmark_cumrets = benchmark_rets.vbt.returns.cumulative(start_value=start_value)
            benchmark_cumrets.vbt.plot(**benchmark_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)
        else:
            benchmark_cumrets = None

        # Plot main
        if main_kwargs is None:
            main_kwargs = {}
        main_kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['purple']
                )
            ),
            other_trace_kwargs='hidden'
        ), main_kwargs)
        cumrets = self.cumulative(start_value=start_value)
        if fill_to_benchmark:
            cumrets.vbt.plot_against(benchmark_cumrets, **main_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)
        else:
            cumrets.vbt.plot_against(start_value, **main_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)

        # Plot hline
        if hline_shape_kwargs is None:
            hline_shape_kwargs = {}
        fig.add_shape(**merge_dicts(dict(
            type='line',
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=start_value,
            x1=x_domain[1],
            y1=start_value,
            line=dict(
                color="gray",
                dash="dash",
            )
        ), hline_shape_kwargs))

        return fig


@register_dataframe_vbt_accessor('returns')
class ReturnsDFAccessor(ReturnsAccessor, GenericDFAccessor):
    """
    DataFrame专用收益率访问器
    
    该类是专门为pandas DataFrame设计的收益率分析访问器，继承了ReturnsAccessor
    的所有功能，并针对多列时间序列数据进行了优化。支持对多个资产或策略
    进行批量收益率分析，是构建投资组合分析和多策略比较的核心工具。
    
    设计特点：
        - **多序列处理**：同时处理多个收益率时间序列
        - **批量计算**：高效的向量化批量计算
        - **列间分析**：支持不同列（资产/策略）间的比较分析
        - **统一接口**：与Series访问器保持一致的API接口
    
    主要功能：
        - **批量指标计算**：同时计算多个资产的所有收益率指标
        - **策略比较**：比较不同投资策略的风险收益特征
        - **投资组合分析**：分析投资组合中各成分的表现
        - **基准对比**：与单一或多个基准进行批量比较
        - **风险归因**：分析不同资产对组合风险的贡献
    
    使用场景：
        - **多资产分析**：同时分析股票、债券、商品等多类资产
        - **策略筛选**：从多个候选策略中筛选最优策略
        - **组合构建**：基于各资产的风险收益特征构建投资组合
        - **绩效监控**：监控投资组合中各成分的表现变化
        - **风险管理**：评估和控制投资组合的整体风险
    
    返回值特征：
        - 大多数方法返回Series，每列对应一个指标值
        - 滚动方法返回DataFrame，保持时间序列结构
        - 支持按列进行分组和聚合操作
    
    Accessor on top of return series. For DataFrames only.

    Accessible through `pd.DataFrame.vbt.returns`.
    """

    def __init__(self,
                 obj: tp.Frame,
                 benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                 year_freq: tp.Optional[tp.FrequencyLike] = None,
                 defaults: tp.KwargsLike = None,
                 **kwargs) -> None:
        """
        初始化DataFrame收益率访问器
        
        参数说明：
            obj (tp.Frame): 收益率数据DataFrame
                - 每列代表一个资产或策略的收益率序列
                - 行索引通常为时间索引
            benchmark_rets (tp.Optional[tp.ArrayLike]): 基准收益率
                - 可以是Series（单一基准）或DataFrame（多基准）
                - 会自动广播到与obj相同的维度
            year_freq (tp.Optional[tp.FrequencyLike]): 年化频率
            defaults (tp.KwargsLike): 默认参数配置
            **kwargs: 传递给父类的其他参数
        """
        # 初始化GenericDFAccessor，提供基础DataFrame访问器功能
        GenericDFAccessor.__init__(self, obj, **kwargs)
        
        # 初始化ReturnsAccessor，提供收益率分析功能
        ReturnsAccessor.__init__(
            self,
            obj,
            benchmark_rets=benchmark_rets,
            year_freq=year_freq,
            defaults=defaults,
            **kwargs
        )
