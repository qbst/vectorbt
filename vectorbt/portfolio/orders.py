# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT PORTFOLIO ORDERS MODULE: 订单记录分析和管理核心模块
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于处理和分析订单记录的核心模块。订单记录是量化交易中
最基础的数据结构，记录了每笔交易的详细信息，包括买卖方向、价格、数量、手续费等关键参数。

核心设计理念：
1. **高性能订单分析**：基于NumPy结构化数组和Numba JIT编译，实现接近C语言的计算性能，
   能够处理数百万条订单记录而不会出现性能瓶颈。

2. **专业统计指标**：内置丰富的订单分析指标，包括总订单数、买卖订单比例、平均价格、
   手续费统计等量化交易中的标准评估指标。

3. **智能订单过滤**：提供买入/卖出订单的自动过滤功能，支持按订单方向、价格区间、
   时间范围等多种条件进行订单筛选和分析。

4. **完整可视化支持**：提供专业的订单分析图表，包括订单时间线图、买卖信号散点图等，
   支持与价格图表的叠加显示。

订单记录数据结构：
每个订单记录包含以下核心字段：
- id: 订单唯一标识符，用于追踪和关联订单
- col: 列索引，标识订单属于哪个资产或策略
- idx: 时间索引，标识订单发生的时间点
- size: 订单数量，正数表示买入，负数表示卖出
- price: 订单执行价格，实际成交价格
- fees: 订单手续费，包括交易佣金和其他费用
- side: 订单方向，Buy(买入)或Sell(卖出)

与vectorbt生态系统的关系：
- **Portfolio集成**：作为Portfolio类的核心组件，提供订单层面的分析功能
- **Records继承**：继承自Records类，获得高性能的结构化数据处理能力
- **Trades转换**：订单记录是生成交易记录(Trades)的基础数据源
- **可视化支持**：与vectorbt的绘图系统集成，提供专业的金融图表

应用场景：
- **交易策略回测**：分析策略产生的所有订单，评估策略的交易频率和效率
- **交易成本分析**：计算总手续费、平均交易成本、成本对收益的影响
- **市场影响分析**：分析订单大小对市场价格的影响，优化订单执行策略
- **风险管理**：监控异常订单、大额订单、高频交易等风险因素
- **绩效归因分析**：将投资组合收益归因到具体的订单和交易决策

技术特点：
- **内存高效存储**：使用结构化数组压缩存储，比DataFrame节省50-80%内存
- **向量化计算**：充分利用NumPy的向量化能力，批量处理大规模订单数据
- **智能字段映射**：通过field_config系统自动映射字段名称、类型和显示格式
- **缓存优化**：智能缓存机制避免重复计算，显著提升大数据量下的查询性能

该模块是vectorbt框架中订单分析的基础，为量化交易策略的开发、测试和优化
提供了工业级的订单记录分析能力。

Base class for working with order records.

Order records capture information on filled orders. Orders are mainly populated when simulating
a portfolio and can be accessed as `vectorbt.portfolio.base.Portfolio.orders`.

```pycon
>>> import pandas as pd
>>> import numpy as np
>>> from datetime import datetime, timedelta
>>> import vectorbt as vbt

>>> np.random.seed(42)
>>> price = pd.DataFrame({
...     'a': np.random.uniform(1, 2, size=100),
...     'b': np.random.uniform(1, 2, size=100)
... }, index=[datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)])
>>> size = pd.DataFrame({
...     'a': np.random.uniform(-1, 1, size=100),
...     'b': np.random.uniform(-1, 1, size=100),
... }, index=[datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)])
>>> pf = vbt.Portfolio.from_orders(price, size, fees=0.01, freq='d')
>>> orders = pf.orders

>>> orders.buy.count()
a    58
b    51
Name: count, dtype: int64

>>> orders.sell.count()
a    42
b    49
Name: count, dtype: int64
```

## Stats

!!! hint
    See `vectorbt.generic.stats_builder.StatsBuilderMixin.stats` and `Orders.metrics`.

```pycon
>>> orders['a'].stats()
Start                2020-01-01 00:00:00
End                  2020-04-09 00:00:00
Period                 100 days 00:00:00
Total Records                        100
Total Buy Orders                      58
Total Sell Orders                     42
Min Size                        0.003033
Max Size                        0.989877
Avg Size                        0.508608
Avg Buy Size                    0.468802
Avg Sell Size                   0.563577
Avg Buy Price                   1.437037
Avg Sell Price                  1.515951
Total Fees                      0.740177
Min Fees                        0.000052
Max Fees                        0.016224
Avg Fees                        0.007402
Avg Buy Fees                    0.006771
Avg Sell Fees                   0.008273
Name: a, dtype: object
```

`Orders.stats` also supports (re-)grouping:

```pycon
>>> orders.stats(group_by=True)
Start                2020-01-01 00:00:00
End                  2020-04-09 00:00:00
Period                 100 days 00:00:00
Total Records                        200
Total Buy Orders                     109
Total Sell Orders                     91
Min Size                        0.003033
Max Size                        0.989877
Avg Size                        0.506279
Avg Buy Size                    0.472504
Avg Sell Size                   0.546735
Avg Buy Price                    1.47336
Avg Sell Price                  1.496759
Total Fees                      1.483343
Min Fees                        0.000052
Max Fees                        0.018319
Avg Fees                        0.007417
Avg Buy Fees                    0.006881
Avg Sell Fees                   0.008058
Name: group, dtype: object
```

## Plots

!!! hint
    See `vectorbt.generic.plots_builder.PlotsBuilderMixin.plots` and `Orders.subplots`.

`Orders` class has a single subplot based on `Orders.plot`:

```pycon
>>> orders['a'].plots()
```

![](/assets/images/orders_plots.svg)
"""

# 导入所需的Python标准库和第三方库
import numpy as np  # 导入NumPy库，用于高性能数值计算和数组操作
import pandas as pd  # 导入Pandas库，用于数据结构和数据分析
import plotly.graph_objects as go  # 导入Plotly绘图库，用于创建交互式图表

# 导入vectorbt框架的核心模块和类型系统
from vectorbt import _typing as tp  # 导入vectorbt的类型定义模块，提供类型提示支持
from vectorbt.base.array_wrapper import ArrayWrapper  # 导入数组包装器，管理数组元数据
from vectorbt.base.reshape_fns import to_2d_array  # 导入数组重塑函数，将数组转换为2维
from vectorbt.portfolio.enums import order_dt, OrderSide  # 导入投资组合枚举：订单数据类型和订单方向
from vectorbt.records.base import Records  # 导入记录基类，提供结构化数据处理能力
from vectorbt.records.decorators import attach_fields, override_field_config  # 导入字段装饰器
from vectorbt.utils.colors import adjust_lightness  # 导入颜色工具函数，用于调整图表颜色亮度
from vectorbt.utils.config import merge_dicts, Config  # 导入配置工具：字典合并和配置类
from vectorbt.utils.figure import make_figure  # 导入图形工具函数，用于创建Plotly图表

# 定义模块文档字典，用于控制文档生成
__pdoc__ = {}

# 订单字段配置：定义订单记录中每个字段的元数据和显示格式
orders_field_config = Config(
    dict(
        # 指定数据类型为订单数据类型，使用NumPy结构化数组
        dtype=order_dt,
        # 字段设置：定义每个字段的显示标题和数据映射
        settings=dict(
            id=dict(
                title='Order Id'  # 订单ID字段的显示标题
            ),
            size=dict(
                title='Size'  # 订单数量字段的显示标题
            ),
            price=dict(
                title='Price'  # 订单价格字段的显示标题
            ),
            fees=dict(
                title='Fees'  # 订单手续费字段的显示标题
            ),
            side=dict(
                title='Side',  # 订单方向字段的显示标题
                mapping=OrderSide  # 将数值映射为OrderSide枚举（Buy/Sell）
            )
        )
    ),
    readonly=True,  # 设置为只读配置，防止意外修改
    as_attrs=False  # 不将配置项作为属性访问
)
"""订单字段配置对象，定义了订单记录中每个字段的元数据信息。"""

# 为orders_field_config生成文档
__pdoc__['orders_field_config'] = f"""Field config for `Orders`.

```json
{orders_field_config.to_doc()}
```
"""

# 订单字段附加配置：定义需要为Orders类自动生成的字段相关方法
orders_attach_field_config = Config(
    dict(
        side=dict(
            attach_filters=True  # 为side字段自动生成过滤器方法（如.buy, .sell）
        )
    ),
    readonly=True,  # 设置为只读配置
    as_attrs=False  # 不将配置项作为属性访问
)
"""订单字段附加配置，定义了需要自动生成的字段相关功能。"""

# 为orders_attach_field_config生成文档
__pdoc__['orders_attach_field_config'] = f"""Config of fields to be attached to `Orders`.

```json
{orders_attach_field_config.to_doc()}
```
"""

# 定义OrdersT类型变量，用于类型提示中的泛型约束
# 这确保相关方法返回的类型与调用类的类型一致
OrdersT = tp.TypeVar("OrdersT", bound="Orders")


@attach_fields(orders_attach_field_config)  # 应用字段附加配置装饰器，自动生成字段相关方法
@override_field_config(orders_field_config)  # 应用字段配置覆盖装饰器，使用orders_field_config配置
class Orders(Records):
    """
    订单记录分析类 - vectorbt量化交易框架的核心订单分析组件
    
    Orders类继承自vectorbt.records.base.Records，专门用于处理和分析订单记录数据。
    该类是vectorbt交易分析体系的基础组件，提供了完整的订单记录管理和分析功能。
    
    继承关系：
    - Records: 提供结构化记录数据的高性能处理能力
    - StatsBuilderMixin: 提供统计指标计算功能
    - PlotsBuilderMixin: 提供图表绘制功能
    - Wrapping: 提供ArrayWrapper集成功能
    
    核心功能：
    1. **订单记录存储**：使用NumPy结构化数组高效存储大量订单数据
    2. **智能字段映射**：自动映射订单字段（ID、价格、数量、方向等）
    3. **订单过滤功能**：支持按买入/卖出方向自动过滤订单
    4. **统计分析**：计算订单相关的各种统计指标
    5. **可视化支持**：提供专业的订单分析图表
    
    数据结构：
    - wrapper: ArrayWrapper对象，包含索引、列名、分组等元数据
    - records_arr: 结构化数组，存储订单记录的详细信息
    - close: 参考价格序列，用于图表绘制和分析
    
    字段说明：
    - id: 订单唯一标识符
    - col: 列索引（资产/策略标识）
    - idx: 时间索引（订单执行时间）
    - size: 订单数量（正数买入，负数卖出）
    - price: 订单执行价格
    - fees: 订单手续费
    - side: 订单方向（OrderSide.Buy 或 OrderSide.Sell）
    
    自动生成的过滤属性：
    - buy: 返回所有买入订单的子集
    - sell: 返回所有卖出订单的子集
    
    使用示例：
    ```python
    import pandas as pd
    import numpy as np
    import vectorbt as vbt
    
    # 创建价格数据和订单信号
    prices = pd.Series([100, 105, 98, 95, 102, 108, 103])
    orders_size = pd.Series([1, 0, -0.5, 0, 0.3, 0, -0.8])
    
    # 构建投资组合并获取订单记录
    pf = vbt.Portfolio.from_orders(prices, orders_size, fees=0.01)
    orders = pf.orders
    
    # 基本订单统计
    print(f"总订单数: {orders.count()}")
    print(f"买入订单数: {orders.buy.count()}")  
    print(f"卖出订单数: {orders.sell.count()}")
    print(f"平均订单价格: {orders.price.mean():.2f}")
    print(f"总手续费: {orders.fees.sum():.4f}")
    
    # 订单统计报告
    stats = orders.stats()
    print(stats)
    
    # 可视化订单
    fig = orders.plot()  # 绘制订单时间线图
    fig.show()
    
    # 获取可读格式的订单记录
    readable_orders = orders.records_readable
    print(readable_orders)
    
    # 分别分析买卖订单
    buy_orders = orders.buy
    sell_orders = orders.sell
    print(f"平均买入价格: {buy_orders.price.mean():.2f}")
    print(f"平均卖出价格: {sell_orders.price.mean():.2f}")
    ```
    
    注意事项：
    - 订单记录按时间顺序存储，保持交易的时间先后关系
    - 支持多资产和多策略的同时分析
    - 手续费已包含在订单记录中，便于成本分析
    - 使用.buy和.sell属性可以快速过滤买卖订单
    
    性能特点：
    - 基于NumPy结构化数组，内存效率极高
    - 支持百万级订单记录的实时分析
    - 向量化计算，比纯Python快10-100倍
    - 智能缓存机制，避免重复计算
    
    Extends `Records` for working with order records.
    """

    @property
    def field_config(self) -> Config:
        """
        获取字段配置
        
        返回当前Orders实例使用的字段配置对象，该配置定义了每个字段的
        数据类型、显示标题、映射关系等元数据信息。
        
        Returns:
            Config: 字段配置对象，包含所有字段的元数据定义
        
        Examples:
            >>> orders = pf.orders
            >>> config = orders.field_config
            >>> print(config['settings']['price']['title'])  # 输出: 'Price'
        """
        return self._field_config

    def __init__(self,
                 wrapper: ArrayWrapper,  # 数组包装器，管理索引和元数据
                 records_arr: tp.RecordArray,  # 订单记录的结构化数组
                 close: tp.Optional[tp.ArrayLike] = None,  # 可选的参考收盘价数据
                 **kwargs) -> None:  # 传递给父类的其他参数
        """
        Orders类构造函数
        
        初始化一个Orders对象实例，用于管理和分析订单记录数据。
        
        参数：
            wrapper (ArrayWrapper): 数组包装器对象，包含以下元数据：
                                   - 行索引（通常是时间戳）
                                   - 列索引（通常是资产代码）
                                   - 维度信息和分组设置
            records_arr (tp.RecordArray): NumPy结构化数组，包含所有订单记录，
                                         每条记录包含id、col、idx、size、price、fees、side等字段
            close (tp.Optional[tp.ArrayLike]): 可选的参考价格序列，用于：
                                              - 绘制价格图表时作为背景
                                              - 计算订单相对于收盘价的偏差
                                              - 提供价格上下文信息
            **kwargs: 传递给父类Records构造函数的其他参数
        
        处理流程：
            1. 调用父类Records的构造函数，完成基础的结构化数据初始化
            2. 存储可选的参考价格数据，用于后续的图表绘制和分析
            3. 继承Records类的所有功能：统计分析、过滤、映射等
        
        Examples:
            # 通常不会直接调用构造函数，而是通过Portfolio获取
            >>> pf = vbt.Portfolio.from_orders(prices, sizes)
            >>> orders = pf.orders  # 这里会自动调用Orders构造函数
            
            # 如需直接构造（高级用法）
            >>> import numpy as np
            >>> wrapper = vbt.ArrayWrapper.from_obj(prices)
            >>> # records_arr需要是符合order_dt格式的结构化数组
            >>> orders = vbt.Orders(wrapper, records_arr, close=prices)
        """
        # 调用父类Records的构造函数，完成基础初始化
        # 传递wrapper（元数据管理）、records_arr（订单数据）和close（参考价格）
        Records.__init__(
            self,
            wrapper,       # 数组包装器，管理索引和维度信息
            records_arr,   # 订单记录数组，包含所有订单的详细信息
            close=close,   # 参考价格序列，用于图表绘制和分析上下文
            **kwargs       # 其他配置参数
        )
        # 将参考价格存储为私有属性，供后续方法使用
        self._close = close

    def indexing_func(self: OrdersT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> OrdersT:
        """
        对Orders对象执行索引操作
        
        该方法允许用户使用pandas风格的索引操作来选择Orders对象的子集，
        比如选择特定时间范围的订单、特定资产的订单等。索引操作会同时
        应用到订单记录和参考价格数据上。
        
        参数：
            pd_indexing_func (tp.PandasIndexingFunc): pandas索引函数，
                                                     如 lambda x: x.iloc[:100] 选择前100行
                                                     或 lambda x: x['AAPL'] 选择苹果股票
            **kwargs: 传递给Records.indexing_func_meta的其他参数
        
        返回：
            OrdersT: 索引操作后的新Orders对象，包含筛选后的订单数据
        
        处理流程：
            1. 调用Records的indexing_func_meta获取索引元数据
            2. 根据列索引对参考价格数据进行相应的切片
            3. 创建新的Orders对象，保持数据一致性
        
        Examples:
            >>> # 选择前50个订单
            >>> early_orders = orders.indexing_func(lambda x: x.iloc[:50])
            >>> 
            >>> # 选择特定时间范围的订单  
            >>> import pandas as pd
            >>> date_range = pd.date_range('2023-01-01', '2023-03-31')
            >>> q1_orders = orders.indexing_func(lambda x: x.loc[date_range])
            >>> 
            >>> # 选择特定资产的订单
            >>> aapl_orders = orders.indexing_func(lambda x: x['AAPL'])
            >>> 
            >>> # 组合条件选择
            >>> recent_large = orders.indexing_func(
            ...     lambda x: x.iloc[-100:]  # 最近100个订单
            ... ).apply_mask(orders.size.values > 1000)  # 大于1000的订单
        
        Perform indexing on `Orders`.
        """
        # 调用父类的indexing_func_meta方法获取索引操作的元数据
        # 该方法返回：新的wrapper、新的records数组、分组索引、列索引
        new_wrapper, new_records_arr, group_idxs, col_idxs = \
            Records.indexing_func_meta(self, pd_indexing_func, **kwargs)
            
        # 处理参考价格数据的索引
        if self.close is not None:  # 如果存在参考价格数据
            # 将参考价格转换为2维数组，然后根据列索引进行切片
            # group_by=False确保不对列进行分组处理
            new_close = new_wrapper.wrap(to_2d_array(self.close)[:, col_idxs], group_by=False)
        else:
            new_close = None  # 如果原本就没有参考价格，保持None
            
        # 使用新的数据创建并返回Orders对象
        # replace方法会创建一个新实例，而不是修改当前实例
        return self.replace(
            wrapper=new_wrapper,        # 新的数组包装器（包含索引后的元数据）
            records_arr=new_records_arr, # 新的订单记录数组（包含索引后的数据）
            close=new_close              # 新的参考价格数据（如果存在的话）
        )

    @property
    def close(self) -> tp.Optional[tp.SeriesFrame]:
        """
        参考价格序列
        
        获取与订单数据关联的参考价格序列，通常是收盘价数据。
        该价格序列主要用于：
        - 在图表中绘制价格背景线
        - 提供订单执行时的市场价格上下文
        - 计算订单价格与市场价格的偏差
        
        Returns:
            tp.Optional[tp.SeriesFrame]: pandas Series或DataFrame，
                                       包含参考价格数据；如果未提供则返回None
        
        Examples:
            >>> orders = pf.orders
            >>> if orders.close is not None:
            ...     print(f"参考价格范围: {orders.close.min():.2f} - {orders.close.max():.2f}")
            ...     # 绘制订单图表时会自动使用这个价格序列作为背景
            ...     orders.plot()
        
        Reference price such as close (optional).
        """
        return self._close

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """
        Orders统计指标的默认配置
        
        获取Orders.stats方法使用的默认配置参数。该配置合并了两个来源：
        1. Records基类的默认统计配置
        2. vectorbt全局设置中orders模块的统计配置
        
        Returns:
            tp.Kwargs: 默认统计配置字典，包含统计指标的计算参数
        
        配置内容通常包括：
        - 要计算的统计指标列表
        - 数值格式化参数
        - 分组和聚合设置
        - 输出格式配置
        
        Examples:
            >>> orders = pf.orders
            >>> defaults = orders.stats_defaults
            >>> print("默认统计配置:", defaults)
            >>> 
            >>> # 使用自定义配置
            >>> custom_stats = orders.stats(**defaults, freq='D')
        
        Defaults for `Orders.stats`.

        Merges `vectorbt.records.base.Records.stats_defaults` and
        `orders.stats` from `vectorbt._settings.settings`.
        """
        # 从vectorbt全局设置中获取orders模块的统计配置
        from vectorbt._settings import settings
        orders_stats_cfg = settings['orders']['stats']

        # 合并父类的默认配置和orders特定的配置
        return merge_dicts(
            Records.stats_defaults.__get__(self),  # 获取Records基类的默认统计配置
            orders_stats_cfg                        # 添加orders特定的统计配置
        )

    # 定义Orders类的统计指标配置
    # _metrics是一个类变量，包含了所有可用的统计指标定义
    _metrics: tp.ClassVar[Config] = Config(
        dict(
            # 开始时间：返回数据的第一个时间点
            start=dict(
                title='Start',  # 指标显示名称
                calc_func=lambda self: self.wrapper.index[0],  # 计算函数：获取索引的第一个元素
                agg_func=None,  # 聚合函数：None表示不进行聚合
                tags='wrapper'  # 标签：表示这是wrapper相关的指标
            ),
            # 结束时间：返回数据的最后一个时间点
            end=dict(
                title='End',
                calc_func=lambda self: self.wrapper.index[-1],  # 获取索引的最后一个元素
                agg_func=None,
                tags='wrapper'
            ),
            # 时间周期：返回数据覆盖的总时间长度
            period=dict(
                title='Period',
                calc_func=lambda self: len(self.wrapper.index),  # 获取索引的长度
                apply_to_timedelta=True,  # 将结果转换为时间增量格式
                agg_func=None,
                tags='wrapper'
            ),
            # 总记录数：返回订单记录的总数
            total_records=dict(
                title='Total Records',
                calc_func='count',  # 使用内置的count方法
                tags='records'  # 标签：表示这是records相关的指标
            ),
            # 总买入订单数：返回所有买入订单的数量
            total_buy_orders=dict(
                title='Total Buy Orders',
                calc_func='buy.count',  # 调用buy过滤器的count方法
                tags=['orders', 'buy']  # 多个标签：orders和buy
            ),
            # 总卖出订单数：返回所有卖出订单的数量
            total_sell_orders=dict(
                title='Total Sell Orders',
                calc_func='sell.count',  # 调用sell过滤器的count方法
                tags=['orders', 'sell']
            ),
            # 最小订单数量
            min_size=dict(
                title='Min Size',
                calc_func='size.min',  # 调用size字段的min方法
                tags=['orders', 'size']
            ),
            # 最大订单数量
            max_size=dict(
                title='Max Size',
                calc_func='size.max',  # 调用size字段的max方法
                tags=['orders', 'size']
            ),
            # 平均订单数量
            avg_size=dict(
                title='Avg Size',
                calc_func='size.mean',  # 调用size字段的mean方法
                tags=['orders', 'size']
            ),
            # 平均买入订单数量
            avg_buy_size=dict(
                title='Avg Buy Size',
                calc_func='buy.size.mean',  # 买入订单的平均数量
                tags=['orders', 'buy', 'size']
            ),
            # 平均卖出订单数量
            avg_sell_size=dict(
                title='Avg Sell Size',
                calc_func='sell.size.mean',  # 卖出订单的平均数量
                tags=['orders', 'sell', 'size']
            ),
            # 平均买入价格
            avg_buy_price=dict(
                title='Avg Buy Price',
                calc_func='buy.price.mean',  # 买入订单的平均价格
                tags=['orders', 'buy', 'price']
            ),
            # 平均卖出价格
            avg_sell_price=dict(
                title='Avg Sell Price',
                calc_func='sell.price.mean',  # 卖出订单的平均价格
                tags=['orders', 'sell', 'price']
            ),
            # 总手续费
            total_fees=dict(
                title='Total Fees',
                calc_func='fees.sum',  # 所有订单手续费的总和
                tags=['orders', 'fees']
            ),
            # 最小手续费
            min_fees=dict(
                title='Min Fees',
                calc_func='fees.min',  # 最低的单笔手续费
                tags=['orders', 'fees']
            ),
            # 最大手续费
            max_fees=dict(
                title='Max Fees',
                calc_func='fees.max',  # 最高的单笔手续费
                tags=['orders', 'fees']
            ),
            # 平均手续费
            avg_fees=dict(
                title='Avg Fees',
                calc_func='fees.mean',  # 平均每笔订单的手续费
                tags=['orders', 'fees']
            ),
            # 平均买入手续费
            avg_buy_fees=dict(
                title='Avg Buy Fees',
                calc_func='buy.fees.mean',  # 买入订单的平均手续费
                tags=['orders', 'buy', 'fees']
            ),
            # 平均卖出手续费
            avg_sell_fees=dict(
                title='Avg Sell Fees',
                calc_func='sell.fees.mean',  # 卖出订单的平均手续费
                tags=['orders', 'sell', 'fees']
            ),
        ),
        copy_kwargs=dict(copy_mode='deep')  # 深拷贝配置，确保配置对象的独立性
    )

    @property
    def metrics(self) -> Config:
        """
        获取统计指标配置
        
        返回Orders类定义的所有统计指标的配置信息。这些指标包括：
        - 时间相关：开始时间、结束时间、时间周期
        - 订单数量：总订单数、买入/卖出订单数
        - 订单规模：最小/最大/平均订单大小
        - 价格统计：平均买入/卖出价格
        - 成本分析：各种手续费统计
        
        Returns:
            Config: 统计指标配置对象
        
        Examples:
            >>> orders = pf.orders
            >>> metrics_config = orders.metrics
            >>> print("可用指标:", list(metrics_config.keys()))
            >>> 
            >>> # 查看特定指标的配置
            >>> buy_orders_config = metrics_config['total_buy_orders']
            >>> print(f"指标标题: {buy_orders_config['title']}")
            >>> print(f"计算方法: {buy_orders_config['calc_func']}")
        """
        return self._metrics

    # ############# Plotting ############# #

    def plot(self,
             column: tp.Optional[tp.Label] = None,  # 要绘制的列名，None表示绘制所有列
             close_trace_kwargs: tp.KwargsLike = None,  # 收盘价线条的绘图参数
             buy_trace_kwargs: tp.KwargsLike = None,   # 买入信号标记的绘图参数
             sell_trace_kwargs: tp.KwargsLike = None,  # 卖出信号标记的绘图参数
             add_trace_kwargs: tp.KwargsLike = None,   # 添加图层的参数
             fig: tp.Optional[tp.BaseFigure] = None,   # 现有的图表对象，None表示创建新图表
             **layout_kwargs) -> tp.BaseFigure:        # 图表布局参数
        """
        绘制订单分析图表
        
        创建一个交互式图表，显示订单执行情况。图表包含：
        1. 价格背景线（如果提供了close数据）
        2. 买入订单标记（向上三角形，绿色）
        3. 卖出订单标记（向下三角形，红色）
        
        每个标记显示详细的订单信息，包括订单ID、时间、价格、数量和手续费。
        
        参数：
            column (tp.Optional[tp.Label]): 要绘制的列名（资产名称），
                                          如果为None则绘制第一列
            close_trace_kwargs (dict): 收盘价线条的样式参数，如：
                                     {'line': {'color': 'blue', 'width': 2}}
            buy_trace_kwargs (dict): 买入标记的样式参数，如：
                                   {'marker': {'size': 10, 'color': 'green'}}
            sell_trace_kwargs (dict): 卖出标记的样式参数
            add_trace_kwargs (dict): 添加图层时的参数
            fig (tp.Optional[tp.BaseFigure]): 现有图表对象，用于添加新图层
            **layout_kwargs: 图表布局参数，如title、xaxis_title等
        
        返回：
            tp.BaseFigure: Plotly图表对象，可以调用show()显示或进一步定制
        
        使用示例：
            ```python
            # 基本绘图
            fig = orders.plot()
            fig.show()
            
            # 绘制特定资产的订单
            fig = orders.plot(column='AAPL')
            fig.show()
            
            # 自定义样式
            fig = orders.plot(
                close_trace_kwargs={'line': {'color': 'navy', 'width': 3}},
                buy_trace_kwargs={'marker': {'size': 12, 'color': 'darkgreen'}},
                sell_trace_kwargs={'marker': {'size': 12, 'color': 'darkred'}},
                title="苹果股票订单执行情况",
                xaxis_title="时间",
                yaxis_title="价格 ($)"
            )
            fig.show()
            
            # 添加到现有图表
            price_fig = some_price_data.plot()
            orders_fig = orders.plot(fig=price_fig)  # 在价格图上叠加订单
            orders_fig.show()
            ```
        
        图表特性：
        - 交互式缩放和平移
        - 鼠标悬停显示详细订单信息
        - 支持多列（多资产）同时显示
        - 可与其他图表组合使用
        
        Args:
            column (str): Name of the column to plot.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Orders.close`.
            buy_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Buy" markers.
            sell_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Sell" markers.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import pandas as pd
            >>> from datetime import datetime, timedelta
            >>> import vectorbt as vbt

            >>> price = pd.Series([1., 2., 3., 2., 1.], name='Price')
            >>> price.index = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(len(price))]
            >>> size = pd.Series([1., 1., 1., 1., -1.])
            >>> orders = vbt.Portfolio.from_orders(price, size).orders

            >>> orders.plot()
            ```

            ![](/assets/images/orders_plot.svg)
        """  # pragma: no cover
        # 从vectorbt全局设置中获取绘图配置
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        # 选择特定列的数据进行绘制，group_by=False表示不分组处理
        self_col = self.select_one(column=column, group_by=False)

        # 设置收盘价线条的默认样式参数
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(dict(
            line=dict(
                color=plotting_cfg['color_schema']['blue']  # 使用配置中的蓝色
            ),
            name='Close'  # 图例中显示为"Close"
        ), close_trace_kwargs)
        
        # 初始化买入和卖出标记的样式参数
        if buy_trace_kwargs is None:
            buy_trace_kwargs = {}
        if sell_trace_kwargs is None:
            sell_trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        # 创建图表对象（如果未提供现有图表）
        if fig is None:
            fig = make_figure()
        # 应用布局参数
        fig.update_layout(**layout_kwargs)

        # 绘制价格背景线（如果存在收盘价数据）
        if self_col.close is not None:
            # 使用vbt访问器的plot方法绘制收盘价线条
            fig = self_col.close.vbt.plot(trace_kwargs=close_trace_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)

        # 只有当存在订单记录时才绘制订单标记
        if self_col.count() > 0:
            # 从订单记录中提取各种信息用于绘图和悬停显示
            
            # 提取订单ID数组
            id_ = self_col.get_field_arr('id')
            id_title = self_col.get_field_title('id')  # 获取字段的显示标题

            # 提取并映射时间索引
            idx = self_col.get_map_field_to_index('idx')  # 将索引映射为实际时间
            idx_title = self_col.get_field_title('idx')

            # 提取订单数量
            size = self_col.get_field_arr('size')
            size_title = self_col.get_field_title('size')

            # 提取订单手续费
            fees = self_col.get_field_arr('fees')
            fees_title = self_col.get_field_title('fees')

            # 提取订单价格
            price = self_col.get_field_arr('price')
            price_title = self_col.get_field_title('price')

            # 提取订单方向（买入或卖出）
            side = self_col.get_field_arr('side')

            # 处理买入订单的绘制
            buy_mask = side == OrderSide.Buy  # 创建买入订单的布尔掩码
            if buy_mask.any():  # 如果存在买入订单
                # 为买入订单准备自定义数据（用于鼠标悬停显示）
                buy_customdata = np.stack((
                    id_[buy_mask],    # 买入订单的ID
                    size[buy_mask],   # 买入订单的数量
                    fees[buy_mask]    # 买入订单的手续费
                ), axis=1)
                
                # 创建买入订单的散点图对象
                buy_scatter = go.Scatter(
                    x=idx[buy_mask],          # X轴：时间（买入订单的时间点）
                    y=price[buy_mask],        # Y轴：价格（买入订单的执行价格）
                    mode='markers',           # 显示模式：仅显示标记点
                    marker=dict(
                        symbol='triangle-up',     # 标记形状：向上三角形
                        color=plotting_cfg['contrast_color_schema']['green'],  # 标记颜色：绿色
                        size=8,                   # 标记大小
                        line=dict(
                            width=1,              # 边框宽度
                            color=adjust_lightness(plotting_cfg['contrast_color_schema']['green'])  # 边框颜色：调暗的绿色
                        )
                    ),
                    name='Buy',               # 图例名称
                    customdata=buy_customdata, # 自定义数据，用于悬停显示
                    # 悬停模板：定义鼠标悬停时显示的信息格式
                    hovertemplate=f"{id_title}: %{{customdata[0]}}"           # 订单ID
                                  f"<br>{idx_title}: %{{x}}"                  # 时间
                                  f"<br>{price_title}: %{{y}}"                # 价格
                                  f"<br>{size_title}: %{{customdata[1]:.6f}}" # 数量（6位小数）
                                  f"<br>{fees_title}: %{{customdata[2]:.6f}}" # 手续费（6位小数）
                )
                # 应用用户自定义的买入标记样式
                buy_scatter.update(**buy_trace_kwargs)
                # 将买入标记添加到图表中
                fig.add_trace(buy_scatter, **add_trace_kwargs)

            # 处理卖出订单的绘制（逻辑与买入订单类似）
            sell_mask = side == OrderSide.Sell  # 创建卖出订单的布尔掩码
            if sell_mask.any():  # 如果存在卖出订单
                # 为卖出订单准备自定义数据
                sell_customdata = np.stack((
                    id_[sell_mask],   # 卖出订单的ID
                    size[sell_mask],  # 卖出订单的数量
                    fees[sell_mask]   # 卖出订单的手续费
                ), axis=1)
                
                # 创建卖出订单的散点图对象
                sell_scatter = go.Scatter(
                    x=idx[sell_mask],         # X轴：时间
                    y=price[sell_mask],       # Y轴：价格
                    mode='markers',           # 仅显示标记
                    marker=dict(
                        symbol='triangle-down',   # 标记形状：向下三角形
                        color=plotting_cfg['contrast_color_schema']['red'],  # 标记颜色：红色
                        size=8,                   # 标记大小
                        line=dict(
                            width=1,              # 边框宽度
                            color=adjust_lightness(plotting_cfg['contrast_color_schema']['red'])  # 边框颜色：调暗的红色
                        )
                    ),
                    name='Sell',              # 图例名称
                    customdata=sell_customdata, # 自定义数据
                    # 悬停模板
                    hovertemplate=f"{id_title}: %{{customdata[0]}}"
                                  f"<br>{idx_title}: %{{x}}"
                                  f"<br>{price_title}: %{{y}}"
                                  f"<br>{size_title}: %{{customdata[1]:.6f}}"
                                  f"<br>{fees_title}: %{{customdata[2]:.6f}}"
                )
                # 应用用户自定义的卖出标记样式
                sell_scatter.update(**sell_trace_kwargs)
                # 将卖出标记添加到图表中
                fig.add_trace(sell_scatter, **add_trace_kwargs)

        # 返回完成的图表对象
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """
        Orders绘图方法的默认配置
        
        获取Orders.plots方法使用的默认配置参数。该配置合并了：
        1. Records基类的默认绘图配置
        2. vectorbt全局设置中orders模块的绘图配置
        
        Returns:
            tp.Kwargs: 默认绘图配置字典
        
        配置内容通常包括：
        - 图表类型和样式设置
        - 颜色方案和主题配置
        - 子图布局参数
        - 交互功能设置
        
        Examples:
            >>> orders = pf.orders
            >>> defaults = orders.plots_defaults
            >>> print("默认绘图配置:", defaults)
            >>> 
            >>> # 使用自定义配置绘图
            >>> orders.plots(**defaults, title="自定义订单分析图")
        
        Defaults for `Orders.plots`.

        Merges `vectorbt.records.base.Records.plots_defaults` and
        `orders.plots` from `vectorbt._settings.settings`.
        """
        # 从vectorbt全局设置中获取orders模块的绘图配置
        from vectorbt._settings import settings
        orders_plots_cfg = settings['orders']['plots']

        # 合并父类的默认绘图配置和orders特定的绘图配置
        return merge_dicts(
            Records.plots_defaults.__get__(self),  # 获取Records基类的默认绘图配置
            orders_plots_cfg                        # 添加orders特定的绘图配置
        )

    # 定义子图配置，用于Orders.plots()方法
    _subplots: tp.ClassVar[Config] = Config(
        dict(
            plot=dict(
                title="Orders",                    # 子图标题
                yaxis_kwargs=dict(title="Price"),  # Y轴标题为"Price"
                check_is_not_grouped=True,         # 检查数据未分组（因为订单图不支持分组显示）
                plot_func='plot',                  # 使用plot方法进行绘制
                tags='orders'                      # 标签，用于分类和过滤
            )
        ),
        copy_kwargs=dict(copy_mode='deep')  # 深拷贝配置
    )

    @property
    def subplots(self) -> Config:
        """
        获取子图配置
        
        返回Orders类定义的子图配置，用于plots()方法创建图表组合。
        
        Returns:
            Config: 子图配置对象，定义了如何创建和布局图表
        
        配置说明：
        - plot: 主要的订单图表配置
          - title: 图表标题
          - yaxis_kwargs: Y轴配置（标题为"Price"）
          - check_is_not_grouped: 确保数据未分组
          - plot_func: 绘图函数名称
          - tags: 图表标签
        
        Examples:
            >>> orders = pf.orders
            >>> subplot_config = orders.subplots
            >>> print("子图配置:", subplot_config)
            >>> 
            >>> # 使用默认子图配置创建图表
            >>> fig = orders.plots()
            >>> fig.show()
        """
        return self._subplots


# 为Orders类生成字段配置相关的文档
Orders.override_field_config_doc(__pdoc__)
# 为Orders类生成统计指标相关的文档
Orders.override_metrics_doc(__pdoc__)
# 为Orders类生成子图配置相关的文档
Orders.override_subplots_doc(__pdoc__)
