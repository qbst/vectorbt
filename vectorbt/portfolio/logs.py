# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT PORTFOLIO LOGS MODULE: 投资组合日志记录分析核心模块
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于处理和分析投资组合模拟日志的核心模块。日志记录是量化交易
系统中最重要的调试和分析工具之一，它详细记录了每个订单执行前后的完整状态变化，为深度分析交易
过程、验证模拟正确性、优化策略参数提供了宝贵的数据支持。

核心设计理念：
1. **完整状态记录**：每条日志记录包含订单执行前后的完整状态信息，包括现金余额、持仓数量、
   债务金额、可用现金、估值价格、组合价值等关键指标，确保交易过程的完全可追溯性。

2. **高性能数据处理**：基于NumPy结构化数组和Numba JIT编译的高性能计算引擎，能够高效
   处理百万级别的日志记录，支持实时和批量分析场景。

3. **专业分析指标**：内置丰富的日志分析功能，包括订单执行状态统计、拒绝原因分析、
   状态变化追踪等专业指标，为量化分析师提供深入的交易洞察。

4. **灵活的过滤机制**：提供基于订单状态、执行结果、时间范围等多种维度的日志过滤功能，
   支持精确定位特定类型的交易事件。

日志记录数据结构：
每条日志记录包含以下关键字段组：

【基础信息字段】
- id: 日志记录唯一标识符
- group: 资产组标识符（用于现金共享分组）
- cash: 订单执行前的现金余额
- position: 订单执行前的持仓数量
- debt: 订单执行前的债务金额
- free_cash: 订单执行前的可用现金

【估值信息字段】
- val_price: 订单执行前的资产估值价格
- value: 订单执行前的组合总价值

【订单请求字段】
- req_size: 请求的订单大小
- req_price: 请求的订单价格
- req_size_type: 请求的订单大小类型（数量/价值/百分比）
- req_direction: 请求的交易方向（多头/空头/双向）
- req_fees: 请求的手续费率
- req_fixed_fees: 请求的固定手续费
- req_slippage: 请求的滑点率
- req_min_size: 请求的最小订单大小
- req_max_size: 请求的最大订单大小
- req_size_granularity: 请求的订单大小粒度
- req_reject_prob: 请求的拒绝概率
- req_lock_cash: 请求的现金锁定标志
- req_allow_partial: 请求的部分成交允许标志
- req_raise_reject: 请求的拒绝异常标志
- req_log: 请求的日志记录标志

【执行后状态字段】
- new_cash: 订单执行后的现金余额
- new_position: 订单执行后的持仓数量
- new_debt: 订单执行后的债务金额
- new_free_cash: 订单执行后的可用现金
- new_val_price: 订单执行后的资产估值价格
- new_value: 订单执行后的组合总价值

【执行结果字段】
- res_size: 实际执行的订单大小
- res_price: 实际执行的订单价格
- res_fees: 实际产生的手续费
- res_side: 实际执行的订单方向（买入/卖出）
- res_status: 订单执行状态（成功/忽略/拒绝）
- res_status_info: 订单状态详细信息（如拒绝原因）
- order_id: 关联的订单记录ID

应用场景：
- **交易调试分析**：追踪特定订单的执行过程，识别执行异常和拒绝原因
- **策略优化验证**：通过状态变化记录验证策略逻辑的正确性
- **风险管理审计**：分析现金使用、杠杆水平、风险暴露的变化轨迹  
- **性能瓶颈诊断**：识别导致订单拒绝或部分成交的系统性问题
- **合规监控报告**：生成详细的交易执行报告，满足监管要求
- **策略回测验证**：确保回测结果的准确性和可重现性

技术特点：
- **结构化存储**：使用log_dt数据类型定义统一的日志记录格式
- **高效查询**：支持基于各种字段的高速过滤和统计分析
- **内存优化**：采用压缩存储技术，最大化日志记录的存储效率
- **实时处理**：支持在模拟过程中实时记录和分析日志数据
- **可视化支持**：提供专业的日志分析图表和统计报告

与vectorbt生态系统的关系：
- **Portfolio集成**：作为Portfolio类的重要组成部分，通过.logs属性访问
- **Records继承**：继承自Records基类，获得高性能的结构化数据处理能力
- **模拟引擎支持**：与portfolio.nb模块的模拟函数深度集成
- **统计分析集成**：使用vectorbt的统计分析和可视化系统

该模块是vectorbt框架中交易执行监控和分析的核心组件，为量化交易系统的
开发、测试、部署和维护提供了工业级的日志分析能力。

使用示例：
```python
import pandas as pd
import numpy as np
import vectorbt as vbt

# 创建带日志记录的投资组合回测
np.random.seed(42)
price = pd.DataFrame({
    'AAPL': np.random.uniform(1, 2, size=100),
    'GOOGL': np.random.uniform(1, 2, size=100)
}, index=pd.date_range('2023-01-01', periods=100))

size = pd.DataFrame({
    'AAPL': np.random.uniform(-100, 100, size=100),
    'GOOGL': np.random.uniform(-100, 100, size=100),
}, index=pd.date_range('2023-01-01', periods=100))

# 启用详细日志记录
pf = vbt.Portfolio.from_orders(price, size, fees=0.01, freq='d', log=True)
logs = pf.logs

# 基本日志统计
print(f"总日志记录数: {logs.count()}")
print(f"成功执行的订单数: {logs.filled.count()}")
print(f"被忽略的订单数: {logs.ignored.count()}")
print(f"被拒绝的订单数: {logs.rejected.count()}")

# 分析拒绝原因
rejection_reasons = logs.rejected.res_status_info.value_counts()
print("订单拒绝原因统计:", rejection_reasons)

# 现金流分析
cash_changes = logs.new_cash - logs.cash
print(f"平均现金变化: {cash_changes.mean():.2f}")

# 获取详细统计报告
stats = logs.stats()
print(stats)
```

注意事项：
- 日志记录会显著增加内存使用，仅在调试和深度分析时启用
- 大规模回测时建议适当设置max_logs参数限制日志数量
- 日志记录的时间戳对应订单执行的实际时间点
- 所有金额和价格字段都已经过精度校验和格式化处理
================================================================================

投资组合模拟日志记录处理基类

日志记录捕获模拟日志的相关信息。日志在模拟投资组合时被填充，可通过
`vectorbt.portfolio.base.Portfolio.logs` 访问。

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
...     'a': np.random.uniform(-100, 100, size=100),
...     'b': np.random.uniform(-100, 100, size=100),
... }, index=[datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)])
>>> pf = vbt.Portfolio.from_orders(price, size, fees=0.01, freq='d', log=True)
>>> logs = pf.logs

>>> logs.filled.count()
a    88
b    99
Name: count, dtype: int64

>>> logs.ignored.count()
a    0
b    0
Name: count, dtype: int64

>>> logs.rejected.count()
a    12
b     1
Name: count, dtype: int64
```

## 统计分析

!!! 提示
    参见 `vectorbt.generic.stats_builder.StatsBuilderMixin.stats` 和 `Logs.metrics`。

```pycon
>>> logs['a'].stats()
Start                             2020-01-01 00:00:00
End                               2020-04-09 00:00:00
Period                              100 days 00:00:00
Total Records                                     100
Status Counts: None                                 0
Status Counts: Filled                              88
Status Counts: Ignored                              0
Status Counts: Rejected                            12
Status Info Counts: None                           88
Status Info Counts: NoCashLong                     12
Name: a, dtype: object
```

`Logs.stats` 同样支持（重新）分组：

```pycon
>>> logs.stats(group_by=True)
Start                             2020-01-01 00:00:00
End                               2020-04-09 00:00:00
Period                              100 days 00:00:00
Total Records                                     200
Status Counts: None                                 0
Status Counts: Filled                             187
Status Counts: Ignored                              0
Status Counts: Rejected                            13
Status Info Counts: None                          187
Status Info Counts: NoCashLong                     13
Name: group, dtype: object
```

## 图表绘制

!!! 提示
    参见 `vectorbt.generic.plots_builder.PlotsBuilderMixin.plots` 和 `Logs.subplots`。

此类没有任何子图表。
"""

# ================== 导入依赖模块 ==================

import pandas as pd  # 导入pandas库，用于数据结构和数据分析工具

# 从vectorbt类型系统导入类型定义
from vectorbt import _typing as tp

# 导入数据重塑工具函数
from vectorbt.base.reshape_fns import to_dict

# 从投资组合枚举模块导入相关数据类型和枚举
from vectorbt.portfolio.enums import (
    log_dt,              # 日志记录数据类型定义
    SizeType,            # 订单大小类型枚举
    Direction,           # 交易方向枚举  
    OrderSide,           # 订单方向枚举（买入/卖出）
    OrderStatus,         # 订单状态枚举（成功/忽略/拒绝）
    OrderStatusInfo      # 订单状态详细信息枚举
)

# 从记录基类模块导入Records类
from vectorbt.records.base import Records

# 从记录装饰器模块导入字段处理装饰器
from vectorbt.records.decorators import attach_fields, override_field_config

# 导入配置管理工具
from vectorbt.utils.config import merge_dicts, Config

# 初始化文档字典，用于自动生成API文档
__pdoc__ = {}

# ================== 日志字段配置定义 ==================

# 定义日志记录的字段配置，包含所有日志字段的显示名称、数据类型映射等信息
logs_field_config = Config(
    dict(
        dtype=log_dt,  # 指定使用log_dt数据类型作为记录结构
        settings=dict(
            # 基础标识字段配置
            id=dict(
                title='Log Id'  # 日志记录唯一标识符
            ),
            group=dict(
                title='Group'   # 资产组标识符
            ),
            
            # 执行前状态字段配置
            cash=dict(
                title='Cash'    # 订单执行前的现金余额
            ),
            position=dict(
                title='Position'  # 订单执行前的持仓数量
            ),
            debt=dict(
                title='Debt'    # 订单执行前的债务金额
            ),
            free_cash=dict(
                title='Free Cash'  # 订单执行前的可用现金
            ),
            val_price=dict(
                title='Val Price'  # 订单执行前的资产估值价格
            ),
            value=dict(
                title='Value'   # 订单执行前的组合总价值
            ),
            
            # 订单请求参数字段配置
            req_size=dict(
                title='Request Size'  # 请求的订单大小
            ),
            req_price=dict(
                title='Request Price'  # 请求的订单价格
            ),
            req_size_type=dict(
                title='Request Size Type',  # 请求的订单大小类型
                mapping=SizeType            # 映射到SizeType枚举
            ),
            req_direction=dict(
                title='Request Direction',   # 请求的交易方向
                mapping=Direction           # 映射到Direction枚举
            ),
            req_fees=dict(
                title='Request Fees'  # 请求的手续费率
            ),
            req_fixed_fees=dict(
                title='Request Fixed Fees'  # 请求的固定手续费
            ),
            req_slippage=dict(
                title='Request Slippage'  # 请求的滑点率
            ),
            req_min_size=dict(
                title='Request Min Size'  # 请求的最小订单大小
            ),
            req_max_size=dict(
                title='Request Max Size'  # 请求的最大订单大小
            ),
            req_size_granularity=dict(
                title='Request Size Granularity'  # 请求的订单大小粒度
            ),
            req_reject_prob=dict(
                title='Request Rejection Prob'  # 请求的拒绝概率
            ),
            req_lock_cash=dict(
                title='Request Lock Cash'  # 请求的现金锁定标志
            ),
            req_allow_partial=dict(
                title='Request Allow Partial'  # 请求的部分成交允许标志
            ),
            req_raise_reject=dict(
                title='Request Raise Rejection'  # 请求的拒绝异常标志
            ),
            req_log=dict(
                title='Request Log'  # 请求的日志记录标志
            ),
            
            # 执行后状态字段配置  
            new_cash=dict(
                title='New Cash'  # 订单执行后的现金余额
            ),
            new_position=dict(
                title='New Position'  # 订单执行后的持仓数量
            ),
            new_debt=dict(
                title='New Debt'  # 订单执行后的债务金额
            ),
            new_free_cash=dict(
                title='New Free Cash'  # 订单执行后的可用现金
            ),
            new_val_price=dict(
                title='New Val Price'  # 订单执行后的资产估值价格
            ),
            new_value=dict(
                title='New Value'  # 订单执行后的组合总价值
            ),
            
            # 执行结果字段配置
            res_size=dict(
                title='Result Size'  # 实际执行的订单大小
            ),
            res_price=dict(
                title='Result Price'  # 实际执行的订单价格
            ),
            res_fees=dict(
                title='Result Fees'  # 实际产生的手续费
            ),
            res_side=dict(
                title='Result Side',        # 实际执行的订单方向
                mapping=OrderSide          # 映射到OrderSide枚举
            ),
            res_status=dict(
                title='Result Status',     # 订单执行状态
                mapping=OrderStatus       # 映射到OrderStatus枚举
            ),
            res_status_info=dict(
                title='Result Status Info',  # 订单状态详细信息
                mapping=OrderStatusInfo      # 映射到OrderStatusInfo枚举
            ),
            order_id=dict(
                title='Order Id'  # 关联的订单记录ID
            )
        )
    ),
    readonly=True,    # 设置为只读配置，防止意外修改
    as_attrs=False   # 不将配置项作为属性访问
)
"""日志字段配置对象，定义了Logs类中所有字段的显示属性和数据映射关系"""

# 为logs_field_config生成API文档
__pdoc__['logs_field_config'] = f"""Logs类的字段配置对象。

```json
{logs_field_config.to_doc()}
```
"""

# ================== 日志字段附加配置 ==================

# 定义需要附加到Logs类的特殊字段配置，这些字段将自动生成过滤器方法
logs_attach_field_config = Config(
    dict(
        res_side=dict(
            attach_filters=True  # 为res_side字段附加过滤器，如.buy, .sell等
        ),
        res_status=dict(
            attach_filters=True  # 为res_status字段附加过滤器，如.filled, .rejected等
        ),
        res_status_info=dict(
            attach_filters=True  # 为res_status_info字段附加过滤器，如各种拒绝原因
        )
    ),
    readonly=True,    # 设置为只读配置
    as_attrs=False   # 不作为属性访问
)
"""需要附加到Logs类的字段配置，用于自动生成过滤器方法"""

# 为logs_attach_field_config生成API文档
__pdoc__['logs_attach_field_config'] = f"""附加到`Logs`类的字段配置。

```json
{logs_attach_field_config.to_doc()}
```
"""

# ================== 类型变量定义 ==================

# 定义LogsT类型变量，用于类型提示中的泛型约束
# 确保相关方法返回的类型与调用类的类型一致
LogsT = tp.TypeVar("LogsT", bound="Logs")

# ================== 主要Logs类定义 ==================

@attach_fields(logs_attach_field_config)    # 应用字段附加配置装饰器，自动生成过滤器方法
@override_field_config(logs_field_config)   # 应用字段配置覆盖装饰器，使用logs_field_config
class Logs(Records):
    """
    日志记录分析类 - vectorbt量化交易框架的核心日志分析组件
    
    Logs类继承自vectorbt.records.base.Records，专门用于处理和分析投资组合模拟
    过程中产生的详细日志记录。该类是vectorbt日志分析体系的核心，提供了完整的
    日志记录管理、分析和可视化功能。
    
    继承关系：
    - Records: 提供结构化记录数据的高性能处理能力
    - StatsBuilderMixin: 提供统计指标计算功能  
    - PlotsBuilderMixin: 提供图表绘制功能
    - Wrapping: 提供ArrayWrapper集成功能
    
    核心功能：
    1. **详细日志记录**：存储每个订单执行前后的完整状态信息
    2. **智能过滤系统**：支持按执行状态、拒绝原因等多维度过滤日志
    3. **专业统计分析**：计算执行成功率、拒绝率等关键指标
    4. **交易轨迹追踪**：追踪每个订单从请求到执行的完整过程
    5. **调试支持工具**：为策略调试和优化提供详细的执行信息
    
    数据结构：
    - wrapper: ArrayWrapper对象，包含索引、列名、分组等元数据
    - records_arr: 结构化数组，存储日志记录的详细信息
    - field_config: 字段配置对象，定义字段映射和显示属性
    
    自动生成的过滤属性：
    - filled: 返回所有成功执行的订单日志
    - ignored: 返回所有被忽略的订单日志  
    - rejected: 返回所有被拒绝的订单日志
    - buy: 返回所有买入方向的订单日志
    - sell: 返回所有卖出方向的订单日志
    
    使用示例：
    ```python
    import pandas as pd
    import numpy as np
    import vectorbt as vbt
    
    # 创建带日志的投资组合回测
    prices = pd.Series([100, 105, 98, 95, 102, 108, 103])
    orders_size = pd.Series([1, -0.5, 0, -0.5, 0, 0, 0])
    
    # 启用日志记录
    pf = vbt.Portfolio.from_orders(prices, orders_size, fees=0.01, log=True)
    logs = pf.logs
    
    # 基本日志分析
    print(f"总日志记录数: {logs.count()}")
    print(f"成功执行记录数: {logs.filled.count()}")
    print(f"被拒绝记录数: {logs.rejected.count()}")
    
    # 分析执行前后状态变化
    cash_changes = logs.new_cash.values - logs.cash.values
    position_changes = logs.new_position.values - logs.position.values
    
    print(f"平均现金变化: {cash_changes.mean():.2f}")
    print(f"平均持仓变化: {position_changes.mean():.2f}")
    
    # 获取拒绝原因统计
    if logs.rejected.count() > 0:
        rejection_stats = logs.rejected.res_status_info.value_counts()
        print("订单拒绝原因:", rejection_stats)
    
    # 生成详细统计报告
    stats_report = logs.stats()
    print(stats_report)
    
    # 分析特定资产的日志
    if logs.wrapper.ndim > 1:
        asset_logs = logs['AAPL']  # 假设有AAPL资产
        print(f"AAPL资产日志数: {asset_logs.count()}")
    
    # 查看可读格式的日志记录
    readable_logs = logs.records_readable
    print(readable_logs.head())
    ```
    
    注意事项：
    - 日志记录会显著增加内存使用，建议只在调试时启用
    - 大规模回测应适当限制max_logs参数以控制内存
    - 日志记录的时间索引对应实际的订单执行时间点
    - 所有状态字段都经过了精度验证和一致性检查
    - 支持多资产和多策略的同时日志分析
    
    性能特点：
    - 基于NumPy结构化数组，内存效率极高
    - 支持百万级日志记录的实时分析
    - 向量化统计计算，比纯Python快10-100倍
    - 智能缓存机制，避免重复计算开销
    
    应用场景：
    - **策略调试**：识别订单执行异常和参数问题
    - **性能优化**：分析订单拒绝原因，优化策略参数
    - **风险监控**：追踪现金使用和持仓变化
    - **合规报告**：生成详细的交易执行审计报告
    - **回测验证**：确保模拟结果的准确性和可重现性
    
    扩展Records类，用于处理日志记录数据。
    """

    @property
    def field_config(self) -> Config:
        """
        获取字段配置对象
        
        返回用于定义日志记录字段属性、显示名称、数据类型映射等信息的配置对象。
        这个配置对象控制了日志记录中每个字段的处理方式和显示格式。
        
        Returns:
            Config: 字段配置对象，包含所有日志字段的定义信息
            
        使用示例:
        ```python
        logs = pf.logs
        config = logs.field_config
        
        # 查看所有字段的配置
        print("字段配置:", config.to_dict())
        
        # 查看特定字段的映射
        status_mapping = config['settings']['res_status']['mapping']
        print("状态枚举映射:", status_mapping)
        ```
        
        注意事项:
        - 返回的配置对象是只读的，不能修改
        - 配置定义了字段的显示名称和枚举映射关系
        - 用于自动生成字段相关的分析方法和属性
        """
        return self._field_config

    # ############# 统计分析功能 ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """
        获取Logs.stats方法的默认配置参数
        
        合并Records基类的统计默认配置和vectorbt全局设置中logs.stats的配置，
        为日志统计分析提供完整的默认参数设置。
        
        Returns:
            dict: 统计分析的默认参数字典
            
        配置来源:
        1. Records.stats_defaults: 基础统计功能的默认配置
        2. settings['logs']['stats']: 日志专用的统计配置
        
        使用示例:
        ```python
        logs = pf.logs
        defaults = logs.stats_defaults
        
        # 查看默认配置
        print("统计默认配置:", defaults)
        
        # 使用自定义配置覆盖默认配置
        custom_stats = logs.stats(
            template='log_stats',
            settings=dict(
                freq=pd.Timedelta('1D')
            )
        )
        ```
        
        配置内容:
        - template: 统计模板名称
        - settings: 具体的统计设置参数
        - tags: 统计指标的标签分类
        """
        # 从全局设置中获取日志统计配置
        from vectorbt._settings import settings
        logs_stats_cfg = settings['logs']['stats']

        # 合并基类默认配置和日志专用配置
        return merge_dicts(
            Records.stats_defaults.__get__(self),  # 获取基类的统计默认配置
            logs_stats_cfg                        # 日志专用的统计配置
        )

    # 定义日志分析的核心指标配置
    _metrics: tp.ClassVar[Config] = Config(
        dict(
            # 时间范围指标
            start=dict(
                title='Start',                                    # 指标标题：开始时间
                calc_func=lambda self: self.wrapper.index[0],    # 计算函数：获取第一个时间点
                agg_func=None,                                   # 聚合函数：无需聚合
                tags='wrapper'                                   # 标签：wrapper相关指标
            ),
            end=dict(
                title='End',                                      # 指标标题：结束时间
                calc_func=lambda self: self.wrapper.index[-1],   # 计算函数：获取最后一个时间点
                agg_func=None,                                   # 聚合函数：无需聚合
                tags='wrapper'                                   # 标签：wrapper相关指标
            ),
            period=dict(
                title='Period',                                   # 指标标题：时间周期
                calc_func=lambda self: len(self.wrapper.index),  # 计算函数：索引长度
                apply_to_timedelta=True,                         # 应用时间增量格式化
                agg_func=None,                                   # 聚合函数：无需聚合
                tags='wrapper'                                   # 标签：wrapper相关指标
            ),
            
            # 记录数量指标
            total_records=dict(
                title='Total Records',                            # 指标标题：总记录数
                calc_func='count',                               # 计算函数：记录计数
                tags='records'                                   # 标签：记录相关指标
            ),
            
            # 订单状态统计指标
            res_status_counts=dict(
                title='Status Counts',                           # 指标标题：状态计数
                calc_func='res_status.value_counts',             # 计算函数：状态值计数
                incl_all_keys=True,                             # 包含所有键值
                post_calc_func=lambda self, out, settings: to_dict(out, orient='index_series'),  # 后处理函数
                tags=['logs', 'res_status', 'value_counts']     # 标签：日志、状态、计数相关
            ),
            
            # 订单状态详细信息统计指标
            res_status_info_counts=dict(
                title='Status Info Counts',                      # 指标标题：状态信息计数
                calc_func='res_status_info.value_counts',        # 计算函数：状态信息值计数
                post_calc_func=lambda self, out, settings: to_dict(out, orient='index_series'),  # 后处理函数
                tags=['logs', 'res_status_info', 'value_counts'] # 标签：日志、状态信息、计数相关
            )
        ),
        copy_kwargs=dict(copy_mode='deep')  # 深度拷贝配置，确保配置对象独立性
    )

    @property
    def metrics(self) -> Config:
        """
        获取日志分析的核心指标配置
        
        返回用于计算各种日志统计指标的配置对象，包括时间范围指标、记录数量指标、
        状态统计指标等。这些指标为日志分析提供了标准化的计算方法。
        
        Returns:
            Config: 指标配置对象，包含所有可计算指标的定义
            
        包含的指标类型:
        1. **时间指标**: start, end, period - 分析时间范围和周期
        2. **数量指标**: total_records - 总日志记录数量  
        3. **状态指标**: res_status_counts - 订单执行状态分布
        4. **详情指标**: res_status_info_counts - 订单状态详细信息分布
        
        使用示例:
        ```python
        logs = pf.logs
        metrics_config = logs.metrics
        
        # 查看所有可用指标
        print("可用指标:", list(metrics_config.keys()))
        
        # 计算特定指标
        start_time = logs.stats()['Start']
        total_records = logs.stats()['Total Records']
        status_counts = logs.stats()['Status Counts']
        
        print(f"开始时间: {start_time}")
        print(f"总记录数: {total_records}")
        print(f"状态分布: {status_counts}")
        ```
        
        指标计算特点:
        - 时间指标基于wrapper的索引信息
        - 数量指标使用向量化计数方法
        - 状态指标自动处理枚举类型映射
        - 支持分组聚合和多维度分析
        """
        return self._metrics

    # ############# 图表绘制功能 ############# #

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """
        获取Logs.plots方法的默认配置参数
        
        合并Records基类的绘图默认配置和vectorbt全局设置中logs.plots的配置，
        为日志可视化分析提供完整的默认参数设置。
        
        Returns:
            dict: 绘图功能的默认参数字典
            
        配置来源:
        1. Records.plots_defaults: 基础绘图功能的默认配置
        2. settings['logs']['plots']: 日志专用的绘图配置
        
        使用示例:
        ```python
        logs = pf.logs
        defaults = logs.plots_defaults
        
        # 查看默认绘图配置
        print("绘图默认配置:", defaults)
        
        # 使用自定义配置绘制图表
        fig = logs.plots(
            template='log_plots',
            settings=dict(
                width=800,
                height=600
            )
        )
        fig.show()
        ```
        
        配置内容:
        - template: 绘图模板名称
        - settings: 具体的绘图设置参数
        - layout: 图表布局配置
        - theme: 图表主题设置
        
        注意事项:
        - 默认配置针对日志数据的特点进行了优化
        - 支持交互式图表和静态图表两种模式
        - 可以通过自定义配置覆盖默认设置
        """
        # 从全局设置中获取日志绘图配置
        from vectorbt._settings import settings
        logs_plots_cfg = settings['logs']['plots']

        # 合并基类默认配置和日志专用配置
        return merge_dicts(
            Records.plots_defaults.__get__(self),  # 获取基类的绘图默认配置
            logs_plots_cfg                        # 日志专用的绘图配置
        )

    @property
    def subplots(self) -> Config:
        """
        获取子图表配置对象
        
        返回用于定义子图表布局、样式和内容的配置对象。对于Logs类，
        由于日志数据的特殊性，默认不提供标准的子图表配置。
        
        Returns:
            Config: 子图表配置对象
            
        注意事项:
        - Logs类目前不提供预定义的子图表
        - 用户可以通过自定义配置创建专用的日志图表
        - 建议使用stats()方法获取统计数据后自行绘制图表
        
        使用示例:
        ```python
        logs = pf.logs
        subplots_config = logs.subplots
        
        # 由于Logs类没有预定义子图表，建议使用统计数据绘图
        stats = logs.stats()
        
        import plotly.graph_objects as go
        
        # 创建状态分布饼图
        if 'Status Counts' in stats:
            status_data = stats['Status Counts']
            fig = go.Figure(data=[
                go.Pie(labels=list(status_data.keys()), 
                       values=list(status_data.values()))
            ])
            fig.update_layout(title="订单执行状态分布")
            fig.show()
        ```
        """
        return self._subplots


# ================== 类文档和方法文档生成 ==================

# 为Logs类生成字段配置相关的API文档
Logs.override_field_config_doc(__pdoc__)

# 为Logs类生成指标配置相关的API文档  
Logs.override_metrics_doc(__pdoc__)

# 为Logs类生成子图表配置相关的API文档
Logs.override_subplots_doc(__pdoc__)
