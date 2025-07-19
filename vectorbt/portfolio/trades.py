# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT PORTFOLIO TRADES MODULE: 交易记录分析和管理核心模块
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于处理和分析交易记录的核心模块。该模块提供了完整的
交易记录管理体系，包括入场交易(EntryTrades)、退出交易(ExitTrades)和持仓(Positions)的
统一处理和分析功能。这是量化交易分析中最重要的模块之一，直接关系到策略绩效评估、
风险控制和交易决策优化。

核心设计理念：
1. **分层交易抽象**：将复杂的交易记录抽象为三个层次，便于不同角度的分析
   - EntryTrades: 从入场视角分析每笔开仓交易的表现
   - ExitTrades: 从退出视角分析每笔平仓交易的效果  
   - Positions: 从持仓视角分析完整的买入-持有-卖出周期

2. **高性能计算架构**：基于NumPy结构化数组和Numba JIT编译，实现接近C语言的计算性能，
   能够处理数百万条交易记录而不会出现性能瓶颈。

3. **专业交易指标**：内置丰富的交易分析指标，包括胜率、盈利因子、期望收益、SQN等
   量化交易中的标准评估指标，为策略优化提供量化依据。

4. **完整可视化支持**：提供专业的交易分析图表，包括交易PnL散点图、交易时间线图、
   收益分布图等，支持交互式分析和结果展示。

交易记录数据结构：
每个交易记录包含以下核心字段：
- id: 交易唯一标识符
- col: 列索引(资产/策略标识) 
- entry_idx/exit_idx: 入场/出场时间索引
- entry_price/exit_price: 入场/出场价格
- entry_fees/exit_fees: 入场/出场手续费
- size: 交易数量(正数买入，负数卖出)
- pnl: 净盈亏
- return: 收益率
- direction: 交易方向(Long/Short)
- status: 交易状态(Open/Closed)
- parent_id: 父级交易ID(用于交易聚合)

三种交易类型的区别：

【入场交易 EntryTrades】
- 以开仓订单为基础创建交易记录
- 每个开仓订单对应一个入场交易
- 退出信息是所有对应平仓订单的加权平均
- 适用于分析入场时机和入场策略的有效性

【退出交易 ExitTrades】  
- 以平仓订单为基础创建交易记录
- 每个平仓订单对应一个退出交易
- 入场信息是对应开仓订单的分摊结果
- 适用于分析出场时机和止盈止损策略的有效性

【持仓 Positions】
- 将连续的入场或退出交易聚合为持仓记录
- 一个持仓可能包含多次加仓和减仓操作
- 反映完整的投资周期表现
- 适用于分析整体投资策略的效果

应用场景：
- **策略绩效评估**：计算夏普比率、最大回撤、胜率等关键指标
- **交易成本分析**：评估手续费、滑点等交易成本对收益的影响
- **风险管理优化**：分析止损止盈策略的有效性
- **策略参数优化**：基于历史交易表现优化策略参数
- **交易心理分析**：分析连胜连败streaks对交易行为的影响

与vectorbt生态系统的关系：
- 从Orders模块接收订单数据并转换为交易记录
- 为Portfolio模块提供交易层面的分析功能
- 集成Records模块的高性能数据处理能力
- 使用Generic模块的统计分析和绘图功能

技术特点：
- 基于结构化数组的内存高效存储
- Numba编译的高性能计算函数
- 完整的字段配置和映射系统
- 丰富的统计指标和可视化功能
- 支持开放和封闭交易的分别处理

该模块是vectorbt框架中交易分析的核心，为量化交易策略的开发、测试和优化
提供了工业级的交易记录分析能力。

使用示例：
```python
import pandas as pd
import numpy as np
import vectorbt as vbt

# 1. 创建简单的价格数据和交易信号
prices = pd.Series([100, 105, 98, 95, 102, 108, 103])
orders = pd.Series([1, 0, 0, -0.5, 0, 0, -0.5])

# 2. 从订单创建投资组合
pf = vbt.Portfolio.from_orders(prices, orders)

# 3. 获取不同层次的交易分析
entry_trades = pf.entry_trades          # 入场交易分析
exit_trades = pf.exit_trades            # 退出交易分析  
positions = pf.positions                # 持仓分析

# 4. 计算关键交易指标
print("总交易次数:", entry_trades.count())
print("胜率:", entry_trades.win_rate())
print("盈利因子:", entry_trades.profit_factor())
print("期望收益:", entry_trades.expectancy())
print("夏普比率:", entry_trades.sqn())

# 5. 可视化交易结果
entry_trades.plot()                     # 交易时间线图
entry_trades.plot_pnl()                 # 盈亏散点图

# 6. 获取详细交易记录
readable_records = entry_trades.records_readable
print(readable_records)
```

警告和注意事项：
- 所有交易类型都返回开放和封闭的交易，可能影响绩效计算结果
- 要仅考虑封闭交易，应明确查询closed属性
- 交易记录的时间戳使用索引映射，确保时区和频率设置正确
- 手续费计算包含在PnL中，分析时需要考虑交易成本的影响

Base class for working with trade records.

Trade records capture information on trades.

In vectorbt, a trade is a sequence of orders that starts with an opening order and optionally ends
with a closing order. Every pair of opposite orders can be represented by a trade. Each trade has a PnL
info attached to quickly assess its performance. An interesting effect of this representation
is the ability to aggregate trades: if two or more trades are happening one after another in time,
they can be aggregated into a bigger trade. This way, for example, single-order trades can be aggregated
into positions; but also multiple positions can be aggregated into a single blob that reflects the performance
of the entire symbol.

!!! warning
    All classes return both closed AND open trades/positions, which may skew your performance results.
    To only consider closed trades/positions, you should explicitly query the `closed` attribute.

## Trade types

There are three main types of trades.

### Entry trades

An entry trade is created from each order that opens or adds to a position.

For example, if we have a single large buy order and 100 smaller sell orders, we will see
a single trade with the entry information copied from the buy order and the exit information being
a size-weighted average over the exit information of all sell orders. On the other hand,
if we have 100 smaller buy orders and a single sell order, we will see 100 trades,
each with the entry information copied from the buy order and the exit information being
a size-based fraction of the exit information of the sell order.

Use `vectorbt.portfolio.trades.EntryTrades.from_orders` to build entry trades from orders.
Also available as `vectorbt.portfolio.base.Portfolio.entry_trades`.

### Exit trades

An exit trade is created from each order that closes or removes from a position.

Use `vectorbt.portfolio.trades.ExitTrades.from_orders` to build exit trades from orders.
Also available as `vectorbt.portfolio.base.Portfolio.exit_trades`.

### Positions

A position is created from a sequence of entry or exit trades.

Use `vectorbt.portfolio.trades.Positions.from_trades` to build positions from entry or exit trades.
Also available as `vectorbt.portfolio.base.Portfolio.positions`.

## Example

* Increasing position:

```pycon
>>> import pandas as pd
>>> import numpy as np
>>> from datetime import datetime, timedelta
>>> import vectorbt as vbt

>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([1., 1., 1., 1., -4.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0
1         1       0   1.0                1              2.0         1.0
2         2       0   1.0                2              3.0         1.0
3         3       0   1.0                3              4.0         1.0

   Exit Timestamp  Avg Exit Price  Exit Fees   PnL  Return Direction  Status  \\
0               4             5.0       0.25  2.75  2.7500      Long  Closed
1               4             5.0       0.25  1.75  0.8750      Long  Closed
2               4             5.0       0.25  0.75  0.2500      Long  Closed
3               4             5.0       0.25 -0.25 -0.0625      Long  Closed

   Parent Id
0          0
1          0
2          0
3          0

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   4.0                0              2.5         4.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               4             5.0        1.0  5.0     0.5      Long  Closed

   Parent Id
0          0

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   4.0                0              2.5         4.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               4             5.0        1.0  5.0     0.5      Long  Closed

   Parent Id
0          0

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

* Decreasing position:

```pycon
>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([4., -1., -1., -1., -1.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   4.0                0              1.0         1.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               4             3.5        4.0  5.0    1.25      Long  Closed

   Parent Id
0          0

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0        0.25
1         1       0   1.0                0              1.0        0.25
2         2       0   1.0                0              1.0        0.25
3         3       0   1.0                0              1.0        0.25

   Exit Timestamp  Avg Exit Price  Exit Fees   PnL  Return Direction  Status  \\
0               1             2.0        1.0 -0.25   -0.25      Long  Closed
1               2             3.0        1.0  0.75    0.75      Long  Closed
2               3             4.0        1.0  1.75    1.75      Long  Closed
3               4             5.0        1.0  2.75    2.75      Long  Closed

   Parent Id
0          0
1          0
2          0
3          0

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   4.0                0              1.0         1.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               4             3.5        4.0  5.0    1.25      Long  Closed

   Parent Id
0          0

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

* Multiple reversing positions:

```pycon
>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([1., -2., 2., -2., 1.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0
1         1       0   1.0                1              2.0         0.5
2         2       0   1.0                2              3.0         0.5
3         3       0   1.0                3              4.0         0.5

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               1             2.0        0.5 -0.5  -0.500      Long  Closed
1               2             3.0        0.5 -2.0  -1.000     Short  Closed
2               3             4.0        0.5  0.0   0.000      Long  Closed
3               4             5.0        1.0 -2.5  -0.625     Short  Closed

   Parent Id
0          0
1          1
2          2
3          3

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0
1         1       0   1.0                1              2.0         0.5
2         2       0   1.0                2              3.0         0.5
3         3       0   1.0                3              4.0         0.5

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               1             2.0        0.5 -0.5  -0.500      Long  Closed
1               2             3.0        0.5 -2.0  -1.000     Short  Closed
2               3             4.0        0.5  0.0   0.000      Long  Closed
3               4             5.0        1.0 -2.5  -0.625     Short  Closed

   Parent Id
0          0
1          1
2          2
3          3

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0
1         1       0   1.0                1              2.0         0.5
2         2       0   1.0                2              3.0         0.5
3         3       0   1.0                3              4.0         0.5

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction  Status  \\
0               1             2.0        0.5 -0.5  -0.500      Long  Closed
1               2             3.0        0.5 -2.0  -1.000     Short  Closed
2               3             4.0        0.5  0.0   0.000      Long  Closed
3               4             5.0        1.0 -2.5  -0.625     Short  Closed

   Parent Id
0          0
1          1
2          2
3          3

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

* Open position:

```pycon
>>> # Entry trades
>>> pf_kwargs = dict(
...     close=pd.Series([1., 2., 3., 4., 5.]),
...     size=pd.Series([1., 0., 0., 0., 0.]),
...     fixed_fees=1.
... )
>>> entry_trades = vbt.Portfolio.from_orders(**pf_kwargs).entry_trades
>>> entry_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction Status  \\
0               4             5.0        0.0  3.0     3.0      Long   Open

   Parent Id
0          0

>>> # Exit trades
>>> exit_trades = vbt.Portfolio.from_orders(**pf_kwargs).exit_trades
>>> exit_trades.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction Status  \\
0               4             5.0        0.0  3.0     3.0      Long   Open

   Parent Id
0          0

>>> # Positions
>>> positions = vbt.Portfolio.from_orders(**pf_kwargs).positions
>>> positions.records_readable
   Trade Id  Column  Size  Entry Timestamp  Avg Entry Price  Entry Fees  \\
0         0       0   1.0                0              1.0         1.0

   Exit Timestamp  Avg Exit Price  Exit Fees  PnL  Return Direction Status  \\
0               4             5.0        0.0  3.0     3.0      Long   Open

   Parent Id
0          0

>>> entry_trades.pnl.sum() == exit_trades.pnl.sum() == positions.pnl.sum()
True
```

Get trade count, trade PnL, and winning trade PnL:

```pycon
>>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.])
>>> size = pd.Series([1., -0.5, -0.5, 2., -0.5, -0.5, -0.5])
>>> trades = vbt.Portfolio.from_orders(price, size).trades

>>> trades.count()
6

>>> trades.pnl.sum()
-3.0

>>> trades.winning.count()
2

>>> trades.winning.pnl.sum()
1.5
```

Get count and PnL of trades with duration of more than 2 days:

```pycon
>>> mask = (trades.records['exit_idx'] - trades.records['entry_idx']) > 2
>>> trades_filtered = trades.apply_mask(mask)
>>> trades_filtered.count()
2

>>> trades_filtered.pnl.sum()
-3.0
```

## Stats

!!! hint
    See `vectorbt.generic.stats_builder.StatsBuilderMixin.stats` and `Trades.metrics`.

```pycon
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

>>> pf.trades['a'].stats()
Start                                2020-01-01 00:00:00
End                                  2020-04-09 00:00:00
Period                                 100 days 00:00:00
First Trade Start                    2020-01-01 00:00:00
Last Trade End                       2020-04-09 00:00:00
Coverage                               100 days 00:00:00
Overlap Coverage                        97 days 00:00:00
Total Records                                         48
Total Long Trades                                     22
Total Short Trades                                    26
Total Closed Trades                                   47
Total Open Trades                                      1
Open Trade PnL                                 -1.290981
Win Rate [%]                                    51.06383
Max Win Streak                                         3
Max Loss Streak                                        3
Best Trade [%]                                 43.326077
Worst Trade [%]                               -59.478304
Avg Winning Trade [%]                          21.418522
Avg Losing Trade [%]                          -18.856792
Avg Winning Trade Duration              22 days 22:00:00
Avg Losing Trade Duration     29 days 01:02:36.521739130
Profit Factor                                   0.976634
Expectancy                                     -0.001569
SQN                                            -0.064929
Name: a, dtype: object
```

Positions share almost identical metrics with trades:

```pycon
>>> pf.positions['a'].stats()
Start                            2020-01-01 00:00:00
End                              2020-04-09 00:00:00
Period                             100 days 00:00:00
Coverage [%]                                   100.0
First Position Start             2020-01-01 00:00:00
Last Position End                2020-04-09 00:00:00
Total Records                                      3
Total Long Positions                               2
Total Short Positions                              1
Total Closed Positions                             2
Total Open Positions                               1
Open Position PnL                          -0.929746
Win Rate [%]                                    50.0
Max Win Streak                                     1
Max Loss Streak                                    1
Best Position [%]                          39.498421
Worst Position [%]                          -3.32533
Avg Winning Position [%]                   39.498421
Avg Losing Position [%]                     -3.32533
Avg Winning Position Duration        1 days 00:00:00
Avg Losing Position Duration        47 days 00:00:00
Profit Factor                               0.261748
Expectancy                                 -0.217492
SQN                                        -0.585103
Name: a, dtype: object
```

To also include open trades/positions when calculating metrics such as win rate, pass `incl_open=True`:

```pycon
>>> pf.trades['a'].stats(settings=dict(incl_open=True))
Start                         2020-01-01 00:00:00
End                           2020-04-09 00:00:00
Period                          100 days 00:00:00
First Trade Start             2020-01-01 00:00:00
Last Trade End                2020-04-09 00:00:00
Coverage                        100 days 00:00:00
Overlap Coverage                 97 days 00:00:00
Total Records                                  48
Total Long Trades                              22
Total Short Trades                             26
Total Closed Trades                            47
Total Open Trades                               1
Open Trade PnL                          -1.290981
Win Rate [%]                             51.06383
Max Win Streak                                  3
Max Loss Streak                                 3
Best Trade [%]                          43.326077
Worst Trade [%]                        -59.478304
Avg Winning Trade [%]                   21.418522
Avg Losing Trade [%]                   -19.117677
Avg Winning Trade Duration       22 days 22:00:00
Avg Losing Trade Duration        30 days 00:00:00
Profit Factor                            0.693135
Expectancy                              -0.028432
SQN                                     -0.794284
Name: a, dtype: object
```

`Trades.stats` also supports (re-)grouping:

```pycon
>>> pf.trades.stats(group_by=True)
Start                                2020-01-01 00:00:00
End                                  2020-04-09 00:00:00
Period                                 100 days 00:00:00
First Trade Start                    2020-01-01 00:00:00
Last Trade End                       2020-04-09 00:00:00
Coverage                               100 days 00:00:00
Overlap Coverage                       100 days 00:00:00
Total Records                                        104
Total Long Trades                                     32
Total Short Trades                                    72
Total Closed Trades                                  102
Total Open Trades                                      2
Open Trade PnL                                 -1.790938
Win Rate [%]                                   46.078431
Max Win Streak                                         5
Max Loss Streak                                        5
Best Trade [%]                                 43.326077
Worst Trade [%]                               -87.793448
Avg Winning Trade [%]                          19.023926
Avg Losing Trade [%]                          -20.605892
Avg Winning Trade Duration    24 days 08:40:51.063829787
Avg Losing Trade Duration     25 days 11:20:43.636363636
Profit Factor                                   0.909581
Expectancy                                     -0.006035
SQN                                            -0.365593
Name: group, dtype: object
```

## Plots

!!! hint
    See `vectorbt.generic.plots_builder.PlotsBuilderMixin.plots` and `Trades.subplots`.

`Trades` class has two subplots based on `Trades.plot` and `Trades.plot_pnl`:

```pycon
>>> pf.trades['a'].plots(settings=dict(plot_zones=False)).show_svg()
```

![](/assets/images/trades_plots.svg)
"""

# 导入必要的库和模块
import numpy as np                           # NumPy数值计算库，用于高性能数组操作
import pandas as pd                          # Pandas数据分析库，用于时间序列和表格数据处理
import plotly.graph_objects as go            # Plotly图形对象，用于创建交互式金融图表

# 导入vectorbt内部模块
from vectorbt import _typing as tp                                    # 类型提示定义
from vectorbt.base.array_wrapper import ArrayWrapper                  # 数组包装器，提供pandas兼容的数组操作
from vectorbt.base.reshape_fns import to_1d_array, to_2d_array      # 数组维度转换函数
from vectorbt.generic.ranges import Ranges                           # 范围分析基类，提供时间区间分析功能
from vectorbt.portfolio import nb                                    # 投资组合的Numba编译函数模块
from vectorbt.portfolio.enums import TradeDirection, TradeStatus, trade_dt  # 交易方向、状态枚举和数据类型定义
from vectorbt.portfolio.orders import Orders                         # 订单记录处理类
from vectorbt.records.decorators import attach_fields, override_field_config  # 字段附加和配置覆盖装饰器
from vectorbt.records.mapped_array import MappedArray               # 映射数组类，用于稀疏数据高效处理
from vectorbt.utils.array_ import min_rel_rescale, max_rel_rescale  # 数组相对缩放工具函数
from vectorbt.utils.colors import adjust_lightness                  # 颜色亮度调整函数，用于图表美化
from vectorbt.utils.config import merge_dicts, Config              # 配置合并和配置类
from vectorbt.utils.decorators import cached_method, cached_property # 缓存装饰器，用于性能优化
from vectorbt.utils.figure import make_figure, get_domain          # 图形创建和域获取函数
from vectorbt.utils.template import RepEval                        # 模板表达式求值工具

__pdoc__ = {}  # 文档配置字典，用于控制API文档的生成

# ############# Trades 交易记录类 ############# #

# 交易记录字段配置 - 定义交易记录中各字段的属性和显示设置
trades_field_config = Config(
    dict(
        dtype=trade_dt,  # 使用trade_dt数据类型定义交易记录的结构
        settings={
            'id': dict(
                title='Trade Id'  # 交易唯一标识符，用于区分不同的交易记录
            ),
            'idx': dict(
                name='exit_idx'  # 将Records基类的idx字段重映射为exit_idx
            ),
            'start_idx': dict(
                name='entry_idx'  # 将Ranges基类的start_idx字段重映射为entry_idx
            ),
            'end_idx': dict(
                name='exit_idx'  # 将Ranges基类的end_idx字段重映射为exit_idx
            ),
            'size': dict(
                title='Size'  # 交易数量，正数表示买入，负数表示卖出
            ),
            'entry_idx': dict(
                title='Entry Timestamp',  # 入场时间戳
                mapping='index'  # 映射到索引，用于时间显示
            ),
            'entry_price': dict(
                title='Avg Entry Price'  # 平均入场价格，多次入场时的加权平均价
            ),
            'entry_fees': dict(
                title='Entry Fees'  # 入场手续费总额
            ),
            'exit_idx': dict(
                title='Exit Timestamp',  # 出场时间戳  
                mapping='index'  # 映射到索引，用于时间显示
            ),
            'exit_price': dict(
                title='Avg Exit Price'  # 平均出场价格，多次出场时的加权平均价
            ),
            'exit_fees': dict(
                title='Exit Fees'  # 出场手续费总额
            ),
            'pnl': dict(
                title='PnL'  # 净盈亏（Profit and Loss），包含手续费的总收益
            ),
            'return': dict(
                title='Return'  # 收益率，相对于投入资本的收益百分比
            ),
            'direction': dict(
                title='Direction',  # 交易方向
                mapping=TradeDirection  # 映射到TradeDirection枚举（Long/Short）
            ),
            'status': dict(
                title='Status',  # 交易状态
                mapping=TradeStatus  # 映射到TradeStatus枚举（Open/Closed）
            ),
            'parent_id': dict(
                title='Position Id'  # 父级持仓ID，用于将交易聚合为持仓
            )
        }
    ),
    readonly=True,    # 配置为只读，防止意外修改
    as_attrs=False    # 不作为属性访问，使用字典方式访问
)
"""_"""

__pdoc__['trades_field_config'] = f"""Field config for `Trades`.

```json
{trades_field_config.to_doc()}
```
"""

trades_attach_field_config = Config(
    {
        'return': dict(
            attach='returns'
        ),
        'direction': dict(
            attach_filters=True
        ),
        'status': dict(
            attach_filters=True,
            on_conflict='ignore'
        )
    },
    readonly=True,
    as_attrs=False
)
"""_"""

__pdoc__['trades_attach_field_config'] = f"""Config of fields to be attached to `Trades`.

```json
{trades_attach_field_config.to_doc()}
```
"""

# Trades类的类型变量，用于类型提示中的泛型约束，确保方法返回类型与调用类一致
TradesT = tp.TypeVar("TradesT", bound="Trades")


@attach_fields(trades_attach_field_config)  # 附加字段配置装饰器，自动添加字段相关的属性和方法
@override_field_config(trades_field_config)  # 覆盖字段配置装饰器，应用trades_field_config配置
class Trades(Ranges):
    """
    交易记录分析基类 - vectorbt量化交易框架的核心交易分析类
    
    Trades类继承自vectorbt.generic.ranges.Ranges，专门用于处理交易类记录数据，
    包括入场交易、出场交易和持仓记录。该类是vectorbt交易分析体系的基础，
    提供了完整的交易记录管理和分析功能。
    
    继承关系：
    - Ranges: 提供时间区间分析的基础功能
    - Records: 提供结构化记录数据的处理能力
    - StatsBuilderMixin: 提供统计指标计算功能
    - PlotsBuilderMixin: 提供图表绘制功能
    
    核心功能：
    1. **交易记录管理**：高效存储和访问交易记录数据
    2. **交易分析指标**：计算胜率、盈利因子、期望收益等关键指标
    3. **交易过滤功能**：筛选盈利/亏损交易、开放/封闭交易等
    4. **可视化分析**：提供专业的交易分析图表
    5. **数据聚合**：支持按时间、资产、策略等维度聚合分析
    
    数据结构：
    - wrapper: ArrayWrapper对象，包含索引、列名、分组等元数据
    - records_arr: 结构化数组，存储交易记录的详细信息
    - close: 参考价格序列，用于计算未实现收益
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建价格数据
    prices = pd.Series([100, 105, 98, 95, 102, 108, 103])
    
    # 创建交易订单
    orders = pd.Series([1, -0.5, 0, -0.5, 0, 0, 0])
    
    # 构建投资组合并获取交易记录
    pf = vbt.Portfolio.from_orders(prices, orders)
    trades = pf.trades
    
    # 分析交易表现
    print(f"总交易数: {trades.count()}")
    print(f"胜率: {trades.win_rate():.2%}")
    print(f"盈利因子: {trades.profit_factor():.2f}")
    print(f"期望收益: {trades.expectancy():.2f}")
    
    # 可视化分析
    trades.plot()        # 交易时间线图
    trades.plot_pnl()    # 盈亏散点图
    
    # 获取详细记录
    print(trades.records_readable)
    ```
    
    注意事项：
    - 默认返回所有交易（包括开放和封闭），使用.closed属性获取仅封闭交易
    - 交易记录按时间顺序排列，保持交易的先后关系
    - 支持多资产和多策略的同时分析
    - 手续费已计入PnL计算，无需额外处理
    
    参数：
        wrapper (ArrayWrapper): 数组包装器，包含索引和列信息
        records_arr (tp.RecordArray): 交易记录的结构化数组
        close (tp.ArrayLike): 参考收盘价数据，用于计算开放交易的未实现收益
        **kwargs: 传递给父类的其他参数
    """

    @property
    def field_config(self) -> Config:
        """
        获取字段配置对象
        
        返回当前交易记录类的字段配置，包含各字段的名称、类型、映射关系等信息。
        该配置控制着数据的显示格式、字段映射和类型转换等行为。
        
        Returns:
            Config: 字段配置对象
        """
        return self._field_config

    def __init__(self,
                 wrapper: ArrayWrapper,
                 records_arr: tp.RecordArray,
                 close: tp.ArrayLike,
                 **kwargs) -> None:
        """
        初始化Trades交易记录对象
        
        构造函数初始化交易记录对象，设置必要的数据结构和配置。
        继承自Ranges类的初始化过程，并额外保存参考价格数据。
        
        参数：
            wrapper (ArrayWrapper): 数组包装器，包含时间索引、资产列名、分组信息等元数据
            records_arr (tp.RecordArray): 交易记录的NumPy结构化数组，包含所有交易详细信息
            close (tp.ArrayLike): 参考收盘价序列，用于计算开放交易的市值和未实现盈亏
            **kwargs: 传递给父类Ranges的其他初始化参数
        
        处理流程：
            1. 调用父类Ranges的初始化方法，设置基础的区间分析功能
            2. 保存参考价格数据，用于后续的收益计算
            3. 应用字段配置，设置数据访问和显示格式
        """
        # 调用父类Ranges的初始化方法，传递所有必要参数
        Ranges.__init__(
            self,
            wrapper,
            records_arr,
            close=close,
            **kwargs
        )
        # 保存参考价格数据，用于计算开放交易的当前市值
        self._close = close

    def indexing_func(self: TradesT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> TradesT:
        """
        执行Trades对象的索引操作
        
        该方法实现了Trades对象的索引和切片功能，支持pandas风格的索引操作。
        当对Trades对象进行索引时（如trades['AAPL']或trades.iloc[:100]），
        该方法会被自动调用，确保索引操作正确地应用到交易记录数据和相关的元数据上。
        
        参数：
            pd_indexing_func (tp.PandasIndexingFunc): pandas索引函数，定义具体的索引操作
            **kwargs: 索引操作的额外参数
        
        返回：
            TradesT: 索引后的新Trades对象，包含筛选后的交易记录和相应的元数据
        
        处理流程：
            1. 调用父类的索引元数据处理方法，获取新的包装器和记录数组
            2. 如果存在参考价格数据，对其应用相同的列索引
            3. 创建并返回新的Trades对象
        
        示例：
            ```python
            # 获取特定资产的交易记录
            aapl_trades = trades['AAPL']
            
            # 获取前100条交易记录
            recent_trades = trades.iloc[:100]
            
            # 获取特定时间段的交易记录
            period_trades = trades.loc['2023-01-01':'2023-12-31']
            ```
        """
        # 调用父类Ranges的索引元数据处理方法，获取索引后的基础数据
        new_wrapper, new_records_arr, group_idxs, col_idxs = \
            Ranges.indexing_func_meta(self, pd_indexing_func, **kwargs)
        
        # 如果存在参考价格数据，需要对其应用相同的列索引操作
        if self.close is not None:
            # 将参考价格转换为2D数组并应用列索引，然后重新包装
            new_close = new_wrapper.wrap(to_2d_array(self.close)[:, col_idxs], group_by=False)
        else:
            new_close = None
        
        # 使用新的数据创建并返回新的Trades对象
        return self.replace(
            wrapper=new_wrapper,
            records_arr=new_records_arr,
            close=new_close
        )

    @property
    def close(self) -> tp.Optional[tp.SeriesFrame]:
        """
        获取参考价格数据（如收盘价）
        
        返回用作参考的价格序列，通常是收盘价数据。这些数据用于：
        1. 计算开放交易的当前市值
        2. 计算未实现盈亏
        3. 作为交易分析的价格基准
        
        Returns:
            tp.Optional[tp.SeriesFrame]: 参考价格的Series或DataFrame，如果未提供则为None
        
        使用示例：
            ```python
            # 获取参考价格数据
            reference_prices = trades.close
            
            # 检查是否有参考价格
            if trades.close is not None:
                print("当前参考价格:", trades.close.iloc[-1])
            ```
        """
        return self._close

    @classmethod
    def from_ts(cls: tp.Type[TradesT], *args, **kwargs) -> TradesT:
        """
        从时间序列创建Trades对象（未实现方法）
        
        该方法是预留的类方法，用于从时间序列数据直接创建Trades对象。
        目前尚未实现，调用时会抛出NotImplementedError异常。
        
        在未来的版本中，该方法可能用于：
        - 从价格序列和信号直接生成交易记录
        - 从外部交易数据源导入交易记录
        - 提供更灵活的Trades对象创建方式
        
        参数：
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            TradesT: Trades对象（当前未实现）
        
        Raises:
            NotImplementedError: 方法尚未实现
        """
        raise NotImplementedError

    @cached_property
    def winning(self: TradesT) -> TradesT:
        """
        获取盈利交易记录
        
        筛选出所有盈利的交易记录（PnL > 0）。这是交易分析中的重要子集，
        用于分析盈利交易的特征、频率和模式。
        
        Returns:
            TradesT: 包含所有盈利交易的Trades对象
        
        应用场景：
            - 分析盈利交易的平均持续时间
            - 计算盈利交易的平均收益率
            - 研究盈利交易的时间分布模式
            - 评估策略的盈利能力
        
        使用示例：
            ```python
            # 获取所有盈利交易
            winning_trades = trades.winning
            
            # 分析盈利交易的统计信息
            print(f"盈利交易数量: {winning_trades.count()}")
            print(f"平均盈利: {winning_trades.pnl.mean():.2f}")
            print(f"最大盈利: {winning_trades.pnl.max():.2f}")
            print(f"平均盈利率: {winning_trades.returns.mean():.2%}")
            
            # 获取盈利交易的详细记录
            winning_records = winning_trades.records_readable
            print(winning_records)
            ```
        """
        # 创建过滤掩码，选择PnL大于0的交易
        filter_mask = self.values['pnl'] > 0.
        # 应用掩码过滤，返回新的Trades对象
        return self.apply_mask(filter_mask)

    @cached_property
    def losing(self: TradesT) -> TradesT:
        """
        获取亏损交易记录
        
        筛选出所有亏损的交易记录（PnL < 0）。这是风险分析的重要子集，
        用于研究亏损模式、风险控制和止损策略的有效性。
        
        Returns:
            TradesT: 包含所有亏损交易的Trades对象
        
        应用场景：
            - 分析亏损交易的平均持续时间
            - 计算平均亏损幅度和最大单笔亏损
            - 研究亏损交易的时间聚集性
            - 评估止损策略的有效性
            - 分析风险控制措施的执行情况
        
        使用示例：
            ```python
            # 获取所有亏损交易
            losing_trades = trades.losing
            
            # 分析亏损交易的统计信息
            print(f"亏损交易数量: {losing_trades.count()}")
            print(f"平均亏损: {losing_trades.pnl.mean():.2f}")
            print(f"最大亏损: {losing_trades.pnl.min():.2f}")
            print(f"平均亏损率: {losing_trades.returns.mean():.2%}")
            
            # 分析亏损交易的持续时间
            avg_losing_duration = losing_trades.duration.mean()
            print(f"平均亏损交易持续时间: {avg_losing_duration}")
            
            # 获取亏损交易的详细记录
            losing_records = losing_trades.records_readable
            print(losing_records)
            ```
        """
        # 创建过滤掩码，选择PnL小于0的交易
        filter_mask = self.values['pnl'] < 0.
        # 应用掩码过滤，返回新的Trades对象
        return self.apply_mask(filter_mask)

    @cached_property
    def winning_streak(self) -> MappedArray:
        """
        计算每笔交易的连胜次数
        
        对于每笔交易，计算在其之前（包括自身）连续盈利交易的数量。
        这是交易心理分析的重要指标，用于研究连胜对后续交易决策的影响。
        
        Returns:
            MappedArray: 映射数组，包含每笔交易对应的连胜次数
        
        计算逻辑：
            - 从第一笔交易开始，逐笔检查交易结果
            - 如果当前交易盈利，连胜计数+1
            - 如果当前交易亏损，连胜计数重置为0
            - 返回每笔交易时刻的连胜状态
        
        应用场景：
            - 分析交易员在连胜后的行为变化
            - 研究连胜对风险承受能力的影响
            - 评估连胜后的决策质量
            - 识别过度自信导致的交易错误
        
        使用示例：
            ```python
            # 获取每笔交易的连胜次数
            win_streaks = trades.winning_streak
            
            # 找出最大连胜记录
            max_winning_streak = win_streaks.max()
            print(f"最大连胜次数: {max_winning_streak}")
            
            # 分析连胜分布
            streak_distribution = win_streaks.value_counts()
            print("连胜次数分布:")
            print(streak_distribution)
            
            # 获取连胜期间的交易记录
            long_streak_mask = win_streaks >= 3  # 连胜3次以上
            long_streak_trades = trades.apply_mask(long_streak_mask)
            print(f"连胜3次以上的交易数: {long_streak_trades.count()}")
            ```
        
        See Also:
            vectorbt.portfolio.nb.trade_winning_streak_nb: 底层Numba实现函数
        """
        # 应用Numba编译的连胜计算函数，返回int64类型的映射数组
        return self.apply(nb.trade_winning_streak_nb, dtype=np.int64)

    @cached_property
    def losing_streak(self) -> MappedArray:
        """
        计算每笔交易的连败次数
        
        对于每笔交易，计算在其之前（包括自身）连续亏损交易的数量。
        这是风险管理分析的重要指标，用于研究连败对交易心理和策略执行的影响。
        
        Returns:
            MappedArray: 映射数组，包含每笔交易对应的连败次数
        
        计算逻辑：
            - 从第一笔交易开始，逐笔检查交易结果
            - 如果当前交易亏损，连败计数+1
            - 如果当前交易盈利，连败计数重置为0
            - 返回每笔交易时刻的连败状态
        
        应用场景：
            - 分析交易员在连败后的心理状态变化
            - 研究连败对风险管理策略的影响
            - 评估连败后的恢复能力
            - 制定连败后的资金管理方案
            - 识别需要暂停交易的风险信号
        
        使用示例：
            ```python
            # 获取每笔交易的连败次数
            loss_streaks = trades.losing_streak
            
            # 找出最大连败记录
            max_losing_streak = loss_streaks.max()
            print(f"最大连败次数: {max_losing_streak}")
            
            # 分析连败分布
            streak_distribution = loss_streaks.value_counts()
            print("连败次数分布:")
            print(streak_distribution)
            
            # 获取连败期间的交易记录
            long_streak_mask = loss_streaks >= 3  # 连败3次以上
            long_streak_trades = trades.apply_mask(long_streak_mask)
            print(f"连败3次以上的交易数: {long_streak_trades.count()}")
            
            # 分析连败后的恢复情况
            recovery_trades = trades.apply_mask(loss_streaks == 0)  # 连败结束的交易
            recovery_win_rate = recovery_trades.winning.count() / recovery_trades.count()
            print(f"连败后恢复的胜率: {recovery_win_rate:.2%}")
            ```
        
        See Also:
            vectorbt.portfolio.nb.trade_losing_streak_nb: 底层Numba实现函数
        """
        # 应用Numba编译的连败计算函数，返回int64类型的映射数组
        return self.apply(nb.trade_losing_streak_nb, dtype=np.int64)

    @cached_method
    def win_rate(self, group_by: tp.GroupByLike = None,
                 wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算胜率（盈利交易比例）
        
        胜率是量化交易中最重要的指标之一，表示盈利交易占总交易数的比例。
        这是评估交易策略成功率的直观指标，但需要结合盈亏比一起分析。
        
        参数：
            group_by (tp.GroupByLike, optional): 分组方式，用于按资产、策略等维度分组计算
            wrap_kwargs (tp.KwargsLike, optional): 包装器参数，用于控制返回结果的格式
        
        返回：
            tp.MaybeSeries: 胜率，范围在0-1之间，可以是标量或Series
        
        计算公式：
            胜率 = 盈利交易数 / 总交易数
        
        解释说明：
            - 胜率 > 0.5：表示盈利交易多于亏损交易
            - 胜率 < 0.5：表示亏损交易多于盈利交易
            - 胜率 = 0.5：盈利和亏损交易数量相等
            - 单纯的高胜率并不保证整体盈利，还需考虑平均盈亏比
        
        应用场景：
            - 评估交易策略的基础表现
            - 比较不同策略的成功率
            - 制定资金管理和仓位控制策略
            - 评估交易员的技能水平
        
        使用示例：
            ```python
            # 计算整体胜率
            overall_win_rate = trades.win_rate()
            print(f"整体胜率: {overall_win_rate:.2%}")
            
            # 按资产分组计算胜率
            asset_win_rates = trades.win_rate(group_by=None)  # 每个资产的胜率
            print("各资产胜率:")
            print(asset_win_rates)
            
            # 结合盈亏比分析
            avg_win = trades.winning.pnl.mean()
            avg_loss = abs(trades.losing.pnl.mean())
            win_loss_ratio = avg_win / avg_loss
            print(f"胜率: {overall_win_rate:.2%}")
            print(f"盈亏比: {win_loss_ratio:.2f}")
            print(f"预期收益: {overall_win_rate * avg_win - (1-overall_win_rate) * avg_loss:.2f}")
            ```
        
        注意事项：
            - 胜率高不一定代表策略好，需要结合盈亏比分析
            - 样本量太小时胜率可能不具代表性
            - 考虑交易成本后的实际胜率可能会降低
        """
        # 获取盈利交易数量，转换为1维数组
        win_count = to_1d_array(self.winning.count(group_by=group_by))
        # 获取总交易数量，转换为1维数组
        total_count = to_1d_array(self.count(group_by=group_by))
        
        # 使用numpy的错误状态上下文，忽略除法错误和无效值错误（如0/0的情况）
        with np.errstate(divide='ignore', invalid='ignore'):
            # 计算胜率：盈利交易数 / 总交易数
            win_rate = win_count / total_count
        
        # 合并包装器参数，设置默认名称
        wrap_kwargs = merge_dicts(dict(name_or_index='win_rate'), wrap_kwargs)
        # 使用包装器包装结果并返回
        return self.wrapper.wrap_reduced(win_rate, group_by=group_by, **wrap_kwargs)

    @cached_method
    def profit_factor(self, group_by: tp.GroupByLike = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算盈利因子（盈利总额与亏损总额的比值）
        
        盈利因子是衡量交易策略盈利能力的重要指标，表示总盈利与总亏损的比值。
        这个指标综合考虑了胜率和盈亏比，是评估策略整体表现的关键指标。
        
        参数：
            group_by (tp.GroupByLike, optional): 分组方式，用于按资产、策略等维度分组计算
            wrap_kwargs (tp.KwargsLike, optional): 包装器参数，用于控制返回结果的格式
        
        返回：
            tp.MaybeSeries: 盈利因子，可以是标量或Series
        
        计算公式：
            盈利因子 = 总盈利金额 / |总亏损金额|
        
        解释说明：
            - 盈利因子 > 1.0：策略整体盈利，盈利总额超过亏损总额
            - 盈利因子 = 1.0：盈亏平衡，盈利总额等于亏损总额  
            - 盈利因子 < 1.0：策略整体亏损，亏损总额超过盈利总额
            - 盈利因子越高，策略的盈利能力越强
        
        行业标准：
            - 盈利因子 > 2.0：优秀的交易策略
            - 盈利因子 > 1.5：良好的交易策略
            - 盈利因子 > 1.0：可接受的交易策略
            - 盈利因子 < 1.0：需要改进的策略
        
        应用场景：
            - 评估策略的整体盈利能力
            - 比较不同策略的表现
            - 策略参数优化的目标函数
            - 风险调整后的收益评估
        
        使用示例：
            ```python
            # 计算整体盈利因子
            pf = trades.profit_factor()
            print(f"盈利因子: {pf:.2f}")
            
            # 判断策略表现
            if pf > 2.0:
                print("优秀的交易策略")
            elif pf > 1.5:
                print("良好的交易策略") 
            elif pf > 1.0:
                print("可接受的策略，但有改进空间")
            else:
                print("策略需要重新设计")
            
            # 按资产分组计算盈利因子
            asset_pf = trades.profit_factor(group_by=None)
            print("各资产盈利因子:")
            print(asset_pf.sort_values(ascending=False))
            
            # 分析盈利因子的构成
            total_profit = trades.winning.pnl.sum()
            total_loss = abs(trades.losing.pnl.sum())
            print(f"总盈利: {total_profit:.2f}")
            print(f"总亏损: {total_loss:.2f}")
            print(f"盈利因子: {total_profit / total_loss:.2f}")
            ```
        
        注意事项：
            - 如果没有亏损交易，盈利因子将为无穷大
            - 如果没有盈利交易，盈利因子将为0
            - 该指标不考虑交易次数和时间因素
            - 需要结合其他指标综合评估策略表现
        """
        # 计算总盈利金额，转换为1维数组
        total_win = to_1d_array(self.winning.pnl.sum(group_by=group_by))
        # 计算总亏损金额，转换为1维数组
        total_loss = to_1d_array(self.losing.pnl.sum(group_by=group_by))

        # 处理特殊情况：只有盈利或只有亏损的列，避免出现NaN值
        has_values = to_1d_array(self.count(group_by=group_by)) > 0
        # 如果没有盈利交易但有交易记录，将总盈利设为0
        total_win[np.isnan(total_win) & has_values] = 0.
        # 如果没有亏损交易但有交易记录，将总亏损设为0
        total_loss[np.isnan(total_loss) & has_values] = 0.

        # 使用numpy的错误状态上下文，忽略除0等错误
        with np.errstate(divide='ignore', invalid='ignore'):
            # 计算盈利因子：总盈利 / 总亏损的绝对值
            profit_factor = total_win / np.abs(total_loss)
        
        # 合并包装器参数，设置默认名称
        wrap_kwargs = merge_dicts(dict(name_or_index='profit_factor'), wrap_kwargs)
        # 使用包装器包装结果并返回
        return self.wrapper.wrap_reduced(profit_factor, group_by=group_by, **wrap_kwargs)

    @cached_method
    def expectancy(self, group_by: tp.GroupByLike = None,
                   wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算期望收益（平均盈利能力）
        
        期望收益是衡量交易策略长期盈利能力的重要指标，表示每笔交易的平均预期收益。
        该指标综合考虑了胜率、平均盈利和平均亏损，是评估策略可行性的关键数值。
        
        参数：
            group_by (tp.GroupByLike, optional): 分组方式，用于按资产、策略等维度分组计算
            wrap_kwargs (tp.KwargsLike, optional): 包装器参数，用于控制返回结果的格式
        
        返回：
            tp.MaybeSeries: 期望收益，可以是标量或Series
        
        计算公式：
            期望收益 = 胜率 × 平均盈利 - (1 - 胜率) × |平均亏损|
            
        数学表达：
            E(R) = P(Win) × Avg(Win) + P(Loss) × Avg(Loss)
            其中：P(Loss) = 1 - P(Win)，Avg(Loss) < 0
        
        解释说明：
            - 期望收益 > 0：策略长期来看是盈利的，值越大越好
            - 期望收益 = 0：策略长期盈亏平衡
            - 期望收益 < 0：策略长期来看是亏损的，需要改进
            - 该指标反映了每笔交易的平均预期收益
        
        应用场景：
            - 评估策略的长期盈利潜力
            - 策略参数优化的目标函数
            - 资金管理和仓位控制的依据
            - 比较不同策略的预期表现
            - 计算最优凯利比例等
        
        使用示例：
            ```python
            # 计算整体期望收益
            exp = trades.expectancy()
            print(f"期望收益: {exp:.2f}")
            
            # 判断策略可行性
            if exp > 0:
                print(f"策略可行，每笔交易平均预期收益: {exp:.2f}")
            else:
                print(f"策略不可行，每笔交易平均预期亏损: {abs(exp):.2f}")
            
            # 按资产分组计算期望收益
            asset_exp = trades.expectancy(group_by=None)
            print("各资产期望收益:")
            print(asset_exp.sort_values(ascending=False))
            
            # 分析期望收益的构成
            win_rate = trades.win_rate()
            avg_win = trades.winning.pnl.mean()
            avg_loss = abs(trades.losing.pnl.mean())
            print(f"胜率: {win_rate:.2%}")
            print(f"平均盈利: {avg_win:.2f}")
            print(f"平均亏损: {avg_loss:.2f}")
            print(f"期望收益: {win_rate * avg_win - (1-win_rate) * avg_loss:.2f}")
            
            # 计算所需胜率（盈亏平衡）
            breakeven_win_rate = avg_loss / (avg_win + avg_loss)
            print(f"盈亏平衡所需胜率: {breakeven_win_rate:.2%}")
            ```
        
        注意事项：
            - 期望收益基于历史数据，未来表现可能不同
            - 该指标假设交易结果独立分布
            - 需要足够的样本量确保统计显著性
            - 考虑交易成本对期望收益的影响
        """
        # 获取胜率，转换为1维数组
        win_rate = to_1d_array(self.win_rate(group_by=group_by))
        # 获取平均盈利，转换为1维数组
        avg_win = to_1d_array(self.winning.pnl.mean(group_by=group_by))
        # 获取平均亏损，转换为1维数组
        avg_loss = to_1d_array(self.losing.pnl.mean(group_by=group_by))

        # 处理特殊情况：只有盈利或只有亏损的列，避免出现NaN值
        has_values = to_1d_array(self.count(group_by=group_by)) > 0
        # 如果没有盈利交易但有交易记录，将平均盈利设为0
        avg_win[np.isnan(avg_win) & has_values] = 0.
        # 如果没有亏损交易但有交易记录，将平均亏损设为0
        avg_loss[np.isnan(avg_loss) & has_values] = 0.

        # 计算期望收益：胜率×平均盈利 - 败率×平均亏损的绝对值
        expectancy = win_rate * avg_win - (1 - win_rate) * np.abs(avg_loss)
        
        # 合并包装器参数，设置默认名称
        wrap_kwargs = merge_dicts(dict(name_or_index='expectancy'), wrap_kwargs)
        # 使用包装器包装结果并返回
        return self.wrapper.wrap_reduced(expectancy, group_by=group_by, **wrap_kwargs)

    @cached_method
    def sqn(self, group_by: tp.GroupByLike = None,
            wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算系统质量数（SQN - System Quality Number）
        
        SQN是由Van Tharp提出的系统质量评估指标，用于衡量交易系统的整体质量。
        该指标结合了交易次数、平均收益和收益稳定性，是评估交易系统可靠性的重要工具。
        
        参数：
            group_by (tp.GroupByLike, optional): 分组方式，用于按资产、策略等维度分组计算
            wrap_kwargs (tp.KwargsLike, optional): 包装器参数，用于控制返回结果的格式
        
        返回：
            tp.MaybeSeries: SQN值，可以是标量或Series
        
        计算公式：
            SQN = √交易次数 × 平均PnL / PnL标准差
        
        Van Tharp的SQN评级标准：
            - SQN ≥ 2.5：优秀系统（Excellent）
            - 1.6 ≤ SQN < 2.5：良好系统（Good）  
            - 0.6 ≤ SQN < 1.6：一般系统（Average）
            - 0 ≤ SQN < 0.6：较差系统（Below Average）
            - SQN < 0：劣质系统（Poor）
        
        指标意义：
            - SQN越高，系统质量越好
            - 考虑了交易频率对系统评估的影响
            - 综合评估收益与风险的平衡
            - 较高的SQN表示系统具有稳定的盈利能力
        
        应用场景：
            - 评估交易系统的整体质量
            - 比较不同策略的系统性表现
            - 策略优化和参数调整的指导
            - 风险调整后的表现评估
        
        使用示例：
            ```python
            # 计算系统质量数
            sqn_value = trades.sqn()
            print(f"SQN: {sqn_value:.2f}")
            
            # 根据SQN值评估系统质量
            if sqn_value >= 2.5:
                print("优秀的交易系统")
            elif sqn_value >= 1.6:
                print("良好的交易系统")
            elif sqn_value >= 0.6:
                print("一般的交易系统")
            elif sqn_value >= 0:
                print("较差的交易系统，需要改进")
            else:
                print("劣质系统，不建议使用")
            
            # 按资产分组计算SQN
            asset_sqn = trades.sqn(group_by=None)
            print("各资产SQN:")
            print(asset_sqn.sort_values(ascending=False))
            
            # 分析SQN的构成要素
            count = trades.count()
            pnl_mean = trades.pnl.mean()
            pnl_std = trades.pnl.std()
            print(f"交易次数: {count}")
            print(f"平均PnL: {pnl_mean:.4f}")
            print(f"PnL标准差: {pnl_std:.4f}")
            print(f"SQN: {np.sqrt(count) * pnl_mean / pnl_std:.2f}")
            
            # SQN与其他指标的比较
            profit_factor = trades.profit_factor()
            win_rate = trades.win_rate()
            print(f"SQN: {sqn_value:.2f}, 盈利因子: {profit_factor:.2f}, 胜率: {win_rate:.2%}")
            ```
        
        注意事项：
            - SQN对交易次数敏感，样本量过小时不够可靠
            - 该指标假设交易结果服从正态分布
            - 需要结合其他指标综合评估系统表现
            - 不同市场环境下的SQN可能存在差异
        """
        # 获取交易次数，转换为1维数组
        count = to_1d_array(self.count(group_by=group_by))
        # 获取PnL平均值，转换为1维数组
        pnl_mean = to_1d_array(self.pnl.mean(group_by=group_by))
        # 获取PnL标准差，转换为1维数组
        pnl_std = to_1d_array(self.pnl.std(group_by=group_by))
        
        # 计算SQN：√交易次数 × 平均PnL / PnL标准差
        sqn = np.sqrt(count) * pnl_mean / pnl_std
        
        # 合并包装器参数，设置默认名称
        wrap_kwargs = merge_dicts(dict(name_or_index='sqn'), wrap_kwargs)
        # 使用包装器包装结果并返回
        return self.wrapper.wrap_reduced(sqn, group_by=group_by, **wrap_kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Trades.stats`.

        Merges `vectorbt.generic.ranges.Ranges.stats_defaults` and
        `trades.stats` from `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        trades_stats_cfg = settings['trades']['stats']

        return merge_dicts(
            Ranges.stats_defaults.__get__(self),
            trades_stats_cfg
        )

    # 交易统计指标配置 - 定义了Trades类支持的所有统计指标和计算方法
    _metrics: tp.ClassVar[Config] = Config(
        dict(
            # 基础时间信息指标
            start=dict(
                title='Start',  # 分析开始时间
                calc_func=lambda self: self.wrapper.index[0],  # 获取第一个时间点
                agg_func=None,  # 不需要聚合函数
                tags='wrapper'  # 标记为包装器相关指标
            ),
            end=dict(
                title='End',  # 分析结束时间
                calc_func=lambda self: self.wrapper.index[-1],  # 获取最后一个时间点
                agg_func=None,  # 不需要聚合函数
                tags='wrapper'  # 标记为包装器相关指标
            ),
            period=dict(
                title='Period',  # 分析时间周期长度
                calc_func=lambda self: len(self.wrapper.index),  # 计算时间点总数
                apply_to_timedelta=True,  # 应用时间差处理，将数值转换为时间间隔
                agg_func=None,  # 不需要聚合函数
                tags='wrapper'  # 标记为包装器相关指标
            ),
            # 交易时间相关指标
            first_trade_start=dict(
                title='First Trade Start',  # 首次交易开始时间
                calc_func='entry_idx.nth',  # 获取第n个入场时间索引
                n=0,  # 获取第0个（第一个）交易
                wrap_kwargs=dict(to_index=True),  # 转换为索引时间格式
                tags=['trades', 'index']  # 标记为交易和索引相关
            ),
            last_trade_end=dict(
                title='Last Trade End',  # 最后交易结束时间
                calc_func='exit_idx.nth',  # 获取第n个退出时间索引
                n=-1,  # 获取第-1个（最后一个）交易
                wrap_kwargs=dict(to_index=True),  # 转换为索引时间格式
                tags=['trades', 'index']  # 标记为交易和索引相关
            ),
            
            # 时间覆盖度指标
            coverage=dict(
                title='Coverage',  # 交易时间覆盖度（非重叠）
                calc_func='coverage',  # 调用coverage计算方法
                overlapping=False,  # 不考虑重叠的交易时间
                normalize=False,  # 不进行标准化，返回绝对时间
                apply_to_timedelta=True,  # 应用时间差处理
                tags=['ranges', 'coverage']  # 标记为范围和覆盖度相关
            ),
            overlap_coverage=dict(
                title='Overlap Coverage',  # 交易时间覆盖度（含重叠）
                calc_func='coverage',  # 调用coverage计算方法
                overlapping=True,  # 考虑重叠的交易时间
                normalize=False,  # 不进行标准化，返回绝对时间
                apply_to_timedelta=True,  # 应用时间差处理
                tags=['ranges', 'coverage']  # 标记为范围和覆盖度相关
            ),
            
            # 交易数量统计指标
            total_records=dict(
                title='Total Records',  # 交易记录总数
                calc_func='count',  # 调用count计算方法
                tags='records'  # 标记为记录相关
            ),
            total_long_trades=dict(
                title='Total Long Trades',  # 多头交易总数
                calc_func='long.count',  # 调用多头交易的count方法
                tags=['trades', 'long']  # 标记为交易和多头相关
            ),
            total_short_trades=dict(
                title='Total Short Trades',  # 空头交易总数
                calc_func='short.count',  # 调用空头交易的count方法
                tags=['trades', 'short']  # 标记为交易和空头相关
            ),
            total_closed_trades=dict(
                title='Total Closed Trades',  # 已关闭交易总数
                calc_func='closed.count',  # 调用已关闭交易的count方法
                tags=['trades', 'closed']  # 标记为交易和已关闭相关
            ),
            total_open_trades=dict(
                title='Total Open Trades',  # 开放交易总数
                calc_func='open.count',  # 调用开放交易的count方法
                tags=['trades', 'open']  # 标记为交易和开放相关
            ),
            open_trade_pnl=dict(
                title='Open Trade PnL',  # 开放交易总盈亏
                calc_func='open.pnl.sum',  # 调用开放交易的pnl总和方法
                tags=['trades', 'open']  # 标记为交易和开放相关
            ),
            # 交易表现核心指标
            win_rate=dict(
                title='Win Rate [%]',  # 胜率（百分比）
                calc_func='closed.win_rate',  # 基于已关闭交易计算胜率
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理：转换为百分比格式
                tags=RepEval("['trades', *incl_open_tags]")  # 动态标记：交易相关，是否包含开放交易
            ),
            
            # 连胜连败指标
            winning_streak=dict(
                title='Max Win Streak',  # 最大连胜次数
                calc_func=RepEval("'winning_streak.max' if incl_open else 'closed.winning_streak.max'"),  # 根据是否包含开放交易选择计算函数
                wrap_kwargs=dict(dtype=pd.Int64Dtype()),  # 指定数据类型为可空整型
                tags=RepEval("['trades', *incl_open_tags, 'streak']")  # 动态标记：交易、连胜相关
            ),
            losing_streak=dict(
                title='Max Loss Streak',  # 最大连败次数
                calc_func=RepEval("'losing_streak.max' if incl_open else 'closed.losing_streak.max'"),  # 根据是否包含开放交易选择计算函数
                wrap_kwargs=dict(dtype=pd.Int64Dtype()),  # 指定数据类型为可空整型
                tags=RepEval("['trades', *incl_open_tags, 'streak']")  # 动态标记：交易、连败相关
            ),
            
            # 极值表现指标
            best_trade=dict(
                title='Best Trade [%]',  # 最佳交易收益率（百分比）
                calc_func=RepEval("'returns.max' if incl_open else 'closed.returns.max'"),  # 根据设置选择最大收益率
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理：转换为百分比格式
                tags=RepEval("['trades', *incl_open_tags]")  # 动态标记：交易相关
            ),
            worst_trade=dict(
                title='Worst Trade [%]',  # 最差交易收益率（百分比）
                calc_func=RepEval("'returns.min' if incl_open else 'closed.returns.min'"),  # 根据设置选择最小收益率
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理：转换为百分比格式
                tags=RepEval("['trades', *incl_open_tags]")  # 动态标记：交易相关
            ),
            
            # 平均表现指标
            avg_winning_trade=dict(
                title='Avg Winning Trade [%]',  # 平均盈利交易收益率（百分比）
                calc_func=RepEval("'winning.returns.mean' if incl_open else 'closed.winning.returns.mean'"),  # 根据设置选择盈利交易平均收益率
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理：转换为百分比格式
                tags=RepEval("['trades', *incl_open_tags, 'winning']")  # 动态标记：交易、盈利相关
            ),
            avg_losing_trade=dict(
                title='Avg Losing Trade [%]',  # 平均亏损交易收益率（百分比）
                calc_func=RepEval("'losing.returns.mean' if incl_open else 'closed.losing.returns.mean'"),  # 根据设置选择亏损交易平均收益率
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理：转换为百分比格式
                tags=RepEval("['trades', *incl_open_tags, 'losing']")  # 动态标记：交易、亏损相关
            ),
            
            # 持续时间指标
            avg_winning_trade_duration=dict(
                title='Avg Winning Trade Duration',  # 平均盈利交易持续时间
                calc_func=RepEval("'winning.avg_duration' if incl_open else 'closed.winning.avg_duration'"),  # 根据设置选择盈利交易平均持续时间
                fill_wrap_kwargs=True,  # 自动填充包装器参数
                tags=RepEval("['trades', *incl_open_tags, 'winning', 'duration']")  # 动态标记：交易、盈利、持续时间相关
            ),
            avg_losing_trade_duration=dict(
                title='Avg Losing Trade Duration',  # 平均亏损交易持续时间
                calc_func=RepEval("'losing.avg_duration' if incl_open else 'closed.losing.avg_duration'"),  # 根据设置选择亏损交易平均持续时间
                fill_wrap_kwargs=True,  # 自动填充包装器参数
                tags=RepEval("['trades', *incl_open_tags, 'losing', 'duration']")  # 动态标记：交易、亏损、持续时间相关
            ),
            
            # 高级分析指标
            profit_factor=dict(
                title='Profit Factor',  # 盈利因子
                calc_func=RepEval("'profit_factor' if incl_open else 'closed.profit_factor'"),  # 根据设置调用对应的盈利因子计算方法
                tags=RepEval("['trades', *incl_open_tags]")  # 动态标记：交易相关
            ),
            expectancy=dict(
                title='Expectancy',  # 期望收益
                calc_func=RepEval("'expectancy' if incl_open else 'closed.expectancy'"),  # 根据设置调用对应的期望收益计算方法
                tags=RepEval("['trades', *incl_open_tags]")  # 动态标记：交易相关
            ),
            sqn=dict(
                title='SQN',  # 系统质量数（System Quality Number）
                calc_func=RepEval("'sqn' if incl_open else 'closed.sqn'"),  # 根据设置调用对应的SQN计算方法
                tags=RepEval("['trades', *incl_open_tags]")  # 动态标记：交易相关
            )
        ),
        copy_kwargs=dict(copy_mode='deep')  # 配置深拷贝模式，确保配置修改不影响原始配置
    )

    @property
    def metrics(self) -> Config:
        """
        获取交易统计指标配置
        
        返回完整的交易统计指标配置对象，包含所有可计算的统计指标定义。
        该配置定义了统计报告中包含哪些指标、如何计算这些指标、
        以及指标的显示格式和分组标签。
        
        Returns:
            Config: 交易统计指标配置对象，包含以下类型的指标：
            
            - **基础时间信息指标**：分析开始时间、结束时间、时间周期
            - **交易时间相关指标**：首次交易时间、最后交易时间
            - **时间覆盖度指标**：交易时间覆盖度（含/不含重叠）
            - **交易数量统计指标**：总交易数、多头/空头交易数、开放/关闭交易数
            - **交易表现核心指标**：胜率、开放交易盈亏
            - **连胜连败指标**：最大连胜/连败次数
            - **极值表现指标**：最佳/最差交易收益率
            - **平均表现指标**：平均盈利/亏损交易收益率
            - **持续时间指标**：平均盈利/亏损交易持续时间
            - **高级分析指标**：盈利因子、期望收益、系统质量数(SQN)
        
        应用场景：
            - 自定义统计报告：基于配置生成定制化的统计报告
            - 指标扩展：在现有指标基础上添加新的计算指标
            - 配置定制：修改指标的计算方式或显示格式
            - 指标筛选：基于标签筛选特定类型的指标
        
        使用示例：
            ```python
            # 获取统计指标配置
            metrics_config = trades.metrics
            
            # 查看所有可用指标
            available_metrics = list(metrics_config.keys())
            print("可用指标:", available_metrics)
            
            # 查看特定指标的配置
            win_rate_config = metrics_config['win_rate']
            print("胜率指标配置:", win_rate_config)
            
            # 基于标签筛选指标
            streak_metrics = [
                key for key, config in metrics_config.items()
                if 'streak' in config.get('tags', [])
            ]
            print("连胜连败相关指标:", streak_metrics)
            
            # 查看高级分析指标
            advanced_metrics = [
                key for key, config in metrics_config.items()  
                if 'profit_factor' in key or 'expectancy' in key or 'sqn' in key
            ]
            print("高级分析指标:", advanced_metrics)
            ```
        
        注意事项：
            - 配置对象是只读的，修改不会影响原始配置
            - 部分指标使用RepEval进行动态计算，会根据incl_open设置调整
            - 所有百分比指标在后处理中会自动乘以100
            - 持续时间指标会自动转换为时间间隔格式
        """
        return self._metrics

    # ############# Plotting 绘图功能模块 ############# #

    def plot_pnl(self,
                 column: tp.Optional[tp.Label] = None,
                 pct_scale: bool = True,
                 marker_size_range: tp.Tuple[float, float] = (7, 14),
                 opacity_range: tp.Tuple[float, float] = (0.75, 0.9),
                 closed_profit_trace_kwargs: tp.KwargsLike = None,
                 closed_loss_trace_kwargs: tp.KwargsLike = None,
                 open_trace_kwargs: tp.KwargsLike = None,
                 hline_shape_kwargs: tp.KwargsLike = None,
                 add_trace_kwargs: tp.KwargsLike = None,
                 xref: str = 'x',
                 yref: str = 'y',
                 fig: tp.Optional[tp.BaseFigure] = None,
                 **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制交易盈亏散点图 - 专业的交易分析可视化工具
        
        该方法创建交互式的交易盈亏散点图，以时间为x轴，盈亏金额或收益率为y轴，
        通过不同颜色的散点直观展示每笔交易的表现。散点的大小和透明度根据
        收益率的绝对值动态调整，提供丰富的视觉信息。
        
        图表特征：
        - **盈利交易**：绿色散点，位于零线上方
        - **亏损交易**：红色散点，位于零线下方
        - **开放交易**：橙色散点，表示未完成的交易
        - **散点大小**：根据收益率绝对值调整，重要交易更显眼
        - **散点透明度**：根据收益率绝对值调整，增强视觉层次
        
        参数说明：
            column (tp.Optional[tp.Label], optional): 要绘制的列名，None表示绘制第一列
            pct_scale (bool, default=True): 是否使用百分比刻度
                - True: y轴显示收益率百分比（推荐）
                - False: y轴显示绝对盈亏金额
            marker_size_range (tp.Tuple[float, float], default=(7, 14)): 散点大小范围
                - 元组格式：(最小大小, 最大大小)
                - 大小根据收益率绝对值动态调整
            opacity_range (tp.Tuple[float, float], default=(0.75, 0.9)): 散点透明度范围
                - 元组格式：(最小透明度, 最大透明度)
                - 透明度根据收益率绝对值动态调整
                
            closed_profit_trace_kwargs (tp.KwargsLike, optional): 盈利交易散点的样式参数
                传递给plotly.graph_objects.Scatter的参数字典
            closed_loss_trace_kwargs (tp.KwargsLike, optional): 亏损交易散点的样式参数
                传递给plotly.graph_objects.Scatter的参数字典  
            open_trace_kwargs (tp.KwargsLike, optional): 开放交易散点的样式参数
                传递给plotly.graph_objects.Scatter的参数字典
            hline_shape_kwargs (tp.KwargsLike, optional): 零线的样式参数
                传递给plotly.graph_objects.Figure.add_shape的参数字典
            add_trace_kwargs (tp.KwargsLike, optional): 添加轨迹的参数
                传递给add_trace方法的参数字典
                
            xref (str, default='x'): X轴坐标系引用
            yref (str, default='y'): Y轴坐标系引用
            fig (tp.Optional[tp.BaseFigure], optional): 现有图表对象
                如果提供，将在现有图表上添加散点；否则创建新图表
            **layout_kwargs: 传递给图表布局的其他参数
        
        返回：
            tp.BaseFigure: Plotly图表对象，支持交互式操作和进一步定制
        
        应用场景：
            - **交易表现分析**：快速识别盈利和亏损交易的分布
            - **时间序列分析**：观察交易表现随时间的变化趋势
            - **异常值识别**：通过散点大小快速定位重要的盈亏交易
            - **策略评估**：评估交易策略在不同时间段的有效性
            - **风险管理**：识别大额亏损交易的时间聚集性

        使用示例：
            ```python
            import pandas as pd
            import numpy as np
            import vectorbt as vbt
            from datetime import datetime, timedelta
            
            # 1. 创建示例数据
            dates = pd.date_range('2023-01-01', periods=20, freq='D')
            prices = pd.Series(np.random.randn(20).cumsum() + 100, index=dates)
            
            # 创建随机交易信号
            np.random.seed(42)
            signals = np.random.choice([-1, 0, 1], size=20, p=[0.15, 0.7, 0.15])
            orders = pd.Series(signals, index=dates)
            
            # 2. 创建投资组合并获取交易记录
            pf = vbt.Portfolio.from_orders(prices, orders, fees=0.01)
            trades = pf.trades
            
            # 3. 基础盈亏散点图（推荐）
            fig1 = trades.plot_pnl(pct_scale=True, title='交易收益率散点图')
            fig1.show()
            
            # 4. 绝对金额散点图
            fig2 = trades.plot_pnl(pct_scale=False, title='交易盈亏金额散点图')
            fig2.show()
            
            # 5. 自定义样式的高级图表
            fig3 = trades.plot_pnl(
                pct_scale=True,
                marker_size_range=(10, 20),        # 增大散点尺寸
                opacity_range=(0.6, 1.0),          # 调整透明度范围
                closed_profit_trace_kwargs={       # 自定义盈利散点样式
                    'marker': {'symbol': 'triangle-up', 'line': {'width': 2}},
                    'name': '盈利交易'
                },
                closed_loss_trace_kwargs={         # 自定义亏损散点样式
                    'marker': {'symbol': 'triangle-down', 'line': {'width': 2}},
                    'name': '亏损交易'
                },
                open_trace_kwargs={                # 自定义开放交易样式
                    'marker': {'symbol': 'diamond', 'line': {'width': 2}},
                    'name': '未完成交易'
                },
                hline_shape_kwargs={               # 自定义零线样式
                    'line': {'color': 'black', 'width': 2, 'dash': 'dot'}
                },
                title='自定义样式交易分析图',
                xaxis_title='交易时间',
                yaxis_title='收益率 (%)'
            )
            fig3.show()
            
            # 6. 多资产对比分析
            multi_prices = pd.DataFrame({
                'AAPL': np.random.randn(20).cumsum() + 150,
                'GOOGL': np.random.randn(20).cumsum() + 2500
            }, index=dates)
            
            multi_pf = vbt.Portfolio.from_orders(multi_prices, orders, fees=0.01)
            
            # 分别绘制每个资产的交易表现
            fig4 = multi_pf.trades['AAPL'].plot_pnl(title='AAPL 交易表现')
            fig5 = multi_pf.trades['GOOGL'].plot_pnl(title='GOOGL 交易表现')
            
            # 7. 在现有图表上叠加
            base_fig = vbt.make_figure()
            trades.plot_pnl(fig=base_fig, pct_scale=True)
            # 可以继续添加其他图层...
            base_fig.show()
            ```
            
        进阶分析技巧：
            ```python
            # 1. 结合统计分析
            trades_stats = trades.stats()
            print(f"胜率: {trades_stats['Win Rate [%]']:.1f}%")
            print(f"盈利因子: {trades_stats['Profit Factor']:.2f}")
            
            # 绘制盈亏图，标题包含关键指标
            fig = trades.plot_pnl(
                title=f'交易分析 - 胜率: {trades_stats["Win Rate [%]"]:.1f}%, '
                      f'盈利因子: {trades_stats["Profit Factor"]:.2f}'
            )
            
            # 2. 筛选特定类型交易
            profitable_trades = trades.winning
            losing_trades = trades.losing
            
            # 分别绘制盈利和亏损交易
            fig_profit = profitable_trades.plot_pnl(title='仅盈利交易')
            fig_loss = losing_trades.plot_pnl(title='仅亏损交易')
            
            # 3. 时间段分析
            recent_trades = trades.iloc[-10:]  # 最近10笔交易
            fig_recent = recent_trades.plot_pnl(title='最近交易表现')
            
            # 4. 自动添加注释
            fig = trades.plot_pnl()
            
            # 添加最佳和最差交易的注释
            best_return = trades.returns.max()
            worst_return = trades.returns.min()
            
            fig.add_annotation(
                text=f"最佳交易: {best_return:.2%}",
                x=trades.exit_idx[trades.returns.idxmax()],
                y=best_return,
                showarrow=True
            )
            
            fig.add_annotation(
                text=f"最差交易: {worst_return:.2%}",
                x=trades.exit_idx[trades.returns.idxmin()], 
                y=worst_return,
                showarrow=True
            )
            ```
        
        可视化解读指南：
            - **散点位置**：x轴表示交易完成时间，y轴表示收益率或盈亏金额
            - **颜色含义**：绿色=盈利，红色=亏损，橙色=未完成交易
            - **大小含义**：散点越大表示收益率绝对值越大，即影响越重要
            - **透明度含义**：越不透明表示交易表现越极端
            - **零线参考**：水平零线帮助快速区分盈亏交易
            - **时间趋势**：观察散点的时间分布可发现策略表现的周期性
        
        注意事项：
            - 开放交易的盈亏基于当前参考价格，可能随市场变化而变动
            - 散点大小和透明度的计算基于收益率，适合相对比较分析
            - 建议使用pct_scale=True进行不同资产间的比较
            - 图表支持缩放、平移等交互操作，便于详细分析特定时间段
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> pf.trades.plot_pnl()
            ```

            ![](/assets/images/trades_plot_pnl.svg)
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        self_col = self.select_one(column=column, group_by=False)

        if closed_profit_trace_kwargs is None:
            closed_profit_trace_kwargs = {}
        if closed_loss_trace_kwargs is None:
            closed_loss_trace_kwargs = {}
        if open_trace_kwargs is None:
            open_trace_kwargs = {}
        if hline_shape_kwargs is None:
            hline_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        marker_size_range = tuple(marker_size_range)
        xaxis = 'xaxis' + xref[1:]
        yaxis = 'yaxis' + yref[1:]

        if fig is None:
            fig = make_figure()
        if pct_scale:
            _layout_kwargs = dict()
            _layout_kwargs[yaxis] = dict(tickformat='.2%')
            fig.update_layout(**_layout_kwargs)
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(xref, fig)

        if self_col.count() > 0:
            # Extract information
            id_ = self_col.get_field_arr('id')
            id_title = self_col.get_field_title('id')

            exit_idx = self_col.get_map_field_to_index('exit_idx')
            exit_idx_title = self_col.get_field_title('exit_idx')

            pnl = self_col.get_field_arr('pnl')
            pnl_title = self_col.get_field_title('pnl')

            returns = self_col.get_field_arr('return')
            return_title = self_col.get_field_title('return')

            status = self_col.get_field_arr('status')

            neutral_mask = pnl == 0
            profit_mask = pnl > 0
            loss_mask = pnl < 0

            marker_size = min_rel_rescale(np.abs(returns), marker_size_range)
            opacity = max_rel_rescale(np.abs(returns), opacity_range)

            open_mask = status == TradeStatus.Open
            closed_profit_mask = (~open_mask) & profit_mask
            closed_loss_mask = (~open_mask) & loss_mask
            open_mask &= ~neutral_mask

            def _plot_scatter(mask: tp.Array1d, name: tp.TraceName, color: tp.Any, kwargs: tp.Kwargs) -> None:
                if np.any(mask):
                    if self_col.get_field_setting('parent_id', 'ignore', False):
                        customdata = np.stack((
                            id_[mask],
                            pnl[mask],
                            returns[mask]
                        ), axis=1)
                        hovertemplate = f"{id_title}: %{{customdata[0]}}" \
                                        f"<br>{exit_idx_title}: %{{x}}" \
                                        f"<br>{pnl_title}: %{{customdata[1]:.6f}}" \
                                        f"<br>{return_title}: %{{customdata[2]:.2%}}"
                    else:
                        parent_id = self_col.get_field_arr('parent_id')
                        parent_id_title = self_col.get_field_title('parent_id')
                        customdata = np.stack((
                            id_[mask],
                            parent_id[mask],
                            pnl[mask],
                            returns[mask]
                        ), axis=1)
                        hovertemplate = f"{id_title}: %{{customdata[0]}}" \
                                        f"<br>{parent_id_title}: %{{customdata[1]}}" \
                                        f"<br>{exit_idx_title}: %{{x}}" \
                                        f"<br>{pnl_title}: %{{customdata[2]:.6f}}" \
                                        f"<br>{return_title}: %{{customdata[3]:.2%}}"
                    scatter = go.Scatter(
                        x=exit_idx[mask],
                        y=returns[mask] if pct_scale else pnl[mask],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            color=color,
                            size=marker_size[mask],
                            opacity=opacity[mask],
                            line=dict(
                                width=1,
                                color=adjust_lightness(color)
                            ),
                        ),
                        name=name,
                        customdata=customdata,
                        hovertemplate=hovertemplate
                    )
                    scatter.update(**kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_profit_mask,
                'Closed - Profit',
                plotting_cfg['contrast_color_schema']['green'],
                closed_profit_trace_kwargs
            )

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_loss_mask,
                'Closed - Loss',
                plotting_cfg['contrast_color_schema']['red'],
                closed_loss_trace_kwargs
            )

            # Plot Open scatter
            _plot_scatter(
                open_mask,
                'Open',
                plotting_cfg['contrast_color_schema']['orange'],
                open_trace_kwargs
            )

        # Plot zeroline
        fig.add_shape(**merge_dicts(dict(
            type='line',
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0,
            x1=x_domain[1],
            y1=0,
            line=dict(
                color="gray",
                dash="dash",
            )
        ), hline_shape_kwargs))
        return fig

    def plot(self,
             column: tp.Optional[tp.Label] = None,
             plot_zones: bool = True,
             close_trace_kwargs: tp.KwargsLike = None,
             entry_trace_kwargs: tp.KwargsLike = None,
             exit_trace_kwargs: tp.KwargsLike = None,
             exit_profit_trace_kwargs: tp.KwargsLike = None,
             exit_loss_trace_kwargs: tp.KwargsLike = None,
             active_trace_kwargs: tp.KwargsLike = None,
             profit_shape_kwargs: tp.KwargsLike = None,
             loss_shape_kwargs: tp.KwargsLike = None,
             add_trace_kwargs: tp.KwargsLike = None,
             xref: str = 'x',
             yref: str = 'y',
             fig: tp.Optional[tp.BaseFigure] = None,
             **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot orders.

        Args:
            column (str): Name of the column to plot.
            plot_zones (bool): Whether to plot zones.

                Set to False if there are many trades within one position.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Trades.close`.
            entry_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Entry" markers.
            exit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit" markers.
            exit_profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit - Profit" markers.
            exit_loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit - Loss" markers.
            active_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Active" markers.
            profit_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for profit zones.
            loss_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for loss zones.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> import pandas as pd
            >>> from datetime import datetime, timedelta
            >>> import vectorbt as vbt

            >>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.], name='Price')
            >>> price.index = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(len(price))]
            >>> orders = pd.Series([1., -0.5, -0.5, 2., -0.5, -0.5, -0.5])
            >>> pf = vbt.Portfolio.from_orders(price, orders)
            >>> pf.trades.plot()
            ```

            ![](/assets/images/trades_plot.svg)
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        self_col = self.select_one(column=column, group_by=False)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(dict(
            line=dict(
                color=plotting_cfg['color_schema']['blue']
            ),
            name='Close'
        ), close_trace_kwargs)
        if entry_trace_kwargs is None:
            entry_trace_kwargs = {}
        if exit_trace_kwargs is None:
            exit_trace_kwargs = {}
        if exit_profit_trace_kwargs is None:
            exit_profit_trace_kwargs = {}
        if exit_loss_trace_kwargs is None:
            exit_loss_trace_kwargs = {}
        if active_trace_kwargs is None:
            active_trace_kwargs = {}
        if profit_shape_kwargs is None:
            profit_shape_kwargs = {}
        if loss_shape_kwargs is None:
            loss_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # Plot close
        if self_col.close is not None:
            fig = self_col.close.vbt.plot(trace_kwargs=close_trace_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)

        if self_col.count() > 0:
            # Extract information
            id_ = self_col.get_field_arr('id')
            id_title = self_col.get_field_title('id')

            size = self_col.get_field_arr('size')
            size_title = self_col.get_field_title('size')

            entry_idx = self_col.get_map_field_to_index('entry_idx')
            entry_idx_title = self_col.get_field_title('entry_idx')

            entry_price = self_col.get_field_arr('entry_price')
            entry_price_title = self_col.get_field_title('entry_price')

            entry_fees = self_col.get_field_arr('entry_fees')
            entry_fees_title = self_col.get_field_title('entry_fees')

            exit_idx = self_col.get_map_field_to_index('exit_idx')
            exit_idx_title = self_col.get_field_title('exit_idx')

            exit_price = self_col.get_field_arr('exit_price')
            exit_price_title = self_col.get_field_title('exit_price')

            exit_fees = self_col.get_field_arr('exit_fees')
            exit_fees_title = self_col.get_field_title('exit_fees')

            direction = self_col.get_apply_mapping_arr('direction')
            direction_title = self_col.get_field_title('direction')

            pnl = self_col.get_field_arr('pnl')
            pnl_title = self_col.get_field_title('pnl')

            returns = self_col.get_field_arr('return')
            return_title = self_col.get_field_title('return')

            status = self_col.get_field_arr('status')

            duration = np.vectorize(str)(self_col.wrapper.to_timedelta(
                self_col.duration.values, to_pd=True, silence_warnings=True))

            # Plot Entry markers
            if self_col.get_field_setting('parent_id', 'ignore', False):
                entry_customdata = np.stack((
                    id_,
                    size,
                    entry_fees,
                    direction
                ), axis=1)
                entry_hovertemplate = f"{id_title}: %{{customdata[0]}}" \
                                      f"<br>{size_title}: %{{customdata[1]:.6f}}" \
                                      f"<br>{entry_idx_title}: %{{x}}" \
                                      f"<br>{entry_price_title}: %{{y}}" \
                                      f"<br>{entry_fees_title}: %{{customdata[2]:.6f}}" \
                                      f"<br>{direction_title}: %{{customdata[3]}}"
            else:
                parent_id = self_col.get_field_arr('parent_id')
                parent_id_title = self_col.get_field_title('parent_id')
                entry_customdata = np.stack((
                    id_,
                    parent_id,
                    size,
                    entry_fees,
                    direction
                ), axis=1)
                entry_hovertemplate = f"{id_title}: %{{customdata[0]}}" \
                                      f"<br>{parent_id_title}: %{{customdata[1]}}" \
                                      f"<br>{size_title}: %{{customdata[2]:.6f}}" \
                                      f"<br>{entry_idx_title}: %{{x}}" \
                                      f"<br>{entry_price_title}: %{{y}}" \
                                      f"<br>{entry_fees_title}: %{{customdata[3]:.6f}}" \
                                      f"<br>{direction_title}: %{{customdata[4]}}"
            entry_scatter = go.Scatter(
                x=entry_idx,
                y=entry_price,
                mode='markers',
                marker=dict(
                    symbol='square',
                    color=plotting_cfg['contrast_color_schema']['blue'],
                    size=7,
                    line=dict(
                        width=1,
                        color=adjust_lightness(plotting_cfg['contrast_color_schema']['blue'])
                    )
                ),
                name='Entry',
                customdata=entry_customdata,
                hovertemplate=entry_hovertemplate
            )
            entry_scatter.update(**entry_trace_kwargs)
            fig.add_trace(entry_scatter, **add_trace_kwargs)

            # Plot end markers
            def _plot_end_markers(mask: tp.Array1d, name: tp.TraceName, color: tp.Any, kwargs: tp.Kwargs) -> None:
                if np.any(mask):
                    if self_col.get_field_setting('parent_id', 'ignore', False):
                        exit_customdata = np.stack((
                            id_[mask],
                            size[mask],
                            exit_fees[mask],
                            pnl[mask],
                            returns[mask],
                            direction[mask],
                            duration[mask]
                        ), axis=1)
                        exit_hovertemplate = f"{id_title}: %{{customdata[0]}}" \
                                             f"<br>{size_title}: %{{customdata[1]:.6f}}" \
                                             f"<br>{exit_idx_title}: %{{x}}" \
                                             f"<br>{exit_price_title}: %{{y}}" \
                                             f"<br>{exit_fees_title}: %{{customdata[2]:.6f}}" \
                                             f"<br>{pnl_title}: %{{customdata[3]:.6f}}" \
                                             f"<br>{return_title}: %{{customdata[4]:.2%}}" \
                                             f"<br>{direction_title}: %{{customdata[5]}}" \
                                             f"<br>Duration: %{{customdata[6]}}"
                    else:
                        parent_id = self_col.get_field_arr('parent_id')
                        parent_id_title = self_col.get_field_title('parent_id')
                        exit_customdata = np.stack((
                            id_[mask],
                            parent_id[mask],
                            size[mask],
                            exit_fees[mask],
                            pnl[mask],
                            returns[mask],
                            direction[mask],
                            duration[mask]
                        ), axis=1)
                        exit_hovertemplate = f"{id_title}: %{{customdata[0]}}" \
                                             f"<br>{parent_id_title}: %{{customdata[1]}}" \
                                             f"<br>{size_title}: %{{customdata[2]:.6f}}" \
                                             f"<br>{exit_idx_title}: %{{x}}" \
                                             f"<br>{exit_price_title}: %{{y}}" \
                                             f"<br>{exit_fees_title}: %{{customdata[3]:.6f}}" \
                                             f"<br>{pnl_title}: %{{customdata[4]:.6f}}" \
                                             f"<br>{return_title}: %{{customdata[5]:.2%}}" \
                                             f"<br>{direction_title}: %{{customdata[6]}}" \
                                             f"<br>Duration: %{{customdata[7]}}"
                    scatter = go.Scatter(
                        x=exit_idx[mask],
                        y=exit_price[mask],
                        mode='markers',
                        marker=dict(
                            symbol='square',
                            color=color,
                            size=7,
                            line=dict(
                                width=1,
                                color=adjust_lightness(color)
                            )
                        ),
                        name=name,
                        customdata=exit_customdata,
                        hovertemplate=exit_hovertemplate
                    )
                    scatter.update(**kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Exit markers
            _plot_end_markers(
                (status == TradeStatus.Closed) & (pnl == 0.),
                'Exit',
                plotting_cfg['contrast_color_schema']['gray'],
                exit_trace_kwargs
            )

            # Plot Exit - Profit markers
            _plot_end_markers(
                (status == TradeStatus.Closed) & (pnl > 0.),
                'Exit - Profit',
                plotting_cfg['contrast_color_schema']['green'],
                exit_profit_trace_kwargs
            )

            # Plot Exit - Loss markers
            _plot_end_markers(
                (status == TradeStatus.Closed) & (pnl < 0.),
                'Exit - Loss',
                plotting_cfg['contrast_color_schema']['red'],
                exit_loss_trace_kwargs
            )

            # Plot Active markers
            _plot_end_markers(
                status == TradeStatus.Open,
                'Active',
                plotting_cfg['contrast_color_schema']['orange'],
                active_trace_kwargs
            )

            if plot_zones:
                profit_mask = pnl > 0.
                if np.any(profit_mask):
                    # Plot profit zones
                    for i in np.flatnonzero(profit_mask):
                        fig.add_shape(**merge_dicts(dict(
                            type="rect",
                            xref=xref,
                            yref=yref,
                            x0=entry_idx[i],
                            y0=entry_price[i],
                            x1=exit_idx[i],
                            y1=exit_price[i],
                            fillcolor='green',
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        ), profit_shape_kwargs))

                loss_mask = pnl < 0.
                if np.any(loss_mask):
                    # Plot loss zones
                    for i in np.flatnonzero(loss_mask):
                        fig.add_shape(**merge_dicts(dict(
                            type="rect",
                            xref=xref,
                            yref=yref,
                            x0=entry_idx[i],
                            y0=entry_price[i],
                            x1=exit_idx[i],
                            y1=exit_price[i],
                            fillcolor='red',
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        ), loss_shape_kwargs))

        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Trades.plots`.

        Merges `vectorbt.generic.ranges.Ranges.plots_defaults` and
        `trades.plots` from `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        trades_plots_cfg = settings['trades']['plots']

        # 合并父类和当前类的绘图默认配置
        return merge_dicts(
            Ranges.plots_defaults.__get__(self),  # 继承Ranges类的绘图默认配置
            trades_plots_cfg                       # 应用交易特定的绘图配置
        )

    # 子图配置 - 定义Trades类支持的标准子图类型和绘制参数
    _subplots: tp.ClassVar[Config] = Config(
        dict(
            # 交易时间线图配置
            plot=dict(
                title="Trades",  # 子图标题
                yaxis_kwargs=dict(title="Price"),  # Y轴配置：价格轴
                check_is_not_grouped=True,  # 检查数据是否未分组，确保绘图正确性
                plot_func='plot',  # 绘图函数名称，对应plot方法
                tags='trades'  # 标签，标识为交易相关图表
            ),
            # 交易盈亏散点图配置
            plot_pnl=dict(
                title="Trade PnL",  # 子图标题：交易盈亏
                yaxis_kwargs=dict(title="Trade PnL"),  # Y轴配置：盈亏轴
                check_is_not_grouped=True,  # 检查数据是否未分组，确保绘图正确性
                plot_func='plot_pnl',  # 绘图函数名称，对应plot_pnl方法
                tags='trades'  # 标签，标识为交易相关图表
            )
        ),
        copy_kwargs=dict(copy_mode='deep')  # 深拷贝配置，确保修改不影响原始配置
    )

    @property
    def subplots(self) -> Config:
        """
        获取子图配置对象
        
        返回Trades类支持的所有子图类型的配置信息。这些配置定义了
        如何创建标准化的交易分析图表，包括图表标题、轴标签、
        绘图函数等设置。
        
        Returns:
            Config: 子图配置对象，包含以下子图类型：
            
            - **plot**: 交易时间线图
              - 显示交易的入场和退出点
              - Y轴为价格，便于观察交易时机
              - 包含交易区域的颜色填充
              
            - **plot_pnl**: 交易盈亏散点图  
              - 显示每笔交易的盈亏表现
              - Y轴为盈亏金额或收益率
              - 通过颜色区分盈利和亏损交易
        
        应用场景：
            - 创建标准化的交易分析报告
            - 快速生成多个相关的交易图表
            - 自定义图表布局和样式
            - 批量生成不同资产的交易分析图
        
        使用示例：
            ```python
            # 获取子图配置
            subplot_config = trades.subplots
            
            # 查看可用的子图类型
            available_plots = list(subplot_config.keys())
            print("可用图表类型:", available_plots)  # ['plot', 'plot_pnl']
            
            # 查看特定子图配置
            pnl_config = subplot_config['plot_pnl']
            print("盈亏图配置:", pnl_config)
            
            # 使用plots方法创建标准子图
            fig = trades.plots()  # 创建包含所有子图的组合图表
            fig.show()
            
            # 创建特定子图
            pnl_fig = trades.plots(subplots=['plot_pnl'])
            timeline_fig = trades.plots(subplots=['plot'])
            
            # 自定义子图设置
            custom_fig = trades.plots(
                subplots=['plot_pnl'],
                subplot_kwargs=dict(
                    plot_pnl=dict(
                        pct_scale=True,
                        marker_size_range=(10, 20)
                    )
                )
            )
            ```
        
        注意事项：
            - 所有子图都要求数据未分组（check_is_not_grouped=True）
            - 子图配置可以通过plots方法的参数进行覆盖
            - 每个子图都有对应的绘图方法（plot_func指定）
        """
        return self._subplots


# 文档系统集成 - 将类的配置文档集成到vectorbt的文档系统中
Trades.override_field_config_doc(__pdoc__)  # 覆盖字段配置文档，将配置信息集成到API文档
Trades.override_metrics_doc(__pdoc__)        # 覆盖统计指标文档，将指标信息集成到API文档  
Trades.override_subplots_doc(__pdoc__)       # 覆盖子图配置文档，将绘图信息集成到API文档

# ############# EntryTrades 入场交易分析类 ############# #

# 入场交易字段配置 - 专门针对入场交易记录的字段设置
entry_trades_field_config = Config(
    dict(
        settings={
            'id': dict(
                title='Entry Trade Id'  # 入场交易唯一标识符，区分不同的开仓交易
            ),
            'idx': dict(
                name='entry_idx'  # 将基类的idx字段重映射为entry_idx，强调入场时间
            )
        }
    ),
    readonly=True,    # 配置为只读，防止意外修改
    as_attrs=False    # 不作为属性访问，使用字典方式访问
)
"""_"""

__pdoc__['entry_trades_field_config'] = f"""入场交易字段配置文档。

```json
{entry_trades_field_config.to_doc()}
```
"""

# 入场交易类的类型变量，用于类型提示中的泛型约束
EntryTradesT = tp.TypeVar("EntryTradesT", bound="EntryTrades")


@override_field_config(entry_trades_field_config)  # 应用入场交易专用字段配置
class EntryTrades(Trades):
    """
    入场交易分析类 - 专门分析开仓交易的表现
    
    EntryTrades类继承自Trades基类，专门用于处理和分析入场（开仓）交易。
    该类以每个开仓订单为基础创建交易记录，对应的退出信息是所有相关
    平仓订单的加权平均结果。
    
    设计理念：
    - **入场视角分析**：从开仓的角度分析每笔交易的表现
    - **加权平均退出**：多次退出时，退出价格和手续费采用加权平均
    - **入场策略评估**：专门用于评估入场时机和入场策略的有效性
    
    数据特点：
    - 每个开仓订单对应一条入场交易记录
    - 如果一笔开仓分多次退出，退出信息是加权平均值
    - 交易ID以入场订单为准，便于追踪入场决策的效果
    
    适用场景：
    - 分析入场时机的准确性
    - 评估不同入场信号的有效性
    - 研究入场价格对最终收益的影响
    - 优化开仓策略和时机选择
    
    与其他交易类型的区别：
    - EntryTrades: 以开仓为基准，分析入场策略效果
    - ExitTrades: 以平仓为基准，分析出场策略效果
    - Positions: 聚合完整持仓，分析整体投资表现
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建价格数据
    prices = pd.Series([100, 105, 98, 95, 102, 108, 103])
    
    # 创建订单：在第0、2、4天开仓，在第1、3、5天平仓
    orders = pd.Series([1, -0.5, 0.5, -0.5, 0.5, -1, 0])
    
    # 创建投资组合并获取入场交易
    pf = vbt.Portfolio.from_orders(prices, orders)
    entry_trades = pf.entry_trades
    
    # 分析入场交易表现
    print(f"入场交易数量: {entry_trades.count()}")
    print(f"入场平均收益率: {entry_trades.returns.mean():.2%}")
    print(f"最佳入场交易收益: {entry_trades.returns.max():.2%}")
    
    # 分析入场时机
    entry_prices = entry_trades.entry_price
    print(f"平均入场价格: {entry_prices.mean():.2f}")
    
    # 获取详细入场交易记录
    entry_records = entry_trades.records_readable
    print("入场交易详情:")
    print(entry_records)
    ```
    
    注意事项：
    - 入场交易记录包含开放和封闭的交易
    - 如需仅分析已完成交易，使用.closed属性
    - 退出信息是多次退出的加权平均，可能与实际单次退出不同
    """

    @classmethod
    def from_orders(cls: tp.Type[EntryTradesT],
                    orders: Orders,
                    close: tp.Optional[tp.ArrayLike] = None,
                    attach_close: bool = True,
                    **kwargs) -> EntryTradesT:
        """
        从订单记录构建入场交易对象
        
        该类方法是创建EntryTrades对象的主要方式，从Orders对象中提取
        开仓订单信息，并计算对应的退出信息，生成以入场为视角的交易记录。
        
        参数：
            orders (Orders): 包含所有订单信息的Orders对象
            close (tp.Optional[tp.ArrayLike], optional): 参考收盘价序列，
                用于计算开放交易的未实现收益。如果为None，则使用orders.close
            attach_close (bool, default=True): 是否将参考价格附加到结果对象中
            **kwargs: 传递给EntryTrades构造函数的其他参数
        
        返回：
            EntryTradesT: 构建好的入场交易对象
        
        处理流程：
            1. 从Orders对象中提取订单记录数组
            2. 调用底层Numba编译函数计算入场交易记录
            3. 每个开仓订单生成一条入场交易记录
            4. 对应的退出信息通过加权平均计算得出
            5. 创建并返回EntryTrades对象
        
        算法逻辑：
            - 遍历所有订单，识别开仓订单
            - 对每个开仓订单，寻找对应的所有平仓订单
            - 计算加权平均的退出价格、手续费和时间
            - 生成完整的入场交易记录
        
        使用示例：
            ```python
            # 从现有的投资组合获取订单
            orders = portfolio.orders
            
            # 构建入场交易分析
            entry_trades = EntryTrades.from_orders(orders)
            
            # 使用自定义参考价格
            custom_close = pd.Series([...])  # 自定义收盘价序列
            entry_trades = EntryTrades.from_orders(
                orders, 
                close=custom_close,
                attach_close=True
            )
            
            # 不附加参考价格（节省内存）
            entry_trades = EntryTrades.from_orders(
                orders,
                attach_close=False
            )
            ```
        
        注意事项：
            - 该方法会调用高性能的Numba编译函数进行计算
            - 处理大量订单时性能优异
            - 退出信息的计算采用加权平均方式
        """
        # 如果没有提供参考价格，使用订单对象中的收盘价
        if close is None:
            close = orders.close
        
        # 调用Numba编译的函数计算入场交易记录
        # 该函数会分析所有订单，识别开仓订单并计算对应的退出信息
        trade_records_arr = nb.get_entry_trades_nb(
            orders.values,                    # 订单记录数组
            to_2d_array(close),              # 参考价格的2D数组
            orders.col_mapper.col_map        # 列映射信息
        )
        
        # 创建并返回EntryTrades对象
        return cls(
            orders.wrapper,                                      # 数组包装器
            trade_records_arr,                                  # 入场交易记录数组
            close=close if attach_close else None,              # 参考价格（可选）
            **kwargs                                            # 其他参数
        )


# ############# ExitTrades 退出交易分析类 ############# #

# 退出交易字段配置 - 专门针对退出交易记录的字段设置
exit_trades_field_config = Config(
    dict(
        settings={
            'id': dict(
                title='Exit Trade Id'  # 退出交易唯一标识符，区分不同的平仓交易
            )
        }
    ),
    readonly=True,    # 配置为只读，防止意外修改
    as_attrs=False    # 不作为属性访问，使用字典方式访问
)
"""_"""

__pdoc__['exit_trades_field_config'] = f"""退出交易字段配置文档。

```json
{exit_trades_field_config.to_doc()}
```
"""

# 退出交易类的类型变量，用于类型提示中的泛型约束
ExitTradesT = tp.TypeVar("ExitTradesT", bound="ExitTrades")


@override_field_config(exit_trades_field_config)  # 应用退出交易专用字段配置
class ExitTrades(Trades):
    """
    退出交易分析类 - 专门分析平仓交易的表现
    
    ExitTrades类继承自Trades基类，专门用于处理和分析退出（平仓）交易。
    该类以每个平仓订单为基础创建交易记录，对应的入场信息是相关开仓
    订单的分摊结果。
    
    设计理念：
    - **退出视角分析**：从平仓的角度分析每笔交易的表现
    - **入场信息分摊**：多次入场的成本按比例分摊到每次退出
    - **退出策略评估**：专门用于评估出场时机和止盈止损策略的有效性
    
    数据特点：
    - 每个平仓订单对应一条退出交易记录
    - 如果持仓来源于多次开仓，入场成本按比例分摊
    - 交易ID以平仓订单为准，便于追踪退出决策的效果
    
    适用场景：
    - 分析出场时机的准确性
    - 评估止盈止损策略的有效性
    - 研究退出价格对收益的影响
    - 优化平仓策略和时机选择
    - 分析不同退出信号的效果
    
    与其他交易类型的区别：
    - EntryTrades: 以开仓为基准，分析入场策略效果
    - ExitTrades: 以平仓为基准，分析出场策略效果
    - Positions: 聚合完整持仓，分析整体投资表现
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建价格数据
    prices = pd.Series([100, 105, 98, 95, 102, 108, 103])
    
    # 创建订单：多次入场，分批退出
    orders = pd.Series([0.5, 0.5, 0, -0.3, 0, -0.4, -0.3])
    
    # 创建投资组合并获取退出交易
    pf = vbt.Portfolio.from_orders(prices, orders)
    exit_trades = pf.exit_trades
    
    # 分析退出交易表现
    print(f"退出交易数量: {exit_trades.count()}")
    print(f"退出平均收益率: {exit_trades.returns.mean():.2%}")
    print(f"最佳退出交易收益: {exit_trades.returns.max():.2%}")
    
    # 分析退出时机
    exit_prices = exit_trades.exit_price
    print(f"平均退出价格: {exit_prices.mean():.2f}")
    
    # 分析止盈止损效果
    profitable_exits = exit_trades.winning
    loss_exits = exit_trades.losing
    print(f"盈利退出比例: {profitable_exits.count() / exit_trades.count():.2%}")
    print(f"平均盈利退出收益: {profitable_exits.returns.mean():.2%}")
    print(f"平均亏损退出收益: {loss_exits.returns.mean():.2%}")
    
    # 获取详细退出交易记录
    exit_records = exit_trades.records_readable
    print("退出交易详情:")
    print(exit_records)
    ```
    
    注意事项：
    - 退出交易记录包含所有平仓操作
    - 入场信息是按持仓比例分摊的平均成本
    - 适合分析各种退出策略的有效性
    """

    @classmethod
    def from_orders(cls: tp.Type[ExitTradesT],
                    orders: Orders,
                    close: tp.Optional[tp.ArrayLike] = None,
                    attach_close: bool = True,
                    **kwargs) -> ExitTradesT:
        """
        从订单记录构建退出交易对象
        
        该类方法是创建ExitTrades对象的主要方式，从Orders对象中提取
        平仓订单信息，并计算对应的入场成本分摊，生成以退出为视角的交易记录。
        
        参数：
            orders (Orders): 包含所有订单信息的Orders对象
            close (tp.Optional[tp.ArrayLike], optional): 参考收盘价序列，
                用于计算开放交易的未实现收益。如果为None，则使用orders.close
            attach_close (bool, default=True): 是否将参考价格附加到结果对象中
            **kwargs: 传递给ExitTrades构造函数的其他参数
        
        返回：
            ExitTradesT: 构建好的退出交易对象
        
        处理流程：
            1. 从Orders对象中提取订单记录数组
            2. 调用底层Numba编译函数计算退出交易记录
            3. 每个平仓订单生成一条退出交易记录
            4. 对应的入场成本通过比例分摊计算得出
            5. 创建并返回ExitTrades对象
        
        算法逻辑：
            - 遍历所有订单，识别平仓订单
            - 对每个平仓订单，寻找对应的开仓订单
            - 按持仓比例分摊入场成本和手续费
            - 生成完整的退出交易记录
        
        使用示例：
            ```python
            # 从现有的投资组合获取订单
            orders = portfolio.orders
            
            # 构建退出交易分析
            exit_trades = ExitTrades.from_orders(orders)
            
            # 分析止损策略效果
            stop_loss_trades = exit_trades.losing
            print(f"止损交易数量: {stop_loss_trades.count()}")
            print(f"平均止损收益率: {stop_loss_trades.returns.mean():.2%}")
            
            # 分析止盈策略效果
            take_profit_trades = exit_trades.winning
            print(f"止盈交易数量: {take_profit_trades.count()}")
            print(f"平均止盈收益率: {take_profit_trades.returns.mean():.2%}")
            
            # 使用自定义参考价格
            custom_close = pd.Series([...])
            exit_trades = ExitTrades.from_orders(
                orders,
                close=custom_close,
                attach_close=True
            )
            ```
        
        注意事项：
            - 该方法会调用高性能的Numba编译函数进行计算
            - 入场成本分摊采用先进先出(FIFO)原则
            - 适合分析各种退出策略的表现
        """
        # 如果没有提供参考价格，使用订单对象中的收盘价
        if close is None:
            close = orders.close
        
        # 调用Numba编译的函数计算退出交易记录
        # 该函数会分析所有订单，识别平仓订单并计算对应的入场成本分摊
        trade_records_arr = nb.get_exit_trades_nb(
            orders.values,                    # 订单记录数组
            to_2d_array(close),              # 参考价格的2D数组
            orders.col_mapper.col_map        # 列映射信息
        )
        
        # 创建并返回ExitTrades对象
        return cls(
            orders.wrapper,                                      # 数组包装器
            trade_records_arr,                                  # 退出交易记录数组
            close=close if attach_close else None,              # 参考价格（可选）
            **kwargs                                            # 其他参数
        )


# ############# Positions 持仓分析类 ############# #

# 持仓字段配置 - 专门针对持仓记录的字段设置
positions_field_config = Config(
    dict(
        settings={
            'id': dict(
                title='Position Id'  # 持仓唯一标识符，区分不同的持仓周期
            ),
            'parent_id': dict(
                title='Parent Id',  # 父级ID，在持仓层面不使用
                ignore=True         # 忽略该字段，持仓是最高层级的聚合
            )
        }
    ),
    readonly=True,    # 配置为只读，防止意外修改
    as_attrs=False    # 不作为属性访问，使用字典方式访问
)
"""_"""

__pdoc__['positions_field_config'] = f"""持仓字段配置文档。

```json
{positions_field_config.to_doc()}
```
"""

# 持仓类的类型变量，用于类型提示中的泛型约束
PositionsT = tp.TypeVar("PositionsT", bound="Positions")


@override_field_config(positions_field_config)  # 应用持仓专用字段配置
class Positions(Trades):
    """
    持仓分析类 - 分析完整持仓周期的投资表现
    
    Positions类继承自Trades基类，专门用于处理和分析持仓记录。
    持仓是交易记录的最高层级聚合，将连续的入场交易或退出交易
    聚合为完整的持仓周期，反映整个投资过程的表现。
    
    设计理念：
    - **完整投资周期**：从建仓到清仓的完整投资过程分析
    - **交易聚合**：将多次交易聚合为持仓，简化分析复杂度
    - **整体表现评估**：关注整个持仓周期的综合表现
    - **投资决策分析**：评估投资决策的整体效果
    
    数据特点：
    - 一个持仓可能包含多次加仓和减仓操作
    - 持仓记录反映从开始建仓到完全清仓的整个过程
    - 入场和出场信息是所有相关交易的综合结果
    - 持仓ID是唯一的，不存在parent_id概念
    
    适用场景：
    - 分析长期投资策略的表现
    - 评估完整投资决策的效果
    - 研究持仓周期对收益的影响
    - 分析资产配置和仓位管理策略
    - 评估投资组合的整体表现
    
    与其他交易类型的区别：
    - EntryTrades: 分析开仓策略，关注入场时机
    - ExitTrades: 分析平仓策略，关注出场时机  
    - Positions: 分析完整持仓，关注整体投资表现
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建价格数据
    prices = pd.Series([100, 105, 98, 95, 102, 108, 103, 99, 104, 110])
    
    # 创建复杂的交易序列：分批建仓、持有、分批清仓
    orders = pd.Series([0.3, 0.2, 0, 0, 0.5, 0, -0.4, 0, -0.3, -0.3])
    
    # 创建投资组合并获取持仓分析
    pf = vbt.Portfolio.from_orders(prices, orders)
    positions = pf.positions
    
    # 分析持仓表现
    print(f"持仓数量: {positions.count()}")
    print(f"平均持仓收益率: {positions.returns.mean():.2%}")
    print(f"最佳持仓收益: {positions.returns.max():.2%}")
    print(f"最差持仓收益: {positions.returns.min():.2%}")
    
    # 分析持仓周期
    durations = positions.duration
    print(f"平均持仓周期: {durations.mean()}")
    print(f"最长持仓周期: {durations.max()}")
    
    # 分析持仓规模和成本
    position_sizes = positions.size
    avg_entry_prices = positions.entry_price
    print(f"平均持仓规模: {position_sizes.mean():.2f}")
    print(f"平均入场成本: {avg_entry_prices.mean():.2f}")
    
    # 持仓盈亏分析
    profitable_positions = positions.winning
    losing_positions = positions.losing
    print(f"盈利持仓比例: {profitable_positions.count() / positions.count():.2%}")
    print(f"平均盈利持仓收益: {profitable_positions.returns.mean():.2%}")
    print(f"平均亏损持仓收益: {losing_positions.returns.mean():.2%}")
    
    # 获取详细持仓记录
    position_records = positions.records_readable
    print("持仓详细记录:")
    print(position_records)
    ```
    
    注意事项：
    - 持仓分析提供最高层次的投资表现视角
    - 持仓记录包含开放和封闭的持仓
    - 适合分析长期投资策略和资产配置效果
    - 持仓聚合可能掩盖交易层面的细节
    """

    @property
    def field_config(self) -> Config:
        """
        获取持仓字段配置对象
        
        返回持仓记录类的字段配置，包含各字段的名称、类型、映射关系等信息。
        持仓配置忽略了parent_id字段，因为持仓是最高层级的聚合。
        
        Returns:
            Config: 持仓字段配置对象
        """
        return self._field_config

    @classmethod
    def from_trades(cls: tp.Type[PositionsT],
                    trades: Trades,
                    close: tp.Optional[tp.ArrayLike] = None,
                    attach_close: bool = True,
                    **kwargs) -> PositionsT:
        """
        从交易记录构建持仓对象
        
        该类方法是创建Positions对象的主要方式，从任意类型的Trades对象
        （EntryTrades或ExitTrades）中聚合连续的交易，生成完整的持仓记录。
        
        参数：
            trades (Trades): 输入的交易记录对象，可以是EntryTrades或ExitTrades
            close (tp.Optional[tp.ArrayLike], optional): 参考收盘价序列，
                用于计算开放持仓的未实现收益。如果为None，则使用trades.close
            attach_close (bool, default=True): 是否将参考价格附加到结果对象中
            **kwargs: 传递给Positions构造函数的其他参数
        
        返回：
            PositionsT: 构建好的持仓对象
        
        处理流程：
            1. 从Trades对象中提取交易记录数组
            2. 调用底层Numba编译函数聚合连续交易为持仓
            3. 计算每个持仓的综合入场和出场信息
            4. 创建并返回Positions对象
        
        聚合逻辑：
            - 识别同一资产的连续交易序列
            - 将时间上连续的同向操作聚合为一个持仓
            - 计算聚合后的平均入场价格、总手续费
            - 确定持仓的开始和结束时间
        
        使用示例：
            ```python
            # 从入场交易构建持仓
            entry_trades = portfolio.entry_trades
            positions = Positions.from_trades(entry_trades)
            
            # 从退出交易构建持仓
            exit_trades = portfolio.exit_trades
            positions = Positions.from_trades(exit_trades)
            
            # 使用自定义参考价格
            custom_close = pd.Series([...])
            positions = Positions.from_trades(
                trades,
                close=custom_close,
                attach_close=True
            )
            
            # 不附加参考价格（节省内存）
            positions = Positions.from_trades(
                trades,
                attach_close=False
            )
            
            # 分析持仓与交易的关系
            print(f"交易数量: {trades.count()}")
            print(f"持仓数量: {positions.count()}")
            print(f"每个持仓平均交易数: {trades.count() / positions.count():.1f}")
            ```
        
        注意事项：
            - 该方法会调用高性能的Numba编译函数进行聚合
            - 聚合过程中会合并连续的同向交易
            - 结果持仓数量通常少于输入交易数量
            - 持仓聚合可能会丢失交易层面的时间细节
        """
        # 如果没有提供参考价格，使用交易对象中的参考价格
        if close is None:
            close = trades.close
        
        # 调用Numba编译的函数将交易记录聚合为持仓记录
        # 该函数会分析交易序列，将连续的交易聚合为持仓周期
        position_records_arr = nb.get_positions_nb(
            trades.values,                    # 输入的交易记录数组
            trades.col_mapper.col_map        # 列映射信息
        )
        
        # 创建并返回Positions对象
        return cls(
            trades.wrapper,                                      # 复用输入交易的数组包装器
            position_records_arr,                               # 聚合后的持仓记录数组
            close=close if attach_close else None,              # 参考价格（可选）
            **kwargs                                            # 其他参数
        )
