# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT PORTFOLIO ENUMS MODULE: 投资组合枚举类型和数据结构核心定义模块
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中最核心的枚举类型和数据结构定义模块，为整个投资组合管理
系统提供了完整的类型系统和数据格式规范。该模块是vectorbt框架"类型安全+高性能"设计
理念的重要体现，通过精心设计的枚举类型和数据结构确保了系统的可靠性和扩展性。

核心设计理念：

1. **完整的类型系统**：
   定义了投资组合管理中所有核心概念的类型，包括订单、交易、持仓、方向、状态等，
   为整个系统提供了统一、严谨的类型基础。

2. **高性能数据结构**：
   使用NumPy数据类型定义结构化记录格式，确保数据存储和访问的极致性能，
   同时保持内存使用的高效性。

3. **上下文驱动设计**：
   通过丰富的上下文类（Context classes）为不同执行阶段提供准确的状态信息，
   支持复杂的事件驱动交易逻辑。

4. **可扩展的枚举系统**：
   通过NamedTuple实现的伪枚举系统，既保持了类型安全性，又提供了良好的
   可读性和扩展性。

主要功能模块：

【异常定义模块】
- RejectedOrderError: 订单拒绝异常类，用于处理订单被系统拒绝的情况

【核心枚举类型模块】
- InitCashMode: 初始资金设置模式（自动/自动对齐）
- CallSeqType: 订单调用序列类型（默认/反向/随机/自动）
- AccumulationMode: 仓位累积模式（禁用/双向/仅增加/仅减少）
- ConflictMode: 信号冲突处理模式
- DirectionConflictMode: 方向冲突处理模式
- OppositeEntryMode: 反向开仓模式
- StopEntryPrice/StopExitPrice: 止损止盈价格参考类型
- StopExitMode/StopUpdateMode: 止损退出和更新模式
- SizeType: 订单大小类型（数量/价值/百分比/目标类型）
- Direction: 交易方向（仅多头/仅空头/双向）
- OrderStatus/OrderSide: 订单状态和方向
- OrderStatusInfo: 订单状态详细信息
- TradeDirection/TradeStatus/TradesType: 交易相关类型

【状态管理模块】
- ProcessOrderState: 订单处理状态
- ExecuteOrderState: 订单执行状态

【上下文系统模块】
- SimulationContext: 模拟全局上下文
- GroupContext: 资产组上下文
- RowContext: 时间行上下文  
- SegmentContext: 分段上下文
- OrderContext: 订单上下文
- PostOrderContext: 订单后处理上下文
- FlexOrderContext: 灵活订单上下文
- AdjustSLContext/AdjustTPContext: 止损止盈调整上下文
- SignalContext: 信号生成上下文

【订单和结果模块】
- Order: 订单定义，包含完整的订单参数
- NoOrder: 空订单标识
- OrderResult: 订单执行结果

【记录数据类型模块】
- order_dt: 订单记录数据类型
- trade_dt: 交易记录数据类型
- log_dt: 日志记录数据类型

应用场景：
- **类型安全检查**：为编译器和IDE提供准确的类型信息，减少运行时错误
- **数据结构定义**：为高性能的NumPy数组操作提供结构化数据格式
- **配置参数管理**：提供标准化的配置选项，确保参数使用的一致性
- **上下文信息传递**：在复杂的模拟过程中传递准确的状态信息
- **扩展和定制**：为用户自定义功能提供标准的接口和数据格式

技术特点：
- **零成本抽象**：枚举类型在运行时表现为整数，无额外性能开销
- **内存对齐优化**：结构化数据类型使用align=True优化内存访问
- **文档自动生成**：集成文档系统，自动生成API文档
- **类型提示支持**：完整的类型注解，支持静态类型检查

与vectorbt生态系统的关系：
- **核心基础设施**：为整个portfolio模块提供类型和数据结构基础
- **性能优化支持**：为Numba编译函数提供兼容的数据类型
- **扩展性保障**：为用户自定义策略和分析提供标准接口
- **文档系统集成**：与vectorbt的文档生成系统深度集成

该模块是vectorbt框架的基石之一，为量化交易系统的可靠运行提供了坚实的类型
和数据结构基础。其设计充分考虑了性能、安全性、可扩展性和易用性的平衡。

使用示例：
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import *

# 1. 使用枚举类型配置策略
pf = vbt.Portfolio.from_orders(
    close=price_data,
    size=order_sizes,
    direction=Direction.LongOnly,     # 仅允许多头交易
    size_type=SizeType.TargetPercent, # 按目标百分比下单
    fees=0.001
)

# 2. 检查订单状态
orders = pf.orders
filled_orders = orders.res_status == OrderStatus.Filled
rejected_orders = orders.res_status == OrderStatus.Rejected

# 3. 分析订单拒绝原因
rejection_info = orders[rejected_orders].res_status_info
cash_shortage = rejection_info == OrderStatusInfo.NoCashLong

# 4. 使用上下文信息（在自定义函数中）
@njit
def custom_order_func(ctx, *args):
    # 访问当前市场状态
    current_price = ctx.close[ctx.i, ctx.col]
    current_cash = ctx.cash_now
    current_position = ctx.position_now
    
    # 基于上下文信息做出交易决策
    if current_cash > current_price * 100:
        return Order(size=100, price=current_price)
    else:
        return NoOrder
        
# 5. 创建自定义记录分析
import numpy as np

# 访问订单记录
order_records = pf.orders.records
print(f"订单记录类型: {order_records.dtype}")
print(f"总订单数: {len(order_records)}")

# 按订单方向统计
buy_orders = order_records[order_records['side'] == OrderSide.Buy]
sell_orders = order_records[order_records['side'] == OrderSide.Sell]
```

注意事项：
- 所有枚举值在运行时都是整数，便于NumPy数组处理和Numba编译
- 上下文类提供的信息在不同执行阶段会有所不同，需要根据具体场景使用
- 记录数据类型专为高性能计算优化，直接使用NumPy操作即可获得最佳性能
- 扩展枚举类型时需要保持向后兼容性，避免破坏现有代码
================================================================================

命名元组和枚举类型定义

为 `vectorbt.portfolio` 模块定义枚举和其他数据结构模式。
"""

# ================== 导入依赖模块 ==================

import numpy as np  # 导入NumPy库，用于高性能数值计算和数据类型定义

# 从vectorbt类型系统导入类型定义
from vectorbt import _typing as tp

# 从文档工具模块导入文档生成函数
from vectorbt.utils.docs import to_doc

# 定义模块公开接口，列出所有可供外部使用的类和常量
__all__ = [
    # 异常类
    'RejectedOrderError',               # 订单拒绝异常
    
    # 核心配置枚举
    'InitCashMode',                    # 初始资金模式
    'CallSeqType',                     # 调用序列类型
    'AccumulationMode',                # 仓位累积模式
    'ConflictMode',                    # 信号冲突模式
    'DirectionConflictMode',           # 方向冲突模式
    'OppositeEntryMode',               # 反向开仓模式
    'StopEntryPrice',                  # 止损入场价格类型
    'StopExitPrice',                   # 止损出场价格类型
    'StopExitMode',                    # 止损退出模式
    'StopUpdateMode',                  # 止损更新模式
    
    # 订单和交易相关枚举
    'SizeType',                        # 订单大小类型
    'Direction',                       # 交易方向
    'OrderStatus',                     # 订单状态
    'OrderSide',                       # 订单方向（买卖）
    'OrderStatusInfo',                 # 订单状态详细信息
    'TradeDirection',                  # 交易方向
    'TradeStatus',                     # 交易状态
    'TradesType',                      # 交易类型
    
    # 状态和上下文类
    'ProcessOrderState',               # 订单处理状态
    'ExecuteOrderState',               # 订单执行状态
    'SimulationContext',               # 模拟全局上下文
    'GroupContext',                    # 资产组上下文
    'RowContext',                      # 时间行上下文
    'SegmentContext',                  # 分段上下文
    'OrderContext',                    # 订单上下文
    'PostOrderContext',                # 订单后处理上下文
    'FlexOrderContext',                # 灵活订单上下文
    
    # 订单相关类
    'Order',                           # 订单定义
    'NoOrder',                         # 空订单
    'OrderResult',                     # 订单结果
    
    # 调整上下文类
    'AdjustSLContext',                 # 止损调整上下文
    'AdjustTPContext',                 # 止盈调整上下文
    'SignalContext',                   # 信号上下文
    
    # 记录数据类型
    'order_dt',                        # 订单记录数据类型
    'trade_dt',                        # 交易记录数据类型
    'log_dt'                          # 日志记录数据类型
]

# 初始化文档字典，用于API文档的自动生成
__pdoc__ = {}


# ############# 异常类定义 ############# #


class RejectedOrderError(Exception):
    """
    订单拒绝异常类
    
    当订单因各种原因被投资组合模拟系统拒绝时抛出的异常。这是一个自定义异常类，
    继承自Python的标准Exception类，专门用于处理订单执行过程中的拒绝情况。
    
    常见的订单拒绝原因包括：
    - 资金不足：没有足够的现金来执行买入订单
    - 持仓不足：没有足够的持仓来执行卖出订单
    - 订单大小不符合要求：订单大小超出限制或低于最小值
    - 价格异常：订单价格为NaN或其他无效值
    - 随机拒绝：模拟真实交易中的随机拒绝事件
    
    使用示例：
    ```python
    import vectorbt as vbt
    from vectorbt.portfolio.enums import RejectedOrderError
    
    try:
        # 创建一个可能导致订单拒绝的投资组合
        pf = vbt.Portfolio.from_orders(
            close=price_data,
            size=large_order_sizes,  # 可能超出资金限制的大订单
            raise_reject=True        # 启用拒绝异常抛出
        )
    except RejectedOrderError as e:
        print(f"订单被拒绝: {e}")
        # 处理订单拒绝的情况
        # 可能需要调整订单大小或检查资金状况
    
    # 检查订单状态而不抛出异常
    pf = vbt.Portfolio.from_orders(
        close=price_data,
        size=order_sizes,
        raise_reject=False  # 不抛出异常，通过状态检查处理拒绝
    )
    
    # 分析被拒绝的订单
    orders = pf.orders
    rejected_mask = orders.res_status == vbt.OrderStatus.Rejected
    rejected_orders = orders.iloc[rejected_mask]
    print(f"被拒绝的订单数: {rejected_mask.sum()}")
    ```
    
    注意事项：
    - 只有在Order.raise_reject=True时才会抛出此异常
    - 在生产环境中，建议通过状态检查而非异常来处理订单拒绝
    - 异常会中断模拟过程，因此主要用于调试和错误检测
    - 可以通过OrderStatusInfo枚举获取具体的拒绝原因
    
    订单拒绝异常。当投资组合模拟系统拒绝执行订单时抛出。
    """
    pass


# ############# 枚举类型定义 ############# #


class InitCashModeT(tp.NamedTuple):
    """
    初始资金模式类型定义
    
    使用NamedTuple定义的伪枚举类，用于指定投资组合模拟中初始资金的设置模式。
    这种设计在保持类型安全的同时，提供了良好的性能和可读性。
    """
    Auto: int = 0        # 自动模式：模拟内资金无限，然后设置为总花费金额
    AutoAlign: int = 1   # 自动对齐模式：设置为所有列总花费金额的对齐值


InitCashMode = InitCashModeT()
"""初始资金模式枚举实例，提供两种初始资金设置策略"""

__pdoc__['InitCashMode'] = f"""初始资金模式枚举

```json
{to_doc(InitCashMode)}
```

该枚举定义了投资组合模拟中初始资金的不同设置模式，用于控制模拟开始时的资金配置策略。

属性说明:
    Auto (0): 自动模式
        - 模拟过程中资金视为无限
        - 模拟结束后设置为实际花费的总金额
        - 适用于想要了解策略实际资金需求的场景
        
    AutoAlign (1): 自动对齐模式  
        - 设置为所有列（资产）总花费金额的统一值
        - 确保所有资产具有相同的初始资金配置
        - 适用于公平比较不同资产表现的场景

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import InitCashMode

# 使用自动模式
pf1 = vbt.Portfolio.from_orders(
    close=price_data,
    size=order_sizes,
    init_cash=InitCashMode.Auto  # 资金无限，后续调整为实际需求
)

# 使用自动对齐模式
pf2 = vbt.Portfolio.from_orders(
    close=price_data,
    size=order_sizes, 
    init_cash=InitCashMode.AutoAlign  # 所有列使用相同的初始资金
)

print(f"自动模式实际资金需求: {pf1.init_cash}")
print(f"对齐模式统一初始资金: {pf2.init_cash}")
```

应用场景:
- 策略资金需求分析：使用Auto模式了解策略的实际资金需求
- 多资产对比分析：使用AutoAlign模式确保公平的比较基础
- 动态资金配置：根据不同策略特点选择合适的资金模式
"""


class CallSeqTypeT(tp.NamedTuple):
    """
    调用序列类型定义
    
    定义在模拟过程中多个资产（列）的处理顺序，影响资金分配和订单执行的优先级。
    在现金共享的场景中，调用顺序会直接影响策略的执行结果。
    """
    Default: int = 0    # 默认顺序：从左到右依次处理
    Reversed: int = 1   # 反向顺序：从右到左依次处理
    Random: int = 2     # 随机顺序：随机打乱处理顺序
    Auto: int = 3       # 自动顺序：基于订单价值动态排序


CallSeqType = CallSeqTypeT()
"""调用序列类型枚举实例，控制多资产处理的顺序策略"""

__pdoc__['CallSeqType'] = f"""调用序列类型枚举

```json
{to_doc(CallSeqType)}
```

该枚举控制投资组合模拟中多个资产（列）的处理顺序，特别是在现金共享的场景下，
调用顺序会显著影响订单的执行和资金的分配。

属性说明:
    Default (0): 默认顺序
        - 按列的自然顺序从左到右处理
        - 第一列优先获得资金分配
        - 最简单和最常用的处理方式
        
    Reversed (1): 反向顺序
        - 按列的相反顺序从右到左处理  
        - 最后一列优先获得资金分配
        - 用于测试顺序敏感性
        
    Random (2): 随机顺序
        - 每个时间步随机打乱处理顺序
        - 消除顺序偏差，提供更公平的资金分配
        - 用于风险管理和鲁棒性测试
        
    Auto (3): 自动顺序
        - 根据订单的价值动态排序处理
        - 卖单优先执行以释放资金供买单使用
        - 实现更智能的资金利用，但可能引入前瞻偏差

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import CallSeqType

# 默认顺序处理
pf_default = vbt.Portfolio.from_orders(
    close=price_data,
    size=order_sizes,
    call_seq=CallSeqType.Default,
    cash_sharing=True  # 启用现金共享以观察顺序影响
)

# 自动智能排序
pf_auto = vbt.Portfolio.from_orders(
    close=price_data,
    size=order_sizes,
    call_seq=CallSeqType.Auto,  # 基于订单价值自动排序
    cash_sharing=True
)

# 随机顺序以消除偏差
pf_random = vbt.Portfolio.from_orders(
    close=price_data,
    size=order_sizes,
    call_seq=CallSeqType.Random,  # 随机处理顺序
    cash_sharing=True,
    seed=42  # 设置随机种子确保可重现
)

# 比较不同顺序的影响
print("默认顺序总收益:", pf_default.total_return())
print("自动顺序总收益:", pf_auto.total_return()) 
print("随机顺序总收益:", pf_random.total_return())
```

注意事项:
- Auto模式虽然智能，但可能引入前瞻偏差，在实盘中需要谨慎使用
- Random模式需要设置种子以确保结果的可重现性
- 在单资产或无现金共享的场景下，调用顺序不会产生影响
- 复杂的多资产策略应该测试不同调用顺序的敏感性
"""


class AccumulationModeT(tp.NamedTuple):
    """
    仓位累积模式类型定义
    
    定义了如何处理仓位的逐步增加和减少。累积模式允许策略通过多次交易
    逐渐建立或缩减仓位，而不是一次性完成所有交易操作。
    """
    Disabled: int = 0    # 禁用累积：不允许仓位累积
    Both: int = 1        # 双向累积：允许增加和减少仓位
    AddOnly: int = 2     # 仅增加：只允许增加仓位
    RemoveOnly: int = 3  # 仅减少：只允许减少仓位


AccumulationMode = AccumulationModeT()
"""仓位累积模式枚举实例，控制仓位的逐步建立和调整策略"""

__pdoc__['AccumulationMode'] = f"""仓位累积模式枚举

```json
{to_doc(AccumulationMode)}
```

该枚举控制投资组合中仓位的累积行为，允许策略通过多次较小的交易逐步建立或调整仓位，
而不是通过单次大额交易完成所有操作。这种机制在风险管理和市场影响控制方面非常重要。

属性说明:
    Disabled (0): 禁用累积
        - 不允许任何形式的仓位累积
        - 每次信号都会完整执行，不考虑现有仓位
        - 适用于简单的买入-持有-卖出策略
        
    Both (1): 双向累积
        - 允许在现有仓位基础上继续增加或减少
        - 支持分批建仓和分批减仓
        - 提供最大的灵活性，适合复杂策略
        
    AddOnly (2): 仅增加模式
        - 只允许在现有仓位基础上增加
        - 不允许减少现有仓位
        - 适用于趋势跟踪和动量策略
        
    RemoveOnly (3): 仅减少模式
        - 只允许减少现有仓位
        - 不允许增加现有仓位
        - 适用于止盈和风险控制策略

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import AccumulationMode

# 创建分批建仓的信号
entries = [True, False, True, False, False]  # 第1、3天有入场信号
exits = [False, False, False, False, True]   # 第5天有出场信号

# 禁用累积模式 - 第3天的信号会被忽略，因为已有仓位
pf_disabled = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entries,
    exits=exits,
    size=100,
    accumulate=AccumulationMode.Disabled
)

# 双向累积模式 - 第3天会增加仓位
pf_both = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entries,
    exits=exits,
    size=100,
    accumulate=AccumulationMode.Both  # 允许累积
)

# 仅增加模式 - 只能在现有多头基础上继续增加
pf_add_only = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entries,
    exits=exits,
    size=100,
    accumulate=AccumulationMode.AddOnly
)

print(f"禁用累积最大仓位: {pf_disabled.asset_flow().max()}")  # 100
print(f"双向累积最大仓位: {pf_both.asset_flow().max()}")    # 200  
print(f"仅增加最大仓位: {pf_add_only.asset_flow().max()}")  # 200
```

重要说明:
!!! note
    累积模式在退出和反向入场时表现不同：
    - 退出：减少当前仓位但不进入反向仓位
    - 反向入场：减少同等数量的仓位，一旦仓位关闭，开始建立反向仓位
    
    反向入场行为可通过 `OppositeEntryMode` 调整，止损订单行为可通过 `StopExitMode` 调整。

应用场景:
- **分批建仓策略**: 使用AddOnly模式逐步建立仓位，降低市场冲击
- **动态仓位管理**: 使用Both模式根据市场条件灵活调整仓位大小
- **风险控制策略**: 使用RemoveOnly模式实现分批止盈和风险管理
- **趋势跟踪策略**: 在趋势确认后逐步加仓，充分捕获趋势收益
"""


class ConflictModeT(tp.NamedTuple):
    """
    信号冲突处理模式类型定义
    
    定义当同一时间点同时出现入场和出场信号时的处理策略。这种情况在
    实际交易中经常发生，需要明确的规则来决定优先执行哪个信号。
    """
    Ignore: int = 0    # 忽略：同时忽略两个冲突信号
    Entry: int = 1     # 入场优先：优先执行入场信号
    Exit: int = 2      # 出场优先：优先执行出场信号  
    Adjacent: int = 3  # 相邻优先：执行与当前状态相邻的信号
    Opposite: int = 4  # 相反优先：执行与当前状态相反的信号


ConflictMode = ConflictModeT()
"""信号冲突处理模式枚举实例，控制同时出现的入场出场信号的优先级"""

__pdoc__['ConflictMode'] = f"""信号冲突处理模式枚举

```json
{to_doc(ConflictMode)}
```

该枚举定义了当同一时间点同时出现入场和出场信号时的处理策略。在复杂的交易策略中，
不同的技术指标可能会产生相互冲突的信号，需要明确的优先级规则。

属性说明:
    Ignore (0): 忽略冲突
        - 当入场和出场信号同时出现时，两个信号都被忽略
        - 维持当前仓位不变
        - 适用于保守的策略，避免在不确定时执行交易
        
    Entry (1): 入场信号优先
        - 优先执行入场信号，忽略出场信号
        - 倾向于建立新仓位或增加仓位
        - 适用于积极的趋势跟踪策略
        
    Exit (2): 出场信号优先
        - 优先执行出场信号，忽略入场信号
        - 倾向于平仓或减少仓位
        - 适用于风险规避和止损策略
        
    Adjacent (3): 相邻信号优先
        - 只在有持仓时生效，否则忽略所有信号
        - 执行与当前状态"相邻"的操作（如持多头时的增仓或减仓）
        - 避免不合理的仓位跳跃
        
    Opposite (4): 相反信号优先
        - 只在有持仓时生效，否则忽略所有信号  
        - 执行与当前状态"相反"的操作（如持多头时的空头信号）
        - 适用于反转策略

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import ConflictMode

# 创建冲突信号：第3天同时有入场和出场信号
long_entries = [True, False, True, False, False]   # 第1、3天多头入场
long_exits = [False, False, True, False, False]    # 第3天多头出场
short_entries = [False, False, False, False, False]
short_exits = [False, False, False, False, False]

# 忽略冲突模式
pf_ignore = vbt.Portfolio.from_signals(
    close=price_data,
    entries=long_entries,
    exits=long_exits, 
    upon_long_conflict=ConflictMode.Ignore  # 冲突时忽略
)

# 入场优先模式 
pf_entry = vbt.Portfolio.from_signals(
    close=price_data,
    entries=long_entries,
    exits=long_exits,
    upon_long_conflict=ConflictMode.Entry  # 入场信号优先
)

# 出场优先模式
pf_exit = vbt.Portfolio.from_signals(
    close=price_data,
    entries=long_entries,
    exits=long_exits, 
    upon_long_conflict=ConflictMode.Exit   # 出场信号优先
)

# 检查第3天的仓位变化
print(f"忽略模式第3天仓位: {pf_ignore.asset_flow().iloc[2]}")  # 0（忽略）
print(f"入场优先第3天仓位: {pf_entry.asset_flow().iloc[2]}")   # 增加仓位
print(f"出场优先第3天仓位: {pf_exit.asset_flow().iloc[2]}")    # 平仓
```

应用场景:
- **保守策略**: 使用Ignore模式在信号不明确时保持观望
- **趋势策略**: 使用Entry模式在趋势确认时积极建仓  
- **风控策略**: 使用Exit模式在风险信号出现时及时止损
- **智能策略**: 使用Adjacent/Opposite模式实现更精细的仓位控制

注意事项:
- Adjacent和Opposite模式只在有持仓时才会生效
- 冲突处理的选择应该与整体策略逻辑保持一致
- 建议通过回测验证不同冲突处理模式对策略表现的影响
"""


class DirectionConflictModeT(tp.NamedTuple):
    """
    方向冲突处理模式类型定义
    
    定义当同一时间点同时出现多头和空头入场信号时的处理策略。在复杂策略中，
    不同指标可能同时产生相反方向的信号，需要明确的规则来决定优先级。
    """
    Ignore: int = 0    # 忽略：同时忽略多头和空头信号
    Long: int = 1      # 多头优先：优先执行多头入场信号
    Short: int = 2     # 空头优先：优先执行空头入场信号
    Adjacent: int = 3  # 相邻优先：执行与当前仓位相邻的信号
    Opposite: int = 4  # 相反优先：执行与当前仓位相反的信号


DirectionConflictMode = DirectionConflictModeT()
"""方向冲突处理模式枚举实例，控制多头空头入场信号的优先级策略"""

__pdoc__['DirectionConflictMode'] = f"""方向冲突处理模式枚举

```json
{to_doc(DirectionConflictMode)}
```

该枚举定义了当同一时间点同时出现多头和空头入场信号时的处理策略。在多策略组合或
复杂技术分析系统中，经常出现相反方向的信号同时触发的情况。

属性说明:
    Ignore (0): 忽略所有冲突
        - 当多头和空头入场信号同时出现时，两个信号都被忽略
        - 保持当前仓位状态不变
        - 适用于保守的策略，避免在方向不明确时建仓
        
    Long (1): 多头信号优先
        - 优先执行多头入场信号，忽略空头入场信号
        - 倾向于建立或增加多头仓位
        - 适用于偏向看涨的策略
        
    Short (2): 空头信号优先
        - 优先执行空头入场信号，忽略多头入场信号
        - 倾向于建立或增加空头仓位
        - 适用于偏向看跌的策略
        
    Adjacent (3): 相邻信号优先
        - 只在有持仓时生效，无仓位时忽略所有信号
        - 执行与当前仓位方向相同的信号（如持多头时的多头信号）
        - 避免频繁的方向转换，适用于趋势跟踪策略
        
    Opposite (4): 相反信号优先
        - 只在有持仓时生效，无仓位时忽略所有信号
        - 执行与当前仓位方向相反的信号（如持多头时的空头信号）
        - 适用于反转策略和风险管理

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import DirectionConflictMode

# 创建方向冲突信号：第3天同时有多头和空头入场信号
long_entries = [True, False, True, False, False]   # 第1、3天多头信号
short_entries = [False, False, True, False, False]  # 第3天空头信号
long_exits = [False, False, False, False, True]
short_exits = [False, False, False, False, True]

# 忽略冲突模式
pf_ignore = vbt.Portfolio.from_signals(
    close=price_data,
    entries=long_entries,
    short_entries=short_entries,
    exits=long_exits,
    short_exits=short_exits,
    upon_dir_conflict=DirectionConflictMode.Ignore  # 方向冲突时忽略
)

# 多头优先模式
pf_long = vbt.Portfolio.from_signals(
    close=price_data,
    entries=long_entries,
    short_entries=short_entries,
    exits=long_exits,
    short_exits=short_exits,
    upon_dir_conflict=DirectionConflictMode.Long   # 多头信号优先
)

# 空头优先模式
pf_short = vbt.Portfolio.from_signals(
    close=price_data,
    entries=long_entries,
    short_entries=short_entries,
    exits=long_exits,
    short_exits=short_exits,
    upon_dir_conflict=DirectionConflictMode.Short  # 空头信号优先
)

# 比较不同处理方式的结果
print(f"忽略模式收益: {pf_ignore.total_return()}")
print(f"多头优先收益: {pf_long.total_return()}")
print(f"空头优先收益: {pf_short.total_return()}")
```

应用场景:
- **多策略组合**: 不同子策略产生相反信号时的统一处理
- **技术分析系统**: 多个指标产生冲突信号时的优先级管理
- **风险管理**: 通过Adjacent/Opposite模式控制仓位变化频率
- **市场中性策略**: 通过Ignore模式在信号不明确时保持中性

注意事项:
- Adjacent和Opposite模式只在有持仓时才生效
- 应该结合具体的市场环境和策略特点选择合适的模式
- 建议通过历史数据测试不同模式的效果
"""


class OppositeEntryModeT(tp.NamedTuple):
    """
    反向入场模式类型定义
    
    定义当持有某个方向的仓位时，如果在退出信号之前收到反向入场信号时的处理策略。
    这是一个常见的交易场景，需要明确的规则来处理仓位转换。
    """
    Ignore: int = 0          # 忽略：忽略反向入场信号
    Close: int = 1           # 平仓：直接平掉当前仓位
    CloseReduce: int = 2     # 平仓减少：平仓或在累积模式下减少仓位
    Reverse: int = 3         # 反转：反转为反向仓位
    ReverseReduce: int = 4   # 反转减少：反转或在累积模式下减少后反转


OppositeEntryMode = OppositeEntryModeT()
"""反向入场模式枚举实例，控制持仓时反向入场信号的处理策略"""

__pdoc__['OppositeEntryMode'] = f"""反向入场模式枚举

```json
{to_doc(OppositeEntryMode)}
```

该枚举定义了当持有某个方向的仓位时，在退出信号之前收到反向入场信号时的处理策略。
这种情况在趋势反转或策略调整时经常发生。

属性说明:
    Ignore (0): 忽略反向信号
        - 完全忽略反向入场信号，保持当前仓位不变
        - 等待正常的退出信号才平仓
        - 适用于坚持原有方向判断的策略
        
    Close (1): 直接平仓
        - 收到反向信号时立即平掉当前仓位
        - 不建立反向仓位，回到空仓状态
        - 适用于保守的仓位管理策略
        
    CloseReduce (2): 平仓或减少
        - 禁用累积时：直接平掉当前仓位
        - 启用累积时：按信号大小减少当前仓位
        - 提供更灵活的仓位调整机制
        
    Reverse (3): 完全反转
        - 平掉当前仓位并建立等量的反向仓位
        - 实现从多头到空头（或相反）的直接转换
        - 适用于快速响应市场变化的策略
        
    ReverseReduce (4): 反转或减少
        - 禁用累积时：完全反转仓位方向
        - 启用累积时：先减少当前仓位，完全平仓后建立反向仓位
        - 在累积模式下提供渐进式的方向转换

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import OppositeEntryMode

# 创建仓位反转场景：持有多头后收到空头信号
long_entries = [True, False, False, False, False]   # 第1天建立多头仓位
short_entries = [False, False, True, False, False]  # 第3天收到空头信号
long_exits = [False, False, False, False, True]     # 第5天多头退出信号
short_exits = [False, False, False, False, True]

# 忽略反向信号
pf_ignore = vbt.Portfolio.from_signals(
    close=price_data,
    entries=long_entries,
    short_entries=short_entries,
    exits=long_exits,
    short_exits=short_exits,
    upon_opposite_entry=OppositeEntryMode.Ignore  # 忽略反向入场
)

# 直接反转仓位
pf_reverse = vbt.Portfolio.from_signals(
    close=price_data,
    entries=long_entries,
    short_entries=short_entries,
    exits=long_exits,
    short_exits=short_exits,
    upon_opposite_entry=OppositeEntryMode.Reverse  # 反转仓位方向
)

# 先平仓再观察
pf_close = vbt.Portfolio.from_signals(
    close=price_data,
    entries=long_entries,
    short_entries=short_entries,
    exits=long_exits,
    short_exits=short_exits,
    upon_opposite_entry=OppositeEntryMode.Close    # 平仓但不反转
)

# 分析不同处理方式的仓位变化
print("忽略模式仓位变化:", pf_ignore.asset_flow().values)
print("反转模式仓位变化:", pf_reverse.asset_flow().values)  
print("平仓模式仓位变化:", pf_close.asset_flow().values)
```

应用场景:
- **趋势反转策略**: 使用Reverse模式快速调整仓位方向
- **风险管理策略**: 使用Close模式在不确定时先平仓观察
- **渐进调整策略**: 使用CloseReduce/ReverseReduce配合累积模式
- **保守交易策略**: 使用Ignore模式坚持原有方向判断

与累积模式的配合:
- 当accumulate=True时，CloseReduce和ReverseReduce会表现不同
- 累积模式允许分批调整仓位，而非一次性全部转换
- 建议根据具体策略需求选择合适的组合方式
"""


class StopEntryPriceT(tp.NamedTuple):
    """
    止损入场价格类型定义
    
    定义在设置止损订单时使用哪种价格作为初始止损价格的基准。
    不同的价格基准会影响止损订单的触发时机和执行效果。
    """
    ValPrice: int = 0   # 估值价格：使用资产估值价格
    Price: int = 1      # 默认价格：使用默认价格
    FillPrice: int = 2  # 成交价格：使用成交价格（已包含滑点）
    Close: int = 3      # 收盘价格：使用收盘价格


StopEntryPrice = StopEntryPriceT()
"""止损入场价格枚举实例，定义止损订单的价格基准选择"""

__pdoc__['StopEntryPrice'] = f"""止损入场价格枚举

```json
{to_doc(StopEntryPrice)}
```

该枚举定义了在设置止损订单时使用哪种价格作为初始止损价格的基准。
选择不同的价格基准会影响止损的触发条件和风险控制效果。

属性说明:
    ValPrice (0): 资产估值价格
        - 使用资产的当前估值价格作为止损基准
        - 通常是最新的市场价格或理论价值
        - 适用于需要精确估值的复杂策略
        
    Price (1): 默认价格
        - 使用系统默认的价格作为基准
        - 通常是当前K线的某个标准价格
        - 最常用的选择，适合大多数场景
        
    FillPrice (2): 实际成交价格
        - 使用订单的实际成交价格（已包含滑点影响）
        - 反映真实的交易成本
        - 适用于精确的风险控制和盈亏计算
        
    Close (3): 收盘价格
        - 使用K线的收盘价作为基准
        - 提供稳定的价格参考
        - 适用于日线或更长周期的策略

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import StopEntryPrice

# 使用收盘价设置止损
pf_close = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entry_signals,
    exits=exit_signals,
    sl_stop=0.05,  # 5%止损
    stop_entry_price=StopEntryPrice.Close,  # 基于收盘价
    size=100
)

# 使用成交价设置止损（考虑滑点）
pf_fill = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entry_signals,
    exits=exit_signals,
    sl_stop=0.05,  # 5%止损
    stop_entry_price=StopEntryPrice.FillPrice,  # 基于实际成交价
    slippage=0.001,  # 0.1%滑点
    size=100
)

print(f"基于收盘价止损的收益: {pf_close.total_return()}")
print(f"基于成交价止损的收益: {pf_fill.total_return()}")
```

选择建议:
- **日内交易**: 推荐使用Price或FillPrice获得更准确的止损
- **长期投资**: 可以使用Close价格，更加稳定可靠
- **精确风控**: 使用FillPrice确保止损基于真实成交成本
- **简单策略**: 使用Price作为默认选择
"""


class StopExitPriceT(tp.NamedTuple):
    """
    止损退出价格类型定义
    
    定义当止损信号触发时，使用哪种价格执行平仓操作。不同的价格选择
    会影响止损的执行效果和最终的盈亏结果。
    """
    StopLimit: int = 0   # 止损限价：以止损价格作为限价单执行
    StopMarket: int = 1  # 止损市价：以止损价格作为市价单执行  
    Price: int = 2       # 默认价格：使用默认价格执行
    Close: int = 3       # 收盘价格：使用收盘价执行


StopExitPrice = StopExitPriceT()
"""止损退出价格枚举实例，定义止损触发时的执行价格策略"""

__pdoc__['StopExitPrice'] = f"""止损退出价格枚举

```json
{to_doc(StopExitPrice)}
```

该枚举定义了当止损信号触发时，使用哪种价格执行平仓操作。不同的价格选择
会显著影响止损的实际执行效果和最终收益。

属性说明:
    StopLimit (0): 止损限价单
        - 以止损触发价格作为限价单执行
        - 如果止损之前已被触发，使用下一K线的开盘价
        - 不应用用户自定义的滑点设置
        - 提供价格保护但可能无法成交
        
    StopMarket (1): 止损市价单
        - 以止损触发价格作为市价单执行
        - 如果止损之前已被触发，使用下一K线的开盘价
        - 应用用户自定义的滑点设置
        - 确保成交但价格可能不理想
        
    Price (2): 默认价格
        - 使用系统默认价格执行止损
        - 应用用户自定义的滑点设置
        - 最常用的选择
        
    Close (3): 收盘价格
        - 使用K线收盘价执行止损
        - 应用用户自定义的滑点设置
        - 适用于较长周期的策略

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import StopExitPrice

# 使用止损限价单（价格保护）
pf_limit = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entry_signals,
    exits=exit_signals,
    sl_stop=0.05,  # 5%止损
    stop_exit_price=StopExitPrice.StopLimit,  # 限价单执行
    size=100
)

# 使用止损市价单（确保成交）
pf_market = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entry_signals,
    exits=exit_signals,
    sl_stop=0.05,  # 5%止损  
    stop_exit_price=StopExitPrice.StopMarket,  # 市价单执行
    slippage=0.001,  # 应用滑点
    size=100
)

# 使用收盘价执行（稳定但可能滞后）
pf_close = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entry_signals,
    exits=exit_signals,
    sl_stop=0.05,  # 5%止损
    stop_exit_price=StopExitPrice.Close,  # 收盘价执行
    size=100
)

print(f"限价止损收益: {pf_limit.total_return()}")
print(f"市价止损收益: {pf_market.total_return()}")
print(f"收盘价止损收益: {pf_close.total_return()}")
```

重要说明:
!!! note
    止损订单的执行限制：
    
    1) 止损信号不能在入场信号的同一K线处理
    
    2) 止损信号可能与用户信号冲突。系统假设止损信号在时间上优先于其他信号。
       因此，在使用止损订单时，建议其他信号使用收盘价执行，避免前瞻偏差。

!!! warning
    注意Price选项的使用：
    确保只在与StopEntryPrice.Close配合使用StopExitPrice.Price，
    否则无法保证价格的时间顺序正确性。

应用场景:
- **风险优先**: 使用StopMarket确保止损一定执行
- **价格优先**: 使用StopLimit在可接受价格范围内止损  
- **简单策略**: 使用Price作为通用选择
- **长期持仓**: 使用Close价格，降低短期波动影响
"""


class StopExitModeT(tp.NamedTuple):
    """
    止损退出模式类型定义
    
    定义当止损信号触发时如何处理当前仓位。不同的退出模式适用于
    不同的交易策略和风险管理需求。
    """
    Close: int = 0           # 平仓：直接平掉当前仓位
    CloseReduce: int = 1     # 平仓减少：平仓或在累积模式下减少仓位
    Reverse: int = 2         # 反转：反转为反向仓位
    ReverseReduce: int = 3   # 反转减少：反转或在累积模式下减少后反转


StopExitMode = StopExitModeT()
"""止损退出模式枚举实例，控制止损触发时的仓位处理策略"""

__pdoc__['StopExitMode'] = f"""止损退出模式枚举

```json
{to_doc(StopExitMode)}
```

该枚举定义了当止损信号触发时如何处理当前仓位。不同的退出模式提供了从简单平仓
到复杂仓位反转的多种策略，以适应不同的交易需求和风险管理要求。

属性说明:
    Close (0): 直接平仓
        - 止损触发时立即平掉当前全部仓位
        - 回到空仓状态，不建立新的仓位
        - 最简单和最常用的止损处理方式
        - 适用于绝大多数风险管理场景
        
    CloseReduce (1): 平仓或减少
        - 禁用累积模式时：直接平掉当前仓位
        - 启用累积模式时：按照止损信号大小减少仓位
        - 提供更灵活的仓位管理能力
        - 适用于需要渐进式风险控制的策略
        
    Reverse (2): 完全反转
        - 平掉当前仓位并建立等量的反向仓位
        - 从多头直接转为等量空头（或相反）
        - 适用于认为市场将反向运行的策略
        - 风险较高，需要谨慎使用
        
    ReverseReduce (3): 反转或减少
        - 禁用累积模式时：完全反转仓位方向
        - 启用累积模式时：先减少当前仓位，完全平仓后建立反向仓位
        - 在累积模式下提供渐进式的反转能力
        - 结合了反转策略和渐进式调整的优点

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import StopExitMode

# 传统止损：直接平仓
pf_close = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entry_signals,
    exits=exit_signals,
    sl_stop=0.05,  # 5%止损
    stop_exit_mode=StopExitMode.Close,  # 直接平仓
    size=100
)

# 反转策略：止损时建立反向仓位
pf_reverse = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entry_signals,
    exits=exit_signals,
    sl_stop=0.05,  # 5%止损
    stop_exit_mode=StopExitMode.Reverse,  # 反转仓位
    size=100
)

# 累积模式下的渐进式止损
pf_reduce = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entry_signals,
    exits=exit_signals,
    sl_stop=0.05,  # 5%止损
    stop_exit_mode=StopExitMode.CloseReduce,  # 减少仓位
    accumulate=True,  # 启用累积模式
    size=50  # 较小的单次交易规模
)

print(f"直接平仓止损收益: {pf_close.total_return()}")
print(f"反转止损策略收益: {pf_reverse.total_return()}")
print(f"渐进止损策略收益: {pf_reduce.total_return()}")

# 分析仓位变化模式
print("\\n直接平仓模式仓位变化:")
print(pf_close.asset_flow().dropna())

print("\\n反转模式仓位变化:")  
print(pf_reverse.asset_flow().dropna())
```

策略选择建议:
- **保守投资者**: 使用Close模式，确保风险及时控制
- **激进交易者**: 考虑使用Reverse模式，但需要充分的市场分析支持
- **渐进式管理**: 配合累积模式使用CloseReduce或ReverseReduce
- **复杂策略**: 根据不同市场条件动态选择退出模式

风险提示:
- Reverse和ReverseReduce模式会建立新的仓位，增加了额外的市场风险
- 反转策略需要对市场趋势有准确判断，否则可能加大损失
- 在高波动市场中，反转策略可能导致频繁的方向变换
- 建议结合其他技术指标确认反转信号的有效性

与其他参数的配合:
- accumulate参数：影响CloseReduce和ReverseReduce的具体行为
- stop_exit_price参数：决定退出时使用的价格类型
- 信号优先级：止损信号通常具有最高优先级
            
"""


class StopUpdateModeT(tp.NamedTuple):
    """
    止损更新模式类型定义
    
    定义当建立新仓位时如何处理已有的止损设置。这在累积交易和仓位调整时
    特别重要，决定了止损策略的连续性和有效性。
    """
    Keep: int = 0           # 保持：保留旧的止损设置
    Override: int = 1       # 覆盖：用新的止损覆盖旧设置（如果新止损不为NaN）
    OverrideNaN: int = 2    # 强制覆盖：用新的止损覆盖旧设置（即使新止损为NaN）


StopUpdateMode = StopUpdateModeT()
"""止损更新模式枚举实例，控制新仓位建立时的止损设置策略"""

__pdoc__['StopUpdateMode'] = f"""止损更新模式枚举

```json
{to_doc(StopUpdateMode)}
```

该枚举定义了当建立新仓位时如何处理已有止损设置的策略。在累积交易、仓位调整或
重新入场时，需要明确的规则来决定是保持原有止损还是更新为新的止损水平。

属性说明:
    Keep (0): 保持原有止损
        - 保留旧的止损设置不变
        - 新的仓位继续使用原有的止损水平
        - 适用于希望维持一致止损策略的场景
        - 确保止损策略的连续性
        
    Override (1): 条件覆盖模式
        - 只有当新的止损不为NaN时才覆盖旧设置
        - 如果新止损为NaN，则保持原有止损
        - 提供了智能的止损更新机制
        - 避免因数据问题导致止损失效
        
    OverrideNaN (2): 强制覆盖模式
        - 无论新止损是否为NaN都覆盖旧设置
        - 即使新止损为NaN也会替换原有止损
        - 可能导致止损保护失效，需要谨慎使用
        - 适用于需要完全重置止损的场景

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import StopUpdateMode

# 保持原有止损策略
pf_keep = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entry_signals,
    exits=exit_signals,
    sl_stop=0.05,  # 5%止损
    accumulate=True,  # 允许累积建仓
    stop_update_mode=StopUpdateMode.Keep,  # 保持原有止损
    size=100
)

# 智能覆盖止损
pf_override = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entry_signals,
    exits=exit_signals,
    sl_stop=pd.Series([0.05, np.nan, 0.03]),  # 变化的止损设置
    accumulate=True,
    stop_update_mode=StopUpdateMode.Override,  # 有效值时覆盖
    size=100
)

# 强制覆盖所有止损
pf_force = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entry_signals,
    exits=exit_signals,
    sl_stop=pd.Series([0.05, np.nan, 0.03]),  # 包含NaN的止损
    accumulate=True,
    stop_update_mode=StopUpdateMode.OverrideNaN,  # 强制覆盖
    size=100
)

print(f"保持模式收益: {pf_keep.total_return():.2%}")
print(f"智能覆盖收益: {pf_override.total_return():.2%}")
print(f"强制覆盖收益: {pf_force.total_return():.2%}")
```

策略应用场景:
- **Keep模式**：
  - 长期持仓策略，希望维持一致的风险控制
  - 分批建仓时保持统一的止损水平
  - 避免因短期调整影响整体风险管理
  
- **Override模式**：
  - 动态调整止损的策略
  - 根据市场条件优化风险控制
  - 智能处理数据缺失情况
  
- **OverrideNaN模式**：
  - 需要完全重置止损的场景
  - 清除所有止损保护的特殊策略
  - 高级用户的自定义止损逻辑

风险提示:
- OverrideNaN模式可能导致止损保护失效，使用时需要特别谨慎
- 频繁更新止损可能影响策略的稳定性
- 建议在回测中充分验证不同模式的影响
- 实盘交易中应优先考虑风险保护的连续性

与其他参数的配合:
- accumulate参数：影响何时触发止损更新
- sl_stop/tp_stop参数：提供新的止损水平
- 仓位管理策略：决定止损更新的时机和方式
"""


class SizeTypeT(tp.NamedTuple):
    """
    订单大小类型定义
    
    定义不同的订单大小指定方式。这是投资组合系统中最重要的参数之一，
    直接影响资金管理、仓位控制和风险管理的精确度。
    """
    Amount: int = 0          # 绝对数量：指定交易的资产绝对数量
    Value: int = 1           # 价值金额：指定交易的资产价值金额
    Percent: int = 2         # 可用资源百分比：基于可用资源的百分比
    TargetAmount: int = 3    # 目标数量：指定目标持仓的绝对数量
    TargetValue: int = 4     # 目标价值：指定目标持仓的价值金额
    TargetPercent: int = 5   # 目标百分比：指定目标持仓占总价值的百分比


SizeType = SizeTypeT()
"""订单大小类型枚举实例，定义各种资金和仓位管理方式"""

__pdoc__['SizeType'] = f"""订单大小类型枚举

```json
{to_doc(SizeType)}
```

该枚举定义了指定订单大小的不同方式，是vectorbt投资组合管理系统中最核心的参数之一。
不同的大小类型适用于不同的交易策略和资金管理需求。

属性说明:
    Amount (0): 绝对数量模式
        - 直接指定要交易的资产数量（如股数、手数等）
        - 最直接和常用的方式
        - 适用于明确知道交易数量的策略
        
    Value (1): 价值金额模式
        - 指定要交易的资产价值金额
        - 通过 OrderContext.val_price_now 转换为Amount模式
        - 适用于基于固定金额的投资策略
        
    Percent (2): 可用资源百分比模式
        - 基于当前可用资源的百分比（注意：不是仓位价值的百分比！）
        - 买入时：基于 OrderContext.cash_now 的百分比
        - 卖出时：基于 OrderContext.position_now 的百分比  
        - 做空时：基于 OrderContext.free_cash_now 的百分比
        - 反转仓位时：同时考虑position_now和free_cash_now
        - 自动考虑手续费和滑点限制
        
    TargetAmount (3): 目标数量模式
        - 指定最终要持有的资产目标数量（=目标仓位）
        - 使用 OrderContext.position_now 获取当前仓位
        - 自动计算需要交易的数量，转换为Amount模式
        - 适用于仓位再平衡策略
        
    TargetValue (4): 目标价值模式
        - 指定最终要持有的资产目标价值
        - 使用 OrderContext.val_price_now 获取当前资产价值
        - 转换为TargetAmount模式执行
        - 适用于基于价值的仓位管理
        
    TargetPercent (5): 目标百分比模式
        - 指定目标仓位占总投资组合价值的百分比
        - 使用 OrderContext.value_now 获取当前总价值
        - 转换为TargetValue模式执行
        - 适用于动态再平衡和风险平价策略

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType

# 绝对数量交易
pf_amount = vbt.Portfolio.from_orders(
    close=price_data,
    size=pd.Series([100, -50, 25]),  # 买100股，卖50股，买25股
    size_type=SizeType.Amount,  # 绝对数量模式
    init_cash=10000
)

# 固定金额交易
pf_value = vbt.Portfolio.from_orders(
    close=price_data,
    size=pd.Series([1000, -500, 300]),  # 买1000元，卖500元价值，买300元
    size_type=SizeType.Value,   # 价值金额模式
    init_cash=10000
)

# 百分比资金管理
pf_percent = vbt.Portfolio.from_orders(
    close=price_data,
    size=pd.Series([0.5, -0.3, 0.2]),  # 50%资金买入，30%仓位卖出，20%资金买入
    size_type=SizeType.Percent,  # 百分比模式
    init_cash=10000
)

# 目标仓位管理
pf_target = vbt.Portfolio.from_orders(
    close=price_data,
    size=pd.Series([100, 50, 0]),  # 目标100股，目标50股，目标0股（清仓）
    size_type=SizeType.TargetAmount,  # 目标数量模式
    init_cash=10000
)

# 目标价值百分比（投资组合再平衡）
pf_rebalance = vbt.Portfolio.from_orders(
    close=price_data,
    size=pd.Series([0.6, 0.4, 0.5]),  # 目标60%，40%，50%的总价值
    size_type=SizeType.TargetPercent,  # 目标百分比模式
    init_cash=10000
)

print(f"绝对数量模式收益: {pf_amount.total_return():.2%}")
print(f"价值金额模式收益: {pf_value.total_return():.2%}")
print(f"百分比模式收益: {pf_percent.total_return():.2%}")
print(f"目标仓位模式收益: {pf_target.total_return():.2%}")
print(f"再平衡模式收益: {pf_rebalance.total_return():.2%}")
```

重要说明:
!!! note
    Percent模式的费用计算：
    系统会自动考虑手续费和滑点来计算实际可用的资金限制。
    但在现实中，滑点和手续费在下单前是未知的，这可能导致轻微的偏差。

应用策略:
- **固定股数策略**: 使用Amount模式，适合明确的买卖信号
- **固定金额策略**: 使用Value模式，适合定期定额投资
- **比例资金管理**: 使用Percent模式，适合风险控制和资金管理
- **仓位再平衡**: 使用Target*模式，适合多资产组合管理
- **动态调仓策略**: 结合不同模式实现复杂的资金分配逻辑

选择建议:
- 新手投资者：推荐Amount或Value模式，简单直观
- 专业交易者：使用Percent模式实现更精确的资金管理
- 组合管理：使用Target*模式进行动态再平衡
- 量化策略：根据策略逻辑灵活选择不同模式
"""


class DirectionT(tp.NamedTuple):
    """
    仓位方向类型定义
    
    定义投资组合中允许的仓位方向。这个设置直接影响策略的交易范围
    和风险特征，是策略设计中的重要约束条件。
    """
    LongOnly: int = 0   # 仅多头：只允许多头（买入持有）仓位
    ShortOnly: int = 1  # 仅空头：只允许空头（卖空）仓位  
    Both: int = 2       # 双向：允许多头和空头仓位


Direction = DirectionT()
"""仓位方向枚举实例，定义策略允许的交易方向约束"""

__pdoc__['Direction'] = f"""仓位方向枚举

```json
{to_doc(Direction)}
```

该枚举定义了投资组合中允许的仓位方向，是策略风险特征和交易范围的重要约束。
不同的方向设置适用于不同的市场环境和投资目标。

属性说明:
    LongOnly (0): 仅多头方向
        - 只允许买入持有（做多）操作
        - 不能进行卖空操作
        - 适用于传统的长期投资策略
        - 风险相对较低，适合保守投资者
        
    ShortOnly (1): 仅空头方向
        - 只允许卖空（做空）操作
        - 不能进行买入持有操作
        - 适用于熊市或特定的做空策略
        - 风险较高，需要专业知识和技能
        
    Both (2): 双向交易
        - 同时允许多头和空头操作
        - 可以根据市场条件灵活调整仓位方向
        - 适用于对冲策略和市场中性策略
        - 提供最大的策略灵活性

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import Direction

# 仅多头策略（传统投资）
pf_long_only = vbt.Portfolio.from_signals(
    close=price_data,
    entries=long_signals,
    exits=exit_signals,
    direction=Direction.LongOnly,  # 限制为仅多头
    size=100,
    init_cash=10000
)

# 仅空头策略（纯做空策略）
pf_short_only = vbt.Portfolio.from_signals(
    close=price_data,
    entries=short_signals,  # 注意：这里用short_entries参数
    exits=exit_signals,
    direction=Direction.ShortOnly,  # 限制为仅空头
    size=100,
    init_cash=10000
)

# 双向策略（多空都可以）
pf_both = vbt.Portfolio.from_signals(
    close=price_data,
    entries=long_signals,
    short_entries=short_signals,
    exits=long_exits,
    short_exits=short_exits,
    direction=Direction.Both,  # 允许双向交易
    size=100,
    init_cash=10000
)

print(f"仅多头策略收益: {pf_long_only.total_return():.2%}")
print(f"仅空头策略收益: {pf_short_only.total_return():.2%}")  
print(f"双向策略收益: {pf_both.total_return():.2%}")

# 分析不同方向的风险收益特征
print(f"\\n多头策略夏普比率: {pf_long_only.sharpe_ratio():.2f}")
print(f"空头策略夏普比率: {pf_short_only.sharpe_ratio():.2f}")
print(f"双向策略夏普比率: {pf_both.sharpe_ratio():.2f}")
```

策略应用:
- **牛市环境**: 使用LongOnly方向捕获上涨趋势
- **熊市环境**: 考虑ShortOnly方向或Both方向对冲风险
- **震荡市场**: 使用Both方向实现双向获利
- **风险管理**: 通过方向限制控制策略的风险暴露

与其他参数配合:
- 配合accumulate参数：控制仓位累积方向
- 配合conflict_mode：处理不同方向信号的冲突
- 配合止损设置：不同方向需要不同的风险控制策略

注意事项:
- ShortOnly策略需要充足的保证金和风险管理
- Both方向策略复杂度较高，需要更精细的信号设计
- 不同交易所和资产类别对做空有不同的限制
- 建议根据市场环境和个人风险承受能力选择合适的方向
"""


class OrderStatusT(tp.NamedTuple):
    """
    订单状态类型定义
    
    定义订单在投资组合模拟系统中的最终执行状态。每个订单都会有一个明确的状态，
    用于追踪和分析交易执行的结果。
    """
    Filled: int = 0     # 已成交：订单已成功执行
    Ignored: int = 1    # 已忽略：订单被系统忽略
    Rejected: int = 2   # 已拒绝：订单被系统拒绝执行


OrderStatus = OrderStatusT()
"""订单状态枚举实例，用于标识订单的最终执行状态"""

__pdoc__['OrderStatus'] = f"""订单状态枚举

```json
{to_doc(OrderStatus)}
```

该枚举定义了订单在投资组合模拟系统中的最终执行状态，是交易分析和调试的重要指标。
每个订单都会被归类到这三种状态之一。

属性说明:
    Filled (0): 已成交状态
        - 订单已成功执行并产生了实际的交易记录
        - 资金和仓位已发生相应变化
        - 这是正常交易的期望状态
        
    Ignored (1): 已忽略状态
        - 订单被系统主动忽略，通常是由于策略规则限制
        - 例如：信号冲突时选择忽略、累积模式下的重复信号等
        - 不会产生任何资金或仓位变化
        
    Rejected (2): 已拒绝状态
        - 订单因各种限制条件无法执行而被拒绝
        - 如资金不足、仓位不足、订单大小不符合要求等
        - 具体拒绝原因可通过OrderStatusInfo查看

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import OrderStatus

# 创建一个可能产生各种订单状态的投资组合
pf = vbt.Portfolio.from_orders(
    close=price_data,
    size=pd.Series([100, -200, 50]),  # 可能导致资金不足的订单
    init_cash=1000,  # 较少的初始资金
    log=True  # 启用日志记录
)

# 分析订单执行状态
orders = pf.orders
logs = pf.logs

# 统计不同状态的订单数量
filled_count = (orders.status == OrderStatus.Filled).sum()
ignored_count = (orders.status == OrderStatus.Ignored).sum() 
rejected_count = (orders.status == OrderStatus.Rejected).sum()

print(f"已成交订单: {filled_count}")
print(f"已忽略订单: {ignored_count}")
print(f"已拒绝订单: {rejected_count}")

# 计算订单成功率
total_orders = len(orders)
success_rate = filled_count / total_orders if total_orders > 0 else 0
print(f"订单成功率: {success_rate:.2%}")

# 分析被拒绝的订单
if rejected_count > 0:
    rejected_orders = orders[orders.status == OrderStatus.Rejected]
    print("\\n被拒绝的订单详情:")
    print(rejected_orders[['size', 'price', 'fees']])
```

策略分析意义:
- **成交率分析**: Filled状态的比例反映策略的可执行性
- **风险识别**: Rejected状态过多可能表明资金管理有问题
- **策略优化**: 通过分析不同状态的分布优化策略参数
- **系统调试**: 识别和解决策略执行中的问题

与其他系统组件的关系:
- **日志系统**: 所有状态变化都会记录在日志中
- **统计分析**: 状态分布是重要的策略评估指标
- **风险管理**: 拒绝状态可以作为风险预警信号
"""


class OrderSideT(tp.NamedTuple):
    """
    订单方向类型定义
    
    定义订单的交易方向，区分买入和卖出操作。这是最基础的交易属性，
    直接影响资金流向和仓位变化。
    """
    Buy: int = 0    # 买入：购买资产，增加仓位或减少空头仓位
    Sell: int = 1   # 卖出：出售资产，减少多头仓位或增加空头仓位


OrderSide = OrderSideT()
"""订单方向枚举实例，标识订单是买入还是卖出操作"""

__pdoc__['OrderSide'] = f"""订单方向枚举

```json
{to_doc(OrderSide)}
```

该枚举定义了订单的基本交易方向，是所有交易操作的基础分类。
每个订单都必须明确指定是买入还是卖出。

属性说明:
    Buy (0): 买入方向
        - 购买资产，使用现金换取资产
        - 对多头仓位：增加持仓数量
        - 对空头仓位：减少空头数量（买入平仓）
        - 资金流向：现金减少，资产增加
        
    Sell (1): 卖出方向  
        - 出售资产，使用资产换取现金
        - 对多头仓位：减少持仓数量（卖出平仓）
        - 对空头仓位：增加空头数量（卖空开仓）
        - 资金流向：资产减少，现金增加

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import OrderSide

# 创建包含买卖操作的投资组合
pf = vbt.Portfolio.from_orders(
    close=price_data,
    size=pd.Series([100, -50, 30]),  # 正数为买，负数为卖
    init_cash=10000
)

orders = pf.orders

# 分析买卖订单分布
buy_orders = orders[orders.side == OrderSide.Buy]
sell_orders = orders[orders.side == OrderSide.Sell]

print(f"买入订单数量: {len(buy_orders)}")
print(f"卖出订单数量: {len(sell_orders)}")

# 计算买卖金额统计
buy_volume = (buy_orders.size * buy_orders.price).sum()
sell_volume = (sell_orders.size * sell_orders.price).sum()

print(f"总买入金额: {buy_volume:.2f}")
print(f"总卖出金额: {sell_volume:.2f}")
print(f"净买入金额: {buy_volume - sell_volume:.2f}")

# 分析买卖订单的平均规模
avg_buy_size = buy_orders.size.mean() if len(buy_orders) > 0 else 0
avg_sell_size = sell_orders.size.mean() if len(sell_orders) > 0 else 0

print(f"平均买入规模: {avg_buy_size:.2f}")
print(f"平均卖出规模: {avg_sell_size:.2f}")
```

交易逻辑中的应用:
- **仓位管理**: 根据订单方向计算仓位变化
- **资金流分析**: 追踪买卖资金的流入流出
- **交易成本**: 不同方向可能有不同的手续费率
- **市场影响**: 大量买单可能推高价格，大量卖单可能压低价格

与其他组件的关系:
- **Size参数**: 正值通常对应Buy，负值对应Sell
- **Direction设置**: 限制允许的订单方向
- **手续费计算**: 某些市场买卖手续费不同
- **滑点影响**: 买卖订单可能有不同的滑点设置
"""


class OrderStatusInfoT(tp.NamedTuple):
    """
    订单状态详细信息类型定义
    
    提供订单被忽略或拒绝的具体原因，用于深度分析和调试。
    这些详细信息对于策略优化和问题诊断至关重要。
    """
    SizeNaN: int = 0           # 订单大小为NaN
    PriceNaN: int = 1          # 订单价格为NaN  
    ValPriceNaN: int = 2       # 估值价格为NaN
    ValueNaN: int = 3          # 价值为NaN
    ValueZeroNeg: int = 4      # 价值为零或负数
    SizeZero: int = 5          # 订单大小为零
    NoCashShort: int = 6       # 做空时现金不足
    NoCashLong: int = 7        # 做多时现金不足
    NoOpenPosition: int = 8    # 没有可平仓的持仓
    MaxSizeExceeded: int = 9   # 超过最大订单大小限制
    RandomEvent: int = 10      # 随机拒绝事件
    CantCoverFees: int = 11    # 无法支付手续费
    MinSizeNotReached: int = 12 # 未达到最小订单大小
    PartialFill: int = 13      # 部分成交


OrderStatusInfo = OrderStatusInfoT()
"""订单状态详细信息枚举实例，提供订单执行失败的具体原因"""

__pdoc__['OrderStatusInfo'] = f"""订单状态详细信息枚举

```json
{to_doc(OrderStatusInfo)}
```

该枚举提供了订单被忽略或拒绝的详细原因，是策略调试和优化的重要工具。
通过分析这些详细信息，可以识别策略中的问题并进行针对性改进。

属性说明:
    SizeNaN (0): 订单大小为NaN
        - 订单大小参数包含无效数值
        - 通常由计算错误或数据问题导致
        - 需要检查size参数的计算逻辑
        
    PriceNaN (1): 订单价格为NaN
        - 订单执行价格无效
        - 可能是价格数据缺失或计算错误
        - 需要检查价格数据的完整性
        
    ValPriceNaN (2): 估值价格为NaN
        - 资产估值价格无效
        - 影响价值计算和仓位评估
        - 需要检查估值数据源
        
    ValueNaN (3): 价值为NaN
        - 计算得出的订单价值无效
        - 通常是价格或数量计算错误的结果
        
    ValueZeroNeg (4): 价值为零或负数
        - 订单价值不合理（零或负数）
        - 可能是价格或数量设置错误
        
    SizeZero (5): 订单大小为零
        - 计算得出的订单大小为零
        - 通常发生在目标仓位等于当前仓位时
        
    NoCashShort (6): 做空时现金不足
        - 做空订单所需的保证金不足
        - 需要增加现金或减少订单大小
        
    NoCashLong (7): 做多时现金不足  
        - 买入订单所需的资金不足
        - 最常见的拒绝原因之一
        
    NoOpenPosition (8): 没有可平仓的持仓
        - 尝试卖出但没有相应的多头仓位
        - 或尝试买入平仓但没有空头仓位
        
    MaxSizeExceeded (9): 超过最大订单大小限制
        - 订单大小超过了预设的最大值
        - 用于风险控制和仓位管理
        
    RandomEvent (10): 随机拒绝事件
        - 模拟真实市场中的随机拒绝情况
        - 通过reject_prob参数控制
        
    CantCoverFees (11): 无法支付手续费
        - 剩余资金不足以支付交易手续费
        - 需要考虑手续费对小额交易的影响
        
    MinSizeNotReached (12): 未达到最小订单大小
        - 订单大小低于预设的最小值
        - 用于避免过小的无意义交易
        
    PartialFill (13): 部分成交
        - 订单只能部分执行
        - 通常发生在资金或仓位不足时

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import OrderStatus, OrderStatusInfo

# 创建一个资金紧张的投资组合来观察各种拒绝情况
pf = vbt.Portfolio.from_orders(
    close=price_data,
    size=pd.Series([1000, -2000, 500]),  # 可能导致资金不足
    init_cash=1000,  # 较少的初始资金
    fees=0.01,       # 1%手续费增加拒绝概率
    log=True         # 启用详细日志
)

# 分析订单拒绝原因
orders = pf.orders
logs = pf.logs

# 获取被拒绝的订单
rejected_orders = orders[orders.status == OrderStatus.Rejected]
rejected_logs = logs[logs.res_status == OrderStatus.Rejected]

if len(rejected_logs) > 0:
    # 统计拒绝原因分布
    rejection_reasons = rejected_logs.res_status_info.value_counts()
    print("订单拒绝原因统计:")
    for reason_code, count in rejection_reasons.items():
        reason_name = OrderStatusInfo._fields[reason_code]
        print(f"  {reason_name}: {count} 次")
    
    # 分析最常见的拒绝原因
    most_common_reason = rejection_reasons.index[0]
    print(f"\\n最常见拒绝原因: {OrderStatusInfo._fields[most_common_reason]}")
    
    # 查看具体的拒绝详情
    cash_shortage = rejected_logs[rejected_logs.res_status_info == OrderStatusInfo.NoCashLong]
    if len(cash_shortage) > 0:
        print(f"\\n资金不足情况: {len(cash_shortage)} 次")
        print("当时现金状况:", cash_shortage.cash.values)
        print("所需资金:", cash_shortage.req_size.values * cash_shortage.req_price.values)
```

策略优化指导:
- **NoCashLong/Short**: 优化资金管理，使用百分比大小类型
- **SizeZero**: 检查目标仓位计算逻辑
- **MaxSizeExceeded**: 调整订单大小限制参数
- **MinSizeNotReached**: 设置合理的最小订单大小
- **CantCoverFees**: 考虑手续费对小额交易的影响
- **PartialFill**: 启用allow_partial参数或调整订单大小

调试最佳实践:
1. 启用日志记录获取详细的执行信息
2. 定期统计各种拒绝原因的分布
3. 针对最常见的拒绝原因优化策略参数
4. 使用模拟环境测试极端情况下的策略表现
5. 建立监控系统追踪订单执行质量
"""

status_info_desc = [
    "订单大小为NaN",                    # Size is NaN
    "价格为NaN",                      # Price is NaN
    "资产估值价格为NaN",                # Asset valuation price is NaN
    "资产/组价值为NaN",                # Asset/group value is NaN
    "资产/组价值为零或负数",             # Asset/group value is zero or negative
    "订单大小为零",                    # Size is zero
    "做空资金不足",                    # Not enough cash to short
    "做多资金不足",                    # Not enough cash to long
    "没有可减少/关闭的持仓",            # No open position to reduce/close
    "订单大小超过最大允许值",           # Size is greater than maximum allowed
    "发生随机拒绝事件",               # Random event happened
    "资金不足以支付手续费",            # Not enough cash to cover fees
    "最终大小低于最小允许值",           # Final size is less than minimum allowed
    "最终大小小于请求大小"             # Final size is less than requested
]
"""订单状态详细描述列表，与OrderStatusInfo枚举值对应的可读性描述"""

__pdoc__['status_info_desc'] = f"""订单状态描述信息

```json
{to_doc(status_info_desc)}
```

该列表提供了与OrderStatusInfo枚举值对应的中文描述信息，用于提供更直观的订单状态说明。
每个索引位置对应OrderStatusInfo中相应的枚举值，便于在日志分析和调试时理解具体的拒绝或忽略原因。
"""


class TradeDirectionT(tp.NamedTuple):
    """
    交易方向类型定义
    
    定义交易记录中的方向属性，区分多头交易和空头交易。
    这与OrderSide不同，TradeDirection描述的是整个交易的方向特征。
    """
    Long: int = 0   # 多头交易：买入开仓到卖出平仓的完整交易
    Short: int = 1  # 空头交易：卖空开仓到买入平仓的完整交易


TradeDirection = TradeDirectionT()
"""交易方向枚举实例，标识完整交易的方向特征"""

__pdoc__['TradeDirection'] = f"""交易方向枚举

```json
{to_doc(TradeDirection)}
```

该枚举定义了交易记录的方向属性，描述完整交易的方向特征。
与OrderSide（订单方向）不同，TradeDirection描述的是从开仓到平仓的完整交易周期的方向。

属性说明:
    Long (0): 多头交易
        - 从买入开仓开始，到卖出平仓结束的完整交易
        - 盈利来源于价格上涨
        - 传统的"低买高卖"交易模式
        
    Short (1): 空头交易
        - 从卖空开仓开始，到买入平仓结束的完整交易
        - 盈利来源于价格下跌
        - "高卖低买"的交易模式

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import TradeDirection

# 创建包含多空交易的投资组合
pf = vbt.Portfolio.from_signals(
    close=price_data,
    entries=long_entries,
    short_entries=short_entries,
    exits=long_exits,
    short_exits=short_exits,
    direction=Direction.Both  # 允许双向交易
)

trades = pf.trades

# 分析多头和空头交易
long_trades = trades[trades.direction == TradeDirection.Long]
short_trades = trades[trades.direction == TradeDirection.Short]

print(f"多头交易数量: {len(long_trades)}")
print(f"空头交易数量: {len(short_trades)}")

# 计算不同方向的盈亏
long_pnl = long_trades.pnl.sum()
short_pnl = short_trades.pnl.sum()

print(f"多头交易总盈亏: {long_pnl:.2f}")
print(f"空头交易总盈亏: {short_pnl:.2f}")

# 分析胜率
long_win_rate = (long_trades.pnl > 0).mean()
short_win_rate = (short_trades.pnl > 0).mean()

print(f"多头交易胜率: {long_win_rate:.2%}")
print(f"空头交易胜率: {short_win_rate:.2%}")
```

应用场景:
- **交易分析**: 分别评估多头和空头策略的表现
- **风险管理**: 了解不同方向的风险暴露
- **策略优化**: 针对不同市场环境优化多空策略
- **绩效归因**: 分析盈亏来源于哪个交易方向
"""


class TradeStatusT(tp.NamedTuple):
    """
    交易状态类型定义
    
    定义交易记录的当前状态，区分正在进行的交易和已完成的交易。
    这对于分析未平仓盈亏和已实现盈亏非常重要。
    """
    Open: int = 0    # 开放状态：交易仍在进行中（未平仓）
    Closed: int = 1  # 关闭状态：交易已完成（已平仓）


TradeStatus = TradeStatusT()
"""交易状态枚举实例，标识交易的当前执行状态"""

__pdoc__['TradeStatus'] = f"""交易状态枚举

```json
{to_doc(TradeStatus)}
```

该枚举定义了交易记录的当前状态，区分正在进行的持仓交易和已经完成的历史交易。
这对于分析未实现盈亏和已实现盈亏具有重要意义。

属性说明:
    Open (0): 开放状态（未平仓）
        - 交易仍在进行中，持有未平仓的仓位
        - 盈亏为浮动盈亏，随市价变化
        - 交易的最终结果尚未确定
        
    Closed (1): 关闭状态（已平仓）
        - 交易已完全完成，所有仓位已平掉
        - 盈亏为已实现盈亏，结果确定
        - 可以计算准确的投资回报率

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import TradeStatus

# 创建投资组合并分析交易状态
pf = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entry_signals,
    exits=exit_signals,
    direction=Direction.Both
)

trades = pf.trades

# 分离开放和关闭的交易
open_trades = trades[trades.status == TradeStatus.Open]
closed_trades = trades[trades.status == TradeStatus.Closed]

print(f"未平仓交易数量: {len(open_trades)}")
print(f"已平仓交易数量: {len(closed_trades)}")

# 计算已实现和未实现盈亏
realized_pnl = closed_trades.pnl.sum() if len(closed_trades) > 0 else 0
unrealized_pnl = open_trades.pnl.sum() if len(open_trades) > 0 else 0

print(f"已实现盈亏: {realized_pnl:.2f}")
print(f"未实现盈亏: {unrealized_pnl:.2f}")
print(f"总盈亏: {realized_pnl + unrealized_pnl:.2f}")

# 分析交易持续时间
if len(closed_trades) > 0:
    avg_duration = closed_trades.duration.mean()
    print(f"平均交易持续期: {avg_duration:.1f} 个周期")

# 分析当前持仓情况
if len(open_trades) > 0:
    current_positions = open_trades.groupby('column').size()
    print(f"当前各资产持仓数: \\n{current_positions}")
```

投资分析意义:
- **风险评估**: 未平仓交易代表当前的风险暴露
- **绩效计算**: 只有已平仓交易的盈亏是确定的
- **资金管理**: 了解有多少资金被占用在未平仓交易中
- **策略评估**: 分析策略的持仓管理能力

注意事项:
- Open状态的交易盈亏会随着价格变动而变化
- 只有Closed状态的交易才能用于计算最终的投资回报
- 在回测结束时，所有Open状态的交易都应该被强制平仓
"""


class TradesTypeT(tp.NamedTuple):
    """
    交易记录类型定义
    
    定义不同类型的交易记录分析，用于从不同角度观察和分析交易行为。
    提供了多种交易分析视角，满足不同的分析需求。
    """
    EntryTrades: int = 0  # 入场交易：从入场角度分析的交易记录
    ExitTrades: int = 1   # 出场交易：从出场角度分析的交易记录
    Positions: int = 2    # 仓位记录：完整的仓位生命周期记录


TradesType = TradesTypeT()
"""交易记录类型枚举实例，定义不同的交易分析视角"""

__pdoc__['TradesType'] = f"""交易记录类型枚举

```json
{to_doc(TradesType)}
```

该枚举定义了不同类型的交易记录分析方式，每种类型从不同的角度观察和分析交易行为，
为投资者提供全面的交易分析视角。

属性说明:
    EntryTrades (0): 入场交易分析
        - 以入场信号为起点的交易分析
        - 关注入场时机、入场价格、入场后的表现
        - 适用于分析入场策略的有效性
        - 每个入场信号产生一个交易记录
        
    ExitTrades (1): 出场交易分析  
        - 以出场信号为终点的交易分析
        - 关注出场时机、出场价格、持仓期间的表现
        - 适用于分析出场策略的有效性
        - 每个出场信号产生一个交易记录
        
    Positions (2): 仓位生命周期分析
        - 完整的仓位从建立到清空的分析
        - 包含完整的买入-持有-卖出周期
        - 适用于分析整体仓位管理效果
        - 每个完整的仓位周期产生一个记录

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import TradesType

# 创建投资组合
pf = vbt.Portfolio.from_signals(
    close=price_data,
    entries=entry_signals,
    exits=exit_signals
)

# 获取不同类型的交易记录
entry_trades = pf.get_trades(TradesType.EntryTrades)
exit_trades = pf.get_trades(TradesType.ExitTrades) 
positions = pf.get_trades(TradesType.Positions)

# 比较不同分析角度的结果
print(f"入场交易记录数: {len(entry_trades)}")
print(f"出场交易记录数: {len(exit_trades)}")
print(f"完整仓位记录数: {len(positions)}")

# 分析入场策略效果
if len(entry_trades) > 0:
    entry_win_rate = (entry_trades.pnl > 0).mean()
    avg_entry_return = entry_trades.return_pct.mean()
    print(f"\\n入场策略胜率: {entry_win_rate:.2%}")
    print(f"入场策略平均收益率: {avg_entry_return:.2%}")

# 分析出场策略效果  
if len(exit_trades) > 0:
    exit_win_rate = (exit_trades.pnl > 0).mean()
    avg_exit_return = exit_trades.return_pct.mean()
    print(f"\\n出场策略胜率: {exit_win_rate:.2%}")
    print(f"出场策略平均收益率: {avg_exit_return:.2%}")

# 分析完整仓位管理效果
if len(positions) > 0:
    position_win_rate = (positions.pnl > 0).mean()
    avg_position_return = positions.return_pct.mean()
    avg_holding_period = positions.duration.mean()
    print(f"\\n仓位管理胜率: {position_win_rate:.2%}")
    print(f"仓位管理平均收益率: {avg_position_return:.2%}")
    print(f"平均持仓周期: {avg_holding_period:.1f}")
```

分析应用场景:
- **入场分析**: 使用EntryTrades评估信号识别能力
- **出场分析**: 使用ExitTrades评估风险控制能力  
- **整体评估**: 使用Positions评估完整的交易策略
- **策略对比**: 比较不同角度下的策略表现

选择建议:
- **策略开发阶段**: 使用EntryTrades优化入场逻辑
- **风控优化阶段**: 使用ExitTrades优化出场策略
- **综合评估阶段**: 使用Positions进行整体性能评估
- **细节分析**: 结合使用三种类型获得全面分析
"""


# ############# 命名元组数据结构 ############# #


class ProcessOrderState(tp.NamedTuple):
    """
    订单处理状态数据结构
    
    记录订单处理前后的完整投资组合状态信息。这是vectorbt模拟系统中的核心状态对象，
    包含了所有必要的财务和仓位信息，用于订单处理和状态跟踪。
    
    该数据结构在订单处理的各个阶段被使用：
    - 订单处理前：记录当前状态作为基准
    - 订单处理后：记录更新后的状态
    - 状态比较：计算订单执行的影响
    
    字段说明:
        cash: 当前列或现金共享组的现金余额
        position: 当前列的持仓数量
        debt: 当前列做空产生的债务金额
        free_cash: 当前列或现金共享组的可用现金
        val_price: 当前列的资产估值价格
        value: 当前列或现金共享组的总价值
        oidx: 订单记录索引
        lidx: 日志记录索引
    """
    cash: float      # 现金余额：当前列或现金共享组的现金总额
    position: float  # 持仓数量：当前列的资产持仓数量（正数多头，负数空头）
    debt: float      # 做空债务：当前列做空操作产生的债务总额
    free_cash: float # 可用现金：当前列或现金共享组的可用于交易的现金
    val_price: float # 估值价格：当前列资产的估值价格（用于价值计算）
    value: float     # 总价值：当前列或现金共享组的总价值（现金+持仓价值-债务）
    oidx: int        # 订单索引：对应的订单记录在order_records数组中的索引位置
    lidx: int        # 日志索引：对应的日志记录在log_records数组中的索引位置


__pdoc__['ProcessOrderState'] = """订单处理状态数据结构

该命名元组记录了订单处理前后的完整投资组合状态信息，是vectorbt模拟引擎的核心状态对象。
每个字段都代表着投资组合在特定时刻的关键财务指标。

使用场景:
- 订单执行前状态记录
- 订单执行后状态更新  
- 状态变化分析和审计
- 风险管理和监控

该结构确保了交易过程的完整性和可追溯性，是实现精确模拟的重要基础。
"""

__pdoc__['ProcessOrderState.cash'] = """现金余额

当前列或现金共享组的现金总额。

详细说明:
- 对于独立列：该列的专有现金余额
- 对于现金共享组：整个组共享的现金池总额
- 包含所有可用于投资的流动资金
- 会因买入（减少）和卖出（增加）而变化
- 不包括债务抵押的资金

注意事项:
- 现金余额可能为负数（表示借贷或保证金透支）
- 与free_cash的区别：cash是总现金，free_cash是可用现金
"""

__pdoc__['ProcessOrderState.position'] = """持仓数量

当前列的资产持仓数量。

详细说明:
- 正数：多头持仓数量（拥有的资产数量）
- 负数：空头持仓数量（借入卖出的资产数量）
- 零值：无持仓状态（现金状态）
- 单位取决于资产类型（股数、手数等）

计算影响:
- 影响组合总价值计算：position * val_price
- 影响资金占用和风险暴露
- 影响后续订单的可执行性
"""

__pdoc__['ProcessOrderState.debt'] = """做空债务

当前列做空操作产生的债务总额。

详细说明:
- 记录所有未偿还的做空债务价值
- 用于计算可用现金：free_cash = cash - debt
- 只有在允许做空的策略中才会产生
- 债务需要通过买入平仓来偿还

风险考量:
- 债务过高可能导致强制平仓
- 影响后续交易的资金可用性
- 需要考虑利息成本（如果模拟包含）
"""

__pdoc__['ProcessOrderState.free_cash'] = """可用现金

当前列或现金共享组的可用于交易的现金。

详细说明:
- 计算公式：cash - debt（现金减去债务）
- 代表真正可用于新投资的资金
- 永远不会超过初始现金水平（因为交易总有成本）
- 是订单可执行性检查的重要依据

应用场景:
- 订单大小验证
- 资金管理决策
- 风险控制检查
- 杠杆计算
"""

__pdoc__['ProcessOrderState.val_price'] = """估值价格

当前列资产的估值价格。

详细说明:
- 用于计算持仓价值：position * val_price
- 默认为当前K线的收盘价
- 可通过pre_segment_func_nb自定义覆盖
- 支持前向填充处理缺失值

重要用途:
- 支持SizeType.Value订单大小类型
- 支持SizeType.TargetValue和SizeType.TargetPercent
- 计算组合总价值的基础
- 风险价值（VaR）计算的价格基准

注意事项:
- 不能使用-np.inf或np.inf，只能是有限值
- NaN值可能导致整个组价值为NaN
"""

__pdoc__['ProcessOrderState.value'] = """总价值

当前列或现金共享组的总价值。

详细说明:
- 计算公式：cash + position * val_price
- 对于现金共享组：所有列的价值总和
- 反映投资组合的当前市值
- 用于计算收益率和风险指标

更新时机:
- 在pre_segment_func_nb之后更新
- 在post_segment_func_nb之前更新
- 如果update_value=True，在order_func_nb后也会更新

应用意义:
- 绩效评估的基础
- 仓位再平衡的依据
- 风险管理的重要指标
- 目标百分比订单的计算基准
"""

__pdoc__['ProcessOrderState.oidx'] = """订单记录索引

对应的订单记录在order_records数组中的索引位置。

详细说明:
- 指向SimulationContext.order_records数组
- 用于关联状态与具体的订单记录
- 便于订单追踪和调试分析
- -1表示尚未有订单记录

使用场景:
- 订单历史查询
- 执行路径跟踪
- 性能分析和调试
- 状态与订单的关联分析
"""

__pdoc__['ProcessOrderState.lidx'] = """日志记录索引

对应的日志记录在log_records数组中的索引位置。

详细说明:
- 指向SimulationContext.log_records数组
- 用于关联状态与详细的执行日志
- 便于深度调试和状态分析
- -1表示尚未有日志记录

应用价值:
- 详细的执行过程追踪
- 订单拒绝原因分析
- 状态变化历史记录
- 系统调试和优化支持
"""


class ExecuteOrderState(tp.NamedTuple):
    """
    订单执行状态数据结构
    
    记录订单执行完成后的核心状态信息。这是ProcessOrderState的简化版本，
    只包含订单执行后必须更新的关键财务状态字段。
    
    该结构用于：
    - 订单执行结果的状态更新
    - 执行前后状态的简化比较
    - 高频状态更新场景的性能优化
    - 核心状态信息的传递
    
    字段说明:
        cash: 执行后的现金余额
        position: 执行后的持仓数量
        debt: 执行后的做空债务
        free_cash: 执行后的可用现金
    """
    cash: float      # 现金余额：订单执行后的现金总额
    position: float  # 持仓数量：订单执行后的持仓数量
    debt: float      # 做空债务：订单执行后的债务总额
    free_cash: float # 可用现金：订单执行后的可用现金


__pdoc__['ExecuteOrderState'] = """订单执行状态数据结构

该命名元组记录了订单执行完成后的核心投资组合状态信息。它是ProcessOrderState的精简版本，
专注于订单执行直接影响的关键财务指标。

设计目的:
- 简化状态更新过程
- 提高执行效率
- 减少内存使用
- 突出核心变化

与ProcessOrderState的关系:
- 包含ProcessOrderState中最重要的4个字段
- 省略了估值、价值和索引信息
- 专用于订单执行结果的状态传递
- 可以转换为完整的ProcessOrderState

使用场景:
- 订单执行引擎的内部状态更新
- 高频交易模拟的性能优化
- 状态变化的增量更新
- 核心指标的快速访问
"""

__pdoc__['ExecuteOrderState.cash'] = """执行后现金余额

参见 `ProcessOrderState.cash` 的详细说明。

该字段记录订单执行完成后的现金余额状态，
反映了买入（现金减少）或卖出（现金增加）对现金的直接影响。
"""

__pdoc__['ExecuteOrderState.position'] = """执行后持仓数量

参见 `ProcessOrderState.position` 的详细说明。

该字段记录订单执行完成后的持仓数量状态，
反映了买入（持仓增加）或卖出（持仓减少）对仓位的直接影响。
"""

__pdoc__['ExecuteOrderState.debt'] = """执行后做空债务

参见 `ProcessOrderState.debt` 的详细说明。

该字段记录订单执行完成后的做空债务状态，
主要在空头交易中发生变化。
"""

__pdoc__['ExecuteOrderState.free_cash'] = """执行后可用现金

参见 `ProcessOrderState.free_cash` 的详细说明。

该字段记录订单执行完成后的可用现金状态，
是后续订单执行能力评估的重要依据。
"""


class SimulationContext(tp.NamedTuple):
    """
    模拟上下文数据结构
    
    这是vectorbt投资组合模拟系统的核心上下文对象，包含了整个模拟过程中
    所有必需的配置参数、状态信息和记录数组。该上下文在模拟的各个阶段
    被传递给不同的回调函数，提供了完整的模拟环境信息。
    
    设计理念：
    - **全局性**：包含所有模拟相关的全局信息
    - **不可变性**：作为命名元组，确保数据的不可变性
    - **完整性**：提供模拟执行所需的所有必要信息
    - **层次性**：为其他更具体的上下文提供基础信息
    
    主要用途：
    - 传递给 pre_sim_func_nb 和 post_sim_func_nb 函数
    - 为其他所有上下文类提供基础字段
    - 存储模拟的全局配置和状态
    - 管理订单记录和日志记录的存储
    """
    # 模拟配置参数
    target_shape: tp.Shape          # 目标形状：(行数, 列数) 
    group_lens: tp.Array1d          # 每组的列数数组
    init_cash: tp.Array1d           # 初始资金数组
    cash_sharing: bool              # 是否启用现金共享
    call_seq: tp.Optional[tp.Array2d] # 调用序列矩阵
    segment_mask: tp.ArrayLike      # 段执行掩码
    call_pre_segment: bool          # 是否调用段前函数
    call_post_segment: bool         # 是否调用段后函数
    close: tp.ArrayLike            # 收盘价数据
    ffill_val_price: bool          # 是否前向填充估值价格
    update_value: bool             # 是否在订单后更新价值
    fill_pos_record: bool          # 是否填充仓位记录
    flex_2d: bool                  # 是否使用灵活的二维索引
    
    # 记录存储数组
    order_records: tp.RecordArray   # 订单记录数组
    log_records: tp.RecordArray     # 日志记录数组
    
    # 最新状态数组
    last_cash: tp.Array1d           # 最新现金余额
    last_position: tp.Array1d       # 最新持仓数量
    last_debt: tp.Array1d           # 最新做空债务
    last_free_cash: tp.Array1d      # 最新可用现金
    last_val_price: tp.Array1d      # 最新估值价格
    last_value: tp.Array1d          # 最新组合价值
    second_last_value: tp.Array1d   # 次新组合价值
    last_return: tp.Array1d         # 最新收益率
    last_oidx: tp.Array1d           # 最新订单记录索引
    last_lidx: tp.Array1d           # 最新日志记录索引
    last_pos_record: tp.RecordArray # 最新仓位记录


__pdoc__['SimulationContext'] = """模拟上下文命名元组

该命名元组表示整个投资组合模拟的上下文环境，包含了所有其他上下文可用的通用信息。
它是vectorbt模拟系统的核心数据结构，定义了模拟执行的完整环境。

**上下文层次结构:**
```
SimulationContext (全局模拟环境)
├── GroupContext (组级上下文)  
├── RowContext (行级上下文)
└── SegmentContext (段级上下文)
    ├── OrderContext (订单上下文)
    ├── PostOrderContext (订单后上下文)
    └── FlexOrderContext (灵活订单上下文)
```

**传递目标:**
- `pre_sim_func_nb`: 模拟开始前的预处理函数
- `post_sim_func_nb`: 模拟完成后的后处理函数

**核心功能:**
1. **全局配置管理**: 存储模拟的所有配置参数
2. **状态追踪**: 维护所有资产和组的最新状态
3. **记录管理**: 管理订单记录和日志记录的存储
4. **上下文基础**: 为所有子上下文提供基础信息

使用示例:
```python
def pre_sim_func_nb(c: SimulationContext) -> None:
    '''模拟开始前的初始化函数'''
    print(f"开始模拟，目标形状: {c.target_shape}")
    print(f"资产组配置: {c.group_lens}")
    print(f"初始资金: {c.init_cash}")
    print(f"现金共享: {'启用' if c.cash_sharing else '禁用'}")

def post_sim_func_nb(c: SimulationContext) -> None:
    '''模拟完成后的清理函数'''
    print(f"模拟完成，总订单数: {c.last_oidx.max() + 1}")
    print(f"最终现金: {c.last_cash}")
    print(f"最终持仓: {c.last_position}")
    print(f"最终价值: {c.last_value}")

# 在投资组合中使用
pf = vbt.Portfolio.from_order_func(
    close=price_data,
    order_func_nb=my_order_func,
    pre_sim_func_nb=pre_sim_func_nb,
    post_sim_func_nb=post_sim_func_nb
)
```

**设计原则:**
- **不可变性**: 作为命名元组，确保上下文数据的完整性
- **完整性**: 包含模拟所需的所有必要信息
- **层次性**: 为不同层级的上下文提供统一的基础
- **高效性**: 优化的数据结构，支持高性能计算

**注意事项:**
- 上下文对象是只读的，不能直接修改其字段
- 状态更新通过模拟引擎自动完成
- 记录数组在模拟过程中动态填充
- 所有数组字段都经过优化，支持向量化操作
"""
__pdoc__['SimulationContext.target_shape'] = """模拟的目标形状

包含恰好两个元素的元组：行数和列数。

**结构说明:**
- 第一个元素：时间步数（行数）
- 第二个元素：资产数量（列数）

**应用意义:**
- 定义了整个模拟空间的维度
- 决定了价格数据和其他输入的预期形状
- 影响内存分配和计算效率

使用示例:
```python
# 一天的分钟数据，三个资产
target_shape = (1440, 3)  # 1440分钟 × 3个资产

# 100天的日数据，5个资产  
target_shape = (100, 5)   # 100天 × 5个资产

# 在回调函数中使用
def order_func_nb(c):
    rows, cols = c.target_shape
    print(f"模拟空间: {rows}个时间步 × {cols}个资产")
```

**注意事项:**
- 第一轴是时间（行），第二轴是资产（列）
- 所有输入数据都应该与此形状兼容
- 形状确定后不能在模拟过程中更改
"""

__pdoc__['SimulationContext.group_lens'] = """每组中的列数

数组，指定每个资产组包含的列数。即使列没有分组，也包含1（每组一列）。

**分组逻辑:**
- 未分组：每个资产独立，`group_lens = [1, 1, 1, ...]`
- 分组：相关资产归为一组，如配对交易 `group_lens = [2]`
- 混合：部分分组，如 `group_lens = [2, 1, 3]`

**与现金共享的关系:**
- 同组内的资产可以共享现金池
- 不同组之间的资产资金独立
- 影响资金管理和风险控制策略

使用示例:
```python
# 配对交易：两个资产为一组
group_lens = np.array([2])

# 三个独立资产
group_lens = np.array([1, 1, 1])

# 混合分组：一个配对 + 两个独立资产
group_lens = np.array([2, 1, 1])

# 在回调函数中访问
def pre_group_func_nb(c):
    current_group_size = c.group_lens[c.group]
    print(f"当前组大小: {current_group_size}")
```

**应用场景:**
- **配对交易**: 两个相关资产组成一组
- **行业轮动**: 同行业资产组成一组
- **多策略**: 不同策略的资产分别分组
- **风险管理**: 按风险级别分组管理
"""

__pdoc__['SimulationContext.init_cash'] = """每列或现金共享组的初始资金

根据是否启用现金共享，数组形状有所不同。

**形状规则:**
- 启用现金共享：形状为 `(group_lens.shape[0],)` - 每组一个值
- 禁用现金共享：形状为 `(target_shape[1],)` - 每列一个值

**资金分配逻辑:**
- 现金共享组：组内所有资产共用一个资金池
- 独立列：每个资产有独立的资金账户

使用示例:
```python
# 场景1：三列，每列$100，现金共享组(2列) + 独立组(1列)
# 启用现金共享时
init_cash = np.array([200, 100])  # 第一组$200，第二组$100

# 禁用现金共享时  
init_cash = np.array([100, 100, 100])  # 每列$100

# 场景2：不同资金配置
init_cash = np.array([10000, 5000, 15000])  # 不同资产不同资金

# 在回调函数中使用
def order_func_nb(c):
    if c.cash_sharing:
        available_cash = c.cash_now  # 组共享资金
    else:
        available_cash = c.cash_now  # 列独立资金
```

**策略意义:**
- **风险平衡**: 不同资产配置不同资金
- **策略权重**: 通过资金分配体现策略重要性
- **资金管理**: 灵活的资金分配和使用策略
"""

__pdoc__['SimulationContext.cash_sharing'] = """是否启用现金共享

控制同组内资产是否可以共享现金池的布尔标志。

**启用现金共享 (True):**
- 同组资产共用一个现金账户
- 一个资产的交易可以使用组内其他资产的现金
- 提高资金使用效率，适合相关性高的资产

**禁用现金共享 (False):**  
- 每个资产有独立的现金账户
- 资产间资金隔离，降低相互影响
- 更严格的风险控制，适合独立策略

使用示例:
```python
# 配对交易策略 - 启用现金共享
pf_pairs = vbt.Portfolio.from_signals(
    close=pair_prices,
    entries=pair_entries, 
    exits=pair_exits,
    group_by=[0, 0],  # 两个资产为一组
    cash_sharing=True  # 启用现金共享
)

# 多策略组合 - 禁用现金共享
pf_multi = vbt.Portfolio.from_signals(
    close=multi_prices,
    entries=multi_entries,
    exits=multi_exits, 
    cash_sharing=False  # 每个资产独立资金
)

# 在回调函数中检查
def order_func_nb(c):
    if c.cash_sharing:
        print(f"组共享现金: {c.cash_now}")
    else:
        print(f"列独立现金: {c.cash_now}")
```

**应用场景:**
- **启用共享**: 配对交易、套利策略、相关资产组合
- **禁用共享**: 独立策略、风险隔离、多策略组合

**风险考虑:**
- 共享现金可能导致风险集中
- 独立现金可能降低资金使用效率
- 需根据策略特点和风险偏好选择
"""
__pdoc__['SimulationContext.call_seq'] = """Default sequence of calls per segment.

Controls the sequence in which `order_func_nb` is executed within each segment.

Has shape `SimulationContext.target_shape` and each value must exist in the range `[0, group_len)`.

!!! note
    To use `sort_call_seq_nb`, should be generated using `CallSeqType.Default`.

    To change the call sequence dynamically, better change `SegmentContext.call_seq_now` in-place.
    
Example:
    The default call sequence for three data points and two groups with three columns each:
    
    ```python
    np.array([
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2],
        [0, 1, 2, 0, 1, 2]
    ])
    ```
"""
__pdoc__['SimulationContext.segment_mask'] = """Mask of whether a particular segment should be executed.

A segment is simply a sequence of `order_func_nb` calls under the same group and row.

If a segment is inactive, any callback function inside of it will not be executed.
You can still execute the segment's pre- and postprocessing function by enabling 
`SimulationContext.call_pre_segment` and `SimulationContext.call_post_segment` respectively.

Utilizes flexible indexing using `vectorbt.base.reshape_fns.flex_select_auto_nb` and `flex_2d`, 
so it can be passed as 

* 2-dim array, 
* 1-dim array per column (requires `flex_2d=True`), 
* 1-dim array per row (requires `flex_2d=False`), and
* a scalar. 

Broadcasts to the shape `(target_shape[0], group_lens.shape[0])`.

!!! note
    To modify the array in place, make sure to build an array of the full shape.

Example:
    Consider two groups with two columns each and the following activity mask:
    
    ```python
    np.array([[ True, False], 
              [False,  True]])
    ```
    
    The first group is only executed in the first row and the second group is only executed in the second row.
"""
__pdoc__['SimulationContext.call_pre_segment'] = """Whether to call `pre_segment_func_nb` regardless of 
`SimulationContext.segment_mask`."""
__pdoc__['SimulationContext.call_post_segment'] = """Whether to call `post_segment_func_nb` regardless of 
`SimulationContext.segment_mask`.

Allows, for example, to write user-defined arrays such as returns at the end of each segment."""
__pdoc__['SimulationContext.close'] = """Latest asset price at each time step.

Utilizes flexible indexing using `vectorbt.base.reshape_fns.flex_select_auto_nb` and `flex_2d`, 
so it can be passed as 

* 2-dim array, 
* 1-dim array per column (requires `flex_2d=True`), 
* 1-dim array per row (requires `flex_2d=False`), and
* a scalar. 

Broadcasts to the shape `SimulationContext.target_shape`.

!!! note
    To modify the array in place, make sure to build an array of the full shape.
"""
__pdoc__['SimulationContext.ffill_val_price'] = """Whether to track valuation price only if it's known.

Otherwise, unknown `SimulationContext.close` will lead to NaN in valuation price at the next timestamp."""
__pdoc__['SimulationContext.update_value'] = """Whether to update group value after each filled order.

Otherwise, stays the same for all columns in the group (the value is calculated
only once, before executing any order).

The change is marginal and mostly driven by transaction costs and slippage."""
__pdoc__['SimulationContext.fill_pos_record'] = """Whether to fill position record.

Disable this to make simulation a bit faster for simple use cases."""
__pdoc__['SimulationContext.flex_2d'] = """Whether the elements in a 1-dim array should be treated per
column rather than per row.

This flag is set automatically when using `vectorbt.portfolio.base.Portfolio.from_order_func` depending upon 
whether there is any argument that has been broadcast to 2 dimensions.

Has only effect when using flexible indexing, for example, with `vectorbt.base.reshape_fns.flex_select_auto_nb`.
"""
__pdoc__['SimulationContext.order_records'] = """Order records.

It's a 1-dimensional array with records of type `order_dt`.

The array is initialized with empty records first (they contain random data), and then 
gradually filled with order data. The number of initialized records depends upon `max_orders`, 
but usually it's `target_shape[0] * target_shape[1]`, meaning there is maximal one order record per element.
`max_orders` can be chosen lower if not every `order_func_nb` leads to a filled order, to save memory.

You can use `SimulationContext.last_oidx` to get the index of the latest filled order of each column.

Example:
    Before filling, each order record looks like this:
    
    ```python
    np.array([(-8070450532247928832, -8070450532247928832, 4, 0., 0., 0., 5764616306889786413)]
    ```
    
    After filling, it becomes like this:
    
    ```python
    np.array([(0, 0, 1, 50., 1., 0., 1)]
    ```
"""
__pdoc__['SimulationContext.log_records'] = """Log records.

Similar to `SimulationContext.order_records` but of type `log_dt` and index `SimulationContext.last_lidx`."""
__pdoc__['SimulationContext.last_cash'] = """Latest cash per column or group with cash sharing.

Has the same shape as `SimulationContext.init_cash`.

Gets updated right after `order_func_nb`.
"""
__pdoc__['SimulationContext.last_position'] = """Latest position per column.

Has shape `(target_shape[1],)`.

Gets updated right after `order_func_nb`.
"""
__pdoc__['SimulationContext.last_debt'] = """Latest debt from shorting per column.

Debt is the total value from shorting that hasn't been covered yet. Used to update `OrderContext.free_cash_now`.

Has shape `(target_shape[1],)`. 

Gets updated right after `order_func_nb`.
"""
__pdoc__['SimulationContext.last_free_cash'] = """Latest free cash per column or group with cash sharing.

Free cash never goes above the initial level, because an operation always costs money.

Has shape `(target_shape[1],)`. 

Gets updated right after `order_func_nb`.
"""
__pdoc__['SimulationContext.last_val_price'] = """Latest valuation price per column.

Has shape `(target_shape[1],)`.

Enables `SizeType.Value`, `SizeType.TargetValue`, and `SizeType.TargetPercent`.

Gets multiplied by the current position to get the value of the column (see `SimulationContext.last_value`).

Defaults to the `SimulationContext.close` before `post_segment_func_nb`.
If `SimulationContext.ffill_val_price`, gets updated only if `SimulationContext.close` is not NaN.
For example, close of `[1, 2, np.nan, np.nan, 5]` yields valuation price of `[1, 2, 2, 2, 5]`.

Also gets updated right after `pre_segment_func_nb` - you can use `pre_segment_func_nb` to
override `last_val_price` in-place, such that `order_func_nb` can use the new group value. 
You are not allowed to use `-np.inf` or `np.inf` - only finite values.
If `SimulationContext.update_value`, gets also updated right after `order_func_nb` using 
filled order price as the latest known price.

!!! note
    Since the previous `SimulationContext.close` is NaN in the first row, the first `last_val_price` is also NaN.
    
    Overriding `last_val_price` with NaN won't apply `SimulationContext.ffill_val_price`,
    so your entire group will become NaN.

Example:
    Consider 10 units in column 1 and 20 units in column 2. The previous close of them is
    $40 and $50 respectively, which is also the default valuation price in the current row,
    available as `last_val_price` in `pre_segment_func_nb`. If both columns are in the same group 
    with cash sharing, the group is valued at $1400 before any `order_func_nb` is called, and can 
    be later accessed via `OrderContext.value_now`.
"""
__pdoc__['SimulationContext.last_value'] = """Latest value per column or group with cash sharing.

Has the same shape as `SimulationContext.init_cash`.

Calculated by multiplying valuation price by the current position.
The value of each column in a group with cash sharing is summed to get the value of the entire group.

Gets updated using `SimulationContext.last_val_price` after `pre_segment_func_nb` and 
before `post_segment_func_nb`. If `SimulationContext.update_value`, gets also updated right after 
`order_func_nb` using filled order price as the latest known price (the difference will be minimal, 
only affected by costs).
"""
__pdoc__['SimulationContext.second_last_value'] = """Second-latest value per column or group with cash sharing.

Has the same shape as `SimulationContext.last_value`.

Contains the latest known value two rows before (`i - 2`) to be compared either with the latest known value 
one row before (`i - 1`) or now (`i`).

Gets updated at the end of each segment/row. 
"""
__pdoc__['SimulationContext.last_return'] = """Latest return per column or group with cash sharing.

Has the same shape as `SimulationContext.last_value`.

Calculated by comparing `SimulationContext.last_value` to `SimulationContext.second_last_value`.

Gets updated each time `SimulationContext.last_value` is updated.
"""
__pdoc__['SimulationContext.last_oidx'] = """Index of the latest order record of each column.

Points to `SimulationContext.order_records` and has shape `(target_shape[1],)`.

Example:
    `last_oidx` of `np.array([1, 100, -1])` means the latest filled order is `order_records[1]` for the
    first column, `order_records[100]` for the second column, and no orders have been filled yet
    for the third column.
"""
__pdoc__['SimulationContext.last_lidx'] = """Index of the latest log record of each column.

Similar to `SimulationContext.last_oidx` but for log records.
"""
__pdoc__['SimulationContext.last_pos_record'] = """Latest position record of each column.

It's a 1-dimensional array with records of type `trade_dt`.

Has shape `(target_shape[1],)`.

The array is initialized with empty records first (they contain random data)
and the field `id` is set to -1. Once the first position is entered in a column,
the `id` becomes 0 and the record materializes. Once the position is closed, the record
fixes its identifier and other data until the next position is entered. 

The fields `entry_price` and `exit_price` are average entry and exit price respectively.
The fields `pnl` and `return` contain statistics as if the position has been closed and are 
re-calculated using `SimulationContext.last_val_price` after `pre_segment_func_nb` 
(in case `SimulationContext.last_val_price` has been overridden) and before `post_segment_func_nb`.

!!! note
    In an open position record, the field `exit_price` doesn't reflect the latest valuation price,
    but keeps the average price at which the position has been reduced.

The position record is updated after successfully filling an order (after `order_func_nb` and
before `post_order_func_nb`).

Example:
    Consider a simulation that orders `order_size` for `order_price` and $1 fixed fees.
    Here's order info from `order_func_nb` and the updated position info from `post_order_func_nb`:
    
    ```plaintext
        order_size  order_price  id  col  size  entry_idx  entry_price  \\
    0          NaN            1  -1    0   1.0         13    14.000000   
    1          0.5            2   0    0   0.5          1     2.000000   
    2          1.0            3   0    0   1.5          1     2.666667   
    3          NaN            4   0    0   1.5          1     2.666667   
    4         -1.0            5   0    0   1.5          1     2.666667   
    5         -0.5            6   0    0   1.5          1     2.666667   
    6          NaN            7   0    0   1.5          1     2.666667   
    7         -0.5            8   1    0   0.5          7     8.000000   
    8         -1.0            9   1    0   1.5          7     8.666667   
    9          1.0           10   1    0   1.5          7     8.666667   
    10         0.5           11   1    0   1.5          7     8.666667   
    11         1.0           12   2    0   1.0         11    12.000000   
    12        -2.0           13   3    0   1.0         12    13.000000   
    13         2.0           14   4    0   1.0         13    14.000000   
    
        entry_fees  exit_idx  exit_price  exit_fees   pnl    return  direction  status
    0          0.5        -1         NaN        0.0 -0.50 -0.035714          0       0
    1          1.0        -1         NaN        0.0 -1.00 -1.000000          0       0
    2          2.0        -1         NaN        0.0 -1.50 -0.375000          0       0
    3          2.0        -1         NaN        0.0 -0.75 -0.187500          0       0
    4          2.0        -1    5.000000        1.0  0.50  0.125000          0       0
    5          2.0         5    5.333333        2.0  0.00  0.000000          0       1
    6          2.0         5    5.333333        2.0  0.00  0.000000          0       1
    7          1.0        -1         NaN        0.0 -1.00 -0.250000          1       0
    8          2.0        -1         NaN        0.0 -2.50 -0.192308          1       0
    9          2.0        -1   10.000000        1.0 -5.00 -0.384615          1       0
    10         2.0        10   10.333333        2.0 -6.50 -0.500000          1       1
    11         1.0        -1         NaN        0.0 -1.00 -0.083333          0       0
    12         0.5        -1         NaN        0.0 -0.50 -0.038462          1       0
    13         0.5        -1         NaN        0.0 -0.50 -0.035714          0       0
    ```
"""


class GroupContext(tp.NamedTuple):
    """
    资产组上下文数据结构
    
    表示当前处理的资产组的上下文信息。资产组是一组相关的列（资产），
    它们可能共享现金或具有其他关联关系。该上下文包含了SimulationContext
    的所有字段，并添加了描述当前组的特定信息。
    
    设计用途：
    - 传递给 pre_group_func_nb 和 post_group_func_nb 函数
    - 为组级别的操作提供上下文信息
    - 支持组级别的资金管理和策略决策
    - 管理组内资产的协调和同步
    
    组的概念：
    - 组是一组相关列的集合（例如，通过共享资本相关）
    - 在每一行中，同一组下的列被绑定到同一个段
    - 组可以实现资产间的协调交易和风险管理
    """
    # 继承自SimulationContext的所有字段
    target_shape: tp.Shape          # 模拟目标形状
    group_lens: tp.Array1d          # 每组列数
    init_cash: tp.Array1d           # 初始资金
    cash_sharing: bool              # 现金共享标志
    call_seq: tp.Optional[tp.Array2d] # 调用序列
    segment_mask: tp.ArrayLike      # 段掩码
    call_pre_segment: bool          # 调用段前函数标志
    call_post_segment: bool         # 调用段后函数标志
    close: tp.ArrayLike            # 收盘价数据
    ffill_val_price: bool          # 前向填充估值价格标志
    update_value: bool             # 更新价值标志
    fill_pos_record: bool          # 填充仓位记录标志
    flex_2d: bool                  # 灵活二维索引标志
    order_records: tp.RecordArray   # 订单记录数组
    log_records: tp.RecordArray     # 日志记录数组
    last_cash: tp.Array1d           # 最新现金状态
    last_position: tp.Array1d       # 最新仓位状态
    last_debt: tp.Array1d           # 最新债务状态
    last_free_cash: tp.Array1d      # 最新可用现金
    last_val_price: tp.Array1d      # 最新估值价格
    last_value: tp.Array1d          # 最新组合价值
    second_last_value: tp.Array1d   # 次新组合价值
    last_return: tp.Array1d         # 最新收益率
    last_oidx: tp.Array1d           # 最新订单索引
    last_lidx: tp.Array1d           # 最新日志索引
    last_pos_record: tp.RecordArray # 最新仓位记录
    
    # GroupContext特有字段
    group: int        # 当前组的索引
    group_len: int    # 当前组中的列数
    from_col: int     # 当前组第一列的索引
    to_col: int       # 当前组最后一列的索引+1


__pdoc__['GroupContext'] = """资产组上下文命名元组

该命名元组表示一个资产组的上下文环境。资产组是一组相关的列（资产），
它们通过某种方式关联（例如，共享相同的资本）。在每一行中，同一组下的列被绑定到同一个段。

**核心概念:**
- **资产组**: 相关资产的逻辑分组，支持协调交易和风险管理
- **组内协调**: 同组资产可以共享资金、信息和交易决策
- **段绑定**: 每行中同组的所有列形成一个处理段

**包含字段:**
- 继承SimulationContext的所有字段
- 添加描述当前组的特定字段

**传递目标:**
- `pre_group_func_nb`: 组处理前的预处理函数
- `post_group_func_nb`: 组处理后的后处理函数

**组配置示例:**
考虑一个包含6个资产的投资组合，分为三组：

| group | group_len | from_col | to_col | 描述 |
| ----- | --------- | -------- | ------ | ---- |
| 0     | 3         | 0        | 3      | 第一组：3个资产（列0-2） |
| 1     | 2         | 3        | 5      | 第二组：2个资产（列3-4） |
| 2     | 1         | 5        | 6      | 第三组：1个资产（列5）   |

使用示例:
```python
def pre_group_func_nb(c: GroupContext) -> None:
    '''组处理前的准备函数'''
    print(f"处理第{c.group}组")
    print(f"组大小: {c.group_len}")
    print(f"列范围: [{c.from_col}, {c.to_col})")
    
    # 组级别的资金管理
    if c.cash_sharing:
        group_cash = c.last_cash[c.group]
        print(f"组共享现金: {group_cash}")
    
    # 组内资产状态分析
    group_positions = c.last_position[c.from_col:c.to_col]
    group_values = c.last_val_price[c.from_col:c.to_col] * group_positions
    print(f"组内持仓价值: {group_values}")

def post_group_func_nb(c: GroupContext) -> None:
    '''组处理后的清理函数'''
    # 组级别的风险检查
    group_exposure = sum(abs(c.last_position[c.from_col:c.to_col]))
    print(f"组风险暴露: {group_exposure}")

# 在投资组合中使用
pf = vbt.Portfolio.from_order_func(
    close=price_data,
    order_func_nb=my_order_func,
    pre_group_func_nb=pre_group_func_nb,
    post_group_func_nb=post_group_func_nb,
    group_by=[0, 0, 1, 1, 2]  # 分组配置
)
```

**应用场景:**
- **配对交易**: 两个相关资产为一组，实现配对策略
- **行业轮动**: 同行业资产分组，实现行业级别的资金配置
- **多策略组合**: 不同策略的资产分组，实现策略隔离
- **风险管理**: 按风险级别分组，实现分层风险控制

**组级别操作:**
- 组内资金重新分配
- 组级别的风险监控
- 组内资产的协调交易
- 组级别的绩效评估
"""
for field in GroupContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['GroupContext.' + field] = f"See `SimulationContext.{field}`."
__pdoc__['GroupContext.group'] = """Index of the current group.

Has range `[0, group_lens.shape[0])`.
"""
__pdoc__['GroupContext.group_len'] = """Number of columns in the current group.

Scalar value. Same as `group_lens[group]`.
"""
__pdoc__['GroupContext.from_col'] = """Index of the first column in the current group.

Has range `[0, target_shape[1])`.
"""
__pdoc__['GroupContext.to_col'] = """Index of the last column in the current group plus one.

Has range `[1, target_shape[1] + 1)`. 

If columns are not grouped, equals to `from_col + 1`.

!!! warning
    In the last group, `to_col` points at a column that doesn't exist.
"""


class RowContext(tp.NamedTuple):
    """
    行上下文数据结构
    
    表示当前时间步（行）的上下文信息。行是执行段的时间步，
    包含了SimulationContext的所有字段，并添加了当前行的特定信息。
    
    设计用途：
    - 传递给 pre_row_func_nb 和 post_row_func_nb 函数
    - 为行级别（时间步级别）的操作提供上下文
    - 支持时间序列相关的分析和处理
    - 管理每个时间点的全局状态
    """
    # 继承自SimulationContext的所有字段
    target_shape: tp.Shape          # 模拟目标形状
    group_lens: tp.Array1d          # 每组列数
    init_cash: tp.Array1d           # 初始资金
    cash_sharing: bool              # 现金共享标志
    call_seq: tp.Optional[tp.Array2d] # 调用序列
    segment_mask: tp.ArrayLike      # 段掩码
    call_pre_segment: bool          # 调用段前函数标志
    call_post_segment: bool         # 调用段后函数标志
    close: tp.ArrayLike            # 收盘价数据
    ffill_val_price: bool          # 前向填充估值价格标志
    update_value: bool             # 更新价值标志
    fill_pos_record: bool          # 填充仓位记录标志
    flex_2d: bool                  # 灵活二维索引标志
    order_records: tp.RecordArray   # 订单记录数组
    log_records: tp.RecordArray     # 日志记录数组
    last_cash: tp.Array1d           # 最新现金状态
    last_position: tp.Array1d       # 最新仓位状态
    last_debt: tp.Array1d           # 最新债务状态
    last_free_cash: tp.Array1d      # 最新可用现金
    last_val_price: tp.Array1d      # 最新估值价格
    last_value: tp.Array1d          # 最新组合价值
    second_last_value: tp.Array1d   # 次新组合价值
    last_return: tp.Array1d         # 最新收益率
    last_oidx: tp.Array1d           # 最新订单索引
    last_lidx: tp.Array1d           # 最新日志索引
    last_pos_record: tp.RecordArray # 最新仓位记录
    
    # RowContext特有字段
    i: int                          # 当前行（时间步）索引


__pdoc__['RowContext'] = """行上下文命名元组

该命名元组表示一行（时间步）的上下文环境。行是执行段的时间单位，
在每个时间步中会执行相应的段操作。

**核心概念:**
- **时间步**: 每一行代表一个时间点（如一天、一小时、一分钟等）
- **段执行**: 在每个时间步中，所有活跃的段都会被执行
- **全局状态**: 维护整个投资组合在当前时间点的状态

**包含字段:**
- 继承SimulationContext的所有字段
- 添加描述当前行的特定字段

**传递目标:**
- `pre_row_func_nb`: 行处理前的预处理函数
- `post_row_func_nb`: 行处理后的后处理函数

使用示例:
```python
def pre_row_func_nb(c: RowContext) -> None:
    '''每个时间步开始前的处理'''
    current_time = c.i
    print(f"处理时间步 {current_time}")
    
    # 时间相关的逻辑
    if current_time == 0:
        print("模拟开始")
    elif current_time == c.target_shape[0] - 1:
        print("即将结束模拟")
    
    # 全局市场状态分析
    current_prices = c.close[c.i]  # 当前时间步的所有资产价格
    print(f"当前市场价格: {current_prices}")

def post_row_func_nb(c: RowContext) -> None:
    '''每个时间步结束后的处理'''
    # 计算当前时间步的组合表现
    total_value = c.last_value.sum()
    print(f"时间步 {c.i} 结束，总价值: {total_value}")
    
    # 记录时间序列数据
    if hasattr(c, 'custom_metrics'):
        c.custom_metrics[c.i] = total_value

# 在投资组合中使用
pf = vbt.Portfolio.from_order_func(
    close=price_data,
    order_func_nb=my_order_func,
    pre_row_func_nb=pre_row_func_nb,
    post_row_func_nb=post_row_func_nb
)
```

**应用场景:**
- **时间序列分析**: 记录和分析每个时间点的状态变化
- **市场状态监控**: 监控整体市场状况和组合表现
- **时间相关策略**: 实现基于时间的策略逻辑
- **数据记录**: 记录自定义的时间序列指标

**时间步处理流程:**
1. 调用 `pre_row_func_nb` - 行前处理
2. 遍历所有活跃组，执行段操作
3. 调用 `post_row_func_nb` - 行后处理
4. 更新全局状态，进入下一时间步
"""
for field in RowContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['RowContext.' + field] = f"See `SimulationContext.{field}`."
__pdoc__['RowContext.i'] = """Index of the current row.

Has range `[0, target_shape[0])`.
"""


class SegmentContext(tp.NamedTuple):
    """
    段上下文数据结构
    
    表示一个段的上下文信息。段是组和行的交集，定义了在同一组和行内
    元素的处理方式和顺序。包含了多个上下文的所有字段，并添加了
    描述当前段的特定字段。
    
    设计用途：
    - 传递给 pre_segment_func_nb 和 post_segment_func_nb 函数
    - 为段级别的操作提供上下文信息
    - 控制组内资产的处理顺序和协调
    - 管理段级别的状态和配置
    
    段的概念：
    - 段是组和行的交集实体
    - 定义了同一组和行内元素的处理方式和顺序
    - 是订单执行的基本单位
    """
    # 继承自多个上下文的字段
    target_shape: tp.Shape          # 模拟目标形状
    group_lens: tp.Array1d          # 每组列数
    init_cash: tp.Array1d           # 初始资金
    cash_sharing: bool              # 现金共享标志
    call_seq: tp.Optional[tp.Array2d] # 调用序列
    segment_mask: tp.ArrayLike      # 段掩码
    call_pre_segment: bool          # 调用段前函数标志
    call_post_segment: bool         # 调用段后函数标志
    close: tp.ArrayLike            # 收盘价数据
    ffill_val_price: bool          # 前向填充估值价格标志
    update_value: bool             # 更新价值标志
    fill_pos_record: bool          # 填充仓位记录标志
    flex_2d: bool                  # 灵活二维索引标志
    order_records: tp.RecordArray   # 订单记录数组
    log_records: tp.RecordArray     # 日志记录数组
    last_cash: tp.Array1d           # 最新现金状态
    last_position: tp.Array1d       # 最新仓位状态
    last_debt: tp.Array1d           # 最新债务状态
    last_free_cash: tp.Array1d      # 最新可用现金
    last_val_price: tp.Array1d      # 最新估值价格
    last_value: tp.Array1d          # 最新组合价值
    second_last_value: tp.Array1d   # 次新组合价值
    last_return: tp.Array1d         # 最新收益率
    last_oidx: tp.Array1d           # 最新订单索引
    last_lidx: tp.Array1d           # 最新日志索引
    last_pos_record: tp.RecordArray # 最新仓位记录
    group: int                      # 当前组索引
    group_len: int                  # 当前组大小
    from_col: int                   # 组起始列索引
    to_col: int                     # 组结束列索引+1
    i: int                          # 当前行索引
    
    # SegmentContext特有字段
    call_seq_now: tp.Optional[tp.Array1d] # 当前段内的调用序列


__pdoc__['SegmentContext'] = """段上下文命名元组

该命名元组表示一个段的上下文环境。段是组和行的交集，是一个实体，
定义了在同一组和行内元素的处理方式和顺序。

**核心概念:**
- **段 = 组 × 行**: 段是特定组在特定时间步的处理单元
- **处理顺序**: 通过call_seq_now控制组内资产的处理顺序
- **协调执行**: 实现组内资产的协调和同步交易

**包含字段:**
- 继承SimulationContext、GroupContext和RowContext的所有字段
- 添加描述当前段的特定字段

**传递目标:**
- `pre_segment_func_nb`: 段处理前的预处理函数
- `post_segment_func_nb`: 段处理后的后处理函数

使用示例:
```python
def pre_segment_func_nb(c: SegmentContext) -> None:
    '''段开始前的准备处理'''
    print(f"处理段: 组{c.group}, 时间步{c.i}")
    print(f"组大小: {c.group_len}, 列范围: [{c.from_col}, {c.to_col})")
    
    # 动态调整调用序列
    if c.call_seq_now is not None:
        # 根据某种策略重新排序
        # 例如：按持仓价值排序，价值高的先交易
        positions = c.last_position[c.from_col:c.to_col]
        values = c.last_val_price[c.from_col:c.to_col] * positions
        sorted_indices = np.argsort(-values)  # 降序排列
        c.call_seq_now[:] = sorted_indices
        print(f"调整后的调用序列: {c.call_seq_now}")
    
    # 组级别的风险检查
    group_exposure = np.sum(np.abs(c.last_position[c.from_col:c.to_col]))
    print(f"组风险暴露: {group_exposure}")

def post_segment_func_nb(c: SegmentContext) -> None:
    '''段结束后的清理处理'''
    # 计算段执行后的组状态变化
    group_value = c.last_value[c.group] if c.cash_sharing else \
                  np.sum(c.last_value[c.from_col:c.to_col])
    print(f"段执行后组价值: {group_value}")
    
    # 记录段级别的统计信息
    orders_in_segment = np.sum(c.last_oidx[c.from_col:c.to_col] >= 0)
    print(f"本段执行订单数: {orders_in_segment}")

# 在投资组合中使用
pf = vbt.Portfolio.from_order_func(
    close=price_data,
    order_func_nb=my_order_func,
    pre_segment_func_nb=pre_segment_func_nb,
    post_segment_func_nb=post_segment_func_nb,
    group_by=[0, 0, 1, 1]  # 两组，每组两个资产
)
```

**段执行流程:**
1. 调用 `pre_segment_func_nb` - 段前处理
2. 按 `call_seq_now` 顺序调用 `order_func_nb`
3. 调用 `post_segment_func_nb` - 段后处理

**应用场景:**
- **动态排序**: 根据市场条件动态调整资产处理顺序
- **组内协调**: 实现组内资产的协调交易策略
- **风险控制**: 在段级别实施风险管理措施
- **状态同步**: 确保组内资产状态的一致性

**调用序列控制:**
- `call_seq_now` 控制组内资产的处理顺序
- 可以在 `pre_segment_func_nb` 中动态修改
- 影响资金使用的优先级和交易执行顺序
"""
for field in SegmentContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['SegmentContext.' + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__['SegmentContext.' + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__['SegmentContext.' + field] = f"See `RowContext.{field}`."
__pdoc__['SegmentContext.call_seq_now'] = """Sequence of calls within the current segment.

Has shape `(group_len,)`. 

Each value in this sequence should indicate the position of column in the group to
call next. Processing goes always from left to right.

You can use `pre_segment_func_nb` to override `call_seq_now`.
    
Example:
    `[2, 0, 1]` would first call column 2, then 0, and finally 1.
"""


class OrderContext(tp.NamedTuple):
    """
    订单上下文数据结构
    
    表示当前订单执行时的完整上下文信息。这是最详细的上下文，包含了
    SegmentContext的所有字段，并添加了描述当前状态的特定字段。
    该上下文为订单函数提供了做出交易决策所需的所有信息。
    
    设计用途：
    - 传递给 order_func_nb 订单函数
    - 提供当前资产和组的完整状态信息
    - 支持基于当前状态的智能交易决策
    - 包含所有必要的历史和实时数据
    
    核心特点：
    - 包含当前列（资产）的所有状态信息
    - 提供组级别和全局级别的上下文
    - 支持复杂的多资产交易策略
    - 实时反映最新的市场和持仓状态
    """
    # 继承自上级上下文的所有字段
    target_shape: tp.Shape          # 模拟目标形状
    group_lens: tp.Array1d          # 每组列数
    init_cash: tp.Array1d           # 初始资金
    cash_sharing: bool              # 现金共享标志
    call_seq: tp.Optional[tp.Array2d] # 调用序列
    segment_mask: tp.ArrayLike      # 段掩码
    call_pre_segment: bool          # 调用段前函数标志
    call_post_segment: bool         # 调用段后函数标志
    close: tp.ArrayLike            # 收盘价数据
    ffill_val_price: bool          # 前向填充估值价格标志
    update_value: bool             # 更新价值标志
    fill_pos_record: bool          # 填充仓位记录标志
    flex_2d: bool                  # 灵活二维索引标志
    order_records: tp.RecordArray   # 订单记录数组
    log_records: tp.RecordArray     # 日志记录数组
    last_cash: tp.Array1d           # 最新现金状态
    last_position: tp.Array1d       # 最新仓位状态
    last_debt: tp.Array1d           # 最新债务状态
    last_free_cash: tp.Array1d      # 最新可用现金
    last_val_price: tp.Array1d      # 最新估值价格
    last_value: tp.Array1d          # 最新组合价值
    second_last_value: tp.Array1d   # 次新组合价值
    last_return: tp.Array1d         # 最新收益率
    last_oidx: tp.Array1d           # 最新订单索引
    last_lidx: tp.Array1d           # 最新日志索引
    last_pos_record: tp.RecordArray # 最新仓位记录
    group: int                      # 当前组索引
    group_len: int                  # 当前组大小
    from_col: int                   # 组起始列索引
    to_col: int                     # 组结束列索引+1
    i: int                          # 当前行索引
    call_seq_now: tp.Optional[tp.Array1d] # 当前段调用序列
    
    # OrderContext特有字段 - 当前状态
    col: int                        # 当前列（资产）索引
    call_idx: int                   # 当前调用索引
    cash_now: float                 # 当前现金余额
    position_now: float             # 当前持仓数量
    debt_now: float                 # 当前做空债务
    free_cash_now: float            # 当前可用现金
    val_price_now: float            # 当前估值价格
    value_now: float                # 当前组合价值
    return_now: float               # 当前收益率
    pos_record_now: tp.Record       # 当前仓位记录


__pdoc__['OrderContext'] = """订单上下文命名元组

该命名元组表示订单执行时的完整上下文环境。它包含了SegmentContext的所有字段，
并添加了描述当前状态的特定字段，为订单函数提供了制定交易决策所需的全部信息。

**上下文层次:**
```
OrderContext (订单级上下文)
├── 继承 SegmentContext (段级上下文)
│   ├── 继承 GroupContext (组级上下文)
│   ├── 继承 RowContext (行级上下文)
│   └── 继承 SimulationContext (全局上下文)
└── 添加当前资产的实时状态信息
```

**传递目标:**
- `order_func_nb`: 订单生成函数，核心的交易逻辑函数

**核心功能:**
1. **状态查询**: 获取当前资产的所有状态信息
2. **决策支持**: 提供制定交易决策所需的完整数据
3. **风险评估**: 评估当前的风险暴露和资金状况
4. **策略执行**: 支持复杂的多资产交易策略

**关键状态字段:**
- `cash_now`: 当前可用现金（考虑现金共享）
- `position_now`: 当前持仓数量
- `val_price_now`: 当前资产价格
- `free_cash_now`: 当前自由现金（扣除做空保证金）

使用示例:
```python
def order_func_nb(c: OrderContext) -> Order:
    '''智能订单生成函数'''
    
    # 获取当前状态
    current_price = c.val_price_now
    current_position = c.position_now
    available_cash = c.free_cash_now
    
    # 获取历史价格用于技术分析
    if c.i >= 20:  # 确保有足够历史数据
        prices = c.close[:c.i+1, c.col]
        sma_20 = prices[-20:].mean()
        
        # 简单的移动平均策略
        if current_price > sma_20 and current_position == 0:
            # 价格突破20日均线且无持仓，买入
            max_shares = available_cash // current_price
            return Order(size=min(100, max_shares))
            
        elif current_price < sma_20 and current_position > 0:
            # 价格跌破20日均线且有持仓，卖出
            return Order(size=-current_position)
    
    # 风险管理：止损
    if current_position > 0:
        entry_price = c.pos_record_now['entry_price']
        if current_price < entry_price * 0.95:  # 5%止损
            return Order(size=-current_position)
    
    # 资金管理：控制单次投资金额
    if available_cash > c.init_cash[c.group if c.cash_sharing else c.col] * 0.1:
        # 当可用现金超过初始资金的10%时考虑投资
        investment_size = available_cash * 0.05  # 使用5%的现金
        shares = investment_size // current_price
        if shares > 0:
            return Order(size=shares)
    
    return NoOrder  # 无交易信号

# 在投资组合中使用
pf = vbt.Portfolio.from_order_func(
    close=price_data,
    order_func_nb=order_func_nb,
    init_cash=10000,
    fees=0.001
)
```

**高级应用:**
- **多资产协调**: 通过组信息协调多个资产的交易
- **动态资金分配**: 根据市场条件动态调整资金分配
- **复杂策略**: 实现基于多种指标的复杂交易策略
- **风险控制**: 实时监控和控制投资风险

**注意事项:**
- 上下文是只读的，不能直接修改状态
- 状态反映的是订单执行前的情况
- 需要返回有效的Order对象或NoOrder
- 复杂策略应考虑计算性能和内存使用
"""
for field in OrderContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['OrderContext.' + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__['OrderContext.' + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__['OrderContext.' + field] = f"See `RowContext.{field}`."
    elif field in SegmentContext._fields:
        __pdoc__['OrderContext.' + field] = f"See `SegmentContext.{field}`."
__pdoc__['OrderContext.col'] = """Current column.

Has range `[0, target_shape[1])` and is always within `[from_col, to_col)`.
"""
__pdoc__['OrderContext.call_idx'] = """Index of the current call in `SegmentContext.call_seq_now`.

Has range `[0, group_len)`.
"""
__pdoc__['OrderContext.cash_now'] = "`SimulationContext.last_cash` for the current column/group."
__pdoc__['OrderContext.position_now'] = "`SimulationContext.last_position` for the current column."
__pdoc__['OrderContext.debt_now'] = "`SimulationContext.last_debt` for the current column."
__pdoc__['OrderContext.free_cash_now'] = "`SimulationContext.last_free_cash` for the current column/group."
__pdoc__['OrderContext.val_price_now'] = "`SimulationContext.last_val_price` for the current column."
__pdoc__['OrderContext.value_now'] = "`SimulationContext.last_value` for the current column/group."
__pdoc__['OrderContext.return_now'] = "`SimulationContext.last_return` for the current column/group."
__pdoc__['OrderContext.pos_record_now'] = "`SimulationContext.last_pos_record` for the current column."


class PostOrderContext(tp.NamedTuple):
    """
    订单后上下文数据结构
    
    表示订单处理完成后的上下文信息。包含了OrderContext的所有字段，
    并添加了订单执行结果和执行前状态的字段。该上下文提供了订单
    执行前后的完整对比信息，用于后续处理和分析。
    
    设计用途：
    - 传递给 post_order_func_nb 订单后处理函数
    - 提供订单执行前后的状态对比
    - 支持基于执行结果的后续处理
    - 记录和分析订单执行效果
    
    核心特点：
    - 包含订单执行前的完整状态快照
    - 包含订单执行结果的详细信息
    - 包含订单执行后的最新状态
    - 支持执行效果的分析和验证
    """
    # 继承自上级上下文的所有字段（省略重复字段注释）
    target_shape: tp.Shape
    group_lens: tp.Array1d
    init_cash: tp.Array1d
    cash_sharing: bool
    call_seq: tp.Optional[tp.Array2d]
    segment_mask: tp.ArrayLike
    call_pre_segment: bool
    call_post_segment: bool
    close: tp.ArrayLike
    ffill_val_price: bool
    update_value: bool
    fill_pos_record: bool
    flex_2d: bool
    order_records: tp.RecordArray
    log_records: tp.RecordArray
    last_cash: tp.Array1d
    last_position: tp.Array1d
    last_debt: tp.Array1d
    last_free_cash: tp.Array1d
    last_val_price: tp.Array1d
    last_value: tp.Array1d
    second_last_value: tp.Array1d
    last_return: tp.Array1d
    last_oidx: tp.Array1d
    last_lidx: tp.Array1d
    last_pos_record: tp.RecordArray
    group: int
    group_len: int
    from_col: int
    to_col: int
    i: int
    call_seq_now: tp.Optional[tp.Array1d]
    col: int
    call_idx: int
    
    # 执行前状态字段
    cash_before: float          # 执行前现金余额
    position_before: float      # 执行前持仓数量
    debt_before: float          # 执行前做空债务
    free_cash_before: float     # 执行前可用现金
    val_price_before: float     # 执行前估值价格
    value_before: float         # 执行前组合价值
    
    # 执行结果字段
    order_result: "OrderResult" # 订单执行结果
    
    # 执行后状态字段
    cash_now: float             # 执行后现金余额
    position_now: float         # 执行后持仓数量
    debt_now: float             # 执行后做空债务
    free_cash_now: float        # 执行后可用现金
    val_price_now: float        # 执行后估值价格
    value_now: float            # 执行后组合价值
    return_now: float           # 执行后收益率
    pos_record_now: tp.Record   # 执行后仓位记录


__pdoc__['PostOrderContext'] = """订单后上下文命名元组

该命名元组表示订单处理完成后的上下文环境。它包含了OrderContext的所有字段，
并添加了描述订单执行结果和执行前状态的字段。

**核心功能:**
- **状态对比**: 提供执行前后的完整状态对比
- **结果分析**: 包含详细的订单执行结果信息
- **效果评估**: 支持订单执行效果的分析和验证
- **后续处理**: 为基于执行结果的后续处理提供支持

**包含字段:**
- 继承OrderContext的所有字段
- 添加执行前状态字段（*_before）
- 添加订单执行结果（order_result）
- 更新执行后状态字段（*_now）

**传递目标:**
- `post_order_func_nb`: 订单处理后的后续处理函数

**状态字段对比:**
- `*_before`: 订单执行前的状态
- `order_result`: 订单执行的详细结果
- `*_now`: 订单执行后的最新状态

使用示例:
```python
def post_order_func_nb(c: PostOrderContext) -> None:
    '''订单执行后的分析和处理'''
    
    # 分析订单执行结果
    result = c.order_result
    if result.status == OrderStatus.Filled:
        print(f"订单成功执行: {result.size}股 @ {result.price}")
        
        # 计算执行效果
        cash_change = c.cash_now - c.cash_before
        position_change = c.position_now - c.position_before
        
        print(f"现金变化: {cash_change}")
        print(f"持仓变化: {position_change}")
        print(f"手续费: {result.fees}")
        
        # 验证计算正确性
        expected_cash_change = -(result.size * result.price + result.fees)
        if abs(cash_change - expected_cash_change) > 1e-6:
            print("警告: 现金变化计算异常!")
            
    elif result.status == OrderStatus.Rejected:
        print(f"订单被拒绝: {result.status_info}")
        
        # 分析拒绝原因
        if result.status_info == OrderStatusInfo.NoCashLong:
            print("拒绝原因: 资金不足")
        elif result.status_info == OrderStatusInfo.MinSizeNotReached:
            print("拒绝原因: 订单大小低于最小限制")
            
        # 状态应该没有变化
        assert c.cash_now == c.cash_before
        assert c.position_now == c.position_before
        
    else:  # Ignored
        print("订单被忽略")
    
    # 风险监控
    if c.position_now != 0:
        # 计算当前持仓价值
        position_value = abs(c.position_now * c.val_price_now)
        portfolio_value = c.value_now
        
        # 检查仓位集中度
        concentration = position_value / portfolio_value if portfolio_value > 0 else 0
        if concentration > 0.5:  # 超过50%
            print(f"警告: 单一资产仓位过于集中 ({concentration:.1%})")
    
    # 记录交易统计
    if hasattr(c, 'trade_stats') and result.status == OrderStatus.Filled:
        c.trade_stats['total_trades'] += 1
        c.trade_stats['total_fees'] += result.fees
        if result.side == OrderSide.Buy:
            c.trade_stats['buy_trades'] += 1
        else:
            c.trade_stats['sell_trades'] += 1

# 在投资组合中使用
pf = vbt.Portfolio.from_order_func(
    close=price_data,
    order_func_nb=my_order_func,
    post_order_func_nb=post_order_func_nb
)
```

**应用场景:**
- **执行验证**: 验证订单执行的正确性和一致性
- **风险监控**: 监控订单执行后的风险状况变化
- **统计记录**: 记录交易统计信息和绩效指标
- **异常处理**: 处理订单执行中的异常情况
- **策略调整**: 根据执行结果调整后续策略

**执行结果分析:**
- 成功执行: 分析执行价格、数量、费用等
- 被拒绝: 分析拒绝原因，调整策略参数
- 被忽略: 检查忽略的合理性

**状态一致性检查:**
- 验证状态变化的逻辑正确性
- 确保资金和持仓的平衡
- 检查计算精度和数值稳定性
"""
for field in PostOrderContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['PostOrderContext.' + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__['PostOrderContext.' + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__['PostOrderContext.' + field] = f"See `RowContext.{field}`."
    elif field in SegmentContext._fields:
        __pdoc__['PostOrderContext.' + field] = f"See `SegmentContext.{field}`."
    elif field in OrderContext._fields:
        __pdoc__['PostOrderContext.' + field] = f"See `OrderContext.{field}`."
__pdoc__['PostOrderContext.cash_before'] = "`OrderContext.cash_now` before execution."
__pdoc__['PostOrderContext.position_before'] = "`OrderContext.position_now` before execution."
__pdoc__['PostOrderContext.debt_before'] = "`OrderContext.debt_now` before execution."
__pdoc__['PostOrderContext.free_cash_before'] = "`OrderContext.free_cash_now` before execution."
__pdoc__['PostOrderContext.val_price_before'] = "`OrderContext.val_price_now` before execution."
__pdoc__['PostOrderContext.value_before'] = "`OrderContext.value_now` before execution."
__pdoc__['PostOrderContext.order_result'] = """Order result of type `OrderResult`.

Can be used to check whether the order has been filled, ignored, or rejected.
"""
__pdoc__['PostOrderContext.cash_now'] = "`OrderContext.cash_now` after execution."
__pdoc__['PostOrderContext.position_now'] = "`OrderContext.position_now` after execution."
__pdoc__['PostOrderContext.debt_now'] = "`OrderContext.debt_now` after execution."
__pdoc__['PostOrderContext.free_cash_now'] = "`OrderContext.free_cash_now` after execution."
__pdoc__['PostOrderContext.val_price_now'] = """`OrderContext.val_price_now` after execution.

If `SimulationContext.update_value`, gets replaced with the fill price, 
as it becomes the most recently known price. Otherwise, stays the same.
"""
__pdoc__['PostOrderContext.value_now'] = """`OrderContext.value_now` after execution.

If `SimulationContext.update_value`, gets updated with the new cash and value of the column. Otherwise, stays the same.
"""
__pdoc__['PostOrderContext.return_now'] = "`OrderContext.return_now` after execution."
__pdoc__['PostOrderContext.pos_record_now'] = "`OrderContext.pos_record_now` after execution."


class FlexOrderContext(tp.NamedTuple):
    """
    灵活订单上下文数据结构
    
    表示灵活订单执行时的上下文信息。与OrderContext不同，FlexOrderContext
    不绑定到特定的列，而是提供更灵活的订单生成方式。包含了SegmentContext
    的所有字段，并添加了当前调用索引。
    
    设计用途：
    - 传递给 flex_order_func_nb 灵活订单函数
    - 支持不按固定列顺序的订单生成
    - 实现更复杂的多资产协调策略
    - 提供最大的订单生成灵活性
    
    核心特点：
    - 不绑定到特定列，可以为任意列生成订单
    - 支持动态的资产选择和订单分配
    - 适合复杂的多资产交易策略
    - 提供完全的订单生成控制权
    """
    # 继承自上级上下文的所有字段
    target_shape: tp.Shape          # 模拟目标形状
    group_lens: tp.Array1d          # 每组列数
    init_cash: tp.Array1d           # 初始资金
    cash_sharing: bool              # 现金共享标志
    call_seq: tp.Optional[tp.Array2d] # 调用序列
    segment_mask: tp.ArrayLike      # 段掩码
    call_pre_segment: bool          # 调用段前函数标志
    call_post_segment: bool         # 调用段后函数标志
    close: tp.ArrayLike            # 收盘价数据
    ffill_val_price: bool          # 前向填充估值价格标志
    update_value: bool             # 更新价值标志
    fill_pos_record: bool          # 填充仓位记录标志
    flex_2d: bool                  # 灵活二维索引标志
    order_records: tp.RecordArray   # 订单记录数组
    log_records: tp.RecordArray     # 日志记录数组
    last_cash: tp.Array1d           # 最新现金状态
    last_position: tp.Array1d       # 最新仓位状态
    last_debt: tp.Array1d           # 最新债务状态
    last_free_cash: tp.Array1d      # 最新可用现金
    last_val_price: tp.Array1d      # 最新估值价格
    last_value: tp.Array1d          # 最新组合价值
    second_last_value: tp.Array1d   # 次新组合价值
    last_return: tp.Array1d         # 最新收益率
    last_oidx: tp.Array1d           # 最新订单索引
    last_lidx: tp.Array1d           # 最新日志索引
    last_pos_record: tp.RecordArray # 最新仓位记录
    group: int                      # 当前组索引
    group_len: int                  # 当前组大小
    from_col: int                   # 组起始列索引
    to_col: int                     # 组结束列索引+1
    i: int                          # 当前行索引
    
    # FlexOrderContext特有字段
    call_seq_now: None              # 无调用序列（灵活模式）
    call_idx: int                   # 当前调用索引


__pdoc__['FlexOrderContext'] = """灵活订单上下文命名元组

该命名元组表示灵活订单的上下文环境。与标准的OrderContext不同，
FlexOrderContext不绑定到特定的列，提供了更大的订单生成灵活性。

**核心概念:**
- **灵活性**: 不受固定列顺序约束，可以为任意列生成订单
- **多资产协调**: 支持复杂的多资产交易策略和资金分配
- **动态选择**: 根据市场条件动态选择交易的资产
- **完全控制**: 提供对订单生成过程的完全控制权

**包含字段:**
- 继承SegmentContext的所有字段
- 添加当前调用索引

**传递目标:**
- `flex_order_func_nb`: 灵活订单生成函数

**与OrderContext的区别:**
- OrderContext: 绑定到特定列，按固定顺序调用
- FlexOrderContext: 不绑定列，可以灵活选择目标资产

使用示例:
```python
def flex_order_func_nb(c: FlexOrderContext) -> tp.Tuple[int, Order]:
    '''灵活订单生成函数'''
    
    # 分析组内所有资产的状态
    group_positions = c.last_position[c.from_col:c.to_col]
    group_prices = c.last_val_price[c.from_col:c.to_col]
    group_values = group_positions * group_prices
    
    # 策略1: 动态资产轮换
    if c.call_idx == 0:
        # 第一次调用：选择表现最差的资产买入
        if len(group_values) > 1:
            worst_col = c.from_col + np.argmin(group_values)
            available_cash = c.last_cash[c.group] if c.cash_sharing else c.last_cash[worst_col]
            
            if available_cash > group_prices[worst_col - c.from_col] * 100:
                return worst_col, Order(size=100)
    
    elif c.call_idx == 1:
        # 第二次调用：选择表现最好的资产减仓
        if len(group_values) > 1:
            best_col = c.from_col + np.argmax(group_values)
            if group_positions[best_col - c.from_col] > 0:
                return best_col, Order(size=-50)  # 减仓50股
    
    # 策略2: 配对交易
    elif c.call_idx == 2 and c.group_len == 2:
        # 配对交易：两个资产的价差交易
        col1, col2 = c.from_col, c.from_col + 1
        price1, price2 = group_prices[0], group_prices[1]
        pos1, pos2 = group_positions[0], group_positions[1]
        
        # 计算价格比率
        ratio = price1 / price2 if price2 != 0 else 1
        
        # 如果比率偏离历史均值，执行配对交易
        if ratio > 1.2:  # 资产1相对贵，卖出资产1，买入资产2
            if pos1 > 0:
                return col1, Order(size=-min(50, pos1))
        elif ratio < 0.8:  # 资产1相对便宜，买入资产1
            available_cash = c.last_cash[c.group] if c.cash_sharing else c.last_cash[col1]
            if available_cash > price1 * 50:
                return col1, Order(size=50)
    
    # 策略3: 风险平衡
    elif c.call_idx == 3:
        # 重新平衡组内资产权重
        total_value = np.sum(group_values)
        if total_value > 0:
            target_weight = 1.0 / c.group_len  # 等权重
            
            for i in range(c.group_len):
                col = c.from_col + i
                current_weight = group_values[i] / total_value
                
                # 如果权重偏差超过5%，进行调整
                if abs(current_weight - target_weight) > 0.05:
                    target_value = total_value * target_weight
                    current_value = group_values[i]
                    
                    if current_value > target_value:
                        # 减仓
                        excess_value = current_value - target_value
                        shares_to_sell = min(group_positions[i], 
                                           excess_value / group_prices[i])
                        if shares_to_sell > 0:
                            return col, Order(size=-shares_to_sell)
    
    # 默认：无交易
    return -1, NoOrder

# 在投资组合中使用
pf = vbt.Portfolio.from_order_func(
    close=price_data,
    flex_order_func_nb=flex_order_func_nb,
    group_by=[0, 0, 1, 1, 1],  # 两组：2+3资产
    max_orders_per_segment=5,   # 每段最多5个订单
    max_orders=1000
)
```

**高级应用场景:**
- **动态资产轮换**: 根据表现动态选择交易资产
- **配对交易**: 实现复杂的配对和套利策略
- **风险平衡**: 动态调整投资组合权重和风险暴露
- **多策略融合**: 在同一段内执行多种不同的交易策略
- **资金优化**: 优化资金在不同资产间的分配

**返回值格式:**
- 返回元组 `(column_index, order)`
- `column_index`: 目标列索引，-1表示无交易
- `order`: Order对象或NoOrder

**性能考虑:**
- 灵活性带来了更高的计算复杂度
- 需要合理控制调用次数和策略复杂度
- 建议设置合适的max_orders_per_segment参数

**注意事项:**
- 必须返回有效的列索引（在组范围内）
- 列索引必须在[from_col, to_col)范围内
- 返回-1表示本次调用不生成订单
- 需要手动管理调用次数和终止条件
"""
for field in FlexOrderContext._fields:
    if field in SimulationContext._fields:
        __pdoc__['FlexOrderContext.' + field] = f"See `SimulationContext.{field}`."
    elif field in GroupContext._fields:
        __pdoc__['FlexOrderContext.' + field] = f"See `GroupContext.{field}`."
    elif field in RowContext._fields:
        __pdoc__['FlexOrderContext.' + field] = f"See `RowContext.{field}`."
    elif field in SegmentContext._fields:
        __pdoc__['FlexOrderContext.' + field] = f"See `SegmentContext.{field}`."
__pdoc__['FlexOrderContext.call_idx'] = "Index of the current call."


class Order(tp.NamedTuple):
    """
    订单定义数据结构
    
    这是vectorbt投资组合系统中最核心的订单定义类，包含了执行一个订单所需的
    所有参数和控制选项。该类采用命名元组设计，确保了高性能和不可变性。
    
    设计特点：
    - 所有参数都有合理的默认值
    - 支持复杂的订单类型和执行控制
    - 内置风险管理和错误处理机制
    - 与vectorbt模拟引擎深度集成
    
    字段分类：
    【核心订单参数】
    - size: 订单大小
    - price: 订单价格  
    - size_type: 大小类型
    - direction: 允许方向
    
    【成本控制参数】
    - fees: 手续费率
    - fixed_fees: 固定手续费
    - slippage: 滑点率
    
    【风险控制参数】
    - min_size: 最小订单大小
    - max_size: 最大订单大小
    - size_granularity: 大小粒度
    - reject_prob: 拒绝概率
    
    【执行控制参数】
    - lock_cash: 锁定现金标志
    - allow_partial: 允许部分成交
    - raise_reject: 拒绝时抛异常
    - log: 记录日志标志
    """
    size: float = np.inf              # 订单大小：要交易的数量或金额
    price: float = np.inf             # 订单价格：每单位的交易价格
    size_type: int = SizeType.Amount  # 大小类型：订单大小的解释方式
    direction: int = Direction.Both   # 允许方向：订单允许的交易方向
    fees: float = 0.0                # 手续费率：按订单价值的百分比收费
    fixed_fees: float = 0.0          # 固定手续费：每笔订单的固定费用
    slippage: float = 0.0            # 滑点率：价格滑动的百分比
    min_size: float = 0.0            # 最小大小：订单的最小允许大小
    max_size: float = np.inf         # 最大大小：订单的最大允许大小
    size_granularity: float = np.nan # 大小粒度：订单大小的最小调整单位
    reject_prob: float = 0.0         # 拒绝概率：随机拒绝订单的概率
    lock_cash: bool = False          # 锁定现金：做空时是否锁定现金
    allow_partial: bool = True       # 允许部分成交：是否接受部分填充
    raise_reject: bool = False       # 拒绝异常：拒绝时是否抛出异常
    log: bool = False               # 日志记录：是否记录此订单的详细日志


__pdoc__['Order'] = """订单定义数据结构

该命名元组代表vectorbt系统中的一个完整订单定义，包含了执行订单所需的所有参数。
这是投资组合模拟的核心数据结构，每个字段都经过精心设计以支持复杂的交易策略。

设计原则:
- **高性能**: 使用命名元组确保最佳性能
- **类型安全**: 每个字段都有明确的类型定义
- **默认友好**: 所有字段都有合理的默认值
- **风险可控**: 内置多种风险控制机制

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import Order, SizeType, Direction

# 创建基本订单
basic_order = Order(
    size=100,           # 买入100股
    price=50.0,         # 每股50元
    fees=0.001         # 0.1%手续费
)

# 创建百分比订单
percent_order = Order(
    size=0.5,                    # 使用50%可用资金
    size_type=SizeType.Percent,  # 百分比模式
    price=np.inf                # 市价单（使用收盘价）
)

# 创建限价订单
limit_order = Order(
    size=200,           # 200股
    price=45.0,         # 限价45元
    min_size=50,        # 最小50股
    allow_partial=True  # 允许部分成交
)

# 创建风控订单
risk_controlled_order = Order(
    size=1000,
    max_size=500,       # 最多只能成交500股
    reject_prob=0.05,   # 5%随机拒绝概率
    log=True           # 记录详细日志
)
```

!!! 重要提示
    由于Numba当前在处理带默认值的命名元组时存在问题，
    建议使用 `vectorbt.portfolio.nb.order_nb` 函数来创建订单对象。

订单生命周期:
1. **订单创建**: 通过Order类或order_nb函数创建
2. **参数验证**: 系统验证所有参数的有效性
3. **执行检查**: 检查资金、仓位等执行条件
4. **订单执行**: 根据参数执行实际交易
5. **结果记录**: 生成OrderResult并更新状态
6. **日志记录**: 如果启用，记录详细执行日志

高级特性:
- **智能大小处理**: 支持固定大小、百分比、目标仓位等多种模式
- **自动价格处理**: 支持市价、限价、开盘价等多种价格类型
- **风险管理**: 内置大小限制、拒绝概率等风险控制机制
- **成本控制**: 精确的手续费和滑点计算
- **调试支持**: 可选的详细日志记录功能
"""
__pdoc__['Order.size'] = """订单大小（以单位计）

订单大小的行为取决于 `Order.size_type` 和 `Order.direction` 的设置。

**固定大小模式（Amount/Value）:**

* 设置为任意数字：买入/卖出指定数量或价值
    - 多头订单受当前现金余额限制
    - 空头订单仅在启用 `Order.lock_cash` 时受限制
* 设置为 `np.inf`：使用全部现金买入
* 设置为 `-np.inf`：卖出全部可用现金对应的仓位
    - 如果 `Order.direction` 不是 `Direction.Both`，`-np.inf` 将平掉当前仓位
* 设置为 `np.nan` 或 0：跳过此订单

**百分比模式（Percent）:**

* 设置为 0.0-1.0 之间的数字：使用相应比例的可用资源
* 设置为 `np.nan`：跳过此订单

**目标大小模式（Target系列）:**

* 设置为任意数字：买入/卖出至目标仓位或价值
* 设置为 0：平掉当前仓位
* 设置为 `np.nan`：跳过此订单

使用示例:
```python
# 固定数量订单
Order(size=100)  # 买入100股

# 使用全部现金
Order(size=np.inf)  # 全仓买入

# 平掉所有仓位  
Order(size=-np.inf, direction=Direction.LongOnly)  # 卖出平仓

# 百分比订单
Order(size=0.5, size_type=SizeType.Percent)  # 使用50%资金

# 目标仓位订单
Order(size=200, size_type=SizeType.TargetAmount)  # 调整至200股
```
"""

__pdoc__['Order.price'] = """每单位价格

最终价格将受滑点影响。

**特殊价格值:**
* `-np.inf`：使用当前开盘价（如果可用）或前一收盘价（≈ 加密货币的当前开盘价）
* `np.inf`：使用当前收盘价（市价单）
* 具体数值：使用指定的限价

**价格处理逻辑:**
1. 系统首先获取指定的价格
2. 应用滑点调整：final_price = price * (1 + slippage * side)
3. 使用调整后的价格执行订单

使用示例:
```python
# 市价单（使用收盘价）
Order(price=np.inf)

# 限价单
Order(price=50.25)

# 开盘价单
Order(price=-np.inf)

# 带滑点的订单
Order(price=50.0, slippage=0.001)  # 0.1%滑点
```

!!! 重要提示
    确保使用的时间戳在当前开盘价和收盘价之间（最好不包括开盘和收盘时刻）。
"""

__pdoc__['Order.size_type'] = """订单大小类型

参见 `SizeType` 枚举的详细说明。

该字段决定了如何解释 `Order.size` 参数：
- `SizeType.Amount`：绝对数量
- `SizeType.Value`：价值金额  
- `SizeType.Percent`：可用资源百分比
- `SizeType.TargetAmount`：目标绝对数量
- `SizeType.TargetValue`：目标价值金额
- `SizeType.TargetPercent`：目标价值百分比
"""

__pdoc__['Order.direction'] = """允许的交易方向

参见 `Direction` 枚举的详细说明。

该字段限制订单可以执行的方向：
- `Direction.LongOnly`：仅允许多头操作（买入）
- `Direction.ShortOnly`：仅允许空头操作（卖出）  
- `Direction.Both`：允许双向操作（默认）

与 `Order.size` 的符号配合使用来确定最终的订单行为。
"""

__pdoc__['Order.fees'] = """手续费率（按订单价值的百分比）

以订单价值的百分比形式收取的手续费。

**费率说明:**
- 正值：支付手续费（如 0.001 = 0.1%）
- 负值：获得返佣（如 -0.0005 = 获得0.05%返佣）
- 零值：无手续费交易

**计算方式:**
```
手续费金额 = 订单价值 × fees
订单价值 = abs(size) × price
```

使用示例:
```python
# 0.1%手续费
Order(fees=0.001)

# 0.25%手续费  
Order(fees=0.0025)

# 返佣交易
Order(fees=-0.0005)

# 免费交易
Order(fees=0.0)
```

!!! 注意
    0.01 = 1%，请注意小数点位置。
"""

__pdoc__['Order.fixed_fees'] = """固定手续费金额

每笔订单收取的固定费用，与订单大小无关。

**费用说明:**
- 正值：支付固定费用
- 负值：获得固定返佣
- 零值：无固定费用

**与比例费用的关系:**
总手续费 = 比例费用 + 固定费用

使用示例:
```python
# 每笔订单5元固定费用
Order(fixed_fees=5.0)

# 同时收取比例费用和固定费用
Order(fees=0.001, fixed_fees=2.0)

# 固定返佣
Order(fixed_fees=-1.0)
```
"""

__pdoc__['Order.slippage'] = """滑点率（按订单价格的百分比）

滑点是对价格施加的惩罚，模拟真实市场中的价格冲击。

**滑点计算:**
```
最终价格 = 原始价格 × (1 + 滑点率 × 方向系数)
方向系数：买入为+1，卖出为-1
```

**滑点影响:**
- 买入时：价格上升（不利）
- 卖出时：价格下降（不利）
- 总是对交易者不利

使用示例:
```python
# 0.1%滑点
Order(price=100, slippage=0.001)
# 买入最终价格: 100 × (1 + 0.001) = 100.1
# 卖出最终价格: 100 × (1 - 0.001) = 99.9

# 0.5%滑点
Order(slippage=0.005)

# 无滑点
Order(slippage=0.0)
```

!!! 注意
    0.01 = 1%，滑点总是对交易者不利。
"""

__pdoc__['Order.min_size'] = """最小订单大小限制

双向生效的最小订单大小限制。

**限制规则:**
- 如果计算出的订单大小小于此值，订单将被拒绝
- 适用于买入和卖出方向
- 用于避免过小的无意义交易

**应用场景:**
- 避免微小订单产生过高的相对成本
- 符合交易所的最小订单要求
- 减少过于频繁的小额交易

使用示例:
```python
# 最小10股
Order(min_size=10)

# 最小100元价值
Order(min_size=100, size_type=SizeType.Value)

# 无最小限制
Order(min_size=0.0)
```
"""

__pdoc__['Order.max_size'] = """最大订单大小限制

双向生效的最大订单大小限制。

**限制规则:**
- 如果计算出的订单大小大于此值，将部分成交至最大值
- 适用于买入和卖出方向  
- 用于风险控制和分批执行

**应用场景:**
- 控制单笔订单的最大风险暴露
- 分批执行大额订单以减少市场冲击
- 遵守监管或交易所的单笔限额要求

使用示例:
```python
# 最大1000股
Order(max_size=1000)

# 最大10000元价值
Order(max_size=10000, size_type=SizeType.Value)

# 无最大限制
Order(max_size=np.inf)
```
"""

__pdoc__['Order.size_granularity'] = """订单大小粒度

订单大小的最小调整单位，用于将连续值离散化。

**粒度规则:**
- 最终订单大小将调整为粒度的整数倍
- 调整方向为向下取整（减少风险）
- NaN表示无粒度限制

**应用场景:**
- 模拟股票交易的整手要求（如100股为一手）
- 符合交易所的最小变动单位
- 简化订单管理和风险计算

使用示例:
```python
# 按1股粒度调整（整股交易）
Order(size=12.7, size_granularity=1.0)  # 实际执行12股

# 按100股粒度调整（整手交易）
Order(size=250, size_granularity=100)   # 实际执行200股

# 按0.01粒度调整
Order(size=10.567, size_granularity=0.01)  # 实际执行10.56

# 无粒度限制
Order(size_granularity=np.nan)
```

!!! 注意
    成交大小仍然是浮点数，粒度只影响计算过程。
"""

__pdoc__['Order.reject_prob'] = """订单随机拒绝概率

模拟随机拒绝事件的概率，用于测试订单管理的健壮性。

**概率范围:**
- 0.0：从不拒绝
- 1.0：总是拒绝  
- 0.0-1.0：相应概率的随机拒绝

**应用目的:**
- 模拟真实市场中的不确定性
- 测试策略对执行失败的适应能力
- 验证风险管理和错误处理机制
- 压力测试交易系统的健壮性

**实际意义:**
真实交易中，订单可能因各种原因被拒绝：
- 流动性不足
- 价格波动过大
- 系统故障
- 监管限制

使用示例:
```python
# 5%拒绝概率
Order(reject_prob=0.05)

# 10%拒绝概率（高压力测试）
Order(reject_prob=0.1)

# 从不拒绝
Order(reject_prob=0.0)

# 总是拒绝（用于测试）
Order(reject_prob=1.0)
```

!!! 提示
    现实交易并非总是顺利，使用随机拒绝来测试订单管理的健壮性。
"""
__pdoc__['Order.lock_cash'] = """做空时是否锁定现金

控制做空操作时的现金管理策略。

**启用时（True）:**
- 防止买入或做空时 `free_cash` 变为负数
- 每列必须有足够的现金支持其操作
- 避免跨列资金互相抵押的情况

**禁用时（False）:**
- 允许 `free_cash` 为负数
- 一个列可以使用另一个列的抵押品
- 更灵活的资金使用，但风险更高

**负数free_cash的含义:**
当free_cash为负数时，表示一个列使用了另一个列的抵押品，
这在一般情况下是不希望出现的情况。

使用示例:
```python
# 锁定现金（保守策略）
Order(size=-100, lock_cash=True)   # 做空时必须有足够现金

# 不锁定现金（激进策略）
Order(size=-100, lock_cash=False)  # 允许使用抵押品做空

# 多资产组合中的应用
pf = vbt.Portfolio.from_orders(
    price_data,
    orders,
    lock_cash=True,  # 全局设置
    group_by=None    # 每列独立管理现金
)
```

风险提示:
- 禁用现金锁定可能导致资金管理混乱
- 建议在充分理解风险的情况下使用
- 对于新手建议保持默认值（False）
"""

__pdoc__['Order.allow_partial'] = """是否允许部分成交

控制当订单无法完全执行时的处理策略。

**允许部分成交（True）:**
- 订单可以部分执行，成交可用的数量
- 剩余部分被忽略，不会产生错误
- 适用于大额订单或流动性有限的情况

**禁止部分成交（False）:**
- 订单必须完全执行，否则被拒绝
- 保证订单的完整性和策略的一致性
- 适用于对执行精度要求高的策略

**特殊情况:**
当 `Order.size` 为 `np.inf` 时，此设置不适用，
因为无限大小的订单本身就是要使用全部可用资源。

使用示例:
```python
# 允许部分成交（灵活策略）
Order(size=1000, allow_partial=True)   # 可能只成交800股

# 禁止部分成交（严格策略）
Order(size=1000, allow_partial=False)  # 要么成交1000股，要么拒绝

# 结合最小大小使用
Order(
    size=1000,
    min_size=800,         # 最少800股
    allow_partial=True    # 允许部分成交
)  # 可以成交800-1000股之间的任意数量

# 全仓订单（此设置无效）
Order(size=np.inf, allow_partial=False)  # 设置被忽略
```

策略考虑:
- 部分成交可能影响策略的预期效果
- 建议根据策略类型和市场条件选择
- 可以配合min_size参数使用以保证最小执行量
"""

__pdoc__['Order.raise_reject'] = """订单被拒绝时是否抛出异常

控制订单执行失败时的错误处理策略。

**抛出异常（True）:**
- 订单被拒绝时立即抛出异常
- 终止整个模拟过程
- 适用于调试和严格的策略验证

**不抛出异常（False）:**
- 订单被拒绝时静默处理
- 模拟继续进行，记录拒绝状态
- 适用于生产环境和健壮的策略

**应用场景:**

*调试模式:*
```python
# 调试时发现问题立即停止
Order(size=1000, raise_reject=True)
```

*生产模式:*
```python
# 生产环境中优雅处理失败
Order(size=1000, raise_reject=False)

# 通过日志分析拒绝原因
pf = vbt.Portfolio.from_orders(..., log=True)
rejected_orders = pf.logs.rejected
```

*错误处理策略:*
```python
# 结合异常处理使用
try:
    pf = vbt.Portfolio.from_orders(
        price_data,
        orders,
        raise_reject=True  # 启用异常
    )
except Exception as e:
    print(f"订单执行失败: {e}")
    # 实施备用策略
```

注意事项:
- 启用异常会中断整个模拟过程
- 建议在开发阶段启用，生产环境禁用
- 可以配合日志记录进行详细的错误分析
"""

__pdoc__['Order.log'] = """是否为此订单记录详细日志

控制是否为当前订单生成详细的执行日志记录。

**启用日志（True）:**
- 记录订单执行前后的完整状态信息
- 包括现金、仓位、价格等所有相关数据
- 便于详细的调试和分析

**禁用日志（False）:**
- 不记录此订单的日志信息
- 节省内存和提高性能
- 适用于不需要详细分析的订单

**重要提醒:**
启用日志记录时，请确保增加 `max_logs` 参数，
否则可能因为日志缓冲区满而无法记录。

使用示例:
```python
# 为重要订单启用日志
important_order = Order(
    size=1000,
    price=50.0,
    log=True  # 记录详细日志
)

# 批量订单中选择性记录
orders = [
    Order(size=100, log=False),  # 小订单不记录
    Order(size=1000, log=True),  # 大订单记录日志
    Order(size=50, log=False)
]

# 在投资组合中启用日志
pf = vbt.Portfolio.from_orders(
    price_data,
    orders,
    max_logs=10000,  # 增加日志缓冲区
    log=True         # 全局启用日志
)

# 分析日志记录
logs = pf.logs
detailed_logs = logs[logs.req_log == True]  # 获取启用日志的订单
```

性能考虑:
- 日志记录会显著增加内存使用
- 建议只为关键订单启用日志
- 大规模回测时应谨慎使用
- 可以通过max_logs限制日志数量

分析价值:
- 提供订单执行的完整追踪
- 便于识别执行异常和优化机会
- 支持详细的策略调试和验证
- 有助于理解复杂策略的执行过程
"""

NoOrder = Order(
    size=np.nan,             # 无效大小，表示跳过
    price=np.nan,            # 无效价格
    size_type=-1,            # 无效大小类型
    direction=-1,            # 无效方向
    fees=np.nan,             # 无效手续费
    fixed_fees=np.nan,       # 无效固定费用
    slippage=np.nan,         # 无效滑点
    min_size=np.nan,         # 无效最小大小
    max_size=np.nan,         # 无效最大大小
    size_granularity=np.nan, # 无效粒度
    reject_prob=np.nan,      # 无效拒绝概率
    lock_cash=False,         # 不锁定现金
    allow_partial=False,     # 不允许部分成交
    raise_reject=False,      # 不抛出异常
    log=False               # 不记录日志
)
"""空订单实例，表示不应该被处理的订单，用于跳过特定的交易时点"""

__pdoc__['NoOrder'] = """空订单常量

预定义的特殊订单实例，表示不应该被处理的订单。所有关键参数都设置为NaN或无效值，
确保此订单会被系统跳过而不执行任何交易操作。

**设计目的:**
- 提供统一的"无操作"订单表示
- 在订单序列中标记跳过的时间点
- 简化条件订单逻辑的实现
- 避免需要特殊的空值处理

**主要特征:**
- 所有数值参数都是NaN
- 所有枚举参数都是无效值(-1)
- 所有布尔参数都是False
- 保证不会执行任何实际交易

使用示例:
```python
import vectorbt as vbt
from vectorbt.portfolio.enums import Order, NoOrder

# 创建订单序列，在某些时点跳过交易
orders = [
    Order(size=100, price=50.0),  # 第一天买入
    NoOrder,                      # 第二天跳过
    Order(size=-50, price=55.0),  # 第三天部分卖出
    NoOrder,                      # 第四天跳过
    Order(size=-50, price=60.0)   # 第五天剩余卖出
]

# 条件订单逻辑
def create_order(condition, size, price):
    if condition:
        return Order(size=size, price=price)
    else:
        return NoOrder

# 在投资组合中使用
orders = [create_order(signal, 100, price) for signal, price in zip(signals, prices)]
pf = vbt.Portfolio.from_orders(price_data, orders)

# 动态订单生成
def order_func(context):
    if should_trade(context):
        return Order(size=calculate_size(context))
    return NoOrder
```

技术细节:
- 系统在处理时会自动识别并跳过NoOrder
- 不会产生任何记录或状态变化
- 不会消耗计算资源进行无意义的处理
- 可以安全地在任何需要Order对象的地方使用

与None的区别:
- NoOrder是一个有效的Order对象，只是参数无效
- None表示缺少订单对象，可能导致类型错误
- NoOrder提供了更明确的语义和更好的类型安全性
"""


class OrderResult(tp.NamedTuple):
    """
    订单执行结果数据结构
    
    记录单个订单执行完成后的结果信息。这是订单处理流程的输出，
    包含了执行的详细结果和状态信息，用于后续的状态更新和分析。
    
    该结构体是订单执行引擎与上层逻辑之间的接口，提供了
    标准化的执行结果表示方式。
    """
    size: float    # 实际成交大小
    price: float   # 实际成交价格（含滑点调整）
    fees: float    # 实际支付的总手续费
    side: int      # 实际执行的订单方向（OrderSide枚举）
    status: int    # 订单执行状态（OrderStatus枚举）
    status_info: int # 订单状态详细信息（OrderStatusInfo枚举）


__pdoc__['OrderResult'] = """订单执行结果数据结构

该命名元组表示单个订单的执行结果，包含了执行过程的所有关键信息。
它是订单处理引擎的输出，用于更新投资组合状态和生成执行记录。

**数据流程:**
1. 订单请求（Order）→ 执行引擎处理
2. 执行引擎 → 生成执行结果（OrderResult）
3. 执行结果 → 更新投资组合状态
4. 执行结果 → 生成订单记录和日志

**结果分析:**
通过OrderResult可以分析：
- 订单是否成功执行
- 实际成交与预期的差异
- 执行成本和滑点影响
- 失败原因和改进方向

使用示例:
```python
# 通常从投资组合执行过程中获得
# 以下是概念性示例，展示结果的结构

# 成功执行的订单结果
successful_result = OrderResult(
    size=100.0,           # 成交100股
    price=50.25,          # 成交价格50.25（含滑点）
    fees=1.25,           # 总手续费1.25
    side=OrderSide.Buy,   # 买入方向
    status=OrderStatus.Filled,  # 成功成交
    status_info=0        # 无额外状态信息
)

# 被拒绝的订单结果
rejected_result = OrderResult(
    size=0.0,            # 未成交
    price=0.0,           # 无成交价格
    fees=0.0,            # 无手续费
    side=OrderSide.Buy,   # 原始买入方向
    status=OrderStatus.Rejected,  # 被拒绝
    status_info=OrderStatusInfo.NoCashLong  # 资金不足
)

# 部分成交的订单结果
partial_result = OrderResult(
    size=80.0,           # 只成交80股（原计划100股）
    price=50.30,         # 成交价格
    fees=1.00,           # 按实际成交计算的手续费
    side=OrderSide.Buy,   # 买入方向
    status=OrderStatus.Filled,  # 仍然算作成交
    status_info=OrderStatusInfo.PartialFill  # 部分成交信息
)

# 结果分析
def analyze_result(result):
    if result.status == OrderStatus.Filled:
        print(f"成交{result.size}股，价格{result.price}，费用{result.fees}")
    elif result.status == OrderStatus.Rejected:
        reason = OrderStatusInfo._fields[result.status_info]
        print(f"订单被拒绝，原因：{reason}")
    else:
        print("订单被忽略")
```

状态组合分析:
- status + status_info 提供完整的执行状态信息
- size + price + fees 提供具体的执行数据
- side 确认实际的执行方向

注意事项:
- 被拒绝或忽略的订单，size通常为0
- price已经包含滑点调整
- fees是最终实际支付的总费用
- status_info提供了失败的具体原因
"""

__pdoc__['OrderResult.size'] = """实际成交大小

订单的实际成交数量，可能与请求大小不同。

**取值情况:**
- 成功订单：实际成交的数量
- 部分成交：部分成交的数量（小于请求大小）
- 被拒绝/忽略订单：通常为0

**影响因素:**
- 资金限制：可用资金不足导致的减少
- 仓位限制：当前仓位限制导致的调整
- 最大大小限制：max_size参数的约束
- 粒度调整：size_granularity导致的取整

使用示例:
```python
# 分析成交情况
if result.size > 0:
    fill_rate = result.size / requested_size
    print(f"成交率: {fill_rate:.2%}")
else:
    print("未成交")
```
"""

__pdoc__['OrderResult.price'] = """实际成交价格（每单位，已调整滑点）

订单的实际成交价格，已经包含了滑点的影响。

**价格构成:**
- 基础价格：原始请求价格或市价
- 滑点调整：根据slippage参数调整
- 方向影响：买入时价格上升，卖出时价格下降

**计算公式:**
```
实际价格 = 基础价格 × (1 + 滑点率 × 方向系数)
方向系数：买入为+1，卖出为-1
```

**特殊情况:**
- 被拒绝/忽略的订单：通常为0或NaN
- 市价单：使用收盘价加滑点调整
- 限价单：使用指定价格加滑点调整

使用示例:
```python
# 分析滑点影响
slippage_impact = result.price - original_price
print(f"滑点影响: {slippage_impact:.4f}")

# 计算总交易价值
trade_value = result.size * result.price
print(f"交易价值: {trade_value:.2f}")
```
"""

__pdoc__['OrderResult.fees'] = """实际支付的总手续费

订单执行时实际支付的手续费总额。

**费用构成:**
- 比例费用：基于交易价值的百分比费用
- 固定费用：每笔订单的固定费用
- 总费用 = 比例费用 + 固定费用

**计算方式:**
```
比例费用 = abs(size) × price × fees_rate
固定费用 = fixed_fees
总费用 = 比例费用 + 固定费用
```

**特殊情况:**
- 被拒绝/忽略的订单：通常为0
- 返佣交易：可能为负数（表示获得返佣）
- 部分成交：按实际成交量计算

使用示例:
```python
# 分析手续费影响
fee_rate = result.fees / (result.size * result.price) if result.size > 0 else 0
print(f"实际费率: {fee_rate:.4%}")

# 计算净交易价值
net_value = result.size * result.price - result.fees
print(f"净交易价值: {net_value:.2f}")
```
"""

__pdoc__['OrderResult.side'] = """实际执行的订单方向

参见 `OrderSide` 枚举的详细说明。

**取值范围:**
- `OrderSide.Buy` (0)：买入操作
- `OrderSide.Sell` (1)：卖出操作

**确定逻辑:**
根据最终的size符号和direction设置确定：
- 正数size通常对应Buy
- 负数size通常对应Sell
- 受direction参数限制影响

使用示例:
```python
if result.side == OrderSide.Buy:
    print("执行了买入操作")
elif result.side == OrderSide.Sell:
    print("执行了卖出操作")
```
"""

__pdoc__['OrderResult.status'] = """订单执行状态

参见 `OrderStatus` 枚举的详细说明。

**状态类型:**
- `OrderStatus.Filled` (0)：成功成交
- `OrderStatus.Ignored` (1)：被忽略
- `OrderStatus.Rejected` (2)：被拒绝

**状态意义:**
- Filled：订单成功执行，产生了实际的交易
- Ignored：订单被系统忽略，通常是策略逻辑决定
- Rejected：订单因各种限制无法执行

使用示例:
```python
if result.status == OrderStatus.Filled:
    print("订单成功执行")
elif result.status == OrderStatus.Rejected:
    print("订单被拒绝")
else:
    print("订单被忽略")
```
"""

__pdoc__['OrderResult.status_info'] = """订单状态详细信息

参见 `OrderStatusInfo` 枚举的详细说明。

**信息类型:**
提供订单被忽略或拒绝的具体原因：
- 资金相关：NoCashLong, NoCashShort
- 大小相关：SizeZero, MinSizeNotReached, MaxSizeExceeded
- 数据相关：SizeNaN, PriceNaN, ValueNaN
- 其他：RandomEvent, CantCoverFees, PartialFill等

**分析价值:**
- 识别策略中的问题
- 优化订单参数设置
- 改进风险管理机制
- 调试执行异常

使用示例:
```python
if result.status == OrderStatus.Rejected:
    reason = OrderStatusInfo._fields[result.status_info]
    print(f"拒绝原因: {reason}")
    
    # 针对性处理
    if result.status_info == OrderStatusInfo.NoCashLong:
        print("建议：减少订单大小或增加资金")
    elif result.status_info == OrderStatusInfo.MinSizeNotReached:
        print("建议：调整最小订单大小设置")
```
"""


class AdjustSLContext(tp.NamedTuple):
    i: int
    col: int
    position_now: float
    val_price_now: float
    init_i: int
    init_price: float
    curr_i: int
    curr_price: float
    curr_stop: float
    curr_trail: bool


__pdoc__['AdjustSLContext'] = "A named tuple representing the context for adjusting (trailing) stop loss."
__pdoc__['AdjustSLContext.i'] = """Index of the current row.

Has range `[0, target_shape[0])`."""
__pdoc__['AdjustSLContext.col'] = """Current column.

Has range `[0, target_shape[1])` and is always within `[from_col, to_col)`."""
__pdoc__['AdjustSLContext.position_now'] = "Latest position."
__pdoc__['AdjustSLContext.val_price_now'] = "Latest valuation price."
__pdoc__['AdjustSLContext.init_i'] = """Index of the row of the initial stop.

Doesn't change."""
__pdoc__['AdjustSLContext.init_price'] = """Price of the initial stop.

Doesn't change."""
__pdoc__['AdjustSLContext.curr_i'] = """Index of the row of the updated stop.

Gets updated once the price is updated."""
__pdoc__['AdjustSLContext.curr_price'] = """Current stop price.

Gets updated in trailing SL once a higher price is discovered."""
__pdoc__['AdjustSLContext.curr_stop'] = """Current stop value.

Can be updated by adjustment function."""
__pdoc__['AdjustSLContext.curr_trail'] = """Current trailing flag.

Can be updated by adjustment function."""


class AdjustTPContext(tp.NamedTuple):
    i: int
    col: int
    position_now: float
    val_price_now: float
    init_i: int
    init_price: float
    curr_stop: float


__pdoc__['AdjustTPContext'] = "A named tuple representing the context for adjusting take profit."
__pdoc__['AdjustTPContext.i'] = "See `AdjustSLContext.i`."
__pdoc__['AdjustTPContext.col'] = "See `AdjustSLContext.col`."
__pdoc__['AdjustTPContext.position_now'] = "See `AdjustSLContext.position_now`."
__pdoc__['AdjustTPContext.val_price_now'] = "See `AdjustSLContext.val_price_now`."
__pdoc__['AdjustTPContext.init_i'] = "See `AdjustSLContext.init_i`."
__pdoc__['AdjustTPContext.init_price'] = "See `AdjustSLContext.curr_price`."
__pdoc__['AdjustTPContext.curr_stop'] = "See `AdjustSLContext.curr_stop`."


class SignalContext(tp.NamedTuple):
    i: int
    col: int
    position_now: float
    val_price_now: float
    flex_2d: bool


__pdoc__['AdjustSLContext'] = "A named tuple representing the context for generation of signals."
__pdoc__['AdjustSLContext.i'] = """Index of the current row.

Has range `[0, target_shape[0])`."""
__pdoc__['AdjustSLContext.col'] = """Current column.

Has range `[0, target_shape[1])` and is always within `[from_col, to_col)`."""
__pdoc__['AdjustSLContext.position_now'] = "Latest position."
__pdoc__['AdjustSLContext.val_price_now'] = "Latest valuation price."
__pdoc__['AdjustSLContext.flex_2d'] = "See `vectorbt.base.reshape_fns.flex_select_auto_nb`."

# ############# Records ############# #

order_dt = np.dtype([
    ('id', np.int64),        # 订单唯一标识符
    ('col', np.int64),       # 列索引（资产索引）
    ('idx', np.int64),       # 时间索引（行索引）
    ('size', np.float64),    # 订单大小（实际成交数量）
    ('price', np.float64),   # 订单价格（实际成交价格）
    ('fees', np.float64),    # 手续费金额
    ('side', np.int64),      # 订单方向（买入/卖出）
], align=True)
"""订单记录数据类型，定义了订单记录的完整结构化数据格式"""

__pdoc__['order_dt'] = f"""订单记录数据类型

```json
{to_doc(order_dt)}
```

该NumPy数据类型定义了订单记录的结构化格式，用于存储投资组合模拟中每个订单的详细信息。
这种结构化设计确保了高效的内存使用和快速的数据访问。

字段详细说明:

**id (int64)**: 订单唯一标识符
- 每个订单的全局唯一ID，从0开始递增
- 用于订单追踪和关联分析
- 在整个模拟过程中保持唯一性

**col (int64)**: 列索引（资产索引）
- 指示订单所属的资产列
- 范围：[0, 资产数量-1]
- 用于多资产组合的订单分组和分析

**idx (int64)**: 时间索引（行索引）
- 订单执行的时间点索引
- 对应价格数据的行索引
- 用于时间序列分析和回测验证

**size (float64)**: 订单大小（实际成交数量）
- 订单的实际成交数量，而非请求数量
- 正数表示买入数量，负数表示卖出数量
- 可能因资金限制或部分成交而小于请求数量

**price (float64)**: 订单价格（实际成交价格）
- 订单的实际成交价格，包含滑点影响
- 用于计算交易成本和盈亏
- 可能与请求价格有差异

**fees (float64)**: 手续费金额
- 该订单产生的实际手续费
- 包括比例费用和固定费用的总和
- 影响净收益的重要成本因素

**side (int64)**: 订单方向
- 使用OrderSide枚举值：0=Buy（买入），1=Sell（卖出）
- 区分资金流入和流出方向
- 用于订单分类和统计分析

技术特点:
- 使用align=True优化内存对齐，提升访问性能
- 所有数值字段使用64位精度，确保计算准确性
- 紧凑的结构设计，最小化内存占用
- 与NumPy数组完全兼容，支持向量化操作

应用场景:
- 订单执行历史记录
- 交易成本分析
- 执行质量评估
- 策略回测验证
- 合规监控和审计

使用示例:
```python
import numpy as np
from vectorbt.portfolio.enums import order_dt

# 创建订单记录数组
orders = np.array([
    (0, 0, 10, 100.0, 50.25, 1.25, 0),  # 买入订单
    (1, 0, 15, -50.0, 52.10, 0.65, 1),  # 卖出订单
], dtype=order_dt)

# 访问订单信息
print(f"订单ID: {orders['id']}")
print(f"成交价格: {orders['price']}")
print(f"手续费: {orders['fees']}")

# 分析买卖订单
buy_orders = orders[orders['side'] == 0]
sell_orders = orders[orders['side'] == 1]
```
"""

_trade_fields = [
    ('id', np.int64),           # 交易唯一标识符
    ('col', np.int64),          # 列索引（资产索引）
    ('size', np.float64),       # 交易大小（持仓数量）
    ('entry_idx', np.int64),    # 入场时间索引
    ('entry_price', np.float64), # 平均入场价格
    ('entry_fees', np.float64), # 入场总手续费
    ('exit_idx', np.int64),     # 出场时间索引
    ('exit_price', np.float64), # 平均出场价格
    ('exit_fees', np.float64),  # 出场总手续费
    ('pnl', np.float64),        # 盈亏金额
    ('return', np.float64),     # 收益率
    ('direction', np.int64),    # 交易方向（多头/空头）
    ('status', np.int64),       # 交易状态（开放/关闭）
    ('parent_id', np.int64)     # 父交易ID（用于关联分析）
]

trade_dt = np.dtype(_trade_fields, align=True)
"""交易记录数据类型，定义了完整交易生命周期的结构化数据格式"""

__pdoc__['trade_dt'] = f"""交易记录数据类型

```json
{to_doc(trade_dt)}
```

该NumPy数据类型定义了交易记录的结构化格式，用于存储从开仓到平仓的完整交易信息。
与order_dt不同，trade_dt记录的是完整的交易周期，而不是单个订单操作。

字段详细说明:

**id (int64)**: 交易唯一标识符
- 每个交易的全局唯一ID，从0开始递增
- 用于交易追踪和绩效分析
- 区分不同的交易实例

**col (int64)**: 列索引（资产索引）
- 指示交易所属的资产列
- 范围：[0, 资产数量-1]
- 用于多资产组合的交易分组

**size (float64)**: 交易大小（持仓数量）
- 交易的最大持仓数量
- 正数表示多头交易，负数表示空头交易
- 反映交易的规模和风险暴露

**entry_idx (int64)**: 入场时间索引
- 交易开始（首次建仓）的时间点
- 对应价格数据的行索引
- 用于计算持仓时间

**entry_price (float64)**: 平均入场价格
- 所有入场订单的加权平均价格
- 考虑了分批建仓的情况
- 用于计算交易盈亏的成本基础

**entry_fees (float64)**: 入场总手续费
- 建仓过程中产生的所有手续费总和
- 包括多次加仓的累计费用
- 影响交易净收益的成本因素

**exit_idx (int64)**: 出场时间索引
- 交易结束（完全平仓）的时间点
- 对于未平仓交易为-1
- 用于计算实际持仓期间

**exit_price (float64)**: 平均出场价格
- 所有出场订单的加权平均价格
- 对于未平仓交易，使用当前估值价格
- 用于计算最终交易盈亏

**exit_fees (float64)**: 出场总手续费
- 平仓过程中产生的所有手续费总和
- 包括分批平仓的累计费用
- 对于未平仓交易可能为0

**pnl (float64)**: 盈亏金额
- 交易的绝对盈亏金额（包含手续费）
- 计算公式：(exit_price - entry_price) * size - entry_fees - exit_fees
- 正数表示盈利，负数表示亏损

**return (float64)**: 收益率
- 交易的相对收益率
- 计算公式：pnl / (entry_price * abs(size) + entry_fees)
- 用于标准化的绩效比较

**direction (int64)**: 交易方向
- 使用TradeDirection枚举值：0=Long（多头），1=Short（空头）
- 区分交易的基本策略方向
- 用于方向性绩效分析

**status (int64)**: 交易状态
- 使用TradeStatus枚举值：0=Open（开放），1=Closed（关闭）
- 区分活跃交易和历史交易
- 影响盈亏计算的方式

**parent_id (int64)**: 父交易ID
- 用于关联相关交易的父级标识
- 支持复杂的交易关系建模
- 便于分层交易分析

技术特点:
- 内存对齐优化，支持高效的向量化计算
- 完整的交易生命周期记录
- 支持未平仓交易的实时盈亏计算
- 兼容多种交易分析需求

应用场景:
- 交易绩效评估
- 风险收益分析
- 持仓时间统计
- 策略有效性验证
- 交易行为研究

使用示例:
```python
import numpy as np
from vectorbt.portfolio.enums import trade_dt, TradeDirection, TradeStatus

# 创建交易记录数组
trades = np.array([
    (0, 0, 100.0, 10, 50.25, 1.25, 15, 52.10, 0.65, 184.35, 0.0365, 0, 1, -1),
    (1, 0, -50.0, 20, 51.80, 0.65, -1, 51.50, 0.0, -15.0, -0.0058, 1, 0, -1),
], dtype=trade_dt)

# 分析交易表现
completed_trades = trades[trades['status'] == TradeStatus.Closed]
open_trades = trades[trades['status'] == TradeStatus.Open]

print(f"已完成交易数: {len(completed_trades)}")
print(f"未平仓交易数: {len(open_trades)}")

# 计算总盈亏
total_pnl = trades['pnl'].sum()
avg_return = trades[trades['status'] == TradeStatus.Closed]['return'].mean()

print(f"总盈亏: {total_pnl:.2f}")
print(f"平均收益率: {avg_return:.2%}")
```
"""

_log_fields = [
    # 基础标识字段
    ('id', np.int64),                    # 日志记录唯一标识符
    ('group', np.int64),                 # 资产组索引
    ('col', np.int64),                   # 列索引（资产索引）
    ('idx', np.int64),                   # 时间索引（行索引）
    
    # 执行前状态字段
    ('cash', np.float64),                # 执行前现金余额
    ('position', np.float64),            # 执行前持仓数量
    ('debt', np.float64),                # 执行前做空债务
    ('free_cash', np.float64),           # 执行前可用现金
    ('val_price', np.float64),           # 执行前资产估值价格
    ('value', np.float64),               # 执行前组合总价值
    
    # 订单请求参数字段
    ('req_size', np.float64),            # 请求的订单大小
    ('req_price', np.float64),           # 请求的订单价格
    ('req_size_type', np.int64),         # 请求的订单大小类型
    ('req_direction', np.int64),         # 请求的交易方向
    ('req_fees', np.float64),            # 请求的手续费率
    ('req_fixed_fees', np.float64),      # 请求的固定手续费
    ('req_slippage', np.float64),        # 请求的滑点率
    ('req_min_size', np.float64),        # 请求的最小订单大小
    ('req_max_size', np.float64),        # 请求的最大订单大小
    ('req_size_granularity', np.float64), # 请求的订单大小粒度
    ('req_reject_prob', np.float64),     # 请求的拒绝概率
    ('req_lock_cash', np.bool_),         # 请求的现金锁定标志
    ('req_allow_partial', np.bool_),     # 请求的部分成交允许标志
    ('req_raise_reject', np.bool_),      # 请求的拒绝异常标志
    ('req_log', np.bool_),               # 请求的日志记录标志
    
    # 执行后状态字段
    ('new_cash', np.float64),            # 执行后现金余额
    ('new_position', np.float64),        # 执行后持仓数量
    ('new_debt', np.float64),            # 执行后做空债务
    ('new_free_cash', np.float64),       # 执行后可用现金
    ('new_val_price', np.float64),       # 执行后资产估值价格
    ('new_value', np.float64),           # 执行后组合总价值
    
    # 执行结果字段
    ('res_size', np.float64),            # 实际执行的订单大小
    ('res_price', np.float64),           # 实际执行的订单价格
    ('res_fees', np.float64),            # 实际产生的手续费
    ('res_side', np.int64),              # 实际执行的订单方向
    ('res_status', np.int64),            # 订单执行状态
    ('res_status_info', np.int64),       # 订单状态详细信息
    ('order_id', np.int64)               # 关联的订单记录ID
]

log_dt = np.dtype(_log_fields, align=True)
"""日志记录数据类型，定义了订单执行过程的完整状态记录格式"""

__pdoc__['log_dt'] = f"""日志记录数据类型

```json
{to_doc(log_dt)}
```

该NumPy数据类型定义了日志记录的结构化格式，用于存储订单执行前后的完整状态信息。
这是vectorbt中最详细的记录类型，包含了订单处理的全过程数据，是调试和分析的重要工具。

字段分组说明:

**基础标识字段:**
- **id (int64)**: 日志记录唯一标识符，全局递增的记录ID
- **group (int64)**: 资产组索引，用于现金共享分组
- **col (int64)**: 列索引（资产索引），指示相关资产
- **idx (int64)**: 时间索引（行索引），记录发生的时间点

**执行前状态字段:**
- **cash (float64)**: 订单执行前的现金余额
- **position (float64)**: 订单执行前的持仓数量
- **debt (float64)**: 订单执行前的做空债务
- **free_cash (float64)**: 订单执行前的可用现金
- **val_price (float64)**: 订单执行前的资产估值价格
- **value (float64)**: 订单执行前的组合总价值

**订单请求字段（req_前缀）:**
- **req_size (float64)**: 请求的订单大小
- **req_price (float64)**: 请求的订单价格
- **req_size_type (int64)**: 请求的订单大小类型（SizeType枚举）
- **req_direction (int64)**: 请求的交易方向（Direction枚举）
- **req_fees (float64)**: 请求的手续费率
- **req_fixed_fees (float64)**: 请求的固定手续费
- **req_slippage (float64)**: 请求的滑点率
- **req_min_size (float64)**: 请求的最小订单大小
- **req_max_size (float64)**: 请求的最大订单大小
- **req_size_granularity (float64)**: 请求的订单大小粒度
- **req_reject_prob (float64)**: 请求的拒绝概率
- **req_lock_cash (bool)**: 请求的现金锁定标志
- **req_allow_partial (bool)**: 请求的部分成交允许标志
- **req_raise_reject (bool)**: 请求的拒绝异常标志
- **req_log (bool)**: 请求的日志记录标志

**执行后状态字段（new_前缀）:**
- **new_cash (float64)**: 订单执行后的现金余额
- **new_position (float64)**: 订单执行后的持仓数量
- **new_debt (float64)**: 订单执行后的做空债务
- **new_free_cash (float64)**: 订单执行后的可用现金
- **new_val_price (float64)**: 订单执行后的资产估值价格
- **new_value (float64)**: 订单执行后的组合总价值

**执行结果字段（res_前缀）:**
- **res_size (float64)**: 实际执行的订单大小
- **res_price (float64)**: 实际执行的订单价格
- **res_fees (float64)**: 实际产生的手续费
- **res_side (int64)**: 实际执行的订单方向（OrderSide枚举）
- **res_status (int64)**: 订单执行状态（OrderStatus枚举）
- **res_status_info (int64)**: 订单状态详细信息（OrderStatusInfo枚举）
- **order_id (int64)**: 关联的订单记录ID

设计特点:
- **完整性**: 记录订单处理的完整生命周期
- **可追溯性**: 每个字段都有明确的含义和用途
- **调试友好**: 提供详细的执行过程信息
- **高性能**: 优化的内存布局和数据类型
- **标准化**: 统一的字段命名和数据格式

应用场景:
- **策略调试**: 追踪订单执行异常和拒绝原因
- **性能分析**: 分析订单执行效率和成功率
- **风险监控**: 监控资金使用和仓位变化
- **合规审计**: 生成详细的交易执行报告
- **策略优化**: 基于执行数据优化策略参数

使用示例:
```python
import numpy as np
from vectorbt.portfolio.enums import log_dt, OrderStatus, OrderStatusInfo

# 访问日志记录（通常从Portfolio.logs获得）
# logs = portfolio.logs.records_arr  # 获取日志记录数组

# 分析订单执行状态
# filled_logs = logs[logs['res_status'] == OrderStatus.Filled]
# rejected_logs = logs[logs['res_status'] == OrderStatus.Rejected]

# 分析拒绝原因
# rejection_reasons = rejected_logs['res_status_info']
# cash_shortage = np.sum(rejection_reasons == OrderStatusInfo.NoCashLong)

# 计算状态变化
# cash_changes = logs['new_cash'] - logs['cash']
# position_changes = logs['new_position'] - logs['position']

# 分析执行效率
# execution_rate = len(filled_logs) / len(logs) if len(logs) > 0 else 0

print("日志记录数据结构已定义，用于详细的订单执行分析")
```

与其他数据类型的关系:
- **order_dt**: 日志记录可通过order_id关联到具体订单
- **trade_dt**: 多个日志记录可构成一个完整交易
- **ProcessOrderState**: 日志记录包含了状态对象的所有信息

注意事项:
- 日志记录会显著增加内存使用，建议只在调试时启用
- 大规模回测时应适当限制max_logs参数
- 所有状态字段都经过精度验证和一致性检查
- 支持实时和批量的日志分析操作
"""
