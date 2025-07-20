# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT SIGNALS ENUMS MODULE: 信号系统枚举类型定义模块
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中signals模块的核心枚举定义文件，为整个信号系统
提供了标准化的类型定义和常量管理。该模块采用命名元组(NamedTuple)的设计模式，
确保了类型安全性和代码可读性，是vectorbt信号处理系统的重要基础设施。

核心设计理念：
1. **类型安全**：使用NamedTuple确保枚举值的类型安全和不可变性
2. **语义清晰**：通过直观的命名约定表达业务逻辑含义
3. **系统集成**：与vectorbt其他模块无缝集成，提供统一的类型接口
4. **扩展友好**：支持未来功能扩展，保持向后兼容性

主要功能模块：

【止损类型枚举 - StopType】
定义量化交易中常见的止损策略类型，为风险管理提供标准化的类型支持：
- StopLoss: 固定止损，基于入场价格的固定百分比止损
- TrailStop: 追踪止损，随价格有利移动而动态调整的止损
- TakeProfit: 止盈策略，达到目标收益时的获利了结

【工厂模式枚举 - FactoryMode】  
定义信号生成器的四种工作模式，支持不同的信号生成策略：
- Entries: 仅生成入场信号，适用于只需要买入信号的策略
- Exits: 仅生成出场信号，需要基于现有入场信号生成对应的出场
- Both: 同时生成入场和出场信号，适用于完整的交易策略
- Chain: 链式处理模式，基于输入信号生成新的入场和出场信号

技术特点：
- **类型安全**：使用NamedTuple确保枚举值的不可变性和类型检查
- **文档集成**：与vectorbt文档系统深度集成，支持自动文档生成
- **性能优化**：轻量级设计，零运行时开销
- **IDE支持**：完整的类型注解，支持智能提示和静态检查

应用场景：
- **信号工厂配置**：为SignalFactory提供模式选择支持
- **止损策略定义**：为风险管理模块提供标准化的止损类型
- **代码可读性**：通过语义化的枚举值提高代码可维护性
- **API一致性**：确保vectorbt各模块间的接口一致性

与vectorbt生态系统的关系：
- 为signals.factory模块提供模式枚举支持
- 与signals.nb模块的止损函数协同工作
- 支持signals.generators模块的生成器配置
- 与portfolio模块的风险管理功能集成

使用示例：
```python
import vectorbt as vbt
from vectorbt.signals.enums import StopType, FactoryMode

# 使用止损类型枚举
print(f"固定止损类型: {StopType.StopLoss}")  # 输出: 0
print(f"追踪止损类型: {StopType.TrailStop}")  # 输出: 1
print(f"止盈类型: {StopType.TakeProfit}")     # 输出: 2

# 使用工厂模式枚举
print(f"仅入场模式: {FactoryMode.Entries}")   # 输出: 0
print(f"仅出场模式: {FactoryMode.Exits}")     # 输出: 1
print(f"双向模式: {FactoryMode.Both}")        # 输出: 2
print(f"链式模式: {FactoryMode.Chain}")       # 输出: 3

# 在信号工厂中使用
factory = vbt.SignalFactory(mode=FactoryMode.Both)
signals = factory.from_choice_func(
    entry_choice_func=my_entry_func,
    exit_choice_func=my_exit_func
)
```

该模块是vectorbt信号系统的基础类型定义，为整个量化交易框架提供了
标准化、类型安全的枚举支持，是构建可靠量化交易系统的重要组件。
================================================================================

命名元组和枚举类型定义。

为 `vectorbt.signals` 定义枚举和其他模式。
"""

from vectorbt import _typing as tp  # 导入vectorbt类型系统，提供类型注解支持
from vectorbt.utils.docs import to_doc  # 导入文档工具函数，用于生成枚举的JSON文档

__all__ = [
    'StopType',      # 止损类型枚举，定义不同的止损策略类型
    'FactoryMode'    # 工厂模式枚举，定义信号生成器的工作模式
]

__pdoc__ = {}  # 文档控制字典，用于控制自动文档生成


# ############# 枚举类型定义 ############# #


class StopTypeT(tp.NamedTuple):
    """
    止损类型命名元组类
    
    功能概述：
    定义量化交易中常见的止损策略类型，为风险管理提供标准化的类型支持。
    使用NamedTuple确保类型安全性和不可变性，是vectorbt止损系统的核心类型定义。
    
    枚举值说明：
        StopLoss (int = 0): 固定止损类型
            - 基于入场价格的固定百分比止损
            - 止损位在整个持仓期间保持不变
            - 适用于震荡市场或需要严格风险控制的策略
            
        TrailStop (int = 1): 追踪止损类型  
            - 随价格有利移动而动态调整的止损
            - 止损位只能向有利方向移动，不能回撤
            - 适用于趋势市场，能够锁定更多利润
            
        TakeProfit (int = 2): 止盈类型
            - 达到目标收益时的获利了结策略
            - 基于入场价格计算目标盈利价位
            - 用于控制风险收益比，实现利润保护
    
    技术特点：
        - 使用NamedTuple确保类型安全和不可变性
        - 整数枚举值便于存储和序列化
        - 与vectorbt文档系统深度集成
        - 支持IDE智能提示和静态类型检查
    
    使用场景：
        - 止损策略的类型标识和分类
        - 风险管理系统的策略选择
        - 信号生成器的参数配置
        - 回测结果的止损类型分析
    
    示例用法：
        ```python
        from vectorbt.signals.enums import StopType
        
        # 获取止损类型值
        fixed_stop = StopType.StopLoss      # 0
        trail_stop = StopType.TrailStop     # 1
        take_profit = StopType.TakeProfit   # 2
        
        # 在止损逻辑中使用
        if stop_type == StopType.StopLoss:
            # 执行固定止损逻辑
            stop_price = entry_price * (1 - stop_percent)
        elif stop_type == StopType.TrailStop:
            # 执行追踪止损逻辑
            stop_price = max_price * (1 - stop_percent)
        elif stop_type == StopType.TakeProfit:
            # 执行止盈逻辑
            stop_price = entry_price * (1 + profit_percent)
        
        # 在信号生成器中使用
        ohlc_stop = vbt.OHLCSTX.run(
            entries, open, high, low, close,
            sl_stop=0.05,  # 5%止损
            tp_stop=0.10   # 10%止盈
        )
        
        # 分析止损类型分布
        stop_types = ohlc_stop.stop_type_readable
        stop_loss_count = (stop_types == 'StopLoss').sum()
        trail_stop_count = (stop_types == 'TrailStop').sum()
        take_profit_count = (stop_types == 'TakeProfit').sum()
        ```
    
    与其他模块的关系：
        - 与signals.nb模块的止损函数协同工作
        - 为signals.generators模块的OHLCST系列提供类型支持
        - 与portfolio模块的风险管理功能集成
        - 支持signals.accessors模块的止损分析功能
    
    注意事项：
        - 枚举值是整数类型，便于高效存储和比较
        - 使用NamedTuple确保类型安全，避免运行时错误
        - 与vectorbt的文档系统集成，支持自动生成API文档
        - 建议在代码中使用枚举值而非硬编码数字
    """
    StopLoss: int = 0    # 固定止损类型：基于入场价格的固定百分比止损
    TrailStop: int = 1   # 追踪止损类型：随价格有利移动而动态调整的止损
    TakeProfit: int = 2  # 止盈类型：达到目标收益时的获利了结策略


StopType = StopTypeT()  # 创建止损类型枚举实例，提供全局访问点
"""止损类型枚举实例

提供量化交易中常见止损策略类型的标准化定义。

```json
{to_doc(StopType)}
```

枚举值说明：
- **StopLoss (0)**: 固定止损类型
  - 基于入场价格的固定百分比止损
  - 止损位在整个持仓期间保持不变
  - 适用于震荡市场或需要严格风险控制的策略
  
- **TrailStop (1)**: 追踪止损类型
  - 随价格有利移动而动态调整的止损
  - 止损位只能向有利方向移动，不能回撤
  - 适用于趋势市场，能够锁定更多利润
  
- **TakeProfit (2)**: 止盈类型
  - 达到目标收益时的获利了结策略
  - 基于入场价格计算目标盈利价位
  - 用于控制风险收益比，实现利润保护

使用示例：
```python
import vectorbt as vbt
from vectorbt.signals.enums import StopType

# 创建OHLC止损信号生成器
ohlc_stop = vbt.OHLCSTX.run(
    entries, open, high, low, close,
    sl_stop=0.05,  # 5%固定止损
    sl_trail=False,  # 不使用追踪止损
    tp_stop=0.10   # 10%止盈
)

# 分析止损类型分布
stop_types = ohlc_stop.stop_type_readable
print(f"固定止损触发次数: {(stop_types == 'StopLoss').sum()}")
print(f"追踪止损触发次数: {(stop_types == 'TrailStop').sum()}")
print(f"止盈触发次数: {(stop_types == 'TakeProfit').sum()}")
```
"""

__pdoc__['StopType'] = f"""止损类型枚举。

```json
{to_doc(StopType)}
```
"""


class FactoryModeT(tp.NamedTuple):
    """
    工厂模式命名元组类
    
    功能概述：
    定义信号生成器的四种工作模式，为SignalFactory提供标准化的模式选择支持。
    这些模式决定了信号生成器如何生成和处理入场与出场信号，是vectorbt信号系统
    的核心配置参数。
    
    枚举值说明：
        Entries (int = 0): 仅生成入场信号模式
            - 只使用generate_func生成入场信号
            - 不接收任何输入信号数组
            - 产生一个输出信号数组：entries
            - 此类生成器通常没有后缀标识
            
        Exits (int = 1): 仅生成出场信号模式
            - 只使用generate_ex_func生成出场信号
            - 接收一个输入信号数组：entries（入场信号）
            - 产生一个输出信号数组：exits
            - 此类生成器通常有后缀'X'标识
            
        Both (int = 2): 同时生成入场和出场信号模式
            - 使用generate_enex_func同时生成入场和出场信号
            - 不接收任何输入信号数组
            - 产生两个输出信号数组：entries和exits
            - 此类生成器通常有后缀'NX'标识
            
        Chain (int = 3): 链式处理模式
            - 使用generate_enex_func进行链式信号处理
            - 接收一个输入信号数组：entries（入场信号）
            - 产生两个输出信号数组：new_entries和exits
            - 此类生成器通常有后缀'CX'标识
    
    技术特点：
        - 使用NamedTuple确保类型安全和不可变性
        - 整数枚举值便于存储和序列化
        - 与SignalFactory深度集成，支持模式切换
        - 提供清晰的信号生成逻辑分离
    
    使用场景：
        - SignalFactory的模式配置
        - 信号生成器的行为控制
        - 复杂信号策略的构建
        - 信号处理管道的设计
    
    示例用法：
        ```python
        from vectorbt.signals.enums import FactoryMode
        from vectorbt.signals.factory import SignalFactory
        
        # 模式1：仅生成入场信号
        entry_factory = SignalFactory(mode=FactoryMode.Entries)
        EntrySignals = entry_factory.from_choice_func(
            entry_choice_func=my_entry_func
        )
        entries = EntrySignals.run(input_shape=(100, 3))
        
        # 模式2：仅生成出场信号（需要入场信号作为输入）
        exit_factory = SignalFactory(mode=FactoryMode.Exits)
        ExitSignals = exit_factory.from_choice_func(
            exit_choice_func=my_exit_func
        )
        exits = ExitSignals.run(entries)  # 基于入场信号生成出场信号
        
        # 模式3：同时生成入场和出场信号
        both_factory = SignalFactory(mode=FactoryMode.Both)
        BothSignals = both_factory.from_choice_func(
            entry_choice_func=my_entry_func,
            exit_choice_func=my_exit_func
        )
        result = BothSignals.run(input_shape=(100, 3))
        entries = result.entries
        exits = result.exits
        
        # 模式4：链式处理模式
        chain_factory = SignalFactory(mode=FactoryMode.Chain)
        ChainSignals = chain_factory.from_choice_func(
            exit_choice_func=my_exit_func  # 只需要出场函数
        )
        new_entries, exits = ChainSignals.run(entries)  # 返回新的入场信号和出场信号
        ```
    
    与其他模块的关系：
        - 与SignalFactory类深度集成，控制信号生成行为
        - 影响signals.nb模块中不同生成函数的调用
        - 决定signals.generators模块中生成器的后缀命名
        - 与portfolio模块的信号处理逻辑协同工作
    
    注意事项：
        - 不同模式对输入输出数组的要求不同
        - 模式选择会影响信号生成器的性能和复杂度
        - 建议根据具体需求选择最合适的模式
        - 模式一旦设置，在SignalFactory实例化后不可更改
    """
    Entries: int = 0  # 仅生成入场信号模式：只使用generate_func生成入场信号
    Exits: int = 1    # 仅生成出场信号模式：只使用generate_ex_func生成出场信号
    Both: int = 2     # 同时生成入场和出场信号模式：使用generate_enex_func同时生成
    Chain: int = 3    # 链式处理模式：使用generate_enex_func进行链式信号处理


FactoryMode = FactoryModeT()  # 创建工厂模式枚举实例，提供全局访问点
"""工厂模式枚举实例

定义信号生成器的四种工作模式，控制信号生成器的行为逻辑。

```json
{to_doc(FactoryMode)}
```

模式详解：

**Entries (0)**: 仅生成入场信号模式
- 使用 `generate_func` 生成入场信号
- 不接收任何输入信号数组
- 产生一个输出信号数组：`entries`
- 此类生成器通常没有后缀标识
- 适用场景：只需要买入信号的策略，如趋势跟踪策略

**Exits (1)**: 仅生成出场信号模式  
- 使用 `generate_ex_func` 生成出场信号
- 接收一个输入信号数组：`entries`（入场信号）
- 产生一个输出信号数组：`exits`
- 此类生成器通常有后缀'X'标识
- 适用场景：基于现有入场信号生成出场信号，如止损策略

**Both (2)**: 同时生成入场和出场信号模式
- 使用 `generate_enex_func` 同时生成入场和出场信号
- 不接收任何输入信号数组
- 产生两个输出信号数组：`entries` 和 `exits`
- 此类生成器通常有后缀'NX'标识
- 适用场景：完整的交易策略，如均值回归策略

**Chain (3)**: 链式处理模式
- 使用 `generate_enex_func` 进行链式信号处理
- 接收一个输入信号数组：`entries`
- 产生两个输出信号数组：`new_entries` 和 `exits`
- 此类生成器通常有后缀'CX'标识
- 适用场景：信号过滤和重组，如信号去重和优化

使用示例：
```python
import vectorbt as vbt
from vectorbt.signals.enums import FactoryMode

# 示例1：仅生成入场信号
entry_factory = vbt.SignalFactory(mode=FactoryMode.Entries)
EntrySignals = entry_factory.from_choice_func(
    entry_choice_func=lambda from_i, to_i, col: np.array([from_i])
)
entries = EntrySignals.run(input_shape=(100, 3))

# 示例2：仅生成出场信号
exit_factory = vbt.SignalFactory(mode=FactoryMode.Exits)
ExitSignals = exit_factory.from_choice_func(
    exit_choice_func=lambda from_i, to_i, col: np.array([from_i + 5])
)
exits = ExitSignals.run(entries)  # 基于入场信号生成出场信号

# 示例3：同时生成入场和出场信号
both_factory = vbt.SignalFactory(mode=FactoryMode.Both)
BothSignals = both_factory.from_choice_func(
    entry_choice_func=lambda from_i, to_i, col: np.array([from_i]),
    exit_choice_func=lambda from_i, to_i, col: np.array([from_i + 10])
)
result = BothSignals.run(input_shape=(100, 3))
print(f"入场信号数: {result.entries.sum()}")
print(f"出场信号数: {result.exits.sum()}")

# 示例4：链式处理模式
chain_factory = vbt.SignalFactory(mode=FactoryMode.Chain)
ChainSignals = chain_factory.from_choice_func(
    exit_choice_func=lambda from_i, to_i, col: np.array([from_i + 3])
)
new_entries, exits = ChainSignals.run(entries)
print(f"原始入场信号数: {entries.sum()}")
print(f"新入场信号数: {new_entries.sum()}")
print(f"出场信号数: {exits.sum()}")
```

模式选择建议：
- **Entries**: 适用于只需要入场信号的简单策略
- **Exits**: 适用于基于技术指标或时间规则的出场策略
- **Both**: 适用于完整的交易策略，需要同时控制入场和出场
- **Chain**: 适用于信号优化和过滤，如去除过于频繁的信号
"""

__pdoc__['FactoryMode'] = f"""工厂模式枚举。

```json
{to_doc(FactoryMode)}
```

模式详解：

**Entries**: 仅生成入场信号模式
- 使用 `generate_func` 生成入场信号
- 不接收任何输入信号数组
- 产生一个输出信号数组：`entries`
- 此类生成器通常没有后缀标识

**Exits**: 仅生成出场信号模式
- 使用 `generate_ex_func` 生成出场信号
- 接收一个输入信号数组：`entries`
- 产生一个输出信号数组：`exits`
- 此类生成器通常有后缀'X'标识

**Both**: 同时生成入场和出场信号模式
- 使用 `generate_enex_func` 同时生成入场和出场信号
- 不接收任何输入信号数组
- 产生两个输出信号数组：`entries` 和 `exits`
- 此类生成器通常有后缀'NX'标识

**Chain**: 链式处理模式
- 使用 `generate_enex_func` 进行链式信号处理
- 接收一个输入信号数组：`entries`
- 产生两个输出信号数组：`new_entries` 和 `exits`
- 此类生成器通常有后缀'CX'标识
"""
