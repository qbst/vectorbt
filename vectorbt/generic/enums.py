# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT GENERIC MODULE: ENUMERATED TYPES AND DATA SCHEMAS
================================================================================

文件作用概述：
本文件是vectorbt量化交易框架中枚举类型和数据结构定义的核心模块，专门为范围分析（Range Analysis）
和回撤分析（Drawdown Analysis）提供标准化的数据表示格式。该模块通过定义命名元组枚举和结构化
数据类型，为vectorbt框架的记录系统（Records System）奠定了类型安全的基础。

核心设计逻辑：
1. **状态枚举标准化**：通过RangeStatus和DrawdownStatus枚举，为时间序列分析中的状态转换
   提供标准化表示，确保整个框架对数据状态理解的一致性。

2. **结构化记录定义**：使用NumPy的dtype定义range_dt和drawdown_dt，创建内存对齐的
   结构化数组格式，为高性能数据处理和Numba编译优化提供支持。

3. **类型安全保障**：通过命名元组（NamedTuple）提供编译时类型检查，避免魔法数字的使用，
   提高代码可读性和维护性。

4. **性能优化设计**：所有数据类型都使用align=True进行内存对齐，为vectorbt的
   高频数据处理和并行计算提供最佳性能。
"""

# 导入Python标准库中的数据类型操作模块
import numpy as np  # 导入NumPy库，用于高性能数值计算和数组操作

# 导入vectorbt的类型定义模块，提供完整的类型注解支持
from vectorbt import _typing as tp
# 导入vectorbt的文档生成工具，用于自动生成API文档
from vectorbt.utils.docs import to_doc

# 定义模块的公开API，这些是可以被外部导入使用的对象
# 这种方式明确了模块的公共接口，提高API的可维护性
__all__ = [
    'RangeStatus',      # 范围状态枚举：标识时间范围的开放/关闭状态
    'DrawdownStatus',   # 回撤状态枚举：标识回撤的活跃/恢复状态
    'drawdown_dt',      # 回撤记录数据类型：定义回撤分析的结构化数据格式
    'range_dt'          # 范围记录数据类型：定义范围分析的结构化数据格式
]

# 文档配置字典，用于自定义特定对象的文档生成行为
__pdoc__ = {}


# ############# Enums ############# #
# 枚举类型定义部分：为vectorbt的状态管理提供类型安全的枚举

class RangeStatusT(tp.NamedTuple):
    """
    范围状态类型定义类 - 用于定义时间范围的状态枚举
    
    该类使用命名元组（NamedTuple）实现类型安全的枚举，为vectorbt框架中的
    范围分析功能提供标准化的状态表示。在量化交易分析中，经常需要识别和
    分析各种时间范围，如趋势持续期、信号有效期、价格区间等。
    
    设计优势：
    - 类型安全：使用NamedTuple提供编译时类型检查
    - 内存高效：整数枚举减少内存占用和提高比较性能
    - 语义清晰：命名常量避免魔法数字，提高代码可读性
    - Numba兼容：支持Numba JIT编译，确保高性能计算
    
    状态语义：
    - Open: 表示范围仍在进行中，终点尚未确定
    - Closed: 表示范围已经结束，起止点都已确定
    
    应用示例：
    - 趋势分析：标识一个上升趋势是否还在继续
    - 信号分析：标识一个交易信号是否还有效
    - 区间分析：标识一个价格区间是否已经突破
    """
    Open: int     # 开放状态(值为0)：表示范围正在进行中，结束时间点尚未确定
    Closed: int   # 关闭状态(值为1)：表示范围已经结束，具有明确的起止时间点


# 创建RangeStatus枚举实例，使用range(2)生成连续整数值(0, 1)
# 这种设计确保了枚举值的连续性和可预测性，便于数组索引和条件判断
RangeStatus = RangeStatusT(*range(2))
"""
范围状态枚举常量 - vectorbt框架中范围分析的核心状态标识

该枚举定义了时间范围的两种基本状态，广泛应用于vectorbt的ranges模块中。
在量化分析中，时间范围的状态识别对于正确计算持续时间、覆盖率等指标至关重要。

枚举值说明：
- RangeStatus.Open (0): 开放状态
  * 表示范围仍在进行，终止时间未确定
  * 在计算持续时间时需要特殊处理（包含当前时间点）
  * 常见于实时分析中正在发生的事件

- RangeStatus.Closed (1): 关闭状态  
  * 表示范围已经结束，具有明确的起止时间
  * 可以精确计算持续时间和相关统计指标
  * 历史分析中的完整事件记录

使用示例：
```python
import vectorbt as vbt
import pandas as pd

# 创建布尔信号序列（True表示信号激活）
signals = pd.Series([True, True, False, False, True, True, True])

# 生成范围记录，识别信号的连续区间
ranges = vbt.Ranges.from_ts(signals)

# 查看范围状态
print(ranges.status.values)
# 输出: [1 1] (所有范围都是Closed状态，因为序列已结束)

# 检查特定状态的范围
closed_ranges = ranges.status == vbt.RangeStatus.Closed
print(f"关闭范围数量: {closed_ranges.sum()}")

# 实时场景：最后一个范围可能是Open状态
live_signals = pd.Series([False, True, True, True])  # 正在进行的信号
live_ranges = vbt.Ranges.from_ts(live_signals)
print(f"最后范围状态: {live_ranges.status.values[-1]}")
# 可能输出: 0 (Open状态，信号仍在继续)
```

性能特点：
- 整数比较：枚举值为整数，支持高效的数值比较操作
- 内存对齐：在结构化数组中实现最佳内存布局
- Numba兼容：支持JIT编译，在循环中实现零开销抽象
"""

# 配置RangeStatus的文档生成，使用to_doc工具自动生成JSON格式的文档
__pdoc__['RangeStatus'] = f"""Range status.

```json
{to_doc(RangeStatus)}
```
"""


class DrawdownStatusT(tp.NamedTuple):
    """
    回撤状态类型定义类 - 用于定义回撤分析的状态枚举
    
    该类为vectorbt框架中的回撤分析功能提供状态标识。回撤分析是量化交易中
    风险管理的核心组成部分，用于衡量投资组合或策略的最大损失和恢复能力。
    
    设计目标：
    - 风险量化：精确识别和量化投资风险
    - 状态跟踪：实时跟踪回撤的发展和恢复过程
    - 性能评估：为策略评估提供关键的风险指标
    - 预警机制：为风险控制提供状态变化的及时通知
    
    状态定义：
    - Active: 回撤正在进行中，投资组合价值尚未恢复到历史最高点
    - Recovered: 回撤已经恢复，投资组合价值已达到或超过历史最高点
    
    应用场景：
    - 投资组合风险监控：实时监控投资组合的回撤状态
    - 策略性能评估：评估交易策略的风险控制能力
    - 风险预警系统：当回撤达到阈值时触发预警
    - 历史回撤分析：分析历史上的回撤事件和恢复时间
    """
    Active: int      # 活跃状态(值为0)：回撤正在进行中，价值尚未恢复到峰值
    Recovered: int   # 恢复状态(值为1)：回撤已经恢复，价值达到或超过了历史峰值


# 创建DrawdownStatus枚举实例，使用range(2)生成连续整数值(0, 1)
# 整数编码便于在NumPy数组中高效存储和处理
DrawdownStatus = DrawdownStatusT(*range(2))
"""
回撤状态枚举常量 - vectorbt框架中回撤分析的状态标识系统

该枚举是量化风险管理的核心组件，提供了回撤生命周期的标准化状态定义。
回撤分析是评估投资策略风险和收益特征的重要工具。

枚举值详解：
- DrawdownStatus.Active (0): 活跃回撤状态
  * 表示投资组合正经历价值回撤
  * 当前价值低于历史最高水位
  * 需要持续监控和风险管理
  * 常用于实时风险预警系统

- DrawdownStatus.Recovered (1): 恢复完成状态
  * 表示投资组合已从回撤中恢复
  * 当前价值达到或超过历史峰值
  * 可以计算完整的回撤周期指标
  * 用于历史回撤事件分析

实际应用示例：
```python
import vectorbt as vbt
import pandas as pd
import numpy as np

# 模拟投资组合价值序列
portfolio_values = pd.Series([
    100, 110, 105, 95, 85, 90, 100, 115, 108, 120
], index=pd.date_range('2023-01-01', periods=10, freq='D'))

# 生成回撤分析
drawdowns = vbt.Drawdowns.from_ts(portfolio_values)

# 查看回撤记录
print("回撤记录:")
print(drawdowns.records_readable)

# 分析活跃回撤
active_drawdowns = drawdowns.active
print(f"当前活跃回撤数量: {len(active_drawdowns.records)}")

# 分析已恢复回撤
recovered_drawdowns = drawdowns.recovered  
print(f"已恢复回撤数量: {len(recovered_drawdowns.records)}")

# 获取最大回撤信息
max_dd = drawdowns.max_drawdown()
print(f"最大回撤: {max_dd:.2%}")

# 风险监控：检查当前是否存在活跃回撤
current_status = drawdowns.records['status'][-1] if len(drawdowns.records) > 0 else None
if current_status == vbt.DrawdownStatus.Active:
    print("⚠️ 警告：投资组合当前处于回撤状态")
else:
    print("✅ 投资组合当前无活跃回撤")

# 回撤恢复分析
recovery_times = drawdowns.recovery_duration.mean()
print(f"平均恢复时间: {recovery_times}")
```

性能优化特性：
- 整数存储：枚举值使用整数，减少内存占用和提高比较速度
- 向量化操作：支持NumPy向量化操作，适合大规模数据分析
- Numba加速：兼容Numba JIT编译，在循环中实现最高性能
- 内存对齐：在结构化数组中保证最佳内存访问模式
"""

# 配置DrawdownStatus的文档生成
__pdoc__['DrawdownStatus'] = f"""Drawdown status.

```json
{to_doc(DrawdownStatus)}
```
"""

# ############# Records ############# #
# 记录数据类型定义部分：为vectorbt的结构化数据存储提供Schema定义

# 定义范围记录的数据类型（range_dt）
# 使用NumPy的dtype创建结构化数组格式，这是vectorbt记录系统的核心数据结构
range_dt = np.dtype([
    ('id', np.int64),        # 记录唯一标识符：用于区分不同的范围记录，支持记录关联和查询
    ('col', np.int64),       # 列索引：指示该范围属于哪一列数据，支持多资产/多策略并行分析
    ('start_idx', np.int64), # 起始索引：范围开始的时间点索引，对应时间序列的位置
    ('end_idx', np.int64),   # 结束索引：范围结束的时间点索引，用于计算持续时间
    ('status', np.int64)     # 状态标识：使用RangeStatus枚举值，标识范围的开放/关闭状态
], align=True)  # align=True确保字段在内存中对齐，提高访问性能和缓存效率
"""
范围记录数据类型 - vectorbt框架中时间范围分析的标准化数据结构

该数据类型定义了范围记录的完整schema，为vectorbt的ranges模块提供底层数据支持。
结构化数组设计确保了高性能的数据处理和内存效率。

字段详细说明：
- id (int64): 记录唯一标识符
  * 作用：为每个范围记录分配唯一ID，便于记录的索引和关联
  * 范围：从0开始的连续整数，按创建顺序递增
  * 用途：记录查询、关联分析、数据完整性验证

- col (int64): 列索引
  * 作用：标识该范围记录属于哪一列数据
  * 范围：0到数据框列数减1
  * 用途：多资产分析、列级别统计、分组计算

- start_idx (int64): 起始时间索引
  * 作用：记录范围开始的时间点在原始时间序列中的位置
  * 范围：0到时间序列长度减1
  * 用途：时间定位、持续时间计算、时间窗口分析

- end_idx (int64): 结束时间索引
  * 作用：记录范围结束的时间点在原始时间序列中的位置
  * 范围：start_idx到时间序列长度减1
  * 注意：对于Open状态的范围，end_idx指向当前最后一个时间点

- status (int64): 状态标识
  * 作用：使用RangeStatus枚举值标识范围状态
  * 值域：0(Open) 或 1(Closed)
  * 用途：状态过滤、持续时间计算调整、实时分析

内存布局优化：
- 字段对齐：align=True确保字段按照64位边界对齐
- 缓存友好：紧凑的内存布局提高CPU缓存命中率
- 向量化：支持SIMD指令集加速批量操作
- Numba兼容：结构完全兼容Numba JIT编译

应用示例：
```python
import vectorbt as vbt
import pandas as pd
import numpy as np

# 创建示例数据：股票价格的布尔条件（如价格大于移动平均线）
prices = pd.Series([100, 105, 102, 98, 103, 108, 106, 110])
ma = prices.rolling(3).mean()
above_ma = prices > ma

# 生成范围记录
ranges = vbt.Ranges.from_ts(above_ma.fillna(False))

# 访问底层记录数组
records_array = ranges.values
print("范围记录结构:")
print(f"记录数量: {len(records_array)}")
print(f"字段名称: {records_array.dtype.names}")

# 查看具体记录
for i, record in enumerate(records_array):
    print(f"记录 {i}:")
    print(f"  ID: {record['id']}")
    print(f"  列: {record['col']}")  
    print(f"  起始: {record['start_idx']}")
    print(f"  结束: {record['end_idx']}")
    print(f"  状态: {'Open' if record['status'] == 0 else 'Closed'}")
    print()

# 计算自定义指标
durations = records_array['end_idx'] - records_array['start_idx']
print(f"范围持续时间: {durations}")
print(f"平均持续时间: {durations.mean():.2f}")
```

性能特点：
- 内存效率：紧凑的二进制格式，最小化内存占用
- 访问速度：结构化访问避免了字典查找的开销
- 批量操作：支持NumPy的向量化操作
- 类型安全：强类型字段定义避免类型错误
"""

# 配置range_dt的文档生成
__pdoc__['range_dt'] = f"""`np.dtype` of range records.

```json
{to_doc(range_dt)}
```
"""

# 定义回撤记录的数据类型（drawdown_dt）
# 相比范围记录，回撤记录包含更丰富的信息，支持详细的回撤分析
drawdown_dt = np.dtype([
    ('id', np.int64),         # 记录唯一标识符：回撤事件的唯一ID
    ('col', np.int64),        # 列索引：标识属于哪一列数据（资产/策略）
    ('peak_idx', np.int64),   # 峰值索引：回撤开始前的最高点时间索引
    ('start_idx', np.int64),  # 开始索引：回撤正式开始的时间索引（峰值后第一个下跌点）
    ('valley_idx', np.int64), # 谷值索引：回撤过程中的最低点时间索引
    ('end_idx', np.int64),    # 结束索引：回撤结束的时间索引（恢复到峰值或序列结束）
    ('peak_val', np.float64), # 峰值价格：回撤开始前的最高价格值
    ('valley_val', np.float64), # 谷值价格：回撤过程中的最低价格值
    ('end_val', np.float64),  # 结束价格：回撤结束时的价格值
    ('status', np.int64),     # 状态标识：使用DrawdownStatus枚举值
], align=True)  # 内存对齐优化，确保高效的数据访问
"""
回撤记录数据类型 - vectorbt框架中回撤分析的完整数据结构定义

该数据类型是量化风险管理的核心数据结构，包含了回撤事件的全生命周期信息。
相比简单的范围记录，回撤记录提供了价格变化的详细信息，支持深度的风险分析。

字段详细说明：

时间维度字段：
- id (int64): 回撤事件唯一标识
  * 功能：全局唯一的回撤事件编号
  * 用途：事件索引、历史查询、性能追踪

- col (int64): 数据列索引  
  * 功能：标识回撤事件所属的数据列
  * 用途：多资产分析、组合风险分解

- peak_idx (int64): 峰值时间索引
  * 功能：标识回撤开始前的历史最高点位置
  * 用途：回撤起点确定、周期性分析

- start_idx (int64): 回撤开始时间索引
  * 功能：标识价格开始下跌的时间点
  * 关系：通常为peak_idx + 1
  * 用途：回撤开始时间定位

- valley_idx (int64): 最低点时间索引
  * 功能：标识回撤过程中的价格最低点
  * 用途：最大损失时点确定、恢复时间计算

- end_idx (int64): 回撤结束时间索引
  * 功能：标识回撤恢复或结束的时间点
  * 状态相关：Active状态时指向当前时间，Recovered状态时指向恢复时间

价格维度字段：
- peak_val (float64): 峰值价格
  * 功能：记录回撤前的历史最高价格
  * 精度：双精度浮点数，确保价格精确性
  * 用途：回撤幅度计算的基准值

- valley_val (float64): 谷值价格
  * 功能：记录回撤过程中的最低价格
  * 用途：最大回撤幅度计算、风险量化

- end_val (float64): 结束价格
  * 功能：记录回撤结束时的价格水平
  * 状态相关：Active状态为当前价格，Recovered状态为恢复价格
  * 用途：恢复程度评估、当前风险状态

- status (int64): 回撤状态
  * 功能：使用DrawdownStatus枚举标识回撤状态
  * 值域：0(Active活跃) 或 1(Recovered恢复)
  * 用途：状态过滤、实时监控、历史分析

核心计算公式：
- 回撤幅度 = (valley_val - peak_val) / peak_val
- 恢复幅度 = (end_val - valley_val) / (peak_val - valley_val)  
- 回撤持续期 = valley_idx - start_idx + 1
- 恢复持续期 = end_idx - valley_idx
- 总周期 = end_idx - start_idx + 1

应用示例：
```python
import vectorbt as vbt
import pandas as pd
import numpy as np

# 创建模拟投资组合价值序列
np.random.seed(42)
returns = np.random.normal(0.001, 0.02, 252)  # 252个交易日的收益率
prices = pd.Series(100 * (1 + returns).cumprod(), 
                  index=pd.date_range('2023-01-01', periods=252, freq='D'))

# 生成回撤分析
drawdowns = vbt.Drawdowns.from_ts(prices)

# 访问回撤记录数组
dd_records = drawdowns.values
print(f"检测到 {len(dd_records)} 个回撤事件")

# 分析每个回撤事件
for i, dd in enumerate(dd_records):
    # 计算关键指标
    dd_pct = (dd['valley_val'] - dd['peak_val']) / dd['peak_val'] * 100
    duration = dd['valley_idx'] - dd['start_idx'] + 1
    recovery_duration = dd['end_idx'] - dd['valley_idx'] if dd['status'] == 1 else None
    
    print(f"\n回撤事件 {dd['id']}:")
    print(f"  峰值时间: {prices.index[dd['peak_idx']]}")
    print(f"  峰值价格: ${dd['peak_val']:.2f}")
    print(f"  谷值时间: {prices.index[dd['valley_idx']]}")
    print(f"  谷值价格: ${dd['valley_val']:.2f}")
    print(f"  回撤幅度: {dd_pct:.2f}%")
    print(f"  下跌天数: {duration}")
    print(f"  状态: {'活跃' if dd['status'] == 0 else '已恢复'}")
    
    if recovery_duration is not None:
        print(f"  恢复天数: {recovery_duration}")
        recovery_pct = (dd['end_val'] - dd['valley_val']) / (dd['peak_val'] - dd['valley_val']) * 100
        print(f"  恢复程度: {recovery_pct:.1f}%")

# 统计分析
max_dd_idx = np.argmin([(dd['valley_val'] - dd['peak_val']) / dd['peak_val'] for dd in dd_records])
max_dd = dd_records[max_dd_idx]
max_dd_pct = (max_dd['valley_val'] - max_dd['peak_val']) / max_dd['peak_val'] * 100

print(f"\n风险统计:")
print(f"最大回撤: {max_dd_pct:.2f}%")
print(f"回撤事件数量: {len(dd_records)}")
print(f"活跃回撤数量: {sum(dd['status'] == 0 for dd in dd_records)}")
print(f"平均回撤幅度: {np.mean([(dd['valley_val'] - dd['peak_val']) / dd['peak_val'] for dd in dd_records]) * 100:.2f}%")
```

内存和性能优化：
- 紧凑存储：使用结构化数组减少内存碎片
- 字段对齐：64位对齐确保最佳CPU访问性能
- 批量计算：支持向量化操作，避免Python循环开销
- Numba兼容：完全支持JIT编译，实现C级别性能
- 缓存优化：顺序访问模式提高缓存命中率
"""

# 配置drawdown_dt的文档生成
__pdoc__['drawdown_dt'] = f"""`np.dtype` of drawdown records.

```json
{to_doc(drawdown_dt)}
```
"""
