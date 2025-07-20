# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT SIGNALS ACCESSORS MODULE: 信号数据访问器模块
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中signals模块的核心访问器实现，为pandas的Series和DataFrame
提供了专门的信号数据处理接口。该模块通过pandas accessor模式，将复杂的信号处理功能
封装成简洁易用的API，是vectorbt信号系统的重要组成部分。

核心设计理念：
1. **访问器模式架构**：通过pandas accessor扩展机制，为Series和DataFrame提供.vbt.signals接口
2. **功能模块化**：将信号处理功能分为生成、过滤、随机、出场、范围、排序、索引等多个模块
3. **类型安全**：严格限制输入数据类型为布尔型，确保信号数据的语义正确性
4. **高性能计算**：底层调用Numba编译的函数，提供接近C语言的执行性能
5. **可视化集成**：内置专业的信号可视化功能，支持入场出场标记的图形化展示

主要功能模块：

【信号生成模块 Generation】
- generate(): 基于自定义选择函数生成信号
- generate_both(): 同时生成入场和出场信号
- generate_exits(): 基于现有入场信号生成出场信号

【信号过滤模块 Filtering】
- clean(): 清理和优化信号序列，确保逻辑一致性

【随机信号模块 Random】
- generate_random(): 生成随机信号，支持数量或概率控制
- generate_random_both(): 生成随机入场出场信号对
- generate_random_exits(): 为现有入场信号生成随机出场信号

【止损出场模块 Exits】
- generate_stop_exits(): 基于价格阈值生成止损出场信号
- generate_ohlc_stop_exits(): 基于OHLC数据生成高级止损出场信号

【范围分析模块 Ranges】
- between_ranges(): 分析信号之间的时间间隔范围
- partition_ranges(): 识别连续信号的分区范围
- between_partition_ranges(): 分析分区之间的间隔范围

【排序系统模块 Ranking】
- rank(): 通用信号排序功能
- pos_rank(): 信号位置排序
- partition_pos_rank(): 分区位置排序

【索引管理模块 Index】
- nth_index(): 获取第N个信号的索引位置
- norm_avg_index(): 计算归一化平均索引位置
- index_mapped(): 获取信号的映射索引数组

【统计分析模块 Stats】
- total(): 统计信号总数
- rate(): 计算信号出现率
- total_partitions(): 统计分区总数
- partition_rate(): 计算分区率

【逻辑运算模块 Logical】
- AND(): 逻辑与运算
- OR(): 逻辑或运算
- XOR(): 逻辑异或运算

【可视化模块 Plotting】
- plot(): 基础信号可视化
- plot_as_markers(): 标记式信号可视化
- plot_as_entry_markers(): 入场信号标记可视化
- plot_as_exit_markers(): 出场信号标记可视化

技术特点：
- **类型安全**：严格验证输入数据类型，确保信号数据的布尔语义
- **高性能**：底层使用Numba编译函数，支持大规模数据处理
- **内存高效**：使用就地操作和缓存友好的数据访问模式
- **API一致**：统一的访问器接口，支持Series和DataFrame
- **扩展性强**：模块化设计，易于添加新的信号处理功能

应用场景：
- **量化策略开发**：为交易策略提供信号生成和处理能力
- **信号分析**：分析信号的时间分布、频率和模式
- **风险管理**：实现止损止盈等风险控制信号
- **策略回测**：为回测系统提供信号数据支持
- **可视化分析**：提供专业的信号可视化功能

与vectorbt生态系统的关系：
- 继承自GenericAccessor，扩展了基础访问器功能
- 与signals.nb模块协同工作，提供底层计算支持
- 与signals.factory模块配合，支持信号生成器配置
- 与portfolio模块集成，支持策略回测和风险管理
- 与generic模块协作，提供统计分析和可视化功能

使用模式：
所有功能都通过pandas accessor模式访问：
1. Series访问：pd.Series.vbt.signals.*
2. DataFrame访问：pd.DataFrame.vbt.signals.*
3. 输入数据必须是布尔型，表示信号状态
4. 支持分组操作和批量处理

该模块是vectorbt框架中连接信号数据与用户应用的重要桥梁，为量化交易
提供了工业级的信号处理和分析能力。

使用示例：
```python
import pandas as pd
import vectorbt as vbt
import numpy as np

# 创建示例信号数据
signals = pd.DataFrame({
    'a': [True, False, False, True, False],
    'b': [False, True, True, False, True],
    'c': [True, True, False, False, True]
})

# 基础统计分析
print("信号总数:", signals.vbt.signals.total())
print("信号出现率:", signals.vbt.signals.rate())

# 生成随机出场信号
exits = signals.vbt.signals.generate_random_exits(prob=0.3, seed=42)

# 生成止损出场信号
prices = pd.Series([100, 105, 98, 110, 95])
stop_exits = signals.vbt.signals.generate_stop_exits(prices, stop=-0.05)

# 可视化信号
fig = signals.vbt.signals.plot(title="信号可视化示例")
```
================================================================================

自定义pandas访问器，用于信号数据处理。

可通过以下方式访问方法：

* `SignalsSRAccessor` -> `pd.Series.vbt.signals.*`
* `SignalsDFAccessor` -> `pd.DataFrame.vbt.signals.*`

```pycon
>>> import pandas as pd
>>> import vectorbt as vbt

>>> # vectorbt.signals.accessors.SignalsAccessor.pos_rank
>>> pd.Series([False, True, True, True, False]).vbt.signals.pos_rank()
0    0
1    1
2    2
3    3
4    0
dtype: int64
```

访问器继承自 `vectorbt.generic.accessors`。

!!! note
    底层Series/DataFrame应该已经是信号序列。
    
    输入数组应该是 `np.bool_` 类型。
    
    分组操作仅支持接受 `group_by` 参数的方法。
    
    访问器不使用缓存机制。

运行以下示例：

```pycon
>>> import vectorbt as vbt
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from datetime import datetime

>>> mask = pd.DataFrame({
...     'a': [True, False, False, False, False],
...     'b': [True, False, True, False, True],
...     'c': [True, True, True, False, False]
... }, index=pd.Index([
...     datetime(2020, 1, 1),
...     datetime(2020, 1, 2),
...     datetime(2020, 1, 3),
...     datetime(2020, 1, 4),
...     datetime(2020, 1, 5)
... ]))
>>> mask
                a      b      c
2020-01-01   True   True   True
2020-01-02  False  False   True
2020-01-03  False   True   True
2020-01-04  False  False  False
2020-01-05  False   True  False
```

## 统计分析

!!! hint
    参见 `vectorbt.generic.stats_builder.StatsBuilderMixin.stats` 和 `SignalsAccessor.metrics`。

```pycon
>>> mask.vbt.signals.stats(column='a')
Start                       2020-01-01 00:00:00
End                         2020-01-05 00:00:00
Period                          5 days 00:00:00
Total                                         1
Rate [%]                                     20
First Index                 2020-01-01 00:00:00
Last Index                  2020-01-01 00:00:00
Norm Avg Index [-1, 1]                       -1
Distance: Min                               NaT
Distance: Max                               NaT
Distance: Mean                              NaT
Distance: Std                               NaT
Total Partitions                              1
Partition Rate [%]                          100
Partition Length: Min           1 days 00:00:00
Partition Length: Max           1 days 00:00:00
Partition Length: Mean          1 days 00:00:00
Partition Length: Std                       NaT
Partition Distance: Min                     NaT
Partition Distance: Max                     NaT
Partition Distance: Mean                    NaT
Partition Distance: Std                     NaT
Name: a, dtype: object
```

我们可以传递另一个信号数组来比较：

```pycon
>>> mask.vbt.signals.stats(column='a', settings=dict(other=mask['b']))
Start                       2020-01-01 00:00:00
End                         2020-01-05 00:00:00
Period                          5 days 00:00:00
Total                                         1
Rate [%]                                     20
Total Overlapping                             1
Overlapping Rate [%]                    33.3333
First Index                 2020-01-01 00:00:00
Last Index                  2020-01-01 00:00:00
Norm Avg Index [-1, 1]                       -1
Distance -> Other: Min          0 days 00:00:00
Distance -> Other: Max          0 days 00:00:00
Distance -> Other: Mean         0 days 00:00:00
Distance -> Other: Std                      NaT
Total Partitions                              1
Partition Rate [%]                          100
Partition Length: Min           1 days 00:00:00
Partition Length: Max           1 days 00:00:00
Partition Length: Mean          1 days 00:00:00
Partition Length: Std                       NaT
Partition Distance: Min                     NaT
Partition Distance: Max                     NaT
Partition Distance: Mean                    NaT
Partition Distance: Std                     NaT
Name: a, dtype: object
```

我们也可以将持续时间作为浮点数而不是timedelta返回：

```pycon
>>> mask.vbt.signals.stats(column='a', settings=dict(to_timedelta=False))
Start                       2020-01-01 00:00:00
End                         2020-01-05 00:00:00
Period                                        5
Total                                         1
Rate [%]                                     20
First Index                 2020-01-01 00:00:00
Last Index                  2020-01-01 00:00:00
Norm Avg Index [-1, 1]                       -1
Distance: Min                               NaN
Distance: Max                               NaN
Distance: Mean                              NaN
Distance: Std                               NaN
Total Partitions                              1
Partition Rate [%]                          100
Partition Length: Min                         1
Partition Length: Max                         1
Partition Length: Mean                        1
Partition Length: Std                       NaN
Partition Distance: Min                     NaN
Partition Distance: Max                     NaN
Partition Distance: Mean                    NaN
Partition Distance: Std                     NaN
Name: a, dtype: object
```

`SignalsAccessor.stats` 也支持（重新）分组：

```pycon
>>> mask.vbt.signals.stats(column=0, group_by=[0, 0, 1])
Start                       2020-01-01 00:00:00
End                         2020-01-05 00:00:00
Period                          5 days 00:00:00
Total                                         4
Rate [%]                                     40
First Index                 2020-01-01 00:00:00
Last Index                  2020-01-05 00:00:00
Norm Avg Index [-1, 1]                    -0.25
Distance: Min                   2 days 00:00:00
Distance: Max                   2 days 00:00:00
Distance: Mean                  2 days 00:00:00
Distance: Std                   0 days 00:00:00
Total Partitions                              4
Partition Rate [%]                          100
Partition Length: Min           1 days 00:00:00
Partition Length: Max           1 days 00:00:00
Partition Length: Mean          1 days 00:00:00
Partition Length: Std           0 days 00:00:00
Partition Distance: Min         2 days 00:00:00
Partition Distance: Max         2 days 00:00:00
Partition Distance: Mean        2 days 00:00:00
Partition Distance: Std         0 days 00:00:00
Name: 0, dtype: object
```

## 绘图功能

!!! hint
    参见 `vectorbt.generic.plots_builder.PlotsBuilderMixin.plots` 和 `SignalsAccessor.subplots`。

此类继承自 `vectorbt.generic.accessors.GenericAccessor` 的子图功能。
"""

import warnings  # 导入警告模块，用于处理函数参数警告

import numpy as np  # 导入NumPy数值计算库，用于数组操作和数学计算
import pandas as pd  # 导入pandas数据处理库，用于Series和DataFrame操作

from vectorbt import _typing as tp  # 导入vectorbt类型系统，提供类型注解支持
from vectorbt.base import reshape_fns  # 导入数组重塑函数，用于数据形状转换
from vectorbt.base.array_wrapper import ArrayWrapper  # 导入数组包装器，用于数据包装和元数据管理
from vectorbt.generic import nb as generic_nb  # 导入通用Numba函数，用于基础数值计算
from vectorbt.generic import plotting  # 导入绘图模块，用于数据可视化
from vectorbt.generic.accessors import GenericAccessor, GenericSRAccessor, GenericDFAccessor  # 导入通用访问器基类
from vectorbt.generic.ranges import Ranges  # 导入范围类，用于时间范围分析
from vectorbt.records.mapped_array import MappedArray  # 导入映射数组，用于索引映射
from vectorbt.root_accessors import register_dataframe_vbt_accessor, register_series_vbt_accessor  # 导入访问器注册函数
from vectorbt.signals import nb  # 导入信号Numba函数，用于底层信号计算
from vectorbt.utils import checks  # 导入检查工具，用于参数验证
from vectorbt.utils.colors import adjust_lightness  # 导入颜色调整函数，用于可视化颜色处理
from vectorbt.utils.config import merge_dicts, Config  # 导入配置工具，用于字典合并和配置管理
from vectorbt.utils.decorators import class_or_instancemethod  # 导入装饰器，用于类或实例方法
from vectorbt.utils.template import RepEval  # 导入模板评估器，用于动态字符串生成

__pdoc__ = {}  # 文档控制字典，用于控制自动文档生成


class SignalsAccessor(GenericAccessor):
    """
    信号访问器类 - 为信号数据提供专门的访问接口
    
    功能概述：
    这是vectorbt信号系统的核心访问器类，为pandas的Series和DataFrame提供了专门的
    信号数据处理接口。该类继承自GenericAccessor，扩展了信号生成、过滤、分析、
    可视化等专业功能，是量化交易信号处理的重要基础设施。
    
    核心特性：
    - 支持Series和DataFrame两种数据类型的信号处理
    - 提供完整的信号生命周期管理功能
    - 集成高性能的Numba编译计算函数
    - 内置专业的信号可视化和统计分析功能
    - 支持分组操作和批量处理
    
    主要功能模块：
    - **信号生成**：基于自定义函数或随机算法生成信号
    - **信号过滤**：清理和优化信号序列，确保逻辑一致性
    - **止损管理**：基于价格阈值生成止损出场信号
    - **范围分析**：分析信号的时间分布和间隔特征
    - **排序系统**：为信号分配优先级和位置排序
    - **统计分析**：计算信号的频率、分布等统计指标
    - **逻辑运算**：支持信号间的AND、OR、XOR等逻辑操作
    - **可视化**：提供专业的信号图形化展示功能
    
    技术特点：
    - 严格类型检查，确保输入数据为布尔型
    - 高性能计算，底层使用Numba编译优化
    - 内存高效，支持大规模数据处理
    - API一致，提供统一的访问接口
    - 扩展性强，易于添加新功能
    
    使用场景：
    - 量化交易策略的信号生成和处理
    - 技术分析指标的信号转换
    - 风险管理系统的止损信号生成
    - 策略回测的信号数据准备
    - 信号模式分析和可视化
    
    示例用法：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建信号数据
    signals = pd.DataFrame({
        'a': [True, False, True, False, True],
        'b': [False, True, False, True, False]
    })
    
    # 基础统计分析
    total_signals = signals.vbt.signals.total()
    signal_rate = signals.vbt.signals.rate()
    
    # 生成随机出场信号
    exits = signals.vbt.signals.generate_random_exits(prob=0.3)
    
    # 生成止损信号
    prices = pd.Series([100, 105, 98, 110, 95])
    stop_exits = signals.vbt.signals.generate_stop_exits(prices, stop=-0.05)
    
    # 可视化信号
    fig = signals.vbt.signals.plot(title="信号分析")
    ```
    
    与vectorbt生态系统的关系：
    - 继承自GenericAccessor，扩展基础访问器功能
    - 与signals.nb模块协同，提供底层计算支持
    - 与signals.factory模块配合，支持信号生成器
    - 与portfolio模块集成，支持策略回测
    - 与generic模块协作，提供统计和可视化功能
    """

    def __init__(self, obj: tp.SeriesFrame, **kwargs) -> None:
        """
        初始化信号访问器
        
        功能说明：
        创建信号访问器实例，验证输入数据的类型和格式，确保数据符合信号处理的要求。
        该初始化方法会严格检查输入数据类型，确保所有数据都是布尔型，这是信号处理的基础要求。
        
        参数说明：
            obj (Series/DataFrame): 要处理的信号数据对象
                - 必须是pandas的Series或DataFrame类型
                - 数据类型必须是布尔型(np.bool_)
                - True表示有信号，False表示无信号
            **kwargs: 传递给父类GenericAccessor的关键字参数
        
        验证逻辑：
        - 检查数据类型是否为布尔型
        - 确保数据格式符合信号处理要求
        - 初始化父类访问器功能
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        
        # 创建布尔型信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False],
            'b': [False, True, False, True]
        })
        
        # 创建信号访问器
        accessor = signals.vbt.signals
        print(f"信号总数: {accessor.total()}")
        ```
        
        注意事项：
        - 输入数据必须是布尔型，否则会抛出类型错误
        - 建议使用True/False而不是1/0来表示信号状态
        - 数据应该具有时间索引，便于后续的时间序列分析
        """
        checks.assert_dtype(obj, np.bool_)  # 验证数据类型必须是布尔型

        GenericAccessor.__init__(self, obj, **kwargs)  # 调用父类初始化方法

    @property
    def sr_accessor_cls(self) -> tp.Type["SignalsSRAccessor"]:
        """
        Series访问器类属性
        
        功能说明：
        返回专门用于pandas Series的信号访问器类，用于处理单列信号数据。
        该属性是访问器工厂模式的一部分，根据数据类型自动选择合适的访问器类。
        
        返回值：
            tp.Type[SignalsSRAccessor]: Series专用的信号访问器类
        
        使用场景：
        - 当处理单列信号数据时自动使用
        - 需要明确指定Series访问器时使用
        - 访问器内部自动分发时使用
        
        示例用法：
        ```python
        import pandas as pd
        
        # 创建单列信号数据
        signals = pd.Series([True, False, True, False, True])
        
        # 自动使用Series访问器
        accessor = signals.vbt.signals
        print(type(accessor))  # <class 'vectorbt.signals.accessors.SignalsSRAccessor'>
        ```
        """
        return SignalsSRAccessor  # 返回Series专用的信号访问器类

    @property
    def df_accessor_cls(self) -> tp.Type["SignalsDFAccessor"]:
        """
        DataFrame访问器类属性
        
        功能说明：
        返回专门用于pandas DataFrame的信号访问器类，用于处理多列信号数据。
        该属性是访问器工厂模式的一部分，根据数据类型自动选择合适的访问器类。
        
        返回值：
            tp.Type[SignalsDFAccessor]: DataFrame专用的信号访问器类
        
        使用场景：
        - 当处理多列信号数据时自动使用
        - 需要明确指定DataFrame访问器时使用
        - 访问器内部自动分发时使用
        
        示例用法：
        ```python
        import pandas as pd
        
        # 创建多列信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False],
            'b': [False, True, False, True]
        })
        
        # 自动使用DataFrame访问器
        accessor = signals.vbt.signals
        print(type(accessor))  # <class 'vectorbt.signals.accessors.SignalsDFAccessor'>
        ```
        """
        return SignalsDFAccessor  # 返回DataFrame专用的信号访问器类

    # ############# 方法重写 ############# #

    def bshift(self, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """
        向后移动信号数据
        
        功能说明：
        重写父类的bshift方法，为信号数据提供专门的向后移动功能。
        该方法将信号数据向后移动指定的周期数，并用指定的填充值填充空位。
        对于信号数据，默认填充值为False，这符合信号处理的语义。
        
        参数说明：
            *args: 传递给父类bshift方法的位置参数
                - 通常包含移动的周期数
            fill_value (bool): 填充值，默认为False
                - False: 表示无信号状态，符合信号处理语义
                - True: 表示有信号状态，通常不推荐使用
            **kwargs: 传递给父类bshift方法的关键字参数
        
        返回值：
            tp.SeriesFrame: 移动后的信号数据，保持原始数据类型和索引
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.Series([True, False, True, False, True])
        
        # 向后移动1个周期
        shifted = signals.vbt.signals.bshift(1)
        print("原始信号:", signals.values)
        print("移动后信号:", shifted.values)
        # 输出:
        # 原始信号: [ True False  True False  True]
        # 移动后信号: [False  True False  True False]
        ```
        
        应用场景：
        - 信号延迟分析
        - 滞后信号生成
        - 信号时序调整
        - 策略参数优化
        
        注意事项：
        - 移动操作会改变信号的时序关系
        - 填充值的选择影响后续分析结果
        - 建议在移动后检查数据的完整性
        """
        return GenericAccessor.bshift(self, *args, fill_value=fill_value, **kwargs)  # 调用父类方法，使用False作为默认填充值

    def fshift(self, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """
        向前移动信号数据
        
        功能说明：
        重写父类的fshift方法，为信号数据提供专门的向前移动功能。
        该方法将信号数据向前移动指定的周期数，并用指定的填充值填充空位。
        对于信号数据，默认填充值为False，这符合信号处理的语义。
        
        参数说明：
            *args: 传递给父类fshift方法的位置参数
                - 通常包含移动的周期数
            fill_value (bool): 填充值，默认为False
                - False: 表示无信号状态，符合信号处理语义
                - True: 表示有信号状态，通常不推荐使用
            **kwargs: 传递给父类fshift方法的关键字参数
        
        返回值：
            tp.SeriesFrame: 移动后的信号数据，保持原始数据类型和索引
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.Series([True, False, True, False, True])
        
        # 向前移动1个周期
        shifted = signals.vbt.signals.fshift(1)
        print("原始信号:", signals.values)
        print("移动后信号:", shifted.values)
        # 输出:
        # 原始信号: [ True False  True False  True]
        # 移动后信号: [False  True False  True False]
        ```
        
        应用场景：
        - 信号提前分析
        - 前瞻信号生成
        - 信号时序调整
        - 策略参数优化
        
        注意事项：
        - 移动操作会改变信号的时序关系
        - 填充值的选择影响后续分析结果
        - 建议在移动后检查数据的完整性
        """
        return GenericAccessor.fshift(self, *args, fill_value=fill_value, **kwargs)  # 调用父类方法，使用False作为默认填充值

    @classmethod
    def empty(cls, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """
        创建空的信号数据对象
        
        功能说明：
        重写父类的empty方法，创建指定形状的空信号数据对象。
        该方法用于初始化信号数据结构，所有元素都填充为指定的布尔值。
        对于信号数据，默认填充值为False，表示无信号状态。
        
        参数说明：
            *args: 传递给父类empty方法的位置参数
                - 通常包含数据的形状信息
            fill_value (bool): 填充值，默认为False
                - False: 表示无信号状态，符合信号处理语义
                - True: 表示有信号状态，通常不推荐使用
            **kwargs: 传递给父类empty方法的关键字参数
                - 包含索引、列名等元数据信息
        
        返回值：
            tp.SeriesFrame: 空的信号数据对象，所有元素为fill_value
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建空的信号Series
        empty_series = pd.Series.vbt.signals.empty(5, fill_value=False)
        print("空Series:", empty_series.values)
        # 输出: [False False False False False]
        
        # 创建空的信号DataFrame
        empty_df = pd.DataFrame.vbt.signals.empty(
            (5, 3), 
            fill_value=False,
            columns=['a', 'b', 'c']
        )
        print("空DataFrame:\n", empty_df)
        ```
        
        应用场景：
        - 信号数据初始化
        - 模板数据结构创建
        - 批量信号生成准备
        - 测试数据创建
        
        注意事项：
        - 创建的对象数据类型为布尔型
        - 填充值的选择影响后续操作
        - 建议提供有意义的索引和列名
        """
        return GenericAccessor.empty(*args, fill_value=fill_value, dtype=np.bool_, **kwargs)  # 调用父类方法，指定布尔型数据类型

    @classmethod
    def empty_like(cls, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """
        创建与指定对象形状相同的空信号数据
        
        功能说明：
        重写父类的empty_like方法，创建与指定对象具有相同形状和元数据的空信号数据。
        该方法用于快速创建与现有信号数据相同结构的空数据对象，便于后续填充。
        
        参数说明：
            *args: 传递给父类empty_like方法的位置参数
                - 通常包含参考对象
            fill_value (bool): 填充值，默认为False
                - False: 表示无信号状态，符合信号处理语义
                - True: 表示有信号状态，通常不推荐使用
            **kwargs: 传递给父类empty_like方法的关键字参数
        
        返回值：
            tp.SeriesFrame: 与参考对象形状相同的空信号数据
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建参考信号数据
        reference = pd.DataFrame({
            'a': [True, False, True],
            'b': [False, True, False]
        })
        
        # 创建相同形状的空信号数据
        empty_like = pd.DataFrame.vbt.signals.empty_like(reference, fill_value=False)
        print("参考数据:\n", reference)
        print("空数据:\n", empty_like)
        ```
        
        应用场景：
        - 信号数据模板创建
        - 批量信号初始化
        - 数据结构复制
        - 测试环境准备
        
        注意事项：
        - 创建的对象具有与参考对象相同的索引和列名
        - 数据类型固定为布尔型
        - 填充值的选择影响后续操作
        """
        return GenericAccessor.empty_like(*args, fill_value=fill_value, dtype=np.bool_, **kwargs)  # 调用父类方法，指定布尔型数据类型

    # ############# 信号生成 ############# #

    @classmethod
    def generate(cls,
                 shape: tp.RelaxedShape,
                 choice_func_nb: tp.ChoiceFunc, *args,
                 pick_first: bool = False,
                 **kwargs) -> tp.SeriesFrame:
        """
        基于自定义选择函数生成信号
        
        功能说明：
        使用Numba编译的选择函数生成信号数据，这是信号生成的核心方法。
        该方法提供了灵活的信号生成机制，允许用户通过自定义函数控制信号的生成逻辑。
        选择函数应该返回信号位置的索引数组，该方法会自动将这些位置转换为布尔型信号。
        
        参数说明：
            shape (tuple/int): 输出信号的形状
                - 整数: 表示一维信号的长度
                - 元组: 表示二维信号的(行数, 列数)
            choice_func_nb (callable): Numba编译的选择函数
                - 函数签名: def func(from_i, to_i, col, *args) -> np.ndarray
                - 返回信号位置的索引数组
                - 必须使用@njit装饰器编译
            *args: 传递给选择函数的额外参数
            pick_first (bool): 是否只选择第一个信号
                - True: 在每个搜索范围内只选择第一个信号
                - False: 选择所有符合条件的信号
            **kwargs: 传递给pandas构造函数的关键字参数
                - 包含index、columns等元数据信息
        
        返回值：
            tp.SeriesFrame: 生成的信号数据，True表示有信号，False表示无信号
        
        选择函数要求：
        ```python
        from numba import njit
        import numpy as np
        
        @njit
        def my_choice_func(from_i, to_i, col, *args):
            # from_i: 搜索起始索引
            # to_i: 搜索结束索引
            # col: 当前列索引
            # *args: 额外参数
            
            # 返回信号位置的索引数组
            return np.array([from_i + col])  # 示例：在每列的第一个位置生成信号
        ```
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        from numba import njit
        
        # 定义选择函数：每5个周期生成一个信号
        @njit
        def periodic_choice(from_i, to_i, col):
            if (from_i // 5) * 5 < to_i:
                return np.array([(from_i // 5) * 5])
            return np.array([], dtype=np.int64)
        
        # 生成信号
        signals = pd.DataFrame.vbt.signals.generate(
            (20, 3),  # 20行3列的信号数据
            periodic_choice,
            index=pd.date_range('2020-01-01', periods=20),
            columns=['a', 'b', 'c']
        )
        print("生成的信号:\n", signals)
        ```
        
        应用场景：
        - 技术指标信号生成
        - 自定义交易策略信号
        - 信号模式模拟
        - 策略测试数据生成
        
        注意事项：
        - 选择函数必须是Numba编译的
        - 函数应该返回有效的索引数组
        - 索引应该在有效范围内
        - 建议使用向量化操作提高性能
        """
        checks.assert_numba_func(choice_func_nb)  # 验证选择函数是否为Numba编译函数

        # 处理形状参数，确保是二维元组
        if not isinstance(shape, tuple):
            shape = (shape, 1)  # 一维形状转换为二维
        elif isinstance(shape, tuple) and len(shape) == 1:
            shape = (shape[0], 1)  # 一维元组转换为二维

        # 调用底层Numba函数生成信号
        result = nb.generate_nb(shape, pick_first, choice_func_nb, *args)

        # 根据形状返回适当的数据类型
        if cls.is_series():
            if shape[1] > 1:
                raise ValueError("Use DataFrame accessor")  # Series访问器不支持多列
            return pd.Series(result[:, 0], **kwargs)  # 返回单列Series
        return pd.DataFrame(result, **kwargs)  # 返回多列DataFrame

    @classmethod
    def generate_both(cls,
                      shape: tp.RelaxedShape,
                      entry_choice_func_nb: tp.Optional[tp.ChoiceFunc] = None,
                      entry_args: tp.ArgsLike = None,
                      exit_choice_func_nb: tp.Optional[tp.ChoiceFunc] = None,
                      exit_args: tp.ArgsLike = None,
                      entry_wait: int = 1,
                      exit_wait: int = 1,
                      entry_pick_first: bool = True,
                      exit_pick_first: bool = True,
                      **kwargs) -> tp.Tuple[tp.SeriesFrame, tp.SeriesFrame]:
        """
        同时生成入场和出场信号
        
        功能说明：
        使用两个独立的Numba编译选择函数同时生成入场和出场信号。
        该方法实现了完整的交易信号生成逻辑，能够创建配对的入场和出场信号序列。
        入场和出场信号可以有不同的生成逻辑和时间间隔，支持复杂的交易策略需求。
        
        参数说明：
            shape (tuple/int): 输出信号的形状
                - 整数: 表示一维信号的长度
                - 元组: 表示二维信号的(行数, 列数)
            entry_choice_func_nb (callable): 入场信号选择函数
                - 函数签名: def func(from_i, to_i, col, *args) -> np.ndarray
                - 返回入场信号位置的索引数组
                - 必须使用@njit装饰器编译
            entry_args (tuple): 传递给入场选择函数的参数
                - 包含入场函数需要的额外参数
            exit_choice_func_nb (callable): 出场信号选择函数
                - 函数签名: def func(from_i, to_i, col, *args) -> np.ndarray
                - 返回出场信号位置的索引数组
                - 必须使用@njit装饰器编译
            exit_args (tuple): 传递给出场选择函数的参数
                - 包含出场函数需要的额外参数
            entry_wait (int): 入场信号间的最小等待周期
                - 控制入场信号的密度
            exit_wait (int): 出场信号间的最小等待周期
                - 控制出场信号的密度
            entry_pick_first (bool): 是否只选择第一个入场信号
                - True: 在每个搜索范围内只选择第一个入场信号
                - False: 选择所有符合条件的入场信号
            exit_pick_first (bool): 是否只选择第一个出场信号
                - True: 在每个搜索范围内只选择第一个出场信号
                - False: 选择所有符合条件的出场信号
            **kwargs: 传递给pandas构造函数的关键字参数
        
        返回值：
            tuple: (入场信号数据, 出场信号数据)
                - 两个数据对象都是布尔型，True表示有信号
        
        选择函数示例：
        ```python
        from numba import njit
        import numpy as np
        
        @njit
        def entry_choice_func(from_i, to_i, col, temp_idx_arr):
            # 在搜索范围的开始位置生成入场信号
            temp_idx_arr[0] = from_i
            return temp_idx_arr[:1]
        
        @njit
        def exit_choice_func(from_i, to_i, col, temp_idx_arr):
            # 在入场信号后等待col个周期生成出场信号
            wait = col
            temp_idx_arr[0] = from_i + wait
            if temp_idx_arr[0] < to_i:
                return temp_idx_arr[:1]
            return temp_idx_arr[:0]  # 返回空数组
        ```
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        from numba import njit
        
        # 定义入场选择函数：在每列的第一个位置入场
        @njit
        def entry_func(from_i, to_i, col, temp_idx_arr):
            temp_idx_arr[0] = from_i
            return temp_idx_arr[:1]
        
        # 定义出场选择函数：在入场后等待不同周期出场
        @njit
        def exit_func(from_i, to_i, col, temp_idx_arr):
            wait = col + 1  # 不同列等待不同周期
            temp_idx_arr[0] = from_i + wait
            if temp_idx_arr[0] < to_i:
                return temp_idx_arr[:1]
            return temp_idx_arr[:0]
        
        # 创建临时数组（重用内存）
        temp_idx_arr = np.empty((1,), dtype=np.int64)
        
        # 生成入场和出场信号
        entries, exits = pd.DataFrame.vbt.signals.generate_both(
            (10, 3),  # 10行3列
            entry_func, (temp_idx_arr,),
            exit_func, (temp_idx_arr,),
            entry_wait=2,  # 入场信号间隔2个周期
            exit_wait=1,   # 出场信号间隔1个周期
            index=pd.date_range('2020-01-01', periods=10),
            columns=['a', 'b', 'c']
        )
        
        print("入场信号:\n", entries)
        print("出场信号:\n", exits)
        ```
        
        应用场景：
        - 完整的交易策略信号生成
        - 配对交易信号创建
        - 策略回测数据准备
        - 信号模式研究
        
        注意事项：
        - 两个选择函数都必须是Numba编译的
        - 入场和出场信号应该有合理的时序关系
        - wait参数控制信号密度，避免过于频繁的交易
        - 建议使用临时数组重用内存，提高性能
        """
        checks.assert_not_none(entry_choice_func_nb)  # 验证入场选择函数不为空
        checks.assert_not_none(exit_choice_func_nb)   # 验证出场选择函数不为空
        checks.assert_numba_func(entry_choice_func_nb)  # 验证入场函数为Numba编译
        checks.assert_numba_func(exit_choice_func_nb)   # 验证出场函数为Numba编译
        
        # 设置默认参数
        if entry_args is None:
            entry_args = ()
        if exit_args is None:
            exit_args = ()

        # 处理形状参数，确保是二维元组
        if not isinstance(shape, tuple):
            shape = (shape, 1)  # 一维形状转换为二维
        elif isinstance(shape, tuple) and len(shape) == 1:
            shape = (shape[0], 1)  # 一维元组转换为二维

        # 调用底层Numba函数生成入场和出场信号
        result1, result2 = nb.generate_enex_nb(
            shape,
            entry_wait,
            exit_wait,
            entry_pick_first,
            exit_pick_first,
            entry_choice_func_nb, entry_args,
            exit_choice_func_nb, exit_args
        )
        
        # 根据形状返回适当的数据类型
        if cls.is_series():
            if shape[1] > 1:
                raise ValueError("Use DataFrame accessor")  # Series访问器不支持多列
            return pd.Series(result1[:, 0], **kwargs), pd.Series(result2[:, 0], **kwargs)  # 返回单列Series对
        return pd.DataFrame(result1, **kwargs), pd.DataFrame(result2, **kwargs)  # 返回多列DataFrame对

    def generate_exits(self,
                       exit_choice_func_nb: tp.ChoiceFunc, *args,
                       wait: int = 1,
                       until_next: bool = True,
                       skip_until_exit: bool = False,
                       pick_first: bool = False,
                       wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        基于现有入场信号生成出场信号
        
        功能说明：
        使用Numba编译的选择函数为现有的入场信号生成对应的出场信号。
        该方法实现了从入场信号到出场信号的转换逻辑，支持复杂的出场策略。
        出场信号可以基于时间、价格、技术指标等多种条件生成，满足不同的交易需求。
        
        参数说明：
            exit_choice_func_nb (callable): 出场信号选择函数
                - 函数签名: def func(from_i, to_i, col, *args) -> np.ndarray
                - 返回出场信号位置的索引数组
                - 必须使用@njit装饰器编译
            *args: 传递给出场选择函数的额外参数
            wait (int): 出场信号延迟周期数
                - 控制出场信号相对于入场信号的延迟
                - 0: 可能在同一bar产生两个信号
                - >0: 出场信号至少延迟wait个周期
            until_next (bool): 出场搜索范围控制
                - True: 在下一个入场信号之前搜索出场信号
                - False: 在整个剩余时间范围内搜索出场信号
            skip_until_exit (bool): 入场信号跳过控制
                - True: 跳过直到找到出场信号
                - False: 不跳过任何入场信号
            pick_first (bool): 是否只选择第一个出场信号
                - True: 在每个搜索范围内只选择第一个出场信号
                - False: 选择所有符合条件的出场信号
            wrap_kwargs (dict): 包装参数
                - 传递给ArrayWrapper.wrap的关键字参数
        
        返回值：
            tp.SeriesFrame: 生成的出场信号数据，True表示有出场信号
        
        选择函数示例：
        ```python
        from numba import njit
        import numpy as np
        
        @njit
        def exit_choice_func(from_i, to_i, col, temp_range):
            # 在搜索范围内生成连续的出场信号
            return temp_range[from_i:to_i]
        
        @njit
        def fixed_wait_exit(from_i, to_i, col, wait_periods):
            # 在入场后等待固定周期出场
            exit_idx = from_i + wait_periods
            if exit_idx < to_i:
                return np.array([exit_idx])
            return np.array([], dtype=np.int64)
        ```
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        from numba import njit
        
        # 创建入场信号
        entries = pd.DataFrame({
            'a': [True, False, False, False, False],
            'b': [True, False, True, False, True],
            'c': [True, True, True, False, False]
        })
        
        # 定义出场选择函数：填充入场信号后的所有空间
        @njit
        def fill_exit_func(from_i, to_i, col, temp_range):
            return temp_range[from_i:to_i]
        
        # 生成出场信号
        exits = entries.vbt.signals.generate_exits(
            fill_exit_func,
            np.arange(entries.shape[0]),  # 传递索引范围
            wait=1,  # 延迟1个周期
            until_next=True,  # 在下一个入场前搜索
            pick_first=False  # 选择所有出场信号
        )
        
        print("入场信号:\n", entries)
        print("出场信号:\n", exits)
        ```
        
        应用场景：
        - 基于入场信号的出场策略实现
        - 止损止盈信号生成
        - 时间基出场策略
        - 技术指标出场信号
        
        注意事项：
        - 出场选择函数必须是Numba编译的
        - until_next和skip_until_exit参数控制出场逻辑
        - wait参数影响出场信号的及时性
        - 建议根据实际交易需求调整参数
        """
        checks.assert_numba_func(exit_choice_func_nb)  # 验证出场选择函数为Numba编译

        # 调用底层Numba函数生成出场信号
        exits = nb.generate_ex_nb(
            self.to_2d_array(),  # 将当前信号数据转换为2D数组
            wait,
            until_next,
            skip_until_exit,
            pick_first,
            exit_choice_func_nb,
            *args
        )
        return self.wrapper.wrap(exits, group_by=False, **merge_dicts({}, wrap_kwargs))  # 包装并返回出场信号

    # ############# Filtering ############# #

    @class_or_instancemethod
    def clean(cls_or_self,
              *args,
              entry_first: bool = True,
              broadcast_kwargs: tp.KwargsLike = None,
              wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeTuple[tp.SeriesFrame]:
        """
        清理和优化信号序列
        
        功能说明：
        清理信号数据，确保信号序列的逻辑一致性和有效性。
        该方法支持两种模式：单信号清理和双信号（入场/出场）清理。
        对于单信号，使用first()方法选择第一个信号；对于双信号，使用clean_enex_nb
        确保入场和出场信号的配对关系正确。
        
        参数说明：
            *args: 输入信号数组
                - 1个数组: 单信号清理模式
                - 2个数组: 双信号清理模式（入场信号, 出场信号）
            entry_first (bool): 入场信号优先标志
                - True: 入场信号优先，确保每个入场都有对应的出场
                - False: 出场信号优先，确保每个出场都有对应的入场
            broadcast_kwargs (dict): 广播参数
                - 控制信号数组的广播行为
            wrap_kwargs (dict): 包装参数
                - 传递给ArrayWrapper.wrap的关键字参数
        
        返回值：
            单信号模式: tp.SeriesFrame - 清理后的信号数据
            双信号模式: tuple - (清理后的入场信号, 清理后的出场信号)
        
        清理逻辑：
        【单信号清理】：
        - 使用first()方法选择每个分区中的第一个信号
        - 移除重复和冲突的信号
        - 确保信号序列的简洁性
        
        【双信号清理】：
        - 确保入场和出场信号的一对一关系
        - 移除无法配对的信号
        - 根据entry_first参数调整配对优先级
        - 清理重叠和冲突的信号对
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        
        # 单信号清理示例
        signals = pd.Series([True, True, False, True, False, True])
        cleaned = signals.vbt.signals.clean()
        print("原始信号:", signals.values)
        print("清理后信号:", cleaned.values)
        # 输出:
        # 原始信号: [ True  True False  True False  True]
        # 清理后信号: [ True False False False False False]
        
        # 双信号清理示例
        entries = pd.Series([True, False, True, False, True])
        exits = pd.Series([False, True, False, True, False])
        
        clean_entries, clean_exits = pd.Series.vbt.signals.clean(
            entries, exits,
            entry_first=True  # 入场信号优先
        )
        
        print("原始入场:", entries.values)
        print("原始出场:", exits.values)
        print("清理后入场:", clean_entries.values)
        print("清理后出场:", clean_exits.values)
        ```
        
        应用场景：
        - 信号数据预处理和清洗
        - 入场出场信号配对优化
        - 策略信号逻辑验证
        - 回测数据质量保证
        
        注意事项：
        - 清理操作会改变原始信号序列
        - entry_first参数影响最终的配对结果
        - 建议在策略回测前进行信号清理
        - 清理后的信号更适合用于实际交易
        """
        # 处理类方法或实例方法调用
        if not isinstance(cls_or_self, type):
            args = (cls_or_self.obj, *args)  # 实例方法：添加当前对象作为第一个参数
        
        # 单信号清理模式
        if len(args) == 1:
            obj = args[0]
            if not isinstance(obj, (pd.Series, pd.DataFrame)):
                wrapper = ArrayWrapper.from_shape(np.asarray(obj).shape)
                obj = wrapper.wrap(obj)
            return obj.vbt.signals.first(wrap_kwargs=wrap_kwargs)  # 使用first方法清理单信号
        
        # 双信号清理模式
        elif len(args) == 2:
            if broadcast_kwargs is None:
                broadcast_kwargs = {}
            entries, exits = reshape_fns.broadcast(*args, **broadcast_kwargs)  # 广播信号数组
            entries_out, exits_out = nb.clean_enex_nb(  # 调用底层清理函数
                reshape_fns.to_2d_array(entries),
                reshape_fns.to_2d_array(exits),
                entry_first
            )
            return (
                ArrayWrapper.from_obj(entries).wrap(entries_out, group_by=False, **merge_dicts({}, wrap_kwargs)),  # 包装清理后的入场信号
                ArrayWrapper.from_obj(exits).wrap(exits_out, group_by=False, **merge_dicts({}, wrap_kwargs))  # 包装清理后的出场信号
            )
        else:
            raise ValueError("Either one or two arrays must be passed")  # 参数数量错误

    # ############# Random ############# #

    @classmethod
    def generate_random(cls,
                        shape: tp.RelaxedShape,
                        n: tp.Optional[tp.ArrayLike] = None,
                        prob: tp.Optional[tp.ArrayLike] = None,
                        pick_first: bool = False,
                        seed: tp.Optional[int] = None,
                        **kwargs) -> tp.SeriesFrame:
        """
        生成随机信号
        
        功能说明：
        使用随机算法生成信号数据，支持两种模式：数量控制和概率控制。
        该方法提供了灵活的信号生成机制，适用于策略测试、蒙特卡洛模拟和基准对比。
        生成的信号具有随机性，但可以通过seed参数确保结果的可重现性。
        
        参数说明：
            shape (tuple/int): 输出信号的形状
                - 整数: 表示一维信号的长度
                - 元组: 表示二维信号的(行数, 列数)
            n (array-like, optional): 信号数量控制
                - 标量: 每列生成相同数量的信号
                - 数组: 每列生成不同数量的信号，会广播到列数
                - 与prob参数互斥，只能设置其中一个
            prob (array-like, optional): 信号概率控制
                - 标量: 每个时点按相同概率生成信号
                - 数组: 每个时点按不同概率生成信号，会广播到shape
                - 与n参数互斥，只能设置其中一个
            pick_first (bool): 是否只选择第一个信号
                - True: 在每个搜索范围内只选择第一个信号
                - False: 选择所有符合条件的信号
                - 仅在概率模式下有效
            seed (int, optional): 随机种子
                - 确保结果的可重现性
                - None: 使用系统默认随机种子
            **kwargs: 传递给pandas构造函数的关键字参数
                - 包含index、columns等元数据信息
        
        返回值：
            tp.SeriesFrame: 生成的随机信号数据，True表示有信号
        
        生成模式：
        【数量控制模式 (n参数)】：
        - 为每列生成指定数量的随机信号
        - 信号位置在时间范围内随机分布
        - 适用于需要固定信号数量的场景
        
        【概率控制模式 (prob参数)】：
        - 每个时点按指定概率生成信号
        - 支持不同时点使用不同概率
        - 适用于需要控制信号密度的场景
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        
        # 数量控制模式：每列生成不同数量的信号
        signals_n = pd.DataFrame.vbt.signals.generate_random(
            (10, 3),  # 10行3列
            n=[2, 3, 1],  # 第1列2个信号，第2列3个信号，第3列1个信号
            seed=42,
            index=pd.date_range('2020-01-01', periods=10),
            columns=['a', 'b', 'c']
        )
        print("数量控制信号:\n", signals_n)
        
        # 概率控制模式：每个时点按概率生成信号
        signals_prob = pd.DataFrame.vbt.signals.generate_random(
            (10, 3),
            prob=0.3,  # 30%概率生成信号
            seed=42,
            index=pd.date_range('2020-01-01', periods=10),
            columns=['a', 'b', 'c']
        )
        print("概率控制信号:\n", signals_prob)
        
        # 不同概率模式：不同时点使用不同概率
        time_prob = np.linspace(0.1, 0.5, 10)  # 概率从10%增加到50%
        signals_time_prob = pd.DataFrame.vbt.signals.generate_random(
            (10, 3),
            prob=time_prob[:, None],  # 广播到3列
            seed=42,
            index=pd.date_range('2020-01-01', periods=10),
            columns=['a', 'b', 'c']
        )
        print("时间变化概率信号:\n", signals_time_prob)
        ```
        
        应用场景：
        - 策略压力测试和鲁棒性验证
        - 蒙特卡洛模拟和情景分析
        - 基准策略对比和性能评估
        - 信号模式研究和统计分析
        - 回测系统的数据准备
        
        注意事项：
        - n和prob参数不能同时设置
        - seed参数确保结果可重现，便于调试和对比
        - 生成的信号具有随机性，每次运行结果可能不同
        - 建议在策略开发初期使用随机信号进行快速测试
        """
        # 处理形状参数，确保是二维元组
        flex_2d = True
        if not isinstance(shape, tuple):
            flex_2d = False
            shape = (shape, 1)  # 一维形状转换为二维
        elif isinstance(shape, tuple) and len(shape) == 1:
            flex_2d = False
            shape = (shape[0], 1)  # 一维元组转换为二维

        # 参数验证：n和prob不能同时设置
        if n is not None and prob is not None:
            raise ValueError("Either n or prob should be set, not both")
        
        # 数量控制模式
        if n is not None:
            n = np.broadcast_to(n, shape[1])  # 广播到列数
            result = nb.generate_rand_nb(shape, n, seed=seed)  # 调用数量控制生成函数
        
        # 概率控制模式
        elif prob is not None:
            prob = np.broadcast_to(prob, shape)  # 广播到完整形状
            result = nb.generate_rand_by_prob_nb(shape, prob, pick_first, flex_2d, seed=seed)  # 调用概率控制生成函数
        
        else:
            raise ValueError("At least n or prob should be set")  # 至少需要设置一个参数

        # 根据形状返回适当的数据类型
        if cls.is_series():
            if shape[1] > 1:
                raise ValueError("Use DataFrame accessor")  # Series访问器不支持多列
            return pd.Series(result[:, 0], **kwargs)  # 返回单列Series
        return pd.DataFrame(result, **kwargs)  # 返回多列DataFrame

    @classmethod
    def generate_random_both(cls,
                             shape: tp.RelaxedShape,
                             n: tp.Optional[tp.ArrayLike] = None,
                             entry_prob: tp.Optional[tp.ArrayLike] = None,
                             exit_prob: tp.Optional[tp.ArrayLike] = None,
                             seed: tp.Optional[int] = None,
                             entry_wait: int = 1,
                             exit_wait: int = 1,
                             entry_pick_first: bool = True,
                             exit_pick_first: bool = True,
                             **kwargs) -> tp.Tuple[tp.SeriesFrame, tp.SeriesFrame]:
        """
        随机生成入场和出场信号对
        
        功能说明：
        使用随机算法同时生成入场和出场信号，支持两种模式：数量控制和概率控制。
        该方法创建配对的交易信号序列，适用于完整的交易策略测试和模拟。
        入场和出场信号可以有不同的生成逻辑和时间间隔，支持复杂的交易场景。
        
        参数说明：
            shape (tuple/int): 输出信号的形状
                - 整数: 表示一维信号的长度
                - 元组: 表示二维信号的(行数, 列数)
            n (array-like, optional): 信号对数量控制
                - 标量: 每列生成相同数量的信号对
                - 数组: 每列生成不同数量的信号对，会广播到列数
                - 与概率参数互斥，只能设置其中一个
            entry_prob (array-like, optional): 入场信号概率
                - 标量: 每个时点按相同概率生成入场信号
                - 数组: 每个时点按不同概率生成入场信号，会广播到shape
                - 需要与exit_prob同时设置
            exit_prob (array-like, optional): 出场信号概率
                - 标量: 每个时点按相同概率生成出场信号
                - 数组: 每个时点按不同概率生成出场信号，会广播到shape
                - 需要与entry_prob同时设置
            seed (int, optional): 随机种子
                - 确保结果的可重现性
                - None: 使用系统默认随机种子
            entry_wait (int): 入场信号间的最小等待周期
                - 控制入场信号的密度
            exit_wait (int): 出场信号间的最小等待周期
                - 控制出场信号的密度
            entry_pick_first (bool): 是否只选择第一个入场信号
                - True: 在每个搜索范围内只选择第一个入场信号
                - False: 选择所有符合条件的入场信号
            exit_pick_first (bool): 是否只选择第一个出场信号
                - True: 在每个搜索范围内只选择第一个出场信号
                - False: 选择所有符合条件的出场信号
            **kwargs: 传递给pandas构造函数的关键字参数
        
        返回值：
            tuple: (入场信号数据, 出场信号数据)
                - 两个数据对象都是布尔型，True表示有信号
        
        生成模式：
        【数量控制模式 (n参数)】：
        - 为每列生成指定数量的入场和出场信号对
        - 信号位置在时间范围内随机分布
        - 入场和出场信号交替生成
        
        【概率控制模式 (entry_prob + exit_prob)】：
        - 每个时点按指定概率生成入场和出场信号
        - 支持不同时点使用不同概率
        - 入场和出场信号独立生成
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        
        # 数量控制模式：每列生成不同数量的信号对
        entries_n, exits_n = pd.DataFrame.vbt.signals.generate_random_both(
            (10, 3),  # 10行3列
            n=[2, 3, 1],  # 第1列2对信号，第2列3对信号，第3列1对信号
            seed=42,
            entry_wait=2,  # 入场信号间隔2个周期
            exit_wait=1,   # 出场信号间隔1个周期
            index=pd.date_range('2020-01-01', periods=10),
            columns=['a', 'b', 'c']
        )
        print("数量控制入场信号:\n", entries_n)
        print("数量控制出场信号:\n", exits_n)
        
        # 概率控制模式：每个时点按概率生成信号
        entries_prob, exits_prob = pd.DataFrame.vbt.signals.generate_random_both(
            (10, 3),
            entry_prob=0.3,  # 30%概率生成入场信号
            exit_prob=0.2,   # 20%概率生成出场信号
            seed=42,
            entry_wait=1, exit_wait=1,
            index=pd.date_range('2020-01-01', periods=10),
            columns=['a', 'b', 'c']
        )
        print("概率控制入场信号:\n", entries_prob)
        print("概率控制出场信号:\n", exits_prob)
        ```
        
        应用场景：
        - 完整交易策略的随机测试
        - 配对交易信号生成
        - 策略回测数据准备
        - 信号模式研究和分析
        - 蒙特卡洛模拟和压力测试
        
        注意事项：
        - n参数与概率参数不能同时设置
        - entry_prob和exit_prob必须同时设置
        - wait参数控制信号密度，避免过于频繁的交易
        - seed参数确保结果可重现，便于调试和对比
        """
        # 处理形状参数，确保是二维元组
        flex_2d = True
        if not isinstance(shape, tuple):
            flex_2d = False
            shape = (shape, 1)  # 一维形状转换为二维
        elif isinstance(shape, tuple) and len(shape) == 1:
            flex_2d = False
            shape = (shape[0], 1)  # 一维元组转换为二维

        # 参数验证：n与概率参数不能同时设置
        if n is not None and (entry_prob is not None or exit_prob is not None):
            raise ValueError("Either n or any of the entry_prob and exit_prob should be set, not both")
        
        # 数量控制模式
        if n is not None:
            n = np.broadcast_to(n, shape[1])  # 广播到列数
            entries, exits = nb.generate_rand_enex_nb(shape, n, entry_wait, exit_wait, seed=seed)  # 调用数量控制生成函数
        
        # 概率控制模式
        elif entry_prob is not None and exit_prob is not None:
            entry_prob = np.broadcast_to(entry_prob, shape)  # 广播入场概率到完整形状
            exit_prob = np.broadcast_to(exit_prob, shape)    # 广播出场概率到完整形状
            entries, exits = nb.generate_rand_enex_by_prob_nb(  # 调用概率控制生成函数
                shape,
                entry_prob,
                exit_prob,
                entry_wait,
                exit_wait,
                entry_pick_first,
                exit_pick_first,
                flex_2d,
                seed=seed
            )
        else:
            raise ValueError("At least n, or entry_prob and exit_prob should be set")  # 参数设置错误

        # 根据形状返回适当的数据类型
        if cls.is_series():
            if shape[1] > 1:
                raise ValueError("Use DataFrame accessor")  # Series访问器不支持多列
            return pd.Series(entries[:, 0], **kwargs), pd.Series(exits[:, 0], **kwargs)  # 返回单列Series对
        return pd.DataFrame(entries, **kwargs), pd.DataFrame(exits, **kwargs)  # 返回多列DataFrame对

    def generate_random_exits(self,
                              prob: tp.Optional[tp.ArrayLike] = None,
                              seed: tp.Optional[int] = None,
                              wait: int = 1,
                              until_next: bool = True,
                              skip_until_exit: bool = False,
                              wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        为现有入场信号生成随机出场信号
        
        功能说明：
        基于现有的入场信号生成随机的出场信号，支持两种模式：固定出场和概率出场。
        该方法为每个入场信号生成对应的出场信号，适用于需要随机出场策略的场景。
        生成的出场信号具有随机性，但可以通过seed参数确保结果的可重现性。
        
        参数说明：
            prob (array-like, optional): 出场信号概率
                - None: 为每个入场信号生成恰好一个出场信号
                - 标量: 每个时点按相同概率生成出场信号
                - 数组: 每个时点按不同概率生成出场信号，会广播到信号形状
            seed (int, optional): 随机种子
                - 确保结果的可重现性
                - None: 使用系统默认随机种子
            wait (int): 出场信号延迟周期数
                - 控制出场信号相对于入场信号的延迟
                - 0: 可能在同一bar产生两个信号
                - >0: 出场信号至少延迟wait个周期
            until_next (bool): 出场搜索范围控制
                - True: 在下一个入场信号之前搜索出场信号
                - False: 在整个剩余时间范围内搜索出场信号
            skip_until_exit (bool): 入场信号跳过控制
                - True: 跳过直到找到出场信号
                - False: 不跳过任何入场信号
            wrap_kwargs (dict): 包装参数
                - 传递给ArrayWrapper.wrap的关键字参数
        
        返回值：
            tp.SeriesFrame: 生成的随机出场信号数据，True表示有出场信号
        
        生成模式：
        【固定出场模式 (prob=None)】：
        - 为每个入场信号生成恰好一个出场信号
        - 出场位置在入场后的时间范围内随机分布
        - 确保每个入场都有对应的出场
        
        【概率出场模式 (prob设置)】：
        - 每个时点按指定概率生成出场信号
        - 支持不同时点使用不同概率
        - 出场信号数量不固定
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        
        # 创建入场信号
        entries = pd.DataFrame({
            'a': [True, False, False, False, False],
            'b': [True, False, True, False, True],
            'c': [True, True, True, False, False]
        })
        
        # 固定出场模式：为每个入场生成恰好一个出场
        exits_fixed = entries.vbt.signals.generate_random_exits(
            seed=42,
            wait=1,  # 延迟1个周期
            until_next=True,  # 在下一个入场前搜索
            skip_until_exit=False  # 不跳过入场信号
        )
        print("固定出场模式:\n", exits_fixed)
        
        # 概率出场模式：每个时点按概率生成出场
        exits_prob = entries.vbt.signals.generate_random_exits(
            prob=0.3,  # 30%概率生成出场信号
            seed=42,
            wait=1,
            until_next=True,
            skip_until_exit=False
        )
        print("概率出场模式:\n", exits_prob)
        
        # 时间变化概率：不同时点使用不同概率
        time_prob = np.linspace(0.1, 0.5, 5)  # 概率从10%增加到50%
        exits_time_prob = entries.vbt.signals.generate_random_exits(
            prob=time_prob[:, None],  # 广播到3列
            seed=42,
            wait=1,
            until_next=True,
            skip_until_exit=False
        )
        print("时间变化概率出场:\n", exits_time_prob)
        ```
        
        应用场景：
        - 随机出场策略测试
        - 策略鲁棒性验证
        - 蒙特卡洛模拟
        - 基准策略对比
        - 信号模式研究
        
        注意事项：
        - prob参数控制出场信号的生成方式
        - seed参数确保结果可重现，便于调试和对比
        - wait参数影响出场信号的及时性
        - until_next和skip_until_exit参数控制出场逻辑
        - 生成的信号具有随机性，每次运行结果可能不同
        """
        # 概率出场模式
        if prob is not None:
            obj, prob = reshape_fns.broadcast(self.obj, prob, keep_raw=[False, True])  # 广播信号和概率
            exits = nb.generate_rand_ex_by_prob_nb(  # 调用概率出场生成函数
                reshape_fns.to_2d_array(obj),
                prob,
                wait,
                until_next,
                skip_until_exit,
                obj.ndim == 2,
                seed=seed
            )
            return ArrayWrapper.from_obj(obj).wrap(exits, group_by=False, **merge_dicts({}, wrap_kwargs))  # 包装并返回出场信号
        
        # 固定出场模式
        exits = nb.generate_rand_ex_nb(  # 调用固定出场生成函数
            self.to_2d_array(),
            wait,
            until_next,
            skip_until_exit,
            seed=seed
        )
        return self.wrapper.wrap(exits, group_by=False, **merge_dicts({}, wrap_kwargs))  # 包装并返回出场信号

    def generate_stop_exits(self,
                            ts: tp.ArrayLike,
                            stop: tp.ArrayLike,
                            trailing: tp.ArrayLike = False,
                            entry_wait: int = 1,
                            exit_wait: int = 1,
                            until_next: bool = True,
                            skip_until_exit: bool = False,
                            pick_first: bool = True,
                            chain: bool = False,
                            broadcast_kwargs: tp.KwargsLike = None,
                            wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeTuple[tp.SeriesFrame]:
        """
        基于价格阈值生成止损出场信号
        
        功能说明：
        根据时间序列数据(ts)和止损阈值(stop)为现有的入场信号生成对应的出场信号。
        该方法实现了基于价格触发的止损机制，支持固定止损和追踪止损两种模式。
        当价格达到止损条件时，自动生成出场信号，实现风险控制。
        
        参数说明：
            ts (array_like): 时间序列价格数据
                - 用于计算价格变化和触发止损条件
                - 可以是Series、DataFrame或numpy数组
            stop (array_like): 止损阈值
                - 正数: 表示止盈阈值(价格上升触发)
                - 负数: 表示止损阈值(价格下降触发)
                - 0: 表示不设置止损
            trailing (array_like): 是否启用追踪止损
                - True: 启用追踪止损，止损价格会随价格变动调整
                - False: 使用固定止损价格
            entry_wait (int): 入场信号间的最小等待周期
                - 控制入场信号的密度，避免过于频繁的交易
            exit_wait (int): 出场信号间的最小等待周期
                - 控制出场信号的密度，确保出场信号的稳定性
            until_next (bool): 出场搜索范围控制
                - True: 在下一个入场信号之前搜索出场信号
                - False: 在整个剩余时间范围内搜索出场信号
            skip_until_exit (bool): 入场信号跳过控制
                - True: 跳过直到找到出场信号
                - False: 不跳过任何入场信号
            pick_first (bool): 是否只选择第一个出场信号
                - True: 在每个搜索范围内只选择第一个出场信号
                - False: 选择所有符合条件的出场信号
            chain (bool): 是否启用链式模式
                - True: 返回新的入场信号和出场信号对
                - False: 只返回出场信号
            broadcast_kwargs (dict): 广播参数
                - 控制输入数组的广播行为
            wrap_kwargs (dict): 包装参数
                - 传递给ArrayWrapper.wrap的关键字参数
        
        返回值：
            非链式模式: tp.SeriesFrame - 生成的出场信号数据
            链式模式: tuple - (新的入场信号, 出场信号)
        
        止损逻辑：
        【固定止损】：
        - 基于入场价格计算固定的止损价格
        - 当价格触及止损价格时触发出场信号
        - 止损价格在整个持仓期间保持不变
        
        【追踪止损】：
        - 止损价格会随着价格变动而调整
        - 当价格向有利方向移动时，止损价格相应调整
        - 当价格向不利方向移动时，止损价格保持不变
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        
        # 创建入场信号和价格数据
        entries = pd.DataFrame({
            'a': [True, False, False, False, False],
            'b': [True, False, True, False, True],
            'c': [True, True, True, False, False]
        })
        
        # 价格数据
        prices = pd.Series([100, 105, 98, 110, 95])
        
        # 生成固定止损出场信号(5%止损)
        stop_exits = entries.vbt.signals.generate_stop_exits(
            prices, 
            stop=-0.05,  # 5%止损
            trailing=False,  # 固定止损
            exit_wait=1,  # 延迟1个周期
            until_next=True  # 在下一个入场前搜索
        )
        
        # 生成追踪止损出场信号(3%追踪止损)
        trail_exits = entries.vbt.signals.generate_stop_exits(
            prices,
            stop=-0.03,  # 3%追踪止损
            trailing=True,  # 追踪止损
            exit_wait=1,
            until_next=True
        )
        
        print("入场信号:\n", entries)
        print("固定止损出场信号:\n", stop_exits)
        print("追踪止损出场信号:\n", trail_exits)
        ```
        
        应用场景：
        - 风险管理系统的止损信号生成
        - 量化交易策略的风险控制
        - 技术分析指标的止损应用
        - 策略回测的止损逻辑实现
        
        注意事项：
        - 止损阈值的选择直接影响策略的风险收益比
        - 追踪止损适合趋势性市场，固定止损适合震荡市场
        - 建议根据市场特性和策略需求调整wait参数
        - 链式模式会修改入场信号，需要谨慎使用
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        entries = self.obj  # 获取当前信号数据作为入场信号

        keep_raw = (False, True, True, True)  # 指定哪些参数保持原始形状
        broadcast_kwargs = merge_dicts(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)  # 合并广播参数
        entries, ts, stop, trailing = reshape_fns.broadcast(  # 广播所有输入数组到相同形状
            entries, ts, stop, trailing, **broadcast_kwargs, keep_raw=keep_raw)

        # 执行信号生成
        if chain:  # 链式模式：生成新的入场和出场信号对
            new_entries, exits = nb.generate_stop_enex_nb(  # 调用Numba函数生成入场和出场信号
                reshape_fns.to_2d_array(entries),  # 转换为2D数组
                ts,  # 时间序列数据
                stop,  # 止损阈值
                trailing,  # 追踪止损标志
                entry_wait,  # 入场等待周期
                exit_wait,  # 出场等待周期
                pick_first,  # 是否选择第一个信号
                entries.ndim == 2  # 是否为2D数组
            )
            return ArrayWrapper.from_obj(entries).wrap(new_entries, group_by=False, **merge_dicts({}, wrap_kwargs)), \
                   ArrayWrapper.from_obj(entries).wrap(exits, group_by=False, **merge_dicts({}, wrap_kwargs))  # 返回新的入场和出场信号
        else:  # 非链式模式：只生成出场信号
            if skip_until_exit and until_next:
                warnings.warn("skip_until_exit=True has only effect when until_next=False", stacklevel=2)  # 参数冲突警告
            exits = nb.generate_stop_ex_nb(  # 调用Numba函数生成出场信号
                reshape_fns.to_2d_array(entries),  # 转换为2D数组
                ts,  # 时间序列数据
                stop,  # 止损阈值
                trailing,  # 追踪止损标志
                exit_wait,  # 出场等待周期
                until_next,  # 搜索范围控制
                skip_until_exit,  # 跳过控制
                pick_first,  # 是否选择第一个信号
                entries.ndim == 2  # 是否为2D数组
            )
            return ArrayWrapper.from_obj(entries).wrap(exits, group_by=False, **merge_dicts({}, wrap_kwargs))  # 返回出场信号

    def generate_ohlc_stop_exits(self,
                                 open: tp.ArrayLike,
                                 high: tp.Optional[tp.ArrayLike] = None,
                                 low: tp.Optional[tp.ArrayLike] = None,
                                 close: tp.Optional[tp.ArrayLike] = None,
                                 is_open_safe: bool = True,
                                 out_dict: tp.Optional[tp.Dict[str, tp.ArrayLike]] = None,
                                 sl_stop: tp.ArrayLike = np.nan,
                                 sl_trail: tp.ArrayLike = False,
                                 tp_stop: tp.ArrayLike = np.nan,
                                 reverse: tp.ArrayLike = False,
                                 entry_wait: int = 1,
                                 exit_wait: int = 1,
                                 until_next: bool = True,
                                 skip_until_exit: bool = False,
                                 pick_first: bool = True,
                                 chain: bool = False,
                                 broadcast_kwargs: tp.KwargsLike = None,
                                 wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeTuple[tp.SeriesFrame]:
        """
        基于OHLC价格数据生成止损止盈出场信号
        
        功能说明：
        使用完整的OHLC(开高低收)价格数据为入场信号生成止损和止盈出场信号。
        该方法支持更精确的价格触发机制，能够处理开盘价、最高价、最低价、收盘价的不同组合。
        支持止损、止盈、追踪止损等多种风险控制策略，是量化交易中重要的风险管理工具。
        
        参数说明：
            open (array_like): 开盘价数据
                - 用于计算止损止盈的基准价格
                - 可以是Series、DataFrame或numpy数组
            high (array_like): 最高价数据，默认为None
                - None时使用open作为最高价
                - 用于止盈触发条件判断
            low (array_like): 最低价数据，默认为None
                - None时使用open作为最低价
                - 用于止损触发条件判断
            close (array_like): 收盘价数据，默认为None
                - None时使用open作为收盘价
                - 用于价格变化计算
            is_open_safe (bool): 开盘价安全标志
                - True: 使用开盘价作为基准价格
                - False: 使用前一日收盘价作为基准价格
            out_dict (dict): 输出字典，默认为None
                - 用于返回止损价格和止损类型信息
                - 包含'stop_price'和'stop_type'两个键
            sl_stop (array_like): 止损阈值，默认为np.nan
                - 负数: 表示止损百分比
                - np.nan: 表示不设置止损
            sl_trail (array_like): 是否启用追踪止损，默认为False
                - True: 启用追踪止损
                - False: 使用固定止损
            tp_stop (array_like): 止盈阈值，默认为np.nan
                - 正数: 表示止盈百分比
                - np.nan: 表示不设置止盈
            reverse (array_like): 是否反向操作，默认为False
                - True: 做空策略，止损在上方，止盈在下方
                - False: 做多策略，止损在下方，止盈在上方
            entry_wait (int): 入场信号间的最小等待周期
            exit_wait (int): 出场信号间的最小等待周期
            until_next (bool): 出场搜索范围控制
            skip_until_exit (bool): 入场信号跳过控制
            pick_first (bool): 是否只选择第一个出场信号
            chain (bool): 是否启用链式模式
            broadcast_kwargs (dict): 广播参数
            wrap_kwargs (dict): 包装参数
        
        返回值：
            非链式模式: tp.SeriesFrame - 生成的出场信号数据
            链式模式: tuple - (新的入场信号, 出场信号)
            同时会更新out_dict中的'stop_price'和'stop_type'信息
        
        止损止盈逻辑：
        【做多策略】：
        - 止损价格 = 入场价格 * (1 + sl_stop)
        - 止盈价格 = 入场价格 * (1 + tp_stop)
        - 当最低价触及止损价格时触发止损
        - 当最高价触及止盈价格时触发止盈
        
        【做空策略】：
        - 止损价格 = 入场价格 * (1 - sl_stop)
        - 止盈价格 = 入场价格 * (1 - tp_stop)
        - 当最高价触及止损价格时触发止损
        - 当最低价触及止盈价格时触发止盈
        
        【追踪止损】：
        - 止损价格会随着价格变动而调整
        - 保持与当前价格的固定距离
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        from vectorbt.signals.enums import StopType
        
        # 创建入场信号
        entries = pd.DataFrame({
            'a': [True, False, False, False, False],
            'b': [True, False, True, False, True],
            'c': [True, True, True, False, False]
        })
        
        # 创建OHLC价格数据
        price_data = pd.DataFrame({
            'open': [100, 101, 99, 102, 98],
            'high': [102, 103, 101, 104, 100],
            'low': [98, 99, 97, 100, 96],
            'close': [101, 99, 102, 98, 103]
        })
        
        # 创建输出字典
        out_dict = {}
        
        # 生成止损止盈出场信号
        exits = entries.vbt.signals.generate_ohlc_stop_exits(
            price_data['open'], 
            price_data['high'], 
            price_data['low'], 
            price_data['close'],
            sl_stop=-0.05,  # 5%止损
            sl_trail=True,  # 追踪止损
            tp_stop=0.10,   # 10%止盈
            out_dict=out_dict
        )
        
        print("入场信号:\n", entries)
        print("出场信号:\n", exits)
        print("止损价格:\n", out_dict['stop_price'])
        print("止损类型:\n", out_dict['stop_type'].vbt(mapping=StopType).apply_mapping())
        ```
        
        应用场景：
        - 完整的量化交易策略风险控制
        - 技术分析指标的止损止盈实现
        - 高频交易的风险管理
        - 策略回测的精确模拟
        
        注意事项：
        - OHLC数据必须具有时间对齐性
        - 止损止盈阈值的选择需要平衡风险和收益
        - 追踪止损适合趋势性市场
        - 建议使用out_dict获取详细的止损信息
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        entries = self.obj  # 获取当前信号数据作为入场信号

        # 设置默认价格数据
        if high is None:
            high = open  # 如果未提供最高价，使用开盘价
        if low is None:
            low = open   # 如果未提供最低价，使用开盘价
        if close is None:
            close = open # 如果未提供收盘价，使用开盘价
        
        # 处理输出字典
        if out_dict is None:
            out_dict_passed = False  # 标记是否传入了out_dict
            out_dict = {}  # 创建空字典
        else:
            out_dict_passed = True   # 标记传入了out_dict
        
        # 获取止损价格和止损类型输出数组
        stop_price_out = out_dict.get('stop_price', np.nan if out_dict_passed else None)
        stop_type_out = out_dict.get('stop_type', -1 if out_dict_passed else None)
        out_args = ()  # 初始化输出参数元组
        
        # 构建输出参数
        if stop_price_out is not None:
            out_args += (stop_price_out,)  # 添加止损价格输出
        if stop_type_out is not None:
            out_args += (stop_type_out,)   # 添加止损类型输出

        keep_raw = (False, True, True, True, True, True, True, True, True) + (False,) * len(out_args)  # 指定保持原始形状的参数
        broadcast_kwargs = merge_dicts(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)  # 合并广播参数
        
        # 广播所有输入数组到相同形状
        entries, open, high, low, close, sl_stop, sl_trail, tp_stop, reverse, *out_args = reshape_fns.broadcast(
            entries, open, high, low, close, sl_stop, sl_trail, tp_stop, reverse, *out_args,
            **broadcast_kwargs, keep_raw=keep_raw)
        
        # 处理止损价格输出数组
        if stop_price_out is None:
            stop_price_out = np.empty_like(entries, dtype=np.float64)  # 创建空数组
        else:
            stop_price_out = out_args[0]  # 获取第一个输出参数
            out_args = out_args[1:]       # 移除已处理的参数
        
        # 处理止损类型输出数组
        if stop_type_out is None:
            stop_type_out = np.empty_like(entries, dtype=np.int64)  # 创建空数组
        else:
            stop_type_out = out_args[0]   # 获取第一个输出参数
        
        # 转换为2D数组
        stop_price_out = reshape_fns.to_2d_array(stop_price_out)
        stop_type_out = reshape_fns.to_2d_array(stop_type_out)

        # 执行信号生成
        if chain:  # 链式模式：生成新的入场和出场信号对
            new_entries, exits = nb.generate_ohlc_stop_enex_nb(  # 调用Numba函数生成入场和出场信号
                reshape_fns.to_2d_array(entries),  # 转换为2D数组
                open,    # 开盘价
                high,    # 最高价
                low,     # 最低价
                close,   # 收盘价
                stop_price_out,  # 止损价格输出
                stop_type_out,   # 止损类型输出
                sl_stop,         # 止损阈值
                sl_trail,        # 追踪止损标志
                tp_stop,         # 止盈阈值
                reverse,         # 反向操作标志
                is_open_safe,    # 开盘价安全标志
                entry_wait,      # 入场等待周期
                exit_wait,       # 出场等待周期
                pick_first,      # 是否选择第一个信号
                entries.ndim == 2  # 是否为2D数组
            )
            # 更新输出字典
            out_dict['stop_price'] = ArrayWrapper.from_obj(entries).wrap(
                stop_price_out, group_by=False, **merge_dicts({}, wrap_kwargs))
            out_dict['stop_type'] = ArrayWrapper.from_obj(entries).wrap(
                stop_type_out, group_by=False, **merge_dicts({}, wrap_kwargs))
            return ArrayWrapper.from_obj(entries).wrap(new_entries, group_by=False, **merge_dicts({}, wrap_kwargs)), \
                   ArrayWrapper.from_obj(entries).wrap(exits, group_by=False, **merge_dicts({}, wrap_kwargs))  # 返回新的入场和出场信号
        else:  # 非链式模式：只生成出场信号
            if skip_until_exit and until_next:
                warnings.warn("skip_until_exit=True has only effect when until_next=False", stacklevel=2)  # 参数冲突警告
            exits = nb.generate_ohlc_stop_ex_nb(  # 调用Numba函数生成出场信号
                reshape_fns.to_2d_array(entries),  # 转换为2D数组
                open,    # 开盘价
                high,    # 最高价
                low,     # 最低价
                close,   # 收盘价
                stop_price_out,  # 止损价格输出
                stop_type_out,   # 止损类型输出
                sl_stop,         # 止损阈值
                sl_trail,        # 追踪止损标志
                tp_stop,         # 止盈阈值
                reverse,         # 反向操作标志
                is_open_safe,    # 开盘价安全标志
                exit_wait,       # 出场等待周期
                until_next,      # 搜索范围控制
                skip_until_exit, # 跳过控制
                pick_first,      # 是否选择第一个信号
                entries.ndim == 2  # 是否为2D数组
            )
            # 更新输出字典
            out_dict['stop_price'] = ArrayWrapper.from_obj(entries).wrap(
                stop_price_out, group_by=False, **merge_dicts({}, wrap_kwargs))
            out_dict['stop_type'] = ArrayWrapper.from_obj(entries).wrap(
                stop_type_out, group_by=False, **merge_dicts({}, wrap_kwargs))
            return ArrayWrapper.from_obj(entries).wrap(exits, group_by=False, **merge_dicts({}, wrap_kwargs))  # 返回出场信号

    # ############# Ranges ############# #

    def between_ranges(self,
                       other: tp.Optional[tp.ArrayLike] = None,
                       from_other: bool = False,
                       broadcast_kwargs: tp.KwargsLike = None,
                       group_by: tp.GroupByLike = None,
                       attach_ts: bool = True,
                       attach_other: bool = False,
                       **kwargs) -> Ranges:
        """
        分析信号之间的时间范围
        
        功能说明：
        分析信号数据中的时间范围，计算信号之间的间隔和持续时间。
        该方法支持单信号分析和双信号对比分析，能够识别信号的时间分布特征。
        返回Ranges对象，包含详细的范围信息，便于后续的时间序列分析。
        
        参数说明：
            other (array_like): 第二个信号数组，默认为None
                - None: 单信号分析模式，分析当前信号的范围
                - 数组: 双信号分析模式，分析两个信号之间的关系
            from_other (bool): 遍历方向控制，默认为False
                - True: 以other信号为基准进行遍历
                - False: 以当前信号为基准进行遍历
            broadcast_kwargs (dict): 广播参数，默认为None
                - 控制输入数组的广播行为
            group_by (GroupByLike): 分组参数，默认为None
                - 指定分组方式，支持按列分组
            attach_ts (bool): 是否附加时间序列，默认为True
                - True: 在Ranges对象中包含时间序列信息
                - False: 不包含时间序列信息
            attach_other (bool): 是否附加other信号，默认为False
                - True: 在Ranges对象中包含other信号信息
                - False: 不包含other信号信息
            **kwargs: 传递给Ranges构造函数的关键字参数
        
        返回值：
            Ranges: 范围分析结果对象，包含以下信息：
                - 范围ID、列索引、开始时间、结束时间
                - 持续时间、状态等详细信息
        
        分析模式：
        【单信号分析】：
        - 分析当前信号中True值之间的时间间隔
        - 计算每个信号分区的持续时间
        - 识别信号的时间分布模式
        
        【双信号分析】：
        - 分析两个信号之间的时间关系
        - 计算从一个信号到另一个信号的时间间隔
        - 支持不同的遍历方向
        
        使用示例：
        ```python
        import pandas as pd
        
        # 单信号范围分析
        signals = pd.Series([True, False, False, True, False, True, True])
        ranges = signals.vbt.signals.between_ranges()
        print("范围记录:\n", ranges.records_readable)
        print("持续时间:", ranges.duration.values)
        
        # 双信号范围分析
        signals1 = pd.Series([True, True, True, False, False])
        signals2 = pd.Series([False, False, True, False, True])
        
        # 以signals1为基准分析
        ranges1 = signals1.vbt.signals.between_ranges(other=signals2)
        print("以signals1为基准的范围:\n", ranges1.records_readable)
        
        # 以signals2为基准分析
        ranges2 = signals1.vbt.signals.between_ranges(other=signals2, from_other=True)
        print("以signals2为基准的范围:\n", ranges2.records_readable)
        ```
        
        应用场景：
        - 信号时间分布分析
        - 交易策略的时间特征研究
        - 信号间隔统计和优化
        - 策略回测的时间分析
        
        注意事项：
        - 输入信号必须是布尔型数据
        - 时间索引有助于更精确的分析
        - 双信号分析需要确保信号对齐
        - 分组分析可以处理多列信号数据
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}

        if other is None:  # 单信号分析模式
            # 分析当前信号的范围
            range_records = nb.between_ranges_nb(self.to_2d_array())  # 调用Numba函数分析范围
            wrapper = self.wrapper  # 使用当前对象的包装器
            to_attach = self.obj    # 附加当前信号数据
        else:  # 双信号分析模式
            # 广播两个信号数组到相同形状
            obj, other = reshape_fns.broadcast(self.obj, other, **broadcast_kwargs)
            # 分析两个信号之间的范围关系
            range_records = nb.between_two_ranges_nb(
                reshape_fns.to_2d_array(obj),      # 第一个信号数组
                reshape_fns.to_2d_array(other),    # 第二个信号数组
                from_other=from_other              # 遍历方向
            )
            wrapper = ArrayWrapper.from_obj(obj)   # 创建新的包装器
            to_attach = other if attach_other else obj  # 选择要附加的信号数据
        
        # 创建并返回Ranges对象
        return Ranges(
            wrapper,                    # 数组包装器
            range_records,              # 范围记录
            ts=to_attach if attach_ts else None,  # 时间序列数据
            **kwargs                    # 其他参数
        ).regroup(group_by)            # 应用分组并返回

    def partition_ranges(self, group_by: tp.GroupByLike = None, attach_ts: bool = True, **kwargs) -> Ranges:
        """
        分析信号分区的范围信息
        
        功能说明：
        将连续的True信号识别为一个分区，分析每个分区的范围特征。
        该方法能够识别信号的分区结构，计算每个分区的开始时间、结束时间和持续时间。
        对于量化交易策略分析，分区范围信息有助于理解信号的聚集特征。
        
        参数说明：
            group_by (GroupByLike): 分组参数，默认为None
                - 指定分组方式，支持按列分组
            attach_ts (bool): 是否附加时间序列，默认为True
                - True: 在Ranges对象中包含时间序列信息
                - False: 不包含时间序列信息
            **kwargs: 传递给Ranges构造函数的关键字参数
        
        返回值：
            Ranges: 分区范围分析结果对象，包含以下信息：
                - 分区ID、列索引、开始时间、结束时间
                - 分区持续时间、状态等详细信息
        
        分区识别逻辑：
        - 连续的True值被识别为一个分区
        - 分区从第一个True值开始，到最后一个True值结束
        - 分区之间用False值分隔
        - 每个分区都有唯一的ID和范围信息
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建包含分区的信号数据
        signals = pd.Series([True, True, True, False, True, True])
        ranges = signals.vbt.signals.partition_ranges()
        
        print("分区范围记录:\n", ranges.records_readable)
        print("分区持续时间:", ranges.duration.values)
        # 输出示例：
        # 分区范围记录:
        #    Range Id  Column  Start Timestamp  End Timestamp  Status
        # 0         0       0                0              3  Closed
        # 1         1       0                4              5    Open
        # 分区持续时间: [3, 1]
        ```
        
        应用场景：
        - 信号聚集性分析
        - 交易策略的持仓时间分析
        - 信号模式识别
        - 策略优化和参数调整
        
        注意事项：
        - 分区分析基于连续的True值
        - 最后一个分区可能是开放状态(未结束)
        - 时间索引有助于更精确的分区分析
        """
        range_records = nb.partition_ranges_nb(self.to_2d_array())  # 调用Numba函数分析分区范围
        return Ranges(
            self.wrapper,                    # 数组包装器
            range_records,                   # 分区范围记录
            ts=self.obj if attach_ts else None,  # 时间序列数据
            **kwargs                         # 其他参数
        ).regroup(group_by)                 # 应用分组并返回

    def between_partition_ranges(self, group_by: tp.GroupByLike = None, attach_ts: bool = True, **kwargs) -> Ranges:
        """
        分析分区之间的时间间隔
        
        功能说明：
        计算信号分区之间的时间间隔，分析分区的时间分布特征。
        该方法识别每个分区的边界，计算相邻分区之间的间隔时间。
        有助于理解信号的周期性特征和分区的时间分布模式。
        
        参数说明：
            group_by (GroupByLike): 分组参数，默认为None
                - 指定分组方式，支持按列分组
            attach_ts (bool): 是否附加时间序列，默认为True
                - True: 在Ranges对象中包含时间序列信息
                - False: 不包含时间序列信息
            **kwargs: 传递给Ranges构造函数的关键字参数
        
        返回值：
            Ranges: 分区间隔分析结果对象，包含以下信息：
                - 间隔ID、列索引、开始时间、结束时间
                - 间隔持续时间、状态等详细信息
        
        间隔计算逻辑：
        - 识别每个分区的开始和结束位置
        - 计算相邻分区之间的时间间隔
        - 间隔从上一个分区的结束到下一个分区的开始
        - 第一个间隔从数据开始到第一个分区开始
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建包含多个分区的信号数据
        signals = pd.Series([True, False, False, True, False, True, True])
        ranges = signals.vbt.signals.between_partition_ranges()
        
        print("分区间隔记录:\n", ranges.records_readable)
        print("间隔持续时间:", ranges.duration.values)
        # 输出示例：
        # 分区间隔记录:
        #    Range Id  Column  Start Timestamp  End Timestamp  Status
        # 0         0       0                0              3  Closed
        # 1         1       0                3              5  Closed
        # 间隔持续时间: [3, 2]
        ```
        
        应用场景：
        - 信号周期性分析
        - 交易策略的间隔优化
        - 信号时间分布研究
        - 策略参数调优
        
        注意事项：
        - 间隔分析基于分区的边界
        - 第一个间隔可能从数据开始位置开始
        - 时间索引有助于更精确的间隔分析
        - 分组分析可以处理多列信号数据
        """
        range_records = nb.between_partition_ranges_nb(self.to_2d_array())  # 调用Numba函数分析分区间隔
        return Ranges(
            self.wrapper,                    # 数组包装器
            range_records,                   # 分区间隔记录
            ts=self.obj if attach_ts else None,  # 时间序列数据
            **kwargs                         # 其他参数
        ).regroup(group_by)                 # 应用分组并返回

    # ############# Ranking ############# #

    def rank(self,
             rank_func_nb: tp.RankFunc, *args,
             prepare_func: tp.Optional[tp.Callable] = None,
             reset_by: tp.Optional[tp.ArrayLike] = None,
             after_false: bool = False,
             broadcast_kwargs: tp.KwargsLike = None,
             wrap_kwargs: tp.KwargsLike = None,
             as_mapped: bool = False,
             **kwargs) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """
        基于自定义排序函数对信号进行排序
        
        功能说明：
        使用Numba编译的排序函数对信号数据进行排序分析。
        该方法提供了灵活的排序机制，支持自定义排序逻辑和重置条件。
        可以分析信号的位置排序、分区排序等多种排序特征，为策略分析提供重要信息。
        
        参数说明：
            rank_func_nb (callable): Numba编译的排序函数
                - 函数签名: def func(rank_arr, from_i, to_i, col, *args) -> None
                - 必须使用@njit装饰器编译
                - 函数应该修改rank_arr数组来设置排序值
            *args: 传递给排序函数的额外参数
            prepare_func (callable): 准备函数，默认为None
                - 用于准备传递给排序函数的临时数组等参数
                - 函数签名: def func(obj_arr, reset_by) -> tuple
                - 返回的元组会传递给排序函数
            reset_by (array_like): 重置条件数组，默认为None
                - 指定何时重置排序计数器
                - True值表示重置排序
                - None表示不重置
            after_false (bool): 是否在False值后重置，默认为False
                - True: 在每个False值后重置排序
                - False: 不重置排序
            broadcast_kwargs (dict): 广播参数，默认为None
                - 控制输入数组的广播行为
            wrap_kwargs (dict): 包装参数，默认为None
                - 传递给ArrayWrapper.wrap的关键字参数
            as_mapped (bool): 是否返回映射数组，默认为False
                - True: 返回MappedArray对象，-1值被替换为NaN
                - False: 返回普通的SeriesFrame对象
            **kwargs: 传递给MappedArray构造函数的关键字参数
        
        返回值：
            tp.Union[tp.SeriesFrame, MappedArray]: 排序结果
                - 数值表示排序位置，-1表示无排序
                - as_mapped=True时返回MappedArray，-1被替换为NaN
        
        排序函数要求：
        ```python
        from numba import njit
        import numpy as np
        
        @njit
        def my_rank_func(rank_arr, from_i, to_i, col, *args):
            # rank_arr: 排序数组，需要修改此数组
            # from_i: 排序起始索引
            # to_i: 排序结束索引
            # col: 当前列索引
            # *args: 额外参数
            
            # 示例：为每个True值分配递增的排序号
            rank = 0
            for i in range(from_i, to_i):
                if rank_arr[i, col]:  # 如果是True值
                    rank_arr[i, col] = rank
                    rank += 1
                else:
                    rank_arr[i, col] = -1  # False值设为-1
        ```
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        from numba import njit
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False],
            'c': [False, True, True, True, False]
        })
        
        # 定义排序函数：为每个True值分配位置排序
        @njit
        def position_rank_func(rank_arr, from_i, to_i, col):
            rank = 0
            for i in range(from_i, to_i):
                if rank_arr[i, col]:
                    rank_arr[i, col] = rank
                    rank += 1
                else:
                    rank_arr[i, col] = -1
        
        # 执行排序
        ranks = signals.vbt.signals.rank(
            position_rank_func,
            after_false=False  # 不在False后重置
        )
        
        print("原始信号:\n", signals)
        print("位置排序:\n", ranks)
        ```
        
        应用场景：
        - 信号位置分析
        - 分区排序研究
        - 策略信号优先级分析
        - 信号模式识别
        
        注意事项：
        - 排序函数必须是Numba编译的
        - 函数应该修改rank_arr数组
        - reset_by参数控制排序重置逻辑
        - 建议使用prepare_func准备临时数组
        """
        checks.assert_not_none(rank_func_nb)  # 验证排序函数不为空
        checks.assert_numba_func(rank_func_nb)  # 验证排序函数为Numba编译
        if broadcast_kwargs is None:
            broadcast_kwargs = {}

        if reset_by is not None:  # 如果提供了重置条件
            obj, reset_by = reshape_fns.broadcast(self.obj, reset_by, **broadcast_kwargs)  # 广播数组
            reset_by = reshape_fns.to_2d_array(reset_by)  # 转换为2D数组
        else:
            obj = self.obj  # 使用当前对象
        
        obj_arr = reshape_fns.to_2d_array(obj)  # 转换为2D数组
        
        if prepare_func is not None:  # 如果提供了准备函数
            temp_arrs = prepare_func(obj_arr, reset_by)  # 准备临时数组
        else:
            temp_arrs = ()  # 空元组
        
        # 调用Numba函数执行排序
        rank = nb.rank_nb(
            obj_arr,           # 信号数组
            reset_by,          # 重置条件
            after_false,       # 是否在False后重置
            rank_func_nb,      # 排序函数
            *temp_arrs,        # 临时数组
            *args              # 额外参数
        )
        
        # 包装排序结果
        rank_wrapped = ArrayWrapper.from_obj(obj).wrap(rank, group_by=False, **merge_dicts({}, wrap_kwargs))
        
        if as_mapped:  # 如果需要返回映射数组
            rank_wrapped = rank_wrapped.replace(-1, np.nan)  # 将-1替换为NaN
            return rank_wrapped.vbt.to_mapped(
                dropna=True,    # 删除NaN值
                dtype=np.int64, # 整数类型
                **kwargs        # 其他参数
            )
        return rank_wrapped  # 返回排序结果

    def pos_rank(self, allow_gaps: bool = False, **kwargs) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """
        获取信号位置排序
        
        功能说明：
        为信号数据中的每个True值分配位置排序号，分析信号在时间序列中的位置分布。
        该方法使用内置的sig_pos_rank_nb函数，为每个信号分区内的True值分配递增的排序号。
        有助于理解信号的时间分布特征和位置模式。
        
        参数说明：
            allow_gaps (bool): 是否允许排序间隔，默认为False
                - True: 允许排序号之间有间隔，保持连续性
                - False: 排序号连续，无间隔
            **kwargs: 传递给rank方法的关键字参数
                - 包括reset_by、after_false、as_mapped等参数
        
        返回值：
            tp.Union[tp.SeriesFrame, MappedArray]: 位置排序结果
                - 数值表示在分区内的位置排序(从0开始)
                - -1表示无信号位置
        
        排序逻辑：
        - 每个信号分区内的True值按时间顺序分配排序号
        - 排序号从0开始递增
        - False值的位置设为-1
        - 支持按条件重置排序计数器
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False],
            'c': [False, True, True, True, False]
        })
        
        # 基础位置排序
        ranks = signals.vbt.signals.pos_rank()
        print("位置排序:\n", ranks)
        
        # 允许间隔的位置排序
        ranks_gaps = signals.vbt.signals.pos_rank(allow_gaps=True)
        print("允许间隔的位置排序:\n", ranks_gaps)
        
        # 在False后重置的排序
        ranks_reset = signals.vbt.signals.pos_rank(after_false=True)
        print("重置排序:\n", ranks_reset)
        ```
        
        应用场景：
        - 信号时间分布分析
        - 交易策略的信号位置研究
        - 信号模式识别
        - 策略优化和参数调整
        
        注意事项：
        - 排序基于信号分区进行
        - allow_gaps参数影响排序的连续性
        - 重置条件可以改变排序逻辑
        - 返回映射数组时-1值被替换为NaN
        """
        prepare_func = lambda obj, reset_by: (np.full(obj.shape[1], -1, dtype=np.int64),)  # 准备临时数组
        return self.rank(
            nb.sig_pos_rank_nb,  # 使用内置的信号位置排序函数
            allow_gaps,          # 是否允许间隔
            prepare_func=prepare_func,  # 准备函数
            **kwargs             # 其他参数
        )

    def partition_pos_rank(self, **kwargs) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """
        获取分区位置排序
        
        功能说明：
        为信号分区分配排序号，分析分区的顺序和分布特征。
        该方法使用内置的part_pos_rank_nb函数，为每个信号分区分配唯一的排序号。
        有助于理解信号的聚集特征和分区的时间分布模式。
        
        参数说明：
            **kwargs: 传递给rank方法的关键字参数
                - 包括reset_by、after_false、as_mapped等参数
        
        返回值：
            tp.Union[tp.SeriesFrame, MappedArray]: 分区排序结果
                - 数值表示分区的排序号(从0开始)
                - -1表示无分区
        
        排序逻辑：
        - 连续的True值被识别为一个分区
        - 每个分区分配一个唯一的排序号
        - 分区内的所有True值共享相同的排序号
        - False值的位置设为-1
        - 支持按条件重置分区排序
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建包含分区的信号数据
        signals = pd.DataFrame({
            'a': [True, True, False, True, False],
            'b': [True, False, True, True, True],
            'c': [False, True, True, False, True]
        })
        
        # 分区位置排序
        ranks = signals.vbt.signals.partition_pos_rank()
        print("分区排序:\n", ranks)
        
        # 在False后重置的分区排序
        ranks_reset = signals.vbt.signals.partition_pos_rank(after_false=True)
        print("重置分区排序:\n", ranks_reset)
        
        # 返回映射数组
        ranks_mapped = signals.vbt.signals.partition_pos_rank(as_mapped=True)
        print("映射分区排序:\n", ranks_mapped)
        ```
        
        应用场景：
        - 信号聚集性分析
        - 交易策略的持仓模式研究
        - 分区时间分布分析
        - 策略优化和参数调整
        
        注意事项：
        - 分区基于连续的True值识别
        - 分区排序反映信号的聚集特征
        - 重置条件可以改变分区排序逻辑
        - 返回映射数组时-1值被替换为NaN
        """
        prepare_func = lambda obj, reset_by: (np.full(obj.shape[1], -1, dtype=np.int64),)  # 准备临时数组
        return self.rank(
            nb.part_pos_rank_nb,  # 使用内置的分区位置排序函数
            prepare_func=prepare_func,  # 准备函数
            **kwargs              # 其他参数
        )

    def first(self, wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.SeriesFrame:
        """
        选择第一个信号
        
        功能说明：
        基于位置排序选择每个分区中的第一个信号。
        该方法使用pos_rank方法获取位置排序，然后选择排序号为0的信号。
        常用于信号过滤，只保留每个分区的第一个信号。
        
        参数说明：
            wrap_kwargs (dict): 包装参数，默认为None
                - 传递给ArrayWrapper.wrap的关键字参数
            **kwargs: 传递给pos_rank方法的关键字参数
                - 包括allow_gaps、reset_by、after_false等参数
        
        返回值：
            tp.SeriesFrame: 过滤后的信号数据
                - 只包含每个分区的第一个信号
                - 其他位置设为False
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, True, False, True, False],
            'b': [True, False, True, True, True],
            'c': [False, True, True, True, False]
        })
        
        # 选择第一个信号
        first_signals = signals.vbt.signals.first()
        print("原始信号:\n", signals)
        print("第一个信号:\n", first_signals)
        ```
        
        应用场景：
        - 信号去重和过滤
        - 策略信号优化
        - 减少交易频率
        - 信号质量提升
        """
        pos_rank = self.pos_rank(**kwargs).values  # 获取位置排序
        return self.wrapper.wrap(pos_rank == 0, group_by=False, **merge_dicts({}, wrap_kwargs))  # 选择排序号为0的信号

    def nth(self, n: int, wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.SeriesFrame:
        """
        选择第n个信号
        
        功能说明：
        基于位置排序选择每个分区中的第n个信号。
        该方法使用pos_rank方法获取位置排序，然后选择排序号为n的信号。
        支持选择任意位置的信号，常用于信号分析和策略研究。
        
        参数说明：
            n (int): 要选择的信号位置
                - 0: 第一个信号
                - 1: 第二个信号
                - -1: 最后一个信号
            wrap_kwargs (dict): 包装参数，默认为None
                - 传递给ArrayWrapper.wrap的关键字参数
            **kwargs: 传递给pos_rank方法的关键字参数
        
        返回值：
            tp.SeriesFrame: 过滤后的信号数据
                - 只包含每个分区的第n个信号
                - 其他位置设为False
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, True, False, True, False],
            'b': [True, False, True, True, True],
            'c': [False, True, True, False, True]
        })
        
        # 选择第二个信号
        second_signals = signals.vbt.signals.nth(1)
        print("第二个信号:\n", second_signals)
        
        # 选择最后一个信号
        last_signals = signals.vbt.signals.nth(-1)
        print("最后一个信号:\n", last_signals)
        ```
        
        应用场景：
        - 信号位置分析
        - 策略信号选择
        - 信号模式研究
        - 策略优化和测试
        """
        pos_rank = self.pos_rank(**kwargs).values  # 获取位置排序
        return self.wrapper.wrap(pos_rank == n, group_by=False, **merge_dicts({}, wrap_kwargs))  # 选择排序号为n的信号

    def from_nth(self, n: int, wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.SeriesFrame:
        """
        选择从第n个开始的信号
        
        功能说明：
        基于位置排序选择每个分区中从第n个开始的所有信号。
        该方法使用pos_rank方法获取位置排序，然后选择排序号大于等于n的信号。
        常用于信号过滤，保留分区中特定位置之后的所有信号。
        
        参数说明：
            n (int): 起始信号位置
                - 0: 从第一个信号开始(包含所有信号)
                - 1: 从第二个信号开始
                - 2: 从第三个信号开始
            wrap_kwargs (dict): 包装参数，默认为None
                - 传递给ArrayWrapper.wrap的关键字参数
            **kwargs: 传递给pos_rank方法的关键字参数
        
        返回值：
            tp.SeriesFrame: 过滤后的信号数据
                - 包含每个分区中从第n个开始的所有信号
                - 其他位置设为False
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, True, False, True, False],
            'b': [True, False, True, True, True],
            'c': [False, True, True, False, True]
        })
        
        # 从第二个信号开始选择
        from_second = signals.vbt.signals.from_nth(1)
        print("从第二个信号开始:\n", from_second)
        
        # 从第三个信号开始选择
        from_third = signals.vbt.signals.from_nth(2)
        print("从第三个信号开始:\n", from_third)
        ```
        
        应用场景：
        - 信号过滤和优化
        - 策略信号选择
        - 减少早期信号干扰
        - 策略参数调优
        """
        pos_rank = self.pos_rank(**kwargs).values  # 获取位置排序
        return self.wrapper.wrap(pos_rank >= n, group_by=False, **merge_dicts({}, wrap_kwargs))  # 选择排序号大于等于n的信号

    def pos_rank_mapped(self, group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """
        获取映射格式的位置排序
        
        功能说明：
        返回位置排序的映射数组格式，-1值被替换为NaN。
        该方法调用pos_rank方法并设置as_mapped=True，返回MappedArray对象。
        便于后续的统计分析和可视化处理。
        
        参数说明：
            group_by (GroupByLike): 分组参数，默认为None
                - 指定分组方式，支持按列分组
            **kwargs: 传递给pos_rank方法的关键字参数
        
        返回值：
            MappedArray: 映射格式的位置排序结果
                - -1值被替换为NaN
                - 便于统计分析和可视化
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False],
            'c': [False, True, True, True, False]
        })
        
        # 获取映射格式的位置排序
        mapped_ranks = signals.vbt.signals.pos_rank_mapped()
        print("映射格式位置排序:\n", mapped_ranks)
        
        # 统计分析
        print("排序统计:", mapped_ranks.describe())
        ```
        
        应用场景：
        - 信号排序统计分析
        - 数据可视化
        - 信号模式研究
        - 策略性能分析
        """
        return self.pos_rank(as_mapped=True, group_by=group_by, **kwargs)  # 返回映射格式的位置排序

    def partition_pos_rank_mapped(self, group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """
        获取映射格式的分区排序
        
        功能说明：
        返回分区排序的映射数组格式，-1值被替换为NaN。
        该方法调用partition_pos_rank方法并设置as_mapped=True，返回MappedArray对象。
        便于后续的统计分析和可视化处理。
        
        参数说明：
            group_by (GroupByLike): 分组参数，默认为None
                - 指定分组方式，支持按列分组
            **kwargs: 传递给partition_pos_rank方法的关键字参数
        
        返回值：
            MappedArray: 映射格式的分区排序结果
                - -1值被替换为NaN
                - 便于统计分析和可视化
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, True, False, True, False],
            'b': [True, False, True, True, True],
            'c': [False, True, True, False, True]
        })
        
        # 获取映射格式的分区排序
        mapped_part_ranks = signals.vbt.signals.partition_pos_rank_mapped()
        print("映射格式分区排序:\n", mapped_part_ranks)
        
        # 统计分析
        print("分区排序统计:", mapped_part_ranks.describe())
        ```
        
        应用场景：
        - 分区排序统计分析
        - 数据可视化
        - 信号聚集性研究
        - 策略性能分析
        """
        return self.partition_pos_rank(as_mapped=True, group_by=group_by, **kwargs)  # 返回映射格式的分区排序

    # ############# Index ############# #

    def nth_index(self, n: int, return_labels: bool = True, group_by: tp.GroupByLike = None,
                  wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        获取第n个信号的索引位置
        
        功能说明：
        获取信号数据中第n个True值的索引位置。
        该方法基于位置排序找到第n个信号的具体时间位置，返回对应的索引标签。
        支持正数和负数索引，便于分析信号的时间分布特征。
        
        参数说明：
            n (int): 要查找的信号位置
                - 0: 第一个信号
                - 1: 第二个信号
                - -1: 最后一个信号
                - -2: 倒数第二个信号
            return_labels (bool): 是否返回标签，默认为True
                - True: 返回时间索引标签(如时间戳)
                - False: 返回数值索引位置
            group_by (GroupByLike): 分组参数，默认为None
                - 指定分组方式，支持按列分组
            wrap_kwargs (dict): 包装参数，默认为None
                - 传递给ArrayWrapper.wrap的关键字参数
        
        返回值：
            tp.MaybeSeries: 第n个信号的索引位置
                - return_labels=True: 返回时间索引标签
                - return_labels=False: 返回数值索引位置
                - 如果不存在第n个信号，返回NaN或NaT
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建带时间索引的信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False],
            'c': [False, True, True, True, False]
        }, index=pd.date_range('2020-01-01', periods=5))
        
        # 获取第一个信号的索引
        first_idx = signals.vbt.signals.nth_index(0)
        print("第一个信号索引:", first_idx)
        
        # 获取第二个信号的索引
        second_idx = signals.vbt.signals.nth_index(1)
        print("第二个信号索引:", second_idx)
        
        # 获取最后一个信号的索引
        last_idx = signals.vbt.signals.nth_index(-1)
        print("最后一个信号索引:", last_idx)
        
        # 获取数值索引位置
        numeric_idx = signals.vbt.signals.nth_index(0, return_labels=False)
        print("数值索引位置:", numeric_idx)
        ```
        
        应用场景：
        - 信号时间分布分析
        - 交易策略的时间特征研究
        - 信号间隔计算
        - 策略优化和参数调整
        
        注意事项：
        - 索引从0开始计数
        - 负数索引从-1开始(最后一个)
        - 如果不存在第n个信号，返回NaN
        - 分组分析会返回聚合结果
        """
        if self.is_frame() and self.wrapper.grouper.is_grouped(group_by=group_by):  # 如果是分组数据
            squeezed = self.squeeze_grouped(generic_nb.any_squeeze_nb, group_by=group_by)  # 压缩分组数据
            arr = reshape_fns.to_2d_array(squeezed)  # 转换为2D数组
        else:
            arr = self.to_2d_array()  # 直接转换为2D数组
        
        nth_index = nb.nth_index_nb(arr, n)  # 调用Numba函数获取第n个索引
        
        if return_labels:  # 如果需要返回标签
            minus_one_mask = nth_index == -1  # 标记无效索引
            nth_index = nth_index.astype(object)  # 转换为对象类型
            nth_index[minus_one_mask] = np.nan  # 无效索引设为NaN
            nth_index[~minus_one_mask] = self.wrapper.index[nth_index[~minus_one_mask].astype(np.int64)]  # 获取对应的索引标签
        
        wrap_kwargs = merge_dicts(dict(name_or_index='nth_index'), wrap_kwargs)  # 合并包装参数
        return self.wrapper.wrap_reduced(nth_index, group_by=group_by, **wrap_kwargs)  # 包装并返回结果

    def norm_avg_index(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算归一化平均索引位置
        
        功能说明：
        计算信号的平均位置相对于数据中间位置的归一化值。
        该方法测量信号分布的中心趋势，帮助快速识别信号的时间分布特征。
        归一化值范围在[-1, 1]之间，便于比较不同信号的分布特征。
        
        参数说明：
            group_by (GroupByLike): 分组参数，默认为None
                - 指定分组方式，支持按列分组
            wrap_kwargs (dict): 包装参数，默认为None
                - 传递给ArrayWrapper.wrap的关键字参数
        
        返回值：
            tp.MaybeSeries: 归一化平均索引位置
                - 范围: [-1.0, 1.0]
                - -1.0: 信号集中在开始位置
                - 0.0: 信号分布对称
                - 1.0: 信号集中在结束位置
        
        归一化逻辑：
        - 计算所有信号位置的平均值
        - 相对于数据中间位置进行归一化
        - 结果范围在[-1, 1]之间
        - 负值表示信号偏左，正值表示信号偏右
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建不同分布的信号数据
        signals_left = pd.Series([True, False, False, False, False])  # 信号在左侧
        signals_right = pd.Series([False, False, False, False, True])  # 信号在右侧
        signals_center = pd.Series([False, False, True, False, False])  # 信号在中间
        signals_symmetric = pd.Series([True, False, False, False, True])  # 信号对称分布
        
        # 计算归一化平均索引
        left_avg = signals_left.vbt.signals.norm_avg_index()
        right_avg = signals_right.vbt.signals.norm_avg_index()
        center_avg = signals_center.vbt.signals.norm_avg_index()
        symmetric_avg = signals_symmetric.vbt.signals.norm_avg_index()
        
        print("左侧信号平均索引:", left_avg)      # 接近 -1.0
        print("右侧信号平均索引:", right_avg)     # 接近 1.0
        print("中间信号平均索引:", center_avg)    # 接近 0.0
        print("对称信号平均索引:", symmetric_avg)  # 接近 0.0
        ```
        
        应用场景：
        - 信号时间分布特征分析
        - 交易策略的时间偏好研究
        - 信号模式识别
        - 策略优化和参数调整
        
        注意事项：
        - 归一化值反映信号分布的中心趋势
        - 分组分析会计算加权平均
        - 单个信号时返回极值(-1或1)
        - 对称分布时接近0
        """
        norm_avg_index = nb.norm_avg_index_nb(self.to_2d_array())  # 调用Numba函数计算归一化平均索引
        wrap_kwargs = merge_dicts(dict(name_or_index='norm_avg_index'), wrap_kwargs)  # 合并包装参数
        norm_avg_index = self.wrapper.wrap_reduced(norm_avg_index, group_by=False, **wrap_kwargs)  # 包装结果
        
        if self.is_frame() and self.wrapper.grouper.is_grouped(group_by=group_by):  # 如果是分组数据
            # 分组索引是组内列索引的加权平均
            if group_by is None:
                group_by = self.wrapper.grouper.group_by  # 使用默认分组
            
            col_total = self.total(group_by=False)  # 获取每列的信号总数
            norm_avg_index *= col_total  # 乘以信号总数
            norm_avg_index = norm_avg_index.vbt.squeeze_grouped(  # 压缩分组数据
                generic_nb.sum_squeeze_nb, group_by=group_by)
            group_total = col_total.vbt.squeeze_grouped(  # 获取分组总数
                generic_nb.sum_squeeze_nb, group_by=group_by)
            norm_avg_index /= group_total  # 除以分组总数得到加权平均
        
        return norm_avg_index  # 返回归一化平均索引

    def index_mapped(self, group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """
        获取映射格式的索引数组
        
        功能说明：
        返回信号位置的映射数组，只包含True值的位置信息。
        该方法创建一个与信号数据相同形状的数组，True值位置保留索引值，False值位置设为NaN。
        便于后续的统计分析和可视化处理。
        
        参数说明：
            group_by (GroupByLike): 分组参数，默认为None
                - 指定分组方式，支持按列分组
            **kwargs: 传递给to_mapped方法的关键字参数
        
        返回值：
            MappedArray: 映射格式的索引数组
                - 只包含True值的位置信息
                - False值位置为NaN
                - 便于统计分析和可视化
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False],
            'c': [False, True, True, True, False]
        })
        
        # 获取映射格式的索引
        mapped_indices = signals.vbt.signals.index_mapped()
        print("映射索引:\n", mapped_indices)
        
        # 统计分析
        print("索引统计:", mapped_indices.describe())
        ```
        
        应用场景：
        - 信号位置统计分析
        - 数据可视化
        - 信号模式研究
        - 策略性能分析
        """
        indices = np.arange(len(self.wrapper.index), dtype=np.float64)[:, None]  # 创建索引数组
        indices = np.tile(indices, (1, len(self.wrapper.columns)))  # 平铺到所有列
        indices = reshape_fns.soft_to_ndim(indices, self.wrapper.ndim)  # 调整维度
        indices[~self.obj.values] = np.nan  # False值位置设为NaN
        return self.wrapper.wrap(indices).vbt.to_mapped(  # 转换为映射数组
            dropna=True,    # 删除NaN值
            dtype=np.int64, # 整数类型
            group_by=group_by,  # 分组参数
            **kwargs        # 其他参数
        )

    def total(self, wrap_kwargs: tp.KwargsLike = None,
              group_by: tp.GroupByLike = None) -> tp.MaybeSeries:
        """
        计算信号总数
        
        功能说明：
        计算每列或每组中True值的总数。
        该方法统计信号数据的密度，是信号分析的基础指标。
        支持分组统计，便于多维度分析。
        
        参数说明：
            wrap_kwargs (dict): 包装参数，默认为None
                - 传递给ArrayWrapper.wrap的关键字参数
            group_by (GroupByLike): 分组参数，默认为None
                - 指定分组方式，支持按列分组
        
        返回值：
            tp.MaybeSeries: 信号总数
                - 每列或每组的True值数量
                - 整数类型
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False],
            'c': [False, True, True, True, False]
        })
        
        # 计算信号总数
        total_signals = signals.vbt.signals.total()
        print("信号总数:", total_signals)
        
        # 分组统计
        grouped_total = signals.vbt.signals.total(group_by=True)
        print("分组总数:", grouped_total)
        ```
        
        应用场景：
        - 信号密度分析
        - 策略信号频率统计
        - 信号质量评估
        - 策略性能分析
        """
        wrap_kwargs = merge_dicts(dict(name_or_index='total'), wrap_kwargs)  # 合并包装参数
        return self.sum(group_by=group_by, wrap_kwargs=wrap_kwargs)  # 调用sum方法计算总数

    def rate(self, wrap_kwargs: tp.KwargsLike = None,
             group_by: tp.GroupByLike = None, **kwargs) -> tp.MaybeSeries:
        """
        计算信号比率
        
        功能说明：
        计算信号密度，即True值数量占总数据长度的比例。
        该方法提供标准化的信号密度指标，便于比较不同长度的数据。
        比率范围在[0, 1]之间，0表示无信号，1表示全为信号。
        
        参数说明：
            wrap_kwargs (dict): 包装参数，默认为None
                - 传递给ArrayWrapper.wrap的关键字参数
            group_by (GroupByLike): 分组参数，默认为None
                - 指定分组方式，支持按列分组
            **kwargs: 传递给total方法的关键字参数
        
        返回值：
            tp.MaybeSeries: 信号比率
                - 范围: [0.0, 1.0]
                - 0.0: 无信号
                - 1.0: 全为信号
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False, True],  # 3/5 = 0.6
            'b': [True, True, False, True, False],  # 3/5 = 0.6
            'c': [False, True, True, True, False]   # 3/5 = 0.6
        })
        
        # 计算信号比率
        signal_rate = signals.vbt.signals.rate()
        print("信号比率:", signal_rate)
        
        # 分组统计
        grouped_rate = signals.vbt.signals.rate(group_by=True)
        print("分组比率:", grouped_rate)
        ```
        
        应用场景：
        - 信号密度标准化分析
        - 策略信号频率比较
        - 信号质量评估
        - 策略优化和参数调整
        """
        total = reshape_fns.to_1d_array(self.total(group_by=group_by, **kwargs))  # 获取信号总数
        wrap_kwargs = merge_dicts(dict(name_or_index='rate'), wrap_kwargs)  # 合并包装参数
        total_steps = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]  # 计算总步数
        return self.wrapper.wrap_reduced(total / total_steps, group_by=group_by, **wrap_kwargs)  # 计算比率并返回

    def total_partitions(self, wrap_kwargs: tp.KwargsLike = None,
                         group_by: tp.GroupByLike = None, **kwargs) -> tp.MaybeSeries:
        """
        计算分区总数
        
        功能说明：
        计算信号数据中分区的总数。
        该方法统计连续True值形成的分区数量，反映信号的聚集特征。
        分区总数是信号模式分析的重要指标。
        
        参数说明：
            wrap_kwargs (dict): 包装参数，默认为None
                - 传递给ArrayWrapper.wrap的关键字参数
            group_by (GroupByLike): 分组参数，默认为None
                - 指定分组方式，支持按列分组
            **kwargs: 传递给partition_ranges方法的关键字参数
        
        返回值：
            tp.MaybeSeries: 分区总数
                - 每列或每组的分区数量
                - 整数类型
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, True, False, True, False],  # 2个分区
            'b': [True, False, True, True, True],   # 2个分区
            'c': [False, True, True, False, True]   # 2个分区
        })
        
        # 计算分区总数
        total_parts = signals.vbt.signals.total_partitions()
        print("分区总数:", total_parts)
        
        # 分组统计
        grouped_parts = signals.vbt.signals.total_partitions(group_by=True)
        print("分组分区数:", grouped_parts)
        ```
        
        应用场景：
        - 信号聚集性分析
        - 交易策略的持仓模式研究
        - 信号模式识别
        - 策略优化和参数调整
        """
        wrap_kwargs = merge_dicts(dict(name_or_index='total_partitions'), wrap_kwargs)  # 合并包装参数
        return self.partition_ranges(**kwargs).count(group_by=group_by, wrap_kwargs=wrap_kwargs)  # 计算分区数量

    def partition_rate(self, wrap_kwargs: tp.KwargsLike = None,
                       group_by: tp.GroupByLike = None, **kwargs) -> tp.MaybeSeries:
        """
        计算分区比率
        
        功能说明：
        计算分区密度，即分区数量与信号总数的比例。
        该方法反映信号的聚集程度，高比率表示信号分散，低比率表示信号聚集。
        是信号模式分析的重要指标。
        
        参数说明：
            wrap_kwargs (dict): 包装参数，默认为None
                - 传递给ArrayWrapper.wrap的关键字参数
            group_by (GroupByLike): 分组参数，默认为None
                - 指定分组方式，支持按列分组
            **kwargs: 传递给total_partitions和total方法的关键字参数
        
        返回值：
            tp.MaybeSeries: 分区比率
                - 范围: [0.0, 1.0]
                - 高值: 信号分散
                - 低值: 信号聚集
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建不同聚集程度的信号数据
        signals_scattered = pd.DataFrame({
            'a': [True, False, True, False, True],  # 3个分区，3个信号，比率=1.0
            'b': [True, True, False, True, True],   # 2个分区，4个信号，比率=0.5
            'c': [True, True, True, False, True]    # 2个分区，4个信号，比率=0.5
        })
        
        # 计算分区比率
        part_rate = signals_scattered.vbt.signals.partition_rate()
        print("分区比率:", part_rate)
        
        # 分组统计
        grouped_rate = signals_scattered.vbt.signals.partition_rate(group_by=True)
        print("分组分区比率:", grouped_rate)
        ```
        
        应用场景：
        - 信号聚集性分析
        - 交易策略的持仓模式研究
        - 信号质量评估
        - 策略优化和参数调整
        
        注意事项：
        - 分区比率反映信号的聚集程度
        - 高比率表示信号分散，适合短线策略
        - 低比率表示信号聚集，适合长线策略
        - 需要结合具体策略需求分析
        """
        total_partitions = reshape_fns.to_1d_array(self.total_partitions(group_by=group_by, *kwargs))  # 获取分区总数
        total = reshape_fns.to_1d_array(self.total(group_by=group_by, *kwargs))  # 获取信号总数
        wrap_kwargs = merge_dicts(dict(name_or_index='partition_rate'), wrap_kwargs)  # 合并包装参数
        return self.wrapper.wrap_reduced(total_partitions / total, group_by=group_by, **wrap_kwargs)  # 计算比率并返回

    # ############# Logical operations ############# #

    def AND(self, other: tp.ArrayLike, **kwargs) -> tp.SeriesFrame:
        """
        与另一个信号进行逻辑AND运算
        
        功能说明：
        将当前信号与另一个信号进行逻辑AND运算，返回两个信号的交集。
        该方法实现信号的条件组合，只有当两个信号都为True时，结果才为True。
        常用于多条件信号的组合和过滤。
        
        参数说明：
            other (array_like): 另一个信号数组
                - 可以是Series、DataFrame或numpy数组
                - 必须与当前信号具有兼容的形状
            **kwargs: 传递给combine方法的关键字参数
                - 包括broadcast_kwargs、concat等参数
        
        返回值：
            tp.SeriesFrame: 逻辑AND运算结果
                - True: 两个信号都为True的位置
                - False: 其他位置
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建两个信号
        signal1 = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False]
        })
        
        signal2 = pd.DataFrame({
            'a': [True, True, False, True, False],
            'b': [False, True, True, False, True]
        })
        
        # 逻辑AND运算
        and_result = signal1.vbt.signals.AND(signal2)
        print("AND结果:\n", and_result)
        
        # 与条件组合
        condition = pd.Series([True, False, True, True, False])
        filtered = signal1.vbt.signals.AND(condition)
        print("条件过滤结果:\n", filtered)
        ```
        
        应用场景：
        - 多条件信号组合
        - 信号过滤和筛选
        - 策略信号优化
        - 风险管理信号生成
        
        注意事项：
        - 输入信号必须是布尔型
        - 支持广播和形状兼容
        - 结果保持原始数据的索引和列名
        """
        return self.combine(other, combine_func=np.logical_and, **kwargs)  # 使用逻辑AND函数组合信号

    def OR(self, other: tp.ArrayLike, **kwargs) -> tp.SeriesFrame:
        """
        与另一个信号进行逻辑OR运算
        
        功能说明：
        将当前信号与另一个信号进行逻辑OR运算，返回两个信号的并集。
        该方法实现信号的合并，当任一信号为True时，结果就为True。
        常用于多条件信号的合并和扩展。
        
        参数说明：
            other (array_like): 另一个信号数组
                - 可以是Series、DataFrame或numpy数组
                - 必须与当前信号具有兼容的形状
            **kwargs: 传递给combine方法的关键字参数
                - 包括broadcast_kwargs、concat等参数
        
        返回值：
            tp.SeriesFrame: 逻辑OR运算结果
                - True: 任一信号为True的位置
                - False: 两个信号都为False的位置
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建两个信号
        signal1 = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False]
        })
        
        signal2 = pd.DataFrame({
            'a': [True, True, False, True, False],
            'b': [False, True, True, False, True]
        })
        
        # 逻辑OR运算
        or_result = signal1.vbt.signals.OR(signal2)
        print("OR结果:\n", or_result)
        
        # 多条件合并
        conditions = [signal1 > 0.5, signal2 < -0.5]
        merged = signal1.vbt.signals.OR(conditions, concat=True, keys=['>0.5', '<-0.5'])
        print("多条件合并:\n", merged)
        ```
        
        应用场景：
        - 多条件信号合并
        - 信号扩展和增强
        - 策略信号组合
        - 风险信号汇总
        
        注意事项：
        - 输入信号必须是布尔型
        - 支持广播和形状兼容
        - 可以使用concat参数合并多个条件
        - 结果保持原始数据的索引和列名
        """
        return self.combine(other, combine_func=np.logical_or, **kwargs)  # 使用逻辑OR函数组合信号

    def XOR(self, other: tp.ArrayLike, **kwargs) -> tp.SeriesFrame:
        """
        与另一个信号进行逻辑XOR运算
        
        功能说明：
        将当前信号与另一个信号进行逻辑XOR运算，返回两个信号的异或结果。
        该方法实现信号的差异检测，只有当两个信号不同时，结果才为True。
        常用于信号变化检测和异常识别。
        
        参数说明：
            other (array_like): 另一个信号数组
                - 可以是Series、DataFrame或numpy数组
                - 必须与当前信号具有兼容的形状
            **kwargs: 传递给combine方法的关键字参数
                - 包括broadcast_kwargs、concat等参数
        
        返回值：
            tp.SeriesFrame: 逻辑XOR运算结果
                - True: 两个信号不同的位置
                - False: 两个信号相同的位置
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建两个信号
        signal1 = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False]
        })
        
        signal2 = pd.DataFrame({
            'a': [True, True, False, True, False],
            'b': [False, True, True, False, True]
        })
        
        # 逻辑XOR运算
        xor_result = signal1.vbt.signals.XOR(signal2)
        print("XOR结果:\n", xor_result)
        
        # 信号变化检测
        lagged_signal = signal1.shift(1)
        changes = signal1.vbt.signals.XOR(lagged_signal)
        print("信号变化:\n", changes)
        ```
        
        应用场景：
        - 信号变化检测
        - 异常信号识别
        - 策略信号差异分析
        - 信号质量评估
        
        注意事项：
        - 输入信号必须是布尔型
        - 支持广播和形状兼容
        - 常用于检测信号状态变化
        - 结果保持原始数据的索引和列名
        """
        return self.combine(other, combine_func=np.logical_xor, **kwargs)  # 使用逻辑XOR函数组合信号

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """
        统计分析的默认参数
        
        功能说明：
        获取信号统计分析的默认配置参数。
        该方法合并了GenericAccessor的默认参数和vectorbt设置中的信号统计配置。
        为统计分析提供统一的参数配置。
        
        返回值：
            tp.Kwargs: 统计分析的默认参数字典
                - 包含各种统计指标的配置
                - 合并了基础配置和信号专用配置
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False]
        })
        
        # 获取默认统计参数
        defaults = signals.vbt.signals.stats_defaults
        print("默认统计参数:", defaults)
        
        # 使用默认参数进行统计分析
        stats = signals.vbt.signals.stats()
        print("统计分析结果:", stats)
        ```
        
        应用场景：
        - 统计分析参数配置
        - 自定义统计指标设置
        - 批量统计分析
        - 策略性能评估
        """
        from vectorbt._settings import settings
        signals_stats_cfg = settings['signals']['stats']  # 获取信号统计配置

        return merge_dicts(
            GenericAccessor.stats_defaults.__get__(self),  # 基础默认参数
            signals_stats_cfg  # 信号专用配置
        )

    _metrics: tp.ClassVar[Config] = Config(
        dict(
            # 基础时间指标
            start=dict(
                title='Start',  # 标题：开始时间
                calc_func=lambda self: self.wrapper.index[0],  # 计算函数：获取第一个索引（开始时间）
                agg_func=None,  # 聚合函数：无（直接使用原始值）
                tags='wrapper'  # 标签：包装器相关
            ),
            end=dict(
                title='End',  # 标题：结束时间
                calc_func=lambda self: self.wrapper.index[-1],  # 计算函数：获取最后一个索引（结束时间）
                agg_func=None,  # 聚合函数：无（直接使用原始值）
                tags='wrapper'  # 标签：包装器相关
            ),
            period=dict(
                title='Period',  # 标题：周期长度
                calc_func=lambda self: len(self.wrapper.index),  # 计算函数：获取索引长度（总周期数）
                apply_to_timedelta=True,  # 应用到时间差：是（支持时间差计算）
                agg_func=None,  # 聚合函数：无（直接使用原始值）
                tags='wrapper'  # 标签：包装器相关
            ),
            
            # 基础信号统计指标
            total=dict(
                title='Total',  # 标题：信号总数
                calc_func='total',  # 计算函数：调用total方法计算信号总数
                tags='signals'  # 标签：信号相关
            ),
            rate=dict(
                title='Rate [%]',  # 标题：信号比率（百分比）
                calc_func='rate',  # 计算函数：调用rate方法计算信号比率
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理函数：转换为百分比
                tags='signals'  # 标签：信号相关
            ),
            
            # 信号重叠分析指标
            total_overlapping=dict(
                title='Total Overlapping',  # 标题：重叠信号总数
                calc_func=lambda self, other, group_by:
                (self & other).vbt.signals.total(group_by=group_by),  # 计算函数：两个信号的交集总数
                check_silent_has_other=True,  # 静默检查其他信号：是（自动检查是否有其他信号）
                tags=['signals', 'other']  # 标签：信号和其他信号相关
            ),
            overlapping_rate=dict(
                title='Overlapping Rate [%]',  # 标题：重叠信号比率（百分比）
                calc_func=lambda self, other, group_by:
                (self & other).vbt.signals.total(group_by=group_by) /
                (self | other).vbt.signals.total(group_by=group_by),  # 计算函数：交集总数除以并集总数
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理函数：转换为百分比
                check_silent_has_other=True,  # 静默检查其他信号：是
                tags=['signals', 'other']  # 标签：信号和其他信号相关
            ),
            
            # 信号位置分析指标
            first_index=dict(
                title='First Index',  # 标题：第一个信号索引
                calc_func='nth_index',  # 计算函数：调用nth_index方法
                n=0,  # 参数：获取第0个信号（第一个）
                return_labels=True,  # 返回标签：是（返回时间索引标签）
                tags=['signals', 'index']  # 标签：信号和索引相关
            ),
            last_index=dict(
                title='Last Index',  # 标题：最后一个信号索引
                calc_func='nth_index',  # 计算函数：调用nth_index方法
                n=-1,  # 参数：获取第-1个信号（最后一个）
                return_labels=True,  # 返回标签：是（返回时间索引标签）
                tags=['signals', 'index']  # 标签：信号和索引相关
            ),
            norm_avg_index=dict(
                title='Norm Avg Index [-1, 1]',  # 标题：归一化平均索引（范围-1到1）
                calc_func='norm_avg_index',  # 计算函数：调用norm_avg_index方法
                tags=['signals', 'index']  # 标签：信号和索引相关
            ),
            
            # 信号距离分析指标
            distance=dict(
                title=RepEval("f'Distance {\"<-\" if from_other else \"->\"} {other_name}' "
                              "if other is not None else 'Distance'"),  # 标题：动态距离标题（根据方向和其他信号名称）
                calc_func='between_ranges.duration',  # 计算函数：调用between_ranges.duration方法
                post_calc_func=lambda self, out, settings: {
                    'Min': out.min(),  # 最小值
                    'Max': out.max(),  # 最大值
                    'Mean': out.mean(),  # 平均值
                    'Std': out.std(ddof=settings.get('ddof', 1))  # 标准差
                },  # 后处理函数：计算统计摘要
                apply_to_timedelta=True,  # 应用到时间差：是（支持时间差计算）
                tags=RepEval("['signals', 'distance', 'other'] if other is not None else ['signals', 'distance']")  # 动态标签
            ),
            
            # 信号分区分析指标
            total_partitions=dict(
                title='Total Partitions',  # 标题：分区总数
                calc_func='total_partitions',  # 计算函数：调用total_partitions方法
                tags=['signals', 'partitions']  # 标签：信号和分区相关
            ),
            partition_rate=dict(
                title='Partition Rate [%]',  # 标题：分区比率（百分比）
                calc_func='partition_rate',  # 计算函数：调用partition_rate方法
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理函数：转换为百分比
                tags=['signals', 'partitions']  # 标签：信号和分区相关
            ),
            partition_len=dict(
                title='Partition Length',  # 标题：分区长度
                calc_func='partition_ranges.duration',  # 计算函数：调用partition_ranges.duration方法
                post_calc_func=lambda self, out, settings: {
                    'Min': out.min(),  # 最小值
                    'Max': out.max(),  # 最大值
                    'Mean': out.mean(),  # 平均值
                    'Std': out.std(ddof=settings.get('ddof', 1))  # 标准差
                },  # 后处理函数：计算统计摘要
                apply_to_timedelta=True,  # 应用到时间差：是（支持时间差计算）
                tags=['signals', 'partitions', 'distance']  # 标签：信号、分区和距离相关
            ),
            partition_distance=dict(
                title='Partition Distance',  # 标题：分区间隔
                calc_func='between_partition_ranges.duration',  # 计算函数：调用between_partition_ranges.duration方法
                post_calc_func=lambda self, out, settings: {
                    'Min': out.min(),  # 最小值
                    'Max': out.max(),  # 最大值
                    'Mean': out.mean(),  # 平均值
                    'Std': out.std(ddof=settings.get('ddof', 1))  # 标准差
                },  # 后处理函数：计算统计摘要
                apply_to_timedelta=True,  # 应用到时间差：是（支持时间差计算）
                tags=['signals', 'partitions', 'distance']  # 标签：信号、分区和距离相关
            ),
        ),
        copy_kwargs=dict(copy_mode='deep')  # 复制参数：深度复制模式
    )

    @property
    def metrics(self) -> Config:
        """
        信号分析指标配置
        
        功能说明：
        获取信号分析的所有可用指标配置。
        该属性提供了完整的信号分析指标体系，包括基础统计、位置分析、距离分析、分区分析等。
        每个指标都有详细的配置信息，便于自动化的统计分析。
        
        返回值：
            Config: 信号分析指标配置对象
                - 包含所有预定义的信号分析指标
                - 每个指标都有标题、计算函数、标签等信息
        
        指标分类：
        【基础统计指标】：
        - total: 信号总数
        - rate: 信号比率
        
        【位置分析指标】：
        - first_index: 第一个信号索引
        - last_index: 最后一个信号索引
        - norm_avg_index: 归一化平均索引
        
        【距离分析指标】：
        - avg_distance: 平均距离
        - other_distance: 到其他信号的距离
        
        【分区分析指标】：
        - total_partitions: 分区总数
        - partition_rate: 分区比率
        - partition_len: 分区长度
        - partition_distance: 分区间隔
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False]
        })
        
        # 获取指标配置
        metrics_config = signals.vbt.signals.metrics
        print("可用指标:", list(metrics_config.keys()))
        
        # 使用指标进行统计分析
        stats = signals.vbt.signals.stats()
        print("统计分析:", stats)
        ```
        
        应用场景：
        - 自动化信号分析
        - 策略性能评估
        - 信号质量分析
        - 批量策略测试
        """
        return self._metrics  # 返回指标配置

    # ############# Plotting ############# #

    def plot(self, yref: str = 'y', **kwargs) -> tp.Union[tp.BaseFigure, plotting.Scatter]:
        """
        绘制信号图表
        
        功能说明：
        创建信号数据的可视化图表，展示信号的时间分布和模式。
        该方法提供了专业的信号可视化功能，支持自定义图表样式和布局。
        图表使用布尔值刻度，便于直观理解信号状态。
        
        参数说明：
            yref (str): Y轴引用，默认为'y'
                - 用于多子图布局中的Y轴标识
            **kwargs: 传递给lineplot方法的关键字参数
                - 包括图表样式、布局、标题等参数
        
        返回值：
            tp.Union[tp.BaseFigure, plotting.Scatter]: 图表对象
                - 可能是完整的图表或散点图对象
        
        图表特性：
        - Y轴刻度为布尔值(False/True)
        - 支持时间序列索引
        - 多列信号自动分色显示
        - 可自定义样式和布局
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False],
            'c': [False, True, True, True, False]
        }, index=pd.date_range('2020-01-01', periods=5))
        
        # 绘制信号图表
        fig = signals.vbt.signals.plot(
            title="信号分析图表",
            ylabel="信号状态",
            showlegend=True
        )
        
        # 显示图表
        fig.show()
        ```
        
        应用场景：
        - 信号模式可视化
        - 策略信号分析
        - 信号质量评估
        - 策略报告生成
        
        注意事项：
        - 需要安装plotly库
        - 支持交互式图表
        - 可以导出为多种格式
        - 适合嵌入到报告中
        """
        default_layout = dict()  # 默认布局
        default_layout['yaxis' + yref[1:]] = dict(  # 设置Y轴属性
            tickmode='array',     # 数组刻度模式
            tickvals=[0, 1],      # 刻度值：0和1
            ticktext=['false', 'true']  # 刻度标签：false和true
        )
        return self.obj.vbt.lineplot(**merge_dicts(default_layout, kwargs))  # 调用lineplot方法并合并布局

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """
        图表绘制的默认参数
        
        功能说明：
        获取信号图表绘制的默认配置参数。
        该方法合并了GenericAccessor的默认参数和vectorbt设置中的信号图表配置。
        为图表绘制提供统一的参数配置。
        
        返回值：
            tp.Kwargs: 图表绘制的默认参数字典
                - 包含图表样式、布局、颜色等配置
                - 合并了基础配置和信号专用配置
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False]
        })
        
        # 获取默认图表参数
        defaults = signals.vbt.signals.plots_defaults
        print("默认图表参数:", defaults)
        
        # 使用默认参数绘制图表
        fig = signals.vbt.signals.plot()
        ```
        
        应用场景：
        - 图表样式配置
        - 批量图表生成
        - 报告图表标准化
        - 可视化参数管理
        """
        from vectorbt._settings import settings
        signals_plots_cfg = settings['signals']['plots']  # 获取信号图表配置

        return merge_dicts(
            GenericAccessor.plots_defaults.__get__(self),  # 基础默认参数
            signals_plots_cfg  # 信号专用配置
        )

    @property
    def subplots(self) -> Config:
        """
        子图配置
        
        功能说明：
        获取信号分析子图的配置信息。
        该属性定义了信号分析中各种子图的布局和配置。
        支持多子图布局，便于同时展示不同的信号分析结果。
        
        返回值：
            Config: 子图配置对象
                - 包含各种子图的布局配置
                - 支持自定义子图组合
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建信号数据
        signals = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [True, True, False, True, False]
        })
        
        # 获取子图配置
        subplot_config = signals.vbt.signals.subplots
        print("子图配置:", subplot_config)
        
        # 创建子图
        fig = signals.vbt.signals.plot_subplots()
        ```
        
        应用场景：
        - 多维度信号分析
        - 综合策略展示
        - 信号对比分析
        - 专业报告生成
        """
        return self._subplots  # 返回子图配置


SignalsAccessor.override_metrics_doc(__pdoc__)
SignalsAccessor.override_subplots_doc(__pdoc__)


@register_series_vbt_accessor('signals')
class SignalsSRAccessor(SignalsAccessor, GenericSRAccessor):
    """
    Series信号访问器类 - 专门处理pandas Series类型的信号数据
    
    功能概述：
    这是vectorbt信号系统中专门用于pandas Series的访问器类，继承自SignalsAccessor和GenericSRAccessor。
    该类为单列信号数据提供了专门的访问接口，包括信号生成、分析、可视化等功能。
    通过pandas accessor模式，可以通过pd.Series.vbt.signals访问所有信号处理功能。
    
    核心特性：
    - 专门处理pandas Series类型的信号数据
    - 继承SignalsAccessor的所有信号处理功能
    - 继承GenericSRAccessor的通用访问器功能
    - 提供Series专用的可视化方法
    - 支持单列信号的完整生命周期管理
    
    主要功能：
    - **信号生成**：基于自定义函数或随机算法生成信号
    - **信号分析**：统计分析、排序、索引管理等
    - **信号可视化**：专业的Series信号图形化展示
    - **信号操作**：清理、过滤、逻辑运算等
    
    技术特点：
    - 严格类型检查，确保输入数据为布尔型
    - 高性能计算，底层使用Numba编译优化
    - 内存高效，支持大规模数据处理
    - API一致，提供统一的访问接口
    
    使用场景：
    - 单列信号数据的处理和分析
    - 技术指标信号的生成和转换
    - 交易策略的入场出场信号管理
    - 信号模式的可视化分析
    
    示例用法：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建单列信号数据
    signals = pd.Series([True, False, True, False, True])
    
    # 基础统计分析
    total_signals = signals.vbt.signals.total()
    signal_rate = signals.vbt.signals.rate()
    
    # 生成随机出场信号
    exits = signals.vbt.signals.generate_random_exits(prob=0.3)
    
    # 可视化信号
    fig = signals.vbt.signals.plot_as_markers()
    ```
    
    与vectorbt生态系统的关系：
    - 继承自SignalsAccessor，获得完整的信号处理能力
    - 继承自GenericSRAccessor，获得通用访问器功能
    - 与signals.nb模块协同，提供底层计算支持
    - 与plotting模块集成，提供可视化功能
    """

    def __init__(self, obj: tp.Series, **kwargs) -> None:
        """
        初始化Series信号访问器
        
        功能说明：
        创建Series信号访问器实例，初始化父类功能。
        该方法会调用GenericSRAccessor和SignalsAccessor的初始化方法，
        确保访问器具有完整的功能。
        
        参数说明：
            obj (pd.Series): 要处理的信号Series对象
                - 必须是pandas的Series类型
                - 数据类型必须是布尔型(np.bool_)
                - True表示有信号，False表示无信号
            **kwargs: 传递给父类的关键字参数
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建布尔型信号Series
        signals = pd.Series([True, False, True, False, True])
        
        # 创建Series信号访问器
        accessor = signals.vbt.signals
        print(f"信号总数: {accessor.total()}")
        ```
        
        注意事项：
        - 输入数据必须是pandas Series类型
        - 数据类型必须是布尔型
        - 建议使用True/False而不是1/0来表示信号状态
        """
        GenericSRAccessor.__init__(self, obj, **kwargs)  # 初始化GenericSRAccessor父类
        SignalsAccessor.__init__(self, obj, **kwargs)    # 初始化SignalsAccessor父类

    def plot_as_markers(self, y: tp.Optional[tp.ArrayLike] = None,
                        **kwargs) -> tp.Union[tp.BaseFigure, plotting.Scatter]:  # pragma: no cover
        """
        将信号绘制为标记点
        
        功能说明：
        将Series信号数据绘制为散点图标记，便于在时间序列图表中标记信号位置。
        该方法为信号数据提供基础的可视化功能，支持自定义Y轴数据和样式配置。
        常用于在价格图表或其他时间序列图表中标记交易信号。
        
        参数说明：
            y (array_like, optional): Y轴数据，默认为None
                - None: 使用默认的Y轴数据（全为1的Series）
                - array_like: 自定义的Y轴数据，如价格序列
            **kwargs: 传递给scatterplot方法的关键字参数
                - 包括trace_kwargs、fig等参数
        
        返回值：
            tp.Union[tp.BaseFigure, plotting.Scatter]: 绘制的图表对象
                - 可能是完整的图表或散点图对象
        
        默认样式：
        - 标记形状：圆形(circle)
        - 标记颜色：蓝色(blue)
        - 标记大小：7
        - 边框宽度：1
        - 边框颜色：调整亮度的蓝色
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        
        # 创建信号数据
        signals = pd.Series([True, False, True, False, True])
        
        # 创建价格数据
        prices = pd.Series([100, 105, 98, 110, 95])
        
        # 绘制信号标记
        fig = signals.vbt.signals.plot_as_markers(y=prices)
        
        # 自定义样式
        fig = signals.vbt.signals.plot_as_markers(
            y=prices,
            trace_kwargs=dict(
                marker=dict(
                    symbol='diamond',
                    color='red',
                    size=10
                )
            )
        )
        ```
        
        应用场景：
        - 在价格图表中标记交易信号
        - 可视化技术指标的信号点
        - 展示策略的入场出场时机
        - 信号模式的可视化分析
        
        注意事项：
        - 信号数据必须是布尔型
        - Y轴数据应该与信号数据长度相同
        - 可以通过kwargs自定义标记样式
        - 支持与其他图表叠加显示
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']  # 获取绘图配置

        if y is None:
            # 如果没有提供Y轴数据，创建默认的全1序列
            y = pd.Series.vbt.empty_like(self.obj, 1)
        else:
            # 将Y轴数据转换为pandas数组格式
            y = reshape_fns.to_pd_array(y)

        # 调用scatterplot方法绘制标记，合并默认样式和自定义参数
        return y[self.obj].vbt.scatterplot(**merge_dicts(dict(
            trace_kwargs=dict(
                marker=dict(
                    symbol='circle',  # 默认圆形标记
                    color=plotting_cfg['contrast_color_schema']['blue'],  # 默认蓝色
                    size=7,  # 默认大小7
                    line=dict(
                        width=1,  # 边框宽度1
                        color=adjust_lightness(plotting_cfg['contrast_color_schema']['blue'])  # 边框颜色
                    )
                )
            )
        ), kwargs))

    def plot_as_entry_markers(self, y: tp.Optional[tp.ArrayLike] = None,
                              **kwargs) -> tp.Union[tp.BaseFigure, plotting.Scatter]:  # pragma: no cover
        """
        将信号绘制为入场标记
        
        功能说明：
        将Series信号数据绘制为入场标记，使用向上的三角形标记表示买入信号。
        该方法专门用于可视化交易策略的入场信号，具有特定的样式和语义。
        入场标记通常用绿色向上三角形表示，便于与出场标记区分。
        
        参数说明：
            y (array_like, optional): Y轴数据，默认为None
                - None: 使用默认的Y轴数据（全为1的Series）
                - array_like: 自定义的Y轴数据，如价格序列
            **kwargs: 传递给plot_as_markers方法的关键字参数
        
        返回值：
            tp.Union[tp.BaseFigure, plotting.Scatter]: 绘制的图表对象
        
        入场标记样式：
        - 标记形状：向上三角形(triangle-up)
        - 标记颜色：绿色(green)
        - 标记大小：8
        - 边框宽度：1
        - 边框颜色：调整亮度的绿色
        - 标记名称：'Entry'
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        
        # 创建入场信号数据
        entry_signals = pd.Series([True, False, True, False, True])
        
        # 创建价格数据
        prices = pd.Series([100, 105, 98, 110, 95])
        
        # 绘制入场标记
        fig = entry_signals.vbt.signals.plot_as_entry_markers(y=prices)
        
        # 在现有图表上添加入场标记
        fig = prices.vbt.lineplot()
        entry_signals.vbt.signals.plot_as_entry_markers(y=prices, fig=fig)
        ```
        
        应用场景：
        - 在价格图表中标记买入信号
        - 可视化策略的入场时机
        - 与出场标记配合使用
        - 交易策略的可视化分析
        
        注意事项：
        - 信号数据必须是布尔型
        - 入场标记使用绿色向上三角形
        - 可以与出场标记在同一图表中显示
        - 支持自定义样式覆盖默认设置
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']  # 获取绘图配置

        # 调用plot_as_markers方法，使用入场标记的默认样式
        return self.plot_as_markers(y=y, **merge_dicts(dict(
            trace_kwargs=dict(
                marker=dict(
                    symbol='triangle-up',  # 向上三角形表示入场
                    color=plotting_cfg['contrast_color_schema']['green'],  # 绿色表示买入
                    size=8,  # 稍大的标记
                    line=dict(
                        width=1,  # 边框宽度1
                        color=adjust_lightness(plotting_cfg['contrast_color_schema']['green'])  # 边框颜色
                    )
                ),
                name='Entry'  # 标记名称为Entry
            )
        ), kwargs))

    def plot_as_exit_markers(self, y: tp.Optional[tp.ArrayLike] = None,
                             **kwargs) -> tp.Union[tp.BaseFigure, plotting.Scatter]:  # pragma: no cover
        """
        将信号绘制为出场标记
        
        功能说明：
        将Series信号数据绘制为出场标记，使用向下的三角形标记表示卖出信号。
        该方法专门用于可视化交易策略的出场信号，具有特定的样式和语义。
        出场标记通常用红色向下三角形表示，便于与入场标记区分。
        
        参数说明：
            y (array_like, optional): Y轴数据，默认为None
                - None: 使用默认的Y轴数据（全为1的Series）
                - array_like: 自定义的Y轴数据，如价格序列
            **kwargs: 传递给plot_as_markers方法的关键字参数
        
        返回值：
            tp.Union[tp.BaseFigure, plotting.Scatter]: 绘制的图表对象
        
        出场标记样式：
        - 标记形状：向下三角形(triangle-down)
        - 标记颜色：红色(red)
        - 标记大小：8
        - 边框宽度：1
        - 边框颜色：调整亮度的红色
        - 标记名称：'Exit'
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        
        # 创建出场信号数据
        exit_signals = pd.Series([False, True, False, True, False])
        
        # 创建价格数据
        prices = pd.Series([100, 105, 98, 110, 95])
        
        # 绘制出场标记
        fig = exit_signals.vbt.signals.plot_as_exit_markers(y=prices)
        
        # 在现有图表上添加出场标记
        fig = prices.vbt.lineplot()
        exit_signals.vbt.signals.plot_as_exit_markers(y=prices, fig=fig)
        ```
        
        应用场景：
        - 在价格图表中标记卖出信号
        - 可视化策略的出场时机
        - 与入场标记配合使用
        - 交易策略的可视化分析
        
        注意事项：
        - 信号数据必须是布尔型
        - 出场标记使用红色向下三角形
        - 可以与入场标记在同一图表中显示
        - 支持自定义样式覆盖默认设置
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']  # 获取绘图配置

        # 调用plot_as_markers方法，使用出场标记的默认样式
        return self.plot_as_markers(y=y, **merge_dicts(dict(
            trace_kwargs=dict(
                marker=dict(
                    symbol='triangle-down',  # 向下三角形表示出场
                    color=plotting_cfg['contrast_color_schema']['red'],  # 红色表示卖出
                    size=8,  # 稍大的标记
                    line=dict(
                        width=1,  # 边框宽度1
                        color=adjust_lightness(plotting_cfg['contrast_color_schema']['red'])  # 边框颜色
                    )
                ),
                name='Exit'  # 标记名称为Exit
            )
        ), kwargs))


@register_dataframe_vbt_accessor('signals')
class SignalsDFAccessor(SignalsAccessor, GenericDFAccessor):
    """
    DataFrame信号访问器类 - 专门处理pandas DataFrame类型的信号数据
    
    功能概述：
    这是vectorbt信号系统中专门用于pandas DataFrame的访问器类，继承自SignalsAccessor和GenericDFAccessor。
    该类为多列信号数据提供了专门的访问接口，包括信号生成、分析、可视化等功能。
    通过pandas accessor模式，可以通过pd.DataFrame.vbt.signals访问所有信号处理功能。
    
    核心特性：
    - 专门处理pandas DataFrame类型的信号数据
    - 继承SignalsAccessor的所有信号处理功能
    - 继承GenericDFAccessor的通用访问器功能
    - 支持多列信号的批量处理
    - 提供DataFrame专用的统计和分析功能
    
    主要功能：
    - **信号生成**：基于自定义函数或随机算法生成多列信号
    - **信号分析**：多列信号的统计分析、排序、索引管理等
    - **信号可视化**：专业的DataFrame信号图形化展示
    - **信号操作**：多列信号的清理、过滤、逻辑运算等
    - **分组分析**：支持按列分组的统计分析
    
    技术特点：
    - 严格类型检查，确保输入数据为布尔型
    - 高性能计算，底层使用Numba编译优化
    - 内存高效，支持大规模多列数据处理
    - API一致，提供统一的访问接口
    - 支持分组操作和批量处理
    
    使用场景：
    - 多列信号数据的处理和分析
    - 多策略信号的批量管理
    - 多资产信号的对比分析
    - 信号模式的多维度研究
    
    示例用法：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建多列信号数据
    signals = pd.DataFrame({
        'strategy1': [True, False, True, False, True],
        'strategy2': [False, True, False, True, False],
        'strategy3': [True, True, False, False, True]
    })
    
    # 基础统计分析
    total_signals = signals.vbt.signals.total()
    signal_rate = signals.vbt.signals.rate()
    
    # 生成随机出场信号
    exits = signals.vbt.signals.generate_random_exits(prob=0.3)
    
    # 可视化信号
    fig = signals.vbt.signals.plot(title="多策略信号分析")
    ```
    
    与vectorbt生态系统的关系：
    - 继承自SignalsAccessor，获得完整的信号处理能力
    - 继承自GenericDFAccessor，获得通用DataFrame访问器功能
    - 与signals.nb模块协同，提供底层计算支持
    - 与portfolio模块集成，支持多策略回测
    - 与generic模块协作，提供统计和可视化功能
    """

    def __init__(self, obj: tp.Frame, **kwargs) -> None:
        """
        初始化DataFrame信号访问器
        
        功能说明：
        创建DataFrame信号访问器实例，初始化父类功能。
        该方法会调用GenericDFAccessor和SignalsAccessor的初始化方法，
        确保访问器具有完整的功能。
        
        参数说明：
            obj (pd.DataFrame): 要处理的信号DataFrame对象
                - 必须是pandas的DataFrame类型
                - 数据类型必须是布尔型(np.bool_)
                - True表示有信号，False表示无信号
                - 每列代表一个独立的信号序列
            **kwargs: 传递给父类的关键字参数
        
        使用示例：
        ```python
        import pandas as pd
        
        # 创建布尔型信号DataFrame
        signals = pd.DataFrame({
            'a': [True, False, True, False, True],
            'b': [False, True, False, True, False]
        })
        
        # 创建DataFrame信号访问器
        accessor = signals.vbt.signals
        print(f"信号总数: {accessor.total()}")
        ```
        
        注意事项：
        - 输入数据必须是pandas DataFrame类型
        - 所有列的数据类型必须是布尔型
        - 每列代表一个独立的信号序列
        - 支持分组操作和批量处理
        """
        GenericDFAccessor.__init__(self, obj, **kwargs)  # 初始化GenericDFAccessor父类
        SignalsAccessor.__init__(self, obj, **kwargs)    # 初始化SignalsAccessor父类
