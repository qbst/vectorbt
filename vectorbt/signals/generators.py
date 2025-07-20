# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT 信号生成器模块 - 量化交易信号生成的完整工具集
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中的信号生成器核心模块，通过SignalFactory工厂类构建了一整套
专业的交易信号生成器。这些生成器是量化交易策略开发的重要基础设施，为用户提供了从简单到复杂
的各种信号生成解决方案。

核心设计理念：
1. **工厂模式架构**：所有信号生成器都基于SignalFactory统一构建，确保API一致性和扩展性
2. **多元化信号类型**：涵盖随机信号、概率信号、技术信号和止损信号等多个维度
3. **高性能计算**：底层使用Numba JIT编译的函数，实现接近C语言的执行速度  
4. **灵活参数配置**：支持多种参数组合和广播，适应不同市场环境和交易策略
5. **可视化集成**：内置绘图功能，支持信号的直观分析和策略验证

信号生成器分类体系：

【随机信号生成器族 - RAND系列】
用于生成随机交易信号，主要用于策略压力测试、蒙特卡洛模拟和基准对比：
- RAND: 随机入场信号生成器，根据指定数量随机生成买入信号
- RANDX: 随机出场信号生成器，为每个入场信号随机生成对应的卖出信号
- RANDNX: 随机入场出场协同生成器，同时生成入场和出场信号

【概率信号生成器族 - RPROB系列】  
基于概率分布生成交易信号，支持更精细的随机化控制：
- RPROB: 基于概率的入场信号生成器，每个时点按概率决定是否生成信号
- RPROBX: 基于概率的出场信号生成器，为入场信号按概率生成出场
- RPROBCX: 基于概率的链式出场信号生成器，支持信号链式传递
- RPROBNX: 基于概率的双向信号生成器，同时控制入场和出场概率

【止损止盈信号生成器族 - ST系列】
实现专业的风险管理信号，支持各种止损止盈策略：  
- STX: 基于价格阈值的止损出场信号生成器，支持固定止损和追踪止损
- STCX: 基于价格阈值的链式止损信号生成器，支持连续止损策略

【OHLC高级止损信号生成器族 - OHLCST系列】
基于完整K线数据的高级止损策略，提供更精确的风险控制：
- OHLCSTX: OHLC止损出场信号生成器，支持止损、追踪止损、止盈等复合策略
- OHLCSTCX: OHLC链式止损信号生成器，支持连续复合止损策略

技术特点：
- **工厂统一接口**：所有生成器都通过SignalFactory构建，提供统一的run()方法
- **参数灵活配置**：支持标量、数组和多维参数配置，满足复杂策略需求
- **高性能计算**：底层使用Numba编译的函数，支持大规模数据处理
- **可视化支持**：内置plot()方法，支持信号的图形化展示和分析
- **类型安全**：完整的类型注解，支持IDE智能提示和静态检查

应用场景：
- **策略开发**：为量化交易策略提供入场和出场信号
- **风险管理**：实现止损止盈等风险控制机制
- **回测验证**：生成各种测试信号用于策略回测
- **压力测试**：使用随机信号测试策略鲁棒性
- **策略优化**：通过参数扫描优化信号生成参数

使用模式：
所有信号生成器都遵循统一的使用模式：
1. 使用run()方法运行生成器
2. 传入必要的输入数据(如价格、时间窗口)
3. 配置生成参数(如窗口大小、概率、阈值等)
4. 获取生成的信号数组(entries/exits)
5. 可选择使用plot()方法可视化结果

与vectorbt生态系统的关系：
- 基于signals.factory.SignalFactory构建
- 使用signals.nb模块的底层Numba函数
- 与portfolio模块协作进行策略回测
- 支持与indicators模块的技术指标集成

该模块是vectorbt框架中连接信号理论与实践应用的重要桥梁，为量化交易提供了工业级的
信号生成解决方案。

典型使用示例：
```python
import vectorbt as vbt
import pandas as pd
import numpy as np

# 创建示例价格数据
price_data = pd.Series(np.random.randn(100).cumsum() + 100)

# 1. 随机信号生成
rand_signals = vbt.RAND.run(input_shape=(100,), n=5, seed=42)
print("随机入场信号:", rand_signals.entries.sum())

# 2. 概率信号生成  
prob_signals = vbt.RPROB.run(input_shape=(100,), prob=0.1, seed=42)
print("概率入场信号:", prob_signals.entries.sum())

# 3. 止损信号生成(需要先有入场信号)
entries = rand_signals.entries
stop_exits = vbt.STX.run(entries, ts=price_data, stop=0.05, trailing=False)
print("止损出场信号:", stop_exits.exits.sum())

# 4. 可视化信号
rand_signals.plot(title="随机信号生成示例")
```
================================================================================

基于 vectorbt.signals.factory.SignalFactory 构建的信号生成器集合"""

import numpy as np  # 导入NumPy数值计算库，用于数组操作和数学计算
import plotly.graph_objects as go  # 导入Plotly图形对象，用于交互式数据可视化

from vectorbt import _typing as tp  # 导入vectorbt类型系统，提供类型注解支持
from vectorbt.indicators.configs import flex_col_param_config, flex_elem_param_config  # 导入灵活参数配置，支持多种参数广播模式
from vectorbt.signals.enums import StopType  # 导入止损类型枚举，定义不同的止损策略类型
from vectorbt.signals.factory import SignalFactory  # 导入信号工厂类，用于构建各种信号生成器
from vectorbt.signals.nb import (  # 导入Numba编译的底层信号处理函数
    rand_enex_apply_nb,      # 随机入场出场应用函数
    rand_by_prob_choice_nb,  # 基于概率的随机选择函数
    stop_choice_nb,          # 止损选择函数
    ohlc_stop_choice_nb,     # OHLC止损选择函数
    rand_choice_nb           # 随机选择函数
)
from vectorbt.utils.config import Config  # 导入配置类，用于管理生成器配置
from vectorbt.utils.figure import make_figure  # 导入图形创建工具，用于生成可视化图表

# ========================================
# 随机信号生成器族 (RAND Series)
# ========================================
# 
# RAND系列提供基于随机数生成的交易信号，主要用于:
# - 策略基准测试和对比分析  
# - 蒙特卡洛模拟和压力测试
# - 随机交易策略的构建和验证
# - 信号生成算法的性能基准

# 随机入场信号生成器
# 功能：根据指定数量随机生成入场信号位置
# 应用：策略压力测试、随机交易基准、蒙特卡洛模拟
RAND = SignalFactory(
    class_name='RAND',           # 生成的类名，用于标识这个随机入场信号生成器
    module_name=__name__,        # 模块名，指向当前文件
    short_name='rand',           # 简短名称，用于参数前缀和内部标识
    mode='entries',              # 信号模式：仅生成入场信号(entries)
    param_names=['n']            # 参数名列表：n表示要生成的信号数量
).from_choice_func(              # 使用选择函数方法构建信号生成器
    entry_choice_func=rand_choice_nb,  # 入场选择函数：使用Numba编译的随机选择函数
    entry_settings=dict(         # 入场函数设置
        pass_params=['n']        # 向选择函数传递参数n(信号数量)
    ),
    param_settings=dict(         # 参数设置配置
        n=flex_col_param_config  # n参数使用灵活列参数配置，支持标量或数组
    ),
    seed=None                    # 随机种子：默认不设置，每次运行结果不同
)


class _RAND(RAND):
    """
    随机入场信号生成器类
    
    功能概述：
    根据指定的信号数量随机生成入场信号，是策略测试和基准对比的重要工具。
    该生成器使用vectorbt.signals.nb.rand_choice_nb函数作为底层实现，
    通过随机选择算法在指定的时间窗口内生成预定数量的入场信号。
    
    核心特性：
    - 支持固定数量的随机信号生成
    - 支持多种参数配置模式(标量、数组、列表)
    - 提供随机种子控制，确保结果可复现
    - 支持多列并行处理，适用于多资产组合
    
    参数说明：
        n (int/array/list): 要生成的入场信号数量
            - 标量：所有列使用相同的信号数量
            - 数组：每列可以有不同的信号数量  
            - 列表：生成多个参数组合进行对比测试
    
    算法逻辑：
    1. 根据输入的时间序列长度确定可选择的时间点范围
    2. 使用随机数生成器从可用时间点中随机选择n个位置
    3. 将选中的时间点标记为True，其余位置标记为False
    4. 返回布尔类型的信号数组
    
    应用场景：
    - **策略基准测试**：生成随机入场信号作为策略表现的基准
    - **蒙特卡洛模拟**：模拟随机交易行为的长期统计特性  
    - **压力测试**：测试交易系统在随机信号下的鲁棒性
    - **算法验证**：验证信号处理算法的正确性
    
    使用示例：
        ```python
        # 示例1：测试三种不同的入场信号数量
        >>> import vectorbt as vbt
        >>> rand = vbt.RAND.run(input_shape=(6,), n=[1, 2, 3], seed=42)
        >>> print(rand.entries)
        rand_n      1      2      3
        0        True   True   True
        1       False  False   True
        2       False  False  False
        3       False   True  False
        4       False  False   True
        5       False  False  False

        # 示例2：每列设置不同的信号数量
        >>> import numpy as np
        >>> rand = vbt.RAND.run(input_shape=(8, 2), n=[np.array([1, 2]), 3], seed=42)
        >>> print(rand.entries)
        rand_n      1      2      3      3
                    0      1      0      1
        0       False  False   True  False
        1        True  False  False  False
        2       False  False  False   True
        3       False   True   True  False
        4       False  False  False  False
        5       False  False  False   True
        6       False  False   True  False
        7       False   True  False   True
        
        # 示例3：单一配置的简单使用
        >>> simple_rand = vbt.RAND.run(input_shape=(10,), n=3, seed=123)
        >>> print(f"生成的信号数量: {simple_rand.entries.sum()}")
        >>> print(f"信号位置: {simple_rand.entries[simple_rand.entries].index.tolist()}")
        ```
    
    性能特点：
    - 使用Numba编译的底层函数，执行速度快
    - 支持向量化操作，可同时处理多个资产
    - 内存使用高效，适合大规模数据处理
    
    注意事项：
    - 信号数量n不能超过时间序列的长度
    - 设置随机种子(seed)可确保结果可重现
    - 生成的信号位置是完全随机的，不依赖任何市场数据
    """
    pass


setattr(RAND, '__doc__', _RAND.__doc__)  # 将文档字符串绑定到RAND类

# 随机出场信号生成器
# 功能：为每个入场信号随机生成对应的出场信号
# 应用：完整的随机交易策略构建、信号配对测试
RANDX = SignalFactory(
    class_name='RANDX',          # 生成的类名，X表示退出(eXit)信号
    module_name=__name__,        # 模块名，指向当前文件
    short_name='randx',          # 简短名称，x后缀表示出场信号
    mode='exits'                 # 信号模式：仅生成出场信号(exits)
).from_choice_func(              # 使用选择函数方法构建信号生成器
    exit_choice_func=rand_choice_nb,  # 出场选择函数：使用随机选择函数
    exit_settings=dict(          # 出场函数设置
        pass_kwargs=dict(n=1)    # 传递关键字参数：每个入场信号对应1个出场信号
    ),
    seed=None                    # 随机种子：默认不设置
)


class _RANDX(RANDX):
    """
    随机出场信号生成器类
    
    功能概述：
    基于已有的入场信号，为每个入场信号随机生成对应的出场信号。该生成器是构建完整
    随机交易策略的重要组件，能够与入场信号配对形成完整的交易周期。
    
    核心特性：
    - 依赖输入的入场信号数组(entries)
    - 为每个入场信号生成一个随机出场信号
    - 确保出场信号在对应入场信号之后发生
    - 支持随机种子控制，保证结果可重现
    
    算法逻辑：
    1. 扫描输入的入场信号数组，找到所有True的位置
    2. 对于每个入场信号，在其之后的时间范围内随机选择一个出场位置
    3. 将选中的位置标记为True，生成出场信号数组
    4. 确保出场信号不会在入场信号之前或同时发生
    
    应用场景：
    - **完整随机策略**：与RAND配合构建完整的随机买卖策略
    - **信号配对测试**：测试入场出场信号的配对逻辑
    - **交易周期分析**：分析随机交易的持仓时间分布
    - **策略基准构建**：作为其他策略的随机基准对比
    
    参数继承：
    继承RAND的参数说明，但在出场信号生成中，n固定为1(每个入场对应1个出场)
    
    使用示例：
        ```python
        # 为给定的入场信号生成随机出场信号
        >>> import vectorbt as vbt
        >>> import pandas as pd
        >>> 
        >>> # 定义入场信号：在第0和第3个时间点入场
        >>> entries = pd.Series([True, False, False, True, False, False])
        >>> randx = vbt.RANDX.run(entries, seed=42)
        >>> 
        >>> print("入场信号:", entries.tolist())
        >>> print("出场信号:", randx.exits.tolist())
        >>> # 输出结果:
        >>> # 入场信号: [True, False, False, True, False, False]
        >>> # 出场信号: [False, False, True, False, True, False]
        >>> 
        >>> # 验证信号配对关系
        >>> print("第1笔交易: 入场第0天, 出场第2天, 持仓2天")
        >>> print("第2笔交易: 入场第3天, 出场第4天, 持仓1天")
        
        # 批量处理多个资产
        >>> import numpy as np
        >>> entries_multi = pd.DataFrame({
        ...     'asset1': [True, False, True, False, False],
        ...     'asset2': [False, True, False, True, False]
        ... })
        >>> randx_multi = vbt.RANDX.run(entries_multi, seed=42)
        >>> print("多资产出场信号:")
        >>> print(randx_multi.exits)
        ```
    
    技术特点：
    - 自动处理信号时序关系，确保出场在入场之后
    - 支持多列并行处理，适用于多资产组合
    - 使用高效的随机选择算法
    - 与portfolio模块完美集成
    
    注意事项：
    - 必须提供有效的入场信号作为输入
    - 出场信号的生成范围受到时间序列长度限制
    - 如果入场信号在序列末尾，可能无法生成对应的出场信号
    - 设置相同的随机种子可确保结果一致性
    """
    pass


setattr(RANDX, '__doc__', _RANDX.__doc__)  # 将文档字符串绑定到RANDX类

# 随机入场出场协同生成器
# 功能：同时生成随机的入场和出场信号，形成完整的交易信号对
# 应用：完整随机策略构建、交易系统压力测试
RANDNX = SignalFactory(
    class_name='RANDNX',         # 生成的类名，NX表示同时生成入场和出场信号
    module_name=__name__,        # 模块名，指向当前文件
    short_name='randnx',         # 简短名称，nx后缀表示入场+出场
    mode='both',                 # 信号模式：同时生成入场和出场信号
    param_names=['n']            # 参数名列表：n表示要生成的信号对数量
).from_apply_func(               # 使用应用函数方法构建(函数几乎是向量化的)
    rand_enex_apply_nb,          # 随机入场出场应用函数：Numba编译的协同生成函数
    require_input_shape=True,    # 需要输入形状：因为不依赖外部数据，需要指定数据维度
    param_settings=dict(         # 参数设置配置
        n=flex_col_param_config  # n参数使用灵活列参数配置
    ),
    kwargs_to_args=['entry_wait', 'exit_wait'],  # 关键字参数转位置参数
    entry_wait=1,                # 入场等待期：入场信号后需要等待的周期数
    exit_wait=1,                 # 出场等待期：出场信号后需要等待的周期数
    seed=None                    # 随机种子：默认不设置
)


class _RANDNX(RANDNX):
    """
    随机入场出场协同信号生成器类
    
    功能概述：
    同时生成指定数量的随机入场和出场信号对，形成完整的交易信号序列。该生成器
    能够确保入场和出场信号的正确配对，是构建完整随机交易策略的核心工具。
    
    核心特性：
    - 同时生成入场和出场信号(entries & exits)
    - 自动处理信号配对关系，确保每个入场都有对应的出场
    - 支持指定交易对数量，便于策略参数优化
    - 提供等待期控制，模拟实际交易中的延迟
    
    算法逻辑：
    1. 根据指定的信号对数量n，在时间序列中随机选择n个入场点
    2. 为每个入场点在其后的时间范围内随机选择对应的出场点
    3. 应用入场等待期和出场等待期，模拟实际交易延迟
    4. 生成两个布尔数组：entries(入场信号) 和 exits(出场信号)
    
    参数说明：
        n (int/array/list): 要生成的交易信号对数量
            - 与RAND中的参数含义相同
        entry_wait (int): 入场等待期，入场信号触发后需要等待的周期数
        exit_wait (int): 出场等待期，出场信号触发后需要等待的周期数
    
    应用场景：
    - **完整随机策略**：构建包含完整交易逻辑的随机策略
    - **策略性能基准**：作为其他策略的随机基准对比
    - **交易系统测试**：测试交易系统的信号处理能力
    - **参数优化验证**：验证参数优化算法的有效性
    
    使用示例：
        ```python
        # 测试三种不同的交易对数量
        >>> import vectorbt as vbt
        >>> randnx = vbt.RANDNX.run(
        ...     input_shape=(6,),
        ...     n=[1, 2, 3],
        ...     seed=42)

        >>> print("入场信号:")
        >>> print(randnx.entries)
        randnx_n      1      2      3
        0          True   True   True
        1         False  False  False
        2         False   True   True
        3         False  False  False
        4         False  False   True
        5         False  False  False

        >>> print("出场信号:")
        >>> print(randnx.exits)
        randnx_n      1      2      3
        0         False  False  False
        1          True   True   True
        2         False  False  False
        3         False   True   True
        4         False  False  False
        5         False  False   True
        
        # 分析交易对的配对关系
        >>> for col in randnx.entries.columns:
        ...     entries = randnx.entries[col]
        ...     exits = randnx.exits[col] 
        ...     print(f"\n{col}列的交易对:")
        ...     entry_idx = entries[entries].index.tolist()
        ...     exit_idx = exits[exits].index.tolist()
        ...     for i, (e, x) in enumerate(zip(entry_idx, exit_idx)):
        ...         print(f"  交易{i+1}: 入场第{e}天, 出场第{x}天, 持仓{x-e}天")
        ```
    
    技术特点：
    - 使用向量化算法，支持高效批量处理
    - 自动确保信号时序的合理性
    - 支持多种等待期配置，模拟真实交易环境
    - 与portfolio模块无缝集成，支持完整的回测流程
    
    与其他生成器的关系：
    - 相比RAND：同时生成出场信号，更完整
    - 相比RANDX：不需要预先提供入场信号，更独立
    - 可作为其他复杂策略的随机化基准
    
    注意事项：
    - 信号对数量受时间序列长度和等待期参数限制
    - 等待期参数会影响可用的信号生成空间
    - 在短时间序列中使用大的n值可能导致信号生成失败
    """
    pass


setattr(RANDNX, '__doc__', _RANDNX.__doc__)  # 将文档字符串绑定到RANDNX类

# ========================================  
# 概率信号生成器族 (RPROB Series)
# ========================================
# 
# RPROB系列提供基于概率分布的信号生成，特点是：
# - 每个时间点都有独立的信号生成概率
# - 支持动态概率调整和时变概率分布
# - 提供更精细的随机化控制机制
# - 适用于复杂的随机化交易策略构建

# 概率入场信号生成器
# 功能：根据指定概率在每个时间点随机生成入场信号  
# 应用：概率交易策略、动态信号生成、复杂随机模拟
RPROB = SignalFactory(
    class_name='RPROB',          # 生成的类名，PROB表示基于概率
    module_name=__name__,        # 模块名，指向当前文件
    short_name='rprob',          # 简短名称，rprob表示随机概率
    mode='entries',              # 信号模式：仅生成入场信号
    param_names=['prob']         # 参数名列表：prob表示信号生成概率
).from_choice_func(              # 使用选择函数方法构建
    entry_choice_func=rand_by_prob_choice_nb,  # 入场选择函数：基于概率的随机选择
    entry_settings=dict(         # 入场函数设置
        pass_params=['prob'],    # 传递概率参数
        pass_kwargs=['pick_first', 'temp_idx_arr', 'flex_2d']  # 传递辅助参数
    ),
    pass_flex_2d=True,           # 启用灵活的2D数组处理
    param_settings=dict(         # 参数设置配置  
        prob=flex_elem_param_config,  # prob参数使用灵活元素参数配置，支持各种广播
    ),
    seed=None                    # 随机种子：默认不设置
)


class _RPROB(RPROB):
    """
    概率入场信号生成器类
    
    功能概述：
    基于概率分布生成随机入场信号，每个时间点都有独立的信号生成概率。
    相比固定数量的随机信号生成，这种方式提供了更灵活的概率控制，
    能够模拟更接近真实市场的随机性特征。
    
    核心特性：
    - 基于概率分布的信号生成机制
    - 每个时间点独立计算信号生成概率
    - 支持时变概率(每个时间点不同概率)
    - 支持空间变概率(每列、每行、每个元素不同概率)
    - 提供完全的概率控制和统计特性
    
    算法逻辑：
    1. 对每个时间点生成一个0-1之间的随机数
    2. 将随机数与指定的概率阈值进行比较
    3. 如果随机数小于概率值，则在该位置生成信号(True)
    4. 否则不生成信号(False)
    5. 重复此过程直到处理完所有时间点
    
    参数说明：
        prob (float/array/list): 信号生成概率，范围[0, 1]
            - 标量：所有时间点使用相同概率
            - 行数组：每个时间点使用不同概率(时变概率)
            - 列数组：每列使用不同概率(资产特定概率)
            - 2D数组：每个位置使用不同概率(完全自定义)
            - 列表：生成多个概率组合用于参数对比
    
    概率含义：
    - prob=0.0: 永不生成信号
    - prob=0.1: 平均每10个时间点生成1个信号
    - prob=0.5: 平均每2个时间点生成1个信号  
    - prob=1.0: 每个时间点都生成信号
    
    应用场景：
    - **概率交易策略**：基于市场状态概率调整信号频率
    - **动态信号生成**：根据市场波动性调整信号概率
    - **随机策略基准**：提供可控的随机信号基准
    - **A/B测试**：对比不同概率参数的策略效果
    
    使用示例：
        ```python
        # 示例1：生成三列不同概率的入场信号
        >>> import vectorbt as vbt
        >>> rprob = vbt.RPROB.run(input_shape=(5,), prob=[0., 0.5, 1.], seed=42)
        >>> print(rprob.entries)
        rprob_prob    0.0    0.5   1.0
        0           False   True  True
        1           False   True  True
        2           False  False  True
        3           False  False  True
        4           False  False  True
        
        # 分析信号生成统计
        >>> for col in rprob.entries.columns:
        ...     signal_count = rprob.entries[col].sum()
        ...     total_periods = len(rprob.entries)
        ...     actual_prob = signal_count / total_periods
        ...     expected_prob = float(col.split('_')[-1])
        ...     print(f"概率{expected_prob}: 期望{expected_prob*total_periods:.1f}个信号, "
        ...           f"实际{signal_count}个信号, 实际概率{actual_prob:.2f}")

        # 示例2：时变概率 - 每个时间点不同概率
        >>> import numpy as np
        >>> time_varying_prob = np.array([0., 0., 1., 1., 1.])  # 前两天概率0，后三天概率1
        >>> rprob_tv = vbt.RPROB.run(input_shape=(5,), prob=time_varying_prob, seed=42)
        >>> print("时变概率信号:")
        >>> print(rprob_tv.entries)
        0    False
        1    False
        2     True   # 概率从这里开始变为1
        3     True
        4     True
        
        # 示例3：多资产不同概率
        >>> multi_asset_prob = np.array([[0.1, 0.3, 0.5, 0.7, 0.9]]).T  # 每行不同概率
        >>> rprob_ma = vbt.RPROB.run(input_shape=(5, 5), prob=multi_asset_prob, seed=42)
        >>> print("多资产概率信号生成率:")
        >>> for i, col in enumerate(rprob_ma.entries.columns):
        ...     rate = rprob_ma.entries.iloc[:, i].sum() / len(rprob_ma.entries)
        ...     print(f"资产{i}: 期望概率{multi_asset_prob[i, 0]:.1f}, 实际概率{rate:.2f}")
        ```
    
    统计特性：
    - 在大样本情况下，实际信号频率会收敛到设定的概率值
    - 信号生成符合伯努利分布的统计特性
    - 支持概率的各种统计分析和验证
    
    技术特点：
    - 使用高效的向量化概率计算
    - 支持复杂的概率广播机制
    - 内置统计验证和分析功能
    - 与其他模块完美集成
    
    注意事项：
    - 概率值必须在[0, 1]范围内
    - 实际信号数量可能与期望值有随机偏差
    - 在短时间序列中，统计特性可能不够稳定
    - 设置随机种子可确保结果可重现
    """
    pass


setattr(RPROB, '__doc__', _RPROB.__doc__)  # 将文档字符串绑定到RPROB类

# RPROBX生成器的配置对象
# 功能：定义概率出场信号生成器的基础配置参数
rprobx_config = Config(
    dict(
        class_name='RPROBX',         # 类名：概率出场信号生成器
        module_name=__name__,        # 模块名：当前文件
        short_name='rprobx',         # 简短名称：rprobx表示随机概率出场
        mode='exits',                # 信号模式：仅生成出场信号
        param_names=['prob']         # 参数列表：prob表示出场概率
    )
)
"""RPROBX生成器的工厂配置对象 - 定义概率出场信号生成器的基本结构"""

# RPROBX生成器的函数配置对象  
# 功能：定义概率出场信号生成器的函数调用配置
rprobx_func_config = Config(
    dict(
        exit_choice_func=rand_by_prob_choice_nb,  # 出场选择函数：基于概率的随机选择
        exit_settings=dict(          # 出场设置
            pass_params=['prob'],    # 传递概率参数
            pass_kwargs=['pick_first', 'temp_idx_arr', 'flex_2d']  # 传递辅助参数
        ),
        pass_flex_2d=True,           # 启用灵活2D处理
        param_settings=dict(         # 参数设置
            prob=flex_elem_param_config  # 概率参数的灵活配置
        ),
        seed=None                    # 随机种子
    )
)
"""RPROBX生成器的函数配置对象 - 定义概率出场信号的具体生成逻辑"""

# 概率出场信号生成器
# 功能：基于概率为入场信号生成对应的出场信号
# 应用：完整概率策略构建、灵活出场控制
RPROBX = SignalFactory(
    **rprobx_config              # 展开基础配置参数
).from_choice_func(
    **rprobx_func_config         # 展开函数配置参数
)


class _RPROBX(RPROBX):
    """
    概率出场信号生成器类
    
    功能概述：
    基于概率为已有的入场信号生成对应的出场信号，每个入场信号都有独立的
    出场概率计算。这种方式相比固定出场策略更加灵活，能够适应不同的
    市场环境和交易策略需求。
    
    核心特性：
    - 依赖输入的入场信号数组作为基础
    - 每个入场信号对应的出场概率可独立设置
    - 支持动态概率调整和时变概率控制
    - 保持入场出场信号的时序关系正确性
    
    算法逻辑：
    1. 扫描入场信号数组，识别所有入场点
    2. 对于每个入场点，从下一个时间点开始计算出场概率
    3. 在每个候选出场点生成随机数并与概率比较
    4. 一旦生成出场信号，该入场信号的出场过程结束
    5. 继续处理下一个入场信号
    
    参数继承：
    继承RPROB的概率参数说明，支持相同的灵活配置方式
    
    应用场景：
    - **灵活出场策略**：基于市场条件动态调整出场概率
    - **风险管理优化**：在不同市场状态下采用不同出场概率
    - **策略参数测试**：测试不同出场概率对策略的影响
    - **随机出场基准**：作为复杂出场策略的随机基准
    
    技术特点：
    - 继承RPROB系列的所有概率配置特性
    - 自动处理信号时序关系
    - 支持多资产并行处理
    - 与入场信号生成器完美搭配
    
    使用说明：
    参数配置和使用方法与RPROB类似，详见RPROB的参数说明文档
    """
    pass


setattr(RPROBX, '__doc__', _RPROBX.__doc__)  # 将文档字符串绑定到RPROBX类

# 概率链式出场信号生成器
# 功能：基于概率生成链式出场信号，支持信号的连续传递
# 应用：复杂交易策略、信号链式处理、动态信号管理
RPROBCX = SignalFactory(
    **rprobx_config.merge_with(  # 基于rprobx配置，并合并新的参数
        dict(
            class_name='RPROBCX',    # 类名：概率链式出场信号生成器，CX表示Chain eXit
            short_name='rprobcx',    # 简短名称：rprobcx表示随机概率链式出场
            mode='chain'             # 信号模式：链式模式，生成new_entries和exits
        )
    )
).from_choice_func(
    **rprobx_func_config         # 使用相同的函数配置
)


class _RPROBCX(RPROBCX):
    """
    概率链式出场信号生成器类
    
    功能概述：
    生成基于概率的链式出场信号序列，能够在原有入场信号基础上产生新的
    入场信号和对应的出场信号。这种链式机制特别适用于需要连续信号处理
    的复杂交易策略。
    
    核心特性：
    - 链式信号生成模式(Chain mode)
    - 基于概率的灵活出场控制  
    - 生成new_entries和exits两个输出
    - 支持信号的连续传递和处理
    
    信号生成逻辑：
    1. 接收原始的入场信号(entries)作为输入
    2. 基于概率为每个入场信号生成出场信号
    3. 同时生成新的入场信号(new_entries)用于链式处理
    4. 返回两个信号数组供下游使用
    
    链式模式优势：
    - 支持多层信号处理和传递
    - 适用于复杂的交易逻辑构建
    - 便于与其他信号生成器组合使用
    - 提供更灵活的信号管理机制
    
    应用场景：
    - **多阶段交易策略**：实现复杂的多步骤交易逻辑
    - **信号链式处理**：构建信号处理管道
    - **动态策略调整**：根据信号结果动态调整后续策略
    - **复合信号生成**：组合多种信号生成器
    
    参数继承：
    继承RPROB系列的完整概率参数配置能力
    
    技术特点：
    - 完全兼容vectorbt的链式处理架构
    - 支持复杂的信号流管理
    - 高效的链式计算实现
    - 与其他链式生成器无缝集成
    """
    pass


setattr(RPROBCX, '__doc__', _RPROBCX.__doc__)  # 将文档字符串绑定到RPROBCX类

# 概率双向信号生成器
# 功能：同时生成概率控制的入场和出场信号
# 应用：完整概率策略、独立双概率控制、复杂随机策略
RPROBNX = SignalFactory(
    class_name='RPROBNX',        # 类名：概率双向信号生成器，NX表示同时生成入场和出场
    module_name=__name__,        # 模块名：当前文件
    short_name='rprobnx',        # 简短名称：rprobnx表示随机概率双向
    mode='both',                 # 信号模式：同时生成入场和出场信号
    param_names=['entry_prob', 'exit_prob']  # 参数列表：分别控制入场和出场概率
).from_choice_func(              # 使用选择函数方法构建
    entry_choice_func=rand_by_prob_choice_nb,  # 入场选择函数：基于概率的随机选择
    entry_settings=dict(         # 入场设置
        pass_params=['entry_prob'],    # 传递入场概率参数
        pass_kwargs=['pick_first', 'temp_idx_arr', 'flex_2d']  # 传递辅助参数
    ),
    exit_choice_func=rand_by_prob_choice_nb,   # 出场选择函数：基于概率的随机选择
    exit_settings=dict(          # 出场设置
        pass_params=['exit_prob'],     # 传递出场概率参数
        pass_kwargs=['pick_first', 'temp_idx_arr', 'flex_2d']  # 传递辅助参数
    ),
    pass_flex_2d=True,           # 启用灵活2D处理
    param_settings=dict(         # 参数设置
        entry_prob=flex_elem_param_config,    # 入场概率的灵活配置
        exit_prob=flex_elem_param_config      # 出场概率的灵活配置
    ),
    seed=None                    # 随机种子
)


class _RPROBNX(RPROBNX):
    """
    概率双向信号生成器类
    
    功能概述：
    同时基于独立的概率参数生成入场和出场信号，提供了最灵活的概率控制机制。
    该生成器允许用户分别设置入场概率和出场概率，构建完全定制化的概率交易策略。
    
    核心特性：
    - 双独立概率控制：入场概率和出场概率完全独立
    - 同时生成entries和exits两个信号数组
    - 支持复杂的概率组合和参数扫描
    - 提供完整的概率交易策略构建能力
    
    算法逻辑：
    1. 根据entry_prob参数在每个时间点生成入场信号
    2. 根据exit_prob参数在每个时间点生成出场信号
    3. 两个过程完全独立，不存在依赖关系
    4. 最终输出两个独立的布尔信号数组
    
    参数说明：
        entry_prob (float/array/list): 入场信号生成概率，范围[0, 1]
            - 继承RPROB系列的完整概率配置能力
        exit_prob (float/array/list): 出场信号生成概率，范围[0, 1]
            - 同样支持所有灵活的概率配置方式
    
    应用场景：
    - **完全概率策略**：构建基于双概率控制的完整交易策略
    - **参数敏感性分析**：分析不同概率组合对策略的影响
    - **策略对比测试**：通过参数笛卡尔积进行策略对比
    - **复杂随机模拟**：模拟复杂的随机交易行为
    
    使用示例：
        ```python
        # 示例1：测试所有概率组合(笛卡尔积)
        >>> import vectorbt as vbt
        >>> rprobnx = vbt.RPROBNX.run(
        ...     input_shape=(5,),
        ...     entry_prob=[0.5, 1.],
        ...     exit_prob=[0.5, 1.],
        ...     param_product=True,    # 启用参数笛卡尔积
        ...     seed=42)

        >>> print("入场信号矩阵:")
        >>> print(rprobnx.entries)
        rprobnx_entry_prob    0.5    0.5    1.0    0.5
        rprobnx_exit_prob     0.5    1.0    0.5    1.0
        0                    True   True   True   True
        1                   False  False  False  False
        2                   False  False  False   True
        3                   False  False  False  False
        4                   False  False   True   True

        >>> print("出场信号矩阵:")
        >>> print(rprobnx.exits)
        rprobnx_entry_prob    0.5    0.5    1.0    1.0
        rprobnx_exit_prob     0.5    1.0    0.5    1.0
        0                   False  False  False  False
        1                   False   True  False   True
        2                   False  False  False  False
        3                   False  False   True   True
        4                    True  False  False  False
        
        # 分析不同概率组合的信号特征
        >>> for col in rprobnx.entries.columns:
        ...     entry_rate = rprobnx.entries[col].mean()
        ...     exit_rate = rprobnx.exits[col].mean()
        ...     print(f"{col}: 实际入场率{entry_rate:.2f}, 实际出场率{exit_rate:.2f}")

        # 示例2：时变概率设计 - 每个时间点使用不同概率
        >>> import numpy as np
        >>> entry_prob1 = np.asarray([1., 0., 1., 0., 1.])  # 奇数时点入场
        >>> entry_prob2 = np.asarray([0., 1., 0., 1., 0.])  # 偶数时点入场  
        >>> rprobnx_tv = vbt.RPROBNX.run(
        ...     input_shape=(5,),
        ...     entry_prob=[entry_prob1, entry_prob2],
        ...     exit_prob=1.,    # 所有时点都尝试出场
        ...     seed=42)

        >>> print("时变入场概率信号:")
        >>> print(rprobnx_tv.entries)
        rprobnx_entry_prob array_0 array_1
        rprobnx_exit_prob      1.0     1.0
        0                     True   False   # array_0在偶数位置入场
        1                    False    True   # array_1在奇数位置入场
        2                     True   False
        3                    False    True
        4                     True   False

        >>> print("对应的出场信号:")
        >>> print(rprobnx_tv.exits)
        rprobnx_entry_prob array_0 array_1
        rprobnx_exit_prob      1.0     1.0
        0                    False   False
        1                     True   False   # array_0的出场信号
        2                    False    True   # array_1的出场信号
        3                     True   False
        4                    False    True
        ```
    
    策略构建优势：
    - **完全控制**：对入场和出场时机提供精确的概率控制
    - **独立优化**：可以独立优化入场和出场策略
    - **复杂组合**：支持构建非常复杂的概率组合策略
    - **统计验证**：便于进行大样本统计验证
    
    技术特点：
    - 支持多维参数广播和笛卡尔积组合
    - 高效的向量化概率计算
    - 完整的统计特性支持
    - 与其他模块无缝集成
    
    注意事项：
    - 入场和出场信号是独立生成的，可能出现同时为True的情况
    - 在实际应用中需要处理信号冲突的逻辑
    - 概率参数必须在有效范围[0, 1]内
    - 建议使用大样本进行统计验证
    """
    pass


setattr(RPROBNX, '__doc__', _RPROBNX.__doc__)  # 将文档字符串绑定到RPROBNX类

# ========================================
# 止损止盈信号生成器族 (ST Series)
# ========================================
# 
# ST系列提供专业的风险管理信号生成，特点是：
# - 基于价格阈值的止损止盈逻辑
# - 支持固定止损和追踪止损策略
# - 提供精确的风险控制机制
# - 适用于各种风险管理策略

# STX生成器的基础配置对象
# 功能：定义止损出场信号生成器的核心结构
stx_config = Config(
    dict(
        class_name='STX',            # 类名：止损出场信号生成器，ST表示STop，X表示eXit
        module_name=__name__,        # 模块名：当前文件
        short_name='stx',            # 简短名称：stx表示止损出场
        mode='exits',                # 信号模式：仅生成出场信号
        input_names=['ts'],          # 输入名称：ts表示时间序列(通常是价格数据)
        param_names=['stop', 'trailing']  # 参数名称：stop止损阈值，trailing是否追踪止损
    )
)
"""STX生成器的工厂配置对象 - 定义基于价格的止损出场信号生成器结构"""

# STX生成器的函数配置对象
# 功能：定义止损出场信号的具体计算逻辑
stx_func_config = Config(
    dict(
        exit_choice_func=stop_choice_nb,  # 出场选择函数：基于止损条件的选择函数
        exit_settings=dict(          # 出场设置配置
            pass_inputs=['ts'],      # 传递时间序列数据(价格)
            pass_params=['stop', 'trailing'],  # 传递止损参数
            pass_kwargs=['wait', 'pick_first', 'temp_idx_arr', 'flex_2d']  # 传递辅助参数
        ),
        pass_flex_2d=True,           # 启用灵活2D处理
        param_settings=dict(         # 参数设置
            stop=flex_elem_param_config,      # 止损阈值的灵活配置
            trailing=flex_elem_param_config   # 追踪止损标志的灵活配置
        ),
        trailing=False               # 默认值：不使用追踪止损
    )
)
"""STX生成器的函数配置对象 - 定义止损出场信号的计算和选择逻辑"""

# 止损出场信号生成器
# 功能：基于价格阈值生成止损出场信号
# 应用：风险管理、止损策略、资金保护
STX = SignalFactory(
    **stx_config                 # 展开基础配置参数
).from_choice_func(
    **stx_func_config            # 展开函数配置参数
)


class _STX(STX):
    """
    止损出场信号生成器类
    
    功能概述：
    基于价格阈值生成止损出场信号，是专业风险管理系统的核心组件。该生成器
    支持固定止损和追踪止损两种模式，能够在价格达到预设条件时自动触发出场信号，
    有效保护投资者的资金安全。
    
    核心特性：
    - 基于实时价格数据的止损判断
    - 支持固定止损(Fixed Stop Loss)和追踪止损(Trailing Stop)
    - 提供灵活的止损阈值设置
    - 自动处理止损信号的时序逻辑
    
    算法逻辑：
    【固定止损模式 (trailing=False)】：
    1. 记录入场时的参考价格
    2. 持续监控当前价格与参考价格的偏离度
    3. 当价格偏离超过止损阈值时触发出场信号
    4. 止损阈值在整个持仓期间保持不变
    
    【追踪止损模式 (trailing=True)】：
    1. 记录入场后的最有利价格(最高价或最低价)
    2. 根据最有利价格动态调整止损线
    3. 当价格回撤超过止损阈值时触发出场信号
    4. 止损线只向有利方向移动，不会回撤
    
    参数说明：
        ts (array): 时间序列价格数据，通常是收盘价或其他代表价格
        stop (float/array): 止损阈值
            - 正值：表示止损幅度，如0.05表示5%止损
            - 负值：表示价格绝对值，如-95表示价格跌至95时止损
        trailing (bool/array): 是否启用追踪止损
            - False: 固定止损模式
            - True: 追踪止损模式
    
    应用场景：
    - **风险控制**：为交易策略添加基础的风险保护机制
    - **资金保护**：防止单笔交易损失过大
    - **趋势跟随**：通过追踪止损锁定趋势利润
    - **止盈策略**：结合止盈逻辑构建完整的风险管理体系
    
    使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        import numpy as np

        # 创建示例数据
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(50).cumsum())
        entries = pd.Series([True] + [False] * 49)  # 第一天入场

        # 示例1：固定5%止损
        fixed_stop = vbt.STX.run(entries, prices, stop=0.05, trailing=False)
        print("固定止损信号:")
        print(fixed_stop.exits[fixed_stop.exits].index.tolist())

        # 示例2：5%追踪止损  
        trailing_stop = vbt.STX.run(entries, prices, stop=0.05, trailing=True)
        print("追踪止损信号:")
        print(trailing_stop.exits[trailing_stop.exits].index.tolist())

        # 示例3：多种止损阈值对比
        multi_stop = vbt.STX.run(
            entries, prices, 
            stop=[0.03, 0.05, 0.10],  # 3%, 5%, 10%止损
            trailing=False
        )
        print("不同止损阈值的触发时间:")
        for col in multi_stop.exits.columns:
            trigger_day = multi_stop.exits[col][multi_stop.exits[col]].index
            if len(trigger_day) > 0:
                print(f"  {col}: 第{trigger_day[0]}天触发")
            else:
                print(f"  {col}: 未触发")

        # 示例4：固定止损vs追踪止损对比
        comparison = vbt.STX.run(
            entries, prices,
            stop=0.08,
            trailing=[False, True]  # 同时测试两种模式
        )
        print("止损模式对比:")
        for col in comparison.exits.columns:
            trigger_day = comparison.exits[col][comparison.exits[col]].index
            if len(trigger_day) > 0:
                trigger_price = prices.iloc[trigger_day[0]]
                print(f"  {col}: 第{trigger_day[0]}天触发, 价格{trigger_price:.2f}")
        ```
    
    技术特点：
    - 使用高效的Numba编译函数进行价格比较
    - 支持向量化处理多个资产的止损计算
    - 自动处理价格序列的边界条件
    - 与portfolio模块完美集成
    
    止损策略选择：
    - **固定止损**：适用于震荡市场，防止频繁触发
    - **追踪止损**：适用于趋势市场，能够锁定更多利润
    - **组合使用**：可以结合两种模式的优势
    
    注意事项：
    - 止损阈值设置需要考虑市场波动性
    - 过小的止损阈值可能导致频繁触发
    - 追踪止损在震荡市场中可能表现不佳
    - 止损信号一旦触发，当前持仓周期结束
    """
    pass


setattr(STX, '__doc__', _STX.__doc__)  # 将文档字符串绑定到STX类

# 链式止损出场信号生成器
# 功能：基于止损条件生成链式出场信号，支持连续止损策略
# 应用：多级止损策略、复合风险管理、动态止损调整
STCX = SignalFactory(
    **stx_config.merge_with(     # 基于stx配置，并合并链式模式参数
        dict(
            class_name='STCX',   # 类名：链式止损出场信号生成器，CX表示Chain eXit
            short_name='stcx',   # 简短名称：stcx表示止损链式出场
            mode='chain'         # 信号模式：链式模式，生成new_entries和exits
        )
    )
).from_choice_func(
    **stx_func_config            # 使用相同的止损函数配置
)


class _STCX(STCX):
    """
    链式止损出场信号生成器类
    
    功能概述：
    生成基于止损条件的链式出场信号，能够在原有入场信号基础上产生新的入场信号
    和对应的止损出场信号。这种链式机制特别适用于需要连续风险管理的复杂交易策略。
    
    核心特性：
    - 链式信号处理模式(Chain mode)
    - 基于价格阈值的专业止损逻辑
    - 生成new_entries和exits两个输出
    - 支持连续的风险管理策略
    
    信号生成逻辑：
    1. 接收原始的入场信号(entries)作为输入
    2. 基于价格数据和止损参数为每个入场信号生成止损出场信号
    3. 同时生成新的入场信号(new_entries)用于链式处理
    4. 实现连续的风险管理和信号传递
    
    链式止损优势：
    - 支持多级止损策略的实现
    - 适用于复杂的风险管理体系构建
    - 便于与其他风险控制模块组合
    - 提供动态止损调整能力
    
    应用场景：
    - **分级止损策略**：实现多层次的止损保护
    - **动态风险管理**：根据市场情况调整止损策略
    - **复合止损系统**：组合多种止损条件
    - **风险控制流水线**：构建完整的风险管理流程
    
    参数继承：
    继承STX的完整止损参数配置能力，包括：
    - stop: 止损阈值设置
    - trailing: 追踪止损开关
    - ts: 价格时间序列数据
    
    技术特点：
    - 完全兼容vectorbt的链式处理架构
    - 支持复杂的止损策略组合
    - 高效的链式止损计算
    - 与其他链式生成器协调工作
    
    典型应用流程：
    1. 第一级：基础入场信号 -> STCX -> 第一级止损保护
    2. 第二级：new_entries -> 其他策略模块 -> 第二级风险控制
    3. 多级组合：构建完整的风险管理体系
    
    注意事项：
    - 链式处理需要合理设计信号流
    - 多级止损可能增加策略复杂度
    - 需要平衡风险控制与策略收益
    - 建议进行充分的回测验证
    """
    pass


setattr(STCX, '__doc__', _STCX.__doc__)  # 将文档字符串绑定到STCX类

# ========================================
# OHLC高级止损信号生成器族 (OHLCST Series)  
# ========================================
# 
# OHLCST系列是最高级的止损信号生成器，特点是：
# - 基于完整OHLC数据的精确止损计算
# - 支持止损(SL)、追踪止损(TSL)、止盈(TP)的复合策略
# - 提供止损类型和触发价格的详细信息
# - 适用于专业的风险管理和量化交易系统

# OHLCSTX生成器的基础配置对象
# 功能：定义基于OHLC数据的高级止损出场信号生成器结构
ohlcstx_config = Config(
    dict(
        class_name='OHLCSTX',        # 类名：OHLC止损出场信号生成器
        module_name=__name__,        # 模块名：当前文件
        short_name='ohlcstx',        # 简短名称：ohlcstx表示OHLC止损出场
        mode='exits',                # 信号模式：仅生成出场信号
        input_names=['open', 'high', 'low', 'close'],  # 输入名称：完整的OHLC数据
        in_output_names=['stop_price', 'stop_type'],   # 就地输出：止损触发价格和类型
        param_names=['sl_stop', 'sl_trail', 'tp_stop', 'reverse'],  # 参数名称列表
        attr_settings=dict(          # 属性设置
            stop_type=dict(dtype=StopType)  # 止损类型使用StopType枚举，自动创建可读版本
        )
    )
)
"""OHLCSTX生成器的工厂配置对象 - 定义基于完整K线数据的高级止损信号生成器"""

# OHLCSTX生成器的函数配置对象
# 功能：定义基于OHLC数据的复杂止损计算逻辑
ohlcstx_func_config = Config(
    dict(
        exit_choice_func=ohlc_stop_choice_nb,    # 出场选择函数：基于OHLC的止损选择
        exit_settings=dict(          # 出场设置配置
            pass_inputs=['open', 'high', 'low', 'close'],  # 传递完整OHLC数据（不传递entries）
            pass_in_outputs=['stop_price', 'stop_type'],   # 传递输出数组用于记录止损信息
            pass_params=['sl_stop', 'sl_trail', 'tp_stop', 'reverse'],  # 传递所有止损参数
            pass_kwargs=[('is_open_safe', True), 'wait', 'pick_first', 'temp_idx_arr', 'flex_2d'],  # 传递辅助参数
        ),
        pass_flex_2d=True,           # 启用灵活2D处理
        in_output_settings=dict(     # 就地输出设置
            stop_price=dict(         # 止损价格输出配置
                dtype=np.float64     # 价格数据类型为64位浮点数
            ),
            stop_type=dict(          # 止损类型输出配置
                dtype=np.int64       # 类型数据为64位整数
            )
        ),
        param_settings=dict(         # 参数设置配置
            sl_stop=flex_elem_param_config,     # 止损阈值的灵活配置
            sl_trail=flex_elem_param_config,    # 追踪止损标志的灵活配置
            tp_stop=flex_elem_param_config,     # 止盈阈值的灵活配置
            reverse=flex_elem_param_config      # 反向交易标志的灵活配置
        ),
        # 参数默认值设置
        sl_stop=np.nan,              # 默认止损：不设置（NaN表示禁用）
        sl_trail=False,              # 默认追踪止损：关闭
        tp_stop=np.nan,              # 默认止盈：不设置（NaN表示禁用）
        reverse=False,               # 默认反向：关闭（False表示多头交易）
        stop_price=np.nan,           # 默认止损价格：未触发
        stop_type=-1                 # 默认止损类型：无类型
    )
)
"""OHLCSTX生成器的函数配置对象 - 定义基于OHLC数据的复杂止损逻辑和参数"""

# OHLC高级止损出场信号生成器
# 功能：基于完整K线数据生成专业的止损出场信号
# 应用：专业交易系统、高精度风险管理、复合止损策略
OHLCSTX = SignalFactory(
    **ohlcstx_config             # 展开基础配置参数
).from_choice_func(
    **ohlcstx_func_config        # 展开函数配置参数
)


def _bind_ohlcstx_plot(base_cls: type, entries_attr: str) -> tp.Callable:  # pragma: no cover
    """
    OHLC止损信号生成器的绘图函数绑定器
    
    功能概述：
    为OHLCST系列信号生成器动态绑定专业的绘图函数，该函数能够同时显示
    OHLC价格数据和对应的入场出场信号，提供直观的可视化分析能力。
    
    核心特性：
    - 动态函数绑定：根据不同的生成器类动态创建绘图函数
    - 专业OHLC图表：支持K线图和OHLC线图两种显示模式
    - 信号标记显示：在价格图上标记入场和出场信号位置
    - 止损信息展示：显示止损触发价格和止损类型
    
    参数说明：
        base_cls (type): 基础类，通常是OHLCSTX或OHLCSTCX
        entries_attr (str): 入场信号属性名，用于确定显示哪个信号数组
            - 'entries': 用于OHLCSTX，显示原始入场信号
            - 'new_entries': 用于OHLCSTCX，显示链式入场信号
    
    返回值：
        tp.Callable: 绑定后的绘图函数，具有完整的OHLC可视化功能
    
    技术实现：
    - 使用闭包技术捕获基础类的绘图方法
    - 动态创建新的绘图函数并绑定到目标类
    - 支持Plotly交互式图表的所有功能
    - 自动处理多列数据的可视化需求
    
    应用场景：
    - **策略可视化**：直观展示止损策略的执行过程
    - **信号分析**：分析入场出场信号的时机和效果
    - **风险监控**：监控止损触发情况和风险控制效果
    - **策略优化**：通过可视化对比不同参数的效果
    """

    base_cls_plot = base_cls.plot  # 获取基础类的绘图方法

    def plot(self,
             plot_type: tp.Union[None, str, tp.BaseTraceType] = None,  # 图表类型：None/OHLC/Candlestick或自定义
             ohlc_kwargs: tp.KwargsLike = None,                        # OHLC图表的样式参数
             entry_trace_kwargs: tp.KwargsLike = None,                 # 入场信号标记的样式参数
             exit_trace_kwargs: tp.KwargsLike = None,                  # 出场信号标记的样式参数
             add_trace_kwargs: tp.KwargsLike = None,                   # 添加图层的通用参数
             fig: tp.Optional[tp.BaseFigure] = None,                   # 可选的现有图表对象
             _base_cls_plot: tp.Callable = base_cls_plot,              # 基础绘图函数（内部使用）
             **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover   # 布局参数和其他关键字参数
        from vectorbt._settings import settings  # 导入vectorbt全局设置
        ohlcv_cfg = settings['ohlcv']            # 获取OHLCV相关配置
        plotting_cfg = settings['plotting']      # 获取绘图相关配置

        # 检查数据维度：只支持单列数据的可视化
        if self.wrapper.ndim > 1:
            raise TypeError("Select a column first. Use indexing.")

        # 初始化默认参数
        if ohlc_kwargs is None:
            ohlc_kwargs = {}                     # OHLC图表样式参数默认空字典
        if add_trace_kwargs is None:
            add_trace_kwargs = {}                # 添加图层参数默认空字典

        # 创建或使用现有图表对象
        if fig is None:
            fig = make_figure()                  # 创建新的图表对象
            fig.update_layout(                   # 设置默认布局
                showlegend=True,                 # 显示图例
                xaxis_rangeslider_visible=False, # 隐藏x轴范围滑块
                xaxis_showgrid=True,             # 显示x轴网格
                yaxis_showgrid=True              # 显示y轴网格
            )
        fig.update_layout(**layout_kwargs)       # 应用用户自定义布局参数

        # 确定图表类型和绘图对象
        if plot_type is None:
            plot_type = ohlcv_cfg['plot_type']   # 使用默认图表类型
        if isinstance(plot_type, str):
            if plot_type.lower() == 'ohlc':      # OHLC线图模式
                plot_type = 'OHLC'
                plot_obj = go.Ohlc               # 使用Plotly的OHLC对象
            elif plot_type.lower() == 'candlestick':  # K线图模式
                plot_type = 'Candlestick'
                plot_obj = go.Candlestick        # 使用Plotly的Candlestick对象
            else:
                raise ValueError("Plot type can be either 'OHLC' or 'Candlestick'")
        else:
            plot_obj = plot_type                 # 使用用户自定义的绘图对象
        
        # 创建OHLC价格图表
        ohlc = plot_obj(
            x=self.wrapper.index,                # x轴：时间索引
            open=self.open,                      # 开盘价数据
            high=self.high,                      # 最高价数据
            low=self.low,                        # 最低价数据
            close=self.close,                    # 收盘价数据
            name=plot_type,                      # 图表名称
            increasing=dict(                     # 上涨K线的样式
                line=dict(
                    color=plotting_cfg['color_schema']['increasing']  # 使用配置的上涨颜色
                )
            ),
            decreasing=dict(                     # 下跌K线的样式
                line=dict(
                    color=plotting_cfg['color_schema']['decreasing']  # 使用配置的下跌颜色
                )
            )
        )
        ohlc.update(**ohlc_kwargs)               # 应用用户自定义的OHLC样式
        fig.add_trace(ohlc, **add_trace_kwargs)  # 将OHLC图表添加到主图表

        # 绘制入场和出场信号标记
        _base_cls_plot(                          # 调用基础绘图函数绘制信号标记
            self,
            entry_y=self.open,                   # 入场信号标记在开盘价位置
            exit_y=self.stop_price,              # 出场信号标记在止损价格位置
            exit_types=self.stop_type_readable,  # 出场信号类型（可读格式）
            entry_trace_kwargs=entry_trace_kwargs,  # 入场标记样式参数
            exit_trace_kwargs=exit_trace_kwargs,    # 出场标记样式参数
            add_trace_kwargs=add_trace_kwargs,      # 添加图层参数
            fig=fig                              # 目标图表对象
        )
        return fig                               # 返回完整的图表对象

    plot.__doc__ = """绘制OHLC图表、`{0}.{1}`入场信号和`{0}.exits`出场信号。
    
    功能说明：
    创建专业的OHLC价格图表，并在其上叠加显示入场信号标记和出场信号标记。
    支持K线图和OHLC线图两种显示模式，提供完整的交易信号可视化分析。
    
    参数说明：
        plot_type: 图表类型，可选'OHLC'、'Candlestick'或自定义Plotly图表对象
        ohlc_kwargs (dict): 传递给OHLC图表的关键字参数，用于自定义样式
        entry_trace_kwargs (dict): 传递给入场信号标记的关键字参数，用于自定义入场标记样式
        exit_trace_kwargs (dict): 传递给出场信号标记的关键字参数，用于自定义出场标记样式
        fig (Figure or FigureWidget): 可选的现有图表对象，用于添加新的图层
        **layout_kwargs: 传递给图表布局的关键字参数，用于自定义整体布局
    
    图表特性：
    - 支持OHLC线图和K线图两种显示模式
    - 自动应用vectorbt的颜色主题配置
    - 入场信号标记在开盘价位置
    - 出场信号标记在止损触发价格位置
    - 显示止损类型信息（StopLoss、TrailStop、TakeProfit等）
    - 支持交互式缩放、平移和悬停显示
    
    使用场景：
    - **策略回测可视化**：直观展示止损策略的执行过程
    - **信号时机分析**：分析入场出场信号的时机和效果
    - **风险控制监控**：监控止损触发情况和风险控制效果
    - **参数优化对比**：通过可视化对比不同止损参数的效果
    
    技术特点：
    - 基于Plotly构建，支持交互式操作
    - 自动处理多列数据的单列显示
    - 集成vectorbt的全局配置系统
    - 支持自定义样式和布局参数""".format(base_cls.__name__, entries_attr)

    if entries_attr == 'entries':
        plot.__doc__ += """
    使用示例：
        ```python
        # 基础使用：绘制OHLC图表和信号标记
        >>> ohlcstx.iloc[:, 0].plot()
        
        # 自定义样式：使用K线图模式
        >>> ohlcstx.iloc[:, 0].plot(plot_type='Candlestick')
        
        # 自定义标记样式
        >>> ohlcstx.iloc[:, 0].plot(
        ...     entry_trace_kwargs=dict(marker=dict(color='green', size=8)),
        ...     exit_trace_kwargs=dict(marker=dict(color='red', size=8))
        ... )
        
        # 自定义布局
        >>> ohlcstx.iloc[:, 0].plot(
        ...     title="止损策略可视化",
        ...     xaxis_title="时间",
        ...     yaxis_title="价格"
        ... )
        ```
        
        ![](/assets/images/OHLCSTX.svg)
    """
    return plot  # 返回绑定后的绘图函数


class _OHLCSTX(OHLCSTX):
    """
    OHLC高级止损出场信号生成器类
    
    功能概述：
    基于完整的OHLC（开盘价、最高价、最低价、收盘价）数据生成专业的止损出场信号，
    是vectorbt框架中最高级的风险管理工具。该生成器支持多种止损策略的组合使用，
    包括固定止损、追踪止损和止盈，能够提供精确的风险控制和利润保护。
    
    核心特性：
    - 基于完整K线数据的精确止损计算
    - 支持止损(SL)、追踪止损(TSL)、止盈(TP)的复合策略
    - 提供止损触发价格和止损类型的详细信息
    - 内置专业的可视化功能，支持OHLC图表和信号标记
    
    算法逻辑：
    【固定止损 (sl_stop)】：
    1. 记录入场时的参考价格（通常是开盘价）
    2. 监控价格下跌幅度，当跌幅超过sl_stop时触发止损
    3. 止损价格 = 参考价格 × (1 - sl_stop)
    
    【追踪止损 (sl_trail=True)】：
    1. 记录入场后的最高价格作为追踪基准
    2. 动态调整止损线：止损价格 = 最高价 × (1 - sl_stop)
    3. 当价格回撤超过sl_stop时触发止损
    
    【止盈 (tp_stop)】：
    1. 监控价格上涨幅度，当涨幅超过tp_stop时触发止盈
    2. 止盈价格 = 参考价格 × (1 + tp_stop)
    
    参数说明：
        entries (array): 入场信号数组，True表示入场点
        open (array): 开盘价时间序列
        high (array): 最高价时间序列  
        low (array): 最低价时间序列
        close (array): 收盘价时间序列
        sl_stop (float/array): 止损阈值，如0.05表示5%止损
        sl_trail (bool/array): 是否启用追踪止损
        tp_stop (float/array): 止盈阈值，如0.10表示10%止盈
        reverse (bool/array): 是否反向交易（做空）
    
    输出属性：
        exits (array): 出场信号数组，True表示出场点
        stop_price (array): 止损触发价格，显示具体的触发价格
        stop_type_readable (array): 可读的止损类型
            - StopLoss: 固定止损触发
            - TrailStop: 追踪止损触发  
            - TakeProfit: 止盈触发
            - None: 未触发任何止损
    
    应用场景：
    - **专业交易系统**：为机构级交易系统提供精确的风险控制
    - **高频交易策略**：基于实时OHLC数据的快速止损决策
    - **多策略组合**：支持复杂的多策略风险管理系统
    - **策略回测验证**：提供完整的止损策略回测能力
    
    使用示例：
        ```python
        # 示例1：测试不同止损类型
        >>> import vectorbt as vbt
        >>> import pandas as pd
        >>> import numpy as np

        >>> # 创建示例数据
        >>> entries = pd.Series([True, False, False, False, False, False])
        >>> price = pd.DataFrame({
        ...     'open': [10, 11, 12, 11, 10, 9],
        ...     'high': [11, 12, 13, 12, 11, 10],
        ...     'low': [9, 10, 11, 10, 9, 8],
        ...     'close': [10, 11, 12, 11, 10, 9]
        ... })
        
        >>> # 运行OHLC止损生成器
        >>> ohlcstx = vbt.OHLCSTX.run(
        ...     entries,
        ...     price['open'], price['high'], price['low'], price['close'],
        ...     sl_stop=[0.1, 0.1, np.nan],      # 10%固定止损，10%固定止损，无止损
        ...     sl_trail=[False, True, False],   # 固定止损，追踪止损，固定止损
        ...     tp_stop=[np.nan, np.nan, 0.1])   # 无止盈，无止盈，10%止盈

        >>> # 查看入场信号
        >>> print("入场信号:")
        >>> print(ohlcstx.entries)
        ohlcstx_sl_stop     0.1    0.1    NaN
        ohlcstx_sl_trail  False   True  False
        ohlcstx_tp_stop     NaN    NaN    0.1
        0                  True   True   True
        1                 False  False  False
        2                 False  False  False
        3                 False  False  False
        4                 False  False  False
        5                 False  False  False

        >>> # 查看出场信号
        >>> print("出场信号:")
        >>> print(ohlcstx.exits)
        ohlcstx_sl_stop     0.1    0.1    NaN
        ohlcstx_sl_trail  False   True  False
        ohlcstx_tp_stop     NaN    NaN    0.1
        0                 False  False  False
        1                 False  False   True    # 第1列：第1天止盈触发
        2                 False  False  False
        3                 False   True  False    # 第2列：第3天追踪止损触发
        4                  True  False  False    # 第1列：第4天固定止损触发
        5                 False  False  False

        >>> # 查看止损触发价格
        >>> print("止损触发价格:")
        >>> print(ohlcstx.stop_price)
        ohlcstx_sl_stop     0.1    0.1    NaN
        ohlcstx_sl_trail  False   True  False
        ohlcstx_tp_stop     NaN    NaN    0.1
        0                   NaN    NaN    NaN
        1                   NaN    NaN   11.0   # 止盈价格：10 * 1.1 = 11.0
        2                   NaN    NaN    NaN
        3                   NaN   11.7    NaN   # 追踪止损价格：13 * 0.9 = 11.7
        4                   9.0    NaN    NaN   # 固定止损价格：10 * 0.9 = 9.0
        5                   NaN    NaN    NaN

        >>> # 查看止损类型
        >>> print("止损类型:")
        >>> print(ohlcstx.stop_type_readable)
        ohlcstx_sl_stop        0.1        0.1         NaN
        ohlcstx_sl_trail     False       True       False
        ohlcstx_tp_stop        NaN        NaN         0.1
        0                     None       None        None
        1                     None       None  TakeProfit
        2                     None       None        None
        3                     None  TrailStop        None
        4                 StopLoss       None        None
        5                     None       None        None
        
        # 可视化分析
        >>> ohlcstx.iloc[:, 0].plot(title="固定止损策略可视化")
        >>> ohlcstx.iloc[:, 1].plot(title="追踪止损策略可视化")  
        >>> ohlcstx.iloc[:, 2].plot(title="止盈策略可视化")
        ```
    
    技术特点：
    - 使用Numba编译的高性能计算函数
    - 支持向量化处理多个资产和参数组合
    - 提供完整的止损信息记录和追踪
    - 集成专业的可视化分析功能
    
    与其他止损生成器的区别：
    - 相比STX：基于完整OHLC数据，提供更精确的止损计算
    - 相比其他生成器：专门针对风险管理设计，功能最全面
    - 支持复合止损策略，可同时使用多种止损方式
    
    注意事项：
    - 止损阈值设置需要考虑市场波动性和交易成本
    - 追踪止损在震荡市场中可能频繁触发
    - 止盈设置过高可能错过早期获利机会
    - 建议结合其他技术指标进行参数优化
    """

    plot = _bind_ohlcstx_plot(OHLCSTX, 'entries')  # 绑定专业的OHLC绘图函数


setattr(OHLCSTX, '__doc__', _OHLCSTX.__doc__)  # 将文档字符串绑定到OHLCSTX类
setattr(OHLCSTX, 'plot', _OHLCSTX.plot)         # 将绘图函数绑定到OHLCSTX类

# OHLC链式止损出场信号生成器
# 功能：基于OHLC数据生成链式止损出场信号，支持连续的风险管理策略
# 应用：多级风险管理、复合止损系统、动态风险控制
OHLCSTCX = SignalFactory(
    **ohlcstx_config.merge_with(     # 基于ohlcstx配置，并合并链式模式参数
        dict(
            class_name='OHLCSTCX',   # 类名：OHLC链式止损出场信号生成器，CX表示Chain eXit
            short_name='ohlcstcx',   # 简短名称：ohlcstcx表示OHLC止损链式出场
            mode='chain'             # 信号模式：链式模式，生成new_entries和exits
        )
    )
).from_choice_func(
    **ohlcstx_func_config            # 使用相同的OHLC止损函数配置
)


class _OHLCSTCX(OHLCSTCX):
    """
    OHLC链式止损出场信号生成器类
    
    功能概述：
    生成基于OHLC数据的链式止损出场信号，能够在原有入场信号基础上产生新的
    入场信号和对应的止损出场信号。这种链式机制特别适用于需要连续风险管理的
    复杂交易策略，是构建多层次风险控制系统的核心工具。
    
    核心特性：
    - 链式信号处理模式(Chain mode)
    - 基于完整OHLC数据的专业止损逻辑
    - 生成new_entries和exits两个输出
    - 支持连续的风险管理策略和信号传递
    
    信号生成逻辑：
    1. 接收原始的入场信号(entries)作为输入
    2. 基于OHLC数据和止损参数为每个入场信号生成止损出场信号
    3. 同时生成新的入场信号(new_entries)用于链式处理
    4. 实现连续的风险管理和信号传递机制
    
    链式止损优势：
    - 支持多级止损策略的实现和组合
    - 适用于复杂的风险管理体系构建
    - 便于与其他风险控制模块组合使用
    - 提供动态止损调整和策略切换能力
    
    应用场景：
    - **分级风险管理**：实现多层次的风险控制体系
    - **动态止损策略**：根据市场情况动态调整止损策略
    - **复合止损系统**：组合多种止损条件和策略
    - **风险控制流水线**：构建完整的风险管理流程
    
    参数继承：
    继承OHLCSTX的完整参数配置能力，包括：
    - sl_stop: 止损阈值设置
    - sl_trail: 追踪止损开关
    - tp_stop: 止盈阈值设置
    - reverse: 反向交易标志
    - open/high/low/close: 完整的OHLC数据
    
    技术特点：
    - 完全兼容vectorbt的链式处理架构
    - 支持复杂的OHLC止损策略组合
    - 高效的链式止损计算和信号传递
    - 与其他链式生成器协调工作
    
    典型应用流程：
    1. 第一级：基础入场信号 -> OHLCSTCX -> 第一级OHLC止损保护
    2. 第二级：new_entries -> 其他策略模块 -> 第二级风险控制
    3. 多级组合：构建完整的OHLC风险管理体系
    
    可视化支持：
    - 继承OHLCSTX的专业可视化功能
    - 支持OHLC图表和信号标记的完整显示
    - 提供链式信号的直观分析能力
    
    注意事项：
    - 链式处理需要合理设计信号流和风险控制逻辑
    - 多级止损可能增加策略复杂度和计算开销
    - 需要平衡风险控制与策略收益的关系
    - 建议进行充分的回测验证和参数优化
    """

    plot = _bind_ohlcstx_plot(OHLCSTCX, 'new_entries')  # 绑定专业的OHLC绘图函数，显示链式入场信号


setattr(OHLCSTCX, '__doc__', _OHLCSTCX.__doc__)  # 将文档字符串绑定到OHLCSTCX类
setattr(OHLCSTCX, 'plot', _OHLCSTCX.plot)         # 将绘图函数绑定到OHLCSTCX类
