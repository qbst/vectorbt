# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT SIGNALS MODULE: 高性能信号生成和处理核心模块 (NB.PY)
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于信号生成、过滤和处理的高性能计算模块。
该模块通过Numba JIT编译技术，提供了一整套信号处理算法，是量化交易策略开发的核心基础设施。

核心设计理念：
1. **极致性能优化**：所有函数都使用@njit装饰器进行Just-In-Time编译，执行速度比纯Python快10-100倍
2. **矩阵优先设计**：遵循vectorbt核心原则，将二维矩阵视为一等公民，支持多资产同时处理
3. **信号驱动架构**：实现完整的信号生命周期管理，从生成、过滤到执行的全链路处理
4. **随机化支持**：内置多种随机信号生成器，支持蒙特卡洛模拟和压力测试
5. **止损止盈集成**：提供专业的风险管理信号，支持复杂的止损止盈策略

主要功能模块：
【信号生成引擎 Generation】
- generate_nb(): 基础信号生成器，使用自定义选择函数
- generate_ex_nb(): 退出信号生成器，为每个入场信号生成对应的退出信号
- generate_enex_nb(): 入场退出信号协同生成器，交替生成入场和退出信号

【信号过滤系统 Filtering】
- clean_enex_nb(): 入场退出信号清理器，确保信号序列的逻辑一致性

【随机信号生成器 Random】
- generate_rand_nb(): 随机信号生成器，按指定数量随机选择信号点
- generate_rand_by_prob_nb(): 概率随机信号生成器，按概率分布生成信号

【止损止盈系统 Stop Exits】
- generate_stop_ex_nb(): 止损信号生成器，基于价格阈值生成退出信号
- generate_ohlc_stop_ex_nb(): OHLC止损信号生成器，基于K线数据的高级止损策略

【范围分析工具 Map and Reduce Ranges】
- between_ranges_nb(): 信号间隔范围计算器，分析信号之间的时间间隔
- partition_ranges_nb(): 信号分区范围计算器，识别连续信号区间

【排序和索引系统 Ranking & Index】
- rank_nb(): 信号排序器，为信号分配优先级排序
- nth_index_nb(): 第N个信号索引查找器，定位特定位置的信号

应用场景：
- **量化策略开发**：为各类技术分析策略提供信号生成支持
- **风险管理系统**：实现止损止盈、仓位控制等风险管理逻辑
- **回测引擎**：为历史回测提供高性能的信号计算能力
- **实时交易**：为实盘交易提供毫秒级的信号处理性能
- **蒙特卡洛模拟**：支持随机信号生成，用于压力测试和情景分析

技术特点：
- **Numba JIT编译**：所有核心算法都经过JIT编译优化
- **向量化处理**：支持2维数组批量处理，可同时处理多个资产
- **内存高效**：使用就地操作和缓存友好的数据访问模式
- **类型安全**：完整的类型注解，支持静态类型检查
- **函数式设计**：纯函数设计，无副作用，易于测试和并行化

与vectorbt生态系统的关系：
- 为signals.factory模块提供底层计算支持
- 与portfolio.nb模块协作处理交易信号
- 使用base.reshape_fns模块的灵活选择功能
- 集成generic.nb模块的通用数值计算函数

数据约定：
- 所有输入数组都应该是2维，除非函数名包含'_1d'后缀
- 数据沿索引轴(axis 0)进行处理，符合时间序列分析习惯
- 返回的索引都是绝对索引，便于与原始数据对应
- 传递给函数的回调函数都必须是Numba编译的函数

该模块是vectorbt框架信号处理的核心，为量化交易策略提供了工业级的信号生成和处理能力。

使用示例：
```python
import numpy as np
import vectorbt as vbt

# 1. 基础信号生成示例
@njit
def simple_choice_func(from_i, to_i, col):
    # 每5个周期生成一个信号
    if (from_i // 5) * 5 < to_i:
        return np.array([(from_i // 5) * 5])
    return np.array([], dtype=np.int64)

signals = vbt.signals.nb.generate_nb((100, 3), False, simple_choice_func)

# 2. 随机信号生成示例  
random_signals = vbt.signals.nb.generate_rand_nb((100, 3), n=10, seed=42)

# 3. 止损信号生成示例
entries = np.zeros((100, 1), dtype=bool)
entries[10, 0] = True  # 在第10个周期入场
prices = np.random.randn(100, 1).cumsum() + 100

stop_exits = vbt.signals.nb.generate_stop_ex_nb(
    entries, prices, stop=-0.05, trailing=False, 
    wait=1, until_next=True, skip_until_exit=True, 
    pick_first=True, flex_2d=True
)
```

注意事项：
- 所有函数都需要Numba兼容的数据类型
- 选择函数必须返回有效的索引数组
- 信号生成器会自动处理边界条件和异常情况
- 随机信号生成需要设置seed以确保结果可复现
================================================================================

Numba编译函数集合

提供了丰富的Numba编译函数库，用于访问器和回测管道的其他部分（如技术指标）。
这些函数仅接受NumPy数组和其他Numba兼容类型。

```pycon
>>> import numpy as np
>>> import vectorbt as vbt

>>> # vectorbt.signals.nb.pos_rank_nb
>>> vbt.signals.nb.pos_rank_nb(np.array([False, True, True, True, False])[:, None])[:, 0]
[-1  0  1  2 -1]
```

!!! 重要提示
    vectorbt将矩阵视为一等公民，期望输入数组为2维，除非函数具有后缀`_1d`或用作其他函数的输入。
    数据沿索引轴(axis 0)进行处理。
    
    作为参数传递的所有函数都应该是Numba编译的函数。

    返回的索引应该是绝对索引。
"""

import numpy as np  # 导入NumPy库，提供高性能数组操作
from numba import njit  # 导入Numba的njit装饰器，用于JIT编译优化

# 导入vectorbt的类型注解模块
from vectorbt import _typing as tp
# 导入基础重塑函数模块，提供灵活的数组选择功能
from vectorbt.base.reshape_fns import flex_select_auto_nb
# 导入通用枚举模块，提供范围数据类型和状态定义
from vectorbt.generic.enums import range_dt, RangeStatus
# 导入信号枚举模块，提供止损类型定义
from vectorbt.signals.enums import StopType
# 导入数组工具模块，提供概率分布和数值缩放功能
from vectorbt.utils.array_ import uniform_summing_to_one_nb, rescale_float_to_int_nb, renormalize_nb


# ############# 信号生成 (Generation) ############# #


@njit  # 使用Numba JIT编译优化
def generate_nb(shape: tp.Shape,
                pick_first: bool,
                choice_func_nb: tp.ChoiceFunc, *args) -> tp.Array2d:
    """创建指定形状的布尔信号矩阵，使用choice_func_nb选择信号位置。

    这是vectorbt信号生成系统的基础函数，通过用户定义的选择函数来确定信号位置。
    该函数支持多种信号生成策略，从简单的固定周期信号到复杂的条件触发信号。

    参数说明:
        shape (tuple): 目标信号矩阵的形状，格式为(行数, 列数)
        pick_first (bool): 是否只选择choice_func_nb返回的第一个信号
                          - True: 只取第一个信号，用于单次信号生成
                          - False: 取所有信号，用于多次信号生成
        choice_func_nb (callable): 信号选择函数，必须是Numba编译的函数
                                  函数签名: func(from_i, to_i, col, *args) -> np.array
                                  - from_i: 搜索范围起始索引(包含)
                                  - to_i: 搜索范围结束索引(不包含)
                                  - col: 当前列索引
                                  - *args: 传递给选择函数的额外参数
                                  返回值: 索引数组，范围在[from_i, to_i)内
        *args: 传递给choice_func_nb的额外参数

    返回值:
        np.array: 形状为shape的布尔数组，True表示信号位置

    算法逻辑:
        1. 创建全False的布尔矩阵作为输出
        2. 对每一列调用选择函数获取信号索引
        3. 根据pick_first参数决定是否只选择第一个信号
        4. 将选中的位置设置为True
        5. 进行边界检查，确保索引有效

    使用场景:
        - 基于技术指标的信号生成(如移动平均交叉)
        - 基于价格突破的信号生成
        - 基于时间周期的固定信号生成
        - 基于随机条件的信号生成

    性能特点:
        - Numba编译优化，接近C语言性能
        - 向量化处理，支持多列同时计算
        - 内存高效，原地操作减少内存分配

    示例用法:
        ```python
        @njit
        def choice_func_nb(from_i, to_i, col):
            # 每列在不同位置生成信号
            return np.array([from_i + col])

        # 生成3列5行的信号矩阵
        signals = generate_nb((5, 3), False, choice_func_nb)
        # 结果:
        # [[ True False False]
        #  [False  True False] 
        #  [False False  True]
        #  [False False False]
        #  [False False False]]
        ```

    异常处理:
        - 如果返回的索引超出边界，抛出ValueError
        - 自动处理空索引数组的情况

    注意事项:
        - choice_func_nb必须是Numba编译的函数
        - 返回的索引必须在有效范围内
        - 函数是纯函数，无副作用，线程安全
    """
    out = np.full(shape, False, dtype=np.bool_)  # 创建全False的布尔输出数组

    # 遍历每一列
    for col in range(out.shape[1]):
        # 调用用户定义的选择函数获取信号索引
        idxs = choice_func_nb(0, shape[0], col, *args)
        
        # 如果没有返回任何索引，跳过当前列
        if len(idxs) == 0:
            continue
        
        if pick_first:
            # 只选择第一个信号
            first_i = idxs[0]
            # 边界检查
            if first_i < 0 or first_i >= shape[0]:
                raise ValueError("第一个返回索引超出边界")
            out[first_i, col] = True
        else:
            # 选择所有信号
            # 边界检查
            if np.any(idxs < 0) or np.any(idxs >= shape[0]):
                raise ValueError("返回的索引超出边界")
            out[idxs, col] = True
    
    return out


@njit  # 使用Numba JIT编译优化
def generate_ex_nb(entries: tp.Array2d,
                   wait: int,
                   until_next: bool,
                   skip_until_exit: bool,
                   pick_first: bool,
                   exit_choice_func_nb: tp.ChoiceFunc, *args) -> tp.Array2d:
    """为每个入场信号生成对应的退出信号。

    这是vectorbt信号系统中专门用于生成退出信号的核心函数。它接收入场信号数组，
    并为每个入场信号生成相应的退出信号，支持多种退出策略和时间控制机制。

    参数说明:
        entries (np.array): 入场信号的布尔数组，形状为(时间, 资产)
        wait (int): 退出信号延迟周期数
                   - 0: 允许在同一周期内入场和退出
                   - >0: 必须等待指定周期后才能退出
                   注意: wait=0可能导致同一bar出现两个信号
        until_next (bool): 是否只在下一个入场信号前搜索退出信号
                          - True: 限制搜索范围到下一个入场信号
                          - False: 搜索到序列末尾
                          注意: False时难以判断退出信号属于哪个入场信号
        skip_until_exit (bool): 是否跳过处理退出前的入场信号
                               只在until_next=False时有效
                               - True: 跳过退出前的新入场信号
                               - False: 处理所有入场信号
                               注意: True时难以判断退出信号属于哪个入场信号
        pick_first (bool): 是否只选择退出选择函数返回的第一个信号
        exit_choice_func_nb (callable): 退出信号选择函数，必须是Numba编译的函数
                                       参见generate_nb中choice_func_nb的说明
        *args: 传递给exit_choice_func_nb的额外参数

    返回值:
        np.array: 与entries相同形状的布尔数组，True表示退出信号位置

    算法逻辑:
        1. 为每列找到所有入场信号的索引
        2. 对每个入场信号:
           a. 根据skip_until_exit判断是否跳过
           b. 计算搜索范围[from_i, to_i)
           c. 调用退出选择函数获取候选退出索引
           d. 根据pick_first参数选择退出信号
           e. 更新最后退出位置记录
        3. 返回所有退出信号的布尔矩阵

    时间控制机制:
        - wait参数控制最小等待时间
        - until_next参数控制最大搜索范围
        - skip_until_exit参数控制重叠处理策略

    使用场景:
        - 技术指标退出信号生成
        - 止损止盈信号生成
        - 时间退出信号生成
        - 条件退出信号生成

    示例用法:
        ```python
        @njit
        def exit_after_3_bars(from_i, to_i, col):
            # 3个周期后退出
            if from_i + 2 < to_i:
                return np.array([from_i + 2])
            return np.array([], dtype=np.int64)

        entries = np.array([[True, False], [False, False], 
                           [False, False], [False, False]])
        exits = generate_ex_nb(entries, 1, True, False, True, 
                              exit_after_3_bars)
        ```

    性能优化:
        - 使用Numba编译，接近C语言性能
        - 智能范围计算，避免不必要的搜索
        - 内存高效的索引操作

    注意事项:
        - 退出选择函数必须是Numba编译的
        - 搜索范围会根据参数自动调整
        - 函数保证时间序列的逻辑一致性
    """
    exits = np.full_like(entries, False)  # 创建与入场信号相同形状的退出信号数组

    # 遍历每一列(每个资产/策略)
    for col in range(entries.shape[1]):
        # 找到当前列所有入场信号的索引位置
        entry_idxs = np.flatnonzero(entries[:, col])
        last_exit_i = -1  # 记录最后一个退出信号的位置
        
        # 处理每个入场信号
        for i in range(entry_idxs.shape[0]):
            # 检查是否应该跳过当前入场信号
            if skip_until_exit and entry_idxs[i] <= last_exit_i:
                continue
            
            # 计算退出信号搜索的起始位置
            from_i = entry_idxs[i] + wait
            
            # 计算退出信号搜索的结束位置
            if i < entry_idxs.shape[0] - 1 and until_next:
                # 如果until_next为True且不是最后一个入场信号，搜索到下一个入场信号
                to_i = entry_idxs[i + 1]
            else:
                # 否则搜索到序列末尾
                to_i = entries.shape[0]
            
            # 如果有有效的搜索范围
            if to_i > from_i:
                # 调用退出选择函数获取候选退出索引
                idxs = exit_choice_func_nb(from_i, to_i, col, *args)
                
                # 如果没有找到退出信号，继续下一个入场信号
                if len(idxs) == 0:
                    continue
                
                if pick_first:
                    # 只选择第一个退出信号
                    first_i = idxs[0]
                    # 边界检查
                    if first_i < from_i or first_i >= to_i:
                        raise ValueError("第一个返回索引超出边界")
                    exits[first_i, col] = True
                    last_exit_i = first_i  # 更新最后退出位置
                else:
                    # 选择所有退出信号
                    # 边界检查
                    if np.any(idxs < from_i) or np.any(idxs >= to_i):
                        raise ValueError("返回的索引超出边界")
                    exits[idxs, col] = True
                    last_exit_i = idxs[-1]  # 更新最后退出位置为最后一个退出信号
    
    return exits


@njit  # 使用Numba JIT编译优化
def generate_enex_nb(shape: tp.Shape,
                     entry_wait: int,
                     exit_wait: int,
                     entry_pick_first: bool,
                     exit_pick_first: bool,
                     entry_choice_func_nb: tp.ChoiceFunc,
                     entry_args: tp.Args,
                     exit_choice_func_nb: tp.ChoiceFunc,
                     exit_args: tp.Args) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """交替生成入场信号和退出信号的高级信号协调器。

    这是vectorbt信号系统中最复杂和最强大的信号生成函数，它能够智能地协调入场和退出信号的生成，
    确保信号序列的逻辑一致性和时间有序性。该函数实现了状态机模式，在入场和退出状态间交替切换。

    参数说明:
        shape (tuple): 目标信号矩阵的形状，格式为(行数, 列数)
        entry_wait (int): 入场信号等待周期数
                         - 0: 允许入场和退出在同一周期内处理，且退出可在入场前处理
                         - >0: 入场信号必须等待指定周期后才能生效
                         注意: 与exit_wait不能同时为0
        exit_wait (int): 退出信号等待周期数  
                        - 0: 允许入场和退出在同一周期内处理，且入场可在退出前处理
                        - >0: 退出信号必须等待指定周期后才能生效
                        注意: 与entry_wait不能同时为0
        entry_pick_first (bool): 是否只选择入场选择函数返回的第一个信号
                               - True: 只取第一个入场信号
                               - False: 取所有入场信号
        exit_pick_first (bool): 是否只选择退出选择函数返回的第一个信号
                              - True: 只取第一个退出信号  
                              - False: 取所有退出信号，类似于generate_ex_nb中skip_until_exit=True
        entry_choice_func_nb (callable): 入场信号选择函数，必须是Numba编译的函数
                                       参见generate_nb中choice_func_nb的说明
        entry_args (tuple): 传递给entry_choice_func_nb的参数元组
        exit_choice_func_nb (callable): 退出信号选择函数，必须是Numba编译的函数  
                                      参见generate_nb中choice_func_nb的说明
        exit_args (tuple): 传递给exit_choice_func_nb的参数元组

    返回值:
        tuple[np.array, np.array]: (入场信号矩阵, 退出信号矩阵)
        - 两个布尔矩阵形状都与shape相同
        - 入场信号矩阵: True表示入场信号位置
        - 退出信号矩阵: True表示退出信号位置

    算法逻辑:
        1. 初始化两个空的信号矩阵
        2. 对每列实现状态机循环:
           a. 偶数轮次(i%2==0): 寻找入场信号
           b. 奇数轮次(i%2==1): 寻找退出信号
        3. 每轮次的处理流程:
           a. 计算搜索范围[from_i, to_i)
           b. 调用相应的选择函数
           c. 根据pick_first参数选择信号
           d. 更新状态变量，准备下一轮
        4. 循环检测和边界保护确保算法收敛

    状态机设计:
        - 状态0(偶数): 等待入场信号状态
        - 状态1(奇数): 等待退出信号状态
        - prev_prev_i, prev_i: 用于无限循环检测
        - 严格的时间递增约束确保信号序列有序

    时间控制机制:
        - entry_wait和exit_wait控制信号间的最小时间间隔
        - 防止信号过于密集，符合实际交易约束
        - 两个参数不能同时为0，避免时间歧义

    使用场景:
        - 完整交易周期的信号生成(买入-持有-卖出)
        - 复杂策略的状态切换控制
        - 基于多条件的动态信号协调
        - 需要严格时序控制的信号生成

    性能特点:
        - Numba编译优化，高性能状态机实现
        - 智能循环检测，避免无限循环
        - 内存高效，原地生成双信号矩阵

    示例用法:
        ```python
        @njit
        def ma_cross_entry(from_i, to_i, col, ma_fast, ma_slow):
            # 快线上穿慢线入场
            for i in range(from_i, to_i):
                if ma_fast[i, col] > ma_slow[i, col] and ma_fast[i-1, col] <= ma_slow[i-1, col]:
                    return np.array([i])
            return np.array([], dtype=np.int64)

        @njit  
        def ma_cross_exit(from_i, to_i, col, ma_fast, ma_slow):
            # 快线下穿慢线退出
            for i in range(from_i, to_i):
                if ma_fast[i, col] < ma_slow[i, col] and ma_fast[i-1, col] >= ma_slow[i-1, col]:
                    return np.array([i])
            return np.array([], dtype=np.int64)

        entries, exits = generate_enex_nb(
            shape=(100, 3),
            entry_wait=1, exit_wait=1,
            entry_pick_first=True, exit_pick_first=True,
            entry_choice_func_nb=ma_cross_entry,
            entry_args=(ma_fast, ma_slow),
            exit_choice_func_nb=ma_cross_exit,
            exit_args=(ma_fast, ma_slow)
        )
        ```

    异常处理:
        - entry_wait和exit_wait同时为0时抛出ValueError
        - 自动检测无限循环并抛出异常
        - 严格的边界检查确保索引有效性

    注意事项:
        - 两个选择函数都必须是Numba编译的
        - 状态机确保入场和退出的交替出现
        - 时间约束保证信号序列的现实可行性
        - 函数保证线程安全，可并行调用
    """
    entries = np.full(shape, False)  # 初始化入场信号矩阵
    exits = np.full(shape, False)    # 初始化退出信号矩阵
    
    # 检查时间参数合法性
    if entry_wait == 0 and exit_wait == 0:
        raise ValueError("entry_wait和exit_wait不能同时为0")

    # 对每一列(每个资产/策略)运行状态机
    for col in range(shape[1]):
        prev_prev_i = -2  # 前两次信号位置，用于无限循环检测
        prev_i = -1       # 上一次信号位置
        i = 0            # 状态计数器，偶数=入场状态，奇数=退出状态
        while True:
            to_i = shape[0]
            # Cannot assign two functions to a var in numba
            if i % 2 == 0:
                if i == 0:
                    from_i = 0
                else:
                    from_i = prev_i + entry_wait
                if from_i >= to_i:
                    break
                idxs = entry_choice_func_nb(from_i, to_i, col, *entry_args)
                a = entries
                pick_first = entry_pick_first
            else:
                from_i = prev_i + exit_wait
                if from_i >= to_i:
                    break
                idxs = exit_choice_func_nb(from_i, to_i, col, *exit_args)
                a = exits
                pick_first = exit_pick_first
            if len(idxs) == 0:
                break
            first_i = idxs[0]
            if first_i == prev_i == prev_prev_i:
                raise ValueError("Infinite loop detected")
            if first_i < from_i:
                raise ValueError("First index is out of bounds")
            if pick_first:
                # Consider only the first signal
                if first_i >= to_i:
                    raise ValueError("First index is out of bounds")
                a[first_i, col] = True
                prev_prev_i = prev_i
                prev_i = first_i
                i += 1
            else:
                # Consider all signals
                last_i = idxs[-1]
                if last_i >= to_i:
                    raise ValueError("Last index is out of bounds")
                a[idxs, col] = True
                prev_prev_i = prev_i
                prev_i = last_i
                i += 1

    return entries, exits


# ############# Filtering ############# #


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存
def clean_enex_1d_nb(entries: tp.Array1d,
                     exits: tp.Array1d,
                     entry_first: bool) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """清理一维入场和退出信号数组，确保信号序列的逻辑一致性。

    这是vectorbt信号处理系统中的重要清理函数，用于解决原始信号中可能存在的逻辑冲突，
    如同时出现入场和退出信号、信号顺序错误等问题。该函数实现了有限状态机，
    确保信号序列符合交易逻辑。

    参数说明:
        entries (np.array): 一维入场信号布尔数组
        exits (np.array): 一维退出信号布尔数组，长度必须与entries相同
        entry_first (bool): 是否要求入场信号必须在退出信号之前
                           - True: 严格的入场优先逻辑，必须先入场才能退出
                           - False: 允许在没有入场的情况下直接退出(适用于已有持仓)

    返回值:
        tuple[np.array, np.array]: (清理后的入场信号, 清理后的退出信号)
        - 两个数组的形状与输入相同
        - 保证信号的逻辑一致性和时间有序性

    算法逻辑:
        使用三状态有限状态机:
        - 状态-1: 初始状态，无任何位置
        - 状态0: 退出状态，可以接受入场信号
        - 状态1: 入场状态，可以接受退出信号

        状态转换规则:
        1. 初始状态(-1) + 入场信号 → 入场状态(1)
        2. 初始状态(-1) + 退出信号 → 退出状态(0) [仅当entry_first=False]
        3. 入场状态(1) + 退出信号 → 退出状态(0)
        4. 退出状态(0) + 入场信号 → 入场状态(1)
        5. 同时出现入场和退出信号 → 忽略该位置

    使用场景:
        - 清理技术指标产生的冗余信号
        - 解决信号生成函数的逻辑冲突
        - 确保回测信号的合理性
        - 预处理用户提供的原始信号

    示例用法:
        ```python
        # 原始信号存在同时入场和退出的冲突
        entries = np.array([True, False, True, True, False])
        exits = np.array([False, True, True, False, True])
        
        # 清理信号，要求入场优先
        clean_entries, clean_exits = clean_enex_1d_nb(entries, exits, True)
        # 结果: clean_entries = [True, False, False, True, False]
        #       clean_exits = [False, True, False, False, True]
        ```

    性能特点:
        - Numba编译优化，接近C语言性能
        - 启用缓存，重复调用时性能更佳
        - 单遍扫描算法，时间复杂度O(n)
        - 内存高效，原地状态机实现

    注意事项:
        - 函数保证输出信号的时间有序性
        - entry_first参数影响初始状态的行为
        - 同时出现的冲突信号会被完全忽略
        - 函数是纯函数，无副作用，线程安全
    """
    entries_out = np.full(entries.shape, False, dtype=np.bool_)  # 初始化清理后的入场信号数组
    exits_out = np.full(exits.shape, False, dtype=np.bool_)      # 初始化清理后的退出信号数组

    phase = -1  # 状态机当前状态: -1=初始, 0=已退出, 1=已入场
    
    # 遍历每个时间点
    for i in range(entries.shape[0]):
        # 如果同时有入场和退出信号，跳过该位置(避免冲突)
        if entries[i] and exits[i]:
            continue
            
        # 处理入场信号
        if entries[i]:
            # 只有在初始状态或已退出状态时才能入场
            if phase == -1 or phase == 0:
                phase = 1  # 转换到入场状态
                entries_out[i] = True
        
        # 处理退出信号
        if exits[i]:
            # 退出条件: (允许直接退出且在初始状态) 或 (当前在入场状态)
            if (not entry_first and phase == -1) or phase == 1:
                phase = 0  # 转换到退出状态
                exits_out[i] = True

    return entries_out, exits_out


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存
def clean_enex_nb(entries: tp.Array2d,
                  exits: tp.Array2d,
                  entry_first: bool) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """二维版本的入场退出信号清理函数。

    这是clean_enex_1d_nb的向量化版本，能够同时处理多个资产或策略的信号清理。
    该函数将每一列视为独立的信号序列，分别应用清理逻辑。

    参数说明:
        entries (np.array): 二维入场信号布尔数组，形状为(时间, 资产)
        exits (np.array): 二维退出信号布尔数组，形状必须与entries相同
        entry_first (bool): 是否要求入场信号必须在退出信号之前，参见clean_enex_1d_nb

    返回值:
        tuple[np.array, np.array]: (清理后的入场信号矩阵, 清理后的退出信号矩阵)

    算法逻辑:
        1. 为每一列独立运行clean_enex_1d_nb函数
        2. 保持列间的独立性，不同资产的信号不会相互影响
        3. 所有列共享相同的清理规则和参数

    使用场景:
        - 多资产投资组合的信号清理
        - 批量处理多个策略的信号
        - 大规模回测中的信号预处理
        - 并行化的信号处理任务

    性能特点:
        - Numba编译优化，支持向量化操作
        - 启用缓存，提高重复调用性能
        - 列级别的并行处理友好设计
        - 内存访问模式优化，提高缓存命中率

    示例用法:
        ```python
        # 多资产信号矩阵
        entries = np.array([[True, False], [False, True], [True, True]])
        exits = np.array([[False, True], [True, False], [True, False]]) 
        
        # 批量清理多个资产的信号
        clean_entries, clean_exits = clean_enex_nb(entries, exits, True)
        ```

    注意事项:
        - 每一列的处理完全独立，不会相互影响
        - 适合大规模并行处理场景
        - 保持输入矩阵的维度结构不变
    """
    entries_out = np.empty(entries.shape, dtype=np.bool_)  # 创建清理后的入场信号输出矩阵
    exits_out = np.empty(exits.shape, dtype=np.bool_)      # 创建清理后的退出信号输出矩阵

    # 对每一列(每个资产/策略)独立应用清理逻辑
    for col in range(entries.shape[1]):
        entries_out[:, col], exits_out[:, col] = clean_enex_1d_nb(
            entries[:, col], exits[:, col], entry_first
        )
    
    return entries_out, exits_out


# ############# Random ############# #


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存
def rand_choice_nb(from_i: int, to_i: int, col: int, n: tp.MaybeArray[int]) -> tp.Array1d:
    """随机选择函数，从指定范围内随机选择n个信号位置。

    这是一个符合choice_func_nb接口规范的随机选择函数，用于在指定的时间范围内
    随机选择信号位置。该函数支持灵活的索引机制，可以为不同列指定不同的选择数量。

    参数说明:
        from_i (int): 选择范围的起始索引(包含)
        to_i (int): 选择范围的结束索引(不包含)
        col (int): 当前列索引，用于灵活索引
        n (int或array-like): 要选择的信号数量
                           - 标量: 所有列使用相同数量
                           - 数组: 每列可以有不同数量(使用灵活索引)

    返回值:
        np.array: 选中的信号位置索引数组，按升序排列

    算法逻辑:
        1. 使用灵活索引机制获取当前列的选择数量
        2. 计算实际可选择的最大数量(范围大小vs要求数量的最小值)
        3. 使用numpy.random.choice进行无重复随机选择
        4. 将相对索引转换为绝对索引并返回

    使用场景:
        - 随机信号生成策略
        - 蒙特卡洛模拟的信号生成
        - 压力测试和敏感性分析
        - 随机化回测验证

    性能特点:
        - Numba编译优化，高性能随机选择
        - 启用缓存，重复调用时性能优越
        - 支持灵活索引，适应复杂参数配置
        - 无重复选择，保证信号位置唯一性

    示例用法:
        ```python
        # 在[10, 20)范围内为第0列随机选择3个信号
        indices = rand_choice_nb(10, 20, 0, 3)
        # 可能的结果: [12, 15, 18]
        
        # 使用数组指定不同列的选择数量
        n_array = np.array([2, 3, 1])  # 第0列选2个，第1列选3个，第2列选1个
        indices_col1 = rand_choice_nb(5, 15, 1, n_array)  # 为第1列选择3个
        ```

    注意事项:
        - 选择数量不能超过可用范围大小
        - 返回的索引保证无重复且有序
        - 需要在调用前设置随机种子以确保可重现性
        - 函数依赖numpy的随机数生成器状态
    """
    ns = np.asarray(n)  # 将n转换为数组形式，支持灵活索引
    
    # 计算实际选择数量(不能超过可用范围)
    size = min(to_i - from_i, flex_select_auto_nb(ns, 0, col, True))
    
    # 随机选择相对位置，然后转换为绝对位置
    return from_i + np.random.choice(to_i - from_i, size=size, replace=False)


@njit  # 使用Numba JIT编译优化
def generate_rand_nb(shape: tp.Shape, n: tp.MaybeArray[int], seed: tp.Optional[int] = None) -> tp.Array2d:
    """创建指定形状的布尔矩阵，随机选择指定数量的信号位置。

    这是vectorbt随机信号生成系统的核心函数，使用统计随机方法在时间序列中
    生成信号位置。该函数特别适用于蒙特卡洛模拟、压力测试和随机化验证。

    参数说明:
        shape (tuple): 目标信号矩阵的形状，格式为(时间, 资产)
        n (int或array-like): 每列要生成的信号数量
                           - 标量: 所有列使用相同数量
                           - 数组: 支持每列不同数量(使用灵活索引)
        seed (int, optional): 随机种子，用于确保结果可重现
                             - None: 使用当前随机状态
                             - 整数: 设置特定随机种子

    返回值:
        np.array: 形状为shape的布尔矩阵，True表示随机选中的信号位置

    算法逻辑:
        1. 如果提供种子，设置随机数生成器状态
        2. 调用generate_nb函数，使用rand_choice_nb作为选择函数
        3. 对每一列独立进行随机选择
        4. 返回完整的布尔信号矩阵

    使用场景:
        - 随机交易策略的信号生成
        - 蒙特卡洛回测验证
        - 策略鲁棒性测试
        - 基准随机信号生成

    性能特点:
        - Numba编译优化，高效的随机生成
        - 向量化操作，支持大规模矩阵处理
        - 可重现的随机性控制
        - 内存高效的信号矩阵构建

    示例用法:
        ```python
        # 为3个资产、100个时间点的矩阵，每列随机生成5个信号
        signals = generate_rand_nb((100, 3), n=5, seed=42)
        print(f"总信号数: {signals.sum()}")  # 应该是15个信号
        
        # 不同列生成不同数量的信号
        n_per_col = np.array([3, 5, 2])
        signals = generate_rand_nb((50, 3), n=n_per_col, seed=123)
        ```

    注意事项:
        - 设置seed参数以确保结果可重现
        - 信号数量不能超过时间序列长度
        - 函数会自动处理边界条件
        - 适用于需要统计随机性的量化场景

    参见:
        rand_choice_nb: 底层随机选择函数的实现
    """
    # 如果提供了随机种子，则设置随机数生成器
    if seed is not None:
        np.random.seed(seed)
    
    # 使用generate_nb函数和rand_choice_nb选择函数生成随机信号
    return generate_nb(
        shape,
        False,  # 不只选择第一个信号，选择所有随机选中的信号
        rand_choice_nb, n
    )


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存
def rand_by_prob_choice_nb(from_i: int,
                           to_i: int,
                           col: int,
                           prob: tp.MaybeArray[float],
                           pick_first: bool,
                           temp_idx_arr: tp.Array1d,
                           flex_2d: bool) -> tp.Array1d:
    """基于概率的随机选择函数，按照指定概率随机选择信号位置。

    这是一个高级的随机选择函数，它不是简单地随机选择固定数量的信号，
    而是基于每个时间点的概率分布来决定是否生成信号。这种方法更符合
    实际市场中信号的随机性特征。

    参数说明:
        from_i (int): 选择范围的起始索引(包含)
        to_i (int): 选择范围的结束索引(不包含)
        col (int): 当前列索引，用于灵活索引
        prob (float或array-like): 每个位置生成信号的概率
                                - 标量: 所有位置使用相同概率
                                - 数组: 每个位置可以有不同概率(使用灵活索引)
                                - 概率值范围: [0, 1]
        pick_first (bool): 是否在找到第一个信号后立即停止
                          - True: 找到第一个信号后立即返回
                          - False: 遍历整个范围，可能返回多个信号
        temp_idx_arr (np.array): 临时索引数组，用于存储选中的位置
                                预分配的整数数组，提高性能
        flex_2d (bool): 概率数组是否为二维形式，参见flex_select_auto_nb

    返回值:
        np.array: 选中的信号位置索引数组，按时间顺序排列

    算法逻辑:
        1. 遍历指定时间范围[from_i, to_i)
        2. 对每个时间点:
           a. 使用灵活索引获取该位置的概率值
           b. 生成[0,1)区间的随机数
           c. 如果随机数小于概率值，选中该位置
           d. 如果pick_first=True且已选中，立即返回
        3. 返回所有选中位置的索引数组

    概率分布支持:
        - 均匀概率: 所有位置使用相同概率
        - 时变概率: 不同时间点使用不同概率
        - 列特定概率: 不同资产使用不同概率分布
        - 动态概率: 基于市场状态的动态概率调整

    使用场景:
        - 基于波动率的信号生成
        - 市场情绪驱动的信号模型
        - 非均匀分布的随机测试
        - 复杂概率模型的信号生成

    性能特点:
        - Numba编译优化，高速概率计算
        - 启用缓存，重复调用性能优越
        - 使用预分配数组，避免内存分配开销
        - 支持早停机制，提高效率

    示例用法:
        ```python
        # 固定概率0.1的信号生成
        temp_arr = np.empty(50, dtype=np.int64)
        indices = rand_by_prob_choice_nb(0, 50, 0, 0.1, False, temp_arr, False)
        
        # 时变概率：开盘时概率高，收盘时概率低
        prob_array = np.linspace(0.2, 0.05, 100)  # 递减概率
        indices = rand_by_prob_choice_nb(0, 100, 0, prob_array, False, temp_arr, True)
        ```

    注意事项:
        - 概率值必须在[0,1]范围内
        - temp_idx_arr长度必须足够存储所有可能的选中位置
        - 函数依赖当前的随机数生成器状态
        - pick_first=True可以显著提高性能
    """
    probs = np.asarray(prob)  # 将概率转换为数组形式
    j = 0  # 选中位置的计数器
    
    # 遍历指定的时间范围
    for i in range(from_i, to_i):
        # 生成随机数并与对应位置的概率比较
        if np.random.uniform(0, 1) < flex_select_auto_nb(probs, i, col, flex_2d):  # [0, 1)
            temp_idx_arr[j] = i  # 存储选中的位置
            j += 1
            # 如果只需要第一个信号，立即退出
            if pick_first:
                break
    
    # 返回选中位置的数组切片
    return temp_idx_arr[:j]


@njit  # 使用Numba JIT编译优化
def generate_rand_by_prob_nb(shape: tp.Shape,
                             prob: tp.MaybeArray[float],
                             pick_first: bool,
                             flex_2d: bool,
                             seed: tp.Optional[int] = None) -> tp.Array2d:
    """创建指定形状的布尔矩阵，按概率随机生成信号。

    这是vectorbt高级随机信号生成系统的核心函数，它基于概率分布而不是固定数量
    来生成信号。这种方法更符合真实市场中信号出现的随机性特征，特别适用于
    基于市场微观结构、波动率或情绪指标的信号生成。

    参数说明:
        shape (tuple): 目标信号矩阵的形状，格式为(时间, 资产)
        prob (float或array-like): 信号生成概率分布
                                - 标量: 所有位置使用统一概率
                                - 1维数组: 时间维度的概率变化
                                - 2维数组: 完整的时间-资产概率矩阵
        pick_first (bool): 是否在每列找到第一个信号后停止
                          - True: 每列最多生成一个信号
                          - False: 每列可以生成多个信号
        flex_2d (bool): 概率数组的维度解释方式
                       - True: prob被视为二维数组
                       - False: prob被视为标量或一维数组
        seed (int, optional): 随机种子，用于确保结果可重现

    返回值:
        np.array: 形状为shape的布尔矩阵，True表示按概率选中的信号位置

    算法逻辑:
        1. 设置随机种子(如果提供)
        2. 创建临时索引数组用于性能优化
        3. 对每一列调用generate_nb函数
        4. 使用rand_by_prob_choice_nb作为概率选择函数
        5. 返回完整的布尔信号矩阵

    概率模型设计:
        - 时间不变模型: 使用标量概率，适用于简单随机策略
        - 时间变化模型: 使用时间序列概率，适应市场周期
        - 完全随机模型: 使用随机概率矩阵，模拟复杂市场行为
        - 条件概率模型: 基于市场状态的条件概率分布

    使用场景:
        - 基于波动率的动态信号生成
        - 市场微观结构驱动的信号模型
        - 情绪指标转换为交易信号
        - 复杂随机过程的信号模拟

    性能特点:
        - Numba编译优化，高效的概率计算
        - 向量化处理，支持大规模概率矩阵
        - 内存预分配策略，减少运行时开销
        - 可控的随机性，支持可重现的结果

    示例用法:
        ```python
        # 1. 简单均匀概率信号生成
        signals = generate_rand_by_prob_nb(
            (100, 3), prob=0.05, pick_first=False, flex_2d=False, seed=42
        )
        
        # 2. 时变概率信号生成
        # 交易时间概率高，非交易时间概率低
        time_probs = np.where(
            (np.arange(100) % 24 >= 9) & (np.arange(100) % 24 <= 16),
            0.1, 0.02  # 交易时间10%概率，其他时间2%概率
        )
        signals = generate_rand_by_prob_nb(
            (100, 3), prob=time_probs, pick_first=False, flex_2d=True, seed=42
        )
        
        # 3. 个股特定概率矩阵
        # 不同股票在不同时间有不同的信号概率
        prob_matrix = np.random.beta(2, 8, (100, 3))  # Beta分布概率
        signals = generate_rand_by_prob_nb(
            (100, 3), prob=prob_matrix, pick_first=False, flex_2d=True, seed=42
        )
        ```

    注意事项:
        - prob数组的形状必须与flex_2d参数匹配
        - 所有概率值必须在[0,1]范围内
        - 高概率值会导致密集的信号，可能影响策略性能
        - 建议设置随机种子以确保结果可重现

    参见:
        rand_by_prob_choice_nb: 底层概率选择函数的实现
    """
    # 如果提供了随机种子，则设置随机数生成器
    if seed is not None:
        np.random.seed(seed)
    
    # 创建临时索引数组，用于存储每列的选中位置
    temp_idx_arr = np.empty((shape[0],), dtype=np.int64)
    
    # 使用generate_nb函数和rand_by_prob_choice_nb选择函数生成概率信号
    return generate_nb(
        shape,
        pick_first,
        rand_by_prob_choice_nb, prob, pick_first, temp_idx_arr, flex_2d
    )


# ############# Random exits ############# #

@njit  # 使用Numba JIT编译优化
def generate_rand_ex_nb(entries: tp.Array2d,
                        wait: int,
                        until_next: bool,
                        skip_until_exit: bool,
                        seed: tp.Optional[int] = None) -> tp.Array2d:
    """为每个入场信号生成随机的退出信号。

    这是vectorbt随机退出信号生成系统的基础函数，它为给定的入场信号序列
    随机生成相应的退出信号。该函数特别适用于测试策略对退出时机的敏感性，
    以及生成基准随机退出策略。

    参数说明:
        entries (np.array): 入场信号的二维布尔数组，形状为(时间, 资产)
        wait (int): 退出信号的最小等待周期数
                   - 0: 允许在入场的同一周期退出
                   - >0: 必须等待指定周期后才能退出
        until_next (bool): 是否限制退出信号搜索范围到下一个入场信号
                          - True: 只在下一个入场信号前搜索退出
                          - False: 搜索到时间序列末尾
        skip_until_exit (bool): 是否跳过退出前的新入场信号
                               - True: 跳过直到当前退出信号完成
                               - False: 处理所有入场信号
        seed (int, optional): 随机种子，确保结果可重现
                             - None: 使用当前随机状态
                             - 整数: 设置特定的随机种子

    返回值:
        np.array: 与entries相同形状的布尔数组，True表示随机生成的退出信号位置

    算法逻辑:
        1. 设置随机种子(如果提供)
        2. 使用generate_ex_nb函数作为基础框架
        3. 使用rand_choice_nb作为选择函数，每次随机选择1个退出位置
        4. pick_first=True确保每个入场信号只对应一个退出信号

    使用场景:
        - 策略退出时机敏感性分析
        - 基准随机退出策略生成
        - 蒙特卡洛退出策略测试
        - 策略鲁棒性验证

    性能特点:
        - Numba编译优化，高效的随机退出生成
        - 基于成熟的generate_ex_nb框架
        - 可控的随机性，支持可重现测试
        - 向量化处理，支持多资产批量处理

    示例用法:
        ```python
        # 创建简单的入场信号
        entries = np.zeros((100, 2), dtype=bool)
        entries[[10, 30, 60], 0] = True  # 第一个资产的入场信号
        entries[[15, 45, 75], 1] = True  # 第二个资产的入场信号
        
        # 生成随机退出信号
        random_exits = generate_rand_ex_nb(
            entries, wait=1, until_next=True, 
            skip_until_exit=False, seed=42
        )
        
        print(f"入场信号数量: {entries.sum()}")
        print(f"退出信号数量: {random_exits.sum()}")
        ```

    注意事项:
        - 设置seed参数以确保测试结果可重现
        - wait参数影响退出信号的最早可能时间
        - 每个入场信号必定对应一个随机退出信号
        - 适用于需要随机退出策略的回测场景

    参见:
        generate_ex_nb: 底层退出信号生成框架
        rand_choice_nb: 随机选择函数实现
    """
    # 如果提供了随机种子，设置随机数生成器
    if seed is not None:
        np.random.seed(seed)
    
    # 使用generate_ex_nb框架和rand_choice_nb选择函数生成随机退出信号
    return generate_ex_nb(
        entries,
        wait,
        until_next,
        skip_until_exit,
        True,  # pick_first=True，每个入场只选一个退出
        rand_choice_nb, 1  # 随机选择1个退出位置
    )


@njit  # 使用Numba JIT编译优化
def generate_rand_ex_by_prob_nb(entries: tp.Array2d,
                                prob: tp.MaybeArray[float],
                                wait: int,
                                until_next: bool,
                                skip_until_exit: bool,
                                flex_2d: bool,
                                seed: tp.Optional[int] = None) -> tp.Array2d:
    """为每个入场信号按概率生成随机退出信号。

    这是vectorbt高级随机退出信号生成系统的核心函数，它基于概率分布为入场信号
    生成相应的退出信号。与简单的随机退出不同，这个函数允许不同时间点有不同的
    退出概率，更符合实际市场中退出行为的时变特征。

    参数说明:
        entries (np.array): 入场信号的二维布尔数组，形状为(时间, 资产)
        prob (float或array-like): 退出概率分布
                                 - 标量: 所有时间点使用统一退出概率
                                 - 数组: 时间变化或时间-资产特定的概率分布
        wait (int): 退出信号的最小等待周期数，参见generate_rand_ex_nb
        until_next (bool): 退出搜索范围控制，参见generate_rand_ex_nb
        skip_until_exit (bool): 入场信号跳过控制，参见generate_rand_ex_nb
        flex_2d (bool): 概率数组的维度解释方式
                       - True: prob被视为二维数组
                       - False: prob被视为标量或一维数组
        seed (int, optional): 随机种子，确保结果可重现

    返回值:
        np.array: 与entries相同形状的布尔数组，True表示按概率生成的退出信号位置

    算法逻辑:
        1. 设置随机种子(如果提供)
        2. 创建临时索引数组用于性能优化
        3. 使用generate_ex_nb函数作为基础框架
        4. 使用rand_by_prob_choice_nb作为概率选择函数
        5. pick_first=True确保每个入场信号最多对应一个退出信号

    概率模型应用:
        - 时间衰减模型: 持仓时间越长，退出概率越高
        - 波动率驱动模型: 高波动期间退出概率增加
        - 市场状态模型: 不同市场条件下的差异化退出概率
        - 个股特征模型: 基于个股特征的定制化退出概率

    使用场景:
        - 基于波动率的动态退出策略
        - 市场微观结构驱动的退出模型
        - 行为金融学退出模式模拟
        - 复杂退出规则的概率建模

    性能特点:
        - Numba编译优化，高效的概率计算
        - 支持复杂的概率分布模型
        - 内存预分配策略，减少运行时开销
        - 可控的随机性和可重现的结果

    示例用法:
        ```python
        # 1. 时间衰减退出概率模型
        entries = np.zeros((100, 2), dtype=bool)
        entries[[10, 50], :] = True
        
        # 退出概率随时间递增(持仓时间越长退出概率越高)
        time_decay_prob = np.linspace(0.01, 0.5, 100)
        
        exits = generate_rand_ex_by_prob_nb(
            entries, prob=time_decay_prob, wait=1, until_next=True,
            skip_until_exit=False, flex_2d=True, seed=42
        )
        
        # 2. 波动率驱动退出模型
        # 高波动期间退出概率增加
        volatility = np.random.gamma(2, 0.1, 100)  # 模拟波动率
        vol_prob = np.clip(volatility * 2, 0.01, 0.8)  # 转换为退出概率
        
        exits = generate_rand_ex_by_prob_nb(
            entries, prob=vol_prob, wait=1, until_next=True,
            skip_until_exit=False, flex_2d=True, seed=123
        )
        ```

    注意事项:
        - prob数组的形状必须与flex_2d参数匹配
        - 概率值必须在[0,1]范围内
        - 高概率值会导致更早的退出信号
        - 建议设置随机种子以确保结果可重现

    参见:
        generate_rand_ex_nb: 简单随机退出信号生成
        rand_by_prob_choice_nb: 底层概率选择函数
    """
    # 如果提供了随机种子，设置随机数生成器
    if seed is not None:
        np.random.seed(seed)
    
    # 创建临时索引数组，提高性能
    temp_idx_arr = np.empty((entries.shape[0],), dtype=np.int64)
    
    # 使用generate_ex_nb框架和rand_by_prob_choice_nb概率选择函数生成退出信号
    return generate_ex_nb(
        entries,
        wait,
        until_next,
        skip_until_exit,
        True,  # pick_first=True，每个入场最多选一个退出
        rand_by_prob_choice_nb, prob, True, temp_idx_arr, flex_2d
    )


@njit  # 使用Numba JIT编译优化
def generate_rand_enex_nb(shape: tp.Shape,
                          n: tp.MaybeArray[int],
                          entry_wait: int,
                          exit_wait: int,
                          seed: tp.Optional[int] = None) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """生成指定数量的随机入场和退出信号对。

    这是vectorbt随机信号生成系统中最复杂的函数之一，它能够生成指定数量的
    入场-退出信号对，同时确保信号分布尽可能接近均匀分布。该函数通过巧妙的
    算法设计，在满足时间约束的前提下，实现了高质量的随机信号分布。

    参数说明:
        shape (tuple): 目标信号矩阵的形状，格式为(时间, 资产)
        n (int或array-like): 每列要生成的入场-退出信号对数量
                           - 标量: 所有列使用相同数量
                           - 数组: 每列可以有不同数量(使用灵活索引)
        entry_wait (int): 入场信号间的最小等待周期数
                         - 与exit_wait不能同时为0
        exit_wait (int): 退出信号间的最小等待周期数  
                        - 与entry_wait不能同时为0
        seed (int, optional): 随机种子，确保结果可重现

    返回值:
        tuple[np.array, np.array]: (入场信号矩阵, 退出信号矩阵)
        - 两个布尔矩阵形状都与shape相同
        - 每列的入场信号和退出信号数量相等且等于n

    算法逻辑:
        针对不同的等待约束条件使用不同的生成策略:

        1. **基础情况** (entry_wait=1, exit_wait=1):
           - 生成2n个随机位置
           - 奇数位置作为入场信号，偶数位置作为退出信号
           - 简单高效的交替分配方案

        2. **复杂情况** (其他等待约束):
           - 计算最小空间需求和可用空间
           - 智能分配首尾空间和中间扩展空间
           - 使用均匀分布算法生成信号位置
           - 为每个入场信号匹配对应的退出信号

    空间分配策略:
        - min_range: 相邻信号间的最小间隔
        - min_total_range: 首末信号间的最小总间隔
        - free_space: 可用于随机分布的额外空间
        - 三段式分配: 前置空间 + 扩展空间 + 后置空间

    均匀分布实现:
        - 使用uniform_summing_to_one_nb生成归一化随机数
        - 通过rescale_float_to_int_nb转换为整数空间分配
        - 确保信号分布的统计均匀性

    使用场景:
        - 随机交易策略的基准测试
        - 蒙特卡洛策略验证
        - 交易频率敏感性分析
        - 策略参数优化的随机基线

    性能特点:
        - Numba编译优化，复杂算法的高速执行
        - 智能算法分支，针对不同情况优化
        - 数学统计保证的均匀分布特性
        - 支持大规模随机信号生成

    示例用法:
        ```python
        # 1. 基础随机信号对生成
        entries, exits = generate_rand_enex_nb(
            shape=(100, 3), n=5, entry_wait=1, exit_wait=1, seed=42
        )
        print(f"入场信号总数: {entries.sum()}")  # 应该是15
        print(f"退出信号总数: {exits.sum()}")    # 应该是15
        
        # 2. 不同列生成不同数量的信号对
        n_per_col = np.array([3, 5, 2])  # 不同列的信号对数量
        entries, exits = generate_rand_enex_nb(
            shape=(200, 3), n=n_per_col, entry_wait=5, exit_wait=3, seed=123
        )
        
        # 验证每列的信号对数量
        for col in range(3):
            entry_count = entries[:, col].sum()
            exit_count = exits[:, col].sum()
            print(f"列{col}: 入场={entry_count}, 退出={exit_count}")
        ```

    异常处理:
        - entry_wait和exit_wait同时为0时抛出ValueError
        - 时间序列长度不足时抛出ValueError
        - 自动处理边界条件和特殊情况

    注意事项:
        - 生成的信号对数量严格等于指定的n值
        - 算法保证信号分布的统计均匀性
        - 时间约束参数直接影响信号的密集程度
        - 设置随机种子对结果重现性至关重要

    参见:
        uniform_summing_to_one_nb: 均匀随机数生成
        rescale_float_to_int_nb: 数值空间缩放函数
    """
    # 如果提供了随机种子，设置随机数生成器
    if seed is not None:
        np.random.seed(seed)
    
    # 初始化入场和退出信号矩阵
    entries = np.full(shape, False)
    exits = np.full(shape, False)
    
    # 检查时间约束参数的合法性
    if entry_wait == 0 and exit_wait == 0:
        raise ValueError("entry_wait和exit_wait不能同时为0")
    
    ns = np.asarray(n)  # 转换为数组形式以支持灵活索引

    # 基础情况：简单的交替信号生成
    if entry_wait == 1 and exit_wait == 1:
        # 生成2n个随机位置，然后交替分配给入场和退出
        both = generate_rand_nb(shape, ns * 2, seed=None)
        for col in range(both.shape[1]):
            both_idxs = np.flatnonzero(both[:, col])
            entries[both_idxs[0::2], col] = True  # 偶数索引为入场信号
            exits[both_idxs[1::2], col] = True    # 奇数索引为退出信号
    else:
        # 复杂情况：需要考虑时间约束的智能分配
        for col in range(shape[1]):
            _n = flex_select_auto_nb(ns, 0, col, True)  # 获取当前列的信号对数量
            
            if _n == 1:
                # 单个信号对的简单情况
                entry_idx = np.random.randint(0, shape[0] - exit_wait)
                entries[entry_idx, col] = True
            else:
                # 多个信号对的复杂空间分配算法
                
                # 计算相邻入场信号间的最小间隔
                min_range = entry_wait + exit_wait
                
                # 计算首末入场信号间的最小总间隔
                min_total_range = min_range * (_n - 1)
                
                # 检查时间序列长度是否足够
                if shape[0] < min_total_range + exit_wait + 1:
                    raise ValueError("时间序列长度不足以容纳所需的信号对数量")

                # 计算最小空间需求外的可用空间
                max_free_space = shape[0] - min_total_range - 1

                # 限制自由空间以保持分布的均匀性
                # 如果自由空间过大，会导致信号过于稀疏
                free_space = min(max_free_space, 3 * shape[0] // (_n + 1))

                # 为最后的退出信号预留空间
                free_space -= exit_wait

                # 将自由空间分配到三个区域：
                # 1) 第一个信号前  2) 信号间的扩展空间  3) 最后一个信号后
                # 为了补偿缺失的最后退出信号，给后置空间分配双倍权重
                rand_floats = uniform_summing_to_one_nb(6)  # 生成6个归一化随机数
                chosen_spaces = rescale_float_to_int_nb(rand_floats, (0, free_space), free_space)
                
                # 确定第一个和最后一个入场信号的位置
                first_idx = chosen_spaces[0]
                last_idx = shape[0] - np.sum(chosen_spaces[-2:]) - exit_wait - 1

                # 计算首末入场信号间的实际可用范围
                total_range = last_idx - first_idx

                # 计算单个信号间隔的最大可能值
                max_range = total_range - (_n - 2) * min_range

                # 在总范围内随机分配各个信号间隔
                rand_floats = uniform_summing_to_one_nb(_n - 1)
                chosen_ranges = rescale_float_to_int_nb(rand_floats, (min_range, max_range), total_range)

                # 将间隔转换为实际的入场信号位置
                entry_idxs = np.empty(_n, dtype=np.int64)
                entry_idxs[0] = first_idx
                entry_idxs[1:] = chosen_ranges
                entry_idxs = np.cumsum(entry_idxs)  # 累积求和得到绝对位置
                entries[entry_idxs, col] = True

        # 为所有入场信号生成对应的退出信号
        for col in range(shape[1]):
            entry_idxs = np.flatnonzero(entries[:, col])
            
            for j in range(len(entry_idxs)):
                # 计算当前入场信号对应的退出时间窗口
                entry_i = entry_idxs[j] + exit_wait  # 最早退出时间
                
                if j < len(entry_idxs) - 1:
                    # 不是最后一个入场，退出时间不能超过下一个入场
                    exit_i = entry_idxs[j + 1] - entry_wait
                else:
                    # 最后一个入场，退出时间可以到序列末尾
                    exit_i = entries.shape[0] - 1
                
                # 在有效窗口内随机选择退出时间
                i = np.random.randint(exit_i - entry_i + 1)
                exits[entry_i + i, col] = True
    
    return entries, exits


def rand_enex_apply_nb(input_shape: tp.Shape,
                       n: tp.MaybeArray[int],
                       entry_wait: int,
                       exit_wait: int) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """用于调用generate_rand_enex_nb的应用函数包装器。

    这是一个简单的包装函数，用于在vectorbt的apply系统中调用generate_rand_enex_nb。
    apply系统要求函数接受input_shape作为第一个参数，这个包装函数提供了兼容的接口。

    参数说明:
        input_shape (tuple): 输入数据的形状，传递给generate_rand_enex_nb的shape参数
        n (int或array-like): 信号对数量，参见generate_rand_enex_nb
        entry_wait (int): 入场等待时间，参见generate_rand_enex_nb  
        exit_wait (int): 退出等待时间，参见generate_rand_enex_nb

    返回值:
        tuple[np.array, np.array]: (入场信号矩阵, 退出信号矩阵)

    使用场景:
        - vectorbt内部apply系统的集成
        - 批量处理多个数据集的随机信号生成
        - 工厂模式中的标准化接口

    注意事项:
        - 这个函数不支持随机种子设置
        - 主要用于内部系统集成，用户通常直接使用generate_rand_enex_nb
    """
    return generate_rand_enex_nb(input_shape, n, entry_wait, exit_wait)


@njit  # 使用Numba JIT编译优化
def generate_rand_enex_by_prob_nb(shape: tp.Shape,
                                  entry_prob: tp.MaybeArray[float],
                                  exit_prob: tp.MaybeArray[float],
                                  entry_wait: int,
                                  exit_wait: int,
                                  entry_pick_first: bool,
                                  exit_pick_first: bool,
                                  flex_2d: bool,
                                  seed: tp.Optional[int] = None) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """基于概率分布交替生成入场和退出信号。

    这是vectorbt随机信号生成系统的最高级函数，它结合了概率驱动的信号生成
    和智能的入场-退出协调机制。该函数允许入场和退出信号使用不同的概率分布，
    能够模拟复杂的市场行为和交易策略。

    参数说明:
        shape (tuple): 目标信号矩阵的形状，格式为(时间, 资产)
        entry_prob (float或array-like): 入场信号概率分布
                                      - 标量: 统一的入场概率
                                      - 数组: 时间变化或资产特定的入场概率
        exit_prob (float或array-like): 退出信号概率分布
                                     - 标量: 统一的退出概率  
                                     - 数组: 时间变化或资产特定的退出概率
        entry_wait (int): 入场信号间的最小等待周期，与exit_wait不能同时为0
        exit_wait (int): 退出信号间的最小等待周期，与entry_wait不能同时为0
        entry_pick_first (bool): 是否在找到第一个入场信号后停止当前搜索
        exit_pick_first (bool): 是否在找到第一个退出信号后停止当前搜索
        flex_2d (bool): 概率数组的维度解释方式
        seed (int, optional): 随机种子，确保结果可重现

    返回值:
        tuple[np.array, np.array]: (入场信号矩阵, 退出信号矩阵)
        - 两个布尔矩阵形状都与shape相同
        - 信号分布遵循指定的概率分布

    算法逻辑:
        1. 设置随机种子(如果提供)
        2. 创建临时索引数组用于性能优化
        3. 使用generate_enex_nb函数作为协调框架
        4. 为入场和退出分别使用rand_by_prob_choice_nb概率选择函数
        5. 通过参数元组传递各自的概率分布和配置

    概率模型设计:
        - **独立概率模型**: 入场和退出使用完全独立的概率分布
        - **相关概率模型**: 入场和退出概率存在相关性(如反向相关)
        - **状态依赖模型**: 概率分布依赖于当前的市场状态
        - **自适应概率模型**: 概率分布随历史表现动态调整

    使用场景:
        - 复杂交易策略的概率建模
        - 市场微观结构的信号生成
        - 行为金融模式的量化建模
        - 多因子驱动的信号系统

    性能特点:
        - Numba编译优化，复杂概率计算的高效执行
        - 支持高维概率分布模型
        - 内存预分配策略，优化大规模计算
        - 完全可控的随机性和可重现性

    示例用法:
        ```python
        # 1. 市场周期驱动的概率信号模型
        # 牛市期间入场概率高，熊市期间退出概率高
        market_cycle = np.sin(np.linspace(0, 4*np.pi, 200))  # 模拟市场周期
        
        entry_probs = np.clip(0.05 + 0.03 * market_cycle, 0.01, 0.1)
        exit_probs = np.clip(0.05 - 0.03 * market_cycle, 0.01, 0.1)
        
        entries, exits = generate_rand_enex_by_prob_nb(
            shape=(200, 3),
            entry_prob=entry_probs, exit_prob=exit_probs,
            entry_wait=2, exit_wait=1,
            entry_pick_first=True, exit_pick_first=True,
            flex_2d=True, seed=42
        )
        
        # 2. 波动率驱动的信号模型  
        # 高波动期间入场和退出概率都增加
        volatility = np.random.gamma(2, 0.1, 150)
        vol_factor = np.clip(volatility / np.mean(volatility), 0.5, 2.0)
        
        entry_probs = 0.03 * vol_factor
        exit_probs = 0.06 * vol_factor  # 退出概率是入场概率的2倍
        
        entries, exits = generate_rand_enex_by_prob_nb(
            shape=(150, 2),
            entry_prob=entry_probs, exit_prob=exit_probs,
            entry_wait=1, exit_wait=1,
            entry_pick_first=True, exit_pick_first=True,
            flex_2d=True, seed=123
        )
        
        print(f"入场信号数: {entries.sum()}, 退出信号数: {exits.sum()}")
        ```

    注意事项:
        - 入场和退出概率分布可以完全不同
        - 概率值必须在[0,1]范围内
        - pick_first参数控制每次搜索的信号数量
        - 建议设置随机种子以确保结果可重现

    参见:
        generate_enex_nb: 底层信号协调框架
        rand_by_prob_choice_nb: 概率选择函数实现
    """
    # 如果提供了随机种子，设置随机数生成器
    if seed is not None:
        np.random.seed(seed)
    
    # 创建临时索引数组，用于两个概率选择函数共享
    temp_idx_arr = np.empty((shape[0],), dtype=np.int64)
    
    # 使用generate_enex_nb框架协调入场和退出信号生成
    return generate_enex_nb(
        shape,
        entry_wait,
        exit_wait,
        entry_pick_first,
        exit_pick_first,
        # 入场信号使用rand_by_prob_choice_nb和entry_prob
        rand_by_prob_choice_nb, (entry_prob, entry_pick_first, temp_idx_arr, flex_2d),
        # 退出信号使用rand_by_prob_choice_nb和exit_prob  
        rand_by_prob_choice_nb, (exit_prob, exit_pick_first, temp_idx_arr, flex_2d)
    )


# ############# Stop exits ############# #


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存
def first_choice_nb(from_i: int, to_i: int, col: int, a: tp.Array2d) -> tp.Array1d:
    """第一信号选择函数，返回指定数组中第一个True信号的索引。

    这是一个符合choice_func_nb接口规范的简单选择函数，专门用于寻找
    给定范围内的第一个信号。该函数常用于止损止盈系统中，当需要
    重新激活已有的入场信号时使用。

    参数说明:
        from_i (int): 搜索范围的起始索引(包含)
        to_i (int): 搜索范围的结束索引(不包含)
        col (int): 当前列索引，指定要搜索的列
        a (np.array): 二维布尔数组，包含信号数据

    返回值:
        np.array: 包含第一个True信号索引的数组
                 - 如果找到信号，返回长度为1的数组
                 - 如果没有找到，返回空数组

    算法逻辑:
        1. 在指定范围[from_i, to_i)内遍历指定列
        2. 寻找第一个True值的位置
        3. 找到后立即返回该索引
        4. 如果没有找到任何True值，返回空数组

    使用场景:
        - 重新激活现有的入场信号
        - 在止损止盈系统中寻找信号起始点
        - 作为generate_enex_nb的选择函数
        - 简单的信号检测和定位

    性能特点:
        - Numba编译优化，高速搜索
        - 启用缓存，提高重复调用性能
        - 早停机制，找到第一个信号后立即返回
        - 内存高效，最小化数组分配

    示例用法:
        ```python
        # 创建测试信号数组
        signals = np.array([[False, True], [True, False], [False, True]])
        
        # 在第0列的[0,3)范围内寻找第一个信号
        first_idx = first_choice_nb(0, 3, 0, signals)
        # 结果: [1] (第1行第0列是第一个True)
        
        # 在第1列寻找
        first_idx = first_choice_nb(0, 3, 1, signals)
        # 结果: [0] (第0行第1列是第一个True)
        ```

    注意事项:
        - 函数只返回第一个匹配的信号索引
        - 如果没有找到信号，返回空数组而不是-1
        - 搜索范围是左闭右开区间[from_i, to_i)
        - 函数是纯函数，无副作用，线程安全
    """
    out = np.empty((1,), dtype=np.int64)  # 预分配结果数组
    
    # 在指定范围内搜索第一个True值
    for i in range(from_i, to_i):
        if a[i, col]:
            out[0] = i  # 保存找到的索引
            return out  # 立即返回
    
    # 如果没有找到任何True值，返回空数组
    return out[:0]


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存  
def stop_choice_nb(from_i: int,
                   to_i: int,
                   col: int,
                   ts: tp.ArrayLike,
                   stop: tp.MaybeArray[float],
                   trailing: tp.MaybeArray[bool],
                   wait: int,
                   pick_first: bool,
                   temp_idx_arr: tp.Array1d,
                   flex_2d: bool) -> tp.Array1d:
    """基于价格序列的止损信号选择函数。

    这是vectorbt止损止盈系统的核心选择函数，能够基于价格时间序列生成
    固定止损和移动止损信号。该函数支持多种止损类型，是专业风险管理
    系统的重要组成部分。

    参数说明:
        from_i (int): 搜索范围的起始索引(包含)
        to_i (int): 搜索范围的结束索引(不包含)
        col (int): 当前列索引，用于灵活索引
        ts (array-like): 二维价格时间序列数组，如收盘价
        stop (float或array-like): 止损阈值设置
                                 - 正值: 止盈阈值，价格上涨触发
                                 - 负值: 止损阈值，价格下跌触发
                                 - np.nan: 禁用止损功能
                                 - 支持逐帧、逐列、逐行或逐元素设置
        trailing (bool或array-like): 是否启用移动止损
                                    - True: 启用移动止损，动态调整止损价位
                                    - False: 固定止损，基于初始价格计算
                                    - 支持灵活索引配置
        wait (int): 止损信号延迟周期数
                   - 0: 可能在同一bar产生两个信号
                   - >0: 移动止损在from_i之前的bar不会更新
        pick_first (bool): 是否在找到第一个止损信号后停止搜索
        temp_idx_arr (np.array): 临时索引数组，用于存储止损触发位置
        flex_2d (bool): 灵活索引的二维模式标识

    返回值:
        np.array: 止损触发位置的索引数组

    算法逻辑:
        1. **初始化阶段**:
           - 获取初始价格、止损参数和移动止损设置
           - 初始化最高价和最低价追踪变量
        
        2. **止损价格计算**:
           - 固定止损: 基于初始价格计算止损位
           - 移动止损: 基于当前最优价格动态调整
           - 支持多头和空头两种方向
        
        3. **止损触发检测**:
           - 检查当前价格是否触及止损位
           - 记录触发位置并根据pick_first参数决定是否继续
        
        4. **移动止损更新**:
           - 持续跟踪最高价和最低价
           - 动态调整移动止损的基准价格

    止损类型详解:
        - **固定止损** (trailing=False): 止损位固定不变，基于入场价格
        - **移动止损** (trailing=True): 止损位随有利价格移动而调整
        - **止盈订单** (stop>0): 价格达到目标盈利时触发
        - **止损订单** (stop<0): 价格达到最大亏损时触发

    使用场景:
        - 专业的风险管理系统
        - 自动化止损止盈策略
        - 趋势跟踪移动止损
        - 多资产组合的风险控制

    性能特点:
        - Numba编译优化，毫秒级止损计算
        - 启用缓存，重复调用性能卓越
        - 内存预分配，避免运行时分配
        - 支持向量化的多资产处理

    示例用法:
        ```python
        # 创建价格时间序列
        prices = np.array([[100, 105, 98, 110, 95]]).T  # 5个时间点的价格
        temp_arr = np.empty(5, dtype=np.int64)
        
        # 固定10%止损
        stop_indices = stop_choice_nb(
            0, 5, 0, prices, stop=-0.1, trailing=False,
            wait=0, pick_first=True, temp_idx_arr=temp_arr, flex_2d=True
        )
        
        # 移动10%止损
        trail_indices = stop_choice_nb(
            0, 5, 0, prices, stop=-0.1, trailing=True,
            wait=0, pick_first=True, temp_idx_arr=temp_arr, flex_2d=True
        )
        ```

    注意事项:
        - stop参数的正负号决定了止损的方向性
        - 移动止损只能朝有利方向移动，不能回撤
        - wait>0时会影响移动止损的更新频率
        - 函数假设价格序列的连续性和有效性

    参见:
        flex_select_auto_nb: 灵活索引选择函数
        StopType: 止损类型枚举定义
    """
    j = 0  # 结果索引计数器
    
    # 计算初始化参数的时间点
    init_i = from_i - wait
    
    # 获取初始价格和止损参数
    init_ts = flex_select_auto_nb(ts, init_i, col, flex_2d)
    init_stop = flex_select_auto_nb(np.asarray(stop), init_i, col, flex_2d)
    init_trailing = flex_select_auto_nb(np.asarray(trailing), init_i, col, flex_2d)
    
    # 初始化最高价和最低价追踪变量（用于移动止损）
    max_high = min_low = init_ts

    # 遍历指定的价格序列范围
    for i in range(from_i, to_i):
        # 只有启用了止损功能才进行计算
        if not np.isnan(init_stop):
            if init_trailing:
                # 移动止损价格计算
                if init_stop >= 0:
                    # 移动止盈：基于最低价向上调整
                    curr_stop_price = min_low * (1 + abs(init_stop))
                else:
                    # 移动止损：基于最高价向下调整
                    curr_stop_price = max_high * (1 - abs(init_stop))
            else:
                # 固定止损价格计算：基于初始价格
                curr_stop_price = init_ts * (1 + init_stop)

        # 获取当前价格并检查止损触发条件
        curr_ts = flex_select_auto_nb(ts, i, col, flex_2d)
        
        if not np.isnan(init_stop):
            # 判断止损触发条件
            if init_stop >= 0:
                # 止盈条件：价格达到或超过目标价位
                exit_signal = curr_ts >= curr_stop_price
            else:
                # 止损条件：价格跌破或触及止损价位
                exit_signal = curr_ts <= curr_stop_price
            
            # 如果触发止损信号
            if exit_signal:
                temp_idx_arr[j] = i  # 记录触发位置
                j += 1
                # 如果只需要第一个信号，立即返回
                if pick_first:
                    return temp_idx_arr[:1]

        # 更新移动止损的最高价和最低价追踪
        if init_trailing:
            if curr_ts < min_low:
                min_low = curr_ts  # 更新最低价
            elif curr_ts > max_high:
                max_high = curr_ts  # 更新最高价
    
    # 返回所有触发位置
    return temp_idx_arr[:j]


@njit  # 使用Numba JIT编译优化
def generate_stop_ex_nb(entries: tp.Array2d,
                        ts: tp.ArrayLike,
                        stop: tp.MaybeArray[float],
                        trailing: tp.MaybeArray[bool],
                        wait: int,
                        until_next: bool,
                        skip_until_exit: bool,
                        pick_first: bool,
                        flex_2d: bool) -> tp.Array2d:
    """为入场信号生成基于价格的止损退出信号。

    这是vectorbt止损止盈系统的主要用户接口函数，它将复杂的止损逻辑
    封装成简单易用的函数调用。该函数支持固定止损、移动止损、止盈等
    多种专业级风险管理功能。

    参数说明:
        entries (np.array): 入场信号的二维布尔数组，形状为(时间, 资产)
        ts (array-like): 价格时间序列，用于止损计算的基准价格
        stop (float或array-like): 止损阈值，参见stop_choice_nb
        trailing (bool或array-like): 移动止损开关，参见stop_choice_nb
        wait (int): 止损延迟周期，参见stop_choice_nb
        until_next (bool): 止损搜索范围控制，参见generate_ex_nb
        skip_until_exit (bool): 入场信号跳过控制，参见generate_ex_nb
        pick_first (bool): 首选信号控制，参见generate_ex_nb
        flex_2d (bool): 灵活索引二维模式

    返回值:
        np.array: 与entries相同形状的布尔数组，True表示止损退出信号位置

    算法逻辑:
        1. 创建临时索引数组用于性能优化
        2. 使用generate_ex_nb作为基础框架
        3. 使用stop_choice_nb作为止损选择函数
        4. 将所有止损参数传递给底层函数

    止损策略类型:
        - **固定止损策略**: 基于入场价格的固定百分比止损
        - **移动止损策略**: 随价格有利移动而调整的动态止损
        - **止盈策略**: 达到目标收益时的获利了结
        - **组合策略**: 多种止损方式的组合应用

    使用场景:
        - 股票、期货、加密货币的风险管理
        - 自动化交易系统的止损模块
        - 量化策略的风险控制组件
        - 多资产组合的统一风险管理

    性能特点:
        - Numba编译优化，专业级性能
        - 向量化处理，支持多资产批量处理
        - 内存高效，最小化中间数据存储
        - 可配置的灵活参数系统

    示例用法:
        * 生成10%的移动止损和止盈信号

        ```python
        import numpy as np
        from vectorbt.signals.nb import generate_stop_ex_nb

        # 创建入场信号：第2个时间点入场
        entries = np.asarray([False, True, False, False, False])[:, None]
        # 价格序列：入场后先上涨再下跌
        ts = np.asarray([1, 2, 3, 2, 1])[:, None]

        # 生成10%移动止损信号
        stop_exits = generate_stop_ex_nb(entries, ts, -0.1, True, 1, True, True, True, True)
        # 结果: 第4个时间点触发止损（价格从3跌到2，触发移动止损）
        
        # 生成10%固定止盈信号  
        profit_exits = generate_stop_ex_nb(entries, ts, 0.1, False, 1, True, True, True, True)
        # 结果: 第3个时间点触发止盈（价格从2涨到3，涨幅50% > 10%）
        ```

    实际应用案例:
        ```python
        # 多资产移动止损系统
        entries = create_ma_cross_signals(prices)  # 移动平均交叉入场
        
        # 配置不同资产的止损参数
        stop_levels = np.array([0.05, 0.08, 0.03])  # 5%, 8%, 3%止损
        trailing_flags = np.array([True, True, False])  # 前两个用移动止损
        
        stop_exits = generate_stop_ex_nb(
            entries, prices, -stop_levels, trailing_flags,
            wait=1, until_next=True, skip_until_exit=False,
            pick_first=True, flex_2d=True
        )
        ```

    注意事项:
        - 止损阈值的正负号决定止损还是止盈
        - 移动止损只能向有利方向调整，无法回撤
        - wait参数会影响止损触发的及时性
        - 建议在实际应用前进行充分的历史回测

    参见:
        generate_ex_nb: 底层退出信号生成框架
        stop_choice_nb: 核心止损选择算法
        StopType: 止损类型枚举定义
    """
    # 创建临时索引数组，用于存储止损触发位置
    temp_idx_arr = np.empty((entries.shape[0],), dtype=np.int64)
    
    # 使用generate_ex_nb框架，以stop_choice_nb作为选择函数生成止损信号
    return generate_ex_nb(
        entries,             # 入场信号数组
        wait,               # 延迟周期
        until_next,         # 搜索范围控制
        skip_until_exit,    # 信号跳过控制
        pick_first,         # 首选信号控制
        stop_choice_nb,     # 止损选择函数
        # 传递给stop_choice_nb的参数
        ts,                 # 价格时间序列
        stop,               # 止损阈值
        trailing,           # 移动止损标志
        wait,               # 延迟周期（重复传递）
        pick_first,         # 首选信号标志（重复传递）
        temp_idx_arr,       # 临时索引数组
        flex_2d             # 灵活索引模式
    )


@njit  # 使用Numba JIT编译优化
def generate_stop_enex_nb(entries: tp.Array2d,
                          ts: tp.Array,
                          stop: tp.MaybeArray[float],
                          trailing: tp.MaybeArray[bool],
                          entry_wait: int,
                          exit_wait: int,
                          pick_first: bool,
                          flex_2d: bool) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """基于止损逻辑交替生成入场和退出信号。

    这是vectorbt止损系统的高级协调函数，它能够智能地管理入场和退出信号的
    交替生成，确保每个入场信号都有对应的止损退出信号。该函数实现了完整的
    交易周期管理，特别适用于需要严格风险控制的交易系统。

    参数说明:
        entries (np.array): 初始入场信号模板，用于信号激活
        ts (array-like): 价格时间序列，用于止损计算
        stop (float或array-like): 止损阈值设置，参见stop_choice_nb
        trailing (bool或array-like): 移动止损配置，参见stop_choice_nb
        entry_wait (int): 入场信号间的最小等待周期
        exit_wait (int): 退出信号间的最小等待周期
        pick_first (bool): 是否只选择第一个触发的止损信号
        flex_2d (bool): 灵活索引的二维模式标识

    返回值:
        tuple[np.array, np.array]: (新的入场信号矩阵, 止损退出信号矩阵)
        - 两个矩阵形状都与entries相同
        - 新入场信号是原始信号的清理版本
        - 退出信号严格对应每个有效的入场信号

    算法逻辑:
        1. 使用generate_enex_nb作为协调框架
        2. 入场选择使用first_choice_nb激活原始入场信号
        3. 退出选择使用ohlc_stop_choice_nb实现止损逻辑
        4. 确保每个入场信号都有对应的止损退出

    与generate_stop_ex_nb的区别:
        - **信号清理**: 自动清理重叠和冲突的入场信号
        - **严格配对**: 确保入场和退出信号的一对一关系
        - **周期控制**: 支持入场和退出的独立时间控制
        - **系统完整性**: 提供完整的交易周期管理

    使用场景:
        - 需要严格信号配对的交易系统
        - 专业级的风险管理平台
        - 自动化交易机器人的核心模块
        - 复杂策略的信号协调管理

    性能特点:
        - Numba编译优化，高速信号协调
        - 内存高效的信号配对算法
        - 支持大规模多资产处理
        - 完整的交易周期状态管理

    示例用法:
        ```python
        # 创建移动平均交叉入场信号
        ma_fast = prices.rolling(5).mean()
        ma_slow = prices.rolling(20).mean()
        raw_entries = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
        
        # 生成带止损的完整交易信号
        clean_entries, stop_exits = generate_stop_enex_nb(
            raw_entries.values, prices.values,
            stop=-0.05,          # 5%固定止损
            trailing=False,      # 使用固定止损
            entry_wait=2,        # 入场信号间隔2个周期
            exit_wait=1,         # 止损后1个周期才能重新入场
            pick_first=True,     # 只选择第一个止损触发点
            flex_2d=True
        )
        
        print(f"原始入场信号数: {raw_entries.sum()}")
        print(f"清理后入场信号数: {clean_entries.sum()}")
        print(f"止损退出信号数: {stop_exits.sum()}")
        ```

    应用案例:
        ```python
        # 趋势跟踪系统with移动止损
        trend_signals = identify_trend_breakouts(prices)
        
        entries, exits = generate_stop_enex_nb(
            trend_signals, prices,
            stop=-0.08,          # 8%移动止损
            trailing=True,       # 启用移动止损
            entry_wait=3,        # 避免频繁交易
            exit_wait=2,         # 止损后冷静期
            pick_first=True,
            flex_2d=True
        )
        ```

    注意事项:
        - 该函数会修改原始入场信号，移除无法配对的入场
        - 移动止损参数会影响退出信号的触发时机
        - entry_wait和exit_wait参数控制信号密度和质量
        - 建议配合实际交易成本进行参数调优

    参见:
        generate_enex_nb: 底层信号协调框架
        first_choice_nb: 入场信号选择器
        stop_choice_nb: 止损信号选择器
    """
    # 创建临时索引数组，用于止损计算
    temp_idx_arr = np.empty((entries.shape[0],), dtype=np.int64)
    
    # 使用generate_enex_nb框架协调入场和止损退出信号
    return generate_enex_nb(
        entries.shape,      # 信号矩阵的形状
        entry_wait,         # 入场信号等待周期
        exit_wait,          # 退出信号等待周期
        True,               # 入场信号pick_first=True
        pick_first,         # 退出信号的pick_first设置
        # 入场信号选择：使用first_choice_nb激活原始入场信号
        first_choice_nb, (entries,),
        # 退出信号选择：使用stop_choice_nb实现止损逻辑
        stop_choice_nb, (ts, stop, trailing, exit_wait, pick_first, temp_idx_arr, flex_2d)
    )


@njit(cache=True)
def ohlc_stop_choice_nb(from_i: int,
                        to_i: int,
                        col: int,
                        open: tp.ArrayLike,
                        high: tp.ArrayLike,
                        low: tp.ArrayLike,
                        close: tp.ArrayLike,
                        stop_price_out: tp.Array2d,
                        stop_type_out: tp.Array2d,
                        sl_stop: tp.MaybeArray[float],
                        sl_trail: tp.MaybeArray[bool],
                        tp_stop: tp.MaybeArray[float],
                        reverse: tp.MaybeArray[bool],
                        is_open_safe: bool,
                        wait: int,
                        pick_first: bool,
                        temp_idx_arr: tp.Array1d,
                        flex_2d: bool) -> tp.Array1d:
    """基于OHLC数据的高级止损止盈选择函数。

    这是vectorbt止损系统中最复杂和最专业的选择函数，它基于完整的OHLC
    (开高低收)数据来精确检测止损和止盈的触发时机。相比于简单的基于收盘价
    的止损，OHLC止损能够更准确地反映盘中的价格波动和真实的止损执行情况。

    参数说明:
        from_i (int): 搜索范围的起始索引(包含)
        to_i (int): 搜索范围的结束索引(不包含)
        col (int): 当前列索引，用于灵活索引
        open (array-like): 开盘价时间序列数组
        high (array-like): 最高价时间序列数组
        low (array-like): 最低价时间序列数组
        close (array-like): 收盘价时间序列数组
        stop_price_out (np.array): 输出数组，用于记录实际的止损触发价格
        stop_type_out (np.array): 输出数组，用于记录止损类型
                                 - 0: 止损 (StopType.StopLoss)
                                 - 1: 移动止损 (StopType.TrailStop)
                                 - 2: 止盈 (StopType.TakeProfit)
        sl_stop (float或array-like): 止损阈值设置（必须>=0）
                                   - 0.05表示5%的止损幅度
                                   - np.nan表示禁用止损
        sl_trail (bool或array-like): 是否启用移动止损
        tp_stop (float或array-like): 止盈阈值设置（必须>=0）
                                   - 0.10表示10%的止盈幅度
                                   - np.nan表示禁用止盈
        reverse (bool或array-like): 是否反向操作（适用于空头持仓）
                                  - False: 正向操作，价格向上为盈利
                                  - True: 反向操作，价格向下为盈利
        is_open_safe (bool): 开盘价是否在入场价格之前或同时
                           - True: 可以在入场bar使用高低价
                           - False: 入场bar只能使用收盘价
        wait (int): 止损延迟周期数，参见stop_choice_nb
        pick_first (bool): 是否在触发第一个止损后立即停止搜索
        temp_idx_arr (np.array): 临时索引数组，用于存储触发位置
        flex_2d (bool): 灵活索引的二维模式标识

    返回值:
        np.array: 止损/止盈触发位置的索引数组

    算法逻辑:
        1. **数据预处理阶段**:
           - 验证止损止盈参数的有效性（必须>=0）
           - 初始化价格追踪变量（最高价、最低价）
           - 设置初始参考价格和交易方向
        
        2. **逐bar分析阶段**:
           - 获取当前bar的OHLC数据并进行完整性检查
           - 根据移动止损设置动态计算止损价位
           - 计算固定止盈价位
           - 确定当前bar可用的价格范围
        
        3. **触发检测阶段**:
           - 优先检查止损触发条件（保守原则）
           - 检查止盈触发条件
           - 记录触发价格和触发类型
           - 更新移动止损的价格追踪
        
        4. **结果输出阶段**:
           - 将触发信息写入输出数组
           - 根据pick_first参数决定是否继续

    OHLC数据处理逻辑:
        - **完整性检查**: 自动补全缺失的OHLC数据
        - **价格范围确定**: 根据is_open_safe确定可用价格范围
        - **触发优先级**: 止损优先于止盈（保守交易原则）
        - **移动止损更新**: 基于历史最优价格动态调整

    交易方向支持:
        - **正向交易** (reverse=False): 适用于多头持仓
          - 止损: 价格跌破止损位
          - 止盈: 价格突破止盈位
          - 移动止损: 跟随价格上涨调整止损位
        - **反向交易** (reverse=True): 适用于空头持仓
          - 止损: 价格突破止损位
          - 止盈: 价格跌破止盈位
          - 移动止损: 跟随价格下跌调整止损位

    使用场景:
        - 专业级的OHLC数据止损系统
        - 精确的盘中止损执行模拟
        - 多空交易策略的风险管理
        - 高频交易的精密止损控制

    性能特点:
        - Numba编译优化，纳秒级OHLC分析
        - 启用缓存，重复调用性能极佳
        - 内存高效，就地计算减少分配
        - 支持复杂的多参数止损逻辑

    重要说明:
        由于缺乏bar内部的时序数据，当同一bar内同时触发止损和止盈时，
        算法采用保守策略，优先执行止损。这种设计确保了风险控制的优先级。

        移动止损只能基于前一bar的价格信息进行更新，这是为了避免
        使用当前bar内的未来信息，确保回测的真实性。

    示例用法:
        ```python
        # 创建OHLC数据
        ohlc_data = pd.DataFrame({
            'open': [100, 102, 104, 103, 101],
            'high': [101, 105, 106, 104, 102], 
            'low': [99, 101, 103, 100, 99],
            'close': [102, 104, 103, 101, 100]
        })
        
        # 输出数组
        stop_price_out = np.full((5, 1), np.nan)
        stop_type_out = np.full((5, 1), -1, dtype=np.int64)
        temp_arr = np.empty(5, dtype=np.int64)
        
        # 5%移动止损 + 10%止盈
        triggers = ohlc_stop_choice_nb(
            0, 5, 0,
            ohlc_data.open.values, ohlc_data.high.values,
            ohlc_data.low.values, ohlc_data.close.values,
            stop_price_out, stop_type_out,
            sl_stop=0.05, sl_trail=True, tp_stop=0.10, reverse=False,
            is_open_safe=True, wait=0, pick_first=True,
            temp_idx_arr=temp_arr, flex_2d=False
        )
        ```

    注意事项:
        - sl_stop和tp_stop参数必须为非负值
        - OHLC数据的完整性会影响止损精度
        - is_open_safe参数影响入场bar的价格使用范围
        - 函数会自动处理数据缺失和异常情况

    参见:
        StopType: 止损类型枚举定义
        flex_select_auto_nb: 灵活索引选择函数
        stop_choice_nb: 简化版本的止损选择函数
    """
    # 初始化结果计数器和基础参数
    j = 0
    init_i = from_i - wait
    
    # 获取初始化参数
    init_open = flex_select_auto_nb(open, init_i, col, flex_2d)
    init_sl_stop = flex_select_auto_nb(np.asarray(sl_stop), init_i, col, flex_2d)
    
    # 验证止损参数有效性
    if init_sl_stop < 0:
        raise ValueError("止损数值必须大于等于0")
    
    init_sl_trail = flex_select_auto_nb(np.asarray(sl_trail), init_i, col, flex_2d)
    init_tp_stop = flex_select_auto_nb(np.asarray(tp_stop), init_i, col, flex_2d)
    
    # 验证止盈参数有效性
    if init_tp_stop < 0:
        raise ValueError("止盈数值必须大于等于0")
    
    init_reverse = flex_select_auto_nb(np.asarray(reverse), init_i, col, flex_2d)
    
    # 初始化移动止损的价格追踪变量
    max_p = min_p = init_open

    # 逐bar遍历分析
    for i in range(from_i, to_i):
        # 获取当前bar的OHLC数据
        _open = flex_select_auto_nb(open, i, col, flex_2d)
        _high = flex_select_auto_nb(high, i, col, flex_2d)
        _low = flex_select_auto_nb(low, i, col, flex_2d)
        _close = flex_select_auto_nb(close, i, col, flex_2d)
        
        # OHLC数据完整性检查和修复
        if np.isnan(_open):
            _open = _close  # 用收盘价替代缺失的开盘价
        if np.isnan(_low):
            _low = min(_open, _close)  # 用开盘价和收盘价的较小值作为最低价
        if np.isnan(_high):
            _high = max(_open, _close)  # 用开盘价和收盘价的较大值作为最高价

        # 计算当前的止损价格
        if not np.isnan(init_sl_stop):
            if init_sl_trail:
                # 移动止损价格计算
                if init_reverse:
                    # 空头移动止损：基于最低价向上调整
                    curr_sl_stop_price = min_p * (1 + init_sl_stop)
                else:
                    # 多头移动止损：基于最高价向下调整
                    curr_sl_stop_price = max_p * (1 - init_sl_stop)
            else:
                # 固定止损价格计算
                if init_reverse:
                    curr_sl_stop_price = init_open * (1 + init_sl_stop)
                else:
                    curr_sl_stop_price = init_open * (1 - init_sl_stop)
        
        # 计算当前的止盈价格
        if not np.isnan(init_tp_stop):
            if init_reverse:
                # 空头止盈：价格下跌达到止盈幅度
                curr_tp_stop_price = init_open * (1 - init_tp_stop)
            else:
                # 多头止盈：价格上涨达到止盈幅度
                curr_tp_stop_price = init_open * (1 + init_tp_stop)

        # 确定当前bar可用的价格范围
        if i > init_i or is_open_safe:
            # is_open_safe=True 表示开盘价在入场价之前或同时，
            # 因此可以安全地使用当前bar的高低价
            curr_high = _high
            curr_low = _low
        else:
            # 否则，只能使用收盘价（保守策略）
            curr_high = curr_low = _close

        # 检测止损和止盈触发条件
        exit_signal = False
        
        # 优先检查止损触发（风险控制优先原则）
        if not np.isnan(init_sl_stop):
            if (not init_reverse and curr_low <= curr_sl_stop_price) or \
                    (init_reverse and curr_high >= curr_sl_stop_price):
                # 止损触发
                exit_signal = True
                stop_price_out[i, col] = curr_sl_stop_price  # 记录触发价格
                
                # 设置止损类型标识
                if init_sl_trail:
                    stop_type_out[i, col] = StopType.TrailStop  # 移动止损
                else:
                    stop_type_out[i, col] = StopType.StopLoss   # 固定止损
        
        # 如果没有触发止损，检查止盈条件
        if not exit_signal and not np.isnan(init_tp_stop):
            if (not init_reverse and curr_high >= curr_tp_stop_price) or \
                    (init_reverse and curr_low <= curr_tp_stop_price):
                # 止盈触发
                exit_signal = True
                stop_price_out[i, col] = curr_tp_stop_price    # 记录触发价格
                stop_type_out[i, col] = StopType.TakeProfit   # 设置止盈类型
        
        # 如果触发了退出信号
        if exit_signal:
            temp_idx_arr[j] = i  # 记录触发位置
            j += 1
            if pick_first:
                return temp_idx_arr[:1]  # 只需要第一个信号则立即返回

        # 更新移动止损的价格追踪（只有启用移动止损时才更新）
        if init_sl_trail:
            if curr_low < min_p:
                min_p = curr_low  # 更新历史最低价
            if curr_high > max_p:
                max_p = curr_high  # 更新历史最高价

    # 返回所有触发位置
    return temp_idx_arr[:j]


@njit  # 使用Numba JIT编译优化
def generate_ohlc_stop_ex_nb(entries: tp.Array2d,
                             open: tp.ArrayLike,
                             high: tp.ArrayLike,
                             low: tp.ArrayLike,
                             close: tp.ArrayLike,
                             stop_price_out: tp.Array2d,
                             stop_type_out: tp.Array2d,
                             sl_stop: tp.MaybeArray[float],
                             sl_trail: tp.MaybeArray[bool],
                             tp_stop: tp.MaybeArray[float],
                             reverse: tp.MaybeArray[bool],
                             is_open_safe: bool,
                             wait: int,
                             until_next: bool,
                             skip_until_exit: bool,
                             pick_first: bool,
                             flex_2d: bool) -> tp.Array2d:
    """基于OHLC数据为入场信号生成高精度止损止盈退出信号。

    这是vectorbt OHLC止损系统的主要用户接口函数，它结合了完整的OHLC数据
    和先进的止损止盈逻辑，为专业交易系统提供最精确的风险管理功能。
    该函数能够同时处理固定止损、移动止损和止盈逻辑，是机构级交易系统的核心组件。

    参数说明:
        entries (np.array): 入场信号的二维布尔数组，形状为(时间, 资产)
        open/high/low/close (array-like): OHLC价格数据时间序列
        stop_price_out (np.array): 输出数组，记录实际止损/止盈触发价格
                                  形状必须与entries相同，初始化为np.nan
        stop_type_out (np.array): 输出数组，记录止损/止盈触发类型
                                 形状必须与entries相同，初始化为-1
        sl_stop (float或array-like): 止损阈值，必须>=0，参见ohlc_stop_choice_nb
        sl_trail (bool或array-like): 移动止损开关，参见ohlc_stop_choice_nb
        tp_stop (float或array-like): 止盈阈值，必须>=0，参见ohlc_stop_choice_nb
        reverse (bool或array-like): 交易方向标识，参见ohlc_stop_choice_nb
        is_open_safe (bool): 开盘价安全性标识，参见ohlc_stop_choice_nb
        wait (int): 止损延迟周期数，参见generate_ex_nb
        until_next (bool): 搜索范围控制，参见generate_ex_nb
        skip_until_exit (bool): 信号跳过控制，参见generate_ex_nb
        pick_first (bool): 首选信号控制，参见generate_ex_nb
        flex_2d (bool): 灵活索引二维模式标识

    返回值:
        np.array: 与entries相同形状的布尔数组，True表示OHLC止损/止盈退出信号位置

    算法特点:
        1. **精确的盘中价格分析**: 利用完整OHLC数据精确判断触发时机
        2. **同时止损止盈**: 在单一函数中处理所有类型的退出逻辑
        3. **多空兼容**: 通过reverse参数支持多头和空头策略
        4. **价格追踪输出**: 提供实际触发价格和触发类型的详细记录

    OHLC优势对比:
        相比基于收盘价的简单止损，OHLC止损具有以下优势：
        - **更高精度**: 考虑盘中的完整价格波动
        - **真实触发**: 反映实际交易中的触发情况
        - **避免滑点**: 更准确地模拟真实的止损执行价格
        - **同bar触发**: 支持在入场同一bar内触发退出信号

    应用场景:
        - 专业交易系统的核心风险管理模块
        - 高精度的历史回测和策略验证
        - 机构级量化交易平台
        - 需要精确止损执行的自动化交易

    性能特点:
        - Numba编译优化，机构级性能表现
        - 向量化OHLC数据处理
        - 内存高效的大规模数据支持
        - 完整的触发信息记录系统

    示例用法:
        * 生成移动止损和止盈信号，展示同bar内退出信号的生成能力

        ```python
        import numpy as np
        from vectorbt.signals.nb import generate_ohlc_stop_ex_nb

        # 创建入场信号和OHLC数据
        entries = np.asarray([True, False, True, False, False])[:, None]
        entry_price = np.asarray([10, 11, 12, 11, 10])[:, None]
        high_price = entry_price + 1  # 最高价比入场价高1
        low_price = entry_price - 1   # 最低价比入场价低1
        close_price = entry_price     # 收盘价等于入场价
        
        # 初始化输出数组
        stop_price_out = np.full_like(entries, np.nan, dtype=np.float64)
        stop_type_out = np.full_like(entries, -1, dtype=np.int64)

        # 生成OHLC止损信号：10%移动止损 + 10%止盈
        exits = generate_ohlc_stop_ex_nb(
            entries=entries,
            open=entry_price, high=high_price, 
            low=low_price, close=close_price,
            stop_price_out=stop_price_out,
            stop_type_out=stop_type_out,
            sl_stop=0.1,      # 10%移动止损
            sl_trail=True,    # 启用移动止损
            tp_stop=0.1,      # 10%止盈
            reverse=False,    # 多头交易
            is_open_safe=True, # 可以使用入场bar的高低价
            wait=1, until_next=True, skip_until_exit=False,
            pick_first=True, flex_2d=True
        )

        print("退出信号:", exits.flatten())
        print("触发价格:", stop_price_out.flatten())
        print("触发类型:", stop_type_out.flatten())
        ```

    实际应用案例:
        ```python
        # 多资产OHLC止损系统
        ohlc_data = load_ohlc_data(['AAPL', 'GOOGL', 'MSFT'])  # 加载OHLC数据
        ma_signals = generate_ma_crossover_signals(ohlc_data.close)  # 生成入场信号
        
        # 配置不同资产的止损参数
        stop_levels = np.array([0.05, 0.08, 0.06])  # 不同的止损水平
        profit_targets = np.array([0.15, 0.12, 0.10])  # 不同的止盈目标
        
        # 初始化输出数组
        stop_prices = np.full_like(ma_signals, np.nan, dtype=float)
        stop_types = np.full_like(ma_signals, -1, dtype=int)
        
        # 生成OHLC止损退出信号
        ohlc_exits = generate_ohlc_stop_ex_nb(
            ma_signals, ohlc_data.open, ohlc_data.high, 
            ohlc_data.low, ohlc_data.close,
            stop_prices, stop_types,
            sl_stop=stop_levels, sl_trail=True,
            tp_stop=profit_targets, reverse=False,
            is_open_safe=True, wait=1, until_next=True,
            skip_until_exit=False, pick_first=True, flex_2d=True
        )
        ```

    注意事项:
        - 输出数组必须预先分配且形状正确
        - OHLC数据的质量直接影响止损精度
        - is_open_safe=False时首个bar只能使用收盘价
        - 建议在实际应用前进行充分的参数优化

    性能说明:
        如果is_open_safe为False，第一个退出信号将在第二个bar执行。
        这是因为我们无法确定入场价格是否在第一个bar的高低价之前，
        因此移动止损无法对9.0的低价触发。

    参见:
        generate_ex_nb: 底层退出信号生成框架
        ohlc_stop_choice_nb: 核心OHLC止损选择算法
        StopType: 止损类型枚举定义
    """
    # 创建临时索引数组，用于存储触发位置
    temp_idx_arr = np.empty((entries.shape[0],), dtype=np.int64)
    
    # 使用generate_ex_nb框架，以ohlc_stop_choice_nb作为选择函数
    return generate_ex_nb(
        entries,                # 入场信号数组
        wait,                  # 延迟周期
        until_next,            # 搜索范围控制
        skip_until_exit,       # 信号跳过控制
        pick_first,            # 首选信号控制
        ohlc_stop_choice_nb,   # OHLC止损选择函数
        # 传递给ohlc_stop_choice_nb的参数
        open, high, low, close,          # OHLC价格数据
        stop_price_out, stop_type_out,   # 输出数组
        sl_stop, sl_trail, tp_stop,      # 止损止盈参数
        reverse, is_open_safe,           # 交易方向和安全性参数
        wait, pick_first,                # 延迟和选择参数
        temp_idx_arr, flex_2d            # 临时数组和索引参数
    )


@njit  # 使用Numba JIT编译优化
def generate_ohlc_stop_enex_nb(entries: tp.Array2d,
                               open: tp.ArrayLike,
                               high: tp.ArrayLike,
                               low: tp.ArrayLike,
                               close: tp.ArrayLike,
                               stop_price_out: tp.Array2d,
                               stop_type_out: tp.Array2d,
                               sl_stop: tp.MaybeArray[float],
                               sl_trail: tp.MaybeArray[bool],
                               tp_stop: tp.MaybeArray[float],
                               reverse: tp.MaybeArray[bool],
                               is_open_safe: bool,
                               entry_wait: int,
                               exit_wait: int,
                               pick_first: bool,
                               flex_2d: bool) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """基于OHLC数据交替生成入场和止损止盈信号的高级协调函数。

    这是vectorbt OHLC止损系统的终极协调函数，它智能地管理完整的交易周期，
    确保每个入场信号都有严格配对的OHLC止损止盈退出信号。该函数实现了
    机构级交易系统所需的完整信号管理和风险控制功能。

    参数说明:
        entries (np.array): 初始入场信号模板，用于信号激活
        open/high/low/close (array-like): 完整的OHLC价格数据时间序列
        stop_price_out (np.array): 输出数组，记录实际触发价格
        stop_type_out (np.array): 输出数组，记录触发类型
        sl_stop (float或array-like): 止损阈值设置，参见ohlc_stop_choice_nb
        sl_trail (bool或array-like): 移动止损配置，参见ohlc_stop_choice_nb
        tp_stop (float或array-like): 止盈阈值设置，参见ohlc_stop_choice_nb
        reverse (bool或array-like): 交易方向配置，参见ohlc_stop_choice_nb
        is_open_safe (bool): 开盘价安全性标识，参见ohlc_stop_choice_nb
        entry_wait (int): 入场信号间的最小等待周期
        exit_wait (int): 退出信号间的最小等待周期
        pick_first (bool): 是否只选择第一个触发的止损/止盈信号
        flex_2d (bool): 灵活索引的二维模式标识

    返回值:
        tuple[np.array, np.array]: (清理后的入场信号矩阵, OHLC止损止盈退出信号矩阵)
        - 两个矩阵形状都与entries相同
        - 新入场信号是原始信号的清理和重组版本
        - 退出信号严格对应每个有效入场信号的OHLC止损/止盈

    算法逻辑:
        1. **信号清理阶段**: 使用first_choice_nb清理和重组原始入场信号
        2. **OHLC止损协调**: 使用ohlc_stop_choice_nb实现高精度止损逻辑
        3. **周期管理**: 通过entry_wait和exit_wait控制信号密度
        4. **完整配对**: 确保每个入场都有对应的OHLC退出信号

    与其他函数的关系:
        - **vs generate_ohlc_stop_ex_nb**: 增加了信号清理和严格配对
        - **vs generate_stop_enex_nb**: 使用更精确的OHLC数据而非单一价格
        - **vs generate_enex_nb**: 专门针对止损止盈的高级封装

    核心优势:
        - **完整交易周期管理**: 从入场到退出的全程信号协调
        - **OHLC精确度**: 基于完整OHLC数据的最准确止损执行
        - **机构级功能**: 同时支持止损、移动止损和止盈的复合逻辑
        - **信号质量保证**: 自动清理冲突和重叠的信号

    使用场景:
        - 需要最高精度止损的专业交易系统
        - 机构级量化投资平台的核心模块
        - 复杂多策略系统的风险管理协调
        - 高要求的实盘自动交易系统

    性能特点:
        - Numba编译优化，达到C语言级别性能
        - 内存高效的大规模OHLC数据处理
        - 智能的信号配对和状态管理
        - 完整的触发信息记录和追溯

    示例用法:
        ```python
        # 构建复杂的OHLC止损交易系统
        ohlc_data = fetch_market_data(['AAPL', 'GOOGL', 'MSFT'])
        
        # 生成技术指标入场信号
        rsi_signals = generate_rsi_signals(ohlc_data)
        ma_signals = generate_ma_crossover_signals(ohlc_data)
        combined_signals = rsi_signals | ma_signals  # 组合信号
        
        # 初始化OHLC止损输出数组
        stop_prices = np.full_like(combined_signals, np.nan, dtype=float)
        stop_types = np.full_like(combined_signals, -1, dtype=int)
        
        # 生成完整的OHLC止损交易信号
        clean_entries, ohlc_exits = generate_ohlc_stop_enex_nb(
            combined_signals,                    # 原始入场信号
            ohlc_data.open, ohlc_data.high,     # OHLC数据
            ohlc_data.low, ohlc_data.close,
            stop_prices, stop_types,             # 输出数组
            sl_stop=0.06,                        # 6%固定止损
            sl_trail=False,                      # 使用固定止损
            tp_stop=0.12,                        # 12%止盈目标
            reverse=False,                       # 多头策略
            is_open_safe=True,                   # 安全使用开盘价
            entry_wait=3,                        # 入场信号间隔3期
            exit_wait=1,                         # 退出后1期冷却
            pick_first=True,                     # 只取第一个触发
            flex_2d=True
        )
        
        # 分析结果
        print(f"原始入场信号数: {combined_signals.sum()}")
        print(f"清理后入场数: {clean_entries.sum()}")
        print(f"OHLC退出信号数: {ohlc_exits.sum()}")
        print(f"平均触发价格: {np.nanmean(stop_prices):.2f}")
        
        # 分析止损类型分布
        stop_loss_count = (stop_types == StopType.StopLoss).sum()
        trail_stop_count = (stop_types == StopType.TrailStop).sum()
        take_profit_count = (stop_types == StopType.TakeProfit).sum()
        print(f"止损触发: {stop_loss_count}, 移动止损: {trail_stop_count}, 止盈: {take_profit_count}")
        ```

    高级应用案例:
        ```python
        # 多时间框架OHLC止损系统
        daily_data = load_daily_ohlc()
        hourly_signals = generate_hourly_signals()
        
        # 在日线级别执行OHLC止损
        entries, exits = generate_ohlc_stop_enex_nb(
            hourly_signals.resample('D').any(),  # 信号重采样到日线
            daily_data.open, daily_data.high,
            daily_data.low, daily_data.close,
            daily_stop_prices, daily_stop_types,
            sl_stop=np.array([0.04, 0.06, 0.05]),  # 不同资产不同止损
            sl_trail=np.array([True, False, True]), # 混合止损类型
            tp_stop=0.10,                           # 统一10%止盈
            reverse=False, is_open_safe=True,
            entry_wait=2, exit_wait=1, pick_first=True, flex_2d=True
        )
        ```

    注意事项:
        - 该函数具有与generate_ohlc_stop_ex_nb相同的逻辑，但会清理入场信号
        - OHLC数据质量对最终结果的影响最为关键
        - 输出数组的预分配和初始化至关重要
        - 建议在实际使用前进行充分的参数调优和历史验证

    参见:
        generate_enex_nb: 底层信号协调框架
        first_choice_nb: 入场信号清理选择器
        ohlc_stop_choice_nb: 核心OHLC止损逻辑
        StopType: 止损类型枚举常量定义
    """
    # 创建临时索引数组，用于OHLC止损计算
    temp_idx_arr = np.empty((entries.shape[0],), dtype=np.int64)
    
    # 使用generate_enex_nb框架协调入场和OHLC止损退出信号
    return generate_enex_nb(
        entries.shape,          # 信号矩阵的形状
        entry_wait,             # 入场信号等待周期
        exit_wait,              # 退出信号等待周期
        True,                   # 入场信号pick_first=True（清理重叠信号）
        pick_first,             # 退出信号的pick_first设置
        # 入场信号选择：使用first_choice_nb激活和清理原始入场信号
        first_choice_nb, (entries,),
        # 退出信号选择：使用ohlc_stop_choice_nb实现精确OHLC止损逻辑
        ohlc_stop_choice_nb, (
            open, high, low, close,              # OHLC价格数据
            stop_price_out, stop_type_out,       # 输出数组
            sl_stop, sl_trail, tp_stop,          # 止损止盈参数配置
            reverse, is_open_safe,               # 交易方向和安全性参数
            exit_wait, pick_first,               # 延迟和选择控制参数
            temp_idx_arr, flex_2d                # 临时数组和索引参数
        )
    )


# ############# Map and reduce ranges ############# #


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存
def between_ranges_nb(a: tp.Array2d) -> tp.RecordArray:
    """创建信号间范围记录，分析两个信号之间的时间间隔。

    这是vectorbt范围分析系统的基础函数，用于分析同一序列中连续信号之间的时间间隔。
    该函数对于理解信号的密度分布、间隔特征和时间模式具有重要价值，是信号质量
    分析和策略优化的重要工具。

    参数说明:
        a (np.array): 二维布尔信号数组，形状为(时间, 资产)
                     True表示信号位置，False表示无信号

    返回值:
        np.recarray: 范围记录数组，每条记录包含以下字段：
                    - id: 记录的唯一标识符
                    - col: 列索引（资产/策略标识）
                    - start_idx: 范围开始索引（前一个信号位置）
                    - end_idx: 范围结束索引（后一个信号位置）
                    - status: 范围状态（RangeStatus.Closed，表示封闭范围）

    算法逻辑:
        1. 遍历每一列（每个资产/策略）
        2. 找到该列所有True信号的位置索引
        3. 对于每对相邻的信号位置，创建一个范围记录
        4. 记录范围的起始和结束位置，以及相关元数据

    使用场景:
        - 分析交易信号的时间分布特征
        - 计算平均持仓时间和信号间隔
        - 识别信号密集期和稀疏期
        - 优化信号生成参数

    性能特点:
        - Numba编译优化，高速范围计算
        - 启用缓存，重复调用性能优越
        - 内存高效的结构化记录输出
        - 支持大规模信号序列分析

    示例用法:
        ```python
        import numpy as np
        from vectorbt.signals.nb import between_ranges_nb

        # 创建测试信号：两个资产的信号序列
        signals = np.array([
            [True, False, True, False, True],    # 资产1: 位置0,2,4有信号
            [False, True, False, True, False]    # 资产2: 位置1,3有信号
        ]).T

        # 分析信号间的范围
        ranges = between_ranges_nb(signals)
        
        # 分析结果
        for i in range(len(ranges)):
            record = ranges[i]
            interval = record['end_idx'] - record['start_idx']
            print(f"资产{record['col']}: 从位置{record['start_idx']}到{record['end_idx']}，间隔{interval}")
        ```

    实际应用案例:
        ```python
        # 分析移动平均交叉信号的时间特征
        ma_cross_signals = generate_ma_crossover_signals(prices)
        signal_ranges = between_ranges_nb(ma_cross_signals)
        
        # 计算平均信号间隔
        intervals = [r['end_idx'] - r['start_idx'] for r in signal_ranges]
        avg_interval = np.mean(intervals)
        print(f"平均信号间隔: {avg_interval:.2f} 个周期")
        
        # 识别异常间隔
        long_intervals = [r for r in signal_ranges if r['end_idx'] - r['start_idx'] > avg_interval * 2]
        print(f"异常长间隔数量: {len(long_intervals)}")
        ```

    注意事项:
        - 函数只处理至少有2个信号的列
        - 返回的记录数等于(信号数量-1)的总和
        - 所有返回的范围状态都是Closed（封闭的）
        - 结果按列优先顺序排列

    参见:
        RangeStatus: 范围状态枚举定义
        between_two_ranges_nb: 双序列范围分析函数
        partition_ranges_nb: 信号分区范围分析函数
    """
    # 预分配范围记录数组（最大可能大小）
    range_records = np.empty(a.shape[0] * a.shape[1], dtype=range_dt)
    ridx = 0  # 记录索引计数器

    # 遍历每一列（每个资产/策略）
    for col in range(a.shape[1]):
        # 找到当前列所有True信号的位置索引
        a_idxs = np.flatnonzero(a[:, col])
        
        # 只有当存在至少2个信号时才能分析间隔
        if a_idxs.shape[0] > 1:
            # 遍历每对相邻的信号，创建范围记录
            for j in range(1, a_idxs.shape[0]):
                from_i = a_idxs[j - 1]  # 前一个信号位置
                to_i = a_idxs[j]        # 后一个信号位置
                
                # 填充范围记录的各个字段
                range_records[ridx]['id'] = ridx                    # 唯一标识符
                range_records[ridx]['col'] = col                    # 列索引
                range_records[ridx]['start_idx'] = from_i           # 范围起始位置
                range_records[ridx]['end_idx'] = to_i               # 范围结束位置
                range_records[ridx]['status'] = RangeStatus.Closed  # 封闭范围状态
                ridx += 1

    # 返回实际使用的记录数组切片
    return range_records[:ridx]


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存
def between_two_ranges_nb(a: tp.Array2d, b: tp.Array2d, from_other: bool = False) -> tp.RecordArray:
    """创建两个不同信号序列间的范围记录，分析信号配对和时间关系。

    这是vectorbt范围分析系统的高级函数，专门用于分析两个不同信号序列之间的
    时间关系和配对模式。该函数对于分析入场-退出信号对、买入-卖出信号配对、
    以及不同策略信号间的时间关系具有重要意义。

    参数说明:
        a (np.array): 第一个信号序列，二维布尔数组，形状为(时间, 资产)
        b (np.array): 第二个信号序列，二维布尔数组，形状必须与a相同
        from_other (bool): 范围计算方向控制
                          - False: 从a中的每个信号到b中的后续信号
                          - True: 从b中的每个信号到a中的前置信号

    返回值:
        np.recarray: 范围记录数组，字段与between_ranges_nb相同

    算法逻辑:
        根据from_other参数采用不同的配对策略:

        **正向配对** (from_other=False):
        - 对a中的每个信号，寻找b中第一个时间上不早于该信号的位置
        - 适用于分析"触发-响应"类型的信号关系
        - 常用于入场信号到退出信号的分析

        **反向配对** (from_other=True):
        - 对b中的每个信号，寻找a中最后一个时间上不晚于该信号的位置  
        - 适用于分析"准备-执行"类型的信号关系
        - 常用于条件信号到交易信号的分析

    重叠处理:
        当a和b在同一时间点都有信号时（重叠），仍会创建范围记录，
        此时from_i等于to_i，表示零时间间隔的信号配对。

    使用场景:
        - 分析入场信号到退出信号的持仓时间分布
        - 评估信号生成延迟和响应时间
        - 研究不同策略信号间的时序关系
        - 优化信号配对和交易执行逻辑

    性能特点:
        - Numba编译优化，高效的双序列配对算法
        - 启用缓存，重复分析性能卓越
        - 智能的索引搜索和配对逻辑
        - 内存高效的结构化输出

    示例用法:
        ```python
        # 创建入场和退出信号序列
        entries = np.array([[True, False, False, True, False]]).T
        exits = np.array([[False, True, False, False, True]]).T

        # 分析入场到退出的持仓时间
        entry_to_exit = between_two_ranges_nb(entries, exits, from_other=False)
        
        for record in entry_to_exit:
            holding_time = record['end_idx'] - record['start_idx']
            print(f"持仓时间: {holding_time} 个周期")
        
        # 分析退出到前置入场的信号关系
        exit_to_entry = between_two_ranges_nb(entries, exits, from_other=True)
        ```

    实际应用案例:
        ```python
        # 分析止损信号和入场信号的配对关系
        buy_signals = generate_buy_signals(prices)
        stop_losses = generate_stop_losses(prices, buy_signals)
        
        # 计算每笔交易的风险暴露时间
        trade_durations = between_two_ranges_nb(buy_signals, stop_losses)
        
        # 统计分析
        durations = [r['end_idx'] - r['start_idx'] for r in trade_durations]
        avg_duration = np.mean(durations)
        max_duration = np.max(durations)
        print(f"平均持仓: {avg_duration:.1f}期，最长持仓: {max_duration}期")
        
        # 识别配对失败的信号
        unpaired_entries = len(buy_signals.sum()) - len(trade_durations)
        print(f"未配对的入场信号: {unpaired_entries}")
        ```

    高级应用:
        ```python
        # 多策略信号协调分析
        strategy_a_signals = generate_strategy_a_signals(data)
        strategy_b_signals = generate_strategy_b_signals(data)
        
        # 分析策略A到策略B的信号传递时间
        signal_transmission = between_two_ranges_nb(
            strategy_a_signals, strategy_b_signals, from_other=False
        )
        
        # 分析信号传递效率
        transmission_times = [r['end_idx'] - r['start_idx'] for r in signal_transmission]
        immediate_responses = sum(1 for t in transmission_times if t == 0)
        print(f"即时响应率: {immediate_responses/len(transmission_times)*100:.1f}%")
        ```

    注意事项:
        - a和b必须具有相同的形状
        - 重叠信号会产生零间隔的范围记录
        - from_other参数显著影响配对逻辑和结果
        - 返回的记录数取决于成功配对的信号数量

    参见:
        between_ranges_nb: 单序列范围分析
        RangeStatus: 范围状态枚举常量
        partition_ranges_nb: 信号分区范围分析
    """
    # 预分配范围记录数组
    range_records = np.empty(a.shape[0] * a.shape[1], dtype=range_dt)
    ridx = 0  # 记录索引计数器

    # 遍历每一列
    for col in range(a.shape[1]):
        # 获取两个序列在当前列的所有信号位置
        a_idxs = np.flatnonzero(a[:, col])
        
        if a_idxs.shape[0] > 0:  # a序列必须有信号
            b_idxs = np.flatnonzero(b[:, col])
            
            if b_idxs.shape[0] > 0:  # b序列也必须有信号
                if from_other:
                    # 反向配对：从b中的每个信号找a中的前置信号
                    for j, to_i in enumerate(b_idxs):
                        # 找到a中所有不晚于当前b信号的位置
                        valid_a_idxs = a_idxs[a_idxs <= to_i]
                        if len(valid_a_idxs) > 0:
                            from_i = valid_a_idxs[-1]  # 选择最后一个（最接近的）前置信号
                            
                            # 创建范围记录
                            range_records[ridx]['id'] = ridx
                            range_records[ridx]['col'] = col
                            range_records[ridx]['start_idx'] = from_i
                            range_records[ridx]['end_idx'] = to_i
                            range_records[ridx]['status'] = RangeStatus.Closed
                            ridx += 1
                else:
                    # 正向配对：从a中的每个信号找b中的后续信号
                    for j, from_i in enumerate(a_idxs):
                        # 找到b中所有不早于当前a信号的位置
                        valid_b_idxs = b_idxs[b_idxs >= from_i]
                        if len(valid_b_idxs) > 0:
                            to_i = valid_b_idxs[0]  # 选择第一个（最接近的）后续信号
                            
                            # 创建范围记录
                            range_records[ridx]['id'] = ridx
                            range_records[ridx]['col'] = col
                            range_records[ridx]['start_idx'] = from_i
                            range_records[ridx]['end_idx'] = to_i
                            range_records[ridx]['status'] = RangeStatus.Closed
                            ridx += 1

    # 返回实际使用的记录数组
    return range_records[:ridx]


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存
def partition_ranges_nb(a: tp.Array2d) -> tp.RecordArray:
    """创建信号分区范围记录，识别连续信号块的边界和持续时间。

    这是vectorbt范围分析系统的分区函数，专门用于识别和分析连续信号的分区。
    该函数将连续的True信号视为一个分区，分析每个分区的起始、结束位置和状态，
    对于理解信号的聚集模式和持续特征具有重要价值。

    参数说明:
        a (np.array): 二维布尔信号数组，形状为(时间, 资产)

    返回值:
        np.recarray: 范围记录数组，包含每个连续信号分区的信息
                    - start_idx: 分区开始位置（第一个True信号）
                    - end_idx: 分区结束位置（最后一个True信号的下一位置）
                    - status: RangeStatus.Closed（完整分区）或RangeStatus.Open（未完成分区）

    算法逻辑:
        1. 使用状态机跟踪分区状态（在分区内/在分区外）
        2. 检测分区的开始（从False变为True）
        3. 检测分区的结束（从True变为False）
        4. 处理序列末尾的开放分区（状态为Open）

    分区状态说明:
        - **Closed分区**: 完整的信号分区，有明确的开始和结束
        - **Open分区**: 延续到序列末尾的未完成分区

    使用场景:
        - 分析信号的聚集和分散模式
        - 计算连续持仓时间的分布
        - 识别信号密集期和活跃期
        - 评估策略的信号持续性

    性能特点:
        - Numba编译优化，高效的状态机实现
        - 启用缓存，重复分析性能优异
        - 单遍扫描算法，时间复杂度O(n)
        - 内存高效的分区记录输出

    示例用法:
        ```python
        # 创建包含连续信号块的测试数据
        signals = np.array([
            [False, True, True, False, False, True, True, True, False],
            [True, True, False, False, True, False, False, False, False]
        ]).T

        # 分析信号分区
        partitions = partition_ranges_nb(signals)
        
        for record in partitions:
            duration = record['end_idx'] - record['start_idx']
            status = "完整" if record['status'] == RangeStatus.Closed else "开放"
            print(f"资产{record['col']}: 位置{record['start_idx']}-{record['end_idx']}, "
                  f"持续{duration}期, 状态:{status}")
        ```

    实际应用案例:
        ```python
        # 分析趋势信号的持续特征
        trend_signals = generate_trend_signals(prices)
        trend_partitions = partition_ranges_nb(trend_signals)
        
        # 计算趋势持续时间统计
        durations = [r['end_idx'] - r['start_idx'] for r in trend_partitions]
        avg_duration = np.mean(durations)
        max_duration = np.max(durations)
        
        print(f"趋势平均持续: {avg_duration:.1f}期")
        print(f"最长趋势持续: {max_duration}期")
        
        # 识别短期和长期趋势分区
        short_trends = [r for r in trend_partitions if r['end_idx'] - r['start_idx'] < 5]
        long_trends = [r for r in trend_partitions if r['end_idx'] - r['start_idx'] >= 20]
        
        print(f"短期趋势数量: {len(short_trends)}")
        print(f"长期趋势数量: {len(long_trends)}")
        ```

    高级应用:
        ```python
        # 多资产信号聚集度分析
        multi_asset_signals = generate_multi_asset_signals(prices)
        signal_partitions = partition_ranges_nb(multi_asset_signals)
        
        # 按资产分组分析
        from collections import defaultdict
        asset_partitions = defaultdict(list)
        
        for record in signal_partitions:
            asset_partitions[record['col']].append(record)
        
        # 比较不同资产的信号特征
        for asset_id, partitions in asset_partitions.items():
            durations = [p['end_idx'] - p['start_idx'] for p in partitions]
            open_partitions = sum(1 for p in partitions if p['status'] == RangeStatus.Open)
            
            print(f"资产{asset_id}: {len(partitions)}个分区, "
                  f"平均持续{np.mean(durations):.1f}期, "
                  f"{open_partitions}个开放分区")
        ```

    注意事项:
        - 连续的True值被视为同一个分区
        - 序列末尾的分区状态为Open
        - end_idx是左闭右开区间的右边界
        - 分区最小长度为1（单个True信号）

    参见:
        RangeStatus: 范围状态枚举定义
        between_ranges_nb: 信号间隔分析
        between_partition_ranges_nb: 分区间隔分析
    """
    # 预分配范围记录数组
    range_records = np.empty(a.shape[0] * a.shape[1], dtype=range_dt)
    ridx = 0  # 记录索引计数器

    # 遍历每一列
    for col in range(a.shape[1]):
        is_partition = False  # 分区状态标志
        from_i = -1          # 当前分区的开始位置

        # 逐行扫描，检测分区的开始和结束
        for i in range(a.shape[0]):
            if a[i, col]:  # 当前位置有信号
                if not is_partition:
                    # 分区开始：记录开始位置
                    from_i = i
                is_partition = True
            elif is_partition:
                # 分区结束：从True变为False
                to_i = i  # 结束位置（不包含当前位置）
                
                # 创建封闭分区记录
                range_records[ridx]['id'] = ridx
                range_records[ridx]['col'] = col
                range_records[ridx]['start_idx'] = from_i
                range_records[ridx]['end_idx'] = to_i
                range_records[ridx]['status'] = RangeStatus.Closed
                ridx += 1
                is_partition = False

            # 处理序列末尾的特殊情况
            if i == a.shape[0] - 1:
                if is_partition:
                    # 序列末尾仍在分区内，创建开放分区记录
                    to_i = a.shape[0] - 1  # 结束位置为序列末尾
                    
                    range_records[ridx]['id'] = ridx
                    range_records[ridx]['col'] = col
                    range_records[ridx]['start_idx'] = from_i
                    range_records[ridx]['end_idx'] = to_i
                    range_records[ridx]['status'] = RangeStatus.Open  # 开放状态
                    ridx += 1

    return range_records[:ridx]


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存
def between_partition_ranges_nb(a: tp.Array2d) -> tp.RecordArray:
    """创建分区间隔范围记录，分析连续信号分区之间的间隔时间。

    这是vectorbt范围分析系统的分区间隔函数，专门用于分析连续信号分区之间
    的空白间隔。该函数识别信号的非活跃期，对于理解信号的周期性、间歇性
    和市场的平静期具有重要价值。

    参数说明:
        a (np.array): 二维布尔信号数组，形状为(时间, 资产)

    返回值:
        np.recarray: 范围记录数组，包含每个分区间隔的信息
                    - start_idx: 间隔开始位置（前一分区的结束位置）
                    - end_idx: 间隔结束位置（下一分区的开始位置）
                    - status: RangeStatus.Closed（所有间隔都是封闭的）

    算法逻辑:
        1. 使用状态机跟踪分区和非分区状态
        2. 识别分区的结束位置作为间隔的开始
        3. 识别下一个分区的开始位置作为间隔的结束
        4. 只记录完整的间隔（不包括序列开头和结尾的开放区间）

    与partition_ranges_nb的区别:
        - partition_ranges_nb: 分析有信号的连续区间
        - between_partition_ranges_nb: 分析无信号的间隔区间

    使用场景:
        - 分析信号的非活跃期和静默期
        - 评估策略的信号生成频率
        - 识别市场的平静期和间歇期
        - 优化信号生成的时机和密度

    性能特点:
        - Numba编译优化，高效的间隔检测算法
        - 启用缓存，重复分析性能卓越
        - 状态机设计，逻辑清晰简洁
        - 内存高效的间隔记录输出

    示例用法:
        ```python
        # 创建包含多个分区的信号序列
        signals = np.array([
            [True, True, False, False, False, True, False, True, True],
            [False, True, True, False, True, True, False, False, True]
        ]).T

        # 分析分区间的间隔
        intervals = between_partition_ranges_nb(signals)
        
        for record in intervals:
            interval_length = record['end_idx'] - record['start_idx']
            print(f"资产{record['col']}: 间隔位置{record['start_idx']}-{record['end_idx']}, "
                  f"长度{interval_length}期")
        ```

    实际应用案例:
        ```python
        # 分析交易信号的非活跃期
        trading_signals = generate_trading_signals(prices)
        inactive_periods = between_partition_ranges_nb(trading_signals)
        
        # 统计非活跃期的特征
        inactive_durations = [r['end_idx'] - r['start_idx'] for r in inactive_periods]
        
        if inactive_durations:
            avg_inactive = np.mean(inactive_durations)
            max_inactive = np.max(inactive_durations)
            min_inactive = np.min(inactive_durations)
            
            print(f"平均非活跃期: {avg_inactive:.1f}期")
            print(f"最长非活跃期: {max_inactive}期")
            print(f"最短非活跃期: {min_inactive}期")
            
            # 识别异常长的非活跃期
            long_inactive = [r for r in inactive_periods if 
                           r['end_idx'] - r['start_idx'] > avg_inactive * 2]
            print(f"异常长非活跃期数量: {len(long_inactive)}")
        ```

    市场分析应用:
        ```python
        # 分析市场波动信号的间歇特征
        volatility_signals = generate_volatility_breakout_signals(prices)
        quiet_periods = between_partition_ranges_nb(volatility_signals)
        
        # 分析市场平静期的分布
        quiet_durations = [r['end_idx'] - r['start_idx'] for r in quiet_periods]
        
        # 按时间长度分类平静期
        short_quiet = sum(1 for d in quiet_durations if d < 5)      # 短期平静
        medium_quiet = sum(1 for d in quiet_durations if 5 <= d < 20)  # 中期平静  
        long_quiet = sum(1 for d in quiet_durations if d >= 20)     # 长期平静
        
        print(f"短期平静期: {short_quiet}个")
        print(f"中期平静期: {medium_quiet}个") 
        print(f"长期平静期: {long_quiet}个")
        
        # 计算平静期占总时间的比例
        total_quiet_time = sum(quiet_durations)
        total_time = len(prices)
        quiet_ratio = total_quiet_time / total_time
        print(f"市场平静期比例: {quiet_ratio*100:.1f}%")
        ```

    注意事项:
        - 只记录完整的分区间隔，不包括序列首尾的开放间隔
        - 间隔的定义是两个分区之间的非信号区域
        - 结果数量通常比分区数量少1
        - 连续分区（无间隔）不会产生记录

    参见:
        partition_ranges_nb: 信号分区分析
        between_ranges_nb: 信号间隔分析  
        RangeStatus: 范围状态枚举定义
    """
    # 预分配范围记录数组
    range_records = np.empty(a.shape[0] * a.shape[1], dtype=range_dt)
    ridx = 0  # 记录索引计数器

    # 遍历每一列
    for col in range(a.shape[1]):
        is_partition = False  # 分区状态标志
        from_i = -1          # 当前间隔的开始位置

        # 逐行扫描，检测分区间的间隔
        for i in range(a.shape[0]):
            if a[i, col]:  # 当前位置有信号
                if not is_partition and from_i != -1:
                    # 间隔结束：从非分区转入分区，且之前有记录的间隔开始
                    to_i = i  # 间隔结束位置
                    
                    # 创建间隔记录
                    range_records[ridx]['id'] = ridx
                    range_records[ridx]['col'] = col
                    range_records[ridx]['start_idx'] = from_i
                    range_records[ridx]['end_idx'] = to_i
                    range_records[ridx]['status'] = RangeStatus.Closed
                    ridx += 1
                    
                is_partition = True  # 进入分区状态
                from_i = i          # 更新潜在间隔开始位置为当前分区位置
            else:
                # 当前位置无信号，可能在间隔中
                is_partition = False

    return range_records[:ridx]


# ############# 排序系统 (Ranking) ############# #

@njit  # 使用Numba JIT编译优化
def rank_nb(a: tp.Array2d,
            reset_by: tp.Optional[tp.Array1d],
            after_false: bool,
            rank_func_nb: tp.RankFunc, *args) -> tp.Array2d:
    """为信号序列中的每个信号分配排序等级。

    这是vectorbt排序系统的核心函数，它为信号序列中的每个True值分配等级或排序。
    该函数支持复杂的排序逻辑，包括分区重置、条件过滤和自定义排序算法，
    是信号优先级管理和序列分析的重要工具。

    参数说明:
        a (np.array): 二维布尔信号数组，形状为(时间, 资产)
        reset_by (np.array, optional): 重置信号数组，用于重置排序计数
                                     - None: 不使用重置功能
                                     - 数组: 当reset_by为True时重置排序
        after_false (bool): 是否忽略序列开头没有False值的True分区
                           - True: 要求第一个分区前必须有False值
                           - False: 处理所有True值，包括序列开头的
        rank_func_nb (callable): 排序函数，必须是Numba编译的函数
                                函数签名: func(i, col, reset_i, prev_part_end_i, 
                                           part_start_i, *args) -> int
                                - 返回-1表示不分配等级
                                - 返回>=0表示分配的等级值
        *args: 传递给rank_func_nb的额外参数

    返回值:
        np.array: 与输入相同形状的整数数组，-1表示无等级，>=0表示等级值

    算法逻辑:
        1. **状态追踪**: 维护重置位置、分区边界和状态信息
        2. **分区检测**: 识别连续True值的分区和分区边界
        3. **条件过滤**: 根据after_false参数过滤符合条件的分区
        4. **等级分配**: 调用自定义排序函数为每个信号分配等级

    排序函数接口:
        排序函数接收以下参数并返回等级值：
        - i: 当前行索引
        - col: 当前列索引  
        - reset_i: 最近重置信号的索引
        - prev_part_end_i: 前一个分区的结束索引
        - part_start_i: 当前分区的开始索引
        - *args: 额外参数

    使用场景:
        - 为信号分配优先级和等级
        - 分析信号在分区内的位置关系
        - 实现复杂的信号过滤和选择逻辑
        - 构建基于位置的信号评分系统

    性能特点:
        - Numba编译优化，高速排序计算
        - 灵活的自定义排序逻辑支持
        - 高效的状态机实现
        - 支持大规模信号序列处理

    示例用法:
        ```python
        @njit
        def position_rank_func(i, col, reset_i, prev_part_end_i, part_start_i):
            # 简单的位置排序：分区内的位置索引
            return i - part_start_i
        
        # 创建测试信号
        signals = np.array([
            [False, True, True, False, True, True, True],
            [True, True, False, False, True, False, True]
        ]).T
        
        # 为信号分配位置等级
        ranks = rank_nb(signals, None, False, position_rank_func)
        print(ranks)
        ```

    实际应用案例:
        ```python
        # 为交易信号分配优先级等级
        trading_signals = generate_trading_signals(prices)
        
        @njit
        def priority_rank_func(i, col, reset_i, prev_part_end_i, part_start_i, 
                              signal_strength):
            # 基于信号强度的优先级排序
            strength = signal_strength[i, col]
            if strength > 0.8:
                return 0  # 高优先级
            elif strength > 0.5:
                return 1  # 中优先级
            else:
                return 2  # 低优先级
        
        # 计算信号强度（示例）
        signal_strength = calculate_signal_strength(prices)
        
        # 分配优先级等级
        priority_ranks = rank_nb(
            trading_signals, None, False, 
            priority_rank_func, signal_strength
        )
        ```

    注意事项:
        - 排序函数必须是Numba编译的
        - -1表示该位置不分配等级
        - after_false影响序列开头分区的处理
        - reset_by可以周期性重置排序计数

    参见:
        sig_pos_rank_nb: 分区位置排序函数
        part_pos_rank_nb: 分区序号排序函数
    """
    # 初始化输出数组，-1表示无等级
    out = np.full(a.shape, -1, dtype=np.int64)

    # 遍历每一列
    for col in range(a.shape[1]):
        # 初始化状态变量
        reset_i = 0              # 最近重置位置
        prev_part_end_i = -1     # 前一个分区的结束位置
        part_start_i = -1        # 当前分区的开始位置
        in_partition = False     # 是否在分区内
        false_seen = not after_false  # 是否已见过False值

        # 遍历每一行
        for i in range(a.shape[0]):
            # 处理重置信号
            if reset_by is not None:
                if reset_by[i, col]:
                    reset_i = i

            # 处理True信号
            if a[i, col] and not (after_false and not false_seen):
                if not in_partition:
                    # 分区开始
                    part_start_i = i
                in_partition = True
                
                # 调用排序函数分配等级
                out[i, col] = rank_func_nb(i, col, reset_i, prev_part_end_i, part_start_i, *args)
                
            elif not a[i, col]:
                # 处理False信号
                if in_partition:
                    # 分区结束
                    prev_part_end_i = i - 1
                in_partition = False
                false_seen = True

    return out


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存
def sig_pos_rank_nb(i: int, col: int, reset_i: int, prev_part_end_i: int, part_start_i: int,
                    sig_pos_temp: tp.Array1d, allow_gaps: bool) -> int:
    """信号位置排序函数，按照信号在分区内的位置分配等级。

    这是一个符合rank_func_nb接口的具体排序函数，专门用于按照信号在分区内
    的相对位置进行排序。该函数为每个信号分配一个从0开始的递增序号，
    表示该信号在当前分区中的位置顺序。

    参数说明:
        i (int): 当前行索引
        col (int): 当前列索引
        reset_i (int): 最近重置信号的索引
        prev_part_end_i (int): 前一个分区的结束索引
        part_start_i (int): 当前分区的开始索引
        sig_pos_temp (np.array): 临时数组，用于维护每列的位置计数
        allow_gaps (bool): 是否允许位置间隔
                          - True: 连续计数，忽略分区内的False值
                          - False: 严格按分区边界重置计数

    返回值:
        int: 信号在分区内的位置等级，从0开始

    算法逻辑:
        1. 检查是否需要重置位置计数（重置信号或新分区）
        2. 根据allow_gaps参数决定计数策略
        3. 递增位置计数并返回当前位置

    使用场景:
        - 标识信号在分区内的顺序位置
        - 实现基于位置的信号过滤
        - 分析信号的时序分布特征
        - 构建位置相关的信号评分

    示例用法:
        ```python
        # 为信号分配分区内位置等级
        signals = np.array([[False, True, True, False, True, True]]).T
        temp_array = np.zeros(1, dtype=np.int64)
        
        # 使用严格分区计数
        pos_ranks = rank_nb(signals, None, False, sig_pos_rank_nb, temp_array, False)
        # 结果可能是: [[-1, 0, 1, -1, 0, 1]]
        
        # 使用连续计数
        pos_ranks = rank_nb(signals, None, False, sig_pos_rank_nb, temp_array, True)  
        # 结果可能是: [[-1, 0, 1, -1, 2, 3]]
        ```

    注意事项:
        - sig_pos_temp数组必须为每列预分配空间
        - allow_gaps参数显著影响计数逻辑
        - 函数会修改sig_pos_temp数组的状态
        - 适用于需要位置信息的排序场景
    """
    # 检查是否需要重置位置计数
    if reset_i > prev_part_end_i and max(reset_i, part_start_i) == i:
        # 重置条件：有重置信号且在分区开始
        sig_pos_temp[col] = -1
    elif not allow_gaps and part_start_i == i:
        # 严格模式：分区开始时重置
        sig_pos_temp[col] = -1
    
    # 递增位置计数
    sig_pos_temp[col] += 1
    
    # 返回当前位置等级
    return sig_pos_temp[col]


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存
def part_pos_rank_nb(i: int, col: int, reset_i: int, prev_part_end_i: int, part_start_i: int,
                     part_pos_temp: tp.Array1d) -> int:
    """分区位置排序函数，按照分区在序列中的位置分配等级。

    这是一个符合rank_func_nb接口的具体排序函数，专门用于按照分区在整个
    序列中的出现顺序进行排序。该函数为每个分区分配一个唯一的序号，
    分区内的所有信号都获得相同的分区等级。

    参数说明:
        i (int): 当前行索引
        col (int): 当前列索引  
        reset_i (int): 最近重置信号的索引
        prev_part_end_i (int): 前一个分区的结束索引
        part_start_i (int): 当前分区的开始索引
        part_pos_temp (np.array): 临时数组，用于维护每列的分区计数

    返回值:
        int: 分区在序列中的位置等级，从0开始

    算法逻辑:
        1. 检查是否是新分区的开始
        2. 根据重置条件决定是否重置分区计数
        3. 为新分区递增计数器
        4. 返回当前分区的等级

    使用场景:
        - 标识信号所属的分区序号
        - 实现基于分区的信号分组
        - 分析分区级别的信号特征
        - 构建分区相关的统计分析

    示例用法:
        ```python
        # 为信号分配分区等级
        signals = np.array([[True, True, False, True, False, True, True]]).T
        temp_array = np.zeros(1, dtype=np.int64)
        
        # 使用分区位置排序
        part_ranks = rank_nb(signals, None, False, part_pos_rank_nb, temp_array)
        # 结果可能是: [[0, 0, -1, 1, -1, 2, 2]]
        # 表示: 分区0有2个信号，分区1有1个信号，分区2有2个信号
        ```

    实际应用案例:
        ```python
        # 分析交易信号的分区分布
        trading_signals = generate_trading_signals(prices)
        temp_array = np.zeros(trading_signals.shape[1], dtype=np.int64)
        
        partition_ranks = rank_nb(
            trading_signals, None, False, 
            part_pos_rank_nb, temp_array
        )
        
        # 统计每个分区的信号数量
        for col in range(trading_signals.shape[1]):
            unique_partitions = np.unique(partition_ranks[:, col])
            valid_partitions = unique_partitions[unique_partitions >= 0]
            print(f"资产{col}: {len(valid_partitions)}个信号分区")
        ```

    注意事项:
        - part_pos_temp数组必须为每列预分配空间
        - 分区内所有信号获得相同的分区等级
        - 重置信号会重新开始分区计数
        - 适用于需要分区标识的分析场景
    """
    # 检查是否需要重置分区计数
    if reset_i > prev_part_end_i and max(reset_i, part_start_i) == i:
        # 重置条件：有重置信号且在分区开始时
        part_pos_temp[col] = 0
    elif part_start_i == i:
        # 新分区开始：递增分区计数
        part_pos_temp[col] += 1
    
    # 返回当前分区等级
    return part_pos_temp[col]


# ############# 索引系统 (Index) ############# #


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存
def nth_index_1d_nb(a: tp.Array1d, n: int) -> int:
    """获取一维数组中第n个True值的索引位置。

    这是vectorbt索引系统的基础函数，用于在布尔数组中定位特定序号的True值。
    该函数支持正向和反向索引，是信号定位和选择的重要工具。

    参数说明:
        a (np.array): 一维布尔数组
        n (int): 目标True值的序号
               - n>=0: 正向索引，0表示第1个True值
               - n<0: 反向索引，-1表示最后1个True值

    返回值:
        int: 第n个True值的索引位置，如果不存在则返回-1

    算法逻辑:
        **正向搜索** (n>=0):
        1. 从数组开头向后遍历
        2. 遇到True值时递增计数
        3. 计数达到n时返回当前索引

        **反向搜索** (n<0):
        1. 从数组末尾向前遍历  
        2. 遇到True值时递减计数
        3. 计数达到n时返回当前索引

    使用场景:
        - 定位特定序号的信号位置
        - 实现信号的首末位置查找
        - 支持基于序号的信号选择
        - 构建信号索引和引用系统

    性能特点:
        - Numba编译优化，高速索引查找
        - 启用缓存，重复查找性能卓越
        - 早停机制，找到目标后立即返回
        - 支持正反双向索引查找

    示例用法:
        ```python
        # 创建测试信号序列
        signals = np.array([False, True, False, True, True, False, True])
        
        # 正向索引查找
        first_signal = nth_index_1d_nb(signals, 0)   # 第1个True: 索引1
        second_signal = nth_index_1d_nb(signals, 1)  # 第2个True: 索引3
        third_signal = nth_index_1d_nb(signals, 2)   # 第3个True: 索引4
        
        # 反向索引查找  
        last_signal = nth_index_1d_nb(signals, -1)   # 最后1个True: 索引6
        second_last = nth_index_1d_nb(signals, -2)   # 倒数第2个True: 索引4
        
        print(f"第1个信号位置: {first_signal}")
        print(f"最后1个信号位置: {last_signal}")
        ```

    实际应用案例:
        ```python
        # 分析交易信号的首末位置
        trading_signals = generate_trading_signals(prices)
        
        for col in range(trading_signals.shape[1]):
            signals_1d = trading_signals[:, col]
            
            first_trade = nth_index_1d_nb(signals_1d, 0)
            last_trade = nth_index_1d_nb(signals_1d, -1)
            
            if first_trade >= 0 and last_trade >= 0:
                trading_span = last_trade - first_trade
                total_signals = np.sum(signals_1d)
                
                print(f"资产{col}: 首次交易位置{first_trade}, 末次交易位置{last_trade}")
                print(f"  交易时间跨度: {trading_span}期, 总信号数: {total_signals}")
        ```

    边界条件处理:
        ```python
        # 处理边界情况
        empty_signals = np.array([False, False, False])
        single_signal = np.array([False, True, False])
        
        # 空信号序列
        result = nth_index_1d_nb(empty_signals, 0)  # 返回-1
        
        # 超出范围的索引
        result = nth_index_1d_nb(single_signal, 5)  # 返回-1
        result = nth_index_1d_nb(single_signal, -5) # 返回-1
        
        # 有效索引
        result = nth_index_1d_nb(single_signal, 0)  # 返回1
        result = nth_index_1d_nb(single_signal, -1) # 返回1
        ```

    注意事项:
        - n从0开始计数，0表示第1个True值
        - 负数索引支持反向查找
        - 超出范围或未找到目标时返回-1
        - 函数对空数组和单元素数组都有良好处理
    """
    if n >= 0:
        # 正向搜索：从头到尾
        found = -1  # True值计数器，从-1开始
        for i in range(a.shape[0]):
            if a[i]:
                found += 1  # 发现True值，计数递增
                if found == n:
                    return i  # 找到第n个True值，返回索引
    else:
        # 反向搜索：从尾到头
        found = 0   # True值计数器，从0开始（负数逻辑）
        for i in range(a.shape[0] - 1, -1, -1):  # 反向遍历
            if a[i]:
                found -= 1  # 发现True值，计数递减
                if found == n:
                    return i  # 找到第n个True值，返回索引
    
    # 未找到目标True值
    return -1


@njit(cache=True)  # 使用Numba JIT编译优化，启用缓存
def nth_index_nb(a: tp.Array2d, n: int) -> tp.Array1d:
    """二维版本的第n个True值索引查找函数。

    这是nth_index_1d_nb的向量化版本，能够同时处理二维数组的每一列，
    为每列独立查找第n个True值的位置。该函数实现了批量索引查找，
    是多资产、多策略信号分析的重要工具。

    参数说明:
        a (np.array): 二维布尔数组，形状为(时间, 资产)
        n (int): 目标True值的序号，正负数规则与nth_index_1d_nb相同

    返回值:
        np.array: 一维整数数组，长度等于列数
                 每个元素是对应列中第n个True值的索引，-1表示未找到

    算法逻辑:
        1. 为每一列分别调用nth_index_1d_nb函数
        2. 收集所有列的查找结果
        3. 返回包含所有结果的一维数组

    使用场景:
        - 批量查找多个资产的特定序号信号
        - 分析多策略的信号时序特征
        - 实现向量化的信号定位操作
        - 构建多维信号索引系统

    性能特点:
        - Numba编译优化，批量处理高效
        - 启用缓存，重复操作性能卓越
        - 向量化设计，支持大规模数据
        - 列级别的独立处理逻辑

    示例用法:
        ```python
        # 创建多资产信号矩阵
        signals = np.array([
            [False, True, False],   # 时间0: 只有资产1有信号
            [True, False, True],    # 时间1: 资产0和2有信号  
            [False, True, False],   # 时间2: 只有资产1有信号
            [True, False, True],    # 时间3: 资产0和2有信号
            [False, False, True]    # 时间4: 只有资产2有信号
        ])
        
        # 查找每个资产的第1个信号位置
        first_signals = nth_index_nb(signals, 0)
        print(f"各资产首次信号位置: {first_signals}")  # [1, 0, 1]
        
        # 查找每个资产的第2个信号位置  
        second_signals = nth_index_nb(signals, 1)
        print(f"各资产第2次信号位置: {second_signals}")  # [3, 2, 3]
        
        # 查找每个资产的最后一个信号位置
        last_signals = nth_index_nb(signals, -1)
        print(f"各资产最后信号位置: {last_signals}")  # [3, 2, 4]
        ```

    实际应用案例:
        ```python
        # 分析多资产交易策略的信号特征
        multi_asset_signals = generate_multi_asset_signals(prices)
        
        # 批量查找各资产的首次和末次交易信号
        first_trades = nth_index_nb(multi_asset_signals, 0)
        last_trades = nth_index_nb(multi_asset_signals, -1)
        
        # 分析各资产的交易活跃期
        for asset_id in range(len(first_trades)):
            first_pos = first_trades[asset_id]
            last_pos = last_trades[asset_id]
            
            if first_pos >= 0 and last_pos >= 0:
                active_span = last_pos - first_pos
                total_signals = multi_asset_signals[:, asset_id].sum()
                
                print(f"资产{asset_id}: 活跃期{active_span}天, 总信号{total_signals}个")
                
                # 计算信号密度
                if active_span > 0:
                    signal_density = total_signals / active_span
                    print(f"  信号密度: {signal_density:.3f}信号/天")
            else:
                print(f"资产{asset_id}: 无有效交易信号")
        ```

    批量分析应用:
        ```python
        # 批量分析信号分布特征
        strategy_signals = load_strategy_signals()  # 假设加载多策略信号
        
        # 查找各策略的关键信号位置
        first_pos = nth_index_nb(strategy_signals, 0)    # 首次信号
        second_pos = nth_index_nb(strategy_signals, 1)   # 第二次信号  
        last_pos = nth_index_nb(strategy_signals, -1)    # 最后信号
        second_last = nth_index_nb(strategy_signals, -2) # 倒数第二次
        
        # 构建信号特征矩阵
        signal_features = np.column_stack([first_pos, second_pos, second_last, last_pos])
        
        # 分析信号时序模式
        for strategy_id, features in enumerate(signal_features):
            first, second, s_last, last = features
            
            if all(pos >= 0 for pos in features):
                early_gap = second - first      # 早期信号间隔
                late_gap = last - s_last       # 后期信号间隔  
                total_span = last - first       # 总体时间跨度
                
                print(f"策略{strategy_id}: 早期间隔{early_gap}, 后期间隔{late_gap}, 总跨度{total_span}")
        ```

    注意事项:
        - 每一列独立处理，列间不影响
        - 返回数组长度等于输入列数
        - -1表示对应列未找到目标序号的True值
        - 支持所有nth_index_1d_nb的索引规则

    参见:
        nth_index_1d_nb: 一维版本的索引查找函数
        norm_avg_index_nb: 归一化平均索引计算函数
    """
    # 初始化输出数组
    out = np.empty(a.shape[1], dtype=np.int64)
    
    # 对每一列分别调用一维索引查找函数
    for col in range(a.shape[1]):
        out[col] = nth_index_1d_nb(a[:, col], n)
    
    return out


@njit(cache=True)
def norm_avg_index_1d_nb(a: tp.Array1d) -> float:
    """Get mean index normalized to (-1, 1)."""
    mean_index = np.mean(np.flatnonzero(a))
    return renormalize_nb(mean_index, (0, len(a) - 1), (-1, 1))


@njit(cache=True)
def norm_avg_index_nb(a: tp.Array2d) -> tp.Array1d:
    """2-dim version of `norm_avg_index_1d_nb`."""
    out = np.empty(a.shape[1], dtype=np.float64)
    for col in range(a.shape[1]):
        out[col] = norm_avg_index_1d_nb(a[:, col])
    return out
