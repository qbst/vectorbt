# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT 随机数生成工具模块 - 统一的随机种子管理和控制系统
================================================================================

文件作用概述：
本模块是vectorbt量化交易框架中专门负责随机数生成和种子管理的核心工具模块。
该模块提供了统一的随机种子控制接口，确保在量化分析、回测和模拟过程中
能够获得可重复、可验证的随机结果。

"""

import random

import numpy as np
from numba import njit


@njit(cache=True)
def set_seed_nb(seed: int) -> None:
    """
    在Numba编译环境中设置随机种子
    
    该函数专门用于在Numba Just-In-Time编译环境中设置NumPy随机数生成器的种子。
    由于Numba编译的函数运行在独立的环境中，需要专门的函数来设置随机种子，
    以确保在高性能计算场景下的随机数生成具有可重复性。
    
    参数说明：
        seed (int): 随机种子值，必须为整数
            - 相同的种子值将产生相同的随机数序列
            - 不同的种子值将产生不同的随机数序列
            - 建议使用正整数，如42、123、2024等
    
    返回值：
        None: 该函数不返回任何值，仅设置随机种子状态
    
    使用示例：
        >>> import vectorbt as vbt
        >>> from numba import njit
        >>> import numpy as np
        
        >>> # 示例1：在Numba函数中使用随机种子
        >>> @njit
        ... def generate_random_returns(n_periods, seed_value):
        ...     vbt.utils.random_.set_seed_nb(seed_value)
        ...     return np.random.normal(0.001, 0.02, n_periods)
        
        >>> # 生成可重复的随机收益率序列
        >>> returns1 = generate_random_returns(100, 42)
        >>> returns2 = generate_random_returns(100, 42)
        >>> print(f"两次生成的序列相同: {np.allclose(returns1, returns2)}")
    
    技术实现：
        - 直接调用np.random.seed()在Numba环境中设置NumPy随机种子
        - 由于Numba编译的限制，只能设置NumPy随机数生成器
        - 不影响Python标准库random模块的状态
    """
    np.random.seed(seed)  # 在Numba环境中设置NumPy随机数生成器的种子


def set_seed(seed: int) -> None:
    """
    统一设置所有随机数生成器的种子
    
    该函数是vectorbt框架中随机种子管理的核心入口点，它会同时设置Python标准库、
    NumPy和Numba环境中的所有随机数生成器种子，确保在整个量化分析流程中
    获得完全一致和可重复的随机数序列。
    
    功能特点：
    - **全覆盖设置**：同时设置Python、NumPy、Numba三个环境的随机种子
    - **一致性保证**：确保所有随机数生成器使用相同的种子值
    - **简单易用**：单一函数调用即可完成所有环境的种子设置
    - **框架集成**：与vectorbt的全局设置系统完美集成
    
    参数说明：
        seed (int): 随机种子值，必须为整数
            - 推荐使用有意义的整数，如日期、版本号等
            - 常用种子值：42（经典选择）、123、2024等
            - 避免使用0或负数作为种子值
    
    返回值：
        None: 该函数不返回任何值，仅设置全局随机种子状态
    
    实现原理：
        1. 设置Python标准库random模块的种子
        2. 设置NumPy随机数生成器的种子
        3. 调用set_seed_nb设置Numba环境的种子
        4. 确保所有环境使用相同的种子值
    
    """
    random.seed(seed)  # 设置Python标准库random模块的随机种子
    np.random.seed(seed)  # 设置NumPy随机数生成器的种子
    set_seed_nb(seed)  # 调用Numba编译的函数设置Numba环境的随机种子
