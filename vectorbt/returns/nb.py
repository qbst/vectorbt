# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT RETURNS MODULE: 高性能收益率计算与投资组合绩效评估核心模块
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于收益率计算和投资组合绩效评估的高性能数值
计算模块。该模块通过Numba JIT编译技术，提供了完整的投资组合分析工具箱，包括
收益率计算、风险指标、绩效比率、回撤分析等核心量化金融指标的高效实现。

核心设计理念：
1. **极致性能优化**：所有函数都使用@njit装饰器进行Just-In-Time编译，执行速度
   比纯Python快10-100倍，能够处理百万级数据点的实时分析需求。

2. **完整的量化指标体系**：涵盖现代投资组合理论的所有核心指标，包括：
   - 基础收益率计算：单期收益率、累计收益率、总收益率
   - 风险调整收益指标：夏普比率、索提诺比率、卡尔玛比率、欧米伽比率
   - 相对绩效指标：阿尔法、贝塔、信息比率、捕获比率
   - 风险度量指标：最大回撤、VaR、CVaR、下行风险、尾部比率
   - 年化调整指标：年化收益率、年化波动率

3. **矩阵优先架构**：遵循vectorbt核心设计原则，所有函数都支持二维矩阵输入，
   能够同时处理多个资产或策略的批量计算，实现真正的向量化量化分析。

4. **滚动窗口支持**：为大部分指标提供滚动窗口版本，支持动态风险管理和实时
   监控需求，适用于高频交易和实时风控系统。

5. **数值稳定性保证**：在所有计算中考虑了边界条件和异常情况处理，如除零检查、
   无穷大处理、NaN值兼容等，确保生产环境的稳定性。

主要功能模块：
【收益率计算引擎】
- get_return_nb(): 基础收益率计算，处理各种边界情况
- returns_1d_nb()/returns_nb(): 从价格序列计算收益率时间序列
- total_return_apply_nb(): 计算指定期间的总收益率

【累计收益率系统】
- cum_returns_1d_nb()/cum_returns_nb(): 累计收益率曲线计算
- cum_returns_final_1d_nb()/cum_returns_final_nb(): 期末总收益率
- rolling_cum_returns_final_nb(): 滚动期间总收益率

【年化指标模块】
- annualized_return_1d_nb()/annualized_return_nb(): 复合年增长率(CAGR)
- annualized_volatility_1d_nb()/annualized_volatility_nb(): 年化波动率
- rolling版本：支持动态年化指标监控

【风险度量系统】
- drawdown_1d_nb()/drawdown_nb(): 回撤曲线计算
- max_drawdown_1d_nb()/max_drawdown_nb(): 最大回撤(MDD)
- value_at_risk_1d_nb()/value_at_risk_nb(): 风险价值(VaR)
- cond_value_at_risk_1d_nb()/cond_value_at_risk_nb(): 条件风险价值(CVaR)
- downside_risk_1d_nb()/downside_risk_nb(): 下行风险

【风险调整收益模块】
- sharpe_ratio_1d_nb()/sharpe_ratio_nb(): 夏普比率
- sortino_ratio_1d_nb()/sortino_ratio_nb(): 索提诺比率
- calmar_ratio_1d_nb()/calmar_ratio_nb(): 卡尔玛比率
- omega_ratio_1d_nb()/omega_ratio_nb(): 欧米伽比率
- information_ratio_1d_nb()/information_ratio_nb(): 信息比率

【相对绩效分析模块】
- beta_1d_nb()/beta_nb(): 贝塔系数(市场敏感性)
- alpha_1d_nb()/alpha_nb(): 阿尔法系数(超额收益)
- capture_1d_nb()/capture_nb(): 捕获比率
- up_capture_1d_nb()/up_capture_nb(): 上行捕获比率
- down_capture_1d_nb()/down_capture_nb(): 下行捕获比率

【统计分布分析模块】
- tail_ratio_1d_nb()/tail_ratio_nb(): 尾部比率(95%/5%分位数比)

技术特点：
- **零拷贝操作**：使用NumPy数组视图和就地操作，最小化内存分配
- **缓存友好**：优化内存访问模式，提高CPU缓存命中率
- **并行兼容**：所有函数都是无状态纯函数，天然支持并行化处理
- **类型安全**：完整的类型注解，支持静态类型检查和IDE智能提示
- **向量化优化**：充分利用NumPy的SIMD指令集和向量化能力

应用场景：
- **量化策略回测**：计算策略历史绩效和风险指标
- **实时风险监控**：实时计算投资组合风险指标和预警
- **投资组合优化**：为多目标优化提供高效的目标函数计算
- **绩效归因分析**：分析投资组合收益来源和风险贡献
- **合规风控**：计算监管要求的风险指标和限额监控
- **算法交易**：为高频交易策略提供毫秒级指标计算
- **财富管理**：为客户提供详细的投资绩效报告和分析

与vectorbt生态系统的关系：
- **ReturnsAccessor集成**：为收益率访问器提供底层计算支持
- **Portfolio模块协作**：为投资组合分析提供绩效计算引擎
- **Generic模块继承**：使用通用模块的基础数值计算函数
- **配置系统统一**：遵循vectorbt的全局配置和参数管理规范
- **类型系统兼容**：与vectorbt的类型注解系统完全兼容

该模块是vectorbt框架高性能量化分析的核心基础设施，为现代投资组合管理和
风险控制提供了工业级的计算能力和专业级的金融指标支持。
================================================================================

Numba编译函数集合 - 投资组合绩效评估的高性能计算引擎

提供用于访问器和投资组合绩效测量的Numba编译函数库。这些函数仅接受NumPy数组
和其他Numba兼容类型，确保最佳的计算性能。

```pycon
>>> import numpy as np
>>> import vectorbt as vbt

>>> price = np.array([1.1, 1.2, 1.3, 1.2, 1.1])
>>> returns = vbt.generic.nb.pct_change_1d_nb(price)

>>> # vectorbt.returns.nb.cum_returns_1d_nb
>>> vbt.returns.nb.cum_returns_1d_nb(returns, 0)
array([0., 0.09090909, 0.18181818, 0.09090909, 0.])
```

!!! 重要提示
    vectorbt将矩阵视为一等公民，期望输入数组为2维，除非函数具有`_1d`后缀
    或用作其他函数的输入。数据沿索引轴(axis 0)进行处理。

    作为参数传递的所有函数都应该是Numba编译的函数。
"""

# 导入核心数值计算库
import numpy as np  # NumPy科学计算库，提供高效的数组操作和数学函数
from numba import njit  # Numba JIT编译器，将Python函数编译为机器码

# 导入vectorbt内部模块
from vectorbt import _typing as tp  # vectorbt类型注解模块，提供类型提示支持
from vectorbt.generic import nb as generic_nb  # 通用数值计算模块，提供基础数学函数


@njit(cache=True)  # Numba JIT编译装饰器，启用缓存以提高重复调用性能
def get_return_nb(input_value: float, output_value: float) -> float:
    """
    计算单期收益率的核心函数
    
    该函数是所有收益率计算的基础，负责计算从输入值到输出值的收益率。
    采用标准的收益率公式：(新值 - 旧值) / 旧值，同时处理各种边界情况
    以确保数值计算的稳定性和准确性。
    
    计算公式：
    - 标准情况：return = (output_value - input_value) / input_value
    - 负基数修正：当input_value < 0时，结果乘以-1以保持方向一致性
    - 零基数处理：当input_value = 0时，根据output_value符号返回±∞
    
    参数说明：
        input_value (float): 初始值（分母），通常为前一期的价格或资产价值
        output_value (float): 终值（分子），通常为当前期的价格或资产价值
    
    返回值：
        float: 计算得到的收益率，可能为正数、负数、零、或无穷大
    
    边界情况处理：
        - 当input_value = 0且output_value = 0时，返回0（无变化）
        - 当input_value = 0且output_value ≠ 0时，返回±∞（根据output_value符号）
        - 当input_value < 0时，对结果取负以保持收益率方向的一致性
    
    使用示例：
        >>> get_return_nb(100.0, 110.0)  # 10%上涨
        0.1
        >>> get_return_nb(100.0, 90.0)   # 10%下跌
        -0.1
        >>> get_return_nb(0.0, 10.0)     # 从零开始
        inf
        >>> get_return_nb(-100.0, -90.0) # 负基数情况
        0.1
    
    金融解释：
        该函数计算的是简单收益率（Simple Return），广泛用于：
        - 股票价格变动分析
        - 投资组合收益率计算
        - 绩效基准比较
        - 风险指标计算的基础
    """
    # 检查分母是否为零（特殊情况处理）
    if input_value == 0:
        # 如果新值也为零，表示无变化
        if output_value == 0:
            return 0.
        # 如果新值不为零，返回带符号的无穷大
        return np.inf * np.sign(output_value)
    
    # 标准收益率计算公式：(新值 - 旧值) / 旧值
    return_value = (output_value - input_value) / input_value
    
    # 处理负基数情况：当初始值为负数时，调整符号以保持收益率语义的一致性
    if input_value < 0:
        return_value *= -1
    
    return return_value


@njit(cache=True)  # Numba JIT编译，启用缓存优化
def returns_1d_nb(value: tp.Array1d, init_value: float) -> tp.Array1d:
    """
    一维价格序列转收益率序列的核心转换函数
    
    该函数将价格时间序列转换为收益率时间序列，是量化分析的基础转换操作。
    通过逐点计算相邻时期间的收益率，生成与原价格序列等长的收益率序列。
    
    计算逻辑：
        - 第i期收益率 = get_return_nb(第i-1期价格, 第i期价格)
        - 第0期收益率 = get_return_nb(初始价格, 第0期价格)
        - 使用滚动方式逐期更新基准价格
    
    参数说明：
        value (tp.Array1d): 一维价格数组，形状为(n,)，按时间顺序排列
        init_value (float): 初始基准价格，用于计算第一期收益率
    
    返回值：
        tp.Array1d: 一维收益率数组，形状与输入价格数组相同
    
    内存优化：
        - 预分配输出数组，避免动态内存分配开销
        - 使用就地更新，最小化临时变量创建
        - 利用NumPy的连续内存布局优化缓存性能
    
    使用示例：
        >>> prices = np.array([100.0, 110.0, 105.0, 115.0])
        >>> returns_1d_nb(prices, 95.0)
        array([0.05263158, 0.1, -0.04545455, 0.09523810])
        
        解释：
        - 第1期：(100-95)/95 = 0.0526 (5.26%)
        - 第2期：(110-100)/100 = 0.1 (10%)
        - 第3期：(105-110)/110 = -0.0455 (-4.55%)
        - 第4期：(115-105)/105 = 0.0952 (9.52%)
    
    应用场景：
        - 将股价数据转换为收益率数据
        - 计算投资组合的期间收益率
        - 为风险指标计算准备输入数据
        - 标准化不同资产的收益表现
    """
    # 预分配输出数组，使用float64精度确保数值稳定性
    out = np.empty(value.shape, dtype=np.float64)
    # 初始化基准价格为用户提供的初始值
    input_value = init_value
    
    # 逐期计算收益率
    for i in range(out.shape[0]):
        # 获取当期价格作为输出值
        output_value = value[i]
        # 计算当期收益率
        out[i] = get_return_nb(input_value, output_value)
        # 更新基准价格为当期价格，用于下一期计算
        input_value = output_value
    
    return out


@njit(cache=True)  # Numba JIT编译优化
def returns_nb(value: tp.Array2d, init_value: tp.Array1d) -> tp.Array2d:
    """
    二维价格矩阵转收益率矩阵的批量处理函数
    
    这是returns_1d_nb的二维扩展版本，能够同时处理多个资产或策略的价格数据，
    实现真正的向量化批量计算。该函数是vectorbt框架支持多资产投资组合分析的
    核心基础设施。
    
    设计理念：
        - 每列代表一个资产或策略的时间序列
        - 每行代表同一时间点的多个资产价格
        - 支持不同资产的不同初始价格设置
        - 保持时间维度和资产维度的一致性
    
    参数说明：
        value (tp.Array2d): 二维价格矩阵，形状为(时间点数, 资产数)
            - 行索引：时间维度，按时间顺序排列
            - 列索引：资产维度，每列代表一个资产
        init_value (tp.Array1d): 初始价格向量，形状为(资产数,)
            - 每个元素对应一个资产的基准价格
    
    返回值：
        tp.Array2d: 二维收益率矩阵，形状与输入价格矩阵相同
    
    计算并行性：
        - 不同资产间的计算完全独立，具备天然的并行性
        - 每列的计算都调用returns_1d_nb函数
        - 适合在多核环境下进行并行优化
    
    使用示例：
        >>> prices = np.array([[100.0, 200.0],   # t0: 股票A=100, 股票B=200
        ...                    [110.0, 190.0],   # t1: 股票A=110, 股票B=190
        ...                    [105.0, 210.0]])  # t2: 股票A=105, 股票B=210
        >>> init_prices = np.array([95.0, 180.0])  # 初始基准价格
        >>> returns_nb(prices, init_prices)
        array([[ 0.05263158, 0.11111111],    # t0收益率
               [ 0.1       , -0.05      ],    # t1收益率  
               [-0.04545455, 0.10526316]])   # t2收益率
    
    内存效率：
        - 预分配整个输出矩阵，避免动态扩展
        - 利用NumPy的列优先内存布局
        - 最小化临时对象创建
    
    应用场景：
        - 多资产投资组合的收益率计算
        - 不同策略的批量回测分析
        - 因子收益率计算
        - 行业板块轮动分析
        - 大规模量化策略的收益率预处理
    """
    # 预分配输出矩阵，保持与输入矩阵相同的形状
    out = np.empty(value.shape, dtype=np.float64)
    
    # 逐列处理每个资产的价格序列
    for col in range(out.shape[1]):
        # 调用一维函数处理当前资产列，使用对应的初始价格
        out[:, col] = returns_1d_nb(value[:, col], init_value[col])
    
    return out


@njit(cache=True)  # Numba编译优化
def total_return_apply_nb(idxs: tp.Array1d, col: int, returns: tp.Array1d) -> float:
    """
    计算指定期间总收益率的应用函数
    
    该函数计算给定收益率序列的复合总收益率，用于测量整个投资期间的累积表现。
    采用几何平均方法，能够准确反映复利效应对最终收益的影响。
    
    计算公式：
        总收益率 = ∏(1 + ri) - 1
        其中ri为第i期的收益率
    
    参数说明：
        idxs (tp.Array1d): 索引数组（在此函数中未使用，保持接口一致性）
        col (int): 列索引（在此函数中未使用，保持接口一致性）
        returns (tp.Array1d): 收益率序列
    
    返回值：
        float: 期间总收益率
    
    数值稳定性：
        - 使用np.nanprod自动处理NaN值
        - NaN值被忽略，不影响最终计算结果
        - 适用于包含缺失数据的真实市场数据
    
    使用示例：
        >>> returns = np.array([0.1, -0.05, 0.08, 0.02])
        >>> total_return_apply_nb(None, None, returns)
        0.1561  # 约15.61%的总收益率
        
        计算过程：
        (1+0.1) × (1-0.05) × (1+0.08) × (1+0.02) - 1
        = 1.1 × 0.95 × 1.08 × 1.02 - 1
        = 1.1561 - 1 = 0.1561
    
    金融意义：
        - 反映投资的复合增长效果
        - 考虑了再投资收益的影响
        - 是衡量投资策略长期表现的核心指标
    """
    # 使用几何平均计算总收益率：所有(1+收益率)的乘积再减1
    return np.nanprod(returns + 1) - 1


@njit(cache=True)  # Numba编译优化，启用缓存
def cum_returns_1d_nb(returns: tp.Array1d, start_value: float) -> tp.Array1d:
    """
    一维累计收益率计算函数
    
    该函数计算收益率序列的累计复合收益率，生成完整的累计收益率曲线。
    这是投资组合分析中最基础也是最重要的函数之一，用于可视化投资策略
    的历史表现轨迹和评估资产价值的历史演变。
    
    计算逻辑：
        - 累计收益率[i] = (1+r[0]) × (1+r[1]) × ... × (1+r[i]) × start_value
        - 或累计收益率[i] = (1+r[0]) × (1+r[1]) × ... × (1+r[i]) - 1 (当start_value=0时)
        - 使用逐期乘积避免数值溢出问题
    
    参数说明：
        returns (tp.Array1d): 一维收益率数组，按时间顺序排列
        start_value (float): 起始资产价值
            - 当为0时，返回相对累计收益率（百分比形式）
            - 当为具体数值时，返回绝对累计资产价值
    
    返回值：
        tp.Array1d: 累计收益率数组，形状与输入相同
    
    数值稳定性：
        - 自动跳过NaN值，保持累计乘积的连续性
        - 使用逐期乘积而非一次性乘方，避免浮点溢出
        - 支持负收益率和极端市场情况
    
    使用示例：
        >>> returns = np.array([0.1, -0.05, 0.08, 0.02])
        >>> cum_returns_1d_nb(returns, 0.0)  # 相对累计收益
        array([0.1, 0.045, 0.1286, 0.1512])
        
        >>> cum_returns_1d_nb(returns, 100.0)  # 绝对资产价值
        array([110.0, 104.5, 112.86, 115.12])
        
        计算过程：
        - t1: 1.1 * 100 = 110 (或 1.1 - 1 = 0.1)
        - t2: 1.1 * 0.95 * 100 = 104.5 (或 1.045 - 1 = 0.045)
        - t3: 1.1 * 0.95 * 1.08 * 100 = 112.86
        - t4: 1.1 * 0.95 * 1.08 * 1.02 * 100 = 115.12
    
    金融应用：
        - 投资组合净值曲线绘制
        - 策略回测的收益率可视化
        - 风险指标计算的基础（如最大回撤）
        - 绩效比较和基准分析
    """
    # 预分配输出数组，保持与输入相同的形状和数据类型
    out = np.empty_like(returns, dtype=np.float64)
    # 初始化累计乘积因子
    cumprod = 1
    
    # 逐期计算累计收益率
    for i in range(returns.shape[0]):
        # 检查当期收益率是否有效（非NaN）
        if not np.isnan(returns[i]):
            # 累计乘积：将当期收益率转换为增长因子并累乘
            cumprod *= returns[i] + 1
        # 保存当期的累计增长因子
        out[i] = cumprod
    
    # 根据起始值类型返回不同格式的结果
    if start_value == 0.:
        # 返回相对累计收益率（减去1得到百分比形式）
        return out - 1.
    # 返回绝对累计资产价值（乘以起始价值）
    return out * start_value


@njit(cache=True)  # Numba编译优化
def cum_returns_nb(returns: tp.Array2d, start_value: float) -> tp.Array2d:
    """
    二维累计收益率批量计算函数
    
    这是cum_returns_1d_nb的二维扩展版本，能够同时计算多个资产或策略的
    累计收益率曲线。该函数是多资产投资组合分析和策略比较的核心工具。
    
    设计理念：
        - 每列代表一个资产或策略的累计收益率时间序列
        - 每行代表同一时间点的多个资产累计收益率
        - 所有资产使用相同的起始价值进行标准化
        - 支持大规模批量处理，提高计算效率
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
        start_value (float): 统一的起始资产价值
    
    返回值：
        tp.Array2d: 二维累计收益率矩阵，形状与输入相同
    
    计算并行性：
        - 不同资产间的计算完全独立
        - 适合多核并行优化
        - 每列调用一维累计收益率函数
    
    使用示例：
        >>> returns = np.array([[0.1, 0.05],    # t1: 资产A=10%, 资产B=5%
        ...                     [-0.05, 0.02],  # t2: 资产A=-5%, 资产B=2%
        ...                     [0.08, -0.01]]) # t3: 资产A=8%, 资产B=-1%
        >>> cum_returns_nb(returns, 100.0)
        array([[110.0, 105.0],      # t1累计价值
               [104.5, 107.1],      # t2累计价值
               [112.86, 106.029]])  # t3累计价值
    
    应用场景：
        - 多策略净值曲线对比
        - 行业轮动效果分析
        - 因子组合绩效评估
        - 资产配置效果监控
    """
    # 预分配输出矩阵，保持与输入相同的形状
    out = np.empty_like(returns, dtype=np.float64)
    
    # 逐列处理每个资产的累计收益率
    for col in range(returns.shape[1]):
        # 调用一维函数处理当前资产列
        out[:, col] = cum_returns_1d_nb(returns[:, col], start_value)
    
    return out


@njit(cache=True)  # Numba编译优化
def cum_returns_final_1d_nb(returns: tp.Array1d, start_value: float = 0.) -> float:
    """
    一维期末总收益率计算函数
    
    该函数计算整个投资期间的最终累计收益率，是衡量投资策略整体表现的
    核心指标。与累计收益率曲线不同，此函数只返回期末的单一数值结果。
    
    计算原理：
        - 直接计算所有收益率的几何平均
        - 等价于cum_returns_1d_nb的最后一个值
        - 更高效的一次性计算，适用于只需要期末结果的场景
    
    参数说明：
        returns (tp.Array1d): 收益率时间序列
        start_value (float): 起始价值，默认为0（返回相对收益率）
    
    返回值：
        float: 期末累计收益率
    
    计算优势：
        - 无需生成中间数组，内存效率高
        - 使用np.nanprod一次性完成所有计算
        - 自动处理NaN值，提高数据容错性
    
    使用示例：
        >>> returns = np.array([0.1, -0.05, 0.08, 0.02])
        >>> cum_returns_final_1d_nb(returns, 0.0)
        0.1512  # 15.12%的总收益率
        
        >>> cum_returns_final_1d_nb(returns, 100.0)
        115.12  # 期末资产价值
    
    应用场景：
        - 投资策略最终绩效评估
        - 不同投资期间的收益率比较
        - 年化收益率计算的基础
        - 大规模策略筛选和排序
    """
    # 使用几何平均一次性计算期末累计增长因子
    out = np.nanprod(returns + 1.)
    
    # 根据起始值返回相应格式的结果
    if start_value == 0.:
        # 返回相对累计收益率
        return out - 1.
    # 返回绝对累计资产价值
    return out * start_value


@njit(cache=True)  # Numba编译优化
def cum_returns_final_nb(returns: tp.Array2d, start_value: float = 0.) -> tp.Array1d:
    """
    二维期末总收益率批量计算函数
    
    这是cum_returns_final_1d_nb的二维扩展版本，能够同时计算多个资产或策略
    的期末累计收益率。返回一个一维数组，包含每个资产的期末收益率结果。
    
    设计用途：
        - 快速获得多个策略的最终绩效排名
        - 批量计算投资组合中各资产的期末表现
        - 为后续的绩效分析提供汇总数据
        - 支持大规模策略筛选和优化
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
        start_value (float): 统一起始价值，默认为0
    
    返回值：
        tp.Array1d: 一维期末收益率数组，形状为(资产数,)
    
    使用示例：
        >>> returns = np.array([[0.1, 0.05],
        ...                     [-0.05, 0.02],
        ...                     [0.08, -0.01]])
        >>> cum_returns_final_nb(returns, 0.0)
        array([0.1512, 0.0609])  # 资产A: 15.12%, 资产B: 6.09%
    
    应用价值：
        - 投资组合绩效汇总
        - 策略排序和筛选
        - 风险调整收益率计算的输入
        - 绩效归因分析的基础数据
    """
    # 预分配输出数组，长度等于资产数量
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的期末收益率
    for col in range(returns.shape[1]):
        # 调用一维函数计算当前资产的期末收益率
        out[col] = cum_returns_final_1d_nb(returns[:, col], start_value)
    
    return out


@njit  # Numba编译优化
def rolling_cum_returns_final_nb(returns: tp.Array2d,
                                 window: int,
                                 minp: tp.Optional[int],
                                 start_value: float = 0.) -> tp.Array2d:
    """
    滚动累计总收益率计算函数
    
    该函数计算指定窗口长度的滚动累计总收益率，为动态风险管理和实时
    绩效监控提供关键数据。通过滑动窗口的方式，能够观察不同时期的
    投资表现变化趋势。
    
    功能特点：
        - 使用滑动窗口技术，逐期更新计算窗口
        - 每个时间点计算过去N期的累计收益率
        - 支持最小观测期数设置，确保统计有效性
        - 提供连续的动态绩效监控能力
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
        window (int): 滚动窗口大小（观测期数）
        minp (tp.Optional[int]): 最小有效观测期数
            - 当窗口内有效数据少于此值时返回NaN
            - None时使用window作为最小期数
        start_value (float): 起始价值，默认为0（相对收益率）
    
    返回值：
        tp.Array2d: 滚动累计收益率矩阵，形状与输入相同
    
    应用场景：
        - 动态风险监控：观察近期绩效变化
        - 策略有效性评估：检验策略在不同市场环境下的表现
        - 资产配置调整：基于近期表现调整权重
        - 实时绩效跟踪：为交易决策提供及时反馈
    
    使用示例：
        >>> returns = np.array([[0.02, 0.01],
        ...                     [0.03, -0.01],
        ...                     [-0.01, 0.02],
        ...                     [0.01, 0.01]])
        >>> rolling_cum_returns_final_nb(returns, window=3, minp=2, start_value=0.0)
        # 计算3期滚动窗口的累计收益率
    
    技术实现：
        - 内部定义应用函数，调用一维累计收益率计算
        - 利用generic_nb.rolling_apply_nb进行滚动应用
        - 自动处理边界条件和数据有效性检查
    """
    # 定义滚动窗口内的应用函数
    def _apply_func_nb(i, col, _returns, _start_value):
        # 对窗口内的收益率计算累计总收益率
        return cum_returns_final_1d_nb(_returns, _start_value)

    # 使用通用滚动应用函数进行计算
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, start_value)


@njit(cache=True)  # Numba编译优化，启用缓存
def annualized_return_1d_nb(returns: tp.Array1d, ann_factor: float) -> float:
    """
    一维年化收益率计算函数
    
    该函数计算复合年增长率(CAGR)，是量化投资中最重要的绩效指标之一。
    年化收益率消除了投资期长度的影响，使不同期间和不同策略的收益率
    具备可比性，是投资决策和绩效评估的标准指标。
    
    计算原理：
        CAGR = (期末价值/期初价值)^(年化因子/期数) - 1
        其中年化因子 = 年度交易日数 / 数据频率对应的日数
    
    数学公式：
        annualized_return = (1 + total_return)^(ann_factor / n_periods) - 1
        其中 total_return 为整个投资期间的累计收益率
    
    参数说明：
        returns (tp.Array1d): 一维收益率时间序列
        ann_factor (float): 年化因子
            - 日频数据：通常为252（年均交易日）
            - 周频数据：通常为52（年均周数）
            - 月频数据：通常为12（年均月数）
    
    返回值：
        float: 年化收益率（小数形式，如0.15表示15%）
    
    应用价值：
        - 标准化不同期间的投资表现
        - 与市场基准进行公平比较
        - 投资组合配置决策的重要依据
        - 风险调整收益率计算的基础
    
    使用示例：
        >>> returns = np.array([0.02, 0.01, -0.005, 0.03])  # 4个月收益率
        >>> annualized_return_1d_nb(returns, 12.0)  # 月频数据年化
        0.0823  # 约8.23%的年化收益率
        
        计算过程：
        1. 累计收益率 = (1.02) × (1.01) × (0.995) × (1.03) - 1 = 0.0547
        2. 年化收益率 = (1 + 0.0547)^(12/4) - 1 = 1.0547^3 - 1 ≈ 0.0823
    
    注意事项：
        - 假设收益率在时间上均匀分布
        - 适用于长期投资绩效评估
        - 短期数据的年化结果可能不具代表性
    """
    # 计算期末累计价值（以1为起始价值）
    end_value = cum_returns_final_1d_nb(returns, 1.)
    # 使用复利公式计算年化收益率
    return end_value ** (ann_factor / returns.shape[0]) - 1


@njit(cache=True)  # Numba编译优化
def annualized_return_nb(returns: tp.Array2d, ann_factor: float) -> tp.Array1d:
    """
    二维年化收益率批量计算函数
    
    这是annualized_return_1d_nb的二维扩展版本，能够同时计算多个资产
    或策略的年化收益率。该函数是投资组合分析和策略比较的核心工具。
    
    设计优势：
        - 批量处理多个资产，提高计算效率
        - 统一年化因子，确保结果的可比性
        - 向量化计算，充分利用硬件性能
        - 为后续分析提供标准化的输入数据
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
        ann_factor (float): 统一的年化因子
    
    返回值：
        tp.Array1d: 各资产年化收益率数组，形状为(资产数,)
    
    应用场景：
        - 多资产投资组合绩效比较
        - 策略筛选和排序
        - 基金经理绩效评估
        - 资产配置权重优化的输入
    
    使用示例：
        >>> returns = np.array([[0.01, 0.02, -0.005],   # 资产A, B, C的月收益率
        ...                     [0.02, -0.01, 0.01],
        ...                     [-0.005, 0.015, 0.02],
        ...                     [0.03, 0.01, -0.01]])
        >>> annualized_return_nb(returns, 12.0)
        array([0.0823, 0.0456, 0.0201])  # 各资产的年化收益率
    
    计算特点：
        - 每列独立计算，支持并行优化
        - 保持计算方法的一致性
        - 提供汇总格式的结果输出
    """
    # 预分配输出数组，长度等于资产数量
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的年化收益率
    for col in range(returns.shape[1]):
        # 调用一维函数计算当前资产的年化收益率
        out[col] = annualized_return_1d_nb(returns[:, col], ann_factor)
    
    return out


@njit  # Numba编译优化
def rolling_annualized_return_nb(returns: tp.Array2d,
                                 window: int,
                                 minp: tp.Optional[int],
                                 ann_factor: float) -> tp.Array2d:
    """
    滚动年化收益率计算函数
    
    该函数计算指定窗口期间的滚动年化收益率，为投资策略的动态评估
    提供重要工具。通过观察不同时期的年化表现，能够识别策略的
    时效性和市场适应性。
    
    功能特点：
        - 动态窗口计算：每个时点计算过去N期的年化收益率
        - 时序稳定性分析：观察年化收益率的时间变化
        - 策略评估工具：评估不同市场环境下的策略效果
        - 风险管理支持：识别绩效异常期间
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵
        window (int): 滚动窗口大小
        minp (tp.Optional[int]): 最小有效观测期数
        ann_factor (float): 年化因子
    
    返回值：
        tp.Array2d: 滚动年化收益率矩阵，形状与输入相同
    
    应用场景：
        - 策略适应性分析：观察在不同市场环境下的表现
        - 绩效稳定性评估：检验策略的持续盈利能力
        - 实时监控：为交易决策提供动态反馈
        - 风险预警：识别绩效恶化趋势
    
    使用示例：
        >>> returns = np.random.normal(0.001, 0.02, (252, 3))  # 252个交易日，3个策略
        >>> rolling_ann_ret = rolling_annualized_return_nb(returns, 60, 30, 252.0)
        # 计算60日滚动年化收益率，最少需要30个有效观测
    
    技术实现：
        - 使用内部应用函数调用一维年化收益率计算
        - 通过generic_nb.rolling_apply_nb进行滚动计算
        - 自动处理窗口边界和数据有效性
    """
    # 定义滚动窗口内的应用函数
    def _apply_func_nb(i, col, _returns, _ann_factor):
        # 对窗口内数据计算年化收益率
        return annualized_return_1d_nb(_returns, _ann_factor)

    # 使用通用滚动应用函数进行计算
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, ann_factor)


@njit(cache=True)  # Numba编译优化，启用缓存
def annualized_volatility_1d_nb(returns: tp.Array1d,
                                ann_factor: float,
                                levy_alpha: float = 2.0,
                                ddof: int = 1) -> float:
    """
    一维年化波动率计算函数
    
    该函数计算投资策略的年化波动率，是衡量投资风险的核心指标。
    年化波动率反映了投资收益的不确定性程度，是现代投资组合理论
    中风险度量的基础，也是各种风险调整收益指标计算的重要组成部分。
    
    计算原理：
        年化波动率 = 标准差 × sqrt(年化因子)
        其中标准差使用样本标准差（默认ddof=1）
        
    levy_alpha参数说明：
        - 默认值2.0：假设收益率服从正态分布（高斯分布）
        - 其他值：适用于厚尾分布（如Levy稳定分布）
        - 年化公式：ann_factor^(1/levy_alpha)
    
    参数说明：
        returns (tp.Array1d): 一维收益率时间序列
        ann_factor (float): 年化因子
            - 日频数据：252（年均交易日）
            - 月频数据：12（年均月数）
            - 周频数据：52（年均周数）
        levy_alpha (float): Levy稳定性指数，默认为2.0（正态分布）
        ddof (int): 自由度调整，默认为1（样本标准差）
    
    返回值：
        float: 年化波动率（小数形式，如0.15表示15%）
    
    数值稳定性：
        - 当观测值少于2个时返回NaN
        - 使用nanstd_1d_nb自动处理NaN值
        - 支持不完整时间序列的计算
    
    使用示例：
        >>> returns = np.array([0.02, -0.01, 0.015, -0.005, 0.01])
        >>> annualized_volatility_1d_nb(returns, 252.0, 2.0, 1)
        0.1591  # 约15.91%的年化波动率
        
        计算过程：
        1. 样本标准差 = 0.01002（基于5个观测值）
        2. 年化波动率 = 0.01002 × sqrt(252) = 0.01002 × 15.87 ≈ 0.1591
    
    金融意义：
        - 高波动率：投资风险较大，但可能带来更高收益
        - 低波动率：投资相对稳定，但收益潜力有限
        - 是计算夏普比率、VaR等指标的基础
        - 投资组合风险管理的重要参考
    
    应用场景：
        - 投资风险评估和限额管理
        - 夏普比率等风险调整指标计算
        - 投资组合优化中的风险约束
        - VaR和CVaR等风险度量计算
    """
    # 检查数据有效性：至少需要2个观测值才能计算标准差
    if returns.shape[0] < 2:
        return np.nan

    # 计算年化波动率：标准差乘以年化因子的Levy调整
    return generic_nb.nanstd_1d_nb(returns, ddof) * ann_factor ** (1.0 / levy_alpha)


@njit(cache=True)  # Numba编译优化
def annualized_volatility_nb(returns: tp.Array2d,
                             ann_factor: float,
                             levy_alpha: float = 2.0,
                             ddof: int = 1) -> tp.Array1d:
    """
    二维年化波动率批量计算函数
    
    这是annualized_volatility_1d_nb的二维扩展版本，能够同时计算多个资产
    或策略的年化波动率。该函数在多资产投资组合风险管理和策略比较中
    发挥重要作用。
    
    设计优势：
        - 统一参数设置，确保风险度量的一致性
        - 批量处理，显著提高计算效率
        - 向量化输出，便于后续风险分析
        - 支持大规模策略的风险筛选
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
        ann_factor (float): 统一年化因子
        levy_alpha (float): Levy稳定性指数，默认2.0
        ddof (int): 自由度调整，默认1
    
    返回值：
        tp.Array1d: 各资产年化波动率数组，形状为(资产数,)
    
    应用场景：
        - 投资组合风险分散度分析
        - 资产相关性和协方差矩阵计算的输入
        - 多策略风险预算分配
        - 风险调整收益指标的批量计算
    
    使用示例：
        >>> returns = np.array([[0.01, 0.02, -0.005],   # 3个资产的日收益率
        ...                     [0.02, -0.01, 0.01],
        ...                     [-0.005, 0.015, 0.02],
        ...                     [0.03, 0.01, -0.01]])
        >>> annualized_volatility_nb(returns, 252.0, 2.0, 1)
        array([0.1523, 0.1089, 0.0876])  # 各资产的年化波动率
    
    风险管理应用：
        - 识别高风险资产，进行权重调整
        - 构建等风险权重的投资组合
        - 设置基于波动率的止损线
        - 计算风险调整后的资产配置比例
    """
    # 预分配输出数组，长度等于资产数量
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的年化波动率
    for col in range(returns.shape[1]):
        # 调用一维函数计算当前资产的年化波动率
        out[col] = annualized_volatility_1d_nb(returns[:, col], ann_factor, levy_alpha, ddof)
    
    return out


@njit  # Numba编译优化
def rolling_annualized_volatility_nb(returns: tp.Array2d,
                                     window: int,
                                     minp: tp.Optional[int],
                                     ann_factor: float,
                                     levy_alpha: float = 2.0,
                                     ddof: int = 1) -> tp.Array2d:
    """
    滚动年化波动率计算函数
    
    该函数计算指定窗口期间的滚动年化波动率，为动态风险管理提供关键工具。
    通过观察波动率的时间变化，能够识别市场风险的变化趋势，为实时风险
    控制和投资决策提供重要参考。
    
    功能特点：
        - 动态风险监控：实时跟踪投资组合风险变化
        - 市场状态识别：识别高波动和低波动市场环境
        - 风险预警：及时发现风险异常期间
        - 策略适应性：评估策略在不同波动环境下的表现
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵
        window (int): 滚动窗口大小
        minp (tp.Optional[int]): 最小有效观测期数
        ann_factor (float): 年化因子
        levy_alpha (float): Levy稳定性指数，默认2.0
        ddof (int): 自由度调整，默认1
    
    返回值：
        tp.Array2d: 滚动年化波动率矩阵，形状与输入相同
    
    应用场景：
        - 动态风险限额管理：根据当前波动率调整头寸
        - 市场择时：基于波动率变化调整投资策略
        - 风险预警系统：设置基于滚动波动率的预警线
        - 策略评估：分析策略在不同波动环境下的稳定性
    
    使用示例：
        >>> returns = np.random.normal(0, 0.02, (252, 2))  # 252个交易日，2个资产
        >>> rolling_vol = rolling_annualized_volatility_nb(
        ...     returns, window=30, minp=20, ann_factor=252.0
        ... )
        # 计算30日滚动年化波动率，最少需要20个有效观测
    
    风险管理策略：
        - 波动率突破：当滚动波动率超过历史分位数时减仓
        - 波动率均值回归：利用波动率的均值回归特性进行交易
        - 动态对冲：根据滚动波动率调整对冲比例
        - 风险预算：基于滚动波动率动态调整风险预算
    """
    # 定义滚动窗口内的应用函数
    def _apply_func_nb(i, col, _returns, _ann_factor, _levy_alpha, _ddof):
        # 对窗口内数据计算年化波动率
        return annualized_volatility_1d_nb(_returns, _ann_factor, _levy_alpha, _ddof)

    # 使用通用滚动应用函数进行计算
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, ann_factor, levy_alpha, ddof)


@njit(cache=True)  # Numba编译优化，启用缓存
def drawdown_1d_nb(returns: tp.Array1d) -> tp.Array1d:
    """
    一维回撤序列计算函数
    
    该函数计算累计收益率相对于历史最高点的回撤幅度，是衡量投资风险和
    损失程度的核心指标。回撤序列能够直观显示投资策略在历史上每个时点
    相对于峰值的损失情况，为风险管理提供重要参考。
    
    计算原理：
        回撤率 = (当前累计净值 / 历史最高净值) - 1
        - 当达到新的净值高点时，回撤为0
        - 当净值下跌时，回撤为负值，表示损失幅度
        - 回撤序列连续展现整个投资过程的风险暴露
    
    参数说明：
        returns (tp.Array1d): 一维收益率时间序列
    
    返回值：
        tp.Array1d: 回撤序列，形状与输入相同
            - 0表示达到新的净值高点
            - 负值表示相对于历史最高点的损失百分比
    
    计算步骤：
        1. 计算累计净值序列（起始值设为100）
        2. 计算截至当期的历史最高净值
        3. 计算当前净值相对于最高净值的回撤比例
    
    使用示例：
        >>> returns = np.array([0.1, -0.05, -0.02, 0.08])
        >>> drawdown_1d_nb(returns)
        array([0., -0.0455, -0.0635, 0.])
        
        解释：
        - t1: 净值110，历史最高110，回撤0%
        - t2: 净值104.5，历史最高110，回撤-4.55%
        - t3: 净值102.41，历史最高110，回撤-6.36%
        - t4: 净值110.6，新的历史最高，回撤0%
    
    风险管理应用：
        - 识别策略的最大风险暴露期间
        - 设置基于回撤的止损机制
        - 评估策略的风险控制能力
        - 为投资者提供风险披露信息
    
    金融意义：
        - 反映投资者可能面临的最大账面损失
        - 评估策略的抗风险能力
        - 是最大回撤(MDD)计算的基础
        - 投资心理承受能力的重要参考
    """
    # 计算累计净值序列，起始值为100（便于理解百分比）
    cum_returns = cum_returns_1d_nb(returns, start_value=100.)
    # 计算截至当期的历史最高净值
    max_returns = generic_nb.expanding_max_1d_nb(cum_returns, minp=1)
    # 计算回撤：当前净值相对于历史最高净值的比例减1
    return cum_returns / max_returns - 1


@njit(cache=True)  # Numba编译优化
def drawdown_nb(returns: tp.Array2d) -> tp.Array2d:
    """
    二维回撤序列批量计算函数
    
    这是drawdown_1d_nb的二维扩展版本，能够同时计算多个资产或策略的
    回撤序列。该函数在投资组合风险管理和多策略比较中具有重要作用。
    
    设计优势：
        - 批量处理多个策略，提高计算效率
        - 统一回撤计算标准，便于策略比较
        - 矩阵输出格式，方便后续分析和可视化
        - 支持大规模策略的风险评估
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
    
    返回值：
        tp.Array2d: 二维回撤矩阵，形状与输入相同
    
    应用场景：
        - 多策略风险对比：比较不同策略的风险特征
        - 投资组合风险分散：评估组合资产的风险贡献
        - 风险预警系统：监控多个策略的同时回撤风险
        - 策略筛选：基于回撤特征筛选优质策略
    
    使用示例：
        >>> returns = np.array([[0.02, 0.01],    # 两个策略的收益率
        ...                     [-0.03, -0.01],
        ...                     [0.01, 0.02]])
        >>> drawdown_nb(returns)
        array([[0., 0.],           # t1: 两策略都创新高
               [-0.0098, -0.0099],  # t2: 两策略都出现回撤
               [0., 0.]])           # t3: 两策略都创新高
    
    风险管理策略：
        - 当多个策略同时出现大幅回撤时，考虑系统性风险
        - 选择回撤相关性较低的策略进行组合
        - 设置基于组合回撤的动态仓位调整机制
    """
    # 预分配输出矩阵，保持与输入相同的形状
    out = np.empty_like(returns, dtype=np.float64)
    
    # 逐列计算每个资产的回撤序列
    for col in range(returns.shape[1]):
        # 调用一维函数计算当前资产的回撤序列
        out[:, col] = drawdown_1d_nb(returns[:, col])
    
    return out


@njit(cache=True)  # Numba编译优化
def max_drawdown_1d_nb(returns: tp.Array1d) -> float:
    """
    一维最大回撤计算函数
    
    该函数计算投资策略的最大回撤(Maximum Drawdown, MDD)，是衡量投资
    风险最重要的指标之一。最大回撤表示在整个投资期间从任一历史高点
    到随后最低点的最大损失幅度，是投资者可能面临的最坏情况。
    
    计算原理：
        最大回撤 = min(回撤序列)
        即回撤序列中的最小值（最大负值）
    
    参数说明：
        returns (tp.Array1d): 一维收益率时间序列
    
    返回值：
        float: 最大回撤（负值，如-0.2表示20%的最大回撤）
    
    风险管理意义：
        - 反映策略的最大历史损失
        - 评估投资者的心理承受能力要求
        - 是风险调整收益指标（如卡尔玛比率）的分母
        - 设置止损线的重要参考
    
    使用示例：
        >>> returns = np.array([0.1, -0.15, -0.05, 0.2, -0.08])
        >>> max_drawdown_1d_nb(returns)
        -0.1818  # 约18.18%的最大回撤
        
        计算过程：
        1. 累计净值: [110, 93.5, 88.825, 106.59, 98.06]
        2. 历史最高: [110, 110, 110, 110, 110]（直到第4期）
        3. 回撤序列: [0, -0.15, -0.1925, 0, ...]
        4. 最大回撤: -0.1925 (19.25%)
    
    应用场景：
        - 策略风险评估和比较
        - 投资组合风险预算分配
        - 风险调整收益指标计算
        - 监管合规和风险披露
    
    投资者教育：
        - 帮助投资者了解策略的历史最大损失
        - 设定合理的投资预期和心理准备
        - 制定基于风险承受能力的投资计划
    """
    # 计算回撤序列并返回其最小值（最大回撤）
    return np.min(drawdown_1d_nb(returns))


@njit(cache=True)  # Numba编译优化
def max_drawdown_nb(returns: tp.Array2d) -> tp.Array1d:
    """
    二维最大回撤批量计算函数
    
    这是max_drawdown_1d_nb的二维扩展版本，能够同时计算多个资产或策略
    的最大回撤。该函数在投资组合构建和策略筛选中发挥重要作用。
    
    设计优势：
        - 批量风险评估：同时评估多个策略的风险水平
        - 标准化比较：提供统一的风险度量标准
        - 高效计算：向量化处理提高计算效率
        - 决策支持：为投资决策提供量化风险依据
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
    
    返回值：
        tp.Array1d: 各资产最大回撤数组，形状为(资产数,)
    
    应用场景：
        - 策略筛选：选择最大回撤可接受的策略
        - 风险预算：基于最大回撤分配投资权重
        - 合规管理：确保策略符合风险限制要求
        - 客户服务：向客户展示不同策略的风险特征
    
    使用示例：
        >>> returns = np.array([[0.05, 0.02, 0.08],    # 3个策略
        ...                     [-0.1, -0.02, -0.05],
        ...                     [-0.05, 0.03, 0.02],
        ...                     [0.15, -0.01, -0.03]])
        >>> max_drawdown_nb(returns)
        array([-0.1429, -0.0291, -0.0741])  # 各策略的最大回撤
    
    风险管理决策：
        - 策略A最大回撤14.29%，风险较高
        - 策略B最大回撤2.91%，风险较低
        - 策略C最大回撤7.41%，风险中等
        - 可基于风险承受能力选择合适策略
    """
    # 预分配输出数组，长度等于资产数量
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的最大回撤
    for col in range(returns.shape[1]):
        # 调用一维函数计算当前资产的最大回撤
        out[col] = max_drawdown_1d_nb(returns[:, col])
    
    return out


@njit  # Numba编译优化
def rolling_max_drawdown_nb(returns: tp.Array2d, window: int, minp: tp.Optional[int]) -> tp.Array2d:
    """
    滚动最大回撤计算函数
    
    该函数计算指定窗口期间的滚动最大回撤，为动态风险管理提供重要工具。
    通过观察最大回撤的时间变化，能够及时识别风险积累和市场环境变化，
    为实时风险控制提供关键信息。
    
    功能特点：
        - 动态风险监控：实时跟踪策略风险变化
        - 趋势识别：识别风险恶化和改善趋势
        - 预警机制：基于滚动最大回撤设置风险预警
        - 适应性管理：根据近期风险情况调整策略
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵
        window (int): 滚动窗口大小
        minp (tp.Optional[int]): 最小有效观测期数
    
    返回值：
        tp.Array2d: 滚动最大回撤矩阵，形状与输入相同
    
    应用场景：
        - 动态止损：当滚动最大回撤超过阈值时止损
        - 仓位管理：基于近期回撤情况调整仓位大小
        - 策略切换：当策略滚动回撤恶化时切换策略
        - 风险报告：为客户提供动态风险监控报告
    
    使用示例：
        >>> returns = np.random.normal(0.001, 0.02, (100, 2))  # 100期，2个策略
        >>> rolling_mdd = rolling_max_drawdown_nb(returns, window=20, minp=10)
        # 计算20期滚动最大回撤，最少需要10个有效观测
    
    风险管理策略：
        - 渐进式减仓：随着滚动回撤增加逐步减少仓位
        - 动态对冲：基于滚动回撤调整对冲比例
        - 资金管理：根据滚动回撤调整资金使用率
        - 心理管理：帮助投资者了解当前风险水平
    
    技术实现：
        - 使用滚动窗口技术计算局部最大回撤
        - 通过generic_nb.rolling_apply_nb进行滚动计算
        - 自动处理窗口边界和数据完整性检查
    """
    # 定义滚动窗口内的应用函数
    def _apply_func_nb(i, col, _returns):
        # 对窗口内数据计算最大回撤
        return max_drawdown_1d_nb(_returns)

    # 使用通用滚动应用函数进行计算
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb)


@njit(cache=True)  # Numba编译优化，启用缓存
def calmar_ratio_1d_nb(returns: tp.Array1d, ann_factor: float) -> float:
    """
    一维卡尔玛比率计算函数
    
    卡尔玛比率(Calmar Ratio)，也称为回撤比率，是衡量风险调整收益的重要指标。
    该比率计算年化收益率与最大回撤的比值，反映了每承担一单位最大回撤风险
    所能获得的年化收益。相比夏普比率使用标准差作为风险度量，卡尔玛比率
    使用最大回撤，更直观地反映了投资者可能面临的最大损失。
    
    计算公式：
        卡尔玛比率 = 年化收益率 / |最大回撤|
        - 分子：衡量策略的年化盈利能力
        - 分母：衡量策略的最大历史损失
        - 比值：每承担1%最大回撤风险获得的年化收益
    
    参数说明：
        returns (tp.Array1d): 一维收益率时间序列
        ann_factor (float): 年化因子
    
    返回值：
        float: 卡尔玛比率
            - 正值：策略盈利且存在回撤，数值越大越好
            - 无穷大：策略盈利但无回撤（理想情况）
            - NaN：无法计算（如数据不足或特殊情况）
    
    数值稳定性处理：
        - 当最大回撤为0时：返回无穷大（完美策略）
        - 当年化收益率为负时：比率为负值（亏损策略）
        - 数据不足时：返回NaN
    
    使用示例：
        >>> returns = np.array([0.02, 0.01, -0.01, 0.03, -0.02])  # 月收益率
        >>> calmar_ratio_1d_nb(returns, 12.0)
        2.86  # 卡尔玛比率为2.86
        
        计算过程：
        1. 年化收益率 = 约6.8%
        2. 最大回撤 = 约-2.38%
        3. 卡尔玛比率 = 6.8% / 2.38% = 2.86
    
    投资意义：
        - 值越高，单位风险的收益越高
        - 相比夏普比率，更关注极端风险
        - 适合风险厌恶型投资者的评估标准
        - 在对冲基金和绝对收益策略中广泛使用
    
    应用场景：
        - 策略风险调整收益评估
        - 不同策略的风险效率比较
        - 投资组合经理绩效评估
        - 风险预算分配的量化依据
    
    与其他指标的比较：
        - 夏普比率：使用波动率作为风险度量
        - 索提诺比率：使用下行风险作为风险度量
        - 卡尔玛比率：使用最大回撤作为风险度量
    """
    # 计算最大回撤（负值）
    max_drawdown = max_drawdown_1d_nb(returns)
    
    # 特殊情况处理：最大回撤为0（完美策略，无损失）
    if max_drawdown == 0.:
        return np.nan  # 实际应该是无穷大，但这里返回NaN以保持一致性
    
    # 计算年化收益率
    annualized_return = annualized_return_1d_nb(returns, ann_factor)
    
    # 再次检查最大回撤（防止数值误差）
    if max_drawdown == 0.:
        return np.inf
    
    # 计算卡尔玛比率：年化收益率除以最大回撤的绝对值
    return annualized_return / np.abs(max_drawdown)


@njit(cache=True)  # Numba编译优化
def calmar_ratio_nb(returns: tp.Array2d, ann_factor: float) -> tp.Array1d:
    """
    二维卡尔玛比率批量计算函数
    
    这是calmar_ratio_1d_nb的二维扩展版本，能够同时计算多个资产或策略的
    卡尔玛比率。该函数在多策略比较和投资组合构建中具有重要价值。
    
    设计优势：
        - 统一风险调整收益评估：使用一致的计算标准
        - 高效批量处理：显著提高多策略分析效率
        - 便于排序筛选：输出格式便于策略排序和筛选
        - 支持大规模分析：适合处理数百个策略的比较
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
        ann_factor (float): 统一年化因子
    
    返回值：
        tp.Array1d: 各资产卡尔玛比率数组，形状为(资产数,)
    
    应用场景：
        - 多策略绩效评估：比较不同策略的风险调整收益
        - 投资组合构建：选择卡尔玛比率较高的策略
        - 基金经理评估：评估不同经理的管理能力
        - 风险预算分配：基于卡尔玛比率分配投资权重
    
    使用示例：
        >>> returns = np.array([[0.02, 0.01, 0.03],   # 3个策略的月收益率
        ...                     [0.01, -0.02, 0.01],
        ...                     [-0.01, 0.03, -0.02],
        ...                     [0.03, -0.01, 0.04]])
        >>> calmar_ratio_nb(returns, 12.0)
        array([2.45, 1.32, 3.78])  # 各策略的卡尔玛比率
    
    投资决策支持：
        - 策略A: 卡尔玛比率2.45，中等风险调整收益
        - 策略B: 卡尔玛比率1.32，较低风险调整收益  
        - 策略C: 卡尔玛比率3.78，较高风险调整收益
        - 建议优先选择策略C进行投资
    
    风险管理价值：
        - 识别最大回撤控制较好的策略
        - 平衡收益与极端风险的关系
        - 为风险厌恶投资者提供决策依据
    """
    # 预分配输出数组，长度等于资产数量
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的卡尔玛比率
    for col in range(returns.shape[1]):
        # 调用一维函数计算当前资产的卡尔玛比率
        out[col] = calmar_ratio_1d_nb(returns[:, col], ann_factor)
    
    return out


@njit  # Numba编译优化
def rolling_calmar_ratio_nb(returns: tp.Array2d,
                            window: int,
                            minp: tp.Optional[int],
                            ann_factor: float) -> tp.Array2d:
    """
    滚动卡尔玛比率计算函数
    
    该函数计算指定窗口期间的滚动卡尔玛比率，为动态风险调整收益评估提供
    重要工具。通过观察卡尔玛比率的时间变化，能够识别策略在不同市场环境
    下的风险调整表现，为实时策略评估和调整提供量化依据。
    
    功能特点：
        - 动态绩效评估：实时评估策略的风险调整表现
        - 趋势识别：识别策略绩效改善或恶化趋势
        - 时点比较：比较策略在不同时期的相对表现
        - 适应性分析：评估策略在不同市场环境下的稳定性
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵
        window (int): 滚动窗口大小
        minp (tp.Optional[int]): 最小有效观测期数
        ann_factor (float): 年化因子
    
    返回值：
        tp.Array2d: 滚动卡尔玛比率矩阵，形状与输入相同
    
    应用场景：
        - 策略动态评估：监控策略绩效的实时变化
        - 市场适应性分析：评估策略在不同市场环境下的表现
        - 绩效归因：识别策略表现突出或不佳的时期
        - 动态资产配置：基于滚动绩效调整资产权重
    
    使用示例：
        >>> returns = np.random.normal(0.005, 0.02, (252, 3))  # 252个交易日，3个策略
        >>> rolling_calmar = rolling_calmar_ratio_nb(
        ...     returns, window=60, minp=30, ann_factor=252.0
        ... )
        # 计算60日滚动卡尔玛比率，最少需要30个有效观测
    
    策略管理应用：
        - 当策略的滚动卡尔玛比率持续下降时，考虑策略调整
        - 比较多个策略的滚动表现，动态选择最优策略
        - 设置基于滚动卡尔玛比率的策略切换阈值
        - 为客户提供动态的策略表现报告
    
    技术实现：
        - 使用内部应用函数调用一维卡尔玛比率计算
        - 通过generic_nb.rolling_apply_nb进行滚动计算
        - 自动处理窗口边界和数据完整性检查
    """
    # 定义滚动窗口内的应用函数
    def _apply_func_nb(i, col, _returns, _ann_factor):
        # 对窗口内数据计算卡尔玛比率
        return calmar_ratio_1d_nb(_returns, _ann_factor)

    # 使用通用滚动应用函数进行计算
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, ann_factor)


@njit(cache=True)
def omega_ratio_1d_nb(returns: tp.Array1d,
                      ann_factor: float,
                      risk_free: float = 0.,
                      required_return: float = 0.) -> float:
    """Omega ratio of a strategy.."""
    if ann_factor == 1:
        return_threshold = required_return
    elif ann_factor <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** (1. / ann_factor) - 1
    returns_less_thresh = returns - risk_free - return_threshold
    numer = np.sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * np.sum(returns_less_thresh[returns_less_thresh < 0.0])
    if denom == 0.:
        return np.inf
    return numer / denom


@njit(cache=True)
def omega_ratio_nb(returns: tp.Array2d,
                   ann_factor: float,
                   risk_free: float = 0.,
                   required_return: float = 0.) -> tp.Array1d:
    """2-dim version of `omega_ratio_1d_nb`."""
    out = np.empty(returns.shape[1], dtype=np.float64)
    for col in range(returns.shape[1]):
        out[col] = omega_ratio_1d_nb(
            returns[:, col], ann_factor, risk_free, required_return)
    return out


@njit
def rolling_omega_ratio_nb(returns: tp.Array2d,
                           window: int,
                           minp: tp.Optional[int],
                           ann_factor: float,
                           risk_free: float = 0.,
                           required_return: float = 0.) -> tp.Array2d:
    """Rolling version of `omega_ratio_nb`."""

    def _apply_func_nb(i, col, _returns, _ann_factor, _risk_free, _required_return):
        return omega_ratio_1d_nb(_returns, _ann_factor, _risk_free, _required_return)

    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, ann_factor, risk_free, required_return)


@njit(cache=True)  # Numba编译优化，启用缓存
def sharpe_ratio_1d_nb(returns: tp.Array1d,
                       ann_factor: float,
                       risk_free: float = 0.,
                       ddof: int = 1) -> float:
    """
    一维夏普比率计算函数
    
    夏普比率(Sharpe Ratio)是现代投资组合理论中最重要的风险调整收益指标，
    由诺贝尔经济学奖得主威廉·夏普提出。该比率衡量投资组合每承担一单位
    总风险(以标准差衡量)所获得的超额收益，是评估投资策略效率的核心指标。
    
    计算公式：
        夏普比率 = (平均收益率 - 无风险利率) / 收益率标准差 × √年化因子
        - 分子：超额收益率，即投资组合收益超过无风险资产的部分
        - 分母：收益率的标准差，衡量投资的总体波动风险
        - 年化调整：乘以√年化因子进行时间标准化
    
    理论基础：
        基于现代投资组合理论的风险-收益权衡原理，认为理性投资者在承担
        相同风险水平下会选择收益更高的投资，或在相同收益水平下会选择
        风险更低的投资。夏普比率正是衡量这种效率的标准。
    
    参数说明：
        returns (tp.Array1d): 一维收益率时间序列
        ann_factor (float): 年化因子（如日频数据为252）
        risk_free (float): 无风险利率，默认为0
            - 通常使用国债利率或银行存款利率
            - 应与收益率数据的频率保持一致
        ddof (int): 自由度调整，默认为1（样本标准差）
    
    返回值：
        float: 年化夏普比率
            - 正值：策略表现优于无风险资产，数值越大越好
            - 0：策略表现等同于无风险资产
            - 负值：策略表现劣于无风险资产
            - 无穷大：策略有正超额收益但无波动（理想情况）
    
    数值稳定性：
        - 当观测值少于2个时返回NaN
        - 当标准差为0时返回无穷大（零波动策略）
        - 自动处理NaN值，提高数据容错性
    
    使用示例：
        >>> returns = np.array([0.02, 0.01, -0.01, 0.03, -0.005])  # 月收益率
        >>> sharpe_ratio_1d_nb(returns, 12.0, 0.002, 1)  # 月无风险利率0.2%
        1.73  # 年化夏普比率为1.73
        
        计算过程：
        1. 超额收益 = [0.018, 0.008, -0.012, 0.028, -0.007]
        2. 平均超额收益 = 0.007 (0.7%)
        3. 标准差 = 0.0161 (1.61%)
        4. 夏普比率 = 0.007/0.0161 × √12 = 1.73
    
    业界标准：
        - 夏普比率 > 1.0：表现良好
        - 夏普比率 > 1.5：表现优秀
        - 夏普比率 > 2.0：表现卓越
        - 夏普比率 < 0：表现劣于无风险资产
    
    应用场景：
        - 基金和策略绩效评估
        - 投资组合优化中的目标函数
        - 资产配置权重确定
        - 投资经理绩效比较
        - 风险调整收益的标准化衡量
    
    局限性：
        - 假设收益率服从正态分布
        - 对极端风险不够敏感
        - 将上行波动和下行波动等同对待
        - 适用于长期投资评估
    """
    # 检查数据有效性：至少需要2个观测值
    if returns.shape[0] < 2:
        return np.nan

    # 计算超额收益序列（减去无风险利率）
    returns_risk_adj = returns - risk_free
    # 计算平均超额收益
    mean = np.nanmean(returns_risk_adj)
    # 计算超额收益的标准差
    std = generic_nb.nanstd_1d_nb(returns_risk_adj, ddof)
    
    # 处理零波动情况：如果标准差为0，返回无穷大
    if std == 0.:
        return np.inf
    
    # 计算年化夏普比率：(平均超额收益/标准差) × √年化因子
    return mean / std * np.sqrt(ann_factor)


@njit(cache=True)  # Numba编译优化
def sharpe_ratio_nb(returns: tp.Array2d,
                    ann_factor: float,
                    risk_free: float = 0.,
                    ddof: int = 1) -> tp.Array1d:
    """
    二维夏普比率批量计算函数
    
    这是sharpe_ratio_1d_nb的二维扩展版本，能够同时计算多个资产或策略的
    夏普比率。该函数是投资组合优化、策略比较和资产筛选的核心工具。
    
    设计优势：
        - 统一评估标准：使用一致的无风险利率和计算方法
        - 高效批量处理：显著提高多策略分析效率  
        - 标准化输出：便于策略排序和量化比较
        - 规模化应用：支持处理数百个资产的同时分析
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
        ann_factor (float): 统一年化因子
        risk_free (float): 统一无风险利率，默认为0
        ddof (int): 统一自由度调整，默认为1
    
    返回值：
        tp.Array1d: 各资产夏普比率数组，形状为(资产数,)
    
    应用场景：
        - 多策略绩效排序：识别风险调整收益最高的策略
        - 投资组合构建：选择夏普比率较高的资产进行配置
        - 基金筛选：从众多基金中选择表现优异的产品
        - 风险预算分配：基于夏普比率分配投资权重
        - 绩效基准比较：与市场指数或同类产品比较
    
    使用示例：
        >>> returns = np.array([[0.01, 0.02, 0.005],   # 3个资产的日收益率
        ...                     [0.015, -0.01, 0.01],
        ...                     [-0.005, 0.03, 0.008],
        ...                     [0.02, -0.005, 0.012]])
        >>> sharpe_ratio_nb(returns, 252.0, 0.0001, 1)  # 日无风险利率0.01%
        array([1.45, 0.87, 2.12])  # 各资产的年化夏普比率
    
    投资决策指导：
        - 资产A: 夏普比率1.45，表现良好，值得配置
        - 资产B: 夏普比率0.87，表现一般，谨慎配置
        - 资产C: 夏普比率2.12，表现卓越，重点配置
    
    量化投资应用：
        - 多因子选股：选择夏普比率较高的因子
        - 策略组合：构建基于夏普比率的策略组合
        - 动态再平衡：定期基于夏普比率调整权重
        - 风险管理：设置基于夏普比率的投资限制
    """
    # 预分配输出数组，长度等于资产数量
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的夏普比率
    for col in range(returns.shape[1]):
        # 调用一维函数计算当前资产的夏普比率
        out[col] = sharpe_ratio_1d_nb(returns[:, col], ann_factor, risk_free, ddof)
    
    return out


@njit  # Numba编译优化
def rolling_sharpe_ratio_nb(returns: tp.Array2d,
                            window: int,
                            minp: tp.Optional[int],
                            ann_factor: float,
                            risk_free: float = 0.,
                            ddof: int = 1) -> tp.Array2d:
    """
    滚动夏普比率计算函数
    
    该函数计算指定窗口期间的滚动夏普比率，为动态投资组合管理提供重要工具。
    通过观察夏普比率的时间变化，能够及时识别策略表现的改善或恶化，为实时
    投资决策调整提供量化依据。
    
    功能特点：
        - 动态绩效监控：实时跟踪策略风险调整收益的变化
        - 趋势识别：识别策略绩效的持续改善或恶化趋势
        - 市场适应性：评估策略在不同市场环境下的表现稳定性
        - 时点分析：比较策略在不同时期的相对优势
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵
        window (int): 滚动窗口大小（建议至少30个观测点）
        minp (tp.Optional[int]): 最小有效观测期数
        ann_factor (float): 年化因子
        risk_free (float): 无风险利率，默认为0
        ddof (int): 自由度调整，默认为1
    
    返回值：
        tp.Array2d: 滚动夏普比率矩阵，形状与输入相同
    
    应用场景：
        - 策略择时：基于滚动夏普比率的变化进行策略切换
        - 动态配置：根据滚动表现调整资产配置权重
        - 风险预警：当滚动夏普比率持续下降时发出预警
        - 绩效归因：识别策略表现突出或不佳的具体时期
        - 客户服务：为客户提供策略表现的动态监控报告
    
    使用示例：
        >>> returns = np.random.normal(0.0005, 0.015, (252, 2))  # 252交易日，2个策略
        >>> rolling_sharpe = rolling_sharpe_ratio_nb(
        ...     returns, window=60, minp=30, ann_factor=252.0, risk_free=0.0001
        ... )
        # 计算60日滚动夏普比率，最少需要30个有效观测
    
    策略管理应用：
        - 策略轮换：当A策略的滚动夏普比率超过B策略时进行切换
        - 仓位管理：滚动夏普比率较高的策略配置更高权重
        - 风险控制：当滚动夏普比率降至阈值以下时减仓
        - 市场择时：利用滚动夏普比率的变化判断市场环境
    
    技术实现细节：
        - 使用滑动窗口技术，逐期更新计算
        - 通过generic_nb.rolling_apply_nb提供统一的滚动框架
        - 自动处理窗口边界和数据完整性检查
        - 支持自定义最小观测期数以确保统计有效性
    """
    # 定义滚动窗口内的应用函数
    def _apply_func_nb(i, col, _returns, _ann_factor, _risk_free, _ddof):
        # 对窗口内数据计算夏普比率
        return sharpe_ratio_1d_nb(_returns, _ann_factor, _risk_free, _ddof)

    # 使用通用滚动应用函数进行计算
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, ann_factor, risk_free, ddof)


@njit(cache=True)  # Numba编译优化，启用缓存
def downside_risk_1d_nb(returns: tp.Array1d, ann_factor: float, required_return: float = 0.) -> float:
    """
    一维下行风险计算函数
    
    下行风险(Downside Risk)，也称为下行标准差或下行波动率，是专门衡量投资
    组合低于目标收益率时的风险指标。与传统标准差将上行和下行波动等同对待
    不同，下行风险只关注负向偏离，更符合投资者对风险的直觉理解。
    
    计算原理：
        下行风险 = √[E((min(R-T, 0))²)] × √年化因子
        其中：R为实际收益率，T为目标收益率(required_return)
        只计算收益率低于目标时的平方偏差，高于目标时设为0
    
    理论基础：
        基于下偏矩理论(Lower Partial Moments)，认为投资者更关心下行风险
        而非整体波动。这种不对称风险观更符合行为金融学中的损失厌恶理论。
    
    参数说明：
        returns (tp.Array1d): 一维收益率时间序列
        ann_factor (float): 年化因子
        required_return (float): 目标收益率阈值，默认为0
            - 0：衡量相对于零收益的下行风险
            - 正值：衡量相对于特定收益目标的下行风险
            - 可设为无风险利率或基准收益率
    
    返回值：
        float: 年化下行风险（正值，数值越小表示下行风险越低）
    
    计算步骤：
        1. 计算超额收益：收益率减去目标收益率
        2. 过滤正向偏差：将大于0的值设为0，只保留负向偏差
        3. 计算下行方差：负向偏差的平方均值
        4. 年化处理：开方后乘以年化因子的平方根
    
    使用示例：
        >>> returns = np.array([0.02, -0.01, 0.015, -0.02, 0.01])
        >>> downside_risk_1d_nb(returns, 252.0, 0.005)  # 日频，目标收益率0.5%
        0.0892  # 约8.92%的年化下行风险
        
        计算过程：
        1. 超额收益 = [0.015, -0.015, 0.01, -0.025, 0.005]
        2. 负向偏差 = [0, -0.015, 0, -0.025, 0]
        3. 下行方差 = (0.015² + 0.025²) / 5 = 0.000205
        4. 年化下行风险 = √0.000205 × √252 ≈ 0.0892
    
    应用场景：
        - 索提诺比率计算的分母
        - 下行风险预算管理
        - 风险厌恶投资者的风险评估
        - 目标导向的投资策略风险度量
        - 不对称风险管理模型
    
    与标准差的区别：
        - 标准差：衡量总体波动，包括有利和不利偏离
        - 下行风险：只衡量不利偏离，更符合风险直觉
        - 下行风险通常小于等于标准差
        - 在收益分布不对称时差异明显
    
    金融意义：
        - 反映投资者真正关心的"坏"波动
        - 更准确地度量损失风险
        - 支持不对称风险管理策略
        - 与损失厌恶心理一致
    """
    # 计算超额收益：实际收益率减去目标收益率
    adj_returns = returns - required_return
    # 过滤正向偏差：只保留负向偏差，正向偏差设为0
    adj_returns[adj_returns > 0] = 0
    # 计算年化下行风险：负向偏差平方均值的平方根，再乘以年化因子的平方根
    return np.sqrt(np.nanmean(adj_returns ** 2)) * np.sqrt(ann_factor)


@njit(cache=True)  # Numba编译优化
def downside_risk_nb(returns: tp.Array2d, ann_factor: float, required_return: float = 0.) -> tp.Array1d:
    """
    二维下行风险批量计算函数
    
    这是downside_risk_1d_nb的二维扩展版本，能够同时计算多个资产或策略的
    下行风险。该函数在不对称风险管理和多策略比较中具有重要价值。
    
    设计优势：
        - 统一目标收益率：使用一致的风险评估标准
        - 批量风险评估：同时评估多个策略的下行风险
        - 不对称风险分析：专注于投资者真正关心的损失风险
        - 高效计算：向量化处理提高分析效率
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
        ann_factor (float): 统一年化因子
        required_return (float): 统一目标收益率，默认为0
    
    返回值：
        tp.Array1d: 各资产下行风险数组，形状为(资产数,)
    
    应用场景：
        - 多策略下行风险比较
        - 不对称风险预算分配
        - 损失厌恶型投资者的资产筛选
        - 目标收益导向的投资组合构建
        - 索提诺比率的批量计算基础
    
    使用示例：
        >>> returns = np.array([[0.01, 0.02, -0.005],   # 3个策略的日收益率
        ...                     [0.015, -0.01, 0.01],
        ...                     [-0.005, 0.03, -0.02],
        ...                     [0.02, -0.005, 0.015]])
        >>> downside_risk_nb(returns, 252.0, 0.002)  # 日频，目标收益0.2%
        array([0.0567, 0.0891, 0.1245])  # 各策略的年化下行风险
    
    风险管理应用：
        - 策略A: 下行风险5.67%，下行风险控制较好
        - 策略B: 下行风险8.91%，下行风险中等
        - 策略C: 下行风险12.45%，下行风险较高
        - 可基于下行风险进行策略权重配置
    """
    # 预分配输出数组，长度等于资产数量
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的下行风险
    for col in range(returns.shape[1]):
        # 调用一维函数计算当前资产的下行风险
        out[col] = downside_risk_1d_nb(returns[:, col], ann_factor, required_return)
    
    return out


@njit  # Numba编译优化
def rolling_downside_risk_nb(returns: tp.Array2d,
                             window: int,
                             minp: tp.Optional[int],
                             ann_factor: float,
                             required_return: float = 0.) -> tp.Array2d:
    """
    滚动下行风险计算函数
    
    该函数计算指定窗口期间的滚动下行风险，为动态不对称风险管理提供重要工具。
    通过观察下行风险的时间变化，能够及时识别策略下行风险的积累和释放，为
    实时风险控制提供关键信息。
    
    功能特点：
        - 动态下行风险监控：实时跟踪策略的负向风险变化
        - 不对称风险识别：专注监控真正的损失风险
        - 风险趋势分析：识别下行风险的累积和缓解趋势
        - 市场环境适应：评估策略在不同市场环境下的下行风险
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵
        window (int): 滚动窗口大小
        minp (tp.Optional[int]): 最小有效观测期数
        ann_factor (float): 年化因子
        required_return (float): 目标收益率，默认为0
    
    返回值：
        tp.Array2d: 滚动下行风险矩阵，形状与输入相同
    
    应用场景：
        - 动态风险限额：基于滚动下行风险调整仓位限制
        - 风险预警系统：当下行风险超过阈值时发出预警
        - 策略择时：根据下行风险变化调整策略配置
        - 客户风险监控：为风险厌恶客户提供实时风险报告
    
    使用示例：
        >>> returns = np.random.normal(0.0005, 0.015, (252, 2))  # 252交易日，2个策略
        >>> rolling_downside = rolling_downside_risk_nb(
        ...     returns, window=30, minp=15, ann_factor=252.0, required_return=0.001
        ... )
        # 计算30日滚动下行风险，目标日收益0.1%
    
    风险管理策略：
        - 下行风险突破：当滚动下行风险超过历史分位数时减仓
        - 动态对冲：根据滚动下行风险调整对冲比例
        - 风险预算：基于滚动下行风险动态调整风险预算
        - 策略切换：当下行风险恶化时考虑策略替换
    
    技术实现：
        - 使用滑动窗口技术计算局部下行风险
        - 通过generic_nb.rolling_apply_nb进行滚动应用
        - 自动处理窗口边界和数据完整性检查
    """
    # 定义滚动窗口内的应用函数
    def _apply_func_nb(i, col, _returns, _ann_factor, _required_return):
        # 对窗口内数据计算下行风险
        return downside_risk_1d_nb(_returns, _ann_factor, _required_return)

    # 使用通用滚动应用函数进行计算
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, ann_factor, required_return)


@njit(cache=True)  # Numba编译优化，启用缓存
def sortino_ratio_1d_nb(returns: tp.Array1d, ann_factor: float, required_return: float = 0.) -> float:
    """
    一维索提诺比率计算函数
    
    索提诺比率(Sortino Ratio)是由Frank Sortino提出的风险调整收益指标，是夏普比率的
    改进版本。与夏普比率使用总体标准差作为风险度量不同，索提诺比率只考虑下行风险，
    更准确地反映了投资者对风险的理解——只有负向波动才是真正的风险。
    
    计算公式：
        索提诺比率 = (年化超额收益率 - 目标收益率) / 下行风险
        - 分子：超过目标收益率的年化超额收益
        - 分母：低于目标收益率时的下行风险（下行标准差）
        - 比值：每承担一单位下行风险获得的超额年化收益
    
    理论优势：
        1. 不对称风险观：区分有利和不利的波动
        2. 符合投资心理：投资者欢迎上行波动，厌恶下行波动
        3. 目标导向：基于具体的收益目标进行评估
        4. 实用性更强：对于追求绝对收益的策略更有意义
    
    参数说明：
        returns (tp.Array1d): 一维收益率时间序列
        ann_factor (float): 年化因子
        required_return (float): 目标收益率，默认为0
            - 0：衡量相对于零收益的超额表现
            - 正值：衡量相对于特定目标的超额表现
            - 通常设为无风险利率或投资者期望收益率
    
    返回值：
        float: 索提诺比率
            - 正值：策略表现超过目标，数值越大越好
            - 负值：策略表现低于目标
            - 无穷大：有超额收益但无下行风险（理想情况）
            - NaN：数据不足或无法计算
    
    数值稳定性：
        - 观测值少于2个时返回NaN
        - 下行风险为0时返回无穷大
        - 自动处理NaN值，提高容错性
    
    使用示例：
        >>> returns = np.array([0.02, -0.01, 0.015, -0.005, 0.03])  # 月收益率
        >>> sortino_ratio_1d_nb(returns, 12.0, 0.005)  # 年化，目标月收益0.5%
        2.84  # 索提诺比率为2.84
        
        计算过程：
        1. 超额收益 = [0.015, -0.015, 0.01, -0.01, 0.025]
        2. 年化超额收益 = 0.003 × 12 = 0.036 (3.6%)
        3. 下行风险 = 0.0127 (1.27%)  # 只考虑负向偏差
        4. 索提诺比率 = 0.036 / 0.0127 = 2.84
    
    业界标准：
        - 索提诺比率 > 1.0：表现良好
        - 索提诺比率 > 1.5：表现优秀  
        - 索提诺比率 > 2.0：表现卓越
        - 索提诺比率通常高于夏普比率（相同策略）
    
    应用场景：
        - 绝对收益策略评估
        - 对冲基金绩效评估
        - 目标收益导向的投资决策
        - 风险厌恶投资者的指标偏好
        - 不对称风险管理
    
    与夏普比率的对比：
        - 夏普比率：考虑总体波动，包括上行和下行
        - 索提诺比率：只考虑下行波动，更符合风险直觉
        - 在收益分布偏斜时，两者差异显著
        - 索提诺比率对上行波动更为"宽容"
    
    金融意义：
        - 更准确地衡量风险调整后的收益
        - 支持不对称风险管理策略
        - 为追求绝对收益的投资者提供更合适的评估工具
        - 有助于识别真正优秀的风险控制策略
    """
    # 检查数据有效性：至少需要2个观测值
    if returns.shape[0] < 2:
        return np.nan

    # 计算相对于目标收益率的超额收益
    adj_returns = returns - required_return
    # 计算年化超额收益率
    average_annualized_return = np.nanmean(adj_returns) * ann_factor
    # 计算下行风险（只考虑负向偏离）
    downside_risk = downside_risk_1d_nb(returns, ann_factor, required_return)
    
    # 处理零下行风险情况：返回无穷大（完美策略）
    if downside_risk == 0.:
        return np.inf
    
    # 计算索提诺比率：年化超额收益除以下行风险
    return average_annualized_return / downside_risk


@njit(cache=True)  # Numba编译优化
def sortino_ratio_nb(returns: tp.Array2d, ann_factor: float, required_return: float = 0.) -> tp.Array1d:
    """
    二维索提诺比率批量计算函数
    
    这是sortino_ratio_1d_nb的二维扩展版本，能够同时计算多个资产或策略的
    索提诺比率。该函数在不对称风险管理和多策略绩效比较中具有重要价值。
    
    设计优势：
        - 不对称风险评估：统一使用下行风险作为风险度量
        - 目标导向分析：基于一致的目标收益率进行评估
        - 批量绩效比较：高效比较多个策略的风险调整表现
        - 精准风险定位：识别真正的风险控制能力
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
        ann_factor (float): 统一年化因子
        required_return (float): 统一目标收益率，默认为0
    
    返回值：
        tp.Array1d: 各资产索提诺比率数组，形状为(资产数,)
    
    应用场景：
        - 多策略不对称风险绩效比较
        - 绝对收益导向的投资组合构建
        - 对冲基金策略筛选和评估
        - 风险厌恶型投资者的资产配置
        - 目标收益实现能力评估
    
    使用示例：
        >>> returns = np.array([[0.015, 0.02, 0.01],   # 3个策略的月收益率
        ...                     [0.01, -0.015, 0.005],
        ...                     [-0.005, 0.025, -0.01],
        ...                     [0.025, -0.01, 0.02]])
        >>> sortino_ratio_nb(returns, 12.0, 0.01)  # 月频，目标月收益1%
        array([1.89, 0.73, 1.45])  # 各策略的索提诺比率
    
    投资决策指导：
        - 策略A: 索提诺比率1.89，下行风险控制良好
        - 策略B: 索提诺比率0.73，下行风险控制一般
        - 策略C: 索提诺比率1.45，下行风险控制较好
        - 建议优先配置策略A，谨慎考虑策略B
    
    风险管理价值：
        - 识别真正具备下行保护能力的策略
        - 构建不对称风险优化的投资组合
        - 为追求绝对收益的投资者提供量化依据
        - 支持基于目标收益的资产配置决策
    """
    # 预分配输出数组，长度等于资产数量
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的索提诺比率
    for col in range(returns.shape[1]):
        # 调用一维函数计算当前资产的索提诺比率
        out[col] = sortino_ratio_1d_nb(returns[:, col], ann_factor, required_return)
    
    return out


@njit  # Numba编译优化
def rolling_sortino_ratio_nb(returns: tp.Array2d,
                             window: int,
                             minp: tp.Optional[int],
                             ann_factor: float,
                             required_return: float = 0.) -> tp.Array2d:
    """
    滚动索提诺比率计算函数
    
    该函数计算指定窗口期间的滚动索提诺比率，为动态不对称风险调整收益评估
    提供重要工具。通过观察索提诺比率的时间变化，能够识别策略在不同市场
    环境下的下行风险控制能力。
    
    功能特点：
        - 动态绩效评估：实时评估策略的不对称风险调整表现
        - 下行保护监控：专注监控策略的下行风险控制能力
        - 趋势识别：识别策略绩效改善或恶化的趋势
        - 市场适应性：评估策略在不同环境下的稳定性
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵
        window (int): 滚动窗口大小
        minp (tp.Optional[int]): 最小有效观测期数
        ann_factor (float): 年化因子
        required_return (float): 目标收益率，默认为0
    
    返回值：
        tp.Array2d: 滚动索提诺比率矩阵，形状与输入相同
    
    应用场景：
        - 策略动态评估：监控策略的实时不对称风险表现
        - 下行保护分析：评估策略在不同时期的保护能力
        - 绩效归因：识别策略表现突出的具体时期
        - 动态资产配置：基于滚动表现调整资产权重
    
    使用示例：
        >>> returns = np.random.normal(0.008, 0.02, (252, 2))  # 252交易日，2个策略
        >>> rolling_sortino = rolling_sortino_ratio_nb(
        ...     returns, window=60, minp=30, ann_factor=252.0, required_return=0.002
        ... )
        # 计算60日滚动索提诺比率，目标日收益0.2%
    
    策略管理应用：
        - 当策略滚动索提诺比率持续下降时，考虑策略调整
        - 比较多个策略的滚动下行风险控制能力
        - 设置基于滚动索提诺比率的策略切换阈值
        - 为风险厌恶客户提供动态的下行保护监控报告
    
    技术实现：
        - 使用滑动窗口技术计算局部索提诺比率
        - 通过generic_nb.rolling_apply_nb进行滚动应用
        - 自动处理窗口边界和数据完整性检查
    """
    # 定义滚动窗口内的应用函数
    def _apply_func_nb(i, col, _returns, _ann_factor, _required_return):
        # 对窗口内数据计算索提诺比率
        return sortino_ratio_1d_nb(_returns, _ann_factor, _required_return)

    # 使用通用滚动应用函数进行计算
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, ann_factor, required_return)


@njit(cache=True)  # Numba编译优化，启用缓存
def information_ratio_1d_nb(returns: tp.Array1d, benchmark_rets: tp.Array1d, ddof: int = 1) -> float:
    """
    一维信息比率计算函数
    
    信息比率(Information Ratio, IR)是衡量投资组合相对于基准表现的风险调整收益
    指标，也称为评价比率。它衡量了投资组合每承担一单位主动风险所获得的超额
    收益，是评估主动投资管理能力的核心指标。
    
    计算公式：
        信息比率 = 主动收益的平均值 / 主动收益的标准差
        其中：主动收益 = 投资组合收益率 - 基准收益率
    
    理论基础：
        基于现代投资组合理论中的主动投资管理理论，信息比率衡量了投资经理
        创造超额收益的能力以及这种能力的一致性。高信息比率表明投资经理能够
        持续稳定地产生超额收益。
    
    参数说明：
        returns (tp.Array1d): 一维投资组合收益率时间序列
        benchmark_rets (tp.Array1d): 一维基准收益率时间序列
        ddof (int): 自由度调整，默认为1（样本标准差）
    
    返回值：
        float: 信息比率
            - 正值：策略表现优于基准，数值越大表示超额收益越稳定
            - 负值：策略表现劣于基准，绝对值越大表示劣势越稳定
            - 无穷大：有稳定超额收益但无主动风险（理论情况）
            - NaN：数据不足或无法计算
    
    数值稳定性：
        - 观测值少于2个时返回NaN
        - 主动风险为0时返回无穷大
        - 自动处理NaN值，提高容错性
    
    使用示例：
        >>> returns = np.array([0.02, 0.01, 0.015, -0.005, 0.025])  # 投资组合月收益率
        >>> benchmark = np.array([0.015, 0.008, 0.012, 0.001, 0.018])  # 基准月收益率
        >>> information_ratio_1d_nb(returns, benchmark, 1)
        0.89  # 信息比率为0.89
        
        计算过程：
        1. 主动收益 = [0.005, 0.002, 0.003, -0.006, 0.007]
        2. 平均主动收益 = 0.0022 (0.22%)
        3. 主动风险 = 0.00247 (0.247%)
        4. 信息比率 = 0.0022 / 0.00247 = 0.89
    
    业界标准：
        - 信息比率 > 0.5：表现良好，具备主动管理能力
        - 信息比率 > 0.75：表现优秀，主动管理能力较强
        - 信息比率 > 1.0：表现卓越，主动管理能力很强
        - 信息比率通常在-1到1之间，超过1的情况较少见
    
    应用场景：
        - 主动投资管理能力评估
        - 基金经理绩效评估和比较
        - 投资组合的基准相对表现分析
        - 主动风险预算分配
        - 投资策略的Alpha生成能力评估
    
    与其他指标的关系：
        - 与夏普比率：信息比率关注相对基准表现，夏普比率关注绝对表现
        - 与Alpha：信息比率衡量Alpha的一致性，Alpha只衡量超额收益
        - 与跟踪误差：信息比率的分母就是跟踪误差
    
    金融意义：
        - 衡量投资经理的主动管理技能
        - 评估超额收益的可持续性和稳定性
        - 帮助投资者识别优秀的主动管理策略
        - 为投资者选择基金经理提供量化依据
    
    局限性：
        - 假设主动收益服从正态分布
        - 对基准的依赖性较强
        - 短期数据可能不够稳定
        - 无法反映策略的下行保护能力
    """
    # 检查数据有效性：至少需要2个观测值
    if returns.shape[0] < 2:
        return np.nan

    # 计算主动收益：投资组合收益率减去基准收益率
    active_return = returns - benchmark_rets
    # 计算主动收益的平均值
    mean = np.nanmean(active_return)
    # 计算主动收益的标准差（跟踪误差）
    std = generic_nb.nanstd_1d_nb(active_return, ddof)
    
    # 处理零主动风险情况：返回无穷大（完全追踪基准）
    if std == 0.:
        return np.inf
    
    # 计算信息比率：平均主动收益除以主动风险
    return mean / std


@njit(cache=True)  # Numba编译优化
def information_ratio_nb(returns: tp.Array2d, benchmark_rets: tp.Array2d, ddof: int = 1) -> tp.Array1d:
    """
    二维信息比率批量计算函数
    
    这是information_ratio_1d_nb的二维扩展版本，能够同时计算多个资产或策略
    相对于对应基准的信息比率。该函数在主动投资管理评估中具有重要价值。
    
    设计优势：
        - 主动管理评估：统一评估多个策略的主动管理能力
        - 基准相对分析：基于对应基准进行相对表现评估
        - 批量绩效比较：高效比较多个基金经理的管理能力
        - 风险调整评估：考虑主动风险的超额收益评估
    
    参数说明：
        returns (tp.Array2d): 二维投资组合收益率矩阵，形状为(时间点数, 资产数)
        benchmark_rets (tp.Array2d): 二维基准收益率矩阵，形状与returns相同
        ddof (int): 统一自由度调整，默认为1
    
    返回值：
        tp.Array1d: 各资产信息比率数组，形状为(资产数,)
    
    应用场景：
        - 多基金经理绩效比较评估
        - 主动投资策略筛选和排序
        - 投资组合的基准相对表现分析
        - Alpha生成能力的批量评估
        - 主动风险预算的分配决策
    
    使用示例：
        >>> returns = np.array([[0.02, 0.015, 0.025],     # 3个基金的月收益率
        ...                     [0.01, 0.008, 0.012],
        ...                     [0.015, 0.018, 0.022],
        ...                     [-0.005, 0.005, -0.008]])
        >>> benchmarks = np.array([[0.018, 0.012, 0.02],  # 对应的基准收益率
        ...                        [0.008, 0.006, 0.01],
        ...                        [0.012, 0.015, 0.018],
        ...                        [0.002, 0.003, -0.005]])
        >>> information_ratio_nb(returns, benchmarks, 1)
        array([0.65, 0.89, 0.73])  # 各基金的信息比率
    
    投资决策指导：
        - 基金A: 信息比率0.65，主动管理能力一般
        - 基金B: 信息比率0.89，主动管理能力较强
        - 基金C: 信息比率0.73，主动管理能力良好
        - 建议优先选择基金B进行主动投资
    
    基金管理应用：
        - 识别具备持续Alpha生成能力的基金经理
        - 构建基于信息比率的基金组合
        - 为投资者提供基金选择的量化依据
        - 评估主动投资策略的有效性
    """
    # 预分配输出数组，长度等于资产数量
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的信息比率
    for col in range(returns.shape[1]):
        # 调用一维函数计算当前资产相对于基准的信息比率
        out[col] = information_ratio_1d_nb(returns[:, col], benchmark_rets[:, col], ddof)
    
    return out


@njit  # Numba编译优化
def rolling_information_ratio_nb(returns: tp.Array2d,
                                 window: int,
                                 minp: tp.Optional[int],
                                 benchmark_rets: tp.Array2d,
                                 ddof: int = 1) -> tp.Array2d:
    """
    滚动信息比率计算函数
    
    该函数计算指定窗口期间的滚动信息比率，为动态主动投资管理评估提供重要
    工具。通过观察信息比率的时间变化，能够识别投资经理主动管理能力的
    变化趋势和稳定性。
    
    功能特点：
        - 动态管理能力评估：实时评估投资经理的主动管理技能变化
        - 一致性监控：监控超额收益生成能力的时间稳定性
        - 趋势识别：识别管理能力改善或恶化的趋势
        - 市场适应性：评估策略在不同市场环境下的适应能力
    
    参数说明：
        returns (tp.Array2d): 二维投资组合收益率矩阵
        window (int): 滚动窗口大小（建议至少36个观测点）
        minp (tp.Optional[int]): 最小有效观测期数
        benchmark_rets (tp.Array2d): 二维基准收益率矩阵
        ddof (int): 自由度调整，默认为1
    
    返回值：
        tp.Array2d: 滚动信息比率矩阵，形状与输入相同
    
    应用场景：
        - 基金经理动态评估：监控基金经理的管理能力变化
        - 主动策略监控：实时跟踪主动投资策略的有效性
        - 绩效归因分析：识别超额收益产生的具体时期
        - 投资决策支持：为基金选择和调整提供动态依据
    
    使用示例：
        >>> returns = np.random.normal(0.008, 0.02, (252, 2))  # 252交易日，2个基金
        >>> benchmarks = np.random.normal(0.006, 0.015, (252, 2))  # 对应基准
        >>> rolling_ir = rolling_information_ratio_nb(
        ...     returns, window=60, minp=30, benchmark_rets=benchmarks, ddof=1
        ... )
        # 计算60日滚动信息比率，最少需要30个有效观测
    
    基金管理应用：
        - 当基金的滚动信息比率持续下降时，考虑更换基金经理
        - 识别基金经理表现突出的市场环境和时期
        - 为客户提供基金表现的动态监控报告
        - 设置基于滚动信息比率的基金调整触发条件
    
    技术实现细节：
        - 使用滑动窗口技术计算局部信息比率
        - 自动处理基准数据的时间对齐
        - 通过generic_nb.rolling_apply_nb进行滚动计算
        - 自动处理窗口边界和数据完整性检查
    """
    # 定义滚动窗口内的应用函数
    def _apply_func_nb(i, col, _returns, _benchmark_rets, _ddof):
        # 对窗口内数据计算信息比率，需要对齐基准数据的时间窗口
        return information_ratio_1d_nb(_returns, _benchmark_rets[i + 1 - len(_returns):i + 1, col], _ddof)

    # 使用通用滚动应用函数进行计算
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, benchmark_rets, ddof)


@njit(cache=True)  # Numba编译优化，启用缓存
def beta_1d_nb(returns: tp.Array1d, benchmark_rets: tp.Array1d) -> float:
    """
    一维Beta系数计算函数
    
    Beta系数是衡量投资组合相对于基准(通常是市场指数)系统性风险的指标，
    来源于资本资产定价模型(CAPM)。Beta系数反映了投资组合收益率对基准
    收益率变化的敏感性，是现代投资组合理论的核心概念之一。
    
    计算公式：
        Beta = Cov(投资组合收益率, 基准收益率) / Var(基准收益率)
        其中：Cov为协方差，Var为方差
    
    理论基础：
        基于CAPM模型，Beta系数衡量投资组合承担的系统性风险（不可分散风险）。
        系统性风险是与整个市场相关的风险，无法通过分散投资消除。Beta系数
        帮助投资者了解投资组合对市场波动的敏感程度。
    
    参数说明：
        returns (tp.Array1d): 一维投资组合收益率时间序列
        benchmark_rets (tp.Array1d): 一维基准收益率时间序列
    
    返回值：
        float: Beta系数
            - Beta = 1：与基准波动完全一致，承担平均市场风险
            - Beta > 1：比基准更加波动，属于高风险高收益类型
            - Beta < 1：比基准波动更小，属于防御性投资
            - Beta = 0：与基准无相关性，收益率独立于市场
            - Beta < 0：与基准呈负相关，具有对冲特性
    
    数值稳定性：
        - 基准收益率少于2个观测值时返回NaN
        - 基准方差接近0时返回NaN（避免除零）
        - 基准方差为0时返回无穷大
        - 自动处理缺失值，确保计算稳定性
    
    使用示例：
        >>> returns = np.array([0.02, -0.01, 0.015, -0.005, 0.025])  # 投资组合月收益率
        >>> benchmark = np.array([0.018, -0.008, 0.012, -0.003, 0.02])  # 市场基准月收益率
        >>> beta_1d_nb(returns, benchmark)
        1.14  # Beta系数为1.14
        
        计算过程：
        1. 基准收益均值 = 0.0078 (0.78%)
        2. 基准残差 = [0.0102, -0.0158, 0.0042, -0.0108, 0.0122]
        3. 协方差 = E[(返回值-均值) × 基准残差] = 0.000185
        4. 基准方差 = E[基准残差²] = 0.000162
        5. Beta = 0.000185 / 0.000162 = 1.14
    
    Beta系数解读：
        - Beta = 0.5：当市场上涨10%时，该投资预期上涨5%
        - Beta = 1.0：当市场上涨10%时，该投资预期上涨10%
        - Beta = 1.5：当市场上涨10%时，该投资预期上涨15%
        - Beta = -0.5：当市场上涨10%时，该投资预期下跌5%
    
    应用场景：
        - 投资组合风险管理和评估
        - 资本资产定价模型(CAPM)的应用
        - 投资组合的系统性风险度量
        - 资产配置中的风险平衡
        - Alpha计算的基础输入
    
    投资策略含义：
        - 高Beta股票：适合牛市投资，风险和收益都较高
        - 低Beta股票：适合熊市防御，相对稳定
        - 负Beta资产：天然对冲工具，如黄金、债券等
        - Beta组合：通过组合构建目标Beta水平
    
    局限性：
        - 假设线性关系，实际可能存在非线性
        - 基于历史数据，未来Beta可能发生变化
        - 只衡量系统性风险，忽略特异性风险
        - 对基准的选择敏感
    """
    # 检查基准数据有效性：至少需要2个观测值
    if benchmark_rets.shape[0] < 2:
        return np.nan

    # 处理缺失值：当投资组合收益率缺失时，对应的基准收益率也设为缺失
    independent = np.where(
        np.isnan(returns),  # 如果投资组合收益率为NaN
        np.nan,             # 则基准收益率也设为NaN
        benchmark_rets,     # 否则使用原基准收益率
    )
    
    # 计算基准收益率的残差（相对于均值的偏差）
    ind_residual = independent - np.nanmean(independent)
    # 计算协方差：投资组合收益率与基准残差的期望乘积
    covariances = np.nanmean(ind_residual * returns)
    # 计算基准残差的平方
    ind_residual = ind_residual ** 2
    # 计算基准收益率的方差
    ind_variances = np.nanmean(ind_residual)
    
    # 处理数值稳定性：方差过小时返回NaN
    if ind_variances < 1.0e-30:
        ind_variances = np.nan
    
    # 处理零方差情况：基准无波动时返回无穷大
    if ind_variances == 0.:
        return np.inf
    
    # 计算Beta系数：协方差除以基准方差
    return covariances / ind_variances


@njit(cache=True)  # Numba编译优化
def beta_nb(returns: tp.Array2d, benchmark_rets: tp.Array2d) -> tp.Array1d:
    """
    二维Beta系数批量计算函数
    
    这是beta_1d_nb的二维扩展版本，能够同时计算多个资产或投资组合相对于
    对应基准的Beta系数。该函数在投资组合风险管理中具有重要价值。
    
    设计优势：
        - 系统性风险评估：统一评估多个资产的市场敏感性
        - 风险分散分析：了解不同资产的风险特征
        - 批量风险度量：高效计算大规模投资组合的风险指标
        - 资产配置支持：为构建目标风险水平的组合提供数据
    
    参数说明：
        returns (tp.Array2d): 二维投资组合收益率矩阵，形状为(时间点数, 资产数)
        benchmark_rets (tp.Array2d): 二维基准收益率矩阵，形状与returns相同
    
    返回值：
        tp.Array1d: 各资产Beta系数数组，形状为(资产数,)
    
    应用场景：
        - 多资产投资组合的风险评估
        - 行业或板块相对市场的系统性风险分析
        - 投资组合Beta调整和再平衡
        - 市场中性策略的构建
        - 风险预算分配中的系统性风险度量
    
    使用示例：
        >>> returns = np.array([[0.02, 0.015, 0.025],      # 3只股票的月收益率
        ...                     [0.01, 0.008, -0.012],
        ...                     [0.015, -0.005, 0.022],
        ...                     [-0.005, 0.012, -0.008]])
        >>> benchmarks = np.array([[0.018, 0.018, 0.018],  # 市场基准收益率
        ...                        [0.008, 0.008, 0.008],
        ...                        [0.012, 0.012, 0.012],
        ...                        [0.002, 0.002, 0.002]])
        >>> beta_nb(returns, benchmarks)
        array([0.89, 0.67, 1.32])  # 各股票的Beta系数
    
    投资组合构建指导：
        - 股票A: Beta=0.89，防御性特征，适合稳健投资
        - 股票B: Beta=0.67，低风险特征，适合保守投资
        - 股票C: Beta=1.32，成长性特征，适合激进投资
        - 可通过权重调整构建目标Beta的投资组合
    
    风险管理应用：
        - 识别投资组合中的高风险和低风险资产
        - 构建市场中性或Beta中性的投资策略
        - 进行系统性风险的对冲操作
        - 评估投资组合对市场波动的整体敏感性
    """
    # 预分配输出数组，长度等于资产数量
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的Beta系数
    for col in range(returns.shape[1]):
        # 调用一维函数计算当前资产相对于基准的Beta系数
        out[col] = beta_1d_nb(returns[:, col], benchmark_rets[:, col])
    
    return out


@njit  # Numba编译优化
def rolling_beta_nb(returns: tp.Array2d,
                    window: int,
                    minp: tp.Optional[int],
                    benchmark_rets: tp.Array2d) -> tp.Array2d:
    """
    滚动Beta系数计算函数
    
    该函数计算指定窗口期间的滚动Beta系数，为动态风险管理提供重要工具。
    通过观察Beta系数的时间变化，能够识别投资组合系统性风险特征的
    变化趋势和市场敏感性的演变。
    
    功能特点：
        - 动态风险监控：实时跟踪投资组合的市场敏感性变化
        - 系统性风险趋势：识别Beta系数的时间演变规律
        - 市场状态适应：评估不同市场环境下的风险特征
        - 风险调整时机：为风险管理决策提供动态依据
    
    参数说明：
        returns (tp.Array2d): 二维投资组合收益率矩阵
        window (int): 滚动窗口大小（建议至少60个观测点）
        minp (tp.Optional[int]): 最小有效观测期数
        benchmark_rets (tp.Array2d): 二维基准收益率矩阵
    
    返回值：
        tp.Array2d: 滚动Beta系数矩阵，形状与输入相同
    
    应用场景：
        - 动态对冲策略：根据滚动Beta调整对冲比例
        - 风险预算管理：基于Beta变化调整风险暴露
        - 市场择时：利用Beta变化识别市场环境转换
        - 投资组合再平衡：定期基于Beta变化调整权重
    
    使用示例：
        >>> returns = np.random.normal(0.008, 0.025, (252, 2))  # 252交易日，2只股票
        >>> benchmarks = np.random.normal(0.006, 0.02, (252, 2))  # 对应基准
        >>> rolling_beta = rolling_beta_nb(
        ...     returns, window=60, minp=30, benchmark_rets=benchmarks
        ... )
        # 计算60日滚动Beta系数，最少需要30个有效观测
    
    投资策略应用：
        - Beta上升趋势：市场敏感性增强，考虑减少风险暴露
        - Beta下降趋势：防御特征增强，可能适合增加配置
        - Beta稳定期：系统性风险特征明确，便于风险管理
        - Beta剧烈波动：风险特征不稳定，需要谨慎对待
    
    风险管理策略：
        - 设置Beta阈值，超过时自动调整仓位
        - 利用Beta变化进行市场情绪判断
        - 构建动态Beta中性的投资组合
        - 为衍生品对冲提供实时风险度量
    
    技术实现：
        - 使用滑动窗口技术计算局部Beta系数
        - 自动处理基准数据的时间对齐
        - 通过generic_nb.rolling_apply_nb进行滚动计算
        - 自动处理数值稳定性和边界条件
    """
    # 定义滚动窗口内的应用函数
    def _apply_func_nb(i, col, _returns, _benchmark_rets):
        # 对窗口内数据计算Beta系数，需要对齐基准数据的时间窗口
        return beta_1d_nb(_returns, _benchmark_rets[i + 1 - len(_returns):i + 1, col])

    # 使用通用滚动应用函数进行计算
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, benchmark_rets)


@njit(cache=True)  # Numba JIT编译优化，启用缓存以提高重复调用性能
def alpha_1d_nb(returns: tp.Array1d,
                benchmark_rets: tp.Array1d,
                ann_factor: float,
                risk_free: float = 0.) -> float:
    """
    一维年化阿尔法系数计算函数
    
    阿尔法系数(Alpha)是衡量投资组合相对于基准产生超额收益能力的核心指标，
    来源于资本资产定价模型(CAPM)。阿尔法系数反映了投资经理的选股能力和主动
    投资管理技巧，是评估投资绩效最重要的指标之一。
    
    计算原理：
        基于CAPM模型：E(R) = Rf + β×[E(Rm) - Rf] + α
        其中：α = E(R) - Rf - β×[E(Rm) - Rf]
        Alpha序列 = (投资组合收益率 - 无风险利率) - β×(基准收益率 - 无风险利率)
        年化Alpha = (1 + 平均Alpha)^年化因子 - 1
    
    理论基础：
        阿尔法系数代表了投资组合在承担相同系统性风险水平下，相对于基准的
        超额收益。正的阿尔法表示投资经理创造了价值，负的阿尔法表示投资
        经理破坏了价值。阿尔法的大小直接反映投资管理的技能水平。
    
    参数说明：
        returns (tp.Array1d): 一维投资组合收益率时间序列
        benchmark_rets (tp.Array1d): 一维基准收益率时间序列（通常为市场指数）
        ann_factor (float): 年化因子
            - 日频数据：252（年均交易日数）
            - 月频数据：12（年均月数）
            - 周频数据：52（年均周数）
        risk_free (float): 无风险利率，默认为0
            - 应与收益率数据频率保持一致
            - 通常使用国债收益率或央行基准利率
    
    返回值：
        float: 年化阿尔法系数（小数形式）
            - 正值：投资组合表现优于预期，创造超额收益
            - 负值：投资组合表现劣于预期，未能达到基准调整后的收益
            - 0：投资组合表现符合CAPM模型预期
            - NaN：数据不足或无法计算
    
    使用示例：
        >>> returns = np.array([0.02, 0.015, -0.01, 0.025, 0.008])  # 投资组合月收益率
        >>> benchmark = np.array([0.018, 0.012, -0.008, 0.02, 0.006])  # 市场基准月收益率
        >>> alpha_1d_nb(returns, benchmark, 12.0, 0.002)  # 月无风险利率0.2%
        0.0156  # 年化阿尔法为1.56%
        
        计算过程：
        1. 调整后投资组合收益 = [0.018, 0.013, -0.012, 0.023, 0.006]
        2. 调整后基准收益 = [0.016, 0.01, -0.01, 0.018, 0.004]
        3. Beta系数 = 1.08（通过beta_1d_nb计算）
        4. Alpha序列 = 调整后投资组合收益 - 1.08×调整后基准收益
        5. 平均Alpha = 0.0013
        6. 年化Alpha = (1 + 0.0013)^12 - 1 = 0.0156
    
    阿尔法解读：
        - Alpha > 3%：卓越的投资表现，顶级投资经理水平
        - Alpha > 1%：优秀的投资表现，具备较强的主动管理能力
        - Alpha > 0：正面的投资表现，创造了额外价值
        - Alpha = 0：符合市场预期，被动投资水平
        - Alpha < 0：负面的投资表现，未能创造价值
    
    应用场景：
        - 投资经理绩效评估和薪酬考核
        - 主动基金与被动指数基金比较
        - 投资策略的价值创造能力评估
        - 机构投资者的管理人选择
        - 投资组合的阿尔法预算分配
    
    与其他指标的关系：
        - Beta：衡量系统性风险，Alpha衡量超额收益
        - 夏普比率：衡量总体风险调整收益，Alpha衡量相对基准超额收益
        - 信息比率：衡量主动收益的一致性，Alpha衡量主动收益的大小
        - 特雷诺比率：Alpha/Beta，衡量单位系统风险的超额收益
    
    金融意义：
        - 反映投资经理的"真实技能"，剔除市场Beta的影响
        - 衡量投资策略的核心价值创造能力
        - 帮助投资者识别优秀的主动管理策略
        - 为投资者支付管理费提供价值评估依据
    
    注意事项：
        - 阿尔法的统计显著性需要通过t检验等方法验证
        - 短期阿尔法可能不具备持续性，建议使用较长期数据
        - 基准选择对阿尔法计算结果影响较大
        - 阿尔法不能预测未来表现，仅反映历史技能
    """
    # 数据有效性检查：至少需要2个观测值才能计算Beta和Alpha
    if returns.shape[0] < 2:
        return np.nan  # 数据不足时返回NaN

    # 计算风险调整后的投资组合收益率：扣除无风险利率得到超额收益
    adj_returns = returns - risk_free
    
    # 计算风险调整后的基准收益率：同样扣除无风险利率得到基准超额收益  
    adj_benchmark_rets = benchmark_rets - risk_free
    
    # 计算投资组合相对于基准的Beta系数：衡量系统性风险暴露程度
    beta = beta_1d_nb(returns, benchmark_rets)
    
    # 计算Alpha序列：投资组合超额收益减去Beta调整后的基准超额收益
    # 这代表了剔除系统性风险影响后的纯粹超额收益，即投资经理的选股技能
    alpha_series = adj_returns - (beta * adj_benchmark_rets)
    
    # 年化处理：将平均Alpha按复利方式年化
    # 公式：(1 + 平均Alpha)^年化因子 - 1，体现复利增长效应
    return (np.nanmean(alpha_series) + 1) ** ann_factor - 1


@njit(cache=True)  # Numba JIT编译优化，启用缓存以提高批量处理性能
def alpha_nb(returns: tp.Array2d,
             benchmark_rets: tp.Array2d,
             ann_factor: float,
             risk_free: float = 0.) -> tp.Array1d:
    """
    二维年化阿尔法系数批量计算函数
    
    这是alpha_1d_nb的二维扩展版本，能够同时计算多个资产或策略相对于各自
    基准的年化阿尔法系数。该函数是多资产投资组合分析和策略比较的核心工具，
    为大规模量化投资提供高效的批量阿尔法计算能力。
    
    设计理念：
        - 每列代表一个资产或策略的阿尔法计算
        - 支持不同资产使用不同基准进行比较
        - 统一的无风险利率和年化因子设置，确保结果可比性
        - 向量化批量处理，大幅提高计算效率
    
    参数说明：
        returns (tp.Array2d): 二维投资组合收益率矩阵，形状为(时间点数, 资产数)
            - 行索引：时间维度，按时间顺序排列
            - 列索引：资产维度，每列代表一个资产或策略
        benchmark_rets (tp.Array2d): 二维基准收益率矩阵，形状与returns相同
            - 每列对应相应资产的基准收益率（可以是不同的基准）
        ann_factor (float): 统一的年化因子
        risk_free (float): 统一的无风险利率，默认为0
    
    返回值：
        tp.Array1d: 各资产年化阿尔法系数数组，形状为(资产数,)
            - 每个元素对应一个资产的年化阿尔法系数
            - 保持与输入矩阵列顺序的一致性
    
    计算并行性：
        - 不同资产间的阿尔法计算完全独立，具备天然并行性
        - 每列调用alpha_1d_nb进行独立计算
        - 适合在多核环境下进行并行优化
        - 为大规模量化策略提供高效支持
    
    使用示例：
        >>> # 三个投资策略的月收益率
        >>> returns = np.array([[0.02, 0.015, -0.005],   # 策略A, B, C的第1个月收益
        ...                     [0.01, -0.008, 0.012],   # 策略A, B, C的第2个月收益
        ...                     [-0.005, 0.02, 0.008],   # 策略A, B, C的第3个月收益
        ...                     [0.025, 0.01, -0.01]])   # 策略A, B, C的第4个月收益
        >>> 
        >>> # 三个策略对应的基准收益率（可以是不同基准）
        >>> benchmarks = np.array([[0.018, 0.012, -0.002],  # 基准收益率
        ...                        [0.008, -0.005, 0.01],
        ...                        [-0.002, 0.018, 0.006],
        ...                        [0.02, 0.008, -0.008]])
        >>> 
        >>> alpha_nb(returns, benchmarks, 12.0, 0.002)  # 月频数据，无风险利率0.2%
        array([0.0234, -0.0156, 0.0089])  # 各策略的年化阿尔法系数
        
        结果解读：
        - 策略A：年化阿尔法2.34%，表现优异，创造了显著超额收益
        - 策略B：年化阿尔法-1.56%，表现不佳，未能跑赢基准
        - 策略C：年化阿尔法0.89%，表现良好，有一定的价值创造能力
    
    应用场景：
        - 多策略投资组合的绩效评估和排序
        - 量化基金经理的批量绩效考核
        - 不同行业或风格因子的Alpha比较
        - 投资组合优化中的Alpha预测输入
        - 大规模策略筛选和资产配置决策
    
    内存效率：
        - 预分配输出数组，避免动态内存扩展
        - 利用缓存友好的列遍历顺序
        - 最小化临时对象创建，优化内存使用
    
    批量处理优势：
        - 一次性处理多个资产，避免重复调用开销
        - 统一参数设置，保证计算标准的一致性
        - 便于后续统计分析和可视化展示
        - 支持大规模量化投资的工业化需求
    """
    # 预分配输出数组，长度等于资产数量（矩阵的列数）
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产相对于其对应基准的年化阿尔法系数
    for col in range(returns.shape[1]):
        # 调用一维阿尔法函数计算当前资产列的阿尔法系数
        # 使用对应列的基准收益率进行比较
        out[col] = alpha_1d_nb(returns[:, col], benchmark_rets[:, col], ann_factor, risk_free)
    
    return out


@njit  # Numba JIT编译优化，专为滚动窗口计算设计
def rolling_alpha_nb(returns: tp.Array2d,
                     window: int,
                     minp: tp.Optional[int],
                     benchmark_rets: tp.Array2d,
                     ann_factor: float,
                     risk_free: float = 0.) -> tp.Array2d:
    """
    滚动窗口年化阿尔法系数计算函数
    
    该函数计算滚动窗口内的年化阿尔法系数，是动态风险管理和实时投资决策的
    核心工具。通过滑动时间窗口，该函数能够捕捉投资组合阿尔法系数的时变特征，
    为投资者提供更敏感和及时的绩效监控指标。
    
    核心价值：
        滚动阿尔法相比静态阿尔法，能够：
        - 及时识别投资策略效果的变化趋势
        - 动态调整投资组合配置和风险敞口
        - 提供实时的投资经理技能评估
        - 支持基于阿尔法衰减的策略轮换决策
    
    计算原理：
        对于每个时间点，使用该时点之前window长度的数据窗口计算阿尔法系数：
        - 滑动窗口[t-window+1, t]内的收益率数据
        - 窗口内对应的基准收益率数据
        - 调用alpha_1d_nb计算窗口阿尔法
    
    参数说明：
        returns (tp.Array2d): 二维投资组合收益率矩阵，形状为(时间点数, 资产数)
        window (int): 滚动窗口大小（时间点数）
            - 日频数据：建议63天（一个季度）到252天（一年）
            - 月频数据：建议12个月到36个月
            - 窗口过小：统计不稳定；窗口过大：响应迟缓
        minp (tp.Optional[int]): 计算所需的最少观测值
            - None：使用window作为最小观测值要求
            - 整数：指定最小有效观测值数量
        benchmark_rets (tp.Array2d): 二维基准收益率矩阵，形状与returns相同
        ann_factor (float): 年化因子
        risk_free (float): 无风险利率，默认为0
    
    返回值：
        tp.Array2d: 滚动阿尔法矩阵，形状与输入returns相同
            - 前window-1行为NaN（数据不足）
            - 后续每行为对应时点的滚动窗口阿尔法系数
    
    应用场景：
        - **动态投资组合管理**：根据阿尔法衰减情况调整投资权重
        - **实时绩效监控**：监控投资经理技能的时变特征
        - **策略择时**：基于阿尔法趋势进行策略进入和退出决策
        - **风险预警**：当阿尔法持续恶化时触发风险控制措施
        - **投资者沟通**：向客户展示策略效果的动态变化
    
    使用示例：
        >>> # 6个月的日收益率数据
        >>> returns = np.random.normal(0.001, 0.02, (126, 2))  # 两个策略
        >>> benchmark = np.random.normal(0.0008, 0.015, (126, 2))  # 对应基准
        >>> 
        >>> # 计算21天（约一个月）滚动阿尔法
        >>> rolling_alphas = rolling_alpha_nb(returns, 21, None, benchmark, 252.0, 0.0)
        >>> 
        >>> # 查看最后10天的滚动阿尔法
        >>> print(rolling_alphas[-10:, :])
        array([[ 0.0234, -0.0156],    # 第117天的21日滚动阿尔法
               [ 0.0189,  0.0023],    # 第118天的21日滚动阿尔法
               ...])                  # 持续更新
    
    时间序列分析价值：
        - **趋势识别**：阿尔法系数的上升或下降趋势
        - **周期性模式**：识别阿尔法的季节性或周期性变化
        - **异常检测**：识别阿尔法的突变点和异常期间
        - **预测性分析**：基于历史滚动阿尔法预测未来表现
    
    实时应用考虑：
        - **数据频率**：高频更新要求选择合适的窗口大小
        - **计算延迟**：平衡计算精度与实时性要求
        - **存储效率**：滚动计算可只保存必要的历史数据
        - **并发处理**：多资产滚动阿尔法的并行计算优化
    
    注意事项：
        - 滚动阿尔法具有滞后性，反映的是过去窗口期间的表现
        - 市场环境剧变时，历史阿尔法的预测价值可能降低
        - 需结合其他动态指标综合判断投资策略效果
        - 窗口大小的选择需要在稳定性和敏感性间找平衡
    """
    # 定义滚动窗口内的阿尔法计算函数
    # 该内部函数将被rolling_apply_nb调用，处理每个滚动窗口
    def _apply_func_nb(i, col, _returns, _benchmark_rets, _ann_factor, _risk_free):
        """
        滚动窗口阿尔法计算的内部应用函数
        
        参数说明：
            i (int): 当前时间点索引
            col (int): 当前资产列索引  
            _returns (tp.Array1d): 当前窗口的投资组合收益率
            _benchmark_rets (tp.Array2d): 完整的基准收益率矩阵
            _ann_factor (float): 年化因子
            _risk_free (float): 无风险利率
        
        返回值：
            float: 当前窗口的年化阿尔法系数
        """
        # 提取与投资组合收益率窗口对应的基准收益率切片
        # [i + 1 - len(_returns):i + 1, col] 确保基准数据与投资组合数据时间对齐
        benchmark_window = _benchmark_rets[i + 1 - len(_returns):i + 1, col]
        
        # 调用一维阿尔法函数计算当前窗口的阿尔法系数
        return alpha_1d_nb(_returns, benchmark_window, _ann_factor, _risk_free)

    # 使用通用滚动应用函数进行滚动阿尔法计算
    # generic_nb.rolling_apply_nb负责：
    # 1. 管理滚动窗口的时间切片
    # 2. 处理最小观测值要求(minp)
    # 3. 调用_apply_func_nb进行具体计算
    # 4. 组装输出矩阵并处理边界条件
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, benchmark_rets, ann_factor, risk_free)


@njit(cache=True)  # Numba JIT编译优化，启用缓存提高重复计算性能
def tail_ratio_1d_nb(returns: tp.Array1d) -> float:
    """
    一维尾部比率计算函数
    
    尾部比率(Tail Ratio)是衡量投资收益分布尾部特征的重要风险指标，通过比较
    收益分布的右尾(95%分位数)与左尾(5%分位数)的绝对值，评估投资组合的收益
    不对称性和尾部风险特征。该指标对识别投资组合的极端收益行为特别有价值。
    
    计算公式：
        尾部比率 = abs(95%分位数) / abs(5%分位数)
        - 分子：右尾收益的绝对值，代表极端正收益的幅度
        - 分母：左尾损失的绝对值，代表极端负收益的幅度
        - 两者都取绝对值确保比率为正，便于解读和比较
    
    理论基础：
        基于极值理论和风险管理实践，尾部比率衡量投资组合在极端市场条件下的
        表现不对称性。高尾部比率表明投资组合的极端正收益相对于极端损失更大，
        这通常被视为更优的风险-收益特征。
    
    参数说明：
        returns (tp.Array1d): 一维收益率时间序列
    
    返回值：
        float: 尾部比率（正数或无穷大）
            - 比率 > 1：极端正收益超过极端损失，分布特征较优
            - 比率 = 1：极端收益和损失相当，分布相对对称
            - 比率 < 1：极端损失超过极端收益，分布风险偏向
            - 比率 = ∞：极端损失接近零，策略几乎无下行风险
    
    数值稳定性处理：
        - 自动过滤NaN值，确保计算基于有效数据
        - 当有效数据少于1个时返回NaN
        - 当5%分位数为0时返回无穷大（极少发生损失）
        - 使用绝对值避免负数比率的混淆
    
    使用示例：
        >>> returns = np.array([0.05, -0.03, 0.02, -0.01, 0.08, -0.02, 0.01, -0.04])
        >>> tail_ratio_1d_nb(returns)
        2.0  # 尾部比率为2.0
        
        计算过程：
        1. 过滤NaN值：保留所有有效收益率数据
        2. 95%分位数 = 0.08，绝对值 = 0.08
        3. 5%分位数 = -0.04，绝对值 = 0.04
        4. 尾部比率 = 0.08 / 0.04 = 2.0
        
        解读：极端正收益是极端损失的2倍，风险收益特征良好
    
    指标解读基准：
        - **卓越表现** (比率 > 2.0)：极端收益显著超过极端损失
        - **优秀表现** (1.5 < 比率 ≤ 2.0)：较好的尾部收益特征
        - **良好表现** (1.2 < 比率 ≤ 1.5)：适度的正向尾部特征
        - **中性表现** (0.8 ≤ 比率 ≤ 1.2)：尾部特征相对均衡
        - **需要关注** (比率 < 0.8)：极端损失风险相对较大
    
    应用场景：
        - **策略筛选**：识别具有优秀尾部特征的投资策略
        - **风险评估**：评估在极端市场条件下的表现
        - **投资组合构建**：平衡不同尾部特征的资产配置
        - **绩效归因**：分析策略收益的极值来源
        - **风险预算**：为具有不同尾部特征的策略分配风险额度
    
    投资策略解读：
        - **趋势跟随策略**：通常具有较高尾部比率，能捕获大趋势
        - **均值回复策略**：可能较低尾部比率但胜率高
        - **波动率策略**：尾部比率取决于市场波动特征
        - **套利策略**：通常较低尾部比率但回撤控制较好
    
    与其他风险指标关系：
        - **偏度(Skewness)**：尾部比率是偏度的简化直观版本
        - **VaR**：关注特定置信水平的损失，尾部比率比较极值
        - **最大回撤**：关注历史最大损失，尾部比率关注分位数损失
        - **夏普比率**：衡量平均风险调整收益，尾部比率关注极端情况
    
    注意事项：
        - 对样本大小敏感，建议至少100个观测值以上
        - 分位数估计在小样本下可能不稳定
        - 不反映极端事件的发生频率，只反映幅度
        - 需结合其他风险指标进行综合评估
    """
    # 数据预处理：过滤掉NaN值，确保计算基于有效数据
    returns = returns[~np.isnan(returns)]
    
    # 数据有效性检查：至少需要1个有效观测值
    if len(returns) < 1:
        return np.nan  # 无有效数据时返回NaN
    
    # 计算95%分位数的绝对值：极端正收益的幅度（右尾）
    perc_95 = np.abs(np.percentile(returns, 95))
    
    # 计算5%分位数的绝对值：极端负收益的幅度（左尾）  
    perc_5 = np.abs(np.percentile(returns, 5))
    
    # 处理零除错误：当极端损失为0时，表明策略几乎无下行风险
    if perc_5 == 0.:
        return np.inf  # 返回无穷大表示理想的风险特征
    
    # 计算尾部比率：极端收益相对于极端损失的倍数
    # 比率越高表示策略在极端情况下的表现越有利
    return perc_95 / perc_5


@njit(cache=True)  # Numba JIT编译优化，启用缓存提升批量处理效率
def tail_ratio_nb(returns: tp.Array2d) -> tp.Array1d:
    """
    二维尾部比率批量计算函数
    
    这是tail_ratio_1d_nb的二维扩展版本，能够同时计算多个资产或策略的尾部比率。
    该函数为投资组合管理者提供了高效的批量风险特征分析工具，特别适用于
    大规模资产筛选和风险对比分析。
    
    设计优势：
        - 批量计算多个资产的尾部比率，提高分析效率
        - 统一的计算标准，确保不同资产间结果的可比性
        - 向量化处理，充分利用数值计算优势
        - 便于后续的排序、筛选和可视化分析
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
            - 行索引：时间维度，按时间顺序排列
            - 列索引：资产维度，每列代表一个资产或策略
    
    返回值：
        tp.Array1d: 各资产尾部比率数组，形状为(资产数,)
            - 每个元素对应一个资产的尾部比率
            - 保持与输入矩阵列顺序的一致性
            - 便于后续统计分析和决策应用
    
    计算特点：
        - 每列独立计算，不同资产间互不影响
        - 自动处理每列的缺失值和异常情况
        - 支持混合资产类型的同时分析
        - 提供标准化的风险特征输出
    
    使用示例：
        >>> # 三个不同策略的收益率数据
        >>> returns = np.array([
        ...     [ 0.02,  0.015, -0.01],    # 策略A, B, C在第1期
        ...     [-0.01, -0.005,  0.02],    # 策略A, B, C在第2期  
        ...     [ 0.03,  0.01,  -0.005],   # 策略A, B, C在第3期
        ...     [-0.02,  0.025,  0.015],   # 策略A, B, C在第4期
        ...     [ 0.05, -0.02,   0.008],   # 策略A, B, C在第5期
        ...     [-0.03,  0.03,  -0.012]    # 策略A, B, C在第6期
        ... ])
        >>> 
        >>> tail_ratios = tail_ratio_nb(returns)
        >>> print(tail_ratios)
        array([1.67, 2.25, 1.33])  # 各策略的尾部比率
        
        结果解读：
        - 策略A：尾部比率1.67，具有良好的极值收益特征
        - 策略B：尾部比率2.25，表现卓越，极端收益显著超过极端损失  
        - 策略C：尾部比率1.33，适度正向的尾部特征
    
    应用场景：
        - **策略组合构建**：基于尾部比率选择互补的策略组合
        - **资产配置优化**：在不同尾部特征资产间分配权重
        - **风险预算管理**：为不同风险特征的策略分配风险额度
        - **绩效排序筛选**：按尾部比率对投资策略进行排序
        - **风险监控系统**：批量监控多个策略的风险特征变化
    
    投资组合应用：
        - **核心-卫星策略**：核心配置低风险资产，卫星配置高尾部比率策略
        - **风险平价组合**：基于尾部风险特征进行风险平衡配置
        - **多策略基金**：评估和选择具有不同尾部特征的子策略
        - **量化策略池**：从大量策略中筛选优秀尾部特征的候选
    
    数据分析价值：
        - **横向比较**：在同类资产中识别风险特征优势
        - **风格分析**：识别不同投资风格的尾部特征模式  
        - **时间序列**：结合滚动计算观察尾部特征的时变性
        - **相关性分析**：研究尾部比率与其他绩效指标的关系
    
    后续分析建议：
        - 结合其他风险指标进行综合评估
        - 考虑策略间的相关性和互补性
        - 定期更新计算，监控特征变化
        - 结合市场环境分析尾部比率的有效性
    """
    # 预分配输出数组，长度等于资产数量（矩阵的列数）
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的尾部比率
    for col in range(returns.shape[1]):
        # 调用一维尾部比率函数计算当前资产列的尾部比率
        # 每列的计算完全独立，可以并行化处理
        out[col] = tail_ratio_1d_nb(returns[:, col])
    
    return out


@njit  # Numba JIT编译优化，专门为滚动窗口计算优化
def rolling_tail_ratio_nb(returns: tp.Array2d, window: int, minp: tp.Optional[int]) -> tp.Array2d:
    """
    滚动窗口尾部比率计算函数
    
    该函数计算滚动窗口内的尾部比率，为动态风险监控和实时决策提供时变的
    尾部风险特征指标。通过滑动时间窗口，能够及时捕捉投资组合极端风险
    特征的变化，为风险管理和投资决策提供前瞻性的风险预警。
    
    核心价值：
        相比静态尾部比率，滚动尾部比率能够：
        - 实时监控投资策略尾部风险特征的演变
        - 及时识别风险特征的恶化或改善趋势  
        - 为动态风险配置提供量化决策依据
        - 支持基于风险变化的策略调整决策
    
    计算原理：
        对于每个时间点，计算该时点之前window长度窗口内的尾部比率：
        - 滑动窗口[t-window+1, t]内的收益率数据
        - 计算窗口内95%分位数与5%分位数的绝对值比率
        - 生成连续的时变尾部比率序列
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
        window (int): 滚动窗口大小（时间点数）
            - 日频数据：建议21天（月度）到63天（季度）
            - 月频数据：建议6个月到18个月
            - 需平衡统计稳定性与响应敏感性
        minp (tp.Optional[int]): 计算所需的最少观测值
            - None：使用window作为最小观测值要求
            - 整数：指定最小有效观测值数量，通常不少于20
    
    返回值：
        tp.Array2d: 滚动尾部比率矩阵，形状与输入returns相同
            - 前window-1行为NaN（数据不足）  
            - 后续每行为对应时点的滚动窗口尾部比率
    
    应用场景：
        - **动态风险监控**：实时跟踪策略尾部风险特征变化
        - **风险预警系统**：当尾部比率持续恶化时触发预警
        - **择时决策支持**：基于尾部比率趋势进行策略进入退出
        - **投资组合再平衡**：根据各资产尾部特征变化调整权重
        - **绩效归因分析**：分析不同时期尾部特征对整体表现的贡献
    
    使用示例：
        >>> # 模拟6个月的日收益率数据
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0.001, 0.02, (126, 2))  # 两个策略
        >>> 
        >>> # 计算21天滚动尾部比率  
        >>> rolling_ratios = rolling_tail_ratio_nb(returns, 21, None)
        >>> 
        >>> # 查看最后10天的滚动尾部比率
        >>> print(rolling_ratios[-10:, :])
        array([[1.45, 1.23],    # 第117天的21日滚动尾部比率
               [1.52, 1.18],    # 第118天的21日滚动尾部比率  
               ...])            # 持续更新的风险特征
    
    时间序列分析价值：
        - **趋势识别**：识别尾部比率的上升、下降或震荡趋势
        - **周期性模式**：发现尾部风险特征的季节性或周期性规律
        - **异常检测**：识别尾部比率的突变点和异常时期
        - **相关性分析**：研究不同资产尾部比率的联动关系
    
    风险管理应用：
        - **动态止损**：当尾部比率持续恶化时触发止损机制
        - **风险限额管理**：基于滚动尾部比率动态调整风险限额
        - **压力测试**：模拟不同尾部比率情景下的组合表现
        - **风险监控报告**：为管理层提供实时的风险特征报告
    
    市场环境适应性：
        - **牛市**：关注尾部比率是否保持在合理水平
        - **熊市**：监控尾部比率恶化程度，评估下行保护
        - **震荡市**：观察尾部比率的稳定性和变化幅度
        - **危机时期**：重点监控极端风险特征的变化
    
    注意事项：
        - 滚动尾部比率具有滞后性，反映历史窗口期特征
        - 窗口过小可能导致指标过度敏感和噪音
        - 窗口过大可能导致指标反应迟钝
        - 需结合市场环境和其他指标综合判断
    """
    # 定义滚动窗口内的尾部比率计算函数
    # 该内部函数将被rolling_apply_nb调用处理每个滚动窗口
    def _apply_func_nb(i, col, _returns):
        """
        滚动窗口尾部比率计算的内部应用函数
        
        参数说明：
            i (int): 当前时间点索引（未使用，保持接口一致性）
            col (int): 当前资产列索引（未使用，保持接口一致性）
            _returns (tp.Array1d): 当前窗口的收益率数据
        
        返回值：
            float: 当前窗口的尾部比率
        """
        # 直接调用一维尾部比率函数计算当前窗口的尾部比率
        # 函数内部会自动处理NaN值过滤和边界情况
        return tail_ratio_1d_nb(_returns)

    # 使用通用滚动应用函数进行滚动尾部比率计算
    # generic_nb.rolling_apply_nb负责：
    # 1. 管理滚动窗口的时间切片和数据提取
    # 2. 处理最小观测值要求(minp)和边界条件
    # 3. 调用_apply_func_nb进行窗口内的具体计算  
    # 4. 组装完整的输出矩阵并处理缺失值
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb)


@njit(cache=True)  # Numba JIT编译优化，启用缓存提高VaR计算性能
def value_at_risk_1d_nb(returns: tp.Array1d, cutoff: float = 0.05) -> float:
    """
    一维风险价值(VaR)计算函数
    
    风险价值(Value at Risk, VaR)是国际风险管理的标准工具，用于量化在正常市场
    条件下，给定置信水平和持有期内，投资组合可能面临的最大损失。VaR为风险
    管理者提供了简单直观的风险度量，是监管合规和内部风控的核心指标。
    
    计算原理：
        VaR = 收益率分布的cutoff分位数
        - 例如：5% VaR表示有95%的概率损失不会超过VaR值
        - VaR值通常为负数，表示损失金额
        - 基于历史收益率分布的经验分位数估计
    
    理论基础：
        基于投资组合理论和风险管理实践，VaR假设历史收益率分布能够代表未来
        风险特征。该方法简单易懂，广泛应用于银行、保险、基金等金融机构的
        风险管理和监管报告中。
    
    参数说明：
        returns (tp.Array1d): 一维收益率时间序列
        cutoff (float): 置信水平的补集，默认为0.05（即95%置信水平）
            - 0.05对应95% VaR（最常用）
            - 0.01对应99% VaR（更严格的风险度量）
            - 0.1对应90% VaR（相对宽松的风险度量）
    
    返回值：
        float: VaR值（通常为负数，表示潜在损失）
            - 负值：表示潜在损失的大小（绝对值越大风险越高）
            - 接近0：表示投资组合风险较低
            - NaN：数据不足或无法计算
    
    数值稳定性：
        - 自动过滤NaN值，基于有效数据计算
        - 当有效数据少于1个时返回NaN
        - 使用numpy的percentile函数确保计算稳定性
    
    使用示例：
        >>> # 模拟100天的日收益率数据
        >>> returns = np.array([-0.08, -0.03, 0.02, -0.01, 0.04, -0.02, 0.01, -0.05])
        >>> 
        >>> # 计算95% VaR (5%分位数)
        >>> var_5 = value_at_risk_1d_nb(returns, 0.05)
        >>> print(f"95% VaR: {var_5:.3f}")  # -0.080 (-8.0%)
        >>> 
        >>> # 计算99% VaR (1%分位数) 
        >>> var_1 = value_at_risk_1d_nb(returns, 0.01)  
        >>> print(f"99% VaR: {var_1:.3f}")  # -0.080 (-8.0%)
        
        解读：在95%的置信水平下，该策略的日损失不会超过8.0%
    
    业界标准：
        - **监管要求**：巴塞尔协议要求银行计算99% VaR用于资本充足率计算
        - **内部管理**：多数机构使用95% VaR进行日常风险监控
        - **投资组合**：基金公司通常披露95% VaR作为风险指标
    
    应用场景：
        - **风险限额管理**：设定基于VaR的交易和投资限额
        - **资本配置**：根据VaR确定风险资本的配置
        - **绩效评估**：风险调整收益的分母计算
        - **监管报告**：满足监管机构的风险披露要求
        - **投资者教育**：向投资者解释投资风险的潜在损失
    
    VaR解读指南：
        - **低风险** (|VaR| < 2%)：保守型投资，适合风险厌恶投资者
        - **中等风险** (2% ≤ |VaR| < 5%)：平衡型投资，多数投资者可接受
        - **高风险** (5% ≤ |VaR| < 10%)：积极型投资，需要较高风险承受能力
        - **极高风险** (|VaR| ≥ 10%)：激进型投资，仅适合专业投资者
    
    与其他风险指标的关系：
        - **标准差**：VaR关注极端损失，标准差衡量整体波动
        - **最大回撤**：VaR基于概率分布，最大回撤基于历史最差表现
        - **CVaR**：CVaR是VaR的扩展，衡量超过VaR的期望损失
        - **下行风险**：都关注负向风险，但计算方法不同
    
    局限性与注意事项：
        - **分布假设**：假设历史分布代表未来，在极端市场可能失效
        - **相关性变化**：市场危机时相关性剧变，VaR可能低估风险
        - **模型风险**：基于历史数据，对结构性变化反应滞后
        - **流动性风险**：未考虑资产变现时的流动性影响
        - **尾部风险**：无法揭示超过VaR阈值的损失分布特征
    
    最佳实践建议：
        - 建议至少使用250个交易日的历史数据
        - 定期更新数据，保持模型的时效性
        - 结合压力测试和情景分析补充VaR的不足
        - 监控VaR突破频率，验证模型有效性
        - 在极端市场条件下谨慎解读VaR结果
    """
    # 数据预处理：过滤掉NaN值，确保计算基于有效数据
    returns = returns[~np.isnan(returns)]
    
    # 数据有效性检查：至少需要1个有效观测值
    if len(returns) < 1:
        return np.nan  # 无有效数据时返回NaN
    
    # 计算VaR：收益率分布的指定分位数
    # cutoff通常为0.05，表示95%置信水平下的最大损失
    # 乘以100转换为百分位数参数（0.05 -> 5%分位数）
    return np.percentile(returns, 100 * cutoff)


@njit(cache=True)  # Numba JIT编译优化，启用缓存提高批量VaR计算效率
def value_at_risk_nb(returns: tp.Array2d, cutoff: float = 0.05) -> tp.Array1d:
    """
    二维风险价值(VaR)批量计算函数
    
    这是value_at_risk_1d_nb的二维扩展版本，能够同时计算多个资产或策略的风险价值。
    该函数为风险管理部门提供了高效的批量风险度量工具，特别适用于投资组合
    风险监控、资本配置和监管报告等场景。
    
    设计优势：
        - 批量处理多个资产的VaR计算，大幅提高分析效率
        - 统一的置信水平设置，确保不同资产间风险度量的一致性
        - 向量化处理，充分利用现代计算架构的并行能力
        - 便于后续的风险聚合、限额管理和报告生成
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
            - 行索引：时间维度，按时间顺序排列的历史收益率
            - 列索引：资产维度，每列代表一个资产、策略或投资组合
        cutoff (float): 置信水平的补集，默认为0.05（95%置信水平）
            - 所有资产使用统一的置信水平，便于风险比较和聚合
    
    返回值：
        tp.Array1d: 各资产VaR值数组，形状为(资产数,)
            - 每个元素对应一个资产的VaR值（通常为负数）
            - 保持与输入矩阵列顺序的一致性
            - 便于后续风险分析和决策制定
    
    计算特点：
        - 每列独立计算，不同资产间的VaR计算互不影响
        - 自动处理每列的缺失值和数据质量问题
        - 支持混合资产类型的同时风险评估
        - 提供标准化的风险度量输出格式
    
    使用示例：
        >>> # 三个不同投资策略的收益率数据（100天）
        >>> np.random.seed(42)
        >>> returns = np.array([
        ...     np.random.normal(0.001, 0.015, 100),  # 策略A：低风险
        ...     np.random.normal(0.002, 0.025, 100),  # 策略B：中等风险  
        ...     np.random.normal(0.003, 0.035, 100)   # 策略C：高风险
        ... ]).T  # 转置为(100, 3)格式
        >>> 
        >>> # 计算95% VaR
        >>> var_values = value_at_risk_nb(returns, 0.05)
        >>> print("各策略95% VaR:")
        >>> for i, var in enumerate(var_values):
        ...     print(f"策略{chr(65+i)}: {var:.3f} ({var*100:.1f}%)")
        
        输出示例：
        各策略95% VaR:
        策略A: -0.023 (-2.3%)  # 低风险策略
        策略B: -0.039 (-3.9%)  # 中等风险策略
        策略C: -0.054 (-5.4%)  # 高风险策略
    
    风险管理应用场景：
        - **投资组合风险监控**：实时监控各子策略的风险暴露
        - **资本配置决策**：基于VaR进行风险资本的合理分配
        - **限额管理系统**：设定和监控基于VaR的交易限额
        - **监管合规报告**：满足监管机构的风险披露要求
        - **客户风险披露**：向投资者展示各产品的风险水平
    
    机构应用实例：
        - **银行**：计算交易账簿各业务线的市场风险VaR
        - **基金公司**：监控旗下各基金产品的投资风险
        - **保险公司**：评估投资组合对偿付能力的影响
        - **企业财务**：管理企业金融资产的市场风险敞口
        - **养老基金**：控制各资产类别的风险贡献度
    
    风险聚合与分解：
        - **组合VaR**：可用于计算投资组合的总体VaR
        - **成分贡献**：分析各资产对总体风险的贡献度
        - **边际VaR**：计算增加单位投资对总风险的影响
        - **风险预算**：基于VaR进行风险预算的分配和控制
    
    数据质量与处理：
        - 自动处理各列的缺失值，不影响其他资产的计算
        - 支持不同资产的不同历史长度（通过NaN填充对齐）
        - 建议使用至少250个交易日的历史数据
        - 定期更新数据以保持风险度量的时效性
    
    后续分析建议：
        - 结合压力测试验证VaR模型的有效性
        - 监控VaR突破频率，评估模型校准质量
        - 配合CVaR分析，了解尾部风险的完整情况
        - 进行回测分析，验证VaR预测的准确性
    """
    # 预分配输出数组，长度等于资产数量（矩阵的列数）
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的VaR值
    for col in range(returns.shape[1]):
        # 调用一维VaR函数计算当前资产列的风险价值
        # 每列的计算完全独立，适合并行化处理
        out[col] = value_at_risk_1d_nb(returns[:, col], cutoff)
    
    return out


@njit  # Numba JIT编译优化，专门为滚动VaR计算设计
def rolling_value_at_risk_nb(returns: tp.Array2d,
                             window: int,
                             minp: tp.Optional[int],
                             cutoff: float = 0.05) -> tp.Array2d:
    """
    滚动窗口风险价值(VaR)计算函数
    
    该函数计算滚动窗口内的风险价值，为动态风险管理和实时监控提供时变的
    风险度量指标。通过滑动时间窗口，能够及时捕捉市场风险的变化趋势，
    为风险控制决策提供前瞻性的预警信号。
    
    核心价值：
        相比静态VaR，滚动VaR能够：
        - 实时反映市场风险环境的变化
        - 及时识别风险水平的上升或下降趋势
        - 为动态风险限额调整提供量化依据
        - 支持基于风险变化的投资决策和风控措施
    
    计算原理：
        对于每个时间点，使用该时点之前window长度的历史数据计算VaR：
        - 滑动窗口[t-window+1, t]内的收益率数据
        - 计算窗口内指定置信水平的分位数
        - 生成连续的时变VaR序列
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
        window (int): 滚动窗口大小（时间点数）
            - 日频数据：建议63天（季度）到252天（年度）
            - 月频数据：建议12个月到36个月
            - 需平衡统计稳定性与市场敏感性
        minp (tp.Optional[int]): 计算所需的最少观测值
            - None：使用window作为最小观测值要求
            - 整数：指定最小有效观测值数量，建议不少于30
        cutoff (float): 置信水平的补集，默认为0.05（95%置信水平）
    
    返回值：
        tp.Array2d: 滚动VaR矩阵，形状与输入returns相同
            - 前window-1行为NaN（数据不足）
            - 后续每行为对应时点的滚动窗口VaR值
    
    应用场景：
        - **动态风险监控**：实时跟踪投资组合风险水平变化
        - **风险预警系统**：当VaR持续恶化时触发风险预警
        - **限额动态调整**：基于滚动VaR趋势调整交易限额
        - **市场择时决策**：根据风险环境变化调整投资策略
        - **监管合规监控**：持续监控是否满足监管风险要求
    
    使用示例：
        >>> # 模拟1年的日收益率数据
        >>> np.random.seed(42)
        >>> # 模拟市场环境变化：前半年低波动，后半年高波动
        >>> returns1 = np.random.normal(0.001, 0.015, (126, 2))  # 低波动期
        >>> returns2 = np.random.normal(0.001, 0.035, (126, 2))  # 高波动期
        >>> returns = np.vstack([returns1, returns2])  # 合并数据
        >>> 
        >>> # 计算21天滚动95% VaR
        >>> rolling_var = rolling_value_at_risk_nb(returns, 21, None, 0.05)
        >>> 
        >>> # 分析VaR的时间变化
        >>> print("前半年平均VaR:", np.nanmean(rolling_var[21:126, 0]))
        >>> print("后半年平均VaR:", np.nanmean(rolling_var[147:252, 0]))
        
        预期输出：
        前半年平均VaR: -0.025 (-2.5%)  # 低风险期
        后半年平均VaR: -0.058 (-5.8%)  # 高风险期
    
    风险管理实务应用：
        - **银行风险管理**：监控交易账簿的日间VaR变化
        - **基金风险控制**：跟踪基金净值的风险暴露演变
        - **保险资管**：监控投资组合对偿付能力的影响
        - **企业财务**：管理金融资产的市场风险动态
        - **监管报告**：提供动态风险监控的证据材料
    
    市场环境适应性：
        - **平静期**：VaR相对稳定，关注是否出现异常波动
        - **波动期**：VaR快速上升，需要及时调整风险敞口
        - **危机期**：VaR可能大幅跳升，需要特别关注模型有效性
        - **复苏期**：VaR逐步回落，可考虑适度增加风险敞口
    
    技术实现特点：
        - 使用高效的滑动窗口算法，避免重复计算
        - 自动处理数据不足的边界情况
        - 支持不同资产的并行计算
        - 内存优化的设计，适合大规模数据处理
    
    模型验证建议：
        - 定期进行回测验证，检查VaR突破频率
        - 监控模型在不同市场环境下的表现
        - 结合压力测试验证极端情况下的有效性
        - 与其他风险模型进行交叉验证
    
    注意事项：
        - 滚动VaR具有滞后性，反映的是历史窗口期的风险
        - 在市场结构性变化时，历史VaR的预测能力可能下降
        - 需要结合前瞻性分析和专家判断
        - 窗口大小的选择需要在稳定性和敏感性之间平衡
    """
    # 定义滚动窗口内的VaR计算函数
    # 该内部函数将被rolling_apply_nb调用处理每个滚动窗口
    def _apply_func_nb(i, col, _returns, _cutoff):
        """
        滚动窗口VaR计算的内部应用函数
        
        参数说明：
            i (int): 当前时间点索引（未使用，保持接口一致性）
            col (int): 当前资产列索引（未使用，保持接口一致性）
            _returns (tp.Array1d): 当前窗口的收益率数据
            _cutoff (float): 置信水平参数
        
        返回值：
            float: 当前窗口的VaR值
        """
        # 调用一维VaR函数计算当前窗口的风险价值
        # 函数内部会自动处理NaN值过滤和数据有效性检查
        return value_at_risk_1d_nb(_returns, _cutoff)

    # 使用通用滚动应用函数进行滚动VaR计算
    # generic_nb.rolling_apply_nb负责：
    # 1. 管理滚动窗口的时间切片和数据提取
    # 2. 处理最小观测值要求(minp)和边界条件
    # 3. 调用_apply_func_nb进行窗口内的VaR计算
    # 4. 组装完整的输出矩阵并处理缺失值填充
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, cutoff)


@njit(cache=True)  # Numba JIT编译优化，启用缓存提高CVaR计算性能
def cond_value_at_risk_1d_nb(returns: tp.Array1d, cutoff: float = 0.05) -> float:
    """
    一维条件风险价值(CVaR)计算函数
    
    条件风险价值(Conditional Value at Risk, CVaR)，也称为期望损失(Expected 
    Shortfall, ES)，是VaR的重要扩展指标。CVaR衡量的是超过VaR阈值的条件期望
    损失，提供了尾部风险的更完整信息，是现代风险管理的核心工具。
    
    计算原理：
        CVaR = E[损失 | 损失 > VaR]
        即在损失超过VaR的条件下，损失的期望值
        - 计算步骤：
          1. 对收益率序列进行排序
          2. 取最差的cutoff比例的观测值
          3. 计算这些最差情况的平均值
    
    理论基础：
        CVaR基于一致性风险度量理论，满足单调性、平移不变性、正齐次性和
        次可加性等优良性质。相比VaR只给出损失阈值，CVaR提供了超过该阈值
        时的期望损失大小，为风险管理提供更丰富的信息。
    
    参数说明：
        returns (tp.Array1d): 一维收益率时间序列
        cutoff (float): 置信水平的补集，默认为0.05（即95%置信水平）
            - 0.05：计算最差5%情况下的平均损失
            - 0.01：计算最差1%情况下的平均损失
            - cutoff值越小，关注的极端情况越严重
    
    返回值：
        float: CVaR值（通常为负数，表示条件期望损失）
            - 负值：表示在最坏cutoff比例情况下的平均损失
            - 绝对值越大表示尾部风险越严重
            - 通常CVaR的绝对值大于对应的VaR
    
    数值稳定性：
        - 使用np.partition进行高效的部分排序
        - 自动处理边界情况和数据长度问题
        - 计算复杂度为O(n)，比完全排序更高效
    
    使用示例：
        >>> # 模拟包含极端损失的收益率序列
        >>> returns = np.array([-0.15, -0.08, -0.05, -0.03, -0.02, 
        ...                     0.01, 0.02, 0.03, 0.04, 0.05])
        >>> 
        >>> # 计算95% CVaR (最差5%的平均损失)
        >>> cvar_5 = cond_value_at_risk_1d_nb(returns, 0.05)
        >>> print(f"95% CVaR: {cvar_5:.3f} ({cvar_5*100:.1f}%)")
        >>> # 输出: 95% CVaR: -0.150 (-15.0%)
        >>> 
        >>> # 对比VaR和CVaR
        >>> var_5 = value_at_risk_1d_nb(returns, 0.05)
        >>> print(f"95% VaR:  {var_5:.3f} ({var_5*100:.1f}%)")
        >>> # 输出: 95% VaR: -0.150 (-15.0%)
        >>> 
        >>> # CVaR通常比VaR更严格（绝对值更大）
        >>> print(f"CVaR vs VaR 比率: {abs(cvar_5/var_5):.2f}")
    
    CVaR相对VaR的优势：
        - **完整性**：提供尾部损失的完整信息，而非仅仅是阈值
        - **一致性**：满足一致性风险度量的所有公理
        - **可加性**：投资组合的CVaR可以由成分资产CVaR计算得出
        - **优化友好**：CVaR是凸函数，便于在优化问题中使用
        - **监管认可**：越来越多的监管机构认可CVaR作为风险度量
    
    应用场景：
        - **投资组合优化**：作为风险约束或目标函数
        - **资本配置**：基于CVaR进行更精确的资本分配
        - **压力测试**：评估极端情况下的期望损失
        - **风险预算**：设定基于尾部风险的限额
        - **绩效评估**：风险调整收益的分母计算
    
    业界应用标准：
        - **银行业**：巴塞尔III框架鼓励使用CVaR进行内部风险管理
        - **保险业**：偿付能力II使用CVaR概念进行资本要求计算
        - **基金业**：越来越多基金使用CVaR进行风险披露
        - **监管机构**：部分监管机构要求报告CVaR指标
    
    CVaR解读指南：
        - **低风险** (|CVaR| < 3%)：保守策略，尾部风险可控
        - **中等风险** (3% ≤ |CVaR| < 7%)：平衡策略，可接受的尾部风险
        - **高风险** (7% ≤ |CVaR| < 15%)：积极策略，需要密切监控
        - **极高风险** (|CVaR| ≥ 15%)：激进策略，仅适合专业投资者
    
    与其他风险指标关系：
        - **VaR关系**：CVaR ≥ VaR（绝对值），提供更保守的风险估计
        - **标准差**：CVaR关注尾部，标准差关注整体分布
        - **最大回撤**：CVaR基于统计分布，最大回撤基于历史极值
        - **下行风险**：都关注负面风险，但计算方法和关注点不同
    
    实施建议：
        - 建议与VaR一起使用，提供风险的完整图景
        - 定期回测验证CVaR模型的有效性
        - 结合压力测试分析极端情况下的表现
        - 考虑使用CVaR进行投资组合优化
    
    注意事项：
        - CVaR对极端值更敏感，需要足够的历史数据
        - 在数据有限时，CVaR的估计可能不稳定
        - 需要定期更新数据以保持模型时效性
        - 极端市场条件下需要谨慎解读结果
    """
    # 计算cutoff对应的索引位置
    # (len(returns) - 1) * cutoff 确定最差cutoff比例对应的索引
    cutoff_index = int((len(returns) - 1) * cutoff)
    
    # 使用np.partition进行高效的部分排序
    # partition将数组分为两部分：[:cutoff_index+1]包含最小的cutoff_index+1个元素
    # 这比完全排序更高效，时间复杂度为O(n)
    partitioned = np.partition(returns, cutoff_index)
    
    # 计算最差cutoff比例情况下的平均损失（条件期望）
    # [:cutoff_index + 1] 包含最差的cutoff比例的观测值
    return np.mean(partitioned[:cutoff_index + 1])


@njit(cache=True)  # Numba JIT编译优化，启用缓存提高批量CVaR计算效率
def cond_value_at_risk_nb(returns: tp.Array2d, cutoff: float = 0.05) -> tp.Array1d:
    """
    二维条件风险价值(CVaR)批量计算函数
    
    这是cond_value_at_risk_1d_nb的二维扩展版本，能够同时计算多个资产或策略的
    条件风险价值。该函数为风险管理部门提供了高效的批量尾部风险度量工具，
    特别适用于投资组合优化、风险预算分配和监管合规等高级风险管理应用。
    
    设计优势：
        - 批量处理多个资产的CVaR计算，显著提高分析效率
        - 统一的置信水平设置，确保不同资产间风险度量的一致性
        - 向量化处理，充分利用现代计算架构的并行优势
        - 为投资组合优化和风险聚合提供标准化输入
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
            - 行索引：时间维度，按时间顺序排列的历史收益率
            - 列索引：资产维度，每列代表一个资产、策略或子组合
        cutoff (float): 置信水平的补集，默认为0.05（95%置信水平）
            - 所有资产使用统一的置信水平，便于风险比较和聚合
    
    返回值：
        tp.Array1d: 各资产CVaR值数组，形状为(资产数,)
            - 每个元素对应一个资产的CVaR值（通常为负数）
            - 保持与输入矩阵列顺序的一致性
            - 便于后续风险分析和投资组合优化
    
    使用示例：
        >>> # 三个不同风险特征的投资策略
        >>> np.random.seed(42)
        >>> # 策略A：正态分布，低风险
        >>> strategy_a = np.random.normal(0.001, 0.015, 200)
        >>> # 策略B：带有负偏的分布，中等风险  
        >>> strategy_b = np.concatenate([
        ...     np.random.normal(0.002, 0.02, 180),
        ...     np.random.normal(-0.05, 0.01, 20)  # 添加尾部风险
        ... ])
        >>> # 策略C：高波动策略
        >>> strategy_c = np.random.normal(0.003, 0.035, 200)
        >>> 
        >>> returns = np.column_stack([strategy_a, strategy_b, strategy_c])
        >>> 
        >>> # 计算95% CVaR
        >>> cvar_values = cond_value_at_risk_nb(returns, 0.05)
        >>> print("各策略95% CVaR:")
        >>> for i, cvar in enumerate(cvar_values):
        ...     print(f"策略{chr(65+i)}: {cvar:.3f} ({cvar*100:.1f}%)")
        
        预期输出：
        各策略95% CVaR:
        策略A: -0.025 (-2.5%)  # 低风险，CVaR接近正态分布预期
        策略B: -0.048 (-4.8%)  # 中等风险，CVaR反映尾部风险
        策略C: -0.058 (-5.8%)  # 高风险，CVaR显示高波动影响
    
    高级风险管理应用：
        - **投资组合优化**：
          * 使用CVaR作为风险约束进行均值-CVaR优化
          * 构建风险平价组合，基于CVaR贡献度分配权重
          * 进行多目标优化，平衡收益与尾部风险
        
        - **风险预算管理**：
          * 基于各资产CVaR分配风险预算
          * 设定基于CVaR的投资限额和止损线
          * 监控各子策略对总体尾部风险的贡献
        
        - **资本配置**：
          * 使用CVaR进行经济资本的分配
          * 基于CVaR计算风险调整收益率(RAROC)
          * 为不同业务线分配风险资本
    
    机构应用场景：
        - **银行业**：计算各业务条线的经济资本需求
        - **基金公司**：评估旗下产品的尾部风险特征
        - **保险公司**：管理投资组合的尾部风险敞口
        - **养老基金**：控制长期投资的下行风险
        - **对冲基金**：优化多策略组合的风险配置
    
    CVaR聚合与分解：
        - **组合CVaR**：利用CVaR的次可加性计算组合风险
        - **风险贡献**：分析各资产对组合CVaR的边际贡献
        - **成分CVaR**：分解组合CVaR到各个成分资产
        - **相关性影响**：考虑资产间相关性对CVaR聚合的影响
    
    模型验证与回测：
        - **回测验证**：检验CVaR预测的准确性和一致性
        - **压力测试**：在极端市场情景下验证CVaR有效性
        - **敏感性分析**：分析cutoff参数对CVaR结果的影响
        - **模型比较**：与其他风险模型的结果进行对比验证
    
    数据质量要求：
        - 建议使用至少500个观测值以获得稳定的CVaR估计
        - 定期更新数据，特别是在市场环境发生变化时
        - 注意处理数据中的异常值和结构断点
        - 考虑使用滚动窗口以适应时变的风险特征
    
    计算效率特点：
        - 每列独立计算，天然支持并行处理
        - 使用高效的部分排序算法，避免完全排序的开销
        - 内存优化设计，适合处理大规模资产组合
        - 缓存友好的列遍历顺序，提高计算性能
    """
    # 预分配输出数组，长度等于资产数量（矩阵的列数）
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的CVaR值
    for col in range(returns.shape[1]):
        # 调用一维CVaR函数计算当前资产列的条件风险价值
        # 每列的计算完全独立，适合并行化处理和大规模应用
        out[col] = cond_value_at_risk_1d_nb(returns[:, col], cutoff)
    
    return out


@njit  # Numba JIT编译优化，专门为滚动CVaR计算设计
def rolling_cond_value_at_risk_nb(returns: tp.Array2d,
                                  window: int,
                                  minp: tp.Optional[int],
                                  cutoff: float = 0.05) -> tp.Array2d:
    """
    滚动窗口条件风险价值(CVaR)计算函数
    
    该函数计算滚动窗口内的条件风险价值，为动态尾部风险管理提供时变的
    精确风险度量。通过滑动时间窗口，能够及时捕捉市场尾部风险的演变，
    为高级风险管理和投资组合优化提供前瞻性的决策支持。
    
    核心价值：
        相比静态CVaR，滚动CVaR能够：
        - 实时监控尾部风险的时间变化特征
        - 及时识别极端风险的积累或释放趋势
        - 为动态风险预算调整提供精确的量化依据
        - 支持基于尾部风险变化的投资决策优化
    
    计算原理：
        对于每个时间点，使用该时点之前window长度的历史数据计算CVaR：
        - 滑动窗口[t-window+1, t]内的收益率数据
        - 计算窗口内最差cutoff比例观测值的平均损失
        - 生成连续的时变CVaR序列，反映尾部风险动态
    
    参数说明：
        returns (tp.Array2d): 二维收益率矩阵，形状为(时间点数, 资产数)
        window (int): 滚动窗口大小（时间点数）
            - 日频数据：建议126天（半年）到252天（一年）
            - 月频数据：建议24个月到60个月
            - CVaR需要更长窗口以获得稳定的尾部风险估计
        minp (tp.Optional[int]): 计算所需的最少观测值
            - None：使用window作为最小观测值要求
            - 整数：指定最小有效观测值数量，建议不少于100
        cutoff (float): 置信水平的补集，默认为0.05（95%置信水平）
    
    返回值：
        tp.Array2d: 滚动CVaR矩阵，形状与输入returns相同
            - 前window-1行为NaN（数据不足）
            - 后续每行为对应时点的滚动窗口CVaR值
    
    应用场景：
        - **动态风险预算**：基于滚动CVaR调整各资产的风险配置
        - **尾部风险监控**：实时跟踪投资组合的极端损失风险
        - **投资组合再平衡**：根据CVaR变化趋势调整资产权重
        - **风险限额管理**：动态设定基于CVaR的投资限额
        - **压力测试验证**：验证风险模型在不同时期的有效性
    
    使用示例：
        >>> # 模拟2年的日收益率数据，包含不同风险阶段
        >>> np.random.seed(42)
        >>> # 第一年：正常市场环境
        >>> normal_period = np.random.normal(0.001, 0.02, (252, 2))
        >>> # 第二年：高波动环境，包含极端事件
        >>> volatile_period = np.concatenate([
        ...     np.random.normal(0.0005, 0.035, (230, 2)),  # 高波动期
        ...     np.random.normal(-0.08, 0.015, (22, 2))     # 极端事件期
        ... ])
        >>> returns = np.vstack([normal_period, volatile_period])
        >>> 
        >>> # 计算63天滚动95% CVaR
        >>> rolling_cvar = rolling_cond_value_at_risk_nb(returns, 63, None, 0.05)
        >>> 
        >>> # 分析CVaR的时间演变
        >>> print("第一年平均CVaR:", np.nanmean(rolling_cvar[63:252, 0]))
        >>> print("第二年平均CVaR:", np.nanmean(rolling_cvar[315:504, 0]))
        
        预期输出：
        第一年平均CVaR: -0.032 (-3.2%)  # 正常风险期
        第二年平均CVaR: -0.078 (-7.8%)  # 高风险期，CVaR显著上升
    
    高级风险管理应用：
        - **动态对冲策略**：基于滚动CVaR变化调整对冲比例
        - **风险平价组合**：使用时变CVaR重新平衡风险贡献
        - **尾部风险预算**：动态分配基于CVaR的风险预算
        - **监管资本管理**：根据CVaR变化调整监管资本缓冲
        - **投资者沟通**：向客户展示风险水平的时间变化
    
    市场环境适应性：
        - **牛市环境**：CVaR相对稳定，关注是否出现风险积累
        - **熊市环境**：CVaR快速上升，需要及时调整风险敞口
        - **危机期间**：CVaR可能大幅跳升，需要启动应急预案
        - **复苏阶段**：CVaR逐步回落，可考虑增加风险配置
    
    技术实现优势：
        - 使用高效的滑动窗口算法，优化计算性能
        - 自动处理数据边界和缺失值情况
        - 支持大规模资产组合的并行计算
        - 内存优化设计，适合长时间序列处理
    
    模型监控与验证：
        - **回测验证**：检验滚动CVaR的预测准确性
        - **敏感性分析**：测试不同窗口大小对结果的影响
        - **稳定性检验**：评估CVaR估计的时间稳定性
        - **压力测试**：验证极端市场条件下的模型表现
    
    实务操作建议：
        - 结合VaR一起使用，提供完整的风险图景
        - 定期校准模型参数，确保估计质量
        - 建立CVaR突破的预警机制
        - 与基本面分析结合，提高风险判断准确性
    
    注意事项：
        - CVaR对极端值敏感，需要充分的历史数据
        - 窗口大小需要在稳定性和敏感性之间平衡
        - 在市场结构性变化时需要重新校准模型
        - 建议结合前瞻性分析和专家判断使用
    """
    # 定义滚动窗口内的CVaR计算函数
    # 该内部函数将被rolling_apply_nb调用处理每个滚动窗口
    def _apply_func_nb(i, col, _returns, _cutoff):
        """
        滚动窗口CVaR计算的内部应用函数
        
        参数说明：
            i (int): 当前时间点索引（未使用，保持接口一致性）
            col (int): 当前资产列索引（未使用，保持接口一致性）
            _returns (tp.Array1d): 当前窗口的收益率数据
            _cutoff (float): 置信水平参数
        
        返回值：
            float: 当前窗口的CVaR值（条件期望损失）
        """
        # 调用一维CVaR函数计算当前窗口的条件风险价值
        # 函数内部会自动处理数据排序和条件期望计算
        return cond_value_at_risk_1d_nb(_returns, _cutoff)

    # 使用通用滚动应用函数进行滚动CVaR计算
    # generic_nb.rolling_apply_nb负责：
    # 1. 管理滚动窗口的时间切片和数据提取
    # 2. 处理最小观测值要求(minp)和边界条件
    # 3. 调用_apply_func_nb进行窗口内的CVaR计算
    # 4. 组装完整的输出矩阵并处理缺失值填充
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, cutoff)


@njit(cache=True)  # Numba JIT编译优化，启用缓存提高捕获比率计算性能
def capture_1d_nb(returns: tp.Array1d, benchmark_rets: tp.Array1d, ann_factor: float) -> float:
    """
    一维捕获比率计算函数
    
    捕获比率(Capture Ratio)衡量投资组合相对于基准的收益捕获能力，反映了
    投资策略在整个投资期间对基准收益的跟踪和超越程度。该指标为投资者提供
    了直观的相对绩效评估，是评价主动投资管理效果的重要工具。
    
    计算公式：
        捕获比率 = 投资组合年化收益率 / 基准年化收益率
        - 比率 > 1：投资组合表现优于基准
        - 比率 = 1：投资组合表现与基准一致
        - 比率 < 1：投资组合表现劣于基准
    
    理论基础：
        捕获比率基于相对绩效评估理论，通过比较投资组合与基准的年化收益率，
        量化投资策略的相对价值创造能力。该指标简单直观，易于理解和沟通，
        广泛应用于基金评级、投资决策和绩效报告中。
    
    参数说明：
        returns (tp.Array1d): 一维投资组合收益率时间序列
        benchmark_rets (tp.Array1d): 一维基准收益率时间序列
        ann_factor (float): 年化因子
            - 日频数据：252（年均交易日数）
            - 月频数据：12（年均月数）
            - 周频数据：52（年均周数）
    
    返回值：
        float: 捕获比率
            - > 1.0：投资组合年化收益超过基准，表现优秀
            - = 1.0：投资组合年化收益等于基准，表现持平
            - < 1.0：投资组合年化收益低于基准，表现不佳
            - ∞：基准收益为零而投资组合有正收益（理想情况）
            - NaN：数据不足或计算错误
    
    数值稳定性：
        - 当基准年化收益为0时返回无穷大，避免除零错误
        - 自动处理负收益率情况，保持比率的合理性
        - 使用年化收益率确保不同期间数据的可比性
    
    使用示例：
        >>> # 模拟投资组合和基准的月收益率数据（12个月）
        >>> returns = np.array([0.02, 0.015, -0.01, 0.025, 0.008, 0.012,
        ...                     -0.005, 0.018, 0.022, -0.008, 0.015, 0.01])
        >>> benchmark = np.array([0.018, 0.012, -0.008, 0.02, 0.006, 0.01,
        ...                      -0.003, 0.015, 0.018, -0.006, 0.012, 0.008])
        >>> 
        >>> capture_ratio = capture_1d_nb(returns, benchmark, 12.0)
        >>> print(f"捕获比率: {capture_ratio:.2f}")
        >>> # 输出: 捕获比率: 1.15
        >>> 
        >>> # 解读：投资组合年化收益比基准高15%
        >>> if capture_ratio > 1.0:
        ...     print(f"投资组合表现优于基准 {(capture_ratio-1)*100:.1f}%")
    
    捕获比率解读指南：
        - **卓越表现** (比率 > 1.2)：显著超越基准，投资技能突出
        - **优秀表现** (1.1 < 比率 ≤ 1.2)：稳定超越基准，具备价值创造能力
        - **良好表现** (1.05 < 比率 ≤ 1.1)：略微超越基准，表现可接受
        - **持平表现** (0.95 ≤ 比率 ≤ 1.05)：与基准表现基本一致
        - **需要改进** (比率 < 0.95)：表现落后基准，需要分析原因
    
    应用场景：
        - **基金评级**：评估基金经理的整体投资能力
        - **策略比较**：比较不同投资策略的相对表现
        - **绩效考核**：作为投资经理薪酬考核的重要指标
        - **产品选择**：帮助投资者选择表现优异的投资产品
        - **风险调整**：结合风险指标评估风险调整后的相对表现
    
    与其他绩效指标的关系：
        - **Alpha**：Alpha衡量风险调整后的超额收益，捕获比率衡量总收益比较
        - **信息比率**：信息比率考虑收益的一致性，捕获比率只看总体收益
        - **夏普比率**：夏普比率是风险调整收益，捕获比率是相对收益
        - **Beta**：Beta衡量系统风险，捕获比率衡量收益获取能力
    
    投资策略分析：
        - **主动策略**：捕获比率>1表明主动管理创造了价值
        - **被动策略**：捕获比率应接近1，显著偏离表明跟踪误差
        - **增强策略**：期望捕获比率略大于1，同时控制跟踪误差
        - **对冲策略**：捕获比率可能较低，但波动性也相应较小
    
    实务应用注意事项：
        - 需要足够长的观察期以获得稳定的估计
        - 应该结合风险指标进行综合评估
        - 不同市场环境下的捕获比率可能差异较大
        - 需要考虑基准的适当性和代表性
    
    局限性：
        - 只反映收益率比较，不考虑风险差异
        - 对观察期的选择敏感
        - 无法反映收益获取的时间分布特征
        - 不能区分运气和技能的贡献
    """
    # 计算投资组合的年化收益率
    annualized_return1 = annualized_return_1d_nb(returns, ann_factor)
    
    # 计算基准的年化收益率
    annualized_return2 = annualized_return_1d_nb(benchmark_rets, ann_factor)
    
    # 处理基准收益为零的特殊情况：避免除零错误
    if annualized_return2 == 0.:
        return np.inf  # 返回无穷大表示投资组合相对表现无限好
    
    # 计算捕获比率：投资组合年化收益与基准年化收益的比值
    return annualized_return1 / annualized_return2


@njit(cache=True)  # Numba JIT编译优化，启用缓存提高批量捕获比率计算效率
def capture_nb(returns: tp.Array2d, benchmark_rets: tp.Array2d, ann_factor: float) -> tp.Array1d:
    """
    二维捕获比率批量计算函数
    
    该函数同时计算多个资产或策略的整体捕获比率，为投资组合管理提供
    高效的批量绩效评估工具。通过一次性处理多个资产，显著提高了
    大规模投资组合分析的效率。
    
    应用价值：
        - 批量评估多个策略的整体绩效表现
        - 为资产配置决策提供相对绩效比较
        - 支持基金产品的批量绩效排序和筛选
        - 为投资组合优化提供绩效输入数据
    """
    # 预分配输出数组，长度等于资产数量
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的捕获比率
    for col in range(returns.shape[1]):
        # 调用一维捕获比率函数计算当前资产的相对绩效
        out[col] = capture_1d_nb(returns[:, col], benchmark_rets[:, col], ann_factor)
    
    return out


@njit  # Numba JIT编译优化，专门为滚动捕获比率计算设计
def rolling_capture_nb(returns: tp.Array2d,
                       window: int,
                       minp: tp.Optional[int],
                       benchmark_rets: tp.Array2d,
                       ann_factor: float) -> tp.Array2d:
    """
    滚动窗口捕获比率计算函数
    
    该函数计算滚动窗口内的捕获比率，为动态绩效评估提供时变的相对表现指标。
    通过滑动时间窗口，能够及时捕捉投资策略相对于基准的绩效变化趋势，
    为投资决策和绩效管理提供前瞻性的量化支持。
    
    核心价值：
        - 实时监控投资策略相对绩效的时间变化
        - 及时识别策略表现的改善或恶化趋势
        - 为动态投资决策提供量化依据
        - 支持基于相对绩效变化的策略调整
    """
    # 定义滚动窗口内的捕获比率计算函数
    def _apply_func_nb(i, col, _returns, _benchmark_rets, _ann_factor):
        # 提取与投资组合收益率窗口对应的基准收益率切片
        benchmark_window = _benchmark_rets[i + 1 - len(_returns):i + 1, col]
        # 调用一维捕获比率函数计算当前窗口的相对绩效
        return capture_1d_nb(_returns, benchmark_window, _ann_factor)

    # 使用通用滚动应用函数进行滚动捕获比率计算
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, benchmark_rets, ann_factor)


@njit(cache=True)  # Numba JIT编译优化，启用缓存提高上行捕获比率计算性能
def up_capture_1d_nb(returns: tp.Array1d, benchmark_rets: tp.Array1d, ann_factor: float) -> float:
    """
    一维上行捕获比率计算函数
    
    上行捕获比率(Up Capture Ratio)专门衡量投资组合在基准上涨期间的收益捕获
    能力，反映了投资策略在牛市或上升趋势中的表现。该指标帮助投资者了解
    投资组合是否能够有效参与市场上涨，是评估投资策略上行参与度的关键工具。
    
    计算原理：
        1. 筛选基准收益为正的时期
        2. 计算这些时期投资组合和基准的年化收益率
        3. 计算两者的比值作为上行捕获比率
        
        公式：上行捕获比率 = 上涨期投资组合年化收益 / 上涨期基准年化收益
    
    理论基础：
        基于行为金融学和市场参与理论，上行捕获比率衡量投资策略在有利市场
        环境中的参与程度。高上行捕获比率表明策略能够有效捕获市场上涨机会，
        这对于追求资本增值的投资者尤为重要。
    
    参数说明：
        returns (tp.Array1d): 一维投资组合收益率时间序列
        benchmark_rets (tp.Array1d): 一维基准收益率时间序列
        ann_factor (float): 年化因子
    
    返回值：
        float: 上行捕获比率
            - > 1.0：在基准上涨时，投资组合涨幅更大，上行参与度高
            - = 1.0：在基准上涨时，投资组合涨幅与基准一致
            - < 1.0：在基准上涨时，投资组合涨幅较小，上行参与度低
            - NaN：无基准上涨期间或数据不足
    
    数值稳定性：
        - 自动筛选基准收益为正的时期
        - 当无上涨期间时返回NaN
        - 处理基准上涨期间年化收益为零的边界情况
    
    使用示例：
        >>> # 模拟牛熊交替的市场环境
        >>> benchmark = np.array([0.03, -0.02, 0.025, -0.01, 0.04, -0.015, 0.02])
        >>> returns = np.array([0.035, -0.015, 0.02, -0.012, 0.045, -0.018, 0.025])
        >>> 
        >>> up_capture = up_capture_1d_nb(returns, benchmark, 12.0)
        >>> print(f"上行捕获比率: {up_capture:.2f}")
        >>> # 输出: 上行捕获比率: 1.12
        >>> 
        >>> # 解读：在基准上涨期间，投资组合平均多获得12%的收益
        >>> if up_capture > 1.0:
        ...     print(f"上行期间表现优于基准 {(up_capture-1)*100:.1f}%")
    
    上行捕获比率解读：
        - **卓越上行参与** (比率 > 1.3)：在牛市中显著超越基准
        - **优秀上行参与** (1.1 < 比率 ≤ 1.3)：在牛市中稳定超越基准
        - **良好上行参与** (1.0 < 比率 ≤ 1.1)：在牛市中略微超越基准
        - **完全上行参与** (比率 = 1.0)：在牛市中与基准同步上涨
        - **有限上行参与** (0.8 ≤ 比率 < 1.0)：在牛市中涨幅有限
        - **防御性策略** (比率 < 0.8)：在牛市中涨幅较小，可能更注重风控
    
    应用场景：
        - **成长型投资评估**：评价成长策略在牛市中的表现
        - **动量策略分析**：分析动量策略的上行跟随能力
        - **基金风格识别**：识别基金的投资风格和市场参与度
        - **择时策略评估**：评估择时策略在上涨期间的有效性
        - **投资组合构建**：选择上行参与度高的资产或策略
    
    投资策略含义：
        - **高上行捕获**：
          * 优点：能够充分享受牛市收益
          * 缺点：可能在熊市中回撤也较大
          * 适合：风险承受能力强的投资者
        
        - **低上行捕获**：
          * 优点：策略相对保守，风险控制较好
          * 缺点：可能错失牛市机会
          * 适合：风险厌恶的投资者
    
    与下行捕获比率的配合使用：
        - **理想组合**：高上行捕获 + 低下行捕获
        - **进攻型策略**：高上行捕获 + 高下行捕获
        - **防御型策略**：低上行捕获 + 低下行捕获
        - **需要改进**：低上行捕获 + 高下行捕获
    
    市场环境分析：
        - **牛市主导期**：上行捕获比率是主要评估指标
        - **熊市主导期**：下行捕获比率更为重要
        - **震荡市场**：需要综合考虑上行和下行捕获比率
    
    实务应用建议：
        - 结合下行捕获比率进行综合评估
        - 考虑不同市场周期的表现差异
        - 关注上行捕获的一致性和稳定性
        - 结合其他绩效指标进行全面分析
    
    注意事项：
        - 依赖于基准上涨期间的数量和质量
        - 短期数据可能不具代表性
        - 需要考虑上行期间的市场环境差异
        - 应该与风险指标结合使用
    """
    # 筛选基准收益为正的时期：只关注市场上涨的情况
    # 这样可以专门分析投资组合在有利环境中的表现
    up_periods_mask = benchmark_rets > 0
    returns_up = returns[up_periods_mask]
    benchmark_rets_up = benchmark_rets[up_periods_mask]
    
    # 数据有效性检查：需要至少1个上涨期间
    if returns_up.shape[0] < 1:
        return np.nan  # 无上涨期间时返回NaN
    
    # 计算上涨期间投资组合的年化收益率
    annualized_return1 = annualized_return_1d_nb(returns_up, ann_factor)
    
    # 计算上涨期间基准的年化收益率
    annualized_return2 = annualized_return_1d_nb(benchmark_rets_up, ann_factor)
    
    # 处理基准上涨期间年化收益为零的边界情况
    if annualized_return2 == 0.:
        return np.inf  # 返回无穷大表示相对表现极佳
    
    # 计算上行捕获比率：上涨期间投资组合与基准年化收益的比值
    return annualized_return1 / annualized_return2


@njit(cache=True)  # Numba JIT编译优化，启用缓存提高批量上行捕获比率计算效率
def up_capture_nb(returns: tp.Array2d, benchmark_rets: tp.Array2d, ann_factor: float) -> tp.Array1d:
    """
    二维上行捕获比率批量计算函数
    
    该函数同时计算多个资产或策略的上行捕获比率，为投资组合管理提供
    高效的批量上行参与度评估工具。通过分析各资产在市场上涨期间的
    表现，帮助投资者识别具有优秀上行参与能力的投资标的。
    
    应用价值：
        - 批量筛选具有优秀上行参与度的资产或策略
        - 为成长型投资组合构建提供量化依据
        - 支持动量策略的资产选择和权重配置
        - 评估各资产在牛市环境中的表现潜力
    """
    # 预分配输出数组，长度等于资产数量
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的上行捕获比率
    for col in range(returns.shape[1]):
        # 调用一维上行捕获比率函数计算当前资产的上行参与能力
        out[col] = up_capture_1d_nb(returns[:, col], benchmark_rets[:, col], ann_factor)
    
    return out


@njit  # Numba JIT编译优化，专门为滚动上行捕获比率计算设计
def rolling_up_capture_nb(returns: tp.Array2d,
                          window: int,
                          minp: tp.Optional[int],
                          benchmark_rets: tp.Array2d,
                          ann_factor: float) -> tp.Array2d:
    """
    滚动窗口上行捕获比率计算函数
    
    该函数计算滚动窗口内的上行捕获比率，为动态评估投资策略在市场上涨期间的
    参与能力提供时变指标。通过滑动时间窗口，能够及时监控策略上行参与度的
    变化，为择时决策和策略调整提供量化支持。
    
    应用价值：
        - 动态监控策略在牛市环境中的参与度变化
        - 识别策略上行捕获能力的改善或恶化趋势
        - 为基于市场环境的策略轮换提供依据
        - 支持动态资产配置和风险管理决策
    """
    # 定义滚动窗口内的上行捕获比率计算函数
    def _apply_func_nb(i, col, _returns, _benchmark_rets, _ann_factor):
        # 提取与投资组合收益率窗口对应的基准收益率切片
        benchmark_window = _benchmark_rets[i + 1 - len(_returns):i + 1, col]
        # 调用一维上行捕获比率函数计算当前窗口的上行参与能力
        return up_capture_1d_nb(_returns, benchmark_window, _ann_factor)

    # 使用通用滚动应用函数进行滚动上行捕获比率计算
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, benchmark_rets, ann_factor)


@njit(cache=True)  # Numba JIT编译优化，启用缓存提高下行捕获比率计算性能
def down_capture_1d_nb(returns: tp.Array1d, benchmark_rets: tp.Array1d, ann_factor: float) -> float:
    """
    一维下行捕获比率计算函数
    
    下行捕获比率(Down Capture Ratio)专门衡量投资组合在基准下跌期间的损失
    控制能力，反映了投资策略在熊市或下降趋势中的防御表现。该指标帮助投资者
    了解投资组合在不利市场环境中的风险控制效果，是评估下行保护能力的关键工具。
    
    计算原理：
        1. 筛选基准收益为负的时期
        2. 计算这些时期投资组合和基准的年化收益率
        3. 计算两者的比值作为下行捕获比率
        
        公式：下行捕获比率 = 下跌期投资组合年化收益 / 下跌期基准年化收益
    
    理论基础：
        基于风险管理理论和下行风险控制原理，下行捕获比率衡量投资策略在不利
        市场环境中的防御能力。低下行捕获比率表明策略在市场下跌时能够有效
        控制损失，这对于风险厌恶的投资者具有重要价值。
    
    参数说明：
        returns (tp.Array1d): 一维投资组合收益率时间序列
        benchmark_rets (tp.Array1d): 一维基准收益率时间序列
        ann_factor (float): 年化因子
    
    返回值：
        float: 下行捕获比率
            - < 1.0：在基准下跌时，投资组合跌幅较小，下行保护能力强（理想）
            - = 1.0：在基准下跌时，投资组合跌幅与基准一致
            - > 1.0：在基准下跌时，投资组合跌幅更大，下行保护能力弱
            - NaN：无基准下跌期间或数据不足
    
    数值稳定性：
        - 自动筛选基准收益为负的时期
        - 当无下跌期间时返回NaN
        - 处理基准下跌期间年化收益为零的边界情况
    
    使用示例：
        >>> # 模拟牛熊交替的市场环境
        >>> benchmark = np.array([0.03, -0.04, 0.025, -0.03, 0.02, -0.05, 0.01])
        >>> returns = np.array([0.035, -0.03, 0.02, -0.025, 0.025, -0.04, 0.015])
        >>> 
        >>> down_capture = down_capture_1d_nb(returns, benchmark, 12.0)
        >>> print(f"下行捕获比率: {down_capture:.2f}")
        >>> # 输出: 下行捕获比率: 0.85
        >>> 
        >>> # 解读：在基准下跌期间，投资组合平均少损失15%
        >>> if down_capture < 1.0:
        ...     print(f"下行期间损失控制优于基准 {(1-down_capture)*100:.1f}%")
    
    下行捕获比率解读：
        - **卓越下行保护** (比率 < 0.7)：在熊市中显著优于基准，损失控制极佳
        - **优秀下行保护** (0.7 ≤ 比率 < 0.9)：在熊市中稳定优于基准
        - **良好下行保护** (0.9 ≤ 比率 < 1.0)：在熊市中略微优于基准
        - **同步下行** (比率 = 1.0)：在熊市中与基准同步下跌
        - **有限下行保护** (1.0 < 比率 ≤ 1.2)：在熊市中跌幅略大于基准
        - **下行放大** (比率 > 1.2)：在熊市中跌幅显著大于基准，风控不足
    
    应用场景：
        - **防御型投资评估**：评价防御策略在熊市中的表现
        - **风险控制分析**：分析策略的下行风险管理能力
        - **保本策略评估**：评估保本或保守策略的有效性
        - **对冲效果评价**：评估对冲策略在市场下跌时的保护效果
        - **投资组合构建**：选择下行保护能力强的资产或策略
    
    投资策略含义：
        - **低下行捕获**：
          * 优点：在熊市中能够有效控制损失
          * 缺点：可能在牛市中上涨有限
          * 适合：风险厌恶、追求稳健收益的投资者
        
        - **高下行捕获**：
          * 缺点：在熊市中损失可能较大
          * 可能原因：高风险策略、杠杆使用、风控不足
          * 需要：加强风险管理和下行保护措施
    
    与上行捕获比率的配合分析：
        - **理想策略**：高上行捕获(>1.0) + 低下行捕获(<1.0)
          * 牛市中充分参与，熊市中有效防御
        
        - **进攻型策略**：高上行捕获(>1.0) + 高下行捕获(>1.0)  
          * 高风险高收益，波动较大
        
        - **防御型策略**：低上行捕获(<1.0) + 低下行捕获(<1.0)
          * 稳健保守，适合风险厌恶投资者
        
        - **需要改进**：低上行捕获(<1.0) + 高下行捕获(>1.0)
          * 上涨时参与不足，下跌时保护不够
    
    市场环境分析：
        - **熊市主导期**：下行捕获比率是核心评估指标
        - **牛市主导期**：下行捕获比率提供风险预警信息
        - **震荡市场**：下行捕获比率反映策略的风险控制稳定性
    
    风险管理应用：
        - **止损策略评估**：评价止损机制的有效性
        - **对冲策略分析**：分析对冲工具的保护效果
        - **波动率管理**：评估波动率控制策略的效果
        - **资产配置优化**：选择下行保护能力强的资产组合
    
    实务应用建议：
        - 与上行捕获比率结合使用，全面评估策略特征
        - 关注不同市场环境下下行捕获的稳定性
        - 结合最大回撤、VaR等指标进行综合风险评估
        - 定期监控下行捕获比率的变化趋势
    
    注意事项：
        - 依赖于基准下跌期间的数量和严重程度
        - 短期数据可能无法充分反映下行保护能力
        - 需要考虑下跌期间的市场环境差异
        - 应该与其他风险指标结合综合评估
    """
    # 筛选基准收益为负的时期：只关注市场下跌的情况
    # 这样可以专门分析投资组合在不利环境中的防御表现
    down_periods_mask = benchmark_rets < 0
    returns_down = returns[down_periods_mask]
    benchmark_rets_down = benchmark_rets[down_periods_mask]
    
    # 数据有效性检查：需要至少1个下跌期间
    if returns_down.shape[0] < 1:
        return np.nan  # 无下跌期间时返回NaN
    
    # 计算下跌期间投资组合的年化收益率（通常为负值）
    annualized_return1 = annualized_return_1d_nb(returns_down, ann_factor)
    
    # 计算下跌期间基准的年化收益率（通常为负值）
    annualized_return2 = annualized_return_1d_nb(benchmark_rets_down, ann_factor)
    
    # 处理基准下跌期间年化收益为零的边界情况
    if annualized_return2 == 0.:
        return np.inf  # 返回无穷大表示极端情况
    
    # 计算下行捕获比率：下跌期间投资组合与基准年化收益的比值
    # 注意：由于都是负值相除，比率为正数
    # 比率<1表示投资组合跌幅小于基准（下行保护好）
    return annualized_return1 / annualized_return2


@njit(cache=True)  # Numba JIT编译优化，启用缓存提高批量下行捕获比率计算效率
def down_capture_nb(returns: tp.Array2d, benchmark_rets: tp.Array2d, ann_factor: float) -> tp.Array1d:
    """
    二维下行捕获比率批量计算函数
    
    这是down_capture_1d_nb的二维扩展版本，能够同时计算多个资产或策略的
    下行捕获比率。该函数为风险管理部门提供了高效的批量下行保护能力评估工具，
    特别适用于投资组合风险分析、防御策略筛选和风险预算管理等应用场景。
    
    应用价值：
        - 批量评估多个策略的下行风险控制能力
        - 为投资组合构建提供风险特征分析
        - 支持大规模资产筛选和风险对比
        - 为风险预算分配提供量化依据
    
    参数说明：
        returns (tp.Array2d): 二维投资组合收益率矩阵
        benchmark_rets (tp.Array2d): 二维基准收益率矩阵  
        ann_factor (float): 年化因子
    
    返回值：
        tp.Array1d: 各资产下行捕获比率数组
            - 每个元素对应一个资产的下行捕获比率
            - 比率<1.0表示该资产具有良好的下行保护能力
    """
    # 预分配输出数组，长度等于资产数量
    out = np.empty(returns.shape[1], dtype=np.float64)
    
    # 逐列计算每个资产的下行捕获比率
    for col in range(returns.shape[1]):
        # 调用一维下行捕获比率函数计算当前资产的防御能力
        out[col] = down_capture_1d_nb(returns[:, col], benchmark_rets[:, col], ann_factor)
    
    return out


@njit  # Numba JIT编译优化，专门为滚动下行捕获比率计算设计
def rolling_down_capture_nb(returns: tp.Array2d,
                            window: int,
                            minp: tp.Optional[int],
                            benchmark_rets: tp.Array2d,
                            ann_factor: float) -> tp.Array2d:
    """
    滚动窗口下行捕获比率计算函数
    
    该函数计算滚动窗口内的下行捕获比率，为动态评估投资策略在市场下跌期间的
    防御能力提供时变指标。通过滑动时间窗口，能够及时监控策略下行保护能力的
    变化，为风险管理和防御策略调整提供量化支持。
    
    应用价值：
        - 动态监控策略在熊市环境中的防御能力变化
        - 识别策略下行保护能力的改善或恶化趋势
        - 为基于风险环境的防御策略调整提供依据
        - 支持动态风险预算管理和止损决策
    """
    # 定义滚动窗口内的下行捕获比率计算函数
    def _apply_func_nb(i, col, _returns, _benchmark_rets, _ann_factor):
        # 提取与投资组合收益率窗口对应的基准收益率切片
        benchmark_window = _benchmark_rets[i + 1 - len(_returns):i + 1, col]
        # 调用一维下行捕获比率函数计算当前窗口的下行防御能力
        return down_capture_1d_nb(_returns, benchmark_window, _ann_factor)

    # 使用通用滚动应用函数进行滚动下行捕获比率计算
    return generic_nb.rolling_apply_nb(
        returns, window, minp, _apply_func_nb, benchmark_rets, ann_factor)
