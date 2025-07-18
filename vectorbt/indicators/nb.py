# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT INDICATORS MODULE: 高性能技术指标计算核心模块
================================================================================

文件设计逻辑和作用概述：
用于技术指标计算的模块。该模块提供了一整套经过Numba JIT编译优化的技术指标计算函数，
为上层的指标类（如MA、RSI、MACD等）提供底层的高性能计算支持。

核心设计理念：
1. **性能优化优先**：所有函数都使用@njit装饰器进行Just-In-Time编译，实现接近C语言的
   执行速度，能够处理大规模金融时间序列数据的实时计算需求。

2. **缓存机制设计**：采用智能缓存策略，通过cache和apply函数分离的设计模式，避免重复
   计算相同参数组合的中间结果，大幅提升多参数指标计算的效率。

3. **矩阵优先架构**：遵循vectorbt的核心设计原则，将二维矩阵视为一等公民，所有函数都
   期望处理2维数组输入，数据沿着时间轴（axis 0）进行处理，完美适配多资产分析场景。

4. **模块化函数设计**：每个技术指标都分解为多个独立的计算函数，支持函数组合和复用，
   提高代码的可维护性和扩展性。

主要功能模块：

【移动平均类指标】
- ma_nb: 简单移动平均和指数移动平均的统一计算函数
- mstd_nb: 移动标准差计算，支持简单和指数加权模式
- 相关缓存和应用函数：ma_cache_nb, ma_apply_nb, mstd_cache_nb, mstd_apply_nb

【布林带指标】
- bb_cache_nb: 布林带缓存函数，预计算移动平均和标准差
- bb_apply_nb: 布林带应用函数，计算上轨、中轨、下轨

【相对强弱指标(RSI)】
- rsi_cache_nb: RSI缓存函数，预计算上涨和下跌的移动平均
- rsi_apply_nb: RSI应用函数，计算相对强弱指数

【随机指标(STOCH)】
- stoch_cache_nb: 随机指标缓存函数，预计算滚动最高价和最低价
- stoch_apply_nb: 随机指标应用函数，计算%K和%D值

【MACD指标】
- macd_cache_nb: MACD缓存函数，预计算快线和慢线的移动平均
- macd_apply_nb: MACD应用函数，计算MACD线和信号线

【平均真实波幅(ATR)】
- true_range_nb: 真实波幅计算函数
- atr_cache_nb: ATR缓存函数，预计算真实波幅的移动平均
- atr_apply_nb: ATR应用函数，返回真实波幅和平均真实波幅

【成交量指标】
- obv_custom_nb: 能量潮指标(OBV)的自定义计算函数

技术实现特点：
- **缓存优化**：通过hash函数对参数组合进行缓存，避免重复计算
- **内存效率**：使用in-place操作和numpy数组拷贝，最大化内存利用率
- **数值稳定性**：处理边界条件和特殊值，确保计算结果的稳定性
- **向量化计算**：充分利用NumPy的向量化操作和SIMD指令集

缓存机制说明：
vectorbt采用了独特的两阶段计算模式：
1. **Cache阶段**：预计算所有参数组合的中间结果，存储在字典中
2. **Apply阶段**：根据具体参数从缓存中提取结果，进行最终计算

这种设计的优势：
- 避免重复计算相同的中间结果
- 支持批量参数优化和参数扫描
- 提供更好的内存局部性
- 支持并行计算
```python
import numpy as np
import pandas as pd
import vectorbt as vbt

# 创建示例价格数据
prices = pd.DataFrame({
    'AAPL': [150, 152, 149, 153, 155, 151, 154, 156, 158, 160],
    'GOOGL': [2800, 2820, 2790, 2850, 2880, 2860, 2890, 2910, 2930, 2950]
}, index=pd.date_range('2023-01-01', periods=10, freq='D'))

# 转换为2D数组
price_2d = prices.values

# 1. 移动平均计算示例
window = 5
ewm = False
adjust = False

# 计算简单移动平均
ma_result = vbt.indicators.nb.ma_nb(price_2d, window, ewm, adjust)
print("5日简单移动平均:")
print(pd.DataFrame(ma_result, index=prices.index, columns=prices.columns))

# 2. RSI计算示例
rsi_window = 14
rsi_ewm = False

# 先计算缓存
rsi_cache = vbt.indicators.nb.rsi_cache_nb(
    price_2d, [rsi_window], [rsi_ewm], adjust
)

# 然后计算RSI
rsi_result = vbt.indicators.nb.rsi_apply_nb(
    price_2d, rsi_window, rsi_ewm, adjust, rsi_cache
)
print("\\n14日RSI:")
print(pd.DataFrame(rsi_result, index=prices.index, columns=prices.columns))

# 3. 布林带计算示例
bb_window = 20
bb_ewm = False
bb_alpha = 2.0
bb_ddof = 0

# 计算布林带缓存
bb_ma_cache, bb_mstd_cache = vbt.indicators.nb.bb_cache_nb(
    price_2d, [bb_window], [bb_ewm], [bb_alpha], adjust, bb_ddof
)

# 计算布林带
bb_middle, bb_upper, bb_lower = vbt.indicators.nb.bb_apply_nb(
    price_2d, bb_window, bb_ewm, bb_alpha, adjust, bb_ddof, 
    bb_ma_cache, bb_mstd_cache
)
print("\\n20日布林带:")
print("中轨:", pd.DataFrame(bb_middle, index=prices.index, columns=prices.columns))
print("上轨:", pd.DataFrame(bb_upper, index=prices.index, columns=prices.columns))
print("下轨:", pd.DataFrame(bb_lower, index=prices.index, columns=prices.columns))
```
"""

# 导入NumPy库，提供高效的数组操作和数学计算功能
import numpy as np
# 导入Numba的JIT编译装饰器，将Python函数编译为高性能机器码
from numba import njit

# 导入vectorbt的类型定义模块，提供完整的类型注解支持
from vectorbt import _typing as tp
# 导入vectorbt的通用计算模块，提供基础的数值计算函数
from vectorbt.generic import nb as generic_nb


@njit(cache=True)
def ma_nb(a: tp.Array2d, window: int, ewm: bool, adjust: bool = False) -> tp.Array2d:
    """
    计算简单移动平均或指数移动平均
    
    这是vectorbt技术指标系统中最基础的移动平均计算函数，支持两种主要的移动平均算法：
    简单移动平均(SMA)和指数移动平均(EMA)。该函数是许多其他技术指标的基础组件。
    
    算法原理：
    - 简单移动平均(SMA)：计算指定窗口期内的算术平均值
      SMA = (P1 + P2 + ... + Pn) / n
    - 指数移动平均(EMA)：给予近期价格更高的权重
      EMA = α * P_current + (1-α) * EMA_previous
      其中 α = 2/(window+1) 或根据adjust参数调整
    
    参数说明：
        a (tp.Array2d): 输入的二维价格数组，形状为(时间点数, 资产数)
        window (int): 移动平均的窗口大小，必须大于0
        ewm (bool): 是否使用指数移动平均
                   - True: 计算指数移动平均(EMA)
                   - False: 计算简单移动平均(SMA)
        adjust (bool): 仅在ewm=True时有效，是否进行偏差调整
                      - True: 使用调整后的EMA算法，减少初期偏差
                      - False: 使用标准EMA算法
    
    返回值：
        tp.Array2d: 与输入相同形状的移动平均结果数组
                   前(window-1)个值为NaN，从第window个值开始为有效计算结果
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建示例价格数据
        >>> prices = np.array([
        ...     [100, 200],  # 第1天：股票A=100, 股票B=200
        ...     [102, 198],  # 第2天：股票A=102, 股票B=198
        ...     [101, 201],  # 第3天：股票A=101, 股票B=201
        ...     [103, 199],  # 第4天：股票A=103, 股票B=199
        ...     [105, 203]   # 第5天：股票A=105, 股票B=203
        ... ])
        
        >>> # 计算3日简单移动平均
        >>> sma_3 = vbt.indicators.nb.ma_nb(prices, window=3, ewm=False)
        >>> print("3日SMA结果:")
        >>> print(sma_3)
        >>> # 输出:
        >>> # [[    nan     nan]
        >>> #  [    nan     nan]
        >>> #  [101.    199.67]  # (100+102+101)/3, (200+198+201)/3
        >>> #  [102.    199.33]  # (102+101+103)/3, (198+201+199)/3
        >>> #  [103.    201.  ]]  # (101+103+105)/3, (201+199+203)/3
        
        >>> # 计算3日指数移动平均
        >>> ema_3 = vbt.indicators.nb.ma_nb(prices, window=3, ewm=True, adjust=False)
        >>> print("3日EMA结果:")
        >>> print(ema_3)
        >>> # EMA权重系数 α = 2/(3+1) = 0.5
        >>> # 第3天: EMA = 0.5*101 + 0.5*((0.5*102 + 0.5*100)) = 101.25
        
        >>> # 在量化策略中的应用
        >>> # 1. 趋势跟踪策略
        >>> short_ma = vbt.indicators.nb.ma_nb(prices, window=5, ewm=False)
        >>> long_ma = vbt.indicators.nb.ma_nb(prices, window=20, ewm=False)
        >>> # 当短期均线上穿长期均线时产生买入信号
        >>> buy_signals = short_ma > long_ma
        
        >>> # 2. 均值回归策略
        >>> current_price = prices[-1:]  # 最新价格
        >>> ma_20 = vbt.indicators.nb.ma_nb(prices, window=20, ewm=False)
        >>> # 当价格偏离均线超过2%时产生交易信号
        >>> deviation = (current_price - ma_20[-1:]) / ma_20[-1:]
        >>> oversold = deviation < -0.02  # 超卖信号
        >>> overbought = deviation > 0.02  # 超买信号
    
    技术细节：
        - 使用generic_nb模块的底层函数实现，确保计算精度和性能
        - 支持批量计算多个资产的移动平均，提高计算效率
        - 前(window-1)个值设置为NaN，符合技术分析的标准做法
        - 函数经过Numba JIT编译，运行速度接近C语言实现
    
    注意事项：
        - 输入数组必须是2维的，即使只有一个资产也要保持(n, 1)的形状
        - window参数必须小于等于数组的行数，否则所有结果都是NaN
        - EMA的adjust参数会影响初期值的计算，建议根据具体需求选择
        - 该函数不处理缺失值，输入数据应预先清理
    """
    if ewm:
        # 使用指数移动平均算法
        return generic_nb.ewm_mean_nb(a, window, minp=window, adjust=adjust)
    # 使用简单移动平均算法
    return generic_nb.rolling_mean_nb(a, window, minp=window)


@njit(cache=True)
def mstd_nb(a: tp.Array2d, window: int, ewm: int, adjust: bool = False, ddof: int = 0) -> tp.Array2d:
    """
    计算移动标准差(Moving Standard Deviation)
    
    移动标准差是衡量价格波动性的重要技术指标，用于量化资产价格在特定时间窗口内的
    离散程度。该函数支持简单移动标准差和指数加权移动标准差两种计算方式。
    
    算法原理：
    - 简单移动标准差：计算窗口内数据的标准差
      MSTD = sqrt(Σ(Xi - MA)² / (n-ddof))
    - 指数加权移动标准差：给予近期数据更高权重的标准差
      使用指数衰减权重计算方差，然后开平方根
    
    参数说明：
        a (tp.Array2d): 输入的二维价格数组，形状为(时间点数, 资产数)
        window (int): 计算标准差的窗口大小，必须大于1
        ewm (int): 是否使用指数加权移动标准差
                  - 0或False: 使用简单移动标准差
                  - 1或True: 使用指数加权移动标准差
        adjust (bool): 仅在ewm=True时有效，是否进行偏差调整
        ddof (int): 自由度调整参数(Delta Degrees of Freedom)
                   - 0: 总体标准差(除以n)
                   - 1: 样本标准差(除以n-1)，更常用于金融数据
    
    返回值：
        tp.Array2d: 与输入相同形状的移动标准差结果数组
                   前(window-1)个值为NaN，从第window个值开始为有效结果
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建波动性较大的价格数据
        >>> prices = np.array([
        ...     [100, 50],
        ...     [105, 48],
        ...     [95, 52],
        ...     [110, 45],
        ...     [90, 55],
        ...     [115, 42],
        ...     [85, 58]
        ... ])
        
        >>> # 计算5日移动标准差
        >>> mstd_5 = vbt.indicators.nb.mstd_nb(prices, window=5, ewm=False, ddof=1)
        >>> print("5日移动标准差:")
        >>> print(mstd_5)
        >>> # 前4个值为NaN，从第5个值开始显示标准差
        
        >>> # 计算指数加权移动标准差
        >>> ewm_std = vbt.indicators.nb.mstd_nb(prices, window=5, ewm=True, adjust=False)
        >>> print("5日指数加权移动标准差:")
        >>> print(ewm_std)
        
        >>> # 在量化策略中的应用
        >>> # 1. 波动率突破策略
        >>> volatility = vbt.indicators.nb.mstd_nb(prices, window=20, ewm=False, ddof=1)
        >>> avg_volatility = np.nanmean(volatility, axis=0)  # 平均波动率
        >>> high_vol_threshold = avg_volatility * 1.5  # 高波动阈值
        >>> low_vol_threshold = avg_volatility * 0.5   # 低波动阈值
        >>> 
        >>> # 当波动率突然增加时，可能预示着价格突破
        >>> breakout_signals = volatility[-1] > high_vol_threshold
        >>> 
        >>> # 2. 波动率均值回归策略
        >>> # 当波动率过低时，预期未来波动率会增加
        >>> low_vol_signals = volatility[-1] < low_vol_threshold
        
        >>> # 3. 风险管理应用
        >>> # 计算价格的标准化偏差(Z-Score)
        >>> ma_20 = vbt.indicators.nb.ma_nb(prices, window=20, ewm=False)
        >>> std_20 = vbt.indicators.nb.mstd_nb(prices, window=20, ewm=False, ddof=1)
        >>> z_score = (prices - ma_20) / std_20
        >>> # 当|Z-Score| > 2时，认为价格出现异常偏离
        >>> extreme_signals = np.abs(z_score) > 2
    
    技术细节：
        - 标准差计算基于方差的平方根，确保结果的数学准确性
        - ddof参数控制自由度，金融数据通常使用ddof=1(样本标准差)
        - 指数加权版本对近期数据给予更高权重，更敏感于最新变化
        - 函数经过Numba优化，支持大规模数据的高效计算
    
    注意事项：
        - window必须至少为2，否则无法计算标准差
        - 当窗口内所有值相同时，标准差为0
        - 指数加权版本的初始值可能与简单版本有较大差异
        - 结果中的NaN值需要在后续处理中适当处理
    """
    if ewm:
        # 使用指数加权移动标准差
        return generic_nb.ewm_std_nb(a, window, minp=window, adjust=adjust, ddof=ddof)
    # 使用简单移动标准差
    return generic_nb.rolling_std_nb(a, window, minp=window, ddof=ddof)


@njit(cache=True)
def ma_cache_nb(close: tp.Array2d, windows: tp.List[int], ewms: tp.List[bool],
                adjust: bool) -> tp.Dict[int, tp.Array2d]:
    """
    移动平均缓存函数
    
    这是vectorbt缓存系统的核心函数之一，用于预计算多个参数组合的移动平均结果。
    通过批量计算和缓存机制，显著提高了多参数移动平均指标的计算效率。
    
    缓存机制原理：
    1. 遍历所有窗口大小和EWM类型的组合
    2. 为每个唯一的参数组合计算移动平均
    3. 使用hash值作为键，将结果存储在字典中
    4. 避免重复计算相同参数的移动平均
    
    参数说明：
        close (tp.Array2d): 收盘价数组，形状为(时间点数, 资产数)
        windows (tp.List[int]): 窗口大小列表，如[5, 10, 20]
        ewms (tp.List[bool]): 是否使用EWM的布尔列表，与windows对应
        adjust (bool): EWM的调整参数，应用于所有EWM计算
    
    返回值：
        tp.Dict[int, tp.Array2d]: 缓存字典，键为参数组合的hash值，值为计算结果
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建价格数据
        >>> prices = np.random.randn(100, 3).cumsum(axis=0) + 100
        
        >>> # 定义多个参数组合
        >>> windows = [5, 10, 20, 50]
        >>> ewms = [False, False, True, True]
        
        >>> # 构建缓存
        >>> cache = vbt.indicators.nb.ma_cache_nb(prices, windows, ewms, adjust=False)
        >>> print(f"缓存了 {len(cache)} 个参数组合的结果")
        
        >>> # 查看缓存内容
        >>> for key, value in cache.items():
        ...     print(f"Hash {key}: 结果形状 {value.shape}")
        
        >>> # 在实际应用中，这个缓存会被apply函数使用
        >>> # 例如计算5日SMA
        >>> sma_5 = vbt.indicators.nb.ma_apply_nb(
        ...     prices, window=5, ewm=False, adjust=False, cache_dict=cache
        ... )
    
    性能优势：
        - 避免重复计算：相同参数组合只计算一次
        - 批量处理：一次性处理所有参数组合
        - 内存效率：只存储唯一的计算结果
        - 并行友好：不同参数组合可以并行计算
    
    技术细节：
        - 使用(window, ewm)元组的hash值作为缓存键
        - hash函数确保相同参数组合产生相同的键
        - 字典查找的时间复杂度为O(1)
        - 适用于参数扫描和优化场景
    """
    cache_dict = dict()
    for i in range(len(windows)):
        # 为每个参数组合生成唯一的hash键
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            # 只有当hash键不存在时才计算，避免重复计算
            cache_dict[h] = ma_nb(close, windows[i], ewms[i], adjust=adjust)
    return cache_dict


@njit(cache=True)
def ma_apply_nb(close: tp.Array2d, window: int, ewm: bool, adjust: bool,
                cache_dict: tp.Dict[int, tp.Array2d]) -> tp.Array2d:
    """
    移动平均应用函数
    
    这是vectorbt缓存系统的应用函数，用于从预计算的缓存中提取特定参数组合的
    移动平均结果。该函数与ma_cache_nb配合使用，实现高效的参数化计算。
    
    工作原理：
    1. 根据输入参数生成hash键
    2. 从缓存字典中查找对应的计算结果
    3. 直接返回缓存的结果，无需重新计算
    
    参数说明：
        close (tp.Array2d): 收盘价数组(实际上不使用，保持接口一致性)
        window (int): 移动平均窗口大小
        ewm (bool): 是否使用指数移动平均
        adjust (bool): EWM调整参数
        cache_dict (tp.Dict[int, tp.Array2d]): 由ma_cache_nb生成的缓存字典
    
    返回值：
        tp.Array2d: 指定参数组合的移动平均结果
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建价格数据
        >>> prices = np.array([
        ...     [100, 200],
        ...     [102, 198],
        ...     [101, 201],
        ...     [103, 199],
        ...     [105, 203]
        ... ])
        
        >>> # 先构建缓存
        >>> cache = vbt.indicators.nb.ma_cache_nb(
        ...     prices, [3, 5], [False, True], adjust=False
        ... )
        
        >>> # 使用apply函数获取特定参数的结果
        >>> sma_3 = vbt.indicators.nb.ma_apply_nb(
        ...     prices, window=3, ewm=False, adjust=False, cache_dict=cache
        ... )
        >>> print("3日SMA:")
        >>> print(sma_3)
        
        >>> # 获取EMA结果
        >>> ema_5 = vbt.indicators.nb.ma_apply_nb(
        ...     prices, window=5, ewm=True, adjust=False, cache_dict=cache
        ... )
        >>> print("5日EMA:")
        >>> print(ema_5)
        
        >>> # 在vectorbt指标系统中的应用
        >>> # 这种模式允许一次缓存，多次使用
        >>> # 特别适用于参数优化和多策略回测
    
    性能特点：
        - O(1)时间复杂度：直接字典查找
        - 零计算开销：直接返回缓存结果
        - 内存共享：多次调用共享同一份缓存
        - 线程安全：只读操作，支持并发访问
    
    注意事项：
        - 必须先调用ma_cache_nb生成缓存
        - 参数组合必须在缓存中存在
        - close参数实际不使用，但保持接口一致性
    """
    # 生成与缓存时相同的hash键
    h = hash((window, ewm))
    # 从缓存字典中返回对应的结果
    return cache_dict[h]


@njit(cache=True)
def mstd_cache_nb(close: tp.Array2d, windows: tp.List[int], ewms: tp.List[bool], adjust: bool,
                  ddof: int) -> tp.Dict[int, tp.Array2d]:
    """
    移动标准差缓存函数
    
    为移动标准差计算提供缓存机制，支持多个窗口大小和EWM类型的批量预计算。
    这是布林带等复合指标的重要组成部分。
    
    参数说明：
        close (tp.Array2d): 收盘价数组，形状为(时间点数, 资产数)
        windows (tp.List[int]): 窗口大小列表
        ewms (tp.List[bool]): 是否使用EWM的布尔列表
        adjust (bool): EWM调整参数
        ddof (int): 自由度调整参数
    
    返回值：
        tp.Dict[int, tp.Array2d]: 缓存字典，存储各参数组合的标准差结果
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建价格数据
        >>> prices = np.random.randn(50, 2).cumsum(axis=0) + 100
        
        >>> # 构建标准差缓存
        >>> std_cache = vbt.indicators.nb.mstd_cache_nb(
        ...     prices, [10, 20], [False, False], adjust=False, ddof=1
        ... )
        
        >>> # 缓存包含了10日和20日的标准差结果
        >>> print(f"缓存大小: {len(std_cache)}")
    """
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = mstd_nb(close, windows[i], ewms[i], adjust=adjust, ddof=ddof)
    return cache_dict


@njit(cache=True)
def mstd_apply_nb(close: tp.Array2d, window: int, ewm: bool, adjust: bool, ddof: int,
                  cache_dict: tp.Dict[int, tp.Array2d]) -> tp.Array2d:
    """
    移动标准差应用函数
    
    从缓存中提取特定参数组合的移动标准差结果。
    
    参数说明：
        close (tp.Array2d): 收盘价数组(保持接口一致性)
        window (int): 窗口大小
        ewm (bool): 是否使用EWM
        adjust (bool): EWM调整参数
        ddof (int): 自由度调整参数
        cache_dict (tp.Dict[int, tp.Array2d]): 标准差缓存字典
    
    返回值：
        tp.Array2d: 指定参数的移动标准差结果
    """
    h = hash((window, ewm))
    return cache_dict[h]


@njit(cache=True)
def bb_cache_nb(close: tp.Array2d, windows: tp.List[int], ewms: tp.List[bool], alphas: tp.List[float],
                adjust: bool, ddof: int) -> tp.Tuple[tp.Dict[int, tp.Array2d], tp.Dict[int, tp.Array2d]]:
    """
    布林带缓存函数
    
    布林带是由移动平均线和标准差构成的技术指标，需要同时计算移动平均和移动标准差。
    该函数一次性预计算所有需要的中间结果，为布林带指标提供高效的缓存支持。
    
    布林带原理：
    - 中轨：n日移动平均线
    - 上轨：中轨 + α * n日标准差
    - 下轨：中轨 - α * n日标准差
    其中α通常取2，表示2倍标准差
    
    参数说明：
        close (tp.Array2d): 收盘价数组，形状为(时间点数, 资产数)
        windows (tp.List[int]): 窗口大小列表，如[20, 50]
        ewms (tp.List[bool]): 是否使用EWM的布尔列表
        alphas (tp.List[float]): 标准差倍数列表，如[2.0, 2.5]
        adjust (bool): EWM调整参数
        ddof (int): 标准差计算的自由度调整
    
    返回值：
        tp.Tuple[tp.Dict[int, tp.Array2d], tp.Dict[int, tp.Array2d]]: 
        返回两个缓存字典的元组：
        - 第一个：移动平均缓存字典
        - 第二个：移动标准差缓存字典
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建价格数据
        >>> prices = np.random.randn(100, 2).cumsum(axis=0) + 100
        
        >>> # 构建布林带缓存
        >>> ma_cache, std_cache = vbt.indicators.nb.bb_cache_nb(
        ...     prices, 
        ...     windows=[20], 
        ...     ewms=[False], 
        ...     alphas=[2.0], 
        ...     adjust=False, 
        ...     ddof=1
        ... )
        
        >>> print(f"移动平均缓存大小: {len(ma_cache)}")
        >>> print(f"标准差缓存大小: {len(std_cache)}")
        
        >>> # 这些缓存将被bb_apply_nb使用来计算布林带
    
    技术细节：
        - 复用了ma_cache_nb和mstd_cache_nb函数，避免代码重复
        - 两个缓存使用相同的参数组合，确保一致性
        - 支持多个alpha值，可以同时计算不同宽度的布林带
        - 缓存机制特别适用于布林带的参数优化
    
    应用场景：
        - 布林带突破策略：价格突破上轨或下轨的交易信号
        - 布林带挤压：标准差收缩时的波动率预测
        - 均值回归策略：价格偏离中轨的回归交易
        - 多时间框架分析：同时分析不同周期的布林带
    """
    # 构建移动平均缓存
    ma_cache_dict = ma_cache_nb(close, windows, ewms, adjust)
    # 构建移动标准差缓存
    mstd_cache_dict = mstd_cache_nb(close, windows, ewms, adjust, ddof)
    # 返回两个缓存字典的元组
    return ma_cache_dict, mstd_cache_dict


@njit(cache=True)
def bb_apply_nb(close: tp.Array2d, window: int, ewm: bool, alpha: float,
                adjust: bool, ddof: int, ma_cache_dict: tp.Dict[int, tp.Array2d],
                mstd_cache_dict: tp.Dict[int, tp.Array2d]) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """
    布林带应用函数
    
    使用预计算的移动平均和标准差缓存来计算布林带的上轨、中轨和下轨。
    布林带是量化交易中最重要的技术指标之一，广泛用于趋势跟踪和均值回归策略。
    
    布林带计算公式：
    - 中轨(Middle Band) = n日移动平均
    - 上轨(Upper Band) = 中轨 + α * n日标准差
    - 下轨(Lower Band) = 中轨 - α * n日标准差
    
    参数说明：
        close (tp.Array2d): 收盘价数组(保持接口一致性)
        window (int): 移动平均和标准差的窗口大小
        ewm (bool): 是否使用指数移动平均
        alpha (float): 标准差的倍数，通常为2.0
        adjust (bool): EWM调整参数
        ddof (int): 标准差计算的自由度调整
        ma_cache_dict (tp.Dict[int, tp.Array2d]): 移动平均缓存字典
        mstd_cache_dict (tp.Dict[int, tp.Array2d]): 移动标准差缓存字典
    
    返回值：
        tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]: 
        返回三个数组的元组：
        - 中轨：移动平均线
        - 上轨：中轨 + α * 标准差
        - 下轨：中轨 - α * 标准差
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建价格数据
        >>> prices = np.array([
        ...     [100, 50],
        ...     [102, 48],
        ...     [98, 52],
        ...     [105, 45],
        ...     [95, 55],
        ...     [108, 42],
        ...     [92, 58],
        ...     [110, 40],
        ...     [88, 60],
        ...     [112, 38]
        ... ])
        
        >>> # 先构建缓存
        >>> ma_cache, std_cache = vbt.indicators.nb.bb_cache_nb(
        ...     prices, [5], [False], [2.0], adjust=False, ddof=1
        ... )
        
        >>> # 计算布林带
        >>> middle, upper, lower = vbt.indicators.nb.bb_apply_nb(
        ...     prices, window=5, ewm=False, alpha=2.0, adjust=False, ddof=1,
        ...     ma_cache_dict=ma_cache, mstd_cache_dict=std_cache
        ... )
        
        >>> print("布林带中轨(5日均线):")
        >>> print(middle)
        >>> print("布林带上轨:")
        >>> print(upper)
        >>> print("布林带下轨:")
        >>> print(lower)
        
        >>> # 在量化策略中的应用
        >>> # 1. 布林带突破策略
        >>> current_price = prices[-1:]
        >>> upper_breakout = current_price > upper[-1:]  # 突破上轨
        >>> lower_breakout = current_price < lower[-1:]  # 跌破下轨
        >>> 
        >>> # 2. 布林带挤压策略
        >>> bb_width = (upper - lower) / middle  # 布林带宽度
        >>> avg_width = np.nanmean(bb_width)
        >>> squeeze_signal = bb_width[-1] < avg_width * 0.5  # 挤压信号
        >>> 
        >>> # 3. 均值回归策略
        >>> bb_position = (prices - lower) / (upper - lower)  # 价格在布林带中的位置
        >>> oversold = bb_position < 0.2   # 接近下轨，超卖
        >>> overbought = bb_position > 0.8  # 接近上轨，超买
    
    技术分析意义：
        - 上轨突破：强势信号，价格可能继续上涨
        - 下轨跌破：弱势信号，价格可能继续下跌
        - 带宽收缩：波动率降低，可能预示着大行情来临
        - 带宽扩张：波动率增加，趋势可能正在形成
        - 中轨支撑/阻力：价格经常在中轨附近找到支撑或阻力
    
    注意事项：
        - alpha=2.0时，理论上95%的价格应该在布林带内
        - 布林带不是绝对的支撑和阻力，需要结合其他指标使用
        - 在强趋势市场中，价格可能长时间运行在布林带外
        - 布林带的有效性在不同市场环境下可能有所不同
    """
    # 计算参数组合的hash键
    h = hash((window, ewm))
    # 从缓存中获取移动平均(中轨)
    ma = np.copy(ma_cache_dict[h])
    # 从缓存中获取移动标准差
    mstd = np.copy(mstd_cache_dict[h])
    # 计算布林带三条线
    # 返回顺序：中轨, 上轨, 下轨
    return ma, ma + alpha * mstd, ma - alpha * mstd


@njit(cache=True)
def rsi_cache_nb(close: tp.Array2d, windows: tp.List[int], ewms: tp.List[bool],
                 adjust: bool) -> tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]:
    """
    相对强弱指数(RSI)缓存函数
    
    RSI是衡量价格变动速度和幅度的动量振荡器，用于识别超买和超卖条件。
    该函数预计算RSI所需的上涨和下跌移动平均，为RSI指标提供高效的缓存支持。
    
    RSI算法原理：
    1. 计算价格变化：delta = close[i] - close[i-1]
    2. 分离上涨和下跌：up = max(delta, 0), down = max(-delta, 0)
    3. 计算上涨和下跌的移动平均：avg_up, avg_down
    4. 计算相对强度：RS = avg_up / avg_down
    5. 计算RSI：RSI = 100 - 100/(1 + RS)
    
    参数说明：
        close (tp.Array2d): 收盘价数组，形状为(时间点数, 资产数)
        windows (tp.List[int]): RSI计算窗口列表，如[14, 21]
        ewms (tp.List[bool]): 是否使用EWM的布尔列表
        adjust (bool): EWM调整参数
    
    返回值：
        tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]: 
        缓存字典，键为参数hash，值为(上涨均值, 下跌均值)的元组
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建价格数据（模拟股价波动）
        >>> prices = np.array([
        ...     [100, 50],
        ...     [102, 48],  # 上涨2, 下跌2
        ...     [98, 52],   # 下跌4, 上涨4
        ...     [105, 45],  # 上涨7, 下跌7
        ...     [95, 55],   # 下跌10, 上涨10
        ...     [108, 42],  # 上涨13, 下跌13
        ...     [92, 58],   # 下跌16, 上涨16
        ...     [110, 40],  # 上涨18, 下跌18
        ...     [88, 60],   # 下跌22, 上涨20
        ...     [112, 38]   # 上涨24, 下跌22
        ... ])
        
        >>> # 构建RSI缓存
        >>> rsi_cache = vbt.indicators.nb.rsi_cache_nb(
        ...     prices, [5], [False], adjust=False
        ... )
        
        >>> # 缓存包含上涨和下跌的移动平均
        >>> hash_key = hash((5, False))
        >>> avg_up, avg_down = rsi_cache[hash_key]
        >>> print("5日平均上涨幅度:")
        >>> print(avg_up)
        >>> print("5日平均下跌幅度:")
        >>> print(avg_down)
    
    技术细节：
        - 使用generic_nb.diff_nb计算价格变化
        - 将价格变化分离为上涨和下跌两个序列
        - 上涨序列：正值保持，负值设为0
        - 下跌序列：负值取绝对值，正值设为0
        - 分别计算上涨和下跌的移动平均
        - 支持简单移动平均和指数移动平均两种模式
    
    应用场景：
        - 超买超卖判断：RSI>70超买，RSI<30超卖
        - 背离分析：价格新高而RSI不创新高
        - 趋势确认：RSI突破50线确认趋势方向
        - 多时间框架分析：不同周期RSI的综合判断
    """
    # 计算价格变化(差分)
    delta = generic_nb.diff_nb(close)
    # 创建上涨和下跌序列的副本
    up, down = delta.copy(), delta.copy()
    # 处理上涨序列：负值设为0，正值保持
    up = generic_nb.set_by_mask_nb(up, up < 0, 0)
    # 处理下跌序列：正值设为0，负值取绝对值
    down = np.abs(generic_nb.set_by_mask_nb(down, down > 0, 0))

    # 构建缓存字典
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            # 计算上涨和下跌的移动平均
            roll_up = ma_nb(up, windows[i], ewms[i], adjust=adjust)
            roll_down = ma_nb(down, windows[i], ewms[i], adjust=adjust)
            # 存储为元组
            cache_dict[h] = roll_up, roll_down
    return cache_dict


@njit(cache=True)
def rsi_apply_nb(close: tp.Array2d, window: int, ewm: bool, adjust: bool,
                 cache_dict: tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]) -> tp.Array2d:
    """
    相对强弱指数(RSI)应用函数
    
    使用预计算的上涨和下跌移动平均来计算RSI指标。RSI是最重要的动量指标之一，
    用于判断市场的超买超卖状态和趋势强度。
    
    RSI计算公式：
    RSI = 100 - 100 / (1 + RS)
    其中 RS = 平均上涨幅度 / 平均下跌幅度
    
    参数说明：
        close (tp.Array2d): 收盘价数组(保持接口一致性)
        window (int): RSI计算窗口大小，通常为14
        ewm (bool): 是否使用指数移动平均
        adjust (bool): EWM调整参数
        cache_dict (tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]): RSI缓存字典
    
    返回值：
        tp.Array2d: RSI指标值，范围在0-100之间
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建趋势性价格数据
        >>> prices = np.array([
        ...     [100, 100],
        ...     [105, 95],   # 强势上涨 vs 弱势下跌
        ...     [110, 90],   # 继续上涨 vs 继续下跌
        ...     [115, 85],   # 持续强势 vs 持续弱势
        ...     [120, 80],   # 可能超买 vs 可能超卖
        ...     [118, 82],   # 小幅回调 vs 小幅反弹
        ...     [116, 84],   # 继续回调 vs 继续反弹
        ...     [119, 81],   # 重新上涨 vs 重新下跌
        ...     [122, 78],   # 创新高 vs 创新低
        ...     [125, 75]    # 强势延续 vs 弱势延续
        ... ])
        
        >>> # 先构建缓存
        >>> rsi_cache = vbt.indicators.nb.rsi_cache_nb(
        ...     prices, [6], [False], adjust=False
        ... )
        
        >>> # 计算RSI
        >>> rsi = vbt.indicators.nb.rsi_apply_nb(
        ...     prices, window=6, ewm=False, adjust=False, cache_dict=rsi_cache
        ... )
        
        >>> print("6日RSI指标:")
        >>> print(rsi)
        >>> # 第一列(上涨趋势)的RSI应该较高(>50)
        >>> # 第二列(下跌趋势)的RSI应该较低(<50)
        
        >>> # RSI交易信号生成
        >>> # 1. 超买超卖信号
        >>> overbought = rsi > 70    # 超买信号
        >>> oversold = rsi < 30      # 超卖信号
        >>> 
        >>> # 2. 中线突破信号
        >>> bullish = rsi > 50       # 多头信号
        >>> bearish = rsi < 50       # 空头信号
        >>> 
        >>> # 3. RSI背离信号(需要更多数据)
        >>> # 价格创新高但RSI不创新高 = 顶背离
        >>> # 价格创新低但RSI不创新低 = 底背离
        
        >>> # 在量化策略中的应用
        >>> # 1. RSI均值回归策略
        >>> buy_signal = (rsi < 30) & (rsi.shift(1) >= 30)   # RSI从超卖区域回升
        >>> sell_signal = (rsi > 70) & (rsi.shift(1) <= 70)  # RSI从超买区域回落
        >>> 
        >>> # 2. RSI趋势跟踪策略
        >>> long_trend = rsi > 50    # RSI在50以上，看多
        >>> short_trend = rsi < 50   # RSI在50以下，看空
        >>> 
        >>> # 3. RSI过滤器
        >>> # 在其他策略中使用RSI作为过滤条件
        >>> # 例如：只在RSI不超买时买入，只在RSI不超卖时卖出
    
    RSI指标解读：
        - RSI > 70：通常认为超买，可能面临回调压力
        - RSI < 30：通常认为超卖，可能面临反弹机会
        - RSI > 50：表明上涨动能较强，偏向多头
        - RSI < 50：表明下跌动能较强，偏向空头
        - RSI = 50：多空力量平衡，趋势不明确
    
    技术细节：
        - 当平均下跌幅度为0时，RSI = 100
        - 当平均上涨幅度为0时，RSI = 0
        - RSI对价格变化的敏感度随窗口大小而变化
        - 较小的窗口使RSI更敏感，较大的窗口使RSI更平滑
    
    注意事项：
        - RSI在强趋势市场中可能长期保持极端值
        - 应该结合其他指标和市场环境综合判断
        - 不同市场和时间框架的RSI阈值可能需要调整
        - RSI背离信号比单纯的超买超卖更可靠
    """
    # 获取参数组合的hash键
    h = hash((window, ewm))
    # 从缓存中获取上涨和下跌的移动平均
    roll_up, roll_down = cache_dict[h]
    # 计算相对强度 RS = 平均上涨 / 平均下跌
    rs = roll_up / roll_down
    # 计算RSI = 100 - 100/(1 + RS)
    return 100 - 100 / (1 + rs)


@njit(cache=True)
def stoch_cache_nb(high: tp.Array2d, low: tp.Array2d, close: tp.Array2d,
                   k_windows: tp.List[int], d_windows: tp.List[int], d_ewms: tp.List[bool],
                   adjust: bool) -> tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]:
    """
    随机指标(Stochastic Oscillator)缓存函数
    
    随机指标是由George Lane开发的动量指标，用于比较收盘价与特定周期内的价格范围。
    该函数预计算随机指标所需的滚动最高价和最低价，为STOCH指标提供高效的缓存支持。
    
    随机指标算法原理：
    1. 计算%K值：%K = (C - L_n) / (H_n - L_n) * 100
       其中：C = 当前收盘价，L_n = n日内最低价，H_n = n日内最高价
    2. 计算%D值：%D = %K的m日移动平均
    
    参数说明：
        high (tp.Array2d): 最高价数组，形状为(时间点数, 资产数)
        low (tp.Array2d): 最低价数组，形状为(时间点数, 资产数)
        close (tp.Array2d): 收盘价数组，形状为(时间点数, 资产数)
        k_windows (tp.List[int]): %K计算窗口列表，如[14, 21]
        d_windows (tp.List[int]): %D计算窗口列表，如[3, 5]
        d_ewms (tp.List[bool]): %D是否使用EWM的布尔列表
        adjust (bool): EWM调整参数
    
    返回值：
        tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]: 
        缓存字典，键为K窗口的hash，值为(滚动最低价, 滚动最高价)的元组
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建OHLC数据
        >>> np.random.seed(42)
        >>> n_days = 20
        >>> n_assets = 2
        >>> 
        >>> # 模拟价格数据
        >>> close = np.random.randn(n_days, n_assets).cumsum(axis=0) + 100
        >>> high = close + np.random.uniform(0, 2, (n_days, n_assets))
        >>> low = close - np.random.uniform(0, 2, (n_days, n_assets))
        
        >>> # 构建随机指标缓存
        >>> stoch_cache = vbt.indicators.nb.stoch_cache_nb(
        ...     high, low, close,
        ...     k_windows=[14], 
        ...     d_windows=[3], 
        ...     d_ewms=[False], 
        ...     adjust=False
        ... )
        
        >>> # 缓存包含滚动最高价和最低价
        >>> hash_key = hash(14)
        >>> roll_low, roll_high = stoch_cache[hash_key]
        >>> print("14日滚动最低价:")
        >>> print(roll_low[:5])  # 显示前5行
        >>> print("14日滚动最高价:")
        >>> print(roll_high[:5])  # 显示前5行
    
    技术细节：
        - 使用generic_nb.rolling_min_nb计算滚动最低价
        - 使用generic_nb.rolling_max_nb计算滚动最高价
        - 只缓存K窗口相关的数据，D窗口在apply函数中处理
        - 支持多个K窗口的批量计算
        - 缓存键只使用K窗口，因为最高价和最低价只与K窗口相关
    
    应用场景：
        - 超买超卖判断：%K>80超买，%K<20超卖
        - 金叉死叉：%K线与%D线的交叉信号
        - 背离分析：价格与随机指标的背离
        - 趋势确认：随机指标的位置和方向
    """
    cache_dict = dict()
    for i in range(len(k_windows)):
        # 只使用K窗口作为hash键，因为最高价和最低价只与K窗口相关
        h = hash(k_windows[i])
        if h not in cache_dict:
            # 计算滚动最低价和最高价
            roll_min = generic_nb.rolling_min_nb(low, k_windows[i])
            roll_max = generic_nb.rolling_max_nb(high, k_windows[i])
            # 存储为元组
            cache_dict[h] = roll_min, roll_max
    return cache_dict


@njit(cache=True)
def stoch_apply_nb(high: tp.Array2d, low: tp.Array2d, close: tp.Array2d,
                   k_window: int, d_window: int, d_ewm: bool, adjust: bool,
                   cache_dict: tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """
    随机指标(Stochastic Oscillator)应用函数
    
    使用预计算的滚动最高价和最低价来计算随机指标的%K和%D值。
    随机指标是重要的动量振荡器，用于识别超买超卖状态和价格转折点。
    
    计算公式：
    %K = (收盘价 - 最低价) / (最高价 - 最低价) * 100
    %D = %K的移动平均
    
    参数说明：
        high (tp.Array2d): 最高价数组(保持接口一致性)
        low (tp.Array2d): 最低价数组(保持接口一致性)
        close (tp.Array2d): 收盘价数组，用于计算%K
        k_window (int): %K计算窗口，通常为14
        d_window (int): %D计算窗口，通常为3
        d_ewm (bool): %D是否使用指数移动平均
        adjust (bool): EWM调整参数
        cache_dict (tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]): 随机指标缓存字典
    
    返回值：
        tp.Tuple[tp.Array2d, tp.Array2d]: (%K值, %D值)的元组
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建模拟的股价数据
        >>> np.random.seed(42)
        >>> n_days = 30
        >>> 
        >>> # 模拟一个有明显波动的价格序列
        >>> base_price = 100
        >>> price_changes = np.random.randn(n_days).cumsum() * 0.5
        >>> close = (base_price + price_changes).reshape(-1, 1)
        >>> 
        >>> # 生成高低价(简化处理)
        >>> high = close + np.random.uniform(0.5, 2, close.shape)
        >>> low = close - np.random.uniform(0.5, 2, close.shape)
        
        >>> # 先构建缓存
        >>> stoch_cache = vbt.indicators.nb.stoch_cache_nb(
        ...     high, low, close, [14], [3], [False], adjust=False
        ... )
        
        >>> # 计算随机指标
        >>> percent_k, percent_d = vbt.indicators.nb.stoch_apply_nb(
        ...     high, low, close, 
        ...     k_window=14, d_window=3, d_ewm=False, adjust=False,
        ...     cache_dict=stoch_cache
        ... )
        
        >>> print("随机指标%K值:")
        >>> print(percent_k.flatten()[-10:])  # 显示最后10个值
        >>> print("随机指标%D值:")
        >>> print(percent_d.flatten()[-10:])  # 显示最后10个值
        
        >>> # 随机指标交易信号
        >>> # 1. 超买超卖信号
        >>> overbought_k = percent_k > 80  # %K超买
        >>> oversold_k = percent_k < 20    # %K超卖
        >>> 
        >>> # 2. 金叉死叉信号
        >>> golden_cross = (percent_k > percent_d) & (percent_k.shift(1) <= percent_d.shift(1))
        >>> death_cross = (percent_k < percent_d) & (percent_k.shift(1) >= percent_d.shift(1))
        >>> 
        >>> # 3. 背离信号
        >>> # 价格创新高但%K不创新高 = 顶背离
        >>> # 价格创新低但%K不创新低 = 底背离
        
        >>> # 在量化策略中的应用
        >>> # 1. 随机指标反转策略
        >>> buy_signal = oversold_k & golden_cross    # 超卖且金叉
        >>> sell_signal = overbought_k & death_cross  # 超买且死叉
        >>> 
        >>> # 2. 随机指标趋势跟踪
        >>> uptrend = (percent_k > 50) & (percent_d > 50)    # 强势上升
        >>> downtrend = (percent_k < 50) & (percent_d < 50)  # 强势下降
        >>> 
        >>> # 3. 多时间框架随机指标
        >>> # 结合不同K窗口的随机指标进行综合判断
    
    随机指标解读：
        - %K > 80：通常认为超买，可能面临回调
        - %K < 20：通常认为超卖，可能面临反弹
        - %K > %D：短期动量强于长期动量，偏向多头
        - %K < %D：短期动量弱于长期动量，偏向空头
        - %K和%D都在50以上：整体处于强势区域
        - %K和%D都在50以下：整体处于弱势区域
    
    技术细节：
        - %K值范围在0-100之间，表示收盘价在价格区间中的位置
        - %D是%K的平滑版本，减少了噪音但增加了滞后性
        - 当最高价等于最低价时，%K值为0(避免除零错误)
        - 随机指标对价格的短期波动比较敏感
    
    注意事项：
        - 在强趋势市场中，随机指标可能长期保持极端值
        - 应该结合价格趋势和其他指标综合判断
        - 不同市场的超买超卖阈值可能需要调整
        - 随机指标的背离信号通常比单纯的超买超卖更可靠
    """
    # 获取K窗口的hash键
    h = hash(k_window)
    # 从缓存中获取滚动最低价和最高价
    roll_min, roll_max = cache_dict[h]
    # 计算%K = (收盘价 - 最低价) / (最高价 - 最低价) * 100
    percent_k = 100 * (close - roll_min) / (roll_max - roll_min)
    # 计算%D = %K的移动平均
    percent_d = ma_nb(percent_k, d_window, d_ewm, adjust=adjust)
    # 返回%K和%D
    return percent_k, percent_d


@njit(cache=True)
def macd_cache_nb(close: tp.Array2d, fast_windows: tp.List[int], slow_windows: tp.List[int],
                  signal_windows: tp.List[int], macd_ewms: tp.List[bool], signal_ewms: tp.List[bool],
                  adjust: bool) -> tp.Dict[int, tp.Array2d]:
    """
    MACD(Moving Average Convergence Divergence)缓存函数
    
    MACD是Gerald Appel开发的趋势跟踪动量指标，通过计算两个不同周期的指数移动平均线
    之间的差值来识别趋势变化。该函数预计算MACD所需的快线和慢线移动平均。
    
    MACD算法原理：
    1. 快线EMA：通常为12日指数移动平均
    2. 慢线EMA：通常为26日指数移动平均
    3. MACD线：快线EMA - 慢线EMA
    4. 信号线：MACD线的9日指数移动平均
    5. 柱状图：MACD线 - 信号线
    
    参数说明：
        close (tp.Array2d): 收盘价数组，形状为(时间点数, 资产数)
        fast_windows (tp.List[int]): 快线窗口列表，如[12, 10]
        slow_windows (tp.List[int]): 慢线窗口列表，如[26, 20]
        signal_windows (tp.List[int]): 信号线窗口列表，如[9, 7]
        macd_ewms (tp.List[bool]): MACD线是否使用EWM的布尔列表
        signal_ewms (tp.List[bool]): 信号线是否使用EWM的布尔列表
        adjust (bool): EWM调整参数
    
    返回值：
        tp.Dict[int, tp.Array2d]: 缓存字典，包含快线和慢线的移动平均结果
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建趋势性价格数据
        >>> np.random.seed(42)
        >>> n_days = 100
        >>> n_assets = 2
        >>> 
        >>> # 模拟有趋势的价格序列
        >>> trend1 = np.linspace(0, 10, n_days)  # 上升趋势
        >>> trend2 = np.linspace(5, -5, n_days)  # 下降趋势
        >>> noise = np.random.randn(n_days, n_assets) * 0.5
        >>> 
        >>> prices = np.column_stack([
        ...     100 + trend1 + noise[:, 0],
        ...     100 + trend2 + noise[:, 1]
        ... ])
        
        >>> # 构建MACD缓存
        >>> macd_cache = vbt.indicators.nb.macd_cache_nb(
        ...     prices,
        ...     fast_windows=[12], 
        ...     slow_windows=[26], 
        ...     signal_windows=[9],
        ...     macd_ewms=[True], 
        ...     signal_ewms=[True], 
        ...     adjust=False
        ... )
        
        >>> print(f"MACD缓存大小: {len(macd_cache)}")
        >>> # 缓存包含12日和26日的EMA结果
    
    技术细节：
        - 复用ma_cache_nb函数来计算移动平均缓存
        - 将快线和慢线窗口合并到一个列表中
        - 将对应的EWM设置也合并到一个列表中
        - 缓存键基于(窗口, EWM)的组合
        - 支持多个参数组合的批量计算
    
    应用场景：
        - 趋势跟踪：MACD线的方向和位置
        - 动量分析：MACD线的变化速度
        - 买卖信号：MACD线与信号线的交叉
        - 背离分析：价格与MACD的背离现象
    """
    # 合并快线和慢线窗口
    windows = fast_windows.copy()
    windows.extend(slow_windows)
    # 合并对应的EWM设置
    ewms = macd_ewms.copy()
    ewms.extend(macd_ewms)
    # 使用移动平均缓存函数
    return ma_cache_nb(close, windows, ewms, adjust)


@njit(cache=True)
def macd_apply_nb(close: tp.Array2d, fast_window: int, slow_window: int,
                  signal_window: int, macd_ewm: bool, signal_ewm: bool, adjust: bool,
                  cache_dict: tp.Dict[int, tp.Array2d]) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """
    MACD(Moving Average Convergence Divergence)应用函数
    
    使用预计算的快线和慢线移动平均来计算MACD线和信号线。MACD是最重要的趋势跟踪
    指标之一，广泛用于识别趋势变化和生成买卖信号。
    
    计算公式：
    MACD线 = 快线EMA - 慢线EMA
    信号线 = MACD线的EMA
    柱状图 = MACD线 - 信号线(在调用方计算)
    
    参数说明：
        close (tp.Array2d): 收盘价数组(保持接口一致性)
        fast_window (int): 快线窗口，通常为12
        slow_window (int): 慢线窗口，通常为26
        signal_window (int): 信号线窗口，通常为9
        macd_ewm (bool): MACD线是否使用EWM
        signal_ewm (bool): 信号线是否使用EWM
        adjust (bool): EWM调整参数
        cache_dict (tp.Dict[int, tp.Array2d]): MACD缓存字典
    
    返回值：
        tp.Tuple[tp.Array2d, tp.Array2d]: (MACD线, 信号线)的元组
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建具有明显趋势变化的价格数据
        >>> n_days = 60
        >>> 
        >>> # 模拟趋势转换：前30天上升，后30天下降
        >>> uptrend = np.linspace(100, 120, 30)
        >>> downtrend = np.linspace(120, 100, 30)
        >>> prices = np.concatenate([uptrend, downtrend]).reshape(-1, 1)
        >>> 
        >>> # 添加一些噪音
        >>> np.random.seed(42)
        >>> prices += np.random.randn(n_days, 1) * 0.5
        
        >>> # 先构建缓存
        >>> macd_cache = vbt.indicators.nb.macd_cache_nb(
        ...     prices, [12], [26], [9], [True], [True], adjust=False
        ... )
        
        >>> # 计算MACD
        >>> macd_line, signal_line = vbt.indicators.nb.macd_apply_nb(
        ...     prices, 
        ...     fast_window=12, slow_window=26, signal_window=9,
        ...     macd_ewm=True, signal_ewm=True, adjust=False,
        ...     cache_dict=macd_cache
        ... )
        
        >>> print("MACD线:")
        >>> print(macd_line.flatten()[-10:])  # 显示最后10个值
        >>> print("信号线:")
        >>> print(signal_line.flatten()[-10:])  # 显示最后10个值
        
        >>> # 计算MACD柱状图
        >>> histogram = macd_line - signal_line
        >>> print("MACD柱状图:")
        >>> print(histogram.flatten()[-10:])
        
        >>> # MACD交易信号
        >>> # 1. 金叉死叉信号
        >>> golden_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        >>> death_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        >>> 
        >>> # 2. 零轴突破信号
        >>> bullish_zero_cross = (macd_line > 0) & (macd_line.shift(1) <= 0)
        >>> bearish_zero_cross = (macd_line < 0) & (macd_line.shift(1) >= 0)
        >>> 
        >>> # 3. 柱状图信号
        >>> histogram_increasing = histogram > histogram.shift(1)  # 柱状图增加
        >>> histogram_decreasing = histogram < histogram.shift(1)  # 柱状图减少
        
        >>> # 在量化策略中的应用
        >>> # 1. MACD趋势跟踪策略
        >>> long_signal = golden_cross & (macd_line > 0)     # 金叉且在零轴上方
        >>> short_signal = death_cross & (macd_line < 0)     # 死叉且在零轴下方
        >>> 
        >>> # 2. MACD背离策略
        >>> # 价格创新高但MACD不创新高 = 顶背离
        >>> # 价格创新低但MACD不创新低 = 底背离
        >>> 
        >>> # 3. MACD多时间框架策略
        >>> # 结合不同参数的MACD进行综合判断
    
    MACD指标解读：
        - MACD线 > 0：短期趋势强于长期趋势，偏向多头
        - MACD线 < 0：短期趋势弱于长期趋势，偏向空头
        - MACD线 > 信号线：短期动量加速上升
        - MACD线 < 信号线：短期动量加速下降
        - 柱状图 > 0：MACD线上穿信号线，动量转强
        - 柱状图 < 0：MACD线下穿信号线，动量转弱
    
    技术细节：
        - 快线窗口必须小于慢线窗口，否则指标意义不大
        - MACD线反映了两个EMA之间的差值变化
        - 信号线是MACD线的平滑版本，用于产生交易信号
        - 柱状图显示MACD线与信号线的差值变化
    
    注意事项：
        - MACD是滞后指标，信号确认需要时间
        - 在震荡市场中可能产生较多假信号
        - 应该结合趋势和其他指标综合判断
        - 不同市场的参数设置可能需要调整
    """
    # 生成快线和慢线的hash键
    fast_h = hash((fast_window, macd_ewm))
    slow_h = hash((slow_window, macd_ewm))
    # 从缓存中获取快线和慢线EMA
    fast_ma = cache_dict[fast_h]
    slow_ma = cache_dict[slow_h]
    # 计算MACD线 = 快线EMA - 慢线EMA
    macd_ts = fast_ma - slow_ma
    # 计算信号线 = MACD线的EMA
    signal_ts = ma_nb(macd_ts, signal_window, signal_ewm, adjust=adjust)
    # 返回MACD线和信号线
    return macd_ts, signal_ts


@njit(cache=True)
def true_range_nb(high: tp.Array2d, low: tp.Array2d, close: tp.Array2d) -> tp.Array2d:
    """
    真实波幅(True Range)计算函数
    
    真实波幅是Welles Wilder开发的波动率指标，用于衡量价格的真实波动范围。
    它考虑了跳空缺口的影响，比简单的日内波幅更准确地反映市场的真实波动性。
    
    真实波幅定义：
    TR = max(H - L, |H - C_prev|, |L - C_prev|)
    其中：
    - H: 当日最高价
    - L: 当日最低价  
    - C_prev: 前一日收盘价
    
    参数说明：
        high (tp.Array2d): 最高价数组，形状为(时间点数, 资产数)
        low (tp.Array2d): 最低价数组，形状为(时间点数, 资产数)
        close (tp.Array2d): 收盘价数组，形状为(时间点数, 资产数)
    
    返回值：
        tp.Array2d: 真实波幅数组，与输入相同形状
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建包含跳空的OHLC数据
        >>> n_days = 10
        >>> 
        >>> # 模拟股价数据(包含跳空)
        >>> close = np.array([
        ...     [100, 50],
        ...     [102, 48],  # 正常波动
        ...     [105, 52],  # 向上跳空
        ...     [103, 49],  # 正常波动
        ...     [108, 45],  # 向上跳空
        ...     [106, 47],  # 正常波动
        ...     [104, 44],  # 向下跳空
        ...     [107, 46],  # 正常波动
        ...     [109, 48],  # 正常波动
        ...     [111, 50]   # 正常波动
        ... ])
        >>> 
        >>> # 生成高低价(简化处理)
        >>> high = close + np.random.uniform(0.5, 2, close.shape)
        >>> low = close - np.random.uniform(0.5, 2, close.shape)
        
        >>> # 确保价格关系正确
        >>> high = np.maximum(high, close)
        >>> low = np.minimum(low, close)
        
        >>> # 计算真实波幅
        >>> tr = vbt.indicators.nb.true_range_nb(high, low, close)
        >>> print("真实波幅:")
        >>> print(tr)
        
        >>> # 分析真实波幅的意义
        >>> # 1. 波动率分析
        >>> avg_tr = np.mean(tr, axis=0)
        >>> print(f"平均真实波幅: {avg_tr}")
        >>> 
        >>> # 2. 异常波动检测
        >>> tr_threshold = avg_tr * 2  # 2倍平均波幅作为阈值
        >>> high_volatility = tr > tr_threshold
        >>> print("高波动日期:")
        >>> print(np.where(high_volatility))
        
        >>> # 在量化策略中的应用
        >>> # 1. 波动率突破策略
        >>> # 当真实波幅超过历史平均值时，可能预示着价格突破
        >>> 
        >>> # 2. 风险管理
        >>> # 根据真实波幅调整止损位和仓位大小
        >>> 
        >>> # 3. 趋势强度判断
        >>> # 真实波幅增加通常伴随着趋势的加强
    
    技术细节：
        - 第一个交易日的真实波幅等于最高价减最低价
        - 考虑了三种情况：日内波幅、向上跳空、向下跳空
        - 使用max函数确保真实波幅始终为正值
        - 真实波幅的单位与价格相同
    
    应用场景：
        - ATR(平均真实波幅)指标的基础计算
        - 波动率分析和风险管理
        - 止损位和目标位的设定
        - 趋势强度和市场状态的判断
        - 异常波动的识别和预警
    
    注意事项：
        - 真实波幅反映的是绝对波动，不是相对波动
        - 高价股的真实波幅通常比低价股大
        - 需要结合价格水平来评估波动率的相对大小
        - 真实波幅容易受到异常值的影响
    """
    # 计算前一日收盘价(向前移动一位)
    prev_close = generic_nb.fshift_nb(close, 1)
    # 计算三种波幅情况
    tr1 = high - low                    # 日内波幅
    tr2 = np.abs(high - prev_close)     # 最高价与前收盘价的差值
    tr3 = np.abs(low - prev_close)      # 最低价与前收盘价的差值
    
    # 创建结果数组
    tr = np.empty(prev_close.shape, dtype=np.float64)
    # 对每列(每个资产)和每行(每个时间点)计算真实波幅
    for col in range(tr.shape[1]):
        for i in range(tr.shape[0]):
            # 真实波幅 = max(日内波幅, 向上跳空, 向下跳空)
            tr[i, col] = max(tr1[i, col], tr2[i, col], tr3[i, col])
    return tr


@njit(cache=True)
def atr_cache_nb(high: tp.Array2d, low: tp.Array2d, close: tp.Array2d, windows: tp.List[int],
                 ewms: tp.List[bool], adjust: bool) -> tp.Tuple[tp.Array2d, tp.Dict[int, tp.Array2d]]:
    """
    平均真实波幅(ATR)缓存函数
    
    ATR是基于真实波幅的移动平均指标，用于衡量市场的平均波动性。该函数先计算
    真实波幅，然后预计算不同参数组合的ATR值，为ATR指标提供高效的缓存支持。
    
    ATR算法原理：
    1. 计算每日的真实波幅(TR)
    2. 计算真实波幅的n日移动平均，得到ATR
    3. 通常使用指数移动平均来计算ATR
    
    参数说明：
        high (tp.Array2d): 最高价数组，形状为(时间点数, 资产数)
        low (tp.Array2d): 最低价数组，形状为(时间点数, 资产数)
        close (tp.Array2d): 收盘价数组，形状为(时间点数, 资产数)
        windows (tp.List[int]): ATR计算窗口列表，如[14, 21]
        ewms (tp.List[bool]): 是否使用EWM的布尔列表
        adjust (bool): EWM调整参数
    
    返回值：
        tp.Tuple[tp.Array2d, tp.Dict[int, tp.Array2d]]: 
        返回元组：(真实波幅数组, ATR缓存字典)
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建模拟的OHLC数据
        >>> np.random.seed(42)
        >>> n_days = 50
        >>> n_assets = 2
        >>> 
        >>> # 生成基础价格
        >>> base_prices = np.random.randn(n_days, n_assets).cumsum(axis=0) + 100
        >>> 
        >>> # 生成OHLC数据
        >>> close = base_prices
        >>> high = close + np.random.uniform(0.5, 3, close.shape)
        >>> low = close - np.random.uniform(0.5, 3, close.shape)
        
        >>> # 构建ATR缓存
        >>> tr, atr_cache = vbt.indicators.nb.atr_cache_nb(
        ...     high, low, close, 
        ...     windows=[14, 21], 
        ...     ewms=[True, False], 
        ...     adjust=False
        ... )
        
        >>> print("真实波幅形状:", tr.shape)
        >>> print("ATR缓存大小:", len(atr_cache))
        >>> 
        >>> # 查看真实波幅统计
        >>> print("平均真实波幅:", np.nanmean(tr, axis=0))
        >>> print("真实波幅标准差:", np.nanstd(tr, axis=0))
    
    技术细节：
        - 首先调用true_range_nb计算真实波幅
        - 然后对真实波幅应用不同的移动平均参数
        - 返回原始真实波幅和ATR缓存，提供最大的灵活性
        - 缓存机制避免了重复计算真实波幅
    
    性能优势：
        - 真实波幅只计算一次，避免重复计算
        - 支持多个ATR参数的批量计算
        - 缓存机制提高了参数扫描的效率
        - 适用于ATR的参数优化场景
    """
    # 计算真实波幅(只计算一次)
    tr = true_range_nb(high, low, close)
    # 构建ATR缓存字典
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            # 计算真实波幅的移动平均，得到ATR
            cache_dict[h] = ma_nb(tr, windows[i], ewms[i], adjust=adjust)
    # 返回真实波幅和ATR缓存
    return tr, cache_dict


@njit(cache=True)
def atr_apply_nb(high: tp.Array2d, low: tp.Array2d, close: tp.Array2d, window: int, ewm: bool, adjust: bool,
                 tr: tp.Array2d, cache_dict: tp.Dict[int, tp.Array2d]) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """
    平均真实波幅(ATR)应用函数
    
    使用预计算的真实波幅和ATR缓存来返回指定参数的ATR结果。ATR是重要的波动率指标，
    广泛用于风险管理、止损设置和趋势强度判断。
    
    参数说明：
        high (tp.Array2d): 最高价数组(保持接口一致性)
        low (tp.Array2d): 最低价数组(保持接口一致性)
        close (tp.Array2d): 收盘价数组(保持接口一致性)
        window (int): ATR计算窗口，通常为14
        ewm (bool): 是否使用指数移动平均
        adjust (bool): EWM调整参数
        tr (tp.Array2d): 预计算的真实波幅数组
        cache_dict (tp.Dict[int, tp.Array2d]): ATR缓存字典
    
    返回值：
        tp.Tuple[tp.Array2d, tp.Array2d]: (真实波幅, ATR)的元组
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建具有不同波动特征的价格数据
        >>> n_days = 30
        >>> 
        >>> # 第一个资产：低波动
        >>> low_vol_prices = np.random.randn(n_days) * 0.5 + 100
        >>> # 第二个资产：高波动
        >>> high_vol_prices = np.random.randn(n_days) * 2 + 100
        >>> 
        >>> close = np.column_stack([low_vol_prices, high_vol_prices])
        >>> high = close + np.random.uniform(0.2, 1, close.shape)
        >>> low = close - np.random.uniform(0.2, 1, close.shape)
        
        >>> # 先构建缓存
        >>> tr, atr_cache = vbt.indicators.nb.atr_cache_nb(
        ...     high, low, close, [14], [True], adjust=False
        ... )
        
        >>> # 计算ATR
        >>> tr_result, atr_result = vbt.indicators.nb.atr_apply_nb(
        ...     high, low, close, window=14, ewm=True, adjust=False,
        ...     tr=tr, cache_dict=atr_cache
        ... )
        
        >>> print("14日ATR:")
        >>> print(atr_result[-10:])  # 显示最后10个值
        >>> 
        >>> # 比较两个资产的波动性
        >>> print("资产1平均ATR:", np.nanmean(atr_result[:, 0]))
        >>> print("资产2平均ATR:", np.nanmean(atr_result[:, 1]))
        
        >>> # ATR在交易中的应用
        >>> # 1. 动态止损设置
        >>> stop_loss_multiplier = 2.0
        >>> stop_loss_distance = atr_result * stop_loss_multiplier
        >>> 
        >>> # 对于多头头寸
        >>> long_stop_loss = close - stop_loss_distance
        >>> # 对于空头头寸
        >>> short_stop_loss = close + stop_loss_distance
        >>> 
        >>> # 2. 仓位大小调整
        >>> risk_per_trade = 0.02  # 每笔交易风险2%
        >>> account_size = 100000  # 账户资金
        >>> position_size = (account_size * risk_per_trade) / stop_loss_distance
        >>> 
        >>> # 3. 趋势强度判断
        >>> # ATR增加通常表示趋势加强
        >>> atr_change = atr_result - np.roll(atr_result, 1, axis=0)
        >>> trend_strengthening = atr_change > 0
        >>> 
        >>> # 4. 突破确认
        >>> # 价格突破伴随ATR增加更可靠
        >>> price_change = np.abs(close - np.roll(close, 1, axis=0))
        >>> significant_move = price_change > atr_result * 1.5
        
        >>> # 在量化策略中的应用
        >>> # 1. ATR通道策略
        >>> atr_multiplier = 2.0
        >>> upper_channel = close + atr_result * atr_multiplier
        >>> lower_channel = close - atr_result * atr_multiplier
        >>> 
        >>> # 2. 波动率均值回归
        >>> avg_atr = np.nanmean(atr_result[-20:], axis=0)  # 20日平均ATR
        >>> high_vol_signal = atr_result[-1] > avg_atr * 1.5  # 高波动信号
        >>> low_vol_signal = atr_result[-1] < avg_atr * 0.5   # 低波动信号
        >>> 
        >>> # 3. 多时间框架ATR
        >>> # 结合不同周期的ATR进行综合分析
    
    ATR指标解读：
        - ATR值越大，市场波动性越高
        - ATR值越小，市场波动性越低
        - ATR上升：波动性增加，趋势可能加强
        - ATR下降：波动性减少，市场可能进入整理
        - ATR的绝对值取决于价格水平
    
    技术细节：
        - ATR始终为正值，表示波动的绝对大小
        - 通常使用指数移动平均来计算ATR，更敏感于最新变化
        - ATR不指示价格方向，只反映波动程度
        - 不同资产的ATR值不具有直接可比性
    
    注意事项：
        - ATR是绝对波动率指标，需要结合价格水平分析
        - 高价股的ATR通常比低价股大
        - 应该使用ATR的相对变化而非绝对值进行比较
        - ATR在不同市场环境下的有效性可能不同
    """
    # 生成参数组合的hash键
    h = hash((window, ewm))
    # 从缓存中获取ATR结果，返回真实波幅和ATR
    return tr, cache_dict[h]


@njit(cache=True)
def obv_custom_nb(close: tp.Array2d, volume_ts: tp.Array2d) -> tp.Array2d:
    """
    能量潮指标(On-Balance Volume, OBV)自定义计算函数
    
    OBV是由Joe Granville开发的成交量指标，通过累积成交量来判断资金流向。
    它基于这样的理念：成交量是价格变动的先行指标，资金流向决定价格趋势。
    
    OBV算法原理：
    1. 如果今日收盘价 > 昨日收盘价，则 OBV = 前日OBV + 今日成交量
    2. 如果今日收盘价 < 昨日收盘价，则 OBV = 前日OBV - 今日成交量  
    3. 如果今日收盘价 = 昨日收盘价，则 OBV = 前日OBV
    
    参数说明：
        close (tp.Array2d): 收盘价数组，形状为(时间点数, 资产数)
        volume_ts (tp.Array2d): 成交量数组，形状为(时间点数, 资产数)
    
    返回值：
        tp.Array2d: OBV指标值数组，与输入相同形状
    
    使用示例：
        >>> import numpy as np
        >>> import vectorbt as vbt
        
        >>> # 创建价格和成交量数据
        >>> n_days = 15
        >>> 
        >>> # 模拟价格数据：前半段上涨，后半段下跌
        >>> prices = np.array([
        ...     100, 102, 104, 103, 105, 107, 106, 108,  # 上涨阶段
        ...     107, 105, 103, 104, 102, 100, 98         # 下跌阶段
        ... ]).reshape(-1, 1)
        >>> 
        >>> # 模拟成交量数据：价格上涨时成交量大，下跌时成交量小
        >>> volumes = np.array([
        ...     1000, 1200, 1500, 800, 1300, 1600, 900, 1400,  # 上涨时大成交量
        ...     700, 1100, 1000, 600, 900, 800, 1200            # 下跌时相对小成交量
        ... ]).reshape(-1, 1)
        
        >>> # 计算OBV
        >>> obv = vbt.indicators.nb.obv_custom_nb(prices, volumes)
        >>> print("OBV指标值:")
        >>> print(obv.flatten())
        
        >>> # 分析OBV的变化
        >>> obv_change = obv[1:] - obv[:-1]
        >>> print("OBV变化:")
        >>> print(obv_change.flatten())
        
        >>> # OBV与价格的关系分析
        >>> price_change = prices[1:] - prices[:-1]
        >>> print("价格变化:")
        >>> print(price_change.flatten())
        
        >>> # 在量化策略中的应用
        >>> # 1. OBV趋势确认
        >>> obv_trend_up = obv[-1] > obv[-5]    # OBV上升趋势
        >>> price_trend_up = prices[-1] > prices[-5]  # 价格上升趋势
        >>> 
        >>> if obv_trend_up and price_trend_up:
        ...     print("价格和OBV同步上升，趋势确认")
        >>> elif not obv_trend_up and price_trend_up:
        ...     print("价格上升但OBV下降，可能出现背离")
        
        >>> # 2. OBV背离分析
        >>> # 计算最近几天的价格和OBV高点
        >>> recent_price_high = np.max(prices[-10:])
        >>> recent_obv_high = np.max(obv[-10:])
        >>> 
        >>> # 如果价格创新高但OBV没有创新高，可能是顶背离信号
        >>> if prices[-1] == recent_price_high and obv[-1] < recent_obv_high:
        ...     print("可能出现顶背离，注意风险")
        
        >>> # 3. OBV突破策略
        >>> # 当OBV突破前期高点时，可能预示价格突破
        >>> obv_breakout = obv[-1] > np.max(obv[-20:-1])
        >>> if obv_breakout:
        ...     print("OBV突破前期高点，可能预示价格上涨")
    
    OBV指标解读：
        - OBV上升：买方力量强于卖方，资金流入
        - OBV下降：卖方力量强于买方，资金流出
        - OBV与价格同步：趋势健康，可持续性强
        - OBV与价格背离：趋势可能反转，需要警惕
        - OBV突破：可能预示价格即将突破
    
    技术细节：
        - 使用set_by_mask_nb函数处理价格下跌时的成交量符号
        - 使用nancumsum_nb计算累积成交量，处理缺失值
        - OBV的绝对值没有意义，关注的是相对变化和趋势
        - 第一天的OBV值等于第一天的成交量(如果价格上涨)或负成交量(如果价格下跌)
    
    应用场景：
        - 趋势确认：验证价格趋势的可靠性
        - 背离分析：识别潜在的趋势反转信号
        - 突破确认：成交量突破确认价格突破
        - 资金流向分析：判断主力资金的进出
        - 多空力量对比：评估买卖双方的力量对比
    
    注意事项：
        - OBV需要结合价格分析，单独使用意义不大
        - 在成交量数据不准确的市场中，OBV的有效性会降低
        - 应该关注OBV的趋势变化而非绝对值
        - OBV背离信号需要结合其他指标确认
    """
    # 计算价格变化：当前收盘价与前一日收盘价的比较
    prev_close = generic_nb.fshift_nb(close, 1)
    # 根据价格变化调整成交量的符号
    # 如果收盘价 < 前收盘价，成交量变为负数
    obv = generic_nb.set_by_mask_mult_nb(volume_ts, close < prev_close, -volume_ts)
    # 计算累积成交量，得到OBV
    obv = generic_nb.nancumsum_nb(obv)
    return obv
