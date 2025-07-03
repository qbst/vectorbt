# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
自定义数据类模块 - vectorbt框架的多数据源支持

本模块提供了vectorbt框架中Data基类的多种自定义实现，支持从不同数据源获取金融数据，
包括合成数据生成、传统金融数据提供商以及加密货币交易所的数据接口。

模块设计架构：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                           vectorbt.data.base.Data (基类)
                                        │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
            SyntheticData          │              │         External APIs
         (合成数据生成基类)          │              │        (外部API数据类)
                    │               │              │              │
              GBMData               │              │              │
        (几何布朗运动模拟)            │              │              │
                                   │              │              │
                              YFData              │        ┌─────┴─────┐
                         (Yahoo Finance)          │        │           │
                                                 │    BinanceData  CCXTData
                                                 │   (币安交易所)  (多交易所)
                                                 │              │
                                            AlpacaData         │
                                           (Alpaca券商)    (支持多个
                                                          加密货币交易所)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心功能特性：

1. 合成数据生成
   ├── SyntheticData: 合成数据生成的抽象基类
   └── GBMData: 基于几何布朗运动的股价模拟

2. 传统金融数据源  
   ├── YFData: Yahoo Finance免费股票数据
   └── AlpacaData: Alpaca券商的实时和历史数据

3. 加密货币数据源
   ├── BinanceData: 币安交易所专用接口
   └── CCXTData: 统一的多交易所接口

数据处理流程：
1. 数据获取 → 2. 格式标准化 → 3. 时区处理 → 4. 数据对齐 → 5. 缓存更新

每个数据类都提供：
- download_symbol(): 下载单个符号的数据
- update_symbol(): 增量更新数据
- 自动错误处理和重试机制
- 统一的pandas DataFrame/Series输出格式
- 完整的时区支持

使用场景：
- 量化策略回测：获取历史价格数据用于策略验证
- 实时数据监控：获取最新市场数据用于实时决策
- 数据模拟研究：生成合成数据用于算法研究和压力测试
- 多资产分析：同时处理股票、期货、加密货币等多种资产
- 跨市场套利：比较不同交易所的价格差异

技术特点：
- 异步数据获取：支持并发下载多个符号
- 智能缓存机制：避免重复请求相同数据
- 网络容错处理：自动重试和错误恢复
- 数据质量验证：自动检测和处理异常数据
- 扩展友好设计：易于添加新的数据源
"""

import time
import warnings
from functools import wraps

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from vectorbt import _typing as tp
from vectorbt.data.base import Data
from vectorbt.utils.config import merge_dicts, get_func_kwargs
from vectorbt.utils.datetime_ import (
    get_utc_tz,
    get_local_tz,
    to_tzaware_datetime,
    datetime_to_ms
)

try:
    from binance.client import Client as ClientT
except ImportError:
    ClientT = tp.Any
try:
    from ccxt.base.exchange import Exchange as ExchangeT
except ImportError:
    ExchangeT = tp.Any


class SyntheticData(Data):
    """
    合成数据生成基类 - 用于生成人工合成金融数据的抽象基类
    
    SyntheticData是vectorbt框架中专门用于生成合成金融数据的基类，主要用于：
    1. 量化策略的模拟和测试
    2. 算法性能的压力测试  
    3. 缺失数据的插值和补全
    4. 金融建模和学术研究
    
    设计特点：
    - 抽象基类：定义了合成数据生成的标准接口
    - 时间驱动：基于指定的时间索引生成数据
    - 灵活扩展：支持各种数学模型和随机过程
    - 参数控制：通过kwargs传递生成参数
    
    继承结构：
    SyntheticData (抽象基类)
    └── GBMData (几何布朗运动实现)
    └── 其他自定义模型...
    
    与真实数据的区别：
    - 数据来源：数学模型生成 vs API接口获取
    - 时间控制：完全可控的时间序列 vs 真实市场时间
    - 参数调节：模型参数可调 vs 真实市场数据固定
    - 重现性：可重复生成 vs 历史数据不可重现
    
    使用场景举例：
    ```python
    # 示例1：继承SyntheticData实现自定义模型
    class MeanReversionData(SyntheticData):
        @classmethod
        def generate_symbol(cls, symbol, index, initial_price=100, 
                          mean_revert_speed=0.1, volatility=0.2, **kwargs):
            # 实现均值回归模型的价格生成逻辑
            prices = []
            current_price = initial_price
            for i in range(len(index)):
                # 均值回归公式实现
                drift = mean_revert_speed * (initial_price - current_price)
                shock = np.random.normal(0, volatility)
                current_price += drift + shock
                prices.append(current_price)
            return pd.Series(prices, index=index, name=symbol)
    
    # 示例2：生成测试数据
    synthetic_data = MeanReversionData.download(
        'TEST_STOCK',
        start='2020-01-01',
        end='2023-12-31',
        freq='1D',
        initial_price=100,
        mean_revert_speed=0.05,
        volatility=0.15
    )
    ```
    """

    @classmethod
    def generate_symbol(cls, symbol: tp.Label, index: tp.Index, **kwargs) -> tp.SeriesFrame:
        """
        生成单个符号数据的抽象方法 - 子类必须实现的数据生成接口
        
        这是SyntheticData类的核心抽象方法，定义了合成数据生成的标准接口。
        每个继承SyntheticData的子类都必须实现这个方法来生成具体的数据。
        
        Args:
            symbol (tp.Label): 符号标识符，用于标识和命名生成的数据系列
                例如：'AAPL_SIM', 'BTC_SYNTHETIC', 'TEST_001'等
            index (tp.Index): 时间索引，定义数据生成的时间点序列
                通常是pandas DatetimeIndex，指定数据的时间维度
            **kwargs: 模型特定的生成参数，根据具体模型需要传递不同参数
                常见参数包括：
                - 初始值：initial_price, start_value等
                - 波动性：volatility, sigma等  
                - 趋势参数：drift, mu等
                - 随机种子：seed等
                - 模型参数：根据具体数学模型而定
                
        Returns:
            tp.SeriesFrame: 返回pandas Series或DataFrame，包含生成的合成数据
                - Series：单一时间序列数据（如股价）
                - DataFrame：多列数据（如OHLCV）
                
        实现要求：
        1. 数据长度必须与index长度一致
        2. 返回的数据索引必须与传入的index匹配
        3. 数据类型应该是数值类型（float, int）
        4. 建议处理边界情况（如空索引、无效参数等）
        
        实现示例：
        ```python
        @classmethod
        def generate_symbol(cls, symbol, index, initial_price=100, 
                          volatility=0.2, drift=0.05, seed=None, **kwargs):
            if seed is not None:
                np.random.seed(seed)
                
            # 生成随机游走价格序列
            n_periods = len(index)
            returns = np.random.normal(drift/252, volatility/np.sqrt(252), n_periods)
            prices = [initial_price]
            
            for i in range(1, n_periods):
                prices.append(prices[-1] * (1 + returns[i]))
                
            return pd.Series(prices, index=index, name=symbol)
        ```
        
        注意事项：
        - 子类必须重写此方法，否则调用时会抛出NotImplementedError
        - 建议在实现中添加参数验证和异常处理
        - 可以根据需要返回单列或多列数据
        - 考虑数据的现实合理性（如价格不能为负等）
        """
        raise NotImplementedError

    @classmethod
    def download_symbol(cls,
                        symbol: tp.Label,
                        start: tp.DatetimeLike = 0,
                        end: tp.DatetimeLike = 'now',
                        freq: tp.Union[None, str, pd.DateOffset] = None,
                        date_range_kwargs: tp.KwargsLike = None,
                        **kwargs) -> tp.SeriesFrame:
        """
        下载符号数据方法 - 生成时间索引并调用generate_symbol生成合成数据
        
        这是SyntheticData类对Data基类download_symbol方法的实现，主要功能是：
        1. 根据起止时间和频率生成时间索引
        2. 调用generate_symbol方法生成实际的合成数据
        3. 处理时区转换和参数验证
        
        Args:
            symbol (tp.Label): 符号标识符，传递给generate_symbol方法
            start (tp.DatetimeLike, optional): 数据开始时间，默认为0
                支持的格式：
                - 字符串：'2020-01-01', '2020-01-01 09:30:00'
                - 时间戳：int类型的Unix时间戳
                - datetime对象：datetime.datetime实例
                - 相对时间：'1 year ago', '6 months ago'
            end (tp.DatetimeLike, optional): 数据结束时间，默认为'now'
                支持格式同start参数
            freq (str/pd.DateOffset, optional): 时间频率，默认为None
                常用频率：
                - 'D': 日频率
                - 'H': 小时频率  
                - '15min': 15分钟频率
                - 'B': 工作日频率
                - pd.DateOffset自定义频率
            date_range_kwargs (dict, optional): 传递给pd.date_range的额外参数
                可用参数：
                - periods: 指定期数而非结束时间
                - tz: 时区设置
                - normalize: 是否标准化到午夜
                - name: 索引名称
            **kwargs: 传递给generate_symbol方法的模型参数
            
        Returns:
            tp.SeriesFrame: 生成的合成数据，格式由generate_symbol决定
            
        Raises:
            ValueError: 当生成的时间范围为空时抛出异常
            
        工作流程：
        1. 参数预处理：设置默认的date_range_kwargs
        2. 时区转换：将start和end转换为UTC时区的datetime对象
        3. 时间索引生成：使用pd.date_range生成时间序列
        4. 索引验证：检查生成的索引是否为空
        5. 数据生成：调用generate_symbol生成实际数据
        
        使用示例：
        ```python
        # 示例1：基本用法
        data = MysyntheticData.download_symbol(
            'TEST_SYMBOL',
            start='2020-01-01',
            end='2020-12-31',
            freq='D'
        )
        
        # 示例2：指定期数
        data = MysyntheticData.download_symbol(
            'TEST_SYMBOL',
            start='2020-01-01',
            freq='D',
            date_range_kwargs={'periods': 100}  # 生成100个交易日
        )
        
        # 示例3：包含模型参数
        data = MysyntheticData.download_symbol(
            'GBM_STOCK',
            start='2020-01-01',
            end='2020-12-31',
            freq='D',
            S0=100,      # 初始价格
            mu=0.05,     # 漂移率
            sigma=0.2,   # 波动率
            seed=42      # 随机种子
        )
        ```
        
        注意事项：
        - 时间参数会自动转换为UTC时区
        - 空的时间范围会抛出ValueError异常
        - 所有kwargs会直接传递给generate_symbol方法
        - 生成的数据时间索引与计算的index完全一致
        """
        # 设置默认的date_range参数
        if date_range_kwargs is None:
            date_range_kwargs = {}
        
        # 生成时间索引：使用pd.date_range创建时间序列
        index = pd.date_range(
            start=to_tzaware_datetime(start, tz=get_utc_tz()),    # 开始时间转换为UTC时区
            end=to_tzaware_datetime(end, tz=get_utc_tz()),        # 结束时间转换为UTC时区
            freq=freq,                                            # 时间频率
            **date_range_kwargs                                   # 额外的date_range参数
        )
        
        # 验证时间索引：确保生成的索引不为空
        if len(index) == 0:
            raise ValueError("Date range is empty")
        
        # 调用generate_symbol生成实际的合成数据
        return cls.generate_symbol(symbol, index, **kwargs)

    def update_symbol(self, symbol: tp.Label, **kwargs) -> tp.SeriesFrame:
        """
        更新符号数据方法 - 在现有合成数据基础上生成新的数据点
        
        该方法用于为已存在的合成数据生成新的数据点，通常用于：
        1. 模拟实时数据流
        2. 扩展现有的时间序列
        3. 增量数据生成和测试
        
        Args:
            symbol (tp.Label): 要更新的符号标识符，必须已存在于self.data中
            **kwargs: 更新参数，会与原始下载参数合并
                常用参数：
                - end: 新的结束时间（扩展数据到此时间）
                - freq: 数据频率（通常与原数据保持一致）
                - 其他模型参数会覆盖原有设置
                
        Returns:
            tp.SeriesFrame: 新生成的数据，通常是从最后一个时间点开始的增量数据
            
        工作原理：
        1. 获取符号的原始下载参数
        2. 将开始时间设置为现有数据的最后时间点
        3. 合并原始参数和新传入的参数
        4. 调用download_symbol生成新数据
        
        使用示例：
        ```python
        # 示例1：基本数据更新
        # 假设原数据到2020-12-31，现在要更新到2021-12-31
        gbm_data = vbt.GBMData.download(
            'TEST_GBM',
            start='2020-01-01',
            end='2020-12-31',
            freq='D'
        )
        
        # 更新数据到2021年底
        updated_data = gbm_data.update_symbol(
            'TEST_GBM',
            end='2021-12-31'  # 只需指定新的结束时间
        )
        
        # 示例2：修改模型参数的更新
        updated_data = gbm_data.update_symbol(
            'TEST_GBM',
            end='2021-06-30',
            sigma=0.3,  # 增加波动率
            mu=0.1      # 增加漂移率
        )
        
        # 示例3：实时数据模拟
        import time
        for i in range(10):
            # 每次更新一天的数据
            current_end = pd.Timestamp.now() - pd.Timedelta(days=10-i)
            new_data = gbm_data.update_symbol(
                'TEST_GBM',
                end=current_end
            )
            print(f"更新到: {current_end}, 最新价格: {new_data.iloc[-1]}")
            time.sleep(1)  # 模拟实时更新间隔
        ```
        
        注意事项：
        - 符号必须已存在于当前Data实例中
        - 默认从现有数据的最后时间点开始生成新数据
        - 新参数会覆盖原始下载时的对应参数
        - 生成的是增量数据，需要通过update()方法合并到主数据中
        - 对于确定性模型（有seed），需要注意随机种子的连续性
        """
        # 获取符号的原始下载参数
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        
        # 设置更新的起始时间为现有数据的最后时间点
        download_kwargs['start'] = self.data[symbol].index[-1]
        
        # 合并原始参数和新传入的参数（新参数具有更高优先级）
        kwargs = merge_dicts(download_kwargs, kwargs)
        
        # 调用download_symbol生成新的数据
        return self.download_symbol(symbol, **kwargs)


def generate_gbm_paths(S0: float, mu: float, sigma: float, T: int, M: int, I: int,
                       seed: tp.Optional[int] = None) -> tp.Array2d:
    """
    几何布朗运动路径生成函数 - 基于GBM模型生成多条价格路径
    
    几何布朗运动(Geometric Brownian Motion, GBM)是金融数学中最重要的随机过程之一，
    被广泛用于股票价格建模和期权定价。该函数实现了标准的GBM数值模拟算法。
    
    数学模型：
    dS_t = μ * S_t * dt + σ * S_t * dW_t
    
    其中：
    - S_t: t时刻的资产价格
    - μ: 漂移率(drift rate)，表示预期收益率
    - σ: 波动率(volatility)，表示价格波动的幅度
    - dW_t: 维纳过程(Wiener process)的微分，即布朗运动
    
    离散化公式：
    S_{t+dt} = S_t * exp((μ - 0.5*σ²)*dt + σ*√dt*Z)
    
    其中Z~N(0,1)是标准正态分布的随机变量。
    
    Args:
        S0 (float): 初始价格（t=0时刻的资产价格）
            例如：100.0 表示初始股价为100元
        mu (float): 年化漂移率（无风险利率 + 风险溢价）
            例如：0.05 表示年化预期收益率为5%
            注意：可以为负值，表示预期下跌
        sigma (float): 年化波动率（价格波动的标准差）
            例如：0.2 表示年化波动率为20%
            必须为正值，通常在0.1-0.5之间
        T (int): 总时间长度（单位与mu、sigma保持一致）
            例如：如果mu、sigma是年化的，T=1表示1年
        M (int): 时间步数（将总时间T分割成M个小段）
            时间步长 dt = T/M
            M越大，模拟越精确，但计算量越大
        I (int): 路径数量（需要生成的独立路径数）
            用于蒙特卡洛模拟时的路径数量
        seed (int, optional): 随机种子，用于结果的可重现性
            设置后每次运行会产生相同的随机序列
            
    Returns:
        tp.Array2d: 形状为(M+1, I)的二维numpy数组
            - 行数：M+1 (包含初始时刻，共M+1个时间点)
            - 列数：I (每列代表一条独立的价格路径)
            - paths[0, :] = S0 (所有路径的初始价格都是S0)
            - paths[t, i] 表示第i条路径在第t个时间点的价格
            
    使用示例：
    ```python
    # 示例1：单条路径的股价模拟
    paths = generate_gbm_paths(
        S0=100,      # 初始股价100元
        mu=0.05,     # 年化收益率5%
        sigma=0.2,   # 年化波动率20%
        T=1,         # 模拟1年
        M=252,       # 一年252个交易日
        I=1,         # 生成1条路径
        seed=42      # 设置随机种子
    )
    
    # 提取股价序列（第一条也是唯一的路径）
    stock_prices = paths[:, 0]
    print(f"初始价格: {stock_prices[0]:.2f}")
    print(f"最终价格: {stock_prices[-1]:.2f}")
    
    # 示例2：蒙特卡洛模拟 - 生成多条路径
    paths = generate_gbm_paths(
        S0=100, mu=0.05, sigma=0.2,
        T=1, M=252, I=10000,  # 生成10000条路径
        seed=123
    )
    
    # 分析最终价格分布
    final_prices = paths[-1, :]  # 所有路径的最终价格
    print(f"最终价格均值: {np.mean(final_prices):.2f}")
    print(f"最终价格标准差: {np.std(final_prices):.2f}")
    print(f"理论期望价格: {S0 * np.exp(mu * T):.2f}")
    
    # 示例3：期权定价中的应用
    # 计算欧式看涨期权的蒙特卡洛价格
    strike = 105  # 行权价
    payoffs = np.maximum(final_prices - strike, 0)  # 期权收益
    option_price = np.exp(-0.05 * 1) * np.mean(payoffs)  # 贴现期望收益
    print(f"期权理论价格: {option_price:.2f}")
    ```
    
    数值稳定性注意事项：
    - 当σ很大或dt很大时，exp函数可能产生数值溢出
    - 建议M至少为250（对于年化参数），确保dt足够小
    - 对于高波动率(σ>0.5)，建议增加M值
    
    与其他模型的比较：
    - vs 算术布朗运动：GBM确保价格始终为正
    - vs 均值回归模型：GBM没有长期均值回归特性
    - vs 跳跃扩散模型：GBM没有价格跳跃，路径连续
    
    参考文献：
    - Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities
    - Hull, J. (2017). Options, Futures, and Other Derivatives
    - 实现参考：https://stackoverflow.com/a/45036114/8141780
    """
    # 设置随机种子以确保结果可重现
    if seed is not None:
        np.random.seed(seed)

    # 计算时间步长：将总时间T分割成M个相等的小段
    dt = float(T) / M
    
    # 初始化路径矩阵：(M+1)行（包含t=0），I列（I条独立路径）
    paths = np.zeros((M + 1, I), np.float64)
    
    # 设置所有路径的初始价格为S0
    paths[0] = S0
    
    # 按时间步逐步生成每个时间点的价格
    for t in range(1, M + 1):
        # 生成I个独立的标准正态分布随机数
        rand = np.random.standard_normal(I)
        
        # 应用GBM的离散化公式更新价格
        # 公式：S_{t+dt} = S_t * exp((μ - 0.5*σ²)*dt + σ*√dt*Z)
        paths[t] = paths[t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt    # 漂移项：调整期望收益率
            + sigma * np.sqrt(dt) * rand     # 扩散项：随机波动部分
        )
    
    return paths


class GBMData(SyntheticData):
    """
    几何布朗运动数据类 - 基于GBM模型生成金融时间序列数据
    
    GBMData是SyntheticData的具体实现，专门用于生成基于几何布朗运动模型的金融数据。
    这个类广泛应用于量化金融的多个领域，是期权定价、风险管理和策略回测的重要工具。
    
    核心特性：
    1. 数学严谨性：基于Black-Scholes模型的理论基础
    2. 参数可控性：完全控制价格动态的关键参数
    3. 可重现性：通过随机种子确保结果一致性  
    4. 多路径支持：可同时生成多条独立的价格路径
    5. 实时模拟：支持增量数据更新模拟实时交易
    
    应用场景：
    
    1. 期权定价与风险管理
       - 蒙特卡洛期权定价
       - VaR和CVaR风险度量
       - 希腊字母敏感性分析
       - 压力测试和情景分析
    
    2. 量化策略研究
       - 策略回测的基准数据
       - 算法交易的压力测试
       - 策略参数的敏感性分析
       - 多资产相关性建模
    
    3. 学术研究与教学
       - 金融工程课程的教学演示
       - 研究论文的数值实验
       - 新算法的性能验证
       - 理论模型的实证检验
    
    4. 实际交易应用
       - 高频交易策略的预测
       - 波动率交易的基准
       - 套利策略的机会识别
       - 流动性风险的评估
    
    数学模型详解：
    
    连续时间模型：
    dS_t = μ * S_t * dt + σ * S_t * dW_t
    
    解析解：
    S_t = S_0 * exp((μ - σ²/2) * t + σ * W_t)
    
    其中：
    - S_t: t时刻的资产价格
    - S_0: 初始资产价格
    - μ: 瞬时漂移率（年化收益率）
    - σ: 瞬时波动率（年化标准差）
    - W_t: 标准布朗运动
    - t: 时间（年为单位）
    
    统计性质：
    - E[S_t] = S_0 * exp(μ * t)  （价格期望值）
    - Var[S_t] = S_0² * exp(2μt) * (exp(σ²t) - 1)  （价格方差）
    - ln(S_t/S_0) ~ N((μ - σ²/2)t, σ²t)  （对数收益率分布）
    
    使用示例：
    
    ```python
    import vectorbt as vbt
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 示例1：基本股价模拟
    gbm_data = vbt.GBMData.download(
        'AAPL_SIM',           # 符号名称
        start='2 hours ago',   # 开始时间
        end='now',             # 结束时间  
        freq='1min',           # 1分钟频率
        S0=150.0,              # 初始价格150美元
        mu=0.10,               # 年化收益率10%
        sigma=0.25,            # 年化波动率25%
        seed=42                # 随机种子
    )
    
    # 获取生成的价格序列
    prices = gbm_data.get()
    print(f"初始价格: {prices.iloc[0]:.2f}")
    print(f"最终价格: {prices.iloc[-1]:.2f}")
    print(f"价格变化: {(prices.iloc[-1]/prices.iloc[0]-1)*100:.2f}%")
    
    # 示例2：多路径蒙特卡洛模拟
    gbm_paths = vbt.GBMData.download(
        ['PATH_1', 'PATH_2', 'PATH_3'],  # 多个路径
        start='2023-01-01',
        end='2023-12-31', 
        freq='D',                        # 日频率
        S0=100.0,
        mu=0.08,
        sigma=0.20,
        I=3,                            # 生成3条路径
        seed=123
    )
    
    # 分析路径统计特性
    all_paths = gbm_paths.get()
    print(f"所有路径最终价格均值: {all_paths.iloc[-1].mean():.2f}")
    print(f"理论期望价格: {100 * np.exp(0.08 * 1):.2f}")
    
    # 示例3：实时数据更新模拟
    # 初始化数据
    live_data = vbt.GBMData.download(
        'LIVE_STOCK',
        start='2023-01-01 09:30:00',
        end='2023-01-01 16:00:00',
        freq='1H',
        S0=200.0,
        mu=0.05,
        sigma=0.15
    )
    
    # 模拟实时更新
    import time
    for i in range(3):
        time.sleep(1)  # 模拟等待新数据
        live_data = live_data.update(
            end=f'2023-01-0{2+i} 16:00:00'  # 扩展到下一天
        )
        current_price = live_data.get().iloc[-1]
        print(f"第{i+1}次更新后最新价格: {current_price:.2f}")
    
    # 示例4：期权定价应用
    # 生成大量路径用于蒙特卡洛期权定价
    option_simulation = vbt.GBMData.download(
        'OPTION_UNDERLYING',
        start='2023-01-01',
        end='2023-04-01',  # 3个月期权
        freq='D',
        S0=100.0,          # 当前股价
        mu=0.05,           # 无风险利率
        sigma=0.20,        # 历史波动率
        I=10000,           # 10000条路径
        seed=999
    )
    
    # 计算欧式看涨期权价格
    strike_price = 105      # 行权价
    maturity = 3/12         # 3个月到期
    risk_free_rate = 0.05   # 无风险利率
    
    final_prices = option_simulation.get().iloc[-1]  # 到期日价格
    payoffs = np.maximum(final_prices - strike_price, 0)  # 期权收益
    option_price = np.exp(-risk_free_rate * maturity) * payoffs.mean()
    
    print(f"蒙特卡洛期权价格: {option_price:.4f}")
    
    # 与Black-Scholes公式比较
    from scipy.stats import norm
    d1 = (np.log(100/105) + (0.05 + 0.20**2/2) * maturity) / (0.20 * np.sqrt(maturity))
    d2 = d1 - 0.20 * np.sqrt(maturity)
    bs_price = 100 * norm.cdf(d1) - 105 * np.exp(-0.05 * maturity) * norm.cdf(d2)
    print(f"Black-Scholes理论价格: {bs_price:.4f}")
    print(f"价格差异: {abs(option_price - bs_price):.4f}")
    ```
    
    参数选择指南：
    
    1. 初始价格(S0)：
       - 股票：当前市价（如100-500美元）
       - 指数：当前点位（如3000-5000点）
       - 外汇：当前汇率（如1.1-1.3）
    
    2. 漂移率(μ)：
       - 股票：历史年化收益率（通常0.05-0.15）
       - 无风险资产：国债收益率（通常0.01-0.05）
       - 高风险资产：可能为负值
    
    3. 波动率(σ)：
       - 大盘股：0.15-0.25（年化）
       - 小盘股：0.25-0.40（年化）
       - 加密货币：0.50-1.00（年化）
       - 债券：0.05-0.15（年化）
    
    4. 时间参数：
       - 高频交易：分钟或秒级别
       - 日内交易：分钟到小时级别
       - 中长期投资：日到月级别
    
    注意事项与限制：
    
    1. 模型假设：
       - 收益率服从正态分布（现实中可能有厚尾）
       - 波动率恒定（现实中波动率聚集）
       - 无跳跃（现实中可能有价格跳跃）
       - 流动性无限（现实中存在流动性约束）
    
    2. 数值精度：
       - 时间步长要足够小（建议dt < 1/250）
       - 避免极端参数组合导致数值溢出
       - 长期模拟时考虑累积误差
    
    3. 实际应用：
       - 需要定期校准模型参数
       - 结合其他模型验证结果
       - 考虑市场微观结构影响
    """

    @classmethod
    def generate_symbol(cls,
                        symbol: tp.Label,
                        index: tp.Index,
                        S0: float = 100.,
                        mu: float = 0.,
                        sigma: float = 0.05,
                        T: tp.Optional[int] = None,
                        I: int = 1,
                        seed: tp.Optional[int] = None) -> tp.SeriesFrame:
        """
        生成符号数据的核心方法 - 使用GBM模型生成指定符号的价格序列
        
        这是GBMData类的核心方法，实现了SyntheticData抽象类的generate_symbol接口。
        该方法调用generate_gbm_paths函数生成原始路径数据，然后转换为pandas格式。
        
        Args:
            symbol (tp.Label): 符号标识符，用于标识生成的数据系列
                例如：'AAPL_GBM', 'BTC_SIM', 'SPY_SYNTHETIC'等
                
            index (pd.Index): 时间索引，指定生成数据的时间点序列
                通常是pandas DatetimeIndex，定义价格序列的时间维度
                例如：pd.date_range('2023-01-01', '2023-12-31', freq='D')
                
            S0 (float, optional): 初始价格，默认100.0
                表示t=0时刻的资产价格，作为整个价格路径的起点
                注意：该值不会出现在最终输出数据中（被跳过了第一个点）
                实际意义：前一个交易日的收盘价或开盘基准价
                
            mu (float, optional): 漂移率，默认0.0
                年化预期收益率，表示价格变化的趋势方向
                - 正值：预期价格上涨趋势
                - 负值：预期价格下跌趋势  
                - 零值：无明确趋势的随机游走
                单位：年化比率（如0.05表示5%年化收益率）
                
            sigma (float, optional): 波动率，默认0.05
                年化标准差，表示价格变化的不确定性程度
                值越大，价格波动越剧烈；值越小，价格变化越平稳
                单位：年化标准差（如0.2表示20%年化波动率）
                
            T (int, optional): 时间步数，默认为None
                表示模拟的总时间长度，以步数为单位
                如果为None，则自动设置为index的长度
                与时间索引结合确定时间步长：dt = T / len(index)
                
            I (int, optional): 路径数量，默认1
                需要生成的独立价格路径数量
                - I=1：生成单一价格序列（返回Series）
                - I>1：生成多条独立路径（返回DataFrame）
                用于蒙特卡洛模拟时的路径数量
                
            seed (int, optional): 随机种子，默认None
                用于控制随机数生成，确保结果可重现
                设置相同种子会产生完全相同的价格序列
                None表示使用真随机种子
                
        Returns:
            tp.SeriesFrame: 生成的价格数据，格式取决于路径数量
                - 单路径(I=1)：pandas Series，索引为时间，值为价格
                - 多路径(I>1)：pandas DataFrame，索引为时间，列为不同路径
                
        数据生成流程：
        1. 参数预处理：设置默认的时间步数T
        2. 调用generate_gbm_paths生成原始路径矩阵
        3. 去除第一行（初始价格S0不包含在输出中）
        4. 根据路径数量决定返回格式（Series或DataFrame）
        5. 设置适当的索引和列名
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        
        # 示例1：生成单一股价序列
        index = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        stock_price = vbt.GBMData.generate_symbol(
            symbol='AAPL_SIM',
            index=index,
            S0=150.0,    # 苹果股票起始价格150美元
            mu=0.12,     # 年化收益率12%
            sigma=0.25,  # 年化波动率25%
            seed=42      # 固定随机种子
        )
        
        print(f"数据类型: {type(stock_price)}")  # pandas Series
        print(f"数据长度: {len(stock_price)}")   # 365个交易日
        print(f"初始价格: {stock_price.iloc[0]:.2f}")
        print(f"最终价格: {stock_price.iloc[-1]:.2f}")
        
        # 示例2：生成多条路径用于蒙特卡洛模拟
        multi_paths = vbt.GBMData.generate_symbol(
            symbol='MC_SIMULATION',
            index=index,
            S0=100.0,
            mu=0.08,
            sigma=0.20,
            I=1000,      # 生成1000条路径
            seed=123
        )
        
        print(f"数据类型: {type(multi_paths)}")     # pandas DataFrame
        print(f"数据形状: {multi_paths.shape}")     # (365, 1000)
        print(f"列名: {multi_paths.columns.name}")  # 'path'
        
        # 分析最终价格分布
        final_prices = multi_paths.iloc[-1]
        print(f"最终价格均值: {final_prices.mean():.2f}")
        print(f"最终价格标准差: {final_prices.std():.2f}")
        
        # 示例3：短期高频数据生成
        intraday_index = pd.date_range(
            '2023-01-01 09:30:00', 
            '2023-01-01 16:00:00', 
            freq='1min'
        )
        
        hft_data = vbt.GBMData.generate_symbol(
            symbol='HFT_STOCK',
            index=intraday_index,
            S0=200.0,
            mu=0.00,     # 日内无明显趋势
            sigma=0.30,  # 较高的短期波动率  
            T=1,         # 时间单位为天
            seed=456
        )
        
        print(f"日内数据点数: {len(hft_data)}")
        print(f"价格变化范围: {hft_data.min():.2f} - {hft_data.max():.2f}")
        ```
        
        技术细节说明：
        
        1. 时间步长计算：
           dt = T / len(index)
           这意味着如果T=1（1年）且index有252个交易日，
           则dt=1/252≈0.004（约1.5个交易日）
           
        2. 初始价格处理：
           generate_gbm_paths返回(M+1, I)的数组，第一行是S0
           通过[1:]切片去除第一行，使输出长度与index一致
           
        3. 数据格式转换：
           - 单路径：转换为Series，name为symbol
           - 多路径：转换为DataFrame，列名为'path_0', 'path_1'等
           
        4. 随机性控制：
           种子设置在generate_gbm_paths函数内部
           每次调用都会重新设置种子（如果提供）
           
        注意事项：
        - S0不出现在最终数据中，它仅作为生成算法的起点
        - 时间索引必须与生成的数据长度完全匹配
        - 多路径情况下，所有路径使用相同参数但独立随机过程
        - 波动率和漂移率应与时间单位保持一致（通常年化）
        """
        # 设置默认时间步数：如果未指定T，则使用索引长度
        if T is None:
            T = len(index)
            
        # 生成GBM路径：调用核心算法函数
        # 返回形状为(M+1, I)的数组，第一行为初始价格S0
        out = generate_gbm_paths(S0, mu, sigma, T, len(index), I, seed=seed)[1:]
        
        # 根据路径数量决定返回格式
        if out.shape[1] == 1:
            # 单路径：返回Series格式
            return pd.Series(out[:, 0], index=index)
        
        # 多路径：返回DataFrame格式
        # 创建列索引，命名为'path'，列标签为0, 1, 2, ...
        columns = pd.RangeIndex(stop=out.shape[1], name='path')
        return pd.DataFrame(out, index=index, columns=columns)

    def update_symbol(self, symbol: tp.Label, **kwargs) -> tp.SeriesFrame:
        """
        更新GBM符号数据方法 - 基于现有数据的最后价格继续生成新的GBM序列
        
        该方法专门为GBM数据实现了智能的增量更新逻辑，确保新生成的数据
        与现有数据在价格连续性上保持一致，避免出现价格跳跃。
        
        Args:
            symbol (tp.Label): 要更新的符号标识符
            **kwargs: 更新参数，会与原始下载参数合并
                常用参数包括：
                - end: 新的结束时间
                - mu: 新的漂移率（可以调整未来预期）
                - sigma: 新的波动率（可以调整波动程度）
                - 其他generate_symbol支持的参数
                
        Returns:
            tp.SeriesFrame: 新生成的增量数据
            
        核心改进特性：
        1. 价格连续性：使用现有数据的倒数第二个价格作为新的S0
        2. 时间连续性：自动从现有数据的最后时间点开始
        3. 随机性重置：清除原始随机种子，避免重复模式
        4. 参数灵活性：支持修改模型参数以适应市场变化
        
        与基类update_symbol的区别：
        - 智能S0设置：使用倒数第二个价格而非最后一个价格
        - 参数清理：自动移除不适用的T和seed参数
        - 连续性保证：确保新旧数据的平滑连接
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 示例1：基本数据更新
        # 初始生成2023年上半年的数据
        gbm_data = vbt.GBMData.download(
            'TSLA_SIM',
            start='2023-01-01',
            end='2023-06-30',
            freq='D',
            S0=200.0,
            mu=0.15,
            sigma=0.40,
            seed=42
        )
        
        print(f"上半年最后价格: {gbm_data.get().iloc[-1]:.2f}")
        
        # 更新到2023年底，使用相同参数
        updated_data = gbm_data.update_symbol(
            'TSLA_SIM',
            end='2023-12-31'
        )
        
        print(f"下半年第一个价格: {updated_data.iloc[0]:.2f}")
        print(f"价格连续性检查: 连续" if abs(gbm_data.get().iloc[-1] - updated_data.iloc[0]) < 0.01 else "不连续")
        
        # 示例2：调整模型参数的更新
        # 假设下半年市场环境变化，需要调整参数
        adjusted_update = gbm_data.update_symbol(
            'TSLA_SIM',
            end='2023-12-31',
            mu=0.05,     # 降低预期收益率
            sigma=0.60   # 增加波动率
        )
        
        print(f"调整参数后的数据长度: {len(adjusted_update)}")
        
        # 示例3：实时交易模拟
        # 模拟盘中实时更新
        live_gbm = vbt.GBMData.download(
            'LIVE_STOCK',
            start='2023-12-01 09:30:00',
            end='2023-12-01 12:00:00',
            freq='5min',
            S0=100.0,
            mu=0.0,      # 盘中无明显趋势
            sigma=0.25
        )
        
        # 模拟午后交易时段
        afternoon_data = live_gbm.update_symbol(
            'LIVE_STOCK', 
            end='2023-12-01 16:00:00',
            sigma=0.35   # 午后波动率可能增加
        )
        
        print(f"上午结束价格: {live_gbm.get().iloc[-1]:.2f}")
        print(f"午后开始价格: {afternoon_data.iloc[0]:.2f}")
        ```
        
        技术实现细节：
        
        1. S0智能设置逻辑：
           ```python
           # 不使用最后一个价格：self.data[symbol].iloc[-1]
           # 而使用倒数第二个价格：self.data[symbol].iloc[-2]
           ```
           原因：避免数据重叠和价格跳跃，保持平滑过渡
           
        2. 参数清理机制：
           - 移除原始S0：因为需要重新计算
           - 移除原始T：因为时间范围发生变化
           - 清空seed：避免重复随机模式
           
        3. 参数合并优先级：
           新传入的kwargs > 处理后的download_kwargs > 默认值
           
        注意事项：
        - 确保现有数据至少有2个数据点（用于获取倒数第二个价格）
        - 新的时间范围应该从现有数据的最后时间点开始
        - 参数修改会影响整个新生成的序列
        - 建议在市场环境显著变化时调整mu和sigma参数
        """
        # 获取该符号的原始下载参数
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        
        # 设置更新的起始时间为现有数据的最后时间点
        download_kwargs['start'] = self.data[symbol].index[-1]
        
        # 移除原始S0参数，因为需要基于现有数据重新设置
        _ = download_kwargs.pop('S0', None)
        
        # 智能设置新的S0：使用倒数第二个价格，确保价格连续性
        # 这样可以避免价格跳跃，保持数据的平滑过渡
        S0 = self.data[symbol].iloc[-2]
        
        # 移除原始T参数，因为时间范围发生了变化
        _ = download_kwargs.pop('T', None)
        
        # 清空随机种子，避免重复相同的随机模式
        # 这样每次更新都会产生新的随机序列
        download_kwargs['seed'] = None
        
        # 合并处理后的原始参数和新传入的参数
        kwargs = merge_dicts(download_kwargs, kwargs)
        
        # 调用download_symbol生成新数据，使用计算得到的S0
        return self.download_symbol(symbol, S0=S0, **kwargs)


class YFData(Data):
    """
    Yahoo Finance数据类 - 基于yfinance库获取免费金融数据
    
    YFData是vectorbt框架中最常用的真实市场数据源之一，通过Yahoo Finance的免费API
    提供股票、ETF、期货、外汇和加密货币的历史和近实时数据。这是量化分析入门的理想选择。
    
    数据源特点：
    1. 免费访问：无需API密钥，完全免费使用
    2. 广泛覆盖：全球主要市场的股票、ETF、指数、商品
    3. 多时间框架：从1分钟到月度的各种时间间隔
    4. 实时性：提供15-20分钟延迟的准实时数据
    5. 历史数据：可追溯数十年的历史价格数据
    
    支持的资产类型：
    - 股票：AAPL, GOOGL, MSFT等美股和国际股票
    - ETF：SPY, QQQ, ARKK等交易所交易基金
    - 指数：^GSPC (S&P 500), ^IXIC (NASDAQ)等
    - 加密货币：BTC-USD, ETH-USD等主流数字货币
    - 外汇：EURUSD=X, GBPUSD=X等货币对
    - 商品：GC=F (黄金), CL=F (原油)等期货
    
    数据格式：
    返回标准的OHLCV数据，包含以下列：
    - Open: 开盘价
    - High: 最高价  
    - Low: 最低价
    - Close: 收盘价
    - Volume: 成交量
    - Dividends: 股息（如适用）
    - Stock Splits: 股票拆分（如适用）
    
    时区处理：
    - 股票：通常使用市场本地时区（如美股为EST/EDT）
    - 加密货币：通常使用UTC时区
    - 自动转换：vectorbt会自动处理时区标准化
    
    数据质量注意事项：
    ⚠️ 重要警告：
    Yahoo Finance的数据仅用于教育和研究目的，不建议用于实际交易决策，原因：
    1. 数据可能被操纵或添加噪音
    2. 可能存在缺失数据点（特别是成交量数据）
    3. 历史数据可能被事后调整
    4. 服务稳定性无法保证
    5. 精度可能不足以支持高频交易策略
    
    使用场景：
    1. 学习和教育：量化金融课程和教学演示
    2. 策略开发：初步策略测试和概念验证
    3. 回测研究：长期历史数据的趋势分析
    4. 市场监控：个人投资的市场观察
    5. 原型开发：快速搭建量化交易原型
    
    使用示例：
    ```python
    import vectorbt as vbt
    
    # 示例1：获取单只股票的日线数据
    aapl_data = vbt.YFData.download(
        "AAPL",                    # 苹果股票
        start='2022-01-01',        # 开始日期
        end='2023-12-31',          # 结束日期
        interval='1d'              # 日线数据
    )
    
    # 获取收盘价序列
    aapl_close = aapl_data.get('Close')
    print(f"数据点数: {len(aapl_close)}")
    print(f"价格范围: ${aapl_close.min():.2f} - ${aapl_close.max():.2f}")
    
    # 示例2：获取多只股票的数据进行比较
    tech_stocks = vbt.YFData.download(
        ["AAPL", "GOOGL", "MSFT", "AMZN"],  # 科技巨头股票
        period='2y',                         # 最近2年数据
        interval='1d'
    )
    
    # 获取所有股票的收盘价
    tech_closes = tech_stocks.get('Close')
    print(f"股票数量: {len(tech_closes.columns)}")
    print(f"数据形状: {tech_closes.shape}")
    
    # 计算相对表现（标准化到起始价格）
    normalized_prices = tech_closes / tech_closes.iloc[0] * 100
    print("相对表现（%）:")
    print(normalized_prices.iloc[-1].round(2))
    
    # 示例3：获取高频数据（分钟级）
    spy_intraday = vbt.YFData.download(
        "SPY",                           # S&P 500 ETF
        start='2023-12-01 09:30:00',     # 交易日开盘
        end='2023-12-01 16:00:00',       # 交易日收盘
        interval='1m'                    # 1分钟数据
    )
    
    # 分析日内波动
    spy_prices = spy_intraday.get('Close')
    daily_return = (spy_prices.iloc[-1] / spy_prices.iloc[0] - 1) * 100
    daily_volatility = spy_prices.pct_change().std() * 100
    print(f"日内收益率: {daily_return:.2f}%")
    print(f"日内波动率: {daily_volatility:.2f}%")
    
    # 示例4：加密货币数据
    crypto_data = vbt.YFData.download(
        ["BTC-USD", "ETH-USD"],          # 比特币和以太坊
        period='1y',                     # 最近1年
        interval='1d'
    )
    
    # 分析加密货币相关性
    crypto_returns = crypto_data.get('Close').pct_change().dropna()
    correlation = crypto_returns.corr()
    print("加密货币相关性矩阵:")
    print(correlation.round(3))
    
    # 示例5：实时数据更新模拟
    # 获取最近的数据
    live_data = vbt.YFData.download(
        "TSLA",
        period='5d',                     # 最近5天
        interval='1h'                    # 小时数据
    )
    
    print(f"最新价格: ${live_data.get('Close').iloc[-1]:.2f}")
    
    # 更新到更近的时间点
    updated_data = live_data.update(period='3d')
    print(f"更新后数据点数: {len(updated_data.get('Close'))}")
    
    # 示例6：指数和ETF数据
    market_data = vbt.YFData.download(
        {
            '^GSPC': 'S&P 500 Index',    # 标普500指数
            'SPY': 'S&P 500 ETF',        # 标普500 ETF
            '^VIX': 'VIX Index'          # 恐慌指数
        },
        period='1y',
        interval='1d'
    )
    
    # 分析市场情绪
    sp500_returns = market_data.get('Close')['^GSPC'].pct_change()
    vix_levels = market_data.get('Close')['^VIX']
    
    # 计算VIX与收益率的相关性
    fear_correlation = sp500_returns.corr(vix_levels.pct_change())
    print(f"恐慌指数与市场相关性: {fear_correlation:.3f}")
    ```
    
    最佳实践：
    
    1. 数据验证：
       - 检查缺失数据点
       - 验证异常价格波动
       - 对比其他数据源确认关键数据点
    
    2. 时间处理：
       - 注意不同市场的交易时间
       - 考虑节假日和停牌的影响
       - 正确处理时区转换
    
    3. 性能优化：
       - 避免频繁的小批量请求
       - 缓存数据减少重复下载
       - 使用合适的时间间隔
    
    4. 风险管理：
       - 不依赖于Yahoo数据做实盘交易
       - 对关键数据进行多源验证
       - 建立数据质量监控机制
    
    替代方案：
    对于专业应用，建议考虑：
    - Alpha Vantage (免费额度有限)
    - IEX Cloud (付费但稳定)
    - Quandl (高质量金融数据)
    - 专业数据提供商 (Bloomberg, Refinitiv)
    """

    @classmethod
    def download_symbol(cls,
                        symbol: tp.Label,
                        period: str = 'max',
                        start: tp.Optional[tp.DatetimeLike] = None,
                        end: tp.Optional[tp.DatetimeLike] = None,
                        ticker_kwargs: tp.KwargsLike = None,
                        **kwargs) -> tp.Frame:
        """
        下载单个符号的Yahoo Finance数据
        
        该方法通过yfinance库从Yahoo Finance获取指定符号的历史股价数据，
        支持灵活的时间范围设置和参数配置。
        
        Args:
            symbol (str): 股票符号代码
            period (str): 预设时间周期，默认'max'
            start (any): 开始时间
            end (any): 结束时间  
            ticker_kwargs (dict): 传递给yfinance.Ticker的参数
            **kwargs: 传递给yfinance历史数据方法的参数
            
        Returns:
            tp.Frame: 包含OHLCV数据的DataFrame
        """
        # 导入yfinance库用于数据获取
        import yfinance as yf

        # yfinance仍然使用mktime，它假设传入的日期是本地时间
        # 因此需要转换为本地时区的时间格式
        if start is not None:
            # 将开始时间转换为本地时区的时间格式
            start = to_tzaware_datetime(start, tz=get_local_tz())
        if end is not None:
            # 将结束时间转换为本地时区的时间格式
            end = to_tzaware_datetime(end, tz=get_local_tz())

        # 设置默认的ticker参数为空字典
        if ticker_kwargs is None:
            ticker_kwargs = {}
        
        # 创建yfinance的Ticker对象并获取历史数据
        # 调用history方法获取指定时间范围的股价数据
        return yf.Ticker(symbol, **ticker_kwargs).history(period=period, start=start, end=end, **kwargs)

    def update_symbol(self, symbol: tp.Label, **kwargs) -> tp.Frame:
        """
        更新指定符号的数据
        
        该方法用于获取现有数据之后的增量数据，实现数据的实时更新。
        
        Args:
            symbol (tp.Label): 要更新的符号
            **kwargs: 额外参数，会覆盖原始下载参数
            
        Returns:
            tp.Frame: 新获取的增量数据
        """
        # 获取该符号原始的下载参数配置
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        
        # 将开始时间设置为现有数据的最后一个时间点
        # 这样可以获取从上次更新后的新数据
        download_kwargs['start'] = self.data[symbol].index[-1]
        
        # 合并原始参数和新传入的参数，新参数优先级更高
        kwargs = merge_dicts(download_kwargs, kwargs)
        
        # 调用download_symbol方法获取新的数据
        return self.download_symbol(symbol, **kwargs)


BinanceDataT = tp.TypeVar("BinanceDataT", bound="BinanceData")


class BinanceData(Data):
    """
    币安交易所数据类 - 基于python-binance库获取加密货币交易数据
    
    BinanceData专门用于从全球最大的加密货币交易所Binance获取实时和历史数据。
    币安提供高质量、高频率的加密货币数据，是加密货币量化交易的首选数据源。
    
    主要特点：
    1. 高质量数据：直接来自交易所，数据准确可靠
    2. 实时性强：提供近实时的市场数据
    3. 多时间框架：支持1分钟到1月的各种K线间隔
    4. 丰富信息：包含成交量、交易数量等详细市场数据
    5. 免费使用：无需API密钥即可获取公开数据
    
    支持的交易对：
    - 主流币种：BTC/USDT, ETH/USDT, BNB/USDT等
    - 山寨币：ADA/USDT, DOT/USDT, LINK/USDT等  
    - 稳定币对：BTC/BUSD, ETH/BUSD等
    - 法币对：BTC/EUR, ETH/GBP等
    
    数据格式：
    返回详细的K线数据，包含以下字段：
    - Open: 开盘价
    - High: 最高价
    - Low: 最低价  
    - Close: 收盘价
    - Volume: 基础资产成交量
    - Close time: 收盘时间
    - Quote volume: 计价资产成交量
    - Number of trades: 交易笔数
    - Taker base volume: 主动买入基础资产量
    - Taker quote volume: 主动买入计价资产量
    
    使用场景：
    1. 加密货币量化交易策略开发
    2. 数字资产投资组合管理
    3. 加密货币市场分析和研究
    4. 套利策略和高频交易
    5. 区块链金融产品开发
    
    使用示例：
    ```python
    # 获取比特币1分钟K线数据
    btc_data = vbt.BinanceData.download(
        "BTCUSDT",              # 比特币对USDT交易对
        start='2 hours ago UTC', # 2小时前开始
        end='now UTC',          # 到现在为止
        interval='1m'           # 1分钟K线
    )
    
    # 获取价格数据
    btc_prices = btc_data.get('Close')
    print(f"最新BTC价格: ${btc_prices.iloc[-1]:,.2f}")
    
    # 实时更新数据
    updated_data = btc_data.update()
    print(f"更新后数据点数: {len(updated_data.get())}")
    ```
    
    注意事项：
    - 需要安装python-binance库
    - 数据请求有频率限制，建议适当延迟
    - 时间均为UTC时区
    - 支持无API密钥的公开数据访问
    """

    @classmethod
    def download(cls: tp.Type[BinanceDataT],
                 symbols: tp.Labels,
                 client: tp.Optional["ClientT"] = None,
                 **kwargs) -> BinanceDataT:
        """
        覆盖基类的download方法以自动实例化Binance客户端
        
        该方法自动处理Binance客户端的创建和配置，简化用户的使用过程。
        
        Args:
            symbols: 要下载的符号列表
            client: 可选的Binance客户端实例
            **kwargs: 其他下载参数
            
        Returns:
            BinanceDataT: 包含下载数据的BinanceData实例
        """
        # 导入必要的模块
        from binance.client import Client
        from vectorbt._settings import settings
        
        # 获取binance相关的配置设置
        binance_cfg = settings['data']['binance']

        # 初始化客户端参数字典
        client_kwargs = dict()
        
        # 遍历Client构造函数的所有参数
        for k in get_func_kwargs(Client):
            # 如果kwargs中包含Client的参数，则提取到client_kwargs中
            if k in kwargs:
                client_kwargs[k] = kwargs.pop(k)
        
        # 合并默认配置和用户提供的客户端参数
        client_kwargs = merge_dicts(binance_cfg, client_kwargs)
        
        # 如果没有提供client实例，则创建一个新的客户端
        if client is None:
            client = Client(**client_kwargs)
        
        # 调用父类的download方法，传入创建的client
        return super(BinanceData, cls).download(symbols, client=client, **kwargs)

    @classmethod
    def download_symbol(cls,
                        symbol: str,
                        client: tp.Optional["ClientT"] = None,
                        interval: str = '1d',
                        start: tp.DatetimeLike = 0,
                        end: tp.DatetimeLike = 'now UTC',
                        delay: tp.Optional[float] = 500,
                        limit: int = 500,
                        show_progress: bool = True,
                        tqdm_kwargs: tp.KwargsLike = None) -> tp.Frame:
        """Download the symbol.

        Args:
            symbol (str): Symbol.
            client (binance.client.Client): Binance client of type `binance.client.Client`.
            interval (str): Kline interval.

                See `binance.enums`.
            start (any): Start datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            delay (float): Time to sleep after each request (in milliseconds).
            limit (int): The maximum number of returned items.
            show_progress (bool): Whether to show the progress bar.
            tqdm_kwargs (dict): Keyword arguments passed to `tqdm`.

        For defaults, see `data.binance` in `vectorbt._settings.settings`.
        """
        # 验证客户端是否已提供，必须有客户端才能访问API
        if client is None:
            raise ValueError("client must be provided")

        # 设置默认的进度条参数
        if tqdm_kwargs is None:
            tqdm_kwargs = {}
            
        # 建立时间戳范围 - 将开始时间转换为毫秒时间戳
        start_ts = datetime_to_ms(to_tzaware_datetime(start, tz=get_utc_tz()))
        
        try:
            # 尝试获取该符号的第一条可用数据，用于确定数据起始时间
            first_data = client.get_klines(
                symbol=symbol,       # 交易对符号
                interval=interval,   # K线间隔
                limit=1,            # 只获取1条数据
                startTime=0,        # 从最早时间开始
                endTime=None        # 不设置结束时间
            )
            # 获取第一条有效数据的时间戳
            first_valid_ts = first_data[0][0]
            # 取用户指定开始时间和数据实际开始时间的较大值
            next_start_ts = start_ts = max(start_ts, first_valid_ts)
        except:
            # 如果获取失败，使用用户指定的开始时间
            next_start_ts = start_ts
            
        # 将结束时间转换为毫秒时间戳
        end_ts = datetime_to_ms(to_tzaware_datetime(end, tz=get_utc_tz()))

        # 定义时间戳转字符串的辅助函数，用于显示进度信息
        def _ts_to_str(ts: tp.DatetimeLike) -> str:
            return str(pd.Timestamp(to_tzaware_datetime(ts, tz=get_utc_tz())))

        # 迭代收集数据 - 初始化数据存储列表
        data: tp.List[list] = []
        
        # 使用进度条显示下载进度
        with tqdm(disable=not show_progress, **tqdm_kwargs) as pbar:
            # 设置进度条初始描述为开始时间
            pbar.set_description(_ts_to_str(start_ts))
            
            # 开始数据获取循环
            while True:
                # 获取下一个时间间隔的K线数据
                next_data = client.get_klines(
                    symbol=symbol,              # 交易对符号
                    interval=interval,          # K线时间间隔
                    limit=limit,               # 单次请求的最大条数
                    startTime=next_start_ts,   # 开始时间戳
                    endTime=end_ts             # 结束时间戳
                )
                
                # 根据是否已有数据来过滤新获取的数据
                if len(data) > 0:
                    # 如果已有数据，过滤掉重复和超出范围的数据
                    next_data = list(filter(lambda d: next_start_ts < d[0] < end_ts, next_data))
                else:
                    # 如果是第一次获取，只过滤超出结束时间的数据
                    next_data = list(filter(lambda d: d[0] < end_ts, next_data))

                # 更新时间戳和进度条显示
                if not len(next_data):
                    # 如果没有新数据，跳出循环
                    break
                    
                # 将新数据添加到主数据列表中
                data += next_data
                
                # 更新进度条描述，显示时间范围
                pbar.set_description("{} - {}".format(
                    _ts_to_str(start_ts),          # 开始时间
                    _ts_to_str(next_data[-1][0])   # 当前最新时间
                ))
                
                # 更新进度条
                pbar.update(1)
                
                # 设置下次请求的开始时间为当前批次的最后时间
                next_start_ts = next_data[-1][0]
                
                # 如果设置了延迟，则等待指定时间（对API友好）
                if delay is not None:
                    time.sleep(delay / 1000)  # 延迟单位从毫秒转换为秒

        # 将原始数据转换为DataFrame格式
        # 创建DataFrame并指定列名，对应Binance API返回的数据结构
        df = pd.DataFrame(data, columns=[
            'Open time',         # 开盘时间（毫秒时间戳）
            'Open',              # 开盘价
            'High',              # 最高价
            'Low',               # 最低价
            'Close',             # 收盘价
            'Volume',            # 基础资产成交量
            'Close time',        # 收盘时间（毫秒时间戳）
            'Quote volume',      # 计价资产成交量
            'Number of trades',  # 交易笔数
            'Taker base volume', # 主动买入的基础资产量
            'Taker quote volume', # 主动买入的计价资产量
            'Ignore'             # 忽略字段（API返回但不使用）
        ])
        
        # 设置DataFrame的索引为开盘时间
        # 将毫秒时间戳转换为pandas时间格式，并设置为UTC时区
        df.index = pd.to_datetime(df['Open time'], unit='ms', utc=True)
        
        # 删除原始的开盘时间列，因为已经设置为索引
        del df['Open time']
        
        # 数据类型转换：将字符串格式的价格数据转换为浮点数
        df['Open'] = df['Open'].astype(float)                    # 开盘价转浮点数
        df['High'] = df['High'].astype(float)                    # 最高价转浮点数
        df['Low'] = df['Low'].astype(float)                      # 最低价转浮点数
        df['Close'] = df['Close'].astype(float)                  # 收盘价转浮点数
        df['Volume'] = df['Volume'].astype(float)                # 成交量转浮点数
        
        # 收盘时间也转换为pandas时间格式
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms', utc=True)
        
        # 其他数值字段的类型转换
        df['Quote volume'] = df['Quote volume'].astype(float)     # 计价资产成交量转浮点数
        df['Number of trades'] = df['Number of trades'].astype(int)  # 交易笔数转整数
        df['Taker base volume'] = df['Taker base volume'].astype(float)  # 主动买入基础资产量转浮点数
        df['Taker quote volume'] = df['Taker quote volume'].astype(float) # 主动买入计价资产量转浮点数
        
        # 删除不需要的忽略字段
        del df['Ignore']

        # 返回处理完成的DataFrame
        return df

    def update_symbol(self, symbol: str, **kwargs) -> tp.Frame:
        """
        更新指定符号的Binance数据
        
        该方法获取现有数据之后的新数据，用于实现数据的增量更新。
        
        Args:
            symbol (str): 要更新的交易对符号
            **kwargs: 额外参数，会覆盖原始下载参数
            
        Returns:
            tp.Frame: 新获取的增量数据
        """
        # 获取该符号的原始下载参数配置
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        
        # 将开始时间设置为现有数据的最后一个时间点
        # 确保获取的是最新的增量数据，避免重复
        download_kwargs['start'] = self.data[symbol].index[-1]
        
        # 关闭进度条显示，因为更新操作通常是后台进行
        download_kwargs['show_progress'] = False
        
        # 合并原始参数和新传入的参数，新参数具有更高优先级
        kwargs = merge_dicts(download_kwargs, kwargs)
        
        # 调用download_symbol方法获取新的数据
        return self.download_symbol(symbol, **kwargs)


class CCXTData(Data):
    """
    CCXT多交易所数据类 - 基于ccxt库统一访问多个加密货币交易所
    
    CCXTData通过CCXT(CryptoCurrency eXchange Trading)库提供统一的接口，
    支持超过120个加密货币交易所的数据获取，是跨交易所分析的理想工具。
    
    主要优势：
    1. 多交易所支持：统一接口访问Binance、Coinbase、Kraken等主流交易所
    2. 标准化数据：自动处理不同交易所的数据格式差异
    3. 简化接入：无需学习每个交易所的独特API
    4. 套利支持：便于跨交易所价格比较和套利策略
    5. 容错处理：内建重试机制和错误处理
    
    支持的交易所（部分）：
    - 中心化交易所：binance, coinbase, kraken, bitfinex, huobi等
    - 去中心化交易所：部分支持DEX
    - 衍生品交易所：bitmex, okex, ftx等
    
    数据格式：
    返回标准OHLCV格式：
    - Open: 开盘价
    - High: 最高价
    - Low: 最低价
    - Close: 收盘价
    - Volume: 成交量
    
    使用场景：
    1. 跨交易所套利策略开发
    2. 多交易所数据聚合分析
    3. 交易所流动性比较研究
    4. 统一的加密货币数据获取接口
    
    使用示例：
    ```python
    # 从Binance获取BTC/USDT数据
    btc_data = vbt.CCXTData.download(
        "BTC/USDT",
        exchange='binance',      # 指定交易所
        start='2 hours ago UTC',
        end='now UTC',
        timeframe='1m'
    )
    
    # 比较不同交易所的价格
    exchanges = ['binance', 'coinbase', 'kraken']
    for exchange in exchanges:
        data = vbt.CCXTData.download(
            "BTC/USD", 
            exchange=exchange,
            period='1d'
        )
        price = data.get('Close').iloc[-1]
        print(f"{exchange} BTC价格: ${price:,.2f}")
    ```
    
    注意事项：
    - 需要安装ccxt库
    - 不同交易所的符号格式可能不同
    - 部分交易所可能需要API密钥
    - 请求频率限制因交易所而异
    """

    @classmethod
    def download_symbol(cls,
                        symbol: str,
                        exchange: tp.Union[str, "ExchangeT"] = 'binance',
                        config: tp.Optional[dict] = None,
                        timeframe: str = '1d',
                        start: tp.DatetimeLike = 0,
                        end: tp.DatetimeLike = 'now UTC',
                        delay: tp.Optional[float] = None,
                        limit: tp.Optional[int] = 500,
                        retries: int = 3,
                        show_progress: bool = True,
                        params: tp.Optional[dict] = None,
                        tqdm_kwargs: tp.KwargsLike = None) -> tp.Frame:
        """Download the symbol.

        Args:
            symbol (str): Symbol.
            exchange (str or object): Exchange identifier or an exchange object of type
                `ccxt.base.exchange.Exchange`.
            config (dict): Config passed to the exchange upon instantiation.

                Will raise an exception if exchange has been already instantiated.
            timeframe (str): Timeframe supported by the exchange.
            start (any): Start datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            delay (float): Time to sleep after each request (in milliseconds).

                !!! note
                    Use only if `enableRateLimit` is not set.
            limit (int): The maximum number of returned items.
            retries (int): The number of retries on failure to fetch data.
            show_progress (bool): Whether to show the progress bar.
            tqdm_kwargs (dict): Keyword arguments passed to `tqdm`.
            params (dict): Exchange-specific key-value parameters.

        For defaults, see `data.ccxt` in `vectorbt._settings.settings`.
        """
        # 导入必要的模块
        import ccxt
        from vectorbt._settings import settings
        
        # 获取CCXT相关的配置设置
        ccxt_cfg = settings['data']['ccxt']

        # 初始化默认参数
        if config is None:
            config = {}
        if tqdm_kwargs is None:
            tqdm_kwargs = {}
        if params is None:
            params = {}
            
        # 处理交易所实例化逻辑
        if isinstance(exchange, str):
            # 如果exchange是字符串，需要创建交易所实例
            if not hasattr(ccxt, exchange):
                raise ValueError(f"Exchange {exchange} not found")
                
            # 解析配置参数：合并全局配置和交易所特定配置
            default_config = {}
            for k, v in ccxt_cfg.items():
                # 获取通用配置（不是针对特定交易所的设置）
                if k in ccxt.exchanges:
                    continue
                default_config[k] = v
                
            # 如果有该交易所的特定配置，则合并进来
            if exchange in ccxt_cfg:
                default_config = merge_dicts(default_config, ccxt_cfg[exchange])
                
            # 合并默认配置和用户提供的配置
            config = merge_dicts(default_config, config)
            
            # 创建交易所实例
            exchange = getattr(ccxt, exchange)(config)
        else:
            # 如果exchange已经是实例，则不能再应用配置
            if len(config) > 0:
                raise ValueError("Cannot apply config after instantiation of the exchange")
                
        # 验证交易所是否支持OHLCV数据获取
        if not exchange.has['fetchOHLCV']:
            raise ValueError(f"Exchange {exchange} does not support OHLCV")
            
        # 验证交易所是否支持指定的时间框架
        if timeframe not in exchange.timeframes:
            raise ValueError(f"Exchange {exchange} does not support {timeframe} timeframe")
            
        # 如果使用的是模拟OHLCV（可能不够准确），发出警告
        if exchange.has['fetchOHLCV'] == 'emulated':
            warnings.warn("Using emulated OHLCV candles", stacklevel=2)

        # 定义重试装饰器，用于处理网络错误和交易所错误
        def _retry(method):
            @wraps(method)
            def retry_method(*args, **kwargs):
                # 执行指定次数的重试
                for i in range(retries):
                    try:
                        # 尝试执行原方法
                        return method(*args, **kwargs)
                    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                        # 如果是最后一次重试，抛出异常
                        if i == retries - 1:
                            raise e
                    # 如果设置了延迟，在重试之间等待
                    if delay is not None:
                        time.sleep(delay / 1000)

            return retry_method

        # 使用重试装饰器包装数据获取函数
        @_retry
        def _fetch(_since, _limit):
            # 调用交易所的OHLCV数据获取方法
            return exchange.fetch_ohlcv(
                symbol,                 # 交易对符号
                timeframe=timeframe,    # 时间框架
                since=_since,          # 开始时间戳
                limit=_limit,          # 数据条数限制
                params=params          # 额外参数
            )

        # 建立时间戳范围
        # 将开始时间转换为毫秒时间戳
        start_ts = datetime_to_ms(to_tzaware_datetime(start, tz=get_utc_tz()))
        
        try:
            # 尝试获取第一条数据以确定实际的数据起始时间
            first_data = _fetch(0, 1)
            first_valid_ts = first_data[0][0]
            # 使用用户指定时间和实际可用时间中的较大值
            next_start_ts = start_ts = max(start_ts, first_valid_ts)
        except:
            # 如果获取失败，使用用户指定的开始时间
            next_start_ts = start_ts
            
        # 将结束时间转换为毫秒时间戳
        end_ts = datetime_to_ms(to_tzaware_datetime(end, tz=get_utc_tz()))

        # 定义时间戳转字符串的辅助函数，用于进度显示
        def _ts_to_str(ts):
            return str(pd.Timestamp(to_tzaware_datetime(ts, tz=get_utc_tz())))

        # 迭代收集数据 - 初始化数据存储列表
        data: tp.List[list] = []
        
        # 使用进度条显示数据下载进度
        with tqdm(disable=not show_progress, **tqdm_kwargs) as pbar:
            # 设置进度条初始描述为开始时间
            pbar.set_description(_ts_to_str(start_ts))
            
            # 开始数据获取循环
            while True:
                # 获取下一批OHLCV数据
                next_data = _fetch(next_start_ts, limit)
                
                # 根据是否已有数据来过滤新获取的数据
                if len(data) > 0:
                    # 如果已有数据，过滤掉重复和超出范围的数据
                    next_data = list(filter(lambda d: next_start_ts < d[0] < end_ts, next_data))
                else:
                    # 如果是第一次获取，只过滤超出结束时间的数据
                    next_data = list(filter(lambda d: d[0] < end_ts, next_data))

                # 更新时间戳和进度条显示
                if not len(next_data):
                    # 如果没有新数据，跳出循环
                    break
                    
                # 将新数据添加到主数据列表中
                data += next_data
                
                # 更新进度条描述，显示当前下载的时间范围
                pbar.set_description("{} - {}".format(
                    _ts_to_str(start_ts),          # 开始时间
                    _ts_to_str(next_data[-1][0])   # 当前最新时间
                ))
                
                # 更新进度条
                pbar.update(1)
                
                # 设置下次请求的开始时间为当前批次的最后时间
                next_start_ts = next_data[-1][0]
                
                # 如果设置了延迟，则等待指定时间（对API友好）
                if delay is not None:
                    time.sleep(delay / 1000)  # 延迟单位从毫秒转换为秒

        # 将原始数据转换为DataFrame格式
        # 创建DataFrame并指定列名，对应CCXT标准的OHLCV格式
        df = pd.DataFrame(data, columns=[
            'Open time',    # 开盘时间（毫秒时间戳）
            'Open',         # 开盘价
            'High',         # 最高价
            'Low',          # 最低价
            'Close',        # 收盘价
            'Volume'        # 成交量
        ])
        
        # 设置DataFrame的索引为开盘时间
        # 将毫秒时间戳转换为pandas时间格式，并设置为UTC时区
        df.index = pd.to_datetime(df['Open time'], unit='ms', utc=True)
        
        # 删除原始的开盘时间列，因为已经设置为索引
        del df['Open time']
        
        # 数据类型转换：将字符串格式的数据转换为浮点数
        df['Open'] = df['Open'].astype(float)     # 开盘价转浮点数
        df['High'] = df['High'].astype(float)     # 最高价转浮点数
        df['Low'] = df['Low'].astype(float)       # 最低价转浮点数
        df['Close'] = df['Close'].astype(float)   # 收盘价转浮点数
        df['Volume'] = df['Volume'].astype(float) # 成交量转浮点数

        # 返回处理完成的DataFrame
        return df

    def update_symbol(self, symbol: str, **kwargs) -> tp.Frame:
        """
        更新指定符号的CCXT数据
        
        该方法获取现有数据之后的新数据，用于实现跨交易所数据的增量更新。
        
        Args:
            symbol (str): 要更新的交易对符号
            **kwargs: 额外参数，会覆盖原始下载参数
            
        Returns:
            tp.Frame: 新获取的增量数据
        """
        # 获取该符号的原始下载参数配置
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        
        # 将开始时间设置为现有数据的最后一个时间点
        # 确保获取的是从上次更新之后的新数据
        download_kwargs['start'] = self.data[symbol].index[-1]
        
        # 关闭进度条显示，更新操作通常在后台进行
        download_kwargs['show_progress'] = False
        
        # 合并原始参数和新传入的参数，新参数具有更高优先级
        kwargs = merge_dicts(download_kwargs, kwargs)
        
        # 调用download_symbol方法获取新的数据
        return self.download_symbol(symbol, **kwargs)


class AlpacaData(Data):
    """
    Alpaca证券数据类 - 基于alpaca-py库获取美股和加密货币数据
    
    AlpacaData通过Alpaca Markets提供的免佣金经纪商API获取高质量的美股和加密货币数据。
    Alpaca是美国知名的金融科技公司，专为算法交易和量化投资提供基础设施服务。
    
    主要特点：
    1. 专业级数据：来自专业券商，数据质量高于免费源
    2. 实时性强：提供近实时的市场数据（免费账户有15分钟延迟）
    3. 多资产支持：美股、ETF、加密货币等
    4. API友好：专为程序化交易设计的现代RESTful API
    5. 免费使用：提供免费的纸质交易账户和数据访问
    
    数据源优势：
    - 数据可靠性：直接来自证券交易所和做市商
    - 低延迟：专业级的数据传输基础设施
    - 完整性：包含完整的OHLCV数据和市场深度信息
    - 合规性：受美国SEC和FINRA监管
    
    支持的资产类型：
    - 美股：NYSE、NASDAQ上市的所有股票
    - ETF：主要的交易所交易基金
    - 加密货币：BTC、ETH等主流数字货币
    - 期权：股票期权数据（高级功能）
    
    账户类型：
    - 纸质交易账户：免费，模拟交易，数据延迟15分钟
    - 真实交易账户：实时数据，支持真实交易
    - 专业账户：更高级的数据和功能
    
    使用场景：
    1. 美股量化策略开发和回测
    2. 算法交易系统的数据支持
    3. 投资组合管理和风险控制
    4. 学术研究和金融教育
    5. 个人投资的数据分析
    
    使用示例：
    ```python
    # 需要先注册Alpaca账户并获取API密钥
    # 注册地址：https://app.alpaca.markets/signup
    
    # 获取苹果股票分钟级数据
    aapl_data = vbt.AlpacaData.download(
        "AAPL",                     # 苹果股票代码
        start='2 hours ago UTC',    # 2小时前
        end='15 minutes ago UTC',   # 15分钟前（免费账户延迟）
        timeframe='1m'              # 1分钟K线
    )
    
    # 获取收盘价数据
    aapl_prices = aapl_data.get('Close')
    print(f"AAPL最新价格: ${aapl_prices.iloc[-1]:.2f}")
    
    # 获取多只科技股数据比较
    tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    for symbol in tech_stocks:
        data = vbt.AlpacaData.download(
            symbol,
            timeframe='1d',
            start='30 days ago'
        )
        latest_price = data.get('Close').iloc[-1]
        print(f"{symbol}: ${latest_price:.2f}")
    ```
    
    配置要求：
    1. 安装alpaca-py库：pip install alpaca-py
    2. 注册Alpaca账户获取API密钥
    3. 在vectorbt配置中设置API密钥
    
    注意事项：
    - 免费账户的数据有15分钟延迟
    - 需要有效的Alpaca API密钥
    - 受美国市场交易时间限制
    - API有请求频率限制
    - 某些高级功能需要付费账户
    """

    @classmethod
    def download_symbol(cls,
                        symbol: str,
                        timeframe: str = '1d',
                        start: tp.DatetimeLike = 0,
                        end: tp.DatetimeLike = 'now UTC',
                        adjustment: tp.Optional[str] = 'all',
                        limit: int = 500,
                        feed: tp.Optional[str] = None,
                        **kwargs) -> tp.Frame:
        """Download the symbol.

        Args:
            symbol (str): Symbol.
            timeframe (str): Timeframe of data.

                Must be integer multiple of 'm' (minute), 'h' (hour) or 'd' (day). i.e. '15m'.
                See https://alpaca.markets/data.

                !!! note
                    Data from the latest 15 minutes is not available with a free data plan.

            start (any): Start datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            adjustment (str): Specifies the corporate action adjustment for the stocks. 

                Allowed are `raw`, `split`, `dividend` or `all`.
            limit (int): The maximum number of returned items.
            feed (str): The feed to pull market data from.

                This is either "iex", "otc", or "sip". Feeds "sip" and "otc" are only available to
                those with a subscription. Default is "iex" for free plans and "sip" for paid.

        For defaults, see `data.alpaca` in `vectorbt._settings.settings`.
        """
        # 导入必要的模块
        from vectorbt._settings import settings
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

        # 获取Alpaca相关的配置设置
        alpaca_cfg = settings['data']['alpaca']

        # 根据符号类型选择合适的客户端类型
        if "/" in symbol:
            # 包含"/"的符号被视为加密货币交易对（如BTC/USD）
            REST = CryptoHistoricalDataClient
        else:
            # 不包含"/"的符号被视为股票代码（如AAPL）
            REST = StockHistoricalDataClient

        # 提取客户端构造函数的参数
        client_kwargs = dict()
        for k in get_func_kwargs(REST):
            # 如果kwargs中包含客户端参数，则提取到client_kwargs中
            if k in kwargs:
                client_kwargs[k] = kwargs.pop(k)

        # 合并默认配置和用户提供的客户端参数
        client_kwargs = merge_dicts(alpaca_cfg, client_kwargs)

        # 创建适当的历史数据客户端实例
        client = REST(**client_kwargs)

        # 定义时间单位映射表
        _timeframe_units = {
            'd': TimeFrameUnit.Day,     # 日
            'h': TimeFrameUnit.Hour,    # 小时
            'm': TimeFrameUnit.Minute   # 分钟
        }

        # 验证时间框架格式
        if len(timeframe) < 2:
            raise ValueError("invalid timeframe")

        # 解析时间框架字符串（如"15m"拆分为"15"和"m"）
        amount_str = timeframe[:-1]  # 数量部分
        unit_str = timeframe[-1]     # 单位部分

        # 验证数量是否为数字，单位是否在支持的范围内
        if not amount_str.isnumeric() or unit_str not in _timeframe_units:
            raise ValueError("invalid timeframe")

        # 转换为整数和对应的时间单位枚举
        amount = int(amount_str)
        unit = _timeframe_units[unit_str]

        # 创建Alpaca的TimeFrame对象
        _timeframe = TimeFrame(amount, unit)

        # 将开始和结束时间转换为ISO格式字符串
        start_ts = to_tzaware_datetime(start, tz=get_utc_tz()).isoformat()
        end_ts = to_tzaware_datetime(end, tz=get_utc_tz()).isoformat()

        # 根据符号类型调用不同的API获取数据
        if "/" in symbol:
            # 获取加密货币K线数据
            df = client.get_crypto_bars(CryptoBarsRequest(
                symbol_or_symbols=symbol,  # 加密货币交易对
                timeframe=_timeframe,      # 时间框架
                start=start_ts,           # 开始时间
                end=end_ts,               # 结束时间
                limit=limit,              # 数据条数限制
            )).df
        else:
            # 获取股票K线数据
            df = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol,  # 股票代码
                timeframe=_timeframe,      # 时间框架
                start=start_ts,           # 开始时间
                end=end_ts,               # 结束时间
                adjustment=adjustment,     # 价格调整类型
                limit=limit,              # 数据条数限制
                feed=feed,                # 数据源类型
            )).df

        # 筛选OHLCV数据并移除额外列
        # 移除多余的列数据（交易数量和成交量加权平均价格）
        df.drop(['trade_count', 'vwap'], axis=1, errors='ignore', inplace=True)

        # 标准化列名：将小写列名转换为标准的大写格式
        # 确保与vectorbt框架的数据格式保持一致
        df.rename(columns={
            'open': 'Open',       # 开盘价
            'high': 'High',       # 最高价
            'low': 'Low',         # 最低价
            'close': 'Close',     # 收盘价
            'volume': 'Volume',   # 成交量
        }, inplace=True)

        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        return df

    def update_symbol(self, symbol: str, **kwargs) -> tp.Frame:
        """
        更新指定符号的Alpaca数据
        
        该方法获取现有数据之后的新数据，用于实现美股和加密货币数据的增量更新。
        
        Args:
            symbol (str): 要更新的符号（股票代码或加密货币交易对）
            **kwargs: 额外参数，会覆盖原始下载参数
            
        Returns:
            tp.Frame: 新获取的增量数据
        """
        # 获取该符号的原始下载参数配置
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        
        # 将开始时间设置为现有数据的最后一个时间点
        # 确保获取的是从上次更新之后的新数据
        download_kwargs['start'] = self.data[symbol].index[-1]
        
        # 关闭进度条显示，更新操作通常在后台进行
        download_kwargs['show_progress'] = False
        
        # 合并原始参数和新传入的参数，新参数具有更高优先级
        kwargs = merge_dicts(download_kwargs, kwargs)
        
        # 调用download_symbol方法获取新的数据
        return self.download_symbol(symbol, **kwargs)
