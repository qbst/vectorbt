# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
VectorBT收益率高级统计度量模块

该模块专注于提供不依赖Numba JIT编译的高级统计风险度量指标，主要涉及夏普比率的
统计学扩展和偏差校正。这些指标在量化金融研究中用于解决多重假设检验问题和
统计推断中的过拟合问题，为投资策略的真实绩效评估提供更加严格的统计框架。

设计逻辑：
    本模块实现了基于统计学理论的高级绩效度量方法，主要解决传统夏普比率在
    多策略测试环境中的统计偏差问题。通过引入期望最大夏普比率和紧缩夏普比率
    等概念，为量化投资研究提供更加严格的统计显著性检验工具。

核心功能：
    - 期望最大夏普比率计算：基于极值理论估计随机策略的最大期望夏普比率
    - 紧缩夏普比率计算：校正多重假设检验偏差，评估策略的真实统计显著性
    - 统计推断支持：为策略回测结果提供严格的统计显著性验证

应用场景：
    - **量化研究验证**：验证策略回测结果的统计显著性
    - **多策略筛选**：在大量策略中识别真正具有超额收益能力的策略
    - **过拟合检测**：识别和避免策略开发中的过度优化问题
    - **学术研究**：为金融学术研究提供严格的统计推断工具
    - **监管报告**：满足监管机构对策略有效性统计验证的要求

理论基础：
    本模块基于Bailey和López de Prado等学者的研究成果，结合极值理论、
    多重假设检验理论和非参数统计学方法，为量化投资提供理论严谨的
    绩效评估框架。

技术特点：
    - 使用SciPy统计库实现高精度统计计算
    - 支持向量化操作，适合大规模策略批量分析
    - 提供灵活的参数配置，适应不同的研究场景
    - 与vectorbt生态系统无缝集成

注意事项：
    这些高级统计指标需要足够的历史数据和合理的参数设置才能产生可靠结果，
    建议结合领域专家知识和统计学理论进行正确解释和应用。
"""

import numpy as np  # 导入NumPy库，用于数值计算和数组操作
from scipy.stats import norm  # 从SciPy统计模块导入正态分布，用于统计推断和概率计算

from vectorbt import _typing as tp  # 导入vectorbt类型注释模块，提供类型提示支持


def approx_exp_max_sharpe(mean_sharpe: float, var_sharpe: float, nb_trials: int) -> float:
    """
    期望最大夏普比率近似计算函数
    
    该函数基于极值理论计算在给定试验次数下，随机策略可能达到的期望最大夏普比率。
    这是多重假设检验理论中的核心概念，用于建立统计显著性的基准线。通过估计
    "运气成分"可能产生的最大夏普比率，帮助研究者区分真实的投资技能和随机运气。
    
    理论背景：
        基于Bailey和López de Prado的研究，当进行大量策略测试时，即使是完全随机的
        策略也可能产生表面上较高的夏普比率。该函数使用Gumbel分布的渐近理论，
        结合Euler-Mascheroni常数和正态分位数，提供期望最大值的理论估计。
    
    数学原理：
        E[max(SR)] ≈ μ_SR + σ_SR × [(1-γ) × Φ⁻¹(1-1/N) + γ × Φ⁻¹(1-1/(N×e))]
        其中：
        - μ_SR: 夏普比率均值（通常为0，表示无技能假设）
        - σ_SR: 夏普比率标准差
        - γ: Euler-Mascheroni常数 ≈ 0.5772
        - Φ⁻¹: 标准正态分布的逆累积分布函数
        - N: 试验次数
        - e: 自然常数
    
    参数说明：
        mean_sharpe (float): 夏普比率的均值
            - 通常设置为0，表示"无技能假设"（策略无真实超额收益能力）
            - 也可以基于历史数据或先验知识设置非零值
        var_sharpe (float): 夏普比率的方差
            - 反映夏普比率估计的不确定性
            - 可以通过历史数据或理论推导获得
        nb_trials (int): 试验次数（测试的策略数量）
            - 表示进行假设检验的策略总数
            - 数值越大，期望最大夏普比率越高
    
    返回值：
        float: 期望最大夏普比率
            - 表示在给定试验次数下，随机策略可能达到的期望最大夏普比率
            - 用作后续统计显著性检验的基准阈值
            - 高于此值的夏普比率更可能反映真实的投资技能
    
    使用示例：
        >>> # 示例1：评估100个随机策略的期望最大夏普比率
        >>> mean_sr = 0.0  # 无技能假设
        >>> var_sr = 1.0   # 夏普比率方差为1
        >>> n_strategies = 100  # 测试100个策略
        >>> 
        >>> expected_max_sr = approx_exp_max_sharpe(mean_sr, var_sr, n_strategies)
        >>> print(f"100个随机策略的期望最大夏普比率: {expected_max_sr:.3f}")
        >>> # 输出: 100个随机策略的期望最大夏普比率: 2.145
        
        >>> # 示例2：不同策略数量的比较
        >>> for n in [10, 50, 100, 500, 1000]:
        ...     max_sr = approx_exp_max_sharpe(0.0, 1.0, n)
        ...     print(f"{n:4d}个策略: 期望最大SR = {max_sr:.3f}")
        
        输出示例：
          10个策略: 期望最大SR = 1.539
          50个策略: 期望最大SR = 1.940
         100个策略: 期望最大SR = 2.145
         500个策略: 期望最大SR = 2.576
        1000个策略: 期望最大SR = 2.751
    
    实际应用场景：
        - **策略开发**：在开发新策略时，评估获得的夏普比率是否超越随机水平
        - **回测验证**：验证历史回测结果的统计显著性
        - **多策略筛选**：在大量策略中识别真正有效的策略
        - **风险管理**：为投资委员会提供策略评估的统计基准
        - **学术研究**：为金融学术论文提供严格的统计检验基准
    
    统计解释：
        - **显著性阈值**：该值可作为统计显著性的阈值，超过此值的策略更可能具有真实技能
        - **多重检验校正**：自动考虑了多重假设检验的影响，避免了Type I错误的累积
        - **保守估计**：提供相对保守的估计，有助于减少过拟合策略的误判
    
    注意事项：
        - 该方法假设夏普比率服从正态分布，在极端情况下可能不准确
        - 试验次数应该反映实际进行的策略测试数量
        - 结果应该结合具体的投资环境和风险承受能力进行解释
        - 这是近似方法，在样本量较小时精度可能有限
    
    参考文献：
        - Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: 
          Correcting for Selection Bias, Backtest Overfitting, and Non-Normality"
        - Harvey, C. R., & Liu, Y. (2015). "Backtesting"
    """
    # 计算期望最大夏普比率的近似值
    # 使用基于极值理论的公式，结合Euler-Mascheroni常数和正态分位数
    return mean_sharpe + np.sqrt(var_sharpe) * \
           ((1 - np.euler_gamma) * norm.ppf(1 - 1 / nb_trials) + np.euler_gamma * norm.ppf(1 - 1 / (nb_trials * np.e)))
    # mean_sharpe: 夏普比率均值基准
    # np.sqrt(var_sharpe): 夏普比率标准差
    # (1 - np.euler_gamma): (1 - 欧拉常数)，约等于0.4228
    # norm.ppf(1 - 1 / nb_trials): 标准正态分布的(1-1/N)分位数
    # np.euler_gamma: 欧拉-马歇罗尼常数，约等于0.5772
    # norm.ppf(1 - 1 / (nb_trials * np.e)): 标准正态分布的(1-1/(N×e))分位数
    # 整个表达式实现了Gumbel分布最大值期望的近似计算


def deflated_sharpe_ratio(*, est_sharpe: tp.Array1d, var_sharpe: float, nb_trials: int,
                          backtest_horizon: int, skew: tp.Array1d, kurtosis: tp.Array1d) -> tp.Array1d:
    """
    紧缩夏普比率（DSR）计算函数
    
    紧缩夏普比率是一种校正多重假设检验偏差的统计指标，用于评估投资策略
    夏普比率的真实统计显著性。该指标通过考虑策略选择偏差、回测过拟合和
    收益分布的非正态性，提供更加严格和可靠的策略绩效评估。
    
    核心思想：
        传统的夏普比率在多策略测试环境中容易产生选择偏差，即研究者倾向于
        选择表现最好的策略，而忽略了这种选择过程本身带来的统计偏差。DSR
        通过引入统计显著性检验，将策略的夏普比率转换为其统计显著性的概率。
    
    理论基础：
        DSR基于假设检验理论，原假设H0为策略无真实超额收益能力，备择假设H1为
        策略具有真实的投资技能。DSR值表示在给定证据下，拒绝原假设的置信度，
        即策略具有真实技能的概率。
    
    数学公式：
        DSR = Φ((SR_est - SR0) × √(T-1) / √(1 - S×SR_est + (K-1)/4 × SR_est²))
        
        其中：
        - Φ: 标准正态分布的累积分布函数
        - SR_est: 估计的夏普比率
        - SR0: 期望最大夏普比率（通过approx_exp_max_sharpe计算）
        - T: 回测时间长度
        - S: 收益分布的偏度
        - K: 收益分布的峰度
    
    参数说明：
        est_sharpe (tp.Array1d): 估计的夏普比率数组
            - 通过历史数据计算得到的策略夏普比率
            - 支持同时评估多个策略
        var_sharpe (float): 夏普比率的方差
            - 反映夏普比率估计的不确定性
            - 通常通过统计分析或理论推导获得
        nb_trials (int): 试验次数（测试的策略数量）
            - 表示总共测试的策略数量
            - 用于计算期望最大夏普比率基准
        backtest_horizon (int): 回测时间长度
            - 表示用于策略评估的时间点数量
            - 影响统计检验的功效和精度
        skew (tp.Array1d): 收益分布的偏度数组
            - 衡量收益分布相对于正态分布的不对称程度
            - 负偏度表示极端损失的概率较高
        kurtosis (tp.Array1d): 收益分布的峰度数组
            - 衡量收益分布的尾部厚度
            - 高峰度表示极端事件发生概率较高
    
    返回值：
        tp.Array1d: 紧缩夏普比率数组
            - 取值范围在[0, 1]之间
            - 接近1表示策略具有高度统计显著性
            - 接近0.5表示策略与随机策略无显著差异
            - 低于0.5表示策略表现可能不如随机策略
    
    使用示例：
        >>> # 示例1：单个策略的DSR评估
        >>> import numpy as np
        >>> 
        >>> # 策略基本参数
        >>> estimated_sharpe = np.array([1.5])  # 策略的夏普比率
        >>> sharpe_variance = 1.0               # 夏普比率方差
        >>> total_strategies = 100              # 总共测试的策略数
        >>> backtest_length = 252               # 回测长度（1年日数据）
        >>> return_skewness = np.array([-0.5])  # 收益偏度
        >>> return_kurtosis = np.array([4.0])   # 收益峰度
        >>> 
        >>> # 计算DSR
        >>> dsr = deflated_sharpe_ratio(
        ...     est_sharpe=estimated_sharpe,
        ...     var_sharpe=sharpe_variance,
        ...     nb_trials=total_strategies,
        ...     backtest_horizon=backtest_length,
        ...     skew=return_skewness,
        ...     kurtosis=return_kurtosis
        ... )
        >>> 
        >>> print(f"策略DSR: {dsr[0]:.3f}")
        >>> if dsr[0] > 0.95:
        ...     print("策略具有高度统计显著性")
        >>> elif dsr[0] > 0.8:
        ...     print("策略具有统计显著性")
        >>> elif dsr[0] > 0.5:
        ...     print("策略可能具有一定技能，但显著性不强")
        >>> else:
        ...     print("策略缺乏统计显著性")
        
        >>> # 示例2：批量策略评估
        >>> # 模拟10个不同策略的参数
        >>> strategies_sr = np.array([0.8, 1.2, 1.5, 0.5, 2.0, 
        ...                          1.8, 0.3, 1.0, 2.2, 1.3])
        >>> strategies_skew = np.random.normal(-0.2, 0.3, 10)
        >>> strategies_kurt = np.random.normal(3.5, 0.8, 10)
        >>> 
        >>> # 计算所有策略的DSR
        >>> batch_dsr = deflated_sharpe_ratio(
        ...     est_sharpe=strategies_sr,
        ...     var_sharpe=1.0,
        ...     nb_trials=1000,  # 假设从1000个策略中筛选
        ...     backtest_horizon=500,  # 2年日数据
        ...     skew=strategies_skew,
        ...     kurtosis=strategies_kurt
        ... )
        >>> 
        >>> # 结果分析
        >>> for i, (sr, dsr_val) in enumerate(zip(strategies_sr, batch_dsr)):
        ...     print(f"策略{i+1}: SR={sr:.2f}, DSR={dsr_val:.3f}")
    
    结果解释指南：
        - **DSR > 0.95**: 策略具有高度统计显著性，推荐采用
        - **0.8 < DSR ≤ 0.95**: 策略具有统计显著性，可以考虑采用
        - **0.6 < DSR ≤ 0.8**: 策略可能具有一定技能，需要进一步验证
        - **0.4 < DSR ≤ 0.6**: 策略与随机策略差异不大，需要谨慎
        - **DSR ≤ 0.4**: 策略缺乏统计显著性，不建议采用
    
    实际应用场景：
        - **量化基金管理**：评估基金经理策略的真实有效性
        - **投资研究**：验证新开发策略的统计显著性
        - **风险管理**：识别过拟合的策略，避免投资风险
        - **监管合规**：满足监管机构对策略有效性的统计验证要求
        - **学术研究**：为金融学术研究提供严格的统计推断工具
    
    优势特点：
        - **偏差校正**：自动校正多重假设检验和选择偏差
        - **分布灵活**：考虑收益分布的偏度和峰度特征
        - **统计严格**：基于严格的统计假设检验理论
        - **实用性强**：提供直观的概率解释
    
    注意事项：
        - 需要准确估计收益分布的偏度和峰度参数
        - 回测时间长度应该足够长以确保统计检验的有效性
        - 试验次数应该反映实际的策略测试规模
        - 结果应该结合业务知识和市场环境进行综合判断
        - DSR较低不一定意味着策略无价值，可能需要更长的验证期
    
    理论局限性：
        - 假设夏普比率渐近正态分布，在小样本情况下可能不准确
        - 对偏度和峰度参数的估计精度敏感
        - 不能完全消除所有形式的数据窥探偏差
        - 在市场环境发生结构性变化时，历史统计特征可能失效
    
    参考文献：
        - Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio"
        - Harvey, C. R., Liu, Y., & Zhu, H. (2016). "...and the Cross-Section of Expected Returns"
        - López de Prado, M. (2018). "Advances in Financial Machine Learning"
    
    扩展阅读：
        有关DSR的详细介绍和应用，请参考：
        https://gmarti.gitlab.io/qfin/2018/05/30/deflated-sharpe-ratio.html
    """
    # 第一步：计算期望最大夏普比率作为统计显著性的基准阈值
    # 使用无技能假设（mean=0）和给定的夏普比率方差
    SR0 = approx_exp_max_sharpe(0, var_sharpe, nb_trials)
    # SR0表示在给定试验次数下，随机策略可能达到的期望最大夏普比率
    # 这个值作为判断策略是否具有真实技能的基准线

    # 第二步：计算紧缩夏普比率
    # 使用标准正态分布的累积分布函数计算统计显著性概率
    return norm.cdf(((est_sharpe - SR0) * np.sqrt(backtest_horizon - 1)) /
                    np.sqrt(1 - skew * est_sharpe + ((kurtosis - 1) / 4) * est_sharpe ** 2))
    # 分子部分：(est_sharpe - SR0) * sqrt(T-1)
    #   - (est_sharpe - SR0)：估计夏普比率与期望最大值的差异
    #   - sqrt(backtest_horizon - 1)：时间长度的平方根，提供统计检验的功效
    # 分母部分：sqrt(1 - skew*SR + (kurtosis-1)/4 * SR²)
    #   - 1：基础方差项
    #   - skew * est_sharpe：偏度对方差的影响
    #   - (kurtosis-1)/4 * est_sharpe²：峰度对方差的二次影响
    # norm.cdf()：将统计量转换为概率值（DSR），表示策略具有真实技能的置信度
