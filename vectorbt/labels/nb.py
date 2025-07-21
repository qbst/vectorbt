# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT LABELS MODULE: 机器学习标签生成高性能计算核心模块
================================================================================

专门用于生成机器学习训练标签的高性能计算模块。

该模块提供了一整套基于Numba JIT编译的标签生成算法，主要用于创建监督学习模型的
目标变量（labels），特别适用于金融时间序列预测任务。

核心设计理念：
1. **前瞻性标签生成**：基于未来信息生成标签，用于训练预测模型而非实时交易
2. **高性能计算**：所有函数都使用@njit装饰器进行JIT编译，实现接近C语言的执行速度
3. **趋势识别算法**：实现了多种趋势识别和极值检测算法，为机器学习提供丰富的特征
4. **突破检测机制**：提供价格突破阈值的检测功能，用于生成交易信号标签

主要功能模块：
- **未来统计指标计算**：计算未来一段时间内的均值、标准差、最值等统计指标
- **固定窗口标签生成**：计算当前值与未来固定时间点的百分比变化
- **趋势标签生成**：基于局部极值识别价格趋势方向和强度
- **突破标签生成**：检测价格是否突破预设的上涨或下跌阈值
- **对称阈值计算**：计算对称的涨跌阈值，确保统计上的公平性

设计警告：
⚠️ 重要提醒：本模块中的所有函数都包含前瞻性偏差（Look-ahead Bias），这些函数使用了未来的数据信息，因此：
- 仅适用于离线训练机器学习模型
- 绝对不能用于实时交易系统
- 用于回测时需要确保没有引入未来信息泄露
"""

# 导入必要的库
import numpy as np  # 导入NumPy库，用于数值计算和数组操作
from numba import njit  # 导入Numba的JIT编译装饰器，用于高性能计算

# 导入vectorbt的类型定义
from vectorbt import _typing as tp  # 导入vectorbt的类型注解模块
from vectorbt.base.reshape_fns import flex_select_auto_nb  # 导入灵活选择函数，用于多维数组的智能索引
from vectorbt.generic import nb as generic_nb  # 导入通用的Numba编译函数库
from vectorbt.labels.enums import TrendMode  # 导入趋势模式枚举类型


@njit(cache=True)  # 使用Numba JIT编译，cache=True表示缓存编译结果以提高后续调用性能
def future_mean_apply_nb(close: tp.Array2d,  # 输入的价格数组，形状为(时间, 资产)
                         window: int,  # 统计窗口大小，表示未来多少个时间点
                         ewm: bool,  # 是否使用指数加权移动平均，True表示EWM，False表示简单移动平均
                         wait: int = 1,  # 等待期数，默认为1，表示排除当前值避免前瞻性偏差
                         adjust: bool = False) -> tp.Array2d:  # 是否调整权重，仅在EWM模式下有效
    """
    计算未来时间窗口内的均值标签生成函数
    
    该函数用于生成机器学习模型的标签，计算每个时间点未来指定窗口内价格的平均值。
    这是一种前瞻性指标，利用了未来信息，因此仅用于模型训练而非实时交易。
    
    参数说明：
        close: 价格数据的2维数组，行表示时间，列表示不同资产
        window: 未来统计窗口的大小（时间点数量）
        ewm: 是否使用指数加权移动平均
             - True: 使用指数加权，近期数据权重更高
             - False: 使用简单移动平均，所有数据权重相等
        wait: 等待期数，用于避免使用当前时间点的数据
              - 1: 从下一个时间点开始计算（推荐）
              - 0: 包含当前时间点（可能引入偏差）
        adjust: 指数加权移动平均的权重调整参数
                - True: 使用调整后的权重
                - False: 使用标准权重
    
    返回值：
        与输入形状相同的2维数组，包含每个时间点的未来均值标签
    
    使用场景：
        - 预测模型的回归标签生成
        - 趋势强度评估
        - 价格目标预测
        - 均值回归策略的基准计算
    
    示例：
        >>> import numpy as np
        >>> prices = np.array([[100, 200], [101, 201], [102, 202], [103, 203], [104, 204]])
        >>> future_mean = future_mean_apply_nb(prices, window=3, ewm=False, wait=1)
        >>> # 返回每个时间点未来3天的平均价格
    """
    if ewm:  # 如果使用指数加权移动平均
        # 将数组反转（[::-1]），计算EWM，然后再反转回来
        # 这样可以计算"未来"的EWM，而不是传统的"过去"EWM
        out = generic_nb.ewm_mean_nb(close[::-1], window, minp=window, adjust=adjust)[::-1]
    else:  # 如果使用简单移动平均
        # 同样先反转数组，计算滚动均值，然后反转回来
        out = generic_nb.rolling_mean_nb(close[::-1], window, minp=window)[::-1]
    
    if wait > 0:  # 如果设置了等待期
        # 使用后向移位函数，将结果向后移动wait个位置
        # 这样可以避免使用当前时间点的信息，减少前瞻性偏差
        return generic_nb.bshift_nb(out, wait)
    return out  # 如果没有等待期，直接返回结果


@njit(cache=True)  # 使用Numba JIT编译优化性能
def future_std_apply_nb(close: tp.Array2d,  # 输入的价格数组
                        window: int,  # 统计窗口大小
                        ewm: bool,  # 是否使用指数加权移动标准差
                        wait: int = 1,  # 等待期数
                        adjust: bool = False,  # 权重调整参数
                        ddof: int = 0) -> tp.Array2d:  # 自由度调整参数
    """
    计算未来时间窗口内的标准差标签生成函数
    
    该函数计算每个时间点未来指定窗口内价格的标准差，用于度量未来价格的波动性。
    这是一种前瞻性波动率指标，常用于风险预测模型的标签生成。
    
    参数说明：
        close: 价格数据的2维数组
        window: 未来统计窗口的大小
        ewm: 是否使用指数加权移动标准差
        wait: 等待期数，避免使用当前时间点数据
        adjust: 指数加权的权重调整参数
        ddof: 自由度增量（Delta Degrees of Freedom）
              - 0: 使用总体标准差公式（除以N）
              - 1: 使用样本标准差公式（除以N-1）
    
    返回值：
        与输入形状相同的2维数组，包含每个时间点的未来标准差标签
    
    使用场景：
        - 波动率预测模型
        - 风险评估标签生成
        - 期权定价模型的波动率输入
        - 动态风险管理系统
    
    示例：
        >>> prices = np.array([[100, 200], [105, 195], [98, 210], [102, 205], [107, 190]])
        >>> future_vol = future_std_apply_nb(prices, window=3, ewm=False, wait=1)
        >>> # 返回每个时间点未来3天的价格波动率
    """
    if ewm:  # 如果使用指数加权移动标准差
        # 计算指数加权移动标准差，使用反转数组实现"未来"计算
        out = generic_nb.ewm_std_nb(close[::-1], window, minp=window, adjust=adjust, ddof=ddof)[::-1]
    else:  # 如果使用简单移动标准差
        # 计算滚动标准差，使用反转数组实现"未来"计算
        out = generic_nb.rolling_std_nb(close[::-1], window, minp=window, ddof=ddof)[::-1]
    
    if wait > 0:  # 如果设置了等待期
        # 向后移位，避免使用当前时间点信息
        return generic_nb.bshift_nb(out, wait)
    return out  # 返回计算结果


@njit(cache=True)  # 使用Numba JIT编译优化
def future_min_apply_nb(close: tp.Array2d,  # 输入的价格数组
                        window: int,  # 统计窗口大小
                        wait: int = 1) -> tp.Array2d:  # 等待期数
    """
    计算未来时间窗口内的最小值标签生成函数
    
    该函数计算每个时间点未来指定窗口内的最低价格，用于生成支撑位预测标签
    或风险管理中的最坏情况估计。
    
    参数说明：
        close: 价格数据的2维数组
        window: 未来统计窗口的大小
        wait: 等待期数，避免使用当前时间点数据
    
    返回值：
        与输入形状相同的2维数组，包含每个时间点的未来最小值标签
    
    使用场景：
        - 支撑位预测模型
        - 止损点设置参考
        - 风险管理中的最坏情况分析
        - 价格底部预测
    
    示例：
        >>> prices = np.array([[100, 200], [105, 195], [98, 210], [102, 205], [107, 190]])
        >>> future_min = future_min_apply_nb(prices, window=3, wait=1)
        >>> # 返回每个时间点未来3天的最低价格
    """
    # 使用反转数组计算未来的滚动最小值
    out = generic_nb.rolling_min_nb(close[::-1], window, minp=window)[::-1]
    
    if wait > 0:  # 如果设置了等待期
        # 向后移位，避免使用当前时间点信息
        return generic_nb.bshift_nb(out, wait)
    return out  # 返回计算结果


@njit(cache=True)  # 使用Numba JIT编译优化
def future_max_apply_nb(close: tp.Array2d,  # 输入的价格数组
                        window: int,  # 统计窗口大小
                        wait: int = 1) -> tp.Array2d:  # 等待期数
    """
    计算未来时间窗口内的最大值标签生成函数
    
    该函数计算每个时间点未来指定窗口内的最高价格，用于生成阻力位预测标签
    或收益潜力评估。
    
    参数说明：
        close: 价格数据的2维数组
        window: 未来统计窗口的大小
        wait: 等待期数，避免使用当前时间点数据
    
    返回值：
        与输入形状相同的2维数组，包含每个时间点的未来最大值标签
    
    使用场景：
        - 阻力位预测模型
        - 收益目标设置参考
        - 趋势强度评估
        - 价格顶部预测
    
    示例：
        >>> prices = np.array([[100, 200], [105, 195], [98, 210], [102, 205], [107, 190]])
        >>> future_max = future_max_apply_nb(prices, window=3, wait=1)
        >>> # 返回每个时间点未来3天的最高价格
    """
    # 使用反转数组计算未来的滚动最大值
    out = generic_nb.rolling_max_nb(close[::-1], window, minp=window)[::-1]
    
    if wait > 0:  # 如果设置了等待期
        # 向后移位，避免使用当前时间点信息
        return generic_nb.bshift_nb(out, wait)
    return out  # 返回计算结果


@njit(cache=True)  # 使用Numba JIT编译优化
def fixed_labels_apply_nb(close: tp.Array2d,  # 输入的价格数组
                          n: int) -> tp.Array2d:  # 固定的未来时间步数
    """
    生成固定时间点的价格变化百分比标签
    
    该函数计算当前价格与未来第n个时间点价格的百分比变化，这是最简单直接的
    标签生成方法，广泛用于股价涨跌预测模型。
    
    参数说明：
        close: 价格数据的2维数组，行表示时间，列表示不同资产
        n: 未来的时间步数
           - 1: 下一个时间点（如明日收盘价）
           - 5: 第5个时间点（如一周后收盘价）
           - 20: 第20个时间点（如一个月后收盘价）
    
    返回值：
        与输入形状相同的2维数组，包含价格变化百分比标签
        - 正值：价格上涨
        - 负值：价格下跌
        - 计算公式：(未来价格 - 当前价格) / 当前价格
    
    使用场景：
        - 股价涨跌预测模型的目标变量
        - 分类模型的回归标签生成
        - 策略收益率计算
        - 基准收益率对比
    
    示例：
        >>> prices = np.array([[100, 200], [105, 195], [98, 210], [102, 205], [107, 190]])
        >>> labels = fixed_labels_apply_nb(prices, n=2)
        >>> # 计算每个时间点2天后的价格变化百分比
        >>> # 对于第0天的价格100，第2天的价格是98，变化率为(98-100)/100=-0.02
    
    注意事项：
        - 该函数包含严重的前瞻性偏差，仅用于模型训练
        - 返回的标签数组末尾n个元素将为NaN（因为没有足够的未来数据）
        - 适合用作监督学习的回归目标或分类阈值的基础
    """
    # 使用后向移位函数获取未来第n个时间点的价格，然后计算百分比变化
    # generic_nb.bshift_nb(close, n) 将数组向后移位n个位置，相当于获取未来n步的价格
    # 公式：(future_price - current_price) / current_price
    return (generic_nb.bshift_nb(close, n) - close) / close


@njit(cache=True)  # 使用Numba JIT编译优化
def mean_labels_apply_nb(close: tp.Array2d,  # 输入的价格数组
                         window: int,  # 统计窗口大小
                         ewm: bool,  # 是否使用指数加权移动平均
                         wait: int = 1,  # 等待期数
                         adjust: bool = False) -> tp.Array2d:  # 权重调整参数
    """
    生成基于未来均值的价格变化百分比标签
    
    该函数计算当前价格与未来一段时间内平均价格的百分比变化，相比固定时间点标签
    更加平滑和稳定，有助于减少标签的噪声。
    
    参数说明：
        close: 价格数据的2维数组
        window: 未来统计窗口的大小
        ewm: 是否使用指数加权移动平均
        wait: 等待期数，避免使用当前时间点数据
        adjust: 指数加权的权重调整参数
    
    返回值：
        与输入形状相同的2维数组，包含基于未来均值的价格变化百分比标签
        - 正值：未来平均价格高于当前价格
        - 负值：未来平均价格低于当前价格
        - 计算公式：(未来均价 - 当前价格) / 当前价格
    
    使用场景：
        - 平滑的价格趋势预测标签
        - 减少噪声的回归目标
        - 中长期趋势方向预测
        - 均值回归策略的基准
    
    示例：
        >>> prices = np.array([[100, 200], [105, 195], [98, 210], [102, 205], [107, 190]])
        >>> labels = mean_labels_apply_nb(prices, window=3, ewm=False, wait=1)
        >>> # 计算每个时间点与未来3天平均价格的变化百分比
    
    优势：
        - 相比固定时间点标签更加平滑
        - 减少了极端值的影响
        - 提供更稳定的训练目标
    """
    # 先计算未来的均值，然后计算与当前价格的百分比变化
    # 使用之前定义的future_mean_apply_nb函数获取未来均值
    # 公式：(future_mean - current_price) / current_price
    return (future_mean_apply_nb(close, window, ewm, wait, adjust) - close) / close


@njit(cache=True)  # 使用Numba JIT编译优化
def get_symmetric_pos_th_nb(neg_th: tp.MaybeArray[float]) -> tp.MaybeArray[float]:
    """
    计算与负收益率对称的正收益率阈值
    
    该函数用于确保涨跌阈值在统计上的对称性。由于收益率的非对称性质，
    50%的下跌需要100%的上涨才能回到原来的水平。
    
    参数说明：
        neg_th: 负收益率阈值（下跌幅度）
                - 应该是正数，表示下跌的百分比
                - 支持标量或数组格式
    
    返回值：
        对称的正收益率阈值
        - 计算公式：neg_th / (1 - neg_th)
    
    数学原理：
        假设初始价格为P，下跌neg_th后价格为P*(1-neg_th)
        要回到原价格P，需要上涨的百分比为：
        (P - P*(1-neg_th)) / (P*(1-neg_th)) = neg_th / (1-neg_th)
    
    使用场景：
        - 确保涨跌阈值的统计对称性
        - 风险管理中的对称止损止盈设置
        - 期权策略中的对称执行价格计算
        - 波动率模型的对称参数设置
    
    示例：
        >>> neg_threshold = 0.5  # 50%下跌
        >>> pos_threshold = get_symmetric_pos_th_nb(neg_threshold)
        >>> print(pos_threshold)  # 输出：1.0 (100%上涨)
        >>> 
        >>> # 验证对称性
        >>> initial_price = 100
        >>> down_price = initial_price * (1 - 0.5)  # 50下跌后
        >>> up_price = down_price * (1 + 1.0)       # 100%上涨后
        >>> print(up_price)  # 输出：100.0 (回到原价)
    """
    # 计算对称的正收益率阈值
    # 公式：neg_th / (1 - neg_th)
    # 例如：50%的下跌对应100%的上涨才能回到原价
    return neg_th / (1 - neg_th)


@njit(cache=True)  # 使用Numba JIT编译优化
def get_symmetric_neg_th_nb(pos_th: tp.MaybeArray[float]) -> tp.MaybeArray[float]:
    """
    计算与正收益率对称的负收益率阈值
    
    该函数是get_symmetric_pos_th_nb的逆函数，用于从正收益率阈值
    计算对应的负收益率阈值。
    
    参数说明：
        pos_th: 正收益率阈值（上涨幅度）
                - 应该是正数，表示上涨的百分比
                - 支持标量或数组格式
    
    返回值：
        对称的负收益率阈值
        - 计算公式：pos_th / (1 + pos_th)
    
    数学原理：
        假设初始价格为P，上涨pos_th后价格为P*(1+pos_th)
        要回到原价格P，需要下跌的百分比为：
        (P*(1+pos_th) - P) / (P*(1+pos_th)) = pos_th / (1+pos_th)
    
    使用场景：
        - 从上涨阈值计算对称的下跌阈值
        - 期权策略中的对称定价
        - 风险管理中的对称风险控制
        - 统计模型中的对称参数设置
    
    示例：
        >>> pos_threshold = 1.0  # 100%上涨
        >>> neg_threshold = get_symmetric_neg_th_nb(pos_threshold)
        >>> print(neg_threshold)  # 输出：0.5 (50%下跌)
        >>> 
        >>> # 验证对称性
        >>> initial_price = 100
        >>> up_price = initial_price * (1 + 1.0)    # 100%上涨后
        >>> down_price = up_price * (1 - 0.5)       # 50%下跌后
        >>> print(down_price)  # 输出：100.0 (回到原价)
    """
    # 计算对称的负收益率阈值
    # 公式：pos_th / (1 + pos_th)
    # 例如：100%的上涨对应50%的下跌才能回到原价
    return pos_th / (1 + pos_th)


@njit(cache=True)  # 使用Numba JIT编译优化
def local_extrema_apply_nb(close: tp.Array2d,  # 输入的价格数组
                           pos_th: tp.MaybeArray[float],  # 正向（上涨）阈值
                           neg_th: tp.MaybeArray[float],  # 负向（下跌）阈值
                           flex_2d: bool = True) -> tp.Array2d:  # 是否使用2维灵活索引
    """
    识别价格序列中的局部极值点（波峰和波谷）
    
    该函数实现了一个高效的局部极值检测算法，能够识别价格序列中的重要转折点。
    该算法基于动态阈值检测，只有当价格变化超过设定的阈值时才认为是有效的极值点。
    
    核心算法逻辑：
    1. 从价格序列开始遍历，寻找第一个满足阈值条件的转折点
    2. 一旦确定了方向（上涨或下跌），就持续更新当前的极值点
    3. 当价格反向变化超过阈值时，确认前一个极值点并改变方向
    4. 重复此过程直到序列结束
    
    参数说明：
        close: 价格数据的2维数组，行表示时间，列表示不同资产
        pos_th: 正向（上涨）阈值，用于识别价格上涨的极值点
                - 范围：(0, 1]，例如0.02表示2%
                - 支持标量、1维数组或2维数组
        neg_th: 负向（下跌）阈值，用于识别价格下跌的极值点
                - 范围：(0, 1]，例如0.02表示2%
                - 支持标量、1维数组或2维数组
        flex_2d: 是否使用2维灵活索引，支持不同位置使用不同阈值
    
    返回值：
        与输入形状相同的2维数组，包含局部极值标记
        - 1: 波峰（局部最高点）
        - -1: 波谷（局部最低点）
        - 0: 非极值点
    
    算法特点：
        - 动态阈值检测：避免噪声造成的虚假极值
        - 实时更新：在确定方向后持续更新极值点位置
        - 双向检测：同时检测上涨和下跌的极值点
        - 最后点处理：序列结束时的特殊处理
    
    使用场景：
        - 技术分析中的波峰波谷识别
        - 趋势转折点检测
        - 支撑阻力位自动识别
        - 波动交易策略的信号生成
        - 市场结构分析
    
    示例：
        >>> prices = np.array([[100, 105, 98, 110, 95, 108]]).T  # 转置为列向量
        >>> extrema = local_extrema_apply_nb(prices, pos_th=0.05, neg_th=0.05)
        >>> # 在5%的阈值下识别局部极值点
    
    注意事项：
        - 阈值设置需要平衡敏感性和稳定性
        - 过小的阈值可能产生过多的虚假信号
        - 过大的阈值可能遗漏重要的转折点
        - 该算法具有一定的滞后性，极值点在事后才能确认
    """
    # 将阈值转换为NumPy数组格式，确保类型一致性
    pos_th = np.asarray(pos_th)  # 转换正向阈值为数组
    neg_th = np.asarray(neg_th)  # 转换负向阈值为数组
    
    # 初始化输出数组，全部设为0（非极值点）
    out = np.full(close.shape, 0, dtype=np.int64)

    # 遍历每一列（每个资产）
    for col in range(close.shape[1]):
        prev_i = 0  # 记录上一个潜在极值点的索引
        direction = 0  # 记录当前的趋势方向：0=未确定，1=上涨，-1=下跌

        # 遍历每个时间点（从第1个开始，因为需要与前一个点比较）
        for i in range(1, close.shape[0]):
            # 使用灵活选择函数获取当前位置的阈值
            _pos_th = abs(flex_select_auto_nb(pos_th, prev_i, col, flex_2d))
            _neg_th = abs(flex_select_auto_nb(neg_th, prev_i, col, flex_2d))
            
            # 验证阈值的有效性
            if _pos_th == 0:
                raise ValueError("Positive threshold cannot be 0")  # 正向阈值不能为0
            if _neg_th == 0:
                raise ValueError("Negative threshold cannot be 0")  # 负向阈值不能为0

            if direction == 1:  # 当前处于上涨趋势中
                # 在上涨趋势中寻找下一个高点，同时更新当前的低点
                if close[i, col] < close[prev_i, col]:
                    # 如果当前价格低于之前的价格，更新最低点位置
                    prev_i = i
                elif close[i, col] >= close[prev_i, col] * (1 + _pos_th):
                    # 如果当前价格超过最低点的上涨阈值，确认之前的点为波谷
                    out[prev_i, col] = -1  # 标记为波谷
                    prev_i = i  # 更新极值点位置
                    direction = -1  # 改变趋势方向为下跌
                    
            elif direction == -1:  # 当前处于下跌趋势中
                # 在下跌趋势中寻找下一个低点，同时更新当前的高点
                if close[i, col] > close[prev_i, col]:
                    # 如果当前价格高于之前的价格，更新最高点位置
                    prev_i = i
                elif close[i, col] <= close[prev_i, col] * (1 - _neg_th):
                    # 如果当前价格低于最高点的下跌阈值，确认之前的点为波峰
                    out[prev_i, col] = 1  # 标记为波峰
                    prev_i = i  # 更新极值点位置
                    direction = 1  # 改变趋势方向为上涨
                    
            else:  # 趋势方向未确定，寻找第一个有效的极值点
                if close[i, col] >= close[prev_i, col] * (1 + _pos_th):
                    # 如果价格上涨超过正向阈值，确认起始点为波谷
                    out[prev_i, col] = -1  # 标记为波谷
                    prev_i = i  # 更新极值点位置
                    direction = -1  # 设置趋势方向为下跌
                elif close[i, col] <= close[prev_i, col] * (1 - _neg_th):
                    # 如果价格下跌超过负向阈值，确认起始点为波峰
                    out[prev_i, col] = 1  # 标记为波峰
                    prev_i = i  # 更新极值点位置
                    direction = 1  # 设置趋势方向为上涨

            # 特殊处理：如果到达序列的最后一个点
            if i == close.shape[0] - 1:
                # 如果已经确定了趋势方向，将最后的极值点标记为相反方向
                if direction != 0:
                    out[prev_i, col] = -direction  # 标记最后的极值点
    
    return out  # 返回极值点标记数组


@njit(cache=True)  # 使用Numba JIT编译优化
def bn_trend_labels_nb(close: tp.Array2d,  # 输入的价格数组
                       local_extrema: tp.Array2d) -> tp.Array2d:  # 局部极值点数组
    """
    生成二进制趋势标签（波峰到波谷 = 0，波谷到波峰 = 1）
    
    该函数基于局部极值点来生成简单的二进制趋势标签。在两个相邻的极值点之间，
    根据价格的变化方向分配标签：下跌趋势为0，上涨趋势为1。
    
    参数说明：
        close: 价格数据的2维数组
        local_extrema: 局部极值点数组，由local_extrema_apply_nb函数生成
                       - 1: 波峰点
                       - -1: 波谷点
                       - 0: 非极值点
    
    返回值：
        与输入形状相同的2维数组，包含二进制趋势标签
        - 0: 下跌趋势（从波峰到波谷）
        - 1: 上涨趋势（从波谷到波峰）
        - NaN: 极值点之外的区域
    
    算法逻辑：
        1. 找到所有的极值点（非零值）
        2. 遍历相邻的极值点对
        3. 比较两个极值点的价格
        4. 如果下一个极值点价格更高，标记为上涨趋势(1)
        5. 如果下一个极值点价格更低，标记为下跌趋势(0)
    
    使用场景：
        - 趋势分类模型的标签生成
        - 趋势跟踪策略的信号生成
        - 市场状态分类
        - 简单的趋势识别系统
    
    示例：
        >>> prices = np.array([[100, 105, 98, 110, 95, 108]]).T
        >>> extrema = local_extrema_apply_nb(prices, pos_th=0.05, neg_th=0.05)
        >>> trend_labels = bn_trend_labels_nb(prices, extrema)
        >>> # 在极值点之间生成上涨(1)或下跌(0)的趋势标签
    
    注意事项：
        - 只有在极值点之间的区域才有标签值
        - 需要先调用local_extrema_apply_nb函数生成极值点
        - 标签具有明确的时间边界（由极值点定义）
    """
    # 初始化输出数组，全部设为NaN
    out = np.full_like(close, np.nan, dtype=np.float64)

    # 遍历每一列（每个资产）
    for col in range(close.shape[1]):
        # 找到所有非零的极值点索引
        idxs = np.flatnonzero(local_extrema[:, col])
        
        # 如果没有找到极值点，跳过当前列
        if idxs.shape[0] == 0:
            continue

        # 遍历相邻的极值点对
        for k in range(1, idxs.shape[0]):
            prev_i = idxs[k - 1]  # 前一个极值点的索引
            next_i = idxs[k]      # 下一个极值点的索引

            # 比较两个极值点的价格，确定趋势方向
            if close[next_i, col] > close[prev_i, col]:
                # 如果下一个极值点价格更高，标记为上涨趋势
                out[prev_i:next_i, col] = 1
            else:
                # 如果下一个极值点价格更低，标记为下跌趋势
                out[prev_i:next_i, col] = 0

    return out  # 返回趋势标签数组


@njit(cache=True)  # 使用Numba JIT编译优化
def bn_cont_trend_labels_nb(close: tp.Array2d,  # 输入的价格数组
                            local_extrema: tp.Array2d) -> tp.Array2d:  # 局部极值点数组
    """
    生成连续趋势标签（在每个极值区间内将价格标准化为0到1）
    
    该函数在每个极值区间内，将价格标准化到0-1的范围内，其中0表示最低点，1表示最高点。
    然后使用反向映射（1减去标准化值）来表示未来的趋势方向：
    - 接近0的值表示将会上涨
    - 接近1的值表示将会下跌
    
    参数说明：
        close: 价格数据的2维数组
        local_extrema: 局部极值点数组，由local_extrema_apply_nb函数生成
    
    返回值：
        与输入形状相同的2维数组，包含连续趋势标签
        - 0: 表示将会下跌（当前价格接近区间最高点）
        - 1: 表示将会上涨（当前价格接近区间最低点）
        - 中间值: 表示上涨或下跌的程度
        - NaN: 极值点之外的区域
    
    算法逻辑：
        1. 在每个极值区间内找到最高价和最低价
        2. 将区间内的所有价格标准化到[0,1]范围
        3. 使用1减去标准化值作为标签
        4. 这样低价格点（接近最低点）得到接近1的标签
        5. 高价格点（接近最高点）得到接近0的标签
    
    使用场景：
        - 连续趋势预测模型的标签生成
        - 价格在趋势区间内的相对位置评估
        - 动态止损止盈点的设置参考
        - 趋势强度的量化评估
    
    示例：
        >>> prices = np.array([[100, 105, 98, 110, 95, 108]]).T
        >>> extrema = local_extrema_apply_nb(prices, pos_th=0.05, neg_th=0.05)
        >>> cont_labels = bn_cont_trend_labels_nb(prices, extrema)
        >>> # 在极值区间内生成0-1的连续趋势标签
    
    优势：
        - 提供比二进制标签更丰富的信息
        - 反映价格在趋势中的相对位置
        - 适用于回归模型的连续标签
    """
    # 初始化输出数组，全部设为NaN
    out = np.full_like(close, np.nan, dtype=np.float64)

    # 遍历每一列（每个资产）
    for col in range(close.shape[1]):
        # 找到所有非零的极值点索引
        idxs = np.flatnonzero(local_extrema[:, col])
        
        # 如果没有找到极值点，跳过当前列
        if idxs.shape[0] == 0:
            continue

        # 遍历相邻的极值点对
        for k in range(1, idxs.shape[0]):
            prev_i = idxs[k - 1]  # 前一个极值点的索引
            next_i = idxs[k]      # 下一个极值点的索引

            # 计算当前区间内的最高价和最低价
            # 注意：这里使用next_i+1是为了包含下一个极值点
            _min = np.min(close[prev_i:next_i + 1, col])
            _max = np.max(close[prev_i:next_i + 1, col])
            
            # 对区间内的每个价格进行标准化处理
            # 使用反向映射：1 - (price - min) / (max - min)
            # 这样最低价格得到标签1，最高价格得到标签0
            out[prev_i:next_i, col] = 1 - (close[prev_i:next_i, col] - _min) / (_max - _min)

    return out  # 返回连续趋势标签数组


@njit(cache=True)  # 使用Numba JIT编译优化
def bn_cont_sat_trend_labels_nb(close: tp.Array2d,  # 输入的价格数组
                                local_extrema: tp.Array2d,  # 局部极值点数组
                                pos_th: tp.MaybeArray[float],  # 正向阈值
                                neg_th: tp.MaybeArray[float],  # 负向阈值
                                flex_2d: bool = True) -> tp.Array2d:  # 是否使用2维灵活索引
    """
    生成带饱和处理的连续趋势标签
    
    该函数类似于bn_cont_trend_labels_nb，但增加了饱和处理机制。当价格变化
    超过设定的阈值时，标签会被设置为饱和值（0或1），否则使用连续的插值。
    
    饱和处理机制：
    - 当预期上涨且价格足够低时，标签饱和为1
    - 当预期下跌且价格足够高时，标签饱和为0
    - 其他情况下使用线性插值
    
    参数说明：
        close: 价格数据的2维数组
        local_extrema: 局部极值点数组
        pos_th: 正向阈值，用于判断饱和条件
        neg_th: 负向阈值，用于判断饱和条件
        flex_2d: 是否使用2维灵活索引
    
    返回值：
        与输入形状相同的2维数组，包含带饱和的连续趋势标签
        - 0: 强烈的下跌信号（饱和状态）
        - 1: 强烈的上涨信号（饱和状态）
        - 中间值: 线性插值的趋势强度
        - NaN: 极值点之外的区域
    
    算法逻辑：
        1. 确定极值区间和趋势方向
        2. 根据趋势方向计算饱和边界
        3. 检查当前价格是否满足饱和条件
        4. 如果满足，设置为饱和值(0或1)
        5. 否则使用线性插值计算标签
    
    使用场景：
        - 需要强化极端信号的趋势预测
        - 动态调整预测信心的模型
        - 风险控制中的极端情况处理
        - 自适应阈值的趋势分析
    
    示例：
        >>> prices = np.array([[100, 105, 98, 110, 95, 108]]).T
        >>> extrema = local_extrema_apply_nb(prices, pos_th=0.05, neg_th=0.05)
        >>> sat_labels = bn_cont_sat_trend_labels_nb(prices, extrema, 0.03, 0.03)
        >>> # 生成带饱和处理的连续趋势标签
    
    优势：
        - 对极端情况提供更强的信号
        - 平衡了连续性和离散性
        - 适应性更强的标签生成
    """
    # 将阈值转换为NumPy数组格式
    pos_th = np.asarray(pos_th)
    neg_th = np.asarray(neg_th)
    
    # 初始化输出数组，全部设为NaN
    out = np.full_like(close, np.nan, dtype=np.float64)

    # 遍历每一列（每个资产）
    for col in range(close.shape[1]):
        # 找到所有非零的极值点索引
        idxs = np.flatnonzero(local_extrema[:, col])
        
        # 如果没有找到极值点，跳过当前列
        if idxs.shape[0] == 0:
            continue

        # 遍历相邻的极值点对
        for k in range(1, idxs.shape[0]):
            prev_i = idxs[k - 1]  # 前一个极值点的索引
            next_i = idxs[k]      # 下一个极值点的索引

            # 获取当前区间的阈值
            _pos_th = abs(flex_select_auto_nb(pos_th, prev_i, col, flex_2d))
            _neg_th = abs(flex_select_auto_nb(neg_th, prev_i, col, flex_2d))
            
            # 验证阈值的有效性
            if _pos_th == 0:
                raise ValueError("Positive threshold cannot be 0")
            if _neg_th == 0:
                raise ValueError("Negative threshold cannot be 0")
                
            # 计算当前区间内的最高价和最低价
            _min = np.min(close[prev_i:next_i + 1, col])
            _max = np.max(close[prev_i:next_i + 1, col])

            # 处理区间内的每个价格点
            for i in range(prev_i, next_i):
                # 判断趋势方向（下一个极值点是否更高）
                if close[next_i, col] > close[prev_i, col]:
                    # 上涨趋势：计算饱和边界
                    _start = _max / (1 + _pos_th)  # 饱和起始价格
                    _end = _min * (1 + _pos_th)    # 饱和结束价格
                    
                    # 检查是否满足饱和条件
                    if _max >= _end and close[i, col] <= _start:
                        # 价格足够低，设置为强烈上涨信号
                        out[i, col] = 1
                    else:
                        # 使用线性插值计算标签
                        out[i, col] = 1 - (close[i, col] - _start) / (_max - _start)
                else:
                    # 下跌趋势：计算饱和边界
                    _start = _min / (1 - _neg_th)  # 饱和起始价格
                    _end = _max * (1 - _neg_th)    # 饱和结束价格
                    
                    # 检查是否满足饱和条件
                    if _min <= _end and close[i, col] >= _start:
                        # 价格足够高，设置为强烈下跌信号
                        out[i, col] = 0
                    else:
                        # 使用线性插值计算标签
                        out[i, col] = 1 - (close[i, col] - _min) / (_start - _min)

    return out  # 返回带饱和处理的连续趋势标签数组


@njit(cache=True)  # 使用Numba JIT编译优化
def pct_trend_labels_nb(close: tp.Array2d,  # 输入的价格数组
                        local_extrema: tp.Array2d,  # 局部极值点数组
                        normalize: bool) -> tp.Array2d:  # 是否使用标准化计算
    """
    生成基于百分比变化的趋势标签
    
    该函数计算每个时间点到下一个极值点的百分比变化，提供了直观的
    收益率预测标签。支持两种计算模式：标准模式和标准化模式。
    
    参数说明：
        close: 价格数据的2维数组
        local_extrema: 局部极值点数组
        normalize: 是否使用标准化计算
                   - True: 使用未来价格作为分母 (future_price - current_price) / future_price
                   - False: 使用当前价格作为分母 (future_price - current_price) / current_price
    
    返回值：
        与输入形状相同的2维数组，包含百分比变化标签
        - 正值: 价格将上涨的百分比
        - 负值: 价格将下跌的百分比
        - NaN: 极值点之外的区域
    
    计算公式：
        - 标准模式 (normalize=False): (P_future - P_current) / P_current
        - 标准化模式 (normalize=True): (P_future - P_current) / P_future
    
    两种模式的差异：
        - 标准模式：传统的收益率计算，直观易懂
        - 标准化模式：对称性更好，适用于某些机器学习模型
    
    使用场景：
        - 回归模型的收益率预测标签
        - 量化交易中的收益率目标设置
        - 风险调整后的收益率分析
        - 投资组合优化中的预期收益
    
    示例：
        >>> prices = np.array([[100, 105, 98, 110, 95, 108]]).T
        >>> extrema = local_extrema_apply_nb(prices, pos_th=0.05, neg_th=0.05)
        >>> pct_labels = pct_trend_labels_nb(prices, extrema, normalize=False)
        >>> # 生成到下一个极值点的百分比变化标签
    
    优势：
        - 提供直观的收益率预测
        - 支持两种计算模式
        - 适用于回归和分类任务
    """
    # 初始化输出数组，全部设为NaN
    out = np.full_like(close, np.nan, dtype=np.float64)

    # 遍历每一列（每个资产）
    for col in range(close.shape[1]):
        # 找到所有非零的极值点索引
        idxs = np.flatnonzero(local_extrema[:, col])
        
        # 如果没有找到极值点，跳过当前列
        if idxs.shape[0] == 0:
            continue

        # 遍历相邻的极值点对
        for k in range(1, idxs.shape[0]):
            prev_i = idxs[k - 1]  # 前一个极值点的索引
            next_i = idxs[k]      # 下一个极值点的索引

            # 计算区间内每个点到下一个极值点的百分比变化
            for i in range(prev_i, next_i):
                # 根据normalize参数选择计算方式
                if close[next_i, col] > close[prev_i, col] and normalize:
                    # 上涨趋势且使用标准化模式
                    # 公式：(未来价格 - 当前价格) / 未来价格
                    out[i, col] = (close[next_i, col] - close[i, col]) / close[next_i, col]
                else:
                    # 下跌趋势或使用标准模式
                    # 公式：(未来价格 - 当前价格) / 当前价格
                    out[i, col] = (close[next_i, col] - close[i, col]) / close[i, col]

    return out  # 返回百分比变化标签数组


@njit(cache=True)  # 使用Numba JIT编译优化
def trend_labels_apply_nb(close: tp.Array2d,  # 输入的价格数组
                          pos_th: tp.MaybeArray[float],  # 正向阈值
                          neg_th: tp.MaybeArray[float],  # 负向阈值
                          mode: int,  # 趋势模式，对应TrendMode枚举
                          flex_2d: bool = True) -> tp.Array2d:  # 是否使用2维灵活索引
    """
    趋势标签生成的统一接口函数
    
    该函数是趋势标签生成的主要入口点，它首先识别局部极值点，然后根据
    指定的模式生成相应的趋势标签。支持多种趋势标签生成模式。
    
    参数说明：
        close: 价格数据的2维数组
        pos_th: 正向（上涨）阈值
        neg_th: 负向（下跌）阈值
        mode: 趋势模式，对应TrendMode枚举值
               - TrendMode.Binary (0): 二进制趋势标签
               - TrendMode.BinaryCont (1): 连续趋势标签
               - TrendMode.BinaryContSat (2): 饱和连续趋势标签
               - TrendMode.PctChange (3): 百分比变化标签
               - TrendMode.PctChangeNorm (4): 标准化百分比变化标签
        flex_2d: 是否使用2维灵活索引
    
    返回值：
        与输入形状相同的2维数组，包含相应模式的趋势标签
    
    算法流程：
        1. 首先调用local_extrema_apply_nb识别局部极值点
        2. 根据mode参数选择相应的标签生成函数
        3. 返回生成的趋势标签
    
    支持的趋势模式：
        - Binary: 简单的0/1二进制标签
        - BinaryCont: 0-1连续标签
        - BinaryContSat: 带饱和的0-1连续标签
        - PctChange: 百分比变化标签
        - PctChangeNorm: 标准化百分比变化标签
    
    使用场景：
        - 统一的趋势标签生成接口
        - 支持多种机器学习模型的需求
        - 策略开发中的标签实验
        - 自动化的标签生成系统
    
    示例：
        >>> import vectorbt as vbt
        >>> prices = np.array([[100, 105, 98, 110, 95, 108]]).T
        >>> 
        >>> # 生成二进制趋势标签
        >>> binary_labels = trend_labels_apply_nb(
        ...     prices, 0.05, 0.05, vbt.labels.enums.TrendMode.Binary
        ... )
        >>> 
        >>> # 生成百分比变化标签
        >>> pct_labels = trend_labels_apply_nb(
        ...     prices, 0.05, 0.05, vbt.labels.enums.TrendMode.PctChange
        ... )
    
    优势：
        - 统一的接口，便于使用
        - 支持多种标签类型
        - 自动处理极值点识别
        - 灵活的参数配置
    """
    # 首先识别局部极值点
    local_extrema = local_extrema_apply_nb(close, pos_th, neg_th, flex_2d)
    
    # 根据模式选择相应的标签生成函数
    if mode == TrendMode.Binary:
        # 二进制趋势标签
        return bn_trend_labels_nb(close, local_extrema)
    if mode == TrendMode.BinaryCont:
        # 连续趋势标签
        return bn_cont_trend_labels_nb(close, local_extrema)
    if mode == TrendMode.BinaryContSat:
        # 饱和连续趋势标签
        return bn_cont_sat_trend_labels_nb(close, local_extrema, pos_th, neg_th, flex_2d)
    if mode == TrendMode.PctChange:
        # 百分比变化标签（标准模式）
        return pct_trend_labels_nb(close, local_extrema, False)
    if mode == TrendMode.PctChangeNorm:
        # 百分比变化标签（标准化模式）
        return pct_trend_labels_nb(close, local_extrema, True)
    
    # 如果模式不被识别，抛出错误
    raise ValueError("Trend mode is not recognized")


@njit(cache=True)  # 使用Numba JIT编译优化
def breakout_labels_nb(close: tp.Array2d,  # 输入的价格数组
                       window: int,  # 未来观察窗口大小
                       pos_th: tp.MaybeArray[float],  # 正向突破阈值
                       neg_th: tp.MaybeArray[float],  # 负向突破阈值
                       wait: int = 1,  # 等待期数
                       flex_2d: bool = True) -> tp.Array2d:  # 是否使用2维灵活索引
    """
    生成价格突破标签
    
    该函数检测在指定的未来时间窗口内，价格是否突破了预设的阈值。
    这是一种常用的技术分析方法，用于识别价格的显著变化。
    
    突破检测机制：
    - 在未来window个时间点内寻找突破
    - 正向突破：价格上涨超过pos_th阈值
    - 负向突破：价格下跌超过neg_th阈值
    - 首次突破原则：一旦检测到突破，立即标记并停止搜索
    
    参数说明：
        close: 价格数据的2维数组
        window: 未来观察窗口大小（时间点数量）
        pos_th: 正向突破阈值
                - 例如0.05表示5%的上涨突破
                - 支持标量或数组格式
        neg_th: 负向突破阈值
                - 例如0.05表示5%的下跌突破
                - 支持标量或数组格式
        wait: 等待期数，避免使用当前时间点
        flex_2d: 是否使用2维灵活索引
    
    返回值：
        与输入形状相同的2维数组，包含突破标签
        - 1: 正向突破（上涨突破）
        - -1: 负向突破（下跌突破）
        - 0: 无突破
    
    算法逻辑：
        1. 对每个时间点，检查未来window个时间点
        2. 计算价格相对于当前价格的变化
        3. 如果上涨超过pos_th，标记为正向突破(1)
        4. 如果下跌超过neg_th，标记为负向突破(-1)
        5. 首次突破原则：检测到突破后立即停止
        6. 如果在窗口内没有突破，标记为无突破(0)
    
    使用场景：
        - 技术分析中的突破信号生成
        - 动量策略的信号标签
        - 短期价格波动预测
        - 止损止盈点的动态设置
        - 波动率突破策略
    
    示例：
        >>> prices = np.array([[100, 105, 98, 110, 95, 108]]).T
        >>> breakout_labels = breakout_labels_nb(
        ...     prices, 
        ...     window=3,      # 未来3个时间点
        ...     pos_th=0.05,   # 5%上涨突破
        ...     neg_th=0.05,   # 5%下跌突破
        ...     wait=1         # 等待1个时间点
        ... )
        >>> # 检测未来3天内是否发生5%的价格突破
    
    特点：
        - 前瞻性检测：使用未来信息，仅用于标签生成
        - 首次突破原则：避免重复计算
        - 灵活阈值设置：支持动态阈值
        - 高效算法：使用Numba编译优化
    
    注意事项：
        - 该函数包含前瞻性偏差，仅用于模型训练
        - 返回的标签数组末尾部分可能没有足够的未来数据
        - 阈值设置需要根据市场特点进行调整
    """
    # 将阈值转换为NumPy数组格式
    pos_th = np.asarray(pos_th)
    neg_th = np.asarray(neg_th)
    
    # 初始化输出数组，全部设为0（无突破）
    out = np.full_like(close, 0, dtype=np.float64)

    # 遍历每一列（每个资产）
    for col in range(close.shape[1]):
        # 遍历每个时间点
        for i in range(close.shape[0]):
            # 获取当前位置的阈值
            _pos_th = abs(flex_select_auto_nb(pos_th, i, col, flex_2d))
            _neg_th = abs(flex_select_auto_nb(neg_th, i, col, flex_2d))

            # 在未来window个时间点内寻找突破
            # 搜索范围：从i+wait到i+window+wait
            for j in range(i + wait, min(i + window + wait, close.shape[0])):
                # 检查正向突破（上涨突破）
                if _pos_th > 0 and close[j, col] >= close[i, col] * (1 + _pos_th):
                    out[i, col] = 1  # 标记为正向突破
                    break  # 首次突破原则，立即停止搜索
                
                # 检查负向突破（下跌突破）
                if _neg_th > 0 and close[j, col] <= close[i, col] * (1 - _neg_th):
                    out[i, col] = -1  # 标记为负向突破
                    break  # 首次突破原则，立即停止搜索

    return out  # 返回突破标签数组
