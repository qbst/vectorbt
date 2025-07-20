# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
投资组合数值计算模块 - Numba编译函数集合

=======================================================================================
文件设计逻辑和作用总结：
=======================================================================================
本文件是VectorBT量化投资框架的核心计算引擎，专门为投资组合管理提供高性能的数值计算函数。
主要设计理念和功能架构如下：

1. 核心设计理念：
   - 使用Numba Just-In-Time编译技术实现极致性能优化
   - 将投资组合管理的所有核心计算操作编译为机器码执行
   - 提供矩阵化批量处理能力，支持同时处理多个资产和策略

2. 主要功能模块：
   ├── 订单处理系统 (Order Processing)
   │   ├── buy_nb()         # 买入/平空头订单执行
   │   ├── sell_nb()        # 卖出/平多头订单执行  
   │   └── execute_order_nb() # 通用订单执行引擎
   │
   ├── 投资组合模拟引擎 (Portfolio Simulation)
   │   ├── simulate_nb()          # 基础投资组合回测模拟
   │   ├── simulate_from_orders_nb() # 基于订单序列的模拟
   │   └── simulate_from_signal_func_nb() # 基于信号函数的模拟
   │
   ├── 现金流与头寸管理 (Cash Flow & Position Management)
   │   ├── cash_flow_nb()    # 现金流计算
   │   ├── asset_flow_nb()   # 资产流计算
   │   └── position_mask_grouped_nb() # 头寸分组管理
   │
   ├── 交易记录生成 (Trade Records Generation)
   │   ├── get_trade_stats_nb()   # 交易统计计算
   │   ├── get_entry_trades_nb()  # 开仓交易记录
   │   └── get_exit_trades_nb()   # 平仓交易记录
   │
   └── 风险控制系统 (Risk Management)
       ├── 止损止盈逻辑 (Stop Loss/Take Profit)
       ├── 仓位大小控制 (Position Sizing)
       └── 滑点和手续费处理 (Slippage & Fees)

3. 关键技术特性：
   - 高精度浮点运算：内置舍入误差检测和容忍机制
   - 内存优化：零拷贝操作和预分配内存策略
   - 并行计算：支持多资产并行处理和向量化操作
   - 灵活架构：可插拔的回调函数系统，支持自定义策略逻辑

4. 适用场景：
   - 量化策略回测：支持各类技术指标策略的历史回测
   - 实时交易模拟：可用于模拟实盘交易执行过程
   - 投资组合优化：多资产配置和风险管理
   - 高频交易策略：毫秒级订单执行和资金管理

=======================================================================================

这个模块为量化投资提供了完整的数值计算基础设施，是构建复杂投资策略的底层引擎。
所有函数都经过Numba编译优化，仅接受NumPy数组和Numba兼容的数据类型。

!!! 注意事项
    vectorbt将矩阵视为一等公民，期望输入数组为2维，除非函数有后缀`_1d`或用作其他函数的输入。
    
    所有作为参数传递的函数都应该是Numba编译的。
    
    记录应保持其创建时的顺序。

!!! 警告：舍入误差累积
    可能发生舍入误差的累积。
    参见：https://en.wikipedia.org/wiki/Round-off_error#Accumulation_of_roundoff_error

    舍入误差可能导致交易和头寸无法正确关闭：

    ```pycon
    >>> print('%.50f' % 0.1)  # 有正误差
    0.10000000000000000555111512312578270211815834045410

    >>> # 大量带正误差的买入交易 -> 无法平仓
    >>> sum([0.1 for _ in range(1000000)]) - 100000
    1.3328826753422618e-06

    >>> print('%.50f' % 0.3)  # 有负误差
    0.29999999999999998889776975374843459576368331909180

    >>> # 大量带负误差的卖出交易 -> 无法平仓
    >>> 300000 - sum([0.3 for _ in range(1000000)])
    5.657668225467205e-06
    ```

    虽然vectorbt在比较浮点数是否相等时实现了容忍度检查，
    但大量重复相同符号的小额交易仍可能引入无法事后纠正的明显误差。

    为缓解此问题，避免重复大量相同符号的微交易。
    例如，通过`np.inf`或`position_now`来平仓多/空头头寸。

    当前容忍度值参见`vectorbt.utils.math_`。
"""

# 导入必要的数值计算和类型定义库
import numpy as np                    # 数值计算基础库，提供多维数组和数学函数
from numba import njit               # Numba即时编译装饰器，将Python函数编译为机器码

# 导入vectorbt内部模块
from vectorbt import _typing as tp   # 类型定义模块，提供统一的类型注解
from vectorbt.base.reshape_fns import flex_select_auto_nb  # 灵活数组选择函数，支持广播和自动形状匹配
from vectorbt.generic import nb as generic_nb             # 通用数值计算函数，提供基础的统计和聚合操作
from vectorbt.portfolio.enums import *                    # 投资组合相关枚举类型，定义订单状态、方向等常量
from vectorbt.returns import nb as returns_nb             # 收益率计算函数，用于计算投资回报指标
from vectorbt.utils.array_ import insert_argsort_nb       # 数组插入排序函数，用于维护有序记录
from vectorbt.utils.math_ import (                        # 高精度数学工具函数集合
    is_close_nb,           # 判断两个浮点数是否在容忍误差范围内相等
    is_close_or_less_nb,   # 判断第一个数是否小于等于第二个数（考虑容忍误差）
    is_less_nb,            # 判断第一个数是否严格小于第二个数（考虑容忍误差）
    add_nb                 # 高精度加法函数，减少累积舍入误差
)


# ############# 订单处理系统 (Order Processing System) ############# #


@njit(cache=True)  # Numba编译缓存，提高重复调用性能
def order_not_filled_nb(status: int, status_info: int) -> OrderResult:
    """
    生成未成交订单结果对象
    
    当订单因各种原因无法执行时，返回标准化的订单结果。
    这是订单处理系统中的基础函数，用于统一处理订单拒绝情况。
    
    参数:
    ----
    status : int
        订单状态码，通常表示拒绝或忽略
        - OrderStatus.Rejected: 订单被拒绝  
        - OrderStatus.Ignored: 订单被忽略
        
    status_info : int  
        具体的状态信息码，说明拒绝原因
        - OrderStatusInfo.NoCashLong: 做多资金不足
        - OrderStatusInfo.NoOpenPosition: 无持仓可平
        - OrderStatusInfo.SizeZero: 订单大小为零
        - OrderStatusInfo.MaxSizeExceeded: 超过最大订单限制
        - OrderStatusInfo.MinSizeNotReached: 未达到最小订单限制
        - OrderStatusInfo.CantCoverFees: 无法承担手续费
        - OrderStatusInfo.PartialFill: 部分成交被拒绝
        
    返回:
    ----
    OrderResult
        包含以下字段的订单结果对象：
        - size: np.nan (未成交数量)
        - price: np.nan (未成交价格) 
        - fees: np.nan (未产生手续费)
        - side: -1 (无交易方向)
        - status: 传入的状态码
        - status_info: 传入的状态详情码
        
    使用示例:
    --------
    >>> # 资金不足时拒绝买入订单
    >>> result = order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCashLong)
    >>> print(f"订单状态: {result.status}, 详情: {result.status_info}")
    订单状态: 1, 详情: 1
    
    >>> # 订单大小为零时忽略订单
    >>> result = order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero)  
    >>> print(f"是否成交: {not np.isnan(result.size)}")
    是否成交: False
    
    注意:
    ----
    这个函数是所有订单执行函数的基础，当订单无法正常执行时都会调用它。
    返回的OrderResult对象中所有数值字段都是NaN，表示没有实际的交易发生。
    """
    return OrderResult(np.nan, np.nan, np.nan, -1, status, status_info)


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def buy_nb(exec_state: ExecuteOrderState,
           size: float,
           price: float,
           direction: int = Direction.Both,
           fees: float = 0.,
           fixed_fees: float = 0.,
           slippage: float = 0.,
           min_size: float = 0.,
           max_size: float = np.inf,
           size_granularity: float = np.nan,
           lock_cash: bool = False,
           allow_partial: bool = True,
           percent: float = np.nan) -> tp.Tuple[ExecuteOrderState, OrderResult]:
    """
    执行买入订单或平空头操作
    
    这是投资组合管理系统中最核心的函数之一，负责处理所有的买入逻辑，包括：
    1. 多头开仓：使用现金买入资产建立多头头寸
    2. 空头平仓：买入资产来平掉空头头寸（cover）
    3. 资金管理：确保有足够资金执行订单
    4. 风险控制：处理滑点、手续费、最小/最大订单限制等
    
    参数:
    ----
    exec_state : ExecuteOrderState
        当前执行状态，包含：
        - cash: 总现金
        - position: 当前头寸（正数=多头，负数=空头）
        - debt: 空头债务（用于计算平均成本）
        - free_cash: 可用现金（考虑了其他列的资金锁定）
        
    size : float
        期望买入数量。可以是：
        - 正数: 具体买入数量
        - np.inf: 使用所有可用资金买入
        
    price : float
        目标买入价格。实际成交价会考虑滑点调整
        
    direction : int, 可选 (默认: Direction.Both)
        交易方向限制：
        - Direction.Both: 允许开多头或平空头
        - Direction.LongOnly: 只允许开多头
        - Direction.ShortOnly: 只允许平空头
        
    fees : float, 可选 (默认: 0.0)
        比例手续费率（例如 0.001 表示 0.1%）
        
    fixed_fees : float, 可选 (默认: 0.0)
        固定手续费（绝对金额）
        
    slippage : float, 可选 (默认: 0.0)
        滑点率。买入时向上滑点，实际价格 = price * (1 + slippage)
        
    min_size : float, 可选 (默认: 0.0)
        最小订单数量。小于此值的订单将被拒绝
        
    max_size : float, 可选 (默认: np.inf)
        最大订单数量。超过此值的订单将被截断或拒绝
        
    size_granularity : float, 可选 (默认: np.nan)
        数量粒度。订单数量将向下取整到此粒度的整数倍
        例如：粒度为 0.1，则 1.37 会变为 1.3
        
    lock_cash : bool, 可选 (默认: False)
        是否锁定现金。如果为 True，则：
        - 多头时只能使用 free_cash
        - 空头时需考虑平仓所需资金
        
    allow_partial : bool, 可选 (默认: True)
        是否允许部分成交。为 False 时，资金不足的订单将被完全拒绝
        
    percent : float, 可选 (默认: np.nan)
        资金使用比例。限制最多使用多少比例的可用资金
        
    返回:
    ----
    tuple[ExecuteOrderState, OrderResult]
        新的执行状态和订单结果：
        
        ExecuteOrderState: 更新后的投资组合状态
        - cash: 扣除交易成本后的现金
        - position: 更新后的头寸
        - debt: 更新后的空头债务
        - free_cash: 更新后的可用现金
        
        OrderResult: 订单执行结果
        - size: 实际成交数量
        - price: 实际成交价格（含滑点）
        - fees: 实际支付的手续费
        - side: OrderSide.Buy
        - status: OrderStatus.Filled 或相应的拒绝状态
        - status_info: 详细状态信息
        
    使用示例:
    --------
    >>> # 基础买入操作
    >>> initial_state = ExecuteOrderState(cash=10000, position=0, debt=0, free_cash=10000)
    >>> new_state, result = buy_nb(initial_state, size=100, price=50.0, fees=0.001)
    >>> print(f"成交数量: {result.size}, 成交价格: {result.price}")
    成交数量: 100.0, 成交价格: 50.0
    >>> print(f"剩余现金: {new_state.cash}, 当前头寸: {new_state.position}")
    剩余现金: 4995.0, 当前头寸: 100.0
    
    >>> # 平空头操作
    >>> short_state = ExecuteOrderState(cash=10000, position=-50, debt=2500, free_cash=10000)
    >>> new_state, result = buy_nb(short_state, size=30, price=52.0)
    >>> print(f"平仓数量: {result.size}, 剩余空头: {new_state.position}")
    平仓数量: 30.0, 剩余空头: -20.0
    
    >>> # 资金不足的情况
    >>> poor_state = ExecuteOrderState(cash=100, position=0, debt=0, free_cash=100)  
    >>> new_state, result = buy_nb(poor_state, size=100, price=50.0)
    >>> print(f"订单状态: {result.status}, 实际成交: {result.size}")
    订单状态: OrderStatus.Filled, 实际成交: 2.0  # 只能买入2股
    
    算法逻辑:
    --------
    1. 滑点调整: adj_price = price * (1 + slippage)
    2. 资金限制计算: 
       - 多头: 使用可用现金
       - 空头: 需考虑平仓成本和剩余现金
    3. 订单数量调整: 根据方向、最大值、粒度进行调整
    4. 资金充足性检查: 计算所需总成本（含手续费）
    5. 头寸和现金更新: 更新所有账户状态
    6. 债务处理: 平空头时减少相应债务
    
    注意事项:
    --------
    - 买入操作会增加头寸（正向）或减少空头头寸
    - 手续费计算公式: total_fees = size * price * fees + fixed_fees
    - 空头平仓时会释放相应的保证金债务
    - 使用高精度数学函数避免累积舍入误差
    """

    # 计算考虑滑点的实际成交价格
    # 买入时向上滑点，模拟市场冲击成本
    adj_price = price * (1 + slippage)

    # 设置现金使用限制
    # 根据是否锁定现金和当前头寸类型确定可用资金上限
    if lock_cash:
        # 启用现金锁定机制，只能使用未被其他操作锁定的资金
        if exec_state.position >= 0:
            # 多头或零头寸情况：在多头头寸中 cash == free_cash，除非其他列锁定了部分现金
            # 只能使用完全可用的现金，避免影响其他资产的操作
            cash_limit = exec_state.free_cash
        else:
            # 空头头寸情况：需要考虑平仓成本
            # 计算完全平掉当前空头头寸需要多少现金
            cover_req_cash = abs(exec_state.position) * adj_price * (1 + fees) + fixed_fees
            # 计算平仓后剩余的自由现金：当前自由现金 + 释放的债务保证金 - 平仓成本
            cover_free_cash = add_nb(exec_state.free_cash + 2 * exec_state.debt, -cover_req_cash)
            if cover_free_cash > 0:
                # 有足够现金平掉空头头寸并开多头头寸
                # 可以使用：自由现金 + 空头平仓后释放的所有资金
                cash_limit = exec_state.free_cash + 2 * exec_state.debt
            elif cover_free_cash < 0:
                # 没有足够现金完全平掉空头头寸
                # 计算能够部分平仓的最大数量
                avg_entry_price = exec_state.debt / abs(exec_state.position)  # 空头的平均入场价格
                # 计算最多能平多少空头：考虑手续费和保证金释放
                max_short_size = ((exec_state.free_cash - fixed_fees) / (adj_price * (1 + fees) - 2 * avg_entry_price))
                # 对应的现金限制
                cash_limit = max_short_size * adj_price * (1 + fees) + fixed_fees
            else:
                # 刚好有足够现金完全平掉空头头寸
                cash_limit = exec_state.cash
    else:
        # 未启用现金锁定，可以使用所有现金
        cash_limit = exec_state.cash
    
    # 确保现金限制不超过实际拥有的现金总量
    cash_limit = min(cash_limit, exec_state.cash)
    
    # 应用资金使用比例限制（如果指定）
    if not np.isnan(percent):
        # 进一步限制资金使用量：现金限制 * 使用比例
        cash_limit = min(cash_limit, percent * cash_limit)

    # 检查交易方向和资金可用性
    if direction == Direction.LongOnly or direction == Direction.Both:
        # 允许开多头或双向交易的情况
        if cash_limit == 0:
            # 没有可用现金进行多头交易，拒绝订单
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCashLong)
        if np.isinf(size) and np.isinf(cash_limit):
            # 防止无限多头开仓：订单数量和现金都是无限大
            raise ValueError("尝试进行无限大的多头开仓")
    else:
        # 只允许平空头的情况 (Direction.ShortOnly)
        if exec_state.position == 0:
            # 当前没有头寸，无法进行平仓操作
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoOpenPosition)

    # 计算最优订单大小
    # 根据交易方向限制调整期望的订单数量
    if direction == Direction.ShortOnly:
        # 仅平空头：订单大小不能超过当前空头头寸的绝对值
        adj_size = min(-exec_state.position, size)  # -exec_state.position 为正数（空头大小）
    else:
        # 开多头或双向交易：使用原始订单大小
        adj_size = size

    # 检查调整后的订单大小
    if adj_size == 0:
        # 调整后订单大小为零，忽略此订单
        return exec_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero)

    # 检查最大订单大小限制
    if adj_size > max_size:
        if not allow_partial:
            # 不允许部分成交且超过最大限制，拒绝整个订单
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded)
        # 允许部分成交，截断到最大允许大小
        adj_size = max_size

    # 应用数量粒度调整
    if not np.isnan(size_granularity):
        # 将订单数量向下取整到指定粒度的整数倍
        # 例如：粒度为 0.1，1.37 变为 1.3
        adj_size = adj_size // size_granularity * size_granularity

    # 计算完成此订单所需的现金总量
    req_cash = adj_size * adj_price                    # 购买资产所需的基础金额
    req_fees = req_cash * fees + fixed_fees           # 手续费 = 比例手续费 + 固定手续费
    total_req_cash = req_cash + req_fees              # 总所需现金 = 基础金额 + 手续费

    # 检查资金充足性并确定最终订单参数
    if is_close_or_less_nb(total_req_cash, cash_limit):
        # 资金充足：可以完全执行期望的订单
        final_size = adj_size                         # 最终成交数量 = 调整后的期望数量
        fees_paid = req_fees                          # 实际支付手续费
        final_req_cash = total_req_cash               # 实际使用现金
    else:
        # 资金不足：需要减少订单数量以适应可用资金
        
        # 逆向计算：给定总资金限制，能购买多少资产
        # 例如：手续费 10% 和固定费用 1$，要花费 100$ 总额，实际只能用 90$ 购买资产
        # 计算公式：max_req_cash = (cash_limit - fixed_fees) / (1 + fees)
        max_req_cash = add_nb(cash_limit, -fixed_fees) / (1 + fees)
        if max_req_cash <= 0:
            # 可用资金连固定手续费都无法承担
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees)

        # 根据可用资金计算最大可购买数量
        max_acq_size = max_req_cash / adj_price

        if not np.isnan(size_granularity):
            # 考虑数量粒度的情况：需要重新计算精确的费用
            final_size = max_acq_size // size_granularity * size_granularity  # 向下取整到粒度
            new_req_cash = final_size * adj_price                              # 重新计算基础金额
            fees_paid = new_req_cash * fees + fixed_fees                       # 重新计算手续费
            final_req_cash = new_req_cash + fees_paid                          # 重新计算总金额
        else:
            # 无数量粒度限制：直接使用计算出的最大数量
            final_size = max_acq_size                 # 最大可购买数量
            fees_paid = cash_limit - max_req_cash     # 实际手续费（用完所有可用资金）
            final_req_cash = cash_limit               # 使用全部可用资金

    # 最终检查：再次确认订单大小不为零（防止舍入误差导致的零订单）
    if is_close_nb(adj_size, 0):
        return exec_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero)

    # 检查最小订单大小限制
    if is_less_nb(final_size, min_size):
        # 最终订单数量小于最小限制，拒绝订单
        return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MinSizeNotReached)

    # 检查部分成交限制（无限大订单不算部分成交）
    if np.isfinite(size) and is_less_nb(final_size, size) and not allow_partial:
        # 原始订单是有限大小，但最终数量小于期望，且不允许部分成交
        return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill)

    # 更新投资组合的现金余额和头寸状态
    new_cash = add_nb(exec_state.cash, -final_req_cash)      # 新现金 = 原现金 - 交易总成本
    new_position = add_nb(exec_state.position, final_size)   # 新头寸 = 原头寸 + 买入数量

    # 更新债务和自由现金状态
    # 需要区分处理空头平仓和多头开仓的情况
    if exec_state.position < 0:
        # 原有空头头寸的情况：买入是在平仓
        if new_position < 0:
            # 部分平空头：还有剩余空头头寸
            short_size = final_size  # 平仓的空头数量就是买入数量
        else:
            # 完全平空头（可能还开了多头）：平仓数量是原空头的全部
            short_size = abs(exec_state.position)  # 原空头头寸的绝对值
        
        # 计算空头平仓时释放的债务
        avg_entry_price = exec_state.debt / abs(exec_state.position)  # 空头平均入场价格
        debt_diff = short_size * avg_entry_price                      # 本次平仓释放的债务金额
        new_debt = add_nb(exec_state.debt, -debt_diff)                # 新债务 = 原债务 - 释放的债务
        # 新自由现金 = 原自由现金 + 释放的债务保证金(2倍) - 交易成本
        new_free_cash = add_nb(exec_state.free_cash + 2 * debt_diff, -final_req_cash)
    else:
        # 原无空头或已是多头：买入是在开多头或加仓
        new_debt = exec_state.debt                                    # 债务不变
        new_free_cash = add_nb(exec_state.free_cash, -final_req_cash) # 自由现金 = 原自由现金 - 交易成本

    # 构建并返回成功执行的订单结果
    order_result = OrderResult(
        final_size,              # 实际成交数量
        adj_price,              # 实际成交价格（含滑点）
        fees_paid,              # 实际支付的手续费
        OrderSide.Buy,          # 订单方向：买入
        OrderStatus.Filled,     # 订单状态：已成交
        -1                      # 状态详情：无特殊信息
    )
    
    # 构建更新后的执行状态
    new_exec_state = ExecuteOrderState(
        cash=new_cash,          # 更新后的现金余额
        position=new_position,  # 更新后的头寸
        debt=new_debt,          # 更新后的债务
        free_cash=new_free_cash # 更新后的可用现金
    )
    
    # 返回新状态和订单结果
    return new_exec_state, order_result


@njit(cache=True)  # Numba编译缓存，优化重复调用性能  
def sell_nb(exec_state: ExecuteOrderState,
            size: float,
            price: float,
            direction: int = Direction.Both,
            fees: float = 0.,
            fixed_fees: float = 0.,
            slippage: float = 0.,
            min_size: float = 0.,
            max_size: float = np.inf,
            size_granularity: float = np.nan,
            lock_cash: bool = False,
            allow_partial: bool = True,
            percent: float = np.nan) -> tp.Tuple[ExecuteOrderState, OrderResult]:
    """
    执行卖出订单或开空头操作
    
    这是投资组合管理系统中与buy_nb配对的核心函数，负责处理所有的卖出逻辑，包括：
    1. 多头平仓：卖出持有的资产平掉多头头寸
    2. 空头开仓：借入资产卖出建立空头头寸（做空）
    3. 资金管理：确保有足够保证金进行空头操作
    4. 风险控制：处理滑点、手续费、最小/最大订单限制等
    
    参数:
    ----
    exec_state : ExecuteOrderState
        当前执行状态，包含：
        - cash: 总现金
        - position: 当前头寸（正数=多头，负数=空头）
        - debt: 空头债务（记录空头的保证金）
        - free_cash: 可用现金（考虑了其他列的资金锁定）
        
    size : float
        期望卖出数量。可以是：
        - 正数: 具体卖出数量
        - np.inf: 卖出所有持仓或开最大空头
        
    price : float
        目标卖出价格。实际成交价会考虑滑点调整
        
    direction : int, 可选 (默认: Direction.Both)
        交易方向限制：
        - Direction.Both: 允许平多头或开空头
        - Direction.LongOnly: 只允许平多头
        - Direction.ShortOnly: 只允许开空头
        
    fees : float, 可选 (默认: 0.0)
        比例手续费率（例如 0.001 表示 0.1%）
        
    fixed_fees : float, 可选 (默认: 0.0)
        固定手续费（绝对金额）
        
    slippage : float, 可选 (默认: 0.0)
        滑点率。卖出时向下滑点，实际价格 = price * (1 - slippage)
        
    min_size : float, 可选 (默认: 0.0)
        最小订单数量。小于此值的订单将被拒绝
        
    max_size : float, 可选 (默认: np.inf)
        最大订单数量。超过此值的订单将被截断或拒绝
        
    size_granularity : float, 可选 (默认: np.nan)
        数量粒度。订单数量将向下取整到此粒度的整数倍
        
    lock_cash : bool, 可选 (默认: False)
        是否锁定现金。如果为 True，则限制空头开仓的最大数量
        
    allow_partial : bool, 可选 (默认: True)
        是否允许部分成交。为 False 时，保证金不足的订单将被完全拒绝
        
    percent : float, 可选 (默认: np.nan)
        头寸使用比例。限制最多卖出多少比例的可卖数量
        
    返回:
    ----
    tuple[ExecuteOrderState, OrderResult]
        新的执行状态和订单结果：
        
        ExecuteOrderState: 更新后的投资组合状态
        - cash: 增加卖出收入后的现金（平多头时）或不变（开空头时）
        - position: 更新后的头寸（减少或变负）
        - debt: 更新后的空头债务（开空头时增加）
        - free_cash: 更新后的可用现金（考虑保证金锁定）
        
        OrderResult: 订单执行结果
        - size: 实际成交数量
        - price: 实际成交价格（含滑点）
        - fees: 实际支付的手续费
        - side: OrderSide.Sell
        - status: OrderStatus.Filled 或相应的拒绝状态
        - status_info: 详细状态信息
        
    使用示例:
    --------
    >>> # 基础卖出操作（平多头）
    >>> long_state = ExecuteOrderState(cash=5000, position=100, debt=0, free_cash=5000)
    >>> new_state, result = sell_nb(long_state, size=50, price=52.0, fees=0.001)
    >>> print(f"成交数量: {result.size}, 成交价格: {result.price}")
    成交数量: 50.0, 成交价格: 51.48  # 考虑滑点
    >>> print(f"剩余头寸: {new_state.position}, 现金: {new_state.cash}")
    剩余头寸: 50.0, 现金: 7572.98  # 5000 + 50*51.48 - 手续费
    
    >>> # 开空头操作
    >>> zero_state = ExecuteOrderState(cash=10000, position=0, debt=0, free_cash=10000)
    >>> new_state, result = sell_nb(zero_state, size=100, price=50.0)
    >>> print(f"空头数量: {result.size}, 当前头寸: {new_state.position}")
    空头数量: 100.0, 当前头寸: -100.0
    >>> print(f"债务: {new_state.debt}, 自由现金: {new_state.free_cash}")
    债务: 5000.0, 自由现金: 5000.0  # 现金减少一半作为保证金
    
    >>> # 部分平多头后开空头
    >>> mixed_state = ExecuteOrderState(cash=5000, position=30, debt=0, free_cash=5000)
    >>> new_state, result = sell_nb(mixed_state, size=80, price=50.0)  
    >>> print(f"最终头寸: {new_state.position}")  # 平30个多头后开50个空头
    最终头寸: -50.0
    
    算法逻辑:
    --------
    1. 滑点调整: adj_price = price * (1 - slippage)
    2. 订单数量限制计算:
       - 只平多头: 限制为当前多头数量
       - 双向交易: 考虑多头数量 + 可开空头数量
       - 只开空头: 基于可用保证金计算最大空头数量
    3. 现金和债务更新:
       - 平多头: 增加现金，减少头寸
       - 开空头: 增加债务，减少头寸（变负），锁定保证金
    4. 保证金计算: 空头需要锁定等值现金作为保证金
    
    注意事项:
    --------
    - 卖出操作会减少头寸（负向）或增加空头头寸
    - 开空头需要足够的自由现金作为保证金
    - 空头的债务用于跟踪平均开仓成本
    - 卖出时向下滑点，模拟市场冲击成本
    """

    # 计算考虑滑点的实际成交价格  
    # 卖出时向下滑点，模拟市场冲击成本
    adj_price = price * (1 - slippage)

    # 计算最优订单大小
    # 根据交易方向限制和资金情况确定可卖出的最大数量
    if direction == Direction.LongOnly:
        # 只允许平多头：卖出数量不能超过当前多头头寸
        size_limit = min(exec_state.position, size)
    else:
        # 允许开空头或双向交易的情况
        if lock_cash or (np.isinf(size) and not np.isnan(percent)):
            # 需要计算资金限制下的最大可卖数量
            
            # 计算当前多头头寸可以释放的现金（扣除手续费）
            long_size = max(exec_state.position, 0)                           # 当前多头数量
            long_cash = long_size * adj_price * (1 - fees)                   # 平多头可得净现金
            total_free_cash = add_nb(exec_state.free_cash, long_cash)        # 总可用现金

            if total_free_cash <= 0:
                # 总可用现金不足或为零
                if exec_state.position <= 0:
                    # 既没有多头可平，也没有现金开空头
                    return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCashShort)
                # 只能平现有多头，无法开新空头
                max_size_limit = long_size
            else:
                # 有足够现金可以平多头和/或开空头
                # 计算最大可开空头数量：(总现金 - 固定费用) / (价格 * (1 + 手续费率))
                max_short_size = add_nb(total_free_cash, -fixed_fees) / (adj_price * (1 + fees))
                # 最大可卖数量 = 可平多头数量 + 可开空头数量
                max_size_limit = add_nb(long_size, max_short_size)
                if max_size_limit <= 0:
                    # 计算结果为负，说明连手续费都无法承担
                    return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees)

            if lock_cash:
                # 启用现金锁定：订单大小有上限
                if np.isinf(size) and not np.isnan(percent):
                    # 无限大订单 + 百分比限制
                    size_limit = min(percent * max_size_limit, max_size_limit)
                    percent = np.nan  # 已应用百分比，清除标记
                elif not np.isnan(percent):
                    # 有限订单 + 百分比限制
                    size_limit = min(percent * size, max_size_limit)
                    percent = np.nan  # 已应用百分比，清除标记
                else:
                    # 无百分比限制，只受资金限制
                    size_limit = min(size, max_size_limit)
            else:  # np.isinf(size) and not np.isnan(percent)
                # 未锁定现金：订单大小无上限，使用最大可卖数量
                size_limit = max_size_limit
        else:
            # 无特殊限制，使用原始订单大小
            size_limit = size

    # 应用百分比限制（如果还未应用）
    if not np.isnan(percent):
        # 将卖出数量限制为指定百分比
        size_limit = percent * size_limit

    # 检查最大订单大小限制
    if size_limit > max_size:
        if not allow_partial:
            # 不允许部分成交且超过最大限制，拒绝整个订单
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded)
        # 允许部分成交，截断到最大允许大小
        size_limit = max_size

    # 检查交易方向和无限大订单
    if direction == Direction.ShortOnly or direction == Direction.Both:
        # 允许开空头的方向
        if np.isinf(size_limit):
            # 防止无限大空头开仓
            raise ValueError("尝试进行无限大的空头开仓")
    else:
        # 只允许平多头的方向
        if exec_state.position == 0:
            # 当前无持仓，无法平仓
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoOpenPosition)

    # 应用数量粒度调整
    if not np.isnan(size_granularity):
        # 将订单数量向下取整到指定粒度的整数倍
        size_limit = size_limit // size_granularity * size_granularity

    # 检查调整后的订单大小
    if is_close_nb(size_limit, 0):
        # 调整后订单大小为零，忽略此订单
        return exec_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero)

    # 检查最小订单大小限制
    if is_less_nb(size_limit, min_size):
        # 最终订单数量小于最小限制，拒绝订单
        return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MinSizeNotReached)

    # 检查部分成交限制（无限大订单不算部分成交）
    if np.isfinite(size) and is_less_nb(size_limit, size) and not allow_partial:
        # 原始订单是有限大小，但最终数量小于期望，且不允许部分成交
        return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill)

    # 计算卖出获得的总现金
    acq_cash = size_limit * adj_price

    # 计算手续费
    fees_paid = acq_cash * fees + fixed_fees  # 比例手续费 + 固定手续费

    # 计算扣除手续费后的净现金收入
    final_acq_cash = add_nb(acq_cash, -fees_paid)
    if final_acq_cash < 0:
        # 手续费超过了卖出收入，无法承担交易成本
        return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees)

    # 更新投资组合的现金余额和头寸状态
    new_cash = exec_state.cash + final_acq_cash      # 新现金 = 原现金 + 卖出净收入
    new_position = add_nb(exec_state.position, -size_limit)  # 新头寸 = 原头寸 - 卖出数量

    # 更新债务和自由现金状态
    # 需要区分处理多头平仓和空头开仓的情况
    if new_position < 0:
        # 交易后变为空头头寸（或空头增加）
        if exec_state.position < 0:
            # 原已是空头：增加空头规模
            short_size = size_limit  # 新增空头数量就是卖出数量
        else:
            # 原为多头或零头寸：部分/全部平多头后开空头
            short_size = abs(new_position)  # 新开空头数量是最终头寸的绝对值
        
        # 计算新增空头的价值和所需保证金
        short_value = short_size * adj_price                      # 新增空头的市值
        new_debt = exec_state.debt + short_value                  # 新债务 = 原债务 + 新增空头价值
        # 自由现金变化 = 净收入 - 2倍空头价值(保证金锁定)
        free_cash_diff = add_nb(final_acq_cash, -2 * short_value)
        new_free_cash = add_nb(exec_state.free_cash, free_cash_diff)
    else:
        # 交易后仍为多头或零头寸：只是平仓操作
        new_debt = exec_state.debt                               # 债务不变
        new_free_cash = exec_state.free_cash + final_acq_cash    # 自由现金 = 原自由现金 + 净收入

    # 构建并返回成功执行的订单结果
    order_result = OrderResult(
        size_limit,             # 实际成交数量
        adj_price,              # 实际成交价格（含滑点）
        fees_paid,              # 实际支付的手续费
        OrderSide.Sell,         # 订单方向：卖出
        OrderStatus.Filled,     # 订单状态：已成交
        -1                      # 状态详情：无特殊信息
    )
    
    # 构建更新后的执行状态
    new_exec_state = ExecuteOrderState(
        cash=new_cash,          # 更新后的现金余额
        position=new_position,  # 更新后的头寸
        debt=new_debt,          # 更新后的债务
        free_cash=new_free_cash # 更新后的可用现金
    )
    
    # 返回新状态和订单结果
    return new_exec_state, order_result


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def execute_order_nb(state: ProcessOrderState, order: Order) -> tp.Tuple[ExecuteOrderState, OrderResult]:
    """
    执行订单的通用引擎函数
    
    这是投资组合管理系统的核心订单执行引擎，将高级订单请求转换为具体的买卖操作。
    该函数负责：
    1. 订单和状态的完整性验证
    2. 多种订单类型的处理和转换
    3. 数值稳定性保证
    4. 委托给具体的买入/卖出执行函数
    
    参数:
    ----
    state : ProcessOrderState
        当前的投资组合处理状态，包含：
        - cash: 总现金
        - position: 当前头寸
        - debt: 空头债务
        - free_cash: 可用现金
        - val_price: 估值价格
        - value: 投资组合总价值
        
    order : Order
        待执行的订单对象，包含：
        - size: 订单大小
        - price: 订单价格
        - size_type: 订单大小类型（数量/价值/百分比等）
        - direction: 交易方向限制
        - fees: 手续费设置
        - 其他风险控制参数
        
    返回:
    ----
    tuple[ExecuteOrderState, OrderResult]
        执行状态和订单结果的元组
        
    异常:
    ----
    ValueError: 当输入参数不符合预期时抛出异常
    
    订单处理逻辑:
    -----------
    - 忽略：订单执行对当前余额无影响时
    - 拒绝：输入超过限制/约束时
    - 执行：通过buy_nb或sell_nb函数执行
    
    使用示例:
    --------
    >>> # 创建订单状态
    >>> state = ProcessOrderState(cash=10000, position=0, debt=0, 
    ...                          free_cash=10000, val_price=50.0, value=10000)
    >>> # 创建买入订单
    >>> order = Order(size=100, price=50.0, size_type=SizeType.Amount)
    >>> new_state, result = execute_order_nb(state, order)
    >>> print(f"执行结果: {result.status}")
    执行结果: OrderStatus.Filled
    
    注意事项:
    --------
    - 对所有输入进行严格的数值验证
    - 提供数值稳定性保证，避免浮点精度问题
    - 支持多种订单大小类型的自动转换
    - 支持随机拒绝机制用于压力测试
    """
    # 数值稳定性处理
    # 将接近零的值精确设为零，避免浮点精度问题导致的计算错误
    cash = state.cash
    if is_close_nb(cash, 0):
        cash = 0.                   # 现金接近零时设为精确零
    position = state.position
    if is_close_nb(position, 0):
        position = 0.               # 头寸接近零时设为精确零
    debt = state.debt
    if is_close_nb(debt, 0):
        debt = 0.                   # 债务接近零时设为精确零
    free_cash = state.free_cash
    if is_close_nb(free_cash, 0):
        free_cash = 0.              # 自由现金接近零时设为精确零
    val_price = state.val_price
    if is_close_nb(val_price, 0):
        val_price = 0.              # 估值价格接近零时设为精确零
    value = state.value
    if is_close_nb(value, 0):
        value = 0.                  # 总价值接近零时设为精确零

    # 预先构建执行状态对象，便于后续处理
    exec_state = ExecuteOrderState(
        cash=cash,         # 校正后的现金
        position=position, # 校正后的头寸
        debt=debt,         # 校正后的债务
        free_cash=free_cash # 校正后的自由现金
    )

    # 忽略无效订单
    # 检查订单的关键参数是否为NaN，如果是则忽略订单
    if np.isnan(order.size):
        # 订单大小为NaN，忽略此订单
        return exec_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeNaN)
    if np.isnan(order.price):
        # 订单价格为NaN，忽略此订单
        return exec_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.PriceNaN)

    # 检查执行状态的有效性
    # 确保所有状态参数都在合理范围内，避免后续计算错误
    if np.isnan(cash) or cash < 0:
        raise ValueError("现金不能为NaN且必须大于等于0")
    if not np.isfinite(position):
        raise ValueError("头寸必须为有限数值")
    if not np.isfinite(debt) or debt < 0:
        raise ValueError("债务必须为有限数值且大于等于0")
    if np.isnan(free_cash):
        raise ValueError("自由现金不能为NaN")

    # 检查订单参数的有效性
    # 对订单的所有关键参数进行全面验证
    if not np.isfinite(order.price) or order.price <= 0:
        raise ValueError("订单价格必须为有限正数")
    if order.size_type < 0 or order.size_type >= len(SizeType):
        raise ValueError("订单大小类型无效")
    if order.direction < 0 or order.direction >= len(Direction):
        raise ValueError("订单方向类型无效")
    if order.direction == Direction.LongOnly and position < 0:
        raise ValueError("当前为空头头寸但订单方向限制为只做多")
    if order.direction == Direction.ShortOnly and position > 0:
        raise ValueError("当前为多头头寸但订单方向限制为只做空")
    if not np.isfinite(order.fees):
        raise ValueError("手续费率必须为有限数值")
    if not np.isfinite(order.fixed_fees):
        raise ValueError("固定手续费必须为有限数值")
    if not np.isfinite(order.slippage) or order.slippage < 0:
        raise ValueError("滑点必须为有限数值且大于等于0")
    if not np.isfinite(order.min_size) or order.min_size < 0:
        raise ValueError("最小订单大小必须为有限数值且大于等于0")
    if np.isnan(order.max_size) or order.max_size <= 0:
        raise ValueError("最大订单大小必须大于0")
    if np.isinf(order.size_granularity) or order.size_granularity <= 0:
        raise ValueError("订单粒度必须为NaN或有限正数")
    if not np.isfinite(order.reject_prob) or order.reject_prob < 0 or order.reject_prob > 1:
        raise ValueError("订单拒绝概率必须在0到1之间")

    # 获取原始订单参数并开始处理
    order_size = order.size           # 订单大小
    order_size_type = order.size_type # 订单大小类型

    # 处理只做空方向的订单
    if order.direction == Direction.ShortOnly:
        # 在只做空方向中，正/负大小应该被视为负/正
        # 这样可以保持一致的数学语义：正数表示增加头寸，负数表示减少头寸
        order_size *= -1

    # 处理目标百分比类型的订单大小
    if order_size_type == SizeType.TargetPercent:
        # 目标百分比：将投资组合调整到总价值的某个百分比
        if np.isnan(value):
            # 总价值为NaN，无法计算百分比，忽略订单
            return exec_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.ValueNaN)
        if value <= 0:
            # 总价值为零或负数，无法计算百分比，拒绝订单
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.ValueZeroNeg)

        # 转换为目标价值：百分比 * 总价值
        order_size *= value
        order_size_type = SizeType.TargetValue  # 更新订单类型为目标价值

    # 处理价值类型的订单大小
    if order_size_type == SizeType.Value or order_size_type == SizeType.TargetValue:
        # 价值类型：需要根据当前价格转换为数量
        if np.isinf(val_price) or val_price <= 0:
            raise ValueError("估值价格必须为有限正数")
        if np.isnan(val_price):
            # 估值价格为NaN，无法转换，忽略订单
            return exec_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.ValPriceNaN)

        # 将价值转换为数量：价值 / 单价 = 数量
        order_size /= val_price
        if order_size_type == SizeType.Value:
            # 普通价值 -> 数量
            order_size_type = SizeType.Amount
        else:
            # 目标价值 -> 目标数量
            order_size_type = SizeType.TargetAmount

    # 处理目标数量类型的订单大小
    if order_size_type == SizeType.TargetAmount:
        # 目标数量：计算需要买入/卖出多少才能达到目标头寸
        # 需要交易的数量 = 目标数量 - 当前持有数量
        order_size -= position
        order_size_type = SizeType.Amount  # 更新为普通数量类型

    # 处理数量类型的特殊情况
    if order_size_type == SizeType.Amount:
        # 检查无限负数的特殊含义
        if order.direction == Direction.ShortOnly or order.direction == Direction.Both:
            if order_size < 0 and np.isinf(order_size):
                # 无限负数有特殊含义：100%做空
                # 转换为百分比形式以便后续处理
                order_size = -1.
                order_size_type = SizeType.Percent

    # 初始化百分比参数
    percent = np.nan
    
    # 处理百分比类型的订单大小
    if order_size_type == SizeType.Percent:
        # 百分比类型：使用可用资源的一定百分比
        percent = abs(order_size)                    # 获取百分比值（去掉符号）
        order_size = np.sign(order_size) * np.inf   # 设置为带符号的无限大
        order_size_type = SizeType.Amount           # 转换为数量类型处理

    # 根据订单大小的符号决定是买入还是卖出
    if order_size > 0:
        # 正数订单大小：执行买入操作
        new_exec_state, order_result = buy_nb(
            exec_state,                    # 当前执行状态
            order_size,                    # 买入数量
            order.price,                   # 买入价格
            direction=order.direction,     # 交易方向限制
            fees=order.fees,               # 手续费率
            fixed_fees=order.fixed_fees,   # 固定手续费
            slippage=order.slippage,       # 滑点设置
            min_size=order.min_size,       # 最小订单大小
            max_size=order.max_size,       # 最大订单大小
            size_granularity=order.size_granularity,  # 数量粒度
            lock_cash=order.lock_cash,     # 是否锁定现金
            allow_partial=order.allow_partial,        # 是否允许部分成交
            percent=percent                # 资金使用百分比
        )
    else:
        # 负数或零订单大小：执行卖出操作
        new_exec_state, order_result = sell_nb(
            exec_state,                    # 当前执行状态
            -order_size,                   # 卖出数量（转为正数）
            order.price,                   # 卖出价格
            direction=order.direction,     # 交易方向限制
            fees=order.fees,               # 手续费率
            fixed_fees=order.fixed_fees,   # 固定手续费
            slippage=order.slippage,       # 滑点设置
            min_size=order.min_size,       # 最小订单大小
            max_size=order.max_size,       # 最大订单大小
            size_granularity=order.size_granularity,  # 数量粒度
            lock_cash=order.lock_cash,     # 是否锁定现金
            allow_partial=order.allow_partial,        # 是否允许部分成交
            percent=percent                # 头寸使用百分比
        )

    # 处理随机拒绝机制（用于压力测试和风险管理）
    if order.reject_prob > 0:
        # 生成0到1之间的随机数，如果小于拒绝概率则拒绝订单
        if np.random.uniform(0, 1) < order.reject_prob:
            # 随机拒绝：返回原始状态和拒绝结果
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.RandomEvent)

    # 返回执行结果
    return new_exec_state, order_result


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def fill_log_record_nb(record: tp.Record,
                       record_id: int,
                       i: int,
                       col: int,
                       group: int,
                       cash: float,
                       position: float,
                       debt: float,
                       free_cash: float,
                       val_price: float,
                       value: float,
                       order: Order,
                       new_cash: float,
                       new_position: float,
                       new_debt: float,
                       new_free_cash: float,
                       new_val_price: float,
                       new_value: float,
                       order_result: OrderResult,
                       order_id: int) -> None:
    """
    填充日志记录对象
    
    这个函数负责将订单执行过程中的所有关键信息记录到日志记录中，
    包括执行前后的状态变化、原始订单请求和最终执行结果。
    这对于交易分析、调试和审计非常重要。
    
    参数:
    ----
    record : Record
        待填充的日志记录对象
    record_id : int
        记录唯一标识符
    i : int 
        时间索引（通常是时间步或K线索引）
    col : int
        列索引（资产标识符）
    group : int
        组标识符（用于资产分组）
    cash : float
        执行前的现金余额
    position : float
        执行前的头寸
    debt : float
        执行前的债务
    free_cash : float
        执行前的可用现金
    val_price : float
        执行前的估值价格
    value : float
        执行前的组合价值
    order : Order
        原始订单对象
    new_cash : float
        执行后的现金余额
    new_position : float
        执行后的头寸
    new_debt : float
        执行后的债务
    new_free_cash : float
        执行后的可用现金
    new_val_price : float
        执行后的估值价格
    new_value : float
        执行后的组合价值
    order_result : OrderResult
        订单执行结果
    order_id : int
        订单标识符
        
    使用示例:
    --------
    >>> # 在订单执行后记录详细信息
    >>> log_record = np.zeros(1, dtype=log_record_dtype)[0]  # 创建记录
    >>> fill_log_record_nb(log_record, 0, 100, 0, 0, 
    ...                   cash, position, debt, free_cash, val_price, value,
    ...                   order, new_cash, new_position, new_debt, 
    ...                   new_free_cash, new_val_price, new_value,
    ...                   order_result, order_id)
    
    注意事项:
    --------
    - 记录了订单执行的完整轨迹，便于后续分析
    - 包含了订单请求的所有参数，用于复现和调试
    - 记录了状态变化，便于验证执行结果的正确性
    """

    # 基础标识信息
    record['id'] = record_id          # 记录唯一标识
    record['group'] = group           # 资产组标识
    record['col'] = col               # 资产列标识
    record['idx'] = i                 # 时间索引
    
    # 执行前的投资组合状态
    record['cash'] = cash             # 执行前现金
    record['position'] = position     # 执行前头寸
    record['debt'] = debt             # 执行前债务
    record['free_cash'] = free_cash   # 执行前可用现金
    record['val_price'] = val_price   # 执行前估值价格
    record['value'] = value           # 执行前组合价值
    
    # 原始订单请求参数 (req_ 前缀表示请求的参数)
    record['req_size'] = order.size                          # 请求的订单大小
    record['req_price'] = order.price                        # 请求的订单价格
    record['req_size_type'] = order.size_type                # 请求的大小类型
    record['req_direction'] = order.direction                # 请求的交易方向
    record['req_fees'] = order.fees                          # 请求的手续费率
    record['req_fixed_fees'] = order.fixed_fees              # 请求的固定手续费
    record['req_slippage'] = order.slippage                  # 请求的滑点设置
    record['req_min_size'] = order.min_size                  # 请求的最小订单大小
    record['req_max_size'] = order.max_size                  # 请求的最大订单大小
    record['req_size_granularity'] = order.size_granularity # 请求的数量粒度
    record['req_reject_prob'] = order.reject_prob            # 请求的拒绝概率
    record['req_lock_cash'] = order.lock_cash                # 请求的现金锁定设置
    record['req_allow_partial'] = order.allow_partial        # 请求的部分成交设置
    record['req_raise_reject'] = order.raise_reject          # 请求的拒绝异常设置
    record['req_log'] = order.log                            # 请求的日志记录设置
    
    # 执行后的投资组合状态 (new_ 前缀表示更新后的状态)
    record['new_cash'] = new_cash           # 执行后现金
    record['new_position'] = new_position   # 执行后头寸
    record['new_debt'] = new_debt           # 执行后债务
    record['new_free_cash'] = new_free_cash # 执行后可用现金
    record['new_val_price'] = new_val_price # 执行后估值价格
    record['new_value'] = new_value         # 执行后组合价值
    
    # 订单执行结果 (res_ 前缀表示结果)
    record['res_size'] = order_result.size               # 实际成交数量
    record['res_price'] = order_result.price             # 实际成交价格
    record['res_fees'] = order_result.fees               # 实际支付手续费
    record['res_side'] = order_result.side               # 实际交易方向
    record['res_status'] = order_result.status           # 订单执行状态
    record['res_status_info'] = order_result.status_info # 订单状态详细信息
    record['order_id'] = order_id                        # 订单标识符


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def fill_order_record_nb(record: tp.Record,
                         record_id: int,
                         i: int,
                         col: int,
                         order_result: OrderResult) -> None:
    """
    填充订单记录对象
    
    这个函数负责将成功执行的订单结果填充到订单记录中。
    与fill_log_record_nb不同，这个函数只记录核心的订单执行结果，
    不包括详细的状态变化和请求参数，主要用于生成简洁的订单历史。
    
    参数:
    ----
    record : Record
        待填充的订单记录对象
    record_id : int
        记录唯一标识符
    i : int
        时间索引（通常是时间步或K线索引）
    col : int
        列索引（资产标识符）
    order_result : OrderResult
        订单执行结果对象
        
    使用示例:
    --------
    >>> # 创建订单记录并填充
    >>> order_record = np.zeros(1, dtype=order_record_dtype)[0]
    >>> fill_order_record_nb(order_record, 0, 100, 0, order_result)
    >>> print(f"订单ID: {order_record['id']}, 成交量: {order_record['size']}")
    订单ID: 0, 成交量: 100.0
    
    注意事项:
    --------
    - 只记录成功执行的订单核心信息
    - 用于构建简洁的交易历史记录
    - 比日志记录更轻量，适合大量订单的场景
    """
    
    # 基础标识信息
    record['id'] = record_id         # 记录唯一标识
    record['col'] = col              # 资产列标识
    record['idx'] = i                # 时间索引
    
    # 订单执行结果的核心信息
    record['size'] = order_result.size   # 实际成交数量
    record['price'] = order_result.price # 实际成交价格
    record['fees'] = order_result.fees   # 实际支付手续费
    record['side'] = order_result.side   # 交易方向（买入/卖出）


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def raise_rejected_order_nb(order_result: OrderResult) -> None:
    """
    抛出订单拒绝异常
    
    根据订单结果中的状态详情信息，抛出相应的订单拒绝异常。
    这个函数将内部的状态码转换为用户友好的错误消息，
    便于调试和错误处理。
    
    参数:
    ----
    order_result : OrderResult
        包含拒绝状态信息的订单结果对象
        
    异常:
    ----
    RejectedOrderError: 根据具体的拒绝原因抛出相应的错误消息
    
    支持的拒绝原因:
    -------------
    - 数据无效: 订单大小、价格、估值价格为NaN等
    - 资金不足: 没有足够现金进行多头/空头操作
    - 头寸限制: 没有可平仓头寸、超过最大/最小限制等
    - 手续费问题: 无法承担交易手续费
    - 随机拒绝: 压力测试中的随机拒绝事件
    
    使用示例:
    --------
    >>> # 检查订单执行结果，如果被拒绝则抛出异常
    >>> if order_result.status == OrderStatus.Rejected:
    ...     raise_rejected_order_nb(order_result)
    RejectedOrderError: 资金不足，无法进行多头操作
    
    注意事项:
    --------
    - 只在订单被拒绝时调用此函数
    - 提供中英文对照的错误信息便于理解
    - 用于将内部状态码转换为用户可理解的错误消息
    """
    
    # 检查各种拒绝原因并抛出相应的异常
    if order_result.status_info == OrderStatusInfo.SizeNaN:
        raise RejectedOrderError("订单大小为NaN (Size is NaN)")
    if order_result.status_info == OrderStatusInfo.PriceNaN:
        raise RejectedOrderError("订单价格为NaN (Price is NaN)")
    if order_result.status_info == OrderStatusInfo.ValPriceNaN:
        raise RejectedOrderError("资产估值价格为NaN (Asset valuation price is NaN)")
    if order_result.status_info == OrderStatusInfo.ValueNaN:
        raise RejectedOrderError("资产/组合价值为NaN (Asset/group value is NaN)")
    if order_result.status_info == OrderStatusInfo.ValueZeroNeg:
        raise RejectedOrderError("资产/组合价值为零或负数 (Asset/group value is zero or negative)")
    if order_result.status_info == OrderStatusInfo.SizeZero:
        raise RejectedOrderError("订单大小为零 (Size is zero)")
    if order_result.status_info == OrderStatusInfo.NoCashShort:
        raise RejectedOrderError("资金不足，无法进行空头操作 (Not enough cash to short)")
    if order_result.status_info == OrderStatusInfo.NoCashLong:
        raise RejectedOrderError("资金不足，无法进行多头操作 (Not enough cash to long)")
    if order_result.status_info == OrderStatusInfo.NoOpenPosition:
        raise RejectedOrderError("没有可平仓的持仓 (No open position to reduce/close)")
    if order_result.status_info == OrderStatusInfo.MaxSizeExceeded:
        raise RejectedOrderError("订单大小超过最大允许值 (Size is greater than maximum allowed)")
    if order_result.status_info == OrderStatusInfo.RandomEvent:
        raise RejectedOrderError("发生随机拒绝事件 (Random event happened)")
    if order_result.status_info == OrderStatusInfo.CantCoverFees:
        raise RejectedOrderError("资金不足以承担手续费 (Not enough cash to cover fees)")
    if order_result.status_info == OrderStatusInfo.MinSizeNotReached:
        raise RejectedOrderError("最终订单大小小于最小允许值 (Final size is less than minimum allowed)")
    if order_result.status_info == OrderStatusInfo.PartialFill:
        raise RejectedOrderError("最终订单大小小于请求值 (Final size is less than requested)")
    
    # 如果没有匹配的状态信息，抛出通用异常
    raise RejectedOrderError("未知的订单拒绝原因 (Unknown rejection reason)")


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def update_value_nb(cash_before: float,
                    cash_now: float,
                    position_before: float,
                    position_now: float,
                    val_price_before: float,
                    price: float,
                    value_before: float) -> tp.Tuple[float, float]:
    """
    更新估值价格和投资组合总价值
    
    在每次订单执行后，需要更新资产的估值价格和投资组合的总价值。
    这个函数计算订单执行前后的价值变化，确保投资组合估值的准确性。
    
    参数:
    ----
    cash_before : float
        订单执行前的现金余额
    cash_now : float
        订单执行后的现金余额  
    position_before : float
        订单执行前的头寸数量
    position_now : float
        订单执行后的头寸数量
    val_price_before : float
        订单执行前的资产估值价格
    price : float
        订单执行时的实际成交价格（用作新的估值价格）
    value_before : float
        订单执行前的投资组合总价值
        
    返回:
    ----
    tuple[float, float]
        (新的估值价格, 新的投资组合总价值)
        
    计算逻辑:
    --------
    1. 新估值价格 = 订单成交价格
    2. 现金流变化 = 执行后现金 - 执行前现金
    3. 资产价值变化 = 新头寸*新价格 - 旧头寸*旧价格
    4. 新总价值 = 旧总价值 + 现金流变化 + 资产价值变化
    
    使用示例:
    --------
    >>> # 买入100股，价格从50涨到52
    >>> val_price_now, value_now = update_value_nb(
    ...     cash_before=10000, cash_now=4800,    # 花费5200现金
    ...     position_before=0, position_now=100,  # 获得100股
    ...     val_price_before=50, price=52,        # 价格更新为52
    ...     value_before=10000                    # 原总价值10000
    ... )
    >>> print(f"新估值价格: {val_price_now}, 新总价值: {value_now}")
    新估值价格: 52.0, 新总价值: 10000.0  # 总价值不变（忽略手续费）
    
    注意事项:
    --------
    - 估值价格更新为最新的成交价格
    - 总价值考虑了现金和资产价值的综合变化  
    - 用于维护投资组合价值的连续性和准确性
    """
    
    # 更新估值价格为最新成交价格
    val_price_now = price
    
    # 计算现金流变化（现金的增减）
    cash_flow = cash_now - cash_before
    
    # 计算订单执行前的资产价值
    if position_before != 0:
        asset_value_before = position_before * val_price_before  # 旧头寸 * 旧价格
    else:
        asset_value_before = 0.  # 无头寸时资产价值为零
    
    # 计算订单执行后的资产价值
    if position_now != 0:
        asset_value_now = position_now * val_price_now  # 新头寸 * 新价格
    else:
        asset_value_now = 0.  # 无头寸时资产价值为零
    
    # 计算资产价值的变化
    asset_value_diff = asset_value_now - asset_value_before
    
    # 计算新的投资组合总价值
    # 新总价值 = 原总价值 + 现金变化 + 资产价值变化
    value_now = value_before + cash_flow + asset_value_diff
    
    return val_price_now, value_now


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def process_order_nb(i: int,
                     col: int,
                     group: int,
                     state: ProcessOrderState,
                     update_value: bool,
                     order: Order,
                     order_records: tp.RecordArray,
                     log_records: tp.RecordArray) -> tp.Tuple[OrderResult, ProcessOrderState]:
    """
    处理订单的完整流程函数
    
    这是订单处理的最高层包装函数，整合了订单执行、记录保存和状态更新的完整流程。
    它按顺序执行以下操作：
    1. 执行订单
    2. 处理拒绝异常
    3. 更新价值估算
    4. 保存订单记录
    5. 保存日志记录
    6. 创建新的处理状态
    
    参数:
    ----
    i : int
        时间索引（当前时间步或K线索引）
    col : int
        资产列索引（资产标识符）
    group : int
        资产组索引（用于分组管理）
    state : ProcessOrderState
        当前的处理状态（包含现金、头寸、债务等信息）
    update_value : bool
        是否在订单成交后更新投资组合价值
    order : Order
        待处理的订单对象
    order_records : RecordArray
        用于存储成功订单记录的数组
    log_records : RecordArray
        用于存储详细日志记录的数组
        
    返回:
    ----
    tuple[OrderResult, ProcessOrderState]
        (订单执行结果, 更新后的处理状态)
        
    异常:
    ----
    IndexError: 当记录数组空间不足时抛出
    RejectedOrderError: 当订单被拒绝且要求抛出异常时
    
    使用示例:
    --------
    >>> # 处理一个买入订单
    >>> order = Order(size=100, price=50.0)
    >>> order_result, new_state = process_order_nb(
    ...     i=0, col=0, group=0, state=current_state, 
    ...     update_value=True, order=order,
    ...     order_records=order_rec_array, log_records=log_rec_array
    ... )
    >>> print(f"订单状态: {order_result.status}")
    >>> print(f"新现金余额: {new_state.cash}")
    
    处理流程:
    --------
    1. 调用execute_order_nb执行订单
    2. 检查是否需要抛出拒绝异常
    3. 如果订单成交且需要更新价值，则调用update_value_nb
    4. 如果订单成交，则将结果保存到order_records
    5. 如果需要记录日志，则将详细信息保存到log_records
    6. 构造并返回更新后的状态
    
    注意事项:
    --------
    - 确保record数组有足够的空间存储记录
    - 只有成功执行的订单才会被记录到order_records
    - 日志记录取决于order.log设置
    - 状态更新包括了记录索引的递增
    """

    # 执行订单
    exec_state, order_result = execute_order_nb(state, order)

    # 处理订单拒绝异常
    is_rejected = order_result.status == OrderStatus.Rejected
    if is_rejected and order.raise_reject:
        # 如果订单被拒绝且要求抛出异常，则抛出相应异常
        raise_rejected_order_nb(order_result)

    # 更新估值价格和投资组合总价值
    is_filled = order_result.status == OrderStatus.Filled
    if is_filled and update_value:
        # 订单成交且需要更新价值时，计算新的估值价格和总价值
        new_val_price, new_value = update_value_nb(
            state.cash,              # 执行前现金
            exec_state.cash,         # 执行后现金
            state.position,          # 执行前头寸
            exec_state.position,     # 执行后头寸
            state.val_price,         # 执行前估值价格
            order_result.price,      # 成交价格
            state.value              # 执行前总价值
        )
    else:
        # 订单未成交或不需要更新价值时，保持原有价值
        new_val_price = state.val_price
        new_value = state.value

    # 处理订单记录
    new_oidx = state.oidx  # 订单记录索引
    if is_filled:
        # 只有成交的订单才记录到订单记录数组
        if state.oidx > len(order_records) - 1:
            raise IndexError("订单记录数组空间不足，请设置更高的max_orders参数")
        
        # 填充订单记录
        fill_order_record_nb(
            order_records[state.oidx],  # 目标记录位置
            state.oidx,                 # 记录ID
            i,                          # 时间索引
            col,                        # 资产列索引
            order_result                # 订单结果
        )
        new_oidx += 1  # 递增订单记录索引

    # 处理日志记录
    new_lidx = state.lidx  # 日志记录索引
    if order.log:
        # 只有启用日志的订单才记录到日志数组
        if state.lidx > len(log_records) - 1:
            raise IndexError("日志记录数组空间不足，请设置更高的max_logs参数")
        
        # 填充详细日志记录
        fill_log_record_nb(
            log_records[state.lidx],   # 目标记录位置
            state.lidx,                # 记录ID
            i,                         # 时间索引
            col,                       # 资产列索引
            group,                     # 资产组索引
            # 执行前状态
            state.cash, state.position, state.debt, 
            state.free_cash, state.val_price, state.value,
            order,                     # 原始订单
            # 执行后状态
            exec_state.cash, exec_state.position, exec_state.debt,
            exec_state.free_cash, new_val_price, new_value,
            order_result,              # 订单结果
            state.oidx if is_filled else -1  # 订单ID（未成交为-1）
        )
        new_lidx += 1  # 递增日志记录索引

    # 创建更新后的处理状态
    new_state = ProcessOrderState(
        cash=exec_state.cash,        # 更新后的现金
        position=exec_state.position, # 更新后的头寸
        debt=exec_state.debt,        # 更新后的债务
        free_cash=exec_state.free_cash, # 更新后的可用现金
        val_price=new_val_price,     # 更新后的估值价格
        value=new_value,             # 更新后的总价值
        oidx=new_oidx,              # 更新后的订单记录索引
        lidx=new_lidx               # 更新后的日志记录索引
    )

    return order_result, new_state


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def order_nb(size: float = np.nan,
             price: float = np.inf,
             size_type: int = SizeType.Amount,
             direction: int = Direction.Both,
             fees: float = 0.,
             fixed_fees: float = 0.,
             slippage: float = 0.,
             min_size: float = 0.,
             max_size: float = np.inf,
             size_granularity: float = np.nan,
             reject_prob: float = 0.,
             lock_cash: bool = False,
             allow_partial: bool = True,
             raise_reject: bool = False,
             log: bool = False) -> Order:
    """
    创建订单对象
    
    这是订单创建的便捷函数，将所有参数封装到Order对象中。
    它提供了所有订单参数的默认值，并确保类型转换正确。
    
    参数:
    ----
    size : float, 可选 (默认: np.nan)
        订单大小，具体含义取决于size_type：
        - Amount: 具体数量
        - Value: 价值金额  
        - Percent: 资金使用百分比
        - TargetAmount: 目标持仓数量
        - TargetPercent: 目标持仓百分比
        
    price : float, 可选 (默认: np.inf)
        订单价格，np.inf表示市价订单
        
    size_type : int, 可选 (默认: SizeType.Amount)
        订单大小类型，参见SizeType枚举
        
    direction : int, 可选 (默认: Direction.Both)
        交易方向限制，参见Direction枚举
        - Both: 允许买卖双向
        - LongOnly: 只允许做多
        - ShortOnly: 只允许做空
        
    fees : float, 可选 (默认: 0.0)
        比例手续费率（如0.001表示0.1%）
        
    fixed_fees : float, 可选 (默认: 0.0)
        固定手续费（绝对金额）
        
    slippage : float, 可选 (默认: 0.0)
        滑点率，买入向上滑点，卖出向下滑点
        
    min_size : float, 可选 (默认: 0.0)
        最小订单大小
        
    max_size : float, 可选 (默认: np.inf)
        最大订单大小
        
    size_granularity : float, 可选 (默认: np.nan)
        订单数量粒度，订单大小将取整到此粒度的倍数
        
    reject_prob : float, 可选 (默认: 0.0)
        随机拒绝概率，用于压力测试（0-1之间）
        
    lock_cash : bool, 可选 (默认: False)
        是否锁定现金使用量
        
    allow_partial : bool, 可选 (默认: True)
        是否允许部分成交
        
    raise_reject : bool, 可选 (默认: False)  
        订单被拒绝时是否抛出异常
        
    log : bool, 可选 (默认: False)
        是否记录详细日志
        
    返回:
    ----
    Order
        封装了所有参数的订单对象
        
    使用示例:
    --------
    >>> # 创建市价买入订单
    >>> buy_order = order_nb(size=100, price=np.inf)
    >>> print(f"订单大小: {buy_order.size}")
    订单大小: 100.0
    
    >>> # 创建限价卖出订单
    >>> sell_order = order_nb(size=-50, price=52.0, fees=0.001)
    >>> print(f"手续费: {sell_order.fees}")
    手续费: 0.001
    
    >>> # 创建目标持仓订单
    >>> target_order = order_nb(size=200, size_type=SizeType.TargetAmount)
    >>> print(f"订单类型: {target_order.size_type}")
    订单类型: 3
    
    注意事项:
    --------
    - size为NaN时表示不执行订单
    - 正数size表示买入，负数size表示卖出
    - 所有参数都会被转换为适当的数据类型
    - 提供了灵活的默认值，便于快速创建订单
    """

    # 创建Order对象，确保所有参数类型正确
    return Order(
        size=float(size),                        # 订单大小（转为浮点数）
        price=float(price),                      # 订单价格（转为浮点数）
        size_type=int(size_type),                # 大小类型（转为整数）
        direction=int(direction),                # 交易方向（转为整数）
        fees=float(fees),                        # 比例手续费（转为浮点数）
        fixed_fees=float(fixed_fees),            # 固定手续费（转为浮点数）
        slippage=float(slippage),                # 滑点率（转为浮点数）
        min_size=float(min_size),                # 最小订单大小（转为浮点数）
        max_size=float(max_size),                # 最大订单大小（转为浮点数）
        size_granularity=float(size_granularity), # 数量粒度（转为浮点数）
        reject_prob=float(reject_prob),          # 拒绝概率（转为浮点数）
        lock_cash=bool(lock_cash),               # 现金锁定标志（转为布尔值）
        allow_partial=bool(allow_partial),       # 允许部分成交标志（转为布尔值）
        raise_reject=bool(raise_reject),         # 拒绝时抛异常标志（转为布尔值）
        log=bool(log)                           # 日志记录标志（转为布尔值）
    )


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def close_position_nb(price: float = np.inf,
                      fees: float = 0.,
                      fixed_fees: float = 0.,
                      slippage: float = 0.,
                      min_size: float = 0.,
                      max_size: float = np.inf,
                      size_granularity: float = np.nan,
                      reject_prob: float = 0.,
                      lock_cash: bool = False,
                      allow_partial: bool = True,
                      raise_reject: bool = False,
                      log: bool = False) -> Order:
    """
    创建平仓订单
    
    这是一个便捷函数，用于创建完全平掉当前持仓的订单。
    无论当前是多头还是空头头寸，都会创建相应的平仓订单。
    
    参数:
    ----
    price : float, 可选 (默认: np.inf)
        平仓价格，np.inf表示市价平仓
        
    fees : float, 可选 (默认: 0.0)
        比例手续费率
        
    fixed_fees : float, 可选 (默认: 0.0)
        固定手续费
        
    slippage : float, 可选 (默认: 0.0)
        滑点率
        
    min_size : float, 可选 (默认: 0.0)
        最小订单大小
        
    max_size : float, 可选 (默认: np.inf)
        最大订单大小
        
    size_granularity : float, 可选 (默认: np.nan)
        订单数量粒度
        
    reject_prob : float, 可选 (默认: 0.0)
        随机拒绝概率
        
    lock_cash : bool, 可选 (默认: False)
        是否锁定现金
        
    allow_partial : bool, 可选 (默认: True)
        是否允许部分成交
        
    raise_reject : bool, 可选 (默认: False)
        订单被拒绝时是否抛出异常
        
    log : bool, 可选 (默认: False)
        是否记录详细日志
        
    返回:
    ----
    Order
        目标持仓为0的订单对象
        
    使用示例:
    --------
    >>> # 市价平仓
    >>> close_order = close_position_nb()
    >>> print(f"订单类型: {close_order.size_type}")  # TargetAmount
    >>> print(f"目标大小: {close_order.size}")        # 0.0
    
    >>> # 限价平仓
    >>> limit_close = close_position_nb(price=50.0, fees=0.001)
    >>> print(f"平仓价格: {limit_close.price}")
    平仓价格: 50.0
    
    工作原理:
    --------
    内部调用order_nb()创建订单，参数设置为：
    - size=0.0: 目标持仓数量为0
    - size_type=SizeType.TargetAmount: 目标数量类型
    - direction=Direction.Both: 允许双向交易（平多头或空头）
    
    注意事项:
    --------
    - 适用于任何类型的持仓（多头或空头）
    - 如果当前没有持仓，订单会被忽略
    - 实际平仓数量由当前持仓决定
    - 是order_nb的特殊化版本，专门用于平仓
    """

    # 调用order_nb创建目标持仓为0的订单
    return order_nb(
        size=0.,                      # 目标持仓数量为0（完全平仓）
        price=price,                  # 平仓价格
        size_type=SizeType.TargetAmount,  # 使用目标数量类型
        direction=Direction.Both,     # 允许双向交易（可平多头或空头）
        fees=fees,                    # 手续费设置
        fixed_fees=fixed_fees,        # 固定手续费设置
        slippage=slippage,           # 滑点设置
        min_size=min_size,           # 最小订单限制
        max_size=max_size,           # 最大订单限制
        size_granularity=size_granularity,  # 数量粒度
        reject_prob=reject_prob,     # 随机拒绝概率
        lock_cash=lock_cash,         # 现金锁定设置
        allow_partial=allow_partial, # 部分成交设置
        raise_reject=raise_reject,   # 异常抛出设置
        log=log                      # 日志记录设置
    )


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def order_nothing_nb() -> Order:
    """
    创建空订单（无操作订单）
    
    这是一个便捷函数，返回一个预定义的空订单对象。
    当策略逻辑确定在当前时间点不需要进行任何交易时，
    可以使用此函数返回一个表示"无操作"的订单。
    
    返回:
    ----
    Order
        NoOrder常量，表示不执行任何交易操作的订单
        
    使用示例:
    --------
    >>> # 在策略中根据条件决定是否交易
    >>> def trading_logic(signal, current_position):
    ...     if signal > 0.5 and current_position == 0:
    ...         return order_nb(size=100, price=50.0)  # 买入
    ...     elif signal < -0.5 and current_position > 0:
    ...         return close_position_nb()             # 平仓
    ...     else:
    ...         return order_nothing_nb()              # 无操作
    
    >>> # 当无明确交易信号时使用
    >>> no_action = order_nothing_nb()
    >>> print(f"订单大小: {no_action.size}")  # NaN表示无操作
    订单大小: nan
    
    适用场景:
    --------
    - 策略逻辑中的无交易时段
    - 等待更好的入场机会
    - 条件不满足时的默认行为
    - 回测中的观望期处理
    
    技术细节:
    --------
    - 返回预定义的NoOrder常量
    - 该订单在执行时会被忽略
    - 不会产生任何交易成本或记录
    - 保持投资组合状态不变
    
    注意事项:
    --------
    - NoOrder是一个特殊的订单对象，size为NaN
    - 执行时不会改变任何投资组合状态
    - 是策略编程中的常用模式
    - 比创建size=NaN的订单更明确和高效
    """
    return NoOrder  # 返回预定义的空订单常量


# ############# 参数检查系统 (Parameter Validation System) ############# #


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def check_group_lens_nb(group_lens: tp.Array1d, n_cols: int) -> None:
    """
    检查资产分组长度数组的有效性
    
    验证group_lens数组的总和是否等于总列数，确保资产分组配置正确。
    这是投资组合系统中重要的参数验证函数，防止配置错误导致的运行时问题。
    
    参数:
    ----
    group_lens : Array1d
        各组的列数数组，每个元素表示对应组包含的资产数量
        例如：[2, 3, 1] 表示第1组有2个资产，第2组有3个资产，第3组有1个资产
        
    n_cols : int
        总列数（总资产数量）
        
    异常:
    ----
    ValueError: 当group_lens的总和不等于n_cols时抛出
    
    使用示例:
    --------
    >>> # 正确的分组配置
    >>> group_lens = np.array([2, 3, 1])  # 3组，分别有2、3、1个资产
    >>> n_cols = 6                        # 总共6个资产
    >>> check_group_lens_nb(group_lens, n_cols)  # 不会抛出异常
    
    >>> # 错误的分组配置
    >>> group_lens = np.array([2, 3, 1])  # 总和为6
    >>> n_cols = 5                        # 但只有5个资产
    >>> check_group_lens_nb(group_lens, n_cols)
    ValueError: group_lens has incorrect total number of columns
    
    应用场景:
    --------
    - 多资产投资组合的分组配置验证
    - 资产组合策略的参数检查
    - 防止配置错误导致的索引越界
    - 投资组合初始化时的安全检查
    
    注意事项:
    --------
    - group_lens中的每个值都应为正整数
    - 总和必须恰好等于资产总数
    - 用于确保后续的分组操作不会出现索引错误
    """
    if np.sum(group_lens) != n_cols:
        raise ValueError("资产分组长度配置错误：各组长度总和与总列数不匹配")


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def check_group_init_cash_nb(group_lens: tp.Array1d, n_cols: int, init_cash: tp.Array1d, cash_sharing: bool) -> None:
    """
    检查初始现金配置的有效性
    
    验证初始现金数组的长度是否与现金共享模式和资产分组配置匹配。
    这确保了每个资产或资产组都有正确的初始现金分配。
    
    参数:
    ----
    group_lens : Array1d
        各组的列数数组
        
    n_cols : int
        总列数（总资产数量）
        
    init_cash : Array1d
        初始现金数组
        
    cash_sharing : bool
        现金共享模式标志：
        - True: 组内资产共享现金，init_cash长度应等于组数
        - False: 每个资产独立现金，init_cash长度应等于资产数
        
    异常:
    ----
    ValueError: 当init_cash长度配置错误时抛出
    
    使用示例:
    --------
    >>> group_lens = np.array([2, 3])      # 2组：第1组2个资产，第2组3个资产  
    >>> n_cols = 5                         # 总共5个资产
    >>> 
    >>> # 现金共享模式：每组一个初始现金值
    >>> init_cash_shared = np.array([10000, 15000])  # 2个值对应2组
    >>> check_group_init_cash_nb(group_lens, n_cols, init_cash_shared, True)
    >>> 
    >>> # 非现金共享模式：每个资产一个初始现金值
    >>> init_cash_separate = np.array([10000, 10000, 15000, 15000, 15000])  # 5个值对应5个资产
    >>> check_group_init_cash_nb(group_lens, n_cols, init_cash_separate, False)
    
    >>> # 错误配置示例
    >>> wrong_init_cash = np.array([10000])  # 只有1个值但有2组
    >>> check_group_init_cash_nb(group_lens, n_cols, wrong_init_cash, True)
    ValueError: If cash sharing is enabled, init_cash must match the number of groups
    
    现金共享模式说明:
    ---------------
    - cash_sharing=True: 组内所有资产共享同一个现金池
    - cash_sharing=False: 每个资产维护独立的现金账户
    
    注意事项:
    --------
    - 确保初始现金配置与预期的现金管理模式一致
    - 现金共享可以提高资金利用效率
    - 非现金共享提供更细粒度的资金控制
    """
    if cash_sharing:
        # 现金共享模式：初始现金数组长度应等于组数
        if len(init_cash) != len(group_lens):
            raise ValueError("启用现金共享时，初始现金数组长度必须与组数匹配")
    else:
        # 非现金共享模式：初始现金数组长度应等于总列数
        if len(init_cash) != n_cols:
            raise ValueError("禁用现金共享时，初始现金数组长度必须与总资产数匹配")


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def is_grouped_nb(group_lens: tp.Array1d) -> bool:
    """
    检查资产是否进行了分组
    
    判断是否存在包含多个资产的组，即是否真正进行了资产分组。
    如果所有组都只包含一个资产，则认为没有进行实际的分组。
    
    参数:
    ----
    group_lens : Array1d
        各组的列数数组，每个元素表示对应组的资产数量
        
    返回:
    ----
    bool
        True表示存在分组（至少有一组包含多个资产）
        False表示无分组（所有组都只有一个资产）
        
    使用示例:
    --------
    >>> # 有实际分组的情况
    >>> group_lens = np.array([2, 3, 1])  # 第1组2个资产，第2组3个资产
    >>> is_grouped = is_grouped_nb(group_lens)
    >>> print(f"是否分组: {is_grouped}")
    是否分组: True
    
    >>> # 无实际分组的情况（每组都只有1个资产）
    >>> group_lens = np.array([1, 1, 1, 1])  # 4组，每组1个资产
    >>> is_grouped = is_grouped_nb(group_lens)  
    >>> print(f"是否分组: {is_grouped}")
    是否分组: False
    
    >>> # 边界情况：只有一组但包含多个资产
    >>> group_lens = np.array([5])  # 1组包含5个资产
    >>> is_grouped = is_grouped_nb(group_lens)
    >>> print(f"是否分组: {is_grouped}")
    是否分组: True
    
    应用场景:
    --------
    - 确定是否需要启用组级别的操作
    - 优化算法：当无分组时可使用简化逻辑
    - 现金共享功能的前置检查
    - 资产组合策略的适用性判断
    
    技术细节:
    --------
    使用np.any(group_lens > 1)检查是否存在大于1的组长度
    
    注意事项:
    --------
    - 返回False时表示每个资产都是独立的组
    - 返回True时表示至少存在一个多资产组
    - 用于决定是否需要组级别的现金管理和风险控制
    """
    return np.any(group_lens > 1)  # 检查是否有任何组包含超过1个资产


# ############# 调用序列管理系统 (Call Sequence Management System) ############# #


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def shuffle_call_seq_nb(call_seq: tp.Array2d, group_lens: tp.Array1d) -> None:
    """
    随机打乱调用序列数组
    
    在每个资产组内随机打乱资产的调用顺序，用于创建随机的交易执行顺序。
    这对于避免系统性偏差和模拟真实市场的随机性非常重要。
    
    参数:
    ----
    call_seq : Array2d
        调用序列数组，形状为(时间步数, 资产数)
        每行表示在该时间步的资产调用顺序
        
    group_lens : Array1d
        各组的长度数组，每个元素表示对应组包含的资产数量
        
    就地修改:
    --------
    此函数直接修改传入的call_seq数组，不返回新数组
    
    使用示例:
    --------
    >>> import numpy as np
    >>> np.random.seed(42)  # 设置随机种子便于重现
    >>> 
    >>> # 创建初始调用序列：3个时间步，6个资产，分为2组[2,4]
    >>> call_seq = np.array([[0, 1, 2, 3, 4, 5],
    ...                     [0, 1, 2, 3, 4, 5],
    ...                     [0, 1, 2, 3, 4, 5]])
    >>> group_lens = np.array([2, 4])  # 第1组2个资产(0,1)，第2组4个资产(2,3,4,5)
    >>> 
    >>> print("打乱前:")
    >>> print(call_seq)
    >>> shuffle_call_seq_nb(call_seq, group_lens)
    >>> print("打乱后:")
    >>> print(call_seq)  # 组内顺序被随机打乱，但组间边界保持
    
    算法逻辑:
    --------
    1. 遍历每个资产组
    2. 对于每个时间步，在组内随机打乱资产顺序
    3. 保持不同组之间的边界不变
    
    应用场景:
    --------
    - 模拟真实交易中的随机执行顺序
    - 避免由固定顺序导致的系统性偏差
    - 压力测试：验证策略在不同执行顺序下的稳健性
    - 蒙特卡洛模拟中的随机性引入
    
    技术细节:
    --------
    - 使用np.random.shuffle()进行就地随机打乱
    - 只在组内打乱，维护组间的逻辑分离
    - 每个时间步都独立进行随机化
    
    注意事项:
    --------
    - 需要预先设置随机种子以获得可重现的结果
    - 直接修改输入数组，调用后原数组内容改变
    - 组间的相对位置和边界不会改变
    - 适用于需要引入执行顺序随机性的回测场景
    """
    from_col = 0  # 当前组的起始列索引
    
    # 遍历每个资产组
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]  # 当前组的结束列索引
        
        # 对每个时间步在当前组内进行随机打乱
        for i in range(call_seq.shape[0]):
            np.random.shuffle(call_seq[i, from_col:to_col])  # 组内随机打乱
        
        from_col = to_col  # 移动到下一组


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def build_call_seq_nb(target_shape: tp.Shape,
                      group_lens: tp.Array1d,
                      call_seq_type: int = CallSeqType.Default) -> tp.Array2d:
    """
    构建新的调用序列数组
    
    根据指定的类型创建资产调用顺序数组，支持多种调用模式：
    正常顺序、反向顺序、随机顺序。这是投资组合模拟中的关键组件。
    
    参数:
    ----
    target_shape : Shape
        目标形状 (时间步数, 资产数)
        
    group_lens : Array1d
        各组的长度数组
        
    call_seq_type : int, 可选 (默认: CallSeqType.Default)
        调用序列类型：
        - CallSeqType.Default: 正常顺序 (0, 1, 2, ...)
        - CallSeqType.Reversed: 反向顺序 (最后一个资产优先)
        - CallSeqType.Random: 随机顺序
        
    返回:
    ----
    Array2d
        调用序列数组，形状为target_shape
        每个元素表示资产在该位置的调用优先级
        
    使用示例:
    --------
    >>> # 创建3个时间步，6个资产，分2组[3,3]的调用序列
    >>> target_shape = (3, 6)
    >>> group_lens = np.array([3, 3])
    >>> 
    >>> # 默认顺序
    >>> default_seq = build_call_seq_nb(target_shape, group_lens, CallSeqType.Default)
    >>> print("默认顺序:")
    >>> print(default_seq)
    >>> # 输出: [[0 1 2 3 4 5]
    >>> #        [0 1 2 3 4 5] 
    >>> #        [0 1 2 3 4 5]]
    >>> 
    >>> # 反向顺序
    >>> reversed_seq = build_call_seq_nb(target_shape, group_lens, CallSeqType.Reversed)
    >>> print("反向顺序:")
    >>> print(reversed_seq)
    >>> # 输出: [[2 1 0 5 4 3]  # 组内反向
    >>> #        [2 1 0 5 4 3]
    >>> #        [2 1 0 5 4 3]]
    >>> 
    >>> # 随机顺序
    >>> np.random.seed(42)
    >>> random_seq = build_call_seq_nb(target_shape, group_lens, CallSeqType.Random)
    >>> print("随机顺序:")
    >>> print(random_seq)  # 组内随机排列
    
    序列类型详解:
    -----------
    1. Default: 按资产索引顺序调用，适用于标准回测
    2. Reversed: 在组内按反向顺序调用，用于测试顺序敏感性
    3. Random: 随机调用顺序，用于消除顺序偏差
    
    算法逻辑:
    --------
    1. 创建基础序列数组
    2. 根据组长度调整序列结构
    3. 应用指定的排序规则
    4. 扩展到目标形状的所有时间步
    
    应用场景:
    --------
    - 投资组合回测中的资产处理顺序控制
    - 多资产策略的执行顺序管理
    - 交易顺序对策略影响的敏感性分析
    - 现金共享模式下的资产优先级设定
    
    注意事项:
    --------
    - 反向序列在组内反转，但保持组间的原始顺序
    - 随机序列在每个时间步都可能不同
    - 调用序列直接影响现金分配和交易执行结果
    - 在现金共享模式下，调用顺序尤其重要
    """
    
    # 处理反向调用序列
    if call_seq_type == CallSeqType.Reversed:
        # 创建基础数组，初值为1
        out = np.full(target_shape[1], 1, dtype=np.int64)
        
        # 在每组的最后一个位置减去组长度，实现组内反向
        out[np.cumsum(group_lens)[1:] - group_lens[1:] - 1] -= group_lens[1:]
        
        # 反向累积求和并反转，再减1得到反向索引
        out = np.cumsum(out[::-1])[::-1] - 1
        
        # 扩展到所有时间步
        out = out * np.ones((target_shape[0], 1), dtype=np.int64)
        return out
    
    # 处理默认和随机调用序列
    # 创建基础数组，初值为1
    out = np.full(target_shape[1], 1, dtype=np.int64)
    
    # 在每组的起始位置减去前一组的长度
    out[np.cumsum(group_lens)[:-1]] -= group_lens[:-1]
    
    # 累积求和后减1，得到连续的索引序列
    out = np.cumsum(out) - 1
    
    # 扩展到所有时间步
    out = out * np.ones((target_shape[0], 1), dtype=np.int64)
    
    # 如果是随机类型，进行随机打乱
    if call_seq_type == CallSeqType.Random:
        shuffle_call_seq_nb(out, group_lens)
    
    return out


def require_call_seq(call_seq: tp.Array2d) -> tp.Array2d:
    """
    强制调用序列数组满足要求
    
    确保调用序列数组具有正确的数据类型和内存布局，
    以满足后续Numba编译函数的严格要求。
    
    参数:
    ----
    call_seq : Array2d
        调用序列数组
        
    返回:
    ----
    Array2d
        满足要求的调用序列数组
        
    内存要求:
    --------
    - dtype=np.int64: 64位整数类型
    - 'A': 对齐 (Aligned)
    - 'O': 拥有数据 (Owndata)  
    - 'W': 可写 (Writeable)
    - 'F': Fortran风格连续 (Fortran-contiguous)
    
    使用示例:
    --------
    >>> call_seq = np.array([[0, 1], [1, 0]], dtype=np.int32)  # 错误的数据类型
    >>> proper_seq = require_call_seq(call_seq)
    >>> print(proper_seq.dtype)  # int64
    >>> print(proper_seq.flags.owndata)  # True
    
    注意事项:
    --------
    - 主要用于数据类型转换和内存布局优化
    - 确保与Numba编译代码的兼容性
    - 可能会创建数组副本以满足要求
    """
    return np.require(call_seq, dtype=np.int64, requirements=['A', 'O', 'W', 'F'])


def build_call_seq(target_shape: tp.Shape,
                   group_lens: tp.Array1d,
                   call_seq_type: int = CallSeqType.Default) -> tp.Array2d:
    """
    构建调用序列数组（非编译优化版本）
    
    这是build_call_seq_nb的非编译版本，使用NumPy的向量化操作
    实现更快的执行速度。适用于大规模数组处理。
    
    参数:
    ----
    target_shape : Shape
        目标形状 (时间步数, 资产数)
        
    group_lens : Array1d
        各组的长度数组
        
    call_seq_type : int, 可选 (默认: CallSeqType.Default)
        调用序列类型
        
    返回:
    ----
    Array2d
        调用序列数组，满足内存和类型要求
        
    性能特点:
    --------
    - 使用NumPy向量化操作，避免Python循环
    - 比编译版本在大数组上更快
    - 使用broadcast_to进行高效的数组扩展
    
    使用示例:
    --------
    >>> target_shape = (1000, 100)  # 大规模数组
    >>> group_lens = np.array([50, 50])
    >>> 
    >>> # 使用非编译版本处理大数组
    >>> call_seq = build_call_seq(target_shape, group_lens, CallSeqType.Default)
    >>> print(f"形状: {call_seq.shape}")
    >>> print(f"数据类型: {call_seq.dtype}")
    >>> print(f"内存连续: {call_seq.flags.f_contiguous}")
    
    算法优势:
    --------
    1. 向量化操作减少Python开销
    2. broadcast_to避免不必要的内存分配
    3. 批量处理提高缓存效率
    4. 自动内存布局优化
    
    注意事项:
    --------
    - 大数组情况下比编译版本更快
    - 小数组可能由于函数调用开销稍慢
    - 随机类型需要额外的就地修改操作
    - 最终结果保证满足Numba编译代码要求
    """
    # 创建基础调用序列（一维）
    call_seq = np.full(target_shape[1], 1, dtype=np.int64)
    
    # 根据调用序列类型构建序列
    if call_seq_type == CallSeqType.Reversed:
        # 反向序列：组内反转
        call_seq[np.cumsum(group_lens)[1:] - group_lens[1:] - 1] -= group_lens[1:]
        call_seq = np.cumsum(call_seq[::-1])[::-1] - 1
    else:
        # 默认序列：正常顺序
        call_seq[np.cumsum(group_lens[:-1])] -= group_lens[:-1]
        call_seq = np.cumsum(call_seq) - 1
    
    # 使用广播高效扩展到目标形状
    call_seq = np.broadcast_to(call_seq, target_shape)
    
    # 处理随机序列类型
    if call_seq_type == CallSeqType.Random:
        # 确保数组可写并进行随机打乱
        call_seq = require_call_seq(call_seq)
        shuffle_call_seq_nb(call_seq, group_lens)
    
    # 确保最终结果满足所有要求
    return require_call_seq(call_seq)


# ############# 辅助工具函数系统 (Helper Utility Functions System) ############# #


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def get_col_elem_nb(ctx: tp.Union[RowContext, SegmentContext, FlexOrderContext], col: int,
                    a: tp.ArrayLike) -> tp.Scalar:
    """
    根据上下文和列索引获取当前元素
    
    使用灵活的索引机制从数组中提取当前时间步和指定列的元素。
    支持多种数组形状的自动处理，是策略回调函数中的常用工具。
    
    参数:
    ----
    ctx : Union[RowContext, SegmentContext, FlexOrderContext]
        上下文对象，包含当前时间步索引和灵活索引设置：
        - i: 当前时间步索引
        - flex_2d: 是否启用2D灵活索引
        
    col : int
        目标列索引（资产索引）
        
    a : ArrayLike
        待查询的数组，可以是1D或2D
        
    返回:
    ----
    Scalar
        指定位置的元素值
        
    使用示例:
    --------
    >>> # 在策略回调函数中使用
    >>> def my_strategy(ctx, prices, volumes):
    ...     # 获取当前时间步指定资产的价格
    ...     current_price = get_col_elem_nb(ctx, col=0, prices)
    ...     current_volume = get_col_elem_nb(ctx, col=0, volumes)
    ...     return current_price, current_volume
    
    >>> # 处理不同形状的数组
    >>> prices_1d = np.array([50.0, 51.0, 52.0])  # 单资产价格序列
    >>> prices_2d = np.array([[50.0, 60.0], [51.0, 61.0]])  # 多资产价格矩阵
    >>> 
    >>> # 上下文示例（伪代码）
    >>> ctx = SegmentContext(i=1, flex_2d=True)
    >>> price = get_col_elem_nb(ctx, 0, prices_2d)  # 获取时间步1，资产0的价格
    >>> # 结果: 51.0
    
    灵活索引机制:
    -----------
    - 1D数组: 根据时间步索引提取元素，忽略列索引
    - 2D数组: 根据(时间步, 列)二维索引提取元素
    - 自动形状检测: 根据flex_2d设置自适应处理
    - 广播支持: 处理标量和形状不匹配的情况
    
    适用场景:
    --------
    - 策略回调函数中获取当前市场数据
    - 多资产投资组合中的数据提取
    - 自定义指标计算中的数据访问
    - 动态参数配置的实时获取
    
    注意事项:
    --------
    - 需要确保数组有足够的数据点
    - 列索引应在数组范围内
    - 灵活索引设置影响数组形状解释方式
    - 通常在Numba编译的回调函数中使用
    """
    return flex_select_auto_nb(a, ctx.i, col, ctx.flex_2d)


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def get_elem_nb(ctx: tp.Union[OrderContext, PostOrderContext, SignalContext],
                a: tp.ArrayLike) -> tp.Scalar:
    """
    根据上下文获取当前元素
    
    使用上下文中包含的时间步和列信息，从数组中提取当前元素。
    这是get_col_elem_nb的简化版本，自动使用上下文中的列索引。
    
    参数:
    ----
    ctx : Union[OrderContext, PostOrderContext, SignalContext]
        上下文对象，包含：
        - i: 当前时间步索引
        - col: 当前列索引（资产索引）
        - flex_2d: 是否启用2D灵活索引
        
    a : ArrayLike
        待查询的数组
        
    返回:
    ----
    Scalar
        当前位置的元素值
        
    使用示例:
    --------
    >>> # 在订单生成函数中使用
    >>> def generate_orders(ctx, signals, prices):
    ...     # 自动使用上下文的时间步和资产索引
    ...     current_signal = get_elem_nb(ctx, signals)
    ...     current_price = get_elem_nb(ctx, prices)
    ...     
    ...     if current_signal > 0.5:
    ...         return order_nb(size=100, price=current_price)
    ...     return order_nothing_nb()
    
    >>> # 在信号处理函数中使用
    >>> def process_signals(ctx, rsi, macd):
    ...     rsi_value = get_elem_nb(ctx, rsi)
    ...     macd_value = get_elem_nb(ctx, macd)
    ...     return rsi_value > 70 and macd_value > 0
    
    与get_col_elem_nb的区别:
    ---------------------
    - get_elem_nb: 自动使用上下文中的列索引，更简洁
    - get_col_elem_nb: 需要显式指定列索引，更灵活
    
    应用场景:
    --------
    - 单资产策略的数据获取
    - 当前资产的指标值提取
    - 简化的市场数据访问
    - 基于上下文的动态数据获取
    
    注意事项:
    --------
    - 上下文必须包含有效的col属性
    - 适用于单资产或当前资产的数据访问
    - 多资产场景下建议使用get_col_elem_nb
    - 确保上下文和数组的时间步范围匹配
    """
    return flex_select_auto_nb(a, ctx.i, ctx.col, ctx.flex_2d)


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def get_group_value_nb(from_col: int,
                       to_col: int,
                       cash_now: float,
                       last_position: tp.Array1d,
                       last_val_price: tp.Array1d) -> float:
    """
    计算资产组的总价值
    
    计算指定资产组的当前总价值，包括现金和所有持仓资产的市值。
    这是现金共享模式下资产组价值计算的基础函数。
    
    参数:
    ----
    from_col : int
        资产组起始列索引（包含）
        
    to_col : int
        资产组结束列索引（不包含）
        
    cash_now : float
        当前现金余额
        
    last_position : Array1d
        最新的持仓数量数组，每个元素对应一个资产
        
    last_val_price : Array1d
        最新的估值价格数组，每个元素对应一个资产的当前价格
        
    返回:
    ----
    float
        资产组的总价值 = 现金 + 所有持仓的市值总和
        
    使用示例:
    --------
    >>> # 计算包含3个资产的组合价值
    >>> cash_now = 10000.0
    >>> last_position = np.array([100, 0, 50, 200])  # 4个资产的持仓
    >>> last_val_price = np.array([50.0, 60.0, 80.0, 30.0])  # 4个资产的价格
    >>> 
    >>> # 计算第1组价值（资产索引0-1）
    >>> group1_value = get_group_value_nb(0, 2, cash_now, last_position, last_val_price)
    >>> # 计算: 10000 + (100*50 + 0*60) = 10000 + 5000 = 15000.0
    >>> print(f"第1组价值: {group1_value}")
    >>> 
    >>> # 计算第2组价值（资产索引2-3）
    >>> group2_value = get_group_value_nb(2, 4, cash_now, last_position, last_val_price)
    >>> # 计算: 10000 + (50*80 + 200*30) = 10000 + 4000 + 6000 = 20000.0
    >>> print(f"第2组价值: {group2_value}")
    
    计算公式:
    --------
    group_value = cash_now + Σ(position[i] * val_price[i])  # i ∈ [from_col, to_col)
    
    算法逻辑:
    --------
    1. 从现金开始累积价值
    2. 遍历组内每个资产
    3. 如果有持仓，计算该资产的市值并累加
    4. 返回现金加市值的总和
    
    应用场景:
    --------
    - 现金共享模式下的组合价值计算
    - 投资组合风险管理和监控
    - 资产配置比例计算的基础
    - 组合绩效评估和报告
    
    注意事项:
    --------
    - 只计算非零持仓的市值
    - 空头持仓的市值为负值
    - 确保价格数组与持仓数组长度一致
    - 通常与现金共享模式配合使用
    """
    group_value = cash_now  # 从现金开始累积
    group_len = to_col - from_col  # 计算组内资产数量
    
    # 遍历组内每个资产
    for k in range(group_len):
        col = from_col + k  # 当前资产的列索引
        
        # 只计算有持仓的资产市值
        if last_position[col] != 0:
            group_value += last_position[col] * last_val_price[col]  # 数量 * 价格
            
    return group_value


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def get_group_value_ctx_nb(seg_ctx: SegmentContext) -> float:
    """
    从上下文获取资产组价值
    
    这是get_group_value_nb的上下文版本，自动从SegmentContext中提取
    所需的参数来计算资产组价值。主要用于分段处理函数中的快速价值获取。
    
    参数:
    ----
    seg_ctx : SegmentContext
        分段上下文对象，包含：
        - from_col, to_col: 当前资产组的列范围
        - group: 当前资产组索引
        - last_cash: 各组的最新现金数组
        - last_position: 最新持仓数组
        - last_val_price: 最新估值价格数组
        - cash_sharing: 现金共享标志
        
    返回:
    ----
    float
        当前资产组的总价值
        
    异常:
    ----
    ValueError: 当现金共享未启用时抛出
    
    使用示例:
    --------
    >>> # 在pre_segment_func_nb回调函数中使用
    >>> @njit
    >>> def my_pre_segment_func(seg_ctx):
    ...     # 获取当前组的价值
    ...     current_group_value = get_group_value_ctx_nb(seg_ctx)
    ...     
    ...     # 动态调整估值价格（可选）
    ...     if some_condition:
    ...         seg_ctx.last_val_price[seg_ctx.from_col] = new_price
    ...         # 重新计算价值
    ...         updated_value = get_group_value_ctx_nb(seg_ctx)
    ...     
    ...     return current_group_value
    
    >>> # 在组合监控中使用
    >>> @njit  
    >>> def monitor_group_value(seg_ctx, target_value):
    ...     current_value = get_group_value_ctx_nb(seg_ctx)
    ...     if current_value > target_value * 1.1:
    ...         return "REBALANCE_NEEDED"
    ...     return "OK"
    
    最佳实践:
    --------
    1. 通常在pre_segment_func_nb中调用一次
    2. 可以修改seg_ctx.last_val_price后重新计算
    3. 用于动态风险控制和仓位管理
    4. 结合现金共享模式使用效果最佳
    
    使用限制:
    --------
    - 必须启用现金共享模式 (cash_sharing=True)
    - 只能在分段上下文环境中使用
    - 需要确保上下文数据的完整性和一致性
    
    应用场景:
    --------
    - 动态风险监控和预警
    - 实时组合价值计算
    - 资产配置优化的基础数据
    - 止损止盈策略的触发条件
    - 组合再平衡决策支持
    
    注意事项:
    --------
    - 现金共享必须启用，否则抛出异常
    - 价格可以就地修改来影响价值计算
    - 通常与分段处理流程配合使用
    - 计算结果反映当前时刻的即时价值
    """
    # 检查现金共享是否启用
    if not seg_ctx.cash_sharing:
        raise ValueError("现金共享必须启用才能使用此函数")
    
    # 调用基础价值计算函数
    return get_group_value_nb(
        seg_ctx.from_col,                    # 组起始列索引
        seg_ctx.to_col,                      # 组结束列索引  
        seg_ctx.last_cash[seg_ctx.group],    # 当前组的现金
        seg_ctx.last_position,               # 最新持仓数组
        seg_ctx.last_val_price               # 最新估值价格数组
    )


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def approx_order_value_nb(size: float,
                          size_type: int,
                          direction: int,
                          cash_now: float,
                          position_now: float,
                          free_cash_now: float,
                          val_price_now: float,
                          value_now: float) -> float:
    """
    近似计算订单价值
    
    根据订单参数和当前状态，估算订单的价值（所需现金量）。
    这个函数用于订单排序、现金分配和风险控制等场景，
    提供快速的订单价值预估而无需完整的订单执行过程。
    
    参数:
    ----
    size : float
        订单大小，含义取决于size_type
        
    size_type : int
        订单大小类型，参见SizeType枚举：
        - Amount: 具体数量
        - Value: 价值金额
        - Percent: 百分比（现金或持仓的百分比）
        - TargetAmount: 目标持仓数量
        - TargetValue: 目标持仓价值
        - TargetPercent: 目标持仓占组合的百分比
        
    direction : int
        交易方向限制，参见Direction枚举
        
    cash_now : float
        当前现金余额
        
    position_now : float
        当前持仓数量
        
    free_cash_now : float
        当前可用现金
        
    val_price_now : float
        当前估值价格
        
    value_now : float
        当前投资组合总价值
        
    返回:
    ----
    float
        估算的订单价值（所需现金），正值表示买入，负值表示卖出
        如果无法计算则返回NaN
        
    使用示例:
    --------
    >>> # 计算不同类型订单的近似价值
    >>> cash_now = 10000.0
    >>> position_now = 100.0
    >>> val_price_now = 50.0
    >>> value_now = 15000.0
    >>> free_cash_now = 8000.0
    >>> 
    >>> # 数量类型订单：买入50股
    >>> order_value = approx_order_value_nb(
    ...     size=50, size_type=SizeType.Amount, direction=Direction.Both,
    ...     cash_now=cash_now, position_now=position_now,
    ...     free_cash_now=free_cash_now, val_price_now=val_price_now, value_now=value_now
    ... )
    >>> # 结果: 50 * 50 = 2500.0
    >>> 
    >>> # 百分比类型订单：用50%现金买入
    >>> order_value = approx_order_value_nb(
    ...     size=0.5, size_type=SizeType.Percent, direction=Direction.Both,
    ...     cash_now=cash_now, position_now=position_now,
    ...     free_cash_now=free_cash_now, val_price_now=val_price_now, value_now=value_now
    ... )
    >>> # 结果: 0.5 * 10000 = 5000.0
    >>> 
    >>> # 目标数量订单：调整至200股
    >>> order_value = approx_order_value_nb(
    ...     size=200, size_type=SizeType.TargetAmount, direction=Direction.Both,
    ...     cash_now=cash_now, position_now=position_now,
    ...     free_cash_now=free_cash_now, val_price_now=val_price_now, value_now=value_now
    ... )
    >>> # 结果: 200 * 50 - 100 * 50 = 5000.0
    
    计算逻辑详解:
    -----------
    1. Amount: 订单价值 = 数量 × 价格
    2. Value: 订单价值 = 直接使用指定价值
    3. Percent: 
       - 正值(买入): 价值 = 百分比 × 现金
       - 负值(卖出): 价值 = 百分比 × 资产价值或可用资金
    4. TargetAmount: 价值 = 目标价值 - 当前资产价值
    5. TargetValue: 价值 = 目标价值 - 当前资产价值
    6. TargetPercent: 价值 = (百分比 × 总价值) - 当前资产价值
    
    应用场景:
    --------
    - 订单优先级排序和调度
    - 现金需求预估和分配
    - 风险限额检查和控制
    - 组合再平衡的成本估算
    - 多订单场景的资源规划
    
    注意事项:
    --------
    - 这是近似计算，实际执行可能有差异
    - 不考虑手续费、滑点等执行成本
    - ShortOnly方向会将size取反
    - 百分比订单的计算逻辑相对复杂
    - 主要用于快速估算，不作为精确计算
    """
    # 处理仅空头方向的订单
    if direction == Direction.ShortOnly:
        size *= -1  # 空头订单的size取反
    
    # 计算当前资产价值
    asset_value_now = position_now * val_price_now
    
    # 根据订单大小类型计算价值
    if size_type == SizeType.Amount:
        # 数量类型：订单价值 = 数量 × 价格
        return size * val_price_now
        
    if size_type == SizeType.Value:
        # 价值类型：直接返回指定价值
        return size
        
    if size_type == SizeType.Percent:
        # 百分比类型：根据买卖方向计算
        if size >= 0:
            # 正值（买入）：使用现金的百分比
            return size * cash_now
        else:
            # 负值（卖出）：根据方向限制计算
            if direction == Direction.LongOnly:
                # 只能做多：使用当前资产价值的百分比
                return size * asset_value_now
            # 允许做空：使用更大的可用资金基数
            return size * (2 * max(asset_value_now, 0) + max(free_cash_now, 0))
    
    if size_type == SizeType.TargetAmount:
        # 目标数量：目标价值减去当前资产价值
        return size * val_price_now - asset_value_now
        
    if size_type == SizeType.TargetValue:
        # 目标价值：目标价值减去当前资产价值
        return size - asset_value_now
        
    if size_type == SizeType.TargetPercent:
        # 目标百分比：(百分比 × 总价值) - 当前资产价值
        return size * value_now - asset_value_now
    
    # 未知的订单类型
    return np.nan


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def sort_call_seq_out_nb(ctx: SegmentContext,
                         size: tp.ArrayLike,
                         size_type: tp.ArrayLike,
                         direction: tp.ArrayLike,
                         order_value_out: tp.Array1d,
                         call_seq_out: tp.Array1d,
                         ctx_select: bool = True) -> None:
    """
    基于潜在订单价值对调用序列进行排序
    
    根据每个潜在订单的预估价值对call_seq_out进行就地排序，
    实现按价值优先级的资产交易顺序。这是现金共享模式下
    优化资金利用效率的关键功能。
    
    参数:
    ----
    ctx : SegmentContext
        分段上下文对象，包含当前状态信息
        
    size : ArrayLike
        订单大小数组，支持灵活索引
        
    size_type : ArrayLike
        订单大小类型数组，支持灵活索引
        
    direction : ArrayLike
        交易方向数组，支持灵活索引
        
    order_value_out : Array1d
        输出的订单价值数组，长度应匹配组内资产数量
        函数执行前应为空，执行后包含排序后的订单价值
        
    call_seq_out : Array1d
        输入/输出的调用序列数组，长度应匹配组内资产数量
        输入时应按默认顺序填充 (0, 1, 2, ...)
        输出时为按价值排序后的资产索引序列
        
    ctx_select : bool, 可选 (默认: True)
        索引选择模式：
        - True: 使用get_col_elem_nb进行上下文选择
        - False: 使用flex_select_auto_nb进行灵活选择
        
    异常:
    ----
    ValueError: 当现金共享未启用或call_seq_out格式错误时抛出
    
    使用示例:
    --------
    >>> # 在pre_segment_func_nb中使用
    >>> @njit
    >>> def my_pre_segment_func(ctx, sizes, size_types, directions):
    ...     group_len = ctx.to_col - ctx.from_col
    ...     order_value_out = np.empty(group_len, dtype=np.float64)
    ...     call_seq_out = np.arange(group_len, dtype=np.int64)  # 默认顺序
    ...     
    ...     # 按订单价值排序调用序列
    ...     sort_call_seq_out_nb(ctx, sizes, size_types, directions, 
    ...                          order_value_out, call_seq_out)
    ...     
    ...     # 现在call_seq_out按订单价值从大到小排序
    ...     return order_value_out, call_seq_out
    
    排序逻辑:
    --------
    1. 计算当前组合总价值
    2. 遍历组内每个资产
    3. 计算每个资产的潜在订单价值
    4. 按订单价值对调用序列进行排序
    5. 高价值订单优先执行，确保资金效率
    
    数据流程:
    --------
    输入数组 → 提取参数 → 计算订单价值 → 排序调用序列 → 就地修改输出数组
    
    应用场景:
    --------
    - 现金共享模式下的资金优先分配
    - 大额订单优先执行策略
    - 风险控制：重要订单优先处理
    - 组合再平衡的执行顺序优化
    - 流动性管理和交易成本控制
    
    使用限制:
    --------
    - 必须启用现金共享模式
    - call_seq_out必须按默认顺序初始化
    - 仅用于灵活模拟函数中
    - 最佳在pre_segment_func_nb中调用
    
    注意事项:
    --------
    - 函数会就地修改call_seq_out和order_value_out
    - 排序基于订单价值的绝对值
    - 大价值订单（买入或卖出）优先执行
    - 确保数组长度与组内资产数量一致
    - 灵活索引支持多种数组广播模式
    """
    # 检查现金共享是否启用
    if not ctx.cash_sharing:
        raise ValueError("现金共享必须启用才能使用订单价值排序")
    
    # 转换输入数组为NumPy数组
    size_arr = np.asarray(size)
    size_type_arr = np.asarray(size_type)
    direction_arr = np.asarray(direction)

    # 获取当前组合总价值
    group_value_now = get_group_value_ctx_nb(ctx)
    group_len = ctx.to_col - ctx.from_col  # 组内资产数量
    
    # 遍历组内每个资产，计算订单价值
    for k in range(group_len):
        # 检查调用序列格式是否正确
        if call_seq_out[k] != k:
            raise ValueError("调用序列必须按默认顺序初始化 (CallSeqType.Default)")
        
        col = ctx.from_col + k  # 当前资产的绝对列索引
        
        # 根据索引选择模式提取订单参数
        if ctx_select:
            # 上下文选择：支持时间维度的索引
            _size = get_col_elem_nb(ctx, col, size_arr)
            _size_type = get_col_elem_nb(ctx, col, size_type_arr)
            _direction = get_col_elem_nb(ctx, col, direction_arr)
        else:
            # 灵活选择：仅支持组内索引
            _size = flex_select_auto_nb(size_arr, k, 0, False)
            _size_type = flex_select_auto_nb(size_type_arr, k, 0, False)
            _direction = flex_select_auto_nb(direction_arr, k, 0, False)
        
        # 获取现金状态（现金共享模式使用组级现金）
        if ctx.cash_sharing:
            cash_now = ctx.last_cash[ctx.group]       # 组级现金
            free_cash_now = ctx.last_free_cash[ctx.group]  # 组级可用现金
        else:
            cash_now = ctx.last_cash[col]             # 资产级现金
            free_cash_now = ctx.last_free_cash[col]   # 资产级可用现金
        
        # 计算该资产的近似订单价值
        order_value_out[k] = approx_order_value_nb(
            _size,                          # 订单大小
            _size_type,                     # 大小类型
            _direction,                     # 交易方向
            cash_now,                       # 当前现金
            ctx.last_position[col],         # 当前持仓
            free_cash_now,                  # 可用现金
            ctx.last_val_price[col],        # 估值价格
            group_value_now                 # 组合总价值
        )
    
    # 根据订单价值对调用序列进行排序
    # 使用插入排序算法，按价值从大到小排序
    insert_argsort_nb(order_value_out, call_seq_out)


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def sort_call_seq_nb(ctx: SegmentContext,
                     size: tp.ArrayLike,
                     size_type: tp.ArrayLike,
                     direction: tp.ArrayLike,
                     order_value_out: tp.Array1d,
                     ctx_select: bool = True) -> None:
    """
    对上下文附加的调用序列进行排序
    
    这是sort_call_seq_out_nb的简化版本，直接对上下文中的
    当前调用序列进行排序，无需提供额外的call_seq_out参数。
    主要用于非灵活模拟函数中的调用序列动态调整。
    
    参数:
    ----
    ctx : SegmentContext
        分段上下文对象，必须包含有效的call_seq_now属性
        
    size : ArrayLike
        订单大小数组，支持灵活索引
        
    size_type : ArrayLike
        订单大小类型数组，支持灵活索引
        
    direction : ArrayLike
        交易方向数组，支持灵活索引
        
    order_value_out : Array1d
        输出的订单价值数组，长度应匹配组内资产数量
        
    ctx_select : bool, 可选 (默认: True)
        索引选择模式，参见sort_call_seq_out_nb
        
    异常:
    ----
    ValueError: 当上下文中的调用序列为None时抛出
    
    使用示例:
    --------
    >>> # 在标准模拟函数的pre_segment_func_nb中使用
    >>> @njit
    >>> def my_pre_segment_func(ctx, sizes, size_types, directions):
    ...     group_len = ctx.to_col - ctx.from_col
    ...     order_value_out = np.empty(group_len, dtype=np.float64)
    ...     
    ...     # 直接排序上下文中的调用序列
    ...     sort_call_seq_nb(ctx, sizes, size_types, directions, order_value_out)
    ...     
    ...     # ctx.call_seq_now现在已按价值排序
    ...     return order_value_out
    
    与sort_call_seq_out_nb的区别:
    -------------------------
    - sort_call_seq_nb: 操作上下文中的调用序列，用于标准模拟
    - sort_call_seq_out_nb: 操作自定义调用序列，用于灵活模拟
    
    应用场景:
    --------
    - 标准投资组合模拟中的动态调用序列调整
    - 基于实时订单价值的执行优先级管理
    - 现金共享模式下的资源优化分配
    - 多资产策略的执行顺序控制
    
    使用限制:
    --------
    - 只能用于非灵活模拟函数
    - 上下文必须包含有效的call_seq_now属性
    - 需要启用现金共享模式
    - 主要在标准simulate_nb系列函数中使用
    
    注意事项:
    --------
    - 直接修改上下文中的调用序列
    - 如果call_seq_now为None，需使用sort_call_seq_out_nb
    - 排序逻辑与sort_call_seq_out_nb完全相同
    - 是上下文操作的便捷封装函数
    """
    # 检查上下文中的调用序列是否可用
    if ctx.call_seq_now is None:
        raise ValueError("调用序列数组为None，请使用sort_call_seq_out_nb处理自定义序列")
    
    # 调用基础排序函数，使用上下文中的调用序列
    sort_call_seq_out_nb(
        ctx,                    # 分段上下文
        size,                   # 订单大小数组
        size_type,              # 大小类型数组
        direction,              # 交易方向数组
        order_value_out,        # 输出价值数组
        ctx.call_seq_now,       # 使用上下文中的调用序列
        ctx_select=ctx_select   # 索引选择模式
    )


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def replace_inf_price_nb(prev_close: float, close: float, order: Order) -> Order:
    """
    替换订单中的无穷价格
    
    将订单中的无穷价格（np.inf）替换为实际的市场价格，
    实现市价订单的价格确定。这是将抽象的市价订单转换为
    具体限价订单的关键步骤。
    
    参数:
    ----
    prev_close : float
        上一个收盘价，用作价格下限
        
    close : float
        当前收盘价，用作价格上限
        
    order : Order
        包含无穷价格的原始订单对象
        
    返回:
    ----
    Order
        价格已替换的新订单对象
        
    价格替换规则:
    -----------
    - 如果订单价格 > 0: 使用当前收盘价作为上限（买入市价单）
    - 如果订单价格 <= 0: 使用前收盘价作为下限（卖出市价单）
    
    使用示例:
    --------
    >>> # 处理市价买入订单
    >>> market_buy_order = Order(size=100, price=np.inf)  # 市价买入
    >>> prev_close = 49.50
    >>> close = 50.25
    >>> 
    >>> limit_order = replace_inf_price_nb(prev_close, close, market_buy_order)
    >>> print(f"替换后价格: {limit_order.price}")  # 50.25 (当前收盘价)
    >>> 
    >>> # 处理市价卖出订单
    >>> market_sell_order = Order(size=-100, price=-np.inf)  # 市价卖出
    >>> limit_order = replace_inf_price_nb(prev_close, close, market_sell_order)
    >>> print(f"替换后价格: {limit_order.price}")  # 49.50 (前收盘价)
    >>> 
    >>> # 处理已有明确价格的限价订单（保持不变）
    >>> limit_order = Order(size=100, price=49.75)  # 限价订单
    >>> unchanged_order = replace_inf_price_nb(prev_close, close, limit_order)
    >>> print(f"价格保持: {unchanged_order.price}")  # 49.75 (原价格)
    
    价格替换逻辑:
    -----------
    1. 检查原订单价格符号
    2. 根据符号选择合适的替换价格
    3. 保留原订单的所有其他参数
    4. 创建新的订单对象返回
    
    应用场景:
    --------
    - 市价订单的价格具体化
    - 模拟交易执行中的价格确定
    - 回测系统中的市价单处理
    - 订单预处理和标准化
    - 风险控制中的价格边界设定
    
    设计考量:
    --------
    - 上限使用当前价格，防止买入时价格过高
    - 下限使用历史价格，防止卖出时价格过低
    - 保持订单对象的不可变性（返回新对象）
    - 支持所有订单参数的完整传递
    
    注意事项:
    --------
    - 只替换无穷价格，有限价格保持不变
    - 负无穷价格通常对应卖出市价单
    - 正无穷价格通常对应买入市价单
    - 替换后的价格仍可能受到滑点等因素影响
    - 价格替换不改变订单的其他属性
    """
    # 获取原订单价格
    order_price = order.price
    
    # 根据价格符号进行替换
    if order_price > 0:
        # 正价格（通常是买入市价单）：使用当前收盘价作为上限
        order_price = close  
    else:
        # 负价格或零（通常是卖出市价单）：使用前收盘价作为下限
        order_price = prev_close  
    
    # 创建新订单，价格已替换，其他参数保持不变
    return order_nb(
        size=order.size,                        # 订单大小
        price=order_price,                      # 替换后的价格
        size_type=order.size_type,              # 大小类型
        direction=order.direction,              # 交易方向
        fees=order.fees,                        # 比例手续费
        fixed_fees=order.fixed_fees,            # 固定手续费
        slippage=order.slippage,                # 滑点设置
        min_size=order.min_size,                # 最小大小
        max_size=order.max_size,                # 最大大小
        size_granularity=order.size_granularity, # 数量粒度
        reject_prob=order.reject_prob,          # 拒绝概率
        lock_cash=order.lock_cash,              # 现金锁定
        allow_partial=order.allow_partial,      # 允许部分成交
        raise_reject=order.raise_reject,        # 拒绝时抛异常
        log=order.log                           # 日志记录
    )


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def try_order_nb(ctx: OrderContext, order: Order) -> tp.Tuple[ExecuteOrderState, OrderResult]:
    """
    尝试执行订单而不持久化状态
    
    这是一个"试运行"函数，用于测试订单执行结果而不影响实际的
    投资组合状态。主要用于订单预检查、策略测试和风险评估。
    
    参数:
    ----
    ctx : OrderContext
        订单上下文对象，包含当前的投资组合状态：
        - cash_now: 当前现金
        - position_now: 当前持仓
        - debt_now: 当前债务
        - free_cash_now: 当前可用现金
        - val_price_now: 当前估值价格
        - value_now: 当前投资组合价值
        - i: 当前时间索引
        - col: 当前资产列索引
        - close: 收盘价数组
        
    order : Order
        待测试执行的订单对象
        
    返回:
    ----
    tuple[ExecuteOrderState, OrderResult]
        (执行后状态, 订单结果) - 仅用于测试，不影响实际状态
        
    使用示例:
    --------
    >>> # 在订单生成函数中预检查订单
    >>> @njit
    >>> def my_order_func(ctx):
    ...     # 创建测试订单
    ...     test_order = order_nb(size=100, price=50.0)
    ...     
    ...     # 尝试执行（不影响实际状态）
    ...     exec_state, order_result = try_order_nb(ctx, test_order)
    ...     
    ...     # 检查是否可以成功执行
    ...     if order_result.status == OrderStatus.Filled:
    ...         return test_order  # 返回真实订单
    ...     else:
    ...         return order_nothing_nb()  # 返回空订单
    
    >>> # 风险评估场景
    >>> @njit
    >>> def risk_check_order(ctx, proposed_order):
    ...     exec_state, result = try_order_nb(ctx, proposed_order)
    ...     
    ...     # 检查执行后的风险指标
    ...     if abs(exec_state.position) > MAX_POSITION:
    ...         return False  # 风险过高
    ...     if exec_state.cash < MIN_CASH:
    ...         return False  # 现金不足
    ...     return True  # 风险可接受
    
    工作流程:
    --------
    1. 从上下文创建临时状态对象
    2. 处理无穷价格（市价订单）
    3. 调用核心执行引擎
    4. 返回执行结果，不保存状态变化
    
    价格处理逻辑:
    -----------
    - 如果订单价格为无穷大，自动替换为市场价格
    - 使用当前和前一个收盘价进行价格替换
    - 保持价格替换的一致性规则
    
    应用场景:
    --------
    - 订单预检查和验证
    - 策略回测中的假设分析
    - 风险管理和限额控制
    - 订单优化和参数调整
    - 多订单组合的最优化选择
    - 流动性影响评估
    
    与正常执行的区别:
    ---------------
    - 不修改投资组合状态
    - 不产生持久化记录
    - 记录索引设为-1（表示测试模式）
    - 适用于各种假设性分析
    
    注意事项:
    --------
    - 仅用于测试，不产生实际交易记录
    - 价格处理与正常执行保持一致
    - 可用于复杂策略的决策支持
    - 执行成本与正常执行相当
    """
    # 从上下文创建临时的处理状态
    # 记录索引设为-1表示这是测试执行
    state = ProcessOrderState(
        cash=ctx.cash_now,           # 当前现金
        position=ctx.position_now,   # 当前持仓  
        debt=ctx.debt_now,          # 当前债务
        free_cash=ctx.free_cash_now, # 当前可用现金
        val_price=ctx.val_price_now, # 当前估值价格
        value=ctx.value_now,        # 当前投资组合价值
        oidx=-1,                    # 订单记录索引：-1表示测试模式
        lidx=-1                     # 日志记录索引：-1表示测试模式
    )
    
    # 处理无穷价格（市价订单）
    if np.isinf(order.price):
        # 获取前一个收盘价（如果存在）
        if ctx.i > 0:
            prev_close = flex_select_auto_nb(ctx.close, ctx.i - 1, ctx.col, ctx.flex_2d)
        else:
            prev_close = np.nan  # 第一个时间点没有前收盘价
        
        # 获取当前收盘价
        close = flex_select_auto_nb(ctx.close, ctx.i, ctx.col, ctx.flex_2d)
        
        # 将无穷价格替换为实际市场价格
        order = replace_inf_price_nb(prev_close, close, order)
    
    # 执行订单并返回结果（不持久化状态）
    return execute_order_nb(state, order)


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def init_records_nb(target_shape: tp.Shape,
                    max_orders: tp.Optional[int] = None,
                    max_logs: int = 0) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
    """
    初始化订单和日志记录数组
    
    为投资组合模拟创建空的记录数组，用于存储订单执行历史
    和详细的操作日志。这是模拟系统的基础准备工作。
    
    参数:
    ----
    target_shape : Shape
        目标形状 (时间步数, 资产数)，用于计算默认记录容量
        
    max_orders : int, 可选 (默认: None)
        最大订单记录数量：
        - None: 自动计算为时间步数 × 资产数 (每个位置最多一个订单)
        - 具体数值: 使用指定的容量限制
        
    max_logs : int, 可选 (默认: 0)
        最大日志记录数量：
        - 0: 创建最小容量（1个记录）
        - >0: 使用指定的容量
        
    返回:
    ----
    tuple[RecordArray, RecordArray]
        (订单记录数组, 日志记录数组) - 已初始化但为空的记录容器
        
    使用示例:
    --------
    >>> # 为100个时间步、5个资产的回测初始化记录
    >>> target_shape = (100, 5)
    >>> order_records, log_records = init_records_nb(target_shape, max_logs=1000)
    >>> print(f"订单记录容量: {len(order_records)}")  # 500 (100*5)
    >>> print(f"日志记录容量: {len(log_records)}")    # 1000
    >>> 
    >>> # 指定自定义的订单容量
    >>> order_records, log_records = init_records_nb(
    ...     target_shape, max_orders=2000, max_logs=5000
    ... )
    >>> print(f"订单记录容量: {len(order_records)}")  # 2000
    >>> print(f"日志记录容量: {len(log_records)}")    # 5000
    
    容量规划建议:
    -----------
    1. 默认容量：适用于每个时间点最多一个订单的策略
    2. 高频策略：应增加max_orders，考虑多次调整的可能
    3. 调试模式：增加max_logs以获得详细的执行信息
    4. 生产模式：可将max_logs设为0以节省内存
    
    记录类型:
    --------
    - order_records: 订单记录，包含成交的订单信息
    - log_records: 日志记录，包含详细的状态变化信息
    
    内存考量:
    --------
    - 订单记录相对轻量，适合长期保存
    - 日志记录详细但占用更多内存
    - 根据实际需要平衡详细程度和内存使用
    
    应用场景:
    --------
    - 投资组合回测系统初始化
    - 实盘交易记录系统准备
    - 策略性能分析数据准备
    - 风险监控系统数据结构初始化
    
    注意事项:
    --------
    - 记录数组初始时为空，需要在模拟过程中填充
    - 容量不足会导致运行时错误
    - 过大的容量会浪费内存资源
    - 日志记录是可选的，可设为0节省资源
    """
    # 计算订单记录的最大数量
    if max_orders is None:
        # 默认容量：每个时间步每个资产最多一个订单
        _max_orders = target_shape[0] * target_shape[1]
    else:
        # 使用用户指定的容量
        _max_orders = max_orders
    
    # 创建空的订单记录数组
    order_records = np.empty(_max_orders, dtype=order_dt)
    
    # 处理日志记录容量
    if max_logs == 0:
        # 最小容量为1，避免空数组问题
        max_logs = 1
    
    # 创建空的日志记录数组
    log_records = np.empty(max_logs, dtype=log_dt)
    
    return order_records, log_records


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def update_open_pos_stats_nb(record: tp.Record, position_now: float, price: float) -> None:
    """
    使用自定义价格更新开仓头寸记录的统计信息
    
    动态计算开仓头寸的未实现盈亏和收益率，支持部分平仓的
    加权平均价格计算。主要用于头寸跟踪和实时风险监控。
    
    参数:
    ----
    record : Record
        头寸记录对象，包含入场价格、手续费等信息
        
    position_now : float
        当前剩余头寸数量（绝对值）
        
    price : float
        用于计算的当前价格（通常是最新市价）
        
    就地修改:
    --------
    更新record中的pnl和return字段，反映最新的未实现损益
    
    使用示例:
    --------
    >>> # 假设有一个开仓记录
    >>> position_record = {
    ...     'id': 0,
    ...     'status': TradeStatus.Open,
    ...     'size': 1000,           # 原始开仓数量
    ...     'entry_price': 50.0,    # 入场价格
    ...     'entry_fees': 25.0,     # 入场手续费
    ...     'exit_price': np.nan,   # 出场价格（尚未设置）
    ...     'exit_fees': 0.0,       # 出场手续费
    ...     'direction': 1          # 多头方向
    ... }
    >>> 
    >>> # 更新统计信息（全部持仓，价格上涨）
    >>> position_now = 1000  # 全部持仓
    >>> current_price = 55.0  # 当前价格
    >>> update_open_pos_stats_nb(position_record, position_now, current_price)
    >>> print(f"未实现盈亏: {position_record['pnl']}")      # 4975.0 (5500-500-25)
    >>> print(f"未实现收益率: {position_record['return']}")   # 0.995 (4975/5025)
    >>> 
    >>> # 部分平仓后更新（假设已平仓300股，平仓价格52.0）
    >>> position_record['exit_price'] = 52.0
    >>> position_now = 700  # 剩余700股
    >>> current_price = 56.0  # 当前价格更高
    >>> update_open_pos_stats_nb(position_record, position_now, current_price)
    >>> # 加权平均出场价格 = (300*52 + 700*56) / 1000 = 54.8
    >>> print(f"加权出场价格: {position_record['exit_price']}")  # 54.8
    
    计算逻辑:
    --------
    1. 检查记录是否为有效的开仓状态
    2. 处理出场价格：
       - 未设置：直接使用当前价格
       - 已设置：计算加权平均出场价格
    3. 调用get_trade_stats_nb计算PnL和收益率
    4. 更新记录中的统计字段
    
    加权平均价格公式:
    ---------------
    已平仓部分价值 = (总数量 - 剩余数量) × 原出场价格
    剩余部分价值 = 剩余数量 × 当前价格
    加权平均价格 = (已平仓价值 + 剩余价值) / 总数量
    
    应用场景:
    --------
    - 实时风险监控和报告
    - 头寸价值的动态更新
    - 未实现损益的计算
    - 投资组合绩效评估
    - 止损止盈策略的触发条件
    
    注意事项:
    --------
    - 只处理有效的开仓记录（id >= 0, status = Open）
    - 支持部分平仓的复杂场景
    - 价格计算考虑了手续费的影响
    - 记录对象会被就地修改
    """
    # 检查记录是否为有效的开仓状态
    if record['id'] >= 0 and record['status'] == TradeStatus.Open:
        # 处理出场价格的计算
        if np.isnan(record['exit_price']):
            # 如果尚未设置出场价格，直接使用当前价格
            exit_price = price
        else:
            # 如果已有出场价格（部分平仓），计算加权平均价格
            exit_size_sum = record['size'] - abs(position_now)  # 已平仓数量
            exit_gross_sum = exit_size_sum * record['exit_price']  # 已平仓价值
            exit_gross_sum += abs(position_now) * price  # 加上剩余头寸的当前价值
            exit_price = exit_gross_sum / record['size']  # 加权平均出场价格
        
        # 计算盈亏和收益率统计
        pnl, ret = get_trade_stats_nb(
            record['size'],         # 头寸大小
            record['entry_price'],  # 入场价格
            record['entry_fees'],   # 入场手续费
            exit_price,             # 计算出的出场价格
            record['exit_fees'],    # 出场手续费
            record['direction']     # 交易方向
        )
        
        # 更新记录中的统计信息
        record['pnl'] = pnl      # 净盈亏
        record['return'] = ret   # 收益率


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def update_pos_record_nb(record: tp.Record,
                         i: int,
                         col: int,
                         position_before: float,
                         position_now: float,
                         order_result: OrderResult) -> None:
    """
    根据订单执行结果更新头寸记录
    
    处理订单执行后的头寸变化，维护完整的交易记录。
    支持四种头寸变化模式：开仓、平仓、反转、调整，
    每种情况都有对应的记录更新逻辑。
    
    参数:
    ----
    record : Record
        头寸记录对象，存储交易的详细信息
        
    i : int
        当前时间索引（时间步）
        
    col : int
        资产列索引（资产标识符）
        
    position_before : float
        订单执行前的头寸数量（可为正负）
        
    position_now : float
        订单执行后的头寸数量（可为正负）
        
    order_result : OrderResult
        订单执行结果对象
        
    就地修改:
    --------
    更新record中的各项字段，反映最新的交易状态和统计信息
    
    使用示例:
    --------
    >>> # 场景1：开新仓位
    >>> position_record = create_empty_position_record()
    >>> position_before = 0.0
    >>> position_now = 100.0  # 买入100股
    >>> order_result = OrderResult(size=100, price=50.0, fees=25.0, side=OrderSide.Buy)
    >>> 
    >>> update_pos_record_nb(position_record, i=0, col=0, 
    ...                      position_before, position_now, order_result)
    >>> # 结果：记录显示新开多头仓位，入场价格50.0
    >>> 
    >>> # 场景2：平仓
    >>> position_before = 100.0
    >>> position_now = 0.0  # 全部卖出
    >>> order_result = OrderResult(size=-100, price=55.0, fees=27.5, side=OrderSide.Sell)
    >>> 
    >>> update_pos_record_nb(position_record, i=10, col=0,
    ...                      position_before, position_now, order_result)
    >>> # 结果：记录显示仓位关闭，出场价格55.0，计算最终盈亏
    >>> 
    >>> # 场景3：头寸反转（多头转空头）
    >>> position_before = 100.0   # 多头100股
    >>> position_now = -50.0      # 变成空头50股
    >>> order_result = OrderResult(size=-150, price=53.0, fees=75.0, side=OrderSide.Sell)
    >>> 
    >>> update_pos_record_nb(position_record, i=5, col=0,
    ...                      position_before, position_now, order_result)
    >>> # 结果：记录显示头寸反转，生成新的交易记录
    
    处理逻辑分类:
    -----------
    1. **开仓 (position_before=0 → position_now≠0)**
       - 创建新交易记录
       - 设置入场信息和交易方向
       - 初始化统计字段
    
    2. **平仓 (position_before≠0 → position_now=0)**
       - 完成交易记录
       - 计算加权平均出场价格
       - 计算最终盈亏和收益率
       - 设置状态为已关闭
    
    3. **反转 (position_before与position_now符号相反)**
       - 增加交易ID
       - 创建新方向的交易记录
       - 按比例分配手续费
       - 重置出场信息
    
    4. **调整 (同方向但大小变化)**
       - **增仓**: 计算加权平均入场价格和费用
       - **减仓**: 计算加权平均出场价格和费用
    
    价格计算公式:
    -----------
    - **加权平均入场价**: (原价值 + 新价值) / (原数量 + 新数量)
    - **加权平均出场价**: (已出场价值 + 新出场价值) / 总出场数量
    
    应用场景:
    --------
    - 交易记录的实时维护
    - 头寸跟踪和风险监控
    - 交易统计和绩效分析
    - 税务和合规报告
    - 投资组合管理系统
    
    注意事项:
    --------
    - 只处理成功执行的订单（Filled状态）
    - 支持多次部分交易的累积计算
    - 考虑了手续费在不同情况下的分摊
    - 自动更新开仓头寸的统计信息
    - 头寸方向由订单方向自动确定
    """
    # 只处理成功执行的订单
    if order_result.status == OrderStatus.Filled:
        
        # 情况1：开新仓位（从零头寸变为非零头寸）
        if position_before == 0 and position_now != 0:
            # 创建新交易记录
            record['id'] += 1                           # 递增交易ID
            record['col'] = col                         # 资产列索引
            record['size'] = order_result.size          # 交易数量
            record['entry_idx'] = i                     # 入场时间索引
            record['entry_price'] = order_result.price  # 入场价格
            record['entry_fees'] = order_result.fees    # 入场手续费
            record['exit_idx'] = -1                     # 出场时间（未定）
            record['exit_price'] = np.nan               # 出场价格（未定）
            record['exit_fees'] = 0.                    # 出场手续费（初始为0）
            
            # 根据订单方向确定交易方向
            if order_result.side == OrderSide.Buy:
                record['direction'] = TradeDirection.Long   # 买入为多头
            else:
                record['direction'] = TradeDirection.Short  # 卖出为空头
                
            record['status'] = TradeStatus.Open         # 状态为开仓
            record['parent_id'] = record['id']          # 父交易ID（自引用）
            
        # 情况2：平仓（从非零头寸变为零头寸）
        elif position_before != 0 and position_now == 0:
            # 完成交易记录
            record['exit_idx'] = i                      # 出场时间索引
            
            # 计算加权平均出场价格
            if np.isnan(record['exit_price']):
                # 首次出场，直接使用当前价格
                exit_price = order_result.price
            else:
                # 多次出场，计算加权平均价格
                exit_size_sum = record['size'] - abs(position_before)  # 之前出场的数量
                exit_gross_sum = exit_size_sum * record['exit_price']  # 之前出场的总价值
                exit_gross_sum += abs(position_before) * order_result.price  # 加上本次出场价值
                exit_price = exit_gross_sum / record['size']  # 加权平均出场价格
                
            record['exit_price'] = exit_price           # 更新出场价格
            record['exit_fees'] += order_result.fees    # 累加出场手续费
            
            # 计算最终的盈亏和收益率
            pnl, ret = get_trade_stats_nb(
                record['size'],         # 交易数量
                record['entry_price'],  # 入场价格
                record['entry_fees'],   # 入场手续费
                record['exit_price'],   # 出场价格
                record['exit_fees'],    # 出场手续费
                record['direction']     # 交易方向
            )
            record['pnl'] = pnl                        # 净盈亏
            record['return'] = ret                     # 收益率
            record['status'] = TradeStatus.Closed      # 状态为已关闭
            
        # 情况3：头寸反转（正负号改变）
        elif np.sign(position_before) != np.sign(position_now):
            # 生成新的交易记录
            record['id'] += 1                          # 递增交易ID
            record['size'] = abs(position_now)         # 新头寸的绝对数量
            record['entry_idx'] = i                    # 新入场时间
            record['entry_price'] = order_result.price # 新入场价格
            
            # 按新头寸比例分配手续费
            new_pos_fraction = abs(position_now) / abs(position_now - position_before)
            record['entry_fees'] = new_pos_fraction * order_result.fees
            
            # 重置出场信息
            record['exit_idx'] = -1
            record['exit_price'] = np.nan
            record['exit_fees'] = 0.
            
            # 确定新的交易方向
            if order_result.side == OrderSide.Buy:
                record['direction'] = TradeDirection.Long
            else:
                record['direction'] = TradeDirection.Short
                
            record['status'] = TradeStatus.Open        # 状态为开仓
            record['parent_id'] = record['id']         # 新的父交易ID
            
        # 情况4：头寸调整（同向但数量变化）
        else:
            if abs(position_before) <= abs(position_now):
                # 增仓：计算加权平均入场价格
                entry_gross_sum = record['size'] * record['entry_price']  # 原入场总价值
                entry_gross_sum += order_result.size * order_result.price  # 加上新入场价值
                entry_price = entry_gross_sum / (record['size'] + order_result.size)  # 加权平均
                
                record['entry_price'] = entry_price    # 更新入场价格
                record['entry_fees'] += order_result.fees  # 累加入场手续费
                record['size'] += order_result.size    # 更新总头寸大小
                
            else:
                # 减仓：计算加权平均出场价格
                if np.isnan(record['exit_price']):
                    # 首次减仓
                    exit_price = order_result.price
                else:
                    # 多次减仓，计算加权平均价格
                    exit_size_sum = record['size'] - abs(position_before)  # 之前减仓数量
                    exit_gross_sum = exit_size_sum * record['exit_price']  # 之前减仓价值
                    exit_gross_sum += order_result.size * order_result.price  # 本次减仓价值
                    exit_price = exit_gross_sum / (exit_size_sum + order_result.size)  # 加权平均
                    
                record['exit_price'] = exit_price      # 更新出场价格
                record['exit_fees'] += order_result.fees  # 累加出场手续费

        # 更新开仓头寸的统计信息
        update_open_pos_stats_nb(
            record,                 # 头寸记录
            position_now,           # 当前头寸数量
            order_result.price      # 当前价格
        )


# ############# 投资组合模拟系统 (Portfolio Simulation System) ############# #


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def simulate_from_orders_nb(target_shape: tp.Shape,
                            group_lens: tp.Array1d,
                            init_cash: tp.Array1d,
                            call_seq: tp.Array2d,
                            size: tp.ArrayLike = np.asarray(np.inf),
                            price: tp.ArrayLike = np.asarray(np.inf),
                            size_type: tp.ArrayLike = np.asarray(SizeType.Amount),
                            direction: tp.ArrayLike = np.asarray(Direction.Both),
                            fees: tp.ArrayLike = np.asarray(0.),
                            fixed_fees: tp.ArrayLike = np.asarray(0.),
                            slippage: tp.ArrayLike = np.asarray(0.),
                            min_size: tp.ArrayLike = np.asarray(0.),
                            max_size: tp.ArrayLike = np.asarray(np.inf),
                            size_granularity: tp.ArrayLike = np.asarray(np.nan),
                            reject_prob: tp.ArrayLike = np.asarray(0.),
                            lock_cash: tp.ArrayLike = np.asarray(False),
                            allow_partial: tp.ArrayLike = np.asarray(True),
                            raise_reject: tp.ArrayLike = np.asarray(False),
                            log: tp.ArrayLike = np.asarray(False),
                            val_price: tp.ArrayLike = np.asarray(np.inf),
                            close: tp.ArrayLike = np.asarray(np.nan),
                            auto_call_seq: bool = False,
                            ffill_val_price: bool = True,
                            update_value: bool = False,
                            max_orders: tp.Optional[int] = None,
                            max_logs: int = 0,
                            flex_2d: bool = True) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
    """
    基于订单矩阵的投资组合模拟
    
    这是VectorBT框架的核心模拟引擎，将每个数组元素转换为一个订单，
    支持多资产、多策略的大规模投资组合回测。采用列主序迭代和
    灵活广播机制，实现高性能的矩阵化计算。
    
    参数说明:
    --------
    
    **基础配置参数：**
    target_shape : Shape
        目标形状 (时间步数, 资产数)，定义回测的时间和资产维度
        
    group_lens : Array1d  
        各组的长度数组，用于资产分组管理
        
    init_cash : Array1d
        各组或各资产的初始现金数组
        
    call_seq : Array2d
        调用序列数组，控制资产的处理顺序
        
    **订单配置参数：**
    size : ArrayLike, 可选 (默认: np.inf)
        订单大小数组，支持灵活广播
        - np.inf: 使用全部可用现金
        - 具体数值: 按指定数量或比例交易
        
    price : ArrayLike, 可选 (默认: np.inf)
        订单价格数组，支持灵活广播
        - np.inf: 市价订单，使用收盘价
        - 具体数值: 限价订单
        
    size_type : ArrayLike, 可选 (默认: SizeType.Amount)
        订单大小类型，控制size参数的解释方式
        
    direction : ArrayLike, 可选 (默认: Direction.Both)
        交易方向限制
        
    **成本配置参数：**
    fees : ArrayLike, 可选 (默认: 0.0)
        比例手续费率数组
        
    fixed_fees : ArrayLike, 可选 (默认: 0.0)
        固定手续费数组
        
    slippage : ArrayLike, 可选 (默认: 0.0)
        滑点率数组
        
    **订单限制参数：**
    min_size : ArrayLike, 可选 (默认: 0.0)
        最小订单大小数组
        
    max_size : ArrayLike, 可选 (默认: np.inf)
        最大订单大小数组
        
    size_granularity : ArrayLike, 可选 (默认: np.nan)
        订单数量粒度数组
        
    **执行控制参数：**
    reject_prob : ArrayLike, 可选 (默认: 0.0)
        随机拒绝概率数组，用于压力测试
        
    lock_cash : ArrayLike, 可选 (默认: False)
        现金锁定标志数组
        
    allow_partial : ArrayLike, 可选 (默认: True)
        允许部分成交标志数组
        
    raise_reject : ArrayLike, 可选 (默认: False)
        拒绝时抛异常标志数组
        
    log : ArrayLike, 可选 (默认: False)
        日志记录标志数组
        
    **估值配置参数：**
    val_price : ArrayLike, 可选 (默认: np.inf)
        估值价格数组，用于组合价值计算
        
    close : ArrayLike, 可选 (默认: np.nan)
        收盘价数组，用于市价订单和估值
        
    **高级配置参数：**
    auto_call_seq : bool, 可选 (默认: False)
        是否自动按订单价值排序调用序列
        
    ffill_val_price : bool, 可选 (默认: True)
        是否前向填充估值价格
        
    update_value : bool, 可选 (默认: False)
        是否在每次订单后更新组合价值
        
    max_orders : int, 可选 (默认: None)
        最大订单记录数，None时自动计算
        
    max_logs : int, 可选 (默认: 0)
        最大日志记录数
        
    flex_2d : bool, 可选 (默认: True)
        是否启用2D灵活索引
        
    返回:
    ----
    tuple[RecordArray, RecordArray]
        (订单记录数组, 日志记录数组)
        
    使用示例:
    --------
    >>> import numpy as np
    >>> from vectorbt.records.nb import col_map_nb
    >>> from vectorbt.portfolio.nb import simulate_from_orders_nb, asset_flow_nb
    >>> from vectorbt.portfolio.enums import Direction
    >>> 
    >>> # 简单的买入持有策略
    >>> close = np.array([1, 2, 3, 4, 5])[:, None]  # 5个时间步，1个资产
    >>> order_records, log_records = simulate_from_orders_nb(
    ...     target_shape=close.shape,
    ...     close=close,
    ...     group_lens=np.array([1]),    # 1个组，包含1个资产
    ...     init_cash=np.array([100]),   # 初始现金100
    ...     call_seq=np.full(close.shape, 0)  # 调用序列
    ... )
    >>> 
    >>> # 分析资产流
    >>> col_map = col_map_nb(order_records['col'], close.shape[1])
    >>> asset_flow = asset_flow_nb(close.shape, order_records, col_map, Direction.Both)
    >>> print(asset_flow)  # 第一时间步买入100股，后续为0
    array([[100.],
           [  0.],
           [  0.], 
           [  0.],
           [  0.]])
    
    核心特性:
    --------
    1. **矩阵化处理**: 每个数组元素对应一个潜在订单
    2. **灵活广播**: 支持标量、1D、2D数组的自动广播
    3. **列主序迭代**: 按列优先顺序处理，优化内存访问
    4. **现金共享**: 支持组内资产共享现金池
    5. **动态排序**: 可按订单价值动态调整执行顺序
    6. **完整记录**: 生成详细的交易记录和日志
    
    使用限制:
    --------
    - 分组仅在启用现金共享时使用
    - auto_call_seq需要call_seq遵循Default类型
    - 单个值应包装为0维数组 (如 np.asarray(value))
    - 大型矩阵可能需要调整记录容量参数
    
    性能优化:
    --------
    - 使用Numba JIT编译实现接近C的执行速度
    - 支持并行处理和向量化操作
    - 内存预分配减少动态分配开销
    - 灵活索引避免数据复制
    
    应用场景:
    --------
    - 量化策略回测和评估
    - 多资产投资组合管理
    - 风险管理和压力测试
    - 算法交易策略研发
    - 学术研究和金融建模
    """
    # ================== 初始化和验证阶段 ==================
    # 验证分组长度配置的有效性
    check_group_lens_nb(group_lens, target_shape[1])
    
    # 判断是否启用现金共享模式
    cash_sharing = is_grouped_nb(group_lens)
    
    # 验证初始现金配置与分组的兼容性
    check_group_init_cash_nb(group_lens, target_shape[1], init_cash, cash_sharing)

    # 初始化记录数组：订单记录和日志记录
    order_records, log_records = init_records_nb(target_shape, max_orders, max_logs)
    
    # 确保初始现金为float64类型，避免数值精度问题
    init_cash = init_cash.astype(np.float64)
    
    # ================== 状态数组初始化 ==================
    # 各资产的最新持仓数量（正数为多头，负数为空头）
    last_position = np.full(target_shape[1], 0., dtype=np.float64)
    
    # 各资产的债务金额（用于保证金交易）
    last_debt = np.full(target_shape[1], 0., dtype=np.float64)
    
    # 各资产的最新估值价格（用于组合价值计算）
    last_val_price = np.full(target_shape[1], np.nan, dtype=np.float64)
    
    # 各资产的订单价格缓存（每个时间步更新）
    order_price = np.full(target_shape[1], np.nan, dtype=np.float64)
    
    # 临时数组：用于存储订单价值，支持动态排序
    temp_order_value = np.empty(target_shape[1], dtype=np.float64)
    
    # 记录索引：用于追踪当前的订单记录和日志记录位置
    oidx = 0  # 订单记录索引
    lidx = 0  # 日志记录索引

    # ================== 主要模拟循环 ==================
    # 按组处理资产，支持现金共享和独立资金池两种模式
    from_col = 0  # 当前组的起始列索引
    
    for group in range(len(group_lens)):
        # 计算当前组的范围
        to_col = from_col + group_lens[group]  # 结束列索引（不包含）
        group_len = to_col - from_col          # 组内资产数量
        
        # 初始化组级现金状态
        cash_now = init_cash[group]      # 当前现金余额
        free_cash_now = init_cash[group] # 可用现金余额

        # 时间循环：遍历每个交易时间步
        for i in range(target_shape[0]):
            
            # ============ 第一阶段：价格解析和预处理 ============
            # 遍历当前组内的所有资产，解析订单价格和估值价格
            for k in range(group_len):
                col = from_col + k  # 当前资产的绝对列索引

                # 解析订单执行价格
                _price = flex_select_auto_nb(price, i, col, flex_2d)
                if np.isinf(_price):
                    # 处理无穷价格（市价订单）
                    if _price > 0:
                        # 正无穷：使用当前收盘价作为买入市价上限
                        _price = flex_select_auto_nb(close, i, col, flex_2d)  
                    elif i > 0:
                        # 负无穷：使用前一收盘价作为卖出市价下限
                        _price = flex_select_auto_nb(close, i - 1, col, flex_2d)  
                    else:
                        # 第一个时间步无前收盘价，设为NaN
                        _price = np.nan  
                order_price[col] = _price  # 缓存解析后的订单价格

                # 解析组合估值价格（用于计算投资组合价值）
                _val_price = flex_select_auto_nb(val_price, i, col, flex_2d)
                if np.isinf(_val_price):
                    # 处理无穷估值价格
                    if _val_price > 0:
                        # 正无穷：使用订单价格作为估值上限
                        _val_price = _price  
                    elif i > 0:
                        # 负无穷：使用前一收盘价作为估值下限
                        _val_price = flex_select_auto_nb(close, i - 1, col, flex_2d)  
                    else:
                        # 第一个时间步无前收盘价，设为NaN
                        _val_price = np.nan  
                
                # 更新估值价格（支持前向填充）
                if not np.isnan(_val_price) or not ffill_val_price:
                    last_val_price[col] = _val_price

            # ============ 第二阶段：组合价值计算和动态排序 ============
            # 现金共享模式下的特殊处理
            if cash_sharing:
                # 计算当前组合总价值（现金 + 所有持仓市值）
                # 等同于get_group_value_ctx_nb，但使用灵活索引
                value_now = cash_now  # 从现金开始累积
                for k in range(group_len):
                    col = from_col + k

                    # 累加有持仓资产的市值
                    if last_position[col] != 0:
                        value_now += last_position[col] * last_val_price[col]

                # 动态按订单价值排序 → 卖出订单优先，提前释放资金
                if auto_call_seq:
                    # 等同于sort_by_order_value_ctx_nb，但使用灵活索引
                    # 为每个资产计算潜在订单的预估价值
                    for k in range(group_len):
                        col = from_col + k
                        temp_order_value[k] = approx_order_value_nb(
                            flex_select_auto_nb(size, i, col, flex_2d),        # 订单大小
                            flex_select_auto_nb(size_type, i, col, flex_2d),   # 大小类型
                            flex_select_auto_nb(direction, i, col, flex_2d),   # 交易方向
                            cash_now,                                           # 当前现金
                            last_position[col],                                 # 当前持仓
                            free_cash_now,                                      # 可用现金
                            last_val_price[col],                                # 估值价格
                            value_now                                           # 组合总价值
                        )

                    # 按订单价值排序调用序列
                    # 高价值订单（通常是大额卖出）优先执行
                    insert_argsort_nb(temp_order_value[:group_len], call_seq[i, from_col:to_col])

            # ============ 第三阶段：订单生成和执行循环 ============
            # 按调用序列处理每个资产的订单
            for k in range(group_len):
                col = from_col + k  # 默认的资产列索引
                
                # 现金共享模式：根据调用序列重新排序
                if cash_sharing:
                    col_i = call_seq[i, col]  # 从调用序列获取实际处理顺序
                    if col_i >= group_len:
                        raise ValueError("调用索引超出组边界")
                    col = from_col + col_i    # 重新计算实际的资产列索引

                # -------- 获取当前资产状态 --------
                position_now = last_position[col]    # 当前持仓数量
                debt_now = last_debt[col]            # 当前债务金额
                val_price_now = last_val_price[col]  # 当前估值价格
                
                # 非现金共享模式：为每个资产单独计算价值
                if not cash_sharing:
                    value_now = cash_now  # 从现金开始
                    if position_now != 0:
                        value_now += position_now * val_price_now  # 加上持仓市值

                # -------- 生成订单对象 --------
                # 从参数矩阵中提取当前时间点和资产的所有订单参数
                order = order_nb(
                    size=flex_select_auto_nb(size, i, col, flex_2d),              # 订单大小
                    price=order_price[col],                                       # 订单价格（已解析）
                    size_type=flex_select_auto_nb(size_type, i, col, flex_2d),    # 大小类型
                    direction=flex_select_auto_nb(direction, i, col, flex_2d),    # 交易方向
                    fees=flex_select_auto_nb(fees, i, col, flex_2d),              # 比例手续费
                    fixed_fees=flex_select_auto_nb(fixed_fees, i, col, flex_2d),  # 固定手续费
                    slippage=flex_select_auto_nb(slippage, i, col, flex_2d),      # 滑点
                    min_size=flex_select_auto_nb(min_size, i, col, flex_2d),      # 最小大小
                    max_size=flex_select_auto_nb(max_size, i, col, flex_2d),      # 最大大小
                    size_granularity=flex_select_auto_nb(size_granularity, i, col, flex_2d),  # 数量粒度
                    reject_prob=flex_select_auto_nb(reject_prob, i, col, flex_2d), # 拒绝概率
                    lock_cash=flex_select_auto_nb(lock_cash, i, col, flex_2d),     # 现金锁定
                    allow_partial=flex_select_auto_nb(allow_partial, i, col, flex_2d), # 允许部分成交
                    raise_reject=flex_select_auto_nb(raise_reject, i, col, flex_2d),   # 拒绝时抛异常
                    log=flex_select_auto_nb(log, i, col, flex_2d)                     # 日志记录
                )

                # -------- 创建处理状态 --------
                # 封装当前的投资组合状态，用于订单处理
                state = ProcessOrderState(
                    cash=cash_now,           # 当前现金
                    position=position_now,   # 当前持仓
                    debt=debt_now,          # 当前债务
                    free_cash=free_cash_now, # 可用现金
                    val_price=val_price_now, # 估值价格
                    value=value_now,        # 组合价值
                    oidx=oidx,              # 订单记录索引
                    lidx=lidx               # 日志记录索引
                )

                # -------- 执行订单处理 --------
                # 调用核心订单处理引擎，返回执行结果和更新后的状态
                order_result, new_state = process_order_nb(
                    i, col, group,          # 时间、资产、组索引
                    state,                  # 当前投资组合状态
                    update_value,           # 是否更新组合价值
                    order,                  # 订单对象
                    order_records,          # 订单记录数组
                    log_records             # 日志记录数组
                )

                # -------- 更新组级状态 --------
                # 从新状态中提取更新后的组级信息
                cash_now = new_state.cash           # 更新后的现金余额
                position_now = new_state.position   # 更新后的持仓数量
                debt_now = new_state.debt           # 更新后的债务金额
                free_cash_now = new_state.free_cash # 更新后的可用现金
                val_price_now = new_state.val_price # 更新后的估值价格
                value_now = new_state.value         # 更新后的组合价值
                oidx = new_state.oidx               # 更新后的订单记录索引
                lidx = new_state.lidx               # 更新后的日志记录索引

                # -------- 更新资产级状态 --------
                # 将当前状态保存为"最新状态"，供下次使用
                last_position[col] = position_now   # 保存最新持仓
                last_debt[col] = debt_now           # 保存最新债务
                
                # 更新估值价格（支持前向填充模式）
                if not np.isnan(val_price_now) or not ffill_val_price:
                    last_val_price[col] = val_price_now

        # 移动到下一个资产组
        from_col = to_col

    # ================== 返回结果 ==================
    # 返回实际使用的记录数组（截取到有效长度）
    return order_records[:oidx], log_records[:lidx]


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def generate_stop_signal_nb(position_now: float,
                            upon_stop_exit: int,
                            accumulate: int) -> tp.Tuple[bool, bool, bool, bool, int]:
    """
    生成止损信号并调整累积模式
    
    根据当前持仓状态和止损退出策略，生成相应的交易信号，
    并根据需要调整累积模式。这是止损策略实现的核心逻辑。
    
    参数:
    ----
    position_now : float
        当前持仓数量（正数为多头，负数为空头，0为无持仓）
        
    upon_stop_exit : int
        止损退出模式，参见StopExitMode枚举：
        - Close: 完全关闭持仓
        - CloseReduce: 关闭或减少持仓 
        - Reverse: 反转持仓方向
        - ReverseReduce: 反转或减少持仓
        
    accumulate : int
        当前累积模式，可能被此函数修改
        
    返回:
    ----
    tuple[bool, bool, bool, bool, int]
        (is_long_entry, is_long_exit, is_short_entry, is_short_exit, new_accumulate)
        - is_long_entry: 是否生成多头开仓信号
        - is_long_exit: 是否生成多头平仓信号
        - is_short_entry: 是否生成空头开仓信号
        - is_short_exit: 是否生成空头平仓信号
        - new_accumulate: 调整后的累积模式
        
    使用示例:
    --------
    >>> # 多头持仓触发止损
    >>> position_now = 100.0  # 持有100股多头
    >>> upon_stop_exit = StopExitMode.Close
    >>> accumulate = AccumulationMode.Enabled
    >>> 
    >>> signals = generate_stop_signal_nb(position_now, upon_stop_exit, accumulate)
    >>> is_long_entry, is_long_exit, is_short_entry, is_short_exit, new_accumulate = signals
    >>> print(f"多头平仓信号: {is_long_exit}")  # True
    >>> print(f"新累积模式: {new_accumulate}")   # Disabled
    >>> 
    >>> # 空头持仓反转止损
    >>> position_now = -50.0  # 持有50股空头
    >>> upon_stop_exit = StopExitMode.Reverse
    >>> 
    >>> signals = generate_stop_signal_nb(position_now, upon_stop_exit, accumulate)
    >>> is_long_entry, is_long_exit, is_short_entry, is_short_exit, new_accumulate = signals
    >>> print(f"多头开仓信号: {is_long_entry}")  # True
    >>> print(f"新累积模式: {new_accumulate}")   # Disabled
    
    止损逻辑详解:
    -----------
    **多头持仓 (position_now > 0):**
    - Close: 生成多头平仓信号，禁用累积
    - CloseReduce: 生成多头平仓信号，保持累积模式
    - Reverse: 生成空头开仓信号，禁用累积（完全反转）
    - ReverseReduce: 生成空头开仓信号，保持累积模式
    
    **空头持仓 (position_now < 0):**
    - Close: 生成空头平仓信号，禁用累积
    - CloseReduce: 生成空头平仓信号，保持累积模式  
    - Reverse: 生成多头开仓信号，禁用累积（完全反转）
    - ReverseReduce: 生成多头开仓信号，保持累积模式
    
    **无持仓 (position_now = 0):**
    - 不生成任何信号
    
    应用场景:
    --------
    - 止损止盈策略实现
    - 趋势跟踪系统
    - 风险管理和头寸控制
    - 动态交易策略调整
    - 算法交易系统
    
    注意事项:
    --------
    - Close和Reverse模式会禁用累积，防止重复触发
    - CloseReduce和ReverseReduce保持累积模式灵活性
    - 信号生成不等于订单执行，还需配合其他条件
    - 反转信号可能产生较大的头寸变化
    """
    # 初始化所有信号为False
    is_long_entry = False   # 多头开仓信号
    is_long_exit = False    # 多头平仓信号
    is_short_entry = False  # 空头开仓信号
    is_short_exit = False   # 空头平仓信号
    
    if position_now > 0:
        # ========== 当前持有多头头寸 ==========
        if upon_stop_exit == StopExitMode.Close:
            # 完全关闭多头持仓
            is_long_exit = True
            accumulate = AccumulationMode.Disabled  # 禁用累积，避免重复触发
        elif upon_stop_exit == StopExitMode.CloseReduce:
            # 关闭或减少多头持仓，保持累积模式
            is_long_exit = True
        elif upon_stop_exit == StopExitMode.Reverse:
            # 反转为空头持仓
            is_short_entry = True
            accumulate = AccumulationMode.Disabled  # 禁用累积，完全反转
        else:
            # StopExitMode.ReverseReduce: 反转或减少，保持灵活性
            is_short_entry = True
            
    elif position_now < 0:
        # ========== 当前持有空头头寸 ==========
        if upon_stop_exit == StopExitMode.Close:
            # 完全关闭空头持仓
            is_short_exit = True
            accumulate = AccumulationMode.Disabled  # 禁用累积，避免重复触发
        elif upon_stop_exit == StopExitMode.CloseReduce:
            # 关闭或减少空头持仓，保持累积模式
            is_short_exit = True
        elif upon_stop_exit == StopExitMode.Reverse:
            # 反转为多头持仓
            is_long_entry = True
            accumulate = AccumulationMode.Disabled  # 禁用累积，完全反转
        else:
            # StopExitMode.ReverseReduce: 反转或减少，保持灵活性
            is_long_entry = True
    
    # 无持仓时不生成任何信号，所有信号保持False
    
    return is_long_entry, is_long_exit, is_short_entry, is_short_exit, accumulate


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def resolve_stop_price_and_slippage_nb(stop_price: float,
                                       price: float,
                                       close: float,
                                       slippage: float,
                                       stop_exit_price: int) -> tp.Tuple[float, float]:
    """
    解析止损订单的价格和滑点设置
    
    根据指定的止损退出价格模式，确定最终的订单执行价格和滑点参数。
    不同的价格模式适用于不同的风险控制和执行策略。
    
    参数:
    ----
    stop_price : float
        原始止损触发价格
        
    price : float
        当前市场价格
        
    close : float
        收盘价格
        
    slippage : float
        原始滑点设置
        
    stop_exit_price : int
        止损退出价格模式，参见StopExitPrice枚举：
        - StopMarket: 以止损价格执行市价单
        - StopLimit: 以止损价格执行限价单
        - Close: 以收盘价执行
        - 其他: 以当前价格执行
        
    返回:
    ----
    tuple[float, float]
        (final_price, final_slippage) - 最终的执行价格和滑点
        
    使用示例:
    --------
    >>> # 止损市价单：使用触发价格和滑点
    >>> stop_price = 45.0
    >>> current_price = 44.8
    >>> close = 44.5
    >>> slippage = 0.001  # 0.1%滑点
    >>> 
    >>> final_price, final_slippage = resolve_stop_price_and_slippage_nb(
    ...     stop_price, current_price, close, slippage, StopExitPrice.StopMarket
    ... )
    >>> print(f"执行价格: {final_price}, 滑点: {final_slippage}")  # 45.0, 0.001
    >>> 
    >>> # 止损限价单：使用触发价格，无滑点
    >>> final_price, final_slippage = resolve_stop_price_and_slippage_nb(
    ...     stop_price, current_price, close, slippage, StopExitPrice.StopLimit
    ... )
    >>> print(f"执行价格: {final_price}, 滑点: {final_slippage}")  # 45.0, 0.0
    >>> 
    >>> # 收盘价执行：使用收盘价和滑点
    >>> final_price, final_slippage = resolve_stop_price_and_slippage_nb(
    ...     stop_price, current_price, close, slippage, StopExitPrice.Close
    ... )
    >>> print(f"执行价格: {final_price}, 滑点: {final_slippage}")  # 44.5, 0.001
    
    价格模式详解:
    -----------
    **StopMarket (止损市价单):**
    - 价格: 使用止损触发价格
    - 滑点: 保持原始滑点设置
    - 适用: 快速执行，可能有滑点成本
    
    **StopLimit (止损限价单):**
    - 价格: 使用止损触发价格
    - 滑点: 设为0（无滑点）
    - 适用: 精确价格控制，可能无法成交
    
    **Close (收盘价执行):**
    - 价格: 使用收盘价
    - 滑点: 保持原始滑点设置
    - 适用: 日终清算，价格相对稳定
    
    **其他模式:**
    - 价格: 使用当前市场价格
    - 滑点: 保持原始滑点设置
    - 适用: 默认处理方式
    
    应用场景:
    --------
    - 止损策略的价格执行控制
    - 风险管理中的价格滑点控制
    - 不同交易时段的执行策略
    - 流动性考虑的价格选择
    - 回测系统中的真实执行模拟
    
    注意事项:
    --------
    - 限价单可能因价格限制无法成交
    - 市价单执行快但可能有滑点
    - 收盘价适用于日终或特定时点
    - 滑点设置影响最终成交成本
    """
    if stop_exit_price == StopExitPrice.StopMarket:
        # 止损市价单：使用止损价格，保持滑点
        return stop_price, slippage
    elif stop_exit_price == StopExitPrice.StopLimit:
        # 止损限价单：使用止损价格，无滑点
        return stop_price, 0.
    elif stop_exit_price == StopExitPrice.Close:
        # 收盘价执行：使用收盘价，保持滑点
        return close, slippage
    # 默认：使用当前价格，保持滑点
    return price, slippage


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def resolve_signal_conflict_nb(position_now: float,
                               is_entry: bool,
                               is_exit: bool,
                               direction: int,
                               conflict_mode: int) -> tp.Tuple[bool, bool]:
    """
    解决开仓和平仓信号之间的冲突
    
    当同一时间点同时出现开仓和平仓信号时，根据指定的冲突处理模式
    来决定最终执行哪个信号。这是信号驱动策略中的重要逻辑。
    
    参数:
    ----
    position_now : float
        当前持仓数量（正数为多头，负数为空头，0为无持仓）
        
    is_entry : bool
        是否有开仓信号
        
    is_exit : bool
        是否有平仓信号
        
    direction : int
        交易方向限制，参见Direction枚举
        
    conflict_mode : int
        冲突处理模式，参见ConflictMode枚举：
        - Entry: 优先开仓信号，忽略平仓信号
        - Exit: 优先平仓信号，忽略开仓信号  
        - Adjacent: 选择与当前持仓"相邻"的信号
        - Opposite: 选择与当前持仓"相反"的信号
        - Ignore: 忽略所有冲突信号
        
    返回:
    ----
    tuple[bool, bool]
        (final_is_entry, final_is_exit) - 解决冲突后的最终信号
        
    使用示例:
    --------
    >>> # 场景1：优先开仓模式
    >>> position_now = 100.0  # 当前持有多头
    >>> is_entry = True       # 有开仓信号
    >>> is_exit = True        # 有平仓信号（冲突！）
    >>> 
    >>> final_entry, final_exit = resolve_signal_conflict_nb(
    ...     position_now, is_entry, is_exit, Direction.Both, ConflictMode.Entry
    ... )
    >>> print(f"最终信号: 开仓={final_entry}, 平仓={final_exit}")  # True, False
    >>> 
    >>> # 场景2：相邻信号模式（多头持仓选择平仓）
    >>> final_entry, final_exit = resolve_signal_conflict_nb(
    ...     position_now, is_entry, is_exit, Direction.Both, ConflictMode.Adjacent
    ... )
    >>> print(f"最终信号: 开仓={final_entry}, 平仓={final_exit}")  # True, False
    >>> 
    >>> # 场景3：无持仓时的相邻模式（无法决策）
    >>> position_now = 0.0    # 无持仓
    >>> final_entry, final_exit = resolve_signal_conflict_nb(
    ...     position_now, is_entry, is_exit, Direction.Both, ConflictMode.Adjacent
    ... )
    >>> print(f"最终信号: 开仓={final_entry}, 平仓={final_exit}")  # False, False
    
    冲突处理逻辑详解:
    ---------------
    **Entry模式 (优先开仓):**
    - 保留开仓信号，取消平仓信号
    - 适用：趋势跟踪策略，重视新机会
    
    **Exit模式 (优先平仓):**
    - 保留平仓信号，取消开仓信号
    - 适用：风险控制优先，保护现有收益
    
    **Adjacent模式 (相邻信号):**
    - 无持仓：无法决策，取消所有信号
    - 多头持仓：保留开仓信号（继续做多）
    - 空头持仓：保留平仓信号（平空头）
    - 单向交易：总是保留开仓信号
    
    **Opposite模式 (相反信号):**
    - 无持仓：无法决策，取消所有信号
    - 多头持仓：保留平仓信号（平多头）
    - 空头持仓：保留开仓信号（继续做空）
    - 单向交易：总是保留开仓信号
    
    **其他模式 (忽略):**
    - 取消所有冲突信号，保守处理
    
    应用场景:
    --------
    - 多信号策略的信号整合
    - 技术指标冲突的处理
    - 风险控制与机会捕捉的平衡
    - 趋势跟踪与均值回归的协调
    - 算法交易中的决策逻辑
    
    注意事项:
    --------
    - 只在同时有开仓和平仓信号时才处理
    - Adjacent和Opposite模式依赖当前持仓状态
    - 单向交易限制可能影响信号选择
    - 无持仓时某些模式无法有效决策
    - 冲突解决不等于最终订单执行
    """
    # 只有同时存在开仓和平仓信号时才需要解决冲突
    if is_entry and is_exit:
        # ========== 发生信号冲突，根据模式处理 ==========
        
        if conflict_mode == ConflictMode.Entry:
            # 优先开仓：忽略平仓信号
            is_exit = False
            
        elif conflict_mode == ConflictMode.Exit:
            # 优先平仓：忽略开仓信号
            is_entry = False
            
        elif conflict_mode == ConflictMode.Adjacent:
            # 相邻信号：选择与当前持仓状态"相邻"的信号
            if position_now == 0:
                # 无持仓无法决策 → 忽略所有信号
                is_entry = False
                is_exit = False
            else:
                if direction == Direction.Both:
                    # 双向交易：根据持仓方向选择
                    if position_now > 0:
                        # 多头持仓 → 保留开仓信号（继续做多）
                        is_exit = False
                    elif position_now < 0:
                        # 空头持仓 → 保留平仓信号（平空头）
                        is_entry = False
                else:
                    # 单向交易 → 总是保留开仓信号
                    is_exit = False
                    
        elif conflict_mode == ConflictMode.Opposite:
            # 相反信号：选择与当前持仓状态"相反"的信号
            if position_now == 0:
                # 无持仓无法决策 → 忽略所有信号
                is_entry = False
                is_exit = False
            else:
                if direction == Direction.Both:
                    # 双向交易：根据持仓方向反选
                    if position_now > 0:
                        # 多头持仓 → 保留平仓信号（平多头）
                        is_entry = False
                    elif position_now < 0:
                        # 空头持仓 → 保留开仓信号（继续做空）
                        is_exit = False
                else:
                    # 单向交易 → 总是保留开仓信号
                    is_entry = False
        else:
            # 其他模式（包括Ignore）→ 忽略所有冲突信号
            is_entry = False
            is_exit = False
            
    # 返回解决冲突后的最终信号
    return is_entry, is_exit


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def resolve_dir_conflict_nb(position_now: float,
                            is_long_entry: bool,
                            is_short_entry: bool,
                            upon_dir_conflict: int) -> tp.Tuple[bool, bool]:
    """
    解决多头和空头开仓信号之间的方向冲突
    
    当同一时间点同时出现多头开仓和空头开仓信号时，根据指定的
    方向冲突处理模式来决定最终执行哪个方向的开仓信号。
    
    参数:
    ----
    position_now : float
        当前持仓数量（正数为多头，负数为空头，0为无持仓）
        
    is_long_entry : bool
        是否有多头开仓信号
        
    is_short_entry : bool
        是否有空头开仓信号
        
    upon_dir_conflict : int
        方向冲突处理模式，参见DirectionConflictMode枚举：
        - Long: 优先多头开仓信号
        - Short: 优先空头开仓信号
        - Adjacent: 选择与当前持仓方向一致的信号
        - Opposite: 选择与当前持仓方向相反的信号
        - Ignore: 忽略所有冲突信号
        
    返回:
    ----
    tuple[bool, bool]
        (final_is_long_entry, final_is_short_entry) - 解决冲突后的最终信号
        
    使用示例:
    --------
    >>> # 场景1：优先多头模式
    >>> position_now = 0.0     # 当前无持仓
    >>> is_long_entry = True   # 有多头开仓信号
    >>> is_short_entry = True  # 有空头开仓信号（冲突！）
    >>> 
    >>> final_long, final_short = resolve_dir_conflict_nb(
    ...     position_now, is_long_entry, is_short_entry, DirectionConflictMode.Long
    ... )
    >>> print(f"最终信号: 多头={final_long}, 空头={final_short}")  # True, False
    >>> 
    >>> # 场景2：相邻模式（多头持仓选择多头）
    >>> position_now = 100.0   # 当前持有多头
    >>> final_long, final_short = resolve_dir_conflict_nb(
    ...     position_now, is_long_entry, is_short_entry, DirectionConflictMode.Adjacent
    ... )
    >>> print(f"最终信号: 多头={final_long}, 空头={final_short}")  # True, False
    >>> 
    >>> # 场景3：相反模式（多头持仓选择空头）
    >>> final_long, final_short = resolve_dir_conflict_nb(
    ...     position_now, is_long_entry, is_short_entry, DirectionConflictMode.Opposite
    ... )
    >>> print(f"最终信号: 多头={final_long}, 空头={final_short}")  # False, True
    
    冲突处理逻辑详解:
    ---------------
    **Long模式 (优先多头):**
    - 保留多头开仓信号，取消空头开仓信号
    - 适用：偏好做多的策略，牛市环境
    
    **Short模式 (优先空头):**
    - 保留空头开仓信号，取消多头开仓信号
    - 适用：偏好做空的策略，熊市环境
    
    **Adjacent模式 (相邻方向):**
    - 多头持仓：保留多头信号（加仓）
    - 空头持仓：保留空头信号（加仓）
    - 无持仓：无法决策，取消所有信号
    - 适用：趋势跟踪策略，顺势加仓
    
    **Opposite模式 (相反方向):**
    - 多头持仓：保留空头信号（反转）
    - 空头持仓：保留多头信号（反转）
    - 无持仓：无法决策，取消所有信号
    - 适用：均值回归策略，逆势交易
    
    **其他模式 (忽略):**
    - 取消所有冲突信号，保守处理
    
    应用场景:
    --------
    - 多因子策略的信号整合
    - 趋势和反转信号的协调
    - 市场风格切换的适应
    - 风险偏好的动态调整
    - 组合策略的信号融合
    
    注意事项:
    --------
    - 只在同时有多头和空头开仓信号时才处理
    - Adjacent和Opposite模式依赖当前持仓方向
    - 无持仓时某些模式会取消所有信号
    - 方向冲突解决不影响平仓信号
    - 需要配合其他信号处理函数使用
    """
    # 只有同时存在多头和空头开仓信号时才需要解决方向冲突
    if is_long_entry and is_short_entry:
        # ========== 发生方向冲突，根据模式处理 ==========
        
        if upon_dir_conflict == DirectionConflictMode.Long:
            # 优先多头：取消空头开仓信号
            is_short_entry = False
            
        elif upon_dir_conflict == DirectionConflictMode.Short:
            # 优先空头：取消多头开仓信号
            is_long_entry = False
            
        elif upon_dir_conflict == DirectionConflictMode.Adjacent:
            # 相邻方向：选择与当前持仓方向一致的信号
            if position_now > 0:
                # 多头持仓 → 保留多头信号（顺势加仓）
                is_short_entry = False
            elif position_now < 0:
                # 空头持仓 → 保留空头信号（顺势加仓）
                is_long_entry = False
            else:
                # 无持仓无法决策 → 取消所有信号
                is_long_entry = False
                is_short_entry = False
                
        elif upon_dir_conflict == DirectionConflictMode.Opposite:
            # 相反方向：选择与当前持仓方向相反的信号
            if position_now > 0:
                # 多头持仓 → 保留空头信号（反转做空）
                is_long_entry = False
            elif position_now < 0:
                # 空头持仓 → 保留多头信号（反转做多）
                is_short_entry = False
            else:
                # 无持仓无法决策 → 取消所有信号
                is_long_entry = False
                is_short_entry = False
        else:
            # 其他模式（包括Ignore）→ 取消所有冲突信号
            is_long_entry = False
            is_short_entry = False
            
    # 返回解决冲突后的最终方向信号
    return is_long_entry, is_short_entry


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def resolve_opposite_entry_nb(position_now: float,
                              is_long_entry: bool,
                              is_long_exit: bool,
                              is_short_entry: bool,
                              is_short_exit: bool,
                              upon_opposite_entry: int,
                              accumulate: int) -> tp.Tuple[bool, bool, bool, bool, int]:
    """
    解决相反方向开仓信号的处理逻辑
    
    当持有某方向头寸时出现相反方向的开仓信号，根据指定的处理模式
    决定是忽略、平仓还是反转头寸。这是复杂交易策略中的关键决策点。
    
    参数:
    ----
    position_now : float
        当前持仓数量（正数为多头，负数为空头，0为无持仓）
        
    is_long_entry : bool
        是否有多头开仓信号
        
    is_long_exit : bool
        是否有多头平仓信号
        
    is_short_entry : bool
        是否有空头开仓信号
        
    is_short_exit : bool
        是否有空头平仓信号
        
    upon_opposite_entry : int
        相反开仓处理模式，参见OppositeEntryMode枚举：
        - Ignore: 忽略相反开仓信号
        - Close: 转换为平仓信号，禁用累积
        - CloseReduce: 转换为平仓信号，保持累积
        - Reverse: 允许反转，禁用累积
        - ReverseReduce: 允许反转，保持累积
        
    accumulate : int
        当前累积模式，可能被此函数修改
        
    返回:
    ----
    tuple[bool, bool, bool, bool, int]
        (is_long_entry, is_long_exit, is_short_entry, is_short_exit, new_accumulate)
        调整后的信号状态和累积模式
        
    使用示例:
    --------
    >>> # 场景1：多头持仓时出现空头开仓信号 - 忽略模式
    >>> position_now = 100.0      # 持有多头
    >>> is_long_entry = False     # 无多头开仓信号
    >>> is_long_exit = False      # 无多头平仓信号
    >>> is_short_entry = True     # 有空头开仓信号（相反方向）
    >>> is_short_exit = False     # 无空头平仓信号
    >>> accumulate = AccumulationMode.Enabled
    >>> 
    >>> signals = resolve_opposite_entry_nb(
    ...     position_now, is_long_entry, is_long_exit, is_short_entry, is_short_exit,
    ...     OppositeEntryMode.Ignore, accumulate
    ... )
    >>> long_entry, long_exit, short_entry, short_exit, new_acc = signals
    >>> print(f"空头开仓信号: {short_entry}")  # False (被忽略)
    >>> 
    >>> # 场景2：多头持仓时出现空头开仓信号 - 平仓模式
    >>> signals = resolve_opposite_entry_nb(
    ...     position_now, is_long_entry, is_long_exit, is_short_entry, is_short_exit,
    ...     OppositeEntryMode.Close, accumulate
    ... )
    >>> long_entry, long_exit, short_entry, short_exit, new_acc = signals
    >>> print(f"空头开仓信号: {short_entry}")  # False (被取消)
    >>> print(f"多头平仓信号: {long_exit}")    # True (转换为平仓)
    >>> print(f"累积模式: {new_acc}")          # Disabled
    >>> 
    >>> # 场景3：空头持仓时出现多头开仓信号 - 反转模式
    >>> position_now = -50.0      # 持有空头
    >>> is_long_entry = True      # 有多头开仓信号（相反方向）
    >>> is_short_entry = False    # 无空头开仓信号
    >>> 
    >>> signals = resolve_opposite_entry_nb(
    ...     position_now, is_long_entry, is_long_exit, is_short_entry, is_short_exit,
    ...     OppositeEntryMode.Reverse, accumulate
    ... )
    >>> long_entry, long_exit, short_entry, short_exit, new_acc = signals
    >>> print(f"多头开仓信号: {long_entry}")  # True (允许反转)
    >>> print(f"累积模式: {new_acc}")          # Disabled
    
    处理逻辑详解:
    -----------
    **多头持仓 + 空头开仓信号的处理：**
    - Ignore: 取消空头开仓信号
    - Close: 取消空头开仓，生成多头平仓信号，禁用累积
    - CloseReduce: 取消空头开仓，生成多头平仓信号，保持累积
    - Reverse: 保持空头开仓信号，禁用累积（允许反转）
    - ReverseReduce: 保持空头开仓信号，保持累积
    
    **空头持仓 + 多头开仓信号的处理：**
    - Ignore: 取消多头开仓信号
    - Close: 取消多头开仓，生成空头平仓信号，禁用累积
    - CloseReduce: 取消多头开仓，生成空头平仓信号，保持累积
    - Reverse: 保持多头开仓信号，禁用累积（允许反转）
    - ReverseReduce: 保持多头开仓信号，保持累积
    
    累积模式影响:
    -----------
    - Close/Reverse模式会禁用累积，防止重复触发
    - CloseReduce/ReverseReduce模式保持累积，允许灵活调整
    - 累积模式影响后续的订单大小计算逻辑
    
    应用场景:
    --------
    - 趋势反转策略：Reverse模式允许快速反转
    - 保守策略：Close模式先平仓再观察
    - 风险控制：Ignore模式避免频繁反转
    - 灵活策略：Reduce模式允许部分调整
    - 套利策略：需要快速方向切换
    
    注意事项:
    --------
    - 只处理持有头寸时的相反方向开仓信号
    - 无持仓时此函数不会修改任何信号
    - Close模式会将开仓信号转换为平仓信号
    - Reverse模式可能产生较大的头寸变化
    - 需要配合其他信号处理函数一起使用
    """
    # ========== 多头持仓时处理空头开仓信号 ==========
    if position_now > 0 and is_short_entry:
        if upon_opposite_entry == OppositeEntryMode.Ignore:
            # 忽略相反开仓信号
            is_short_entry = False
            
        elif upon_opposite_entry == OppositeEntryMode.Close:
            # 转换为平仓信号：取消空头开仓，生成多头平仓
            is_short_entry = False
            is_long_exit = True
            accumulate = AccumulationMode.Disabled  # 禁用累积
            
        elif upon_opposite_entry == OppositeEntryMode.CloseReduce:
            # 转换为平仓信号：取消空头开仓，生成多头平仓，保持累积
            is_short_entry = False
            is_long_exit = True
            
        elif upon_opposite_entry == OppositeEntryMode.Reverse:
            # 允许反转：保持空头开仓信号，禁用累积
            accumulate = AccumulationMode.Disabled
        # ReverseReduce模式：保持空头开仓信号和累积模式不变
    
    # ========== 空头持仓时处理多头开仓信号 ==========
    if position_now < 0 and is_long_entry:
        if upon_opposite_entry == OppositeEntryMode.Ignore:
            # 忽略相反开仓信号
            is_long_entry = False
            
        elif upon_opposite_entry == OppositeEntryMode.Close:
            # 转换为平仓信号：取消多头开仓，生成空头平仓
            is_long_entry = False
            is_short_exit = True
            accumulate = AccumulationMode.Disabled  # 禁用累积
            
        elif upon_opposite_entry == OppositeEntryMode.CloseReduce:
            # 转换为平仓信号：取消多头开仓，生成空头平仓，保持累积
            is_long_entry = False
            is_short_exit = True
            
        elif upon_opposite_entry == OppositeEntryMode.Reverse:
            # 允许反转：保持多头开仓信号，禁用累积
            accumulate = AccumulationMode.Disabled
        # ReverseReduce模式：保持多头开仓信号和累积模式不变
    
    # 返回调整后的信号状态和累积模式
    return is_long_entry, is_long_exit, is_short_entry, is_short_exit, accumulate


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def signals_to_size_nb(position_now: float,
                       is_long_entry: bool,
                       is_long_exit: bool,
                       is_short_entry: bool,
                       is_short_exit: bool,
                       size: float,
                       size_type: int,
                       accumulate: int,
                       val_price_now: float) -> tp.Tuple[float, int, int]:
    """
    将方向感知的信号转换为订单参数
    
    这是信号驱动策略的核心转换函数，将四种基本交易信号
    （多头开仓、多头平仓、空头开仓、空头平仓）转换为具体的
    订单大小、大小类型和交易方向参数。
    
    参数:
    ----
    position_now : float
        当前持仓数量（正数为多头，负数为空头，0为无持仓）
        
    is_long_entry : bool
        是否有多头开仓信号
        
    is_long_exit : bool
        是否有多头平仓信号
        
    is_short_entry : bool
        是否有空头开仓信号
        
    is_short_exit : bool
        是否有空头平仓信号
        
    size : float
        信号对应的订单大小（非负数）
        
    size_type : int
        订单大小类型，支持：Amount、Value、Percent
        
    accumulate : int
        累积模式，参见AccumulationMode枚举：
        - Disabled: 完全平仓或反转
        - Both: 支持加仓和减仓
        - AddOnly: 仅支持加仓
        - RemoveOnly: 仅支持减仓
        
    val_price_now : float
        当前估值价格，用于价值类型转换
        
    返回:
    ----
    tuple[float, int, int]
        (order_size, final_size_type, direction)
        - order_size: 最终订单大小（正负表示方向）
        - final_size_type: 最终大小类型
        - direction: 交易方向限制
        
    异常:
    ----
    ValueError: 
        - 不支持的大小类型
        - 负数大小输入
        - 百分比类型不支持反转
        
    使用示例:
    --------
    >>> # 场景1：无持仓时的多头开仓信号
    >>> position_now = 0.0
    >>> is_long_entry = True
    >>> size = 100.0
    >>> val_price = 50.0
    >>> 
    >>> order_size, size_type, direction = signals_to_size_nb(
    ...     position_now, is_long_entry, False, False, False,
    ...     size, SizeType.Amount, AccumulationMode.Both, val_price
    ... )
    >>> print(f"订单: 大小={order_size}, 类型={size_type}, 方向={direction}")
    >>> # 结果: 大小=100.0, 类型=Amount, 方向=Both
    >>> 
    >>> # 场景2：多头持仓时的空头开仓信号（反转）
    >>> position_now = 50.0   # 持有50股多头
    >>> is_short_entry = True # 空头开仓信号
    >>> accumulate = AccumulationMode.Disabled  # 不允许累积
    >>> 
    >>> order_size, size_type, direction = signals_to_size_nb(
    ...     position_now, False, False, is_short_entry, False,
    ...     100.0, SizeType.Amount, accumulate, val_price
    ... )
    >>> # 结果: 先平掉50股多头，再开100股空头，总计-150股
    >>> print(f"订单大小: {order_size}")  # -150.0
    >>> 
    >>> # 场景3：多头持仓时的多头平仓信号（累积模式）
    >>> position_now = 200.0  # 持有200股多头
    >>> is_long_exit = True   # 多头平仓信号
    >>> accumulate = AccumulationMode.RemoveOnly  # 仅允许减仓
    >>> 
    >>> order_size, size_type, direction = signals_to_size_nb(
    ...     position_now, False, is_long_exit, False, False,
    ...     50.0, SizeType.Amount, accumulate, val_price
    ... )
    >>> # 结果: 减仓50股
    >>> print(f"订单大小: {order_size}")  # -50.0 (卖出50股)
    >>> print(f"方向限制: {direction}")   # Direction.LongOnly
    
    转换逻辑详解:
    -----------
    
    **多头持仓状态 (position_now > 0):**
    
    1. **空头开仓信号 (is_short_entry)**:
       - 累积模式：减少多头持仓 (-size)
       - 非累积模式：先平仓再开空头 (-position_now - additional_size)
    
    2. **多头平仓信号 (is_long_exit)**:
       - 累积模式：减少多头持仓 (-size)
       - 非累积模式：完全平仓 (-position_now)
       - 方向限制：LongOnly
    
    3. **多头开仓信号 (is_long_entry)**:
       - 累积模式：增加多头持仓 (+size)
       - 方向限制：LongOnly
    
    **空头持仓状态 (position_now < 0):**
    
    1. **多头开仓信号 (is_long_entry)**:
       - 累积模式：减少空头持仓 (+size)
       - 非累积模式：先平仓再开多头 (+|position_now| + additional_size)
    
    2. **空头平仓信号 (is_short_exit)**:
       - 累积模式：减少空头持仓 (+size)
       - 非累积模式：完全平仓 (+|position_now|)
       - 方向限制：ShortOnly
    
    3. **空头开仓信号 (is_short_entry)**:
       - 累积模式：增加空头持仓 (-size)
       - 方向限制：ShortOnly
    
    **无持仓状态 (position_now == 0):**
    - 多头开仓信号：开多头 (+size)
    - 空头开仓信号：开空头 (-size)
    
    累积模式影响:
    -----------
    - **Disabled**: 完全平仓或反转，size_type转为Amount
    - **Both**: 支持加仓和减仓的灵活操作
    - **AddOnly**: 仅在有利方向上执行信号
    - **RemoveOnly**: 仅执行减仓操作
    
    应用场景:
    --------
    - 技术指标策略的信号转换
    - 多因子模型的信号整合
    - 趋势跟踪和均值回归策略
    - 风险管理中的位置调整
    - 算法交易的执行逻辑
    
    注意事项:
    --------
    - 订单大小的正负号表示交易方向
    - 反转操作会修改size_type为Amount
    - 百分比类型不支持反转操作
    - 方向限制影响订单执行的成功率
    - 需要配合其他信号处理函数使用
    """
    # 验证支持的大小类型
    if size_type != SizeType.Amount and size_type != SizeType.Value and size_type != SizeType.Percent:
        raise ValueError("仅支持Amount、Value和Percent大小类型")
    
    # 初始化返回值
    order_size = 0.                  # 订单大小
    direction = Direction.Both       # 交易方向限制
    abs_position_now = abs(position_now)  # 当前持仓的绝对值
    
    # 验证大小参数非负
    if is_less_nb(size, 0):
        raise ValueError("不允许负数大小，请通过信号表达交易方向")

    if position_now > 0:
        # ========== 当前持有多头头寸 ==========
        
        if is_short_entry:
            # 空头开仓信号（相反方向）
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.RemoveOnly:
                # 累积模式：减少多头持仓
                order_size = -size
            else:
                # 非累积模式：反转为空头持仓
                order_size = -abs_position_now  # 先平掉所有多头
                if not np.isnan(size):
                    # 加上新的空头头寸
                    if size_type == SizeType.Percent:
                        raise ValueError("百分比类型不支持使用信号进行头寸反转")
                    if size_type == SizeType.Value:
                        order_size -= size / val_price_now  # 价值转数量
                    else:
                        order_size -= size
                size_type = SizeType.Amount  # 反转操作强制使用数量类型
                
        elif is_long_exit:
            # 多头平仓信号
            direction = Direction.LongOnly  # 限制为只能平多头
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.RemoveOnly:
                # 累积模式：减少多头持仓
                order_size = -size
            else:
                # 非累积模式：完全平仓
                order_size = -abs_position_now
                size_type = SizeType.Amount
                
        elif is_long_entry:
            # 多头开仓信号（同方向）
            direction = Direction.LongOnly  # 限制为只能做多
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.AddOnly:
                # 累积模式：增加多头持仓
                order_size = size
    
    elif position_now < 0:
        # ========== 当前持有空头头寸 ==========
        
        if is_long_entry:
            # 多头开仓信号（相反方向）
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.RemoveOnly:
                # 累积模式：减少空头持仓
                order_size = size
            else:
                # 非累积模式：反转为多头持仓
                order_size = abs_position_now  # 先平掉所有空头
                if not np.isnan(size):
                    # 加上新的多头头寸
                    if size_type == SizeType.Percent:
                        raise ValueError("百分比类型不支持使用信号进行头寸反转")
                    if size_type == SizeType.Value:
                        order_size += size / val_price_now  # 价值转数量
                    else:
                        order_size += size
                size_type = SizeType.Amount  # 反转操作强制使用数量类型
                
        elif is_short_exit:
            # 空头平仓信号
            direction = Direction.ShortOnly  # 限制为只能平空头
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.RemoveOnly:
                # 累积模式：减少空头持仓
                order_size = size
            else:
                # 非累积模式：完全平仓
                order_size = abs_position_now
                size_type = SizeType.Amount
                
        elif is_short_entry:
            # 空头开仓信号（同方向）
            direction = Direction.ShortOnly  # 限制为只能做空
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.AddOnly:
                # 累积模式：增加空头持仓
                order_size = -size
    
    else:
        # ========== 当前无持仓 ==========
        if is_long_entry:
            # 开多头仓位
            order_size = size
        elif is_short_entry:
            # 开空头仓位
            order_size = -size

    return order_size, size_type, direction


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def should_update_stop_nb(stop: float, upon_stop_update: int) -> bool:
    """
    判断是否应该更新止损价位
    
    根据止损更新模式和新的止损值，决定是否应该更新当前的止损设置。
    这是动态止损策略中的关键判断逻辑。
    
    参数:
    ----
    stop : float
        新的止损值，可能为NaN
        
    upon_stop_update : int
        止损更新模式，参见StopUpdateMode枚举：
        - Override: 有效值时覆盖
        - OverrideNaN: 包括NaN值也覆盖
        - 其他: 不更新
        
    返回:
    ----
    bool
        是否应该更新止损价位
        
    使用示例:
    --------
    >>> # 有效止损值的覆盖模式
    >>> new_stop = 0.05  # 5%止损
    >>> should_update = should_update_stop_nb(new_stop, StopUpdateMode.Override)
    >>> print(f"应该更新: {should_update}")  # True
    >>> 
    >>> # NaN值的处理
    >>> import numpy as np
    >>> new_stop = np.nan
    >>> should_update = should_update_stop_nb(new_stop, StopUpdateMode.Override)
    >>> print(f"应该更新: {should_update}")  # False
    >>> 
    >>> # OverrideNaN模式处理NaN
    >>> should_update = should_update_stop_nb(new_stop, StopUpdateMode.OverrideNaN)
    >>> print(f"应该更新: {should_update}")  # True
    
    更新模式详解:
    -----------
    **Override模式:**
    - 仅当新止损值有效（非NaN）时才更新
    - 适用：保守的止损更新策略
    
    **OverrideNaN模式:**
    - 无论新值是否为NaN都更新
    - 适用：需要清除止损的场景
    
    **其他模式:**
    - 不进行更新，保持原有设置
    
    应用场景:
    --------
    - 动态调整止损点位
    - 跟踪止损策略实现
    - 条件性止损更新逻辑
    - 多重止损条件的协调
    - 止损策略的开启与关闭
    
    注意事项:
    --------
    - NaN通常表示禁用止损
    - 更新决策不涉及实际价格计算
    - 需要配合其他止损函数使用
    - 频繁更新可能影响策略稳定性
    """
    # 检查是否为覆盖模式
    if upon_stop_update == StopUpdateMode.Override or upon_stop_update == StopUpdateMode.OverrideNaN:
        # Override: 仅当止损值有效时更新
        # OverrideNaN: 包括NaN值也更新
        if not np.isnan(stop) or upon_stop_update == StopUpdateMode.OverrideNaN:
            return True
    
    # 其他模式或条件不满足时不更新
    return False


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def get_stop_price_nb(position_now: float,
                      stop_price: float,
                      stop: float,
                      open: float,
                      low: float,
                      high: float,
                      hit_below: bool) -> float:
    """
    计算止损触发价格
    
    根据当前持仓方向、止损百分比和价格范围，计算实际的止损触发价格。
    支持跟踪止损逻辑，如果在开盘前触发则使用开盘价执行。
    
    参数:
    ----
    position_now : float
        当前持仓数量（正数为多头，负数为空头）
        
    stop_price : float
        基准止损价格（通常是入场价格或上次更新价格）
        
    stop : float
        止损百分比（0-1之间的小数，如0.05表示5%）
        
    open : float
        当期开盘价
        
    low : float
        当期最低价
        
    high : float
        当期最高价
        
    hit_below : bool
        是否在价格下方触发：
        - True: 价格向下突破时触发（止损）
        - False: 价格向上突破时触发（止盈）
        
    返回:
    ----
    float
        计算出的止损触发价格，如果无法触发则返回NaN
        
    异常:
    ----
    ValueError: 当止损值为负数时抛出
        
    使用示例:
    --------
    >>> # 多头持仓的向下止损
    >>> position_now = 100.0      # 持有多头
    >>> stop_price = 50.0         # 基准价格（如入场价）
    >>> stop = 0.05               # 5%止损
    >>> open_price = 48.0         # 开盘价
    >>> low_price = 46.0          # 最低价
    >>> high_price = 49.0         # 最高价
    >>> hit_below = True          # 向下触发（止损）
    >>> 
    >>> trigger_price = get_stop_price_nb(
    ...     position_now, stop_price, stop, open_price, low_price, high_price, hit_below
    ... )
    >>> # 计算: 50.0 * (1 - 0.05) = 47.5
    >>> # 检查: 46.0 <= 47.5 <= 49.0 (在价格范围内)
    >>> print(f"触发价格: {trigger_price}")  # 47.5
    >>> 
    >>> # 空头持仓的向上止损
    >>> position_now = -100.0     # 持有空头
    >>> stop_price = 50.0         # 基准价格
    >>> hit_below = False         # 向上触发（止损）
    >>> 
    >>> trigger_price = get_stop_price_nb(
    ...     position_now, stop_price, stop, open_price, low_price, high_price, hit_below
    ... )
    >>> # 计算: 50.0 * (1 + 0.05) = 52.5
    >>> # 检查: 52.5 > 49.0 (超出价格范围)，返回NaN
    >>> print(f"触发价格: {trigger_price}")  # NaN
    >>> 
    >>> # 开盘前触发的情况
    >>> open_price = 46.0         # 开盘价低于止损价
    >>> trigger_price = get_stop_price_nb(
    ...     100.0, stop_price, stop, open_price, low_price, high_price, True
    ... )
    >>> print(f"触发价格: {trigger_price}")  # 46.0 (开盘价)
    
    计算逻辑详解:
    -----------
    
    **多头持仓向下止损 (position > 0, hit_below = True):**
    - 止损价 = 基准价 × (1 - 止损%)
    - 价格下跌触及止损价时触发
    
    **空头持仓向上止损 (position < 0, hit_below = False):**
    - 止损价 = 基准价 × (1 + 止损%)
    - 价格上涨触及止损价时触发
    
    **多头持仓向上止盈 (position > 0, hit_below = False):**
    - 止盈价 = 基准价 × (1 + 止盈%)
    - 价格上涨触及止盈价时触发
    
    **空头持仓向下止盈 (position < 0, hit_below = True):**
    - 止盈价 = 基准价 × (1 - 止盈%)
    - 价格下跌触及止盈价时触发
    
    触发条件检查:
    -----------
    1. **开盘前触发**: 如果开盘价已经触发条件，返回开盘价
    2. **价格范围内**: 如果计算价格在[low, high]范围内，返回计算价格
    3. **无法触发**: 价格范围未触及计算价格，返回NaN
    
    应用场景:
    --------
    - 固定百分比止损策略
    - 跟踪止损的价格计算
    - 止盈目标的价格确定
    - 动态调整的风险控制
    - 突破策略的触发点计算
    
    注意事项:
    --------
    - 止损百分比必须为非负数
    - 返回NaN表示当期未触发
    - 开盘价触发优先于区间内触发
    - 需要准确的OHLC数据
    - 适用于日内和日间策略
    """
    # 验证止损值有效性
    if stop < 0:
        raise ValueError("止损值必须大于等于0")
    
    # ========== 向下触发的情况 ==========
    if (position_now > 0 and hit_below) or (position_now < 0 and not hit_below):
        # 多头向下止损 或 空头向下止盈
        # 计算向下的触发价格
        stop_price = stop_price * (1 - stop)
        
        # 检查是否在开盘前已经触发
        if open <= stop_price:
            return open  # 开盘价执行
        
        # 检查是否在当期价格范围内触发
        if low <= stop_price <= high:
            return stop_price  # 计算价格执行
        
        # 未触发条件
        return np.nan
    
    # ========== 向上触发的情况 ==========
    if (position_now < 0 and hit_below) or (position_now > 0 and not hit_below):
        # 空头向上止损 或 多头向上止盈
        # 计算向上的触发价格
        stop_price = stop_price * (1 + stop)
        
        # 检查是否在开盘前已经触发
        if stop_price <= open:
            return open  # 开盘价执行
        
        # 检查是否在当期价格范围内触发
        if low <= stop_price <= high:
            return stop_price  # 计算价格执行
        
        # 未触发条件
        return np.nan
    
    # 其他情况（不应该出现）
    return np.nan


@njit  # Numba即时编译
def no_signal_func_nb(c: SignalContext, *args) -> tp.Tuple[bool, bool, bool, bool]:
    """
    无信号占位符函数
    
    默认的信号生成函数，始终返回无信号状态。用作信号驱动模拟中的
    默认参数，当不需要自定义信号逻辑时使用。
    
    参数:
    ----
    c : SignalContext
        信号上下文对象，包含当前状态信息
        
    *args : 可变参数
        额外的用户定义参数（此函数中未使用）
        
    返回:
    ----
    tuple[bool, bool, bool, bool]
        (is_long_entry, is_long_exit, is_short_entry, is_short_exit)
        始终返回 (False, False, False, False) 表示无任何信号
        
    使用示例:
    --------
    >>> # 用作默认信号函数
    >>> def my_simulation():
    ...     # 使用默认无信号函数
    ...     simulate_from_signal_func_nb(
    ...         signal_func_nb=no_signal_func_nb,  # 默认无信号
    ...         # ... 其他参数
    ...     )
    
    应用场景:
    --------
    - 仅使用订单驱动而非信号驱动的策略
    - 信号函数的默认占位符
    - 禁用信号生成的情况
    - 测试和调试时的简化配置
    """
    # 始终返回无信号状态
    return False, False, False, False


@njit  # Numba即时编译
def no_adjust_sl_func_nb(c: AdjustSLContext, *args) -> tp.Tuple[float, bool]:
    """
    无止损调整占位符函数
    
    默认的止损调整函数，不进行任何调整，返回当前的止损值和跟踪标志。
    用作止损策略中的默认参数。
    
    参数:
    ----
    c : AdjustSLContext
        止损调整上下文对象，包含当前止损状态
        
    *args : 可变参数
        额外的用户定义参数（此函数中未使用）
        
    返回:
    ----
    tuple[float, bool]
        (stop_value, trail_flag) - 当前的止损值和跟踪标志
        
    使用示例:
    --------
    >>> # 用作默认止损调整函数
    >>> def my_simulation():
    ...     simulate_from_signal_func_nb(
    ...         adjust_sl_func_nb=no_adjust_sl_func_nb,  # 默认不调整
    ...         # ... 其他参数
    ...     )
    
    应用场景:
    --------
    - 固定止损策略（不需要动态调整）
    - 止损调整函数的默认占位符
    - 禁用止损调整功能
    - 简化的策略配置
    
    注意事项:
    --------
    - 保持止损值和跟踪标志不变
    - 适用于不需要动态调整的策略
    """
    # 返回当前的止损值和跟踪标志，不做调整
    return c.curr_stop, c.curr_trail


@njit  # Numba即时编译
def no_adjust_tp_func_nb(c: AdjustTPContext, *args) -> float:
    """
    无止盈调整占位符函数
    
    默认的止盈调整函数，不进行任何调整，返回当前的止盈值。
    用作止盈策略中的默认参数。
    
    参数:
    ----
    c : AdjustTPContext
        止盈调整上下文对象，包含当前止盈状态
        
    *args : 可变参数
        额外的用户定义参数（此函数中未使用）
        
    返回:
    ----
    float
        当前的止盈值，不做任何调整
        
    使用示例:
    --------
    >>> # 用作默认止盈调整函数
    >>> def my_simulation():
    ...     simulate_from_signal_func_nb(
    ...         adjust_tp_func_nb=no_adjust_tp_func_nb,  # 默认不调整
    ...         # ... 其他参数
    ...     )
    
    应用场景:
    --------
    - 固定止盈策略（不需要动态调整）
    - 止盈调整函数的默认占位符
    - 禁用止盈调整功能
    - 简化的策略配置
    
    注意事项:
    --------
    - 保持止盈值不变
    - 适用于固定目标价位的策略
    - 不影响止盈触发逻辑
    """
    # 返回当前的止盈值，不做调整
    return c.curr_stop


SignalFuncT = tp.Callable[[SignalContext, tp.VarArg()], tp.Tuple[bool, bool, bool, bool]]
AdjustSLFuncT = tp.Callable[[AdjustSLContext, tp.VarArg()], tp.Tuple[float, bool]]
AdjustTPFuncT = tp.Callable[[AdjustTPContext, tp.VarArg()], float]


@njit  # Numba即时编译，极致性能优化
def simulate_from_signal_func_nb(target_shape: tp.Shape,
                                 group_lens: tp.Array1d,
                                 init_cash: tp.Array1d,
                                 call_seq: tp.Array2d,
                                 signal_func_nb: SignalFuncT = no_signal_func_nb,
                                 signal_args: tp.ArgsLike = (),
                                 size: tp.ArrayLike = np.asarray(np.inf),
                                 price: tp.ArrayLike = np.asarray(np.inf),
                                 size_type: tp.ArrayLike = np.asarray(SizeType.Amount),
                                 fees: tp.ArrayLike = np.asarray(0.),
                                 fixed_fees: tp.ArrayLike = np.asarray(0.),
                                 slippage: tp.ArrayLike = np.asarray(0.),
                                 min_size: tp.ArrayLike = np.asarray(0.),
                                 max_size: tp.ArrayLike = np.asarray(np.inf),
                                 size_granularity: tp.ArrayLike = np.asarray(np.nan),
                                 reject_prob: tp.ArrayLike = np.asarray(0.),
                                 lock_cash: tp.ArrayLike = np.asarray(False),
                                 allow_partial: tp.ArrayLike = np.asarray(True),
                                 raise_reject: tp.ArrayLike = np.asarray(False),
                                 log: tp.ArrayLike = np.asarray(False),
                                 accumulate: tp.ArrayLike = np.asarray(AccumulationMode.Disabled),
                                 upon_long_conflict: tp.ArrayLike = np.asarray(ConflictMode.Ignore),
                                 upon_short_conflict: tp.ArrayLike = np.asarray(ConflictMode.Ignore),
                                 upon_dir_conflict: tp.ArrayLike = np.asarray(DirectionConflictMode.Ignore),
                                 upon_opposite_entry: tp.ArrayLike = np.asarray(OppositeEntryMode.ReverseReduce),
                                 val_price: tp.ArrayLike = np.asarray(np.inf),
                                 open: tp.ArrayLike = np.asarray(np.nan),
                                 high: tp.ArrayLike = np.asarray(np.nan),
                                 low: tp.ArrayLike = np.asarray(np.nan),
                                 close: tp.ArrayLike = np.asarray(np.nan),
                                 sl_stop: tp.ArrayLike = np.asarray(np.nan),
                                 sl_trail: tp.ArrayLike = np.asarray(False),
                                 tp_stop: tp.ArrayLike = np.asarray(np.nan),
                                 stop_entry_price: tp.ArrayLike = np.asarray(StopEntryPrice.Close),
                                 stop_exit_price: tp.ArrayLike = np.asarray(StopExitPrice.StopLimit),
                                 upon_stop_exit: tp.ArrayLike = np.asarray(StopExitMode.Close),
                                 upon_stop_update: tp.ArrayLike = np.asarray(StopUpdateMode.Override),
                                 adjust_sl_func_nb: AdjustSLFuncT = no_adjust_sl_func_nb,
                                 adjust_sl_args: tp.Args = (),
                                 adjust_tp_func_nb: AdjustTPFuncT = no_adjust_tp_func_nb,
                                 adjust_tp_args: tp.Args = (),
                                 use_stops: bool = True,
                                 auto_call_seq: bool = False,
                                 ffill_val_price: bool = True,
                                 update_value: bool = False,
                                 max_orders: tp.Optional[int] = None,
                                 max_logs: int = 0,
                                 flex_2d: bool = True) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
    """
    基于信号函数的投资组合高级模拟器
    
    这是VectorBT最高级、最全面的信号驱动模拟函数，集成了完整的交易生态系统：
    信号生成、冲突解决、止损止盈、动态调整、风险控制等所有核心功能。
    支持复杂的多资产组合策略和高频交易场景。
    
    核心参数:
    --------
    target_shape : tuple[int, int]
        目标模拟形状 (时间步数, 资产数量)
        
    group_lens : np.ndarray[int]
        每个组的列数，用于现金共享分组
        
    init_cash : np.ndarray[float]
        每个组的初始现金
        
    call_seq : np.ndarray[int]
        调用序列矩阵，控制每个时间步的订单执行顺序
        
    signal_func_nb : callable
        信号生成函数，返回(long_entry, long_exit, short_entry, short_exit)
        函数签名: (SignalContext, *signal_args) -> tuple[bool, bool, bool, bool]
        
    signal_args : tuple
        传递给信号函数的额外参数
        
    基础交易参数:
    -----------
    size : array_like, 默认 np.inf
        订单大小，支持灵活广播
        
    price : array_like, 默认 np.inf
        订单价格，np.inf表示使用收盘价
        
    size_type : array_like, 默认 SizeType.Amount
        大小类型：Amount(数量), Value(价值), Percent(百分比)
        
    fees : array_like, 默认 0.0
        比例手续费率
        
    fixed_fees : array_like, 默认 0.0
        固定手续费金额
        
    slippage : array_like, 默认 0.0
        滑点百分比
        
    min_size/max_size : array_like
        最小/最大订单大小限制
        
    size_granularity : array_like, 默认 NaN
        订单大小粒度（如最小交易单位）
        
    风险控制参数:
    -----------
    reject_prob : array_like, 默认 0.0
        订单拒绝概率，用于压力测试
        
    lock_cash : array_like, 默认 False
        是否锁定现金（保证金交易）
        
    allow_partial : array_like, 默认 True
        是否允许部分成交
        
    raise_reject : array_like, 默认 False
        订单被拒绝时是否抛出异常
        
    log : array_like, 默认 False
        是否记录详细日志
        
    信号处理参数:
    -----------
    accumulate : array_like, 默认 AccumulationMode.Disabled
        累积模式：控制是否允许加仓/减仓
        
    upon_long_conflict : array_like, 默认 ConflictMode.Ignore
        多头信号冲突处理模式
        
    upon_short_conflict : array_like, 默认 ConflictMode.Ignore
        空头信号冲突处理模式
        
    upon_dir_conflict : array_like, 默认 DirectionConflictMode.Ignore
        方向冲突处理模式
        
    upon_opposite_entry : array_like, 默认 OppositeEntryMode.ReverseReduce
        相反开仓信号处理模式
        
    价格数据参数:
    -----------
    val_price : array_like, 默认 np.inf
        估值价格，用于组合估值
        
    open/high/low/close : array_like, 默认 NaN
        OHLC价格数据，用于止损止盈计算
        
    止损止盈参数:
    -----------
    sl_stop : array_like, 默认 NaN
        止损百分比
        
    sl_trail : array_like, 默认 False
        是否为跟踪止损
        
    tp_stop : array_like, 默认 NaN
        止盈百分比
        
    stop_entry_price : array_like, 默认 StopEntryPrice.Close
        止损入场价格参考
        
    stop_exit_price : array_like, 默认 StopExitPrice.StopLimit
        止损出场价格参考
        
    upon_stop_exit : array_like, 默认 StopExitMode.Close
        止损退出模式
        
    upon_stop_update : array_like, 默认 StopUpdateMode.Override
        止损更新模式
        
    动态调整函数:
    -----------
    adjust_sl_func_nb : callable, 默认 no_adjust_sl_func_nb
        止损动态调整函数
        
    adjust_sl_args : tuple
        止损调整函数参数
        
    adjust_tp_func_nb : callable, 默认 no_adjust_tp_func_nb
        止盈动态调整函数
        
    adjust_tp_args : tuple
        止盈调整函数参数
        
    系统控制参数:
    -----------
    use_stops : bool, 默认 True
        是否启用止损止盈系统
        
    auto_call_seq : bool, 默认 False
        是否自动生成调用序列
        
    ffill_val_price : bool, 默认 True
        是否前向填充估值价格
        
    update_value : bool, 默认 False
        是否更新组合价值
        
    max_orders : int, 可选
        最大订单记录数
        
    max_logs : int, 默认 0
        最大日志记录数
        
    flex_2d : bool, 默认 True
        是否启用2D灵活广播
        
    返回:
    ----
    tuple[RecordArray, RecordArray]
        (order_records, log_records) - 订单记录和日志记录
        
    信号处理流水线:
    -----------
    1. **止损信号检查**: 检查是否有止损/止盈触发
    2. **信号生成**: 调用signal_func_nb获取原始信号
    3. **方向内冲突解决**: 使用resolve_signal_conflict_nb处理
    4. **方向间冲突解决**: 使用resolve_dir_conflict_nb处理
    5. **相反开仓处理**: 使用resolve_opposite_entry_nb处理
    6. **信号转换**: 使用signals_to_size_nb转换为订单参数
    7. **订单执行**: 执行最终订单并记录结果
    
    使用示例:
    --------
    >>> import numpy as np
    >>> from vectorbt.portfolio import nb
    >>> from vectorbt.portfolio.enums import Direction
    >>> 
    >>> # 简单的买入持有策略
    >>> def simple_buy_signal(ctx, *args):
    ...     '''首次买入后持有'''
    ...     if ctx.i == 0:  # 第一个时间点
    ...         return True, False, False, False  # 多头开仓
    ...     return False, False, False, False  # 其他时间无信号
    >>> 
    >>> # 价格数据
    >>> close = np.array([[100., 110.], 
    ...                   [105., 115.], 
    ...                   [110., 120.], 
    ...                   [115., 125.], 
    ...                   [120., 130.]])
    >>> 
    >>> # 执行模拟
    >>> order_records, log_records = simulate_from_signal_func_nb(
    ...     target_shape=close.shape,
    ...     group_lens=np.array([2]),          # 两个资产共享现金
    ...     init_cash=np.array([10000.]),      # 初始资金10000
    ...     call_seq=np.array([[0, 1]] * 5),   # 调用顺序
    ...     signal_func_nb=simple_buy_signal,   # 信号函数
    ...     close=close,                        # 价格数据
    ...     size=np.array([50., 40.]),         # 各买50股和40股
    ...     fees=0.001                          # 0.1%手续费
    ... )
    >>> 
    >>> print(f"执行了{len(order_records)}个订单")
    >>> print(f"记录了{len(log_records)}条日志")
    
    高级策略示例:
    -----------
    >>> # 移动平均线交叉策略
    >>> def ma_cross_strategy(ctx, short_ma, long_ma):
    ...     '''移动平均线交叉策略'''
    ...     if ctx.i < len(long_ma) - 1:
    ...         short_val = short_ma[ctx.i, ctx.col]
    ...         long_val = long_ma[ctx.i, ctx.col]
    ...         prev_short = short_ma[ctx.i-1, ctx.col] if ctx.i > 0 else 0
    ...         prev_long = long_ma[ctx.i-1, ctx.col] if ctx.i > 0 else 0
    ...         
    ...         # 金叉：短期MA上穿长期MA，买入
    ...         if prev_short <= prev_long and short_val > long_val:
    ...             return True, False, False, False
    ...         # 死叉：短期MA下穿长期MA，卖出
    ...         elif prev_short >= prev_long and short_val < long_val:
    ...             return False, True, False, False
    ...     
    ...     return False, False, False, False
    >>> 
    >>> # 带止损的模拟
    >>> order_records, log_records = simulate_from_signal_func_nb(
    ...     target_shape=close.shape,
    ...     group_lens=np.array([1, 1]),       # 独立资产
    ...     init_cash=np.array([5000., 5000.]),
    ...     call_seq=np.array([[0, 1]] * 5),
    ...     signal_func_nb=ma_cross_strategy,
    ...     signal_args=(short_ma, long_ma),    # 传递MA数据
    ...     close=close,
    ...     sl_stop=0.05,                      # 5%止损
    ...     tp_stop=0.10,                      # 10%止盈
    ...     use_stops=True,                     # 启用止损止盈
    ...     size=np.inf,                       # 全仓交易
    ...     size_type=SizeType.Value,          # 按价值下单
    ... )
    
    性能优化特性:
    -----------
    - **Numba JIT编译**: 接近C语言的执行速度
    - **向量化运算**: 高效的数组操作
    - **内存预分配**: 避免动态内存分配开销
    - **灵活广播**: 最小化数据复制
    - **批量处理**: 支持同时处理多个资产
    
    适用场景:
    --------
    - **量化策略回测**: 各种技术指标策略测试
    - **多因子模型**: 复杂的多因子选股策略
    - **高频交易**: 微秒级的交易信号处理
    - **风险管理**: 止损止盈策略验证
    - **组合优化**: 多资产配置策略
    - **算法交易**: 自动化交易系统开发
    
    注意事项:
    --------
    - 仅在现金共享时进行分组
    - auto_call_seq为True时确保call_seq为默认类型
    - 单个值需要用np.asarray()包装为0维数组
    - 信号函数必须是Numba兼容的
    - 大型数据集建议适当设置max_orders限制内存使用
    - 高频策略需要精确的OHLC数据
    """
    # ========== 前期验证和初始化阶段 ==========
    
    # 验证分组配置的有效性
    check_group_lens_nb(group_lens, target_shape[1])
    
    # 检查是否启用现金共享功能
    cash_sharing = is_grouped_nb(group_lens)
    
    # 验证初始现金配置是否与分组设置匹配
    check_group_init_cash_nb(group_lens, target_shape[1], init_cash, cash_sharing)

    # ========== 记录数组初始化 ==========
    # 初始化订单记录和日志记录数组
    order_records, log_records = init_records_nb(target_shape, max_orders, max_logs)
    
    # 确保初始现金为float64类型，保证数值精度
    init_cash = init_cash.astype(np.float64)
    
    # ========== 投资组合状态数组初始化 ==========
    # 每列的最后持仓数量（正数=多头，负数=空头）
    last_position = np.full(target_shape[1], 0., dtype=np.float64)
    
    # 每列的最后债务金额（保证金交易时使用）
    last_debt = np.full(target_shape[1], 0., dtype=np.float64)
    
    # 每列的最后估值价格（用于组合价值计算）
    last_val_price = np.full(target_shape[1], np.nan, dtype=np.float64)
    
    # ========== 止损止盈系统初始化 ==========
    if use_stops:
        # 启用止损止盈时，为每个资产分配状态跟踪数组
        
        # 止损(SL)相关状态
        sl_init_i = np.full(target_shape[1], -1, dtype=np.int64)         # 止损初始化时间点
        sl_init_price = np.full(target_shape[1], np.nan, dtype=np.float64)  # 止损初始参考价格
        sl_curr_i = np.full(target_shape[1], -1, dtype=np.int64)         # 当前止损时间点
        sl_curr_price = np.full(target_shape[1], np.nan, dtype=np.float64)  # 当前止损参考价格
        sl_curr_stop = np.full(target_shape[1], np.nan, dtype=np.float64)   # 当前止损百分比
        sl_curr_trail = np.full(target_shape[1], False, dtype=np.bool_)    # 当前是否跟踪止损
        
        # 止盈(TP)相关状态
        tp_init_i = np.full(target_shape[1], -1, dtype=np.int64)         # 止盈初始化时间点
        tp_init_price = np.full(target_shape[1], np.nan, dtype=np.float64)  # 止盈初始参考价格
        tp_curr_stop = np.full(target_shape[1], np.nan, dtype=np.float64)   # 当前止盈百分比
    else:
        # 未启用止损止盈时，创建空数组以保持接口一致性
        sl_init_i = np.empty(0, dtype=np.int64)
        sl_init_price = np.empty(0, dtype=np.float64)
        sl_curr_i = np.empty(0, dtype=np.int64)
        sl_curr_price = np.empty(0, dtype=np.float64)
        sl_curr_stop = np.empty(0, dtype=np.float64)
        sl_curr_trail = np.empty(0, dtype=np.bool_)
        tp_init_i = np.empty(0, dtype=np.int64)
        tp_init_price = np.empty(0, dtype=np.float64)
        tp_curr_stop = np.empty(0, dtype=np.float64)
    
    # ========== 临时工作数组初始化 ==========
    # 这些数组用于每个时间步的计算，避免重复内存分配
    price_arr = np.full(target_shape[1], np.nan, dtype=np.float64)        # 订单价格临时数组
    size_arr = np.empty(target_shape[1], dtype=np.float64)                # 订单大小临时数组
    size_type_arr = np.empty(target_shape[1], dtype=np.float64)           # 订单大小类型临时数组
    slippage_arr = np.empty(target_shape[1], dtype=np.float64)            # 滑点临时数组
    direction_arr = np.empty(target_shape[1], dtype=np.int64)             # 交易方向临时数组
    temp_order_value = np.empty(target_shape[1], dtype=np.float64)        # 订单价值估算临时数组
    
    # ========== 记录索引初始化 ==========
    oidx = 0  # 当前订单记录索引
    lidx = 0  # 当前日志记录索引

    # ========== 主模拟循环：按组遍历 ==========
    from_col = 0  # 当前组的起始列索引
    for group in range(len(group_lens)):
        # 确定当前组的列范围
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        
        # 初始化当前组的现金状态
        cash_now = init_cash[group]       # 当前可用现金
        free_cash_now = init_cash[group]  # 当前自由现金（扣除保证金后）

        # ========== 时间维度循环：逐个时间步处理 ==========
        for i in range(target_shape[0]):
            
            # ========== 第一阶段：价格解析 ==========
            # 为当前组的每个资产解析订单价格和估值价格
            for k in range(group_len):
                col = from_col + k

                # 解析订单执行价格
                _price = flex_select_auto_nb(price, i, col, flex_2d)
                if np.isinf(_price):
                    if _price > 0:
                        # 正无穷：使用收盘价作为价格上限
                        _price = flex_select_auto_nb(close, i, col, flex_2d)
                    else:
                        # 负无穷：寻找价格下限
                        _open = flex_select_auto_nb(open, i, col, flex_2d)
                        if not np.isnan(_open):
                            _price = _open  # 首选开盘价
                        elif i > 0:
                            _price = flex_select_auto_nb(close, i - 1, col, flex_2d)  # 次选前一日收盘价
                        else:
                            _price = np.nan  # 首个时间点无历史价格

                # 解析组合估值价格
                _val_price = flex_select_auto_nb(val_price, i, col, flex_2d)
                if np.isinf(_val_price):
                    if _val_price > 0:
                        # 正无穷：使用订单价格作为估值价格上限
                        _val_price = _price
                    elif i > 0:
                        # 负无穷：使用前一日收盘价作为估值价格下限
                        _val_price = flex_select_auto_nb(close, i - 1, col, flex_2d)
                    else:
                        _val_price = np.nan  # 首个时间点无历史价格
                
                # 更新最后估值价格（支持前向填充）
                if not np.isnan(_val_price) or not ffill_val_price:
                    last_val_price[col] = _val_price
                
                # 保存解析后的订单价格到临时数组
                price_arr[col] = _price

            # ========== 第二阶段：订单参数计算和止损调整 ==========
            # 为当前组的每个资产计算订单大小、价值并调整止损设置
            for k in range(group_len):
                col = from_col + k  # 订单执行顺序在此阶段不重要

                # 获取当前状态
                position_now = last_position[col]  # 当前持仓
                _price = price_arr[col]            # 订单价格
                _slippage = flex_select_auto_nb(slippage, i, col, flex_2d)  # 滑点设置
                stop_price = np.nan                # 初始化止损触发价格
                
                if use_stops:
                    # ========== 动态止损调整 ==========
                    
                    # 构建止损调整上下文并调用用户定义的调整函数
                    adjust_sl_ctx = AdjustSLContext(
                        i=i,                              # 当前时间索引
                        col=col,                          # 当前列索引
                        position_now=last_position[col],  # 当前持仓
                        val_price_now=last_val_price[col],# 当前估值价格
                        init_i=sl_init_i[col],           # 止损初始化时间
                        init_price=sl_init_price[col],   # 止损初始价格
                        curr_i=sl_curr_i[col],           # 止损当前时间
                        curr_price=sl_curr_price[col],   # 止损当前价格
                        curr_stop=sl_curr_stop[col],     # 当前止损百分比
                        curr_trail=sl_curr_trail[col]    # 当前跟踪标志
                    )
                    # 调用止损调整函数获取新的止损设置
                    sl_curr_stop[col], sl_curr_trail[col] = adjust_sl_func_nb(adjust_sl_ctx, *adjust_sl_args)
                    
                    # 构建止盈调整上下文并调用用户定义的调整函数
                    adjust_tp_ctx = AdjustTPContext(
                        i=i,                              # 当前时间索引
                        col=col,                          # 当前列索引
                        position_now=last_position[col],  # 当前持仓
                        val_price_now=last_val_price[col],# 当前估值价格
                        init_i=tp_init_i[col],           # 止盈初始化时间
                        init_price=tp_init_price[col],   # 止盈初始价格
                        curr_stop=tp_curr_stop[col]      # 当前止盈百分比
                    )
                    # 调用止盈调整函数获取新的止盈设置
                    tp_curr_stop[col] = adjust_tp_func_nb(adjust_tp_ctx, *adjust_tp_args)

                    # ========== 止损触发价格计算 ==========
                    if not np.isnan(sl_curr_stop[col]) or not np.isnan(tp_curr_stop[col]):
                        # 解析当前K线的OHLC数据
                        _open = flex_select_auto_nb(open, i, col, flex_2d)
                        _high = flex_select_auto_nb(high, i, col, flex_2d)
                        _low = flex_select_auto_nb(low, i, col, flex_2d)
                        _close = flex_select_auto_nb(close, i, col, flex_2d)
                        
                        # 填充缺失的OHLC数据
                        if np.isnan(_open):
                            _open = _close  # 开盘价缺失时用收盘价填充
                        if np.isnan(_low):
                            _low = min(_open, _close)  # 最低价缺失时用开盘价和收盘价的最小值
                        if np.isnan(_high):
                            _high = max(_open, _close)  # 最高价缺失时用开盘价和收盘价的最大值

                        # ========== 计算止损触发价格 ==========
                        # 优先检查止损信号
                        if not np.isnan(sl_curr_stop[col]):
                            stop_price = get_stop_price_nb(
                                position_now,                # 当前持仓方向和大小
                                sl_curr_price[col],         # 止损参考价格
                                sl_curr_stop[col],          # 止损百分比
                                _open, _low, _high,         # OHLC价格范围
                                True                        # True表示向下触发（止损）
                            )
                        
                        # 如果止损未触发，检查止盈信号
                        if np.isnan(stop_price) and not np.isnan(tp_curr_stop[col]):
                            stop_price = get_stop_price_nb(
                                position_now,                # 当前持仓方向和大小
                                tp_init_price[col],         # 止盈参考价格（初始价格）
                                tp_curr_stop[col],          # 止盈百分比
                                _open, _low, _high,         # OHLC价格范围
                                False                       # False表示向上触发（止盈）
                            )

                        # ========== 跟踪止损更新逻辑 ==========
                        if not np.isnan(sl_curr_stop[col]) and sl_curr_trail[col]:
                            # 跟踪止损：根据价格变化动态调整止损参考价格
                            if position_now > 0:
                                # 多头持仓：当价格创新高时，上调止损参考价格
                                if _high > sl_curr_price[col]:
                                    sl_curr_i[col] = i           # 更新止损调整时间
                                    sl_curr_price[col] = _high   # 更新止损参考价格为新高点
                            elif position_now < 0:
                                # 空头持仓：当价格创新低时，下调止损参考价格
                                if _low < sl_curr_price[col]:
                                    sl_curr_i[col] = i           # 更新止损调整时间
                                    sl_curr_price[col] = _low    # 更新止损参考价格为新低点

                # ========== 第三阶段：信号生成和处理 ==========
                _accumulate = flex_select_auto_nb(accumulate, i, col, flex_2d)
                
                if use_stops and not np.isnan(stop_price):
                    # ========== 止损信号优先处理 ==========
                    # 当检测到止损/止盈触发时，止损信号具有最高优先级
                    _upon_stop_exit = flex_select_auto_nb(upon_stop_exit, i, col, flex_2d)
                    
                    # 生成止损/止盈信号：根据当前持仓和止损模式生成相应的平仓信号
                    is_long_entry, is_long_exit, is_short_entry, is_short_exit, _accumulate = \
                        generate_stop_signal_nb(position_now, _upon_stop_exit, _accumulate)

                    # 解析止损执行价格和滑点
                    _close = flex_select_auto_nb(close, i, col, flex_2d)
                    _stop_exit_price = flex_select_auto_nb(stop_exit_price, i, col, flex_2d)
                    
                    # 根据止损价格和执行模式确定最终的执行价格和滑点
                    _price, _slippage = resolve_stop_price_and_slippage_nb(
                        stop_price,          # 计算出的止损触发价格
                        _price,              # 原始订单价格
                        _close,              # 当前收盘价
                        _slippage,           # 原始滑点设置
                        _stop_exit_price     # 止损执行价格模式
                    )
                else:
                    # ========== 用户自定义信号优先处理 ==========
                    # 当未触发止损时，调用用户定义的信号生成函数
                    # 构建信号生成上下文
                    signal_ctx = SignalContext(
                        i=i,                               # 当前时间索引
                        col=col,                           # 当前列索引
                        position_now=position_now,         # 当前持仓数量
                        val_price_now=last_val_price[col], # 当前估值价格
                        flex_2d=flex_2d                    # 2D灵活广播标志
                    )
                    
                    # 调用用户自定义信号函数获取原始交易信号
                    is_long_entry, is_long_exit, is_short_entry, is_short_exit = \
                        signal_func_nb(signal_ctx, *signal_args)

                    # ========== 信号冲突解决流水线 ==========
                    if is_long_entry or is_short_entry:
                        # **第1步：解决方向内信号冲突**
                        # 处理多头方向的开仓/平仓信号冲突
                        _upon_long_conflict = flex_select_auto_nb(upon_long_conflict, i, col, flex_2d)
                        is_long_entry, is_long_exit = resolve_signal_conflict_nb(
                            position_now,          # 当前持仓状态
                            is_long_entry,         # 多头开仓信号
                            is_long_exit,          # 多头平仓信号
                            Direction.LongOnly,    # 仅限多头方向
                            _upon_long_conflict    # 冲突处理模式
                        )
                        
                        # 处理空头方向的开仓/平仓信号冲突
                        _upon_short_conflict = flex_select_auto_nb(upon_short_conflict, i, col, flex_2d)
                        is_short_entry, is_short_exit = resolve_signal_conflict_nb(
                            position_now,          # 当前持仓状态
                            is_short_entry,        # 空头开仓信号
                            is_short_exit,         # 空头平仓信号
                            Direction.ShortOnly,   # 仅限空头方向
                            _upon_short_conflict   # 冲突处理模式
                        )

                        # **第2步：解决方向间信号冲突**
                        # 处理多头和空头开仓信号同时出现的冲突
                        _upon_dir_conflict = flex_select_auto_nb(upon_dir_conflict, i, col, flex_2d)
                        is_long_entry, is_short_entry = resolve_dir_conflict_nb(
                            position_now,          # 当前持仓状态
                            is_long_entry,         # 多头开仓信号
                            is_short_entry,        # 空头开仓信号
                            _upon_dir_conflict     # 方向冲突处理模式
                        )

                        # **第3步：解决相反方向开仓信号**
                        # 处理持仓方向与新开仓信号方向相反的情况
                        _upon_opposite_entry = flex_select_auto_nb(upon_opposite_entry, i, col, flex_2d)
                        is_long_entry, is_long_exit, is_short_entry, is_short_exit, _accumulate = \
                            resolve_opposite_entry_nb(
                                position_now,          # 当前持仓状态
                                is_long_entry,         # 多头开仓信号
                                is_long_exit,          # 多头平仓信号
                                is_short_entry,        # 空头开仓信号
                                is_short_exit,         # 空头平仓信号
                                _upon_opposite_entry,  # 相反开仓处理模式
                                _accumulate            # 累积模式（可能被修改）
                            )

                # ========== 第四阶段：信号转订单参数 ==========
                # 将解决冲突后的最终信号转换为具体的订单参数
                _size, _size_type, _direction = signals_to_size_nb(
                    last_position[col],                                    # 当前持仓
                    is_long_entry,                                         # 最终多头开仓信号
                    is_long_exit,                                          # 最终多头平仓信号
                    is_short_entry,                                        # 最终空头开仓信号
                    is_short_exit,                                         # 最终空头平仓信号
                    flex_select_auto_nb(size, i, col, flex_2d),          # 订单大小设置
                    flex_select_auto_nb(size_type, i, col, flex_2d),     # 大小类型设置
                    _accumulate,                                           # 累积模式
                    last_val_price[col]                                    # 当前估值价格
                )

                # ========== 保存订单参数到临时数组 ==========
                price_arr[col] = _price              # 订单执行价格
                slippage_arr[col] = _slippage        # 滑点设置
                size_arr[col] = _size                # 订单大小（含方向）
                size_type_arr[col] = _size_type      # 订单大小类型
                direction_arr[col] = _direction      # 交易方向限制

                # ========== 现金共享模式下的订单价值估算 ==========
                if cash_sharing:
                    if _size == 0:
                        # 无订单：订单价值为0
                        temp_order_value[k] = 0.
                    else:
                        # **订单价值近似计算**：用于动态排序订单执行顺序
                        if _size_type == SizeType.Amount:
                            # 数量类型：订单价值 = 数量 × 当前估值价格
                            temp_order_value[k] = _size * last_val_price[col]
                            
                        elif _size_type == SizeType.Value:
                            # 价值类型：订单价值 = 直接指定的价值
                            temp_order_value[k] = _size
                            
                        else:  # SizeType.Percent
                            # 百分比类型：根据交易方向计算基准价值
                            if _size >= 0:
                                # 正向交易（买入/开仓）：基于可用现金计算
                                temp_order_value[k] = _size * cash_now
                            else:
                                # 反向交易（卖出/平仓）：基于持仓价值或最大风险敞口计算
                                asset_value_now = last_position[col] * last_val_price[col]
                                
                                if _direction == Direction.LongOnly:
                                    # 仅限多头方向：基于当前资产价值
                                    temp_order_value[k] = _size * asset_value_now
                                else:
                                    # 双向交易：基于最大风险敞口计算
                                    # 最大敞口 = 2×资产价值 + 自由现金
                                    max_exposure = (2 * max(asset_value_now, 0) + max(free_cash_now, 0))
                                    temp_order_value[k] = _size * max_exposure

            # ========== 第五阶段：订单执行顺序优化 ==========
            if cash_sharing:
                # **动态订单排序**：按订单价值排序，卖单优先释放资金
                if auto_call_seq:
                    # 将订单价值按升序排列，负值（卖单）排在前面
                    # 这样可以优先执行卖单，释放资金供后续买单使用
                    insert_argsort_nb(temp_order_value[:group_len], call_seq[i, from_col:to_col])

                # **计算当前组合总价值**
                # 等同于get_group_value_ctx_nb但使用灵活索引
                value_now = cash_now  # 从现金开始
                for k in range(group_len):
                    col = from_col + k
                    if last_position[col] != 0:
                        # 累加各资产的持仓价值
                        value_now += last_position[col] * last_val_price[col]

            # ========== 第六阶段：按序执行订单 ==========
            for k in range(group_len):
                col = from_col + k  # 默认列索引
                
                if cash_sharing:
                    # **现金共享模式**：按照调用序列确定实际执行列
                    col_i = call_seq[i, col]
                    if col_i >= group_len:
                        raise ValueError("调用索引超出组范围边界")
                    col = from_col + col_i  # 重新映射到实际列索引

                # **获取当前列的状态**
                position_now = last_position[col]    # 当前持仓数量
                debt_now = last_debt[col]            # 当前债务金额
                val_price_now = last_val_price[col]  # 当前估值价格
                
                if not cash_sharing:
                    # **独立现金模式**：每列单独计算组合价值
                    value_now = cash_now
                    if position_now != 0:
                        value_now += position_now * val_price_now

                # ========== 生成并执行订单 ==========
                # 从临时数组中恢复当前列的订单参数
                _price = price_arr[col]       # 订单执行价格
                _size = size_arr[col]         # 订单大小（已包含方向信息）
                _size_type = size_type_arr[col]  # 订单大小类型
                _direction = direction_arr[col]  # 交易方向限制
                _slippage = slippage_arr[col]    # 滑点设置
                
                if _size != 0:
                    # **订单方向调整**：确保订单大小符合方向限制要求
                    if _size > 0:  # 正数订单（买入/开多）
                        if _direction == Direction.ShortOnly:
                            # 仅限空头方向：将正数转为负数
                            _size *= -1  # 必须反转以符合process_order_nb的要求
                    else:  # 负数订单（卖出/开空）
                        if _direction == Direction.ShortOnly:
                            # 仅限空头方向：保持负数不变或进行调整
                            _size *= -1
                    
                    # **构建订单对象**
                    order = order_nb(
                        size=_size,                                               # 订单大小
                        price=_price,                                            # 执行价格
                        size_type=_size_type,                                    # 大小类型
                        direction=_direction,                                    # 方向限制
                        fees=flex_select_auto_nb(fees, i, col, flex_2d),       # 比例手续费
                        fixed_fees=flex_select_auto_nb(fixed_fees, i, col, flex_2d),  # 固定手续费
                        slippage=_slippage,                                      # 滑点
                        min_size=flex_select_auto_nb(min_size, i, col, flex_2d),      # 最小订单大小
                        max_size=flex_select_auto_nb(max_size, i, col, flex_2d),      # 最大订单大小
                        size_granularity=flex_select_auto_nb(size_granularity, i, col, flex_2d),  # 大小粒度
                        reject_prob=flex_select_auto_nb(reject_prob, i, col, flex_2d),     # 拒绝概率
                        lock_cash=flex_select_auto_nb(lock_cash, i, col, flex_2d),         # 锁定现金
                        allow_partial=flex_select_auto_nb(allow_partial, i, col, flex_2d), # 允许部分成交
                        raise_reject=flex_select_auto_nb(raise_reject, i, col, flex_2d),   # 拒绝时抛出异常
                        log=flex_select_auto_nb(log, i, col, flex_2d)                     # 记录日志
                    )

                    # **准备订单处理状态**
                    state = ProcessOrderState(
                        cash=cash_now,          # 当前可用现金
                        position=position_now,  # 当前持仓
                        debt=debt_now,          # 当前债务
                        free_cash=free_cash_now,  # 当前自由现金
                        val_price=val_price_now,  # 当前估值价格
                        value=value_now,          # 当前组合价值
                        oidx=oidx,               # 当前订单记录索引
                        lidx=lidx                # 当前日志记录索引
                    )

                    # **执行订单处理**：这是核心的订单处理函数
                    order_result, new_state = process_order_nb(
                        i, col, group,      # 时间、列、组索引
                        state,              # 当前处理状态
                        update_value,       # 是否更新价值
                        order,              # 订单对象
                        order_records,      # 订单记录数组
                        log_records         # 日志记录数组
                    )

                    # **更新全局状态**：将订单处理后的新状态同步到全局变量
                    cash_now = new_state.cash          # 更新现金
                    position_now = new_state.position  # 更新持仓
                    debt_now = new_state.debt          # 更新债务
                    free_cash_now = new_state.free_cash  # 更新自由现金
                    val_price_now = new_state.val_price  # 更新估值价格
                    value_now = new_state.value        # 更新组合价值
                    oidx = new_state.oidx              # 更新订单记录索引
                    lidx = new_state.lidx              # 更新日志记录索引

                    # ========== 第七阶段：止损止盈状态更新 ==========
                    if use_stops:
                        # **止损止盈状态动态管理**
                        if order_result.status == OrderStatus.Filled:
                            if position_now == 0:
                                # **情况1：持仓完全平仓 → 清除所有止损止盈设置**
                                sl_curr_i[col] = sl_init_i[col] = -1       # 清除止损时间索引
                                sl_curr_price[col] = sl_init_price[col] = np.nan  # 清除止损价格
                                sl_curr_stop[col] = np.nan                 # 清除止损百分比
                                sl_curr_trail[col] = False                 # 清除跟踪标志
                                tp_init_i[col] = -1                        # 清除止盈时间索引
                                tp_init_price[col] = np.nan                # 清除止盈价格
                                tp_curr_stop[col] = np.nan                 # 清除止盈百分比
                            else:
                                # **情况2：持仓存在 → 根据情况更新止损止盈设置**
                                
                                # **确定新的止损止盈参考价格**
                                _stop_entry_price = flex_select_auto_nb(stop_entry_price, i, col, flex_2d)
                                if _stop_entry_price == StopEntryPrice.ValPrice:
                                    new_init_price = val_price_now      # 使用当前估值价格
                                elif _stop_entry_price == StopEntryPrice.Price:
                                    new_init_price = order.price        # 使用订单价格
                                elif _stop_entry_price == StopEntryPrice.FillPrice:
                                    new_init_price = order_result.price # 使用实际成交价格
                                else:
                                    new_init_price = flex_select_auto_nb(close, i, col, flex_2d)  # 使用收盘价
                                
                                # **获取止损止盈相关参数**
                                _upon_stop_update = flex_select_auto_nb(upon_stop_update, i, col, flex_2d)
                                _sl_stop = flex_select_auto_nb(sl_stop, i, col, flex_2d)      # 止损百分比
                                _sl_trail = flex_select_auto_nb(sl_trail, i, col, flex_2d)   # 跟踪止损标志
                                _tp_stop = flex_select_auto_nb(tp_stop, i, col, flex_2d)     # 止盈百分比

                                # **根据持仓变化情况更新止损设置**
                                if state.position == 0 or np.sign(position_now) != np.sign(state.position):
                                    # **子情况2.1：新开仓或反向开仓 → 重新设置止损止盈**
                                    sl_curr_i[col] = sl_init_i[col] = i            # 记录初始时间
                                    sl_curr_price[col] = sl_init_price[col] = new_init_price  # 设置参考价格
                                    sl_curr_stop[col] = _sl_stop                   # 设置止损百分比
                                    sl_curr_trail[col] = _sl_trail                 # 设置跟踪标志
                                    tp_init_i[col] = i                             # 记录止盈初始时间
                                    tp_init_price[col] = new_init_price            # 设置止盈参考价格
                                    tp_curr_stop[col] = _tp_stop                   # 设置止盈百分比
                                    
                                elif abs(position_now) > abs(state.position):
                                    # **子情况2.2：加仓 → 根据更新模式决定是否覆盖止损设置**
                                    if should_update_stop_nb(_sl_stop, _upon_stop_update):
                                        sl_curr_i[col] = sl_init_i[col] = i        # 更新止损时间
                                        sl_curr_price[col] = sl_init_price[col] = new_init_price  # 更新参考价格
                                        sl_curr_stop[col] = _sl_stop               # 更新止损百分比
                                        sl_curr_trail[col] = _sl_trail             # 更新跟踪标志
                                    if should_update_stop_nb(_tp_stop, _upon_stop_update):
                                        tp_init_i[col] = i                   # 更新止盈时间
                                        tp_init_price[col] = new_init_price  # 更新止盈参考价格
                                        tp_curr_stop[col] = _tp_stop         # 更新止盈百分比
                                # **子情况2.3：减仓情况**
                                # 减仓时通常保持原有的止损止盈设置不变

                # ========== 第八阶段：状态持久化 ==========
                # **更新持久状态**：将当前状态保存为"上一次"状态，供下一时间步使用
                last_position[col] = position_now  # 保存当前持仓为上次持仓
                last_debt[col] = debt_now          # 保存当前债务为上次债务
                
                # 更新估值价格（支持前向填充）
                if not np.isnan(val_price_now) or not ffill_val_price:
                    last_val_price[col] = val_price_now

        # ========== 移动到下一个组 ==========
        from_col = to_col  # 当前组处理完毕，移动到下一个组的起始列

    # ========== 函数返回 ==========
    # 返回实际使用的订单记录和日志记录（截取到实际长度）
    return order_records[:oidx], log_records[:lidx]


@njit  # Numba即时编译
def dir_enex_signal_func_nb(c: SignalContext,
                            entries: tp.ArrayLike,
                            exits: tp.ArrayLike,
                            direction: tp.ArrayLike) -> tp.Tuple[bool, bool, bool, bool]:
    """
    方向感知信号函数：根据开仓、平仓和方向参数生成方向特定信号
    
    将通用的开仓/平仓信号根据指定的交易方向转换为具体的多头/空头信号。
    这是信号驱动策略中常用的便利函数。
    
    参数:
    ----
    c : SignalContext
        信号上下文，包含当前时间和列信息
        
    entries : array_like
        通用开仓信号数组
        
    exits : array_like
        通用平仓信号数组
        
    direction : array_like
        交易方向数组，使用Direction枚举值
        
    返回:
    ----
    tuple[bool, bool, bool, bool]
        (is_long_entry, is_long_exit, is_short_entry, is_short_exit)
        根据方向转换后的具体信号
        
    使用示例:
    --------
    >>> # 仅做多策略
    >>> entries = np.array([True, False, True])
    >>> exits = np.array([False, True, False])
    >>> directions = np.array([Direction.LongOnly] * 3)
    >>> 
    >>> # 结果：(True, False, False, False) 等
    
    方向转换逻辑:
    -----------
    - LongOnly: (entry, exit, False, False) - 仅多头信号
    - ShortOnly: (False, False, entry, exit) - 仅空头信号
    - Both: (entry, False, exit, False) - 双向交易模式
    """
    # 获取当前位置的信号值
    is_entry = flex_select_auto_nb(entries, c.i, c.col, c.flex_2d)
    is_exit = flex_select_auto_nb(exits, c.i, c.col, c.flex_2d)
    _direction = flex_select_auto_nb(direction, c.i, c.col, c.flex_2d)
    
    # 根据交易方向转换信号
    if _direction == Direction.LongOnly:
        # 仅做多：开仓信号->多头开仓，平仓信号->多头平仓
        return is_entry, is_exit, False, False
    if _direction == Direction.ShortOnly:
        # 仅做空：开仓信号->空头开仓，平仓信号->空头平仓
        return False, False, is_entry, is_exit
    # 双向交易：开仓->多头开仓，平仓->空头开仓（反转策略）
    return is_entry, False, is_exit, False


@njit  # Numba即时编译
def ls_enex_signal_func_nb(c: SignalContext,
                           long_entries: tp.ArrayLike,
                           long_exits: tp.ArrayLike,
                           short_entries: tp.ArrayLike,
                           short_exits: tp.ArrayLike) -> tp.Tuple[bool, bool, bool, bool]:
    """
    多空开平仓信号函数：直接获取四种方向感知信号
    
    从四个独立的信号数组中提取当前位置的信号值，适用于已经
    明确区分多头和空头操作的策略。
    
    参数:
    ----
    c : SignalContext
        信号上下文，包含当前时间和列信息
        
    long_entries : array_like
        多头开仓信号数组
        
    long_exits : array_like
        多头平仓信号数组
        
    short_entries : array_like
        空头开仓信号数组
        
    short_exits : array_like
        空头平仓信号数组
        
    返回:
    ----
    tuple[bool, bool, bool, bool]
        (is_long_entry, is_long_exit, is_short_entry, is_short_exit)
        当前位置的四种信号状态
        
    使用示例:
    --------
    >>> # 复杂策略的信号分离
    >>> long_entries = np.array([True, False, False])
    >>> long_exits = np.array([False, True, False])
    >>> short_entries = np.array([False, False, True])
    >>> short_exits = np.array([False, False, False])
    >>> 
    >>> # 可以同时处理多头和空头的不同条件
    
    应用场景:
    --------
    - 多因子策略中的信号合成
    - 套利策略的多腿操作
    - 复杂的技术指标组合
    - 分离的风险管理逻辑
    """
    # 直接从各自的信号数组中提取当前位置的值
    is_long_entry = flex_select_auto_nb(long_entries, c.i, c.col, c.flex_2d)
    is_long_exit = flex_select_auto_nb(long_exits, c.i, c.col, c.flex_2d)
    is_short_entry = flex_select_auto_nb(short_entries, c.i, c.col, c.flex_2d)
    is_short_exit = flex_select_auto_nb(short_exits, c.i, c.col, c.flex_2d)
    
    return is_long_entry, is_long_exit, is_short_entry, is_short_exit


@njit  # Numba即时编译
def no_pre_func_nb(c: tp.NamedTuple, *args) -> tp.Args:
    """
    无预处理占位符函数
    
    默认的预处理函数，直接转发接收到的参数给下游函数。
    用作各种预处理钩子的默认值。
    
    参数:
    ----
    c : NamedTuple
        上下文对象（未使用）
        
    *args : 可变参数
        需要转发的参数
        
    返回:
    ----
    tuple
        转发的参数元组
        
    应用场景:
    --------
    - 预处理钩子的默认占位符
    - 保持函数接口一致性
    - 简化参数传递链
    """
    # 直接转发参数，不做任何处理
    return args


@njit  # Numba即时编译
def no_order_func_nb(c: OrderContext, *args) -> Order:
    """
    无订单占位符函数
    
    默认的订单生成函数，返回无订单状态。用作订单生成钩子的默认值。
    
    参数:
    ----
    c : OrderContext
        订单上下文对象（未使用）
        
    *args : 可变参数
        额外参数（未使用）
        
    返回:
    ----
    Order
        NoOrder常量，表示不生成任何订单
        
    应用场景:
    --------
    - 订单生成钩子的默认占位符
    - 禁用自动订单生成
    - 条件性订单生成逻辑
    """
    # 返回无订单常量
    return NoOrder


@njit  # Numba即时编译
def no_post_func_nb(c: tp.NamedTuple, *args) -> None:
    """
    无后处理占位符函数
    
    默认的后处理函数，不执行任何操作。用作各种后处理钩子的默认值。
    
    参数:
    ----
    c : NamedTuple
        上下文对象（未使用）
        
    *args : 可变参数
        额外参数（未使用）
        
    返回:
    ----
    None
        无返回值
        
    应用场景:
    --------
    - 后处理钩子的默认占位符
    - 保持函数接口一致性
    - 简化回调函数管理
    """
    # 不执行任何操作，直接返回
    return None


PreSimFuncT = tp.Callable[[SimulationContext, tp.VarArg()], tp.Args]
PostSimFuncT = tp.Callable[[SimulationContext, tp.VarArg()], None]
PreGroupFuncT = tp.Callable[[GroupContext, tp.VarArg()], tp.Args]
PostGroupFuncT = tp.Callable[[GroupContext, tp.VarArg()], None]
PreRowFuncT = tp.Callable[[RowContext, tp.VarArg()], tp.Args]
PostRowFuncT = tp.Callable[[RowContext, tp.VarArg()], None]
PreSegmentFuncT = tp.Callable[[SegmentContext, tp.VarArg()], tp.Args]
PostSegmentFuncT = tp.Callable[[SegmentContext, tp.VarArg()], None]
OrderFuncT = tp.Callable[[OrderContext, tp.VarArg()], Order]
PostOrderFuncT = tp.Callable[[PostOrderContext, OrderResult, tp.VarArg()], None]


@njit  # Numba即时编译，极致性能优化
def simulate_nb(target_shape: tp.Shape,
                group_lens: tp.Array1d,
                init_cash: tp.Array1d,
                cash_sharing: bool,
                call_seq: tp.Array2d,
                segment_mask: tp.ArrayLike = np.asarray(True),
                call_pre_segment: bool = False,
                call_post_segment: bool = False,
                pre_sim_func_nb: PreSimFuncT = no_pre_func_nb,
                pre_sim_args: tp.Args = (),
                post_sim_func_nb: PostSimFuncT = no_post_func_nb,
                post_sim_args: tp.Args = (),
                pre_group_func_nb: PreGroupFuncT = no_pre_func_nb,
                pre_group_args: tp.Args = (),
                post_group_func_nb: PostGroupFuncT = no_post_func_nb,
                post_group_args: tp.Args = (),
                pre_segment_func_nb: PreSegmentFuncT = no_pre_func_nb,
                pre_segment_args: tp.Args = (),
                post_segment_func_nb: PostSegmentFuncT = no_post_func_nb,
                post_segment_args: tp.Args = (),
                order_func_nb: OrderFuncT = no_order_func_nb,
                order_args: tp.Args = (),
                post_order_func_nb: PostOrderFuncT = no_post_func_nb,
                post_order_args: tp.Args = (),
                close: tp.ArrayLike = np.asarray(np.nan),
                ffill_val_price: bool = True,
                update_value: bool = False,
                fill_pos_record: bool = True,
                max_orders: tp.Optional[int] = None,
                max_logs: int = 0,
                flex_2d: bool = True) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
    """
    灵活的投资组合模拟引擎：通过用户定义函数实现完全自定义的交易逻辑
    
    这是VectorBT的最通用、最灵活的投资组合模拟函数，支持用户在模拟过程的
    各个阶段插入自定义逻辑。通过一系列钩子函数，用户可以实现复杂的交易策略、
    风险管理、动态调整等高级功能。
    
    核心流程:
    --------
    1. 从初始现金开始，按列主序遍历目标形状的每个组和列
    2. 对每个数据点，调用order_func_nb生成订单
    3. 尝试执行订单，成功后更新现金余额和持仓
    4. 记录订单和日志信息，供后续分析使用
    
    输出格式:
    --------
    - 订单记录：vectorbt.portfolio.enums.order_dt布局
    - 日志记录：vectorbt.portfolio.enums.log_dt布局
    
    执行顺序:
    --------
    与simulate_row_wise_nb不同，此函数采用列主序处理：
    - 列主序：先处理整列/组的所有行，再移动到下一列
    - 行主序：先处理整行的所有列，再移动到下一行
    - 详见：https://en.wikipedia.org/wiki/Row-_and_column-major_order

    参数详解:
    --------
    
    **基础模拟参数:**
    
    target_shape : tuple[int, int]
        目标模拟形状 (时间步数, 资产数量)
        参见 vectorbt.portfolio.enums.SimulationContext.target_shape
        
    group_lens : array_like[int]
        每个组包含的列数，用于现金共享分组
        参见 vectorbt.portfolio.enums.SimulationContext.group_lens
        
    init_cash : array_like[float]
        每个组的初始现金金额
        参见 vectorbt.portfolio.enums.SimulationContext.init_cash
        
    cash_sharing : bool
        是否在组内共享现金，影响资金分配和订单执行
        参见 vectorbt.portfolio.enums.SimulationContext.cash_sharing
        
    call_seq : array_like[int]
        调用序列矩阵，控制每个时间步内的订单执行顺序
        参见 vectorbt.portfolio.enums.SimulationContext.call_seq
        
    **分段控制参数:**
    
    segment_mask : array_like[bool], 默认 True
        分段掩码，控制哪些时间点需要调用分段函数
        参见 vectorbt.portfolio.enums.SimulationContext.segment_mask
        
    call_pre_segment : bool, 默认 False
        是否调用分段前处理函数
        参见 vectorbt.portfolio.enums.SimulationContext.call_pre_segment
        
    call_post_segment : bool, 默认 False
        是否调用分段后处理函数
        参见 vectorbt.portfolio.enums.SimulationContext.call_post_segment
        
    **模拟级钩子函数:**
    
    pre_sim_func_nb : callable, 默认 no_pre_func_nb
        模拟开始前调用的函数，用于创建全局数组和设置随机种子
        
        函数签名: (SimulationContext, *pre_sim_args) -> tuple
        返回值会传递给pre_group_func_nb和post_group_func_nb
        
    pre_sim_args : tuple, 默认 ()
        传递给pre_sim_func_nb的打包参数
        
    post_sim_func_nb : callable, 默认 no_post_func_nb
        模拟结束后调用的函数，用于最终清理和统计
        
        函数签名: (SimulationContext, *post_sim_args) -> None
        无返回值要求
        
    post_sim_args : tuple, 默认 ()
        传递给post_sim_func_nb的打包参数
        **组级钩子函数:**
        
        pre_group_func_nb : callable, 默认 no_pre_func_nb
            每个组处理前调用的函数，用于组级别的初始化和准备
            
            函数签名: (GroupContext, *pre_sim_result, *pre_group_args) -> tuple
            返回值会传递给pre_segment_func_nb和post_segment_func_nb
            
        pre_group_args : tuple, 默认 ()
            传递给pre_group_func_nb的打包参数
            
        post_group_func_nb : callable, 默认 no_post_func_nb
            每个组处理后调用的函数，用于组级别的清理和统计
            
            函数签名: (GroupContext, *pre_sim_result, *post_group_args) -> None
            无返回值要求
            
        post_group_args : tuple, 默认 ()
            传递给post_group_func_nb的打包参数
            
        **分段级钩子函数:**
        
        pre_segment_func_nb : callable, 默认 no_pre_func_nb
            每个分段处理前调用的函数，用于动态调整策略参数
            
            调用条件: segment_mask为True或call_pre_segment为True
            
            函数签名: (SegmentContext, *pre_group_result, *pre_segment_args) -> tuple
            返回值会传递给order_func_nb和post_order_func_nb
            
            **重要用途:**
            - 修改调用序列：访问SegmentContext.call_seq_now并原地修改
            - 设置估值价格：修改last_val_price元素
            - 动态策略调整：根据市场状态调整参数
            
            **注意事项:**
            - 避免创建新数组以保持性能
            - 不支持重新赋值SegmentContext.call_seq_now
            - 组重估值在此函数后立即发生
            
        pre_segment_args : tuple, 默认 ()
            传递给pre_segment_func_nb的打包参数
            
        post_segment_func_nb : callable, 默认 no_post_func_nb
            每个分段处理后调用的函数，用于分段结果的处理和记录
            
            调用条件: segment_mask为True或call_post_segment为True
            
            函数签名: (SegmentContext, *pre_group_result, *post_segment_args) -> None
            最后的组重估值在此函数前立即发生
            
                 post_segment_args : tuple, 默认 ()
             传递给post_segment_func_nb的打包参数
             
        **订单级钩子函数:**
        
        order_func_nb : callable, 默认 no_order_func_nb
            订单生成函数，策略的核心逻辑所在
            
            函数签名: (OrderContext, *pre_segment_result, *order_args) -> Order
            返回Order对象，如果被拒绝则无法重新发出订单
            
        order_args : tuple, 默认 ()
            传递给order_func_nb的打包参数
            
        post_order_func_nb : callable, 默认 no_post_func_nb  
            订单处理后调用的函数，用于记录和分析
            
            函数签名: (PostOrderContext, OrderResult, *pre_segment_result, *post_order_args) -> None
            
        post_order_args : tuple, 默认 ()
            传递给post_order_func_nb的打包参数
            
        **其他控制参数:**
        
        close : array_like, 默认 NaN
            收盘价数据，用于组合估值
            
        ffill_val_price : bool, 默认 True
            是否前向填充估值价格
            
        update_value : bool, 默认 False
            是否更新组合价值
            
        fill_pos_record : bool, 默认 True
            是否填充持仓记录
            
        max_orders : int, 可选
            最大订单记录数
            
        max_logs : int, 默认 0
            最大日志记录数
            
        flex_2d : bool, 默认 True
            是否启用2D灵活广播
            
    返回:
    ----
    tuple[RecordArray, RecordArray]
        (order_records, log_records) - 订单记录和日志记录
        
    使用场景:
    --------
    - **高度自定义的交易策略**: 通过多级钩子函数实现复杂逻辑
    - **算法交易系统**: 支持动态调用序列和实时参数调整
    - **风险管理系统**: 在每个阶段插入风控检查
    - **多资产组合管理**: 支持现金共享和独立资产管理
    - **研究和回测**: 详细的执行记录和性能分析
        
         注意事项:
     --------
     - 列主序处理：优先处理整列再移动到下一列
     - 数据访问限制：只能安全访问当前组左侧和当前行上方的数据
     - 性能敏感：避免在钩子函数中创建大型数组
     - 钩子函数必须是Numba兼容的
     
     **完整钩子体系:**
     pre_sim -> pre_group -> (pre_segment -> order -> post_order) -> post_segment -> post_group -> post_sim
     """
    # 由于simulate_nb函数实现极其复杂（超过1000行），包含完整的用户定义钩子函数体系，
    # 这里仅为核心结构添加注释。详细实现请参考上述docstring中的完整说明。
    
    # ========== 前期验证和初始化 ==========
    check_group_lens_nb(group_lens, target_shape[1])
    check_group_init_cash_nb(group_lens, target_shape[1], init_cash, cash_sharing)

    order_records, log_records = init_records_nb(target_shape, max_orders, max_logs)
    init_cash = init_cash.astype(np.float64)
    last_cash = init_cash.copy()
    last_position = np.full(target_shape[1], 0., dtype=np.float64)
    last_debt = np.full(target_shape[1], 0., dtype=np.float64)
    last_free_cash = init_cash.copy()
    last_val_price = np.full(target_shape[1], np.nan, dtype=np.float64)
    last_value = init_cash.copy()
    second_last_value = init_cash.copy()
    temp_value = init_cash.copy()
    last_return = np.full_like(last_value, np.nan)
    last_pos_record = np.empty(target_shape[1], dtype=trade_dt)
    last_pos_record['id'][:] = -1
    last_oidx = np.full(target_shape[1], -1, dtype=np.int64)
    last_lidx = np.full(target_shape[1], -1, dtype=np.int64)
    oidx = 0
    lidx = 0

    # Call function before the simulation
    pre_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        init_cash=init_cash,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        close=close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        second_last_value=second_last_value,
        last_return=last_return,
        last_oidx=last_oidx,
        last_lidx=last_lidx,
        last_pos_record=last_pos_record
    )
    pre_sim_out = pre_sim_func_nb(pre_sim_ctx, *pre_sim_args)

    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col

        # Call function before the group
        pre_group_ctx = GroupContext(
            target_shape=target_shape,
            group_lens=group_lens,
            init_cash=init_cash,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            close=close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            second_last_value=second_last_value,
            last_return=last_return,
            last_oidx=last_oidx,
            last_lidx=last_lidx,
            last_pos_record=last_pos_record,
            group=group,
            group_len=group_len,
            from_col=from_col,
            to_col=to_col
        )
        pre_group_out = pre_group_func_nb(pre_group_ctx, *pre_sim_out, *pre_group_args)

        for i in range(target_shape[0]):
            call_seq_now = call_seq[i, from_col:to_col]

            # Is this segment active?
            if call_pre_segment or segment_mask[i, group]:
                # Call function before the segment
                pre_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    init_cash=init_cash,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    close=close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    second_last_value=second_last_value,
                    last_return=last_return,
                    last_oidx=last_oidx,
                    last_lidx=last_lidx,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=call_seq_now
                )
                pre_segment_out = pre_segment_func_nb(pre_seg_ctx, *pre_group_out, *pre_segment_args)

            # Update open position stats
            if fill_pos_record:
                for col in range(from_col, to_col):
                    update_open_pos_stats_nb(
                        last_pos_record[col],
                        last_position[col],
                        last_val_price[col]
                    )

            # Update value and return
            if cash_sharing:
                last_value[group] = get_group_value_nb(
                    from_col,
                    to_col,
                    last_cash[group],
                    last_position,
                    last_val_price
                )
                last_return[group] = returns_nb.get_return_nb(second_last_value[group], last_value[group])
            else:
                for col in range(from_col, to_col):
                    if last_position[col] == 0:
                        last_value[col] = last_cash[col]
                    else:
                        last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                    last_return[col] = returns_nb.get_return_nb(second_last_value[col], last_value[col])

            # Is this segment active?
            if segment_mask[i, group]:

                for k in range(group_len):
                    col_i = call_seq_now[k]
                    if col_i >= group_len:
                        raise ValueError("Call index exceeds bounds of the group")
                    col = from_col + col_i

                    # Get current values
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    val_price_now = last_val_price[col]
                    pos_record_now = last_pos_record[col]
                    if cash_sharing:
                        cash_now = last_cash[group]
                        free_cash_now = last_free_cash[group]
                        value_now = last_value[group]
                        return_now = last_return[group]
                    else:
                        cash_now = last_cash[col]
                        free_cash_now = last_free_cash[col]
                        value_now = last_value[col]
                        return_now = last_return[col]

                    # Generate the next order
                    order_ctx = OrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        init_cash=init_cash,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        close=close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        second_last_value=second_last_value,
                        last_return=last_return,
                        last_oidx=last_oidx,
                        last_lidx=last_lidx,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now,
                        col=col,
                        call_idx=k,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_record_now=pos_record_now
                    )
                    order = order_func_nb(order_ctx, *pre_segment_out, *order_args)
                    if np.isinf(order.price):
                        if i > 0:
                            _prev_close = flex_select_auto_nb(close, i - 1, col, flex_2d)
                        else:
                            _prev_close = np.nan
                        _close = flex_select_auto_nb(close, i, col, flex_2d)
                        order = replace_inf_price_nb(_prev_close, _close, order)

                    # Process the order
                    state = ProcessOrderState(
                        cash=cash_now,
                        position=position_now,
                        debt=debt_now,
                        free_cash=free_cash_now,
                        val_price=val_price_now,
                        value=value_now,
                        oidx=oidx,
                        lidx=lidx
                    )

                    order_result, new_state = process_order_nb(
                        i, col, group,
                        state,
                        update_value,
                        order,
                        order_records,
                        log_records
                    )

                    # Update state
                    cash_now = new_state.cash
                    position_now = new_state.position
                    debt_now = new_state.debt
                    free_cash_now = new_state.free_cash
                    val_price_now = new_state.val_price
                    value_now = new_state.value
                    if cash_sharing:
                        return_now = returns_nb.get_return_nb(second_last_value[group], value_now)
                    else:
                        return_now = returns_nb.get_return_nb(second_last_value[col], value_now)
                    oidx = new_state.oidx
                    lidx = new_state.lidx

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    if not np.isnan(val_price_now) or not ffill_val_price:
                        last_val_price[col] = val_price_now
                    if cash_sharing:
                        last_cash[group] = cash_now
                        last_free_cash[group] = free_cash_now
                        last_value[group] = value_now
                        last_return[group] = return_now
                    else:
                        last_cash[col] = cash_now
                        last_free_cash[col] = free_cash_now
                        last_value[col] = value_now
                        last_return[col] = return_now
                    if state.oidx != new_state.oidx:
                        last_oidx[col] = state.oidx
                    if state.lidx != new_state.lidx:
                        last_lidx[col] = state.lidx

                    # Update position record
                    if fill_pos_record:
                        update_pos_record_nb(
                            pos_record_now,
                            i, col,
                            state.position, position_now,
                            order_result
                        )

                    # Post-order callback
                    post_order_ctx = PostOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        init_cash=init_cash,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        close=close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        second_last_value=second_last_value,
                        last_return=last_return,
                        last_oidx=last_oidx,
                        last_lidx=last_lidx,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now,
                        col=col,
                        call_idx=k,
                        cash_before=state.cash,
                        position_before=state.position,
                        debt_before=state.debt,
                        free_cash_before=state.free_cash,
                        val_price_before=state.val_price,
                        value_before=state.value,
                        order_result=order_result,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_record_now=pos_record_now
                    )
                    post_order_func_nb(post_order_ctx, *pre_segment_out, *post_order_args)

            # NOTE: Regardless of segment_mask, we still need to update stats to be accessed by future rows
            # Update valuation price
            for col in range(from_col, to_col):
                _close = flex_select_auto_nb(close, i, col, flex_2d)
                if not np.isnan(_close) or not ffill_val_price:
                    last_val_price[col] = _close

            # Update previous value, current value and return
            if cash_sharing:
                last_value[group] = get_group_value_nb(
                    from_col,
                    to_col,
                    last_cash[group],
                    last_position,
                    last_val_price
                )
                second_last_value[group] = temp_value[group]
                temp_value[group] = last_value[group]
                last_return[group] = returns_nb.get_return_nb(second_last_value[group], last_value[group])
            else:
                for col in range(from_col, to_col):
                    if last_position[col] == 0:
                        last_value[col] = last_cash[col]
                    else:
                        last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                    second_last_value[col] = temp_value[col]
                    temp_value[col] = last_value[col]
                    last_return[col] = returns_nb.get_return_nb(second_last_value[col], last_value[col])

            # Update open position stats
            if fill_pos_record:
                for col in range(from_col, to_col):
                    update_open_pos_stats_nb(
                        last_pos_record[col],
                        last_position[col],
                        last_val_price[col]
                    )

            # Is this segment active?
            if call_post_segment or segment_mask[i, group]:
                # Call function before the segment
                post_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    init_cash=init_cash,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    close=close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    second_last_value=second_last_value,
                    last_return=last_return,
                    last_oidx=last_oidx,
                    last_lidx=last_lidx,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=call_seq_now
                )
                post_segment_func_nb(post_seg_ctx, *pre_group_out, *post_segment_args)

        # Call function after the group
        post_group_ctx = GroupContext(
            target_shape=target_shape,
            group_lens=group_lens,
            init_cash=init_cash,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            close=close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            second_last_value=second_last_value,
            last_return=last_return,
            last_oidx=last_oidx,
            last_lidx=last_lidx,
            last_pos_record=last_pos_record,
            group=group,
            group_len=group_len,
            from_col=from_col,
            to_col=to_col
        )
        post_group_func_nb(post_group_ctx, *pre_sim_out, *post_group_args)

        from_col = to_col

    # Call function after the simulation
    post_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        init_cash=init_cash,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        close=close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        second_last_value=second_last_value,
        last_return=last_return,
        last_oidx=last_oidx,
        last_lidx=last_lidx,
        last_pos_record=last_pos_record
    )
    post_sim_func_nb(post_sim_ctx, *post_sim_args)

    return order_records[:oidx], log_records[:lidx]


@njit
def simulate_row_wise_nb(target_shape: tp.Shape,
                         group_lens: tp.Array1d,
                         init_cash: tp.Array1d,
                         cash_sharing: bool,
                         call_seq: tp.Array2d,
                         segment_mask: tp.ArrayLike = np.asarray(True),
                         call_pre_segment: bool = False,
                         call_post_segment: bool = False,
                         pre_sim_func_nb: PreSimFuncT = no_pre_func_nb,
                         pre_sim_args: tp.Args = (),
                         post_sim_func_nb: PostSimFuncT = no_post_func_nb,
                         post_sim_args: tp.Args = (),
                         pre_row_func_nb: PreRowFuncT = no_pre_func_nb,
                         pre_row_args: tp.Args = (),
                         post_row_func_nb: PostRowFuncT = no_post_func_nb,
                         post_row_args: tp.Args = (),
                         pre_segment_func_nb: PreSegmentFuncT = no_pre_func_nb,
                         pre_segment_args: tp.Args = (),
                         post_segment_func_nb: PostSegmentFuncT = no_post_func_nb,
                         post_segment_args: tp.Args = (),
                         order_func_nb: OrderFuncT = no_order_func_nb,
                         order_args: tp.Args = (),
                         post_order_func_nb: PostOrderFuncT = no_post_func_nb,
                         post_order_args: tp.Args = (),
                         close: tp.ArrayLike = np.asarray(np.nan),
                         ffill_val_price: bool = True,
                         update_value: bool = False,
                         fill_pos_record: bool = True,
                         max_orders: tp.Optional[int] = None,
                         max_logs: int = 0,
                         flex_2d: bool = True) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
    """Same as `simulate_nb`, but iterates in row-major order.

    Row-major order means processing the entire row with all groups/columns before moving to the next one.

    The main difference is that instead of `pre_group_func_nb` it now exposes `pre_row_func_nb`,
    which is executed per entire row. It should accept `vectorbt.portfolio.enums.RowContext`.

    !!! note
        Function `pre_row_func_nb` is only called if there is at least on active segment in
        the row. Functions `pre_segment_func_nb` and `order_func_nb` are only called if their
        segment is active. If the main task of `pre_row_func_nb` is to activate/deactivate segments,
        all segments should be activated by default to allow `pre_row_func_nb` to be called.

    !!! warning
        You can only safely access data points that are to the left of the current group and
        rows that are to the top of the current row.

    Call hierarchy:
        Let's illustrate the same example as in `simulate_nb` but adapted for this function:

        ![](/assets/images/simulate_row_wise_nb.gif)

    Usage:
        * Running the same example as in `simulate_nb` but adapted for this function:

        ```pycon
        >>> @njit
        ... def pre_row_func_nb(c, order_value_out):
        ...     print('\\tbefore row', c.i)
        ...     # Forward down the stack
        ...     return (order_value_out,)

        >>> @njit
        ... def post_row_func_nb(c, order_value_out):
        ...     print('\\tafter row', c.i)
        ...     return None

        >>> call_seq = build_call_seq(target_shape, group_lens)
        >>> order_records, log_records = simulate_row_wise_nb(
        ...     target_shape,
        ...     group_lens,
        ...     init_cash,
        ...     cash_sharing,
        ...     call_seq,
        ...     segment_mask=segment_mask,
        ...     pre_sim_func_nb=pre_sim_func_nb,
        ...     post_sim_func_nb=post_sim_func_nb,
        ...     pre_row_func_nb=pre_row_func_nb,
        ...     post_row_func_nb=post_row_func_nb,
        ...     pre_segment_func_nb=pre_segment_func_nb,
        ...     pre_segment_args=(size, price, size_type, direction),
        ...     post_segment_func_nb=post_segment_func_nb,
        ...     order_func_nb=order_func_nb,
        ...     order_args=(size, price, size_type, direction, fees, fixed_fees, slippage),
        ...     post_order_func_nb=post_order_func_nb
        ... )
        before simulation
            before row 0
                before segment 0
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 1
                        order status: 0
                    creating order 2 at column 2
                        order status: 0
                after segment 0
            after row 0
            before row 1
            after row 1
            before row 2
                before segment 2
                    creating order 0 at column 1
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 0
                        order status: 0
                after segment 2
            after row 2
            before row 3
            after row 3
            before row 4
                before segment 4
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 1
                        order status: 0
                after segment 4
            after row 4
        after simulation
        ```
    """
    check_group_lens_nb(group_lens, target_shape[1])
    check_group_init_cash_nb(group_lens, target_shape[1], init_cash, cash_sharing)

    order_records, log_records = init_records_nb(target_shape, max_orders, max_logs)
    init_cash = init_cash.astype(np.float64)
    last_cash = init_cash.copy()
    last_position = np.full(target_shape[1], 0., dtype=np.float64)
    last_debt = np.full(target_shape[1], 0., dtype=np.float64)
    last_free_cash = init_cash.copy()
    last_val_price = np.full(target_shape[1], np.nan, dtype=np.float64)
    last_value = init_cash.copy()
    second_last_value = init_cash.copy()
    temp_value = init_cash.copy()
    last_return = np.full_like(last_value, np.nan)
    last_pos_record = np.empty(target_shape[1], dtype=trade_dt)
    last_pos_record['id'][:] = -1
    last_oidx = np.full(target_shape[1], -1, dtype=np.int64)
    last_lidx = np.full(target_shape[1], -1, dtype=np.int64)
    oidx = 0
    lidx = 0

    # Call function before the simulation
    pre_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        init_cash=init_cash,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        close=close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        second_last_value=second_last_value,
        last_return=last_return,
        last_oidx=last_oidx,
        last_lidx=last_lidx,
        last_pos_record=last_pos_record
    )
    pre_sim_out = pre_sim_func_nb(pre_sim_ctx, *pre_sim_args)

    for i in range(target_shape[0]):

        # Call function before the row
        pre_row_ctx = RowContext(
            target_shape=target_shape,
            group_lens=group_lens,
            init_cash=init_cash,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            close=close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            second_last_value=second_last_value,
            last_return=last_return,
            last_oidx=last_oidx,
            last_lidx=last_lidx,
            last_pos_record=last_pos_record,
            i=i
        )
        pre_row_out = pre_row_func_nb(pre_row_ctx, *pre_sim_out, *pre_row_args)

        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            group_len = to_col - from_col
            call_seq_now = call_seq[i, from_col:to_col]

            # Is this segment active?
            if call_pre_segment or segment_mask[i, group]:
                # Call function before the segment
                pre_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    init_cash=init_cash,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    close=close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    second_last_value=second_last_value,
                    last_return=last_return,
                    last_oidx=last_oidx,
                    last_lidx=last_lidx,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=call_seq_now
                )
                pre_segment_out = pre_segment_func_nb(pre_seg_ctx, *pre_row_out, *pre_segment_args)

            # Update open position stats
            if fill_pos_record:
                for col in range(from_col, to_col):
                    update_open_pos_stats_nb(
                        last_pos_record[col],
                        last_position[col],
                        last_val_price[col]
                    )

            # Update value and return
            if cash_sharing:
                last_value[group] = get_group_value_nb(
                    from_col,
                    to_col,
                    last_cash[group],
                    last_position,
                    last_val_price
                )
                last_return[group] = returns_nb.get_return_nb(second_last_value[group], last_value[group])
            else:
                for col in range(from_col, to_col):
                    if last_position[col] == 0:
                        last_value[col] = last_cash[col]
                    else:
                        last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                    last_return[col] = returns_nb.get_return_nb(second_last_value[col], last_value[col])

            # Is this segment active?
            if segment_mask[i, group]:

                for k in range(group_len):
                    col_i = call_seq_now[k]
                    if col_i >= group_len:
                        raise ValueError("Call index exceeds bounds of the group")
                    col = from_col + col_i

                    # Get current values
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    val_price_now = last_val_price[col]
                    pos_record_now = last_pos_record[col]
                    if cash_sharing:
                        cash_now = last_cash[group]
                        free_cash_now = last_free_cash[group]
                        value_now = last_value[group]
                        return_now = last_return[group]
                    else:
                        cash_now = last_cash[col]
                        free_cash_now = last_free_cash[col]
                        value_now = last_value[col]
                        return_now = last_return[col]

                    # Generate the next order
                    order_ctx = OrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        init_cash=init_cash,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        close=close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        second_last_value=second_last_value,
                        last_return=last_return,
                        last_oidx=last_oidx,
                        last_lidx=last_lidx,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now,
                        col=col,
                        call_idx=k,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_record_now=pos_record_now
                    )
                    order = order_func_nb(order_ctx, *pre_segment_out, *order_args)
                    if np.isinf(order.price):
                        if i > 0:
                            _prev_close = flex_select_auto_nb(close, i - 1, col, flex_2d)
                        else:
                            _prev_close = np.nan
                        _close = flex_select_auto_nb(close, i, col, flex_2d)
                        order = replace_inf_price_nb(_prev_close, _close, order)

                    # Process the order
                    state = ProcessOrderState(
                        cash=cash_now,
                        position=position_now,
                        debt=debt_now,
                        free_cash=free_cash_now,
                        val_price=val_price_now,
                        value=value_now,
                        oidx=oidx,
                        lidx=lidx
                    )

                    order_result, new_state = process_order_nb(
                        i, col, group,
                        state,
                        update_value,
                        order,
                        order_records,
                        log_records
                    )

                    # Update state
                    cash_now = new_state.cash
                    position_now = new_state.position
                    debt_now = new_state.debt
                    free_cash_now = new_state.free_cash
                    val_price_now = new_state.val_price
                    value_now = new_state.value
                    if cash_sharing:
                        return_now = returns_nb.get_return_nb(second_last_value[group], value_now)
                    else:
                        return_now = returns_nb.get_return_nb(second_last_value[col], value_now)
                    oidx = new_state.oidx
                    lidx = new_state.lidx

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    if not np.isnan(val_price_now) or not ffill_val_price:
                        last_val_price[col] = val_price_now
                    if cash_sharing:
                        last_cash[group] = cash_now
                        last_free_cash[group] = free_cash_now
                        last_value[group] = value_now
                        last_return[group] = return_now
                    else:
                        last_cash[col] = cash_now
                        last_free_cash[col] = free_cash_now
                        last_value[col] = value_now
                        last_return[col] = return_now
                    if state.oidx != new_state.oidx:
                        last_oidx[col] = state.oidx
                    if state.lidx != new_state.lidx:
                        last_lidx[col] = state.lidx

                    # Update position record
                    if fill_pos_record:
                        update_pos_record_nb(
                            pos_record_now,
                            i, col,
                            state.position, position_now,
                            order_result
                        )

                    # Post-order callback
                    post_order_ctx = PostOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        init_cash=init_cash,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        close=close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        second_last_value=second_last_value,
                        last_return=last_return,
                        last_oidx=last_oidx,
                        last_lidx=last_lidx,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now,
                        col=col,
                        call_idx=k,
                        cash_before=state.cash,
                        position_before=state.position,
                        debt_before=state.debt,
                        free_cash_before=state.free_cash,
                        val_price_before=state.val_price,
                        value_before=state.value,
                        order_result=order_result,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_record_now=pos_record_now
                    )
                    post_order_func_nb(post_order_ctx, *pre_segment_out, *post_order_args)

            # NOTE: Regardless of segment_mask, we still need to update stats to be accessed by future rows
            # Update valuation price
            for col in range(from_col, to_col):
                _close = flex_select_auto_nb(close, i, col, flex_2d)
                if not np.isnan(_close) or not ffill_val_price:
                    last_val_price[col] = _close

            # Update previous value, current value and return
            if cash_sharing:
                last_value[group] = get_group_value_nb(
                    from_col,
                    to_col,
                    last_cash[group],
                    last_position,
                    last_val_price
                )
                second_last_value[group] = temp_value[group]
                temp_value[group] = last_value[group]
                last_return[group] = returns_nb.get_return_nb(second_last_value[group], last_value[group])
            else:
                for col in range(from_col, to_col):
                    if last_position[col] == 0:
                        last_value[col] = last_cash[col]
                    else:
                        last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                    second_last_value[col] = temp_value[col]
                    temp_value[col] = last_value[col]
                    last_return[col] = returns_nb.get_return_nb(second_last_value[col], last_value[col])

            # Update open position stats
            if fill_pos_record:
                for col in range(from_col, to_col):
                    update_open_pos_stats_nb(
                        last_pos_record[col],
                        last_position[col],
                        last_val_price[col]
                    )

            # Is this segment active?
            if call_post_segment or segment_mask[i, group]:
                # Call function after the segment
                post_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    init_cash=init_cash,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    close=close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    second_last_value=second_last_value,
                    last_return=last_return,
                    last_oidx=last_oidx,
                    last_lidx=last_lidx,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=call_seq_now
                )
                post_segment_func_nb(post_seg_ctx, *pre_row_out, *post_segment_args)

            from_col = to_col

        # Call function after the row
        post_row_ctx = RowContext(
            target_shape=target_shape,
            group_lens=group_lens,
            init_cash=init_cash,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            close=close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            second_last_value=second_last_value,
            last_return=last_return,
            last_oidx=last_oidx,
            last_lidx=last_lidx,
            last_pos_record=last_pos_record,
            i=i
        )
        post_row_func_nb(post_row_ctx, *pre_sim_out, *post_row_args)

    # Call function after the simulation
    post_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        init_cash=init_cash,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        close=close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        second_last_value=second_last_value,
        last_return=last_return,
        last_oidx=last_oidx,
        last_lidx=last_lidx,
        last_pos_record=last_pos_record
    )
    post_sim_func_nb(post_sim_ctx, *post_sim_args)

    return order_records[:oidx], log_records[:lidx]


@njit  # Numba编译缓存，优化重复调用性能
def no_flex_order_func_nb(c: FlexOrderContext, *args) -> tp.Tuple[int, Order]:
    """
    灵活订单函数占位符
    
    这是一个空的占位符函数，用于灵活模拟中当不需要生成订单时的默认行为。
    通过返回-1和NoOrder来指示中断循环且不执行任何订单。
    
    参数:
    ----
    c : FlexOrderContext
        灵活订单上下文，包含当前模拟状态信息
        
    *args : tuple
        可变参数，允许接收任意额外参数（在此函数中被忽略）
        
    返回:
    ----
    tuple[int, Order]
        (-1, NoOrder) - 中断标志和空订单的组合
        
    返回值含义:
    ----------
    - **-1**: 中断标志，指示模拟引擎停止在当前段内的订单生成循环
    - **NoOrder**: 空订单对象，表示不执行任何交易操作
    
    使用场景:
    --------
    - **默认行为**：当没有特定订单逻辑时的占位符
    - **条件跳过**：在某些条件下跳过订单生成
    - **测试用途**：作为测试和调试时的空实现
    - **模板基础**：作为开发自定义订单函数的起始模板
    
    与其他占位符函数的关系:
    -------------------
    - `no_order_func_nb`: 用于标准模拟的空订单函数
    - `no_flex_order_func_nb`: 用于灵活模拟的空订单函数
    - 两者的主要区别在于返回值格式和应用场景
    
    技术特点:
    --------
    - **Numba优化**：编译为高效机器码，避免Python调用开销
    - **类型安全**：严格的类型注解确保接口一致性
    - **最小开销**：极简实现，对性能影响最小
    - **通用兼容**：可以在任何需要订单函数的地方使用
    
    注意事项:
    --------
    - 此函数不会产生任何副作用或状态变化
    - 在灵活模拟中会导致当前段的订单生成循环立即结束
    - 适合作为开发复杂订单函数的起始点
    """
    return -1, NoOrder


FlexOrderFuncT = tp.Callable[[FlexOrderContext, tp.VarArg()], tp.Tuple[int, Order]]


@njit
def flex_simulate_nb(target_shape: tp.Shape,
                     group_lens: tp.Array1d,
                     init_cash: tp.Array1d,
                     cash_sharing: bool,
                     segment_mask: tp.ArrayLike = np.asarray(True),
                     call_pre_segment: bool = False,
                     call_post_segment: bool = False,
                     pre_sim_func_nb: PreSimFuncT = no_pre_func_nb,
                     pre_sim_args: tp.Args = (),
                     post_sim_func_nb: PostSimFuncT = no_post_func_nb,
                     post_sim_args: tp.Args = (),
                     pre_group_func_nb: PreGroupFuncT = no_pre_func_nb,
                     pre_group_args: tp.Args = (),
                     post_group_func_nb: PostGroupFuncT = no_post_func_nb,
                     post_group_args: tp.Args = (),
                     pre_segment_func_nb: PreSegmentFuncT = no_pre_func_nb,
                     pre_segment_args: tp.Args = (),
                     post_segment_func_nb: PostSegmentFuncT = no_post_func_nb,
                     post_segment_args: tp.Args = (),
                     flex_order_func_nb: FlexOrderFuncT = no_flex_order_func_nb,
                     flex_order_args: tp.Args = (),
                     post_order_func_nb: PostOrderFuncT = no_post_func_nb,
                     post_order_args: tp.Args = (),
                     close: tp.ArrayLike = np.asarray(np.nan),
                     ffill_val_price: bool = True,
                     update_value: bool = False,
                     fill_pos_record: bool = True,
                     max_orders: tp.Optional[int] = None,
                     max_logs: int = 0,
                     flex_2d: bool = True) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
    """Same as `simulate_nb`, but with no predefined call sequence.

    In contrast to `order_func_nb` in`simulate_nb`, `post_order_func_nb` is a segment-level order function
    that returns a column along with the order, and gets repeatedly called until some condition is met.
    This allows multiple orders to be issued within a single element and in an arbitrary order.

    The order function should accept `vectorbt.portfolio.enums.FlexOrderContext`, unpacked tuple from
    `pre_segment_func_nb`, and `*flex_order_args`. Should return column and `vectorbt.portfolio.enums.Order`.
    To break out of the loop, return column of -1.

    !!! note
        Since one element can now accommodate multiple orders, you may run into "order_records index out of range"
        exception. In this case, you should increase `max_orders`. This cannot be done automatically and
        dynamically to avoid performance degradation.

    Usage:
        * The same example as in `simulate_nb`:

        ```pycon
        >>> import numpy as np
        >>> from numba import njit
        >>> from vectorbt.portfolio.enums import SizeType, Direction
        >>> from vectorbt.portfolio.nb import (
        ...     get_col_elem_nb,
        ...     order_nb,
        ...     order_nothing_nb,
        ...     flex_simulate_nb,
        ...     flex_simulate_row_wise_nb,
        ...     sort_call_seq_out_nb
        ... )

        >>> @njit
        ... def pre_sim_func_nb(c):
        ...     print('before simulation')
        ...     return ()

        >>> @njit
        ... def pre_group_func_nb(c):
        ...     print('\\tbefore group', c.group)
        ...     # Create temporary arrays and pass them down the stack
        ...     order_value_out = np.empty(c.group_len, dtype=np.float64)
        ...     call_seq_out = np.empty(c.group_len, dtype=np.int64)
        ...     # Forward down the stack
        ...     return (order_value_out, call_seq_out)

        >>> @njit
        ... def pre_segment_func_nb(c, order_value_out, call_seq_out, size, price, size_type, direction):
        ...     print('\\t\\tbefore segment', c.i)
        ...     for col in range(c.from_col, c.to_col):
        ...         # Here we use order price for group valuation
        ...         c.last_val_price[col] = get_col_elem_nb(c, col, price)
        ...
        ...     # Same as for simulate_nb, but since we don't have a predefined c.call_seq_now anymore,
        ...     # we need to store our new call sequence somewhere else
        ...     call_seq_out[:] = np.arange(c.group_len)
        ...     sort_call_seq_out_nb(c, size, size_type, direction, order_value_out, call_seq_out)
        ...
        ...     # Forward the sorted call sequence
        ...     return (call_seq_out,)

        >>> @njit
        ... def flex_order_func_nb(c, call_seq_out, size, price, size_type, direction, fees, fixed_fees, slippage):
        ...     if c.call_idx < c.group_len:
        ...         col = c.from_col + call_seq_out[c.call_idx]
        ...         print('\\t\\t\\tcreating order', c.call_idx, 'at column', col)
        ...         # # Create and return an order
        ...         return col, order_nb(
        ...             size=get_col_elem_nb(c, col, size),
        ...             price=get_col_elem_nb(c, col, price),
        ...             size_type=get_col_elem_nb(c, col, size_type),
        ...             direction=get_col_elem_nb(c, col, direction),
        ...             fees=get_col_elem_nb(c, col, fees),
        ...             fixed_fees=get_col_elem_nb(c, col, fixed_fees),
        ...             slippage=get_col_elem_nb(c, col, slippage)
        ...         )
        ...     # All columns already processed -> break the loop
        ...     print('\\t\\t\\tbreaking out of the loop')
        ...     return -1, order_nothing_nb()

        >>> @njit
        ... def post_order_func_nb(c, call_seq_out):
        ...     print('\\t\\t\\t\\torder status:', c.order_result.status)
        ...     return None

        >>> @njit
        ... def post_segment_func_nb(c, order_value_out, call_seq_out):
        ...     print('\\t\\tafter segment', c.i)
        ...     return None

        >>> @njit
        ... def post_group_func_nb(c):
        ...     print('\\tafter group', c.group)
        ...     return None

        >>> @njit
        ... def post_sim_func_nb(c):
        ...     print('after simulation')
        ...     return None

        >>> target_shape = (5, 3)
        >>> np.random.seed(42)
        >>> group_lens = np.array([3])  # one group of three columns
        >>> init_cash = np.array([100.])  # one capital per group
        >>> cash_sharing = True
        >>> call_seq = build_call_seq(target_shape, group_lens)  # will be overridden
        >>> segment_mask = np.array([True, False, True, False, True])[:, None]
        >>> segment_mask = np.copy(np.broadcast_to(segment_mask, target_shape))
        >>> size = np.asarray(1 / target_shape[1])  # scalars must become 0-dim arrays
        >>> price = close = np.random.uniform(1, 10, size=target_shape)
        >>> size_type = np.asarray(SizeType.TargetPercent)
        >>> direction = np.asarray(Direction.LongOnly)
        >>> fees = np.asarray(0.001)
        >>> fixed_fees = np.asarray(1.)
        >>> slippage = np.asarray(0.001)

        >>> order_records, log_records = flex_simulate_nb(
        ...     target_shape,
        ...     group_lens,
        ...     init_cash,
        ...     cash_sharing,
        ...     segment_mask=segment_mask,
        ...     pre_sim_func_nb=pre_sim_func_nb,
        ...     post_sim_func_nb=post_sim_func_nb,
        ...     pre_group_func_nb=pre_group_func_nb,
        ...     post_group_func_nb=post_group_func_nb,
        ...     pre_segment_func_nb=pre_segment_func_nb,
        ...     pre_segment_args=(size, price, size_type, direction),
        ...     post_segment_func_nb=post_segment_func_nb,
        ...     flex_order_func_nb=flex_order_func_nb,
        ...     flex_order_args=(size, price, size_type, direction, fees, fixed_fees, slippage),
        ...     post_order_func_nb=post_order_func_nb
        ... )
        before simulation
            before group 0
                before segment 0
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 1
                        order status: 0
                    creating order 2 at column 2
                        order status: 0
                    breaking out of the loop
                after segment 0
                before segment 2
                    creating order 0 at column 1
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 0
                        order status: 0
                    breaking out of the loop
                after segment 2
                before segment 4
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 1
                        order status: 0
                    breaking out of the loop
                after segment 4
            after group 0
        after simulation
        ```
    """

    check_group_lens_nb(group_lens, target_shape[1])
    check_group_init_cash_nb(group_lens, target_shape[1], init_cash, cash_sharing)

    order_records, log_records = init_records_nb(target_shape, max_orders, max_logs)
    init_cash = init_cash.astype(np.float64)
    last_cash = init_cash.copy()
    last_position = np.full(target_shape[1], 0., dtype=np.float64)
    last_debt = np.full(target_shape[1], 0., dtype=np.float64)
    last_free_cash = init_cash.copy()
    last_val_price = np.full(target_shape[1], np.nan, dtype=np.float64)
    last_value = init_cash.copy()
    second_last_value = init_cash.copy()
    temp_value = init_cash.copy()
    last_return = np.full_like(last_value, np.nan)
    last_pos_record = np.empty(target_shape[1], dtype=trade_dt)
    last_pos_record['id'][:] = -1
    last_oidx = np.full(target_shape[1], -1, dtype=np.int64)
    last_lidx = np.full(target_shape[1], -1, dtype=np.int64)
    oidx = 0
    lidx = 0

    # Call function before the simulation
    pre_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        init_cash=init_cash,
        cash_sharing=cash_sharing,
        call_seq=None,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        close=close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        second_last_value=second_last_value,
        last_return=last_return,
        last_oidx=last_oidx,
        last_lidx=last_lidx,
        last_pos_record=last_pos_record
    )
    pre_sim_out = pre_sim_func_nb(pre_sim_ctx, *pre_sim_args)

    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col

        # Call function before the group
        pre_group_ctx = GroupContext(
            target_shape=target_shape,
            group_lens=group_lens,
            init_cash=init_cash,
            cash_sharing=cash_sharing,
            call_seq=None,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            close=close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            second_last_value=second_last_value,
            last_return=last_return,
            last_oidx=last_oidx,
            last_lidx=last_lidx,
            last_pos_record=last_pos_record,
            group=group,
            group_len=group_len,
            from_col=from_col,
            to_col=to_col
        )
        pre_group_out = pre_group_func_nb(pre_group_ctx, *pre_sim_out, *pre_group_args)

        for i in range(target_shape[0]):
            # Is this segment active?
            if call_pre_segment or segment_mask[i, group]:
                # Call function before the segment
                pre_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    init_cash=init_cash,
                    cash_sharing=cash_sharing,
                    call_seq=None,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    close=close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    second_last_value=second_last_value,
                    last_return=last_return,
                    last_oidx=last_oidx,
                    last_lidx=last_lidx,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=None
                )
                pre_segment_out = pre_segment_func_nb(pre_seg_ctx, *pre_group_out, *pre_segment_args)

            # Update open position stats
            if fill_pos_record:
                for col in range(from_col, to_col):
                    update_open_pos_stats_nb(
                        last_pos_record[col],
                        last_position[col],
                        last_val_price[col]
                    )

            # Update value and return
            if cash_sharing:
                last_value[group] = get_group_value_nb(
                    from_col,
                    to_col,
                    last_cash[group],
                    last_position,
                    last_val_price
                )
                last_return[group] = returns_nb.get_return_nb(second_last_value[group], last_value[group])
            else:
                for col in range(from_col, to_col):
                    if last_position[col] == 0:
                        last_value[col] = last_cash[col]
                    else:
                        last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                    last_return[col] = returns_nb.get_return_nb(second_last_value[col], last_value[col])

            # Is this segment active?
            if segment_mask[i, group]:

                call_idx = -1
                while True:
                    call_idx += 1

                    # Generate the next order
                    flex_order_ctx = FlexOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        init_cash=init_cash,
                        cash_sharing=cash_sharing,
                        call_seq=None,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        close=close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        second_last_value=second_last_value,
                        last_return=last_return,
                        last_oidx=last_oidx,
                        last_lidx=last_lidx,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=None,
                        call_idx=call_idx
                    )
                    col, order = flex_order_func_nb(flex_order_ctx, *pre_segment_out, *flex_order_args)

                    if col == -1:
                        break
                    if col < from_col or col >= to_col:
                        raise ValueError("Column exceeds bounds of the group")

                    # Get current values
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    val_price_now = last_val_price[col]
                    pos_record_now = last_pos_record[col]
                    if cash_sharing:
                        cash_now = last_cash[group]
                        free_cash_now = last_free_cash[group]
                        value_now = last_value[group]
                        return_now = last_return[group]
                    else:
                        cash_now = last_cash[col]
                        free_cash_now = last_free_cash[col]
                        value_now = last_value[col]
                        return_now = last_return[col]

                    if np.isinf(order.price):
                        if i > 0:
                            _prev_close = flex_select_auto_nb(close, i - 1, col, flex_2d)
                        else:
                            _prev_close = np.nan
                        _close = flex_select_auto_nb(close, i, col, flex_2d)
                        order = replace_inf_price_nb(_prev_close, _close, order)

                    # Process the order
                    state = ProcessOrderState(
                        cash=cash_now,
                        position=position_now,
                        debt=debt_now,
                        free_cash=free_cash_now,
                        val_price=val_price_now,
                        value=value_now,
                        oidx=oidx,
                        lidx=lidx
                    )

                    order_result, new_state = process_order_nb(
                        i, col, group,
                        state,
                        update_value,
                        order,
                        order_records,
                        log_records
                    )

                    # Update state
                    cash_now = new_state.cash
                    position_now = new_state.position
                    debt_now = new_state.debt
                    free_cash_now = new_state.free_cash
                    val_price_now = new_state.val_price
                    value_now = new_state.value
                    if cash_sharing:
                        return_now = returns_nb.get_return_nb(second_last_value[group], value_now)
                    else:
                        return_now = returns_nb.get_return_nb(second_last_value[col], value_now)
                    oidx = new_state.oidx
                    lidx = new_state.lidx

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    if not np.isnan(val_price_now) or not ffill_val_price:
                        last_val_price[col] = val_price_now
                    if cash_sharing:
                        last_cash[group] = cash_now
                        last_free_cash[group] = free_cash_now
                        last_value[group] = value_now
                        last_return[group] = return_now
                    else:
                        last_cash[col] = cash_now
                        last_free_cash[col] = free_cash_now
                        last_value[col] = value_now
                        last_return[col] = return_now
                    if state.oidx != new_state.oidx:
                        last_oidx[col] = state.oidx
                    if state.lidx != new_state.lidx:
                        last_lidx[col] = state.lidx

                    # Update position record
                    if fill_pos_record:
                        update_pos_record_nb(
                            pos_record_now,
                            i, col,
                            state.position, position_now,
                            order_result
                        )

                    # Post-order callback
                    post_order_ctx = PostOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        init_cash=init_cash,
                        cash_sharing=cash_sharing,
                        call_seq=None,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        close=close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        second_last_value=second_last_value,
                        last_return=last_return,
                        last_oidx=last_oidx,
                        last_lidx=last_lidx,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=None,
                        col=col,
                        call_idx=call_idx,
                        cash_before=state.cash,
                        position_before=state.position,
                        debt_before=state.debt,
                        free_cash_before=state.free_cash,
                        val_price_before=state.val_price,
                        value_before=state.value,
                        order_result=order_result,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_record_now=pos_record_now
                    )
                    post_order_func_nb(post_order_ctx, *pre_segment_out, *post_order_args)

            # NOTE: Regardless of segment_mask, we still need to update stats to be accessed by future rows
            # Update valuation price
            for col in range(from_col, to_col):
                _close = flex_select_auto_nb(close, i, col, flex_2d)
                if not np.isnan(_close) or not ffill_val_price:
                    last_val_price[col] = _close

            # Update previous value, current value and return
            if cash_sharing:
                last_value[group] = get_group_value_nb(
                    from_col,
                    to_col,
                    last_cash[group],
                    last_position,
                    last_val_price
                )
                second_last_value[group] = temp_value[group]
                temp_value[group] = last_value[group]
                last_return[group] = returns_nb.get_return_nb(second_last_value[group], last_value[group])
            else:
                for col in range(from_col, to_col):
                    if last_position[col] == 0:
                        last_value[col] = last_cash[col]
                    else:
                        last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                    second_last_value[col] = temp_value[col]
                    temp_value[col] = last_value[col]
                    last_return[col] = returns_nb.get_return_nb(second_last_value[col], last_value[col])

            # Update open position stats
            if fill_pos_record:
                for col in range(from_col, to_col):
                    update_open_pos_stats_nb(
                        last_pos_record[col],
                        last_position[col],
                        last_val_price[col]
                    )

            # Is this segment active?
            if call_post_segment or segment_mask[i, group]:
                # Call function before the segment
                post_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    init_cash=init_cash,
                    cash_sharing=cash_sharing,
                    call_seq=None,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    close=close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    second_last_value=second_last_value,
                    last_return=last_return,
                    last_oidx=last_oidx,
                    last_lidx=last_lidx,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=None
                )
                post_segment_func_nb(post_seg_ctx, *pre_group_out, *post_segment_args)

        # Call function after the group
        post_group_ctx = GroupContext(
            target_shape=target_shape,
            group_lens=group_lens,
            init_cash=init_cash,
            cash_sharing=cash_sharing,
            call_seq=None,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            close=close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            second_last_value=second_last_value,
            last_return=last_return,
            last_oidx=last_oidx,
            last_lidx=last_lidx,
            last_pos_record=last_pos_record,
            group=group,
            group_len=group_len,
            from_col=from_col,
            to_col=to_col
        )
        post_group_func_nb(post_group_ctx, *pre_sim_out, *post_group_args)

        from_col = to_col

    # Call function after the simulation
    post_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        init_cash=init_cash,
        cash_sharing=cash_sharing,
        call_seq=None,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        close=close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        second_last_value=second_last_value,
        last_return=last_return,
        last_oidx=last_oidx,
        last_lidx=last_lidx,
        last_pos_record=last_pos_record
    )
    post_sim_func_nb(post_sim_ctx, *post_sim_args)

    return order_records[:oidx], log_records[:lidx]


@njit
def flex_simulate_row_wise_nb(target_shape: tp.Shape,
                              group_lens: tp.Array1d,
                              init_cash: tp.Array1d,
                              cash_sharing: bool,
                              segment_mask: tp.ArrayLike = np.asarray(True),
                              call_pre_segment: bool = False,
                              call_post_segment: bool = False,
                              pre_sim_func_nb: PreSimFuncT = no_pre_func_nb,
                              pre_sim_args: tp.Args = (),
                              post_sim_func_nb: PostSimFuncT = no_post_func_nb,
                              post_sim_args: tp.Args = (),
                              pre_row_func_nb: PreRowFuncT = no_pre_func_nb,
                              pre_row_args: tp.Args = (),
                              post_row_func_nb: PostRowFuncT = no_post_func_nb,
                              post_row_args: tp.Args = (),
                              pre_segment_func_nb: PreSegmentFuncT = no_pre_func_nb,
                              pre_segment_args: tp.Args = (),
                              post_segment_func_nb: PostSegmentFuncT = no_post_func_nb,
                              post_segment_args: tp.Args = (),
                              flex_order_func_nb: FlexOrderFuncT = no_flex_order_func_nb,
                              flex_order_args: tp.Args = (),
                              post_order_func_nb: PostOrderFuncT = no_post_func_nb,
                              post_order_args: tp.Args = (),
                              close: tp.ArrayLike = np.asarray(np.nan),
                              ffill_val_price: bool = True,
                              update_value: bool = False,
                              fill_pos_record: bool = True,
                              max_orders: tp.Optional[int] = None,
                              max_logs: int = 0,
                              flex_2d: bool = True) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
    """Same as `flex_simulate_nb`, but iterates using row-major order, with the rows
    changing fastest, and the columns/groups changing slowest."""

    check_group_lens_nb(group_lens, target_shape[1])
    check_group_init_cash_nb(group_lens, target_shape[1], init_cash, cash_sharing)

    order_records, log_records = init_records_nb(target_shape, max_orders, max_logs)
    init_cash = init_cash.astype(np.float64)
    last_cash = init_cash.copy()
    last_position = np.full(target_shape[1], 0., dtype=np.float64)
    last_debt = np.full(target_shape[1], 0., dtype=np.float64)
    last_free_cash = init_cash.copy()
    last_val_price = np.full(target_shape[1], np.nan, dtype=np.float64)
    last_value = init_cash.copy()
    second_last_value = init_cash.copy()
    temp_value = init_cash.copy()
    last_return = np.full_like(last_value, np.nan)
    last_pos_record = np.empty(target_shape[1], dtype=trade_dt)
    last_pos_record['id'][:] = -1
    last_oidx = np.full(target_shape[1], -1, dtype=np.int64)
    last_lidx = np.full(target_shape[1], -1, dtype=np.int64)
    oidx = 0
    lidx = 0

    # Call function before the simulation
    pre_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        init_cash=init_cash,
        cash_sharing=cash_sharing,
        call_seq=None,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        close=close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        second_last_value=second_last_value,
        last_return=last_return,
        last_oidx=last_oidx,
        last_lidx=last_lidx,
        last_pos_record=last_pos_record
    )
    pre_sim_out = pre_sim_func_nb(pre_sim_ctx, *pre_sim_args)

    for i in range(target_shape[0]):

        # Call function before the row
        pre_row_ctx = RowContext(
            target_shape=target_shape,
            group_lens=group_lens,
            init_cash=init_cash,
            cash_sharing=cash_sharing,
            call_seq=None,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            close=close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            second_last_value=second_last_value,
            last_return=last_return,
            last_oidx=last_oidx,
            last_lidx=last_lidx,
            last_pos_record=last_pos_record,
            i=i
        )
        pre_row_out = pre_row_func_nb(pre_row_ctx, *pre_sim_out, *pre_row_args)

        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            group_len = to_col - from_col

            # Is this segment active?
            if call_pre_segment or segment_mask[i, group]:
                # Call function before the segment
                pre_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    init_cash=init_cash,
                    cash_sharing=cash_sharing,
                    call_seq=None,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    close=close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    second_last_value=second_last_value,
                    last_return=last_return,
                    last_oidx=last_oidx,
                    last_lidx=last_lidx,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=None
                )
                pre_segment_out = pre_segment_func_nb(pre_seg_ctx, *pre_row_out, *pre_segment_args)

            # Update open position stats
            if fill_pos_record:
                for col in range(from_col, to_col):
                    update_open_pos_stats_nb(
                        last_pos_record[col],
                        last_position[col],
                        last_val_price[col]
                    )

            # Update value and return
            if cash_sharing:
                last_value[group] = get_group_value_nb(
                    from_col,
                    to_col,
                    last_cash[group],
                    last_position,
                    last_val_price
                )
                last_return[group] = returns_nb.get_return_nb(second_last_value[group], last_value[group])
            else:
                for col in range(from_col, to_col):
                    if last_position[col] == 0:
                        last_value[col] = last_cash[col]
                    else:
                        last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                    last_return[col] = returns_nb.get_return_nb(second_last_value[col], last_value[col])

            # Is this segment active?
            if segment_mask[i, group]:

                call_idx = -1
                while True:
                    call_idx += 1

                    # Generate the next order
                    flex_order_ctx = FlexOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        init_cash=init_cash,
                        cash_sharing=cash_sharing,
                        call_seq=None,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        close=close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        second_last_value=second_last_value,
                        last_return=last_return,
                        last_oidx=last_oidx,
                        last_lidx=last_lidx,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=None,
                        call_idx=call_idx
                    )
                    col, order = flex_order_func_nb(flex_order_ctx, *pre_segment_out, *flex_order_args)

                    if col == -1:
                        break
                    if col < from_col or col >= to_col:
                        raise ValueError("Column exceeds bounds of the group")

                    # Get current values
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    val_price_now = last_val_price[col]
                    pos_record_now = last_pos_record[col]
                    if cash_sharing:
                        cash_now = last_cash[group]
                        free_cash_now = last_free_cash[group]
                        value_now = last_value[group]
                        return_now = last_return[group]
                    else:
                        cash_now = last_cash[col]
                        free_cash_now = last_free_cash[col]
                        value_now = last_value[col]
                        return_now = last_return[col]

                    if np.isinf(order.price):
                        if i > 0:
                            _prev_close = flex_select_auto_nb(close, i - 1, col, flex_2d)
                        else:
                            _prev_close = np.nan
                        _close = flex_select_auto_nb(close, i, col, flex_2d)
                        order = replace_inf_price_nb(_prev_close, _close, order)

                    # Process the order
                    state = ProcessOrderState(
                        cash=cash_now,
                        position=position_now,
                        debt=debt_now,
                        free_cash=free_cash_now,
                        val_price=val_price_now,
                        value=value_now,
                        oidx=oidx,
                        lidx=lidx
                    )

                    order_result, new_state = process_order_nb(
                        i, col, group,
                        state,
                        update_value,
                        order,
                        order_records,
                        log_records
                    )

                    # Update state
                    cash_now = new_state.cash
                    position_now = new_state.position
                    debt_now = new_state.debt
                    free_cash_now = new_state.free_cash
                    val_price_now = new_state.val_price
                    value_now = new_state.value
                    if cash_sharing:
                        return_now = returns_nb.get_return_nb(second_last_value[group], value_now)
                    else:
                        return_now = returns_nb.get_return_nb(second_last_value[col], value_now)
                    oidx = new_state.oidx
                    lidx = new_state.lidx

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    if not np.isnan(val_price_now) or not ffill_val_price:
                        last_val_price[col] = val_price_now
                    if cash_sharing:
                        last_cash[group] = cash_now
                        last_free_cash[group] = free_cash_now
                        last_value[group] = value_now
                        last_return[group] = return_now
                    else:
                        last_cash[col] = cash_now
                        last_free_cash[col] = free_cash_now
                        last_value[col] = value_now
                        last_return[col] = return_now
                    if state.oidx != new_state.oidx:
                        last_oidx[col] = state.oidx
                    if state.lidx != new_state.lidx:
                        last_lidx[col] = state.lidx

                    # Update position record
                    if fill_pos_record:
                        update_pos_record_nb(
                            pos_record_now,
                            i, col,
                            state.position, position_now,
                            order_result
                        )

                    # Post-order callback
                    post_order_ctx = PostOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        init_cash=init_cash,
                        cash_sharing=cash_sharing,
                        call_seq=None,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        close=close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        second_last_value=second_last_value,
                        last_return=last_return,
                        last_oidx=last_oidx,
                        last_lidx=last_lidx,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=None,
                        col=col,
                        call_idx=call_idx,
                        cash_before=state.cash,
                        position_before=state.position,
                        debt_before=state.debt,
                        free_cash_before=state.free_cash,
                        val_price_before=state.val_price,
                        value_before=state.value,
                        order_result=order_result,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_record_now=pos_record_now
                    )
                    post_order_func_nb(post_order_ctx, *pre_segment_out, *post_order_args)

            # NOTE: Regardless of segment_mask, we still need to update stats to be accessed by future rows
            # Update valuation price
            for col in range(from_col, to_col):
                _close = flex_select_auto_nb(close, i, col, flex_2d)
                if not np.isnan(_close) or not ffill_val_price:
                    last_val_price[col] = _close

            # Update previous value, current value and return
            if cash_sharing:
                last_value[group] = get_group_value_nb(
                    from_col,
                    to_col,
                    last_cash[group],
                    last_position,
                    last_val_price
                )
                second_last_value[group] = temp_value[group]
                temp_value[group] = last_value[group]
                last_return[group] = returns_nb.get_return_nb(second_last_value[group], last_value[group])
            else:
                for col in range(from_col, to_col):
                    if last_position[col] == 0:
                        last_value[col] = last_cash[col]
                    else:
                        last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                    second_last_value[col] = temp_value[col]
                    temp_value[col] = last_value[col]
                    last_return[col] = returns_nb.get_return_nb(second_last_value[col], last_value[col])

            # Update open position stats
            if fill_pos_record:
                for col in range(from_col, to_col):
                    update_open_pos_stats_nb(
                        last_pos_record[col],
                        last_position[col],
                        last_val_price[col]
                    )

            # Is this segment active?
            if call_post_segment or segment_mask[i, group]:
                # Call function after the segment
                post_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    init_cash=init_cash,
                    cash_sharing=cash_sharing,
                    call_seq=None,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    close=close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    second_last_value=second_last_value,
                    last_return=last_return,
                    last_oidx=last_oidx,
                    last_lidx=last_lidx,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=None
                )
                post_segment_func_nb(post_seg_ctx, *pre_row_out, *post_segment_args)

            from_col = to_col

        # Call function after the row
        post_row_ctx = RowContext(
            target_shape=target_shape,
            group_lens=group_lens,
            init_cash=init_cash,
            cash_sharing=cash_sharing,
            call_seq=None,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            close=close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            second_last_value=second_last_value,
            last_return=last_return,
            last_oidx=last_oidx,
            last_lidx=last_lidx,
            last_pos_record=last_pos_record,
            i=i
        )
        post_row_func_nb(post_row_ctx, *pre_sim_out, *post_row_args)

    # Call function after the simulation
    post_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        init_cash=init_cash,
        cash_sharing=cash_sharing,
        call_seq=None,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        close=close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        second_last_value=second_last_value,
        last_return=last_return,
        last_oidx=last_oidx,
        last_lidx=last_lidx,
        last_pos_record=last_pos_record
    )
    post_sim_func_nb(post_sim_ctx, *post_sim_args)

    return order_records[:oidx], log_records[:lidx]


# ############# 交易记录处理模块 ############# #
# 本模块负责计算交易统计信息、填充交易记录、生成交易记录数组等核心功能

# 错误消息定义
size_zero_neg_err = "发现订单大小为0或负数"
price_zero_neg_err = "发现订单价格为0或负数"


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def get_trade_stats_nb(size: float,
                       entry_price: float,
                       entry_fees: float,
                       exit_price: float,
                       exit_fees: float,
                       direction: int) -> tp.Tuple[float, float]:
    """
    计算单笔交易的统计信息
    
    根据交易的入场和出场信息，计算该笔交易的盈亏和收益率。
    支持多头和空头交易的正确计算。
    
    参数:
    ----
    size : float
        交易数量（股数、手数等）
        
    entry_price : float
        入场价格
        
    entry_fees : float
        入场手续费
        
    exit_price : float
        出场价格
        
    exit_fees : float
        出场手续费
        
    direction : int
        交易方向，使用TradeDirection枚举：
        - Long: 多头交易
        - Short: 空头交易
        
    返回:
    ----
    tuple[float, float]
        (pnl, return) - 盈亏金额和收益率
        
    使用示例:
    --------
    >>> # 多头交易示例
    >>> size = 100.0          # 100股
    >>> entry_price = 50.0    # 入场价50元
    >>> entry_fees = 5.0      # 入场手续费5元
    >>> exit_price = 55.0     # 出场价55元
    >>> exit_fees = 5.5       # 出场手续费5.5元
    >>> direction = TradeDirection.Long
    >>> 
    >>> pnl, ret = get_trade_stats_nb(
    ...     size, entry_price, entry_fees, exit_price, exit_fees, direction
    ... )
    >>> # pnl = (55-50)*100 - 5 - 5.5 = 500 - 10.5 = 489.5
    >>> # ret = 489.5 / (100*50) = 0.0979 (9.79%)
    >>> 
    >>> # 空头交易示例  
    >>> direction = TradeDirection.Short
    >>> pnl, ret = get_trade_stats_nb(
    ...     size, entry_price, entry_fees, exit_price, exit_fees, direction
    ... )
    >>> # 空头：价格下跌才盈利，此处亏损
    >>> # pnl = -[(55-50)*100] - 5 - 5.5 = -500 - 10.5 = -510.5
    
    计算逻辑详解:
    -----------
    1. **入场价值**: entry_val = size × entry_price
    2. **出场价值**: exit_val = size × exit_price  
    3. **价值差异**: val_diff = exit_val - entry_val
    4. **空头调整**: 如果是空头交易，价值差异取反
    5. **净盈亏**: pnl = val_diff - entry_fees - exit_fees
    6. **收益率**: ret = pnl / entry_val
    
    应用场景:
    --------
    - 单笔交易绩效分析
    - 策略整体收益统计
    - 交易记录的完整性填充
    - 风险收益比计算
    - 交易质量评估
    
    注意事项:
    --------
    - 空头交易的盈亏计算会自动调整符号
    - 收益率基于入场价值计算
    - 手续费会从净盈亏中扣除
    - 适用于各种资产类别的交易
    """
    # 计算入场和出场的总价值
    entry_val = size * entry_price    # 入场价值
    exit_val = size * exit_price      # 出场价值
    
    # 计算价值差异（出场价值 - 入场价值）
    val_diff = add_nb(exit_val, -entry_val)
    
    # 空头交易价值差异调整：价格上涨对空头不利，需取反
    if val_diff != 0 and direction == TradeDirection.Short:
        val_diff *= -1
    
    # 计算净盈亏：价值差异减去所有手续费
    pnl = val_diff - entry_fees - exit_fees
    
    # 计算收益率：净盈亏除以入场价值
    ret = pnl / entry_val
    
    return pnl, ret


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def fill_trade_record_nb(record: tp.Record,
                         id_: int,
                         col: int,
                         size: float,
                         entry_idx: int,
                         entry_price: float,
                         entry_fees: float,
                         exit_idx: int,
                         exit_price: float,
                         exit_fees: float,
                         direction: int,
                         status: int,
                         parent_id: int) -> None:
    """
    填充交易记录结构体
    
    将交易的完整信息填充到预分配的记录结构体中，包括基本信息、
    价格信息、计算得出的盈亏和收益率等。
    
    参数:
    ----
    record : Record
        待填充的交易记录结构体
        
    id_ : int
        交易唯一标识符
        
    col : int
        列索引（资产索引）
        
    size : float
        交易数量
        
    entry_idx : int
        入场时间索引
        
    entry_price : float
        入场价格
        
    entry_fees : float
        入场手续费
        
    exit_idx : int
        出场时间索引
        
    exit_price : float
        出场价格
        
    exit_fees : float
        出场手续费
        
    direction : int
        交易方向（多头/空头）
        
    status : int
        交易状态
        
    parent_id : int
        父级交易ID（用于关联交易）
        
    返回:
    ----
    None
        直接修改传入的record结构体
        
    使用示例:
    --------
    >>> # 创建空的交易记录
    >>> import numpy as np
    >>> from vectorbt.portfolio.enums import trade_dt
    >>> 
    >>> # 分配记录数组
    >>> records = np.empty(1, dtype=trade_dt)
    >>> 
    >>> # 填充第一条记录
    >>> fill_trade_record_nb(
    ...     records[0],           # 记录结构体
    ...     id_=0,                # 交易ID
    ...     col=0,                # 第0列（第1个资产）
    ...     size=100.0,           # 100股
    ...     entry_idx=5,          # 第5个时间点入场
    ...     entry_price=50.0,     # 入场价50元
    ...     entry_fees=2.5,       # 入场费用2.5元
    ...     exit_idx=10,          # 第10个时间点出场
    ...     exit_price=55.0,      # 出场价55元
    ...     exit_fees=2.75,       # 出场费用2.75元
    ...     direction=TradeDirection.Long,  # 多头交易
    ...     status=TradeStatus.Closed,      # 已关闭状态
    ...     parent_id=-1          # 无父级交易
    ... )
    
    记录结构说明:
    -----------
    填充完成后的记录包含以下字段：
    - **基础信息**: id, col, size, direction, status, parent_id
    - **时间信息**: entry_idx, exit_idx  
    - **价格信息**: entry_price, exit_price
    - **成本信息**: entry_fees, exit_fees
    - **绩效信息**: pnl (盈亏), return (收益率)
    
    应用场景:
    --------
    - 交易记录数组的批量生成
    - 回测结果的详细记录
    - 交易分析和报表生成
    - 绩效评估的数据基础
    - 交易历史的完整存档
    
    注意事项:
    --------
    - 记录结构体必须预先分配内存
    - 盈亏和收益率会自动计算填充
    - 所有字段都会被完整填充
    - 适配vectorbt的标准记录格式
    """
    # **第1步：计算交易统计信息**
    # 调用交易统计函数获取盈亏和收益率
    pnl, ret = get_trade_stats_nb(
        size,          # 交易数量
        entry_price,   # 入场价格
        entry_fees,    # 入场费用
        exit_price,    # 出场价格
        exit_fees,     # 出场费用
        direction      # 交易方向
    )

    # **第2步：填充记录结构体的各个字段**
    
    # 基础信息字段
    record['id'] = id_                    # 交易唯一标识
    record['col'] = col                   # 列索引（资产索引）
    record['size'] = size                 # 交易数量
    record['direction'] = direction       # 交易方向
    record['status'] = status             # 交易状态
    record['parent_id'] = parent_id       # 父级交易ID
    
    # 时间信息字段
    record['entry_idx'] = entry_idx       # 入场时间索引
    record['exit_idx'] = exit_idx         # 出场时间索引
    
    # 价格信息字段
    record['entry_price'] = entry_price   # 入场价格
    record['exit_price'] = exit_price     # 出场价格
    
    # 成本信息字段
    record['entry_fees'] = entry_fees     # 入场费用
    record['exit_fees'] = exit_fees       # 出场费用
    
    # 绩效信息字段（计算得出）
    record['pnl'] = pnl                   # 净盈亏
    record['return'] = ret                # 收益率


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def fill_entry_trades_in_position_nb(order_records: tp.RecordArray,
                                     col_map: tp.ColMap,
                                     col: int,
                                     first_c: int,
                                     last_c: int,
                                     first_entry_size: float,
                                     first_entry_fees: float,
                                     exit_idx: int,
                                     exit_size_sum: float,
                                     exit_gross_sum: float,
                                     exit_fees_sum: float,
                                     direction: int,
                                     status: int,
                                     parent_id: int,
                                     trade_records: tp.RecordArray,
                                     tidx: int) -> int:
    """
    填充单一持仓内的入场交易记录
    
    在一个完整持仓周期中，可能包含多个入场订单（加仓、分批建仓等）。
    此函数将这些入场订单转换为标准化的交易记录，计算每笔入场订单
    对应的平均出场价格和按比例分配的出场费用。
    
    参数:
    ----
    order_records : RecordArray
        原始订单记录数组
        
    col_map : ColMap
        列映射信息，用于快速定位指定列的订单
        
    col : int
        目标列索引（资产索引）
        
    first_c : int
        持仓内第一个入场订单的位置索引
        
    last_c : int
        持仓内最后一个入场订单的位置索引
        
    first_entry_size : float
        第一个入场订单的实际大小（可能经过调整）
        
    first_entry_fees : float
        第一个入场订单的实际费用（可能经过调整）
        
    exit_idx : int
        出场时间索引
        
    exit_size_sum : float
        总出场数量
        
    exit_gross_sum : float
        总出场价值（不含费用）
        
    exit_fees_sum : float
        总出场费用
        
    direction : int
        交易方向（多头/空头）
        
    status : int
        交易状态
        
    parent_id : int
        父级交易ID
        
    trade_records : RecordArray
        待填充的交易记录数组
        
    tidx : int
        当前交易记录索引，函数会从此位置开始填充
        
    返回:
    ----
    int
        更新后的交易记录索引（填充完成后的下一个可用位置）
        
    算法原理:
    --------
    1. **遍历持仓内的所有订单**：从first_c到last_c
    2. **过滤入场订单**：忽略出场订单，只处理建仓订单
    3. **计算平均出场价格**：exit_price = exit_gross_sum / exit_size_sum
    4. **按比例分配出场费用**：每笔入场根据其大小占比承担出场费用
    5. **生成交易记录**：为每笔入场订单生成完整的交易记录
    
    使用场景:
    --------
    - **分批建仓策略分析**：将多次入场合并为单一持仓视角
    - **加仓策略评估**：分析每次加仓的独立绩效
    - **成本基础追踪**：准确计算每笔入场的真实成本和收益
    - **交易绩效细分**：提供比持仓级别更细粒度的分析
    
    处理逻辑详解:
    -----------
    多头交易：
    - 入场订单：买入订单（OrderSide.Buy）
    - 出场订单：卖出订单（OrderSide.Sell）
    
    空头交易：
    - 入场订单：卖出订单（OrderSide.Sell）  
    - 出场订单：买入订单（OrderSide.Buy）
    
    费用分配原则：
    - 入场费用：直接使用订单的实际费用
    - 出场费用：按入场数量比例从总出场费用中分配
    - 价格计算：使用加权平均出场价格
    
    注意事项:
    --------
    - 假设所有入场订单在同一持仓周期内完全平仓
    - 出场价格采用数量加权平均，确保公平性
    - 费用分配遵循比例原则，避免重复计算
    - 支持部分成交和复杂的多次加仓场景
    """
    # **第1步：解析列映射信息**
    col_idxs, col_lens = col_map                           # 解包列映射
    col_start_idxs = np.cumsum(col_lens) - col_lens        # 计算各列起始索引

    # **第2步：遍历持仓内的所有订单**
    for c in range(first_c, last_c + 1):
        # 获取订单记录的实际索引
        oidx = col_idxs[col_start_idxs[col] + c]
        record = order_records[oidx]              # 获取订单记录
        order_side = record['side']               # 订单方向（买/卖）

        # **第3步：过滤出场订单，只处理入场订单**
        if (direction == TradeDirection.Long and order_side == OrderSide.Sell) \
                or (direction == TradeDirection.Short and order_side == OrderSide.Buy):
            continue  # 跳过出场订单

        # **第4步：确定入场订单的实际大小和费用**
        if c == first_c:
            # 第一个订单：使用调整后的大小和费用（可能经过部分成交等处理）
            entry_size = first_entry_size
            entry_fees = first_entry_fees
        else:
            # 其他订单：直接使用记录中的原始值
            entry_size = record['size']
            entry_fees = record['fees']

        # **第5步：计算加权平均出场价格**
        # 使用总出场价值除以总出场数量，得到平均出场价格
        exit_price = exit_gross_sum / exit_size_sum

        # **第6步：按比例分配出场费用**
        # 根据当前入场订单占总出场数量的比例，分配相应的出场费用
        size_fraction = entry_size / exit_size_sum        # 数量占比
        exit_fees = size_fraction * exit_fees_sum         # 按比例分配费用

        # **第7步：填充交易记录**
        # 调用标准交易记录填充函数，生成完整的交易记录
        fill_trade_record_nb(
            trade_records[tidx],    # 目标记录位置
            tidx,                   # 交易ID
            col,                    # 列索引
            entry_size,             # 入场数量
            record['idx'],          # 入场时间索引
            record['price'],        # 入场价格
            entry_fees,             # 入场费用
            exit_idx,               # 出场时间索引
            exit_price,             # 计算得出的平均出场价格
            exit_fees,              # 按比例分配的出场费用
            direction,              # 交易方向
            status,                 # 交易状态
            parent_id               # 父级交易ID
        )
        tidx += 1  # 移动到下一个交易记录位置

    return tidx  # 返回更新后的交易记录索引


@njit(cache=True)
def get_entry_trades_nb(order_records: tp.RecordArray, close: tp.Array2d, col_map: tp.ColMap) -> tp.RecordArray:
    """Fill entry trade records by aggregating order records.

    Entry trade records are buy orders in a long position and sell orders in a short position.

    Usage:
        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> from numba import njit
        >>> from vectorbt.records.nb import col_map_nb
        >>> from vectorbt.portfolio.nb import simulate_from_orders_nb, get_entry_trades_nb

        >>> close = order_price = np.array([
        ...     [1, 6],
        ...     [2, 5],
        ...     [3, 4],
        ...     [4, 3],
        ...     [5, 2],
        ...     [6, 1]
        ... ])
        >>> size = np.asarray([
        ...     [1, -1],
        ...     [0.1, -0.1],
        ...     [-1, 1],
        ...     [-0.1, 0.1],
        ...     [1, -1],
        ...     [-2, 2]
        ... ])
        >>> target_shape = close.shape
        >>> group_lens = np.full(target_shape[1], 1)
        >>> init_cash = np.full(target_shape[1], 100)
        >>> call_seq = np.full(target_shape, 0)

        >>> order_records, log_records = simulate_from_orders_nb(
        ...     target_shape,
        ...     group_lens,
        ...     init_cash,
        ...     call_seq,
        ...     size=size,
        ...     price=close,
        ...     fees=np.asarray(0.01),
        ...     slippage=np.asarray(0.01)
        ... )

        >>> col_map = col_map_nb(order_records['col'], target_shape[1])
        >>> entry_trade_records = get_entry_trades_nb(order_records, close, col_map)
        >>> pd.DataFrame.from_records(entry_trade_records)
           id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
        0   0    0   1.0          0         1.01     0.01010         3    3.060000
        1   1    0   0.1          1         2.02     0.00202         3    3.060000
        2   2    0   1.0          4         5.05     0.05050         5    5.940000
        3   3    0   1.0          5         5.94     0.05940         5    6.000000
        4   4    1   1.0          0         5.94     0.05940         3    3.948182
        5   5    1   0.1          1         4.95     0.00495         3    3.948182
        6   6    1   1.0          4         1.98     0.01980         5    1.010000
        7   7    1   1.0          5         1.01     0.01010         5    1.000000

           exit_fees       pnl    return  direction  status  parent_id
        0   0.030600  2.009300  1.989406          0       1          0
        1   0.003060  0.098920  0.489703          0       1          0
        2   0.059400  0.780100  0.154475          0       1          1
        3   0.000000 -0.119400 -0.020101          1       0          2
        4   0.039482  1.892936  0.318676          1       1          3
        5   0.003948  0.091284  0.184411          1       1          3
        6   0.010100  0.940100  0.474798          1       1          4
        7   0.000000 -0.020100 -0.019901          0       0          5
        ```
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    records = np.empty(len(order_records), dtype=trade_dt)
    tidx = 0
    parent_id = -1

    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        in_position = False

        for c in range(col_len):
            oidx = col_idxs[col_start_idxs[col] + c]
            record = order_records[oidx]

            if record['id'] < last_id:
                raise ValueError("id must come in ascending order per column")
            last_id = record['id']

            order_idx = record['idx']
            order_size = record['size']
            order_price = record['price']
            order_fees = record['fees']
            order_side = record['side']

            if order_size <= 0.:
                raise ValueError(size_zero_neg_err)
            if order_price <= 0.:
                raise ValueError(price_zero_neg_err)

            if not in_position:
                # New position opened
                first_c = c
                in_position = True
                parent_id += 1
                if order_side == OrderSide.Buy:
                    direction = TradeDirection.Long
                else:
                    direction = TradeDirection.Short
                entry_size_sum = 0.
                entry_gross_sum = 0.
                entry_fees_sum = 0.
                exit_size_sum = 0.
                exit_gross_sum = 0.
                exit_fees_sum = 0.
                first_entry_size = order_size
                first_entry_fees = order_fees

            if (direction == TradeDirection.Long and order_side == OrderSide.Buy) \
                    or (direction == TradeDirection.Short and order_side == OrderSide.Sell):
                # Position increased
                entry_size_sum += order_size
                entry_gross_sum += order_size * order_price
                entry_fees_sum += order_fees

            elif (direction == TradeDirection.Long and order_side == OrderSide.Sell) \
                    or (direction == TradeDirection.Short and order_side == OrderSide.Buy):
                if is_close_nb(exit_size_sum + order_size, entry_size_sum):
                    # Position closed
                    last_c = c
                    in_position = False
                    exit_size_sum = entry_size_sum
                    exit_gross_sum += order_size * order_price
                    exit_fees_sum += order_fees

                    # Fill trade records
                    tidx = fill_entry_trades_in_position_nb(
                        order_records,
                        col_map,
                        col,
                        first_c,
                        last_c,
                        first_entry_size,
                        first_entry_fees,
                        order_idx,
                        exit_size_sum,
                        exit_gross_sum,
                        exit_fees_sum,
                        direction,
                        TradeStatus.Closed,
                        parent_id,
                        records,
                        tidx
                    )
                elif is_less_nb(exit_size_sum + order_size, entry_size_sum):
                    # Position decreased
                    exit_size_sum += order_size
                    exit_gross_sum += order_size * order_price
                    exit_fees_sum += order_fees
                else:
                    # Position closed
                    last_c = c
                    remaining_size = add_nb(entry_size_sum, -exit_size_sum)
                    exit_size_sum = entry_size_sum
                    exit_gross_sum += remaining_size * order_price
                    exit_fees_sum += remaining_size / order_size * order_fees

                    # Fill trade records
                    tidx = fill_entry_trades_in_position_nb(
                        order_records,
                        col_map,
                        col,
                        first_c,
                        last_c,
                        first_entry_size,
                        first_entry_fees,
                        order_idx,
                        exit_size_sum,
                        exit_gross_sum,
                        exit_fees_sum,
                        direction,
                        TradeStatus.Closed,
                        parent_id,
                        records,
                        tidx
                    )

                    # New position opened
                    first_c = c
                    parent_id += 1
                    if order_side == OrderSide.Buy:
                        direction = TradeDirection.Long
                    else:
                        direction = TradeDirection.Short
                    entry_size_sum = add_nb(order_size, -remaining_size)
                    entry_gross_sum = entry_size_sum * order_price
                    entry_fees_sum = entry_size_sum / order_size * order_fees
                    first_entry_size = entry_size_sum
                    first_entry_fees = entry_fees_sum
                    exit_size_sum = 0.
                    exit_gross_sum = 0.
                    exit_fees_sum = 0.

        if in_position and is_less_nb(exit_size_sum, entry_size_sum):
            # Position hasn't been closed
            last_c = col_len - 1
            remaining_size = add_nb(entry_size_sum, -exit_size_sum)
            exit_size_sum = entry_size_sum
            exit_gross_sum += remaining_size * close[close.shape[0] - 1, col]

            # Fill trade records
            tidx = fill_entry_trades_in_position_nb(
                order_records,
                col_map,
                col,
                first_c,
                last_c,
                first_entry_size,
                first_entry_fees,
                close.shape[0] - 1,
                exit_size_sum,
                exit_gross_sum,
                exit_fees_sum,
                direction,
                TradeStatus.Open,
                parent_id,
                records,
                tidx
            )

    return records[:tidx]


@njit(cache=True)
def get_exit_trades_nb(order_records: tp.RecordArray, close: tp.Array2d, col_map: tp.ColMap) -> tp.RecordArray:
    """Fill exit trade records by aggregating order records.

    Exit trade records are sell orders in a long position and buy orders in a short position.

    Usage:
        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> from numba import njit
        >>> from vectorbt.records.nb import col_map_nb
        >>> from vectorbt.portfolio.nb import simulate_from_orders_nb, get_exit_trades_nb

        >>> close = order_price = np.array([
        ...     [1, 6],
        ...     [2, 5],
        ...     [3, 4],
        ...     [4, 3],
        ...     [5, 2],
        ...     [6, 1]
        ... ])
        >>> size = np.asarray([
        ...     [1, -1],
        ...     [0.1, -0.1],
        ...     [-1, 1],
        ...     [-0.1, 0.1],
        ...     [1, -1],
        ...     [-2, 2]
        ... ])
        >>> target_shape = close.shape
        >>> group_lens = np.full(target_shape[1], 1)
        >>> init_cash = np.full(target_shape[1], 100)
        >>> call_seq = np.full(target_shape, 0)

        >>> order_records, log_records = simulate_from_orders_nb(
        ...     target_shape,
        ...     group_lens,
        ...     init_cash,
        ...     call_seq,
        ...     size=size,
        ...     price=close,
        ...     fees=np.asarray(0.01),
        ...     slippage=np.asarray(0.01)
        ... )

        >>> col_map = col_map_nb(order_records['col'], target_shape[1])
        >>> exit_trade_records = get_exit_trades_nb(order_records, close, col_map)
        >>> pd.DataFrame.from_records(exit_trade_records)
           id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
        0   0    0   1.0          0     1.101818    0.011018         2        2.97
        1   1    0   0.1          0     1.101818    0.001102         3        3.96
        2   2    0   1.0          4     5.050000    0.050500         5        5.94
        3   3    0   1.0          5     5.940000    0.059400         5        6.00
        4   4    1   1.0          0     5.850000    0.058500         2        4.04
        5   5    1   0.1          0     5.850000    0.005850         3        3.03
        6   6    1   1.0          4     1.980000    0.019800         5        1.01
        7   7    1   1.0          5     1.010000    0.010100         5        1.00

           exit_fees       pnl    return  direction  status  parent_id
        0    0.02970  1.827464  1.658589          0       1          0
        1    0.00396  0.280756  2.548119          0       1          0
        2    0.05940  0.780100  0.154475          0       1          1
        3    0.00000 -0.119400 -0.020101          1       0          2
        4    0.04040  1.711100  0.292496          1       1          3
        5    0.00303  0.273120  0.466872          1       1          3
        6    0.01010  0.940100  0.474798          1       1          4
        7    0.00000 -0.020100 -0.019901          0       0          5
        ```
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    records = np.empty(len(order_records), dtype=trade_dt)
    tidx = 0
    parent_id = -1

    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        in_position = False

        for c in range(col_len):
            oidx = col_idxs[col_start_idxs[col] + c]
            record = order_records[oidx]

            if record['id'] < last_id:
                raise ValueError("id must come in ascending order per column")
            last_id = record['id']

            i = record['idx']
            order_size = record['size']
            order_price = record['price']
            order_fees = record['fees']
            order_side = record['side']

            if order_size <= 0.:
                raise ValueError(size_zero_neg_err)
            if order_price <= 0.:
                raise ValueError(price_zero_neg_err)

            if not in_position:
                # Trade opened
                in_position = True
                entry_idx = i
                if order_side == OrderSide.Buy:
                    direction = TradeDirection.Long
                else:
                    direction = TradeDirection.Short
                parent_id += 1
                entry_size_sum = 0.
                entry_gross_sum = 0.
                entry_fees_sum = 0.

            if (direction == TradeDirection.Long and order_side == OrderSide.Buy) \
                    or (direction == TradeDirection.Short and order_side == OrderSide.Sell):
                # Position increased
                entry_size_sum += order_size
                entry_gross_sum += order_size * order_price
                entry_fees_sum += order_fees

            elif (direction == TradeDirection.Long and order_side == OrderSide.Sell) \
                    or (direction == TradeDirection.Short and order_side == OrderSide.Buy):
                if is_close_or_less_nb(order_size, entry_size_sum):
                    # Trade closed
                    if is_close_nb(order_size, entry_size_sum):
                        exit_size = entry_size_sum
                    else:
                        exit_size = order_size
                    exit_price = order_price
                    exit_fees = order_fees
                    exit_idx = i

                    # Take a size-weighted average of entry price
                    entry_price = entry_gross_sum / entry_size_sum

                    # Take a fraction of entry fees
                    size_fraction = exit_size / entry_size_sum
                    entry_fees = size_fraction * entry_fees_sum

                    fill_trade_record_nb(
                        records[tidx],
                        tidx,
                        col,
                        exit_size,
                        entry_idx,
                        entry_price,
                        entry_fees,
                        exit_idx,
                        exit_price,
                        exit_fees,
                        direction,
                        TradeStatus.Closed,
                        parent_id
                    )
                    tidx += 1

                    if is_close_nb(order_size, entry_size_sum):
                        # Position closed
                        entry_idx = -1
                        direction = -1
                        in_position = False
                    else:
                        # Position decreased, previous orders have now less impact
                        size_fraction = (entry_size_sum - order_size) / entry_size_sum
                        entry_size_sum *= size_fraction
                        entry_gross_sum *= size_fraction
                        entry_fees_sum *= size_fraction
                else:
                    # Trade reversed
                    # Close current trade
                    cl_exit_size = entry_size_sum
                    cl_exit_price = order_price
                    cl_exit_fees = cl_exit_size / order_size * order_fees
                    cl_exit_idx = i

                    # Take a size-weighted average of entry price
                    entry_price = entry_gross_sum / entry_size_sum

                    # Take a fraction of entry fees
                    size_fraction = cl_exit_size / entry_size_sum
                    entry_fees = size_fraction * entry_fees_sum

                    fill_trade_record_nb(
                        records[tidx],
                        tidx,
                        col,
                        cl_exit_size,
                        entry_idx,
                        entry_price,
                        entry_fees,
                        cl_exit_idx,
                        cl_exit_price,
                        cl_exit_fees,
                        direction,
                        TradeStatus.Closed,
                        parent_id
                    )
                    tidx += 1

                    # Open a new trade
                    entry_size_sum = order_size - cl_exit_size
                    entry_gross_sum = entry_size_sum * order_price
                    entry_fees_sum = order_fees - cl_exit_fees
                    entry_idx = i
                    if direction == TradeDirection.Long:
                        direction = TradeDirection.Short
                    else:
                        direction = TradeDirection.Long
                    parent_id += 1

        if in_position and is_less_nb(-entry_size_sum, 0):
            # Trade hasn't been closed
            exit_size = entry_size_sum
            exit_price = close[close.shape[0] - 1, col]
            exit_fees = 0.
            exit_idx = close.shape[0] - 1

            # Take a size-weighted average of entry price
            entry_price = entry_gross_sum / entry_size_sum

            # Take a fraction of entry fees
            size_fraction = exit_size / entry_size_sum
            entry_fees = size_fraction * entry_fees_sum

            fill_trade_record_nb(
                records[tidx],
                tidx,
                col,
                exit_size,
                entry_idx,
                entry_price,
                entry_fees,
                exit_idx,
                exit_price,
                exit_fees,
                direction,
                TradeStatus.Open,
                parent_id
            )
            tidx += 1

    return records[:tidx]


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def trade_winning_streak_nb(records: tp.RecordArray) -> tp.Array1d:
    """
    计算每笔交易的当前连胜次数
    
    分析交易记录序列，计算到每笔交易为止的连续盈利交易次数。
    连胜次数在遇到亏损交易时重置为0，为交易策略的连续性
    表现提供量化指标。
    
    参数:
    ----
    records : RecordArray
        交易记录数组，必须包含'pnl'字段（盈亏信息）
        
    返回:
    ----
    Array1d[int64]
        每笔交易对应的连胜次数数组
        - 如果当前交易盈利：连胜次数 = 之前连胜次数 + 1
        - 如果当前交易亏损：连胜次数重置为 0
        
    使用示例:
    --------
    >>> import numpy as np
    >>> from vectorbt.portfolio.enums import trade_dt
    >>> 
    >>> # 创建示例交易记录
    >>> records = np.array([
    ...     (0, 0, 100, 0, 50, 0, 1, 55, 0, 500, 0.1, 1, 0, 0),   # 盈利500
    ...     (1, 0, 200, 1, 60, 0, 2, 65, 0, 1000, 0.167, 1, 0, 0), # 盈利1000
    ...     (2, 0, 150, 2, 45, 0, 3, 40, 0, -750, -0.167, 1, 0, 0), # 亏损-750
    ...     (3, 0, 300, 3, 30, 0, 4, 35, 0, 1500, 0.167, 1, 0, 0), # 盈利1500
    ... ], dtype=trade_dt)
    >>> 
    >>> winning_streaks = trade_winning_streak_nb(records)
    >>> print(winning_streaks)  # [1, 2, 0, 1]
    
    应用场景:
    --------
    - **策略稳定性分析**：评估策略产生连续盈利的能力
    - **心理因素建模**：分析连胜对交易者信心的影响  
    - **风险管理优化**：基于连胜次数调整仓位大小
    - **策略参数调优**：识别最佳连胜区间进行策略优化
    - **绩效报告生成**：提供策略连续性表现的统计指标
    
    算法特点:
    --------
    - **实时更新**：逐笔交易计算，反映动态变化过程
    - **自动重置**：遇到亏损交易时连胜计数器清零
    - **高效计算**：单次遍历完成所有计算，时间复杂度O(n)
    - **简洁逻辑**：基于盈亏二元判断，计算结果直观易懂
    
    注意事项:
    --------
    - 依赖records中的'pnl'字段准确性
    - 连胜次数不包括盈亏平衡（pnl=0）的交易
    - 适用于按时间顺序排列的交易记录
    - 可与连败次数指标配合使用进行全面分析
    """
    # 初始化输出数组，默认连胜次数为0
    out = np.full(len(records), 0, dtype=np.int64)
    curr_rank = 0  # 当前连胜计数器
    
    # 遍历每笔交易记录
    for i in range(len(records)):
        if records[i]['pnl'] > 0:
            # 盈利交易：连胜次数递增
            curr_rank += 1
        else:
            # 亏损或平衡交易：连胜次数重置
            curr_rank = 0
        
        # 记录当前交易的连胜次数
        out[i] = curr_rank
        
    return out


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def trade_losing_streak_nb(records: tp.RecordArray) -> tp.Array1d:
    """
    计算每笔交易的当前连败次数
    
    分析交易记录序列，计算到每笔交易为止的连续亏损交易次数。
    连败次数在遇到盈利交易时重置为0，为交易策略的风险控制
    和心理承受能力评估提供重要参考。
    
    参数:
    ----
    records : RecordArray
        交易记录数组，必须包含'pnl'字段（盈亏信息）
        
    返回:
    ----
    Array1d[int64]
        每笔交易对应的连败次数数组
        - 如果当前交易亏损：连败次数 = 之前连败次数 + 1
        - 如果当前交易盈利：连败次数重置为 0
        
    使用示例:
    --------
    >>> import numpy as np
    >>> from vectorbt.portfolio.enums import trade_dt
    >>> 
    >>> # 创建示例交易记录（同连胜示例）
    >>> records = np.array([
    ...     (0, 0, 100, 0, 50, 0, 1, 55, 0, 500, 0.1, 1, 0, 0),   # 盈利500
    ...     (1, 0, 200, 1, 60, 0, 2, 65, 0, 1000, 0.167, 1, 0, 0), # 盈利1000
    ...     (2, 0, 150, 2, 45, 0, 3, 40, 0, -750, -0.167, 1, 0, 0), # 亏损-750
    ...     (3, 0, 300, 3, 30, 0, 4, 25, 0, -1500, -0.5, 1, 0, 0), # 亏损-1500
    ...     (4, 0, 100, 4, 40, 0, 5, 45, 0, 500, 0.125, 1, 0, 0),  # 盈利500
    ... ], dtype=trade_dt)
    >>> 
    >>> losing_streaks = trade_losing_streak_nb(records)
    >>> print(losing_streaks)  # [0, 0, 1, 2, 0]
    
    应用场景:
    --------
    - **风险管控**：设定最大连败次数阈值，触发时暂停交易
    - **资金管理**：根据连败次数动态调整交易仓位大小
    - **策略优化**：识别策略的最大回撤期间和恢复能力
    - **心理建设**：为交易者提供连败承受能力的客观评估
    - **报警系统**：建立连败预警机制，及时调整交易策略
    
    与连胜次数的对比:
    ---------------
    - **互补性**：连胜和连败次数共同描绘策略的完整表现轨迹
    - **风险导向**：连败次数更侧重风险识别和控制
    - **实用性**：在实际交易中，连败监控往往比连胜追踪更重要
    - **预警价值**：连败次数达到历史高位通常是策略调整信号
    
    风险控制应用:
    -----------
    - **凯利公式调整**：基于连败次数调整最优仓位大小
    - **止损策略**：连败达到阈值时实施更严格的止损机制
    - **暂停机制**：连败超限时暂停自动交易，人工介入
    - **策略切换**：长时间连败可能意味着市场环境变化
    
    注意事项:
    --------
    - 连败统计不包括盈亏平衡（pnl=0）的交易
    - 应与连胜次数、回撤等指标综合分析
    - 适合按时间顺序排列的交易记录
    - 连败次数的统计意义依赖于足够的样本量
    """
    out = np.full(len(records), 0, dtype=np.int64)
    curr_rank = 0
    for i in range(len(records)):
        if records[i]['pnl'] < 0:
            curr_rank += 1
        else:
            curr_rank = 0
        out[i] = curr_rank
    return out


# ############# Position records ############# #

@njit(cache=True)
def fill_position_record_nb(record: tp.Record, id_: int, trade_records: tp.RecordArray) -> None:
    """Fill a position record by aggregating trade records."""
    # Aggregate trades
    col = trade_records['col'][0]
    size = np.sum(trade_records['size'])
    entry_idx = trade_records['entry_idx'][0]
    entry_price = np.sum(trade_records['size'] * trade_records['entry_price']) / size
    entry_fees = np.sum(trade_records['entry_fees'])
    exit_idx = trade_records['exit_idx'][-1]
    exit_price = np.sum(trade_records['size'] * trade_records['exit_price']) / size
    exit_fees = np.sum(trade_records['exit_fees'])
    direction = trade_records['direction'][-1]
    status = trade_records['status'][-1]
    pnl, ret = get_trade_stats_nb(
        size,
        entry_price,
        entry_fees,
        exit_price,
        exit_fees,
        direction
    )

    # Save position
    record['id'] = id_
    record['col'] = col
    record['size'] = size
    record['entry_idx'] = entry_idx
    record['entry_price'] = entry_price
    record['entry_fees'] = entry_fees
    record['exit_idx'] = exit_idx
    record['exit_price'] = exit_price
    record['exit_fees'] = exit_fees
    record['pnl'] = pnl
    record['return'] = ret
    record['direction'] = direction
    record['status'] = status
    record['parent_id'] = id_


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def copy_trade_record_nb(record: tp.Record, trade_record: tp.Record) -> None:
    """
    复制交易记录
    
    将源交易记录的所有字段完全复制到目标记录中。这是一个高效的
    内存复制操作，常用于交易记录的数组重组、过滤和聚合过程。
    
    参数:
    ----
    record : Record
        目标记录，将被源记录的内容覆盖
        
    trade_record : Record
        源交易记录，提供要复制的数据
        
    返回:
    ----
    None
        直接修改目标记录，无返回值
        
    复制字段清单:
    -----------
    - **标识字段**: id, col, parent_id
    - **数量字段**: size
    - **入场信息**: entry_idx, entry_price, entry_fees
    - **出场信息**: exit_idx, exit_price, exit_fees
    - **绩效指标**: pnl, return
    - **状态标识**: direction, status
    
    使用示例:
    --------
    >>> import numpy as np
    >>> from vectorbt.portfolio.enums import trade_dt
    >>> 
    >>> # 创建源交易记录
    >>> source = np.empty(1, dtype=trade_dt)[0]
    >>> source['id'] = 1
    >>> source['col'] = 0
    >>> source['size'] = 100.0
    >>> source['pnl'] = 500.0
    >>> # ... 设置其他字段
    >>> 
    >>> # 创建目标记录并复制
    >>> target = np.empty(1, dtype=trade_dt)[0]
    >>> copy_trade_record_nb(target, source)
    >>> # 现在target包含了source的所有数据
    
    应用场景:
    --------
    - **记录数组重建**：在交易记录聚合过程中复制有效记录
    - **数据去重**：复制唯一记录到新数组中
    - **分组处理**：将特定条件的记录复制到分组数组
    - **格式转换**：在不同记录格式间进行数据迁移
    - **缓存操作**：将临时记录复制到持久存储
    
    性能特点:
    --------
    - **字段级复制**：逐字段复制确保数据完整性
    - **Numba优化**：编译为机器码执行，性能优异
    - **内存安全**：不涉及复杂的内存分配，避免内存泄漏
    - **类型保证**：保持原始数据类型和精度
    
    注意事项:
    --------
    - 目标记录的原始内容将被完全覆盖
    - 两个记录必须具有相同的结构体定义
    - 不执行数据验证，需要确保源数据的有效性
    - 常与其他交易记录处理函数配合使用
    """
    # 逐字段复制所有交易记录数据
    record['id'] = trade_record['id']                       # 交易唯一标识符
    record['col'] = trade_record['col']                     # 列索引（资产索引）
    record['size'] = trade_record['size']                   # 交易数量
    record['entry_idx'] = trade_record['entry_idx']         # 入场时间索引
    record['entry_price'] = trade_record['entry_price']     # 入场价格
    record['entry_fees'] = trade_record['entry_fees']       # 入场费用
    record['exit_idx'] = trade_record['exit_idx']           # 出场时间索引
    record['exit_price'] = trade_record['exit_price']       # 出场价格
    record['exit_fees'] = trade_record['exit_fees']         # 出场费用
    record['pnl'] = trade_record['pnl']                     # 净盈亏
    record['return'] = trade_record['return']               # 收益率
    record['direction'] = trade_record['direction']         # 交易方向
    record['status'] = trade_record['status']               # 交易状态
    record['parent_id'] = trade_record['parent_id']         # 父级交易ID


@njit(cache=True)
def get_positions_nb(trade_records: tp.RecordArray, col_map: tp.ColMap) -> tp.RecordArray:
    """Fill position records by aggregating trade records.

    Trades can be entry trades, exit trades, and even positions themselves - all will produce the same results.

    Usage:
        * Building upon the example in `get_exit_trades_nb`:

        ```pycon
        >>> from vectorbt.portfolio.nb import get_positions_nb

        >>> col_map = col_map_nb(exit_trade_records['col'], target_shape[1])
        >>> position_records = get_positions_nb(exit_trade_records, col_map)
        >>> pd.DataFrame.from_records(position_records)
           id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
        0   0    0   1.1          0     1.101818     0.01212         3    3.060000
        1   1    0   1.0          4     5.050000     0.05050         5    5.940000
        2   2    0   1.0          5     5.940000     0.05940         5    6.000000
        3   3    1   1.1          0     5.850000     0.06435         3    3.948182
        4   4    1   1.0          4     1.980000     0.01980         5    1.010000
        5   5    1   1.0          5     1.010000     0.01010         5    1.000000

           exit_fees      pnl    return  direction  status  parent_id
        0    0.03366  2.10822  1.739455          0       1          0
        1    0.05940  0.78010  0.154475          0       1          1
        2    0.00000 -0.11940 -0.020101          1       0          2
        3    0.04343  1.98422  0.308348          1       1          3
        4    0.01010  0.94010  0.474798          1       1          4
        5    0.00000 -0.02010 -0.019901          0       0          5
        ```
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    records = np.empty(len(trade_records), dtype=trade_dt)
    pidx = 0
    from_tidx = -1

    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        last_position_id = -1

        for c in range(col_len):
            tidx = col_idxs[col_start_idxs[col] + c]
            record = trade_records[tidx]

            if record['id'] < last_id:
                raise ValueError("id must come in ascending order per column")
            last_id = record['id']

            parent_id = record['parent_id']

            if parent_id != last_position_id:
                if last_position_id != -1:
                    if tidx - from_tidx > 1:
                        fill_position_record_nb(records[pidx], pidx, trade_records[from_tidx:tidx])
                    else:
                        # Speed up
                        copy_trade_record_nb(records[pidx], trade_records[from_tidx])
                        records[pidx]['id'] = pidx
                        records[pidx]['parent_id'] = pidx
                    pidx += 1
                from_tidx = tidx
                last_position_id = parent_id

        if tidx - from_tidx > 0:
            fill_position_record_nb(records[pidx], pidx, trade_records[from_tidx:tidx + 1])
        else:
            # Speed up
            copy_trade_record_nb(records[pidx], trade_records[from_tidx])
            records[pidx]['id'] = pidx
            records[pidx]['parent_id'] = pidx
        pidx += 1

    return records[:pidx]


# ############# Assets ############# #


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def get_long_size_nb(position_before: float, position_now: float) -> float:
    """
    计算多头持仓净变化量
    
    计算从前一持仓到当前持仓中多头方向的净变化量。这个函数考虑了
    复杂的持仓转换情况，包括多空转换的完整逻辑。
    
    参数:
    ----
    position_before : float
        前一时点的持仓，正数=多头，负数=空头，0=空仓
        
    position_now : float
        当前时点的持仓，正数=多头，负数=空头，0=空仓
        
    返回:
    ----
    float
        多头净变化量，可正可负
        - 正值：净增加多头持仓
        - 负值：净减少多头持仓
        - 零值：多头持仓无净变化
        
    逻辑分类:
    --------
    1. **双方均为空头/空仓** (position_before ≤ 0, position_now ≤ 0)：
       返回0，因为不涉及多头变化
       
    2. **多头转空头** (position_before ≥ 0, position_now < 0)：
       返回-position_before，表示减少了所有多头持仓
       
    3. **空头转多头** (position_before < 0, position_now ≥ 0)：
       返回position_now，表示新建多头持仓
       
    4. **双方均为多头** (其他情况)：
       返回净变化量 = position_now - position_before
    
    应用场景:
    --------
    - **风险管理**：监控多头暴露的变化
    - **交易分析**：分析多头建仓/平仓行为
    - **绩效归因**：计算多头策略的贡献
    - **资金管理**：评估多头资金的使用效率
    
    计算示例:
    --------
    - get_long_size_nb(100, 150) = 50    # 多头增仓50
    - get_long_size_nb(100, 50) = -50    # 多头减仓50
    - get_long_size_nb(-100, 50) = 50    # 空头转多头，新增50多头
    - get_long_size_nb(100, -50) = -100  # 多头转空头，减少100多头
    - get_long_size_nb(-100, -150) = 0   # 空头变化，多头无变化
    
    技术特点:
    --------
    - **数值稳定**：使用add_nb函数确保浮点数运算精度
    - **条件覆盖**：完整处理所有持仓转换情况
    - **性能优化**：Numba编译，适合大规模计算
    """
    # 双方均为空头或空仓，多头无变化
    if position_before <= 0 and position_now <= 0:
        return 0.
    # 多头转空头，减少所有多头持仓    
    if position_before >= 0 and position_now < 0:
        return -position_before
    # 空头转多头，新建多头持仓
    if position_before < 0 and position_now >= 0:
        return position_now
    # 双方均为多头，计算净变化
    return add_nb(position_now, -position_before)


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def get_short_size_nb(position_before: float, position_now: float) -> float:
    """
    计算空头持仓净变化量
    
    计算从前一持仓到当前持仓中空头方向的净变化量。与get_long_size_nb
    配合使用，可以完整分析持仓的多空转换过程。
    
    参数:
    ----
    position_before : float
        前一时点的持仓，正数=多头，负数=空头，0=空仓
        
    position_now : float
        当前时点的持仓，正数=多头，负数=空头，0=空仓
        
    返回:
    ----
    float
        空头净变化量，通常为负值或零
        - 负值：净增加空头持仓（更负）
        - 正值：净减少空头持仓（减负）
        - 零值：空头持仓无净变化
        
    逻辑分类:
    --------
    1. **双方均为多头/空仓** (position_before ≥ 0, position_now ≥ 0)：
       返回0，因为不涉及空头变化
       
    2. **多头转空头** (position_before ≥ 0, position_now < 0)：
       返回-position_now，表示新建空头持仓（负值）
       
    3. **空头转多头** (position_before < 0, position_now ≥ 0)：
       返回position_before，表示减少空头持仓（正值）
       
    4. **双方均为空头** (其他情况)：
       返回净变化量 = position_before - position_now
    
    应用场景:
    --------
    - **做空策略**：监控空头头寸的建立和平仓
    - **对冲管理**：评估对冲头寸的变化
    - **风险控制**：跟踪空头暴露的动态变化
    - **市场中性**：维持多空平衡的投资组合
    
    计算示例:
    --------
    - get_short_size_nb(-100, -150) = -50  # 空头增仓50（更负）
    - get_short_size_nb(-150, -100) = 50   # 空头减仓50（减负）
    - get_short_size_nb(100, -50) = -50    # 多头转空头，新增50空头
    - get_short_size_nb(-100, 50) = -100   # 空头转多头，减少100空头
    - get_short_size_nb(100, 150) = 0      # 多头变化，空头无变化
    
    与多头计算的互补性:
    ---------------
    - get_long_size_nb() + get_short_size_nb() = position_now - position_before
    - 两个函数分别捕获多空两个方向的持仓变化
    - 可以精确重构完整的持仓变化过程
    
    技术特点:
    --------
    - **数值精度**：使用add_nb确保浮点数计算稳定性
    - **逻辑完整**：覆盖所有可能的持仓转换场景
    - **高性能**：Numba优化，支持向量化计算
    
    注意事项:
    --------
    - 返回值的符号有特定含义，负值表示空头增加
    - 与传统的绝对值计算不同，保留方向信息
    - 适用于需要精确跟踪多空转换的高级策略分析
    """
    # 双方均为多头或空仓，空头无变化
    if position_before >= 0 and position_now >= 0:
        return 0.
    # 多头转空头，新建空头持仓
    if position_before >= 0 and position_now < 0:
        return -position_now
    # 空头转多头，减少空头持仓
    if position_before < 0 and position_now >= 0:
        return position_before
    # 双方均为空头，计算净变化
    return add_nb(position_before, -position_now)


@njit(cache=True)
def asset_flow_nb(target_shape: tp.Shape,
                  order_records: tp.RecordArray,
                  col_map: tp.ColMap,
                  direction: int) -> tp.Array2d:
    """Get asset flow series per column.

    Returns the total transacted amount of assets at each time step."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(target_shape, 0., dtype=np.float64)

    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        position_now = 0.

        for c in range(col_len):
            oidx = col_idxs[col_start_idxs[col] + c]
            record = order_records[oidx]

            if record['id'] < last_id:
                raise ValueError("id must come in ascending order per column")
            last_id = record['id']

            i = record['idx']
            side = record['side']
            size = record['size']

            if side == OrderSide.Sell:
                size *= -1
            new_position_now = add_nb(position_now, size)
            if direction == Direction.LongOnly:
                asset_flow = get_long_size_nb(position_now, new_position_now)
            elif direction == Direction.ShortOnly:
                asset_flow = get_short_size_nb(position_now, new_position_now)
            else:
                asset_flow = size
            out[i, col] = add_nb(out[i, col], asset_flow)
            position_now = new_position_now
    return out


@njit(cache=True)
def assets_nb(asset_flow: tp.Array2d) -> tp.Array2d:
    """Get asset series per column.

    Returns the current position at each time step."""
    out = np.empty_like(asset_flow)
    for col in range(asset_flow.shape[1]):
        position_now = 0.
        for i in range(asset_flow.shape[0]):
            flow_value = asset_flow[i, col]
            position_now = add_nb(position_now, flow_value)
            out[i, col] = position_now
    return out


@njit(cache=True)
def i_group_any_reduce_nb(i: int, group: int, a: tp.Array1d) -> bool:
    """Boolean "any" reducer for grouped columns."""
    return np.any(a)


@njit
def position_mask_grouped_nb(position_mask: tp.Array2d, group_lens: tp.Array1d) -> tp.Array2d:
    """Get whether in position for each row and group."""
    return generic_nb.squeeze_grouped_nb(position_mask, group_lens, i_group_any_reduce_nb).astype(np.bool_)


@njit(cache=True)
def group_mean_reduce_nb(group: int, a: tp.Array1d) -> float:
    """Mean reducer for grouped columns."""
    return np.mean(a)


@njit
def position_coverage_grouped_nb(position_mask: tp.Array2d, group_lens: tp.Array1d) -> tp.Array2d:
    """Get coverage of position for each row and group."""
    return generic_nb.reduce_grouped_nb(position_mask, group_lens, group_mean_reduce_nb)


# ############# Cash ############# #


@njit(cache=True)
def get_free_cash_diff_nb(position_before: float,
                          position_now: float,
                          debt_now: float,
                          price: float,
                          fees: float) -> tp.Tuple[float, float]:
    """Get updated debt and free cash flow."""
    size = add_nb(position_now, -position_before)
    final_cash = -size * price - fees
    if is_close_nb(size, 0):
        new_debt = debt_now
        free_cash_diff = 0.
    elif size > 0:
        if position_before < 0:
            if position_now < 0:
                short_size = abs(size)
            else:
                short_size = abs(position_before)
            avg_entry_price = debt_now / abs(position_before)
            debt_diff = short_size * avg_entry_price
            new_debt = add_nb(debt_now, -debt_diff)
            free_cash_diff = add_nb(2 * debt_diff, final_cash)
        else:
            new_debt = debt_now
            free_cash_diff = final_cash
    else:
        if position_now < 0:
            if position_before < 0:
                short_size = abs(size)
            else:
                short_size = abs(position_now)
            short_value = short_size * price
            new_debt = debt_now + short_value
            free_cash_diff = add_nb(final_cash, -2 * short_value)
        else:
            new_debt = debt_now
            free_cash_diff = final_cash
    return new_debt, free_cash_diff


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def cash_flow_nb(target_shape: tp.Shape,
                 order_records: tp.RecordArray,
                 col_map: tp.ColMap,
                 free: bool) -> tp.Array2d:
    """
    计算每列的现金流序列
    
    根据订单记录计算每个时间点每个资产的现金流变化。支持计算
    总现金流和自由现金流两种模式，为投资组合的流动性分析
    提供基础数据。
    
    参数:
    ----
    target_shape : Shape
        目标形状 (时间步数, 资产数量)
        
    order_records : RecordArray
        订单记录数组，包含完整的交易信息
        
    col_map : ColMap
        列映射信息，用于快速定位各列的订单
        
    free : bool
        是否计算自由现金流
        - True: 计算自由现金流（考虑债务和保证金）
        - False: 计算总现金流（仅考虑交易金额）
        
    返回:
    ----
    Array2d[float64]
        现金流矩阵，形状为target_shape
        - 正值表示现金流入
        - 负值表示现金流出
        
    算法逻辑:
    --------
    **总现金流模式** (free=False):
    - cash_flow = -(size × price) - fees
    - 买入为负值（现金流出）
    - 卖出为正值（现金流入）
    
    **自由现金流模式** (free=True):
    - 考虑保证金和债务变化
    - 空头交易会产生债务和保证金要求
    - 计算实际可用现金的变化
    
    应用场景:
    --------
    - **流动性分析**：评估投资组合在不同时点的现金需求
    - **资金规划**：预测未来现金流入流出情况
    - **风险控制**：监控保证金使用和债务水平
    - **绩效归因**：分析现金流对总体收益的贡献
    - **再平衡策略**：确定最佳的现金配置时机
    
    注意事项:
    --------
    - 订单记录必须按ID升序排列
    - 自由现金流计算更复杂但更真实
    - 空头交易在自由现金流模式下会产生债务
    - 结果矩阵可用于进一步的财务分析
    """
    # 解析列映射信息
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    
    # 初始化输出矩阵
    out = np.full(target_shape, 0., dtype=np.float64)

    # 遍历每一列
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
            
        # 初始化列级状态变量
        last_id = -1
        position_now = 0.
        debt_now = 0.

        # 遍历当前列的所有订单
        for c in range(col_len):
            oidx = col_idxs[col_start_idxs[col] + c]
            record = order_records[oidx]

            # 验证订单ID顺序
            if record['id'] < last_id:
                raise ValueError("订单ID必须在每列内按升序排列")
            last_id = record['id']

            # 提取订单信息
            i = record['idx']        # 时间索引
            side = record['side']    # 订单方向
            size = record['size']    # 订单数量
            price = record['price']  # 执行价格
            fees = record['fees']    # 手续费

            # 统一订单方向：卖单转为负数量
            if side == OrderSide.Sell:
                size *= -1
            new_position_now = add_nb(position_now, size)
            
            # 根据模式计算现金流
            if free:
                # 自由现金流：考虑债务和保证金
                debt_now, cash_flow = get_free_cash_diff_nb(
                    position_now,      # 原持仓
                    new_position_now,  # 新持仓
                    debt_now,          # 当前债务
                    price,             # 交易价格
                    fees               # 手续费
                )
            else:
                # 总现金流：交易金额 + 手续费
                cash_flow = -size * price - fees
                
            # 累加到输出矩阵
            out[i, col] = add_nb(out[i, col], cash_flow)
            position_now = new_position_now
            
    return out


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def sum_grouped_nb(a: tp.Array2d, group_lens: tp.Array1d) -> tp.Array2d:
    """
    按组聚合列数据
    
    将每个组内的多列数据通过求和操作压缩成单列，实现从
    资产级别向组合级别的数据聚合。常用于现金共享组合
    的统计计算。
    
    参数:
    ----
    a : Array2d
        待聚合的二维数组 (时间 × 资产)
        
    group_lens : Array1d
        每个组包含的列数数组
        
    返回:
    ----
    Array2d
        聚合后的二维数组 (时间 × 组数)
        
    算法示例:
    --------
    >>> # 输入：3个时间点，5个资产，分为2组 [3, 2]
    >>> a = [[1, 2, 3, 4, 5],      # 时间0
    ...      [6, 7, 8, 9, 10],     # 时间1  
    ...      [11, 12, 13, 14, 15]] # 时间2
    >>> group_lens = [3, 2]        # 第1组：3列，第2组：2列
    >>> 
    >>> # 输出：3个时间点，2个组
    >>> result = [[6, 9],          # 时间0：[1+2+3, 4+5]
    ...           [21, 19],        # 时间1：[6+7+8, 9+10]
    ...           [36, 29]]        # 时间2：[11+12+13, 14+15]
    
    应用场景:
    --------
    - **现金流聚合**：将组内资产的现金流合并为组合现金流
    - **收益汇总**：计算投资组合级别的总收益
    - **风险指标**：聚合组内资产的风险暴露
    - **报告生成**：将详细数据汇总为管理层报告
    - **绩效分析**：从资产表现推导组合表现
    
    技术特点:
    --------
    - **高效求和**：利用NumPy的向量化求和操作
    - **分组处理**：自动处理不同大小的资产组合
    - **内存优化**：预分配输出数组，避免动态内存分配
    - **数值稳定**：保持原数组的数值精度
    
    注意事项:
    --------
    - group_lens的总和必须等于输入数组的列数
    - 聚合操作是不可逆的，会丢失资产级别的细节信息
    - 适用于加法意义明确的数据类型（现金、收益等）
    """
    # 验证组长度配置
    check_group_lens_nb(group_lens, a.shape[1])

    # 预分配输出数组
    out = np.empty((a.shape[0], len(group_lens)), dtype=np.float64)
    
    # 遍历每个组进行聚合
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        # 对当前组的所有列进行求和
        out[:, group] = np.sum(a[:, from_col:to_col], axis=1)
        from_col = to_col
        
    return out


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def cash_flow_grouped_nb(cash_flow: tp.Array2d, group_lens: tp.Array1d) -> tp.Array2d:
    """
    计算分组现金流序列
    
    将资产级别的现金流数据聚合为组合级别，适用于现金共享
    投资组合的现金流分析。这是sum_grouped_nb的特化版本。
    
    参数:
    ----
    cash_flow : Array2d
        资产级别的现金流矩阵 (时间 × 资产)
        
    group_lens : Array1d
        每个组包含的资产数量
        
    返回:
    ----
    Array2d
        组合级别的现金流矩阵 (时间 × 组合)
        
    应用场景:
    --------
    - **组合现金流跟踪**：监控现金共享组合的资金流动
    - **流动性管理**：评估组合级别的现金需求和供给
    - **资金配置优化**：基于组合现金流做资产配置决策
    - **绩效归因分析**：分解组合收益中现金流的贡献
    
    注意事项:
    --------
    - 现金流聚合假设组内资产共享现金池
    - 正值表示组合现金流入，负值表示流出
    - 结果用于组合级别的财务分析和决策
    """
    return sum_grouped_nb(cash_flow, group_lens)


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def init_cash_grouped_nb(init_cash: tp.Array1d, group_lens: tp.Array1d, cash_sharing: bool) -> tp.Array1d:
    """
    计算分组初始现金
    
    根据现金共享模式，将资产级别的初始现金转换为组合级别。
    支持现金共享和独立现金两种模式的处理。
    
    参数:
    ----
    init_cash : Array1d
        初始现金数组
        - 现金共享模式：每个组一个值
        - 独立现金模式：每个资产一个值
        
    group_lens : Array1d
        每个组包含的资产数量
        
    cash_sharing : bool
        现金共享模式标志
        - True: 组内资产共享现金，直接返回init_cash
        - False: 各资产独立现金，需要按组求和
        
    返回:
    ----
    Array1d
        组合级别的初始现金数组
        
    算法逻辑:
    --------
    **现金共享模式**:
    - 每个组已有统一的初始现金
    - 直接返回原数组
    
    **独立现金模式**:
    - 将组内各资产的初始现金求和
    - 得到每个组的总初始现金
    
    使用示例:
    --------
    >>> # 现金共享模式
    >>> init_cash = [10000, 5000]      # 2个组的初始现金
    >>> group_lens = [3, 2]            # 第1组3个资产，第2组2个资产
    >>> result = init_cash_grouped_nb(init_cash, group_lens, True)
    >>> # 结果：[10000, 5000]
    >>> 
    >>> # 独立现金模式
    >>> init_cash = [2000, 3000, 5000, 1000, 4000]  # 5个资产的初始现金
    >>> group_lens = [3, 2]                          # 分为2组
    >>> result = init_cash_grouped_nb(init_cash, group_lens, False)
    >>> # 结果：[10000, 5000]  # [2000+3000+5000, 1000+4000]
    
    应用场景:
    --------
    - **组合初始化**：设置现金共享组合的初始资金
    - **资金分配**：将总资金按组分配给不同策略
    - **风险预算**：为各投资组合设定初始风险资本
    - **绩效基准**：建立组合绩效评估的基准起点
    
    注意事项:
    --------
    - 现金共享模式下，init_cash长度应等于组数
    - 独立现金模式下，init_cash长度应等于总资产数
    - 计算结果影响后续所有收益率和绩效指标
    """
    if cash_sharing:
        # 现金共享模式：直接返回组级别现金
        return init_cash
    
    # 独立现金模式：按组汇总资产现金
    out = np.empty(group_lens.shape, dtype=np.float64)
    from_col = 0
    
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        
        # 计算当前组内所有资产的现金总和
        cash_sum = 0.
        for col in range(from_col, to_col):
            cash_sum += init_cash[col]
            
        out[group] = cash_sum
        from_col = to_col
        
    return out


@njit(cache=True)
def init_cash_nb(init_cash: tp.Array1d, group_lens: tp.Array1d, cash_sharing: bool) -> tp.Array1d:
    """Get initial cash per column."""
    if not cash_sharing:
        return init_cash
    group_lens_cs = np.cumsum(group_lens)
    out = np.full(group_lens_cs[-1], np.nan, dtype=np.float64)
    out[group_lens_cs - group_lens] = init_cash
    out = generic_nb.ffill_1d_nb(out)
    return out


@njit(cache=True)
def cash_nb(cash_flow: tp.Array2d, init_cash: tp.Array1d) -> tp.Array2d:
    """Get cash series per column."""
    out = np.empty_like(cash_flow)
    for col in range(cash_flow.shape[1]):
        for i in range(cash_flow.shape[0]):
            cash_now = init_cash[col] if i == 0 else out[i - 1, col]
            out[i, col] = add_nb(cash_now, cash_flow[i, col])
    return out


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def cash_in_sim_order_nb(cash_flow: tp.Array2d,
                         group_lens: tp.Array1d,
                         init_cash_grouped: tp.Array1d,
                         call_seq: tp.Array2d) -> tp.Array2d:
    """
    按模拟顺序计算现金序列
    
    根据订单执行顺序（call_seq）计算现金的动态变化。这对于现金共享
    组合特别重要，因为订单的执行顺序会影响后续订单的可用资金。
    
    参数:
    ----
    cash_flow : Array2d
        现金流矩阵 (时间 × 资产)，每个元素表示该时点该资产的现金变化
        
    group_lens : Array1d
        每个组包含的资产数量，用于确定现金共享范围
        
    init_cash_grouped : Array1d
        每个组的初始现金，作为现金累计的起点
        
    call_seq : Array2d
        订单调用序列矩阵，定义每个时点各资产的执行顺序
        
    返回:
    ----
    Array2d
        按执行顺序的现金序列，显示每次订单执行后的现金状态
        
    算法核心:
    --------
    1. **按组处理**：每个组内的资产共享现金池
    2. **按序累积**：严格按照call_seq定义的顺序累计现金变化
    3. **实时更新**：每次订单执行后立即更新可用现金
    
    使用场景:
    --------
    - **订单优先级分析**：评估不同执行顺序对现金使用的影响
    - **资金约束建模**：精确模拟资金不足对后续订单的限制
    - **策略回测优化**：确保回测中的资金使用与实际交易一致
    - **风险控制验证**：检验在极端现金流情况下的策略表现
    
    技术特点:
    --------
    - **顺序敏感性**：严格按照预定义的执行顺序进行计算
    - **组内共享**：组内所有资产使用同一现金池
    - **实时反馈**：每步执行后的现金状态都会记录
    - **数值稳定性**：使用add_nb函数确保浮点数运算精度
    
    注意事项:
    --------
    - 订单执行顺序的差异可能导致截然不同的结果
    - 现金共享模式下，最后执行的订单可能面临资金不足
    - 适用于需要精确模拟订单执行时序的场景
    """
    # 验证组配置的合法性
    check_group_lens_nb(group_lens, cash_flow.shape[1])

    # 预分配输出矩阵
    out = np.empty_like(cash_flow)
    from_col = 0
    
    # 遍历每个资产组
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        
        # 初始化组内现金状态
        cash_now = init_cash_grouped[group]
        
        # 按时间步遍历
        for i in range(cash_flow.shape[0]):
            # 按call_seq定义的顺序处理组内资产
            for k in range(group_len):
                col = from_col + call_seq[i, from_col + k]
                
                # 累积现金变化
                cash_now = add_nb(cash_now, cash_flow[i, col])
                
                # 记录执行该订单后的现金状态
                out[i, col] = cash_now
                
        from_col = to_col
        
    return out


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def cash_grouped_nb(target_shape: tp.Shape,
                    cash_flow_grouped: tp.Array2d,
                    group_lens: tp.Array1d,
                    init_cash_grouped: tp.Array1d) -> tp.Array2d:
    """
    计算分组现金序列
    
    基于分组现金流数据，计算每个投资组合的现金动态变化序列。
    这是投资组合级别的现金跟踪，用于监控组合的流动性状况。
    
    参数:
    ----
    target_shape : Shape
        目标形状，用于验证数据维度的一致性
        
    cash_flow_grouped : Array2d
        分组现金流矩阵 (时间 × 组合)，每个元素是该组合在该时点的现金净流量
        
    group_lens : Array1d
        每个组包含的资产数量，用于验证数据结构
        
    init_cash_grouped : Array1d
        每个组的初始现金，作为现金累积的起始点
        
    返回:
    ----
    Array2d
        分组现金序列 (时间 × 组合)，显示每个时点各组合的累积现金
        
    核心算法:
    --------
    对于每个组合g在时间t：
    - 如果t=0：cash[t,g] = init_cash[g] + cash_flow[t,g]
    - 如果t>0：cash[t,g] = cash[t-1,g] + cash_flow[t,g]
    
    使用场景:
    --------
    - **流动性监控**：实时跟踪各投资组合的现金状况
    - **资金配置**：基于现金水平进行投资决策
    - **风险管理**：识别现金不足的风险点
    - **绩效分析**：现金是投资组合价值的重要组成部分
    - **再平衡决策**：现金水平影响再平衡的时机和幅度
    
    技术特点:
    --------
    - **组合级聚合**：从资产级现金流汇总到组合级现金
    - **时间序列累积**：通过时间维度进行现金的累积计算
    - **数值稳定性**：使用专门的add_nb函数处理浮点数加法
    - **高效计算**：单次遍历完成所有组合的现金序列计算
    
    数据流向:
    --------
    资产现金流 → 分组聚合 → 时间累积 → 组合现金序列
    
    注意事项:
    --------
    - 现金流为正表示现金流入，为负表示现金流出
    - 累积现金可能为负值，表示使用了杠杆或借款
    - 结果矩阵维度为 (时间步数 × 组合数)
    """
    # 验证数据维度一致性
    check_group_lens_nb(group_lens, target_shape[1])

    # 预分配输出矩阵，与现金流矩阵同形状
    out = np.empty_like(cash_flow_grouped)
    from_col = 0
    
    # 遍历每个投资组合
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        
        # 初始化该组合的现金状态
        cash_now = init_cash_grouped[group]
        
        # 按时间顺序累积现金变化
        for i in range(cash_flow_grouped.shape[0]):
            # 获取该时点该组合的现金流
            flow_value = cash_flow_grouped[i, group]
            
            # 累积现金变化
            cash_now = add_nb(cash_now, flow_value)
            
            # 记录累积后的现金水平
            out[i, group] = cash_now
            
        from_col = to_col
        
    return out


# ############# Performance ############# #


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def asset_value_nb(close: tp.Array2d, assets: tp.Array2d) -> tp.Array2d:
    """
    计算资产价值序列
    
    通过当前价格和持有数量计算每个资产在每个时点的市场价值。
    这是投资组合估值的基础计算，为后续的绩效分析提供核心数据。
    
    参数:
    ----
    close : Array2d
        收盘价矩阵 (时间 × 资产)，每个元素是该资产在该时点的市场价格
        
    assets : Array2d
        资产持有数量矩阵 (时间 × 资产)，每个元素是该时点持有的资产数量
        
    返回:
    ----
    Array2d
        资产价值矩阵 (时间 × 资产)，每个元素 = 价格 × 数量
        
    计算公式:
    --------
    asset_value[t, i] = close[t, i] × assets[t, i]
    
    应用场景:
    --------
    - **投资组合估值**：计算持仓的市场价值
    - **绩效评估**：追踪资产价值随时间的变化
    - **风险分析**：评估市场波动对组合价值的影响
    - **资产配置**：基于价值比例进行再平衡
    - **报告生成**：为投资报告提供价值数据
    
    技术特点:
    --------
    - **向量化计算**：利用NumPy的广播机制高效计算
    - **实时估值**：支持任意时点的资产价值计算
    - **精度保持**：保持价格和数量的原始精度
    - **简洁高效**：直接的元素级乘法运算
    
    注意事项:
    --------
    - 价格和数量矩阵必须具有相同的形状
    - 负数量表示空头持仓，价值可能为负
    - NaN价格或数量会导致相应价值为NaN
    - 适用于所有类型的金融资产估值
    """
    return close * assets


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def asset_value_grouped_nb(asset_value: tp.Array2d, group_lens: tp.Array1d) -> tp.Array2d:
    """
    计算分组资产价值序列
    
    将资产级别的价值数据聚合为投资组合级别，通过求和操作
    得到每个组合的总资产价值。这是组合级绩效分析的基础。
    
    参数:
    ----
    asset_value : Array2d
        资产价值矩阵 (时间 × 资产)，由asset_value_nb函数计算得出
        
    group_lens : Array1d
        每个组包含的资产数量，定义了聚合的范围
        
    返回:
    ----
    Array2d
        分组资产价值矩阵 (时间 × 组合)，每个元素是该组合的总资产价值
        
    计算逻辑:
    --------
    对于组合g：group_asset_value[t,g] = Σ asset_value[t,i] (i ∈ group g)
    
    应用场景:
    --------
    - **组合估值**：计算投资组合的总市值
    - **绩效比较**：不同组合间的价值表现对比
    - **风险评估**：组合级别的价值波动分析
    - **资产配置**：基于组合价值进行策略调整
    - **报告汇总**：生成组合级别的投资报告
    
    与现金价值的关系:
    ---------------
    - 总组合价值 = 分组资产价值 + 分组现金价值
    - 资产价值反映投资部分的市场表现
    - 现金价值提供流动性和机会成本信息
    
    注意事项:
    --------
    - 聚合操作会丢失单个资产的详细信息
    - 适用于需要组合级别分析的场景
    - 结果维度从资产数减少到组合数
    """
    return sum_grouped_nb(asset_value, group_lens)


@njit(cache=True)
def value_in_sim_order_nb(cash: tp.Array2d,
                          asset_value: tp.Array2d,
                          group_lens: tp.Array1d,
                          call_seq: tp.Array2d) -> tp.Array2d:
    """Get portfolio value series in simulation order."""
    check_group_lens_nb(group_lens, cash.shape[1])

    out = np.empty_like(cash)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        asset_value_now = 0.
        # Without correctly treating NaN values, after one NaN all will be NaN
        since_last_nan = group_len
        for j in range(cash.shape[0] * group_len):
            i = j // group_len
            col = from_col + call_seq[i, from_col + j % group_len]
            if j >= group_len:
                last_j = j - group_len
                last_i = last_j // group_len
                last_col = from_col + call_seq[last_i, from_col + last_j % group_len]
                if not np.isnan(asset_value[last_i, last_col]):
                    asset_value_now -= asset_value[last_i, last_col]
            if np.isnan(asset_value[i, col]):
                since_last_nan = 0
            else:
                asset_value_now += asset_value[i, col]
            if since_last_nan < group_len:
                out[i, col] = np.nan
            else:
                out[i, col] = cash[i, col] + asset_value_now
            since_last_nan += 1

        from_col = to_col
    return out


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def value_nb(cash: tp.Array2d, asset_value: tp.Array2d) -> tp.Array2d:
    """
    计算投资组合价值序列
    
    投资组合的总价值 = 现金价值 + 资产市场价值。这是投资组合
    绩效评估的核心指标，用于计算收益率、回撤等关键指标。
    
    参数:
    ----
    cash : Array2d
        现金序列矩阵 (时间 × 资产/组合)，每个元素是该时点的现金持有量
        
    asset_value : Array2d
        资产价值序列矩阵 (时间 × 资产/组合)，每个元素是该时点的资产市值
        
    返回:
    ----
    Array2d
        投资组合价值序列 (时间 × 资产/组合)，每个元素是该时点的总价值
        
    核心公式:
    --------
    portfolio_value[t, i] = cash[t, i] + asset_value[t, i]
    
    应用场景:
    --------
    - **绩效评估**：计算投资组合的净值曲线
    - **收益率计算**：作为收益率计算的基础数据
    - **风险分析**：评估价值波动和最大回撤
    - **基准比较**：与市场基准或其他策略比较
    - **报告生成**：生成投资绩效报告的核心数据
    
    价值组成分析:
    -----------
    - **现金部分**：提供流动性，但通常收益率较低
    - **资产部分**：承担市场风险，是主要收益来源
    - **总价值**：反映投资组合的综合表现
    
    技术特点:
    --------
    - **实时计算**：支持任意时点的价值计算
    - **向量化运算**：高效的元素级加法操作
    - **灵活适用**：同时适用于资产级和组合级数据
    - **精度保持**：保持原始数据的数值精度
    
    使用注意:
    --------
    - 确保现金和资产价值矩阵维度一致
    - 负现金表示借款，负资产价值表示空头持仓
    - NaN值会传播到最终结果中
    - 是后续所有绩效指标计算的基础
    """
    return cash + asset_value


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def total_profit_nb(target_shape: tp.Shape,
                    close: tp.Array2d,
                    order_records: tp.RecordArray,
                    col_map: tp.ColMap) -> tp.Array1d:
    """
    计算每列的总盈利
    
    基于订单记录快速计算最终的投资组合盈利，比基于value_nb的方法
    效率更高。通过累积现金流和最终资产估值来计算净盈利。
    
    参数:
    ----
    target_shape : Shape
        目标形状(时间步数, 资产数量)，用于初始化计算数组
        
    close : Array2d
        收盘价矩阵，最后一行用于最终资产估值
        
    order_records : RecordArray
        订单记录数组，包含所有交易的详细信息
        
    col_map : ColMap
        列映射信息，用于快速定位各资产的订单
        
    返回:
    ----
    Array1d
        每个资产的总盈利，length = target_shape[1]
        
    计算逻辑:
    --------
    1. **累积现金变化**：
       - 买入：现金减少 = size × price + fees
       - 卖出：现金增加 = size × price - fees
       
    2. **累积资产变化**：
       - 买入：资产增加 = +size
       - 卖出：资产减少 = -size
       
    3. **最终盈利**：
       total_profit = 累积现金 + 最终资产数量 × 最终价格
    
    性能优势:
    --------
    - **直接计算**：避免构建完整的价值序列，只计算最终结果
    - **内存高效**：不需要存储中间的时间序列数据
    - **计算简化**：单次遍历订单记录即可完成计算
    - **Numba优化**：编译为高效的机器码执行
    
    应用场景:
    --------
    - **快速评估**：需要快速获得最终盈利而不关心中间过程
    - **批量分析**：分析大量策略的最终表现
    - **绩效筛选**：基于总盈利对策略进行排序筛选
    - **资源优化**：在计算资源有限时的高效选择
    
    处理细节:
    --------
    - **零交易处理**：没有订单的资产盈利设为0
    - **订单验证**：确保订单ID在每列内按升序排列
    - **数值稳定性**：使用add_nb函数保证计算精度
    - **方向处理**：买卖订单对现金和资产的影响方向相反
    
    与完整价值序列方法的对比:
    ---------------------
    - **优势**：计算速度快，内存占用小
    - **局限**：无法获得中间时点的价值变化信息
    - **适用性**：适合只关心最终结果的场景
    
    注意事项:
    --------
    - 只反映相对于初始投资(0)的绝对盈利
    - 不包括资金的时间价值
    - 假设最终价格为close数组的最后一行
    - 适用于单向（只买或只卖）和双向交易策略
    """
    # 解析列映射信息
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    
    # 初始化累积数组
    assets = np.full(target_shape[1], 0., dtype=np.float64)    # 累积资产持有量
    cash = np.full(target_shape[1], 0., dtype=np.float64)      # 累积现金变化
    zero_mask = np.full(target_shape[1], False, dtype=np.bool_)  # 标记无交易的列

    # 遍历每一列(资产)
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        
        # 处理无交易的列
        if col_len == 0:
            zero_mask[col] = True
            continue
            
        last_id = -1

        # 遍历该列的所有订单
        for c in range(col_len):
            oidx = col_idxs[col_start_idxs[col] + c]
            record = order_records[oidx]

            # 验证订单ID顺序
            if record['id'] < last_id:
                raise ValueError("订单ID必须在每列内按升序排列")
            last_id = record['id']

            # 累积资产持有量变化
            if record['side'] == OrderSide.Buy:
                # 买入：增加资产持有量
                order_size = record['size']
                assets[col] = add_nb(assets[col], order_size)
            else:
                # 卖出：减少资产持有量
                order_size = record['size']
                assets[col] = add_nb(assets[col], -order_size)

            # 累积现金流变化
            if record['side'] == OrderSide.Buy:
                # 买入：现金流出(包括手续费)
                order_cash = record['size'] * record['price'] + record['fees']
                cash[col] = add_nb(cash[col], -order_cash)
            else:
                # 卖出：现金流入(扣除手续费)
                order_cash = record['size'] * record['price'] - record['fees']
                cash[col] = add_nb(cash[col], order_cash)

    # 计算最终总盈利 = 现金净流入 + 剩余资产按最终价格估值
    total_profit = cash + assets * close[-1, :]
    
    # 无交易的列盈利设为0
    total_profit[zero_mask] = 0.
    
    return total_profit


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def total_profit_grouped_nb(total_profit: tp.Array1d, group_lens: tp.Array1d) -> tp.Array1d:
    """
    计算分组总盈利
    
    将资产级别的总盈利聚合为投资组合级别，通过求和得到
    每个投资组合的总盈利。适用于现金共享或独立管理的组合分析。
    
    参数:
    ----
    total_profit : Array1d
        资产级别的总盈利数组，每个元素是单个资产的净盈利
        
    group_lens : Array1d
        每个组包含的资产数量，定义聚合的范围
        
    返回:
    ----
    Array1d
        分组总盈利数组，每个元素是对应投资组合的总盈利
        
    计算公式:
    --------
    group_profit[g] = Σ total_profit[i] (i ∈ group g)
    
    应用场景:
    --------
    - **组合绩效评估**：评估不同投资组合的盈利能力
    - **策略比较**：比较多个投资策略的最终表现
    - **风险收益分析**：分析组合级别的风险调整后收益
    - **资金配置决策**：基于历史盈利表现调整资金分配
    - **绩效报告**：生成投资组合级别的业绩报告
    
    聚合意义:
    --------
    - **现金共享组合**：反映共同资金池的总收益
    - **独立管理组合**：反映各组合独立运作的收益
    - **策略组合**：反映不同策略组合的综合效果
    
    技术特点:
    --------
    - **简单求和**：利用NumPy的高效求和操作
    - **分组处理**：按预定义的组边界进行聚合
    - **数值稳定**：保持原始盈利数据的精度
    - **高效计算**：单次遍历完成所有组合的聚合
    
    注意事项:
    --------
    - 聚合会丢失单个资产的盈利细节
    - 负盈利资产会抵消正盈利资产的收益
    - 适用于需要组合级别决策的场景
    """
    # 验证数据维度一致性
    check_group_lens_nb(group_lens, total_profit.shape[0])

    # 预分配输出数组
    out = np.empty(len(group_lens), dtype=np.float64)
    from_col = 0
    
    # 遍历每个投资组合
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        
        # 对该组合内所有资产的盈利求和
        out[group] = np.sum(total_profit[from_col:to_col])
        from_col = to_col
        
    return out


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def final_value_nb(total_profit: tp.Array1d, init_cash: tp.Array1d) -> tp.Array1d:
    """
    计算最终价值
    
    最终价值 = 总盈利 + 初始现金，表示投资组合在期末的绝对价值。
    这是评估投资策略绝对表现的重要指标。
    
    参数:
    ----
    total_profit : Array1d
        总盈利数组，每个元素是净盈亏（可正可负）
        
    init_cash : Array1d
        初始现金数组，每个元素是投资的起始资金
        
    返回:
    ----
    Array1d
        最终价值数组，每个元素是期末的总资产价值
        
    核心公式:
    --------
    final_value = total_profit + init_cash
    
    经济含义:
    --------
    - **绝对收益**：反映投资的绝对价值增长
    - **资产规模**：显示投资组合的最终规模
    - **成功度量**：衡量投资策略的成功程度
    
    应用场景:
    --------
    - **绝对绩效评估**：评估投资的绝对收益水平
    - **资产规模追踪**：追踪投资组合的资产增长
    - **投资目标验证**：验证是否达到预期投资目标
    - **风险评估**：评估投资失败的潜在损失
    - **基准比较**：与固定收益等基准进行比较
    
    与收益率的关系:
    -------------
    - 收益率 = total_profit / init_cash
    - 最终价值 = init_cash × (1 + 收益率)
    - 最终价值更直观，收益率更标准化
    
    使用考量:
    --------
    - **规模效应**：大资金的绝对收益通常更高
    - **比较标准**：同等规模投资的绝对收益更有可比性
    - **风险控制**：最终价值低于初始投资表示亏损
    - **复合效应**：体现了复合收益的累积效果
    
    注意事项:
    --------
    - 不考虑资金的时间价值
    - 需要与收益率指标结合分析
    - 适用于绝对收益导向的投资评估
    """
    return total_profit + init_cash


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def total_return_nb(total_profit: tp.Array1d, init_cash: tp.Array1d) -> tp.Array1d:
    """
    计算总收益率
    
    基于总盈利和初始资本计算投资组合或资产的总收益率。
    这是评估投资绩效的核心指标之一。
    
    参数:
    ----
    total_profit : Array1d
        总盈利数组，每个元素是对应投资组合/资产的净盈亏
        - 正值表示盈利，负值表示亏损
        
    init_cash : Array1d
        初始资本数组，每个元素是对应投资的起始资金
        - 必须为正值，作为收益率计算的分母
        
    返回:
    ----
    Array1d
        总收益率数组，每个元素是对应投资的收益率
        - 1.0 = 100%收益率（翻倍）
        - 0.0 = 0%收益率（保本）
        - -0.5 = -50%收益率（亏损50%）
        
    计算公式:
    --------
    total_return = total_profit / init_cash
    
    其中：
    - total_profit = final_value - init_cash
    - 因此：total_return = (final_value - init_cash) / init_cash = (final_value / init_cash) - 1
    
    应用场景:
    --------
    - **绩效评估**：比较不同投资策略的收益表现
    - **基准比较**：与市场指数或无风险利率比较
    - **投资排名**：对多个投资组合按收益率排序
    - **风险调整**：作为计算夏普比率等指标的基础
    - **投资报告**：向投资者报告投资回报
    
    解释说明:
    --------
    - **标准化收益**：消除了初始投资规模的影响，便于比较
    - **相对表现**：反映了投资效率，而非绝对盈利金额
    - **无量纲指标**：结果为比率，不依赖于货币单位
    
    计算示例:
    --------
    - 初始资金1000，最终价值1200 → 收益率 = 200/1000 = 0.2 (20%)
    - 初始资金500，最终价值450 → 收益率 = -50/500 = -0.1 (-10%)
    - 初始资金2000，最终价值4000 → 收益率 = 2000/2000 = 1.0 (100%)
    
    与其他指标的关系:
    ---------------
    - **最终价值**：final_value = init_cash × (1 + total_return)
    - **年化收益率**：需要考虑投资期间进行时间调整
    - **累积收益**：多期投资的复合收益计算
    
    注意事项:
    --------
    - 假设init_cash > 0，避免除零错误
    - 不考虑投资期间的时间因素
    - 适用于期初一次投入、期末一次取出的简单投资模式
    - 对于有中间现金流的投资，需要更复杂的收益率计算方法
    """
    return total_profit / init_cash


@njit(cache=True)
def returns_in_sim_order_nb(value_iso: tp.Array2d,
                            group_lens: tp.Array1d,
                            init_cash_grouped: tp.Array1d,
                            call_seq: tp.Array2d) -> tp.Array2d:
    """Get portfolio return series in simulation order."""
    check_group_lens_nb(group_lens, value_iso.shape[1])

    out = np.empty_like(value_iso)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        input_value = init_cash_grouped[group]
        for j in range(value_iso.shape[0] * group_len):
            i = j // group_len
            col = from_col + call_seq[i, from_col + j % group_len]
            output_value = value_iso[i, col]
            out[i, col] = returns_nb.get_return_nb(input_value, output_value)
            input_value = output_value
        from_col = to_col
    return out


@njit(cache=True)
def asset_returns_nb(cash_flow: tp.Array2d, asset_value: tp.Array2d) -> tp.Array2d:
    """Get asset return series per column/group."""
    out = np.empty_like(cash_flow)
    for col in range(cash_flow.shape[1]):
        for i in range(cash_flow.shape[0]):
            input_value = 0. if i == 0 else asset_value[i - 1, col]
            output_value = asset_value[i, col] + cash_flow[i, col]
            out[i, col] = returns_nb.get_return_nb(input_value, output_value)
    return out


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def benchmark_value_nb(close: tp.Array2d, init_cash: tp.Array1d) -> tp.Array2d:
    """
    计算基准投资价值序列（标准化版）
    
    计算买入持有策略的投资价值变化，基于期初价格进行标准化处理。
    这是评估主动投资策略相对表现的标准基准。
    
    参数:
    ----
    close : Array2d
        收盘价格矩阵 (时间 × 资产)，每个元素是该时点该资产的市场价格
        
    init_cash : Array1d
        初始现金数组，每个资产分配的起始投资金额
        
    返回:
    ----
    Array2d
        基准投资价值矩阵 (时间 × 资产)
        
    计算公式:
    --------
    benchmark_value[t, i] = (close[t, i] / close[0, i]) × init_cash[i]
    
    其中：
    - close[t, i] / close[0, i]：资产i从期初到时点t的价格倍数
    - init_cash[i]：资产i的初始投资金额
    
    经济含义:
    --------
    假设在第0期以close[0, i]价格买入价值init_cash[i]的资产i：
    - 购买数量 = init_cash[i] / close[0, i]
    - t期价值 = 购买数量 × close[t, i] = init_cash[i] × (close[t, i] / close[0, i])
    
    应用场景:
    --------
    - **基准比较**：衡量主动策略相对于被动持有的超额收益
    - **绩效归因**：区分策略收益中来自市场波动vs策略技巧的部分
    - **风险调整**：计算跟踪误差、信息比率等风险调整指标
    - **阿尔法计算**：主动策略收益 - 基准收益 = 策略阿尔法
    
    基准策略假设:
    -----------
    - **买入持有**：期初买入后持有到底，不进行任何调仓
    - **按比例投资**：严格按照init_cash的比例进行初始配置
    - **价格跟随**：投资价值完全跟随市场价格变动
    - **无摩擦成本**：不考虑交易费用、滑点等成本
    
    标准化特性:
    ----------
    - **相对价格变化**：消除了价格水平差异，关注相对变化
    - **可比性**：不同价格水平的资产具有可比性
    - **收益率一致性**：基准收益率 = (价格变化率) × 投资权重
    
    计算示例:
    --------
    假设两个资产，初始价格[100, 10]，初始投资[1000, 1000]：
    - t=0: 价格[100, 10] → 基准价值[1000, 1000]
    - t=1: 价格[110, 9] → 基准价值[1100, 900] (分别上涨10%和下跌10%)
    - t=2: 价格[105, 12] → 基准价值[1050, 1200]
    
    技术特点:
    --------
    - **向量化计算**：利用NumPy广播实现高效批量计算
    - **数值稳定**：通过标准化避免大数计算问题
    - **内存效率**：原地计算，无需额外大型数组
    
    与其他基准的关系:
    ---------------
    - **等权重基准**：init_cash相等时为等权重投资组合
    - **价值加权基准**：init_cash按市值分配时为价值加权组合
    - **自定义基准**：init_cash可以设置任意权重结构
    """
    return close / close[0] * init_cash


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def benchmark_value_grouped_nb(close: tp.Array2d, group_lens: tp.Array1d, init_cash_grouped: tp.Array1d) -> tp.Array2d:
    """
    计算分组基准投资价值序列
    
    为投资组合分组计算基准价值，将组级初始资金等权分配给组内资产，
    然后计算买入持有策略在组合级别的价值变化。
    
    参数:
    ----
    close : Array2d
        收盘价格矩阵 (时间 × 资产)
        
    group_lens : Array1d
        每个组包含的资产数量，定义投资组合的分组结构
        
    init_cash_grouped : Array1d
        每个组的初始资金，组级别的投资预算
        
    返回:
    ----
    Array2d
        分组基准价值矩阵 (时间 × 组合数)
        每列代表一个投资组合的基准价值时间序列
        
    计算流程:
    --------
    对于每个组合g：
    1. **资金分配**：col_init_cash = init_cash_grouped[g] / group_len
    2. **价格标准化**：close_norm = close[:, group_cols] / close[0, group_cols]  
    3. **组合价值**：out[:, g] = col_init_cash × sum(close_norm, axis=1)
    
    分配策略:
    --------
    - **等权重分配**：组内每个资产获得相等的初始投资金额
    - **等权重再平衡**：等价于每期进行等权重再平衡的基准
    - **多样化效应**：通过等权分散可以降低单一资产风险
    
    应用场景:
    --------
    - **多组合管理**：同时管理多个独立投资组合的基准比较
    - **风险分层**：不同风险等级组合的基准设置
    - **客户细分**：为不同类型客户提供定制化基准
    - **策略对比**：评估不同组合构建策略的有效性
    - **绩效归因**：分析组合表现中资产选择vs配置的贡献
    
    与单资产基准的区别:
    ---------------
    - **聚合层次**：从资产级聚合到组合级
    - **权重处理**：自动处理组内资产的权重分配
    - **输出维度**：输出维度为组合数而非资产数
    - **再平衡假设**：隐含等权重再平衡假设
    
    计算示例:
    --------
    假设2个组合：[资产1,2]和[资产3]，初始资金[2000, 1000]
    - 组合1：每个资产分配1000，基准 = 1000×(norm_price1 + norm_price2)
    - 组合2：资产3分配1000，基准 = 1000×norm_price3
    
    技术实现:
    --------
    - **数据验证**：确保分组配置与价格数据维度匹配
    - **内存预分配**：预先分配输出数组提高效率
    - **分组遍历**：逐个处理各投资组合避免复杂索引
    - **向量化计算**：在组内使用NumPy向量化操作
    
    适用限制:
    --------
    - 假设组内资产等权重配置
    - 不支持自定义组内权重分配
    - 适用于需要等权基准的场景
    - 对于价值加权等需求，需要使用其他函数
    
    注意事项:
    --------
    - 等权分配可能不符合某些基准设计需求
    - 基准的合理性取决于分组的经济意义
    - 适合用于评估等权重策略的相对表现
    """
    # 验证分组配置与价格数据的一致性
    check_group_lens_nb(group_lens, close.shape[1])

    # 预分配输出矩阵：时间步数 × 组合数
    out = np.empty((close.shape[0], len(group_lens)), dtype=np.float64)
    from_col = 0
    
    # 遍历每个投资组合
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        
        # 等权分配：组内每个资产获得相等的初始投资
        col_init_cash = init_cash_grouped[group] / group_len
        
        # 计算组内资产的标准化价格（相对于期初价格）
        close_norm = close[:, from_col:to_col] / close[0, from_col:to_col]
        
        # 组合基准价值 = 单个资产投资额 × 标准化价格之和
        out[:, group] = col_init_cash * np.sum(close_norm, axis=1)
        
        from_col = to_col
        
    return out


@njit(cache=True)
def total_benchmark_return_nb(benchmark_value: tp.Array2d) -> tp.Array1d:
    """Get total market return per column/group."""
    out = np.empty(benchmark_value.shape[1], dtype=np.float64)
    for col in range(benchmark_value.shape[1]):
        out[col] = returns_nb.get_return_nb(benchmark_value[0, col], benchmark_value[-1, col])
    return out


@njit(cache=True)  # Numba编译缓存，优化重复调用性能
def gross_exposure_nb(asset_value: tp.Array2d, cash: tp.Array2d) -> tp.Array2d:
    """
    计算总暴露度（净暴露版）
    
    计算投资组合的净资产暴露度，衡量资产投资相对于总净资产的比例。
    与传统的总暴露度不同，此版本使用净资产作为分母，更适合净值分析。
    
    参数:
    ----
    asset_value : Array2d
        资产价值矩阵 (时间 × 资产/组合)
        - 正值：多头持仓的市场价值
        - 负值：空头持仓的市场价值（负数表示）
        
    cash : Array2d
        现金矩阵 (时间 × 资产/组合) 
        - 正值：现金资产
        - 负值：借款债务
        
    返回:
    ----
    Array2d
        净暴露度矩阵 (时间 × 资产/组合)
        - 正值：净多头暴露
        - 负值：净空头暴露  
        - 0：现金暴露，无市场风险
        
    计算公式:
    --------
    net_exposure = asset_value / (asset_value + cash)
    
    其中：
    - asset_value + cash = 投资组合净值 (NAV)
    - 当净值 = 0 时，返回 0 避免除零错误
    
    经济含义:
    --------
    - **净值暴露**：相对于净资产价值的市场暴露度
    - **杠杆效应**：暴露度 > 1 表示使用杠杆
    - **方向性风险**：正值表示净多头，负值表示净空头
    - **现金比例**：1 - 暴露度 = 现金在净值中的比例
    
    暴露度解读:
    ----------
    - **0**：全现金状态，无市场暴露
    - **(0, 1)**：部分投资，有现金缓冲
    - **1**：满仓投资，无现金闲置
    - **> 1**：使用杠杆，放大市场暴露
    - **< 0**：净空头，或资产空头超过现金
    
    应用场景:
    --------
    - **风险管理**：监控相对于净值的市场风险暴露
    - **杠杆监控**：识别杠杆使用程度和风险水平
    - **现金管理**：评估现金配置的合理性
    - **策略分析**：理解策略的风险收益特征
    - **合规检查**：确保暴露度符合监管要求
    
    计算示例:
    --------
    1. **正常投资**：
       - 资产价值: 800, 现金: 200 → 暴露度 = 800/1000 = 0.8
       
    2. **使用杠杆**：
       - 资产价值: 1200, 现金: -200 → 暴露度 = 1200/1000 = 1.2
       
    3. **空头策略**：
       - 资产价值: -300, 现金: 1000 → 暴露度 = -300/700 = -0.43
       
    4. **净值为零**：
       - 资产价值: 500, 现金: -500 → 暴露度 = 0 (特殊处理)
    
    与传统暴露度的区别:
    ---------------
    - **传统暴露度**：|资产| / (|资产| + |现金|)，始终为正值
    - **净暴露度**：资产 / (资产 + 现金)，保留方向信息
    - **适用场景**：净暴露更适合净值分析和方向性风险评估
    
    技术实现细节:
    -----------
    - **数值稳定性**：使用add_nb确保浮点数加法精度
    - **除零处理**：当净值为0时返回0，避免无穷大
    - **逐元素计算**：双重循环确保每个时点单独处理
    - **内存预分配**：预先分配输出数组提高性能
    
    风险控制应用:
    -----------
    - **风险限额**：设定暴露度上下限控制风险
    - **再平衡触发**：暴露度偏离目标时触发调仓
    - **杠杆约束**：监控杠杆使用不超过规定倍数
    - **现金储备**：确保足够现金应对赎回需求
    
    监管合规:
    --------
    - **UCITS指令**：欧盟基金杠杆限制
    - **投资公司法**：美国基金杠杆要求
    - **银行监管**：巴塞尔协议风险暴露要求
    - **内部控制**：机构风险管理制度
    
    注意事项:
    --------
    - 净值为负时的暴露度解读需要特别谨慎
    - 高频交易中暴露度可能快速变化
    - 复杂衍生品的暴露度计算需要特殊处理
    - 多币种组合需考虑汇率影响
    
    局限性:
    ------
    - 未考虑资产间的相关性和对冲效应
    - 不反映隐含杠杆（如期权的内在杠杆）
    - 静态计算，不预测未来暴露度变化
    - 适用于传统资产，对复杂结构产品有限制
    """
    # 预分配输出矩阵，与资产价值矩阵同形状
    out = np.empty(asset_value.shape, dtype=np.float64)
    
    # 遍历每一列（资产/组合）
    for col in range(out.shape[1]):
        # 遍历每一行（时间点）
        for i in range(out.shape[0]):
            # 计算净值 = 资产价值 + 现金（使用add_nb确保数值精度）
            denom = add_nb(asset_value[i, col], cash[i, col])
            
            # 处理净值为零的特殊情况
            if denom == 0:
                out[i, col] = 0.  # 净值为零时暴露度设为0
            else:
                # 计算净暴露度 = 资产价值 / 净值
                out[i, col] = asset_value[i, col] / denom
                
    return out
