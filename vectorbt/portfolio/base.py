# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)
# 版权所有 (c) 2021 Oleg Polakow. 保留所有权利.
# 此代码基于Apache 2.0许可证和Commons Clause许可证发布（详细信息请参见LICENSE.md）

"""
投资组合建模和绩效测量的基础类模块

本模块是vectorbt量化投资框架的核心组件，提供了Portfolio类用于：
1. 建模投资组合性能和计算各种风险与绩效指标
2. 使用Numba编译函数进行高性能计算
3. 基于记录类评估订单、日志、交易、仓位和回撤等事件

设计逻辑：
- Portfolio类负责创建资产配置序列，生成权益曲线
- 纳入基本交易成本，产生绩效统计信息
- 输出仓位/利润指标和回撤信息
- 支持三种主要模拟模式：从订单、从信号、从订单函数
"""
# 导入必要的库和模块
import warnings  # 用于发出警告信息

import numpy as np  # 数值计算库，提供高性能数组操作
import pandas as pd  # 数据分析库，提供DataFrame和Series数据结构

# 从vectorbt导入类型提示相关模块
from vectorbt import _typing as tp  # vectorbt的类型提示定义

# 导入基础模块：数组包装器和包装基类
from vectorbt.base.array_wrapper import ArrayWrapper, Wrapping  # 数组包装器和包装基类
from vectorbt.base.reshape_fns import to_1d_array, to_2d_array, broadcast, broadcast_to, to_pd_array  # 数组形状重塑函数

# 导入通用模块：回撤分析和构建器混合类
from vectorbt.generic.drawdowns import Drawdowns  # 回撤分析模块
from vectorbt.generic.plots_builder import PlotsBuilderMixin  # 绘图构建器混合类
from vectorbt.generic.stats_builder import StatsBuilderMixin  # 统计构建器混合类

# 导入投资组合核心模块
from vectorbt.portfolio import nb  # Numba编译的投资组合函数
from vectorbt.portfolio.decorators import attach_returns_acc_methods  # 投资组合装饰器
from vectorbt.portfolio.enums import *  # 投资组合枚举定义
from vectorbt.portfolio.logs import Logs  # 日志模块
from vectorbt.portfolio.orders import Orders  # 订单模块
from vectorbt.portfolio.trades import Trades, EntryTrades, ExitTrades, Positions  # 交易模块

# 导入收益计算模块
from vectorbt.returns import nb as returns_nb  # 收益计算的Numba函数
from vectorbt.returns.accessors import ReturnsAccessor  # 收益访问器

# 导入信号生成器
from vectorbt.signals.generators import RANDNX, RPROBNX  # 随机信号生成器

# 导入工具模块
from vectorbt.utils import checks  # 检查工具函数
from vectorbt.utils.colors import adjust_opacity  # 颜色调整工具
from vectorbt.utils.config import merge_dicts, Config  # 配置工具
from vectorbt.utils.decorators import cached_property, cached_method  # 缓存装饰器
from vectorbt.utils.enum_ import map_enum_fields  # 枚举映射工具
from vectorbt.utils.figure import get_domain  # 绘图工具
from vectorbt.utils.random_ import set_seed  # 随机种子设置
from vectorbt.utils.template import RepEval, deep_substitute  # 模板工具

# 尝试导入quantstats库（可选的量化分析库）
try:
    import quantstats as qs  # QuantStats是一个用于量化绩效分析的Python库
except ImportError:
    QSAdapterT = tp.Any  # 如果导入失败，设为Any类型
else:
    from vectorbt.returns.qs_adapter import QSAdapter as QSAdapterT  # 导入QuantStats适配器

__pdoc__ = {}  # 文档字典初始化，用于存储文档字符串

# 收益访问器配置：定义要添加到Portfolio类的收益计算方法
returns_acc_config = Config(
    {
        # 日收益率计算配置
        'daily_returns': dict(source_name='daily'),  # 日收益率
        'annual_returns': dict(source_name='annual'),  # 年收益率
        'cumulative_returns': dict(source_name='cumulative'),  # 累计收益率
        'annualized_return': dict(source_name='annualized'),  # 年化收益率
        
        # 风险指标配置
        'annualized_volatility': dict(),  # 年化波动率
        'calmar_ratio': dict(),  # 卡尔玛比率（年化收益/最大回撤）
        'omega_ratio': dict(),  # 欧米茄比率（收益概率加权/损失概率加权）
        'sharpe_ratio': dict(),  # 夏普比率（风险调整收益）
        'deflated_sharpe_ratio': dict(),  # 紧缩夏普比率
        'downside_risk': dict(),  # 下行风险
        'sortino_ratio': dict(),  # 索提诺比率（下行风险调整收益）
        'information_ratio': dict(),  # 信息比率
        
        # 市场敏感性指标
        'beta': dict(),  # 贝塔系数（系统性风险）
        'alpha': dict(),  # 阿尔法系数（超额收益）
        
        # 极端风险指标  
        'tail_ratio': dict(),  # 尾部比率
        'value_at_risk': dict(),  # 风险价值（VaR）
        'cond_value_at_risk': dict(),  # 条件风险价值（CVaR）
        
        # 捕获率指标
        'capture': dict(),  # 捕获率
        'up_capture': dict(),  # 上行捕获率
        'down_capture': dict(),  # 下行捕获率
        
        # 回撤指标
        'drawdown': dict(),  # 回撤
        'max_drawdown': dict()  # 最大回撤
    },
    readonly=True,  # 只读配置，防止意外修改
    as_attrs=False  # 不作为属性访问
)
"""收益访问器配置字典，定义了Portfolio类中可用的收益和风险计算方法"""

# 为returns_acc_config生成文档
__pdoc__['returns_acc_config'] = f"""要添加到Portfolio类的收益访问器方法配置。

这个配置字典定义了Portfolio类中可用的各种收益率和风险指标计算方法。
每个键对应一个可以在Portfolio实例上调用的方法名。

```json
{returns_acc_config.to_doc()}
```
"""

# 定义Portfolio类的类型变量，用于类型提示
PortfolioT = tp.TypeVar("PortfolioT", bound="Portfolio")


class MetaPortfolio(type(StatsBuilderMixin), type(PlotsBuilderMixin)):
    """
    Portfolio类的元类
    
    继承自StatsBuilderMixin和PlotsBuilderMixin的元类，
    用于提供统计分析和绘图功能的类级别支持。
    """
    pass


@attach_returns_acc_methods(returns_acc_config)  # 装饰器：自动附加收益访问器方法
class Portfolio(Wrapping, StatsBuilderMixin, PlotsBuilderMixin, metaclass=MetaPortfolio):
    """
    投资组合建模和性能测量的核心类
    
    这是vectorbt量化投资框架的核心组件，提供完整的投资组合建模、
    回测、绩效评估和可视化功能。
    
    主要功能模块：
    1. 投资组合构建：支持从订单、信号、自定义函数构建投资组合
    2. 绩效计算：提供各种风险和收益指标（夏普比率、最大回撤等）
    3. 回测功能：支持现实的交易成本、滑点、资金管理等
    4. 数据管理：高效的内存使用和缓存机制
    5. 可视化：丰富的图表和报告功能
    
    Class for modeling portfolio and measuring its performance.

    参数说明 (Args):
        wrapper (ArrayWrapper): 数组包装器
            负责管理数据的索引、形状、分组等操作。
            参见 `vectorbt.base.array_wrapper.ArrayWrapper`。
            
        close (array_like): 每个时间步的最后资产价格
            用于计算未实现盈亏和投资组合价值。
            
        order_records (array_like): 订单记录的结构化NumPy数组
            包含所有已执行订单的详细信息，如价格、数量、费用等。
            
        log_records (array_like): 日志记录的结构化NumPy数组
            包含模拟过程中的详细日志信息，用于调试和分析。
            
        init_cash (InitCashMode, float or array_like of float): 初始资本
            投资组合的起始资金。支持自动对齐模式和手动设置。
            
        cash_sharing (bool): 是否在同一组内共享现金
            启用后允许组内不同资产间的资金调配。
            
        call_seq (array_like of int): 每行每组的调用序列，默认为None
            控制订单在同一时间步内的执行顺序。
            
        fillna_close (bool): 是否前向和后向填充close中的NaN值
            在模拟后应用，以避免资产价值计算中的NaN值。
            参见 `Portfolio.get_filled_close`。
            
        trades_type (str or int): 默认的交易类型
            指定在整个Portfolio中使用的默认交易类型。
            参见 `vectorbt.portfolio.enums.TradesType`。

    默认值配置参见 `vectorbt._settings.settings` 中的 'portfolio' 部分。

    !!! note
        推荐使用带有 `from_` 前缀的类方法来构建投资组合。
        `__init__` 方法主要用于索引操作。

    !!! note
        此类设计为不可变的。要更改任何属性，请使用 `Portfolio.replace` 方法。
    """

    def __init__(self,
                 wrapper: ArrayWrapper,  # 数组包装器，管理索引和形状
                 close: tp.ArrayLike,  # 收盘价数组
                 order_records: tp.RecordArray,  # 订单记录数组
                 log_records: tp.RecordArray,  # 日志记录数组
                 init_cash: tp.ArrayLike,  # 初始现金
                 cash_sharing: bool,  # 现金共享标志
                 call_seq: tp.Optional[tp.Array2d] = None,  # 调用序列（可选）
                 fillna_close: tp.Optional[bool] = None,  # 填充收盘价NaN的标志（可选）
                 trades_type: tp.Optional[tp.Union[int, str]] = None  # 交易类型（可选）
                 ) -> None:
        """
        初始化Portfolio实例
        
        注意：这个方法主要用于内部使用和索引操作。
        推荐使用类方法如from_orders(), from_signals()等来创建Portfolio实例。
        """
        # 初始化父类：Wrapping类负责基本的数组包装功能
        Wrapping.__init__(
            self,
            wrapper,
            close=close,
            order_records=order_records,
            log_records=log_records,
            init_cash=init_cash,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            fillna_close=fillna_close,
            trades_type=trades_type
        )
        # 初始化统计构建器混合类，提供统计计算功能
        StatsBuilderMixin.__init__(self)
        # 初始化绘图构建器混合类，提供可视化功能
        PlotsBuilderMixin.__init__(self)

        # 获取默认配置参数
        from vectorbt._settings import settings
        portfolio_cfg = settings['portfolio']  # 获取投资组合相关的全局配置

        # 设置默认值：如果参数未指定，则从全局配置中获取
        if fillna_close is None:
            fillna_close = portfolio_cfg['fillna_close']  # 获取fillna_close的默认值
        if trades_type is None:
            trades_type = portfolio_cfg['trades_type']  # 获取trades_type的默认值
        if isinstance(trades_type, str):
            # 如果trades_type是字符串，则映射为对应的枚举值
            trades_type = map_enum_fields(trades_type, TradesType)

        # 存储传递的参数为私有属性
        # 将收盘价广播到正确的形状（不分组）
        self._close = broadcast_to(close, wrapper.dummy(group_by=False))
        self._order_records = order_records  # 存储订单记录
        self._log_records = log_records  # 存储日志记录
        self._init_cash = init_cash  # 存储初始现金
        self._cash_sharing = cash_sharing  # 存储现金共享标志
        self._call_seq = call_seq  # 存储调用序列
        self._fillna_close = fillna_close  # 存储填充标志
        self._trades_type = trades_type  # 存储交易类型

    def indexing_func(self: PortfolioT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> PortfolioT:
        """
        对Portfolio对象执行索引操作
        
        这个方法允许像处理pandas对象一样对Portfolio进行切片和索引操作，
        例如pf['AAPL']或pf.iloc[0]等。索引操作会相应地调整所有相关的数据结构。
        
        Args:
            pd_indexing_func: pandas索引函数
            **kwargs: 传递给索引函数的额外参数
            
        Returns:
            PortfolioT: 经过索引操作后的新Portfolio实例
            
        示例:
            >>> # 选择特定资产
            >>> pf_btc = pf['BTC-USD']  # 选择BTC的投资组合
            >>> # 选择多个资产
            >>> pf_crypto = pf[['BTC-USD', 'ETH-USD']]  # 选择加密货币投资组合
        """
        # 获取索引操作的元数据信息：新包装器、组索引、列索引
        new_wrapper, _, group_idxs, col_idxs = \
            self.wrapper.indexing_func_meta(pd_indexing_func, column_only_select=True, **kwargs)
        
        # 根据列索引选择收盘价数据
        new_close = new_wrapper.wrap(to_2d_array(self.close)[:, col_idxs], group_by=False)
        
        # 根据列索引选择订单记录
        new_order_records = self.orders.get_by_col_idxs(col_idxs)
        
        # 根据列索引选择日志记录
        new_log_records = self.logs.get_by_col_idxs(col_idxs)
        
        # 处理初始现金：如果是整数（枚举模式），保持不变；否则根据索引选择
        if isinstance(self._init_cash, int):
            new_init_cash = self._init_cash  # 枚举模式，保持原值
        else:
            # 根据现金共享设置选择适当的索引（组索引或列索引）
            new_init_cash = to_1d_array(self._init_cash)[group_idxs if self.cash_sharing else col_idxs]
        
        # 处理调用序列：如果存在，则根据列索引选择相应的序列
        if self.call_seq is not None:
            new_call_seq = self.call_seq.values[:, col_idxs]  # 选择对应列的调用序列
        else:
            new_call_seq = None  # 如果原本就没有调用序列，保持None

        # 返回一个新的Portfolio实例，包含索引操作后的所有数据
        return self.replace(
            wrapper=new_wrapper,
            close=new_close,
            order_records=new_order_records,
            log_records=new_log_records,
            init_cash=new_init_cash,
            call_seq=new_call_seq
        )

    # ############# Class methods ############# #

    @classmethod  # 类方法装饰器
    def from_orders(cls: tp.Type[PortfolioT],
                    # 核心价格数据
                    close: tp.ArrayLike,  # 收盘价数组，用于计算未实现盈亏和投资组合价值
                    
                    # 订单基本参数
                    size: tp.Optional[tp.ArrayLike] = None,  # 订单大小（股数或金额）
                    size_type: tp.Optional[tp.ArrayLike] = None,  # 订单大小类型（股数/金额/百分比等）
                    direction: tp.Optional[tp.ArrayLike] = None,  # 交易方向（多头/空头/双向）
                    price: tp.Optional[tp.ArrayLike] = None,  # 订单价格
                    
                    # 交易成本参数
                    fees: tp.Optional[tp.ArrayLike] = None,  # 费用比例（订单价值的百分比）
                    fixed_fees: tp.Optional[tp.ArrayLike] = None,  # 固定费用（每笔订单固定金额）
                    slippage: tp.Optional[tp.ArrayLike] = None,  # 滑点比例（价格的百分比）
                    
                    # 订单限制参数
                    min_size: tp.Optional[tp.ArrayLike] = None,  # 最小订单大小
                    max_size: tp.Optional[tp.ArrayLike] = None,  # 最大订单大小
                    size_granularity: tp.Optional[tp.ArrayLike] = None,  # 订单大小粒度
                    
                    # 订单执行参数
                    reject_prob: tp.Optional[tp.ArrayLike] = None,  # 订单拒绝概率
                    lock_cash: tp.Optional[tp.ArrayLike] = None,  # 做空时是否锁定现金
                    allow_partial: tp.Optional[tp.ArrayLike] = None,  # 是否允许部分成交
                    raise_reject: tp.Optional[tp.ArrayLike] = None,  # 订单被拒绝时是否抛出异常
                    log: tp.Optional[tp.ArrayLike] = None,  # 是否记录日志
                    
                    # 估值参数
                    val_price: tp.Optional[tp.ArrayLike] = None,  # 资产估值价格
                    
                    # 投资组合设置参数
                    init_cash: tp.Optional[tp.ArrayLike] = None,  # 初始现金
                    cash_sharing: tp.Optional[bool] = None,  # 是否在组内共享现金
                    call_seq: tp.Optional[tp.ArrayLike] = None,  # 调用序列
                    
                    # 模拟控制参数
                    ffill_val_price: tp.Optional[bool] = None,  # 是否前向填充估值价格
                    update_value: tp.Optional[bool] = None,  # 是否在订单后更新组价值
                    max_orders: tp.Optional[int] = None,  # 最大订单记录数
                    max_logs: tp.Optional[int] = None,  # 最大日志记录数
                    seed: tp.Optional[int] = None,  # 随机种子
                    
                    # 数据处理参数
                    group_by: tp.GroupByLike = None,  # 分组方式
                    broadcast_kwargs: tp.KwargsLike = None,  # 广播参数
                    wrapper_kwargs: tp.KwargsLike = None,  # 包装器参数
                    freq: tp.Optional[tp.FrequencyLike] = None,  # 时间频率
                    attach_call_seq: tp.Optional[bool] = None,  # 是否附加调用序列
                    **kwargs) -> PortfolioT:  # 其他参数传递给构造函数
        """
        从订单模拟投资组合 - 根据订单大小、价格、费用等信息构建投资组合
        
        这是Portfolio类最直接、最快速的构建方法。它接受订单的各种参数（大小、价格、费用等），
        将它们广播到一致的形状，然后为每个时间步和资产创建订单并执行模拟。
        
        Simulate portfolio from orders - size, price, fees, and other information.

        参数说明 (Args):
            close (array_like): 每个时间步的最后资产价格
                将被广播到所有其他参数的形状。
                用于计算未实现盈亏和投资组合价值。
                
            size (float or array_like): 要订购的订单大小
                参见 `vectorbt.portfolio.enums.Order.size`。将被广播。
                可以是股数、金额或百分比，具体含义由size_type决定。
                
            size_type (SizeType or array_like): 订单大小类型
                参见 `vectorbt.portfolio.enums.SizeType` 和 `vectorbt.portfolio.enums.Order.size_type`。
                将被广播。
                可选值：Amount(股数)、Value(金额)、Percent(百分比)等。

                !!! note
                    `SizeType.Percent` 不支持仓位反转。请切换到单一方向。

                !!! warning
                    使用 `SizeType.Percent` 时要谨慎设置 `call_seq` 为 'auto'。
                    要在买入订单之前执行卖出订单，需要提前近似计算组中每个订单的价值。
                    但由于 `SizeType.Percent` 依赖于现金余额，而现金余额无法提前计算
                    （因为每个订单后都可能改变），这可能产生非最优的调用序列。
                    
            direction (Direction or array_like): 交易方向
                参见 `vectorbt.portfolio.enums.Direction` 和 `vectorbt.portfolio.enums.Order.direction`。
                将被广播。可选值：LongOnly(仅多头)、ShortOnly(仅空头)、Both(双向)。
                
            price (array_like of float): 订单价格
                参见 `vectorbt.portfolio.enums.Order.price`。默认为 `np.inf`。将被广播。
                指定订单的执行价格，通常使用开盘价或收盘价。

                !!! note
                    确保在启用现金共享和 `call_seq` 设为 `CallSeqType.Auto` 的组中，
                    所有订单价格使用相同的时间戳。
                    
            fees (float or array_like): 费用比例
                参见 `vectorbt.portfolio.enums.Order.fees`。将被广播。
                以订单价值的百分比形式表示的交易费用（如0.001表示0.1%）。
                
            fixed_fees (float or array_like): 固定费用
                参见 `vectorbt.portfolio.enums.Order.fixed_fees`。将被广播。
                每笔订单需要支付的固定金额费用，不依赖于订单大小。
                
            slippage (float or array_like): 滑点比例
                参见 `vectorbt.portfolio.enums.Order.slippage`。将被广播。
                以价格百分比形式表示的滑点成本（如0.001表示0.1%的价格偏移）。
                
            min_size (float or array_like): 最小订单大小
                参见 `vectorbt.portfolio.enums.Order.min_size`。将被广播。
                订单被接受的最小规模，低于此值的订单将被拒绝。
                
            max_size (float or array_like): 最大订单大小
                参见 `vectorbt.portfolio.enums.Order.max_size`。将被广播。
                单笔订单的最大规模。如果超过此值，订单将被部分成交。

                超出时将部分成交。
                
            size_granularity (float or array_like): 订单大小粒度
                参见 `vectorbt.portfolio.enums.Order.size_granularity`。将被广播。
                订单大小的最小变动单位，用于确保订单大小符合交易规则。
                
            reject_prob (float or array_like): 订单拒绝概率
                参见 `vectorbt.portfolio.enums.Order.reject_prob`。将被广播。
                订单被随机拒绝的概率（0-1之间），用于模拟市场流动性不足等情况。
                
            lock_cash (bool or array_like): 做空时是否锁定现金
                参见 `vectorbt.portfolio.enums.Order.lock_cash`。将被广播。
                在进行空头交易时是否需要锁定等额现金作为保证金。
                
            allow_partial (bool or array_like): 是否允许部分成交
                参见 `vectorbt.portfolio.enums.Order.allow_partial`。将被广播。
                当资金或仓位不足时，是否允许订单部分执行。

                当订单大小为 `np.inf` 时不适用。
                
            raise_reject (bool or array_like): 订单被拒绝时是否抛出异常
                参见 `vectorbt.portfolio.enums.Order.raise_reject`。将被广播。
                当订单因各种原因被拒绝时，是否抛出异常中断执行。
                
            log (bool or array_like): 是否记录订单日志
                参见 `vectorbt.portfolio.enums.Order.log`。将被广播。
                是否在模拟过程中记录详细的订单执行日志，用于调试和分析。
                
            val_price (array_like of float): 资产估值价格
                将被广播。用于决策时计算组中每个资产价值的价格参考。

                * 任何 `-np.inf` 元素会被最新的估值价格替换（前一个 `close` 或
                    如果 `ffill_val_price` 为真则是最新已知的估值价格）。
                * 任何 `np.inf` 元素会被当前订单价格替换。

                在决策时用于计算组中每个资产的价值，
                例如，将目标价值转换为目标数量。

                !!! note
                    与 `Portfolio.from_order_func` 相比，订单价格是事先已知的（某种程度上），
                    因此 `val_price` 默认设置为当前订单价格（使用 `np.inf`）。
                    要使用前一个收盘价进行估值，请在设置中将其设为 `-np.inf`。

                !!! note
                    确保 `val_price` 使用的时间戳在启用现金共享的组中所有订单时间戳之前
                    （例如前一个 `close`），否则您是在作弊。
                    
            init_cash (InitCashMode, float or array_like of float): 初始资本
                投资组合的起始资金量。

                默认情况下，将广播到列数。
                如果启用现金共享，将广播到组数。
                参见 `vectorbt.portfolio.enums.InitCashMode` 来寻找最优初始资金。

                !!! note
                    模式 `InitCashMode.AutoAlign` 在投资组合初始化后应用，
                    为所有列/组设置相同的初始现金。更改分组将改变初始现金，
                    所以在索引时要注意。
                    
            cash_sharing (bool): 是否在同一组内共享现金
                启用后，组内所有资产共享同一个现金池。

                如果 `group_by` 为 None，`group_by` 变为 True 以形成带有现金共享的单一组。

                !!! warning
                    引入跨资产依赖关系。

                    此方法假设在共享同一资本的资产组中，所有订单将在同一时点内执行
                    并保持其价格，无论它们在队列中的位置如何，即使它们相互依赖
                    因此无法并行执行。
                    
            call_seq (CallSeqType or array_like): 每行每组的默认调用序列
                控制同一时间步内订单的执行顺序。

                此序列中的每个值应指示组中要调用的下一列的位置。
                `call_seq` 的处理总是从左到右进行。
                例如，`[2, 0, 1]` 将首先调用列 'c'，然后是 'a'，最后是 'b'。

                * 使用 `vectorbt.portfolio.enums.CallSeqType` 选择序列类型。
                * 设置为数组以指定自定义序列。不会广播。

                如果选择 `CallSeqType.Auto`，将根据订单价值动态重新排列调用。
                计算每行每组所有订单的价值，并按此价值排序。
                卖出订单将首先执行以为买入订单释放资金。

                !!! warning
                    `CallSeqType.Auto` 应谨慎使用：

                    * 它不仅假设订单价格事先已知，而且假设订单可以按任意顺序执行
                        并仍保持其价格。实际上，这很难做到：处理一个资产后，
                        时间已过去，其他资产的价格可能已经改变。
                    * 即使您能够指定足够大的滑点来补偿此行为，
                        滑点本身也应该取决于执行顺序。此方法不允许您这样做。
                    * 如果一个订单被拒绝，它仍可能执行下一个订单，
                        并可能使它们没有所需的资金。

                    要获得更多控制，请使用 `Portfolio.from_order_func`。
                    
            ffill_val_price (bool): 是否仅在已知时跟踪估值价格
                前向填充估值价格的标志。否则，未知的 `close` 将导致
                下一个时间戳的估值价格为 NaN。
                
            update_value (bool): 是否在每个订单成交后更新组价值
                控制是否在订单执行后立即更新投资组合价值计算。
                
            max_orders (int): 订单记录数组的大小
                订单记录数组的最大容量。默认为广播形状中的元素数。
                如果内存不足，请设置为较低的数字。

            max_logs (int): 日志记录数组的大小
                日志记录数组的最大容量。如果任何 `log` 为 True，
                默认为广播形状中的元素数，否则为 1。
                如果内存不足，请设置为较低的数字。

            seed (int): 随机种子
                为 `call_seq` 和模拟开始时设置的随机种子，确保结果可重现。
                
            group_by (any): 列分组方式
                定义如何对列进行分组。参见 `vectorbt.base.column_grouper.ColumnGrouper`。
                
            broadcast_kwargs (dict): 广播关键字参数
                传递给 `vectorbt.base.reshape_fns.broadcast` 的关键字参数。
                
            wrapper_kwargs (dict): 包装器关键字参数
                传递给 `vectorbt.base.array_wrapper.ArrayWrapper` 的关键字参数。
                
            freq (any): 索引频率
                如果无法从 `close` 中解析时间频率，手动指定的索引频率。
                
            attach_call_seq (bool): 是否将 `call_seq` 传递给构造函数
                如果您想按模拟顺序分析某些指标，则有意义。
                否则，只是占用内存。
                
            **kwargs: 传递给 `__init__` 方法的关键字参数
                其他所有参数都会传递给Portfolio的构造函数。

        所有可广播的参数都将使用 `vectorbt.base.reshape_fns.broadcast` 进行广播，
        但保持原始形状以利用灵活索引并节省内存。

        默认值参见 `vectorbt._settings.settings` 中的 'portfolio' 配置。

        !!! note
            当 `call_seq` 不是 `CallSeqType.Auto` 时，在每个时间戳，
            组中资产的处理严格按照 `call_seq` 中定义的顺序进行。此顺序无法动态更改。

            这对此特定方法有一个重大影响：调用栈中的最后一个资产无法处理，
            直到其他资产被处理。这就是为什么重新平衡在此设置中无法正常工作的原因：
            必须事先为所有资产指定百分比，然后调整处理顺序，先卖出要卖出的资产，
            以为要买入的资产释放资金。这可以通过使用 `CallSeqType.Auto` 自动完成。

        !!! hint
            所有可广播的参数都可以按帧、序列、行、列或元素设置。

        使用示例 (Usage):
            * 每个时点买入10单位：

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])  # 价格序列
            >>> pf = vbt.Portfolio.from_orders(close, 10)  # 每个时点买入10单位

            >>> pf.assets()  # 查看资产持有量
            0    10.0
            1    20.0
            2    30.0
            3    40.0
            4    40.0
            dtype: float64
            >>> pf.cash()  # 查看现金余额
            0    90.0
            1    70.0
            2    40.0
            3     0.0
            4     0.0
            dtype: float64
            ```

            * 通过先平仓来反转每个仓位：

            ```pycon
            >>> size = [1, 0, -1, 0, 1]  # 目标仓位百分比序列
            >>> pf = vbt.Portfolio.from_orders(close, size, size_type='targetpercent')

            >>> pf.assets()  # 查看资产持有量（包括负值即空头仓位）
            0    100.000000
            1      0.000000
            2    -66.666667  # 空头仓位
            3      0.000000
            4     26.666667
            dtype: float64
            >>> pf.cash()  # 查看现金余额
            0      0.000000
            1    200.000000
            2    400.000000
            3    133.333333
            4      0.000000
            dtype: float64
            ```

            * 等权重投资组合，如 `vectorbt.portfolio.nb.simulate_nb` 示例所示
            （更紧凑但对执行控制较少）：

            ```pycon
            >>> np.random.seed(42)
            >>> close = pd.DataFrame(np.random.uniform(1, 10, size=(5, 3)))  # 3个资产，5个时点
            >>> size = pd.Series(np.full(5, 1/3))  # 每列33.3%的权重
            >>> size[1::2] = np.nan  # 跳过每个第二个时点

            >>> pf = vbt.Portfolio.from_orders(
            ...     close,  # 既作为参考价格也作为订单价格
            ...     size,
            ...     size_type='targetpercent',  # 目标百分比类型
            ...     call_seq='auto',  # 先卖后买的自动调用序列
            ...     group_by=True,  # 单一组
            ...     cash_sharing=True,  # 资产共享相同现金
            ...     fees=0.001, fixed_fees=1., slippage=0.001  # 交易成本
            ... )

            >>> pf.asset_value(group_by=False).vbt.plot()  # 绘制资产价值图表
            ```

            ![](/assets/images/simulate_nb.svg)
        """
        # 获取默认配置参数
        from vectorbt._settings import settings
        portfolio_cfg = settings['portfolio']  # 获取投资组合配置

        # 设置默认参数值：如果参数未提供，则使用配置文件中的默认值
        if size is None:
            size = portfolio_cfg['size']  # 默认订单大小
        if size_type is None:
            size_type = portfolio_cfg['size_type']  # 默认订单大小类型
        size_type = map_enum_fields(size_type, SizeType)  # 将字符串映射为枚举值
        if direction is None:
            direction = portfolio_cfg['order_direction']  # 默认交易方向
        direction = map_enum_fields(direction, Direction)  # 将字符串映射为枚举值
        if price is None:
            price = np.inf  # 默认价格设为无穷大，表示使用市价
        if size is None:  # 这里可能是重复检查，但保持原有逻辑
            size = portfolio_cfg['size']
        if fees is None:
            fees = portfolio_cfg['fees']  # 默认费用比例
        if fixed_fees is None:
            fixed_fees = portfolio_cfg['fixed_fees']  # 默认固定费用
        if slippage is None:
            slippage = portfolio_cfg['slippage']  # 默认滑点
        if min_size is None:
            min_size = portfolio_cfg['min_size']  # 默认最小订单大小
        if max_size is None:
            max_size = portfolio_cfg['max_size']  # 默认最大订单大小
        if size_granularity is None:
            size_granularity = portfolio_cfg['size_granularity']  # 默认订单大小粒度
        if reject_prob is None:
            reject_prob = portfolio_cfg['reject_prob']  # 默认订单拒绝概率
        if lock_cash is None:
            lock_cash = portfolio_cfg['lock_cash']  # 默认是否锁定现金
        if allow_partial is None:
            allow_partial = portfolio_cfg['allow_partial']  # 默认是否允许部分成交
        if raise_reject is None:
            raise_reject = portfolio_cfg['raise_reject']  # 默认是否在拒绝时抛出异常
        if log is None:
            log = portfolio_cfg['log']  # 默认是否记录日志
        if val_price is None:
            val_price = portfolio_cfg['val_price']  # 默认估值价格
        if init_cash is None:
            init_cash = portfolio_cfg['init_cash']  # 默认初始现金
        # 处理初始现金设置
        if isinstance(init_cash, str):
            init_cash = map_enum_fields(init_cash, InitCashMode)  # 将字符串映射为枚举
        if isinstance(init_cash, int) and init_cash in InitCashMode:
            init_cash_mode = init_cash  # 保存初始现金模式
            init_cash = np.inf  # 临时设为无穷大，稍后会重新计算
        else:
            init_cash_mode = None  # 非枚举模式
            
        # 处理现金共享设置
        if cash_sharing is None:
            cash_sharing = portfolio_cfg['cash_sharing']  # 从配置获取默认值
        if cash_sharing and group_by is None:
            group_by = True  # 现金共享时自动启用分组
            
        # 处理调用序列设置
        if call_seq is None:
            call_seq = portfolio_cfg['call_seq']  # 从配置获取默认调用序列
        auto_call_seq = False  # 是否使用自动调用序列
        if isinstance(call_seq, str):
            call_seq = map_enum_fields(call_seq, CallSeqType)  # 字符串映射为枚举
        if isinstance(call_seq, int):
            if call_seq == CallSeqType.Auto:
                call_seq = CallSeqType.Default  # 自动模式时先设为默认
                auto_call_seq = True  # 标记使用自动调用序列
                
        # 处理其他模拟参数
        if ffill_val_price is None:
            ffill_val_price = portfolio_cfg['ffill_val_price']  # 是否前向填充估值价格
        if update_value is None:
            update_value = portfolio_cfg['update_value']  # 是否在每次订单后更新价值
        if seed is None:
            seed = portfolio_cfg['seed']  # 随机种子
        if seed is not None:
            set_seed(seed)  # 设置随机种子以确保可重复性
            
        # 处理数据处理参数
        if freq is None:
            freq = portfolio_cfg['freq']  # 时间频率
        if attach_call_seq is None:
            attach_call_seq = portfolio_cfg['attach_call_seq']  # 是否附加调用序列
        if broadcast_kwargs is None:
            broadcast_kwargs = {}  # 广播参数
        if wrapper_kwargs is None:
            wrapper_kwargs = {}  # 包装器参数
            
        # 参数验证：现金共享时不能禁用组选择
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")

        # 准备模拟
        # 由于灵活索引，只有收盘价需要广播，其他可以保持原样
        broadcastable_args = (
            size,  # 订单大小
            price,  # 订单价格
            size_type,  # 订单大小类型
            direction,  # 交易方向
            fees,  # 费用比例
            fixed_fees,  # 固定费用
            slippage,  # 滑点
            min_size,  # 最小订单大小
            max_size,  # 最大订单大小
            size_granularity,  # 订单大小粒度
            reject_prob,  # 订单拒绝概率
            lock_cash,  # 做空时是否锁定现金
            allow_partial,  # 是否允许部分成交
            raise_reject,  # 拒绝时是否抛出异常
            log,  # 是否记录日志
            val_price,  # 资产估值价格
            close  # 收盘价（需要完全广播）
        )
        
        # 设置广播选项：除了收盘价外，其他参数保持原始形状以节省内存
        keep_raw = [True] * len(broadcastable_args)  # 创建保持原始形状的标记列表
        keep_raw[-1] = False  # 收盘价需要完全广播
        broadcast_kwargs = merge_dicts(dict(
            keep_raw=keep_raw,  # 保持原始形状的参数列表
            require_kwargs=dict(requirements='W')  # 广播要求：可写
        ), broadcast_kwargs)
        
        # 执行参数广播
        broadcasted_args = broadcast(*broadcastable_args, **broadcast_kwargs)
        close = broadcasted_args[-1]  # 获取广播后的收盘价
        
        # 确保收盘价是pandas对象
        if not checks.is_pandas(close):
            close = pd.Series(close) if close.ndim == 1 else pd.DataFrame(close)
            
        # 确定目标形状（二维）
        target_shape_2d = (close.shape[0], close.shape[1] if close.ndim > 1 else 1)
        
        # 创建数组包装器，管理索引和分组
        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        
        # 获取现金共享组的长度（如果启用现金共享）
        cs_group_lens = wrapper.grouper.get_group_lens(group_by=None if cash_sharing else False)
        
        # 将初始现金广播到合适的形状并确保为float64类型
        init_cash = np.require(np.broadcast_to(init_cash, (len(cs_group_lens),)), dtype=np.float64)
        
        # 获取用户指定分组的长度
        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        
        # 处理调用序列
        if checks.is_any_array(call_seq):
            # 如果是数组，则广播到目标形状并验证
            call_seq = nb.require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
        else:
            # 否则根据序列类型构建调用序列
            call_seq = nb.build_call_seq(target_shape_2d, group_lens, call_seq_type=call_seq)
            
        # 设置最大订单数和日志数的默认值
        if max_orders is None:
            max_orders = target_shape_2d[0] * target_shape_2d[1]  # 默认为总元素数
        if max_logs is None:
            max_logs = target_shape_2d[0] * target_shape_2d[1]  # 默认为总元素数
        if not np.any(log):
            max_logs = 1  # 如果不记录日志，则最小化日志数组大小

        # 执行投资组合模拟
        # 调用Numba编译的核心模拟函数来处理订单执行
        order_records, log_records = nb.simulate_from_orders_nb(
            target_shape_2d,  # 目标二维形状(时间步数, 资产数)
            cs_group_lens,  # 现金共享组长度（仅在启用现金共享时分组以提速）
            init_cash,  # 初始现金数组
            call_seq,  # 调用序列数组
            *map(np.asarray, broadcasted_args),  # 将所有广播参数转换为numpy数组
            auto_call_seq,  # 是否使用自动调用序列
            ffill_val_price,  # 是否前向填充估值价格
            update_value,  # 是否在订单后更新价值
            max_orders,  # 最大订单记录数
            max_logs,  # 最大日志记录数
            close.ndim == 2  # 是否为二维数据（多资产）
        )

        # 创建Portfolio实例
        # 使用模拟结果和所有配置参数创建新的Portfolio对象
        return cls(
            wrapper,  # 数组包装器，管理索引和分组
            close,  # 收盘价数据
            order_records,  # 执行的订单记录
            log_records,  # 模拟过程中的日志记录
            init_cash if init_cash_mode is None else init_cash_mode,  # 初始现金（保持模式或实际值）
            cash_sharing,  # 现金共享设置
            call_seq=call_seq if attach_call_seq else None,  # 调用序列（如果需要附加）
            **kwargs  # 传递给构造函数的其他参数
        )

    @classmethod  # 类方法装饰器
    def from_signals(cls: tp.Type[PortfolioT],
                     # 核心价格数据
                     close: tp.ArrayLike,  # 收盘价数组，用于计算未实现盈亏和投资组合价值
                     
                     # 基本信号参数
                     entries: tp.Optional[tp.ArrayLike] = None,  # 入场信号（多头或通用入场）
                     exits: tp.Optional[tp.ArrayLike] = None,  # 出场信号（多头或通用出场）
                     short_entries: tp.Optional[tp.ArrayLike] = None,  # 空头入场信号
                     short_exits: tp.Optional[tp.ArrayLike] = None,  # 空头出场信号
                     
                     # 自定义信号函数
                     signal_func_nb: nb.SignalFuncT = nb.no_signal_func_nb,  # 自定义信号生成函数
                     signal_args: tp.ArgsLike = (),  # 传递给信号函数的参数
                     
                     # 订单基本参数（继承自from_orders）
                     size: tp.Optional[tp.ArrayLike] = None,  # 订单大小
                     size_type: tp.Optional[tp.ArrayLike] = None,  # 订单大小类型
                     price: tp.Optional[tp.ArrayLike] = None,  # 订单价格
                     
                     # 交易成本参数
                     fees: tp.Optional[tp.ArrayLike] = None,  # 费用比例
                     fixed_fees: tp.Optional[tp.ArrayLike] = None,  # 固定费用
                     slippage: tp.Optional[tp.ArrayLike] = None,  # 滑点
                     
                     # 订单限制参数
                     min_size: tp.Optional[tp.ArrayLike] = None,  # 最小订单大小
                     max_size: tp.Optional[tp.ArrayLike] = None,  # 最大订单大小
                     size_granularity: tp.Optional[tp.ArrayLike] = None,  # 订单大小粒度
                     
                     # 订单执行参数
                     reject_prob: tp.Optional[tp.ArrayLike] = None,  # 订单拒绝概率
                     lock_cash: tp.Optional[tp.ArrayLike] = None,  # 做空时是否锁定现金
                     allow_partial: tp.Optional[tp.ArrayLike] = None,  # 是否允许部分成交
                     raise_reject: tp.Optional[tp.ArrayLike] = None,  # 拒绝时是否抛出异常
                     log: tp.Optional[tp.ArrayLike] = None,  # 是否记录日志
                     
                     # 信号处理参数
                     accumulate: tp.Optional[tp.ArrayLike] = None,  # 是否允许累积仓位
                     upon_long_conflict: tp.Optional[tp.ArrayLike] = None,  # 多头信号冲突时的处理方式
                     upon_short_conflict: tp.Optional[tp.ArrayLike] = None,  # 空头信号冲突时的处理方式
                     upon_dir_conflict: tp.Optional[tp.ArrayLike] = None,  # 方向冲突时的处理方式
                     upon_opposite_entry: tp.Optional[tp.ArrayLike] = None,  # 相反入场信号的处理方式
                     direction: tp.Optional[tp.ArrayLike] = None,  # 交易方向限制
                     
                     # 估值参数
                     val_price: tp.Optional[tp.ArrayLike] = None,  # 资产估值价格
                     
                     # OHLC数据（用于止损止盈）
                     open: tp.Optional[tp.ArrayLike] = None,  # 开盘价
                     high: tp.Optional[tp.ArrayLike] = None,  # 最高价
                     low: tp.Optional[tp.ArrayLike] = None,  # 最低价
                     
                     # 止损止盈参数
                     sl_stop: tp.Optional[tp.ArrayLike] = None,  # 止损水平
                     sl_trail: tp.Optional[tp.ArrayLike] = None,  # 是否为追踪止损
                     tp_stop: tp.Optional[tp.ArrayLike] = None,  # 止盈水平
                     stop_entry_price: tp.Optional[tp.ArrayLike] = None,  # 止损入场价格类型
                     stop_exit_price: tp.Optional[tp.ArrayLike] = None,  # 止损出场价格类型
                     upon_stop_exit: tp.Optional[tp.ArrayLike] = None,  # 止损出场时的处理方式
                     upon_stop_update: tp.Optional[tp.ArrayLike] = None,  # 止损更新时的处理方式
                     
                     # 止损止盈调整函数
                     adjust_sl_func_nb: nb.AdjustSLFuncT = nb.no_adjust_sl_func_nb,  # 调整止损的函数
                     adjust_sl_args: tp.Args = (),  # 调整止损函数的参数
                     adjust_tp_func_nb: nb.AdjustTPFuncT = nb.no_adjust_tp_func_nb,  # 调整止盈的函数
                     adjust_tp_args: tp.Args = (),  # 调整止盈函数的参数
                     use_stops: tp.Optional[bool] = None,  # 是否使用止损止盈
                     
                     # 投资组合设置参数（继承自from_orders）
                     init_cash: tp.Optional[tp.ArrayLike] = None,  # 初始现金
                     cash_sharing: tp.Optional[bool] = None,  # 是否在组内共享现金
                     call_seq: tp.Optional[tp.ArrayLike] = None,  # 调用序列
                     
                     # 模拟控制参数
                     ffill_val_price: tp.Optional[bool] = None,  # 是否前向填充估值价格
                     update_value: tp.Optional[bool] = None,  # 是否在订单后更新价值
                     max_orders: tp.Optional[int] = None,  # 最大订单记录数
                     max_logs: tp.Optional[int] = None,  # 最大日志记录数
                     seed: tp.Optional[int] = None,  # 随机种子
                     
                     # 数据处理参数
                     group_by: tp.GroupByLike = None,  # 分组方式
                     broadcast_named_args: tp.KwargsLike = None,  # 命名参数广播字典
                     broadcast_kwargs: tp.KwargsLike = None,  # 广播关键字参数
                     template_mapping: tp.Optional[tp.Mapping] = None,  # 模板映射字典
                     wrapper_kwargs: tp.KwargsLike = None,  # 包装器关键字参数
                     freq: tp.Optional[tp.FrequencyLike] = None,  # 时间频率
                     attach_call_seq: tp.Optional[bool] = None,  # 是否附加调用序列
                     **kwargs) -> PortfolioT:  # 其他参数传递给构造函数
        """
        从入场和出场信号模拟投资组合
        
        这个方法是基于交易信号的投资组合构建方法。它在from_orders之上添加了一个抽象层，
        自动化了一些信号处理过程，如防止重复入场、止损止盈等功能。
        
        Simulate portfolio from entry and exit signals.

        参见 `vectorbt.portfolio.nb.simulate_from_signal_func_nb`。

        您有三种提供信号的选项：

        * `entries` 和 `exits`：每对信号的方向取自 `direction` 参数。
            最适合在方向不随时间变化的情况下使用。

            使用 `vectorbt.portfolio.nb.dir_enex_signal_func_nb` 作为 `signal_func_nb`。

            !!! hint
                `entries` 和 `exits` 可以轻松转换为方向感知信号：

                * (True, True, 'longonly') -> True, True, False, False
                * (True, True, 'shortonly') -> False, False, True, True
                * (True, True, 'both') -> True, False, True, False

        * `entries`（作为多头）、`exits`（作为多头）、`short_entries` 和 `short_exits`：
            方向已经内置在数组中。最适合在方向频繁变化时使用
            （例如，如果您有一个指标提供多头信号，另一个提供空头信号）。

            使用 `vectorbt.portfolio.nb.ls_enex_signal_func_nb` 作为 `signal_func_nb`。

        * `signal_func_nb` 和 `signal_args`：返回方向感知信号的自定义信号函数。
            最适合在信号应该基于自定义条件动态放置时使用。

        参数说明 (Args):
            close (array_like): 参见 `Portfolio.from_orders`。
            
            entries (array_like of bool): 入场信号的布尔数组
                如果所有其他信号数组都未设置，则默认为 True，否则为 False。将被广播。

                * 如果未设置 `short_entries` 和 `short_exits`：如果 `direction` 是 `all` 或 `longonly`
                    则作为多头信号，否则作为空头信号。
                * 如果设置了 `short_entries` 或 `short_exits`：作为 `long_entries`。
                
            exits (array_like of bool): 出场信号的布尔数组
                默认为 False。将被广播。

                * 如果未设置 `short_entries` 和 `short_exits`：如果 `direction` 是 `all` 或 `longonly`
                    则作为空头信号，否则作为多头信号。
                * 如果设置了 `short_entries` 或 `short_exits`：作为 `long_exits`。
                
            short_entries (array_like of bool): 空头入场信号的布尔数组
                默认为 False。将被广播。
                
            short_exits (array_like of bool): 空头出场信号的布尔数组
                默认为 False。将被广播。
                
            signal_func_nb (callable): 调用以生成信号的函数
                应接受 `vectorbt.portfolio.enums.SignalContext` 和 `*signal_args`。
                应返回多头入场信号、多头出场信号、空头入场信号和空头出场信号。

                !!! note
                    止损信号具有优先级：仅当没有止损信号时才执行 `signal_func_nb`。
                    
            signal_args (tuple): 传递给 `signal_func_nb` 的打包参数
                默认为 `()`。
                
            size (float or array_like): 订单大小，参见 `Portfolio.from_orders`
                控制每次信号触发时的交易数量。

                !!! note
                    不允许负值大小。您应该使用信号来表达方向。
                    
            size_type (SizeType or array_like): 订单大小类型，参见 `Portfolio.from_orders`
                控制如何解释size参数。

                仅支持 `SizeType.Amount`、`SizeType.Value` 和 `SizeType.Percent`。
                其他模式（如目标百分比）与信号不兼容，因为它们的逻辑可能与信号方向矛盾。

                !!! note
                    `SizeType.Percent` 不支持仓位反转。切换到单一方向或使用 
                    `vectorbt.portfolio.enums.OppositeEntryMode.Close` 先平仓。

                参见 `Portfolio.from_orders` 中的警告。
                
            price (array_like of float): 订单价格，参见 `Portfolio.from_orders`
            fees (float or array_like): 费用比例，参见 `Portfolio.from_orders`
            fixed_fees (float or array_like): 固定费用，参见 `Portfolio.from_orders`
            slippage (float or array_like): 滑点，参见 `Portfolio.from_orders`
            min_size (float or array_like): 最小订单大小，参见 `Portfolio.from_orders`
            max_size (float or array_like): 最大订单大小，参见 `Portfolio.from_orders`

                如果超出限制则部分成交。如果启用累积且 `max_size` 太低，
                您可能无法正确平仓。
                
            size_granularity (float or array_like): 订单大小粒度，参见 `Portfolio.from_orders`
            reject_prob (float or array_like): 订单拒绝概率，参见 `Portfolio.from_orders`
            lock_cash (bool or array_like): 做空时是否锁定现金，参见 `Portfolio.from_orders`
            allow_partial (bool or array_like): 是否允许部分成交，参见 `Portfolio.from_orders`
            raise_reject (bool or array_like): 拒绝时是否抛出异常，参见 `Portfolio.from_orders`
            log (bool or array_like): 是否记录日志，参见 `Portfolio.from_orders`
            
            accumulate (bool, AccumulationMode or array_like): 累积模式
                参见 `vectorbt.portfolio.enums.AccumulationMode`。
                如果为 True，则变为 'both'。如果为 False，则变为 'disabled'。将被广播。

                启用时，`Portfolio.from_signals` 的行为类似于 `Portfolio.from_orders`。
                
            upon_long_conflict (ConflictMode or array_like): 多头信号冲突模式
                当多个多头信号冲突时的处理方式。
                参见 `vectorbt.portfolio.enums.ConflictMode`。将被广播。
                
            upon_short_conflict (ConflictMode or array_like): 空头信号冲突模式
                当多个空头信号冲突时的处理方式。
                参见 `vectorbt.portfolio.enums.ConflictMode`。将被广播。
                
            upon_dir_conflict (DirectionConflictMode or array_like): 方向冲突模式
                当多头和空头信号同时出现时的处理方式。
                参见 `vectorbt.portfolio.enums.DirectionConflictMode`。将被广播。
                
            upon_opposite_entry (OppositeEntryMode or array_like): 相反入场模式
                当出现与当前仓位相反的入场信号时的处理方式。
                参见 `vectorbt.portfolio.enums.OppositeEntryMode`。将被广播。
                
            direction (Direction or array_like): 交易方向限制，参见 `Portfolio.from_orders`

                仅在 `short_entries` 和 `short_exits` 未设置时生效。
                
            val_price (array_like of float): 资产估值价格，参见 `Portfolio.from_orders`
            open (array_like of float): 每个时间步的开盘价
                默认为 `np.nan`，会被 `close` 替换。将被广播。

                仅用于止损信号。
                
            high (array_like of float): 每个时间步的最高价
                默认为 `np.nan`，会被 `open` 和 `close` 的最大值替换。将被广播。

                仅用于止损信号。
                
            low (array_like of float): 每个时间步的最低价
                默认为 `np.nan`，会被 `open` 和 `close` 的最小值替换。将被广播。

                仅用于止损信号。
                
            sl_stop (array_like of float): 止损水平
                止损触发的价格水平。将被广播。

                对于多头/空头仓位，是相对于获得价格的下方/上方百分比。
                注意：0.01 = 1%。
                
            sl_trail (array_like of bool): `sl_stop` 是否为追踪止损
                控制止损是否会随着有利价格变动而调整。将被广播。
                
            tp_stop (array_like of float): 止盈水平
                止盈触发的价格水平。将被广播。

                对于多头/空头仓位，是相对于获得价格的上方/下方百分比。
                注意：0.01 = 1%。
                
            stop_entry_price (StopEntryPrice or array_like): 止损入场价格类型
                确定用于计算止损的入场价格类型。
                参见 `vectorbt.portfolio.enums.StopEntryPrice`。将被广播。

                如果按元素基础提供，则在入场时应用。
                
            stop_exit_price (StopExitPrice or array_like): 止损出场价格类型
                确定触发止损的出场价格类型。
                参见 `vectorbt.portfolio.enums.StopExitPrice`。将被广播。

                如果按元素基础提供，则在出场时应用。
                
            upon_stop_exit (StopExitMode or array_like): 止损出场模式
                确定止损触发后的处理方式。
                参见 `vectorbt.portfolio.enums.StopExitMode`。将被广播。

                如果按元素基础提供，则在出场时应用。
                
            upon_stop_update (StopUpdateMode or array_like): 止损更新模式
                确定重复入场时如何更新止损。
                参见 `vectorbt.portfolio.enums.StopUpdateMode`。将被广播。

                仅在启用累积时生效。

                如果按元素基础提供，则在重复入场时应用。
                
            adjust_sl_func_nb (callable): 调整止损的函数
                用于动态调整止损水平的自定义函数。
                默认为 `vectorbt.portfolio.nb.no_adjust_sl_func_nb`。

                在每行之前为每个元素调用。

                应接受 `vectorbt.portfolio.enums.AdjustSLContext` 和 `*adjust_sl_args`。
                应返回新的止损值和追踪标志的元组。
                
            adjust_sl_args (tuple): 传递给 `adjust_sl_func_nb` 的打包参数
                调整止损函数的额外参数。默认为 `()`。
            adjust_tp_func_nb (callable): 调整止盈的函数
                用于动态调整止盈水平的自定义函数。
                默认为 `vectorbt.portfolio.nb.no_adjust_tp_func_nb`。

                在每行之前为每个元素调用。

                应接受 `vectorbt.portfolio.enums.AdjustTPContext` 和 `*adjust_tp_args`。
                应返回新的止盈值。
                
            adjust_tp_args (tuple): 传递给 `adjust_tp_func_nb` 的打包参数
                调整止盈函数的额外参数。默认为 `()`。
                
            use_stops (bool): 是否使用止损止盈
                默认为 None，如果任何止损止盈不为 NaN 或任何调整函数是自定义的，
                则变为 True。

                禁用此选项可使简单用例的模拟速度更快。
                
            init_cash (InitCashMode, float or array_like of float): 初始现金，参见 `Portfolio.from_orders`
            cash_sharing (bool): 现金共享，参见 `Portfolio.from_orders`
            call_seq (CallSeqType or array_like): 调用序列，参见 `Portfolio.from_orders`
            ffill_val_price (bool): 前向填充估值价格，参见 `Portfolio.from_orders`
            update_value (bool): 更新价值，参见 `Portfolio.from_orders`
            max_orders (int): 最大订单数，参见 `Portfolio.from_orders`
            max_logs (int): 最大日志数，参见 `Portfolio.from_orders`
            seed (int): 随机种子，参见 `Portfolio.from_orders`
            group_by (any): 分组方式，参见 `Portfolio.from_orders`
            
            broadcast_named_args (dict): 包含要广播的命名参数的字典

                然后您可以将参数名称传递给函数，此方法将用它们对应的广播对象替换它们。
                
            broadcast_kwargs (dict): 广播参数，参见 `Portfolio.from_orders`
            template_mapping (mapping): 替换参数中模板的映射
            wrapper_kwargs (dict): 包装器参数，参见 `Portfolio.from_orders`
            freq (any): 频率，参见 `Portfolio.from_orders`
            attach_call_seq (bool): 附加调用序列，参见 `Portfolio.from_orders`
            **kwargs: 传递给 `__init__` 方法的关键字参数

        所有可广播的参数都将使用 `vectorbt.base.reshape_fns.broadcast` 进行广播，
        但保持原始形状以利用灵活索引并节省内存。

        默认值参见 `vectorbt._settings.settings` 中的 `portfolio`。

        !!! note
            止损信号具有优先级 - 它在同一时间条内在其他信号之前执行。
            也就是说，如果存在止损信号，就不会生成和执行其他信号，
            因为每个代码和时间条只能有一个订单。

        !!! hint
            如果您使用收盘价生成信号，不要忘记将信号向前移动一个时点，
            例如使用 `signals.vbt.fshift(1)`。通常，确保使用信号之后的价格。

        另请参见 `Portfolio.from_orders` 的注释和提示。

        使用示例 (Usage):
            * 默认情况下，如果所有信号数组都为 None，`entries` 变为 True，
            在第一个时点开仓，其他不做任何操作：

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])  # 价格序列
            >>> pf = vbt.Portfolio.from_signals(close, size=1)  # 默认信号策略
            >>> pf.asset_flow()  # 查看资产流动
            0    1.0  # 第一个时点买入1单位
            1    0.0  # 其他时点无操作
            2    0.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            * 入场开多头，出场平多头：

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),    # 入场信号
            ...     exits=pd.Series([False, False, True, True, True]),      # 出场信号
            ...     size=1,         # 每次交易1单位
            ...     direction='longonly'  # 仅多头方向
            ... )
            >>> pf.asset_flow()  # 查看资产流动
            0    1.0  # 入场买入
            1    0.0  # 已有仓位，不重复买入
            2    0.0  # 入场和出场信号同时，入场优先
            3   -1.0  # 出场卖出
            4    0.0  # 无仓位，出场信号无效
            dtype: float64

            >>> # 使用方向感知数组而不是 `direction` 参数
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),  # long_entries
            ...     exits=pd.Series([False, False, True, True, True]),  # long_exits
            ...     short_entries=False,
            ...     short_exits=False,
            ...     size=1
            ... )
            >>> pf.asset_flow()
            0    1.0
            1    0.0
            2    0.0
            3   -1.0
            4    0.0
            dtype: float64
            ```

            Notice how both `short_entries` and `short_exits` are provided as constants - as any other
            broadcastable argument, they are treated as arrays where each element is False.

            * Entry opens short, exit closes short:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='shortonly'
            ... )
            >>> pf.asset_flow()
            0   -1.0
            1    0.0
            2    0.0
            3    1.0
            4    0.0
            dtype: float64

            >>> # Using direction-aware arrays instead of `direction`
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=False,  # long_entries
            ...     exits=False,  # long_exits
            ...     short_entries=pd.Series([True, True, True, False, False]),
            ...     short_exits=pd.Series([False, False, True, True, True]),
            ...     size=1
            ... )
            >>> pf.asset_flow()
            0   -1.0
            1    0.0
            2    0.0
            3    1.0
            4    0.0
            dtype: float64
            ```

            * Entry opens long and closes short, exit closes long and opens short:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='both'
            ... )
            >>> pf.asset_flow()
            0    1.0
            1    0.0
            2    0.0
            3   -2.0
            4    0.0
            dtype: float64

            >>> # Using direction-aware arrays instead of `direction`
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),  # long_entries
            ...     exits=False,  # long_exits
            ...     short_entries=pd.Series([False, False, True, True, True]),
            ...     short_exits=False,
            ...     size=1
            ... )
            >>> pf.asset_flow()
            0    1.0
            1    0.0
            2    0.0
            3   -2.0
            4    0.0
            dtype: float64
            ```

            * More complex signal combinations are best expressed using direction-aware arrays.
            For example, ignore opposite signals as long as the current position is open:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries      =pd.Series([True, False, False, False, False]),  # long_entries
            ...     exits        =pd.Series([False, False, True, False, False]),  # long_exits
            ...     short_entries=pd.Series([False, True, False, True, False]),
            ...     short_exits  =pd.Series([False, False, False, False, True]),
            ...     size=1,
            ...     upon_opposite_entry='ignore'
            ... )
            >>> pf.asset_flow()
            0    1.0
            1    0.0
            2   -1.0
            3   -1.0
            4    1.0
            dtype: float64
            ```

            * First opposite signal closes the position, second one opens a new position:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='both',
            ...     upon_opposite_entry='close'
            ... )
            >>> pf.asset_flow()
            0    1.0
            1    0.0
            2    0.0
            3   -1.0
            4   -1.0
            dtype: float64
            ```

            * If both long entry and exit signals are True (a signal conflict), choose exit:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='longonly',
            ...     upon_long_conflict='exit')
            >>> pf.asset_flow()
            0    1.0
            1    0.0
            2   -1.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            * If both long entry and short entry signal are True (a direction conflict), choose short:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='both',
            ...     upon_dir_conflict='short')
            >>> pf.asset_flow()
            0    1.0
            1    0.0
            2   -2.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            !!! note
                Remember that when direction is set to 'both', entries become `long_entries` and exits become
                `short_entries`, so this becomes a conflict of directions rather than signals.

            * If there are both signal and direction conflicts:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=True,  # long_entries
            ...     exits=True,  # long_exits
            ...     short_entries=True,
            ...     short_exits=True,
            ...     size=1,
            ...     upon_long_conflict='entry',
            ...     upon_short_conflict='entry',
            ...     upon_dir_conflict='short'
            ... )
            >>> pf.asset_flow()
            0   -1.0
            1    0.0
            2    0.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            * Turn on accumulation of signals. Entry means long order, exit means short order
            (acts similar to `from_orders`):

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='both',
            ...     accumulate=True)
            >>> pf.asset_flow()
            0    1.0
            1    1.0
            2    0.0
            3   -1.0
            4   -1.0
            dtype: float64
            ```

            * Allow increasing a position (of any direction), deny decreasing a position:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='both',
            ...     accumulate='addonly')
            >>> pf.asset_flow()
            0    1.0  << open a long position
            1    1.0  << add to the position
            2    0.0
            3   -3.0  << close and open a short position
            4   -1.0  << add to the position
            dtype: float64
            ```

            * Testing multiple parameters (via broadcasting):

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     direction=[list(Direction)],
            ...     broadcast_kwargs=dict(columns_from=Direction._fields))
            >>> pf.asset_flow()
                Long  Short    All
            0  100.0 -100.0  100.0
            1    0.0    0.0    0.0
            2    0.0    0.0    0.0
            3 -100.0   50.0 -200.0
            4    0.0    0.0    0.0
            ```

            * Set risk/reward ratio by passing trailing stop loss and take profit thresholds:

            ```pycon
            >>> close = pd.Series([10, 11, 12, 11, 10, 9])
            >>> entries = pd.Series([True, False, False, False, False, False])
            >>> exits = pd.Series([False, False, False, False, False, True])
            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     sl_stop=0.1, sl_trail=True, tp_stop=0.2)  # take profit hit
            >>> pf.asset_flow()
            0    10.0
            1     0.0
            2   -10.0
            3     0.0
            4     0.0
            5     0.0
            dtype: float64

            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     sl_stop=0.1, sl_trail=True, tp_stop=0.3)  # stop loss hit
            >>> pf.asset_flow()
            0    10.0
            1     0.0
            2     0.0
            3     0.0
            4   -10.0
            5     0.0
            dtype: float64

            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     sl_stop=np.inf, sl_trail=True, tp_stop=np.inf)  # nothing hit, exit as usual
            >>> pf.asset_flow()
            0    10.0
            1     0.0
            2     0.0
            3     0.0
            4     0.0
            5   -10.0
            dtype: float64
            ```

            !!! note
                When the stop price is hit, the stop signal invalidates any other signal defined for this bar.
                Thus, make sure that your signaling logic happens at the very end of the bar
                (for example, by using the closing price), otherwise you may expose yourself to a look-ahead bias.

                See `vectorbt.portfolio.enums.StopExitPrice` for more details.

            * We can implement our own stop loss or take profit, or adjust the existing one at each time step.
            Let's implement [stepped stop-loss](https://www.freqtrade.io/en/stable/strategy-advanced/#stepped-stoploss):

            ```pycon
            >>> @njit
            ... def adjust_sl_func_nb(c):
            ...     current_profit = (c.val_price_now - c.init_price) / c.init_price
            ...     if current_profit >= 0.40:
            ...         return 0.25, True
            ...     elif current_profit >= 0.25:
            ...         return 0.15, True
            ...     elif current_profit >= 0.20:
            ...         return 0.07, True
            ...     return c.curr_stop, c.curr_trail

            >>> close = pd.Series([10, 11, 12, 11, 10])
            >>> pf = vbt.Portfolio.from_signals(close, adjust_sl_func_nb=adjust_sl_func_nb)
            >>> pf.asset_flow()
            0    10.0
            1     0.0
            2     0.0
            3   -10.0  # 7% from 12 hit
            4    11.0
            dtype: float64
            ```

            * Sometimes there is a need to provide or transform signals dynamically. For this, we can implement
            a custom signal function `signal_func_nb`. For example, let's implement a signal function that
            takes two numerical arrays - long and short one - and transforms them into 4 direction-aware boolean
            arrays that vectorbt understands:

            ```pycon
            >>> @njit
            ... def signal_func_nb(c, long_num_arr, short_num_arr):
            ...     long_num = nb.get_elem_nb(c, long_num_arr)
            ...     short_num = nb.get_elem_nb(c, short_num_arr)
            ...     is_long_entry = long_num > 0
            ...     is_long_exit = long_num < 0
            ...     is_short_entry = short_num > 0
            ...     is_short_exit = short_num < 0
            ...     return is_long_entry, is_long_exit, is_short_entry, is_short_exit

            >>> pf = vbt.Portfolio.from_signals(
            ...     pd.Series([1, 2, 3, 4, 5]),
            ...     signal_func_nb=signal_func_nb,
            ...     signal_args=(vbt.Rep('long_num_arr'), vbt.Rep('short_num_arr')),
            ...     broadcast_named_args=dict(
            ...         long_num_arr=pd.Series([1, 0, -1, 0, 0]),
            ...         short_num_arr=pd.Series([0, 1, 0, 1, -1])
            ...     ),
            ...     size=1,
            ...     upon_opposite_entry='ignore'
            ... )
            >>> pf.asset_flow()
            0    1.0
            1    0.0
            2   -1.0
            3   -1.0
            4    1.0
            dtype: float64
            ```

            Passing both arrays as `broadcast_named_args` broadcasts them internally as any other array,
            so we don't have to worry about their dimensions every time we change our data.
        """
        # Get defaults
        from vectorbt._settings import settings
        portfolio_cfg = settings['portfolio']

        ls_mode = short_entries is not None or short_exits is not None
        signal_func_mode = signal_func_nb is not nb.no_signal_func_nb
        if (entries is not None or exits is not None or ls_mode) and signal_func_mode:
            raise ValueError("Either any of the signal arrays or signal_func_nb should be set, not both")
        if entries is None:
            if exits is None and not ls_mode:
                entries = True
            else:
                entries = False
        if exits is None:
            exits = False
        if short_entries is None:
            short_entries = False
        if short_exits is None:
            short_exits = False
        if signal_func_nb is nb.no_signal_func_nb:
            if ls_mode:
                signal_func_nb = nb.ls_enex_signal_func_nb
            else:
                signal_func_nb = nb.dir_enex_signal_func_nb
        if size is None:
            size = portfolio_cfg['size']
        if size_type is None:
            size_type = portfolio_cfg['size_type']
        size_type = map_enum_fields(size_type, SizeType)
        if price is None:
            price = np.inf
        if fees is None:
            fees = portfolio_cfg['fees']
        if fixed_fees is None:
            fixed_fees = portfolio_cfg['fixed_fees']
        if slippage is None:
            slippage = portfolio_cfg['slippage']
        if min_size is None:
            min_size = portfolio_cfg['min_size']
        if max_size is None:
            max_size = portfolio_cfg['max_size']
        if size_granularity is None:
            size_granularity = portfolio_cfg['size_granularity']
        if reject_prob is None:
            reject_prob = portfolio_cfg['reject_prob']
        if lock_cash is None:
            lock_cash = portfolio_cfg['lock_cash']
        if allow_partial is None:
            allow_partial = portfolio_cfg['allow_partial']
        if raise_reject is None:
            raise_reject = portfolio_cfg['raise_reject']
        if log is None:
            log = portfolio_cfg['log']
        if accumulate is None:
            accumulate = portfolio_cfg['accumulate']
        accumulate = map_enum_fields(accumulate, AccumulationMode, ignore_type=(int, bool))
        if upon_long_conflict is None:
            upon_long_conflict = portfolio_cfg['upon_long_conflict']
        upon_long_conflict = map_enum_fields(upon_long_conflict, ConflictMode)
        if upon_short_conflict is None:
            upon_short_conflict = portfolio_cfg['upon_short_conflict']
        upon_short_conflict = map_enum_fields(upon_short_conflict, ConflictMode)
        if upon_dir_conflict is None:
            upon_dir_conflict = portfolio_cfg['upon_dir_conflict']
        upon_dir_conflict = map_enum_fields(upon_dir_conflict, DirectionConflictMode)
        if upon_opposite_entry is None:
            upon_opposite_entry = portfolio_cfg['upon_opposite_entry']
        upon_opposite_entry = map_enum_fields(upon_opposite_entry, OppositeEntryMode)
        if direction is not None and ls_mode:
            warnings.warn("direction has no effect if short_entries and short_exits are set", stacklevel=2)
        if direction is None:
            direction = portfolio_cfg['signal_direction']
        direction = map_enum_fields(direction, Direction)
        if val_price is None:
            val_price = portfolio_cfg['val_price']
        if open is None:
            open = np.nan
        if high is None:
            high = np.nan
        if low is None:
            low = np.nan
        if sl_stop is None:
            sl_stop = portfolio_cfg['sl_stop']
        if sl_trail is None:
            sl_trail = portfolio_cfg['sl_trail']
        if tp_stop is None:
            tp_stop = portfolio_cfg['tp_stop']
        if stop_entry_price is None:
            stop_entry_price = portfolio_cfg['stop_entry_price']
        stop_entry_price = map_enum_fields(stop_entry_price, StopEntryPrice)
        if stop_exit_price is None:
            stop_exit_price = portfolio_cfg['stop_exit_price']
        stop_exit_price = map_enum_fields(stop_exit_price, StopExitPrice)
        if upon_stop_exit is None:
            upon_stop_exit = portfolio_cfg['upon_stop_exit']
        upon_stop_exit = map_enum_fields(upon_stop_exit, StopExitMode)
        if upon_stop_update is None:
            upon_stop_update = portfolio_cfg['upon_stop_update']
        upon_stop_update = map_enum_fields(upon_stop_update, StopUpdateMode)
        if use_stops is None:
            use_stops = portfolio_cfg['use_stops']
        if use_stops is None:
            if isinstance(sl_stop, float) and \
                    np.isnan(sl_stop) and \
                    isinstance(tp_stop, float) and \
                    np.isnan(tp_stop) and \
                    adjust_sl_func_nb == nb.no_adjust_sl_func_nb and \
                    adjust_tp_func_nb == nb.no_adjust_tp_func_nb:
                use_stops = False
            else:
                use_stops = True

        if init_cash is None:
            init_cash = portfolio_cfg['init_cash']
        if isinstance(init_cash, str):
            init_cash = map_enum_fields(init_cash, InitCashMode)
        if isinstance(init_cash, int) and init_cash in InitCashMode:
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if cash_sharing is None:
            cash_sharing = portfolio_cfg['cash_sharing']
        if cash_sharing and group_by is None:
            group_by = True
        if call_seq is None:
            call_seq = portfolio_cfg['call_seq']
        auto_call_seq = False
        if isinstance(call_seq, str):
            call_seq = map_enum_fields(call_seq, CallSeqType)
        if isinstance(call_seq, int):
            if call_seq == CallSeqType.Auto:
                call_seq = CallSeqType.Default
                auto_call_seq = True
        if ffill_val_price is None:
            ffill_val_price = portfolio_cfg['ffill_val_price']
        if update_value is None:
            update_value = portfolio_cfg['update_value']
        if seed is None:
            seed = portfolio_cfg['seed']
        if seed is not None:
            set_seed(seed)
        if freq is None:
            freq = portfolio_cfg['freq']
        if attach_call_seq is None:
            attach_call_seq = portfolio_cfg['attach_call_seq']
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_mapping is None:
            template_mapping = {}
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")

        # Prepare the simulation
        broadcastable_args = dict(
            size=size,
            price=price,
            size_type=size_type,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            reject_prob=reject_prob,
            lock_cash=lock_cash,
            allow_partial=allow_partial,
            raise_reject=raise_reject,
            log=log,
            accumulate=accumulate,
            upon_long_conflict=upon_long_conflict,
            upon_short_conflict=upon_short_conflict,
            upon_dir_conflict=upon_dir_conflict,
            upon_opposite_entry=upon_opposite_entry,
            val_price=val_price,
            open=open,
            high=high,
            low=low,
            close=close,
            sl_stop=sl_stop,
            sl_trail=sl_trail,
            tp_stop=tp_stop,
            stop_entry_price=stop_entry_price,
            stop_exit_price=stop_exit_price,
            upon_stop_exit=upon_stop_exit,
            upon_stop_update=upon_stop_update
        )
        if not signal_func_mode:
            if ls_mode:
                broadcastable_args['entries'] = entries
                broadcastable_args['exits'] = exits
                broadcastable_args['short_entries'] = short_entries
                broadcastable_args['short_exits'] = short_exits
            else:
                broadcastable_args['entries'] = entries
                broadcastable_args['exits'] = exits
                broadcastable_args['direction'] = direction
        broadcastable_args = {**broadcastable_args, **broadcast_named_args}
        # Only close is broadcast, others can remain unchanged thanks to flexible indexing
        close_idx = list(broadcastable_args.keys()).index('close')
        keep_raw = [True] * len(broadcastable_args)
        keep_raw[close_idx] = False
        broadcast_kwargs = merge_dicts(dict(
            keep_raw=keep_raw,
            require_kwargs=dict(requirements='W')
        ), broadcast_kwargs)
        broadcasted_args = broadcast(*broadcastable_args.values(), **broadcast_kwargs)
        broadcasted_args = dict(zip(broadcastable_args.keys(), broadcasted_args))
        close = broadcasted_args['close']
        if not checks.is_pandas(close):
            close = pd.Series(close) if close.ndim == 1 else pd.DataFrame(close)
        broadcasted_args['close'] = to_2d_array(close)
        target_shape_2d = (close.shape[0], close.shape[1] if close.ndim > 1 else 1)
        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_lens = wrapper.grouper.get_group_lens(group_by=None if cash_sharing else False)
        init_cash = np.require(np.broadcast_to(init_cash, (len(cs_group_lens),)), dtype=np.float64)
        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        if checks.is_any_array(call_seq):
            call_seq = nb.require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
        else:
            call_seq = nb.build_call_seq(target_shape_2d, group_lens, call_seq_type=call_seq)
        if max_orders is None:
            max_orders = target_shape_2d[0] * target_shape_2d[1]
        if max_logs is None:
            max_logs = target_shape_2d[0] * target_shape_2d[1]
        if not np.any(log):
            max_logs = 1
        template_mapping = {**broadcasted_args, **dict(
            target_shape=target_shape_2d,
            group_lens=cs_group_lens,
            init_cash=init_cash,
            call_seq=call_seq,
            adjust_sl_func_nb=adjust_sl_func_nb,
            adjust_sl_args=adjust_sl_args,
            adjust_tp_func_nb=adjust_tp_func_nb,
            adjust_tp_args=adjust_tp_args,
            use_stops=use_stops,
            auto_call_seq=auto_call_seq,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            max_orders=max_orders,
            max_logs=max_logs,
            flex_2d=close.ndim == 2,
            wrapper=wrapper
        ), **template_mapping}
        adjust_sl_args = deep_substitute(adjust_sl_args, template_mapping)
        adjust_tp_args = deep_substitute(adjust_tp_args, template_mapping)
        if signal_func_mode:
            signal_args = deep_substitute(signal_args, template_mapping)
        else:
            if ls_mode:
                signal_args = (
                    broadcasted_args['entries'],
                    broadcasted_args['exits'],
                    broadcasted_args['short_entries'],
                    broadcasted_args['short_exits']
                )
            else:
                signal_args = (
                    broadcasted_args['entries'],
                    broadcasted_args['exits'],
                    broadcasted_args['direction']
                )
        checks.assert_numba_func(signal_func_nb)
        checks.assert_numba_func(adjust_sl_func_nb)
        checks.assert_numba_func(adjust_tp_func_nb)

        # Perform the simulation
        order_records, log_records = nb.simulate_from_signal_func_nb(
            target_shape_2d,
            cs_group_lens,  # group only if cash sharing is enabled to speed up
            init_cash,
            call_seq,
            signal_func_nb=signal_func_nb,
            signal_args=signal_args,
            size=broadcasted_args['size'],
            price=broadcasted_args['price'],
            size_type=broadcasted_args['size_type'],
            fees=broadcasted_args['fees'],
            fixed_fees=broadcasted_args['fixed_fees'],
            slippage=broadcasted_args['slippage'],
            min_size=broadcasted_args['min_size'],
            max_size=broadcasted_args['max_size'],
            size_granularity=broadcasted_args['size_granularity'],
            reject_prob=broadcasted_args['reject_prob'],
            lock_cash=broadcasted_args['lock_cash'],
            allow_partial=broadcasted_args['allow_partial'],
            raise_reject=broadcasted_args['raise_reject'],
            log=broadcasted_args['log'],
            accumulate=broadcasted_args['accumulate'],
            upon_long_conflict=broadcasted_args['upon_long_conflict'],
            upon_short_conflict=broadcasted_args['upon_short_conflict'],
            upon_dir_conflict=broadcasted_args['upon_dir_conflict'],
            upon_opposite_entry=broadcasted_args['upon_opposite_entry'],
            val_price=broadcasted_args['val_price'],
            open=broadcasted_args['open'],
            high=broadcasted_args['high'],
            low=broadcasted_args['low'],
            close=broadcasted_args['close'],
            sl_stop=broadcasted_args['sl_stop'],
            sl_trail=broadcasted_args['sl_trail'],
            tp_stop=broadcasted_args['tp_stop'],
            stop_entry_price=broadcasted_args['stop_entry_price'],
            stop_exit_price=broadcasted_args['stop_exit_price'],
            upon_stop_exit=broadcasted_args['upon_stop_exit'],
            upon_stop_update=broadcasted_args['upon_stop_update'],
            adjust_sl_func_nb=adjust_sl_func_nb,
            adjust_sl_args=adjust_sl_args,
            adjust_tp_func_nb=adjust_tp_func_nb,
            adjust_tp_args=adjust_tp_args,
            use_stops=use_stops,
            auto_call_seq=auto_call_seq,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            max_orders=max_orders,
            max_logs=max_logs,
            flex_2d=close.ndim == 2
        )

        # Create an instance
        return cls(
            wrapper,
            close,
            order_records,
            log_records,
            init_cash if init_cash_mode is None else init_cash_mode,
            cash_sharing,
            call_seq=call_seq if attach_call_seq else None,
            **kwargs
        )

    @classmethod  # 类方法装饰器
    def from_holding(cls: tp.Type[PortfolioT], close: tp.ArrayLike, **kwargs) -> PortfolioT:
        """
        从持有策略模拟投资组合（买入并持有）
        
        这是一个简化的投资组合构建方法，实现买入并持有策略。
        在第一个时点买入并一直持有到最后，不进行任何交易。
        
        Args:
            close: 收盘价数据
            **kwargs: 传递给from_signals的其他参数
            
        Returns:
            Portfolio对象，实现买入并持有策略
            
        示例:
            >>> close = pd.Series([1, 2, 3, 4, 5])  # 价格从1涨到5
            >>> pf = vbt.Portfolio.from_holding(close)  # 买入并持有
            >>> pf.final_value()  # 最终价值：初始现金100，全部买入100股，最终价值=100*5=500
            500.0
            
        Simulate portfolio from holding.

        基于 `Portfolio.from_signals`。
        """
        return cls.from_signals(close, entries=True, exits=False, **kwargs)

    @classmethod  # 类方法装饰器
    def from_random_signals(cls: tp.Type[PortfolioT],
                            close: tp.ArrayLike,  # 收盘价数据
                            n: tp.Optional[tp.ArrayLike] = None,  # 信号数量
                            prob: tp.Optional[tp.ArrayLike] = None,  # 信号概率
                            entry_prob: tp.Optional[tp.ArrayLike] = None,  # 入场概率
                            exit_prob: tp.Optional[tp.ArrayLike] = None,  # 出场概率
                            param_product: bool = False,  # 是否生成参数乘积组合
                            seed: tp.Optional[int] = None,  # 随机种子
                            run_kwargs: tp.KwargsLike = None,  # 运行参数
                            **kwargs) -> PortfolioT:
        """
        从随机入场和出场信号模拟投资组合
        
        这个方法用于测试和分析随机交易策略的效果。通过生成随机的交易信号
        来模拟投资组合表现，常用于策略回测和风险分析。
        
        Args:
            close: 收盘价数据
            n: 信号数量，如果设置则基于固定信号数量生成
            prob: 遇到信号的概率，如果设置则基于概率生成信号  
            entry_prob: 入场信号的概率
            exit_prob: 出场信号的概率
            param_product: 是否生成参数的乘积组合
            seed: 随机种子，确保结果可重现
            run_kwargs: 运行参数字典
            **kwargs: 传递给from_signals的其他参数
            
        Returns:
            Portfolio对象，基于随机信号的投资组合
            
        信号生成方式:
        * 如果设置了 `n`，参见 `vectorbt.signals.generators.RANDNX`
        * 如果设置了 `prob`，参见 `vectorbt.signals.generators.RPROBNX`

        基于 `Portfolio.from_signals`。

        !!! note
            生成随机信号时使用 `close` 的形状。与其他数组的广播在生成后进行。

        使用示例 (Usage):
            * 测试随机入场和出场的多种组合：

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_random_signals(close, n=[2, 1, 0], seed=42)
            >>> pf.orders.count()
            randnx_n
            2    4
            1    2
            0    0
            Name: count, dtype: int64
            ```

            * Test the Cartesian product of entry and exit encounter probabilities:

            ```pycon
            >>> pf = vbt.Portfolio.from_random_signals(
            ...     close,
            ...     entry_prob=[0, 0.5, 1],
            ...     exit_prob=[0, 0.5, 1],
            ...     param_product=True,
            ...     seed=42)
            >>> pf.orders.count()
            rprobnx_entry_prob  rprobnx_exit_prob
            0.0                 0.0                  0
                                0.5                  0
                                1.0                  0
            0.5                 0.0                  1
                                0.5                  4
                                1.0                  3
            1.0                 0.0                  1
                                0.5                  4
                                1.0                  5
            Name: count, dtype: int64
            ```
        """
        from vectorbt._settings import settings
        portfolio_cfg = settings['portfolio']

        close = to_pd_array(close)
        close_wrapper = ArrayWrapper.from_obj(close)
        if entry_prob is None:
            entry_prob = prob
        if exit_prob is None:
            exit_prob = prob
        if seed is None:
            seed = portfolio_cfg['seed']
        if run_kwargs is None:
            run_kwargs = {}

        if n is not None and (entry_prob is not None or exit_prob is not None):
            raise ValueError("Either n or entry_prob and exit_prob should be set")
        if n is not None:
            rand = RANDNX.run(
                n=n,
                input_shape=close.shape,
                input_index=close_wrapper.index,
                input_columns=close_wrapper.columns,
                seed=seed,
                **run_kwargs
            )
            entries = rand.entries
            exits = rand.exits
        elif entry_prob is not None and exit_prob is not None:
            rprobnx = RPROBNX.run(
                entry_prob=entry_prob,
                exit_prob=exit_prob,
                param_product=param_product,
                input_shape=close.shape,
                input_index=close_wrapper.index,
                input_columns=close_wrapper.columns,
                seed=seed,
                **run_kwargs
            )
            entries = rprobnx.entries
            exits = rprobnx.exits
        else:
            raise ValueError("At least n or entry_prob and exit_prob should be set")

        return cls.from_signals(close, entries, exits, seed=seed, **kwargs)

    @classmethod  # 类方法装饰器
    def from_order_func(cls: tp.Type[PortfolioT],
                        close: tp.ArrayLike,  # 收盘价数据
                        order_func_nb: tp.Union[nb.OrderFuncT, nb.FlexOrderFuncT],  # 订单生成函数
                        *order_args,  # 传递给订单函数的参数
                        flexible: tp.Optional[bool] = None,  # 是否使用灵活模式
                        init_cash: tp.Optional[tp.ArrayLike] = None,  # 初始现金
                        cash_sharing: tp.Optional[bool] = None,  # 现金共享
                        call_seq: tp.Optional[tp.ArrayLike] = None,  # 调用序列
                        segment_mask: tp.Optional[tp.ArrayLike] = None,  # 段掩码
                        call_pre_segment: tp.Optional[bool] = None,  # 调用段前函数
                        call_post_segment: tp.Optional[bool] = None,  # 调用段后函数
                        # 模拟生命周期回调函数
                        pre_sim_func_nb: nb.PreSimFuncT = nb.no_pre_func_nb,  # 模拟前函数
                        pre_sim_args: tp.Args = (),  # 模拟前函数参数
                        post_sim_func_nb: nb.PostSimFuncT = nb.no_post_func_nb,  # 模拟后函数
                        post_sim_args: tp.Args = (),  # 模拟后函数参数
                        pre_group_func_nb: nb.PreGroupFuncT = nb.no_pre_func_nb,  # 组前函数
                        pre_group_args: tp.Args = (),  # 组前函数参数
                        post_group_func_nb: nb.PostGroupFuncT = nb.no_post_func_nb,  # 组后函数
                        post_group_args: tp.Args = (),  # 组后函数参数
                        pre_row_func_nb: nb.PreRowFuncT = nb.no_pre_func_nb,  # 行前函数
                        pre_row_args: tp.Args = (),  # 行前函数参数
                        post_row_func_nb: nb.PostRowFuncT = nb.no_post_func_nb,  # 行后函数
                        post_row_args: tp.Args = (),  # 行后函数参数
                        pre_segment_func_nb: nb.PreSegmentFuncT = nb.no_pre_func_nb,  # 段前函数
                        pre_segment_args: tp.Args = (),  # 段前函数参数
                        post_segment_func_nb: nb.PostSegmentFuncT = nb.no_post_func_nb,  # 段后函数
                        post_segment_args: tp.Args = (),  # 段后函数参数
                        post_order_func_nb: nb.PostOrderFuncT = nb.no_post_func_nb,  # 订单后函数
                        post_order_args: tp.Args = (),  # 订单后函数参数
                        # 模拟控制参数
                        ffill_val_price: tp.Optional[bool] = None,  # 前向填充估值价格
                        update_value: tp.Optional[bool] = None,  # 更新价值
                        fill_pos_record: tp.Optional[bool] = None,  # 填充仓位记录
                        row_wise: tp.Optional[bool] = None,  # 按行处理
                        use_numba: tp.Optional[bool] = None,  # 使用Numba加速
                        max_orders: tp.Optional[int] = None,  # 最大订单数
                        max_logs: tp.Optional[int] = None,  # 最大日志数
                        seed: tp.Optional[int] = None,  # 随机种子
                        group_by: tp.GroupByLike = None,  # 分组方式
                        broadcast_named_args: tp.KwargsLike = None,  # 广播命名参数
                        broadcast_kwargs: tp.KwargsLike = None,  # 广播参数
                        template_mapping: tp.Optional[tp.Mapping] = None,  # 模板映射
                        wrapper_kwargs: tp.KwargsLike = None,  # 包装器参数
                        freq: tp.Optional[tp.FrequencyLike] = None,  # 数据频率
                        attach_call_seq: tp.Optional[bool] = None,  # 附加调用序列
                        **kwargs) -> PortfolioT:
        """
        从自定义订单函数构建投资组合
        
        这是vectorbt中最强大和灵活的投资组合构建方法，允许用户完全自定义
        订单生成逻辑。通过提供一个订单函数，可以实现任意复杂的交易策略。
        
        这个方法提供了完整的模拟生命周期回调系统，可以在模拟的不同阶段
        插入自定义逻辑，实现高度定制化的策略开发。
        
        Args:
            close (array_like): 每个时间步的最后资产价格
                将广播到目标形状。用于计算未实现盈亏和投资组合价值。
                
            order_func_nb (callable): 订单生成函数
                这是核心函数，决定何时何地生成什么样的订单。
                
            *order_args: 传递给 `order_func_nb` 的参数
            
            flexible (bool): 是否使用灵活的订单函数进行模拟
                这消除了每个时点和代码只能有一个订单的限制。
                
            init_cash (InitCashMode, float or array_like of float): 初始资本
                参见 `Portfolio.from_orders` 中的 `init_cash`。
                
            cash_sharing (bool): 是否在同一组内共享现金
                如果 `group_by` 为 None，`group_by` 变为 True 形成带现金共享的单一组。
                
            call_seq (CallSeqType or array_like): 每行每组的默认调用序列
                * 使用 `vectorbt.portfolio.enums.CallSeqType` 选择序列类型
                * 设置为数组以指定自定义序列。不会广播。

                !!! note
                    CallSeqType.Auto 应手动实现。
                    在 `pre_segment_func_nb` 中使用 `vectorbt.portfolio.nb.sort_call_seq_nb` 
                    或 `vectorbt.portfolio.nb.sort_call_seq_out_nb`。
                    
            segment_mask (int or array_like of bool): 特定段是否应执行的掩码
                提供整数将激活每第n行。
                提供布尔值或布尔数组将广播到行数和组数。
                不与 `close` 和 `broadcast_named_args` 一起广播，仅对最终形状广播。
                
            call_pre_segment (bool): 是否无论 `segment_mask` 如何都调用 `pre_segment_func_nb`
            call_post_segment (bool): 是否无论 `segment_mask` 如何都调用 `post_segment_func_nb`
            
            # 生命周期回调函数 - 提供完整的模拟控制能力
            pre_sim_func_nb (callable): 模拟开始前调用的函数
                默认为 `vectorbt.portfolio.nb.no_pre_func_nb`。
            pre_sim_args (tuple): 传递给 `pre_sim_func_nb` 的打包参数
                默认为 `()`。
            post_sim_func_nb (callable): 模拟结束后调用的函数
                默认为 `vectorbt.portfolio.nb.no_post_func_nb`。
            post_sim_args (tuple): 传递给 `post_sim_func_nb` 的打包参数
                默认为 `()`。
            pre_group_func_nb (callable): 每个组开始前调用的函数
                默认为 `vectorbt.portfolio.nb.no_pre_func_nb`。

        Build portfolio from a custom order function.

        !!! hint
            参见 `vectorbt.portfolio.nb.simulate_nb` 了解图解和参数定义。

        个别模拟函数的详细信息:

        * 非 `row_wise` 且非 `flexible`: 参见 `vectorbt.portfolio.nb.simulate_nb`
        * 非 `row_wise` 且 `flexible`: 参见 `vectorbt.portfolio.nb.flex_simulate_nb`
        * `row_wise` 且非 `flexible`: 参见 `vectorbt.portfolio.nb.simulate_row_wise_nb`
        * `row_wise` 且 `flexible`: 参见 `vectorbt.portfolio.nb.flex_simulate_row_wise_nb`

                Called only if `row_wise` is False.
            pre_group_args (tuple): Packed arguments passed to `pre_group_func_nb`.
                Defaults to `()`.
            post_group_func_nb (callable): Function called after each group.
                Defaults to `vectorbt.portfolio.nb.no_post_func_nb`.

                Called only if `row_wise` is False.
            post_group_args (tuple): Packed arguments passed to `post_group_func_nb`.
                Defaults to `()`.
            pre_row_func_nb (callable): Function called before each row.
                Defaults to `vectorbt.portfolio.nb.no_pre_func_nb`.

                Called only if `row_wise` is True.
            pre_row_args (tuple): Packed arguments passed to `pre_row_func_nb`.
                Defaults to `()`.
            post_row_func_nb (callable): Function called after each row.
                Defaults to `vectorbt.portfolio.nb.no_post_func_nb`.

                Called only if `row_wise` is True.
            post_row_args (tuple): Packed arguments passed to `post_row_func_nb`.
                Defaults to `()`.
            pre_segment_func_nb (callable): Function called before each segment.
                Defaults to `vectorbt.portfolio.nb.no_pre_func_nb`.
            pre_segment_args (tuple): Packed arguments passed to `pre_segment_func_nb`.
                Defaults to `()`.
            post_segment_func_nb (callable): Function called after each segment.
                Defaults to `vectorbt.portfolio.nb.no_post_func_nb`.
            post_segment_args (tuple): Packed arguments passed to `post_segment_func_nb`.
                Defaults to `()`.
            post_order_func_nb (callable): Callback that is called after the order has been processed.
            post_order_args (tuple): Packed arguments passed to `post_order_func_nb`.
                Defaults to `()`.
            ffill_val_price (bool): Whether to track valuation price only if it's known.

                Otherwise, unknown `close` will lead to NaN in valuation price at the next timestamp.
            update_value (bool): Whether to update group value after each filled order.
            fill_pos_record (bool): Whether to fill position record.

                Disable this to make simulation a bit faster for simple use cases.
            row_wise (bool): Whether to iterate over rows rather than columns/groups.
            use_numba (bool): Whether to run the main simulation function using Numba.

                !!! note
                    Disabling it does not disable Numba for other functions.
                    If necessary, you should ensure that every other function does not uses Numba as well.
                    You can do this by using the `py_func` attribute of that function.
                    Or, you could disable Numba globally by doing `os.environ['NUMBA_DISABLE_JIT'] = '1'`.
            max_orders (int): Size of the order records array.
                Defaults to the number of elements in the broadcasted shape.

                Set to a lower number if you run out of memory.
            max_logs (int): Size of the log records array.
                Defaults to the number of elements in the broadcasted shape.

                Set to a lower number if you run out of memory.
            seed (int): See `Portfolio.from_orders`.
            group_by (any): See `Portfolio.from_orders`.
            broadcast_named_args (dict): See `Portfolio.from_signals`.
            broadcast_kwargs (dict): See `Portfolio.from_orders`.
            template_mapping (mapping): See `Portfolio.from_signals`.
            wrapper_kwargs (dict): See `Portfolio.from_orders`.
            freq (any): See `Portfolio.from_orders`.
            attach_call_seq (bool): See `Portfolio.from_orders`.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `portfolio` in `vectorbt._settings.settings`.

        !!! note
            All passed functions should be Numba-compiled if Numba is enabled.

            Also see notes on `Portfolio.from_orders`.

        !!! note
            In contrast to other methods, the valuation price is previous `close` instead of the order price
            since the price of an order is unknown before the call (which is more realistic by the way).
            You can still override the valuation price in `pre_segment_func_nb`.

        Usage:
            * Buy 10 units each tick using closing price:

            ```pycon
            >>> @njit
            ... def order_func_nb(c, size):
            ...     return nb.order_nb(size=size)

            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_order_func(close, order_func_nb, 10)

            >>> pf.assets()
            0    10.0
            1    20.0
            2    30.0
            3    40.0
            4    40.0
            dtype: float64
            >>> pf.cash()
            0    90.0
            1    70.0
            2    40.0
            3     0.0
            4     0.0
            dtype: float64
            ```

            * Reverse each position by first closing it. Keep state of last position to determine
            which position to open next (just as an example, there are easier ways to do this):

            ```pycon
            >>> @njit
            ... def pre_group_func_nb(c):
            ...     last_pos_state = np.array([-1])
            ...     return (last_pos_state,)

            >>> @njit
            ... def order_func_nb(c, last_pos_state):
            ...     if c.position_now != 0:
            ...         return nb.close_position_nb()
            ...
            ...     if last_pos_state[0] == 1:
            ...         size = -np.inf  # open short
            ...         last_pos_state[0] = -1
            ...     else:
            ...         size = np.inf  # open long
            ...         last_pos_state[0] = 1
            ...     return nb.order_nb(size=size)

            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     order_func_nb,
            ...     pre_group_func_nb=pre_group_func_nb
            ... )

            >>> pf.assets()
            0    100.000000
            1      0.000000
            2    -66.666667
            3      0.000000
            4     26.666667
            dtype: float64
            >>> pf.cash()
            0      0.000000
            1    200.000000
            2    400.000000
            3    133.333333
            4      0.000000
            dtype: float64
            ```

            * Equal-weighted portfolio as in the example under `vectorbt.portfolio.nb.simulate_nb`:

            ```pycon
            >>> @njit
            ... def pre_group_func_nb(c):
            ...     order_value_out = np.empty(c.group_len, dtype=np.float64)
            ...     return (order_value_out,)

            >>> @njit
            ... def pre_segment_func_nb(c, order_value_out, size, price, size_type, direction):
            ...     for col in range(c.from_col, c.to_col):
            ...         c.last_val_price[col] = nb.get_col_elem_nb(c, col, price)
            ...     nb.sort_call_seq_nb(c, size, size_type, direction, order_value_out)
            ...     return ()

            >>> @njit
            ... def order_func_nb(c, size, price, size_type, direction, fees, fixed_fees, slippage):
            ...     return nb.order_nb(
            ...         size=nb.get_elem_nb(c, size),
            ...         price=nb.get_elem_nb(c, price),
            ...         size_type=nb.get_elem_nb(c, size_type),
            ...         direction=nb.get_elem_nb(c, direction),
            ...         fees=nb.get_elem_nb(c, fees),
            ...         fixed_fees=nb.get_elem_nb(c, fixed_fees),
            ...         slippage=nb.get_elem_nb(c, slippage)
            ...     )

            >>> np.random.seed(42)
            >>> close = np.random.uniform(1, 10, size=(5, 3))
            >>> size_template = vbt.RepEval('np.asarray(1 / group_lens[0])')

            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     order_func_nb,
            ...     size_template,  # order_args as *args
            ...     vbt.Rep('price'),
            ...     vbt.Rep('size_type'),
            ...     vbt.Rep('direction'),
            ...     vbt.Rep('fees'),
            ...     vbt.Rep('fixed_fees'),
            ...     vbt.Rep('slippage'),
            ...     segment_mask=2,  # rebalance every second tick
            ...     pre_group_func_nb=pre_group_func_nb,
            ...     pre_segment_func_nb=pre_segment_func_nb,
            ...     pre_segment_args=(
            ...         size_template,
            ...         vbt.Rep('price'),
            ...         vbt.Rep('size_type'),
            ...         vbt.Rep('direction')
            ...     ),
            ...     broadcast_named_args=dict(  # broadcast against each other
            ...         price=close,
            ...         size_type=SizeType.TargetPercent,
            ...         direction=Direction.LongOnly,
            ...         fees=0.001,
            ...         fixed_fees=1.,
            ...         slippage=0.001
            ...     ),
            ...     template_mapping=dict(np=np),  # required by size_template
            ...     cash_sharing=True, group_by=True,  # one group with cash sharing
            ... )

            >>> pf.asset_value(group_by=False).vbt.plot()
            ```

            ![](/assets/images/simulate_nb.svg)

            Templates are a very powerful tool to prepare any custom arguments after they are broadcast and
            before they are passed to the simulation function. In the example above, we use `broadcast_named_args`
            to broadcast some arguments against each other and templates to pass those objects to callbacks.
            Additionally, we used an evaluation template to compute the size based on the number of assets in each group.

            You may ask: why should we bother using broadcasting and templates if we could just pass `size=1/3`?
            Because of flexibility those features provide: we can now pass whatever parameter combinations we want
            and it will work flawlessly. For example, to create two groups of equally-allocated positions,
            we need to change only two parameters:

            ```pycon
            >>> close = np.random.uniform(1, 10, size=(5, 6))  # 6 columns instead of 3
            >>> group_by = ['g1', 'g1', 'g1', 'g2', 'g2', 'g2']  # 2 groups instead of 1

            >>> pf['g1'].asset_value(group_by=False).vbt.plot()
            >>> pf['g2'].asset_value(group_by=False).vbt.plot()
            ```

            ![](/assets/images/from_order_func_g1.svg)

            ![](/assets/images/from_order_func_g2.svg)

            * Combine multiple exit conditions. Exit early if the price hits some threshold before an actual exit:

            ```pycon
            >>> @njit
            ... def pre_sim_func_nb(c):
            ...     # We need to define stop price per column once
            ...     stop_price = np.full(c.target_shape[1], np.nan, dtype=np.float64)
            ...     return (stop_price,)

            >>> @njit
            ... def order_func_nb(c, stop_price, entries, exits, size):
            ...     # Select info related to this order
            ...     entry_now = nb.get_elem_nb(c, entries)
            ...     exit_now = nb.get_elem_nb(c, exits)
            ...     size_now = nb.get_elem_nb(c, size)
            ...     price_now = nb.get_elem_nb(c, c.close)
            ...     stop_price_now = stop_price[c.col]
            ...
            ...     # Our logic
            ...     if entry_now:
            ...         if c.position_now == 0:
            ...             return nb.order_nb(
            ...                 size=size_now,
            ...                 price=price_now,
            ...                 direction=Direction.LongOnly)
            ...     elif exit_now or price_now >= stop_price_now:
            ...         if c.position_now > 0:
            ...             return nb.order_nb(
            ...                 size=-size_now,
            ...                 price=price_now,
            ...                 direction=Direction.LongOnly)
            ...     return NoOrder

            >>> @njit
            ... def post_order_func_nb(c, stop_price, stop):
            ...     # Same broadcasting as for size
            ...     stop_now = nb.get_elem_nb(c, stop)
            ...
            ...     if c.order_result.status == OrderStatus.Filled:
            ...         if c.order_result.side == OrderSide.Buy:
            ...             # Position entered: Set stop condition
            ...             stop_price[c.col] = (1 + stop_now) * c.order_result.price
            ...         else:
            ...             # Position exited: Remove stop condition
            ...             stop_price[c.col] = np.nan

            >>> def simulate(close, entries, exits, size, threshold):
            ...     return vbt.Portfolio.from_order_func(
            ...         close,
            ...         order_func_nb,
            ...         vbt.Rep('entries'), vbt.Rep('exits'), vbt.Rep('size'),  # order_args
            ...         pre_sim_func_nb=pre_sim_func_nb,
            ...         post_order_func_nb=post_order_func_nb,
            ...         post_order_args=(vbt.Rep('threshold'),),
            ...         broadcast_named_args=dict(  # broadcast against each other
            ...             entries=entries,
            ...             exits=exits,
            ...             size=size,
            ...             threshold=threshold
            ...         )
            ...     )

            >>> close = pd.Series([10, 11, 12, 13, 14])
            >>> entries = pd.Series([True, True, False, False, False])
            >>> exits = pd.Series([False, False, False, True, True])
            >>> simulate(close, entries, exits, np.inf, 0.1).asset_flow()
            0    10.0
            1     0.0
            2   -10.0
            3     0.0
            4     0.0
            dtype: float64

            >>> simulate(close, entries, exits, np.inf, 0.2).asset_flow()
            0    10.0
            1     0.0
            2   -10.0
            3     0.0
            4     0.0
            dtype: float64

            >>> simulate(close, entries, exits, np.nan).asset_flow()
            0    10.0
            1     0.0
            2     0.0
            3   -10.0
            4     0.0
            dtype: float64
            ```

            The reason why stop of 10% does not result in an order at the second time step is because
            it comes at the same time as entry, so it must wait until no entry is present.
            This can be changed by replacing the statement "elif" with "if", which would execute
            an exit regardless if an entry is present (similar to using `ConflictMode.Opposite` in
            `Portfolio.from_signals`).

            We can also test the parameter combinations above all at once (thanks to broadcasting):

            ```pycon
            >>> size = pd.DataFrame(
            ...     [[0.1, 0.2, np.nan]],
            ...     columns=pd.Index(['0.1', '0.2', 'nan'], name='size')
            ... )
            >>> simulate(close, entries, exits, np.inf, size).asset_flow()
            size   0.1   0.2   nan
            0     10.0  10.0  10.0
            1      0.0   0.0   0.0
            2    -10.0 -10.0   0.0
            3      0.0   0.0 -10.0
            4      0.0   0.0   0.0
            ```

            * Let's illustrate how to generate multiple orders per symbol and bar.
            For each bar, buy at open and sell at close:

            ```pycon
            >>> @njit
            ... def flex_order_func_nb(c, open, size):
            ...     if c.call_idx == 0:
            ...         return c.from_col, nb.order_nb(size=size, price=open[c.i, c.from_col])
            ...     if c.call_idx == 1:
            ...         return c.from_col, nb.close_position_nb(price=c.close[c.i, c.from_col])
            ...     return -1, NoOrder

            >>> open = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> close = pd.DataFrame({'a': [2, 3, 4], 'b': [3, 4, 5]})
            >>> size = 1
            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     flex_order_func_nb,
            ...     to_2d_array(open), size,
            ...     flexible=True, max_orders=close.shape[0] * close.shape[1] * 2)

            >>> pf.orders.records_readable
                Order Id  Timestamp Column  Size  Price  Fees  Side
            0          0          0      a   1.0    1.0   0.0   Buy
            1          1          0      a   1.0    2.0   0.0  Sell
            2          2          1      a   1.0    2.0   0.0   Buy
            3          3          1      a   1.0    3.0   0.0  Sell
            4          4          2      a   1.0    3.0   0.0   Buy
            5          5          2      a   1.0    4.0   0.0  Sell
            6          6          0      b   1.0    4.0   0.0   Buy
            7          7          0      b   1.0    3.0   0.0  Sell
            8          8          1      b   1.0    5.0   0.0   Buy
            9          9          1      b   1.0    4.0   0.0  Sell
            10        10          2      b   1.0    6.0   0.0   Buy
            11        11          2      b   1.0    5.0   0.0  Sell
            ```

            !!! warning
                Each bar is effectively a black box - we don't know how the price moves inside.
                Since trades must come in an order that replicates that of the real world, the only reliable
                pieces of information are the opening and the closing price.
        """
        # Get defaults
        from vectorbt._settings import settings
        portfolio_cfg = settings['portfolio']

        close = to_pd_array(close)
        if flexible is None:
            flexible = portfolio_cfg['flexible']
        if init_cash is None:
            init_cash = portfolio_cfg['init_cash']
        if isinstance(init_cash, str):
            init_cash = map_enum_fields(init_cash, InitCashMode)
        if isinstance(init_cash, int) and init_cash in InitCashMode:
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if cash_sharing is None:
            cash_sharing = portfolio_cfg['cash_sharing']
        if cash_sharing and group_by is None:
            group_by = True
        if not flexible:
            if call_seq is None:
                call_seq = portfolio_cfg['call_seq']
            call_seq = map_enum_fields(call_seq, CallSeqType)
            if isinstance(call_seq, int):
                if call_seq == CallSeqType.Auto:
                    raise ValueError("CallSeqType.Auto must be implemented manually. "
                                     "Use sort_call_seq_nb in pre_segment_func_nb.")
        if segment_mask is None:
            segment_mask = True
        if call_pre_segment is None:
            call_pre_segment = portfolio_cfg['call_pre_segment']
        if call_post_segment is None:
            call_post_segment = portfolio_cfg['call_post_segment']
        if ffill_val_price is None:
            ffill_val_price = portfolio_cfg['ffill_val_price']
        if update_value is None:
            update_value = portfolio_cfg['update_value']
        if fill_pos_record is None:
            fill_pos_record = portfolio_cfg['fill_pos_record']
        if row_wise is None:
            row_wise = portfolio_cfg['row_wise']
        if use_numba is None:
            use_numba = portfolio_cfg['use_numba']
        if seed is None:
            seed = portfolio_cfg['seed']
        if seed is not None:
            set_seed(seed)
        if freq is None:
            freq = portfolio_cfg['freq']
        if attach_call_seq is None:
            attach_call_seq = portfolio_cfg['attach_call_seq']
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        require_kwargs = dict(require_kwargs=dict(requirements='W'))
        broadcast_kwargs = merge_dicts(require_kwargs, broadcast_kwargs)
        if template_mapping is None:
            template_mapping = {}
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")

        # Prepare the simulation
        broadcastable_args = {**dict(close=close), **broadcast_named_args}
        if len(broadcastable_args) > 1:
            close_idx = list(broadcastable_args.keys()).index('close')
            keep_raw = [True] * len(broadcastable_args)
            keep_raw[close_idx] = False
            broadcast_kwargs = merge_dicts(dict(
                keep_raw=keep_raw,
                require_kwargs=dict(requirements='W')
            ), broadcast_kwargs)
            broadcasted_args = broadcast(*broadcastable_args.values(), **broadcast_kwargs)
            broadcasted_args = dict(zip(broadcastable_args.keys(), broadcasted_args))
            close = broadcasted_args['close']
            if not checks.is_pandas(close):
                close = pd.Series(close) if close.ndim == 1 else pd.DataFrame(close)
        else:
            broadcasted_args = broadcastable_args
        broadcasted_args['close'] = to_2d_array(close)
        target_shape_2d = (close.shape[0], close.shape[1] if close.ndim > 1 else 1)
        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_lens = wrapper.grouper.get_group_lens(group_by=None if cash_sharing else False)
        init_cash = np.require(np.broadcast_to(init_cash, (len(cs_group_lens),)), dtype=np.float64)
        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        if isinstance(segment_mask, int):
            _segment_mask = np.full((target_shape_2d[0], len(group_lens)), False)
            _segment_mask[0::segment_mask] = True
            segment_mask = _segment_mask
        else:
            segment_mask = broadcast(
                segment_mask,
                to_shape=(target_shape_2d[0], len(group_lens)),
                to_pd=False,
                **require_kwargs
            )
        if not flexible:
            if checks.is_any_array(call_seq):
                call_seq = nb.require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
            else:
                call_seq = nb.build_call_seq(target_shape_2d, group_lens, call_seq_type=call_seq)
        if max_orders is None:
            max_orders = target_shape_2d[0] * target_shape_2d[1]
        if max_logs is None:
            max_logs = target_shape_2d[0] * target_shape_2d[1]
        template_mapping = {**broadcasted_args, **dict(
            target_shape=target_shape_2d,
            group_lens=group_lens,
            init_cash=init_cash,
            cash_sharing=cash_sharing,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=pre_sim_args,
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=post_sim_args,
            pre_group_func_nb=pre_group_func_nb,
            pre_group_args=pre_group_args,
            post_group_func_nb=post_group_func_nb,
            post_group_args=post_group_args,
            pre_row_func_nb=pre_row_func_nb,
            pre_row_args=pre_row_args,
            post_row_func_nb=post_row_func_nb,
            post_row_args=post_row_args,
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=pre_segment_args,
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=post_segment_args,
            flex_order_func_nb=order_func_nb,
            flex_order_args=order_args,
            post_order_func_nb=post_order_func_nb,
            post_order_args=post_order_args,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            max_orders=max_orders,
            max_logs=max_logs,
            flex_2d=close.ndim == 2,
            wrapper=wrapper
        ), **template_mapping}
        pre_sim_args = deep_substitute(pre_sim_args, template_mapping)
        post_sim_args = deep_substitute(post_sim_args, template_mapping)
        pre_group_args = deep_substitute(pre_group_args, template_mapping)
        post_group_args = deep_substitute(post_group_args, template_mapping)
        pre_row_args = deep_substitute(pre_row_args, template_mapping)
        post_row_args = deep_substitute(post_row_args, template_mapping)
        pre_segment_args = deep_substitute(pre_segment_args, template_mapping)
        post_segment_args = deep_substitute(post_segment_args, template_mapping)
        order_args = deep_substitute(order_args, template_mapping)
        post_order_args = deep_substitute(post_order_args, template_mapping)
        if use_numba:
            checks.assert_numba_func(pre_sim_func_nb)
            checks.assert_numba_func(post_sim_func_nb)
            checks.assert_numba_func(pre_group_func_nb)
            checks.assert_numba_func(post_group_func_nb)
            checks.assert_numba_func(pre_row_func_nb)
            checks.assert_numba_func(post_row_func_nb)
            checks.assert_numba_func(pre_segment_func_nb)
            checks.assert_numba_func(post_segment_func_nb)
            checks.assert_numba_func(order_func_nb)
            checks.assert_numba_func(post_order_func_nb)

        # Perform the simulation
        if row_wise:
            if flexible:
                simulate_func = nb.flex_simulate_row_wise_nb
                if not use_numba and hasattr(simulate_func, 'py_func'):
                    simulate_func = simulate_func.py_func
                order_records, log_records = simulate_func(
                    target_shape=target_shape_2d,
                    group_lens=group_lens,
                    init_cash=init_cash,
                    cash_sharing=cash_sharing,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    pre_sim_func_nb=pre_sim_func_nb,
                    pre_sim_args=pre_sim_args,
                    post_sim_func_nb=post_sim_func_nb,
                    post_sim_args=post_sim_args,
                    pre_row_func_nb=pre_row_func_nb,
                    pre_row_args=pre_row_args,
                    post_row_func_nb=post_row_func_nb,
                    post_row_args=post_row_args,
                    pre_segment_func_nb=pre_segment_func_nb,
                    pre_segment_args=pre_segment_args,
                    post_segment_func_nb=post_segment_func_nb,
                    post_segment_args=post_segment_args,
                    flex_order_func_nb=order_func_nb,
                    flex_order_args=order_args,
                    post_order_func_nb=post_order_func_nb,
                    post_order_args=post_order_args,
                    close=broadcasted_args['close'],
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    max_orders=max_orders,
                    max_logs=max_logs,
                    flex_2d=close.ndim == 2
                )
            else:
                simulate_func = nb.simulate_row_wise_nb
                if not use_numba and hasattr(simulate_func, 'py_func'):
                    simulate_func = simulate_func.py_func
                order_records, log_records = simulate_func(
                    target_shape=target_shape_2d,
                    group_lens=group_lens,
                    init_cash=init_cash,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    pre_sim_func_nb=pre_sim_func_nb,
                    pre_sim_args=pre_sim_args,
                    post_sim_func_nb=post_sim_func_nb,
                    post_sim_args=post_sim_args,
                    pre_row_func_nb=pre_row_func_nb,
                    pre_row_args=pre_row_args,
                    post_row_func_nb=post_row_func_nb,
                    post_row_args=post_row_args,
                    pre_segment_func_nb=pre_segment_func_nb,
                    pre_segment_args=pre_segment_args,
                    post_segment_func_nb=post_segment_func_nb,
                    post_segment_args=post_segment_args,
                    order_func_nb=order_func_nb,
                    order_args=order_args,
                    post_order_func_nb=post_order_func_nb,
                    post_order_args=post_order_args,
                    close=broadcasted_args['close'],
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    max_orders=max_orders,
                    max_logs=max_logs,
                    flex_2d=close.ndim == 2
                )
        else:
            if flexible:
                simulate_func = nb.flex_simulate_nb
                if not use_numba and hasattr(simulate_func, 'py_func'):
                    simulate_func = simulate_func.py_func
                order_records, log_records = simulate_func(
                    target_shape=target_shape_2d,
                    group_lens=group_lens,
                    init_cash=init_cash,
                    cash_sharing=cash_sharing,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    pre_sim_func_nb=pre_sim_func_nb,
                    pre_sim_args=pre_sim_args,
                    post_sim_func_nb=post_sim_func_nb,
                    post_sim_args=post_sim_args,
                    pre_group_func_nb=pre_group_func_nb,
                    pre_group_args=pre_group_args,
                    post_group_func_nb=post_group_func_nb,
                    post_group_args=post_group_args,
                    pre_segment_func_nb=pre_segment_func_nb,
                    pre_segment_args=pre_segment_args,
                    post_segment_func_nb=post_segment_func_nb,
                    post_segment_args=post_segment_args,
                    flex_order_func_nb=order_func_nb,
                    flex_order_args=order_args,
                    post_order_func_nb=post_order_func_nb,
                    post_order_args=post_order_args,
                    close=broadcasted_args['close'],
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    max_orders=max_orders,
                    max_logs=max_logs,
                    flex_2d=close.ndim == 2
                )
            else:
                simulate_func = nb.simulate_nb
                if not use_numba and hasattr(simulate_func, 'py_func'):
                    simulate_func = simulate_func.py_func
                order_records, log_records = simulate_func(
                    target_shape=target_shape_2d,
                    group_lens=group_lens,
                    init_cash=init_cash,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    pre_sim_func_nb=pre_sim_func_nb,
                    pre_sim_args=pre_sim_args,
                    post_sim_func_nb=post_sim_func_nb,
                    post_sim_args=post_sim_args,
                    pre_group_func_nb=pre_group_func_nb,
                    pre_group_args=pre_group_args,
                    post_group_func_nb=post_group_func_nb,
                    post_group_args=post_group_args,
                    pre_segment_func_nb=pre_segment_func_nb,
                    pre_segment_args=pre_segment_args,
                    post_segment_func_nb=post_segment_func_nb,
                    post_segment_args=post_segment_args,
                    order_func_nb=order_func_nb,
                    order_args=order_args,
                    post_order_func_nb=post_order_func_nb,
                    post_order_args=post_order_args,
                    close=broadcasted_args['close'],
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    max_orders=max_orders,
                    max_logs=max_logs,
                    flex_2d=close.ndim == 2
                )

        # Create an instance
        return cls(
            wrapper,
            close,
            order_records,
            log_records,
            init_cash if init_cash_mode is None else init_cash_mode,
            cash_sharing,
            call_seq=call_seq if not flexible and attach_call_seq else None,
            **kwargs
        )

    # ############# Properties ############# #

    @property  # 属性装饰器
    def wrapper(self) -> ArrayWrapper:
        """
        数组包装器
        
        管理数据的索引、形状、分组等操作的核心组件。
        在启用现金共享时，会限制分组的修改权限。
        
        Array wrapper.
        """
        if self.cash_sharing:
            # 启用现金共享时，只允许在需要时禁用分组（但不是全局的，参见regroup）
            return self._wrapper.replace(
                allow_enable=False,  # 不允许启用分组
                allow_modify=False   # 不允许修改分组
            )
        return self._wrapper  # 返回原始包装器

    def regroup(self: PortfolioT, group_by: tp.GroupByLike, **kwargs) -> PortfolioT:
        """
        重新分组此对象
        
        允许用户更改投资组合的分组方式，例如将多个资产合并为一个组进行分析。
        
        Args:
            group_by: 新的分组方式
            **kwargs: 传递给regroup的其他参数
            
        Returns:
            重新分组后的Portfolio实例
            
        Regroup this object.

        参见 `vectorbt.base.array_wrapper.Wrapping.regroup`。

        !!! note
            所有缓存的对象都将丢失。
        """
        if self.cash_sharing:
            if self.wrapper.grouper.is_grouping_modified(group_by=group_by):
                raise ValueError("当 cash_sharing=True 时，无法全局修改分组")
        return Wrapping.regroup(self, group_by, **kwargs)

    @property
    def cash_sharing(self) -> bool:
        """
        是否在同一组内共享现金
        
        当为True时，同一组内的资产共享同一个现金池，
        允许资金在资产间自由流动。
        
        Whether to share cash within the same group.
        """
        return self._cash_sharing

    @property
    def call_seq(self, wrap_kwargs: tp.KwargsLike = None) -> tp.Optional[tp.SeriesFrame]:
        """
        每行每组的调用序列
        
        控制同一时间步内不同资产的订单执行顺序。
        对于现金共享的组合，这个顺序很重要。
        
        Args:
            wrap_kwargs: 包装参数
            
        Returns:
            调用序列的Series/DataFrame，如果未设置则返回None
            
        Sequence of calls per row and group.
        """
        if self._call_seq is None:
            return None
        return self.wrapper.wrap(self._call_seq, group_by=False, **merge_dicts({}, wrap_kwargs))

    @property
    def fillna_close(self) -> bool:
        """
        是否在Portfolio.close中前向后向填充NaN值
        
        用于处理价格数据中的缺失值，确保计算的连续性。
        
        Whether to forward-backward fill NaN values in `Portfolio.close`.
        """
        return self._fillna_close

    @property
    def trades_type(self) -> int:
        """
        在Portfolio中使用的默认交易类型
        
        决定如何解释和分析交易记录（入场交易、出场交易或仓位）。
        
        Default `vectorbt.portfolio.trades.Trades` to use across `Portfolio`.
        """
        return self._trades_type

    # ############# 参考价格 Reference price ############# #

    @property
    def close(self) -> tp.SeriesFrame:
        """
        每单位价格序列（收盘价）
        
        这是投资组合计算的基础价格数据，用于：
        - 计算未实现盈亏
        - 计算投资组合总价值
        - 作为订单执行的参考价格
        
        Price per unit series.
        """
        return self._close

    @cached_method  # 缓存方法装饰器
    def get_filled_close(self, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        对Portfolio.close中的NaN值进行前向-后向填充
        
        处理价格数据中的缺失值，通过前向填充和后向填充确保数据的完整性。
        这对于计算投资组合价值和收益率至关重要。
        
        Args:
            wrap_kwargs: 包装参数
            
        Returns:
            填充后的收盘价数据
            
        Forward-backward-fill NaN values in `Portfolio.close`
        """
        close = to_2d_array(self.close.ffill().bfill())  # 前向和后向填充NaN值
        return self.wrapper.wrap(close, group_by=False, **merge_dicts({}, wrap_kwargs))

    # ############# 记录 Records ############# #

    @property
    def order_records(self) -> tp.RecordArray:
        """
        订单记录的结构化NumPy数组
        
        包含所有已执行订单的详细信息，是投资组合分析的基础数据。
        每条记录包含订单ID、时间戳、大小、价格、费用等信息。
        
        A structured NumPy array of order records.
        """
        return self._order_records

    @cached_property  # 缓存属性装饰器
    def orders(self) -> Orders:
        """
        使用默认参数的Portfolio.get_orders
        
        返回Orders对象，提供对订单记录的高级分析功能。
        
        `Portfolio.get_orders` with default arguments.
        """
        return self.get_orders()

    @cached_method
    def get_orders(self, group_by: tp.GroupByLike = None, **kwargs) -> Orders:
        """
        获取订单记录
        
        创建Orders对象来分析和操作订单数据。Orders对象提供了
        丰富的方法来过滤、聚合和分析订单记录。
        
        Args:
            group_by: 分组方式
            **kwargs: 传递给Orders构造函数的其他参数
            
        Returns:
            Orders对象，包含订单记录和分析方法
        
        Get order records.

        参见 `vectorbt.portfolio.orders.Orders`。
        """
        return Orders(self.wrapper, self.order_records, close=self.close, **kwargs).regroup(group_by)

    @property
    def log_records(self) -> tp.RecordArray:
        """
        日志记录的结构化NumPy数组
        
        包含模拟过程中的详细日志信息，用于调试和深度分析。
        每条记录包含时间戳、现金状态、仓位状态等详细信息。
        
        A structured NumPy array of log records.
        """
        return self._log_records

    @cached_property
    def logs(self) -> Logs:
        """`Portfolio.get_logs` with default arguments."""
        return self.get_logs()

    @cached_method
    def get_logs(self, group_by: tp.GroupByLike = None, **kwargs) -> Logs:
        """Get log records.

        See `vectorbt.portfolio.logs.Logs`."""
        return Logs(self.wrapper, self.log_records, **kwargs).regroup(group_by)

    @cached_property  # 缓存属性装饰器
    def entry_trades(self) -> EntryTrades:
        """
        入场交易记录
        
        使用默认参数的Portfolio.get_entry_trades
        
        返回所有入场交易的详细记录，包括入场时间、价格、数量、盈亏等信息。
        入场交易是指建立新仓位或增加现有仓位的交易。
        
        Returns:
            EntryTrades对象，包含入场交易的分析功能
            
        示例:
            >>> pf.entry_trades.count()  # 入场交易数量
            >>> pf.entry_trades.win_rate()  # 入场交易胜率
            >>> pf.entry_trades.pnl.sum()  # 入场交易总盈亏
            
        `Portfolio.get_entry_trades` with default arguments.
        """
        return self.get_entry_trades()

    @cached_method  # 缓存方法装饰器
    def get_entry_trades(self, group_by: tp.GroupByLike = None, **kwargs) -> EntryTrades:
        """
        获取入场交易记录
        
        从订单记录中提取并分析所有的入场交易，提供详细的交易统计和分析功能。
        
        Args:
            group_by: 分组方式
            **kwargs: 传递给EntryTrades的其他参数
            
        Returns:
            EntryTrades对象，包含丰富的交易分析方法
            
        Get entry trade records.

        参见 `vectorbt.portfolio.trades.EntryTrades`。
        """
        return EntryTrades.from_orders(self.orders, **kwargs).regroup(group_by)

    @cached_property  # 缓存属性装饰器
    def exit_trades(self) -> ExitTrades:
        """
        出场交易记录
        
        使用默认参数的Portfolio.get_exit_trades
        
        返回所有出场交易的详细记录，包括出场时间、价格、数量、盈亏等信息。
        出场交易是指减少仓位或完全平仓的交易。
        
        Returns:
            ExitTrades对象，包含出场交易的分析功能
            
        示例:
            >>> pf.exit_trades.count()  # 出场交易数量  
            >>> pf.exit_trades.win_rate()  # 出场交易胜率
            >>> pf.exit_trades.avg_return()  # 平均收益率
            
        `Portfolio.get_exit_trades` with default arguments.
        """
        return self.get_exit_trades()

    @cached_method  # 缓存方法装饰器
    def get_exit_trades(self, group_by: tp.GroupByLike = None, **kwargs) -> ExitTrades:
        """
        获取出场交易记录
        
        从订单记录中提取并分析所有的出场交易，提供详细的平仓分析功能。
        
        Args:
            group_by: 分组方式
            **kwargs: 传递给ExitTrades的其他参数
            
        Returns:
            ExitTrades对象，包含丰富的平仓分析方法
            
        Get exit trade records.

        参见 `vectorbt.portfolio.trades.ExitTrades`。
        """
        return ExitTrades.from_orders(self.orders, **kwargs).regroup(group_by)

    @cached_property  # 缓存属性装饰器
    def trades(self) -> Trades:
        """
        交易记录（根据trades_type决定类型）
        
        使用默认参数的Portfolio.get_trades
        
        根据Portfolio.trades_type的设置返回相应类型的交易记录。
        这是一个统一的交易访问接口。
        
        `Portfolio.get_trades` with default arguments.
        """
        return self.get_trades()

    @cached_property  # 缓存属性装饰器
    def positions(self) -> Positions:
        """
        仓位记录
        
        使用默认参数的Portfolio.get_positions
        
        返回所有仓位的详细记录，包括持仓期间、最大/最小价值、总盈亏等。
        仓位是从建仓到平仓的完整交易周期。
        
        Returns:
            Positions对象，包含仓位分析功能
            
        示例:
            >>> pf.positions.count()  # 仓位数量
            >>> pf.positions.duration.mean()  # 平均持仓时间
            >>> pf.positions.pnl.sum()  # 总盈亏
            
        `Portfolio.get_positions` with default arguments.
        """
        return self.get_positions()

    @cached_method  # 缓存方法装饰器
    def get_positions(self, group_by: tp.GroupByLike = None, **kwargs) -> Positions:
        """
        获取仓位记录
        
        从出场交易中构建仓位记录，分析完整的持仓周期表现。
        
        Args:
            group_by: 分组方式
            **kwargs: 传递给Positions的其他参数
            
        Returns:
            Positions对象，包含仓位分析方法
            
        Get position records.

        参见 `vectorbt.portfolio.trades.Positions`。
        """
        return Positions.from_trades(self.exit_trades, **kwargs).regroup(group_by)

    @cached_method  # 缓存方法装饰器
    def get_trades(self, group_by: tp.GroupByLike = None, **kwargs) -> Trades:
        """
        根据trades_type获取交易/仓位记录
        
        这是一个统一的接口方法，根据Portfolio.trades_type的设置
        返回相应的交易类型记录。
        
        Args:
            group_by: 分组方式
            **kwargs: 传递给相应交易类的其他参数
            
        Returns:
            根据trades_type返回相应的交易对象
            
        Get trade/position records depending upon `Portfolio.trades_type`.
        """
        if self.trades_type == TradesType.EntryTrades:
            return self.get_entry_trades(group_by=group_by, **kwargs)
        elif self.trades_type == TradesType.ExitTrades:
            return self.get_exit_trades(group_by=group_by, **kwargs)
        return self.get_positions(group_by=group_by, **kwargs)

    @cached_property  # 缓存属性装饰器
    def drawdowns(self) -> Drawdowns:
        """
        回撤记录
        
        使用默认参数的Portfolio.get_drawdowns
        
        返回投资组合的回撤分析，包括最大回撤、回撤持续时间、
        回撤恢复时间等风险指标。
        
        Returns:
            Drawdowns对象，包含回撤分析功能
            
        示例:
            >>> pf.drawdowns.max_drawdown()  # 最大回撤
            >>> pf.drawdowns.avg_duration()  # 平均回撤持续时间
            
        `Portfolio.get_drawdowns` with default arguments.
        """
        return self.get_drawdowns()

    @cached_method  # 缓存方法装饰器
    def get_drawdowns(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None,
                      wrapper_kwargs: tp.KwargsLike = None, **kwargs) -> Drawdowns:
        """
        从投资组合价值获取回撤记录
        
        分析投资组合价值的回撤情况，识别所有回撤事件并计算相关统计指标。
        
        Args:
            group_by: 分组方式
            wrap_kwargs: 包装参数
            wrapper_kwargs: 包装器参数
            **kwargs: 传递给Drawdowns的其他参数
            
        Returns:
            Drawdowns对象，包含详细的回撤分析
            
        Get drawdown records from `Portfolio.value`.

        参见 `vectorbt.generic.drawdowns.Drawdowns`。
        """
        value = self.value(group_by=group_by, wrap_kwargs=wrap_kwargs)
        wrapper_kwargs = merge_dicts(self.orders.wrapper.config, wrapper_kwargs, dict(group_by=None))
        return Drawdowns.from_ts(value, wrapper_kwargs=wrapper_kwargs, **kwargs)

    # ############# Assets ############# #

    @cached_method
    def asset_flow(self, direction: str = 'both', wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Get asset flow series per column.

        Returns the total transacted amount of assets at each time step."""
        direction = map_enum_fields(direction, Direction)
        asset_flow = nb.asset_flow_nb(
            self.wrapper.shape_2d,
            self.orders.values,
            self.orders.col_mapper.col_map,
            direction
        )
        return self.wrapper.wrap(asset_flow, group_by=False, **merge_dicts({}, wrap_kwargs))

    @cached_method  # 缓存方法装饰器
    def assets(self, direction: str = 'both', wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        获取每列的资产序列
        
        返回每个时间步的当前仓位大小。正值表示多头仓位，负值表示空头仓位。
        
        Args:
            direction: 方向筛选
                - 'both': 所有方向（默认）
                - 'longonly': 仅多头仓位
                - 'shortonly': 仅空头仓位  
            wrap_kwargs: 包装参数
            
        Returns:
            每个时间步的资产持有量序列
            
        示例:
            >>> pf.assets()  # 获取所有资产持有量
            >>> pf.assets(direction='longonly')  # 仅获取多头仓位
        
        Get asset series per column.

        Returns the current position at each time step.
        """
        direction = map_enum_fields(direction, Direction)  # 将字符串转换为枚举
        asset_flow = to_2d_array(self.asset_flow(direction='both'))  # 获取资产流动
        assets = nb.assets_nb(asset_flow)  # 计算累计资产
        if direction == Direction.LongOnly:
            assets = np.where(assets > 0, assets, 0.)  # 仅保留正值（多头）
        if direction == Direction.ShortOnly:
            assets = np.where(assets < 0, -assets, 0.)  # 仅保留负值转正（空头）
        return self.wrapper.wrap(assets, group_by=False, **merge_dicts({}, wrap_kwargs))

    @cached_method
    def position_mask(self, direction: str = 'both', group_by: tp.GroupByLike = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        获取每列/组的仓位掩码
        
        如果资产在该时点有仓位（无论多头还是空头），元素为True，否则为False。
        
        Args:
            direction: 方向筛选（'both', 'longonly', 'shortonly'）
            group_by: 分组方式
            wrap_kwargs: 包装参数
            
        Returns:
            仓位掩码的布尔数组
            
        Get position mask per column/group.

        An element is True if the asset is in the market at this tick.
        """
        direction = map_enum_fields(direction, Direction)
        assets = to_2d_array(self.assets(direction=direction))
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            # 如果有分组，使用分组逻辑
            position_mask = to_2d_array(self.position_mask(direction=direction, group_by=False))
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            position_mask = nb.position_mask_grouped_nb(position_mask, group_lens)
        else:
            position_mask = assets != 0  # 非零仓位为True
        return self.wrapper.wrap(position_mask, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    @cached_method
    def position_coverage(self, direction: str = 'both', group_by: tp.GroupByLike = None,
                          wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        获取每列/组的仓位覆盖率
        
        计算有仓位的时间占总时间的比例，用于衡量策略的活跃程度。
        
        Args:
            direction: 方向筛选（'both', 'longonly', 'shortonly'）
            group_by: 分组方式
            wrap_kwargs: 包装参数
            
        Returns:
            仓位覆盖率（0-1之间的值）
            
        Get position coverage per column/group.
        """
        direction = map_enum_fields(direction, Direction)
        assets = to_2d_array(self.assets(direction=direction))
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            # 分组情况下的覆盖率计算
            position_mask = to_2d_array(self.position_mask(direction=direction, group_by=False))
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            position_coverage = nb.position_coverage_grouped_nb(position_mask, group_lens)
        else:
            position_coverage = np.mean(assets != 0, axis=0)  # 非零仓位的平均比例
        wrap_kwargs = merge_dicts(dict(name_or_index='position_coverage'), wrap_kwargs)
        return self.wrapper.wrap_reduced(position_coverage, group_by=group_by, **wrap_kwargs)

    # ############# 现金 Cash ############# #

    @cached_method
    def cash_flow(self, group_by: tp.GroupByLike = None, free: bool = False,
                  wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        获取每列/组的现金流序列
        
        显示每个时间步的现金变化量，正值表示现金流入，负值表示现金流出。
        
        Args:
            group_by: 分组方式
            free: 是否返回自由现金流
                - True: 返回自由现金流（永远不超过初始水平，因为操作总是花钱）
                - False: 返回总现金流（默认）
            wrap_kwargs: 包装参数
            
        Returns:
            现金流序列
            
        Get cash flow series per column/group.

        Use `free` to return the flow of the free cash, which never goes above the initial level,
        because an operation always costs money.
        """
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            cash_flow = to_2d_array(self.cash_flow(group_by=False, free=free))
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            cash_flow = nb.cash_flow_grouped_nb(cash_flow, group_lens)
        else:
            cash_flow = nb.cash_flow_nb(
                self.wrapper.shape_2d,
                self.orders.values,
                self.orders.col_mapper.col_map,
                free
            )
        return self.wrapper.wrap(cash_flow, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    @cached_property  # 缓存属性装饰器
    def init_cash(self) -> tp.MaybeSeries:
        """
        使用默认参数的Portfolio.get_init_cash
        
        返回初始现金金额，这是投资组合计算的基础。
        
        `Portfolio.get_init_cash` with default arguments.
        """
        return self.get_init_cash()

    @cached_method
    def get_init_cash(self, group_by: tp.GroupByLike = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        获取每列/组的初始现金金额
        
        计算投资组合开始时的现金金额。如果使用自动模式，
        会根据最大现金需求来确定合适的初始现金。
        
        Args:
            group_by: 分组方式
            wrap_kwargs: 包装参数
            
        Returns:
            初始现金金额（Series或标量值）
            
        Initial amount of cash per column/group with default arguments.

        !!! note
            如果初始现金余额是自动发现的，并且在整个模拟过程中没有使用自有现金
            （例如，做空时），它将被设置为1而不是0，以便顺利计算收益。
        """
        if isinstance(self._init_cash, int):
            cash_flow = to_2d_array(self.cash_flow(group_by=group_by))
            cash_min = np.min(np.cumsum(cash_flow, axis=0), axis=0)
            init_cash = np.where(cash_min < 0, np.abs(cash_min), 1.)
            if self._init_cash == InitCashMode.AutoAlign:
                init_cash = np.full(init_cash.shape, np.max(init_cash))
        else:
            init_cash = to_1d_array(self._init_cash)
            if self.wrapper.grouper.is_grouped(group_by=group_by):
                group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
                init_cash = nb.init_cash_grouped_nb(init_cash, group_lens, self.cash_sharing)
            else:
                group_lens = self.wrapper.grouper.get_group_lens()
                init_cash = nb.init_cash_nb(init_cash, group_lens, self.cash_sharing)
        wrap_kwargs = merge_dicts(dict(name_or_index='init_cash'), wrap_kwargs)
        return self.wrapper.wrap_reduced(init_cash, group_by=group_by, **wrap_kwargs)

    @cached_method
    def cash(self, group_by: tp.GroupByLike = None, in_sim_order: bool = False, free: bool = False,
             wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        获取每列/组的现金余额序列
        
        计算每个时间步的现金余额，这是投资组合价值计算的重要组成部分。
        
        Args:
            group_by: 分组方式
            in_sim_order: 是否按模拟顺序返回现金
                - True: 按订单执行的实际顺序返回现金（需要启用现金共享）
                - False: 按时间顺序返回现金（默认）
            free: 是否返回自由现金余额
                - True: 返回自由现金（考虑保证金等约束）
                - False: 返回总现金余额（默认）
            wrap_kwargs: 包装参数
            
        Returns:
            现金余额序列
            
        示例:
            >>> pf.cash()  # 获取总现金余额
            >>> pf.cash(free=True)  # 获取自由现金余额
            
        Get cash balance series per column/group.

        参见 `Portfolio.value` 中对 `in_sim_order` 的解释。
        关于 `free` 参数，参见 `Portfolio.cash_flow`。
        """
        # 参数验证：按模拟顺序时必须启用现金共享
        if in_sim_order and not self.cash_sharing:
            raise ValueError("当 in_sim_order=True 时，必须启用现金共享")

        # 获取现金流数据
        cash_flow = to_2d_array(self.cash_flow(group_by=group_by, free=free))
        
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            # 分组情况：计算组级别的现金余额
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            init_cash = to_1d_array(self.get_init_cash(group_by=group_by))
            cash = nb.cash_grouped_nb(
                self.wrapper.shape_2d,  # 数组形状
                cash_flow,  # 现金流数据
                group_lens,  # 组长度
                init_cash   # 初始现金
            )
        else:
            # 非分组情况
            if self.wrapper.grouper.is_grouping_disabled(group_by=group_by) and in_sim_order:
                # 按模拟顺序计算现金余额（考虑调用序列）
                if self.call_seq is None:
                    raise ValueError("没有附加调用序列。请在类方法中传递 `attach_call_seq=True`"
                                     "（不支持灵活模拟）")
                group_lens = self.wrapper.grouper.get_group_lens()
                init_cash = to_1d_array(self.init_cash)
                call_seq = to_2d_array(self.call_seq)
                cash = nb.cash_in_sim_order_nb(cash_flow, group_lens, init_cash, call_seq)
            else:
                # 标准现金余额计算
                init_cash = to_1d_array(self.get_init_cash(group_by=False))
                cash = nb.cash_nb(cash_flow, init_cash)
        return self.wrapper.wrap(cash, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    # ############# 投资组合性能 Performance ############# #

    @cached_method
    def asset_value(self, direction: str = 'both', group_by: tp.GroupByLike = None,
                    wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        获取每列/组的资产价值序列
        
        计算持有资产的市场价值，即仓位数量乘以当前价格。
        这是投资组合总价值的重要组成部分。
        
        Args:
            direction: 方向筛选
                - 'both': 所有方向的资产价值（默认）
                - 'longonly': 仅多头仓位的价值
                - 'shortonly': 仅空头仓位的价值
            group_by: 分组方式
            wrap_kwargs: 包装参数
            
        Returns:
            资产价值序列
            
        示例:
            >>> pf.asset_value()  # 获取总资产价值
            >>> pf.asset_value(direction='longonly')  # 获取多头资产价值
            
        Get asset value series per column/group.
        """
        direction = map_enum_fields(direction, Direction)
        
        # 获取价格数据，如果启用则填充缺失值
        if self.fillna_close:
            close = to_2d_array(self.get_filled_close()).copy()
        else:
            close = to_2d_array(self.close).copy()
            
        # 获取资产持有量
        assets = to_2d_array(self.assets(direction=direction))
        close[assets == 0] = 0.  # 当资产为0时，将价格设为0（处理NaN价格）
        
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            # 分组情况：先计算列级别，再聚合到组级别
            asset_value = to_2d_array(self.asset_value(direction=direction, group_by=False))
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            asset_value = nb.asset_value_grouped_nb(asset_value, group_lens)
        else:
            # 非分组情况：直接计算资产价值（价格 × 持有量）
            asset_value = nb.asset_value_nb(close, assets)
        return self.wrapper.wrap(asset_value, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    @cached_method
    def gross_exposure(self, direction: str = 'both', group_by: tp.GroupByLike = None,
                       wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        获取总敞口
        
        计算投资组合的总敞口，即资产价值相对于总资本（资产价值+现金）的比例。
        总敞口反映了投资组合的杠杆程度。
        
        Args:
            direction: 方向筛选（'both', 'longonly', 'shortonly'）
            group_by: 分组方式
            wrap_kwargs: 包装参数
            
        Returns:
            总敞口比例序列（通常在0-1之间，>1表示使用杠杆）
            
        Get gross exposure.
        """
        asset_value = to_2d_array(self.asset_value(group_by=group_by, direction=direction))
        cash = to_2d_array(self.cash(group_by=group_by, free=True))  # 使用自由现金
        gross_exposure = nb.gross_exposure_nb(asset_value, cash)
        return self.wrapper.wrap(gross_exposure, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    @cached_method
    def net_exposure(self, group_by: tp.GroupByLike = None,
                     wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        获取净敞口
        
        计算投资组合的净敞口，即多头敞口减去空头敞口。
        净敞口反映了投资组合的方向性偏向。
        
        Args:
            group_by: 分组方式
            wrap_kwargs: 包装参数
            
        Returns:
            净敞口比例序列
            - 正值：偏向多头
            - 负值：偏向空头  
            - 0：市场中性
            
        Get net exposure.
        """
        long_exposure = to_2d_array(self.gross_exposure(direction='longonly', group_by=group_by))
        short_exposure = to_2d_array(self.gross_exposure(direction='shortonly', group_by=group_by))
        net_exposure = long_exposure - short_exposure  # 净敞口 = 多头敞口 - 空头敞口
        return self.wrapper.wrap(net_exposure, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    @cached_method
    def value(self, group_by: tp.GroupByLike = None, in_sim_order: bool = False,
              wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        获取每列/组的投资组合价值序列
        
        这是投资组合分析的核心方法，计算投资组合的总价值（现金 + 资产价值）。
        
        Args:
            group_by: 分组方式
            in_sim_order: 是否按模拟顺序返回价值
                - False: 按时间顺序生成独立的投资组合价值（默认）
                  对每个资产基于现金流生成价值，因此独立于其他资产
                - True: 按模拟顺序返回价值（行主序），不能直接用于生成收益率
                  有助于分析价值在模拟过程中的演变
            wrap_kwargs: 包装参数
            
        Returns:
            投资组合价值序列
            
        示例:
            >>> pf.value()  # 获取投资组合总价值
            >>> pf.value(in_sim_order=True)  # 获取按模拟顺序的价值
            
        Get portfolio value series per column/group.

        默认情况下，将基于现金流为每个资产生成投资组合价值，因此独立于其他资产，
        初始现金余额和头寸为整个组的。用于生成收益率和比较同一组内的资产。

        当 `group_by` 为 False 且 `in_sim_order` 为 True 时，
        返回按模拟顺序生成的价值（参见行主序）。
        此价值不能直接用于生成收益率。有助于分析价值在模拟过程中的演变。
        """
        # 获取现金余额和资产价值
        cash = to_2d_array(self.cash(group_by=group_by, in_sim_order=in_sim_order))
        asset_value = to_2d_array(self.asset_value(group_by=group_by))
        
        if self.wrapper.grouper.is_grouping_disabled(group_by=group_by) and in_sim_order:
            # 按模拟顺序计算价值（考虑调用序列）
            if self.call_seq is None:
                raise ValueError("没有附加调用序列。请在类方法中传递 `attach_call_seq=True`"
                                 "（不支持灵活模拟）")
            group_lens = self.wrapper.grouper.get_group_lens()
            call_seq = to_2d_array(self.call_seq)
            value = nb.value_in_sim_order_nb(cash, asset_value, group_lens, call_seq)
            # 价格为NaN的情况已经在ungrouped_value_nb中处理
        else:
            # 标准价值计算：现金 + 资产价值
            value = nb.value_nb(cash, asset_value)
        return self.wrapper.wrap(value, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    @cached_method
    def total_profit(self, group_by: tp.GroupByLike = None,
                     wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        获取每列/组的总利润
        
        计算投资组合的累计利润，即最终价值减去初始现金。
        这个方法直接从订单记录计算，速度很快。
        
        Args:
            group_by: 分组方式
            wrap_kwargs: 包装参数
            
        Returns:
            总利润（Series或标量值）
            - 正值：获得利润
            - 负值：发生亏损
            - 0：盈亏平衡
            
        示例:
            >>> pf.total_profit()  # 获取总利润
            
        Get total profit per column/group.

        直接从订单记录计算（速度快）。
        """
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            # 分组情况：先计算每列的利润，然后聚合到组级别
            total_profit = to_1d_array(self.total_profit(group_by=False))
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            total_profit = nb.total_profit_grouped_nb(
                total_profit,  # 每列的总利润
                group_lens     # 组长度
            )
        else:
            # 非分组情况：直接从订单记录计算总利润
            if self.fillna_close:
                close = to_2d_array(self.get_filled_close())  # 使用填充后的收盘价
            else:
                close = to_2d_array(self.close)  # 使用原始收盘价
            total_profit = nb.total_profit_nb(
                self.wrapper.shape_2d,        # 数据形状
                close,                        # 收盘价数据
                self.orders.values,          # 订单记录
                self.orders.col_mapper.col_map  # 列映射
            )
        wrap_kwargs = merge_dicts(dict(name_or_index='total_profit'), wrap_kwargs)
        return self.wrapper.wrap_reduced(total_profit, group_by=group_by, **wrap_kwargs)

    @cached_method
    def final_value(self, group_by: tp.GroupByLike = None,
                    wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        获取每列/组的最终价值
        
        计算投资组合在期末的总价值，即初始现金加上总利润。
        
        Args:
            group_by: 分组方式
            wrap_kwargs: 包装参数
            
        Returns:
            最终价值（Series或标量值）
            
        示例:
            >>> pf.final_value()  # 获取最终价值
            
        Get final value per column/group.
        """
        init_cash = to_1d_array(self.get_init_cash(group_by=group_by))  # 获取初始现金
        total_profit = to_1d_array(self.total_profit(group_by=group_by))  # 获取总利润
        final_value = nb.final_value_nb(total_profit, init_cash)  # 最终价值 = 初始现金 + 总利润
        wrap_kwargs = merge_dicts(dict(name_or_index='final_value'), wrap_kwargs)
        return self.wrapper.wrap_reduced(final_value, group_by=group_by, **wrap_kwargs)

    @cached_method
    def total_return(self, group_by: tp.GroupByLike = None,
                     wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        获取每列/组的总收益率
        
        计算投资组合的总收益率，即总利润除以初始现金。
        这是衡量投资组合整体表现的重要指标。
        
        Args:
            group_by: 分组方式
            wrap_kwargs: 包装参数
            
        Returns:
            总收益率（Series或标量值）
            - 正值：获得收益，如0.15表示15%收益
            - 负值：发生亏损，如-0.10表示10%亏损
            - 0：盈亏平衡
            
        示例:
            >>> pf.total_return()  # 获取总收益率
            
        Get total return per column/group.
        """
        init_cash = to_1d_array(self.get_init_cash(group_by=group_by))  # 获取初始现金
        total_profit = to_1d_array(self.total_profit(group_by=group_by))  # 获取总利润
        total_return = nb.total_return_nb(total_profit, init_cash)  # 总收益率 = 总利润 / 初始现金
        wrap_kwargs = merge_dicts(dict(name_or_index='total_return'), wrap_kwargs)
        return self.wrapper.wrap_reduced(total_return, group_by=group_by, **wrap_kwargs)

    @cached_method
    def returns(self, group_by: tp.GroupByLike = None, in_sim_order=False,
                wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        获取基于投资组合价值的每列/组收益率序列
        
        计算每个时间步的收益率，即价值的百分比变化。
        这是进行风险分析和绩效评估的基础数据。
        
        Args:
            group_by: 分组方式
            in_sim_order: 是否按模拟顺序计算收益率
                - False: 按时间顺序计算（默认）
                - True: 按模拟顺序计算（需要启用现金共享）
            wrap_kwargs: 包装参数
            
        Returns:
            收益率序列
            - 正值：该时点获得收益
            - 负值：该时点发生亏损
            - 0：该时点无变化
            
        示例:
            >>> pf.returns()  # 获取收益率序列
            >>> pf.returns().mean()  # 获取平均收益率
            >>> pf.returns().std()   # 获取收益率波动率
            
        Get return series per column/group based on portfolio value.
        """
        value = to_2d_array(self.value(group_by=group_by, in_sim_order=in_sim_order))
        
        if self.wrapper.grouper.is_grouping_disabled(group_by=group_by) and in_sim_order:
            # 按模拟顺序计算收益率
            if self.call_seq is None:
                raise ValueError("没有附加调用序列。请在类方法中传递 `attach_call_seq=True`"
                                 "（不支持灵活模拟）")
            group_lens = self.wrapper.grouper.get_group_lens()
            init_cash_grouped = to_1d_array(self.init_cash)
            call_seq = to_2d_array(self.call_seq)
            returns = nb.returns_in_sim_order_nb(value, group_lens, init_cash_grouped, call_seq)
        else:
            # 标准收益率计算：(当前价值 - 前一价值) / 前一价值
            init_cash = to_1d_array(self.get_init_cash(group_by=group_by))
            returns = returns_nb.returns_nb(value, init_cash)
        return self.wrapper.wrap(returns, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    @cached_method
    def asset_returns(self, group_by: tp.GroupByLike = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        获取每列/组的资产收益率序列
        
        这种收益率类型仅基于现金流和资产价值，而不是投资组合价值。
        它忽略被动现金，因此无论当前可用现金数量多少都会返回相同的数字，即使是无穷大。
        收益率的规模相当于全仓投入并保持可用现金为零。
        
        Args:
            group_by: 分组方式
            wrap_kwargs: 包装参数
            
        Returns:
            资产收益率序列
            - 基于实际交易的现金流计算
            - 不受闲置现金影响
            - 反映资产本身的表现
            
        示例:
            >>> pf.asset_returns()  # 获取资产收益率
            
        Get asset return series per column/group.

        这种收益率类型仅基于现金流和资产价值而不是投资组合价值。
        它忽略被动现金，因此无论当前可用现金数量多少都会返回相同的数字，即使是 `np.inf`。
        收益率的规模相当于全仓投入并保持可用现金为零。
        """
        cash_flow = to_2d_array(self.cash_flow(group_by=group_by))  # 获取现金流
        asset_value = to_2d_array(self.asset_value(group_by=group_by))  # 获取资产价值
        asset_returns = nb.asset_returns_nb(cash_flow, asset_value)  # 计算资产收益率
        return self.wrapper.wrap(asset_returns, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    @property
    def returns_acc(self) -> ReturnsAccessor:
        """
        使用默认参数的Portfolio.get_returns_acc
        
        返回收益率访问器，提供丰富的收益率分析方法。
        
        `Portfolio.get_returns_acc` with default arguments.
        """
        return self.get_returns_acc()

    @cached_method
    def get_returns_acc(self,
                        group_by: tp.GroupByLike = None,
                        benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                        freq: tp.Optional[tp.FrequencyLike] = None,
                        year_freq: tp.Optional[tp.FrequencyLike] = None,
                        use_asset_returns: bool = False,
                        defaults: tp.KwargsLike = None,
                        **kwargs) -> ReturnsAccessor:
        """
        获取收益率访问器
        
        创建收益率访问器对象，提供专业的收益率分析功能，如夏普比率、
        最大回撤、波动率等风险和收益指标。
        
        Args:
            group_by: 分组方式
            benchmark_rets: 基准收益率（用于比较分析）
            freq: 数据频率
            year_freq: 年化频率
            use_asset_returns: 是否使用资产收益率（而非投资组合收益率）
            defaults: 默认参数
            **kwargs: 其他参数
            
        Returns:
            ReturnsAccessor对象，包含丰富的收益率分析方法
            
        示例:
            >>> returns_acc = pf.get_returns_acc()
            >>> returns_acc.sharpe_ratio()  # 夏普比率
            >>> returns_acc.max_drawdown()  # 最大回撤
            >>> returns_acc.volatility()    # 波动率
            
        Get returns accessor of type `vectorbt.returns.accessors.ReturnsAccessor`.

        !!! hint
            您可以在此投资组合的（可缓存）属性中找到此访问器的大部分方法。
        """
        if freq is None:
            freq = self.wrapper.freq  # 使用包装器的默认频率
            
        # 根据参数选择收益率类型
        if use_asset_returns:
            returns = self.asset_returns(group_by=group_by)  # 使用资产收益率
        else:
            returns = self.returns(group_by=group_by)  # 使用投资组合收益率
            
        # 设置基准收益率
        if benchmark_rets is None:
            benchmark_rets = self.benchmark_returns(group_by=group_by)
            
        # 创建并返回收益率访问器
        return returns.vbt.returns(
            benchmark_rets=benchmark_rets,  # 基准收益率
            freq=freq,                      # 数据频率
            year_freq=year_freq,           # 年化频率
            defaults=defaults,             # 默认参数
            **kwargs                       # 其他参数
        )

    @cached_property
    def qs(self) -> QSAdapterT:
        """
        使用默认参数的Portfolio.get_qs
        
        返回quantstats适配器，提供专业的量化投资分析功能。
        quantstats是一个流行的Python量化分析库。
        
        `Portfolio.get_qs` with default arguments.
        """
        return self.get_qs()

    @cached_method
    def get_qs(self,
               group_by: tp.GroupByLike = None,
               benchmark_rets: tp.Optional[tp.ArrayLike] = None,
               freq: tp.Optional[tp.FrequencyLike] = None,
               year_freq: tp.Optional[tp.FrequencyLike] = None,
               use_asset_returns: bool = False,
               **kwargs) -> QSAdapterT:
        """
        获取quantstats适配器
        
        创建quantstats适配器对象，提供专业的量化投资分析功能，
        包括详细的绩效报告、风险指标、回撤分析等。
        
        Args:
            group_by: 分组方式
            benchmark_rets: 基准收益率
            freq: 数据频率
            year_freq: 年化频率
            use_asset_returns: 是否使用资产收益率
            **kwargs: 传递给适配器构造函数的其他参数
            
        Returns:
            QSAdapter对象，包含quantstats的所有分析功能
            
        示例:
            >>> qs = pf.get_qs()
            >>> qs.plot_snapshot()    # 生成绩效快照图
            >>> qs.stats()           # 获取详细统计信息
            >>> qs.plot_returns()    # 绘制收益率图表
            
        Get quantstats adapter of type `vectorbt.returns.qs_adapter.QSAdapter`.

        `**kwargs` 传递给适配器构造函数。
        """
        from vectorbt.returns.qs_adapter import QSAdapter

        # 获取收益率访问器
        returns_acc = self.get_returns_acc(
            group_by=group_by,
            benchmark_rets=benchmark_rets,
            freq=freq,
            year_freq=year_freq,
            use_asset_returns=use_asset_returns
        )
        return QSAdapter(returns_acc, **kwargs)

    @cached_method
    def benchmark_value(self, group_by: tp.GroupByLike = None,
                        wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        获取每列/组的市场基准价值序列
        
        计算基于买入并持有策略的基准投资组合价值，用于与实际投资组合进行比较。
        
        Args:
            group_by: 分组方式
            wrap_kwargs: 包装参数
            
        Returns:
            基准价值序列
            - 基于买入并持有策略
            - 用于投资组合绩效比较
            
        示例:
            >>> pf.benchmark_value()  # 获取基准价值
            
        Get market benchmark value series per column/group.

        如果分组，会在组内资产之间平均分配初始现金。

        !!! note
            不考虑费用和滑点。要考虑这些因素，请创建单独的投资组合。
        """
        # 获取价格数据
        if self.fillna_close:
            close = to_2d_array(self.get_filled_close())  # 使用填充后的收盘价
        else:
            close = to_2d_array(self.close)  # 使用原始收盘价
            
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            # 分组情况：在组内平均分配现金
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            init_cash_grouped = to_1d_array(self.get_init_cash(group_by=group_by))
            benchmark_value = nb.benchmark_value_grouped_nb(close, group_lens, init_cash_grouped)
        else:
            # 非分组情况：每列使用各自的初始现金
            init_cash = to_1d_array(self.get_init_cash(group_by=False))
            benchmark_value = nb.benchmark_value_nb(close, init_cash)
        return self.wrapper.wrap(benchmark_value, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    @cached_method
    def benchmark_returns(self, group_by: tp.GroupByLike = None,
                          wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        获取基于基准价值的每列/组收益率序列
        
        计算基于买入并持有策略的基准收益率，用于评估投资组合的相对表现。
        
        Args:
            group_by: 分组方式
            wrap_kwargs: 包装参数
            
        Returns:
            基准收益率序列
            - 基于买入并持有策略
            - 用于计算阿尔法、贝塔等指标
            
        示例:
            >>> pf.benchmark_returns()  # 获取基准收益率
            
        Get return series per column/group based on benchmark value.
        """
        benchmark_value = to_2d_array(self.benchmark_value(group_by=group_by))  # 获取基准价值
        init_cash = to_1d_array(self.get_init_cash(group_by=group_by))  # 获取初始现金
        benchmark_returns = returns_nb.returns_nb(benchmark_value, init_cash)  # 计算基准收益率
        return self.wrapper.wrap(benchmark_returns, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    benchmark_rets = benchmark_returns  # benchmark_returns的别名

    @cached_method
    def total_benchmark_return(self, group_by: tp.GroupByLike = None,
                               wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        获取总基准收益率
        
        计算基准投资组合的总收益率，用于与实际投资组合的总收益率进行比较。
        
        Args:
            group_by: 分组方式
            wrap_kwargs: 包装参数
            
        Returns:
            总基准收益率（Series或标量值）
            
        示例:
            >>> pf.total_benchmark_return()  # 获取总基准收益率
            
        Get total benchmark return.
        """
        benchmark_value = to_2d_array(self.benchmark_value(group_by=group_by))  # 获取基准价值
        total_benchmark_return = nb.total_benchmark_return_nb(benchmark_value)  # 计算总基准收益率
        wrap_kwargs = merge_dicts(dict(name_or_index='total_benchmark_return'), wrap_kwargs)
        return self.wrapper.wrap_reduced(total_benchmark_return, group_by=group_by, **wrap_kwargs)

    # ############# 属性解析 Resolution ############# #

    @property
    def self_aliases(self) -> tp.Set[str]:
        """
        与此对象关联的名称
        
        定义可以用来引用此Portfolio对象的别名集合。
        
        Names to associate with this object.
        """
        return {'self', 'portfolio', 'pf'}

    def pre_resolve_attr(self, attr: str, final_kwargs: tp.KwargsLike = None) -> str:
        """
        在解析前预处理属性
        
        在属性解析之前对属性名进行预处理，支持属性别名和动态解析。
        
        Args:
            attr: 属性名
            final_kwargs: 最终关键字参数
            
        Returns:
            处理后的属性名
            
        Pre-process an attribute before resolution.

        Uses the following keys:

        * `use_asset_returns`: Whether to use `Portfolio.asset_returns` when resolving `returns` argument.
        * `trades_type`: Which trade type to use when resolving `trades` argument."""
        if 'use_asset_returns' in final_kwargs:
            if attr == 'returns' and final_kwargs['use_asset_returns']:
                attr = 'asset_returns'
        if 'trades_type' in final_kwargs:
            trades_type = final_kwargs['trades_type']
            if isinstance(final_kwargs['trades_type'], str):
                trades_type = map_enum_fields(trades_type, TradesType)
            if attr == 'trades' and trades_type != self.trades_type:
                if trades_type == TradesType.EntryTrades:
                    attr = 'entry_trades'
                elif trades_type == TradesType.ExitTrades:
                    attr = 'exit_trades'
                else:
                    attr = 'positions'
        return attr

    def post_resolve_attr(self, attr: str, out: tp.Any, final_kwargs: tp.KwargsLike = None) -> str:
        """Post-process an object after resolution.

        Uses the following keys:

        * `incl_open`: Whether to include open trades/positions when resolving an argument
            that is an instance of `vectorbt.portfolio.trades.Trades`."""
        if 'incl_open' in final_kwargs:
            if isinstance(out, Trades) and not final_kwargs['incl_open']:
                out = out.closed
        return out

    # ############# 统计指标 Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """
        Portfolio.stats的默认参数
        
        合并统计构建器混入类和设置文件中的默认统计参数，
        为投资组合统计分析提供全面的默认配置。
        
        Returns:
            统计分析的默认参数字典
            
        Defaults for `Portfolio.stats`.

        合并 `vectorbt.generic.stats_builder.StatsBuilderMixin.stats_defaults` 和
        `vectorbt._settings.settings` 中的 `portfolio.stats`。
        """
        from vectorbt._settings import settings
        returns_cfg = settings['returns']  # 收益率配置
        portfolio_stats_cfg = settings['portfolio']['stats']  # 投资组合统计配置

        return merge_dicts(
            StatsBuilderMixin.stats_defaults.__get__(self),  # 统计构建器的默认参数
            dict(
                settings=dict(
                    year_freq=returns_cfg['year_freq'],  # 年化频率
                    trades_type=self.trades_type         # 交易类型
                )
            ),
            portfolio_stats_cfg  # 投资组合统计配置
        )

    # 预定义的投资组合性能指标配置
    # 这些指标用于生成详细的投资组合统计报告
    _metrics: tp.ClassVar[Config] = Config(
        dict(
            # 基本时间信息
            start=dict(
                title='Start',              # 指标标题：开始时间
                calc_func=lambda self: self.wrapper.index[0],  # 计算函数：获取第一个时间点
                agg_func=None,              # 聚合函数：无需聚合
                tags='wrapper'              # 标签：属于包装器类型
            ),
            end=dict(
                title='End',                # 指标标题：结束时间
                calc_func=lambda self: self.wrapper.index[-1], # 计算函数：获取最后一个时间点
                agg_func=None,              # 聚合函数：无需聚合
                tags='wrapper'              # 标签：属于包装器类型
            ),
            period=dict(
                title='Period',             # 指标标题：投资期间
                calc_func=lambda self: len(self.wrapper.index), # 计算函数：获取时间长度
                apply_to_timedelta=True,    # 应用到时间差
                agg_func=None,              # 聚合函数：无需聚合
                tags='wrapper'              # 标签：属于包装器类型
            ),
            
            # 基本价值指标
            start_value=dict(
                title='Start Value',        # 指标标题：起始价值
                calc_func='get_init_cash',  # 计算函数：获取初始现金
                tags='portfolio'            # 标签：属于投资组合类型
            ),
            end_value=dict(
                title='End Value',          # 指标标题：最终价值
                calc_func='final_value',    # 计算函数：获取最终价值
                tags='portfolio'            # 标签：属于投资组合类型
            ),
            total_return=dict(
                title='Total Return [%]',   # 指标标题：总收益率(百分比)
                calc_func='total_return',   # 计算函数：获取总收益率
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理：转换为百分比
                tags='portfolio'            # 标签：属于投资组合类型
            ),
            benchmark_return=dict(
                title='Benchmark Return [%]',
                calc_func='benchmark_rets.vbt.returns.total',
                post_calc_func=lambda self, out, settings: out * 100,
                tags='portfolio'
            ),
            max_gross_exposure=dict(
                title='Max Gross Exposure [%]',
                calc_func='gross_exposure.vbt.max',
                post_calc_func=lambda self, out, settings: out * 100,
                tags='portfolio'
            ),
            total_fees_paid=dict(
                title='Total Fees Paid',
                calc_func='orders.fees.sum',
                tags=['portfolio', 'orders']
            ),
            max_dd=dict(
                title='Max Drawdown [%]',
                calc_func='drawdowns.max_drawdown',
                post_calc_func=lambda self, out, settings: -out * 100,
                tags=['portfolio', 'drawdowns']
            ),
            max_dd_duration=dict(
                title='Max Drawdown Duration',
                calc_func='drawdowns.max_duration',
                fill_wrap_kwargs=True,
                tags=['portfolio', 'drawdowns', 'duration']
            ),
            total_trades=dict(
                title='Total Trades',
                calc_func='trades.count',
                incl_open=True,
                tags=['portfolio', 'trades']
            ),
            total_closed_trades=dict(
                title='Total Closed Trades',
                calc_func='trades.closed.count',
                tags=['portfolio', 'trades', 'closed']
            ),
            total_open_trades=dict(
                title='Total Open Trades',
                calc_func='trades.open.count',
                incl_open=True,
                tags=['portfolio', 'trades', 'open']
            ),
            open_trade_pnl=dict(
                title='Open Trade PnL',
                calc_func='trades.open.pnl.sum',
                incl_open=True,
                tags=['portfolio', 'trades', 'open']
            ),
            win_rate=dict(
                title='Win Rate [%]',
                calc_func='trades.win_rate',
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]")
            ),
            best_trade=dict(
                title='Best Trade [%]',
                calc_func='trades.returns.max',
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]")
            ),
            worst_trade=dict(
                title='Worst Trade [%]',
                calc_func='trades.returns.min',
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]")
            ),
            avg_winning_trade=dict(
                title='Avg Winning Trade [%]',
                calc_func='trades.winning.returns.mean',
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'winning']")
            ),
            avg_losing_trade=dict(
                title='Avg Losing Trade [%]',
                calc_func='trades.losing.returns.mean',
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'losing']")
            ),
            avg_winning_trade_duration=dict(
                title='Avg Winning Trade Duration',
                calc_func='trades.winning.duration.mean',
                apply_to_timedelta=True,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'winning', 'duration']")
            ),
            avg_losing_trade_duration=dict(
                title='Avg Losing Trade Duration',
                calc_func='trades.losing.duration.mean',
                apply_to_timedelta=True,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'losing', 'duration']")
            ),
            profit_factor=dict(
                title='Profit Factor',
                calc_func='trades.profit_factor',
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]")
            ),
            expectancy=dict(
                title='Expectancy',
                calc_func='trades.expectancy',
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]")
            ),
            sharpe_ratio=dict(
                title='Sharpe Ratio',
                calc_func='returns_acc.sharpe_ratio',
                check_has_freq=True,
                check_has_year_freq=True,
                tags=['portfolio', 'returns']
            ),
            calmar_ratio=dict(
                title='Calmar Ratio',
                calc_func='returns_acc.calmar_ratio',
                check_has_freq=True,
                check_has_year_freq=True,
                tags=['portfolio', 'returns']
            ),
            omega_ratio=dict(
                title='Omega Ratio',
                calc_func='returns_acc.omega_ratio',
                check_has_freq=True,
                check_has_year_freq=True,
                tags=['portfolio', 'returns']
            ),
            sortino_ratio=dict(
                title='Sortino Ratio',
                calc_func='returns_acc.sortino_ratio',
                check_has_freq=True,
                check_has_year_freq=True,
                tags=['portfolio', 'returns']
            )
        ),
        copy_kwargs=dict(copy_mode='deep')
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    def returns_stats(self,
                      group_by: tp.GroupByLike = None,
                      benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                      freq: tp.Optional[tp.FrequencyLike] = None,
                      year_freq: tp.Optional[tp.FrequencyLike] = None,
                      use_asset_returns: bool = False,
                      defaults: tp.KwargsLike = None,
                      **kwargs) -> tp.SeriesFrame:
        """Compute various statistics on returns of this portfolio.

        See `Portfolio.returns_acc` and `vectorbt.returns.accessors.ReturnsAccessor.metrics`.

        `kwargs` will be passed to `vectorbt.returns.accessors.ReturnsAccessor.stats` method.
        If `benchmark_rets` is not set, uses `Portfolio.benchmark_returns`."""
        returns_acc = self.get_returns_acc(
            group_by=group_by,
            benchmark_rets=benchmark_rets,
            freq=freq,
            year_freq=year_freq,
            use_asset_returns=use_asset_returns,
            defaults=defaults
        )
        return getattr(returns_acc, 'stats')(**kwargs)

    # ############# 绘图方法 Plotting ############# #

    def plot_orders(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的订单图表
        
        在价格图表上显示订单的买入和卖出点，帮助分析交易策略的执行情况。
        
        Args:
            column: 要绘制的列名，如果为None则绘制第一列
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象
            
        示例:
            >>> pf.plot_orders()  # 绘制订单图表
            >>> pf.plot_orders(column='AAPL')  # 绘制指定列的订单
            
        Plot one column/group of orders.
        """
        kwargs = merge_dicts(dict(close_trace_kwargs=dict(name='Close')), kwargs)
        return self.orders.regroup(False).plot(column=column, **kwargs)

    def plot_trades(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的交易图表
        
        显示完整的交易（从入场到出场），包括盈利和亏损的交易。
        
        Args:
            column: 要绘制的列名，如果为None则绘制第一列
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象
            
        示例:
            >>> pf.plot_trades()  # 绘制交易图表
            
        Plot one column/group of trades.
        """
        kwargs = merge_dicts(dict(close_trace_kwargs=dict(name='Close')), kwargs)
        return self.trades.regroup(False).plot(column=column, **kwargs)

    def plot_trade_pnl(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的交易盈亏图表
        
        显示每笔交易的盈亏情况，帮助分析策略的盈利能力分布。
        
        Args:
            column: 要绘制的列名，如果为None则绘制第一列
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象
            
        示例:
            >>> pf.plot_trade_pnl()  # 绘制交易盈亏分布
            
        Plot one column/group of trade PnL.
        """
        return self.trades.regroup(False).plot_pnl(column=column, **kwargs)

    def plot_positions(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的仓位图表
        
        显示仓位的变化情况，包括多头和空头仓位的时间分布。
        
        Args:
            column: 要绘制的列名，如果为None则绘制第一列
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象
            
        示例:
            >>> pf.plot_positions()  # 绘制仓位图表
            
        Plot one column/group of positions.
        """
        kwargs = merge_dicts(dict(close_trace_kwargs=dict(name='Close')), kwargs)
        return self.positions.regroup(False).plot(column=column, **kwargs)

    def plot_position_pnl(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的仓位盈亏图表
        
        显示持仓期间的未实现盈亏变化，帮助分析仓位管理效果。
        
        Args:
            column: 要绘制的列名，如果为None则绘制第一列
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象
            
        示例:
            >>> pf.plot_position_pnl()  # 绘制仓位盈亏
            
        Plot one column/group of position PnL.
        """
        return self.positions.regroup(False).plot_pnl(column=column, **kwargs)

    def plot_asset_flow(self,
                        column: tp.Optional[tp.Label] = None,
                        direction: str = 'both',
                        xref: str = 'x',
                        yref: str = 'y',
                        hline_shape_kwargs: tp.KwargsLike = None,
                        **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列的资产流动图表
        
        显示资产数量随时间的变化，包括买入（正值）和卖出（负值）的资产流动。
        
        Args:
            column: 要绘制的列名
            direction: 方向筛选，参见 `vectorbt.portfolio.enums.Direction`
                - 'both': 显示所有方向
                - 'longonly': 仅显示多头方向  
                - 'shortonly': 仅显示空头方向
            xref: X坐标轴引用
            yref: Y坐标轴引用
            hline_shape_kwargs: 传递给零线的形状参数
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象，包含零基准线
            
        示例:
            >>> pf.plot_asset_flow()  # 绘制资产流动图
            >>> pf.plot_asset_flow(direction='longonly')  # 仅显示多头流动
            
        Plot one column of asset flow.

        Args:
            column (str): Name of the column to plot.
            direction (Direction): See `vectorbt.portfolio.enums.Direction`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zeroline.
            **kwargs: Keyword arguments passed to `vectorbt.generic.accessors.GenericAccessor.plot`.
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['brown']
                ),
                name='Assets'
            )
        ), kwargs)
        asset_flow = self.asset_flow(direction=direction)
        asset_flow = self.select_one_from_obj(asset_flow, self.wrapper.regroup(False), column=column)
        fig = asset_flow.vbt.plot(**kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0,
            x1=x_domain[1],
            y1=0
        ), hline_shape_kwargs))
        return fig

    def plot_cash_flow(self,
                       column: tp.Optional[tp.Label] = None,
                       group_by: tp.GroupByLike = None,
                       free: bool = False,
                       xref: str = 'x',
                       yref: str = 'y',
                       hline_shape_kwargs: tp.KwargsLike = None,
                       **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的现金流图表
        
        显示现金流入（正值）和流出（负值）随时间的变化情况。
        现金流图表帮助理解投资组合的资金使用效率。
        
        Args:
            column: 要绘制的列名/组名
            group_by: 分组或取消分组列，参见 `vectorbt.base.column_grouper.ColumnGrouper`
            free: 是否绘制自由现金流
                - True: 绘制自由现金流（考虑保证金等约束）
                - False: 绘制总现金流（默认）
            xref: X坐标轴引用
            yref: Y坐标轴引用
            hline_shape_kwargs: 传递给零线的形状参数
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象，包含零基准线
            
        示例:
            >>> pf.plot_cash_flow()  # 绘制总现金流
            >>> pf.plot_cash_flow(free=True)  # 绘制自由现金流
            
        Plot one column/group of cash flow.

        Args:
            column (str): Name of the column/group to plot.
            group_by (any): Group or ungroup columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            free (bool): Whether to plot the flow of the free cash.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zeroline.
            **kwargs: Keyword arguments passed to `vectorbt.generic.accessors.GenericAccessor.plot`.
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['green']
                ),
                name='Cash'
            )
        ), kwargs)
        cash_flow = self.cash_flow(group_by=group_by, free=free)
        cash_flow = self.select_one_from_obj(cash_flow, self.wrapper.regroup(group_by), column=column)
        fig = cash_flow.vbt.plot(**kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0.,
            x1=x_domain[1],
            y1=0.
        ), hline_shape_kwargs))
        return fig

    def plot_assets(self,
                    column: tp.Optional[tp.Label] = None,
                    direction: str = 'both',
                    xref: str = 'x',
                    yref: str = 'y',
                    hline_shape_kwargs: tp.KwargsLike = None,
                    **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列的资产持有量图表
        
        显示资产持有量随时间的变化，包括多头（正值）和空头（负值）仓位。
        此图表与价格图表叠加显示，帮助分析仓位管理策略。
        
        Args:
            column: 要绘制的列名
            direction: 方向筛选，参见 `vectorbt.portfolio.enums.Direction`
                - 'both': 显示所有方向的仓位（默认）
                - 'longonly': 仅显示多头仓位  
                - 'shortonly': 仅显示空头仓位
            xref: X坐标轴引用
            yref: Y坐标轴引用
            hline_shape_kwargs: 传递给零线的形状参数
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象，显示资产持有量与价格的对比
            
        示例:
            >>> pf.plot_assets()  # 绘制资产持有量图
            >>> pf.plot_assets(direction='longonly')  # 仅显示多头仓位
            
        Plot one column of assets.

        Args:
            column (str): Name of the column to plot.
            direction (Direction): See `vectorbt.portfolio.enums.Direction`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zeroline.
            **kwargs: Keyword arguments passed to `vectorbt.generic.accessors.GenericSRAccessor.plot_against`.
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['brown']
                ),
                name='Assets'
            ),
            pos_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['brown'], 0.3)
            ),
            neg_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['orange'], 0.3)
            ),
            other_trace_kwargs='hidden'
        ), kwargs)
        assets = self.assets(direction=direction)
        assets = self.select_one_from_obj(assets, self.wrapper.regroup(False), column=column)
        fig = assets.vbt.plot_against(0, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0.,
            x1=x_domain[1],
            y1=0.
        ), hline_shape_kwargs))
        return fig

    def plot_cash(self,
                  column: tp.Optional[tp.Label] = None,
                  group_by: tp.GroupByLike = None,
                  free: bool = False,
                  xref: str = 'x',
                  yref: str = 'y',
                  hline_shape_kwargs: tp.KwargsLike = None,
                  **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的现金余额图表
        
        显示现金余额随时间的变化情况。现金余额图表帮助监控
        投资组合的流动性状况和资金配置效率。
        
        Args:
            column: 要绘制的列名/组名
            group_by: 分组或取消分组列，参见 `vectorbt.base.column_grouper.ColumnGrouper`
            free: 是否绘制自由现金余额
                - True: 绘制自由现金余额（考虑保证金等约束）
                - False: 绘制总现金余额（默认）
            xref: X坐标轴引用
            yref: Y坐标轴引用
            hline_shape_kwargs: 传递给零线的形状参数
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象，显示现金余额的时间序列
            
        示例:
            >>> pf.plot_cash()  # 绘制总现金余额
            >>> pf.plot_cash(free=True)  # 绘制自由现金余额
            
        Plot one column/group of cash balance.

        Args:
            column (str): Name of the column/group to plot.
            group_by (any): Group or ungroup columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            free (bool): Whether to plot the flow of the free cash.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zeroline.
            **kwargs: Keyword arguments passed to `vectorbt.generic.accessors.GenericSRAccessor.plot_against`.
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['green']
                ),
                name='Cash'
            ),
            pos_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['green'], 0.3)
            ),
            neg_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['red'], 0.3)
            ),
            other_trace_kwargs='hidden'
        ), kwargs)
        init_cash = self.get_init_cash(group_by=group_by)
        init_cash = self.select_one_from_obj(init_cash, self.wrapper.regroup(group_by), column=column)
        cash = self.cash(group_by=group_by, free=free)
        cash = self.select_one_from_obj(cash, self.wrapper.regroup(group_by), column=column)
        fig = cash.vbt.plot_against(init_cash, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=init_cash,
            x1=x_domain[1],
            y1=init_cash
        ), hline_shape_kwargs))
        return fig

    def plot_asset_value(self,
                         column: tp.Optional[tp.Label] = None,
                         group_by: tp.GroupByLike = None,
                         direction: str = 'both',
                         xref: str = 'x',
                         yref: str = 'y',
                         hline_shape_kwargs: tp.KwargsLike = None,
                         **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的资产价值图表
        
        显示持有资产的市场价值随时间的变化。资产价值等于持有数量乘以当前价格，
        这是投资组合总价值的重要组成部分。
        
        Args:
            column: 要绘制的列名/组名
            group_by: 分组或取消分组列，参见 `vectorbt.base.column_grouper.ColumnGrouper`
            direction: 方向筛选，参见 `vectorbt.portfolio.enums.Direction`
                - 'both': 显示所有方向的资产价值（默认）
                - 'longonly': 仅显示多头资产价值
                - 'shortonly': 仅显示空头资产价值
            xref: X坐标轴引用
            yref: Y坐标轴引用
            hline_shape_kwargs: 传递给零线的形状参数
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象，显示资产价值与零线的对比
            
        示例:
            >>> pf.plot_asset_value()  # 绘制总资产价值
            >>> pf.plot_asset_value(direction='longonly')  # 仅显示多头资产价值
            
        Plot one column/group of asset value.

        Args:
            column (str): Name of the column/group to plot.
            group_by (any): Group or ungroup columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            direction (Direction): See `vectorbt.portfolio.enums.Direction`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zeroline.
            **kwargs: Keyword arguments passed to `vectorbt.generic.accessors.GenericSRAccessor.plot_against`.
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['cyan']
                ),
                name='Asset Value'
            ),
            pos_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['cyan'], 0.3)
            ),
            neg_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['orange'], 0.3)
            ),
            other_trace_kwargs='hidden'
        ), kwargs)
        asset_value = self.asset_value(direction=direction, group_by=group_by)
        asset_value = self.select_one_from_obj(asset_value, self.wrapper.regroup(group_by), column=column)
        fig = asset_value.vbt.plot_against(0, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0.,
            x1=x_domain[1],
            y1=0.
        ), hline_shape_kwargs))
        return fig

    def plot_value(self,
                   column: tp.Optional[tp.Label] = None,
                   group_by: tp.GroupByLike = None,
                   xref: str = 'x',
                   yref: str = 'y',
                   hline_shape_kwargs: tp.KwargsLike = None,
                   **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的投资组合价值图表
        
        显示投资组合总价值（现金 + 资产价值）随时间的变化。
        这是最重要的绩效图表，直观显示投资组合的整体表现。
        
        Args:
            column: 要绘制的列名/组名
            group_by: 分组或取消分组列，参见 `vectorbt.base.column_grouper.ColumnGrouper`
            xref: X坐标轴引用
            yref: Y坐标轴引用
            hline_shape_kwargs: 传递给基准线（初始现金）的形状参数
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象，显示投资组合价值与初始现金的对比
            
        示例:
            >>> pf.plot_value()  # 绘制投资组合总价值
            >>> pf.plot_value(column='AAPL')  # 绘制特定列的价值
            
        Plot one column/group of value.

        Args:
            column (str): Name of the column/group to plot.
            group_by (any): Group or ungroup columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for baseline.
            **kwargs: Keyword arguments passed to `vectorbt.generic.accessors.GenericSRAccessor.plot_against`.
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['purple']
                ),
                name='Value'
            ),
            other_trace_kwargs='hidden'
        ), kwargs)
        init_cash = self.get_init_cash(group_by=group_by)
        init_cash = self.select_one_from_obj(init_cash, self.wrapper.regroup(group_by), column=column)
        value = self.value(group_by=group_by)
        value = self.select_one_from_obj(value, self.wrapper.regroup(group_by), column=column)
        fig = value.vbt.plot_against(init_cash, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=init_cash,
            x1=x_domain[1],
            y1=init_cash
        ), hline_shape_kwargs))
        return fig

    def plot_cum_returns(self,
                         column: tp.Optional[tp.Label] = None,
                         group_by: tp.GroupByLike = None,
                         benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                         use_asset_returns: bool = False,
                         **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的累计收益率图表
        
        显示累计收益率随时间的变化，并与基准收益率进行对比。
        这是评估投资组合相对表现的重要图表。
        
        Args:
            column: 要绘制的列名/组名
            group_by: 分组或取消分组列，参见 `vectorbt.base.column_grouper.ColumnGrouper`
            benchmark_rets: 基准收益率数据

                如果为 None，将使用 `Portfolio.benchmark_returns`。
            use_asset_returns: 是否绘制资产收益率
                - True: 使用资产收益率（不受现金影响）
                - False: 使用投资组合收益率（默认）
            **kwargs: 传递给 `vectorbt.returns.accessors.ReturnsSRAccessor.plot_cumulative` 的参数
            
        Returns:
            Plotly图表对象，显示累计收益率与基准的对比
            
        示例:
            >>> pf.plot_cum_returns()  # 绘制累计收益率
            >>> pf.plot_cum_returns(use_asset_returns=True)  # 使用资产收益率
            
        Plot one column/group of cumulative returns.

        Args:
            column (str): Name of the column/group to plot.
            group_by (any): Group or ungroup columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            benchmark_rets (array_like): Benchmark returns.

                If None, will use `Portfolio.benchmark_returns`.
            use_asset_returns (bool): Whether to plot asset returns.
            **kwargs: Keyword arguments passed to `vectorbt.returns.accessors.ReturnsSRAccessor.plot_cumulative`.
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        if benchmark_rets is None:
            benchmark_rets = self.benchmark_returns(group_by=group_by)
        else:
            benchmark_rets = broadcast_to(benchmark_rets, self.obj)
        benchmark_rets = self.select_one_from_obj(benchmark_rets, self.wrapper.regroup(group_by), column=column)
        kwargs = merge_dicts(dict(
            benchmark_rets=benchmark_rets,
            main_kwargs=dict(
                trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg['color_schema']['purple']
                    ),
                    name='Value'
                )
            ),
            hline_shape_kwargs=dict(
                type='line',
                line=dict(
                    color='gray',
                    dash="dash",
                )
            )
        ), kwargs)
        if use_asset_returns:
            returns = self.asset_returns(group_by=group_by)
        else:
            returns = self.returns(group_by=group_by)
        returns = self.select_one_from_obj(returns, self.wrapper.regroup(group_by), column=column)
        return returns.vbt.returns.plot_cumulative(**kwargs)

    def plot_drawdowns(self,
                       column: tp.Optional[tp.Label] = None,
                       group_by: tp.GroupByLike = None,
                       **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的回撤图表
        
        显示投资组合的回撤事件，包括回撤幅度、持续时间和恢复过程。
        回撤图表是风险评估的重要工具。
        
        Args:
            column: 要绘制的列名/组名
            group_by: 分组或取消分组列，参见 `vectorbt.base.column_grouper.ColumnGrouper`
            **kwargs: 传递给 `vectorbt.generic.drawdowns.Drawdowns.plot` 的参数
            
        Returns:
            Plotly图表对象，显示回撤事件的详细信息
            
        示例:
            >>> pf.plot_drawdowns()  # 绘制回撤图表
            
        Plot one column/group of drawdowns.

        Args:
            column (str): Name of the column/group to plot.
            group_by (any): Group or ungroup columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            **kwargs: Keyword arguments passed to `vectorbt.generic.drawdowns.Drawdowns.plot`.
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        kwargs = merge_dicts(dict(
            ts_trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['purple']
                ),
                name='Value'
            )
        ), kwargs)
        return self.get_drawdowns(group_by=group_by).plot(column=column, **kwargs)

    def plot_underwater(self,
                        column: tp.Optional[tp.Label] = None,
                        group_by: tp.GroupByLike = None,
                        xref: str = 'x',
                        yref: str = 'y',
                        hline_shape_kwargs: tp.KwargsLike = None,
                        **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的水下图（回撤百分比）
        
        显示投资组合相对于历史最高点的回撤百分比。水下图提供了
        连续的回撤视图，帮助理解投资组合的风险暴露程度。
        
        Args:
            column: 要绘制的列名/组名
            group_by: 分组或取消分组列，参见 `vectorbt.base.column_grouper.ColumnGrouper`
            xref: X坐标轴引用
            yref: Y坐标轴引用
            hline_shape_kwargs: 传递给零线的形状参数
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象，显示回撤百分比的时间序列
            
        示例:
            >>> pf.plot_underwater()  # 绘制水下图
            
        Plot one column/group of underwater.

        Args:
            column (str): Name of the column/group to plot.
            group_by (any): Group or ungroup columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zeroline.
            **kwargs: Keyword arguments passed to `vectorbt.generic.accessors.GenericAccessor.plot`.
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['red']
                ),
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['red'], 0.3),
                fill='tozeroy',
                name='Drawdown'
            )
        ), kwargs)
        drawdown = self.drawdown(group_by=group_by)
        drawdown = self.select_one_from_obj(drawdown, self.wrapper.regroup(group_by), column=column)
        fig = drawdown.vbt.plot(**kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0,
            x1=x_domain[1],
            y1=0
        ), hline_shape_kwargs))
        yaxis = 'yaxis' + yref[1:]
        fig.layout[yaxis]['tickformat'] = '%'
        return fig

    def plot_gross_exposure(self,
                            column: tp.Optional[tp.Label] = None,
                            group_by: tp.GroupByLike = None,
                            direction: str = 'both',
                            xref: str = 'x',
                            yref: str = 'y',
                            hline_shape_kwargs: tp.KwargsLike = None,
                            **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的总敞口图表
        
        显示投资组合的总敞口随时间的变化。总敞口反映了投资组合的
        杠杆程度，值为1表示全仓，大于1表示使用杠杆。
        
        Args:
            column: 要绘制的列名/组名
            group_by: 分组或取消分组列，参见 `vectorbt.base.column_grouper.ColumnGrouper`
            direction: 方向筛选，参见 `vectorbt.portfolio.enums.Direction`
                - 'both': 显示所有方向的敞口（默认）
                - 'longonly': 仅显示多头敞口
                - 'shortonly': 仅显示空头敞口
            xref: X坐标轴引用
            yref: Y坐标轴引用
            hline_shape_kwargs: 传递给基准线（1.0）的形状参数
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象，显示总敞口与基准线的对比
            
        示例:
            >>> pf.plot_gross_exposure()  # 绘制总敞口
            >>> pf.plot_gross_exposure(direction='longonly')  # 仅多头敞口
            
        Plot one column/group of gross exposure.

        Args:
            column (str): Name of the column/group to plot.
            group_by (any): Group or ungroup columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            direction (Direction): See `vectorbt.portfolio.enums.Direction`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for baseline.
            **kwargs: Keyword arguments passed to `vectorbt.generic.accessors.GenericSRAccessor.plot_against`.
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['pink']
                ),
                name='Exposure'
            ),
            pos_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['orange'], 0.3)
            ),
            neg_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['pink'], 0.3)
            ),
            other_trace_kwargs='hidden'
        ), kwargs)
        gross_exposure = self.gross_exposure(direction=direction, group_by=group_by)
        gross_exposure = self.select_one_from_obj(gross_exposure, self.wrapper.regroup(group_by), column=column)
        fig = gross_exposure.vbt.plot_against(1, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=1,
            x1=x_domain[1],
            y1=1
        ), hline_shape_kwargs))
        return fig

    def plot_net_exposure(self,
                          column: tp.Optional[tp.Label] = None,
                          group_by: tp.GroupByLike = None,
                          xref: str = 'x',
                          yref: str = 'y',
                          hline_shape_kwargs: tp.KwargsLike = None,
                          **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制一列/组的净敞口图表
        
        显示投资组合的净敞口随时间的变化。净敞口等于多头敞口减去空头敞口，
        反映了投资组合的方向性偏向。正值偏向多头，负值偏向空头，0为市场中性。
        
        Args:
            column: 要绘制的列名/组名
            group_by: 分组或取消分组列，参见 `vectorbt.base.column_grouper.ColumnGrouper`
            xref: X坐标轴引用
            yref: Y坐标轴引用
            hline_shape_kwargs: 传递给零线的形状参数
            **kwargs: 传递给绘图函数的其他参数
            
        Returns:
            Plotly图表对象，显示净敞口与零线的对比
            
        示例:
            >>> pf.plot_net_exposure()  # 绘制净敞口
            
        Plot one column/group of net exposure.

        Args:
            column (str): Name of the column/group to plot.
            group_by (any): Group or ungroup columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zeroline.
            **kwargs: Keyword arguments passed to `vectorbt.generic.accessors.GenericSRAccessor.plot_against`.
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['pink']
                ),
                name='Exposure'
            ),
            pos_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['pink'], 0.3)
            ),
            neg_trace_kwargs=dict(
                fillcolor=adjust_opacity(plotting_cfg['color_schema']['orange'], 0.3)
            ),
            other_trace_kwargs='hidden'
        ), kwargs)
        net_exposure = self.net_exposure(group_by=group_by)
        net_exposure = self.select_one_from_obj(net_exposure, self.wrapper.regroup(group_by), column=column)
        fig = net_exposure.vbt.plot_against(0, **kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(**merge_dicts(dict(
            type='line',
            line=dict(
                color='gray',
                dash="dash",
            ),
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0,
            x1=x_domain[1],
            y1=0
        ), hline_shape_kwargs))
        return fig

    @property  # 属性装饰器
    def plots_defaults(self) -> tp.Kwargs:
        """
        Portfolio.plot的默认参数
        
        合并绘图构建器混入类和设置文件中的默认绘图参数，
        为投资组合绘图提供全面的默认配置。
        
        Returns:
            绘图的默认参数字典
            
        Defaults for `Portfolio.plot`.

        合并 `vectorbt.generic.plots_builder.PlotsBuilderMixin.plots_defaults` 和
        `vectorbt._settings.settings` 中的 `portfolio.plots`。
        """
        from vectorbt._settings import settings
        returns_cfg = settings['returns']  # 收益率配置
        portfolio_plots_cfg = settings['portfolio']['plots']  # 投资组合绘图配置

        return merge_dicts(
            PlotsBuilderMixin.plots_defaults.__get__(self),  # 绘图构建器的默认参数
            dict(
                settings=dict(
                    year_freq=returns_cfg['year_freq'],  # 年化频率
                    trades_type=self.trades_type         # 交易类型
                )
            ),
            portfolio_plots_cfg  # 投资组合绘图配置
        )

    # 预定义的投资组合子图配置
    # 这些子图用于生成完整的投资组合分析报告
    _subplots: tp.ClassVar[Config] = Config(
        dict(
            # 交易相关图表
            orders=dict(
                title="Orders",                    # 图表标题：订单
                yaxis_kwargs=dict(title="Price"), # Y轴标题：价格
                check_is_not_grouped=True,        # 检查是否未分组
                plot_func='orders.plot',          # 绘图函数：订单绘图
                tags=['portfolio', 'orders']      # 标签：投资组合、订单
            ),
            trades=dict(
                title="Trades",                    # 图表标题：交易
                yaxis_kwargs=dict(title="Price"), # Y轴标题：价格
                check_is_not_grouped=True,        # 检查是否未分组
                plot_func='trades.plot',          # 绘图函数：交易绘图
                tags=['portfolio', 'trades']      # 标签：投资组合、交易
            ),
            trade_pnl=dict(
                title="Trade PnL",                  # 图表标题：交易盈亏
                yaxis_kwargs=dict(title="Trade PnL"), # Y轴标题：交易盈亏
                check_is_not_grouped=True,          # 检查是否未分组
                plot_func='trades.plot_pnl',        # 绘图函数：交易盈亏绘图
                tags=['portfolio', 'trades']        # 标签：投资组合、交易
            ),
            
            # 资产流动图表
            asset_flow=dict(
                title="Asset Flow",                 # 图表标题：资产流动
                yaxis_kwargs=dict(title="Asset flow"), # Y轴标题：资产流动
                check_is_not_grouped=True,          # 检查是否未分组
                plot_func='plot_asset_flow',        # 绘图函数：资产流动绘图
                pass_add_trace_kwargs=True,         # 传递追加轨迹参数
                tags=['portfolio', 'assets']        # 标签：投资组合、资产
            ),
            cash_flow=dict(
                title="Cash Flow",                  # 图表标题：现金流
                yaxis_kwargs=dict(title="Cash flow"), # Y轴标题：现金流
                plot_func='plot_cash_flow',         # 绘图函数：现金流绘图
                pass_add_trace_kwargs=True,         # 传递追加轨迹参数
                tags=['portfolio', 'cash']          # 标签：投资组合、现金
            ),
            
            # 资产和现金图表
            assets=dict(
                title="Assets",                     # 图表标题：资产
                yaxis_kwargs=dict(title="Assets"), # Y轴标题：资产
                check_is_not_grouped=True,         # 检查是否未分组
                plot_func='plot_assets',           # 绘图函数：资产绘图
                pass_add_trace_kwargs=True,        # 传递追加轨迹参数
                tags=['portfolio', 'assets']       # 标签：投资组合、资产
            ),
            cash=dict(
                title="Cash",                      # 图表标题：现金
                yaxis_kwargs=dict(title="Cash"),  # Y轴标题：现金
                plot_func='plot_cash',            # 绘图函数：现金绘图
                pass_add_trace_kwargs=True,       # 传递追加轨迹参数
                tags=['portfolio', 'cash']        # 标签：投资组合、现金
            ),
            
            # 价值图表
            asset_value=dict(
                title="Asset Value",                   # 图表标题：资产价值
                yaxis_kwargs=dict(title="Asset value"), # Y轴标题：资产价值
                plot_func='plot_asset_value',          # 绘图函数：资产价值绘图
                pass_add_trace_kwargs=True,            # 传递追加轨迹参数
                tags=['portfolio', 'assets', 'value']  # 标签：投资组合、资产、价值
            ),
            value=dict(
                title="Value",                     # 图表标题：价值
                yaxis_kwargs=dict(title="Value"), # Y轴标题：价值
                plot_func='plot_value',           # 绘图函数：价值绘图
                pass_add_trace_kwargs=True,       # 传递追加轨迹参数
                tags=['portfolio', 'value']       # 标签：投资组合、价值
            ),
            
            # 收益率图表
            cum_returns=dict(
                title="Cumulative Returns",               # 图表标题：累计收益率
                yaxis_kwargs=dict(title="Cumulative returns"), # Y轴标题：累计收益率
                plot_func='plot_cum_returns',             # 绘图函数：累计收益率绘图
                pass_hline_shape_kwargs=True,             # 传递水平线形状参数
                pass_add_trace_kwargs=True,               # 传递追加轨迹参数
                pass_xref=True,                           # 传递X轴引用
                pass_yref=True,                           # 传递Y轴引用
                tags=['portfolio', 'returns']             # 标签：投资组合、收益率
            ),
            
            # 回撤图表
            drawdowns=dict(
                title="Drawdowns",                 # 图表标题：回撤
                yaxis_kwargs=dict(title="Value"), # Y轴标题：价值
                plot_func='plot_drawdowns',       # 绘图函数：回撤绘图
                pass_add_trace_kwargs=True,       # 传递追加轨迹参数
                pass_xref=True,                   # 传递X轴引用
                pass_yref=True,                   # 传递Y轴引用
                tags=['portfolio', 'value', 'drawdowns']  # 标签：投资组合、价值、回撤
            ),
            underwater=dict(
                title="Underwater",                  # 图表标题：水下图
                yaxis_kwargs=dict(title="Drawdown"), # Y轴标题：回撤
                plot_func='plot_underwater',         # 绘图函数：水下图绘图
                pass_add_trace_kwargs=True,          # 传递追加轨迹参数
                tags=['portfolio', 'value', 'drawdowns']  # 标签：投资组合、价值、回撤
            ),
            
            # 敞口图表
            gross_exposure=dict(
                title="Gross Exposure",                    # 图表标题：总敞口
                yaxis_kwargs=dict(title="Gross exposure"), # Y轴标题：总敞口
                plot_func='plot_gross_exposure',           # 绘图函数：总敞口绘图
                pass_add_trace_kwargs=True,                # 传递追加轨迹参数
                tags=['portfolio', 'exposure']             # 标签：投资组合、敞口
            ),
            net_exposure=dict(
                title="Net Exposure",                     # 图表标题：净敞口
                yaxis_kwargs=dict(title="Net exposure"),  # Y轴标题：净敞口
                plot_func='plot_net_exposure',            # 绘图函数：净敞口绘图
                pass_add_trace_kwargs=True,               # 传递追加轨迹参数
                tags=['portfolio', 'exposure']            # 标签：投资组合、敞口
            )
        ),
        copy_kwargs=dict(copy_mode='deep')  # 深拷贝参数
    )

    plot = PlotsBuilderMixin.plots  # 继承绘图构建器的绘图方法

    @property  # 属性装饰器
    def subplots(self) -> Config:
        """
        子图配置
        
        返回预定义的子图配置，用于创建完整的投资组合分析仪表板。
        
        Returns:
            包含所有子图配置的Config对象
        """
        return self._subplots


# 重写文档字符串
Portfolio.override_metrics_doc(__pdoc__)   # 重写指标文档
Portfolio.override_subplots_doc(__pdoc__)  # 重写子图文档

# 设置绘图方法的文档引用
__pdoc__['Portfolio.plot'] = "参见 `vectorbt.generic.plots_builder.PlotsBuilderMixin.plots`。"
