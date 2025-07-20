# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT PORTFOLIO DECORATORS MODULE: 投资组合装饰器模块
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于Portfolio类功能扩展的装饰器模块。该模块的核心
设计理念是通过装饰器模式实现Portfolio类与ReturnsAccessor功能的无缝集成，为用户提供
便捷的收益率分析方法访问接口。

核心设计理念：
1. **自动方法注入**：通过装饰器自动为Portfolio类添加ReturnsAccessor的方法
2. **配置驱动**：基于配置字典动态生成方法，提高代码的可维护性和扩展性
3. **方法代理模式**：生成的方法作为ReturnsAccessor方法的代理，保持API一致性
4. **缓存优化**：所有生成的方法都使用cached_method装饰器，提高访问性能

技术特点：
- **动态方法生成**：运行时动态创建方法并绑定到类上
- **闭包技术**：使用闭包保持方法参数的正确性
- **反射机制**：通过getattr动态调用ReturnsAccessor的方法
- **文档字符串自动生成**：为生成的方法自动创建文档字符串

应用场景：
- Portfolio类的功能扩展
- 第三方库方法的集成
- API接口的统一化
- 性能优化和缓存管理

与vectorbt生态系统的关系：
- **Portfolio集成**：主要服务于Portfolio类的功能扩展
- **ReturnsAccessor桥接**：建立Portfolio与ReturnsAccessor之间的桥梁
- **配置系统集成**：使用vectorbt的Config系统管理方法配置
- **缓存系统集成**：与vectorbt的缓存系统深度集成

该模块虽然代码量不大，但在vectorbt架构中起到了关键的桥梁作用，使得Portfolio类
能够直接访问丰富的收益率分析功能，大大提升了用户体验和API的易用性。

使用示例：
```python
import pandas as pd
import numpy as np
import vectorbt as vbt

# Portfolio类通过装饰器自动获得了收益率分析方法
np.random.seed(42)
price = pd.DataFrame({
    'AAPL': np.random.uniform(100, 200, size=100),
    'GOOGL': np.random.uniform(2000, 3000, size=100)
}, index=pd.date_range('2023-01-01', periods=100))

size = pd.DataFrame({
    'AAPL': np.random.uniform(-10, 10, size=100),
    'GOOGL': np.random.uniform(-5, 5, size=100)
}, index=pd.date_range('2023-01-01', periods=100))

# 创建投资组合
pf = vbt.Portfolio.from_orders(price, size, fees=0.01)

# 直接调用收益率分析方法（由装饰器自动添加）
daily_returns = pf.daily_returns()      # 日收益率
annual_returns = pf.annual_returns()    # 年收益率
sharpe_ratio = pf.sharpe_ratio()        # 夏普比率
max_drawdown = pf.max_drawdown()        # 最大回撤

print(f"夏普比率: {sharpe_ratio}")
print(f"最大回撤: {max_drawdown}")
```

注意事项：
- 装饰器必须在类定义完成前应用
- 生成的方法会覆盖同名的现有方法
- 所有生成的方法都支持缓存，提高性能
- 方法的参数和行为与ReturnsAccessor保持一致
================================================================================

类和函数装饰器模块

该模块提供了用于Portfolio类功能扩展的装饰器，主要用于自动添加收益率分析方法。
"""

# 导入类型注解模块，用于类型提示和泛型定义
from vectorbt import _typing as tp
# 导入检查工具模块，用于参数验证和类型检查
from vectorbt.utils import checks
# 导入配置管理模块，用于处理方法配置
from vectorbt.utils.config import Config
# 导入装饰器工具模块，用于方法缓存
from vectorbt.utils.decorators import cached_method

# 定义包装器函数类型，用于类装饰器的类型提示
# 该类型表示一个接受类类型参数并返回同类型的可调用对象
WrapperFuncT = tp.Callable[[tp.Type[tp.T]], tp.Type[tp.T]]


def attach_returns_acc_methods(config: Config) -> WrapperFuncT:
    """
    类装饰器：为Portfolio类自动添加ReturnsAccessor的方法
    
    这是本模块的核心函数，实现了Portfolio类与ReturnsAccessor功能的无缝集成。
    该装饰器通过配置驱动的方式，自动为Portfolio类生成对应的收益率分析方法，
    使用户能够直接在Portfolio实例上调用各种收益率分析功能。
    
    设计原理：
    1. **配置解析**：解析传入的配置字典，获取方法名称和参数
    2. **动态方法生成**：为每个配置项动态创建对应的方法
    3. **方法代理**：生成的方法作为ReturnsAccessor方法的代理
    4. **缓存集成**：所有生成的方法都集成缓存功能
    
    技术实现：
    - 使用闭包技术保持方法参数的正确性
    - 通过反射机制动态调用ReturnsAccessor的方法
    - 自动生成方法的文档字符串和元数据
    - 集成vectorbt的缓存系统提高性能
    
    参数：
        config (Config): 方法配置字典，包含目标方法名（键）和配置信息（值）
            配置字典的结构：
            {
                'target_method_name': {
                    'source_name': str,     # 源方法名称，默认为目标名称
                    'docstring': str        # 方法文档字符串，默认自动生成
                }
            }
    
    返回：
        WrapperFuncT: 类装饰器函数，用于装饰Portfolio类
    
    异常：
        AssertionError: 当被装饰的类不是Portfolio子类时抛出
    
    配置示例：
    ```python
    from vectorbt.utils.config import Config
    
    # 基础配置示例
    returns_config = Config({
        'daily_returns': {
            'source_name': 'daily',
            'docstring': '获取日收益率数据'
        },
        'sharpe_ratio': {
            'source_name': 'sharpe_ratio',
            'docstring': '计算夏普比率'
        },
        'max_drawdown': {
            'docstring': '计算最大回撤'  # source_name默认为'max_drawdown'
        }
    })
    
    @attach_returns_acc_methods(returns_config)
    class MyPortfolio(Portfolio):
        pass
    ```
    
    使用示例：
    ```python
    import pandas as pd
    import numpy as np
    import vectorbt as vbt
    from vectorbt.utils.config import Config
    
    # 1. 基础使用示例
    # Portfolio类已经通过该装饰器预配置了常用的收益率方法
    np.random.seed(42)
    prices = pd.DataFrame({
        'AAPL': np.random.uniform(100, 200, 100),
        'GOOGL': np.random.uniform(2000, 3000, 100),
        'MSFT': np.random.uniform(200, 400, 100)
    }, index=pd.date_range('2023-01-01', periods=100))
    
    orders = pd.DataFrame({
        'AAPL': np.random.uniform(-5, 5, 100),
        'GOOGL': np.random.uniform(-2, 2, 100),
        'MSFT': np.random.uniform(-3, 3, 100)
    }, index=pd.date_range('2023-01-01', periods=100))
    
    # 创建投资组合
    pf = vbt.Portfolio.from_orders(prices, orders, fees=0.001)
    
    # 直接调用装饰器添加的方法
    daily_rets = pf.daily_returns()           # 日收益率
    annual_rets = pf.annual_returns()         # 年收益率
    cumulative_rets = pf.cumulative_returns() # 累积收益率
    
    # 风险指标
    sharpe = pf.sharpe_ratio()                # 夏普比率
    sortino = pf.sortino_ratio()              # 索提诺比率
    max_dd = pf.max_drawdown()                # 最大回撤
    
    # 波动率指标
    volatility = pf.annualized_volatility()   # 年化波动率
    downside_risk = pf.downside_risk()        # 下行风险
    
    print(f"年化收益率: {annual_rets.iloc[-1]:.2%}")
    print(f"夏普比率: {sharpe:.4f}")
    print(f"最大回撤: {max_dd:.2%}")
    print(f"年化波动率: {volatility:.2%}")
    
    # 2. 自定义装饰器配置示例
    custom_config = Config({
        'custom_return_metric': {
            'source_name': 'total',
            'docstring': '自定义总收益率指标'
        },
        'risk_adjusted_return': {
            'source_name': 'calmar_ratio',
            'docstring': '风险调整收益率（卡尔马比率）'
        },
        'tail_risk': {
            'source_name': 'value_at_risk',
            'docstring': '尾部风险（VaR）'
        }
    })
    
    @attach_returns_acc_methods(custom_config)
    class CustomPortfolio(vbt.Portfolio):
        '''自定义投资组合类，包含额外的收益率分析方法'''
        pass
    
    # 使用自定义Portfolio类
    custom_pf = CustomPortfolio.from_orders(prices, orders, fees=0.001)
    
    # 调用自定义添加的方法
    total_return = custom_pf.custom_return_metric()
    calmar = custom_pf.risk_adjusted_return()
    var = custom_pf.tail_risk()
    
    print(f"总收益率: {total_return:.2%}")
    print(f"卡尔马比率: {calmar:.4f}")
    print(f"风险价值: {var:.4f}")
    
    # 3. 方法参数传递示例
    # 装饰器生成的方法支持所有ReturnsAccessor方法的参数
    benchmark_prices = pd.Series(
        np.random.uniform(2500, 3500, 100),
        index=pd.date_range('2023-01-01', periods=100)
    )
    benchmark_returns = benchmark_prices.pct_change().dropna()
    
    # 使用基准收益率计算指标
    alpha = pf.alpha(benchmark_rets=benchmark_returns)
    beta = pf.beta(benchmark_rets=benchmark_returns)
    info_ratio = pf.information_ratio(benchmark_rets=benchmark_returns)
    
    print(f"Alpha: {alpha:.4f}")
    print(f"Beta: {beta:.4f}")
    print(f"信息比率: {info_ratio:.4f}")
    
    # 4. 分组和频率设置示例
    # 按资产类别分组
    sectors = ['Tech', 'Tech', 'Tech']  # 假设都是科技股
    
    # 使用分组参数
    sector_returns = pf.annual_returns(group_by=sectors)
    sector_sharpe = pf.sharpe_ratio(group_by=sectors)
    
    print(f"科技板块年化收益率: {sector_returns:.2%}")
    print(f"科技板块夏普比率: {sector_sharpe:.4f}")
    
    # 使用不同频率
    monthly_returns = pf.returns(freq='M')  # 月度收益率
    weekly_vol = pf.annualized_volatility(freq='W')  # 周频年化波动率
    
    print(f"月度收益率样本:\n{monthly_returns.head()}")
    print(f"周频年化波动率: {weekly_vol:.2%}")
    ```
    
    实际应用场景：
    1. **量化策略回测**：快速评估策略的收益率特征
    2. **风险管理**：计算各种风险指标和风险调整收益率
    3. **组合优化**：比较不同组合的收益率表现
    4. **绩效归因**：分析投资组合的收益来源
    5. **合规报告**：生成监管要求的收益率报告
    
    性能优化：
    - 所有生成的方法都使用缓存，避免重复计算
    - 支持向量化操作，处理大规模数据集
    - 集成numba加速，提高数值计算性能
    
    注意事项：
    1. 被装饰的类必须是Portfolio的子类
    2. 生成的方法会覆盖同名的现有方法
    3. 方法参数与ReturnsAccessor保持一致
    4. 缓存机制依赖于参数的哈希值
    5. 配置字典的键名将成为生成的方法名
    """

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        """
        内部包装器函数：执行实际的类装饰逻辑
        
        该函数是装饰器的核心实现，负责验证目标类、解析配置、
        生成方法并将其绑定到目标类上。
        
        参数：
            cls: 要被装饰的类，必须是Portfolio的子类
            
        返回：
            装饰后的类，包含新添加的方法
            
        异常：
            AssertionError: 当cls不是Portfolio子类时抛出
        """
        # 验证被装饰的类必须是Portfolio的子类
        # 这确保了生成的方法能够正确访问Portfolio的get_returns_acc方法
        checks.assert_subclass_of(cls, "Portfolio")

        # 遍历配置字典中的每个方法配置项
        for target_name, settings in config.items():
            # 获取源方法名称，如果未指定则使用目标名称
            # 这允许为ReturnsAccessor的方法创建别名
            source_name = settings.get('source_name', target_name)
            
            # 获取或生成方法文档字符串
            # 默认文档字符串指向ReturnsAccessor的对应方法
            docstring = settings.get('docstring', f"See `vectorbt.returns.accessors.ReturnsAccessor.{source_name}`.")

            # 定义新方法的实现
            # 使用闭包技术保持source_name的正确引用
            def new_method(self,
                           *,  # 强制使用关键字参数，提高API的清晰性
                           group_by: tp.GroupByLike = None,                    # 分组参数
                           benchmark_rets: tp.Optional[tp.ArrayLike] = None,   # 基准收益率
                           freq: tp.Optional[tp.FrequencyLike] = None,         # 时间频率
                           year_freq: tp.Optional[tp.FrequencyLike] = None,    # 年化频率
                           use_asset_returns: bool = False,                    # 是否使用资产收益率
                           _source_name: str = source_name,                    # 内部参数：源方法名
                           **kwargs) -> tp.Any:                               # 其他参数
                """
                动态生成的方法实现
                
                该方法作为ReturnsAccessor方法的代理，负责：
                1. 获取Portfolio的ReturnsAccessor实例
                2. 调用对应的源方法
                3. 返回计算结果
                
                参数说明与Portfolio.get_returns_acc方法一致
                """
                # 获取Portfolio的ReturnsAccessor实例
                # 这里使用了Portfolio类的get_returns_acc方法
                returns_acc = self.get_returns_acc(
                    group_by=group_by,                    # 传递分组参数
                    benchmark_rets=benchmark_rets,        # 传递基准收益率
                    freq=freq,                            # 传递频率参数
                    year_freq=year_freq,                  # 传递年化频率
                    use_asset_returns=use_asset_returns   # 传递资产收益率标志
                )
                # 使用反射机制调用ReturnsAccessor的对应方法
                # getattr动态获取方法，然后调用并传递额外参数
                return getattr(returns_acc, _source_name)(**kwargs)

            # 设置新方法的元数据
            new_method.__name__ = target_name                           # 设置方法名
            new_method.__qualname__ = f"{cls.__name__}.{target_name}"   # 设置限定名
            new_method.__doc__ = docstring                              # 设置文档字符串
            
            # 将新方法绑定到目标类上，并应用缓存装饰器
            # cached_method装饰器提供了方法级别的缓存功能
            setattr(cls, target_name, cached_method(new_method))
        
        # 返回装饰后的类
        return cls

    # 返回包装器函数，完成装饰器的定义
    return wrapper
