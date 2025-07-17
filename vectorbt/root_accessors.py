# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
VectorBT 根级 pandas 访问器模块

本模块是 vectorbt 量化分析框架的核心入口点，负责在 pandas 的 Series 和 DataFrame 对象上
注册和管理自定义的 `.vbt` 访问器。通过这个模块，用户可以直接在 pandas 对象上调用 vectorbt
的所有功能。

设计逻辑：
1. **访问器架构**：使用 pandas 的访问器模式，通过 `pd.Series.vbt.*` 和 `pd.DataFrame.vbt.*` 
   提供统一的访问接口，使得 vectorbt 的功能能够与 pandas 无缝集成
2. **分层设计**：构建了一个分层的访问器继承体系，根访问器作为所有其他专用访问器的基础
3. **动态注册**：提供了灵活的访问器注册机制，允许在运行时动态添加新的访问器
4. **非缓存设计**：与标准的 pandas 访问器不同，本模块的访问器不使用缓存，确保每次访问都获得最新数据
5. **模块化扩展**：为专门的访问器（如 signals、returns、ohlcv 等）提供了可扩展的基础架构

访问器继承层次结构：
```
vbt.base.accessors.BaseSR/DFAccessor           -> pd.Series/DataFrame.vbt.*
vbt.generic.accessors.GenericSR/DFAccessor     -> pd.Series/DataFrame.vbt.*
vbt.signals.accessors.SignalsSR/DFAccessor     -> pd.Series/DataFrame.vbt.signals.*
vbt.returns.accessors.ReturnsSR/DFAccessor     -> pd.Series/DataFrame.vbt.returns.*
vbt.ohlcv.accessors.OHLCVDFAccessor            -> pd.DataFrame.vbt.ohlc.* and pd.DataFrame.vbt.ohlcv.*
vbt.px_accessors.PXAccessor                    -> pd.DataFrame.vbt.px.*
```

主要功能：
- 在 pandas 对象上注册 vbt 访问器
- 提供访问器的动态注册和管理功能
- 建立访问器之间的继承关系
- 确保访问器的非缓存特性以保证数据一致性

使用示例：
```python
import pandas as pd
import vectorbt as vbt
import numpy as np

# 创建示例数据
data = pd.Series([100, 105, 98, 95, 102, 108, 103])
df = pd.DataFrame({
    'price': [100, 105, 98, 95, 102, 108, 103],
    'volume': [1000, 1200, 800, 600, 1100, 1500, 900]
})

# 通过 vbt 访问器使用 vectorbt 功能
# 1. 基本统计分析
stats = data.vbt.describe()
print("基本统计信息:", stats)

# 2. 滚动窗口计算
rolling_mean = data.vbt.rolling_mean(window=3)
print("3期滚动均值:", rolling_mean)

# 3. 专用访问器使用
# 信号分析访问器
signals = (data > data.shift(1)).vbt.signals
print("上涨信号:", signals.sum())

# 收益率分析访问器
returns = data.pct_change().vbt.returns
print("收益率统计:", returns.describe())

# 4. 绘图功能
fig = data.vbt.plot()
fig.show()
```

注意事项：
- 本模块的访问器不使用缓存，每次访问都会创建新的访问器实例
- 访问器支持链式调用，如 `df.vbt.signals.plot()`
- 所有访问器都继承了 DirNamesMixin，支持在 IPython/Jupyter 中的自动补全
"""

# 导入警告模块，用于在访问器注册时发出警告
import warnings

# 导入 pandas 库和相关类
import pandas as pd
from pandas.core.accessor import DirNamesMixin  # pandas 访问器基类，支持目录名管理

# 导入 vectorbt 内部模块
from vectorbt import _typing as tp  # 类型提示模块
from vectorbt.generic.accessors import GenericSRAccessor, GenericDFAccessor  # 通用访问器类
from vectorbt.utils.config import Configured  # 配置管理类

# 定义泛型类型变量，用于类型提示
ParentAccessorT = tp.TypeVar("ParentAccessorT", bound=object)  # 父访问器类型变量
AccessorT = tp.TypeVar("AccessorT", bound=object)  # 访问器类型变量


class Accessor:
    """
    自定义属性访问器类
    
    这是一个类似于属性的对象，用于在 pandas 对象上动态创建访问器实例。
    与标准的 pandas 访问器不同，此访问器不使用缓存机制，确保每次访问都获得最新的数据。
    
    设计原理：
    - 使用描述符协议 (__get__ 方法) 实现类似属性的行为
    - 每次访问都创建新的访问器实例，避免使用过时的数据
    - 支持多种对象类型（pd.Series、pd.DataFrame、Configured等）
    
    参数：
        name (str): 访问器的名称，用于标识和注册
        accessor (tp.Type[AccessorT]): 访问器类，将被实例化的类型
    
    使用示例：
    ```python
    # 内部使用示例（通常不直接使用）
    accessor = Accessor("my_accessor", MyAccessorClass)
    
    # 当访问 obj.my_accessor 时，会调用 __get__ 方法
    # 返回 MyAccessorClass(obj) 的实例
    ```
    
    注意：
        与其他 pandas 访问器不同，此访问器不使用缓存！
        这防止了在对象原地修改后使用旧数据的问题。
    """

    def __init__(self, name: str, accessor: tp.Type[AccessorT]) -> None:
        """
        初始化访问器
        
        参数：
            name: 访问器名称
            accessor: 访问器类
        """
        self._name = name  # 存储访问器名称
        self._accessor = accessor  # 存储访问器类

    def __get__(self, obj: ParentAccessorT, cls: DirNamesMixin) -> AccessorT:
        """
        描述符协议的 __get__ 方法，在访问访问器时被调用
        
        此方法实现了描述符协议，当通过对象访问此属性时会被自动调用。
        根据不同的对象类型创建相应的访问器实例。
        
        参数：
            obj: 被访问的对象实例（如 pd.Series、pd.DataFrame 等）
            cls: 对象的类（实现 DirNamesMixin 的类）
        
        返回：
            AccessorT: 访问器实例
        
        处理逻辑：
        1. 如果 obj 为 None（类级别访问），返回访问器类本身
        2. 如果 obj 是 pandas 对象，直接使用 obj 创建访问器实例
        3. 如果 obj 是 Configured 对象，使用 replace 方法创建实例
        4. 其他情况，使用 obj.obj 创建访问器实例
        """
        if obj is None:  # 类级别访问，返回访问器类本身
            return self._accessor
        if isinstance(obj, (pd.Series, pd.DataFrame)):  # pandas 对象
            accessor_obj = self._accessor(obj)  # 直接创建访问器实例
        elif isinstance(obj, Configured):  # 配置对象
            accessor_obj = obj.replace(cls_=self._accessor)  # 使用配置替换方法
        else:  # 其他对象类型
            accessor_obj = self._accessor(obj.obj)  # 使用 obj.obj 创建实例
        return accessor_obj


def register_accessor(name: str, cls: tp.Type[DirNamesMixin]) -> tp.Callable:
    """
    注册自定义访问器的通用函数
    
    这是一个装饰器工厂函数，用于在指定的类上注册自定义访问器。
    它会检查名称冲突并发出警告，然后将访问器添加到类的属性中。
    
    参数：
        name (str): 访问器的名称，将作为属性名添加到类中
        cls (tp.Type[DirNamesMixin]): 目标类，访问器将被添加到此类上
    
    返回：
        tp.Callable: 装饰器函数，用于装饰访问器类
    
    使用示例：
    ```python
    # 在 pd.Series 上注册名为 'my_accessor' 的访问器
    @register_accessor('my_accessor', pd.Series)
    class MySeriesAccessor:
        def __init__(self, obj):
            self._obj = obj
        
        def my_method(self):
            return self._obj.sum()
    
    # 使用注册的访问器
    series = pd.Series([1, 2, 3, 4, 5])
    result = series.my_accessor.my_method()  # 返回 15
    ```
    
    注意：
        cls 应该是 pandas.core.accessor.DirNamesMixin 的子类
    """

    def decorator(accessor: tp.Type[AccessorT]) -> tp.Type[AccessorT]:
        """
        装饰器函数，实际执行访问器注册的逻辑
        
        参数：
            accessor: 要注册的访问器类
        
        返回：
            访问器类本身（用于链式调用）
        """
        if hasattr(cls, name):  # 检查是否存在同名属性
            warnings.warn(  # 发出警告
                f"registration of accessor {repr(accessor)} under name "
                f"{repr(name)} for type {repr(cls)} is overriding a preexisting "
                f"attribute with the same name.",
                UserWarning,  # 用户警告类型
                stacklevel=2,  # 堆栈级别，指向调用者
            )
        setattr(cls, name, Accessor(name, accessor))  # 设置访问器属性
        cls._accessors.add(name)  # 将访问器名称添加到类的访问器集合中
        return accessor  # 返回访问器类

    return decorator  # 返回装饰器函数


def register_series_accessor(name: str) -> tp.Callable:
    """
    注册 pd.Series 自定义访问器的装饰器函数
    
    这是一个便捷函数，专门用于在 pd.Series 上注册自定义访问器。
    它是 register_accessor 的一个特化版本。
    
    参数：
        name (str): 访问器名称，访问时使用 series.{name} 的形式
    
    返回：
        tp.Callable: 装饰器函数
    
    使用示例：
    ```python
    @register_series_accessor('my_stats')
    class MySeriesStatsAccessor:
        def __init__(self, obj):
            self._obj = obj
        
        def moving_average(self, window=5):
            '''计算移动平均'''
            return self._obj.rolling(window=window).mean()
        
        def volatility(self, window=20):
            '''计算波动率'''
            return self._obj.rolling(window=window).std()
    
    # 使用示例
    prices = pd.Series([100, 105, 98, 95, 102, 108, 103, 99, 104, 110])
    ma5 = prices.my_stats.moving_average(window=5)  # 5期移动平均
    vol = prices.my_stats.volatility(window=5)       # 5期波动率
    ```
    """
    return register_accessor(name, pd.Series)  # 调用通用注册函数，指定 pd.Series 作为目标类


def register_dataframe_accessor(name: str) -> tp.Callable:
    """
    注册 pd.DataFrame 自定义访问器的装饰器函数
    
    这是一个便捷函数，专门用于在 pd.DataFrame 上注册自定义访问器。
    它是 register_accessor 的一个特化版本。
    
    参数：
        name (str): 访问器名称，访问时使用 dataframe.{name} 的形式
    
    返回：
        tp.Callable: 装饰器函数
    
    使用示例：
    ```python
    @register_dataframe_accessor('my_finance')
    class MyFinanceAccessor:
        def __init__(self, obj):
            self._obj = obj
        
        def sharpe_ratio(self, risk_free_rate=0.02):
            '''计算夏普比率'''
            returns = self._obj.pct_change().dropna()
            excess_returns = returns.mean() - risk_free_rate
            return excess_returns / returns.std()
        
        def max_drawdown(self):
            '''计算最大回撤'''
            cumulative = (1 + self._obj.pct_change()).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            return drawdown.min()
    
    # 使用示例
    df = pd.DataFrame({
        'stock_a': [100, 105, 98, 95, 102, 108, 103],
        'stock_b': [50, 52, 49, 47, 51, 54, 53]
    })
    sharpe = df.my_finance.sharpe_ratio()        # 计算夏普比率
    max_dd = df.my_finance.max_drawdown()        # 计算最大回撤
    ```
    """
    return register_accessor(name, pd.DataFrame)  # 调用通用注册函数，指定 pd.DataFrame 作为目标类


# 通过继承 DirNamesMixin，我们可以在彼此之上构建访问器
@register_series_accessor("vbt")  # 注册 "vbt" 访问器到 pd.Series
class Vbt_SRAccessor(DirNamesMixin, GenericSRAccessor):
    """
    pd.Series 的主要 vectorbt 访问器类
    
    这是 vectorbt 框架为 pd.Series 提供的主要访问器，它继承了 DirNamesMixin 和 GenericSRAccessor，
    为 Series 对象提供了全面的量化分析功能。
    
    继承关系：
    - DirNamesMixin: 提供目录名管理功能，支持动态属性访问和 IPython 自动补全
    - GenericSRAccessor: 提供通用的 Series 分析功能，包括统计、绘图、变换等
    
    主要功能模块：
    1. **统计分析**：describe、value_counts、各种统计函数
    2. **时间序列分析**：滚动窗口、扩展窗口、指数加权移动平均
    3. **数据变换**：填充、差分、标准化、映射等
    4. **绘图可视化**：线图、直方图、箱线图等
    5. **分组聚合**：groupby、resample、reduce 等操作
    6. **专用访问器**：signals、returns 等专门的分析工具
    
    参数：
        obj (tp.Series): 要分析的 pandas Series 对象
        **kwargs: 传递给父类构造函数的其他参数
    
    使用示例：
    ```python
    import pandas as pd
    import numpy as np
    
    # 创建示例时间序列数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates, name='price')
    
    # 基本统计分析
    stats = prices.vbt.describe()
    print("基本统计信息:", stats)
    
    # 滚动窗口分析
    ma_5 = prices.vbt.rolling_mean(window=5)      # 5日移动平均
    ma_20 = prices.vbt.rolling_mean(window=20)    # 20日移动平均
    volatility = prices.vbt.rolling_std(window=20)  # 20日波动率
    
    # 数据变换
    returns = prices.vbt.pct_change()             # 收益率
    normalized = prices.vbt.normalize()           # 标准化
    filled = prices.vbt.fillna(method='ffill')    # 填充缺失值
    
    # 绘图可视化
    fig = prices.vbt.plot(title='价格走势')
    fig.show()
    
    # 使用专用访问器
    # 信号分析（需要布尔序列）
    signals = (prices > ma_20).vbt.signals
    print("信号统计:", signals.describe())
    
    # 收益率分析
    returns_analysis = returns.vbt.returns
    print("收益率统计:", returns_analysis.describe())
    ```
    
    注意：
    - 访问器不使用缓存，每次访问都会创建新实例
    - 支持链式调用，如 prices.vbt.returns.sharpe_ratio()
    - 所有方法都针对量化分析场景进行了优化
    """

    def __init__(self, obj: tp.Series, **kwargs) -> None:
        """
        初始化 Series 访问器
        
        参数：
            obj: pandas Series 对象
            **kwargs: 传递给父类的其他参数
        """
        self._obj = obj  # 存储 Series 对象的引用

        # 初始化父类
        DirNamesMixin.__init__(self)  # 初始化目录名管理功能
        GenericSRAccessor.__init__(self, obj, **kwargs)  # 初始化通用 Series 访问器


@register_dataframe_accessor("vbt")  # 注册 "vbt" 访问器到 pd.DataFrame
class Vbt_DFAccessor(DirNamesMixin, GenericDFAccessor):
    """
    pd.DataFrame 的主要 vectorbt 访问器类
    
    这是 vectorbt 框架为 pd.DataFrame 提供的主要访问器，它继承了 DirNamesMixin 和 GenericDFAccessor，
    为 DataFrame 对象提供了全面的量化分析功能，特别适用于多资产组合分析。
    
    继承关系：
    - DirNamesMixin: 提供目录名管理功能，支持动态属性访问和 IPython 自动补全
    - GenericDFAccessor: 提供通用的 DataFrame 分析功能，包括统计、绘图、变换等
    
    主要功能模块：
    1. **多资产统计分析**：describe、correlation、covariance 等
    2. **时间序列分析**：滚动窗口、扩展窗口、指数加权移动平均
    3. **数据变换**：填充、差分、标准化、映射等
    4. **绘图可视化**：热力图、线图、箱线图等
    5. **分组聚合**：groupby、resample、reduce 等操作
    6. **专用访问器**：ohlcv、px、signals、returns 等
    
    参数：
        obj (tp.Frame): 要分析的 pandas DataFrame 对象
        **kwargs: 传递给父类构造函数的其他参数
    
    使用示例：
    ```python
    import pandas as pd
    import numpy as np
    
    # 创建示例多资产价格数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = pd.DataFrame({
        'AAPL': np.random.randn(100).cumsum() + 150,
        'GOOGL': np.random.randn(100).cumsum() + 2500,
        'MSFT': np.random.randn(100).cumsum() + 300,
        'TSLA': np.random.randn(100).cumsum() + 200
    }, index=dates)
    
    # 基本统计分析
    stats = prices.vbt.describe()
    print("多资产统计信息:", stats)
    
    # 相关性分析
    correlation = prices.vbt.corr()
    print("相关性矩阵:", correlation)
    
    # 滚动窗口分析
    ma_20 = prices.vbt.rolling_mean(window=20)    # 20日移动平均
    volatility = prices.vbt.rolling_std(window=20)  # 20日波动率
    
    # 数据变换
    returns = prices.vbt.pct_change()             # 收益率矩阵
    normalized = prices.vbt.normalize()           # 标准化处理
    
    # 绘图可视化
    fig = prices.vbt.plot(title='多资产价格走势')
    fig.show()
    
    # 热力图
    heatmap = prices.vbt.heatmap(title='价格热力图')
    heatmap.show()
    
    # 使用专用访问器
    # OHLCV 数据分析（如果有 OHLCV 格式数据）
    if 'Open' in prices.columns:
        ohlcv = prices.vbt.ohlcv
        print("OHLCV 统计:", ohlcv.describe())
    
    # 收益率分析
    returns_analysis = returns.vbt.returns
    print("收益率统计:", returns_analysis.describe())
    
    # 组合分析
    portfolio_returns = returns.mean(axis=1)  # 等权重组合
    portfolio_stats = portfolio_returns.vbt.returns.describe()
    print("组合统计:", portfolio_stats)
    ```
    
    注意：
    - 访问器不使用缓存，每次访问都会创建新实例
    - 支持链式调用，如 prices.vbt.returns.sharpe_ratio()
    - 特别适用于多资产和投资组合分析
    - 所有方法都针对量化分析场景进行了优化
    """

    def __init__(self, obj: tp.Frame, **kwargs) -> None:
        """
        初始化 DataFrame 访问器
        
        参数：
            obj: pandas DataFrame 对象
            **kwargs: 传递给父类的其他参数
        """
        self._obj = obj  # 存储 DataFrame 对象的引用

        # 初始化父类
        DirNamesMixin.__init__(self)  # 初始化目录名管理功能
        GenericDFAccessor.__init__(self, obj, **kwargs)  # 初始化通用 DataFrame 访问器


def register_series_vbt_accessor(name: str, parent: tp.Type[DirNamesMixin] = Vbt_SRAccessor) -> tp.Callable:
    """
    在父访问器之上注册 pd.Series 访问器的装饰器函数
    
    这个函数允许在现有的访问器（默认是 Vbt_SRAccessor）之上构建新的访问器，
    形成访问器的层次结构。这样可以创建专门的访问器，如 signals、returns 等。
    
    参数：
        name (str): 新访问器的名称
        parent (tp.Type[DirNamesMixin]): 父访问器类，默认为 Vbt_SRAccessor
    
    返回：
        tp.Callable: 装饰器函数
    
    使用示例：
    ```python
    @register_series_vbt_accessor('signals')
    class SignalsAccessor:
        def __init__(self, obj):
            self._obj = obj
        
        def count_signals(self):
            '''计算信号数量'''
            return self._obj.sum()
        
        def signal_duration(self):
            '''计算信号持续时间'''
            # 计算连续信号的持续时间
            groups = (self._obj != self._obj.shift()).cumsum()
            return self._obj.groupby(groups).size()
    
    # 使用示例
    prices = pd.Series([100, 105, 98, 95, 102, 108, 103])
    signals = prices > prices.shift(1)  # 上涨信号
    
    # 通过 vbt 访问器使用 signals 访问器
    count = signals.vbt.signals.count_signals()      # 信号数量
    duration = signals.vbt.signals.signal_duration()  # 信号持续时间
    ```
    
    访问器层次结构：
    ```
    pd.Series.vbt                    # 主访问器
    pd.Series.vbt.signals           # 信号分析访问器
    pd.Series.vbt.returns           # 收益率分析访问器
    pd.Series.vbt.{custom_name}     # 自定义访问器
    ```
    """
    return register_accessor(name, parent)  # 调用通用注册函数，指定父访问器作为目标类


def register_dataframe_vbt_accessor(name: str, parent: tp.Type[DirNamesMixin] = Vbt_DFAccessor) -> tp.Callable:
    """
    在父访问器之上注册 pd.DataFrame 访问器的装饰器函数
    
    这个函数允许在现有的访问器（默认是 Vbt_DFAccessor）之上构建新的访问器，
    形成访问器的层次结构。这样可以创建专门的访问器，如 ohlcv、px、signals 等。
    
    参数：
        name (str): 新访问器的名称
        parent (tp.Type[DirNamesMixin]): 父访问器类，默认为 Vbt_DFAccessor
    
    返回：
        tp.Callable: 装饰器函数
    
    使用示例：
    ```python
    @register_dataframe_vbt_accessor('portfolio')
    class PortfolioAccessor:
        def __init__(self, obj):
            self._obj = obj
        
        def equal_weight_returns(self):
            '''计算等权重组合收益率'''
            returns = self._obj.pct_change().dropna()
            return returns.mean(axis=1)
        
        def sharpe_ratio(self, risk_free_rate=0.02):
            '''计算组合夏普比率'''
            portfolio_returns = self.equal_weight_returns()
            excess_returns = portfolio_returns.mean() - risk_free_rate
            return excess_returns / portfolio_returns.std()
        
        def max_drawdown(self):
            '''计算组合最大回撤'''
            portfolio_returns = self.equal_weight_returns()
            cumulative = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            return drawdown.min()
    
    # 使用示例
    prices = pd.DataFrame({
        'AAPL': [100, 105, 98, 95, 102, 108, 103],
        'GOOGL': [2500, 2520, 2480, 2450, 2510, 2580, 2550],
        'MSFT': [300, 310, 295, 290, 305, 320, 315]
    })
    
    # 通过 vbt 访问器使用 portfolio 访问器
    portfolio_returns = prices.vbt.portfolio.equal_weight_returns()
    sharpe = prices.vbt.portfolio.sharpe_ratio()
    max_dd = prices.vbt.portfolio.max_drawdown()
    ```
    
    访问器层次结构：
    ```
    pd.DataFrame.vbt                    # 主访问器
    pd.DataFrame.vbt.ohlcv             # OHLCV 数据访问器
    pd.DataFrame.vbt.px                # 价格数据访问器
    pd.DataFrame.vbt.signals           # 信号分析访问器
    pd.DataFrame.vbt.returns           # 收益率分析访问器
    pd.DataFrame.vbt.{custom_name}     # 自定义访问器
    ```
    """
    return register_accessor(name, parent)  # 调用通用注册函数，指定父访问器作为目标类
