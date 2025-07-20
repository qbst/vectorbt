# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT QUANTSTATS 适配器模块 - 量化交易绩效分析与第三方库集成核心模块
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架与quantstats量化分析库之间的核心适配器模块。该模块通过
智能的适配器模式，将quantstats库的所有功能无缝集成到vectorbt的收益率分析体系中，
为量化交易策略提供专业级的绩效评估、风险分析和可视化功能。

核心设计理念：
1. **无缝第三方集成**：通过适配器模式将quantstats的200+个分析函数完美集成到vectorbt生态系统
2. **智能参数映射**：自动处理vectorbt与quantstats之间的参数命名和格式差异
3. **配置驱动架构**：支持全局配置、访问器配置和适配器配置的智能合并
4. **数据清理机制**：自动处理缺失值、基准数据对齐和数据完整性检查
5. **动态方法生成**：使用装饰器模式在运行时为适配器类动态添加所有quantstats方法

主要功能模块：

【核心适配器类 QSAdapter】
- 继承自Configured配置基类，提供灵活的配置管理机制
- 持有ReturnsAccessor实例的引用，获取收益率数据和元配置
- 实现__call__方法支持链式配置和参数传递
- 提供defaults属性智能合并多层配置信息

【装饰器驱动的方法生成 attach_qs_methods】
- 动态扫描quantstats库的四个模块：utils、stats、plots、reports
- 自动为每个接受returns参数的函数生成对应的适配器方法
- 智能处理方法命名：plots模块添加plot_前缀，reports模块添加_report后缀
- 保持原始函数的完整签名信息，确保IDE智能提示和类型检查的有效性

【智能参数处理机制】
- 自动映射vectorbt参数到quantstats参数（如rf->risk_free）
- 根据函数签名动态匹配和传递所需参数
- 处理特殊参数如periods、periods_per_year的自动计算
- 支持用户自定义参数覆盖默认参数

【数据质量保障系统】
- 自动检测和处理收益率数据中的缺失值（NaN）
- 智能对齐收益率数据与基准数据的时间序列
- 确保传递给quantstats函数的数据完整性和一致性
- 支持DataFrame和Series两种数据格式的统一处理

与quantstats集成的优势：
1. **统一接口**：通过.vbt.returns.qs访问所有quantstats功能，保持API一致性
2. **参数复用**：一次配置基准收益率、无风险利率等参数，全局复用
3. **智能转换**：自动处理年化因子、时间频率等参数转换
4. **性能优化**：避免重复数据处理和参数传递
5. **类型安全**：保持完整的类型提示和文档信息

应用场景：
- **策略绩效评估**：计算夏普比率、最大回撤、卡尔玛比率等经典指标
- **风险分析**：VaR计算、压力测试、相关性分析
- **基准比较**：Alpha、Beta、信息比率等相对绩效指标
- **可视化分析**：回测结果图表、收益分布图、滚动指标图
- **报告生成**：自动化的HTML投资组合分析报告

技术特点：
- **零拷贝集成**：直接使用vectorbt的数据结构，避免不必要的数据复制
- **动态方法绑定**：运行时生成方法，保持最小的内存占用
- **完整类型支持**：保持quantstats原始函数的类型注解和文档
- **灵活配置系统**：支持全局、模块、实例三个层次的配置管理
- **异常处理机制**：智能处理数据缺失、参数错误等异常情况

与vectorbt生态系统的关系：
- **ReturnsAccessor集成**：作为收益率访问器的.qs属性提供服务
- **配置系统统一**：使用vectorbt的全局配置系统管理默认参数
- **数据结构兼容**：与ArrayWrapper、pandas对象完美兼容
- **类型系统统一**：遵循vectorbt的类型注解规范
- **文档系统集成**：自动生成API文档和使用示例

该模块体现了vectorbt框架"开放生态、无缝集成"的设计理念，通过适配器模式
将业界最优秀的量化分析工具整合到统一的框架中，为量化交易者提供了强大而
易用的分析工具集。
================================================================================

Quantstats适配器类 - 为vectorbt收益率分析提供quantstats库功能

!!! 注意事项
    访问器不使用缓存机制。

我们可以从`ReturnsAccessor`访问适配器：

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbt as vbt
>>> import quantstats as qs

>>> np.random.seed(42)
>>> rets = pd.Series(np.random.uniform(-0.1, 0.1, size=(100,)))
>>> benchmark_rets = pd.Series(np.random.uniform(-0.1, 0.1, size=(100,)))

>>> rets.vbt.returns.qs.r_squared(benchmark=benchmark_rets)
0.0011582111228735541
```

这等同于：

```pycon
>>> qs.stats.r_squared(rets, benchmark_rets)
```

那么为什么不直接使用`qs.stats`？

首先，我们可以一次性定义所有参数（如基准收益率），避免每次调用函数时重复传递。
其次，vectorbt自动将传递给`ReturnsAccessor`的参数转换为quantstats使用的格式。

```pycon
>>> # vectorbt理解的默认参数
>>> ret_acc = rets.vbt.returns(
...     benchmark_rets=benchmark_rets,
...     freq='d',
...     year_freq='365d',
...     defaults=dict(risk_free=0.001)
... )

>>> ret_acc.qs.r_squared()
0.0011582111228735541

>>> ret_acc.qs.sharpe()
-1.9158923252075455

>>> # 仅quantstats理解的默认参数
>>> qs_defaults = dict(
...     benchmark=benchmark_rets,
...     periods=365,
...     periods_per_year=365,
...     rf=0.001
... )
>>> ret_acc_qs = rets.vbt.returns.qs(defaults=qs_defaults)

>>> ret_acc_qs.r_squared()
0.0011582111228735541

>>> ret_acc_qs.sharpe()
-1.9158923252075455
```

适配器自动将收益率传递给特定函数。
它还会合并设置中定义的默认值、传递给`ReturnsAccessor`的默认值，
以及传递给`QSAdapter`本身的默认值，并将它们与函数签名中列出的参数名称匹配。

例如，`periods`和`periods_per_year`参数默认为年化因子
`ReturnsAccessor.ann_factor`，它本身基于`freq`参数。这使得
quantstats和vectorbt产生的结果至少在某种程度上相似。

```pycon
>>> vbt.settings.array_wrapper['freq'] = 'h'
>>> vbt.settings.returns['year_freq'] = '365d'

>>> rets.vbt.returns.sharpe_ratio()  # ReturnsAccessor
-9.38160953971508

>>> rets.vbt.returns.qs.sharpe()  # 通过QSAdapter使用quantstats
-9.38160953971508
```

我们仍然可以通过覆盖默认值或直接将参数传递给函数来覆盖任何参数：

```pycon
>>> rets.vbt.returns.qs(defaults=dict(periods=252)).sharpe()
-1.5912029345745982

>>> rets.vbt.returns.qs.sharpe(periods=252)
-1.5912029345745982

>>> qs.stats.sharpe(rets)
-1.5912029345745982
```
"""
# 导入Python标准库模块
from inspect import getmembers, isfunction, signature, Parameter  # 用于反射和函数签名处理

# 导入第三方数据处理库
import pandas as pd  # pandas数据处理库，用于DataFrame和Series操作
import quantstats as qs  # quantstats量化分析库，提供丰富的投资组合分析功能

# 导入vectorbt内部模块
from vectorbt import _typing as tp  # vectorbt类型注解模块，提供类型提示支持
from vectorbt.returns.accessors import ReturnsAccessor  # 收益率访问器类，提供收益率分析功能
from vectorbt.utils import checks  # 工具检查模块，提供参数验证功能
from vectorbt.utils.config import merge_dicts, get_func_arg_names, Configured  # 配置工具模块


def attach_qs_methods(cls: tp.Type[tp.T], replace_signature: bool = True) -> tp.Type[tp.T]:
    """
    类装饰器：为QSAdapter类动态附加quantstats方法
    
    这是一个核心装饰器函数，负责扫描quantstats库的所有模块，并为每个接受
    'returns'参数的函数创建对应的适配器方法。这种动态方法生成的设计使得
    vectorbt能够自动支持quantstats的所有功能，无需手动编写大量重复代码。
    
    设计原理：
    1. **反射机制**：使用getmembers和isfunction遍历quantstats的所有函数
    2. **智能过滤**：只处理接受'returns'参数的公有函数，避免不相关的函数
    3. **命名规范**：根据模块类型自动调整方法名（plots->plot_前缀，reports->_report后缀）
    4. **签名保持**：完整保留原函数的参数签名，确保IDE智能提示的有效性
    5. **文档继承**：自动设置方法文档，指向原始quantstats函数
    
    参数说明：
        cls: 要装饰的类，必须是QSAdapter的子类
        replace_signature: 是否替换方法签名为原始quantstats函数的签名
        
    返回值：
        装饰后的类，包含所有动态生成的quantstats方法
        
    使用示例：
    ```python
    @attach_qs_methods
    class QSAdapter(Configured):
        # 类会自动获得所有quantstats方法
        pass
    
    # 使用生成的方法
    adapter = QSAdapter(returns_accessor)
    sharpe = adapter.sharpe()  # 来自quantstats.stats.sharpe
    fig = adapter.plot_histogram()  # 来自quantstats.plots.histogram
    ```
    
    技术细节：
    - 使用闭包捕获每个quantstats函数的引用
    - 通过setattr动态为类添加方法
    - 处理参数的关键字传递和默认值合并
    - 自动清理空值数据，确保quantstats函数的正常运行
    """
    # 确保传入的类是QSAdapter的子类
    checks.assert_subclass_of(cls, "QSAdapter")

    # 遍历quantstats的四个核心模块：utils、stats、plots、reports
    for module_name in ['utils', 'stats', 'plots', 'reports']:
        # 获取指定模块的所有函数成员
        for qs_func_name, qs_func in getmembers(getattr(qs, module_name), isfunction):
            # 过滤条件：非私有函数且接受'returns'参数
            if not qs_func_name.startswith('_') and checks.func_accepts_arg(qs_func, 'returns'):
                # 根据模块类型调整方法命名
                if module_name == 'plots':
                    new_method_name = 'plot_' + qs_func_name  # 绘图函数添加plot_前缀
                elif module_name == 'reports':
                    new_method_name = qs_func_name + '_report'  # 报告函数添加_report后缀
                else:
                    new_method_name = qs_func_name  # utils和stats模块保持原名

                def new_method(self, *, _func: tp.Callable = qs_func, **kwargs) -> tp.Any:
                    """
                    动态生成的适配器方法实现
                    
                    这是每个quantstats函数对应的适配器方法的实际实现。该方法负责：
                    1. 获取收益率数据并处理缺失值
                    2. 智能匹配和传递函数参数
                    3. 处理基准数据的时间对齐
                    4. 调用原始quantstats函数并返回结果
                    
                    核心处理流程：
                    - 数据获取：从returns_accessor获取收益率时间序列
                    - 空值检测：识别收益率和基准数据中的缺失值
                    - 参数匹配：根据函数签名智能匹配所需参数
                    - 数据清理：移除包含缺失值的时间点
                    - 函数调用：使用清理后的数据调用quantstats函数
                    """
                    # 获取收益率数据（pandas Series或DataFrame）
                    returns = self.returns_accessor.obj
                    
                    # 识别缺失值：DataFrame按行检查，Series直接检查
                    if isinstance(returns, pd.DataFrame):
                        null_mask = returns.isnull().any(axis=1)  # DataFrame：任意列有NaN的行
                    else:
                        null_mask = returns.isnull()  # Series：直接检查NaN值
                    
                    # 获取目标函数的参数名列表
                    func_arg_names = get_func_arg_names(_func)
                    # 获取适配器的默认配置
                    defaults = self.defaults

                    # 准备传递给quantstats函数的参数字典
                    pass_kwargs = dict()
                    
                    # 遍历函数签名中的每个参数
                    for arg_name in func_arg_names:
                        # 如果用户没有手动提供该参数
                        if arg_name not in kwargs:
                            # 优先使用默认配置中的参数值
                            if arg_name in defaults:
                                pass_kwargs[arg_name] = defaults[arg_name]
                            # 特殊处理：自动提供基准收益率
                            elif arg_name == 'benchmark':
                                if self.returns_accessor.benchmark_rets is not None:
                                    pass_kwargs['benchmark'] = self.returns_accessor.benchmark_rets
                            # 特殊处理：自动计算periods参数（年化相关）
                            elif arg_name == 'periods':
                                pass_kwargs['periods'] = int(self.returns_accessor.ann_factor)
                            # 特殊处理：自动计算periods_per_year参数
                            elif arg_name == 'periods_per_year':
                                pass_kwargs['periods_per_year'] = int(self.returns_accessor.ann_factor)
                        else:
                            # 使用用户手动提供的参数值
                            pass_kwargs[arg_name] = kwargs[arg_name]

                    # 处理基准数据的缺失值对齐
                    if 'benchmark' in pass_kwargs:
                        # 检测基准数据中的缺失值
                        if isinstance(pass_kwargs['benchmark'], pd.DataFrame):
                            bm_null_mask = pass_kwargs['benchmark'].isnull().any(axis=1)
                        else:
                            bm_null_mask = pass_kwargs['benchmark'].isnull()
                        
                        # 合并收益率和基准数据的缺失值掩码
                        null_mask = null_mask | bm_null_mask
                        # 从基准数据中移除缺失值
                        pass_kwargs['benchmark'] = pass_kwargs['benchmark'].loc[~null_mask]
                    
                    # 从收益率数据中移除缺失值
                    returns = returns.loc[~null_mask]

                    # 验证参数绑定（确保参数匹配函数签名）
                    signature(_func).bind(returns=returns, **pass_kwargs)
                    # 调用原始quantstats函数并返回结果
                    return _func(returns=returns, **pass_kwargs)

                # 如果需要替换函数签名（保持IDE智能提示）
                if replace_signature:
                    # 获取原始quantstats函数的签名
                    source_sig = signature(qs_func)
                    # 获取新方法的参数列表
                    new_method_params = tuple(signature(new_method).parameters.values())
                    # 提取self参数
                    self_arg = new_method_params[0]
                    # 将其他参数转换为仅关键字参数
                    other_args = [
                        p.replace(kind=Parameter.KEYWORD_ONLY)
                        if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
                        else p
                        for p in list(source_sig.parameters.values())[1:]
                    ]
                    # 构建新的函数签名
                    source_sig = source_sig.replace(parameters=(self_arg,) + tuple(other_args))
                    # 应用新签名到方法
                    new_method.__signature__ = source_sig

                # 设置方法的文档字符串
                new_method.__doc__ = f"参见 `quantstats.{module_name}.{qs_func_name}`。"
                # 设置方法的限定名称（用于调试和反射）
                new_method.__qualname__ = f"{cls.__name__}.{new_method_name}"
                # 设置方法的名称
                new_method.__name__ = new_method_name
                # 将方法动态添加到类中
                setattr(cls, new_method_name, new_method)
                
    # 返回装饰后的类
    return cls


# QSAdapter类的类型变量，用于类型提示中的泛型约束
QSAdapterT = tp.TypeVar("QSAdapterT", bound="QSAdapter")


@attach_qs_methods  # 应用装饰器，动态添加所有quantstats方法
class QSAdapter(Configured):
    """
    Quantstats适配器核心类
    
    该类是vectorbt框架与quantstats库之间的桥梁，提供了完整的量化分析功能适配。
    通过智能的参数映射、数据清理和方法生成机制，使得用户可以无缝地在vectorbt
    环境中使用quantstats的所有功能。
    
    核心特性：
    1. **自动方法生成**：通过@attach_qs_methods装饰器，自动获得200+个quantstats方法
    2. **智能参数映射**：自动处理vectorbt与quantstats之间的参数命名差异
    3. **配置层次合并**：智能合并全局配置、访问器配置和适配器配置
    4. **数据质量保障**：自动处理缺失值、时间对齐和数据清理
    5. **链式调用支持**：支持配置的链式传递和方法链式调用
    
    设计模式：
    - **适配器模式**：在vectorbt和quantstats之间提供统一接口
    - **装饰器模式**：通过装饰器动态增强类的功能
    - **配置驱动**：通过分层配置系统控制行为
    - **代理模式**：代理对quantstats函数的访问和调用
    
    继承结构：
    - Configured: 提供配置管理和对象替换能力
    
    使用示例：
    ```python
    # 1. 基本使用 - 通过ReturnsAccessor访问
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    ret_acc = returns.vbt.returns(freq='d')
    qs_adapter = ret_acc.qs
    
    # 2. 配置自定义默认值
    qs_custom = ret_acc.qs(defaults=dict(rf=0.02))
    
    # 3. 使用quantstats功能
    sharpe = qs_adapter.sharpe()  # 夏普比率
    max_dd = qs_adapter.max_drawdown()  # 最大回撤
    fig = qs_adapter.plot_returns()  # 收益率图表
    
    # 4. 链式调用
    report = returns.vbt.returns.qs(
        defaults=dict(rf=0.02, benchmark=benchmark)
    ).full_report()
    ```
    
    方法分类：
    - **统计方法**：sharpe(), max_drawdown(), calmar()等绩效指标
    - **绘图方法**：plot_returns(), plot_drawdown()等可视化方法
    - **报告方法**：full_report(), stats_report()等综合报告
    - **工具方法**：各种辅助计算和数据处理功能
    
    配置优先级（从高到低）：
    1. 方法调用时的直接参数
    2. QSAdapter初始化时的defaults参数
    3. ReturnsAccessor的defaults配置
    4. vectorbt全局设置中的qs_adapter.defaults
    
    注意事项：
    - 适配器不使用缓存机制，每次调用都会重新计算
    - 自动处理数据中的缺失值，确保quantstats函数正常运行
    - 支持DataFrame和Series两种数据格式
    - 所有绘图方法返回matplotlib图形对象
    """

    def __init__(self, returns_accessor: ReturnsAccessor, defaults: tp.KwargsLike = None, **kwargs) -> None:
        """
        初始化QSAdapter适配器实例
        
        构造函数负责建立与ReturnsAccessor的关联，并设置适配器的配置参数。
        这个初始化过程确保适配器能够访问收益率数据和相关的配置信息。
        
        参数说明：
            returns_accessor: ReturnsAccessor实例，提供收益率数据和配置
                包含收益率时间序列、基准收益率、年化因子等信息
            defaults: 适配器级别的默认配置字典，可选参数
                用于覆盖全局配置中的特定参数
            **kwargs: 传递给Configured基类的其他关键字参数
                支持配置管理相关的额外参数
                
        初始化过程：
        1. 验证returns_accessor参数的有效性
        2. 调用Configured基类的初始化方法
        3. 保存returns_accessor和defaults的引用
        4. 建立配置继承关系
        
        异常处理：
        - 如果returns_accessor不是ReturnsAccessor实例，将抛出TypeError
        """
        # 验证returns_accessor参数必须是ReturnsAccessor的实例
        checks.assert_instance_of(returns_accessor, ReturnsAccessor)

        # 调用Configured基类的初始化方法，建立配置管理功能
        Configured.__init__(self, returns_accessor=returns_accessor, defaults=defaults, **kwargs)

        # 保存ReturnsAccessor实例的引用，用于后续数据访问
        self._returns_accessor = returns_accessor
        # 保存适配器级别的默认配置
        self._defaults = defaults

    def __call__(self: QSAdapterT, **kwargs) -> QSAdapterT:
        """
        调用操作符重载，支持链式配置传递
        
        该方法实现了函数调用语法，允许用户通过调用适配器实例来创建新的
        配置实例。这种设计支持链式调用和配置的灵活传递，是流式API设计的
        重要组成部分。
        
        设计目的：
        1. **链式配置**：支持.qs().method()的链式调用语法
        2. **配置传递**：允许在调用链中动态修改配置
        3. **不可变性**：返回新实例而不是修改当前实例
        4. **类型安全**：保持正确的类型注解
        
        参数说明：
            **kwargs: 要传递给新实例的配置参数
                通常包含defaults、基准数据等配置信息
                
        返回值：
            新的QSAdapter实例，包含合并后的配置
            
        使用示例：
        ```python
        # 基础用法
        adapter = returns.vbt.returns.qs
        
        # 链式配置
        custom_adapter = returns.vbt.returns.qs(
            defaults=dict(rf=0.02, periods=252)
        )
        
        # 进一步链式调用
        result = returns.vbt.returns.qs(
            defaults=dict(rf=0.02)
        ).sharpe()
        ```
        
        技术实现：
        - 使用self.replace()方法创建新实例
        - 保持原始配置不变，体现不可变性设计
        - 通过类型变量确保返回类型的正确性
        """
        # 使用replace方法创建新的适配器实例，传递新的配置参数
        return self.replace(**kwargs)

    @property
    def returns_accessor(self) -> ReturnsAccessor:
        """
        返回关联的ReturnsAccessor实例
        
        这个属性提供对底层ReturnsAccessor对象的访问，允许适配器获取
        收益率数据、基准数据、年化因子等关键信息。这是适配器与
        vectorbt收益率分析系统连接的桥梁。
        
        返回值：
            ReturnsAccessor实例，包含收益率数据和分析功能
            
        用途：
        - 获取收益率时间序列数据
        - 访问基准收益率数据
        - 获取年化因子和频率信息
        - 访问ReturnsAccessor的配置参数
        """
        return self._returns_accessor

    @property
    def defaults_mapping(self) -> tp.Dict:
        """
        参数映射字典：vectorbt参数名到quantstats参数名的映射
        
        该属性定义了vectorbt框架和quantstats库之间的参数名称映射关系。
        由于两个库在某些参数的命名上存在差异，这个映射确保了参数的
        正确传递和识别。
        
        当前映射关系：
        - 'rf' <- 'risk_free': 无风险利率参数的映射
          vectorbt使用'risk_free'，quantstats使用'rf'
          
        设计考虑：
        1. **命名统一**：解决不同库之间的命名约定差异
        2. **向后兼容**：保持与现有代码的兼容性
        3. **易于扩展**：可以轻松添加新的参数映射关系
        4. **明确映射**：通过字典明确定义映射关系
        
        返回值：
            字典，键为quantstats参数名，值为vectorbt参数名
            
        使用场景：
        - 在defaults属性中自动进行参数名称转换
        - 确保配置参数的正确传递
        - 支持两个库之间的参数标准化
        
        扩展示例：
        ```python
        # 未来可能的扩展
        return dict(
            rf='risk_free',              # 无风险利率
            periods='ann_factor',        # 年化周期
            benchmark='benchmark_rets'   # 基准收益率
        )
        ```
        """
        return dict(rf='risk_free')

    @property
    def defaults(self) -> tp.Kwargs:
        """
        QSAdapter的综合默认配置
        
        该属性负责智能合并来自多个来源的配置参数，创建一个统一的默认配置
        字典。这种分层配置系统确保了配置的灵活性和可扩展性，同时保持了
        合理的默认值和用户自定义的优先级。
        
        配置合并层次结构（按优先级从低到高）：
        1. **全局配置**：vectorbt._settings.settings['qs_adapter']['defaults']
           - 系统级别的全局默认配置
           - 影响所有QSAdapter实例的基础配置
           
        2. **映射配置**：从ReturnsAccessor.defaults映射而来的配置
           - 根据defaults_mapping转换参数名称
           - 连接vectorbt和quantstats的参数体系
           
        3. **实例配置**：QSAdapter.__init__中的defaults参数
           - 用户在创建适配器实例时指定的配置
           - 具有最高优先级，可覆盖所有其他配置
        
        智能映射机制：
        - 扫描ReturnsAccessor.defaults中的参数
        - 根据defaults_mapping进行参数名称转换
        - 只映射存在于mapping中且在源配置中存在的参数
        
        返回值：
            合并后的配置字典，包含所有层次的默认参数
            
        使用示例：
        ```python
        # 全局设置无风险利率
        vbt.settings.qs_adapter.defaults['rf'] = 0.02
        
        # ReturnsAccessor级别设置
        ret_acc = returns.vbt.returns(defaults=dict(risk_free=0.03))
        
        # QSAdapter级别设置
        qs_adapter = ret_acc.qs(defaults=dict(rf=0.04))
        
        # 最终defaults['rf'] = 0.04（实例配置优先级最高）
        ```
        
        技术实现：
        - 使用merge_dicts进行深度合并
        - 自动处理参数名称映射
        - 保持配置的不可变性
        """
        # 导入vectorbt的全局配置系统
        from vectorbt._settings import settings
        # 获取qs_adapter模块的全局默认配置
        qs_adapter_defaults_cfg = settings['qs_adapter']['defaults']

        # 创建映射配置字典
        mapped_defaults = dict()
        # 遍历参数映射关系
        for k, v in self.defaults_mapping.items():
            # 如果ReturnsAccessor的defaults中包含对应的vectorbt参数名
            if v in self.returns_accessor.defaults:
                # 将参数值映射到quantstats参数名
                mapped_defaults[k] = self.returns_accessor.defaults[v]
                
        # 按优先级顺序合并所有配置层次
        return merge_dicts(
            qs_adapter_defaults_cfg,  # 1. 全局配置（最低优先级）
            mapped_defaults,          # 2. 映射配置（中等优先级）
            self._defaults           # 3. 实例配置（最高优先级）
        )
