# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
统计构建器混入模块 (Statistics Builder Mixin Module)

本模块是vectorbt量化分析框架中负责统计指标计算的核心组件，提供了一个强大且灵活的
统计指标构建系统。该系统能够将复杂的量化分析结果转换为标准化的统计指标。

设计理念与架构：
1. 模块化设计：通过指标(metric)的概念，将复杂的统计分析分解为多个独立的计算单元
2. 配置驱动：使用声明式配置定义指标结构，支持动态参数替换和条件计算
3. 混入模式：通过Mixin模式为不同的量化分析类提供统一的统计接口
4. 模板系统：集成模板引擎，支持动态配置和参数化计算
5. 标签过滤：通过标签系统实现指标的条件计算和分组管理

主要功能特性：
- 指标管理：支持多指标计算，自动处理依赖关系和计算顺序
- 动态配置：运行时根据数据特征和用户参数动态调整指标配置
- 标签过滤：使用布尔表达式进行指标的条件计算
- 模板替换：支持配置参数的动态替换和计算
- 聚合处理：智能处理多列数据的聚合和选择
- 缓存优化：内置缓存机制提升重复计算的性能

应用场景：
- 量化策略性能评估：收益率、夏普比率、最大回撤、胜率等核心指标
- 风险管理分析：VaR、CVaR、波动率、相关性等风险指标
- 投资组合分析：资产配置效率、分散化指标、再平衡频率等
- 交易行为分析：交易频率、持仓时间、成交量分布等
- 回测结果评估：信息比率、卡尔马比率、索提诺比率等高级指标

技术实现：
- 基于pandas和numpy构建，确保高性能的数值计算
- 使用元类(Metaclass)模式实现配置的继承和扩展
- 集成vectorbt的数组包装器(ArrayWrapper)系统
- 支持多种数据源：pandas DataFrame/Series、NumPy数组等
- 提供完整的错误处理和警告系统

该模块与vectorbt的其他核心模块紧密集成：
- 与plots_builder模块共享配置管理和过滤机制
- 与array_wrapper模块协作处理数据包装和索引
- 与template模块配合实现动态配置替换
- 与config模块集成提供配置管理功能
"""

import inspect  # 导入inspect模块，用于函数签名检查和代码内省
import string  # 导入string模块，用于字符串模板处理
import warnings  # 导入warnings模块，用于发出运行时警告
from collections import Counter  # 导入Counter类，用于统计指标名称的重复次数

import numpy as np  # 导入numpy，用于数值计算和统计函数
import pandas as pd  # 导入pandas，用于数据结构和数据分析

from vectorbt import _typing as tp  # 导入vectorbt类型定义模块
from vectorbt.base.array_wrapper import Wrapping  # 导入数组包装器基类
from vectorbt.utils import checks  # 导入检查工具模块
from vectorbt.utils.attr_ import get_dict_attr  # 导入属性获取工具函数
from vectorbt.utils.config import Config, merge_dicts, get_func_arg_names  # 导入配置管理工具
from vectorbt.utils.tags import match_tags  # 导入标签匹配工具函数
from vectorbt.utils.template import deep_substitute  # 导入深度模板替换工具函数


class MetaStatsBuilderMixin(type):
    """
    统计构建器混入类的元类
    
    该元类为StatsBuilderMixin提供了一个只读的类属性`metrics`，通过元类机制
    实现了类级别的属性访问控制。这种设计模式确保了指标配置的封装性和一致性。
    
    设计目的：
    - 提供类级别的配置访问：允许通过类名直接访问指标配置
    - 确保配置的只读性：防止意外修改类级别的默认配置
    - 支持继承机制：子类可以继承并扩展父类的指标配置
    
    使用示例：
        >>> # 通过类名访问指标配置
        >>> config = SomeAnalyzer.metrics
        >>> print(config.keys())  # 显示所有可用的指标类型
        
        >>> # 在子类中扩展配置
        >>> class CustomAnalyzer(StatsBuilderMixin):
        ...     _metrics = Config({
        ...         'custom_ratio': {'calc_func': 'calculate_custom_ratio'}
        ...     })
    """

    @property
    def metrics(cls) -> Config:
        """
        获取类支持的指标配置
        
        这是一个类属性装饰器，提供对_metrics配置的只读访问。
        每个继承StatsBuilderMixin的类都可以定义自己的_metrics配置。
        
        Returns:
            Config: 包含所有支持的指标定义的配置对象
                   每个指标包含计算函数、标签、参数等信息
        """
        return cls._metrics


class StatsBuilderMixin(metaclass=MetaStatsBuilderMixin):
    """
    统计构建器混入类
    
    这是vectorbt框架中统计指标计算的核心混入类，为量化分析对象提供统一的统计接口。
    该类实现了一个完整的统计指标构建系统，支持多指标、动态配置、标签过滤等高级功能。
    
    核心特性：
    1. 多指标支持：可以同时计算多个不同类型的统计指标
    2. 配置驱动：通过声明式配置定义指标结构和计算方法
    3. 动态参数：支持运行时参数替换和条件计算
    4. 标签过滤：使用标签系统控制指标的计算条件
    5. 模板系统：支持配置参数的动态替换
    6. 智能聚合：自动处理多列数据的聚合和选择
    
    设计模式：
    - 混入模式(Mixin Pattern)：为不同类型的分析对象提供统计能力
    - 策略模式(Strategy Pattern)：通过calc_func支持多种计算策略
    - 模板方法模式(Template Method)：定义了统计计算的标准流程
    - 配置模式(Configuration Pattern)：使用配置对象管理复杂参数
    
    必要条件：
    - 必须继承自vectorbt.base.array_wrapper.Wrapping类
    - 需要定义_metrics类变量来指定支持的指标类型
    
    使用示例：
        >>> class PortfolioAnalyzer(Wrapping, StatsBuilderMixin):
        ...     _metrics = Config({
        ...         'total_return': {
        ...             'title': 'Total Return',
        ...             'calc_func': 'total_return',
        ...             'tags': ['returns', 'basic']
        ...         },
        ...         'sharpe_ratio': {
        ...             'title': 'Sharpe Ratio',
        ...             'calc_func': 'sharpe_ratio',
        ...             'tags': ['returns', 'risk']
        ...         }
        ...     })
        ...     
        ...     def total_return(self):
        ...         # 计算总收益率的具体实现
        ...         return (self.value[-1] / self.value[0]) - 1
        
        >>> analyzer = PortfolioAnalyzer(wrapper)
        >>> # 计算所有指标
        >>> stats = analyzer.stats()
        >>> # 只计算收益相关指标
        >>> stats = analyzer.stats(tags='returns')
        >>> # 计算特定指标
        >>> stats = analyzer.stats(metrics=['total_return', 'sharpe_ratio'])
    
    配置结构：
        每个指标配置包含以下可选字段：
        - title: 指标标题
        - calc_func: 计算函数名称或可调用对象
        - tags: 指标标签列表，用于过滤
        - agg_func: 聚合函数，用于多列数据
        - apply_to_timedelta: 是否应用时间差转换
        - template_mapping: 模板参数映射
        - 其他自定义参数
    """

    def __init__(self) -> None:
        """
        初始化统计构建器混入类
        
        执行必要的类型检查并初始化实例级别的配置。每个实例都会获得
        一个独立的_metrics配置副本，避免不同实例间的配置干扰。
        
        Raises:
            AssertionError: 如果当前类不是Wrapping的子类
        """
        # 类型检查：确保当前类继承自Wrapping基类
        # 这是使用统计构建器的必要条件，因为需要访问数组包装器的功能
        checks.assert_instance_of(self, Wrapping)

        # 复制可写属性：将类级别的指标配置复制到实例级别
        # 使用深拷贝确保每个实例都有独立的配置副本
        # 这样可以在实例级别修改配置而不影响其他实例或类定义
        self._metrics = self.__class__._metrics.copy()

    @property
    def writeable_attrs(self) -> tp.Set[str]:
        """
        可写属性集合
        
        定义了在配置保存和复制操作中需要包含的可写属性。这些属性的状态
        会在对象序列化、复制或配置更新时被保留。
        
        Returns:
            Set[str]: 包含可写属性名称的集合，目前只包含'_metrics'
        """
        return {'_metrics'}

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """
        统计计算的默认配置参数
        
        从vectorbt全局设置中获取统计构建器的默认配置，并结合当前对象的
        特定信息（如时间频率）生成完整的默认参数字典。
        
        配置来源：
        1. vectorbt._settings.settings中的'stats_builder'配置
        2. 当前对象的时间频率信息（从wrapper.freq获取）
        
        Returns:
            dict: 包含以下默认配置的字典：
                - metrics: 默认要计算的指标列表
                - tags: 默认的标签过滤条件
                - silence_warnings: 是否静默警告
                - template_mapping: 模板参数映射
                - filters: 过滤器配置
                - settings: 包含频率信息的设置字典
                - metric_settings: 指标特定设置
        
        使用示例：
            >>> analyzer = SomeAnalyzer(wrapper)
            >>> defaults = analyzer.stats_defaults
            >>> print(defaults['settings']['freq'])  # 时间频率信息
        """
        from vectorbt._settings import settings  # 导入vectorbt全局设置
        stats_builder_cfg = settings['stats_builder']  # 获取统计构建器配置

        # 合并全局配置和实例特定配置
        return merge_dicts(
            stats_builder_cfg,  # 全局默认配置
            dict(settings=dict(freq=self.wrapper.freq))  # 添加当前对象的频率信息
        )

    # 类变量：定义默认的指标配置
    # 使用ClassVar类型注解表明这是类级别的变量
    # 子类应该重写这个配置来定义自己支持的指标类型
    _metrics: tp.ClassVar[Config] = Config(
        dict(
            # 开始时间指标：获取数据的起始时间点
            start=dict(
                title='Start',  # 指标标题，用于显示
                calc_func=lambda self: self.wrapper.index[0],  # 计算函数：获取索引的第一个元素
                agg_func=None,  # 聚合函数：None表示不需要聚合
                tags='wrapper'  # 标签：标记为包装器相关指标
            ),
            # 结束时间指标：获取数据的结束时间点
            end=dict(
                title='End',  # 指标标题
                calc_func=lambda self: self.wrapper.index[-1],  # 计算函数：获取索引的最后一个元素
                agg_func=None,  # 聚合函数：None表示不需要聚合
                tags='wrapper'  # 标签：标记为包装器相关指标
            ),
            # 时间周期指标：获取数据的时间跨度
            period=dict(
                title='Period',  # 指标标题
                calc_func=lambda self: len(self.wrapper.index),  # 计算函数：获取索引的长度
                apply_to_timedelta=True,  # 应用时间差转换：将数值转换为时间差格式
                agg_func=None,  # 聚合函数：None表示不需要聚合
                tags='wrapper'  # 标签：标记为包装器相关指标
            )
        ),
        copy_kwargs=dict(copy_mode='deep')  # 配置复制参数：使用深拷贝模式确保配置独立性
    )

    @property
    def metrics(self) -> Config:
        """
        获取当前实例支持的指标配置
        
        这是实例级别的指标配置访问器，返回在__init__中创建的配置副本。
        与类级别的配置不同，修改这个配置只会影响当前实例。
        
        配置结构：
            每个指标配置是一个字典，包含以下可选字段：
            - title (str): 指标标题，默认使用指标名称
            - calc_func (str|callable): 计算函数，必需字段
            - tags (list): 指标标签列表，用于过滤和分组
            - agg_func (callable): 聚合函数，用于多列数据
            - apply_to_timedelta (bool): 是否应用时间差转换
            - template_mapping (dict): 模板参数映射
            - check_{filter} (bool): 过滤器检查条件
            - inv_check_{filter} (bool): 反向过滤器检查条件
            - resolve_calc_func (bool): 是否解析计算函数，默认True
            - pass_{arg} (bool): 是否传递特定参数
            - resolve_path_{arg} (bool): 是否解析路径参数
            - resolve_{arg} (bool): 是否解析特定参数
            - 其他自定义参数
        
        Returns:
            Config: 当前实例的指标配置对象，可以进行修改而不影响类定义
        
        使用方式：
            1. 直接修改配置：obj.metrics['new_metric'] = {...}
            2. 重写此属性：在子类中定义自定义的metrics属性
            3. 重写实例变量：obj._metrics = new_config
        
        示例：
            >>> analyzer = PortfolioAnalyzer(wrapper)
            >>> # 查看所有可用指标
            >>> print(list(analyzer.metrics.keys()))
            >>> # 添加新的指标配置
            >>> analyzer.metrics['custom_ratio'] = {
            ...     'title': 'Custom Ratio',
            ...     'calc_func': 'calculate_custom_ratio',
            ...     'tags': ['custom']
            ... }
        """
        return self._metrics

    def stats(self,
              metrics: tp.Optional[tp.MaybeIterable[tp.Union[str, tp.Tuple[str, tp.Kwargs]]]] = None,  # 要计算的指标列表
              tags: tp.Optional[tp.MaybeIterable[str]] = None,  # 标签过滤条件
              column: tp.Optional[tp.Label] = None,  # 指定的列名或组名
              group_by: tp.GroupByLike = None,  # 列分组方式
              agg_func: tp.Optional[tp.Callable] = np.mean,  # 聚合函数，默认为均值
              silence_warnings: tp.Optional[bool] = None,  # 是否静默警告
              template_mapping: tp.Optional[tp.Mapping] = None,  # 全局模板参数映射
              settings: tp.KwargsLike = None,  # 全局设置参数
              filters: tp.KwargsLike = None,  # 过滤器配置
              metric_settings: tp.KwargsLike = None) -> tp.Optional[tp.SeriesFrame]:  # 指标特定设置
        """
        计算对象的各种统计指标
        
        这是统计构建器的核心方法，实现了一个完整的统计指标计算系统。
        该方法支持多指标计算、动态配置、标签过滤、模板替换等高级功能，
        是量化分析中进行性能评估和风险分析的重要工具。
        
        核心工作流程：
        1. 参数解析和默认值设置：处理输入参数，应用默认配置
        2. 指标配置准备：标准化指标配置格式，处理重复名称
        3. 配置合并和模板替换：合并各级配置，替换模板参数
        4. 标签过滤：根据标签条件过滤要计算的指标
        5. 过滤器应用：应用高级过滤条件
        6. 指标计算：执行具体的计算函数
        7. 结果后处理：聚合、格式化和返回结果

        Args:
            metrics (str, tuple, iterable, or dict): Metrics to calculate.

                Each element can be either:

                * a metric name (see keys in `StatsBuilderMixin.metrics`)
                * a tuple of a metric name and a settings dict as in `StatsBuilderMixin.metrics`.

                The settings dict can contain the following keys:

                * `title`: Title of the metric. Defaults to the name.
                * `tags`: Single or multiple tags to associate this metric with.
                    If any of these tags is in `tags`, keeps this metric.
                * `check_{filter}` and `inv_check_{filter}`: Whether to check this metric against a
                    filter defined in `filters`. True (or False for inverse) means to keep this metric.
                * `calc_func` (required): Calculation function for custom metrics.
                    Should return either a scalar for one column/group, pd.Series for multiple columns/groups,
                    or a dict of such for multiple sub-metrics.
                * `resolve_calc_func`: whether to resolve `calc_func`. If the function can be accessed
                    by traversing attributes of this object, you can specify the path to this function
                    as a string (see `vectorbt.utils.attr_.deep_getattr` for the path format).
                    If `calc_func` is a function, arguments from merged metric settings are matched with
                    arguments in the signature (see below). If `resolve_calc_func` is False, `calc_func`
                    should accept (resolved) self and dictionary of merged metric settings.
                    Defaults to True.
                * `post_calc_func`: Function to post-process the result of `calc_func`.
                    Should accept (resolved) self, output of `calc_func`, and dictionary of merged metric settings,
                    and return whatever is acceptable to be returned by `calc_func`. Defaults to None.
                * `fill_wrap_kwargs`: Whether to fill `wrap_kwargs` with `to_timedelta` and `silence_warnings`.
                    Defaults to False.
                * `apply_to_timedelta`: Whether to apply `vectorbt.base.array_wrapper.ArrayWrapper.to_timedelta`
                    on the result. To disable this globally, pass `to_timedelta=False` in `settings`.
                    Defaults to False.
                * `pass_{arg}`: Whether to pass any argument from the settings (see below). Defaults to True if
                    this argument was found in the function's signature. Set to False to not pass.
                    If argument to be passed was not found, `pass_{arg}` is removed.
                * `resolve_path_{arg}`: Whether to resolve an argument that is meant to be an attribute of
                    this object and is the first part of the path of `calc_func`. Passes only optional arguments.
                    Defaults to True. See `vectorbt.utils.attr_.AttrResolver.resolve_attr`.
                * `resolve_{arg}`: Whether to resolve an argument that is meant to be an attribute of
                    this object and is present in the function's signature. Defaults to False.
                    See `vectorbt.utils.attr_.AttrResolver.resolve_attr`.
                * `template_mapping`: Mapping to replace templates in metric settings. Used across all settings.
                * Any other keyword argument that overrides the settings or is passed directly to `calc_func`.

                If `resolve_calc_func` is True, the calculation function may "request" any of the
                following arguments by accepting them or if `pass_{arg}` was found in the settings dict:

                * Each of `vectorbt.utils.attr_.AttrResolver.self_aliases`: original object
                    (ungrouped, with no column selected)
                * `group_by`: won't be passed if it was used in resolving the first attribute of `calc_func`
                    specified as a path, use `pass_group_by=True` to pass anyway
                * `column`
                * `metric_name`
                * `agg_func`
                * `silence_warnings`
                * `to_timedelta`: replaced by True if None and frequency is set
                * Any argument from `settings`
                * Any attribute of this object if it meant to be resolved
                    (see `vectorbt.utils.attr_.AttrResolver.resolve_attr`)

                Pass `metrics='all'` to calculate all supported metrics.
            tags (str or iterable): Tags to select.

                See `vectorbt.utils.tags.match_tags`.
            column (str): Name of the column/group.

                !!! hint
                    There are two ways to select a column: `obj['a'].stats()` and `obj.stats(column='a')`.
                    They both accomplish the same thing but in different ways: `obj['a'].stats()` computes
                    statistics of the column 'a' only, while `obj.stats(column='a')` computes statistics of
                    all columns first and only then selects the column 'a'. The first method is preferred
                    when you have a lot of data or caching is disabled. The second method is preferred when
                    most attributes have already been cached.
            group_by (any): Group or ungroup columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            agg_func (callable): Aggregation function to aggregate statistics across all columns.
                Defaults to mean.

                Should take `pd.Series` and return a const.

                Has only effect if `column` was specified or this object contains only one column of data.

                If `agg_func` has been overridden by a metric:

                * it only takes effect if global `agg_func` is not None
                * will raise a warning if it's None but the result of calculation has multiple values
            silence_warnings (bool): Whether to silence all warnings.
            template_mapping (mapping): Global mapping to replace templates.

                Gets merged over `template_mapping` from `StatsBuilderMixin.stats_defaults`.

                Applied on `settings` and then on each metric settings.
            filters (dict): Filters to apply.

                Each item consists of the filter name and settings dict.

                The settings dict can contain the following keys:

                * `filter_func`: Filter function that should accept resolved self and
                    merged settings for a metric, and return either True or False.
                * `warning_message`: Warning message to be shown when skipping a metric.
                    Can be a template that will be substituted using merged metric settings as mapping.
                    Defaults to None.
                * `inv_warning_message`: Same as `warning_message` but for inverse checks.

                Gets merged over `filters` from `StatsBuilderMixin.stats_defaults`.
            settings (dict): Global settings and resolution arguments.

                Extends/overrides `settings` from `StatsBuilderMixin.stats_defaults`.
                Gets extended/overridden by metric settings.
            metric_settings (dict): Keyword arguments for each metric.

                Extends/overrides all global and metric settings.

        For template logic, see `vectorbt.utils.template`.

        For defaults, see `StatsBuilderMixin.stats_defaults`.

        !!! hint
            There are two types of arguments: optional (or resolution) and mandatory arguments.
            Optional arguments are only passed if they are found in the function's signature.
            Mandatory arguments are passed regardless of this. Optional arguments can only be defined
            using `settings` (that is, globally), while mandatory arguments can be defined both using
            default metric settings and `{metric_name}_kwargs`. Overriding optional arguments using default
            metric settings or `{metric_name}_kwargs` won't turn them into mandatory. For this, pass `pass_{arg}=True`.

        !!! hint
            Make sure to resolve and then to re-use as many object attributes as possible to
            utilize built-in caching (even if global caching is disabled).

        Usage:
            See `vectorbt.portfolio.base` for examples.
        
        Returns:
            pd.Series或pd.DataFrame或None: 计算结果
                - 单列数据或指定column时返回pd.Series
                - 多列数据时返回pd.DataFrame，行为列名，列为指标名
                - 没有指标需要计算时返回None
        
        Raises:
            ValueError: 当指标配置无效或过滤器不存在时
            TypeError: 当计算函数返回不支持的数据类型时
            Exception: 计算过程中的其他异常会被重新抛出
        
        使用示例：
            >>> # 基本用法：计算所有指标
            >>> stats = analyzer.stats()
            
            >>> # 计算特定指标
            >>> stats = analyzer.stats(metrics=['total_return', 'sharpe_ratio'])
            
            >>> # 使用标签过滤
            >>> stats = analyzer.stats(tags=['returns', 'risk'])
            
            >>> # 指定列和聚合函数
            >>> stats = analyzer.stats(column='AAPL', agg_func=np.median)
            
            >>> # 使用自定义设置
            >>> stats = analyzer.stats(
            ...     settings={'window': 252},
            ...     metric_settings={'sharpe_ratio': {'risk_free': 0.02}}
            ... )
            
            >>> # 使用模板参数
            >>> stats = analyzer.stats(
            ...     template_mapping={'window': 252, 'freq': 'D'},
            ...     settings={'rolling_window': '$window', 'frequency': '$freq'}
            ... )
        """
        # 第一步：解析默认值和配置合并
        # 处理输入参数，将None值替换为默认配置，合并各级配置参数
        if silence_warnings is None:
            # 从默认配置中获取是否静默警告的设置
            silence_warnings = self.stats_defaults.get('silence_warnings', False)
        # 合并模板映射：全局默认 + 用户提供的映射
        template_mapping = merge_dicts(self.stats_defaults.get('template_mapping', {}), template_mapping)
        # 合并过滤器配置：全局默认 + 用户提供的过滤器
        filters = merge_dicts(self.stats_defaults.get('filters', {}), filters)
        # 合并全局设置：全局默认 + 用户提供的设置
        settings = merge_dicts(self.stats_defaults.get('settings', {}), settings)
        # 合并指标特定设置：全局默认 + 用户提供的指标设置
        metric_settings = merge_dicts(self.stats_defaults.get('metric_settings', {}), metric_settings)

        # 第二步：全局模板替换
        # 在指标级别处理之前，先对全局设置进行模板参数替换
        if len(template_mapping) > 0:
            # 使用模板映射对设置进行深度替换
            sub_settings = deep_substitute(settings, mapping=template_mapping)
        else:
            # 没有模板映射时直接使用原设置
            sub_settings = settings

        # 第三步：解析self对象
        # 根据条件参数解析self对象，获得一个可能经过变换的对象实例
        # impacts_caching=False表示这个解析不会影响缓存策略
        reself = self.resolve_self(
            cond_kwargs=sub_settings,  # 条件参数，用于决定如何解析对象
            impacts_caching=False,  # 不影响缓存
            silence_warnings=silence_warnings  # 警告控制
        )

        # 第四步：准备指标列表
        # 处理metrics参数，将其标准化为统一的格式
        if metrics is None:
            # 如果未指定指标，从默认配置中获取，默认为'all'
            metrics = reself.stats_defaults.get('metrics', 'all')
        if metrics == 'all':
            # 'all'表示使用所有可用的指标配置
            metrics = reself.metrics
        if isinstance(metrics, dict):
            # 如果是字典格式，转换为(name, settings)元组列表
            metrics = list(metrics.items())
        if isinstance(metrics, (str, tuple)):
            # 如果是单个指标，包装成列表
            metrics = [metrics]

        # 第五步：准备标签过滤条件
        # 处理tags参数，用于后续的指标过滤
        if tags is None:
            # 如果未指定标签，从默认配置中获取
            tags = reself.stats_defaults.get('tags', 'all')
        if isinstance(tags, str) and tags == 'all':
            # 'all'表示不进行标签过滤
            tags = None
        if isinstance(tags, (str, tuple)):
            # 如果是单个标签，包装成列表
            tags = [tags]

        # 第六步：标准化指标格式
        # 将所有指标转换为(name, settings)元组的统一格式
        new_metrics = []
        for i, metric in enumerate(metrics):
            if isinstance(metric, str):
                # 字符串指标名：从配置中获取对应的设置
                metric = (metric, reself.metrics[metric])
            if not isinstance(metric, tuple):
                # 验证格式：必须是元组格式
                raise TypeError(f"Metric at index {i} must be either a string or a tuple")
            new_metrics.append(metric)
        metrics = new_metrics

        # 第七步：处理重复的指标名称
        # 当同一个指标被多次指定时，为其添加数字后缀以区分
        metric_counts = Counter(list(map(lambda x: x[0], metrics)))  # 统计每个指标名的出现次数
        metric_i = {k: -1 for k in metric_counts.keys()}  # 为每个指标名初始化计数器
        metrics_dct = {}  # 最终的指标配置字典
        for i, (metric_name, _metric_settings) in enumerate(metrics):
            if metric_counts[metric_name] > 1:
                # 如果指标名重复，添加数字后缀
                metric_i[metric_name] += 1
                metric_name = metric_name + '_' + str(metric_i[metric_name])
            metrics_dct[metric_name] = _metric_settings

        # 第八步：验证指标特定设置
        # 检查metric_settings中的键是否都能匹配到有效的指标
        missed_keys = set(metric_settings.keys()).difference(set(metrics_dct.keys()))
        if len(missed_keys) > 0:
            raise ValueError(f"Keys {missed_keys} in metric_settings could not be matched with any metric")

        # 第九步：合并各级配置设置
        # 为每个指标合并全局设置、指标默认设置和用户指定设置
        opt_arg_names_dct = {}  # 存储每个指标的可选参数名称集合
        custom_arg_names_dct = {}  # 存储每个指标的自定义参数名称集合
        resolved_self_dct = {}  # 存储每个指标解析后的self对象
        mapping_dct = {}  # 存储每个指标的模板映射字典
        
        for metric_name, _metric_settings in list(metrics_dct.items()):
            # 构建可选设置：包含系统提供的标准参数
            opt_settings = merge_dicts(
                # self别名映射：为不同的self别名提供相同的对象引用
                {name: reself for name in reself.self_aliases},
                # 标准参数：stats方法的基本参数
                dict(
                    column=column,  # 列名参数
                    group_by=group_by,  # 分组参数
                    metric_name=metric_name,  # 当前指标名称
                    agg_func=agg_func,  # 聚合函数
                    silence_warnings=silence_warnings,  # 警告控制
                    to_timedelta=None  # 时间差转换标志，稍后会根据频率设置
                ),
                settings  # 用户提供的全局设置
            )
            
            # 复制指标默认设置，避免修改原始配置
            _metric_settings = _metric_settings.copy()
            # 获取用户为该指标提供的特定设置
            passed_metric_settings = metric_settings.get(metric_name, {})
            
            # 按优先级合并设置：可选设置 < 指标默认设置 < 用户特定设置
            merged_settings = merge_dicts(
                opt_settings,  # 最低优先级：系统默认参数
                _metric_settings,  # 中等优先级：指标默认配置
                passed_metric_settings  # 最高优先级：用户特定设置
            )
            
            # 处理指标级别的模板映射
            metric_template_mapping = merged_settings.pop('template_mapping', {})
            # 合并全局模板映射和指标特定模板映射
            template_mapping_merged = merge_dicts(template_mapping, metric_template_mapping)
            # 对模板映射本身进行模板替换（支持嵌套模板）
            template_mapping_merged = deep_substitute(template_mapping_merged, mapping=merged_settings)
            # 创建完整的映射字典，用于后续的模板替换
            mapping = merge_dicts(template_mapping_merged, merged_settings)
            # 对合并后的设置进行最终的模板替换
            merged_settings = deep_substitute(merged_settings, mapping=mapping)

            # 第十步：标签过滤
            # 根据用户指定的标签条件过滤指标
            if tags is not None:
                # 获取当前指标的标签设置
                in_tags = merged_settings.get('tags', None)
                # 如果指标没有标签或标签不匹配，则跳过该指标
                if in_tags is None or not match_tags(tags, in_tags):
                    metrics_dct.pop(metric_name, None)  # 从指标字典中移除
                    continue  # 跳过后续处理

            # 第十一步：参数名称分类
            # 将参数名称分为自定义参数和可选参数两类，用于后续的参数解析
            custom_arg_names = set(_metric_settings.keys()).union(set(passed_metric_settings.keys()))  # 自定义参数：来自配置的参数
            opt_arg_names = set(opt_settings.keys())  # 可选参数：系统提供的标准参数
            
            # 第十二步：解析指标特定的self对象
            # 为每个指标创建一个定制化的self对象，支持缓存和条件解析
            custom_reself = reself.resolve_self(
                cond_kwargs=merged_settings,  # 条件参数：用于决定如何解析对象
                custom_arg_names=custom_arg_names,  # 自定义参数名称
                impacts_caching=True,  # 影响缓存：这个解析会影响缓存策略
                silence_warnings=merged_settings['silence_warnings']  # 警告控制
            )

            # 第十三步：存储处理结果
            # 将处理后的配置和对象存储到对应的字典中，供后续使用
            metrics_dct[metric_name] = merged_settings  # 最终的合并设置
            custom_arg_names_dct[metric_name] = custom_arg_names  # 自定义参数名称集合
            opt_arg_names_dct[metric_name] = opt_arg_names  # 可选参数名称集合
            resolved_self_dct[metric_name] = custom_reself  # 解析后的self对象
            mapping_dct[metric_name] = mapping  # 模板映射字典

        # 第十四步：应用高级过滤器
        # 根据配置的过滤器条件进一步筛选指标，支持正向和反向过滤
        for metric_name, _metric_settings in list(metrics_dct.items()):
            # 获取该指标的相关信息
            custom_reself = resolved_self_dct[metric_name]  # 解析后的self对象
            mapping = mapping_dct[metric_name]  # 模板映射字典
            _silence_warnings = _metric_settings.get('silence_warnings')  # 警告控制

            # 收集该指标需要的过滤器
            metric_filters = set()
            for k in _metric_settings.keys():
                filter_name = None
                if k.startswith('check_'):
                    # 正向过滤器：check_xxx表示需要xxx过滤器返回True
                    filter_name = k[len('check_'):]
                elif k.startswith('inv_check_'):
                    # 反向过滤器：inv_check_xxx表示需要xxx过滤器返回False
                    filter_name = k[len('inv_check_'):]
                if filter_name is not None:
                    # 验证过滤器是否存在
                    if filter_name not in filters:
                        raise ValueError(f"Metric '{metric_name}' requires filter '{filter_name}'")
                    metric_filters.add(filter_name)

            # 应用每个过滤器
            for filter_name in metric_filters:
                # 获取过滤器配置
                filter_settings = filters[filter_name]
                # 对过滤器设置进行模板替换
                _filter_settings = deep_substitute(filter_settings, mapping=mapping)
                # 提取过滤器函数和警告消息
                filter_func = _filter_settings['filter_func']  # 过滤器函数
                warning_message = _filter_settings.get('warning_message', None)  # 正向过滤警告消息
                inv_warning_message = _filter_settings.get('inv_warning_message', None)  # 反向过滤警告消息
                # 检查该指标是否需要应用此过滤器
                to_check = _metric_settings.get('check_' + filter_name, False)  # 是否需要正向检查
                inv_to_check = _metric_settings.get('inv_check_' + filter_name, False)  # 是否需要反向检查

                # 如果需要检查，则执行过滤器函数
                if to_check or inv_to_check:
                    # 调用过滤器函数，传入解析后的self对象和指标设置
                    whether_true = filter_func(custom_reself, _metric_settings)
                    # 判断是否需要移除该指标
                    # 正向检查：过滤器返回False时移除
                    # 反向检查：过滤器返回True时移除
                    to_remove = (to_check and not whether_true) or (inv_to_check and whether_true)
                    
                    if to_remove:
                        # 发出相应的警告消息
                        if to_check and warning_message is not None and not _silence_warnings:
                            warnings.warn(warning_message)
                        if inv_to_check and inv_warning_message is not None and not _silence_warnings:
                            warnings.warn(inv_warning_message)

                        # 从所有相关字典中移除该指标
                        metrics_dct.pop(metric_name, None)
                        custom_arg_names_dct.pop(metric_name, None)
                        opt_arg_names_dct.pop(metric_name, None)
                        resolved_self_dct.pop(metric_name, None)
                        mapping_dct.pop(metric_name, None)
                        break  # 一旦被移除，就不需要检查其他过滤器了

        # 第十五步：检查是否还有指标需要计算
        # 经过标签过滤和高级过滤后，可能没有指标剩余
        if len(metrics_dct) == 0:
            if not silence_warnings:
                warnings.warn("No metrics to calculate", stacklevel=2)
            return None  # 没有指标需要计算，返回None

        # 第十六步：开始计算统计指标
        # 初始化计算所需的数据结构
        arg_cache_dct = {}  # 参数缓存字典，用于提高重复计算的性能
        stats_dct = {}  # 统计结果字典，存储所有计算出的指标值
        used_agg_func = False  # 标记是否使用了聚合函数
        
        # 遍历每个需要计算的指标
        for i, (metric_name, _metric_settings) in enumerate(metrics_dct.items()):
            try:
                # 准备计算参数
                final_kwargs = _metric_settings.copy()  # 复制设置，避免修改原始配置
                opt_arg_names = opt_arg_names_dct[metric_name]  # 可选参数名称
                custom_arg_names = custom_arg_names_dct[metric_name]  # 自定义参数名称
                custom_reself = resolved_self_dct[metric_name]  # 解析后的self对象

                # 清理不需要传递给计算函数的键
                for k, v in list(final_kwargs.items()):
                    if k.startswith('check_') or k.startswith('inv_check_') or k in ('tags',):
                        final_kwargs.pop(k, None)  # 移除过滤器相关的键和标签

                # 提取指标特定的配置值
                _column = final_kwargs.get('column')  # 列名
                _group_by = final_kwargs.get('group_by')  # 分组方式
                _agg_func = final_kwargs.get('agg_func')  # 聚合函数
                _silence_warnings = final_kwargs.get('silence_warnings')  # 警告控制
                
                # 设置时间差转换标志
                if final_kwargs['to_timedelta'] is None:
                    # 如果未明确设置，根据频率信息自动决定
                    final_kwargs['to_timedelta'] = custom_reself.wrapper.freq is not None
                to_timedelta = final_kwargs.get('to_timedelta')
                
                # 提取计算相关的配置
                title = final_kwargs.pop('title', metric_name)  # 指标标题，默认使用指标名
                calc_func = final_kwargs.pop('calc_func')  # 计算函数
                resolve_calc_func = final_kwargs.pop('resolve_calc_func', True)  # 是否解析计算函数
                post_calc_func = final_kwargs.pop('post_calc_func', None)  # 后处理函数
                use_caching = final_kwargs.pop('use_caching', True)  # 是否使用缓存
                fill_wrap_kwargs = final_kwargs.pop('fill_wrap_kwargs', False)  # 是否填充包装参数
                
                # 处理包装参数
                if fill_wrap_kwargs:
                    # 自动填充包装参数，用于数组包装器的配置
                    final_kwargs['wrap_kwargs'] = merge_dicts(
                        dict(to_timedelta=to_timedelta, silence_warnings=_silence_warnings),
                        final_kwargs.get('wrap_kwargs', None)
                    )
                apply_to_timedelta = final_kwargs.pop('apply_to_timedelta', False)  # 是否对结果应用时间差转换

                # Resolve calc_func
                if resolve_calc_func:
                    if not callable(calc_func):
                        passed_kwargs_out = {}

                        def _getattr_func(obj: tp.Any,
                                          attr: str,
                                          args: tp.ArgsLike = None,
                                          kwargs: tp.KwargsLike = None,
                                          call_attr: bool = True,
                                          _final_kwargs: tp.Kwargs = final_kwargs,
                                          _opt_arg_names: tp.Set[str] = opt_arg_names,
                                          _custom_arg_names: tp.Set[str] = custom_arg_names,
                                          _arg_cache_dct: tp.Kwargs = arg_cache_dct) -> tp.Any:
                            if attr in final_kwargs:
                                return final_kwargs[attr]
                            if args is None:
                                args = ()
                            if kwargs is None:
                                kwargs = {}
                            if obj is custom_reself and _final_kwargs.pop('resolve_path_' + attr, True):
                                if call_attr:
                                    return custom_reself.resolve_attr(
                                        attr,
                                        args=args,
                                        cond_kwargs={k: v for k, v in _final_kwargs.items() if k in _opt_arg_names},
                                        kwargs=kwargs,
                                        custom_arg_names=_custom_arg_names,
                                        cache_dct=_arg_cache_dct,
                                        use_caching=use_caching,
                                        passed_kwargs_out=passed_kwargs_out
                                    )
                                return getattr(obj, attr)
                            out = getattr(obj, attr)
                            if callable(out) and call_attr:
                                return out(*args, **kwargs)
                            return out

                        calc_func = custom_reself.deep_getattr(
                            calc_func,
                            getattr_func=_getattr_func,
                            call_last_attr=False
                        )

                        if 'group_by' in passed_kwargs_out:
                            if 'pass_group_by' not in final_kwargs:
                                final_kwargs.pop('group_by', None)

                    # Resolve arguments
                    if callable(calc_func):
                        func_arg_names = get_func_arg_names(calc_func)
                        for k in func_arg_names:
                            if k not in final_kwargs:
                                if final_kwargs.pop('resolve_' + k, False):
                                    try:
                                        arg_out = custom_reself.resolve_attr(
                                            k,
                                            cond_kwargs=final_kwargs,
                                            custom_arg_names=custom_arg_names,
                                            cache_dct=arg_cache_dct,
                                            use_caching=use_caching
                                        )
                                    except AttributeError:
                                        continue
                                    final_kwargs[k] = arg_out
                        for k in list(final_kwargs.keys()):
                            if k in opt_arg_names:
                                if 'pass_' + k in final_kwargs:
                                    if not final_kwargs.get('pass_' + k):  # first priority
                                        final_kwargs.pop(k, None)
                                elif k not in func_arg_names:  # second priority
                                    final_kwargs.pop(k, None)
                        for k in list(final_kwargs.keys()):
                            if k.startswith('pass_') or k.startswith('resolve_'):
                                final_kwargs.pop(k, None)  # cleanup

                        # Call calc_func
                        out = calc_func(**final_kwargs)
                    else:
                        # calc_func is already a result
                        out = calc_func
                else:
                    # Do not resolve calc_func
                    out = calc_func(custom_reself, _metric_settings)

                # Call post_calc_func
                if post_calc_func is not None:
                    out = post_calc_func(custom_reself, out, _metric_settings)

                # Post-process and store the metric
                multiple = True
                if not isinstance(out, dict):
                    multiple = False
                    out = {None: out}
                for k, v in out.items():
                    # Resolve title
                    if multiple:
                        if title is None:
                            t = str(k)
                        else:
                            t = title + ': ' + str(k)
                    else:
                        t = title

                    # Check result type
                    if checks.is_any_array(v) and not checks.is_series(v):
                        raise TypeError("calc_func must return either a scalar for one column/group, "
                                        "pd.Series for multiple columns/groups, or a dict of such. "
                                        f"Not {type(v)}.")

                    # Handle apply_to_timedelta
                    if apply_to_timedelta and to_timedelta:
                        v = custom_reself.wrapper.to_timedelta(v, silence_warnings=_silence_warnings)

                    # Select column or aggregate
                    if checks.is_series(v):
                        if _column is not None:
                            v = custom_reself.select_one_from_obj(
                                v, custom_reself.wrapper.regroup(_group_by), column=_column)
                        elif _agg_func is not None and agg_func is not None:
                            v = _agg_func(v)
                            used_agg_func = True
                        elif _agg_func is None and agg_func is not None:
                            if not _silence_warnings:
                                warnings.warn(f"Metric '{metric_name}' returned multiple values "
                                              f"despite having no aggregation function", stacklevel=2)
                            continue

                    # Store metric
                    if t in stats_dct:
                        if not _silence_warnings:
                            warnings.warn(f"Duplicate metric title '{t}'", stacklevel=2)
                    stats_dct[t] = v
            except Exception as e:
                warnings.warn(f"Metric '{metric_name}' raised an exception", stacklevel=2)
                raise e

        # Return the stats
        if reself.wrapper.get_ndim(group_by=group_by) == 1:
            return pd.Series(stats_dct, name=reself.wrapper.get_name(group_by=group_by))
        if column is not None:
            return pd.Series(stats_dct, name=column)
        if agg_func is not None:
            if used_agg_func and not silence_warnings:
                warnings.warn(f"Object has multiple columns. Aggregating using {agg_func}. "
                              f"Pass column to select a single column/group.", stacklevel=2)
            return pd.Series(stats_dct, name='agg_func_' + agg_func.__name__)
        new_index = reself.wrapper.grouper.get_columns(group_by=group_by)
        stats_df = pd.DataFrame(stats_dct, index=new_index)
        return stats_df

    # ############# 文档生成方法 ############# #

    @classmethod
    def build_metrics_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """
        构建指标配置的文档字符串
        
        这是一个类方法，用于自动生成指标配置的文档。它会提取指标配置信息
        并格式化为标准的文档字符串，支持自动文档生成工具。
        
        Args:
            source_cls (type, optional): 源类，用于获取文档模板
                                       默认为StatsBuilderMixin
        
        Returns:
            str: 格式化后的指标文档字符串
        
        使用示例：
            >>> # 为自定义分析器类生成文档
            >>> doc = CustomAnalyzer.build_metrics_doc()
            >>> print(doc)  # 显示所有指标的详细文档
        """
        if source_cls is None:
            source_cls = StatsBuilderMixin  # 默认使用基类作为文档模板源
        
        # 使用字符串模板进行文档生成
        return string.Template(
            inspect.cleandoc(get_dict_attr(source_cls, 'metrics').__doc__)  # 获取模板文档字符串
        ).substitute(
            {
                'metrics': cls.metrics.to_doc(),  # 将指标配置转换为文档格式
                'cls_name': cls.__name__  # 插入当前类名
            }
        )

    @classmethod
    def override_metrics_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """
        重写指标文档
        
        这个方法用于在子类中重写metrics属性的文档字符串。它会自动调用
        build_metrics_doc方法生成新的文档并更新__pdoc__字典。
        
        Args:
            __pdoc__ (dict): 文档字典，用于存储类和方法的文档字符串
            source_cls (type, optional): 源类，用于获取文档模板
        
        使用示例：
            >>> # 在子类中重写文档
            >>> __pdoc__ = {}
            >>> CustomAnalyzer.override_metrics_doc(__pdoc__)
            >>> # 现在__pdoc__包含了更新后的文档
        """
        # 生成并设置新的metrics属性文档
        __pdoc__[cls.__name__ + '.metrics'] = cls.build_metrics_doc(source_cls=source_cls)


# 文档字典初始化
# 这个字典用于存储自动生成的文档字符串，供文档生成工具使用
__pdoc__ = dict()
# 为基类生成默认的指标文档
StatsBuilderMixin.override_metrics_doc(__pdoc__)
