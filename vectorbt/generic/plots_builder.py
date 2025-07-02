# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
图表构建器混入模块 (Plots Builder Mixin Module)。
支持将复杂的量化分析结果转换为直观的可视化图表。
"""

import inspect
import string
import warnings
from collections import Counter

from vectorbt import _typing as tp 
from vectorbt.base.array_wrapper import Wrapping
from vectorbt.utils import checks
from vectorbt.utils.attr_ import get_dict_attr
from vectorbt.utils.config import Config, merge_dicts, get_func_arg_names
from vectorbt.utils.figure import make_subplots, get_domain
from vectorbt.utils.tags import match_tags
from vectorbt.utils.template import deep_substitute


class MetaPlotsBuilderMixin(type):
    """
    图表构建器混入类的元类。为PlotsBuilderMixin提供了一个只读的类属性`subplots`。
    """

    @property
    def subplots(cls) -> Config:
        """
        获取类支持的子图配置
        
        这是一个类属性装饰器，提供对_subplots配置的只读访问。
        每个继承PlotsBuilderMixin的类都可以定义自己的_subplots配置。
        
        Returns:
            Config: 包含所有支持的子图定义的配置对象
                   每个子图包含绘图函数、标签、参数等信息
        """
        return cls._subplots


class PlotsBuilderMixin(metaclass=MetaPlotsBuilderMixin):
    """
    图表构建器混入类。为量化分析对象提供统一的绘图接口。
    
    核心特性：
    1. 多子图支持：可以在一个图表中组合多个不同类型的子图
    2. 配置驱动：通过声明式配置定义图表结构和样式
    3. 动态参数：支持运行时参数替换和条件渲染
    4. 标签过滤：使用标签系统控制子图的显示条件
    5. 模板系统：支持配置参数的动态替换
    6. 自动布局：智能计算图表尺寸和子图间距
    
    必要条件：
    - 必须继承自vectorbt.base.array_wrapper.Wrapping类
    - 需要定义_subplots类变量来指定支持的子图类型
    
    使用示例：
        >>> class PriceAnalyzer(Wrapping, PlotsBuilderMixin):
        ...     _subplots = Config({
        ...         'price': {
        ...             'title': 'Price Chart',
        ...             'plot_func': 'plot_price',
        ...             'tags': ['basic', 'price']
        ...         },
        ...         'volume': {
        ...             'title': 'Volume',
        ...             'plot_func': 'plot_volume', 
        ...             'tags': ['basic', 'volume']
        ...         }
        ...     })
        ...     
        ...     def plot_price(self, fig, **kwargs):
        ...         # 绘制价格图表的具体实现
        ...         pass
        
        >>> analyzer = PriceAnalyzer(wrapper)
        >>> # 绘制所有子图
        >>> fig = analyzer.plots()
        >>> # 只绘制价格相关图表
        >>> fig = analyzer.plots(tags='price')
        >>> # 绘制特定子图
        >>> fig = analyzer.plots(subplots=['price', 'volume'])
    
    配置结构：
        每个子图配置包含以下可选字段：
        - title: 子图标题
        - plot_func: 绘图函数名称或可调用对象
        - tags: 子图标签列表，用于过滤
        - xaxis_kwargs: X轴配置参数
        - yaxis_kwargs: Y轴配置参数
        - template_mapping: 模板参数映射
        - 其他自定义参数
    """

    def __init__(self):
        """
        初始化图表构建器混入类
        
        执行必要的类型检查并初始化实例级别的配置。每个实例都会获得
        一个独立的_subplots配置副本，避免不同实例间的配置干扰。
        
        Raises:
            AssertionError: 如果当前类不是Wrapping的子类
        """
        # 类型检查：确保当前类继承自Wrapping基类
        # 这是使用图表构建器的必要条件，因为需要访问数组包装器的功能
        checks.assert_instance_of(self, Wrapping)

        # 复制类级别的子图配置到实例级别
        # 使用深拷贝确保每个实例都有独立的配置副本
        # 这样可以在实例级别修改配置而不影响其他实例或类定义
        self._subplots = self.__class__._subplots.copy()

    @property
    def writeable_attrs(self) -> tp.Set[str]:
        """
        可写属性集合
        
        定义了在配置保存和复制操作中需要包含的可写属性。这些属性的状态
        会在对象序列化、复制或配置更新时被保留。
        
        Returns:
            Set[str]: 包含可写属性名称的集合，目前只包含'_subplots'
        """
        return {'_subplots'}

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """
        图表绘制的默认配置参数
        
        从vectorbt全局设置中获取图表构建器的默认配置，并结合当前对象的
        特定信息（如时间频率）生成完整的默认参数字典。
        
        配置来源：
        1. vectorbt._settings.settings中的'plots_builder'配置
        2. 当前对象的时间频率信息（从wrapper.freq获取）
        
        Returns:
            dict: 包含以下默认配置的字典：
                - subplots: 默认要绘制的子图列表
                - tags: 默认的标签过滤条件
                - silence_warnings: 是否静默警告
                - show_titles: 是否显示子图标题
                - hide_id_labels: 是否隐藏重复的图例标签
                - group_id_labels: 是否分组相同的图例标签
                - template_mapping: 模板参数映射
                - filters: 过滤器配置
                - settings: 包含频率信息的设置字典
                - subplot_settings: 子图特定设置
                - make_subplots_kwargs: 传递给make_subplots的参数
                - layout_kwargs: 图表布局参数
        
        使用示例：
            >>> analyzer = SomeAnalyzer(wrapper)
            >>> defaults = analyzer.plots_defaults
            >>> print(defaults['show_titles'])  # True或False
            >>> print(defaults['settings']['freq'])  # 时间频率信息
        """
        from vectorbt._settings import settings  # 导入vectorbt全局设置
        plots_builder_cfg = settings['plots_builder']  # 获取图表构建器配置

        # 合并全局配置和实例特定配置
        return merge_dicts(
            plots_builder_cfg,  # 全局默认配置
            dict(settings=dict(freq=self.wrapper.freq))  # 添加当前对象的频率信息
        )

    # 类变量：定义默认的空子图配置
    # 使用ClassVar类型注解表明这是类级别的变量
    # 子类应该重写这个配置来定义自己支持的子图类型
    _subplots: tp.ClassVar[Config] = Config(
        dict(),  # 空字典作为默认配置
        copy_kwargs=dict(copy_mode='deep')  # 使用深拷贝模式确保配置独立性
    )

    @property
    def subplots(self) -> Config:
        """
        获取当前实例支持的子图配置
        
        这是实例级别的子图配置访问器，返回在__init__中创建的配置副本。
        与类级别的配置不同，修改这个配置只会影响当前实例。
        
        配置结构：
            每个子图配置是一个字典，包含以下可选字段：
            - title (str): 子图标题，默认使用子图名称
            - plot_func (str|callable): 绘图函数，必需字段
            - tags (list): 子图标签列表，用于过滤和分组
            - xaxis_kwargs (dict): X轴配置参数
            - yaxis_kwargs (dict): Y轴配置参数
            - template_mapping (dict): 模板参数映射
            - check_{filter} (bool): 过滤器检查条件
            - inv_check_{filter} (bool): 反向过滤器检查条件
            - resolve_plot_func (bool): 是否解析绘图函数，默认True
            - pass_{arg} (bool): 是否传递特定参数
            - resolve_path_{arg} (bool): 是否解析路径参数
            - resolve_{arg} (bool): 是否解析特定参数
            - 其他自定义参数
        
        Returns:
            Config: 当前实例的子图配置对象，可以进行修改而不影响类定义
        
        使用方式：
            1. 直接修改配置：obj.subplots['new_plot'] = {...}
            2. 重写此属性：在子类中定义自定义的subplots属性
            3. 重写实例变量：obj._subplots = new_config
        
        示例：
            >>> analyzer = PriceAnalyzer(wrapper)
            >>> # 查看所有可用子图
            >>> print(list(analyzer.subplots.keys()))
            >>> # 添加新的子图配置
            >>> analyzer.subplots['custom'] = {
            ...     'title': 'Custom Plot',
            ...     'plot_func': 'plot_custom',
            ...     'tags': ['custom']
            ... }
        """
        return self._subplots

    def plots(self,
              subplots: tp.Optional[tp.MaybeIterable[tp.Union[str, tp.Tuple[str, tp.Kwargs]]]] = None,  # 要绘制的子图规范
              tags: tp.Optional[tp.MaybeIterable[str]] = None,  # 标签过滤条件
              column: tp.Optional[tp.Label] = None,  # 列选择器
              group_by: tp.GroupByLike = None,  # 分组依据
              silence_warnings: tp.Optional[bool] = None,  # 是否静默警告
              template_mapping: tp.Optional[tp.Mapping] = None,  # 模板参数映射
              settings: tp.KwargsLike = None,  # 全局设置参数
              filters: tp.KwargsLike = None,  # 过滤器配置
              subplot_settings: tp.KwargsLike = None,  # 子图特定设置
              show_titles: bool = None,  # 是否显示子图标题
              hide_id_labels: bool = None,  # 是否隐藏重复图例标签
              group_id_labels: bool = None,  # 是否分组相同图例标签
              make_subplots_kwargs: tp.KwargsLike = None,  # make_subplots函数参数
              **layout_kwargs) -> tp.Optional[tp.BaseFigure]:  # 图表布局参数
        """
        绘制对象的各个部分的图表
        
        这是图表构建器的核心方法，提供了一个统一的接口来创建复杂的多子图可视化。
        该方法支持灵活的子图配置、动态参数替换、标签过滤和自动布局等高级功能。
        
        核心工作流程：
        1. 参数解析和默认值设置
        2. 子图配置的标准化和验证
        3. 标签过滤和条件检查
        4. 模板参数替换和动态配置
        5. 图表布局计算和子图创建
        6. 逐个子图的绘制和渲染
        7. 图例优化和最终布局调整
        
        Args:
            subplots: 要绘制的子图规范，支持多种格式：
                - None: 使用默认子图配置
                - 'all': 绘制所有支持的子图
                - str: 单个子图名称
                - tuple: (子图名称, 配置字典) 的元组
                - list: 包含上述格式的列表
                - dict: {子图名称: 配置字典} 的字典
                
                每个子图配置字典可以包含：
                - title: 子图标题，默认使用子图名称
                - plot_func: 绘图函数，必需字段，可以是函数名或可调用对象
                - xaxis_kwargs: X轴配置参数，默认 {'title': 'Index'}
                - yaxis_kwargs: Y轴配置参数，默认为空字典
                - tags: 子图标签列表，用于过滤
                - check_{filter}: 过滤器检查条件
                - inv_check_{filter}: 反向过滤器检查条件
                - resolve_plot_func: 是否解析绘图函数，默认True
                - pass_{arg}: 是否传递特定参数到绘图函数
                - resolve_path_{arg}: 是否解析路径参数
                - resolve_{arg}: 是否解析特定参数
                - template_mapping: 子图级别的模板参数映射
                - 其他自定义参数会传递给绘图函数
                
                当resolve_plot_func=True时，绘图函数可以请求以下参数：
                - self的各种别名：原始对象（未分组、未选择列）
                - group_by: 分组参数（某些情况下不会传递）
                - column: 列选择参数
                - subplot_name: 当前子图名称
                - trace_names: 包含子图名称的列表
                - add_trace_kwargs: 包含子图行列索引的字典
                - xref, yref: X/Y轴引用
                - xaxis, yaxis: X/Y轴名称
                - x_domain, y_domain: X/Y轴域范围
                - fig: Plotly图形对象
                - silence_warnings: 警告控制参数
                - settings中的任何参数
                - 对象的任何可解析属性
                
            tags: 标签过滤条件，支持：
                - None: 不进行标签过滤
                - 'all': 显示所有标签的子图
                - str: 单个标签名称
                - list: 标签列表，使用OR逻辑
                - 支持布尔表达式，如 'tag1 and not tag2'
                
            column: 列选择器，用于选择特定的数据列进行分析
            
            group_by: 分组依据，用于数据的分组操作
            
            silence_warnings: 是否静默警告信息，None时使用默认配置
            
            template_mapping: 全局模板参数映射，用于动态替换配置中的占位符
                应用顺序：settings -> make_subplots_kwargs -> layout_kwargs -> 各子图设置
                
            filters: 过滤器配置字典，定义各种过滤条件
            
            settings: 全局设置参数，会传递给所有子图的绘图函数
            
            subplot_settings: 子图特定设置，格式为 {子图名称: 设置字典}
            
            show_titles: 是否显示子图标题，None时使用默认配置
            
            hide_id_labels: 是否隐藏相同的图例标签
                两个标签被认为相同当它们的名称、标记样式和线条样式都匹配时
                
            group_id_labels: 是否将相同的图例标签分组显示
            
            make_subplots_kwargs: 传递给plotly.subplots.make_subplots的参数
                常用参数包括：
                - rows, cols: 子图行列数
                - shared_xaxes, shared_yaxes: 是否共享坐标轴
                - subplot_titles: 子图标题列表
                - vertical_spacing, horizontal_spacing: 子图间距
                
            **layout_kwargs: 图表布局参数，用于更新图形的整体布局
                常用参数包括：
                - width, height: 图形尺寸
                - title: 整体标题
                - legend: 图例配置
                - font: 字体配置
                - showlegend: 是否显示图例
                
        Returns:
            Optional[BaseFigure]: Plotly图形对象，如果没有子图需要绘制则返回None
            
        Raises:
            TypeError: 当子图配置格式不正确时
            ValueError: 当过滤器配置缺失或子图设置键不匹配时
            Exception: 子图绘制过程中的各种异常
            
        使用示例：
            >>> # 基本用法 - 绘制所有默认子图
            >>> fig = analyzer.plots()
            
            >>> # 绘制特定子图
            >>> fig = analyzer.plots(subplots=['price', 'volume'])
            
            >>> # 使用标签过滤
            >>> fig = analyzer.plots(tags='technical')
            >>> fig = analyzer.plots(tags=['price', 'volume'])
            >>> fig = analyzer.plots(tags='price and not volume')
            
            >>> # 自定义子图配置
            >>> fig = analyzer.plots(subplots=[
            ...     'price',
            ...     ('volume', {'title': 'Custom Volume', 'yaxis_kwargs': {'title': 'Vol'}})
            ... ])
            
            >>> # 高级配置
            >>> fig = analyzer.plots(
            ...     subplots='all',
            ...     show_titles=True,
            ...     hide_id_labels=True,
            ...     make_subplots_kwargs={'rows': 2, 'cols': 1, 'shared_xaxes': True},
            ...     width=800,
            ...     height=600
            ... )
            
            >>> # 使用模板参数
            >>> fig = analyzer.plots(
            ...     template_mapping={'symbol': 'AAPL', 'period': '1D'},
            ...     subplot_settings={
            ...         'price': {'title': 'Price of ${symbol} (${period})'}
            ...     }
            ... )
        
        注意事项：
            - PlotsBuilderMixin与StatsBuilderMixin设计相似，概念对应关系：
              plots_defaults ↔ stats_defaults
              subplots ↔ metrics  
              subplot_settings ↔ metric_settings
            - 布局相关的解析参数（如add_trace_kwargs）在过滤之前不可用，
              不能在模板中使用但可以被覆盖
            - 子图绘制过程中的异常会被捕获并重新抛出，便于调试
        """
        from vectorbt._settings import settings as _settings  # 导入vectorbt全局设置
        plotting_cfg = _settings['plotting']  # 获取绘图相关配置

        # 解析默认值：将None参数替换为默认配置中的值
        # 这种设计允许用户选择性地覆盖特定参数，而保持其他参数使用默认值
        if silence_warnings is None:
            silence_warnings = self.plots_defaults['silence_warnings']  # 是否静默警告
        if show_titles is None:
            show_titles = self.plots_defaults['show_titles']  # 是否显示子图标题
        if hide_id_labels is None:
            hide_id_labels = self.plots_defaults['hide_id_labels']  # 是否隐藏重复图例标签
        if group_id_labels is None:
            group_id_labels = self.plots_defaults['group_id_labels']  # 是否分组相同图例标签
            
        # 合并配置字典：将默认配置与用户提供的配置合并，用户配置优先
        # merge_dicts函数会深度合并字典，确保嵌套配置也能正确合并
        template_mapping = merge_dicts(self.plots_defaults['template_mapping'], template_mapping)  # 模板参数映射
        filters = merge_dicts(self.plots_defaults['filters'], filters)  # 过滤器配置
        settings = merge_dicts(self.plots_defaults['settings'], settings)  # 全局设置参数
        subplot_settings = merge_dicts(self.plots_defaults['subplot_settings'], subplot_settings)  # 子图特定设置
        make_subplots_kwargs = merge_dicts(self.plots_defaults['make_subplots_kwargs'], make_subplots_kwargs)  # make_subplots参数
        layout_kwargs = merge_dicts(self.plots_defaults['layout_kwargs'], layout_kwargs)  # 布局参数

        # 全局模板替换：在子图级别之前进行全局配置的模板参数替换
        # 这一步处理的是应用于整个图表的模板参数，不包括子图特定的模板
        if len(template_mapping) > 0:  # 如果存在模板参数映射
            # 使用deep_substitute函数递归替换配置中的模板占位符
            sub_settings = deep_substitute(settings, mapping=template_mapping)  # 替换全局设置中的模板
            sub_make_subplots_kwargs = deep_substitute(make_subplots_kwargs, mapping=template_mapping)  # 替换子图创建参数中的模板
            sub_layout_kwargs = deep_substitute(layout_kwargs, mapping=template_mapping)  # 替换布局参数中的模板
        else:
            # 如果没有模板参数，直接使用原始配置
            sub_settings = settings
            sub_make_subplots_kwargs = make_subplots_kwargs
            sub_layout_kwargs = layout_kwargs

        # 解析self对象：创建一个解析后的self副本，用于后续的属性访问和方法调用
        # 这个步骤确保后续操作使用的是经过条件配置解析的对象实例
        reself = self.resolve_self(
            cond_kwargs=sub_settings,  # 传入解析后的设置作为条件参数
            impacts_caching=False,  # 不影响缓存，因为这是临时解析
            silence_warnings=silence_warnings  # 传递警告控制参数
        )

        # 准备子图配置：标准化子图参数为统一的格式
        # 支持多种输入格式，最终转换为 [(子图名称, 配置字典), ...] 的列表格式
        if subplots is None:
            subplots = reself.plots_defaults['subplots']  # 使用默认子图配置
        if subplots == 'all':
            subplots = reself.subplots  # 使用所有可用的子图
        if isinstance(subplots, dict):
            subplots = list(subplots.items())  # 将字典转换为键值对列表
        if isinstance(subplots, (str, tuple)):
            subplots = [subplots]  # 将单个子图转换为列表

        # 准备标签过滤条件：标准化标签参数为统一的格式
        # 支持单个标签、标签列表或特殊值'all'
        if tags is None:
            tags = reself.plots_defaults['tags']  # 使用默认标签配置
        if isinstance(tags, str) and tags == 'all':
            tags = None  # 'all'表示不进行标签过滤
        if isinstance(tags, (str, tuple)):
            tags = [tags]  # 将单个标签或元组转换为列表

        # 统一子图格式：将所有子图转换为 (名称, 配置) 元组格式
        # 这一步确保后续处理可以统一处理所有子图，无论原始输入格式如何
        new_subplots = []
        for i, subplot in enumerate(subplots):
            if isinstance(subplot, str):
                # 字符串格式：从类配置中获取对应的子图配置
                subplot = (subplot, reself.subplots[subplot])
            if not isinstance(subplot, tuple):
                # 类型检查：确保每个子图都是有效的格式
                raise TypeError(f"Subplot at index {i} must be either a string or a tuple")
            new_subplots.append(subplot)
        subplots = new_subplots

        # 处理重复的子图名称：为重复的子图名称添加数字后缀
        # 这允许用户多次使用同一个子图类型但使用不同的配置
        subplot_counts = Counter(list(map(lambda x: x[0], subplots)))  # 统计每个子图名称的出现次数
        subplot_i = {k: -1 for k in subplot_counts.keys()}  # 为每个子图名称初始化计数器
        subplots_dct = {}  # 最终的子图配置字典
        for i, (subplot_name, _subplot_settings) in enumerate(subplots):
            if subplot_counts[subplot_name] > 1:
                # 如果子图名称重复，添加数字后缀
                subplot_i[subplot_name] += 1  # 增加计数器
                subplot_name = subplot_name + '_' + str(subplot_i[subplot_name])  # 添加后缀
            subplots_dct[subplot_name] = _subplot_settings  # 存储子图配置

        # 检查子图设置的有效性：确保subplot_settings中的所有键都对应实际的子图
        # 这个检查可以帮助用户发现配置错误，避免静默忽略无效的设置
        missed_keys = set(subplot_settings.keys()).difference(set(subplots_dct.keys()))
        if len(missed_keys) > 0:
            raise ValueError(f"Keys {missed_keys} in subplot_settings could not be matched with any subplot")

        # 合并设置并准备解析：为每个子图创建完整的配置和解析环境
        # 这一步是整个绘图流程的核心，为每个子图准备所有必要的参数和上下文
        opt_arg_names_dct = {}  # 存储每个子图的可选参数名称集合
        custom_arg_names_dct = {}  # 存储每个子图的自定义参数名称集合
        resolved_self_dct = {}  # 存储每个子图的解析后的self对象
        mapping_dct = {}  # 存储每个子图的模板参数映射
        for subplot_name, _subplot_settings in list(subplots_dct.items()):
            # 构建可选设置字典：包含所有可能传递给绘图函数的参数
            # 这些参数按优先级排列：self别名 < 基础参数 < 全局设置
            opt_settings = merge_dicts(
                {name: reself for name in reself.self_aliases},  # self的各种别名，用于属性解析
                dict(
                    column=column,  # 列选择参数
                    group_by=group_by,  # 分组参数
                    subplot_name=subplot_name,  # 当前子图名称
                    trace_names=[subplot_name],  # 轨迹名称列表，用于图例
                    silence_warnings=silence_warnings  # 警告控制参数
                ),
                settings  # 全局设置参数
            )
            
            # 复制子图设置以避免修改原始配置
            _subplot_settings = _subplot_settings.copy()
            
            # 获取该子图的特定设置（如果有的话）
            passed_subplot_settings = subplot_settings.get(subplot_name, {})
            
            # 合并所有设置：可选设置 < 子图默认设置 < 用户传递的子图设置
            # 这种优先级确保用户的设置能够覆盖默认配置
            merged_settings = merge_dicts(
                opt_settings,  # 可选参数（优先级最低）
                _subplot_settings,  # 子图默认配置
                passed_subplot_settings  # 用户传递的子图设置（优先级最高）
            )
            
            # 处理子图级别的模板参数：提取并合并模板映射
            subplot_template_mapping = merged_settings.pop('template_mapping', {})  # 提取子图模板映射
            template_mapping_merged = merge_dicts(template_mapping, subplot_template_mapping)  # 合并全局和子图模板
            template_mapping_merged = deep_substitute(template_mapping_merged, mapping=merged_settings)  # 使用当前设置解析模板
            mapping = merge_dicts(template_mapping_merged, merged_settings)  # 创建最终的参数映射
            
            # 安全的模板替换：使用safe=True因为稍后会再次进行深度替换
            # 这里只是预处理，真正的替换会在布局参数已知后进行
            merged_settings = deep_substitute(merged_settings, mapping=mapping, safe=True)

            # 标签过滤：根据tags参数决定是否包含当前子图
            # 这是一个重要的过滤机制，允许用户选择性地显示特定类型的子图
            if tags is not None:
                in_tags = merged_settings.get('tags', None)  # 获取子图的标签列表
                # 如果子图没有标签或标签不匹配，则跳过该子图
                if in_tags is None or not match_tags(tags, in_tags):
                    subplots_dct.pop(subplot_name, None)  # 从子图字典中移除
                    continue  # 跳过后续处理，继续下一个子图

            # 收集参数名称：区分自定义参数和可选参数
            # 这有助于后续的参数解析和传递过程
            custom_arg_names = set(_subplot_settings.keys()).union(set(passed_subplot_settings.keys()))  # 自定义参数名称
            opt_arg_names = set(opt_settings.keys())  # 可选参数名称
            
            # 解析自定义的self对象：为每个子图创建专门的解析上下文
            # 这确保每个子图都有正确的数据访问环境
            custom_reself = reself.resolve_self(
                cond_kwargs=merged_settings,  # 使用合并后的设置作为条件参数
                custom_arg_names=custom_arg_names,  # 传递自定义参数名称
                impacts_caching=True,  # 影响缓存，因为这是子图特定的解析
                silence_warnings=merged_settings['silence_warnings']  # 使用子图的警告设置
            )

            # 存储子图的所有配置和解析结果
            # 这些数据将在后续的过滤和绘制阶段使用
            subplots_dct[subplot_name] = merged_settings  # 合并后的设置
            custom_arg_names_dct[subplot_name] = custom_arg_names  # 自定义参数名称集合
            opt_arg_names_dct[subplot_name] = opt_arg_names  # 可选参数名称集合
            resolved_self_dct[subplot_name] = custom_reself  # 解析后的self对象
            mapping_dct[subplot_name] = mapping  # 模板参数映射

        # 过滤子图：根据配置的过滤器条件进一步筛选子图
        # 这是一个高级过滤机制，允许基于复杂的条件动态决定是否显示子图
        for subplot_name, _subplot_settings in list(subplots_dct.items()):
            custom_reself = resolved_self_dct[subplot_name]  # 获取该子图的解析后self对象
            mapping = mapping_dct[subplot_name]  # 获取该子图的模板参数映射
            _silence_warnings = _subplot_settings.get('silence_warnings')  # 获取警告控制设置

            # 收集子图需要的所有过滤器：扫描配置键找出所有过滤器引用
            subplot_filters = set()
            for k in _subplot_settings.keys():
                filter_name = None
                if k.startswith('check_'):
                    # 正向过滤器：check_{filter_name} = True 表示必须满足条件
                    filter_name = k[len('check_'):]
                elif k.startswith('inv_check_'):
                    # 反向过滤器：inv_check_{filter_name} = True 表示必须不满足条件
                    filter_name = k[len('inv_check_'):]
                if filter_name is not None:
                    # 验证过滤器是否在配置中定义
                    if filter_name not in filters:
                        raise ValueError(f"Metric '{subplot_name}' requires filter '{filter_name}'")
                    subplot_filters.add(filter_name)

            # 逐个执行过滤器检查
            for filter_name in subplot_filters:
                filter_settings = filters[filter_name]  # 获取过滤器配置
                _filter_settings = deep_substitute(filter_settings, mapping=mapping)  # 应用模板替换
                filter_func = _filter_settings['filter_func']  # 获取过滤器函数
                warning_message = _filter_settings.get('warning_message', None)  # 获取警告消息
                inv_warning_message = _filter_settings.get('inv_warning_message', None)  # 获取反向警告消息
                to_check = _subplot_settings.get('check_' + filter_name, False)  # 是否进行正向检查
                inv_to_check = _subplot_settings.get('inv_check_' + filter_name, False)  # 是否进行反向检查

                # 执行过滤器检查
                if to_check or inv_to_check:
                    whether_true = filter_func(custom_reself, _subplot_settings)  # 执行过滤器函数
                    # 确定是否需要移除子图：
                    # - 正向检查失败：要求为True但结果为False
                    # - 反向检查失败：要求为False但结果为True
                    to_remove = (to_check and not whether_true) or (inv_to_check and whether_true)
                    
                    if to_remove:
                        # 发出适当的警告消息
                        if to_check and warning_message is not None and not _silence_warnings:
                            warnings.warn(warning_message)
                        if inv_to_check and inv_warning_message is not None and not _silence_warnings:
                            warnings.warn(inv_warning_message)

                        # 从所有相关字典中移除该子图
                        subplots_dct.pop(subplot_name, None)
                        custom_arg_names_dct.pop(subplot_name, None)
                        opt_arg_names_dct.pop(subplot_name, None)
                        resolved_self_dct.pop(subplot_name, None)
                        mapping_dct.pop(subplot_name, None)
                        break  # 一旦决定移除，就不需要检查其他过滤器了

        # 检查是否还有子图需要绘制：经过所有过滤后可能没有子图剩余
        if len(subplots_dct) == 0:
            if not silence_warnings:
                warnings.warn("No subplots to plot", stacklevel=2)
            return None  # 没有子图可绘制，返回None

        # 设置图形布局：计算子图的行列布局和规格
        # 这一步确定了整个图表的基本结构和尺寸
        rows = sub_make_subplots_kwargs.pop('rows', len(subplots_dct))  # 行数，默认为子图数量（垂直排列）
        cols = sub_make_subplots_kwargs.pop('cols', 1)  # 列数，默认为1（单列布局）
        specs = sub_make_subplots_kwargs.pop('specs', [[{} for _ in range(cols)] for _ in range(rows)])  # 子图规格，默认为标准网格
        # 计算行列位置映射：确定每个子图在网格中的实际位置
        row_col_tuples = []
        for row, row_spec in enumerate(specs):
            for col, col_spec in enumerate(row_spec):
                if col_spec is not None:  # 只有非空的规格才对应实际的子图位置
                    row_col_tuples.append((row + 1, col + 1))  # Plotly使用1基索引
                    
        # 坐标轴共享设置：决定子图间是否共享X/Y轴
        shared_xaxes = sub_make_subplots_kwargs.pop('shared_xaxes', True)  # 默认共享X轴
        shared_yaxes = sub_make_subplots_kwargs.pop('shared_yaxes', False)  # 默认不共享Y轴
        
        # 布局尺寸常量：定义各种间距和尺寸的基准值
        default_height = plotting_cfg['layout']['height']  # 从全局配置获取默认高度
        default_width = plotting_cfg['layout']['width'] + 50  # 从全局配置获取默认宽度并增加边距
        min_space = 10  # 子图间的最小间距（像素）
        max_title_spacing = 30  # 标题占用的最大空间（像素）
        max_xaxis_spacing = 50  # X轴标签占用的最大空间（像素）
        max_yaxis_spacing = 100  # Y轴标签占用的最大空间（像素）
        legend_height = 50  # 图例占用的高度（像素）
        # 计算实际间距：根据配置决定各种间距的实际值
        if show_titles:
            title_spacing = max_title_spacing  # 显示标题时需要预留标题空间
        else:
            title_spacing = 0  # 不显示标题时无需额外空间
            
        if not shared_xaxes and rows > 1:
            xaxis_spacing = max_xaxis_spacing  # 不共享X轴且多行时需要为每行的X轴标签预留空间
        else:
            xaxis_spacing = 0  # 共享X轴或单行时无需额外空间
            
        if not shared_yaxes and cols > 1:
            yaxis_spacing = max_yaxis_spacing  # 不共享Y轴且多列时需要为每列的Y轴标签预留空间
        else:
            yaxis_spacing = 0  # 共享Y轴或单列时无需额外空间

        # 计算图表高度：优先使用用户指定的高度，否则自动计算
        if 'height' in sub_layout_kwargs:
            height = sub_layout_kwargs.pop('height')  # 使用用户指定的高度
        else:
            # 自动计算高度：基础高度 + 标题空间 + 多行调整
            height = default_height + title_spacing
            if rows > 1:
                height *= rows  # 多行时高度按行数倍增
                height += min_space * rows - min_space  # 添加行间间距（n行需要n-1个间距）
                height += legend_height - legend_height * rows  # 调整图例高度（多行时图例相对占用更少空间）
                if shared_xaxes:
                    # 共享X轴时，只有最后一行需要X轴标签空间
                    height += max_xaxis_spacing - max_xaxis_spacing * rows
                    
        # 计算图表宽度：优先使用用户指定的宽度，否则自动计算
        if 'width' in sub_layout_kwargs:
            width = sub_layout_kwargs.pop('width')  # 使用用户指定的宽度
        else:
            # 自动计算宽度：基础宽度 + 多列调整
            width = default_width
            if cols > 1:
                width *= cols  # 多列时宽度按列数倍增
                width += min_space * cols - min_space  # 添加列间间距（n列需要n-1个间距）
                if shared_yaxes:
                    # 共享Y轴时，只有第一列需要Y轴标签空间
                    width += max_yaxis_spacing - max_yaxis_spacing * cols
        # 计算垂直间距：确定子图间的垂直间距
        if height is not None:
            if 'vertical_spacing' in sub_make_subplots_kwargs:
                vertical_spacing = sub_make_subplots_kwargs.pop('vertical_spacing')  # 使用用户指定的垂直间距
            else:
                # 自动计算垂直间距：基础间距 + 标题空间 + X轴空间
                vertical_spacing = min_space + title_spacing + xaxis_spacing
            # 将像素值转换为相对比例（Plotly要求0-1之间的值）
            if vertical_spacing is not None and vertical_spacing > 1:
                vertical_spacing /= height
            # 计算图例的Y位置：位于图表上方
            legend_y = 1 + (min_space + title_spacing) / height
        else:
            # 高度未指定时使用默认间距
            vertical_spacing = sub_make_subplots_kwargs.pop('vertical_spacing', None)
            legend_y = 1.02  # 默认图例位置
            
        # 计算水平间距：确定子图间的水平间距
        if width is not None:
            if 'horizontal_spacing' in sub_make_subplots_kwargs:
                horizontal_spacing = sub_make_subplots_kwargs.pop('horizontal_spacing')  # 使用用户指定的水平间距
            else:
                # 自动计算水平间距：基础间距 + Y轴空间
                horizontal_spacing = min_space + yaxis_spacing
            # 将像素值转换为相对比例（Plotly要求0-1之间的值）
            if horizontal_spacing is not None and horizontal_spacing > 1:
                horizontal_spacing /= width
        else:
            # 宽度未指定时使用默认间距
            horizontal_spacing = sub_make_subplots_kwargs.pop('horizontal_spacing', None)
            
        # 准备子图标题：为每个子图创建占位符标题
        if show_titles:
            _subplot_titles = []
            for i in range(len(subplots_dct)):
                _subplot_titles.append('$title_' + str(i))  # 使用占位符，稍后替换为实际标题
        else:
            _subplot_titles = None  # 不显示标题
        # 创建多子图布局：使用Plotly的make_subplots函数创建基础图形结构
        fig = make_subplots(
            rows=rows,  # 子图行数
            cols=cols,  # 子图列数
            specs=specs,  # 子图规格定义
            shared_xaxes=shared_xaxes,  # X轴共享设置
            shared_yaxes=shared_yaxes,  # Y轴共享设置
            subplot_titles=_subplot_titles,  # 子图标题列表
            vertical_spacing=vertical_spacing,  # 垂直间距
            horizontal_spacing=horizontal_spacing,  # 水平间距
            **sub_make_subplots_kwargs  # 其他用户指定的参数
        )
        
        # 合并布局参数：设置图形的整体布局属性
        sub_layout_kwargs = merge_dicts(dict(
            showlegend=True,  # 显示图例
            width=width,  # 图形宽度
            height=height,  # 图形高度
            legend=dict(  # 图例配置
                orientation="h",  # 水平方向排列
                yanchor="bottom",  # Y轴锚点在底部
                y=legend_y,  # Y位置（相对值）
                xanchor="right",  # X轴锚点在右侧
                x=1,  # X位置（相对值，1表示最右侧）
                traceorder='normal'  # 图例项的排序方式
            )
        ), sub_layout_kwargs)  # 合并用户指定的布局参数
        
        # 应用布局设置：更新图形的布局配置
        fig.update_layout(**sub_layout_kwargs)  # sub_layout_kwargs的最终应用位置

        # 绘制子图：逐个处理每个子图的绘制
        # 这是整个绘图流程的核心执行阶段
        arg_cache_dct = {}  # 参数缓存字典，用于优化重复的属性解析
        for i, (subplot_name, _subplot_settings) in enumerate(subplots_dct.items()):
            try:
                # 准备子图绘制的参数：复制配置并获取相关的解析对象
                final_kwargs = _subplot_settings.copy()  # 复制子图设置作为最终参数
                opt_arg_names = opt_arg_names_dct[subplot_name]  # 可选参数名称集合
                custom_arg_names = custom_arg_names_dct[subplot_name]  # 自定义参数名称集合
                custom_reself = resolved_self_dct[subplot_name]  # 解析后的self对象
                mapping = mapping_dct[subplot_name]  # 模板参数映射

                # 计算图形相关参数：确定子图在整个图形中的位置和引用
                row, col = row_col_tuples[i]  # 获取子图的行列位置
                xref = 'x' if i == 0 else 'x' + str(i + 1)  # X轴引用名称（第一个子图为'x'，其他为'x2', 'x3'等）
                yref = 'y' if i == 0 else 'y' + str(i + 1)  # Y轴引用名称（第一个子图为'y'，其他为'y2', 'y3'等）
                xaxis = 'xaxis' + xref[1:]  # X轴配置键名（'xaxis', 'xaxis2', 'xaxis3'等）
                yaxis = 'yaxis' + yref[1:]  # Y轴配置键名（'yaxis', 'yaxis2', 'yaxis3'等）
                x_domain = get_domain(xref, fig)  # 获取X轴的域范围（0-1之间的相对位置）
                y_domain = get_domain(yref, fig)  # 获取Y轴的域范围（0-1之间的相对位置）
                
                # 构建子图布局参数：包含绘图函数需要的所有布局相关信息
                subplot_layout_kwargs = dict(
                    add_trace_kwargs=dict(row=row, col=col),  # 添加轨迹时需要的行列参数
                    xref=xref,  # X轴引用
                    yref=yref,  # Y轴引用
                    xaxis=xaxis,  # X轴配置键
                    yaxis=yaxis,  # Y轴配置键
                    x_domain=x_domain,  # X轴域范围
                    y_domain=y_domain,  # Y轴域范围
                    fig=fig,  # Plotly图形对象
                    pass_fig=True  # 强制传递fig参数
                )
                
                # 更新参数名称集合：将布局参数添加到相应的集合中
                for k in subplot_layout_kwargs:
                    opt_arg_names.add(k)  # 添加到可选参数集合
                    if k in final_kwargs:
                        custom_arg_names.add(k)  # 如果在最终参数中也存在，添加到自定义参数集合
                        
                # 合并参数：将布局参数与子图设置合并
                final_kwargs = merge_dicts(subplot_layout_kwargs, final_kwargs)  # 布局参数优先级较低
                mapping = merge_dicts(subplot_layout_kwargs, mapping)  # 同时更新模板映射
                
                # 最终的模板替换：现在布局参数已知，可以进行完整的模板替换
                final_kwargs = deep_substitute(final_kwargs, mapping=mapping)

                # 清理配置键：移除不需要传递给绘图函数的内部配置键
                for k, v in list(final_kwargs.items()):
                    if k.startswith('check_') or k.startswith('inv_check_') or k in ('tags',):
                        final_kwargs.pop(k, None)  # 移除过滤器相关的键和标签键

                # 提取子图特定的配置值：从最终参数中提取绘图控制参数
                _column = final_kwargs.get('column')  # 列选择参数（保留用于信息）
                _group_by = final_kwargs.get('group_by')  # 分组参数（保留用于信息）
                _silence_warnings = final_kwargs.get('silence_warnings')  # 警告控制参数
                title = final_kwargs.pop('title', subplot_name)  # 子图标题，默认使用子图名称
                plot_func = final_kwargs.pop('plot_func', None)  # 绘图函数，必需参数
                xaxis_kwargs = final_kwargs.pop('xaxis_kwargs', None)  # X轴配置参数
                yaxis_kwargs = final_kwargs.pop('yaxis_kwargs', None)  # Y轴配置参数
                resolve_plot_func = final_kwargs.pop('resolve_plot_func', True)  # 是否解析绘图函数
                use_caching = final_kwargs.pop('use_caching', True)  # 是否使用缓存

                # 执行绘图函数：这是子图实际渲染的核心步骤
                if plot_func is not None:
                    # 解析绘图函数：将字符串函数名转换为可调用对象并处理参数
                    if resolve_plot_func:
                        if not callable(plot_func):
                            # 如果绘图函数不是可调用对象，需要从对象中解析获取
                            passed_kwargs_out = {}  # 用于收集解析过程中传递的参数

                            def _getattr_func(obj: tp.Any,
                                              attr: str,
                                              args: tp.ArgsLike = None,
                                              kwargs: tp.KwargsLike = None,
                                              call_attr: bool = True,
                                              _final_kwargs: tp.Kwargs = final_kwargs,
                                              _opt_arg_names: tp.Set[str] = opt_arg_names,
                                              _custom_arg_names: tp.Set[str] = custom_arg_names,
                                              _arg_cache_dct: tp.Kwargs = arg_cache_dct) -> tp.Any:
                                """
                                自定义属性获取函数：用于在解析绘图函数时控制属性访问行为
                                
                                这个函数提供了一个灵活的属性解析机制，支持：
                                - 优先从final_kwargs中获取属性值
                                - 对custom_reself对象使用特殊的属性解析逻辑
                                - 缓存重复的属性解析结果以提高性能
                                - 控制是否调用可调用属性
                                """
                                if attr in final_kwargs:
                                    return final_kwargs[attr]  # 优先从最终参数中获取属性值
                                if args is None:
                                    args = ()
                                if kwargs is None:
                                    kwargs = {}
                                if obj is custom_reself and _final_kwargs.pop('resolve_path_' + attr, True):
                                    # 对于custom_reself对象，使用特殊的属性解析逻辑
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
                                # 对于其他对象，使用标准的属性获取
                                out = getattr(obj, attr)
                                if callable(out) and call_attr:
                                    return out(*args, **kwargs)  # 如果是可调用对象且需要调用，则执行调用
                                return out

                            # 使用深度属性获取来解析绘图函数
                            plot_func = custom_reself.deep_getattr(
                                plot_func,  # 要解析的属性路径（可能是嵌套的）
                                getattr_func=_getattr_func,  # 自定义的属性获取函数
                                call_last_attr=False  # 不调用最后的属性（因为我们要获取函数本身）
                            )

                            # 处理group_by参数的特殊逻辑
                            if 'group_by' in passed_kwargs_out:
                                if 'pass_group_by' not in final_kwargs:
                                    final_kwargs.pop('group_by', None)  # 如果没有明确要求传递group_by，则移除它
                                    
                        # 验证绘图函数的有效性
                        if not callable(plot_func):
                            raise TypeError("plot_func must be callable")

                        # 解析绘图函数的参数：自动解析函数需要的参数
                        func_arg_names = get_func_arg_names(plot_func)  # 获取函数的参数名列表
                        for k in func_arg_names:
                            if k not in final_kwargs:  # 如果参数不在最终参数中
                                if final_kwargs.pop('resolve_' + k, False):  # 检查是否需要解析该参数
                                    try:
                                        # 尝试从对象中解析该属性
                                        arg_out = custom_reself.resolve_attr(
                                            k,
                                            cond_kwargs=final_kwargs,
                                            custom_arg_names=custom_arg_names,
                                            cache_dct=arg_cache_dct,
                                            use_caching=use_caching
                                        )
                                    except AttributeError:
                                        continue  # 如果属性不存在，跳过
                                    final_kwargs[k] = arg_out  # 将解析的属性添加到最终参数中
                                    
                        # 清理不需要的可选参数：移除函数不需要的可选参数
                        for k in list(final_kwargs.keys()):
                            if k in opt_arg_names:  # 如果是可选参数
                                if 'pass_' + k in final_kwargs:
                                    if not final_kwargs.get('pass_' + k):  # 第一优先级：明确设置不传递
                                        final_kwargs.pop(k, None)
                                elif k not in func_arg_names:  # 第二优先级：函数不需要该参数
                                    final_kwargs.pop(k, None)
                                    
                        # 清理控制参数：移除以'pass_'或'resolve_'开头的控制参数
                        for k in list(final_kwargs.keys()):
                            if k.startswith('pass_') or k.startswith('resolve_'):
                                final_kwargs.pop(k, None)  # 清理控制参数

                        # 调用绘图函数：执行实际的绘图操作
                        plot_func(**final_kwargs)
                    else:
                        # 不解析绘图函数：直接调用，传递原始参数
                        plot_func(custom_reself, _subplot_settings)

                # 更新全局布局：设置子图的标题和坐标轴
                for annotation in fig.layout.annotations:
                    # 替换占位符标题为实际标题
                    if 'text' in annotation and annotation['text'] == '$title_' + str(i):
                        annotation['text'] = title
                        
                # 配置子图的坐标轴
                subplot_layout = dict()
                subplot_layout[xaxis] = merge_dicts(dict(title='Index'), xaxis_kwargs)  # X轴配置，默认标题为'Index'
                subplot_layout[yaxis] = merge_dicts(dict(), yaxis_kwargs)  # Y轴配置
                fig.update_layout(**subplot_layout)  # 应用坐标轴配置
                
            except Exception as e:
                # 异常处理：捕获子图绘制过程中的异常
                warnings.warn(f"Subplot '{subplot_name}' raised an exception", stacklevel=2)
                raise e  # 重新抛出异常以便调试

        # 移除重复的图例标签：优化图例显示，避免重复标签
        # 这是一个重要的后处理步骤，提升图表的可读性
        found_ids = dict()  # 存储已发现的图例ID
        unique_idx = 0  # 唯一索引计数器
        for trace in fig.data:
            # 提取轨迹的标识信息：名称、标记样式、线条样式等
            if 'name' in trace:
                name = trace['name']  # 轨迹名称
            else:
                name = None
            if 'marker' in trace:
                marker = trace['marker']  # 标记配置
            else:
                marker = {}
            if 'symbol' in marker:
                marker_symbol = marker['symbol']  # 标记符号
            else:
                marker_symbol = None
            if 'color' in marker:
                marker_color = marker['color']  # 标记颜色
            else:
                marker_color = None
            if 'line' in trace:
                line = trace['line']  # 线条配置
            else:
                line = {}
            if 'dash' in line:
                line_dash = line['dash']  # 线条虚线样式
            else:
                line_dash = None
            if 'color' in line:
                line_color = line['color']  # 线条颜色
            else:
                line_color = None

            # 创建轨迹的唯一标识：基于所有视觉属性
            id = (name, marker_symbol, marker_color, line_dash, line_color)
            if id in found_ids:
                # 如果已经存在相同的标识，处理重复
                if hide_id_labels:
                    trace['showlegend'] = False  # 隐藏重复的图例标签
                if group_id_labels:
                    trace['legendgroup'] = found_ids[id]  # 将相同标识的轨迹分组
            else:
                # 如果是新的标识，记录并分配唯一索引
                if group_id_labels:
                    trace['legendgroup'] = unique_idx  # 分配图例分组
                found_ids[id] = unique_idx  # 记录标识
                unique_idx += 1  # 增加唯一索引

        # 移除共享坐标轴时多余的标题：优化共享坐标轴的显示
        # 当坐标轴共享时，只在边缘显示标题，避免重复
        if shared_xaxes:
            # 处理共享X轴：只在最后一行显示X轴标题
            i = 0
            for row in range(rows):
                for col in range(cols):
                    if specs[row][col] is not None:
                        xaxis = 'xaxis' if i == 0 else 'xaxis' + str(i + 1)
                        if row < rows - 1:  # 如果不是最后一行
                            fig.layout[xaxis]['title'] = None  # 移除X轴标题
                        i += 1
                        
        if shared_yaxes:
            # 处理共享Y轴：只在第一列显示Y轴标题
            i = 0
            for row in range(rows):
                for col in range(cols):
                    if specs[row][col] is not None:
                        yaxis = 'yaxis' if i == 0 else 'yaxis' + str(i + 1)
                        if col > 0:  # 如果不是第一列
                            fig.layout[yaxis]['title'] = None  # 移除Y轴标题
                        i += 1

        # 返回完成的图形对象
        return fig

    # ############# 文档生成相关方法 ############# #

    @classmethod
    def build_subplots_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """
        构建子图配置的文档字符串
        
        这个类方法用于动态生成子图配置的文档，通过模板替换的方式
        将类的实际子图配置插入到文档模板中。
        
        Args:
            source_cls: 源类，用于获取文档模板，默认为PlotsBuilderMixin
            
        Returns:
            str: 格式化后的子图配置文档字符串
        """
        if source_cls is None:
            source_cls = PlotsBuilderMixin
        return string.Template(
            inspect.cleandoc(get_dict_attr(source_cls, 'subplots').__doc__)
        ).substitute(
            {'subplots': cls.subplots.to_doc(), 'cls_name': cls.__name__}
        )

    @classmethod
    def override_subplots_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """
        重写子图配置的文档
        
        这个方法应该在每个重写了`subplots`配置的子类中调用，
        用于更新文档系统中的子图配置文档。
        
        Args:
            __pdoc__: 文档字典，用于存储类的文档字符串
            source_cls: 源类，用于获取文档模板，默认为PlotsBuilderMixin
        """
        __pdoc__[cls.__name__ + '.subplots'] = cls.build_subplots_doc(source_cls=source_cls)


# 全局文档字典：用于存储动态生成的文档
__pdoc__ = dict()
# 为PlotsBuilderMixin类重写子图文档
PlotsBuilderMixin.override_subplots_doc(__pdoc__)
