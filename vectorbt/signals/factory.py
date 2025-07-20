# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
信号工厂模块 - 用于轻松构建新的信号生成器

设计逻辑与作用总结：
===================

该模块是vectorbt量化交易框架中的核心信号生成组件，主要功能包括：

1. **信号工厂设计模式**：
   - 采用工厂模式设计，允许用户通过简单的配置创建复杂的信号生成器
   - 继承自IndicatorFactory，扩展了选择函数功能
   - 支持多种信号生成模式：仅入场信号、仅出场信号、同时生成入场和出场信号、链式信号处理

2. **核心功能**：
   - 通过提供入场和出场函数以及输入、参数、输出名称等信息，创建独立的信号生成类
   - 支持任意复杂度的信号生成逻辑
   - 提供灵活的参数组合和批量处理能力
   - 集成Numba加速，支持高性能信号计算

3. **应用场景**：
   - 技术指标信号生成（如MACD、RSI等交叉信号）
   - 自定义交易策略信号
   - 多信号组合和过滤
   - 回测系统中的信号模拟

4. **架构特点**：
   - 模块化设计，易于扩展和维护
   - 支持缓存机制，提高重复计算效率
   - 提供可视化功能，便于信号分析和调试
   - 与vectorbt其他组件无缝集成

使用示例：
---------
```python
# 创建简单的交叉信号生成器
from vectorbt.signals.factory import SignalFactory
from numba import njit
import numpy as np

@njit
def crossover_entry(close, ma1, ma2, from_i, to_i, col):
    # 当短期均线上穿长期均线时产生入场信号
    if from_i > 0 and ma1[from_i-1, col] <= ma2[from_i-1, col] and ma1[from_i, col] > ma2[from_i, col]:
        return np.array([from_i])
    return np.array([], dtype=np.int64)

@njit  
def crossover_exit(close, ma1, ma2, from_i, to_i, col):
    # 当短期均线下穿长期均线时产生出场信号
    if from_i > 0 and ma1[from_i-1, col] >= ma2[from_i-1, col] and ma1[from_i, col] < ma2[from_i, col]:
        return np.array([from_i])
    return np.array([], dtype=np.int64)

# 构建信号生成器
CrossoverSignals = SignalFactory(
    mode='both',  # 同时生成入场和出场信号
    input_names=['close', 'ma1', 'ma2'],  # 输入数据名称
    param_names=['short_period', 'long_period']  # 参数名称
).from_choice_func(
    entry_choice_func=crossover_entry,
    exit_choice_func=crossover_exit,
    entry_settings=dict(
        pass_inputs=['close', 'ma1', 'ma2']  # 传递给入场函数的输入
    ),
    exit_settings=dict(
        pass_inputs=['close', 'ma1', 'ma2']  # 传递给出场函数的输入
    )
)

# 使用信号生成器
close_prices = np.array([100, 101, 102, 103, 104])
ma1 = np.array([99, 100, 101, 102, 103])  # 短期均线
ma2 = np.array([100, 100, 100, 100, 100])  # 长期均线

signals = CrossoverSignals.run(close_prices, ma1, ma2)
print("入场信号:", signals.entries)
print("出场信号:", signals.exits)
```

The signal factory class `SignalFactory` extends `vectorbt.indicators.factory.IndicatorFactory`
to offer a convenient way to create signal generators of any complexity. By providing it with information
such as entry and exit functions and the names of inputs, parameters, and outputs, it will create a
stand-alone class capable of generating signals for an arbitrary combination of inputs and parameters.
"""

import inspect

import numpy as np
from numba import njit

from vectorbt import _typing as tp
from vectorbt.base import combine_fns
from vectorbt.indicators.factory import IndicatorFactory, IndicatorBase, CacheOutputT
from vectorbt.signals.enums import FactoryMode
from vectorbt.signals.nb import (
    generate_nb,
    generate_ex_nb,
    generate_enex_nb,
    first_choice_nb
)
from vectorbt.utils import checks
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.enum_ import map_enum_fields
from vectorbt.utils.params import to_typed_list


class SignalFactory(IndicatorFactory):
    """
    信号工厂类 - 用于构建信号生成器的工厂类
    
    继承自 `vectorbt.indicators.factory.IndicatorFactory`，扩展了选择函数功能。
    根据指定的模式生成固定数量的输出。如果需要生成其他输出，可以使用就地输出（通过 `in_output_names`）。
    
    支持的生成模式请参考 `vectorbt.signals.enums.FactoryMode`。
    其他参数会传递给 `vectorbt.indicators.factory.IndicatorFactory`。
    
    主要特性：
    - 支持多种信号生成模式（仅入场、仅出场、同时生成、链式处理）
    - 提供灵活的参数配置和输入输出管理
    - 集成Numba加速，支持高性能计算
    - 内置可视化功能，便于信号分析
    
    使用示例：
    ```python
    # 创建简单的信号工厂
    factory = SignalFactory(
        mode='both',  # 同时生成入场和出场信号
        input_names=['price', 'volume'],  # 输入数据名称
        param_names=['threshold']  # 参数名称
    )
    
    # 使用工厂创建信号生成器
    MySignals = factory.from_choice_func(
        entry_choice_func=my_entry_func,
        exit_choice_func=my_exit_func
    )
    
    # 运行信号生成器
    signals = MySignals.run(price_data, volume_data, threshold=0.5)
    ```
    """

    def __init__(self,
                 *args,
                 mode: tp.Union[str, int] = FactoryMode.Both,
                 input_names: tp.Optional[tp.Sequence[str]] = None,
                 attr_settings: tp.KwargsLike = None,
                 **kwargs) -> None:
        """
        初始化信号工厂
        
        Args:
            *args: 传递给父类IndicatorFactory的位置参数
            mode: 信号生成模式，可选值：
                - FactoryMode.Entries: 仅生成入场信号
                - FactoryMode.Exits: 仅生成出场信号  
                - FactoryMode.Both: 同时生成入场和出场信号
                - FactoryMode.Chain: 链式处理模式，基于输入入场信号生成新的入场和出场信号
            input_names: 输入数据名称列表，用于标识传递给信号函数的输入数据
            attr_settings: 属性设置字典，用于配置输出属性的数据类型等
            **kwargs: 传递给父类IndicatorFactory的关键字参数
        """
        # 将模式字符串或整数映射为FactoryMode枚举值
        mode = map_enum_fields(mode, FactoryMode)
        
        # 初始化输入名称列表
        if input_names is None:
            input_names = []
        else:
            input_names = list(input_names)
            
        # 初始化属性设置字典
        if attr_settings is None:
            attr_settings = {}

        # 验证输入名称，确保不会与内置的entries和exits冲突
        if 'entries' in input_names:
            raise ValueError("entries cannot be used in input_names")
        if 'exits' in input_names:
            raise ValueError("exits cannot be used in input_names")
            
        # 根据模式设置输出名称和输入名称
        if mode == FactoryMode.Entries:
            # 仅入场模式：只输出entries
            output_names = ['entries']
        elif mode == FactoryMode.Exits:
            # 仅出场模式：需要entries作为输入，输出exits
            input_names = ['entries'] + input_names
            output_names = ['exits']
        elif mode == FactoryMode.Both:
            # 同时模式：输出entries和exits
            output_names = ['entries', 'exits']
        else:
            # 链式模式：需要entries作为输入，输出new_entries和exits
            input_names = ['entries'] + input_names
            output_names = ['new_entries', 'exits']
            
        # 设置entries输入的数据类型为布尔型
        if 'entries' in input_names:
            attr_settings['entries'] = dict(dtype=np.bool_)
            
        # 设置所有输出属性的数据类型为布尔型
        for output_name in output_names:
            attr_settings[output_name] = dict(dtype=np.bool_)

        # 调用父类初始化方法
        IndicatorFactory.__init__(
            self,
            *args,
            input_names=input_names,
            output_names=output_names,
            attr_settings=attr_settings,
            **kwargs
        )
        # 保存模式设置
        self.mode = mode

        # 定义plot方法，用于可视化信号
        def plot(_self,
                 entry_y: tp.Optional[tp.ArrayLike] = None,
                 exit_y: tp.Optional[tp.ArrayLike] = None,
                 entry_types: tp.Optional[tp.ArrayLikeSequence] = None,
                 exit_types: tp.Optional[tp.ArrayLikeSequence] = None,
                 entry_trace_kwargs: tp.KwargsLike = None,
                 exit_trace_kwargs: tp.KwargsLike = None,
                 fig: tp.Optional[tp.BaseFigure] = None,
                 **kwargs) -> tp.BaseFigure:  # pragma: no cover
            """
            绘制入场和出场信号标记
            
            Args:
                _self: 信号对象实例
                entry_y: 入场信号标记的Y轴值
                exit_y: 出场信号标记的Y轴值  
                entry_types: 入场信号类型数组
                exit_types: 出场信号类型数组
                entry_trace_kwargs: 入场信号轨迹的关键字参数
                exit_trace_kwargs: 出场信号轨迹的关键字参数
                fig: 要添加轨迹的图形对象
                **kwargs: 传递给信号标记绘制方法的关键字参数
                
            Returns:
                包含信号标记的图形对象
            """
            # 检查维度，如果大于1维需要先选择列
            if _self.wrapper.ndim > 1:
                raise TypeError("Select a column first. Use indexing.")

            # 初始化轨迹参数
            if entry_trace_kwargs is None:
                entry_trace_kwargs = {}
            if exit_trace_kwargs is None:
                exit_trace_kwargs = {}
                
            # 设置入场信号轨迹名称
            entry_trace_kwargs = merge_dicts(
                dict(name="New Entry" if mode == FactoryMode.Chain else "Entry"),
                entry_trace_kwargs
            )
            # 设置出场信号轨迹名称
            exit_trace_kwargs = merge_dicts(
                dict(name="Exit"),
                exit_trace_kwargs
            )
                
            # 添加入场信号类型信息到悬停模板
            if entry_types is not None:
                entry_types = np.asarray(entry_types)
                entry_trace_kwargs = merge_dicts(dict(
                    customdata=entry_types,
                    hovertemplate="(%{x}, %{y})<br>Type: %{customdata}"
                ), entry_trace_kwargs)
                
            # 添加出场信号类型信息到悬停模板
            if exit_types is not None:
                exit_types = np.asarray(exit_types)
                exit_trace_kwargs = merge_dicts(dict(
                    customdata=exit_types,
                    hovertemplate="(%{x}, %{y})<br>Type: %{customdata}"
                ), exit_trace_kwargs)
                
            # 根据模式绘制相应的信号标记
            if mode == FactoryMode.Entries:
                # 仅入场模式：只绘制入场信号
                fig = _self.entries.vbt.signals.plot_as_entry_markers(
                    y=entry_y, trace_kwargs=entry_trace_kwargs, fig=fig, **kwargs)
            elif mode == FactoryMode.Exits:
                # 仅出场模式：绘制输入入场信号和输出出场信号
                fig = _self.entries.vbt.signals.plot_as_entry_markers(
                    y=entry_y, trace_kwargs=entry_trace_kwargs, fig=fig, **kwargs)
                fig = _self.exits.vbt.signals.plot_as_exit_markers(
                    y=exit_y, trace_kwargs=exit_trace_kwargs, fig=fig, **kwargs)
            elif mode == FactoryMode.Both:
                # 同时模式：绘制输出入场信号和输出出场信号
                fig = _self.entries.vbt.signals.plot_as_entry_markers(
                    y=entry_y, trace_kwargs=entry_trace_kwargs, fig=fig, **kwargs)
                fig = _self.exits.vbt.signals.plot_as_exit_markers(
                    y=exit_y, trace_kwargs=exit_trace_kwargs, fig=fig, **kwargs)
            else:
                # 链式模式：绘制新入场信号和输出出场信号
                fig = _self.new_entries.vbt.signals.plot_as_entry_markers(
                    y=entry_y, trace_kwargs=entry_trace_kwargs, fig=fig, **kwargs)
                fig = _self.exits.vbt.signals.plot_as_exit_markers(
                    y=exit_y, trace_kwargs=exit_trace_kwargs, fig=fig, **kwargs)

            return fig

        # 设置plot方法的文档字符串
        plot.__doc__ = """Plot `{0}.{1}` and `{0}.exits`.

        Args:
            entry_y (array_like): Y-axis values to plot entry markers on.
            exit_y (array_like): Y-axis values to plot exit markers on.
            entry_types (array_like): Entry types in string format.
            exit_types (array_like): Exit types in string format.
            entry_trace_kwargs (dict): Keyword arguments passed to \
            `vectorbt.signals.accessors.SignalsSRAccessor.plot_as_entry_markers` for `{0}.{1}`.
            exit_trace_kwargs (dict): Keyword arguments passed to \
            `vectorbt.signals.accessors.SignalsSRAccessor.plot_as_exit_markers` for `{0}.exits`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **kwargs: Keyword arguments passed to `vectorbt.signals.accessors.SignalsSRAccessor.plot_as_markers`.
        """.format(
            self.class_name, 'new_entries' if mode == FactoryMode.Chain else 'entries'
        )

        # 将plot方法添加到生成的指标类中
        setattr(self.Indicator, 'plot', plot)

    def from_choice_func(
            self,
            entry_choice_func: tp.Optional[tp.ChoiceFunc] = None,
            exit_choice_func: tp.Optional[tp.ChoiceFunc] = None,
            generate_func: tp.Callable = generate_nb,
            generate_ex_func: tp.Callable = generate_ex_nb,
            generate_enex_func: tp.Callable = generate_enex_nb,
            cache_func: tp.Callable = None,
            entry_settings: tp.KwargsLike = None,
            exit_settings: tp.KwargsLike = None,
            cache_settings: tp.KwargsLike = None,
            numba_loop: bool = False,
            **kwargs) -> tp.Type[IndicatorBase]:
        """
        基于入场和出场选择函数构建信号生成器类
        
        选择函数是返回信号索引的简单函数。有两种类型：入场选择函数和出场选择函数。
        每个选择函数接收广播时间序列、广播就地输出时间序列、广播参数数组和其他参数，
        并返回对应于所选信号的索引数组。详见 `vectorbt.signals.nb.generate_nb`。
        
        Args:
            entry_choice_func (callable): 返回入场信号索引的 `choice_func_nb` 函数
                - 对于 `FactoryMode.Chain` 模式，默认为 `vectorbt.signals.nb.first_choice_nb`
                - 该函数应该使用 @njit 装饰器进行Numba编译
                - 函数签名：def func(from_i, to_i, col, *args) -> np.ndarray
                
            exit_choice_func (callable): 返回出场信号索引的 `choice_func_nb` 函数
                - 该函数应该使用 @njit 装饰器进行Numba编译
                - 函数签名：def func(from_i, to_i, col, *args) -> np.ndarray
                
            generate_func (callable): 入场信号生成函数
                - 默认为 `vectorbt.signals.nb.generate_nb`
                - 负责根据选择函数的结果生成实际的入场信号
                
            generate_ex_func (callable): 出场信号生成函数
                - 默认为 `vectorbt.signals.nb.generate_ex_nb`
                - 负责根据选择函数的结果生成实际的出场信号
                
            generate_enex_func (callable): 入场和出场信号同时生成函数
                - 默认为 `vectorbt.signals.nb.generate_enex_nb`
                - 用于同时生成入场和出场信号的情况
                
            cache_func (callable): 缓存函数，用于预处理数据
                - 所有返回的对象将作为最后几个参数传递给选择函数
                - 用于提高重复计算的效率
                
            entry_settings (dict): 入场选择函数的设置字典
                - 控制哪些输入、参数和参数传递给入场函数
                - 详见下面的设置字典说明
                
            exit_settings (dict): 出场选择函数的设置字典
                - 控制哪些输入、参数和参数传递给出场函数
                - 详见下面的设置字典说明
                
            cache_settings (dict): 缓存函数的设置字典
                - 控制缓存函数的参数传递
                - 详见下面的设置字典说明
                
            numba_loop (bool): 是否使用Numba进行循环
                - 当对小型输入进行大量迭代时设置为True
                - 可以提高性能但会增加编译时间
                
            **kwargs: 传递给 `IndicatorFactory.from_custom_func` 的关键字参数
        
        重要说明：
        ----------
        - 选择函数应该是Numba编译的（使用@njit装饰器）
        - 每个函数要传递哪些输入、参数和参数应该在函数的设置字典中明确指示
        - 默认情况下，不传递任何内容
        - 不支持直接向选择函数传递关键字参数，使用设置字典中的 `pass_kwargs` 将关键字参数作为位置参数传递
        
        设置字典可以包含以下键：
        -------------------------
        
        Attributes:
            pass_inputs (list of str): 要传递给选择函数的输入名称列表
                - 默认为 []。顺序很重要。每个名称必须在 `input_names` 中
                - 示例：['close', 'volume'] 会将close和volume数据传递给函数
                
            pass_in_outputs (list of str): 要传递给选择函数的就地输出名称列表
                - 默认为 []。顺序很重要。每个名称必须在 `in_output_names` 中
                - 用于在函数间共享状态信息
                
            pass_params (list of str): 要传递给选择函数的参数名称列表
                - 默认为 []。顺序很重要。每个名称必须在 `param_names` 中
                - 示例：['threshold', 'period'] 会将阈值和周期参数传递给函数
                
            pass_kwargs (dict, list of str or list of tuple): 从 `kwargs` 字典中要作为位置参数传递给选择函数的关键字参数
                - 默认为 []。顺序很重要
                - 如果任何元素是元组，应包含名称和默认值
                - 如果任何元素是字符串，默认值为None
                
                内置键包括：
                - `input_shape`: 如果没有传递输入时间序列，则为输入形状
                    - 如果 `pass_input_shape` 为True，则由管道提供默认值
                - `wait`: 放置信号前等待的刻度数
                    - 默认为1
                - `until_next`: 是否将信号放置到下一个入场信号
                    - 默认为True，仅在 `generate_ex_func` 中应用
                - `skip_until_exit`: 是否跳过处理入场信号直到下一个出场
                    - 默认为False，仅在 `generate_ex_func` 中应用
                - `pick_first`: 是否在找到第一个出场信号时立即停止
                    - 对于 `FactoryMode.Entries` 默认为False，否则为True
                - `temp_idx_arr`: 用于临时存储索引的空整数数组
                    - 默认为自动生成的形状为 `input_shape[0]` 的数组
                    - 也可以传递 `temp_idx_arr1`, `temp_idx_arr2` 等来生成多个
                - `flex_2d`: 参见 `vectorbt.base.reshape_fns.flex_select_auto_nb`
                    - 如果 `pass_flex_2d` 为True，则由管道提供默认值
                    
            pass_cache (bool): 是否将缓存从 `cache_func` 传递给选择函数
                - 默认为False。缓存以解包形式传递
        
        可以传递给 `run` 和 `run_combs` 方法的参数：
        -----------------------------------------
        
        Args:
            *args: 对于 `FactoryMode.Entries` 应使用此参数代替 `entry_args`，
                   对于 `FactoryMode.Exits` 和默认 `entry_choice_func` 的 `FactoryMode.Chain` 应使用此参数代替 `exit_args`
            entry_args (tuple): 传递给入场选择函数的参数
            exit_args (tuple): 传递给出场选择函数的参数
            cache_args (tuple): 传递给缓存函数的参数
            entry_kwargs (tuple): 入场选择函数的设置。如果 `pass_kwargs` 中有参数，也包含作为位置参数传递的参数
            exit_kwargs (tuple): 出场选择函数的设置。如果 `pass_kwargs` 中有参数，也包含作为位置参数传递的参数
            cache_kwargs (tuple): 缓存函数的设置。如果 `pass_kwargs` 中有参数，也包含作为位置参数传递的参数
            return_cache (bool): 是否仅返回缓存
            use_cache (any): 要使用的缓存
            **kwargs: 对于 `FactoryMode.Entries` 应使用此参数代替 `entry_kwargs`，
                      对于 `FactoryMode.Exits` 和默认 `entry_choice_func` 的 `FactoryMode.Chain` 应使用此参数代替 `exit_kwargs`
        
        更多参数请参见 `vectorbt.indicators.factory.run_pipeline`。
        
        使用示例：
        ---------
        
        1. 最简单的信号指标，在第一个索引处放置True：
        
        ```python
        from numba import njit
        import vectorbt as vbt
        import numpy as np

        @njit
        def entry_choice_func(from_i, to_i, col):
            return np.array([from_i])

        @njit
        def exit_choice_func(from_i, to_i, col):
            return np.array([from_i])

        MySignals = vbt.SignalFactory().from_choice_func(
            entry_choice_func=entry_choice_func,
            exit_choice_func=exit_choice_func,
            entry_kwargs=dict(wait=1),
            exit_kwargs=dict(wait=1)
        )

        my_sig = MySignals.run(input_shape=(3, 3))
        print(my_sig.entries)
        print(my_sig.exits)
        ```
        
        2. 取第一个入场信号，等待n个刻度后放置出场信号。找到下一个入场信号并重复。测试三个不同的n值：
        
        ```python
        from numba import njit
        from vectorbt.signals.factory import SignalFactory

        @njit
        def wait_choice_nb(from_i, to_i, col, n, temp_idx_arr):
            temp_idx_arr[0] = from_i + n  # 下一个出场的索引
            if temp_idx_arr[0] < to_i:
                return temp_idx_arr[:1]
            return temp_idx_arr[:0]  # 必须返回数组

        # 构建信号生成器
        MySignals = SignalFactory(
            mode='chain',
            param_names=['n']
        ).from_choice_func(
            exit_choice_func=wait_choice_nb,
            exit_settings=dict(
                pass_params=['n'],
                pass_kwargs=['temp_idx_arr']  # 内置关键字参数
            )
        )

        # 运行信号生成器
        entries = [True, True, True, True, True]
        my_sig = MySignals.run(entries, [0, 1, 2])
        ```
        
        3. 要组合多个迭代信号，需要创建自定义选择函数。以下是使用"OR"规则组合两个随机生成器的示例（第一个信号获胜）：
        
        ```python
        from numba import njit
        from collections import namedtuple
        from vectorbt.indicators.configs import flex_elem_param_config
        from vectorbt.signals.factory import SignalFactory
        from vectorbt.signals.nb import rand_by_prob_choice_nb

        # 枚举以区分随机生成器
        RandType = namedtuple('RandType', ['R1', 'R2'])(0, 1)

        # 定义出场选择函数
        @njit
        def rand_exit_choice_nb(from_i, to_i, col, rand_type, prob1,
                                prob2, temp_idx_arr1, temp_idx_arr2, flex_2d):
            idxs1 = rand_by_prob_choice_nb(from_i, to_i, col, prob1, True, temp_idx_arr1, flex_2d)
            if len(idxs1) > 0:
                to_i = idxs1[0]  # 不需要超过第一个找到的信号
            idxs2 = rand_by_prob_choice_nb(from_i, to_i, col, prob2, True, temp_idx_arr2, flex_2d)
            if len(idxs2) > 0:
                rand_type[idxs2[0], col] = RandType.R2
                return idxs2
            if len(idxs1) > 0:
                rand_type[idxs1[0], col] = RandType.R1
                return idxs1
            return temp_idx_arr1[:0]

        # 构建信号生成器
        MySignals = SignalFactory(
            mode='chain',
            in_output_names=['rand_type'],
            param_names=['prob1', 'prob2'],
            attr_settings=dict(
                rand_type=dict(dtype=RandType)  # 创建rand_type_readable
            )
        ).from_choice_func(
            exit_choice_func=rand_exit_choice_nb,
            exit_settings=dict(
                pass_in_outputs=['rand_type'],
                pass_params=['prob1', 'prob2'],
                pass_kwargs=['temp_idx_arr1', 'temp_idx_arr2', 'flex_2d']
            ),
            param_settings=dict(
                prob1=flex_elem_param_config,  # 每个框架/行/列/元素的参数
                prob2=flex_elem_param_config
            ),
            pass_flex_2d=True,
            rand_type=-1  # 用此值填充
        )

        # 运行信号生成器
        entries = [True, True, True, True, True]
        my_sig = MySignals.run(entries, [0., 1.], [0., 1.], param_product=True)
        ```
        
        Returns:
            tp.Type[IndicatorBase]: 生成的信号指标类，可以用于创建信号实例
        """

        # 获取当前模式和相关名称列表
        mode = self.mode
        input_names = self.input_names
        param_names = self.param_names
        in_output_names = self.in_output_names

        # 根据模式验证和设置函数要求
        if mode == FactoryMode.Entries:
            # 仅入场模式：需要输入形状，必须提供入场选择函数
            require_input_shape = True
            checks.assert_not_none(entry_choice_func)
            checks.assert_numba_func(entry_choice_func)
            if exit_choice_func is not None:
                raise ValueError("exit_choice_func cannot be used with FactoryMode.Entries")
        elif mode == FactoryMode.Exits:
            # 仅出场模式：不需要输入形状，必须提供出场选择函数
            require_input_shape = False
            if entry_choice_func is not None:
                raise ValueError("entry_choice_func cannot be used with FactoryMode.Exits")
            checks.assert_not_none(exit_choice_func)
            checks.assert_numba_func(exit_choice_func)
        elif mode == FactoryMode.Both:
            # 同时模式：需要输入形状，必须同时提供入场和出场选择函数
            require_input_shape = True
            checks.assert_not_none(entry_choice_func)
            checks.assert_numba_func(entry_choice_func)
            checks.assert_not_none(exit_choice_func)
            checks.assert_numba_func(exit_choice_func)
        else:
            # 链式模式：不需要输入形状，可以设置默认的入场选择函数
            require_input_shape = False
            if entry_choice_func is None:
                entry_choice_func = first_choice_nb
            if entry_settings is None:
                entry_settings = {}
            # 为链式模式设置默认的入场设置，传递entries输入
            entry_settings = merge_dicts(dict(
                pass_inputs=['entries']
            ), entry_settings)
            checks.assert_not_none(entry_choice_func)
            checks.assert_numba_func(entry_choice_func)
            checks.assert_not_none(exit_choice_func)
            checks.assert_numba_func(exit_choice_func)
            
        # 从kwargs中获取require_input_shape，如果未提供则使用上面计算的值
        require_input_shape = kwargs.pop('require_input_shape', require_input_shape)

        # 初始化设置字典
        if entry_settings is None:
            entry_settings = {}
        if exit_settings is None:
            exit_settings = {}
        if cache_settings is None:
            cache_settings = {}

        # 验证设置字典中的键是否有效
        valid_keys = [
            'pass_inputs',
            'pass_in_outputs',
            'pass_params',
            'pass_kwargs',
            'pass_cache'
        ]
        checks.assert_dict_valid(entry_settings, valid_keys)
        checks.assert_dict_valid(exit_settings, valid_keys)
        checks.assert_dict_valid(cache_settings, valid_keys)

        # 定义辅助函数：从函数设置中获取指定类型的名称列表
        def _get_func_names(func_settings: tp.Kwargs, setting: str, all_names: tp.Sequence[str]) -> tp.List[str]:
            """
            从函数设置中获取指定类型的名称列表
            
            Args:
                func_settings: 函数设置字典
                setting: 设置键名（如'pass_inputs'）
                all_names: 所有可用名称的列表
                
            Returns:
                从设置中获取的名称列表
            """
            func_input_names = func_settings.get(setting, None)
            if func_input_names is None:
                return []
            else:
                # 验证所有名称都在可用名称列表中
                for name in func_input_names:
                    checks.assert_in(name, all_names)
            return func_input_names

        # 获取各函数需要的输入名称
        entry_input_names = _get_func_names(entry_settings, 'pass_inputs', input_names)
        exit_input_names = _get_func_names(exit_settings, 'pass_inputs', input_names)
        cache_input_names = _get_func_names(cache_settings, 'pass_inputs', input_names)

        # 获取各函数需要的就地输出名称
        entry_in_output_names = _get_func_names(entry_settings, 'pass_in_outputs', in_output_names)
        exit_in_output_names = _get_func_names(exit_settings, 'pass_in_outputs', in_output_names)
        cache_in_output_names = _get_func_names(cache_settings, 'pass_in_outputs', in_output_names)

        # 获取各函数需要的参数名称
        entry_param_names = _get_func_names(entry_settings, 'pass_params', param_names)
        exit_param_names = _get_func_names(exit_settings, 'pass_params', param_names)
        cache_param_names = _get_func_names(cache_settings, 'pass_params', param_names)

        # 构建选择参数元组的函数
        # 根据不同的模式构建不同的apply_func函数
        if mode == FactoryMode.Entries:
            # 仅入场模式：构建入场信号生成函数
            _0 = "i"  # 参数索引
            _0 += ", shape"  # 输入形状
            _0 += ", entry_pick_first"  # 是否选择第一个信号
            _0 += ", entry_input_tuple"  # 入场输入元组
            if len(entry_in_output_names) > 0:
                _0 += ", entry_in_output_tuples"  # 入场就地输出元组
            if len(entry_param_names) > 0:
                _0 += ", entry_param_tuples"  # 入场参数元组
            _0 += ", entry_args"  # 入场函数参数
            
            _1 = "shape"  # 传递给generate_func的参数
            _1 += ", entry_pick_first"
            _1 += ", entry_choice_func"
            _1 += ", *entry_input_tuple"  # 解包入场输入
            if len(entry_in_output_names) > 0:
                _1 += ", *entry_in_output_tuples[i]"  # 解包当前索引的就地输出
            if len(entry_param_names) > 0:
                _1 += ", *entry_param_tuples[i]"  # 解包当前索引的参数
            _1 += ", *entry_args"  # 解包入场函数参数
            
            # 构建函数字符串并编译
            func_str = "def apply_func({0}):\n   return generate_func({1})".format(_0, _1)
            scope = {
                'generate_func': generate_func,
                'entry_choice_func': entry_choice_func
            }
            filename = inspect.getfile(lambda: None)
            code = compile(func_str, filename, 'single')
            exec(code, scope)
            apply_func = scope['apply_func']
            
            # 根据numba_loop设置选择连接函数
            if numba_loop:
                apply_func = njit(apply_func)
                apply_and_concat_func = combine_fns.apply_and_concat_one_nb
            else:
                apply_and_concat_func = combine_fns.apply_and_concat_one

        elif mode == FactoryMode.Exits:
            # 仅出场模式：构建出场信号生成函数
            _0 = "i"  # 参数索引
            _0 += ", entries"  # 输入入场信号
            _0 += ", exit_wait"  # 出场等待时间
            _0 += ", until_next"  # 是否到下一个入场信号
            _0 += ", skip_until_exit"  # 是否跳过直到出场
            _0 += ", exit_pick_first"  # 是否选择第一个出场信号
            _0 += ", exit_input_tuple"  # 出场输入元组
            if len(exit_in_output_names) > 0:
                _0 += ", exit_in_output_tuples"  # 出场就地输出元组
            if len(exit_param_names) > 0:
                _0 += ", exit_param_tuples"  # 出场参数元组
            _0 += ", exit_args"  # 出场函数参数
            
            _1 = "entries"  # 传递给generate_ex_func的参数
            _1 += ", exit_wait"
            _1 += ", until_next"
            _1 += ", skip_until_exit"
            _1 += ", exit_pick_first"
            _1 += ", exit_choice_func"
            _1 += ", *exit_input_tuple"  # 解包出场输入
            if len(exit_in_output_names) > 0:
                _1 += ", *exit_in_output_tuples[i]"  # 解包当前索引的就地输出
            if len(exit_param_names) > 0:
                _1 += ", *exit_param_tuples[i]"  # 解包当前索引的参数
            _1 += ", *exit_args"  # 解包出场函数参数
            
            # 构建函数字符串并编译
            func_str = "def apply_func({0}):\n   return generate_ex_func({1})".format(_0, _1)
            scope = {
                'generate_ex_func': generate_ex_func,
                'exit_choice_func': exit_choice_func
            }
            filename = inspect.getfile(lambda: None)
            code = compile(func_str, filename, 'single')
            exec(code, scope)
            apply_func = scope['apply_func']
            
            # 根据numba_loop设置选择连接函数
            if numba_loop:
                apply_func = njit(apply_func)
                apply_and_concat_func = combine_fns.apply_and_concat_one_nb
            else:
                apply_and_concat_func = combine_fns.apply_and_concat_one

        else:
            # 同时模式或链式模式：构建入场和出场信号同时生成函数
            _0 = "i"  # 参数索引
            _0 += ", shape"  # 输入形状
            _0 += ", entry_wait"  # 入场等待时间
            _0 += ", exit_wait"  # 出场等待时间
            _0 += ", entry_pick_first"  # 是否选择第一个入场信号
            _0 += ", exit_pick_first"  # 是否选择第一个出场信号
            _0 += ", entry_input_tuple"  # 入场输入元组
            _0 += ", exit_input_tuple"  # 出场输入元组
            if len(entry_in_output_names) > 0:
                _0 += ", entry_in_output_tuples"  # 入场就地输出元组
            if len(exit_in_output_names) > 0:
                _0 += ", exit_in_output_tuples"  # 出场就地输出元组
            if len(entry_param_names) > 0:
                _0 += ", entry_param_tuples"  # 入场参数元组
            if len(exit_param_names) > 0:
                _0 += ", exit_param_tuples"  # 出场参数元组
            _0 += ", entry_args"  # 入场函数参数
            _0 += ", exit_args"  # 出场函数参数
            
            _1 = "shape"  # 传递给generate_enex_func的参数
            _1 += ", entry_wait"
            _1 += ", exit_wait"
            _1 += ", entry_pick_first"
            _1 += ", exit_pick_first"
            _1 += ", entry_choice_func"
            _1 += ", (*entry_input_tuple"  # 入场输入元组
            if len(entry_in_output_names) > 0:
                _1 += ", *entry_in_output_tuples[i]"  # 解包当前索引的入场就地输出
            if len(entry_param_names) > 0:
                _1 += ", *entry_param_tuples[i]"  # 解包当前索引的入场参数
            _1 += ", *entry_args)"  # 解包入场函数参数
            _1 += ", exit_choice_func"
            _1 += ", (*exit_input_tuple"  # 出场输入元组
            if len(exit_in_output_names) > 0:
                _1 += ", *exit_in_output_tuples[i]"  # 解包当前索引的出场就地输出
            if len(exit_param_names) > 0:
                _1 += ", *exit_param_tuples[i]"  # 解包当前索引的出场参数
            _1 += ", *exit_args)"  # 解包出场函数参数
            
            # 构建函数字符串并编译
            func_str = "def apply_func({0}):\n   return generate_enex_func({1})".format(_0, _1)
            scope = {
                'generate_enex_func': generate_enex_func,
                'entry_choice_func': entry_choice_func,
                'exit_choice_func': exit_choice_func
            }
            filename = inspect.getfile(lambda: None)
            code = compile(func_str, filename, 'single')
            exec(code, scope)
            apply_func = scope['apply_func']
            
            # 根据numba_loop设置选择连接函数
            if numba_loop:
                apply_func = njit(apply_func)
                apply_and_concat_func = combine_fns.apply_and_concat_multiple_nb
            else:
                apply_and_concat_func = combine_fns.apply_and_concat_multiple

        # 定义自定义函数，这是信号生成的核心逻辑
        def custom_func(input_list: tp.List[tp.AnyArray],
                        in_output_list: tp.List[tp.List[tp.AnyArray]],
                        param_list: tp.List[tp.List[tp.Param]],
                        *args,
                        input_shape: tp.Optional[tp.Shape] = None,
                        flex_2d: tp.Optional[bool] = None,
                        entry_args: tp.Optional[tp.Args] = None,
                        exit_args: tp.Optional[tp.Args] = None,
                        cache_args: tp.Optional[tp.Args] = None,
                        entry_kwargs: tp.KwargsLike = None,
                        exit_kwargs: tp.KwargsLike = None,
                        cache_kwargs: tp.KwargsLike = None,
                        return_cache: bool = False,
                        use_cache: tp.Optional[CacheOutputT] = None,
                        **_kwargs) -> tp.Union[CacheOutputT, tp.Array2d, tp.List[tp.Array2d]]:
            """
            自定义信号生成函数
            
            这是信号工厂的核心函数，负责协调各个选择函数和生成函数，
            处理参数传递、缓存机制和信号生成逻辑。
            
            Args:
                input_list: 输入数据列表，包含所有传递给信号函数的时间序列数据
                in_output_list: 就地输出列表，用于在函数间共享状态信息
                param_list: 参数列表，包含所有传递给信号函数的参数
                *args: 位置参数，根据模式不同有不同的含义
                input_shape: 输入数据的形状，如果没有传递输入时间序列则需要提供
                flex_2d: 是否使用灵活的2D处理模式
                entry_args: 传递给入场选择函数的参数元组
                exit_args: 传递给出场选择函数的参数元组
                cache_args: 传递给缓存函数的参数元组
                entry_kwargs: 入场选择函数的设置字典
                exit_kwargs: 出场选择函数的设置字典
                cache_kwargs: 缓存函数的设置字典
                return_cache: 是否仅返回缓存结果
                use_cache: 要使用的缓存对象
                **_kwargs: 其他关键字参数
                
            Returns:
                根据模式返回不同的结果：
                - 仅入场模式：返回入场信号数组
                - 仅出场模式：返回出场信号数组
                - 同时模式：返回[入场信号数组, 出场信号数组]
                - 缓存模式：返回缓存对象
            """
            # 获取输入形状
            if len(input_list) == 0:
                if input_shape is None:
                    raise ValueError("Pass input_shape if no input time series were passed")
            else:
                input_shape = input_list[0].shape

            # 初始化参数元组
            if entry_args is None:
                entry_args = ()
            if exit_args is None:
                exit_args = ()
            if cache_args is None:
                cache_args = ()
                
            # 根据模式处理位置参数
            if mode == FactoryMode.Entries:
                # 仅入场模式：使用*args作为入场参数
                if len(entry_args) > 0:
                    raise ValueError("Use *args instead of entry_args with FactoryMode.Entries")
                entry_args = args
            elif mode == FactoryMode.Exits or (mode == FactoryMode.Chain and entry_choice_func == first_choice_nb):
                # 仅出场模式或链式模式：使用*args作为出场参数
                if len(exit_args) > 0:
                    raise ValueError("Use *args instead of exit_args "
                                     "with FactoryMode.Exits or FactoryMode.Chain")
                exit_args = args
            else:
                # 同时模式：不允许使用*args
                if len(args) > 0:
                    raise ValueError("*args cannot be used with FactoryMode.Both")

            # 初始化关键字参数字典
            if entry_kwargs is None:
                entry_kwargs = {}
            if exit_kwargs is None:
                exit_kwargs = {}
            if cache_kwargs is None:
                cache_kwargs = {}
                
            # 根据模式处理关键字参数
            if mode == FactoryMode.Entries:
                # 仅入场模式：使用**_kwargs作为入场关键字参数
                if len(entry_kwargs) > 0:
                    raise ValueError("Use **kwargs instead of entry_kwargs with FactoryMode.Entries")
                entry_kwargs = _kwargs
            elif mode == FactoryMode.Exits or (mode == FactoryMode.Chain and entry_choice_func == first_choice_nb):
                # 仅出场模式或链式模式：使用**_kwargs作为出场关键字参数
                if len(exit_kwargs) > 0:
                    raise ValueError("Use **kwargs instead of exit_kwargs "
                                     "with FactoryMode.Exits or FactoryMode.Chain")
                exit_kwargs = _kwargs
            else:
                # 同时模式：不允许使用**_kwargs
                if len(_kwargs) > 0:
                    raise ValueError("*args cannot be used with FactoryMode.Both")

            # 设置默认的关键字参数
            kwargs_defaults = dict(
                input_shape=input_shape,
                wait=1,  # 默认等待1个刻度
                until_next=True,  # 默认到下一个入场信号
                skip_until_exit=False,  # 默认不跳过直到出场
                pick_first=True,  # 默认选择第一个信号
                flex_2d=flex_2d,  # 灵活的2D处理模式
            )
            # 仅入场模式：不选择第一个信号
            if mode == FactoryMode.Entries:
                kwargs_defaults['pick_first'] = False
                
            # 合并默认参数和用户提供的参数
            entry_kwargs = merge_dicts(kwargs_defaults, entry_kwargs)
            exit_kwargs = merge_dicts(kwargs_defaults, exit_kwargs)
            cache_kwargs = merge_dicts(kwargs_defaults, cache_kwargs)
            
            # 提取关键参数
            entry_wait = entry_kwargs['wait']
            exit_wait = exit_kwargs['wait']
            entry_pick_first = entry_kwargs['pick_first']
            exit_pick_first = exit_kwargs['pick_first']
            until_next = exit_kwargs['until_next']
            skip_until_exit = exit_kwargs['skip_until_exit']

            # 分发参数到各个函数
            # 构建入场输入元组
            entry_input_tuple = ()
            for input_name in entry_input_names:
                entry_input_tuple += (input_list[input_names.index(input_name)],)
                
            # 构建出场输入元组
            exit_input_tuple = ()
            for input_name in exit_input_names:
                exit_input_tuple += (input_list[input_names.index(input_name)],)
                
            # 构建缓存输入元组
            cache_input_tuple = ()
            for input_name in cache_input_names:
                cache_input_tuple += (input_list[input_names.index(input_name)],)

            # 分发就地输出到各个函数
            # 构建入场就地输出列表
            entry_in_output_list = []
            for in_output_name in entry_in_output_names:
                entry_in_output_list.append(in_output_list[in_output_names.index(in_output_name)])
                
            # 构建出场就地输出列表
            exit_in_output_list = []
            for in_output_name in exit_in_output_names:
                exit_in_output_list.append(in_output_list[in_output_names.index(in_output_name)])
                
            # 构建缓存就地输出列表
            cache_in_output_list = []
            for in_output_name in cache_in_output_names:
                cache_in_output_list.append(in_output_list[in_output_names.index(in_output_name)])

            # 分发参数到各个函数
            # 构建入场参数列表
            entry_param_list = []
            for param_name in entry_param_names:
                entry_param_list.append(param_list[param_names.index(param_name)])
                
            # 构建出场参数列表
            exit_param_list = []
            for param_name in exit_param_names:
                exit_param_list.append(param_list[param_names.index(param_name)])
                
            # 构建缓存参数列表
            cache_param_list = []
            for param_name in cache_param_names:
                cache_param_list.append(param_list[param_names.index(param_name)])

            # 计算参数数量并构建参数元组
            n_params = len(param_list[0]) if len(param_list) > 0 else 1
            entry_in_output_tuples = list(zip(*entry_in_output_list))
            exit_in_output_tuples = list(zip(*exit_in_output_list))
            entry_param_tuples = list(zip(*entry_param_list))
            exit_param_tuples = list(zip(*exit_param_list))

            # 定义辅助函数：构建额外的参数
            def _build_more_args(func_settings: tp.Kwargs, func_kwargs: tp.Kwargs) -> tp.Args:
                """
                根据函数设置构建额外的参数
                
                Args:
                    func_settings: 函数设置字典
                    func_kwargs: 函数关键字参数字典
                    
                Returns:
                    额外的参数元组
                """
                pass_kwargs = func_settings.get('pass_kwargs', [])
                if isinstance(pass_kwargs, dict):
                    pass_kwargs = list(pass_kwargs.items())
                more_args = ()
                for key in pass_kwargs:
                    value = None
                    if isinstance(key, tuple):
                        key, value = key
                    else:
                        # 为临时索引数组设置默认值
                        if key.startswith('temp_idx_arr'):
                            value = np.empty((input_shape[0],), dtype=np.int64)
                    value = func_kwargs.get(key, value)
                    more_args += (value,)
                return more_args

            # 构建各函数的额外参数
            entry_more_args = _build_more_args(entry_settings, entry_kwargs)
            exit_more_args = _build_more_args(exit_settings, exit_kwargs)
            cache_more_args = _build_more_args(cache_settings, cache_kwargs)

            # 缓存处理
            cache = use_cache
            if cache is None and cache_func is not None:
                # 准备缓存函数的参数
                _cache_in_output_list = cache_in_output_list
                _cache_param_list = cache_param_list
                
                # 如果是Numba函数，需要转换参数类型
                if checks.is_numba_func(cache_func):
                    if len(_cache_in_output_list) > 0:
                        _cache_in_output_list = [to_typed_list(in_outputs) for in_outputs in _cache_in_output_list]
                    if len(_cache_param_list) > 0:
                        _cache_param_list = [to_typed_list(params) for params in _cache_param_list]

                # 调用缓存函数
                cache = cache_func(
                    *cache_input_tuple,
                    *_cache_in_output_list,
                    *_cache_param_list,
                    *cache_args,
                    *cache_more_args
                )
                
            # 如果只需要返回缓存，直接返回
            if return_cache:
                return cache
                
            # 处理缓存结果
            if cache is None:
                cache = ()
            if not isinstance(cache, tuple):
                cache = (cache,)

            # 准备传递给各函数的缓存
            entry_cache = ()
            exit_cache = ()
            if entry_settings.get('pass_cache', False):
                entry_cache = cache
            if exit_settings.get('pass_cache', False):
                exit_cache = cache

            # 应用并连接信号生成结果
            # 根据不同的模式调用相应的信号生成逻辑
            if mode == FactoryMode.Entries:
                # 仅入场模式：只生成入场信号
                
                # 处理入场就地输出元组
                if len(entry_in_output_names) > 0:
                    if numba_loop:
                        _entry_in_output_tuples = (to_typed_list(entry_in_output_tuples),)
                    else:
                        _entry_in_output_tuples = (entry_in_output_tuples,)
                else:
                    _entry_in_output_tuples = ()
                    
                # 处理入场参数元组
                if len(entry_param_names) > 0:
                    if numba_loop:
                        _entry_param_tuples = (to_typed_list(entry_param_tuples),)
                    else:
                        _entry_param_tuples = (entry_param_tuples,)
                else:
                    _entry_param_tuples = ()

                # 调用入场信号生成函数
                return apply_and_concat_func(
                    n_params,  # 参数数量
                    apply_func,  # 应用函数
                    input_shape,  # 输入形状
                    entry_pick_first,  # 是否选择第一个信号
                    entry_input_tuple,  # 入场输入元组
                    *_entry_in_output_tuples,  # 入场就地输出元组
                    *_entry_param_tuples,  # 入场参数元组
                    entry_args + entry_more_args + entry_cache  # 入场函数参数
                )

            elif mode == FactoryMode.Exits:
                # 仅出场模式：只生成出场信号
                
                # 处理出场就地输出元组
                if len(exit_in_output_names) > 0:
                    if numba_loop:
                        _exit_in_output_tuples = (to_typed_list(exit_in_output_tuples),)
                    else:
                        _exit_in_output_tuples = (exit_in_output_tuples,)
                else:
                    _exit_in_output_tuples = ()
                    
                # 处理出场参数元组
                if len(exit_param_names) > 0:
                    if numba_loop:
                        _exit_param_tuples = (to_typed_list(exit_param_tuples),)
                    else:
                        _exit_param_tuples = (exit_param_tuples,)
                else:
                    _exit_param_tuples = ()

                # 调用出场信号生成函数
                return apply_and_concat_func(
                    n_params,  # 参数数量
                    apply_func,  # 应用函数
                    input_list[0],  # 输入入场信号
                    exit_wait,  # 出场等待时间
                    until_next,  # 是否到下一个入场信号
                    skip_until_exit,  # 是否跳过直到出场
                    exit_pick_first,  # 是否选择第一个出场信号
                    exit_input_tuple,  # 出场输入元组
                    *_exit_in_output_tuples,  # 出场就地输出元组
                    *_exit_param_tuples,  # 出场参数元组
                    exit_args + exit_more_args + exit_cache  # 出场函数参数
                )

            else:
                # 同时模式或链式模式：同时生成入场和出场信号
                
                # 处理入场就地输出元组
                if len(entry_in_output_names) > 0:
                    if numba_loop:
                        _entry_in_output_tuples = (to_typed_list(entry_in_output_tuples),)
                    else:
                        _entry_in_output_tuples = (entry_in_output_tuples,)
                else:
                    _entry_in_output_tuples = ()
                    
                # 处理入场参数元组
                if len(entry_param_names) > 0:
                    if numba_loop:
                        _entry_param_tuples = (to_typed_list(entry_param_tuples),)
                    else:
                        _entry_param_tuples = (entry_param_tuples,)
                else:
                    _entry_param_tuples = ()
                    
                # 处理出场就地输出元组
                if len(exit_in_output_names) > 0:
                    if numba_loop:
                        _exit_in_output_tuples = (to_typed_list(exit_in_output_tuples),)
                    else:
                        _exit_in_output_tuples = (exit_in_output_tuples,)
                else:
                    _exit_in_output_tuples = ()
                    
                # 处理出场参数元组
                if len(exit_param_names) > 0:
                    if numba_loop:
                        _exit_param_tuples = (to_typed_list(exit_param_tuples),)
                    else:
                        _exit_param_tuples = (exit_param_tuples,)
                else:
                    _exit_param_tuples = ()

                # 调用入场和出场信号同时生成函数
                return apply_and_concat_func(
                    n_params,  # 参数数量
                    apply_func,  # 应用函数
                    input_shape,  # 输入形状
                    entry_wait,  # 入场等待时间
                    exit_wait,  # 出场等待时间
                    entry_pick_first,  # 是否选择第一个入场信号
                    exit_pick_first,  # 是否选择第一个出场信号
                    entry_input_tuple,  # 入场输入元组
                    exit_input_tuple,  # 出场输入元组
                    *_entry_in_output_tuples,  # 入场就地输出元组
                    *_exit_in_output_tuples,  # 出场就地输出元组
                    *_entry_param_tuples,  # 入场参数元组
                    *_exit_param_tuples,  # 出场参数元组
                    entry_args + entry_more_args + entry_cache,  # 入场函数参数
                    exit_args + exit_more_args + exit_cache  # 出场函数参数
                )

        # 调用父类的from_custom_func方法创建最终的信号生成器类
        return self.from_custom_func(
            custom_func,  # 自定义函数
            as_lists=True,  # 使用列表形式
            require_input_shape=require_input_shape,  # 是否需要输入形状
            **kwargs  # 其他关键字参数
        )
