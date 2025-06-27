# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
数组包装器模块

本模块是vectorbt量化交易框架的核心基础模块之一，主要提供NumPy数组与pandas对象之间的
高效转换和元数据管理功能。

主要功能：
1. 数组元数据管理：存储和管理NumPy数组的索引、列名和形状信息
2. 类型转换：在NumPy数组和pandas Series/DataFrame之间无缝转换
3. 分组操作：与ColumnGrouper集成，支持列的分组操作
4. 索引操作：提供高效的数组索引和切片功能
5. 时间序列支持：处理时间频率和时间增量转换
6. 缓存优化：使用装饰器实现方法结果缓存，提升性能

设计理念：
- 元数据分离：将数组数据与元数据分离，提高内存效率
- 不可变性：ArrayWrapper设计为不可变对象，确保数据一致性
- 延迟计算：通过缓存机制实现延迟计算，优化性能
- 类型安全：提供严格的类型检查和验证
- 扩展性：支持自定义包装器和扩展功能

应用场景：
- 量化策略回测：管理价格、成交量等时间序列数据
- 技术指标计算：为指标计算提供统一的数据接口
- 投资组合分析：处理多资产的收益和风险数据
- 数据可视化：为图表绘制提供结构化数据

该模块是vectorbt框架中数据处理层的核心组件，为上层的策略、指标、统计分析等模块
提供了统一、高效的数据包装和操作接口。
"""

import warnings  # 导入警告模块，用于发出运行时警告

import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import pandas as pd  # 导入Pandas库，用于数据结构和数据分析

from vectorbt import _typing as tp  # 导入vectorbt类型定义模块
from vectorbt.base import index_fns, reshape_fns  # 导入基础模块：索引函数和重塑函数
from vectorbt.base.column_grouper import ColumnGrouper  # 导入列分组器
from vectorbt.base.indexing import IndexingError, PandasIndexer  # 导入索引相关类和异常
from vectorbt.base.reshape_fns import to_pd_array  # 导入数组转pandas对象函数
from vectorbt.utils import checks  # 导入检查工具模块
from vectorbt.utils.array_ import get_ranges_arr  # 导入数组工具函数
from vectorbt.utils.attr_ import AttrResolver, AttrResolverT  # 导入属性解析器
from vectorbt.utils.config import Configured, merge_dicts  # 导入配置相关工具
from vectorbt.utils.datetime_ import freq_to_timedelta, DatetimeIndexes  # 导入时间处理工具
from vectorbt.utils.decorators import cached_method  # 导入缓存装饰器

# 定义ArrayWrapper类型变量，用于类型提示中的自引用
ArrayWrapperT = tp.TypeVar("ArrayWrapperT", bound="ArrayWrapper")
# 定义索引元数据类型，包含包装器和索引信息的元组
IndexingMetaT = tp.Tuple[ArrayWrapperT, tp.MaybeArray, tp.MaybeArray, tp.Array1d]


class ArrayWrapper(Configured, PandasIndexer):
    """
    数组包装器类
    
    用于存储和管理NumPy数组的索引、列名和形状元数据的核心类。该类与ColumnGrouper
    紧密集成，提供了高效的数组包装和操作功能。

    主要特性：
    - 元数据管理：存储索引、列名、维度和频率信息
    - 不可变设计：确保对象状态的一致性和线程安全
    - 分组支持：与ColumnGrouper集成，支持列的分组操作
    - 缓存优化：使用@cached_method装饰器优化性能
    - 类型转换：支持NumPy数组与pandas对象的相互转换

    设计模式：
    - 策略模式：通过不同的包装策略处理不同类型的数据
    - 装饰器模式：使用缓存装饰器增强方法功能
    - 适配器模式：适配NumPy和pandas之间的接口差异

    Args:
        index: 行索引，类似pandas的行索引
        columns: 列索引，类似pandas的列索引  
        ndim: 数组维度数量（1或2）
        freq: 时间序列频率，用于时间相关计算
        column_only_select: 是否仅对列进行索引操作
        group_select: 是否对分组进行索引操作
        grouped_ndim: 分组后的维度数量
        **kwargs: 传递给ColumnGrouper的额外参数

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbt.base.array_wrapper import ArrayWrapper
        
        # 创建基本的数组包装器
        >>> index = pd.DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03'])
        >>> columns = pd.Index(['AAPL', 'GOOGL', 'MSFT'])
        >>> wrapper = ArrayWrapper(index, columns, ndim=2)
        >>> print(wrapper.shape)
        (3, 3)
        
        # 包装NumPy数组为DataFrame
        >>> data = np.random.randn(3, 3)
        >>> df = wrapper.wrap(data)
        >>> print(type(df))
        <class 'pandas.core.frame.DataFrame'>
        
        # 创建时间序列包装器
        >>> ts_wrapper = ArrayWrapper(
        ...     index=pd.date_range('2024-01-01', periods=100, freq='D'),
        ...     columns=['price'],
        ...     ndim=1,
        ...     freq='D'
        ... )
        >>> print(ts_wrapper.freq)
        Timedelta('1 days 00:00:00')
        
        # 分组操作示例
        >>> sectors = ['Tech', 'Tech', 'Tech']  # 行业分组
        >>> grouped_wrapper = ArrayWrapper(
        ...     index, columns, ndim=2, 
        ...     group_by=sectors
        ... )
        >>> print(grouped_wrapper.grouper.get_group_count())
        1

    Note:
        - 该类设计为不可变对象，任何修改都需要通过replace()方法创建新实例
        - 对于Series数据，应传递[sr.name]作为columns参数
        - 使用get_开头的方法获取分组感知的结果
    """

    def __init__(self,
                 index: tp.IndexLike,  # 行索引，支持类似pandas索引的对象
                 columns: tp.IndexLike,  # 列索引，支持类似pandas索引的对象
                 ndim: int,  # 数组维度数量，通常为1（Series）或2（DataFrame）
                 freq: tp.Optional[tp.FrequencyLike] = None,  # 时间序列频率，可选
                 column_only_select: tp.Optional[bool] = None,  # 是否仅对列进行索引，可选
                 group_select: tp.Optional[bool] = None,  # 是否对分组进行索引，可选
                 grouped_ndim: tp.Optional[int] = None,  # 分组后的维度数量，可选
                 **kwargs) -> None:  # 传递给ColumnGrouper的额外参数
        # 创建配置字典，用于存储所有构造参数
        config = dict(
            index=index,
            columns=columns,
            ndim=ndim,
            freq=freq,
            column_only_select=column_only_select,
            group_select=group_select,
            grouped_ndim=grouped_ndim,
        )

        # 参数验证：确保关键参数不为None
        checks.assert_not_none(index)  # 验证索引不为空
        checks.assert_not_none(columns)  # 验证列不为空
        checks.assert_not_none(ndim)  # 验证维度不为空
        
        # 索引标准化：将输入转换为pandas Index对象
        if not isinstance(index, pd.Index):  # 如果不是pandas Index
            index = pd.Index(index)  # 转换为pandas Index
        if not isinstance(columns, pd.Index):  # 如果不是pandas Index
            columns = pd.Index(columns)  # 转换为pandas Index

        # 实例属性初始化：使用下划线前缀表示私有属性
        self._index = index  # 存储行索引
        self._columns = columns  # 存储列索引
        self._ndim = ndim  # 存储维度数量
        self._freq = freq  # 存储频率信息
        self._column_only_select = column_only_select  # 存储列选择模式
        self._group_select = group_select  # 存储分组选择模式
        self._grouper = ColumnGrouper(columns, **kwargs)  # 创建列分组器实例
        self._grouped_ndim = grouped_ndim  # 存储分组后的维度

        # 父类初始化：按照多重继承的MRO顺序初始化
        PandasIndexer.__init__(self)  # 初始化pandas索引器功能
        Configured.__init__(self, **merge_dicts(config, self._grouper._config))  # 初始化配置管理功能

    @cached_method  # 缓存装饰器：避免重复计算，提高性能
    def indexing_func_meta(self: ArrayWrapperT,
                           pd_indexing_func: tp.PandasIndexingFunc,  # pandas索引函数
                           index: tp.Optional[tp.IndexLike] = None,  # 可选的新索引
                           columns: tp.Optional[tp.IndexLike] = None,  # 可选的新列索引
                           column_only_select: tp.Optional[bool] = None,  # 是否仅选择列
                           group_select: tp.Optional[bool] = None,  # 是否选择分组
                           group_by: tp.GroupByLike = None) -> IndexingMetaT:  # 分组依据
        """
        执行索引操作并返回索引元数据

        Args:
            pd_indexing_func: pandas索引函数，如obj.iloc[:, :2]中的lambda
            index: 新的行索引，如果为None则使用当前索引
            columns: 新的列索引，如果为None则自动推断
            column_only_select: 是否仅对列进行索引操作
            group_select: 是否对分组进行索引操作
            group_by: 重新分组的依据

        Returns:
            IndexingMetaT: 包含以下元素的元组
                - 新的ArrayWrapper实例
                - 行索引数组
                - 列/分组索引数组  
                - 未分组的列索引数组

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from vectorbt.base.array_wrapper import ArrayWrapper
            
            # 基本索引操作
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B', 'C']),
            ...     columns=pd.Index(['X', 'Y', 'Z']),
            ...     ndim=2
            ... )
            >>> # 选择前两列
            >>> new_wrapper, idx_idxs, col_idxs, ungrouped_col_idxs = wrapper.indexing_func_meta(
            ...     lambda x: x.iloc[:, :2]
            ... )
            >>> print(new_wrapper.columns)
            Index(['X', 'Y'], dtype='object')
        
        """
        from vectorbt._settings import settings  
        array_wrapper_cfg = settings['array_wrapper'] 

        if column_only_select is None:  
            column_only_select = self.column_only_select 
        if column_only_select is None:
            column_only_select = array_wrapper_cfg['column_only_select']
        if group_select is None: 
            group_select = self.group_select 
        if group_select is None: 
            group_select = array_wrapper_cfg['group_select']
            
        # 根据group_by参数创建新的ArrayWrapper实例_self
        _self = self.regroup(group_by) 
        group_select = group_select and _self.grouper.is_grouped() 
        
        # index = _self.index ∪ index
        if index is None:  
            index = _self.index 
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
            
        # columns = (group_select ? _self.grouper.get_columns() : _self.columns) ∪ columns
        if columns is None:
            if group_select:
                columns = _self.grouper.get_columns()
            else:
                columns = _self.columns 
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns) 
            
        # 创建中间包装器：用于索引操作的临时包装器
        if group_select: 
            # 分组作为列：使用分组后的维度
            i_wrapper = ArrayWrapper(index, columns, _self.get_ndim())
        else:
            i_wrapper = ArrayWrapper(index, columns, _self.ndim)
            
        n_rows = len(index)
        n_cols = len(columns)

        # 列选择模式处理：将包装器视为列的Series进行索引
        if column_only_select:
            if i_wrapper.ndim == 1:  # 如果已经是1维（单列）
                raise IndexingError("Columns only: This object already contains one column of data")
            try:
                # 创建列映射器：使用列索引创建一个Series进行索引操作
                col_mapper = pd_indexing_func(i_wrapper.wrap_reduced(np.arange(n_cols), columns=columns))
            except pd.core.indexing.IndexingError as e:
                warnings.warn("Columns only: Make sure to treat this object "
                              "as a Series of columns rather than a DataFrame", stacklevel=2)
                raise e
            # 处理索引结果：确定新的列索引和维度
            if checks.is_series(col_mapper):  # 如果返回Series（多列选择）
                new_columns = col_mapper.index  # 新列索引
                col_idxs = col_mapper.values  # 列索引数组
                new_ndim = 2  # 新维度为2
            else:  # 如果返回标量（单列选择）
                new_columns = columns[[col_mapper]]  # 选中的单列
                col_idxs = col_mapper  # 列索引
                new_ndim = 1  # 新维度为1
            new_index = index  # 索引保持不变
            idx_idxs = np.arange(len(index))  # 所有行索引
        else:
            # 标准索引模式：同时处理行和列的索引
            # 创建行映射器：用于确定选中的行索引
            idx_mapper = pd_indexing_func(i_wrapper.wrap(
                np.broadcast_to(np.arange(n_rows)[:, None], (n_rows, n_cols)),  # 广播行索引
                index=index,
                columns=columns
            ))
            # 处理1维情况
            if i_wrapper.ndim == 1:  # 如果是1维数组
                if not checks.is_series(idx_mapper):  # 必须返回Series
                    raise IndexingError("Selection of a scalar is not allowed")  # 不允许选择标量
                idx_idxs = idx_mapper.values  # 行索引数组
                col_idxs = 0  # 列索引为0（单列）
            else:
                # 2维情况：同时处理行和列索引
                # 创建列映射器：用于确定选中的列索引  
                col_mapper = pd_indexing_func(i_wrapper.wrap(
                    np.broadcast_to(np.arange(n_cols), (n_rows, n_cols)),  # 广播列索引
                    index=index,
                    columns=columns
                ))
                # 根据返回结果类型进行不同处理
                if checks.is_frame(idx_mapper):  # 如果返回DataFrame（多行多列）
                    idx_idxs = idx_mapper.values[:, 0]  # 行索引数组
                    col_idxs = col_mapper.values[0]  # 列索引数组
                elif checks.is_series(idx_mapper):  # 如果返回Series
                    # 检查选择模式：是单列还是单行
                    one_col = np.all(col_mapper.values == col_mapper.values.item(0))  # 是否选择单列
                    one_idx = np.all(idx_mapper.values == idx_mapper.values.item(0))  # 是否选择单行
                    if one_col and one_idx:  # 如果同时选择单行单列且重复
                        # 不允许在两个轴上都只选择一个索引且重复多次
                        raise IndexingError("Must select at least two unique indices in one of both axes")
                    elif one_col:  # 如果选择单列
                        # 单列选择：多行单列
                        idx_idxs = idx_mapper.values  # 行索引数组
                        col_idxs = col_mapper.values[0]  # 列索引（标量）
                    elif one_idx:  # 如果选择单行
                        # 单行选择：单行多列
                        idx_idxs = idx_mapper.values[0]  # 行索引（标量）
                        col_idxs = col_mapper.values  # 列索引数组
                    else:
                        raise IndexingError  # 其他情况抛出错误
                else:
                    raise IndexingError("Selection of a scalar is not allowed")  # 不允许选择标量
            # 从索引结果提取新的索引和列信息
            new_index = index_fns.get_index(idx_mapper, 0)  # 获取新的行索引
            if not isinstance(idx_idxs, np.ndarray):  # 如果选择单行
                # 单行选择：使用索引作为列名
                new_columns = index[[idx_idxs]]
            elif not isinstance(col_idxs, np.ndarray):  # 如果选择单列
                # 单列选择：使用列名
                new_columns = columns[[col_idxs]]
            else:
                # 多行多列选择：从结果中获取列索引
                new_columns = index_fns.get_index(idx_mapper, 1)
            new_ndim = idx_mapper.ndim  # 新维度

        # 分组处理：处理启用列分组的复杂情况
        if _self.grouper.is_grouped():  # 如果启用了列分组
            # 分组启用时的约束检查
            if np.asarray(idx_idxs).ndim == 0:  # 如果行索引是标量
                raise IndexingError("Flipping index and columns is not allowed")  # 不允许翻转索引和列

            if group_select:  # 如果基于分组进行选择
                # 基于分组的选择逻辑
                # 获取对应于所选分组的列索引
                group_idxs = col_idxs  # 分组索引就是列索引
                group_idxs_arr = reshape_fns.to_1d_array(group_idxs)  # 转换为1维数组
                # 获取每个分组的起始和结束列索引
                group_start_idxs = _self.grouper.get_group_start_idxs()[group_idxs_arr]  # 分组起始索引
                group_end_idxs = _self.grouper.get_group_end_idxs()[group_idxs_arr]  # 分组结束索引
                # 生成未分组的列索引数组：展开分组为具体的列索引
                ungrouped_col_idxs = get_ranges_arr(group_start_idxs, group_end_idxs)
                ungrouped_columns = _self.columns[ungrouped_col_idxs]  # 对应的未分组列名
                
                # 确定未分组后的维度
                if new_ndim == 1 and len(ungrouped_columns) == 1:  # 如果新维度是1且只有一列
                    ungrouped_ndim = 1  # 未分组维度为1
                    ungrouped_col_idxs = ungrouped_col_idxs[0]  # 取第一个索引
                else:
                    ungrouped_ndim = 2  # 未分组维度为2

                # 获取对应于新列的选定分组索引
                # 我们可以使用_self.group_by[ungrouped_col_idxs]，但索引操作可能改变了标签
                group_lens = _self.grouper.get_group_lens()[group_idxs_arr]  # 获取分组长度
                ungrouped_group_idxs = np.full(len(ungrouped_columns), 0)  # 初始化分组索引数组
                ungrouped_group_idxs[group_lens[:-1]] = 1  # 在分组边界处标记
                ungrouped_group_idxs = np.cumsum(ungrouped_group_idxs)  # 累积求和得到分组索引

                # 返回基于分组选择的结果
                return _self.replace(
                    index=new_index,  # 新的行索引
                    columns=ungrouped_columns,  # 未分组的列
                    ndim=ungrouped_ndim,  # 未分组的维度
                    grouped_ndim=new_ndim,  # 分组后的维度
                    group_by=new_columns[ungrouped_group_idxs]  # 新的分组依据
                ), idx_idxs, group_idxs, ungrouped_col_idxs

            # 基于列的选择（在分组模式下）
            col_idxs_arr = reshape_fns.to_1d_array(col_idxs)  # 转换列索引为1维数组
            return _self.replace(
                index=new_index,  # 新的行索引
                columns=new_columns,  # 新的列索引
                ndim=new_ndim,  # 新的维度
                grouped_ndim=None,  # 清除分组维度信息
                group_by=_self.grouper.group_by[col_idxs_arr]  # 根据列索引更新分组
            ), idx_idxs, col_idxs, col_idxs

        # 分组禁用时的处理：直接返回简单的替换结果
        return _self.replace(
            index=new_index,  # 新的行索引
            columns=new_columns,  # 新的列索引
            ndim=new_ndim,  # 新的维度
            grouped_ndim=None,  # 无分组维度
            group_by=None  # 无分组信息
        ), idx_idxs, col_idxs, col_idxs

    def indexing_func(self: ArrayWrapperT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> ArrayWrapperT:
        """
        执行索引操作
        
        这是indexing_func_meta的简化版本，只返回新的ArrayWrapper实例，
        不返回索引元数据。适用于只需要结果包装器的场景。
        
        Args:
            pd_indexing_func: pandas索引函数
            **kwargs: 传递给indexing_func_meta的额外参数
            
        Returns:
            ArrayWrapperT: 索引后的新ArrayWrapper实例
            
        Examples:
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B', 'C']),
            ...     columns=pd.Index(['X', 'Y', 'Z']),
            ...     ndim=2
            ... )
            >>> # 选择前两列
            >>> new_wrapper = wrapper.indexing_func(lambda x: x.iloc[:, :2])
            >>> print(new_wrapper.shape)
            (3, 2)
        """
        return self.indexing_func_meta(pd_indexing_func, **kwargs)[0]  # 只返回第一个元素（新包装器）

    @classmethod  # 类方法装饰器：可以通过类或实例调用
    def from_obj(cls: tp.Type[ArrayWrapperT], obj: tp.ArrayLike, *args, **kwargs) -> ArrayWrapperT:
        """
        从对象派生元数据创建ArrayWrapper
        
        该类方法从pandas对象或类似数组的对象中自动提取索引、列和维度信息，
        创建对应的ArrayWrapper实例。这是创建包装器的便捷方法。
        
        Args:
            obj: 类似数组的对象（pandas Series/DataFrame、NumPy数组等）
            *args: 传递给构造函数的位置参数
            **kwargs: 传递给构造函数的关键字参数
            
        Returns:
            ArrayWrapperT: 新创建的ArrayWrapper实例
            
        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            
            # 从DataFrame创建包装器
            >>> df = pd.DataFrame({
            ...     'A': [1, 2, 3],
            ...     'B': [4, 5, 6]
            ... }, index=['x', 'y', 'z'])
            >>> wrapper = ArrayWrapper.from_obj(df)
            >>> print(wrapper.shape)
            (3, 2)
            >>> print(wrapper.columns.tolist())
            ['A', 'B']
            
            # 从Series创建包装器
            >>> sr = pd.Series([1, 2, 3], index=['x', 'y', 'z'], name='values')
            >>> wrapper = ArrayWrapper.from_obj(sr)
            >>> print(wrapper.shape)
            (3,)
            >>> print(wrapper.name)
            values
            
            # 从NumPy数组创建包装器
            >>> arr = np.random.randn(4, 3)
            >>> wrapper = ArrayWrapper.from_obj(arr)
            >>> print(wrapper.shape)
            (4, 3)
        """
        pd_obj = to_pd_array(obj)  # 将输入对象转换为pandas对象
        index = index_fns.get_index(pd_obj, 0)  # 提取行索引（轴0）
        columns = index_fns.get_index(pd_obj, 1)  # 提取列索引（轴1）
        ndim = pd_obj.ndim  # 获取维度数量
        # 从kwargs中移除可能冲突的参数
        kwargs.pop('index', None)  # 移除index参数
        kwargs.pop('columns', None)  # 移除columns参数
        kwargs.pop('ndim', None)  # 移除ndim参数
        return cls(index, columns, ndim, *args, **kwargs)  # 创建新实例

    @classmethod  # 类方法装饰器
    def from_shape(cls: tp.Type[ArrayWrapperT], shape: tp.Shape, *args, **kwargs) -> ArrayWrapperT:
        """
        从形状创建ArrayWrapper
        
        该类方法根据给定的数组形状创建ArrayWrapper实例，使用默认的范围索引。
        适用于已知数组形状但还未有实际数据的场景。
        
        Args:
            shape: 数组形状元组，如(rows, cols)或(rows,)
            *args: 传递给构造函数的位置参数
            **kwargs: 传递给构造函数的关键字参数
            
        Returns:
            ArrayWrapperT: 新创建的ArrayWrapper实例
            
        Examples:
            >>> # 创建2D包装器
            >>> wrapper_2d = ArrayWrapper.from_shape((100, 5))
            >>> print(wrapper_2d.shape)
            (100, 5)
            >>> print(wrapper_2d.index[:3].tolist())
            [0, 1, 2]
            >>> print(wrapper_2d.columns.tolist())
            [0, 1, 2, 3, 4]
            
            # 创建1D包装器
            >>> wrapper_1d = ArrayWrapper.from_shape((50,))
            >>> print(wrapper_1d.shape)
            (50,)
            
            # 添加时间索引
            >>> import pandas as pd
            >>> wrapper_ts = ArrayWrapper.from_shape(
            ...     (30, 3),
            ...     index=pd.date_range('2024-01-01', periods=30),
            ...     columns=['A', 'B', 'C']
            ... )
            >>> print(wrapper_ts.index[0])
            2024-01-01 00:00:00
        """
        # 创建行索引：从0开始的范围索引
        index = pd.RangeIndex(start=0, step=1, stop=shape[0])
        # 创建列索引：如果是2D则使用第二维大小，否则为1
        columns = pd.RangeIndex(start=0, step=1, stop=shape[1] if len(shape) > 1 else 1)
        ndim = len(shape)  # 维度数量等于形状元组的长度
        return cls(index, columns, ndim, *args, **kwargs)  # 创建新实例

    @property  # 属性装饰器：将方法转换为只读属性
    def index(self) -> tp.Index:
        """
        行索引
        
        获取数组包装器的行索引，类似于pandas DataFrame或Series的index。
        
        Returns:
            tp.Index: pandas Index对象，表示行标签
            
        Examples:
            >>> wrapper = ArrayWrapper(
            ...     index=pd.DatetimeIndex(['2024-01-01', '2024-01-02']),
            ...     columns=['A', 'B'],
            ...     ndim=2
            ... )
            >>> print(wrapper.index)
            DatetimeIndex(['2024-01-01', '2024-01-02'], dtype='datetime64[ns]', freq=None)
        """
        return self._index  # 返回私有属性_index

    @property  # 属性装饰器
    def columns(self) -> tp.Index:
        """
        列索引
        
        获取数组包装器的列索引，类似于pandas DataFrame的columns。
        
        Returns:
            tp.Index: pandas Index对象，表示列标签
            
        Examples:
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['row1', 'row2']),
            ...     columns=pd.Index(['col1', 'col2', 'col3']),
            ...     ndim=2
            ... )
            >>> print(wrapper.columns)
            Index(['col1', 'col2', 'col3'], dtype='object')
        """
        return self._columns  # 返回私有属性_columns

    def get_columns(self, group_by: tp.GroupByLike = None) -> tp.Index:
        """
        获取分组感知的列索引
        
        根据分组情况返回相应的列索引。如果启用分组，返回分组标签；
        否则返回原始列标签。
        
        Args:
            group_by: 分组依据，可选
            
        Returns:
            tp.Index: 分组感知的列索引
            
        Examples:
            >>> # 无分组情况
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B']),
            ...     columns=pd.Index(['X', 'Y', 'Z']),
            ...     ndim=2
            ... )
            >>> print(wrapper.get_columns())
            Index(['X', 'Y', 'Z'], dtype='object')
            
            # 有分组情况
            >>> grouped_wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B']),
            ...     columns=pd.Index(['X1', 'X2', 'Y']),
            ...     ndim=2,
            ...     group_by=['group1', 'group1', 'group2']
            ... )
            >>> print(grouped_wrapper.get_columns())
            Index(['group1', 'group2'], dtype='object')
        """
        return self.resolve(group_by=group_by).columns  # 解析分组后返回列索引

    @property  # 属性装饰器
    def name(self) -> tp.Any:
        """
        名称
        
        对于1维数组包装器，返回其名称（类似Series.name）；
        对于2维数组包装器，返回None。
        
        Returns:
            tp.Any: 名称，1维时返回列名，2维时返回None
            
        Examples:
            >>> # 1维包装器有名称
            >>> wrapper_1d = ArrayWrapper(
            ...     index=pd.Index(['A', 'B', 'C']),
            ...     columns=pd.Index(['price']),
            ...     ndim=1
            ... )
            >>> print(wrapper_1d.name)
            price
            
            # 2维包装器无名称
            >>> wrapper_2d = ArrayWrapper(
            ...     index=pd.Index(['A', 'B']),
            ...     columns=pd.Index(['X', 'Y']),
            ...     ndim=2
            ... )
            >>> print(wrapper_2d.name)
            None
        """
        if self.ndim == 1:  # 如果是1维数组
            if self.columns[0] == 0:  # 如果列名是默认的0
                return None  # 返回None（类似Series没有名称）
            return self.columns[0]  # 返回第一个（也是唯一的）列名
        return None  # 2维数组返回None

    def get_name(self, group_by: tp.GroupByLike = None) -> tp.Any:
        """
        获取分组感知的名称
        
        Args:
            group_by: 分组依据，可选
            
        Returns:
            tp.Any: 分组感知的名称
            
        Examples:
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B']),
            ...     columns=pd.Index(['value']),
            ...     ndim=1
            ... )
            >>> print(wrapper.get_name())
            value
        """
        return self.resolve(group_by=group_by).name  # 解析分组后返回名称

    @property  # 属性装饰器
    def ndim(self) -> int:
        """
        维度数量
        
        返回数组的维度数量，1表示Series类型，2表示DataFrame类型。
        
        Returns:
            int: 维度数量
            
        Examples:
            >>> wrapper_1d = ArrayWrapper(pd.Index(['A']), pd.Index(['X']), ndim=1)
            >>> print(wrapper_1d.ndim)
            1
            
            >>> wrapper_2d = ArrayWrapper(pd.Index(['A']), pd.Index(['X', 'Y']), ndim=2)
            >>> print(wrapper_2d.ndim)
            2
        """
        return self._ndim  # 返回私有属性_ndim

    def get_ndim(self, group_by: tp.GroupByLike = None) -> int:
        return self.resolve(group_by=group_by).ndim

    @property
    def shape(self) -> tp.Shape:
        """
        数组形状
        
        返回数组的形状，类似于NumPy数组的shape属性。
        对于1维数组返回(rows,)，对于2维数组返回(rows, cols)。
        
        Returns:
            tp.Shape: 形状元组
            
        Examples:
            >>> # 1维包装器
            >>> wrapper_1d = ArrayWrapper(
            ...     index=pd.Index(['A', 'B', 'C']),
            ...     columns=pd.Index(['price']),
            ...     ndim=1
            ... )
            >>> print(wrapper_1d.shape)
            (3,)
            
            # 2维包装器
            >>> wrapper_2d = ArrayWrapper(
            ...     index=pd.Index(['A', 'B']),
            ...     columns=pd.Index(['X', 'Y', 'Z']),
            ...     ndim=2
            ... )
            >>> print(wrapper_2d.shape)
            (2, 3)
        """
        if self.ndim == 1:
            return len(self.index),
        return len(self.index), len(self.columns)

    def get_shape(self, group_by: tp.GroupByLike = None) -> tp.Shape:
        """
        获取分组感知的数组形状
        
        Args:
            group_by: 分组依据，可选
            
        Returns:
            tp.Shape: 分组感知的形状元组
            
        Examples:
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B']),
            ...     columns=pd.Index(['X1', 'X2', 'Y']),
            ...     ndim=2,
            ...     group_by=['G1', 'G1', 'G2']
            ... )
            >>> print(wrapper.get_shape())  # 分组后：2行，2个分组
            (2, 2)
        """
        return self.resolve(group_by=group_by).shape

    @property
    def shape_2d(self) -> tp.Shape:
        """
        二维形状
        
        将数组形状视为二维返回。对于1维数组，返回(rows, 1)；
        对于2维数组，返回实际形状。这对于统一处理不同维度的数组很有用。
        
        Returns:
            tp.Shape: 二维形状元组
            
        Examples:
            >>> wrapper_1d = ArrayWrapper(pd.Index(['A', 'B']), pd.Index(['X']), ndim=1)
            >>> print(wrapper_1d.shape_2d)
            (2, 1)
            
            >>> wrapper_2d = ArrayWrapper(pd.Index(['A', 'B']), pd.Index(['X', 'Y']), ndim=2)
            >>> print(wrapper_2d.shape_2d)
            (2, 2)
        """
        if self.ndim == 1:
            return self.shape[0], 1
        return self.shape

    def get_shape_2d(self, group_by: tp.GroupByLike = None) -> tp.Shape:
        """
        获取分组感知的二维形状
        
        Args:
            group_by: 分组依据，可选
            
        Returns:
            tp.Shape: 分组感知的二维形状元组
        """
        return self.resolve(group_by=group_by).shape_2d

    @property
    def freq(self) -> tp.Optional[pd.Timedelta]:
        """
        索引频率
        
        获取时间序列索引的频率。按优先级顺序检查：
        1. 构造函数中设置的频率
        2. 全局配置中的频率
        3. 时间索引的内置频率
        4. 时间索引的推断频率
        
        Returns:
            tp.Optional[pd.Timedelta]: 时间增量对象，如果无法确定频率则返回None
            
        Examples:
            >>> import pandas as pd
            >>> from vectorbt.base.array_wrapper import ArrayWrapper
            
            # 显式设置频率
            >>> wrapper = ArrayWrapper(
            ...     index=pd.date_range('2024-01-01', periods=5),
            ...     columns=['price'],
            ...     ndim=1,
            ...     freq='D'
            ... )
            >>> print(wrapper.freq)
            Timedelta('1 days 00:00:00')
            
            # 从时间索引自动推断频率
            >>> ts_index = pd.date_range('2024-01-01', periods=10, freq='H')
            >>> wrapper = ArrayWrapper(ts_index, ['value'], ndim=1)
            >>> print(wrapper.freq)
            Timedelta('0 days 01:00:00')
        """
        from vectorbt._settings import settings
        array_wrapper_cfg = settings['array_wrapper']

        freq = self._freq
        if freq is None:
            freq = array_wrapper_cfg['freq']
        if freq is not None:
            return freq_to_timedelta(freq)
        if isinstance(self.index, DatetimeIndexes):
            if self.index.freq is not None:
                return freq_to_timedelta(self.index.freq)
            if self.index.inferred_freq is not None:
                return freq_to_timedelta(self.index.inferred_freq)
        return freq

    def to_timedelta(self, a: tp.MaybeArray[float], to_pd: bool = False,
                     silence_warnings: tp.Optional[bool] = None) -> tp.Union[pd.Timedelta, np.timedelta64, tp.Array]:
        """
        将数组转换为时间增量
        
        使用ArrayWrapper的频率信息将数值数组转换为时间增量。
        这在计算持续时间、时间间隔等场景中非常有用。
        
        Args:
            a: 输入数值或数组
            to_pd: 是否转换为pandas Timedelta对象
            silence_warnings: 是否静默警告信息
            
        Returns:
            时间增量对象或数组
            
        Examples:
            >>> wrapper = ArrayWrapper(
            ...     index=pd.date_range('2024-01-01', periods=5, freq='D'),
            ...     columns=['price'],
            ...     ndim=1
            ... )
            >>> # 转换数值为时间增量
            >>> result = wrapper.to_timedelta(2.5)
            >>> print(result)  # 2.5天
            2.5 days 00:00:00
            
            # 转换数组
            >>> import numpy as np
            >>> result = wrapper.to_timedelta(np.array([1, 2, 3]))
            >>> print(result)
            [1 days 2 days 3 days]
            
        Raises:
            UserWarning: 当无法解析频率时发出警告
        """
        from vectorbt._settings import settings
        array_wrapper_cfg = settings['array_wrapper']

        if silence_warnings is None:
            silence_warnings = array_wrapper_cfg['silence_warnings']

        if self.freq is None:
            if not silence_warnings:
                warnings.warn("Couldn't parse the frequency of index. Pass it as `freq` or "
                              "define it globally under `settings.array_wrapper`.", stacklevel=2)
            return a
        if to_pd:
            return pd.to_timedelta(a * self.freq)
        return a * self.freq

    @property
    def column_only_select(self) -> tp.Optional[bool]:
        """
        是否仅对列执行索引
        
        当设置为True时，索引操作将对象视为列的Series而不是DataFrame。
        这在处理列向量化操作时很有用。
        
        Returns:
            tp.Optional[bool]: 是否启用列选择模式
        """
        return self._column_only_select

    @property
    def group_select(self) -> tp.Optional[bool]:
        """
        是否对分组执行索引
        
        当设置为True时，索引操作将基于列分组而不是单个列。
        只有在启用列分组时才有效。
        
        Returns:
            tp.Optional[bool]: 是否启用分组选择模式
        """
        return self._group_select

    @property
    def grouper(self) -> ColumnGrouper:
        """
        列分组器
        
        管理列分组功能的ColumnGrouper实例。提供分组相关的所有操作，
        如分组创建、分组索引获取、分组统计等。
        
        Returns:
            ColumnGrouper: 列分组器实例
        """
        return self._grouper

    @property
    def grouped_ndim(self) -> int:
        """
        分组后的维度数量
        
        在启用列分组时返回有效的维度数量。如果分组后只有一个分组，
        则返回1；如果有多个分组，则返回2。
        
        Returns:
            int: 分组后的维度数量
            
        Examples:
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B']),
            ...     columns=pd.Index(['X1', 'X2', 'Y1', 'Y2']),
            ...     ndim=2,
            ...     group_by=['G1', 'G1', 'G2', 'G2']
            ... )
            >>> print(wrapper.grouped_ndim)  # 2个分组，所以是2维
            2
        """
        if self._grouped_ndim is None:
            if self.grouper.is_grouped():
                return 2 if self.grouper.get_group_count() > 1 else 1
            return self.ndim
        return self._grouped_ndim

    def regroup(self: ArrayWrapperT, group_by: tp.GroupByLike, **kwargs) -> ArrayWrapperT:
        """
        基于self使用group_by重新构建一个
        重新分组对象
        
        根据新的分组依据创建新的ArrayWrapper实例。只有在分组发生变化时
        才创建新实例，否则返回自身以保持缓存有效性。
        
        Args:
            group_by: 新的分组依据
            **kwargs: 传递给replace方法的额外参数
            
        Returns:
            ArrayWrapperT: 重新分组后的ArrayWrapper实例
            
        Examples:
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B']),
            ...     columns=pd.Index(['X', 'Y', 'Z']),
            ...     ndim=2
            ... )
            >>> # 重新分组
            >>> grouped = wrapper.regroup(['G1', 'G1', 'G2'])
            >>> print(grouped.grouper.get_group_count())
            2
            
            # 取消分组
            >>> ungrouped = grouped.regroup(None)
            >>> print(ungrouped.grouper.is_grouped())
            False
        """
        if self.grouper.is_grouping_changed(group_by=group_by):
            self.grouper.check_group_by(group_by=group_by)
            grouped_ndim = None
            if self.grouper.is_grouped(group_by=group_by):
                if not self.grouper.is_group_count_changed(group_by=group_by):
                    grouped_ndim = self.grouped_ndim
            return self.replace(grouped_ndim=grouped_ndim, group_by=group_by, **kwargs)
        return self  # important for keeping cache

    @cached_method
    def resolve(self: ArrayWrapperT, group_by: tp.GroupByLike = None, **kwargs) -> ArrayWrapperT:
        """
        解析对象
        
        将列和其他元数据替换为分组信息，创建一个"解析"后的包装器，
        其中分组被视为实际的列。这是分组操作的核心方法。
        
        Args:
            group_by: 分组依据，可选
            **kwargs: 传递给regroup方法的额外参数
            
        Returns:
            ArrayWrapperT: 解析后的ArrayWrapper实例
            
        Examples:
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B']),
            ...     columns=pd.Index(['X', 'Y', 'Z']),
            ...     ndim=2,
            ...     group_by=['G1', 'G1', 'G2']
            ... )
            >>> resolved = wrapper.resolve()
            >>> print(resolved.columns)  # 显示分组标签
            Index(['G1', 'G2'], dtype='object')
            >>> print(resolved.shape)    # 形状变为(2, 2)
            (2, 2)
        """
        _self = self.regroup(group_by=group_by, **kwargs)
        if _self.grouper.is_grouped():
            return _self.replace(
                columns=_self.grouper.get_columns(),
                ndim=_self.grouped_ndim,
                grouped_ndim=None,
                group_by=None
            )
        return _self  # important for keeping cache

    def wrap(self,
             arr: tp.ArrayLike,
             index: tp.Optional[tp.IndexLike] = None,
             columns: tp.Optional[tp.IndexLike] = None,
             fillna: tp.Optional[tp.Scalar] = None,
             dtype: tp.Optional[tp.PandasDTypeLike] = None,
             group_by: tp.GroupByLike = None,
             to_timedelta: bool = False,
             to_index: bool = False,
             silence_warnings: tp.Optional[bool] = None) -> tp.SeriesFrame:
        """
        使用存储的元数据包装NumPy数组
        
        这是ArrayWrapper的核心功能之一，将NumPy数组转换为带有适当索引、
        列名和数据类型的pandas Series或DataFrame。执行以下处理流程：
        
        1) 转换为NumPy数组
        2) 填充NaN值（可选）
        3) 使用索引、列名和数据类型包装（可选）
        4) 转换为索引（可选）
        5) 使用ArrayWrapper.to_timedelta转换为时间增量（可选）
        
        Args:
            arr: 要包装的类似数组对象
            index: 行索引，默认使用包装器的索引
            columns: 列索引，默认使用包装器的列索引
            fillna: 用于填充NaN的值
            dtype: pandas数据类型
            group_by: 分组依据
            to_timedelta: 是否转换为时间增量
            to_index: 是否转换为索引值
            silence_warnings: 是否静默警告
            
        Returns:
            tp.SeriesFrame: 包装后的pandas Series或DataFrame
            
        Examples:
            >>> import numpy as np
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B', 'C']),
            ...     columns=pd.Index(['X', 'Y']),
            ...     ndim=2
            ... )
            >>> # 包装2D数组为DataFrame
            >>> data = np.random.randn(3, 2)
            >>> df = wrapper.wrap(data)
            >>> print(type(df))
            <class 'pandas.core.frame.DataFrame'>
            >>> print(df.shape)
            (3, 2)
            >>> print(df.index.tolist())
            ['A', 'B', 'C']
            
            # 包装1D数组为Series
            >>> wrapper_1d = ArrayWrapper(
            ...     index=pd.Index(['A', 'B', 'C']),
            ...     columns=pd.Index(['value']),
            ...     ndim=1
            ... )
            >>> data_1d = np.array([1, 2, 3])
            >>> sr = wrapper_1d.wrap(data_1d)
            >>> print(type(sr))
            <class 'pandas.core.series.Series'>
            
            # 填充NaN值
            >>> data_with_nan = np.array([1.0, np.nan, 3.0])
            >>> sr_filled = wrapper_1d.wrap(data_with_nan, fillna=0)
            >>> print(sr_filled.values)
            [1. 0. 3.]
            
            # 转换为时间增量（需要设置频率）
            >>> ts_wrapper = ArrayWrapper(
            ...     index=pd.date_range('2024-01-01', periods=3),
            ...     columns=['duration'],
            ...     ndim=1,
            ...     freq='D'
            ... )
            >>> durations = np.array([1, 2, 3])
            >>> sr_td = ts_wrapper.wrap(durations, to_timedelta=True)
            >>> print(sr_td.dtype)
            timedelta64[ns]
        """
        from vectorbt._settings import settings
        array_wrapper_cfg = settings['array_wrapper']

        if silence_warnings is None:
            silence_warnings = array_wrapper_cfg['silence_warnings']

        _self = self.resolve(group_by=group_by)

        if index is None:
            index = _self.index
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if columns is None:
            columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if len(columns) == 1:
            name = columns[0]
            if name == 0:  # was a Series before
                name = None
        else:
            name = None

        def _wrap(arr):
            # 转换为NumPy数组
            arr = np.asarray(arr)
            checks.assert_ndim(arr, (1, 2))
            # 填充NaN值
            if fillna is not None:
                arr[pd.isnull(arr)] = fillna
            # 调整数组维度
            arr = reshape_fns.soft_to_ndim(arr, self.ndim)
            # 验证形状匹配
            checks.assert_shape_equal(arr, index, axis=(0, 0))
            if arr.ndim == 2:
                checks.assert_shape_equal(arr, columns, axis=(1, 0))
            # 创建pandas对象
            if arr.ndim == 1:
                return pd.Series(arr, index=index, name=name, dtype=dtype)
            if arr.ndim == 2:
                if arr.shape[1] == 1 and _self.ndim == 1:
                    return pd.Series(arr[:, 0], index=index, name=name, dtype=dtype)
                return pd.DataFrame(arr, index=index, columns=columns, dtype=dtype)
            raise ValueError(f"{arr.ndim}-d input is not supported")

        out = _wrap(arr)
        if to_index:
            # Convert to index
            if checks.is_series(out):
                out = out.map(lambda x: self.index[x] if x != -1 else np.nan)
            else:
                out = out.applymap(lambda x: self.index[x] if x != -1 else np.nan)
        if to_timedelta:
            # Convert to timedelta
            out = self.to_timedelta(out, silence_warnings=silence_warnings)
        return out

    def wrap_reduced(self,
                     arr: tp.ArrayLike,
                     name_or_index: tp.NameIndex = None,
                     columns: tp.Optional[tp.IndexLike] = None,
                     fillna: tp.Optional[tp.Scalar] = None,
                     dtype: tp.Optional[tp.PandasDTypeLike] = None,
                     group_by: tp.GroupByLike = None,
                     to_timedelta: bool = False,
                     to_index: bool = False,
                     silence_warnings: tp.Optional[bool] = None) -> tp.MaybeSeriesFrame:
        """
        包装缩减操作的结果
        
        专门用于包装经过缩减操作（如求和、平均值、最大值等）后的结果。
        与wrap方法不同，这个方法能够智能处理维度降低的情况。
        
        name_or_index参数的用法：
        - 如果缩减为每列一个标量，则作为结果Series的名称
        - 如果缩减为每列一个数组，则作为结果Series/DataFrame的索引
        
        Args:
            arr: 缩减后的数组
            name_or_index: 结果的名称或索引
            columns: 列索引，默认使用包装器的列索引
            fillna: 用于填充NaN的值
            dtype: pandas数据类型
            group_by: 分组依据
            to_timedelta: 是否转换为时间增量
            to_index: 是否转换为索引值
            silence_warnings: 是否静默警告
            
        Returns:
            tp.MaybeSeriesFrame: 包装后的标量、Series或DataFrame
            
        Examples:
            >>> import numpy as np
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B', 'C']),
            ...     columns=pd.Index(['X', 'Y', 'Z']),
            ...     ndim=2
            ... )
            
            # 缩减为每列一个标量（如求和）
            >>> sums = np.array([10, 20, 30])  # 每列的和
            >>> result = wrapper.wrap_reduced(sums, name_or_index='sum')
            >>> print(type(result))
            <class 'pandas.core.series.Series'>
            >>> print(result.index.tolist())
            ['X', 'Y', 'Z']
            >>> print(result.name)
            sum
            
            # 缩减为每列一个数组（如滚动窗口）
            >>> rolling_means = np.random.randn(2, 3)  # 2个时间点，3列
            >>> time_index = pd.Index(['T1', 'T2'])
            >>> result = wrapper.wrap_reduced(rolling_means, name_or_index=time_index)
            >>> print(type(result))
            <class 'pandas.core.frame.DataFrame'>
            >>> print(result.shape)
            (2, 3)
            
            # 1维数组的缩减
            >>> wrapper_1d = ArrayWrapper(
            ...     index=pd.Index(['A', 'B', 'C']),
            ...     columns=pd.Index(['value']),
            ...     ndim=1
            ... )
            >>> # 缩减为标量
            >>> total = 15.0
            >>> result = wrapper_1d.wrap_reduced(total)
            >>> print(result)  # 返回标量值
            15.0
            
            # 缩减为数组
            >>> cumsum = np.array([1, 3, 6])
            >>> result = wrapper_1d.wrap_reduced(cumsum)
            >>> print(type(result))
            <class 'pandas.core.series.Series'>
            >>> print(result.name)
            value
        """
        from vectorbt._settings import settings
        array_wrapper_cfg = settings['array_wrapper']

        if silence_warnings is None:
            silence_warnings = array_wrapper_cfg['silence_warnings']

        checks.assert_not_none(self.ndim)
        _self = self.resolve(group_by=group_by)

        if columns is None:
            columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)

        if to_index:
            if dtype is None:
                dtype = np.int64
            if fillna is None:
                fillna = -1

        def _wrap_reduced(arr):
            nonlocal name_or_index  # 声明使用外层作用域中的name_or_index变量，允许函数内部修改该变量

            arr = np.asarray(arr)  # 将输入数组转换为NumPy数组格式，确保后续操作的兼容性
            if fillna is not None:  # 检查是否需要填充缺失值
                arr[pd.isnull(arr)] = fillna  # 使用pandas的isnull函数检测缺失值，并用fillna指定的值进行填充
            if arr.ndim == 0:  # 处理0维数组（标量）的情况
                # Scalar per Series/DataFrame
                return pd.Series(arr, dtype=dtype)[0]  # 将标量包装为Series再取第一个元素，确保返回正确的数据类型
            if arr.ndim == 1:  # 处理1维数组的情况
                if _self.ndim == 1:  # 如果原始包装器是1维的（Series类型）
                    if arr.shape[0] == 1:  # 如果数组只有一个元素
                        return pd.Series(arr, dtype=dtype)[0]  # 返回标量值，从单元素Series中提取
                    sr_name = columns[0]  # 获取第一个（也是唯一的）列名作为Series名称
                    if sr_name == 0:  # was arr Series before  # 如果列名是默认的0（表示原来是无名Series）
                        sr_name = None  # 将Series名称设为None，保持pandas的默认行为
                    if isinstance(name_or_index, str):  # 如果name_or_index是字符串类型
                        name_or_index = None  # 将其设为None，因为在1维情况下字符串应该用作名称而不是索引
                    return pd.Series(arr, index=name_or_index, name=sr_name, dtype=dtype)  # 创建Series，使用name_or_index作为索引，sr_name作为名称
                # Scalar per column in arr DataFrame  # 处理2维包装器缩减为1维的情况（每列一个标量）
                return pd.Series(arr, index=columns, name=name_or_index, dtype=dtype)  # 创建Series，列名作为索引，name_or_index作为Series名称
            if arr.ndim == 2:  # 处理2维数组的情况
                if arr.shape[1] == 1 and _self.ndim == 1:  # 如果数组只有一列且原始包装器是1维的
                    arr = reshape_fns.soft_to_ndim(arr, 1)  # 将2维数组转换为1维数组，去除多余的维度
                    # Array per Series  # 按Series处理
                    sr_name = columns[0]  # 获取列名作为Series名称
                    if sr_name == 0:  # was arr Series before  # 如果列名是默认的0
                        sr_name = None  # 设为None，保持原Series的无名状态
                    if isinstance(name_or_index, str):  # 如果name_or_index是字符串
                        name_or_index = None  # 设为None，字符串不能作为索引使用
                    return pd.Series(arr, index=name_or_index, name=sr_name, dtype=dtype)  # 创建Series
                # Array per column in DataFrame  # 处理标准的2维DataFrame情况
                if isinstance(name_or_index, str):  # 如果name_or_index是字符串类型
                    name_or_index = None  # 设为None，因为DataFrame不能使用字符串作为索引
                return pd.DataFrame(arr, index=name_or_index, columns=columns, dtype=dtype)  # 创建DataFrame，使用name_or_index作为行索引，columns作为列索引
            raise ValueError(f"{arr.ndim}-d input is not supported")  # 如果数组维度不是0、1、2，抛出不支持的维度错误

        out = _wrap_reduced(arr)
        if to_index:
            # Convert to index
            if checks.is_series(out):
                out = out.map(lambda x: self.index[x] if x != -1 else np.nan)
            elif checks.is_frame(out):
                out = out.applymap(lambda x: self.index[x] if x != -1 else np.nan)
            else:
                out = self.index[out] if out != -1 else np.nan
        if to_timedelta:
            # Convert to timedelta
            out = self.to_timedelta(out, silence_warnings=silence_warnings)
        return out

    def dummy(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """
        创建虚拟Series/DataFrame
        
        创建一个具有正确形状和元数据但包含未初始化数据的pandas对象。
        主要用于测试、占位符或作为后续填充的模板。
        
        Args:
            group_by: 分组依据，可选
            **kwargs: 传递给wrap方法的额外参数
            
        Returns:
            tp.SeriesFrame: 包含未初始化数据的pandas Series或DataFrame
            
        Examples:
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B', 'C']),
            ...     columns=pd.Index(['X', 'Y']),
            ...     ndim=2
            ... )
            >>> dummy_df = wrapper.dummy()
            >>> print(dummy_df.shape)
            (3, 2)
            >>> print(dummy_df.index.tolist())
            ['A', 'B', 'C']
            >>> print(dummy_df.columns.tolist())
            ['X', 'Y']
            # 注意：数据内容是未初始化的，可能包含随机值
        """
        _self = self.resolve(group_by=group_by)
        return _self.wrap(np.empty(_self.shape), **kwargs)

    def fill(self, fill_value: tp.Scalar, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """
        填充Series/DataFrame
        
        创建一个用指定値填充的pandas Series或DataFrame。
        所有元素都将被设置为相同的填充值。
        
        Args:
            fill_value: 填充值
            group_by: 分组依据，可选
            **kwargs: 传递给wrap方法的额外参数
            
        Returns:
            tp.SeriesFrame: 填充后的pandas Series或DataFrame
            
        Examples:
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B', 'C']),
            ...     columns=pd.Index(['X', 'Y']),
            ...     ndim=2
            ... )
            >>> # 用0填充
            >>> df_zeros = wrapper.fill(0)
            >>> print(df_zeros)
                X    Y
            A  0.0  0.0
            B  0.0  0.0
            C  0.0  0.0
            
            # 用NaN填充
            >>> import numpy as np
            >>> df_nan = wrapper.fill(np.nan)
            >>> print(df_nan.isna())
                X     Y
            A  True  True
            B  True  True
            C  True  True
            
            # 用字符串填充
            >>> df_str = wrapper.fill('empty')
            >>> print(df_str)
                    X      Y
            A  empty  empty
            B  empty  empty
            C  empty  empty
        """
        _self = self.resolve(group_by=group_by)
        return _self.wrap(np.full(_self.shape_2d, fill_value), **kwargs)

    def fill_reduced(self, fill_value: tp.Scalar, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """
        填充缩减后的Series/DataFrame
        
        创建一个用指定值填充的缩减形状的pandas对象。对于2维包装器，
        创建一个每列一个值的Series；对于1维包装器，创建一个标量值。
        
        Args:
            fill_value: 填充值
            group_by: 分组依据，可选
            **kwargs: 传递给wrap方法的额外参数
            
        Returns:
            tp.SeriesFrame: 填充后的缩减pandas对象
            
        Examples:
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B', 'C']),
            ...     columns=pd.Index(['X', 'Y', 'Z']),
            ...     ndim=2
            ... )
            >>> # 每列填充一个值
            >>> sr_reduced = wrapper.fill_reduced(1.0)
            >>> print(sr_reduced)
            X    1.0
            Y    1.0
            Z    1.0
            dtype: float64
            >>> print(type(sr_reduced))
            <class 'pandas.core.series.Series'>
            
            # 1维包装器的缩减
            >>> wrapper_1d = ArrayWrapper(
            ...     index=pd.Index(['A', 'B', 'C']),
            ...     columns=pd.Index(['value']),
            ...     ndim=1
            ... )
            >>> scalar_reduced = wrapper_1d.fill_reduced(42)
            >>> print(scalar_reduced)
            42
            >>> print(type(scalar_reduced))
            <class 'numpy.int64'>  # 或类似的标量类型
        """
        _self = self.resolve(group_by=group_by)
        return _self.wrap(np.full(_self.shape_2d[1], fill_value), **kwargs)


WrappingT = tp.TypeVar("WrappingT", bound="Wrapping")


class Wrapping(Configured, PandasIndexer, AttrResolver):
    """
    使用ArrayWrapper的全局包装类
    
    这是一个抽象基类，为需要使用ArrayWrapper功能的类提供统一的接口。
    它继承了三个重要的基类：
    - Configured: 提供配置管理功能
    - PandasIndexer: 提供pandas风格的索引操作
    - AttrResolver: 提供属性解析功能
    
    该类的主要目的是为vectorbt中的各种数据结构提供一致的包装器接口，
    使它们都能够享受ArrayWrapper提供的元数据管理、分组操作、索引功能等。
    
    主要特性：
    - 包装器集成：将ArrayWrapper作为核心组件
    - 索引操作：支持pandas风格的索引操作
    - 分组功能：支持列分组和重新分组
    - 属性解析：支持动态属性解析和频率处理
    - 列选择：提供便捷的单列/单组选择功能
    
    应用场景：
    - 技术指标类：如MACD、RSI等技术指标
    - 回测结果类：如Portfolio、Trades等
    - 统计分析类：如Returns、Drawdowns等
    - 自定义数据结构：需要包装器功能的用户自定义类
    
    Examples:
        >>> # 这是一个抽象基类，通常通过继承使用
        >>> class MyIndicator(Wrapping):
        ...     def __init__(self, data, **kwargs):
        ...         wrapper = ArrayWrapper.from_obj(data)
        ...         super().__init__(wrapper, **kwargs)
        ...         self._data = data
        ...     
        ...     @property
        ...     def data(self):
        ...         return self._data
        
        >>> # 使用示例
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> indicator = MyIndicator(df)
        >>> print(indicator.wrapper.shape)
        (3, 2)
        
        # 索引操作
        >>> sliced = indicator.iloc[:2]
        >>> print(sliced.wrapper.shape)
        (2, 2)
        
        # 分组操作
        >>> grouped = indicator.regroup(['G1', 'G2'])
        >>> print(grouped.wrapper.grouper.get_group_count())
        2
    
    Note:
        - 这是一个抽象基类，通常不直接实例化
        - 子类需要在构造函数中提供ArrayWrapper实例
        - 所有包装器相关的操作都会自动传播到新实例
    """

    def __init__(self, wrapper: ArrayWrapper, **kwargs) -> None:
        """
        构造函数
        
        Args:
            wrapper: ArrayWrapper实例，提供元数据管理功能
            **kwargs: 传递给父类的额外参数
            
        Raises:
            AssertionError: 当wrapper不是ArrayWrapper实例时
            
        Examples:
            >>> wrapper = ArrayWrapper(
            ...     index=pd.Index(['A', 'B']),
            ...     columns=pd.Index(['X', 'Y']),
            ...     ndim=2
            ... )
            >>> wrapping = Wrapping(wrapper)
            >>> print(wrapping.wrapper.shape)
            (2, 2)
        """
        checks.assert_instance_of(wrapper, ArrayWrapper)  # 验证wrapper类型
        self._wrapper = wrapper  # 存储包装器实例

        # 按照MRO顺序初始化父类
        Configured.__init__(self, wrapper=wrapper, **kwargs)
        PandasIndexer.__init__(self)
        AttrResolver.__init__(self)

    def indexing_func(self: WrappingT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> WrappingT:
        """
        对Wrapping对象执行索引操作
        
        通过pandas风格的索引函数对包装的对象进行索引操作，
        并返回新的Wrapping实例，其中包含更新后的ArrayWrapper。
        
        Args:
            pd_indexing_func: pandas索引函数
            **kwargs: 传递给ArrayWrapper.indexing_func的额外参数
            
        Returns:
            WrappingT: 索引后的新Wrapping实例
            
        Examples:
            # 假设有一个继承自Wrapping的类MyData
            >>> my_data = MyData(some_dataframe)
            >>> # 选择前两行
            >>> sliced = my_data.indexing_func(lambda x: x.iloc[:2])
            >>> print(sliced.wrapper.shape[0])  # 行数变为2
            
            # 选择特定列
            >>> selected = my_data.indexing_func(lambda x: x.iloc[:, [0, 2]])
            >>> print(selected.wrapper.shape[1])  # 列数变为2
        """
        return self.replace(wrapper=self.wrapper.indexing_func(pd_indexing_func, **kwargs))

    @property
    def wrapper(self) -> ArrayWrapper:
        """
        数组包装器
        
        获取与此Wrapping实例关联的ArrayWrapper对象。
        包装器提供了所有的元数据管理、索引操作、分组功能等。
        
        Returns:
            ArrayWrapper: 数组包装器实例
            
        Examples:
            >>> wrapping = MyWrappingClass(data)
            >>> wrapper = wrapping.wrapper
            >>> print(wrapper.shape)
            (100, 5)
            >>> print(wrapper.columns.tolist())
            ['col1', 'col2', 'col3', 'col4', 'col5']
            
            # 访问包装器的功能
            >>> print(wrapper.freq)  # 时间频率
            >>> print(wrapper.grouper.is_grouped())  # 是否分组
        """
        return self._wrapper

    def regroup(self: WrappingT, group_by: tp.GroupByLike, **kwargs) -> WrappingT:
        """
        重新分组此对象
        
        根据新的分组依据创建新的Wrapping实例。只有在分组发生变化时
        才创建新实例，否则返回自身以保持缓存有效性。
        
        Args:
            group_by: 分组依据，可以是列表或其他分组标识
            **kwargs: 传递给ArrayWrapper.regroup的额外参数
            
        Returns:
            WrappingT: 重新分组后的Wrapping实例
            
        Examples:
            >>> # 原始数据有4列
            >>> data = MyWrappingClass(df_4_columns)
            >>> print(data.wrapper.shape)
            (100, 4)
            
            # 将4列分为2组
            >>> grouped = data.regroup(['G1', 'G1', 'G2', 'G2'])
            >>> print(grouped.wrapper.get_shape())  # 分组后形状
            (100, 2)
            >>> print(grouped.wrapper.grouper.get_group_count())
            2
            
            # 取消分组
            >>> ungrouped = grouped.regroup(None)
            >>> print(ungrouped.wrapper.shape)
            (100, 4)
            
            # 如果分组没有变化，返回相同实例（保持缓存）
            >>> same_grouped = grouped.regroup(['G1', 'G1', 'G2', 'G2'])
            >>> print(same_grouped is grouped)
            True
        """
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            self.wrapper.grouper.check_group_by(group_by=group_by)
            return self.replace(wrapper=self.wrapper.regroup(group_by, **kwargs))
        return self  # important for keeping cache

    def resolve_self(self: AttrResolverT,
                     cond_kwargs: tp.KwargsLike = None,
                     custom_arg_names: tp.Optional[tp.Set[str]] = None,
                     impacts_caching: bool = True,
                     silence_warnings: tp.Optional[bool] = None) -> AttrResolverT:
        """
        解析自身
        
        如果在cond_kwargs中发现不同的频率，则创建此实例的副本。
        这是AttrResolver功能的一部分，用于动态处理参数变化。
        
        Args:
            cond_kwargs: 条件参数字典
            custom_arg_names: 自定义参数名称集合
            impacts_caching: 是否影响缓存
            silence_warnings: 是否静默警告
            
        Returns:
            AttrResolverT: 解析后的实例
            
        Examples:
            >>> data = MyWrappingClass(time_series_data)
            >>> print(data.wrapper.freq)
            Timedelta('1 days 00:00:00')
            
            # 传入不同的频率参数时会创建新实例
            >>> resolved = data.resolve_self({'freq': 'H'})
            >>> print(resolved.wrapper.freq)  
            Timedelta('0 days 01:00:00')
            >>> print(resolved is data)  # 不是同一个实例
            False
            
            # 如果频率相同，返回原实例
            >>> same_resolved = data.resolve_self({'freq': 'D'})
            >>> print(same_resolved is data)
            True
            
        Note:
            - 频率变化会创建对象副本，可能影响性能
            - 建议在对象创建时设置正确的频率以重用缓存
        """
        from vectorbt._settings import settings
        array_wrapper_cfg = settings['array_wrapper']

        if cond_kwargs is None:
            cond_kwargs = {}
        if custom_arg_names is None:
            custom_arg_names = set()
        if silence_warnings is None:
            silence_warnings = array_wrapper_cfg['silence_warnings']

        # 检查频率参数是否发生变化
        if 'freq' in cond_kwargs:
            wrapper_copy = self.wrapper.replace(freq=cond_kwargs['freq'])

            if wrapper_copy.freq != self.wrapper.freq:
                if not silence_warnings:
                    warnings.warn(f"Changing the frequency will create a copy of this object. "
                                  f"Consider setting it upon object creation to re-use existing cache.", stacklevel=2)
                self_copy = self.replace(wrapper=wrapper_copy)
                # 更新条件参数中的自引用
                for alias in self.self_aliases:
                    if alias not in custom_arg_names:
                        cond_kwargs[alias] = self_copy
                cond_kwargs['freq'] = self_copy.wrapper.freq
                if impacts_caching:
                    cond_kwargs['use_caching'] = False
                return self_copy
        return self

    def select_one(self: WrappingT, column: tp.Any = None, group_by: tp.GroupByLike = None, **kwargs) -> WrappingT:
        """
        选择一列/一组
        
        从多列/多组的对象中选择单列或单组，返回新的Wrapping实例。
        column参数可以是基于标签的位置，也可以是整数位置（如果标签查找失败）。
        
        Args:
            column: 要选择的列名或位置，None表示根据当前状态自动选择
            group_by: 分组依据，可选
            **kwargs: 传递给regroup的额外参数
            
        Returns:
            WrappingT: 选择后的Wrapping实例
            
        Raises:
            TypeError: 当对象已经只有一列/组或返回多列/组时
            KeyError: 当指定的列名/组名不存在时
            
        Examples:
            >>> # 多列数据
            >>> data = MyWrappingClass(dataframe_with_columns_ABC)
            >>> print(data.wrapper.shape)
            (100, 3)
            
            # 按标签选择
            >>> selected_A = data.select_one('A')
            >>> print(selected_A.wrapper.shape)
            (100,)  # 1维
            >>> print(selected_A.wrapper.name)
            A
            
            # 按位置选择
            >>> selected_0 = data.select_one(0)  # 选择第一列
            >>> print(selected_0.wrapper.name)
            A  # 第一列的名称
            
            # 分组场景
            >>> grouped_data = data.regroup(['G1', 'G1', 'G2'])
            >>> selected_G1 = grouped_data.select_one('G1')
            >>> print(selected_G1.wrapper.shape)
            (100,)  # 选择了G1组（对应2列）
            
            # 自动选择（当已经是单列时）
            >>> single_col_data = MyWrappingClass(series_data)
            >>> auto_selected = single_col_data.select_one()
            >>> print(auto_selected is single_col_data)  # 返回自身
            True
            
            # 错误情况：多列时未指定column
            >>> try:
            ...     data.select_one()  # 会抛出TypeError
            ... except TypeError as e:
            ...     print(e)
            Only one column is allowed. Use indexing or column argument.
        """
        _self = self.regroup(group_by, **kwargs)

        def _check_out_dim(out: WrappingT) -> WrappingT:
            """检查输出维度是否正确"""
            if _self.wrapper.grouper.is_grouped():
                if out.wrapper.grouped_ndim != 1:
                    raise TypeError("Could not select one group: multiple groups returned")
            else:
                if out.wrapper.ndim != 1:
                    raise TypeError("Could not select one column: multiple columns returned")
            return out

        if column is not None:
            # 指定了要选择的列/组
            if _self.wrapper.grouper.is_grouped():
                # 分组模式
                if _self.wrapper.grouped_ndim == 1:
                    raise TypeError("This object already contains one group of data")
                if column not in _self.wrapper.get_columns():
                    if isinstance(column, int):
                        # 尝试按位置索引
                        if _self.wrapper.column_only_select:
                            return _check_out_dim(_self.iloc[column])
                        return _check_out_dim(_self.iloc[:, column])
                    raise KeyError(f"Group '{column}' not found")
            else:
                # 非分组模式
                if _self.wrapper.ndim == 1:
                    raise TypeError("This object already contains one column of data")
                if column not in _self.wrapper.columns:
                    if isinstance(column, int):
                        # 尝试按位置索引
                        if _self.wrapper.column_only_select:
                            return _check_out_dim(_self.iloc[column])
                        return _check_out_dim(_self.iloc[:, column])
                    raise KeyError(f"Column '{column}' not found")
            return _check_out_dim(_self[column])
        
        # 未指定column，根据当前状态自动处理
        if not _self.wrapper.grouper.is_grouped():
            # 非分组模式
            if _self.wrapper.ndim == 1:
                return _self  # 已经是单列，直接返回
            raise TypeError("Only one column is allowed. Use indexing or column argument.")
        
        # 分组模式
        if _self.wrapper.grouped_ndim == 1:
            return _self  # 已经是单组，直接返回
        raise TypeError("Only one group is allowed. Use indexing or column argument.")

    @staticmethod
    def select_one_from_obj(obj: tp.SeriesFrame, wrapper: ArrayWrapper, column: tp.Any = None) -> tp.MaybeSeries:
        """
        从pandas对象中选择一列/一组
        
        这是select_one方法的静态版本，直接操作pandas对象而不是Wrapping实例。
        适用于在不创建Wrapping实例的情况下执行列选择操作。
        
        Args:
            obj: pandas Series或DataFrame对象
            wrapper: 关联的ArrayWrapper实例
            column: 要选择的列名或位置，None表示自动选择
            
        Returns:
            tp.MaybeSeries: 选择后的pandas Series，或在某些情况下返回原对象
            
        Raises:
            TypeError: 当对象已经只有一列/组或选择失败时
            KeyError: 当指定的列名/组名不存在时
            
        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
            >>> wrapper = ArrayWrapper.from_obj(df)
            
            # 按标签选择
            >>> series_A = Wrapping.select_one_from_obj(df, wrapper, 'A')
            >>> print(type(series_A))
            <class 'pandas.core.series.Series'>
            >>> print(series_A.name)
            A
            
            # 按位置选择
            >>> series_0 = Wrapping.select_one_from_obj(df, wrapper, 0)
            >>> print(series_0.name)
            A
            
            # 分组场景
            >>> grouped_wrapper = wrapper.regroup(['G1', 'G1', 'G2'])
            >>> # 注意：这里obj仍然是原始DataFrame，但wrapper是分组的
            >>> series_G1 = Wrapping.select_one_from_obj(df, grouped_wrapper, 'G1')
            >>> print(type(series_G1))
            <class 'pandas.core.series.Series'>
            
            # 自动选择单列
            >>> sr = pd.Series([1, 2, 3], name='value')
            >>> wrapper_1d = ArrayWrapper.from_obj(sr)
            >>> result = Wrapping.select_one_from_obj(sr, wrapper_1d)
            >>> print(result is sr)  # 返回原对象
            True
            
            # 错误情况
            >>> try:
            ...     Wrapping.select_one_from_obj(df, wrapper)  # 多列但未指定column
            ... except TypeError as e:
            ...     print(e)
            Only one column is allowed. Use indexing or column argument.
        """
        if column is not None:
            # 指定了要选择的列/组
            if wrapper.ndim == 1:
                raise TypeError("This object already contains one column of data")
            
            if wrapper.grouper.is_grouped():
                # 分组模式
                if column not in wrapper.get_columns():
                    if isinstance(column, int):
                        # 按位置选择
                        if isinstance(obj, pd.DataFrame):
                            return obj.iloc[:, column]
                        return obj.iloc[column]
                    raise KeyError(f"Group '{column}' not found")
            else:
                # 非分组模式
                if column not in wrapper.columns:
                    if isinstance(column, int):
                        # 按位置选择
                        if isinstance(obj, pd.DataFrame):
                            return obj.iloc[:, column]
                        return obj.iloc[column]
                    raise KeyError(f"Column '{column}' not found")
            return obj[column]
        
        # 未指定column，根据当前状态自动处理
        if not wrapper.grouper.is_grouped():
            # 非分组模式
            if wrapper.ndim == 1:
                return obj  # 已经是单列，直接返回
            raise TypeError("Only one column is allowed. Use indexing or column argument.")
        
        # 分组模式
        if wrapper.grouped_ndim == 1:
            return obj  # 已经是单组，直接返回
        raise TypeError("Only one group is allowed. Use indexing or column argument.")
