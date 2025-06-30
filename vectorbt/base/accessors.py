# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)


import numpy as np 
import pandas as pd  

from vectorbt import _typing as tp  
from vectorbt.base import combine_fns, index_fns, reshape_fns  

from vectorbt.base.array_wrapper import ArrayWrapper, Wrapping  

from vectorbt.base.column_grouper import ColumnGrouper 

from vectorbt.utils import checks  
from vectorbt.utils.config import merge_dicts, get_func_arg_names  

from vectorbt.utils.decorators import class_or_instancemethod, attach_binary_magic_methods, attach_unary_magic_methods

BaseAccessorT = tp.TypeVar("BaseAccessorT", bound="BaseAccessor")


@attach_binary_magic_methods(
    lambda self, other, np_func: self.combine(other, allow_multiple=False, combine_func=np_func))
@attach_unary_magic_methods(lambda self, np_func: self.apply(apply_func=np_func))
class BaseAccessor(Wrapping):
    """
    pandas Series和DataFrame的基础访问器类
    
    该类是vectorbt框架中所有访问器的基类，为pandas对象提供高性能的数组操作和量化分析功能。
    通过pandas的访问器机制（pd.Series.vbt和pd.DataFrame.vbt），用户可以无缝使用vectorbt
    的所有高级功能，而无需显式类型转换。
    
    核心特性：
    - **统一接口**: 为Series（1维）和DataFrame（2维）提供统一的操作接口
    - **高性能计算**: 内部使用NumPy进行计算，相比pandas提供10-100倍性能提升
    - **自动广播**: 实现类似NumPy的广播机制，自动处理不同形状数组的运算
    - **运算符重载**: 重载了算术、比较和逻辑运算符，提供直观的向量化操作
    - **维度处理**: 自动处理Series到DataFrame的转换，确保矩阵计算的一致性
    - **元数据保持**: 通过ArrayWrapper保持索引、列名等元数据信息
    
    设计原理：
    由于Series本质上是只有一列的DataFrame，为了避免为1维数据专门定义方法，
    内部会将任何Series转换为DataFrame进行矩阵计算，然后通过ArrayWrapper
    将2维输出转换回Series（如果需要）。这种设计确保了API的一致性和代码的简洁性。
    
    魔法方法装饰器说明：
    - @attach_binary_magic_methods: 自动生成二元运算符（+、-、*、/、//、%、**、&、|、^等）
      每个运算符都会调用combine方法，并传入对应的NumPy函数进行高性能计算
    - @attach_unary_magic_methods: 自动生成一元运算符（-、+、~、abs等）
      每个运算符都会调用apply方法，并传入对应的NumPy函数
    
    Args:
        obj (tp.SeriesFrame): pandas Series或DataFrame对象
        wrapper (tp.Optional[ArrayWrapper]): 可选的数组包装器，如果为None则自动创建
        **kwargs: 传递给ArrayWrapper和ColumnGrouper的配置参数
            
        Raises:
            AssertionError: 当obj不是pandas Series或DataFrame时抛出
    """

    def __init__(self, obj: tp.SeriesFrame, wrapper: tp.Optional[ArrayWrapper] = None, **kwargs) -> None:
        """
        初始化BaseAccessor实例
        
        该构造函数负责创建访问器实例，设置数组包装器，并配置各种计算参数。
        构造过程包括参数验证、包装器创建/配置、以及父类初始化等步骤。
        
        Args:
            obj: pandas Series或DataFrame对象，作为访问器操作的目标数据
            wrapper: 可选的ArrayWrapper实例，如果提供则使用其配置；否则自动创建
            **kwargs: 传递给ArrayWrapper和ColumnGrouper的配置参数
            
        Raises:
            AssertionError: 当obj不是pandas Series或DataFrame时抛出
        """
        # 参数验证：确保输入对象是pandas Series或DataFrame
        checks.assert_instance_of(obj, (pd.Series, pd.DataFrame))

        # 存储原始pandas对象的引用
        self._obj = obj

        # 分离包装器参数和分组器参数
        # 这一步将kwargs中的参数按照目标类（ArrayWrapper/ColumnGrouper）进行分类
        wrapper_arg_names = get_func_arg_names(ArrayWrapper.__init__)  # 获取ArrayWrapper构造函数的参数名
        grouper_arg_names = get_func_arg_names(ColumnGrouper.__init__)  # 获取ColumnGrouper构造函数的参数名
        wrapping_kwargs = dict()  # 存储包装相关的参数
        
        # 遍历kwargs，将相关参数移动到wrapping_kwargs中
        for k in list(kwargs.keys()):
            if k in wrapper_arg_names or k in grouper_arg_names:
                wrapping_kwargs[k] = kwargs.pop(k)  # 移动参数并从原kwargs中删除
        
        # 创建或配置ArrayWrapper实例
        if wrapper is None:
            # 如果没有提供wrapper，则从obj自动创建
            wrapper = ArrayWrapper.from_obj(obj, **wrapping_kwargs)
        else:
            # 如果提供了wrapper，则使用其配置并应用新的参数
            wrapper = wrapper.replace(**wrapping_kwargs)
        
        # 调用父类Wrapping的构造函数，完成基础设置
        Wrapping.__init__(self, wrapper, obj=obj, **kwargs)

    def __call__(self: BaseAccessorT, **kwargs) -> BaseAccessorT:
        """
        允许向初始化器传递参数的调用方法
        
        该方法使访问器实例可以像函数一样被调用，主要用于动态修改访问器的配置参数。
        这在需要临时改变分组方式、频率设置或其他配置时特别有用，而无需重新创建访问器。
        
        Args:
            **kwargs: 要传递给访问器初始化器的关键字参数，包括：
                     - group_by: 重新设置列分组方式
                     - freq: 重新设置时间序列频率
                     - column_only_select: 修改列选择模式
                     - group_select: 修改分组选择模式
                     以及其他ArrayWrapper和ColumnGrouper支持的参数
        
        Returns:
            BaseAccessorT: 具有新配置的访问器实例
            
        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'AAPL': [100, 102, 98],
            ...     'GOOGL': [2800, 2820, 2790], 
            ...     'MSFT': [300, 305, 295]
            ... })
            >>> 
            >>> # 原始访问器
            >>> acc = df.vbt
            >>> print(acc.wrapper.grouper.is_grouped())  # False
            >>> 
            >>> # 重新配置分组
            >>> sectors = ['Tech', 'Tech', 'Tech']
            >>> grouped_acc = acc(group_by=sectors)
            >>> print(grouped_acc.wrapper.grouper.is_grouped())  # True
        """
        return self.replace(**kwargs)

    @property
    def sr_accessor_cls(self) -> tp.Type["BaseSRAccessor"]:
        """
        Series访问器类的属性
        
        Returns:
            tp.Type["BaseSRAccessor"]: BaseSRAccessor类型，用于Series对象的访问器
        """
        return BaseSRAccessor

    @property
    def df_accessor_cls(self) -> tp.Type["BaseDFAccessor"]:
        """
        DataFrame访问器类的属性
        
        Returns:
            tp.Type["BaseDFAccessor"]: BaseDFAccessor类型，用于DataFrame对象的访问器
        """
        return BaseDFAccessor

    def indexing_func(self: BaseAccessorT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> BaseAccessorT:
        """
        对BaseAccessor执行索引操作
        
        该方法允许使用pandas风格的索引函数对访问器进行索引操作，并返回新的访问器实例。
        内部会调用包装器的indexing_func_meta方法获取索引元数据，然后基于结果创建相应的
        访问器类型（Series访问器或DataFrame访问器）。
        
        Args:
            pd_indexing_func: pandas索引函数，例如lambda x: x.iloc[:10, :2]
            **kwargs: 传递给wrapper.indexing_func_meta的额外参数
            
        Returns:
            BaseAccessorT: 索引后的新访问器实例，类型可能是Series或DataFrame访问器
            
        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'A': [1, 2, 3, 4],
            ...     'B': [5, 6, 7, 8],
            ...     'C': [9, 10, 11, 12]
            ... })
            >>> 
            >>> # 选择前两行和前两列
            >>> sub_acc = df.vbt.indexing_func(lambda x: x.iloc[:2, :2])
            >>> print(sub_acc.obj.shape)  # (2, 2)
            >>> 
            >>> # 选择单列（返回Series访问器）
            >>> col_acc = df.vbt.indexing_func(lambda x: x['A'])
            >>> print(type(col_acc).__name__)  # BaseSRAccessor
        """
        # 获取索引操作的元数据：新包装器、行索引、列索引、未分组列索引
        new_wrapper, idx_idxs, _, col_idxs = self.wrapper.indexing_func_meta(pd_indexing_func, **kwargs)
        
        # 使用索引结果从原始数组中提取数据
        new_obj = new_wrapper.wrap(self.to_2d_array()[idx_idxs, :][:, col_idxs], group_by=False)
        
        # 根据结果对象类型返回相应的访问器
        if checks.is_series(new_obj):
            # 如果结果是Series，返回Series访问器
            return self.replace(
                cls_=self.sr_accessor_cls,  # 使用Series访问器类
                obj=new_obj,               # 新的Series对象
                wrapper=new_wrapper        # 新的包装器
            )
        # 如果结果是DataFrame，返回DataFrame访问器
        return self.replace(
            cls_=self.df_accessor_cls,     # 使用DataFrame访问器类
            obj=new_obj,                   # 新的DataFrame对象
            wrapper=new_wrapper            # 新的包装器
        )

    @property
    def obj(self):
        """
        获取pandas对象
        
        Returns:
            pandas对象: 访问器包装的原始pandas Series或DataFrame对象
        """
        return self._obj

    @class_or_instancemethod
    def is_series(cls_or_self) -> bool:
        """
        判断是否为Series访问器（抽象方法）
        
        该方法在基类中未实现，需要在子类中重写。
        
        Returns:
            bool: 如果是Series访问器返回True，否则返回False
            
        Raises:
            NotImplementedError: 在基类中调用时抛出此异常
        """
        raise NotImplementedError

    @class_or_instancemethod
    def is_frame(cls_or_self) -> bool:
        """
        判断是否为DataFrame访问器（抽象方法）
        
        该方法在基类中未实现，需要在子类中重写。
        
        Returns:
            bool: 如果是DataFrame访问器返回True，否则返回False
            
        Raises:
            NotImplementedError: 在基类中调用时抛出此异常
        """
        raise NotImplementedError

    # ############# 创建方法 ############# #

    @classmethod
    def empty(cls, shape: tp.Shape, fill_value: tp.Scalar = np.nan, **kwargs) -> tp.SeriesFrame:
        """生成指定形状的空Series/DataFrame并用指定值填充"""
        if not isinstance(shape, tuple) or (isinstance(shape, tuple) and len(shape) == 1):
            # 单一整数或长度为1的元组：创建Series
            return pd.Series(np.full(shape, fill_value), **kwargs)
        # 多维元组：创建DataFrame
        return pd.DataFrame(np.full(shape, fill_value), **kwargs)

    @classmethod
    def empty_like(cls, other: tp.SeriesFrame, fill_value: tp.Scalar = np.nan, **kwargs) -> tp.SeriesFrame:
        """生成与指定Series/DataFrame结构相同的空对象并用指定值填充"""
        if checks.is_series(other):
            # 如果模板是Series，提取其形状、索引和名称
            return cls.empty(other.shape, fill_value=fill_value, 
                           index=other.index, name=other.name, **kwargs)
        # 如果模板是DataFrame，提取其形状、索引和列
        return cls.empty(other.shape, fill_value=fill_value, 
                       index=other.index, columns=other.columns, **kwargs)

    # ############# 索引和列操作 ############# #

    def apply_on_index(self, apply_func: tp.Callable, *args, axis: int = 1,
                       inplace: bool = False, **kwargs) -> tp.Optional[tp.SeriesFrame]:
        """
        对pandas对象的索引应用函数
        
        该方法允许对pandas对象的行索引（axis=0）或列索引（axis=1）应用自定义函数，
        实现索引的变换、重命名、重新排序等操作。这在数据预处理和索引标准化中非常有用。
        
        Args:
            apply_func: 要应用到索引上的可调用函数
                       函数签名：apply_func(index, *args, **kwargs) -> new_index
                       其中index是pandas.Index对象，返回值也应该是pandas.Index对象
            *args: 传递给apply_func的位置参数
            axis: 指定操作的轴
                 - 1: 操作列索引（默认）
                 - 0: 操作行索引
            inplace: 是否原地修改pandas对象
                    - True: 直接修改原对象，返回None
                    - False: 返回修改后的副本，原对象不变
            **kwargs: 传递给apply_func的关键字参数
            
        Returns:
            tp.Optional[tp.SeriesFrame]: 
            - 如果inplace=True，返回None
            - 如果inplace=False，返回修改后的pandas对象副本
            
        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'A': [1, 2, 3],
            ...     'B': [4, 5, 6]
            ... }, index=[10, 20, 30])
            >>> 
            >>> # 将列名转换为大写
            >>> def uppercase_columns(idx):
            ...     return idx.str.upper()
            >>> 
            >>> new_df = df.vbt.apply_on_index(uppercase_columns, axis=1)
            >>> print(new_df.columns.tolist())  # ['A', 'B'] -> ['A', 'B'] (已经是大写)
            >>> 
            >>> # 给行索引添加前缀
            >>> def add_prefix(idx, prefix):
            ...     return [f"{prefix}_{i}" for i in idx]
            >>> 
            >>> new_df = df.vbt.apply_on_index(add_prefix, "row", axis=0)
            >>> print(new_df.index.tolist())  # ['row_10', 'row_20', 'row_30']
            >>> 
            >>> # 原地修改
            >>> df.vbt.apply_on_index(add_prefix, "idx", axis=0, inplace=True)
            >>> print(df.index.tolist())  # ['idx_10', 'idx_20', 'idx_30']
        """
        # 验证axis参数的有效性
        checks.assert_in(axis, (0, 1))

        # 根据axis选择要操作的索引
        if axis == 1:
            obj_index = self.wrapper.columns  # 操作列索引
        else:
            obj_index = self.wrapper.index    # 操作行索引
        
        # 对选定的索引应用函数
        obj_index = apply_func(obj_index, *args, **kwargs)
        
        if inplace:
            # 原地修改：直接修改原始pandas对象的索引
            if axis == 1:
                self.obj.columns = obj_index
            else:
                self.obj.index = obj_index
            return None
        else:
            # 非原地修改：创建对象的副本并修改其索引
            obj = self.obj.copy()
            if axis == 1:
                obj.columns = obj_index
            else:
                obj.index = obj_index
            return obj

    def stack_index(self, index: tp.Index, on_top: bool = True, axis: int = 1,
                    inplace: bool = False, **kwargs) -> tp.Optional[tp.SeriesFrame]:
        """
        在指定轴上堆叠索引，创建多级索引结构
        
        该方法将新的索引与现有索引进行堆叠，创建MultiIndex结构。这在需要为数据添加
        额外的层次结构（如时间维度、分组维度）时非常有用。
        
        参考：vectorbt.base.index_fns.stack_indexes
        
        Args:
            index: 要堆叠的新索引，可以是pandas.Index或类似索引的对象
            on_top: 新索引的位置
                   - True: 新索引作为外层（顶层）索引
                   - False: 新索引作为内层（底层）索引
            axis: 指定操作的轴
                 - 1: 操作列索引（默认）
                 - 0: 操作行索引
            inplace: 是否原地修改
            **kwargs: 传递给index_fns.stack_indexes的额外参数
            
        Returns:
            tp.Optional[tp.SeriesFrame]: 根据inplace参数返回None或修改后的对象
            
        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'A': [1, 2, 3],
            ...     'B': [4, 5, 6]
            ... })
            >>> 
            >>> # 在列上添加顶层索引（策略层）
            >>> strategies = pd.Index(['Strategy1', 'Strategy1'], name='strategy')
            >>> new_df = df.vbt.stack_index(strategies, on_top=True, axis=1)
            >>> print(new_df.columns)
            >>> # MultiIndex([('Strategy1', 'A'), ('Strategy1', 'B')], names=['strategy', None])
            >>> 
            >>> # 在行上添加时间层
            >>> dates = pd.date_range('2024-01-01', periods=3, name='date')
            >>> new_df = df.vbt.stack_index(dates, on_top=True, axis=0)
            >>> print(new_df.index.names)  # ['date', None]
        """
        def apply_func(obj_index: tp.Index) -> tp.Index:
            """内部应用函数：根据on_top参数决定堆叠顺序"""
            if on_top:
                # 新索引在顶层：[new_index, existing_index]
                return index_fns.stack_indexes([index, obj_index], **kwargs)
            # 新索引在底层：[existing_index, new_index]
            return index_fns.stack_indexes([obj_index, index], **kwargs)

        # 调用apply_on_index执行索引操作
        return self.apply_on_index(apply_func, axis=axis, inplace=inplace)

    def drop_levels(self, levels: tp.MaybeLevelSequence, axis: int = 1,
                    inplace: bool = False, strict: bool = True) -> tp.Optional[tp.SeriesFrame]:
        """
        删除多级索引中的指定层级
        
        该方法用于删除MultiIndex中的一个或多个层级，简化索引结构。
        在数据分析中，当某些层级不再需要时，可以使用此方法清理索引。
        
        参考：vectorbt.base.index_fns.drop_levels
        
        Args:
            levels: 要删除的层级，可以是：
                   - 整数：层级的位置索引
                   - 字符串：层级的名称
                   - 列表：多个层级的位置索引或名称
            axis: 指定操作的轴（1为列，0为行）
            inplace: 是否原地修改
            strict: 是否严格模式
                   - True: 如果指定的层级不存在，会抛出异常
                   - False: 忽略不存在的层级
                   
        Returns:
            tp.Optional[tp.SeriesFrame]: 根据inplace参数返回None或修改后的对象
            
        Examples:
            >>> import pandas as pd
            >>> # 创建多级列索引的DataFrame
            >>> columns = pd.MultiIndex.from_tuples([
            ...     ('A', 'X'), ('A', 'Y'), ('B', 'X'), ('B', 'Y')
            ... ], names=['group', 'item'])
            >>> df = pd.DataFrame(np.random.randn(3, 4), columns=columns)
            >>> 
            >>> # 删除第一层级（group层）
            >>> simple_df = df.vbt.drop_levels(0, axis=1)
            >>> print(simple_df.columns.tolist())  # ['X', 'Y', 'X', 'Y']
            >>> 
            >>> # 删除指定名称的层级
            >>> simple_df = df.vbt.drop_levels('group', axis=1)
            >>> print(simple_df.columns.names)  # ['item']
        """
        def apply_func(obj_index: tp.Index) -> tp.Index:
            """内部应用函数：删除指定层级"""
            return index_fns.drop_levels(obj_index, levels, strict=strict)

        return self.apply_on_index(apply_func, axis=axis, inplace=inplace)

    def rename_levels(self, name_dict: tp.Dict[str, tp.Any], axis: int = 1,
                      inplace: bool = False, strict: bool = True) -> tp.Optional[tp.SeriesFrame]:
        """
        重命名多级索引中的层级名称
        
        该方法用于修改MultiIndex中各层级的名称，使索引结构更加语义化和易于理解。
        
        参考：vectorbt.base.index_fns.rename_levels
        
        Args:
            name_dict: 层级名称映射字典
                      键：当前层级名称（字符串）
                      值：新的层级名称（任意类型）
            axis: 指定操作的轴（1为列，0为行）
            inplace: 是否原地修改
            strict: 是否严格模式
                   - True: 如果字典中的键不存在，会抛出异常
                   - False: 忽略不存在的键
                   
        Returns:
            tp.Optional[tp.SeriesFrame]: 根据inplace参数返回None或修改后的对象
            
        Examples:
            >>> import pandas as pd
            >>> # 创建多级索引
            >>> columns = pd.MultiIndex.from_tuples([
            ...     ('A', 'X'), ('A', 'Y'), ('B', 'X'), ('B', 'Y')
            ... ], names=['old_group', 'old_item'])
            >>> df = pd.DataFrame(np.random.randn(3, 4), columns=columns)
            >>> 
            >>> # 重命名层级
            >>> rename_dict = {'old_group': 'sector', 'old_item': 'metric'}
            >>> new_df = df.vbt.rename_levels(rename_dict, axis=1)
            >>> print(new_df.columns.names)  # ['sector', 'metric']
        """
        def apply_func(obj_index: tp.Index) -> tp.Index:
            """内部应用函数：重命名层级"""
            return index_fns.rename_levels(obj_index, name_dict, strict=strict)

        return self.apply_on_index(apply_func, axis=axis, inplace=inplace)

    def select_levels(self, level_names: tp.MaybeLevelSequence, axis: int = 1,
                      inplace: bool = False) -> tp.Optional[tp.SeriesFrame]:
        """
        选择多级索引中的指定层级，删除其他层级
        
        该方法从MultiIndex中选择特定的层级，删除未选择的层级，实现索引的筛选和简化。
        
        参考：vectorbt.base.index_fns.select_levels
        
        Args:
            level_names: 要保留的层级名称或位置索引
                        可以是单个名称/索引或名称/索引的列表
            axis: 指定操作的轴（1为列，0为行）
            inplace: 是否原地修改
            
        Returns:
            tp.Optional[tp.SeriesFrame]: 根据inplace参数返回None或修改后的对象
            
        Examples:
            >>> import pandas as pd
            >>> # 创建三层多级索引
            >>> columns = pd.MultiIndex.from_tuples([
            ...     ('A', 'X', '1'), ('A', 'X', '2'), ('B', 'Y', '1'), ('B', 'Y', '2')
            ... ], names=['group', 'item', 'subitem'])
            >>> df = pd.DataFrame(np.random.randn(3, 4), columns=columns)
            >>> 
            >>> # 只保留第一层和第三层
            >>> selected_df = df.vbt.select_levels(['group', 'subitem'], axis=1)
            >>> print(selected_df.columns.names)  # ['group', 'subitem']
            >>> 
            >>> # 只保留指定位置的层级
            >>> selected_df = df.vbt.select_levels([0, 2], axis=1)
            >>> print(selected_df.columns.names)  # ['group', 'subitem']
        """
        def apply_func(obj_index: tp.Index) -> tp.Index:
            """内部应用函数：选择指定层级"""
            return index_fns.select_levels(obj_index, level_names)

        return self.apply_on_index(apply_func, axis=axis, inplace=inplace)

    def drop_redundant_levels(self, axis: int = 1, inplace: bool = False) -> tp.Optional[tp.SeriesFrame]:
        """
        删除多级索引中的冗余层级
        
        该方法自动识别并删除MultiIndex中只有单一唯一值的层级，这些层级不提供额外信息，
        删除后可以简化索引结构而不丢失数据区分度。
        
        参考：vectorbt.base.index_fns.drop_redundant_levels
        
        Args:
            axis: 指定操作的轴（1为列，0为行）
            inplace: 是否原地修改
            
        Returns:
            tp.Optional[tp.SeriesFrame]: 根据inplace参数返回None或修改后的对象
            
        Examples:
            >>> import pandas as pd
            >>> # 创建包含冗余层级的多级索引
            >>> columns = pd.MultiIndex.from_tuples([
            ...     ('A', 'same', 'X'), ('A', 'same', 'Y'), 
            ...     ('A', 'same', 'Z'), ('A', 'same', 'W')
            ... ], names=['group', 'redundant', 'item'])
            >>> df = pd.DataFrame(np.random.randn(3, 4), columns=columns)
            >>> 
            >>> # 删除冗余层级（'redundant'层所有值都是'same'）
            >>> clean_df = df.vbt.drop_redundant_levels(axis=1)
            >>> print(clean_df.columns.names)  # ['group', 'item']
        """
        def apply_func(obj_index: tp.Index) -> tp.Index:
            """内部应用函数：删除冗余层级"""
            return index_fns.drop_redundant_levels(obj_index)

        return self.apply_on_index(apply_func, axis=axis, inplace=inplace)

    def drop_duplicate_levels(self, keep: tp.Optional[str] = None, axis: int = 1,
                              inplace: bool = False) -> tp.Optional[tp.SeriesFrame]:
        """
        删除多级索引中的重复层级
        
        该方法删除MultiIndex中具有相同标签组合的重复层级，保留指定位置的层级。
        这在数据清理和索引标准化中很有用。
        
        参考：vectorbt.base.index_fns.drop_duplicate_levels
        
        Args:
            keep: 保留重复项的策略
                 - 'first': 保留第一个出现的重复项（默认）
                 - 'last': 保留最后一个出现的重复项
                 - None: 删除所有重复项
            axis: 指定操作的轴（1为列，0为行）
            inplace: 是否原地修改
            
        Returns:
            tp.Optional[tp.SeriesFrame]: 根据inplace参数返回None或修改后的对象
            
        Examples:
            >>> import pandas as pd
            >>> # 创建包含重复层级的多级索引
            >>> columns = pd.MultiIndex.from_tuples([
            ...     ('A', 'X'), ('A', 'Y'), ('A', 'X'), ('B', 'Z')  # ('A', 'X')重复
            ... ], names=['group', 'item'])
            >>> df = pd.DataFrame(np.random.randn(3, 4), columns=columns)
            >>> 
            >>> # 删除重复层级，保留首次出现的
            >>> clean_df = df.vbt.drop_duplicate_levels(keep='first', axis=1)
            >>> print(len(clean_df.columns))  # 3 (删除了一个重复的)
        """
        def apply_func(obj_index: tp.Index) -> tp.Index:
            """内部应用函数：删除重复层级"""
            return index_fns.drop_duplicate_levels(obj_index, keep=keep)

        return self.apply_on_index(apply_func, axis=axis, inplace=inplace)

    # ############# 数组重塑 ############# #

    def to_1d_array(self) -> tp.Array1d:
        """
        将pandas对象转换为1维NumPy数组
        
        该方法将pandas Series或DataFrame转换为一维NumPy数组。对于DataFrame，
        会按行优先（row-major）顺序展平数组。这在需要将多维数据转换为向量进行
        数值计算或机器学习算法处理时非常有用。
        
        参考：vectorbt.base.reshape_fns.to_1d
        
        Returns:
            tp.Array1d: 一维NumPy数组，包含原始数据的所有元素
            
        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> 
            >>> # Series转换为1维数组
            >>> sr = pd.Series([1, 2, 3, 4], name='data')
            >>> arr_1d = sr.vbt.to_1d_array()
            >>> print(arr_1d)  # [1 2 3 4]
            >>> print(arr_1d.shape)  # (4,)
            >>> 
            >>> # DataFrame转换为1维数组（按行展平）
            >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
            >>> arr_1d = df.vbt.to_1d_array()
            >>> print(arr_1d)  # [1 2 3 4]
            >>> print(arr_1d.shape)  # (4,)
        """
        return reshape_fns.to_1d_array(self.obj)

    def to_2d_array(self) -> tp.Array2d:
        """
        将pandas对象转换为2维NumPy数组
        该方法将pandas Series或DataFrame转换为二维NumPy数组。对于Series，
        会将其转换为列向量（shape为(n, 1)）。
        """
        return reshape_fns.to_2d_array(self.obj)

    def tile(self, n: int, keys: tp.Optional[tp.IndexLike] = None, axis: int = 1,
             wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        在指定轴上平铺（复制）pandas对象
        
        该方法将pandas对象沿指定轴复制n次，创建重复的数据结构。这在需要为多个策略、
        多个参数组合或多个场景创建相同基础数据的副本时非常有用。
        
        参考：vectorbt.base.reshape_fns.tile
        
        Args:
            n: 复制次数，必须是正整数
            keys: 可选的索引标签，用作最外层索引
                 如果提供，长度必须等于n，用于标识每个复制的副本
            axis: 复制的轴向
                 - 1: 沿列轴复制（默认），增加列数
                 - 0: 沿行轴复制，增加行数
            wrap_kwargs: 传递给数组包装器的额外参数
            
        Returns:
            tp.SeriesFrame: 平铺后的pandas对象
            
        Examples:
            >>> import pandas as pd
            >>> 
            >>> # Series沿列轴平铺
            >>> sr = pd.Series([1, 2, 3], index=['A', 'B', 'C'], name='original')
            >>> tiled_sr = sr.vbt.tile(3)
            >>> print(tiled_sr.shape)  # (3, 3)
            >>> print(tiled_sr.columns)  # RangeIndex([0, 1, 2])
            >>> 
            >>> # 使用自定义keys作为列名
            >>> strategies = ['momentum', 'reversal', 'arbitrage']
            >>> tiled_sr = sr.vbt.tile(3, keys=strategies, axis=1)
            >>> print(tiled_sr.columns.tolist())  # ['momentum', 'reversal', 'arbitrage']
            >>> 
            >>> # DataFrame沿列轴平铺
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> tiled_df = df.vbt.tile(2, keys=['scenario1', 'scenario2'], axis=1)
            >>> print(tiled_df.columns)
            >>> # MultiIndex([('scenario1', 'A'), ('scenario1', 'B'), 
            >>> #            ('scenario2', 'A'), ('scenario2', 'B')])
            >>> 
            >>> # 沿行轴平铺
            >>> tiled_df = df.vbt.tile(2, keys=['period1', 'period2'], axis=0)
            >>> print(tiled_df.index)
            >>> # MultiIndex([('period1', 0), ('period1', 1), 
            >>> #            ('period2', 0), ('period2', 1)])
        """
        # 调用reshape_fns.tile执行底层平铺操作
        tiled = reshape_fns.tile(self.obj, n, axis=axis)
        
        if keys is not None:
            # 如果提供了keys，创建多级索引结构
            if axis == 1:
                # 沿列轴平铺：keys作为外层列索引
                new_columns = index_fns.combine_indexes([keys, self.wrapper.columns])
                return ArrayWrapper.from_obj(tiled).wrap(
                    tiled.values, **merge_dicts(dict(columns=new_columns), wrap_kwargs))
            else:
                # 沿行轴平铺：keys作为外层行索引
                new_index = index_fns.combine_indexes([keys, self.wrapper.index])
                return ArrayWrapper.from_obj(tiled).wrap(
                    tiled.values, **merge_dicts(dict(index=new_index), wrap_kwargs))
        return tiled

    def repeat(self, n: int, keys: tp.Optional[tp.IndexLike] = None, axis: int = 1,
               wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        在指定轴上重复pandas对象的每个元素
        
        该方法与tile不同，repeat会将每个元素沿指定轴重复n次，而不是将整个对象复制n次。
        这在需要为每个数据点创建多个观测值或进行数据扩展时非常有用。
        
        参考：vectorbt.base.reshape_fns.repeat
        
        Args:
            n: 每个元素的重复次数，必须是正整数
            keys: 可选的索引标签，用作内层索引
                 如果提供，长度必须等于n，用于标识每个重复
            axis: 重复的轴向
                 - 1: 沿列轴重复（默认）
                 - 0: 沿行轴重复
            wrap_kwargs: 传递给数组包装器的额外参数
            
        Returns:
            tp.SeriesFrame: 重复后的pandas对象
            
        Examples:
            >>> import pandas as pd
            >>> 
            >>> # Series沿列轴重复每个元素
            >>> sr = pd.Series([1, 2, 3], index=['A', 'B', 'C'], name='original')
            >>> repeated_sr = sr.vbt.repeat(2)
            >>> print(repeated_sr.shape)  # (3, 2)
            >>> print(repeated_sr.values)
            >>> # [[1 1]
            >>> #  [2 2] 
            >>> #  [3 3]]
            >>> 
            >>> # 使用keys标识重复
            >>> repeated_sr = sr.vbt.repeat(3, keys=['rep1', 'rep2', 'rep3'], axis=1)
            >>> print(repeated_sr.columns)
            >>> # MultiIndex([('original', 'rep1'), ('original', 'rep2'), ('original', 'rep3')])
            >>> 
            >>> # DataFrame沿行轴重复
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> repeated_df = df.vbt.repeat(2, keys=['obs1', 'obs2'], axis=0)
            >>> print(repeated_df.shape)  # (4, 2)
            >>> print(repeated_df.index)
            >>> # MultiIndex([(0, 'obs1'), (0, 'obs2'), (1, 'obs1'), (1, 'obs2')])
        """
        # 调用reshape_fns.repeat执行底层重复操作
        repeated = reshape_fns.repeat(self.obj, n, axis=axis)
        
        if keys is not None:
            # 如果提供了keys，创建多级索引结构
            if axis == 1:
                # 沿列轴重复：原列索引作为外层，keys作为内层
                new_columns = index_fns.combine_indexes([self.wrapper.columns, keys])
                return ArrayWrapper.from_obj(repeated).wrap(
                    repeated.values, **merge_dicts(dict(columns=new_columns), wrap_kwargs))
            else:
                # 沿行轴重复：原行索引作为外层，keys作为内层
                new_index = index_fns.combine_indexes([self.wrapper.index, keys])
                return ArrayWrapper.from_obj(repeated).wrap(
                    repeated.values, **merge_dicts(dict(index=new_index), wrap_kwargs))
        return repeated

    def align_to(self, other: tp.SeriesFrame, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        将当前对象对齐到另一个pandas对象的轴结构
        
        该方法根据另一个pandas对象的索引和列结构，重新组织当前对象的数据，
        实现两个对象在轴维度上的对齐。这在需要确保多个数据源具有相同结构时非常有用。
        
        Args:
            other: 目标pandas Series或DataFrame对象，作为对齐的参考
            wrap_kwargs: 传递给数组包装器的额外参数
            
        Returns:
            tp.SeriesFrame: 对齐后的pandas对象，具有与other相同的索引和列结构
            
        Examples:
            >>> import pandas as pd
            >>> import vectorbt as vbt
            >>> 
            >>> # 创建待对齐的DataFrame
            >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=['x', 'y'], columns=['a', 'b'])
            >>> print(df1)
            >>> #    a  b
            >>> # x  1  2
            >>> # y  3  4
            >>> 
            >>> # 创建目标结构的DataFrame
            >>> df2 = pd.DataFrame([[5, 6, 7, 8], [9, 10, 11, 12]], index=['x', 'y'],
            ...     columns=pd.MultiIndex.from_arrays([[1, 1, 2, 2], ['a', 'b', 'a', 'b']]))
            >>> print(df2)
            >>> #       1       2    
            >>> #    a   b   a   b
            >>> # x  5   6   7   8
            >>> # y  9  10  11  12
            >>> 
            >>> # 将df1对齐到df2的结构
            >>> aligned_df1 = df1.vbt.align_to(df2)
            >>> print(aligned_df1)
            >>> #       1     2
            >>> #    a  b  a  b
            >>> # x  1  2  1  2
            >>> # y  3  4  3  4
            >>> 
            >>> # 列会被重复以匹配目标结构
            >>> print(aligned_df1.shape)  # (2, 4)
            >>> print(df2.shape)          # (2, 4)
        """
        # 验证other是pandas对象
        checks.assert_instance_of(other, (pd.Series, pd.DataFrame))
        
        # 将两个对象都转换为2维以便处理
        obj = reshape_fns.to_2d(self.obj)
        other = reshape_fns.to_2d(other)

        # 对齐索引：找到obj的索引在other索引中的对应位置
        aligned_index = index_fns.align_index_to(obj.index, other.index)
        # 对齐列：找到obj的列在other列中的对应位置
        aligned_columns = index_fns.align_index_to(obj.columns, other.columns)
        
        # 使用对齐的索引位置提取数据
        obj = obj.iloc[aligned_index, aligned_columns]
        
        # 用目标对象的索引和列包装结果
        return self.wrapper.wrap(
            obj.values, group_by=False,
            **merge_dicts(dict(index=other.index, columns=other.columns), wrap_kwargs))

    @class_or_instancemethod
    def broadcast(cls_or_self, *others: tp.Union[tp.ArrayLike, "BaseAccessor"], **kwargs) -> reshape_fns.BCRT:
        """
        广播当前对象与其他对象，使它们具有兼容的形状
        
        该方法实现了类似NumPy的广播机制，将不同形状的数组自动扩展到兼容的形状，
        以便进行元素级运算。这是vectorbt实现高效批量计算的核心功能之一。
        
        参考：vectorbt.base.reshape_fns.broadcast
        
        Args:
            *others: 要进行广播的其他对象，可以是：
                    - 类似数组的对象（NumPy数组、pandas对象等）
                    - BaseAccessor实例
            **kwargs: 传递给reshape_fns.broadcast的额外参数
            
        Returns:
            reshape_fns.BCRT: 广播后的对象元组，所有对象都具有兼容的形状
            
        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> 
            >>> # Series与标量广播
            >>> sr = pd.Series([1, 2, 3])
            >>> broadcasted = sr.vbt.broadcast(5)
            >>> print(broadcasted[0].shape)  # (3, 1)
            >>> print(broadcasted[1].shape)  # (3, 1)
            >>> 
            >>> # Series与DataFrame广播
            >>> df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
            >>> sr = pd.Series([10, 20, 30])
            >>> broadcasted = sr.vbt.broadcast(df)
            >>> print(broadcasted[0].shape)  # (3, 2)
            >>> print(broadcasted[1].shape)  # (3, 2)
            >>> 
            >>> # 多个对象同时广播
            >>> sr1 = pd.Series([1, 2, 3])
            >>> sr2 = pd.Series([10, 20])
            >>> arr = np.array([[100], [200], [300]])
            >>> broadcasted = sr1.vbt.broadcast(sr2, arr)
            >>> for i, obj in enumerate(broadcasted):
            ...     print(f"Object {i} shape: {obj.shape}")
            >>> # Object 0 shape: (3, 2)
            >>> # Object 1 shape: (3, 2) 
            >>> # Object 2 shape: (3, 2)
        """
        # 提取BaseAccessor对象中的pandas对象
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        
        if isinstance(cls_or_self, type):
            # 类方法调用：直接广播others中的对象
            return reshape_fns.broadcast(*others, **kwargs)
        # 实例方法调用：将self.obj加入广播
        return reshape_fns.broadcast(cls_or_self.obj, *others, **kwargs)

    def broadcast_to(self, other: tp.Union[tp.ArrayLike, "BaseAccessor"], **kwargs) -> reshape_fns.BCRT:
        """
        将当前对象广播到另一个对象的形状
        
        该方法将当前对象的形状扩展到与目标对象兼容，实现单向广播。
        这在需要将较小的数组扩展到较大数组的形状时非常有用。
        
        参考：vectorbt.base.reshape_fns.broadcast_to
        
        Args:
            other: 目标对象，可以是类似数组的对象或BaseAccessor实例
            **kwargs: 传递给reshape_fns.broadcast_to的额外参数
            
        Returns:
            reshape_fns.BCRT: 广播后的对象元组
            
        Examples:
            >>> import pandas as pd
            >>> 
            >>> # 将Series广播到DataFrame的形状
            >>> sr = pd.Series([1, 2, 3])
            >>> df = pd.DataFrame([[0, 0], [0, 0], [0, 0]])
            >>> broadcasted = sr.vbt.broadcast_to(df)
            >>> print(broadcasted[0].shape)  # (3, 2)
            >>> print(broadcasted[1].shape)  # (3, 2)
            >>> 
            >>> # 广播后的Series每行都重复了
            >>> print(broadcasted[0])
            >>> #    0  1
            >>> # 0  1  1
            >>> # 1  2  2
            >>> # 2  3  3
        """
        # 如果other是BaseAccessor，提取其pandas对象
        if isinstance(other, BaseAccessor):
            other = other.obj
        return reshape_fns.broadcast_to(self.obj, other, **kwargs)

    def make_symmetric(self) -> tp.Frame:  # pragma: no cover
        """
        将当前对象转换为对称矩阵
        
        该方法将pandas对象转换为对称的DataFrame，主要用于创建相关性矩阵、
        协方差矩阵或距离矩阵等对称数据结构。
        
        参考：vectorbt.base.reshape_fns.make_symmetric
        
        Returns:
            tp.Frame: 对称的DataFrame
            
        Examples:
            >>> import pandas as pd
            >>> 
            >>> # 将Series转换为对称矩阵
            >>> sr = pd.Series([1, 2, 3], name='data')
            >>> symmetric_df = sr.vbt.make_symmetric()
            >>> print(symmetric_df)
            >>> #      0    1    2
            >>> # 0  1.0  2.0  3.0
            >>> # 1  2.0  NaN  NaN
            >>> # 2  3.0  NaN  NaN
        """
        return reshape_fns.make_symmetric(self.obj)

    def unstack_to_array(self, **kwargs) -> tp.Array:  # pragma: no cover
        """
        将多级索引的pandas对象展开为NumPy数组
        
        该方法将具有MultiIndex的pandas对象展开为高维NumPy数组，
        每个索引层级对应数组的一个维度。
        
        参考：vectorbt.base.reshape_fns.unstack_to_array
        
        Args:
            **kwargs: 传递给reshape_fns.unstack_to_array的额外参数
            
        Returns:
            tp.Array: 展开后的多维NumPy数组
        """
        return reshape_fns.unstack_to_array(self.obj, **kwargs)

    def unstack_to_df(self, **kwargs) -> tp.Frame:  # pragma: no cover
        """
        将多级索引的pandas对象展开为DataFrame
        
        该方法将具有MultiIndex的pandas对象的某些层级展开为列，
        创建更宽的DataFrame结构。
        
        参考：vectorbt.base.reshape_fns.unstack_to_df
        
        Args:
            **kwargs: 传递给reshape_fns.unstack_to_df的额外参数
            
        Returns:
            tp.Frame: 展开后的DataFrame
        """
        return reshape_fns.unstack_to_df(self.obj, **kwargs)

    def to_dict(self, **kwargs) -> tp.Mapping:
        """
        将pandas对象转换为字典
        
        该方法将pandas对象转换为嵌套字典结构，保持数据的层次关系。
        这在需要将数据序列化或与其他系统交互时非常有用。
        
        参考：vectorbt.base.reshape_fns.to_dict
        
        Args:
            **kwargs: 传递给reshape_fns.to_dict的额外参数
            
        Returns:
            tp.Mapping: 转换后的字典对象
            
        Examples:
            >>> import pandas as pd
            >>> 
            >>> # Series转换为字典
            >>> sr = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
            >>> result = sr.vbt.to_dict()
            >>> print(result)  # {'a': 1, 'b': 2, 'c': 3}
            >>> 
            >>> # DataFrame转换为嵌套字典
            >>> df = pd.DataFrame({'X': [1, 2], 'Y': [3, 4]}, index=['row1', 'row2'])
            >>> result = df.vbt.to_dict()
            >>> print(result)  # {'X': {'row1': 1, 'row2': 2}, 'Y': {'row1': 3, 'row2': 4}}
        """
        return reshape_fns.to_dict(self.obj, **kwargs)

    # ############# Combining ############# #

    def apply(self, *args, apply_func: tp.Optional[tp.Callable] = None, keep_pd: bool = False,
              to_2d: bool = False, wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.SeriesFrame:
        """
        对当前对象应用自定义函数
        
        该方法是vectorbt中函数应用的核心方法，允许对pandas对象应用任意函数，
        同时保持对象的元数据（索引、列名等）。支持NumPy数组和pandas对象两种输入模式，
        以及可选的维度转换。
        
        Args:
            *args: 传递给apply_func的位置参数
            apply_func: 要应用的函数，必须指定
                       函数签名：apply_func(obj, *args, **kwargs) -> result
                       其中obj是输入数据，result必须与原始数据形状相同
                       可以是Numba编译的函数以获得更好性能
            keep_pd: 是否保持pandas对象格式
                    - True: 将输入作为pandas对象传递给函数
                    - False: 将输入转换为NumPy数组（默认，性能更好）
            to_2d: 是否强制转换为2维
                  - True: 输入转换为2维数组/DataFrame
                  - False: 保持原始维度（默认）
            wrap_kwargs: 传递给ArrayWrapper.wrap的额外参数
            **kwargs: 传递给apply_func的关键字参数
            
        Returns:
            tp.SeriesFrame: 应用函数后的结果，保持原始对象的索引和列结构
            
        Note:
            结果数组必须与原始数组具有相同的形状，否则可能导致索引不匹配的错误。
            
        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> 
            >>> # 基本函数应用
            >>> sr = pd.Series([1, 2, 3, 4], index=['A', 'B', 'C', 'D'])
            >>> squared = sr.vbt.apply(apply_func=lambda x: x ** 2)
            >>> print(squared)
            >>> # A     1
            >>> # B     4
            >>> # C     9
            >>> # D    16
            >>> 
            >>> # 使用额外参数
            >>> def scale_and_shift(data, scale, shift):
            ...     return data * scale + shift
            >>> 
            >>> scaled = sr.vbt.apply(2, 10, apply_func=scale_and_shift)
            >>> print(scaled)  # sr * 2 + 10
            >>> # A    12
            >>> # B    14
            >>> # C    16
            >>> # D    18
            >>> 
            >>> # DataFrame应用
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> normalized = df.vbt.apply(apply_func=lambda x: (x - x.mean()) / x.std())
            >>> 
            >>> # 保持pandas格式进行复杂操作
            >>> def rolling_mean(data, window):
            ...     return data.rolling(window).mean().fillna(data)
            >>> 
            >>> smoothed = sr.vbt.apply(3, apply_func=rolling_mean, keep_pd=True)
        """
        # 验证apply_func参数
        checks.assert_not_none(apply_func)
        
        # 根据参数准备输入数据
        if to_2d:
            # 强制转换为2维格式
            obj = reshape_fns.to_2d(self.obj, raw=not keep_pd)
        else:
            # 保持原始格式
            if not keep_pd:
                obj = np.asarray(self.obj)  # 转换为NumPy数组以提高性能
            else:
                obj = self.obj  # 保持pandas对象格式
        
        # 应用函数
        result = apply_func(obj, *args, **kwargs)
        
        # 包装结果并返回，保持原始对象的元数据
        return self.wrapper.wrap(result, group_by=False, **merge_dicts({}, wrap_kwargs))

    @class_or_instancemethod
    def concat(cls_or_self, *others: tp.ArrayLike, broadcast_kwargs: tp.KwargsLike = None,
               keys: tp.Optional[tp.IndexLike] = None) -> tp.Frame:
        """
        沿列轴连接当前对象与其他对象
        
        该方法将多个类似数组的对象沿列轴进行连接，创建更宽的DataFrame。
        在连接前会自动进行广播以确保所有对象具有兼容的形状。这在需要将多个数据源
        或多个计算结果合并为单一DataFrame时非常有用。
        
        Args:
            *others: 要连接的其他对象列表，可以是：
                    - NumPy数组
                    - pandas Series/DataFrame
                    - BaseAccessor实例
            broadcast_kwargs: 传递给broadcast函数的参数
            keys: 最外层列索引，用于标识不同的数据源
                 如果提供，长度应等于连接的对象总数
                 
        Returns:
            tp.Frame: 连接后的DataFrame
            
        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> 
            >>> # Series与DataFrame连接
            >>> sr = pd.Series([1, 2], index=['x', 'y'], name='series_data')
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])
            >>> result = sr.vbt.concat(df, keys=['series', 'dataframe'])
            >>> print(result)
            >>> #   series dataframe
            >>> #          a        b
            >>> # x      1        3        4
            >>> # y      2        5        6
            >>> 
            >>> # 多个Series连接
            >>> sr1 = pd.Series([1, 2], index=['x', 'y'])
            >>> sr2 = pd.Series([3, 4], index=['x', 'y'])  
            >>> sr3 = pd.Series([5, 6], index=['x', 'y'])
            >>> result = sr1.vbt.concat(sr2, sr3, keys=['col1', 'col2', 'col3'])
            >>> print(result.columns.tolist())  # ['col1', 'col2', 'col3']
            >>> 
            >>> # 不同形状对象的自动广播
            >>> sr = pd.Series([1, 2, 3])
            >>> arr = np.array([[10], [20], [30]])  # 3x1数组
            >>> result = sr.vbt.concat(arr)
            >>> print(result.shape)  # (3, 2)
        """
        # 提取BaseAccessor对象中的pandas对象
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        
        if isinstance(cls_or_self, type):
            # 类方法调用：直接连接others中的对象
            objs = others
        else:
            # 实例方法调用：将self.obj加入连接
            objs = (cls_or_self.obj,) + others
        
        # 设置广播参数
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        
        # 广播所有对象到兼容形状
        broadcasted = reshape_fns.broadcast(*objs, **broadcast_kwargs)
        # 确保所有对象都是2维的
        broadcasted = tuple(map(reshape_fns.to_2d, broadcasted))
        
        # 沿列轴连接
        out = pd.concat(broadcasted, axis=1, keys=keys)
        
        # 处理默认列索引的情况
        if not isinstance(out.columns, pd.MultiIndex) and np.all(out.columns == 0):
            out.columns = pd.RangeIndex(start=0, stop=len(out.columns), step=1)
        
        return out

    def apply_and_concat(self, ntimes: int, *args, apply_func: tp.Optional[tp.Callable] = None,
                         keep_pd: bool = False, to_2d: bool = False, numba_loop: bool = False,
                         use_ray: bool = False, keys: tp.Optional[tp.IndexLike] = None,
                         wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.Frame:
        """
        批量应用函数并沿列轴连接结果
        
        该方法是vectorbt中进行批量计算和参数扫描的核心工具之一。它会多次调用apply_func，
        每次传入不同的索引值（0到ntimes-1），然后将所有结果沿列轴连接成一个DataFrame。
        这在量化交易中进行策略参数优化、技术指标批量计算等场景中极其有用。
        
        参考：vectorbt.base.combine_fns.apply_and_concat_one
        
        Args:
            ntimes: 函数调用次数，必须是正整数
                   对应不同的参数组合、时间窗口或策略配置数量
            *args: 传递给apply_func的位置参数
            apply_func: 要应用的函数，必须指定
                       函数签名：apply_func(i, obj, *args, **kwargs) -> result
                       其中i是当前迭代索引(0到ntimes-1)，obj是输入数据
                       可以是Numba编译的函数以获得最佳性能
            keep_pd: 是否保持pandas对象格式
            to_2d: 是否强制转换为2维
            numba_loop: 是否使用Numba循环优化
                       - True: 使用Numba编译的高性能循环，但不支持关键字参数
                       - False: 使用Python循环，功能完整但速度较慢
            use_ray: 是否使用Ray进行分布式计算
                    - True: 使用Ray并行执行，适用于计算密集型任务
                    - False: 单机执行
                    注意：use_ray和numba_loop不能同时为True
            keys: 最外层列索引，用于标识不同的计算结果
                 如果为None，会自动创建'apply_idx'索引
            wrap_kwargs: 传递给ArrayWrapper.wrap的额外参数
            **kwargs: 传递给apply_func的关键字参数
            
        Returns:
            tp.Frame: 包含所有结果的DataFrame，列数为ntimes × 原始列数
            
        Note:
            - 函数返回的所有数组必须具有相同的形状以便连接
            - 使用Numba循环时性能最佳，但功能受限
            - Ray适用于小数据量但计算时间长的场景
            
        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> 
            >>> # 批量计算不同窗口的移动平均
            >>> prices = pd.Series([100, 102, 98, 105, 103, 107, 101])
            >>> windows = [3, 5, 7, 10]
            >>> 
            >>> def calc_sma(i, data, windows):
            ...     window = windows[i]
            ...     return data.rolling(window).mean().fillna(data)
            >>> 
            >>> smas = prices.vbt.apply_and_concat(
            ...     len(windows), windows,
            ...     apply_func=calc_sma, 
            ...     keep_pd=True,
            ...     keys=[f'SMA_{w}' for w in windows]
            ... )
            >>> print(smas.columns.tolist())  # ['SMA_3', 'SMA_5', 'SMA_7', 'SMA_10']
            >>> 
            >>> # 批量策略回测示例
            >>> def strategy_returns(i, prices, thresholds):
            ...     threshold = thresholds[i]
            ...     signals = (prices.pct_change() > threshold).astype(int)
            ...     return signals * prices.pct_change().shift(-1)  # 下期收益
            >>> 
            >>> thresholds = [0.01, 0.02, 0.03, 0.05]
            >>> results = prices.vbt.apply_and_concat(
            ...     len(thresholds), thresholds,
            ...     apply_func=strategy_returns,
            ...     keep_pd=True,
            ...     keys=[f'thresh_{t}' for t in thresholds]
            ... )
            >>> 
            >>> # 使用Ray进行并行计算（模拟）
            >>> # def slow_computation(i, data):
            >>> #     time.sleep(1)  # 模拟耗时计算
            >>> #     return data * (i + 1)
            >>> # 
            >>> # results = prices.vbt.apply_and_concat(
            >>> #     3, apply_func=slow_computation, use_ray=True
            >>> # )
        """
        # 验证apply_func参数
        checks.assert_not_none(apply_func)
        
        # 根据参数准备输入数据
        if to_2d:
            obj = reshape_fns.to_2d(self.obj, raw=not keep_pd)
        else:
            if not keep_pd:
                obj = np.asarray(self.obj)
            else:
                obj = self.obj
        
        # 根据不同的执行策略调用相应的函数
        if checks.is_numba_func(apply_func) and numba_loop:
            # Numba高性能循环
            if use_ray:
                raise ValueError("Ray cannot be used within Numba")
            result = combine_fns.apply_and_concat_one_nb(ntimes, apply_func, obj, *args, **kwargs)
        else:
            if use_ray:
                # Ray分布式计算
                result = combine_fns.apply_and_concat_one_ray(ntimes, apply_func, obj, *args, **kwargs)
            else:
                # 标准Python循环
                result = combine_fns.apply_and_concat_one(ntimes, apply_func, obj, *args, **kwargs)
        
        # 构建列索引层次结构
        if keys is not None:
            # 使用用户提供的keys
            new_columns = index_fns.combine_indexes([keys, self.wrapper.columns])
        else:
            # 创建默认的索引
            top_columns = pd.Index(np.arange(ntimes), name='apply_idx')
            new_columns = index_fns.combine_indexes([top_columns, self.wrapper.columns])
        
        # 包装结果并返回
        return self.wrapper.wrap(result, group_by=False, 
                               **merge_dicts(dict(columns=new_columns), wrap_kwargs))

    def combine(self, other: tp.MaybeTupleList[tp.Union[tp.ArrayLike, "BaseAccessor"]], *args,
                allow_multiple: bool = True, combine_func: tp.Optional[tp.Callable] = None,
                keep_pd: bool = False, to_2d: bool = False, concat: bool = False, numba_loop: bool = False,
                use_ray: bool = False, broadcast: bool = True, broadcast_kwargs: tp.KwargsLike = None,
                keys: tp.Optional[tp.IndexLike] = None, wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.SeriesFrame:
        """
        使用组合函数将当前对象与其他对象进行组合计算
        
        该方法是vectorbt中最强大和灵活的数组组合工具，支持多种组合模式：
        1. 一对一组合：当前对象与单个其他对象组合
        2. 一对多组合：当前对象与多个其他对象分别组合
        3. 连接模式：将组合结果沿列轴连接
        4. 聚合模式：将多个组合结果聚合为单个结果
        
        该方法广泛用于技术指标计算、策略信号生成、风险度量等量化分析场景。
        
        Args:
            other: 要组合的对象，可以是：
                  - 单个对象：执行一对一组合
                  - 对象列表/元组：执行一对多组合（当allow_multiple=True时）
                  对象类型可以是NumPy数组、pandas对象或BaseAccessor实例
            *args: 传递给combine_func的位置参数
            allow_multiple: 是否允许多对象组合
                          - True: other可以是对象列表，执行一对多组合
                          - False: other必须是单个对象，执行一对一组合
            combine_func: 组合函数，必须指定
                         函数签名：combine_func(obj1, obj2, *args, **kwargs) -> result
                         其中obj1是当前对象，obj2是other中的对象
                         可以是Numba编译的函数以获得最佳性能
            keep_pd: 是否保持pandas对象格式
            to_2d: 是否强制转换为2维
            concat: 组合结果的处理方式
                   - True: 将多个组合结果沿列轴连接（参见combine_and_concat）
                   - False: 将多个组合结果逐对聚合为单个结果（参见combine_multiple）
            numba_loop: 是否使用Numba循环优化
            use_ray: 是否使用Ray进行分布式计算
                    注意：仅在concat=True时支持Ray
            broadcast: 是否在组合前进行广播
                      - True: 自动广播所有输入对象到兼容形状
                      - False: 假设所有对象已具有兼容形状
            broadcast_kwargs: 传递给broadcast函数的参数
            keys: 最外层列索引（仅在concat=True时使用）
            wrap_kwargs: 传递给ArrayWrapper.wrap的额外参数
            **kwargs: 传递给combine_func的关键字参数
            
        Returns:
            tp.SeriesFrame: 组合后的结果
            - 如果concat=True：返回DataFrame，列数等于组合次数×原始列数
            - 如果concat=False：返回与原始对象相同形状的Series/DataFrame
            
        Note:
            - 如果combine_func是Numba编译的，会使用特殊的内存布局要求进行广播
            - 所有输入对象必须具有相同的数据类型（Numba编译函数要求）
            - Ray只能与concat=True一起使用
            
        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> 
            >>> # 基本一对一组合
            >>> sr1 = pd.Series([1, 2, 3])
            >>> sr2 = pd.Series([4, 5, 6])
            >>> result = sr1.vbt.combine(sr2, combine_func=lambda x, y: x + y)
            >>> print(result.values)  # [5, 7, 9]
            >>> 
            >>> # 一对多组合，逐对聚合
            >>> sr = pd.Series([1, 2, 3])
            >>> others = [pd.Series([1, 1, 1]), pd.Series([2, 2, 2]), pd.Series([3, 3, 3])]
            >>> result = sr.vbt.combine(others, combine_func=lambda x, y: x * y)
            >>> print(result.values)  # [6, 24, 54] (1*1*2*3, 2*1*2*3, 3*1*2*3)
            >>> 
            >>> # 一对多组合，连接结果
            >>> result = sr.vbt.combine(
            ...     others, 
            ...     combine_func=lambda x, y: x + y,
            ...     concat=True,
            ...     keys=['add1', 'add2', 'add3']
            ... )
            >>> print(result.columns.tolist())  # ['add1', 'add2', 'add3']
            >>> print(result.values)
            >>> # [[2, 3, 4],   # sr + [1,1,1], sr + [2,2,2], sr + [3,3,3]
            >>> #  [3, 4, 5],
            >>> #  [4, 5, 6]]
            >>> 
            >>> # 技术指标计算示例：计算多周期RSI
            >>> def rsi_like(prices, returns, period):
            ...     # 简化的RSI计算示例
            ...     gain = np.where(returns > 0, returns, 0)
            ...     loss = np.where(returns < 0, -returns, 0)
            ...     return gain / (gain + loss + 1e-8) * 100
            >>> 
            >>> prices = pd.Series([100, 102, 98, 105, 103, 107, 101])
            >>> returns = prices.pct_change().fillna(0)
            >>> periods = [3, 5, 7, 10]
            >>> 
            >>> rsi_results = returns.vbt.combine(
            ...     periods,
            ...     rsi_like,
            ...     combine_func=lambda rets, period: rsi_like(None, rets, period),
            ...     concat=True,
            ...     keys=[f'RSI_{p}' for p in periods]
            ... )
            >>> 
            >>> # 策略信号组合示例
            >>> def generate_signals(prices, ma_short, ma_long):
            ...     short_ma = prices.rolling(ma_short).mean()
            ...     long_ma = prices.rolling(ma_long).mean()
            ...     return (short_ma > long_ma).astype(int)
            >>> 
            >>> ma_pairs = [(5, 10), (5, 20), (10, 20), (10, 30)]
            >>> signals = prices.vbt.combine(
            ...     ma_pairs,
            ...     generate_signals,
            ...     combine_func=lambda p, pair: generate_signals(p, pair[0], pair[1]),
            ...     concat=True,
            ...     keep_pd=True,
            ...     keys=[f'MA_{s}_{l}' for s, l in ma_pairs]
            ... )
            >>> 
            >>> # 使用Ray进行并行计算（注释掉的示例）
            >>> # def slow_combination(x, y):
            >>> #     time.sleep(1)  # 模拟耗时计算
            >>> #     return x + y
            >>> # 
            >>> # result = sr.vbt.combine(
            >>> #     [1, 2, 3], 
            >>> #     combine_func=slow_combination,
            >>> #     concat=True, 
            >>> #     use_ray=True
            >>> # )
        """
        # 处理多对象输入
        if not allow_multiple or not isinstance(other, (tuple, list)):
            others = (other,)
        else:
            others = other
        
        # 提取BaseAccessor对象中的pandas对象
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        
        # 验证combine_func参数
        checks.assert_not_none(combine_func)
        
        # 广播处理
        if broadcast:
            if broadcast_kwargs is None:
                broadcast_kwargs = {}
            
            # 如果是Numba函数，需要特殊的内存布局
            if checks.is_numba_func(combine_func):
                # Numba要求可写数组和C连续内存布局
                broadcast_kwargs = merge_dicts(
                    dict(require_kwargs=dict(requirements=['W', 'C'])), 
                    broadcast_kwargs
                )
            
            # 广播所有对象
            new_obj, *new_others = reshape_fns.broadcast(self.obj, *others, **broadcast_kwargs)
        else:
            new_obj, new_others = self.obj, others
        
        # 确保new_obj是pandas对象
        if not checks.is_pandas(new_obj):
            new_obj = ArrayWrapper.from_shape(new_obj.shape).wrap(new_obj)
        
        # 准备输入数据
        if to_2d:
            inputs = tuple(map(lambda x: reshape_fns.to_2d(x, raw=not keep_pd), (new_obj, *new_others)))
        else:
            if not keep_pd:
                inputs = tuple(map(lambda x: np.asarray(x), (new_obj, *new_others)))
            else:
                inputs = new_obj, *new_others
        
        # 处理单一其他对象的情况
        if len(inputs) == 2:
            result = combine_func(inputs[0], inputs[1], *args, **kwargs)
            return ArrayWrapper.from_obj(new_obj).wrap(result, **merge_dicts({}, wrap_kwargs))
        
        # 处理多个其他对象的情况
        if concat:
            # 连接模式：将结果水平连接
            if checks.is_numba_func(combine_func) and numba_loop:
                if use_ray:
                    raise ValueError("Ray cannot be used within Numba")
                # 验证所有输入的元数据兼容性
                for i in range(1, len(inputs)):
                    checks.assert_meta_equal(inputs[i - 1], inputs[i])
                result = combine_fns.combine_and_concat_nb(
                    inputs[0], inputs[1:], combine_func, *args, **kwargs)
            else:
                if use_ray:
                    result = combine_fns.combine_and_concat_ray(
                        inputs[0], inputs[1:], combine_func, *args, **kwargs)
                else:
                    result = combine_fns.combine_and_concat(
                        inputs[0], inputs[1:], combine_func, *args, **kwargs)
            
            # 构建列索引
            columns = ArrayWrapper.from_obj(new_obj).columns
            if keys is not None:
                new_columns = index_fns.combine_indexes([keys, columns])
            else:
                top_columns = pd.Index(np.arange(len(new_others)), name='combine_idx')
                new_columns = index_fns.combine_indexes([top_columns, columns])
            
            return ArrayWrapper.from_obj(new_obj).wrap(
                result, **merge_dicts(dict(columns=new_columns), wrap_kwargs))
        else:
            # 聚合模式：逐对组合成单一结果
            if use_ray:
                raise ValueError("Ray cannot be used with concat=False")
            
            if checks.is_numba_func(combine_func) and numba_loop:
                # 验证所有输入的数据类型兼容性
                for i in range(1, len(inputs)):
                    checks.assert_dtype_equal(inputs[i - 1], inputs[i])
                result = combine_fns.combine_multiple_nb(inputs, combine_func, *args, **kwargs)
            else:
                result = combine_fns.combine_multiple(inputs, combine_func, *args, **kwargs)
            
            return ArrayWrapper.from_obj(new_obj).wrap(result, **merge_dicts({}, wrap_kwargs))


class BaseSRAccessor(BaseAccessor):
    """
    pandas Series的专用访问器类
    
    该类是BaseAccessor的子类，专门用于处理一维时间序列数据。它继承了BaseAccessor的
    所有功能，并针对Series的特点进行了优化和定制。通过pd.Series.vbt访问器，
    用户可以直接对Series对象使用vectorbt的所有高性能功能。
    
    核心特性：
    - **一维数据处理**: 专门优化用于处理时间序列、价格数据、指标值等一维数据
    - **自动转换**: 内部自动将Series转换为列向量进行矩阵运算，然后转换回Series
    - **元数据保持**: 完整保持Series的索引、名称等元数据信息
    - **性能优化**: 针对一维数据的特点进行了专门的性能优化
    - **完整功能**: 继承BaseAccessor的所有功能，包括重塑、组合、应用等
    
    主要应用场景：
    - **价格序列分析**: 股票价格、商品价格、指数等金融时间序列的分析
    - **技术指标计算**: 移动平均线、RSI、MACD等技术指标的计算
    - **收益率分析**: 收益率序列的统计分析和风险度量
    - **信号生成**: 基于规则的交易信号生成和回测
    - **时间序列变换**: 滚动计算、滞后变换、差分等时间序列操作
    
    与DataFrame访问器的区别：
    - Series访问器专注于一维数据，操作更加直接和高效
    - 某些操作（如tile、repeat）会自动扩展为DataFrame
    - 分组操作通常不适用于Series，除非将其转换为DataFrame
    
    Args:
        obj: pandas Series对象，必须是Series类型
        **kwargs: 传递给BaseAccessor的额外参数
        
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # 创建价格时间序列
        >>> dates = pd.date_range('2024-01-01', periods=100, freq='D')
        >>> prices = pd.Series(
        ...     100 * np.cumprod(1 + np.random.normal(0, 0.02, 100)),
        ...     index=dates,
        ...     name='AAPL'
        ... )
        >>> 
        >>> # 计算收益率
        >>> returns = prices.vbt.apply(apply_func=lambda x: np.diff(x, prepend=x[0]) / x)
        >>> 
        >>> # 批量计算多个窗口的移动平均
        >>> def sma(data, window):
        ...     return data.rolling(window).mean()
        >>> 
        >>> smas = prices.vbt.apply_and_concat(
        ...     4, [5, 10, 20, 50],
        ...     apply_func=lambda i, p, windows: sma(p, windows[i]),
        ...     keep_pd=True,
        ...     keys=[f'SMA_{w}' for w in [5, 10, 20, 50]]
        ... )
        >>> 
        >>> # 技术指标信号生成
        >>> short_ma = prices.rolling(10).mean()
        >>> long_ma = prices.rolling(30).mean()
        >>> signals = (short_ma.vbt > long_ma.vbt).astype(int)
        >>> 
        >>> # 与其他数据组合
        >>> volume = pd.Series(np.random.randint(1000, 10000, 100), index=dates)
        >>> price_volume = prices.vbt.concat(volume, keys=['price', 'volume'])
        >>> 
        >>> # 高性能向量化计算
        >>> normalized_prices = (prices.vbt - prices.vbt.mean()) / prices.vbt.std()
        
    Note:
        - 所有从BaseAccessor继承的方法都可以直接使用
        - 某些操作可能会返回DataFrame而非Series（如tile、repeat、concat等）
        - Series访问器特别适合金融时间序列数据的处理和分析
    """

    def __init__(self, obj: tp.Series, **kwargs) -> None:
        """
        初始化BaseSRAccessor实例
        
        Args:
            obj: pandas Series对象，必须是Series类型
            **kwargs: 传递给BaseAccessor的额外参数
            
        Raises:
            AssertionError: 当obj不是pandas Series时抛出
        """
        # 验证输入对象必须是pandas Series
        checks.assert_instance_of(obj, pd.Series)
        
        # 调用父类构造函数完成初始化
        BaseAccessor.__init__(self, obj, **kwargs)

    @class_or_instancemethod
    def is_series(cls_or_self) -> bool:
        """
        判断是否为Series访问器
        
        Returns:
            bool: 始终返回True，表示这是Series访问器
        """
        return True

    @class_or_instancemethod
    def is_frame(cls_or_self) -> bool:
        """
        判断是否为DataFrame访问器
        
        Returns:
            bool: 始终返回False，表示这不是DataFrame访问器
        """
        return False


class BaseDFAccessor(BaseAccessor):
    """
    pandas DataFrame的专用访问器类
    
    该类是BaseAccessor的子类，专门用于处理二维表格数据。它继承了BaseAccessor的
    所有功能，并针对DataFrame的特点进行了优化和定制。通过pd.DataFrame.vbt访问器，
    用户可以直接对DataFrame对象使用vectorbt的所有高性能功能。
    
    核心特性：
    - **多维数据处理**: 专门优化用于处理多资产、多策略、多指标等二维数据
    - **列级操作**: 支持按列进行操作，每列可以代表不同的资产或策略
    - **分组功能**: 完整支持列分组功能，可以将相关列组合在一起进行分析
    - **广播计算**: 支持复杂的广播操作，实现跨列和跨行的高效计算
    - **批量处理**: 能够同时处理多个时间序列，进行批量分析和比较
    
    主要应用场景：
    - **多资产分析**: 同时分析多只股票、多个资产的价格和收益率
    - **投资组合管理**: 多资产投资组合的收益、风险和权重管理
    - **策略比较**: 多个交易策略的同时回测和性能比较
    - **因子分析**: 多因子模型的构建和分析
    - **风险管理**: 相关性分析、VaR计算、风险暴露分析
    - **技术指标矩阵**: 多资产、多周期技术指标的批量计算
    
    与Series访问器的区别：
    - DataFrame访问器可以处理多列数据，支持更复杂的分析
    - 完整支持分组功能，可以按行业、地区等维度进行分组分析
    - 提供更丰富的数据重塑和组合功能
    - 适合进行多维度的金融数据分析
    
    Args:
        obj: pandas DataFrame对象，必须是DataFrame类型
        **kwargs: 传递给BaseAccessor的额外参数，特别有用的包括：
                 - group_by: 列分组依据，用于将相关列组合分析
                 - freq: 时间序列频率，用于时间相关计算
        
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # 创建多资产价格数据
        >>> dates = pd.date_range('2024-01-01', periods=252, freq='D')
        >>> assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        >>> 
        >>> # 模拟价格数据
        >>> np.random.seed(42)
        >>> returns = np.random.multivariate_normal(
        ...     mean=[0.001] * 4,
        ...     cov=np.eye(4) * 0.02**2,
        ...     size=252
        ... )
        >>> prices = pd.DataFrame(
        ...     100 * np.cumprod(1 + returns, axis=0),
        ...     index=dates,
        ...     columns=assets
        ... )
        >>> 
        >>> # 计算多资产收益率矩阵
        >>> returns_df = prices.vbt.apply(apply_func=lambda x: x.pct_change().fillna(0))
        >>> 
        >>> # 按行业分组分析
        >>> sectors = ['Tech', 'Tech', 'Tech', 'Auto']
        >>> sector_grouped = prices.vbt(group_by=sectors)
        >>> 
        >>> # 批量计算多周期移动平均线
        >>> def multi_sma(data, windows):
        ...     result = []
        ...     for window in windows:
        ...         result.append(data.rolling(window).mean())
        ...     return pd.concat(result, axis=1, keys=[f'SMA_{w}' for w in windows])
        >>> 
        >>> windows = [5, 10, 20, 50]
        >>> sma_matrix = prices.vbt.apply_and_concat(
        ...     len(windows), windows,
        ...     apply_func=lambda i, data, windows: data.rolling(windows[i]).mean(),
        ...     keep_pd=True,
        ...     keys=[f'SMA_{w}' for w in windows]
        ... )
        >>> 
        >>> # 投资组合权重分配
        >>> weights = pd.Series([0.3, 0.3, 0.3, 0.1], index=assets)
        >>> portfolio_returns = (returns_df.vbt * weights).sum(axis=1)
        >>> 
        >>> # 相关性分析
        >>> correlation_matrix = returns_df.vbt.apply(
        ...     apply_func=lambda x: pd.DataFrame(x).corr()
        ... )
        >>> 
        >>> # 风险度量批量计算
        >>> def calculate_var(returns, confidence=0.05):
        ...     return returns.quantile(confidence)
        >>> 
        >>> var_estimates = returns_df.vbt.apply(
        ...     0.05,
        ...     apply_func=calculate_var,
        ...     keep_pd=True
        ... )
        >>> 
        >>> # 策略信号矩阵生成
        >>> short_ma = prices.rolling(10).mean()
        >>> long_ma = prices.rolling(30).mean()
        >>> signals = (short_ma.vbt > long_ma.vbt).astype(int)
        >>> 
        >>> # 多策略组合分析
        >>> strategies = ['momentum', 'reversal']
        >>> strategy_returns = returns_df.vbt.tile(2, keys=strategies)
        >>> 
        >>> # 高性能数据对齐和广播
        >>> benchmark = pd.Series(
        ...     np.random.normal(0.0008, 0.015, 252), 
        ...     index=dates, 
        ...     name='benchmark'
        ... )
        >>> excess_returns = returns_df.vbt - benchmark.vbt
        
    Note:
        - 所有从BaseAccessor继承的方法都可以直接使用
        - 分组功能是DataFrame访问器的重要特性，通过group_by参数实现
        - 特别适合多资产、多策略的量化投资分析
        - 支持复杂的数据重塑和批量计算操作
    """

    def __init__(self, obj: tp.Frame, **kwargs) -> None:
        """
        初始化BaseDFAccessor实例
        
        Args:
            obj: pandas DataFrame对象，必须是DataFrame类型
            **kwargs: 传递给BaseAccessor的额外参数
            
        Raises:
            AssertionError: 当obj不是pandas DataFrame时抛出
        """
        # 验证输入对象必须是pandas DataFrame
        checks.assert_instance_of(obj, pd.DataFrame)
        
        # 调用父类构造函数完成初始化
        BaseAccessor.__init__(self, obj, **kwargs)

    @class_or_instancemethod
    def is_series(cls_or_self) -> bool:
        """
        判断是否为Series访问器
        
        Returns:
            bool: 始终返回False，表示这不是Series访问器
        """
        return False

    @class_or_instancemethod
    def is_frame(cls_or_self) -> bool:
        """
        判断是否为DataFrame访问器
        
        Returns:
            bool: 始终返回True，表示这是DataFrame访问器
        """
        return True
