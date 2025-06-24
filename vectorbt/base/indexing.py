# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
1. 为所有持有pandas对象的自定义类提供一致的索引操作方式，包括iloc、loc、xs等标准pandas索引方法，
   确保可以像操作单个DataFrame一样操作复杂的组合对象。

2. 通过索引转发机制，将对复杂类的索引操作转换为对其内部所有pandas对象的相同索引操作，
   然后重新构造新的类实例，实现了数据结构的不可变性和操作的一致性。

3. 提供了强大的参数化索引功能（ParamLoc），允许用户通过参数值而非列标签来查询数据。

4. 通过build_param_indexer工厂函数，可以动态生成具有特定参数索引能力的类。
"""

import numpy as np  
import pandas as pd  

from vectorbt import _typing as tp  
from vectorbt.base import index_fns, reshape_fns  
from vectorbt.utils import checks  


class IndexingError(Exception):
    """索引操作异常类
    
    当索引操作过程中发生错误时抛出的异常。
    继承自Python标准异常类Exception，用于标识vectorbt索引系统中的特定错误。
    """
    pass


# 定义索引基类的类型变量，用于泛型类型注解
# 这允许IndexingBase的子类方法返回正确的类型
IndexingBaseT = tp.TypeVar("IndexingBaseT", bound="IndexingBase")


class IndexingBase:
    """
    为所有需要支持pandas风格索引操作的类提供了基础接口。
    子类需要重写indexing_func方法，该方法应该：
        1. 接收一个pandas索引函数作为参数 pd_indexing_func
        2. 将该函数应用到所有相关的pandas对象上
        3. 使用索引后的pandas对象构造并返回新的类实例
    """

    def indexing_func(self: IndexingBaseT, pd_indexing_func: tp.Callable, **kwargs) -> IndexingBaseT:
        """索引函数接口
        
        这是索引操作的核心接口方法，子类必须重写此方法来实现具体的索引逻辑。
        
        Args:
            pd_indexing_func: pandas索引函数，如lambda x: x.iloc[0:5]
            **kwargs: 传递给索引操作的额外关键字参数
            
        Returns:
            IndexingBaseT: 索引操作后的新实例
            
        Raises:
            NotImplementedError: 基类中未实现，子类必须重写
            
        示例:
            ```python
            class MyClass(IndexingBase):
                def __init__(self, df1, df2):
                    self.df1 = df1
                    self.df2 = df2
                    
                def indexing_func(self, pd_indexing_func):
                    return MyClass(
                        pd_indexing_func(self.df1),
                        pd_indexing_func(self.df2)
                    )
            ```
        """
        raise NotImplementedError


class LocBase:
    def __init__(self, indexing_func: tp.Callable, **kwargs) -> None:
        self._indexing_func = indexing_func  
        self._indexing_kwargs = kwargs  

    @property
    def indexing_func(self) -> tp.Callable:
        return self._indexing_func

    @property
    def indexing_kwargs(self) -> dict:
        return self._indexing_kwargs

    def __getitem__(self, key: tp.Any) -> tp.Any:
        raise NotImplementedError


class iLoc(LocBase):
    def __getitem__(self, key: tp.Any) -> tp.Any:
        return self.indexing_func(lambda x: x.iloc.__getitem__(key), **self.indexing_kwargs)

class Loc(LocBase):
    def __getitem__(self, key: tp.Any) -> tp.Any:
        return self.indexing_func(lambda x: x.loc.__getitem__(key), **self.indexing_kwargs)


# 定义PandasIndexer的类型变量，用于泛型类型注解
PandasIndexerT = tp.TypeVar("PandasIndexerT", bound="PandasIndexer")


class PandasIndexer(IndexingBase):
    """
    提供了完整的pandas风格索引功能。
    该类实现了iloc、loc、xs和__getitem__等索引方法，使得自定义类可以像pandas对象一样进行索引操作。
    
    继承该类时，实现indexing_func方法（IndexingBase），该方法接收 pd_indexing_func: tp.Callable 来操作子类中的 pandas 属性
        .iloc[key]——>._iloc.__getitem__(key)——>indexing_func(lambda x: x.iloc.__getitem__(key))
        .loc[key]——>._loc.__getitem__(key)——>indexing_func(lambda x: x.loc.__getitem__(key))
        .xs(*args, **kwargs)——>indexing_func(lambda x: x.xs(*args, **kwargs))
        [key]——>.__getitem__(key)——>indexing_func(lambda x: x.__getitem__(key))
    
    示例:
        ```python
        class Portfolio(PandasIndexer):
            def __init__(self, returns_df, positions_df):
                self.returns = returns_df
                self.positions = positions_df
                super().__init__()
                
            def indexing_func(self, pd_indexing_func):
                return Portfolio(
                    pd_indexing_func(self.returns),
                    pd_indexing_func(self.positions)
                )
        
        portfolio = Portfolio(returns_df, positions_df)
        # 现在可以像操作DataFrame一样操作Portfolio
        recent_data = portfolio.iloc[-30:]
        ```
    """

    def __init__(self, **kwargs) -> None:
        """初始化Pandas索引器
        
        Args:
            **kwargs: 传递给索引操作的关键字参数，这些参数会被保存并
                     在每次索引操作时传递给indexing_func方法
        """
        self._iloc = iLoc(self.indexing_func, **kwargs)
        self._loc = Loc(self.indexing_func, **kwargs)
        self._indexing_kwargs = kwargs

    @property
    def indexing_kwargs(self) -> dict:
        return self._indexing_kwargs

    @property
    def iloc(self) -> iLoc:
        return self._iloc

    # 复制iLoc类的文档字符串到iloc属性
    iloc.__doc__ = iLoc.__doc__

    @property
    def loc(self) -> Loc:
        return self._loc

    loc.__doc__ = Loc.__doc__

    def xs(self: PandasIndexerT, *args, **kwargs) -> PandasIndexerT:
        return self.indexing_func(lambda x: x.xs(*args, **kwargs), **self.indexing_kwargs)

    def __getitem__(self: PandasIndexerT, key: tp.Any) -> PandasIndexerT:
        return self.indexing_func(lambda x: x.__getitem__(key), **self.indexing_kwargs)


class ParamLoc(LocBase):
    """参数位置索引器
    
    该类 ParamLoc 的实例 x 作为某个类 A 的属性时，传入一个函数给 self._indexing_func
        这个函数接收一个函数作为参数，并且应当使用接收的这个函数去处理类 A 的其它 DataFrame 属性
    假设类 A 的某实例 *：*.x[key]——>*.x.__getitem__(key)
        获取 key 在 x._mapper.values 上对应的整数索引数组
        然后返回 self._indexing_func(pd_indexing_func)，其中 pd_indexing_func 从参数 obj 中选择 indices 列并删除 x.level_name列，并返回新的 obj
    """

    def __init__(self, mapper: tp.Series, indexing_func: tp.Callable, level_name: tp.Level = None, **kwargs) -> None:
        """初始化参数位置索引器
        
        Args:
            mapper: 参数映射Series，索引为列标签，值为对应的参数值
                   例如：pd.Series(['param1', 'param1', 'param2'], index=['col1', 'col2', 'col3'])
            indexing_func: 索引函数，用于执行实际的索引操作
            level_name: 级别名称，当需要在结果中删除特定级别时使用
            **kwargs: 传递给索引函数的额外关键字参数
            
        Raises:
            AssertionError: 当mapper不是pandas.Series类型时抛出
        """
        checks.assert_instance_of(mapper, pd.Series)

        if mapper.dtype == 'O':  # 'O'表示object类型
            # 如果参数是对象类型，必须先转换为字符串类型
            # 原始mapper不会被修改，这里创建副本进行转换
            mapper = mapper.astype(str)
            
        self._mapper = mapper  # 存储参数映射关系
        self._level_name = level_name  # 存储级别名称

        LocBase.__init__(self, indexing_func, **kwargs)

    @property
    def mapper(self) -> tp.Series:
        return self._mapper

    @property
    def level_name(self) -> tp.Level:
        return self._level_name

    def get_indices(self, key: tp.Any) -> tp.Array1d:
        """获取key在mapper.values上对应的整数索引数组
        
        示例:
            假设有mapper = pd.Series(['A', 'A', 'B'], index=[0, 1, 2])
            - get_indices('A') 返回 [0, 1]
            - get_indices(['A', 'B']) 返回 [0, 1, 2]
            - get_indices(slice('A', 'B')) 返回 [0, 1, 2]
        """
        # 如果mapper是对象类型，需要对key进行相应的字符串转换
        if self.mapper.dtype == 'O':
            if isinstance(key, slice):
                # 处理切片对象，转换start和stop为字符串
                start = str(key.start) if key.start is not None else None
                stop = str(key.stop) if key.stop is not None else None
                key = slice(start, stop, key.step)
            elif isinstance(key, (list, np.ndarray)):
                key = list(map(str, key))
            else:
                key = str(key)
                
        mapper = pd.Series(np.arange(len(self.mapper.index)), index=self.mapper.values)
        
        indices = mapper.loc.__getitem__(key)
        
        if isinstance(indices, pd.Series):
            indices = indices.values
            
        return indices

    def __getitem__(self, key: tp.Any) -> tp.Any:
        """
        [key]——>__getitem__(key)：
            获取key在self._mapper.values上对应的整数索引数组
            定义一个函数，从参数 obj 中选择 indices 列并删除self.level_name列，并返回新的 obj
            然后返回 self.indexing_func(pd_indexing_func, **self.indexing_kwargs)
        """
        # 获取参数对应的列索引位置
        indices = self.get_indices(key)
        
        # 判断是否为多选操作，影响后续的级别删除逻辑
        is_multiple = isinstance(key, (slice, list, np.ndarray))

        def pd_indexing_func(obj: tp.SeriesFrame) -> tp.MaybeSeriesFrame:
            new_obj = obj.iloc[:, indices]
            # 如果只选择了一个参数且指定了级别名称且为DataFrame且具有多级索引列，则删除该级别
            if not is_multiple:
                if self.level_name is not None:
                    if checks.is_frame(new_obj):
                        if isinstance(new_obj.columns, pd.MultiIndex):
                            new_obj.columns = index_fns.drop_levels(new_obj.columns, self.level_name)
                            
            return new_obj

        # 执行索引操作并返回新的类实例
        return self.indexing_func(pd_indexing_func, **self.indexing_kwargs)


def indexing_on_mapper(mapper: tp.Series, ref_obj: tp.SeriesFrame,
                       pd_indexing_func: tp.Callable) -> tp.Optional[tp.Series]:
    checks.assert_instance_of(mapper, pd.Series)
    checks.assert_instance_of(ref_obj, (pd.Series, pd.DataFrame))

    df_range_mapper = reshape_fns.broadcast_to(np.arange(len(mapper.index)), ref_obj)
    
    loced_range_mapper = pd_indexing_func(df_range_mapper)
    
    new_mapper = mapper.iloc[loced_range_mapper.values[0]]
    
    if checks.is_frame(loced_range_mapper):
        return pd.Series(new_mapper.values, index=loced_range_mapper.columns, name=mapper.name)
    elif checks.is_series(loced_range_mapper):
        return pd.Series([new_mapper], index=[loced_range_mapper.name], name=mapper.name)
    
    return None


def build_param_indexer(param_names: tp.Sequence[str], class_name: str = 'ParamIndexer',
                        module_name: tp.Optional[str] = None) -> tp.Type[IndexingBase]:
    
    class ParamIndexer(IndexingBase):
        def __init__(self, param_mappers: tp.Sequence[tp.Series],
                     level_names: tp.Optional[tp.LevelSequence] = None, **kwargs) -> None:
            """初始化参数索引器
            
            Args:
                param_mappers: 参数映射器序列，每个元素对应一个参数的映射关系
                              长度必须与param_names相同
                level_names: 级别名称序列，用于指定每个参数对应的索引级别名称
                            如果提供，长度必须与param_names相同
                **kwargs: 传递给ParamLoc的额外关键字参数
            """
            checks.assert_len_equal(param_names, param_mappers)
            # 为每个参数创建对应的ParamLoc索引器
            for i, param_name in enumerate(param_names):
                level_name = level_names[i] if level_names is not None else None
                _param_loc = ParamLoc(param_mappers[i], self.indexing_func, level_name=level_name, **kwargs)
                # 将ParamLoc实例设置为私有属性，命名格式为_{param_name}_loc
                setattr(self, f'_{param_name}_loc', _param_loc)

    for i, param_name in enumerate(param_names):
        
        def param_loc(self, _param_name=param_name) -> ParamLoc:
            return getattr(self, f'_{_param_name}_loc')

        # 为属性方法设置文档字符串
        param_loc.__doc__ = f"""Access a group of columns by parameter `{param_name}` using `pd.Series.loc`.
        
        Forwards this operation to each Series/DataFrame and returns a new class instance.
        """

        # 将方法设置为类的属性（使用property装饰器）
        setattr(ParamIndexer, param_name + '_loc', property(param_loc))

    # 设置动态生成类的元信息
    ParamIndexer.__name__ = class_name  
    ParamIndexer.__qualname__ = class_name  
    if module_name is not None:
        ParamIndexer.__module__ = module_name  

    return ParamIndexer
