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
    """整数位置索引器
    
    实现类似pandas.DataFrame.iloc的整数位置索引功能。
    该类将iloc索引操作转发到每个Series/DataFrame对象上，并返回新的类实例。
    """

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
    """在映射器上执行索引操作并同步维护参数映射关系
    
    这是一个核心工具函数，用于在对pandas对象执行索引操作时，自动同步更新相应的参数映射器。
    该函数确保在数据筛选、重排、切片等操作后，参数映射关系始终保持与数据结构的一致性。
    
    核心机制:
    1. 通过广播技术将mapper的索引位置追踪到ref_obj的每个位置
    2. 对追踪数组执行与原数据相同的索引操作
    3. 根据操作结果重建新的参数映射器
    4. 返回与索引后数据结构匹配的新mapper
    
    Args:
        mapper (tp.Series): 参数映射Series，建立列标签与参数值的对应关系
            - 索引: 原始数据的列名或行名
            - 值: 对应的参数值（如策略类型、窗口期、风险等级等）
            - 名称: 映射器的名称，用于标识参数类型
            示例: pd.Series(['momentum', 'momentum', 'mean_reversion'], 
                           index=['col1', 'col2', 'col3'], name='strategy_type')
                           
        ref_obj (tp.SeriesFrame): 参考pandas对象，用于确定广播的目标形状
            - 可以是DataFrame或Series
            - 作为mapper广播的模板，确定结果的维度结构
            - 通常是即将被索引的主数据对象
            
        pd_indexing_func (tp.Callable): pandas索引函数，要应用的具体索引操作
            - 接受pandas对象作为参数的函数
            - 返回索引操作后的新pandas对象
            - 示例: lambda x: x.iloc[:, :3] 或 lambda x: x[['col1', 'col2']]
            
    Returns:
        tp.Optional[tp.Series]: 索引操作后的新mapper，如果操作失败则返回None
            - 新mapper的索引对应索引后数据的列名或行名
            - 新mapper的值保持原有的参数映射关系
            - 维护原mapper的名称属性
            
    Raises:
        AssertionError: 当输入参数类型不符合要求时抛出
            - mapper不是pandas.Series类型
            - ref_obj不是pandas.Series或pandas.DataFrame类型
            
    使用场景:
        - ParamLoc参数索引操作中的映射器同步更新
        - 数据重构过程中保持参数一致性
        - 批量操作中自动维护映射关系
        - 动态数据筛选时的参数追踪
        
    实际应用示例:
        # 技术指标窗口期映射的维护
        import pandas as pd
        import numpy as np
        
        # 原始数据: 不同窗口期的移动平均线
        data = pd.DataFrame({
            'MA_5': [100, 101, 102],
            'MA_10': [99, 100, 101], 
            'MA_20': [98, 99, 100]
        })
        
        # 窗口期映射器
        window_mapper = pd.Series([5, 10, 20], 
                                index=['MA_5', 'MA_10', 'MA_20'], 
                                name='window')
        
        # 定义索引操作: 选择前两列
        def select_first_two(obj):
            return obj.iloc[:, :2]
        
        # 同步更新映射器
        new_data = select_first_two(data)  
        # 结果: DataFrame with columns ['MA_5', 'MA_10']
        
        new_mapper = indexing_on_mapper(window_mapper, data, select_first_two)
        # 结果: Series([5, 10], index=['MA_5', 'MA_10'], name='window')
        
        # 验证映射关系保持一致
        assert new_mapper['MA_5'] == 5
        assert new_mapper['MA_10'] == 10
        assert 'MA_20' not in new_mapper.index  # 被正确移除
    """
    # 输入参数类型验证: 确保mapper是pandas Series类型
    # 这是函数正常工作的基础前提，Series提供了索引和值的映射关系
    checks.assert_instance_of(mapper, pd.Series)
    
    # 输入参数类型验证: 确保ref_obj是pandas Series或DataFrame类型  
    # ref_obj作为广播模板，必须具有明确的形状和索引结构
    checks.assert_instance_of(ref_obj, (pd.Series, pd.DataFrame))

    # 创建位置追踪数组: 生成mapper索引位置的整数序列(0,1,2,...)
    # 然后广播到ref_obj的形状，建立位置映射关系
    # 这样每个ref_obj的位置都对应一个mapper中的索引位置
    df_range_mapper = reshape_fns.broadcast_to(np.arange(len(mapper.index)), ref_obj)
    
    # 对位置追踪数组执行相同的索引操作
    # 这一步关键在于模拟原数据的索引过程，记录哪些位置被保留
    # loced_range_mapper记录了索引操作后保留的原始位置信息
    loced_range_mapper = pd_indexing_func(df_range_mapper)
    
    # 根据保留的位置信息重建新的映射器
    # loced_range_mapper.values[0]包含了被保留的原始索引位置
    # 使用iloc根据这些位置从原mapper中提取对应的映射关系
    new_mapper = mapper.iloc[loced_range_mapper.values[0]]
    
    # 根据索引操作结果的类型构建相应的新映射器
    # 情况1: 结果是DataFrame - 返回以列名为索引的Series
    if checks.is_frame(loced_range_mapper):
        # 创建新Series: 值来自重建的mapper，索引来自操作后的列名，保持原名称
        return pd.Series(new_mapper.values, index=loced_range_mapper.columns, name=mapper.name)
    # 情况2: 结果是Series - 返回单元素Series  
    elif checks.is_series(loced_range_mapper):
        # 创建新Series: 值为单个映射值的列表，索引为Series名称，保持原名称
        return pd.Series([new_mapper], index=[loced_range_mapper.name], name=mapper.name)
    
    # 情况3: 结果既不是DataFrame也不是Series - 返回None表示操作失败
    # 这种情况通常表示索引操作返回了标量值或其他不支持的类型
    return None


def build_param_indexer(param_names: tp.Sequence[str], class_name: str = 'ParamIndexer',
                        module_name: tp.Optional[str] = None) -> tp.Type[IndexingBase]:
    """参数索引器工厂函数
    
    这是vectorbt索引系统的高级功能，用于动态生成具有参数索引能力的类。
    该工厂函数为量化分析中的复杂参数化查询提供了强大而灵活的解决方案。
    
    设计理念：
    在量化分析中，经常需要处理具有多个参数维度的数据，比如：
    - 不同时间窗口的移动平均线
    - 不同参数组合的策略回测结果  
    - 不同风险水平的投资组合指标
    
    传统的列标签查询方式不够直观，特别是在pandas多级索引的特定级别查询时。
    该工厂函数通过动态生成类，为每个参数提供专门的索引器，使查询更加直观和便捷。
    
    Args:
        param_names: 参数名称序列，每个名称将生成对应的参数索引器属性
                    例如：['window', 'strategy', 'risk_level']
        class_name: 生成的类名称，默认为'ParamIndexer'
        module_name: 类所属的模块名称，用于设置__module__属性
        
    Returns:
        tp.Type[IndexingBase]: 动态生成的参数索引器类
        
    生成的类特性：
    1. 继承自IndexingBase，具备完整的索引基础设施
    2. 为每个参数名生成对应的*_loc属性（如window_loc、strategy_loc）
    3. 每个*_loc属性都是ParamLoc实例，支持参数化查询
    4. 支持级别名称管理，可自动删除冗余的索引级别
    
    使用模式：
        ```python
        # 1. 创建参数索引器类
        MyIndexer = build_param_indexer(['window', 'strategy'])
        
        # 2. 继承并实现具体的索引逻辑
        class BacktestResults(MyIndexer):
            def __init__(self, returns_df, window_mapper, strategy_mapper):
                self.returns = returns_df
                super().__init__([window_mapper, strategy_mapper])
                
            def indexing_func(self, pd_indexing_func):
                return BacktestResults(
                    pd_indexing_func(self.returns),
                    indexing_on_mapper(self._window_mapper, self.returns, pd_indexing_func),
                    indexing_on_mapper(self._strategy_mapper, self.returns, pd_indexing_func)
                )
        
        # 3. 使用参数化查询
        results = BacktestResults(data, window_map, strategy_map)
        short_term = results.window_loc[10]  # 选择10日窗口的数据
        momentum = results.strategy_loc['momentum']  # 选择动量策略的数据
        combined = results.window_loc[20].strategy_loc['mean_reversion']  # 链式查询
        ```
    
    技术实现：
    - 使用type()动态创建类，确保类型安全
    - 通过property装饰器动态生成属性
    - 利用闭包保持参数名称的正确绑定
    - 支持方法链式调用，保持pandas风格的使用体验
    
    应用场景：
    - 策略回测结果的参数化分析
    - 技术指标的参数敏感性研究
    - 投资组合优化的参数空间探索
    - 因子分析的多维度查询
    """

    class ParamIndexer(IndexingBase):
        """动态生成的参数索引器基类
        
        这个类在运行时被动态创建，为每个指定的参数提供专门的索引器。
        该类的实例需要提供参数映射器序列，用于建立数据列与参数值的对应关系。
        """

        def __init__(self, param_mappers: tp.Sequence[tp.Series],
                     level_names: tp.Optional[tp.LevelSequence] = None, **kwargs) -> None:
            """初始化参数索引器
            
            Args:
                param_mappers: 参数映射器序列，每个元素对应一个参数的映射关系
                              长度必须与param_names相同
                level_names: 级别名称序列，用于指定每个参数对应的索引级别名称
                            如果提供，长度必须与param_names相同
                **kwargs: 传递给ParamLoc的额外关键字参数
                
            Raises:
                AssertionError: 当param_mappers长度与param_names不匹配时抛出
            """
            # 验证参数映射器数量与参数名称数量匹配
            checks.assert_len_equal(param_names, param_mappers)

            # 为每个参数创建对应的ParamLoc索引器
            for i, param_name in enumerate(param_names):
                # 获取对应的级别名称（如果提供）
                level_name = level_names[i] if level_names is not None else None
                
                # 创建ParamLoc实例，绑定到当前实例的indexing_func方法
                _param_loc = ParamLoc(param_mappers[i], self.indexing_func, level_name=level_name, **kwargs)
                
                # 将ParamLoc实例设置为私有属性，命名格式为_{param_name}_loc
                setattr(self, f'_{param_name}_loc', _param_loc)

    # 为每个参数名动态创建对应的属性
    for i, param_name in enumerate(param_names):
        
        def param_loc(self, _param_name=param_name) -> ParamLoc:
            """参数位置索引器属性
            
            这是动态生成的属性方法，为每个参数提供专门的索引器。
            
            Returns:
                ParamLoc: 对应参数的位置索引器
                
            注意:
                这个方法使用闭包捕获参数名称，确保每个属性返回正确的索引器。
            """
            return getattr(self, f'_{_param_name}_loc')

        # 为属性方法设置文档字符串
        param_loc.__doc__ = f"""通过参数 `{param_name}` 访问数据组的位置索引器
        
        该属性提供了基于参数值的数据查询功能，支持单值选择、多值选择、范围切片等操作。
        索引操作会被转发到每个Series/DataFrame，并返回新的类实例。
        
        返回:
            ParamLoc: 参数 `{param_name}` 的位置索引器
            
        使用示例:
            # 选择单个参数值
            result = obj.{param_name}_loc['param_value']
            
            # 选择多个参数值  
            result = obj.{param_name}_loc[['value1', 'value2']]
            
            # 范围选择
            result = obj.{param_name}_loc['start':'end']
            
            # 链式操作
            result = obj.{param_name}_loc['value'].other_param_loc['other_value']
        """

        # 将方法设置为类的属性（使用property装饰器）
        setattr(ParamIndexer, param_name + '_loc', property(param_loc))

    # 设置动态生成类的元信息
    ParamIndexer.__name__ = class_name  # 设置类名
    ParamIndexer.__qualname__ = class_name  # 设置限定名称
    if module_name is not None:
        ParamIndexer.__module__ = module_name  # 设置模块名称

    return ParamIndexer
