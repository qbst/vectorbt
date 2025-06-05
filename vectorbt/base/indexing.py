# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT BASE MODULE: INDEXING SYSTEM
================================================================================

文件设计逻辑与作用概述：
本文件是vectorbt库中索引系统的核心实现，为量化交易分析中复杂的数据结构提供了类似pandas的
索引功能。在vectorbt的架构中，用户经常需要处理包含多个pandas对象（DataFrame、Series）
的复杂类，这些对象可能代表不同的交易策略、时间序列数据、投资组合指标等。

核心设计理念：
1. **统一索引接口**：为所有持有pandas对象的自定义类提供一致的索引操作方式，包括iloc、loc、
   xs等标准pandas索引方法，确保用户可以像操作单个DataFrame一样操作复杂的组合对象。

2. **转发机制**：通过索引转发机制，将对复杂类的索引操作转换为对其内部所有pandas对象的
   相同索引操作，然后重新构造新的类实例，实现了数据结构的不可变性和操作的一致性。

3. **参数化索引**：提供了强大的参数化索引功能（ParamLoc），允许用户通过参数值而非列标签
   来查询数据，这在量化分析中特别有用，比如按策略参数、时间窗口参数等进行数据筛选。

4. **动态类生成**：通过build_param_indexer工厂函数，可以动态生成具有特定参数索引能力的
   类，为不同类型的量化分析场景提供定制化的索引解决方案。

主要应用场景：
- **策略回测结果分析**：对包含多个策略、多个时间周期的回测结果进行统一索引操作
- **投资组合管理**：对包含多个资产、多个指标的投资组合数据进行灵活查询和筛选  
- **风险分析**：对包含多维度风险指标的复杂数据结构进行参数化查询
- **因子分析**：按因子类型、计算参数等对多层次因子数据进行索引操作

技术特色：
- 完全兼容pandas索引语法，无需学习新的API
- 支持链式索引操作，保持pandas的使用习惯
- 提供参数映射功能，支持更灵活的数据查询方式
- 通过抽象基类设计，便于扩展和定制
"""

import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import pandas as pd  # 导入Pandas库，提供数据结构和数据分析工具

from vectorbt import _typing as tp  # 导入vectorbt类型定义模块，提供类型注解支持
from vectorbt.base import index_fns, reshape_fns  # 导入索引函数和重构函数模块
from vectorbt.utils import checks  # 导入检查工具模块，提供数据验证功能


class IndexingError(Exception):
    """索引操作异常类
    
    当索引操作过程中发生错误时抛出的异常。
    继承自Python标准异常类Exception，用于标识vectorbt索引系统中的特定错误。
    
    使用场景：
    - 索引参数不合法时
    - 索引操作失败时  
    - 索引转发过程中出现错误时
    """
    pass


# 定义索引基类的类型变量，用于泛型类型注解
# 这允许IndexingBase的子类方法返回正确的类型
IndexingBaseT = tp.TypeVar("IndexingBaseT", bound="IndexingBase")


class IndexingBase:
    """索引操作基类
    
    这是vectorbt索引系统的核心抽象基类，为所有需要支持pandas风格索引操作的类
    提供了基础接口。该类通过indexing_func方法定义了索引操作的统一接口。
    
    设计模式：
    - 采用模板方法模式，定义了索引操作的统一接口
    - 子类需要实现indexing_func方法来提供具体的索引逻辑
    - 支持方法链式调用，保持pandas的使用习惯
    
    继承指南：
    子类需要重写indexing_func方法，该方法应该：
    1. 接收一个pandas索引函数作为参数
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
    """位置索引基类
    
    这是实现基于位置的索引操作的基础类，为iloc和loc等索引器提供了统一的基础设施。
    该类封装了索引函数和相关参数，为具体的索引操作提供了基础框架。
    
    核心功能：
    - 存储索引函数引用，支持延迟执行
    - 管理索引操作的关键字参数
    - 提供__getitem__接口的抽象定义
    
    设计思想：
    采用策略模式，将索引函数作为策略对象存储，在实际需要时才执行索引操作。
    这样的设计使得索引操作可以被延迟执行，提高了灵活性。
    """

    def __init__(self, indexing_func: tp.Callable, **kwargs) -> None:
        """初始化位置索引基类
        
        Args:
            indexing_func: 索引函数，通常是某个IndexingBase实例的indexing_func方法
            **kwargs: 传递给索引函数的关键字参数
        """
        self._indexing_func = indexing_func  # 存储索引函数引用
        self._indexing_kwargs = kwargs  # 存储索引操作的关键字参数

    @property
    def indexing_func(self) -> tp.Callable:
        """获取索引函数
        
        Returns:
            tp.Callable: 存储的索引函数
        """
        return self._indexing_func

    @property
    def indexing_kwargs(self) -> dict:
        """获取索引关键字参数
        
        Returns:
            dict: 传递给索引函数的关键字参数字典
        """
        return self._indexing_kwargs

    def __getitem__(self, key: tp.Any) -> tp.Any:
        """索引访问接口
        
        这是一个抽象方法，子类必须实现具体的索引逻辑。
        
        Args:
            key: 索引键，可以是切片、列表、标量等任意类型
            
        Returns:
            tp.Any: 索引操作的结果
            
        Raises:
            NotImplementedError: 基类中未实现，子类必须重写
        """
        raise NotImplementedError


class iLoc(LocBase):
    """整数位置索引器
    
    实现类似pandas.DataFrame.iloc的整数位置索引功能。
    该类将iloc索引操作转发到每个Series/DataFrame对象上，并返回新的类实例。
    
    功能特点：
    - 支持所有pandas.iloc支持的索引方式（切片、列表、标量等）
    - 保持与pandas.iloc完全一致的行为和语法
    - 通过索引转发机制实现对复杂对象的统一索引
    
    使用场景：
    当需要按整数位置对包含多个pandas对象的复杂类进行索引时使用，
    比如选择前N行数据、按位置范围切片等。
    """

    def __getitem__(self, key: tp.Any) -> tp.Any:
        """执行整数位置索引
        
        将iloc索引操作应用到所有相关的pandas对象上。
        
        Args:
            key: 索引键，支持pandas.iloc支持的所有格式：
                - 整数：如 5（选择第5行）
                - 切片：如 1:5（选择第1到4行）  
                - 列表：如 [1,3,5]（选择指定位置的行）
                - 元组：如 (1:5, ['col1','col2'])（行列同时索引）
                
        Returns:
            tp.Any: 索引操作后的新对象实例
            
        示例:
            ```python
            # 选择前3行
            result = obj.iloc[:3]
            
            # 选择特定位置的行
            result = obj.iloc[[0, 2, 4]]
            
            # 同时索引行和列
            result = obj.iloc[1:5, 2:7]
            ```
        """
        return self.indexing_func(lambda x: x.iloc.__getitem__(key), **self.indexing_kwargs)


class Loc(LocBase):
    """标签位置索引器
    
    实现类似pandas.DataFrame.loc的标签位置索引功能。
    该类将loc索引操作转发到每个Series/DataFrame对象上，并返回新的类实例。
    
    功能特点：
    - 支持所有pandas.loc支持的索引方式（标签、切片、布尔索引等）
    - 保持与pandas.loc完全一致的行为和语法
    - 支持多级索引的复杂查询操作
    - 支持条件筛选和布尔索引
    
    使用场景：
    当需要按标签对包含多个pandas对象的复杂类进行索引时使用，
    比如按时间范围筛选、按股票代码查询、按条件过滤等。
    """

    def __getitem__(self, key: tp.Any) -> tp.Any:
        """执行标签位置索引
        
        将loc索引操作应用到所有相关的pandas对象上。
        
        Args:
            key: 索引键，支持pandas.loc支持的所有格式：
                - 标签：如 'AAPL'（选择指定标签的行）
                - 切片：如 '2020-01-01':'2020-12-31'（日期范围切片）
                - 列表：如 ['AAPL','MSFT']（选择多个标签）
                - 布尔数组：如条件筛选
                - 元组：如 ('2020-01-01', 'AAPL')（多级索引）
                
        Returns:
            tp.Any: 索引操作后的新对象实例
            
        示例:
            ```python
            # 按日期范围选择
            result = obj.loc['2020-01-01':'2020-12-31']
            
            # 按标签选择
            result = obj.loc[['AAPL', 'MSFT']]
            
            # 条件筛选
            result = obj.loc[obj.returns > 0.05]
            ```
        """
        return self.indexing_func(lambda x: x.loc.__getitem__(key), **self.indexing_kwargs)


# 定义PandasIndexer的类型变量，用于泛型类型注解
PandasIndexerT = tp.TypeVar("PandasIndexerT", bound="PandasIndexer")


class PandasIndexer(IndexingBase):
    """Pandas风格索引器
    
    这是vectorbt索引系统的核心实现类，提供了完整的pandas风格索引功能。
    该类实现了iloc、loc、xs和__getitem__等索引方法，使得自定义类可以像pandas对象一样进行索引操作。
    
    核心特性：
    1. **完整的pandas索引支持**：支持iloc、loc、xs等所有主要的pandas索引方法
    2. **索引转发机制**：自动将索引操作转发到所有内部pandas对象
    3. **类型安全**：通过泛型确保索引操作返回正确的类型
    4. **灵活的参数传递**：支持将额外参数传递给索引操作
    
    使用方式：
    该类通常作为基类被继承，子类需要实现indexing_func方法来定义具体的索引逻辑。
    
    设计模式：
    - 装饰器模式：为现有类添加索引功能
    - 代理模式：代理对内部pandas对象的索引操作
    - 模板方法模式：定义索引操作的统一模板
    
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
        recent_data = portfolio.iloc[-30:]  # 最近30天的数据
        apple_data = portfolio.loc[:, 'AAPL']  # 苹果股票的数据
        ```
    """

    def __init__(self, **kwargs) -> None:
        """初始化Pandas索引器
        
        Args:
            **kwargs: 传递给索引操作的关键字参数，这些参数会被保存并
                     在每次索引操作时传递给indexing_func方法
        """
        # 创建iloc索引器，绑定到当前实例的indexing_func方法
        self._iloc = iLoc(self.indexing_func, **kwargs)
        # 创建loc索引器，绑定到当前实例的indexing_func方法  
        self._loc = Loc(self.indexing_func, **kwargs)
        # 保存索引关键字参数，供其他索引方法使用
        self._indexing_kwargs = kwargs

    @property
    def indexing_kwargs(self) -> dict:
        """获取索引关键字参数
        
        Returns:
            dict: 传递给索引操作的关键字参数字典
        """
        return self._indexing_kwargs

    @property
    def iloc(self) -> iLoc:
        """整数位置索引器属性
        
        提供基于整数位置的索引功能，完全兼容pandas.DataFrame.iloc的API。
        
        Returns:
            iLoc: 整数位置索引器实例
            
        使用示例:
            ```python
            # 选择前10行
            obj.iloc[:10]
            
            # 选择特定行和列
            obj.iloc[5:15, 2:8]
            
            # 选择不连续的行
            obj.iloc[[1, 5, 10, 15]]
            ```
        """
        return self._iloc

    # 复制iLoc类的文档字符串到iloc属性
    iloc.__doc__ = iLoc.__doc__

    @property
    def loc(self) -> Loc:
        """标签位置索引器属性
        
        提供基于标签的索引功能，完全兼容pandas.DataFrame.loc的API。
        
        Returns:
            Loc: 标签位置索引器实例
            
        使用示例:
            ```python
            # 按日期范围选择
            obj.loc['2020-01-01':'2020-12-31']
            
            # 按标签选择列
            obj.loc[:, ['returns', 'volume']]
            
            # 条件筛选
            obj.loc[obj.price > 100]
            ```
        """
        return self._loc

    # 复制Loc类的文档字符串到loc属性
    loc.__doc__ = Loc.__doc__

    def xs(self: PandasIndexerT, *args, **kwargs) -> PandasIndexerT:
        """横截面索引操作
        
        转发pandas.DataFrame.xs操作到每个Series/DataFrame，并返回新的类实例。
        xs方法主要用于从多级索引的DataFrame中提取横截面数据。
        
        Args:
            *args: 传递给pandas.xs的位置参数
                - key: 要选择的标签或标签元组
                - axis: 轴向，0表示索引，1表示列
                - level: 要选择的索引级别
                - drop_level: 是否删除选择的级别
            **kwargs: 传递给pandas.xs的关键字参数
            
        Returns:
            PandasIndexerT: 横截面索引后的新实例
            
        使用场景:
            - 从多级索引中选择特定级别的数据
            - 获取特定时间点或标签的横截面数据
            - 在多维数据分析中提取切片
            
        示例:
            ```python
            # 从多级索引中选择特定日期的数据
            daily_data = obj.xs('2020-01-01', level='date')
            
            # 选择特定资产的所有数据
            asset_data = obj.xs('AAPL', level='symbol', axis=1)
            ```
        """
        return self.indexing_func(lambda x: x.xs(*args, **kwargs), **self.indexing_kwargs)

    def __getitem__(self: PandasIndexerT, key: tp.Any) -> PandasIndexerT:
        """索引访问操作
        
        实现Python的索引语法支持，如obj[key]。
        该方法将__getitem__操作转发到每个内部pandas对象上。
        
        Args:
            key: 索引键，可以是：
                - 字符串：列名或索引标签
                - 列表：多个列名或标签
                - 切片：范围选择
                - 布尔数组：条件筛选
                - 元组：多维索引
                
        Returns:
            PandasIndexerT: 索引操作后的新实例
            
        注意:
            这个方法提供了最灵活的索引方式，但具体行为取决于pandas对象的类型和索引键的类型。
            
        示例:
            ```python
            # 选择单个列
            returns = obj['returns']
            
            # 选择多个列
            subset = obj[['returns', 'volume', 'price']]
            
            # 条件筛选
            positive_returns = obj[obj['returns'] > 0]
            
            # 切片选择
            recent = obj[-30:]  # 最近30个观测值
            ```
        """
        return self.indexing_func(lambda x: x.__getitem__(key), **self.indexing_kwargs)


class ParamLoc(LocBase):
    """参数位置索引器
    
    这是vectorbt索引系统的高级功能，允许通过参数值而非直接的列标签来访问数据组。
    在量化分析中，经常需要按策略参数、计算窗口、风险阈值等参数来筛选数据，
    ParamLoc正是为了解决这类需求而设计的。
    
    核心功能：
    1. **参数映射**：通过mapper Series建立列标签与参数值的映射关系
    2. **类型转换**：自动处理对象类型参数的字符串转换
    3. **级别管理**：支持多级索引中的级别删除操作
    4. **灵活查询**：支持单值、多值、切片等多种查询方式
    
    技术特点：
    - 使用pandas的loc机制实现高效查询
    - 支持对象类型参数的自动字符串化处理
    - 与多级索引无缝集成
    - 提供直观的参数化查询接口
    
    应用场景：
    - 按策略参数筛选回测结果
    - 按计算窗口选择技术指标
    - 按风险水平过滤投资组合
    - 按因子类型查询因子数据
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
        # 验证mapper必须是pandas Series类型
        checks.assert_instance_of(mapper, pd.Series)

        # 处理对象类型的参数值
        if mapper.dtype == 'O':  # 'O'表示object类型
            # 如果参数是对象类型，必须先转换为字符串类型
            # 原始mapper不会被修改，这里创建副本进行转换
            mapper = mapper.astype(str)
            
        self._mapper = mapper  # 存储参数映射关系
        self._level_name = level_name  # 存储级别名称

        # 调用父类初始化方法
        LocBase.__init__(self, indexing_func, **kwargs)

    @property
    def mapper(self) -> tp.Series:
        """获取参数映射器
        
        Returns:
            tp.Series: 存储的参数映射Series，索引为列标签，值为参数值
        """
        return self._mapper

    @property
    def level_name(self) -> tp.Level:
        """获取级别名称
        
        Returns:
            tp.Level: 用于级别删除操作的级别名称
        """
        return self._level_name

    def get_indices(self, key: tp.Any) -> tp.Array1d:
        """根据参数键获取对应的列索引数组
        
        这是ParamLoc的核心方法，将参数值转换为实际的列索引位置。
        
        Args:
            key: 参数键，可以是：
                - 单个参数值：如 'param1'
                - 参数值列表：如 ['param1', 'param2']  
                - 切片对象：如 slice('param1', 'param3')
                - NumPy数组：参数值数组
                
        Returns:
            tp.Array1d: 对应的列索引位置数组
            
        处理逻辑：
        1. 如果mapper是对象类型，则将key也转换为字符串
        2. 使用pandas的loc机制在mapper的值上进行索引
        3. 返回匹配的列位置索引
        
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
                # 处理列表和数组，将每个元素转换为字符串
                key = list(map(str, key))
            else:
                # 处理其他类型（元组、对象等），直接转换为字符串
                key = str(key)
                
        # 创建一个从参数值到列位置的映射
        # mapper.values是参数值，mapper.index是原始列标签
        # 这里创建一个Series，索引是参数值，值是列位置（0到len-1）
        mapper = pd.Series(np.arange(len(self.mapper.index)), index=self.mapper.values)
        
        # 使用pandas的loc机制根据参数值获取列位置
        indices = mapper.loc.__getitem__(key)
        
        # 如果结果是Series，提取其values作为NumPy数组
        if isinstance(indices, pd.Series):
            indices = indices.values
            
        return indices

    def __getitem__(self, key: tp.Any) -> tp.Any:
        """参数索引访问操作
        
        实现通过参数值进行数据访问的核心逻辑。
        
        Args:
            key: 参数键，支持多种格式的参数值查询
            
        Returns:
            tp.Any: 索引操作后的结果对象
            
        处理流程：
        1. 通过get_indices方法将参数值转换为列索引
        2. 判断是否为多选操作（影响级别删除逻辑）
        3. 定义pandas索引函数，使用iloc进行实际索引
        4. 根据需要删除多级索引中的指定级别
        5. 通过indexing_func执行索引操作并返回结果
        
        级别删除逻辑：
        当选择单个参数且指定了level_name时，会自动删除该级别以保持数据结构的简洁性。
        这在处理多级索引的列时特别有用。
        """
        # 获取参数对应的列索引位置
        indices = self.get_indices(key)
        
        # 判断是否为多选操作，影响后续的级别删除逻辑
        is_multiple = isinstance(key, (slice, list, np.ndarray))

        def pd_indexing_func(obj: tp.SeriesFrame) -> tp.MaybeSeriesFrame:
            """pandas索引函数
            
            这是传递给indexing_func的实际索引操作函数。
            
            Args:
                obj: 要进行索引的pandas对象（Series或DataFrame）
                
            Returns:
                tp.MaybeSeriesFrame: 索引后的pandas对象
            """
            # 使用iloc根据列位置进行索引
            new_obj = obj.iloc[:, indices]
            
            # 如果只选择了一个参数且指定了级别名称，则删除该级别
            if not is_multiple:
                if self.level_name is not None:
                    # 检查是否为DataFrame且具有多级索引列
                    if checks.is_frame(new_obj):
                        if isinstance(new_obj.columns, pd.MultiIndex):
                            # 删除指定的级别以简化索引结构
                            new_obj.columns = index_fns.drop_levels(new_obj.columns, self.level_name)
                            
            return new_obj

        # 执行索引操作并返回新的类实例
        return self.indexing_func(pd_indexing_func, **self.indexing_kwargs)


def indexing_on_mapper(mapper: tp.Series, ref_obj: tp.SeriesFrame,
                       pd_indexing_func: tp.Callable) -> tp.Optional[tp.Series]:
    """在映射器上执行索引操作
    
    这是一个工具函数，用于将mapper Series广播到参考对象的维度，然后执行pandas索引操作。
    该函数主要用于参数索引系统中，确保参数映射关系在索引操作后得到正确维护。
    
    工作原理：
    1. 将mapper广播到与ref_obj相同的形状
    2. 对广播后的数据执行索引操作
    3. 根据索引结果重建mapper的映射关系
    4. 返回与索引结果对应的新mapper
    
    Args:
        mapper: 参数映射Series，建立列与参数值的对应关系
        ref_obj: 参考pandas对象，用于确定广播的目标形状
        pd_indexing_func: pandas索引函数，要应用的索引操作
        
    Returns:
        tp.Optional[tp.Series]: 索引操作后的新mapper，如果操作失败则返回None
        
    Raises:
        AssertionError: 当mapper不是Series或ref_obj不是Series/DataFrame时抛出
        
    使用场景：
    - 在参数索引操作中维护映射关系
    - 在数据重构过程中保持参数一致性  
    - 在批量操作中同步更新映射器
    
    示例:
        ```python
        # 原始映射：列A->param1, 列B->param2, 列C->param1
        mapper = pd.Series(['param1', 'param2', 'param1'], 
                          index=['A', 'B', 'C'], name='strategy')
        df = pd.DataFrame(data, columns=['A', 'B', 'C'])
        
        # 选择前两列的索引操作
        indexing_func = lambda x: x.iloc[:, :2]
        new_mapper = indexing_on_mapper(mapper, df, indexing_func)
        # 结果：新mapper对应选择后的列A和B
        ```
    """
    # 验证输入参数类型
    checks.assert_instance_of(mapper, pd.Series)
    checks.assert_instance_of(ref_obj, (pd.Series, pd.DataFrame))

    # 创建范围映射器：将mapper的索引位置（0,1,2,...）广播到ref_obj的形状
    # 这样可以跟踪每个位置在索引操作后的去向
    df_range_mapper = reshape_fns.broadcast_to(np.arange(len(mapper.index)), ref_obj)
    
    # 对范围映射器执行相同的索引操作
    loced_range_mapper = pd_indexing_func(df_range_mapper)
    
    # 根据索引操作的结果重建mapper
    # loced_range_mapper告诉我们哪些原始位置被保留了
    new_mapper = mapper.iloc[loced_range_mapper.values[0]]
    
    # 根据索引结果的类型返回相应的新mapper
    if checks.is_frame(loced_range_mapper):
        # 如果结果是DataFrame，返回以列为索引的Series
        return pd.Series(new_mapper.values, index=loced_range_mapper.columns, name=mapper.name)
    elif checks.is_series(loced_range_mapper):
        # 如果结果是Series，返回单元素Series
        return pd.Series([new_mapper], index=[loced_range_mapper.name], name=mapper.name)
    
    # 如果结果不是DataFrame或Series，返回None
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
