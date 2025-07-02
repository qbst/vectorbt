# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
数据基础类模块 - vectorbt量化分析框架的核心数据管理组件。提供统一的数据下载、存储、更新和管理接口。
"""

import warnings
import numpy as np
import pandas as pd
from vectorbt import _typing as tp
from vectorbt.base.array_wrapper import ArrayWrapper, Wrapping
from vectorbt.generic import plotting
from vectorbt.generic.plots_builder import PlotsBuilderMixin
from vectorbt.generic.stats_builder import StatsBuilderMixin
from vectorbt.utils import checks
from vectorbt.utils.config import merge_dicts, Config
from vectorbt.utils.datetime_ import is_tz_aware, to_timezone
from vectorbt.utils.decorators import cached_method

__pdoc__ = {}


class symbol_dict(dict):
    """
    符号字典类 - 专门用于存储以金融符号为键的配置参数
    
    这是一个特殊的字典类，继承自内置dict，专门用于在Data类中
    为不同的金融符号提供差异化的配置参数。
    
    主要特点：
    - 以金融符号(如'AAPL', 'GOOGL')为键
    - 允许为每个符号设置不同的下载参数
    - 在数据下载过程中被select_symbol_kwargs方法识别和处理
    
    使用示例：
    ```python
    # 为不同符号设置不同的起始值
    start_values = symbol_dict({
        'AAPL': 1000000,    # 苹果股票起始投资100万
        'GOOGL': 500000     # 谷歌股票起始投资50万
    })
    
    # 为不同符号设置不同的时间范围
    start_dates = symbol_dict({
        'AAPL': '2020-01-01',
        'GOOGL': '2021-01-01'
    })
    
    # 在下载时使用
    data = MyData.download(
        ['AAPL', 'GOOGL'], 
        start_value=start_values,
        start_date=start_dates
    )
    ```
    
    注意事项：
    - 如果某个符号在symbol_dict中不存在，将使用默认值
    - 适用于所有需要符号特定配置的场景
    """
    pass


class MetaData(type(StatsBuilderMixin), type(PlotsBuilderMixin)):
    """
    Data类的元类 - 整合统计构建器和图表构建器的元类功能
    
    这个元类通过多重继承的方式，将StatsBuilderMixin和PlotsBuilderMixin
    的元类功能整合到Data类中，使Data类同时具备统计分析和图表绘制的能力。
    
    技术实现：
    - 继承自StatsBuilderMixin和PlotsBuilderMixin的元类
    - 解决多重继承中的方法解析顺序(MRO)问题
    - 确保所有混入类的元类功能正常工作
    
    功能整合：
    - 统计功能：自动注册统计指标，支持stats()方法
    - 图表功能：自动注册子图配置，支持plots()方法
    - 文档生成：整合两个混入类的文档生成功能
    """
    pass


# Data类的类型变量，用于类型提示中的泛型约束
# 确保方法返回的类型与调用类的类型一致
DataT = tp.TypeVar("DataT", bound="Data")


class Data(Wrapping, StatsBuilderMixin, PlotsBuilderMixin, metaclass=MetaData):
    """
    数据管理核心类 - vectorbt框架的统一数据接口
    
    Data类是vectorbt框架中所有数据操作的核心基类，提供了完整的数据生命周期管理，
    包括下载、存储、更新、对齐和访问等功能。该类通过多重继承整合了数组包装、
    统计分析和图表绘制的功能。
    
    核心特性：
    1. 多数据源支持：抽象化不同数据提供商的接口差异
    2. 自动数据对齐：智能处理不同符号间的时间和列差异
    3. 增量数据更新：支持数据的实时或定期更新
    4. 时区智能处理：完整的时区本地化和转换支持
    5. 统计分析集成：内置丰富的统计指标计算功能
    6. 可视化支持：提供灵活的图表绘制接口
    
    设计模式：
    - 模板方法模式：定义数据处理的标准流程，子类实现具体细节
    - 策略模式：支持不同的缺失数据处理策略
    - 观察者模式：支持数据更新时的事件通知
    
    使用示例：
    ```python
    # 1. 基本使用 - 继承Data类实现自定义数据源
    class MyDataSource(vbt.Data):
        @classmethod
        def download_symbol(cls, symbol, **kwargs):
            # 实现具体的数据下载逻辑
            return pd.Series(...)
    
    # 2. 下载多个符号的数据
    data = MyDataSource.download(['AAPL', 'GOOGL', 'MSFT'])
    
    # 3. 访问数据
    apple_data = data.get('AAPL')  # 获取苹果股票数据
    all_data = data.get()          # 获取所有数据的合并结果
    
    # 4. 数据更新
    updated_data = data.update()   # 增量更新数据
    
    # 5. 统计分析
    stats = data.stats()           # 计算统计指标
    
    # 6. 数据可视化
    fig = data.plots()             # 生成图表
    ```
    
    继承结构：
    - Wrapping: 提供pandas索引操作和数组包装功能
    - StatsBuilderMixin: 提供统计指标计算功能
    - PlotsBuilderMixin: 提供图表构建功能
    """

    def __init__(self,
                 wrapper: ArrayWrapper,
                 data: tp.Data,
                 tz_localize: tp.Optional[tp.TimezoneLike],
                 tz_convert: tp.Optional[tp.TimezoneLike],
                 missing_index: str,
                 missing_columns: str,
                 download_kwargs: dict,
                 **kwargs) -> None:
        """
        初始化Data实例
        
        Args:
            wrapper (ArrayWrapper): 数组包装器，提供统一的数组操作接口
            data (tp.Data): 数据字典，以符号为键，pandas对象为值
            tz_localize (tp.TimezoneLike, optional): 时区本地化参数
            tz_convert (tp.TimezoneLike, optional): 时区转换参数  
            missing_index (str): 索引缺失处理策略 ('nan', 'drop', 'raise')
            missing_columns (str): 列缺失处理策略 ('nan', 'drop', 'raise')
            download_kwargs (dict): 下载时使用的关键字参数
            **kwargs: 传递给父类的额外参数
            
        初始化过程：
        1. 调用Wrapping父类初始化，设置数组包装器
        2. 调用StatsBuilderMixin初始化，启用统计功能
        3. 调用PlotsBuilderMixin初始化，启用图表功能
        4. 验证数据字典的格式和一致性
        5. 存储配置参数供后续使用
        """
        # 初始化Wrapping基类，传递所有必要参数
        Wrapping.__init__(
            self,
            wrapper,                    # 数组包装器实例
            data=data,                  # 原始数据字典
            tz_localize=tz_localize,    # 时区本地化设置
            tz_convert=tz_convert,      # 时区转换设置
            missing_index=missing_index,    # 索引缺失处理策略
            missing_columns=missing_columns, # 列缺失处理策略
            download_kwargs=download_kwargs, # 下载参数
            **kwargs                    # 其他额外参数
        )
        # 初始化统计构建器混入类，启用统计分析功能
        StatsBuilderMixin.__init__(self)
        # 初始化图表构建器混入类，启用图表绘制功能
        PlotsBuilderMixin.__init__(self)

        # 验证数据参数必须是字典类型
        checks.assert_instance_of(data, dict)
        # 验证所有数据项具有相同的元数据（索引、列等）
        for k, v in data.items():
            # 检查每个数据项与第一个数据项的元数据是否一致
            checks.assert_meta_equal(v, data[list(data.keys())[0]])
            
        # 存储实例属性，用于后续的数据操作
        self._data = data                           # 数据字典
        self._tz_localize = tz_localize            # 时区本地化参数
        self._tz_convert = tz_convert              # 时区转换参数  
        self._missing_index = missing_index        # 索引缺失处理策略
        self._missing_columns = missing_columns    # 列缺失处理策略
        self._download_kwargs = download_kwargs    # 下载时的关键字参数

    def indexing_func(self: DataT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> DataT:
        """
        执行pandas索引操作的核心方法
        
        该方法是Wrapping基类索引操作的具体实现，支持对Data实例进行
        pandas风格的索引操作，如.loc、.iloc、切片等。
        
        Args:
            pd_indexing_func (tp.PandasIndexingFunc): pandas索引函数
            **kwargs: 传递给索引函数的额外参数
            
        Returns:
            DataT: 返回新的Data实例，包含索引后的数据
            
        工作流程：
        1. 对数组包装器执行索引操作，获取新的包装器
        2. 对数据字典中的每个pandas对象执行相同的索引操作
        3. 使用新的包装器和数据创建新的Data实例
        
        使用示例：
        ```python
        # 时间范围切片
        recent_data = data.loc['2023-01-01':'2023-12-31']
        
        # 行索引切片
        first_100_rows = data.iloc[:100]
        
        # 条件过滤
        filtered_data = data[data.get() > 100]
        ```
        
        注意事项：
        - 返回新的Data实例，不修改原实例
        - 保持所有配置参数不变
        - 确保索引操作在所有符号数据上保持一致
        """
        # 对数组包装器执行索引操作，获取新的包装器实例
        new_wrapper = pd_indexing_func(self.wrapper)
        # 对数据字典中每个pandas对象执行相同的索引操作
        new_data = {k: pd_indexing_func(v) for k, v in self.data.items()}
        # 创建并返回新的Data实例，保持其他配置不变
        return self.replace(
            wrapper=new_wrapper,    # 使用新的包装器
            data=new_data          # 使用索引后的数据
        )

    @property
    def data(self) -> tp.Data:
        return self._data

    @property
    def symbols(self) -> tp.List[tp.Label]:
        return list(self.data.keys())

    @property
    def tz_localize(self) -> tp.Optional[tp.TimezoneLike]:
        return self._tz_localize

    @property
    def tz_convert(self) -> tp.Optional[tp.TimezoneLike]:
        return self._tz_convert

    @property
    def missing_index(self) -> str:
        return self._missing_index

    @property
    def missing_columns(self) -> str:
        return self._missing_columns

    @property
    def download_kwargs(self) -> dict:
        return self._download_kwargs

    @classmethod
    def align_index(cls, data: tp.Data, missing: str = 'nan') -> tp.Data:
        """
        索引对齐方法 - 统一不同符号数据的时间索引
        
        该方法是数据预处理的核心步骤，确保所有符号的数据具有一致的时间索引，
        这对于后续的数据分析和策略回测至关重要。
        
        Args:
            data (tp.Data): 待对齐的数据字典，键为符号，值为pandas对象
            missing (str): 缺失数据处理策略，默认'nan'
                - 'nan': 将缺失的数据点设置为NaN（推荐用于价格数据）
                - 'drop': 删除缺失的数据点（用于严格的时间对齐）
                - 'raise': 遇到不匹配时抛出异常（用于数据质量检查）
                
        Returns:
            tp.Data: 索引对齐后的数据字典
            
        工作原理：
        1. 如果只有一个符号，直接返回原数据
        2. 遍历所有符号，收集所有唯一的索引值
        3. 根据missing策略决定最终的统一索引
        4. 对所有符号的数据进行重新索引
        
        使用场景：
        ```python
        # 场景1：不同符号有不同的交易日历
        data = {
            'AAPL': pd.Series([100, 101, 102], index=['2023-01-01', '2023-01-02', '2023-01-03']),
            'GOOGL': pd.Series([2000, 2010], index=['2023-01-01', '2023-01-03'])  # 缺少01-02
        }
        
        # 使用'nan'策略（默认）
        aligned = Data.align_index(data, missing='nan')
        # 结果：GOOGL在2023-01-02处为NaN
        
        # 使用'drop'策略
        aligned = Data.align_index(data, missing='drop') 
        # 结果：只保留01-01和01-03两个日期
        
        # 使用'raise'策略
        try:
            aligned = Data.align_index(data, missing='raise')
        except ValueError:
            print("索引不匹配，抛出异常")
        ```
        """
        # 如果只有一个符号，无需对齐，直接返回
        if len(data) == 1:
            return data

        # 初始化统一索引为None
        index = None
        # 遍历所有符号的数据，构建统一索引
        for k, v in data.items():
            if index is None:
                # 第一个符号，直接使用其索引
                index = v.index
            else:
                # 检查当前符号的索引是否与统一索引完全匹配
                if len(index.intersection(v.index)) != len(index.union(v.index)):
                    # 索引不匹配，根据策略处理
                    if missing == 'nan':
                        # NaN策略：发出警告，使用并集索引（包含所有时间点）
                        warnings.warn("Symbols have mismatching index. "
                                      "Setting missing data points to NaN.", stacklevel=2)
                        index = index.union(v.index)
                    elif missing == 'drop':
                        # Drop策略：发出警告，使用交集索引（只保留共同时间点）
                        warnings.warn("Symbols have mismatching index. "
                                      "Dropping missing data points.", stacklevel=2)
                        index = index.intersection(v.index)
                    elif missing == 'raise':
                        # Raise策略：抛出异常，不允许索引不匹配
                        raise ValueError("Symbols have mismatching index")
                    else:
                        # 无效策略，抛出异常
                        raise ValueError(f"missing='{missing}' is not recognized")

        # 使用统一索引对所有数据进行重新索引
        new_data = {k: v.reindex(index=index) for k, v in data.items()}
        return new_data

    @classmethod
    def align_columns(cls, data: tp.Data, missing: str = 'raise') -> tp.Data:
        """
        列对齐方法 - 统一不同符号数据的列结构
        
        该方法确保所有符号的数据具有一致的列结构，这对于多符号数据的
        统一处理和分析至关重要。默认策略为'raise'，比索引对齐更严格。
        
        Args:
            data (tp.Data): 待对齐的数据字典
            missing (str): 缺失列处理策略，默认'raise'
                - 'nan': 将缺失的列设置为NaN
                - 'drop': 删除不匹配的列
                - 'raise': 遇到不匹配时抛出异常（默认，更严格）
                
        Returns:
            tp.Data: 列对齐后的数据字典
        """
        # 如果只有一个符号，无需对齐，直接返回
        if len(data) == 1:
            return data

        # 初始化状态变量
        columns = None              # 统一的列索引
        multiple_columns = False    # 是否存在多列数据（DataFrame）
        name_is_none = False       # 是否存在名称为None的Series
        
        # 第一轮遍历：分析数据结构，确定统一的列索引
        for k, v in data.items():
            if isinstance(v, pd.Series):
                # 检查Series的名称是否为None
                if v.name is None:
                    name_is_none = True
                # 临时转换为DataFrame以统一处理列
                v = v.to_frame()
            else:
                # 标记存在多列数据
                multiple_columns = True
                
            if columns is None:
                # 第一个符号，直接使用其列索引
                columns = v.columns
            else:
                # 检查当前符号的列是否与统一列索引完全匹配
                if len(columns.intersection(v.columns)) != len(columns.union(v.columns)):
                    # 列不匹配，根据策略处理
                    if missing == 'nan':
                        # NaN策略：发出警告，使用并集列索引（包含所有列）
                        warnings.warn("Symbols have mismatching columns. "
                                      "Setting missing data points to NaN.", stacklevel=2)
                        columns = columns.union(v.columns)
                    elif missing == 'drop':
                        # Drop策略：发出警告，使用交集列索引（只保留共同列）
                        warnings.warn("Symbols have mismatching columns. "
                                      "Dropping missing data points.", stacklevel=2)
                        columns = columns.intersection(v.columns)
                    elif missing == 'raise':
                        # Raise策略：抛出异常，不允许列不匹配
                        raise ValueError("Symbols have mismatching columns")
                    else:
                        # 无效策略，抛出异常
                        raise ValueError(f"missing='{missing}' is not recognized")

        # 第二轮遍历：使用统一列索引对所有数据进行重新索引
        new_data = {}
        for k, v in data.items():
            # 将Series转换为DataFrame进行统一处理
            if isinstance(v, pd.Series):
                v = v.to_frame()
            # 使用统一列索引进行重新索引
            v = v.reindex(columns=columns)
            
            # 如果原始数据都是单列，则转换回Series
            if not multiple_columns:
                v = v[columns[0]]  # 提取第一列（也是唯一列）
                # 如果原始Series名称为None，则恢复None名称
                if name_is_none:
                    v = v.rename(None)
            new_data[k] = v
        return new_data

    @classmethod
    def select_symbol_kwargs(cls, symbol: tp.Label, kwargs: dict) -> dict:
        """
        符号参数选择方法 - 为特定符号筛选相关的关键字参数
        
        该方法是symbol_dict机制的核心实现，允许为不同符号提供差异化的配置参数。
        这在处理具有不同特性的金融工具时特别有用。
        
        Args:
            symbol (tp.Label): 目标符号名称
            kwargs (dict): 包含参数的字典，值可能是symbol_dict或普通值
            
        Returns:
            dict: 筛选后的参数字典，只包含适用于指定符号的参数
            
        工作原理：
        1. 遍历所有传入的关键字参数
        2. 如果参数值是symbol_dict实例：
           - 检查是否包含目标符号的配置
           - 如果包含，使用符号特定的值
           - 如果不包含，跳过该参数
        3. 如果参数值是普通值，直接使用该值
        
        使用示例：
        ```python
        # 定义差异化参数
        kwargs = {
            'start_date': '2020-01-01',  # 所有符号使用相同日期
            'period': symbol_dict({      # 不同符号使用不同周期
                'AAPL': '1y',
                'GOOGL': '2y'
            }),
            'interval': symbol_dict({    # 不同符号使用不同间隔
                'AAPL': '1d',
                'GOOGL': '1h'
            })
        }
        
        # 为AAPL选择参数
        aapl_kwargs = Data.select_symbol_kwargs('AAPL', kwargs)
        # 结果: {'start_date': '2020-01-01', 'period': '1y', 'interval': '1d'}
        
        # 为GOOGL选择参数
        googl_kwargs = Data.select_symbol_kwargs('GOOGL', kwargs)
        # 结果: {'start_date': '2020-01-01', 'period': '2y', 'interval': '1h'}
        
        # 为不存在的符号选择参数
        msft_kwargs = Data.select_symbol_kwargs('MSFT', kwargs)
        # 结果: {'start_date': '2020-01-01'}  # 只包含通用参数
        ```
        
        实际应用场景：
        - 不同资产类别需要不同的数据获取参数
        - 不同市场有不同的交易时间和数据可用性
        - 个别符号需要特殊的数据处理配置
        """
        # 初始化结果字典
        _kwargs = dict()
        # 遍历所有传入的关键字参数
        for k, v in kwargs.items():
            if isinstance(v, symbol_dict):
                # 如果参数值是symbol_dict，检查是否包含目标符号
                if symbol in v:
                    # 使用符号特定的值
                    _kwargs[k] = v[symbol]
                # 如果不包含目标符号，跳过该参数（不添加到结果中）
            else:
                # 如果是普通值，直接使用
                _kwargs[k] = v
        return _kwargs

    @classmethod
    def from_data(cls: tp.Type[DataT],
                  data: tp.Data,
                  tz_localize: tp.Optional[tp.TimezoneLike] = None,
                  tz_convert: tp.Optional[tp.TimezoneLike] = None,
                  missing_index: tp.Optional[str] = None,
                  missing_columns: tp.Optional[str] = None,
                  wrapper_kwargs: tp.KwargsLike = None,
                  **kwargs) -> DataT:
        """
        从数据字典创建Data实例的工厂方法
        
        这是Data类的核心工厂方法，用于从已有的数据字典创建新的Data实例。
        该方法处理数据预处理、对齐、时区转换等所有必要步骤。
        
        Args:
            data (dict): 以符号为键的数据字典，值为类数组对象
            tz_localize (timezone_like, optional): 时区本地化参数
                如果索引是时区无关的，将其转换为指定时区
                参考 `vectorbt.utils.datetime_.to_timezone`
            tz_convert (timezone_like, optional): 时区转换参数
                将索引从一个时区转换为另一个时区
                参考 `vectorbt.utils.datetime_.to_timezone`
            missing_index (str, optional): 索引缺失处理策略
                参考 `Data.align_index` 方法说明
            missing_columns (str, optional): 列缺失处理策略
                参考 `Data.align_columns` 方法说明
            wrapper_kwargs (dict, optional): 传递给ArrayWrapper的关键字参数
            **kwargs: 传递给__init__方法的其他关键字参数
        """
        # 导入全局设置
        from vectorbt._settings import settings
        data_cfg = settings['data']

        # 获取全局默认配置，如果参数为None则使用配置文件中的默认值
        if tz_localize is None:
            tz_localize = data_cfg['tz_localize']       # 时区本地化默认设置
        if tz_convert is None:
            tz_convert = data_cfg['tz_convert']         # 时区转换默认设置
        if missing_index is None:
            missing_index = data_cfg['missing_index']   # 索引缺失处理默认策略
        if missing_columns is None:
            missing_columns = data_cfg['missing_columns'] # 列缺失处理默认策略
        if wrapper_kwargs is None:
            wrapper_kwargs = {}                         # 包装器参数默认为空字典

        # 创建数据字典的副本，避免修改原始数据
        data = data.copy()
        # 遍历所有符号的数据，进行标准化处理
        for k, v in data.items():
            # 第一步：将数组类型转换为pandas对象
            if not isinstance(v, (pd.Series, pd.DataFrame)):
                # 如果不是pandas对象，先转换为numpy数组
                v = np.asarray(v)
                if v.ndim == 1:
                    # 一维数组转换为Series
                    v = pd.Series(v)
                else:
                    # 多维数组转换为DataFrame
                    v = pd.DataFrame(v)

            # 第二步：处理时间索引相关操作
            if isinstance(v.index, pd.DatetimeIndex):
                # 时区本地化：将时区无关的索引转换为指定时区
                if tz_localize is not None:
                    if not is_tz_aware(v.index):
                        # 只对时区无关的索引进行本地化
                        v = v.tz_localize(to_timezone(tz_localize))
                # 时区转换：将已有时区的索引转换为其他时区
                if tz_convert is not None:
                    v = v.tz_convert(to_timezone(tz_convert))
                # 推断并设置时间频率（如日频、小时频等）
                v.index.freq = v.index.inferred_freq
            # 更新处理后的数据
            data[k] = v

        # 第三步：数据对齐处理
        # 对齐所有符号的时间索引，确保时间点一致
        data = cls.align_index(data, missing=missing_index)
        # 对齐所有符号的列结构，确保数据格式一致
        data = cls.align_columns(data, missing=missing_columns)

        # 第四步：创建新的Data实例
        # 获取符号列表
        symbols = list(data.keys())
        # 使用第一个符号的数据创建ArrayWrapper
        wrapper = ArrayWrapper.from_obj(data[symbols[0]], **wrapper_kwargs)
        # 创建并返回新的Data实例
        return cls(
            wrapper,                    # 数组包装器
            data,                      # 处理后的数据字典
            tz_localize=tz_localize,   # 时区本地化参数
            tz_convert=tz_convert,     # 时区转换参数
            missing_index=missing_index,    # 索引缺失处理策略
            missing_columns=missing_columns, # 列缺失处理策略
            download_kwargs={},        # 空的下载参数（因为是从现有数据创建）
            **kwargs                   # 其他额外参数
        )

    @classmethod
    def download_symbol(cls, symbol: tp.Label, **kwargs) -> tp.SeriesFrame:
        """
        下载单个符号数据的抽象方法
        
        这是一个抽象方法，必须在子类中实现具体的数据下载逻辑。
        不同的数据源（如Yahoo Finance、Alpha Vantage、Quandl等）需要
        实现各自的下载逻辑。
        
        Args:
            symbol (tp.Label): 要下载的金融符号（如'AAPL', 'GOOGL'）
            **kwargs: 下载参数，具体参数取决于数据源的要求
            
        Returns:
            tp.SeriesFrame: 返回pandas Series或DataFrame，包含该符号的数据
            
        实现示例：
        ```python
        class YahooData(vbt.Data):
            @classmethod
            def download_symbol(cls, symbol, period='1y', interval='1d', **kwargs):
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                return data
                
        class AlphaVantageData(vbt.Data):
            @classmethod  
            def download_symbol(cls, symbol, function='TIME_SERIES_DAILY', 
                              api_key=None, **kwargs):
                # 实现Alpha Vantage API调用逻辑
                # ...
                return data
        ```
        
        注意事项：
        - 子类必须实现此方法，否则会抛出NotImplementedError
        - 返回的数据应该是pandas对象（Series或DataFrame）
        - 建议在实现中处理网络异常和数据验证
        """
        raise NotImplementedError

    @classmethod
    def download(cls: tp.Type[DataT],
                 symbols: tp.Union[tp.Label, tp.Labels],
                 tz_localize: tp.Optional[tp.TimezoneLike] = None,
                 tz_convert: tp.Optional[tp.TimezoneLike] = None,
                 missing_index: tp.Optional[str] = None,
                 missing_columns: tp.Optional[str] = None,
                 wrapper_kwargs: tp.KwargsLike = None,
                 **kwargs) -> DataT:
        """
        批量下载多个符号数据的主要方法
        
        这是Data类的核心下载方法，通过调用download_symbol方法来下载
        多个符号的数据，并自动处理数据对齐、时区转换等预处理步骤。
        
        Args:
            symbols (hashable or sequence of hashable): 单个符号或符号列表
                注意：元组被视为单个符号（因为它是可哈希的）
            tz_localize (any): 时区本地化参数，参考Data.from_data
            tz_convert (any): 时区转换参数，参考Data.from_data  
            missing_index (str): 索引缺失处理策略，参考Data.from_data
            missing_columns (str): 列缺失处理策略，参考Data.from_data
            wrapper_kwargs (dict): ArrayWrapper参数，参考Data.from_data
            **kwargs: 传递给Data.download_symbol的参数
                如果不同符号需要不同参数，可使用symbol_dict
                
        Returns:
            DataT: 包含所有下载数据的新Data实例
            
        工作流程：
        1. 验证和标准化符号列表
        2. 遍历每个符号，调用download_symbol方法
        3. 为每个符号选择适用的参数（支持symbol_dict）
        4. 收集所有下载的数据到字典中
        5. 调用from_data方法创建Data实例
        
        使用示例：
        ```python
        # 示例1：下载单个符号
        data = MyData.download('AAPL')
        
        # 示例2：下载多个符号
        data = MyData.download(['AAPL', 'GOOGL', 'MSFT'])
        
        # 示例3：使用统一参数
        data = MyData.download(
            ['AAPL', 'GOOGL'], 
            period='2y', 
            interval='1d'
        )
        
        # 示例4：使用差异化参数
        data = MyData.download(
            ['AAPL', 'GOOGL'],
            period=symbol_dict({
                'AAPL': '1y',
                'GOOGL': '2y'
            }),
            interval='1d'
        )
        
        # 示例5：带时区处理
        data = MyData.download(
            ['AAPL', 'GOOGL'],
            tz_localize='UTC',
            tz_convert='US/Eastern'
        )
        ```
        
        错误处理：
        - 符号类型验证：确保符号是可哈希的或可哈希对象的序列
        - 网络异常：由具体的download_symbol实现处理
        - 数据对齐：由from_data方法自动处理
        """
        # 标准化符号列表：将单个符号转换为列表
        if checks.is_hashable(symbols):
            symbols = [symbols]  # 单个符号转换为包含一个元素的列表
        elif not checks.is_sequence(symbols):
            # 如果既不是可哈希的，也不是序列，则抛出类型错误
            raise TypeError("Symbols must be either hashable or sequence of hashable")

        # 初始化数据字典，用于存储所有符号的数据
        data = dict()
        # 遍历每个符号，逐个下载数据
        for s in symbols:
            # 为当前符号选择适用的关键字参数
            _kwargs = cls.select_symbol_kwargs(s, kwargs)

            # 调用download_symbol方法下载当前符号的数据
            data[s] = cls.download_symbol(s, **_kwargs)

        # 使用from_data工厂方法创建新的Data实例
        return cls.from_data(
            data,                           # 下载的数据字典
            tz_localize=tz_localize,       # 时区本地化参数
            tz_convert=tz_convert,         # 时区转换参数
            missing_index=missing_index,    # 索引缺失处理策略
            missing_columns=missing_columns, # 列缺失处理策略
            wrapper_kwargs=wrapper_kwargs,  # 数组包装器参数
            download_kwargs=kwargs          # 保存原始下载参数供后续更新使用
        )

    def update_symbol(self, symbol: tp.Label, **kwargs) -> tp.SeriesFrame:
        """
        更新单个符号数据的抽象方法
        
        这是一个抽象方法，需要在子类中实现具体的数据更新逻辑。
        与download_symbol不同，这是一个实例方法，可以访问现有数据
        和原始下载参数，通常用于增量更新。
        
        Args:
            symbol (tp.Label): 要更新的金融符号
            **kwargs: 更新参数，会与原始下载参数合并
            
        Returns:
            tp.SeriesFrame: 返回新的或更新的pandas对象
            
        实现要点：
        - 可以访问self.data[symbol]获取现有数据
        - 可以访问self.download_kwargs获取原始参数
        - 应该返回新的数据，而不是修改现有数据
        - 通常实现增量更新逻辑
        
        实现示例：
        ```python
        class YahooData(vbt.Data):
            def update_symbol(self, symbol, **kwargs):
                # 获取现有数据的最后日期
                last_date = self.data[symbol].index[-1]
                
                # 合并原始下载参数和新参数
                download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
                update_kwargs = merge_dicts(download_kwargs, kwargs)
                
                # 设置更新的起始日期
                update_kwargs['start'] = last_date
                
                # 下载新数据
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                new_data = ticker.history(**update_kwargs)
                
                return new_data
        ```
        
        注意事项：
        - 子类必须实现此方法用于数据更新功能
        - 建议处理重复数据点的去重逻辑
        - 返回的数据会与现有数据合并
        """
        raise NotImplementedError

    def update(self: DataT, **kwargs) -> DataT:
        """
        批量更新所有符号数据的主要方法
        
        该方法通过调用update_symbol方法来更新所有符号的数据，
        并返回一个新的Data实例。这是实现数据增量更新的核心方法。
        
        Args:
            **kwargs: 传递给Data.update_symbol的参数
                如果不同符号需要不同参数，可使用symbol_dict
                
        Returns:
            DataT: 包含更新数据的新Data实例
            
        工作流程：
        1. 遍历所有现有符号
        2. 为每个符号选择适用的更新参数
        3. 调用update_symbol方法获取新数据
        4. 处理数组到pandas的转换
        5. 执行时区处理操作
        6. 对齐新数据的索引和列
        7. 合并旧数据和新数据
        8. 去重并保持最新数据
        9. 创建新的Data实例
        """
        # 初始化新数据字典，用于存储更新后的数据
        new_data = dict()
        # 遍历所有现有符号，逐个更新数据
        for k, v in self.data.items():
            # 为当前符号选择适用的更新参数
            _kwargs = self.select_symbol_kwargs(k, kwargs)

            # 调用update_symbol方法获取该符号的新数据
            new_obj = self.update_symbol(k, **_kwargs)

            # 将数组类型转换为pandas对象（如果需要）
            if not isinstance(new_obj, (pd.Series, pd.DataFrame)):
                # 转换为numpy数组
                new_obj = np.asarray(new_obj)
                # 创建连续的索引，从现有数据的最后一个索引开始
                index = pd.RangeIndex(
                    start=v.index[-1],                    # 起始索引
                    stop=v.index[-1] + new_obj.shape[0],  # 结束索引
                    step=1                                # 步长
                )
                if new_obj.ndim == 1:
                    # 一维数组转换为Series
                    new_obj = pd.Series(new_obj, index=index)
                else:
                    # 多维数组转换为DataFrame
                    new_obj = pd.DataFrame(new_obj, index=index)

            # 对新数据执行时区相关操作
            if isinstance(new_obj.index, pd.DatetimeIndex):
                # 时区本地化：如果需要且索引是时区无关的
                if self.tz_localize is not None:
                    if not is_tz_aware(new_obj.index):
                        new_obj = new_obj.tz_localize(to_timezone(self.tz_localize))
                # 时区转换：如果需要转换时区
                if self.tz_convert is not None:
                    new_obj = new_obj.tz_convert(to_timezone(self.tz_convert))

            # 存储处理后的新数据
            new_data[k] = new_obj

        # 对所有新数据进行索引和列对齐
        new_data = self.align_index(new_data, missing=self.missing_index)
        new_data = self.align_columns(new_data, missing=self.missing_columns)

        # 将旧数据和新数据进行合并
        for k, v in new_data.items():
            # 确保新数据与旧数据的结构一致
            if isinstance(self.data[k], pd.Series):
                if isinstance(v, pd.DataFrame):
                    # 如果旧数据是Series，新数据是DataFrame，提取对应列
                    v = v[self.data[k].name]
            else:
                # 如果旧数据是DataFrame，确保新数据包含相同的列
                v = v[self.data[k].columns]
            
            # 沿时间轴连接旧数据和新数据
            v = pd.concat((self.data[k], v), axis=0)
            # 去除重复的索引，保留最新的数据（keep='last'）
            v = v[~v.index.duplicated(keep='last')]
            
            # 如果是时间索引，重新推断频率
            if isinstance(v.index, pd.DatetimeIndex):
                v.index.freq = v.index.inferred_freq
            # 更新合并后的数据
            new_data[k] = v

        # 创建新的Data实例
        # 获取更新后的索引（使用第一个符号的索引作为参考）
        new_index = new_data[self.symbols[0]].index
        # 使用replace方法创建新实例，更新包装器和数据
        return self.replace(
            wrapper=self.wrapper.replace(index=new_index),  # 更新包装器的索引
            data=new_data                                    # 使用合并后的数据
        )

    @cached_method
    def concat(self, level_name: str = 'symbol') -> tp.Data:
        """
        数据连接方法 - 将多符号数据重组为以列名为键的字典
        
        该方法将Data实例中以符号为键的数据字典重新组织为以列名为键的字典，
        其中每个值是包含所有符号数据的Series或DataFrame。这种转换对于
        跨符号分析和比较非常有用。
        
        Args:
            level_name (str): 多符号时用作列索引名称的标签，默认'symbol'
            
        Returns:
            tp.Data: 重组后的数据字典，键为列名，值为包含所有符号的pandas对象
            
        数据转换逻辑：
        - 单符号：返回以列名为键的Series字典
        - 多符号：返回以列名为键的DataFrame字典，符号作为列
        
        使用示例：
        ```python
        # 原始数据结构
        data.data = {
            'AAPL': pd.DataFrame({
                'Open': [100, 101], 'Close': [101, 102], 'Volume': [1000, 1100]
            }),
            'GOOGL': pd.DataFrame({
                'Open': [2000, 2010], 'Close': [2010, 2020], 'Volume': [500, 600]
            })
        }
        
        # 连接后的数据结构
        concat_data = data.concat()
        # 结果：
        # {
        #     'Open': DataFrame with columns ['AAPL', 'GOOGL'],
        #     'Close': DataFrame with columns ['AAPL', 'GOOGL'], 
        #     'Volume': DataFrame with columns ['AAPL', 'GOOGL']
        # }
        
        # 访问特定指标的所有符号数据
        all_close_prices = concat_data['Close']
        #        AAPL  GOOGL
        # 0      101   2010
        # 1      102   2020
        
        # 单符号情况
        single_data.concat()
        # 结果：
        # {
        #     'Open': Series with name 'AAPL',
        #     'Close': Series with name 'AAPL',
        #     'Volume': Series with name 'AAPL'
        # }
        ```
        """
        # 获取第一个符号的数据作为参考
        first_data = self.data[self.symbols[0]]
        # 使用第一个符号的索引作为统一索引
        index = first_data.index
        
        # 确定列名列表
        if isinstance(first_data, pd.Series):
            # 如果是Series，列名为其name属性
            columns = pd.Index([first_data.name])
        else:
            # 如果是DataFrame，使用其列索引
            columns = first_data.columns
            
        # 根据符号数量创建不同的数据结构
        if len(self.symbols) > 1:
            # 多符号：创建DataFrame字典，符号作为列
            new_data = {c: pd.DataFrame(
                index=index,                                    # 使用统一的时间索引
                columns=pd.Index(self.symbols, name=level_name) # 符号作为列，设置列索引名称
            ) for c in columns}
        else:
            # 单符号：创建Series字典
            new_data = {c: pd.Series(
                index=index,           # 使用统一的时间索引
                name=self.symbols[0]   # 使用符号名称作为Series名称
            ) for c in columns}
            
        # 填充数据：遍历每个列和每个符号
        for c in columns:
            for s in self.symbols:
                # 获取当前符号当前列的数据
                if isinstance(self.data[s], pd.Series):
                    # 如果原数据是Series，直接使用
                    col_data = self.data[s]
                else:
                    # 如果原数据是DataFrame，提取指定列
                    col_data = self.data[s][c]
                    
                # 将数据填入相应位置
                if len(self.symbols) > 1:
                    # 多符号：填入DataFrame的对应列
                    new_data[c].loc[:, s] = col_data
                else:
                    # 单符号：填入Series
                    new_data[c].loc[:] = col_data
                    
        # 优化数据类型：推断并设置最适合的数据类型
        for c in columns:
            new_data[c] = new_data[c].infer_objects()
            
        return new_data

    def get(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.MaybeTuple[tp.SeriesFrame]:
        """
        数据获取方法 - 灵活获取指定列或所有数据的统一接口
        
        该方法提供了获取Data实例中数据的统一接口，根据符号数量和列参数
        返回不同格式的数据。这是访问Data中存储数据的主要方法。
        
        Args:
            column (tp.Label or list, optional): 要获取的列名或列名列表
                - None: 获取所有数据
                - str: 获取指定列的数据
                - list: 获取多个指定列的数据
            **kwargs: 传递给concat方法的额外参数
            
        Returns:
            tp.MaybeTuple[tp.SeriesFrame]: 根据输入参数返回不同格式：
                - 单符号单列：返回Series
                - 单符号多列：返回DataFrame
                - 多符号单列：返回DataFrame（符号为列）
                - 多符号多列：返回DataFrame元组
                - 所有数据：返回DataFrame或DataFrame元组
        ```
        
        性能考虑：
        - 多符号数据会调用concat方法（已缓存）
        - 单符号数据直接返回，无额外开销
        - 列选择在数据获取后进行，避免不必要的计算
        
        注意事项：
        - 返回的数据是原始数据的视图或副本，修改可能影响原数据
        - 多符号单列返回的DataFrame以符号为列，时间为索引
        - 列名必须在所有符号的数据中都存在（经过对齐处理）
        """
        # 单符号情况：直接从data字典获取数据
        if len(self.symbols) == 1:
            if column is None:
                # 返回该符号的所有数据（Series或DataFrame）
                return self.data[self.symbols[0]]
            # 返回该符号的指定列数据
            return self.data[self.symbols[0]][column]

        # 多符号情况：需要先进行数据连接
        concat_data = self.concat(**kwargs)
        
        # 如果连接后只有一列数据，直接返回该列
        if len(concat_data) == 1:
            return tuple(concat_data.values())[0]
            
        # 根据column参数返回相应数据
        if column is not None:
            if isinstance(column, list):
                # 多列：返回指定列的元组
                return tuple([concat_data[c] for c in column])
            # 单列：返回指定列的DataFrame
            return concat_data[column]
            
        # 所有列：返回所有列的元组
        return tuple(concat_data.values())

    # ############# 统计分析功能 ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """
        统计分析默认配置属性
        
        该属性返回Data.stats方法的默认配置，合并了StatsBuilderMixin的
        基础配置和vectorbt全局设置中的数据统计配置。
        
        Returns:
            tp.Kwargs: 合并后的默认配置字典
            
        配置来源：
        1. StatsBuilderMixin.stats_defaults: 基础统计构建器的默认配置
        2. settings['data']['stats']: 全局设置中的数据统计专用配置
        
        配置内容通常包括：
        - 默认的统计指标选择
        - 分组和聚合设置
        - 输出格式配置
        - 计算精度设置
        
        使用示例：
        ```python
        # 查看默认配置
        defaults = data.stats_defaults
        print(defaults)
        
        # 使用默认配置计算统计指标
        stats = data.stats()  # 自动应用defaults中的配置
        
        # 覆盖特定配置
        custom_stats = data.stats(
            metrics=['start', 'end', 'total_symbols'],  # 覆盖默认指标
            **defaults  # 其他配置使用默认值
        )
        ```
        """
        # 导入全局设置
        from vectorbt._settings import settings
        # 获取数据统计专用配置
        data_stats_cfg = settings['data']['stats']

        # 合并基础配置和专用配置
        return merge_dicts(
            StatsBuilderMixin.stats_defaults.__get__(self),  # 基础统计构建器默认配置
            data_stats_cfg                                   # 数据统计专用配置
        )

    # 类级别的统计指标配置，定义了Data类支持的所有内置统计指标
    _metrics: tp.ClassVar[Config] = Config(
        dict(
            # 数据起始时间指标
            start=dict(
                title='Start',                                    # 指标显示名称
                calc_func=lambda self: self.wrapper.index[0],    # 计算函数：获取索引的第一个元素
                agg_func=None,                                   # 聚合函数：无需聚合
                tags='wrapper'                                   # 标签：属于包装器相关指标
            ),
            # 数据结束时间指标
            end=dict(
                title='End',                                     # 指标显示名称
                calc_func=lambda self: self.wrapper.index[-1],  # 计算函数：获取索引的最后一个元素
                agg_func=None,                                   # 聚合函数：无需聚合
                tags='wrapper'                                   # 标签：属于包装器相关指标
            ),
            # 数据时间跨度指标
            period=dict(
                title='Period',                                  # 指标显示名称
                calc_func=lambda self: len(self.wrapper.index), # 计算函数：计算索引长度
                apply_to_timedelta=True,                        # 应用时间差：转换为时间跨度格式
                agg_func=None,                                   # 聚合函数：无需聚合
                tags='wrapper'                                   # 标签：属于包装器相关指标
            ),
            # 符号总数指标
            total_symbols=dict(
                title='Total Symbols',                          # 指标显示名称
                calc_func=lambda self: len(self.symbols),      # 计算函数：计算符号数量
                agg_func=None,                                   # 聚合函数：无需聚合
                tags='data'                                      # 标签：属于数据相关指标
            ),
            # 空值统计指标
            null_counts=dict(
                title='Null Counts',                            # 指标显示名称
                # 计算函数：为每个符号计算空值数量
                calc_func=lambda self, group_by:
                {
                    k: v.isnull().vbt(wrapper=self.wrapper).sum(group_by=group_by)
                    for k, v in self.data.items()
                },
                tags='data'                                      # 标签：属于数据相关指标
            )
        ),
        copy_kwargs=dict(copy_mode='deep')  # 深度复制配置，确保配置修改不影响原始定义
    )

    @property
    def metrics(self) -> Config:
        """
        统计指标配置属性
        
        返回当前实例可用的统计指标配置。这个配置定义了Data类支持的
        所有统计指标及其计算方法。
        
        Returns:
            Config: 统计指标配置对象，包含所有可用指标的定义
            
        内置指标说明：
        1. start: 数据开始时间
           - 返回数据的第一个时间点
           - 用于了解数据的时间范围起点
           
        2. end: 数据结束时间
           - 返回数据的最后一个时间点
           - 用于了解数据的时间范围终点
           
        3. period: 数据时间跨度
           - 返回数据的总时间长度
           - 自动转换为时间差格式（如"365 days"）
           
        4. total_symbols: 符号总数
           - 返回Data实例包含的符号数量
           - 用于了解数据集的规模
           
        5. null_counts: 空值统计
           - 返回每个符号的空值数量
           - 用于数据质量评估
        
        使用示例：
        ```python
        # 查看所有可用指标
        available_metrics = list(data.metrics.keys())
        print(f"可用指标: {available_metrics}")
        
        # 查看特定指标的配置
        start_config = data.metrics['start']
        print(f"开始时间指标配置: {start_config}")
        
        # 计算所有指标
        all_stats = data.stats()
        
        # 计算特定标签的指标
        wrapper_stats = data.stats(tags='wrapper')  # 只计算包装器相关指标
        data_stats = data.stats(tags='data')        # 只计算数据相关指标
        
        # 计算特定指标
        basic_stats = data.stats(metrics=['start', 'end', 'total_symbols'])
        ```
        
        扩展指标：
        子类可以通过修改_metrics配置来添加自定义指标：
        ```python
        class ExtendedData(vbt.Data):
            _metrics = vbt.Data._metrics.copy()
            _metrics['custom_metric'] = dict(
                title='Custom Metric',
                calc_func=lambda self: self.custom_calculation(),
                tags='custom'
            )
        ```
        """
        return self._metrics

    # ############# 数据可视化功能 ############# #

    def plot(self,
             column: tp.Optional[tp.Label] = None,
             base: tp.Optional[float] = None,
             **kwargs) -> tp.Union[tp.BaseFigure, plotting.Scatter]:  # pragma: no cover
        """
        数据绘图方法 - 创建金融数据的可视化图表
        
        该方法提供了Data实例的核心绘图功能，支持单列或多列数据的可视化，
        并可选择性地对价格数据进行基准重置。
        
        Args:
            column (str, optional): 要绘制的列名
                - None: 绘制所有列（如果只有一列）或第一列
                - str: 绘制指定列的数据
            base (float, optional): 价格基准重置值
                - None: 使用原始价格数据
                - float: 将所有系列重置到指定的初始基准值
                注意：该列应包含价格数据才有意义
            **kwargs: 传递给vectorbt.generic.accessors.GenericAccessor.plot的关键字参数
            
        Returns:
            tp.Union[tp.BaseFigure, plotting.Scatter]: 
                返回plotly图表对象或vectorbt的Scatter对象
        使用示例：
        ```python
        # 示例1：基本价格图表
        import vectorbt as vbt
        
        # 下载数据
        data = vbt.YFData.download(['AAPL', 'GOOGL', 'MSFT'], period='1y')
        
        # 绘制收盘价
        fig = data.plot(column='Close')
        fig.show()
        
        # 示例2：基准重置比较
        # 将所有股票价格重置到100，便于比较相对表现
        fig = data.plot(column='Close', base=100)
        fig.update_layout(
            title='股价相对表现比较（基准=100）',
            yaxis_title='相对价格'
        )
        fig.show()
        
        # 示例3：自定义样式
        fig = data.plot(
            column='Close',
            base=1,  # 重置到1
            trace_kwargs=dict(
                line=dict(width=2)  # 设置线条宽度
            ),
            layout_kwargs=dict(
                title='标准化价格走势',
                template='plotly_dark'  # 使用暗色主题
            )
        )
        
        # 示例4：成交量图表
        volume_fig = data.plot(
            column='Volume',
            layout_kwargs=dict(
                title='成交量对比',
                yaxis_title='成交量'
            )
        )
        ```
        """
        # 选择指定列的数据，不进行分组
        self_col = self.select_one(column=column, group_by=False)
        # 获取选中列的数据
        data = self_col.get()
        # 如果指定了基准值，对数据进行基准重置
        if base is not None:
            data = data.vbt.rebase(base)
        # 使用vectorbt的绘图接口创建图表
        return data.vbt.plot(**kwargs)

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """
        图表绘制默认配置属性
        
        该属性返回Data.plots方法的默认配置，合并了PlotsBuilderMixin的
        基础配置和vectorbt全局设置中的数据绘图配置。
        
        Returns:
            tp.Kwargs: 合并后的默认绘图配置字典
            
        配置来源：
        1. PlotsBuilderMixin.plots_defaults: 基础图表构建器的默认配置
        2. settings['data']['plots']: 全局设置中的数据绘图专用配置
        
        配置内容通常包括：
        - 默认的子图选择和布局
        - 图表样式和主题设置
        - 轨迹（trace）的默认属性
        - 图例和标题的配置
        
        使用示例：
        ```python
        # 查看默认绘图配置
        defaults = data.plots_defaults
        print(defaults)
        
        # 使用默认配置创建图表
        fig = data.plots()  # 自动应用defaults中的配置
        
        # 覆盖特定配置
        custom_fig = data.plots(
            subplots=['plot'],  # 覆盖默认子图
            **defaults  # 其他配置使用默认值
        )
        ```
        """
        # 导入全局设置
        from vectorbt._settings import settings
        # 获取数据绘图专用配置
        data_plots_cfg = settings['data']['plots']

        # 合并基础配置和专用配置
        return merge_dicts(
            PlotsBuilderMixin.plots_defaults.__get__(self),  # 基础图表构建器默认配置
            data_plots_cfg                                   # 数据绘图专用配置
        )

    # 类级别的子图配置，定义了Data类支持的所有子图类型
    _subplots: tp.ClassVar[Config] = Config(
        dict(
            # 基础绘图子图配置
            plot=dict(
                check_is_not_grouped=True,      # 检查：确保数据未分组
                plot_func='plot',               # 绘图函数：使用plot方法
                pass_add_trace_kwargs=True,     # 传递轨迹参数：允许自定义轨迹属性
                tags='data'                     # 标签：属于数据相关子图
            )
        ),
        copy_kwargs=dict(copy_mode='deep')  # 深度复制配置，确保配置修改不影响原始定义
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


# 自动生成文档：覆盖统计指标的文档字符串
# 这将自动为每个在_metrics中定义的指标生成详细的文档
Data.override_metrics_doc(__pdoc__)

# 自动生成文档：覆盖子图的文档字符串  
# 这将自动为每个在_subplots中定义的子图生成详细的文档
Data.override_subplots_doc(__pdoc__)
