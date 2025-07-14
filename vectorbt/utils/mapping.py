# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT UTILS MODULE: 数据映射和转换工具
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于处理数据映射和转换的核心工具模块。
在量化分析中，经常需要在不同的数据表示形式之间进行转换，如枚举到标签、
索引到值、分类编码到可读文本等。该模块提供了一套完整的映射转换基础设施，
为vectorbt框架的数据标准化和互操作性提供了强大支持。

核心设计理念：
1. **统一映射接口**：提供统一的API来处理各种类型的映射关系
2. **灵活数据转换**：支持标量、数组、Series、DataFrame等多种数据类型的映射
3. **智能类型识别**：自动识别和处理不同数据源的映射格式
4. **容错和兼容性**：提供强大的错误处理和向后兼容性支持

主要功能模块：
- **基础映射操作**：映射反转、映射标准化等基本操作
- **映射转换引擎**：将各种类型的对象转换为标准映射格式
- **智能映射应用**：在复杂数据结构上应用映射规则
- **类型兼容处理**：处理不同数据类型间的映射兼容性

应用场景：
- **枚举处理**：将交易方向、订单类型等枚举值转换为可读标签
- **分类数据转换**：处理股票代码、行业分类、地区编码等分类数据
- **标签标准化**：统一不同数据源的标签格式和命名规范
- **配置参数映射**：将配置文件中的参数映射到内部数据结构
- **记录数据处理**：在记录数组中应用复杂的字段映射关系
- **可视化标签**：为图表生成友好的显示标签

技术特点：
- 支持命名元组、字典、Series、Index等多种映射源
- 提供大小写不敏感和下划线忽略的灵活匹配
- 智能类型检测和兼容性处理
- 高效的向量化映射操作
- 完整的缺失值和异常情况处理

与vectorbt生态系统的关系：
- 为Records模块提供字段映射和数据转换支持
- 为GenericAccessor提供标签映射功能
- 为统计分析模块提供分类数据的标准化处理
- 支持数据可视化中的标签友好化显示
- 为配置系统提供参数映射和验证功能

使用示例：
    >>> import vectorbt as vbt
    >>> from vectorbt.utils.mapping import to_mapping, apply_mapping
    
    >>> # 枚举映射示例
    >>> from collections import namedtuple
    >>> Direction = namedtuple('Direction', ['BUY', 'SELL'])(0, 1)
    >>> direction_map = to_mapping(Direction, reverse=True)
    >>> print(direction_map)  # {'BUY': 0, 'SELL': 1, None: -1}
    
    >>> # 应用映射到数据
    >>> trades = ['BUY', 'SELL', 'BUY']
    >>> mapped_trades = apply_mapping(trades, direction_map)
    >>> print(mapped_trades)  # [0, 1, 0]
    
    >>> # 股票代码映射
    >>> stocks = pd.Series(['AAPL', 'GOOGL', 'MSFT'])
    >>> stock_names = {'AAPL': 'Apple Inc.', 'GOOGL': 'Alphabet Inc.', 'MSFT': 'Microsoft Corp.'}
    >>> readable_names = apply_mapping(stocks, stock_names)
    >>> print(readable_names)

该模块是vectorbt框架数据互操作性的重要基础，确保了不同组件间数据格式的
一致性和转换的可靠性。
"""

import numpy as np  # 导入NumPy库，提供高性能的数值计算和数组操作
import pandas as pd  # 导入Pandas库，提供数据结构和数据分析工具

from vectorbt import _typing as tp  # 导入vectorbt的类型定义模块
from vectorbt.utils import checks  # 导入vectorbt的检查工具模块


def reverse_mapping(mapping: tp.Mapping) -> dict:
    """
    反转映射关系
    
    将输入映射的键值对调，生成新的映射字典。这是映射处理中的基础操作，
    广泛用于双向查找和映射关系的转换。
    
    核心功能：
    - 将原映射的值作为新映射的键
    - 将原映射的键作为新映射的值
    - 确保返回标准的字典格式
    
    参数：
        mapping (tp.Mapping): 要反转的映射对象，可以是字典或其他映射类型
    
    返回：
        dict: 反转后的字典，键值关系完全对调
    
    使用示例：
        >>> # 基本反转操作
        >>> original = {'apple': 1, 'banana': 2, 'orange': 3}
        >>> reversed_map = reverse_mapping(original)
        >>> print(reversed_map)  # {1: 'apple', 2: 'banana', 3: 'orange'}
        
        >>> # 交易方向枚举的反转
        >>> direction_codes = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
        >>> code_to_direction = reverse_mapping(direction_codes)
        >>> print(code_to_direction)  # {'BUY': 0, 'SELL': 1, 'HOLD': 2}
    
    注意事项：
        - 如果原映射中有重复值，反转后可能丢失某些键
        - 返回的总是dict类型，即使输入是其他映射类型
        - 对于None值和特殊值的处理需要特别注意
    """
    # 使用字典推导式实现键值对调
    # 遍历原映射的所有键值对，将值作为新键，键作为新值
    return {v: k for k, v in mapping.items()}


def to_mapping(mapping_like: tp.MappingLike, reverse: bool = False) -> dict:
    """
    将类映射对象转换为标准映射格式
    
    这是vectorbt映射系统的核心转换函数，能够将各种类型的类映射对象
    （命名元组、Series、Index、列表等）统一转换为标准的字典格式，
    为后续的映射操作提供一致的数据接口。
    
    转换策略：
    1. 命名元组：字段名作为值，索引作为键，并自动添加-1->None映射
    2. Series/Index：索引作为键，值作为值
    3. 列表/序列：位置索引作为键，元素作为值
    4. 已有映射：直接转换为字典格式
    
    设计考虑：
    - 自动处理命名元组的特殊性（用于枚举处理）
    - 为命名元组自动添加-1->None映射（vectorbt约定）
    - 统一不同数据源的映射接口
    - 保持映射的语义一致性
    
    参数：
        mapping_like (tp.MappingLike): 类映射对象，支持多种类型：
                                      - 命名元组（枚举）
                                      - pandas Series
                                      - pandas Index  
                                      - 列表、元组等序列
                                      - 字典等映射类型
        reverse (bool, optional): 是否对结果应用反转操作，默认为False
    
    返回：
        dict: 标准化的字典映射
    
    使用示例：
        >>> from collections import namedtuple
        >>> import pandas as pd
        
        >>> # 命名元组转换（枚举处理）
        >>> OrderType = namedtuple('OrderType', ['MARKET', 'LIMIT', 'STOP'])
        >>> order_enum = OrderType(0, 1, 2)
        >>> mapping = to_mapping(order_enum)
        >>> print(mapping)  # {0: 'MARKET', 1: 'LIMIT', 2: 'STOP', -1: None}
        
        >>> # 反转命名元组映射
        >>> reverse_mapping = to_mapping(order_enum, reverse=True)
        >>> print(reverse_mapping)  # {'MARKET': 0, 'LIMIT': 1, 'STOP': 2, None: -1}
        
        >>> # pandas Series转换
        >>> sectors = pd.Series(['Technology', 'Finance', 'Healthcare'], 
        ...                    index=['TECH', 'FIN', 'HEALTH'])
        >>> sector_mapping = to_mapping(sectors)
        >>> print(sector_mapping)  # {'TECH': 'Technology', 'FIN': 'Finance', 'HEALTH': 'Healthcare'}
        
        >>> # 列表转换（自动编号）
        >>> asset_names = ['Bitcoin', 'Ethereum', 'Litecoin']
        >>> asset_mapping = to_mapping(asset_names)
        >>> print(asset_mapping)  # {0: 'Bitcoin', 1: 'Ethereum', 2: 'Litecoin'}
        
        >>> # pandas Index转换
        >>> stock_index = pd.Index(['AAPL', 'GOOGL', 'MSFT'])
        >>> stock_mapping = to_mapping(stock_index)
        >>> print(stock_mapping)  # {0: 'AAPL', 1: 'GOOGL', 2: 'MSFT'}
        
        >>> # 在vectorbt记录处理中的应用
        >>> # 将交易状态枚举转换为映射，用于记录字段的标签化
        >>> TradeStatus = namedtuple('TradeStatus', ['OPEN', 'CLOSED', 'CANCELLED'])
        >>> status_enum = TradeStatus(0, 1, 2)
        >>> status_labels = to_mapping(status_enum, reverse=True)
        >>> # 在统计分析中使用该映射将数字代码转换为可读标签
    
    技术实现：
        - 使用checks模块进行类型检测
        - 特殊处理命名元组的-1键添加
        - 递归处理复杂的嵌套结构
        - 保持性能优化和内存效率
    """
    # 检查是否为命名元组（通常用于枚举定义）
    if checks.is_namedtuple(mapping_like):
        # 命名元组的特殊处理：字段名作为值，索引作为键
        # 使用_asdict()获取字段名到值的映射，然后反转
        mapping = {v: k for k, v in mapping_like._asdict().items()}
        
        # vectorbt约定：为命名元组自动添加-1->None的映射
        # 这用于处理无效或未定义的枚举值
        if -1 not in mapping_like:
            mapping[-1] = None
    
    # 如果不是标准映射类型，需要进行转换
    elif not checks.is_mapping(mapping_like):
        # 检查是否为pandas Index类型
        if checks.is_index(mapping_like):
            # 将Index转换为Series，重置索引以获得位置编号
            mapping_like = mapping_like.to_series().reset_index(drop=True)
        
        # 检查是否为pandas Series类型
        if checks.is_series(mapping_like):
            # Series的索引作为键，值作为值
            mapping = mapping_like.to_dict()
        else:
            # 对于列表、元组等序列类型，使用enumerate生成位置索引映射
            # 位置索引作为键，元素值作为值
            mapping = dict(enumerate(mapping_like))
    
    # 如果已经是映射类型，直接转换为字典
    else:
        mapping = dict(mapping_like)
    
    # 如果需要反转映射关系，应用reverse_mapping函数
    if reverse:
        mapping = reverse_mapping(mapping)
    
    # 返回标准化的字典映射
    return mapping


def apply_mapping(obj: tp.Any,
                  mapping_like: tp.Optional[tp.MappingLike] = None,
                  reverse: bool = False,
                  ignore_case: bool = True,
                  ignore_underscores: bool = True,
                  ignore_type: tp.MaybeTuple[tp.DTypeLike] = None,
                  ignore_missing: bool = False,
                  na_sentinel: tp.Any = None) -> tp.Any:
    """
    在对象上应用映射转换
    
    这是vectorbt映射系统的核心应用函数，提供了强大而灵活的映射应用机制。
    该函数能够处理各种数据类型（标量、数组、Series、DataFrame等），
    并提供多种映射选项来满足不同的业务需求。
    
    核心特性：
    1. **多类型支持**：支持标量、序列、数组、pandas对象等
    2. **智能匹配**：提供大小写不敏感和下划线忽略选项
    3. **类型过滤**：可以跳过特定数据类型的映射
    4. **容错处理**：支持缺失值处理和错误恢复
    5. **递归处理**：自动处理嵌套数据结构
    
    映射处理流程：
    1. 标准化输入映射对象
    2. 构建键处理函数（大小写、下划线等）
    3. 检查类型兼容性和过滤条件
    4. 递归应用映射到各种数据结构
    5. 处理缺失值和异常情况
    
    参数：
        obj (tp.Any): 要应用映射的对象，支持：
                     - 标量值
                     - 元组、列表、集合等序列
                     - NumPy数组
                     - pandas Series、Index、DataFrame
        mapping_like (tp.Optional[tp.MappingLike]): 映射规则，如果为None则返回原对象
        reverse (bool): 是否反转映射，传递给to_mapping函数
        ignore_case (bool): 是否忽略字符串键的大小写差异
        ignore_underscores (bool): 是否忽略字符串键中的下划线
        ignore_type (tp.MaybeTuple[tp.DTypeLike]): 要忽略的数据类型，这些类型不会被映射
        ignore_missing (bool): 是否忽略映射中不存在的键（返回原值而非抛出异常）
        na_sentinel (tp.Any): 用于标记缺失值的哨兵值
    
    返回：
        tp.Any: 应用映射后的对象，类型与输入对象相同
    
    使用示例：
        >>> import numpy as np
        >>> import pandas as pd
        
        >>> # 基本标量映射
        >>> direction_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
        >>> result = apply_mapping('BUY', direction_map)
        >>> print(result)  # 1
        
        >>> # 大小写不敏感映射
        >>> result = apply_mapping('buy', direction_map, ignore_case=True)
        >>> print(result)  # 1
        
        >>> # 列表映射
        >>> trades = ['BUY', 'SELL', 'HOLD', 'BUY']
        >>> mapped_trades = apply_mapping(trades, direction_map)
        >>> print(mapped_trades)  # [1, -1, 0, 1]
        
        >>> # pandas Series映射
        >>> orders = pd.Series(['BUY', 'SELL', 'HOLD'])
        >>> mapped_orders = apply_mapping(orders, direction_map)
        >>> print(mapped_orders.values)  # [1, -1, 0]
        
        >>> # DataFrame列映射
        >>> df = pd.DataFrame({
        ...     'action': ['BUY', 'SELL'],
        ...     'quantity': [100, 200]
        ... })
        >>> # 只映射字符串列，数值列保持不变
        >>> mapped_df = apply_mapping(df, direction_map, ignore_type=np.number)
        >>> print(mapped_df)
        
        >>> # 缺失值处理
        >>> incomplete_trades = ['BUY', 'UNKNOWN', 'SELL']
        >>> # 忽略缺失的映射键，保持原值
        >>> safe_mapping = apply_mapping(
        ...     incomplete_trades, direction_map, ignore_missing=True
        ... )
        >>> print(safe_mapping)  # [1, 'UNKNOWN', -1]
        
        >>> # 在vectorbt记录处理中的应用
        >>> # 将记录字段的数字代码转换为可读标签
        >>> status_codes = np.array([0, 1, 2, 0, 1])
        >>> status_map = {0: 'Open', 1: 'Closed', 2: 'Cancelled'}
        >>> status_labels = apply_mapping(status_codes, status_map)
        >>> print(status_labels)  # ['Open', 'Closed', 'Cancelled', 'Open', 'Closed']
        
        >>> # 复杂的嵌套数据结构映射
        >>> nested_data = [
        ...     {'action': 'BUY', 'status': 'OPEN'},
        ...     ['SELL', 'HOLD'],
        ...     pd.Series(['BUY', 'SELL'])
        ... ]
        >>> # apply_mapping会递归处理嵌套结构
    
    技术实现细节：
        - 使用lambda函数构建键处理逻辑
        - 通过类型检测实现智能分发
        - 向量化操作提高大数据处理性能
        - 完整的异常处理和错误恢复机制
    """
    # 如果没有提供映射，直接返回原对象
    if mapping_like is None:
        return obj

    # 构建键处理函数，根据配置决定如何处理字符串键
    if ignore_case and ignore_underscores:
        # 同时忽略大小写和下划线：转小写并移除下划线
        key_func = lambda x: x.lower().replace('_', '')
    elif ignore_case:
        # 只忽略大小写：转小写
        key_func = lambda x: x.lower()
    elif ignore_underscores:
        # 只忽略下划线：移除下划线
        key_func = lambda x: x.replace('_', '')
    else:
        # 不进行任何处理：保持原样
        key_func = lambda x: x
    
    # 确保ignore_type是元组格式，便于后续类型检查
    if not isinstance(ignore_type, tuple):
        ignore_type = (ignore_type,)

    # 将类映射对象转换为标准字典格式
    mapping = to_mapping(mapping_like, reverse=reverse)

    # 构建处理后的映射字典，应用键处理函数
    new_mapping = dict()
    for k, v in mapping.items():
        # 处理pandas的空值作为特殊键的情况
        if pd.isnull(k):
            na_sentinel = v  # 空键对应的值作为缺失值哨兵
        else:
            # 如果键是字符串，应用键处理函数
            if isinstance(k, str):
                k = key_func(k)
            new_mapping[k] = v

    def _compatible_types(x_type: type, item: tp.Any = None) -> bool:
        """
        检查数据类型是否在忽略类型列表中
        
        该内部函数用于判断给定的数据类型是否应该被忽略映射。
        它支持精确类型匹配和NumPy数据类型的兼容性匹配。
        
        参数：
            x_type (type): 要检查的数据类型
            item (tp.Any, optional): 可选的数据项，用于对象类型的精确判断
        
        返回：
            bool: 如果类型应该被忽略则返回True，否则返回False
        """
        # 如果提供了具体数据项且类型是对象类型，使用实际项的类型
        if item is not None:
            if np.dtype(x_type) == 'O':  # 'O'表示对象类型
                x_type = type(item)
        
        # 遍历所有要忽略的类型
        for y_type in ignore_type:
            if y_type is None:
                return False  # None类型不匹配任何类型
            
            # 精确类型匹配
            if x_type is y_type:
                return True
            
            # NumPy数据类型匹配
            x_dtype = np.dtype(x_type)
            y_dtype = np.dtype(y_type)
            if x_dtype is y_dtype:
                return True
            
            # 数值类型的子类型匹配
            if np.issubdtype(x_dtype, np.integer) and np.issubdtype(y_dtype, np.integer):
                return True  # 整数类型匹配
            if np.issubdtype(x_dtype, np.floating) and np.issubdtype(y_dtype, np.floating):
                return True  # 浮点数类型匹配
            if np.issubdtype(x_dtype, np.bool_) and np.issubdtype(y_dtype, np.bool_):
                return True  # 布尔类型匹配
            if np.issubdtype(x_dtype, np.flexible) and np.issubdtype(y_dtype, np.flexible):
                return True  # 灵活类型（字符串等）匹配
        
        return False  # 没有匹配的类型

    def _converter(x: tp.Any) -> tp.Any:
        """
        单值转换器函数
        
        该内部函数处理单个值的映射转换，包括缺失值处理、
        键处理和映射查找。
        
        参数：
            x (tp.Any): 要转换的单个值
        
        返回：
            tp.Any: 转换后的值
        """
        # 处理pandas缺失值
        if pd.isnull(x):
            return na_sentinel
        
        # 如果是字符串，应用键处理函数
        if isinstance(x, str):
            x = key_func(x)
        
        # 根据ignore_missing决定如何处理缺失的映射键
        if ignore_missing:
            try:
                return new_mapping[x]  # 尝试映射
            except KeyError:
                return x  # 映射失败时返回原值
        
        # 不忽略缺失时，直接进行映射（可能抛出KeyError）
        return new_mapping[x]

    # 根据对象类型选择合适的处理策略
    
    # 处理序列类型（元组、列表、集合等）
    if isinstance(obj, (tuple, list, set, frozenset)):
        # 递归应用映射到每个元素
        result = [apply_mapping(
            v,
            mapping_like=mapping_like,
            reverse=reverse,
            ignore_case=ignore_case,
            ignore_underscores=ignore_underscores,
            ignore_type=ignore_type,
            ignore_missing=ignore_missing,
            na_sentinel=na_sentinel
        ) for v in obj]
        # 保持原始容器类型
        return type(obj)(result)
    
    # 处理NumPy数组
    if isinstance(obj, np.ndarray):
        # 空数组直接返回
        if obj.size == 0:
            return obj
        
        # 检查是否应该忽略此数组的数据类型
        if ignore_type is None or not _compatible_types(obj.dtype, obj.item(0)):
            if obj.ndim == 1:
                # 一维数组：转换为Series进行映射，然后提取值
                return pd.Series(obj).map(_converter).values
            # 多维数组：使用向量化函数进行映射
            return np.vectorize(_converter)(obj)
        # 类型被忽略，返回原数组
        return obj
    
    # 处理pandas Series
    if isinstance(obj, pd.Series):
        # 空Series直接返回
        if obj.size == 0:
            return obj
        
        # 检查是否应该忽略此Series的数据类型
        if ignore_type is None or not _compatible_types(obj.dtype, obj.iloc[0]):
            # 使用pandas的map方法进行高效映射
            return obj.map(_converter)
        # 类型被忽略，返回原Series
        return obj
    
    # 处理pandas Index
    if isinstance(obj, pd.Index):
        # 空Index直接返回
        if obj.size == 0:
            return obj
        
        # 检查是否应该忽略此Index的数据类型
        if ignore_type is None or not _compatible_types(obj.dtype, obj[0]):
            # 使用Index的map方法进行映射
            return obj.map(_converter)
        # 类型被忽略，返回原Index
        return obj
    
    # 处理pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        # 空DataFrame直接返回
        if obj.size == 0:
            return obj
        
        # 对每一列分别处理
        series = []
        for sr_name, sr in obj.items():
            # 检查是否应该忽略此列的数据类型
            if ignore_type is None or not _compatible_types(sr.dtype, sr.iloc[0]):
                # 应用映射到此列
                series.append(sr.map(_converter))
            else:
                # 类型被忽略，保持原列
                series.append(sr)
        
        # 重新组装DataFrame，保持原始列名
        return pd.concat(series, axis=1, keys=obj.columns)
    
    # 处理标量值（其他所有类型）
    if ignore_type is None or not _compatible_types(type(obj)):
        # 应用转换器到标量值
        return _converter(obj)
    
    # 类型被忽略，返回原对象
    return obj
