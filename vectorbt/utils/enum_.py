# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT UTILS MODULE: 枚举映射和转换工具
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于处理枚举映射和转换的核心工具模块。
在vectorbt中，枚举由命名元组的实例表示，以便在Numba编译环境中高效使用。
该模块提供了枚举字段名和数值之间的双向映射转换功能，是框架类型安全和
用户友好接口的重要基础设施。

核心设计理念：
1. **Numba兼容性优先**：枚举使用命名元组实现，确保在Numba JIT编译中的高性能
2. **双向映射支持**：支持字段名到值和值到字段名的双向转换
3. **类型安全保障**：通过类型检查确保映射的准确性和一致性
4. **用户友好接口**：允许用户使用直观的字符串名称而非抽象的数字代码

枚举设计约定：
- 枚举值从0开始按顺序递增：0, 1, 2, 3...
- 特殊值-1表示"无值"或"未定义"状态，对应None
- 字段名通常使用帕斯卡命名法（PascalCase）
- 枚举定义为命名元组，便于内存对齐和快速访问

主要功能模块：
- **字段到值映射**：map_enum_fields函数，将用户友好的字段名转换为内部数值
- **值到字段映射**：map_enum_values函数，将内部数值转换为可读的字段名
- **类型安全检查**：自动验证输入类型，确保映射的正确性
- **灵活数据支持**：支持标量、列表、NumPy数组、pandas Series等多种数据类型

应用场景：
- **交易方向映射**：将'BUY'、'SELL'字符串转换为0、1数值，便于数值计算
- **订单状态转换**：在用户界面显示'FILLED'、'REJECTED'等可读状态
- **枚举参数处理**：在函数调用中支持字符串参数，内部转换为数值处理
- **数据序列化**：在数据存储和传输中进行枚举的编码和解码
- **配置文件处理**：允许配置文件使用字符串枚举，运行时转换为数值

技术特点：
- 基于vectorbt.utils.mapping模块的to_mapping和apply_mapping函数
- 支持大小写不敏感的字段名匹配（通过apply_mapping的选项）
- 自动处理缺失值和无效输入的情况
- 与vectorbt的类型系统完全兼容
- 支持向量化操作，可高效处理大型数据集

与vectorbt生态系统的关系：
- 为Portfolio模块提供订单状态、交易方向等枚举的映射支持
- 为Records模块提供状态字段的标签化显示功能
- 为用户API提供友好的字符串接口，隐藏内部数值实现
- 为配置系统提供枚举参数的验证和转换功能
- 支持Numba编译函数中的高效枚举处理

使用示例：
    >>> import vectorbt as vbt
    >>> from collections import namedtuple
    
    >>> # 定义交易方向枚举
    >>> Direction = namedtuple('Direction', ['BUY', 'SELL', 'HOLD'])
    >>> direction_enum = Direction(0, 1, 2)
    
    >>> # 字段名到值的映射
    >>> signals = ['BUY', 'SELL', 'BUY', 'HOLD']
    >>> numeric_signals = vbt.utils.enum_.map_enum_fields(signals, direction_enum)
    >>> print(numeric_signals)  # [0, 1, 0, 2]
    
    >>> # 值到字段名的映射
    >>> status_values = [0, 1, 0, 2]
    >>> readable_status = vbt.utils.enum_.map_enum_values(status_values, direction_enum)
    >>> print(readable_status)  # ['BUY', 'SELL', 'BUY', 'HOLD']
    
    >>> # 在投资组合分析中的应用
    >>> orders = pd.DataFrame({
    ...     'direction': ['BUY', 'SELL', 'BUY'],
    ...     'size': [100, 200, 150]
    ... })
    >>> orders['direction_code'] = vbt.utils.enum_.map_enum_fields(
    ...     orders['direction'], direction_enum
    ... )

该模块为vectorbt框架的枚举系统提供了完整的映射转换基础设施，
确保了用户接口的友好性和内部处理的高效性。
"""

# 导入vectorbt的类型定义模块，提供类型注解支持
from vectorbt import _typing as tp
# 导入vectorbt的映射工具模块，提供核心的映射转换功能
from vectorbt.utils.mapping import to_mapping, apply_mapping


def map_enum_fields(field: tp.Any, enum: tp.Enum, ignore_type=int, **kwargs) -> tp.Any:
    """
    将枚举字段名映射为对应的数值，是用户友好接口转内部表示的核心函数
    
    该函数是vectorbt枚举系统中最重要的转换工具之一，用于将用户友好的
    字符串字段名转换为内部使用的整数值。这种转换机制使得用户可以使用
    直观的名称（如'BUY'、'SELL'）进行操作，而系统内部使用高效的
    数值进行计算和存储。
    
    核心工作流程：
    1. 使用to_mapping创建字段名到数值的反向映射（reverse=True）
    2. 使用apply_mapping将输入的字段名批量转换为对应数值
    3. 保持数据结构的形状和类型（标量、数组、Series等）
    4. 应用类型检查和错误处理，确保转换的正确性
    
    参数：
        field (tp.Any): 要转换的字段名，支持多种数据类型：
                       - 单个字符串：'BUY'、'SELL'等枚举字段名
                       - 字符串列表：['BUY', 'SELL', 'HOLD']
                       - NumPy字符串数组：np.array(['BUY', 'SELL'])
                       - pandas Series：包含枚举字段名的Series
                       - 已有数值：如果输入已经是数值，直接返回
                       
        enum (tp.Enum): 枚举对象，通常是命名元组实例
                       例如：Direction(BUY=0, SELL=1, HOLD=2)
                       
        ignore_type (type, optional): 忽略类型检查的数据类型，默认为int
                                    对于已经是指定类型的数据，直接返回而不进行映射
                                    
        **kwargs: 传递给apply_mapping函数的额外参数
                 常用选项：
                 - case_sensitive: 是否大小写敏感，默认False
                 - ignore_underscore: 是否忽略下划线，默认True
    
    返回：
        tp.Any: 转换后的数值，保持输入数据的结构和形状
    
    使用示例：
        >>> from collections import namedtuple
        >>> import numpy as np
        >>> import pandas as pd
        >>> 
        >>> # 定义交易方向枚举
        >>> Direction = namedtuple('Direction', ['BUY', 'SELL', 'HOLD'])
        >>> direction_enum = Direction(0, 1, 2)
        >>> 
        >>> # 基本用法：单个字段映射
        >>> signal = 'BUY'
        >>> numeric_signal = map_enum_fields(signal, direction_enum)
        >>> print(numeric_signal)  # 0
        >>> 
        >>> # 列表映射
        >>> signals = ['BUY', 'SELL', 'BUY', 'HOLD']
        >>> numeric_signals = map_enum_fields(signals, direction_enum)
        >>> print(numeric_signals)  # [0, 1, 0, 2]
        >>> 
        >>> # NumPy数组映射
        >>> signal_array = np.array(['BUY', 'SELL', 'HOLD'])
        >>> numeric_array = map_enum_fields(signal_array, direction_enum)
        >>> print(numeric_array)  # array([0, 1, 2])
        >>> 
        >>> # pandas Series映射
        >>> signal_series = pd.Series(['BUY', 'SELL', 'BUY'])
        >>> numeric_series = map_enum_fields(signal_series, direction_enum)
        >>> print(numeric_series)  # Series([0, 1, 0])
        >>> 
        >>> # 已有数值的处理（直接返回）
        >>> existing_numeric = 1
        >>> result = map_enum_fields(existing_numeric, direction_enum)
        >>> print(result)  # 1（直接返回，不进行转换）
        
        >>> # 在量化交易中的应用
        >>> # 订单方向处理
        >>> OrderSide = namedtuple('OrderSide', ['BUY', 'SELL'])
        >>> order_side_enum = OrderSide(0, 1)
        >>> 
        >>> order_signals = pd.DataFrame({
        ...     'symbol': ['AAPL', 'GOOGL', 'MSFT'],
        ...     'side': ['BUY', 'SELL', 'BUY'],
        ...     'quantity': [100, 200, 150]
        ... })
        >>> 
        >>> # 将用户友好的方向转换为数值，用于后续计算
        >>> order_signals['side_code'] = map_enum_fields(
        ...     order_signals['side'], order_side_enum
        ... )
        >>> print(order_signals)
        #   symbol side  quantity  side_code
        # 0   AAPL  BUY       100          0
        # 1  GOOGL SELL       200          1
        # 2   MSFT  BUY       150          0
        
        >>> # 订单状态处理
        >>> OrderStatus = namedtuple('OrderStatus', ['PENDING', 'FILLED', 'REJECTED'])
        >>> status_enum = OrderStatus(0, 1, 2)
        >>> 
        >>> status_updates = ['PENDING', 'FILLED', 'PENDING', 'REJECTED']
        >>> status_codes = map_enum_fields(status_updates, status_enum)
        >>> print(status_codes)  # [0, 1, 0, 2]
        
        >>> # 大小写不敏感的映射
        >>> mixed_case_signals = ['buy', 'SELL', 'Buy']
        >>> numeric_mixed = map_enum_fields(
        ...     mixed_case_signals, direction_enum, case_sensitive=False
        ... )
        >>> print(numeric_mixed)  # [0, 1, 0]
        
        >>> # 在策略配置中的应用
        >>> def create_strategy_config(direction, stop_type, size):
        ...     StopType = namedtuple('StopType', ['STOP_LOSS', 'TAKE_PROFIT'])
        ...     stop_enum = StopType(0, 1)
        ...     
        ...     return {
        ...         'direction_code': map_enum_fields(direction, direction_enum),
        ...         'stop_type_code': map_enum_fields(stop_type, stop_enum),
        ...         'size': size
        ...     }
        >>> 
        >>> config = create_strategy_config('BUY', 'STOP_LOSS', 1000)
        >>> print(config)  # {'direction_code': 0, 'stop_type_code': 0, 'size': 1000}
    
    技术实现细节：
        - 使用to_mapping(enum, reverse=True)创建字段名到数值的映射
        - 通过apply_mapping执行实际的映射转换
        - ignore_type参数允许已有数值直接通过，避免不必要的转换
        - 支持apply_mapping的所有高级选项，如大小写处理等
    
    错误处理：
        - 无效字段名：抛出KeyError或ValueError
        - 类型不匹配：通过ignore_type参数控制处理策略
        - 空值处理：根据映射规则处理None值
    
    性能考虑：
        - 映射字典的创建是一次性开销，后续转换高效
        - 支持向量化操作，可高效处理大型数据集
        - 在Numba编译环境中保持高性能
    
    参见：
        - apply_mapping: 底层映射应用函数
        - map_enum_values: 反向映射函数（数值到字段名）
        - to_mapping: 映射对象创建函数
    """
    # 创建字段名到数值的映射字典（reverse=True表示反向映射）
    mapping = to_mapping(enum, reverse=True)

    # 应用映射转换，将字段名转换为对应的数值
    return apply_mapping(field, mapping, ignore_type=ignore_type, **kwargs)


def map_enum_values(value: tp.Any, enum: tp.Enum, ignore_type=str, **kwargs) -> tp.Any:
    """
    将枚举数值映射为对应的字段名，是内部表示转用户友好接口的核心函数
    
    该函数是map_enum_fields的反向操作，用于将系统内部使用的整数值
    转换为用户友好的字符串字段名。这种转换在数据展示、日志记录、
    用户界面显示等场景中非常重要，使得枚举值以可读的形式呈现。
    
    核心工作流程：
    1. 使用to_mapping创建数值到字段名的正向映射（reverse=False）
    2. 使用apply_mapping将输入的数值批量转换为对应字段名
    3. 保持数据结构的形状和类型（标量、数组、Series等）
    4. 应用类型检查和错误处理，确保转换的正确性
    
    参数：
        value (tp.Any): 要转换的枚举值，支持多种数据类型：
                       - 单个整数：0、1、2等枚举数值
                       - 整数列表：[0, 1, 0, 2]
                       - NumPy整数数组：np.array([0, 1, 2])
                       - pandas Series：包含枚举数值的Series
                       - 已有字符串：如果输入已经是字符串，直接返回
                       
        enum (tp.Enum): 枚举对象，通常是命名元组实例
                       例如：Direction(BUY=0, SELL=1, HOLD=2)
                       
        ignore_type (type, optional): 忽略类型检查的数据类型，默认为str
                                    对于已经是指定类型的数据，直接返回而不进行映射
                                    
        **kwargs: 传递给apply_mapping函数的额外参数
                 通常用于控制映射行为的高级选项
    
    返回：
        tp.Any: 转换后的字段名，保持输入数据的结构和形状
    
    使用示例：
        >>> from collections import namedtuple
        >>> import numpy as np
        >>> import pandas as pd
        >>> 
        >>> # 定义交易方向枚举
        >>> Direction = namedtuple('Direction', ['BUY', 'SELL', 'HOLD'])
        >>> direction_enum = Direction(0, 1, 2)
        >>> 
        >>> # 基本用法：单个数值映射
        >>> code = 0
        >>> field_name = map_enum_values(code, direction_enum)
        >>> print(field_name)  # 'BUY'
        >>> 
        >>> # 列表映射
        >>> codes = [0, 1, 0, 2]
        >>> field_names = map_enum_values(codes, direction_enum)
        >>> print(field_names)  # ['BUY', 'SELL', 'BUY', 'HOLD']
        >>> 
        >>> # NumPy数组映射
        >>> code_array = np.array([0, 1, 2])
        >>> name_array = map_enum_values(code_array, direction_enum)
        >>> print(name_array)  # array(['BUY', 'SELL', 'HOLD'])
        >>> 
        >>> # pandas Series映射
        >>> code_series = pd.Series([0, 1, 0])
        >>> name_series = map_enum_values(code_series, direction_enum)
        >>> print(name_series)  # Series(['BUY', 'SELL', 'BUY'])
        >>> 
        >>> # 已有字符串的处理（直接返回）
        >>> existing_string = 'BUY'
        >>> result = map_enum_values(existing_string, direction_enum)
        >>> print(result)  # 'BUY'（直接返回，不进行转换）
        
        >>> # 在数据分析和可视化中的应用
        >>> # 订单记录的可读化展示
        >>> OrderStatus = namedtuple('OrderStatus', ['PENDING', 'FILLED', 'REJECTED'])
        >>> status_enum = OrderStatus(0, 1, 2)
        >>> 
        >>> order_records = pd.DataFrame({
        ...     'order_id': [1001, 1002, 1003, 1004],
        ...     'status_code': [0, 1, 2, 1],
        ...     'quantity': [100, 200, 150, 300]
        ... })
        >>> 
        >>> # 将数值状态转换为可读字符串，便于报告展示
        >>> order_records['status'] = map_enum_values(
        ...     order_records['status_code'], status_enum
        ... )
        >>> print(order_records)
        #   order_id  status_code  quantity    status
        # 0     1001            0       100   PENDING
        # 1     1002            1       200    FILLED
        # 2     1003            2       150  REJECTED
        # 3     1004            1       300    FILLED
        
        >>> # 交易方向的可视化标签
        >>> trades = pd.DataFrame({
        ...     'timestamp': pd.date_range('2023-01-01', periods=4, freq='H'),
        ...     'direction_code': [0, 1, 0, 1],
        ...     'price': [100.5, 101.2, 99.8, 102.1]
        ... })
        >>> 
        >>> trades['direction'] = map_enum_values(
        ...     trades['direction_code'], direction_enum
        ... )
        >>> 
        >>> # 用于图表标签
        >>> import matplotlib.pyplot as plt
        >>> colors = {'BUY': 'green', 'SELL': 'red'}
        >>> trades['color'] = trades['direction'].map(colors)
        
        >>> # 日志记录和调试
        >>> def log_order_status(order_id, status_code):
        ...     status_name = map_enum_values(status_code, status_enum)
        ...     print(f"订单 {order_id} 状态更新为: {status_name}")
        >>> 
        >>> log_order_status(1001, 1)  # 输出: 订单 1001 状态更新为: FILLED
        
        >>> # 配置文件的反向解析
        >>> def parse_strategy_results(results):
        ...     # results包含数值编码的策略配置
        ...     readable_results = results.copy()
        ...     readable_results['direction'] = map_enum_values(
        ...         results['direction_code'], direction_enum
        ...     )
        ...     return readable_results
        
        >>> # API响应的友好化
        >>> def format_api_response(raw_data):
        ...     formatted = {}
        ...     for key, value in raw_data.items():
        ...         if key.endswith('_code') and isinstance(value, int):
        ...             # 自动将编码字段转换为可读字段
        ...             readable_key = key.replace('_code', '')
        ...             formatted[readable_key] = map_enum_values(value, direction_enum)
        ...         else:
        ...             formatted[key] = value
        ...     return formatted
        
        >>> # 特殊值处理：-1对应None
        >>> special_codes = [0, 1, -1, 2]
        >>> special_names = map_enum_values(special_codes, direction_enum)
        >>> print(special_names)  # ['BUY', 'SELL', None, 'HOLD']
        
        >>> # 在统计分析中的应用
        >>> def analyze_trade_patterns(trade_data):
        ...     # 将数值代码转换为可读标签进行分析
        ...     trade_data['readable_direction'] = map_enum_values(
        ...         trade_data['direction_code'], direction_enum
        ...     )
        ...     
        ...     # 按方向分组统计
        ...     direction_stats = trade_data.groupby('readable_direction').agg({
        ...         'profit': ['mean', 'sum', 'count'],
        ...         'quantity': 'mean'
        ...     })
        ...     
        ...     return direction_stats
    
    应用场景：
        - **数据可视化**：在图表中显示可读的枚举标签
        - **报告生成**：将内部数值转换为业务友好的文本
        - **日志记录**：在日志中输出可读的状态信息
        - **API响应**：为客户端提供可理解的响应数据
        - **数据导出**：将分析结果导出为可读格式
        - **调试和开发**：在开发过程中检查和验证数据状态
    
    技术实现细节：
        - 使用to_mapping(enum, reverse=False)创建数值到字段名的映射
        - 通过apply_mapping执行实际的映射转换
        - ignore_type参数允许已有字符串直接通过，避免不必要的转换
        - 自动处理特殊值-1到None的映射
    
    错误处理：
        - 无效数值：抛出KeyError或ValueError
        - 类型不匹配：通过ignore_type参数控制处理策略
        - 超出范围的值：根据映射规则处理未定义的数值
    
    性能考虑：
        - 映射字典的创建是一次性开销，后续转换高效
        - 支持向量化操作，可高效处理大型数据集
        - 在数据可视化和报告生成中提供良好性能
    
    参见：
        - apply_mapping: 底层映射应用函数
        - map_enum_fields: 反向映射函数（字段名到数值）
        - to_mapping: 映射对象创建函数
    """
    # 创建数值到字段名的映射字典（reverse=False表示正向映射）
    mapping = to_mapping(enum, reverse=False)

    # 应用映射转换，将数值转换为对应的字段名
    return apply_mapping(value, mapping, ignore_type=ignore_type, **kwargs)
