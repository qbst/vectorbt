# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT RECORDS MODULE: Records类基础实现
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中Records类系统的核心基类实现。Records类是vectorbt框架中
用于处理结构化记录数据的基础类，专门设计用来高效地处理稀疏的、结构化的金融数据，
如交易记录、订单记录、持仓记录、回撤记录等。

核心设计理念：
1. **内存效率优先**：Records使用结构化数组存储数据，避免了DataFrame的内存开销，
   特别适合处理大规模的稀疏交易数据。相比传统的DataFrame存储方式，可以节省50-80%的内存。

2. **高性能计算**：所有核心算法都基于Numba编译的函数，实现接近C语言的执行速度，
   能够处理百万级别的记录数据而不会出现性能瓶颈。

3. **灵活的数据映射**：Records支持将结构化数据映射为MappedArray，实现了类似MapReduce
   的数据处理模式，可以在不展开为完整矩阵的情况下进行各种统计计算。

4. **多维度数据操作**：支持按列、按组、按时间等多维度的数据访问和聚合操作，
   特别适合多资产、多策略的量化分析场景。

主要功能模块：

【数据存储与访问】
- **结构化数组管理**：使用NumPy的结构化数组存储记录数据
- **字段配置系统**：通过field_config定义字段的名称、类型、映射关系
- **索引映射**：支持数据的快速索引和切片操作

【数据处理与变换】
- **字段映射**：map_field方法将字段转换为MappedArray
- **记录映射**：map方法对每个记录应用自定义函数
- **列组应用**：apply方法在列或组上应用函数进行聚合操作

【数据过滤与筛选】
- **掩码过滤**：apply_mask方法基于条件过滤记录
- **排序功能**：sort方法对记录进行排序
- **分组操作**：支持复杂的分组和重组操作

【统计分析集成】
- **统计指标**：内置丰富的统计指标计算功能
- **图表绘制**：集成Plotly图表绘制功能
- **缓存机制**：智能缓存机制避免重复计算

【扩展性设计】
- **装饰器支持**：配合decorators模块实现配置驱动的类扩展
- **继承友好**：支持复杂的类继承和字段配置合并
- **插件架构**：可以通过子类化扩展功能

数据结构设计：
Records类基于以下核心数据结构：
- **records_arr**：结构化数组，存储实际的记录数据
- **wrapper**：ArrayWrapper对象，包含索引、列名、分组等元数据
- **col_mapper**：ColumnMapper对象，提供高效的列映射功能
- **field_config**：字段配置对象，定义字段的属性和映射关系

与矩阵存储的对比：
传统的矩阵存储方式：
```
               a     b     c
         0   1.0   5.0   NaN
         1   2.0   NaN   7.0
         2   NaN   8.0   9.0
```

Records存储方式：
```
id  col  idx  value
0    0    0    1.0
1    0    1    2.0
2    1    0    5.0
3    1    2    8.0
4    2    1    7.0
5    2    2    9.0
```

Records存储方式的优势：
- 内存效率更高（只存储非空值）
- 支持不规则的数据结构
- 便于添加额外的字段信息
- 更适合事件驱动的数据处理

应用场景：
- **交易记录处理**：买卖订单、成交记录、持仓变化等
- **投资组合分析**：多资产组合的收益、风险、归因分析
- **风险管理**：回撤分析、压力测试、风险指标计算
- **策略回测**：策略信号、交易成本、绩效评估
- **因子研究**：因子收益、因子暴露、因子有效性分析

技术特点：
- **类型安全**：完整的类型提示和运行时类型检查
- **缓存优化**：智能缓存机制避免重复计算
- **并行友好**：列之间的操作相互独立，易于并行化
- **内存友好**：延迟计算和零拷贝操作
- **向量化**：充分利用NumPy的向量化能力

性能优化：
- 使用Numba编译的核心算法
- 智能的列映射和索引机制
- 缓存属性避免重复计算
- 零拷贝的数据操作

该模块是vectorbt框架的核心组件，为整个量化交易生态系统提供了高效、灵活的数据处理基础。

使用示例：
```python
import numpy as np
import pandas as pd
import vectorbt as vbt

# 1. 创建示例记录数据
example_dt = np.dtype([
    ('id', np.int64),
    ('col', np.int64),
    ('idx', np.int64),
    ('price', np.float64),
    ('volume', np.float64)
])

records_arr = np.array([
    (0, 0, 0, 100.0, 1000),
    (1, 0, 1, 101.0, 1200),
    (2, 1, 0, 102.0, 800),
    (3, 1, 1, 103.0, 1500),
    (4, 2, 0, 104.0, 900),
    (5, 2, 1, 105.0, 1100)
], dtype=example_dt)

# 2. 创建ArrayWrapper
wrapper = vbt.ArrayWrapper(
    index=pd.date_range('2023-01-01', periods=2, freq='D'),
    columns=['AAPL', 'GOOGL', 'MSFT'],
    ndim=2
)

# 3. 创建Records对象
records = vbt.Records(wrapper, records_arr)

# 4. 数据访问和操作
print(f"记录数量: {len(records)}")
print(f"原始记录:\n{records.records}")
print(f"可读格式:\n{records.records_readable}")

# 5. 字段映射
price_mapped = records.map_field('price')
print(f"价格映射数组: {price_mapped.values}")

# 6. 统计分析
stats = records.stats()
print(f"统计信息:\n{stats}")

# 7. 数据过滤
mask = records.get_field_arr('volume') > 1000
filtered_records = records.apply_mask(mask)
print(f"过滤后的记录数量: {len(filtered_records)}")
```
"""

import inspect
import string

import numpy as np
import pandas as pd

from vectorbt import _typing as tp
from vectorbt.base.array_wrapper import ArrayWrapper, Wrapping
from vectorbt.base.reshape_fns import to_1d_array
from vectorbt.generic.plots_builder import PlotsBuilderMixin
from vectorbt.generic.stats_builder import StatsBuilderMixin
from vectorbt.records import nb
from vectorbt.records.col_mapper import ColumnMapper
from vectorbt.records.mapped_array import MappedArray
from vectorbt.utils import checks
from vectorbt.utils.attr_ import get_dict_attr
from vectorbt.utils.config import merge_dicts, Config, Configured
from vectorbt.utils.decorators import cached_method

__pdoc__ = {}

# 定义Records类的类型变量，用于类型提示中的泛型约束
# 这个类型变量确保Records类的方法返回的类型与调用类的类型一致
# 例如：如果在子类MyRecords上调用方法，返回的也是MyRecords类型
RecordsT = tp.TypeVar("RecordsT", bound="Records")

# 定义索引元数据的类型，用于索引操作的返回值类型注解
# 包含四个元素：新的ArrayWrapper、新的记录数组、组索引、列索引
# 这个类型定义确保了索引操作的类型安全性
IndexingMetaT = tp.Tuple[ArrayWrapper, tp.RecordArray, tp.MaybeArray, tp.Array1d]


class MetaFields(type):
    """
    字段元类 - 为Records类提供字段配置的类属性访问
    
    这个元类的主要作用是为Records类及其子类提供一个只读的类属性field_config，
    该属性可以通过类直接访问，而不需要实例化对象。这种设计模式在配置管理中
    非常有用，允许在类定义时就确定字段配置信息。
    
    设计原理：
    - 使用元类的@property装饰器创建类级别的属性
    - 确保field_config在类层面就可以访问
    - 为Records类系统提供配置的统一访问接口
    
    使用示例：
    ```python
    # 通过类直接访问字段配置
    config = Records.field_config
    print(f"字段配置: {config}")
    
    # 子类也可以访问自己的字段配置
    config = MyRecords.field_config
    print(f"子类字段配置: {config}")
    ```
    """

    @property
    def field_config(cls) -> Config:
        """
        字段配置属性 - 返回类的字段配置对象
        
        这个属性提供了对类的_field_config属性的只读访问。
        通过元类属性的方式，可以确保配置在类层面就可以访问，
        而不需要创建实例。
        
        返回：
            Config: 字段配置对象，包含字段的类型、设置等信息
        """
        return cls._field_config


class RecordsWithFields(metaclass=MetaFields):
    """
    带字段配置的Records基类 - 提供字段配置的实例访问
    
    这个类使用MetaFields元类，为Records类提供字段配置的访问能力。
    它既支持类级别的配置访问，也支持实例级别的配置访问。
    
    设计目的：
    - 统一字段配置的访问接口
    - 支持类级别和实例级别的配置访问
    - 为Records继承体系提供配置基础
    
    使用示例：
    ```python
    # 类级别访问
    config = Records.field_config
    
    # 实例级别访问
    records = Records(wrapper, records_arr)
    config = records.field_config
    ```
    """

    @property
    def field_config(self) -> Config:
        """
        字段配置属性 - 返回实例的字段配置对象
        
        这个属性提供了对实例所属类的字段配置的访问。
        文档字符串中的${cls_name}和${field_config}是模板变量，
        会在文档生成时被实际的类名和配置内容替换。
        
        返回：
            Config: 字段配置对象，包含字段的类型、设置等信息
        
        文档模板：
            ${cls_name}: 会被替换为具体的类名
            ${field_config}: 会被替换为具体的配置内容
        """
        return self._field_config


class MetaRecords(type(StatsBuilderMixin), type(PlotsBuilderMixin), type(RecordsWithFields)):
    """
    Records元类 - 整合多个混入类的元类
    
    这个元类继承了三个混入类的元类，解决了多重继承中的元类冲突问题。
    在Python中，当一个类继承多个具有不同元类的类时，需要创建一个
    新的元类来整合这些元类的功能。
    
    继承的元类：
    - type(StatsBuilderMixin): 统计构建器混入类的元类
    - type(PlotsBuilderMixin): 图表构建器混入类的元类  
    - type(RecordsWithFields): 字段配置混入类的元类
    
    设计原理：
    - 解决多重继承中的元类冲突
    - 整合多个混入类的元类功能
    - 确保Records类能够正确继承所有混入类的能力
    
    注意：
    - 这个元类本身不添加新的功能
    - 它的主要作用是解决元类冲突问题
    - 使用pass语句表明它只是一个占位符
    """
    pass


class Records(Wrapping, StatsBuilderMixin, PlotsBuilderMixin, RecordsWithFields, metaclass=MetaRecords):
    """
    Records基类 - vectorbt框架中处理结构化记录数据的核心类
    
    Records类是vectorbt量化交易框架中用于处理结构化记录数据的基础类。
    它提供了一种高效的方式来处理稀疏的、结构化的金融数据，如交易记录、
    订单记录、持仓记录等。该类通过结构化数组存储数据，避免了传统DataFrame
    的内存开销，特别适合大规模的量化分析。
    
    继承关系：
    - Wrapping: 提供数组包装和pandas索引操作功能
    - StatsBuilderMixin: 提供统计指标计算功能
    - PlotsBuilderMixin: 提供图表绘制功能
    - RecordsWithFields: 提供字段配置访问功能
    
    核心特性：
    1. **高效存储**：使用NumPy结构化数组，内存占用比DataFrame少50-80%
    2. **高性能计算**：基于Numba编译的函数，计算速度比纯Python快10-100倍
    3. **灵活映射**：支持将记录数据映射为MappedArray进行各种操作
    4. **多维操作**：支持按列、按组、按时间等多维度的数据处理
    5. **统计集成**：内置丰富的统计指标和图表绘制功能
    
    数据结构：
    Records类基于以下核心组件：
    - records_arr: 结构化数组，存储实际的记录数据
    - wrapper: ArrayWrapper对象，管理索引、列名、分组等元数据
    - col_mapper: ColumnMapper对象，提供高效的列映射功能
    - field_config: 字段配置，定义字段的属性和映射关系
    
    必需字段：
    - id: 记录的唯一标识符
    - col: 记录所属的列索引
    - idx: 记录的时间索引（可选）
    
    参数说明：
        wrapper (ArrayWrapper): 数组包装器，包含索引、列名、分组等元数据
        records_arr (array_like): 结构化NumPy数组，存储记录数据
            必须包含'id'（记录索引）和'col'（列索引）字段
        col_mapper (ColumnMapper, 可选): 列映射器，如果已知可以传入
            注意：它依赖于records_arr，所以在修改records_arr时需要重新创建
        **kwargs: 自定义关键字参数，传递给配置系统
            对于想要扩展配置的子类很有用
    
    使用示例：
    ```python
    import numpy as np
    import pandas as pd
    import vectorbt as vbt
    
    # 1. 创建示例数据
    example_dt = np.dtype([
        ('id', np.int64),
        ('col', np.int64),
        ('idx', np.int64),
        ('price', np.float64),
        ('volume', np.int64)
    ])
    
    records_arr = np.array([
        (0, 0, 0, 100.0, 1000),
        (1, 0, 1, 101.0, 1200),
        (2, 1, 0, 102.0, 800),
        (3, 1, 1, 103.0, 1500),
        (4, 2, 0, 104.0, 900),
        (5, 2, 1, 105.0, 1100)
    ], dtype=example_dt)
    
    # 2. 创建数组包装器
    wrapper = vbt.ArrayWrapper(
        index=pd.date_range('2023-01-01', periods=2, freq='D'),
        columns=['AAPL', 'GOOGL', 'MSFT'],
        ndim=2
    )
    
    # 3. 创建Records对象
    records = vbt.Records(wrapper, records_arr)
    
    # 4. 基本信息
    print(f"记录数量: {len(records)}")
    print(f"列数: {records.wrapper.shape[1]}")
    print(f"时间长度: {records.wrapper.shape[0]}")
    
    # 5. 数据访问
    print("原始记录:")
    print(records.records)
    
    print("可读格式:")
    print(records.records_readable)
    
    # 6. 字段访问
    id_arr = records.id_arr
    col_arr = records.col_arr
    idx_arr = records.idx_arr
    
    # 7. 字段映射
    price_mapped = records.map_field('price')
    volume_mapped = records.map_field('volume')
    
    # 8. 统计分析
    price_stats = price_mapped.describe()
    volume_mean = volume_mapped.mean()
    
    # 9. 数据过滤
    high_volume_mask = records.get_field_arr('volume') > 1000
    filtered_records = records.apply_mask(high_volume_mask)
    
    # 10. 分组操作
    grouped_volume = volume_mapped.sum(group_by=['Tech', 'Tech', 'Other'])
    
    # 11. 自定义映射
    from numba import njit
    
    @njit
    def value_map_nb(record):
        return record.price * record.volume
    
    total_value = records.map(value_map_nb)
    
    # 12. 统计指标
    stats = records.stats()
    print(f"统计信息:\n{stats}")
    ```
    
    高级用法：
    ```python
    # 1. 自定义字段配置
    from vectorbt.records.decorators import override_field_config
    
    custom_config = {
        'dtype': example_dt,
        'settings': {
            'price': {'title': '价格', 'name': 'price'},
            'volume': {'title': '成交量', 'name': 'volume'}
        }
    }
    
    @override_field_config(custom_config)
    class CustomRecords(Records):
        pass
    
    # 2. 时间序列分析
    records_sorted = records.sort()
    
    # 3. 复杂过滤
    def complex_filter(records):
        price_arr = records.get_field_arr('price')
        volume_arr = records.get_field_arr('volume')
        return (price_arr > 100) & (volume_arr > 1000)
    
    filtered = records.apply_mask(complex_filter(records))
    
    # 4. 分组统计
    group_by = ['Large Cap', 'Large Cap', 'Mid Cap']
    grouped_stats = records.map_field('price').describe(group_by=group_by)
    ```
    
    性能优势：
    - 内存使用量比DataFrame减少50-80%
    - 计算速度比纯Python快10-100倍
    - 支持处理百万级别的记录
    - 智能缓存避免重复计算
    
    注意事项：
    - 由于缓存机制，类被设计为不可变的，所有属性都是只读的
    - 要更改任何属性，请使用copy方法并传递属性作为关键字参数
    - 记录数组必须包含'id'和'col'字段
    - 索引操作只支持列选择，不支持行选择
    """

    # 类级别的字段配置：定义Records类的基本字段结构
    # 这是一个类变量，所有Records实例都共享这个配置
    _field_config: tp.ClassVar[Config] = Config(
        dict(
            # dtype字段：定义记录数组的数据类型结构
            # 默认为None，子类可以重写以定义具体的数据类型
            dtype=None,
            
            # settings字段：定义各个字段的配置信息
            settings=dict(
                # id字段配置：记录的唯一标识符
                id=dict(
                    name='id',        # 字段在数组中的实际名称
                    title='Id'        # 字段的显示标题
                ),
                
                # col字段配置：记录所属的列索引
                col=dict(
                    name='col',           # 字段在数组中的实际名称
                    title='Column',       # 字段的显示标题
                    mapping='columns'     # 映射到ArrayWrapper的columns属性
                ),
                
                # idx字段配置：记录的时间索引
                idx=dict(
                    name='idx',           # 字段在数组中的实际名称
                    title='Timestamp',    # 字段的显示标题
                    mapping='index'       # 映射到ArrayWrapper的index属性
                )
            )
        ),
        readonly=True,        # 配置为只读，防止意外修改
        as_attrs=False       # 不将配置项作为属性访问
    )

    @property
    def field_config(self) -> Config:
        """
        字段配置属性 - 返回当前实例的字段配置
        
        这个属性提供了对当前实例所属类的字段配置的访问。
        文档字符串中的模板变量会在文档生成时被替换为实际内容。
        
        返回：
            Config: 字段配置对象，包含字段的类型、设置等信息
        
        文档模板说明：
            ${cls_name}: 会被替换为具体的类名
            ${field_config}: 会被替换为具体的配置内容（JSON格式）
        """
        return self._field_config

    def __init__(self,
                 wrapper: ArrayWrapper,
                 records_arr: tp.RecordArray,
                 col_mapper: tp.Optional[ColumnMapper] = None,
                 **kwargs) -> None:
        """
        Records类的初始化方法
        
        这个方法负责初始化Records对象，包括数据验证、字段检查、
        以及各种组件的初始化。它确保传入的数据符合Records类的要求。
        
        参数：
            wrapper (ArrayWrapper): 数组包装器，包含索引、列名、分组等元数据
            records_arr (tp.RecordArray): 结构化NumPy数组，存储记录数据
            col_mapper (tp.Optional[ColumnMapper]): 列映射器，可选参数
            **kwargs: 传递给配置系统的额外参数
        
        初始化流程：
        1. 调用父类的初始化方法
        2. 验证记录数组的结构
        3. 检查字段配置的完整性
        4. 初始化列映射器
        """
        # 调用Wrapping类的初始化方法
        # 传递包装器和所有参数，建立基础的数组包装功能
        Wrapping.__init__(
            self,
            wrapper,                    # 数组包装器
            records_arr=records_arr,    # 记录数组
            col_mapper=col_mapper,      # 列映射器
            **kwargs                    # 其他配置参数
        )
        
        # 调用StatsBuilderMixin的初始化方法
        # 初始化统计构建器，为统计分析功能做准备
        StatsBuilderMixin.__init__(self)

        # 字段验证部分：确保记录数组具有正确的结构
        
        # 将记录数组转换为NumPy数组，确保类型正确
        records_arr = np.asarray(records_arr)
        
        # 检查记录数组是否为结构化数组（必须有字段定义）
        # 如果不是结构化数组，则抛出异常
        checks.assert_not_none(records_arr.dtype.fields)
        
        # 从字段配置中提取所有字段名称
        # 这个集合包含了配置中定义的所有字段的实际名称
        field_names = {
            dct.get('name', field_name)  # 获取字段的实际名称，如果没有定义则使用字段名
            for field_name, dct in self.field_config.get('settings', {}).items()
        }
        
        # 获取字段配置中定义的数据类型
        dtype = self.field_config.get('dtype', None)
        
        # 如果配置中定义了数据类型，则进行字段完整性检查
        if dtype is not None:
            # 遍历数据类型中定义的所有字段
            for field in dtype.names:
                # 检查字段是否存在于记录数组中
                if field not in records_arr.dtype.names:
                    # 如果字段不在记录数组中，也不在配置的字段名中，则抛出异常
                    if field not in field_names:
                        raise TypeError(f"Field '{field}' from {dtype} cannot be found in records or config")

        # 存储记录数组到实例变量
        # 使用私有变量名，表明这是内部使用的数据
        self._records_arr = records_arr
        
        # 初始化列映射器
        if col_mapper is None:
            # 如果没有提供列映射器，则创建一个新的
            # 列映射器用于优化列级别的数据访问
            col_mapper = ColumnMapper(wrapper, self.col_arr)
        
        # 存储列映射器到实例变量
        self._col_mapper = col_mapper

    def replace(self: RecordsT, **kwargs) -> RecordsT:
        """
        替换Records对象的属性并返回新实例 - 不可变对象的标准模式
        
        这个方法遵循不可变对象的设计模式，通过创建新实例而不是修改现有实例
        来"更改"对象。它是Configured类的replace方法的扩展，添加了
        Records特有的列映射器失效逻辑。
        
        设计原理：
        - 保持对象的不可变性，确保数据的一致性
        - 智能处理列映射器的失效情况
        - 确保新实例的完整性和正确性
        
        参数：
            **kwargs: 要替换的属性及其新值
        
        返回：
            RecordsT: 新的Records实例，包含更新后的属性
        
        注意事项：
        - 当wrapper或records_arr发生变化时，会自动使col_mapper失效
        - 这确保了列映射器与实际数据的一致性
        
        使用示例：
        ```python
        import numpy as np
        import pandas as pd
        import vectorbt as vbt
        
        # 创建原始Records对象
        records = vbt.Records(wrapper, records_arr)
        
        # 替换包装器
        new_wrapper = wrapper.replace(columns=['A', 'B', 'C'])
        new_records = records.replace(wrapper=new_wrapper)
        
        # 替换记录数组
        new_records_arr = records_arr[:5]  # 只保留前5条记录
        filtered_records = records.replace(records_arr=new_records_arr)
        
        # 同时替换多个属性
        updated_records = records.replace(
            wrapper=new_wrapper,
            records_arr=new_records_arr
        )
        ```
        """
        # 检查是否需要使列映射器失效
        # 列映射器依赖于wrapper和records_arr，如果这些发生变化，需要重新创建
        if self.config.get('col_mapper', None) is not None:
            # 如果要替换wrapper，且新的wrapper与现有的不同
            if 'wrapper' in kwargs:
                if self.wrapper is not kwargs.get('wrapper'):
                    # 将col_mapper设置为None，强制重新创建
                    kwargs['col_mapper'] = None
            
            # 如果要替换records_arr，且新的records_arr与现有的不同
            if 'records_arr' in kwargs:
                if self.records_arr is not kwargs.get('records_arr'):
                    # 将col_mapper设置为None，强制重新创建
                    kwargs['col_mapper'] = None
        
        # 调用父类的replace方法创建新实例
        return Configured.replace(self, **kwargs)

    def get_by_col_idxs(self, col_idxs: tp.Array1d) -> tp.RecordArray:
        """
        根据列索引获取对应的记录 - 高效的列级别数据选择
        
        这个方法根据提供的列索引数组，从Records中选择对应列的所有记录。
        它使用了两种不同的算法，根据列映射器的排序状态选择最优的实现。
        
        算法选择：
        - 如果列是排序的：使用range_select算法，基于范围选择，速度更快
        - 如果列不是排序的：使用map_select算法，基于映射选择，更灵活
        
        参数：
            col_idxs (tp.Array1d): 一维数组，包含要选择的列索引
        
        返回：
            tp.RecordArray: 新的记录数组，只包含指定列的记录
        
        性能说明：
        - 对于排序的列，时间复杂度为O(k)，其中k是选择的列数
        - 对于未排序的列，时间复杂度为O(n)，其中n是记录总数
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建Records对象
        records = vbt.Records(wrapper, records_arr)
        
        # 选择第0列和第2列的记录
        col_indices = np.array([0, 2])
        selected_records = records.get_by_col_idxs(col_indices)
        
        # 选择单个列的记录
        single_col_records = records.get_by_col_idxs(np.array([1]))
        
        # 选择多个连续列的记录
        continuous_cols = np.array([0, 1, 2])
        continuous_records = records.get_by_col_idxs(continuous_cols)
        ```
        """
        # 根据列映射器的排序状态选择最优算法
        if self.col_mapper.is_sorted():
            # 列是排序的：使用基于范围的快速选择算法
            # 这种方法利用了排序的特性，通过范围查找实现高效选择
            new_records_arr = nb.record_col_range_select_nb(
                self.values,                      # 原始记录数组
                self.col_mapper.col_range,        # 列范围映射
                to_1d_array(col_idxs)            # 转换为1维数组的列索引
            )  # 这种方法更快，因为利用了排序特性
        else:
            # 列不是排序的：使用基于映射的选择算法
            # 这种方法更灵活，但速度相对较慢
            new_records_arr = nb.record_col_map_select_nb(
                self.values,                      # 原始记录数组
                self.col_mapper.col_map,          # 列映射表
                to_1d_array(col_idxs)            # 转换为1维数组的列索引
            )
        
        # 返回新的记录数组
        return new_records_arr

    def indexing_func_meta(self, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> IndexingMetaT:
        """
        执行索引操作并返回元数据 - 索引操作的核心实现
        
        这个方法是Records类索引操作的核心实现，它处理pandas风格的索引操作
        并返回包含元数据的元组。这个方法主要用于内部实现，为其他索引方法
        提供底层支持。
        
        索引限制：
        - 只支持列选择，不支持行选择（时间轴）
        - 索引行为完全依赖于ArrayWrapper的实现
        - 如果启用了group_select，索引将在组上执行，否则在单列上执行
        
        参数：
            pd_indexing_func (tp.PandasIndexingFunc): pandas索引函数
            **kwargs: 传递给索引函数的额外参数
        
        返回：
            IndexingMetaT: 包含四个元素的元组：
                - new_wrapper: 新的ArrayWrapper对象
                - new_records_arr: 新的记录数组
                - group_idxs: 组索引
                - col_idxs: 列索引
        
        使用示例：
        ```python
        # 这个方法主要用于内部实现，一般不直接调用
        # 但可以用于理解索引操作的内部机制
        
        records = vbt.Records(wrapper, records_arr)
        
        # 内部调用示例（通常不直接使用）
        def select_func(obj):
            return obj.iloc[:, [0, 2]]  # 选择第0列和第2列
        
        new_wrapper, new_records_arr, group_idxs, col_idxs = records.indexing_func_meta(select_func)
        ```
        """
        # 调用ArrayWrapper的索引元数据方法
        # column_only_select=True表示只支持列选择，不支持行选择
        new_wrapper, _, group_idxs, col_idxs = \
            self.wrapper.indexing_func_meta(pd_indexing_func, column_only_select=True, **kwargs)
        
        # 根据列索引获取新的记录数组
        new_records_arr = self.get_by_col_idxs(col_idxs)
        
        # 返回包含所有元数据的元组
        return new_wrapper, new_records_arr, group_idxs, col_idxs

    def indexing_func(self: RecordsT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> RecordsT:
        """
        执行索引操作并返回新的Records实例 - 用户友好的索引接口
        
        这个方法是Records类的主要索引接口，它接受pandas风格的索引操作
        并返回一个新的Records实例。它在indexing_func_meta的基础上
        提供了更友好的用户接口。
        
        索引特性：
        - 支持pandas风格的索引操作（如.loc, .iloc等）
        - 自动处理wrapper和records_arr的更新
        - 保持对象的不可变性
        
        参数：
            pd_indexing_func (tp.PandasIndexingFunc): pandas索引函数
            **kwargs: 传递给索引函数的额外参数
        
        返回：
            RecordsT: 新的Records实例，包含索引后的数据
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建Records对象
        records = vbt.Records(wrapper, records_arr)
        
        # 使用pandas风格的索引
        # 选择特定列
        col_a_records = records['a']
        
        # 选择多个列
        multi_col_records = records[['a', 'c']]
        
        # 使用iloc选择
        first_two_cols = records.iloc[:, :2]
        
        # 使用loc选择
        selected_records = records.loc[:, 'a':'c']
        
        # 条件选择（如果支持）
        filtered_records = records[records.wrapper.columns.isin(['a', 'b'])]
        ```
        """
        # 调用索引元数据方法获取新的wrapper和records_arr
        new_wrapper, new_records_arr, _, _ = self.indexing_func_meta(pd_indexing_func, **kwargs)
        
        # 使用新的wrapper和records_arr创建新的Records实例
        return self.replace(
            wrapper=new_wrapper,
            records_arr=new_records_arr
        )

    @property
    def records_arr(self) -> tp.RecordArray:
        """
        记录数组属性 - 返回底层的结构化数组
        
        这个属性提供对底层记录数组的只读访问。记录数组是Records类的
        核心数据结构，包含了所有的记录信息。
        
        返回：
            tp.RecordArray: 结构化NumPy数组，包含所有记录数据
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 访问底层记录数组
        raw_array = records.records_arr
        print(f"数组类型: {type(raw_array)}")
        print(f"数组形状: {raw_array.shape}")
        print(f"字段名称: {raw_array.dtype.names}")
        
        # 访问特定字段
        ids = raw_array['id']
        cols = raw_array['col']
        prices = raw_array['price']  # 如果存在price字段
        ```
        """
        return self._records_arr

    @property
    def values(self) -> tp.RecordArray:
        """
        值属性 - records_arr的别名，提供更直观的访问方式
        
        这个属性是records_arr的别名，提供了一种更直观的方式来访问
        底层的记录数组。在vectorbt的设计中，values通常表示对象的
        核心数据。
        
        返回：
            tp.RecordArray: 结构化NumPy数组，与records_arr相同
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 两种访问方式是等价的
        array1 = records.records_arr
        array2 = records.values
        print(f"相等: {np.array_equal(array1, array2)}")  # True
        
        # 通常更倾向于使用values
        print(f"记录数量: {len(records.values)}")
        print(f"字段: {records.values.dtype.names}")
        ```
        """
        return self.records_arr

    def __len__(self) -> int:
        """
        长度方法 - 返回记录的数量
        
        这个魔法方法使得Records对象可以使用len()函数，
        返回记录数组中记录的总数。
        
        返回：
            int: 记录的总数
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 获取记录数量
        record_count = len(records)
        print(f"总记录数: {record_count}")
        
        # 也可以直接在条件语句中使用
        if len(records) > 0:
            print("有记录存在")
        
        # 与其他长度比较
        print(f"记录数与数组长度相等: {len(records) == len(records.values)}")
        ```
        """
        return len(self.values)

    @property
    def records(self) -> tp.Frame:
        """
        记录DataFrame属性 - 将结构化数组转换为pandas DataFrame
        
        这个属性将底层的结构化数组转换为pandas DataFrame，
        保持原始的字段名称和数据类型。这为用户提供了熟悉的
        pandas接口来查看和分析数据。
        
        返回：
            tp.Frame: pandas DataFrame，包含所有记录数据
        
        注意：
        - 这个DataFrame保持了原始的字段名称（如'id', 'col', 'idx'）
        - 如果需要更友好的显示，请使用records_readable属性
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 获取DataFrame格式的记录
        df = records.records
        print("原始记录DataFrame:")
        print(df)
        
        # 可以使用pandas的所有功能
        print(f"DataFrame形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"数据类型: {df.dtypes}")
        
        # 进行pandas操作
        filtered_df = df[df['col'] == 0]  # 选择第0列的记录
        grouped_df = df.groupby('col').size()  # 按列分组统计
        ```
        """
        return pd.DataFrame.from_records(self.values)

    @property
    def recarray(self) -> tp.RecArray:
        """
        记录数组视图属性 - 返回NumPy记录数组视图
        
        这个属性将结构化数组转换为NumPy的记录数组视图，
        提供了字段的点表示法访问。记录数组是结构化数组的
        一种特殊视图，允许使用点符号访问字段。
        
        返回：
            tp.RecArray: NumPy记录数组视图
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 获取记录数组视图
        rec_view = records.recarray
        
        # 使用点符号访问字段
        ids = rec_view.id          # 等价于 records.values['id']
        cols = rec_view.col        # 等价于 records.values['col']
        
        # 如果有其他字段
        if hasattr(rec_view, 'price'):
            prices = rec_view.price
        
        print(f"ID数组: {ids}")
        print(f"列数组: {cols}")
        ```
        """
        return self.values.view(np.recarray)

    @property
    def col_mapper(self) -> ColumnMapper:
        """
        列映射器属性 - 返回用于列操作的映射器对象
        
        列映射器是Records类的重要组件，它提供了高效的列级别数据访问
        和操作功能。它维护了列索引到记录的映射关系，使得按列操作
        变得高效。
        
        返回：
            ColumnMapper: 列映射器对象，用于优化列操作
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 访问列映射器
        mapper = records.col_mapper
        
        # 查看映射器信息
        print(f"是否排序: {mapper.is_sorted()}")
        print(f"列映射: {mapper.col_map}")
        
        # 如果列是排序的，可以访问列范围
        if mapper.is_sorted():
            print(f"列范围: {mapper.col_range}")
        
        # 获取特定列的记录数量
        col_counts = mapper.col_lens
        print(f"每列记录数: {col_counts}")
        ```
        """
        return self._col_mapper

    @property
    def records_readable(self) -> tp.Frame:
        """
        可读记录DataFrame属性 - 返回用户友好的记录显示格式
        
        这个属性将底层的结构化数组转换为用户友好的pandas DataFrame，
        考虑了字段配置中的标题、映射和其他显示设置。与records属性不同，
        这个属性会应用字段配置中的转换规则，提供更直观的数据展示。
        
        转换规则：
        - 使用字段配置中的'title'作为列名
        - 应用字段配置中的'mapping'进行值转换
        - 隐藏标记为'ignore'的字段
        - 将索引字段映射到实际的时间索引
        
        返回：
            tp.Frame: pandas DataFrame，包含格式化后的记录数据
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建Records对象
        records = vbt.Records(wrapper, records_arr)
        
        # 比较原始记录和可读记录
        print("原始记录:")
        print(records.records)
        
        print("\n可读记录:")
        print(records.records_readable)
        
        # 可读记录的特点：
        # 1. 列名更友好（如'Id'而不是'id'）
        # 2. 'Column'字段显示实际的列名而不是索引
        # 3. 'Timestamp'字段显示实际的时间而不是索引
        # 4. 枚举字段显示枚举值而不是数字
        ```
        """
        # 创建原始记录DataFrame的副本
        df = self.records.copy()
        
        # 获取字段配置中的设置信息
        field_settings = self.field_config.get('settings', {})
        
        # 遍历DataFrame的所有列，应用字段配置
        for col_name in df.columns:
            # 检查当前列是否有配置
            if col_name in field_settings:
                # 获取该字段的配置字典
                dct = field_settings[col_name]
                
                # 如果字段被标记为忽略，从DataFrame中删除
                if dct.get('ignore', False):
                    df = df.drop(columns=col_name)
                    continue
                
                # 获取字段的实际名称（如果配置中有重命名）
                field_name = dct.get('name', col_name)
                
                # 处理字段标题
                if 'title' in dct:
                    # 使用配置中的标题
                    title = dct['title']
                    # 重命名列
                    new_columns = dict()
                    new_columns[field_name] = title
                    df.rename(columns=new_columns, inplace=True)
                else:
                    # 如果没有标题，使用字段名称
                    title = field_name
                
                # 处理字段映射
                if 'mapping' in dct:
                    # 检查映射类型
                    if isinstance(dct['mapping'], str) and dct['mapping'] == 'index':
                        # 如果映射到索引，将索引值转换为实际的索引标签
                        df[title] = self.get_map_field_to_index(col_name)
                    else:
                        # 如果是其他类型的映射（如枚举），应用映射转换
                        df[title] = self.get_apply_mapping_arr(col_name)
        
        # 返回格式化后的DataFrame
        return df

    def get_field_setting(self, field: str, setting: str, default: tp.Any = None) -> tp.Any:
        """
        获取字段的特定设置值 - 字段配置的统一访问接口
        
        这个方法提供了对字段配置中特定设置的统一访问方式。
        它从字段配置中提取指定字段的指定设置，如果不存在则返回默认值。
        
        参数：
            field (str): 字段名称
            setting (str): 设置名称（如'name', 'title', 'mapping'等）
            default (tp.Any, 可选): 如果设置不存在时的默认值
        
        返回：
            tp.Any: 设置值或默认值
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 获取字段的各种设置
        id_name = records.get_field_setting('id', 'name')
        col_title = records.get_field_setting('col', 'title')
        idx_mapping = records.get_field_setting('idx', 'mapping')
        
        # 使用默认值
        custom_setting = records.get_field_setting('price', 'precision', 2)
        
        print(f"ID字段名称: {id_name}")
        print(f"列字段标题: {col_title}")
        print(f"索引字段映射: {idx_mapping}")
        print(f"价格精度: {custom_setting}")
        ```
        """
        # 从字段配置中获取设置值
        # 使用链式get方法确保每一级都存在，如果不存在则返回默认值
        return self.field_config.get('settings', {}).get(field, {}).get(setting, default)

    def get_field_name(self, field: str) -> str:
        """
        获取字段的实际名称 - 字段名称解析
        
        这个方法解析字段的实际名称。在字段配置中，可能会重命名字段
        （通过'name'设置），这个方法返回字段在记录数组中的实际名称。
        
        参数：
            field (str): 字段的逻辑名称
        
        返回：
            str: 字段在记录数组中的实际名称
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 获取字段的实际名称
        id_name = records.get_field_name('id')        # 通常返回'id'
        col_name = records.get_field_name('col')      # 通常返回'col'
        idx_name = records.get_field_name('idx')      # 通常返回'idx'
        
        # 对于自定义字段配置，可能返回不同的名称
        # 如果配置中设置了 'id': {'name': 'order_id'}
        # 那么 get_field_name('id') 会返回 'order_id'
        
        print(f"ID字段实际名称: {id_name}")
        print(f"列字段实际名称: {col_name}")
        print(f"索引字段实际名称: {idx_name}")
        ```
        """
        # 调用get_field_setting方法获取'name'设置，如果不存在则返回字段名本身
        return self.get_field_setting(field, 'name', field)

    def get_field_title(self, field: str) -> str:
        """
        获取字段的显示标题 - 字段标题解析
        
        这个方法解析字段的显示标题。标题用于在用户界面中显示字段，
        如果字段配置中没有定义标题，则返回字段名称。
        
        参数：
            field (str): 字段名称
        
        返回：
            str: 字段的显示标题
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 获取字段的显示标题
        id_title = records.get_field_title('id')        # 通常返回'Id'
        col_title = records.get_field_title('col')      # 通常返回'Column'
        idx_title = records.get_field_title('idx')      # 通常返回'Timestamp'
        
        # 对于自定义字段
        price_title = records.get_field_title('price')  # 可能返回'Price'或'价格'
        
        print(f"ID字段标题: {id_title}")
        print(f"列字段标题: {col_title}")
        print(f"索引字段标题: {idx_title}")
        print(f"价格字段标题: {price_title}")
        ```
        """
        # 调用get_field_setting方法获取'title'设置，如果不存在则返回字段名
        return self.get_field_setting(field, 'title', field)

    def get_field_mapping(self, field: str) -> tp.Optional[tp.MappingLike]:
        """
        获取字段的映射配置 - 字段映射解析
        
        这个方法解析字段的映射配置。映射用于将字段的数值转换为
        更易理解的表示形式，如将枚举值转换为字符串，或将索引
        转换为实际的时间戳。
        
        参数：
            field (str): 字段名称
        
        返回：
            tp.Optional[tp.MappingLike]: 映射配置，如果不存在则返回None
        
        映射类型：
        - 字符串'index': 映射到ArrayWrapper的index
        - 字符串'columns': 映射到ArrayWrapper的columns
        - 字典: 值到标签的映射
        - 枚举类: 枚举值到名称的映射
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 获取字段的映射配置
        col_mapping = records.get_field_mapping('col')    # 通常返回'columns'
        idx_mapping = records.get_field_mapping('idx')    # 通常返回'index'
        
        # 对于枚举字段
        side_mapping = records.get_field_mapping('side')  # 可能返回OrderSide枚举
        
        # 对于自定义映射
        status_mapping = records.get_field_mapping('status')  # 可能返回状态字典
        
        print(f"列映射: {col_mapping}")
        print(f"索引映射: {idx_mapping}")
        print(f"方向映射: {side_mapping}")
        print(f"状态映射: {status_mapping}")
        ```
        """
        # 调用get_field_setting方法获取'mapping'设置，如果不存在则返回None
        return self.get_field_setting(field, 'mapping', None)

    def get_field_arr(self, field: str) -> tp.Array1d:
        """
        获取字段的数组数据 - 字段数据访问
        
        这个方法获取指定字段的底层数组数据。它会解析字段的实际名称，
        然后从记录数组中提取对应的数据。
        
        参数：
            field (str): 字段的逻辑名称
        
        返回：
            tp.Array1d: 字段的一维数组数据
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 获取基础字段的数组
        id_arr = records.get_field_arr('id')
        col_arr = records.get_field_arr('col')
        idx_arr = records.get_field_arr('idx')
        
        # 获取自定义字段的数组
        price_arr = records.get_field_arr('price')
        volume_arr = records.get_field_arr('volume')
        
        print(f"ID数组: {id_arr}")
        print(f"列数组: {col_arr}")
        print(f"价格数组: {price_arr}")
        
        # 可以对数组进行各种操作
        max_price = np.max(price_arr)
        avg_volume = np.mean(volume_arr)
        
        print(f"最高价格: {max_price}")
        print(f"平均成交量: {avg_volume}")
        ```
        """
        # 首先获取字段的实际名称，然后从记录数组中提取数据
        return self.values[self.get_field_name(field)]

    def get_map_field(self, field: str, **kwargs) -> MappedArray:
        """
        获取字段的映射数组 - 字段映射包装
        
        这个方法将字段数据转换为MappedArray对象，并应用字段配置中的映射。
        它是map_field方法的高级版本，会自动处理字段名称解析和映射应用。
        
        参数：
            field (str): 字段名称
            **kwargs: 传递给map_field方法的额外参数
        
        返回：
            MappedArray: 包含字段数据的映射数组对象
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 获取字段的映射数组
        id_mapped = records.get_map_field('id')
        col_mapped = records.get_map_field('col')
        price_mapped = records.get_map_field('price')
        
        # 对映射数组进行操作
        price_stats = price_mapped.describe()
        price_mean = price_mapped.mean()
        
        # 应用分组
        grouped_price = price_mapped.mean(group_by=['Tech', 'Tech', 'Finance'])
        
        print(f"价格统计: {price_stats}")
        print(f"平均价格: {price_mean}")
        print(f"分组平均价格: {grouped_price}")
        ```
        """
        # 调用map_field方法，传递实际字段名称和映射配置
        return self.map_field(
            self.get_field_name(field),          # 字段的实际名称
            mapping=self.get_field_mapping(field),  # 字段的映射配置
            **kwargs                             # 其他参数
        )

    def get_apply_mapping_arr(self, field: str, **kwargs) -> tp.Array1d:
        """
        获取应用映射后的字段数组 - 映射值转换
        
        这个方法获取字段的映射数组，然后应用映射转换，返回转换后的数组。
        这对于将枚举值转换为字符串或其他可读形式非常有用。
        
        参数：
            field (str): 字段名称
            **kwargs: 传递给get_map_field方法的额外参数
        
        返回：
            tp.Array1d: 应用映射后的一维数组
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 获取应用映射后的数组
        col_names = records.get_apply_mapping_arr('col')  # 返回列名而不是索引
        
        # 对于枚举字段
        if 'side' in records.values.dtype.names:
            side_names = records.get_apply_mapping_arr('side')  # 返回'Buy'/'Sell'而不是0/1
        
        print(f"列名: {col_names}")
        print(f"交易方向: {side_names}")
        ```
        """
        # 获取字段的映射数组，然后应用映射转换
        return self.get_map_field(field, **kwargs).apply_mapping().values

    def get_map_field_to_index(self, field: str, **kwargs) -> tp.Index:
        """
        获取字段映射到索引的结果 - 索引转换
        
        这个方法将字段的映射数组转换为pandas索引。这对于将索引字段
        转换为实际的时间索引特别有用。
        
        参数：
            field (str): 字段名称
            **kwargs: 传递给get_map_field方法的额外参数
        
        返回：
            tp.Index: pandas索引对象
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 将索引字段转换为实际的时间索引
        timestamps = records.get_map_field_to_index('idx')
        
        # 将列索引转换为列名
        column_names = records.get_map_field_to_index('col')
        
        print(f"时间戳: {timestamps}")
        print(f"列名: {column_names}")
        ```
        """
        # 获取字段的映射数组，然后转换为索引
        return self.get_map_field(field, **kwargs).to_index()

    @property
    def id_arr(self) -> tp.Array1d:
        """
        ID数组属性 - 返回记录ID的数组
        
        这个属性返回记录的ID数组。ID是每个记录的唯一标识符，
        用于区分不同的记录。
        
        返回：
            tp.Array1d: 包含所有记录ID的一维数组
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 获取ID数组
        ids = records.id_arr
        
        print(f"记录ID: {ids}")
        print(f"ID数量: {len(ids)}")
        print(f"最小ID: {np.min(ids)}")
        print(f"最大ID: {np.max(ids)}")
        
        # 检查ID的唯一性
        unique_ids = np.unique(ids)
        print(f"唯一ID数量: {len(unique_ids)}")
        print(f"是否有重复ID: {len(ids) != len(unique_ids)}")
        ```
        """
        # 通过get_field_arr方法获取ID字段的数组
        return self.values[self.get_field_name('id')]

    @property
    def col_arr(self) -> tp.Array1d:
        """
        列数组属性 - 返回记录所属列的数组
        
        这个属性返回记录的列数组。列数组表示每个记录属于哪一列
        （如哪个股票、哪个策略等）。
        
        返回：
            tp.Array1d: 包含所有记录列索引的一维数组
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 获取列数组
        cols = records.col_arr
        
        print(f"列索引: {cols}")
        print(f"列数量: {len(np.unique(cols))}")
        print(f"最小列索引: {np.min(cols)}")
        print(f"最大列索引: {np.max(cols)}")
        
        # 统计每列的记录数
        unique_cols, counts = np.unique(cols, return_counts=True)
        for col, count in zip(unique_cols, counts):
            print(f"列 {col}: {count} 条记录")
        ```
        """
        # 通过get_field_arr方法获取列字段的数组
        return self.values[self.get_field_name('col')]

    @property
    def idx_arr(self) -> tp.Optional[tp.Array1d]:
        """
        索引数组属性 - 返回记录时间索引的数组
        
        这个属性返回记录的索引数组。索引数组表示每个记录在时间轴上的位置。
        如果字段配置中没有定义索引字段，则返回None。
        
        返回：
            tp.Optional[tp.Array1d]: 包含所有记录时间索引的一维数组，如果不存在则返回None
        
        使用示例：
        ```python
        records = vbt.Records(wrapper, records_arr)
        
        # 获取索引数组
        indices = records.idx_arr
        
        if indices is not None:
            print(f"时间索引: {indices}")
            print(f"最早时间索引: {np.min(indices)}")
            print(f"最晚时间索引: {np.max(indices)}")
            
            # 统计每个时间点的记录数
            unique_indices, counts = np.unique(indices, return_counts=True)
            for idx, count in zip(unique_indices, counts):
                print(f"时间点 {idx}: {count} 条记录")
        else:
            print("没有时间索引信息")
        ```
        """
        # 获取索引字段的实际名称
        idx_field_name = self.get_field_name('idx')
        
        # 如果索引字段名称为None，返回None
        if idx_field_name is None:
            return None
        
        # 返回索引字段的数组
        return self.values[idx_field_name]

    @cached_method
    def is_sorted(self, incl_id: bool = False) -> bool:
        """
        检查记录是否已排序 - 数据排序状态验证
        
        这个方法检查记录数组是否按照列（主键）和可选的ID（次键）排序。
        排序状态对于某些算法的性能优化非常重要，如范围选择算法。
        
        排序规则：
        - 主键：按列索引(col)排序
        - 次键：按记录ID排序（如果incl_id=True）
        
        参数：
            incl_id (bool, 可选): 是否在排序检查中包含ID字段，默认为False
        
        返回：
            bool: 如果记录已排序则返回True，否则返回False
        
        性能说明：
        - 使用@cached_method装饰器缓存结果，避免重复计算
        - 对于大型数据集，排序检查可能比较耗时
        - 排序状态影响某些算法的选择和性能
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建Records对象
        records = vbt.Records(wrapper, records_arr)
        
        # 检查排序状态
        is_col_sorted = records.is_sorted()
        is_full_sorted = records.is_sorted(incl_id=True)
        
        print(f"按列排序: {is_col_sorted}")
        print(f"按列和ID排序: {is_full_sorted}")
        
        # 根据排序状态选择不同的处理策略
        if is_col_sorted:
            print("数据已按列排序，可以使用快速范围选择算法")
        else:
            print("数据未排序，将使用通用映射选择算法")
        
        # 如果需要排序
        if not is_col_sorted:
            sorted_records = records.sort()
            print(f"排序后状态: {sorted_records.is_sorted()}")
        ```
        """
        # 根据incl_id参数选择不同的排序检查函数
        if incl_id:
            # 检查是否同时按列和ID排序
            # 这提供了最严格的排序检查，确保记录完全有序
            return nb.is_col_idx_sorted_nb(self.col_arr, self.id_arr)
        
        # 只检查是否按列排序
        # 这是最常用的排序检查，满足大部分性能优化需求
        return nb.is_col_sorted_nb(self.col_arr)

    def sort(self: RecordsT, incl_id: bool = False, group_by: tp.GroupByLike = None, **kwargs) -> RecordsT:
        """
        对记录进行排序 - 数据排序操作
        
        这个方法对记录数组进行排序，主要按列排序，可选地按ID进行次级排序。
        排序后的数据可以提高某些操作的性能，如范围选择。
        
        排序策略：
        - 如果数据已经排序，直接返回（避免不必要的计算）
        - 否则使用NumPy的lexsort进行多键排序
        
        参数：
            incl_id (bool, 可选): 是否在排序中包含ID字段，默认为False
            group_by (tp.GroupByLike, 可选): 分组方式，用于排序后的重新分组
            **kwargs: 传递给replace方法的额外参数
        
        返回：
            RecordsT: 排序后的新Records实例
        
        性能警告：
        - 排序操作成本高，特别是对于大型数据集
        - 更好的方法是在创建记录时就保持正确的顺序
        - 对于频繁的排序操作，建议预先排序数据
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建Records对象（可能是无序的）
        records = vbt.Records(wrapper, records_arr)
        
        # 检查是否需要排序
        if not records.is_sorted():
            # 按列排序
            sorted_records = records.sort()
            print(f"排序后状态: {sorted_records.is_sorted()}")
            
            # 按列和ID排序（更严格的排序）
            full_sorted = records.sort(incl_id=True)
            print(f"完全排序状态: {full_sorted.is_sorted(incl_id=True)}")
        
        # 排序后进行分组
        grouped_sorted = records.sort(group_by=['A', 'A', 'B'])
        
        # 性能比较
        import time
        
        # 测试排序性能
        start_time = time.time()
        sorted_records = records.sort()
        sort_time = time.time() - start_time
        print(f"排序耗时: {sort_time:.4f}秒")
        
        # 测试已排序数据的操作性能
        start_time = time.time()
        selected = sorted_records.get_by_col_idxs(np.array([0, 1]))
        select_time = time.time() - start_time
        print(f"选择耗时: {select_time:.4f}秒")
        ```
        """
        # 首先检查是否已经排序，避免不必要的排序操作
        if self.is_sorted(incl_id=incl_id):
            # 如果已经排序，只需要应用其他参数（如kwargs）并重新分组
            return self.replace(**kwargs).regroup(group_by)
        
        # 执行排序操作
        if incl_id:
            # 按列（主键）和ID（次键）进行排序
            # lexsort接受多个键，按从最不重要到最重要的顺序排列
            # 这里ID是次键，col是主键
            ind = np.lexsort((self.id_arr, self.col_arr))  # 昂贵的操作！
        else:
            # 只按列进行排序
            ind = np.argsort(self.col_arr)
        
        # 使用排序索引重新排列记录数组，并创建新的Records实例
        return self.replace(records_arr=self.values[ind], **kwargs).regroup(group_by)

    def apply_mask(self: RecordsT, mask: tp.Array1d, group_by: tp.GroupByLike = None, **kwargs) -> RecordsT:
        """
        应用掩码过滤记录 - 基于条件的记录筛选
        
        这个方法根据提供的布尔掩码过滤记录，返回一个包含筛选后记录的新实例。
        掩码是一个与记录数组长度相同的布尔数组，True表示保留，False表示过滤掉。
        
        过滤原理：
        - 使用np.flatnonzero找到所有True值的索引
        - 使用np.take根据索引选择记录
        - 创建新的Records实例包含过滤后的数据
        
        参数：
            mask (tp.Array1d): 布尔掩码数组，与记录数组长度相同
            group_by (tp.GroupByLike, 可选): 分组方式，用于过滤后的重新分组
            **kwargs: 传递给replace方法的额外参数
        
        返回：
            RecordsT: 包含过滤后记录的新Records实例
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建Records对象
        records = vbt.Records(wrapper, records_arr)
        
        # 1. 基于字段值的简单过滤
        price_mask = records.get_field_arr('price') > 100
        high_price_records = records.apply_mask(price_mask)
        print(f"高价格记录数量: {len(high_price_records)}")
        
        # 2. 基于多个条件的复合过滤
        volume_mask = records.get_field_arr('volume') > 1000
        complex_mask = price_mask & volume_mask
        filtered_records = records.apply_mask(complex_mask)
        print(f"复合条件记录数量: {len(filtered_records)}")
        
        # 3. 基于列的过滤
        col_mask = records.col_arr == 0  # 只保留第0列的记录
        col_records = records.apply_mask(col_mask)
        
        # 4. 基于时间的过滤
        if records.idx_arr is not None:
            time_mask = records.idx_arr >= 5  # 只保留时间索引>=5的记录
            recent_records = records.apply_mask(time_mask)
        
        # 5. 使用自定义函数创建掩码
        def custom_filter(records):
            price_arr = records.get_field_arr('price')
            volume_arr = records.get_field_arr('volume')
            # 价格-成交量关系过滤
            return (price_arr > np.mean(price_arr)) & (volume_arr > np.median(volume_arr))
        
        custom_mask = custom_filter(records)
        custom_filtered = records.apply_mask(custom_mask)
        
        # 6. 过滤后的分组操作
        grouped_filtered = records.apply_mask(
            price_mask, 
            group_by=['Large', 'Large', 'Small']
        )
        
        # 7. 统计过滤效果
        original_count = len(records)
        filtered_count = len(high_price_records)
        filter_ratio = filtered_count / original_count
        print(f"过滤保留比例: {filter_ratio:.2%}")
        ```
        """
        # 找到掩码中所有True值的索引
        # flatnonzero返回非零（True）元素的索引
        mask_indices = np.flatnonzero(mask)
        
        # 使用索引选择记录，创建新的Records实例
        return self.replace(
            records_arr=np.take(self.values, mask_indices),  # 根据索引选择记录
            **kwargs  # 传递其他参数
        ).regroup(group_by)  # 重新分组

    def map_array(self,
                  a: tp.ArrayLike,
                  idx_arr: tp.Optional[tp.ArrayLike] = None,
                  mapping: tp.Optional[tp.MappingLike] = None,
                  group_by: tp.GroupByLike = None,
                  **kwargs) -> MappedArray:
        """
        将数组转换为映射数组 - 数组到映射数组的转换
        
        这个方法将普通数组转换为MappedArray对象，使其能够利用Records的
        列映射和分组功能。数组的长度必须与记录数组的长度相匹配。
        
        转换过程：
        - 验证数组长度与记录数组一致
        - 创建MappedArray对象，关联列信息和索引信息
        - 应用可选的映射和分组
        
        参数：
            a (tp.ArrayLike): 要转换的数组，长度必须与记录数组相匹配
            idx_arr (tp.Optional[tp.ArrayLike], 可选): 索引数组，默认使用Records的idx_arr
            mapping (tp.Optional[tp.MappingLike], 可选): 值映射，用于转换数组值
            group_by (tp.GroupByLike, 可选): 分组方式
            **kwargs: 传递给MappedArray构造函数的额外参数
        
        返回：
            MappedArray: 映射数组对象
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建Records对象
        records = vbt.Records(wrapper, records_arr)
        
        # 1. 转换计算得到的数组
        price_arr = records.get_field_arr('price')
        volume_arr = records.get_field_arr('volume')
        
        # 计算成交金额
        value_arr = price_arr * volume_arr
        value_mapped = records.map_array(value_arr)
        
        # 2. 转换条件数组
        high_price_mask = price_arr > 100
        high_price_mapped = records.map_array(high_price_mask.astype(float))
        
        # 3. 使用自定义映射
        side_mapping = {0: 'Buy', 1: 'Sell'}
        if 'side' in records.values.dtype.names:
            side_arr = records.get_field_arr('side')
            side_mapped = records.map_array(side_arr, mapping=side_mapping)
        
        # 4. 计算复合指标
        # 价格相对于列均值的比例
        col_mean_prices = []
        for col in range(records.wrapper.shape[1]):
            col_mask = records.col_arr == col
            if np.any(col_mask):
                col_mean = np.mean(price_arr[col_mask])
                col_mean_prices.extend([col_mean] * np.sum(col_mask))
        
        price_ratio = price_arr / np.array(col_mean_prices)
        ratio_mapped = records.map_array(price_ratio)
        
        # 5. 分组操作
        grouped_value = records.map_array(
            value_arr, 
            group_by=['Tech', 'Tech', 'Finance']
        )
        
        # 6. 统计分析
        print(f"成交金额统计: {value_mapped.describe()}")
        print(f"价格比例均值: {ratio_mapped.mean()}")
        print(f"分组成交金额: {grouped_value.sum()}")
        
        # 7. 转换为DataFrame进行进一步分析
        df = value_mapped.to_pd()
        print("成交金额DataFrame:")
        print(df)
        ```
        """
        # 将输入转换为NumPy数组
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)
        
        # 验证数组长度与记录数组长度一致
        checks.assert_shape_equal(a, self.values)
        
        # 如果没有提供索引数组，使用Records的索引数组
        if idx_arr is None:
            idx_arr = self.idx_arr
        
        # 创建MappedArray对象
        return MappedArray(
            self.wrapper,           # 数组包装器
            a,                      # 要映射的数组
            self.col_arr,          # 列数组
            id_arr=self.id_arr,    # ID数组
            idx_arr=idx_arr,       # 索引数组
            mapping=mapping,       # 值映射
            col_mapper=self.col_mapper,  # 列映射器
            **kwargs               # 其他参数
        ).regroup(group_by)       # 重新分组

    def map_field(self, field: str, **kwargs) -> MappedArray:
        """
        将字段转换为映射数组 - 字段映射的便捷方法
        
        这个方法将记录中的特定字段转换为MappedArray对象，是map_array
        方法的便捷包装。它直接使用字段名称，无需手动提取数组。
        
        参数：
            field (str): 要映射的字段名称
            **kwargs: 传递给map_array方法的额外参数
        
        返回：
            MappedArray: 包含字段数据的映射数组对象
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建Records对象
        records = vbt.Records(wrapper, records_arr)
        
        # 1. 映射基础字段
        price_mapped = records.map_field('price')
        volume_mapped = records.map_field('volume')
        
        # 2. 计算统计指标
        price_stats = price_mapped.describe()
        volume_mean = volume_mapped.mean()
        
        # 3. 分组分析
        grouped_price = price_mapped.mean(group_by=['Tech', 'Tech', 'Finance'])
        
        # 4. 时间序列分析
        if records.idx_arr is not None:
            price_by_time = price_mapped.groupby(records.idx_arr).mean()
        
        # 5. 相关性分析
        price_volume_corr = np.corrcoef(price_mapped.values, volume_mapped.values)[0, 1]
        
        print(f"价格统计: {price_stats}")
        print(f"成交量均值: {volume_mean}")
        print(f"分组价格: {grouped_price}")
        print(f"价格-成交量相关性: {price_volume_corr:.3f}")
        
        # 6. 可视化
        price_mapped.plots().show()
        ```
        """
        # 提取字段数组
        mapped_arr = self.values[field]
        
        # 使用map_array方法转换为映射数组
        return self.map_array(mapped_arr, **kwargs)

    def map(self,
            map_func_nb: tp.RecordMapFunc, *args,
            dtype: tp.Optional[tp.DTypeLike] = None,
            **kwargs) -> MappedArray:
        """Map each record to a scalar value. Returns mapped array.

        See `vectorbt.records.nb.map_records_nb`.

        `**kwargs` are passed to `Records.map_array`."""
        checks.assert_numba_func(map_func_nb)
        mapped_arr = nb.map_records_nb(self.values, map_func_nb, *args)
        mapped_arr = np.asarray(mapped_arr, dtype=dtype)
        return self.map_array(mapped_arr, **kwargs)

    def apply(self,
              apply_func_nb: tp.RecordApplyFunc, *args,
              group_by: tp.GroupByLike = None,
              apply_per_group: bool = False,
              dtype: tp.Optional[tp.DTypeLike] = None,
              **kwargs) -> MappedArray:
        """Apply function on records per column/group. Returns mapped array.

        Applies per group if `apply_per_group` is True.

        See `vectorbt.records.nb.apply_on_records_nb`.

        `**kwargs` are passed to `Records.map_array`."""
        checks.assert_numba_func(apply_func_nb)
        if apply_per_group:
            col_map = self.col_mapper.get_col_map(group_by=group_by)
        else:
            col_map = self.col_mapper.get_col_map(group_by=False)
        mapped_arr = nb.apply_on_records_nb(self.values, col_map, apply_func_nb, *args)
        mapped_arr = np.asarray(mapped_arr, dtype=dtype)
        return self.map_array(mapped_arr, group_by=group_by, **kwargs)

    @cached_method
    def count(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Return count by column."""
        wrap_kwargs = merge_dicts(dict(name_or_index='count'), wrap_kwargs)
        return self.wrapper.wrap_reduced(
            self.col_mapper.get_col_map(group_by=group_by)[1],
            group_by=group_by, **wrap_kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Records.stats`.

        Merges `vectorbt.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `records.stats` from `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        records_stats_cfg = settings['records']['stats']

        return merge_dicts(
            StatsBuilderMixin.stats_defaults.__get__(self),
            records_stats_cfg
        )

    _metrics: tp.ClassVar[Config] = Config(
        dict(
            start=dict(
                title='Start',
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags='wrapper'
            ),
            end=dict(
                title='End',
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags='wrapper'
            ),
            period=dict(
                title='Period',
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags='wrapper'
            ),
            count=dict(
                title='Count',
                calc_func='count',
                tags='records'
            )
        ),
        copy_kwargs=dict(copy_mode='deep')
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Records.plots`.

        Merges `vectorbt.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `records.plots` from `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        records_plots_cfg = settings['records']['plots']

        return merge_dicts(
            PlotsBuilderMixin.plots_defaults.__get__(self),
            records_plots_cfg
        )

    @property
    def subplots(self) -> Config:
        return self._subplots

    # ############# Docs ############# #

    @classmethod
    def build_field_config_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build field config documentation."""
        if source_cls is None:
            source_cls = Records
        return string.Template(
            inspect.cleandoc(get_dict_attr(source_cls, 'field_config').__doc__)
        ).substitute(
            {'field_config': cls.field_config.to_doc(), 'cls_name': cls.__name__}
        )

    @classmethod
    def override_field_config_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Call this method on each subclass that overrides `field_config`."""
        __pdoc__[cls.__name__ + '.field_config'] = cls.build_field_config_doc(source_cls=source_cls)


Records.override_field_config_doc(__pdoc__)
Records.override_metrics_doc(__pdoc__)
Records.override_subplots_doc(__pdoc__)
