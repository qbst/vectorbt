# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT RECORDS MODULE: 映射数组 (MappedArray)
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中用于处理映射数组的核心模块。MappedArray类是vectorbt框架中
最重要的数据结构之一，它提供了一种高效的方式来处理稀疏的、结构化的金融数据。

核心设计理念：
1. **压缩存储优化**：MappedArray使用压缩存储格式，只存储非空值，大幅节省内存空间。
   与传统的二维DataFrame相比，在处理稀疏交易数据时可以节省80%以上的内存。

2. **高性能计算**：通过Numba JIT编译的底层函数，实现接近C语言速度的数据处理。
   避免了pandas DataFrame的开销，为大规模量化分析提供了性能保障。

3. **灵活的数据访问**：支持按列、按组、按索引等多种维度的数据访问和操作。
   可以在不展开为完整矩阵的情况下进行各种统计计算和数据变换。

4. **无缝集成**：与Records、ColumnMapper等模块深度集成，形成完整的数据处理生态。
   支持与pandas的双向转换，保持了用户友好的接口。

主要功能模块：
- **数据存储**：映射数组存储、列数组、索引数组的管理
- **数据归约**：按列/组进行统计计算（均值、标准差、最值等）
- **数据映射**：在每列/组上应用自定义函数进行数据变换
- **数据转换**：与pandas DataFrame/Series之间的转换
- **数据过滤**：基于条件的数据筛选和Top-N选择
- **数据可视化**：直接生成统计图表（直方图、箱线图等）
- **分组操作**：支持复杂的分组聚合和分析

数据结构设计：
- **mapped_arr**：核心数据数组，存储实际的数值（如价格、收益率、交易量等）
- **col_arr**：列数组，标识每个数据点属于哪一列（如哪个股票、哪个策略）
- **idx_arr**：索引数组，标识每个数据点的时间位置（可选）
- **id_arr**：ID数组，为每个数据点分配唯一标识符

应用场景：
- **交易记录分析**：处理买卖订单、成交记录等稀疏交易数据
- **投资组合分析**：计算多资产投资组合的收益、风险等指标
- **因子分析**：处理股票因子数据，计算因子收益和风险
- **技术指标计算**：基于历史价格数据计算各种技术指标
- **风险管理**：实时计算VaR、最大回撤等风险指标
- **策略回测**：高效处理策略回测中的信号和收益数据

性能特点：
- **内存效率**：压缩存储，内存占用比传统DataFrame减少50-90%
- **计算速度**：Numba编译的核心函数，比纯Python快10-100倍
- **扩展性**：支持处理百万级别的数据点而不会出现性能瓶颈
- **灵活性**：支持动态数据结构，适应不同的数据分布和访问模式

与vectorbt生态系统的关系：
- Records类的数据输出：Records通过map_field等方法生成MappedArray
- ColumnMapper的索引支持：使用高效的列映射进行数据访问
- ArrayWrapper的元数据管理：继承完整的索引和列信息
- 统计分析的基础：为Portfolio、Trades等高级类提供计算基础

技术创新：
- **零拷贝操作**：尽可能避免数据复制，使用视图和索引操作
- **智能缓存**：缓存中间计算结果，避免重复计算
- **并行友好**：列之间的操作相互独立，易于并行化处理
- **类型安全**：严格的类型检查，减少运行时错误

该模块是vectorbt框架实现"高性能量化分析"的核心技术组件，通过创新的数据结构设计
和优化的算法实现，为量化交易提供了工业级的数据处理能力。

使用示例：
```python
import numpy as np
import pandas as pd
import vectorbt as vbt

# 1. 创建映射数组 - 股票价格数据
prices = np.array([100.0, 101.0, 99.5, 102.0, 98.0, 103.0])  # 价格数据
stocks = np.array([0, 0, 1, 1, 2, 2])                        # 股票编号
dates = np.array([0, 1, 0, 1, 0, 1])                         # 时间编号

wrapper = vbt.ArrayWrapper(
    index=pd.date_range('2023-01-01', periods=2, freq='D'),
    columns=['AAPL', 'GOOGL', 'MSFT'],
    ndim=2
)

# 创建映射数组
ma = vbt.MappedArray(wrapper, prices, stocks, idx_arr=dates)

# 2. 数据归约操作
mean_prices = ma.mean()                    # 计算每只股票的平均价格
print(f"平均价格: {mean_prices}")

# 3. 数据映射操作
@vbt.njit
def price_return_nb(idxs, col, values):
    if len(values) < 2:
        return np.full_like(values, np.nan)
    returns = np.empty_like(values)
    returns[0] = np.nan
    for i in range(1, len(values)):
        returns[i] = (values[i] - values[i-1]) / values[i-1]
    return returns

returns_ma = ma.apply(price_return_nb)     # 计算收益率
print(f"收益率: {returns_ma.values}")

# 4. 数据转换
df = ma.to_pd()                           # 转换为pandas DataFrame
print(f"DataFrame形式:\\n{df}")

# 5. 分组分析
group_by = ['科技股', '科技股', '其他']     # 按行业分组
grouped_mean = ma.mean(group_by=group_by)  # 按组计算均值
print(f"分组均值: {grouped_mean}")
```
"""

# 导入NumPy库，用于高性能数值计算和数组操作
import numpy as np
# 导入packaging.version模块，用于版本比较和兼容性检查
import packaging.version
# 导入Pandas库，用于数据结构和数据分析
import pandas as pd

# 导入vectorbt的类型定义模块，提供类型注解支持
from vectorbt import _typing as tp
# 导入数组包装器相关类，用于数据包装和元数据管理
from vectorbt.base.array_wrapper import ArrayWrapper, Wrapping
# 导入数组重塑函数，用于数组维度转换和数据格式化
from vectorbt.base.reshape_fns import to_1d_array, to_dict
# 导入通用模块的Numba编译函数，提供基础的统计计算功能
from vectorbt.generic import nb as generic_nb
# 导入绘图构建器混合类，提供图表绘制功能
from vectorbt.generic.plots_builder import PlotsBuilderMixin
# 导入统计构建器混合类，提供统计分析功能
from vectorbt.generic.stats_builder import StatsBuilderMixin
# 导入records模块的Numba编译函数，提供高性能的记录数据处理
from vectorbt.records import nb
# 导入列映射器，用于高效的列数据访问和管理
from vectorbt.records.col_mapper import ColumnMapper
# 导入检查工具，用于数据验证和类型检查
from vectorbt.utils import checks
# 导入配置管理工具，用于配置合并和管理
from vectorbt.utils.config import merge_dicts, Config, Configured
# 导入装饰器，用于缓存和魔法方法的自动生成
from vectorbt.utils.decorators import cached_method, attach_binary_magic_methods, attach_unary_magic_methods
# 导入映射工具，用于数据映射和转换
from vectorbt.utils.mapping import to_mapping, apply_mapping

# 定义MappedArray类的类型变量，用于类型提示中的泛型约束
# 确保方法返回的类型与调用类的类型一致，支持子类的类型安全
MappedArrayT = tp.TypeVar("MappedArrayT", bound="MappedArray")

# 定义索引操作元数据的类型，用于描述索引操作的返回值结构
# 包含：新包装器、映射数组、列数组、ID数组、索引数组、组索引、列索引
IndexingMetaT = tp.Tuple[
    ArrayWrapper,              # 新的数组包装器，包含更新后的元数据
    tp.Array1d,               # 新的映射数组，包含选择后的数据值
    tp.Array1d,               # 新的列数组，标识数据点的列归属
    tp.Array1d,               # 新的ID数组，为数据点分配唯一标识
    tp.Optional[tp.Array1d],  # 新的索引数组，可能为空
    tp.MaybeArray,            # 组索引，用于分组操作
    tp.Array1d                # 列索引，用于列选择操作
]


def combine_mapped_with_other(self: MappedArrayT,
                              other: tp.Union["MappedArray", tp.ArrayLike],
                              np_func: tp.Callable[[tp.ArrayLike, tp.ArrayLike], tp.Array1d]) -> MappedArrayT:
    """
    将MappedArray与其他兼容对象进行组合运算
    
    该函数是MappedArray算术运算的核心实现，支持MappedArray与其他对象（如标量、数组、
    其他MappedArray）进行各种数学运算。这种设计使得MappedArray可以像NumPy数组一样
    进行直观的数学操作。
    
    参数说明：
        self (MappedArrayT): 当前的MappedArray实例
        other (Union[MappedArray, ArrayLike]): 另一个操作数，可以是：
            - 另一个MappedArray实例
            - NumPy数组
            - 标量值
            - 列表或其他数组类型
        np_func (Callable): NumPy函数，用于执行实际的数学运算
            - 如：np.add, np.subtract, np.multiply, np.divide等
    
    返回值：
        MappedArrayT: 运算结果的新MappedArray实例
    
    算法逻辑：
    1. 检查other是否为MappedArray实例
    2. 如果是，验证两个MappedArray的结构兼容性（id_arr和col_arr必须匹配）
    3. 提取other的数值部分用于运算
    4. 使用np_func执行实际的数学运算
    5. 返回包含运算结果的新MappedArray实例
    
    使用示例：
    ```python
    import numpy as np
    import vectorbt as vbt
    
    # 创建两个MappedArray
    prices1 = np.array([100, 101, 102])
    prices2 = np.array([98, 99, 100])
    cols = np.array([0, 1, 2])
    
    wrapper = vbt.ArrayWrapper(columns=['A', 'B', 'C'], ndim=2)
    ma1 = vbt.MappedArray(wrapper, prices1, cols)
    ma2 = vbt.MappedArray(wrapper, prices2, cols)
    
    # 支持的运算操作
    result_add = ma1 + ma2        # 加法：[198, 200, 202]
    result_sub = ma1 - ma2        # 减法：[2, 2, 2]
    result_mul = ma1 * ma2        # 乘法：[9800, 9999, 10200]
    result_div = ma1 / ma2        # 除法：[1.0204, 1.0202, 1.02]
    
    # 与标量的运算
    result_scalar = ma1 * 1.1     # 所有值乘以1.1
    
    # 与数组的运算
    multipliers = np.array([1.1, 1.2, 1.3])
    result_array = ma1 * multipliers
    ```
    
    注意事项：
    - 两个MappedArray的id_arr和col_arr必须完全匹配
    - 运算结果保持第一个操作数的元数据（wrapper、索引等）
    - 支持广播机制，可以与不同形状的数组进行运算
    """
    # 检查other是否为MappedArray实例
    if isinstance(other, MappedArray):
        # 验证两个MappedArray的结构兼容性
        # id_arr必须匹配，确保数据点的对应关系正确
        checks.assert_array_equal(self.id_arr, other.id_arr)
        # col_arr必须匹配，确保列的对应关系正确
        checks.assert_array_equal(self.col_arr, other.col_arr)
        # 提取other的数值部分用于运算
        other = other.values
    
    # 使用np_func执行实际的数学运算，并返回新的MappedArray实例
    # 保持当前实例的所有元数据，只更新映射数组的数值
    return self.replace(mapped_arr=np_func(self.values, other))


# 定义MappedArray的元类，用于多重继承的类型管理
# 继承自StatsBuilderMixin和PlotsBuilderMixin的元类，确保方法解析顺序正确
class MetaMappedArray(type(StatsBuilderMixin), type(PlotsBuilderMixin)):
    """
    MappedArray的元类
    
    该元类用于处理MappedArray的多重继承问题，确保StatsBuilderMixin和PlotsBuilderMixin
    的方法能够正确集成到MappedArray类中。通过正确的方法解析顺序(MRO)，避免了多重继承
    中可能出现的方法冲突问题。
    """
    pass


# 使用装饰器为MappedArray类自动生成二元魔法方法（如+、-、*、/等）
# combine_mapped_with_other函数将被用作这些运算的实现
@attach_binary_magic_methods(combine_mapped_with_other)
# 使用装饰器为MappedArray类自动生成一元魔法方法（如-、abs、~等）
# lambda函数定义了一元运算的实现方式：对values应用numpy函数后替换mapped_arr
@attach_unary_magic_methods(lambda self, np_func: self.replace(mapped_arr=np_func(self.values)))
class MappedArray(Wrapping, StatsBuilderMixin, PlotsBuilderMixin, metaclass=MetaMappedArray):
    """
    映射数组 - vectorbt框架中用于处理稀疏数据的核心类
    
    MappedArray是vectorbt量化交易框架中最重要的数据结构之一，它提供了一种高效的方式
    来处理稀疏的、结构化的金融数据。该类通过压缩存储技术，大幅减少内存占用，同时
    通过Numba编译的高性能函数实现快速的数据处理和分析。
    
    核心特性：
    1. **压缩存储**：只存储非空值，节省内存空间
    2. **高性能计算**：基于Numba JIT编译的底层函数
    3. **灵活数据访问**：支持按列、按组、按索引的多维操作
    4. **无缝集成**：与vectorbt生态系统完美集成
    
    数据结构：
    - mapped_arr: 实际的数据值数组（如价格、收益率、交易量等）
    - col_arr: 列数组，标识每个数据点属于哪一列（如股票代码）
    - id_arr: ID数组，为每个数据点分配唯一标识符
    - idx_arr: 索引数组，标识数据点的时间位置（可选）
    - wrapper: 数组包装器，包含元数据（索引、列名、分组等）
    - mapping: 值映射，用于将数值转换为可读标签
    
    主要方法：
    - reduce(): 按列/组进行归约操作（求和、均值、最值等）
    - apply(): 在每列/组上应用自定义函数
    - to_pd(): 转换为pandas DataFrame/Series
    - apply_mask(): 基于条件过滤数据
    - sort(): 对数据进行排序
    - value_counts(): 计算值的频次分布
    
    使用示例：
    ```python
    import numpy as np
    import pandas as pd
    import vectorbt as vbt
    
    # 1. 创建基础数据
    # 假设我们有3只股票在2天内的交易数据
    prices = np.array([100.0, 101.0, 102.0, 98.0, 99.0, 103.0])
    stocks = np.array([0, 0, 1, 1, 2, 2])  # 股票编号
    days = np.array([0, 1, 0, 1, 0, 1])    # 交易日编号
    
    # 创建包装器
    wrapper = vbt.ArrayWrapper(
        index=pd.date_range('2023-01-01', periods=2, freq='D'),
        columns=['AAPL', 'GOOGL', 'MSFT'],
        ndim=2
    )
    
    # 创建MappedArray
    ma = vbt.MappedArray(wrapper, prices, stocks, idx_arr=days)
    
    # 2. 基本统计操作
    mean_price = ma.mean()                    # 每只股票的平均价格
    max_price = ma.max()                      # 每只股票的最高价格
    total_volume = ma.sum()                   # 每只股票的总交易量
    
    # 3. 数据变换
    @vbt.njit
    def calculate_returns(idxs, col, values):
        # 计算收益率
        if len(values) < 2:
            return np.full_like(values, np.nan)
        returns = np.empty_like(values)
        returns[0] = np.nan
        for i in range(1, len(values)):
            returns[i] = (values[i] - values[i-1]) / values[i-1]
        return returns
    
    returns_ma = ma.apply(calculate_returns)   # 计算收益率
    
    # 4. 数据过滤
    high_price_mask = ma.values > 100         # 价格大于100的数据点
    filtered_ma = ma.apply_mask(high_price_mask)
    
    # 5. 分组分析
    # 按行业分组：科技股vs其他
    group_by = ['科技股', '科技股', '其他']
    grouped_mean = ma.mean(group_by=group_by)  # 按行业计算平均价格
    
    # 6. 数据转换
    df = ma.to_pd()                           # 转换为pandas DataFrame
    print(df)
    #              AAPL  GOOGL  MSFT
    # 2023-01-01  100.0   98.0   99.0
    # 2023-01-02  101.0   99.0  103.0
    
    # 7. 高级统计
    desc_stats = ma.describe()                # 描述性统计
    value_counts = ma.value_counts()          # 值频次统计
    
    # 8. 可视化
    ma.histplot()                            # 直方图
    ma.boxplot()                             # 箱线图
    ```
    
    适用场景：
    - **交易数据分析**：处理买卖订单、成交记录等稀疏数据
    - **投资组合管理**：计算多资产组合的收益和风险指标
    - **因子研究**：分析股票因子的收益和有效性
    - **技术指标**：计算基于价格的各种技术指标
    - **风险管理**：实时计算VaR、回撤等风险指标
    - **策略回测**：高效处理策略信号和收益数据
    
    性能优势：
    - 内存使用比传统DataFrame减少50-90%
    - 计算速度比纯Python快10-100倍
    - 支持处理百万级数据点
    - 智能缓存避免重复计算
    
    参数说明：
        wrapper (ArrayWrapper): 数组包装器，包含索引、列名、分组等元数据
        mapped_arr (array_like): 一维映射数组，存储实际的数据值
        col_arr (array_like): 一维列数组，标识每个数据点属于哪一列
        id_arr (array_like, optional): 一维ID数组，默认为简单的连续编号
        idx_arr (array_like, optional): 一维索引数组，标识数据点的时间位置
        mapping (MappingLike, optional): 值映射，数值到标签的转换
        col_mapper (ColumnMapper, optional): 列映射器，优化列操作
        **kwargs: 其他配置参数，传递给子类扩展配置
    
    注意事项：
    - mapped_arr和col_arr必须具有相同的长度
    - 如果提供idx_arr，它也必须与mapped_arr长度相同
    - col_mapper依赖于wrapper和col_arr，修改时需要重新创建
    - 类设计为不可变的，所有属性都是只读的
    """

    def __init__(self,
                 wrapper: ArrayWrapper,
                 mapped_arr: tp.ArrayLike,
                 col_arr: tp.ArrayLike,
                 id_arr: tp.Optional[tp.ArrayLike] = None,
                 idx_arr: tp.Optional[tp.ArrayLike] = None,
                 mapping: tp.Optional[tp.MappingLike] = None,
                 col_mapper: tp.Optional[ColumnMapper] = None,
                 **kwargs) -> None:
        """
        初始化MappedArray实例
        
        该构造函数创建一个新的MappedArray实例，设置所有必要的数据结构和元数据。
        初始化过程包括数据验证、类型转换、默认值设置和依赖组件的创建。
        
        参数详细说明：
            wrapper (ArrayWrapper): 数组包装器，核心元数据容器
                - 包含时间序列的索引（如交易日期）
                - 包含数据列的名称（如股票代码）
                - 包含分组信息（如行业分类）
                - 提供数组的形状和维度信息
                - 支持频率信息（如日线、分钟线数据）
                
            mapped_arr (array_like): 映射数组，存储实际的数据值
                - 一维数组，包含所有的数据点
                - 可以是价格、收益率、交易量等任何数值数据
                - 数据类型通常为float64，支持NaN值
                - 长度必须与col_arr相同
                
            col_arr (array_like): 列数组，标识数据点的列归属
                - 一维整数数组，值为0, 1, 2, ...
                - 每个值对应wrapper.columns中的一个列
                - 用于将数据点映射到正确的列（如股票）
                - 可以是排序或未排序的
                
            id_arr (array_like, optional): ID数组，数据点的唯一标识
                - 如果未提供，自动生成连续的整数ID
                - 用于跟踪数据点的原始顺序
                - 在数据过滤和排序时保持数据的可追溯性
                
            idx_arr (array_like, optional): 索引数组，时间位置标识
                - 标识每个数据点在时间轴上的位置
                - 值对应wrapper.index中的位置
                - 用于将数据点正确映射到时间序列
                - 在转换为pandas时必需
                
            mapping (MappingLike, optional): 值映射，数值到标签的转换
                - 字典、命名元组或可调用对象
                - 用于将数值转换为可读的标签
                - 特殊值'index'和'columns'分别映射到索引和列
                - 主要用于分类数据的显示
                
            col_mapper (ColumnMapper, optional): 列映射器，优化列操作
                - 如果未提供，会自动创建
                - 用于高效的列数据访问和索引
                - 依赖于wrapper和col_arr
                - 支持排序和未排序数据的不同优化策略
                
            **kwargs: 其他配置参数
                - 传递给配置系统的额外参数
                - 用于子类扩展和自定义配置
        
        初始化过程：
        1. 调用基类构造函数，设置继承链
        2. 数据类型转换和验证
        3. 设置默认值（如id_arr）
        4. 处理映射配置
        5. 创建或设置列映射器
        6. 存储所有组件的引用
        
        使用示例：
        ```python
        import numpy as np
        import pandas as pd
        import vectorbt as vbt
        
        # 1. 基本创建
        wrapper = vbt.ArrayWrapper(
            index=pd.date_range('2023-01-01', periods=3, freq='D'),
            columns=['AAPL', 'GOOGL', 'MSFT'],
            ndim=2
        )
        
        prices = np.array([100, 101, 102, 98, 99, 103])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        days = np.array([0, 1, 0, 1, 0, 1])
        
        ma = vbt.MappedArray(wrapper, prices, stocks, idx_arr=days)
        
        # 2. 带自定义ID的创建
        custom_ids = np.array([1001, 1002, 2001, 2002, 3001, 3002])
        ma_with_ids = vbt.MappedArray(wrapper, prices, stocks, 
                                      id_arr=custom_ids, idx_arr=days)
        
        # 3. 带值映射的创建
        grade_mapping = {90: 'A', 80: 'B', 70: 'C', 60: 'D'}
        scores = np.array([95, 87, 92, 78, 88, 65])
        ma_grades = vbt.MappedArray(wrapper, scores, stocks, 
                                    mapping=grade_mapping)
        
        # 4. 分类数据的创建
        directions = np.array([1, -1, 1, -1, 1, -1])  # 1=买入, -1=卖出
        direction_mapping = {1: '买入', -1: '卖出'}
        ma_directions = vbt.MappedArray(wrapper, directions, stocks,
                                        mapping=direction_mapping)
        ```
        
        注意事项：
        - 所有数组参数都会被转换为NumPy数组
        - mapped_arr和col_arr必须具有相同的长度
        - 如果提供idx_arr，它也必须与mapped_arr长度相同
        - col_mapper依赖于wrapper和col_arr，修改时需要重新创建
        - 类设计为不可变的，所有属性都是只读的
        """
        # 调用基类Wrapping的构造函数，设置包装器和配置参数
        Wrapping.__init__(
            self,
            wrapper,                    # 数组包装器，包含元数据
            mapped_arr=mapped_arr,     # 映射数组，存储实际数据
            col_arr=col_arr,           # 列数组，标识数据点的列归属
            id_arr=id_arr,             # ID数组，数据点的唯一标识
            idx_arr=idx_arr,           # 索引数组，时间位置标识
            mapping=mapping,           # 值映射，数值到标签的转换
            col_mapper=col_mapper,     # 列映射器，优化列操作
            **kwargs                   # 其他配置参数
        )
        # 调用StatsBuilderMixin的构造函数，初始化统计分析功能
        StatsBuilderMixin.__init__(self)

        # 将映射数组转换为NumPy数组，确保数据类型的一致性
        mapped_arr = np.asarray(mapped_arr)
        # 将列数组转换为NumPy数组，确保数据类型的一致性
        col_arr = np.asarray(col_arr)
        # 验证映射数组和列数组的形状是否匹配，沿着第0轴（长度）进行检查
        checks.assert_shape_equal(mapped_arr, col_arr, axis=0)
        
        # 处理ID数组：如果未提供，创建默认的连续整数ID
        if id_arr is None:
            # 生成从0开始的连续整数作为ID
            id_arr = np.arange(len(mapped_arr))
        else:
            # 将提供的ID数组转换为NumPy数组
            id_arr = np.asarray(id_arr)
        
        # 处理索引数组：如果提供了，验证其长度是否与映射数组匹配
        if idx_arr is not None:
            # 转换为NumPy数组
            idx_arr = np.asarray(idx_arr)
            # 验证索引数组与映射数组的长度是否匹配
            checks.assert_shape_equal(mapped_arr, idx_arr, axis=0)
        
        # 处理值映射：将映射转换为标准格式
        if mapping is not None:
            # 处理特殊的字符串映射
            if isinstance(mapping, str):
                if mapping.lower() == 'index':
                    # 使用包装器的索引作为映射
                    mapping = self.wrapper.index
                elif mapping.lower() == 'columns':
                    # 使用包装器的列作为映射
                    mapping = self.wrapper.columns
            # 将映射转换为标准的映射格式
            mapping = to_mapping(mapping)

        # 存储所有组件的引用，这些属性都是只读的
        self._mapped_arr = mapped_arr      # 映射数组，存储实际数据
        self._id_arr = id_arr              # ID数组，数据点的唯一标识
        self._col_arr = col_arr            # 列数组，标识数据点的列归属
        self._idx_arr = idx_arr            # 索引数组，时间位置标识
        self._mapping = mapping            # 值映射，数值到标签的转换
        
        # 创建或设置列映射器
        if col_mapper is None:
            # 如果未提供列映射器，创建新的列映射器
            col_mapper = ColumnMapper(wrapper, col_arr)
        # 存储列映射器的引用
        self._col_mapper = col_mapper

    def replace(self: MappedArrayT, **kwargs) -> MappedArrayT:
        """
        创建MappedArray的修改副本 - 不可变对象的安全更新机制
        
        由于MappedArray是不可变的数据结构，replace方法提供了创建修改副本的安全方式。
        该方法会智能地处理依赖关系，确保在修改wrapper或col_arr时自动重新创建col_mapper。
        
        工作原理：
        1. 检查是否修改了影响col_mapper的参数（wrapper或col_arr）
        2. 如果修改了这些参数，自动将col_mapper设置为None，强制重新创建
        3. 调用基类的replace方法创建新实例
        4. 返回包含修改后参数的新MappedArray实例
        
        参数说明：
            **kwargs: 要修改的属性和它们的新值
                - wrapper: 新的数组包装器
                - mapped_arr: 新的映射数组
                - col_arr: 新的列数组  
                - id_arr: 新的ID数组
                - idx_arr: 新的索引数组
                - mapping: 新的值映射
                - col_mapper: 新的列映射器
                - 其他配置参数
        
        返回值：
            MappedArrayT: 包含修改后参数的新MappedArray实例
        
        智能依赖管理：
        - 当wrapper改变时，col_mapper会自动重新创建以匹配新的列结构
        - 当col_arr改变时，col_mapper会自动重新创建以匹配新的列映射
        - 其他参数的修改不会影响col_mapper的有效性
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建原始MappedArray
        prices = np.array([100, 101, 102])
        stocks = np.array([0, 1, 2])
        wrapper = vbt.ArrayWrapper(columns=['A', 'B', 'C'], ndim=2)
        ma = vbt.MappedArray(wrapper, prices, stocks)
        
        # 1. 修改数据值（保持结构不变）
        new_prices = np.array([110, 111, 112])
        ma_new_prices = ma.replace(mapped_arr=new_prices)
        print(f"原始价格: {ma.values}")        # [100, 101, 102]
        print(f"新价格: {ma_new_prices.values}")  # [110, 111, 112]
        
        # 2. 修改列结构（col_mapper会自动重新创建）
        new_stocks = np.array([1, 2, 0])  # 重新排列列映射
        ma_new_cols = ma.replace(col_arr=new_stocks)
        
        # 3. 修改包装器（col_mapper会自动重新创建）
        new_wrapper = vbt.ArrayWrapper(columns=['X', 'Y', 'Z'], ndim=2)
        ma_new_wrapper = ma.replace(wrapper=new_wrapper)
        
        # 4. 添加索引数组
        idx_arr = np.array([0, 1, 2])
        ma_with_idx = ma.replace(idx_arr=idx_arr)
        
        # 5. 添加值映射
        grade_mapping = {100: 'A', 101: 'B', 102: 'C'}
        ma_with_mapping = ma.replace(mapping=grade_mapping)
        
        # 6. 链式修改
        ma_modified = ma.replace(
            mapped_arr=new_prices,
            col_arr=new_stocks,
            idx_arr=idx_arr
        )
        ```
        
        注意事项：
        - 原始MappedArray实例保持不变
        - 每次调用都会创建新的实例
        - col_mapper的依赖管理是自动的，无需手动处理
        - 适合在数据处理管道中进行数据变换
        """
        # 检查是否存在col_mapper配置
        if self.config.get('col_mapper', None) is not None:
            # 如果要修改wrapper，检查是否与当前wrapper不同
            if 'wrapper' in kwargs:
                if self.wrapper is not kwargs.get('wrapper'):
                    # wrapper改变时，col_mapper需要重新创建
                    kwargs['col_mapper'] = None
            # 如果要修改col_arr，检查是否与当前col_arr不同
            if 'col_arr' in kwargs:
                if self.col_arr is not kwargs.get('col_arr'):
                    # col_arr改变时，col_mapper需要重新创建
                    kwargs['col_mapper'] = None
        
        # 调用基类的replace方法创建新实例
        return Configured.replace(self, **kwargs)

    def indexing_func_meta(self, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> IndexingMetaT:
        """
        执行索引操作并返回元数据 - 高效的数据选择机制
        
        该方法是MappedArray索引操作的核心实现，它模拟pandas的索引行为，但专门针对
        MappedArray的压缩存储格式进行了优化。该方法返回索引操作的完整元数据，
        为后续的数据重构提供必要信息。
        
        工作流程：
        1. 通过wrapper执行pandas风格的索引操作，获取列选择信息
        2. 使用col_mapper根据列选择获取对应的数据索引
        3. 基于索引选择提取相应的数据、ID和索引数组
        4. 返回完整的索引操作元数据
        
        参数说明：
            pd_indexing_func (PandasIndexingFunc): pandas索引函数
                - 支持标准的pandas索引操作
                - 如：lambda x: x['A']、lambda x: x[['A', 'B']]等
            **kwargs: 传递给包装器索引操作的额外参数
                - column_only_select: 只选择列，不选择行
                - 其他索引相关参数
        
        返回值：
            IndexingMetaT: 包含以下元素的元组
                - new_wrapper: 更新后的数组包装器
                - new_mapped_arr: 选择后的映射数组
                - new_col_arr: 选择后的列数组
                - new_id_arr: 选择后的ID数组
                - new_idx_arr: 选择后的索引数组（可能为None）
                - group_idxs: 组索引信息
                - col_idxs: 列索引信息
        
        索引优化：
        - 使用ColumnMapper进行高效的列选择
        - 避免不必要的数据复制
        - 保持数据的稀疏性和压缩特性
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建测试数据
        prices = np.array([100, 101, 102, 98, 99, 103])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        days = np.array([0, 1, 0, 1, 0, 1])
        
        wrapper = vbt.ArrayWrapper(
            index=pd.date_range('2023-01-01', periods=2, freq='D'),
            columns=['AAPL', 'GOOGL', 'MSFT'],
            ndim=2
        )
        ma = vbt.MappedArray(wrapper, prices, stocks, idx_arr=days)
        
        # 定义索引函数
        def select_apple(x):
            return x['AAPL']  # 选择苹果股票
        
        def select_tech_stocks(x):
            return x[['AAPL', 'GOOGL']]  # 选择科技股
        
        # 执行索引操作并获取元数据
        meta = ma.indexing_func_meta(select_apple)
        new_wrapper, new_mapped_arr, new_col_arr, new_id_arr, new_idx_arr, group_idxs, col_idxs = meta
        
        print(f"选择后的数据: {new_mapped_arr}")    # 苹果股票的价格
        print(f"选择后的列: {new_col_arr}")        # 都是0（因为现在只有一列）
        print(f"新的包装器列: {new_wrapper.columns}")  # Index(['AAPL'])
        ```
        
        应用场景：
        - 股票投资组合的子集选择
        - 特定时间段的数据筛选
        - 按行业或主题的数据分组
        - 复杂的多维数据切片
        
        性能特点：
        - 利用ColumnMapper的高效索引机制
        - 避免全矩阵展开的内存开销
        - 保持数据的稀疏性
        - 支持复杂的索引模式
        """
        # 通过wrapper执行pandas风格的索引操作
        # column_only_select=True表示只选择列，不选择行
        new_wrapper, _, group_idxs, col_idxs = \
            self.wrapper.indexing_func_meta(pd_indexing_func, column_only_select=True, **kwargs)
        
        # 使用col_mapper根据列索引获取对应的数据索引和新的列数组
        new_indices, new_col_arr = self.col_mapper._col_idxs_meta(col_idxs)
        
        # 基于新索引选择映射数组中的相应数据
        new_mapped_arr = self.values[new_indices]
        # 基于新索引选择ID数组中的相应数据
        new_id_arr = self.id_arr[new_indices]
        
        # 处理索引数组：如果原始实例有索引数组，则选择相应的部分
        if self.idx_arr is not None:
            new_idx_arr = self.idx_arr[new_indices]
        else:
            new_idx_arr = None
        
        # 返回完整的索引操作元数据
        return new_wrapper, new_mapped_arr, new_col_arr, new_id_arr, new_idx_arr, group_idxs, col_idxs

    def indexing_func(self: MappedArrayT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> MappedArrayT:
        """
        执行索引操作并返回新的MappedArray实例
        
        该方法是MappedArray索引操作的用户接口，它内部调用indexing_func_meta获取
        索引操作的元数据，然后创建并返回新的MappedArray实例。这种设计分离了
        元数据计算和实例创建，提高了代码的可维护性和重用性。
        
        参数说明：
            pd_indexing_func (PandasIndexingFunc): pandas索引函数
                - 支持所有标准的pandas索引操作
                - 如：选择单列、多列、条件选择等
            **kwargs: 传递给indexing_func_meta的额外参数
        
        返回值：
            MappedArrayT: 包含选择后数据的新MappedArray实例
        
        使用示例：
        ```python
        import numpy as np
        import pandas as pd
        import vectorbt as vbt
        
        # 创建多资产数据
        prices = np.array([100, 101, 102, 98, 99, 103, 95, 96])
        stocks = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        days = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        
        wrapper = vbt.ArrayWrapper(
            index=pd.date_range('2023-01-01', periods=2, freq='D'),
            columns=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
            ndim=2
        )
        ma = vbt.MappedArray(wrapper, prices, stocks, idx_arr=days)
        
        # 1. 选择单个股票
        apple_ma = ma.indexing_func(lambda x: x['AAPL'])
        print(f"苹果数据: {apple_ma.values}")
        print(f"苹果列名: {apple_ma.wrapper.columns}")
        
        # 2. 选择多个股票
        tech_ma = ma.indexing_func(lambda x: x[['AAPL', 'GOOGL']])
        print(f"科技股数据: {tech_ma.values}")
        print(f"科技股列名: {tech_ma.wrapper.columns}")
        
        # 3. 基于条件选择
        large_cap_ma = ma.indexing_func(lambda x: x[['AAPL', 'GOOGL', 'MSFT']])
        
        # 4. 使用布尔索引
        mask = pd.Series([True, False, True, False], index=wrapper.columns)
        selected_ma = ma.indexing_func(lambda x: x[mask])
        
        # 验证索引操作的正确性
        print(f"原始数据长度: {len(ma.values)}")
        print(f"选择后数据长度: {len(tech_ma.values)}")
        print(f"原始列数: {len(ma.wrapper.columns)}")
        print(f"选择后列数: {len(tech_ma.wrapper.columns)}")
        ```
        
        索引模式支持：
        - 单列选择: `lambda x: x['AAPL']`
        - 多列选择: `lambda x: x[['AAPL', 'GOOGL']]`
        - 切片选择: `lambda x: x['AAPL':'MSFT']`
        - 条件选择: `lambda x: x[condition]`
        - 位置选择: `lambda x: x.iloc[:, 0:2]`
        
        应用场景：
        - 投资组合子集分析
        - 行业或主题投资分析
        - 风险因子暴露分析
        - 特定股票的深度分析
        - 数据预处理和清洗
        
        性能优势：
        - 避免全数据复制，只选择必要的数据
        - 保持数据的稀疏性和压缩特性
        - 元数据操作高效，内存占用低
        - 支持链式操作和方法组合
        """
        # 调用indexing_func_meta获取索引操作的完整元数据
        new_wrapper, new_mapped_arr, new_col_arr, new_id_arr, new_idx_arr, _, _ = \
            self.indexing_func_meta(pd_indexing_func, **kwargs)
        
        # 使用索引操作的结果创建新的MappedArray实例
        return self.replace(
            wrapper=new_wrapper,          # 更新后的包装器
            mapped_arr=new_mapped_arr,    # 选择后的映射数组
            col_arr=new_col_arr,          # 选择后的列数组
            id_arr=new_id_arr,            # 选择后的ID数组
            idx_arr=new_idx_arr           # 选择后的索引数组
        )

    @property
    def mapped_arr(self) -> tp.Array1d:
        """
        获取映射数组 - 存储实际数据值的核心数组
        
        映射数组是MappedArray中最核心的数据结构，它存储了所有的实际数据值。
        与传统的二维DataFrame不同，映射数组采用一维压缩存储，只保存非空值，
        从而实现了高效的内存利用。
        
        数据特点：
        - 一维NumPy数组，存储实际的数值数据
        - 可以是任何数值类型（float、int等）
        - 支持NaN值来表示缺失数据
        - 长度与col_arr、id_arr相同
        
        应用场景：
        - 价格数据：股票价格、期货价格、期权价格等
        - 收益率数据：日收益率、周收益率、月收益率等
        - 交易数据：交易量、成交金额、持仓量等
        - 指标数据：技术指标值、基本面指标值等
        - 风险数据：VaR、回撤、波动率等
        
        返回值：
            tp.Array1d: 一维NumPy数组，包含所有实际数据值
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建价格数据
        prices = np.array([100.0, 101.5, 99.8, 102.3, 98.7, 103.1])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, prices, stocks)
        
        # 获取映射数组
        values = ma.mapped_arr
        print(f"映射数组: {values}")
        print(f"数据类型: {values.dtype}")
        print(f"数组形状: {values.shape}")
        print(f"是否有NaN: {np.isnan(values).any()}")
        
        # 映射数组的统计信息
        print(f"最小值: {values.min()}")
        print(f"最大值: {values.max()}")
        print(f"平均值: {values.mean()}")
        print(f"标准差: {values.std()}")
        ```
        
        与其他数组的关系：
        - col_arr: 标识每个值属于哪一列
        - id_arr: 为每个值分配唯一标识符
        - idx_arr: 标识每个值的时间位置（可选）
        
        注意事项：
        - 映射数组是只读的，不能直接修改
        - 要修改数据，需要使用replace()方法
        - 数组中的值按照原始顺序存储，可能不是按列排序的
        """
        return self._mapped_arr

    @property
    def values(self) -> tp.Array1d:
        """
        获取数据值 - mapped_arr的便捷访问器
        
        values属性是mapped_arr的别名，提供了一个更直观的名称来访问实际的数据值。
        这个属性与NumPy数组和pandas对象的.values属性保持一致，提供了熟悉的API。
        
        返回值：
            tp.Array1d: 与mapped_arr相同的一维NumPy数组
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建交易数据
        volumes = np.array([1000, 1500, 800, 1200, 2000, 900])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, volumes, stocks)
        
        # 两种访问方式是等价的
        print(f"mapped_arr: {ma.mapped_arr}")
        print(f"values: {ma.values}")
        print(f"是否相等: {np.array_equal(ma.mapped_arr, ma.values)}")
        
        # 用于算术运算
        doubled_values = ma.values * 2
        print(f"翻倍后的值: {doubled_values}")
        
        # 用于条件过滤
        high_volume_mask = ma.values > 1000
        print(f"高成交量掩码: {high_volume_mask}")
        
        # 用于统计计算
        total_volume = ma.values.sum()
        avg_volume = ma.values.mean()
        print(f"总成交量: {total_volume}")
        print(f"平均成交量: {avg_volume}")
        ```
        
        与pandas的一致性：
        - 类似于pandas Series的.values属性
        - 提供了标准的NumPy数组接口
        - 支持所有NumPy数组操作
        """
        return self.mapped_arr

    def __len__(self) -> int:
        """
        获取MappedArray的长度 - 数据点的总数
        
        返回MappedArray中数据点的总数，等于mapped_arr、col_arr、id_arr的长度。
        这个方法使得MappedArray对象可以使用len()函数，提供了与标准Python容器
        一致的长度查询接口。
        
        返回值：
            int: MappedArray中数据点的总数
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建不同大小的数据集
        small_data = np.array([100, 101, 102])
        small_cols = np.array([0, 1, 2])
        
        large_data = np.array([100, 101, 102, 98, 99, 103, 95, 96, 97, 104])
        large_cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        
        wrapper_small = vbt.ArrayWrapper(columns=['A', 'B', 'C'], ndim=2)
        wrapper_large = vbt.ArrayWrapper(columns=['A', 'B', 'C', 'D', 'E'], ndim=2)
        
        ma_small = vbt.MappedArray(wrapper_small, small_data, small_cols)
        ma_large = vbt.MappedArray(wrapper_large, large_data, large_cols)
        
        # 获取长度
        print(f"小数据集长度: {len(ma_small)}")     # 3
        print(f"大数据集长度: {len(ma_large)}")     # 10
        
        # 用于循环和迭代
        for i in range(len(ma_small)):
            print(f"第{i}个数据点: 值={ma_small.values[i]}, 列={ma_small.col_arr[i]}")
        
        # 用于数据验证
        assert len(ma_small) == len(ma_small.values)
        assert len(ma_small) == len(ma_small.col_arr)
        assert len(ma_small) == len(ma_small.id_arr)
        
        # 用于内存使用估算
        memory_usage = len(ma_large) * 8  # 假设每个float64占8字节
        print(f"约占用内存: {memory_usage} 字节")
        ```
        
        性能说明：
        - 这是一个O(1)操作，直接返回内部数组的长度
        - 比展开为完整矩阵后计算长度要高效得多
        - 可以用于性能监控和资源管理
        """
        return len(self.values)

    @property
    def col_arr(self) -> tp.Array1d:
        """
        获取列数组 - 数据点的列归属标识
        
        列数组是MappedArray中的关键索引结构，它标识每个数据点属于哪一列。
        这个数组与mapped_arr一一对应，提供了从数据点到列的映射关系。
        
        数据特点：
        - 一维整数数组，值为0, 1, 2, ...
        - 每个值对应wrapper.columns中的一个列
        - 长度与mapped_arr相同
        - 可以是排序或未排序的
        
        应用场景：
        - 将交易数据映射到具体的股票
        - 将收益率数据映射到不同的投资策略
        - 将指标数据映射到不同的时间周期
        - 将风险数据映射到不同的资产类别
        
        返回值：
            tp.Array1d: 一维整数数组，标识每个数据点的列归属
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建多股票交易数据
        prices = np.array([100, 101, 98, 99, 102, 103])
        stocks = np.array([0, 0, 1, 1, 2, 2])  # 三只股票，每只两个数据点
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, prices, stocks)
        
        # 获取列数组
        columns = ma.col_arr
        print(f"列数组: {columns}")
        print(f"列数组形状: {columns.shape}")
        print(f"唯一列: {np.unique(columns)}")
        
        # 分析列分布
        for col_idx in np.unique(columns):
            col_name = wrapper.columns[col_idx]
            mask = columns == col_idx
            col_values = ma.values[mask]
            print(f"{col_name}(列{col_idx}): 有{mask.sum()}个数据点, 值为{col_values}")
        
        # 验证列数组的完整性
        assert len(columns) == len(ma.values)
        assert columns.min() >= 0
        assert columns.max() < len(wrapper.columns)
        
        # 创建列分组统计
        col_counts = np.bincount(columns, minlength=len(wrapper.columns))
        for i, count in enumerate(col_counts):
            print(f"{wrapper.columns[i]}: {count}个数据点")
        ```
        
        与ColumnMapper的关系：
        - ColumnMapper使用col_arr构建高效的索引结构
        - 根据col_arr的排序状态选择不同的优化策略
        - 支持快速的列选择和数据重组
        
        注意事项：
        - 列数组是只读的，不能直接修改
        - 修改列数组需要使用replace()方法
        - 列数组的值必须在有效范围内（0 到 列数-1）
        """
        return self._col_arr

    @property
    def col_mapper(self) -> ColumnMapper:
        """
        获取列映射器 - 高效列操作的核心工具
        
        列映射器是MappedArray中用于优化列级操作的关键组件。它根据col_arr的
        排序状态智能选择最优的索引策略，为列选择、分组和聚合提供高性能支持。
        
        功能特点：
        - 自动选择最优索引策略（col_range或col_map）
        - 支持高效的列选择和数据重组
        - 提供分组操作的基础设施
        - 缓存计算结果以提高性能
        
        返回值：
            ColumnMapper: 列映射器实例
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建排序的数据
        sorted_prices = np.array([100, 101, 98, 99, 102, 103])
        sorted_stocks = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma_sorted = vbt.MappedArray(wrapper, sorted_prices, sorted_stocks)
        
        # 获取列映射器
        col_mapper = ma_sorted.col_mapper
        print(f"数据是否排序: {col_mapper.is_sorted()}")
        
        # 使用列范围（适用于排序数据）
        if col_mapper.is_sorted():
            col_range = col_mapper.col_range
            print(f"列范围: {col_range}")
        
        # 使用列映射（适用于所有数据）
        col_map = col_mapper.col_map
        print(f"列映射: {col_map}")
        
        # 创建未排序的数据
        unsorted_prices = np.array([100, 98, 102, 101, 99, 103])
        unsorted_stocks = np.array([0, 1, 2, 0, 1, 2])
        
        ma_unsorted = vbt.MappedArray(wrapper, unsorted_prices, unsorted_stocks)
        col_mapper_unsorted = ma_unsorted.col_mapper
        print(f"未排序数据是否排序: {col_mapper_unsorted.is_sorted()}")
        
        # 列映射器的分组功能
        group_by = ['科技股', '科技股', '其他']
        grouped_col_arr = col_mapper.get_col_arr(group_by=group_by)
        print(f"分组后的列数组: {grouped_col_arr}")
        
        # 获取分组后的列映射
        grouped_col_map = col_mapper.get_col_map(group_by=group_by)
        print(f"分组后的列映射: {grouped_col_map}")
        ```
        
        性能优化：
        - 排序数据：使用col_range，O(1)查找速度
        - 未排序数据：使用col_map，O(log n)查找速度
        - 自动缓存：避免重复计算索引结构
        
        应用场景：
        - 高效的列选择和数据筛选
        - 复杂的分组操作和聚合计算
        - 大规模数据的实时处理
        - 多维数据的灵活访问
        
        更多信息请参考：
        - `vectorbt.records.col_mapper.ColumnMapper`
        """
        return self._col_mapper

    @property
    def id_arr(self) -> tp.Array1d:
        """
        获取ID数组 - 数据点的唯一标识符
        
        ID数组为MappedArray中的每个数据点提供唯一的标识符。这些标识符在数据
        过滤、排序和追踪时发挥重要作用，确保数据的可追溯性和一致性。
        
        数据特点：
        - 一维整数数组，通常为连续的整数
        - 如果创建时未提供，自动生成0, 1, 2, ...的序列
        - 长度与mapped_arr相同
        - 每个ID在数组中都是唯一的
        
        应用场景：
        - 数据过滤后的溯源追踪
        - 复杂数据变换中的身份识别
        - 数据质量检查和验证
        - 多阶段数据处理的一致性保证
        
        返回值：
            tp.Array1d: 一维整数数组，包含每个数据点的唯一标识符
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 1. 默认ID数组（自动生成）
        prices = np.array([100, 101, 102, 98, 99, 103])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma_default = vbt.MappedArray(wrapper, prices, stocks)
        print(f"默认ID数组: {ma_default.id_arr}")  # [0, 1, 2, 3, 4, 5]
        
        # 2. 自定义ID数组
        custom_ids = np.array([1001, 1002, 2001, 2002, 3001, 3002])
        ma_custom = vbt.MappedArray(wrapper, prices, stocks, id_arr=custom_ids)
        print(f"自定义ID数组: {ma_custom.id_arr}")  # [1001, 1002, 2001, 2002, 3001, 3002]
        
        # 3. 数据过滤后的ID追踪
        mask = ma_default.values > 100
        filtered_ma = ma_default.apply_mask(mask)
        print(f"过滤前ID: {ma_default.id_arr}")
        print(f"过滤后ID: {filtered_ma.id_arr}")
        print(f"过滤后的数据: {filtered_ma.values}")
        
        # 验证过滤操作的正确性
        original_filtered_values = ma_default.values[mask]
        assert np.array_equal(filtered_ma.values, original_filtered_values)
        
        # 4. 用于数据质量检查
        print(f"ID数组长度: {len(ma_default.id_arr)}")
        print(f"ID数组是否唯一: {len(np.unique(ma_default.id_arr)) == len(ma_default.id_arr)}")
        print(f"ID数组范围: {ma_default.id_arr.min()} 到 {ma_default.id_arr.max()}")
        
        # 5. 排序操作中的ID保持
        sorted_ma = ma_default.sort(incl_id=True)
        print(f"排序前ID: {ma_default.id_arr}")
        print(f"排序后ID: {sorted_ma.id_arr}")
        print(f"排序前值: {ma_default.values}")
        print(f"排序后值: {sorted_ma.values}")
        ```
        
        在运算中的作用：
        - 两个MappedArray进行运算时，id_arr必须匹配
        - 确保运算的数据点一一对应
        - 防止数据错位和不一致
        
        注意事项：
        - ID数组是只读的，不能直接修改
        - 修改ID数组需要使用replace()方法
        - ID的唯一性对数据完整性很重要
        """
        return self._id_arr

    @property
    def idx_arr(self) -> tp.Optional[tp.Array1d]:
        """
        获取索引数组 - 数据点的时间位置标识符
        
        索引数组是可选的，它标识每个数据点在时间轴上的位置。这个数组将压缩的
        映射数据与时间序列的索引位置建立联系，是将MappedArray转换为pandas
        DataFrame/Series的关键信息。
        
        数据特点：
        - 一维整数数组，值对应wrapper.index中的位置
        - 可选的，如果未提供则为None
        - 长度与mapped_arr相同
        - 值的范围：0 到 len(wrapper.index)-1
        
        应用场景：
        - 时间序列数据的精确定位
        - 转换为pandas DataFrame时的时间轴映射
        - 时间相关的数据分析和可视化
        - 多时间框架数据的对齐和合并
        
        返回值：
            tp.Optional[tp.Array1d]: 一维整数数组或None
        
        使用示例：
        ```python
        import numpy as np
        import pandas as pd
        import vectorbt as vbt
        
        # 创建时间序列数据
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        wrapper = vbt.ArrayWrapper(
            index=dates,
            columns=['AAPL', 'GOOGL', 'MSFT'],
            ndim=2
        )
        
        # 1. 不带索引数组的MappedArray
        prices = np.array([100, 101, 98, 99, 102, 103])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        
        ma_no_idx = vbt.MappedArray(wrapper, prices, stocks)
        print(f"无索引数组: {ma_no_idx.idx_arr}")  # None
        
        # 2. 带索引数组的MappedArray
        days = np.array([0, 1, 0, 1, 0, 1])  # 对应日期索引
        ma_with_idx = vbt.MappedArray(wrapper, prices, stocks, idx_arr=days)
        print(f"有索引数组: {ma_with_idx.idx_arr}")  # [0, 1, 0, 1, 0, 1]
        
        # 3. 索引数组的时间含义
        for i, idx in enumerate(ma_with_idx.idx_arr):
            date = wrapper.index[idx]
            price = ma_with_idx.values[i]
            stock = wrapper.columns[ma_with_idx.col_arr[i]]
            print(f"数据点{i}: {stock}在{date}的价格为{price}")
        
        # 4. 转换为pandas DataFrame
        try:
            df_no_idx = ma_no_idx.to_pd()
            print("无索引数组时无法转换为DataFrame")
        except ValueError as e:
            print(f"转换错误: {e}")
        
        df_with_idx = ma_with_idx.to_pd()
        print(f"转换后的DataFrame:\\n{df_with_idx}")
        
        # 5. 索引数组的统计信息
        if ma_with_idx.idx_arr is not None:
            print(f"索引数组长度: {len(ma_with_idx.idx_arr)}")
            print(f"索引范围: {ma_with_idx.idx_arr.min()} 到 {ma_with_idx.idx_arr.max()}")
            print(f"唯一时间点: {np.unique(ma_with_idx.idx_arr)}")
            
            # 每个时间点的数据量
            for time_idx in np.unique(ma_with_idx.idx_arr):
                count = (ma_with_idx.idx_arr == time_idx).sum()
                date = wrapper.index[time_idx]
                print(f"{date}: {count}个数据点")
        ```
        
        转换到pandas的作用：
        - 提供时间轴的准确映射
        - 确保数据在正确的时间位置
        - 支持时间序列的可视化和分析
        
        注意事项：
        - 如果要转换为pandas DataFrame，idx_arr是必需的
        - 索引数组的值必须在有效范围内
        - 同一位置的多个值会导致转换错误（可使用ignore_index=True）
        """
        return self._idx_arr

    @property
    def mapping(self) -> tp.Optional[tp.Mapping]:
        """
        获取值映射 - 数值到标签的转换器
        
        值映射是可选的，它提供了将数值转换为可读标签的机制。这个功能特别适用于
        分类数据或需要友好显示的数值数据，如交易方向、评级等级、状态标识等。
        
        映射类型：
        - 字典映射：{1: '买入', -1: '卖出'}
        - 命名元组映射：Status(BUY=1, SELL=-1)
        - 可调用对象：lambda x: f"Level_{x}"
        - 特殊字符串：'index' 或 'columns'
        
        应用场景：
        - 交易方向的标签显示（1→'买入', -1→'卖出'）
        - 评级等级的转换（1→'A', 2→'B', 3→'C'）
        - 状态标识的可读化（0→'待处理', 1→'已完成'）
        - 数据报告中的友好显示
        
        返回值：
            tp.Optional[tp.Mapping]: 映射对象或None
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 1. 不带映射的MappedArray
        scores = np.array([85, 92, 78, 90, 88, 94])
        students = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['张三', '李四', '王五'], ndim=2)
        
        ma_no_mapping = vbt.MappedArray(wrapper, scores, students)
        print(f"无映射: {ma_no_mapping.mapping}")  # None
        
        # 2. 带等级映射的MappedArray
        grade_mapping = {
            90: 'A', 85: 'B', 80: 'C', 75: 'D', 70: 'E'
        }
        ma_with_mapping = vbt.MappedArray(wrapper, scores, students, mapping=grade_mapping)
        print(f"有映射: {ma_with_mapping.mapping}")
        
        # 3. 应用映射转换
        mapped_ma = ma_with_mapping.apply_mapping()
        print(f"原始分数: {ma_with_mapping.values}")
        print(f"映射后等级: {mapped_ma.values}")
        
        # 4. 交易方向映射示例
        directions = np.array([1, -1, 1, -1, 1, -1])
        trades = np.array([0, 0, 1, 1, 2, 2])
        direction_mapping = {1: '买入', -1: '卖出'}
        
        ma_trades = vbt.MappedArray(wrapper, directions, trades, mapping=direction_mapping)
        print(f"交易方向映射: {ma_trades.mapping}")
        
        # 应用交易方向映射
        trade_labels = ma_trades.apply_mapping()
        print(f"原始方向: {ma_trades.values}")
        print(f"标签方向: {trade_labels.values}")
        
        # 5. 动态映射示例
        def score_to_grade(score):
            if score >= 90:
                return 'A'
            elif score >= 80:
                return 'B'
            elif score >= 70:
                return 'C'
            else:
                return 'D'
        
        ma_dynamic = vbt.MappedArray(wrapper, scores, students, mapping=score_to_grade)
        dynamic_grades = ma_dynamic.apply_mapping()
        print(f"动态映射结果: {dynamic_grades.values}")
        
        # 6. 统计分析中的映射应用
        if ma_with_mapping.mapping:
            # 值频次统计会使用映射
            value_counts = ma_with_mapping.value_counts()
            print(f"值频次统计:\\n{value_counts}")
            
            # 统计报告会显示映射后的标签
            stats = ma_with_mapping.stats(column='张三')
            print(f"统计报告:\\n{stats}")
        ```
        
        特殊映射：
        - 'index': 使用wrapper.index作为映射
        - 'columns': 使用wrapper.columns作为映射
        
        在统计分析中的作用：
        - 值频次统计时显示标签而不是数值
        - 统计报告中提供可读的分类信息
        - 数据可视化时的图例标签
        
        注意事项：
        - 映射是只读的，不能直接修改
        - 映射不会改变原始数据，只影响显示
        - 如果映射中没有某个值，会保持原始数值
        """
        return self._mapping

    @cached_method
    def is_sorted(self, incl_id: bool = False) -> bool:
        """
        检查映射数组是否已排序 - 性能优化的关键判断
        
        该方法检查col_arr是否已按升序排序，这对于ColumnMapper选择最优索引策略
        非常重要。排序的数据可以使用更高效的col_range索引，而未排序的数据
        需要使用col_map索引。
        
        参数说明：
            incl_id (bool, 可选): 是否同时检查ID数组的排序状态
                - False (默认): 只检查col_arr的排序状态
                - True: 同时检查col_arr和id_arr的字典序排序
        
        返回值：
            bool: 如果数据已排序则返回True，否则返回False
        
        排序检查规则：
        - incl_id=False: 检查col_arr是否非递减排序
        - incl_id=True: 检查(col_arr, id_arr)是否按字典序排序
        
        性能影响：
        - 排序数据：ColumnMapper使用col_range，查找速度O(1)
        - 未排序数据：ColumnMapper使用col_map，查找速度O(log n)
        
        """
        # 如果需要检查ID的排序状态
        if incl_id:
            # 使用Numba编译的函数检查(col_arr, id_arr)的字典序排序
            return nb.is_col_idx_sorted_nb(self.col_arr, self.id_arr)
        
        # 只检查col_arr的排序状态
        return nb.is_col_sorted_nb(self.col_arr)

    def sort(self: MappedArrayT,
             incl_id: bool = False,
             idx_arr: tp.Optional[tp.Array1d] = None,
             group_by: tp.GroupByLike = None,
             **kwargs) -> MappedArrayT:
        """
        对映射数组进行排序 - 优化数据访问性能的关键操作
        
        该方法按列数组（主要）和ID数组（次要，可选）对MappedArray进行排序。
        排序后的数据可以使用更高效的列范围索引，显著提高数据访问性能。
        
        参数说明：
            incl_id (bool, 可选): 是否将ID数组作为次要排序键
                - False (默认): 只按col_arr排序
                - True: 按(col_arr, id_arr)进行字典序排序
            idx_arr (array_like, 可选): 替代的索引数组
                - 如果未提供，使用实例的idx_arr
                - 如果提供，会一起进行排序
            group_by (GroupByLike, 可选): 分组方式
                - 排序后应用分组
            **kwargs: 传递给replace方法的其他参数
        
        返回值：
            MappedArrayT: 排序后的新MappedArray实例
        
        排序逻辑：
        1. 检查数据是否已排序，如果已排序则直接返回
        2. 根据incl_id参数选择排序方式
        3. 使用NumPy的排序函数进行排序
        4. 重新排列所有相关数组
        5. 应用分组（如果指定）
        
        性能影响：
        - 排序后可以使用col_range索引，查找速度O(1)
        - 未排序数据需要使用col_map索引，查找速度O(log n)
        - 排序操作本身的时间复杂度为O(n log n)
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建未排序的数据
        prices = np.array([100, 98, 102, 101, 99, 103])
        stocks = np.array([0, 2, 1, 0, 2, 1])  # 未按列排序
        ids = np.array([1, 2, 3, 4, 5, 6])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma_unsorted = vbt.MappedArray(wrapper, prices, stocks, id_arr=ids)
        print(f"排序前是否排序: {ma_unsorted.is_sorted()}")  # False
        print(f"排序前数据: {ma_unsorted.values}")
        print(f"排序前列: {ma_unsorted.col_arr}")
        print(f"排序前ID: {ma_unsorted.id_arr}")
        
        # 1. 按列排序（不包含ID）
        ma_sorted_col = ma_unsorted.sort(incl_id=False)
        print(f"按列排序后是否排序: {ma_sorted_col.is_sorted()}")  # True
        print(f"排序后数据: {ma_sorted_col.values}")
        print(f"排序后列: {ma_sorted_col.col_arr}")
        print(f"排序后ID: {ma_sorted_col.id_arr}")
        
        # 2. 按列和ID排序（字典序）
        ma_sorted_both = ma_unsorted.sort(incl_id=True)
        print(f"按列和ID排序后是否排序: {ma_sorted_both.is_sorted(incl_id=True)}")  # True
        
        # 3. 带时间索引的排序
        dates = np.array([0, 1, 0, 1, 0, 1])
        ma_with_time = vbt.MappedArray(wrapper, prices, stocks, idx_arr=dates)
        ma_time_sorted = ma_with_time.sort()
        print(f"时间索引排序后: {ma_time_sorted.idx_arr}")
        
        # 4. 性能对比
        import time
        
        # 创建大数据集
        large_data = np.random.randn(100000)
        large_cols = np.random.randint(0, 1000, 100000)  # 未排序
        
        ma_large = vbt.MappedArray(
            vbt.ArrayWrapper(columns=list(range(1000)), ndim=2),
            large_data, large_cols
        )
        
        # 排序前的性能
        start = time.time()
        result_unsorted = ma_large.mean()
        time_unsorted = time.time() - start
        
        # 排序后的性能
        ma_large_sorted = ma_large.sort()
        start = time.time()
        result_sorted = ma_large_sorted.mean()
        time_sorted = time.time() - start
        
        print(f"未排序数据计算耗时: {time_unsorted:.4f}秒")
        print(f"排序数据计算耗时: {time_sorted:.4f}秒")
        print(f"性能提升: {time_unsorted/time_sorted:.2f}倍")
        
        # 5. 分组排序
        group_by = ['科技股', '科技股', '其他']
        ma_grouped_sorted = ma_unsorted.sort(group_by=group_by)
        print(f"分组排序后: {ma_grouped_sorted.wrapper.columns}")
        ```
        
        应用场景：
        - 数据预处理和优化
        - 提高后续操作的性能
        - 数据质量改善
        - 算法输入要求的数据格式
        
        注意事项：
        - 如果数据已排序，会直接返回（避免不必要的排序）
        - incl_id=True时排序开销更大，但提供更严格的顺序
        - 排序会改变数据的原始顺序，可能影响某些分析结果
        - 大数据集的排序可能比较耗时，但能显著提高后续操作性能
        """
        # 如果未提供idx_arr，使用实例的idx_arr
        if idx_arr is None:
            idx_arr = self.idx_arr
        
        # 检查数据是否已排序，如果已排序则直接返回
        if self.is_sorted(incl_id=incl_id):
            return self.replace(idx_arr=idx_arr, **kwargs).regroup(group_by)
        
        # 根据是否包含ID选择排序方式
        if incl_id:
            # 使用字典序排序：先按col_arr排序，再按id_arr排序
            # 注意：lexsort的参数顺序是反的，最后一个参数是主要排序键
            ind = np.lexsort((self.id_arr, self.col_arr))  # 这是一个昂贵的操作！
        else:
            # 只按col_arr排序
            ind = np.argsort(self.col_arr)
        
        # 使用排序索引重新排列所有数组，并创建新的MappedArray实例
        return self.replace(
            mapped_arr=self.values[ind],        # 重新排列映射数组
            col_arr=self.col_arr[ind],          # 重新排列列数组
            id_arr=self.id_arr[ind],            # 重新排列ID数组
            idx_arr=idx_arr[ind] if idx_arr is not None else None,  # 重新排列索引数组
            **kwargs                            # 其他参数传递给replace
        ).regroup(group_by)                     # 应用分组

    def apply_mask(self: MappedArrayT,
                   mask: tp.Array1d,
                   idx_arr: tp.Optional[tp.Array1d] = None,
                   group_by: tp.GroupByLike = None,
                   **kwargs) -> MappedArrayT:
        """
        应用掩码过滤数据 - 基于条件的高效数据筛选
        
        该方法根据布尔掩码过滤MappedArray中的数据点，只保留掩码为True的数据。
        这是一个非常高效的数据筛选方法，广泛用于条件过滤、异常值处理等场景。
        
        参数说明：
            mask (array_like): 布尔掩码数组
                - 长度必须与mapped_arr相同
                - True表示保留该数据点，False表示过滤掉
            idx_arr (array_like, 可选): 替代的索引数组
                - 如果未提供，使用实例的idx_arr
                - 会一起进行过滤
            group_by (GroupByLike, 可选): 分组方式
                - 过滤后应用分组
            **kwargs: 传递给replace方法的其他参数
        
        返回值：
            MappedArrayT: 过滤后的新MappedArray实例
        
        过滤逻辑：
        1. 使用np.flatnonzero找到掩码为True的索引
        2. 使用np.take根据索引提取对应的数据
        3. 重新构建所有相关数组
        4. 应用分组（如果指定）

        """
        # 如果未提供idx_arr，使用实例的idx_arr
        if idx_arr is None:
            idx_arr = self.idx_arr
        
        # 使用np.flatnonzero找到掩码为True的所有索引
        # 这比使用布尔索引更高效
        mask_indices = np.flatnonzero(mask)
        
        # 使用索引提取对应的数据并创建新的MappedArray实例
        return self.replace(
            mapped_arr=np.take(self.values, mask_indices),      # 提取对应的数据值
            col_arr=np.take(self.col_arr, mask_indices),        # 提取对应的列数组
            id_arr=np.take(self.id_arr, mask_indices),          # 提取对应的ID数组
            idx_arr=np.take(idx_arr, mask_indices) if idx_arr is not None else None,  # 提取对应的索引数组
            **kwargs                                            # 其他参数传递给replace
        ).regroup(group_by)                                     # 应用分组

    def map_to_mask(self, inout_map_func_nb: tp.MaskInOutMapFunc, *args,
                    group_by: tp.GroupByLike = None) -> tp.Array1d:
        """
        将映射数组转换为掩码 - 基于自定义函数的条件生成器
        
        该方法使用用户定义的Numba函数在每个列/组上生成布尔掩码。这是一个高性能的
        条件生成工具，允许用户实现复杂的逻辑来决定哪些数据点应该被保留或过滤。
        
        参数说明：
            inout_map_func_nb (MaskInOutMapFunc): Numba编译的映射函数
                - 函数签名：func(idxs, col, values, *args) -> mask
                - idxs: 当前列/组中数据点的索引
                - col: 当前列/组的标识符
                - values: 当前列/组的数据值
                - *args: 额外的参数
                - 返回值: 布尔掩码数组
            *args: 传递给映射函数的额外参数
            group_by (GroupByLike, 可选): 分组方式
                - 在每个组内应用映射函数
        
        返回值：
            tp.Array1d: 布尔掩码数组，长度与mapped_arr相同
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建测试数据
        prices = np.array([100, 101, 98, 99, 102, 103, 95, 96])
        stocks = np.array([0, 0, 1, 1, 2, 2, 0, 0])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, prices, stocks)
        
        # 1. 定义一个简单的阈值过滤函数
        @vbt.njit
        def above_threshold_nb(idxs, col, values, threshold):
            \"\"\"保留高于阈值的数据点\"\"\"
            return values > threshold
        
        # 应用阈值过滤
        threshold_mask = ma.map_to_mask(above_threshold_nb, 100)
        print(f"原始数据: {ma.values}")
        print(f"阈值掩码: {threshold_mask}")
        
        # 2. 定义一个相对过滤函数
        @vbt.njit
        def above_mean_nb(idxs, col, values):
            \"\"\"保留高于列内平均值的数据点\"\"\"
            if len(values) == 0:
                return np.empty(0, dtype=np.bool_)
            mean_val = np.mean(values)
            return values > mean_val
        
        # 应用相对过滤
        mean_mask = ma.map_to_mask(above_mean_nb)
        print(f"高于列内均值的掩码: {mean_mask}")
        
        # 3. 定义一个Top-N过滤函数
        @vbt.njit
        def top_n_nb(idxs, col, values, n):
            \"\"\"保留每列前N个最大值\"\"\"
            if len(values) <= n:
                return np.ones(len(values), dtype=np.bool_)
            
            # 找到前N个最大值的阈值
            sorted_values = np.sort(values)
            threshold = sorted_values[-n]
            
            # 处理相等值的情况
            mask = values > threshold
            if np.sum(mask) < n:
                # 如果不够N个，添加等于阈值的值
                equal_mask = values == threshold
                equal_indices = np.where(equal_mask)[0]
                needed = n - np.sum(mask)
                if needed > 0 and len(equal_indices) > 0:
                    mask[equal_indices[:needed]] = True
            
            return mask
        
        # 应用Top-N过滤
        top2_mask = ma.map_to_mask(top_n_nb, 2)
        print(f"每列前2个最大值的掩码: {top2_mask}")
        ```

        """
        # 获取列映射，用于按列/组组织数据
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        
        # 使用Numba编译的函数生成掩码
        return nb.mapped_to_mask_nb(self.values, col_map, inout_map_func_nb, *args)

    @cached_method
    def top_n_mask(self, n: int, **kwargs) -> tp.Array1d:
        """
        生成每列/组前N个最大值的掩码
        
        该方法为每个列或组生成一个布尔掩码，标识前N个最大值的位置。
        这是一个高效的Top-N选择工具，常用于筛选每个资产的最佳表现期。
        
        参数说明：
            n (int): 要选择的前N个元素数量
            **kwargs: 传递给map_to_mask的其他参数
        
        返回值：
            tp.Array1d: 布尔掩码数组，标识前N个最大值的位置
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建测试数据
        returns = np.array([0.05, 0.02, 0.08, 0.03, 0.01, 0.06])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, returns, stocks)
        
        # 获取每只股票的最佳表现
        top1_mask = ma.top_n_mask(1)
        print(f"原始收益率: {ma.values}")
        print(f"前1个最大值掩码: {top1_mask}")
        
        # 应用掩码获取最佳表现
        best_performance = ma.apply_mask(top1_mask)
        print(f"最佳表现收益率: {best_performance.values}")
        print(f"对应股票: {best_performance.col_arr}")
        ```
        """
        return self.map_to_mask(nb.top_n_inout_map_nb, n, **kwargs)

    @cached_method
    def bottom_n_mask(self, n: int, **kwargs) -> tp.Array1d:
        """
        生成每列/组后N个最小值的掩码
        
        该方法为每个列或组生成一个布尔掩码，标识后N个最小值的位置。
        这是一个高效的Bottom-N选择工具，常用于筛选每个资产的最差表现期。
        
        参数说明：
            n (int): 要选择的后N个元素数量
            **kwargs: 传递给map_to_mask的其他参数
        
        返回值：
            tp.Array1d: 布尔掩码数组，标识后N个最小值的位置
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建测试数据
        returns = np.array([0.05, -0.02, 0.08, -0.03, 0.01, -0.06])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, returns, stocks)
        
        # 获取每只股票的最差表现
        bottom1_mask = ma.bottom_n_mask(1)
        print(f"原始收益率: {ma.values}")
        print(f"后1个最小值掩码: {bottom1_mask}")
        
        # 应用掩码获取最差表现
        worst_performance = ma.apply_mask(bottom1_mask)
        print(f"最差表现收益率: {worst_performance.values}")
        print(f"对应股票: {worst_performance.col_arr}")
        ```
        """
        return self.map_to_mask(nb.bottom_n_inout_map_nb, n, **kwargs)

    @cached_method
    def top_n(self: MappedArrayT, n: int, **kwargs) -> MappedArrayT:
        """
        筛选每列/组的前N个最大值
        
        该方法直接返回包含每列/组前N个最大值的新MappedArray实例。
        这是top_n_mask的便捷包装，常用于投资组合中的资产筛选。
        
        参数说明：
            n (int): 要选择的前N个元素数量
            **kwargs: 传递给apply_mask的其他参数
        
        返回值：
            MappedArrayT: 包含前N个最大值的新MappedArray实例
        """
        return self.apply_mask(self.top_n_mask(n), **kwargs)

    @cached_method
    def bottom_n(self: MappedArrayT, n: int, **kwargs) -> MappedArrayT:
        """
        筛选每列/组的后N个最小值
        
        该方法直接返回包含每列/组后N个最小值的新MappedArray实例。
        这是bottom_n_mask的便捷包装，常用于风险分析中的最差情况筛选。
        
        参数说明：
            n (int): 要选择的后N个元素数量
            **kwargs: 传递给apply_mask的其他参数
        
        返回值：
            MappedArrayT: 包含后N个最小值的新MappedArray实例
        """
        return self.apply_mask(self.bottom_n_mask(n), **kwargs)

    @cached_method
    def is_expandable(self, idx_arr: tp.Optional[tp.Array1d] = None, group_by: tp.GroupByLike = None) -> bool:
        """
        检查映射数组是否可以展开为完整矩阵
        
        该方法检查是否存在多个值指向同一个位置的情况。如果存在，则无法直接展开为
        pandas DataFrame，需要使用ignore_index=True或进行数据聚合。
        
        参数说明：
            idx_arr (array_like, 可选): 索引数组
                - 如果未提供，使用实例的idx_arr
                - 如果实例没有idx_arr，则必须提供
            group_by (GroupByLike, 可选): 分组方式
        
        返回值：
            bool: 如果可以展开则返回True，否则返回False
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建可展开的数据
        prices = np.array([100, 101, 98, 99, 102, 103])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        days = np.array([0, 1, 0, 1, 0, 1])
        
        wrapper = vbt.ArrayWrapper(
            index=pd.date_range('2023-01-01', periods=2, freq='D'),
            columns=['AAPL', 'GOOGL', 'MSFT'],
            ndim=2
        )
        
        ma_expandable = vbt.MappedArray(wrapper, prices, stocks, idx_arr=days)
        print(f"可展开: {ma_expandable.is_expandable()}")  # True
        
        # 创建不可展开的数据（同一位置多个值）
        conflict_days = np.array([0, 0, 0, 1, 1, 1])  # 同一天多个值
        ma_conflict = vbt.MappedArray(wrapper, prices, stocks, idx_arr=conflict_days)
        print(f"有冲突的可展开性: {ma_conflict.is_expandable()}")  # False
        ```
        """
        # 如果未提供idx_arr，尝试使用实例的idx_arr
        if idx_arr is None:
            if self.idx_arr is None:
                raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr
        
        # 获取分组后的列数组
        col_arr = self.col_mapper.get_col_arr(group_by=group_by)
        # 获取目标形状
        target_shape = self.wrapper.get_shape_2d(group_by=group_by)
        
        # 使用Numba编译的函数检查是否可展开
        return nb.is_mapped_expandable_nb(col_arr, idx_arr, target_shape)

    def to_pd(self,
              idx_arr: tp.Optional[tp.Array1d] = None,
              ignore_index: bool = False,
              fill_value: float = np.nan,
              group_by: tp.GroupByLike = None,
              wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        将映射数组展开为pandas Series/DataFrame - 数据格式转换的核心方法
        
        该方法将高效的压缩存储格式转换为标准的pandas格式，便于数据分析、可视化
        和与其他pandas工具的集成。转换过程中会根据idx_arr将数据放置到正确的
        时间-资产位置上。
        
        参数说明：
            idx_arr (array_like, 可选): 索引数组
                - 如果未提供，使用实例的idx_arr
                - 如果实例没有idx_arr，则必须提供
                - 定义每个数据点在时间轴上的位置
            ignore_index (bool, 可选): 是否忽略索引
                - False (默认): 使用idx_arr进行精确的时间定位
                - True: 忽略时间位置，将数据堆叠在每列中
            fill_value (float, 可选): 空位置的填充值
                - 默认: np.nan
                - 用于填充没有数据的位置
            group_by (GroupByLike, 可选): 分组方式
                - 在展开前进行分组
            wrap_kwargs (dict, 可选): 传递给wrapper.wrap的额外参数
        
        返回值：
            tp.SeriesFrame: pandas Series (1D) 或 DataFrame (2D)
        
        展开逻辑：
        1. ignore_index=False时：
           - 使用idx_arr将数据映射到正确的时间位置
           - 创建完整的时间×资产矩阵
           - 空位置填充为fill_value
        
        2. ignore_index=True时：
           - 忽略时间位置，将数据按列堆叠
           - 创建行索引为0, 1, 2, ...的矩阵
           - 适用于不关心时间对齐的场景
        
        使用示例：
        ```python
        import numpy as np
        import pandas as pd
        import vectorbt as vbt
        
        # 创建时间序列数据
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        wrapper = vbt.ArrayWrapper(
            index=dates,
            columns=['AAPL', 'GOOGL', 'MSFT'],
            ndim=2
        )
        
        # 创建稀疏数据
        prices = np.array([100, 101, 98, 99, 102])
        stocks = np.array([0, 0, 1, 1, 2])
        days = np.array([0, 1, 0, 2, 1])
        
        ma = vbt.MappedArray(wrapper, prices, stocks, idx_arr=days)
        
        # 1. 标准展开（保持时间对齐）
        df_standard = ma.to_pd()
        print(f"标准展开:\\n{df_standard}")
        #              AAPL  GOOGL  MSFT
        # 2023-01-01  100.0   98.0   NaN
        # 2023-01-02  101.0   99.0  102.0
        # 2023-01-03    NaN    NaN   NaN
        
        # 2. 忽略索引展开（数据堆叠）
        df_stacked = ma.to_pd(ignore_index=True)
        print(f"堆叠展开:\\n{df_stacked}")
        #      AAPL  GOOGL  MSFT
        # 0  100.0   98.0  102.0
        # 1  101.0   99.0    NaN
        
        # 3. 自定义填充值
        df_filled = ma.to_pd(fill_value=0.0)
        print(f"填充0展开:\\n{df_filled}")
        #              AAPL  GOOGL  MSFT
        # 2023-01-01  100.0   98.0    0.0
        # 2023-01-02  101.0   99.0  102.0
        # 2023-01-03    0.0    0.0    0.0
        
        # 4. 分组展开
        group_by = ['科技股', '科技股', '其他']
        df_grouped = ma.to_pd(group_by=group_by)
        print(f"分组展开:\\n{df_grouped}")
        #              科技股   其他
        # 2023-01-01  100.0   NaN
        # 2023-01-02  101.0  102.0
        # 2023-01-03    NaN   NaN
        
        # 5. 处理冲突数据
        # 创建有冲突的数据（同一位置多个值）
        conflict_prices = np.array([100, 101, 102])
        conflict_stocks = np.array([0, 0, 0])
        conflict_days = np.array([0, 0, 0])  # 同一天多个值
        
        ma_conflict = vbt.MappedArray(wrapper, conflict_prices, conflict_stocks, 
                                      idx_arr=conflict_days)
        
        try:
            df_conflict = ma_conflict.to_pd()
            print("不会到达这里")
        except ValueError as e:
            print(f"冲突错误: {e}")
        
        # 使用ignore_index处理冲突
        df_conflict_stacked = ma_conflict.to_pd(ignore_index=True)
        print(f"冲突数据堆叠展开:\\n{df_conflict_stacked}")
        #      AAPL  GOOGL  MSFT
        # 0  100.0    NaN   NaN
        # 1  101.0    NaN   NaN
        # 2  102.0    NaN   NaN
        
        # 6. 性能和内存考虑
        # 创建大规模稀疏数据
        large_data = np.random.randn(10000)
        large_cols = np.random.randint(0, 1000, 10000)
        large_days = np.random.randint(0, 252, 10000)
        
        large_wrapper = vbt.ArrayWrapper(
            index=pd.date_range('2023-01-01', periods=252, freq='D'),
            columns=list(range(1000)),
            ndim=2
        )
        
        ma_large = vbt.MappedArray(large_wrapper, large_data, large_cols, 
                                   idx_arr=large_days)
        
        import sys
        print(f"MappedArray内存占用: {sys.getsizeof(ma_large)} 字节")
        
        # 展开为DataFrame（内存占用会显著增加）
        df_large = ma_large.to_pd()
        print(f"DataFrame内存占用: {sys.getsizeof(df_large)} 字节")
        print(f"内存放大倍数: {sys.getsizeof(df_large) / sys.getsizeof(ma_large):.1f}x")
        
        # 7. 1维数据展开
        ma_1d = vbt.MappedArray(
            vbt.ArrayWrapper(index=dates, ndim=1),
            np.array([1, 2, 3]),
            np.array([0, 0, 0]),  # 所有数据在同一列
            idx_arr=np.array([0, 1, 2])
        )
        
        series_result = ma_1d.to_pd()
        print(f"1D展开结果:\\n{series_result}")
        # 2023-01-01    1
        # 2023-01-02    2
        # 2023-01-03    3
        ```
        
        展开模式对比：
        - 标准模式：保持时间对齐，适用于时间序列分析
        - 堆叠模式：节省空间，适用于统计分析
        - 分组模式：按类别聚合，适用于行业分析
        
        应用场景：
        - 时间序列数据分析和可视化
        - 与pandas生态系统的集成
        - 数据导出和报告生成
        - 传统金融分析工具的输入
        - 机器学习模型的特征工程
        
        性能考虑：
        - 稀疏数据展开会显著增加内存占用
        - 大规模数据建议使用ignore_index=True
        - 展开操作的时间复杂度为O(n)
        - 结果DataFrame的内存占用与矩阵大小成正比
        
        注意事项：
        - 如果有多个值指向同一位置，会抛出ValueError
        - 使用ignore_index=True可以处理位置冲突
        - 展开后的数据可能占用大量内存
        - 缺失位置会填充为fill_value
        
        错误处理：
        - 缺少idx_arr时抛出ValueError
        - 位置冲突时抛出ValueError（除非ignore_index=True）
        - 内存不足时可能抛出MemoryError
        
        更多信息请参考：
        - `vectorbt.records.nb.expand_mapped_nb`
        - `vectorbt.records.nb.stack_expand_mapped_nb`
        """
        if ignore_index:
            if self.wrapper.ndim == 1:
                return self.wrapper.wrap(
                    self.values,
                    index=np.arange(len(self.values)),
                    group_by=group_by,
                    **merge_dicts({}, wrap_kwargs)
                )
            col_map = self.col_mapper.get_col_map(group_by=group_by)
            out = nb.stack_expand_mapped_nb(self.values, col_map, fill_value)
            return self.wrapper.wrap(
                out, index=np.arange(out.shape[0]),
                group_by=group_by, **merge_dicts({}, wrap_kwargs))
        if idx_arr is None:
            if self.idx_arr is None:
                raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr
        if not self.is_expandable(idx_arr=idx_arr, group_by=group_by):
            raise ValueError("Multiple values are pointing to the same position. Use ignore_index.")
        col_arr = self.col_mapper.get_col_arr(group_by=group_by)
        target_shape = self.wrapper.get_shape_2d(group_by=group_by)
        out = nb.expand_mapped_nb(self.values, col_arr, idx_arr, target_shape, fill_value)
        return self.wrapper.wrap(out, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    def apply(self: MappedArrayT,
              apply_func_nb: tp.MappedApplyFunc, *args,
              group_by: tp.GroupByLike = None,
              apply_per_group: bool = False,
              dtype: tp.Optional[tp.DTypeLike] = None,
              **kwargs) -> MappedArrayT:
        """
        在每列/组上应用函数 - 高性能数据变换的核心方法
        
        该方法在映射数组的每个列或组上应用用户定义的Numba函数，返回变换后的新
        MappedArray实例。这是一个非常强大的数据变换工具，可以实现复杂的数学运算、
        统计计算和数据处理逻辑。
        
        参数说明：
            apply_func_nb (MappedApplyFunc): Numba编译的应用函数
                - 函数签名：func(idxs, col, values, *args) -> new_values
                - idxs: 当前列/组中数据点的索引
                - col: 当前列/组的标识符
                - values: 当前列/组的数据值
                - *args: 额外的参数
                - 返回值: 变换后的数据值数组
            *args: 传递给应用函数的额外参数
            group_by (GroupByLike, 可选): 分组方式
                - 在每个组内应用函数
            apply_per_group (bool, 可选): 是否按组应用
                - False (默认): 按列应用，忽略分组
                - True: 按组应用，考虑分组
            dtype (DTypeLike, 可选): 输出数据类型
                - 如果未指定，保持原数据类型
            **kwargs: 传递给replace方法的其他参数
        
        返回值：
            MappedArrayT: 包含变换后数据的新MappedArray实例
        
        工作原理：
        1. 验证函数是Numba编译的
        2. 根据分组设置获取列映射
        3. 使用Numba函数在每列/组上应用变换
        4. 将结果转换为指定的数据类型
        5. 创建新的MappedArray实例
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建测试数据
        prices = np.array([100, 101, 98, 99, 102, 103])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, prices, stocks)
        
        # 1. 计算收益率
        @vbt.njit
        def calculate_returns_nb(idxs, col, values):
            \"\"\"计算每列的收益率\"\"\"
            if len(values) < 2:
                return np.full_like(values, np.nan)
            
            returns = np.empty_like(values)
            returns[0] = np.nan
            for i in range(1, len(values)):
                returns[i] = (values[i] - values[i-1]) / values[i-1]
            return returns
        
        returns_ma = ma.apply(calculate_returns_nb)
        print(f"原始价格: {ma.values}")
        print(f"收益率: {returns_ma.values}")
        
        # 2. 标准化处理
        @vbt.njit
        def standardize_nb(idxs, col, values):
            \"\"\"标准化每列的数据\"\"\"
            if len(values) < 2:
                return values
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                return np.zeros_like(values)
            
            return (values - mean_val) / std_val
        
        standardized_ma = ma.apply(standardize_nb)
        print(f"标准化后: {standardized_ma.values}")
        
        # 3. 移动平均
        @vbt.njit
        def moving_average_nb(idxs, col, values, window):
            \"\"\"计算移动平均\"\"\"
            if len(values) < window:
                return np.full_like(values, np.nan)
            
            ma_values = np.full_like(values, np.nan)
            for i in range(window-1, len(values)):
                ma_values[i] = np.mean(values[i-window+1:i+1])
            
            return ma_values
        
        ma_2period = ma.apply(moving_average_nb, 2)
        print(f"2期移动平均: {ma_2period.values}")
        
        # 4. 百分位排名
        @vbt.njit
        def percentile_rank_nb(idxs, col, values):
            \"\"\"计算每个值在列内的百分位排名\"\"\"
            if len(values) == 0:
                return values
            
            ranks = np.empty_like(values)
            for i in range(len(values)):
                count_lower = 0
                for j in range(len(values)):
                    if values[j] < values[i]:
                        count_lower += 1
                ranks[i] = count_lower / len(values)
            
            return ranks
        
        rank_ma = ma.apply(percentile_rank_nb)
        print(f"百分位排名: {rank_ma.values}")
        
        # 5. 技术指标计算
        @vbt.njit
        def rsi_nb(idxs, col, values, period=14):
            \"\"\"计算RSI指标\"\"\"
            if len(values) < period + 1:
                return np.full_like(values, np.nan)
            
            rsi_values = np.full_like(values, np.nan)
            
            # 计算价格变化
            changes = np.diff(values)
            gains = np.maximum(changes, 0)
            losses = np.maximum(-changes, 0)
            
            # 计算初始平均涨跌幅
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            for i in range(period, len(values)):
                if i == period:
                    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
                else:
                    gain = gains[i-1]
                    loss = losses[i-1]
                    avg_gain = (avg_gain * (period - 1) + gain) / period
                    avg_loss = (avg_loss * (period - 1) + loss) / period
                    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
                
                rsi_values[i] = 100 - (100 / (1 + rs))
            
            return rsi_values
        
        rsi_ma = ma.apply(rsi_nb, 2)  # 简化为2期RSI
        print(f"RSI指标: {rsi_ma.values}")
        
        # 6. 分组应用
        group_by = ['科技股', '科技股', '其他']
        
        @vbt.njit
        def group_normalize_nb(idxs, col, values):
            \"\"\"组内归一化\"\"\"
            if len(values) == 0:
                return values
            
            min_val = np.min(values)
            max_val = np.max(values)
            
            if max_val == min_val:
                return np.ones_like(values) * 0.5
            
            return (values - min_val) / (max_val - min_val)
        
        # 按组应用
        group_norm_ma = ma.apply(group_normalize_nb, group_by=group_by, apply_per_group=True)
        print(f"组内归一化: {group_norm_ma.values}")
        
        # 7. 数据类型转换
        @vbt.njit
        def to_integer_nb(idxs, col, values):
            \"\"\"转换为整数\"\"\"
            return np.round(values).astype(np.int64)
        
        int_ma = ma.apply(to_integer_nb, dtype=np.int64)
        print(f"整数转换: {int_ma.values}")
        print(f"数据类型: {int_ma.values.dtype}")
        
        # 8. 自定义业务逻辑
        @vbt.njit
        def signal_generation_nb(idxs, col, values, threshold=100):
            \"\"\"生成交易信号\"\"\"
            signals = np.zeros_like(values)
            
            for i in range(1, len(values)):
                if values[i] > threshold and values[i-1] <= threshold:
                    signals[i] = 1  # 买入信号
                elif values[i] < threshold and values[i-1] >= threshold:
                    signals[i] = -1  # 卖出信号
            
            return signals
        
        signals_ma = ma.apply(signal_generation_nb, 100)
        print(f"交易信号: {signals_ma.values}")
        ```
        
        应用场景：
        - 技术指标计算（RSI、MACD、布林带等）
        - 数据标准化和归一化
        - 收益率和风险指标计算
        - 信号生成和策略逻辑
        - 数据清洗和预处理
        - 自定义统计变换
        
        性能优势：
        - 使用Numba编译的函数，接近C语言速度
        - 按列并行处理，充分利用多核CPU
        - 避免Python循环，减少解释器开销
        - 支持复杂的数学运算和条件逻辑
        
        注意事项：
        - 应用函数必须是Numba编译的（使用@njit装饰器）
        - 函数必须能处理空数组和单元素数组
        - 返回值的长度必须与输入values长度相同
        - 复杂的函数可能会影响编译和执行性能
        
        更多信息请参考：
        - `vectorbt.records.nb.apply_on_mapped_nb`
        """
        checks.assert_numba_func(apply_func_nb)
        if apply_per_group:
            col_map = self.col_mapper.get_col_map(group_by=group_by)
        else:
            col_map = self.col_mapper.get_col_map(group_by=False)
        mapped_arr = nb.apply_on_mapped_nb(self.values, col_map, apply_func_nb, *args)
        mapped_arr = np.asarray(mapped_arr, dtype=dtype)
        return self.replace(mapped_arr=mapped_arr, **kwargs).regroup(group_by)

    def reduce(self,
               reduce_func_nb: tp.ReduceFunc, *args,
               idx_arr: tp.Optional[tp.Array1d] = None,
               returns_array: bool = False,
               returns_idx: bool = False,
               to_index: bool = True,
               fill_value: tp.Scalar = np.nan,
               group_by: tp.GroupByLike = None,
               wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeriesFrame:
        """
        归约映射数组 - 高性能数据聚合的核心方法
        
        该方法使用用户定义的Numba函数将每个列/组的多个数据点归约为单个值或数组。
        这是实现各种统计指标（如均值、最大值、方差等）的底层方法，支持灵活的
        返回格式和索引处理。
        
        参数说明：
            reduce_func_nb (ReduceFunc): Numba编译的归约函数
                - 函数签名：func(idxs, col, values, *args) -> result
                - idxs: 当前列/组中数据点的索引
                - col: 当前列/组的标识符
                - values: 当前列/组的数据值
                - *args: 额外的参数
                - 返回值: 归约后的结果（标量或数组）
            *args: 传递给归约函数的额外参数
            idx_arr (array_like, 可选): 索引数组
                - 如果returns_idx=True，必须提供
                - 用于返回索引位置而非值
            returns_array (bool, 可选): 是否返回数组结果
                - False (默认): 返回标量结果
                - True: 返回数组结果
            returns_idx (bool, 可选): 是否返回索引
                - False (默认): 返回数据值
                - True: 返回索引位置
            to_index (bool, 可选): 是否转换为索引标签
                - True (默认): 返回索引标签
                - False: 返回原始位置
            fill_value (Scalar, 可选): 空值填充
                - 默认: np.nan
                - 用于填充空列/组的结果
            group_by (GroupByLike, 可选): 分组方式
                - 在每个组内进行归约
            wrap_kwargs (dict, 可选): 包装参数
                - 传递给wrapper.wrap的额外参数
        
        返回值：
            tp.MaybeSeriesFrame: pandas Series或DataFrame，取决于返回类型
        
        归约模式：
        1. 标量值 (returns_array=False, returns_idx=False)：
           - 每列/组返回一个数值
           - 适用于均值、总和、计数等统计量
        
        2. 索引位置 (returns_array=False, returns_idx=True)：
           - 每列/组返回一个索引位置
           - 适用于最大值位置、最小值位置等
        
        3. 数组结果 (returns_array=True, returns_idx=False)：
           - 每列/组返回一个数值数组
           - 适用于分位数、统计描述等
        
        4. 索引数组 (returns_array=True, returns_idx=True)：
           - 每列/组返回一个索引位置数组
           - 适用于极值位置数组等
        
        """
        # Perform checks
        checks.assert_numba_func(reduce_func_nb)
        if idx_arr is None:
            if self.idx_arr is None:
                if returns_idx:
                    raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr

        # Perform main computation
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        if not returns_array:
            if not returns_idx:
                out = nb.reduce_mapped_nb(
                    self.values,
                    col_map,
                    fill_value,
                    reduce_func_nb,
                    *args
                )
            else:
                out = nb.reduce_mapped_to_idx_nb(
                    self.values,
                    col_map,
                    idx_arr,
                    fill_value,
                    reduce_func_nb,
                    *args
                )
        else:
            if not returns_idx:
                out = nb.reduce_mapped_to_array_nb(
                    self.values,
                    col_map,
                    fill_value,
                    reduce_func_nb,
                    *args
                )
            else:
                out = nb.reduce_mapped_to_idx_array_nb(
                    self.values,
                    col_map,
                    idx_arr,
                    fill_value,
                    reduce_func_nb,
                    *args
                )

        # Perform post-processing
        wrap_kwargs = merge_dicts(dict(
            name_or_index='reduce' if not returns_array else None,
            to_index=returns_idx and to_index,
            fillna=-1 if returns_idx else None,
            dtype=np.int64 if returns_idx else None
        ), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=group_by, **wrap_kwargs)

    @cached_method
    def nth(self, n: int, group_by: tp.GroupByLike = None,
            wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """Return n-th element of each column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index='nth'), wrap_kwargs)
        return self.reduce(
            generic_nb.nth_reduce_nb, n,
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
            **kwargs
        )

    @cached_method
    def nth_index(self, n: int, group_by: tp.GroupByLike = None,
                  wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """Return index of n-th element of each column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index='nth_index'), wrap_kwargs)
        return self.reduce(
            generic_nb.nth_index_reduce_nb, n,
            returns_array=False,
            returns_idx=True,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
            **kwargs
        )

    @cached_method
    def min(self, group_by: tp.GroupByLike = None,
            wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        计算每列/组的最小值 - 寻找最低点的统计方法
        
        该方法计算映射数组中每个列或组的最小值，是风险分析和异常检测中的重要指标。
        在量化交易中，最小值常用于计算最大回撤、支撑位分析等。
        
        参数说明：
            group_by (GroupByLike, 可选): 分组方式
                - 在每个组内计算最小值
            wrap_kwargs (dict, 可选): 包装参数
                - 传递给结果包装的额外参数
            **kwargs: 传递给reduce方法的其他参数
        
        返回值：
            tp.MaybeSeries: 包含每列/组最小值的pandas Series
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建价格数据
        prices = np.array([100, 95, 105, 80, 90, 110])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, prices, stocks)
        
        # 计算最小值
        min_prices = ma.min()
        print(f"最小价格: {min_prices}")
        # AAPL     95
        # GOOGL    80
        # MSFT     90
        
        # 分组计算最小值
        group_by = ['科技股', '科技股', '其他']
        group_min = ma.min(group_by=group_by)
        print(f"分组最小值: {group_min}")
        ```
        """
        wrap_kwargs = merge_dicts(dict(name_or_index='min'), wrap_kwargs)
        return self.reduce(
            generic_nb.min_reduce_nb,
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
            **kwargs
        )

    @cached_method
    def max(self, group_by: tp.GroupByLike = None,
            wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        计算每列/组的最大值 - 寻找最高点的统计方法
        
        该方法计算映射数组中每个列或组的最大值，是性能分析和潜力评估中的重要指标。
        在量化交易中，最大值常用于计算最高收益、阻力位分析等。
        
        参数说明：
            group_by (GroupByLike, 可选): 分组方式
                - 在每个组内计算最大值
            wrap_kwargs (dict, 可选): 包装参数
                - 传递给结果包装的额外参数
            **kwargs: 传递给reduce方法的其他参数
        
        返回值：
            tp.MaybeSeries: 包含每列/组最大值的pandas Series
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建收益率数据
        returns = np.array([0.05, 0.12, 0.03, 0.08, 0.02, 0.15])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, returns, stocks)
        
        # 计算最大收益率
        max_returns = ma.max()
        print(f"最大收益率: {max_returns}")
        # AAPL     0.12
        # GOOGL    0.08
        # MSFT     0.15
        
        # 用于计算最大回撤的分子
        # 通常与cumulative maximum结合使用
        ```
        """
        wrap_kwargs = merge_dicts(dict(name_or_index='max'), wrap_kwargs)
        return self.reduce(
            generic_nb.max_reduce_nb,
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
            **kwargs
        )

    @cached_method
    def mean(self, group_by: tp.GroupByLike = None,
             wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        计算每列/组的平均值 - 中心趋势的经典指标
        
        该方法计算映射数组中每个列或组的算术平均值，是描述性统计和基准比较中
        最常用的指标之一。在量化交易中，平均值用于计算期望收益、策略基准等。
        
        参数说明：
            group_by (GroupByLike, 可选): 分组方式
                - 在每个组内计算平均值
            wrap_kwargs (dict, 可选): 包装参数
                - 传递给结果包装的额外参数
            **kwargs: 传递给reduce方法的其他参数
        
        返回值：
            tp.MaybeSeries: 包含每列/组平均值的pandas Series
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建日收益率数据
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.00, 0.02])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, returns, stocks)
        
        # 计算平均日收益率
        mean_returns = ma.mean()
        print(f"平均日收益率: {mean_returns}")
        # AAPL     0.015
        # GOOGL    0.010
        # MSFT     0.010
        
        # 年化收益率（假设252个交易日）
        annual_returns = mean_returns * 252
        print(f"年化收益率: {annual_returns}")
        
        # 按行业分组计算平均收益率
        group_by = ['科技股', '科技股', '其他']
        sector_returns = ma.mean(group_by=group_by)
        print(f"行业平均收益率: {sector_returns}")
        ```
        """
        wrap_kwargs = merge_dicts(dict(name_or_index='mean'), wrap_kwargs)
        return self.reduce(
            generic_nb.mean_reduce_nb,
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
            **kwargs
        )

    @cached_method
    def median(self, group_by: tp.GroupByLike = None,
               wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        计算每列/组的中位数 - 抗异常值的中心趋势指标
        
        该方法计算映射数组中每个列或组的中位数，相比均值更能抵抗异常值的影响。
        在量化交易中，中位数用于评估典型表现、异常值检测等。
        
        参数说明：
            group_by (GroupByLike, 可选): 分组方式
                - 在每个组内计算中位数
            wrap_kwargs (dict, 可选): 包装参数
                - 传递给结果包装的额外参数
            **kwargs: 传递给reduce方法的其他参数
        
        返回值：
            tp.MaybeSeries: 包含每列/组中位数的pandas Series
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建带异常值的收益率数据
        returns = np.array([0.02, 0.50, 0.01, 0.02, 0.01, 0.01])  # 第2个值是异常值
        stocks = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, returns, stocks)
        
        # 比较均值和中位数
        mean_returns = ma.mean()
        median_returns = ma.median()
        
        print(f"平均收益率: {mean_returns}")
        print(f"中位数收益率: {median_returns}")
        # AAPL的中位数不会受到异常值0.50的严重影响
        
        # 中位数在风险分析中的应用
        # 评估典型的交易表现
        ```
        """
        wrap_kwargs = merge_dicts(dict(name_or_index='median'), wrap_kwargs)
        return self.reduce(
            generic_nb.median_reduce_nb,
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
            **kwargs
        )

    @cached_method
    def std(self, ddof: int = 1, group_by: tp.GroupByLike = None,
            wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        计算每列/组的标准差 - 衡量数据离散程度的重要指标
        
        该方法计算映射数组中每个列或组的标准差，是风险分析中最重要的指标之一。
        在量化交易中，标准差常用于衡量波动率、计算夏普比率等。
        
        参数说明：
            ddof (int, 可选): 自由度修正系数
                - 1 (默认): 样本标准差，分母为n-1
                - 0: 总体标准差，分母为n
            group_by (GroupByLike, 可选): 分组方式
                - 在每个组内计算标准差
            wrap_kwargs (dict, 可选): 包装参数
                - 传递给结果包装的额外参数
            **kwargs: 传递给reduce方法的其他参数
        
        返回值：
            tp.MaybeSeries: 包含每列/组标准差的pandas Series
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建日收益率数据
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.04])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, returns, stocks)
        
        # 计算日收益率标准差（样本标准差）
        daily_std = ma.std()
        print(f"日收益率标准差: {daily_std}")
        
        # 年化波动率（假设252个交易日）
        annual_volatility = daily_std * np.sqrt(252)
        print(f"年化波动率: {annual_volatility}")
        
        # 计算夏普比率
        mean_returns = ma.mean()
        risk_free_rate = 0.02  # 2%的无风险利率
        sharpe_ratio = (mean_returns * 252 - risk_free_rate) / annual_volatility
        print(f"夏普比率: {sharpe_ratio}")
        
        # 使用总体标准差
        population_std = ma.std(ddof=0)
        print(f"总体标准差: {population_std}")
        ```
        """
        wrap_kwargs = merge_dicts(dict(name_or_index='std'), wrap_kwargs)
        return self.reduce(
            generic_nb.std_reduce_nb, ddof,
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
            **kwargs
        )

    @cached_method
    def sum(self, fill_value: tp.Scalar = 0., group_by: tp.GroupByLike = None,
            wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        计算每列/组的总和 - 累积效应的统计指标
        
        该方法计算映射数组中每个列或组的数值总和，常用于计算总收益、累积交易量等。
        在量化交易中，总和用于评估累积效应和总体表现。
        
        参数说明：
            fill_value (Scalar, 可选): 空值填充
                - 默认: 0.0
                - 用于填充空列/组的结果
            group_by (GroupByLike, 可选): 分组方式
                - 在每个组内计算总和
            wrap_kwargs (dict, 可选): 包装参数
                - 传递给结果包装的额外参数
            **kwargs: 传递给reduce方法的其他参数
        
        返回值：
            tp.MaybeSeries: 包含每列/组总和的pandas Series
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建交易量数据
        volumes = np.array([1000, 1500, 2000, 2500, 800, 1200])
        stocks = np.array([0, 0, 1, 1, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, volumes, stocks)
        
        # 计算总交易量
        total_volume = ma.sum()
        print(f"总交易量: {total_volume}")
        # AAPL     2500
        # GOOGL    4500
        # MSFT     2000
        
        # 计算总收益
        pnl = np.array([100, -50, 200, 150, -30, 80])
        pnl_ma = vbt.MappedArray(wrapper, pnl, stocks)
        
        total_pnl = pnl_ma.sum()
        print(f"总盈亏: {total_pnl}")
        # AAPL     50
        # GOOGL    350
        # MSFT     50
        
        # 按行业分组计算总交易量
        group_by = ['科技股', '科技股', '其他']
        sector_volume = ma.sum(group_by=group_by)
        print(f"行业总交易量: {sector_volume}")
        
        # 处理空值的情况
        sparse_data = np.array([100, 200])
        sparse_cols = np.array([0, 2])  # 只有第0列和第2列有数据
        sparse_ma = vbt.MappedArray(wrapper, sparse_data, sparse_cols)
        
        sparse_sum = sparse_ma.sum()
        print(f"稀疏数据总和: {sparse_sum}")
        # AAPL     100
        # GOOGL    0    (使用fill_value=0)
        # MSFT     200
        ```
        """
        wrap_kwargs = merge_dicts(dict(name_or_index='sum'), wrap_kwargs)
        return self.reduce(
            generic_nb.sum_reduce_nb,
            fill_value=fill_value,
            returns_array=False,
            returns_idx=False,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
            **kwargs
        )

    @cached_method
    def idxmin(self, group_by: tp.GroupByLike = None,
               wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """Return index of min by column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index='idxmin'), wrap_kwargs)
        return self.reduce(
            generic_nb.argmin_reduce_nb,
            returns_array=False,
            returns_idx=True,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
            **kwargs
        )

    @cached_method
    def idxmax(self, group_by: tp.GroupByLike = None,
               wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """Return index of max by column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index='idxmax'), wrap_kwargs)
        return self.reduce(
            generic_nb.argmax_reduce_nb,
            returns_array=False,
            returns_idx=True,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
            **kwargs
        )

    @cached_method
    def describe(self,
                 percentiles: tp.Optional[tp.ArrayLike] = None,
                 ddof: int = 1,
                 group_by: tp.GroupByLike = None,
                 wrap_kwargs: tp.KwargsLike = None,
                 **kwargs) -> tp.SeriesFrame:
        """
        计算描述性统计汇总 - 数据分析的核心工具
        
        该方法计算映射数组的完整描述性统计信息，包括计数、均值、标准差、最小值、
        分位数、最大值等。这是数据分析中最常用的汇总统计方法，为投资决策提供
        全面的数据洞察。
        
        参数说明：
            percentiles (ArrayLike, 可选): 要计算的分位数数组
                - 默认: [0.25, 0.5, 0.75] (25%, 50%, 75%分位数)
                - 值应在0.0到1.0之间
                - 会自动包含0.5(中位数)，即使未指定
            ddof (int, 可选): 计算标准差时的自由度修正
                - 默认: 1 (样本标准差)
                - 0: 总体标准差
            group_by (GroupByLike, 可选): 分组方式
                - 按组计算描述性统计
            wrap_kwargs (dict, 可选): 包装参数
                - 传递给结果包装的额外参数
            **kwargs: 传递给reduce方法的其他参数
        
        返回值：
            tp.SeriesFrame: 包含描述性统计的pandas Series/DataFrame
                - 行索引: ['count', 'mean', 'std', 'min', 分位数, 'max']
                - 列索引: 原始列名或分组名
        
        统计指标说明：
            - count: 非空数据点数量
            - mean: 算术平均值
            - std: 标准差
            - min: 最小值
            - 分位数: 指定百分位数的值
            - max: 最大值
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建股票收益率数据
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)  # 正态分布收益率
        stocks = np.random.choice([0, 1, 2], 1000)
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, returns, stocks)
        
        # 1. 基本描述性统计
        desc_stats = ma.describe()
        print(f"基本统计信息:\\n{desc_stats}")
        #              AAPL     GOOGL      MSFT
        # count   334.000000  333.000000  333.000000
        # mean      0.000891    0.001072    0.000964
        # std       0.020156    0.019843    0.020287
        # min      -0.066023   -0.064329   -0.058394
        # 25%      -0.012679   -0.012589   -0.012756
        # 50%       0.000734    0.001089    0.000891
        # 75%       0.013598    0.013789    0.013945
        # max       0.063742    0.059932    0.064123
        
        # 2. 自定义分位数
        custom_percentiles = [0.01, 0.05, 0.1, 0.9, 0.95, 0.99]
        desc_custom = ma.describe(percentiles=custom_percentiles)
        print(f"自定义分位数统计:\\n{desc_custom}") 
        #              AAPL     GOOGL      MSFT
        # count   334.000000  333.000000  333.000000
        # mean      0.000891    0.001072    0.000964
        # std       0.020156    0.019843    0.020287
        # min      -0.066023   -0.064329   -0.058394
        # 1%       -0.012679   -0.012589   -0.012756
        # 5%       -0.012679   -0.012589   -0.012756
        
        # 3. 按行业分组统计
        group_by = ['科技股', '科技股', '其他']
        desc_grouped = ma.describe(group_by=group_by)
        print(f"行业统计:\\n{desc_grouped}")
        
        # 4. 风险分析应用
        # 计算VaR (Value at Risk)
        var_95 = desc_stats.loc['5%']  # 95% VaR
        print(f"95% VaR: {var_95}")
        #              AAPL     GOOGL      MSFT
        # 5%       -0.012679   -0.012589   -0.012756
        
        """
        # 处理分位数参数：如果未提供，使用默认分位数
        if percentiles is not None:
            percentiles = to_1d_array(percentiles)  # 转换为一维数组
        else:
            percentiles = np.array([0.25, 0.5, 0.75])  # 默认四分位数
        
        # 转换为列表便于操作
        percentiles = percentiles.tolist()
        
        # 确保包含中位数(50%)
        if 0.5 not in percentiles:
            percentiles.append(0.5)
        
        # 排序和去重分位数
        percentiles = np.unique(percentiles)
        
        # 格式化分位数标签（如25%、50%、75%）
        perc_formatted = pd.io.formats.format.format_percentiles(percentiles)
        
        # 创建结果的行索引：统计指标名称
        index = pd.Index(['count', 'mean', 'std', 'min', *perc_formatted, 'max'])
        
        # 合并包装参数，设置行索引
        wrap_kwargs = merge_dicts(dict(name_or_index=index), wrap_kwargs)
        
        # 调用reduce方法执行实际的统计计算
        out = self.reduce(
            generic_nb.describe_reduce_nb,  # 使用通用的describe归约函数
            percentiles,                    # 传递分位数参数
            ddof,                          # 传递自由度修正参数
            returns_array=True,            # 返回数组（多个统计指标）
            returns_idx=False,             # 不返回索引
            group_by=group_by,             # 分组参数
            wrap_kwargs=wrap_kwargs,       # 包装参数
            **kwargs                       # 其他参数
        )
        
        # 后处理：处理计数统计的NaN值
        if isinstance(out, pd.DataFrame):
            # 对于DataFrame，将count行的NaN值填充为0
            out.loc['count'].fillna(0., inplace=True)
        else:
            # 对于Series，将count的NaN值填充为0
            if np.isnan(out.loc['count']):
                out.loc['count'] = 0.
        
        return out

    @cached_method
    def count(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算每列/组的数据点数量 - 数据完整性的基础指标
        
        该方法计算映射数组中每个列或组的非空数据点数量，是数据质量评估和
        统计分析中的基础指标。在量化交易中，计数用于评估数据覆盖率、
        样本大小等。
        
        参数说明：
            group_by (GroupByLike, 可选): 分组方式
                - 在每个组内计算数据点数量
            wrap_kwargs (dict, 可选): 包装参数
                - 传递给结果包装的额外参数
        
        返回值：
            tp.MaybeSeries: 包含每列/组数据点数量的pandas Series
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建不同密度的数据
        prices = np.array([100, 101, 102, 98, 99])
        stocks = np.array([0, 0, 0, 1, 2])  # 股票0有3个数据点，股票1和2各有1个
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, prices, stocks)
        
        # 计算每只股票的数据点数量
        data_counts = ma.count()
        print(f"数据点数量: {data_counts}")
        # AAPL     3
        # GOOGL    1
        # MSFT     1
        
        # 数据覆盖率分析
        total_possible = len(wrapper.index)  # 假设时间序列长度
        coverage = data_counts / total_possible
        print(f"数据覆盖率: {coverage}")
        
        # 按行业分组计算数据点数量
        group_by = ['科技股', '科技股', '其他']
        sector_counts = ma.count(group_by=group_by)
        print(f"行业数据点数量: {sector_counts}")
        
        # 用于计算统计指标的置信度
        # 通常需要足够的样本量才能计算可靠的统计指标
        min_samples = 30
        reliable_stocks = data_counts[data_counts >= min_samples]
        print(f"可靠的股票数据: {reliable_stocks}")
        
        # 与其他统计指标结合使用
        mean_returns = ma.mean()
        std_returns = ma.std()
        
        # 只对有足够数据的股票计算夏普比率
        sufficient_data = data_counts >= 10
        reliable_sharpe = (mean_returns[sufficient_data] / 
                          std_returns[sufficient_data])
        print(f"可靠的夏普比率: {reliable_sharpe}")
        ```
        """
        # 合并包装参数，设置结果名称
        wrap_kwargs = merge_dicts(dict(name_or_index='count'), wrap_kwargs)
        
        # 直接从列映射器获取每列的数据点数量，这是最高效的方法
        # col_mapper.get_col_map()返回(col_start_indices, col_counts)
        # 我们只需要col_counts，即每列的数据点数量
        return self.wrapper.wrap_reduced(
            self.col_mapper.get_col_map(group_by=group_by)[1],  # 获取每列的计数
            group_by=group_by,    # 分组参数
            **wrap_kwargs         # 包装参数
        )

    @cached_method
    def value_counts(self,
                     normalize: bool = False,
                     sort_uniques: bool = True,
                     sort: bool = False,
                     ascending: bool = False,
                     dropna: bool = False,
                     group_by: tp.GroupByLike = None,
                     mapping: tp.Optional[tp.MappingLike] = None,
                     incl_all_keys: bool = False,
                     wrap_kwargs: tp.KwargsLike = None,
                     **kwargs) -> tp.SeriesFrame:
        """
        计算值频次分布 - 离散数据分析的核心工具
        
        该方法计算映射数组中每个唯一值的出现频次，支持标准化、排序、映射等高级功能。
        在量化交易中，value_counts用于分析交易信号分布、评级分布、状态转换等。
        
        参数说明：
            normalize (bool, 可选): 是否标准化为概率
                - False (默认): 返回绝对频次
                - True: 返回相对频次(概率)
            sort_uniques (bool, 可选): 是否对唯一值排序
                - True (默认): 对唯一值进行排序
                - False: 保持原始顺序
            sort (bool, 可选): 是否按频次排序
                - False (默认): 不按频次排序
                - True: 按频次排序
            ascending (bool, 可选): 排序方向
                - False (默认): 降序排列
                - True: 升序排列
            dropna (bool, 可选): 是否排除NaN值
                - False (默认): 包含NaN值
                - True: 排除NaN值
            group_by (GroupByLike, 可选): 分组方式
                - 在每个组内计算频次
            mapping (MappingLike, 可选): 值映射
                - 将原始值映射为可读标签
                - 如果为None，使用实例的mapping
            incl_all_keys (bool, 可选): 是否包含所有映射键
                - False (默认): 只包含实际出现的值
                - True: 包含映射中的所有键，未出现的设为0
            wrap_kwargs (dict, 可选): 包装参数
            **kwargs: 传递给映射函数的其他参数
        
        返回值：
            tp.SeriesFrame: 值频次分布表
                - 行索引: 唯一值(或映射后的标签)
                - 列索引: 原始列名或分组名
                - 值: 频次(或概率)
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建交易信号数据
        signals = np.array([1, -1, 0, 1, -1, 1, 0, -1, 1, 0])
        stocks = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        # 创建映射数组
        signal_mapping = {1: '买入', -1: '卖出', 0: '持有'}
        ma = vbt.MappedArray(wrapper, signals, stocks, mapping=signal_mapping)
        
        # 1. 基本频次统计
        counts = ma.value_counts()
        print(f"信号频次分布:\\n{counts}")
        #        AAPL  GOOGL  MSFT
        # 买入     1      2     1
        # 卖出     1      1     1
        # 持有     1      0     2
        
        # 2. 标准化为概率
        probs = ma.value_counts(normalize=True)
        print(f"信号概率分布:\\n{probs}")
        #        AAPL  GOOGL  MSFT
        # 买入   0.33   0.67   0.25
        # 卖出   0.33   0.33   0.25
        # 持有   0.33   0.00   0.50
        
        # 3. 按频次排序
        sorted_counts = ma.value_counts(sort=True, ascending=False)
        print(f"按频次排序:\\n{sorted_counts}")
        
        # 4. 包含所有映射键
        complete_counts = ma.value_counts(incl_all_keys=True)
        print(f"完整映射键:\\n{complete_counts}")
        
        # 5. 分组统计
        group_by = ['科技股', '科技股', '其他']
        group_counts = ma.value_counts(group_by=group_by)
        print(f"行业信号分布:\\n{group_counts}")
        
        # 6. 评级分布分析
        ratings = np.array([1, 2, 3, 2, 1, 3, 2, 1, 3, 2])
        rating_stocks = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        rating_mapping = {1: 'AAA', 2: 'AA', 3: 'A'}
        
        rating_ma = vbt.MappedArray(wrapper, ratings, rating_stocks, 
                                   mapping=rating_mapping)
        
        rating_dist = rating_ma.value_counts(normalize=True)
        print(f"评级分布:\\n{rating_dist}")
        
        # 7. 风险分析应用
        # 计算信号多样性（熵的简化版本）
        entropy_proxy = -(probs * np.log2(probs.replace(0, np.finfo(float).eps))).sum()
        print(f"信号多样性: {entropy_proxy}")
        
        # 8. 异常值检测
        # 识别出现频次极低的信号
        rare_signals = counts[counts == 1]
        print(f"罕见信号: {rare_signals}")
        
        # 9. 时间序列状态分析
        # 用于分析市场状态转换
        market_states = np.array([0, 0, 1, 1, 2, 2, 1, 0, 2, 1])
        state_mapping = {0: '牛市', 1: '震荡', 2: '熊市'}
        
        state_ma = vbt.MappedArray(wrapper, market_states, 
                                  np.zeros(len(market_states)), 
                                  mapping=state_mapping)
        
        state_dist = state_ma.value_counts(normalize=True)
        print(f"市场状态分布: {state_dist}")
        ```
        
        应用场景：
        - 交易信号分析
        - 市场状态分布
        - 评级分布分析
        - 异常值检测
        - 数据质量评估
        - 概率分布建模
        
        注意事项：
        - 不考虑缺失值（与pandas不同）
        - 映射会改变索引标签但不改变统计逻辑
        - 标准化是对所有值的总和进行的
        - 排序优先级：sort_uniques > sort
        """
        # 导入版本解析工具，用于pandas版本兼容性检查
        from pkg_resources import parse_version

        # 处理映射参数：如果未提供，使用实例的mapping
        if mapping is None:
            mapping = self.mapping
        
        # 处理特殊字符串映射
        if isinstance(mapping, str):
            if mapping.lower() == 'index':
                mapping = self.wrapper.index    # 使用索引作为映射
            elif mapping.lower() == 'columns':
                mapping = self.wrapper.columns  # 使用列作为映射
            mapping = to_mapping(mapping)       # 转换为标准映射格式
        
        # 使用pandas的factorize函数进行因子化
        # 这会将原始值转换为整数代码，并返回唯一值
        if parse_version(pd.__version__) < parse_version("1.5.0"):
            # 旧版本pandas的factorize语法
            mapped_codes, mapped_uniques = pd.factorize(self.values, sort=False, na_sentinel=None)
        else:
            # 新版本pandas的factorize语法
            mapped_codes, mapped_uniques = pd.factorize(self.values, sort=False, use_na_sentinel=False)
        
        # 获取列映射，用于按列/组计算频次
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        
        # 使用Numba编译的函数计算值频次
        # 返回一个二维数组：行为唯一值，列为列/组
        value_counts = nb.mapped_value_counts_nb(mapped_codes, len(mapped_uniques), col_map)
        
        # 处理包含所有映射键的情况
        if incl_all_keys and mapping is not None:
            missing_keys = []
            # 查找映射中存在但数据中不存在的键
            for x in mapping:
                # 跳过NaN值的处理
                if pd.isnull(x) and pd.isnull(mapped_uniques).any():
                    continue
                # 如果映射键不在唯一值中，添加为缺失键
                if x not in mapped_uniques:
                    missing_keys.append(x)
            
            # 为缺失键添加零频次行
            if missing_keys:
                value_counts = np.vstack((value_counts, np.full((len(missing_keys), value_counts.shape[1]), 0)))
                mapped_uniques = np.concatenate((mapped_uniques, np.array(missing_keys)))
        
        # 创建NaN值的掩码
        nan_mask = np.isnan(mapped_uniques)
        
        # 如果需要排除NaN值
        if dropna:
            value_counts = value_counts[~nan_mask]      # 排除NaN值对应的行
            mapped_uniques = mapped_uniques[~nan_mask]  # 排除NaN值
        
        # 如果需要对唯一值进行排序
        if sort_uniques:
            new_indices = mapped_uniques.argsort()     # 获取排序索引
            value_counts = value_counts[new_indices]    # 重新排列频次数组
            mapped_uniques = mapped_uniques[new_indices] # 重新排列唯一值
        
        # 计算每个唯一值的总频次（跨列求和）
        value_counts_sum = value_counts.sum(axis=1)
        
        # 如果需要标准化为概率
        if normalize:
            value_counts = value_counts / value_counts_sum.sum()
        
        # 如果需要按频次排序
        if sort:
            if ascending:
                # 升序排列
                new_indices = value_counts_sum.argsort()
            else:
                # 降序排列（默认）
                new_indices = (-value_counts_sum).argsort()
            value_counts = value_counts[new_indices]      # 重新排列频次数组
            mapped_uniques = mapped_uniques[new_indices]  # 重新排列唯一值
        
        # 将NumPy数组包装为pandas对象
        value_counts_pd = self.wrapper.wrap(
            value_counts,
            index=mapped_uniques,     # 使用唯一值作为行索引
            group_by=group_by,        # 分组参数
            **merge_dicts({}, wrap_kwargs)  # 包装参数
        )
        
        # 如果提供了映射，应用映射到索引
        if mapping is not None:
            value_counts_pd.index = apply_mapping(value_counts_pd.index, mapping, **kwargs)
        
        return value_counts_pd

    @cached_method
    def apply_mapping(self: MappedArrayT, mapping: tp.Optional[tp.MappingLike] = None, **kwargs) -> MappedArrayT:
        """
        应用映射变换 - 数据标签化的核心工具
        
        该方法将映射函数应用到映射数组的每个元素上，实现数值到标签的转换。
        在量化交易中，常用于将数字信号转换为可读标签，如将1/-1转换为"买入"/"卖出"。
        
        参数说明：
            mapping (MappingLike, 可选): 映射规则
                - 如果为None，使用实例的mapping属性
                - 字典映射：{1: '买入', -1: '卖出'}
                - 可调用对象：lambda x: f"Level_{x}"
                - 特殊字符串：'index'或'columns'
            **kwargs: 传递给映射函数的其他参数
        
        返回值：
            MappedArrayT: 应用映射后的新MappedArray实例
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建数值信号数据
        signals = np.array([1, -1, 0, 1, -1, 1, 0, -1])
        stocks = np.array([0, 0, 1, 1, 2, 2, 0, 1])
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, signals, stocks)
        
        # 1. 基本映射变换
        signal_mapping = {1: '买入', -1: '卖出', 0: '持有'}
        mapped_ma = ma.apply_mapping(signal_mapping)
        print(f"原始信号: {ma.values}")
        print(f"映射后信号: {mapped_ma.values}")
        # 原始信号: [ 1 -1  0  1 -1  1  0 -1]
        # 映射后信号: ['买入' '卖出' '持有' '买入' '卖出' '买入' '持有' '卖出']
        
        # 2. 使用实例的mapping属性
        ma_with_mapping = vbt.MappedArray(wrapper, signals, stocks, mapping=signal_mapping)
        auto_mapped = ma_with_mapping.apply_mapping()  # 自动使用实例的mapping
        print(f"自动映射: {auto_mapped.values}")
        
        # 3. 函数映射
        def grade_mapper(score):
            if score >= 90:
                return 'A'
            elif score >= 80:
                return 'B'
            elif score >= 70:
                return 'C'
            else:
                return 'D'
        
        scores = np.array([95, 87, 92, 78, 88, 65, 90, 85])
        score_ma = vbt.MappedArray(wrapper, scores, stocks)
        grade_ma = score_ma.apply_mapping(grade_mapper)
        print(f"分数: {score_ma.values}")
        print(f"等级: {grade_ma.values}")
        
        # 4. 特殊字符串映射
        # 使用索引作为映射
        dates = pd.date_range('2023-01-01', periods=8, freq='D')
        wrapper_with_index = vbt.ArrayWrapper(index=dates, columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        ma_with_index = vbt.MappedArray(wrapper_with_index, np.arange(8), stocks)
        
        # 将数值映射为对应的日期
        date_mapped = ma_with_index.apply_mapping('index')
        print(f"日期映射: {date_mapped.values}")
        
        ```
        
        应用场景：
        - 交易信号标签化
        - 数据等级分类
        - 状态转换标记
        - 可视化准备
        - 报告生成
        - 数据解释增强
        
        注意事项：
        - 映射不会改变原始数据结构
        - 未找到映射的值会保持原样
        - 映射后的数据类型可能发生变化
        - 可以与其他MappedArray方法链式使用
        """
        # 处理映射参数：如果未提供，使用实例的mapping属性
        if mapping is None:
            mapping = self.mapping
        
        # 处理特殊字符串映射
        if isinstance(mapping, str):
            if mapping.lower() == 'index':
                mapping = self.wrapper.index    # 使用包装器的索引作为映射
            elif mapping.lower() == 'columns':
                mapping = self.wrapper.columns  # 使用包装器的列作为映射
            mapping = to_mapping(mapping)       # 转换为标准映射格式
        
        # 应用映射到数值并返回新的MappedArray实例
        return self.replace(mapped_arr=apply_mapping(self.values, mapping), **kwargs)

    def to_index(self):
        """
        转换为索引标签 - 数值到时间/索引的转换工具
        
        该方法将映射数组中的数值解释为索引位置，返回对应的索引标签。
        在时间序列分析中，这常用于将数值位置转换为实际的时间戳。
        
        返回值：
            Index: 对应的索引标签数组
        
        使用示例：
        ```python
        import numpy as np
        import pandas as pd
        import vectorbt as vbt
        
        # 创建时间序列数据
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        wrapper = vbt.ArrayWrapper(index=dates, columns=['AAPL'], ndim=2)
        
        # 创建位置数据（索引位置）
        positions = np.array([0, 2, 4, 1, 3])  # 对应不同的日期位置
        stocks = np.array([0, 0, 0, 0, 0])     # 都属于同一列
        
        ma = vbt.MappedArray(wrapper, positions, stocks)
        
        # 转换为对应的日期
        dates_result = ma.to_index()
        print(f"位置: {ma.values}")
        print(f"对应日期: {dates_result}")
        # 位置: [0 2 4 1 3]
        # 对应日期: ['2023-01-01' '2023-01-03' '2023-01-05' '2023-01-02' '2023-01-04']
        
        # 应用场景：将信号发生的位置转换为实际时间
        signal_positions = np.array([10, 25, 50, 75, 100])
        minute_wrapper = vbt.ArrayWrapper(
            index=pd.date_range('2023-01-01 09:30', periods=200, freq='1min'),
            columns=['AAPL'], ndim=2
        )
        
        signal_ma = vbt.MappedArray(minute_wrapper, signal_positions, np.zeros(5))
        signal_times = signal_ma.to_index()
        print(f"信号时间: {signal_times}")
        ```
        
        注意事项：
        - 数值必须是有效的索引位置（0 到 len(index)-1）
        - 超出范围的位置会导致错误
        - 主要用于时间序列和标签化数据
        """
        # 使用包装器的索引进行位置到标签的转换
        return self.wrapper.index[self.values]

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """
        统计默认配置 - 统计分析的配置基础
        
        该属性返回MappedArray统计分析的默认配置，合并了基类的统计配置和
        MappedArray特定的统计设置。这些配置控制着统计指标的计算方式和显示格式。
        
        返回值：
            tp.Kwargs: 统计默认配置字典
        
        配置来源：
            1. StatsBuilderMixin.stats_defaults: 基类统计配置
            2. mapped_array.stats (settings): MappedArray特定统计配置
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建MappedArray实例
        ma = vbt.MappedArray(...)
        
        # 查看默认统计配置
        defaults = ma.stats_defaults
        print(f"默认统计配置: {defaults}")
        
        # 使用自定义配置计算统计信息
        custom_config = {
            'title': '自定义统计报告',
            'settings': {
                'freq': 'D',
                'year_freq': '252D'
            }
        }
        
        # 合并默认配置和自定义配置
        final_config = {**defaults, **custom_config}
        stats = ma.stats(**final_config)
        print(f"统计信息: {stats}")
        ```
        
        配置项说明：
        - freq: 数据频率（如'D'表示日频）
        - year_freq: 年化频率（如'252D'表示252个交易日）
        - title: 统计报告标题
        - template_mapping: 模板映射配置
        - settings: 其他统计设置
        
        注意事项：
        - 该属性是只读的
        - 配置会影响所有统计方法的行为
        - 可以通过settings模块进行全局配置
        """
        # 从设置中获取MappedArray特定的统计配置
        from vectorbt._settings import settings
        mapped_array_stats_cfg = settings['mapped_array']['stats']

        # 合并基类的统计默认配置和MappedArray特定的配置
        return merge_dicts(
            StatsBuilderMixin.stats_defaults.__get__(self),  # 基类的统计默认配置
            mapped_array_stats_cfg                           # MappedArray特定的统计配置
        )

    # 定义统计指标的配置字典
    _metrics: tp.ClassVar[Config] = Config(
        dict(
            # 开始时间指标
            start=dict(
                title='Start',                                          # 指标标题
                calc_func=lambda self: self.wrapper.index[0],         # 计算函数：获取第一个索引
                agg_func=None,                                         # 聚合函数：None表示不聚合
                tags='wrapper'                                         # 标签：标记为包装器相关
            ),
            # 结束时间指标
            end=dict(
                title='End',                                           # 指标标题
                calc_func=lambda self: self.wrapper.index[-1],        # 计算函数：获取最后一个索引
                agg_func=None,                                         # 聚合函数：None表示不聚合
                tags='wrapper'                                         # 标签：标记为包装器相关
            ),
            # 时间周期指标
            period=dict(
                title='Period',                                        # 指标标题
                calc_func=lambda self: len(self.wrapper.index),       # 计算函数：获取索引长度
                apply_to_timedelta=True,                               # 应用到时间增量
                agg_func=None,                                         # 聚合函数：None表示不聚合
                tags='wrapper'                                         # 标签：标记为包装器相关
            ),
            # 数据点数量指标
            count=dict(
                title='Count',                                         # 指标标题
                calc_func='count',                                     # 计算函数：调用count方法
                tags='mapped_array'                                    # 标签：标记为映射数组相关
            ),
            # 均值指标
            mean=dict(
                title='Mean',                                          # 指标标题
                calc_func='mean',                                      # 计算函数：调用mean方法
                inv_check_has_mapping=True,                            # 反向检查是否有映射
                tags=['mapped_array', 'describe']                     # 标签：映射数组和描述统计
            ),
            # 标准差指标
            std=dict(
                title='Std',                                           # 指标标题
                calc_func='std',                                       # 计算函数：调用std方法
                inv_check_has_mapping=True,                            # 反向检查是否有映射
                tags=['mapped_array', 'describe']                     # 标签：映射数组和描述统计
            ),
            # 最小值指标
            min=dict(
                title='Min',                                           # 指标标题
                calc_func='min',                                       # 计算函数：调用min方法
                inv_check_has_mapping=True,                            # 反向检查是否有映射
                tags=['mapped_array', 'describe']                     # 标签：映射数组和描述统计
            ),
            # 中位数指标
            median=dict(
                title='Median',                                        # 指标标题
                calc_func='median',                                    # 计算函数：调用median方法
                inv_check_has_mapping=True,                            # 反向检查是否有映射
                tags=['mapped_array', 'describe']                     # 标签：映射数组和描述统计
            ),
            # 最大值指标
            max=dict(
                title='Max',                                           # 指标标题
                calc_func='max',                                       # 计算函数：调用max方法
                inv_check_has_mapping=True,                            # 反向检查是否有映射
                tags=['mapped_array', 'describe']                     # 标签：映射数组和描述统计
            ),
            # 最小值索引指标
            idx_min=dict(
                title='Min Index',                                     # 指标标题
                calc_func='idxmin',                                    # 计算函数：调用idxmin方法
                inv_check_has_mapping=True,                            # 反向检查是否有映射
                agg_func=None,                                         # 聚合函数：None表示不聚合
                tags=['mapped_array', 'index']                        # 标签：映射数组和索引相关
            ),
            # 最大值索引指标
            idx_max=dict(
                title='Max Index',                                     # 指标标题
                calc_func='idxmax',                                    # 计算函数：调用idxmax方法
                inv_check_has_mapping=True,                            # 反向检查是否有映射
                agg_func=None,                                         # 聚合函数：None表示不聚合
                tags=['mapped_array', 'index']                        # 标签：映射数组和索引相关
            ),
            # 值频次统计指标
            value_counts=dict(
                title='Value Counts',                                  # 指标标题
                calc_func=lambda value_counts: to_dict(value_counts, orient='index_series'),  # 计算函数：转换为字典
                resolve_value_counts=True,                             # 解析值频次
                check_has_mapping=True,                                # 检查是否有映射
                tags=['mapped_array', 'value_counts']                 # 标签：映射数组和值频次相关
            )
        ),
        copy_kwargs=dict(copy_mode='deep')                             # 复制参数：深度复制
    )

    @property
    def metrics(self) -> Config:
        """
        统计指标配置 - 定义可用统计指标的配置对象
        
        该属性返回MappedArray的统计指标配置，定义了所有可用的统计指标及其计算方式。
        这些配置用于stats()方法生成统计报告。
        
        返回值：
            Config: 统计指标配置对象
        
        包含的指标：
            - start: 开始时间
            - end: 结束时间  
            - period: 时间周期
            - count: 数据点数量
            - mean: 均值
            - std: 标准差
            - min: 最小值
            - median: 中位数
            - max: 最大值
            - idx_min: 最小值索引
            - idx_max: 最大值索引
            - value_counts: 值频次统计
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建MappedArray实例
        ma = vbt.MappedArray(...)
        
        # 查看可用的统计指标
        metrics = ma.metrics
        print(f"可用指标: {list(metrics.keys())}")
        
        # 查看特定指标的配置
        mean_config = metrics['mean']
        print(f"均值指标配置: {mean_config}")
        
        # 使用特定指标生成统计报告
        stats = ma.stats(metrics=['mean', 'std', 'count'])
        print(f"选定指标统计: {stats}")
        
        # 自定义指标配置
        custom_metrics = {
            'custom_metric': {
                'title': '自定义指标',
                'calc_func': lambda self: self.mean() * 2,
                'tags': 'custom'
            }
        }
        
        # 合并默认指标和自定义指标
        extended_metrics = {**metrics, **custom_metrics}
        ```
        
        指标配置说明：
            - title: 指标显示名称
            - calc_func: 计算函数（方法名或lambda函数）
            - agg_func: 聚合函数（用于多列汇总）
            - tags: 指标标签（用于分类和过滤）
            - inv_check_has_mapping: 是否检查映射（影响是否显示）
            - apply_to_timedelta: 是否应用到时间增量
            - resolve_value_counts: 是否解析值频次
            - check_has_mapping: 是否检查映射存在
        
        注意事项：
        - 配置是只读的，不能直接修改
        - 可以通过自定义指标扩展功能
        - 标签用于过滤和分类显示
        - 映射检查影响指标的显示与否
        """
        return self._metrics

    # ############# Plotting ############# #

    def histplot(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制直方图 - 数据分布的可视化工具
        
        该方法为每列/组的数据绘制直方图，直观显示数据的分布特征。在量化交易中，
        直方图用于分析收益率分布、风险特征、异常值分布等。
        
        参数说明：
            group_by (GroupByLike, 可选): 分组方式
                - 为每个组绘制直方图
            **kwargs: 传递给底层绘图函数的其他参数
                - bins: 直方图的箱数
                - alpha: 透明度
                - title: 图表标题
                - 其他matplotlib/plotly参数
        
        返回值：
            tp.BaseFigure: 直方图对象
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建收益率数据
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)  # 正态分布的收益率
        stocks = np.random.choice([0, 1, 2], 1000)
        wrapper = vbt.ArrayWrapper(columns=['AAPL', 'GOOGL', 'MSFT'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, returns, stocks)
        
        # 绘制收益率分布直方图
        fig = ma.histplot(bins=50, alpha=0.7, title='收益率分布')
        
        # 按行业分组绘制
        group_by = ['科技股', '科技股', '其他']
        fig_grouped = ma.histplot(group_by=group_by, bins=30)
        
        # 用于风险分析
        # 观察收益率分布的偏斜度和峰态
        # 识别异常值和极端事件
        ```
        """
        # 将数据转换为pandas格式后调用vectorbt的直方图绘制方法
        return self.to_pd(group_by=group_by, ignore_index=True).vbt.histplot(**kwargs)

    def boxplot(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制箱线图 - 数据分布和异常值的可视化工具
        
        该方法为每列/组的数据绘制箱线图，显示数据的中位数、四分位数和异常值。
        在量化交易中，箱线图用于比较不同资产的风险收益特征、识别异常表现等。
        
        参数说明：
            group_by (GroupByLike, 可选): 分组方式
                - 为每个组绘制箱线图
            **kwargs: 传递给底层绘图函数的其他参数
                - whis: 须线长度倍数
                - showfliers: 是否显示异常值
                - title: 图表标题
                - 其他matplotlib/plotly参数
        
        返回值：
            tp.BaseFigure: 箱线图对象
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        
        # 创建不同风险特征的收益率数据
        np.random.seed(42)
        low_risk = np.random.normal(0.005, 0.01, 200)   # 低风险
        med_risk = np.random.normal(0.008, 0.02, 200)   # 中风险
        high_risk = np.random.normal(0.012, 0.04, 200)  # 高风险
        
        returns = np.concatenate([low_risk, med_risk, high_risk])
        stocks = np.concatenate([np.zeros(200), np.ones(200), np.full(200, 2)])
        wrapper = vbt.ArrayWrapper(columns=['低风险', '中风险', '高风险'], ndim=2)
        
        ma = vbt.MappedArray(wrapper, returns, stocks)
        
        # 绘制箱线图比较风险特征
        fig = ma.boxplot(title='不同风险资产的收益率分布')
        
        # 按行业分组比较
        group_by = ['股票', '股票', '债券']
        fig_grouped = ma.boxplot(group_by=group_by)
        
        # 用于风险管理
        # 识别极端收益率事件
        # 比较不同资产的风险水平
        # 发现异常表现的时期
        ```
        """
        # 将数据转换为pandas格式后调用vectorbt的箱线图绘制方法
        return self.to_pd(group_by=group_by, ignore_index=True).vbt.boxplot(**kwargs)

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """
        绘图默认配置 - 图表绘制的配置基础
        
        该属性返回MappedArray绘图的默认配置，合并了基类的绘图配置和
        MappedArray特定的绘图设置。这些配置控制着图表的样式、布局和行为。
        
        返回值：
            tp.Kwargs: 绘图默认配置字典
        
        配置来源：
            1. PlotsBuilderMixin.plots_defaults: 基类绘图配置
            2. mapped_array.plots (settings): MappedArray特定绘图配置
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建MappedArray实例
        ma = vbt.MappedArray(...)
        
        # 查看默认绘图配置
        defaults = ma.plots_defaults
        print(f"默认绘图配置: {defaults}")
        
        # 使用自定义配置绘制图表
        custom_config = {
            'title': '自定义图表',
            'template': 'plotly_dark',
            'width': 800,
            'height': 600
        }
        
        # 合并默认配置和自定义配置
        final_config = {**defaults, **custom_config}
        fig = ma.plots(**final_config)
        ```
        
        配置项说明：
        - template: 图表模板（如'plotly', 'plotly_dark'）
        - width/height: 图表尺寸
        - title: 图表标题
        - xaxis/yaxis: 坐标轴配置
        - legend: 图例配置
        - colors: 颜色配置
        - layout: 布局配置
        
        注意事项：
        - 该属性是只读的
        - 配置会影响所有绘图方法的行为
        - 可以通过settings模块进行全局配置
        """
        # 从设置中获取MappedArray特定的绘图配置
        from vectorbt._settings import settings
        mapped_array_plots_cfg = settings['mapped_array']['plots']

        # 合并基类的绘图默认配置和MappedArray特定的配置
        return merge_dicts(
            PlotsBuilderMixin.plots_defaults.__get__(self),  # 基类的绘图默认配置
            mapped_array_plots_cfg                           # MappedArray特定的绘图配置
        )

    # 定义子图配置字典
    _subplots: tp.ClassVar[Config] = Config(
        dict(
            # pandas转换绘图子图
            to_pd_plot=dict(
                check_is_not_grouped=True,                    # 检查是否未分组
                plot_func='to_pd.vbt.plot',                  # 绘图函数：转换为pandas后使用vbt绘图
                pass_trace_names=False,                       # 不传递轨迹名称
                tags='mapped_array'                           # 标签：标记为映射数组相关
            )
        ),
        copy_kwargs=dict(copy_mode='deep')                    # 复制参数：深度复制
    )

    @property
    def subplots(self) -> Config:
        """
        子图配置 - 定义可用子图的配置对象
        
        该属性返回MappedArray的子图配置，定义了所有可用的子图类型及其绘制方式。
        这些配置用于plots()方法生成复合图表。
        
        返回值：
            Config: 子图配置对象
        
        包含的子图：
            - to_pd_plot: 转换为pandas后的标准绘图
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建MappedArray实例
        ma = vbt.MappedArray(...)
        
        # 查看可用的子图类型
        subplots = ma.subplots
        print(f"可用子图: {list(subplots.keys())}")
        
        # 查看特定子图的配置
        plot_config = subplots['to_pd_plot']
        print(f"绘图配置: {plot_config}")
        
        # 使用特定子图绘制图表
        fig = ma.plots(subplots=['to_pd_plot'])
        
        # 自定义子图配置
        custom_subplots = {
            'custom_plot': {
                'plot_func': 'histplot',
                'title': '自定义直方图',
                'tags': 'custom'
            }
        }
        
        # 合并默认子图和自定义子图
        extended_subplots = {**subplots, **custom_subplots}
        ```
        
        子图配置说明：
            - plot_func: 绘图函数（方法名或函数路径）
            - title: 子图标题
            - tags: 子图标签（用于分类和过滤）
            - check_is_not_grouped: 是否检查未分组状态
            - pass_trace_names: 是否传递轨迹名称
            - subplot_kwargs: 子图特定参数
        
        注意事项：
        - 配置是只读的，不能直接修改
        - 可以通过自定义子图扩展功能
        - 标签用于过滤和分类显示
        - 检查条件影响子图的显示与否
        """
        return self._subplots


# 文档生成相关的字典，用于自动生成API文档
__pdoc__ = dict()
# 重写统计指标的文档
MappedArray.override_metrics_doc(__pdoc__)
# 重写子图的文档
MappedArray.override_subplots_doc(__pdoc__)
