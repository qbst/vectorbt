# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT RECORDS MODULE: 列映射器 (Column Mapper)
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于处理列映射和列索引优化的核心模块。
ColumnMapper类是Records和MappedArray类的重要组成部分，负责管理和优化列数据的索引访问。

核心设计思想：
1. **双重索引策略**：根据数据排序状态智能选择列范围(col_range)或列映射(col_map)
   - 排序数据使用col_range：O(1)查找速度，内存占用少，适合静态数据
   - 未排序数据使用col_map：O(log n)查找速度，内存占用稍高，适合动态数据

2. **高性能优化**：使用Numba JIT编译的底层函数，实现接近C语言速度的列操作
   - col_range_nb：为排序数据构建范围索引，支持二分查找
   - col_map_nb：为未排序数据构建哈希映射，支持快速查找
   - 自动排序检测：智能判断数据状态，选择最优索引策略

3. **分组计算支持**：与ArrayWrapper的分组机制深度集成
   - 支持按组重新映射列数据
   - 保持分组语义的一致性
   - 优化分组操作的性能

4. **缓存优化机制**：使用装饰器缓存计算结果，避免重复计算
   - cached_property：缓存属性计算结果，如col_range、col_map
   - cached_method：缓存方法调用结果，如get_col_range、get_col_map
   - 显著提升大数据量场景下的性能
"""

# 导入vectorbt的类型定义模块，提供类型注解支持
from vectorbt import _typing as tp
# 导入数组包装器相关类，用于数据包装和元数据管理
from vectorbt.base.array_wrapper import ArrayWrapper, Wrapping
# 导入数组重塑函数，用于数组维度转换
from vectorbt.base.reshape_fns import to_1d_array
# 导入records模块的Numba编译函数，提供高性能的列操作
from vectorbt.records import nb
# 导入装饰器，用于缓存属性和方法的计算结果
from vectorbt.utils.decorators import cached_property, cached_method


class ColumnMapper(Wrapping):
    """
    列映射器 - vectorbt框架中专门用于优化列数据访问的核心类
    
    ColumnMapper是Records和MappedArray类使用的重要组件，专门负责管理列和组的元数据，
    为这些类提供高效的列级别操作能力。该类的设计核心是根据数据的排序状态智能选择
    最优的索引策略，从而在保证功能完整性的同时最大化性能。
    
    设计特点：
    1. **自适应索引策略**：根据列数组的排序状态自动选择col_range或col_map
    2. **性能优化**：使用Numba编译的底层函数，实现极致的查找和选择性能
    3. **分组支持**：与ArrayWrapper的分组机制深度集成，支持复杂的分组操作
    4. **缓存机制**：智能缓存计算结果，避免重复计算开销
    
    核心方法：
    - col_range: 适用于排序数据的范围索引（更快，内存占用少）
    - col_map: 适用于未排序数据的映射索引（更灵活，支持乱序数据）
    - is_sorted: 检查数据排序状态，决定使用哪种索引策略
    - _col_idxs_meta: 根据索引策略选择和重组数据
    
    使用示例：
    ```python
    import numpy as np
    import vectorbt as vbt
    
    # 创建数组包装器
    wrapper = vbt.ArrayWrapper(
        index=pd.date_range('2023-01-01', periods=100, freq='D'),
        columns=['AAPL', 'GOOGL', 'MSFT'],
        ndim=2
    )
    
    # 创建列数组（表示数据属于哪一列）
    col_arr = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])  # 交易分别属于不同的股票
    
    # 创建列映射器
    col_mapper = vbt.ColumnMapper(wrapper, col_arr)
    
    # 检查数据排序状态
    print(f"数据已排序: {col_mapper.is_sorted()}")  # True
    
    # 获取列范围索引（适用于排序数据）
    col_range = col_mapper.col_range
    print(f"列范围索引: {col_range}")
    
    # 获取列映射索引（适用于未排序数据）
    col_map = col_mapper.col_map
    print(f"列映射索引: {col_map}")
    
    # 选择特定列的数据
    selected_cols = np.array([0, 2])  # 选择AAPL和MSFT
    indices, new_col_arr = col_mapper._col_idxs_meta(selected_cols)
    print(f"选择的索引: {indices}")
    print(f"新的列数组: {new_col_arr}")
    
    # 分组使用示例
    # 将三个股票分成两组：科技股 (AAPL, GOOGL) 和其他 (MSFT)
    group_by = ['tech', 'tech', 'other']
    grouped_col_arr = col_mapper.get_col_arr(group_by=group_by)
    print(f"分组后的列数组: {grouped_col_arr}")
    ```
    
    性能说明：
    - 排序数据使用col_range：查找时间复杂度O(1)，内存占用最小
    - 未排序数据使用col_map：查找时间复杂度O(log n)，内存占用稍高
    - 自动选择策略：根据数据特性智能选择最优方案
    - 缓存优化：避免重复计算，提升整体性能
    """

    def __init__(self, wrapper: ArrayWrapper, col_arr: tp.Array1d, **kwargs) -> None:
        """
        初始化列映射器
        
        参数说明：
            wrapper (ArrayWrapper): 数组包装器，提供索引、列名和分组等元数据
                - 包含时间索引（如日期序列）
                - 包含列名（如股票代码）
                - 包含分组信息（如按行业分组）
                - 提供数组的形状和维度信息
            col_arr (tp.Array1d): 一维列数组，标识每个数据点属于哪一列
                - 长度与数据点数量相同
                - 值为列索引（0, 1, 2, ...）
                - 可以是排序或未排序的
            **kwargs: 其他关键字参数，传递给基类构造函数
        
        初始化过程：
        1. 调用基类Wrapping的构造函数，设置包装器和参数
        2. 保存包装器和列数组的引用
        3. 后续的索引构建采用延迟计算方式（通过缓存装饰器实现）
        """
        # 调用基类构造函数，初始化包装器和配置
        Wrapping.__init__(
            self,
            wrapper,  # 数组包装器，提供元数据
            col_arr=col_arr,  # 列数组，标识数据点的列归属
            **kwargs  # 其他配置参数
        )
        # 保存包装器引用，用于后续的索引和分组操作
        self._wrapper = wrapper
        # 保存列数组引用，这是列映射器的核心数据
        self._col_arr = col_arr

    def _col_idxs_meta(self, col_idxs: tp.Array1d) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """
        获取列索引的元数据 - 列映射器的核心方法
        
        该方法是ColumnMapper的核心功能，根据数据的排序状态智能选择最优的索引策略：
        - 如果数据已排序：使用col_range_select_nb（更快，O(1)查找）
        - 如果数据未排序：使用col_map_select_nb（更灵活，支持乱序数据）
        
        参数说明：
            col_idxs (tp.Array1d): 要选择的列索引数组
                - 例如：[0, 2] 表示选择第0列和第2列
                - 可以是任意顺序的列索引
        
        返回值：
            tp.Tuple[tp.Array1d, tp.Array1d]: 包含两个数组的元组
                - new_indices: 选择的元素在原数组中的索引位置
                - new_col_arr: 重新编号后的列数组（从0开始连续编号）
        
        算法逻辑：
        1. 检查数据排序状态（通过is_sorted()方法）
        2. 排序数据：使用col_range_select_nb进行快速范围选择
        3. 未排序数据：使用col_map_select_nb进行映射选择
        4. 返回选择的索引和重新编号的列数组
        
        性能特点：
        - 自动选择最优算法：根据数据特性选择最快的实现
        - 使用Numba编译：底层函数经过JIT编译，性能接近C语言
        - 内存效率：避免不必要的数据复制和内存分配
        """
        # 检查列数组是否已排序，这决定了使用哪种索引策略
        if self.is_sorted():
            # 排序数据使用范围选择：更快的O(1)查找算法
            # col_range: 每列的起始和结束位置
            # col_idxs: 要选择的列索引
            new_indices, new_col_arr = nb.col_range_select_nb(self.col_range, to_1d_array(col_idxs))  # faster
        else:
            # 未排序数据使用映射选择：更灵活的查找算法
            # col_map: 列到索引的映射关系
            # col_idxs: 要选择的列索引
            new_indices, new_col_arr = nb.col_map_select_nb(self.col_map, to_1d_array(col_idxs))
        
        # 返回选择的索引和重新编号的列数组
        return new_indices, new_col_arr

    @property
    def wrapper(self) -> ArrayWrapper:
        return self._wrapper

    @property
    def col_arr(self) -> tp.Array1d:
        return self._col_arr

    @cached_method
    def get_col_arr(self, group_by: tp.GroupByLike = None) -> tp.Array1d:
        """
        获取支持分组的列数组
        
        该方法在原始列数组的基础上应用分组逻辑，将列重新映射到组。
        这是vectorbt框架中实现分组分析的核心机制。
        
        参数说明：
            group_by (tp.GroupByLike, optional): 分组参数，可以是：
                - None: 不分组，返回原始列数组
                - 列表/数组: 每个元素指定对应列的分组
                - 字符串: 按列名进行分组
                - 其他pandas groupby支持的格式
        
        返回值：
            tp.Array1d: 分组后的列数组
        
        工作原理：
        1. 通过wrapper.grouper获取分组器
        2. 调用get_groups获取分组数组
        3. 如果有分组，使用分组数组重新映射列
        4. 如果没有分组，返回原始列数组
        
        使用示例：
        ```python
        # 原始列数组：[0, 0, 1, 1, 2, 2] (对应AAPL, GOOGL, MSFT)
        col_arr = np.array([0, 0, 1, 1, 2, 2])
        
        # 按行业分组：科技股(AAPL, GOOGL)和其他(MSFT)
        group_by = ['tech', 'tech', 'other']
        grouped_col_arr = mapper.get_col_arr(group_by=group_by)
        # 结果：[0, 0, 0, 0, 1, 1] (前4个属于科技股组，后2个属于其他组)
        ```
        
        缓存机制：
        - 使用@cached_method装饰器缓存结果
        - 相同参数的调用直接返回缓存结果
        - 显著提升重复调用的性能
        """
        # 从包装器获取分组数组
        group_arr = self.wrapper.grouper.get_groups(group_by=group_by)
        
        # 如果有分组信息，重新映射列数组
        if group_arr is not None:
            # 使用分组数组重新映射：group_arr[col_idx] -> group_idx
            col_arr = group_arr[self.col_arr]
        else:
            # 没有分组，返回原始列数组
            col_arr = self.col_arr
        
        return col_arr

    @cached_property
    def col_range(self) -> tp.ColRange:
        """
        获取列范围索引 - 适用于排序数据的高效索引方案
        
        列范围索引是一种专门为排序数据设计的索引结构，通过记录每列的起始和结束位置
        来实现O(1)时间复杂度的列查找。这种索引方案在数据已排序的情况下比col_map
        更快且内存占用更少。
        
        数据结构：
        - 每个元素包含一列的起始和结束位置
        - 例如：[(0, 3), (3, 6), (6, 9)] 表示第0列占位置0-2，第1列占位置3-5，第2列占位置6-8
        
        适用场景：
        - 数据已按列排序
        - 需要频繁的列查找操作
        - 内存使用要求较高的场景
        - 静态数据（不经常变化）
        
        性能特点：
        - 查找时间复杂度：O(1)
        - 内存占用：O(列数)，非常节省内存
        - 构建时间：O(n)，其中n是数据点数量
        - 最适合：排序的密集数据
        
        使用示例：
        ```python
        # 排序的列数组
        col_arr = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        # 生成的列范围索引
        col_range = [(0, 3), (3, 6), (6, 9)]
        # 表示：第0列在位置0-2，第1列在位置3-5，第2列在位置6-8
        ```
        
        与col_map的对比：
        - col_range: 更快，内存占用少，但要求数据排序
        - col_map: 更灵活，支持乱序数据，但内存占用稍高
        
        缓存机制：
        - 使用@cached_property装饰器缓存结果
        - 首次访问时计算，后续访问直接返回缓存
        - 避免重复计算的开销
        
        返回值：
            tp.ColRange: 列范围索引，包含每列的起始和结束位置
        """
        # 调用Numba编译的高性能函数构建列范围索引
        # col_arr: 列数组，必须是排序的
        # len(self.wrapper.columns): 总列数
        return nb.col_range_nb(self.col_arr, len(self.wrapper.columns))

    @cached_method
    def get_col_range(self, group_by: tp.GroupByLike = None) -> tp.ColRange:
        """
        获取支持分组的列范围索引
        
        该方法在col_range的基础上添加了分组支持，允许在分组后的数据上构建列范围索引。
        这是实现分组分析的重要功能，可以将数据按组重新组织后构建高效的索引。
        
        参数说明：
            group_by (tp.GroupByLike, optional): 分组参数
                - None: 不分组，返回原始列范围索引
                - 其他值: 按指定方式分组后构建索引
        
        返回值：
            tp.ColRange: 分组后的列范围索引
        
        工作原理：
        1. 检查是否需要分组
        2. 如果不需要分组，直接返回原始列范围索引
        3. 如果需要分组，获取分组后的列数组和列信息
        4. 基于分组后的数据构建新的列范围索引
        
        使用示例：
        ```python
        # 原始数据：3个股票，每个股票3条记录
        col_arr = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        # 按行业分组：科技股(股票0,1)，其他(股票2)
        group_by = ['tech', 'tech', 'other']
        
        # 分组后的列范围索引
        grouped_col_range = mapper.get_col_range(group_by=group_by)
        # 结果：[(0, 6), (6, 9)] 表示科技股组占位置0-5，其他组占位置6-8
        ```
        
        性能优化：
        - 使用@cached_method装饰器缓存结果
        - 避免重复计算分组索引的开销
        - 对于频繁使用的分组查询提供高性能支持
        """
        # 检查是否需要分组
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            # 不需要分组，直接返回原始列范围索引
            return self.col_range
        
        # 需要分组，获取分组后的列数组和列信息
        col_arr = self.get_col_arr(group_by=group_by)
        columns = self.wrapper.get_columns(group_by=group_by)
        
        # 基于分组后的数据构建新的列范围索引
        return nb.col_range_nb(col_arr, len(columns))

    @cached_property
    def col_map(self) -> tp.ColMap:
        """
        获取列映射索引 - 适用于未排序数据的灵活索引方案
        
        列映射索引是一种专门为未排序数据设计的索引结构，通过构建列到索引的映射关系
        来实现高效的列查找。虽然比col_range稍慢，但支持任意顺序的数据，提供了更大的灵活性。
        
        数据结构：
        - col_idxs: 重新排列的索引数组，按列分组存储
        - col_lens: 每列的记录数量数组
        - 支持O(log n)时间复杂度的列查找
        
        适用场景：
        - 数据未排序或无法排序
        - 动态数据（经常添加、删除）
        - 需要保持原始数据顺序
        - 实时数据流处理
        
        性能特点：
        - 查找时间复杂度：O(log n)
        - 内存占用：O(数据点数量)，比col_range稍高
        - 构建时间：O(n log n)，其中n是数据点数量
        - 最适合：未排序的稀疏数据
        
        使用示例：
        ```python
        # 未排序的列数组（实时交易数据的典型情况）
        col_arr = np.array([2, 0, 1, 0, 2, 1, 0])
        
        # 生成的列映射索引
        col_map = {
            'col_idxs': [1, 3, 6, 2, 5, 0, 4],  # 按列分组的原始索引
            'col_lens': [3, 2, 2]                # 每列的记录数量
        }
        # 表示：列0有3个记录在位置[1,3,6]，列1有2个记录在位置[2,5]，列2有2个记录在位置[0,4]
        ```
        
        与col_range的对比：
        - col_map: 更灵活，支持乱序数据，但查找稍慢
        - col_range: 更快，内存占用少，但要求数据排序
        
        缓存机制：
        - 使用@cached_property装饰器缓存结果
        - 首次访问时计算，后续访问直接返回缓存
        - 避免重复构建映射的开销
        
        返回值：
            tp.ColMap: 列映射索引，包含列到索引的映射关系
        """
        # 调用Numba编译的高性能函数构建列映射索引
        # col_arr: 列数组，可以是未排序的
        # len(self.wrapper.columns): 总列数
        return nb.col_map_nb(self.col_arr, len(self.wrapper.columns))

    @cached_method
    def get_col_map(self, group_by: tp.GroupByLike = None) -> tp.ColMap:
        """
        获取支持分组的列映射索引
        
        该方法在col_map的基础上添加了分组支持，允许在分组后的数据上构建列映射索引。
        这对于需要在分组数据上进行灵活查找的场景非常有用。
        
        参数说明：
            group_by (tp.GroupByLike, optional): 分组参数
                - None: 不分组，返回原始列映射索引
                - 其他值: 按指定方式分组后构建索引
        
        返回值：
            tp.ColMap: 分组后的列映射索引
        
        工作原理：
        1. 检查是否需要分组
        2. 如果不需要分组，直接返回原始列映射索引
        3. 如果需要分组，获取分组后的列数组和列信息
        4. 基于分组后的数据构建新的列映射索引
        
        使用示例：
        ```python
        # 原始未排序数据：3个股票的交易记录
        col_arr = np.array([2, 0, 1, 0, 2, 1])
        
        # 按行业分组：科技股(股票0,1)，其他(股票2)
        group_by = ['tech', 'tech', 'other']
        
        # 分组后的列映射索引
        grouped_col_map = mapper.get_col_map(group_by=group_by)
        # 结果会将原来的3列重新映射为2个组
        ```
        
        性能优化：
        - 使用@cached_method装饰器缓存结果
        - 避免重复计算分组映射的开销
        - 对于频繁使用的分组查询提供高性能支持
        """
        # 检查是否需要分组
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            # 不需要分组，直接返回原始列映射索引
            return self.col_map
        
        # 需要分组，获取分组后的列数组和列信息
        col_arr = self.get_col_arr(group_by=group_by)
        columns = self.wrapper.get_columns(group_by=group_by)
        
        # 基于分组后的数据构建新的列映射索引
        return nb.col_map_nb(col_arr, len(columns))

    @cached_method
    def is_sorted(self) -> bool:
        return nb.is_col_sorted_nb(self.col_arr)
