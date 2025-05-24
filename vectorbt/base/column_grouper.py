# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT BASE MODULE: COLUMN GROUPER
================================================================================

文件作用概述：
本文件是vectorbt库中负责列分组管理的核心模块，为DataFrame列的分组操作提供了一套完整的
基础设施。在量化交易回测中，经常需要对多个资产、多个策略参数或多个时间周期的数据进行
分组处理，该模块正是为了满足这一需求而设计。

核心设计理念：
1. **统一的分组接口**：提供group_by_to_index函数，将多种形式的分组规范（布尔值、整数、
   字符串、序列等）统一转换为标准的pandas Index格式，确保后续操作的一致性。

2. **高性能分组处理**：通过ColumnGrouper类提供缓存机制和预计算的分组元数据，避免重复
   计算分组信息，显著提升批量数据处理的性能。

3. **灵活的分组策略**：支持多种分组方式包括按名称分组、按位置分组、按级别分组、
   自定义映射分组等，满足不同场景下的分组需求。

4. **分组状态管理**：提供分组启用/禁用/修改的权限控制，确保分组操作的安全性和可控性。

主要应用场景：
- **批量回测**：对不同策略参数组合的回测结果进行分组聚合
- **多资产分析**：按行业、市值、地区等维度对股票进行分组分析  
- **时间序列处理**：按时间周期（日、周、月、年）对数据进行分组统计
- **性能优化**：通过预计算分组信息，加速重复的分组操作

技术特点：
- 使用Numba JIT编译技术加速核心分组长度计算函数
- 提供完整的分组验证和错误处理机制
- 支持MultiIndex多级索引的复杂分组操作
- 集成缓存装饰器，避免重复计算提升性能

与其他模块的协作：
- 与index_fns模块协作处理复杂的索引选择和级别操作
- 与reshape_fns模块配合实现数据的重构和广播
- 为combine_fns模块的批量操作提供分组基础设施
- 作为vectorbt高级功能（如组合优化、风险分析）的底层支持
"""

import numpy as np
import pandas as pd
from numba import njit

from vectorbt import _typing as tp
from vectorbt.base import index_fns
from vectorbt.utils import checks
from vectorbt.utils.array_ import is_sorted
from vectorbt.utils.config import Configured
from vectorbt.utils.decorators import cached_method

GroupByT = tp.Union[None, bool, tp.Index]


def group_by_to_index(index: tp.Index, group_by: tp.GroupByLike) -> GroupByT:
    """
    group_by = None/False时，返回None/False
    group_by = True时，生成一个长度与原始索引相同、所有元素都为'group'的pandas Index
    group_by为int或str时，调用index_fns.select_levels从index中选取对应的Index
    group_by为序列时，如果长度与index不同，并且元素为int或str，则从index中选取对应的Index
    否则，将 group_by 转换为pandas Index 返回 

    Args:
        index (tp.Index): 原始的 pandas Index 对象，表示要进行分组的列索引。
                         通常是 DataFrame 的列索引，用作分组操作的基础。
        group_by (tp.GroupByLike): 分组映射器，支持多种类型：
            - None 或 False: 不进行分组，保持原始结构
            - True: 将所有列归为一个组
            - int 或 str: 按 MultiIndex 的指定级别进行分组
            - 序列: 自定义分组映射，长度需与 index 相同
    
    Returns:
        GroupByT: 转换后的分组对象，类型为 Union[None, bool, tp.Index]：
            - None 或 False: 当不需要分组时即 group_by 为 None 或 False 时
            - pd.Index: 标准化的分组索引
    """
    if group_by is None or group_by is False:
        return group_by
    
    # 当 group_by 为 True 时，生成一个长度与原始索引相同、所有元素都为 'group' 的 pandas Index
    if group_by is True:
        group_by = pd.Index(['group'] * len(index))  # one group
    # 处理 group_by 为单个整数或字符串的情况
    elif isinstance(group_by, (int, str)):
        group_by = index_fns.select_levels(index, group_by)
    # 处理 group_by 为序列（列表、元组等）的情况
    elif checks.is_sequence(group_by):
        if len(group_by) != len(index) \
                and isinstance(group_by[0], (int, str)) \
                and isinstance(index, pd.MultiIndex) \
                and len(group_by) <= len(index.names):
            try:
                group_by = index_fns.select_levels(index, group_by)
            except (IndexError, KeyError):
                pass
    
    # 可能不属于上述任一种情况
    if not isinstance(group_by, pd.Index):
        group_by = pd.Index(group_by)
    
    if len(group_by) != len(index):
        raise ValueError("group_by and index must have the same length")
    
    return group_by


def get_groups_and_index(index: tp.Index, group_by: tp.GroupByLike) -> tp.Tuple[tp.Array1d, tp.Index]:
    """
    根据 group_by_to_index(index, group_by) 的结果，返回 (对应的去重Index下标，去重Index)
    
    Args:
        index (tp.Index): 原始的pandas Index对象，表示要进行分组的源索引
                         通常是DataFrame的列索引或行索引，用作分组操作的基础
                         
        group_by (tp.GroupByLike): 分组映射器，支持多种类型：
                                 - None或False: 不进行分组，每个元素自成一组
                                 - True: 将所有元素归为一个组
                                 - int或str: 按MultiIndex的指定级别进行分组
                                 - 序列: 自定义分组映射，长度需与index相同
    
    Returns:
        tp.Tuple[tp.Array1d, tp.Index]: 包含两个元素的元组：
            - 第一个元素 (tp.Array1d): 组编码数组，整数类型的NumPy数组
              每个元素表示原始索引中对应位置的元素属于哪个组（从0开始编号）
              
            - 第二个元素 (tp.Index): 分组后的新索引，包含所有唯一的分组标识
              保持原始分组值的数据类型和元数据（如名称）
    """
    if group_by is None or group_by is False:
        return np.arange(len(index)), index

    group_by = group_by_to_index(index, group_by)
    
    # factorize函数的作用是将一个包含重复值的序列转换为：
    # - codes: 整数编码数组，相同值对应相同编码，编码从0开始递增
    # - uniques: 按首次出现顺序排列的唯一值数组
    # 例如：MultiIndex([('tech', 'large'), ('tech', 'small'), ('tech', 'large'), ('finance', 'large')])
    # 会被转换为：codes=[0,1,0,2], uniques=[('tech','large'), ('tech','small'), ('finance','large')]
    codes, uniques = pd.factorize(group_by)
    
    # 确保uniques是pandas Index类型
    if not isinstance(uniques, pd.Index):
        new_index = pd.Index(uniques)
    else:
        new_index = uniques
    
    if isinstance(group_by, pd.MultiIndex):
        new_index.names = group_by.names
    elif isinstance(group_by, (pd.Index, pd.Series)):
        new_index.name = group_by.name
    
    return codes, new_index


@njit(cache=True)
def get_group_lens_nb(groups: tp.Array1d) -> tp.Array1d:
    """
    对于升序数组groups，返回其各元素对应数量
    
    注意：该函数要求输入的groups数组必须是连贯且已排序的（如[0,0,1,2,2,...]），
    不支持乱序的分组编码（如[0,2,0,1,2]），这是为了保证算法的高效性和正确性。
    
    Args:
        groups (tp.Array1d): 一维整数数组，包含每个元素的分组编码
                           必须满足以下条件：
                           - 数组元素为非负整数，表示分组编号（从0开始）
                           - 数组必须是连贯且已排序的，相同组的元素必须连续出现
                           - 例如：有效输入 [0,0,0,1,1,2,2,2,2]
                           - 例如：无效输入 [0,2,0,1,2] (不连贯，会抛出异常)
    
    Returns:
        tp.Array1d: 一维整数数组，包含每个分组的元素数量
                   - 数组长度等于唯一分组的数量
                   - 第i个元素表示编号为i的分组包含多少个元素
                   - 数据类型为np.int64，确保能处理大规模数据
                   - 例如：对于输入[0,0,0,1,1,2,2,2,2]，输出[3,2,4]
                          表示组0有3个元素，组1有2个元素，组2有4个元素
    
    Examples:
        >>> import numpy as np
        >>> from numba import njit
        >>> 
        >>> # 示例1：典型的分组长度计算
        >>> groups = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        >>> lengths = get_group_lens_nb(groups)
        >>> print(f"分组编码: {groups}")
        >>> print(f"各组长度: {lengths}")  # [3, 2, 4]
    """
    result = np.empty(groups.shape[0], dtype=np.int64)
    
    j = 0
    last_group = -1
    group_len = 0
    
    for i in range(groups.shape[0]):
        cur_group = groups[i]
        
        if cur_group < last_group:
            raise ValueError("Groups must be coherent and sorted (such as [0, 0, 1, 2, 2, ...])")
        
        if cur_group != last_group:
            if last_group != -1:
                result[j] = group_len
                j += 1
                group_len = 0
            last_group = cur_group
        
        group_len += 1
        
        if i == groups.shape[0] - 1:
            result[j] = group_len
            j += 1
            group_len = 0
    
    return result[:j]


class ColumnGrouper(Configured):
    """列分组器类：用于管理DataFrame列的分组操作的核心工具类。
    
    该类是vectorbt库中列分组功能的核心实现，为DataFrame列提供了完整的分组管理基础设施。
    它不仅支持多种分组方式，还提供了权限控制、缓存优化和分组状态管理等高级功能。
    """

    def __init__(self, columns: tp.Index, group_by: tp.GroupByLike = None, allow_enable: bool = True,
                 allow_disable: bool = True, allow_modify: bool = True) -> None:
        """初始化列分组器实例。
        
        构造函数负责验证输入参数、设置分组映射、配置权限控制，并调用父类的初始化方法。
        所有参数都会被存储为私有属性，以支持不可变设计模式和缓存优化。
        
        Args:
            columns (tp.Index): 要进行分组的原始列索引
                              必须是pandas Index对象，表示DataFrame的列标识
                              支持简单Index和MultiIndex两种类型
                              例如：pd.Index(['A', 'B', 'C']) 或复杂的多级索引
                              
            group_by (tp.GroupByLike, optional): 分组映射器，默认为None
                                               支持多种类型的分组规范：
                                               - None/False: 不分组，每列自成一组
                                               - True: 所有列归为一个组  
                                               - int: 按MultiIndex指定位置级别分组
                                               - str: 按MultiIndex指定名称级别分组
                                               - sequence: 自定义分组映射
                                               
            allow_enable (bool, optional): 是否允许启用分组，默认为True
                                         当group_by为None时，控制是否可以后续启用分组
                                         设为False可防止意外启用分组操作
                                         
            allow_disable (bool, optional): 是否允许禁用分组，默认为True
                                          当group_by不为None时，控制是否可以后续禁用分组
                                          设为False可防止意外禁用分组操作
                                          
            allow_modify (bool, optional): 是否允许修改分组，默认为True
                                         控制是否可以更改分组结构（不影响标签更改）
                                         设为False可防止分组结构被意外修改
        """
        # 调用父类Configured的初始化方法，传递所有参数以支持配置管理功能
        # Configured类提供了配置序列化、缓存和不可变性等基础设施
        Configured.__init__(
            self,
            columns=columns,
            group_by=group_by,
            allow_enable=allow_enable,
            allow_disable=allow_disable,
            allow_modify=allow_modify
        )

        checks.assert_instance_of(columns, pd.Index)
        self._columns = columns  
        
        if group_by is None or group_by is False:
            # 当不需要分组时，将group_by设置为None，表示禁用分组状态
            self._group_by = None
        else:
            self._group_by = group_by_to_index(columns, group_by)

        self._allow_enable = allow_enable    
        self._allow_disable = allow_disable   
        self._allow_modify = allow_modify    

    @property
    def columns(self) -> tp.Index:
        return self._columns

    @property
    def group_by(self) -> GroupByT:
        return self._group_by

    @property
    def allow_enable(self) -> bool:
        return self._allow_enable

    @property
    def allow_disable(self) -> bool:
        return self._allow_disable

    @property  
    def allow_modify(self) -> bool:
        return self._allow_modify

    def is_grouped(self, group_by: tp.GroupByLike = None) -> bool:
        '''判断group_by或self.group_by不为None'''
        if group_by is False:
            return False
        if group_by is None:
            group_by = self.group_by
        return group_by is not None

    def is_grouping_enabled(self, group_by: tp.GroupByLike = None) -> bool:
        '''判断group_by为None而self.group_by不为None'''
        return self.group_by is None and self.is_grouped(group_by=group_by)

    def is_grouping_disabled(self, group_by: tp.GroupByLike = None) -> bool:
        '''判断group_by和self.group_by都不为None'''
        return self.group_by is not None and not self.is_grouped(group_by=group_by)

    @cached_method
    def is_grouping_modified(self, group_by: tp.GroupByLike = None) -> bool:
        """检查列分组是否已被修改。
        
        此方法检测分组结构是否发生变化，但不关心分组标签的更改。
        这是一个缓存方法，避免重复计算复杂的分组比较操作。
        
        分组修改的判断逻辑：
        1. 首先排除无效输入和相同状态的情况
        2. 对于两个都是Index的情况，先比较Index是否相等
        3. 如果Index不相等，则比较实际的分组编码数组
        4. 只有分组编码数组不同时才认为是真正的修改
        
        Args:
            group_by (tp.GroupByLike, optional): 要检查的分组规范，默认为None
                                               用于与当前分组状态进行比较
        
        Returns:
            bool: True表示分组结构已修改，False表示分组结构未修改
                 注意：标签名称的更改不被视为修改
                 
        Examples:
            >>> grouper = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B', 'C', 'D']), 
            ...     group_by=['tech', 'tech', 'finance', 'finance']
            ... )
            >>> # 标签更改但结构相同：不算修改
            >>> is_modified1 = grouper.is_grouping_modified(['technology', 'technology', 'financial', 'financial'])
            >>> print(is_modified1)  # False
            
            >>> # 结构更改：算修改
            >>> is_modified2 = grouper.is_grouping_modified(['tech', 'finance', 'tech', 'finance'])  
            >>> print(is_modified2)  # True
        """
        # 处理无效输入：None或(False且当前为None)的情况
        if group_by is None or (group_by is False and self.group_by is None):
            return False
            
        # 将输入的group_by转换为标准格式以便比较
        group_by = group_by_to_index(self.columns, group_by)
        
        # 当两个分组都是Index对象时，进行详细比较
        if isinstance(group_by, pd.Index) and isinstance(self.group_by, pd.Index):
            # 首先检查Index对象是否完全相等（包括值、顺序、名称等）
            if not pd.Index.equals(group_by, self.group_by):
                # 如果Index不相等，获取实际的分组编码进行比较
                # 分组编码反映了真正的分组结构，不受标签名称影响
                groups1 = get_groups_and_index(self.columns, group_by)[0]
                groups2 = get_groups_and_index(self.columns, self.group_by)[0]
                # 只有当分组编码数组不同时，才认为分组结构真正被修改
                if not np.array_equal(groups1, groups2):
                    return True
            # Index相等或分组编码相等的情况：未修改
            return False
        # 其他情况（类型不同等）：认为已修改
        return True

    @cached_method
    def is_grouping_changed(self, group_by: tp.GroupByLike = None) -> bool:
        """检查列分组是否发生任何形式的变化。
        
        此方法检测分组是否发生任何变化，包括结构修改、标签更改、启用/禁用等。
        这是最宽泛的变化检测方法，用于判断是否需要重新计算相关的缓存数据。
        
        变化检测的逻辑：
        1. 排除无效输入和相同状态的情况
        2. 对于Index类型，使用pandas的equals方法进行精确比较
        3. 任何差异（包括值、顺序、名称、类型）都被视为变化
        
        Args:
            group_by (tp.GroupByLike, optional): 要检查的分组规范，默认为None
                                               用于与当前分组状态进行比较
        
        Returns:
            bool: True表示分组发生任何变化，False表示分组完全相同
                 变化包括：结构修改、标签更改、启用、禁用等所有情况
                 
        Examples:
            >>> grouper = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B', 'C', 'D']), 
            ...     group_by=['tech', 'tech', 'finance', 'finance']
            ... )
            >>> # 标签名称更改：算变化
            >>> is_changed1 = grouper.is_grouping_changed(['technology', 'technology', 'finance', 'finance'])
            >>> print(is_changed1)  # True
            
            >>> # 完全相同：不算变化
            >>> is_changed2 = grouper.is_grouping_changed(['tech', 'tech', 'finance', 'finance'])
            >>> print(is_changed2)  # False
            
            >>> # 禁用分组：算变化
            >>> is_changed3 = grouper.is_grouping_changed(False)
            >>> print(is_changed3)  # True
        """
        # 处理无效输入：None或(False且当前为None)的情况
        if group_by is None or (group_by is False and self.group_by is None):
            return False
            
        # 当两个分组都是Index对象时，使用pandas的精确比较
        if isinstance(group_by, pd.Index) and isinstance(self.group_by, pd.Index):
            # pd.Index.equals会比较所有方面：值、顺序、名称、类型等
            if pd.Index.equals(group_by, self.group_by):
                return False
        # 其他所有情况都被视为变化
        return True

    def is_group_count_changed(self, group_by: tp.GroupByLike = None) -> bool:
        """检查分组数量是否发生变化。
        
        此方法专门检测分组的数量是否改变，不关心分组的具体内容或标签。
        这对于需要根据分组数量调整数据结构或算法的场景非常有用。
        
        Args:
            group_by (tp.GroupByLike, optional): 要检查的分组规范，默认为None
                                               用于与当前分组状态进行比较
        
        Returns:
            bool: True表示分组数量发生变化，False表示分组数量未变化
                 数量比较只在两个都是Index对象时进行
                 
        Examples:
            >>> grouper = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B', 'C', 'D']), 
            ...     group_by=['g1', 'g1', 'g2', 'g3']  # 3个分组
            ... )
            >>> # 相同数量的分组：未变化
            >>> is_count_changed1 = grouper.is_group_count_changed(['x', 'x', 'y', 'z'])  # 3个分组
            >>> print(is_count_changed1)  # False
            
            >>> # 不同数量的分组：已变化  
            >>> is_count_changed2 = grouper.is_group_count_changed(['x', 'x', 'x', 'x'])  # 1个分组
            >>> print(is_count_changed2)  # True
        """
        # 处理无效输入：None或(False且当前为None)的情况
        if group_by is None or (group_by is False and self.group_by is None):
            return False
            
        # 只有当两个分组都是Index对象时才能比较数量
        if isinstance(group_by, pd.Index) and isinstance(self.group_by, pd.Index):
            # 比较两个Index的长度（即唯一分组的数量）
            return len(group_by) != len(self.group_by)
        # 其他情况（类型不同等）：默认认为数量已变化
        return True

    def check_group_by(self, group_by: tp.GroupByLike = None, allow_enable: tp.Optional[bool] = None,
                       allow_disable: tp.Optional[bool] = None, allow_modify: tp.Optional[bool] = None) -> None:
        """根据权限设置检查传入的group_by对象是否被允许。
        
        此方法是权限控制的核心实现，确保所有分组操作都符合创建时设定的权限约束。
        它会检查启用、禁用、修改等操作的权限，并在违反权限时抛出异常。
        
        权限检查逻辑：
        1. 首先解析权限参数（使用传入值或对象默认值）
        2. 然后依次检查启用、禁用、修改权限
        3. 对于每种操作，都会验证是否被明确禁止
        4. 任何权限违反都会立即抛出ValueError异常
        
        Args:
            group_by (tp.GroupByLike, optional): 要检查的分组规范，默认为None
                                               将检查此分组规范是否符合权限要求
            allow_enable (tp.Optional[bool], optional): 是否允许启用分组，默认为None
                                                      None时使用对象的allow_enable设置
            allow_disable (tp.Optional[bool], optional): 是否允许禁用分组，默认为None
                                                       None时使用对象的allow_disable设置
            allow_modify (tp.Optional[bool], optional): 是否允许修改分组，默认为None
                                                      None时使用对象的allow_modify设置
        
        Raises:
            ValueError: 当检测到权限违反时抛出，包含具体的违反类型信息
                       - "Enabling grouping is not allowed": 禁止启用分组
                       - "Disabling grouping is not allowed": 禁止禁用分组  
                       - "Modifying groups is not allowed": 禁止修改分组
                       
        Examples:
            >>> # 创建禁止启用分组的grouper
            >>> grouper = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B']), 
            ...     group_by=None,
            ...     allow_enable=False
            ... )
            >>> try:
            ...     grouper.check_group_by([0, 1])  # 尝试启用分组
            ... except ValueError as e:
            ...     print(e)  # "Enabling grouping is not allowed"
            
            >>> # 创建禁止修改分组的grouper
            >>> grouper2 = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B', 'C', 'D']), 
            ...     group_by=[0, 0, 1, 1],
            ...     allow_modify=False
            ... )
            >>> try:
            ...     grouper2.check_group_by([0, 1, 0, 1])  # 尝试修改分组结构
            ... except ValueError as e:
            ...     print(e)  # "Modifying groups is not allowed"
        """
        # 解析权限参数：使用传入值或对象默认值
        if allow_enable is None:
            allow_enable = self.allow_enable
        if allow_disable is None:
            allow_disable = self.allow_disable
        if allow_modify is None:
            allow_modify = self.allow_modify

        # 检查分组启用权限
        if self.is_grouping_enabled(group_by=group_by):
            if not allow_enable:
                raise ValueError("Enabling grouping is not allowed")
        # 检查分组禁用权限
        elif self.is_grouping_disabled(group_by=group_by):
            if not allow_disable:
                raise ValueError("Disabling grouping is not allowed")
        # 检查分组修改权限
        elif self.is_grouping_modified(group_by=group_by):
            if not allow_modify:
                raise ValueError("Modifying groups is not allowed")

    def resolve_group_by(self, group_by: tp.GroupByLike = None, **kwargs) -> GroupByT:
        """解析group_by参数，从对象变量或关键字参数中获取分组规范。
        
        此方法是分组参数标准化的入口点，它会处理参数解析、权限检查和格式转换。
        大多数需要分组参数的方法都会首先调用此方法来获得统一的分组规范。
        
        解析逻辑：
        1. 参数解析：优先使用传入的group_by，否则使用对象的group_by
        2. 特殊处理：False与None的等价转换
        3. 权限检查：确保操作符合权限设置
        4. 格式转换：统一转换为标准的Index格式
        
        Args:
            group_by (tp.GroupByLike, optional): 分组规范，默认为None
                                               None时使用对象的group_by属性
            **kwargs: 额外的关键字参数，传递给check_group_by方法进行权限检查
        
        Returns:
            GroupByT: 解析后的分组对象，类型为Union[None, bool, tp.Index]
                     已经过权限检查和格式标准化处理
                     
        Examples:
            >>> grouper = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B', 'C', 'D']), 
            ...     group_by=['tech', 'tech', 'finance', 'finance']
            ... )
            >>> # 使用对象默认的group_by
            >>> resolved1 = grouper.resolve_group_by()
            >>> print(type(resolved1))  # <class 'pandas.core.indexes.base.Index'>
            
            >>> # 使用指定的group_by
            >>> resolved2 = grouper.resolve_group_by([0, 0, 1, 1])
            >>> print(type(resolved2))  # <class 'pandas.core.indexes.base.Index'>
            
            >>> # False与None的等价处理
            >>> grouper_none = ColumnGrouper(columns=pd.Index(['A', 'B']), group_by=None)
            >>> resolved3 = grouper_none.resolve_group_by(False)
            >>> print(resolved3)  # None
        """
        # 参数解析：优先使用传入值，否则使用对象属性
        if group_by is None:
            group_by = self.group_by
        # 特殊处理：当对象group_by为None时，False等价于None
        if group_by is False and self.group_by is None:
            group_by = None
        # 权限检查：确保操作符合权限设置
        self.check_group_by(group_by=group_by, **kwargs)
        # 格式转换：统一转换为标准格式
        return group_by_to_index(self.columns, group_by)

    @cached_method
    def get_groups_and_columns(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.Tuple[tp.Array1d, tp.Index]:
        """获取分组编码数组和分组后的列索引。
        
        此方法是分组信息获取的核心接口，返回分组操作所需的两个关键数据：
        分组编码数组（指示每列属于哪个组）和新的列索引（包含分组标识）。
        这是一个缓存方法，避免重复计算相同的分组结果。
        
        Args:
            group_by (tp.GroupByLike, optional): 分组规范，默认为None
                                               使用resolve_group_by方法进行解析
            **kwargs: 额外的关键字参数，传递给resolve_group_by方法
        
        Returns:
            tp.Tuple[tp.Array1d, tp.Index]: 包含两个元素的元组：
                - 第一个元素：分组编码数组，整数数组，指示每列属于哪个组
                - 第二个元素：分组后的新列索引，包含唯一的分组标识
                
        See Also:
            get_groups_and_index: 底层实现函数，处理具体的分组逻辑
            
        Examples:
            >>> grouper = ColumnGrouper(
            ...     columns=pd.Index(['AAPL', 'GOOGL', 'MSFT', 'AMZN']), 
            ...     group_by=['tech', 'tech', 'tech', 'tech']
            ... )
            >>> groups, columns = grouper.get_groups_and_columns()
            >>> print(f"分组编码: {groups}")    # [0, 0, 0, 0]
            >>> print(f"新列索引: {columns}")   # Index(['tech'], dtype='object')
        """
        # 解析分组参数，包括权限检查和格式转换
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        # 调用底层函数进行实际的分组计算
        return get_groups_and_index(self.columns, group_by)

    def get_groups(self, **kwargs) -> tp.Array1d:
        """返回分组编码数组。
        
        此方法是get_groups_and_columns的简化版本，只返回分组编码数组部分。
        分组编码数组是一个整数数组，指示原始列索引中每个位置的元素属于哪个组。
        
        Args:
            **kwargs: 关键字参数，传递给get_groups_and_columns方法
        
        Returns:
            tp.Array1d: 一维整数数组，包含每列的分组编码
                       数组长度等于原始列数，值为0到(分组数-1)的整数
                       
        Examples:
            >>> grouper = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B', 'C', 'D']), 
            ...     group_by=['g1', 'g1', 'g2', 'g2']
            ... )
            >>> groups = grouper.get_groups()
            >>> print(groups)  # [0, 0, 1, 1]
        """
        # 获取分组信息并返回第一个元素（分组编码数组）
        return self.get_groups_and_columns(**kwargs)[0]

    def get_columns(self, **kwargs) -> tp.Index:
        """返回分组后的新列索引。
        
        此方法是get_groups_and_columns的简化版本，只返回分组后的列索引部分。
        新列索引包含所有唯一的分组标识，保持分组的原始顺序和元数据。
        
        Args:
            **kwargs: 关键字参数，传递给get_groups_and_columns方法
        
        Returns:
            tp.Index: 分组后的新列索引，包含唯一的分组标识
                     长度等于分组数量，保持原始分组值的数据类型
                     
        Examples:
            >>> grouper = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B', 'C', 'D']), 
            ...     group_by=['tech', 'tech', 'finance', 'healthcare']
            ... )
            >>> new_columns = grouper.get_columns()
            >>> print(new_columns)  # Index(['tech', 'finance', 'healthcare'], dtype='object')
        """
        # 获取分组信息并返回第二个元素（新列索引）
        return self.get_groups_and_columns(**kwargs)[1]

    @cached_method
    def is_sorted(self, group_by: tp.GroupByLike = None, **kwargs) -> bool:
        """检查分组是否是连贯且已排序的。
        
        此方法验证分组编码是否满足连贯且已排序的要求，这是使用高性能的
        get_group_lens_nb函数的前提条件。连贯且已排序意味着相同组的元素
        必须连续出现，且组编号按顺序递增。
        
        这是一个缓存方法，避免重复检查相同的分组配置。
        
        Args:
            group_by (tp.GroupByLike, optional): 分组规范，默认为None
                                               使用resolve_group_by方法进行解析
            **kwargs: 额外的关键字参数，传递给resolve_group_by方法
        
        Returns:
            bool: True表示分组是连贯且已排序的，False表示不满足要求
                 满足要求的示例：[0, 0, 0, 1, 1, 2, 2, 2]
                 不满足要求的示例：[0, 1, 0, 2, 1]（不连贯）
                 
        Examples:
            >>> # 满足要求的分组
            >>> grouper1 = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B', 'C', 'D', 'E']), 
            ...     group_by=[0, 0, 1, 1, 1]
            ... )
            >>> print(grouper1.is_sorted())  # True
            
            >>> # 不满足要求的分组
            >>> grouper2 = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B', 'C', 'D']), 
            ...     group_by=[0, 1, 0, 1]  # 相同组不连续
            ... )
            >>> print(grouper2.is_sorted())  # False
        """
        # 解析分组参数，包括权限检查和格式转换
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        # 获取分组编码数组
        groups = self.get_groups(group_by=group_by)
        # 使用工具函数检查数组是否已排序
        return is_sorted(groups)

    @cached_method
    def get_group_lens(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.Array1d:
        """获取每个分组的元素数量。
        
        此方法计算每个分组包含的元素数量，返回一个数组，其中第i个元素
        表示第i个分组的大小。这是分组统计和聚合操作的基础信息。
        
        此方法要求分组必须是连贯且已排序的，否则会抛出异常。
        对于无分组的情况，返回一个全为1的数组（每列自成一组）。
        
        Args:
            group_by (tp.GroupByLike, optional): 分组规范，默认为None
                                               使用resolve_group_by方法进行解析
            **kwargs: 额外的关键字参数，传递给resolve_group_by方法
        
        Returns:
            tp.Array1d: 一维整数数组，包含每个分组的元素数量
                       数组长度等于分组数量，每个值为正整数
                       
        Raises:
            ValueError: 当分组不是连贯且已排序时抛出异常
                       错误信息："group_by must lead to groups that are coherent and sorted"
                       
        See Also:
            get_group_lens_nb: 底层的Numba编译函数，提供高性能计算
            
        Examples:
            >>> grouper = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B', 'C', 'D', 'E']), 
            ...     group_by=[0, 0, 0, 1, 1]
            ... )
            >>> lens = grouper.get_group_lens()
            >>> print(lens)  # [3, 2] (第0组有3个元素，第1组有2个元素)
            
            >>> # 无分组情况
            >>> grouper_none = ColumnGrouper(columns=pd.Index(['A', 'B', 'C']), group_by=None)
            >>> lens_none = grouper_none.get_group_lens()
            >>> print(lens_none)  # [1, 1, 1] (每列自成一组)
        """
        # 首先检查分组是否满足连贯且已排序的要求
        if not self.is_sorted(group_by=group_by):
            raise ValueError("group_by must lead to groups that are coherent and sorted "
                             "(such as [0, 0, 1, 2, 2, ...])")
        # 解析分组参数，包括权限检查和格式转换
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        # 处理无分组的情况：每列自成一组，长度都为1
        if group_by is None or group_by is False:  # no grouping
            return np.full(len(self.columns), 1)
        # 获取分组编码数组
        groups = self.get_groups(group_by=group_by)
        # 使用高性能的Numba函数计算分组长度
        return get_group_lens_nb(groups)

    @cached_method
    def get_group_count(self, **kwargs) -> int:
        """获取分组的数量。
        
        此方法返回当前分组配置下的分组总数。这是一个派生属性，
        通过计算分组长度数组的长度来获得。
        
        Args:
            **kwargs: 关键字参数，传递给get_group_lens方法
        
        Returns:
            int: 分组的总数量，始终为正整数
                对于无分组情况，返回列的数量（每列自成一组）
                
        Examples:
            >>> grouper = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B', 'C', 'D']), 
            ...     group_by=['g1', 'g1', 'g2', 'g3']
            ... )
            >>> count = grouper.get_group_count()
            >>> print(count)  # 3 (g1, g2, g3三个分组)
            
            >>> # 无分组情况
            >>> grouper_none = ColumnGrouper(columns=pd.Index(['A', 'B']), group_by=None)
            >>> count_none = grouper_none.get_group_count()
            >>> print(count_none)  # 2 (每列自成一组)
        """
        # 通过分组长度数组的长度获取分组数量
        return len(self.get_group_lens(**kwargs))

    @cached_method
    def get_group_start_idxs(self, **kwargs) -> tp.Array1d:
        """获取每个分组的起始索引数组。
        
        此方法计算每个分组在原始列索引中的起始位置，返回一个数组，
        其中第i个元素表示第i个分组的第一个列的索引位置。
        
        起始索引的计算基于分组长度：第i个分组的起始索引等于
        前i个分组长度的累积和减去第i个分组的长度。
        
        Args:
            **kwargs: 关键字参数，传递给get_group_lens方法
        
        Returns:
            tp.Array1d: 一维整数数组，包含每个分组的起始索引
                       数组长度等于分组数量，值为0到(列数-1)的整数
                       第一个分组的起始索引始终为0
                       
        Examples:
            >>> grouper = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B', 'C', 'D', 'E']), 
            ...     group_by=[0, 0, 0, 1, 1]  # 分组长度为[3, 2]
            ... )
            >>> start_idxs = grouper.get_group_start_idxs()
            >>> print(start_idxs)  # [0, 3] (第0组从索引0开始，第1组从索引3开始)
            
            >>> # 验证：累积和为[3, 5]，减去长度[3, 2]得到[0, 3]
        """
        # 获取每个分组的长度数组
        group_lens = self.get_group_lens(**kwargs)
        # 计算累积和，然后减去对应的长度，得到起始索引
        # np.cumsum(group_lens) - group_lens 相当于每个位置的前缀和
        return np.cumsum(group_lens) - group_lens

    @cached_method
    def get_group_end_idxs(self, **kwargs) -> tp.Array1d:
        """获取每个分组的结束索引数组。
        
        此方法计算每个分组在原始列索引中的结束位置（不包含），返回一个数组，
        其中第i个元素表示第i个分组的最后一个列的索引位置+1。
        
        结束索引的计算直接基于分组长度的累积和，这样可以方便地
        用于切片操作：columns[start_idx:end_idx]。
        
        Args:
            **kwargs: 关键字参数，传递给get_group_lens方法
        
        Returns:
            tp.Array1d: 一维整数数组，包含每个分组的结束索引（不包含）
                       数组长度等于分组数量，值为1到列数的整数
                       最后一个分组的结束索引等于列的总数
                       
        Examples:
            >>> grouper = ColumnGrouper(
            ...     columns=pd.Index(['A', 'B', 'C', 'D', 'E']), 
            ...     group_by=[0, 0, 0, 1, 1]  # 分组长度为[3, 2]
            ... )
            >>> end_idxs = grouper.get_group_end_idxs()
            >>> print(end_idxs)  # [3, 5] (第0组在索引3结束，第1组在索引5结束)
            
            >>> # 验证切片操作
            >>> start_idxs = grouper.get_group_start_idxs()  # [0, 3]
            >>> group_0_columns = grouper.columns[start_idxs[0]:end_idxs[0]]  # ['A', 'B', 'C']
            >>> group_1_columns = grouper.columns[start_idxs[1]:end_idxs[1]]  # ['D', 'E']
        """
        # 获取每个分组的长度数组
        group_lens = self.get_group_lens(**kwargs)
        # 计算累积和，得到每个分组的结束索引（不包含）
        return np.cumsum(group_lens)
