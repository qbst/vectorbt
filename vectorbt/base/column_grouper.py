# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT BASE MODULE: COLUMN GROUPER
================================================================================

文件作用概述：
为Index的分组提供了一套完整的功能。
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

    def __init__(self, 
                columns: tp.Index,  # 要进行分组的原始列索引
                group_by: tp.GroupByLike = None,  # 分组映射器
                allow_enable: bool = True,  # 是否允许启用分组
                allow_disable: bool = True,  # 是否允许禁用分组
                allow_modify: bool = True) -> None: # 是否允许修改分组
        """
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
        """
        # 将参数 columns, group_by, allow_enable, allow_disable, allow_modify 和 settings['configured']['config'] 合并到 self._config
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
        if group_by is False:
            return False
        if group_by is None:
            group_by = self.group_by
        return group_by is not None

    def is_grouping_enabled(self, group_by: tp.GroupByLike = None) -> bool:
        """
        检查是否 self.group_by 为 None 且 group_by 不为 None
        """
        return self.group_by is None and self.is_grouped(group_by=group_by)

    def is_grouping_disabled(self, group_by: tp.GroupByLike = None) -> bool:
        """
        检查是否 self.group_by 不为 None 且 group_by 为 None
        """
        return self.group_by is not None and not self.is_grouped(group_by=group_by)

    @cached_method
    def is_grouping_modified(self, group_by: tp.GroupByLike = None) -> bool:
        """
        检查 self.group_by 是否和 group_by_to_index(self.columns, group_by) 不一致
        """
        if group_by is None or (group_by is False and self.group_by is None):
            return False
            
        group_by = group_by_to_index(self.columns, group_by)
        
        if isinstance(group_by, pd.Index) and isinstance(self.group_by, pd.Index):
            # 首先检查Index对象是否完全相等（包括值、顺序、名称等）
            if not pd.Index.equals(group_by, self.group_by):
                # 如果Index不相等，获取实际的分组编码进行比较，分组编码反映了分组结构，不受标签名称影响
                groups1 = get_groups_and_index(self.columns, group_by)[0]
                groups2 = get_groups_and_index(self.columns, self.group_by)[0]
                if not np.array_equal(groups1, groups2):
                    return True
            return False
        return True

    @cached_method
    def is_grouping_changed(self, group_by: tp.GroupByLike = None) -> bool:
        """
        检查 self.group_by 是否和 group_by 完全不一致
        """
        if group_by is None or (group_by is False and self.group_by is None):
            return False
            
        if isinstance(group_by, pd.Index) and isinstance(self.group_by, pd.Index):
            # pd.Index.equals会比较所有方面：值、顺序、名称、类型等
            if pd.Index.equals(group_by, self.group_by):
                return False
        return True

    def is_group_count_changed(self, group_by: tp.GroupByLike = None) -> bool:
        """
        检查 self.group_by 和 group_by 的长度是否不一致
        """
        if group_by is None or (group_by is False and self.group_by is None):
            return False

        if isinstance(group_by, pd.Index) and isinstance(self.group_by, pd.Index):
            return len(group_by) != len(self.group_by)
        return True

    def check_group_by(self, 
                       group_by: tp.GroupByLike = None, 
                       allow_enable: tp.Optional[bool] = None,
                       allow_disable: tp.Optional[bool] = None, 
                       allow_modify: tp.Optional[bool] = None) -> None:
        """
        如果 self.group_by 为 None 且 group_by 不为 None 并且 allow_enable 为 False，则抛出异常
        如果 self.group_by 不为 None 且 group_by 为 None 并且 allow_disable 为 False，则抛出异常
        如果 self.group_by 和 group_by 不一致 并且 allow_modify 为 False，则抛出异常
        """
        if allow_enable is None:
            allow_enable = self.allow_enable
        if allow_disable is None:
            allow_disable = self.allow_disable
        if allow_modify is None:
            allow_modify = self.allow_modify

        if self.is_grouping_enabled(group_by=group_by):
            if not allow_enable:
                raise ValueError("Enabling grouping is not allowed")
        elif self.is_grouping_disabled(group_by=group_by):
            if not allow_disable:
                raise ValueError("Disabling grouping is not allowed")
        elif self.is_grouping_modified(group_by=group_by):
            if not allow_modify:
                raise ValueError("Modifying groups is not allowed")

    def resolve_group_by(self, group_by: tp.GroupByLike = None, **kwargs) -> GroupByT:
        if group_by is None:
            group_by = self.group_by
        if group_by is False and self.group_by is None:
            group_by = None
        self.check_group_by(group_by=group_by, **kwargs)
        return group_by_to_index(self.columns, group_by)

    @cached_method
    def get_groups_and_columns(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.Tuple[tp.Array1d, tp.Index]:
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        return get_groups_and_index(self.columns, group_by)

    def get_groups(self, **kwargs) -> tp.Array1d:
        return self.get_groups_and_columns(**kwargs)[0]

    def get_columns(self, **kwargs) -> tp.Index:
        return self.get_groups_and_columns(**kwargs)[1]

    @cached_method
    def is_sorted(self, group_by: tp.GroupByLike = None, **kwargs) -> bool:
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        groups = self.get_groups(group_by=group_by)
        return is_sorted(groups)

    @cached_method
    def get_group_lens(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.Array1d:
        group_by = self.resolve_group_by(group_by=group_by, **kwargs)
        if not self.is_sorted(group_by=group_by):
            raise ValueError("group_by must lead to groups that are coherent and sorted "
                             "(such as [0, 0, 1, 2, 2, ...])")
        if group_by is None or group_by is False:  # no grouping
            return np.full(len(self.columns), 1)
        groups = self.get_groups(group_by=group_by)
        return get_group_lens_nb(groups)

    @cached_method
    def get_group_count(self, **kwargs) -> int:
        return len(self.get_group_lens(**kwargs))

    @cached_method
    def get_group_start_idxs(self, **kwargs) -> tp.Array1d:
        group_lens = self.get_group_lens(**kwargs)
        return np.cumsum(group_lens) - group_lens

    @cached_method
    def get_group_end_idxs(self, **kwargs) -> tp.Array1d:
        group_lens = self.get_group_lens(**kwargs)
        return np.cumsum(group_lens)
