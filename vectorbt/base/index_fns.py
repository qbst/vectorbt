# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Functions for working with index/columns.

Index functions perform operations on index objects, such as stacking, combining,
and cleansing MultiIndex levels. "Index" in pandas context is referred to both index and columns."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from numba import njit

from vectorbt import _typing as tp
from vectorbt.utils import checks


def to_any_index(index_like: tp.IndexLike) -> tp.Index:
    """
    将任何类似索引的对象转换为索引。
    
    参数:
        index_like: 类索引对象，可以是列表、数组、元组等
        
    返回:
        pd.Index: 转换后的pandas索引对象
        
    说明:
        如果输入已经是索引对象，则保持不变；
        否则，将其转换为pandas索引对象。
    """
    if not isinstance(index_like, pd.Index):
        return pd.Index(index_like)
    return index_like


def get_index(arg: tp.SeriesFrame, axis: int) -> tp.Index:
    """
    根据指定轴获取数据框或序列的索引。
    
    参数:
        arg: pandas的Series或DataFrame对象
        axis: 轴编号，0表示行索引，1表示列索引
        
    返回:
        pd.Index: 指定轴上的索引对象
        
    异常:
        如果arg不是pandas.Series或pandas.DataFrame，或axis不是0或1，将引发异常
    """
    checks.assert_instance_of(arg, (pd.Series, pd.DataFrame))
    checks.assert_in(axis, (0, 1))

    if axis == 0:
        return arg.index
    else:
        if checks.is_series(arg):
            if arg.name is not None:
                return pd.Index([arg.name])
            return pd.Index([0])  # 与pandas处理方式保持一致
        else:
            return arg.columns

def index_from_values(values: tp.ArrayLikeSequence, name: tp.Optional[str] = None) -> tp.Index:
    """
    通过解析一个可迭代的 `values` 序列来创建一个新的、具有可选名称 `name` 的 `pd.Index` 对象。
    这个函数的目标是根据输入序列中元素的类型和内容，为新索引的每个位置生成一个有意义的标签。

    参数:
        values: tp.ArrayLikeSequence
            一个类似数组的序列（例如列表、元组、NumPy数组或Pandas Series），
            其每个元素将对应新索引中的一个条目。
        name: tp.Optional[str], 默认值: None
            可选参数，用于指定新创建的 `pd.Index` 对象的名称。

    返回:
        pd.Index
            一个新创建的 Pandas 索引对象。

    示例:
        >>> values1 = [10, "hello", None, np.array([1.0, 1.0000001]), np.array([2, 2]), MyObject()]
        >>> index_from_values(values1, name="示例索引")
        Index([10, 'hello', None, 1.0, 2, 'MyObject_5'], dtype='object', name='示例索引')
        
        >>> values2 = [np.array([1, 2, 3]), np.array([4.0, 5.1])]
        >>> index_from_values(values2)
        Index(['array_0', 'array_1'], dtype='object')
    """

    scalar_types = (int, float, complex, str, bool, datetime, timedelta, np.generic)
    value_names = []
    for i in range(len(values)):
        v = values[i]
        if v is None or isinstance(v, scalar_types):
            value_names.append(v)
        # 如果当前元素 `v` 是一个 NumPy 数组
        elif isinstance(v, np.ndarray):
            if np.issubdtype(v.dtype, np.floating):
                if np.isclose(v, v.item(0), equal_nan=True).all():
                    value_names.append(v.item(0))
                else:
                    value_names.append('array_%d' % i)
            else:
                if np.equal(v, v.item(0)).all():
                    value_names.append(v.item(0))
                else:
                    value_names.append('array_%d' % i)
        else:
            value_names.append('%s_%d' % (str(type(v).__name__), i))
    return pd.Index(value_names, name=name)


def repeat_index(index: tp.IndexLike, n: int, ignore_default: tp.Optional[bool] = None) -> tp.Index:
    """
    将输入索引 `index` 中的每个元素重复 `n` 次，生成一个新的 Pandas 索引对象。

    此函数可以处理各种类型的索引，包括简单索引和多级索引 (MultiIndex)。
    它还提供了一个选项 `ignore_default`，用于在处理默认索引（如 RangeIndex(start=0, stop=k, step=1) 且无名称）时，
    可以选择忽略其原始值，并简单地创建一个新的、更长的默认索引。

    参数:
        index: tp.IndexLike
            一个类似索引的对象，可以是 Pandas `Index`、`Series`、`DataFrame` 的索引/列，
            或者任何可以被 `pd.Index()` 接受的类型（如列表、NumPy 数组）。
        n: int
            每个索引元素需要重复的次数。必须为非负整数。
            如果 n 为 0，则返回空索引。
        ignore_default: tp.Optional[bool], 默认值: None
            一个布尔值，指示是否应特殊处理“默认”索引（即，无名称的 `pd.RangeIndex`，通常从0开始，步长为1）。
            - 如果为 `True`，并且输入 `index` 是一个默认索引，则函数将返回一个新的 `pd.RangeIndex`，
              其长度为 `len(index) * n`，而不是重复原始索引的值。
            - 如果为 `False`，则即使是默认索引，其值也会被重复。
            - 如果为 `None` (默认情况)，则此行为由 `vectorbt._settings.broadcasting['ignore_default']` 全局配置决定。
    返回:
        pd.Index
            一个新的 Pandas 索引对象，其中原始索引的每个元素都重复了 `n` 次。

    示例:
        >>> # 示例 1: 重复一个简单的索引
        >>> idx1 = pd.Index(['a', 'b'], name='my_index')
        >>> repeat_index(idx1, 3)
        Index(['a', 'a', 'a', 'b', 'b', 'b'], dtype='object', name='my_index')

        >>> # 示例 2: 重复一个 MultiIndex
        >>> m_idx = pd.MultiIndex.from_tuples([('x', 1), ('y', 2)], names=['char', 'num'])
        >>> repeat_index(m_idx, 2)
        MultiIndex([('x', 1),
                    ('x', 1),
                    ('y', 2),
                    ('y', 2)],
                   names=['char', 'num'])
    """
    from vectorbt._settings import settings
    broadcasting_cfg = settings['broadcasting']

    if ignore_default is None:
        ignore_default = broadcasting_cfg['ignore_default']

    index = to_any_index(index)
    if checks.is_default_index(index) and ignore_default:  # 忽略无名称的简单范围
        return pd.RangeIndex(start=0, stop=len(index) * n, step=1)
    return index.repeat(n)


def tile_index(index: tp.IndexLike, n: int, ignore_default: tp.Optional[bool] = None) -> tp.Index:
    """
    将整个 `index` 对象平铺（完整复制）`n` 次。

    与 `repeat_index` 不同（`repeat_index` 是将索引中的每个元素重复 `n` 次），
    `tile_index` 是将整个索引序列作为一个单元，然后将这个单元重复 `n` 次。

    参数:
        index: tp.IndexLike
            一个类似索引的对象（例如列表、元组、NumPy 数组、Pandas Series 或 Pandas Index）。
            该函数会首先将其转换为 Pandas Index 对象。
        n: int
            整个索引序列需要被平铺（重复）的次数。
        ignore_default: tp.Optional[bool], 默认值: None
            一个布尔值，指示是否忽略 "默认索引"。
            如果为 `True`，并且输入的 `index` 是一个 "默认索引" (例如，一个没有名称的 `pd.RangeIndex`，
            通常是从0开始的连续整数)，则函数会返回一个新的、长度为 `len(index) * n` 的默认索引
            （`pd.RangeIndex(start=0, stop=len(index) * n, step=1)`），
            而不是平铺原始索引的值。
            如果为 `None`，则会从 `vectorbt._settings` 中的广播配置读取此设置。

    返回:
        pd.Index
            一个新创建的 Pandas 索引对象，它是原始索引平铺 `n` 次的结果。

    示例:
        >>> import pandas as pd
        >>> import numpy as np

        >>> simple_index = pd.Index(['a', 'b'], name="L1")
        >>> tile_index(simple_index, 3)
        Index(['a', 'b', 'a', 'b', 'a', 'b'], dtype='object', name='L1')

        >>> multi_index = pd.MultiIndex.from_tuples([('x', 1), ('y', 2)], names=['L1', 'L2'])
        >>> tile_index(multi_index, 2)
        MultiIndex([('x', 1),
                    ('y', 2),
                    ('x', 1),
                    ('y', 2)],
                   names=['L1', 'L2'])
    """
    from vectorbt._settings import settings
    broadcasting_cfg = settings['broadcasting']

    if ignore_default is None:
        ignore_default = broadcasting_cfg['ignore_default']

    index = to_any_index(index)
    if checks.is_default_index(index) and ignore_default:  # 忽略无名称的简单范围
        # 如果是多级索引，则使用 numpy.tile 将其元组表示形式平铺 n 次
        # 然后使用 from_tuples 从平铺后的元组序列创建新的 MultiIndex，并保留原始的级别名称
        return pd.RangeIndex(start=0, stop=len(index) * n, step=1)
    if isinstance(index, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(np.tile(index, n), names=index.names)
    return pd.Index(np.tile(index, n), name=index.name)


def stack_indexes(indexes: tp.Sequence[tp.IndexLike], drop_duplicates: tp.Optional[bool] = None,
                  keep: tp.Optional[str] = None, drop_redundant: tp.Optional[bool] = None) -> tp.Index:
    """
    将序列 `indexes` 中的每个索引垂直堆叠（合并级别）以创建一个新的 `pd.MultiIndex`。

    如果输入序列中的某个索引本身就是 `pd.MultiIndex`，则其所有级别将被展开放入新的 `pd.MultiIndex` 中。
    此函数可以自动处理重复级别和冗余级别的删除。

    参数:
        indexes: tp.Sequence[tp.IndexLike]
            一个包含类索引对象（例如 `pd.Index`, `pd.Series`, 列表, NumPy 数组等）的序列。
            这些索引将被堆叠起来。
        drop_duplicates: tp.Optional[bool], 默认值: None
            是否删除堆叠后 `MultiIndex` 中的重复级别。
            重复级别是指那些具有相同名称和相同所有值的级别。
            如果为 `None`，则从 `vectorbt._settings.broadcasting['drop_duplicates']` 获取默认值。
        keep: tp.Optional[str], 默认值: None
            当 `drop_duplicates` 为 `True` 时，指定保留哪个重复级别。
            可以是 'first'（保留第一个出现的）或 'last'（保留最后一个出现的）。
            如果为 `None`，则从 `vectorbt._settings.broadcasting['keep']` 获取默认值。
        drop_redundant: tp.Optional[bool], 默认值: None
            是否删除堆叠后 `MultiIndex` 中的冗余级别。
            冗余级别通常是指那些对区分数据没有贡献的级别，例如：
            1. 级别中只有一个唯一值且该级别没有名称。
            2. 级别的值序列与默认的 `pd.RangeIndex(start=0, stop=len, step=1)` 相同且无名称。
            如果为 `None`，则从 `vectorbt._settings.broadcasting['drop_redundant']` 获取默认值。

    返回:
        pd.Index
            一个新创建的 `pd.MultiIndex` 对象。如果处理后只剩下一个级别，则返回普通的 `pd.Index`。

    示例:
        >>> import pandas as pd
        >>> from vectorbt.base.index_fns import stack_indexes, to_any_index

        >>> # 示例 1: 堆叠两个简单的索引
        >>> idx1 = pd.Index(['a', 'b'], name='Level_A')
        >>> idx2 = pd.Index([1, 2], name='Level_B')
        >>> stack_indexes([idx1, idx2])
        MultiIndex([('a', 1),
                    ('b', 2)],
                   names=['Level_A', 'Level_B'])

        >>> # 示例 2: 堆叠包括一个 MultiIndex
        >>> idx_multi = pd.MultiIndex.from_tuples([('x', 10), ('y', 20)], names=['M_A', 'M_B'])
        >>> stack_indexes([idx1, idx_multi])
        MultiIndex([('a', 'x', 10),
                    ('b', 'y', 20)],
                   names=['Level_A', 'M_A', 'M_B'])

        >>> # 示例 3: 处理重复级别 (drop_duplicates=True)
        >>> idx_dup1 = pd.Index(['k1', 'k2'], name='Key')
        >>> idx_dup2 = pd.Index(['k1', 'k2'], name='Key') # 与 idx_dup1 名称和值都相同
        >>> idx_val = pd.Index([100, 200], name='Value')
        >>> stack_indexes([idx_dup1, idx_val, idx_dup2], drop_duplicates=True, keep='first')
        MultiIndex([('k1', 100),
                    ('k2', 200)],
                   names=['Key', 'Value'])
        >>> stack_indexes([idx_dup1, idx_val, idx_dup2], drop_duplicates=True, keep='last') # 保留最后一个 Key
        MultiIndex([(100, 'k1'), # 顺序会根据 keep='last' 调整级别处理顺序，保留后出现的Key
                    (200, 'k2')],
                   names=['Value', 'Key']) # 注意：drop_duplicate_levels内部实现会影响保留的级别顺序
    """
    from vectorbt._settings import settings
    broadcasting_cfg = settings['broadcasting']

    if drop_duplicates is None:
        drop_duplicates = broadcasting_cfg['drop_duplicates']
    if keep is None:
        keep = broadcasting_cfg['keep']
    if drop_redundant is None:
        drop_redundant = broadcasting_cfg['drop_redundant']

    levels = []
    for i in range(len(indexes)):
        index = indexes[i]
        if not isinstance(index, pd.MultiIndex):
            levels.append(to_any_index(index))
        else:
            for j in range(index.nlevels):
                levels.append(index.get_level_values(j))

    new_index = pd.MultiIndex.from_arrays(levels)
    if drop_duplicates:
        new_index = drop_duplicate_levels(new_index, keep=keep)
    if drop_redundant:
        new_index = drop_redundant_levels(new_index)
    return new_index


def combine_indexes(indexes: tp.Sequence[tp.IndexLike],
                    ignore_default: tp.Optional[bool] = None, **kwargs) -> tp.Index:
    """
    通过计算输入索引序列 `indexes` 中每个索引的笛卡尔积来组合它们，从而创建一个新的（通常是多级）索引。

    此函数逐对处理输入索引，通过重复和平铺操作生成中间结果，然后将它们堆叠起来。
    最终结果的长度是所有输入索引长度的乘积。
    新索引的级别数量通常等于输入索引的总级别数量（如果每个输入索引都是单级的话），
    或者更复杂，取决于输入索引的结构和 `stack_indexes` 的行为。

    参数:
        indexes: tp.Sequence[tp.IndexLike]
            一个包含类索引对象的序列（例如列表、元组）。序列中的每个元素都将被用于构建笛卡尔积。
            序列至少需要包含一个索引。
        ignore_default: tp.Optional[bool], 默认值: None
            一个布尔值，指示在 `repeat_index` 和 `tile_index` 操作中是否应忽略默认索引
            （例如，没有名称的 `pd.RangeIndex`）。
            如果为 `None`，则将使用 `vectorbt._settings.broadcasting_cfg['ignore_default']` 的设置。
            如果为 `True`，默认索引在重复/平铺时会被替换为新的 `RangeIndex`，而不是复制其值。
        **kwargs:
            其他关键字参数，这些参数将透传给内部调用的 `stack_indexes` 函数。
            这允许用户控制堆叠过程中的行为，例如 `drop_duplicates`, `keep`, `drop_redundant`。

    返回:
        pd.Index
            一个新创建的 Pandas 索引对象，代表了输入索引的笛卡尔积。
            如果输入索引都是单级的，结果通常是一个 `pd.MultiIndex`。

    示例:
        >>> import pandas as pd
        >>> from vectorbt.base.index_fns import to_any_index # 假设 to_any_index 等函数已定义

        >>> idx1 = pd.Index(['a', 'b'], name='level_1')
        >>> idx2 = pd.Index([1, 2], name='level_2')
        >>> idx3 = pd.Index([True], name='level_3')
        >>> combine_indexes([idx1, idx2, idx3])
        MultiIndex([('a', 1, True),
                    ('a', 2, True),
                    ('b', 1, True),
                    ('b', 2, True)],
                   names=['level_1', 'level_2', 'level_3'])

        >>> idx_default1 = pd.RangeIndex(stop=2) # 默认索引
        >>> idx_default2 = pd.RangeIndex(stop=2)
        >>> # 默认情况下，默认索引的值会被使用
        >>> combine_indexes([idx_default1, idx_default2])
        MultiIndex([(0, 0),
                    (0, 1),
                    (1, 0),
                    (1, 1)],
                   )
    """
    new_index = to_any_index(indexes[0])
    for i in range(1, len(indexes)):
        index1, index2 = new_index, to_any_index(indexes[i])
        new_index1 = repeat_index(index1, len(index2), ignore_default=ignore_default)
        new_index2 = tile_index(index2, len(index1), ignore_default=ignore_default)
        new_index = stack_indexes([new_index1, new_index2], **kwargs)
    return new_index


def drop_levels(index: tp.Index, levels: tp.MaybeLevelSequence, strict: bool = True) -> tp.Index:
    """
    从 Pandas 索引对象中删除一个或多个指定的级别。

    如果 `index` 不是 `pd.MultiIndex` 类型，则直接返回原始索引。
    该函数提供了严格模式和非严格模式：
    - 严格模式 (`strict=True`): 直接调用 Pandas 内置的 `index.droplevel()` 方法。如果指定的级别不存在，
      Pandas 通常会抛出错误。
    - 非严格模式 (`strict=False`): 在删除级别前会进行检查。
        - 它会尝试识别有效的级别（通过名称或整数位置）。
        - 如果指定的级别（名称或位置）在索引中不存在或无效，它会忽略这些无效的级别，而不是抛出错误。
        - 只有在删除指定级别后，索引中至少还剩余一个级别时，才会执行删除操作。如果删除操作会导致所有级别都被移除，
          则不进行删除，返回原始的多级索引。

    参数:
        index: tp.Index
            输入的 Pandas 索引对象，可以是单级索引 (`pd.Index`) 或多级索引 (`pd.MultiIndex`)。
        levels: tp.MaybeLevelSequence
            指定要删除的级别。可以是单个级别（整数位置或级别名称字符串）或级别序列（整数位置列表/元组或级别名称列表/元组）。
            例如：`0`, `'level_name'`, `[0, 'level_name_2']`, `(1, 2)`.
        strict: bool, 默认值: True
            一个布尔值，指示是否采用严格模式。
            - `True`: 使用 Pandas 的标准 `droplevel` 行为，对不存在的级别会抛出错误。
            - `False`: 采用更宽松的行为，会忽略不存在的级别，并且只有在删除后仍有级别剩余时才执行删除。

    返回:
        pd.Index
            删除了指定级别后的 Pandas 索引对象。如果输入不是多级索引，或者在非严格模式下删除会导致没有级别剩余，
            则返回原始索引。

    示例:
        >>> import pandas as pd
        >>> # 创建一个多级索引
        >>> arrays = [
        ...     ['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
        ...     ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']
        ... ]
        >>> tuples = list(zip(*arrays))
        >>> multi_index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
        >>> print("原始多级索引:")
        >>> print(multi_index)
        原始多级索引:
        MultiIndex([('bar', 'one'),
                    ('bar', 'two'),
                    ('baz', 'one'),
                    ('baz', 'two'),
                    ('foo', 'one'),
                    ('foo', 'two'),
                    ('qux', 'one'),
                    ('qux', 'two')],
                   names=['first', 'second'])

        >>> # 示例 1: 严格模式下删除一个存在的级别 (按名称)
        >>> dropped_strict_name = drop_levels(multi_index, 'first', strict=True)
        >>> print("\\n严格模式按名称删除 'first':")
        >>> print(dropped_strict_name)
        严格模式按名称删除 'first':
        Index(['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'], dtype='object', name='second')
    """
    if not isinstance(index, pd.MultiIndex):
        return index
    if strict:
        return index.droplevel(levels)

    levels_to_drop = set()
    if isinstance(levels, (int, str)):
        levels = (levels,)
    for level in levels:
        # 检查当前 `level` 是否为多级索引 `index` 的名称之一
        if level in index.names:
            levels_to_drop.add(level)
        # 检查当前 `level` 是否为整数（表示级别位置）
        # 并且该整数位置在有效范围内（0 到 nlevels-1，或 -1 表示最后一个级别）
        elif isinstance(level, int) and 0 <= level < index.nlevels or level == -1:
            levels_to_drop.add(level)
    if len(levels_to_drop) < index.nlevels:
        # 仅当还有一些级别保留时才删除
        return index.droplevel(list(levels_to_drop))
    return index


def rename_levels(index: tp.Index, name_dict: tp.Dict[str, tp.Any], strict: bool = True) -> tp.Index:
    """
    使用 `name_dict` 中定义的映射关系重命名 `index` 对象中的一个或多个级别（level）。

    对于多级索引 (MultiIndex)，此函数会尝试重命名与 `name_dict` 中键匹配的级别名称。
    对于单级索引，此函数会尝试重命名索引本身的名称（如果它与 `name_dict` 中的某个键匹配）。

    参数:
        index: tp.Index
            需要重命名级别的 Pandas 索引对象。可以是单级索引或多级索引。
        name_dict: tp.Dict[str, tp.Any]
            一个字典，用于指定如何重命名级别。
            字典的键 (str) 是当前级别（或索引）的名称，字典的值 (tp.Any) 是对应的新名称。
        strict: bool, 默认值: True
            一个布尔值，指示当尝试重命名一个不存在的级别时是否应引发 `KeyError`。
            如果为 `True`，且 `name_dict` 中的某个键在 `index` 的级别名称中找不到（或对于单级索引，不匹配索引名称），则会引发 `KeyError`。
            如果为 `False`，则会忽略不存在的级别，不会引发错误。

    返回:
        tp.Index
            返回重命名级别后的 Pandas 索引对象。原始索引对象可能会被修改（特别是单级索引的名称），或者返回一个新的索引对象（特别是对于MultiIndex的rename操作）。

    示例:
        >>> import pandas as pd
        >>> # 示例 1: 重命名 MultiIndex 的级别
        >>> m_index = pd.MultiIndex.from_tuples([(1, 'a'), (1, 'b'), (2, 'a')], names=['num', 'char'])
        >>> print("原始 MultiIndex:")
        >>> print(m_index)
        原始 MultiIndex:
        MultiIndex([(1, 'a'),
                    (1, 'b'),
                    (2, 'a')],
                   names=['num', 'char'])
        >>> renamed_m_index = rename_levels(m_index, {'num': 'number_level', 'char': 'character_level'})
        >>> print("\\n重命名后的 MultiIndex:")
        >>> print(renamed_m_index)
        重命名后的 MultiIndex:
        MultiIndex([(1, 'a'),
                    (1, 'b'),
                    (2, 'a')],
                   names=['number_level', 'character_level'])
    """
    for k, v in name_dict.items():
        if isinstance(index, pd.MultiIndex):
            if k in index.names:
                index = index.rename(v, level=k)
            elif strict:
                raise KeyError(f"Level '{k}' not found")
        else:
            if index.name == k:
                index.name = v
            elif strict:
                raise KeyError(f"Level '{k}' not found")
    return index


def select_levels(index: tp.Index, level_names: tp.MaybeLevelSequence) -> tp.Index:
    """
    通过从 `index` (必须是 MultiIndex) 中选择一个或多个由 `level_names` 指定的级别来构建新索引。

    参数:
        index: pd.Index
            一个 Pandas 索引对象。此函数期望它是一个 `pd.MultiIndex`。
        level_names: tp.MaybeLevelSequence
            要选择的级别。可以是：
            - 单个级别的名称 (str)。
            - 单个级别的位置 (int)。
            - 级别名称或位置的序列 (list 或 tuple)。

    返回:
        pd.Index
            一个新的 Pandas 索引对象，它包含从原始 `MultiIndex` 中选择的级别。
            - 如果 `level_names` 指定单个级别，则返回一个单级 `pd.Index`。
            - 如果 `level_names` 指定多个级别，则返回一个新的 `pd.MultiIndex`，其级别顺序与 `level_names` 中指定的顺序一致。

    示例:
        >>> import pandas as pd
        >>> idx = pd.MultiIndex.from_tuples([
        ...     ('bar', 'one', 'x'), ('bar', 'two', 'y'),
        ...     ('foo', 'one', 'z'), ('foo', 'two', 'w')
        ... ], names=['L1', 'L2', 'L3'])
        >>> idx
        MultiIndex([('bar', 'one', 'x'),
                    ('bar', 'two', 'y'),
                    ('foo', 'one', 'z'),
                    ('foo', 'two', 'w')],
                   names=['L1', 'L2', 'L3'])

        >>> # 选择多个级别 (通过名称列表)
        >>> select_levels(idx, ['L3', 'L1'])
        MultiIndex([('x', 'bar'),
                    ('y', 'bar'),
                    ('z', 'foo'),
                    ('w', 'foo')],
                   names=['L3', 'L1'])
    """
    checks.assert_instance_of(index, pd.MultiIndex)

    # 如果 level_names 是一个整数或字符串，则直接返回该级别对应的值
    if isinstance(level_names, (int, str)):
        # 如果是单个级别，则调用 `index.get_level_values()` 方法获取该级别对应的所有值。
        # `get_level_values()` 可以接受级别名称或整数位置作为参数。
        # 返回的是一个单级的 `pd.Index` 对象。
        return index.get_level_values(level_names)
    levels = [index.get_level_values(level_name) for level_name in level_names]
    return pd.MultiIndex.from_arrays(levels)


def drop_redundant_levels(index: tp.Index) -> tp.Index:
    """
    删除 Pandas 索引（尤其是 MultiIndex）中被认为是“冗余”的级别。

    冗余级别主要指那些不提供额外区分信息或其值模式与默认生成的索引相似的级别。
    具体来说，以下类型的级别被视为冗余：
    1. 如果索引长度大于1，且某一级别的所有值都相同（即该级别只有一个唯一值），并且该级别没有名称。
       例如，一个级别的值是 `['X', 'X', 'X']` 且其名称为 `None`。
    2. 某一级别的所有值构成了一个从0开始的默认整数序列（类似于 `pd.RangeIndex(0, N)`），
       并且该级别没有明确的名称（通常由 `vectorbt.utils.checks.is_default_index` 判断）。
       例如，一个级别的值是 `[0, 1, 2]` 且其名称为 `None`。

    此函数仅在删除冗余级别后，索引中至少还保留一个非冗余级别时才执行删除操作。
    如果所有级别都被认为是冗余的，则原始索引将保持不变。

    参数:
        index: tp.Index
            输入的 Pandas 索引对象。如果不是 `pd.MultiIndex`，则函数会直接返回原始索引。

    返回:
        pd.Index
            删除了冗余级别后的新索引对象。如果原始索引不是 `pd.MultiIndex`，
            或者没有找到可删除的冗余级别（或删除后会导致没有级别），则返回原始索引。

    示例:
        >>> import pandas as pd
        >>> from vectorbt.utils import checks # 假设 checks.is_default_index 已定义

        >>> # 示例1: 包含一个单一无名值的级别
        >>> m_index1 = pd.MultiIndex.from_arrays([
        ...     ['A', 'A', 'B', 'B'],  # level 0
        ...     [0, 0, 0, 0],         # level 1 (冗余: 单一值0, 无名)
        ...     ['X', 'Y', 'X', 'Y']   # level 2
        ... ], names=['L0', None, 'L2'])
        >>> print(drop_redundant_levels(m_index1))
        MultiIndex([('A', 'X'),
                    ('A', 'Y'),
                    ('B', 'X'),
                    ('B', 'Y')],
                   names=['L0', 'L2'])
    """
    if not isinstance(index, pd.MultiIndex):
        return index

    levels_to_drop = []
    # 遍历 MultiIndex 的每一级
    for i in range(index.nlevels):
        # 条件1: 检查当前级别是否具有单个无名值
        # - `len(index) > 1`: 确保索引本身有多于一个条目，否则单个值的级别并非冗余。
        # - `len(index.levels[i]) == 1`: 检查当前级别的唯一值数量是否为1。`index.levels[i]` 获取第i级的所有唯一值。
        # - `index.levels[i].name is None`: 检查当前级别的名称是否为 None。
        if len(index) > 1 and len(index.levels[i]) == 1 and index.levels[i].name is None:
            levels_to_drop.append(i)
        # 条件2: 检查当前级别是否是否等同于一个默认的 RangeIndex (例如 RangeIndex(start=0, stop=N, step=1)) 且无名称。
        elif checks.is_default_index(index.get_level_values(i)):
            levels_to_drop.append(i)
    # 安全措施：仅当删除冗余级别后，MultiIndex 中至少还保留一个级别时，才执行删除操作
    if len(levels_to_drop) < index.nlevels:
        return index.droplevel(levels_to_drop)
    return index


def drop_duplicate_levels(index: tp.Index, keep: tp.Optional[str] = None) -> tp.Index:
    """
    删除多级索引MultiIndex `index` 中具有相同名称和值的级别。

    如果一个多级索引中存在多个级别，它们的名称相同（或者都无名称）且包含完全相同的值序列，
    那么此函数会移除这些重复的级别，只保留其中一个。

    参数:
        index: tp.Index
            输入的 Pandas 索引对象。如果不是多级索引（`pd.MultiIndex`），则原样返回。
        keep: tp.Optional[str], 默认值: None
            当检测到重复级别时，决定保留哪一个。
            - 'first': 保留第一个遇到的重复级别（从左到右，即级别0到级别n-1）。
            - 'last': 保留最后一个遇到的重复级别。
            如果为 `None`，则会从 `vectorbt._settings` 中读取广播配置的 'keep' 值。

    返回:
        tp.Index
            删除了重复级别后的索引。如果输入不是多级索引或没有重复级别，则返回原始索引。

    示例:
        >>> import pandas as pd
        >>> # 创建一个包含重复级别的多级索引
        >>> tuples = [('A', 1, 'X'), ('A', 1, 'Y'), ('B', 2, 'X'), ('B', 2, 'Y')]
        >>> multi_index = pd.MultiIndex.from_tuples(tuples, names=['L1', 'L2_dup1', 'L3'])
        >>> # 假设我们要添加一个与 L2_dup1 完全相同的级别
        >>> df = pd.DataFrame(index=multi_index)
        >>> df = df.set_index(df.index.get_level_values('L2_dup1'), append=True, names='L2_dup2')
        >>> print(df.index)
        MultiIndex([('A', 1, 'X', 1),
                    ('A', 1, 'Y', 1),
                    ('B', 2, 'X', 2),
                    ('B', 2, 'Y', 2)],
                   names=['L1', 'L2_dup1', 'L3', 'L2_dup2'])

        >>> # 删除重复级别，保留第一个 (L2_dup1)
        >>> cleaned_index_first = drop_duplicate_levels(df.index, keep='first')
        >>> print(cleaned_index_first)
        MultiIndex([('A', 'X', 1),
                    ('A', 'Y', 1),
                    ('B', 'X', 2),
                    ('B', 'Y', 2)],
                   names=['L1', 'L3', 'L2_dup1']) # 注意L2_dup2被删除，L2_dup1保留
    """
    from vectorbt._settings import settings
    broadcasting_cfg = settings['broadcasting']

    if keep is None:
        keep = broadcasting_cfg['keep']
    if not isinstance(index, pd.MultiIndex):
        return index
    checks.assert_in(keep.lower(), ['first', 'last'])

    levels = []
    levels_to_drop = []
    if keep == 'first':
        # index.nlevels 是 Pandas 中 MultiIndex（多级索引）对象的一个属性，它返回多级索引中的级别数量。
        # 例如，如果一个 MultiIndex 有三个级别（比如国家、城市、街道），那么 index.nlevels 将返回 3。
        r = range(0, index.nlevels)
    else:
        r = range(index.nlevels - 1, -1, -1)  # 向后循环
    # 遍历多级索引的每个级别
    for i in r:
        # 构建一个元组 `level` 来唯一标识当前级别：
        # 第一个元素是当前级别的名称 (index.levels[i].name)
        # 第二个元素是当前级别所有值的元组 (通过 .to_numpy().tolist() 转换为纯 Python 列表再转元组，以确保可哈希和比较)
        level = (index.levels[i].name, tuple(index.get_level_values(i).to_numpy().tolist()))
        if level not in levels:
            levels.append(level)
        else:
            levels_to_drop.append(i)
    return index.droplevel(levels_to_drop)


@njit(cache=True)
def _align_index_to_nb(a: tp.Array1d, b: tp.Array1d) -> tp.Array1d:
    """
    Numba 加速函数：返回将一维数组 `a` 对齐到一维数组 `b` 所需的索引。

    对于 `b` 中的每个元素，此函数会在 `a` 中查找第一个出现的相同元素，
    并记录下该元素在 `a` 中的索引位置。最终返回一个包含这些索引位置的数组。
    这个函数主要用作 `align_index_to` 函数的内部辅助，以提高性能。

    参数:
        a: tp.Array1d
            一维 NumPy 数组，作为源数组，将在其中查找元素。
        b: tp.Array1d
            一维 NumPy 数组，作为目标数组，其元素将被用于在 `a` 中进行匹配。

    返回:
        tp.Array1d
            一个与 `b` 等长的一维 NumPy 数组（`dtype=np.int64`），
            其中每个元素是 `b` 对应位置元素在 `a` 中首次出现的索引。
            如果 `b` 中的某个元素在 `a` 中找不到，则行为未定义（理论上应该能找到，
            因为调用它的 `align_index_to` 函数会预处理数据以确保可对齐性）。
    """
    idxs = np.empty(b.shape[0], dtype=np.int64)
    g = 0
    for i in range(b.shape[0]):
        for j in range(a.shape[0]):
            if b[i] == a[j]:
                idxs[g] = j
                g += 1
                break
    return idxs


def align_index_to(index1: tp.Index, index2: tp.Index) -> pd.IndexSlice:
    """
    如果源索引 `index1` 和目标索引 `index2` 之间存在可用于对齐的共同级别，
    则此函数计算并返回一个 Pandas 索引切片 (`pd.IndexSlice`)。
    应用此切片于 `index1`（或基于 `index1` 的数据结构）可以使其元素顺序和重复模式
    与 `index2` 在这些共同级别上保持一致。

    该函数旨在处理广播或对齐操作中，一个索引需要匹配另一个索引结构的情况。
    它首先会尝试通过级别名称匹配来识别共同级别，并确保 `index2` 中的级别值是 `index1` 相应级别值的子集。
    然后，它会尝试通过简单的平铺（tiling）操作来对齐。如果平铺不可行，
    则会进行更复杂的元素级比较（通过内部的 Numba 加速函数 `_align_index_to_nb`）来找到精确的对齐方式。
    
    index1: Ai index2:Bj
    根据索引的名称找到匹配对应的例如：A1->B1, A2->B2
    构造 Aindex = S[A1(A1 ∩ B1)], S[A2(A2 ∩ B2)] 注意这里不能有重复行
        Bindex = S[B1(A1 ∩ B1)], S[B2(A2 ∩ B2)]
    于是可进一步构造 S[Aindex(Aindex ∩ Bindex)]
                   S[Bindex(Aindex ∩ Bindex)]
    然后调用 _align_index_to_nb(S[Aindex(Aindex ∩ Bindex)], S[Bindex(Aindex ∩ Bindex)]) 即可
    

    参数:
        index1: tp.Index
            源 Pandas 索引对象。此索引将被对齐。
            如果不是 MultiIndex，则会被视为单级 MultiIndex 进行处理。
        index2: tp.Index
            目标 Pandas 索引对象。`index1` 将被对齐到此索引的结构。
            如果不是 MultiIndex，则会被视为单级 MultiIndex 进行处理。

    返回:
        pd.IndexSlice
            一个 Pandas 索引切片对象。当这个切片应用于 `index1` 时（例如 `some_series.loc[slice_obj]`
            或 `index1[slice_obj]`），会得到一个与 `index2` 在共同级别上对齐的新索引。
            如果 `index1` 和 `index2` 已经完全相等，则返回 `pd.IndexSlice[:]` (表示选择所有元素)。

    示例:
        >>> import pandas as pd
        >>> import numpy as np

        >>> # 示例 1: 简单平铺对齐
        >>> idx1_simple = pd.Index(['a', 'b'], name='L1')
        >>> idx2_simple = pd.Index(['a', 'b', 'a', 'b'], name='L1')
        >>> slice_obj_simple = align_index_to(idx1_simple, idx2_simple)
        >>> print(slice_obj_simple)
        [0 1 0 1]
        >>> print(idx1_simple[slice_obj_simple])
        Index(['a', 'b', 'a', 'b'], dtype='object', name='L1')

        >>> # 示例 2: MultiIndex 对齐，基于共同级别
        >>> m_idx1 = pd.MultiIndex.from_tuples([('x', 1), ('y', 2)], names=['char', 'num'])
        >>> m_idx2 = pd.MultiIndex.from_tuples(
        ...     [('x', 1), ('y', 2), ('x', 1)], names=['char', 'num']
        ... )
        >>> slice_obj_multi = align_index_to(m_idx1, m_idx2)
        >>> print(slice_obj_multi)
        [0 1 0]
        >>> print(m_idx1[slice_obj_multi])
        MultiIndex([('x', 1),
                    ('y', 2),
                    ('x', 1)],
                   names=['char', 'num'])

        >>> # 示例 3: 目标索引的共同级别值是源索引的子集
        >>> idx1_sub = pd.Index(['a', 'b', 'c'], name='L1')
        >>> idx2_sub = pd.Index(['b', 'a', 'b'], name='L1')
        >>> slice_obj_sub = align_index_to(idx1_sub, idx2_sub)
        >>> print(slice_obj_sub) # 对应于 idx1 中的 [1, 0, 1]
        [1 0 1]
        >>> print(idx1_sub[slice_obj_sub])
        Index(['b', 'a', 'b'], dtype='object', name='L1')

        >>> # 示例 4: 已经对齐的情况
        >>> idx_eq1 = pd.Index(['a', 'b'])
        >>> idx_eq2 = pd.Index(['a', 'b'])
        >>> print(align_index_to(idx_eq1, idx_eq2))
        slice(None, None, None)

        >>> # 示例 5: 涉及不同级别名称但部分值重叠（需要共同级别名称才能对齐）
        >>> idx1_ex5 = pd.MultiIndex.from_tuples([('a', 1), ('b', 2)], names=['common', 'val1'])
        >>> idx2_ex5 = pd.MultiIndex.from_tuples([('a', 10), ('b', 20), ('a', 30)], names=['common', 'val2'])
        >>> # 结果会是基于 common 对齐
        >>> slice_ex5 = align_index_to(idx1_ex5, idx2_ex5)
        >>> print(slice_ex5) # 预期是 [0, 1, 0]
        [0 1 0]
        >>> print(idx1_ex5[slice_ex5])
        MultiIndex([('a',  1),
                    ('b',  2),
                    ('a',  1)],
                   names=['common', 'val1'])
    """
    # 确保 index1 是 MultiIndex，如果不是，则从其数组形式创建一个单级 MultiIndex
    if not isinstance(index1, pd.MultiIndex):
        index1 = pd.MultiIndex.from_arrays([index1])
    if not isinstance(index2, pd.MultiIndex):
        index2 = pd.MultiIndex.from_arrays([index2])
    # 如果两个索引完全相等（包括值、顺序和名称），则它们已经对齐，返回一个选择所有元素的切片。  
    if pd.Index.equals(index1, index2):
        return pd.IndexSlice[:]
    
    # 例子
    # ex_idx1:
    # MultiIndex([('X', 10, 'alpha'), ('Y', 20, 'beta'), ('X', 30, 'gamma')],
    #            names=['L_common1', 'L_common2', 'L_unique1'])
    # ex_idx2_adj:
    # MultiIndex([('X', 10,   'one'), ('X', 30,   'two'),
    #             ('X', 10, 'three'), ('Y', 20,  'four')],
    #            names=['L_common1', 'L_common2', 'L_unique2'])

    # 初始化一个字典 `mapper`，用于存储 index1 和 index2 之间共同级别的映射关系。
    # 键是 index1 中的级别索引，值是 index2 中对应的级别索引。
    mapper = {}
    for i in range(index1.nlevels):
        for j in range(index2.nlevels):
            name1 = index1.names[i]
            name2 = index2.names[j]
            # 如果两个级别的名称相同
            if name1 == name2:
                # 检查 index2 当前级别 j 的所有唯一值是否都是 index1 对应级别 i 唯一值的子集。
                # 这是对齐的必要条件：目标索引的共同级别值不能超出源索引的范围。
                if set(index2.levels[j]).issubset(set(index1.levels[i])):
                    if i in mapper:
                        raise ValueError(f"在第二个索引中有多个名为{name1}的候选级别")
                    mapper[i] = j
                    continue
                if name1 is not None:
                    raise ValueError(f"第二个索引中的级别{name1}包含第一个索引中不存在的值")
    if len(mapper) == 0:
        raise ValueError("找不到可用于对齐两个索引的共同级别")

    # 初始化一个列表 `factorized`，用于存储共同级别因子化后的整数编码。
    # 因子化是将每个级别的值映射到从0开始的整数，这有助于后续的比较和 Numba 处理。
    factorized = []
    # 遍历 mapper 中记录的共同级别映射关系 (index1的级别k -> index2的级别v)
    for k, v in mapper.items():
        # 将 index1 的级别 k 的值和 index2 的级别 v 的值连接起来，
        # 然后对这个合并后的 Series 进行因子化。
        # `pd.factorize` 返回一个元组：(整数编码数组, 唯一值索引)
        # 我们只需要整数编码数组。
        factorized.append(pd.factorize(pd.concat((
            index1.get_level_values(k).to_series(),
            index2.get_level_values(v).to_series()
        )))[0])
    # 例如
    # [
    #   np.array([0, 1, 0, 0, 0, 0, 1]),  
    #   np.array([0, 1, 2, 0, 2, 0, 1]) 
    # ]
    # np.stack(factorized) ->
    # [[0, 1, 0, 0, 0, 0, 1],
    #  [0, 1, 2, 0, 2, 0, 1]]
    # np.transpose(...) ->
    # [[0, 0],
    #  [1, 1],
    #  [0, 2], 
    #  [0, 0], 
    #  [0, 2], 
    #  [0, 0], 
    #  [1, 1]]
    stacked = np.transpose(np.stack(factorized))
    # [[0, 0],
    #  [1, 1],
    #  [0, 2]]
    indices1 = stacked[:len(index1)]
    # [[0, 0],
    #  [0, 2],
    #  [0, 0],
    #  [1, 1]]
    indices2 = stacked[len(index1):]
    # 检查 `indices1` (基于共同级别的值组合) 是否有重复行。
    # 如果有重复，意味着 `index1` 在这些共同级别上存在相同的条目，这会导致对齐不明确。
    # 对于示例的 indices1: np.unique([[0,0],[1,1],[0,2]], axis=0) 的长度是3，等于 len(indices1)。所以没有重复。
    if len(np.unique(indices1, axis=0)) != len(indices1):
        raise ValueError("不允许在第一个索引中使用重复的值")

    # 尝试通过平铺 (tiling) 操作进行对齐。
    # 检查 index2 的长度是否是 index1 长度的整数倍。
    # 对于示例: len(ex_idx2_adj)=4, len(ex_idx1)=3. 4 % 3 != 0. 此条件为 False.
    if len(index2) % len(index1) == 0:
        tile_times = len(index2) // len(index1)
        index1_tiled = np.tile(indices1, (tile_times, 1))
        if np.array_equal(index1_tiled, indices2):
            return pd.IndexSlice[np.tile(np.arange(len(index1)), tile_times)]

    # [[0, 0], [1, 1], [0, 2], [0, 0], [0, 2], [0, 0], [1, 1]]——>np.array([0, 1, 2, 0, 2, 0, 1])
    unique_indices = np.unique(stacked, axis=0, return_inverse=True)[1]
    # 分离出对应于 index1 的唯一行编码: np.array([0, 1, 2])
    unique1 = unique_indices[:len(index1)]
    # 分离出对应于 index2 的唯一行编码: np.array([0, 2, 0, 1])
    unique2 = unique_indices[len(index1):]
    # 调用 Numba 加速的内部函数 `_align_index_to_nb`，
    # 它会找到将 `unique1` 对齐到 `unique2` 所需的索引。
    return pd.IndexSlice[_align_index_to_nb(unique1, unique2)]


def align_indexes(indexes: tp.Sequence[tp.Index]) -> tp.List[tp.IndexSlice]:
    """
    将序列 `indexes` 中的多个 Pandas 索引对象相互对齐。
    返回[i]对应：indexes中最长（可能很多个）且最前能够对齐到Indexindexs[i]的IndexSlice

    示例:
        >>> import pandas as pd
        >>> index1 = pd.Index([1, 2, 1], name='L1')  # 与index2共享L1级别
        >>> index2 = pd.MultiIndex.from_tuples([('x',1),('x',2),('y',1),('y',2)], names=['L2','L1'])
        >>> index3 = pd.MultiIndex.from_tuples([('x',),('y',)], names=['L2'])
        >>> indices = align_indexes([index1, index2, index3])
        
        # 步骤分解:
        # 1. max_len = 4（index2的长度）
        # 2. 处理index1（长度3）:
        #    - 找到第一个最长索引index2
        #    - 通过align_index_to对齐，返回索引位置数组[0,1,0,1]
        # 3. 处理index2（长度4）: 直接返回[:]
        # 4. 处理index3（长度2）:
        #    - 找到第一个最长索引index2
        #    - 通过align_index_to对齐，返回索引位置数组[0,0,1,1]
        >>> print(indices)
        [<pandas.core.indexing.IndexSlice object at ...>,
         <pandas.core.indexing.IndexSlice object at ...>,
         <pandas.core.indexing.IndexSlice object at ...>]
        >>> index1[indices[0]]  # 返回 [1, 2, 1, 2]
        >>> index3[indices[2]]  # 返回 ['x', 'x', 'y', 'y']
    """
    max_len = max(map(len, indexes))
    indices = []
    for i in range(len(indexes)):
        index_i = indexes[i]
        if len(index_i) == max_len:
            indices.append(pd.IndexSlice[:])
        else:
            for j in range(len(indexes)):
                index_j = indexes[j]
                if len(index_j) == max_len:
                    try:
                        indices.append(align_index_to(index_i, index_j))
                        break
                    except ValueError:
                        pass
            if len(indices) < i + 1:
                raise ValueError(f"位置{i}的索引无法对齐")
    return indices


OptionalLevelSequence = tp.Optional[tp.Sequence[tp.Union[None, tp.Level]]]


def pick_levels(index: tp.Index,
                required_levels: OptionalLevelSequence = None,
                optional_levels: OptionalLevelSequence = None) -> tp.Tuple[tp.List[int], tp.List[int]]:
    """
    从多级索引 (MultiIndex) 中根据用户指定的“必需级别”和“可选级别”列表，
    智能地选择并返回这些级别在原始索引中的整数位置。

    该函数主要用于当用户需要从一个多级索引中提取特定子集级别进行操作时，
    允许用户灵活地指定哪些级别是必须存在的（如果指定为 None，则从剩余级别中自动填充），
    哪些级别是可选的（如果指定为 None，则表示跳过该可选槽位）。

    参数:
        index: tp.Index
            输入的 Pandas 索引对象。此函数期望它是一个 `pd.MultiIndex`。
            如果不是 `pd.MultiIndex`，将引发 `AssertionError`。
        required_levels: OptionalLevelSequence, 默认值: None
            一个序列，定义了必须被选择的级别。序列中的每个元素可以是：
            - 级别名称 (str): 例如 'level_name'。
            - 级别整数位置 (int): 例如 0, 1, -1 (表示最后一个级别)。
            - None: 表示该“必需”槽位将从 `index` 中尚未被其他指定级别占用的级别里按顺序自动选择一个。
            如果 `required_levels` 本身为 `None` 或空列表，则表示没有显式指定的必需级别。
        optional_levels: OptionalLevelSequence, 默认值: None
            一个序列，定义了可选的级别。序列中的每个元素可以是：
            - 级别名称 (str)。
            - 级别整数位置 (int)。
            - None: 表示该“可选”槽位将被忽略，即不选择任何级别来填充它。
            如果 `optional_levels` 本身为 `None` 或空列表，则表示没有可选级别。

    返回:
        tp.Tuple[tp.List[int], tp.List[int]]
            一个元组，包含两个列表：
            1. 第一个列表 (`_required_levels_indices`): 包含所有被选中的必需级别在原始 `index` 中的整数位置。
               列表的顺序与输入 `required_levels` 序列中非None元素的顺序以及自动填充None槽位的顺序一致。
            2. 第二个列表 (`_optional_levels_indices`): 包含所有被选中的可选级别在原始 `index` 中的整数位置。
               如果原始 `optional_levels` 中的某个元素是 `None`，则结果列表的对应位置也会是 `None`。
               列表的顺序与输入 `optional_levels` 序列的顺序一致。

    异常:
        AssertionError:
            - 如果 `index` 不是 `pd.MultiIndex`。
            - 如果 `required_levels` 或 `optional_levels` 中的元素不是 `str`, `int`, 或 `None`。
        ValueError:
            - 如果指定的级别名称在 `index` 的名称中找不到。
            - 如果在 `required_levels` 中指定了 `None` 来自动填充，但实际可用的剩余级别数量
              与需要填充的 `None` 槽位数量不匹配。具体来说，如果 (index.nlevels - (已指定的必需级别数) - (已指定的可选级别数))
              不等于 `required_levels` 中 `None` 的数量。

    示例:
        >>> import pandas as pd
        >>> idx = pd.MultiIndex.from_tuples([
        ...     ('A', 1, 'x', 10.0), ('B', 2, 'y', 20.0), ('C', 3, 'z', 30.0)
        ... ], names=['L1', 'L2_int', 'L3_char', 'L4_float'])
        >>> print("原始索引:")
        >>> print(idx)
        原始索引:
        MultiIndex([('A', 1, 'x', 10.0),
                    ('B', 2, 'y', 20.0),
                    ('C', 3, 'z', 30.0)],
                   names=['L1', 'L2_int', 'L3_char', 'L4_float'])

        >>> # 示例 1: 指定部分必需级别 (名称和None)，无可选级别
        >>> req_lvls, opt_lvls = pick_levels(idx, required_levels=['L1', None, 'L4_float'])
        >>> print(f"示例1 - 必需: {req_lvls}, 可选: {opt_lvls}")
        示例1 - 必需: [0, 1, 3], 可选: []
        >>> # L1 -> 0, L4_float -> 3. 剩余级别有 L2_int(1), L3_char(2).
        >>> # required_levels 中间的 None 会选择 L2_int(1).

        >>> # 示例 2: 指定部分可选级别 (位置和None)，一个必需级别
        >>> req_lvls, opt_lvls = pick_levels(idx, required_levels=[0], optional_levels=[None, 'L3_char', -1])
        >>> print(f"示例2 - 必需: {req_lvls}, 可选: {opt_lvls}")
        示例2 - 必需: [0], 可选: [None, 2, 3]
        >>> # 必需: 级别0 (L1).
        >>> # 可选: 第一个None跳过, L3_char -> 2, 最后一个级别 (-1) -> L4_float(3).

        >>> # 示例 3: 所有必需级别都通过 None 自动选择，指定一个可选级别
        >>> req_lvls, opt_lvls = pick_levels(idx, required_levels=[None, None], optional_levels=['L1'])
        >>> print(f"示例3 - 必需: {req_lvls}, 可选: {opt_lvls}")
        示例3 - 必需: [1, 2], 可选: [0]
        >>> # 可选: L1 -> 0. 剩余级别 L2_int(1), L3_char(2), L4_float(3).
        >>> # 两个None会按顺序选择 L2_int(1) 和 L3_char(2).
    """
    if required_levels is None:
        required_levels = []
    if optional_levels is None:
        optional_levels = []
    checks.assert_instance_of(index, pd.MultiIndex)

    # 计算在 optional_levels 中实际指定了具体级别（非 None）的数量。
    n_opt_set = len(list(filter(lambda x: x is not None, optional_levels)))
    # 计算在 required_levels 中实际指定了具体级别（非 None）的数量。
    n_req_set = len(list(filter(lambda x: x is not None, required_levels)))
    # 计算在减去已指定的可选级别后，索引中还剩下多少级别。
    # 这些是可能用于满足必需级别（包括自动填充None槽位）的候选级别。
    n_levels_left = index.nlevels - n_opt_set
    # 检查是否有必需级别被指定为 None (等待自动填充)。
    if n_req_set < len(required_levels):
        # 如果有必需级别是 None，那么 (剩余可用于必需的级别数) 必须等于 (总共声明的必需级别数)。
        # 换句话说，(index.nlevels - n_opt_set) 必须等于 len(required_levels)。
        # 如果不相等，说明提供的 MultiIndex 的级别总数不足以满足所有声明的必需级别（包括指定和自动填充的）。
        if n_levels_left != len(required_levels):
            n_expected = len(required_levels) + n_opt_set
            raise ValueError(f"预期{n_expected}个级别，找到{index.nlevels}个")

    levels_left = list(range(index.nlevels))

    # 选择可选级别
    _optional_levels = []
    for level in optional_levels:
        level_pos = None
        if level is not None:
            checks.assert_instance_of(level, (int, str))
            if isinstance(level, str):
                level_pos = index.names.index(level)
            else:
                level_pos = level
            if level_pos < 0:
                level_pos = index.nlevels + level_pos
            levels_left.remove(level_pos)
        _optional_levels.append(level_pos)

    # 选择必需级别
    _required_levels = []
    for level in required_levels:
        level_pos = None
        if level is not None:
            checks.assert_instance_of(level, (int, str))
            if isinstance(level, str):
                level_pos = index.names.index(level)
            else:
                level_pos = level
            if level_pos < 0:
                level_pos = index.nlevels + level_pos
            levels_left.remove(level_pos)
        _required_levels.append(level_pos)
    for i, level in enumerate(_required_levels):
        if level is None:
            _required_levels[i] = levels_left.pop(0)

    return _required_levels, _optional_levels


def find_first_occurrence(index_value: tp.Any, index: tp.Index) -> int:
    """
    在给定的 Pandas 索引对象中查找指定值首次出现的整数位置。

    `pandas.Index.get_loc` 方法用于获取标签在索引中的位置。
    然而，`get_loc` 的返回值类型多样：
    - 如果标签唯一，返回整数位置。
    - 如果标签不唯一（重复），可能返回布尔掩码数组或切片。
    - 对于某些类型的索引和标签，也可能返回整数列表。
    此函数旨在统一处理这些情况，并总是返回第一次出现该值的整数索引。

    参数:
        index_value (tp.Any):
            需要在 `index` 中查找其首次出现位置的值。
            该值的类型应与 `index` 中元素的类型兼容。
        index (tp.Index):
            Pandas 索引对象 (例如 `pd.Index`, `pd.DatetimeIndex`, `pd.MultiIndex` 等)，
            将在此索引中搜索 `index_value`。

    返回:
        int:
            `index_value` 在 `index` 中首次出现的整数位置（从0开始计数）。

    示例:
        >>> import pandas as pd
        >>> # 创建一个包含重复值的简单索引
        >>> idx = pd.Index(['a', 'b', 'c', 'a', 'd', 'b'])
        >>> find_first_occurrence('a', idx)
        0
        >>> find_first_occurrence('b', idx)
        1
        >>> find_first_occurrence('d', idx)
        4

        >>> # 创建一个 DatetimeIndex
        >>> dt_idx = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-01 10:00', '2023-01-03'])
        >>> # 查找日期 '2023-01-01' 的首次出现（可能匹配多个时间点）
        >>> find_first_occurrence(pd.Timestamp('2023-01-01'), dt_idx)
        0

        >>> # 如果索引中包含重复标签，get_loc 可能返回布尔数组
        >>> idx_multi_occurrence = pd.Index([10, 20, 10, 30, 10])
        >>> # 假设 idx_multi_occurrence.get_loc(10) 返回类似 array([ True, False,  True, False,  True])
        >>> find_first_occurrence(10, idx_multi_occurrence)
        0
    """
    # 调用 Pandas 索引对象的 get_loc 方法获取指定值的位置信息。
    # get_loc 的返回值可以是整数、切片对象、列表或布尔型 NumPy 数组。
    loc = index.get_loc(index_value)

    # 检查 get_loc 返回的是否为切片对象 (slice)。
    # 例如，在时间序列索引中，查找一个日期可能会返回一个包含该日期所有时间点的切片。
    if isinstance(loc, slice):
        # 如果是切片，第一次出现的位置是切片的起始点。
        return loc.start

    # 检查 get_loc 返回的是否为列表 (list)。
    # 某些情况下（例如特定版本的Pandas或特定索引类型），对于重复值，get_loc可能返回包含所有匹配位置的列表。
    elif isinstance(loc, list):
        # 如果是列表，第一次出现的位置是列表的第一个元素。
        return loc[0]

    # 检查 get_loc 返回的是否为 NumPy 数组 (np.ndarray)。
    # 通常，当索引中存在重复值时，get_loc 会返回一个布尔掩码数组，标记所有匹配值的位置。
    elif isinstance(loc, np.ndarray):
        # 如果是 NumPy 数组（通常是布尔掩码），np.flatnonzero(loc) 会返回所有 True 值（即匹配位置）的扁平化索引。
        # 然后取这个结果数组的第一个元素，即为第一次出现的位置。
        # 例如，如果 loc 是 array([False, True, False, True])，np.flatnonzero(loc) 是 array([1, 3])，则返回 1。
        return np.flatnonzero(loc)[0]

    # 如果 loc 不是以上任何类型（切片、列表、NumPy 数组），
    # 那么它应该是一个整数，表示该值在索引中的唯一位置。
    return loc
