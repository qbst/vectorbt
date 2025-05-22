# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Functions for reshaping arrays.

Reshape functions transform a pandas object/NumPy array in some way, such as tiling, broadcasting,
and unstacking."""

import functools
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numba import njit
try:
    # 尝试从numpy导入broadcast_shapes函数
    # 这在较新版本的numpy中可用
    from numpy import broadcast_shapes

    # 如果成功导入broadcast_shapes，则不需要broadcast_shape
    broadcast_shape = None
except ImportError:
    # 在较旧版本的numpy中，broadcast_shapes不可用
    # 所以我们从stride_tricks模块导入_broadcast_shape作为替代
    from numpy.lib.stride_tricks import _broadcast_shape as broadcast_shape

    # 在这种情况下，我们没有broadcast_shapes函数
    broadcast_shapes = None

from vectorbt import _typing as tp
from vectorbt.base import index_fns, array_wrapper
from vectorbt.utils import checks
from vectorbt.utils.config import resolve_dict


def to_any_array(arg: tp.ArrayLike, raw: bool = False) -> tp.AnyArray:
    """
    将任何类似数组的对象（如列表、元组、NumPy数组、Pandas Series/DataFrame）转换为数组。

    此函数的主要目标是提供一个统一的接口来处理不同类型的数组状数据。
    默认情况下，如果输入已经是 NumPy 数组或 Pandas 对象 (Series/DataFrame)，
    并且 `raw` 参数为 `False`，则会直接返回原始对象，以保留其特定类型和元数据。
    如果 `raw` 参数为 `True`，或者输入不是 NumPy/Pandas 对象，则会尝试将其转换为 NumPy 数组。

    参数:
        arg (tp.ArrayLike):
            输入的类似数组的对象。这可以包括 Python 的列表、元组，
            NumPy 的 ndarray，或者 Pandas 的 Series 和 DataFrame。
        raw (bool, optional):
            一个布尔标志，用于控制转换行为。默认为 `False`。
            - 如果 `raw` 为 `False`（默认）：
                - 若 `arg` 是一个 Pandas 对象 (Series 或 DataFrame) 或 NumPy 数组，
                  则直接返回 `arg` 本身，不进行转换。
                - 若 `arg` 是其他类似数组的对象（如列表），则将其转换为 NumPy 数组。
            - 如果 `raw` 为 `True`：
                - `arg` 将总是被尝试转换为 NumPy 数组，即使它原本是 Pandas 对象。

    返回:
        tp.AnyArray:
            转换后的数组。
            - 如果 `raw` 为 `False` 且 `arg` 是 Pandas 对象或 NumPy 数组，则返回原始的 `arg`。
            - 在其他情况下，返回一个 NumPy 数组 (`np.ndarray`)。
    """
    if not raw and checks.is_any_array(arg):
        return arg
    return np.asarray(arg)


def to_pd_array(arg: tp.ArrayLike) -> tp.SeriesFrame:
    """
    将任何类似数组的对象转换为 Pandas Series 或 DataFrame。

    如果输入对象已经是 Pandas Series 或 DataFrame，则直接返回该对象。
    否则，它会尝试将输入转换为 NumPy 数组。
    如果转换后的 NumPy 数组是一维的，则将其转换为 Pandas Series。
    如果转换后的 NumPy 数组是二维的，则将其转换为 Pandas DataFrame。
    对于其他维度的数组（例如0维或3维及以上），会引发 ValueError。

    参数:
        arg (tp.ArrayLike):
            一个类似数组的对象。它可以是列表、元组、NumPy 数组，
            或者已经是 Pandas Series 或 DataFrame。

    返回:
        tp.SeriesFrame:
            转换后的 Pandas Series 或 DataFrame。

    异常:
        ValueError: 如果输入数组在转换为 NumPy 数组后，其维度不是1或2。

    示例:
        >>> import numpy as np
        >>> import pandas as pd
        >>> # 假设 to_pd_array 在当前命名空间或已正确导入
        >>> # from vectorbt.base.reshape_fns import to_pd_array

        >>> # 示例 1: 输入一个列表
        >>> my_list = [1, 2, 3]
        >>> series_from_list = to_pd_array(my_list)
        >>> print(series_from_list)
        0    1
        1    2
        2    3
        dtype: int64
    """
    # 检查输入参数 `arg` 是否已经是 Pandas 对象 (Series 或 DataFrame)
    if checks.is_pandas(arg):
        return arg
    # 如果 `arg` 不是 Pandas 对象，则尝试将其强制转换为 NumPy 数组。
    arg = np.asarray(arg)
    if arg.ndim == 1:
        # 如果 NumPy 数组是一维的，则将其转换为 Pandas Series。
        return pd.Series(arg)
    if arg.ndim == 2:
        # 如果 NumPy 数组是二维的，则将其转换为 Pandas DataFrame。
        return pd.DataFrame(arg)
    raise ValueError("Wrong number of dimensions: cannot convert to Series or DataFrame")


def soft_to_ndim(arg: tp.ArrayLike, ndim: int, raw: bool = False) -> tp.AnyArray:
    """
    尝试将输入 `arg` 柔和地转换为指定维度 `ndim` (最大为2)。

    “柔和转换”意味着函数会尝试在不改变数据内容的前提下，通过改变数组的形状属性
    (例如，将一个单列的二维数组转换为一维数组，或者反之)来达到目标维度。
    它主要处理一维和二维数组之间的转换。

    参数:
        arg (tp.ArrayLike):
            输入的类数组对象。可以是 Python 列表、元组、NumPy 数组，
            或者 Pandas Series/DataFrame。
        ndim (int):
            目标维度。此函数主要支持目标维度为 1 或 2。
        raw (bool, optional):
            一个布尔标志，控制初始转换行为，默认为 `False`。
            - 如果 `raw` 为 `False` (默认):
                - 若 `arg` 是 Pandas 对象或 NumPy 数组，则直接使用 `arg`。
                - 否则，将 `arg` 转换为 NumPy 数组。
            - 如果 `raw` 为 `True`:
                - `arg` 将总是被尝试转换为 NumPy 数组。
            此参数会传递给 `to_any_array` 函数。

    示例:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbt.base.reshape_fns import soft_to_ndim

        >>> # 示例 1: 将二维 NumPy 数组 (3x1) 转换为一维
        >>> arr_2d = np.array([[1], [2], [3]])
        >>> print(f"原始二维数组:\\n{arr_2d}\\n形状: {arr_2d.shape}")
        >>> arr_1d = soft_to_ndim(arr_2d, ndim=1)
        >>> print(f"转换为一维后:\\n{arr_1d}\\n形状: {arr_1d.shape}")
        原始二维数组:
        [[1]
         [2]
         [3]]
        形状: (3, 1)
        转换为一维后:
        [1 2 3]
        形状: (3,)
    """
    # 首先，使用 to_any_array 函数确保输入 arg 是一个 NumPy 数组或 Pandas 对象。
    # raw 参数决定了如果 arg 已经是 NumPy/Pandas 对象，是否还要强制转换为 NumPy 数组。
    arg = to_any_array(arg, raw=raw)

    # 检查目标维度是否为 1
    if ndim == 1:
        # 如果目标维度是1，并且当前参数的维度是2
        if arg.ndim == 2:
            # 检查参数的第二个维度（列数）是否为1。只有单列的二维数组才能柔和地转换为一维。
            if arg.shape[1] == 1:
                # 如果参数是 Pandas DataFrame (通过 checks.is_frame 判断)
                if checks.is_frame(arg):
                    # 对于单列 DataFrame，使用 .iloc[:, 0] 将其转换为 Pandas Series (一维)。
                    return arg.iloc[:, 0]
                # 如果参数是 NumPy 数组 (不是 DataFrame，但仍是二维且单列)
                # 使用 [:, 0] 将其从 (N, 1) 降维到 (N,)。
                return arg[:, 0]  # 降维操作
    # 检查目标维度是否为 2
    if ndim == 2:
        # 如果目标维度是2，并且当前参数的维度是1
        if arg.ndim == 1:
            # 如果参数是 Pandas Series (通过 checks.is_series 判断)
            if checks.is_series(arg):
                # 对于 Series，使用 .to_frame() 将其转换为单列的 DataFrame (二维)。
                return arg.to_frame()
            # 如果参数是 NumPy 数组 (一维)
            # 使用 [:, None] (或 np.newaxis) 在末尾增加一个新的轴，将其从 (N,) 升级到 (N, 1)。
            return arg[:, None]  # 升维操作
    # 如果以上条件都不满足 (例如，arg 已经是目标维度，或者无法进行柔和转换，
    # 如将一个 (N, M) 且 M > 1 的数组转为一维)，则不执行任何操作，直接返回原始 (或经 to_any_array 处理后的) arg。
    return arg  # 不做任何操作


def to_1d(arg: tp.ArrayLike, raw: bool = False) -> tp.AnyArray1d:
    """
    将输入参数 `arg` 重塑为一维数组。

    此函数会尝试将各种输入（如标量、列表、二维数组/DataFrame）转换为
    一维的 NumPy 数组或 Pandas Series。

    参数:
        arg (tp.ArrayLike):
            输入的类数组对象。可以是 Python 列表、元组、标量值、
            NumPy 数组，或者 Pandas Series/DataFrame。
        raw (bool, optional):
            一个布尔标志，控制输出类型，默认为 `False`。
            - 如果 `raw` 为 `False` (默认):
                - 如果输入 `arg` 可以被转换为 Pandas Series (例如，一维数组，
                  或单列的 DataFrame)，则返回 Pandas Series。
                - 否则，如果结果是一维的，返回 NumPy 数组。
            - 如果 `raw` 为 `True`:
                - 函数将总是尝试返回一个 NumPy 数组。
            此参数主要影响 `to_any_array` 的行为以及最终类型的确定。

    返回:
        tp.AnyArray1d:
            转换后的一维数组。
            - 如果 `raw` 为 `False` 且输入可以表示为 Pandas Series，则返回 Series。
            - 否则返回一维 NumPy 数组。

    示例:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbt.base.reshape_fns import to_1d

        >>> # 示例 1: 输入列表
        >>> my_list = [1, 2, 3]
        >>> arr_1d_list = to_1d(my_list)
        >>> print(f"从列表转换:\\n{arr_1d_list}\\n类型: {type(arr_1d_list)}")
        从列表转换:
        [1 2 3]
        类型: <class 'numpy.ndarray'>

        >>> # 示例 2: 输入 Pandas Series (已经是1D)
        >>> series = pd.Series([4, 5, 6], name='data')
        >>> series_1d = to_1d(series)
        >>> print(f"\\n从 Series 转换 (raw=False):\\n{series_1d}\\n类型: {type(series_1d)}")
        从 Series 转换 (raw=False):
        0    4
        1    5
        2    6
        Name: data, dtype: int64
        类型: <class 'pandas.core.series.Series'>
    """
    # 首先，使用 to_any_array 函数确保输入 arg 是一个 NumPy 数组或 Pandas 对象。
    arg = to_any_array(arg, raw=raw)

    # 检查转换后的 arg 是否为二维
    if arg.ndim == 2:
        # 如果是二维，检查其列数是否为1
        if arg.shape[1] == 1:
            # 如果是单列，检查是否是 DataFrame
            if checks.is_frame(arg):
                # 返回一个 Pandas Series (一维)
                return arg.iloc[:, 0]
            # 如果不是 DataFrame (NumPy 二维数组，因为arg.ndim == 2所以不可能是Series)
            # 则通过 [:, 0] 将其从 (N, 1) 形状转换为 (N,) 形状的一维 NumPy 数组
            return arg[:, 0]
    # 如果 arg 本来就是一维数组
    if arg.ndim == 1:
        # 直接返回该一维数组
        return arg
    # 如果 arg 是一个标量 (0维数组)
    elif arg.ndim == 0:
        # 使用 .reshape((1,)) 将其转换为包含单个元素的一维数组
        return arg.reshape((1,))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 1 dimension")


# 创建一个函数 to_1d_array，它是 to_1d 函数的偏函数版本
# 通过设置 raw=True，确保返回的始终是 NumPy 数组而非 Pandas 对象
# 这个函数在需要强制将任何输入转换为一维 NumPy 数组时非常有用
to_1d_array = functools.partial(to_1d, raw=True)


def to_2d(arg: tp.ArrayLike, raw: bool = False, expand_axis: int = 1) -> tp.AnyArray2d:
    """
    将输入参数 `arg` 重塑为二维数组。

    此函数会尝试将各种输入（如标量、列表、一维数组/Series）转换为
    二维的 NumPy 数组或 Pandas DataFrame。

    参数:
        arg (tp.ArrayLike):
            输入的类数组对象。可以是 Python 列表、元组、标量值、
            NumPy 数组，或者 Pandas Series/DataFrame。
        raw (bool, optional):
            一个布尔标志，控制输出类型，默认为 `False`。
            - 如果 `raw` 为 `False` (默认):
                - 如果输入 `arg` 可以被转换为 Pandas DataFrame (例如，二维数组，
                  或一维的 Series/NumPy 数组)，则返回 Pandas DataFrame。
                - 否则，如果结果是二维的，返回 NumPy 数组。
            - 如果 `raw` 为 `True`:
                - 函数将总是尝试返回一个二维 NumPy 数组。
            此参数主要影响 `to_any_array` 的行为以及最终类型的确定。
        expand_axis (int, optional):
            当输入 `arg` 是一维时，指定扩展新轴的位置。默认为 `1`。
            - `expand_axis = 0`: 将一维数组 `(N,)` 转换为 `(1, N)` 的二维数组。
                                 如果输入是 Pandas Series，其索引将成为列名。
            - `expand_axis = 1`: 将一维数组 `(N,)` 转换为 `(N, 1)` 的二维数组。
                                 如果输入是 Pandas Series，它会变成单列 DataFrame。
            此参数仅在一维输入转换为二维时有效。

    返回:
        tp.AnyArray2d:
            转换后的二维数组。
            - 如果 `raw` 为 `False` 且输入可以表示为 Pandas DataFrame，则返回 DataFrame。
            - 否则返回二维 NumPy 数组。
    """
    arg = to_any_array(arg, raw=raw)
    if arg.ndim == 2:
        return arg
    elif arg.ndim == 1:
        if checks.is_series(arg):
            # 如果是 Series，根据 expand_axis 的值进行转换
            if expand_axis == 0:
                # expand_axis 为 0 时，将 Series 转换为单行 DataFrame。
                # Series 的值 arg.values 变为 DataFrame 的一行 (通过 [None, :] 实现)。
                # Series 的索引 arg.index 变为 DataFrame 的列名。
                return pd.DataFrame(arg.values[None, :], columns=arg.index)
            elif expand_axis == 1:
                # expand_axis 为 1 (默认) 时，使用 Series 的 .to_frame() 方法。
                # 这会将 Series 转换为单列 DataFrame，Series 的 name (如果有) 会成为列名。
                return arg.to_frame()
        # 如果 arg 是一维 NumPy 数组，或者是一个 Pandas Series 但 expand_axis 不是0或1 (虽然上面已经处理)
        # 使用 np.expand_dims 扩展维度。
        # expand_axis 参数指定了新轴插入的位置。
        return np.expand_dims(arg, expand_axis)
    elif arg.ndim == 0:
        return arg.reshape((1, 1))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions")


to_2d_array = functools.partial(to_2d, raw=True)


def to_dict(arg: tp.ArrayLike, orient: str = 'dict') -> dict:
    """
    将输入的类数组对象转换为字典。

    此函数首先会尝试将输入转换为 Pandas Series 或 DataFrame，
    然后利用 Pandas 对象自身的 `to_dict` 方法进行转换，或者根据 `orient` 参数
    采用自定义的转换逻辑。

    参数:
        arg (tp.ArrayLike):
            输入的类数组对象。可以是列表、元组、NumPy 数组，
            或者已经是 Pandas Series 或 DataFrame。
        orient (str, optional):
            指定字典的格式。默认为 `'dict'`。
            - 如果是 Pandas DataFrame，此参数直接传递给 `DataFrame.to_dict()`。
              常见的值包括：
                - `'dict'` (默认): 类似 `{column: {index: value}}`
                - `'list'`: 类似 `{column: [values]}`
                - `'series'`: 类似 `{column: Series(values, index=index)}`
                - `'split'`: 类似 `{'index': [index], 'columns': [columns], 'data': [values]}`
                - `'records'`: 类似 `[{column: value}, ..., {column: value}]`
                - `'index'`: 类似 `{index: {column: value}}`
            - 如果是 Pandas Series，此参数也传递给 `Series.to_dict()`，但行为相对简单，
              通常是 `index: value` 的形式。
            - `'index_series'` (自定义): 这是一个本函数特有的选项。
              如果 `arg` 最终被转换为 Pandas Series (或单列/单行 DataFrame 能被有效视为 Series)，
              此选项会创建一个字典，其中键是 Series 的索引，值是 Series 对应的值。
              这对于需要将 Series 的每个索引-值对明确表示为字典条目的场景很有用。

    返回:
        dict:
            转换后的字典。

    示例:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbt.base.reshape_fns import to_dict

        >>> # 示例 1: 输入列表 (会先转为 Series)
        >>> my_list = [10, 20, 30]
        >>> dict_from_list_default = to_dict(my_list) # orient='dict'
        >>> print(f"从列表转换 (orient='dict'): {dict_from_list_default}")
        从列表转换 (orient='dict'): {0: 10, 1: 20, 2: 30}

        >>> # 示例 2: 输入 Pandas Series
        >>> series = pd.Series([100, 200], index=['a', 'b'], name='data')
        >>> dict_from_series_default = to_dict(series) # orient='dict'
        >>> print(f"\\n从 Series 转换 (orient='dict'): {dict_from_series_default}")
        从 Series 转换 (orient='dict'): {'a': 100, 'b': 200}

        >>> dict_from_series_index = to_dict(series, orient='index_series')
        >>> print(f"从 Series 转换 (orient='index_series'): {dict_from_series_index}")
        从 Series 转换 (orient='index_series'): {'a': 100, 'b': 200}

        >>> # 示例 3: 输入 Pandas DataFrame
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, index=['row1', 'row2'])
        >>> print(f"\\n原始 DataFrame:\\n{df}")
        原始 DataFrame:
              col1  col2
        row1     1     3
        row2     2     4

        >>> dict_from_df_default = to_dict(df) # orient='dict'
        >>> print(f"\\n从 DataFrame 转换 (orient='dict'):\\n{dict_from_df_default}")
        从 DataFrame 转换 (orient='dict'):
        {'col1': {'row1': 1, 'row2': 2}, 'col2': {'row1': 3, 'row2': 4}}

        >>> dict_from_df_list = to_dict(df, orient='list')
        >>> print(f"\\n从 DataFrame 转换 (orient='list'):\\n{dict_from_df_list}")
        从 DataFrame 转换 (orient='list'):
        {'col1': [1, 2], 'col2': [3, 4]}

        >>> dict_from_df_records = to_dict(df, orient='records')
        >>> print(f"\\n从 DataFrame 转换 (orient='records'):\\n{dict_from_df_records}")
        从 DataFrame 转换 (orient='records'):
        [{'col1': 1, 'col2': 3}, {'col1': 2, 'col2': 4}]
    """
    arg = to_pd_array(arg)
    if orient == 'index_series':
        return {arg.index[i]: arg.iloc[i] for i in range(len(arg.index))}
    return arg.to_dict(orient)


def repeat(arg: tp.ArrayLike, n: int, axis: int = 1, raw: bool = False) -> tp.AnyArray:
    """
    沿指定轴重复 `arg` 中的每个元素 `n` 次。

    此函数基于 `np.repeat` 实现，并增加了对 Pandas 对象的支持，
    在重复数据的同时能够正确处理和生成新的索引/列名。

    参数:
        arg (tp.ArrayLike):
            输入的类数组对象。可以是 Python 列表、元组、NumPy 数组，
            或者 Pandas Series/DataFrame。
        n (int):
            每个元素重复的次数。必须是非负整数。
        axis (int, optional):
            执行重复操作的轴。默认为 `1` (按列重复)。
            - `axis = 0`: 沿行方向重复。对于 DataFrame，会重复每一行；
                          对于 Series 或一维数组，会扩展其长度。
            - `axis = 1`: 沿列方向重复。如果输入是一维的，会先将其转换为二维；
                          对于 DataFrame，会重复每一列。
            目前仅支持 `0` 和 `1`。
        raw (bool, optional):
            一个布尔标志，控制输出类型，默认为 `False`。
            - 如果 `raw` 为 `False` (默认) 且输入是 Pandas 对象，则返回 Pandas 对象，
              并相应地调整索引/列名。
            - 如果 `raw` 为 `True` 或输入不是 Pandas 对象，则返回 NumPy 数组。
            此参数主要影响 `to_any_array` 的行为以及最终返回类型。

    返回:
        tp.AnyArray:
            重复后的数组。如果原始输入是 Pandas 对象且 `raw=False`，
            则返回带有新索引/列的 Pandas 对象；否则返回 NumPy 数组。

    示例:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbt.base.reshape_fns import repeat

        >>> # 示例 1: 重复 NumPy 一维数组的元素
        >>> arr_1d = np.array([1, 2])
        >>> repeated_arr_axis0 = repeat(arr_1d, n=3, axis=0) # axis=0 对1D数组作用是扩展
        >>> print(f"NumPy 1D 沿 axis=0 重复:\\n{repeated_arr_axis0}")
        NumPy 1D 沿 axis=0 重复:
        [1 1 1 2 2 2]

        >>> # 注意：对于1D数组，axis=1 会先将其转为2D (N,1)，再按列重复
        >>> repeated_arr_axis1 = repeat(arr_1d, n=2, axis=1)
        >>> print(f"\\nNumPy 1D 沿 axis=1 重复 (隐式转2D):\\n{repeated_arr_axis1}")
        NumPy 1D 沿 axis=1 重复 (隐式转2D):
        [[1 1]
         [2 2]]

        >>> # 示例 2: 重复 Pandas Series 的元素
        >>> series = pd.Series([10, 20], index=['a', 'b'], name='S')
        >>> repeated_series_axis0 = repeat(series, n=2, axis=0)
        >>> print(f"\\nPandas Series 沿 axis=0 重复:\\n{repeated_series_axis0}")
        Pandas Series 沿 axis=0 重复:
        a    10
        a    10
        b    20
        b    20
        Name: S, dtype: int64

        >>> # 示例 3: 重复 Pandas DataFrame 的元素
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
        >>> print(f"\\n原始 DataFrame:\\n{df}")
        原始 DataFrame:
           A  B
        x  1  3
        y  2  4

        >>> repeated_df_axis0 = repeat(df, n=2, axis=0) # 按行重复
        >>> print(f"\\nDataFrame 沿 axis=0 重复:\\n{repeated_df_axis0}")
        DataFrame 沿 axis=0 重复:
           A  B
        x  1  3
        x  1  3
        y  2  4
        y  2  4

        >>> repeated_df_axis1 = repeat(df, n=2, axis=1) # 按列重复
        >>> print(f"\\nDataFrame 沿 axis=1 重复:\\n{repeated_df_axis1}")
        DataFrame 沿 axis=1 重复:
           A  A  B  B
        x  1  1  3  3
        y  2  2  4  4

        >>> # 示例 4: 使用 raw=True
        >>> repeated_df_axis1_raw = repeat(df, n=2, axis=1, raw=True)
        >>> print(f"\\nDataFrame 沿 axis=1 重复 (raw=True):\\n{repeated_df_axis1_raw}")
        DataFrame 沿 axis=1 重复 (raw=True):
        [[1 1 3 3]
         [2 2 4 4]]
    """
    arg = to_any_array(arg, raw=raw)
    if axis == 0:
        # 检查处理后的 arg 是否为 Pandas 对象 (Series 或 DataFrame)
        if checks.is_pandas(arg):
            # 如果是 Pandas 对象，则使用 vectorbt 的 ArrayWrapper 来处理，以保留和更新索引。
            # 1. ArrayWrapper.from_obj(arg): 从 Pandas 对象创建一个 ArrayWrapper 实例。
            # 2. np.repeat(arg.values, n, axis=0): 对 Pandas 对象的底层 NumPy 数据执行元素重复。
            # 3. index=index_fns.repeat_index(arg.index, n): 为重复后的数据生成新的 Pandas 索引。
            # 4. .wrap(...): 使用新的数据和索引包装回原始 Pandas 对象类型。
            return array_wrapper.ArrayWrapper.from_obj(arg).wrap(
                np.repeat(arg.values, n, axis=0), index=index_fns.repeat_index(arg.index, n))
        # 如果 arg 不是 Pandas 对象 (即它是一个 NumPy 数组)
        # 直接使用 NumPy 的 repeat 函数沿 axis 0 重复元素。
        return np.repeat(arg, n, axis=0)
    elif axis == 1:
        arg = to_2d(arg)
        if checks.is_pandas(arg):
            return array_wrapper.ArrayWrapper.from_obj(arg).wrap(
                np.repeat(arg.values, n, axis=1), columns=index_fns.repeat_index(arg.columns, n))
        return np.repeat(arg, n, axis=1)
    else:
        raise ValueError("Only axis 0 and 1 are supported")


def tile(arg: tp.ArrayLike, n: int, axis: int = 1, raw: bool = False) -> tp.AnyArray:
    """
    沿指定轴将整个 `arg` 重复 `n` 次。

    此函数与 `repeat` 不同，`repeat` 是重复数组中的 *每个元素*，
    而 `tile` (平铺) 是重复 *整个数组块*。
    它基于 `np.tile` 实现，并增加了对 Pandas 对象的支持，
    在平铺数据的同时能够正确处理和生成新的索引/列名。

    参数:
        arg (tp.ArrayLike):
            输入的类数组对象。可以是 Python 列表、元组、NumPy 数组，
            或者 Pandas Series/DataFrame。
        n (int):
            整个数组沿指定轴重复的次数。必须是非负整数。
        axis (int, optional):
            执行平铺操作的轴。默认为 `1` (沿列方向平铺)。
            - `axis = 0`: 沿行方向平铺。对于 DataFrame 或二维数组，会将整个数组块
                          在垂直方向上堆叠 `n` 次。对于 Series 或一维数组，
                          会先将其视为列向量 (N,1)，然后垂直堆叠，或者如果 `raw=True`
                          且保持一维，则将整个序列重复 `n` 次。
            - `axis = 1`: 沿列方向平铺。如果输入是一维的，会先将其转换为二维
                          (通常是列向量 (N,1))；对于 DataFrame 或二维数组，
                          会将整个数组块在水平方向上并排 `n` 次。
            目前仅支持 `0` 和 `1`。
        raw (bool, optional):
            一个布尔标志，控制输出类型，默认为 `False`。
            - 如果 `raw` 为 `False` (默认) 且输入是 Pandas 对象，则返回 Pandas 对象，
              并相应地调整索引/列名。
            - 如果 `raw` 为 `True` 或输入不是 Pandas 对象，则返回 NumPy 数组。
            此参数主要影响 `to_any_array` 和 `to_2d` 的行为以及最终返回类型。

    返回:
        tp.AnyArray:
            平铺后的数组。如果原始输入是 Pandas 对象且 `raw=False`，
            则返回带有新索引/列的 Pandas 对象；否则返回 NumPy 数组。

    示例:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbt.base.reshape_fns import tile

        >>> # 示例 1: 平铺 NumPy 一维数组
        >>> arr_1d = np.array([1, 2])
        >>> tiled_arr_axis0 = tile(arr_1d, n=3, axis=0) # 沿 axis=0 平铺
        >>> print(f"NumPy 1D 沿 axis=0 平铺:\\n{tiled_arr_axis0}")
        NumPy 1D 沿 axis=0 平铺:
        [1 2 1 2 1 2]

        >>> # 对于1D数组，axis=1 会先将其转为2D (N,1)，再沿列平铺
        >>> tiled_arr_axis1 = tile(arr_1d, n=2, axis=1)
        >>> print(f"\\nNumPy 1D 沿 axis=1 平铺 (隐式转2D):\\n{tiled_arr_axis1}")
        NumPy 1D 沿 axis=1 平铺 (隐式转2D):
        [[1 1]
         [2 2]]

        >>> # 示例 2: 平铺 Pandas Series
        >>> series = pd.Series([10, 20], index=['a', 'b'], name='S')
        >>> tiled_series_axis0 = tile(series, n=2, axis=0)
        >>> print(f"\\nPandas Series 沿 axis=0 平铺:\\n{tiled_series_axis0}")
        Pandas Series 沿 axis=0 平铺:
        a    10
        b    20
        a    10
        b    20
        Name: S, dtype: int64

        >>> # 示例 3: 平铺 Pandas DataFrame
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
        >>> print(f"\\n原始 DataFrame:\\n{df}")
        原始 DataFrame:
           A  B
        x  1  3
        y  2  4

        >>> tiled_df_axis0 = tile(df, n=2, axis=0) # 沿行平铺
        >>> print(f"\\nDataFrame 沿 axis=0 平铺:\\n{tiled_df_axis0}")
        DataFrame 沿 axis=0 平铺:
           A  B
        x  1  3
        y  2  4
        x  1  3
        y  2  4

        >>> tiled_df_axis1 = tile(df, n=2, axis=1) # 沿列平铺
        >>> print(f"\\nDataFrame 沿 axis=1 平铺:\\n{tiled_df_axis1}")
        DataFrame 沿 axis=1 平铺:
           A  B  A  B
        x  1  3  1  3
        y  2  4  2  4

        >>> # 示例 4: 使用 raw=True
        >>> tiled_df_axis1_raw = tile(df, n=2, axis=1, raw=True)
        >>> print(f"\\nDataFrame 沿 axis=1 平铺 (raw=True):\\n{tiled_df_axis1_raw}")
        DataFrame 沿 axis=1 平铺 (raw=True):
        [[1 3 1 3]
         [2 4 2 4]]
    """
    arg = to_any_array(arg, raw=raw)
    if axis == 0:
        if arg.ndim == 2:
            if checks.is_pandas(arg):
                return array_wrapper.ArrayWrapper.from_obj(arg).wrap(
                    np.tile(arg.values, (n, 1)), index=index_fns.tile_index(arg.index, n))
            return np.tile(arg, (n, 1))
        if checks.is_pandas(arg):
            return array_wrapper.ArrayWrapper.from_obj(arg).wrap(
                np.tile(arg.values, n), index=index_fns.tile_index(arg.index, n))
        return np.tile(arg, n)
    elif axis == 1:
        arg = to_2d(arg)
        if checks.is_pandas(arg):
            return array_wrapper.ArrayWrapper.from_obj(arg).wrap(
                np.tile(arg.values, (1, n)), columns=index_fns.tile_index(arg.columns, n))
        return np.tile(arg, (1, n))
    else:
        raise ValueError("Only axis 0 and 1 are supported")


IndexFromLike = tp.Union[None, str, int, tp.Any]
"""Any object that can be coerced into a `index_from` argument."""


def broadcast_index(args: tp.Sequence[tp.AnyArray],
                    to_shape: tp.Shape,
                    index_from: IndexFromLike = None,
                    axis: int = 0,
                    ignore_sr_names: tp.Optional[bool] = None,
                    **kwargs) -> tp.Optional[tp.Index]:
    """
    根据给定的广播规则，为一组数组/pandas对象生成广播后的 Pandas 索引 (pd.Index) 或列名 (pd.Index)。

    此函数是对数组进行广播操作时，确定结果对象索引/列名的核心逻辑。在 NumPy 的广播操作中，
    数组的值会按规则扩展，但对于 Pandas 对象，还需要同时处理索引或列名的扩展方式。
    该函数提供了多种策略来处理不同情况下的索引/列名广播。

    参数:
        args (tp.Sequence[tp.AnyArray]): 
            要广播的数组或 Pandas 对象的序列。这些对象的索引或列将根据 `index_from` 参数用于创建结果索引。
        to_shape (tp.Shape): 
            目标形状，即广播操作的目标尺寸。这是一个形如 (n,) 或 (n, m) 的元组。
            函数会使用此形状确定结果索引的长度。
        index_from (IndexFromLike, optional): 
            广播规则，指定如何创建新的索引/列名。默认为 None。接受以下值：
            * None 或 'keep' - 保留原始对象的索引/列名 (返回 None，告知调用者保留原索引)
            * 'stack' - 使用 stack_indexes 函数堆叠不同的索引/列名
            * 'strict' - 确保所有 pandas 对象具有相同的索引/列名，如果不同则引发错误
            * 'reset' - 重置所有索引/列名（变为简单的 RangeIndex）
            * 整数 - 使用 args 中对应位置对象的索引/列名
            * 其他任何对象 - 将被转换为 pd.Index 作为新的索引/列名
        axis (int, optional): 
            指定是处理行索引还是列名。默认为 0。
            * 0 表示处理行索引
            * 1 表示处理列名
        ignore_sr_names (bool, optional): 
            是否忽略 Series 对象名称与其他 Series 对象名称之间的冲突。默认为 None（使用配置默认值）。
            * 当为 True 且处理列名 (axis=1) 时，会忽略 Series 的名称
            * 冲突的 Series 名称是指不同但都不为 None 的名称
        **kwargs: 
            传递给 vectorbt.base.index_fns.stack_indexes 函数的关键字参数。
            可用于控制如何堆叠索引创建多级索引的细节。

    返回:
        tp.Optional[tp.Index]: 
            广播后的 Pandas 索引对象，如果 index_from 为 None 或 'keep'，则返回 None。

    示例:

        ```python
        import numpy as np
        import pandas as pd
        from vectorbt.base.reshape_fns import broadcast_index

        # 创建测试数据
        # 不同索引的Series
        s1 = pd.Series([1, 2], index=['a', 'b'], name='S1')
        s2 = pd.Series([3, 4, 5], index=['x', 'y', 'z'], name='S2')
        # 带自定义列名的DataFrame
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['p', 'q'])
        # NumPy数组
        arr = np.array([[1, 2], [3, 4]])
        
        # 示例1: index_from=int - 使用指定位置对象的索引
        # 目标形状(2,3)表示2行3列
        result1 = broadcast_index([s1, s2, df], to_shape=(2, 3), index_from=0, axis=0)
        print("示例1 - 使用s1的索引 (index_from=0, axis=0):")
        print(result1)
        # 输出: Index(['a', 'b'], dtype='object')

        # 示例2: index_from='reset' - 创建新的RangeIndex
        result2 = broadcast_index([s1, s2, df], to_shape=(2, 3), index_from='reset', axis=0)
        print("\n示例2 - 重置索引 (index_from='reset', axis=0):")
        print(result2)
        # 输出: RangeIndex(start=0, stop=2, step=1)

        # 示例3: index_from='stack' - 堆叠不同的索引
        # 为了便于演示，我们使用两个具有不同索引但长度相同的Series
        s3 = pd.Series([1, 2], index=['m', 'n'], name='S3')
        result3 = broadcast_index([s1, s3], to_shape=(2, 2), index_from='stack', axis=0)
        print("\n示例3 - 堆叠索引 (index_from='stack', axis=0):")
        print(result3)
        # 输出: MultiIndex([('a', 'm'), ('b', 'n')], )

        # 示例4: index_from='strict' - 确保所有对象具有相同的索引
        try:
            broadcast_index([s1, s2], to_shape=(2, 2), index_from='strict', axis=0)
        except ValueError as e:
            print("\n示例4 - 严格模式下的索引冲突错误 (index_from='strict', axis=0):")
            print(f"引发错误: {e}")
        
        # 示例5: 传递自定义索引
        custom_index = pd.Index(['custom1', 'custom2'])
        result5 = broadcast_index([s1, s2, arr], to_shape=(2, 3), index_from=custom_index, axis=0)
        print("\n示例5 - 使用自定义索引 (index_from=custom_index, axis=0):")
        print(result5)
        # 输出: Index(['custom1', 'custom2'], dtype='object')

        # 示例6: 处理列 (axis=1)
        result6 = broadcast_index([df], to_shape=(2, 3), index_from='reset', axis=1)
        print("\n示例6 - 重置列索引 (index_from='reset', axis=1):")
        print(result6)
        # 输出: RangeIndex(start=0, stop=3, step=1)

        # 示例7: Series.name作为列 (axis=1)
        # 当axis=1且处理Series时，Series的name被视为列名
        result7 = broadcast_index([s1, s2], to_shape=(2, 2), index_from='stack', axis=1, ignore_sr_names=False)
        print("\n示例7 - 将Series名称堆叠为列 (index_from='stack', axis=1, ignore_sr_names=False):")
        print(result7)
        # 输出: Index(['S1', 'S2'], dtype='object')

        # 示例8: 重复索引以匹配目标长度
        # 长度为1的索引可以被重复以匹配更长的目标形状
        s_short = pd.Series([1], index=['single'])
        result8 = broadcast_index([s_short], to_shape=(3, 1), index_from=0, axis=0)
        print("\n示例8 - 重复短索引 (index_from=0, axis=0, 目标形状=(3,1)):")
        print(result8)
        # 输出: Index(['single', 'single', 'single'], dtype='object')
        ```
    """
    from vectorbt._settings import settings
    broadcasting_cfg = settings['broadcasting']
    
    if ignore_sr_names is None:
        ignore_sr_names = broadcasting_cfg['ignore_sr_names']
        
    index_str = 'columns' if axis == 1 else 'index'

    to_shape_2d = (to_shape[0], 1) if len(to_shape) == 1 else to_shape
    maxlen = to_shape_2d[1] if axis == 1 else to_shape_2d[0]
    
    new_index = None

    # index_from == 'keep' 或 None：直接返回 None
    if index_from is None or (isinstance(index_from, str) and index_from.lower() == 'keep'):
        return None
    
    # 如果 index_from 是一个整数，表示使用 args[index_from]的index/columns
    if isinstance(index_from, int):
        # 如果不是pandas对象则没有index/columns，引发错误
        if not checks.is_pandas(args[index_from]):
            raise TypeError(f"Argument under index {index_from} must be a pandas object")
        new_index = index_fns.get_index(args[index_from], axis)
        
    # 如果 index_from 是一个字符串，表示特定的规则
    elif isinstance(index_from, str):
        # 'reset'：忽略原始索引，生成一个标准的 RangeIndex
        if index_from.lower() == 'reset':
            new_index = pd.RangeIndex(start=0, stop=maxlen, step=1)
        # 如果规则是 'stack' 或 'strict'
        elif index_from.lower() in ('stack', 'strict'):
            # 首先检查所有 Pandas 对象是否具有相同的索引/列
            last_index = None  
            index_conflict = False  # 标记是否存在索引冲突
            for arg in args:
                if checks.is_pandas(arg):
                    index = index_fns.get_index(arg, axis)
                    if last_index is not None:
                        if not checks.is_index_equal(index, last_index):
                            index_conflict = True
                    last_index = index
                    continue
            # 如果检查后没有发现索引冲突
            if not index_conflict:
                # 那么所有 Pandas 对象的索引/列都是相同的 (或者是 None，如果只有一个Pandas对象或没有)
                # 将 new_index 设置为最后记录的那个索引 (它们都一样)
                new_index = last_index
            # 如果存在索引冲突
            else:
                # 如果有索引冲突，但模式是 'stack' 或需要检查更严格的条件
                # 尝试构建一个新的复合索引
                for arg in args:
                    if checks.is_pandas(arg):
                        index = index_fns.get_index(arg, axis)
                        # 特殊处理 Series 的 name 作为列名的情况
                        if axis == 1 and checks.is_series(arg) and ignore_sr_names:
                            # 如果是处理列 (axis=1)，对象是 Series，并且设置了 ignore_sr_names，
                            # 则跳过这个 Series 的 name，不参与堆叠 (因为它被忽略了)
                            continue
                        # 如果索引是默认的 RangeIndex (例如 0, 1, 2, ...) 且没有名称，通常可以忽略
                        if checks.is_default_index(index):
                            continue
                        # 初始化 new_index 或与现有的 new_index 堆叠
                        if new_index is None:
                             # 如果这是第一个要堆叠的非默认索引，直接使用它
                            new_index = index
                        else:
                            # 如果这个索引与已有的新索引相同，则跳过
                            if checks.is_index_equal(index, new_index):
                                continue
                            # 如果模式是'strict'且索引不同，则引发错误
                            if index_from.lower() == 'strict':
                                # 严格模式下，不允许具有不同索引/列的对象进行广播
                                raise ValueError(
                                    f"Broadcasting {index_str} is not allowed when {index_str}_from=strict")
                            # 处理不同长度索引的广播规则
                            # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
                            # 1. 如果索引长度相同，直接堆叠
                            # 2. 如果其中一个索引只有一个元素，复制它以匹配另一个长度再堆叠

                            # 检查两个索引的长度是否不同
                            if len(index) != len(new_index):
                                # 如果两个索引都有多个元素，无法广播
                                if len(index) > 1 and len(new_index) > 1:
                                    raise ValueError("Indexes could not be broadcast together")
                                # 如果当前索引比已有的新索引长，复制新索引
                                if len(index) > len(new_index):
                                    new_index = index_fns.repeat_index(new_index, len(index))
                                # 如果当前索引比已有的新索引短，复制当前索引
                                elif len(index) < len(new_index):
                                    index = index_fns.repeat_index(index, len(new_index))
                            # 堆叠两个索引，创建 MultiIndex
                            # 注意: **kwargs 传递给 stack_indexes 函数，可以控制堆叠行为
                            new_index = index_fns.stack_indexes([new_index, index], **kwargs)
        # 无效的 index_from 字符串值
        else:
            raise ValueError(f"Invalid value '{index_from}' for {'columns' if axis == 1 else 'index'}_from")
    # index_from 为其他类型，直接将其作为索引
    else:
        new_index = index_from
        
    # 如果生成了新索引，检查其长度是否满足要求
    if new_index is not None:
        # 如果新索引长度小于 maxlen（广播后的目标大小）
        if maxlen > len(new_index):
            # 如果是严格模式，则不允许广播索引
            if isinstance(index_from, str) and index_from.lower() == 'strict':
                raise ValueError(f"Broadcasting {index_str} is not allowed when {index_str}_from=strict")
            # 这种情况通常发生在某些 NumPy 对象比新的 pandas 索引长的情况下
            # 在这种情况下，应重复 pandas 索引以匹配 NumPy 对象的长度
            
            # 如果两者都有多个元素且长度不同，无法广播
            if maxlen > 1 and len(new_index) > 1:
                raise ValueError("Indexes could not be broadcast together")
            # 重复索引以达到所需的长度
            new_index = index_fns.repeat_index(new_index, maxlen)
    else:
        # 如果 new_index 仍为 None，但 index_from 不是 None，则创建默认的 RangeIndex
        # 这种情况在 index_from 不是 None 的情况下会发生，即我们明确要生成新索引，而不是保留原索引
        new_index = pd.RangeIndex(start=0, stop=maxlen, step=1)
    return new_index


def wrap_broadcasted(old_arg: tp.AnyArray,
                     new_arg: tp.Array,
                     is_pd: bool = False,
                     new_index: tp.Optional[tp.Index] = None,
                     new_columns: tp.Optional[tp.Index] = None) -> tp.AnyArray:
    """
    将广播后的数组包装回适当的数据类型，并分配相应的索引和列名。
    
    此函数主要用于广播操作后的后处理步骤。当对数组进行广播操作时，结果通常是 NumPy 数组，
    但在许多情况下，我们希望保持原始数据的类型（特别是 Pandas 对象）和元数据（如索引和列名）。
    该函数负责将广播后的 NumPy 数组重新包装为适当的数据类型，并为其分配合适的索引和列名。
    
    参数:
        old_arg (tp.AnyArray): 
            原始输入参数，通常是一个 Pandas 对象（Series 或 DataFrame）或 NumPy 数组。
            如果这是一个 Pandas 对象，且没有提供新的索引new_index或列new_columns，会尝试从这个对象中获取并调整这些信息。
        
        new_arg (tp.Array): 
            经过广播操作后的 NumPy 数组。这通常是广播函数（如 numpy.broadcast_to）的输出结果。
            该数组包含了广播后的数据，但丢失了原始对象的类型和元数据信息。
        
        is_pd (bool, optional): 
            是否将结果包装为 Pandas 对象。默认为 False。
            - 如果为 True，将尝试根据 new_arg 的维度创建 Pandas Series 或 DataFrame。
            - 如果为 False，则直接返回 new_arg（NumPy 数组）不做额外处理。
        
        new_index (tp.Optional[tp.Index], optional): 
            用于结果对象的行索引。默认为 None。
            - 如果提供了该参数，将直接使用它作为结果的索引。
            - 如果为 None 且 old_arg 是 Pandas 对象，会尝试从 old_arg 中提取并适当调整索引。
        
        new_columns (tp.Optional[tp.Index], optional): 
            用于结果对象的列索引（仅适用于二维结果）。默认为 None。
            - 如果提供了该参数，将直接使用它作为结果的列名。
            - 如果为 None 且 old_arg 是 Pandas 对象，会尝试从 old_arg 中提取并适当调整列名。
            - 对于一维结果（Series），如果 new_columns 有恰好一个元素，它可能被用作 Series 的名称。
    
    返回:
        tp.AnyArray: 
            包装后的数组，可能是 Pandas Series、DataFrame 或原始的 NumPy 数组，取决于 is_pd 参数和输入数据的维度。
    
    示例:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbt.base.reshape_fns import wrap_broadcasted
        
        >>> # 示例 1: 广播后的 NumPy 数组重新包装为 Series
        >>> old_series = pd.Series([1, 2], index=['a', 'b'], name='data')
        >>> new_array = np.array([10, 20, 30])  # 假设这是广播后的结果
        >>> # 将 NumPy 数组包装为 Series，使用新的索引
        >>> new_series = wrap_broadcasted(
        ...     old_series, new_array, is_pd=True, new_index=['x', 'y', 'z'])
        >>> print(new_series)
        x    10
        y    20
        z    30
        Name: data, dtype: int64
        
        >>> # 示例 2: 广播后的 NumPy 数组重新包装为 DataFrame
        >>> old_df = pd.DataFrame({'A': [1, 2]}, index=['a', 'b'])
        >>> new_array_2d = np.array([[10, 20], [30, 40]])  # 假设这是广播后的结果
        >>> # 将 NumPy 数组包装为 DataFrame，使用新的列名
        >>> new_df = wrap_broadcasted(
        ...     old_df, new_array_2d, is_pd=True, new_columns=['X', 'Y'])
        >>> print(new_df)
           X   Y
        0  10  20
        1  30  40
    """
    # 检查是否需要将结果包装为 Pandas 对象
    if is_pd:
        # 如果原始参数是 Pandas 对象，尝试获取并调整其索引和列
        if checks.is_pandas(old_arg):
            # 如果没有提供新的行索引
            if new_index is None:
                # 从原始 Pandas 对象中提取行索引
                old_index = index_fns.get_index(old_arg, 0)
                # 如果原始对象和新数组行数相同，直接使用原索引
                if old_arg.shape[0] == new_arg.shape[0]:
                    new_index = old_index
                else:
                    # 否则，重复原索引以匹配新数组的行数
                    new_index = index_fns.repeat_index(old_index, new_arg.shape[0])
            
            # 如果没有提供新的列索引
            if new_columns is None:
                # 从原始 Pandas 对象中提取列索引
                old_columns = index_fns.get_index(old_arg, 1)
                # 计算新数组的列数（如果是二维）或设为1（如果是一维）
                new_ncols = new_arg.shape[1] if new_arg.ndim == 2 else 1
                # 如果原始列数与新列数相同，直接使用原列索引
                if len(old_columns) == new_ncols:
                    new_columns = old_columns
                else:
                    # 否则，重复原列索引以匹配新数组的列数
                    new_columns = index_fns.repeat_index(old_columns, new_ncols)
        
        # 根据新数组的维度决定返回 DataFrame 还是 Series
        if new_arg.ndim == 2:
            # 对于二维数组，返回 DataFrame
            return pd.DataFrame(new_arg, index=new_index, columns=new_columns)
        
        # 处理一维数组的 Series 名称
        # 如果有列索引且只有一个元素，使用它作为 Series 的名称
        if new_columns is not None and len(new_columns) == 1:
            name = new_columns[0]
            # 如果名称是数字0，将其设置为 None（避免使用默认自动索引作为名称）
            if name == 0:
                name = None
        else:
            # 如果没有列索引或列索引不是单一元素，名称设为 None
            name = None
        
        # 返回一维 Series，使用提取的索引和名称
        return pd.Series(new_arg, index=new_index, name=name)
    
    # 如果不需要包装为 Pandas 对象，直接返回 NumPy 数组
    return new_arg


# 定义 BCRT (Broadcast Return Type) 类型：两种可能的返回类型：
# 1. 一个可能的数组元组
# 2. 一个包含四个元素的元组:
#    - 可能的数组元组
#    - 形状信息 (Shape)
#    - 可选的索引对象 (Index)
#    - 可选的列索引对象 (Index)
BCRT = tp.Union[
    tp.MaybeTuple[tp.AnyArray],  # 当 return_meta=False 时返回的类型
    tp.Tuple[tp.MaybeTuple[tp.AnyArray], tp.Shape, tp.Optional[tp.Index], tp.Optional[tp.Index]]  # 当 return_meta=True 时返回的类型
]


def broadcast(*args: tp.ArrayLike,
              to_shape: tp.Optional[tp.RelaxedShape] = None,
              to_pd: tp.Optional[tp.MaybeSequence[bool]] = None,
              to_frame: tp.Optional[bool] = None,
              align_index: tp.Optional[bool] = None,
              align_columns: tp.Optional[bool] = None,
              index_from: tp.Optional[IndexFromLike] = None,
              columns_from: tp.Optional[IndexFromLike] = None,
              require_kwargs: tp.KwargsLikeSequence = None,
              keep_raw: tp.Optional[tp.MaybeSequence[bool]] = False,
              return_meta: bool = False,
              **kwargs) -> BCRT:
    """
    通过使用 NumPy 广播机制，将 `args` 中的任何数组/类数组对象广播到相同的形状。
    
    广播是一种强大的机制，允许不同形状的数组在算术运算中一起使用。此函数扩展了 NumPy 的广播规则，
    不仅处理 NumPy 数组，还能处理 Pandas 对象，保留它们的索引和列名信息。
    
    广播规则遵循 NumPy 的标准规则 (https://numpy.org/doc/stable/user/basics.broadcasting.html)：
    1. 如果数组的维度不同，形状较小的数组会在前面添加 1 直到维度相同
    2. 如果两个数组在某个维度上的大小相同，或其中一个大小为 1，则它们兼容
    3. 如果两个数组在某个维度上都不是 1，且大小不同，则无法广播
    
    参数:
        *args (tp.ArrayLike): 
            要广播的数组/类数组对象。可以是 NumPy 数组、Pandas Series/DataFrame、Python 列表等。
            
        to_shape (tp.Optional[tp.RelaxedShape], optional): 
            目标形状。如果设置，将把 `args` 中的每个元素广播到该形状。
            可以是整数（一维形状）或整数元组（多维形状）。默认为 None，表示自动确定广播后的形状。
            
        to_pd (tp.Optional[tp.MaybeSequence[bool]], optional): 
            是否将所有输出数组转换为 Pandas 对象。
            - 如果为 None (默认)：当 `args` 中至少有一个 Pandas 对象时才转换
            - 如果为 True：强制所有结果转换为 Pandas 对象
            - 如果为 False：返回原始 NumPy 数组
            - 如果是布尔值序列：分别应用到每个参数
            
        to_frame (tp.Optional[bool], optional): 
            是否将所有 Series 转换为 DataFrame。
            - 如果为 None (默认)：根据上下文自动决定
            - 如果为 True：强制将 Series 转换为 DataFrame
            - 如果为 False：保持 Series 不变
            
        align_index (tp.Optional[bool], optional): 
            是否通过多重索引对齐 Pandas 对象的索引。
            - 如果为 None：使用配置默认值
            - 如果为 True：尝试对齐所有索引
            - 如果为 False：不进行索引对齐
            
        align_columns (tp.Optional[bool], optional): 
            是否通过多重索引对齐 Pandas 对象的列。
            - 如果为 None：使用配置默认值
            - 如果为 True：尝试对齐所有列
            - 如果为 False：不进行列对齐
            
        index_from (tp.Optional[IndexFromLike], optional): 
            指定结果索引的来源规则。
            - 如果为 None：使用配置默认值
            - 常见值包括：'keep'（保留原索引）、'stack'（堆叠索引）、
              'strict'（确保索引一致）、'reset'（重置索引）、
              整数（使用第 n 个参数的索引）
            
        columns_from (tp.Optional[IndexFromLike], optional): 
            指定结果列的来源规则。
            - 如果为 None：使用配置默认值
            - 规则同 index_from
            
        require_kwargs (tp.KwargsLikeSequence, optional): 
            传递给 `np.require` 的关键字参数。用于指定结果数组的要求（如类型、内存排列等）。
            - 如果是字典：应用于所有参数
            - 如果是字典序列：分别应用到每个参数
            
        keep_raw (tp.Optional[tp.MaybeSequence[bool]], optional): 
            是否保持未广播版本的数组。默认为 False。
            如果为 True，则仅确保数组可以广播到目标形状，但不实际执行广播。
            - 如果是布尔值序列：分别应用到每个参数
            
        return_meta (bool, optional): 
            是否也返回新形状、索引和列信息。默认为 False。
            - 如果为 True：返回一个四元组 (结果数组, 形状, 索引, 列)
            - 如果为 False：仅返回结果数组
            
        **kwargs: 
            传递给 `broadcast_index` (处理索引广播的函数) 的其他关键字参数。
    
    返回:
        BCRT: 
            如果 return_meta=False (默认)：
                - 如果只有一个参数：返回单个广播后的数组
                - 如果有多个参数：返回广播后数组的元组
            如果 return_meta=True：
                - 返回一个四元组 (广播后的数组或元组, 新形状, 新索引, 新列)
    """
    from vectorbt._settings import settings
    broadcasting_cfg = settings['broadcasting']

    is_pd = False  # 标记是否转成pandas对象(Series/DataFrame)
    is_2d = False  # 标记Series是否转成DataFrame
    
    if require_kwargs is None:
        require_kwargs = {}
    
    # 对于pandas对象，是否对齐index或columns（统一长度）
    if align_index is None:
        align_index = broadcasting_cfg['align_index']
    if align_columns is None:
        align_columns = broadcasting_cfg['align_columns']
    
    # 对于pandas对象，对齐后如何设置最后的index或columns
    if index_from is None:
        index_from = broadcasting_cfg['index_from']
    if columns_from is None:
        columns_from = broadcasting_cfg['columns_from']

    # 将输入参数转换为 NumPy 数组或 Pandas 对象，放入 arr_args
    # 同时标记is_2d和is_pd
    arr_args = []
    for i in range(len(args)):
        arg = to_any_array(args[i])
        if arg.ndim > 1:
            is_2d = True
        if checks.is_pandas(arg):
            is_pd = True
        arr_args.append(arg)

    if to_shape is not None:
        if isinstance(to_shape, int):
            to_shape = (to_shape,)
        checks.assert_instance_of(to_shape, tuple)
        if len(to_shape) > 1:
            is_2d = True

    if to_frame is not None:
        is_2d = to_frame

    if to_pd is not None:
        if isinstance(to_pd, Sequence):
            is_pd = any(to_pd)
        else:
            is_pd = to_pd

    # 把 arr_args 中各pandas对象按照index对齐
    if align_index:
        # 被对齐的pandas对象：其原来每行仍存在，但是顺序可能打乱，数量也可能变多
        # 对齐的pandas对象是arr_args中index长度最大的（可能有多个）
        # 对齐后arr_args中的所有pandas对象的index长度相同
        index_to_align = []
        for i in range(len(arr_args)):
            if checks.is_pandas(arr_args[i]) and len(arr_args[i].index) > 1:
                index_to_align.append(i) 
        if len(index_to_align) > 1:
            indexes = [arr_args[i].index for i in index_to_align]  
            if len(set(map(len, indexes))) > 1:
                index_indices = index_fns.align_indexes(indexes)  
                for i in index_to_align:
                    arr_args[i] = arr_args[i].iloc[index_indices[index_to_align.index(i)]]
                    
    # 把 arr_args 中各DataFrame对象按照columns对齐
    if align_columns:
        cols_to_align = []
        for i in range(len(arr_args)):
            if checks.is_frame(arr_args[i]) and len(arr_args[i].columns) > 1:
                cols_to_align.append(i)
        if len(cols_to_align) > 1:
            indexes = [arr_args[i].columns for i in cols_to_align]
            if len(set(map(len, indexes))) > 1:
                col_indices = index_fns.align_indexes(indexes)
                for i in cols_to_align:
                    arr_args[i] = arr_args[i].iloc[:, col_indices[cols_to_align.index(i)]]

    # is_2d为真，将 arr_args中的所有Series转换为DataFrame，得到arr_args_2d
    arr_args_2d = [arg.to_frame() if is_2d and checks.is_series(arg) else arg for arg in arr_args]

    # 如果未指定to_shape，计算arr_args_2d的广播shape（对其中所有的元素使用广播规则）
    if to_shape is None:
        if broadcast_shapes is not None:
            to_shape = broadcast_shapes(*map(lambda x: np.asarray(x).shape, arr_args_2d))
        else:
            to_shape = broadcast_shape(*map(np.asarray, arr_args_2d))

    # 对于arr_args_2d中的每一项，根据keep_raw决定保持不变，还是转成numpy.ndarray后广播到to_shape，然后放入new_args
    new_args = []
    for i, arg in enumerate(arr_args_2d):
        if isinstance(keep_raw, Sequence):
            _keep_raw = keep_raw[i]
        else:
            _keep_raw = keep_raw
        # 会将arg转成numpy.ndarray，然后广播到to_shape
        bc_arg = np.broadcast_to(arg, to_shape)
        if _keep_raw:
            new_args.append(arg)
            continue
        new_args.append(bc_arg)

    # 根据require_kwargs，决定arr_args_2d中的每一项是否应用np.require（如类型要求、内存排列等）
    for i in range(len(new_args)):
        _require_kwargs = resolve_dict(require_kwargs, i=i)
        new_args[i] = np.require(new_args[i], **_require_kwargs)

    # is_pd为真，生成pandas对象最后统一的index和columns
    if is_pd:
        new_index = broadcast_index(arr_args, to_shape, index_from=index_from, axis=0, **kwargs)
        new_columns = broadcast_index(arr_args, to_shape, index_from=columns_from, axis=1, **kwargs)
    else:
        new_index, new_columns = None, None

    # 将new_args中被转成numpy.ndarray的原pandas对象恢复为原始类型
    for i in range(len(new_args)):
        if isinstance(keep_raw, Sequence):
            _keep_raw = keep_raw[i]
        else:
            _keep_raw = keep_raw
        if _keep_raw:
            continue

        if isinstance(to_pd, Sequence):
            _is_pd = to_pd[i]
        else:
            _is_pd = is_pd

        new_args[i] = wrap_broadcasted(
            arr_args[i],  # 原始参数
            new_args[i],  # 广播后的数组
            is_pd=_is_pd,  # 是否转换为 Pandas
            new_index=new_index,  # 新索引
            new_columns=new_columns  # 新列
        )

    # 根据参数数量和 return_meta 参数决定返回值
    if len(new_args) > 1:
        if return_meta:
            return tuple(new_args), to_shape, new_index, new_columns
        return tuple(new_args)
    
    if return_meta:
        return new_args[0], to_shape, new_index, new_columns
    return new_args[0]


def broadcast_to(arg1: tp.ArrayLike,
                 arg2: tp.ArrayLike,
                 to_pd: tp.Optional[bool] = None,
                 index_from: tp.Optional[IndexFromLike] = None,
                 columns_from: tp.Optional[IndexFromLike] = None,
                 **kwargs) -> BCRT:
    """
    将 `arg1` 广播（broadcast）到 `arg2` 的形状，并可选择保留/恢复为 pandas 对象。

    该函数是 `broadcast` 的简化接口，常用于将一个标量、数组、Series、DataFrame等“对齐”到另一个对象的形状和索引/列。
    广播时会自动处理 numpy/pandas 类型的转换，并可根据参数决定是否保留 pandas 的 index/columns 信息。

    参数说明：
        arg1 (tp.ArrayLike):
            需要被广播的对象。可以是标量、list、np.ndarray、pd.Series、pd.DataFrame 等。
        arg2 (tp.ArrayLike):
            目标对象。其 shape、index、columns 会作为广播的目标。
        to_pd (Optional[bool], 默认 None):
            是否将输出转换为 pandas 对象（Series/DataFrame）。
            - None：自动判断（如果 arg2 是 pandas，则输出也是 pandas，否则为 numpy）。
            - True：强制输出为 pandas。
            - False：强制输出为 numpy。
        index_from (Optional[IndexFromLike], 默认 None):
            指定广播后 index 的来源。
            - None：自动使用 arg2 的 index。
            - 其他值：详见 vectorbt.base.reshape_fns.broadcast 的 index_from 说明。
        columns_from (Optional[IndexFromLike], 默认 None):
            指定广播后 columns 的来源。
            - None：自动使用 arg2 的 columns。
            - 其他值：详见 vectorbt.base.reshape_fns.broadcast 的 columns_from 说明。
        **kwargs:
            其他关键字参数，传递给底层的 broadcast 函数。

    返回值：
        BCRT:
            广播后的对象。类型取决于 to_pd 参数和 arg2 的类型，可能是 numpy.ndarray、pd.Series 或 pd.DataFrame。

    使用示例：
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbt.base.reshape_fns import broadcast_to

        >>> # 示例1：将一维 numpy 数组广播到 Series 的 index
        >>> a = np.array([1, 2, 3])
        >>> sr = pd.Series([4, 5, 6], index=pd.Index(['x', 'y', 'z']), name='a')
        >>> broadcast_to(a, sr)
        x    1
        y    2
        z    3
        Name: a, dtype: int64

        >>> # 示例2：将 Series 广播到 numpy 数组
        >>> broadcast_to(sr, a)
        array([4, 5, 6])

        >>> # 示例3：强制输出为 DataFrame
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        >>> broadcast_to(0, df, to_pd=True)
           A  B
        0  0  0
        1  0  0
    """
    arg1 = to_any_array(arg1)
    arg2 = to_any_array(arg2)
    if to_pd is None:
        to_pd = checks.is_pandas(arg2)
    if to_pd:
        if index_from is None:
            index_from = index_fns.get_index(arg2, 0)
        if columns_from is None:
            columns_from = index_fns.get_index(arg2, 1)
    return broadcast(arg1, to_shape=arg2.shape, to_pd=to_pd, index_from=index_from, columns_from=columns_from, **kwargs)


def broadcast_to_array_of(arg1: tp.ArrayLike, arg2: tp.ArrayLike) -> tp.Array:
    """
    【函数说明】
    将一维 `arg1` 广播到形状为 `(arg1.shape[0], *arg2.shape)` 的多维数组。
    注意：如果 arg1 是多维的，且后面的维度与 arg2 一致，则直接返回 arg1。

    【参数说明】
    arg1 : tp.ArrayLike
        需要被广播的对象。可以是标量、1维数组，或比 arg2 多1个维度的数组。
        - 如果是标量或1维数组，将被扩展为第一个维度与参数数量相同，其余维度与 arg2 相同。
        - 如果已经是 (N, *arg2.shape) 形状，则直接返回。
    arg2 : tp.ArrayLike
        目标数组。其 shape 用于决定广播的目标形状。

    【返回值】
    tp.Array
        广播后的 numpy.ndarray，形状为 (N, *arg2.shape)，N为arg1的长度或1。

    【使用示例】
    >>> import numpy as np
    >>> from vectorbt.base.reshape_fns import broadcast_to_array_of
    >>> # 示例1：arg1为一维数组，arg2为二维
    >>> broadcast_to_array_of([0.1, 0.2], np.empty((2, 2)))
    array([[[0.1, 0.1],
            [0.1, 0.1]],
           [[0.2, 0.2],
            [0.2, 0.2]]])
    >>> # 示例2：arg1为标量
    >>> broadcast_to_array_of(5, np.empty((3, 4)))
    array([[[5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5]]])
    >>> # 示例3：arg1已经是目标形状
    >>> arr = np.arange(6).reshape(2, 3)
    >>> broadcast_to_array_of(arr, np.empty((3,)))
    array([[0, 1, 2],
           [3, 4, 5]])
    """
    arg1 = np.asarray(arg1)
    arg2 = np.asarray(arg2)
    
    # 如果 arg1 的维度比 arg2 多1，并且后面的各维与 arg2 完全一致，说明已经是目标形状，直接返回
    if arg1.ndim == arg2.ndim + 1:
        if arg1.shape[1:] == arg2.shape:
            return arg1
        
    # 如果 arg1 是标量（0维），先转为1维数组，方便后续扩展
    if arg1.ndim == 0:
        arg1 = to_1d(arg1)
        
    checks.assert_ndim(arg1, 1)
    
    # 如果 arg2 是标量（0维），直接返回 arg1（无需扩展）
    if arg2.ndim == 0:
        return arg1
    
    for i in range(arg2.ndim):
        arg1 = np.expand_dims(arg1, axis=-1)
        
    return np.tile(arg1, (1, *arg2.shape))


def broadcast_to_axis_of(arg1: tp.ArrayLike, arg2: tp.ArrayLike, axis: int,
                         require_kwargs: tp.KwargsLike = None) -> tp.Array:
    """
    将数组 `arg1` 广播到目标数组 `arg2` 在指定轴 `axis` 上的大小。
    
    此函数用于将一个数组（通常是一维的）广播到另一个数组特定轴的维度大小，
    常用于参数化操作，如为每个时间点设置不同参数。
    
    参数:
        arg1: tp.ArrayLike
            需要被广播的输入数组。可以是标量、列表、numpy数组或pandas对象。
            此数组将被广播到与 `arg2` 在 `axis` 轴上相同的大小。
        
        arg2: tp.ArrayLike
            目标数组。其在 `axis` 轴上的大小将决定 `arg1` 被广播的目标大小。
        
        axis: int
            指定 `arg2` 的哪个轴作为广播目标。例如，如果 `axis=0`，则 `arg1` 
            将被广播到与 `arg2.shape[0]` 相同的大小。
        
        require_kwargs: tp.KwargsLike, 可选
            传递给 `np.require` 的关键字参数，用于控制输出数组的类型和内存布局。
            例如 {'dtype': np.float64} 可以强制输出为浮点型。
    
    返回:
        tp.Array
            广播后的数组，大小与 `arg2` 在 `axis` 轴上的大小相同。
    
    示例:
        >>> import numpy as np
        >>> from vectorbt.base.reshape_fns import broadcast_to_axis_of
        
        >>> # 示例1：广播标量到一维数组的长度
        >>> arg1 = 5
        >>> arg2 = np.array([1, 2, 3, 4])
        >>> broadcast_to_axis_of(arg1, arg2, axis=0)
        array([5, 5, 5, 5])
        
        >>> # 示例2：广播一维数组到二维数组的行数
        >>> arg1 = [10, 20]
        >>> arg2 = np.array([[1, 2, 3], [4, 5, 6]])
        >>> broadcast_to_axis_of(arg1, arg2, axis=0)
        array([10, 20])
    """
    if require_kwargs is None:
        require_kwargs = {}
    
    arg2 = to_any_array(arg2)
    
    if arg2.ndim < axis + 1:
        return np.broadcast_to(arg1, (1,))[0]  
    
    arg1 = np.broadcast_to(arg1, (arg2.shape[axis],))
    
    # 应用 np.require 对数组进行类型和内存布局转换
    # 可以指定数据类型、内存排列顺序、是否可写等
    arg1 = np.require(arg1, **require_kwargs)
    
    return arg1


def get_multiindex_series(arg: tp.SeriesFrame) -> tp.Series:
    """
    确保输入对象是一个具有多级索引 (MultiIndex) 的 Pandas Series。

    如果传入的是一个 Pandas DataFrame，该函数会尝试将其转换为 Series。
    转换规则如下：
    - 如果 DataFrame 恰好有一行，则该行被转换为 Series，此时原 DataFrame 的列索引将成为新 Series 的索引。
    - 如果 DataFrame 恰好有一列，则该列被转换为 Series，此时原 DataFrame 的行索引将成为新 Series 的索引。
    - 如果 DataFrame 的行数和列数均大于1，则会引发 ValueError。
    最终，函数会校验得到的 Series 是否拥有多级索引，若否则会引发断言错误。

    参数:
        arg (tp.SeriesFrame):
            输入的 Pandas Series 或 DataFrame。
            期望此对象在转换后能形成一个具有多级索引的 Series。

    返回:
        tp.Series:
            一个保证具有多级索引的 Pandas Series。

    异常:
        TypeError: 如果输入 `arg` 不是 Pandas Series 或 DataFrame。
        ValueError: 如果输入的 DataFrame 不是单行或单列，无法明确转换为 Series。
        AssertionError: 如果最终得到的 Series 的索引不是 `pd.MultiIndex` 类型。

    示例:
        >>> import pandas as pd
        >>> # 示例 1: 输入一个已有的多级索引 Series
        >>> index1 = pd.MultiIndex.from_tuples([('A', 1), ('A', 2)], names=['L1', 'L2'])
        >>> series1 = pd.Series([10, 20], index=index1)
        >>> get_multiindex_series(series1)
        L1  L2
        A   1     10
            2     20
        dtype: int64

        >>> # 示例 2: 输入一个单行 DataFrame
        >>> df_single_row = pd.DataFrame([[100, 200]], columns=pd.MultiIndex.from_tuples([('X', 'a'), ('X', 'b')]))
        >>> # 期望结果: 行索引被丢弃，列的多级索引成为 Series 的索引
        >>> get_multiindex_series(df_single_row)
        X  a    100
           b    200
        Name: 0, dtype: int64


        >>> # 示例 3: 输入一个单列 DataFrame
        >>> index3 = pd.MultiIndex.from_tuples([('C', 3), ('D', 4)], names=['L1', 'L2'])
        >>> df_single_col = pd.DataFrame({'Data': [30, 40]}, index=index3)
        >>> # 期望结果: 列名被丢弃，原DataFrame的行多级索引成为 Series 的索引
        >>> get_multiindex_series(df_single_col)
        L1  L2
        C   3     30
        D   4     40
        Name: Data, dtype: int64

        >>> # 示例 4: 输入一个不符合转换条件的 DataFrame (多行多列)
        >>> df_multi = pd.DataFrame({'A': [1,2], 'B': [3,4]})
        >>> try:
        ...     get_multiindex_series(df_multi)
        ... except ValueError as e:
        ...     print(e)
        Supported are either Series or DataFrame with one column/row

        >>> # 示例 5: 输入一个 Series 但其索引不是 MultiIndex
        >>> series_plain_index = pd.Series([1,2,3])
        >>> try:
        ...     get_multiindex_series(series_plain_index)
        ... except AssertionError: # checks.assert_instance_of 会抛出 AssertionError
        ...     print("AssertionError: Index is not a MultiIndex")
        AssertionError: Index is not a MultiIndex
    """
    # 断言输入参数 arg 必须是 pandas.Series 或 pandas.DataFrame 类型。
    checks.assert_instance_of(arg, (pd.Series, pd.DataFrame))

    # 检查 arg 是否为 DataFrame 类型。
    if checks.is_frame(arg):
        # 如果 DataFrame 只有一行。
        if arg.shape[0] == 1:
            # 提取这唯一的一行，将其转换为 Series。
            # 此时，原 DataFrame 的列索引将成为新 Series 的索引。
            arg = arg.iloc[0, :]
        # 如果 DataFrame 只有一列 (而不是只有一行的情况)。
        elif arg.shape[1] == 1:
            # 提取这唯一的一列，将其转换为 Series。
            # 此时，原 DataFrame 的行索引将成为新 Series 的索引。
            arg = arg.iloc[:, 0]
        # 如果 DataFrame 的行数和列数都大于1。
        else:
            # 抛出 ValueError，因为无法明确地将多行多列的 DataFrame 转换为单一的 Series。
            raise ValueError("Supported are either Series or DataFrame with one column/row")

    # 经过上述处理后，arg 要么是原始的 Series，要么是从 DataFrame 转换而来的 Series。
    # 在这里，断言 arg 的索引必须是 pandas.MultiIndex 类型。
    checks.assert_instance_of(arg.index, pd.MultiIndex)

    return arg


def unstack_to_array(arg: tp.SeriesFrame, levels: tp.Optional[tp.MaybeLevelSequence] = None) -> tp.Array:
    """
    将具有多级索引的Pandas Series或DataFrame根据指定的索引级别重塑为一个多维NumPy数组。

    此函数的核心功能是将扁平化的Series/DataFrame数据，根据其MultiIndex的结构，
    “展开”成一个具有与索引层级对应维度的高维数组。缺失值将由NaN填充。

    参数:
        arg (tp.SeriesFrame): 
            输入的Pandas Series或DataFrame。
            - 如果是DataFrame，它必须最多只有一行或一列，以便能明确地转换为一个Series进行处理。
            - 其索引必须是 `pd.MultiIndex` 类型。
        levels (tp.Optional[tp.MaybeLevelSequence], optional): 
            一个整数、字符串，或者它们的序列，用于指定将哪些索引级别（level）以及以何种顺序
            用于构建新NumPy数组的各个维度。
            - 如果为 `None` (默认值)，则使用MultiIndex中的所有级别，顺序与它们在MultiIndex中的定义一致。
            - 级别可以通过其整数位置（从0开始）或名称（如果MultiIndex的级别已命名）来指定。
            - 这些指定的级别将决定输出NumPy数组的维度顺序和大小。

    返回:
        tp.Array: 
            一个多维NumPy数组。
            - 数组的维度数量等于 `levels` 参数中指定的级别数量（或MultiIndex的总级别数，如果`levels`为None）。
            - 每个维度的大小由对应索引级别中的唯一值的数量决定。
            - 数组中的值来源于原始Series，根据索引映射到相应的位置。
            - 如果在unstack后的某个组合在原始数据中不存在，则该位置在新数组中将用 `np.nan` 填充。

    异常:
        ValueError: 
            - 如果输入的DataFrame不符合单行或单列的要求。
            - 如果Series的MultiIndex中包含重复的索引条目（因为重复索引无法唯一确定unstack后的位置）。
        AssertionError:
            - 如果输入 `arg` 的类型不是Pandas Series或DataFrame。
            - 如果转换后的Series的索引类型不是 `pd.MultiIndex`。

    使用示例:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from vectorbt.base import index_fns # 假设 index_fns 在此路径下或已导入
        >>> # from vectorbt.base.reshape_fns import get_multiindex_series, unstack_to_array # 假设在此

        >>> # 构造一个具有三级索引的Series
        >>> index = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2],    # 第0级索引 (例如：'group')
        ...      [3, 4, 3, 4],    # 第1级索引 (例如：'category')
        ...      ['a', 'b', 'c', 'd']], # 第2级索引 (例如：'item')
        ...     names=['group', 'category', 'item']
        ... )
        >>> sr = pd.Series([10, 20, 30, 40], index=index)
        >>> print("原始Series:")
        >>> print(sr)
        原始Series:
        group  category  item
        1      3         a       10
               4         b       20
        2      3         c       30
               4         d       40
        dtype: int64

        >>> # 示例1: 使用所有索引级别 (默认行为)
        >>> arr_all_levels = unstack_to_array(sr)
        >>> print("\\n示例1 - 使用所有级别unstack:")
        >>> print(f"形状: {arr_all_levels.shape}") # 形状将是 (2, 2, 4) -> (len(unique_group), len(unique_cat), len(unique_item))
                                                # 注意：如果某一级别的唯一值在不同组合中不全出现，则该级别在数组中的长度是所有唯一值的总数
                                                # 在这个例子中，item 'a','b','c','d' 都是唯一的，所以第三维是4
        >>> print(arr_all_levels)
        示例1 - 使用所有级别unstack:
        形状: (2, 2, 4)
        [[[10. nan nan nan]
          [nan 20. nan nan]]
        <BLANKLINE>
         [[nan nan 30. nan]
          [nan nan nan 40.]]]
    """
    # 首先，确保输入参数arg被转换为具有多级索引的一维Pandas Series
    # get_multiindex_series会处理arg是Series或单行/单列DataFrame的情况，并检查索引是否为MultiIndex
    sr: tp.Series = to_1d(get_multiindex_series(arg))

    # 检查转换后的Series的索引中是否包含重复条目
    # 如果有重复索引，unstack的结果是不明确的，因此抛出ValueError
    if sr.index.duplicated().any():
        raise ValueError("Index contains duplicate entries, cannot reshape")

    # 初始化两个列表：
    # unique_idx_list: 用于存储每个被选中进行unstack的索引级别中的唯一值数组。
    #                  这些唯一值将决定新NumPy数组对应维度的大小和“刻度”。
    # vals_idx_list: 用于存储原始Series中每个数据点，在其对应的每个被选中的索引级别上，
    #                的索引值在unique_idx_list中对应级别唯一值数组中的整数位置。
    #                这些整数位置将作为填充新NumPy数组时的坐标。
    unique_idx_list = []
    vals_idx_list = []
    # 如果用户没有指定levels参数，则默认使用Series索引的所有级别
    if levels is None:
        levels = range(sr.index.nlevels)
    # 如果levels是单个整数或字符串（代表单个级别），将其转换为单元素元组，以便后续统一处理
    if isinstance(levels, (int, str)):
        levels = (levels,)

    # 遍历由levels参数指定的每一个索引级别
    for level in levels:
        # 从Series的MultiIndex中提取当前level的所有索引值，并转换为NumPy数组
        vals = index_fns.select_levels(sr.index, level).to_numpy()
        # 找到当前level中的所有唯一索引值，这些唯一值将构成新数组在此维度上的“刻度”
        unique_vals = np.unique(vals)
        # 将当前level的唯一值数组存入unique_idx_list
        unique_idx_list.append(unique_vals)

        # 创建一个字典，将当前level的每个唯一索引值映射到它在unique_vals数组中的整数位置（0, 1, 2, ...）
        # 例如，如果unique_vals是['a', 'b', 'c']，则idx_map是{'a':0, 'b':1, 'c':2}
        idx_map = dict(zip(unique_vals, range(len(unique_vals))))
        # 对于原始Series中当前level的每一个索引值，查询它在idx_map中的整数位置
        # 结果vals_idx是一个列表，长度与原始Series相同，包含了每个数据点在当前level上的整数坐标
        vals_idx = list(map(lambda x: idx_map[x], vals))
        # 将当前level的整数坐标列表存入vals_idx_list
        vals_idx_list.append(vals_idx)

    # 根据每个选定level的唯一值的数量，确定新NumPy数组的形状
    # 例如，如果levels有两个级别，第一个级别有U1个唯一值，第二个级别有U2个唯一值，则形状为(U1, U2)
    # 使用np.full创建一个填充了np.nan的多维数组，其形状由unique_idx_list中各唯一值数组的长度决定
    a = np.full(list(map(len, unique_idx_list)), np.nan)

    # 这是关键的赋值步骤：将原始Series的值填充到新创建的多维数组a中
    # tuple(zip(vals_idx_list))的解释：
    #   - vals_idx_list是一个列表的列表，例如 [[level0_pos_for_item1, level0_pos_for_item2, ...],
    #                                       [level1_pos_for_item1, level1_pos_for_item2, ...], ...]
    #   - zip(*vals_idx_list) (如果vals_idx_list是转置的) 或者 zip(vals_idx_list[0], vals_idx_list[1], ...)
    #     这里应该是 zip(*vals_idx_list) 如果 vals_idx_list 是 [[item1_L0, item1_L1], [item2_L0, item2_L1], ...]
    #     或者，如果 vals_idx_list 的结构是 [[all_L0_indices], [all_L1_indices], ...]，
    #     那么 tuple(zip(*vals_idx_list)) 是不正确的，应该是 a[tuple(vals_idx_list)] 或者
    #     NumPy的高级索引期望的是一个元组，元组的每个元素是一个数组，对应一个维度。
    #     例如 a[(coords_dim0, coords_dim1, coords_dim2)]
    #     vals_idx_list中的每个子列表对应一个维度的坐标。
    #     因此，直接用 tuple(vals_idx_list) 作为索引是正确的，它会被NumPy解释为高级索引。
    #     例如，如果 a 是2D，vals_idx_list 是 [[0,1,0], [2,0,1]]，
    #     则 a[([0,1,0], [2,0,1])] 会访问 (0,2), (1,0), (0,1) 这三个位置。
    # 将sr.values（原始Series中的数据值）赋给数组a中由vals_idx_list确定的多维坐标位置
    a[tuple(vals_idx_list)] = sr.values
    
    # 返回填充好数据的多维NumPy数组
    return a


def make_symmetric(arg: tp.SeriesFrame, sort: bool = True) -> tp.Frame:
    """
    将输入的Pandas Series或DataFrame转换为一个对称的DataFrame。

    对称意味着结果DataFrame的行索引和列索引是完全相同的。
    新的行列索引将由原始输入 `arg` 的行索引和列索引中的所有唯一值构成。
    函数会尝试智能地处理原始索引的名称。

    参数:
        arg (tp.SeriesFrame):
            输入的Pandas Series或DataFrame。
            - 如果是Series，它会首先被转换为一个单列的DataFrame。
        sort (bool, optional):
            一个布尔值，指示在合并原始行索引和列索引以形成新的对称索引时，是否应对这些索引用值进行排序。
            - `True` (默认值): 新的对称索引将按值排序。
            - `False`: 新的对称索引将按照原始行索引、列索引值连接后的顺序，并移除重复项，不进行额外排序。

    返回:
        tp.Frame:
            一个对称的Pandas DataFrame。其行索引和列索引相同，包含了输入 `arg`
            原始行索引和列索引的所有唯一值。DataFrame中的值根据原始 `arg` 的值
            及其转置进行填充，以实现对称性。

    要求:
        - 如果输入 `arg` 是一个DataFrame，并且其行索引和列索引都是 `pd.MultiIndex` 类型，
          那么这两个MultiIndex必须具有相同的层级数量（`nlevels`）。

    示例:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from vectorbt.base.reshape_fns import to_2d # 假设 to_2d 在此或已导入
        >>> # from vectorbt.base.reshape_fns import make_symmetric # 假设在此

        >>> # 示例1: 基本的DataFrame
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=['a', 'b'], columns=['c', 'd'])
        >>> print("原始DataFrame df1:")
        >>> print(df1)
        原始DataFrame df1:
           c  d
        a  1  2
        b  3  4
        >>> sym_df1 = make_symmetric(df1)
        >>> print("\\n对称后的df1 (sort=True):")
        >>> print(sym_df1) # 新索引将是 ['a', 'b', 'c', 'd']
        对称后的df1 (sort=True):
             a    b    c    d
        a  NaN  NaN  1.0  2.0  # df1中 a行c列=1, a行d列=2
        b  NaN  NaN  3.0  4.0  # df1中 b行c列=3, b行d列=4
        c  1.0  3.0  NaN  NaN  # df1转置后 c行a列=1, c行b列=3
        d  2.0  4.0  NaN  NaN  # df1转置后 d行a列=2, d行b列=4

        >>> # 示例3: 输入为Series
        >>> sr1 = pd.Series([1, 2, 3], index=pd.Index(['a', 'b', 'c'], name='my_idx'), name='my_series')
        >>> print("\\n原始Series sr1:")
        >>> print(sr1)
        原始Series sr1:
        my_idx
        a    1
        b    2
        c    3
        Name: my_series, dtype: int64
        >>> sym_sr1 = make_symmetric(sr1) # Series会转为单列DataFrame，列名为'my_series'
        >>> print("\\n对称后的sr1:")      # 新索引为 ['a', 'b', 'c', 'my_series'] (排序后)
        >>> print(sym_sr1)
        对称后的sr1:
                   a    b    c  my_series
        a        NaN  NaN  NaN        1.0
        b        NaN  NaN  NaN        2.0
        c        NaN  NaN  NaN        3.0
        my_series  1.0  2.0  3.0        NaN

        >>> # 示例4: MultiIndex (确保层级相同)
        >>> idx_m1 = pd.MultiIndex.from_product([['A'], ['x', 'y']], names=['L1', 'L2'])
        >>> col_m1 = pd.MultiIndex.from_product([['B'], ['z', 'w']], names=['L1', 'L2']) # 注意：层级名可以不同
        >>> df_multi = pd.DataFrame([[1,2],[3,4]], index=idx_m1, columns=col_m1)
        >>> print("\\n原始DataFrame df_multi:")
        >>> print(df_multi)
        原始DataFrame df_multi:
        L1      B   
        L2      z  w
        L1 L2       
        A  x    1  2
           y    3  4
        >>> sym_df_multi = make_symmetric(df_multi)
        >>> print("\\n对称后的df_multi:")
        >>> print(sym_df_multi) # 新的MultiIndex将合并(A,x), (A,y), (B,z), (B,w)并排序
                                # 名称将是 (('L1','L1'), ('L2','L2'))
        对称后的df_multi:
        L1         A         B    
        L2         x    y    z    w
        L1 L2                      
        A  x     NaN  NaN  1.0  2.0
           y     NaN  NaN  3.0  4.0
        B  z     1.0  3.0  NaN  NaN
           w     2.0  4.0  NaN  NaN
    """
    checks.assert_instance_of(arg, (pd.Series, pd.DataFrame))
    df: tp.Frame = to_2d(arg)
    if isinstance(df.index, pd.MultiIndex) or isinstance(df.columns, pd.MultiIndex):
        checks.assert_instance_of(df.index, pd.MultiIndex)
        checks.assert_instance_of(df.columns, pd.MultiIndex)
        checks.assert_array_equal(df.index.nlevels, df.columns.nlevels)
        names1, names2 = tuple(df.index.names), tuple(df.columns.names)
    else:
        names1, names2 = df.index.name, df.columns.name

    if names1 == names2:
        new_name = names1
    else:
        if isinstance(df.index, pd.MultiIndex):
            new_name = tuple(zip(*[names1, names2]))
        else:
            new_name = (names1, names2)
    if sort:
        idx_vals = np.unique(np.concatenate((df.index, df.columns))).tolist()
    else:
        idx_vals = list(dict.fromkeys(np.concatenate((df.index, df.columns))))
    df_index = df.index.copy()
    df_columns = df.columns.copy()
    if isinstance(df.index, pd.MultiIndex):
        unique_index = pd.MultiIndex.from_tuples(idx_vals, names=new_name)
        df_index.names = new_name
        df_columns.names = new_name
    else:
        unique_index = pd.Index(idx_vals, name=new_name)
        df_index.name = new_name
        df_columns.name = new_name
    df = df.copy(deep=False)
    df.index = df_index
    df.columns = df_columns
    df_out_dtype = np.promote_types(df.values.dtype, np.min_scalar_type(np.nan))
    df_out = pd.DataFrame(index=unique_index, columns=unique_index, dtype=df_out_dtype)
    df_out.loc[:, :] = df
    df_out[df_out.isnull()] = df.transpose()
    return df_out


def unstack_to_df(arg: tp.SeriesFrame,
                  index_levels: tp.Optional[tp.MaybeLevelSequence] = None,
                  column_levels: tp.Optional[tp.MaybeLevelSequence] = None,
                  symmetric: bool = False,
                  sort: bool = True) -> tp.Frame:
    """根据多级索引，将输入对象重塑为二维 DataFrame。
    
    该函数将具有多级索引（MultiIndex）的 Series 或 DataFrame 重新组织为二维表格形式，
    通过指定哪些索引级别用于新 DataFrame 的行索引，哪些用于列索引，实现数据的重塑变换。
    这是一种强大的数据透视方法，可以灵活地改变数据的查看维度。
    
    参数:
        arg (tp.SeriesFrame): 
            输入的 Pandas Series 或 DataFrame。要求其索引必须是多级索引（MultiIndex）。
            如果输入是 DataFrame，必须能被转换为具有多级索引的 Series
            （通常是单行或单列的 DataFrame）。
            
        index_levels (tp.Optional[tp.MaybeLevelSequence], 默认为 None): 
            指定哪些索引级别将构成结果 DataFrame 的行索引。
            可以是整数、字符串或它们的序列，用于从原始多级索引中选择级别。
            - 整数表示级别的位置（从0开始）
            - 字符串表示级别的名称（如果多级索引的级别已命名）
            如果原始索引只有两个级别且此参数为 None，默认使用第一个级别（0）。
            
        column_levels (tp.Optional[tp.MaybeLevelSequence], 默认为 None): 
            指定哪些索引级别将构成结果 DataFrame 的列索引。
            格式与 index_levels 相同。
            如果原始索引只有两个级别且此参数为 None，默认使用第二个级别（1）。
            
        symmetric (bool, 默认为 False): 
            是否生成对称 DataFrame。
            - 如果为 True，将使用 make_symmetric 函数将结果转换为对称 DataFrame，
              其中行索引和列索引相同，包含所有选定索引级别的唯一值组合。
            - 如果为 False，返回普通的重塑后的 DataFrame。
            
        sort (bool, 默认为 True): 
            仅当 symmetric=True 时有效。指示在创建对称 DataFrame 时，
            是否对合并后的行列索引进行排序。
            
    返回:
        tp.Frame: 
            重塑后的 DataFrame。
            - 如果 symmetric=False，返回的 DataFrame 行索引来自 index_levels 指定的级别，
              列索引来自 column_levels 指定的级别。
            - 如果 symmetric=True，返回的 DataFrame 行索引和列索引相同，
              包含从 index_levels 和 column_levels 选择的所有唯一值组合。
            
    异常:
        ValueError: 
            - 如果原始索引超过两个级别，但未指定 index_levels 或 column_levels。
            - 如果输入无法转换为具有多级索引的 Series。
            
    示例:
        >>> import pandas as pd
        >>> from vectorbt.base.reshape_fns import unstack_to_df
        >>> 
        >>> # 创建一个具有三级索引的 Series
        >>> index = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2], [3, 4, 3, 4], ['a', 'b', 'c', 'd']],
        ...     names=['x', 'y', 'z'])
        >>> sr = pd.Series([1, 2, 3, 4], index=index)
        >>> 
        >>> # 将 x 和 y 级别用作行索引，z 级别用作列索引
        >>> df1 = unstack_to_df(sr, index_levels=(0, 1), column_levels=2)
        >>> print(df1)
        z      a    b    c    d
        x y                    
        1 3  1.0  NaN  NaN  NaN
          4  NaN  2.0  NaN  NaN
        2 3  NaN  NaN  3.0  NaN
          4  NaN  NaN  NaN  4.0
        >>> 
        >>> # 创建对称 DataFrame
        >>> df2 = unstack_to_df(sr, index_levels=0, column_levels=1, symmetric=True)
        >>> print(df2)
             3    4
        1  NaN  NaN
        2  NaN  NaN
        3  1.0  2.0
        4  3.0  4.0
    """
    # 从输入参数提取 Series
    # 使用 to_1d 确保获得一维序列，get_multiindex_series 确保具有多级索引
    sr: tp.Series = to_1d(get_multiindex_series(arg))
    
    # 当索引级别数大于2时，必须明确指定 index_levels 和 column_levels
    if len(sr.index.levels) > 2:
        if index_levels is None:
            raise ValueError("index_levels must be specified")
        if column_levels is None:
            raise ValueError("column_levels must be specified")
    else:
        # 对于仅有两个级别的索引，默认使用第一个级别作为行索引，第二个级别作为列索引
        if index_levels is None:
            index_levels = 0
        if column_levels is None:
            column_levels = 1
    
    # 构建新的行索引和列索引层次结构
    # select_levels 从多级索引中选择指定级别，unique() 确保索引值不重复
    new_index = index_fns.select_levels(arg.index, index_levels).unique()
    new_columns = index_fns.select_levels(arg.index, column_levels).unique()
    
    # 执行重塑操作并进行后处理
    # unstack_to_array 将 Series 重塑为 NumPy 数组，指定使用哪些级别
    unstacked = unstack_to_array(sr, levels=(index_levels, column_levels))
    # 创建 DataFrame，使用前面生成的新行索引和列索引
    df = pd.DataFrame(unstacked, index=new_index, columns=new_columns)
    
    # 如果要求对称，调用 make_symmetric 函数处理结果
    # 对称 DataFrame 的行索引和列索引完全相同
    if symmetric:
        return make_symmetric(df, sort=sort)
    # 否则直接返回重塑后的 DataFrame
    return df


@njit(cache=True)
def flex_choose_i_and_col_nb(a: tp.Array, flex_2d: bool = True) -> tp.Tuple[int, int]:
    """
    根据标量/一维/二维数组 a 的行/列数是否为1，返回 (0/-1, 0/-1)
    其中一维数组较为特殊：
        flex_2d == True：认为是行向量，返回 (0, 0/-1)
        flex_2d == False：认为是列向量，返回 (0/-1, 0)
    """
    i = -1
    col = -1
    if a.ndim == 0:
        i = 0
        col = 0
    elif a.ndim == 1:
        if flex_2d:
            i = 0
            if a.shape[0] == 1:
                col = 0
        else:
            col = 0
            if a.shape[0] == 1:
                i = 0
    else:
        if a.shape[0] == 1:
            i = 0
        if a.shape[1] == 1:
            col = 0
    return i, col


@njit(cache=True)
def flex_select_nb(a: tp.Array, i: int, col: int, flex_i: int, flex_col: int, flex_2d: bool = True) -> tp.Any:
    if flex_i == -1:
        flex_i = i
    if flex_col == -1:
        flex_col = col
    
    if a.ndim == 0:
        return a.item()
    
    if a.ndim == 1:
        if flex_2d:
            return a[flex_col]
        return a[flex_i]
    return a[flex_i, flex_col]


@njit(cache=True)
def flex_select_auto_nb(a: tp.Array, i: int, col: int, flex_2d: bool = True) -> tp.Any:
    flex_i, flex_col = flex_choose_i_and_col_nb(a, flex_2d)
    return flex_select_nb(a, i, col, flex_i, flex_col, flex_2d)
