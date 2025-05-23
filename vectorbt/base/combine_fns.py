# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Functions for combining arrays.

Combine functions combine two or more NumPy arrays using a custom function. The emphasis here is
done upon stacking the results into one NumPy array - since vectorbt is all about brute-forcing
large spaces of hyperparameters, concatenating the results of each hyperparameter combination into
a single DataFrame is important. All functions are available in both Python and Numba-compiled form."""

import numpy as np
from numba import njit
from tqdm.auto import tqdm

from vectorbt import _typing as tp
from vectorbt.base import reshape_fns


def apply_and_concat_none(n: int,
                          apply_func: tp.Callable, 
                          *args,
                          show_progress: bool = False,
                          tqdm_kwargs: tp.KwargsLike = None,
                          **kwargs) -> None:
    """
    对范围[0, n)内的每个整数值依次调用指定的应用函数，主要用于执行原地操作
    （如文件写入、数据库更新、图像保存等）而不需要收集返回值的场景。
    
    Args:
        n (int): 循环迭代的总次数，函数将对0到n-1的每个整数值调用apply_func
        apply_func (tp.Callable): 要应用的可调用函数，该函数必须接受以下参数：
                                 - 第一个参数：当前迭代的索引值i (int)
                                 - *args：位置参数
                                 - **kwargs：关键字参数
        *args: 传递给apply_func的额外位置参数
        show_progress (bool, optional): 是否显示进度条。默认为False
                                      - True: 显示tqdm进度条
                                      - False: 不显示进度条
        tqdm_kwargs (tp.KwargsLike, optional): 传递给tqdm进度条的配置参数字典
                                             常用参数如desc（描述文本）、unit（单位）等
                                             默认为None，内部会初始化为空字典
        **kwargs: 传递给apply_func的额外关键字参数
    
    Returns:
        None: 该函数不返回任何值，主要用于执行副作用操作
    
    Examples:
        >>> # 示例1：批量保存文件
        >>> def save_file(i, data_list, output_dir):
        ...     filename = f"{output_dir}/file_{i}.txt"
        ...     with open(filename, 'w') as f:
        ...         f.write(data_list[i])
        >>> 
        >>> data = ["content1", "content2", "content3"]
        >>> apply_and_concat_none(3, save_file, data, "/tmp", 
        ...                      show_progress=True,
        ...                      tqdm_kwargs={"desc": "保存文件"})
    """
    if tqdm_kwargs is None:
        tqdm_kwargs = {}
    for i in tqdm(range(n), disable=not show_progress, **tqdm_kwargs):
        apply_func(i, *args, **kwargs)


@njit  # Numba即时编译装饰器，将Python函数编译为高性能的机器码
def apply_and_concat_none_nb(n: int, apply_func_nb: tp.Callable, *args) -> None:
    """
    apply_and_concat_none函数的Numba编译版本，用于高性能的批量函数应用操作
    由于Numba的限制，该函数简化了参数接口，去除了进度条显示和关键字参数支持。
    
    Args:
        n (int): 循环迭代的总次数，函数将对0到n-1的每个整数值调用apply_func_nb
               必须是正整数，表示要执行的批量操作数量
        apply_func_nb (tp.Callable): 要应用的Numba编译函数，该函数必须满足以下要求：
                                   - 必须使用@njit装饰器进行Numba编译
                                   - 第一个参数：当前迭代的索引值i (int)
                                   - 后续参数：*args解包的位置参数
                                   - 不支持关键字参数(**kwargs)
                                   - 函数内部只能使用Numba支持的操作和数据类型
        *args: 传递给apply_func_nb的额外位置参数，必须是Numba兼容的数据类型
              包括NumPy数组、标量数值、元组等，不支持Python对象、字典、列表等
    
    Returns:
        None: 该函数不返回任何值，主要用于执行副作用操作（如原地数组修改）
    
    Examples:
        >>> import numpy as np
        >>> from numba import njit
        >>> 
        >>> # 示例1：批量原地数组操作
        >>> @njit
        ... def multiply_inplace(i, arr, multiplier):
        ...     arr[i] *= multiplier  # 原地修改数组元素
        >>> 
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> apply_and_concat_none_nb(len(data), multiply_inplace, data, 2.0)
        >>> # 结果：data变为[2.0, 4.0, 6.0, 8.0, 10.0]
    """
    for i in range(n):
        apply_func_nb(i, *args)


def apply_and_concat_one(n: int,
                         apply_func: tp.Callable, *args,
                         show_progress: bool = False,
                         tqdm_kwargs: tp.KwargsLike = None,
                         **kwargs) -> tp.Array2d:
    """
    批量应用函数并按列连接结果的核心工具函数
    
    该函数对范围[0, n)内的每个整数值依次调用指定的应用函数，收集每次调用的结果，
    并将所有结果沿着轴1（列方向）连接成一个二维NumPy数组。这是vectorbt库中
    用于批量处理和超参数网格搜索的核心函数之一，特别适用于量化交易中的批量回测、
    特征工程和策略优化等场景。
    
    Args:
        n (int): 循环迭代的总次数，函数将对0到n-1的每个整数值调用apply_func
               必须是正整数，表示要执行的批量操作数量
               
        apply_func (tp.Callable): 要应用的可调用函数，该函数必须满足以下要求：
                                 - 第一个参数：当前迭代的索引值i (int)
                                 - 后续参数：*args解包的位置参数
                                 - 关键字参数：**kwargs解包的关键字参数
                                 - 返回值：必须是1维或2维的NumPy数组
                                 
        *args: 传递给apply_func的额外位置参数
              可以是任何Python对象，常见类型包括：
              - NumPy数组（如价格数据、成交量数据）
              - 标量参数（如技术指标的周期参数）
              - 配置对象（如策略参数字典）
              
        show_progress (bool, optional): 是否显示进度条。默认为False
                                      - True: 显示tqdm进度条，适用于长时间运行的批量计算
                                      - False: 不显示进度条，适用于快速计算或自动化脚本
                                      
        tqdm_kwargs (tp.KwargsLike, optional): 传递给tqdm进度条的配置参数字典
                                             常用参数包括：
                                             - desc: 进度条描述文本（如"计算技术指标"）
                                             - unit: 进度单位（如"股票", "策略", "参数组合"）
                                             - ncols: 进度条显示宽度
                                             - leave: 是否保留完成后的进度条
                                             默认为None，内部会初始化为空字典
                                             
        **kwargs: 传递给apply_func的额外关键字参数
                 可以包含策略参数、计算配置、数据处理选项等
    
    Returns:
        tp.Array2d: 二维NumPy数组，形状为(rows, n * cols)
                   其中rows是apply_func返回数组的行数，cols是返回数组的列数
                   每次apply_func调用的结果按列顺序连接在一起
                   
                   数组结构说明：
                   - 如果apply_func返回1维数组[a1, a2, a3]，结果形状为(3, n)
                   - 如果apply_func返回2维数组shape为(m, k)，结果形状为(m, n*k)
                   - 第i次调用的结果存储在列索引[i*k:(i+1)*k]的位置
    Examples:
        >>> import numpy as np
        >>> 
        >>> # 示例1：批量计算不同周期的简单移动平均线
        >>> def calculate_sma(i, prices, periods):
        ...     period = periods[i]
        ...     # 计算简单移动平均线
        ...     sma = np.convolve(prices, np.ones(period)/period, mode='valid')
        ...     return sma.reshape(-1, 1)  # 返回列向量
        >>> 
        >>> prices = np.random.rand(100) * 100  # 模拟价格数据
        >>> periods = [5, 10, 20, 50]  # 不同的移动平均周期
        >>> results = apply_and_concat_one(
        ...     len(periods), 
        ...     calculate_sma, 
        ...     prices, 
        ...     periods,
        ...     show_progress=True,
        ...     tqdm_kwargs={"desc": "计算SMA", "unit": "周期"}
        ... )
        >>> # 结果：shape为(96, 4)，每列对应一个周期的SMA值
    """
    if tqdm_kwargs is None:
        tqdm_kwargs = {}
    
    outputs = []
    
    for i in tqdm(range(n), disable=not show_progress, **tqdm_kwargs):
        # reshape_fns.to_2d(): 确保apply_func返回的数组被转换为2维格式，来确保后续np.column_stack操作的一致性
        outputs.append(reshape_fns.to_2d(apply_func(i, *args, **kwargs)))
    
    # 使用NumPy的column_stack函数将所有输出数组按列连接
    # column_stack会将输入的数组序列沿着轴1（列方向）堆叠
    # 例如：[array([[1], [2]]), array([[3], [4]])] -> array([[1, 3], [2, 4]])
    return np.column_stack(outputs)


@njit
def to_2d_one_nb(a: tp.Array) -> tp.Array2d:
    """Expand the dimensions of array `a` along axis 1.

    !!! note
        * `a` must be strictly homogeneous"""
    if a.ndim > 1:
        return a
    return np.expand_dims(a, axis=1)


@njit
def apply_and_concat_one_nb(n: int, apply_func_nb: tp.Callable, *args) -> tp.Array2d:
    """A Numba-compiled version of `apply_and_concat_one`.

    !!! note
        * `apply_func_nb` must be Numba-compiled
        * `*args` must be Numba-compatible
        * No support for `**kwargs`
    """
    output_0 = to_2d_one_nb(apply_func_nb(0, *args))
    output = np.empty((output_0.shape[0], n * output_0.shape[1]), dtype=output_0.dtype)
    for i in range(n):
        if i == 0:
            outputs_i = output_0
        else:
            outputs_i = to_2d_one_nb(apply_func_nb(i, *args))
        output[:, i * outputs_i.shape[1]:(i + 1) * outputs_i.shape[1]] = outputs_i
    return output


def apply_and_concat_multiple(n: int,
                              apply_func: tp.Callable, *args,
                              show_progress: bool = False,
                              tqdm_kwargs: tp.KwargsLike = None,
                              **kwargs) -> tp.List[tp.Array2d]:
    """Identical to `apply_and_concat_one`, except that the result of `apply_func` must be
    multiple 1-dim or 2-dim arrays. Each of these arrays at `i` will be concatenated with the
    array at the same position at `i+1`."""
    if tqdm_kwargs is None:
        tqdm_kwargs = {}
    outputs = []
    for i in tqdm(range(n), disable=not show_progress, **tqdm_kwargs):
        outputs.append(tuple(map(reshape_fns.to_2d, apply_func(i, *args, **kwargs))))
    return list(map(np.column_stack, list(zip(*outputs))))


@njit
def to_2d_multiple_nb(a: tp.Iterable[tp.Array]) -> tp.List[tp.Array2d]:
    """Expand the dimensions of each array in `a` along axis 1.

    !!! note
        * `a` must be strictly homogeneous
    """
    lst = list()
    for _a in a:
        lst.append(to_2d_one_nb(_a))
    return lst


@njit
def apply_and_concat_multiple_nb(n: int, apply_func_nb: tp.Callable, *args) -> tp.List[tp.Array2d]:
    """A Numba-compiled version of `apply_and_concat_multiple`.

    !!! note
        * Output of `apply_func_nb` must be strictly homogeneous
        * `apply_func_nb` must be Numba-compiled
        * `*args` must be Numba-compatible
        * No support for `**kwargs`
    """
    outputs = list()
    outputs_0 = to_2d_multiple_nb(apply_func_nb(0, *args))
    for j in range(len(outputs_0)):
        outputs.append(np.empty((outputs_0[j].shape[0], n * outputs_0[j].shape[1]), dtype=outputs_0[j].dtype))
    for i in range(n):
        if i == 0:
            outputs_i = outputs_0
        else:
            outputs_i = to_2d_multiple_nb(apply_func_nb(i, *args))
        for j in range(len(outputs_i)):
            outputs[j][:, i * outputs_i[j].shape[1]:(i + 1) * outputs_i[j].shape[1]] = outputs_i[j]
    return outputs


def select_and_combine(i: int,
                       obj: tp.Any,
                       others: tp.Sequence,
                       combine_func: tp.Callable,
                       *args, **kwargs) -> tp.AnyArray:
    """Combine `obj` and an element from `others` at `i` using `combine_func`."""
    return combine_func(obj, others[i], *args, **kwargs)


def combine_and_concat(obj: tp.Any,
                       others: tp.Sequence,
                       combine_func: tp.Callable,
                       *args, **kwargs) -> tp.Array2d:
    """Use `apply_and_concat_one` to combine `obj` with each element from `others` using `combine_func`."""
    return apply_and_concat_one(len(others), select_and_combine, obj, others, combine_func, *args, **kwargs)


@njit
def select_and_combine_nb(i: int, obj: tp.Any, others: tp.Sequence, combine_func_nb: tp.Callable, *args) -> tp.Array:
    """A Numba-compiled version of `select_and_combine`.

    !!! note
        * `combine_func_nb` must be Numba-compiled
        * `obj`, `others` and `*args` must be Numba-compatible
        * `others` must be strictly homogeneous
        * No support for `**kwargs`
    """
    return combine_func_nb(obj, others[i], *args)


@njit
def combine_and_concat_nb(obj: tp.Any, others: tp.Sequence, combine_func_nb: tp.Callable, *args) -> tp.Array2d:
    """A Numba-compiled version of `combine_and_concat`."""
    return apply_and_concat_one_nb(len(others), select_and_combine_nb, obj, others, combine_func_nb, *args)


def combine_multiple(objs: tp.Sequence, combine_func: tp.Callable, *args, **kwargs) -> tp.AnyArray:
    """Combine `objs` pairwise into a single object."""
    result = objs[0]
    for i in range(1, len(objs)):
        result = combine_func(result, objs[i], *args, **kwargs)
    return result


@njit
def combine_multiple_nb(objs: tp.Sequence, combine_func_nb: tp.Callable, *args) -> tp.Array:
    """A Numba-compiled version of `combine_multiple`.

    !!! note
        * `combine_func_nb` must be Numba-compiled
        * `objs` and `*args` must be Numba-compatible
        * `objs` must be strictly homogeneous
        * No support for `**kwargs`
    """
    result = objs[0]
    for i in range(1, len(objs)):
        result = combine_func_nb(result, objs[i], *args)
    return result


def ray_apply(n: int,
              apply_func: tp.Callable, *args,
              ray_force_init: bool = False,
              ray_func_kwargs: tp.KwargsLike = None,
              ray_init_kwargs: tp.KwargsLike = None,
              ray_shutdown: bool = False,
              **kwargs) -> tp.List[tp.AnyArray]:
    """Run `apply_func` in distributed manner.

    Set `ray_reinit` to True to terminate the Ray runtime and initialize a new one.
    `ray_func_kwargs` will be passed to `ray.remote` and `ray_init_kwargs` to `ray.init`.
    Set `ray_shutdown` to True to terminate the Ray runtime upon the job end.

    """
    import ray

    if ray_init_kwargs is None:
        ray_init_kwargs = {}
    if ray_func_kwargs is None:
        ray_func_kwargs = {}
    if ray_force_init:
        if ray.is_initialized():
            ray.shutdown()
    if not ray.is_initialized():
        ray.init(**ray_init_kwargs)
    if len(ray_func_kwargs) > 0:
        apply_func = ray.remote(**ray_func_kwargs)(apply_func)
    else:
        apply_func = ray.remote(apply_func)
    # args and kwargs don't change -> put to object store
    arg_refs = ()
    for v in args:
        arg_refs += (ray.put(v),)
    kwarg_refs = {}
    for k, v in kwargs.items():
        kwarg_refs[k] = ray.put(v)
    futures = [apply_func.remote(i, *arg_refs, **kwarg_refs) for i in range(n)]
    results = ray.get(futures)
    if ray_shutdown:
        ray.shutdown()
    return results


def apply_and_concat_one_ray(*args, **kwargs) -> tp.Array2d:
    """Distributed version of `apply_and_concat_one`."""
    results = ray_apply(*args, **kwargs)
    return np.column_stack(list(map(reshape_fns.to_2d, results)))


def apply_and_concat_multiple_ray(*args, **kwargs) -> tp.List[tp.Array2d]:
    """Distributed version of `apply_and_concat_multiple`."""
    results = ray_apply(*args, **kwargs)
    return list(map(np.column_stack, list(zip(*results))))


def combine_and_concat_ray(obj: tp.Any,
                           others: tp.Sequence,
                           combine_func: tp.Callable,
                           *args, **kwargs) -> tp.Array2d:
    """Distributed version of `combine_and_concat`."""
    return apply_and_concat_one_ray(len(others), select_and_combine, obj, others, combine_func, *args, **kwargs)
