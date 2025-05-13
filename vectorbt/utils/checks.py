# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
运行时验证工具模块。

该模块提供了一组用于运行时验证的实用函数，主要分为两类：
1. 检查函数(Checks)：返回布尔值，用于测试某种条件是否满足
2. 断言函数(Asserts)：当条件不满足时抛出异常

这些函数主要用于：
- 类型检查和验证
- 数据结构属性验证（形状、维度、数据类型等）
- 索引和对象相等性检查
- 函数参数验证

模块设计遵循了函数式编程风格，提供了清晰的接口和一致的命名约定，
便于在vectorbt库的其他部分进行调用，增强代码的健壮性和可维护性。
"""

import os
from collections.abc import Hashable, Mapping
from inspect import signature, getmro
from keyword import iskeyword

import dill
import numpy as np
import pandas as pd
from numba.core.registry import CPUDispatcher

from vectorbt import _typing as tp


# ############# Checks ############# #


def is_np_array(arg: tp.Any) -> bool:
    """
    检查参数是否为NumPy数组。
    
    参数:
        arg: 要检查的对象
        
    返回:
        如果arg是np.ndarray类型则返回True，否则返回False
    """
    return isinstance(arg, np.ndarray)


def is_series(arg: tp.Any) -> bool:
    """
    检查参数是否为Pandas Series。
    
    参数:
        arg: 要检查的对象
        
    返回:
        如果arg是pd.Series类型则返回True，否则返回False
    """
    return isinstance(arg, pd.Series)


def is_index(arg: tp.Any) -> bool:
    """
    检查参数是否为Pandas Index。
    
    参数:
        arg: 要检查的对象
        
    返回:
        如果arg是pd.Index类型则返回True，否则返回False
    """
    return isinstance(arg, pd.Index)


def is_frame(arg: tp.Any) -> bool:
    """
    检查参数是否为Pandas DataFrame。
    
    参数:
        arg: 要检查的对象
        
    返回:
        如果arg是pd.DataFrame类型则返回True，否则返回False
    """
    return isinstance(arg, pd.DataFrame)


def is_pandas(arg: tp.Any) -> bool:
    """
    检查参数是否为Pandas对象(Series或DataFrame)。
    
    参数:
        arg: 要检查的对象
        
    返回:
        如果arg是pd.Series或pd.DataFrame类型则返回True，否则返回False
    """
    return is_series(arg) or is_frame(arg)


def is_any_array(arg: tp.Any) -> bool:
    """
    检查参数是否为任意数组类型(NumPy数组、Pandas Series或DataFrame)。
    
    参数:
        arg: 要检查的对象
        
    返回:
        如果arg是numpy数组或pandas对象则返回True，否则返回False
    """
    return is_pandas(arg) or isinstance(arg, np.ndarray)


def _to_any_array(arg: tp.ArrayLike) -> tp.AnyArray:
    """
    将任意类数组对象转换为数组。
    
    Pandas对象保持原样不变。
    
    参数:
        arg: 要转换的类数组对象
        
    返回:
        转换后的数组或原始Pandas对象
    """
    if is_any_array(arg):
        return arg
    return np.asarray(arg) # 参考"笔记/vectorbt.ipynb"，须实现 __array__


def is_sequence(arg: tp.Any) -> bool:
    """
    检查参数是否为序列。# 参考"笔记/vectorbt.ipynb"，须实现 __len__ 和 __getitem__
    
    序列需要支持len()和切片操作。
    
    参数:
        arg: 要检查的对象
        
    返回:
        如果arg是序列则返回True，否则返回False
    """
    try:
        len(arg) 
        # 其中 __getitem__ 须支持索引，即
        #     def __getitem__(self, key):
        #         if isinstance(key, slice):
        arg[0:0]    
        return True
    except TypeError:
        return False


def is_iterable(arg: tp.Any) -> bool:
    """
    检查参数是否为可迭代对象。
    
    参数:
        arg: 要检查的对象
        
    返回:
        如果arg是可迭代对象则返回True，否则返回False
    """
    try:
        _ = iter(arg)
        return True
    except TypeError:
        return False


def is_numba_func(arg: tp.Any) -> bool:
    """
    检查参数是否为Numba编译的函数。
    
    根据设置和环境变量决定检查方式：
    1. 如果numba配置禁用了函数类型检查，直接返回True
    2. 如果环境变量NUMBA_DISABLE_JIT=1且不检查函数后缀，直接返回True
    3. 如果环境变量NUMBA_DISABLE_JIT=1且函数名以'_nb'结尾，返回True
    4. 否则检查是否为CPUDispatcher实例
    
    参数:
        arg: 要检查的对象
        
    返回:
        如果arg是Numba编译的函数则返回True，否则返回False
    """
    from vectorbt._settings import settings
    numba_cfg = settings['numba']

    if not numba_cfg['check_func_type']:
        return True
    if 'NUMBA_DISABLE_JIT' in os.environ:
        if os.environ['NUMBA_DISABLE_JIT'] == '1':
            if not numba_cfg['check_func_suffix']:
                return True
            if arg.__name__.endswith('_nb'):
                return True
    return isinstance(arg, CPUDispatcher)


def is_hashable(arg: tp.Any) -> bool:
    """
    检查参数是否可哈希。
    
    不仅检查是否继承自Hashable类，还实际尝试进行哈希操作。
    
    参数:
        arg: 要检查的对象
        
    返回:
        如果arg可哈希则返回True，否则返回False
    """
    if not isinstance(arg, Hashable):
        return False
    # 拥有__hash__()方法不意味着一定可哈希
    try:
        hash(arg)
    except TypeError:
        return False
    return True


def is_index_equal(arg1: tp.Any, arg2: tp.Any, strict: bool = True) -> bool:
    """
    检查两个索引是否相等。
    
    在pd.Index.equals基础上增加了名称测试，但不检查类型。
    
    参数:
        arg1: 第一个索引
        arg2: 第二个索引
        strict: 是否进行严格检查(包括索引名称)
        
    返回:
        如果索引相等则返回True，否则返回False
    """
    if not strict:
        return pd.Index.equals(arg1, arg2)
    if isinstance(arg1, pd.MultiIndex) and isinstance(arg2, pd.MultiIndex):
        if arg1.names != arg2.names:
            return False
    elif isinstance(arg1, pd.MultiIndex) or isinstance(arg2, pd.MultiIndex):
        return False
    else:
        if arg1.name != arg2.name:
            return False
    return pd.Index.equals(arg1, arg2)


def is_default_index(arg: tp.Any) -> bool:
    """
    检查索引是否为基本范围索引(0,1,2,...,len-1)。
    
    参数:
        arg: 要检查的索引
        
    返回:
        如果索引是默认范围索引则返回True，否则返回False
    """
    return is_index_equal(arg, pd.RangeIndex(start=0, stop=len(arg), step=1))


def is_namedtuple(x: tp.Any) -> bool:
    """
    检查对象是否为namedtuple的实例。
    
    参数:
        x: 要检查的对象
        
    返回:
        如果对象是namedtuple实例则返回True，否则返回False
    """
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def func_accepts_arg(func: tp.Callable, arg_name: str, arg_kind: tp.Optional[tp.MaybeTuple[int]] = None) -> bool:
    """
    检查函数是否接受指定名称和类型的参数。
    
    参数:
        func: 要检查的函数
        arg_name: 参数名称
        arg_kind: 参数类型(可选)
            - None: 检查所有普通参数
            - 整数: 检查特定类型参数(0=位置参数, 1=位置或关键字参数, 2=可变位置参数, 3=仅关键字参数, 4=可变关键字参数)
            - 整数元组: 检查多种类型参数
            
    返回:
        如果函数接受指定参数则返回True，否则返回False
    """
    sig = signature(func)
    if arg_kind is not None and isinstance(arg_kind, int):
        arg_kind = (arg_kind,)
    if arg_kind is None:
        if arg_name.startswith('**'):
            return arg_name[2:] in [
                p.name for p in sig.parameters.values()
                if p.kind == p.VAR_KEYWORD
            ]
        if arg_name.startswith('*'):
            return arg_name[1:] in [
                p.name for p in sig.parameters.values()
                if p.kind == p.VAR_POSITIONAL
            ]
        return arg_name in [
            p.name for p in sig.parameters.values()
            if p.kind != p.VAR_POSITIONAL and p.kind != p.VAR_KEYWORD
        ]
    return arg_name in [
        p.name for p in sig.parameters.values()
        if p.kind in arg_kind
    ]


def is_equal(arg1: tp.Any, arg2: tp.Any,
             equality_func: tp.Callable[[tp.Any, tp.Any], bool] = lambda x, y: x == y) -> bool:
    """
    检查两个对象是否相等。
    
    使用指定的相等性函数进行检查，并捕获可能的异常。
    
    参数:
        arg1: 第一个对象
        arg2: 第二个对象
        equality_func: 用于比较相等性的函数，默认使用==操作符
        
    返回:
        如果对象相等则返回True，否则返回False
    """
    try:
        return equality_func(arg1, arg2)
    except:
        pass
    return False


def is_deep_equal(arg1: tp.Any, arg2: tp.Any, check_exact: bool = False, **kwargs) -> bool:
    """
    检查两个对象是否完全相等(深度检查)。
    
    可以比较各种数据类型，包括Pandas对象、NumPy数组、嵌套容器等。
    
    参数:
        arg1: 第一个对象
        arg2: 第二个对象
        check_exact: 是否进行精确检查（对于浮点数）
        **kwargs: 传递给各种断言方法的额外参数
        
    返回:
        如果对象完全相等则返回True，否则返回False
    """
    def _select_kwargs(_method, _kwargs):
        __kwargs = dict()
        if len(kwargs) > 0:
            for k, v in _kwargs.items():
                if func_accepts_arg(_method, k):
                    __kwargs[k] = v
        return __kwargs

    def _check_array(assert_method):
        __kwargs = _select_kwargs(assert_method, kwargs)
        safe_assert(arg1.dtype == arg2.dtype)
        if arg1.dtype.fields is not None:
            for field in arg1.dtype.names:
                assert_method(arg1[field], arg2[field], **__kwargs)
        else:
            assert_method(arg1, arg2, **__kwargs)

    try:
        safe_assert(type(arg1) == type(arg2))
        if isinstance(arg1, pd.Series):
            _kwargs = _select_kwargs(pd.testing.assert_series_equal, kwargs)
            pd.testing.assert_series_equal(arg1, arg2, check_exact=check_exact, **_kwargs)
        elif isinstance(arg1, pd.DataFrame):
            _kwargs = _select_kwargs(pd.testing.assert_frame_equal, kwargs)
            pd.testing.assert_frame_equal(arg1, arg2, check_exact=check_exact, **_kwargs)
        elif isinstance(arg1, pd.Index):
            _kwargs = _select_kwargs(pd.testing.assert_index_equal, kwargs)
            pd.testing.assert_index_equal(arg1, arg2, check_exact=check_exact, **_kwargs)
        elif isinstance(arg1, np.ndarray):
            try:
                _check_array(np.testing.assert_array_equal)
            except:
                if check_exact:
                    return False
                _check_array(np.testing.assert_allclose)
        else:
            if isinstance(arg1, (tuple, list)):
                for i in range(len(arg1)):
                    safe_assert(is_deep_equal(arg1[i], arg2[i], **kwargs))
            elif isinstance(arg1, dict):
                for k in arg1.keys():
                    safe_assert(is_deep_equal(arg1[k], arg2[k], **kwargs))
            else:
                try:
                    if arg1 == arg2:
                        return True
                except:
                    pass
                try:
                    _kwargs = _select_kwargs(dill.dumps, kwargs)
                    if dill.dumps(arg1, **_kwargs) == dill.dumps(arg2, **_kwargs):
                        return True
                except:
                    pass
                return False
    except:
        return False
    return True


def is_subclass_of(arg: tp.Any, types: tp.MaybeTuple[tp.Union[tp.Type, str]]) -> bool:
    """
    检查参数是否为指定类型的子类。
    
    参数:
        arg: 要检查的类
        types: 一个或多个类型或类型名称字符串
        
    返回:
        如果arg是types的子类则返回True，否则返回False
    """
    if isinstance(types, type):
        return issubclass(arg, types)
    if isinstance(types, str):
        for base_t in getmro(arg):
            if str(base_t) == types or base_t.__name__ == types:
                return True
    if isinstance(types, tuple):
        for t in types:
            if is_subclass_of(arg, t):
                return True
    return False


def is_instance_of(arg: tp.Any, types: tp.MaybeTuple[tp.Union[tp.Type, str]]) -> bool:
    """
    检查参数是否为指定类型的实例。
    
    参数:
        arg: 要检查的对象
        types: 一个或多个类型或类型名称字符串
        
    返回:
        如果arg是types的实例则返回True，否则返回False
    """
    return is_subclass_of(type(arg), types)


def is_mapping(arg: tp.Any) -> bool:
    """
    检查参数是否为映射类型。
    
    参数:
        arg: 要检查的对象
        
    返回:
        如果arg是映射类型则返回True，否则返回False
    """
    return isinstance(arg, Mapping)


def is_mapping_like(arg: tp.Any) -> bool:
    """
    检查参数是否为类映射对象。
    
    类映射对象包括映射、Series、Index和命名元组。
    
    参数:
        arg: 要检查的对象
        
    返回:
        如果arg是类映射对象则返回True，否则返回False
    """
    return is_mapping(arg) or is_series(arg) or is_index(arg) or is_namedtuple(arg)


def is_valid_variable_name(arg: str) -> bool:
    """
    检查参数是否为有效的变量名。
    
    有效的变量名必须是合法的标识符且不是Python关键字。
    
    参数:
        arg: 要检查的字符串
        
    返回:
        如果arg是有效变量名则返回True，否则返回False
    """
    return arg.isidentifier() and not iskeyword(arg)


# ############# Asserts ############# #

def safe_assert(arg: tp.Any, msg: tp.Optional[str] = None) -> None:
    """
    安全断言函数，当条件为假时抛出AssertionError。
    
    参数:
        arg: 要断言的条件
        msg: 断言失败时的错误消息
        
    异常:
        AssertionError: 当条件为假时
    """
    if not arg:
        raise AssertionError(msg)


def assert_in(arg1: tp.Any, arg2: tp.Sequence) -> None:
    """
    断言第一个参数在第二个参数中，否则抛出异常。
    
    参数:
        arg1: 要检查的元素
        arg2: 要检查的序列
        
    异常:
        AssertionError: 当arg1不在arg2中时
    """
    if arg1 not in arg2:
        raise AssertionError(f"{arg1} not found in {arg2}")


def assert_numba_func(func: tp.Callable) -> None:
    """
    断言函数是Numba编译的，否则抛出异常。
    
    参数:
        func: 要检查的函数
        
    异常:
        AssertionError: 当func不是Numba编译的函数时
    """
    if not is_numba_func(func):
        raise AssertionError(f"Function {func} must be Numba compiled")


def assert_not_none(arg: tp.Any) -> None:
    """
    断言参数不为None，否则抛出异常。
    
    参数:
        arg: 要检查的对象
        
    异常:
        AssertionError: 当arg为None时
    """
    if arg is None:
        raise AssertionError(f"Argument cannot be None")


def assert_instance_of(arg: tp.Any, types: tp.MaybeTuple[tp.Type]) -> None:
    """
    断言参数为指定类型的实例，否则抛出异常。
    
    参数:
        arg: 要检查的对象
        types: 一个或多个类型
        
    异常:
        AssertionError: 当arg不是types的实例时
    """
    if not is_instance_of(arg, types):
        if isinstance(types, tuple):
            raise AssertionError(f"Type must be one of {types}, not {type(arg)}")
        else:
            raise AssertionError(f"Type must be {types}, not {type(arg)}")


def assert_subclass_of(arg: tp.Type, classes: tp.MaybeTuple[tp.Type]) -> None:
    """
    断言参数为指定类的子类，否则抛出异常。
    
    参数:
        arg: 要检查的类
        classes: 一个或多个类
        
    异常:
        AssertionError: 当arg不是classes的子类时
    """
    if not is_subclass_of(arg, classes):
        if isinstance(classes, tuple):
            raise AssertionError(f"Class must be a subclass of one of {classes}, not {arg}")
        else:
            raise AssertionError(f"Class must be a subclass of {classes}, not {arg}")


def assert_type_equal(arg1: tp.Any, arg2: tp.Any) -> None:
    """
    断言两个参数类型相同，否则抛出异常。
    
    参数:
        arg1: 第一个对象
        arg2: 第二个对象
        
    异常:
        AssertionError: 当arg1和arg2类型不同时
    """
    if type(arg1) != type(arg2):
        raise AssertionError(f"Types {type(arg1)} and {type(arg2)} do not match")


def assert_dtype(arg: tp.ArrayLike, dtype: tp.DTypeLike) -> None:
    """
    断言数组的数据类型为指定类型，否则抛出异常。
    
    参数:
        arg: 要检查的数组
        dtype: 期望的数据类型
        
    异常:
        AssertionError: 当arg的数据类型不等于dtype时
    """
    arg = _to_any_array(arg)
    if isinstance(arg, pd.DataFrame):
        for i, col_dtype in enumerate(arg.dtypes):
            if col_dtype != dtype:
                raise AssertionError(f"Data type of column {i} must be {dtype}, not {col_dtype}")
    else:
        if arg.dtype != dtype:
            raise AssertionError(f"Data type must be {dtype}, not {arg.dtype}")


def assert_subdtype(arg: tp.ArrayLike, dtype: tp.DTypeLike) -> None:
    """
    断言数组的数据类型为指定类型的子类型，否则抛出异常。
    
    参数:
        arg: 要检查的数组
        dtype: 期望的数据类型
        
    异常:
        AssertionError: 当arg的数据类型不是dtype的子类型时
    """
    arg = _to_any_array(arg)
    if isinstance(arg, pd.DataFrame):
        for i, col_dtype in enumerate(arg.dtypes):
            if not np.issubdtype(col_dtype, dtype):
                raise AssertionError(f"Data type of column {i} must be {dtype}, not {col_dtype}")
    else:
        if not np.issubdtype(arg.dtype, dtype):
            raise AssertionError(f"Data type must be {dtype}, not {arg.dtype}")


def assert_dtype_equal(arg1: tp.ArrayLike, arg2: tp.ArrayLike) -> None:
    """
    断言两个数组的数据类型相同，否则抛出异常。
    
    参数:
        arg1: 第一个数组
        arg2: 第二个数组
        
    异常:
        AssertionError: 当arg1和arg2的数据类型不同时
    """
    arg1 = _to_any_array(arg1)
    arg2 = _to_any_array(arg2)
    if isinstance(arg1, pd.DataFrame):
        dtypes1 = arg1.dtypes.to_numpy()
    else:
        dtypes1 = np.asarray([arg1.dtype])
    if isinstance(arg2, pd.DataFrame):
        dtypes2 = arg2.dtypes.to_numpy()
    else:
        dtypes2 = np.asarray([arg2.dtype])
    if len(dtypes1) == len(dtypes2):
        if (dtypes1 == dtypes2).all():
            return
    elif len(np.unique(dtypes1)) == 1 and len(np.unique(dtypes2)) == 1:
        if np.all(np.unique(dtypes1) == np.unique(dtypes2)):
            return
    raise AssertionError(f"Data types {dtypes1} and {dtypes2} do not match")


def assert_ndim(arg: tp.ArrayLike, ndims: tp.MaybeTuple[int]) -> None:
    """
    断言数组的维度为指定值，否则抛出异常。
    
    参数:
        arg: 要检查的数组
        ndims: 期望的维度或维度元组
        
    异常:
        AssertionError: 当arg的维度不符合要求时
    """
    arg = _to_any_array(arg)
    if isinstance(ndims, tuple):
        if arg.ndim not in ndims:
            raise AssertionError(f"Number of dimensions must be one of {ndims}, not {arg.ndim}")
    else:
        if arg.ndim != ndims:
            raise AssertionError(f"Number of dimensions must be {ndims}, not {arg.ndim}")


def assert_len_equal(arg1: tp.Sized, arg2: tp.Sized) -> None:
    """
    断言两个对象长度相同，否则抛出异常。
    
    不将参数转换为NumPy数组。
    
    参数:
        arg1: 第一个对象
        arg2: 第二个对象
        
    异常:
        AssertionError: 当arg1和arg2长度不同时
    """
    if len(arg1) != len(arg2):
        raise AssertionError(f"Lengths of {arg1} and {arg2} do not match")


def assert_shape_equal(arg1: tp.ArrayLike, arg2: tp.ArrayLike,
                       axis: tp.Optional[tp.Union[int, tp.Tuple[int, int]]] = None) -> None:
    """
    断言两个数组在指定轴上的形状相同，否则抛出异常。
    
    参数:
        arg1: 第一个数组
        arg2: 第二个数组
        axis: 要检查的轴或轴对（可选）
        
    异常:
        AssertionError: 当形状不匹配时
    """
    arg1 = _to_any_array(arg1)
    arg2 = _to_any_array(arg2)
    if axis is None:
        if arg1.shape != arg2.shape:
            raise AssertionError(f"Shapes {arg1.shape} and {arg2.shape} do not match")
    else:
        if isinstance(axis, tuple):
            if arg1.shape[axis[0]] != arg2.shape[axis[1]]:
                raise AssertionError(
                    f"Axis {axis[0]} of {arg1.shape} and axis {axis[1]} of {arg2.shape} do not match")
        else:
            if arg1.shape[axis] != arg2.shape[axis]:
                raise AssertionError(f"Axis {axis} of {arg1.shape} and {arg2.shape} do not match")


def assert_index_equal(arg1: pd.Index, arg2: pd.Index, **kwargs) -> None:
    """
    断言两个索引相等，否则抛出异常。
    
    参数:
        arg1: 第一个索引
        arg2: 第二个索引
        **kwargs: 传递给is_index_equal的额外参数
        
    异常:
        AssertionError: 当索引不相等时
    """
    if not is_index_equal(arg1, arg2, **kwargs):
        raise AssertionError(f"Indexes {arg1} and {arg2} do not match")


def assert_meta_equal(arg1: tp.ArrayLike, arg2: tp.ArrayLike) -> None:
    """
    断言两个数组的元数据相等，否则抛出异常。
    
    元数据包括类型、形状和索引（对于Pandas对象）。
    
    参数:
        arg1: 第一个数组
        arg2: 第二个数组
        
    异常:
        AssertionError: 当元数据不匹配时
    """
    arg1 = _to_any_array(arg1)
    arg2 = _to_any_array(arg2)
    assert_type_equal(arg1, arg2)
    assert_shape_equal(arg1, arg2)
    if is_pandas(arg1) and is_pandas(arg2):
        assert_index_equal(arg1.index, arg2.index)
        if is_frame(arg1) and is_frame(arg2):
            assert_index_equal(arg1.columns, arg2.columns)


def assert_array_equal(arg1: tp.ArrayLike, arg2: tp.ArrayLike) -> None:
    """
    断言两个数组的元数据和值完全相等，否则抛出异常。
    
    参数:
        arg1: 第一个数组
        arg2: 第二个数组
        
    异常:
        AssertionError: 当数组不匹配时
    """
    arg1 = _to_any_array(arg1)
    arg2 = _to_any_array(arg2)
    assert_meta_equal(arg1, arg2)
    if is_pandas(arg1) and is_pandas(arg2):
        if arg1.equals(arg2):
            return
    elif not is_pandas(arg1) and not is_pandas(arg2):
        if np.array_equal(arg1, arg2):
            return
    raise AssertionError(f"Arrays {arg1} and {arg2} do not match")


def assert_level_not_exists(arg: pd.Index, level_name: str) -> None:
    """
    断言索引不包含指定的级别名称，否则抛出异常。
    
    参数:
        arg: 要检查的索引
        level_name: 级别名称
        
    异常:
        AssertionError: 当索引包含指定级别名称时
    """
    if isinstance(arg, pd.MultiIndex):
        names = arg.names
    else:
        names = [arg.name]
    if level_name in names:
        raise AssertionError(f"Level {level_name} already exists in {names}")


def assert_equal(arg1: tp.Any, arg2: tp.Any, deep: bool = False) -> None:
    """
    断言两个对象相等，否则抛出异常。
    
    参数:
        arg1: 第一个对象
        arg2: 第二个对象
        deep: 是否进行深度检查
        
    异常:
        AssertionError: 当对象不相等时
    """
    if deep:
        if not is_deep_equal(arg1, arg2):
            raise AssertionError(f"{arg1} and {arg2} do not match (deep check)")
    else:
        if not is_equal(arg1, arg2):
            raise AssertionError(f"{arg1} and {arg2} do not match")


def assert_dict_valid(arg: tp.DictLike, lvl_keys: tp.Sequence[tp.MaybeSequence[str]]) -> None:
    """
    断言字典的键在指定的有效键列表中，否则抛出异常。
    
    lvl_keys应该是列表的列表，每个对应字典中的一个层级。
    
    参数:
        arg: 要检查的字典
        lvl_keys: 有效键的层级列表
        
    异常:
        AssertionError: 当字典包含无效键时
    """
    if arg is None:
        arg = {}
    if len(lvl_keys) == 0:
        return
    if isinstance(lvl_keys[0], str):
        lvl_keys = [lvl_keys]
    set1 = set(arg.keys())
    set2 = set(lvl_keys[0])
    if not set1.issubset(set2):
        raise AssertionError(f"Keys {set1.difference(set2)} are not recognized. Possible keys are {set2}.")
    for k, v in arg.items():
        if isinstance(v, dict):
            assert_dict_valid(v, lvl_keys[1:])


def assert_dict_sequence_valid(arg: tp.DictLikeSequence, lvl_keys: tp.Sequence[tp.MaybeSequence[str]]) -> None:
    """
    断言字典序列中的所有字典键都有效，否则抛出异常。
    
    参数:
        arg: 要检查的字典或字典序列
        lvl_keys: 有效键的层级列表
        
    异常:
        AssertionError: 当任何字典包含无效键时
    """
    if arg is None:
        arg = {}
    if isinstance(arg, dict):
        assert_dict_valid(arg, lvl_keys)
    else:
        for _arg in arg:
            assert_dict_valid(_arg, lvl_keys)


def assert_sequence(arg: tp.Any) -> None:
    """
    断言参数是序列，否则抛出异常。
    
    参数:
        arg: 要检查的对象
        
    异常:
        ValueError: 当参数不是序列时
    """
    if not is_sequence(arg):
        raise ValueError(f"{arg} must be a sequence")


def assert_iterable(arg: tp.Any) -> None:
    """
    断言参数是可迭代对象，否则抛出异常。
    
    参数:
        arg: 要检查的对象
        
    异常:
        ValueError: 当参数不是可迭代对象时
    """
    if not is_iterable(arg):
        raise ValueError(f"{arg} must be an iterable")