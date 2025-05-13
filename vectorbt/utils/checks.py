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
    # 函数文档字符串：详细解释函数的作用、参数、返回值和功能
    """
    检查两个对象是否完全相等(深度检查)。
    
    此函数用于递归地比较两个 Python 对象，包括基本类型、容器（列表、字典、元组）
    以及 NumPy 数组和 Pandas 对象（Series, DataFrame, Index）。
    它会检查对象的内容、结构以及（对于数组和 Pandas 对象）数据类型和索引。
    
    参数:
        arg1: tp.Any
            要比较的第一个对象。
        arg2: tp.Any
            要比较的第二个对象。
        check_exact: bool, default=False
            一个布尔标志，指示对于浮点数比较是否需要精确相等。
            如果为 True，则使用严格相等检查 (例如 np.testing.assert_array_equal)。
            如果为 False，则对于浮点数可能会使用带容差的检查 (例如 np.testing.assert_allclose)。
        **kwargs: Any
            额外的关键字参数，这些参数会被传递给内部调用的特定类型比较函数，
            例如 pd.testing.assert_series_equal, np.testing.assert_allclose 等。
            例如，可以传递 `rtol` 和 `atol` 给 `assert_allclose`。
        
    返回:
        bool
            如果两个对象完全相等（根据深度检查的定义）则返回 True，否则返回 False。
            如果在比较过程中发生任何异常，也返回 False。
    """
    # 内部函数：_select_kwargs 用于从外部kwargs中筛选出适用于特定方法的参数
    def _select_kwargs(_method, _kwargs):
        """
        从外部传入的_kwargs字典中，筛选出包含在_method函数参数列表中的参数。
        这避免了向内部调用的比较函数传递它们不支持的参数。
        """
        __kwargs = dict()
        # 检查外部kwargs字典是否非空
        if len(kwargs) > 0:
            # 遍历外部kwargs中的所有键值对
            for k, v in _kwargs.items():
                # 使用func_accepts_arg检查当前方法(_method)是否接受参数k
                if func_accepts_arg(_method, k):
                    # 如果接受，则将该参数及其值添加到__kwargs字典中
                    __kwargs[k] = v
        # 返回筛选后的参数字典
        return __kwargs

    # 内部函数：_check_array 用于处理NumPy数组的比较逻辑
    def _check_array(assert_method):
        """
        使用指定的断言方法对两个NumPy数组进行比较。
        特别处理了结构化数组(structured arrays)的情况。
        """
        # 调用_select_kwargs筛选出适用于当前断言方法(assert_method)的参数
        __kwargs = _select_kwargs(assert_method, kwargs)
        # 断言两个数组的数据类型(dtype)必须相同，否则抛出AssertionError
        safe_assert(arg1.dtype == arg2.dtype)
        # 检查第一个数组的数据类型是否有字段(fields)信息，即是否为结构化数组
        if arg1.dtype.fields is not None:   # 处理结构化数组的情况
            # 如果是结构化数组，遍历所有字段名称
            for field in arg1.dtype.names:
                # 对每个字段对应的数据（列）递归或分字段地调用assert_method进行比较
                assert_method(arg1[field], arg2[field], **__kwargs)
        # 如果不是结构化数组
        else:
            # 直接对整个数组调用assert_method进行比较
            assert_method(arg1, arg2, **__kwargs)

    # 开始主体的try块，用于捕获比较过程中可能发生的各种异常
    try:
        # 断言两个对象的类型必须完全相同，否则抛出AssertionError
        safe_assert(type(arg1) == type(arg2))
        # 检查第一个对象是否是Pandas Series
        if isinstance(arg1, pd.Series):
            # 如果是Series，筛选出适用于pd.testing.assert_series_equal的参数
            _kwargs = _select_kwargs(pd.testing.assert_series_equal, kwargs)
            # 调用Pandas的assert_series_equal进行比较。如果不同会抛出AssertionError。
            # check_exact 参数直接传递，筛选出的额外参数通过**_kwargs传递。
            pd.testing.assert_series_equal(arg1, arg2, check_exact=check_exact, **_kwargs)
        # 检查第一个对象是否是Pandas DataFrame
        elif isinstance(arg1, pd.DataFrame):
            # 如果是DataFrame，筛选出适用于pd.testing.assert_frame_equal的参数
            _kwargs = _select_kwargs(pd.testing.assert_frame_equal, kwargs)
            # 调用Pandas的assert_frame_equal进行比较
            pd.testing.assert_frame_equal(arg1, arg2, check_exact=check_exact, **_kwargs)
        # 检查第一个对象是否是Pandas Index
        elif isinstance(arg1, pd.Index):
            # 如果是Index，筛选出适用于pd.testing.assert_index_equal的参数
            _kwargs = _select_kwargs(pd.testing.assert_index_equal, kwargs)
            # 调用Pandas的assert_index_equal进行比较
            pd.testing.assert_index_equal(arg1, arg2, check_exact=check_exact, **_kwargs)
        # 检查第一个对象是否是NumPy ndarray (在排除了Pandas对象之后)
        elif isinstance(arg1, np.ndarray):
            # 尝试使用np.testing.assert_array_equal进行严格相等性检查 (不带容差)
            try:
                _check_array(np.testing.assert_array_equal) # 调用内部函数_check_array进行比较
            # 如果np.testing.assert_array_equal抛出异常 (例如，浮点数不完全相等)
            except:
                # 检查是否要求精确匹配 (check_exact)
                if check_exact:
                    # 如果要求精确匹配但严格检查失败，则直接返回False
                    return False
                # 如果不要求精确匹配，则尝试使用np.testing.assert_allclose进行带容差的检查 (主要用于浮点数)
                _check_array(np.testing.assert_allclose) # 调用_check_array进行比较
        # 处理其他类型的对象 (非Pandas, 非NumPy)
        else:
            # 检查对象是否是元组或列表 (序列容器)
            if isinstance(arg1, (tuple, list)):
                # 断言两个序列的长度相同
                safe_assert(len(arg1) == len(arg2))
                # 遍历序列中的每个元素
                for i in range(len(arg1)):
                    # 对对应位置的元素递归调用is_deep_equal进行深度比较
                    # 如果任何一对元素不相等，safe_assert会失败并抛出异常
                    safe_assert(is_deep_equal(arg1[i], arg2[i], **kwargs))
            # 检查对象是否是字典 (映射容器)
            elif isinstance(arg1, dict):
                # 断言两个字典的键集合相同
                safe_assert(set(arg1.keys()) == set(arg2.keys()))
                # 遍历第一个字典的所有键
                for k in arg1.keys():
                    # 对两个字典中对应键的值递归调用is_deep_equal进行深度比较
                    # 如果任何一对值不相等，safe_assert会失败并抛出异常
                    safe_assert(is_deep_equal(arg1[k], arg2[k], **kwargs))
            # 处理既不是Pandas/NumPy对象，也不是常见容器的类型
            else:
                # 尝试使用Python的==操作符进行相等性比较
                try:
                    if arg1 == arg2:
                        # 如果相等，则返回True
                        return True
                # 如果==操作符引发异常 (例如，对象不支持==比较)
                except:
                    pass # 忽略异常，继续尝试其他比较方式
                # 尝试使用dill库将对象序列化为字节串，然后比较字节串是否相等
                # 这是一种判断复杂对象（如函数、自定义类实例）是否相等的方法
                try:
                    # 筛选出适用于dill.dumps的参数
                    _kwargs = _select_kwargs(dill.dumps, kwargs)
                    # 序列化两个对象并比较字节串
                    if dill.dumps(arg1, **_kwargs) == dill.dumps(arg2, **_kwargs):
                        # 如果字节串相等，则认为对象深度相等，返回True
                        return True
                # 如果dill.dumps引发异常 (例如，对象不可序列化)
                except:
                    pass # 忽略异常
                # 如果以上所有比较方式都失败，则认为对象不相等，返回False
                return False
    # 捕获try块中发生的任何异常 (包括内部safe_assert和Pandas/NumPy断言失败抛出的AssertionError)
    except:
        # 如果发生异常，说明对象不相等，返回False
        return False
    # 如果try块中的所有断言和比较都成功完成，说明对象深度相等，返回True
    return True


def is_subclass_of(arg: tp.Any, types: tp.MaybeTuple[tp.Union[tp.Type, str]]) -> bool:
    """
    检查一个对象或类是否为指定类型或类型集合的子类。
    
    此函数用于验证给定的 `arg` 是否是 `types` 参数所指定的类、类名字符串
    或它们的元组中任意一个的子类。它处理了类型对象本身和类型名称字符串两种输入形式。
    
    参数:
        arg: tp.Any
            要检查的对象或类。函数将检查其类型 (type(arg)) 是否为子类。
        types: tp.MaybeTuple[tp.Union[tp.Type, str]]
            一个或多个目标类型或类型名称字符串。
            - 如果是单个类型 (如 `int`): 检查 arg 的类型是否是该类型的子类。
            - 如果是单个字符串 (如 `'int'`): 检查 arg 的类型是否是该字符串对应类型的子类。
            - 如果是元组 (如 `(int, 'str', list)`): 检查 arg 的类型是否是元组中任一类型或字符串对应类型的子类。
        
    返回:
        bool
            如果 arg 的类型是 types 中任一类型或其子类，则返回 True；否则返回 False。
    """
    # 检查传入的types是否是一个单独的类型对象而非实例 (例如 int, list, str, MyClass)
    if isinstance(types, type):
        return issubclass(arg, types)
    # 检查传入的types参数是否是一个单独的字符串实例 (例如 'int', 'list', 'MyClass')
    if isinstance(types, str):
        # 因为无法直接从字符串实例转换到其对应的类型
        # 所以需要遍历arg的类型及其所有基类
        # getmro(arg) 返回arg类型的Method Resolution Order (MRO) 元组，包含arg类型本身及其所有基类
        for base_t in getmro(type(arg)):
            # 将当前遍历到的基类转换为字符串形式进行比较
            # str(base_t) 格式类似 "<class 'module.ClassName'>"
            # base_t.__name__ 格式类似 "ClassName"
            # 如果字符串形式匹配，说明arg的类型是该字符串对应类型的子类（包括自身）
            if str(base_t) == types or base_t.__name__ == types:
                return True
    # 检查传入的types参数是否是一个元组实例 (例如 (int, 'str', list))
    if isinstance(types, tuple):
        # 如果是元组实例，遍历元组中的每一个元素
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
    # 函数文档字符串：详细解释函数的作用、参数、返回值和功能
    """
    断言两个类数组对象的数据类型相同，否则抛出AssertionError。

    此函数用于检查两个输入的类数组对象 (`arg1` 和 `arg2`) 是否具有相同的数据类型。
    它支持比较 NumPy 数组、Pandas Series 和 Pandas DataFrame。
    对于 Pandas DataFrame，它会比较两者的列数据类型序列是否一致。

    参数:
        arg1: tp.ArrayLike
            要检查的第一个类数组对象 (可以是 NumPy 数组、Pandas Series 或 DataFrame)。
        arg2: tp.ArrayLike
            要检查的第二个类数组对象 (可以是 NumPy 数组、Pandas Series 或 DataFrame)。

    返回:
        None
            如果两个对象的数据类型匹配，函数正常返回，不返回任何值。

    异常:
        AssertionError:
            当两个类数组对象的数据类型不相同时抛出此异常。
            错误消息会指示不匹配的数据类型。
    """
    # 将第一个参数转换为NumPy数组或保持为Pandas对象
    arg1 = _to_any_array(arg1)
    # 将第二个参数转换为NumPy数组或保持为Pandas对象
    arg2 = _to_any_array(arg2)

    # 检查第一个参数是否是Pandas DataFrame
    if isinstance(arg1, pd.DataFrame):
        # 如果是DataFrame，提取其所有列的数据类型并转换为NumPy数组
        dtypes1 = arg1.dtypes.to_numpy()
    # 如果第一个参数不是DataFrame
    else:
        # 提取其单一数据类型并包装在NumPy数组中
        dtypes1 = np.asarray([arg1.dtype])

    if isinstance(arg2, pd.DataFrame):
        dtypes2 = arg2.dtypes.to_numpy()
    else:
        dtypes2 = np.asarray([arg2.dtype])

    # 检查两个数据类型数组的长度是否相同 (即列数是否相同，或都是单一数组/Series)
    if len(dtypes1) == len(dtypes2):
        # 如果长度相同，比较两个数据类型数组的元素是否完全相等
        if (dtypes1 == dtypes2).all():
            # 如果所有对应位置的数据类型都相同，断言通过，函数返回
            return
    # 如果数据类型数组长度不同 (例如比较Series和DataFrame，或列数不同的DataFrame)
    elif len(np.unique(dtypes1)) == 1 and len(np.unique(dtypes2)) == 1:
        # 如果两个数组都只包含一个唯一数据类型，并且这两个唯一数据类型相同
        if np.all(np.unique(dtypes1) == np.unique(dtypes2)):
            # 断言通过，函数返回
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
    # 函数文档字符串：详细解释函数的作用、参数、返回值和功能
    """
    断言两个类数组对象（NumPy数组或Pandas对象）的元数据相等，否则抛出AssertionError。

    元数据包括对象的类型、形状以及（对于 Pandas 对象）索引。
    这个函数不会比较对象中的实际数据值，只关注结构性信息。

    参数:
        arg1: tp.ArrayLike
            要检查的第一个类数组对象 (可以是 NumPy 数组、Pandas Series 或 DataFrame)。
        arg2: tp.ArrayLike
            要检查的第二个类数组对象 (可以是 NumPy 数组、Pandas Series 或 DataFrame)。

    返回:
        None
            如果两个对象的元数据匹配，函数正常返回，不返回任何值。

    异常:
        AssertionError:
            当两个类数组对象的元数据不相同时抛出此异常。
            具体原因（类型不匹配、形状不匹配、索引不匹配）会在错误消息中说明。
    """
    # 将第一个参数转换为 NumPy 数组或保持为 Pandas 对象，以便统一处理
    arg1 = _to_any_array(arg1)
    # 将第二个参数转换为 NumPy 数组或保持为 Pandas 对象
    arg2 = _to_any_array(arg2)
    # 断言两个对象的类型必须完全相同，否则抛出 AssertionError
    assert_type_equal(arg1, arg2)
    # 断言两个对象在所有轴上的形状必须完全相同，否则抛出 AssertionError
    # assert_shape_equal 函数会根据需要检查所有维度的大小
    assert_shape_equal(arg1, arg2)
    # 检查两个对象是否都是 Pandas 对象 (Series 或 DataFrame)
    if is_pandas(arg1) and is_pandas(arg2):
        # 如果都是 Pandas 对象，断言它们的索引（行索引）相等
        # is_index_equal 函数会检查索引的值和名称是否匹配
        assert_index_equal(arg1.index, arg2.index)
        # 进一步检查，如果两个对象都是 Pandas DataFrame 类型
        if is_frame(arg1) and is_frame(arg2):
            # 断言它们的列索引相等
            assert_index_equal(arg1.columns, arg2.columns)
    # 函数执行到此，表示所有元数据检查都已通过，无需返回值 (返回 None)


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
    # 函数文档字符串：详细解释函数的作用、参数、返回值和功能
    """
    断言字典的键在指定的有效键列表中，否则抛出AssertionError。

    此函数递归地检查给定的字典 (`arg`) 及其嵌套的字典，
    确保在每个层级上使用的键都包含在对应的有效键列表 (`lvl_keys`) 中。
    这对于验证配置字典或其他结构化字典的合法性非常有用。

    `lvl_keys` 是一个列表的列表，每个内部列表包含一个层级允许的有效键。
    例如：`[['level1_key_a', 'level1_key_b'], ['level2_key_x', 'level2_key_y']]`

    参数:
        arg: tp.DictLike
            要检查的字典或类似字典的对象（如 None）。
            如果为 None，则被视为空字典。
        lvl_keys: tp.Sequence[tp.MaybeSequence[str]]
            一个序列，其中每个元素代表字典一个层级允许的有效键。
            每个元素本身是一个序列 (如列表或元组) 或单个字符串 (表示该层级只有一个允许的键)。

    返回:
        None
            如果字典及其嵌套字典的所有键都在对应的有效键列表中，函数正常返回，不返回任何值。

    异常:
        AssertionError:
            当字典或其嵌套字典包含不在对应 `lvl_keys` 层级中定义的键时抛出。
            错误消息会指出未识别的键和该层级允许的有效键。
    """
    # 如果输入的字典参数为 None，将其视为空字典，无需进一步检查
    if arg is None:
        arg = {}
    # 如果有效键层级列表为空，说明没有限制，任何字典都是有效的，直接返回
    if len(lvl_keys) == 0:
        return
    # 检查 lvl_keys 的第一个元素是否是字符串。如果是，表示 lvl_keys 只有一层，将其包装成列表的列表
    # 例如，将 ['key1', 'key2'] 转换为 [['key1', 'key2']]
    if isinstance(lvl_keys[0], str):
        lvl_keys = [lvl_keys]
    # 获取待检查字典 arg 的所有键，并转换为集合以便进行集合操作
    set1 = set(arg.keys())
    # 获取当前层级 (lvl_keys[0]) 的有效键列表，并转换为集合
    set2 = set(lvl_keys[0])
    # 检查字典 arg 的键集合是否是当前层级有效键集合的子集
    # 如果不是，说明 arg 中包含无效键
    if not set1.issubset(set2):
        # 如果存在无效键，计算出无效键的集合，并抛出 AssertionError
        # 错误消息指明了哪些键是未识别的，以及该层级所有可能的有效键
        raise AssertionError(f"Keys {set1.difference(set2)} are not recognized. Possible keys are {set2}.")
    # 如果当前层级的所有键都有效，则遍历字典 arg 的每个键值对
    for k, v in arg.items():
        # 对于每个值 v，检查它是否是一个字典类型 (包括 Config 等子类，但不包括 atomic_dict)
        # atomic_dict 在 config.py 中定义，用于在合并时被视为单个值处理，不应被递归检查其内部键
        if isinstance(v, dict):
            # 如果 v 是一个字典，则递归调用 assert_dict_valid
            # 对嵌套字典 v 及其剩余的有效键层级 (lvl_keys[1:]) 进行验证
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