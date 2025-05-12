# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
vectorbt中使用的通用类型定义。

此模块定义了vectorbt库中使用的所有类型提示(type hints)。使用类型提示有助于：
1. 提高代码可读性和可维护性
2. 帮助IDE提供更准确的代码补全和错误检查
3. 支持静态类型检查工具(如mypy)进行类型验证
4. 生成更完善的API文档

类型定义分为几大类：
- 基本类型（泛型、标量、序列等）
- 数组相关类型（NumPy数组、Pandas数据结构等）
- 索引和分组类型
- 日期时间相关类型
- 配置和数据类型
- 可调用函数类型（用于回调、自定义操作等）
- 特定领域类型（信号处理、记录处理、指标计算等）

这些类型定义在vectorbt的整个代码库中被广泛使用，确保类型安全和一致性。
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame as Frame, Index
from typing import *
from datetime import datetime, timedelta, tzinfo
from mypy_extensions import VarArg, KwArg
from pandas.tseries.offsets import DateOffset
from plotly.graph_objects import Figure, FigureWidget
from plotly.basedatatypes import BaseFigure, BaseTraceType
from numba.core.registry import CPUDispatcher
from numba.typed import List as NumbaList
from pathlib import Path

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

# ===== 泛型类型 =====
# 定义通用的类型变量，用于泛型编程
T = TypeVar("T")  # 定义一个通用类型变量T，可以表示任何类型
F = TypeVar("F", bound=Callable[..., Any])  # 定义一个表示可调用对象（函数）的类型变量F

# ===== 标量类型 =====
# 定义各种标量类型及其组合
Scalar = Union[str, float, int, complex, bool, object, np.generic]  # 标量类型，包括所有基本数据类型和numpy的标量类型
Number = Union[int, float, complex, np.number, np.bool_]  # 数值类型，包括所有数值数据类型和numpy的数值类型
Int = Union[int, np.integer]  # 整数类型，包括Python整数和numpy的整数类型
Float = Union[float, np.floating]  # 浮点数类型，包括Python浮点数和numpy的浮点数类型
IntFloat = Union[Int, Float]  # 整数或浮点数类型的组合类型

# ===== 基本序列类型 =====
# 定义可能是单个元素或序列的组合类型
MaybeTuple = Union[T, Tuple[T, ...]]  # 可能是单个元素T或元素类型为T的元组
MaybeList = Union[T, List[T]]  # 可能是单个元素T或元素类型为T的列表
TupleList = Union[List[T], Tuple[T, ...]]  # 可能是元素类型为T的列表或元组
MaybeTupleList = Union[T, List[T], Tuple[T, ...]]  # 可能是单个元素T、元素类型为T的列表或元组
MaybeIterable = Union[T, Iterable[T]]  # 可能是单个元素T或元素类型为T的可迭代对象
MaybeSequence = Union[T, Sequence[T]]  # 可能是单个元素T或元素类型为T的序列


# ===== 数组类型 =====
class SupportsArray(Protocol):
    """
    定义支持转换为numpy数组的协议类。
    任何实现了__array__方法的类型都可以被视为支持数组操作。
    这是一个结构化类型，遵循Python的鸭子类型原则。
    """
    def __array__(self) -> np.ndarray: ...  # 必须实现__array__方法返回numpy数组


DTypeLike = Any  # 表示numpy数据类型或可转换为数据类型的对象，如'float64'、np.float64等
PandasDTypeLike = Any  # 表示pandas数据类型或可转换为pandas数据类型的对象
Shape = Tuple[int, ...]  # 表示numpy数组的形状，如(2, 3)表示2行3列
RelaxedShape = Union[int, Shape]  # 宽松的形状表示，可以是单个整数或形状元组
Array = np.ndarray  # 表示任意维度的numpy数组，可用于任何维度的数据
Array1d = np.ndarray  # 表示一维numpy数组
Array2d = np.ndarray  # 表示二维numpy数组
Array3d = np.ndarray  # 表示三维numpy数组
Record = np.void  # 表示numpy的记录类型（结构化数组的单个元素）
RecordArray = np.ndarray  # 表示numpy的记录数组（结构化数组）
RecArray = np.recarray  # 表示numpy的记录数组类型（提供字段访问语法）
MaybeArray = Union[T, Array]  # 可能是单个元素T或numpy数组
SeriesFrame = Union[Series, Frame]  # 可能是pandas的Series或DataFrame
MaybeSeries = Union[T, Series]  # 可能是单个元素T或pandas的Series
MaybeSeriesFrame = Union[T, Series, Frame]  # 可能是单个元素T、pandas的Series或DataFrame
AnyArray = Union[Array, Series, Frame]  # 任何类型的数组：numpy数组、pandas的Series或DataFrame
AnyArray1d = Union[Array1d, Series]  # 任何类型的一维数组：numpy一维数组或pandas的Series
AnyArray2d = Union[Array2d, Frame]  # 任何类型的二维数组：numpy二维数组或pandas的DataFrame
_ArrayLike = Union[Scalar, Sequence[Scalar], Sequence[Sequence[Any]], SupportsArray]  # 内部使用的类数组类型
ArrayLike = Union[_ArrayLike, Array, Index, Series, Frame]  # 表示可转换为数组的类型，必须经过转换
IndexLike = Union[_ArrayLike, Array1d, Index, Series]  # 表示可转换为索引的类型
ArrayLikeSequence = Union[Sequence[T], Array1d, Index, Series]  # 表示一维数据的序列类型

# ===== 标签类型 =====
Label = Hashable  # 表示可哈希的对象，可用作字典键或索引标签
Labels = ArrayLikeSequence[Label]  # 表示标签序列
Level = Union[str, int]  # 表示多级索引中的层级，可以是字符串名称或整数位置
LevelSequence = Sequence[Level]  # 表示层级序列
MaybeLevelSequence = Union[Level, LevelSequence]  # 可能是单个层级或层级序列

# ===== 日期时间类型 =====
FrequencyLike = Union[str, float, pd.Timedelta, timedelta, np.timedelta64, DateOffset]  # 表示可转换为频率的类型
PandasFrequencyLike = Union[str, pd.Timedelta, timedelta, np.timedelta64, DateOffset]  # 表示可转换为pandas频率的类型
TimezoneLike = Union[None, str, float, timedelta, tzinfo]  # 表示可转换为时区的类型
DatetimeLikeIndex = Union[pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex]  # 表示类似日期时间的索引
DatetimeLike = Union[str, float, pd.Timestamp, np.datetime64, datetime]  # 表示可转换为日期时间的类型


class SupportsTZInfo(Protocol):
    """
    定义支持时区信息的协议类。
    任何具有tzinfo属性的类型都被视为支持时区信息。
    """
    tzinfo: tzinfo  # 必须具有tzinfo属性


# ===== 索引类型 =====
PandasIndexingFunc = Callable[[SeriesFrame], MaybeSeriesFrame]  # 表示pandas索引函数，接收Series或DataFrame，返回一个可能的Series或DataFrame

# ===== 分组类型 =====
GroupByLike = Union[None, bool, MaybeLevelSequence, IndexLike]  # 表示可用于分组的类型
PandasGroupByLike = Union[Label, Labels, Callable, Mapping[Label, Any]]  # 表示可用于pandas分组的类型

# ===== 包装类型 =====
NameIndex = Union[None, Any, Index]  # 表示可用作名称或索引的类型

# ===== 配置类型 =====
DictLike = Union[None, dict]  # 表示类似字典的类型
DictLikeSequence = MaybeSequence[DictLike]  # 表示类似字典的序列
Args = Tuple[Any, ...]  # 表示位置参数元组
ArgsLike = Union[None, Args]  # 表示可能为None的位置参数
Kwargs = Dict[str, Any]  # 表示关键字参数字典
KwargsLike = Union[None, Kwargs]  # 表示可能为None的关键字参数
KwargsLikeSequence = MaybeSequence[KwargsLike]  # 表示关键字参数序列
FileName = Union[str, Path]  # 表示文件名，可以是字符串或Path对象

# ===== 数据类型 =====
Data = Dict[Label, SeriesFrame]  # 表示数据字典，键为标签，值为Series或DataFrame

# ===== 绘图类型 =====
TraceName = Union[str, None]  # 表示图表跟踪名称，可以是字符串或None
TraceNames = MaybeSequence[TraceName]  # 表示图表跟踪名称序列

# ===== 通用函数类型 =====
I = TypeVar("I")  # 定义输入类型变量
R = TypeVar("R")  # 定义返回值类型变量
ApplyFunc = Callable[[int, Array1d, VarArg()], MaybeArray]  # 表示应用函数，接收索引、一维数组和可变参数
RowApplyFunc = Callable[[int, Array1d, VarArg()], MaybeArray]  # 表示行应用函数，接收行索引、一维数组和可变参数
RollApplyFunc = Callable[[int, int, Array1d, VarArg()], Scalar]  # 表示滚动应用函数，接收窗口大小、索引、一维数组和可变参数
RollMatrixApplyFunc = Callable[[int, Array2d, VarArg()], MaybeArray]  # 表示滚动矩阵应用函数，接收索引、二维数组和可变参数
GroupByApplyFunc = Callable[[Array1d, int, Array1d, VarArg()], Scalar]  # 表示分组应用函数，接收分组数组、索引、一维数组和可变参数
GroupByMatrixApplyFunc = Callable[[Array1d, Array2d, VarArg()], MaybeArray]  # 表示分组矩阵应用函数，接收分组数组、二维数组和可变参数
ApplyMapFunc = Callable[[int, int, I, VarArg()], Scalar]  # 表示应用映射函数，接收行索引、列索引、输入值和可变参数
FilterFunc = Callable[[int, int, I, VarArg()], bool]  # 表示过滤函数，接收行索引、列索引、输入值和可变参数
ReduceFunc = Callable[[int, Array1d, VarArg()], Scalar]  # 表示归约函数，接收索引、一维数组和可变参数
ReduceArrayFunc = Callable[[int, Array1d, VarArg()], Array1d]  # 表示数组归约函数，接收索引、一维数组和可变参数，返回一维数组
GroupReduceFunc = Callable[[int, Array2d, VarArg()], Scalar]  # 表示组归约函数，接收索引、二维数组和可变参数
FlatGroupReduceFunc = Callable[[int, Array1d, VarArg()], Scalar]  # 表示扁平组归约函数，接收索引、一维数组和可变参数
GroupReduceArrayFunc = Callable[[int, Array2d, VarArg()], Array1d]  # 表示组数组归约函数，接收索引、二维数组和可变参数，返回一维数组
FlatGroupReduceArrayFunc = Callable[[int, Array1d, VarArg()], Array1d]  # 表示扁平组数组归约函数，接收索引、一维数组和可变参数，返回一维数组
GroupSqueezeFunc = Callable[[int, int, Array1d, VarArg()], R]  # 表示组压缩函数，接收行索引、列索引、一维数组和可变参数

# ===== 信号处理类型 =====
ChoiceFunc = Callable[[int, int, int, VarArg()], Array1d]  # 表示选择函数，接收多个整数参数和可变参数，返回一维数组
RankFunc = Callable[[int, int, int, int, int, VarArg()], int]  # 表示排序函数，接收多个整数参数和可变参数，返回整数

# ===== 记录处理类型 =====
ColRange = Array2d  # 表示列范围，二维数组
ColMap = Tuple[Array1d, Array1d]  # 表示列映射，两个一维数组组成的元组
MappedApplyFunc = Callable[[Array1d, int, Array1d, VarArg()], Array1d]  # 表示映射应用函数，接收分组数组、索引、一维数组和可变参数，返回一维数组
RecordApplyFunc = Callable[[RecordArray, VarArg()], Array1d]  # 表示记录应用函数，接收记录数组和可变参数，返回一维数组
RecordMapFunc = Callable[[np.void, VarArg()], R]  # 表示记录映射函数，接收numpy.void类型和可变参数
MaskInOutMapFunc = Callable[[Array1d, Array1d, int, Array1d, VarArg()], None]  # 表示掩码映射函数，接收多个数组和可变参数，无返回值

# ===== 指标类型 =====
Param = Any  # 表示参数，可以是任何类型
Params = Union[List[Param], Tuple[Param, ...], NumbaList, Array1d]  # 表示参数列表，可以是列表、元组、NumbaList或一维数组

# ===== 映射类型 =====
Enum = NamedTuple  # 表示枚举类型，实际是命名元组
MappingLike = Union[str, Mapping, Enum, IndexLike]  # 表示类似映射的类型，可以是字符串、映射、枚举或类似索引的类型
