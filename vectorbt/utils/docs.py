# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
文档实用工具模块。

本模块提供了一系列用于生成和格式化文档内容的辅助工具。
主要功能包括：
1. 定义可被文档化的对象基类 (Documented)。
2. 定义在文档准备过程中可以安全转换为字符串的标记类 (SafeToStr)。
3. prepare_for_doc 用于将各种Python对象转换为适合JSON化的格式。
4. to_doc 用于将 prepare_for_doc 处理后的对象转换为JSON字符串。
"""

# 导入json模块，用于JSON序列化和反序列化
import json
import numpy as np
from vectorbt import _typing as tp


# 定义一个名为Documented的类
class Documented:
    """
    可文档化对象的抽象基类。
    设计目的:
        - 为需要自定义文档表示的类提供一个统一的接口。
        - 区分普通对象和那些需要特殊文档化处理的对象。
    """

    # 定义一个名为to_doc的方法，需要子类实现
    def to_doc(self, **kwargs) -> str:
        """
        将对象转换为文档字符串表示。
        子类必须重写此方法以提供其特定于文档的字符串形式。
        """
        raise NotImplementedError

    def __str__(self):
        try:
            # 尝试调用to_doc方法
            return self.to_doc()
        except NotImplementedError:
            # 如果to_doc未实现，则返回对象的标准repr表示
            return repr(self)


# 定义一个名为SafeToStr的类
class SafeToStr:
    """
    标记类，指示其对象可以在 `prepare_for_doc` 函数中被安全地直接转换为字符串。

    当 `prepare_for_doc`遇到此类的实例时，会直接调用其 `__str__` 方法获取字符串表示，
    而不是进行更复杂的递归处理。

    设计目的:
        - 为那些其 `__str__` 方法已经提供了适合文档的输出的类提供一个简便的标记。
        - 避免对这些对象进行不必要的深度处理。
    """
    pass


def prepare_for_doc(obj: tp.Any, replace: tp.DictLike = None, path: str = None) -> tp.Any:
    """
    准备对象以便在文档中使用，将其转换为更适合展示的格式。
    
    基本情况：numpy.dtype，numpy.ndarray，SafeToStr
    递归情况：namedtuple，tuple，list，dict
    其余情况：直接原obj
    """
    
    if isinstance(obj, SafeToStr):
        return str(obj)
    # 检查obj是否为numpy.dtype类型并且具有fields属性（表明是结构化类型）
    if isinstance(obj, np.dtype) and hasattr(obj, "fields"):
        '''
        dtype = np.dtype([('name', 'U10'), ('age', 'i4'), ('score', 'f8')])
        dtype.fields 类似于 {'name': ('U10', 0), 'age': ('i4', 16), 'score': ('f8', 20)}
        转换为 {'name': 'U10', 'age': 'i4', 'score': 'f8'}
        '''
        return dict(zip(
            dict(obj.fields).keys(),  
            list(map(lambda x: str(x[0]), dict(obj.fields).values()))  
        ))
    # 检查obj是否为元组并且具有_asdict方法（表明是namedtuple）
    if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
        ''' 
        递归处理 obj._asdict()，注意obj._asdict()类似于 OrderedDict([('x', 10), ('y', 20)])
        '''
        return prepare_for_doc(obj._asdict(), replace, path)
    # 检查obj是否为元组或列表
    if isinstance(obj, (tuple, list)):
        # 如果是，递归处理列表或元组中的每个元素
        return [prepare_for_doc(v, replace, path) for v in obj]
    # 检查obj是否为字典
    if isinstance(obj, dict):
        '''
        字典replace相当于预先保存的对应信息
        字典obj的每个键 k 加上指定来源 path + '.'，来查找是否有对应的信息，如果有就替换
        否则要对obj的每个值 v 继续生成信息（注意来源路径要被替换成 path + '.' + k）
        '''
        if replace is None:
            replace = {}
        new_obj = dict()
        for k, v in obj.items():
            if path is None:
                new_path = k
            else:
                new_path = path + '.' + k
            if new_path in replace:
                new_obj[k] = replace[new_path]
            else:
                new_obj[k] = prepare_for_doc(v, replace, new_path)
        return new_obj
    # 检查对象是否具有shape属性，并且shape是元组（通常是numpy.ndarray）
    if hasattr(obj, 'shape') and isinstance(obj.shape, tuple):
        if len(obj.shape) == 0:
            return obj.item()
        return "{} of shape {}".format(object.__repr__(obj), obj.shape)
    # 对于其他所有类型的对象，直接返回原对象
    return obj


def to_doc(obj: tp.Any, replace: tp.DictLike = None, path: str = None, **kwargs) -> str:
    """
    将任意Python对象转换为格式化的JSON字符串，以便在文档中使用。

    首先调用 `prepare_for_doc` 对对象进行预处理，使其结构更适合JOSON化，
    然后使用 `json.dumps` 将处理后的对象序列化为JSON字符串。

    参数:
        obj: tp.Any - 需要转换为JSON字符串的任意Python对象。
        replace: tp.DictLike (可选) - 传递给 `prepare_for_doc` 的替换规则字典。
                 默认为None。
        path: str (可选) - 传递给 `prepare_for_doc` 的初始路径。默认为None。
        **kwargs: 其他传递给 `json.dumps` 的关键字参数，例如 `indent`（缩进级别）、
                  `default`（处理无法直接序列化对象的函数）等。
                  默认缩进为4，默认处理函数为 `str`。

    返回:
        str - 表示对象的格式化JSON字符串。
    """
    # 设置json.dumps的默认参数，如果kwargs中未提供，则使用默认值
    # 默认缩进为4，默认的无法序列化对象的处理函数为str
    kwargs = {**dict(indent=4, default=str), **kwargs}
    # 调用prepare_for_doc处理对象，然后用json.dumps转换为JSON字符串
    return json.dumps(prepare_for_doc(obj, replace, path), **kwargs)
