# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
配置实用工具模块。

本模块提供了一组用于管理配置的工具和类，包括：
1. 字典操作工具（合并、更新、复制等）
2. 配置类（Config、AtomicConfig等）
3. 可序列化类（Pickleable、PickleableDict）
4. 可配置对象基类（Configured）

这些工具和类的设计目的是为了在vectorbt库中提供统一、灵活且可序列化的配置管理方案。
"""

import inspect
import pickle
from collections import namedtuple
from copy import copy, deepcopy

import dill

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.docs import Documented, to_doc


class Default:
    """
    用于包装默认值的类。
    提供一种明确标记值为默认值的方式。
    """

    def __init__(self, value: tp.Any) -> None:
        """
        初始化Default对象。
        
        参数:
            value: 要包装的默认值
        """
        self.value = value

    def __repr__(self) -> str:
        """返回对象的字符串表示。"""
        return "Default(" + self.value.__repr__() + ")"

    def __str__(self) -> str:
        """返回对象的字符串表示。"""
        return self.__repr__()


def resolve_dict(dct: tp.DictLikeSequence, i: tp.Optional[int] = None) -> dict:
    """
    选择并解析字典参数。
    
    参数:
        dct: 字典或字典序列
        i: 如果dct是序列，则为要选择的索引
        
    返回:
        解析后的字典
    
    异常:
        ValueError: 当无法解析字典时
    """
    if dct is None:
        dct = {}
    if isinstance(dct, dict):
        return dict(dct)
    if i is not None:
        _dct = dct[i]
        if _dct is None:
            _dct = {}
        return dict(_dct)
    raise ValueError("Cannot resolve dict")


def get_func_kwargs(func: tp.Callable) -> dict:
    """
    获取函数的关键字参数及其默认值。
    
    参数:
        func: 要检查的函数
        
    返回:
        包含参数名和默认值的字典
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_func_arg_names(func: tp.Callable, arg_kind: tp.Optional[tp.MaybeTuple[int]] = None) -> tp.List[str]:
    """
    获取函数的参数名称列表。
    这个函数用于提取函数的参数名称，可以根据参数类型(arg_kind)筛选特定种类的参数。
    参数:
        func: 要检查的函数
        arg_kind: 参数类型过滤器，可以是以下值:
            - None: 返回所有普通参数（排除*args和**kwargs）
            - 单个整数: 返回特定类型的参数
              * 0: POSITIONAL_ONLY - 仅位置参数 (Python 3.8+中的 `/` 标记前的参数)
              * 1: POSITIONAL_OR_KEYWORD - 普通参数 (可以通过位置或关键字传递)
              * 2: VAR_POSITIONAL - 可变位置参数 (`*args`)
              * 3: KEYWORD_ONLY - 仅关键字参数 (`*` 后的参数)
              * 4: VAR_KEYWORD - 可变关键字参数 (`**kwargs`)
            - 整数元组: 返回多种类型的参数 (例如: (2, 4) 返回 *args 和 **kwargs)
    返回:
        符合条件的参数名称列表
    """
    signature = inspect.signature(func)
    if arg_kind is not None and isinstance(arg_kind, int):
        arg_kind = (arg_kind,)
    if arg_kind is None:
        return [
            p.name for p in signature.parameters.values()
            if p.kind != p.VAR_POSITIONAL and p.kind != p.VAR_KEYWORD
        ]
    return [
        p.name for p in signature.parameters.values()
        if p.kind in arg_kind
    ]


class atomic_dict(dict):
    """
    原子字典类，在合并操作中被视为单个值处理的字典。
    继承自dict但在合并时不会被递归展开。
    """
    pass


InConfigLikeT = tp.Union[None, dict, "ConfigT"]
OutConfigLikeT = tp.Union[dict, "ConfigT"]


def convert_to_dict(dct: InConfigLikeT, nested: bool = True) -> dict:
    """
    提取任何字典的普通dict类型实例（atomic_dict除外）。
    
    对于Config类型会提取.items()
    """
    if dct is None:
        dct = {}
    if isinstance(dct, atomic_dict):
        dct = atomic_dict(dct)
    else:
        # 注意当dct是ConfigT类型时，转换后dct是dict类型
        # 由于ConfigT是字典类，该操作会将原dct的所有顶级键值对dct.items()浅拷贝过去
        dct = dict(dct) 
    if not nested:
        return dct
    for k, v in dct.items():
        if isinstance(v, dict):
            dct[k] = convert_to_dict(v, nested=nested)
        else:
            dct[k] = v
    return dct


def set_dict_item(dct: dict, k: tp.Any, v: tp.Any, force: bool = False) -> None:
    """
    设置字典项（比较特殊的是dct为Config类型）。
    
    如果字典是Config类型，则传递force关键字以覆盖阻止标志。
    
    参数:
        dct: 要设置的字典
        k: 键
        v: 值
        force: 是否强制设置（忽略只读或冻结状态）
    """
    if isinstance(dct, Config):
        dct.__setitem__(k, v, force=force)
    else:
        dct[k] = v


def copy_dict(dct: InConfigLikeT, copy_mode: str = 'shallow', nested: bool = True) -> OutConfigLikeT:
    """
    基于复制模式复制字典。
    
    参数:
        dct: 要复制的字典
        copy_mode: 复制模式，支持
            'shallow'： _v = v
            'hybrid'：_v = copy(v)
            'deep'：_v = deepcopy(v)
        nested: 是否递归复制所有子字典
        
    返回:
        复制后的字典
        
    异常:
        ValueError: 当复制模式不支持时
    """
    if dct is None:
        dct = {}
    checks.assert_instance_of(copy_mode, str)
    copy_mode = copy_mode.lower()
    if copy_mode not in ['shallow', 'hybrid', 'deep']:
        raise ValueError(f"Copy mode '{copy_mode}' not supported")

    # 深拷贝
    if copy_mode == 'deep':
        return deepcopy(dct)
    
    # 如果 dct 是 Config 类的实例 (并且 copy_mode 不是 'deep')
    if isinstance(dct, Config):
        return dct.copy(
            copy_mode=copy_mode,
            nested=nested
        )
        
    # 如果 dct 是普通字典并且 copy_mode 不是 'deep' (例如 'shallow' 或 'hybrid')
    dct_copy = copy(dct) 
    for k, v in dct_copy.items():
        # 如果 nested 为 True 且当前值 v 是一个字典类型
        if nested and isinstance(v, dict):
            # 递归调用 copy_dict 来复制这个嵌套字典 v
            _v = copy_dict(v, copy_mode=copy_mode, nested=nested)
        else:
            # 如果 nested 为 False 或者值 v 不是字典
            # 如果复制模式是 'hybrid'
            if copy_mode == 'hybrid':
                _v = copy(v)  # 使用浅复制来复制值
            else:
                # 如果复制模式是 'shallow' (或其他非 'hybrid' 的情况)
                # 直接使用原始值 v (即 _v 指向与原始字典中相同的值对象)
                _v = v
        # 将处理后的值 _v 设置回 dct_copy 字典中对应的键 k
        # force=True 在这里对于普通字典的 set_dict_item 没有特殊效果，主要是为了 Config 子类兼容
        set_dict_item(dct_copy, k, _v, force=True)
    return dct_copy


def update_dict(x: InConfigLikeT,
                y: InConfigLikeT,
                nested: bool = True,
                force: bool = False,
                same_keys: bool = False) -> None:
    """
    将嵌套字典理解为一颗多叉树：
    
    用字典y.items()更新字典x.items()：
        same_keys决定是否只更新x中存在的键
        nested决定是否考虑嵌套字典
        force决定是否强制更新（实际只对Config类型有效）
    
    参数:
        x: 要更新的字典
        y: 包含更新内容的字典
        nested: 是否递归更新所有子字典
        force: 是否强制更新（忽略只读或冻结状态）
        same_keys: 是否只更新已存在的键
    """
    if x is None:
        return
    if y is None:
        return
    checks.assert_instance_of(x, dict)
    checks.assert_instance_of(y, dict)

    for k, v in y.items():
        if nested \
                and k in x \
                and isinstance(x[k], dict) \
                and isinstance(v, dict) \
                and not isinstance(v, atomic_dict):
        # 嵌套更新nested开启、k在x的键中、x[k]是字典、v是字典、v不是atomic_dict
            update_dict(x[k], v, force=force)
        else:
            # 只更新已经存在的键same_keys开启、k不在x的键中
            if same_keys and k not in x:
                continue
            set_dict_item(x, k, v, force=force)


def merge_dicts(*dicts: InConfigLikeT,
                to_dict: bool = True,
                copy_mode: tp.Optional[str] = 'shallow',
                nested: bool = True,
                same_keys: bool = False) -> OutConfigLikeT:
    """
    合并多个字典。
    
    参数:
        *dicts: 要合并的字典
        to_dict: 是否在复制前将每个字典转换为dict
        copy_mode: 复制模式，用于在合并前复制每个字典
        nested: 是否递归合并所有子字典
        same_keys: 是否只合并重叠的键
        
    返回:
        合并后的字典
    """
    # 仅复制一次
    if to_dict:
        dicts = tuple([convert_to_dict(dct, nested=nested) for dct in dicts])
    if copy_mode is not None:
        if not to_dict or copy_mode != 'shallow':
            # to_dict已经进行了浅复制
            dicts = tuple([copy_dict(dct, copy_mode=copy_mode, nested=nested) for dct in dicts])
    x, y = dicts[0], dicts[1]
    should_update = True
    if x.__class__ is dict and y.__class__ is dict and len(x) == 0:
        x = y
        should_update = False
    if isinstance(x, atomic_dict) or isinstance(y, atomic_dict):
        x = y
        should_update = False
    if should_update:
        update_dict(x, y, nested=nested, force=True, same_keys=same_keys)
    if len(dicts) > 2:
        return merge_dicts(
            x, *dicts[2:],
            to_dict=False,  # 仅执行一次
            copy_mode=None,  # 仅执行一次
            nested=nested,
            same_keys=same_keys
        )
    return x


_RaiseKeyError = object()

DumpTuple = namedtuple('DumpTuple', ('cls', 'dumps'))

PickleableT = tp.TypeVar("PickleableT", bound="Pickleable")


class Pickleable:
    """
    定义可序列化类的抽象属性和方法的超类。
    提供基本的序列化和反序列化功能。
    """

    def dumps(self, **kwargs) -> bytes:
        """
        将对象self序列化为字节。
        假设某个类A继承了Pickleable，a是A的实例，那么a.dumps()就是将a序列化为字节
        """
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def loads(cls: tp.Type[PickleableT], dumps: bytes, **kwargs) -> PickleableT:
        """
        从字节流dumps反序列化得到并返回对象。
        """
        return pickle.loads(dumps)

    def save(self, fname: tp.FileName, **kwargs) -> None:
        """
        将self序列化为字节流写入到路径为fname的文件中。
        """
        dumps = self.dumps(**kwargs)
        with open(fname, "wb") as f:
            f.write(dumps)

    @classmethod
    def load(cls: tp.Type[PickleableT], fname: tp.FileName, **kwargs) -> PickleableT:
        """
        从路径为fname的文件中加载字节流，并反序列化得到并返回对象
        
        参数:
            fname: 文件名
            **kwargs: 传递给loads方法的参数
            
        返回:
            反序列化后的对象
        """
        with open(fname, "rb") as f:
            dumps = f.read()
        return cls.loads(dumps, **kwargs)


PickleableDictT = tp.TypeVar("PickleableDictT", bound="PickleableDict")


class PickleableDict(Pickleable, dict):
    """
    可能包含Pickleable类型值的可序列化字典。
    为字典提供序列化和反序列化功能，同时处理内部的可序列化对象。
    """

    def dumps(self, **kwargs) -> bytes:
        """
        将self.items()序列化为字节流
        
        采用了'双重序列化'的设计：
            如果self.items()中的某个键值对中的值是Pickleable类型，则将该值pickle序列化（适用于有自定义序列化要求的类型）
            然后将整体（一个dict）dill序列化
        """
        dct = dict()
        for k, v in self.items():
            if isinstance(v, Pickleable):
                # k: (v.__class__, v.dumps(**kwargs))
                dct[k] = DumpTuple(cls=v.__class__, dumps=v.dumps(**kwargs))    
            else:
                # k: v
                dct[k] = v  
        return dill.dumps(dct, **kwargs)

    @classmethod
    def loads(cls: tp.Type[PickleableDictT], dumps: bytes, **kwargs) -> PickleableDictT:
        """
        将字节流dumps反序列化并返回PickleableDictT类型的实例对象。
        
        与dumps()的'双重序列化'设计相对应，采用'双重反序列化'设计：
            先将整体进行dill反序列化（得到一个dict）
            如果dict的某个键值对中的值是DumpTuple类型，则将值['dumps']进行pickle反序列化
        """
        config = dill.loads(dumps, **kwargs)
        for k, v in config.items():
            if isinstance(v, DumpTuple):
                config[k] = v.cls.loads(v.dumps, **kwargs)
        return cls(**config)

    def load_update(self, fname: tp.FileName, **kwargs) -> None:
        """
        从路径为fname的文件中加载字节流，然后反序列化并重置到 self.items()
        """
        # 这里clear和update是继承的dict的方法
        self.clear()
        self.update(self.load(fname, **kwargs))


ConfigT = tp.TypeVar("ConfigT", bound="Config")


class Config(PickleableDict, Documented):
    """
    扩展字典，添加配置功能，如嵌套更新、冻结键/值和序列化。
    配置可以通过vectorbt._settings.settings中的config设置来覆盖默认值。
    """

    _copy_kwargs_: tp.Kwargs
    _reset_dct_: dict
    _reset_dct_copy_kwargs_: tp.Kwargs
    _frozen_keys_: bool
    _readonly_: bool
    _nested_: bool
    _convert_dicts_: tp.Union[bool, tp.Type["Config"]]
    _as_attrs_: bool

    def __init__(self,
                 dct: tp.DictLike = None,   # 构造新的Config实例的参考，可以是 None, dict, Config
                 copy_kwargs: tp.KwargsLike = None,  # 设置如何参考dct，例如 {'copy_mode': 'shallow', 'nested': True}
                 reset_dct: tp.DictLike = None,  # 重置Config实例时的参考，可以是 None, dict, Config
                 reset_dct_copy_kwargs: tp.KwargsLike = None,  # 设置如何参考reset_dct
                 frozen_keys: tp.Optional[bool] = None,  # 是否拒绝对配置键的更新
                 readonly: tp.Optional[bool] = None,  # 是否拒绝对配置键和值的更新
                 nested: tp.Optional[bool] = None,  # 是否对每个子字典递归执行操作
                 convert_dicts: tp.Optional[tp.Union[bool, tp.Type["Config"]]] = None,  # 是否将子字典转换为具有相同配置的配置对象
                 as_attrs: tp.Optional[bool] = None) -> None:  # 是否启用通过点符号访问字典键
        """
        初始化Config对象。
        
        解析参数，应用默认值，并设置配置属性。
        """
        try:
            from vectorbt._settings import settings
            configured_cfg = settings['config']
        except ImportError:
            configured_cfg = {}

        if dct is None:
            dct = dict()

        def _resolve_param(pname: str, p: tp.Any, default: tp.Any, merge: bool = False) -> tp.Any:
            '''
            解析参数
                p：Config的__init__方法传入的
                dct_p：dct.__dict__[_pname_]，if Config的__init__方法传入的dct是Config实例
                cfg_default：settings['config'][pname]
                default：传入的
                
                merge：是否合并
                
            优先级：p > dct_p > cfg_default > default
            '''
            cfg_default = configured_cfg.get(pname, None)
            dct_p = getattr(dct, pname + '_') if isinstance(dct, Config) else None

            if merge and isinstance(default, dict):
                return merge_dicts(default, cfg_default, dct_p, p)
            # 非合并模式，直接按优先级返回
            if p is not None:
                return p
            if dct_p is not None:
                return dct_p
            if cfg_default is not None:
                return cfg_default
            return default

        # 相当于没操作，就是__init__传入的
        reset_dct = _resolve_param('reset_dct', reset_dct, None)
        frozen_keys = _resolve_param('frozen_keys', frozen_keys, False)
        readonly = _resolve_param('readonly', readonly, False)
        nested = _resolve_param('nested', nested, False)
        convert_dicts = _resolve_param('convert_dicts', convert_dicts, False)
        as_attrs = _resolve_param('as_attrs', as_attrs, frozen_keys or readonly)
        # 注意 copy_kwargs 和 reset_dct_copy_kwargs 都是 Union[None, Dict[str, Any]] 类型
        # 其中键也只可能是'copy_mode'和'nested'
        reset_dct_copy_kwargs = merge_dicts(copy_kwargs, reset_dct_copy_kwargs)
        copy_kwargs = _resolve_param(
            'copy_kwargs',
            copy_kwargs,
            dict(
                copy_mode='shallow' if readonly else 'hybrid',
                nested=nested
            ),
            merge=True
        )
        reset_dct_copy_kwargs = _resolve_param(
            'reset_dct_copy_kwargs',
            reset_dct_copy_kwargs,
            dict(
                copy_mode='shallow' if readonly else 'hybrid',
                nested=nested
            ),
            merge=True
        )

        # 复制字典
        dct = copy_dict(dict(dct), **copy_kwargs)

        # 转换子字典
        if convert_dicts:
            if not nested:
                raise ValueError("convert_dicts requires nested to be True")
            for k, v in dct.items():
                if isinstance(v, dict) and not isinstance(v, Config):
                    if isinstance(convert_dicts, bool):
                        config_cls = self.__class__
                    elif issubclass(convert_dicts, Config):
                        config_cls = convert_dicts
                    else:
                        raise TypeError("convert_dicts must be either boolean or a subclass of Config")
                    dct[k] = config_cls(
                        v,
                        copy_kwargs=copy_kwargs,
                        reset_dct_copy_kwargs=reset_dct_copy_kwargs,
                        frozen_keys=frozen_keys,
                        readonly=readonly,
                        nested=nested,
                        convert_dicts=convert_dicts,
                        as_attrs=as_attrs
                    )

        # 复制初始配置
        if reset_dct is None:
            reset_dct = dct
        reset_dct = copy_dict(dict(reset_dct), **reset_dct_copy_kwargs)

        dict.__init__(self, dct)

        # 将参数存储在实例变量中
        checks.assert_instance_of(copy_kwargs, dict)
        checks.assert_instance_of(reset_dct, dict)
        checks.assert_instance_of(reset_dct_copy_kwargs, dict)
        checks.assert_instance_of(frozen_keys, bool)
        checks.assert_instance_of(readonly, bool)
        checks.assert_instance_of(nested, bool)
        checks.assert_instance_of(convert_dicts, (bool, type))
        checks.assert_instance_of(as_attrs, bool)

        self.__dict__['_copy_kwargs_'] = copy_kwargs
        self.__dict__['_reset_dct_'] = reset_dct
        self.__dict__['_reset_dct_copy_kwargs_'] = reset_dct_copy_kwargs
        self.__dict__['_frozen_keys_'] = frozen_keys
        self.__dict__['_readonly_'] = readonly
        self.__dict__['_nested_'] = nested
        self.__dict__['_convert_dicts_'] = convert_dicts
        self.__dict__['_as_attrs_'] = as_attrs

        # 将键设置为属性以实现自动完成
        if as_attrs:
            for k, v in self.items():
                if k in self.__dir__():
                    raise ValueError(f"Cannot set key '{k}' as attribute of the config. Disable as_attrs.")
                self.__dict__[k] = v

    @property
    def copy_kwargs_(self) -> tp.Kwargs:
        """用于复制dct的参数。"""
        return self._copy_kwargs_

    @property
    def reset_dct_(self) -> dict:
        """重置时回退的字典。"""
        return self._reset_dct_

    @property
    def reset_dct_copy_kwargs_(self) -> tp.Kwargs:
        """用于复制reset_dct的参数。"""
        return self._reset_dct_copy_kwargs_

    @property
    def frozen_keys_(self) -> bool:
        """是否拒绝对配置键和值的更新。"""
        return self._frozen_keys_

    @property
    def readonly_(self) -> bool:
        """是否拒绝对配置的任何更新。"""
        return self._readonly_

    @property
    def nested_(self) -> bool:
        """是否对每个子字典递归执行操作。"""
        return self._nested_

    @property
    def convert_dicts_(self) -> tp.Union[bool, tp.Type["Config"]]:
        """是否将子字典转换为具有相同配置的配置对象。"""
        return self._convert_dicts_

    @property
    def as_attrs_(self) -> bool:
        """是否启用通过点符号访问字典键。"""
        return self._as_attrs_

    # self 具有 items() 和 __dict__ 两个属性存储结构。
    # _as_attrs_ 决定了是否支持点赋值
    # 当实施赋值时，_readonly_ 和 _frozen_keys_ 决定了是否允许赋值。
    # 允许赋值后，_as_attrs_ 还决定了是否更新 __dict__
    def __setattr__(self, k: str, v: tp.Any) -> None:
        """
        设置属性。
        
        如果as_attrs_为True，则将属性设置转发到__setitem__。
        """
        if self.as_attrs_:
            self.__setitem__(k, v)

    def __setitem__(self, k: str, v: tp.Any, force: bool = False) -> None:
        """
        设置字典项。
        
        考虑只读和冻结键的限制。
        
        参数:
            k: 键
            v: 值
            force: 是否强制设置（忽略只读或冻结状态）
            
        异常:
            TypeError: 当配置为只读时
            KeyError: 当配置键被冻结且键不存在时
        """
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        if not force and self.frozen_keys_:
            if k not in self:
                raise KeyError(f"Config keys are frozen: key '{k}' not found")
        dict.__setitem__(self, k, v)
        if self.as_attrs_:
            self.__dict__[k] = v

    def __delattr__(self, k: str) -> None:
        """
        删除属性。
        
        如果as_attrs_为True，则将删除属性转发到__delitem__。
        """
        if self.as_attrs_:
            self.__delitem__(k)

    def __delitem__(self, k: str, force: bool = False) -> None:
        """
        删除字典项。
        
        考虑只读和冻结键的限制。
        
        参数:
            k: 键
            force: 是否强制删除（忽略只读或冻结状态）
            
        异常:
            TypeError: 当配置为只读时
            KeyError: 当配置键被冻结时
        """
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        if not force and self.frozen_keys_:
            raise KeyError(f"Config keys are frozen")
        dict.__delitem__(self, k)
        if self.as_attrs_:
            del self.__dict__[k]

    def  _clear_attrs(self, prior_keys: tp.Iterable[str]) -> None:
        """
        如果self._as_attrs_为真，删除self.__dict__[{prior_keys - self.keys()}]
        
        参数:
            prior_keys: 删除前的键列表
        """
        if self.as_attrs_:
            for k in set(prior_keys).difference(self.keys()):
                del self.__dict__[k]

    def pop(self, k: str, v: tp.Any = _RaiseKeyError, force: bool = False) -> tp.Any:
        """
        删除并返回指定键的键值对。
        
        参数:
            k: 键
            v: 如果键不存在时的默认值
            force: 是否强制操作（忽略只读或冻结状态）
            
        返回:
            键对应的值
            
        异常:
            TypeError: 当配置为只读时
            KeyError: 当配置键被冻结时
        """
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        if not force and self.frozen_keys_:
            raise KeyError(f"Config keys are frozen")
        prior_keys = list(self.keys())
        if v is _RaiseKeyError:
            result = dict.pop(self, k)
        else:
            result = dict.pop(self, k, v)
        self._clear_attrs(prior_keys)
        return result

    def popitem(self, force: bool = False) -> tp.Tuple[tp.Any, tp.Any]:
        """
        删除并返回某个键值对。
        
        参数:
            force: 是否强制操作（忽略只读或冻结状态）
            
        返回:
            键值对元组
            
        异常:
            TypeError: 当配置为只读时
            KeyError: 当配置键被冻结时
        """
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        if not force and self.frozen_keys_:
            raise KeyError(f"Config keys are frozen")
        prior_keys = list(self.keys())
        result = dict.popitem(self)
        self._clear_attrs(prior_keys)
        return result

    def clear(self, force: bool = False) -> None:
        """
        删除 self.items() 中所有项
        
        参数:
            force: 是否强制操作（忽略只读或冻结状态）
            
        异常:
            TypeError: 当配置为只读时
            KeyError: 当配置键被冻结时
        """
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        if not force and self.frozen_keys_:
            raise KeyError(f"Config keys are frozen")
        prior_keys = list(self.keys())
        dict.clear(self)
        self._clear_attrs(prior_keys)

    def update(self, *args, nested: tp.Optional[bool] = None, force: bool = False, **kwargs) -> None:
        """
        用 dict(*args, **kwargs) 更新 self.items() 和 self.__dict__
        
        参数:
            *args: 要更新的内容
            nested: 是否递归更新所有子字典
            force: 是否强制更新（忽略只读或冻结状态）
            **kwargs: 键值对更新
            
        参见update_dict函数。
        """
        other = dict(*args, **kwargs)
        if nested is None:
            nested = self.nested_
        # 用 other 更新 self.items()
        update_dict(self, other, nested=nested, force=force)

    def __copy__(self: ConfigT) -> ConfigT:
        """
        创建一个新的和self具有相同类型的实例，
        其 __dict__/items() 和 self.__dict__/items() 的对应键完全相同
        """
        # Config 可能有其它子类，其它子类可能实现了 __new__
        cls = self.__class__
        self_copy = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k not in self_copy:  
                self_copy.__dict__[k] = v
        # 删除 self_copy.items() 中所有项
        self_copy.clear(force=True)
        # 用 self.items() 更新 self_copy.items()
        self_copy.update(copy(dict(self)), nested=False, force=True)
        return self_copy

    def __deepcopy__(self: ConfigT, memo: tp.DictLike = None) -> ConfigT:
        """
        创建一个新的和self具有相同类型的实例，
        其 __dict__/items() 的键是 self.__dict__/items() 的键深拷贝得到的
        """
        if memo is None:
            memo = {}
        cls = self.__class__
        self_copy = cls.__new__(cls)
        memo[id(self)] = self_copy
        for k, v in self.__dict__.items():
            if k not in self_copy:  # 否则会复制字典键两次
                self_copy.__dict__[k] = deepcopy(v, memo)
        # 删除 self_copy.items() 中所有项
        self_copy.clear(force=True)
        # 用 self.items() 更新 self_copy.items()
        self_copy.update(deepcopy(dict(self), memo), nested=False, force=True)
        return self_copy

    def copy(self: ConfigT, reset_dct_copy_kwargs: tp.KwargsLike = None, **copy_kwargs) -> ConfigT:
        """
        创建一个新的和self具有相同类型的实例，其 __dict__/items() 和 self.__dict__/items() 的对应键完全相同
        合并获得_reset_dct_ 的复制参数，并按照复制参数重新复制self._reset_dct_给self.__dict__['_reset_dct_']
        合并获得self.items()的复制参数，并按照复制参数重新复制self.items()给self_copy.items()/__dict__
        
        参数:
            reset_dct_copy_kwargs: 通过合并覆盖Config.reset_dct_copy_kwargs_的参数
            **copy_kwargs: 通过合并覆盖Config.copy_kwargs_和Config.reset_dct_copy_kwargs_的参数
            
        返回:
            配置对象的复制
        """
        self_copy = self.__copy__()
        
        reset_dct_copy_kwargs = merge_dicts(self.reset_dct_copy_kwargs_, copy_kwargs, reset_dct_copy_kwargs)
        reset_dct = copy_dict(dict(self.reset_dct_), **reset_dct_copy_kwargs)
        self.__dict__['_reset_dct_'] = reset_dct

        copy_kwargs = merge_dicts(self.copy_kwargs_, copy_kwargs)
        dct = copy_dict(dict(self), **copy_kwargs)
        self_copy.update(dct, nested=False, force=True)

        return self_copy

    def merge_with(self: ConfigT,
                   other: InConfigLikeT,
                   nested: tp.Optional[bool] = None,
                   **kwargs) -> OutConfigLikeT:
        """
        与另一个字典合并为一个字典。
        
        参数:
            other: 要合并的另一个字典
            nested: 是否递归合并所有子字典
            **kwargs: 传递给merge_dicts的参数
            
        返回:
            合并后的字典
            
        参见merge_dicts函数。
        """
        if nested is None:
            nested = self.nested_
        return merge_dicts(self, other, nested=nested, **kwargs)

    def to_dict(self, nested: tp.Optional[bool] = None) -> dict:
        """
        转换为字典。
        
        参数:
            nested: 是否递归转换所有子字典
            
        返回:
            转换后的字典
        """
        return convert_to_dict(self, nested=nested)

    def reset(self, force: bool = False, **reset_dct_copy_kwargs) -> None:
        """
        清除配置并用初始配置更新它。
        
        参数:
            force: 是否强制重置（忽略只读状态）
            **reset_dct_copy_kwargs: 覆盖Config.reset_dct_copy_kwargs_的参数
            
        异常:
            TypeError: 当配置为只读时
        """
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        reset_dct_copy_kwargs = merge_dicts(self.reset_dct_copy_kwargs_, reset_dct_copy_kwargs)
        reset_dct = copy_dict(dict(self.reset_dct_), **reset_dct_copy_kwargs)
        self.clear(force=True)
        self.update(self.reset_dct_, nested=False, force=True)
        self.__dict__['_reset_dct_'] = reset_dct

    def make_checkpoint(self, force: bool = False, **reset_dct_copy_kwargs) -> None:
        """
        用当前状态替换reset_dct。
        
        参数:
            force: 是否强制创建检查点（忽略只读状态）
            **reset_dct_copy_kwargs: 覆盖Config.reset_dct_copy_kwargs_的参数
            
        异常:
            TypeError: 当配置为只读时
        """
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        reset_dct_copy_kwargs = merge_dicts(self.reset_dct_copy_kwargs_, reset_dct_copy_kwargs)
        reset_dct = copy_dict(dict(self), **reset_dct_copy_kwargs)
        self.__dict__['_reset_dct_'] = reset_dct

    def dumps(self, **kwargs) -> bytes:
        """
        序列化为字节。
        
        参数:
            **kwargs: 传递给dill.dumps的参数
            
        返回:
            序列化后的字节
        """
        return dill.dumps(dict(
            dct=PickleableDict(self).dumps(**kwargs),
            copy_kwargs=self.copy_kwargs_,
            reset_dct=PickleableDict(self.reset_dct_).dumps(**kwargs),
            reset_dct_copy_kwargs=self.reset_dct_copy_kwargs_,
            frozen_keys=self.frozen_keys_,
            readonly=self.readonly_,
            nested=self.nested_,
            convert_dicts=self.convert_dicts_,
            as_attrs=self.as_attrs_
        ), **kwargs)

    @classmethod
    def loads(cls: tp.Type[ConfigT], dumps: bytes, **kwargs) -> ConfigT:
        """
        从字节反序列化。
        
        参数:
            dumps: 序列化的字节
            **kwargs: 传递给dill.loads的参数
            
        返回:
            反序列化后的配置对象
        """
        obj = dill.loads(dumps, **kwargs)
        return cls(
            dct=PickleableDict.loads(obj['dct'], **kwargs),
            copy_kwargs=obj['copy_kwargs'],
            reset_dct=PickleableDict.loads(obj['reset_dct'], **kwargs),
            reset_dct_copy_kwargs=obj['reset_dct_copy_kwargs'],
            frozen_keys=obj['frozen_keys'],
            readonly=obj['readonly'],
            nested=obj['nested'],
            convert_dicts=obj['convert_dicts'],
            as_attrs=obj['as_attrs']
        )

    def load_update(self, fname: tp.FileName, **kwargs) -> None:
        """
        从文件加载序列化的配置并更新此实例。
        
        参数:
            fname: 文件名
            **kwargs: 传递给load方法的参数
            
        注意:
            同时更新配置属性和字典内容。
        """
        loaded = self.load(fname, **kwargs)
        self.clear(force=True)
        self.__dict__.clear()
        self.__dict__.update(loaded.__dict__)
        self.update(loaded, nested=False, force=True)

    def __eq__(self, other: tp.Any) -> bool:
        """
        判断两个配置对象是否相等。
        
        参数:
            other: 要比较的对象
            
        返回:
            如果两个对象内容相等则为True，否则为False
        """
        return checks.is_deep_equal(dict(self), dict(other))

    def to_doc(self, with_params: bool = False, **kwargs) -> str:
        """
        转换为文档字符串。
        
        参数:
            with_params: 是否包含参数信息
            **kwargs: 传递给to_doc的参数
            
        返回:
            文档字符串
        """
        doc = self.__class__.__name__ + "(" + to_doc(dict(self), **kwargs) + ")"
        if with_params:
            doc += " with params " + to_doc(dict(
                copy_kwargs=self.copy_kwargs_,
                reset_dct=self.reset_dct_,
                reset_dct_copy_kwargs=self.reset_dct_copy_kwargs_,
                frozen_keys=self.frozen_keys_,
                readonly=self.readonly_,
                nested=self.nested_,
                convert_dicts=self.convert_dicts_,
                as_attrs=self.as_attrs_
            ), **kwargs)
        return doc


class AtomicConfig(Config, atomic_dict):
    """
    原子配置类，在合并时被视为单个值处理的配置对象。
    
    继承自Config和atomic_dict，在合并操作中不会递归展开其内容。
    """
    pass


ConfiguredT = tp.TypeVar("ConfiguredT", bound="Configured")


class Configured(Pickleable, Documented):
    """
    具有初始化配置的类。
    
    所有Configured的子类都使用Config进行初始化，这使得序列化更加容易。
    
    设置定义在vectorbt._settings.settings的configured下。
    
    警告:
        如果任何不在Configured.writeable_attrs中列出的属性被覆盖，
        或者如果任何Configured.__init__参数依赖于全局默认值，
        它们的值将不会被复制。确保显式传递它们以使保存和加载/复制的实例
        对全局变化具有弹性。
    """

    def __init__(self, **config) -> None:
        """
        初始化Configured对象。
        
        参数:
            **config: 初始化配置
        """
        from vectorbt._settings import settings
        configured_cfg = settings['configured']

        self._config = Config(config, **configured_cfg['config'])

    @property
    def config(self) -> Config:
        """初始化配置。"""
        return self._config

    @property
    def writeable_attrs(self) -> tp.Set[str]:
        """
        可写属性集合，这些属性将与配置一起保存/复制。
        
        返回:
            可写属性集合
        """
        return {
            base_cls.writeable_attrs.__get__(self)
            for base_cls in self.__class__.__bases__
            if isinstance(base_cls, Configured)
        }

    def replace(self: ConfiguredT,
                copy_mode_: tp.Optional[str] = 'shallow',
                nested_: tp.Optional[bool] = None,
                cls_: tp.Optional[type] = None,
                **new_config) -> ConfiguredT:
        """
        通过复制和（可选）更改配置创建新实例。
        
        参数:
            copy_mode_: 复制模式
            nested_: 是否递归复制所有子字典
            cls_: 新实例的类
            **new_config: 要合并的新配置
            
        返回:
            新实例
            
        警告:
            此操作不会返回实例的副本，而是返回使用相同配置和可写属性
            （或其副本，取决于copy_mode）初始化的新实例。
        """
        if cls_ is None:
            cls_ = self.__class__
        new_config = self.config.merge_with(new_config, copy_mode=copy_mode_, nested=nested_)
        new_instance = cls_(**new_config)
        for attr in self.writeable_attrs:
            attr_obj = getattr(self, attr)
            if isinstance(attr_obj, Config):
                attr_obj = attr_obj.copy(
                    copy_mode=copy_mode_,
                    nested=nested_
                )
            else:
                if copy_mode_ is not None:
                    if copy_mode_ == 'hybrid':
                        attr_obj = copy(attr_obj)
                    elif copy_mode_ == 'deep':
                        attr_obj = deepcopy(attr_obj)
            setattr(new_instance, attr, attr_obj)
        return new_instance

    def copy(self: ConfiguredT,
             copy_mode: tp.Optional[str] = 'shallow',
             nested: tp.Optional[bool] = None,
             cls: tp.Optional[type] = None) -> ConfiguredT:
        """
        通过复制配置创建新实例。
        
        参数:
            copy_mode: 复制模式
            nested: 是否递归复制所有子字典
            cls: 新实例的类
            
        返回:
            新实例
            
        参见Configured.replace方法。
        """
        return self.replace(copy_mode_=copy_mode, nested_=nested, cls_=cls)

    def dumps(self, **kwargs) -> bytes:
        """
        序列化为字节。
        
        参数:
            **kwargs: 传递给dill.dumps的参数
            
        返回:
            序列化后的字节
        """
        config_dumps = self.config.dumps(**kwargs)
        attr_dct = PickleableDict({attr: getattr(self, attr) for attr in self.writeable_attrs})
        attr_dct_dumps = attr_dct.dumps(**kwargs)
        return dill.dumps((config_dumps, attr_dct_dumps), **kwargs)

    @classmethod
    def loads(cls: tp.Type[ConfiguredT], dumps: bytes, **kwargs) -> ConfiguredT:
        """
        从字节反序列化。
        
        参数:
            dumps: 序列化的字节
            **kwargs: 传递给dill.loads的参数
            
        返回:
            反序列化后的对象
        """
        config_dumps, attr_dct_dumps = dill.loads(dumps, **kwargs)
        config = Config.loads(config_dumps, **kwargs)
        attr_dct = PickleableDict.loads(attr_dct_dumps, **kwargs)
        new_instance = cls(**config)
        for attr, obj in attr_dct.items():
            setattr(new_instance, attr, obj)
        return new_instance

    def __eq__(self, other: tp.Any) -> bool:
        """
        判断两个对象是否相等。
        
        如果它们的配置和可写属性相等，则对象相等。
        
        参数:
            other: 要比较的对象
            
        返回:
            如果两个对象相等则为True，否则为False
        """
        if type(self) != type(other):
            return False
        if self.writeable_attrs != other.writeable_attrs:
            return False
        for attr in self.writeable_attrs:
            if not checks.is_deep_equal(getattr(self, attr), getattr(other, attr)):
                return False
        return self.config == other.config

    def update_config(self, *args, **kwargs) -> None:
        """
        强制更新配置。
        
        参数:
            *args: 传递给config.update的参数
            **kwargs: 传递给config.update的关键字参数
        """
        self.config.update(*args, **kwargs, force=True)

    def to_doc(self, **kwargs) -> str:
        """
        转换为文档字符串。
        
        参数:
            **kwargs: 传递给config.to_doc的参数
            
        返回:
            文档字符串
        """
        return self.__class__.__name__ + "(**" + self.config.to_doc(**kwargs) + ")"