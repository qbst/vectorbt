# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

import inspect
from functools import wraps, lru_cache
from threading import RLock

import numpy as np

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import Config

# 具体原理参考externalLib/vectorbt/vectorbt/笔记/utils/decorators.ipynb

class class_or_instancemethod(classmethod):
    """
    类/实例方法装饰器
    @class_or_instancemethod
    def method():...    # method = class_or_instancemethod(method)
    
    obj.method(***)     # 等价于 method.__get__(obj, type(obj)).__call__(***)
    Class.method(***)   # 等价于 method.__get__(cls, type(cls)).__call__(***)
                        #   或者 MethodType(method, cls).__call__(***)
    """
    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, owner)


class classproperty(object):
    """
    类属性装饰器
    @classproperty
    def property():...  # property = classproperty(property)
    
    obj/Class.property  # 等价于 property(type(obj)/Class)
    obj.property = 1    # 抛出异常
    """

    def __init__(self, func: tp.Callable) -> None:
        self.func = func
        # 复制原函数的文档字符串到描述符对象，这样通过help()或__doc__访问时能显示原函数的文档
        self.__doc__ = getattr(func, '__doc__')

    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        return self.func(owner)

    def __set__(self, instance: object, value: tp.Any) -> None:
        # 抛出异常，明确告知该属性是只读的
        raise AttributeError("can't set attribute")


class class_or_instanceproperty(object):
    """
    类/实例属性装饰器
    @class_or_instanceproperty
    def property():...  # property = class_or_instanceproperty(property)
    obj/Class.property  # 等价于 property(obj/Class)
    obj.property = 1    # 抛出异常
    """

    def __init__(self, func: tp.Callable) -> None:
        self.func = func
        self.__doc__ = getattr(func, '__doc__')

    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        if instance is None:
            return self.func(owner)
        return self.func(instance)

    def __set__(self, instance: object, value: tp.Any) -> None:
        raise AttributeError("can't set attribute")


custom_propertyT = tp.TypeVar("custom_propertyT", bound="custom_property")


class custom_property:
    """
    两种装饰方式：
        @custom_property(a=0, b=0)  # flags
        def property(): ...
        等价于 property = custom_property(a=0, b=0)(property)
        其中 custom_property(a=0, b=0) 返回一个 lambda 表达式，从而进一步构建 cls(property)
    或
        @custom_property
        def property(): ...
        
    
    obj.property        # 等价于 property(obj)
    Class.property      # 返回该描述符property本身
    obj.property = 1    # 抛出异常
    obj.property(***)   # pass，待实现
    """
    def __new__(cls: tp.Type[custom_propertyT], *args, **flags) -> tp.Union[tp.Callable, custom_propertyT]:
        if len(args) == 0:  
            return lambda func: cls(func, **flags)
        elif len(args) == 1:
            return super().__new__(cls)
        raise ValueError("Either function or keyword arguments must be passed")

    def __init__(self, func: tp.Callable, **flags) -> None:
        self.func = func 
        self.name = func.__name__  
        self.flags = flags  
        self.__doc__ = getattr(func, '__doc__')  

    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        if instance is None:
            return self
        return self.func(instance)

    def __set__(self, instance: object, value: tp.Any) -> None:
        raise AttributeError("can't set attribute")

    def __call__(self, *args, **kwargs) -> tp.Any:
        pass  


class CacheCondition(tp.NamedTuple):
    """
    缓存条件类，用于 `should_cache` 函数中定义缓存策略的条件。
    
    该类继承自 NamedTuple，用于封装各种缓存条件参数，通过这些条件可以控制哪些方法或属性应该被缓存。
    条件的匹配遵循优先级排序，越具体的条件优先级越高。
    
    例子：
        # 针对特定实例和方法的缓存条件
        condition1 = CacheCondition(instance=my_obj, func='calculate')
    
    条件优先级排序（数字越小优先级越高）：
        0) instance + func：最具体的条件，针对特定实例的特定方法
        1) instance + flags：针对特定实例的带标志方法
        2) instance：针对特定实例的所有方法
        3) cls + func：针对特定类的特定方法
        4) cls + flags：针对特定类的带标志方法
        5) cls：针对特定类的所有方法
        6) base_cls + func：针对基类的特定方法
        7) base_cls + flags：针对基类的带标志方法
        8) base_cls：针对基类的所有方法
        9) func + flags：针对带标志的特定方法
        10) func：针对特定方法
        11) flags：针对带特定标志的所有方法
    """
    instance: tp.Optional[object] = None
    func: tp.Optional[tp.Union[tp.Callable, "cached_property", str]] = None
    cls: tp.Optional[tp.Union[type, str]] = None
    base_cls: tp.Optional[tp.Union[type, str]] = None
    flags: tp.Optional[dict] = None
    rank: tp.Optional[int] = None

def should_cache(func_name: str, instance: object, func: tp.Optional[tp.Callable] = None, **flags) -> bool:
    """
    根据配置的缓存条件判断是否应该对指定的方法/属性进行缓存。
    
    通过评估一系列预定义的缓存条件来决定
    是否启用缓存。条件按照优先级排序，越具体的条件优先级越高。
    如果同时匹配到白名单和黑名单条件且优先级相同，则由全局缓存开关 `enabled` 决定。
    
    缓存条件优先级排序（数字越小优先级越高）：
        0) instance + func：针对特定实例的特定方法/属性
        1) instance + flags：针对特定实例的带特定标志的方法/属性
        2) instance：针对特定实例的所有方法/属性
        3) cls + func：针对特定类的特定方法/属性
        4) cls + flags：针对特定类的带特定标志的方法/属性
        5) cls：针对特定类的所有方法/属性
        6) base_cls + func：针对基类的特定方法/属性
        7) base_cls + flags：针对基类的带特定标志的方法/属性
        8) base_cls：针对基类的所有方法/属性
        9) func + flags：针对带特定标志的特定方法/属性
        10) func：针对特定方法/属性
        11) flags：针对带特定标志的所有方法/属性
    """
    
    from vectorbt._settings import settings
    caching_cfg = settings['caching']

    # 起始排序值，用于表示未匹配任何条件的情况
    start_rank = 100

    def _get_condition_rank(cond: CacheCondition) -> int:
        """
        使用缓存条件 cond 考察当前的方法调用should_cache的参数：func_name, instance, func, **flags
        并返回相应的优先级排序值，排序值越小表示优先级越高。
        
        Args:
            cond (CacheCondition): 要评估的缓存条件对象
            
        Returns:
            int: 条件的优先级排序值，如果条件不匹配则返回 start_rank (100)
        """

        checks.assert_instance_of(cond, CacheCondition)

        # 检查实例条件：如果指定了实例，必须与当前实例完全匹配
        if cond.instance is not None:
            if instance is not cond.instance:
                return start_rank
          
        # 检查函数/方法条件：支持多种类型的函数匹配
        if cond.func is not None:
            # 如果 cond.func 是 cached_property 实例
            if isinstance(cond.func, cached_property):  
                if func != cond.func.func:
                    return start_rank
            elif callable(cond.func) and hasattr(func, 'func') and hasattr(cond.func, 'func'):  
                if func.func != cond.func.func:
                    return start_rank
            elif isinstance(cond.func, str):
                if func_name != cond.func:
                    return start_rank
            else:
                raise TypeError(f"Caching condition {cond}: func must be either a callable or a string")
                
        # 检查类条件：必须是当前实例的确切类型
        if cond.cls is not None:
            if inspect.isclass(cond.cls):
                if type(instance) != cond.cls:
                    return start_rank
            elif isinstance(cond.cls, str):
                if type(instance).__name__ != cond.cls:
                    return start_rank
            else:
                raise TypeError(f"Caching condition {cond}: cls must be either a class or a string")
                
        # 检查基类条件：检查实例是否为指定基类的子类
        if cond.base_cls is not None:
            if inspect.isclass(cond.base_cls) or isinstance(cond.base_cls, str):
                if not checks.is_instance_of(instance, cond.base_cls):
                    return start_rank
            else:
                raise TypeError(f"Caching condition {cond}: base_cls must be either a class or a string")
                
        # 检查标志条件：所有指定的标志都必须匹配
        if cond.flags is not None:
            if not isinstance(cond.flags, dict):
                raise TypeError(f"Caching condition {cond}: flags must be a dict")
            for k, v in cond.flags.items():
                if k not in flags or flags[k] != v:
                    return start_rank
                    
        # 如果条件指定了自定义排序值，使用自定义值覆盖默认排序
        if cond.rank is not None:
            if not isinstance(cond.rank, int):
                raise TypeError(f"Caching condition {cond}: rank must be an integer")
            # 创建一个包含12个相同自定义排序值的列表
            # 这样无论匹配到哪种条件组合，都使用相同的自定义优先级
            ranks = [cond.rank for _ in range(12)]
        else:
            # 使用默认的优先级排序：0-11，数字越小优先级越高
            ranks = list(range(12))

        # 根据匹配的条件组合返回相应的优先级排序值
        if cond.instance is not None and cond.func is not None:
            return ranks[0]
        if cond.instance is not None and cond.flags is not None:
            return ranks[1]
        if cond.instance is not None:
            return ranks[2]
        if cond.cls is not None and cond.func is not None:
            return ranks[3]
        if cond.cls is not None and cond.flags is not None:
            return ranks[4]
        if cond.cls is not None:
            return ranks[5]
        if cond.base_cls is not None and cond.func is not None:
            return ranks[6]
        if cond.base_cls is not None and cond.flags is not None:
            return ranks[7]
        if cond.base_cls is not None:
            return ranks[8]
        if cond.func is not None and cond.flags is not None:
            return ranks[9]
        if cond.func is not None:
            return ranks[10]
        if cond.flags is not None:
            return ranks[11]

        return start_rank

    # 评估白名单条件，找到最高优先级（最小排序值）的匹配条件
    white_rank = start_rank
    if len(caching_cfg['whitelist']) > 0:
        # 遍历所有白名单条件，找到优先级最高的匹配条件
        for cond in caching_cfg['whitelist']:
            white_rank = min(white_rank, _get_condition_rank(cond))

    # 评估黑名单条件，找到最高优先级（最小排序值）的匹配条件
    black_rank = start_rank
    if len(caching_cfg['blacklist']) > 0:
        # 遍历所有黑名单条件，找到优先级最高的匹配条件
        for cond in caching_cfg['blacklist']:
            black_rank = min(black_rank, _get_condition_rank(cond))

    # 根据白名单和黑名单的匹配结果决定是否缓存
    if white_rank == black_rank:  
        # 如果白名单和黑名单的优先级相同（包括都未匹配的情况）
        # 使用全局缓存开关决定
        return caching_cfg['enabled']  
    # 如果优先级不同，白名单优先级更高则启用缓存，否则禁用缓存
    return white_rank < black_rank


_NOT_FOUND = object()


class cached_property(custom_property):

    def __init__(self, func: tp.Callable, **flags) -> None:
        """
        初始化 cached_property 实例。
        
        Args:
            func (tp.Callable): 要缓存的属性方法
            **flags: 额外的标志参数，用于缓存条件匹配
        """
        super().__init__(func, **flags)
        # 使用 RLock（可重入锁）而不是普通 Lock
        # RLock 允许同一线程多次获取锁，避免死锁问题
        # 这在递归调用或嵌套访问同一属性时特别重要
        self.lock = RLock()

    def clear_cache(self, instance: object) -> None:
        """Clear the cache for this property belonging to `instance`."""
        if hasattr(instance, self.attrname):
            delattr(instance, self.attrname)

    @property
    def attrname(self) -> str:
        """Get name of cached attribute."""
        return '__cached_' + self.name

    def __set_name__(self, owner: tp.Type, name: str) -> None:
        """
        设置描述符在所属类中的属性名
        
        Args:
            owner (tp.Type): 所属的类
            name (str): 属性名称
        例子：
            class MyDescriptor:
                def __set_name__(self, owner, name):...  
                def __get__(self, instance, owner):...

            class MyClass:
                # 该行代码执行时，Python 自动调用 my_attr.__set_name__(MyClass, 'my_attr')
                my_attr = MyDescriptor()
        """
        
        self.name = name

    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        if instance is None:
            return self
        if not should_cache(self.name, instance, func=self.func, **self.flags):
            return super().__get__(instance, owner=owner)
        cache = instance.__dict__
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.func(instance)
                    cache[self.attrname] = val
        return val

    def __call__(self, *args, **kwargs) -> tp.Any:
        ...


class custom_methodT(tp.Protocol):
    func: tp.Callable
    flags: tp.Dict

    def __call__(self, *args, **kwargs) -> tp.Any:
        ...


def custom_method(*args, **flags) -> tp.Union[tp.Callable, custom_methodT]:
    """Custom extensible method that stores function and flags as attributes.

    Can be called both as
    ```pycon
    >>> @cached_method
    ... def user_function(): pass
    ```
    and
    ```pycon
    >>> @cached_method(maxsize=128, typed=False, a=0, b=0)  # flags
    ... def user_function(): pass
    ```
    """

    def decorator(func: tp.Callable) -> custom_methodT:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            return func(*args, **kwargs)

        wrapper.func = func
        wrapper.flags = flags

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


class cached_methodT(custom_methodT):
    maxsize: int
    typed: bool
    name: str
    attrname: str
    lock: RLock
    clear_cache: tp.Callable[[object], None]

    def __call__(self, *args, **kwargs) -> tp.Any:
        ...


def cached_method(*args, maxsize: int = 128, typed: bool = False,
                  **flags) -> tp.Union[tp.Callable, cached_methodT]:
    """Extends `custom_method` with caching.

    Internally uses `functools.lru_cache`.

    Disables caching if `should_cache` yields False or a non-hashable object
    as argument has been passed.

    See notes on `cached_property`."""

    def decorator(func: tp.Callable) -> cached_methodT:
        @wraps(func)
        def wrapper(instance: object, *args, **kwargs) -> tp.Any:
            def partial_func(*args, **kwargs) -> tp.Any:
                # Ignores non-hashable instances
                return func(instance, *args, **kwargs)

            _func = None
            if hasattr(instance, wrapper.name):
                _func = getattr(instance, wrapper.name)
            if not should_cache(wrapper.name, instance, func=_func, **wrapper.flags):
                return func(instance, *args, **kwargs)
            cache = instance.__dict__
            cached_func = cache.get(wrapper.attrname, _NOT_FOUND)
            if cached_func is _NOT_FOUND:
                with wrapper.lock:
                    # check if another thread filled cache while we awaited lock
                    cached_func = cache.get(wrapper.attrname, _NOT_FOUND)
                    if cached_func is _NOT_FOUND:
                        cached_func = lru_cache(maxsize=wrapper.maxsize, typed=wrapper.typed)(partial_func)
                        cache[wrapper.attrname] = cached_func  # store function instead of output

            # Check if object can be hashed
            hashable = True
            for arg in args:
                if not checks.is_hashable(arg):
                    hashable = False
                    break
            for k, v in kwargs.items():
                if not checks.is_hashable(v):
                    hashable = False
                    break
            if not hashable:
                # If not, do not invoke lru_cache
                return func(instance, *args, **kwargs)
            return cached_func(*args, **kwargs)

        def clear_cache(instance):
            """Clear the cache for this method belonging to `instance`."""
            if hasattr(instance, wrapper.attrname):
                delattr(instance, wrapper.attrname)

        wrapper.func = func
        wrapper.flags = flags
        wrapper.maxsize = maxsize
        wrapper.typed = typed
        wrapper.name = func.__name__
        wrapper.attrname = '__cached_' + func.__name__
        wrapper.lock = RLock()
        wrapper.clear_cache = clear_cache

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


# ############# Magic methods ############# #

WrapperFuncT = tp.Callable[[tp.Type[tp.T]], tp.Type[tp.T]]

__pdoc__ = {}

binary_magic_config = Config(
    {
        '__eq__': dict(func=np.equal),
        '__ne__': dict(func=np.not_equal),
        '__lt__': dict(func=np.less),
        '__gt__': dict(func=np.greater),
        '__le__': dict(func=np.less_equal),
        '__ge__': dict(func=np.greater_equal),
        # arithmetic ops
        '__add__': dict(func=np.add),
        '__sub__': dict(func=np.subtract),
        '__mul__': dict(func=np.multiply),
        '__pow__': dict(func=np.power),
        '__mod__': dict(func=np.mod),
        '__floordiv__': dict(func=np.floor_divide),
        '__truediv__': dict(func=np.true_divide),
        '__radd__': dict(func=lambda x, y: np.add(y, x)),
        '__rsub__': dict(func=lambda x, y: np.subtract(y, x)),
        '__rmul__': dict(func=lambda x, y: np.multiply(y, x)),
        '__rpow__': dict(func=lambda x, y: np.power(y, x)),
        '__rmod__': dict(func=lambda x, y: np.mod(y, x)),
        '__rfloordiv__': dict(func=lambda x, y: np.floor_divide(y, x)),
        '__rtruediv__': dict(func=lambda x, y: np.true_divide(y, x)),
        # mask ops
        '__and__': dict(func=np.bitwise_and),
        '__or__': dict(func=np.bitwise_or),
        '__xor__': dict(func=np.bitwise_xor),
        '__rand__': dict(func=lambda x, y: np.bitwise_and(y, x)),
        '__ror__': dict(func=lambda x, y: np.bitwise_or(y, x)),
        '__rxor__': dict(func=lambda x, y: np.bitwise_xor(y, x))
    },
    readonly=True,
    as_attrs=False
)
"""_"""

__pdoc__['binary_magic_config'] = f"""Config of binary magic methods to be added to a class.

```json
{binary_magic_config.to_doc()}
```
"""

BinaryTranslateFuncT = tp.Callable[[tp.Any, tp.Any, tp.Callable], tp.Any]


def attach_binary_magic_methods(translate_func: BinaryTranslateFuncT,
                                config: tp.Optional[Config] = None) -> WrapperFuncT:
    """Class decorator to add binary magic methods to a class.

    `translate_func` should

    * take `self`, `other`, and unary function,
    * perform computation, and
    * return the result.

    `config` defaults to `binary_magic_config` and should contain target method names (keys)
    and dictionaries (values) with the following keys:

    * `func`: Function that combines two array-like objects.
    """
    if config is None:
        config = binary_magic_config

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        for target_name, settings in config.items():
            func = settings['func']

            def new_method(self,
                           other: tp.Any,
                           _translate_func: BinaryTranslateFuncT = translate_func,
                           _func: tp.Callable = func) -> tp.SeriesFrame:
                return _translate_func(self, other, _func)

            new_method.__qualname__ = f"{cls.__name__}.{target_name}"
            new_method.__name__ = target_name
            setattr(cls, target_name, new_method)
        return cls

    return wrapper


unary_magic_config = Config(
    {
        '__neg__': dict(func=np.negative),
        '__pos__': dict(func=np.positive),
        '__abs__': dict(func=np.absolute),
        '__invert__': dict(func=np.invert)
    },
    readonly=True,
    as_attrs=False
)
"""_"""

__pdoc__['unary_magic_config'] = f"""Config of unary magic methods to be added to a class.

```json
{unary_magic_config.to_doc()}
```
"""

UnaryTranslateFuncT = tp.Callable[[tp.Any, tp.Callable], tp.Any]


def attach_unary_magic_methods(translate_func: UnaryTranslateFuncT,
                               config: tp.Optional[Config] = None) -> WrapperFuncT:
    """Class decorator to add unary magic methods to a class.

    `translate_func` should

    * take `self` and unary function,
    * perform computation, and
    * return the result.

    `config` defaults to `unary_magic_config` and should contain target method names (keys)
    and dictionaries (values) with the following keys:

    * `func`: Function that transforms one array-like object.
    """
    if config is None:
        config = unary_magic_config

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        for target_name, settings in config.items():
            func = settings['func']

            def new_method(self,
                           _translate_func: UnaryTranslateFuncT = translate_func,
                           _func: tp.Callable = func) -> tp.SeriesFrame:
                return _translate_func(self, _func)

            new_method.__qualname__ = f"{cls.__name__}.{target_name}"
            new_method.__name__ = target_name
            setattr(cls, target_name, new_method)
        return cls

    return wrapper
