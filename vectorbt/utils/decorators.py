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
    方法装饰器，类/实例方法形式调用
    @class_or_instancemethod
    def method():...    # method = class_or_instancemethod(method)
    
    obj.method(***)     # 等价于 method.__get__(obj, type(obj)).__call__(***)   绑定到实例后调用
    Class.method(***)   # 等价于 method.__get__(cls, type(cls)).__call__(***)   绑定到类后调用
                        #   或者 MethodType(method, cls).__call__(***)
    """
    # 根据 instance 是否为 None，将 method 绑定到类/实例上
    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, owner)


class classproperty(object):
    """
    方法装饰器，类/实例属性形式调用
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
    方法装饰器，类/实例属性形式调用
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
    方法装饰器，实例属性形式调用
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
    使用配置的缓存条件 settings['caching'] 中的白名单['whitelist']和黑名单['blacklist']，
    考察当前的方法调用func_name, instance, func, **flags，分别获得在白名单和黑名单中的最高优先级（最小排序值）数值，
    如果白名单优先级更高则启用缓存，否则禁用缓存。
    
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

    # 评估白名单条件，计算最高优先级（最小排序值）数值
    white_rank = start_rank
    if len(caching_cfg['whitelist']) > 0:
        for cond in caching_cfg['whitelist']:
            white_rank = min(white_rank, _get_condition_rank(cond))

    # 评估黑名单条件，计算最高优先级（最小排序值）数值
    black_rank = start_rank
    if len(caching_cfg['blacklist']) > 0:
        for cond in caching_cfg['blacklist']:
            black_rank = min(black_rank, _get_condition_rank(cond))

    if white_rank == black_rank:  
        return caching_cfg['enabled']  
    return white_rank < black_rank


_NOT_FOUND = object()


class cached_property(custom_property):
    """
    方法装饰器，实例属性形式调用。继承自 custom_property，在此基础上增加了将实例属性调用结果存储到实例字典的功能
    
    两种装饰方式：
        @cached_property(a=0, b=0)  # flags
        def property(): ...
        等价于 property = cached_property(a=0, b=0)(property)
        其中 cached_property(a=0, b=0) 返回一个 lambda 表达式，从而进一步构建 cls(property)
    或
        @cached_property
        def property(): ...
        
    
    obj.property        # 等价于 property(obj)，并且会将 '__cached_property': property(obj) 写入 obj.__dict__
    Class.property      # 返回该描述符property本身
    obj.property = 1    # 抛出异常
    obj.property(***)   # pass，待实现
    """

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
        设置描述符被创建时的名称

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
        """
        调用 self.func(instance)。如果启用了缓存，还会将('__cached_' + self.name: self.func(instance)) 写入instance.__dict__
        """
        if instance is None:
            return self
        
        # 使用配置的缓存条件 settings['caching'] 中的白名单['whitelist']和黑名单['blacklist']，
        # 考察当前的方法调用self.name, instance, self.func, self.flags，分别获得在白名单和黑名单中的最高优先级（最小排序值）数值，
        # 如果白名单优先级更高则启用缓存，否则禁用缓存。
        if not should_cache(self.name, instance, func=self.func, **self.flags):
            # 禁用缓存时，直接调用 self.func(instance)
            return super().__get__(instance, owner=owner)
        cache = instance.__dict__
        # 从instance.__dict__中查找self.attrname（'__cached_' + self.name）
        # 使用 _NOT_FOUND 作为默认值（唯一的哨兵对象），以区分 None 值和未找到的情况
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            # 将 ('__cached_' + self.name: self.func(instance)) 写入instance.__dict__
            # 获取锁，确保线程安全。使用 with 语句确保锁会被正确释放，即使发生异常
            with self.lock:
                # 双重检查：在获得锁后再次检查缓存
                # 这是为了防止在等待锁的过程中，其他线程已经完成了计算并更新了缓存
                # 这种模式称为"双重检查锁定"（Double-Checked Locking）
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
    """
    函数装饰器，给被装饰的函数添加 .func 和 .flags 属性
    (1) 无参装饰
    >>> @custom_method
    ... def func(): pass

    (2) 含字典参数装饰
    >>> @custom_method(flag1=value1, flag2=value2)  # flags
    ... def func(): pass
    # 获得的是 tp.Callable——>custom_methodT
    # 须手动调用 func(target_function) -> wrapper
    """

    def decorator(func: tp.Callable) -> custom_methodT:
        # 使用 functools.wraps 装饰器保持原函数的元数据
        # 这确保了 wrapper.__name__, wrapper.__doc__, wrapper.__module__ 等属性与原函数 func 保持一致
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            return func(*args, **kwargs)

        # 为包装函数添加 func 属性，存储对原函数的引用，从而允许在需要时访问未装饰的原始函数
        # 例如：decorated_func.func(*args, **kwargs) 可以直接调用原函数
        wrapper.func = func
        wrapper.flags = flags

        return wrapper

    if len(args) == 0:
        # 带参数装饰
        # @custom_method(flags) -> custom_method(**flags) -> decorator
        #   -> Python 调用 decorator(target_function) -> 装饰后的函数
        return decorator
    elif len(args) == 1:
        # 直接装饰
        # @custom_method -> custom_method(target_function) -> 装饰后的函数
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
    """
    函数装饰器：
        装饰某个函数后，该函数调用类似于 (instance, *args, **kwargs) 
        调用时，会为 instance 的 __dict__ 中添加 lru_cache 化的原函数
        即可以为 instance 保存：所有使用它的 cached_method 装饰的函数，并且具体调用参数和结果进行了LRU缓存
    
    ① 无参调用：
    >>> @cached_method
    ... def func(): pass
    # 获得的是 wrapper(instance: object, *args, **kwargs)：
        相较于 func 增加了 .func, .flags, .maxsize, .typed, .name, .attrname（'__cached_' + func.__name__ ）
                            .lock, .clear_cache 属性
        调用时
            查找 instance.__dict__ 中对应 wrapper.attrname 的方法 cached_func，如果没有则
                创建一个 lru_cache 装饰 func 后的 cached_func，并将 wrapper.attrname: cached_func 存到 instance.__dict__
            如果 args 和 kwargs 可哈希，调用 cached_func(*args, **kwargs)，否则调用 func(instance, *args, **kwargs)
                
    ② 含字典参数调用：
        >>> @cached_method(flag1=value1, flag2=value2)  # flags
        ... def func(): pass
        # 获得的是 tp.Callable——>cached_methodT
        # 须手动调用 func(target_function) -> wrapper
        
    LRU：基础是一个Hash表，put/get/remove 都是 O(1) 平均时间复杂度
        在此基础上，附加一个双向链表，将最近访问(get/put)的元素放在链表头部。
        当缓存满时，删除链表尾部的元素。
    """

    def decorator(func: tp.Callable) -> cached_methodT:
        # 使用 functools.wraps 保持原方法的元数据，确保 __name__, __doc__, __module__ 等属性正确传递
        @wraps(func)
        def wrapper(instance: object, *args, **kwargs) -> tp.Any:
            def partial_func(*args, **kwargs) -> tp.Any:
                return func(instance, *args, **kwargs)

            # 如果实例已经有了这个方法的引用，获取它，可能是之前装饰过的版本或其他相关方法
            _func = None
            if hasattr(instance, wrapper.name):
                _func = getattr(instance, wrapper.name)
            # 使用配置的缓存条件 settings['caching'] 中的白名单['whitelist']和黑名单['blacklist']，
            # 考察当前的方法调用wrapper.name, instance, _func, wrapper.flags，分别获得在白名单和黑名单中的最高优先级（最小排序值）数值，
            # 如果白名单优先级更高则启用缓存，否则禁用缓存。
            if not should_cache(wrapper.name, instance, func=_func, **wrapper.flags):
                return func(instance, *args, **kwargs)
            # 从instance.__dict__中查找wrapper.attrname
            cache = instance.__dict__
            cached_func = cache.get(wrapper.attrname, _NOT_FOUND)
            if cached_func is _NOT_FOUND:
                # 将 (wrapper.name: wrapper.func(instance)) 写入instance.__dict__
                with wrapper.lock:
                    cached_func = cache.get(wrapper.attrname, _NOT_FOUND)
                    # 如果仍然没有缓存函数，创建新的
                    if cached_func is _NOT_FOUND:
                        # 创建新的 lru_cache 装饰的函数
                        cached_func = lru_cache(maxsize=wrapper.maxsize, typed=wrapper.typed)(partial_func)
                        # 将缓存函数存储到实例字典中
                        # 注意：这里存储的是函数对象，不是函数的输出结果
                        cache[wrapper.attrname] = cached_func  # store function instead of output

            # 检查arg和kwargs所有参数是否可哈希，lru_cache 要求所有参数都必须可哈希才能正常工作
            hashable = True
            for arg in args:
                if not checks.is_hashable(arg):
                    hashable = False
                    break
            for k, v in kwargs.items():
                if not checks.is_hashable(v):
                    hashable = False
                    break
            # 如果参数不可哈希，则不使用缓存
            if not hashable:
                return func(instance, *args, **kwargs)
            # 使用缓存函数
            return cached_func(*args, **kwargs)

        def clear_cache(instance):
            """
            清除指定实例instance的wrapper.attrname方法缓存。
            """
            if hasattr(instance, wrapper.attrname):
                delattr(instance, wrapper.attrname)

        wrapper.func = func                             # 存储对原函数的引用
        wrapper.flags = flags                           # 存储装饰器的配置参数
        wrapper.maxsize = maxsize                       # 设置缓存的最大大小
        wrapper.typed = typed                           # 设置是否启用类型敏感缓存
        wrapper.name = func.__name__                    # 存储原函数的名称
        wrapper.attrname = '__cached_' + func.__name__  # 生成缓存属性的名称
        wrapper.lock = RLock()                          # 创建一个可重入锁，用于线程安全的缓存更新
        wrapper.clear_cache = clear_cache               # 存储清除缓存的方法

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


# ############# Magic methods ############# #

WrapperFuncT = tp.Callable[[tp.Type[tp.T]], tp.Type[tp.T]]

# __pdoc__ 是 pdoc 文档生成工具使用的特殊变量，用于控制哪些对象应该包含在生成的文档中，空字典表示使用默认的文档生成行为
__pdoc__ = {}

# 二元魔术方法配置对象，用于批量为类添加运算符重载功能
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
"""_""" # 空文档字符串，用于 pdoc 文档生成

# 为 binary_magic_config 添加文档字符串
# 使用 f-string 格式化，将配置内容以 JSON 格式展示在文档中
__pdoc__['binary_magic_config'] = f"""Config of binary magic methods to be added to a class.

```json
{binary_magic_config.to_doc()}
```
"""
# 二元函数类型，接受两个操作数和一个 NumPy 函数，返回一个结果
BinaryTranslateFuncT = tp.Callable[[tp.Any, tp.Any, tp.Callable], tp.Any]


def attach_binary_magic_methods(translate_func: BinaryTranslateFuncT,
                                config: tp.Optional[Config] = None) -> WrapperFuncT:
    """
    类装饰器（即装饰一个类），用于为类批量添加二元魔术方法。
    
    假设装饰一个类 cls，则：
    对于装饰参数即二元运算函数translate_func：
        对于装饰参数 config（如果None，则是 binary_magic_config）.items()中的每一项 i（类似于 '__eq__': dict(func=np.equal)）：
            构建函数 new_method(self, other)：return translate_func(self, other, i.value['func'])
            设置 new_method.__qualname__ = f"{cls.__name__}.{i.key}"
            设置 new_method.__name__ = i.key
            设置 cls.__dict__[i.key] = new_method
    返回 cls
    
    使用示例：
        # 定义转换函数 - 实现具体的运算逻辑
        def my_translate_func(self, other, numpy_func):
            '''处理两个操作数之间的运算'''
            # 将操作数转换为 NumPy 数组进行计算
            arr1 = np.asarray(self.data)
            arr2 = np.asarray(other.data if hasattr(other, 'data') else other)
            # 使用 NumPy 函数进行向量化运算
            result = numpy_func(arr1, arr2)
            # 返回与原对象相同类型的新对象
            return MyClass(result)
        
        # 应用装饰器 - 为类添加所有二元运算符
        @attach_binary_magic_methods(my_translate_func)
        class MyClass:
            def __init__(self, data):
                self.data = data
            
            def __repr__(self):
                return f"MyClass({self.data})"
        
        # 现在可以使用各种运算符
        obj1 = MyClass([1, 2, 3])
        obj2 = MyClass([4, 5, 6])
        
        # 算术运算
        result_add = obj1 + obj2      # 调用 __add__ -> [5, 7, 9]
        result_mul = obj1 * 2         # 调用 __mul__ -> [2, 4, 6]
        result_pow = obj1 ** 2        # 调用 __pow__ -> [1, 4, 9]
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
            # 相当于：cls.__dict__[target_name] = new_method
            setattr(cls, target_name, new_method)
        return cls
    return wrapper

# 一元魔术方法配置对象，用于批量为类添加一元魔术方法
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
# 一元函数类型，接受一个操作数和一个 NumPy 函数，返回一个结果
UnaryTranslateFuncT = tp.Callable[[tp.Any, tp.Callable], tp.Any]


def attach_unary_magic_methods(translate_func: UnaryTranslateFuncT,
                               config: tp.Optional[Config] = None) -> WrapperFuncT:
    """
    类装饰器（即装饰一个类），用于为类批量添加一元魔术方法。
    
    假设装饰一个类 cls，则：
    对于装饰参数即一元运算函数translate_func：
        对于装饰参数 config（如果None，则是 unary_magic_config）.items()中的每一项 i（类似于 '__neg__': dict(func=np.negative)）：
            构建函数 new_method(self)：return translate_func(self, i.value['func'])
            设置 new_method.__qualname__ = f"{cls.__name__}.{i.key}"
            设置 new_method.__name__ = i.key
            设置 cls.__dict__[i.key] = new_method
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
