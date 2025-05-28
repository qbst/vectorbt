# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT UTILS MODULE: DECORATORS
================================================================================

文件作用概述：
本文件是vectorbt库中装饰器系统的核心实现，为整个库提供了一套完整的装饰器基础设施。
在vectorbt这样的高性能量化分析库中，装饰器不仅用于代码重用和简化，更重要的是用于
性能优化、缓存管理和动态行为控制，是支撑库高效运行的关键技术组件。

主要功能模块：

一、灵活属性装饰器系统：
- class_or_instancemethod: 支持类方法和实例方法双重调用的装饰器
- classproperty: 类级别属性装饰器，支持类属性访问
- class_or_instanceproperty: 类和实例双重属性装饰器
- custom_property: 可扩展的自定义属性基类，支持标志存储

二、智能缓存管理系统：
- CacheCondition: 缓存条件描述符，支持多维度条件定义
- should_cache: 核心缓存决策函数，基于12级优先级的条件匹配系统
- cached_property: 线程安全的属性缓存装饰器，支持条件缓存控制
- cached_method: 基于LRU的方法缓存装饰器，支持可哈希性检测

三、魔术方法批量生成系统：
- attach_binary_magic_methods: 二元运算符魔术方法批量添加装饰器
- attach_unary_magic_methods: 一元运算符魔术方法批量添加装饰器
- binary_magic_config/unary_magic_config: 预定义的魔术方法配置
"""

import inspect
from functools import wraps, lru_cache
from threading import RLock

import numpy as np

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import Config


class class_or_instancemethod(classmethod):
    '''参考externalLib/vectorbt/vectorbt/笔记/utils/decorators.ipynb'''
    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, owner)


class classproperty(object):
    """类属性装饰器：创建只能在类级别访问的只读属性。
    
    这个装饰器基于Python的描述符协议实现，用于创建类级别的属性，类似于@property
    装饰器，但只能在类上调用而不能在实例上调用。它在vectorbt中用于定义类的元数据、
    配置信息或工厂方法，确保这些属性属于类本身而非实例。
    
    技术实现：
    通过实现__get__和__set__方法，该描述符确保：
    1. 访问时总是将类对象（owner）作为参数传递给被装饰的函数
    2. 禁止设置操作，保持属性的只读特性
    3. 无论从类还是实例访问，都返回相同的结果
    
    应用场景：
    - 类元数据：版本号、作者信息、类型标识等不变的类属性
    - 配置常量：类级别的配置参数，所有实例共享
    - 计算属性：基于类信息计算的动态属性，如子类数量、继承层次等
    
    Examples:
        >>> class QuantStrategy:
        ...     _version = "2.1.0"
        ...     _strategies = ['momentum', 'mean_reversion', 'arbitrage']
        ...     
        ...     @classproperty
        ...     def version(cls):
        ...         '''获取策略库版本号'''
        ...         return f"QuantStrategy v{cls._version}"
        ...     
        ...     @classproperty
        ...     def available_strategies(cls):
        ...         '''获取可用策略列表'''
        ...         return len(cls._strategies)
        ...     
        ...     @classproperty
        ...     def class_info(cls):
        ...         '''获取类的详细信息'''
        ...         return {
        ...             'name': cls.__name__,
        ...             'module': cls.__module__,
        ...             'strategies_count': len(cls._strategies)
        ...         }
        ...
        >>> # 从类访问
        >>> print(QuantStrategy.version)
        QuantStrategy v2.1.0
        >>> print(QuantStrategy.available_strategies)
        3
        
        >>> # 从实例访问（结果相同）
        >>> strategy = QuantStrategy()
        >>> print(strategy.version)
        QuantStrategy v2.1.0
        >>> print(strategy.class_info)
        {'name': 'QuantStrategy', 'module': '__main__', 'strategies_count': 3}
        
        >>> # 尝试设置会抛出异常
        >>> try:
        ...     QuantStrategy.version = "3.0.0"
        ... except AttributeError as e:
        ...     print(f"错误: {e}")
        错误: can't set attribute
    """

    def __init__(self, func: tp.Callable) -> None:
        """初始化类属性装饰器。
        
        Args:
            func (tp.Callable): 被装饰的函数，该函数应该接受一个类对象作为参数
                              并返回计算后的属性值。函数的第一个参数通常命名为cls。
        """
        # 保存被装饰的函数引用，用于后续调用
        self.func = func
        
        # 复制原函数的文档字符串到描述符对象，保持文档的完整性
        # 这样通过help()或__doc__访问时能显示原函数的文档
        self.__doc__ = getattr(func, '__doc__')

    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        """描述符协议的获取方法：当访问属性时调用。
        
        该方法是描述符协议的核心，无论是从类还是从实例访问该属性，
        都会调用此方法。它确保被装饰的函数总是接收类对象作为参数。
        
        Args:
            instance (object): 访问该属性的实例对象
                             - 从类访问时为None（如 MyClass.property）
                             - 从实例访问时为实例对象（如 my_instance.property）
            owner (tp.Optional[tp.Type], optional): 拥有该属性的类对象
                                                   即定义该属性的类，通常不为None
        
        Returns:
            tp.Any: 被装饰函数的返回值，即计算后的属性值
        """
        # 调用被装饰的函数，将类对象（owner）作为参数传递
        # 注意这里忽略了instance参数，因为类属性不依赖于具体实例
        return self.func(owner)

    def __set__(self, instance: object, value: tp.Any) -> None:
        """描述符协议的设置方法：禁止设置属性值。
        
        该方法确保类属性是只读的，任何尝试设置该属性的操作都会抛出异常。
        这样可以防止意外修改类级别的重要属性。
        
        Args:
            instance (object): 尝试设置属性的对象（类或实例）
            value (tp.Any): 尝试设置的值
            
        Raises:
            AttributeError: 总是抛出此异常，提示属性不可设置
        """
        # 抛出异常，明确告知该属性是只读的
        raise AttributeError("can't set attribute")


class class_or_instanceproperty(object):
    """类或实例属性装饰器：根据调用方式智能绑定到类或实例的灵活属性。
    
    这个装饰器结合了classproperty和普通property的功能，创建了一个既可以作为类属性
    又可以作为实例属性的智能属性。当从类访问时，它将类对象传递给被装饰的函数；当从
    实例访问时，它将实例对象传递给函数。这种设计在vectorbt中用于创建既可以获取类级别
    信息又可以获取实例级别信息的多态属性。
    
    技术实现：
    基于Python描述符协议，通过检查instance参数是否为None来决定传递给被装饰函数的参数：
    1. instance为None（类访问）：传递owner（类对象）
    2. instance不为None（实例访问）：传递instance（实例对象）
    3. 禁止设置操作，保持属性的只读特性
    
    应用场景：
    - 动态配置：根据调用方式返回类级别或实例级别的配置
    - 状态信息：显示类的总体状态或特定实例的状态
    - 计算属性：基于类或实例信息计算的动态属性
    - 工厂属性：根据调用方式创建不同类型的对象
    
    Examples:
        >>> class Portfolio:
        ...     total_portfolios = 0
        ...     
        ...     def __init__(self, name, balance=0):
        ...         self.name = name
        ...         self.balance = balance
        ...         Portfolio.total_portfolios += 1
        ...         self.portfolio_id = Portfolio.total_portfolios
        ...     
        ...     @class_or_instanceproperty
        ...     def info(cls_or_self):
        ...         '''获取投资组合信息'''
        ...         if isinstance(cls_or_self, type):
        ...             # 从类调用：返回类级别信息
        ...             return {
        ...                 'type': 'Portfolio Class',
        ...                 'total_portfolios': cls_or_self.total_portfolios,
        ...                 'class_name': cls_or_self.__name__
        ...             }
        ...         else:
        ...             # 从实例调用：返回实例级别信息
        ...             return {
        ...                 'type': 'Portfolio Instance',
        ...                 'name': cls_or_self.name,
        ...                 'balance': cls_or_self.balance,
        ...                 'portfolio_id': cls_or_self.portfolio_id
        ...             }
        ...     
        ...     @class_or_instanceproperty
        ...     def status(cls_or_self):
        ...         '''获取状态信息'''
        ...         if isinstance(cls_or_self, type):
        ...             return f"总共创建了 {cls_or_self.total_portfolios} 个投资组合"
        ...         else:
        ...             status = "盈利" if cls_or_self.balance > 0 else "亏损" if cls_or_self.balance < 0 else "平衡"
        ...             return f"投资组合 '{cls_or_self.name}': {status}"
        ...
        >>> # 从类访问
        >>> print(Portfolio.info)
        {'type': 'Portfolio Class', 'total_portfolios': 0, 'class_name': 'Portfolio'}
        >>> print(Portfolio.status)
        总共创建了 0 个投资组合
        
        >>> # 创建实例并从实例访问
        >>> portfolio1 = Portfolio("成长型", 10000)
        >>> portfolio2 = Portfolio("价值型", -2000)
        
        >>> print(portfolio1.info)
        {'type': 'Portfolio Instance', 'name': '成长型', 'balance': 10000, 'portfolio_id': 1}
        >>> print(portfolio1.status)
        投资组合 '成长型': 盈利
        
        >>> print(portfolio2.status)
        投资组合 '价值型': 亏损
        
        >>> # 现在从类访问会显示更新的信息
        >>> print(Portfolio.status)
        总共创建了 2 个投资组合
        
        >>> # 尝试设置会抛出异常
        >>> try:
        ...     Portfolio.info = "new value"
        ... except AttributeError as e:
        ...     print(f"错误: {e}")
        错误: can't set attribute
    """

    def __init__(self, func: tp.Callable) -> None:
        """初始化类或实例属性装饰器。
        
        Args:
            func (tp.Callable): 被装饰的函数，该函数应该接受一个参数（类对象或实例对象）
                              并根据参数类型返回相应的属性值。函数内部通常使用
                              isinstance(arg, type)来判断参数是类还是实例。
        """
        # 保存被装饰的函数引用，用于后续调用
        self.func = func
        
        # 复制原函数的文档字符串到描述符对象，保持文档的完整性
        # 这样通过help()或__doc__访问时能显示原函数的文档
        self.__doc__ = getattr(func, '__doc__')

    def __get__(self, instance: object, owner: tp.Optional[tp.Type] = None) -> tp.Any:
        """描述符协议的获取方法：根据调用方式智能选择传递的参数。
        
        这是该装饰器的核心方法，实现了智能参数选择逻辑。它检查访问方式
        （类访问还是实例访问），然后将相应的对象传递给被装饰的函数。
        
        Args:
            instance (object): 访问该属性的实例对象
                             - None: 表示从类访问（如 MyClass.property）
                             - 非None: 表示从实例访问（如 my_instance.property）
            owner (tp.Optional[tp.Type], optional): 拥有该属性的类对象
                                                   即定义该属性的类，通常不为None
        
        Returns:
            tp.Any: 被装饰函数的返回值，根据调用方式可能基于类或实例信息计算
        """
        # 检查是从类还是从实例访问该属性
        if instance is None:
            # 从类访问：将类对象（owner）传递给被装饰的函数
            # 这允许函数访问类级别的属性和方法
            return self.func(owner)
        
        # 从实例访问：将实例对象（instance）传递给被装饰的函数
        # 这允许函数访问实例级别的属性和方法
        return self.func(instance)

    def __set__(self, instance: object, value: tp.Any) -> None:
        """描述符协议的设置方法：禁止设置属性值。
        
        该方法确保属性是只读的，无论是类级别还是实例级别的访问都不能修改属性值。
        这样可以防止意外修改重要的计算属性或状态信息。
        
        Args:
            instance (object): 尝试设置属性的对象（类或实例）
            value (tp.Any): 尝试设置的值
            
        Raises:
            AttributeError: 总是抛出此异常，提示属性不可设置
        """
        # 抛出异常，明确告知该属性是只读的
        raise AttributeError("can't set attribute")


custom_propertyT = tp.TypeVar("custom_propertyT", bound="custom_property")


class custom_property:
    """Custom property that stores function and flags as attributes.

    Can be called both as
    ```pycon
    >>> @custom_property
    ... def user_function(self): pass
    ```
    and
    ```plaintext
    >>> @custom_property(a=0, b=0)  # flags
    ... def user_function(self): pass
    ```

    !!! note
        `custom_property` instances belong to classes, not class instances. Thus changing the property,
        for example, by disabling caching, will do the same for each instance of the class where
        the property has been defined."""

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
    """Caching condition for the use in `should_cache`."""

    instance: tp.Optional[object] = None
    """Class instance the method/property is bound to."""

    func: tp.Optional[tp.Union[tp.Callable, "cached_property", str]] = None
    """Method/property or its name (case-sensitive)."""

    cls: tp.Optional[tp.Union[type, str]] = None
    """Class of the instance or its name (case-sensitive)."""

    base_cls: tp.Optional[tp.Union[type, str]] = None
    """Base class of the class or its name (case-sensitive)."""

    flags: tp.Optional[dict] = None
    """Flags to check for in method/property's flags."""

    rank: tp.Optional[int] = None
    """Rank to override the default rank."""


def should_cache(func_name: str, instance: object, func: tp.Optional[tp.Callable] = None, **flags) -> bool:
    """Check whether to cache the method/property based on a range of conditions defined under
    `caching` in `vectorbt._settings.settings`.

    Each condition has its own rank. A narrower condition has a lower (better) rank than a broader condition.
    All supplied keys are checked, and if any condition fails, it's assigned to the highest (worst) rank.

    Here's the condition ranking:

    0) `instance` and `func`
    1) `instance` and `flags`
    2) `instance`
    3) `cls` and `func`
    4) `cls` and `flags`
    5) `cls`
    6) `base_cls` and `func`
    7) `base_cls` and `flags`
    8) `base_cls`
    9) `func` and `flags`
    10) `func`
    11) `flags`
    
    This function goes through all conditions of type `CacheCondition` in `whitelist` and `blacklist`
    and finds the one with the lowest (best) rank. If the search yields the same rank for both lists,
    global caching flag `enabled` decides.

    Usage:
        * Let's evaluate various caching conditions:

        ```pycon
        >>> import vectorbt as vbt

        >>> class A:
        ...     @cached_property(my_flag=True)
        ...     def f(self):
        ...         return None

        >>> class B(A):
        ...     @cached_property(my_flag=False)
        ...     def f(self):
        ...         return None

        >>> a = A()
        >>> b = B()

        >>> vbt.CacheCondition(instance=a, func='f')  # A.f
        >>> vbt.CacheCondition(instance=b, func='f')  # B.f
        >>> vbt.CacheCondition(instance=a, flags=dict(my_flag=True))  # A.f
        >>> vbt.CacheCondition(instance=a, flags=dict(my_flag=False))  # none
        >>> vbt.CacheCondition(instance=b, flags=dict(my_flag=False))  # B.f
        >>> vbt.CacheCondition(instance=a)  # A.f
        >>> vbt.CacheCondition(instance=b)  # B.f
        >>> vbt.CacheCondition(cls=A)  # A.f
        >>> vbt.CacheCondition(cls=B)  # B.f
        >>> vbt.CacheCondition(base_cls=A)  # A.f and B.f
        >>> vbt.CacheCondition(base_cls=B)  # B.f
        >>> vbt.CacheCondition(base_cls=A, flags=dict(my_flag=False))  # B.f
        >>> vbt.CacheCondition(func=A.f)  # A.f
        >>> vbt.CacheCondition(func=B.f)  # B.f
        >>> vbt.CacheCondition(func='f')  # A.f and B.f
        >>> vbt.CacheCondition(func='f', flags=dict(my_flag=False))  # B.f
        >>> vbt.CacheCondition(flags=dict(my_flag=True))  # A.f
        ```
    """
    from vectorbt._settings import settings
    caching_cfg = settings['caching']

    start_rank = 100

    def _get_condition_rank(cond: CacheCondition) -> int:
        # Perform initial checks
        checks.assert_instance_of(cond, CacheCondition)

        if cond.instance is not None:
            if instance is not cond.instance:
                return start_rank
        if cond.func is not None:
            if isinstance(cond.func, cached_property):  # cached_property
                if func != cond.func.func:
                    return start_rank
            elif callable(cond.func) and hasattr(func, 'func') and hasattr(cond.func, 'func'):  # cached_method
                if func.func != cond.func.func:
                    return start_rank
            elif isinstance(cond.func, str):
                if func_name != cond.func:
                    return start_rank
            else:
                raise TypeError(f"Caching condition {cond}: func must be either a callable or a string")
        if cond.cls is not None:
            if inspect.isclass(cond.cls):
                if type(instance) != cond.cls:
                    return start_rank
            elif isinstance(cond.cls, str):
                if type(instance).__name__ != cond.cls:
                    return start_rank
            else:
                raise TypeError(f"Caching condition {cond}: cls must be either a class or a string")
        if cond.base_cls is not None:
            if inspect.isclass(cond.base_cls) or isinstance(cond.base_cls, str):
                if not checks.is_instance_of(instance, cond.base_cls):
                    return start_rank
            else:
                raise TypeError(f"Caching condition {cond}: base_cls must be either a class or a string")
        if cond.flags is not None:
            if not isinstance(cond.flags, dict):
                raise TypeError(f"Caching condition {cond}: flags must be a dict")
            for k, v in cond.flags.items():
                if k not in flags or flags[k] != v:
                    return start_rank
        if cond.rank is not None:
            if not isinstance(cond.rank, int):
                raise TypeError(f"Caching condition {cond}: rank must be an integer")
            ranks = [cond.rank for _ in range(12)]
        else:
            ranks = list(range(12))

        # Rank instance conditions
        if cond.instance is not None and cond.func is not None:
            return ranks[0]
        if cond.instance is not None and cond.flags is not None:
            return ranks[1]
        if cond.instance is not None:
            return ranks[2]

        # Rank class conditions
        if cond.cls is not None and cond.func is not None:
            return ranks[3]
        if cond.cls is not None and cond.flags is not None:
            return ranks[4]
        if cond.cls is not None:
            return ranks[5]

        # Rank base class conditions
        if cond.base_cls is not None and cond.func is not None:
            return ranks[6]
        if cond.base_cls is not None and cond.flags is not None:
            return ranks[7]
        if cond.base_cls is not None:
            return ranks[8]

        # Rank function conditions
        if cond.func is not None and cond.flags is not None:
            return ranks[9]
        if cond.func is not None:
            return ranks[10]
        if cond.flags is not None:
            return ranks[11]

        return start_rank

    white_rank = start_rank
    if len(caching_cfg['whitelist']) > 0:
        for cond in caching_cfg['whitelist']:
            white_rank = min(white_rank, _get_condition_rank(cond))

    black_rank = start_rank
    if len(caching_cfg['blacklist']) > 0:
        for cond in caching_cfg['blacklist']:
            black_rank = min(black_rank, _get_condition_rank(cond))

    if white_rank == black_rank:  # none of the conditions met
        return caching_cfg['enabled']  # global caching decides
    return white_rank < black_rank


_NOT_FOUND = object()


class cached_property(custom_property):
    """Extends `custom_property` with caching.

    Similar to `functools.cached_property`, but without replacing the original attribute
    to be able to re-compute whenever needed.

    Disables caching if `should_cache` yields False.

    Cache can be cleared by calling `clear_cache` with instance as argument.

    !!! note:
        Assumes that the instance (provided as `self`) won't change. If calculation depends
        upon object attributes that can be changed, it won't notice the change."""

    def __init__(self, func: tp.Callable, **flags) -> None:
        super().__init__(func, **flags)
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
