# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
模板工具模块。用于：
1. 动态配置管理 - 允许配置项在运行时根据上下文进行替换
2. 参数化计算 - 支持在不同场景下使用相同的模板但不同的参数
3. 延迟求值 - 将计算推迟到真正需要时执行
4. 条件逻辑 - 支持基于运行时条件的动态行为

主要组件：
- Sub: 字符串模板替换，类似Python的string.Template
- Rep: 简单的键值替换
- RepEval: 表达式求值替换  
- RepFunc: 函数调用替换
- deep_substitute: 深度递归替换函数
"""

from copy import copy
from string import Template

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import set_dict_item, get_func_arg_names, merge_dicts
from vectorbt.utils.docs import SafeToStr, prepare_for_doc


class Sub(SafeToStr):
    """
    字符串模板替换类
    
    - 支持延迟求值：模板在创建时不立即求值，而是在调用substitute时才进行替换
    - 支持映射合并：可以在初始化时提供基础映射，在substitute时提供额外映射
    - 继承SafeToStr：确保在文档生成时的安全字符串表示
    
    使用示例：
    >>> sub = Sub('$greeting $name!', {'greeting': 'Hello'})
    >>> result1 = sub.substitute({'name': 'Bob', 'greeting': 'Hi'})  # Hi Bob!
    >>> result2 = sub.substitute({'name': 'Charlie'})  # Hello Charlie!
    """

    def __init__(self, template: tp.Union[str, Template], mapping: tp.Optional[tp.Mapping] = None) -> None:
        """
        初始化字符串模板替换对象
        
        Args:
            template: 模板字符串或Template对象，包含$占位符的字符串
            mapping: 可选的初始映射字典，提供替换值的基础映射
        """
        self._template = template  # 存储原始模板，可以是字符串或Template对象
        self._mapping = mapping    # 存储初始映射字典

    @property
    def template(self) -> Template:
        if not isinstance(self._template, Template):  # 检查是否已经是Template对象
            return Template(self._template)           # 如果是字符串，转换为Template对象
        return self._template                         # 如果已经是Template对象，直接返回

    @property
    def mapping(self) -> tp.Mapping:
        if self._mapping is None:  # 检查初始映射是否为None
            return {}              # 返回空字典作为默认值
        return self._mapping       # 返回实际的映射字典

    def substitute(self, mapping: tp.Optional[tp.Mapping] = None) -> str:
        """
        执行模板替换操作
        
        使用提供的映射字典替换模板中的占位符。会将初始映射和当前映射进行合并，
        当前映射中的值会覆盖初始映射中的同名键值。
        
        Args:
            mapping: 可选的映射字典，提供替换值
            
        Returns:
            str: 替换后的字符串结果
            
        示例:
            >>> sub = Sub('$name is $age years old', {'name': 'Alice'})
            >>> result = sub.substitute({'age': 25})
            >>> print(result)  # Alice is 25 years old
        """
        mapping = merge_dicts(self.mapping, mapping)  # 合并初始映射和当前映射，当前映射优先
        return self.template.substitute(mapping)      # 使用合并后的映射执行模板替换

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" \
               f"template=\"{self.template.template}\", " \
               f"mapping={prepare_for_doc(self.mapping)})"


class Rep(SafeToStr):
    """
    简单键值替换类
    
    该类提供最基础的键值替换功能，直接从映射字典中获取指定键的值。
    相比Sub类，Rep类不进行字符串模板处理，而是直接返回映射中的值。
    
    主要特性：
    - 直接键值查找：根据键名直接从映射中获取对应的值
    - 类型保持：返回值保持原始类型，不进行字符串转换
    - 映射合并：支持初始映射和运行时映射的合并
    
    使用示例：
    >>> rep = Rep('user_id', {'user_id': 12345})
    >>> result = rep.replace()
    >>> print(result)  # 12345 (整数类型)
    
    >>> # 运行时覆盖初始映射
    >>> rep = Rep('config', {'config': 'default'})
    >>> result = rep.replace({'config': 'production'})
    >>> print(result)  # 'production'
    
    应用场景：
    - 配置参数的动态获取
    - 函数参数的延迟绑定
    - 条件分支中的值选择
    - 数据结构中的占位符替换
    """

    def __init__(self, key: tp.Hashable, mapping: tp.Optional[tp.Mapping] = None) -> None:
        """
        初始化键值替换对象
        
        Args:
            key: 要替换的键名，必须是可哈希的对象（字符串、数字、元组等）
            mapping: 可选的初始映射字典，提供键值对的基础映射
        """
        self._key = key        # 存储要查找的键名
        self._mapping = mapping # 存储初始映射字典

    @property
    def key(self) -> tp.Hashable:
        """
        获取要替换的键名
        
        Returns:
            Hashable: 初始化时指定的键名
        """
        return self._key

    @property
    def mapping(self) -> tp.Mapping:
        """
        获取初始映射字典
        
        Returns:
            Mapping: 初始化时提供的映射字典，如果为None则返回空字典
        """
        if self._mapping is None:  # 检查初始映射是否为None
            return {}              # 返回空字典作为默认值
        return self._mapping       # 返回实际的映射字典

    def replace(self, mapping: tp.Optional[tp.Mapping] = None) -> tp.Any:
        """
        执行键值替换操作
        
        从合并后的映射字典中查找指定键的值并返回。如果键不存在会抛出KeyError。
        
        Args:
            mapping: 可选的映射字典，提供额外的键值对
            
        Returns:
            Any: 映射字典中对应键的值，保持原始类型
            
        Raises:
            KeyError: 当指定的键在映射中不存在时抛出
            
        示例:
            >>> rep = Rep('threshold', {'threshold': 0.5})
            >>> result = rep.replace({'threshold': 0.8})  # 0.8
        """
        mapping = merge_dicts(self.mapping, mapping)  # 合并初始映射和当前映射
        return mapping[self.key]                      # 从合并后的映射中获取指定键的值

    def __str__(self) -> str:
        """
        返回对象的字符串表示形式
        
        Returns:
            str: 包含类名、键名和映射信息的字符串
        """
        return f"{self.__class__.__name__}(" \
               f"key='{self.key}', " \
               f"mapping={prepare_for_doc(self.mapping)})"


class RepEval(SafeToStr):
    """
    表达式求值替换类
    
    该类提供Python表达式的动态求值功能，使用映射字典作为局部变量环境。
    这是一个强大但需要谨慎使用的功能，因为它涉及代码的动态执行。
    
    主要特性：
    - 动态表达式求值：支持任意Python表达式的运行时求值
    - 安全的变量环境：使用提供的映射作为局部变量，全局变量为空字典
    - 类型灵活：表达式结果可以是任意Python对象类型
    - 错误处理：求值过程中的异常会正常抛出
    
    安全注意事项：
    - 不要对不可信的表达式使用此类
    - 表达式中只能访问映射中提供的变量
    - 无法访问内置函数和全局变量（安全设计）
    
    使用示例：
    >>> rep_eval = RepEval('x + y * 2', {'x': 10, 'y': 5})
    >>> result = rep_eval.eval()
    >>> print(result)  # 20
    
    >>> # 条件表达式
    >>> rep_eval = RepEval('max_val if use_max else min_val')
    >>> result = rep_eval.eval({'max_val': 100, 'min_val': 1, 'use_max': True})
    >>> print(result)  # 100
    
    >>> # 列表推导式
    >>> rep_eval = RepEval('[x*2 for x in numbers if x > 0]')
    >>> result = rep_eval.eval({'numbers': [-1, 2, -3, 4]})
    >>> print(result)  # [4, 8]
    
    应用场景：
    - 动态计算配置值
    - 条件逻辑的表达式化
    - 复杂的数据转换规则
    - 运行时的公式计算
    """

    def __init__(self, expression: str, mapping: tp.Optional[tp.Mapping] = None) -> None:
        """
        初始化表达式求值对象
        
        Args:
            expression: 要求值的Python表达式字符串
            mapping: 可选的初始映射字典，作为表达式的局部变量环境
        """
        self._expression = expression  # 存储要求值的表达式字符串
        self._mapping = mapping        # 存储初始映射字典

    @property
    def expression(self) -> str:
        """
        获取要求值的表达式字符串
        
        Returns:
            str: 初始化时指定的Python表达式
        """
        return self._expression

    @property
    def mapping(self) -> tp.Mapping:
        """
        获取初始映射字典
        
        Returns:
            Mapping: 初始化时提供的映射字典，如果为None则返回空字典
        """
        if self._mapping is None:  # 检查初始映射是否为None
            return {}              # 返回空字典作为默认值
        return self._mapping       # 返回实际的映射字典

    def eval(self, mapping: tp.Optional[tp.Mapping] = None) -> tp.Any:
        """
        执行表达式求值操作
        
        使用Python内置的eval函数对表达式进行求值，将合并后的映射作为局部变量环境。
        全局变量环境设置为空字典以提高安全性。
        
        Args:
            mapping: 可选的映射字典，提供额外的局部变量
            
        Returns:
            Any: 表达式求值的结果，类型取决于表达式
            
        Raises:
            各种Python异常: 根据表达式内容可能抛出NameError、TypeError等异常
            
        示例:
            >>> rep_eval = RepEval('len(data) > threshold')
            >>> result = rep_eval.eval({'data': [1,2,3], 'threshold': 2})  # True
        """
        mapping = merge_dicts(self.mapping, mapping)  # 合并初始映射和当前映射
        return eval(self.expression, {}, mapping)     # 使用空的全局环境和合并后的局部环境求值

    def __str__(self) -> str:
        """
        返回对象的字符串表示形式
        
        Returns:
            str: 包含类名、表达式和映射信息的字符串
        """
        return f"{self.__class__.__name__}(" \
               f"expression=\"{self.expression}\", " \
               f"mapping={prepare_for_doc(self.mapping)})"


class RepFunc(SafeToStr):
    """
    函数调用替换类
    
    该类提供函数的延迟调用功能，根据映射字典中的值作为参数来调用指定的函数。
    只有函数签名中存在的参数才会从映射中提取并传递给函数。
    
    主要特性：
    - 智能参数匹配：自动检查函数签名，只传递函数需要的参数
    - 延迟调用：函数在创建对象时不执行，而是在调用call方法时才执行
    - 参数过滤：映射中多余的键值对不会影响函数调用
    - 类型安全：保持函数原有的参数类型和返回类型
    
    使用示例：
    >>> def calculate(x, y, operation='add'):
    ...     if operation == 'add':
    ...         return x + y
    ...     elif operation == 'multiply':
    ...         return x * y
    
    >>> rep_func = RepFunc(calculate, {'x': 10, 'y': 5})
    >>> result = rep_func.call({'operation': 'multiply'})
    >>> print(result)  # 50
    
    >>> # 函数只接收它需要的参数
    >>> def greet(name, title='Mr'):
    ...     return f"{title} {name}"
    
    >>> rep_func = RepFunc(greet)
    >>> result = rep_func.call({'name': 'Smith', 'title': 'Dr', 'age': 30})  # age被忽略
    >>> print(result)  # "Dr Smith"
    
    应用场景：
    - 策略模式的实现
    - 配置化的函数调用
    - 插件系统中的函数执行
    - 延迟计算和惰性求值
    """

    def __init__(self, func: tp.Callable, mapping: tp.Optional[tp.Mapping] = None) -> None:
        """
        初始化函数调用对象
        
        Args:
            func: 要调用的函数对象，必须是可调用对象
            mapping: 可选的初始映射字典，提供函数参数的基础值
        """
        self._func = func       # 存储要调用的函数对象
        self._mapping = mapping # 存储初始映射字典

    @property
    def func(self) -> tp.Callable:
        """
        获取要调用的函数对象
        
        Returns:
            Callable: 初始化时指定的函数对象
        """
        return self._func

    @property
    def mapping(self) -> tp.Mapping:
        """
        获取初始映射字典
        
        Returns:
            Mapping: 初始化时提供的映射字典，如果为None则返回空字典
        """
        if self._mapping is None:  # 检查初始映射是否为None
            return {}              # 返回空字典作为默认值
        return self._mapping       # 返回实际的映射字典

    def call(self, mapping: tp.Optional[tp.Mapping] = None) -> tp.Any:
        """
        执行函数调用操作
        
        首先获取函数的参数名列表，然后从合并后的映射中提取函数需要的参数，
        最后使用这些参数调用函数。
        
        Args:
            mapping: 可选的映射字典，提供额外的函数参数
            
        Returns:
            Any: 函数调用的返回值，类型取决于被调用的函数
            
        示例:
            >>> def power(base, exponent=2):
            ...     return base ** exponent
            >>> rep_func = RepFunc(power, {'base': 3})
            >>> result = rep_func.call({'exponent': 4})  # 81
        """
        mapping = merge_dicts(self.mapping, mapping)  # 合并初始映射和当前映射
        func_arg_names = get_func_arg_names(self.func)  # 获取函数的参数名列表
        func_kwargs = dict()                          # 创建空的关键字参数字典
        for k, v in mapping.items():                  # 遍历合并后的映射
            if k in func_arg_names:                   # 检查键是否是函数的参数名
                func_kwargs[k] = v                    # 如果是，则添加到函数参数字典中
        return self.func(**func_kwargs)               # 使用提取的参数调用函数

    def __str__(self) -> str:
        """
        返回对象的字符串表示形式
        
        Returns:
            str: 包含类名、函数对象和映射信息的字符串
        """
        return f"{self.__class__.__name__}(" \
               f"func={self.func}, " \
               f"mapping={prepare_for_doc(self.mapping)})"


def has_templates(obj: tp.Any) -> tp.Any:
    """
    检查对象是否包含模板元素
    
    该函数递归地检查对象及其嵌套结构，判断是否包含任何模板类的实例。
    支持检查字典、列表、元组、集合等容器类型的嵌套结构。
    
    Args:
        obj: 要检查的对象，可以是任意类型
        
    Returns:
        bool: 如果对象或其嵌套结构中包含模板元素则返回True，否则返回False
        
    检查的模板类型：
        - RepFunc: 函数调用模板
        - RepEval: 表达式求值模板  
        - Rep: 键值替换模板
        - Sub: 字符串模板替换
        - Template: Python标准库模板
        
    示例:
        >>> has_templates("hello")  # False
        >>> has_templates(Rep('key'))  # True
        >>> has_templates([1, 2, Rep('key')])  # True
        >>> has_templates({'a': Sub('$name')})  # True
    """
    if isinstance(obj, RepFunc):     # 检查是否是函数调用模板
        return True
    if isinstance(obj, RepEval):     # 检查是否是表达式求值模板
        return True
    if isinstance(obj, Rep):         # 检查是否是键值替换模板
        return True
    if isinstance(obj, Sub):         # 检查是否是字符串模板替换
        return True
    if isinstance(obj, Template):    # 检查是否是Python标准库模板
        return True
    if isinstance(obj, dict):        # 如果是字典类型
        for k, v in obj.items():     # 遍历字典的所有键值对
            if has_templates(v):     # 递归检查值是否包含模板（注意：不检查键）
                return True
    if isinstance(obj, (tuple, list, set, frozenset)):  # 如果是序列或集合类型
        for v in obj:                # 遍历容器中的所有元素
            if has_templates(v):     # 递归检查每个元素是否包含模板
                return True
    return False                     # 如果没有找到任何模板元素，返回False


def deep_substitute(obj: tp.Any,
                    mapping: tp.Optional[tp.Mapping] = None,
                    safe: bool = False,
                    make_copy: bool = True) -> tp.Any:
    """
    深度递归替换函数
    
    该函数是模板系统的核心功能，能够递归地遍历复杂的数据结构，
    找到其中的模板对象并进行替换。支持嵌套的字典、列表、元组、集合等数据结构。
    
    主要特性：
    - 递归遍历：能够处理任意深度的嵌套数据结构
    - 类型保持：替换后保持原有的数据结构类型
    - 安全模式：可选的错误处理模式
    - 复制控制：可控制是否创建对象副本
    
    Args:
        obj: 要处理的对象，可以包含模板元素的任意数据结构
        mapping: 可选的映射字典，提供模板替换所需的值
        safe: 安全模式标志，为True时遇到错误不抛出异常而是返回原模板
        make_copy: 是否创建对象副本，为True时会复制容器对象以避免修改原对象
        
    Returns:
        Any: 替换后的对象，如果没有模板元素则返回原对象
    
    使用示例：
        >>> # 基本替换
        >>> result = deep_substitute(Sub('$key', {'key': 100}))
        >>> print(result)  # '100'
        
        >>> # 覆盖映射
        >>> result = deep_substitute(Sub('$key', {'key': 100}), {'key': 200})
        >>> print(result)  # '200'
        
        >>> # 复杂数据结构
        >>> data = {
        ...     'config': Rep('env'),
        ...     'values': [Sub('$prefix_$suffix'), Rep('count')],
        ...     'meta': {'template': RepEval('x * 2')}
        ... }
        >>> mapping = {'env': 'prod', 'prefix': 'data', 'suffix': 'file', 'count': 42, 'x': 21}
        >>> result = deep_substitute(data, mapping)
        >>> print(result)  # {'config': 'prod', 'values': ['data_file', 42], 'meta': {'template': 42}}

    """
    if mapping is None:
        mapping = {}
    if not has_templates(obj):
        return obj
    try:
        if isinstance(obj, RepFunc):  
            return obj.call(mapping)
        if isinstance(obj, RepEval):
            return obj.eval(mapping)
        if isinstance(obj, Rep):
            return obj.replace(mapping) 
        if isinstance(obj, Sub):
            return obj.substitute(mapping)
        if isinstance(obj, Template):
            return obj.substitute(mapping)
        if isinstance(obj, dict):                       # 如果是字典类型
            if make_copy:
                obj = copy(obj)                         # 创建字典的浅复制
            for k, v in obj.items():
                set_dict_item(obj, k, deep_substitute(v, mapping=mapping, safe=safe), force=True)  # 递归替换值并强制设置
            return obj
        if isinstance(obj, list):                       # 如果是列表类型
            if make_copy:
                obj = copy(obj)                         # 创建列表的浅复制
            for i in range(len(obj)):
                obj[i] = deep_substitute(obj[i], mapping=mapping, safe=safe)  # 递归替换每个元素
            return obj
        if isinstance(obj, (tuple, set, frozenset)): 
            result = [] 
            for o in obj:
                result.append(deep_substitute(o, mapping=mapping, safe=safe))  # 递归替换每个元素并添加到结果列表
            if checks.is_namedtuple(obj):               # 检查是否是命名元组
                return type(obj)(*result)               # 如果是命名元组，使用原类型和结果重新构造
            return type(obj)(result)                    # 否则使用原类型构造新的容器对象
    except Exception as e:
        if not safe:                                    # 如果不是安全模式
            raise e                                     # 重新抛出异常
    return obj                                          # 如果是安全模式或没有模板，返回原对象
