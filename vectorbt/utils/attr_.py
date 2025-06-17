# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
属性访问和解析工具模块

本模块是vectorbt框架中的核心工具之一，专门用于处理复杂的属性访问、方法调用和对象解析场景。
它为整个vectorbt生态系统提供了统一的属性解析机制，特别是在量化金融分析中的复杂对象操作。

主要功能：
1. 深度链式属性访问 (deep_getattr): 支持复杂的属性链和方法调用链
2. 智能属性解析器 (AttrResolver): 提供缓存、条件参数传递和动态方法调用
3. 字典式属性访问 (get_dict_attr): 绕过标准属性查找机制的直接访问

设计理念：
- 统一性: 为不同类型的对象提供一致的属性访问接口
- 灵活性: 支持从简单的属性访问到复杂的方法调用链
- 性能优化: 内置缓存机制，避免重复计算
- 错误处理: 完善的异常处理和参数验证

应用场景：
- Portfolio对象中的动态指标计算
- GenericAccessor中的链式数据处理
- StatsBuilder中的统计指标解析
- PlotsBuilder中的可视化参数处理

技术特点：
- 支持多种属性访问模式（字符串、元组、可迭代对象）
- 智能参数匹配和传递
- 基于方法签名的动态参数过滤
- 灵活的缓存策略控制
- 完整的类型注解支持
"""

import inspect  # Python内省模块，用于检查对象类型、方法签名等
from collections.abc import Iterable  # 抽象基类，用于检查对象是否可迭代

from vectorbt import _typing as tp  # vectorbt的类型注解模块
from vectorbt.utils import checks  # vectorbt的检查工具模块
from vectorbt.utils.config import merge_dicts, get_func_arg_names  # 配置工具：字典合并和函数参数名获取


def get_dict_attr(obj, attr):
    """直接从对象的__dict__中获取属性，绕过标准的属性查找机制
    
    这个函数用于在不触发属性访问钩子（如__getattr__、__getattribute__）的情况下
    直接获取对象的属性。它会遍历对象的类继承链（MRO），寻找指定的属性。
    
    在vectorbt中主要用于：
    - 避免属性访问时的递归调用
    - 获取类的原始定义而非实例化后的属性
    - 在属性解析过程中绕过可能的属性访问拦截
    
    Args:
        obj: 目标对象，可以是类或实例
        attr: 要获取的属性名称（字符串）
        
    Returns:
        对象的原始属性值
        
    Raises:
        AttributeError: 当在整个继承链中都找不到指定属性时抛出
        
    示例:
        class A:
            x = 1
            
        class B(A):
            y = 2
            
        obj = B()
        # 即使B重写了__getattribute__，这个函数也能直接获取原始属性
        value = get_dict_attr(obj, 'x')  # 返回 1
    """
    # 确定要搜索的类：如果传入的是类，直接使用；如果是实例，获取其类
    if inspect.isclass(obj):
        cls = obj  # 如果obj本身是类，直接使用
    else:
        cls = obj.__class__  # 如果obj是实例，获取其类型
    
    # 遍历继承链：[obj] + cls.mro()确保先搜索obj本身，再搜索其类的方法解析顺序
    # MRO (Method Resolution Order) 是Python的多重继承解析顺序
    for obj in [obj] + cls.mro():
        # 检查当前对象的__dict__中是否包含目标属性
        if attr in obj.__dict__:
            return obj.__dict__[attr]  # 直接返回原始属性值
    
    # 如果在整个继承链中都未找到属性，抛出AttributeError异常
    raise AttributeError


def default_getattr_func(obj: tp.Any,
                         attr: str,
                         args: tp.Optional[tp.Args] = None,
                         kwargs: tp.Optional[tp.Kwargs] = None,
                         call_attr: bool = True) -> tp.Any:
    """默认的属性获取函数，为deep_getattr提供标准的属性访问行为
    
    这是deep_getattr函数的默认属性访问器，定义了标准的属性获取和方法调用逻辑。
    用户可以通过提供自定义的getattr_func来改变属性访问行为。
    
    处理逻辑：
    1. 使用getattr获取属性
    2. 如果属性是可调用的且call_attr为True，则调用该属性
    3. 传递相应的参数和关键字参数
    
    Args:
        obj: 目标对象
        attr: 属性名称
        args: 传递给方法的位置参数（如果属性是方法）
        kwargs: 传递给方法的关键字参数（如果属性是方法）
        call_attr: 是否调用可调用属性，默认为True
        
    Returns:
        属性值或方法调用结果
        
    示例:
        class Calculator:
            def add(self, x, y):
                return x + y
                
        calc = Calculator()
        # 获取并调用方法
        result = default_getattr_func(calc, 'add', args=(1, 2))  # 返回 3
        # 只获取方法对象，不调用
        method = default_getattr_func(calc, 'add', call_attr=False)  # 返回方法对象
    """
    # 初始化默认参数：确保args和kwargs不为None，避免后续操作出错
    if args is None:
        args = ()  # 空元组作为默认位置参数
    if kwargs is None:
        kwargs = {}  # 空字典作为默认关键字参数
    
    # 使用标准getattr获取属性
    out = getattr(obj, attr)
    
    # 判断是否需要调用属性：如果属性是可调用的且call_attr为True
    if callable(out) and call_attr:
        return out(*args, **kwargs)  # 调用方法并传递参数
    
    return out  # 直接返回属性值


def deep_getattr(obj: tp.Any,
                 attr_chain: tp.Union[str, tuple, Iterable],
                 getattr_func: tp.Callable = default_getattr_func,
                 call_last_attr: bool = True) -> tp.Any:
    """递归获取深层属性链，支持复杂的属性访问和方法调用模式
    
    这是vectorbt中最重要的属性访问工具之一，支持多种属性链格式：
    - 字符串形式的点号分隔属性链
    - 元组形式的方法调用（包含参数）
    - 可迭代对象形式的复杂调用链
    
    在量化金融分析中的应用：
    - Portfolio对象的指标计算链：'returns.rolling(20).mean()'
    - 数据处理管道：'data.dropna().fillna(0).normalize()'
    - 统计指标计算：'portfolio.trades.closed.avg_return'
    
    属性链格式说明：
    - 字符串: 'attr1.attr2.method' - 简单的属性和无参方法调用
    - 元组(string,): ('method',) - 无参方法调用
    - 元组(string, tuple): ('method', (arg1, arg2)) - 带位置参数的方法调用
    - 元组(string, tuple, dict): ('method', (args), {kwargs}) - 带位置和关键字参数的方法调用
    - 可迭代对象: [attr1, ('method', (args)), attr2] - 复杂的调用链
    
    Args:
        obj: 起始对象
        attr_chain: 属性链，支持多种格式
        getattr_func: 自定义属性获取函数，默认使用default_getattr_func
        call_last_attr: 是否调用链中最后一个属性（如果它是方法）
        
    Returns:
        属性链访问的最终结果
        
    Raises:
        AttributeError: 当属性链中的某个属性不存在时
        TypeError: 当属性链格式不正确时
        
    示例:
        class Data:
            def __init__(self, values):
                self.values = values
            def filter(self, threshold):
                return Data([v for v in self.values if v > threshold])
            def mean(self):
                return sum(self.values) / len(self.values)
                
        data = Data([1, 2, 3, 4, 5])
        
        # 字符串形式：连续调用属性和方法
        result = deep_getattr(data, 'values')  # [1, 2, 3, 4, 5]
        
        # 元组形式：带参数的方法调用
        filtered = deep_getattr(data, ('filter', (2,)))  # Data([3, 4, 5])
        
        # 复杂链式调用
        chain = [('filter', (2,)), 'mean']
        result = deep_getattr(data, chain)  # 4.0
    """
    # 参数类型验证：确保attr_chain是支持的类型
    checks.assert_instance_of(attr_chain, (str, tuple, Iterable))

    # 情况1：字符串格式的属性链
    if isinstance(attr_chain, str):
        # 如果包含点号，分割成属性链继续递归处理
        if '.' in attr_chain:
            return deep_getattr(
                obj,
                attr_chain.split('.'),  # 按点号分割成列表
                getattr_func=getattr_func,
                call_last_attr=call_last_attr
            )
        # 单个属性名，直接使用getattr_func获取
        return getattr_func(obj, attr_chain, call_attr=call_last_attr)
    
    # 情况2：元组格式的方法调用
    if isinstance(attr_chain, tuple):
        # 元组长度为1：('method',) - 无参方法调用
        if len(attr_chain) == 1 \
                and isinstance(attr_chain[0], str):
            return getattr_func(obj, attr_chain[0])
        
        # 元组长度为2：('method', (args,)) - 带位置参数的方法调用
        if len(attr_chain) == 2 \
                and isinstance(attr_chain[0], str) \
                and isinstance(attr_chain[1], tuple):
            return getattr_func(obj, attr_chain[0], args=attr_chain[1])
        
        # 元组长度为3：('method', (args,), {kwargs}) - 带位置和关键字参数的方法调用
        if len(attr_chain) == 3 \
                and isinstance(attr_chain[0], str) \
                and isinstance(attr_chain[1], tuple) \
                and isinstance(attr_chain[2], dict):
            return getattr_func(obj, attr_chain[0], args=attr_chain[1], kwargs=attr_chain[2])
    
    # 情况3：可迭代对象格式的复杂调用链
    result = obj  # 从初始对象开始
    for i, attr in enumerate(attr_chain):
        # 处理链中的每个元素
        if i < len(attr_chain) - 1:
            # 非最后一个元素：总是调用（call_last_attr=True）
            result = deep_getattr(
                result,
                attr,
                getattr_func=getattr_func,
                call_last_attr=True
            )
        else:
            # 最后一个元素：根据call_last_attr参数决定是否调用
            result = deep_getattr(
                result,
                attr,
                getattr_func=getattr_func,
                call_last_attr=call_last_attr
            )
    return result


# 定义AttrResolver的类型变量，用于支持泛型和类型推断
# 这确保了继承AttrResolver的子类在使用resolve_self等方法时能返回正确的类型
AttrResolverT = tp.TypeVar("AttrResolverT", bound="AttrResolver")


class AttrResolver:
    """智能属性解析器基类，提供高级的属性访问、方法调用和缓存功能
    
    这是vectorbt框架中的核心基类之一，为复杂对象提供统一的属性解析机制。
    它不仅支持标准的属性访问，还提供了智能参数匹配、缓存优化和条件性方法调用。
    
    核心特性：
    1. 智能参数解析：根据方法签名自动过滤和传递参数
    2. 内置缓存机制：避免重复计算，提高性能
    3. 条件性调用：支持基于参数的动态方法选择
    4. 参数预处理和后处理：可以在属性访问前后进行自定义处理
    5. 别名支持：允许为对象定义多个引用名称
    
    在vectorbt中的应用：
    - Portfolio类：智能解析财务指标和统计方法
    - GenericAccessor：动态数据处理和转换
    - ArrayWrapper：数组操作的统一接口
    - StatsBuilder：统计指标的灵活计算
    
    设计模式：
    这个类实现了一种高级的反射模式，允许对象在运行时动态决定如何响应属性访问。
    通过继承这个类，vectorbt的核心对象能够提供更加智能和用户友好的API。
    
    使用示例:
        class SmartCalculator(AttrResolver):
            def __init__(self, data):
                self.data = data
                self._cache = {}
                
            def calculate_mean(self, window=10):
                return self.data[-window:].mean()
                
            def calculate_std(self, window=10, ddof=1):
                return self.data[-window:].std(ddof=ddof)
        
        calc = SmartCalculator(data)
        # 智能参数传递：只传递方法签名中存在的参数
        result = calc.resolve_attr('calculate_mean', cond_kwargs={'window': 20, 'unused_param': 'ignored'})
    """

    @property
    def self_aliases(self) -> tp.Set[str]:
        """返回与当前对象关联的别名集合
        
        这些别名在参数解析过程中会被识别为当前对象的引用。
        在模板和配置系统中，用户可以使用这些别名来引用当前对象。
        
        默认别名：
        - 'self': 标准的自引用别名
        
        子类通常会重写此方法以添加特定的别名：
        - Portfolio类: {'self', 'portfolio', 'pf'}
        - 数据处理类: {'self', 'data', 'df'}
        
        Returns:
            包含对象别名的字符串集合
            
        示例:
            class MyPortfolio(AttrResolver):
                @property
                def self_aliases(self):
                    return {'self', 'portfolio', 'pf', 'my_portfolio'}
        """
        return {'self'}  # 默认只包含'self'别名

    def resolve_self(self: AttrResolverT,
                     cond_kwargs: tp.KwargsLike = None,
                     custom_arg_names: tp.Optional[tp.Set[str]] = None,
                     impacts_caching: bool = True,
                     silence_warnings: bool = False) -> AttrResolverT:
        """解析当前对象实例，支持基于条件参数的对象重构
        
        这个方法允许对象根据传入的条件参数动态调整自身状态或创建新实例。
        它是属性解析链的起点，确保后续的属性访问在正确的对象上下文中进行。
        
        在vectorbt中的应用：
        - Portfolio对象可能根据不同的计算参数创建专门的实例
        - ArrayWrapper可能根据分组参数调整内部结构
        - GenericAccessor可能根据映射参数创建新的访问器
        
        Args:
            cond_kwargs: 条件关键字参数，可能影响对象状态
                        注意：这个字典可能会被就地修改
            custom_arg_names: 自定义参数名称集合，这些参数会影响缓存策略
            impacts_caching: 是否影响缓存，True表示可能影响缓存有效性
            silence_warnings: 是否静默警告信息
            
        Returns:
            解析后的对象实例（可能是新创建的）
            
        Note:
            基类实现直接返回self，子类可以重写以实现更复杂的解析逻辑
            
        示例:
            class ConfigurableCalculator(AttrResolver):
                def __init__(self, data, config=None):
                    self.data = data
                    self.config = config or {}
                    
                def resolve_self(self, cond_kwargs=None, **kwargs):
                    if cond_kwargs and 'new_config' in cond_kwargs:
                        # 创建新配置的实例
                        return ConfigurableCalculator(self.data, cond_kwargs['new_config'])
                    return self
        """
        return self  # 基类实现：直接返回当前实例

    def pre_resolve_attr(self, attr: str, final_kwargs: tp.KwargsLike = None) -> str:
        """属性解析前的预处理钩子，允许动态修改要访问的属性名称
        
        这个方法在实际访问属性之前被调用，允许实现属性名称的动态映射、
        别名解析或基于条件的属性选择。这对于实现灵活的API非常有用。
        
        在vectorbt中的典型应用：
        - Portfolio类中，根据'use_asset_returns'参数将'returns'映射为'asset_returns'
        - 根据'trades_type'参数选择不同类型的交易数据
        - 实现属性的向前兼容性和别名支持
        
        Args:
            attr: 原始属性名称
            final_kwargs: 最终的关键字参数字典，包含所有条件参数
            
        Returns:
            处理后的属性名称
            
        示例:
            class FlexibleData(AttrResolver):
                def pre_resolve_attr(self, attr, final_kwargs=None):
                    # 实现属性别名
                    aliases = {
                        'avg': 'mean',
                        'std_dev': 'std',
                        'variance': 'var'
                    }
                    
                    # 根据条件选择不同的数据源
                    if final_kwargs and final_kwargs.get('use_raw_data'):
                        if attr == 'values':
                            return 'raw_values'
                    
                    return aliases.get(attr, attr)
        """
        return attr  # 基类实现：直接返回原属性名

    def post_resolve_attr(self, attr: str, out: tp.Any, final_kwargs: tp.KwargsLike = None) -> str:
        """属性解析后的后处理钩子，允许对解析结果进行修改
        
        这个方法在属性访问完成后被调用，允许对结果进行过滤、转换或包装。
        这对于实现统一的数据处理逻辑或添加额外的业务逻辑很有用。
        
        在vectorbt中的典型应用：
        - Portfolio类中，根据'incl_open'参数过滤开仓交易
        - 对数值结果进行单位转换或格式化
        - 添加元数据或包装器类
        
        Args:
            attr: 属性名称
            out: 属性解析的原始结果
            final_kwargs: 最终的关键字参数字典
            
        Returns:
            处理后的结果对象
            
        示例:
            class ProcessedData(AttrResolver):
                def post_resolve_attr(self, attr, out, final_kwargs=None):
                    # 对数值结果进行百分比转换
                    if final_kwargs and final_kwargs.get('as_percentage'):
                        if isinstance(out, (int, float)):
                            return out * 100
                    
                    # 过滤负值
                    if final_kwargs and final_kwargs.get('positive_only'):
                        if hasattr(out, '__gt__') and out < 0:
                            return 0
                    
                    return out
        """
        return out  # 基类实现：直接返回原结果

    def resolve_attr(self,
                     attr: str,
                     args: tp.ArgsLike = None,
                     cond_kwargs: tp.KwargsLike = None,
                     kwargs: tp.KwargsLike = None,
                     custom_arg_names: tp.Optional[tp.Container[str]] = None,
                     cache_dct: tp.KwargsLike = None,
                     use_caching: bool = True,
                     passed_kwargs_out: tp.KwargsLike = None) -> tp.Any:
        """核心属性解析方法，提供智能的属性访问、方法调用和缓存功能
        
        这是AttrResolver类的核心方法，实现了复杂的属性解析逻辑：
        1. 智能参数匹配：只传递方法签名中存在的参数
        2. 缓存管理：避免重复计算，提高性能
        3. getter方法支持：自动查找get_<attr>方法
        4. 预处理和后处理：支持属性名和结果的动态转换
        
        解析流程：
        1. 参数初始化和合并
        2. 调用pre_resolve_attr进行属性名预处理
        3. 检查是否存在get_<attr>方法
        4. 区分属性和方法进行不同处理
        5. 智能参数过滤和传递
        6. 缓存检查和存储
        7. 调用post_resolve_attr进行结果后处理
        
        Args:
            attr: 要解析的属性名称
            args: 传递给方法的位置参数
            cond_kwargs: 条件关键字参数，会根据方法签名自动过滤
            kwargs: 显式关键字参数，总是会被传递
            custom_arg_names: 自定义参数名称，影响缓存策略
            cache_dct: 缓存字典，存储已计算的结果
            use_caching: 是否使用缓存
            passed_kwargs_out: 输出参数字典，记录实际传递的参数
            
        Returns:
            属性值或方法调用结果
            
        示例:
            class SmartAnalyzer(AttrResolver):
                def __init__(self, data):
                    self.data = data
                    self._cache = {}
                    
                def get_moving_average(self, window=20, method='simple'):
                    # 复杂的移动平均计算
                    if method == 'simple':
                        return self.data.rolling(window).mean()
                    elif method == 'exponential':
                        return self.data.ewm(span=window).mean()
                        
                def volatility(self, window=20, annualized=True):
                    vol = self.data.rolling(window).std()
                    return vol * np.sqrt(252) if annualized else vol
            
            analyzer = SmartAnalyzer(price_data)
            
            # 智能参数匹配：只传递方法中存在的参数
            ma = analyzer.resolve_attr(
                'moving_average',  # 自动找到get_moving_average方法
                cond_kwargs={
                    'window': 50,
                    'method': 'exponential',
                    'unused_param': 'ignored'  # 这个参数会被忽略
                },
                cache_dct=analyzer._cache,  # 启用缓存
                use_caching=True
            )
        """
        # === 1. 参数默认值初始化 ===
        # 确保所有参数都有默认值，避免None值导致的错误
        if custom_arg_names is None:
            custom_arg_names = list()  # 空列表作为默认自定义参数名
        if cache_dct is None:
            cache_dct = {}  # 空字典作为默认缓存
        if args is None:
            args = ()  # 空元组作为默认位置参数
        if kwargs is None:
            kwargs = {}  # 空字典作为默认关键字参数
        if passed_kwargs_out is None:
            passed_kwargs_out = {}  # 空字典用于记录传递的参数
        
        # 合并条件参数和显式参数，显式参数优先级更高
        final_kwargs = merge_dicts(cond_kwargs, kwargs)

        # === 2. 属性名预处理 ===
        # 获取当前对象的类型，用于后续的属性检查
        cls = type(self)
        
        # 调用预处理钩子，允许动态修改属性名
        _attr = self.pre_resolve_attr(attr, final_kwargs=final_kwargs)
        
        # === 3. getter方法检测 ===
        # 检查是否存在get_<attr>方法，这是vectorbt的约定
        # 例如：访问'returns'属性时，优先查找'get_returns'方法
        if 'get_' + attr in dir(cls):
            _attr = 'get_' + attr

        # === 4. 区分方法和属性进行处理 ===
        # 检查目标属性是否为方法或函数
        if inspect.ismethod(getattr(cls, _attr)) or inspect.isfunction(getattr(cls, _attr)):
            # --- 4a. 方法调用处理 ---
            
            # 获取实例方法对象
            attr_func = getattr(self, _attr)
            attr_func_kwargs = dict()  # 准备传递给方法的参数字典
            
            # 获取方法的参数名列表，用于智能参数过滤
            attr_func_arg_names = get_func_arg_names(attr_func)
            
            custom_k = False  # 标记是否包含自定义参数
            
            # 遍历所有可用参数，进行智能过滤
            for k, v in final_kwargs.items():
                # 只传递方法签名中存在的参数，或者在kwargs中明确指定的参数
                if k in attr_func_arg_names or k in kwargs:
                    # 检查是否为自定义参数（影响缓存策略）
                    if k in custom_arg_names:
                        custom_k = True
                    attr_func_kwargs[k] = v  # 添加到方法参数中
                    passed_kwargs_out[k] = v  # 记录实际传递的参数

            # 缓存检查：如果启用缓存且无自定义参数且缓存中存在结果
            if use_caching and not custom_k and attr in cache_dct:
                out = cache_dct[attr]  # 直接从缓存获取结果
            else:
                # 调用方法并传递筛选后的参数
                out = attr_func(*args, **attr_func_kwargs)
                
                # 存储到缓存（如果满足缓存条件）
                if use_caching and not custom_k:
                    cache_dct[attr] = out
        else:
            # --- 4b. 属性访问处理 ---
            
            # 缓存检查：对于属性也支持缓存
            if use_caching and attr in cache_dct:
                out = cache_dct[attr]  # 从缓存获取
            else:
                # 直接获取属性值
                out = getattr(self, _attr)
                
                # 存储到缓存
                if use_caching:
                    cache_dct[attr] = out

        # === 5. 结果后处理 ===
        # 调用后处理钩子，允许对结果进行修改
        out = self.post_resolve_attr(attr, out, final_kwargs=final_kwargs)
        
        return out  # 返回最终结果

    def deep_getattr(self, *args, **kwargs) -> tp.Any:
        """提供deep_getattr功能的便捷方法
        
        这是一个简单的包装方法，允许AttrResolver的实例直接调用deep_getattr功能，
        而不需要导入和使用模块级的deep_getattr函数。
        
        Args:
            *args: 传递给deep_getattr的位置参数
            **kwargs: 传递给deep_getattr的关键字参数
            
        Returns:
            deep_getattr的执行结果
            
        示例:
            class MyObject(AttrResolver):
                def __init__(self):
                    self.data = SomeDataClass()
                    
            obj = MyObject()
            # 等价于 deep_getattr(obj, 'data.process.normalize')
            result = obj.deep_getattr('data.process.normalize')
        """
        # 直接调用模块级的deep_getattr函数，self作为第一个参数
        return deep_getattr(self, *args, **kwargs)
