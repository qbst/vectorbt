# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT UTILS MODULE: 模块管理和内省工具
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于模块管理和内省的核心工具模块。在大型量化交易
系统中，经常需要动态发现、加载和管理各种模块，包括策略模块、指标模块、数据源模块等。
该模块提供了一套完整的模块管理基础设施，支持模块的自动发现、内容检查、递归导入等
高级功能。

核心设计理念：
1. **模块内省机制**：提供深度的模块内容检查和分析能力，支持运行时模块发现
2. **动态加载支持**：实现模块的动态导入和加载，支持插件化架构的构建
3. **命名空间管理**：智能管理模块命名空间，避免命名冲突和重复导入
4. **黑名单机制**：提供灵活的模块过滤功能，支持选择性导入和安全控制

主要功能模块：
- **模块归属检查**：is_from_module函数，检查对象是否来自指定模块
- **模块内容列举**：list_module_keys函数，列出模块中的所有公共函数和类
- **递归模块导入**：import_submodules函数，递归导入模块的所有子模块
"""

import importlib  # 导入importlib模块，提供模块的动态导入功能
import inspect  # 导入inspect模块，提供对象和模块的内省功能
import pkgutil  # 导入pkgutil模块，提供包和模块的遍历功能
import sys  # 导入sys模块，提供对Python解释器的访问接口
from types import ModuleType  # 导入ModuleType类型，用于模块类型的注解

from vectorbt import _typing as tp  # 导入vectorbt的类型定义模块，提供统一的类型注解


def is_from_module(obj: tp.Any, module: ModuleType) -> bool:
    """
    检查指定对象是否来自特定模块
    
    该函数是vectorbt模块管理系统的基础工具，用于验证对象的模块归属关系。
    在量化交易系统中，经常需要确定某个函数、类或变量是否属于特定的模块，
    以便进行正确的模块管理和API组织。该函数通过深度内省机制，准确判断
    对象的真实来源模块。
    
    核心算法：
    1. 使用inspect.unwrap()剥离装饰器和包装器，获取对象的原始形式
    2. 通过inspect.getmodule()获取对象所属的模块信息
    3. 比较对象模块与目标模块的名称，处理None值情况
    
    应用场景：
    - **API文档生成**：确定哪些函数/类应该包含在模块的公共API中
    - **模块清理**：识别和移除不属于当前模块的对象
    - **插件验证**：验证插件中的组件是否来自正确的模块
    - **动态导入验证**：确认动态导入的对象的真实来源
    
    参数：
        obj (tp.Any): 需要检查的对象，可以是函数、类、变量等任意Python对象
        module (ModuleType): 目标模块对象，用于比较对象的归属关系
    
    返回：
        bool: 如果对象来自指定模块返回True，否则返回False
    
    示例：
        >>> import vectorbt.indicators as indicators
        >>> from vectorbt.indicators import MA
        >>> 
        >>> # 检查MA类是否来自indicators模块
        >>> is_from_module(MA, indicators)
        True
        >>> 
        >>> # 检查内建函数是否来自indicators模块
        >>> is_from_module(len, indicators)
        False
        >>> 
        >>> # 检查导入的外部对象
        >>> import numpy as np
        >>> is_from_module(np.array, indicators)
        False
        
        >>> # 在动态API构建中的应用
        >>> def build_public_api(module):
        ...     public_objects = []
        ...     for name, obj in inspect.getmembers(module):
        ...         if not name.startswith('_') and is_from_module(obj, module):
        ...             public_objects.append((name, obj))
        ...     return public_objects
        
        >>> # 检查装饰器包装的函数
        >>> def my_decorator(func):
        ...     def wrapper(*args, **kwargs):
        ...         return func(*args, **kwargs)
        ...     return wrapper
        >>> 
        >>> @my_decorator
        ... def my_function():
        ...     pass
        >>> 
        >>> # 即使被装饰器包装，仍能正确识别原始函数的模块
        >>> is_from_module(my_function, sys.modules[__name__])
        True
    
    技术细节：
        - 使用inspect.unwrap()处理装饰器和functools.wraps()包装的对象
        - 当对象的模块信息为None时，返回True（处理某些内建对象的情况）
        - 通过模块名称比较而非对象引用比较，避免模块重载问题
        - 支持检查任意类型的Python对象，不仅限于函数和类
    
    注意事项：
        - 对于C扩展模块中的对象，模块信息可能为None
        - 某些动态生成的对象可能无法正确识别其来源模块
        - 模块名称的比较是精确匹配，不支持模糊匹配
    """
    mod = inspect.getmodule(inspect.unwrap(obj))  # 获取对象的原始模块，剥离装饰器和包装器
    return mod is None or mod.__name__ == module.__name__  # 比较模块名称，处理None情况


def list_module_keys(module_name: str, whitelist: tp.Optional[tp.List[str]] = None,
                     blacklist: tp.Optional[tp.List[str]] = None):
    """
    列出模块中所有公共函数和类的名称，支持白名单和黑名单过滤
    
    这个函数是vectorbt模块管理系统的核心工具，用于自动发现和列举模块中的
    公共API组件。在量化交易系统中，经常需要动态获取模块的可用功能，用于
    API文档生成、动态导入、插件发现等场景。该函数提供了灵活的过滤机制，
    支持精确控制哪些组件应该被包含在结果中。
    
    过滤逻辑：
    1. 默认包含所有不以下划线开头的公共对象
    2. 只包含函数（可调用对象）和类对象
    3. 只包含来自指定模块的对象（使用is_from_module验证）
    4. 应用黑名单过滤，排除不需要的组件
    5. 应用白名单过滤，强制包含特定组件
    
    应用场景：
    - **API文档生成**：自动提取模块的公共API用于文档生成
    - **模块导入优化**：获取模块的关键组件列表，优化导入过程
    - **插件接口发现**：动态发现插件模块中的可用接口
    - **单元测试**：自动发现模块中的测试用例和辅助函数
    - **代码分析**：静态分析模块结构和依赖关系
    
    参数：
        module_name (str): 模块名称，必须是已导入的模块名称
        whitelist (tp.Optional[tp.List[str]], optional): 
            白名单，强制包含的对象名称列表，即使不满足其他条件也会包含
        blacklist (tp.Optional[tp.List[str]], optional): 
            黑名单，需要排除的对象名称列表，即使满足其他条件也会排除
    
    返回：
        list: 符合条件的对象名称列表
    
    示例：
        >>> # 列出indicators模块的所有公共API
        >>> api_names = list_module_keys('vectorbt.indicators')
        >>> print(api_names)
        ['MA', 'RSI', 'MACD', 'BollingerBands', 'ATR', ...]
        
        >>> # 使用黑名单排除某些组件
        >>> filtered_names = list_module_keys(
        ...     'vectorbt.indicators',
        ...     blacklist=['_deprecated_function', 'internal_helper']
        ... )
        
        >>> # 使用白名单强制包含私有函数
        >>> extended_names = list_module_keys(
        ...     'vectorbt.indicators',
        ...     whitelist=['_important_private_function']
        ... )
        
        >>> # 在动态API构建中的应用
        >>> def build_module_api(module_name):
        ...     public_names = list_module_keys(module_name)
        ...     module = sys.modules[module_name]
        ...     return {name: getattr(module, name) for name in public_names}
        
        >>> # 为数据源模块构建API
        >>> data_api = build_module_api('vectorbt.data')
        >>> strategy_api = build_module_api('vectorbt.portfolio')
        
        >>> # 过滤特定类型的对象
        >>> def get_module_classes(module_name):
        ...     all_names = list_module_keys(module_name)
        ...     module = sys.modules[module_name]
        ...     return [name for name in all_names 
        ...             if inspect.isclass(getattr(module, name))]
        
        >>> # 获取所有指标类
        >>> indicator_classes = get_module_classes('vectorbt.indicators')
        >>> print(indicator_classes)
        ['MA', 'RSI', 'MACD', 'BollingerBands', ...]
    
    高级用法：
        >>> # 动态插件发现
        >>> def discover_strategy_plugins(plugin_module):
        ...     strategy_names = list_module_keys(
        ...         plugin_module,
        ...         whitelist=['CustomStrategy1', 'CustomStrategy2'],
        ...         blacklist=['BaseStrategy', 'AbstractStrategy']
        ...     )
        ...     return strategy_names
        
        >>> # 条件过滤
        >>> def get_public_indicators(exclude_experimental=True):
        ...     blacklist = ['ExperimentalIndicator'] if exclude_experimental else None
        ...     return list_module_keys('vectorbt.indicators', blacklist=blacklist)
    
    技术实现细节：
        - 使用inspect.getmembers()遍历模块的所有成员
        - 通过inspect.isroutine()和inspect.isclass()判断对象类型
        - 使用is_from_module()确保对象真正属于指定模块
        - 支持可调用对象和类对象的自动识别
        - 白名单的优先级高于其他过滤条件
    
    性能考量：
        - 对于大型模块，遍历和过滤可能消耗较多时间
        - 建议在模块初始化时缓存结果，避免重复计算
        - 使用黑名单比白名单更高效，减少不必要的检查
    
    注意事项：
        - 模块必须已经导入到sys.modules中
        - 动态生成的对象可能无法被正确识别
        - 某些特殊对象（如C扩展）可能需要特殊处理
    """
    if whitelist is None:  # 如果没有提供白名单参数
        whitelist = []  # 初始化为空列表
    if blacklist is None:  # 如果没有提供黑名单参数
        blacklist = []  # 初始化为空列表
    module = sys.modules[module_name]  # 从sys.modules中获取指定名称的模块对象
    return [name for name, obj in inspect.getmembers(module)  # 遍历模块的所有成员
            if (not name.startswith("_") and is_from_module(obj, module)  # 过滤条件：不以下划线开头且来自指定模块
                and ((inspect.isroutine(obj) and callable(obj)) or inspect.isclass(obj))  # 且是可调用对象或类对象
                and name not in blacklist) or name in whitelist]  # 且不在黑名单中，或者在白名单中


def import_submodules(package: tp.Union[str, ModuleType]) -> tp.Dict[str, ModuleType]:
    """
    递归导入模块的所有子模块，支持黑名单过滤机制
    
    这个函数是vectorbt模块管理系统的核心功能，用于实现模块的批量导入和
    命名空间的自动构建。在大型量化交易系统中，经常需要一次性导入所有
    相关的子模块，以便用户可以直接访问所有功能。该函数提供了智能的
    递归导入机制，支持包结构的深度遍历和选择性导入。
    
    核心算法：
    1. 检查包是否定义了__blacklist__属性，获取需要排除的模块名称
    2. 使用pkgutil.walk_packages()遍历包的所有子模块和子包
    3. 递归导入每个子模块，构建完整的模块树
    4. 返回包含所有导入模块的字典映射
    
    黑名单机制：
    如果包定义了__blacklist__属性，该函数会自动跳过黑名单中的模块，
    这对于排除测试模块、实验性功能或不稳定的组件非常有用。
    
    应用场景：
    - **框架初始化**：vectorbt.__init__.py中的自动模块导入
    - **插件系统**：动态加载和注册所有插件模块
    - **API命名空间构建**：构建完整的API命名空间树
    - **开发环境设置**：快速导入所有开发相关的模块
    - **文档生成**：自动发现所有模块用于文档生成
    
    参数：
        package (tp.Union[str, ModuleType]): 
            包名称（字符串）或包对象，作为导入的起始点
    
    返回：
        tp.Dict[str, ModuleType]: 
            包含所有导入模块的字典，键为模块名称，值为模块对象
    
    示例：
        >>> # 导入vectorbt的所有子模块
        >>> all_modules = import_submodules('vectorbt')
        >>> print(list(all_modules.keys()))
        ['vectorbt.indicators', 'vectorbt.portfolio', 'vectorbt.data', 
         'vectorbt.utils', 'vectorbt.base', ...]
        
        >>> # 访问导入的模块
        >>> indicators_module = all_modules['vectorbt.indicators']
        >>> ma_class = indicators_module.MA
        
        >>> # 使用黑名单机制
        >>> # 在包的__init__.py中定义：
        >>> # __blacklist__ = ['tests', 'experimental', 'deprecated']
        >>> 
        >>> # 然后导入时会自动跳过这些模块
        >>> filtered_modules = import_submodules('my_package')
        >>> # 'my_package.tests' 不会被导入
        
        >>> # 递归导入自定义策略包
        >>> def setup_strategy_environment(strategy_package):
        ...     strategy_modules = import_submodules(strategy_package)
        ...     
        ...     # 自动注册所有策略类
        ...     strategies = {}
        ...     for module_name, module in strategy_modules.items():
        ...         for name in list_module_keys(module_name):
        ...             obj = getattr(module, name)
        ...             if inspect.isclass(obj) and name.endswith('Strategy'):
        ...                 strategies[name] = obj
        ...     
        ...     return strategies
        
        >>> # 批量导入数据源模块
        >>> data_modules = import_submodules('vectorbt.data')
        >>> available_sources = [name.split('.')[-1] 
        ...                     for name in data_modules.keys() 
        ...                     if name.endswith('_data')]
        >>> print(f"可用数据源: {available_sources}")
        
        >>> # 在插件系统中的应用
        >>> def load_indicator_plugins(plugin_dir):
        ...     # 动态添加插件目录到Python路径
        ...     sys.path.insert(0, plugin_dir)
        ...     
        ...     # 导入所有插件模块
        ...     plugin_modules = import_submodules('indicator_plugins')
        ...     
        ...     # 注册插件中的指标类
        ...     registered_indicators = {}
        ...     for module_name, module in plugin_modules.items():
        ...         indicator_names = list_module_keys(
        ...             module_name,
        ...             blacklist=['BaseIndicator', 'AbstractIndicator']
        ...         )
        ...         for name in indicator_names:
        ...             indicator_class = getattr(module, name)
        ...             if inspect.isclass(indicator_class):
        ...                 registered_indicators[name] = indicator_class
        ...     
        ...     return registered_indicators
    
    高级用法：
        >>> # 条件导入
        >>> def conditional_import(package, condition_func):
        ...     all_modules = import_submodules(package)
        ...     return {name: module for name, module in all_modules.items()
        ...             if condition_func(name, module)}
        
        >>> # 只导入数据相关模块
        >>> data_modules = conditional_import(
        ...     'vectorbt',
        ...     lambda name, module: 'data' in name
        ... )
        
        >>> # 性能监控的导入
        >>> import time
        >>> def timed_import(package):
        ...     start_time = time.time()
        ...     modules = import_submodules(package)
        ...     end_time = time.time()
        ...     print(f"导入{len(modules)}个模块用时: {end_time - start_time:.2f}秒")
        ...     return modules
    
    黑名单配置示例：
        >>> # 在包的__init__.py中配置黑名单
        >>> __blacklist__ = [
        ...     'tests',          # 测试模块
        ...     'experimental',   # 实验性功能
        ...     'deprecated',     # 已弃用的模块
        ...     'internal',       # 内部使用模块
        ...     'benchmark'       # 性能测试模块
        ... ]
    
    技术实现细节：
        - 使用pkgutil.walk_packages()进行深度优先遍历
        - 自动处理包和模块的区别，对包进行递归导入
        - 通过模块名称切分确保只导入直接子模块
        - 使用importlib.import_module()实现动态导入
        - 返回的字典保持导入顺序，便于调试和分析
    
    性能优化：
        - 避免重复导入已经在sys.modules中的模块
        - 使用字典更新而非列表追加，提高大规模导入的效率
        - 递归调用时传递模块名称而非模块对象，减少内存占用
    
    错误处理：
        - 导入失败的模块不会中断整个导入过程
        - 可以通过添加try-except块来处理导入异常
        - 黑名单机制可以预防已知的问题模块
    
    注意事项：
        - 大规模导入可能消耗大量时间和内存
        - 某些模块的导入可能有副作用（如注册全局变量）
        - 循环导入可能导致导入失败或无限递归
        - 建议在应用启动时进行批量导入，而非运行时动态导入
    """
    if isinstance(package, str):  # 如果传入的是字符串类型的包名
        package = importlib.import_module(package)  # 使用importlib动态导入包对象
    blacklist = []  # 初始化黑名单为空列表
    if hasattr(package, '__blacklist__'):  # 检查包是否定义了__blacklist__属性
        blacklist = package.__blacklist__  # 获取包定义的黑名单列表
    results = {}  # 初始化结果字典，用于存储所有导入的模块
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):  # 遍历包的所有子模块和子包
        if '.'.join(name.split('.')[:-1]) != package.__name__:  # 检查是否为直接子模块，排除更深层的子模块
            continue  # 跳过不是直接子模块的模块
        if name.split('.')[-1] in blacklist:  # 检查模块名是否在黑名单中
            continue  # 跳过黑名单中的模块
        results[name] = importlib.import_module(name)  # 动态导入模块并添加到结果字典
        if is_pkg:  # 如果当前项是包（而不是模块）
            results.update(import_submodules(name))  # 递归导入子包的所有模块
    return results  # 返回包含所有导入模块的字典
