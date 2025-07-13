# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT GENERIC MODULE: CLASS AND FUNCTION DECORATORS
================================================================================

文件作用概述：
本文件是vectorbt量化交易框架中的核心装饰器模块，为类的动态方法生成提供了强大的基础设施。
该模块专门用于解决量化分析中常见的重复代码问题，通过声明式配置自动为类批量添加方法，
极大地提高了开发效率并确保了API的一致性。

核心设计逻辑：
1. **声明式方法生成**：通过配置对象（Config）定义方法规范，装饰器根据配置自动生成方法
2. **类型安全的装饰器**：使用泛型类型约束，确保装饰器只能应用于兼容的类
3. **统一的包装策略**：所有生成的方法都遵循相同的包装和签名替换规则
4. **高性能计算集成**：专门为Numba编译函数和sklearn变换器提供优化的方法生成

设计模式：
- **装饰器模式**：动态为类添加功能，而不修改类的原始定义
- **策略模式**：支持不同类型的方法生成策略（Numba方法、变换方法等）
- **模板方法模式**：定义了方法生成的标准流程，具体实现由配置决定
- **配置驱动模式**：使用配置对象控制方法生成的行为

主要功能模块：
- **Numba方法装饰器**：attach_nb_methods，用于集成高性能Numba编译函数
- **变换方法装饰器**：attach_transform_methods，用于集成sklearn数据预处理器
- **签名管理**：自动处理方法签名的替换和文档生成
- **包装器集成**：与vectorbt的ArrayWrapper系统无缝集成

应用场景：
- **技术指标计算**：为指标类批量添加基于Numba的计算方法
- **数据预处理**：为访问器类批量添加sklearn变换器方法
- **统计分析**：为分析类批量添加统计计算方法
- **性能优化**：避免手动编写大量相似的方法定义

技术优势：
- **零重复代码**：通过配置驱动的方法生成消除代码重复
- **类型安全**：编译时检查装饰器的适用性
- **性能优化**：专门为高性能计算场景优化的方法生成
- **API一致性**：所有生成的方法遵循统一的调用规范

与vectorbt生态系统的关系：
- **BaseAccessor集成**：为pandas访问器提供方法扩展能力
- **ArrayWrapper协作**：与数组包装器协同工作，确保元数据的正确处理
- **Numba优化**：为vectorbt的高性能计算核心提供方法接口
- **sklearn集成**：为量化分析提供标准化的数据预处理能力

该模块是vectorbt框架"高性能+易用性"设计理念的重要体现，通过装饰器的抽象
隐藏了底层的复杂性，同时保持了最佳的计算性能。
"""

# 导入Python标准库中的inspect模块，用于运行时检查对象的内部结构
import inspect

# 导入vectorbt的类型定义模块，提供完整的类型注解支持
from vectorbt import _typing as tp
# 导入vectorbt的检查工具模块，提供类型和条件验证功能
from vectorbt.utils import checks
# 导入vectorbt的配置管理模块，提供配置对象和参数处理功能
from vectorbt.utils.config import merge_dicts, Config, get_func_arg_names

# 定义包装器函数类型，用于类型注解
# 这是一个泛型函数类型，接受一个类型T并返回相同类型T的类
WrapperFuncT = tp.Callable[[tp.Type[tp.T]], tp.Type[tp.T]]


def attach_nb_methods(config: Config) -> WrapperFuncT:
    """
    类装饰器：为类批量添加基于Numba的高性能计算方法
    
    这是vectorbt框架中最重要的装饰器之一，专门用于将Numba编译的高性能函数
    集成到类中作为方法。该装饰器通过配置驱动的方式，自动生成包装方法，
    将底层的NumPy数组操作与上层的pandas对象无缝连接。
    
    设计原理：
    1. **性能优化**：Numba编译的函数运行速度接近C语言，比纯Python快10-100倍
    2. **数据桥接**：自动处理pandas对象到NumPy数组的转换
    3. **元数据保持**：通过ArrayWrapper保持索引、列名等元数据信息
    4. **统一接口**：所有生成的方法都遵循相同的调用规范
    
    配置参数说明：
    config应包含目标方法名称（键）和配置字典（值），配置字典支持以下键：
    
    * `func`: 要包装的Numba函数，第一个参数必须接受2维数组
    * `is_reducing`: 是否为降维函数，默认为False
        - True: 函数返回标量或1维数组（如求和、均值等）
        - False: 函数返回与输入相同形状的数组（如标准化、差分等）
    * `path`: 用于文档的函数路径，默认为func.__name__
    * `replace_signature`: 是否替换目标签名，默认为True
    * `wrap_kwargs`: 默认的包装参数，会与用户提供的参数合并
        对于降维函数，默认为dict(name_or_index=target_name)
    
    被装饰的类必须是vectorbt.base.array_wrapper.Wrapping的子类。
    
    工作流程：
    1. 装饰器检查被装饰类是否为Wrapping的子类
    2. 遍历配置中的每个方法定义
    3. 为每个方法创建包装函数，处理参数转换和结果包装
    4. 替换方法签名以匹配原始Numba函数的签名
    5. 设置方法的文档字符串和元数据
    6. 将生成的方法添加到类中
    
    Args:
        config: 配置对象，定义要添加的方法及其属性
        
    Returns:
        装饰器函数，接受类并返回增强后的类
        
    Raises:
        AssertionError: 当被装饰的类不是Wrapping的子类时抛出
    """

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        # 延迟导入，避免循环导入问题
        from vectorbt.base.array_wrapper import Wrapping

        # 检查类的继承关系，确保被装饰的类是Wrapping的子类
        # 这是必要的，因为生成的方法依赖于Wrapping提供的wrapper属性
        checks.assert_subclass_of(cls, Wrapping)

        # 遍历配置中的每个方法定义
        for target_name, settings in config.items():
            # 提取Numba函数引用
            func = settings['func']
            # 判断是否为降维函数，默认为False
            is_reducing = settings.get('is_reducing', False)
            # 获取文档路径，默认使用函数名
            path = settings.get('path', func.__name__)
            # 是否替换方法签名，默认为True
            replace_signature = settings.get('replace_signature', True)
            # 获取默认的包装参数
            # 对于降维函数，默认使用方法名作为结果的名称或索引
            default_wrap_kwargs = settings.get('wrap_kwargs', dict(name_or_index=target_name) if is_reducing else None)

            # 定义新的方法函数
            # 使用默认参数捕获循环变量，避免闭包问题
            def new_method(self,
                           *args,  # 传递给Numba函数的位置参数
                           _target_name: str = target_name,  # 目标方法名称
                           _func: tp.Callable = func,  # Numba函数引用
                           _is_reducing: bool = is_reducing,  # 是否为降维函数
                           _default_wrap_kwargs: tp.KwargsLike = default_wrap_kwargs,  # 默认包装参数
                           wrap_kwargs: tp.KwargsLike = None,  # 用户提供的包装参数
                           **kwargs) -> tp.SeriesFrame:  # 传递给Numba函数的关键字参数
                # 准备参数：将自身的2维数组作为第一个参数
                args = (self.to_2d_array(),) + args
                # 验证参数绑定，确保参数与函数签名匹配
                inspect.signature(_func).bind(*args, **kwargs)

                # 调用Numba函数进行计算
                a = _func(*args, **kwargs)
                # 合并包装参数
                wrap_kwargs = merge_dicts(_default_wrap_kwargs, wrap_kwargs)
                # 根据函数类型选择包装方式
                if _is_reducing:
                    # 降维函数：使用wrap_reduced方法，通常返回Series或标量
                    return self.wrapper.wrap_reduced(a, **wrap_kwargs)
                # 非降维函数：使用wrap方法，保持原始形状
                return self.wrapper.wrap(a, **wrap_kwargs)

            # 替换方法签名以匹配原始Numba函数
            if replace_signature:
                # 获取原始Numba函数的签名
                source_sig = inspect.signature(func)
                # 获取新方法的参数
                new_method_params = tuple(inspect.signature(new_method).parameters.values())
                # 提取self参数和wrap_kwargs参数
                self_arg = new_method_params[0]
                wrap_kwargs_arg = new_method_params[-2]
                # 构造新的签名：self + 原始函数参数[1:] + wrap_kwargs
                source_sig = source_sig.replace(
                    parameters=(self_arg,) + tuple(source_sig.parameters.values())[1:] + (wrap_kwargs_arg,))
                # 应用新的签名
                new_method.__signature__ = source_sig

            # 设置方法的文档字符串，指向原始函数的文档
            new_method.__doc__ = f"See `{path}`."
            # 设置方法的完全限定名称，用于调试和文档生成
            new_method.__qualname__ = f"{cls.__name__}.{target_name}"
            # 设置方法名称
            new_method.__name__ = target_name
            # 将新方法添加到类中
            setattr(cls, target_name, new_method)
        # 返回增强后的类
        return cls

    # 返回装饰器函数
    return wrapper


def attach_transform_methods(config: Config) -> WrapperFuncT:
    """
    类装饰器：为类批量添加数据变换方法
    
    这是vectorbt框架中专门用于集成sklearn数据预处理器的装饰器。通过该装饰器，
    可以将sklearn的各种变换器（如StandardScaler、MinMaxScaler等）无缝集成到
    vectorbt的数据访问器中，为量化分析提供标准化的数据预处理能力。
    
    设计原理：
    1. **sklearn集成**：无缝集成sklearn的数据预处理生态系统
    2. **自动拟合**：智能检测变换器是否已拟合，自动选择fit_transform或transform
    3. **参数分离**：自动分离变换器初始化参数和变换参数
    4. **签名继承**：保持sklearn变换器的原始方法签名
    
    配置参数说明：
    config应包含目标方法名称（键）和配置字典（值），配置字典支持以下键：
    
    * `transformer`: 变换器类或对象
        - 类：将使用kwargs中的参数初始化
        - 对象：直接使用该实例进行变换
    * `docstring`: 方法文档字符串，默认为"See `{transformer.__name__}`."
    * `replace_signature`: 是否替换目标签名，默认为True
    
    被装饰的类必须是vectorbt.generic.accessors.GenericAccessor的子类。
    
    使用示例：
        ```python
        # 1. 定义变换器配置
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        transform_config = Config({
            'standardize': {
                'transformer': StandardScaler,
                'docstring': 'Standardize features by removing mean and scaling to unit variance.'
            },
            'minmax_scale': {
                'transformer': MinMaxScaler,
                'docstring': 'Scale features to a given range, typically [0, 1].'
            },
            'robust_scale': {
                'transformer': RobustScaler,
                'docstring': 'Scale features using robust statistics.'
            }
        })
        
        # 2. 应用装饰器
        @attach_transform_methods(transform_config)
        class DataPreprocessor(vbt.generic.accessors.GenericAccessor):
            pass
        
        # 3. 使用生成的方法
        import pandas as pd
        import numpy as np
        
        # 创建示例数据
        data = pd.DataFrame({
            'price': [100, 102, 98, 105, 103, 107, 101],
            'volume': [1000, 1200, 800, 1500, 1100, 1800, 900],
            'volatility': [0.1, 0.15, 0.08, 0.2, 0.12, 0.25, 0.09]
        })
        
        # 标准化数据
        standardized = data.vbt.standardize()
        print("标准化后的数据:")
        print(standardized.head())
        
        # 最小-最大缩放，指定范围
        scaled = data.vbt.minmax_scale(feature_range=(-1, 1))
        print("\n缩放到[-1, 1]范围:")
        print(scaled.head())
        
        # 鲁棒缩放，使用不同的分位数
        robust = data.vbt.robust_scale(quantile_range=(10, 90))
        print("\n鲁棒缩放（10%-90%分位数）:")
        print(robust.head())
        
        # 4. 链式变换
        pipeline_result = (data.vbt
                          .standardize()
                          .minmax_scale(feature_range=(0, 1)))
        print("\n链式变换结果:")
        print(pipeline_result.head())
        ```
    
    高级用法示例：
        ```python
        # 预拟合的变换器
        fitted_scaler = StandardScaler().fit(training_data)
        
        # 应用预拟合的变换器到新数据
        @attach_transform_methods(Config({
            'apply_fitted_scaler': {
                'transformer': fitted_scaler,
                'docstring': 'Apply pre-fitted scaler to new data.'
            }
        }))
        class ProductionPreprocessor(vbt.generic.accessors.GenericAccessor):
            pass
        
        # 在生产环境中使用
        new_data = pd.DataFrame({...})
        processed = new_data.vbt.apply_fitted_scaler()
        ```
    
    工作流程：
    1. 装饰器检查被装饰类是否为GenericAccessor的子类
    2. 遍历配置中的每个变换器定义
    3. 为每个变换器创建包装方法
    4. 智能处理变换器的初始化和参数分离
    5. 替换方法签名以匹配变换器的签名
    6. 设置方法的文档字符串和元数据
    7. 将生成的方法添加到类中
    
    Args:
        config: 配置对象，定义要添加的变换器方法及其属性
        
    Returns:
        装饰器函数，接受类并返回增强后的类
        
    Raises:
        AssertionError: 当被装饰的类不是GenericAccessor的子类时抛出
    """

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        # 延迟导入，避免循环导入问题
        from vectorbt.generic.accessors import TransformerT

        # 检查类的继承关系，确保被装饰的类是GenericAccessor的子类
        # 这是必要的，因为生成的方法依赖于GenericAccessor提供的transform方法
        checks.assert_subclass_of(cls, "GenericAccessor")

        # 遍历配置中的每个变换器定义
        for target_name, settings in config.items():
            # 提取变换器类或对象
            transformer = settings['transformer']
            # 获取文档字符串，默认为变换器名称
            docstring = settings.get('docstring', f"See `{transformer.__name__}`.")
            # 是否替换方法签名，默认为True
            replace_signature = settings.get('replace_signature', True)

            # 定义新的方法函数
            # 使用默认参数捕获循环变量，避免闭包问题
            def new_method(self,
                           _target_name: str = target_name,  # 目标方法名称
                           _transformer: tp.Union[tp.Type[TransformerT], TransformerT] = transformer,  # 变换器引用
                           **kwargs) -> tp.SeriesFrame:  # 传递给变换器的关键字参数
                # 检查变换器是否为类（需要实例化）
                if inspect.isclass(_transformer):
                    # 获取变换器构造函数的参数名称
                    arg_names = get_func_arg_names(_transformer.__init__)
                    # 分离变换器初始化参数
                    transformer_kwargs = dict()
                    for arg_name in arg_names:
                        # 如果kwargs中有该参数，则移动到transformer_kwargs中
                        if arg_name in kwargs:
                            transformer_kwargs[arg_name] = kwargs.pop(arg_name)
                    # 实例化变换器并调用transform方法
                    return self.transform(_transformer(**transformer_kwargs), **kwargs)
                # 直接使用变换器对象
                return self.transform(_transformer, **kwargs)

            # 替换方法签名以匹配变换器的签名
            if replace_signature:
                # 获取变换器的构造函数签名
                source_sig = inspect.signature(transformer.__init__)
                # 获取新方法的参数
                new_method_params = tuple(inspect.signature(new_method).parameters.values())
                # 根据变换器类型构造新的签名
                if inspect.isclass(transformer):
                    # 对于类：self + 构造函数参数[1:] + **kwargs
                    transformer_params = tuple(source_sig.parameters.values())
                    source_sig = inspect.Signature(
                        (new_method_params[0],) + transformer_params[1:] + (new_method_params[-1],))
                    new_method.__signature__ = source_sig
                else:
                    # 对于对象：self + **kwargs
                    source_sig = inspect.Signature((new_method_params[0],) + (new_method_params[-1],))
                    new_method.__signature__ = source_sig

            # 设置方法的文档字符串
            new_method.__doc__ = docstring
            # 设置方法的完全限定名称，用于调试和文档生成
            new_method.__qualname__ = f"{cls.__name__}.{target_name}"
            # 设置方法名称
            new_method.__name__ = target_name
            # 将新方法添加到类中
            setattr(cls, target_name, new_method)
        # 返回增强后的类
        return cls

    # 返回装饰器函数
    return wrapper
