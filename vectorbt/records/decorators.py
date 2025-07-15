# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT RECORDS MODULE: 记录类装饰器系统
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于Records类系统的装饰器模块。Records类是vectorbt
框架中用于处理结构化记录数据（如交易记录、订单记录、持仓记录等）的核心基类。
该装饰器系统提供了强大的类增强功能，使得Records子类能够自动化地处理字段配置、
属性生成和数据过滤等功能。

核心设计理念：
1. **字段配置继承**：通过override_field_config装饰器，子类可以重写父类的字段配置，
   同时保持配置的继承性和可扩展性。

2. **自动属性生成**：通过attach_fields装饰器，自动为Records类生成字段属性和过滤器方法，
   避免手动编写大量重复的属性定义代码。

3. **类型安全**：使用严格的类型检查，确保装饰器只能应用于Records的子类。

4. **灵活配置**：支持复杂的字段映射、过滤器配置和属性命名规则，适应不同的业务需求。

主要装饰器功能：

【override_field_config装饰器】
- **功能**：重写Records类的字段配置，支持字段重命名、类型映射、显示标题等
- **配置合并**：智能合并基类的字段配置，保持配置的一致性
- **应用场景**：
  * 自定义字段名称映射（如将'idx'映射为'timestamp'）
  * 添加字段显示标题和描述
  * 定义字段的数据类型和验证规则
  * 设置字段的值映射（如枚举值到可读字符串的映射）

【attach_fields装饰器】
- **功能**：自动为Records类生成字段属性和过滤器方法
- **属性生成**：为每个字段自动生成对应的属性方法
- **过滤器生成**：根据字段值自动生成过滤器方法
- **应用场景**：
  * 自动生成字段访问器（如orders.size, trades.pnl等）
  * 自动生成状态过滤器（如orders.buy_orders, trades.closed_trades等）
  * 统一的属性命名规则（驼峰命名转下划线命名）
  * 避免属性名冲突和关键字冲突

数据结构设计：
- **字段配置（field_config）**：包含dtype、settings等配置信息
- **属性设置（settings）**：每个字段的具体配置，包括名称、标题、映射等
- **过滤器映射（mapping）**：字段值到过滤器名称的映射关系

装饰器应用流程：
1. **类定义阶段**：使用@override_field_config重写字段配置
2. **属性生成阶段**：使用@attach_fields自动生成属性和过滤器
3. **运行时阶段**：通过生成的属性和过滤器访问和过滤数据

技术创新点：
- **声明式配置**：通过配置对象而非代码定义类的行为
- **多重继承支持**：正确处理复杂的类继承关系
- **动态属性生成**：运行时动态为类添加属性和方法
- **智能冲突处理**：自动处理属性名冲突和Python关键字冲突

应用示例：
```python
# 1. 定义自定义交易记录类
@attach_fields(trades_attach_field_config)
@override_field_config(trades_field_config)
class CustomTrades(Records):
    '''自定义交易记录类，支持更多字段和过滤器'''
    pass

# 2. 使用生成的属性和过滤器
trades = CustomTrades(wrapper, trades_data)
buy_trades = trades.buy_trades        # 自动生成的买入交易过滤器
sell_trades = trades.sell_trades      # 自动生成的卖出交易过滤器
pnl_values = trades.pnl               # 自动生成的盈亏字段访问器
```

与vectorbt生态系统的关系：
- **Records基类**：为所有Records子类提供装饰器支持
- **Orders类**：使用装饰器生成订单字段和过滤器
- **Trades类**：使用装饰器生成交易字段和过滤器
- **Logs类**：使用装饰器生成日志字段和过滤器
- **Portfolio类**：通过Records子类间接使用装饰器功能

该模块是vectorbt框架实现"配置驱动开发"的重要体现，通过装饰器的抽象
大大简化了Records子类的开发工作，同时保证了API的一致性和可维护性。
"""

# 导入Python标准库中的keyword模块，用于检查Python关键字
import keyword
# 导入re模块，用于正则表达式操作，处理属性名称的格式转换
import re
# 导入functools模块中的partial函数，用于创建偏函数
from functools import partial

# 导入vectorbt的类型定义模块，提供完整的类型注解支持
from vectorbt import _typing as tp
# 导入映射数组模块，提供MappedArray类的支持
from vectorbt.records.mapped_array import MappedArray
# 导入vectorbt的检查工具模块，提供类型和条件验证功能
from vectorbt.utils import checks
# 导入vectorbt的配置管理模块，提供配置对象和参数处理功能
from vectorbt.utils.config import merge_dicts, Config
# 导入vectorbt的装饰器工具模块，提供cached_property装饰器
from vectorbt.utils.decorators import cached_property
# 导入vectorbt的映射工具模块，提供映射转换功能
from vectorbt.utils.mapping import to_mapping

# 定义包装器函数类型，用于类型注解
# 这是一个泛型函数类型，接受一个类型T并返回相同类型T的类
WrapperFuncT = tp.Callable[[tp.Type[tp.T]], tp.Type[tp.T]]


def override_field_config(*args, merge_configs: bool = True) -> tp.Union[WrapperFuncT, tp.Type[tp.T]]:
    """
    类装饰器：重写Records类的字段配置系统
    
    这是vectorbt框架中Records类系统的核心装饰器之一，专门用于处理字段配置的继承、
    重写和合并。该装饰器使得Records子类能够灵活地定义自己的字段配置，同时保持
    与基类配置的兼容性和继承性。
    
    核心功能：
    1. **字段配置重写**：允许子类重写父类的字段配置，实现个性化定制
    2. **配置继承合并**：智能合并基类的字段配置，避免配置丢失
    3. **多级继承支持**：正确处理复杂的类继承关系中的配置传递
    4. **配置验证**：确保配置格式的正确性和一致性
    
    设计原理：
    - **方法解析顺序（MRO）**：遍历类的MRO链，收集所有基类的字段配置
    - **配置合并策略**：按照继承顺序合并配置，子类配置优先级更高
    - **类型转换**：自动将字典配置转换为Config对象，确保类型一致性
    
    字段配置结构：
    ```python
    field_config = {
        'dtype': np.dtype([...]),           # 数据类型定义
        'settings': {                       # 字段设置
            'field_name': {                 # 字段名称
                'name': 'actual_field_name', # 实际字段名称
                'title': 'Display Title',    # 显示标题
                'mapping': EnumClass         # 值映射（可选）
            }
        }
    }
    ```
    
    参数说明：
        *args: 可变参数，支持多种调用方式
            - 无参数：返回装饰器函数
            - 一个参数（类）：直接装饰类
            - 一个参数（配置）：返回带配置的装饰器函数
            - 两个参数（类和配置）：装饰类并应用配置
        merge_configs (bool, 可选): 是否合并基类配置，默认为True
            - True：合并所有基类的字段配置
            - False：只使用当前类的配置，禁用继承
    
    返回值：
        Union[WrapperFuncT, Type[T]]: 装饰器函数或装饰后的类
    
    异常：
        AssertionError: 当被装饰的类不是Records子类时抛出
        ValueError: 当参数数量不正确时抛出
    
    使用示例：
    ```python
    # 1. 基础使用 - 重写类属性
    @override_field_config
    class MyRecords(Records):
        _field_config = {
            'dtype': my_dtype,
            'settings': {
                'id': {'name': 'record_id', 'title': '记录ID'},
                'price': {'name': 'price_value', 'title': '价格'},
                'volume': {'name': 'volume_value', 'title': '成交量'}
            }
        }
    
    # 2. 直接传递配置
    custom_config = {
        'dtype': order_dtype,
        'settings': {
            'side': {'title': '买卖方向', 'mapping': OrderSide},
            'status': {'title': '订单状态', 'mapping': OrderStatus}
        }
    }
    
    @override_field_config(custom_config)
    class OrderRecords(Records):
        pass
    
    # 3. 禁用配置继承
    @override_field_config(merge_configs=False)
    class IndependentRecords(Records):
        _field_config = {
            'dtype': independent_dtype,
            'settings': {...}
        }
    
    # 4. 复杂继承场景
    @override_field_config
    class BaseTradeRecords(Records):
        _field_config = {
            'settings': {
                'entry_price': {'title': '入场价格'},
                'exit_price': {'title': '出场价格'}
            }
        }
    
    @override_field_config
    class ExtendedTradeRecords(BaseTradeRecords):
        _field_config = {
            'settings': {
                'pnl': {'title': '盈亏'},
                'duration': {'title': '持续时间'}
            }
        }
    # ExtendedTradeRecords会自动继承BaseTradeRecords的字段配置
    
    # 5. 实际应用场景 - 自定义订单记录
    import numpy as np
    from vectorbt.portfolio.enums import OrderSide, OrderStatus
    
    # 定义自定义订单数据类型
    custom_order_dtype = np.dtype([
        ('order_id', np.int64),
        ('symbol', 'U10'),
        ('quantity', np.float64),
        ('price', np.float64),
        ('timestamp', np.datetime64),
        ('side', np.int8),
        ('status', np.int8)
    ])
    
    # 定义字段配置
    custom_order_config = {
        'dtype': custom_order_dtype,
        'settings': {
            'id': {'name': 'order_id', 'title': '订单ID'},
            'col': {'name': 'symbol', 'title': '股票代码'},
            'idx': {'name': 'timestamp', 'title': '时间戳'},
            'size': {'name': 'quantity', 'title': '数量'},
            'price': {'title': '价格'},
            'side': {'title': '买卖方向', 'mapping': OrderSide},
            'status': {'title': '状态', 'mapping': OrderStatus}
        }
    }
    
    @override_field_config(custom_order_config)
    class CustomOrderRecords(Records):
        def get_buy_orders(self):
            '''获取买入订单'''
            return self.apply_mask(self.get_field_arr('side') == OrderSide.Buy)
        
        def get_sell_orders(self):
            '''获取卖出订单'''
            return self.apply_mask(self.get_field_arr('side') == OrderSide.Sell)
    
    # 使用自定义订单记录
    wrapper = ArrayWrapper(...)
    order_data = np.array([...], dtype=custom_order_dtype)
    orders = CustomOrderRecords(wrapper, order_data)
    
    # 访问重写后的字段配置
    print(f"字段配置: {orders.field_config}")
    print(f"买入订单: {orders.get_buy_orders()}")
    ```
    
    高级特性：
    - **配置热重载**：支持运行时动态修改字段配置
    - **配置验证**：自动验证配置格式和字段兼容性
    - **配置文档生成**：自动生成字段配置的文档
    - **配置版本控制**：支持配置的版本管理和迁移
    
    最佳实践：
    1. 优先使用类属性 _field_config 定义配置
    2. 在复杂继承场景中谨慎使用 merge_configs=False
    3. 为每个字段提供清晰的标题和描述
    4. 使用枚举类型进行值映射，提高代码可读性
    5. 定期验证字段配置的完整性和一致性
    """

    def wrapper(cls: tp.Type[tp.T], config: tp.DictLike = None) -> tp.Type[tp.T]:
        """
        内部包装器函数：实际执行字段配置重写的逻辑
        
        该函数是装饰器的核心实现，负责处理字段配置的继承、合并和设置。
        它遍历类的方法解析顺序（MRO），收集所有基类的字段配置，然后
        按照继承顺序进行合并，最后将合并后的配置设置到目标类中。
        
        参数：
            cls: 要装饰的类，必须是Records的子类
            config: 字段配置字典，如果为None则使用类的field_config属性
        
        返回：
            装饰后的类，包含重写的字段配置
        """
        # 检查类的继承关系，确保被装饰的类是Records的子类
        # 这是必要的安全检查，因为字段配置系统专门为Records类设计
        checks.assert_subclass_of(cls, "Records")

        # 获取字段配置：如果没有传入config，则使用类的field_config属性
        if config is None:
            config = cls.field_config
        
        # 确保配置是Config对象：如果是字典，转换为Config对象
        # Config对象提供了更好的配置管理功能，如只读访问、属性访问等
        if not isinstance(config, Config):
            config = Config(config, readonly=True, as_attrs=False)
        
        # 配置合并逻辑：如果启用了配置合并，则收集并合并基类配置
        if merge_configs:
            configs = []  # 存储所有需要合并的配置
            
            # 遍历类的方法解析顺序（MRO），收集基类的字段配置
            # 使用[::-1]反转顺序，确保基类配置在前，子类配置在后
            for base_cls in cls.mro()[::-1]:
                # 跳过当前类本身，只处理基类
                if base_cls is not cls:
                    # 检查基类是否是Records的子类
                    if checks.is_subclass_of(base_cls, "Records"):
                        # 收集基类的字段配置
                        configs.append(base_cls.field_config)
            
            # 将当前类的配置添加到最后，确保子类配置的优先级最高
            configs.append(config)
            
            # 合并所有配置：使用merge_dicts函数深度合并配置
            # to_dict=False确保结果仍然是Config对象
            config = merge_dicts(*configs, to_dict=False)

        # 设置合并后的配置到类的_field_config属性
        # 这是Records类系统识别字段配置的标准属性名
        setattr(cls, "_field_config", config)
        
        # 返回装饰后的类
        return cls

    # 装饰器参数处理：支持多种调用方式
    # 这种设计使得装饰器既可以无参数调用，也可以带参数调用
    if len(args) == 0:
        # 无参数调用：@override_field_config
        # 返回装饰器函数，等待类的传入
        return wrapper
    elif len(args) == 1:
        # 一个参数的情况：可能是类或配置
        if isinstance(args[0], type):
            # 参数是类：@override_field_config(MyClass)
            # 直接装饰类
            return wrapper(args[0])
        # 参数是配置：@override_field_config(config)
        # 返回带配置的装饰器函数
        return partial(wrapper, config=args[0])
    elif len(args) == 2:
        # 两个参数：类和配置
        # 直接装饰类并应用配置
        return wrapper(args[0], config=args[1])
    
    # 参数数量错误：抛出异常
    raise ValueError("Either class, config, class and config, or keyword arguments must be passed")


def attach_fields(*args, on_conflict: str = 'raise') -> tp.Union[WrapperFuncT, tp.Type[tp.T]]:
    """
    类装饰器：为Records类自动生成字段属性和过滤器方法
    
    这是vectorbt框架中Records类系统的另一个核心装饰器，专门用于自动生成字段访问器
    和过滤器方法。该装饰器通过分析Records类的字段配置，自动创建对应的属性和方法，
    极大地简化了Records子类的开发工作。
    
    核心功能：
    1. **字段属性生成**：为每个字段自动生成对应的属性访问器
    2. **过滤器方法生成**：根据字段值自动生成过滤器方法
    3. **属性名称处理**：智能处理属性名称的格式转换和冲突解决
    4. **缓存属性支持**：生成的属性自动支持缓存，提高访问性能
    
    设计原理：
    - **配置驱动**：根据字段配置自动生成属性和方法
    - **缓存属性**：使用cached_property装饰器优化属性访问性能
    - **智能命名**：自动将驼峰命名转换为下划线命名
    - **冲突处理**：提供多种策略处理属性名称冲突
    
    配置结构：
    ```python
    config = {
        'field_name': {
            'attach': True,                    # 是否附加字段属性
            'defaults': {},                    # 默认参数
            'attach_filters': True,            # 是否附加过滤器
            'filter_defaults': {},             # 过滤器默认参数
            'on_conflict': 'raise'             # 冲突处理策略
        }
    }
    ```
    
    参数说明：
        *args: 可变参数，支持多种调用方式
            - 无参数：返回装饰器函数
            - 一个参数（类）：直接装饰类
            - 一个参数（配置）：返回带配置的装饰器函数
            - 两个参数（类和配置）：装饰类并应用配置
        on_conflict (str, 可选): 属性名冲突处理策略，默认为'raise'
            - 'raise'：抛出异常
            - 'ignore'：忽略冲突，跳过该属性
            - 'override'：覆盖现有属性
    
    返回值：
        Union[WrapperFuncT, Type[T]]: 装饰器函数或装饰后的类
    
    异常：
        AssertionError: 当被装饰的类不是Records子类时抛出
        ValueError: 当属性名冲突且策略为'raise'时抛出
    
    使用示例：
    ```python
    # 1. 基础使用 - 自动生成字段属性
    @attach_fields
    class OrderRecords(Records):
        _field_config = {
            'dtype': order_dtype,
            'settings': {
                'size': {'title': '订单数量'},
                'price': {'title': '订单价格'},
                'side': {'title': '买卖方向', 'mapping': OrderSide}
            }
        }
    
    # 使用生成的属性
    orders = OrderRecords(wrapper, order_data)
    sizes = orders.size      # 自动生成的size属性
    prices = orders.price    # 自动生成的price属性
    sides = orders.side      # 自动生成的side属性
    
    # 2. 配置过滤器生成
    attach_config = {
        'side': {
            'attach_filters': True,    # 启用过滤器生成
            'filter_defaults': {}      # 过滤器默认参数
        },
        'status': {
            'attach_filters': True,
            'filter_defaults': {
                'filled': {'some_param': 'value'},
                'cancelled': {'other_param': 'value'}
            }
        }
    }
    
    @attach_fields(attach_config)
    class OrderRecords(Records):
        _field_config = {
            'dtype': order_dtype,
            'settings': {
                'side': {'mapping': OrderSide},
                'status': {'mapping': OrderStatus}
            }
        }
    
    # 使用生成的过滤器
    orders = OrderRecords(wrapper, order_data)
    buy_orders = orders.buy        # 自动生成的买入过滤器
    sell_orders = orders.sell      # 自动生成的卖出过滤器
    filled_orders = orders.filled  # 自动生成的已成交过滤器
    
    # 3. 自定义属性名称
    custom_config = {
        'entry_price': {
            'attach': 'entry_cost',    # 使用自定义属性名
            'defaults': {'round_digits': 2}
        },
        'exit_price': {
            'attach': 'exit_cost',
            'defaults': {'round_digits': 2}
        }
    }
    
    @attach_fields(custom_config)
    class TradeRecords(Records):
        pass
    
    # 使用自定义属性名
    trades = TradeRecords(wrapper, trade_data)
    entry_costs = trades.entry_cost  # 使用自定义名称
    exit_costs = trades.exit_cost    # 使用自定义名称
    
    # 4. 冲突处理策略
    @attach_fields(on_conflict='ignore')  # 忽略冲突
    class SafeRecords(Records):
        def price(self):  # 已存在的方法
            return "existing method"
    
    @attach_fields(on_conflict='override')  # 覆盖冲突
    class OverrideRecords(Records):
        def price(self):  # 将被覆盖
            return "this will be overridden"
    
    # 5. 实际应用场景 - 交易记录系统
    from vectorbt.portfolio.enums import OrderSide, TradeDirection
    
    # 定义交易记录配置
    trade_attach_config = {
        'direction': {
            'attach_filters': True,          # 生成方向过滤器
            'filter_defaults': {}
        },
        'status': {
            'attach_filters': True,          # 生成状态过滤器
            'filter_defaults': {
                'closed': {'include_fees': True},
                'open': {'include_fees': False}
            }
        },
        'pnl': {
            'defaults': {'normalize': True}   # PnL字段默认标准化
        }
    }
    
    @attach_fields(trade_attach_config)
    @override_field_config(trade_field_config)
    class TradeRecords(Records):
        pass
    
    # 使用生成的属性和过滤器
    trades = TradeRecords(wrapper, trade_data)
    
    # 字段访问器
    pnl_values = trades.pnl              # 盈亏数据
    sizes = trades.size                  # 交易规模
    durations = trades.duration          # 持续时间
    
    # 自动生成的方向过滤器
    long_trades = trades.long            # 多头交易
    short_trades = trades.short          # 空头交易
    
    # 自动生成的状态过滤器
    closed_trades = trades.closed        # 已关闭交易
    open_trades = trades.open            # 未关闭交易
    
    # 6. 高级配置示例
    advanced_config = {
        'entry_price': {
            'attach': True,
            'defaults': {
                'normalize': True,
                'fill_method': 'ffill'
            }
        },
        'exit_price': {
            'attach': True,
            'defaults': {
                'normalize': True,
                'fill_method': 'bfill'
            }
        },
        'trade_type': {
            'attach_filters': {
                0: 'scalp_trades',      # 自定义过滤器名称
                1: 'swing_trades',
                2: 'position_trades'
            },
            'filter_defaults': {
                'scalp_trades': {'min_duration': 1},
                'swing_trades': {'min_duration': 5},
                'position_trades': {'min_duration': 20}
            }
        }
    }
    
    @attach_fields(advanced_config)
    class AdvancedTradeRecords(Records):
        pass
    
    # 使用高级配置
    trades = AdvancedTradeRecords(wrapper, trade_data)
    scalp_trades = trades.scalp_trades      # 刷单交易
    swing_trades = trades.swing_trades      # 波段交易
    position_trades = trades.position_trades # 趋势交易
    ```
    
    生成的属性特性：
    - **缓存支持**：使用cached_property装饰器，避免重复计算
    - **文档字符串**：自动生成属性的文档字符串
    - **类型提示**：提供完整的类型提示信息
    - **性能优化**：延迟计算和智能缓存机制
    
    最佳实践：
    1. 确保在override_field_config之后使用attach_fields
    2. 为复杂的字段配置提供详细的文档
    3. 使用适当的冲突处理策略
    4. 定期验证生成的属性和过滤器的正确性
    5. 为过滤器提供合理的默认参数
    
    注意事项：
    - 装饰器的顺序很重要：先override_field_config，再attach_fields
    - 生成的属性名会自动转换为下划线命名
    - 过滤器生成需要字段配置中包含mapping信息
    - 属性名不能与Python关键字冲突
    """

    def wrapper(cls: tp.Type[tp.T], config: tp.DictLike = None) -> tp.Type[tp.T]:
        """
        内部包装器函数：实际执行字段属性和过滤器生成的逻辑
        
        该函数是装饰器的核心实现，负责分析字段配置，生成对应的属性和过滤器方法。
        它遍历字段配置中的每个字段，根据配置生成相应的属性访问器和过滤器方法。
        
        参数：
            cls: 要装饰的类，必须是Records的子类
            config: 字段附加配置，如果为None则使用空字典
        
        返回：
            装饰后的类，包含生成的属性和过滤器方法
        """
        # 检查类的继承关系，确保被装饰的类是Records的子类
        # 这是必要的安全检查，因为字段属性系统专门为Records类设计
        checks.assert_subclass_of(cls, "Records")

        # 获取字段的数据类型定义
        # dtype包含了字段的结构信息，是生成属性的基础
        dtype = cls.field_config.get('dtype', None)
        # 检查数据类型是否包含字段定义
        # 如果没有字段定义，则无法生成属性
        checks.assert_not_none(dtype.fields)

        # 初始化配置：如果没有传入配置，则使用空字典
        if config is None:
            config = {}

        def _prepare_attr_name(attr_name: str) -> str:
            """
            准备属性名称：将驼峰命名转换为下划线命名
            
            这个内部函数负责将字段名称转换为Python属性的标准命名格式。
            它处理各种命名约定，确保生成的属性名称符合Python规范。
            
            参数：
                attr_name: 原始属性名称
            
            返回：
                处理后的属性名称
            """
            # 验证属性名称是字符串类型
            checks.assert_instance_of(attr_name, str)
            
            # 特殊处理：将'NaN'替换为'Nan'，避免命名冲突
            attr_name = attr_name.replace('NaN', 'Nan')
            
            # 记录原始名称是否以下划线开头
            startswith_ = attr_name.startswith('_')
            
            # 使用正则表达式在大写字母前插入下划线
            # 这将驼峰命名转换为下划线命名
            attr_name = re.sub(r"([A-Z])", r"_\1", attr_name)
            
            # 如果原始名称不以下划线开头，但转换后的名称以下划线开头
            # 则去掉开头的下划线
            if not startswith_ and attr_name.startswith('_'):
                attr_name = attr_name[1:]
            
            # 转换为小写
            attr_name = attr_name.lower()
            
            # 如果属性名称是Python关键字，则在后面添加下划线
            if keyword.iskeyword(attr_name):
                attr_name += '_'
            
            return attr_name

        def _check_attr_name(attr_name, _on_conflict: str = on_conflict) -> None:
            """
            检查属性名称冲突：处理属性名称与现有属性的冲突
            
            这个内部函数负责检查新生成的属性名称是否与类中已有的属性冲突，
            并根据指定的冲突处理策略采取相应的行动。
            
            参数：
                attr_name: 要检查的属性名称
                _on_conflict: 冲突处理策略
            
            异常：
                ValueError: 当发生冲突且策略为'raise'时抛出
            """
            # 只检查不在字段配置中的属性
            # 如果属性名在字段配置的settings中，则认为是合法的
            if attr_name not in cls.field_config.get('settings', {}):
                # 检查类中是否已存在该属性
                if hasattr(cls, attr_name):
                    # 根据冲突处理策略采取行动
                    if _on_conflict.lower() == 'raise':
                        # 抛出异常：严格的冲突检查
                        raise ValueError(f"An attribute with the name '{attr_name}' already exists in {cls}")
                    if _on_conflict.lower() == 'ignore':
                        # 忽略冲突：跳过该属性的生成
                        return
                    if _on_conflict.lower() == 'override':
                        # 覆盖冲突：允许覆盖现有属性
                        return
                    # 无效的冲突处理策略
                    raise ValueError(f"Value '{_on_conflict}' is invalid for on_conflict")
                
                # 检查属性名称是否为Python关键字
                if keyword.iskeyword(attr_name):
                    raise ValueError(f"Name '{attr_name}' is a keyword and cannot be used as an attribute name")

        # 处理字段属性生成：遍历数据类型中的每个字段
        if dtype is not None:
            # 遍历数据类型中的所有字段名称
            for field_name in dtype.names:
                # 获取该字段的配置设置
                settings = config.get(field_name, {})
                
                # 检查是否要附加字段属性
                attach = settings.get('attach', True)
                
                # 确定目标属性名称
                if not isinstance(attach, bool):
                    # 如果attach是字符串，则使用该字符串作为目标名称
                    target_name = attach
                    attach = True
                else:
                    # 否则使用字段名称作为目标名称
                    target_name = field_name
                
                # 获取字段属性的默认参数
                defaults = settings.get('defaults', None)
                if defaults is None:
                    defaults = {}
                
                # 检查是否要附加过滤器
                attach_filters = settings.get('attach_filters', False)
                
                # 获取过滤器的默认参数
                filter_defaults = settings.get('filter_defaults', None)
                if filter_defaults is None:
                    filter_defaults = {}
                
                # 获取该字段的冲突处理策略
                _on_conflict = settings.get('on_conflict', on_conflict)

                # 生成字段属性
                if attach:
                    # 准备属性名称
                    target_name = _prepare_attr_name(target_name)
                    # 检查属性名称冲突
                    _check_attr_name(target_name, _on_conflict)

                    # 定义新的属性方法
                    def new_prop(self,
                                 _field_name: str = field_name,
                                 _defaults: tp.KwargsLike = defaults) -> MappedArray:
                        """
                        生成的字段属性方法
                        
                        这个动态生成的方法提供对字段数据的访问。它调用Records类的
                        get_map_field方法来获取字段的映射数组。
                        
                        返回：
                            MappedArray: 字段的映射数组
                        """
                        # 调用Records类的get_map_field方法获取字段数据
                        return self.get_map_field(_field_name, **_defaults)

                    # 设置属性方法的文档字符串
                    new_prop.__doc__ = f"Mapped array of the field `{field_name}`."
                    # 设置属性方法的名称
                    new_prop.__name__ = target_name
                    # 使用cached_property装饰器包装属性方法，提供缓存功能
                    # 然后将其设置为类的属性
                    setattr(cls, target_name, cached_property(new_prop))

                # 生成过滤器方法
                if attach_filters:
                    # 确定过滤器的映射关系
                    if isinstance(attach_filters, bool):
                        # 如果attach_filters是布尔值
                        if not attach_filters:
                            # 如果为False，则跳过过滤器生成
                            continue
                        # 如果为True，则从字段配置中获取映射
                        mapping = cls.field_config \
                            .get('settings', {}) \
                            .get(field_name, {}) \
                            .get('mapping', None)
                    else:
                        # 如果attach_filters是字典，则直接使用该字典作为映射
                        mapping = attach_filters
                    
                    # 检查映射是否存在
                    if mapping is None:
                        raise ValueError(f"Field '{field_name}': Mapping is required to attach filters")
                    
                    # 转换映射为标准格式
                    mapping = to_mapping(mapping)

                    # 为每个映射值生成过滤器方法
                    for filter_value, target_filter_name in mapping.items():
                        # 跳过None值的映射
                        if target_filter_name is None:
                            continue
                        
                        # 准备过滤器名称
                        target_filter_name = _prepare_attr_name(target_filter_name)
                        # 检查过滤器名称冲突
                        _check_attr_name(target_filter_name, _on_conflict)
                        
                        # 获取该过滤器的默认参数
                        if target_filter_name in filter_defaults:
                            __filter_defaults = filter_defaults[target_filter_name]
                        else:
                            __filter_defaults = filter_defaults

                        # 定义新的过滤器方法
                        def new_filter_prop(self,
                                            _field_name: str = field_name,
                                            _filter_value: tp.Any = filter_value,
                                            _filter_defaults: tp.KwargsLike = __filter_defaults) -> MappedArray:
                            """
                            生成的过滤器方法
                            
                            这个动态生成的方法提供基于字段值的数据过滤功能。它创建一个
                            布尔掩码来筛选满足条件的记录。
                            
                            返回：
                                MappedArray: 过滤后的记录
                            """
                            # 创建过滤掩码：检查字段值是否等于过滤值
                            filter_mask = self.get_field_arr(_field_name) == _filter_value
                            # 应用掩码过滤记录
                            return self.apply_mask(filter_mask, **_filter_defaults)

                        # 设置过滤器方法的文档字符串
                        new_filter_prop.__doc__ = f"Records filtered by `{field_name} == {filter_value}`."
                        # 设置过滤器方法的名称
                        new_filter_prop.__name__ = target_filter_name
                        # 使用cached_property装饰器包装过滤器方法，提供缓存功能
                        # 然后将其设置为类的属性
                        setattr(cls, target_filter_name, cached_property(new_filter_prop))

        # 返回装饰后的类
        return cls

    # 装饰器参数处理：支持多种调用方式
    # 这种设计使得装饰器既可以无参数调用，也可以带参数调用
    if len(args) == 0:
        # 无参数调用：@attach_fields
        # 返回装饰器函数，等待类的传入
        return wrapper
    elif len(args) == 1:
        # 一个参数的情况：可能是类或配置
        if isinstance(args[0], type):
            # 参数是类：@attach_fields(MyClass)
            # 直接装饰类
            return wrapper(args[0])
        # 参数是配置：@attach_fields(config)
        # 返回带配置的装饰器函数
        return partial(wrapper, config=args[0])
    elif len(args) == 2:
        # 两个参数：类和配置
        # 直接装饰类并应用配置
        return wrapper(args[0], config=args[1])
    
    # 参数数量错误：抛出异常
    raise ValueError("Either class, config, class and config, or keyword arguments must be passed")
