# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
技术指标配置模块 - 统一的参数配置管理
================================================================================

定义了技术指标系统中的通用配置对象，用于标准化和统一化指标计算时的参数处理行为。

核心功能：
1. 参数广播配置：定义如何将参数广播到输入数据的维度
2. 内存优化配置：通过keep_raw选项优化内存使用
3. 列级处理配置：支持按列独立处理参数
4. 数组类型处理：标准化数组参数的处理方式

设计理念：
- 统一性：所有指标使用相同的配置标准
- 灵活性：支持元素级和列级两种不同的参数处理方式
- 效率性：优化内存使用和计算性能
- 可扩展性：便于添加新的配置选项
"""

# 导入配置管理类
from vectorbt.utils.config import Config  # 导入vectorbt的配置管理工具类

# 创建灵活的元素级参数配置对象
flex_elem_param_config = Config(
    dict(
        is_array_like=True,  # 标识参数为数组类型：传递NumPy数组表示单个值，多个值需使用列表
        bc_to_input=True,  # 启用广播到输入数据：参数将自动广播到与输入数据相同的形状
        broadcast_kwargs=dict(
            keep_raw=True  # 保持原始形状：为灵活索引保留原始数据形状以节省内存
        )
    )
)
"""
灵活元素级参数配置对象

这个配置对象定义了技术指标中元素级参数的标准处理行为。主要用于处理需要
逐个元素应用的参数，如滑动窗口大小、阈值等。

配置说明：
- is_array_like=True: 参数支持数组格式输入
- bc_to_input=True: 自动广播参数到输入数据的形状
- keep_raw=True: 保持原始数据形状，优化内存使用

使用场景：
- 移动平均线的窗口期参数
- RSI指标的超买超卖阈值
- 布林带的标准差倍数参数

使用示例：
```python
# 在IndicatorFactory中使用元素级配置
MA = IndicatorFactory(
    param_names=['window'],
    param_config=dict(
        window=flex_elem_param_config  # 使用灵活元素级配置
    )
)

# 支持多种参数形式
ma_single = MA.run(data, window=20)          # 单个窗口期
ma_multiple = MA.run(data, window=[5,10,20]) # 多个窗口期
ma_array = MA.run(data, window=np.array([20])) # 数组形式
```
"""

# 创建灵活的列级参数配置对象
flex_col_param_config = Config(
    dict(
        is_array_like=True,  # 标识参数为数组类型：支持数组格式的参数输入
        bc_to_input=1,  # 广播到轴1（列）：参数沿列方向广播，适用于多列数据处理
        per_column=True,  # 按列处理：每列显示一个参数值，支持列级独立配置
        broadcast_kwargs=dict(
            keep_raw=True  # 保持原始形状：优化内存使用，保留原始数据结构
        )
    )
)
"""
灵活列级参数配置对象

这个配置对象定义了技术指标中列级参数的标准处理行为。主要用于处理需要
按列独立应用的参数，适用于多资产、多策略的并行计算场景。

配置说明：
- is_array_like=True: 参数支持数组格式输入
- bc_to_input=1: 沿轴1（列）方向广播参数
- per_column=True: 支持每列独立参数设置
- keep_raw=True: 保持原始数据形状，优化内存使用

使用场景：
- 多股票组合的技术指标计算
- 不同参数的并行回测
- 多时间框架的指标分析

使用示例：
```python
# 在IndicatorFactory中使用列级配置
RSI = IndicatorFactory(
    param_names=['window'],
    param_config=dict(
        window=flex_col_param_config  # 使用灵活列级配置
    )
)

# 多列数据，每列使用不同参数
import pandas as pd
prices = pd.DataFrame({
    'AAPL': [100, 102, 101, 103, 105],
    'GOOGL': [2000, 2010, 2005, 2015, 2020],
    'MSFT': [300, 305, 302, 308, 310]
})

# 每列使用不同的RSI窗口期
rsi_result = RSI.run(prices, window=[14, 21, 28])
# AAPL使用14日RSI，GOOGL使用21日RSI，MSFT使用28日RSI
```
"""
