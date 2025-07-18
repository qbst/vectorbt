# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT LABELS MODULE: 枚举类型定义 (Enumerations)
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中labels模块的枚举类型定义文件，主要定义了标签生成器
所使用的各种模式和配置参数。该文件采用命名元组（NamedTuple）来实现枚举类型，
这种设计是为了确保与Numba JIT编译器的完全兼容性，从而获得最佳的执行性能。

核心设计理念：
1. **Numba兼容性优先**：使用NamedTuple而不是标准的Enum类，确保在Numba编译环境中的高性能
2. **类型安全**：通过类型定义确保枚举值的正确性和一致性
3. **语义清晰**：枚举名称直观地反映了其功能和用途
4. **扩展性**：便于添加新的模式和配置选项

技术实现特点：
- 使用tp.NamedTuple作为基类，确保Numba编译兼容性
- 枚举值从0开始的整数序列，便于数组索引和条件判断
- 通过__pdoc__字典提供详细的文档说明，支持自动文档生成
- 集成vectorbt.utils.docs模块，提供JSON格式的文档展示

枚举类型分类：
【趋势模式枚举 (TrendMode)】
- Binary: 二进制趋势标签，用于简单的上涨/下跌分类
- BinaryCont: 连续二进制标签，提供0-1之间的连续值
- BinaryContSat: 饱和连续标签，包含阈值饱和处理
- PctChange: 百分比变化标签，计算实际收益率
- PctChangeNorm: 标准化百分比变化，提供对称性

应用场景：
- 机器学习模型训练：为不同类型的预测任务选择合适的标签模式
- 量化策略开发：根据策略需求选择相应的趋势识别模式
- 回测分析：使用不同模式分析策略在各种市场条件下的表现
- 风险管理：基于不同趋势模式进行风险评估和控制

与其他模块的关系：
- labels.nb模块：各种标签生成算法的底层实现
- labels.generators模块：标签生成器的高层接口
- indicators.factory模块：指标工厂的参数配置
- utils.enum_模块：枚举映射和转换工具

使用示例：
```python
import vectorbt as vbt
import pandas as pd
import numpy as np

# 创建示例价格数据
prices = pd.Series([100, 105, 98, 110, 95, 108, 103, 95, 112, 98])

# 使用不同的趋势模式生成标签
# 1. 二进制模式 - 简单的上涨/下跌标签
binary_labels = vbt.TRENDLB.run(
    prices, 
    pos_th=0.05, 
    neg_th=0.05, 
    mode=vbt.labels.enums.TrendMode.Binary
)

# 2. 连续模式 - 0-1之间的连续值
continuous_labels = vbt.TRENDLB.run(
    prices, 
    pos_th=0.05, 
    neg_th=0.05, 
    mode=vbt.labels.enums.TrendMode.BinaryCont
)

# 3. 百分比变化模式 - 实际收益率
pct_labels = vbt.TRENDLB.run(
    prices, 
    pos_th=0.05, 
    neg_th=0.05, 
    mode=vbt.labels.enums.TrendMode.PctChange
)

# 比较不同模式的输出
print("二进制模式:", binary_labels.labels.dropna().unique())
print("连续模式范围:", continuous_labels.labels.dropna().min(), "到", continuous_labels.labels.dropna().max())
print("百分比变化范围:", pct_labels.labels.dropna().min(), "到", pct_labels.labels.dropna().max())
```

该文件是vectorbt框架中标签生成功能的基础配置文件，为整个标签生成系统提供了
统一的模式定义和类型安全保障。
================================================================================

命名元组和枚举类型定义

为vectorbt.labels模块定义枚举类型和其他模式配置。
"""

# 导入必要的模块
from vectorbt import _typing as tp  # 导入vectorbt的类型定义模块
from vectorbt.utils.docs import to_doc  # 导入文档生成工具，用于生成JSON格式的文档

# 定义模块的公开接口，只导出TrendMode枚举
__all__ = [
    'TrendMode'  # 趋势模式枚举，是本模块的核心导出
]

# 文档生成配置字典，用于控制自动文档生成
__pdoc__ = {}


class TrendModeT(tp.NamedTuple):
    """
    趋势模式类型定义
    
    该类定义了趋势标签生成器支持的所有模式类型。使用NamedTuple实现是为了确保
    与Numba JIT编译器的完全兼容性，从而在大规模数据处理中获得最佳性能。
    
    技术设计说明：
    - 继承自tp.NamedTuple，确保Numba编译时的类型安全
    - 使用整数值作为枚举值，便于数组索引和条件判断
    - 枚举值从0开始递增，遵循vectorbt的编码规范
    - 每个模式对应不同的标签生成算法和应用场景
    
    模式分类：
    1. 二进制模式：提供简单的分类标签（0或1）
    2. 连续模式：提供0-1之间的连续值标签
    3. 百分比模式：提供实际的收益率标签
    
    性能特点：
    - Numba编译优化：在JIT编译环境中达到接近C语言的执行速度
    - 内存效率：整数枚举值占用最少的内存空间
    - 类型安全：编译时类型检查，避免运行时错误
    """
    
    # 二进制趋势模式 (Binary Trend Mode)
    Binary: int = 0
    """
    二进制趋势标签模式
    
    该模式生成简单的二进制标签，用于基本的趋势方向分类。在局部极值点之间的区间内，
    根据趋势方向分配标签：下跌趋势为0，上涨趋势为1。
    
    标签含义：
    - 0: 下跌趋势（从波峰到波谷的区间）
    - 1: 上涨趋势（从波谷到波峰的区间）
    - NaN: 极值点之外的区域
    
    应用场景：
    - 分类模型训练：预测价格上涨或下跌
    - 趋势跟踪策略：识别基本的趋势方向
    - 信号生成：生成买入/卖出信号
    
    优势：
    - 简单直观：标签含义清晰明确
    - 计算高效：二进制操作速度快
    - 适用性广：适合大多数分类任务
    
    示例：
    ```python
    import vectorbt as vbt
    import pandas as pd
    
    prices = pd.Series([100, 105, 98, 110, 95, 108])
    labels = vbt.TRENDLB.run(prices, pos_th=0.05, neg_th=0.05, mode=vbt.labels.enums.TrendMode.Binary)
    print(labels.labels)  # 输出：0和1的序列
    ```
    """
    
    # 连续二进制趋势模式 (Binary Continuous Trend Mode)
    BinaryCont: int = 1
    """
    连续二进制趋势标签模式
    
    该模式在每个极值区间内将价格标准化为0-1的连续值，然后使用反向映射来表示
    未来的趋势方向。提供比二进制模式更丰富的信息。
    
    标签含义：
    - 接近0的值: 当前价格接近区间最高点，预期将下跌
    - 接近1的值: 当前价格接近区间最低点，预期将上涨
    - 中间值: 表示上涨或下跌的程度
    - NaN: 极值点之外的区域
    
    计算原理：
    在每个极值区间内，使用公式：
    label = 1 - (current_price - min_price) / (max_price - min_price)
    
    应用场景：
    - 回归模型训练：预测趋势的强度
    - 连续信号生成：生成渐进的交易信号
    - 风险评估：评估当前价格在趋势中的位置
    
    优势：
    - 信息丰富：提供连续的趋势强度信息
    - 平滑变化：避免二进制标签的突变
    - 位置敏感：反映价格在趋势区间中的相对位置
    
    示例：
    ```python
    import vectorbt as vbt
    import pandas as pd
    
    prices = pd.Series([100, 105, 98, 110, 95, 108])
    labels = vbt.TRENDLB.run(prices, pos_th=0.05, neg_th=0.05, mode=vbt.labels.enums.TrendMode.BinaryCont)
    print(labels.labels)  # 输出：0-1之间的连续值
    ```
    """
    
    # 饱和连续趋势模式 (Binary Continuous Saturated Trend Mode)
    BinaryContSat: int = 2
    """
    饱和连续趋势标签模式
    
    该模式在连续标签的基础上增加了饱和处理机制。当价格变化超过设定的阈值时，
    标签会被设置为饱和值（0或1），否则使用连续的插值。
    
    标签含义：
    - 0: 强烈的下跌信号（饱和状态）
    - 1: 强烈的上涨信号（饱和状态）
    - 中间值: 线性插值的趋势强度
    - NaN: 极值点之外的区域
    
    饱和机制：
    - 当预期上涨且价格足够低时，标签饱和为1
    - 当预期下跌且价格足够高时，标签饱和为0
    - 其他情况下使用线性插值
    
    应用场景：
    - 强化学习：提供明确的奖励信号
    - 风险控制：识别极端市场条件
    - 动态调整：根据市场状态调整策略参数
    
    优势：
    - 极端敏感：对极端情况提供强烈信号
    - 自适应：根据阈值动态调整标签
    - 平衡性：结合连续性和离散性的优势
    
    示例：
    ```python
    import vectorbt as vbt
    import pandas as pd
    
    prices = pd.Series([100, 105, 98, 110, 95, 108])
    labels = vbt.TRENDLB.run(prices, pos_th=0.03, neg_th=0.03, mode=vbt.labels.enums.TrendMode.BinaryContSat)
    print(labels.labels)  # 输出：包含饱和值的0-1连续值
    ```
    """
    
    # 百分比变化趋势模式 (Percentage Change Trend Mode)
    PctChange: int = 3
    """
    百分比变化趋势标签模式
    
    该模式计算每个时间点到下一个极值点的百分比变化，提供直观的收益率预测标签。
    使用当前价格作为分母，符合传统的收益率计算方式。
    
    标签含义：
    - 正值: 到下一个极值点的上涨百分比
    - 负值: 到下一个极值点的下跌百分比
    - 0: 价格无变化
    - NaN: 极值点之外的区域
    
    计算公式：
    label = (next_extrema_price - current_price) / current_price
    
    应用场景：
    - 收益率预测：直接预测未来收益率
    - 策略评估：评估策略的收益潜力
    - 风险分析：分析收益率分布特征
    
    优势：
    - 直观易懂：标签直接对应收益率
    - 实用性强：可直接用于投资决策
    - 兼容性好：与传统金融分析方法兼容
    
    示例：
    ```python
    import vectorbt as vbt
    import pandas as pd
    
    prices = pd.Series([100, 105, 98, 110, 95, 108])
    labels = vbt.TRENDLB.run(prices, pos_th=0.05, neg_th=0.05, mode=vbt.labels.enums.TrendMode.PctChange)
    print(labels.labels)  # 输出：百分比变化值（如0.05表示5%上涨）
    ```
    """
    
    # 标准化百分比变化趋势模式 (Normalized Percentage Change Trend Mode)
    PctChangeNorm: int = 4
    """
    标准化百分比变化趋势标签模式
    
    该模式计算标准化的百分比变化，使用未来价格作为分母，提供更好的数学对称性。
    这种计算方式在某些机器学习模型中具有更好的性质。
    
    标签含义：
    - 正值: 到下一个极值点的标准化上涨百分比
    - 负值: 到下一个极值点的标准化下跌百分比
    - 0: 价格无变化
    - NaN: 极值点之外的区域
    
    计算公式：
    对于上涨趋势：label = (next_extrema_price - current_price) / next_extrema_price
    对于下跌趋势：label = (next_extrema_price - current_price) / current_price
    
    数学特性：
    - 对称性更好：上涨和下跌的标签范围更加对称
    - 数值稳定：避免了极端价格比率的问题
    - 归一化：标签值在更合理的范围内
    
    应用场景：
    - 机器学习模型：提供数值稳定的训练标签
    - 对称分析：需要对称处理上涨和下跌的场景
    - 标准化处理：与其他归一化特征配合使用
    
    优势：
    - 数学性质好：更好的对称性和数值稳定性
    - 模型友好：适合神经网络等模型训练
    - 范围合理：避免极端值对模型的影响
    
    示例：
    ```python
    import vectorbt as vbt
    import pandas as pd
    
    prices = pd.Series([100, 105, 98, 110, 95, 108])
    labels = vbt.TRENDLB.run(prices, pos_th=0.05, neg_th=0.05, mode=vbt.labels.enums.TrendMode.PctChangeNorm)
    print(labels.labels)  # 输出：标准化的百分比变化值
    
    # 比较标准化和非标准化的区别
    normal_labels = vbt.TRENDLB.run(prices, pos_th=0.05, neg_th=0.05, mode=vbt.labels.enums.TrendMode.PctChange)
    print("标准化前:", normal_labels.labels.describe())
    print("标准化后:", labels.labels.describe())
    ```
    """


# 创建TrendMode枚举实例，这是一个全局单例对象
TrendMode = TrendModeT()
"""
趋势模式枚举实例
    
这是TrendModeT类的全局单例实例，提供了所有趋势标签生成模式的访问接口。
该实例在整个vectorbt框架中被广泛使用，确保了模式选择的一致性和类型安全性。

使用方式：
- 直接访问：TrendMode.Binary, TrendMode.BinaryCont 等
- 参数传递：在标签生成器中作为mode参数使用
- 条件判断：在算法中根据模式值进行分支处理

技术特点：
- 单例模式：全局唯一实例，节省内存
- 不可变性：枚举值创建后不可修改
- 类型安全：编译时类型检查
- Numba兼容：在JIT编译环境中高效运行

示例使用：
```python
import vectorbt as vbt

# 直接使用枚举值
mode = vbt.labels.enums.TrendMode.Binary
print(f"二进制模式值: {mode}")

# 作为参数传递
labels = vbt.TRENDLB.run(
    prices, 
    pos_th=0.05, 
    neg_th=0.05, 
    mode=vbt.labels.enums.TrendMode.BinaryCont
)

# 获取所有可用模式
all_modes = [
    vbt.labels.enums.TrendMode.Binary,
    vbt.labels.enums.TrendMode.BinaryCont,
    vbt.labels.enums.TrendMode.BinaryContSat,
    vbt.labels.enums.TrendMode.PctChange,
    vbt.labels.enums.TrendMode.PctChangeNorm
]
print(f"所有模式: {all_modes}")
```
"""

# 为TrendMode添加详细的文档说明，用于自动文档生成
__pdoc__['TrendMode'] = f"""
趋势模式枚举

该枚举定义了vectorbt.labels模块中趋势标签生成器支持的所有模式类型。
每种模式对应不同的标签生成算法和应用场景。

```json
{to_doc(TrendMode)}
```

属性详解：
    Binary: 二进制趋势标签模式
        - 值: 0
        - 算法: vectorbt.labels.nb.bn_trend_labels_nb
        - 输出: 0（下跌）或1（上涨）的离散标签
        - 适用: 简单的趋势分类任务
    
    BinaryCont: 连续二进制趋势标签模式
        - 值: 1
        - 算法: vectorbt.labels.nb.bn_cont_trend_labels_nb
        - 输出: 0-1之间的连续值，表示趋势强度
        - 适用: 需要趋势强度信息的回归任务
    
    BinaryContSat: 饱和连续趋势标签模式
        - 值: 2
        - 算法: vectorbt.labels.nb.bn_cont_sat_trend_labels_nb
        - 输出: 带饱和处理的0-1连续值
        - 适用: 需要强化极端信号的场景
    
    PctChange: 百分比变化趋势标签模式
        - 值: 3
        - 算法: vectorbt.labels.nb.pct_trend_labels_nb (normalize=False)
        - 输出: 实际的百分比变化值
        - 适用: 直接的收益率预测任务
    
    PctChangeNorm: 标准化百分比变化趋势标签模式
        - 值: 4
        - 算法: vectorbt.labels.nb.pct_trend_labels_nb (normalize=True)
        - 输出: 标准化的百分比变化值
        - 适用: 需要数值稳定性的机器学习模型

使用建议：
- 对于简单的趋势分类，使用Binary模式
- 对于需要趋势强度的任务，使用BinaryCont模式
- 对于需要强化极端信号的场景，使用BinaryContSat模式
- 对于直接的收益率预测，使用PctChange模式
- 对于机器学习模型训练，优先考虑PctChangeNorm模式

性能说明：
- 所有模式都经过Numba JIT编译优化
- Binary模式计算最快，PctChangeNorm模式提供最好的数学性质
- 推荐根据具体应用场景选择合适的模式
"""
