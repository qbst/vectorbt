# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
通用数据的自定义 pandas 访问器模块

本模块是 vectorbt 量化分析框架的核心组件之一，为 pandas 的 Series 和 DataFrame 对象提供了
强大的 `.vbt` 访问器，扩展了 pandas 的基础功能，专门针对量化金融分析场景进行了优化。

设计逻辑：
1. **访问器模式**：采用 pandas 的访问器模式，通过 `pd.Series.vbt.*` 和 `pd.DataFrame.vbt.*` 
   提供统一的接口，使得 vectorbt 的功能能够与 pandas 无缝集成
2. **高性能计算**：大量使用 numba 编译的函数来提高计算性能，特别是在滚动窗口、扩展窗口等操作中
3. **模块化设计**：继承了 BaseAccessor、StatsBuilderMixin 和 PlotsBuilderMixin，实现了统计分析和
   绘图功能的模块化
4. **泛型支持**：支持各种数据类型的处理，包括时间序列、数值型、分类型等
5. **可扩展性**：为专门的访问器（如 signals.accessors 和 returns.accessors）提供了基础

主要功能模块：
- 数据统计分析：describe、value_counts、各种统计函数（min、max、mean、std等）
- 滚动和扩展窗口计算：rolling_mean、expanding_mean、ewm_mean 等
- 数据变换：fillna、差分、百分比变化、标准化等
- 分组和聚合：groupby_apply、resample_apply、reduce 等
- 绘图可视化：plot、lineplot、heatmap、boxplot 等
- 时间序列分析：回撤分析、范围分析、交叉分析等
- 数据分割：时间序列分割、滚动分割、扩展分割等
- 映射和转换：apply_mapping、transform、zscore 等

使用示例：
```python
import pandas as pd
import vectorbt as vbt
import numpy as np

# 基本统计分析
df = pd.DataFrame({
    'price': [100, 105, 98, 95, 102, 108, 103],
    'volume': [1000, 1200, 800, 600, 1100, 1500, 900]
})
stats = df.vbt.describe()  # 描述性统计
print(stats)

# 滚动窗口计算
rolling_mean = df['price'].vbt.rolling_mean(window=3)  # 3期滚动均值
print(rolling_mean)

# 绘图功能
fig = df.vbt.plot()  # 创建交互式图表
fig.show()

# 回撤分析
drawdowns = df['price'].vbt.drawdowns  # 自动识别回撤
print(drawdowns.records_readable)

# 数据变换
normalized = df.vbt.zscore()  # Z-score标准化
print(normalized)
```

访问器类型：
- GenericSRAccessor：针对 pd.Series 的访问器，提供 `pd.Series.vbt.*` 功能
- GenericDFAccessor：针对 pd.DataFrame 的访问器，提供 `pd.DataFrame.vbt.*` 功能

注意事项：
- 分组功能只支持接受 `group_by` 参数的方法
- 访问器不使用缓存机制，每次调用都会重新计算
- 大部分计算密集型操作都使用了 numba 编译优化
- 支持多种数据类型和时间频率的处理

继承关系：
- 继承自 vectorbt.base.accessors，获得基础访问器功能
- 被更专业的访问器继承，如 vectorbt.signals.accessors 和 vectorbt.returns.accessors
- 集成了 StatsBuilderMixin 和 PlotsBuilderMixin，提供统计和绘图功能

方法可以通过以下方式访问：
* `GenericSRAccessor` -> `pd.Series.vbt.*`
* `GenericDFAccessor` -> `pd.DataFrame.vbt.*`

示例用法：
```pycon
>>> import pandas as pd
>>> import vectorbt as vbt

>>> # 调用 vectorbt.generic.accessors.GenericAccessor.rolling_mean
>>> pd.Series([1, 2, 3, 4]).vbt.rolling_mean(2)
0    NaN
1    1.5
2    2.5
3    3.5
dtype: float64
```

访问器继承了 `vectorbt.base.accessors` 并被更专业的访问器继承，
如 `vectorbt.signals.accessors` 和 `vectorbt.returns.accessors`。

!!! note
    分组功能只支持接受 `group_by` 参数的方法。

    访问器不使用缓存机制。

运行以下示例需要先执行：
    
```pycon
>>> import vectorbt as vbt
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from datetime import datetime, timedelta

>>> df = pd.DataFrame({
...     'a': [1, 2, 3, 4, 5],
...     'b': [5, 4, 3, 2, 1],
...     'c': [1, 2, 3, 2, 1]
... }, index=pd.Index([
...     datetime(2020, 1, 1),
...     datetime(2020, 1, 2),
...     datetime(2020, 1, 3),
...     datetime(2020, 1, 4),
...     datetime(2020, 1, 5)
... ]))
>>> df
            a  b  c
2020-01-01  1  5  1
2020-01-02  2  4  2
2020-01-03  3  3  3
2020-01-04  4  2  2
2020-01-05  5  1  1

>>> index = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(10)]
>>> sr = pd.Series(np.arange(len(index)), index=index)
>>> sr
2020-01-01    0
2020-01-02    1
2020-01-03    2
2020-01-04    3
2020-01-05    4
2020-01-06    5
2020-01-07    6
2020-01-08    7
2020-01-09    8
2020-01-10    9
dtype: int64
```

## 统计分析

!!! hint
    参见 `vectorbt.generic.stats_builder.StatsBuilderMixin.stats` 和 `GenericAccessor.metrics`。

```pycon
>>> df2 = pd.DataFrame({
...     'a': [np.nan, 2, 3],
...     'b': [4, np.nan, 5],
...     'c': [6, 7, np.nan]
... }, index=['x', 'y', 'z'])

>>> df2.vbt(freq='d').stats(column='a')
Start                      x
End                        z
Period       3 days 00:00:00
Count                      2
Mean                     2.5
Std                 0.707107
Min                      2.0
Median                   2.5
Max                      3.0
Min Index                  y
Max Index                  z
Name: a, dtype: object
```

### 映射功能

映射可以在 `GenericAccessor`（推荐）和 `GenericAccessor.stats` 中设置：

```pycon
>>> mapping = {x: 'test_' + str(x) for x in pd.unique(df2.values.flatten())}
>>> df2.vbt(freq='d', mapping=mapping).stats(column='a')
Start                                   x
End                                     z
Period                    3 days 00:00:00
Count                                   2
Value Counts: test_2.0                  1
Value Counts: test_3.0                  1
Value Counts: test_4.0                  0
Value Counts: test_5.0                  0
Value Counts: test_6.0                  0
Value Counts: test_7.0                  0
Value Counts: test_nan                  1
Name: a, dtype: object

>>> df2.vbt(freq='d').stats(column='a', settings=dict(mapping=mapping))
UserWarning: Changing the mapping will create a copy of this object.
Consider setting it upon object creation to re-use existing cache.

Start                                   x
End                                     z
Period                    3 days 00:00:00
Count                                   2
Value Counts: test_2.0                  1
Value Counts: test_3.0                  1
Value Counts: test_4.0                  0
Value Counts: test_5.0                  0
Value Counts: test_6.0                  0
Value Counts: test_7.0                  0
Value Counts: test_nan                  1
Name: a, dtype: object
```

在调用 `stats` 之前选择列只会考虑来自该列的唯一值：

```pycon
>>> df2['a'].vbt(freq='d', mapping=mapping).stats()
Start                                   x
End                                     z
Period                    3 days 00:00:00
Count                                   2
Value Counts: test_2.0                  1
Value Counts: test_3.0                  1
Value Counts: test_nan                  1
Name: a, dtype: object
```

要包含 `mapping` 中的所有键，传递 `incl_all_keys=True`：

>>> df2['a'].vbt(freq='d', mapping=mapping).stats(settings=dict(incl_all_keys=True))
Start                                   x
End                                     z
Period                    3 days 00:00:00
Count                                   2
Value Counts: test_2.0                  1
Value Counts: test_3.0                  1
Value Counts: test_4.0                  0
Value Counts: test_5.0                  0
Value Counts: test_6.0                  0
Value Counts: test_7.0                  0
Value Counts: test_nan                  1
Name: a, dtype: object
```

`GenericAccessor.stats` 还支持（重新）分组：

```pycon
>>> df2.vbt(freq='d').stats(column=0, group_by=[0, 0, 1])
Start                      x
End                        z
Period       3 days 00:00:00
Count                      4
Mean                     3.5
Std                 1.290994
Min                      2.0
Median                   3.5
Max                      5.0
Min Index                  y
Max Index                  z
Name: 0, dtype: object
```

## 绘图功能

!!! hint
    参见 `vectorbt.generic.plots_builder.PlotsBuilderMixin.plots` 和 `GenericAccessor.subplots`。

`GenericAccessor` 类基于 `GenericAccessor.plot` 有一个单独的子图：

```pycon
>>> df2.vbt.plots()
```

![](/assets/images/generic_plots.svg)
"""

# 标准库导入
import warnings  # 用于处理警告信息

# 第三方库导入
import numpy as np  # 数值计算库
import pandas as pd  # 数据分析库
from numba.typed import Dict  # numba 类型化字典
from scipy import stats  # 统计分析库
from sklearn.exceptions import NotFittedError  # sklearn 异常处理
from sklearn.preprocessing import (  # sklearn 数据预处理工具
    Binarizer,              # 二值化器
    MinMaxScaler,           # 最小-最大缩放器
    MaxAbsScaler,           # 最大绝对值缩放器
    Normalizer,             # 标准化器
    RobustScaler,           # 鲁棒缩放器
    StandardScaler,         # 标准缩放器
    QuantileTransformer,    # 分位数变换器
    PowerTransformer        # 幂变换器
)
from sklearn.utils.validation import check_is_fitted  # 检查模型是否已拟合

# vectorbt 内部导入
from vectorbt import _typing as tp  # 类型定义
from vectorbt.base import index_fns, reshape_fns  # 基础工具函数
from vectorbt.base.accessors import BaseAccessor, BaseDFAccessor, BaseSRAccessor  # 基础访问器
from vectorbt.base.array_wrapper import ArrayWrapper, Wrapping  # 数组包装器
from vectorbt.generic import plotting, nb  # 绘图模块和 numba 函数
from vectorbt.generic.decorators import attach_nb_methods, attach_transform_methods  # 装饰器
from vectorbt.generic.drawdowns import Drawdowns  # 回撤分析类
from vectorbt.generic.plots_builder import PlotsBuilderMixin  # 绘图构建器混入类
from vectorbt.generic.ranges import Ranges  # 范围分析类
from vectorbt.generic.splitters import SplitterT, RangeSplitter, RollingSplitter, ExpandingSplitter  # 分割器类
from vectorbt.generic.stats_builder import StatsBuilderMixin  # 统计构建器混入类
from vectorbt.records.mapped_array import MappedArray  # 映射数组类
from vectorbt.utils import checks  # 检查工具
from vectorbt.utils.config import Config, merge_dicts, resolve_dict  # 配置工具
from vectorbt.utils.figure import make_figure, make_subplots  # 图形工具
from vectorbt.utils.mapping import apply_mapping, to_mapping  # 映射工具

# 尝试导入 bottleneck 库以获得更好的性能
try:  # pragma: no cover
    import bottleneck as bn  # 快速数值计算库

    # 使用 bottleneck 的高性能 NaN 处理函数
    nanmean = bn.nanmean        # 计算忽略 NaN 的均值
    nanstd = bn.nanstd          # 计算忽略 NaN 的标准差
    nansum = bn.nansum          # 计算忽略 NaN 的和
    nanmax = bn.nanmax          # 计算忽略 NaN 的最大值
    nanmin = bn.nanmin          # 计算忽略 NaN 的最小值
    nanmedian = bn.nanmedian    # 计算忽略 NaN 的中位数
    nanargmax = bn.nanargmax    # 计算忽略 NaN 的最大值索引
    nanargmin = bn.nanargmin    # 计算忽略 NaN 的最小值索引
except ImportError:
    # 如果 bottleneck 不可用，使用较慢的 numpy 函数
    nanmean = np.nanmean        # numpy 的忽略 NaN 均值函数
    nanstd = np.nanstd          # numpy 的忽略 NaN 标准差函数
    nansum = np.nansum          # numpy 的忽略 NaN 求和函数
    nanmax = np.nanmax          # numpy 的忽略 NaN 最大值函数
    nanmin = np.nanmin          # numpy 的忽略 NaN 最小值函数
    nanmedian = np.nanmedian    # numpy 的忽略 NaN 中位数函数
    nanargmax = np.nanargmax    # numpy 的忽略 NaN 最大值索引函数
    nanargmin = np.nanargmin    # numpy 的忽略 NaN 最小值索引函数

# 用于存储文档字符串的字典
__pdoc__ = {}


# 通用访问器的元类定义
# 该元类结合了 StatsBuilderMixin 和 PlotsBuilderMixin 的类型，用于创建具有统计和绘图功能的访问器
class MetaGenericAccessor(type(StatsBuilderMixin), type(PlotsBuilderMixin)):
    """
    GenericAccessor 的元类
    
    这个元类结合了 StatsBuilderMixin 和 PlotsBuilderMixin 的类型，
    用于创建具有统计分析和绘图功能的通用访问器。
    """
    pass


# 类型变量定义
GenericAccessorT = tp.TypeVar("GenericAccessorT", bound="GenericAccessor")  # GenericAccessor 的类型变量
SplitOutputT = tp.Union[tp.MaybeTuple[tp.Tuple[tp.Frame, tp.Index]], tp.BaseFigure]  # 分割输出的类型


# 变换器协议定义
class TransformerT(tp.Protocol):
    """
    数据变换器的协议接口
    
    这个协议定义了数据变换器必须实现的方法，主要用于 sklearn 风格的数据预处理。
    任何符合此协议的变换器都可以在 GenericAccessor.transform 方法中使用。
    
    必须实现的方法：
    - __init__: 初始化变换器
    - transform: 对数据进行变换
    - fit_transform: 拟合并变换数据
    
    使用示例：
    ```python
    from sklearn.preprocessing import StandardScaler
    
    # StandardScaler 符合 TransformerT 协议
    scaler = StandardScaler()
    transformed_data = df.vbt.transform(scaler)
    ```
    """
    def __init__(self, **kwargs) -> None:
        """初始化变换器"""
        ...

    def transform(self, *args, **kwargs) -> tp.Array2d:
        """对数据进行变换"""
        ...

    def fit_transform(self, *args, **kwargs) -> tp.Array2d:
        """拟合并变换数据"""
        ...


# Numba 方法配置
# 这个配置定义了将要添加到 GenericAccessor 中的 numba 编译方法
nb_config = Config(
    {
        # 数据洗牌 - 随机打乱数据顺序
        'shuffle': dict(func=nb.shuffle_nb, path='vectorbt.generic.nb.shuffle_nb'),
        # 填充 NaN 值
        'fillna': dict(func=nb.fillna_nb, path='vectorbt.generic.nb.fillna_nb'),
        # 向后移位 - 将数据向后移动指定位数
        'bshift': dict(func=nb.bshift_nb, path='vectorbt.generic.nb.bshift_nb'),
        # 向前移位 - 将数据向前移动指定位数
        'fshift': dict(func=nb.fshift_nb, path='vectorbt.generic.nb.fshift_nb'),
        # 差分计算 - 计算相邻期间的差值
        'diff': dict(func=nb.diff_nb, path='vectorbt.generic.nb.diff_nb'),
        # 百分比变化 - 计算相邻期间的百分比变化
        'pct_change': dict(func=nb.pct_change_nb, path='vectorbt.generic.nb.pct_change_nb'),
        # 向后填充 - 用后面的非 NaN 值填充 NaN
        'bfill': dict(func=nb.bfill_nb, path='vectorbt.generic.nb.bfill_nb'),
        # 向前填充 - 用前面的非 NaN 值填充 NaN
        'ffill': dict(func=nb.ffill_nb, path='vectorbt.generic.nb.ffill_nb'),
        # 累积和 - 计算忽略 NaN 的累积和
        'cumsum': dict(func=nb.nancumsum_nb, path='vectorbt.generic.nb.nancumsum_nb'),
        # 累积积 - 计算忽略 NaN 的累积积
        'cumprod': dict(func=nb.nancumprod_nb, path='vectorbt.generic.nb.nancumprod_nb'),
        # 滚动最小值 - 计算滚动窗口内的最小值
        'rolling_min': dict(func=nb.rolling_min_nb, path='vectorbt.generic.nb.rolling_min_nb'),
        # 滚动最大值 - 计算滚动窗口内的最大值
        'rolling_max': dict(func=nb.rolling_max_nb, path='vectorbt.generic.nb.rolling_max_nb'),
        # 滚动均值 - 计算滚动窗口内的平均值
        'rolling_mean': dict(func=nb.rolling_mean_nb, path='vectorbt.generic.nb.rolling_mean_nb'),
        # 扩展最小值 - 计算扩展窗口内的最小值
        'expanding_min': dict(func=nb.expanding_min_nb, path='vectorbt.generic.nb.expanding_min_nb'),
        # 扩展最大值 - 计算扩展窗口内的最大值
        'expanding_max': dict(func=nb.expanding_max_nb, path='vectorbt.generic.nb.expanding_max_nb'),
        # 扩展均值 - 计算扩展窗口内的平均值
        'expanding_mean': dict(func=nb.expanding_mean_nb, path='vectorbt.generic.nb.expanding_mean_nb'),
        # 乘积 - 计算忽略 NaN 的乘积（归约操作）
        'product': dict(func=nb.nanprod_nb, is_reducing=True, path='vectorbt.generic.nb.nanprod_nb')
    },
    readonly=True,    # 只读配置
    as_attrs=False    # 不作为属性访问
)
"""Numba 方法配置的占位符"""

# 为 nb_config 添加文档字符串
__pdoc__['nb_config'] = f"""要添加到 `GenericAccessor` 的 Numba 方法配置。

这个配置定义了所有将通过装饰器自动添加到 GenericAccessor 类中的 numba 编译方法。
这些方法提供了高性能的数值计算功能，特别适合处理大规模时间序列数据。

配置内容：
```json
{nb_config.to_doc()}
```
"""

# 数据变换方法配置
# 这个配置定义了将要添加到 GenericAccessor 中的 sklearn 风格数据变换方法
transform_config = Config(
    {
        # 二值化 - 将数据转换为二进制形式
        'binarize': dict(
            transformer=Binarizer,
            docstring="参见 `sklearn.preprocessing.Binarizer`。"
        ),
        # 最小-最大缩放 - 将数据缩放到指定范围
        'minmax_scale': dict(
            transformer=MinMaxScaler,
            docstring="参见 `sklearn.preprocessing.MinMaxScaler`。"
        ),
        # 最大绝对值缩放 - 按最大绝对值缩放数据
        'maxabs_scale': dict(
            transformer=MaxAbsScaler,
            docstring="参见 `sklearn.preprocessing.MaxAbsScaler`。"
        ),
        # 标准化 - 将数据标准化为单位长度
        'normalize': dict(
            transformer=Normalizer,
            docstring="参见 `sklearn.preprocessing.Normalizer`。"
        ),
        # 鲁棒缩放 - 使用中位数和四分位数进行缩放
        'robust_scale': dict(
            transformer=RobustScaler,
            docstring="参见 `sklearn.preprocessing.RobustScaler`。"
        ),
        # 标准缩放 - 标准化为零均值单位方差
        'scale': dict(
            transformer=StandardScaler,
            docstring="参见 `sklearn.preprocessing.StandardScaler`。"
        ),
        # 分位数变换 - 将数据变换为均匀或正态分布
        'quantile_transform': dict(
            transformer=QuantileTransformer,
            docstring="参见 `sklearn.preprocessing.QuantileTransformer`。"
        ),
        # 幂变换 - 应用幂变换使数据更接近正态分布
        'power_transform': dict(
            transformer=PowerTransformer,
            docstring="参见 `sklearn.preprocessing.PowerTransformer`。"
        )
    },
    readonly=True,    # 只读配置
    as_attrs=False    # 不作为属性访问
)
"""数据变换方法配置的占位符"""

# 为 transform_config 添加文档字符串
__pdoc__['transform_config'] = f"""要添加到 `GenericAccessor` 的变换方法配置。

这个配置定义了所有将通过装饰器自动添加到 GenericAccessor 类中的数据变换方法。
这些方法基于 sklearn 的预处理器，提供了标准化、缩放、变换等数据预处理功能。

配置内容：
```json
{transform_config.to_doc()}
```
"""


# 使用装饰器自动添加 numba 方法和变换方法到 GenericAccessor 类
@attach_nb_methods(nb_config)          # 添加 numba 编译的高性能计算方法
@attach_transform_methods(transform_config)  # 添加 sklearn 风格的数据变换方法
class GenericAccessor(BaseAccessor, StatsBuilderMixin, PlotsBuilderMixin, metaclass=MetaGenericAccessor):
    """
    通用数据访问器 - 适用于任何类型的数据，同时支持 Series 和 DataFrame
    
    这个类是 vectorbt 框架的核心访问器，为 pandas 的 Series 和 DataFrame 对象提供了
    强大的量化分析功能。它继承了多个混入类，集成了统计分析、绘图、基础数据操作等功能。
    
    核心特性：
    1. **高性能计算**：使用 numba 编译的函数进行快速数值计算
    2. **统计分析**：内置丰富的统计指标和分析功能
    3. **数据可视化**：支持多种图表类型和交互式绘图
    4. **数据变换**：提供 sklearn 风格的数据预处理功能
    5. **时间序列分析**：专门针对金融时间序列的分析工具
    6. **灵活映射**：支持自定义数据映射和标签转换
    
    继承关系：
    - BaseAccessor: 提供基础的访问器功能
    - StatsBuilderMixin: 提供统计分析功能
    - PlotsBuilderMixin: 提供绘图功能
    
    访问方式：
    - 通过 `pd.Series.vbt` 访问 Series 的功能
    - 通过 `pd.DataFrame.vbt` 访问 DataFrame 的功能
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    import numpy as np
    
    # 创建示例数据
    df = pd.DataFrame({
        'price': [100, 105, 98, 95, 102, 108, 103],
        'volume': [1000, 1200, 800, 600, 1100, 1500, 900]
    })
    
    # 基本统计分析
    stats = df.vbt.describe()
    print(stats)
    
    # 滚动窗口计算
    rolling_mean = df['price'].vbt.rolling_mean(window=3)
    print(rolling_mean)
    
    # 数据变换
    standardized = df.vbt.scale()  # 标准化
    normalized = df.vbt.minmax_scale()  # 最小-最大缩放
    
    # 绘图
    fig = df.vbt.plot()
    fig.show()
    
    # 回撤分析
    drawdowns = df['price'].vbt.drawdowns
    print(drawdowns.stats())
    
    # 自定义映射
    mapping = {100: 'low', 105: 'high', 98: 'medium'}
    mapped_accessor = df.vbt(mapping=mapping)
    value_counts = mapped_accessor.value_counts()
    ```
    
    注意事项：
    - 访问器不使用缓存，每次调用都会重新计算
    - 分组功能仅在支持 `group_by` 参数的方法中可用
    - 大多数计算密集型操作都经过 numba 优化
    """

    def __init__(self, obj: tp.SeriesFrame, mapping: tp.Optional[tp.MappingLike] = None, **kwargs) -> None:
        """
        初始化 GenericAccessor 实例
        
        参数：
            obj (tp.SeriesFrame): 要包装的 pandas Series 或 DataFrame 对象
            mapping (tp.Optional[tp.MappingLike], 可选): 数据映射配置
                可以是：
                - 字典：{原值: 新值} 的映射
                - 字符串：'index' 或 'columns'，使用索引或列名作为映射
                - 其他可映射对象
            **kwargs: 传递给基类的额外参数
        
        映射功能：
        - 可以将数据值映射为其他值（如数字映射为标签）
        - 支持在统计分析中使用映射值
        - 提供更好的数据可读性和分析结果
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        
        # 创建示例数据
        df = pd.DataFrame({'score': [1, 2, 3, 2, 1]})
        
        # 使用字典映射
        mapping = {1: 'low', 2: 'medium', 3: 'high'}
        accessor = df.vbt(mapping=mapping)
        value_counts = accessor.value_counts()
        
        # 使用索引映射
        accessor_idx = df.vbt(mapping='index')
        
        # 使用列名映射
        accessor_col = df.vbt(mapping='columns')
        ```
        """
        # 初始化基类
        BaseAccessor.__init__(self, obj, mapping=mapping, **kwargs)  # 初始化基础访问器
        StatsBuilderMixin.__init__(self)                             # 初始化统计构建器
        PlotsBuilderMixin.__init__(self)                             # 初始化绘图构建器

        # 处理映射参数
        if mapping is not None:
            # 如果映射是字符串，则使用特殊处理
            if isinstance(mapping, str):
                if mapping.lower() == 'index':
                    mapping = self.wrapper.index      # 使用索引作为映射
                elif mapping.lower() == 'columns':
                    mapping = self.wrapper.columns    # 使用列名作为映射
            # 将映射转换为标准格式
            mapping = to_mapping(mapping)
        
        # 存储映射配置
        self._mapping = mapping

    @property
    def sr_accessor_cls(self) -> tp.Type["GenericSRAccessor"]:
        """
        Series 访问器类属性
        
        返回：
            tp.Type["GenericSRAccessor"]: 用于 pd.Series 的访问器类
        
        这个属性定义了当数据为 Series 时应该使用的访问器类型。
        """
        return GenericSRAccessor

    @property
    def df_accessor_cls(self) -> tp.Type["GenericDFAccessor"]:
        """
        DataFrame 访问器类属性
        
        返回：
            tp.Type["GenericDFAccessor"]: 用于 pd.DataFrame 的访问器类
        
        这个属性定义了当数据为 DataFrame 时应该使用的访问器类型。
        """
        return GenericDFAccessor

    @property
    def mapping(self) -> tp.Optional[tp.Mapping]:
        """
        映射配置属性
        
        返回：
            tp.Optional[tp.Mapping]: 当前的数据映射配置，如果没有设置则为 None
        
        这个属性提供对当前映射配置的访问，映射用于在统计分析和可视化中
        将原始数据值转换为更有意义的标签。
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        
        # 创建带映射的访问器
        df = pd.DataFrame({'grade': [1, 2, 3, 2, 1]})
        mapping = {1: 'F', 2: 'C', 3: 'A'}
        accessor = df.vbt(mapping=mapping)
        
        # 访问映射配置
        print(accessor.mapping)  # {1: 'F', 2: 'C', 3: 'A'}
        
        # 在统计分析中使用映射
        stats = accessor.value_counts()
        print(stats)  # 显示 F, C, A 的计数而不是 1, 2, 3
        ```
        """
        return self._mapping

    def apply_mapping(self, **kwargs) -> tp.SeriesFrame:
        """
        应用映射到数据 - 使用配置的映射转换数据值
        
        参数：
            **kwargs: 传递给 vectorbt.utils.mapping.apply_mapping 的参数
        
        返回：
            tp.SeriesFrame: 应用映射后的数据
        
        这个方法将当前配置的映射应用到数据上，将原始值转换为映射值。
        如果没有配置映射，则返回原始数据。
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        
        # 创建示例数据
        df = pd.DataFrame({'score': [1, 2, 3, 2, 1]})
        
        # 设置映射
        mapping = {1: 'low', 2: 'medium', 3: 'high'}
        accessor = df.vbt(mapping=mapping)
        
        # 应用映射
        mapped_data = accessor.apply_mapping()
        print(mapped_data)
        # 输出：
        #     score
        # 0     low
        # 1  medium
        # 2    high
        # 3  medium
        # 4     low
        
        # 使用额外参数
        mapped_data_custom = accessor.apply_mapping(na_value='unknown')
        ```
        
        参见：
            vectorbt.utils.mapping.apply_mapping: 底层映射应用函数
        """
        return apply_mapping(self.obj, self.mapping, **kwargs)

    def rolling_std(self, window: int, minp: tp.Optional[int] = None, ddof: int = 1,
                    wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:  # pragma: no cover
        """
        计算滚动窗口标准差 - 计算指定窗口大小的滚动标准差
        
        参数：
            window (int): 滚动窗口大小
            minp (tp.Optional[int], 可选): 窗口内所需的最小观察值数量，默认为 None
            ddof (int, 可选): 自由度增量，默认为 1
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.SeriesFrame: 滚动标准差序列
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103, 99, 101, 104]
        })
        
        # 计算3期滚动标准差
        rolling_std = df['price'].vbt.rolling_std(window=3)
        print(rolling_std)
        
        # 设置最小观察值数量
        rolling_std_minp = df['price'].vbt.rolling_std(window=5, minp=3)
        print(rolling_std_minp)
        ```
        
        参见：
            vectorbt.generic.nb.rolling_std_nb: 底层 numba 实现
        """
        # 调用 numba 编译的滚动标准差函数
        out = nb.rolling_std_nb(self.to_2d_array(), window, minp=minp, ddof=ddof)
        # 包装结果并返回
        return self.wrapper.wrap(out, group_by=False, **merge_dicts({}, wrap_kwargs))

    def expanding_std(self, minp: tp.Optional[int] = 1, ddof: int = 1,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:  # pragma: no cover
        """
        计算扩展窗口标准差 - 计算从开始到当前位置的扩展标准差
        
        参数：
            minp (tp.Optional[int], 可选): 所需的最小观察值数量，默认为 1
            ddof (int, 可选): 自由度增量，默认为 1
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.SeriesFrame: 扩展标准差序列
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103, 99, 101, 104]
        })
        
        # 计算扩展标准差
        expanding_std = df['price'].vbt.expanding_std()
        print(expanding_std)
        
        # 设置最小观察值数量
        expanding_std_minp = df['price'].vbt.expanding_std(minp=3)
        print(expanding_std_minp)
        ```
        
        参见：
            vectorbt.generic.nb.expanding_std_nb: 底层 numba 实现
        """
        # 调用 numba 编译的扩展标准差函数
        out = nb.expanding_std_nb(self.to_2d_array(), minp=minp, ddof=ddof)
        # 包装结果并返回
        return self.wrapper.wrap(out, group_by=False, **merge_dicts({}, wrap_kwargs))

    def ewm_mean(self, span: int, minp: tp.Optional[int] = 0, adjust: bool = True,
                 wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:  # pragma: no cover
        """
        计算指数加权移动平均 (EMA) - 计算指数加权移动平均值
        
        参数：
            span (int): 衰减期数，用于计算平滑因子 α = 2/(span+1)
            minp (tp.Optional[int], 可选): 所需的最小观察值数量，默认为 0
            adjust (bool, 可选): 是否使用调整算法，默认为 True
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.SeriesFrame: 指数加权移动平均序列
        
        算法说明：
        - 平滑因子：α = 2/(span+1)
        - 如果 adjust=True：使用偏差调整的 EMA 算法
        - 如果 adjust=False：使用标准递归算法
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103, 99, 101, 104]
        })
        
        # 计算12期EMA
        ema_12 = df['price'].vbt.ewm_mean(span=12)
        print(ema_12)
        
        # 不使用调整算法
        ema_no_adjust = df['price'].vbt.ewm_mean(span=12, adjust=False)
        print(ema_no_adjust)
        
        # 设置最小观察值数量
        ema_minp = df['price'].vbt.ewm_mean(span=12, minp=5)
        print(ema_minp)
        ```
        
        参见：
            vectorbt.generic.nb.ewm_mean_nb: 底层 numba 实现
        """
        # 调用 numba 编译的指数加权移动平均函数
        out = nb.ewm_mean_nb(self.to_2d_array(), span, minp=minp, adjust=adjust)
        # 包装结果并返回
        return self.wrapper.wrap(out, group_by=False, **merge_dicts({}, wrap_kwargs))

    def ewm_std(self, span: int, minp: tp.Optional[int] = 0, adjust: bool = True, ddof: int = 1,
                wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:  # pragma: no cover
        """
        计算指数加权移动标准差 - 计算指数加权移动标准差
        
        参数：
            span (int): 衰减期数，用于计算平滑因子 α = 2/(span+1)
            minp (tp.Optional[int], 可选): 所需的最小观察值数量，默认为 0
            adjust (bool, 可选): 是否使用调整算法，默认为 True
            ddof (int, 可选): 自由度增量，默认为 1
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.SeriesFrame: 指数加权移动标准差序列
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103, 99, 101, 104]
        })
        
        # 计算12期EMA标准差
        ema_std_12 = df['price'].vbt.ewm_std(span=12)
        print(ema_std_12)
        
        # 计算波动率指标
        volatility = df['price'].vbt.pct_change().vbt.ewm_std(span=20)
        print(volatility)
        
        # 不使用调整算法
        ema_std_no_adjust = df['price'].vbt.ewm_std(span=12, adjust=False)
        print(ema_std_no_adjust)
        ```
        
        参见：
            vectorbt.generic.nb.ewm_std_nb: 底层 numba 实现
        """
        # 调用 numba 编译的指数加权移动标准差函数
        out = nb.ewm_std_nb(self.to_2d_array(), span, minp=minp, adjust=adjust, ddof=ddof)
        # 包装结果并返回
        return self.wrapper.wrap(out, group_by=False, **merge_dicts({}, wrap_kwargs))

    def apply_along_axis(self, apply_func_nb: tp.Union[tp.ApplyFunc, tp.RowApplyFunc], *args, axis: int = 0,
                         wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        沿轴应用函数 - 沿指定轴应用 numba 编译的函数
        
        参数：
            apply_func_nb (tp.Union[tp.ApplyFunc, tp.RowApplyFunc]): numba 编译的应用函数
                - 对于 axis=0: 函数签名为 (col, arr) -> scalar
                - 对于 axis=1: 函数签名为 (row, arr) -> scalar
            *args: 传递给应用函数的额外参数
            axis (int, 可选): 应用函数的轴向，默认为 0
                - 0: 沿列应用（每列计算一个结果）
                - 1: 沿行应用（每行计算一个结果）
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.SeriesFrame: 应用函数后的结果
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        from numba import njit
        
        # 创建示例数据
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10],
            'C': [1, 1, 2, 3, 5]
        })
        
        # 示例1：沿列应用函数（axis=0）
        @njit
        def col_sum_nb(col, arr):
            return np.sum(arr)
        
        # 计算每列的和
        col_sums = df.vbt.apply_along_axis(col_sum_nb, axis=0)
        print("列和:", col_sums)
        
        # 示例2：沿行应用函数（axis=1）
        @njit
        def row_max_nb(row, arr):
            return np.max(arr)
        
        # 计算每行的最大值
        row_maxs = df.vbt.apply_along_axis(row_max_nb, axis=1)
        print("行最大值:", row_maxs)
        
        # 示例3：带参数的函数
        @njit
        def weighted_mean_nb(col, arr, weights):
            return np.average(arr, weights=weights)
        
        weights = np.array([0.5, 0.3, 0.2])
        weighted_means = df.vbt.apply_along_axis(weighted_mean_nb, weights, axis=0)
        print("加权平均:", weighted_means)
        
        # 示例4：金融应用 - 计算夏普比率
        returns = df.pct_change().dropna()
        
        @njit
        def sharpe_ratio_nb(col, arr, risk_free_rate=0.02):
            mean_return = np.mean(arr)
            std_return = np.std(arr)
            return (mean_return - risk_free_rate) * 252 / (std_return * np.sqrt(252))
        
        sharpe_ratios = returns.vbt.apply_along_axis(sharpe_ratio_nb, 0.02, axis=0)
        print("夏普比率:", sharpe_ratios)
        ```
        
        注意：
        - 函数必须是 numba 编译的 (@njit 装饰器)
        - 函数的第一个参数是当前的列/行索引
        - 函数的第二个参数是当前的数组切片
        - 只支持 axis=0 和 axis=1
        
        参见：
            vectorbt.generic.nb.apply_nb: 沿列应用的底层实现
            vectorbt.generic.nb.row_apply_nb: 沿行应用的底层实现
        """
        # 检查函数是否为 numba 编译的函数
        checks.assert_numba_func(apply_func_nb)

        if axis == 0:
            # 沿列应用函数
            out = nb.apply_nb(self.to_2d_array(), apply_func_nb, *args)
        elif axis == 1:
            # 沿行应用函数
            out = nb.row_apply_nb(self.to_2d_array(), apply_func_nb, *args)
        else:
            raise ValueError("Only axes 0 and 1 are supported")
        return self.wrapper.wrap(out, group_by=False, **merge_dicts({}, wrap_kwargs))

    def rolling_apply(self, window: int, apply_func_nb: tp.Union[tp.RollApplyFunc, nb.tp.RollMatrixApplyFunc],
                      *args, minp: tp.Optional[int] = None, on_matrix: bool = False,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        滚动窗口应用函数 - 在滚动窗口上应用 numba 编译的函数
        
        参数：
            window (int): 滚动窗口大小
            apply_func_nb (tp.Union[tp.RollApplyFunc, nb.tp.RollMatrixApplyFunc]): numba 编译的应用函数
                - 对于 on_matrix=False: 函数签名为 (i, col, arr) -> scalar
                - 对于 on_matrix=True: 函数签名为 (i, matrix) -> scalar
            *args: 传递给应用函数的额外参数
            minp (tp.Optional[int], 可选): 窗口内所需的最小观察值数量
            on_matrix (bool, 可选): 是否在整个矩阵上应用函数，默认为 False
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.SeriesFrame: 滚动应用函数后的结果
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        from numba import njit
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103, 99, 101, 104],
            'volume': [1000, 1200, 800, 600, 1100, 1300, 900, 700, 1000, 1150]
        })
        
        # 示例1：滚动窗口均值（按列）
        @njit
        def rolling_mean_nb(i, col, arr):
            return np.nanmean(arr)
        
        rolling_means = df.vbt.rolling_apply(3, rolling_mean_nb)
        print("3期滚动均值:")
        print(rolling_means)
        
        # 示例2：滚动窗口标准差
        @njit
        def rolling_std_nb(i, col, arr):
            return np.nanstd(arr)
        
        rolling_stds = df.vbt.rolling_apply(5, rolling_std_nb, minp=3)
        print("5期滚动标准差（最少3个观察值）:")
        print(rolling_stds)
        
        # 示例3：滚动窗口最大值
        @njit
        def rolling_max_nb(i, col, arr):
            return np.nanmax(arr)
        
        rolling_maxs = df.vbt.rolling_apply(4, rolling_max_nb)
        print("4期滚动最大值:")
        print(rolling_maxs)
        
        # 示例4：基于矩阵的滚动应用（on_matrix=True）
        @njit
        def rolling_corr_nb(i, matrix):
            # 计算两列之间的相关性
            if matrix.shape[1] >= 2:
                return np.corrcoef(matrix[:, 0], matrix[:, 1])[0, 1]
            return np.nan
        
        rolling_corr = df.vbt.rolling_apply(5, rolling_corr_nb, on_matrix=True)
        print("5期滚动相关性:")
        print(rolling_corr)
        
        # 示例5：金融应用 - 滚动波动率
        returns = df['price'].pct_change().dropna()
        
        @njit
        def rolling_volatility_nb(i, col, arr):
            return np.nanstd(arr) * np.sqrt(252)  # 年化波动率
        
        rolling_vol = returns.vbt.rolling_apply(20, rolling_volatility_nb)
        print("20期滚动年化波动率:")
        print(rolling_vol)
        
        # 示例6：滚动偏度
        @njit
        def rolling_skew_nb(i, col, arr):
            # 简化的偏度计算
            mean = np.nanmean(arr)
            std = np.nanstd(arr)
            if std == 0:
                return np.nan
            return np.nanmean(((arr - mean) / std) ** 3)
        
        rolling_skewness = df['price'].vbt.rolling_apply(10, rolling_skew_nb)
        print("10期滚动偏度:")
        print(rolling_skewness)
        ```
        
        注意：
        - 函数必须是 numba 编译的 (@njit 装饰器)
        - 函数的第一个参数是当前索引位置
        - 函数的第二个参数是列索引（on_matrix=False）或整个矩阵（on_matrix=True）
        - 函数的第三个参数是当前窗口的数组切片
        - 窗口大小必须 >= 1
        - 可以设置 minp 来处理窗口内数据不足的情况
        
        参见：
            vectorbt.generic.nb.rolling_apply_nb: 按列滚动应用的底层实现
            vectorbt.generic.nb.rolling_matrix_apply_nb: 基于矩阵滚动应用的底层实现
        """
        # 检查函数是否为 numba 编译的函数
        checks.assert_numba_func(apply_func_nb)

        if on_matrix:
            # 基于矩阵的滚动应用
            out = nb.rolling_matrix_apply_nb(self.to_2d_array(), window, minp, apply_func_nb, *args)
        else:
            # 按列的滚动应用
            out = nb.rolling_apply_nb(self.to_2d_array(), window, minp, apply_func_nb, *args)
        return self.wrapper.wrap(out, group_by=False, **merge_dicts({}, wrap_kwargs))

    def expanding_apply(self, apply_func_nb: tp.Union[tp.RollApplyFunc, nb.tp.RollMatrixApplyFunc],
                        *args, minp: tp.Optional[int] = 1, on_matrix: bool = False,
                        wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        扩展窗口应用函数 - 在扩展窗口上应用 numba 编译的函数
        
        参数：
            apply_func_nb (tp.Union[tp.RollApplyFunc, nb.tp.RollMatrixApplyFunc]): numba 编译的应用函数
                - 对于 on_matrix=False: 函数签名为 (i, col, arr) -> scalar
                - 对于 on_matrix=True: 函数签名为 (i, matrix) -> scalar
            *args: 传递给应用函数的额外参数
            minp (tp.Optional[int], 可选): 窗口内所需的最小观察值数量，默认为 1
            on_matrix (bool, 可选): 是否在整个矩阵上应用函数，默认为 False
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.SeriesFrame: 扩展应用函数后的结果
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        from numba import njit
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103, 99, 101, 104],
            'volume': [1000, 1200, 800, 600, 1100, 1300, 900, 700, 1000, 1150]
        })
        
        # 示例1：扩展窗口均值（按列）
        @njit
        def expanding_mean_nb(i, col, arr):
            return np.nanmean(arr)
        
        expanding_means = df.vbt.expanding_apply(expanding_mean_nb)
        print("扩展均值:")
        print(expanding_means)
        
        # 示例2：扩展窗口标准差
        @njit
        def expanding_std_nb(i, col, arr):
            return np.nanstd(arr)
        
        expanding_stds = df.vbt.expanding_apply(expanding_std_nb, minp=3)
        print("扩展标准差（最少3个观察值）:")
        print(expanding_stds)
        
        # 示例3：扩展窗口最大值
        @njit
        def expanding_max_nb(i, col, arr):
            return np.nanmax(arr)
        
        expanding_maxs = df.vbt.expanding_apply(expanding_max_nb)
        print("扩展最大值:")
        print(expanding_maxs)
        
        # 示例4：基于矩阵的扩展应用（on_matrix=True）
        @njit
        def expanding_corr_nb(i, matrix):
            # 计算两列之间的相关性
            if matrix.shape[1] >= 2 and matrix.shape[0] >= 2:
                return np.corrcoef(matrix[:, 0], matrix[:, 1])[0, 1]
            return np.nan
        
        expanding_corr = df.vbt.expanding_apply(expanding_corr_nb, on_matrix=True)
        print("扩展相关性:")
        print(expanding_corr)
        
        # 示例5：金融应用 - 扩展波动率
        returns = df['price'].pct_change().dropna()
        
        @njit
        def expanding_volatility_nb(i, col, arr):
            return np.nanstd(arr) * np.sqrt(252)  # 年化波动率
        
        expanding_vol = returns.vbt.expanding_apply(expanding_volatility_nb)
        print("扩展年化波动率:")
        print(expanding_vol)
        
        # 示例6：扩展夏普比率
        @njit
        def expanding_sharpe_nb(i, col, arr, risk_free_rate=0.02):
            mean_return = np.nanmean(arr)
            std_return = np.nanstd(arr)
            if std_return == 0:
                return np.nan
            return (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)
        
        expanding_sharpe = returns.vbt.expanding_apply(expanding_sharpe_nb, 0.02)
        print("扩展夏普比率:")
        print(expanding_sharpe)
        
        # 示例7：扩展回撤
        @njit
        def expanding_drawdown_nb(i, col, arr):
            if len(arr) == 0:
                return np.nan
            peak = np.nanmax(arr)
            current = arr[-1]
            return (current - peak) / peak
        
        expanding_dd = df['price'].vbt.expanding_apply(expanding_drawdown_nb)
        print("扩展回撤:")
        print(expanding_dd)
        ```
        
        注意：
        - 函数必须是 numba 编译的 (@njit 装饰器)
        - 函数的第一个参数是当前索引位置
        - 函数的第二个参数是列索引（on_matrix=False）或整个矩阵（on_matrix=True）
        - 函数的第三个参数是从开始到当前位置的数组切片
        - 扩展窗口大小从 1 开始逐步增长到当前位置
        - 可以设置 minp 来处理窗口内数据不足的情况
        
        参见：
            vectorbt.generic.nb.expanding_apply_nb: 按列扩展应用的底层实现
            vectorbt.generic.nb.expanding_matrix_apply_nb: 基于矩阵扩展应用的底层实现
        """
        # 检查函数是否为 numba 编译的函数
        checks.assert_numba_func(apply_func_nb)

        if on_matrix:
            # 基于矩阵的扩展应用
            out = nb.expanding_matrix_apply_nb(self.to_2d_array(), minp, apply_func_nb, *args)
        else:
            # 按列的扩展应用
            out = nb.expanding_apply_nb(self.to_2d_array(), minp, apply_func_nb, *args)
        return self.wrapper.wrap(out, group_by=False, **merge_dicts({}, wrap_kwargs))

    def groupby_apply(self, by: tp.PandasGroupByLike,
                      apply_func_nb: tp.Union[tp.GroupByApplyFunc, tp.GroupByMatrixApplyFunc],
                      *args, on_matrix: bool = False, wrap_kwargs: tp.KwargsLike = None,
                      **kwargs) -> tp.SeriesFrame:
        """
        分组应用函数 - 按分组对数据应用 numba 编译的函数
        
        参数：
            by (tp.PandasGroupByLike): 分组键，同 pandas.DataFrame.groupby 的 by 参数
            apply_func_nb (tp.Union[tp.GroupByApplyFunc, tp.GroupByMatrixApplyFunc]): numba 编译的应用函数
                - 对于 on_matrix=False: 函数签名为 (i, col, arr) -> scalar
                - 对于 on_matrix=True: 函数签名为 (i, matrix) -> scalar
            *args: 传递给应用函数的额外参数
            on_matrix (bool, 可选): 是否在整个矩阵上应用函数，默认为 False
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
            **kwargs: 传递给 pandas.DataFrame.groupby 的参数
        
        返回：
            tp.SeriesFrame: 分组应用函数后的结果
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        from numba import njit
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103, 99, 101, 104],
            'volume': [1000, 1200, 800, 600, 1100, 1300, 900, 700, 1000, 1150],
            'category': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'A', 'A']
        })
        
        # 示例1：按类别分组计算均值
        @njit
        def group_mean_nb(i, col, arr):
            return np.nanmean(arr)
        
        group_means = df[['price', 'volume']].vbt.groupby_apply(df['category'], group_mean_nb)
        print("按类别分组均值:")
        print(group_means)
        
        # 示例2：按数字分组
        groups = [1, 1, 2, 2, 1, 1, 2, 2, 1, 1]
        group_sums = df[['price', 'volume']].vbt.groupby_apply(groups, 
                                                             lambda i, col, arr: np.nansum(arr))
        print("按数字分组求和:")
        print(group_sums)
        
        # 示例3：基于矩阵的分组应用（on_matrix=True）
        @njit
        def group_corr_nb(i, matrix):
            # 计算组内所有列之间的平均相关性
            if matrix.shape[1] >= 2 and matrix.shape[0] >= 2:
                corr_matrix = np.corrcoef(matrix.T)
                # 返回上三角矩阵元素的平均值
                n = matrix.shape[1]
                sum_corr = 0.0
                count = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if not np.isnan(corr_matrix[i, j]):
                            sum_corr += corr_matrix[i, j]
                            count += 1
                return sum_corr / count if count > 0 else np.nan
            return np.nan
        
        group_corr = df[['price', 'volume']].vbt.groupby_apply(df['category'], group_corr_nb, on_matrix=True)
        print("按类别分组相关性:")
        print(group_corr)
        
        # 示例4：金融应用 - 按月份分组计算收益率统计
        dates = pd.date_range('2023-01-01', periods=len(df), freq='3D')
        df.index = dates
        
        # 计算收益率
        returns = df['price'].pct_change().dropna()
        
        @njit
        def group_sharpe_nb(i, col, arr, risk_free_rate=0.02):
            mean_return = np.nanmean(arr)
            std_return = np.nanstd(arr)
            if std_return == 0:
                return np.nan
            return (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)
        
        monthly_sharpe = returns.vbt.groupby_apply(returns.index.month, group_sharpe_nb, 0.02)
        print("月度夏普比率:")
        print(monthly_sharpe)
        
        # 示例5：按周分组计算波动率
        @njit
        def group_volatility_nb(i, col, arr):
            return np.nanstd(arr) * np.sqrt(252)
        
        weekly_vol = returns.vbt.groupby_apply(returns.index.isocalendar().week, group_volatility_nb)
        print("周度波动率:")
        print(weekly_vol)
        ```
        
        注意：
        - 函数必须是 numba 编译的 (@njit 装饰器)
        - 函数的第一个参数是分组索引
        - 函数的第二个参数是列索引（on_matrix=False）或整个矩阵（on_matrix=True）
        - 函数的第三个参数是当前分组的数组切片
        - 支持所有 pandas.DataFrame.groupby 的参数
        
        参见：
            vectorbt.generic.nb.groupby_apply_nb: 按列分组应用的底层实现
            vectorbt.generic.nb.groupby_matrix_apply_nb: 基于矩阵分组应用的底层实现
            pandas.DataFrame.groupby: pandas 分组功能
        """
        # 检查函数是否为 numba 编译的函数
        checks.assert_numba_func(apply_func_nb)

        # 使用 pandas 的 groupby 功能创建分组
        regrouped = self.obj.groupby(by, axis=0, **kwargs)
        groups = Dict()
        for i, (k, v) in enumerate(regrouped.indices.items()):
            groups[i] = np.asarray(v)
        
        if on_matrix:
            # 基于矩阵的分组应用
            out = nb.groupby_matrix_apply_nb(self.to_2d_array(), groups, apply_func_nb, *args)
        else:
            # 按列的分组应用
            out = nb.groupby_apply_nb(self.to_2d_array(), groups, apply_func_nb, *args)
        
        # 设置结果的索引为分组键
        wrap_kwargs = merge_dicts(dict(name_or_index=list(regrouped.indices.keys())), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def resample_apply(self, freq: tp.PandasFrequencyLike,
                       apply_func_nb: tp.Union[tp.GroupByApplyFunc, tp.GroupByMatrixApplyFunc],
                       *args, on_matrix: bool = False, wrap_kwargs: tp.KwargsLike = None,
                       **kwargs) -> tp.SeriesFrame:
        """
        重采样应用函数 - 按时间频率重采样后应用 numba 编译的函数
        
        参数：
            freq (tp.PandasFrequencyLike): 重采样频率，同 pandas.DataFrame.resample 的 freq 参数
            apply_func_nb (tp.Union[tp.GroupByApplyFunc, tp.GroupByMatrixApplyFunc]): numba 编译的应用函数
                - 对于 on_matrix=False: 函数签名为 (i, col, arr) -> scalar
                - 对于 on_matrix=True: 函数签名为 (i, matrix) -> scalar
            *args: 传递给应用函数的额外参数
            on_matrix (bool, 可选): 是否在整个矩阵上应用函数，默认为 False
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
            **kwargs: 传递给 pandas.DataFrame.resample 的参数
        
        返回：
            tp.SeriesFrame: 重采样应用函数后的结果
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        from numba import njit
        
        # 创建示例数据（日度数据）
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'price': np.random.randn(30).cumsum() + 100,
            'volume': np.random.randint(1000, 2000, 30)
        }, index=dates)
        
        # 示例1：按周重采样计算均值
        @njit
        def resample_mean_nb(i, col, arr):
            return np.nanmean(arr)
        
        weekly_means = df.vbt.resample_apply('W', resample_mean_nb)
        print("周度均值:")
        print(weekly_means)
        
        # 示例2：按月重采样计算总和
        @njit
        def resample_sum_nb(i, col, arr):
            return np.nansum(arr)
        
        monthly_sums = df.vbt.resample_apply('M', resample_sum_nb)
        print("月度总和:")
        print(monthly_sums)
        
        # 示例3：基于矩阵的重采样应用（on_matrix=True）
        @njit
        def resample_corr_nb(i, matrix):
            # 计算时间段内的相关性
            if matrix.shape[1] >= 2 and matrix.shape[0] >= 2:
                return np.corrcoef(matrix[:, 0], matrix[:, 1])[0, 1]
            return np.nan
        
        weekly_corr = df.vbt.resample_apply('W', resample_corr_nb, on_matrix=True)
        print("周度相关性:")
        print(weekly_corr)
        
        # 示例4：金融应用 - 按月重采样计算收益率统计
        returns = df['price'].pct_change().dropna()
        
        @njit
        def monthly_sharpe_nb(i, col, arr, risk_free_rate=0.02):
            mean_return = np.nanmean(arr)
            std_return = np.nanstd(arr)
            if std_return == 0:
                return np.nan
            return (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)
        
        monthly_sharpe = returns.vbt.resample_apply('M', monthly_sharpe_nb, 0.02)
        print("月度夏普比率:")
        print(monthly_sharpe)
        
        # 示例5：按周重采样计算波动率
        @njit
        def weekly_volatility_nb(i, col, arr):
            return np.nanstd(arr) * np.sqrt(252)
        
        weekly_vol = returns.vbt.resample_apply('W', weekly_volatility_nb)
        print("周度波动率:")
        print(weekly_vol)
        
        # 示例6：按小时重采样计算 OHLC
        # 对于价格数据，计算开盘、最高、最低、收盘价
        hourly_ohlc = df['price'].to_frame().vbt.resample_apply('H', lambda i, col, arr: np.array([
            arr[0],    # Open
            np.nanmax(arr),   # High
            np.nanmin(arr),   # Low
            arr[-1]    # Close
        ]))
        print("小时 OHLC:")
        print(hourly_ohlc)
        ```
        
        注意：
        - 函数必须是 numba 编译的 (@njit 装饰器)
        - 函数的第一个参数是重采样组索引
        - 函数的第二个参数是列索引（on_matrix=False）或整个矩阵（on_matrix=True）
        - 函数的第三个参数是当前时间段的数组切片
        - 支持所有 pandas.DataFrame.resample 的参数
        - 结果会自动对齐到重采样频率的时间索引
        
        参见：
            vectorbt.generic.nb.groupby_apply_nb: 底层分组应用实现
            vectorbt.generic.nb.groupby_matrix_apply_nb: 基于矩阵的分组应用实现
            pandas.DataFrame.resample: pandas 重采样功能
        """
        # 检查函数是否为 numba 编译的函数
        checks.assert_numba_func(apply_func_nb)

        # 使用 pandas 的 resample 功能创建重采样分组
        resampled = self.obj.resample(freq, axis=0, **kwargs)
        groups = Dict()
        for i, (k, v) in enumerate(resampled.indices.items()):
            groups[i] = np.asarray(v)
        
        if on_matrix:
            # 基于矩阵的重采样应用
            out = nb.groupby_matrix_apply_nb(self.to_2d_array(), groups, apply_func_nb, *args)
        else:
            # 按列的重采样应用
            out = nb.groupby_apply_nb(self.to_2d_array(), groups, apply_func_nb, *args)
        
        # 创建输出对象，使用重采样的键作为索引
        out_obj = self.wrapper.wrap(out, group_by=False, index=list(resampled.indices.keys()))
        
        # 创建完整的重采样结果数组，填充 NaN
        resampled_arr = np.full((resampled.ngroups, self.to_2d_array().shape[1]), np.nan)
        resampled_obj = self.wrapper.wrap(
            resampled_arr,
            index=resampled.asfreq().index,  # 使用重采样频率的完整索引
            group_by=False,
            **merge_dicts({}, wrap_kwargs)
        )
        
        # 将计算结果填充到对应的位置
        resampled_obj.loc[out_obj.index] = out_obj.values
        return resampled_obj

    def applymap(self, apply_func_nb: tp.ApplyMapFunc, *args,
                 wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        元素级应用函数 - 对每个元素应用 numba 编译的函数
        
        参数：
            apply_func_nb (tp.ApplyMapFunc): numba 编译的应用函数
                函数签名为 (i, col, x) -> scalar
                其中 i 是行索引，col 是列索引，x 是元素值
            *args: 传递给应用函数的额外参数
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.SeriesFrame: 元素级应用函数后的结果
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        from numba import njit
        
        # 创建示例数据
        df = pd.DataFrame({
            'A': [1.5, 2.7, 3.2, 4.1, 5.8],
            'B': [-1.2, 0.5, 1.8, -0.3, 2.4],
            'C': [10.1, 20.5, 30.7, 40.2, 50.9]
        })
        
        # 示例1：对每个元素进行平方
        @njit
        def square_nb(i, col, x):
            return x * x
        
        squared_df = df.vbt.applymap(square_nb)
        print("平方结果:")
        print(squared_df)
        
        # 示例2：对每个元素应用 sigmoid 函数
        @njit
        def sigmoid_nb(i, col, x):
            return 1 / (1 + np.exp(-x))
        
        sigmoid_df = df.vbt.applymap(sigmoid_nb)
        print("Sigmoid 结果:")
        print(sigmoid_df)
        
        # 示例3：条件应用 - 根据位置和值进行不同处理
        @njit
        def conditional_nb(i, col, x):
            # 对不同列应用不同的转换
            if col == 0:  # 第一列：取绝对值
                return abs(x)
            elif col == 1:  # 第二列：平方根（对负数返回0）
                return np.sqrt(x) if x >= 0 else 0
            else:  # 其他列：对数变换
                return np.log(x) if x > 0 else np.nan
        
        conditional_df = df.vbt.applymap(conditional_nb)
        print("条件应用结果:")
        print(conditional_df)
        
        # 示例4：带参数的函数
        @njit
        def threshold_nb(i, col, x, threshold, replacement):
            return replacement if x < threshold else x
        
        threshold_df = df.vbt.applymap(threshold_nb, 2.0, 0.0)
        print("阈值处理结果:")
        print(threshold_df)
        
        # 示例5：金融应用 - 收益率转换
        prices = pd.DataFrame({
            'Stock_A': [100, 102, 98, 105, 103],
            'Stock_B': [50, 52, 48, 54, 51],
            'Stock_C': [200, 205, 195, 210, 208]
        })
        
        @njit
        def price_to_return_nb(i, col, x, prev_prices):
            if i == 0:
                return np.nan  # 第一行没有前一期价格
            prev_price = prev_prices[i-1, col]
            return (x - prev_price) / prev_price if prev_price != 0 else np.nan
        
        # 需要传递前一期价格数组
        prev_prices = prices.values
        returns = prices.vbt.applymap(price_to_return_nb, prev_prices)
        print("收益率:")
        print(returns)
        
        # 示例6：标准化处理
        @njit
        def normalize_nb(i, col, x, means, stds):
            mean = means[col]
            std = stds[col]
            return (x - mean) / std if std != 0 else 0
        
        # 计算每列的均值和标准差
        means = df.mean().values
        stds = df.std().values
        
        normalized_df = df.vbt.applymap(normalize_nb, means, stds)
        print("标准化结果:")
        print(normalized_df)
        
        # 示例7：分箱处理
        @njit
        def binning_nb(i, col, x, bins):
            # 将值分配到不同的箱子
            n_bins = len(bins) - 1
            for j in range(n_bins):
                if bins[j] <= x < bins[j+1]:
                    return j
            return n_bins - 1 if x >= bins[-1] else 0
        
        bins = np.array([0, 1, 2, 3, 4, 5, 100])
        binned_df = df.vbt.applymap(binning_nb, bins)
        print("分箱结果:")
        print(binned_df)
        ```
        
        注意：
        - 函数必须是 numba 编译的 (@njit 装饰器)
        - 函数的第一个参数是行索引
        - 函数的第二个参数是列索引
        - 函数的第三个参数是当前元素值
        - 函数必须返回标量值
        - 相比 pandas.DataFrame.applymap，此方法性能更高
        
        参见：
            vectorbt.generic.nb.applymap_nb: 底层元素级应用实现
            pandas.DataFrame.applymap: pandas 元素级应用功能
        """
        checks.assert_numba_func(apply_func_nb)

        out = nb.applymap_nb(self.to_2d_array(), apply_func_nb, *args)
        return self.wrapper.wrap(out, group_by=False, **merge_dicts({}, wrap_kwargs))

    def filter(self, filter_func_nb: tp.FilterFunc, *args,
               wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        过滤数据 - 使用 numba 编译的函数对数据进行过滤
        
        参数：
            filter_func_nb (tp.FilterFunc): numba 编译的过滤函数
                函数签名为 (i, col, x) -> bool
                其中 i 是行索引，col 是列索引，x 是元素值
                返回 True 表示保留该元素，False 表示过滤掉（设为 NaN）
            *args: 传递给过滤函数的额外参数
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.SeriesFrame: 过滤后的数据，不满足条件的元素被设为 NaN
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        from numba import njit
        
        # 创建示例数据
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [1, 3, 2, 4, 1]
        })
        
        # 示例1：过滤大于2的值
        @njit
        def greater_than_2_nb(i, col, x):
            return x > 2
        
        filtered_df = df.vbt.filter(greater_than_2_nb)
        print("过滤大于2的值:")
        print(filtered_df)
        
        # 示例2：根据位置过滤
        @njit
        def even_row_nb(i, col, x):
            return i % 2 == 0  # 只保留偶数行
        
        even_rows = df.vbt.filter(even_row_nb)
        print("偶数行过滤:")
        print(even_rows)
        
        # 示例3：根据列进行不同的过滤
        @njit
        def column_specific_nb(i, col, x):
            if col == 0:  # 第一列：只保留大于2的值
                return x > 2
            elif col == 1:  # 第二列：只保留小于4的值
                return x < 4
            else:  # 其他列：保留所有值
                return True
        
        column_filtered = df.vbt.filter(column_specific_nb)
        print("按列过滤:")
        print(column_filtered)
        
        # 示例4：带参数的过滤
        @njit
        def threshold_filter_nb(i, col, x, lower, upper):
            return lower <= x <= upper
        
        range_filtered = df.vbt.filter(threshold_filter_nb, 2, 4)
        print("范围过滤 [2, 4]:")
        print(range_filtered)
        
        # 示例5：金融应用 - 过滤异常值
        prices = pd.DataFrame({
            'Stock_A': [100, 102, 98, 95, 150, 103, 101],  # 150 是异常值
            'Stock_B': [50, 52, 48, 54, 51, 49, 53],
            'Stock_C': [200, 205, 195, 210, 208, 203, 207]
        })
        
        @njit
        def outlier_filter_nb(i, col, x, prices_array, threshold=0.1):
            if i == 0:
                return True  # 第一行没有前一期价格
            prev_price = prices_array[i-1, col]
            return abs(x - prev_price) / prev_price <= threshold
        
        # 过滤日变动超过10%的异常值
        outlier_filtered = prices.vbt.filter(outlier_filter_nb, prices.values, 0.1)
        print("异常值过滤:")
        print(outlier_filtered)
        
        # 示例6：波动率过滤
        returns = prices.pct_change().dropna()
        
        @njit
        def volatility_filter_nb(i, col, x, volatility_threshold=0.05):
            return abs(x) <= volatility_threshold
        
        low_vol_returns = returns.vbt.filter(volatility_filter_nb, 0.05)
        print("低波动率过滤:")
        print(low_vol_returns)
        
        # 示例7：条件组合过滤
        @njit
        def complex_filter_nb(i, col, x, means, stds, z_threshold=2.0):
            # 过滤 Z-score 大于阈值的值
            mean = means[col]
            std = stds[col]
            z_score = abs(x - mean) / std if std != 0 else 0
            return z_score <= z_threshold
        
        means = df.mean().values
        stds = df.std().values
        z_filtered = df.vbt.filter(complex_filter_nb, means, stds, 2.0)
        print("Z-score 过滤:")
        print(z_filtered)
        ```
        
        注意：
        - 函数必须是 numba 编译的 (@njit 装饰器)
        - 函数的第一个参数是行索引
        - 函数的第二个参数是列索引
        - 函数的第三个参数是当前元素值
        - 函数必须返回布尔值
        - 不满足条件的元素会被设为 NaN
        - 相比 pandas 的过滤操作，此方法性能更高
        
        参见：
            vectorbt.generic.nb.filter_nb: 底层过滤实现
            pandas.DataFrame.where: pandas 条件过滤
            pandas.DataFrame.mask: pandas 掩码过滤
        """
        # 检查函数是否为 numba 编译的函数
        checks.assert_numba_func(filter_func_nb)

        # 调用 numba 编译的过滤函数
        out = nb.filter_nb(self.to_2d_array(), filter_func_nb, *args)
        return self.wrapper.wrap(out, group_by=False, **merge_dicts({}, wrap_kwargs))

    def apply_and_reduce(self, apply_func_nb: tp.ApplyFunc, reduce_func_nb: tp.ReduceFunc,
                         apply_args: tp.Optional[tuple] = None, reduce_args: tp.Optional[tuple] = None,
                         wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        应用和归约操作 - 先应用变换函数，再应用归约函数
        
        参数：
            apply_func_nb (tp.ApplyFunc): numba 编译的应用函数
                函数签名为 (col, arr) -> new_arr
                其中 col 是列索引，arr 是列数组
                返回变换后的数组
            reduce_func_nb (tp.ReduceFunc): numba 编译的归约函数
                函数签名为 (col, arr) -> scalar
                其中 col 是列索引，arr 是应用函数的输出数组
                返回归约后的标量值
            apply_args (tp.Optional[tuple], 可选): 传递给应用函数的参数
            reduce_args (tp.Optional[tuple], 可选): 传递给归约函数的参数
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.MaybeSeries: 应用和归约后的结果
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        from numba import njit
        
        # 创建示例数据
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [1, 3, 2, 4, 1]
        })
        
        # 示例1：过滤大于2的值，然后计算均值
        @njit
        def filter_greater_2_nb(col, arr):
            return arr[arr > 2]
        
        @njit
        def mean_reduce_nb(col, arr):
            return np.nanmean(arr)
        
        result = df.vbt.apply_and_reduce(filter_greater_2_nb, mean_reduce_nb)
        print("过滤大于2的值后的均值:")
        print(result)
        
        # 示例2：计算每列的标准化值，然后求和
        @njit
        def standardize_nb(col, arr):
            mean = np.nanmean(arr)
            std = np.nanstd(arr)
            return (arr - mean) / std if std != 0 else arr
        
        @njit
        def sum_reduce_nb(col, arr):
            return np.nansum(arr)
        
        standardized_sum = df.vbt.apply_and_reduce(standardize_nb, sum_reduce_nb)
        print("标准化后的和:")
        print(standardized_sum)
        
        # 示例3：计算移动平均，然后找最大值
        @njit
        def moving_average_nb(col, arr, window):
            result = np.empty_like(arr)
            for i in range(len(arr)):
                start = max(0, i - window + 1)
                result[i] = np.nanmean(arr[start:i+1])
            return result
        
        @njit
        def max_reduce_nb(col, arr):
            return np.nanmax(arr)
        
        ma_max = df.vbt.apply_and_reduce(
            moving_average_nb, 
            max_reduce_nb,
            apply_args=(3,)  # 窗口大小为3
        )
        print("3期移动平均的最大值:")
        print(ma_max)
        
        # 示例4：金融应用 - 计算收益率，然后计算夏普比率
        prices = pd.DataFrame({
            'Stock_A': [100, 102, 98, 105, 103, 108, 106],
            'Stock_B': [50, 52, 48, 54, 51, 55, 53],
            'Stock_C': [200, 205, 195, 210, 208, 215, 212]
        })
        
        @njit
        def returns_nb(col, arr):
            # 计算收益率
            returns = np.empty(len(arr) - 1)
            for i in range(1, len(arr)):
                returns[i-1] = (arr[i] - arr[i-1]) / arr[i-1]
            return returns
        
        @njit
        def sharpe_ratio_nb(col, arr, risk_free_rate=0.02):
            # 计算夏普比率
            mean_return = np.nanmean(arr)
            std_return = np.nanstd(arr)
            if std_return == 0:
                return np.nan
            return (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)
        
        sharpe_ratios = prices.vbt.apply_and_reduce(
            returns_nb, 
            sharpe_ratio_nb,
            reduce_args=(0.02,)  # 无风险利率
        )
        print("夏普比率:")
        print(sharpe_ratios)
        
        # 示例5：计算滚动波动率，然后找最小值
        @njit
        def rolling_volatility_nb(col, arr, window=5):
            result = np.full_like(arr, np.nan)
            for i in range(window-1, len(arr)):
                window_data = arr[i-window+1:i+1]
                result[i] = np.nanstd(window_data)
            return result
        
        @njit
        def min_reduce_nb(col, arr):
            return np.nanmin(arr[~np.isnan(arr)])  # 忽略NaN
        
        min_volatility = prices.vbt.apply_and_reduce(
            rolling_volatility_nb, 
            min_reduce_nb,
            apply_args=(5,)  # 5期滚动窗口
        )
        print("最小滚动波动率:")
        print(min_volatility)
        
        # 示例6：去除异常值，然后计算中位数
        @njit
        def remove_outliers_nb(col, arr, z_threshold=2.0):
            mean = np.nanmean(arr)
            std = np.nanstd(arr)
            if std == 0:
                return arr
            z_scores = np.abs((arr - mean) / std)
            return arr[z_scores <= z_threshold]
        
        @njit
        def median_reduce_nb(col, arr):
            return np.nanmedian(arr)
        
        outlier_free_median = df.vbt.apply_and_reduce(
            remove_outliers_nb, 
            median_reduce_nb,
            apply_args=(2.0,)  # Z-score阈值
        )
        print("去除异常值后的中位数:")
        print(outlier_free_median)
        ```
        
        注意：
        - 应用函数和归约函数都必须是 numba 编译的 (@njit 装饰器)
        - 应用函数的第一个参数是列索引，第二个参数是列数组
        - 应用函数必须返回数组
        - 归约函数的第一个参数是列索引，第二个参数是应用函数的输出数组
        - 归约函数必须返回标量值
        - 此方法结合了变换和聚合操作，适合复杂的数据处理流程
        
        参见：
            vectorbt.generic.nb.apply_and_reduce_nb: 底层应用和归约实现
            vectorbt.generic.accessors.GenericAccessor.apply_along_axis: 沿轴应用函数
            vectorbt.generic.accessors.GenericAccessor.reduce: 归约操作
        """
        # 检查函数是否为 numba 编译的函数
        checks.assert_numba_func(apply_func_nb)
        checks.assert_numba_func(reduce_func_nb)
        
        # 设置默认参数
        if apply_args is None:
            apply_args = ()
        if reduce_args is None:
            reduce_args = ()

        # 调用 numba 编译的应用和归约函数
        out = nb.apply_and_reduce_nb(self.to_2d_array(), apply_func_nb, apply_args, reduce_func_nb, reduce_args)
        
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='apply_and_reduce'), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def reduce(self,
               reduce_func_nb: tp.Union[
                   tp.FlatGroupReduceFunc,
                   tp.FlatGroupReduceArrayFunc,
                   tp.GroupReduceFunc,
                   tp.GroupReduceArrayFunc,
                   tp.ReduceFunc,
                   tp.ReduceArrayFunc
               ],
               *args,
               returns_array: bool = False,
               returns_idx: bool = False,
               flatten: bool = False,
               order: str = 'C',
               to_index: bool = True,
               group_by: tp.GroupByLike = None,
               wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeriesFrame[float]:
        """
        按列归约数据 - 使用自定义函数对每列数据进行归约操作
        
        这是一个强大的通用归约方法，支持多种不同的归约模式和分组方式。
        可以用于实现各种统计计算、聚合操作和数据压缩。
        
        参数：
            reduce_func_nb: numba 编译的归约函数，根据其他参数使用不同的函数签名：
                - 普通模式：(col, array) -> scalar
                - 数组模式：(col, array) -> array
                - 分组模式：(group, array) -> scalar/array
                - 扁平化模式：(array) -> scalar/array
            *args: 传递给归约函数的额外参数
            returns_array (bool, 可选): 归约函数是否返回数组，默认 False
            returns_idx (bool, 可选): 归约函数是否返回索引/位置，默认 False
            flatten (bool, 可选): 是否在分组前扁平化数据，默认 False
            order (str, 可选): 扁平化顺序，'C' 或 'F'，默认 'C'
            to_index (bool, 可选): 是否将位置转换为索引标签，默认 True
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.MaybeSeriesFrame[float]: 归约后的结果
        
        归约函数类型：
        - 普通归约：每列返回一个标量值
        - 数组归约：每列返回一个数组
        - 分组归约：对分组后的数据进行归约
        - 扁平化归约：先扁平化再归约
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        from numba import njit
        
        # 创建示例数据
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': [1, 2, 3, 2, 1]
        })
        
        # 示例1：基本归约 - 计算均值
        mean_nb = njit(lambda col, a: np.nanmean(a))
        result = df.vbt.reduce(mean_nb)
        print(result)
        # 输出：
        # a    3.0
        # b    3.0
        # c    1.8
        # dtype: float64
        
        # 示例2：返回索引 - 找出最大值位置
        argmax_nb = njit(lambda col, a: np.argmax(a))
        max_idx = df.vbt.reduce(argmax_nb, returns_idx=True)
        print(max_idx)
        # 输出：时间戳索引
        
        # 示例3：返回原始位置而非标签
        max_pos = df.vbt.reduce(argmax_nb, returns_idx=True, to_index=False)
        print(max_pos)
        # 输出：
        # a    4
        # b    0
        # c    2
        # dtype: int64
        
        # 示例4：返回数组 - 计算最小值和最大值
        min_max_nb = njit(lambda col, a: np.array([np.nanmin(a), np.nanmax(a)]))
        min_max = df.vbt.reduce(
            min_max_nb, 
            returns_array=True, 
            wrap_kwargs=dict(name_or_index=['min', 'max'])
        )
        print(min_max)
        # 输出：
        #        a    b    c
        # min  1.0  1.0  1.0
        # max  5.0  5.0  3.0
        
        # 示例5：分组归约
        group_by = pd.Series(['first', 'first', 'second'], name='group')
        grouped_mean = df.vbt.reduce(mean_nb, group_by=group_by)
        print(grouped_mean)
        # 输出：
        # group
        # first     3.0
        # second    1.8
        # dtype: float64
        
        # 示例6：分组数组归约
        grouped_min_max = df.vbt.reduce(
            min_max_nb, 
            returns_array=True, 
            group_by=group_by,
            wrap_kwargs=dict(name_or_index=['min', 'max'])
        )
        print(grouped_min_max)
        # 输出：
        # group  first  second
        # min      1.0     1.0
        # max      5.0     3.0
        
        # 示例7：自定义统计函数
        def custom_stat_nb(col, arr):
            return np.sum(arr > np.mean(arr))  # 计算高于均值的元素数量
        
        custom_stat_nb = njit(custom_stat_nb)
        above_mean_count = df.vbt.reduce(custom_stat_nb)
        print(above_mean_count)
        ```
        
        底层实现：
        - 分组 + 数组 + 扁平化：vectorbt.generic.nb.flat_reduce_grouped_to_array_nb
        - 分组 + 标量 + 扁平化：vectorbt.generic.nb.flat_reduce_grouped_nb
        - 分组 + 数组：vectorbt.generic.nb.reduce_grouped_to_array_nb
        - 分组 + 标量：vectorbt.generic.nb.reduce_grouped_nb
        - 无分组 + 数组：vectorbt.generic.nb.reduce_to_array_nb
        - 无分组 + 标量：vectorbt.generic.nb.reduce_nb
        
        注意事项：
        - 归约函数必须是 numba 编译的函数
        - 当 returns_idx=True 时，函数应返回索引或位置
        - 扁平化模式下，数据会被重新排列为一维数组
        - 分组模式下，数据会按组进行归约
        """
        # 检查归约函数是否为 numba 编译函数
        checks.assert_numba_func(reduce_func_nb)

        # 根据是否分组选择不同的处理路径
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            # 分组归约模式
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            
            if flatten:
                # 扁平化分组归约
                checks.assert_in(order.upper(), ['C', 'F'])  # 检查扁平化顺序
                in_c_order = order.upper() == 'C'
                
                if returns_array:
                    # 扁平化分组归约，返回数组
                    out = nb.flat_reduce_grouped_to_array_nb(
                        self.to_2d_array(), group_lens, in_c_order, reduce_func_nb, *args)
                else:
                    # 扁平化分组归约，返回标量
                    out = nb.flat_reduce_grouped_nb(
                        self.to_2d_array(), group_lens, in_c_order, reduce_func_nb, *args)
                
                # 如果返回的是索引，需要进行位置转换
                if returns_idx:
                    if in_c_order:
                        out //= group_lens  # C 顺序扁平化
                    else:
                        out %= self.wrapper.shape[0]  # F 顺序扁平化
            else:
                # 普通分组归约
                if returns_array:
                    # 分组归约，返回数组
                    out = nb.reduce_grouped_to_array_nb(
                        self.to_2d_array(), group_lens, reduce_func_nb, *args)
                else:
                    # 分组归约，返回标量
                    out = nb.reduce_grouped_nb(
                        self.to_2d_array(), group_lens, reduce_func_nb, *args)
        else:
            # 无分组归约模式
            if returns_array:
                # 按列归约，返回数组
                out = nb.reduce_to_array_nb(
                    self.to_2d_array(), reduce_func_nb, *args)
            else:
                # 按列归约，返回标量
                out = nb.reduce_nb(
                    self.to_2d_array(), reduce_func_nb, *args)

        # 执行后处理
        wrap_kwargs = merge_dicts(dict(
            name_or_index='reduce' if not returns_array else None,  # 设置名称或索引
            to_index=returns_idx and to_index,                      # 是否转换为索引
            fillna=-1 if returns_idx else None,                     # 索引填充值
            dtype=np.int64 if returns_idx else None                 # 索引数据类型
        ), wrap_kwargs)
        
        # 包装结果并返回
        return self.wrapper.wrap_reduced(out, group_by=group_by, **wrap_kwargs)

    def min(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算最小值 - 返回非 NaN 元素的最小值
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.MaybeSeries: 每列的最小值
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103],
            'volume': [1000, 1200, 800, 600, 1100, 1500, 900]
        })
        
        # 计算每列的最小值
        min_values = df.vbt.min()
        print(min_values)
        # 输出：
        # price      95
        # volume    600
        # dtype: int64
        
        # 包含 NaN 的情况
        df_with_nan = df.copy()
        df_with_nan.loc[2, 'price'] = np.nan
        min_values_nan = df_with_nan.vbt.min()
        print(min_values_nan)  # NaN 被忽略
        
        # 分组最小值
        group_by = pd.Series(['metrics', 'metrics'], name='group')
        grouped_min = df.vbt.min(group_by=group_by)
        print(grouped_min)
        # 输出：
        # group
        # metrics    95
        # dtype: int64
        ```
        
        注意事项：
        - 自动忽略 NaN 值
        - 如果所有值都是 NaN，则返回 NaN
        - 支持分组操作
        - 对于非数值类型，使用 numpy.nanmin
        """
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='min'), wrap_kwargs)
        
        # 如果有分组，使用 reduce 方法
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(nb.min_reduce_nb, group_by=group_by, flatten=True, wrap_kwargs=wrap_kwargs)

        # 获取 2D 数组
        arr = self.to_2d_array()
        
        # 根据数据类型选择最优的 nanmin 函数
        if arr.dtype != int and arr.dtype != float:
            # bottleneck 不能处理除整数和浮点数之外的其他类型
            _nanmin = np.nanmin
        else:
            _nanmin = nanmin  # 使用 bottleneck 的高性能版本
        
        # 沿着行方向（axis=0）计算最小值
        return self.wrapper.wrap_reduced(_nanmin(arr, axis=0), group_by=False, **wrap_kwargs)

    def max(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算最大值 - 返回非 NaN 元素的最大值
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.MaybeSeries: 每列的最大值
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103],
            'volume': [1000, 1200, 800, 600, 1100, 1500, 900]
        })
        
        # 计算每列的最大值
        max_values = df.vbt.max()
        print(max_values)
        # 输出：
        # price     108
        # volume   1500
        # dtype: int64
        
        # 包含 NaN 的情况
        df_with_nan = df.copy()
        df_with_nan.loc[2, 'price'] = np.nan
        max_values_nan = df_with_nan.vbt.max()
        print(max_values_nan)  # NaN 被忽略
        
        # 分组最大值
        group_by = pd.Series(['metrics', 'metrics'], name='group')
        grouped_max = df.vbt.max(group_by=group_by)
        print(grouped_max)
        # 输出：
        # group
        # metrics    1500
        # dtype: int64
        
        # 与最小值结合使用
        range_values = df.vbt.max() - df.vbt.min()
        print(f"价格范围: {range_values['price']}")
        print(f"成交量范围: {range_values['volume']}")
        ```
        
        注意事项：
        - 自动忽略 NaN 值
        - 如果所有值都是 NaN，则返回 NaN
        - 支持分组操作
        - 对于非数值类型，使用 numpy.nanmax
        """
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='max'), wrap_kwargs)
        
        # 如果有分组，使用 reduce 方法
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(nb.max_reduce_nb, group_by=group_by, flatten=True, wrap_kwargs=wrap_kwargs)

        # 获取 2D 数组
        arr = self.to_2d_array()
        
        # 根据数据类型选择最优的 nanmax 函数
        if arr.dtype != int and arr.dtype != float:
            # bottleneck 不能处理除整数和浮点数之外的其他类型
            _nanmax = np.nanmax
        else:
            _nanmax = nanmax  # 使用 bottleneck 的高性能版本
        
        # 沿着行方向（axis=0）计算最大值
        return self.wrapper.wrap_reduced(_nanmax(arr, axis=0), group_by=False, **wrap_kwargs)

    def mean(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算平均值 - 返回非 NaN 元素的算术平均值
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.MaybeSeries: 每列的平均值
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103],
            'volume': [1000, 1200, 800, 600, 1100, 1500, 900]
        })
        
        # 计算每列的平均值
        mean_values = df.vbt.mean()
        print(mean_values)
        # 输出：
        # price     101.571429
        # volume   1014.285714
        # dtype: float64
        
        # 包含 NaN 的情况
        df_with_nan = df.copy()
        df_with_nan.loc[2, 'price'] = np.nan
        mean_values_nan = df_with_nan.vbt.mean()
        print(mean_values_nan)  # NaN 被忽略，计算剩余值的平均值
        
        # 分组平均值
        group_by = pd.Series(['metrics', 'metrics'], name='group')
        grouped_mean = df.vbt.mean(group_by=group_by)
        print(grouped_mean)
        # 输出：
        # group
        # metrics    557.928571
        # dtype: float64
        
        # 时间序列平均值分析
        returns = df['price'].pct_change().dropna()
        avg_return = returns.vbt.mean()
        print(f"平均收益率: {avg_return:.4f}")
        
        # 与滚动平均值比较
        rolling_mean = df['price'].vbt.rolling_mean(window=3)
        overall_mean = df['price'].vbt.mean()
        print(f"整体平均价格: {overall_mean:.2f}")
        print(f"最新3期平均价格: {rolling_mean.iloc[-1]:.2f}")
        ```
        
        注意事项：
        - 自动忽略 NaN 值
        - 如果所有值都是 NaN，则返回 NaN
        - 支持分组操作
        - 对于非数值类型，使用 numpy.nanmean
        - 计算的是算术平均值（简单平均）
        """
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='mean'), wrap_kwargs)
        
        # 如果有分组，使用 reduce 方法
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(
                nb.mean_reduce_nb, group_by=group_by, flatten=True, wrap_kwargs=wrap_kwargs)

        # 获取 2D 数组
        arr = self.to_2d_array()
        
        # 根据数据类型选择最优的 nanmean 函数
        if arr.dtype != int and arr.dtype != float:
            # bottleneck 不能处理除整数和浮点数之外的其他类型
            _nanmean = np.nanmean
        else:
            _nanmean = nanmean  # 使用 bottleneck 的高性能版本
        
        # 沿着行方向（axis=0）计算平均值
        return self.wrapper.wrap_reduced(_nanmean(arr, axis=0), group_by=False, **wrap_kwargs)

    def median(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算中位数 - 返回非 NaN 元素的中位数
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.MaybeSeries: 每列的中位数
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103],
            'volume': [1000, 1200, 800, 600, 1100, 1500, 900]
        })
        
        # 计算每列的中位数
        median_values = df.vbt.median()
        print(median_values)
        # 输出：
        # price     102.0
        # volume   1000.0
        # dtype: float64
        
        # 包含 NaN 的情况
        df_with_nan = df.copy()
        df_with_nan.loc[2, 'price'] = np.nan
        median_values_nan = df_with_nan.vbt.median()
        print(median_values_nan)  # NaN 被忽略
        
        # 分组中位数
        group_by = pd.Series(['metrics', 'metrics'], name='group')
        grouped_median = df.vbt.median(group_by=group_by)
        print(grouped_median)
        
        # 与均值比较
        mean_values = df.vbt.mean()
        print("价格 - 均值 vs 中位数:")
        print(f"均值: {mean_values['price']:.2f}")
        print(f"中位数: {median_values['price']:.2f}")
        ```
        
        注意事项：
        - 自动忽略 NaN 值
        - 中位数是排序后的中间值（或中间两个值的平均值）
        - 对于偶数个元素，返回中间两个值的平均值
        - 相比均值，中位数对极端值更稳健
        """
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='median'), wrap_kwargs)
        
        # 如果有分组，使用 reduce 方法
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(nb.median_reduce_nb, group_by=group_by, flatten=True, wrap_kwargs=wrap_kwargs)

        # 获取 2D 数组
        arr = self.to_2d_array()
        
        # 根据数据类型选择最优的 nanmedian 函数
        if arr.dtype != int and arr.dtype != float:
            # bottleneck 不能处理除整数和浮点数之外的其他类型
            _nanmedian = np.nanmedian
        else:
            _nanmedian = nanmedian  # 使用 bottleneck 的高性能版本
        
        # 沿着行方向（axis=0）计算中位数
        return self.wrapper.wrap_reduced(_nanmedian(arr, axis=0), group_by=False, **wrap_kwargs)

    def std(self, ddof: int = 1, group_by: tp.GroupByLike = None,
            wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算标准差 - 返回非 NaN 元素的标准差
        
        参数：
            ddof (int, 可选): 自由度增量，默认为 1
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.MaybeSeries: 每列的标准差
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103],
            'volume': [1000, 1200, 800, 600, 1100, 1500, 900]
        })
        
        # 计算每列的标准差
        std_values = df.vbt.std()
        print(std_values)
        # 输出：
        # price     4.645785
        # volume    313.049517
        # dtype: float64
        
        # 使用总体标准差 (ddof=0)
        std_population = df.vbt.std(ddof=0)
        print(std_population)
        
        # 包含 NaN 的情况
        df_with_nan = df.copy()
        df_with_nan.loc[2, 'price'] = np.nan
        std_values_nan = df_with_nan.vbt.std()
        print(std_values_nan)  # NaN 被忽略
        
        # 分组标准差
        group_by = pd.Series(['metrics', 'metrics'], name='group')
        grouped_std = df.vbt.std(group_by=group_by)
        print(grouped_std)
        
        # 金融应用：计算收益率的波动率
        returns = df['price'].pct_change().dropna()
        volatility = returns.vbt.std()
        annualized_vol = volatility * np.sqrt(252)
        print(f"日波动率: {volatility:.4f}")
        print(f"年化波动率: {annualized_vol:.4f}")
        
        # 变异系数（标准差/均值）
        mean_values = df.vbt.mean()
        cv = std_values / mean_values
        print("变异系数:")
        print(cv)
        ```
        
        注意事项：
        - 自动忽略 NaN 值
        - ddof=1 为样本标准差（默认），ddof=0 为总体标准差
        - 标准差衡量数据的离散程度
        - 标准差的单位与原始数据相同
        - 在金融分析中，标准差常用于衡量波动率
        """
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='std'), wrap_kwargs)
        
        # 如果有分组，使用 reduce 方法
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(nb.std_reduce_nb, ddof, group_by=group_by, flatten=True, wrap_kwargs=wrap_kwargs)

        # 获取 2D 数组
        arr = self.to_2d_array()
        
        # 根据数据类型选择最优的 nanstd 函数
        if arr.dtype != int and arr.dtype != float:
            # bottleneck 不能处理除整数和浮点数之外的其他类型
            _nanstd = np.nanstd
        else:
            _nanstd = nanstd  # 使用 bottleneck 的高性能版本
        
        # 沿着行方向（axis=0）计算标准差
        return self.wrapper.wrap_reduced(_nanstd(arr, ddof=ddof, axis=0), group_by=False, **wrap_kwargs)

    def sum(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算总和 - 返回非 NaN 元素的总和
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.MaybeSeries: 每列的总和
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103],
            'volume': [1000, 1200, 800, 600, 1100, 1500, 900]
        })
        
        # 计算每列的总和
        sum_values = df.vbt.sum()
        print(sum_values)
        # 输出：
        # price      711
        # volume    7100
        # dtype: int64
        
        # 包含 NaN 的情况
        df_with_nan = df.copy()
        df_with_nan.loc[2, 'price'] = np.nan
        sum_values_nan = df_with_nan.vbt.sum()
        print(sum_values_nan)  # NaN 被忽略
        
        # 分组总和
        group_by = pd.Series(['metrics', 'metrics'], name='group')
        grouped_sum = df.vbt.sum(group_by=group_by)
        print(grouped_sum)
        
        # 金融应用：计算总交易量
        total_volume = df['volume'].vbt.sum()
        print(f"总交易量: {total_volume:,}")
        
        # 投资组合价值计算
        holdings = pd.Series([100, 200, 150])  # 持股数量
        portfolio_value = (df['price'] * holdings).vbt.sum()
        print(f"投资组合价值: {portfolio_value}")
        
        # 与均值的关系
        mean_values = df.vbt.mean()
        count_values = df.vbt.count()
        print("验证关系：sum = mean * count")
        print(f"实际总和: {sum_values['price']}")
        print(f"计算总和: {mean_values['price'] * count_values['price']:.0f}")
        ```
        
        注意事项：
        - 自动忽略 NaN 值
        - 如果所有值都是 NaN，则返回 0
        - 支持分组操作
        - 对于布尔类型，True 被视为 1，False 被视为 0
        - 在金融分析中常用于计算总交易量、总收益等
        """
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='sum'), wrap_kwargs)
        
        # 如果有分组，使用 reduce 方法
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(nb.sum_reduce_nb, group_by=group_by, flatten=True, wrap_kwargs=wrap_kwargs)

        # 获取 2D 数组
        arr = self.to_2d_array()
        
        # 根据数据类型选择最优的 nansum 函数
        if arr.dtype != int and arr.dtype != float:
            # bottleneck 不能处理除整数和浮点数之外的其他类型
            _nansum = np.nansum
        else:
            _nansum = nansum  # 使用 bottleneck 的高性能版本
        
        # 沿着行方向（axis=0）计算总和
        return self.wrapper.wrap_reduced(_nansum(arr, axis=0), group_by=False, **wrap_kwargs)

    def count(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        计算计数 - 返回非 NaN 元素的数量
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.MaybeSeries: 每列的非 NaN 元素数量
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        
        # 创建示例数据
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103],
            'volume': [1000, 1200, 800, 600, 1100, 1500, 900]
        })
        
        # 计算每列的计数
        count_values = df.vbt.count()
        print(count_values)
        # 输出：
        # price     7
        # volume    7
        # dtype: int64
        
        # 包含 NaN 的情况
        df_with_nan = df.copy()
        df_with_nan.loc[2, 'price'] = np.nan
        df_with_nan.loc[4, 'volume'] = np.nan
        count_values_nan = df_with_nan.vbt.count()
        print(count_values_nan)
        # 输出：
        # price     6
        # volume    6
        # dtype: int64
        
        # 分组计数
        group_by = pd.Series(['metrics', 'metrics'], name='group')
        grouped_count = df.vbt.count(group_by=group_by)
        print(grouped_count)
        
        # 数据完整性检查
        total_rows = len(df)
        missing_data = total_rows - count_values
        print("数据完整性检查:")
        print(f"总行数: {total_rows}")
        print(f"缺失值数量:")
        print(missing_data)
        
        # 计算缺失值比例
        missing_ratio = missing_data / total_rows
        print("缺失值比例:")
        print(missing_ratio)
        
        # 过滤有效数据
        valid_data_mask = ~df.isna()
        valid_count = valid_data_mask.sum()
        print("使用 pandas 验证:")
        print(valid_count)
        print("与 vbt.count 结果一致:", (valid_count == count_values).all())
        
        # 金融应用：计算交易日数
        trading_days = df['price'].vbt.count()
        print(f"交易日数: {trading_days}")
        
        # 数据质量评估
        data_quality = count_values / total_rows
        print("数据质量评分（完整性）:")
        print(data_quality)
        ```
        
        注意事项：
        - 只统计非 NaN 值的数量
        - 对于空的 Series/DataFrame，返回 0
        - 支持分组操作
        - 返回整数类型 (int64)
        - 在数据清洗和质量评估中非常有用
        """
        # 设置包装器参数，指定返回整数类型
        wrap_kwargs = merge_dicts(dict(name_or_index='count', dtype=np.int64), wrap_kwargs)
        
        # 如果有分组，使用 reduce 方法
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(nb.count_reduce_nb, group_by=group_by, flatten=True, wrap_kwargs=wrap_kwargs)

        # 计算非 NaN 元素的数量
        # 使用 ~np.isnan() 创建布尔掩码，然后求和
        return self.wrapper.wrap_reduced(np.sum(~np.isnan(self.to_2d_array()), axis=0), group_by=False, **wrap_kwargs)

    def idxmin(self, group_by: tp.GroupByLike = None, order: str = 'C',
               wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        返回最小值的标签索引 - 返回非 NaN 元素中最小值的索引标签
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            order (str, 可选): 扁平化顺序，'C' 或 'F'，默认 'C'
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.MaybeSeries: 每列最小值的索引标签
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        
        # 创建示例数据
        dates = pd.date_range('2023-01-01', periods=7, freq='D')
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103],
            'volume': [1000, 1200, 800, 600, 1100, 1500, 900]
        }, index=dates)
        
        # 获取最小值的索引
        min_indices = df.vbt.idxmin()
        print(min_indices)
        # 输出：
        # price     2023-01-04
        # volume    2023-01-04
        # dtype: datetime64[ns]
        
        # 验证结果
        print("验证最小值:")
        print(f"价格最小值: {df.loc[min_indices['price'], 'price']}")
        print(f"成交量最小值: {df.loc[min_indices['volume'], 'volume']}")
        
        # 包含 NaN 的情况
        df_with_nan = df.copy()
        df_with_nan.loc[dates[3], 'price'] = np.nan  # 将最小值设为 NaN
        min_indices_nan = df_with_nan.vbt.idxmin()
        print("包含 NaN 时的最小值索引:")
        print(min_indices_nan)
        
        # 分组最小值索引
        group_by = pd.Series(['metrics', 'metrics'], name='group')
        grouped_min_idx = df.vbt.idxmin(group_by=group_by)
        print(grouped_min_idx)
        
        # 金融应用：找到最低价格日期
        lowest_price_date = df['price'].vbt.idxmin()
        print(f"最低价格日期: {lowest_price_date}")
        print(f"最低价格: {df.loc[lowest_price_date, 'price']}")
        
        # 与 pandas 比较
        pandas_idxmin = df.idxmin()
        print("与 pandas 结果一致:", (pandas_idxmin == min_indices).all())
        
        # 时间序列分析
        returns = df['price'].pct_change().dropna()
        worst_day = returns.vbt.idxmin()
        print(f"最差表现日期: {worst_day}")
        print(f"最差收益率: {returns.loc[worst_day]:.2%}")
        ```
        
        注意事项：
        - 忽略 NaN 值
        - 如果所有值都是 NaN，则返回 NaN
        - 返回的是索引标签，不是位置
        - 支持分组操作
        - 对于有相同最小值的情况，返回第一个出现的索引
        """
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='idxmin'), wrap_kwargs)
        
        # 如果有分组，使用 reduce 方法
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(
                nb.argmin_reduce_nb,
                group_by=group_by,
                flatten=True,
                returns_idx=True,
                order=order,
                wrap_kwargs=wrap_kwargs
            )

        # 获取 2D 数组
        obj = self.to_2d_array()
        
        # 初始化输出数组
        out = np.full(obj.shape[1], np.nan, dtype=object)
        
        # 找出全为 NaN 的列
        nan_mask = np.all(np.isnan(obj), axis=0)
        
        # 对于非全 NaN 的列，找到最小值的索引并转换为索引标签
        out[~nan_mask] = self.wrapper.index[nanargmin(obj[:, ~nan_mask], axis=0)]
        
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def idxmax(self, group_by: tp.GroupByLike = None, order: str = 'C',
               wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        返回最大值的标签索引 - 返回非 NaN 元素中最大值的索引标签
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            order (str, 可选): 扁平化顺序，'C' 或 'F'，默认 'C'
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.MaybeSeries: 每列最大值的索引标签
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        import numpy as np
        
        # 创建示例数据
        dates = pd.date_range('2023-01-01', periods=7, freq='D')
        df = pd.DataFrame({
            'price': [100, 105, 98, 95, 102, 108, 103],
            'volume': [1000, 1200, 800, 600, 1100, 1500, 900]
        }, index=dates)
        
        # 获取最大值的索引
        max_indices = df.vbt.idxmax()
        print(max_indices)
        # 输出：
        # price     2023-01-06
        # volume    2023-01-06
        # dtype: datetime64[ns]
        
        # 验证结果
        print("验证最大值:")
        print(f"价格最大值: {df.loc[max_indices['price'], 'price']}")
        print(f"成交量最大值: {df.loc[max_indices['volume'], 'volume']}")
        
        # 包含 NaN 的情况
        df_with_nan = df.copy()
        df_with_nan.loc[dates[5], 'price'] = np.nan  # 将最大值设为 NaN
        max_indices_nan = df_with_nan.vbt.idxmax()
        print("包含 NaN 时的最大值索引:")
        print(max_indices_nan)
        
        # 分组最大值索引
        group_by = pd.Series(['metrics', 'metrics'], name='group')
        grouped_max_idx = df.vbt.idxmax(group_by=group_by)
        print(grouped_max_idx)
        
        # 金融应用：找到最高价格日期
        highest_price_date = df['price'].vbt.idxmax()
        print(f"最高价格日期: {highest_price_date}")
        print(f"最高价格: {df.loc[highest_price_date, 'price']}")
        
        # 与 pandas 比较
        pandas_idxmax = df.idxmax()
        print("与 pandas 结果一致:", (pandas_idxmax == max_indices).all())
        
        # 时间序列分析
        returns = df['price'].pct_change().dropna()
        best_day = returns.vbt.idxmax()
        print(f"最佳表现日期: {best_day}")
        print(f"最佳收益率: {returns.loc[best_day]:.2%}")
        
        # 波动率分析
        rolling_vol = returns.vbt.rolling_std(window=3)
        highest_vol_date = rolling_vol.vbt.idxmax()
        print(f"最高波动率日期: {highest_vol_date}")
        print(f"最高波动率: {rolling_vol.loc[highest_vol_date]:.4f}")
        
        # 范围分析
        min_indices = df.vbt.idxmin()
        max_indices = df.vbt.idxmax()
        print("价格范围:")
        print(f"从 {min_indices['price']} 到 {max_indices['price']}")
        print(f"价格变化: {df.loc[max_indices['price'], 'price'] - df.loc[min_indices['price'], 'price']}")
        ```
        
        注意事项：
        - 忽略 NaN 值
        - 如果所有值都是 NaN，则返回 NaN
        - 返回的是索引标签，不是位置
        - 支持分组操作
        - 对于有相同最大值的情况，返回第一个出现的索引
        """
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='idxmax'), wrap_kwargs)
        
        # 如果有分组，使用 reduce 方法
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(
                nb.argmax_reduce_nb,
                group_by=group_by,
                flatten=True,
                returns_idx=True,
                order=order,
                wrap_kwargs=wrap_kwargs
            )

        # 获取 2D 数组
        obj = self.to_2d_array()
        
        # 初始化输出数组
        out = np.full(obj.shape[1], np.nan, dtype=object)
        
        # 找出全为 NaN 的列
        nan_mask = np.all(np.isnan(obj), axis=0)
        
        # 对于非全 NaN 的列，找到最大值的索引并转换为索引标签
        out[~nan_mask] = self.wrapper.index[nanargmax(obj[:, ~nan_mask], axis=0)]
        
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def describe(self, percentiles: tp.Optional[tp.ArrayLike] = None, ddof: int = 1,
                 group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        统计描述方法 - 计算数据的描述性统计指标
        
        这个方法类似于pandas的describe方法，但提供了更多的自定义选项。
        它计算数据的主要统计指标，包括计数、均值、标准差、最小值、分位数和最大值。
        
        参数：
            percentiles (tp.Optional[tp.ArrayLike], 可选): 要计算的百分位数
                默认为[0.25, 0.5, 0.75]，即25%、50%（中位数）、75%分位数
            ddof (int, 可选): 计算标准差时的自由度修正，默认为1
                1表示样本标准差，0表示总体标准差
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
        
        返回：
            tp.SeriesFrame: 包含描述性统计指标的DataFrame或Series
        
        统计指标包括：
        - count: 非NaN值的数量
        - mean: 平均值
        - std: 标准差（根据ddof参数调整）
        - min: 最小值
        - 25%/50%/75%: 分位数（根据percentiles参数）
        - max: 最大值
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 示例1：基本描述性统计
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10],
            'C': [1, 1, 2, 2, 3]
        })
        
        # 计算描述性统计
        desc = df.vbt.describe()
        print("基本描述性统计:")
        print(desc)
        
        # 示例2：自定义百分位数
        custom_desc = df.vbt.describe(percentiles=[0.1, 0.5, 0.9])
        print("自定义百分位数:")
        print(custom_desc)
        
        # 示例3：使用总体标准差
        pop_desc = df.vbt.describe(ddof=0)
        print("总体标准差:")
        print(pop_desc)
        
        # 示例4：金融数据分析
        # 模拟股票收益率数据
        np.random.seed(42)
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.08, 0.15, 252),
            'GOOGL': np.random.normal(0.12, 0.20, 252),
            'MSFT': np.random.normal(0.10, 0.18, 252)
        })
        
        # 计算收益率统计
        returns_desc = returns.vbt.describe()
        print("股票收益率统计:")
        print(returns_desc)
        
        # 示例5：风险分析中的分位数
        risk_desc = returns.vbt.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        print("风险分析分位数:")
        print(risk_desc)
        
        # 示例6：分组描述性统计
        portfolio = pd.DataFrame({
            'Tech_Stock1': np.random.normal(0.12, 0.20, 100),
            'Tech_Stock2': np.random.normal(0.10, 0.18, 100),
            'Finance_Stock1': np.random.normal(0.08, 0.15, 100),
            'Finance_Stock2': np.random.normal(0.09, 0.16, 100)
        })
        
        # 按行业分组
        group_desc = portfolio.vbt.describe(
            group_by=['Tech', 'Tech', 'Finance', 'Finance']
        )
        print("按行业分组的描述性统计:")
        print(group_desc)
        
        # 示例7：时间序列数据的描述性统计
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        ts_data = pd.Series(np.random.normal(100, 15, 252), index=dates)
        
        ts_desc = ts_data.vbt.describe()
        print("时间序列描述性统计:")
        print(ts_desc)
        ```
        
        注意：
        - 50%分位数（中位数）总是包含在结果中，即使没有在percentiles中指定
        - percentiles参数会自动去重和排序
        - 该方法基于高效的numba实现，处理大数据集时性能优异
        """
        # 处理百分位数参数
        if percentiles is not None:
            # 将百分位数转换为1D数组
            percentiles = reshape_fns.to_1d_array(percentiles)
        else:
            # 使用默认的25%、50%、75%分位数
            percentiles = np.array([0.25, 0.5, 0.75])
        
        # 转换为列表以便操作
        percentiles = percentiles.tolist()
        
        # 确保50%分位数（中位数）包含在内
        if 0.5 not in percentiles:
            percentiles.append(0.5)
        
        # 去重并排序
        percentiles = np.unique(percentiles)
        
        # 格式化百分位数标签，例如25% -> "25%"
        perc_formatted = pd.io.formats.format.format_percentiles(percentiles)
        
        # 创建结果的索引标签
        index = pd.Index(['count', 'mean', 'std', 'min', *perc_formatted, 'max'])
        
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index=index), wrap_kwargs)
        
        # 根据是否分组选择不同的处理方式
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            # 分组情况下的描述性统计
            return self.reduce(
                nb.describe_reduce_nb, percentiles, ddof,
                group_by=group_by, flatten=True, returns_array=True,
                wrap_kwargs=wrap_kwargs)
        
        # 非分组情况下的描述性统计
        return self.reduce(
            nb.describe_reduce_nb, percentiles, ddof,
            returns_array=True, wrap_kwargs=wrap_kwargs)

    def value_counts(self,
                     normalize: bool = False,
                     sort_uniques: bool = True,
                     sort: bool = False,
                     ascending: bool = False,
                     dropna: bool = False,
                     group_by: tp.GroupByLike = None,
                     mapping: tp.Optional[tp.MappingLike] = None,
                     incl_all_keys: bool = False,
                     wrap_kwargs: tp.KwargsLike = None,
                     **kwargs) -> tp.SeriesFrame:
        """
        值计数方法 - 计算唯一值的出现次数
        
        这个方法类似于pandas的value_counts方法，但提供了更多的自定义选项，
        包括映射功能、分组支持和高级排序选项。
        
        参数：
            normalize (bool, 可选): 是否返回相对频率而不是绝对计数，默认False
            sort_uniques (bool, 可选): 是否对唯一值进行排序，默认True
            sort (bool, 可选): 是否按频率排序，默认False
            ascending (bool, 可选): 排序是否升序，默认False（降序）
            dropna (bool, 可选): 是否排除NaN值，默认False
            group_by (tp.GroupByLike, 可选): 分组方式
            mapping (tp.Optional[tp.MappingLike], 可选): 值映射字典
                可以是字典、Series或字符串（'index'/'columns'）
            incl_all_keys (bool, 可选): 是否包含映射中的所有键，默认False
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
            **kwargs: 传递给映射函数的额外参数
        
        返回：
            tp.SeriesFrame: 包含值计数的DataFrame或Series
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 示例1：基本值计数
        data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        counts = data.vbt.value_counts()
        print("基本值计数:")
        print(counts)
        
        # 示例2：归一化频率
        freq = data.vbt.value_counts(normalize=True)
        print("相对频率:")
        print(freq)
        
        # 示例3：按频率排序
        sorted_counts = data.vbt.value_counts(sort=True, ascending=True)
        print("按频率升序排序:")
        print(sorted_counts)
        
        # 示例4：处理NaN值
        data_with_nan = pd.Series([1, 2, np.nan, 2, 3, np.nan, 3, 3])
        
        # 包含NaN的计数
        counts_with_nan = data_with_nan.vbt.value_counts()
        print("包含NaN的计数:")
        print(counts_with_nan)
        
        # 排除NaN的计数
        counts_no_nan = data_with_nan.vbt.value_counts(dropna=True)
        print("排除NaN的计数:")
        print(counts_no_nan)
        
        # 示例5：使用映射
        categorical_data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A'])
        
        # 定义映射
        mapping = {'A': 'Apple', 'B': 'Banana', 'C': 'Cherry'}
        mapped_counts = categorical_data.vbt.value_counts(mapping=mapping)
        print("映射后的计数:")
        print(mapped_counts)
        
        # 示例6：金融数据分析 - 交易信号统计
        # 模拟交易信号数据
        np.random.seed(42)
        signals = pd.Series(np.random.choice(['BUY', 'SELL', 'HOLD'], 252, p=[0.3, 0.3, 0.4]))
        
        signal_counts = signals.vbt.value_counts(sort=True)
        print("交易信号统计:")
        print(signal_counts)
        
        # 计算信号频率
        signal_freq = signals.vbt.value_counts(normalize=True, sort=True)
        print("交易信号频率:")
        print(signal_freq)
        
        # 示例7：多列数据的值计数
        df = pd.DataFrame({
            'Strategy_A': np.random.choice([0, 1], 100),
            'Strategy_B': np.random.choice([0, 1], 100),
            'Strategy_C': np.random.choice([0, 1], 100)
        })
        
        # 每列的值计数
        multi_counts = df.vbt.value_counts()
        print("多列值计数:")
        print(multi_counts)
        
        # 示例8：等级/评级数据分析
        ratings = pd.Series(['AAA', 'AA', 'A', 'BBB', 'BB', 'B'] * 20)
        
        # 定义评级映射
        rating_mapping = {
            'AAA': 'Investment Grade',
            'AA': 'Investment Grade',
            'A': 'Investment Grade',
            'BBB': 'Investment Grade',
            'BB': 'Speculative Grade',
            'B': 'Speculative Grade'
        }
        
        rating_counts = ratings.vbt.value_counts(
            mapping=rating_mapping,
            sort=True,
            incl_all_keys=True
        )
        print("评级分类统计:")
        print(rating_counts)
        
        # 示例9：时间序列模式分析
        # 创建时间序列方向数据
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        directions = pd.Series(['UP' if x > 0 else 'DOWN' for x in returns])
        
        direction_counts = directions.vbt.value_counts(normalize=True)
        print("市场方向统计:")
        print(direction_counts)
        
        # 示例10：分组值计数
        portfolio_signals = pd.DataFrame({
            'Tech_Stock1': np.random.choice(['BUY', 'SELL', 'HOLD'], 100),
            'Tech_Stock2': np.random.choice(['BUY', 'SELL', 'HOLD'], 100),
            'Finance_Stock1': np.random.choice(['BUY', 'SELL', 'HOLD'], 100),
            'Finance_Stock2': np.random.choice(['BUY', 'SELL', 'HOLD'], 100)
        })
        
        # 按行业分组统计
        grouped_counts = portfolio_signals.vbt.value_counts(
            group_by=['Tech', 'Tech', 'Finance', 'Finance'],
            normalize=True
        )
        print("按行业分组的信号统计:")
        print(grouped_counts)
        ```
        
        注意：
        - 支持多种数据类型的值计数
        - 映射功能可以将原始值转换为更有意义的标签
        - incl_all_keys参数确保映射中的所有键都出现在结果中
        - 性能优化：基于numba的高效实现
        """
        # 导入pandas版本检查
        from pkg_resources import parse_version

        # 如果没有提供映射，使用对象的默认映射
        if mapping is None:
            mapping = self.mapping
        
        # 处理字符串映射类型
        if isinstance(mapping, str):
            if mapping.lower() == 'index':
                mapping = self.wrapper.index
            elif mapping.lower() == 'columns':
                mapping = self.wrapper.columns
            mapping = to_mapping(mapping)
        
        # 根据pandas版本选择factorize方法
        if parse_version(pd.__version__) < parse_version("1.5.0"):
            codes, uniques = pd.factorize(self.obj.values.flatten(), sort=False, na_sentinel=None)
        else:
            codes, uniques = pd.factorize(self.obj.values.flatten(), sort=False, use_na_sentinel=False)
        
        # 重塑codes为2D数组
        codes = codes.reshape(self.wrapper.shape_2d)
        
        # 获取分组长度
        group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
        
        # 使用numba函数计算值计数
        value_counts = nb.value_counts_nb(codes, len(uniques), group_lens)
        
        # 如果需要包含映射中的所有键
        if incl_all_keys and mapping is not None:
            missing_keys = []
            for x in mapping:
                # 处理NaN值
                if pd.isnull(x) and pd.isnull(uniques).any():
                    continue
                # 添加缺失的键
                if x not in uniques:
                    missing_keys.append(x)
            
            # 为缺失的键添加零计数
            value_counts = np.vstack((value_counts, np.full((len(missing_keys), value_counts.shape[1]), 0)))
            uniques = np.concatenate((uniques, np.array(missing_keys)))
        
        # 创建NaN掩码
        nan_mask = np.isnan(uniques)
        
        # 如果需要排除NaN
        if dropna:
            value_counts = value_counts[~nan_mask]
            uniques = uniques[~nan_mask]
        
        # 如果需要对唯一值排序
        if sort_uniques:
            new_indices = uniques.argsort()
            value_counts = value_counts[new_indices]
            uniques = uniques[new_indices]
        
        # 计算每个唯一值的总计数
        value_counts_sum = value_counts.sum(axis=1)
        
        # 如果需要归一化
        if normalize:
            value_counts = value_counts / value_counts_sum.sum()
        
        # 如果需要按频率排序
        if sort:
            if ascending:
                new_indices = value_counts_sum.argsort()
            else:
                new_indices = (-value_counts_sum).argsort()
            value_counts = value_counts[new_indices]
            uniques = uniques[new_indices]
        
        # 包装结果
        value_counts_pd = self.wrapper.wrap(
            value_counts,
            index=uniques,
            group_by=group_by,
            **merge_dicts({}, wrap_kwargs)
        )
        
        # 如果有映射，应用映射到索引
        if mapping is not None:
            value_counts_pd.index = apply_mapping(value_counts_pd.index, mapping, **kwargs)
        
        return value_counts_pd

    # ############# 分辨率处理 ############# #

    def resolve_self(self: GenericAccessorT,
                     cond_kwargs: tp.KwargsLike = None,
                     custom_arg_names: tp.Optional[tp.Set[str]] = None,
                     impacts_caching: bool = True,
                     silence_warnings: bool = False) -> GenericAccessorT:
        """
        解析自身对象 - 根据条件参数解析和创建新的访问器实例
        
        这个方法用于根据条件参数解析自身对象，主要用于处理条件参数中的映射变化。
        当映射（mapping）发生变化时，会创建一个新的访问器实例。
        
        参数：
            cond_kwargs (tp.KwargsLike, 可选): 条件参数字典
            custom_arg_names (tp.Optional[tp.Set[str]], 可选): 自定义参数名称集合
            impacts_caching (bool, 可选): 是否影响缓存，默认True
            silence_warnings (bool, 可选): 是否静默警告，默认False
        
        返回：
            GenericAccessorT: 解析后的访问器实例
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 示例1：基本解析操作
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10]
        })
        
        # 创建访问器
        accessor = df.vbt
        
        # 解析自身
        resolved = accessor.resolve_self()
        print("解析后的访问器类型:", type(resolved))
        
        # 示例2：使用条件参数
        cond_kwargs = {'mapping': {'A': 'Asset_A', 'B': 'Asset_B'}}
        resolved_with_mapping = accessor.resolve_self(cond_kwargs=cond_kwargs)
        print("解析后的映射:", resolved_with_mapping.mapping)
        
        # 示例3：缓存影响控制
        resolved_no_cache = accessor.resolve_self(
            cond_kwargs=cond_kwargs,
            impacts_caching=False
        )
        
        # 示例4：静默警告
        resolved_silent = accessor.resolve_self(
            cond_kwargs=cond_kwargs,
            silence_warnings=True
        )
        
        # 示例5：自定义参数名称
        custom_names = {'self', 'obj'}
        resolved_custom = accessor.resolve_self(
            cond_kwargs=cond_kwargs,
            custom_arg_names=custom_names
        )
        
        # 示例6：金融数据映射解析
        stock_data = pd.DataFrame({
            'AAPL': [100, 102, 98, 105],
            'GOOGL': [200, 205, 195, 210],
            'MSFT': [150, 155, 148, 160]
        })
        
        # 定义股票名称映射
        stock_mapping = {
            'AAPL': 'Apple Inc.',
            'GOOGL': 'Alphabet Inc.',
            'MSFT': 'Microsoft Corp.'
        }
        
        # 解析并应用映射
        stock_accessor = stock_data.vbt.resolve_self(
            cond_kwargs={'mapping': stock_mapping}
        )
        
        # 示例7：条件解析在数据处理中的应用
        portfolio_data = pd.DataFrame({
            'Tech_1': np.random.randn(100),
            'Tech_2': np.random.randn(100),
            'Finance_1': np.random.randn(100),
            'Finance_2': np.random.randn(100)
        })
        
        # 行业映射
        sector_mapping = {
            'Tech_1': 'Technology',
            'Tech_2': 'Technology',
            'Finance_1': 'Finance',
            'Finance_2': 'Finance'
        }
        
        # 解析并应用行业映射
        sector_accessor = portfolio_data.vbt.resolve_self(
            cond_kwargs={'mapping': sector_mapping}
        )
        
        # 示例8：动态解析
        def dynamic_resolve(data, new_mapping):
            \"\"\"动态解析函数\"\"\"
            return data.vbt.resolve_self(
                cond_kwargs={'mapping': new_mapping},
                silence_warnings=True
            )
        
        # 应用动态解析
        dynamic_accessor = dynamic_resolve(portfolio_data, sector_mapping)
        print("动态解析完成")
        ```
        
        注意：
        - 当映射发生变化时，会创建新的对象实例
        - 新实例的缓存设置可能会被重置
        - 自定义参数名称用于控制哪些参数被解析
        - 解析操作主要用于内部处理，用户通常不需要直接调用
        """
        # 初始化条件参数
        if cond_kwargs is None:
            cond_kwargs = {}
        if custom_arg_names is None:
            custom_arg_names = set()

        # 调用基类的resolve_self方法
        reself = Wrapping.resolve_self(
            self,
            cond_kwargs=cond_kwargs,
            custom_arg_names=custom_arg_names,
            impacts_caching=impacts_caching,
            silence_warnings=silence_warnings
        )
        
        # 如果条件参数中包含映射
        if 'mapping' in cond_kwargs:
            # 创建带有新映射的副本
            self_copy = reself.replace(mapping=cond_kwargs['mapping'])

            # 检查映射是否确实发生了变化
            if not checks.is_deep_equal(self_copy.mapping, reself.mapping):
                # 如果映射发生变化，发出警告（除非静默）
                if not silence_warnings:
                    warnings.warn(f"Changing the mapping will create a copy of this object. "
                                  f"Consider setting it upon object creation to re-use existing cache.", stacklevel=2)
                
                # 更新条件参数中的自身别名
                for alias in reself.self_aliases:
                    if alias not in custom_arg_names:
                        cond_kwargs[alias] = self_copy
                
                # 更新映射参数
                cond_kwargs['mapping'] = self_copy.mapping
                
                # 如果影响缓存，禁用缓存
                if impacts_caching:
                    cond_kwargs['use_caching'] = False
                
                return self_copy
        
        return reself

    # ############# 统计分析 ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """
        统计分析默认配置属性 - 返回统计分析方法的默认参数
        
        这个属性合并了基类的统计默认配置和generic模块的特定配置，
        提供了统计分析方法的默认参数设置。
        
        返回：
            tp.Kwargs: 统计分析的默认参数字典
        
        配置来源：
        - StatsBuilderMixin.stats_defaults：基类统计配置
        - settings['generic']['stats']：generic模块特定配置
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        import numpy as np
        
        # 示例1：查看默认配置
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10]
        })
        
        defaults = df.vbt.stats_defaults
        print("统计分析默认配置:")
        for key, value in defaults.items():
            print(f"  {key}: {value}")
        
        # 示例2：使用默认配置进行统计
        stats = df.vbt.stats()
        print("使用默认配置的统计结果:")
        print(stats)
        
        # 示例3：自定义配置覆盖默认配置
        custom_stats = df.vbt.stats(
            metrics=['mean', 'std', 'min', 'max'],
            template_stats=False
        )
        print("自定义配置的统计结果:")
        print(custom_stats)
        
        # 示例4：金融数据统计分析
        np.random.seed(42)
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.002, 0.025, 252),
            'MSFT': np.random.normal(0.0015, 0.022, 252)
        })
        
        # 查看金融数据的默认统计配置
        finance_defaults = returns.vbt.stats_defaults
        print("金融数据统计默认配置:")
        for key, value in finance_defaults.items():
            print(f"  {key}: {value}")
        
        # 应用默认配置
        finance_stats = returns.vbt.stats()
        print("金融数据统计结果:")
        print(finance_stats)
        ```
        
        注意：
        - 默认配置可以通过settings进行全局修改
        - 这些配置影响stats()方法的行为
        - 配置包括要计算的指标、模板设置等
        """
        from vectorbt._settings import settings
        # 获取generic模块的统计配置
        generic_stats_cfg = settings['generic']['stats']

        # 合并基类配置和模块特定配置
        return merge_dicts(
            StatsBuilderMixin.stats_defaults.__get__(self),
            generic_stats_cfg
        )

    # 统计指标配置 - 定义所有可用的统计指标
    _metrics: tp.ClassVar[Config] = Config(
        dict(
            # 起始时间指标
            start=dict(
                title='Start',                               # 指标标题：开始时间
                calc_func=lambda self: self.wrapper.index[0],  # 计算函数：返回第一个索引
                agg_func=None,                               # 聚合函数：无需聚合
                tags='wrapper'                               # 标签：属于包装器相关指标
            ),
            # 结束时间指标
            end=dict(
                title='End',                                 # 指标标题：结束时间
                calc_func=lambda self: self.wrapper.index[-1], # 计算函数：返回最后一个索引
                agg_func=None,                               # 聚合函数：无需聚合
                tags='wrapper'                               # 标签：属于包装器相关指标
            ),
            # 时间周期指标
            period=dict(
                title='Period',                              # 指标标题：时间周期
                calc_func=lambda self: len(self.wrapper.index), # 计算函数：返回索引长度
                apply_to_timedelta=True,                     # 应用时间差转换
                agg_func=None,                               # 聚合函数：无需聚合
                tags='wrapper'                               # 标签：属于包装器相关指标
            ),
            # 计数指标
            count=dict(
                title='Count',                               # 指标标题：计数
                calc_func='count',                           # 计算函数：调用count方法
                inv_check_has_mapping=True,                  # 反向检查是否有映射
                tags=['generic', 'describe']                 # 标签：通用和描述相关
            ),
            # 均值指标
            mean=dict(
                title='Mean',                                # 指标标题：均值
                calc_func='mean',                            # 计算函数：调用mean方法
                inv_check_has_mapping=True,                  # 反向检查是否有映射
                tags=['generic', 'describe']                 # 标签：通用和描述相关
            ),
            # 标准差指标
            std=dict(
                title='Std',                                 # 指标标题：标准差
                calc_func='std',                             # 计算函数：调用std方法
                inv_check_has_mapping=True,                  # 反向检查是否有映射
                tags=['generic', 'describe']                 # 标签：通用和描述相关
            ),
            # 最小值指标
            min=dict(
                title='Min',                                 # 指标标题：最小值
                calc_func='min',                             # 计算函数：调用min方法
                inv_check_has_mapping=True,                  # 反向检查是否有映射
                tags=['generic', 'describe']                 # 标签：通用和描述相关
            ),
            # 中位数指标
            median=dict(
                title='Median',                              # 指标标题：中位数
                calc_func='median',                          # 计算函数：调用median方法
                inv_check_has_mapping=True,                  # 反向检查是否有映射
                tags=['generic', 'describe']                 # 标签：通用和描述相关
            ),
            # 最大值指标
            max=dict(
                title='Max',                                 # 指标标题：最大值
                calc_func='max',                             # 计算函数：调用max方法
                inv_check_has_mapping=True,                  # 反向检查是否有映射
                tags=['generic', 'describe']                 # 标签：通用和描述相关
            ),
            # 最小值索引指标
            idx_min=dict(
                title='Min Index',                           # 指标标题：最小值索引
                calc_func='idxmin',                          # 计算函数：调用idxmin方法
                agg_func=None,                               # 聚合函数：无需聚合
                inv_check_has_mapping=True,                  # 反向检查是否有映射
                tags=['generic', 'index']                    # 标签：通用和索引相关
            ),
            # 最大值索引指标
            idx_max=dict(
                title='Max Index',                           # 指标标题：最大值索引
                calc_func='idxmax',                          # 计算函数：调用idxmax方法
                agg_func=None,                               # 聚合函数：无需聚合
                inv_check_has_mapping=True,                  # 反向检查是否有映射
                tags=['generic', 'index']                    # 标签：通用和索引相关
            ),
            # 值计数指标
            value_counts=dict(
                title='Value Counts',                        # 指标标题：值计数
                calc_func=lambda value_counts: reshape_fns.to_dict(value_counts, orient='index_series'), # 计算函数：转换为字典
                resolve_value_counts=True,                   # 解析值计数
                check_has_mapping=True,                      # 检查是否有映射
                tags=['generic', 'value_counts']             # 标签：通用和值计数相关
            )
        ),
        copy_kwargs=dict(copy_mode='deep')                   # 深拷贝配置
    )

    @property
    def metrics(self) -> Config:
        """
        统计指标配置属性 - 返回可用的统计指标配置
        
        这个属性返回所有可用的统计指标配置，包括基本统计指标、
        索引指标和值计数指标。
        
        返回：
            Config: 统计指标配置对象
        
        可用指标：
        - start/end/period: 时间相关指标
        - count/mean/std/min/median/max: 基本统计指标
        - idx_min/idx_max: 索引指标
        - value_counts: 值计数指标
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        import numpy as np
        
        # 示例1：查看可用指标
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 4, 6, 8, 10]
        })
        
        metrics = df.vbt.metrics
        print("可用的统计指标:")
        for name, config in metrics.items():
            print(f"  {name}: {config.get('title', name)}")
        
        # 示例2：按标签分类指标
        describe_metrics = [name for name, config in metrics.items() 
                          if 'describe' in config.get('tags', [])]
        print("描述性统计指标:", describe_metrics)
        
        wrapper_metrics = [name for name, config in metrics.items() 
                         if 'wrapper' in config.get('tags', [])]
        print("包装器相关指标:", wrapper_metrics)
        
        # 示例3：使用特定指标
        selected_stats = df.vbt.stats(metrics=['mean', 'std', 'min', 'max'])
        print("选定指标的统计结果:")
        print(selected_stats)
        
        # 示例4：金融数据指标应用
        np.random.seed(42)
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.002, 0.025, 252),
            'MSFT': np.random.normal(0.0015, 0.022, 252)
        })
        
        # 计算金融相关指标
        finance_stats = returns.vbt.stats(
            metrics=['count', 'mean', 'std', 'min', 'max']
        )
        print("金融数据统计指标:")
        print(finance_stats)
        
        # 示例5：时间序列指标
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        ts = pd.Series(np.random.randn(100), index=dates)
        
        time_stats = ts.vbt.stats(metrics=['start', 'end', 'period'])
        print("时间序列指标:")
        print(time_stats)
        ```
        
        注意：
        - 不同指标有不同的计算函数和标签
        - 可以根据标签筛选相关指标
        - 指标配置用于stats()方法的内部处理
        """
        return self._metrics

    # ############# 转换方法 ############# #

    def drawdown(self, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        回撤序列计算 - 计算相对于扩展最大值的回撤
        
        这个方法计算每个时间点相对于历史最大值的回撤比例。
        回撤是量化金融中衡量投资组合风险的重要指标。
        
        参数：
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
        
        返回：
            tp.SeriesFrame: 回撤序列（负值表示回撤）
        
        计算公式：
        drawdown = current_value / expanding_max - 1
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 示例1：简单价格序列的回撤
        prices = pd.Series([100, 105, 98, 95, 102, 108, 103])
        drawdown = prices.vbt.drawdown()
        print("回撤序列:")
        print(drawdown)
        
        # 示例2：股票价格回撤分析
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # 模拟股票价格
        returns = np.random.normal(0.001, 0.02, 252)
        prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)
        
        # 计算回撤
        stock_drawdown = prices.vbt.drawdown()
        print("股票回撤:")
        print(stock_drawdown.head())
        
        # 找出最大回撤
        max_drawdown = stock_drawdown.min()
        print(f"最大回撤: {max_drawdown:.2%}")
        
        # 示例3：多资产回撤分析
        portfolio = pd.DataFrame({
            'AAPL': 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 252)),
            'GOOGL': 100 * np.cumprod(1 + np.random.normal(0.0015, 0.025, 252)),
            'MSFT': 100 * np.cumprod(1 + np.random.normal(0.0012, 0.022, 252))
        }, index=dates)
        
        # 计算投资组合回撤
        portfolio_drawdown = portfolio.vbt.drawdown()
        print("投资组合回撤:")
        print(portfolio_drawdown.head())
        
        # 各资产的最大回撤
        max_drawdowns = portfolio_drawdown.min()
        print("各资产最大回撤:")
        print(max_drawdowns)
        
        # 示例4：回撤可视化
        import matplotlib.pyplot as plt
        
        # 绘制价格和回撤
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 价格图
        prices.plot(ax=ax1, title='Stock Price')
        ax1.set_ylabel('Price')
        
        # 回撤图
        stock_drawdown.plot(ax=ax2, title='Drawdown', color='red')
        ax2.fill_between(stock_drawdown.index, stock_drawdown.values, 0, alpha=0.3, color='red')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        
        plt.tight_layout()
        plt.show()
        
        # 示例5：回撤统计分析
        drawdown_stats = stock_drawdown.describe()
        print("回撤统计分析:")
        print(drawdown_stats)
        
        # 计算回撤大于5%的时间比例
        severe_drawdown_ratio = (stock_drawdown < -0.05).sum() / len(stock_drawdown)
        print(f"回撤超过5%的时间比例: {severe_drawdown_ratio:.2%}")
        
        # 示例6：回撤恢复分析
        # 找出回撤期间
        in_drawdown = stock_drawdown < -0.01  # 回撤超过1%
        
        # 计算回撤持续时间
        drawdown_periods = []
        start_idx = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                drawdown_periods.append(i - start_idx)
                start_idx = None
        
        if drawdown_periods:
            avg_drawdown_duration = np.mean(drawdown_periods)
            print(f"平均回撤持续时间: {avg_drawdown_duration:.1f} 天")
        
        # 示例7：风险管理应用
        # 设定风险阈值
        risk_threshold = -0.1  # 10%回撤阈值
        
        # 检查是否触发风险警告
        risk_alerts = stock_drawdown < risk_threshold
        if risk_alerts.any():
            alert_dates = stock_drawdown[risk_alerts].index
            print(f"风险警告日期: {alert_dates.tolist()}")
        
        # 示例8：策略性能评估
        # 计算回撤相关的风险指标
        def calculate_risk_metrics(drawdown_series):
            \"\"\"计算回撤风险指标\"\"\"
            return {
                'Max Drawdown': drawdown_series.min(),
                'Avg Drawdown': drawdown_series[drawdown_series < 0].mean(),
                'Drawdown Std': drawdown_series.std(),
                'Time in Drawdown': (drawdown_series < -0.01).sum() / len(drawdown_series)
            }
        
        risk_metrics = calculate_risk_metrics(stock_drawdown)
        print("风险指标:")
        for metric, value in risk_metrics.items():
            print(f"  {metric}: {value:.4f}")
        ```
        
        注意：
        - 回撤值为负数，表示相对于历史最高点的下跌幅度
        - 基于扩展最大值计算，反映了任何时点的历史最大损失
        - 常用于风险管理和策略评估
        - 可以与其他指标结合使用进行综合分析
        """
        # 计算回撤：当前值 / 扩展最大值 - 1
        out = self.to_2d_array() / nb.expanding_max_nb(self.to_2d_array()) - 1
        
        # 包装结果并返回
        return self.wrapper.wrap(out, group_by=False, **merge_dicts({}, wrap_kwargs))

    @property
    def ranges(self) -> Ranges:
        """
        范围记录属性 - 获取默认参数的范围记录
        
        这个属性是get_ranges()方法的快捷方式，使用默认参数。
        
        返回：
            Ranges: 范围记录对象
        
        使用示例：
        ```python
        import pandas as pd
        import vectorbt as vbt
        
        # 创建布尔序列
        signal = pd.Series([True, True, False, True, True, False])
        
        # 获取范围记录
        ranges = signal.vbt.ranges
        print("范围记录:")
        print(ranges.records_readable)
        ```
        """
        return self.get_ranges()

    def get_ranges(self, wrapper_kwargs: tp.KwargsLike = None, **kwargs) -> Ranges:
        """
        生成范围记录 - 从时间序列中识别和创建范围记录
        
        这个方法基于时间序列数据生成范围记录，用于分析连续的True值、
        非NaN值或其他条件的时间段。
        
        参数：
            wrapper_kwargs (tp.KwargsLike, 可选): 包装器参数
            **kwargs: 传递给Ranges.from_ts的额外参数
        
        返回：
            Ranges: 范围记录对象
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 示例1：基本范围识别
        signal = pd.Series([True, True, False, True, True, True, False])
        ranges = signal.vbt.get_ranges()
        print("基本范围:")
        print(ranges.records_readable)
        
        # 示例2：交易信号范围
        prices = pd.Series([100, 105, 98, 95, 102, 108, 103])
        buy_signal = prices.pct_change() > 0.02
        
        buy_ranges = buy_signal.vbt.get_ranges()
        print("买入信号范围:")
        print(buy_ranges.records_readable)
        
        # 示例3：趋势识别
        ma_short = prices.rolling(2).mean()
        ma_long = prices.rolling(3).mean()
        uptrend = ma_short > ma_long
        
        trend_ranges = uptrend.vbt.get_ranges()
        print("上升趋势范围:")
        print(trend_ranges.records_readable)
        
        # 示例4：自定义间隔值
        custom_data = pd.Series([1, 2, -1, 3, 4, -1, 5])
        custom_ranges = custom_data.vbt.get_ranges(gap_value=-1)
        print("自定义间隔值范围:")
        print(custom_ranges.records_readable)
        
        # 示例5：多列范围分析
        df = pd.DataFrame({
            'Signal_A': [True, True, False, True, False],
            'Signal_B': [False, True, True, False, True]
        })
        
        multi_ranges = df.vbt.get_ranges()
        print("多列范围:")
        print(multi_ranges.records_readable)
        
        # 示例6：金融应用 - 持仓期间
        positions = pd.Series([0, 1, 1, 0, 1, 1, 1, 0])
        position_ranges = positions.vbt.get_ranges()
        print("持仓期间:")
        print(position_ranges.records_readable)
        
        # 计算持仓统计
        print("持仓统计:")
        print(f"平均持仓时间: {position_ranges.avg_duration()}")
        print(f"最长持仓时间: {position_ranges.max_duration()}")
        print(f"持仓覆盖率: {position_ranges.coverage():.2%}")
        ```
        
        注意：
        - 范围记录用于分析连续状态的时间段
        - 支持多种数据类型和自定义间隔值
        - 生成的范围对象提供丰富的分析方法
        """
        # 合并包装器参数
        wrapper_kwargs = merge_dicts(self.wrapper.config, wrapper_kwargs)
        
        # 从时间序列创建范围记录
        return Ranges.from_ts(self.obj, wrapper_kwargs=wrapper_kwargs, **kwargs)

    @property
    def drawdowns(self) -> Drawdowns:
        """
        回撤记录属性 - 获取默认参数的回撤记录
        
        这个属性是get_drawdowns()方法的快捷方式，使用默认参数。
        
        返回：
            Drawdowns: 回撤记录对象
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 创建价格序列
        prices = pd.Series([100, 105, 98, 95, 102, 108, 103])
        
        # 获取回撤记录
        drawdowns = prices.vbt.drawdowns
        print("回撤记录:")
        print(drawdowns.records_readable)
        ```
        """
        return self.get_drawdowns()

    def get_drawdowns(self, wrapper_kwargs: tp.KwargsLike = None, **kwargs) -> Drawdowns:
        """
        生成回撤记录 - 从时间序列中识别和创建回撤记录
        
        这个方法基于价格时间序列自动识别回撤，创建详细的回撤记录，
        包括峰值、谷底、恢复点等信息。
        
        参数：
            wrapper_kwargs (tp.KwargsLike, 可选): 包装器参数
            **kwargs: 传递给Drawdowns.from_ts的额外参数
        
        返回：
            Drawdowns: 回撤记录对象
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 示例1：基本回撤记录
        prices = pd.Series([100, 105, 98, 95, 102, 108, 103])
        drawdowns = prices.vbt.get_drawdowns()
        print("基本回撤记录:")
        print(drawdowns.records_readable)
        
        # 示例2：股票回撤分析
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns = np.random.normal(0.001, 0.02, 252)
        stock_price = pd.Series(100 * np.cumprod(1 + returns), index=dates)
        
        stock_drawdowns = stock_price.vbt.get_drawdowns()
        print("股票回撤统计:")
        print(f"最大回撤: {stock_drawdowns.max_drawdown():.2%}")
        print(f"平均回撤: {stock_drawdowns.avg_drawdown():.2%}")
        print(f"回撤次数: {stock_drawdowns.count()}")
        
        # 示例3：多资产回撤分析
        portfolio = pd.DataFrame({
            'AAPL': 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100)),
            'GOOGL': 100 * np.cumprod(1 + np.random.normal(0.0015, 0.025, 100)),
            'MSFT': 100 * np.cumprod(1 + np.random.normal(0.0012, 0.022, 100))
        })
        
        portfolio_drawdowns = portfolio.vbt.get_drawdowns()
        
        # 各资产回撤分析
        for col in portfolio.columns:
            dd = portfolio_drawdowns[col]
            print(f"{col} - 最大回撤: {dd.max_drawdown():.2%}")
        
        # 示例4：回撤可视化
        fig = stock_drawdowns.plot(top_n=5)
        fig.show()
        
        # 示例5：活跃回撤监控
        active_dd = stock_drawdowns.active_drawdown()
        if active_dd is not None:
            print(f"当前活跃回撤: {active_dd:.2%}")
        
        # 示例6：回撤恢复分析
        recovered_dd = stock_drawdowns.recovered
        if recovered_dd.count() > 0:
            avg_recovery = recovered_dd.avg_recovery_return()
            print(f"平均恢复收益率: {avg_recovery:.2%}")
        
        # 示例7：自定义包装器参数
        custom_drawdowns = stock_price.vbt.get_drawdowns(
            wrapper_kwargs=dict(freq='D'),
            attach_ts=True
        )
        
        # 示例8：风险管理应用
        risk_threshold = -0.1
        severe_drawdowns = stock_drawdowns.apply_mask(
            stock_drawdowns.drawdown < risk_threshold
        )
        
        if severe_drawdowns.count() > 0:
            print(f"严重回撤次数: {severe_drawdowns.count()}")
            print(f"平均严重回撤持续时间: {severe_drawdowns.avg_duration()}")
        ```
        
        注意：
        - 回撤记录包含峰值、谷底、恢复点等详细信息
        - 支持活跃回撤和已恢复回撤的分析
        - 提供丰富的统计指标和可视化功能
        - 是量化金融风险管理的重要工具
        """
        # 合并包装器参数
        wrapper_kwargs = merge_dicts(self.wrapper.config, wrapper_kwargs)
        
        # 从时间序列创建回撤记录
        return Drawdowns.from_ts(self.obj, wrapper_kwargs=wrapper_kwargs, **kwargs)

    def to_mapped(self,
                  dropna: bool = True,
                  dtype: tp.Optional[tp.DTypeLike] = None,
                  group_by: tp.GroupByLike = None,
                  **kwargs) -> MappedArray:
        """
        转换为映射数组 - 将对象转换为MappedArray实例
        
        这个方法将DataFrame或Series转换为MappedArray，这是一种高效的
        存储和处理稀疏数据的格式，特别适用于包含大量NaN值的数据。
        
        参数：
            dropna (bool, 可选): 是否删除NaN值，默认True
            dtype (tp.Optional[tp.DTypeLike], 可选): 输出数据类型
            group_by (tp.GroupByLike, 可选): 分组方式
            **kwargs: 传递给MappedArray构造函数的额外参数
        
        返回：
            MappedArray: 映射数组对象
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 示例1：基本映射数组转换
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [np.nan, 2, 3, np.nan, 5]
        })
        
        mapped = df.vbt.to_mapped()
        print("映射数组:")
        print(f"数据形状: {mapped.wrapper.shape}")
        print(f"非NaN值数量: {len(mapped.values)}")
        
        # 示例2：保留NaN值
        mapped_with_nan = df.vbt.to_mapped(dropna=False)
        print("保留NaN的映射数组:")
        print(f"总值数量: {len(mapped_with_nan.values)}")
        
        # 示例3：指定数据类型
        mapped_int = df.vbt.to_mapped(dtype=int)
        print("整数类型映射数组:")
        print(f"数据类型: {mapped_int.values.dtype}")
        
        # 示例4：分组映射
        grouped_mapped = df.vbt.to_mapped(group_by=['Group1', 'Group1'])
        print("分组映射数组:")
        print(f"分组后形状: {grouped_mapped.wrapper.shape}")
        
        # 示例5：稀疏数据处理
        # 创建稀疏数据
        sparse_data = pd.DataFrame(np.random.random((1000, 100)))
        sparse_data[sparse_data < 0.9] = np.nan  # 90%的数据为NaN
        
        sparse_mapped = sparse_data.vbt.to_mapped()
        print("稀疏数据映射:")
        print(f"原始数据大小: {sparse_data.size}")
        print(f"映射后非NaN值: {len(sparse_mapped.values)}")
        print(f"压缩比: {len(sparse_mapped.values) / sparse_data.size:.2%}")
        
        # 示例6：金融数据应用
        # 创建股票收益率数据（包含缺失值）
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.0015, 0.025, 252),
            'MSFT': np.random.normal(0.0012, 0.022, 252)
        }, index=dates)
        
        # 随机插入缺失值
        mask = np.random.random(returns.shape) < 0.05
        returns[mask] = np.nan
        
        # 转换为映射数组
        returns_mapped = returns.vbt.to_mapped()
        print("收益率映射数组:")
        print(f"完整收益率数据点: {len(returns_mapped.values)}")
        
        # 示例7：时间序列数据映射
        ts = pd.Series(np.random.randn(100))
        ts[ts < 0] = np.nan  # 只保留正值
        
        ts_mapped = ts.vbt.to_mapped()
        print("时间序列映射:")
        print(f"正值数据点: {len(ts_mapped.values)}")
        
        # 示例8：映射数组操作
        # 使用映射数组进行统计计算
        mapped_stats = mapped.reduce(np.mean)
        print("映射数组统计:")
        print(mapped_stats)
        
        # 示例9：内存效率对比
        import sys
        
        # 比较内存使用
        original_memory = sys.getsizeof(sparse_data)
        mapped_memory = sys.getsizeof(sparse_mapped.values) + sys.getsizeof(sparse_mapped.col_arr)
        
        print("内存使用对比:")
        print(f"原始数据: {original_memory:,} bytes")
        print(f"映射数组: {mapped_memory:,} bytes")
        print(f"内存节省: {(1 - mapped_memory/original_memory)*100:.1f}%")
        
        # 示例10：自定义参数
        custom_mapped = df.vbt.to_mapped(
            dropna=True,
            dtype=np.float32,
            group_by=None,
            idx_arr=None  # 自定义索引数组
        )
        print("自定义映射数组:")
        print(f"数据类型: {custom_mapped.values.dtype}")
        ```
        
        注意：
        - MappedArray特别适用于稀疏数据的高效存储
        - dropna=True可以显著减少内存使用
        - 支持分组和自定义数据类型
        - 映射数组保留了原始数据的结构信息
        """
        # 将数据扁平化为一维数组（按列优先顺序）
        mapped_arr = self.to_2d_array().flatten(order='F')
        
        # 创建列索引数组（重复列索引）
        col_arr = np.repeat(np.arange(self.wrapper.shape_2d[1]), self.wrapper.shape_2d[0])
        
        # 创建行索引数组（平铺行索引）
        idx_arr = np.tile(np.arange(self.wrapper.shape_2d[0]), self.wrapper.shape_2d[1])
        
        # 如果需要删除NaN值且数据中存在NaN
        if dropna and np.isnan(mapped_arr).any():
            not_nan_mask = ~np.isnan(mapped_arr)
            mapped_arr = mapped_arr[not_nan_mask]
            col_arr = col_arr[not_nan_mask]
            idx_arr = idx_arr[not_nan_mask]
        
        # 创建并返回MappedArray对象
        return MappedArray(
            self.wrapper,
            np.asarray(mapped_arr, dtype=dtype),
            col_arr,
            idx_arr=idx_arr,
            **kwargs
        ).regroup(group_by)

    def to_returns(self, **kwargs) -> tp.SeriesFrame:
        """
        转换为收益率 - 将价格数据转换为收益率序列
        
        这个方法将价格或价值数据转换为收益率序列，这是金融分析中
        的基本操作。
        
        参数：
            **kwargs: 传递给returns.from_value方法的参数
        
        返回：
            tp.SeriesFrame: 收益率序列
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 示例1：基本收益率转换
        prices = pd.Series([100, 105, 98, 102, 108])
        returns = prices.vbt.to_returns()
        print("基本收益率:")
        print(returns)
        
        # 示例2：股票收益率计算
        stock_prices = pd.DataFrame({
            'AAPL': [100, 105, 98, 102, 108, 103],
            'GOOGL': [200, 210, 195, 205, 215, 208],
            'MSFT': [150, 155, 148, 152, 158, 156]
        })
        
        stock_returns = stock_prices.vbt.to_returns()
        print("股票收益率:")
        print(stock_returns)
        
        # 示例3：对数收益率
        log_returns = stock_prices.vbt.to_returns(log_returns=True)
        print("对数收益率:")
        print(log_returns.head())
        
        # 示例4：年化收益率
        annual_returns = stock_prices.vbt.to_returns(freq='252D')
        print("年化收益率:")
        print(annual_returns.head())
        ```
        
        注意：
        - 默认计算简单收益率
        - 支持对数收益率和年化收益率
        - 第一个值会是NaN（因为没有前一个值）
        """
        # 使用returns模块的from_value方法转换为收益率
        return self.obj.vbt.returns.from_value(self.obj, **kwargs).obj

    # ############# Crossover ############# #

    def crossed_above(self,
                      other: tp.SeriesFrame,
                      wait: int = 0,
                      broadcast_kwargs: tp.KwargsLike = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        向上交叉检测 - 检测数据向上穿越另一个数组的点位
        
        这个方法检测当前数据从下方向上穿越另一个数组的时间点。
        这在金融分析中常用于检测均线交叉、突破等交易信号。
        
        参数：
            other (tp.SeriesFrame): 被穿越的目标数组
            wait (int, 可选): 等待期，默认为0
                指定穿越后需要等待的时间段数才确认信号
            broadcast_kwargs (tp.KwargsLike, 可选): 广播参数
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
        
        返回：
            tp.SeriesFrame: 布尔数组，True表示发生向上穿越
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 示例1：基本向上穿越检测
        prices = pd.Series([100, 102, 105, 108, 106, 109, 111])
        ma = pd.Series([101, 103, 104, 107, 108, 108, 110])
        
        # 检测价格向上穿越移动平均线
        crossover = prices.vbt.crossed_above(ma)
        print("向上穿越信号:")
        print(crossover)
        
        # 示例2：双均线交叉系统
        short_ma = pd.Series([100, 101, 102, 104, 105, 107, 109])
        long_ma = pd.Series([101, 102, 103, 103, 104, 106, 108])
        
        # 短期均线向上穿越长期均线（金叉）
        golden_cross = short_ma.vbt.crossed_above(long_ma)
        print("金叉信号:")
        print(golden_cross)
        
        # 示例3：使用等待期避免假信号
        # 等待1期确认信号
        confirmed_cross = prices.vbt.crossed_above(ma, wait=1)
        print("确认后的穿越信号:")
        print(confirmed_cross)
        
        # 示例4：股票技术分析
        # 模拟股票价格和支撑阻力位
        stock_price = pd.Series([
            95, 97, 99, 101, 103, 105, 107, 109, 111, 113
        ])
        resistance = pd.Series([100] * 10)  # 阻力位
        
        # 检测价格向上突破阻力位
        breakout = stock_price.vbt.crossed_above(resistance)
        print("向上突破信号:")
        print(breakout)
        
        # 示例5：多资产交叉分析
        portfolio_data = pd.DataFrame({
            'Stock_A': [100, 102, 104, 106, 108, 110],
            'Stock_B': [101, 103, 105, 107, 109, 111],
            'Stock_C': [99, 101, 103, 105, 107, 109]
        })
        
        # 各股票与基准的交叉
        benchmark = pd.Series([102, 104, 106, 108, 110, 112])
        
        # 检测各股票向上穿越基准
        cross_signals = portfolio_data.vbt.crossed_above(benchmark)
        print("各股票向上穿越基准:")
        print(cross_signals)
        
        # 示例6：RSI超买超卖信号
        rsi_data = pd.Series([25, 35, 45, 55, 65, 75, 85, 75, 65, 55])
        oversold_level = pd.Series([30] * 10)  # 超卖线
        
        # RSI向上穿越超卖线（买入信号）
        buy_signal = rsi_data.vbt.crossed_above(oversold_level)
        print("RSI买入信号:")
        print(buy_signal)
        
        # 示例7：波动率突破策略
        volatility = pd.Series([0.15, 0.18, 0.22, 0.25, 0.28, 0.32, 0.35])
        vol_threshold = pd.Series([0.20] * 7)  # 波动率阈值
        
        # 波动率向上突破阈值
        vol_breakout = volatility.vbt.crossed_above(vol_threshold)
        print("波动率突破信号:")
        print(vol_breakout)
        
        # 示例8：成交量确认
        volume = pd.Series([1000, 1200, 1500, 1800, 2100, 2400, 2700])
        avg_volume = pd.Series([1300] * 7)  # 平均成交量
        
        # 成交量向上穿越平均成交量
        volume_confirm = volume.vbt.crossed_above(avg_volume)
        print("成交量确认信号:")
        print(volume_confirm)
        
        # 示例9：结合价格和成交量的综合信号
        # 同时满足价格突破和成交量放大
        combined_signal = breakout & volume_confirm
        print("综合突破信号:")
        print(combined_signal)
        
        # 示例10：回测应用
        # 生成交易信号并计算收益
        signals = golden_cross.astype(int)
        returns = short_ma.pct_change().fillna(0)
        
        # 计算策略收益
        strategy_returns = signals.shift(1) * returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        print("策略累计收益:")
        print(cumulative_returns.iloc[-1])
        ```
        
        注意：
        - 向上穿越定义为：前一期self <= other且当前期self > other
        - wait参数可以避免频繁的假信号
        - 结果会自动进行广播对齐
        - 第一个值始终为False（因为需要前一期数据）
        """
        self_obj, other_obj = reshape_fns.broadcast(self.obj, other, **resolve_dict(broadcast_kwargs))
        out = nb.crossed_above_nb(reshape_fns.to_2d_array(self_obj), reshape_fns.to_2d_array(other_obj), wait=wait)
        return ArrayWrapper.from_obj(self_obj).wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def crossed_below(self,
                      other: tp.SeriesFrame,
                      wait: int = 0,
                      broadcast_kwargs: tp.KwargsLike = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        向下交叉检测 - 检测数据向下穿越另一个数组的点位
        
        这个方法检测当前数据从上方向下穿越另一个数组的时间点。
        这在金融分析中常用于检测均线死叉、跌破支撑等交易信号。
        
        参数：
            other (tp.SeriesFrame): 被穿越的目标数组
            wait (int, 可选): 等待期，默认为0
                指定穿越后需要等待的时间段数才确认信号
            broadcast_kwargs (tp.KwargsLike, 可选): 广播参数
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
        
        返回：
            tp.SeriesFrame: 布尔数组，True表示发生向下穿越
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 示例1：基本向下穿越检测
        prices = pd.Series([111, 109, 106, 108, 105, 102, 100])
        ma = pd.Series([110, 108, 107, 107, 106, 104, 101])
        
        # 检测价格向下穿越移动平均线
        crossunder = prices.vbt.crossed_below(ma)
        print("向下穿越信号:")
        print(crossunder)
        
        # 示例2：双均线交叉系统
        short_ma = pd.Series([109, 107, 104, 102, 100, 98, 96])
        long_ma = pd.Series([108, 106, 105, 104, 103, 101, 99])
        
        # 短期均线向下穿越长期均线（死叉）
        death_cross = short_ma.vbt.crossed_below(long_ma)
        print("死叉信号:")
        print(death_cross)
        
        # 示例3：使用等待期避免假信号
        # 等待1期确认信号
        confirmed_cross = prices.vbt.crossed_below(ma, wait=1)
        print("确认后的穿越信号:")
        print(confirmed_cross)
        
        # 示例4：股票技术分析
        # 模拟股票价格和支撑位
        stock_price = pd.Series([
            113, 111, 109, 107, 105, 103, 101, 99, 97, 95
        ])
        support = pd.Series([100] * 10)  # 支撑位
        
        # 检测价格向下跌破支撑位
        breakdown = stock_price.vbt.crossed_below(support)
        print("向下跌破信号:")
        print(breakdown)
        
        # 示例5：多资产交叉分析
        portfolio_data = pd.DataFrame({
            'Stock_A': [110, 108, 106, 104, 102, 100],
            'Stock_B': [111, 109, 107, 105, 103, 101],
            'Stock_C': [109, 107, 105, 103, 101, 99]
        })
        
        # 各股票与基准的交叉
        benchmark = pd.Series([108, 106, 104, 102, 100, 98])
        
        # 检测各股票向下穿越基准
        cross_signals = portfolio_data.vbt.crossed_below(benchmark)
        print("各股票向下穿越基准:")
        print(cross_signals)
        
        # 示例6：RSI超买超卖信号
        rsi_data = pd.Series([75, 65, 55, 45, 35, 25, 15, 25, 35, 45])
        overbought_level = pd.Series([70] * 10)  # 超买线
        
        # RSI向下穿越超买线（卖出信号）
        sell_signal = rsi_data.vbt.crossed_below(overbought_level)
        print("RSI卖出信号:")
        print(sell_signal)
        
        # 示例7：止损信号
        portfolio_value = pd.Series([10000, 9800, 9600, 9400, 9200, 9000, 8800])
        stop_loss = pd.Series([9500] * 7)  # 止损线
        
        # 组合价值向下穿越止损线
        stop_signal = portfolio_value.vbt.crossed_below(stop_loss)
        print("止损信号:")
        print(stop_signal)
        
        # 示例8：波动率回落信号
        volatility = pd.Series([0.35, 0.32, 0.28, 0.25, 0.22, 0.18, 0.15])
        vol_threshold = pd.Series([0.30] * 7)  # 波动率阈值
        
        # 波动率向下穿越阈值
        vol_decline = volatility.vbt.crossed_below(vol_threshold)
        print("波动率回落信号:")
        print(vol_decline)
        
        # 示例9：成交量萎缩信号
        volume = pd.Series([2700, 2400, 2100, 1800, 1500, 1200, 1000])
        avg_volume = pd.Series([2000] * 7)  # 平均成交量
        
        # 成交量向下穿越平均成交量
        volume_decline = volume.vbt.crossed_below(avg_volume)
        print("成交量萎缩信号:")
        print(volume_decline)
        
        # 示例10：综合卖出信号
        # 同时满足价格跌破和成交量萎缩
        combined_sell = breakdown & volume_decline
        print("综合卖出信号:")
        print(combined_sell)
        
        # 示例11：风险管理应用
        # 计算最大回撤信号
        cumulative_returns = pd.Series([1.0, 1.05, 1.03, 1.01, 0.98, 0.95, 0.92])
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        
        # 回撤超过阈值
        drawdown_threshold = pd.Series([-0.05] * 7)  # -5%回撤阈值
        risk_signal = drawdown.vbt.crossed_below(drawdown_threshold)
        print("风险警告信号:")
        print(risk_signal)
        
        # 示例12：回测应用
        # 生成交易信号并计算收益
        signals = death_cross.astype(int) * -1  # 死叉做空
        returns = short_ma.pct_change().fillna(0)
        
        # 计算策略收益
        strategy_returns = signals.shift(1) * returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        print("策略累计收益:")
        print(cumulative_returns.iloc[-1])
        ```
        
        注意：
        - 向下穿越定义为：前一期self >= other且当前期self < other
        - wait参数可以避免频繁的假信号
        - 结果会自动进行广播对齐
        - 第一个值始终为False（因为需要前一期数据）
        - 与crossed_above相反，用于检测向下穿越
        """
        self_obj, other_obj = reshape_fns.broadcast(self.obj, other, **resolve_dict(broadcast_kwargs))
        out = nb.crossed_above_nb(reshape_fns.to_2d_array(other_obj), reshape_fns.to_2d_array(self_obj), wait=wait)
        return ArrayWrapper.from_obj(self_obj).wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    # ############# Transformation ############# #

    def transform(self, transformer: TransformerT, wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.SeriesFrame:
        """
        数据变换方法 - 使用变换器对数据进行转换
        
        这个方法使用scikit-learn风格的变换器对数据进行变换，支持各种预处理操作，
        如标准化、归一化、主成分分析等。变换器会自动检测是否已拟合，如果未拟合会先拟合。
        
        参数：
            transformer (TransformerT): 变换器实例
                必须具有transform和fit_transform方法，理想情况下继承自
                sklearn.base.TransformerMixin和sklearn.base.BaseEstimator
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
            **kwargs: 传递给transform或fit_transform方法的额外参数
        
        返回：
            tp.SeriesFrame: 变换后的数据
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        from sklearn.preprocessing import (
            MinMaxScaler, StandardScaler, RobustScaler, 
            PowerTransformer, QuantileTransformer
        )
        from sklearn.decomposition import PCA
        
        # 创建示例数据
        np.random.seed(42)
        data = pd.DataFrame({
            'A': np.random.normal(100, 20, 100),
            'B': np.random.normal(50, 10, 100),
            'C': np.random.normal(200, 50, 100)
        })
        
        # 示例1：MinMax标准化
        scaler = MinMaxScaler((-1, 1))
        scaled_data = data.vbt.transform(scaler)
        print("MinMax标准化:")
        print(scaled_data.head())
        
        # 示例2：Z-score标准化
        std_scaler = StandardScaler()
        standardized_data = data.vbt.transform(std_scaler)
        print("Z-score标准化:")
        print(standardized_data.head())
        
        # 示例3：鲁棒标准化（对异常值不敏感）
        robust_scaler = RobustScaler()
        robust_data = data.vbt.transform(robust_scaler)
        print("鲁棒标准化:")
        print(robust_data.head())
        
        # 示例4：使用预拟合的变换器
        # 在训练集上拟合变换器
        train_data = data.iloc[:80]
        test_data = data.iloc[80:]
        
        fitted_scaler = MinMaxScaler().fit(train_data)
        
        # 对测试集应用相同的变换
        test_scaled = test_data.vbt.transform(fitted_scaler)
        print("预拟合变换器应用:")
        print(test_scaled.head())
        
        # 示例5：Power变换（提高正态性）
        power_transformer = PowerTransformer(method='yeo-johnson')
        power_data = data.vbt.transform(power_transformer)
        print("Power变换:")
        print(power_data.head())
        
        # 示例6：分位数变换（转换为均匀分布）
        quantile_transformer = QuantileTransformer(output_distribution='uniform')
        quantile_data = data.vbt.transform(quantile_transformer)
        print("分位数变换:")
        print(quantile_data.head())
        
        # 示例7：主成分分析
        pca = PCA(n_components=2)
        pca_data = data.vbt.transform(pca)
        print("主成分分析:")
        print(pca_data.head())
        
        # 示例8：金融数据应用
        # 股票收益率标准化
        stock_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.002, 0.025, 252),
            'MSFT': np.random.normal(0.0015, 0.022, 252)
        })
        
        # 标准化收益率用于风险模型
        returns_scaler = StandardScaler()
        normalized_returns = stock_returns.vbt.transform(returns_scaler)
        print("标准化收益率:")
        print(normalized_returns.head())
        
        # 示例9：时间序列特征工程
        # 创建滑动窗口特征
        from sklearn.preprocessing import PolynomialFeatures
        
        # 生成多项式特征
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = data.vbt.transform(poly)
        print("多项式特征:")
        print(poly_features.head())
        
        # 示例10：自定义变换器
        class LogTransformer:
            def __init__(self, base=np.e):
                self.base = base
            
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                return np.log(X) / np.log(self.base)
            
            def fit_transform(self, X, y=None):
                return self.fit(X, y).transform(X)
        
        # 应用自定义变换器
        log_transformer = LogTransformer(base=2)
        log_data = (data + 1).vbt.transform(log_transformer)  # +1避免log(0)
        print("自定义对数变换:")
        print(log_data.head())
        
        # 示例11：管道式变换
        from sklearn.pipeline import Pipeline
        
        # 创建变换管道
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2))
        ])
        
        pipeline_data = data.vbt.transform(pipeline)
        print("管道式变换:")
        print(pipeline_data.head())
        
        # 示例12：条件变换
        # 只对特定列应用变换
        specific_columns = ['A', 'B']
        specific_data = data[specific_columns].vbt.transform(MinMaxScaler())
        print("特定列变换:")
        print(specific_data.head())
        
        # 示例13：回测中的变换应用
        # 滚动窗口变换
        window_size = 30
        rolling_transformed = []
        
        for i in range(window_size, len(data)):
            window_data = data.iloc[i-window_size:i]
            scaler = StandardScaler().fit(window_data)
            current_data = data.iloc[i:i+1]
            transformed = current_data.vbt.transform(scaler)
            rolling_transformed.append(transformed)
        
        # 示例14：异常值处理
        from sklearn.preprocessing import RobustScaler
        
        # 添加异常值
        data_with_outliers = data.copy()
        data_with_outliers.iloc[10] = [1000, 1000, 1000]  # 异常值
        
        # 使用鲁棒标准化处理异常值
        robust_transformed = data_with_outliers.vbt.transform(RobustScaler())
        print("异常值处理:")
        print(robust_transformed.iloc[8:12])
        
        # 示例15：特征选择变换
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # 假设有目标变量
        target = (data['A'] > data['A'].median()).astype(int)
        
        # 选择最佳特征
        selector = SelectKBest(score_func=f_classif, k=2)
        selector.fit(data, target)
        
        selected_features = data.vbt.transform(selector)
        print("特征选择:")
        print(selected_features.head())
        ```
        
        注意：
        - 变换器会自动检测是否已拟合，如果未拟合会先调用fit_transform
        - 如果已拟合，直接调用transform方法
        - 支持所有scikit-learn风格的变换器
        - 变换后的数据保持原始索引和列名结构
        - 可用于数据预处理、特征工程、降维等任务
        """
        is_fitted = True
        try:
            check_is_fitted(transformer)
        except NotFittedError:
            is_fitted = False
        if not is_fitted:
            result = transformer.fit_transform(self.to_2d_array(), **kwargs)
        else:
            result = transformer.transform(self.to_2d_array(), **kwargs)
        return self.wrapper.wrap(result, group_by=False, **merge_dicts({}, wrap_kwargs))

    def zscore(self, **kwargs) -> tp.SeriesFrame:
        """
        Z-score标准化 - 计算数据的标准化分数
        
        这个方法使用StandardScaler计算数据的Z-score，将数据转换为均值为0、
        标准差为1的标准正态分布。这是数据预处理中最常用的标准化方法。
        
        参数：
            **kwargs: 传递给scale方法的额外参数
        
        返回：
            tp.SeriesFrame: 标准化后的数据（均值为0，标准差为1）
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 示例1：基本Z-score标准化
        data = pd.DataFrame({
            'A': [100, 120, 80, 90, 110],
            'B': [200, 250, 150, 180, 220],
            'C': [10, 12, 8, 9, 11]
        })
        
        # 计算Z-score
        z_scores = data.vbt.zscore()
        print("Z-score标准化:")
        print(z_scores)
        
        # 验证均值和标准差
        print("均值:", z_scores.mean())
        print("标准差:", z_scores.std())
        
        # 示例2：股票收益率标准化
        np.random.seed(42)
        stock_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.002, 0.025, 252),
            'MSFT': np.random.normal(0.0015, 0.022, 252)
        })
        
        # 标准化收益率
        standardized_returns = stock_returns.vbt.zscore()
        print("标准化收益率:")
        print(standardized_returns.head())
        
        # 示例3：时间序列异常值检测
        # 生成带异常值的时间序列
        ts = pd.Series(np.random.normal(0, 1, 100))
        ts[50] = 5  # 插入异常值
        
        # 计算Z-score
        z_scores = ts.vbt.zscore()
        
        # 检测异常值（|Z-score| > 2）
        outliers = np.abs(z_scores) > 2
        print("异常值检测:")
        print(f"异常值数量: {outliers.sum()}")
        print(f"异常值位置: {outliers[outliers].index.tolist()}")
        
        # 示例4：投资组合风险分析
        # 计算各资产的标准化风险
        risk_metrics = pd.DataFrame({
            'Volatility': [0.15, 0.22, 0.18, 0.25, 0.20],
            'Beta': [1.2, 0.8, 1.0, 1.5, 1.1],
            'Sharpe': [0.8, 1.2, 0.9, 0.6, 1.0]
        }, index=['Stock_A', 'Stock_B', 'Stock_C', 'Stock_D', 'Stock_E'])
        
        # 标准化风险指标
        standardized_risk = risk_metrics.vbt.zscore()
        print("标准化风险指标:")
        print(standardized_risk)
        
        # 示例5：特征工程
        # 多个特征的标准化
        features = pd.DataFrame({
            'Price': [100, 110, 95, 105, 120],
            'Volume': [1000, 1200, 800, 1100, 1300],
            'MarketCap': [1000000, 1100000, 950000, 1050000, 1200000]
        })
        
        # 标准化特征
        standardized_features = features.vbt.zscore()
        print("标准化特征:")
        print(standardized_features)
        
        # 示例6：相关性分析准备
        # 标准化后的数据更适合相关性分析
        corr_matrix = standardized_returns.corr()
        print("标准化后的相关性矩阵:")
        print(corr_matrix)
        
        # 示例7：机器学习预处理
        # 准备机器学习的输入数据
        ml_features = pd.DataFrame({
            'Feature1': np.random.normal(50, 20, 100),
            'Feature2': np.random.normal(100, 30, 100),
            'Feature3': np.random.normal(10, 5, 100)
        })
        
        # 标准化特征
        ml_standardized = ml_features.vbt.zscore()
        print("机器学习特征标准化:")
        print(ml_standardized.head())
        
        # 示例8：滚动标准化
        # 滚动窗口标准化
        rolling_data = pd.Series(np.random.randn(100).cumsum())
        window_size = 20
        
        # 滚动标准化
        rolling_standardized = rolling_data.rolling(window_size).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std()
        )
        print("滚动标准化:")
        print(rolling_standardized.tail())
        
        # 示例9：多时间序列标准化
        multi_ts = pd.DataFrame({
            'Series1': np.random.normal(100, 15, 50),
            'Series2': np.random.normal(200, 30, 50),
            'Series3': np.random.normal(50, 10, 50)
        })
        
        # 标准化多个时间序列
        multi_standardized = multi_ts.vbt.zscore()
        print("多时间序列标准化:")
        print(multi_standardized.head())
        
        # 示例10：统计测试准备
        # 为统计测试准备标准化数据
        sample_data = pd.DataFrame({
            'Group_A': np.random.normal(100, 20, 30),
            'Group_B': np.random.normal(110, 25, 30),
            'Group_C': np.random.normal(95, 15, 30)
        })
        
        # 标准化样本数据
        standardized_samples = sample_data.vbt.zscore()
        print("标准化样本数据:")
        print(standardized_samples.describe())
        ```
        
        注意：
        - Z-score = (x - mean) / std
        - 标准化后的数据均值为0，标准差为1
        - 适用于正态分布或近似正态分布的数据
        - 对异常值敏感，异常值会显著影响均值和标准差
        - 常用于机器学习、统计分析和异常值检测
        """
        return self.scale(with_mean=True, with_std=True, **kwargs)

    def rebase(self, base: float, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        重新基准化 - 将所有序列重新调整到给定的初始基准值
        
        这个方法将所有时间序列重新调整到相同的起始基准值，使得不同规模的
        序列能够在同一图表上进行比较。方法会自动处理NaN值。
        
        参数：
            base (float): 新的基准值，所有序列的第一个有效值将被调整到此值
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
        
        返回：
            tp.SeriesFrame: 重新基准化后的数据
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 示例1：基本重新基准化
        data = pd.DataFrame({
            'Stock_A': [100, 105, 98, 102, 108],
            'Stock_B': [50, 52, 48, 51, 53],
            'Stock_C': [200, 210, 195, 205, 220]
        })
        
        # 重新基准化到100
        rebased = data.vbt.rebase(100)
        print("重新基准化到100:")
        print(rebased)
        
        # 示例2：股票价格比较
        # 不同价格水平的股票
        stock_prices = pd.DataFrame({
            'AAPL': [150, 155, 148, 160, 165],
            'GOOGL': [2800, 2850, 2750, 2900, 2950],
            'MSFT': [300, 310, 295, 315, 320]
        })
        
        # 重新基准化到100便于比较
        rebased_stocks = stock_prices.vbt.rebase(100)
        print("股票价格重新基准化:")
        print(rebased_stocks)
        
        # 计算相对表现
        performance = (rebased_stocks.iloc[-1] - 100) / 100
        print("相对表现:")
        print(performance)
        
        # 示例3：投资组合比较
        # 不同规模的投资组合
        portfolios = pd.DataFrame({
            'Portfolio_A': [10000, 10200, 9800, 10500, 10800],
            'Portfolio_B': [50000, 51000, 49500, 52000, 53500],
            'Portfolio_C': [100000, 102000, 98000, 105000, 108000]
        })
        
        # 重新基准化到1000
        rebased_portfolios = portfolios.vbt.rebase(1000)
        print("投资组合重新基准化:")
        print(rebased_portfolios)
        
        # 示例4：处理缺失值
        # 包含NaN的数据
        data_with_nan = pd.DataFrame({
            'Series1': [np.nan, 100, 105, 98, 102],
            'Series2': [200, np.nan, 210, 195, 205],
            'Series3': [50, 52, np.nan, 51, 53]
        })
        
        # 重新基准化（会自动处理NaN）
        rebased_nan = data_with_nan.vbt.rebase(100)
        print("处理NaN的重新基准化:")
        print(rebased_nan)
        
        # 示例5：指数比较
        # 不同基准值的指数
        indices = pd.DataFrame({
            'Index_A': [1000, 1050, 980, 1020, 1080],
            'Index_B': [5000, 5250, 4900, 5100, 5400],
            'Index_C': [2000, 2100, 1950, 2050, 2160]
        })
        
        # 重新基准化到1000
        rebased_indices = indices.vbt.rebase(1000)
        print("指数重新基准化:")
        print(rebased_indices)
        
        # 示例6：汇率比较
        # 不同汇率的比较
        exchange_rates = pd.DataFrame({
            'USD_EUR': [0.85, 0.87, 0.83, 0.86, 0.88],
            'USD_JPY': [110, 112, 108, 111, 114],
            'USD_GBP': [0.75, 0.77, 0.73, 0.76, 0.78]
        })
        
        # 重新基准化到1（相对变化）
        rebased_fx = exchange_rates.vbt.rebase(1)
        print("汇率重新基准化:")
        print(rebased_fx)
        
        # 示例7：经济指标比较
        economic_data = pd.DataFrame({
            'GDP': [20000, 20200, 19800, 20500, 20800],
            'Inflation': [2.0, 2.1, 1.9, 2.2, 2.3],
            'Unemployment': [5.0, 4.8, 5.2, 4.7, 4.5]
        })
        
        # 重新基准化到100
        rebased_econ = economic_data.vbt.rebase(100)
        print("经济指标重新基准化:")
        print(rebased_econ)
        
        # 示例8：时间序列可视化准备
        # 为绘图准备数据
        ts_data = pd.DataFrame({
            'High_Value': [1000, 1100, 950, 1200, 1150],
            'Medium_Value': [100, 110, 95, 120, 115],
            'Low_Value': [10, 11, 9.5, 12, 11.5]
        })
        
        # 重新基准化便于可视化
        rebased_viz = ts_data.vbt.rebase(100)
        print("可视化准备:")
        print(rebased_viz)
        
        # 示例9：相对强弱比较
        # 计算相对强弱指标
        sector_performance = pd.DataFrame({
            'Technology': [100, 108, 95, 115, 125],
            'Healthcare': [100, 103, 98, 108, 112],
            'Finance': [100, 105, 92, 110, 118]
        })
        
        # 重新基准化到100
        rebased_sectors = sector_performance.vbt.rebase(100)
        
        # 计算相对强弱
        relative_strength = rebased_sectors.div(rebased_sectors.mean(axis=1), axis=0) * 100
        print("相对强弱指标:")
        print(relative_strength)
        
        # 示例10：基准比较
        # 与基准的比较
        returns = pd.DataFrame({
            'Strategy': [100, 102, 104, 101, 106],
            'Benchmark': [100, 101, 103, 102, 105]
        })
        
        # 重新基准化
        rebased_returns = returns.vbt.rebase(100)
        
        # 计算相对收益
        relative_return = rebased_returns['Strategy'] - rebased_returns['Benchmark']
        print("相对收益:")
        print(relative_return)
        
        # 示例11：滚动基准化
        # 滚动重新基准化
        long_series = pd.Series(np.random.randn(100).cumsum() + 100)
        window_size = 20
        
        # 滚动基准化
        rolling_rebased = long_series.rolling(window_size).apply(
            lambda x: (x / x.iloc[0]) * 100
        )
        print("滚动基准化:")
        print(rolling_rebased.tail())
        
        # 示例12：多资产组合分析
        # 不同资产类别的比较
        asset_classes = pd.DataFrame({
            'Stocks': [1000, 1080, 950, 1150, 1200],
            'Bonds': [500, 510, 495, 520, 530],
            'Commodities': [200, 220, 180, 240, 250],
            'Real_Estate': [800, 840, 780, 880, 920]
        })
        
        # 重新基准化到100
        rebased_assets = asset_classes.vbt.rebase(100)
        print("多资产类别比较:")
        print(rebased_assets)
        
        # 计算最佳表现资产
        best_performer = rebased_assets.iloc[-1].idxmax()
        best_return = rebased_assets.iloc[-1].max() - 100
        print(f"最佳表现资产: {best_performer}, 收益率: {best_return:.2f}%")
        ```
        
        注意：
        - 重新基准化公式：rebased = (data / first_valid_value) * base
        - 会自动进行前向填充和后向填充处理NaN值
        - 使得不同规模的序列能够在同一尺度上比较
        - 常用于投资组合分析、相对表现比较和数据可视化
        - 保持时间序列的相对变化特征
        """
        result = nb.bfill_nb(nb.ffill_nb(self.to_2d_array()))
        result = result / result[0] * base
        return self.wrapper.wrap(result, group_by=False, **merge_dicts({}, wrap_kwargs))

    # ############# Splitting ############# #

    def split(self, splitter: SplitterT, stack_kwargs: tp.KwargsLike = None, keys: tp.Optional[tp.IndexLike] = None,
              plot: bool = False, trace_names: tp.TraceNames = None, heatmap_kwargs: tp.KwargsLike = None,
              **kwargs) -> SplitOutputT:
        """
        数据分割方法 - 使用分割器对数据进行分割
        
        这个方法使用分割器将数据分割成多个集合（如训练集、验证集、测试集），
        支持时间序列交叉验证和其他分割策略。
        
        参数：
            splitter (SplitterT): 分割器实例
                必须具有split方法，理想情况下继承自sklearn.model_selection.BaseCrossValidator
                或vectorbt.generic.splitters.BaseSplitter
            stack_kwargs (tp.KwargsLike, 可选): 堆叠参数
            keys (tp.Optional[tp.IndexLike], 可选): 分割的键值
            plot (bool, 可选): 是否绘制分割结果，默认False
            trace_names (tp.TraceNames, 可选): 追踪名称
            heatmap_kwargs (tp.KwargsLike, 可选): 热力图参数
            **kwargs: 传递给分割器的额外参数
        
        返回：
            SplitOutputT: 分割结果的元组，每个元组包含数据框和分割索引
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        from sklearn.model_selection import TimeSeriesSplit, KFold
        
        # 创建示例时间序列数据
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
        
        # 示例1：时间序列分割
        splitter = TimeSeriesSplit(n_splits=3)
        (train_df, train_indexes), (test_df, test_indexes) = data.vbt.split(splitter)
        
        print("训练集:")
        print(train_df.head())
        print("测试集:")
        print(test_df.head())
        
        # 示例2：K折交叉验证
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = data.vbt.split(kfold)
        
        print(f"K折分割数量: {len(splits)}")
        
        # 示例3：股票数据回测分割
        stock_data = pd.DataFrame({
            'price': np.random.randn(252).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 252),
            'returns': np.random.normal(0.001, 0.02, 252)
        }, index=pd.date_range('2023-01-01', periods=252, freq='D'))
        
        # 使用时间序列分割进行回测
        ts_splitter = TimeSeriesSplit(n_splits=5, test_size=30)
        (train_data, train_idx), (test_data, test_idx) = stock_data.vbt.split(ts_splitter)
        
        print("股票数据分割:")
        print(f"训练集形状: {train_data.shape}")
        print(f"测试集形状: {test_data.shape}")
        
        # 示例4：自定义分割器
        from vectorbt.generic.splitters import RangeSplitter
        
        # 按日期范围分割
        range_splitter = RangeSplitter(
            ranges=[
                (0, 50),    # 前50天
                (50, 100),  # 后50天
            ]
        )
        
        range_splits = data.vbt.split(range_splitter)
        print("范围分割结果:")
        for i, (split_data, split_idx) in enumerate(range_splits):
            print(f"分割{i}: {len(split_data)} 个样本")
        
        # 示例5：滚动窗口分割
        from vectorbt.generic.splitters import RollingSplitter
        
        # 滚动窗口分割
        rolling_splitter = RollingSplitter(
            window_size=30,
            step=10
        )
        
        rolling_splits = data.vbt.split(rolling_splitter)
        print("滚动窗口分割:")
        for i, (split_data, split_idx) in enumerate(rolling_splits[:3]):
            print(f"窗口{i}: {len(split_data)} 个样本")
        
        # 示例6：扩展窗口分割
        from vectorbt.generic.splitters import ExpandingSplitter
        
        # 扩展窗口分割
        expanding_splitter = ExpandingSplitter(
            min_size=30,
            step=10
        )
        
        expanding_splits = data.vbt.split(expanding_splitter)
        print("扩展窗口分割:")
        for i, (split_data, split_idx) in enumerate(expanding_splits[:3]):
            print(f"扩展窗口{i}: {len(split_data)} 个样本")
        
        # 示例7：带绘图的分割
        splits_with_plot = data.vbt.split(
            TimeSeriesSplit(n_splits=3),
            plot=True
        )
        
        # 示例8：多资产数据分割
        multi_asset = pd.DataFrame({
            'Stock_A': np.random.randn(100).cumsum() + 100,
            'Stock_B': np.random.randn(100).cumsum() + 50,
            'Stock_C': np.random.randn(100).cumsum() + 200
        }, index=pd.date_range('2023-01-01', periods=100, freq='D'))
        
        # 分割多资产数据
        multi_splits = multi_asset.vbt.split(TimeSeriesSplit(n_splits=3))
        (multi_train, multi_train_idx), (multi_test, multi_test_idx) = multi_splits
        
        print("多资产分割:")
        print(f"训练集形状: {multi_train.shape}")
        print(f"测试集形状: {multi_test.shape}")
        
        # 示例9：策略回测中的分割
        # 使用分割进行策略验证
        strategy_results = []
        
        for (train_data, train_idx), (test_data, test_idx) in multi_splits:
            # 在训练集上计算信号
            train_signal = train_data.mean(axis=1).pct_change() > 0
            
            # 在测试集上应用策略
            test_returns = test_data.pct_change().mean(axis=1)
            strategy_return = (train_signal.iloc[-1] * test_returns).sum()
            
            strategy_results.append(strategy_return)
        
        print("策略回测结果:")
        print(f"平均收益: {np.mean(strategy_results):.4f}")
        print(f"收益标准差: {np.std(strategy_results):.4f}")
        
        # 示例10：交叉验证评估
        from sklearn.metrics import mean_squared_error
        
        # 模拟预测模型评估
        cv_scores = []
        
        for (train_data, train_idx), (test_data, test_idx) in multi_splits:
            # 简单的移动平均预测
            train_mean = train_data.mean()
            
            # 在测试集上预测
            predictions = np.full(len(test_data), train_mean)
            
            # 计算MSE
            mse = mean_squared_error(test_data, predictions)
            cv_scores.append(mse)
        
        print("交叉验证评估:")
        print(f"平均MSE: {np.mean(cv_scores):.4f}")
        print(f"MSE标准差: {np.std(cv_scores):.4f}")
        
        # 示例11：时间序列特征工程分割
        # 创建滞后特征
        feature_data = pd.DataFrame({
            'price': stock_data['price'],
            'price_lag1': stock_data['price'].shift(1),
            'price_lag2': stock_data['price'].shift(2),
            'returns': stock_data['returns'],
            'volume': stock_data['volume']
        }).dropna()
        
        # 分割特征数据
        feature_splits = feature_data.vbt.split(TimeSeriesSplit(n_splits=3))
        
        print("特征工程分割:")
        for i, (train_feat, test_feat) in enumerate(feature_splits):
            print(f"分割{i}: 训练特征 {train_feat.shape}, 测试特征 {test_feat.shape}")
        
        # 示例12：自定义键值分割
        custom_keys = ['train_set', 'validation_set', 'test_set']
        keyed_splits = data.vbt.split(
            TimeSeriesSplit(n_splits=3),
            keys=custom_keys
        )
        
        print("自定义键值分割:")
        for key, (split_data, split_idx) in zip(custom_keys, keyed_splits):
            print(f"{key}: {len(split_data)} 个样本")
        ```
        
        注意：
        - 分割操作会丢失日期时间索引格式，需要提前保存索引元数据
        - 返回的是包含数据框和分割索引的元组
        - 支持各种scikit-learn和vectorbt的分割器
        - 可以绘制分割结果的可视化图表
        - 适用于时间序列交叉验证和策略回测
        """
        total_range_sr = pd.Series(np.arange(len(self.wrapper.index)), index=self.wrapper.index)
        set_ranges = list(splitter.split(total_range_sr, **kwargs))
        if len(set_ranges) == 0:
            raise ValueError("No splits were generated")
        idxs_by_split_and_set = list(zip(*set_ranges))

        results = []
        if keys is not None:
            if not isinstance(keys, pd.Index):
                keys = pd.Index(keys)
        for idxs_by_split in idxs_by_split_and_set:
            split_dfs = []
            split_indexes = []
            for split_idx, idxs in enumerate(idxs_by_split):
                split_dfs.append(self.obj.iloc[idxs].reset_index(drop=True))
                if keys is not None:
                    split_name = keys[split_idx]
                else:
                    split_name = 'split_' + str(split_idx)
                split_indexes.append(pd.Index(self.wrapper.index[idxs], name=split_name))
            set_df = pd.concat(split_dfs, axis=1).reset_index(drop=True)
            if keys is not None:
                split_columns = keys
            else:
                split_columns = pd.Index(np.arange(len(split_indexes)), name='split_idx')
            split_columns = index_fns.repeat_index(split_columns, len(self.wrapper.columns))
            if stack_kwargs is None:
                stack_kwargs = {}
            set_df = set_df.vbt.stack_index(split_columns, **stack_kwargs)
            results.append((set_df, split_indexes))

        if plot:  # pragma: no cover
            if trace_names is None:
                trace_names = list(range(len(results)))
            if isinstance(trace_names, str):
                trace_names = [trace_names]
            nan_df = pd.DataFrame(np.nan, columns=pd.RangeIndex(stop=len(results[0][1])), index=self.wrapper.index)
            fig = None
            for i, (_, split_indexes) in enumerate(results):
                heatmap_df = nan_df.copy()
                for j in range(len(split_indexes)):
                    heatmap_df.loc[split_indexes[j], j] = i
                _heatmap_kwargs = resolve_dict(heatmap_kwargs, i=i)
                fig = heatmap_df.vbt.ts_heatmap(fig=fig, **merge_dicts(
                    dict(
                        trace_kwargs=dict(
                            showscale=False,
                            name=str(trace_names[i]),
                            showlegend=True
                        )
                    ),
                    _heatmap_kwargs
                ))
                if fig.layout.colorway is not None:
                    colorway = fig.layout.colorway
                else:
                    colorway = fig.layout.template.layout.colorway
                if 'colorscale' not in _heatmap_kwargs:
                    fig.data[-1].update(colorscale=[colorway[i], colorway[i]])
            return fig

        if len(results) == 1:
            return results[0]
        return tuple(results)

    def range_split(self, **kwargs) -> SplitOutputT:
        """Split using `GenericAccessor.split` on `vectorbt.generic.splitters.RangeSplitter`.

        Usage:
            ```pycon
            >>> range_df, range_indexes = sr.vbt.range_split(n=2)
            >>> range_df
            split_idx  0  1
            0          0  5
            1          1  6
            2          2  7
            3          3  8
            4          4  9
            >>> range_indexes
            [DatetimeIndex(['2020-01-01', ..., '2020-01-05'], dtype='datetime64[ns]', name='split_0'),
             DatetimeIndex(['2020-01-06', ..., '2020-01-10'], dtype='datetime64[ns]', name='split_1')]

            >>> range_df, range_indexes = sr.vbt.range_split(range_len=4)
            >>> range_df
            split_idx  0  1  2  3  4  5  6
            0          0  1  2  3  4  5  6
            1          1  2  3  4  5  6  7
            2          2  3  4  5  6  7  8
            3          3  4  5  6  7  8  9
            >>> range_indexes
            [DatetimeIndex(['2020-01-01', ..., '2020-01-04'], dtype='datetime64[ns]', name='split_0'),
             DatetimeIndex(['2020-01-02', ..., '2020-01-05'], dtype='datetime64[ns]', name='split_1'),
             DatetimeIndex(['2020-01-03', ..., '2020-01-06'], dtype='datetime64[ns]', name='split_2'),
             DatetimeIndex(['2020-01-04', ..., '2020-01-07'], dtype='datetime64[ns]', name='split_3'),
             DatetimeIndex(['2020-01-05', ..., '2020-01-08'], dtype='datetime64[ns]', name='split_4'),
             DatetimeIndex(['2020-01-06', ..., '2020-01-09'], dtype='datetime64[ns]', name='split_5'),
             DatetimeIndex(['2020-01-07', ..., '2020-01-10'], dtype='datetime64[ns]', name='split_6')]

            >>> range_df, range_indexes = sr.vbt.range_split(start_idxs=[0, 2], end_idxs=[5, 7])
            >>> range_df
            split_idx  0  1
            0          0  2
            1          1  3
            2          2  4
            3          3  5
            4          4  6
            5          5  7
            >>> range_indexes
            [DatetimeIndex(['2020-01-01', ..., '2020-01-06'], dtype='datetime64[ns]', name='split_0'),
             DatetimeIndex(['2020-01-03', ..., '2020-01-08'], dtype='datetime64[ns]', name='split_1')]

            >>> range_df, range_indexes = sr.vbt.range_split(start_idxs=[0], end_idxs=[2, 3, 4])
            >>> range_df
            split_idx    0    1  2
            0          0.0  0.0  0
            1          1.0  1.0  1
            2          2.0  2.0  2
            3          NaN  3.0  3
            4          NaN  NaN  4
            >>> range_indexes
            [DatetimeIndex(['2020-01-01', ..., '2020-01-03'], dtype='datetime64[ns]', name='split_0'),
             DatetimeIndex(['2020-01-01', ..., '2020-01-04'], dtype='datetime64[ns]', name='split_1'),
             DatetimeIndex(['2020-01-01', ..., '2020-01-05'], dtype='datetime64[ns]', name='split_2')]

            >>> range_df, range_indexes = sr.vbt.range_split(
            ...     start_idxs=pd.Index(['2020-01-01', '2020-01-02']),
            ...     end_idxs=pd.Index(['2020-01-04', '2020-01-05'])
            ... )
            >>> range_df
            split_idx  0  1
            0          0  1
            1          1  2
            2          2  3
            3          3  4
            >>> range_indexes
            [DatetimeIndex(['2020-01-01', ..., '2020-01-04'], dtype='datetime64[ns]', name='split_0'),
             DatetimeIndex(['2020-01-02', ..., '2020-01-05'], dtype='datetime64[ns]', name='split_1')]

             >>> sr.vbt.range_split(
             ...    start_idxs=pd.Index(['2020-01-01', '2020-01-02', '2020-01-01']),
             ...    end_idxs=pd.Index(['2020-01-08', '2020-01-04', '2020-01-07']),
             ...    plot=True
             ... )
            ```

            ![](/assets/images/range_split_plot.svg)
        """
        return self.split(RangeSplitter(), **kwargs)

    def rolling_split(self, **kwargs) -> SplitOutputT:
        """Split using `GenericAccessor.split` on `vectorbt.generic.splitters.RollingSplitter`.

        Usage:
            ```pycon
            >>> train_set, valid_set, test_set = sr.vbt.rolling_split(
            ...     window_len=5, set_lens=(1, 1), left_to_right=False)
            >>> train_set[0]
            split_idx  0  1  2  3  4  5
            0          0  1  2  3  4  5
            1          1  2  3  4  5  6
            2          2  3  4  5  6  7
            >>> valid_set[0]
            split_idx  0  1  2  3  4  5
            0          3  4  5  6  7  8
            >>> test_set[0]
            split_idx  0  1  2  3  4  5
            0          4  5  6  7  8  9

            >>> sr.vbt.rolling_split(
            ...     window_len=5, set_lens=(1, 1), left_to_right=False,
            ...     plot=True, trace_names=['train', 'valid', 'test'])
            ```

            ![](/assets/images/rolling_split_plot.svg)
        """
        return self.split(RollingSplitter(), **kwargs)

    def expanding_split(self, **kwargs) -> SplitOutputT:
        """Split using `GenericAccessor.split` on `vectorbt.generic.splitters.ExpandingSplitter`.

        Usage:
            ```pycon
            >>> train_set, valid_set, test_set = sr.vbt.expanding_split(
            ...     n=5, set_lens=(1, 1), min_len=3, left_to_right=False)
            >>> train_set[0]
            split_idx    0    1    2    3    4    5    6  7
            0          0.0  0.0  0.0  0.0  0.0  0.0  0.0  0
            1          NaN  1.0  1.0  1.0  1.0  1.0  1.0  1
            2          NaN  NaN  2.0  2.0  2.0  2.0  2.0  2
            3          NaN  NaN  NaN  3.0  3.0  3.0  3.0  3
            4          NaN  NaN  NaN  NaN  4.0  4.0  4.0  4
            5          NaN  NaN  NaN  NaN  NaN  5.0  5.0  5
            6          NaN  NaN  NaN  NaN  NaN  NaN  6.0  6
            7          NaN  NaN  NaN  NaN  NaN  NaN  NaN  7
            >>> valid_set[0]
            split_idx  0  1  2  3  4  5  6  7
            0          1  2  3  4  5  6  7  8
            >>> test_set[0]
            split_idx  0  1  2  3  4  5  6  7
            0          2  3  4  5  6  7  8  9

            >>> sr.vbt.expanding_split(
            ...     set_lens=(1, 1), min_len=3, left_to_right=False,
            ...     plot=True, trace_names=['train', 'valid', 'test'])
            ```

            ![](/assets/images/expanding_split_plot.svg)
        """
        return self.split(ExpandingSplitter(), **kwargs)

    # ############# Plotting ############# #

    def plot(self,
             trace_names: tp.TraceNames = None,
             x_labels: tp.Optional[tp.Labels] = None,
             return_fig: bool = True,
             **kwargs) -> tp.Union[tp.BaseFigure, plotting.Scatter]:  # pragma: no cover
        """Create `vectorbt.generic.plotting.Scatter` and return the figure.

        Usage:
            ```pycon
            >>> df.vbt.plot()
            ```

            ![](/assets/images/df_plot.svg)
        """
        if x_labels is None:
            x_labels = self.wrapper.index
        if trace_names is None:
            if self.is_frame() or (self.is_series() and self.wrapper.name is not None):
                trace_names = self.wrapper.columns
        scatter = plotting.Scatter(
            data=self.to_2d_array(),
            trace_names=trace_names,
            x_labels=x_labels,
            **kwargs
        )
        if return_fig:
            return scatter.fig
        return scatter

    def lineplot(self, **kwargs) -> tp.Union[tp.BaseFigure, plotting.Scatter]:  # pragma: no cover
        """`GenericAccessor.plot` with 'lines' mode.

        Usage:
            ```pycon
            >>> df.vbt.lineplot()
            ```

            ![](/assets/images/df_lineplot.svg)
        """
        return self.plot(**merge_dicts(dict(trace_kwargs=dict(mode='lines')), kwargs))

    def scatterplot(self, **kwargs) -> tp.Union[tp.BaseFigure, plotting.Scatter]:  # pragma: no cover
        """`GenericAccessor.plot` with 'markers' mode.

        Usage:
            ```pycon
            >>> df.vbt.scatterplot()
            ```

            ![](/assets/images/df_scatterplot.svg)
        """
        return self.plot(**merge_dicts(dict(trace_kwargs=dict(mode='markers')), kwargs))

    def barplot(self,
                trace_names: tp.TraceNames = None,
                x_labels: tp.Optional[tp.Labels] = None,
                return_fig: bool = True,
                **kwargs) -> tp.Union[tp.BaseFigure, plotting.Bar]:  # pragma: no cover
        """Create `vectorbt.generic.plotting.Bar` and return the figure.

        Usage:
            ```pycon
            >>> df.vbt.barplot()
            ```

            ![](/assets/images/df_barplot.svg)
        """
        if x_labels is None:
            x_labels = self.wrapper.index
        if trace_names is None:
            if self.is_frame() or (self.is_series() and self.wrapper.name is not None):
                trace_names = self.wrapper.columns
        bar = plotting.Bar(
            data=self.to_2d_array(),
            trace_names=trace_names,
            x_labels=x_labels,
            **kwargs
        )
        if return_fig:
            return bar.fig
        return bar

    def histplot(self,
                 trace_names: tp.TraceNames = None,
                 group_by: tp.GroupByLike = None,
                 return_fig: bool = True,
                 **kwargs) -> tp.Union[tp.BaseFigure, plotting.Histogram]:  # pragma: no cover
        """Create `vectorbt.generic.plotting.Histogram` and return the figure.

        Usage:
            ```pycon
            >>> df.vbt.histplot()
            ```

            ![](/assets/images/df_histplot.svg)
        """
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.flatten_grouped(group_by=group_by).vbt.histplot(trace_names=trace_names, **kwargs)

        if trace_names is None:
            if self.is_frame() or (self.is_series() and self.wrapper.name is not None):
                trace_names = self.wrapper.columns
        hist = plotting.Histogram(
            data=self.to_2d_array(),
            trace_names=trace_names,
            **kwargs
        )
        if return_fig:
            return hist.fig
        return hist

    def boxplot(self,
                trace_names: tp.TraceNames = None,
                group_by: tp.GroupByLike = None,
                return_fig: bool = True,
                **kwargs) -> tp.Union[tp.BaseFigure, plotting.Box]:  # pragma: no cover
        """Create `vectorbt.generic.plotting.Box` and return the figure.

        Usage:
            ```pycon
            >>> df.vbt.boxplot()
            ```

            ![](/assets/images/df_boxplot.svg)
        """
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.flatten_grouped(group_by=group_by).vbt.boxplot(trace_names=trace_names, **kwargs)

        if trace_names is None:
            if self.is_frame() or (self.is_series() and self.wrapper.name is not None):
                trace_names = self.wrapper.columns
        box = plotting.Box(
            data=self.to_2d_array(),
            trace_names=trace_names,
            **kwargs
        )
        if return_fig:
            return box.fig
        return box

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `GenericAccessor.plots`.

        Merges `vectorbt.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `generic.plots` from `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        generic_plots_cfg = settings['generic']['plots']

        return merge_dicts(
            PlotsBuilderMixin.plots_defaults.__get__(self),
            generic_plots_cfg
        )

    _subplots: tp.ClassVar[Config] = Config(
        dict(
            plot=dict(
                check_is_not_grouped=True,
                plot_func='plot',
                pass_trace_names=False,
                tags='generic'
            )
        ),
        copy_kwargs=dict(copy_mode='deep')
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


GenericAccessor.override_metrics_doc(__pdoc__)
GenericAccessor.override_subplots_doc(__pdoc__)


class GenericSRAccessor(GenericAccessor, BaseSRAccessor):
    """Accessor on top of data of any type. For Series only.

    Accessible through `pd.Series.vbt`."""

    def __init__(self, obj: tp.Series, mapping: tp.Optional[tp.MappingLike] = None, **kwargs) -> None:
        BaseSRAccessor.__init__(self, obj, **kwargs)
        GenericAccessor.__init__(self, obj, mapping=mapping, **kwargs)

    def squeeze_grouped(self,
                        squeeze_func_nb: tp.GroupSqueezeFunc, *args,
                        group_by: tp.GroupByLike = None,
                        wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Squeeze each group of elements into a single element.

        Based on `vectorbt.generic.accessors.GenericDFAccessor.squeeze_grouped`."""
        obj_frame = self.obj.to_frame().transpose()
        squeezed = obj_frame.vbt.squeeze_grouped(squeeze_func_nb, *args, group_by=group_by).iloc[0]
        wrap_kwargs = merge_dicts(dict(name_or_index=self.wrapper.name), wrap_kwargs)
        return ArrayWrapper.from_obj(obj_frame).wrap_reduced(squeezed, group_by=group_by, **wrap_kwargs)

    def flatten_grouped(self,
                        group_by: tp.GroupByLike = None,
                        order: str = 'C',
                        wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Flatten each group of elements.

        Based on `vectorbt.generic.accessors.GenericDFAccessor.flatten_grouped`."""
        obj_frame = self.obj.to_frame().transpose()
        return obj_frame.vbt.flatten_grouped(group_by=group_by, order=order, wrap_kwargs=wrap_kwargs)

    def plot_against(self,
                     other: tp.ArrayLike,
                     trace_kwargs: tp.KwargsLike = None,
                     other_trace_kwargs: tp.Union[str, tp.KwargsLike] = None,
                     pos_trace_kwargs: tp.KwargsLike = None,
                     neg_trace_kwargs: tp.KwargsLike = None,
                     hidden_trace_kwargs: tp.KwargsLike = None,
                     add_trace_kwargs: tp.KwargsLike = None,
                     fig: tp.Optional[tp.BaseFigure] = None,
                     **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot Series as a line against another line.

        Args:
            other (array_like): Second array. Will broadcast.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.
            other_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `other`.

                Set to 'hidden' to hide.
            pos_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for positive line.
            neg_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for negative line.
            hidden_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for hidden lines.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> df['a'].vbt.plot_against(df['b'])
            ```

            ![](/assets/images/sr_plot_against.svg)
        """
        if trace_kwargs is None:
            trace_kwargs = {}
        if other_trace_kwargs is None:
            other_trace_kwargs = {}
        if pos_trace_kwargs is None:
            pos_trace_kwargs = {}
        if neg_trace_kwargs is None:
            neg_trace_kwargs = {}
        if hidden_trace_kwargs is None:
            hidden_trace_kwargs = {}
        obj, other = reshape_fns.broadcast(self.obj, other, columns_from='keep')
        checks.assert_instance_of(other, pd.Series)
        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # TODO: Using masks feels hacky
        pos_mask = self.obj > other
        if pos_mask.any():
            # Fill positive area
            pos_obj = self.obj.copy()
            pos_obj[~pos_mask] = other[~pos_mask]
            other.vbt.plot(
                trace_kwargs=merge_dicts(dict(
                    line=dict(
                        color='rgba(0, 0, 0, 0)',
                        width=0
                    ),
                    opacity=0,
                    hoverinfo='skip',
                    showlegend=False,
                    name=None,
                ), hidden_trace_kwargs),
                add_trace_kwargs=add_trace_kwargs,
                fig=fig
            )
            pos_obj.vbt.plot(
                trace_kwargs=merge_dicts(dict(
                    fillcolor='rgba(0, 128, 0, 0.3)',
                    line=dict(
                        color='rgba(0, 0, 0, 0)',
                        width=0
                    ),
                    opacity=0,
                    fill='tonexty',
                    connectgaps=False,
                    hoverinfo='skip',
                    showlegend=False,
                    name=None
                ), pos_trace_kwargs),
                add_trace_kwargs=add_trace_kwargs,
                fig=fig
            )
        neg_mask = self.obj < other
        if neg_mask.any():
            # Fill negative area
            neg_obj = self.obj.copy()
            neg_obj[~neg_mask] = other[~neg_mask]
            other.vbt.plot(
                trace_kwargs=merge_dicts(dict(
                    line=dict(
                        color='rgba(0, 0, 0, 0)',
                        width=0
                    ),
                    opacity=0,
                    hoverinfo='skip',
                    showlegend=False,
                    name=None
                ), hidden_trace_kwargs),
                add_trace_kwargs=add_trace_kwargs,
                fig=fig
            )
            neg_obj.vbt.plot(
                trace_kwargs=merge_dicts(dict(
                    line=dict(
                        color='rgba(0, 0, 0, 0)',
                        width=0
                    ),
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    opacity=0,
                    fill='tonexty',
                    connectgaps=False,
                    hoverinfo='skip',
                    showlegend=False,
                    name=None
                ), neg_trace_kwargs),
                add_trace_kwargs=add_trace_kwargs,
                fig=fig
            )

        # Plot main traces
        self.plot(trace_kwargs=trace_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)
        if other_trace_kwargs == 'hidden':
            other_trace_kwargs = dict(
                line=dict(
                    color='rgba(0, 0, 0, 0)',
                    width=0
                ),
                opacity=0.,
                hoverinfo='skip',
                showlegend=False,
                name=None
            )
        other.vbt.plot(trace_kwargs=other_trace_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)
        return fig

    def overlay_with_heatmap(self,
                             other: tp.ArrayLike,
                             trace_kwargs: tp.KwargsLike = None,
                             heatmap_kwargs: tp.KwargsLike = None,
                             add_trace_kwargs: tp.KwargsLike = None,
                             fig: tp.Optional[tp.BaseFigure] = None,
                             **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot Series as a line and overlays it with a heatmap.

        Args:
            other (array_like): Second array. Will broadcast.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.
            heatmap_kwargs (dict): Keyword arguments passed to `GenericDFAccessor.heatmap`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> df['a'].vbt.overlay_with_heatmap(df['b'])
            ```

            ![](/assets/images/sr_overlay_with_heatmap.svg)
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        if trace_kwargs is None:
            trace_kwargs = {}
        if heatmap_kwargs is None:
            heatmap_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        obj, other = reshape_fns.broadcast(self.obj, other, columns_from='keep')
        checks.assert_instance_of(other, pd.Series)
        if fig is None:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            if 'width' in plotting_cfg['layout']:
                fig.update_layout(width=plotting_cfg['layout']['width'] + 100)
        fig.update_layout(**layout_kwargs)

        other.vbt.ts_heatmap(**heatmap_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)
        self.plot(
            trace_kwargs=merge_dicts(dict(line=dict(color=plotting_cfg['color_schema']['blue'])), trace_kwargs),
            add_trace_kwargs=merge_dicts(dict(secondary_y=True), add_trace_kwargs),
            fig=fig
        )
        return fig

    def heatmap(self,
                x_level: tp.Optional[tp.Level] = None,
                y_level: tp.Optional[tp.Level] = None,
                symmetric: bool = False,
                sort: bool = True,
                x_labels: tp.Optional[tp.Labels] = None,
                y_labels: tp.Optional[tp.Labels] = None,
                slider_level: tp.Optional[tp.Level] = None,
                active: int = 0,
                slider_labels: tp.Optional[tp.Labels] = None,
                return_fig: bool = True,
                fig: tp.Optional[tp.BaseFigure] = None,
                **kwargs) -> tp.Union[tp.BaseFigure, plotting.Heatmap]:  # pragma: no cover
        """Create a heatmap figure based on object's multi-index and values.

        If index is not a multi-index, converts Series into a DataFrame and calls `GenericDFAccessor.heatmap`.

        If multi-index contains more than two levels or you want them in specific order,
        pass `x_level` and `y_level`, each (`int` if index or `str` if name) corresponding
        to an axis of the heatmap. Optionally, pass `slider_level` to use a level as a slider.

        Creates `vectorbt.generic.plotting.Heatmap` and returns the figure.

        Usage:
            ```pycon
            >>> multi_index = pd.MultiIndex.from_tuples([
            ...     (1, 1),
            ...     (2, 2),
            ...     (3, 3)
            ... ])
            >>> sr = pd.Series(np.arange(len(multi_index)), index=multi_index)
            >>> sr
            1  1    0
            2  2    1
            3  3    2
            dtype: int64

            >>> sr.vbt.heatmap()
            ```

            ![](/assets/images/sr_heatmap.svg)

            * Using one level as a slider:

            ```pycon
            >>> multi_index = pd.MultiIndex.from_tuples([
            ...     (1, 1, 1),
            ...     (1, 2, 2),
            ...     (1, 3, 3),
            ...     (2, 3, 3),
            ...     (2, 2, 2),
            ...     (2, 1, 1)
            ... ])
            >>> sr = pd.Series(np.arange(len(multi_index)), index=multi_index)
            >>> sr
            1  1  1    0
               2  2    1
               3  3    2
            2  3  3    3
               2  2    4
               1  1    5
            dtype: int64

            >>> sr.vbt.heatmap(slider_level=0)
            ```

            ![](/assets/images/sr_heatmap_slider.gif)
        """
        if not isinstance(self.wrapper.index, pd.MultiIndex):
            return self.obj.to_frame().vbt.heatmap(
                x_labels=x_labels, y_labels=y_labels,
                return_fig=return_fig, fig=fig, **kwargs)

        (x_level, y_level), (slider_level,) = index_fns.pick_levels(
            self.wrapper.index,
            required_levels=(x_level, y_level),
            optional_levels=(slider_level,)
        )

        x_level_vals = self.wrapper.index.get_level_values(x_level)
        y_level_vals = self.wrapper.index.get_level_values(y_level)
        x_name = x_level_vals.name if x_level_vals.name is not None else 'x'
        y_name = y_level_vals.name if y_level_vals.name is not None else 'y'
        kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                hovertemplate=f"{x_name}: %{{x}}<br>" +
                              f"{y_name}: %{{y}}<br>" +
                              "value: %{z}<extra></extra>"
            ),
            xaxis_title=x_level_vals.name,
            yaxis_title=y_level_vals.name
        ), kwargs)

        if slider_level is None:
            # No grouping
            df = self.unstack_to_df(
                index_levels=y_level, column_levels=x_level,
                symmetric=symmetric, sort=sort
            )
            return df.vbt.heatmap(x_labels=x_labels, y_labels=y_labels, fig=fig, return_fig=return_fig, **kwargs)

        # Requires grouping
        # See https://plotly.com/python/sliders/
        if not return_fig:
            raise ValueError("Cannot use return_fig=False and slider_level simultaneously")
        _slider_labels = []
        for i, (name, group) in enumerate(self.obj.groupby(level=slider_level)):
            if slider_labels is not None:
                name = slider_labels[i]
            _slider_labels.append(name)
            df = group.vbt.unstack_to_df(
                index_levels=y_level, column_levels=x_level,
                symmetric=symmetric, sort=sort
            )
            if x_labels is None:
                x_labels = df.columns
            if y_labels is None:
                y_labels = df.index
            _kwargs = merge_dicts(dict(
                trace_kwargs=dict(
                    name=str(name) if name is not None else None,
                    visible=False
                ),
            ), kwargs)
            default_size = fig is None and 'height' not in _kwargs
            fig = plotting.Heatmap(
                data=reshape_fns.to_2d_array(df),
                x_labels=x_labels,
                y_labels=y_labels,
                fig=fig,
                **_kwargs
            ).fig
            if default_size:
                fig.layout['height'] += 100  # slider takes up space
        fig.data[active].visible = True
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}, {}],
                label=str(_slider_labels[i]) if _slider_labels[i] is not None else None
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)
        prefix = f'{self.wrapper.index.names[slider_level]}: ' \
            if self.wrapper.index.names[slider_level] is not None else None
        sliders = [dict(
            active=active,
            currentvalue={"prefix": prefix},
            pad={"t": 50},
            steps=steps
        )]
        fig.update_layout(
            sliders=sliders
        )
        return fig

    def ts_heatmap(self, **kwargs) -> tp.Union[tp.BaseFigure, plotting.Heatmap]:  # pragma: no cover
        """Heatmap of time-series data."""
        return self.obj.to_frame().vbt.ts_heatmap(**kwargs)

    def volume(self,
               x_level: tp.Optional[tp.Level] = None,
               y_level: tp.Optional[tp.Level] = None,
               z_level: tp.Optional[tp.Level] = None,
               x_labels: tp.Optional[tp.Labels] = None,
               y_labels: tp.Optional[tp.Labels] = None,
               z_labels: tp.Optional[tp.Labels] = None,
               slider_level: tp.Optional[tp.Level] = None,
               slider_labels: tp.Optional[tp.Labels] = None,
               active: int = 0,
               scene_name: str = 'scene',
               fillna: tp.Optional[tp.Number] = None,
               fig: tp.Optional[tp.BaseFigure] = None,
               return_fig: bool = True,
               **kwargs) -> tp.Union[tp.BaseFigure, plotting.Volume]:  # pragma: no cover
        """Create a 3D volume figure based on object's multi-index and values.

        If multi-index contains more than three levels or you want them in specific order, pass
        `x_level`, `y_level`, and `z_level`, each (`int` if index or `str` if name) corresponding
        to an axis of the volume. Optionally, pass `slider_level` to use a level as a slider.

        Creates `vectorbt.generic.plotting.Volume` and returns the figure.

        Usage:
            ```pycon
            >>> multi_index = pd.MultiIndex.from_tuples([
            ...     (1, 1, 1),
            ...     (2, 2, 2),
            ...     (3, 3, 3)
            ... ])
            >>> sr = pd.Series(np.arange(len(multi_index)), index=multi_index)
            >>> sr
            1  1  1    0
            2  2  2    1
            3  3  3    2
            dtype: int64

            >>> sr.vbt.volume().show()
            ```

            ![](/assets/images/sr_volume.svg)
        """
        (x_level, y_level, z_level), (slider_level,) = index_fns.pick_levels(
            self.wrapper.index,
            required_levels=(x_level, y_level, z_level),
            optional_levels=(slider_level,)
        )

        x_level_vals = self.wrapper.index.get_level_values(x_level)
        y_level_vals = self.wrapper.index.get_level_values(y_level)
        z_level_vals = self.wrapper.index.get_level_values(z_level)
        # Labels are just unique level values
        if x_labels is None:
            x_labels = np.unique(x_level_vals)
        if y_labels is None:
            y_labels = np.unique(y_level_vals)
        if z_labels is None:
            z_labels = np.unique(z_level_vals)

        x_name = x_level_vals.name if x_level_vals.name is not None else 'x'
        y_name = y_level_vals.name if y_level_vals.name is not None else 'y'
        z_name = z_level_vals.name if z_level_vals.name is not None else 'z'
        def_kwargs = dict()
        def_kwargs['trace_kwargs'] = dict(
            hovertemplate=f"{x_name}: %{{x}}<br>" +
                          f"{y_name}: %{{y}}<br>" +
                          f"{z_name}: %{{z}}<br>" +
                          "value: %{value}<extra></extra>"
        )
        def_kwargs[scene_name] = dict(
            xaxis_title=x_level_vals.name,
            yaxis_title=y_level_vals.name,
            zaxis_title=z_level_vals.name
        )
        def_kwargs['scene_name'] = scene_name
        kwargs = merge_dicts(def_kwargs, kwargs)

        contains_nan = False
        if slider_level is None:
            # No grouping
            v = self.unstack_to_array(levels=(x_level, y_level, z_level))
            if fillna is not None:
                v = np.nan_to_num(v, nan=fillna)
            if np.isnan(v).any():
                contains_nan = True
            volume = plotting.Volume(
                data=v,
                x_labels=x_labels,
                y_labels=y_labels,
                z_labels=z_labels,
                fig=fig,
                **kwargs
            )
            if return_fig:
                fig = volume.fig
            else:
                fig = volume
        else:
            # Requires grouping
            # See https://plotly.com/python/sliders/
            if not return_fig:
                raise ValueError("Cannot use return_fig=False and slider_level simultaneously")
            _slider_labels = []
            for i, (name, group) in enumerate(self.obj.groupby(level=slider_level)):
                if slider_labels is not None:
                    name = slider_labels[i]
                _slider_labels.append(name)
                v = group.vbt.unstack_to_array(levels=(x_level, y_level, z_level))
                if fillna is not None:
                    v = np.nan_to_num(v, nan=fillna)
                if np.isnan(v).any():
                    contains_nan = True
                _kwargs = merge_dicts(dict(
                    trace_kwargs=dict(
                        name=str(name) if name is not None else None,
                        visible=False
                    )
                ), kwargs)
                default_size = fig is None and 'height' not in _kwargs
                fig = plotting.Volume(
                    data=v,
                    x_labels=x_labels,
                    y_labels=y_labels,
                    z_labels=z_labels,
                    fig=fig,
                    **_kwargs
                ).fig
                if default_size:
                    fig.layout['height'] += 100  # slider takes up space
            fig.data[active].visible = True
            steps = []
            for i in range(len(fig.data)):
                step = dict(
                    method="update",
                    args=[{"visible": [False] * len(fig.data)}, {}],
                    label=str(_slider_labels[i]) if _slider_labels[i] is not None else None
                )
                step["args"][0]["visible"][i] = True
                steps.append(step)
            prefix = f'{self.wrapper.index.names[slider_level]}: ' \
                if self.wrapper.index.names[slider_level] is not None else None
            sliders = [dict(
                active=active,
                currentvalue={"prefix": prefix},
                pad={"t": 50},
                steps=steps
            )]
            fig.update_layout(
                sliders=sliders
            )

        if contains_nan:
            warnings.warn("Data contains NaNs. Use `fillna` argument or "
                          "`show` method in case of visualization issues.", stacklevel=2)
        return fig

    def qqplot(self,
               sparams: tp.Union[tp.Iterable, tuple, None] = (),
               dist: str = 'norm',
               plot_line: bool = True,
               line_shape_kwargs: tp.KwargsLike = None,
               xref: str = 'x',
               yref: str = 'y',
               fig: tp.Optional[tp.BaseFigure] = None,
               **kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot probability plot using `scipy.stats.probplot`.

        `**kwargs` are passed to `GenericAccessor.scatterplot`.

        Usage:
            ```pycon
            >>> pd.Series(np.random.standard_normal(100)).vbt.qqplot()
            ```

            ![](/assets/images/sr_qqplot.svg)
        """
        qq = stats.probplot(self.obj, sparams=sparams, dist=dist)
        fig = pd.Series(qq[0][1], index=qq[0][0]).vbt.scatterplot(fig=fig, **kwargs)

        if plot_line:
            if line_shape_kwargs is None:
                line_shape_kwargs = {}
            x = np.array([qq[0][0][0], qq[0][0][-1]])
            y = qq[1][1] + qq[1][0] * x
            fig.add_shape(**merge_dicts(dict(
                type="line",
                xref=xref,
                yref=yref,
                x0=x[0],
                y0=y[0],
                x1=x[1],
                y1=y[1],
                line=dict(
                    color='red'
                )
            ), line_shape_kwargs))

        return fig


class GenericDFAccessor(GenericAccessor, BaseDFAccessor):
    """Accessor on top of data of any type. For DataFrames only.

    Accessible through `pd.DataFrame.vbt`."""

    def __init__(self, obj: tp.Frame, mapping: tp.Optional[tp.MappingLike] = None, **kwargs) -> None:
        BaseDFAccessor.__init__(self, obj, **kwargs)
        GenericAccessor.__init__(self, obj, mapping=mapping, **kwargs)

    def squeeze_grouped(self,
                        squeeze_func_nb: tp.GroupSqueezeFunc, *args,
                        group_by: tp.GroupByLike = None,
                        wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Squeeze each group of columns into a single column.

        See `vectorbt.generic.nb.squeeze_grouped_nb`.

        Usage:
            ```pycon
            >>> group_by = pd.Series(['first', 'first', 'second'], name='group')
            >>> mean_squeeze_nb = njit(lambda i, group, a: np.nanmean(a))
            >>> df.vbt.squeeze_grouped(mean_squeeze_nb, group_by=group_by)
            group       first  second
            2020-01-01    3.0     1.0
            2020-01-02    3.0     2.0
            2020-01-03    3.0     3.0
            2020-01-04    3.0     2.0
            2020-01-05    3.0     1.0
            ```
        """
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping required")
        checks.assert_numba_func(squeeze_func_nb)

        group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
        out = nb.squeeze_grouped_nb(self.to_2d_array(), group_lens, squeeze_func_nb, *args)
        return self.wrapper.wrap(out, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    def flatten_grouped(self,
                        group_by: tp.GroupByLike = None,
                        order: str = 'C',
                        wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Flatten each group of columns.

        See `vectorbt.generic.nb.flatten_grouped_nb`.
        If all groups have the same length, see `vectorbt.generic.nb.flatten_uniform_grouped_nb`.

        !!! warning
            Make sure that the distribution of group lengths is close to uniform, otherwise
            groups with less columns will be filled with NaN and needlessly occupy memory.

        Usage:
            ```pycon
            >>> group_by = pd.Series(['first', 'first', 'second'], name='group')
            >>> df.vbt.flatten_grouped(group_by=group_by, order='C')
            group       first  second
            2020-01-01    1.0     1.0
            2020-01-01    5.0     NaN
            2020-01-02    2.0     2.0
            2020-01-02    4.0     NaN
            2020-01-03    3.0     3.0
            2020-01-03    3.0     NaN
            2020-01-04    4.0     2.0
            2020-01-04    2.0     NaN
            2020-01-05    5.0     1.0
            2020-01-05    1.0     NaN

            >>> df.vbt.flatten_grouped(group_by=group_by, order='F')
            group       first  second
            2020-01-01    1.0     1.0
            2020-01-02    2.0     2.0
            2020-01-03    3.0     3.0
            2020-01-04    4.0     2.0
            2020-01-05    5.0     1.0
            2020-01-01    5.0     NaN
            2020-01-02    4.0     NaN
            2020-01-03    3.0     NaN
            2020-01-04    2.0     NaN
            2020-01-05    1.0     NaN
            ```
        """
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping required")
        checks.assert_in(order.upper(), ['C', 'F'])

        group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
        if np.all(group_lens == group_lens.item(0)):
            func = nb.flatten_uniform_grouped_nb
        else:
            func = nb.flatten_grouped_nb
        if order.upper() == 'C':
            out = func(self.to_2d_array(), group_lens, True)
            new_index = index_fns.repeat_index(self.wrapper.index, np.max(group_lens))
        else:
            out = func(self.to_2d_array(), group_lens, False)
            new_index = index_fns.tile_index(self.wrapper.index, np.max(group_lens))
        wrap_kwargs = merge_dicts(dict(index=new_index), wrap_kwargs)
        return self.wrapper.wrap(out, group_by=group_by, **wrap_kwargs)

    def heatmap(self,
                x_labels: tp.Optional[tp.Labels] = None,
                y_labels: tp.Optional[tp.Labels] = None,
                return_fig: bool = True,
                **kwargs) -> tp.Union[tp.BaseFigure, plotting.Heatmap]:  # pragma: no cover
        """
        热力图绘制 - 创建数据的热力图可视化
        
        这个方法创建一个热力图，用于可视化二维数据的模式和分布。
        热力图在金融分析中常用于展示相关性矩阵、回报分布等。
        
        参数：
            x_labels (tp.Optional[tp.Labels], 可选): X轴标签，默认使用列名
            y_labels (tp.Optional[tp.Labels], 可选): Y轴标签，默认使用索引
            return_fig (bool, 可选): 是否返回图形对象，默认True
            **kwargs: 传递给Heatmap类的额外参数
        
        返回：
            tp.Union[tp.BaseFigure, plotting.Heatmap]: 热力图对象或图形
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 示例1：基本热力图
        df = pd.DataFrame([
            [0, np.nan, np.nan],
            [np.nan, 1, np.nan],
            [np.nan, np.nan, 2]
        ])
        fig = df.vbt.heatmap()
        fig.show()
        
        # 示例2：金融数据热力图
        # 创建相关性矩阵
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'GOOGL': np.random.normal(0.002, 0.025, 100),
            'MSFT': np.random.normal(0.0015, 0.022, 100)
        })
        
        corr_matrix = returns.corr()
        fig = corr_matrix.vbt.heatmap()
        fig.show()
        
        # 示例3：自定义标签
        custom_fig = df.vbt.heatmap(
            x_labels=['X1', 'X2', 'X3'],
            y_labels=['Y1', 'Y2', 'Y3']
        )
        custom_fig.show()
        ```
        """
        if x_labels is None:
            x_labels = self.wrapper.columns
        if y_labels is None:
            y_labels = self.wrapper.index
        heatmap = plotting.Heatmap(
            data=self.to_2d_array(),
            x_labels=x_labels,
            y_labels=y_labels,
            **kwargs
        )
        if return_fig:
            return heatmap.fig
        return heatmap

    def ts_heatmap(self, is_y_category: bool = True,
                   **kwargs) -> tp.Union[tp.BaseFigure, plotting.Heatmap]:  # pragma: no cover
        """
        时间序列热力图 - 创建时间序列数据的热力图
        
        这个方法专门用于时间序列数据的热力图可视化，
        将数据转置并反转以便更好地展示时间序列模式。
        
        参数：
            is_y_category (bool, 可选): Y轴是否为分类数据，默认True
            **kwargs: 传递给heatmap方法的额外参数
        
        返回：
            tp.Union[tp.BaseFigure, plotting.Heatmap]: 热力图对象或图形
        
        使用示例：
        ```python
        import pandas as pd
        import numpy as np
        import vectorbt as vbt
        
        # 创建时间序列数据
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'Series1': np.random.randn(30),
            'Series2': np.random.randn(30),
            'Series3': np.random.randn(30)
        }, index=dates)
        
        # 创建时间序列热力图
        fig = data.vbt.ts_heatmap()
        fig.show()
        ```
        """
        return self.obj.transpose().iloc[::-1].vbt.heatmap(is_y_category=is_y_category, **kwargs)