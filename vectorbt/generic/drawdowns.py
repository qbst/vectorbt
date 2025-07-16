# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Base class for working with drawdown records.

Drawdown records capture information on drawdowns. Since drawdowns are ranges,
they subclass `vectorbt.generic.ranges.Ranges`.

!!! warning
    `Drawdowns` return both recovered AND active drawdowns, which may skew your performance results.
    To only consider recovered drawdowns, you should explicitly query `recovered` attribute.

Using `Drawdowns.from_ts`, you can generate drawdown records for any time series and analyze them right away.

```pycon
>>> import vectorbt as vbt
>>> import numpy as np
>>> import pandas as pd

>>> start = '2019-10-01 UTC'  # crypto is in UTC
>>> end = '2020-01-01 UTC'
>>> price = vbt.YFData.download('BTC-USD', start=start, end=end).get('Close')
>>> price = price.rename(None)

>>> drawdowns = vbt.Drawdowns.from_ts(price, wrapper_kwargs=dict(freq='d'))

>>> drawdowns.records_readable
   Drawdown Id  Column            Peak Timestamp           Start Timestamp  \\
0            0       0 2019-10-02 00:00:00+00:00 2019-10-03 00:00:00+00:00
1            1       0 2019-10-09 00:00:00+00:00 2019-10-10 00:00:00+00:00
2            2       0 2019-10-27 00:00:00+00:00 2019-10-28 00:00:00+00:00

           Valley Timestamp             End Timestamp   Peak Value  \\
0 2019-10-06 00:00:00+00:00 2019-10-09 00:00:00+00:00  8393.041992
1 2019-10-24 00:00:00+00:00 2019-10-25 00:00:00+00:00  8595.740234
2 2019-12-17 00:00:00+00:00 2020-01-01 00:00:00+00:00  9551.714844

   Valley Value    End Value     Status
0   7988.155762  8595.740234  Recovered
1   7493.488770  8660.700195  Recovered
2   6640.515137  7200.174316     Active

>>> drawdowns.duration.max(wrap_kwargs=dict(to_timedelta=True))
Timedelta('66 days 00:00:00')
```

## From accessors

Moreover, all generic accessors have a property `drawdowns` and a method `get_drawdowns`:

```pycon
>>> # vectorbt.generic.accessors.GenericAccessor.drawdowns.coverage
>>> price.vbt.drawdowns.coverage()
0.9354838709677419
```

## Stats

!!! hint
    See `vectorbt.generic.stats_builder.StatsBuilderMixin.stats` and `Drawdowns.metrics`.

```pycon
>>> df = pd.DataFrame({
...     'a': [1, 2, 1, 3, 2],
...     'b': [2, 3, 1, 2, 1]
... })

>>> drawdowns = df.vbt(freq='d').drawdowns

>>> drawdowns['a'].stats()
Start                                        0
End                                          4
Period                         5 days 00:00:00
Coverage [%]                              40.0
Total Records                                2
Total Recovered Drawdowns                    1
Total Active Drawdowns                       1
Active Drawdown [%]                  33.333333
Active Duration                1 days 00:00:00
Active Recovery [%]                        0.0
Active Recovery Return [%]                 0.0
Active Recovery Duration       0 days 00:00:00
Max Drawdown [%]                          50.0
Avg Drawdown [%]                          50.0
Max Drawdown Duration          1 days 00:00:00
Avg Drawdown Duration          1 days 00:00:00
Max Recovery Return [%]                  200.0
Avg Recovery Return [%]                  200.0
Max Recovery Duration          1 days 00:00:00
Avg Recovery Duration          1 days 00:00:00
Avg Recovery Duration Ratio                1.0
Name: a, dtype: object
```

By default, the metrics `max_dd`, `avg_dd`, `max_dd_duration`, and `avg_dd_duration` do
not include active drawdowns. To change that, pass `incl_active=True`:

```pycon
>>> drawdowns['a'].stats(settings=dict(incl_active=True))
Start                                        0
End                                          4
Period                         5 days 00:00:00
Coverage [%]                              40.0
Total Records                                2
Total Recovered Drawdowns                    1
Total Active Drawdowns                       1
Active Drawdown [%]                  33.333333
Active Duration                1 days 00:00:00
Active Recovery [%]                        0.0
Active Recovery Return [%]                 0.0
Active Recovery Duration       0 days 00:00:00
Max Drawdown [%]                          50.0
Avg Drawdown [%]                     41.666667
Max Drawdown Duration          1 days 00:00:00
Avg Drawdown Duration          1 days 00:00:00
Max Recovery Return [%]                  200.0
Avg Recovery Return [%]                  200.0
Max Recovery Duration          1 days 00:00:00
Avg Recovery Duration          1 days 00:00:00
Avg Recovery Duration Ratio                1.0
Name: a, dtype: object
```

`Drawdowns.stats` also supports (re-)grouping:

```pycon
>>> drawdowns['a'].stats(group_by=True)
UserWarning: Metric 'active_dd' does not support grouped data
UserWarning: Metric 'active_duration' does not support grouped data
UserWarning: Metric 'active_recovery' does not support grouped data
UserWarning: Metric 'active_recovery_return' does not support grouped data
UserWarning: Metric 'active_recovery_duration' does not support grouped data

Start                                        0
End                                          4
Period                         5 days 00:00:00
Coverage [%]                              40.0
Total Records                                2
Total Recovered Drawdowns                    1
Total Active Drawdowns                       1
Max Drawdown [%]                          50.0
Avg Drawdown [%]                          50.0
Max Drawdown Duration          1 days 00:00:00
Avg Drawdown Duration          1 days 00:00:00
Max Recovery Return [%]                  200.0
Avg Recovery Return [%]                  200.0
Max Recovery Duration          1 days 00:00:00
Avg Recovery Duration          1 days 00:00:00
Avg Recovery Duration Ratio                1.0
Name: group, dtype: object
```

## Plots

!!! hint
    See `vectorbt.generic.plots_builder.PlotsBuilderMixin.plots` and `Drawdowns.subplots`.

`Drawdowns` class has a single subplot based on `Drawdowns.plot`:

```pycon
>>> drawdowns['a'].plots()
```

![](/assets/images/drawdowns_plots.svg)
"""

# ========== 导入必要的库和模块 ==========
import numpy as np                          # 导入NumPy库，用于数值计算和数组操作
import pandas as pd                         # 导入Pandas库，用于数据处理和时间序列分析
import plotly.graph_objects as go          # 导入Plotly绘图对象，用于创建交互式图表

# ========== 导入VectorBT框架的核心模块 ==========
from vectorbt import _typing as tp                              # 导入类型提示模块
from vectorbt.base.array_wrapper import ArrayWrapper           # 导入数组包装器，用于处理索引和列名
from vectorbt.base.reshape_fns import to_2d_array, to_pd_array # 导入数组转换函数
from vectorbt.generic import nb                                # 导入Numba编译的核心计算函数
from vectorbt.generic.enums import DrawdownStatus, drawdown_dt # 导入回撤状态枚举和数据类型
from vectorbt.generic.ranges import Ranges                     # 导入范围记录基类
from vectorbt.records.decorators import override_field_config, attach_fields # 导入记录装饰器
from vectorbt.records.mapped_array import MappedArray          # 导入映射数组类
from vectorbt.utils.colors import adjust_lightness             # 导入颜色调整工具
from vectorbt.utils.config import merge_dicts, Config          # 导入配置管理工具
from vectorbt.utils.decorators import cached_property, cached_method # 导入缓存装饰器
from vectorbt.utils.figure import make_figure, get_domain      # 导入图表创建工具
from vectorbt.utils.template import RepEval                    # 导入模板评估工具

# 初始化文档字典，用于存储API文档信息
__pdoc__ = {}

# ========== 回撤记录字段配置 ==========
# 定义Drawdowns类的字段配置，规定了回撤记录的数据结构和字段属性
dd_field_config = Config(
    dict(
        # 指定回撤记录使用的数据类型，包含回撤分析所需的所有字段
        dtype=drawdown_dt,
        
        # 字段设置：定义每个字段的显示标题和映射关系
        settings=dict(
            # 回撤ID字段：用于唯一标识每个回撤记录
            id=dict(
                title='Drawdown Id'            # 字段显示标题
            ),
            
            # 峰值索引字段：标识回撤开始时的峰值位置
            peak_idx=dict(
                title='Peak Timestamp',        # 字段显示标题：峰值时间戳
                mapping='index'               # 映射到ArrayWrapper的index属性
            ),
            
            # 谷底索引字段：标识回撤的最低点位置
            valley_idx=dict(
                title='Valley Timestamp',      # 字段显示标题：谷底时间戳
                mapping='index'               # 映射到ArrayWrapper的index属性
            ),
            
            # 峰值价格字段：记录回撤开始时的价格水平
            peak_val=dict(
                title='Peak Value',           # 字段显示标题：峰值价格
            ),
            
            # 谷底价格字段：记录回撤的最低价格水平
            valley_val=dict(
                title='Valley Value',         # 字段显示标题：谷底价格
            ),
            
            # 结束价格字段：记录回撤结束时的价格水平
            end_val=dict(
                title='End Value',            # 字段显示标题：结束价格
            ),
            
            # 回撤状态字段：标识回撤是否已恢复
            status=dict(
                mapping=DrawdownStatus        # 映射到DrawdownStatus枚举类
            )
        )
    ),
    readonly=True,      # 配置为只读，防止运行时修改
    as_attrs=False     # 不将配置项作为属性访问
)
"""_"""

# 为dd_field_config生成API文档
__pdoc__['dd_field_config'] = f"""Drawdowns类的字段配置。

这个配置定义了回撤记录的数据结构，包括：
- dtype: 指定使用drawdown_dt数据类型
- settings: 定义各字段的显示标题和映射关系
  - id: 回撤的唯一标识符
  - peak_idx: 峰值索引，映射到时间戳
  - valley_idx: 谷底索引，映射到时间戳  
  - peak_val: 峰值价格
  - valley_val: 谷底价格
  - end_val: 结束价格
  - status: 回撤状态，映射到DrawdownStatus枚举

配置内容：
```json
{dd_field_config.to_doc()}
```
"""

# ========== 回撤记录附加字段配置 ==========
# 定义需要附加到Drawdowns类的字段配置，用于自动生成辅助方法
dd_attach_field_config = Config(
    dict(
        # 为status字段启用过滤器功能
        status=dict(
            attach_filters=True        # 自动生成按状态过滤的方法，如filter_by_status
        )
    ),
    readonly=True,    # 配置为只读
    as_attrs=False   # 不将配置项作为属性访问
)
"""_"""

# 为dd_attach_field_config生成API文档
__pdoc__['dd_attach_field_config'] = f"""需要附加到Drawdowns类的字段配置。

这个配置指定了哪些字段需要自动生成过滤器和其他辅助功能：
- status字段启用过滤器：将自动生成按状态过滤的方法

配置内容：
```json
{dd_attach_field_config.to_doc()}
```
"""

# 定义Drawdowns类的类型变量，用于类型提示中的泛型约束
# 这确保了Drawdowns类的方法返回的类型与调用类的类型一致
DrawdownsT = tp.TypeVar("DrawdownsT", bound="Drawdowns")


# 使用装饰器为Drawdowns类附加字段功能和重写字段配置
@attach_fields(dd_attach_field_config)      # 附加字段配置，自动生成辅助方法
@override_field_config(dd_field_config)     # 重写字段配置，使用回撤特有的字段结构
class Drawdowns(Ranges):
    """
    Drawdowns类 - 专门用于处理回撤记录的Ranges子类
    
    这个类扩展了`vectorbt.generic.ranges.Ranges`，专门用于处理回撤记录。
    回撤记录捕获了价格从峰值到谷底再到恢复的完整过程信息，是量化金融
    中风险管理和策略评估的核心工具。
    
    核心功能：
    1. **回撤识别**：从价格时间序列中自动识别回撤期间
    2. **状态区分**：区分已恢复回撤和活跃回撤
    3. **风险分析**：计算最大回撤、平均回撤、恢复时间等指标
    4. **可视化**：提供专门的回撤可视化功能
    
    数据结构：
    - id: 回撤的唯一标识符
    - col: 回撤所属的列索引
    - peak_idx: 峰值索引（回撤开始点）
    - valley_idx: 谷底索引（最大回撤点）
    - end_idx: 结束索引（恢复点或当前点）
    - peak_val: 峰值价格
    - valley_val: 谷底价格
    - end_val: 结束价格
    - status: 回撤状态（已恢复/活跃）
    
    使用示例：
    ```python
    import vectorbt as vbt
    import pandas as pd
    import numpy as np
    
    # 示例1：从价格序列创建回撤记录
    price = pd.Series([100, 105, 98, 95, 102, 108, 103])
    drawdowns = vbt.Drawdowns.from_ts(price)
    
    # 查看回撤记录
    print("回撤记录:")
    print(drawdowns.records_readable)
    
    # 示例2：计算回撤统计指标
    max_dd = drawdowns.max_drawdown()  # 最大回撤
    avg_dd = drawdowns.avg_drawdown()  # 平均回撤
    recovery_time = drawdowns.avg_recovery_duration()  # 平均恢复时间
    
    print(f"最大回撤: {max_dd:.2%}")
    print(f"平均回撤: {avg_dd:.2%}")
    print(f"平均恢复时间: {recovery_time}")
    
    # 示例3：分析已恢复和活跃回撤
    recovered = drawdowns.recovered  # 已恢复回撤
    active = drawdowns.active       # 活跃回撤
    
    print(f"已恢复回撤数量: {recovered.count()}")
    print(f"活跃回撤数量: {active.count()}")
    
    # 示例4：可视化回撤
    fig = drawdowns.plot()
    fig.show()
    ```
    
    继承关系：
    - 继承自Ranges类，获得所有范围记录的功能
    - 重写了字段配置，使用drawdown_dt数据类型
    - 添加了回撤特有的方法和属性
    
    要求：
    - records_arr必须包含drawdown_dt中定义的所有字段
    - 支持从价格时间序列自动创建回撤记录
    - 提供丰富的回撤分析和可视化功能
    """

    @property
    def field_config(self) -> Config:
        """
        字段配置属性 - 返回Drawdowns类的字段配置
        
        这个属性返回专门为Drawdowns类定义的字段配置，包括drawdown_dt数据类型
        和各个字段的设置信息。
        
        返回：
            Config: 包含dtype和settings的配置对象
            
        字段说明：
        - dtype: drawdown_dt数据类型
        - settings: 字段设置字典
          - id: 回撤标识符
          - peak_idx: 峰值索引（映射到时间戳）
          - valley_idx: 谷底索引（映射到时间戳）
          - end_idx: 结束索引（映射到时间戳）
          - peak_val: 峰值价格
          - valley_val: 谷底价格
          - end_val: 结束价格
          - status: 回撤状态（映射到DrawdownStatus枚举）
        """
        return self._field_config

    def __init__(self,
                 wrapper: ArrayWrapper,
                 records_arr: tp.RecordArray,
                 ts: tp.Optional[tp.ArrayLike] = None,
                 **kwargs) -> None:
        """
        Drawdowns类的初始化方法
        
        初始化一个Drawdowns对象，设置数组包装器、记录数组和可选的时间序列数据。
        
        参数：
            wrapper (ArrayWrapper): 数组包装器，包含索引、列名、分组等元数据
            records_arr (tp.RecordArray): 回撤记录的结构化数组
                必须包含drawdown_dt中定义的所有字段：
                - id: 回撤ID
                - col: 列索引
                - peak_idx: 峰值索引
                - valley_idx: 谷底索引
                - end_idx: 结束索引
                - peak_val: 峰值价格
                - valley_val: 谷底价格
                - end_val: 结束价格
                - status: 回撤状态
            ts (tp.Optional[tp.ArrayLike], 可选): 原始时间序列数据
                如果提供，将用于绘图和进一步分析
            **kwargs: 传递给Ranges基类的额外参数
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        from vectorbt.generic.enums import drawdown_dt, DrawdownStatus
        
        # 手动创建回撤记录数组
        records_arr = np.array([
            (0, 0, 0, 2, 4, 100, 90, 95, DrawdownStatus.Recovered),
            (1, 0, 5, 6, 7, 110, 105, 108, DrawdownStatus.Active)
        ], dtype=drawdown_dt)
        
        # 创建包装器
        wrapper = vbt.ArrayWrapper(
            index=pd.date_range('2023-01-01', periods=10, freq='D'),
            columns=['Price'],
            ndim=2
        )
        
        # 创建Drawdowns对象
        drawdowns = vbt.Drawdowns(wrapper, records_arr)
        
        # 查看结果
        print(drawdowns.records_readable)
        ```
        """
        # 调用Ranges基类的初始化方法
        # 这会设置基础的记录功能，包括字段验证和列映射器创建
        Ranges.__init__(
            self,
            wrapper,                    # 数组包装器
            records_arr,               # 记录数组
            ts=ts,                     # 时间序列数据
            **kwargs                   # 其他参数
        )
        
        # 存储原始时间序列数据到私有变量
        # 这个数据在绘图和某些分析中会用到
        self._ts = ts

    def indexing_func(self: DrawdownsT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> DrawdownsT:
        """
        执行索引操作并返回新的Drawdowns实例 - 支持pandas风格索引
        
        这个方法执行pandas风格的索引操作，如切片、选择等，并返回一个新的
        Drawdowns实例。它会正确处理包装器、记录数组和时间序列数据的索引。
        
        参数：
            pd_indexing_func (tp.PandasIndexingFunc): pandas索引函数
            **kwargs: 传递给索引函数的额外参数
        
        返回：
            DrawdownsT: 新的Drawdowns实例，包含索引后的数据
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建多列的价格数据
        prices = pd.DataFrame({
            'AAPL': [100, 105, 98, 95, 102, 108, 103],
            'GOOGL': [200, 210, 195, 190, 205, 215, 208],
            'TSLA': [150, 160, 145, 140, 155, 165, 158]
        })
        
        drawdowns = vbt.Drawdowns.from_ts(prices)
        
        # 选择特定列
        aapl_dd = drawdowns['AAPL']
        print("AAPL的回撤:", aapl_dd.records_readable)
        
        # 选择多列
        tech_dd = drawdowns[['AAPL', 'GOOGL']]
        print("科技股回撤:", tech_dd.records_readable)
        
        # 使用iloc选择
        first_two_dd = drawdowns.iloc[:, :2]
        print("前两只股票回撤:", first_two_dd.records_readable)
        ```
        """
        # 调用Ranges基类的索引元数据方法获取新的组件
        new_wrapper, new_records_arr, _, col_idxs = \
            Ranges.indexing_func_meta(self, pd_indexing_func, **kwargs)
        
        # 处理时间序列数据的索引
        if self.ts is not None:
            # 如果存在时间序列数据，也需要对其进行相应的索引操作
            # 使用列索引选择对应的时间序列数据
            new_ts = new_wrapper.wrap(self.ts.values[:, col_idxs], group_by=False)
        else:
            # 如果没有时间序列数据，设置为None
            new_ts = None
        
        # 创建并返回新的Drawdowns实例
        return self.replace(
            wrapper=new_wrapper,           # 新的包装器
            records_arr=new_records_arr,   # 新的记录数组
            ts=new_ts                      # 新的时间序列数据
        )

    @classmethod
    def from_ts(cls: tp.Type[DrawdownsT],
                ts: tp.ArrayLike,
                attach_ts: bool = True,
                wrapper_kwargs: tp.KwargsLike = None,
                **kwargs) -> DrawdownsT:
        """
        从时间序列创建Drawdowns对象 - 自动回撤识别的核心方法
        
        这个类方法从价格时间序列数据中自动识别回撤，是创建Drawdowns对象的主要方式。
        它会分析价格序列，找出所有的峰值、谷底和恢复点，并生成完整的回撤记录。
        
        识别算法：
        1. 扫描价格序列，识别局部最大值（峰值）
        2. 从每个峰值开始，找到后续的最低点（谷底）
        3. 从谷底开始，找到价格回到峰值的点（恢复点）
        4. 区分已恢复回撤和活跃回撤
        
        参数：
            ts (tp.ArrayLike): 输入的价格时间序列数据
                可以是Series、DataFrame或数组
            attach_ts (bool, 可选): 是否附加原始时间序列，默认True
            wrapper_kwargs (tp.KwargsLike, 可选): 传递给ArrayWrapper的参数
            **kwargs: 传递给Drawdowns构造函数的额外参数
        
        返回：
            DrawdownsT: 从时间序列创建的Drawdowns对象
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        import numpy as np
        
        # 示例1：从简单价格序列创建回撤
        price = pd.Series([100, 105, 98, 95, 102, 108, 103])
        drawdowns = vbt.Drawdowns.from_ts(price)
        
        print("回撤记录:")
        print(drawdowns.records_readable)
        
        # 示例2：从实际股票数据创建回撤
        # 假设我们有股票价格数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        price = pd.Series(100 * np.cumprod(1 + returns), index=dates)
        
        # 创建回撤记录
        drawdowns = vbt.Drawdowns.from_ts(price, wrapper_kwargs=dict(freq='D'))
        
        # 分析回撤
        print("最大回撤:", drawdowns.max_drawdown())
        print("平均回撤:", drawdowns.avg_drawdown())
        print("回撤次数:", drawdowns.count())
        
        # 示例3：多资产回撤分析
        portfolio = pd.DataFrame({
            'Stock_A': [100, 105, 98, 95, 102, 108, 103],
            'Stock_B': [50, 52, 48, 46, 49, 51, 50],
            'Stock_C': [200, 210, 195, 190, 205, 215, 208]
        })
        
        # 创建投资组合回撤
        portfolio_dd = vbt.Drawdowns.from_ts(portfolio)
        
        # 分析各资产回撤
        for col in portfolio.columns:
            dd = portfolio_dd[col]
            print(f"{col} - 最大回撤: {dd.max_drawdown():.2%}")
        
        # 示例4：不保留原始时间序列数据
        drawdowns_no_ts = vbt.Drawdowns.from_ts(price, attach_ts=False)
        print("是否保留原始数据:", drawdowns_no_ts.ts is not None)
        ```
        """
        # 将时间序列转换为pandas对象
        ts_pd = to_pd_array(ts)
        
        # 使用numba编译的函数识别回撤
        # 这个函数会分析价格序列，找出所有的峰值、谷底和恢复点
        records_arr = nb.get_drawdowns_nb(to_2d_array(ts_pd))
        
        # 创建数组包装器
        wrapper = ArrayWrapper.from_obj(ts_pd, **merge_dicts({}, wrapper_kwargs))
        
        # 创建并返回Drawdowns对象
        return cls(
            wrapper,                                    # 数组包装器
            records_arr,                               # 回撤记录数组
            ts=ts_pd if attach_ts else None,          # 可选的时间序列数据
            **kwargs                                   # 其他参数
        )

    @property
    def ts(self) -> tp.Optional[tp.SeriesFrame]:
        """
        时间序列属性 - 返回构建记录时使用的原始时间序列数据
        
        这个属性返回在创建Drawdowns对象时传入的原始时间序列数据。
        如果在初始化时没有提供时间序列数据，则返回None。
        
        返回：
            tp.Optional[tp.SeriesFrame]: 原始时间序列数据，如果不存在则为None
        
        用途：
        - 用于绘图和可视化
        - 用于验证回撤记录的正确性
        - 用于进一步的时间序列分析
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建价格时间序列
        price = pd.Series([100, 105, 98, 95, 102, 108, 103], 
                         index=pd.date_range('2023-01-01', periods=7, freq='D'))
        
        # 从时间序列创建回撤，保留原始数据
        drawdowns = vbt.Drawdowns.from_ts(price, attach_ts=True)
        
        # 访问原始时间序列
        original_price = drawdowns.ts
        print("原始价格序列:")
        print(original_price)
        
        # 创建回撤时不保留原始数据
        drawdowns_no_ts = vbt.Drawdowns.from_ts(price, attach_ts=False)
        print("是否保留原始数据:", drawdowns_no_ts.ts is not None)
        ```
        """
        return self._ts

    # ############# 回撤幅度分析 ############# #

    @cached_property
    def drawdown(self) -> MappedArray:
        """
        回撤幅度属性 - 计算每个回撤的幅度
        
        这个属性计算每个回撤的幅度，即从峰值到谷底的价格跌幅百分比。
        回撤幅度是负值，表示价格下跌的程度。
        
        返回：
            MappedArray: 包含每个回撤幅度的映射数组
        
        计算公式：
        drawdown = (valley_val - peak_val) / peak_val
        
        注意：
        - 回撤幅度为负值，-0.1表示10%的回撤
        - 同时考虑已恢复和活跃回撤
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建价格数据
        price = pd.Series([100, 105, 98, 95, 102, 108, 103])
        drawdowns = vbt.Drawdowns.from_ts(price)
        
        # 获取回撤幅度
        dd_values = drawdowns.drawdown
        print("回撤幅度:")
        print(dd_values.values)
        
        # 转换为百分比显示
        dd_pct = dd_values.values * 100
        print("回撤幅度（百分比）:")
        for i, dd in enumerate(dd_pct):
            print(f"回撤 {i}: {dd:.2f}%")
        
        # 找出最大回撤
        max_dd = dd_values.min()
        print(f"最大回撤: {max_dd:.2%}")
        
        # 分析回撤分布
        print("回撤统计:")
        print(f"平均回撤: {dd_values.mean():.2%}")
        print(f"回撤标准差: {dd_values.std():.2%}")
        print(f"回撤次数: {dd_values.count()}")
        ```
        """
        # 使用numba编译的函数计算回撤幅度
        # 参见 vectorbt.generic.nb.dd_drawdown_nb
        # 计算公式：(valley_val - peak_val) / peak_val
        drawdown = nb.dd_drawdown_nb(
            self.get_field_arr('peak_val'),    # 峰值价格数组
            self.get_field_arr('valley_val')   # 谷底价格数组
        )
        
        # 将结果转换为映射数组并返回
        return self.map_array(drawdown)

    @cached_method
    def avg_drawdown(self, group_by: tp.GroupByLike = None,
                     wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        平均回撤幅度 (ADD - Average Drawdown) - 计算所有回撤的平均幅度
        
        这个方法计算所有回撤的平均幅度，是评估策略或投资组合风险的重要指标。
        平均回撤能够反映策略的整体回撤水平。
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
            **kwargs: 传递给mean方法的其他参数
        
        返回：
            tp.MaybeSeries: 平均回撤幅度（负值）
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建价格数据
        price = pd.Series([100, 105, 98, 95, 102, 108, 103, 100])
        drawdowns = vbt.Drawdowns.from_ts(price)
        
        # 计算平均回撤
        avg_dd = drawdowns.avg_drawdown()
        print(f"平均回撤: {avg_dd:.2%}")
        
        # 多资产投资组合的平均回撤
        portfolio = pd.DataFrame({
            'Stock_A': [100, 105, 98, 95, 102, 108, 103],
            'Stock_B': [50, 52, 48, 46, 49, 51, 50],
            'Stock_C': [200, 210, 195, 190, 205, 215, 208]
        })
        
        portfolio_dd = vbt.Drawdowns.from_ts(portfolio)
        avg_dd_multi = portfolio_dd.avg_drawdown()
        print("各资产平均回撤:")
        print(avg_dd_multi)
        
        # 分组分析
        avg_dd_grouped = portfolio_dd.avg_drawdown(
            group_by=['Growth', 'Value', 'Growth']
        )
        print("按投资风格分组的平均回撤:")
        print(avg_dd_grouped)
        ```
        """
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='avg_drawdown'), wrap_kwargs)
        
        # 计算回撤幅度的平均值
        # 基于 Drawdowns.drawdown 属性
        return self.drawdown.mean(group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_method
    def max_drawdown(self, group_by: tp.GroupByLike = None,
                     wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        最大回撤幅度 (MDD - Maximum Drawdown) - 计算历史最大回撤
        
        这个方法计算历史最大回撤，是量化金融中最重要的风险指标之一。
        最大回撤反映了策略或投资组合可能面临的最大损失。
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
            **kwargs: 传递给min方法的其他参数
        
        返回：
            tp.MaybeSeries: 最大回撤幅度（负值）
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        import numpy as np
        
        # 创建模拟股票价格数据
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # 一年的日收益率
        price = pd.Series(100 * np.cumprod(1 + returns))
        
        # 创建回撤分析
        drawdowns = vbt.Drawdowns.from_ts(price)
        
        # 计算最大回撤
        max_dd = drawdowns.max_drawdown()
        print(f"最大回撤: {max_dd:.2%}")
        
        # 风险评估
        if max_dd < -0.2:
            print("警告：最大回撤超过20%，风险较高")
        elif max_dd < -0.1:
            print("注意：最大回撤在10-20%之间，风险适中")
        else:
            print("风险较低：最大回撤小于10%")
        
        # 投资组合风险分析
        portfolio = pd.DataFrame({
            'Conservative': [100, 102, 101, 103, 102, 104, 103],
            'Aggressive': [100, 110, 95, 85, 105, 120, 108],
            'Balanced': [100, 105, 98, 95, 102, 108, 103]
        })
        
        portfolio_dd = vbt.Drawdowns.from_ts(portfolio)
        max_dd_multi = portfolio_dd.max_drawdown()
        
        print("投资组合最大回撤:")
        for asset, dd in max_dd_multi.items():
            print(f"{asset}: {dd:.2%}")
        
        # 找出风险最高的资产
        riskiest_asset = max_dd_multi.idxmin()
        print(f"风险最高的资产: {riskiest_asset}")
        ```
        """
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='max_drawdown'), wrap_kwargs)
        
        # 计算回撤幅度的最小值（因为回撤是负值，最小值就是最大回撤）
        # 基于 Drawdowns.drawdown 属性
        return self.drawdown.min(group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    # ############# 回撤恢复分析 ############# #

    @cached_property
    def recovery_return(self) -> MappedArray:
        """
        回撤恢复收益率属性 - 计算每个回撤的恢复收益率
        
        这个属性计算每个回撤从谷底到结束点的恢复收益率。
        对于已恢复的回撤，这是从谷底回到峰值的收益率；
        对于活跃回撤，这是从谷底到当前点的收益率。
        
        返回：
            MappedArray: 包含每个回撤恢复收益率的映射数组
        
        计算公式：
        recovery_return = (end_val - valley_val) / valley_val
        
        注意：
        - 恢复收益率为正值，表示从谷底的上涨幅度
        - 同时考虑已恢复和活跃回撤
        - 对于活跃回撤，恢复收益率可能为负值
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建价格数据
        price = pd.Series([100, 105, 98, 95, 102, 108, 103])
        drawdowns = vbt.Drawdowns.from_ts(price)
        
        # 获取恢复收益率
        recovery_returns = drawdowns.recovery_return
        print("恢复收益率:")
        print(recovery_returns.values)
        
        # 转换为百分比显示
        recovery_pct = recovery_returns.values * 100
        print("恢复收益率（百分比）:")
        for i, ret in enumerate(recovery_pct):
            print(f"回撤 {i}: {ret:.2f}%")
        
        # 分析恢复能力
        print("恢复统计:")
        print(f"平均恢复收益率: {recovery_returns.mean():.2%}")
        print(f"最大恢复收益率: {recovery_returns.max():.2%}")
        print(f"恢复收益率标准差: {recovery_returns.std():.2%}")
        
        # 分析已恢复vs活跃回撤的恢复能力
        recovered_dd = drawdowns.recovered
        active_dd = drawdowns.active
        
        if recovered_dd.count() > 0:
            print(f"已恢复回撤的平均恢复收益率: {recovered_dd.recovery_return.mean():.2%}")
        if active_dd.count() > 0:
            print(f"活跃回撤的当前恢复收益率: {active_dd.recovery_return.mean():.2%}")
        ```
        """
        # 使用numba编译的函数计算恢复收益率
        # 参见 vectorbt.generic.nb.dd_recovery_return_nb
        # 计算公式：(end_val - valley_val) / valley_val
        recovery_return = nb.dd_recovery_return_nb(
            self.get_field_arr('valley_val'),   # 谷底价格数组
            self.get_field_arr('end_val')       # 结束价格数组
        )
        
        # 将结果转换为映射数组并返回
        return self.map_array(recovery_return)

    @cached_method
    def avg_recovery_return(self, group_by: tp.GroupByLike = None,
                            wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        平均恢复收益率 - 计算所有回撤的平均恢复收益率
        
        这个方法计算所有回撤的平均恢复收益率，反映了策略或投资组合
        从低点反弹的平均能力。这是评估策略韧性的重要指标。
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
            **kwargs: 传递给mean方法的其他参数
        
        返回：
            tp.MaybeSeries: 平均恢复收益率（正值）
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建价格数据
        price = pd.Series([100, 105, 98, 95, 102, 108, 103, 100])
        drawdowns = vbt.Drawdowns.from_ts(price)
        
        # 计算平均恢复收益率
        avg_recovery = drawdowns.avg_recovery_return()
        print(f"平均恢复收益率: {avg_recovery:.2%}")
        
        # 多资产投资组合的恢复能力分析
        portfolio = pd.DataFrame({
            'Growth_Stock': [100, 110, 95, 85, 105, 120, 115],
            'Value_Stock': [100, 102, 98, 96, 100, 104, 103],
            'Volatile_Stock': [100, 120, 80, 70, 110, 130, 100]
        })
        
        portfolio_dd = vbt.Drawdowns.from_ts(portfolio)
        avg_recovery_multi = portfolio_dd.avg_recovery_return()
        
        print("各资产平均恢复收益率:")
        for asset, recovery in avg_recovery_multi.items():
            print(f"{asset}: {recovery:.2%}")
        
        # 评估恢复能力
        best_recovery = avg_recovery_multi.idxmax()
        print(f"恢复能力最强的资产: {best_recovery}")
        
        # 分组分析
        avg_recovery_grouped = portfolio_dd.avg_recovery_return(
            group_by=['Growth', 'Value', 'Volatile']
        )
        print("按投资风格分组的平均恢复收益率:")
        print(avg_recovery_grouped)
        ```
        """
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='avg_recovery_return'), wrap_kwargs)
        
        # 计算恢复收益率的平均值
        # 基于 Drawdowns.recovery_return 属性
        return self.recovery_return.mean(group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_method
    def max_recovery_return(self, group_by: tp.GroupByLike = None,
                            wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        最大恢复收益率 - 计算历史最大恢复收益率
        
        这个方法计算历史最大恢复收益率，反映了策略或投资组合
        从低点反弹的最大能力。这是评估策略爆发力的重要指标。
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
            **kwargs: 传递给max方法的其他参数
        
        返回：
            tp.MaybeSeries: 最大恢复收益率（正值）
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        import numpy as np
        
        # 创建模拟价格数据，包含大幅回撤和强劲反弹
        price_data = [100, 120, 90, 70, 110, 130, 100, 80, 140, 120]
        price = pd.Series(price_data)
        
        # 创建回撤分析
        drawdowns = vbt.Drawdowns.from_ts(price)
        
        # 计算最大恢复收益率
        max_recovery = drawdowns.max_recovery_return()
        print(f"最大恢复收益率: {max_recovery:.2%}")
        
        # 分析恢复能力
        if max_recovery > 0.5:
            print("优秀：最大恢复收益率超过50%，反弹能力强")
        elif max_recovery > 0.3:
            print("良好：最大恢复收益率在30-50%之间")
        else:
            print("一般：最大恢复收益率低于30%")
        
        # 投资组合恢复能力分析
        portfolio = pd.DataFrame({
            'Defensive': [100, 102, 98, 96, 100, 104, 103, 101],
            'Cyclical': [100, 110, 85, 75, 105, 125, 115, 95],
            'Technology': [100, 130, 80, 60, 120, 150, 130, 100]
        })
        
        portfolio_dd = vbt.Drawdowns.from_ts(portfolio)
        max_recovery_multi = portfolio_dd.max_recovery_return()
        
        print("投资组合最大恢复收益率:")
        for asset, recovery in max_recovery_multi.items():
            print(f"{asset}: {recovery:.2%}")
        
        # 找出反弹能力最强的资产
        strongest_recovery = max_recovery_multi.idxmax()
        print(f"反弹能力最强的资产: {strongest_recovery}")
        
        # 计算恢复收益率与回撤的比率
        max_dd = portfolio_dd.max_drawdown()
        recovery_ratio = max_recovery_multi / abs(max_dd)
        print("恢复收益率/最大回撤比率:")
        print(recovery_ratio)
        ```
        """
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='max_recovery_return'), wrap_kwargs)
        
        # 计算恢复收益率的最大值
        # 基于 Drawdowns.recovery_return 属性
        return self.recovery_return.max(group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    # ############# 回撤持续时间分析 ############# #

    @cached_property
    def decline_duration(self) -> MappedArray:
        """
        下跌持续时间属性 - 计算每个回撤的下跌阶段持续时间
        
        这个属性计算每个回撤从峰值到谷底的下跌阶段持续时间。
        下跌持续时间反映了价格从高点跌到低点所需的时间。
        
        返回：
            MappedArray: 包含每个回撤下跌持续时间的映射数组
        
        计算公式：
        decline_duration = valley_idx - start_idx
        
        注意：
        - 时间单位为索引单位（如交易日、小时等）
        - 同时考虑已恢复和活跃回撤
        - 可以通过wrapper转换为实际时间差
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建带时间索引的价格数据
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        price = pd.Series([100, 105, 98, 95, 102, 108, 103, 100, 95, 105], 
                         index=dates[:10])
        
        # 创建回撤分析
        drawdowns = vbt.Drawdowns.from_ts(price, wrapper_kwargs=dict(freq='D'))
        
        # 获取下跌持续时间
        decline_durations = drawdowns.decline_duration
        print("下跌持续时间（索引单位）:")
        print(decline_durations.values)
        
        # 转换为时间差
        decline_timedelta = decline_durations.to_timedelta()
        print("下跌持续时间（时间差）:")
        print(decline_timedelta.values)
        
        # 分析下跌速度
        print("下跌持续时间统计:")
        print(f"平均下跌持续时间: {decline_durations.mean():.1f}天")
        print(f"最长下跌持续时间: {decline_durations.max():.1f}天")
        print(f"最短下跌持续时间: {decline_durations.min():.1f}天")
        
        # 分析下跌速度与回撤幅度的关系
        drawdown_values = drawdowns.drawdown
        decline_speed = abs(drawdown_values.values) / decline_durations.values
        print("下跌速度统计:")
        print(f"平均下跌速度: {decline_speed.mean():.3f}% per day")
        print(f"最快下跌速度: {decline_speed.max():.3f}% per day")
        ```
        """
        # 使用numba编译的函数计算下跌持续时间
        # 参见 vectorbt.generic.nb.dd_decline_duration_nb
        # 计算公式：valley_idx - start_idx
        decline_duration = nb.dd_decline_duration_nb(
            self.get_field_arr('start_idx'),    # 起始索引数组（峰值）
            self.get_field_arr('valley_idx')    # 谷底索引数组
        )
        
        # 将结果转换为映射数组并返回
        return self.map_array(decline_duration)

    @cached_property
    def recovery_duration(self) -> MappedArray:
        """
        恢复持续时间属性 - 计算每个回撤的恢复阶段持续时间
        
        这个属性计算每个回撤从谷底到结束点的恢复阶段持续时间。
        恢复持续时间反映了价格从低点回升所需的时间。
        
        返回：
            MappedArray: 包含每个回撤恢复持续时间的映射数组
        
        计算公式：
        recovery_duration = end_idx - valley_idx
        
        注意：
        - 时间单位为索引单位（如交易日、小时等）
        - 同时考虑已恢复和活跃回撤
        - 对于活跃回撤，这是从谷底到当前的时间
        - 值大于下跌持续时间意味着恢复比下跌更慢
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建带时间索引的价格数据
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        price = pd.Series([100, 105, 98, 95, 102, 108, 103, 100, 95, 105], 
                         index=dates[:10])
        
        # 创建回撤分析
        drawdowns = vbt.Drawdowns.from_ts(price, wrapper_kwargs=dict(freq='D'))
        
        # 获取恢复持续时间
        recovery_durations = drawdowns.recovery_duration
        print("恢复持续时间（索引单位）:")
        print(recovery_durations.values)
        
        # 转换为时间差
        recovery_timedelta = recovery_durations.to_timedelta()
        print("恢复持续时间（时间差）:")
        print(recovery_timedelta.values)
        
        # 分析恢复速度
        print("恢复持续时间统计:")
        print(f"平均恢复持续时间: {recovery_durations.mean():.1f}天")
        print(f"最长恢复持续时间: {recovery_durations.max():.1f}天")
        print(f"最短恢复持续时间: {recovery_durations.min():.1f}天")
        
        # 比较下跌和恢复速度
        decline_durations = drawdowns.decline_duration
        recovery_decline_ratio = recovery_durations.values / decline_durations.values
        print("恢复/下跌时间比率:")
        print(f"平均比率: {recovery_decline_ratio.mean():.2f}")
        print(f"最大比率: {recovery_decline_ratio.max():.2f}")
        
        # 分析恢复效率
        recovery_returns = drawdowns.recovery_return
        recovery_efficiency = recovery_returns.values / recovery_durations.values
        print("恢复效率（收益率/天）:")
        print(f"平均恢复效率: {recovery_efficiency.mean():.3f}% per day")
        ```
        """
        # 使用numba编译的函数计算恢复持续时间
        # 参见 vectorbt.generic.nb.dd_recovery_duration_nb
        # 计算公式：end_idx - valley_idx
        recovery_duration = nb.dd_recovery_duration_nb(
            self.get_field_arr('valley_idx'),   # 谷底索引数组
            self.get_field_arr('end_idx')       # 结束索引数组
        )
        
        # 将结果转换为映射数组并返回
        return self.map_array(recovery_duration)

    @cached_property
    def recovery_duration_ratio(self) -> MappedArray:
        """
        恢复持续时间比率属性 - 计算恢复时间与下跌时间的比率
        
        这个属性计算每个回撤的恢复持续时间与下跌持续时间的比率。
        比率大于1表示恢复比下跌更慢，比率小于1表示恢复比下跌更快。
        
        返回：
            MappedArray: 包含每个回撤恢复持续时间比率的映射数组
        
        计算公式：
        recovery_duration_ratio = recovery_duration / decline_duration
        
        注意：
        - 比率为无量纲数值
        - 比率大于1：恢复比下跌慢（V型反转较慢）
        - 比率小于1：恢复比下跌快（V型反转较快）
        - 比率等于1：恢复和下跌速度相同
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建不同反转速度的价格数据
        price_data = [100, 105, 98, 95, 102, 108, 103, 100, 95, 105, 110]
        price = pd.Series(price_data)
        
        # 创建回撤分析
        drawdowns = vbt.Drawdowns.from_ts(price)
        
        # 获取恢复持续时间比率
        recovery_ratios = drawdowns.recovery_duration_ratio
        print("恢复持续时间比率:")
        print(recovery_ratios.values)
        
        # 分析反转特征
        print("反转特征分析:")
        for i, ratio in enumerate(recovery_ratios.values):
            if ratio > 1.5:
                print(f"回撤 {i}: 恢复缓慢（比率={ratio:.2f}）")
            elif ratio > 1.0:
                print(f"回撤 {i}: 恢复较慢（比率={ratio:.2f}）")
            elif ratio < 0.5:
                print(f"回撤 {i}: 恢复快速（比率={ratio:.2f}）")
            else:
                print(f"回撤 {i}: 恢复中等（比率={ratio:.2f}）")
        
        # 统计分析
        print("恢复比率统计:")
        print(f"平均恢复比率: {recovery_ratios.mean():.2f}")
        print(f"最大恢复比率: {recovery_ratios.max():.2f}")
        print(f"最小恢复比率: {recovery_ratios.min():.2f}")
        
        # 投资组合恢复速度比较
        portfolio = pd.DataFrame({
            'Fast_Recovery': [100, 110, 90, 85, 105, 108, 106],
            'Slow_Recovery': [100, 110, 90, 85, 88, 92, 95],
            'V_Shape': [100, 110, 90, 85, 110, 108, 106]
        })
        
        portfolio_dd = vbt.Drawdowns.from_ts(portfolio)
        recovery_ratios_multi = portfolio_dd.recovery_duration_ratio
        
        print("投资组合恢复速度比较:")
        for asset in portfolio.columns:
            if asset in recovery_ratios_multi.columns:
                ratios = recovery_ratios_multi[asset]
                print(f"{asset}: 平均恢复比率 = {ratios.mean():.2f}")
        ```
        """
        # 使用numba编译的函数计算恢复持续时间比率
        # 参见 vectorbt.generic.nb.dd_recovery_duration_ratio_nb
        # 计算公式：recovery_duration / decline_duration
        recovery_duration_ratio = nb.dd_recovery_duration_ratio_nb(
            self.get_field_arr('start_idx'),    # 起始索引数组（峰值）
            self.get_field_arr('valley_idx'),   # 谷底索引数组
            self.get_field_arr('end_idx')       # 结束索引数组
        )
        
        # 将结果转换为映射数组并返回
        return self.map_array(recovery_duration_ratio)

    # ############# 活跃回撤状态分析 ############# #

    @cached_method
    def active_drawdown(self, group_by: tp.GroupByLike = None,
                        wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        活跃回撤幅度 - 计算当前活跃回撤的幅度
        
        这个方法计算最后一个活跃回撤的当前幅度，仅考虑状态为Active的回撤。
        活跃回撤幅度反映了当前正在经历的回撤程度。
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
        
        返回：
            tp.MaybeSeries: 活跃回撤幅度（负值）
        
        注意：
        - 不支持分组操作
        - 只考虑最后一个活跃回撤
        - 如果没有活跃回撤，返回NaN
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建包含活跃回撤的价格数据
        price = pd.Series([100, 105, 98, 95, 97])  # 最后没有回到峰值
        drawdowns = vbt.Drawdowns.from_ts(price)
        
        # 获取活跃回撤幅度
        active_dd = drawdowns.active_drawdown()
        print(f"当前活跃回撤: {active_dd:.2%}")
        
        # 风险监控示例
        if active_dd is not None and active_dd < -0.1:
            print("警告：当前回撤超过10%")
        
        # 多资产监控
        portfolio = pd.DataFrame({
            'Stock_A': [100, 105, 98, 95, 97],
            'Stock_B': [100, 110, 90, 85, 88],
            'Stock_C': [100, 102, 101, 103, 104]  # 无活跃回撤
        })
        
        portfolio_dd = vbt.Drawdowns.from_ts(portfolio)
        
        print("各资产活跃回撤:")
        for col in portfolio.columns:
            try:
                active_dd = portfolio_dd[col].active_drawdown()
                if pd.notna(active_dd):
                    print(f"{col}: {active_dd:.2%}")
                else:
                    print(f"{col}: 无活跃回撤")
            except:
                print(f"{col}: 无活跃回撤")
        
        # 实时风险管理
        risk_threshold = -0.15
        for col in portfolio.columns:
            try:
                active_dd = portfolio_dd[col].active_drawdown()
                if pd.notna(active_dd) and active_dd < risk_threshold:
                    print(f"风险警告：{col}的活跃回撤({active_dd:.2%})超过阈值({risk_threshold:.2%})")
            except:
                pass
        ```
        """
        # 检查是否使用了分组，活跃回撤不支持分组
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='active_drawdown'), wrap_kwargs)
        
        # 获取活跃回撤记录
        active = self.active
        
        # 计算当前活跃回撤的幅度
        curr_end_val = active.end_val.nth(-1, group_by=group_by)      # 当前价格
        curr_peak_val = active.peak_val.nth(-1, group_by=group_by)    # 峰值价格
        curr_drawdown = (curr_end_val - curr_peak_val) / curr_peak_val  # 回撤幅度
        
        # 包装结果并返回
        return self.wrapper.wrap_reduced(curr_drawdown, group_by=group_by, **wrap_kwargs)

    @cached_method
    def active_duration(self, group_by: tp.GroupByLike = None,
                        wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        活跃回撤持续时间 - 计算当前活跃回撤的持续时间
        
        这个方法计算最后一个活跃回撤的持续时间，从峰值开始到当前时间。
        活跃回撤持续时间反映了当前回撤已经持续的时间长度。
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
            **kwargs: 传递给nth方法的其他参数
        
        返回：
            tp.MaybeSeries: 活跃回撤持续时间（时间差格式）
        
        注意：
        - 不支持分组操作
        - 只考虑最后一个活跃回撤
        - 如果没有活跃回撤，返回NaN
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建带时间索引的价格数据
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        price = pd.Series([100, 105, 98, 95, 97, 96, 94, 92, 93, 91], index=dates)
        
        # 创建回撤分析
        drawdowns = vbt.Drawdowns.from_ts(price, wrapper_kwargs=dict(freq='D'))
        
        # 获取活跃回撤持续时间
        active_duration = drawdowns.active_duration()
        print(f"活跃回撤持续时间: {active_duration}")
        
        # 转换为天数
        if pd.notna(active_duration):
            days = active_duration.days
            print(f"活跃回撤已持续 {days} 天")
        
        # 多资产监控
        portfolio = pd.DataFrame({
            'Stock_A': [100, 105, 98, 95, 97, 96, 94],
            'Stock_B': [100, 110, 90, 85, 88, 87, 86],
            'Stock_C': [100, 102, 101, 103, 104, 105, 106]
        }, index=pd.date_range('2023-01-01', periods=7, freq='D'))
        
        portfolio_dd = vbt.Drawdowns.from_ts(portfolio, wrapper_kwargs=dict(freq='D'))
        
        print("各资产活跃回撤持续时间:")
        for col in portfolio.columns:
            try:
                duration = portfolio_dd[col].active_duration()
                if pd.notna(duration):
                    print(f"{col}: {duration}")
                else:
                    print(f"{col}: 无活跃回撤")
            except:
                print(f"{col}: 无活跃回撤")
        
        # 持续时间警告
        duration_threshold = pd.Timedelta(days=5)
        for col in portfolio.columns:
            try:
                duration = portfolio_dd[col].active_duration()
                if pd.notna(duration) and duration > duration_threshold:
                    print(f"持续时间警告：{col}的活跃回撤已持续{duration.days}天")
            except:
                pass
        ```
        """
        # 检查是否使用了分组，活跃回撤不支持分组
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        
        # 设置包装器参数，包括转换为时间差
        wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index='active_duration'), wrap_kwargs)
        
        # 获取活跃回撤的持续时间（最后一个活跃回撤）
        return self.active.duration.nth(-1, group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_method
    def active_recovery(self, group_by: tp.GroupByLike = None,
                        wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        活跃回撤恢复程度 - 计算当前活跃回撤的恢复程度
        
        这个方法计算最后一个活跃回撤的恢复程度，即从谷底到当前价格的恢复比例。
        恢复程度用于评估活跃回撤的恢复进展，值为0表示仍在谷底，值为1表示完全恢复。
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
        
        返回：
            tp.MaybeSeries: 活跃回撤恢复程度（0-1之间的值）
        
        计算公式：
        recovery = (current_price - valley_price) / (peak_price - valley_price)
        
        注意：
        - 不支持分组操作
        - 只考虑最后一个活跃回撤
        - 值为0：仍在谷底
        - 值为1：完全恢复到峰值
        - 值大于1：超过了原峰值
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建包含活跃回撤的价格数据
        price = pd.Series([100, 105, 98, 95, 97, 99, 98])  # 正在恢复中
        drawdowns = vbt.Drawdowns.from_ts(price)
        
        # 获取活跃回撤恢复程度
        active_recovery = drawdowns.active_recovery()
        print(f"活跃回撤恢复程度: {active_recovery:.2%}")
        
        # 恢复程度解释
        if active_recovery is not None:
            if active_recovery < 0.2:
                print("恢复程度：刚从谷底开始恢复")
            elif active_recovery < 0.5:
                print("恢复程度：部分恢复")
            elif active_recovery < 0.8:
                print("恢复程度：大部分恢复")
            elif active_recovery < 1.0:
                print("恢复程度：接近完全恢复")
            else:
                print("恢复程度：已超过原峰值")
        
        # 多资产恢复监控
        portfolio = pd.DataFrame({
            'Recovery_Fast': [100, 105, 95, 90, 98, 103, 102],
            'Recovery_Slow': [100, 105, 95, 90, 92, 93, 94],
            'Still_Declining': [100, 105, 95, 90, 88, 85, 83]
        })
        
        portfolio_dd = vbt.Drawdowns.from_ts(portfolio)
        
        print("各资产活跃回撤恢复程度:")
        for col in portfolio.columns:
            try:
                recovery = portfolio_dd[col].active_recovery()
                if pd.notna(recovery):
                    print(f"{col}: {recovery:.2%}")
                    if recovery > 0.8:
                        print(f"  -> {col} 接近完全恢复")
                    elif recovery < 0.2:
                        print(f"  -> {col} 恢复缓慢")
                else:
                    print(f"{col}: 无活跃回撤")
            except:
                print(f"{col}: 无活跃回撤")
        
        # 恢复进度预警
        recovery_threshold = 0.1
        for col in portfolio.columns:
            try:
                recovery = portfolio_dd[col].active_recovery()
                if pd.notna(recovery) and recovery < recovery_threshold:
                    print(f"恢复预警：{col}的恢复程度({recovery:.2%})低于阈值({recovery_threshold:.2%})")
            except:
                pass
        ```
        """
        # 检查是否使用了分组，活跃回撤不支持分组
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='active_recovery'), wrap_kwargs)
        
        # 获取活跃回撤记录
        active = self.active
        
        # 计算当前活跃回撤的恢复程度
        curr_peak_val = active.peak_val.nth(-1, group_by=group_by)        # 峰值价格
        curr_end_val = active.end_val.nth(-1, group_by=group_by)          # 当前价格
        curr_valley_val = active.valley_val.nth(-1, group_by=group_by)    # 谷底价格
        
        # 计算恢复程度：(当前价格 - 谷底价格) / (峰值价格 - 谷底价格)
        curr_recovery = (curr_end_val - curr_valley_val) / (curr_peak_val - curr_valley_val)
        
        # 包装结果并返回
        return self.wrapper.wrap_reduced(curr_recovery, group_by=group_by, **wrap_kwargs)

    @cached_method
    def active_recovery_return(self, group_by: tp.GroupByLike = None,
                               wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        活跃回撤恢复收益率 - 计算当前活跃回撤的恢复收益率
        
        这个方法计算最后一个活跃回撤从谷底到当前价格的恢复收益率。
        恢复收益率反映了活跃回撤从最低点开始的回升表现。
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
            **kwargs: 传递给nth方法的其他参数
        
        返回：
            tp.MaybeSeries: 活跃回撤恢复收益率（正值表示从谷底上涨）
        
        注意：
        - 不支持分组操作
        - 只考虑最后一个活跃回撤
        - 如果没有活跃回撤，返回NaN
        - 正值表示从谷底有所回升，负值表示仍在下跌
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建包含活跃回撤的价格数据
        price = pd.Series([100, 105, 98, 95, 97, 99, 98])  # 从95的谷底恢复到98
        drawdowns = vbt.Drawdowns.from_ts(price)
        
        # 获取活跃回撤恢复收益率
        active_recovery_return = drawdowns.active_recovery_return()
        print(f"活跃回撤恢复收益率: {active_recovery_return:.2%}")
        
        # 恢复收益率解释
        if active_recovery_return is not None:
            if active_recovery_return > 0:
                print("从谷底开始恢复")
            elif active_recovery_return == 0:
                print("仍在谷底")
            else:
                print("仍在下跌")
        
        # 多资产恢复收益率监控
        portfolio = pd.DataFrame({
            'Recovering': [100, 105, 95, 90, 95, 98, 96],
            'Still_Falling': [100, 105, 95, 90, 88, 85, 83],
            'Strong_Recovery': [100, 105, 95, 90, 100, 103, 101]
        })
        
        portfolio_dd = vbt.Drawdowns.from_ts(portfolio)
        
        print("各资产活跃回撤恢复收益率:")
        for col in portfolio.columns:
            try:
                recovery_return = portfolio_dd[col].active_recovery_return()
                if pd.notna(recovery_return):
                    print(f"{col}: {recovery_return:.2%}")
                    if recovery_return > 0.1:
                        print(f"  -> {col} 强劲恢复")
                    elif recovery_return > 0:
                        print(f"  -> {col} 缓慢恢复")
                    elif recovery_return < -0.05:
                        print(f"  -> {col} 仍在恶化")
                else:
                    print(f"{col}: 无活跃回撤")
            except:
                print(f"{col}: 无活跃回撤")
        
        # 恢复收益率预警
        recovery_threshold = -0.05
        for col in portfolio.columns:
            try:
                recovery_return = portfolio_dd[col].active_recovery_return()
                if pd.notna(recovery_return) and recovery_return < recovery_threshold:
                    print(f"恶化预警：{col}的恢复收益率({recovery_return:.2%})低于阈值({recovery_threshold:.2%})")
            except:
                pass
        ```
        """
        # 检查是否使用了分组，活跃回撤不支持分组
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='active_recovery_return'), wrap_kwargs)
        
        # 获取活跃回撤的恢复收益率（最后一个活跃回撤）
        return self.active.recovery_return.nth(-1, group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_method
    def active_recovery_duration(self, group_by: tp.GroupByLike = None,
                                 wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        活跃回撤恢复持续时间 - 计算当前活跃回撤的恢复阶段持续时间
        
        这个方法计算最后一个活跃回撤从谷底到当前时间的恢复持续时间。
        恢复持续时间反映了活跃回撤在恢复阶段已经持续的时间长度。
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
            **kwargs: 传递给nth方法的其他参数
        
        返回：
            tp.MaybeSeries: 活跃回撤恢复持续时间（时间差格式）
        
        注意：
        - 不支持分组操作
        - 只考虑最后一个活跃回撤
        - 如果没有活跃回撤，返回NaN
        - 时间从谷底开始计算到当前时间
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建带时间索引的价格数据
        dates = pd.date_range('2023-01-01', periods=15, freq='D')
        price = pd.Series([100, 105, 98, 95, 97, 99, 98, 100, 99, 97, 96, 98, 99, 97, 95], 
                         index=dates)
        
        # 创建回撤分析
        drawdowns = vbt.Drawdowns.from_ts(price, wrapper_kwargs=dict(freq='D'))
        
        # 获取活跃回撤恢复持续时间
        active_recovery_duration = drawdowns.active_recovery_duration()
        print(f"活跃回撤恢复持续时间: {active_recovery_duration}")
        
        # 转换为天数
        if pd.notna(active_recovery_duration):
            days = active_recovery_duration.days
            print(f"从谷底开始恢复已持续 {days} 天")
        
        # 多资产恢复持续时间监控
        portfolio = pd.DataFrame({
            'Quick_Recovery': [100, 105, 95, 90, 95, 98, 100],
            'Slow_Recovery': [100, 105, 95, 90, 91, 92, 93],
            'No_Recovery': [100, 105, 95, 90, 89, 88, 87]
        }, index=pd.date_range('2023-01-01', periods=7, freq='D'))
        
        portfolio_dd = vbt.Drawdowns.from_ts(portfolio, wrapper_kwargs=dict(freq='D'))
        
        print("各资产活跃回撤恢复持续时间:")
        for col in portfolio.columns:
            try:
                duration = portfolio_dd[col].active_recovery_duration()
                if pd.notna(duration):
                    print(f"{col}: {duration}")
                    if duration.days > 5:
                        print(f"  -> {col} 恢复时间较长")
                    elif duration.days > 2:
                        print(f"  -> {col} 恢复时间适中")
                    else:
                        print(f"  -> {col} 恢复较快")
                else:
                    print(f"{col}: 无活跃回撤")
            except:
                print(f"{col}: 无活跃回撤")
        
        # 恢复持续时间预警
        duration_threshold = pd.Timedelta(days=10)
        for col in portfolio.columns:
            try:
                duration = portfolio_dd[col].active_recovery_duration()
                if pd.notna(duration) and duration > duration_threshold:
                    print(f"恢复缓慢预警：{col}的恢复持续时间({duration.days}天)超过阈值({duration_threshold.days}天)")
            except:
                pass
        ```
        """
        # 检查是否使用了分组，活跃回撤不支持分组
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        
        # 设置包装器参数，包括转换为时间差
        wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index='active_recovery_duration'), wrap_kwargs)
        
        # 获取活跃回撤的恢复持续时间（最后一个活跃回撤）
        return self.active.recovery_duration.nth(-1, group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    # ############# 统计分析配置 ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """
        统计分析默认配置 - 定义Drawdowns.stats的默认参数
        
        这个属性定义了统计分析方法的默认配置，合并了Ranges基类的配置
        和drawdowns模块特有的配置。
        
        返回：
            tp.Kwargs: 统计分析的默认参数字典
        
        配置来源：
        - Ranges基类的stats_defaults
        - settings中的drawdowns.stats配置
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建回撤对象
        drawdowns = vbt.Drawdowns.from_ts(some_price_series)
        
        # 查看默认配置
        defaults = drawdowns.stats_defaults
        print("统计分析默认配置:")
        for key, value in defaults.items():
            print(f"  {key}: {value}")
        
        # 使用默认配置进行统计分析
        stats = drawdowns.stats()
        print("统计结果:")
        print(stats)
        ```
        """
        # 从设置中获取drawdowns模块的统计配置
        from vectorbt._settings import settings
        drawdowns_stats_cfg = settings['drawdowns']['stats']

        # 合并基类配置和drawdowns特有配置
        return merge_dicts(
            Ranges.stats_defaults.__get__(self),  # 获取Ranges基类的默认配置
            drawdowns_stats_cfg                   # 合并drawdowns特有配置
        )

    # 定义回撤记录的统计指标配置
    # 这个配置定义了所有可用的统计指标及其计算方法
    _metrics: tp.ClassVar[Config] = Config(
        dict(
            # ========== 基础时间信息指标 ==========
            
            # 起始时间指标
            start=dict(
                title='Start',                                # 指标标题
                calc_func=lambda self: self.wrapper.index[0],    # 计算函数：返回第一个索引
                agg_func=None,                                # 聚合函数：无需聚合
                tags='wrapper'                                # 标签：属于wrapper相关指标
            ),
            
            # 结束时间指标
            end=dict(
                title='End',                                  # 指标标题
                calc_func=lambda self: self.wrapper.index[-1],   # 计算函数：返回最后一个索引
                agg_func=None,                                # 聚合函数：无需聚合
                tags='wrapper'                                # 标签：属于wrapper相关指标
            ),
            
            # 总时间段指标
            period=dict(
                title='Period',                               # 指标标题
                calc_func=lambda self: len(self.wrapper.index),  # 计算函数：返回索引长度
                apply_to_timedelta=True,                      # 应用时间差转换
                agg_func=None,                                # 聚合函数：无需聚合
                tags='wrapper'                                # 标签：属于wrapper相关指标
            ),
            
            # ========== 覆盖率指标 ==========
            
            # 覆盖率指标（百分比）
            coverage=dict(
                title='Coverage [%]',                         # 指标标题
                calc_func='coverage',                         # 计算函数：调用coverage方法
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理：转换为百分比
                tags=['ranges', 'duration']                   # 标签：范围和持续时间相关
            ),
            
            # ========== 记录数量指标 ==========
            
            # 总记录数指标
            total_records=dict(
                title='Total Records',                        # 指标标题
                calc_func='count',                            # 计算函数：调用count方法
                tags='records'                                # 标签：记录相关指标
            ),
            
            # 已恢复回撤数量指标
            total_recovered=dict(
                title='Total Recovered Drawdowns',           # 指标标题
                calc_func='recovered.count',                  # 计算函数：调用recovered.count方法
                tags='drawdowns'                              # 标签：回撤相关指标
            ),
            
            # 活跃回撤数量指标
            total_active=dict(
                title='Total Active Drawdowns',              # 指标标题
                calc_func='active.count',                     # 计算函数：调用active.count方法
                tags='drawdowns'                              # 标签：回撤相关指标
            ),
            # ========== 活跃回撤指标 ==========
            
            # 活跃回撤幅度指标
            active_dd=dict(
                title='Active Drawdown [%]',                  # 指标标题
                calc_func='active_drawdown',                  # 计算函数：调用active_drawdown方法
                post_calc_func=lambda self, out, settings: -out * 100,  # 后处理：转换为正数百分比
                check_is_not_grouped=True,                    # 检查是否非分组
                tags=['drawdowns', 'active']                  # 标签：回撤和活跃相关
            ),
            
            # 活跃回撤持续时间指标
            active_duration=dict(
                title='Active Duration',                      # 指标标题
                calc_func='active_duration',                  # 计算函数：调用active_duration方法
                fill_wrap_kwargs=True,                        # 填充包装器参数
                check_is_not_grouped=True,                    # 检查是否非分组
                tags=['drawdowns', 'active', 'duration']      # 标签：回撤、活跃和持续时间相关
            ),
            
            # 活跃回撤恢复程度指标
            active_recovery=dict(
                title='Active Recovery [%]',                 # 指标标题
                calc_func='active_recovery',                 # 计算函数：调用active_recovery方法
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理：转换为百分比
                check_is_not_grouped=True,                   # 检查是否非分组
                tags=['drawdowns', 'active']                 # 标签：回撤和活跃相关
            ),
            
            # 活跃回撤恢复收益率指标
            active_recovery_return=dict(
                title='Active Recovery Return [%]',          # 指标标题
                calc_func='active_recovery_return',          # 计算函数：调用active_recovery_return方法
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理：转换为百分比
                check_is_not_grouped=True,                   # 检查是否非分组
                tags=['drawdowns', 'active']                 # 标签：回撤和活跃相关
            ),
            
            # 活跃回撤恢复持续时间指标
            active_recovery_duration=dict(
                title='Active Recovery Duration',            # 指标标题
                calc_func='active_recovery_duration',        # 计算函数：调用active_recovery_duration方法
                fill_wrap_kwargs=True,                       # 填充包装器参数
                check_is_not_grouped=True,                   # 检查是否非分组
                tags=['drawdowns', 'active', 'duration']     # 标签：回撤、活跃和持续时间相关
            ),
            
            # ========== 回撤幅度指标 ==========
            
            # 最大回撤幅度指标
            max_dd=dict(
                title='Max Drawdown [%]',                     # 指标标题：最大回撤百分比
                calc_func=RepEval("'max_drawdown' if incl_active else 'recovered.max_drawdown'"),  # 计算函数：根据是否包含活跃回撤选择
                post_calc_func=lambda self, out, settings: -out * 100,  # 后处理：转换为正数百分比
                tags=RepEval("['drawdowns'] if incl_active else ['drawdowns', 'recovered']")  # 标签：根据设置选择
            ),
            
            # 平均回撤幅度指标
            avg_dd=dict(
                title='Avg Drawdown [%]',                     # 指标标题：平均回撤百分比
                calc_func=RepEval("'avg_drawdown' if incl_active else 'recovered.avg_drawdown'"),  # 计算函数：根据是否包含活跃回撤选择
                post_calc_func=lambda self, out, settings: -out * 100,  # 后处理：转换为正数百分比
                tags=RepEval("['drawdowns'] if incl_active else ['drawdowns', 'recovered']")  # 标签：根据设置选择
            ),
            
            # ========== 回撤持续时间指标 ==========
            
            # 最大回撤持续时间指标
            max_dd_duration=dict(
                title='Max Drawdown Duration',               # 指标标题：最大回撤持续时间
                calc_func=RepEval("'max_duration' if incl_active else 'recovered.max_duration'"),  # 计算函数：根据是否包含活跃回撤选择
                fill_wrap_kwargs=True,                       # 填充包装器参数
                tags=RepEval("['drawdowns', 'duration'] if incl_active else ['drawdowns', 'recovered', 'duration']")  # 标签：根据设置选择
            ),
            
            # 平均回撤持续时间指标
            avg_dd_duration=dict(
                title='Avg Drawdown Duration',               # 指标标题：平均回撤持续时间
                calc_func=RepEval("'avg_duration' if incl_active else 'recovered.avg_duration'"),  # 计算函数：根据是否包含活跃回撤选择
                fill_wrap_kwargs=True,                       # 填充包装器参数
                tags=RepEval("['drawdowns', 'duration'] if incl_active else ['drawdowns', 'recovered', 'duration']")  # 标签：根据设置选择
            ),
            
            # ========== 恢复收益率指标 ==========
            
            # 最大恢复收益率指标
            max_return=dict(
                title='Max Recovery Return [%]',             # 指标标题：最大恢复收益率百分比
                calc_func='recovered.recovery_return.max',   # 计算函数：已恢复回撤的最大恢复收益率
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理：转换为百分比
                tags=['drawdowns', 'recovered']              # 标签：回撤和已恢复相关
            ),
            
            # 平均恢复收益率指标
            avg_return=dict(
                title='Avg Recovery Return [%]',             # 指标标题：平均恢复收益率百分比
                calc_func='recovered.recovery_return.mean',  # 计算函数：已恢复回撤的平均恢复收益率
                post_calc_func=lambda self, out, settings: out * 100,  # 后处理：转换为百分比
                tags=['drawdowns', 'recovered']              # 标签：回撤和已恢复相关
            ),
            
            # ========== 恢复持续时间指标 ==========
            
            # 最大恢复持续时间指标
            max_recovery_duration=dict(
                title='Max Recovery Duration',               # 指标标题：最大恢复持续时间
                calc_func='recovered.recovery_duration.max', # 计算函数：已恢复回撤的最大恢复持续时间
                apply_to_timedelta=True,                     # 应用时间差转换
                tags=['drawdowns', 'recovered', 'duration']  # 标签：回撤、已恢复和持续时间相关
            ),
            
            # 平均恢复持续时间指标
            avg_recovery_duration=dict(
                title='Avg Recovery Duration',               # 指标标题：平均恢复持续时间
                calc_func='recovered.recovery_duration.mean', # 计算函数：已恢复回撤的平均恢复持续时间
                apply_to_timedelta=True,                     # 应用时间差转换
                tags=['drawdowns', 'recovered', 'duration']  # 标签：回撤、已恢复和持续时间相关
            ),
            
            # ========== 恢复持续时间比率指标 ==========
            
            # 恢复持续时间比率指标
            recovery_duration_ratio=dict(
                title='Avg Recovery Duration Ratio',         # 指标标题：平均恢复持续时间比率
                calc_func='recovered.recovery_duration_ratio.mean',  # 计算函数：已恢复回撤的平均恢复持续时间比率
                tags=['drawdowns', 'recovered']              # 标签：回撤和已恢复相关
            )
        ),
        copy_kwargs=dict(copy_mode='deep')
    )

    @property
    def metrics(self) -> Config:
        """
        统计指标配置属性 - 返回Drawdowns类的所有可用统计指标配置
        
        这个属性返回Drawdowns类定义的所有统计指标配置，包括基础时间信息、
        覆盖率指标、记录数量、活跃回撤指标、回撤幅度指标、回撤持续时间指标、
        恢复收益率指标和恢复持续时间指标。
        
        返回：
            Config: 包含所有统计指标配置的Config对象
        
        指标分类：
        - 基础时间信息：start, end, period
        - 覆盖率指标：coverage
        - 记录数量指标：total_records, total_recovered, total_active
        - 活跃回撤指标：active_dd, active_duration, active_recovery, active_recovery_return, active_recovery_duration
        - 回撤幅度指标：max_dd, avg_dd
        - 回撤持续时间指标：max_dd_duration, avg_dd_duration
        - 恢复收益率指标：max_return, avg_return
        - 恢复持续时间指标：max_recovery_duration, avg_recovery_duration
        - 恢复比率指标：recovery_duration_ratio
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建价格数据
        price = pd.Series([100, 105, 98, 95, 102, 108, 103, 100])
        drawdowns = vbt.Drawdowns.from_ts(price)
        
        # 查看可用的统计指标
        metrics = drawdowns.metrics
        print("可用的统计指标:")
        for name, config in metrics.items():
            print(f"  {name}: {config.get('title', name)}")
        
        # 计算特定指标
        selected_stats = drawdowns.stats(metrics=['max_dd', 'avg_dd', 'max_recovery_duration'])
        print("选定指标统计:")
        print(selected_stats)
        
        # 计算所有指标
        all_stats = drawdowns.stats()
        print("所有指标统计:")
        print(all_stats)
        
        # 按标签过滤指标
        duration_stats = drawdowns.stats(tags=['duration'])
        print("持续时间相关指标:")
        print(duration_stats)
        
        # 按分类查看指标
        drawdown_metrics = [name for name, config in metrics.items() 
                          if 'drawdowns' in config.get('tags', [])]
        print("回撤相关指标:", drawdown_metrics)
        
        recovery_metrics = [name for name, config in metrics.items() 
                          if 'recovered' in config.get('tags', [])]
        print("恢复相关指标:", recovery_metrics)
        
        active_metrics = [name for name, config in metrics.items() 
                        if 'active' in config.get('tags', [])]
        print("活跃相关指标:", active_metrics)
        ```
        """
        return self._metrics

    # ############# Plotting ############# #

    def plot(self,
             column: tp.Optional[tp.Label] = None,
             top_n: int = 5,
             plot_zones: bool = True,
             ts_trace_kwargs: tp.KwargsLike = None,
             peak_trace_kwargs: tp.KwargsLike = None,
             valley_trace_kwargs: tp.KwargsLike = None,
             recovery_trace_kwargs: tp.KwargsLike = None,
             active_trace_kwargs: tp.KwargsLike = None,
             decline_shape_kwargs: tp.KwargsLike = None,
             recovery_shape_kwargs: tp.KwargsLike = None,
             active_shape_kwargs: tp.KwargsLike = None,
             add_trace_kwargs: tp.KwargsLike = None,
             xref: str = 'x',
             yref: str = 'y',
             fig: tp.Optional[tp.BaseFigure] = None,
             **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制回撤图表 - 可视化回撤记录的专业绘图方法
        
        这个方法创建一个交互式的Plotly图表，用于可视化回撤记录。
        图表包含原始时间序列（如果存在）、回撤的峰值、谷底和恢复点标记，
        以及可选的回撤区域着色。
        
        参数：
            column (tp.Optional[tp.Label], 可选): 要绘制的列名
                如果为None，则自动选择单列进行绘制
            top_n (int, 可选): 按最大回撤幅度筛选的前N个回撤记录，默认5
                可以设置为None以显示所有回撤
            plot_zones (bool, 可选): 是否绘制回撤区域，默认True
                区域用不同颜色表示下跌、恢复和活跃阶段
            ts_trace_kwargs (dict, 可选): 时间序列线条的Plotly Scatter参数
                用于自定义原始时间序列的外观
            peak_trace_kwargs (dict, 可选): 峰值标记的Plotly Scatter参数
                用于自定义回撤峰值点的外观
            valley_trace_kwargs (dict, 可选): 谷底标记的Plotly Scatter参数
                用于自定义回撤谷底点的外观
            recovery_trace_kwargs (dict, 可选): 恢复点标记的Plotly Scatter参数
                用于自定义回撤恢复点的外观
            active_trace_kwargs (dict, 可选): 活跃回撤标记的Plotly Scatter参数
                用于自定义活跃回撤当前点的外观
            decline_shape_kwargs (dict, 可选): 下跌区域的Plotly add_shape参数
                用于自定义下跌阶段区域的外观
            recovery_shape_kwargs (dict, 可选): 恢复区域的Plotly add_shape参数
                用于自定义恢复阶段区域的外观
            active_shape_kwargs (dict, 可选): 活跃回撤区域的Plotly add_shape参数
                用于自定义活跃回撤区域的外观
            add_trace_kwargs (dict, 可选): 添加轨迹的通用参数
            xref (str, 可选): X轴坐标引用，默认'x'
            yref (str, 可选): Y轴坐标引用，默认'y'
            fig (tp.Optional[tp.BaseFigure], 可选): 现有的Plotly图表对象
                如果提供，将在现有图表上添加内容
            **layout_kwargs: 图表布局的额外参数
        
        返回：
            tp.BaseFigure: 完整的Plotly图表对象
        
        图表元素说明：
        - 蓝色线条：原始时间序列数据（如果存在）
        - 蓝色钻石：回撤峰值点（回撤开始）
        - 红色钻石：谷底点（最大回撤）
        - 绿色钻石：恢复点（回撤结束）
        - 橙色钻石：活跃回撤的当前点
        - 红色半透明区域：下跌阶段
        - 绿色半透明区域：恢复阶段
        - 橙色半透明区域：活跃回撤区域
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # 示例1：基本回撤绘图
        price = pd.Series([100, 105, 98, 95, 102, 108, 103, 100])
        price.index = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(len(price))]
        drawdowns = vbt.Drawdowns.from_ts(price, wrapper_kwargs=dict(freq='1 day'))
        
        # 绘制基本回撤图
        fig = drawdowns.plot()
        fig.show()
        
        # 示例2：自定义绘图样式
        fig = drawdowns.plot(
            plot_zones=True,
            ts_trace_kwargs=dict(line=dict(color='darkblue', width=2)),
            peak_trace_kwargs=dict(marker=dict(size=10, color='blue')),
            valley_trace_kwargs=dict(marker=dict(size=10, color='red')),
            recovery_trace_kwargs=dict(marker=dict(size=10, color='green')),
            decline_shape_kwargs=dict(fillcolor='red', opacity=0.3),
            recovery_shape_kwargs=dict(fillcolor='green', opacity=0.3),
            title="自定义回撤分析图"
        )
        fig.show()
        
        # 示例3：显示特定数量的回撤
        fig = drawdowns.plot(
            top_n=3,  # 只显示最大的3个回撤
            title="Top 3 最大回撤"
        )
        fig.show()
        
        # 示例4：多列数据的回撤绘图
        portfolio = pd.DataFrame({
            'Stock_A': [100, 105, 98, 95, 102, 108, 103],
            'Stock_B': [100, 110, 85, 80, 95, 105, 100],
            'Stock_C': [100, 102, 101, 99, 100, 103, 102]
        })
        portfolio.index = pd.date_range('2023-01-01', periods=7, freq='D')
        
        portfolio_dd = vbt.Drawdowns.from_ts(portfolio, wrapper_kwargs=dict(freq='D'))
        
        # 绘制特定股票的回撤
        fig_a = portfolio_dd.plot(column='Stock_A', title="Stock A 回撤分析")
        fig_a.show()
        
        # 示例5：金融应用 - 实际股票回撤分析
        # 模拟股票价格数据
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # 一年的日收益率
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        price = pd.Series(100 * np.cumprod(1 + returns), index=dates, name='Stock Price')
        
        # 创建回撤分析
        stock_dd = vbt.Drawdowns.from_ts(price, wrapper_kwargs=dict(freq='D'))
        
        # 绘制专业的回撤分析图
        fig = stock_dd.plot(
            top_n=5,  # 显示前5个最大回撤
            title="股票回撤分析 - 前5个最大回撤",
            ts_trace_kwargs=dict(
                name="股票价格",
                line=dict(color='navy', width=1.5)
            ),
            peak_trace_kwargs=dict(
                name="回撤峰值",
                marker=dict(symbol='triangle-up', size=8, color='blue')
            ),
            valley_trace_kwargs=dict(
                name="回撤谷底",
                marker=dict(symbol='triangle-down', size=8, color='red')
            ),
            recovery_trace_kwargs=dict(
                name="回撤恢复",
                marker=dict(symbol='circle', size=8, color='green')
            ),
            decline_shape_kwargs=dict(
                fillcolor='red',
                opacity=0.2,
                line=dict(width=0)
            ),
            recovery_shape_kwargs=dict(
                fillcolor='green',
                opacity=0.2,
                line=dict(width=0)
            )
        )
        
        # 添加图表标题和轴标签
        fig.update_layout(
            title="股票回撤分析",
            xaxis_title="日期",
            yaxis_title="价格",
            showlegend=True,
            legend=dict(x=0, y=1),
            hovermode='x unified'
        )
        
        fig.show()
        
        # 示例6：不显示区域，只显示关键点
        fig = drawdowns.plot(
            plot_zones=False,  # 不显示区域
            ts_trace_kwargs=dict(line=dict(color='black', width=1)),
            title="回撤关键点分析"
        )
        fig.show()
        
        # 示例7：风险管理应用
        # 计算回撤统计
        max_dd = stock_dd.max_drawdown()
        avg_dd = stock_dd.avg_drawdown()
        
        # 在图表中添加统计信息
        fig = stock_dd.plot(
            title=f"风险分析 - 最大回撤: {max_dd:.2%}, 平均回撤: {avg_dd:.2%}",
            top_n=10
        )
        fig.show()
        ```
        """
        # 从设置中获取绘图配置
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        # 选择单列数据进行绘图
        self_col = self.select_one(column=column, group_by=False)
        
        # 如果指定了top_n，则按回撤幅度筛选前N个回撤
        if top_n is not None:
            # 注意：回撤是负值，因此top_n实际上是bottom_n
            self_col = self_col.apply_mask(self_col.drawdown.bottom_n_mask(top_n))

        # 设置时间序列线条的默认参数
        if ts_trace_kwargs is None:
            ts_trace_kwargs = {}
        ts_trace_kwargs = merge_dicts(dict(
            line=dict(
                color=plotting_cfg['color_schema']['blue']  # 默认蓝色
            )
        ), ts_trace_kwargs)
        
        # 初始化各种轨迹参数的默认值
        if peak_trace_kwargs is None:
            peak_trace_kwargs = {}
        if valley_trace_kwargs is None:
            valley_trace_kwargs = {}
        if recovery_trace_kwargs is None:
            recovery_trace_kwargs = {}
        if active_trace_kwargs is None:
            active_trace_kwargs = {}
        if decline_shape_kwargs is None:
            decline_shape_kwargs = {}
        if recovery_shape_kwargs is None:
            recovery_shape_kwargs = {}
        if active_shape_kwargs is None:
            active_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        # 创建图表对象（如果没有提供）
        if fig is None:
            fig = make_figure()
        
        # 更新图表布局
        fig.update_layout(**layout_kwargs)
        
        # 获取Y轴的显示域，用于绘制区域形状
        y_domain = get_domain(yref, fig)

        # 如果存在原始时间序列，则先绘制时间序列线条
        if self_col.ts is not None:
            fig = self_col.ts.vbt.plot(trace_kwargs=ts_trace_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)

        # 如果存在回撤记录，则绘制回撤相关的图表元素
        if self_col.count() > 0:
            # ========== 提取回撤记录信息 ==========
            
            # 提取回撤ID和标题
            id_ = self_col.get_field_arr('id')
            id_title = self_col.get_field_title('id')

            # 提取峰值索引和标题
            peak_idx = self_col.get_map_field_to_index('peak_idx')
            peak_idx_title = self_col.get_field_title('peak_idx')

            # 获取峰值价格：优先使用时间序列数据，否则使用记录中的值
            if self_col.ts is not None:
                peak_val = self_col.ts.loc[peak_idx]
            else:
                peak_val = self_col.get_field_arr('peak_val')
            peak_val_title = self_col.get_field_title('peak_val')

            # 提取谷底索引和标题
            valley_idx = self_col.get_map_field_to_index('valley_idx')
            valley_idx_title = self_col.get_field_title('valley_idx')

            # 获取谷底价格：优先使用时间序列数据，否则使用记录中的值
            if self_col.ts is not None:
                valley_val = self_col.ts.loc[valley_idx]
            else:
                valley_val = self_col.get_field_arr('valley_val')
            valley_val_title = self_col.get_field_title('valley_val')

            # 提取结束索引和标题
            end_idx = self_col.get_map_field_to_index('end_idx')
            end_idx_title = self_col.get_field_title('end_idx')

            # 获取结束价格：优先使用时间序列数据，否则使用记录中的值
            if self_col.ts is not None:
                end_val = self_col.ts.loc[end_idx]
            else:
                end_val = self_col.get_field_arr('end_val')
            end_val_title = self_col.get_field_title('end_val')

            # 获取回撤相关的数值信息
            drawdown = self_col.drawdown.values                    # 回撤幅度
            recovery_return = self_col.recovery_return.values      # 恢复收益率
            
            # 将持续时间转换为字符串格式，用于悬停提示
            decline_duration = np.vectorize(str)(self_col.wrapper.to_timedelta(
                self_col.decline_duration.values, to_pd=True, silence_warnings=True))
            recovery_duration = np.vectorize(str)(self_col.wrapper.to_timedelta(
                self_col.recovery_duration.values, to_pd=True, silence_warnings=True))
            duration = np.vectorize(str)(self_col.wrapper.to_timedelta(
                self_col.duration.values, to_pd=True, silence_warnings=True))

            # 获取回撤状态信息
            status = self_col.get_field_arr('status')

            # ========== 绘制峰值标记 ==========
            
            # 创建峰值掩码：如果峰值和恢复点在同一时间，恢复点优先显示
            peak_mask = peak_idx != np.roll(end_idx, 1)  
            
            if peak_mask.any():
                # 准备峰值标记的自定义数据
                peak_customdata = id_[peak_mask][:, None]
                
                # 创建峰值标记的散点图
                peak_scatter = go.Scatter(
                    x=peak_idx[peak_mask],                          # X坐标：峰值时间索引
                    y=peak_val[peak_mask],                          # Y坐标：峰值价格
                    mode='markers',                                 # 模式：仅显示标记
                    marker=dict(
                        symbol='diamond',                           # 钻石形状
                        color=plotting_cfg['contrast_color_schema']['blue'],  # 蓝色
                        size=7,                                     # 标记大小
                        line=dict(
                            width=1,                                # 边框宽度
                            color=adjust_lightness(plotting_cfg['contrast_color_schema']['blue'])  # 边框颜色
                        )
                    ),
                    name='Peak',                                    # 图例名称：峰值
                    customdata=peak_customdata,                     # 自定义数据用于悬停提示
                    hovertemplate=f"{id_title}: %{{customdata[0]}}"  # 悬停提示模板
                                  f"<br>{peak_idx_title}: %{{x}}"
                                  f"<br>{peak_val_title}: %{{y}}"
                )
                
                # 应用用户自定义的峰值标记参数
                peak_scatter.update(**peak_trace_kwargs)
                
                # 将峰值标记添加到图表
                fig.add_trace(peak_scatter, **add_trace_kwargs)

            # ========== 绘制已恢复回撤的谷底标记 ==========
            
            # 筛选已恢复的回撤记录
            recovered_mask = status == DrawdownStatus.Recovered
            
            if recovered_mask.any():
                # 准备谷底标记的自定义数据（包含回撤ID、回撤幅度和下跌持续时间）
                valley_customdata = np.stack((
                    id_[recovered_mask],                           # 回撤ID
                    drawdown[recovered_mask],                      # 回撤幅度
                    decline_duration[recovered_mask]               # 下跌持续时间
                ), axis=1)
                
                # 创建谷底标记的散点图
                valley_scatter = go.Scatter(
                    x=valley_idx[recovered_mask],                  # X坐标：谷底时间索引
                    y=valley_val[recovered_mask],                  # Y坐标：谷底价格
                    mode='markers',                                # 模式：仅显示标记
                    marker=dict(
                        symbol='diamond',                          # 钻石形状
                        color=plotting_cfg['contrast_color_schema']['red'],  # 红色
                        size=7,                                    # 标记大小
                        line=dict(
                            width=1,                               # 边框宽度
                            color=adjust_lightness(plotting_cfg['contrast_color_schema']['red'])  # 边框颜色
                        )
                    ),
                    name='Valley',                                 # 图例名称：谷底
                    customdata=valley_customdata,                  # 自定义数据用于悬停提示
                    hovertemplate=f"{id_title}: %{{customdata[0]}}"  # 悬停提示模板
                                  f"<br>{valley_idx_title}: %{{x}}"
                                  f"<br>{valley_val_title}: %{{y}}"
                                  f"<br>Drawdown: %{{customdata[1]:.2%}}"    # 显示回撤幅度
                                  f"<br>Duration: %{{customdata[2]}}"        # 显示持续时间
                )
                
                # 应用用户自定义的谷底标记参数
                valley_scatter.update(**valley_trace_kwargs)
                
                # 将谷底标记添加到图表
                fig.add_trace(valley_scatter, **add_trace_kwargs)

                # ========== 绘制下跌区域 ==========

                if plot_zones:
                    # 为每个已恢复的回撤绘制下跌阶段的区域
                    for i in range(len(id_[recovered_mask])):
                        fig.add_shape(**merge_dicts(dict(
                            type="rect",                              # 矩形区域
                            xref=xref,                                # X轴引用
                            yref="paper",                             # Y轴引用（纸张坐标）
                            x0=peak_idx[recovered_mask][i],           # 区域起始X坐标（峰值时间）
                            y0=y_domain[0],                           # 区域起始Y坐标（图表底部）
                            x1=valley_idx[recovered_mask][i],         # 区域结束X坐标（谷底时间）
                            y1=y_domain[1],                           # 区域结束Y坐标（图表顶部）
                            fillcolor='red',                          # 填充颜色：红色（表示下跌）
                            opacity=0.2,                              # 透明度
                            layer="below",                            # 图层：在下方
                            line_width=0,                             # 无边框
                        ), decline_shape_kwargs))

                # Plot recovery markers
                recovery_customdata = np.stack((
                    id_[recovered_mask],
                    recovery_return[recovered_mask],
                    recovery_duration[recovered_mask]
                ), axis=1)
                recovery_scatter = go.Scatter(
                    x=end_idx[recovered_mask],
                    y=end_val[recovered_mask],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        color=plotting_cfg['contrast_color_schema']['green'],
                        size=7,
                        line=dict(
                            width=1,
                            color=adjust_lightness(plotting_cfg['contrast_color_schema']['green'])
                        )
                    ),
                    name='Recovery/Peak',
                    customdata=recovery_customdata,
                    hovertemplate=f"{id_title}: %{{customdata[0]}}"
                                  f"<br>{end_idx_title}: %{{x}}"
                                  f"<br>{end_val_title}: %{{y}}"
                                  f"<br>Return: %{{customdata[1]:.2%}}"
                                  f"<br>Duration: %{{customdata[2]}}"
                )
                recovery_scatter.update(**recovery_trace_kwargs)
                fig.add_trace(recovery_scatter, **add_trace_kwargs)

                if plot_zones:
                    # Plot recovery zones
                    for i in range(len(id_[recovered_mask])):
                        fig.add_shape(**merge_dicts(dict(
                            type="rect",
                            xref=xref,
                            yref="paper",
                            x0=valley_idx[recovered_mask][i],
                            y0=y_domain[0],
                            x1=end_idx[recovered_mask][i],
                            y1=y_domain[1],
                            fillcolor='green',
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        ), recovery_shape_kwargs))

            # Plot active markers
            active_mask = status == DrawdownStatus.Active
            if active_mask.any():
                active_customdata = np.stack((
                    id_[active_mask],
                    drawdown[active_mask],
                    duration[active_mask]
                ), axis=1)
                active_scatter = go.Scatter(
                    x=end_idx[active_mask],
                    y=end_val[active_mask],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        color=plotting_cfg['contrast_color_schema']['orange'],
                        size=7,
                        line=dict(
                            width=1,
                            color=adjust_lightness(plotting_cfg['contrast_color_schema']['orange'])
                        )
                    ),
                    name='Active',
                    customdata=active_customdata,
                    hovertemplate=f"{id_title}: %{{customdata[0]}}"
                                  f"<br>{end_idx_title}: %{{x}}"
                                  f"<br>{end_val_title}: %{{y}}"
                                  f"<br>Return: %{{customdata[1]:.2%}}"
                                  f"<br>Duration: %{{customdata[2]}}"
                )
                active_scatter.update(**active_trace_kwargs)
                fig.add_trace(active_scatter, **add_trace_kwargs)

                if plot_zones:
                    # Plot active drawdown zones
                    for i in range(len(id_[active_mask])):
                        fig.add_shape(**merge_dicts(dict(
                            type="rect",
                            xref=xref,
                            yref="paper",
                            x0=peak_idx[active_mask][i],
                            y0=y_domain[0],
                            x1=end_idx[active_mask][i],
                            y1=y_domain[1],
                            fillcolor='orange',
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        ), active_shape_kwargs))

        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """
        绘图默认配置属性 - 定义Drawdowns.plots方法的默认参数
        
        这个属性定义了绘图方法的默认配置，合并了Ranges基类的配置
        和drawdowns模块特有的绘图配置。
        
        返回：
            tp.Kwargs: 绘图方法的默认参数字典
        
        配置来源：
        - Ranges基类的plots_defaults配置
        - settings中的drawdowns.plots配置
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建回撤对象
        drawdowns = vbt.Drawdowns.from_ts(some_price_series)
        
        # 查看默认绘图配置
        defaults = drawdowns.plots_defaults
        print("绘图默认配置:")
        for key, value in defaults.items():
            print(f"  {key}: {value}")
        
        # 使用默认配置绘图
        fig = drawdowns.plots()
        fig.show()
        
        # 自定义配置覆盖默认配置
        fig = drawdowns.plots(
            plot_zones=True,
            top_n=10,
            ts_trace_kwargs=dict(line=dict(color='red'))
        )
        fig.show()
        ```
        """
        # 从设置中获取drawdowns模块的绘图配置
        from vectorbt._settings import settings
        drawdowns_plots_cfg = settings['drawdowns']['plots']

        # 合并基类配置和drawdowns特有配置
        return merge_dicts(
            Ranges.plots_defaults.__get__(self),  # 获取Ranges基类的默认配置
            drawdowns_plots_cfg                   # 合并drawdowns特有配置
        )

    # 定义回撤图表的子图配置
    # 这个配置定义了可用的子图类型及其属性
    _subplots: tp.ClassVar[Config] = Config(
        dict(
            # 主要回撤绘图子图配置
            plot=dict(
                title="Drawdowns",                      # 子图标题：回撤
                check_is_not_grouped=True,              # 检查是否未分组（回撤图不支持分组）
                plot_func='plot',                       # 绘图函数名称：plot方法
                tags='drawdowns'                        # 标签：回撤相关
            )
        ),
        copy_kwargs=dict(copy_mode='deep')              # 深拷贝配置
    )

    @property
    def subplots(self) -> Config:
        """
        子图配置属性 - 返回Drawdowns类的子图配置
        
        这个属性返回Drawdowns类定义的子图配置，用于plots()方法。
        目前只有一个主要的回撤绘图子图。
        
        返回：
            Config: 包含子图配置的Config对象
        
        子图类型：
        - plot: 主要的回撤绘图子图
          - title: "Drawdowns"
          - check_is_not_grouped: True（不支持分组）
          - plot_func: 'plot'（使用plot方法）
          - tags: 'drawdowns'
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建回撤对象
        drawdowns = vbt.Drawdowns.from_ts(some_price_series)
        
        # 查看可用的子图
        subplots = drawdowns.subplots
        print("可用的子图:")
        for name, config in subplots.items():
            print(f"  {name}: {config.get('title', name)}")
        
        # 使用plots方法创建子图
        fig = drawdowns.plots()
        fig.show()
        
        # 自定义子图标题
        fig = drawdowns.plots(
            plot_kwargs=dict(title="自定义回撤分析")
        )
        fig.show()
        ```
        """
        return self._subplots


# ========== 文档覆盖方法 ========== #

# 覆盖字段配置文档
# 这个方法会自动更新__pdoc__字典中的字段配置文档
Drawdowns.override_field_config_doc(__pdoc__)

# 覆盖统计指标文档
# 这个方法会自动更新__pdoc__字典中的统计指标文档
Drawdowns.override_metrics_doc(__pdoc__)

# 覆盖子图配置文档
# 这个方法会自动更新__pdoc__字典中的子图配置文档
Drawdowns.override_subplots_doc(__pdoc__)
