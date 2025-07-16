# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT范围记录模块 - 专门用于处理时间序列中的范围和持续时间分析
================================================================================

文件设计逻辑和作用概述：
本模块是vectorbt量化交易框架中用于处理范围记录（Range Records）的核心模块。
范围记录用于捕获时间序列中某种状态的起始和结束时间信息，在量化金融分析中具有重要作用。

核心设计理念：
1. **范围定义**：每个范围有一个起始点和结束点，例如range(20)的起始点是0，结束点是20
2. **状态跟踪**：跟踪各种金融状态的持续时间，如回撤期间、持仓期间、信号活跃期间等
3. **高效分析**：基于Records类的高效结构化数组存储，提供快速的范围分析能力
4. **统计集成**：内置丰富的统计指标和可视化功能，支持复杂的金融分析

使用示例：
```python
import vectorbt as vbt
import numpy as np
import pandas as pd

# 创建示例数据
price = pd.Series([100, 102, 101, 99, 98, 100, 102, 104])
ma = price.rolling(3).mean()

# 创建布尔条件（价格高于均线）
above_ma = price > ma

# 从时间序列创建范围记录
ranges = vbt.Ranges.from_ts(above_ma, wrapper_kwargs=dict(freq='d'))

# 分析范围统计
print("范围统计:")
print(ranges.stats())

# 分析持续时间
print("平均持续时间:", ranges.avg_duration())
print("最大持续时间:", ranges.max_duration())

# 分析覆盖率
print("覆盖率:", ranges.coverage())

# 可视化
ranges.plot().show()
```
"""

import numpy as np
import plotly.graph_objects as go

from vectorbt import _typing as tp
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.base.reshape_fns import to_pd_array, to_2d_array
from vectorbt.generic import nb
from vectorbt.generic.enums import RangeStatus, range_dt
from vectorbt.records.base import Records
from vectorbt.records.decorators import override_field_config, attach_fields
from vectorbt.records.mapped_array import MappedArray
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.config import merge_dicts, Config
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.figure import make_figure, get_domain

# 初始化文档字典 - 用于存储API文档信息
__pdoc__ = {}

# 范围记录的字段配置 - 定义Ranges类的字段结构和属性
# 这个配置定义了范围记录的数据结构，包括字段类型、设置和映射关系
ranges_field_config = Config(
    dict(
        # 数据类型定义：指定使用range_dt作为记录的数据类型
        # range_dt包含了范围记录所需的所有字段定义
        dtype=range_dt,
        
        # 字段设置：定义每个字段的显示和映射属性
        settings=dict(
            # 范围ID字段配置
            id=dict(
                title='Range Id'  # 字段显示标题：范围ID
            ),
            
            # 索引字段配置 - 将idx字段重映射为end_idx
            # 这是Records基类的字段重映射功能
            idx=dict(
                name='end_idx'  # 将Records基类的idx字段重命名为end_idx
            ),
            
            # 起始索引字段配置
            start_idx=dict(
                title='Start Timestamp',  # 字段显示标题：起始时间戳
                mapping='index'          # 映射到ArrayWrapper的index属性
            ),
            
            # 结束索引字段配置
            end_idx=dict(
                title='End Timestamp',   # 字段显示标题：结束时间戳
                mapping='index'         # 映射到ArrayWrapper的index属性
            ),
            
            # 状态字段配置
            status=dict(
                title='Status',         # 字段显示标题：状态
                mapping=RangeStatus     # 映射到RangeStatus枚举类
            )
        )
    ),
    readonly=True,    # 配置为只读，防止运行时修改
    as_attrs=False   # 不将配置项作为属性访问
)
"""_"""

# 为ranges_field_config生成文档字符串
__pdoc__['ranges_field_config'] = f"""范围记录的字段配置。

这个配置定义了Ranges类的字段结构，包括：
- dtype: 指定使用range_dt数据类型
- settings: 定义各字段的显示标题和映射关系
  - id: 范围的唯一标识符
  - start_idx: 范围的起始索引，映射到时间戳
  - end_idx: 范围的结束索引，映射到时间戳  
  - status: 范围的状态（开放/封闭），映射到RangeStatus枚举

配置内容：
```json
{ranges_field_config.to_doc()}
```
"""

# 范围记录的附加字段配置 - 定义需要附加到Ranges类的字段
# 这个配置指定了哪些字段需要自动生成过滤器和其他辅助方法
ranges_attach_field_config = Config(
    dict(
        # 状态字段的附加配置
        status=dict(
            attach_filters=True  # 为status字段自动生成过滤器方法
                               # 这将生成诸如filter_by_status等方法
        )
    ),
    readonly=True,    # 配置为只读
    as_attrs=False   # 不将配置项作为属性访问
)
"""_"""

# 为ranges_attach_field_config生成文档字符串
__pdoc__['ranges_attach_field_config'] = f"""需要附加到Ranges类的字段配置。

这个配置指定了哪些字段需要自动生成过滤器和其他辅助功能：
- status字段启用过滤器：将自动生成按状态过滤的方法

配置内容：
```json
{ranges_attach_field_config.to_doc()}
```
"""

# 定义Ranges类的类型变量 - 用于类型提示中的泛型约束
# 这确保了Ranges类的方法返回的类型与调用类的类型一致
RangesT = tp.TypeVar("RangesT", bound="Ranges")


# 使用装饰器为Ranges类附加字段功能
@attach_fields(ranges_attach_field_config)
# 使用装饰器重写字段配置
@override_field_config(ranges_field_config)
class Ranges(Records):
    """
    Ranges类 - 专门用于处理范围记录的Records子类
    
    这个类扩展了Records类，专门用于处理范围记录。范围记录捕获时间序列中
    某种状态的起始和结束时间信息，对于分析过程的持续时间非常有用，
    如回撤、交易和持仓的分析。
    
    核心功能：
    1. **范围识别**：从时间序列中自动识别范围（连续的True值、非NaN值等）
    2. **状态跟踪**：跟踪范围的状态（开放/封闭）
    3. **持续时间分析**：计算范围的持续时间和相关统计
    4. **覆盖率分析**：分析范围覆盖的时间比例
    5. **可视化**：提供专门的范围可视化功能
    
    数据结构：
    - start_idx: 范围的起始索引
    - end_idx: 范围的结束索引  
    - status: 范围的状态（RangeStatus.Open或RangeStatus.Closed）
    - id: 范围的唯一标识符
    - col: 范围所属的列索引
    
    使用示例：
    ```python
    import vectorbt as vbt
    import pandas as pd
    import numpy as np
    
    # 示例1：从布尔时间序列创建范围
    ts = pd.Series([True, True, False, True, True, True, False])
    ranges = vbt.Ranges.from_ts(ts)
    print("范围记录:")
    print(ranges.records_readable)
    
    # 示例2：分析价格回撤
    price = pd.Series([100, 102, 98, 95, 99, 101, 103])
    peak = price.expanding().max()
    drawdown = (price - peak) / peak < -0.02  # 回撤超过2%
    dd_ranges = vbt.Ranges.from_ts(drawdown)
    print("回撤期间:")
    print(dd_ranges.records_readable)
    
    # 示例3：统计分析
    print("平均持续时间:", ranges.avg_duration())
    print("覆盖率:", ranges.coverage())
    print("最长持续时间:", ranges.max_duration())
    
    # 示例4：可视化
    ranges.plot().show()
    ```
    
    继承关系：
    - 继承自Records类，获得所有结构化记录的功能
    - 重写了字段配置，定义了范围记录特有的字段
    - 添加了范围特有的方法和属性
    
    要求：
    - records_arr必须包含range_dt中定义的所有字段
    - 支持开放和封闭两种范围状态
    - 提供从时间序列自动创建范围的能力
    """

    @property
    def field_config(self) -> Config:
        """
        字段配置属性 - 返回Ranges类的字段配置
        
        这个属性返回专门为Ranges类定义的字段配置，包括range_dt数据类型
        和各个字段的设置信息。
        
        返回：
            Config: 包含dtype和settings的配置对象
            
        字段说明：
        - dtype: range_dt数据类型
        - settings: 字段设置字典
          - id: 范围标识符
          - start_idx: 起始索引（映射到时间戳）
          - end_idx: 结束索引（映射到时间戳）
          - status: 范围状态（映射到RangeStatus枚举）
        """
        return self._field_config

    def __init__(self,
                 wrapper: ArrayWrapper,
                 records_arr: tp.RecordArray,
                 ts: tp.Optional[tp.ArrayLike] = None,
                 **kwargs) -> None:
        """
        Ranges类的初始化方法
        
        初始化一个Ranges对象，设置数组包装器、记录数组和可选的时间序列数据。
        
        参数：
            wrapper (ArrayWrapper): 数组包装器，包含索引、列名、分组等元数据
            records_arr (tp.RecordArray): 范围记录的结构化数组
                必须包含range_dt中定义的所有字段：
                - id: 范围ID
                - col: 列索引
                - start_idx: 起始索引
                - end_idx: 结束索引
                - status: 范围状态
            ts (tp.Optional[tp.ArrayLike], 可选): 原始时间序列数据
                如果提供，将用于绘图和进一步分析
            **kwargs: 传递给Records基类的额外参数
        
        使用示例：
        ```python
        import numpy as np
        import vectorbt as vbt
        from vectorbt.generic.enums import range_dt, RangeStatus
        
        # 手动创建范围记录数组
        records_arr = np.array([
            (0, 0, 0, 2, RangeStatus.Closed),  # 范围0: 列0, 从索引0到2
            (1, 0, 4, 6, RangeStatus.Open),    # 范围1: 列0, 从索引4到6（开放）
            (2, 1, 1, 3, RangeStatus.Closed)   # 范围2: 列1, 从索引1到3
        ], dtype=range_dt)
        
        # 创建包装器
        wrapper = vbt.ArrayWrapper(
            index=pd.date_range('2023-01-01', periods=10, freq='D'),
            columns=['A', 'B'],
            ndim=2
        )
        
        # 创建Ranges对象
        ranges = vbt.Ranges(wrapper, records_arr)
        
        # 查看结果
        print(ranges.records_readable)
        print("持续时间:", ranges.duration.values)
        ```
        """
        # 调用Records基类的初始化方法
        # 这会设置基础的记录功能，包括字段验证和列映射器创建
        Records.__init__(
            self,
            wrapper,                    # 数组包装器
            records_arr,               # 记录数组
            ts=ts,                     # 时间序列数据
            **kwargs                   # 其他参数
        )
        
        # 存储原始时间序列数据到私有变量
        # 这个数据在绘图和某些分析中会用到
        self._ts = ts

    def indexing_func(self: RangesT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> RangesT:
        """
        执行索引操作并返回新的Ranges实例 - 支持pandas风格索引
        
        这个方法执行pandas风格的索引操作，如切片、选择等，并返回一个新的
        Ranges实例。它会正确处理包装器、记录数组和时间序列数据的索引。
        
        参数：
            pd_indexing_func (tp.PandasIndexingFunc): pandas索引函数
            **kwargs: 传递给索引函数的额外参数
        
        返回：
            RangesT: 新的Ranges实例，包含索引后的数据
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建多列的范围数据
        ts = pd.DataFrame({
            'A': [True, True, False, True, True],
            'B': [False, True, True, False, True],
            'C': [True, False, False, True, True]
        })
        
        ranges = vbt.Ranges.from_ts(ts)
        
        # 选择特定列
        ranges_a = ranges['A']
        print("列A的范围:", ranges_a.records_readable)
        
        # 选择多列
        ranges_ab = ranges[['A', 'B']]
        print("列A和B的范围:", ranges_ab.records_readable)
        
        # 使用iloc选择
        ranges_first_two = ranges.iloc[:, :2]
        print("前两列的范围:", ranges_first_two.records_readable)
        ```
        """
        # 调用Records基类的索引元数据方法获取新的组件
        new_wrapper, new_records_arr, _, col_idxs = \
            Records.indexing_func_meta(self, pd_indexing_func, **kwargs)
        
        # 处理时间序列数据的索引
        if self.ts is not None:
            # 如果存在时间序列数据，也需要对其进行相应的索引操作
            # 使用列索引选择对应的时间序列数据
            new_ts = new_wrapper.wrap(self.ts.values[:, col_idxs], group_by=False)
        else:
            # 如果没有时间序列数据，设置为None
            new_ts = None
        
        # 创建并返回新的Ranges实例
        return self.replace(
            wrapper=new_wrapper,           # 新的包装器
            records_arr=new_records_arr,   # 新的记录数组
            ts=new_ts                      # 新的时间序列数据
        )

    @classmethod
    def from_ts(cls: tp.Type[RangesT],
                ts: tp.ArrayLike,
                gap_value: tp.Optional[tp.Scalar] = None,
                attach_ts: bool = True,
                wrapper_kwargs: tp.KwargsLike = None,
                **kwargs) -> RangesT:
        """
        从时间序列创建Ranges对象 - 自动范围识别的核心方法
        
        这个类方法从时间序列数据中自动识别范围，是创建Ranges对象的主要方式。
        它会根据数据类型自动选择合适的间隔值，并识别连续的非间隔值序列作为范围。
        
        识别规则：
        - 布尔数据：True值序列为范围，False作为间隔
        - 整数数据：正值序列为范围，-1作为间隔
        - 其他数据：非NaN值序列为范围，NaN作为间隔
        
        参数：
            ts (tp.ArrayLike): 输入的时间序列数据
                可以是Series、DataFrame或数组
            gap_value (tp.Optional[tp.Scalar], 可选): 间隔值
                如果为None，会根据数据类型自动选择
            attach_ts (bool, 可选): 是否附加原始时间序列，默认True
            wrapper_kwargs (tp.KwargsLike, 可选): 传递给ArrayWrapper的参数
            **kwargs: 传递给Ranges构造函数的额外参数
        
        返回：
            RangesT: 从时间序列创建的Ranges对象
        """
        # 设置默认的包装器参数
        if wrapper_kwargs is None:
            wrapper_kwargs = {}

        # 将时间序列转换为pandas对象
        ts_pd = to_pd_array(ts)
        # 将时间序列转换为2D数组以便处理
        ts_arr = to_2d_array(ts_pd)
        
        # 根据数据类型自动确定间隔值
        if gap_value is None:
            if np.issubdtype(ts_arr.dtype, np.bool_):
                # 布尔数据：False作为间隔
                gap_value = False
            elif np.issubdtype(ts_arr.dtype, np.integer):
                # 整数数据：-1作为间隔
                gap_value = -1
            else:
                # 其他数据类型：NaN作为间隔
                gap_value = np.nan
        
        # 使用numba编译的函数查找范围
        # 这个函数会扫描数组，找出所有连续的非间隔值序列
        records_arr = nb.find_ranges_nb(ts_arr, gap_value)
        
        # 创建数组包装器
        wrapper = ArrayWrapper.from_obj(ts_pd, **wrapper_kwargs)
        
        # 创建并返回Ranges对象
        return cls(
            wrapper,                                    # 数组包装器
            records_arr,                               # 范围记录数组
            ts=ts_pd if attach_ts else None,          # 可选的时间序列数据
            **kwargs                                   # 其他参数
        )

    @property
    def ts(self) -> tp.Optional[tp.SeriesFrame]:
        """
        时间序列属性 - 返回构建记录时使用的原始时间序列数据
        
        这个属性返回在创建Ranges对象时传入的原始时间序列数据。
        如果在初始化时没有提供时间序列数据，则返回None。
        
        返回：
            tp.Optional[tp.SeriesFrame]: 原始时间序列数据，如果不存在则为None
        
        用途：
        - 用于绘图和可视化
        - 用于验证范围记录的正确性
        - 用于进一步的时间序列分析
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建时间序列
        ts = pd.Series([True, True, False, True, True], 
                      index=pd.date_range('2023-01-01', periods=5, freq='D'))
        
        # 从时间序列创建范围，保留原始数据
        ranges = vbt.Ranges.from_ts(ts, attach_ts=True)
        
        # 访问原始时间序列
        original_ts = ranges.ts
        print("原始时间序列:")
        print(original_ts)
        
        # 创建范围时不保留原始数据
        ranges_no_ts = vbt.Ranges.from_ts(ts, attach_ts=False)
        print("是否保留原始数据:", ranges_no_ts.ts is not None)
        ```
        """
        return self._ts

    def to_mask(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """
        将范围转换为掩码 - 生成布尔掩码数组
        
        这个方法将范围记录转换为布尔掩码，其中True表示在范围内，False表示不在范围内。
        这对于过滤数据或标识特定时间段非常有用。
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式，用于聚合不同列的范围
            wrap_kwargs (tp.KwargsLike, 可选): 传递给包装器的参数
        
        返回：
            tp.SeriesFrame: 布尔掩码，形状与原始数据相同
        
        注意：
        - 开放范围（Open）的end_idx指向最后一个有效索引
        - 封闭范围（Closed）的end_idx指向范围结束后的下一个索引
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        import numpy as np
        
        # 创建示例数据
        ts = pd.Series([True, True, False, True, True, True, False])
        ranges = vbt.Ranges.from_ts(ts)
        
        # 转换为掩码
        mask = ranges.to_mask()
        print("范围掩码:")
        print(mask)
        
        # 验证掩码与原始数据的一致性
        print("与原始数据一致:", np.array_equal(mask.values, ts.values))
        
        # 多列数据示例
        df = pd.DataFrame({
            'A': [True, True, False, True, False],
            'B': [False, True, True, False, True]
        })
        ranges_multi = vbt.Ranges.from_ts(df)
        mask_multi = ranges_multi.to_mask()
        print("多列掩码:")
        print(mask_multi)
        
        # 分组示例
        grouped_mask = ranges_multi.to_mask(group_by=['Group1', 'Group1'])
        print("分组掩码:")
        print(grouped_mask)
        ```
        """
        # 获取列映射，用于分组操作
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        
        # 使用numba编译的函数生成掩码
        # 这个函数会根据范围的start_idx、end_idx和status生成布尔掩码
        mask = nb.ranges_to_mask_nb(
            self.get_field_arr('start_idx'),    # 起始索引数组
            self.get_field_arr('end_idx'),      # 结束索引数组
            self.get_field_arr('status'),       # 状态数组
            col_map,                            # 列映射
            len(self.wrapper.index)             # 索引长度
        )
        
        # 使用包装器包装结果并返回
        return self.wrapper.wrap(mask, group_by=group_by, **merge_dicts({}, wrap_kwargs))

    @cached_property
    def duration(self) -> MappedArray:
        """
        持续时间属性 - 计算每个范围的持续时间
        
        这个属性计算每个范围的持续时间（以原始格式，即索引单位）。
        持续时间是范围结束索引与起始索引之间的差值。
        
        返回：
            MappedArray: 包含每个范围持续时间的映射数组
        
        计算规则：
        - 对于封闭范围：duration = end_idx - start_idx
        - 对于开放范围：duration = end_idx - start_idx + 1
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建范围数据
        ts = pd.Series([True, True, False, True, True, True, False])
        ranges = vbt.Ranges.from_ts(ts)
        
        # 获取持续时间
        durations = ranges.duration
        print("范围持续时间（索引单位）:")
        print(durations.values)
        
        # 如果有时间频率，转换为时间差
        ts_with_freq = pd.Series(
            [True, True, False, True, True, True, False],
            index=pd.date_range('2023-01-01', periods=7, freq='D')
        )
        ranges_with_freq = vbt.Ranges.from_ts(ts_with_freq)
        
        # 转换为时间差
        time_durations = ranges_with_freq.duration
        print("范围持续时间（时间差）:")
        print(time_durations.to_timedelta())
        
        # 统计分析
        print("平均持续时间:", durations.mean())
        print("最大持续时间:", durations.max())
        print("总持续时间:", durations.sum())
        ```
        """
        # 使用numba编译的函数计算持续时间
        # 这个函数会根据范围的起始索引、结束索引和状态计算持续时间
        duration = nb.range_duration_nb(
            self.get_field_arr('start_idx'),    # 起始索引数组
            self.get_field_arr('end_idx'),      # 结束索引数组
            self.get_field_arr('status')        # 状态数组
        )
        
        # 将结果转换为映射数组并返回
        return self.map_array(duration)

    @cached_method
    def avg_duration(self, group_by: tp.GroupByLike = None,
                     wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        平均持续时间 - 计算范围的平均持续时间
        
        这个方法计算所有范围的平均持续时间，结果自动转换为时间差格式。
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
            **kwargs: 传递给mean方法的其他参数
        
        返回：
            tp.MaybeSeries: 平均持续时间（时间差格式）
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建带时间频率的范围数据
        ts = pd.Series(
            [True, True, False, True, True, True, False, True],
            index=pd.date_range('2023-01-01', periods=8, freq='D')
        )
        ranges = vbt.Ranges.from_ts(ts)
        
        # 计算平均持续时间
        avg_dur = ranges.avg_duration()
        print("平均持续时间:", avg_dur)
        
        # 多列数据的平均持续时间
        df = pd.DataFrame({
            'A': [True, True, False, True, False],
            'B': [False, True, True, False, True]
        }, index=pd.date_range('2023-01-01', periods=5, freq='D'))
        
        ranges_multi = vbt.Ranges.from_ts(df)
        avg_dur_multi = ranges_multi.avg_duration()
        print("多列平均持续时间:")
        print(avg_dur_multi)
        
        # 分组计算
        avg_dur_grouped = ranges_multi.avg_duration(group_by=['Group1', 'Group1'])
        print("分组平均持续时间:", avg_dur_grouped)
        ```
        """
        # 设置包装器参数，包括转换为时间差和设置名称
        wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index='avg_duration'), wrap_kwargs)
        
        # 计算持续时间的平均值
        return self.duration.mean(group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_method
    def max_duration(self, group_by: tp.GroupByLike = None,
                     wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """
        最大持续时间 - 计算范围的最大持续时间
        
        这个方法计算所有范围中的最大持续时间，结果自动转换为时间差格式。
        
        参数：
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
            **kwargs: 传递给max方法的其他参数
        
        返回：
            tp.MaybeSeries: 最大持续时间（时间差格式）
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建范围数据
        ts = pd.Series(
            [True, True, False, True, True, True, True, False],
            index=pd.date_range('2023-01-01', periods=8, freq='H')
        )
        ranges = vbt.Ranges.from_ts(ts)
        
        # 计算最大持续时间
        max_dur = ranges.max_duration()
        print("最大持续时间:", max_dur)
        
        # 分析不同类型的范围
        print("范围详情:")
        print(ranges.records_readable)
        
        # 多列数据的最大持续时间
        df = pd.DataFrame({
            'Stock_A': [True, True, True, False, True, False],
            'Stock_B': [False, True, True, True, True, False]
        }, index=pd.date_range('2023-01-01', periods=6, freq='D'))
        
        ranges_stocks = vbt.Ranges.from_ts(df)
        max_dur_stocks = ranges_stocks.max_duration()
        print("各股票最大持续时间:")
        print(max_dur_stocks)
        
        # 投资组合层面的最大持续时间
        portfolio_max = ranges_stocks.max_duration(group_by=['Portfolio', 'Portfolio'])
        print("投资组合最大持续时间:", portfolio_max)
        ```
        """
        # 设置包装器参数，包括转换为时间差和设置名称
        wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index='max_duration'), wrap_kwargs)
        
        # 计算持续时间的最大值
        return self.duration.max(group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_method
    def coverage(self,
                 overlapping: bool = False,
                 normalize: bool = True,
                 group_by: tp.GroupByLike = None,
                 wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """
        覆盖率分析 - 计算范围覆盖的时间比例
        
        这个方法计算范围记录覆盖的时间步数，可以选择是否考虑重叠和是否标准化。
        覆盖率是量化分析中的重要指标，用于评估策略的活跃度和市场参与度。
        
        参数：
            overlapping (bool, 可选): 是否考虑重叠范围，默认False
                - False: 重叠的范围只计算一次
                - True: 重叠的范围会重复计算
            normalize (bool, 可选): 是否标准化为比例，默认True
                - True: 返回覆盖率（0-1之间）
                - False: 返回覆盖的实际时间步数
            group_by (tp.GroupByLike, 可选): 分组方式
            wrap_kwargs (tp.KwargsLike, 可选): 包装器参数
        
        返回：
            tp.MaybeSeries: 覆盖率或覆盖的时间步数
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        
        # 创建示例数据
        ts = pd.Series([True, True, False, True, True, True, False])
        ranges = vbt.Ranges.from_ts(ts)
        
        # 计算基本覆盖率
        coverage_rate = ranges.coverage()
        print("覆盖率:", coverage_rate)
        
        # 计算覆盖的实际时间步数
        coverage_steps = ranges.coverage(normalize=False)
        print("覆盖的时间步数:", coverage_steps)
        
        # 多列数据的覆盖率分析
        df = pd.DataFrame({
            'Strategy_A': [True, True, False, True, False],
            'Strategy_B': [False, True, True, False, True],
            'Strategy_C': [True, False, True, True, False]
        })
        ranges_multi = vbt.Ranges.from_ts(df)
        
        # 各策略的覆盖率
        coverage_multi = ranges_multi.coverage()
        print("各策略覆盖率:")
        print(coverage_multi)
        
        # 考虑重叠的覆盖率
        coverage_overlap = ranges_multi.coverage(overlapping=True)
        print("考虑重叠的覆盖率:")
        print(coverage_overlap)
        
        # 分组分析（如按策略类型分组）
        coverage_grouped = ranges_multi.coverage(
            group_by=['Trend', 'Mean_Reversion', 'Momentum']
        )
        print("按策略类型分组的覆盖率:")
        print(coverage_grouped)
        
        # 投资组合层面的覆盖率
        portfolio_coverage = ranges_multi.coverage(
            group_by=['Portfolio', 'Portfolio', 'Portfolio']
        )
        print("投资组合覆盖率:", portfolio_coverage)
        
        # 金融分析应用
        # 分析交易信号的市场参与度
        price = pd.Series([100, 102, 101, 103, 105, 104, 106])
        buy_signal = price.pct_change() > 0.01
        buy_ranges = vbt.Ranges.from_ts(buy_signal)
        participation_rate = buy_ranges.coverage()
        print("市场参与度:", participation_rate)
        ```
        """
        # 获取列映射，用于分组操作
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        
        # 获取每个组的索引长度
        # 这用于标准化计算，将绝对时间步数转换为比例
        index_lens = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        
        # 使用numba编译的函数计算覆盖率
        # 这个函数会分析范围的起始、结束时间和状态，计算实际覆盖的时间步数
        coverage = nb.range_coverage_nb(
            self.get_field_arr('start_idx'),    # 起始索引数组
            self.get_field_arr('end_idx'),      # 结束索引数组
            self.get_field_arr('status'),       # 状态数组
            col_map,                            # 列映射
            index_lens,                         # 索引长度
            overlapping=overlapping,            # 是否考虑重叠
            normalize=normalize                 # 是否标准化
        )
        
        # 设置包装器参数
        wrap_kwargs = merge_dicts(dict(name_or_index='coverage'), wrap_kwargs)
        
        # 包装结果并返回
        return self.wrapper.wrap_reduced(coverage, group_by=group_by, **wrap_kwargs)

    # ############# 统计分析部分 ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """
        统计分析默认配置 - 定义Ranges.stats的默认参数
        
        这个属性定义了统计分析方法的默认配置，合并了Records基类的配置
        和ranges模块特有的配置。
        
        返回：
            tp.Kwargs: 统计分析的默认参数字典
        
        配置来源：
        - Records基类的stats_defaults
        - settings中的ranges.stats配置
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建范围对象
        ranges = vbt.Ranges.from_ts(some_time_series)
        
        # 查看默认配置
        defaults = ranges.stats_defaults
        print("统计分析默认配置:")
        for key, value in defaults.items():
            print(f"  {key}: {value}")
        
        # 使用默认配置进行统计分析
        stats = ranges.stats()
        print("统计结果:")
        print(stats)
        ```
        """
        # 从设置中获取ranges模块的统计配置
        from vectorbt._settings import settings
        ranges_stats_cfg = settings['ranges']['stats']

        # 合并基类配置和ranges特有配置
        return merge_dicts(
            Records.stats_defaults.__get__(self),  # 获取Records基类的默认配置
            ranges_stats_cfg                       # 合并ranges特有配置
        )

    # 定义范围记录的统计指标配置
    # 这个配置定义了所有可用的统计指标及其计算方法
    _metrics: tp.ClassVar[Config] = Config(
        dict(
            # 起始时间指标
            start=dict(
                title='Start',                              # 指标标题
                calc_func=lambda self: self.wrapper.index[0],  # 计算函数：返回第一个索引
                agg_func=None,                              # 聚合函数：无需聚合
                tags='wrapper'                              # 标签：属于wrapper相关指标
            ),
            
            # 结束时间指标
            end=dict(
                title='End',                                # 指标标题
                calc_func=lambda self: self.wrapper.index[-1], # 计算函数：返回最后一个索引
                agg_func=None,                              # 聚合函数：无需聚合
                tags='wrapper'                              # 标签：属于wrapper相关指标
            ),
            
            # 总时间段指标
            period=dict(
                title='Period',                             # 指标标题
                calc_func=lambda self: len(self.wrapper.index), # 计算函数：返回索引长度
                apply_to_timedelta=True,                    # 应用时间差转换
                agg_func=None,                              # 聚合函数：无需聚合
                tags='wrapper'                              # 标签：属于wrapper相关指标
            ),
            
            # 覆盖率指标（不考虑重叠）
            coverage=dict(
                title='Coverage',                           # 指标标题
                calc_func='coverage',                       # 计算函数：调用coverage方法
                overlapping=False,                          # 不考虑重叠
                normalize=False,                            # 不标准化（返回实际时间步数）
                apply_to_timedelta=True,                    # 应用时间差转换
                tags=['ranges', 'coverage']                 # 标签：范围和覆盖率相关
            ),
            
            # 重叠覆盖率指标
            overlap_coverage=dict(
                title='Overlap Coverage',                   # 指标标题
                calc_func='coverage',                       # 计算函数：调用coverage方法
                overlapping=True,                           # 考虑重叠
                normalize=False,                            # 不标准化
                apply_to_timedelta=True,                    # 应用时间差转换
                tags=['ranges', 'coverage']                 # 标签：范围和覆盖率相关
            ),
            
            # 总记录数指标
            total_records=dict(
                title='Total Records',                      # 指标标题
                calc_func='count',                          # 计算函数：调用count方法
                tags='records'                              # 标签：记录相关指标
            ),
            
            # 持续时间统计指标
            duration=dict(
                title='Duration',                           # 指标标题
                calc_func='duration.describe',              # 计算函数：调用duration的describe方法
                # 后处理函数：将describe结果转换为具体的统计指标
                post_calc_func=lambda self, out, settings: {
                    'Min': out.loc['min'],                  # 最小持续时间
                    'Median': out.loc['50%'],               # 中位数持续时间
                    'Max': out.loc['max'],                  # 最大持续时间
                    'Mean': out.loc['mean'],                # 平均持续时间
                    'Std': out.loc['std']                   # 持续时间标准差
                },
                apply_to_timedelta=True,                    # 应用时间差转换
                tags=['ranges', 'duration']                 # 标签：范围和持续时间相关
            ),
        ),
        copy_kwargs=dict(copy_mode='deep')                  # 深拷贝配置
    )

    @property
    def metrics(self) -> Config:
        """
        统计指标配置 - 返回可用的统计指标配置
        
        这个属性返回Ranges类可用的所有统计指标配置，包括：
        - 时间相关指标：start, end, period
        - 覆盖率指标：coverage, overlap_coverage
        - 记录数指标：total_records
        - 持续时间指标：duration (包含min, median, max, mean, std)
        
        返回：
            Config: 统计指标配置对象
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建范围对象
        ranges = vbt.Ranges.from_ts(some_time_series)
        
        # 查看可用的统计指标
        metrics = ranges.metrics
        print("可用的统计指标:")
        for name, config in metrics.items():
            print(f"  {name}: {config.get('title', name)}")
        
        # 计算特定指标
        stats = ranges.stats(metrics=['coverage', 'duration'])
        print("选定指标统计:")
        print(stats)
        
        # 计算所有指标
        all_stats = ranges.stats()
        print("所有指标统计:")
        print(all_stats)
        ```
        """
        return self._metrics

    # ############# 绘图相关方法 ############# #

    def plot(self,
             column: tp.Optional[tp.Label] = None,
             top_n: int = 5,
             plot_zones: bool = True,
             ts_trace_kwargs: tp.KwargsLike = None,
             start_trace_kwargs: tp.KwargsLike = None,
             end_trace_kwargs: tp.KwargsLike = None,
             open_shape_kwargs: tp.KwargsLike = None,
             closed_shape_kwargs: tp.KwargsLike = None,
             add_trace_kwargs: tp.KwargsLike = None,
             xref: str = 'x',
             yref: str = 'y',
             fig: tp.Optional[tp.BaseFigure] = None,
             **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover
        """
        绘制范围图表 - 可视化范围记录的专业绘图方法
        
        这个方法创建一个交互式的Plotly图表，用于可视化范围记录。
        图表包含原始时间序列（如果存在）、范围的起始和结束标记，
        以及可选的范围区域着色。
        
        参数：
            column (tp.Optional[tp.Label], 可选): 要绘制的列名，如果为None则选择单列
            top_n (int, 可选): 按最大持续时间筛选的前N个范围，默认5
            plot_zones (bool, 可选): 是否绘制范围区域，默认True
            ts_trace_kwargs (dict, 可选): 时间序列线条的Plotly参数
            start_trace_kwargs (dict, 可选): 起始点标记的Plotly参数
            end_trace_kwargs (dict, 可选): 结束点标记的Plotly参数
            open_shape_kwargs (dict, 可选): 开放范围区域的Plotly参数
            closed_shape_kwargs (dict, 可选): 封闭范围区域的Plotly参数
            add_trace_kwargs (dict, 可选): 添加轨迹的通用参数
            xref (str, 可选): X轴坐标引用，默认'x'
            yref (str, 可选): Y轴坐标引用，默认'y'
            fig (tp.Optional[tp.BaseFigure], 可选): 现有的Plotly图表对象
            **layout_kwargs: 图表布局的额外参数
        
        返回：
            tp.BaseFigure: Plotly图表对象
        
        图表元素：
        - 时间序列线条（如果存在原始数据）
        - 蓝色钻石标记：范围起始点
        - 绿色钻石标记：封闭范围结束点
        - 橙色钻石标记：开放范围结束点
        - 青色半透明区域：封闭范围区域
        - 橙色半透明区域：开放范围区域
        
        使用示例：
        ```python
        import vectorbt as vbt
        import pandas as pd
        import numpy as np
        
        # 示例1：基本范围绘图
        ts = pd.Series([True, True, False, True, True, True, False])
        ranges = vbt.Ranges.from_ts(ts)
        
        # 绘制基本图表
        fig = ranges.plot()
        fig.show()
        
        # 示例2：自定义绘图样式
        fig = ranges.plot(
            plot_zones=True,
            ts_trace_kwargs=dict(line=dict(color='red', width=2)),
            start_trace_kwargs=dict(marker=dict(size=10, color='blue')),
            end_trace_kwargs=dict(marker=dict(size=10, color='green'))
        )
        fig.show()
        
        # 示例3：金融应用 - 绘制回撤期间
        price = pd.Series([100, 102, 98, 95, 97, 101, 103, 99])
        price.index = pd.date_range('2023-01-01', periods=len(price), freq='D')
        
        # 计算回撤
        peak = price.expanding().max()
        drawdown = (price - peak) / peak
        is_drawdown = drawdown < -0.02  # 回撤超过2%
        
        # 创建回撤范围
        dd_ranges = vbt.Ranges.from_ts(is_drawdown, attach_ts=True)
        
        # 绘制回撤图
        fig = dd_ranges.plot(
            title="Price Drawdown Analysis",
            ts_trace_kwargs=dict(name="Drawdown Signal"),
            closed_shape_kwargs=dict(fillcolor='red', opacity=0.3)
        )
        fig.show()
        
        # 示例4：多列数据绘图
        df = pd.DataFrame({
            'Strategy_A': [True, True, False, True, False],
            'Strategy_B': [False, True, True, False, True]
        })
        ranges_multi = vbt.Ranges.from_ts(df)
        
        # 绘制特定列
        fig_a = ranges_multi.plot(column='Strategy_A', title="Strategy A Ranges")
        fig_a.show()
        
        # 示例5：限制显示的范围数量
        fig = ranges.plot(
            top_n=3,  # 只显示持续时间最长的3个范围
            title="Top 3 Longest Ranges"
        )
        fig.show()
        ```
        """
        # 从设置中获取绘图配置
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        # 选择单列数据进行绘图
        self_col = self.select_one(column=column, group_by=False)
        
        # 如果指定了top_n，则按持续时间筛选前N个范围
        if top_n is not None:
            self_col = self_col.apply_mask(self_col.duration.top_n_mask(top_n))

        # 设置时间序列线条的默认参数
        if ts_trace_kwargs is None:
            ts_trace_kwargs = {}
        ts_trace_kwargs = merge_dicts(dict(
            line=dict(
                color=plotting_cfg['color_schema']['blue']  # 使用蓝色作为默认颜色
            )
        ), ts_trace_kwargs)
        
        # 设置各种参数的默认值
        if start_trace_kwargs is None:
            start_trace_kwargs = {}
        if end_trace_kwargs is None:
            end_trace_kwargs = {}
        if open_shape_kwargs is None:
            open_shape_kwargs = {}
        if closed_shape_kwargs is None:
            closed_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        # 创建图表对象（如果没有提供的话）
        if fig is None:
            fig = make_figure()
        
        # 更新图表布局
        fig.update_layout(**layout_kwargs)
        
        # 获取Y轴的显示范围，用于绘制范围区域
        y_domain = get_domain(yref, fig)

        # 如果存在原始时间序列数据，则绘制时间序列线条
        if self_col.ts is not None:
            fig = self_col.ts.vbt.plot(trace_kwargs=ts_trace_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)

        # 如果有范围记录，则绘制范围标记和区域
        if self_col.count() > 0:
            # 提取范围记录的各种信息
            id_ = self_col.get_field_arr('id')                      # 范围ID
            id_title = self_col.get_field_title('id')               # ID字段标题

            start_idx = self_col.get_map_field_to_index('start_idx')  # 起始索引（映射到实际索引）
            start_idx_title = self_col.get_field_title('start_idx')   # 起始索引字段标题
            
            # 获取起始点的Y值
            if self_col.ts is not None:
                start_val = self_col.ts.loc[start_idx]               # 从时间序列中获取起始点的值
            else:
                start_val = np.full(len(start_idx), 0)               # 如果没有时间序列，使用0值

            end_idx = self_col.get_map_field_to_index('end_idx')      # 结束索引（映射到实际索引）
            end_idx_title = self_col.get_field_title('end_idx')       # 结束索引字段标题
            
            # 获取结束点的Y值
            if self_col.ts is not None:
                end_val = self_col.ts.loc[end_idx]                   # 从时间序列中获取结束点的值
            else:
                end_val = np.full(len(end_idx), 0)                   # 如果没有时间序列，使用0值

            # 将持续时间转换为字符串格式，用于悬停提示
            duration = np.vectorize(str)(self_col.wrapper.to_timedelta(
                self_col.duration.values, to_pd=True, silence_warnings=True))

            status = self_col.get_field_arr('status')                # 范围状态数组

            # 绘制范围起始点标记
            start_customdata = id_[:, None]                          # 创建自定义数据数组
            start_scatter = go.Scatter(
                x=start_idx,                                         # X坐标：起始索引
                y=start_val,                                         # Y坐标：起始值
                mode='markers',                                      # 模式：仅标记
                marker=dict(
                    symbol='diamond',                                # 钻石形状
                    color=plotting_cfg['contrast_color_schema']['blue'],  # 蓝色
                    size=7,                                          # 大小
                    line=dict(
                        width=1,                                     # 边框宽度
                        color=adjust_lightness(plotting_cfg['contrast_color_schema']['blue'])  # 边框颜色
                    )
                ),
                name='Start',                                        # 图例名称
                customdata=start_customdata,                         # 自定义数据
                hovertemplate=f"{id_title}: %{{customdata[0]}}"      # 悬停提示模板
                              f"<br>{start_idx_title}: %{{x}}"
            )
            start_scatter.update(**start_trace_kwargs)               # 应用自定义参数
            fig.add_trace(start_scatter, **add_trace_kwargs)         # 添加到图表

            # 处理封闭范围的结束点标记
            closed_mask = status == RangeStatus.Closed               # 筛选封闭范围
            if closed_mask.any():
                # 绘制封闭范围的结束点标记
                closed_end_customdata = np.stack((
                    id_[closed_mask],                                # 范围ID
                    duration[closed_mask]                            # 持续时间
                ), axis=1)
                closed_end_scatter = go.Scatter(
                    x=end_idx[closed_mask],                          # X坐标：结束索引
                    y=end_val[closed_mask],                          # Y坐标：结束值
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        color=plotting_cfg['contrast_color_schema']['green'],  # 绿色表示封闭
                        size=7,
                        line=dict(
                            width=1,
                            color=adjust_lightness(plotting_cfg['contrast_color_schema']['green'])
                        )
                    ),
                    name='Closed',                                   # 图例名称
                    customdata=closed_end_customdata,
                    hovertemplate=f"{id_title}: %{{customdata[0]}}"  # 悬停提示
                                  f"<br>{end_idx_title}: %{{x}}"
                                  f"<br>Duration: %{{customdata[1]}}"
                )
                closed_end_scatter.update(**end_trace_kwargs)
                fig.add_trace(closed_end_scatter, **add_trace_kwargs)

                # 如果启用了区域绘制，绘制封闭范围的区域
                if plot_zones:
                    for i in range(len(id_[closed_mask])):
                        fig.add_shape(**merge_dicts(dict(
                            type="rect",                             # 矩形形状
                            xref=xref,                               # X轴引用
                            yref="paper",                            # Y轴引用（纸张坐标）
                            x0=start_idx[closed_mask][i],            # 起始X坐标
                            y0=y_domain[0],                          # 起始Y坐标
                            x1=end_idx[closed_mask][i],              # 结束X坐标
                            y1=y_domain[1],                          # 结束Y坐标
                            fillcolor='teal',                        # 填充颜色：青色
                            opacity=0.2,                             # 透明度
                            layer="below",                           # 层级：在下方
                            line_width=0,                            # 无边框
                        ), closed_shape_kwargs))

            # 处理开放范围的结束点标记
            open_mask = status == RangeStatus.Open                   # 筛选开放范围
            if open_mask.any():
                # 绘制开放范围的结束点标记
                open_end_customdata = np.stack((
                    id_[open_mask],
                    duration[open_mask]
                ), axis=1)
                open_end_scatter = go.Scatter(
                    x=end_idx[open_mask],
                    y=end_val[open_mask],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        color=plotting_cfg['contrast_color_schema']['orange'],  # 橙色表示开放
                        size=7,
                        line=dict(
                            width=1,
                            color=adjust_lightness(plotting_cfg['contrast_color_schema']['orange'])
                        )
                    ),
                    name='Open',                                     # 图例名称
                    customdata=open_end_customdata,
                    hovertemplate=f"{id_title}: %{{customdata[0]}}"
                                  f"<br>{end_idx_title}: %{{x}}"
                                  f"<br>Duration: %{{customdata[1]}}"
                )
                open_end_scatter.update(**end_trace_kwargs)
                fig.add_trace(open_end_scatter, **add_trace_kwargs)

                # 如果启用了区域绘制，绘制开放范围的区域
                if plot_zones:
                    for i in range(len(id_[open_mask])):
                        fig.add_shape(**merge_dicts(dict(
                            type="rect",
                            xref=xref,
                            yref="paper",
                            x0=start_idx[open_mask][i],
                            y0=y_domain[0],
                            x1=end_idx[open_mask][i],
                            y1=y_domain[1],
                            fillcolor='orange',                      # 填充颜色：橙色
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        ), open_shape_kwargs))

        # 返回完成的图表对象
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """
        绘图默认配置 - 定义Ranges.plots的默认参数
        
        这个属性定义了绘图方法的默认配置，合并了Records基类的配置
        和ranges模块特有的绘图配置。
        
        返回：
            tp.Kwargs: 绘图方法的默认参数字典
        
        配置来源：
        - Records基类的plots_defaults
        - settings中的ranges.plots配置
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建范围对象
        ranges = vbt.Ranges.from_ts(some_time_series)
        
        # 查看默认绘图配置
        defaults = ranges.plots_defaults
        print("绘图默认配置:")
        for key, value in defaults.items():
            print(f"  {key}: {value}")
        
        # 使用默认配置绘图
        fig = ranges.plots()
        fig.show()
        ```
        """
        # 从设置中获取ranges模块的绘图配置
        from vectorbt._settings import settings
        ranges_plots_cfg = settings['ranges']['plots']

        # 合并基类配置和ranges特有配置
        return merge_dicts(
            Records.plots_defaults.__get__(self),  # 获取Records基类的默认配置
            ranges_plots_cfg                       # 合并ranges特有配置
        )

    # 定义子图配置 - 用于plots()方法
    _subplots: tp.ClassVar[Config] = Config(
        dict(
            # 主要的范围绘图子图
            plot=dict(
                title="Ranges",                         # 子图标题
                check_is_not_grouped=True,              # 检查是否未分组
                plot_func='plot',                       # 绘图函数名称
                tags='ranges'                           # 标签
            )
        ),
        copy_kwargs=dict(copy_mode='deep')              # 深拷贝配置
    )

    @property
    def subplots(self) -> Config:
        """
        子图配置 - 返回可用的子图配置
        
        这个属性返回Ranges类可用的子图配置，用于plots()方法。
        目前只有一个主要的范围绘图子图。
        
        返回：
            Config: 子图配置对象
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建范围对象
        ranges = vbt.Ranges.from_ts(some_time_series)
        
        # 查看可用的子图
        subplots = ranges.subplots
        print("可用的子图:")
        for name, config in subplots.items():
            print(f"  {name}: {config.get('title', name)}")
        
        # 使用plots方法创建子图
        fig = ranges.plots()
        fig.show()
        ```
        """
        return self._subplots


# 更新文档字符串 - 为生成的API文档添加字段配置信息
Ranges.override_field_config_doc(__pdoc__)
# 更新文档字符串 - 为生成的API文档添加统计指标信息
Ranges.override_metrics_doc(__pdoc__)
# 更新文档字符串 - 为生成的API文档添加子图配置信息
Ranges.override_subplots_doc(__pdoc__)
