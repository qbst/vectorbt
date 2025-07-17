# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT PX_ACCESSORS MODULE: Plotly Express Pandas访问器模块
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中专门用于集成Plotly Express绘图功能的pandas访问器模块。
该模块通过pandas的访问器机制，为Series和DataFrame对象提供了无缝的Plotly Express
绘图功能，使用户可以直接在pandas数据对象上调用px的所有绘图方法。

核心设计理念：
1. **无缝集成**：通过pandas访问器机制，用户可以通过.vbt.px直接访问Plotly Express的
   所有绘图函数，无需显式的数据转换或函数调用。

2. **智能适配**：自动处理pandas对象与Plotly Express之间的数据格式差异，包括索引标签
   清理、类别排序、Series名称处理等细节。

3. **主题继承**：自动应用vectorbt的全局绘图主题设置，确保生成的图表风格与框架的
   整体视觉风格保持一致。

4. **性能优化**：采用动态方法生成技术，运行时自动为类添加所有支持的绘图方法，
   避免了手动编写大量重复代码的同时保持了高性能。

主要功能特性：
- **动态方法生成**：使用装饰器自动为访问器类添加所有Plotly Express绘图方法
- **数据预处理**：自动处理标签清理、类别排序、数据类型转换等预处理工作
- **主题集成**：自动应用vectorbt的全局绘图配置和主题设置
- **类型区分**：为Series和DataFrame提供专门的访问器实现，确保类型安全

技术实现特点：
- 使用inspect模块动态发现Plotly Express的所有绘图函数
- 通过函数参数检查确定哪些函数支持DataFrame输入
- 使用闭包技术保持函数名称和引用的正确性
- 集成vectorbt的ArrayWrapper和绘图配置系统

应用场景：
- **快速数据探索**：一行代码生成专业的金融图表
- **技术指标可视化**：直接绘制移动平均线、RSI、MACD等技术指标
- **投资组合分析**：可视化投资组合收益、风险分布等
- **时间序列分析**：绘制价格走势、成交量分布等时间序列图表
- **多资产比较**：并排比较不同资产的表现和特征

使用示例：
```python
import pandas as pd
import vectorbt as vbt

# 创建示例数据
price_data = pd.Series([100, 105, 98, 95, 102, 108, 103], 
                      name='Stock Price')

# 使用px访问器绘制线图
fig = price_data.vbt.px.line()
fig.show()

# 多列数据的散点图
df = pd.DataFrame({
    'returns': [0.02, -0.01, 0.03, -0.02, 0.01],
    'volume': [1000, 1200, 800, 1500, 900]
})
fig = df.vbt.px.scatter(x='returns', y='volume')
fig.show()

# 自动应用主题和配置
vbt.settings.set_theme('seaborn')
fig = price_data.vbt.px.bar()  # 自动应用seaborn主题
fig.show()
```

与vectorbt生态系统的关系：
- 集成vectorbt的全局设置系统，自动应用主题和配置
- 使用vectorbt的make_figure函数确保图表格式的一致性
- 通过root_accessors模块注册到pandas的访问器系统
- 为vectorbt的高级分析功能提供可视化支持

该模块是vectorbt框架"易用性"设计理念的重要体现，通过简单的接口为用户提供了
强大的数据可视化能力，特别适合量化分析中的快速数据探索和结果展示。

Plotly Express pandas访问器.

!!! note
    访问器不使用缓存机制，每次调用都会重新生成图表。
"""

from inspect import getmembers, isfunction  # 导入inspect模块的成员检查函数，用于动态发现Plotly Express的所有绘图函数

import pandas as pd  # 导入pandas库，提供数据结构和数据分析功能
import plotly.express as px  # 导入Plotly Express，提供高级绘图API

from vectorbt import _typing as tp  # 导入vectorbt的类型定义模块，提供类型提示支持
from vectorbt.base.accessors import BaseAccessor, BaseDFAccessor, BaseSRAccessor  # 导入基础访问器类
from vectorbt.base.reshape_fns import to_2d_array  # 导入数组重塑函数，用于将数据转换为2D数组格式
from vectorbt.generic.plotting import clean_labels  # 导入标签清理函数，用于处理图表标签的格式化
from vectorbt.root_accessors import register_dataframe_vbt_accessor, register_series_vbt_accessor  # 导入访问器注册函数
from vectorbt.utils import checks  # 导入检查工具模块，提供函数参数检查功能
from vectorbt.utils.config import merge_dicts  # 导入配置合并函数，用于合并字典配置
from vectorbt.utils.figure import make_figure  # 导入图表创建函数，用于生成标准化的Plotly图表


def attach_px_methods(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
    """
    类装饰器：为指定类动态添加Plotly Express的绘图方法
    
    该装饰器是本模块的核心技术实现，它通过反射机制动态发现Plotly Express中的所有绘图函数，
    并为每个函数创建一个对应的类方法。这种设计避免了手动编写大量重复代码，同时确保了
    与Plotly Express的完美兼容性。
    
    工作原理：
    1. 遍历plotly.express模块中的所有函数
    2. 检查函数是否接受'data_frame'参数或为'imshow'函数
    3. 为每个符合条件的函数创建一个包装方法
    4. 将包装方法动态添加到目标类中
    
    参数：
        cls: 需要添加绘图方法的目标类
        
    返回：
        Type[T]: 添加了绘图方法的类
        
    技术细节：
    - 使用闭包技术保持函数名称和引用的正确性
    - 集成vectorbt的主题设置和配置系统
    - 自动处理pandas对象的数据预处理
    - 支持Series和DataFrame的自动类型检测
    
    使用示例：
    ```python
    @attach_px_methods
    class MyPlotAccessor(BaseAccessor):
        pass
    
    # 现在MyPlotAccessor自动拥有所有px绘图方法
    accessor = MyPlotAccessor(data)
    fig = accessor.bar()  # 调用px.bar()方法
    fig = accessor.scatter(x='col1', y='col2')  # 调用px.scatter()方法
    ```
    """

    for px_func_name, px_func in getmembers(px, isfunction):  # 遍历plotly.express模块中的所有函数
        if checks.func_accepts_arg(px_func, 'data_frame') or px_func_name == 'imshow':  # 检查函数是否接受data_frame参数或为imshow函数
            def plot_func(self, *args, _px_func_name: str = px_func_name,
                          _px_func: tp.Callable = px_func, **kwargs) -> tp.BaseFigure:
                """
                动态生成的绘图方法包装器
                
                这个内部函数为每个Plotly Express绘图函数创建一个对应的方法。它负责：
                1. 获取vectorbt的全局绘图配置
                2. 处理数据预处理（标签清理、类别排序等）
                3. 调用原始的Plotly Express函数
                4. 应用vectorbt的图表主题和格式化
                
                参数：
                    self: 访问器实例
                    *args: 位置参数，传递给原始的px函数
                    _px_func_name: 绑定的px函数名称（通过默认参数绑定）
                    _px_func: 绑定的px函数引用（通过默认参数绑定）
                    **kwargs: 关键字参数，传递给原始的px函数
                    
                返回：
                    BaseFigure: 处理后的Plotly图表对象
                """
                from vectorbt._settings import settings  # 导入vectorbt的全局设置对象
                layout_cfg = settings['plotting']['layout']  # 获取绘图布局配置

                layout_kwargs = dict(  # 构建布局参数字典
                    template=kwargs.pop('template', layout_cfg['template']),  # 设置图表模板，优先使用用户指定的模板
                    width=kwargs.pop('width', layout_cfg['width']),  # 设置图表宽度，优先使用用户指定的宽度
                    height=kwargs.pop('height', layout_cfg['height'])  # 设置图表高度，优先使用用户指定的高度
                )
                # 修复category_orders参数以确保分类数据的正确排序
                if 'color' in kwargs:  # 如果指定了颜色参数
                    if isinstance(kwargs['color'], str):  # 如果颜色参数是字符串（列名）
                        if isinstance(self.obj, pd.DataFrame):  # 如果数据对象是DataFrame
                            if kwargs['color'] in self.obj.columns:  # 如果指定的列存在于DataFrame中
                                category_orders = dict()  # 创建分类排序字典
                                category_orders[kwargs['color']] = sorted(self.obj[kwargs['color']].unique())  # 对唯一值进行排序
                                kwargs = merge_dicts(dict(category_orders=category_orders), kwargs)  # 合并分类排序配置

                # 修复Series名称以确保图表标签的正确显示
                obj = self.obj.copy(deep=False)  # 创建数据对象的浅拷贝，避免修改原始数据
                if isinstance(obj, pd.Series):  # 如果数据对象是Series
                    if obj.name is not None:  # 如果Series有名称
                        obj = obj.rename(str(obj.name))  # 将名称转换为字符串格式
                else:  # 如果数据对象是DataFrame
                    obj.columns = clean_labels(obj.columns)  # 清理列名标签，移除特殊字符和格式化
                obj.index = clean_labels(obj.index)  # 清理索引标签，移除特殊字符和格式化

                if _px_func_name == 'imshow':  # 如果是imshow函数（热图显示）
                    return make_figure(_px_func(  # 调用make_figure函数创建标准化图表
                        to_2d_array(obj), *args, **layout_kwargs, **kwargs  # 将数据转换为2D数组格式
                    ), layout=layout_kwargs)  # 应用布局配置
                return make_figure(_px_func(  # 对于其他所有绘图函数
                    obj, *args, **layout_kwargs, **kwargs  # 直接传递处理后的数据对象
                ), layout=layout_kwargs)  # 应用布局配置

            setattr(cls, px_func_name, plot_func)  # 将生成的绘图方法动态添加到目标类中
    return cls  # 返回增强后的类


@attach_px_methods  # 应用装饰器，为PXAccessor类添加所有Plotly Express绘图方法
class PXAccessor(BaseAccessor):
    """
    Plotly Express访问器基类
    
    该类是vectorbt框架中Plotly Express集成的核心访问器类，为pandas的Series和DataFrame
    提供了完整的Plotly Express绘图功能。通过`pd.Series.vbt.px`和`pd.DataFrame.vbt.px`
    可以访问该访问器的所有功能。
    
    核心功能：
    - 提供所有Plotly Express绘图方法的直接访问
    - 自动处理数据预处理和格式化
    - 集成vectorbt的主题设置和配置系统
    - 支持Series和DataFrame的统一接口
    
    设计特点：
    - 继承自BaseAccessor，获得基础的数据处理能力
    - 通过@attach_px_methods装饰器自动添加所有px绘图方法
    - 不使用缓存机制，每次调用都重新生成图表
    - 完全兼容Plotly Express的API和参数
    
    技术实现：
    - 使用动态方法生成技术，避免手动编写重复代码
    - 集成vectorbt的ArrayWrapper系统，保持元数据完整性
    - 自动处理pandas对象与Plotly Express之间的数据格式差异
    
    访问方式：
    可以通过`pd.Series.vbt.px`和`pd.DataFrame.vbt.px`访问
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 设置vectorbt主题
    vbt.settings.set_theme('seaborn')
    
    # 创建示例数据
    price_series = pd.Series([100, 105, 98, 95, 102, 108, 103], name='Stock Price')
    
    # 使用px访问器绘制线图
    fig = price_series.vbt.px.line()
    fig.show()
    
    # 绘制柱状图
    fig = price_series.vbt.px.bar()
    fig.show()
    
    # 对于DataFrame数据
    df = pd.DataFrame({
        'returns': [0.02, -0.01, 0.03, -0.02, 0.01],
        'volume': [1000, 1200, 800, 1500, 900]
    })
    
    # 散点图
    fig = df.vbt.px.scatter(x='returns', y='volume')
    fig.show()
    
    # 直方图
    fig = df['returns'].vbt.px.histogram()
    fig.show()
    
    # 箱线图
    fig = df.vbt.px.box(y='returns')
    fig.show()
    ```
    
    注意事项：
    - 访问器不使用缓存，每次调用都会重新计算
    - 所有Plotly Express的参数都可以正常使用
    - 自动应用vectorbt的全局绘图配置和主题
    - 支持链式调用和方法组合
    
    支持的绘图方法：
    通过@attach_px_methods装饰器，该类自动支持所有Plotly Express的绘图方法，包括：
    - line(): 线图
    - bar(): 柱状图
    - scatter(): 散点图
    - histogram(): 直方图
    - box(): 箱线图
    - violin(): 小提琴图
    - strip(): 条带图
    - area(): 面积图
    - pie(): 饼图
    - sunburst(): 旭日图
    - treemap(): 树状图
    - icicle(): 冰柱图
    - funnel(): 漏斗图
    - timeline(): 时间线图
    - 以及所有其他Plotly Express绘图方法
    
    ![](/assets/images/px_bar.svg)
    """

    def __init__(self, obj: tp.SeriesFrame, **kwargs) -> None:
        """
        初始化PXAccessor实例
        
        该方法初始化Plotly Express访问器，设置必要的数据对象和配置参数。
        
        参数：
            obj: 要包装的pandas Series或DataFrame对象
            **kwargs: 传递给基类的额外配置参数
            
        初始化过程：
        1. 调用基类BaseAccessor的初始化方法
        2. 设置数据对象引用
        3. 应用任何额外的配置参数
        
        使用示例：
        ```python
        # 通常不需要直接调用此方法，而是通过pandas访问器机制自动创建
        series = pd.Series([1, 2, 3])
        accessor = series.vbt.px  # 自动调用__init__方法
        ```
        """
        BaseAccessor.__init__(self, obj, **kwargs)  # 调用基类的初始化方法，传递数据对象和配置参数


@register_series_vbt_accessor('px')  # 注册为Series的vbt访问器，使用'px'作为访问名称
class PXSRAccessor(PXAccessor, BaseSRAccessor):
    """
    专用于Series的Plotly Express访问器
    
    该类是PXAccessor的Series专用版本，提供了针对一维数据优化的Plotly Express绘图功能。
    通过`pd.Series.vbt.px`可以访问该访问器的所有功能。
    
    核心特性：
    - 继承PXAccessor的所有绘图方法
    - 专门针对Series数据进行优化
    - 自动处理Series特有的数据格式问题
    - 集成Series专用的基础访问器功能
    
    类继承关系：
    - PXAccessor: 提供所有Plotly Express绘图方法
    - BaseSRAccessor: 提供Series专用的基础访问器功能
    
    访问方式：
    通过`pd.Series.vbt.px`访问
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建Series数据
    price_series = pd.Series([100, 105, 98, 95, 102, 108, 103], 
                           name='Stock Price',
                           index=pd.date_range('2023-01-01', periods=7))
    
    # 使用Series专用的px访问器
    fig = price_series.vbt.px.line()
    fig.show()
    
    # 柱状图
    fig = price_series.vbt.px.bar()
    fig.show()
    
    # 直方图
    fig = price_series.vbt.px.histogram()
    fig.show()
    
    # 箱线图
    fig = price_series.vbt.px.box()
    fig.show()
    ```
    
    优化特性：
    - 自动处理Series名称的显示
    - 优化了Series索引的标签处理
    - 支持时间序列数据的特殊格式化
    - 自动应用适合一维数据的图表配置
    """

    def __init__(self, obj: tp.Series, **kwargs) -> None:
        """
        初始化Series专用的PX访问器
        
        该方法初始化Series专用的Plotly Express访问器，确保所有相关的基类都正确初始化。
        
        参数：
            obj: 要包装的pandas Series对象
            **kwargs: 传递给基类的额外配置参数
            
        初始化顺序：
        1. 初始化BaseSRAccessor（Series基础访问器）
        2. 初始化PXAccessor（Plotly Express访问器）
        3. 确保所有配置参数正确传递
        
        技术细节：
        - 使用多重继承，需要确保所有基类正确初始化
        - 传递Series对象给所有基类
        - 保持配置参数的一致性
        
        使用示例：
        ```python
        # 通常不需要直接调用此方法
        series = pd.Series([1, 2, 3])
        accessor = series.vbt.px  # 自动调用__init__方法
        ```
        """
        BaseSRAccessor.__init__(self, obj, **kwargs)  # 初始化Series基础访问器
        PXAccessor.__init__(self, obj, **kwargs)  # 初始化Plotly Express访问器


@register_dataframe_vbt_accessor('px')  # 注册为DataFrame的vbt访问器，使用'px'作为访问名称
class PXDFAccessor(PXAccessor, BaseDFAccessor):
    """
    专用于DataFrame的Plotly Express访问器
    
    该类是PXAccessor的DataFrame专用版本，提供了针对二维数据优化的Plotly Express绘图功能。
    通过`pd.DataFrame.vbt.px`可以访问该访问器的所有功能。
    
    核心特性：
    - 继承PXAccessor的所有绘图方法
    - 专门针对DataFrame数据进行优化
    - 自动处理多列数据的复杂绘图需求
    - 集成DataFrame专用的基础访问器功能
    
    类继承关系：
    - PXAccessor: 提供所有Plotly Express绘图方法
    - BaseDFAccessor: 提供DataFrame专用的基础访问器功能
    
    访问方式：
    通过`pd.DataFrame.vbt.px`访问
    
    使用示例：
    ```python
    import pandas as pd
    import vectorbt as vbt
    
    # 创建DataFrame数据
    df = pd.DataFrame({
        'AAPL': [150, 155, 148, 160, 165],
        'GOOGL': [2800, 2850, 2750, 2900, 2950],
        'MSFT': [300, 305, 295, 310, 315]
    }, index=pd.date_range('2023-01-01', periods=5))
    
    # 使用DataFrame专用的px访问器
    fig = df.vbt.px.line()  # 多条线图
    fig.show()
    
    # 相关性热图
    fig = df.corr().vbt.px.imshow()
    fig.show()
    
    # 散点图矩阵
    fig = df.vbt.px.scatter_matrix()
    fig.show()
    
    # 长格式数据的散点图
    df_long = df.reset_index().melt(id_vars='index', var_name='symbol', value_name='price')
    fig = df_long.vbt.px.scatter(x='index', y='price', color='symbol')
    fig.show()
    ```
    
    优化特性：
    - 智能处理多列数据的颜色映射
    - 自动清理列名标签
    - 支持复杂的分类排序
    - 优化了DataFrame索引的标签处理
    - 自动应用适合二维数据的图表配置
    """

    def __init__(self, obj: tp.Frame, **kwargs) -> None:
        """
        初始化DataFrame专用的PX访问器
        
        该方法初始化DataFrame专用的Plotly Express访问器，确保所有相关的基类都正确初始化。
        
        参数：
            obj: 要包装的pandas DataFrame对象
            **kwargs: 传递给基类的额外配置参数
            
        初始化顺序：
        1. 初始化BaseDFAccessor（DataFrame基础访问器）
        2. 初始化PXAccessor（Plotly Express访问器）
        3. 确保所有配置参数正确传递
        
        技术细节：
        - 使用多重继承，需要确保所有基类正确初始化
        - 传递DataFrame对象给所有基类
        - 保持配置参数的一致性
        - 支持复杂的列分组和标签处理
        
        使用示例：
        ```python
        # 通常不需要直接调用此方法
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        accessor = df.vbt.px  # 自动调用__init__方法
        ```
        """
        BaseDFAccessor.__init__(self, obj, **kwargs)  # 初始化DataFrame基础访问器
        PXAccessor.__init__(self, obj, **kwargs)  # 初始化Plotly Express访问器
