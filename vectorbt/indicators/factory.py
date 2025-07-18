# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
技术指标工厂模块 - 用于简化构建复杂技术指标的工厂类

================================================================================
文件设计逻辑和总体作用：
================================================================================
这个文件是vectorbt量化交易框架中的核心组件，用于构建技术指标的工厂类。
主要功能包括：

1. 指标工厂类 (IndicatorFactory)：
   - 提供便捷的方式创建任意复杂度的技术指标
   - 支持多种参数组合和输入数据格式
   - 自动处理数据广播和参数优化

2. 指标管道 (Pipeline)：
   - 标准化指标计算流程
   - 支持输入数组、参数数组和其他相关参数
   - 自动处理参数组合的计算和结果拼接

3. 第三方库集成：
   - 支持与TA-Lib、pandas_ta、ta等主流技术分析库的集成
   - 提供统一的接口和参数管理

4. 性能优化：
   - 支持缓存和智能重用计算结果
   - 提供Numba加速的计算函数
   - 优化内存使用和计算效率

5. 数据处理：
   - 支持pandas DataFrame/Series和NumPy数组
   - 自动处理数据广播和形状匹配
   - 支持多维参数和灵活的参数配置

这个工厂类的设计使得用户可以轻松创建复杂的技术指标，而无需重复编写
相同的数据处理和参数管理代码。
================================================================================

A factory for building new indicators with ease.

The indicator factory class `IndicatorFactory` offers a convenient way to create technical
indicators of any complexity. By providing it with information such as calculation functions and
您的输入、参数和输出的名称，它将创建一个独立的指标类，
能够为您的输入和参数的任意组合运行指标。它还创建了信号生成方法，
并支持常见的pandas和参数索引操作。

每个指标基本上是一个管道，它：

* 接受输入数组列表（例如，OHLCV数据）
* 接受参数数组列表（例如，窗口大小）
* 接受其他相关参数和关键字参数
* 对于每个参数组合，对输入数组执行计算
* 将结果连接到新的输出数组中（例如，滚动平均）

这个管道可以很好地标准化，这是通过 `run_pipeline` 完成的。

`IndicatorFactory` 通过生成和预配置一个新的Python类来简化 `run_pipeline` 的使用，
该类具有用于运行指标的各种类方法。

每个生成的类包含以下特性：

* 通过广播接受任何兼容形状的输入数组
* 接受就地写入的输出数组而不是返回
* 接受任意参数网格
* 开箱即用地支持缓存和其他优化
* 支持pandas和参数索引
* 为所有输入、输出和属性提供辅助方法

考虑以下由两列组成的价格DataFrame，每个资产一列：

```pycon
>>> import vectorbt as vbt
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from datetime import datetime

>>> price = pd.DataFrame({
...     'a': [1, 2, 3, 4, 5],
...     'b': [5, 4, 3, 2, 1]
... }, index=pd.Index([
...     datetime(2020, 1, 1),
...     datetime(2020, 1, 2),
...     datetime(2020, 1, 3),
...     datetime(2020, 1, 4),
...     datetime(2020, 1, 5),
... ])).astype(float)
>>> price
            a    b
2020-01-01  1.0  5.0
2020-01-02  2.0  4.0
2020-01-03  3.0  3.0
2020-01-04  4.0  2.0
2020-01-05  5.0  1.0
```

对于DataFrame中的每一列，让我们计算一个简单移动平均并获得其与价格的交叉点。
特别是，我们想要测试两个不同的窗口大小：2和3。

## Naive approach

A naive way of doing this:

```pycon
>>> ma_df = pd.DataFrame.vbt.concat(
...     price.rolling(window=2).mean(),
...     price.rolling(window=3).mean(),
...     keys=pd.Index([2, 3], name='ma_window'))
>>> ma_df
ma_window          2         3
              a    b    a    b
2020-01-01  NaN  NaN  NaN  NaN
2020-01-02  1.5  4.5  NaN  NaN
2020-01-03  2.5  3.5  2.0  4.0
2020-01-04  3.5  2.5  3.0  3.0
2020-01-05  4.5  1.5  4.0  2.0

>>> above_signals = (price.vbt.tile(2).vbt > ma_df)
>>> above_signals = above_signals.vbt.signals.first(after_false=True)
>>> above_signals
ma_window              2             3
                a      b      a      b
2020-01-01  False  False  False  False
2020-01-02   True  False  False  False
2020-01-03  False  False   True  False
2020-01-04  False  False  False  False
2020-01-05  False  False  False  False

>>> below_signals = (price.vbt.tile(2).vbt < ma_df)
>>> below_signals = below_signals.vbt.signals.first(after_false=True)
>>> below_signals
ma_window              2             3
                a      b      a      b
2020-01-01  False  False  False  False
2020-01-02  False   True  False  False
2020-01-03  False  False  False   True
2020-01-04  False  False  False  False
2020-01-05  False  False  False  False
```

Now the same using `IndicatorFactory`:

```pycon
>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     param_names=['window'],
...     output_names=['ma'],
... ).from_apply_func(vbt.nb.rolling_mean_nb)

>>> myind = MyInd.run(price, [2, 3])
>>> above_signals = myind.price_crossed_above(myind.ma)
>>> below_signals = myind.price_crossed_below(myind.ma)
```

The `IndicatorFactory` class is used to construct indicator classes from UDFs. First, we provide
all the necessary information (indicator config) to build the facade of the indicator, such as the names
of inputs, parameters, and outputs, and the actual calculation function. The factory then generates a
self-contained indicator class capable of running arbitrary configurations of inputs and parameters.
要运行任何配置，我们可以使用 `run` 方法（如我们上面所做的）或 `run_combs` 方法。

## run and run_combs methods

The main method to run an indicator is `run`, which accepts arguments based on the config
provided to the `IndicatorFactory` (see the example above). These arguments include input arrays,
in-place output arrays, parameters, and arguments for `run_pipeline`.

`run_combs` 方法接受与上述方法相同的输入，但基于组合函数计算传递参数的所有组合，
并返回可以相互比较的多个实例。例如，这对于生成多个移动平均的交叉信号很有用：

```pycon
>>> myind1, myind2 = MyInd.run_combs(price, [2, 3, 4])

>>> myind1.ma
myind_1_window                  2         3
                 a    b    a    b    a    b
2020-01-01     NaN  NaN  NaN  NaN  NaN  NaN
2020-01-02     1.5  4.5  1.5  4.5  NaN  NaN
2020-01-03     2.5  3.5  2.5  3.5  2.0  4.0
2020-01-04     3.5  2.5  3.5  2.5  3.0  3.0
2020-01-05     4.5  1.5  4.5  1.5  4.0  2.0

>>> myind2.ma
myind_2_window        3                   4
                 a    b    a    b    a    b
2020-01-01     NaN  NaN  NaN  NaN  NaN  NaN
2020-01-02     NaN  NaN  NaN  NaN  NaN  NaN
2020-01-03     2.0  4.0  NaN  NaN  NaN  NaN
2020-01-04     3.0  3.0  2.5  3.5  2.5  3.5
2020-01-05     4.0  2.0  3.5  2.5  3.5  2.5

>>> myind1.ma_crossed_above(myind2.ma)
myind_1_window                          2             3
myind_2_window            3             4             4
                   a      b      a      b      a      b
2020-01-01     False  False  False  False  False  False
2020-01-02     False  False  False  False  False  False
2020-01-03      True  False  False  False  False  False
2020-01-04     False  False   True  False   True  False
2020-01-05     False  False  False  False  False  False
```

它的主要优势是由于智能缓存，它不需要重新计算每个组合。

要了解任何类方法接受的参数详情，请使用 `help`：

```pycon
>>> help(MyInd.run)
Help on method run:

run(price, window, short_name='custom', hide_params=None, hide_default=True, **kwargs) method of builtins.type instance
    运行 `Indicator` 指标。

    * 输入: `price`
    * 参数: `window`
    * 输出: `ma`

    传递参数名称列表作为 `hide_params` 来隐藏它们的列级别。
    设置 `hide_default` 为 False 以显示具有默认值的参数的列级别。

    其他关键字参数将传递给 `vectorbt.indicators.factory.run_pipeline`。
```

## Parameters

`IndicatorFactory` allows definition of arbitrary parameter grids.

Parameters are variables that can hold one or more values. A single value can be passed as a
scalar, an array, or any other object. Multiple values are passed as a list or an array
(if the flag `is_array_like` is set to False for that parameter). If there are multiple parameters
and each is having multiple values, their values will broadcast to a single shape:

```plaintext
       p1         p2            result
0       0          1          [(0, 1)]
1  [0, 1]        [2]  [(0, 2), (1, 2)]
2  [0, 1]     [2, 3]  [(0, 2), (1, 3)]
3  [0, 1]  [2, 3, 4]             error
```

为了说明指标中参数的使用，让我们构建一个基本指标，如果滚动平均在上下界内则返回1，
如果在界外则返回-1：

```pycon
>>> @njit
... def apply_func_nb(price, window, lower, upper):
...     output = np.full(price.shape, np.nan, dtype=np.float64)
...     for col in range(price.shape[1]):
...         for i in range(window, price.shape[0]):
...             mean = np.mean(price[i - window:i, col])
...             output[i, col] = lower < mean < upper
...     return output

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     param_names=['window', 'lower', 'upper'],
...     output_names=['output']
... ).from_apply_func(apply_func_nb)
```

By default, when `per_column` is set to False, each parameter is applied to the entire input.

One parameter combination:

```pycon
>>> MyInd.run(
...     price,
...     window=2,
...     lower=3,
...     upper=5
... ).output
custom_window         2
custom_lower          3
custom_upper          5
                 a    b
2020-01-01     NaN  NaN
2020-01-02     NaN  NaN
2020-01-03     0.0  1.0
2020-01-04     0.0  1.0
2020-01-05     1.0  0.0
```

Multiple parameter combinations:

```pycon
>>> MyInd.run(
...     price,
...     window=[2, 3],
...     lower=3,
...     upper=5
... ).output
custom_window         2         3
custom_lower          3         3
custom_upper          5         5
                 a    b    a    b
2020-01-01     NaN  NaN  NaN  NaN
2020-01-02     NaN  NaN  NaN  NaN
2020-01-03     0.0  1.0  NaN  NaN
2020-01-04     0.0  1.0  0.0  1.0
2020-01-05     1.0  0.0  0.0  0.0
```

Product of parameter combinations:

```pycon
>>> MyInd.run(
...     price,
...     window=[2, 3],
...     lower=[3, 4],
...     upper=5,
...     param_product=True
... ).output
custom_window                   2                   3
custom_lower          3         4         3         4
custom_upper          5         5         5         5
                 a    b    a    b    a    b    a    b
2020-01-01     NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
2020-01-02     NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
2020-01-03     0.0  1.0  0.0  1.0  NaN  NaN  NaN  NaN
2020-01-04     0.0  1.0  0.0  0.0  0.0  1.0  0.0  0.0
2020-01-05     1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```

Multiple parameter combinations, one per column:

```pycon
>>> MyInd.run(
...     price,
...     window=[2, 3],
...     lower=[3, 4],
...     upper=5,
...     per_column=True
... ).output
custom_window    2    3
custom_lower     3    4
custom_upper     5    5
                 a    b
2020-01-01     NaN  NaN
2020-01-02     NaN  NaN
2020-01-03     0.0  NaN
2020-01-04     0.0  0.0
2020-01-05     1.0  0.0
```

Parameter defaults can be passed directly to the `IndicatorFactory.from_custom_func` and
`IndicatorFactory.from_apply_func`, and overridden in the run method:

```pycon
>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     param_names=['window', 'lower', 'upper'],
...     output_names=['output']
... ).from_apply_func(apply_func_nb, window=2, lower=3, upper=4)

>>> MyInd.run(price, upper=5).output
custom_window         2
custom_lower          3
custom_upper          5
                 a    b
2020-01-01     NaN  NaN
2020-01-02     NaN  NaN
2020-01-03     0.0  1.0
2020-01-04     0.0  1.0
2020-01-05     1.0  0.0
```

某些参数需要按输入的行、列或元素定义。
默认情况下，如果我们传递参数值作为数组，指标会将该数组视为多个值的列表 - 每个输入一个值。
要使指标将此数组视为单个值，请在 `param_settings` 中将标志 `is_array_like` 设置为 True。
此外，要自动将传递的标量/数组广播到输入形状，请将 `bc_to_input` 设置为 True、0（索引轴）或 1（列轴）。

在我们的示例中，参数 `window` 可以按列广播，参数 `lower` 和 `upper` 都可以按元素广播：

```pycon
>>> @njit
... def apply_func_nb(price, window, lower, upper):
...     output = np.full(price.shape, np.nan, dtype=np.float64)
...     for col in range(price.shape[1]):
...         for i in range(window[col], price.shape[0]):
...             mean = np.mean(price[i - window[col]:i, col])
...             output[i, col] = lower[i, col] < mean < upper[i, col]
...     return output

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     param_names=['window', 'lower', 'upper'],
...     output_names=['output']
... ).from_apply_func(
...     apply_func_nb,
...     param_settings=dict(
...         window=dict(is_array_like=True, bc_to_input=1, per_column=True),
...         lower=dict(is_array_like=True, bc_to_input=True),
...         upper=dict(is_array_like=True, bc_to_input=True)
...     )
... )

>>> MyInd.run(
...     price,
...     window=[np.array([2, 3]), np.array([3, 4])],
...     lower=np.array([1, 2]),
...     upper=np.array([3, 4]),
... ).output
custom_window       2       3               4
custom_lower  array_0 array_0 array_1 array_1
custom_upper  array_0 array_0 array_1 array_1
                    a       b       a       b
2020-01-01        NaN     NaN     NaN     NaN
2020-01-02        NaN     NaN     NaN     NaN
2020-01-03        1.0     NaN     NaN     NaN
2020-01-04        1.0     0.0     1.0     NaN
2020-01-05        0.0     1.0     0.0     1.0
```

Broadcasting a huge number of parameters to the input shape can consume lots of memory,
especially when the array materializes. Luckily, vectorbt implements flexible broadcasting,
which preserves the original dimensions of the parameter. This requires two changes:
setting `keep_raw` to True in `broadcast_kwargs` and passing `flex_2d` to the apply function.

There are two configs in `vectorbt.indicators.configs` exactly for this purpose: one for column-wise
broadcasting and one for element-wise broadcasting:

```pycon
>>> from vectorbt.base.reshape_fns import flex_select_auto_nb
>>> from vectorbt.indicators.configs import flex_col_param_config, flex_elem_param_config

>>> @njit
... def apply_func_nb(price, window, lower, upper, flex_2d):
...     output = np.full(price.shape, np.nan, dtype=np.float64)
...     for col in range(price.shape[1]):
...         _window = flex_select_auto_nb(window, 0, col, flex_2d)
...         for i in range(_window, price.shape[0]):
...             _lower = flex_select_auto_nb(lower, i, col, flex_2d)
...             _upper = flex_select_auto_nb(upper, i, col, flex_2d)
...             mean = np.mean(price[i - _window:i, col])
...             output[i, col] = _lower < mean < _upper
...     return output

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     param_names=['window', 'lower', 'upper'],
...     output_names=['output']
... ).from_apply_func(
...     apply_func_nb,
...     param_settings=dict(
...         window=flex_col_param_config,
...         lower=flex_elem_param_config,
...         upper=flex_elem_param_config
...     ),
...     pass_flex_2d=True
... )
```

现在两个界限参数都可以作为标量（整个输入的值）、一维数组（每行或每列的值，
取决于输入是Series还是DataFrame）、二维数组（每个元素的值）或这些的列表来传递。
这允许以最低的内存成本获得最高的参数灵活性。

例如，让我们构建一个包含两个参数组合的网格，每个组合为每列一个窗口大小，
每个元素都有上下界：

```pycon
>>> MyInd.run(
...     price,
...     window=[np.array([2, 3]), np.array([3, 4])],
...     lower=price.values - 3,
...     upper=price.values + 3,
... ).output
custom_window       2       3               4
custom_lower  array_0 array_0 array_1 array_1
custom_upper  array_0 array_0 array_1 array_1
                    a       b       a       b
2020-01-01        NaN     NaN     NaN     NaN
2020-01-02        NaN     NaN     NaN     NaN
2020-01-03        1.0     NaN     NaN     NaN
2020-01-04        1.0     1.0     1.0     NaN
2020-01-05        1.0     1.0     1.0     1.0
```

指标也可以是无参数的。参见 `vectorbt.indicators.basic.OBV`。

## Inputs

`IndicatorFactory` supports passing none, one, or multiple inputs. If multiple inputs are passed,
it tries to broadcast them into a single shape.

请记住，在vectorbt中，每一列意味着一个单独的回测实例。这就是为什么为了使用多个信息片段
（如开盘价、最高价、最低价、收盘价和成交量），我们需要将它们作为单独的pandas对象提供，
而不是单个DataFrame。

让我们创建一个无参数指标，测量每个K线内收盘价的位置：

```pycon
>>> @njit
... def apply_func_nb(high, low, close):
...     return (close - low) / (high - low)

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['high', 'low', 'close'],
...     output_names=['output']
... ).from_apply_func(apply_func_nb)

>>> MyInd.run(price + 1, price - 1, price).output
              a    b
2020-01-01  0.5  0.5
2020-01-02  0.5  0.5
2020-01-03  0.5  0.5
2020-01-04  0.5  0.5
2020-01-05  0.5  0.5
```

为了演示广播，让我们传递high作为DataFrame，low作为Series，close作为标量：

```pycon
>>> df = pd.DataFrame(np.random.uniform(1, 2, size=(5, 2)))
>>> sr = pd.Series(np.random.uniform(0, 1, size=5))
>>> MyInd.run(df, sr, 1).output
          0         1
0  0.960680  0.666820
1  0.400646  0.528456
2  0.093467  0.134777
3  0.037210  0.102411
4  0.529012  0.652602
```

默认情况下，如果传递了Series，它会自动扩展为二维数组。
要保持为一维，请将 `to_2d` 设置为 False。

类似于参数，我们也可以为输入定义默认值。除了使用标量和数组作为默认值外，我们还可以引用其他输入：

```pycon
>>> @njit
... def apply_func_nb(ts1, ts2, ts3):
...     return ts1 + ts2 + ts3

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['ts1', 'ts2', 'ts3'],
...     output_names=['output']
... ).from_apply_func(apply_func_nb, ts2='ts1', ts3='ts1')

>>> MyInd.run(price).output
               a     b
2020-01-01   3.0  15.0
2020-01-02   6.0  12.0
2020-01-03   9.0   9.0
2020-01-04  12.0   6.0
2020-01-05  15.0   3.0

>>> MyInd.run(price, ts2=price * 2).output
               a     b
2020-01-01   4.0  20.0
2020-01-02   8.0  16.0
2020-01-03  12.0  12.0
2020-01-04  16.0   8.0
2020-01-05  20.0   4.0
```

如果指标不接受任何输入数组怎么办？在这种情况下，我们可以强制用户至少提供输入形状。
让我们定义一个生成器，模拟随机回报并生成合成价格：

```pycon
>>> @njit
... def apply_func_nb(input_shape, start, mu, sigma):
...     rand_returns = np.random.normal(mu, sigma, input_shape)
...     return start * vbt.nb.nancumprod_nb(rand_returns + 1)

>>> MyInd = vbt.IndicatorFactory(
...     param_names=['start', 'mu', 'sigma'],
...     output_names=['output']
... ).from_apply_func(
...     apply_func_nb,
...     require_input_shape=True,
...     seed=42
... )

>>> MyInd.run(price.shape, 100, 0, 0.01).output
custom_start                     100
custom_mu                          0
custom_sigma        0.01        0.01
0             100.496714   99.861736
1             101.147620  101.382660
2             100.910779  101.145285
3             102.504375  101.921510
4             102.023143  102.474495
```

我们还可以向运行方法提供pandas元数据，如 `input_index` 和 `input_columns`：

```pycon
>>> MyInd.run(
...     price.shape, 100, 0, 0.01,
...     input_index=price.index, input_columns=price.columns
... ).output
custom_start                     100
custom_mu                          0
custom_sigma        0.01        0.01
                       a           b
2020-01-01    100.496714   99.861736
2020-01-02    101.147620  101.382660
2020-01-03    100.910779  101.145285
2020-01-04    102.504375  101.921510
2020-01-05    102.023143  102.474495
```

One can even build input-less indicator that decides on the output shape dynamically:

```pycon
>>> from vectorbt.base.combine_fns import apply_and_concat_one

>>> def apply_func(i, ps, input_shape):
...      out = np.full(input_shape, 0)
...      out[:ps[i]] = 1
...      return out

>>> def custom_func(ps):
...     input_shape = (np.max(ps),)
...     return apply_and_concat_one(len(ps), apply_func, ps, input_shape)

>>> MyInd = vbt.IndicatorFactory(
...     param_names=['p'],
...     output_names=['output']
... ).from_custom_func(custom_func)

>>> MyInd.run([1, 2, 3, 4, 5]).output
custom_p  1  2  3  4  5
0         1  1  1  1  1
1         0  1  1  1  1
2         0  0  1  1  1
3         0  0  0  1  1
4         0  0  0  0  1
```

## Outputs

There are two types of outputs: regular and in-place outputs:

* Regular outputs are one or more arrays returned by the function. Each should have an exact
same shape and match the number of columns in the input multiplied by the number of parameter values.
* In-place outputs are not returned but modified in-place. They broadcast together with inputs
and are passed to the calculation function as a list, one per parameter.

Two regular outputs:

```pycon
>>> @njit
... def apply_func_nb(price):
...     return price - 1, price + 1

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     output_names=['out1', 'out2']
... ).from_apply_func(apply_func_nb)

>>> myind = MyInd.run(price)
>>> pd.testing.assert_frame_equal(myind.out1, myind.price - 1)
>>> pd.testing.assert_frame_equal(myind.out2, myind.price + 1)
```

One regular output and one in-place output:

```pycon
>>> @njit
... def apply_func_nb(price, in_out2):
...     in_out2[:] = price + 1
...     return price - 1

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     output_names=['out1'],
...     in_output_names=['in_out2']
... ).from_apply_func(apply_func_nb)

>>> myind = MyInd.run(price)
>>> pd.testing.assert_frame_equal(myind.out1, myind.price - 1)
>>> pd.testing.assert_frame_equal(myind.in_out2, myind.price + 1)
```

Two in-place outputs:

```pycon
>>> @njit
... def apply_func_nb(price, in_out1, in_out2):
...     in_out1[:] = price - 1
...     in_out2[:] = price + 1

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     in_output_names=['in_out1', 'in_out2']
... ).from_apply_func(apply_func_nb)

>>> myind = MyInd.run(price)
>>> pd.testing.assert_frame_equal(myind.in_out1, myind.price - 1)
>>> pd.testing.assert_frame_equal(myind.in_out2, myind.price + 1)
```

默认情况下，就地输出创建为具有未初始化值的空数组。
这允许创建可选输出，如果未写入，则不会占用太多内存。
由于不是所有输出都是 `float` 数据类型，我们可以在 `in_output_settings` 中传递 `dtype`。

```pycon
>>> @njit
... def apply_func_nb(price, in_out):
...     in_out[:] = price > np.mean(price)

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     in_output_names=['in_out']
... ).from_apply_func(
...     apply_func_nb,
...     in_output_settings=dict(in_out=dict(dtype=bool))
... )

>>> MyInd.run(price).in_out
                a      b
2020-01-01  False   True
2020-01-02  False   True
2020-01-03  False  False
2020-01-04   True  False
2020-01-05   True  False
```

Another advantage of in-place outputs is that we can provide their initial state:

```pycon
>>> @njit
... def apply_func_nb(price, in_out1, in_out2):
...     in_out1[:] = in_out1 + price
...     in_out2[:] = in_out2 + price

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     in_output_names=['in_out1', 'in_out2']
... ).from_apply_func(
...     apply_func_nb,
...     in_out1=100,
...     in_out2='price'
... )

>>> myind = MyInd.run(price)
>>> myind.in_out1
              a    b
2020-01-01  101  105
2020-01-02  102  104
2020-01-03  103  103
2020-01-04  104  102
2020-01-05  105  101
>>> myind.in_out2
               a     b
2020-01-01   2.0  10.0
2020-01-02   4.0   8.0
2020-01-03   6.0   6.0
2020-01-04   8.0   4.0
2020-01-05  10.0   2.0
```

## Without Numba

也可以提供未经Numba编译的函数。这在使用第三方库时很有用（参见 `IndicatorFactory.from_talib` 的实现）。
此外，我们可以设置 `keep_pd` 为 True 以将所有输入作为pandas对象传递，而不是原始NumPy数组。

!!! note
    将提供已广播的pandas元数据；也就是说，每个输入数组都将具有相同的索引和列。

让我们通过封装一个基本的复合[pandas_ta](https://github.com/twopirllc/pandas-ta)策略来演示这一点：

```pycon
>>> import pandas_ta

>>> def apply_func(open, high, low, close, volume, ema_len, linreg_len):
...     df = pd.DataFrame(dict(open=open, high=high, low=low, close=close, volume=volume))
...     df.ta.strategy(pandas_ta.Strategy("MyStrategy", [
...         dict(kind='ema', length=ema_len),
...         dict(kind='linreg', close='EMA_' + str(ema_len), length=linreg_len)
...     ]))
...     return tuple([df.iloc[:, i] for i in range(5, len(df.columns))])

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['open', 'high', 'low', 'close', 'volume'],
...     param_names=['ema_len', 'linreg_len'],
...     output_names=['ema', 'ema_linreg']
... ).from_apply_func(
...     apply_func,
...     keep_pd=True,
...     to_2d=False
... )

>>> my_ind = MyInd.run(
...     ohlcv['Open'],
...     ohlcv['High'],
...     ohlcv['Low'],
...     ohlcv['Close'],
...     ohlcv['Volume'],
...     ema_len=5,
...     linreg_len=[8, 9, 10]
... )

>>> my_ind.ema_linreg
custom_ema_len                                            5
custom_linreg_len            8             9             10
date
2021-02-02                  NaN           NaN           NaN
2021-02-03                  NaN           NaN           NaN
2021-02-04                  NaN           NaN           NaN
2021-02-05                  NaN           NaN           NaN
2021-02-06                  NaN           NaN           NaN
...                         ...           ...           ...
2021-02-25         52309.302811  52602.005326  52899.576568
2021-02-26         50797.264793  51224.188381  51590.825690
2021-02-28         49217.904905  49589.546052  50066.206828
2021-03-01         48316.305403  48553.540713  48911.701664
2021-03-02         47984.395969  47956.885953  48150.929668
```

在上面的示例中，只能传递每个open、high、low、close和volume的一个Series。
要使指标能够处理二维数据，请将 `to_2d` 设置为 True 并在 `apply_func` 中创建对每列的循环。

!!! hint
    编写原生Numba编译代码可能提供比在pandas上工作的库高出数个数量级的性能。

## Raw outputs and caching

`IndicatorFactory` 尽可能重用计算结果。由于它最初是为超参数优化而设计的，
有时参数值会重复出现，避免一遍又一遍地处理相同参数对于良好的性能是不可避免的。
例如，当使用 `run_combs` 方法并设置 `run_unique` 为 True 时，
它首先计算所有唯一参数组合的原始输出，然后使用它们为整个参数网格构建输出。

让我们首先通过设置 `return_raw` 为 True 来查看典型的原始输出：

```pycon
>>> raw = vbt.MA.run(price, 2, [False, True], return_raw=True)
>>> raw
([array([[       nan,        nan,        nan,        nan],
         [1.5       , 4.5       , 1.66666667, 4.33333333],
         [2.5       , 3.5       , 2.55555556, 3.44444444],
         [3.5       , 2.5       , 3.51851852, 2.48148148],
         [4.5       , 1.5       , 4.50617284, 1.49382716]])],
 [(2, False), (2, True)],
 2,
 [])
```

It consists of a list of the returned output arrays, a list of the zipped parameter combinations,
the number of input columns, and other objects returned along with output arrays but not listed
in `output_names`. The next time we decide to run the indicator on a subset of the parameters above,
we can simply pass this tuple as the `use_raw` argument. This won't call the calculation function and
will throw an error if some of the requested parameter combinations cannot be found in `raw`.

```pycon
>>> vbt.MA.run(price, 2, True, use_raw=raw).ma
ma_window                    2
ma_ewm                    True
                   a         b
2020-01-01       NaN       NaN
2020-01-02  1.666667  4.333333
2020-01-03  2.555556  3.444444
2020-01-04  3.518519  2.481481
2020-01-05  4.506173  1.493827
```

Here is how the performance compares when repeatedly running the same parameter combination
with and without `run_unique`:

```pycon
>>> a = np.random.uniform(size=(1000,))

>>> %timeit vbt.MA.run(a, np.full(1000, 2), run_unique=False)
73.4 ms ± 4.76 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit vbt.MA.run(a, np.full(1000, 2), run_unique=True)
8.99 ms ± 114 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

!!! note
    `run_unique` is disabled by default.

Enable `run_unique` if input arrays have few columns and there are tons of repeated parameter combinations.
Disable `run_unique` if input arrays are very wide, if two identical parameter combinations can lead to
different results, or when requesting raw output, cache, or additional outputs outside of `output_names`.

Another performance enhancement can be introduced by caching, which has to be implemented by the user.
The class method `IndicatorFactory.from_apply_func` has an argument `cache_func`, which is called
prior to the main calculation.

考虑以下情况：我们想要计算两个昂贵的滚动窗口之间的相对距离。我们已经决定了第一个窗口的值，
并且想要测试第二个窗口的数千个值。没有缓存，即使启用了 `run_unique`，
第一个滚动窗口也会反复重新计算并浪费我们的资源：

```pycon
>>> @njit
... def roll_mean_expensive_nb(price, w):
...     for i in range(100):
...         out = vbt.nb.rolling_mean_nb(price, w)
...     return out

>>> @njit
... def apply_func_nb(price, w1, w2):
...     roll_mean1 = roll_mean_expensive_nb(price, w1)
...     roll_mean2 = roll_mean_expensive_nb(price, w2)
...     return (roll_mean2 - roll_mean1) / roll_mean1

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     param_names=['w1', 'w2'],
...     output_names=['output'],
... ).from_apply_func(apply_func_nb)

>>> MyInd.run(price, 2, 3).output
custom_w1                    2
custom_w2                    3
                   a         b
2020-01-01       NaN       NaN
2020-01-02       NaN       NaN
2020-01-03 -0.200000  0.142857
2020-01-04 -0.142857  0.200000
2020-01-05 -0.111111  0.333333

>>> %timeit MyInd.run(price, 2, np.arange(2, 1000))
264 ms ± 3.22 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

为了避免这种情况，让我们缓存所有唯一的滚动窗口：

```pycon
>>> @njit
... def cache_func_nb(price, ws1, ws2):
...     cache_dict = dict()
...     ws = ws1.copy()
...     ws.extend(ws2)
...     for i in range(len(ws)):
...         h = hash((ws[i]))
...         if h not in cache_dict:
...             cache_dict[h] = roll_mean_expensive_nb(price, ws[i])
...     return cache_dict

>>> @njit
... def apply_func_nb(price, w1, w2, cache_dict):
...     return (cache_dict[hash(w2)] - cache_dict[hash(w1)]) / cache_dict[hash(w1)]

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     param_names=['w1', 'w2'],
...     output_names=['output'],
... ).from_apply_func(apply_func_nb, cache_func=cache_func_nb)

>>> MyInd.run(price, 2, 3).output
custom_w1                    2
custom_w2                    3
                   a         b
2020-01-01       NaN       NaN
2020-01-02       NaN       NaN
2020-01-03 -0.200000  0.142857
2020-01-04 -0.142857  0.200000
2020-01-05 -0.111111  0.333333

>>> %timeit MyInd.run(price, 2, np.arange(2, 1000))
145 ms ± 4.55 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

We have cut down the processing time almost in half.

Similar to raw outputs, we can force `IndicatorFactory` to return the cache, so it can be used
in other calculations or even indicators. The clear advantage of this approach is that we don't
rely on some fixed set of parameter combinations any more, but on the values of each parameter,
which gives us more granularity in managing performance.

```pycon
>>> cache = MyInd.run(price, 2, np.arange(2, 1000), return_cache=True)

>>> %timeit MyInd.run(price, np.arange(2, 1000), np.arange(2, 1000), use_cache=cache)
30.1 ms ± 2 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

## Custom properties and methods

Use `custom_output_props` argument when constructing an indicator to define lazy outputs -
只有在显式调用时才会处理的输出。它们将成为缓存属性，与常规输出相比，
它们可以有任意形状。例如，让我们附加一个计算移动平均与价格之间距离的属性。

```pycon
>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     param_names=['window'],
...     output_names=['ma'],
...     custom_output_props=dict(distance=lambda self: (self.price - self.ma) / self.ma)
... ).from_apply_func(vbt.nb.rolling_mean_nb)

>>> MyInd.run(price, [2, 3]).distance
custom_window                   2                   3
                      a         b         a         b
2020-01-01          NaN       NaN       NaN       NaN
2020-01-02     0.333333 -0.111111       NaN       NaN
2020-01-03     0.200000 -0.142857  0.500000 -0.250000
2020-01-04     0.142857 -0.200000  0.333333 -0.333333
2020-01-05     0.111111 -0.333333  0.250000 -0.500000
```

Another way of defining own properties and methods is subclassing:

```pycon
>>> class MyIndExtended(MyInd):
...     def plot(self, column=None, **kwargs):
...         self_col = self.select_one(column=column, group_by=False)
...         return self.ma.vbt.plot(**kwargs)

>>> MyIndExtended.run(price, [2, 3])[(2, 'a')].plot()
```

![](/assets/images/MyInd_plot.svg)

## Helper properties and methods

For all in `input_names`, `in_output_names`, `output_names`, and `custom_output_props`,
`IndicatorFactory` will create a bunch of comparison and combination methods, such as for generating signals.
What kind of methods are created can be regulated using `dtype` in the `attr_settings` dictionary.

```pycon
>>> from collections import namedtuple

>>> MyEnum = namedtuple('MyEnum', ['one', 'two'])(0, 1)

>>> def apply_func_nb(price):
...     out_float = np.empty(price.shape, dtype=np.float64)
...     out_bool = np.empty(price.shape, dtype=np.bool_)
...     out_enum = np.empty(price.shape, dtype=np.int64)
...     return out_float, out_bool, out_enum

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     output_names=['out_float', 'out_bool', 'out_enum'],
...     attr_settings=dict(
...         out_float=dict(dtype=np.float64),
...         out_bool=dict(dtype=np.bool_),
...         out_enum=dict(dtype=MyEnum)
... )).from_apply_func(apply_func_nb)

>>> myind = MyInd.run(price)
>>> dir(myind)
[
    ...
    'out_bool',
    'out_bool_and',
    'out_bool_or',
    'out_bool_stats',
    'out_bool_xor',
    'out_enum',
    'out_enum_readable',
    'out_enum_stats',
    'out_float',
    'out_float_above',
    'out_float_below',
    'out_float_equal',
    'out_float_stats',
    ...
    'price',
    'price_above',
    'price_below',
    'price_equal',
    'price_stats',
    ...
]
```

Each of these methods and properties are created for sheer convenience: to easily combine
boolean arrays using logical rules and to compare numeric arrays. All operations are done
strictly using NumPy. Another advantage is utilization of vectorbt's own broadcasting, such
that one can combine inputs and outputs with an arbitrary array-like object, given their
shapes can broadcast together.

我们还可以通过将它们作为元组/列表传递来同时与多个对象进行比较：

```pycon
>>> myind.price_above([1.5, 2.5])
custom_price_above           1.5           2.5
                        a      b      a      b
2020-01-01          False   True  False   True
2020-01-02           True   True  False   True
2020-01-03           True   True   True   True
2020-01-04           True   True   True  False
2020-01-05           True  False   True  False
```

## Indexing

`IndicatorFactory` attaches pandas indexing to the indicator class thanks to
`vectorbt.base.array_wrapper.ArrayWrapper`. Supported are `iloc`, `loc`,
`*param_name*_loc`, `xs`, and `__getitem__`.

This makes possible accessing rows and columns by labels, integer positions, and parameters.

```pycon
>>> ma = vbt.MA.run(price, [2, 3])

>>> ma[(2, 'b')]
<vectorbt.indicators.basic.MA at 0x7fe4d10ddcc0>

>>> ma[(2, 'b')].ma
2020-01-01    NaN
2020-01-02    4.5
2020-01-03    3.5
2020-01-04    2.5
2020-01-05    1.5
Name: (2, b), dtype: float64

>>> ma.window_loc[2].ma
              a    b
2020-01-01  NaN  NaN
2020-01-02  1.5  4.5
2020-01-03  2.5  3.5
2020-01-04  3.5  2.5
2020-01-05  4.5  1.5
```

## TA-Lib

指标工厂还提供类方法 `IndicatorFactory.from_talib`，
可用于封装TA-Lib中的任何函数。它会自动填充所有必要信息，
如输入、参数和输出名称。

## Stats

!!! hint
    See `vectorbt.generic.stats_builder.StatsBuilderMixin.stats`.

We can attach metrics to any new indicator class:

```pycon
>>> @njit
... def apply_func_nb(price):
...     return price ** 2, price ** 3

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     output_names=['out1', 'out2'],
...     metrics=dict(
...         sum_diff=dict(
...             calc_func=lambda self: self.out2.sum() - self.out1.sum()
...         )
...     )
... ).from_apply_func(
...     apply_func_nb
... )

>>> myind = MyInd.run(price)
>>> myind.stats(column='a')
sum_diff    170.0
Name: a, dtype: float64
```

## Plots

!!! hint
    See `vectorbt.generic.plots_builder.PlotsBuilderMixin.plots`.

Similarly to stats, we can attach subplots to any new indicator class:

```pycon
>>> @njit
... def apply_func_nb(price):
...     return price ** 2, price ** 3

>>> def plot_outputs(out1, out2, column=None, fig=None):
...     fig = out1[column].rename('out1').vbt.plot(fig=fig)
...     fig = out2[column].rename('out2').vbt.plot(fig=fig)

>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price'],
...     output_names=['out1', 'out2'],
...     subplots=dict(
...         plot_outputs=dict(
...             plot_func=plot_outputs,
...             resolve_out1=True,
...             resolve_out2=True
...         )
...     )
... ).from_apply_func(
...     apply_func_nb
... )

>>> myind = MyInd.run(price)
>>> myind.plots(column='a')
```

![](/assets/images/IndicatorFactory_plots.svg)
"""

# ================================================================================
# 导入模块部分 - 引入所需的依赖库和工具函数
# ================================================================================

# 标准库导入
import inspect  # 用于函数签名检查和参数解析
import itertools  # 用于参数组合的迭代器操作
import warnings  # 用于警告信息处理
from collections import Counter  # 用于计数器功能
from collections import OrderedDict  # 用于有序字典
from datetime import datetime, timedelta  # 用于日期时间处理
from types import ModuleType  # 用于模块类型检查

# 第三方库导入
import numpy as np  # NumPy数组操作
import pandas as pd  # Pandas数据结构和操作
from numba import njit  # Numba JIT编译器
from numba.typed import List  # Numba类型化列表

# vectorbt内部模块导入
from vectorbt import _typing as tp  # 类型定义
from vectorbt.base import index_fns, reshape_fns, combine_fns  # 基础功能函数
from vectorbt.base.array_wrapper import ArrayWrapper, Wrapping  # 数组包装器
from vectorbt.base.indexing import build_param_indexer  # 参数索引构建器
from vectorbt.generic import nb as generic_nb  # 通用Numba函数
from vectorbt.generic.accessors import BaseAccessor  # 基础访问器
from vectorbt.generic.plots_builder import PlotsBuilderMixin  # 绘图构建器混合类
from vectorbt.generic.stats_builder import StatsBuilderMixin  # 统计构建器混合类
from vectorbt.utils import checks  # 检查工具函数
from vectorbt.utils.config import merge_dicts, resolve_dict, Config, Default  # 配置处理
from vectorbt.utils.decorators import classproperty, cached_property  # 装饰器工具
from vectorbt.utils.docs import to_doc  # 文档处理工具
from vectorbt.utils.enum_ import map_enum_fields  # 枚举字段映射
from vectorbt.utils.mapping import to_mapping, apply_mapping  # 映射处理工具
from vectorbt.utils.params import to_typed_list, broadcast_params, create_param_product  # 参数处理工具
from vectorbt.utils.random_ import set_seed  # 随机数种子设置

# 可选依赖库导入 - ta技术分析库
try:
    from ta.utils import IndicatorMixin as IndicatorMixinT  # ta库的指标混合类
except ImportError:
    IndicatorMixinT = tp.Any  # 如果导入失败，使用任意类型


def params_to_list(params: tp.Params, is_tuple: bool, is_array_like: bool) -> list:
    """
    将参数转换为列表格式
    
    功能说明：
    - 根据参数的类型标志，将各种形式的参数统一转换为列表格式
    - 支持处理元组、数组等不同的参数类型
    - 为后续的参数处理和广播操作提供统一的数据格式
    
    参数：
        params (tp.Params): 需要转换的参数，可以是标量、列表、元组或数组
        is_tuple (bool): 是否保持元组格式的标志
        is_array_like (bool): 是否将数组视为单个参数的标志
    
    返回值：
        list: 转换后的参数列表
    
    使用示例：
        >>> params_to_list(5, False, False)  # 标量转换为列表
        [5]
        >>> params_to_list([1, 2, 3], False, False)  # 列表保持不变
        [1, 2, 3]
        >>> params_to_list((1, 2), True, False)  # 元组根据is_tuple标志处理
        [1, 2]
    """
    # 定义需要检查的类型列表
    check_against = [list, List]  # 默认包含list和numba.typed.List
    
    # 如果不需要保持元组格式，将tuple添加到检查列表中
    if not is_tuple:
        check_against.append(tuple)
    
    # 如果不将数组视为单个参数，将numpy数组添加到检查列表中
    if not is_array_like:
        check_against.append(np.ndarray)
    
    # 如果参数是检查列表中的类型之一，转换为列表
    if isinstance(params, tuple(check_against)):
        new_params = list(params)  # 转换为列表
    else:
        new_params = [params]  # 将单个参数包装为列表
    
    return new_params


def prepare_params(param_list: tp.Sequence[tp.Params],
                   param_settings: tp.KwargsLikeSequence = None,
                   input_shape: tp.Optional[tp.Shape] = None,
                   to_2d: bool = False) -> tp.List[tp.List]:
    """
    预处理参数列表，进行类型转换、广播和形状调整
    
    功能说明：
    - 对参数列表中的每个参数进行预处理，包括类型转换、映射应用和形状广播
    - 支持将参数广播到输入数据的形状或其特定轴
    - 处理参数的数据类型映射（包括枚举和命名元组）
    - 支持灵活的广播配置和二维形状转换
    
    参数：
        param_list (tp.Sequence[tp.Params]): 待处理的参数列表
        param_settings (tp.KwargsLikeSequence): 参数设置，包含每个参数的配置选项
        input_shape (tp.Optional[tp.Shape]): 输入数据的形状，用于参数广播
        to_2d (bool): 是否将参数转换为二维形状
    
    返回值：
        tp.List[tp.List]: 处理后的参数列表，每个参数都已经过预处理
    
    使用示例：
        >>> param_list = [10, [20, 30]]
        >>> param_settings = [{'bc_to_input': True}, {'is_array_like': True}]
        >>> prepare_params(param_list, param_settings, input_shape=(100, 2))
        # 返回经过广播和处理的参数列表
    """
    new_param_list = []  # 存储处理后的参数列表
    
    # 遍历每个参数进行预处理
    for i, params in enumerate(param_list):
        # 解析当前参数的配置设置
        _param_settings = resolve_dict(param_settings, i=i)
        
        # 获取参数配置选项
        is_tuple = _param_settings.get('is_tuple', False)  # 是否保持元组格式
        dtype = _param_settings.get('dtype', None)  # 数据类型或映射
        
        # 如果指定了数据类型映射，应用类型转换
        if checks.is_mapping_like(dtype):
            if checks.is_namedtuple(dtype):
                params = map_enum_fields(params, dtype)  # 映射枚举字段
            else:
                params = apply_mapping(params, dtype)  # 应用常规映射
        
        # 获取其他配置选项
        is_array_like = _param_settings.get('is_array_like', False)  # 是否将数组视为单个参数
        bc_to_input = _param_settings.get('bc_to_input', False)  # 是否广播到输入形状
        broadcast_kwargs = _param_settings.get('broadcast_kwargs', dict(require_kwargs=dict(requirements='W')))  # 广播配置
        
        # 将参数转换为列表格式
        new_params = params_to_list(params, is_tuple, is_array_like)
        
        # 如果需要广播到输入形状
        if bc_to_input is not False:
            # 检查广播的前提条件
            if is_tuple:
                raise ValueError("Cannot broadcast to input if tuple")  # 元组不能广播
            if input_shape is None:
                raise ValueError("Cannot broadcast to input if input shape is unknown. Pass input_shape.")  # 输入形状未知
            
            # 确定目标广播形状
            if bc_to_input is True:
                to_shape = input_shape  # 广播到完整输入形状
            else:
                checks.assert_in(bc_to_input, (0, 1))  # 只能是0或1
                # 广播到特定轴
                if bc_to_input == 0:
                    to_shape = (input_shape[0],)  # 广播到第0轴（行）
                else:
                    to_shape = (input_shape[1],) if len(input_shape) > 1 else (1,)  # 广播到第1轴（列）
            
            # 执行广播操作
            _new_params = reshape_fns.broadcast(
                *new_params,
                to_shape=to_shape,
                **broadcast_kwargs
            )
            
            # 确保结果是列表格式
            if len(new_params) == 1:
                _new_params = [_new_params]
            else:
                _new_params = list(_new_params)
            
            # 如果需要转换为二维并且广播到完整输入形状
            if to_2d and bc_to_input is True:
                # 如果输入需要转换为2D，参数也需要相应转换
                # 但仅适用于完全匹配输入的参数（非原始格式）
                __new_params = _new_params.copy()
                for j, param in enumerate(__new_params):
                    keep_raw = broadcast_kwargs.get('keep_raw', False)
                    # 如果不保持原始格式，转换为2D
                    if keep_raw is False or (isinstance(keep_raw, (tuple, list)) and not keep_raw[j]):
                        __new_params[j] = reshape_fns.to_2d(param)
                new_params = __new_params
            else:
                new_params = _new_params
        
        # 将处理后的参数添加到结果列表
        new_param_list.append(new_params)
    
    return new_param_list


def build_columns(param_list: tp.Sequence[tp.Sequence[tp.Param]],
                  input_columns: tp.IndexLike,
                  level_names: tp.Optional[tp.Sequence[str]] = None,
                  hide_levels: tp.Optional[tp.Sequence[int]] = None,
                  param_settings: tp.KwargsLikeSequence = None,
                  per_column: bool = False,
                  ignore_default: bool = False,
                  **kwargs) -> tp.Tuple[tp.List[tp.Index], tp.Index]:
    """
    构建多层级列索引，将参数值作为新的列级别叠加在输入列之上
    
    功能说明：
    - 为参数列表中的每个参数创建新的列级别，值为参数值
    - 将参数级别叠加在输入列索引之上，形成多层级索引结构
    - 支持按列处理参数和隐藏特定层级
    - 用于创建指标输出的多维列索引结构
    
    参数：
        param_list (tp.Sequence[tp.Sequence[tp.Param]]): 参数列表，每个参数包含多个值
        input_columns (tp.IndexLike): 输入数据的列索引
        level_names (tp.Optional[tp.Sequence[str]]): 参数级别的名称列表
        hide_levels (tp.Optional[tp.Sequence[int]]): 需要隐藏的级别索引列表
        param_settings (tp.KwargsLikeSequence): 参数设置，包含每个参数的配置选项
        per_column (bool): 是否按列处理参数
        ignore_default (bool): 是否忽略默认值
        **kwargs: 传递给索引堆叠函数的额外参数
    
    返回值：
        tp.Tuple[tp.List[tp.Index], tp.Index]: 参数索引列表和新的列索引
    
    使用示例：
        >>> param_list = [[10, 20], [0.5, 1.0]]
        >>> input_columns = pd.Index(['A', 'B'])
        >>> level_names = ['window', 'alpha']
        >>> param_indexes, new_columns = build_columns(param_list, input_columns, level_names)
        # 返回多层级列索引，包含参数值和原始列名
    """
    # 验证参数和级别名称的长度一致性
    if level_names is not None:
        checks.assert_len_equal(param_list, level_names)
    
    # 初始化隐藏级别列表
    if hide_levels is None:
        hide_levels = []
    
    # 确保输入列为索引对象
    input_columns = index_fns.to_any_index(input_columns)
    
    # 存储参数索引和显示的参数索引
    param_indexes = []  # 所有参数索引
    shown_param_indexes = []  # 需要显示的参数索引
    
    # 遍历每个参数列表，构建参数索引
    for i in range(len(param_list)):
        params = param_list[i]  # 当前参数的值列表
        level_name = None  # 当前级别名称
        
        # 获取级别名称
        if level_names is not None:
            level_name = level_names[i]
        
        # 根据处理模式构建参数索引
        if per_column:
            # 按列处理模式：直接从参数值创建索引
            param_index = index_fns.index_from_values(params, name=level_name)
        else:
            # 解析当前参数的设置
            _param_settings = resolve_dict(param_settings, i=i)
            _per_column = _param_settings.get('per_column', False)
            
            if _per_column:
                # 单个参数的按列处理
                param_index = None
                for param in params:
                    # 将参数广播到输入列的长度
                    bc_param = np.broadcast_to(param, (len(input_columns),))
                    _param_index = index_fns.index_from_values(bc_param, name=level_name)
                    
                    # 合并参数索引
                    if param_index is None:
                        param_index = _param_index
                    else:
                        param_index = param_index.append(_param_index)
                
                # 处理灵活的按列参数情况
                if len(param_index) == 1 and len(input_columns) > 1:
                    # 当使用灵活的按列参数时，重复索引
                    param_index = index_fns.repeat_index(
                        param_index,
                        len(input_columns),
                        ignore_default=ignore_default
                    )
            else:
                # 标准处理模式：从参数值创建索引并重复
                param_index = index_fns.index_from_values(param_list[i], name=level_name)
                param_index = index_fns.repeat_index(
                    param_index,
                    len(input_columns),
                    ignore_default=ignore_default
                )
        
        # 添加到参数索引列表
        param_indexes.append(param_index)
        
        # 如果不在隐藏级别中，添加到显示列表
        if i not in hide_levels:
            shown_param_indexes.append(param_index)
    
    # 构建最终的列索引
    if len(shown_param_indexes) > 0:
        if not per_column:
            # 计算参数值的数量并平铺输入列
            n_param_values = len(param_list[0]) if len(param_list) > 0 else 1
            input_columns = index_fns.tile_index(
                input_columns,
                n_param_values,
                ignore_default=ignore_default
            )
        
        # 堆叠所有显示的参数索引和输入列
        stacked_columns = index_fns.stack_indexes([*shown_param_indexes, input_columns], **kwargs)
        return param_indexes, stacked_columns
    
    # 如果没有显示的参数索引，返回原始输入列
    return param_indexes, input_columns


# ================================================================================
# 类型定义 - 定义指标工厂中使用的各种类型别名
# ================================================================================

# 缓存输出类型：用于存储计算结果的缓存对象
CacheOutputT = tp.Any

# 原始输出类型：包含输出数组列表、参数组合列表、列数和其他对象的元组
RawOutputT = tp.Tuple[tp.List[tp.Array2d], tp.List[tp.Tuple[tp.Param, ...]], int, tp.List[tp.Any]]

# 输入列表类型：二维数组的列表，每个数组代表一个输入
InputListT = tp.List[tp.Array2d]

# 输入映射器类型：可选的一维数组，用于输入映射
InputMapperT = tp.Optional[tp.Array1d]

# 就地输出列表类型：用于就地修改的二维数组列表
InOutputListT = tp.List[tp.Array2d]

# 输出列表类型：二维数组的列表，每个数组代表一个输出
OutputListT = tp.List[tp.Array2d]

# 参数列表类型：嵌套列表，外层列表包含不同的参数，内层列表包含参数的不同值
ParamListT = tp.List[tp.List[tp.Param]]

# 映射器列表类型：索引对象的列表，用于参数映射
MapperListT = tp.List[tp.Index]

# 其他对象列表类型：任意对象的列表，用于存储额外的输出
OtherListT = tp.List[tp.Any]

# 管道输出类型：包含所有管道输出组件的元组
PipelineOutputT = tp.Tuple[
    ArrayWrapper,    # 数组包装器，用于处理索引和列信息
    InputListT,      # 输入列表
    InputMapperT,    # 输入映射器
    InOutputListT,   # 就地输出列表
    OutputListT,     # 输出列表
    ParamListT,      # 参数列表
    MapperListT,     # 映射器列表
    OtherListT       # 其他对象列表
]


def run_pipeline(
        num_ret_outputs: int,
        custom_func: tp.Callable,
        *args,
        require_input_shape: bool = False,
        input_shape: tp.Optional[tp.RelaxedShape] = None,
        input_index: tp.Optional[tp.IndexLike] = None,
        input_columns: tp.Optional[tp.IndexLike] = None,
        input_list: tp.Optional[tp.Sequence[tp.ArrayLike]] = None,
        in_output_list: tp.Optional[tp.Sequence[tp.ArrayLike]] = None,
        in_output_settings: tp.KwargsLikeSequence = None,
        broadcast_kwargs: tp.KwargsLike = None,
        param_list: tp.Optional[tp.Sequence[tp.Param]] = None,
        param_product: bool = False,
        param_settings: tp.KwargsLikeSequence = None,
        run_unique: bool = False,
        silence_warnings: bool = False,
        per_column: bool = False,
        pass_col: bool = False,
        keep_pd: bool = False,
        to_2d: bool = True,
        as_lists: bool = False,
        pass_input_shape: bool = False,
        pass_flex_2d: bool = False,
        level_names: tp.Optional[tp.Sequence[str]] = None,
        hide_levels: tp.Optional[tp.Sequence[int]] = None,
        stacking_kwargs: tp.KwargsLike = None,
        return_raw: bool = False,
        use_raw: tp.Optional[RawOutputT] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        seed: tp.Optional[int] = None,
        **kwargs) -> tp.Union[CacheOutputT, RawOutputT, PipelineOutputT]:
    """
    运行指标计算的核心管道函数，被指标工厂类使用
    
    功能说明：
    这是vectorbt指标系统的核心函数，用于标准化指标计算流程。它负责：
    - 处理和广播输入数据
    - 管理参数组合和参数产品
    - 执行自定义计算函数
    - 处理输出结果和索引
    - 支持缓存、原始输出和各种优化
    
    主要处理流程：
    1. 输入数据预处理和广播
    2. 参数处理和组合生成
    3. 执行自定义计算函数
    4. 输出结果后处理
    5. 构建最终的数据结构
    
    参数说明：
        num_ret_outputs (int): 自定义函数返回的输出数组数量
        custom_func (tp.Callable): 自定义计算函数
        *args: 传递给自定义函数的位置参数
        require_input_shape (bool): 是否需要输入形状，如果为True则会传递input_shape并检查
        input_shape (tp.Optional[tp.RelaxedShape]): 输入数据的广播目标形状
        input_index (tp.Optional[tp.IndexLike]): 输入数据的索引
        input_columns (tp.Optional[tp.IndexLike]): 输入数据的列索引
        input_list (tp.Optional[tp.Sequence[tp.ArrayLike]]): 输入数组列表
        in_output_list (tp.Optional[tp.Sequence[tp.ArrayLike]]): 就地输出数组列表
        in_output_settings (tp.KwargsLikeSequence): 就地输出的设置
        broadcast_kwargs (tp.KwargsLike): 广播函数的关键字参数
        param_list (tp.Optional[tp.Sequence[tp.Param]]): 参数列表
        param_product (bool): 是否构建参数的笛卡尔积
        param_settings (tp.KwargsLikeSequence): 参数设置
        run_unique (bool): 是否只运行唯一的参数组合
        silence_warnings (bool): 是否隐藏警告
        per_column (bool): 是否按列分别处理
        pass_col (bool): 是否传递列索引到自定义函数
        keep_pd (bool): 是否保持pandas对象格式
        to_2d (bool): 是否将输入重塑为二维数组
        as_lists (bool): 是否以列表形式传递输入和参数
        pass_input_shape (bool): 是否传递输入形状到自定义函数
        pass_flex_2d (bool): 是否传递flex_2d参数到自定义函数
        level_names (tp.Optional[tp.Sequence[str]]): 参数级别名称列表
        hide_levels (tp.Optional[tp.Sequence[int]]): 需要隐藏的级别索引
        stacking_kwargs (tp.KwargsLike): 索引堆叠的关键字参数
        return_raw (bool): 是否返回原始输出
        use_raw (tp.Optional[RawOutputT]): 使用原始结果而不是运行计算
        wrapper_kwargs (tp.KwargsLike): 数组包装器的关键字参数
        seed (tp.Optional[int]): 随机种子
        **kwargs: 传递给自定义函数的关键字参数
    
    返回值：
        tp.Union[CacheOutputT, RawOutputT, PipelineOutputT]: 
        - 如果return_raw=True，返回原始输出
        - 否则返回管道输出，包含数组包装器、输入列表、输出列表、参数列表等
    
    使用示例：
        >>> def custom_ma(close, window):
        ...     return np.convolve(close, np.ones(window)/window, mode='valid')
        >>> 
        >>> result = run_pipeline(
        ...     num_ret_outputs=1,
        ...     custom_func=custom_ma,
        ...     input_list=[price_data],
        ...     param_list=[10, 20]
        ... )
        # 返回移动平均线计算结果
    """
    
    # ================================================================================
    # 核心管道函数实现开始
    # ================================================================================
    
    # 设置随机种子以确保结果可重现
    if seed is not None:
        set_seed(seed)
    
    # 验证必需的输入形状参数
    if require_input_shape:
        if input_shape is None:
            raise ValueError("input_shape is required")
        pass_input_shape = True
    
    # 处理参数产品配置
    if param_product:
        if param_list is None:
            param_list = []
        param_list = create_param_product(param_list)
    
    # 处理原始输出使用模式
    if use_raw is not None:
        return use_raw
    
    # 设置随机种子以确保结果可重现
    if seed is not None:
        set_seed(seed)
    
    # ================================================================================
    # 核心管道函数实现开始
    # ================================================================================
    
    # 验证必需的输入形状参数
    if require_input_shape:
        checks.assert_not_none(input_shape)
        pass_input_shape = True
    
    # 处理输入索引和列
    if input_index is not None:
        input_index = index_fns.to_any_index(input_index)
    if input_columns is not None:
        input_columns = index_fns.to_any_index(input_columns)
    
    # 处理输入列表
    if input_list is None:
        input_list = []
    else:
        input_list = list(input_list)
    
    # 处理就地输出列表
    if in_output_list is None:
        in_output_list = []
    else:
        in_output_list = list(in_output_list)
    
    # 处理就地输出设置
    if in_output_settings is None:
        in_output_settings = {}
    checks.assert_dict_sequence_valid(in_output_settings, ['dtype'])
    
    # 处理广播参数
    if broadcast_kwargs is None:
        broadcast_kwargs = {}
    
    # 处理参数列表
    if param_list is None:
        param_list = []
    else:
        param_list = list(param_list)
    
    # 处理参数设置
    if param_settings is None:
        param_settings = {}
    checks.assert_dict_sequence_valid(param_settings, [
        'dtype',
        'is_tuple',
        'is_array_like',
        'bc_to_input',
        'broadcast_kwargs',
        'per_column'
    ])
    
    # 处理隐藏级别和其他参数
    if hide_levels is None:
        hide_levels = []
    if stacking_kwargs is None:
        stacking_kwargs = {}
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    
    # 检查pandas对象与Numba函数的兼容性
    if keep_pd and checks.is_numba_func(custom_func):
        raise ValueError("Cannot pass pandas objects to a Numba-compiled custom_func. Set keep_pd to False.")
    
    # 处理就地输出索引
    in_output_idxs = [i for i, x in enumerate(in_output_list) if x is not None]
    if len(in_output_idxs) > 0:
        # 就地输出应该与输入一起广播
        input_list += [in_output_list[i] for i in in_output_idxs]
    
    # 处理输入列表广播
    if len(input_list) > 0:
        # 广播输入数组
        # 如果提供了input_shape，将所有输入广播到此形状
        broadcast_kwargs = merge_dicts(dict(
            to_shape=input_shape,
            index_from=input_index,
            columns_from=input_columns,
            require_kwargs=dict(requirements='W')
        ), broadcast_kwargs)
        bc_input_list, input_shape, input_index, input_columns = reshape_fns.broadcast(
            *input_list,
            return_meta=True,
            **broadcast_kwargs
        )
        if input_index is None:
            input_index = pd.RangeIndex(start=0, step=1, stop=input_shape[0])
        if input_columns is None:
            input_columns = pd.RangeIndex(start=0, step=1, stop=input_shape[1] if len(input_shape) > 1 else 1)
        if len(input_list) == 1:
            bc_input_list = (bc_input_list,)
        input_list = list(map(np.asarray, bc_input_list))
    
    # 分离输入和就地输出
    if len(in_output_idxs) > 0:
        # 分离输入和就地输出
        in_output_list = input_list[-len(in_output_idxs):]
        input_list = input_list[:-len(in_output_idxs)]

    # 重塑输入形状
    if input_shape is not None and not isinstance(input_shape, tuple):
        input_shape = (input_shape,)
    
    # 保留原始input_shape用于per_column=True的情况
    orig_input_shape = input_shape
    orig_input_shape_2d = input_shape
    if input_shape is not None:
        orig_input_shape_2d = input_shape if len(input_shape) > 1 else (input_shape[0], 1)
    
    # 处理按列模式
    if per_column:
        # input_shape现在是一列的大小
        if input_shape is None:
            raise ValueError("input_shape is required when per_column=True")
        input_shape = (input_shape[0],)
    
    input_shape_ready = input_shape
    input_shape_2d = input_shape
    if input_shape is not None:
        input_shape_2d = input_shape if len(input_shape) > 1 else (input_shape[0], 1)
    
    # 处理二维转换
    if to_2d:
        if input_shape is not None:
            input_shape_ready = input_shape_2d  # 为custom_func准备

    # 预处理参数
    # 注意：使用input_shape而不是input_shape_ready，因为参数应该按照与输入相同的规则广播
    param_list = prepare_params(param_list, param_settings, input_shape=input_shape, to_2d=to_2d)
    
    # 处理多个参数的情况
    if len(param_list) > 1:
        if level_names is not None:
            # 检查级别名称
            checks.assert_len_equal(param_list, level_names)
            # 列应该不包含指定的级别名称
            if input_columns is not None:
                for level_name in level_names:
                    if level_name is not None:
                        checks.assert_level_not_exists(input_columns, level_name)
        if param_product:
            # 从所有参数创建笛卡尔积
            param_list = create_param_product(param_list)
    
    # 广播参数
    if len(param_list) > 0:
        # 广播使每个数组具有相同的长度
        if per_column:
            # 参数数量应该与拆分前的列数匹配
            param_list = broadcast_params(param_list, to_n=orig_input_shape_2d[1])
        else:
            param_list = broadcast_params(param_list)
    
    n_param_values = len(param_list[0]) if len(param_list) > 0 else 1
    use_run_unique = False
    param_list_unique = param_list
    
    # 处理运行唯一参数组合
    if not per_column and run_unique:
        try:
            # 尝试获取所有唯一的参数组合
            param_tuples = list(zip(*param_list))
            unique_param_tuples = list(OrderedDict.fromkeys(param_tuples).keys())
            if len(unique_param_tuples) < len(param_tuples):
                param_list_unique = list(map(list, zip(*unique_param_tuples)))
                use_run_unique = True
        except:
            pass
    
    # 为Numba函数准备参数
    if checks.is_numba_func(custom_func):
        # Numba无法处理无类型列表
        param_list_ready = [to_typed_list(params) for params in param_list_unique]
    else:
        param_list_ready = param_list_unique
    
    n_unique_param_values = len(param_list_unique[0]) if len(param_list_unique) > 0 else 1

    # 准备输入数据（根据per_column模式）
    if per_column:
        # 将每个输入拆分为Series/1维数组，每列一个
        input_list_ready = []
        for input in input_list:
            input_2d = reshape_fns.to_2d(input)
            col_inputs = []
            for i in range(input_2d.shape[1]):
                if to_2d:
                    col_input = input_2d[:, [i]]
                else:
                    col_input = input_2d[:, i]
                if keep_pd:
                    # 保持为pandas对象
                    col_input = ArrayWrapper(input_index, input_columns[[i]], col_input.ndim).wrap(col_input)
                col_inputs.append(col_input)
            input_list_ready.append(col_inputs)
    else:
        input_list_ready = []
        for input in input_list:
            new_input = input
            if to_2d:
                new_input = reshape_fns.to_2d(input)
            if keep_pd:
                # 保持为pandas对象
                new_input = ArrayWrapper(input_index, input_columns, new_input.ndim).wrap(new_input)
            input_list_ready.append(new_input)

    # 准备就地输出
    in_output_list_ready = []
    j = 0
    for i in range(len(in_output_list)):
        if input_shape_2d is None:
            raise ValueError("input_shape is required when using in-place outputs")
        if i in in_output_idxs:
            # 此就地输出已经与输入一起广播
            in_output_wide = np.require(in_output_list[j], requirements='W')
            if not per_column:
                # 每个参数组合一个
                in_output_wide = reshape_fns.tile(in_output_wide, n_unique_param_values, axis=1)
            j += 1
        else:
            # 此就地输出尚未提供，因此创建空的
            _in_output_settings = in_output_settings if isinstance(in_output_settings, dict) else in_output_settings[i]
            dtype = _in_output_settings.get('dtype', None)
            in_output_shape = (input_shape_2d[0], input_shape_2d[1] * n_unique_param_values)
            in_output_wide = np.empty(in_output_shape, dtype=dtype)
        in_output_list[i] = in_output_wide
        in_outputs = []
        # 将每个就地输出拆分为块，每个块为输入形状，并添加到列表中
        for k in range(n_unique_param_values):
            in_output = in_output_wide[:, k * input_shape_2d[1]: (k + 1) * input_shape_2d[1]]
            if len(input_shape_ready) == 1:
                in_output = in_output[:, 0]
            if keep_pd:
                if per_column:
                    in_output = ArrayWrapper(input_index, input_columns[[k]], in_output.ndim).wrap(in_output)
                else:
                    in_output = ArrayWrapper(input_index, input_columns, in_output.ndim).wrap(in_output)
            in_outputs.append(in_output)
        in_output_list_ready.append(in_outputs)
    
    # 为Numba函数准备就地输出
    if checks.is_numba_func(custom_func):
        # Numba无法处理无类型列表
        in_output_list_ready = [to_typed_list(in_outputs) for in_outputs in in_output_list_ready]
        in_output_settings = {}
    checks.assert_dict_sequence_valid(in_output_settings, ['dtype'])
    if broadcast_kwargs is None:
        broadcast_kwargs = {}
    if param_list is None:
        param_list = []
    else:
        param_list = list(param_list)
    if param_settings is None:
        param_settings = {}
    checks.assert_dict_sequence_valid(param_settings, [
        'dtype',
        'is_tuple',
        'is_array_like',
        'bc_to_input',
        'broadcast_kwargs',
        'per_column'
    ])
    if hide_levels is None:
        hide_levels = []
    if stacking_kwargs is None:
        stacking_kwargs = {}
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    if keep_pd and checks.is_numba_func(custom_func):
        raise ValueError("Cannot pass pandas objects to a Numba-compiled custom_func. Set keep_pd to False.")

    in_output_idxs = [i for i, x in enumerate(in_output_list) if x is not None]
    if len(in_output_idxs) > 0:
        # In-place outputs should broadcast together with inputs
        input_list += [in_output_list[i] for i in in_output_idxs]
    if len(input_list) > 0:
        # Broadcast inputs
        # If input_shape is provided, will broadcast all inputs to this shape
        broadcast_kwargs = merge_dicts(dict(
            to_shape=input_shape,
            index_from=input_index,
            columns_from=input_columns,
            require_kwargs=dict(requirements='W')
        ), broadcast_kwargs)
        bc_input_list, input_shape, input_index, input_columns = reshape_fns.broadcast(
            *input_list,
            return_meta=True,
            **broadcast_kwargs
        )
        if input_index is None:
            input_index = pd.RangeIndex(start=0, step=1, stop=input_shape[0])
        if input_columns is None:
            input_columns = pd.RangeIndex(start=0, step=1, stop=input_shape[1] if len(input_shape) > 1 else 1)
        if len(input_list) == 1:
            bc_input_list = (bc_input_list,)
        input_list = list(map(np.asarray, bc_input_list))
    if len(in_output_idxs) > 0:
        # Separate inputs and in-place outputs
        in_output_list = input_list[-len(in_output_idxs):]
        input_list = input_list[:-len(in_output_idxs)]

    # Reshape input shape
    if input_shape is not None and not isinstance(input_shape, tuple):
        input_shape = (input_shape,)
    # Keep original input_shape for per_column=True
    orig_input_shape = input_shape
    orig_input_shape_2d = input_shape
    if input_shape is not None:
        orig_input_shape_2d = input_shape if len(input_shape) > 1 else (input_shape[0], 1)
    if per_column:
        # input_shape is now the size of one column
        if input_shape is None:
            raise ValueError("input_shape is required when per_column=True")
        input_shape = (input_shape[0],)
    input_shape_ready = input_shape
    input_shape_2d = input_shape
    if input_shape is not None:
        input_shape_2d = input_shape if len(input_shape) > 1 else (input_shape[0], 1)
    if to_2d:
        if input_shape is not None:
            input_shape_ready = input_shape_2d  # ready for custom_func

    # Prepare parameters
    # NOTE: input_shape instead of input_shape_ready since parameters should
    # broadcast by the same rules as inputs
    param_list = prepare_params(param_list, param_settings, input_shape=input_shape, to_2d=to_2d)
    if len(param_list) > 1:
        if level_names is not None:
            # Check level names
            checks.assert_len_equal(param_list, level_names)
            # Columns should be free of the specified level names
            if input_columns is not None:
                for level_name in level_names:
                    if level_name is not None:
                        checks.assert_level_not_exists(input_columns, level_name)
        if param_product:
            # Make Cartesian product out of all params
            param_list = create_param_product(param_list)
    if len(param_list) > 0:
        # Broadcast such that each array has the same length
        if per_column:
            # The number of parameters should match the number of columns before split
            param_list = broadcast_params(param_list, to_n=orig_input_shape_2d[1])
        else:
            param_list = broadcast_params(param_list)
    n_param_values = len(param_list[0]) if len(param_list) > 0 else 1
    use_run_unique = False
    param_list_unique = param_list
    if not per_column and run_unique:
        try:
            # Try to get all unique parameter combinations
            param_tuples = list(zip(*param_list))
            unique_param_tuples = list(OrderedDict.fromkeys(param_tuples).keys())
            if len(unique_param_tuples) < len(param_tuples):
                param_list_unique = list(map(list, zip(*unique_param_tuples)))
                use_run_unique = True
        except:
            pass
    if checks.is_numba_func(custom_func):
        # Numba can't stand untyped lists
        param_list_ready = [to_typed_list(params) for params in param_list_unique]
    else:
        param_list_ready = param_list_unique
    n_unique_param_values = len(param_list_unique[0]) if len(param_list_unique) > 0 else 1

    # Prepare inputs
    if per_column:
        # Split each input into Series/1-dim arrays, one per column
        input_list_ready = []
        for input in input_list:
            input_2d = reshape_fns.to_2d(input)
            col_inputs = []
            for i in range(input_2d.shape[1]):
                if to_2d:
                    col_input = input_2d[:, [i]]
                else:
                    col_input = input_2d[:, i]
                if keep_pd:
                    # Keep as pandas object
                    col_input = ArrayWrapper(input_index, input_columns[[i]], col_input.ndim).wrap(col_input)
                col_inputs.append(col_input)
            input_list_ready.append(col_inputs)
    else:
        input_list_ready = []
        for input in input_list:
            new_input = input
            if to_2d:
                new_input = reshape_fns.to_2d(input)
            if keep_pd:
                # Keep as pandas object
                new_input = ArrayWrapper(input_index, input_columns, new_input.ndim).wrap(new_input)
            input_list_ready.append(new_input)

    # Prepare in-place outputs
    in_output_list_ready = []
    j = 0
    for i in range(len(in_output_list)):
        if input_shape_2d is None:
            raise ValueError("input_shape is required when using in-place outputs")
        if i in in_output_idxs:
            # This in-place output has been already broadcast with inputs
            in_output_wide = np.require(in_output_list[j], requirements='W')
            if not per_column:
                # One per parameter combination
                in_output_wide = reshape_fns.tile(in_output_wide, n_unique_param_values, axis=1)
            j += 1
        else:
            # This in-place output hasn't been provided, so create empty
            _in_output_settings = in_output_settings if isinstance(in_output_settings, dict) else in_output_settings[i]
            dtype = _in_output_settings.get('dtype', None)
            in_output_shape = (input_shape_2d[0], input_shape_2d[1] * n_unique_param_values)
            in_output_wide = np.empty(in_output_shape, dtype=dtype)
        in_output_list[i] = in_output_wide
        in_outputs = []
        # Split each in-place output into chunks, each of input shape, and append to a list
        for i in range(n_unique_param_values):
            in_output = in_output_wide[:, i * input_shape_2d[1]: (i + 1) * input_shape_2d[1]]
            if len(input_shape_ready) == 1:
                in_output = in_output[:, 0]
            if keep_pd:
                if per_column:
                    in_output = ArrayWrapper(input_index, input_columns[[i]], in_output.ndim).wrap(in_output)
                else:
                    in_output = ArrayWrapper(input_index, input_columns, in_output.ndim).wrap(in_output)
            in_outputs.append(in_output)
        in_output_list_ready.append(in_outputs)
    if checks.is_numba_func(custom_func):
        # Numba can't stand untyped lists
        in_output_list_ready = [to_typed_list(in_outputs) for in_outputs in in_output_list_ready]

    def _use_raw(_raw):
        # Use raw results of previous run to build outputs
        _output_list, _param_map, _n_input_cols, _other_list = _raw
        idxs = np.array([_param_map.index(param_tuple) for param_tuple in zip(*param_list)])
        _output_list = [
            np.hstack([o[:, idx * _n_input_cols:(idx + 1) * _n_input_cols] for idx in idxs])
            for o in _output_list
        ]
        return _output_list, _param_map, _n_input_cols, _other_list

    # Get raw results
    if use_raw is not None:
        # Use raw results of previous run to build outputs
        output_list, param_map, n_input_cols, other_list = _use_raw(use_raw)
    else:
        # Prepare other arguments
        func_args = args
        func_kwargs = {}
        if pass_input_shape:
            func_kwargs['input_shape'] = input_shape_ready
        if pass_flex_2d:
            if input_shape is None:
                raise ValueError("Cannot determine flex_2d without inputs")
            func_kwargs['flex_2d'] = len(input_shape) == 2
        func_kwargs = merge_dicts(func_kwargs, kwargs)

        # Set seed
        if seed is not None:
            set_seed(seed)

        def _call_custom_func(_input_list_ready, _in_output_list_ready, _param_list_ready, *_func_args, **_func_kwargs):
            # Run the function
            if as_lists:
                if checks.is_numba_func(custom_func):
                    return custom_func(
                        tuple(_input_list_ready),
                        tuple(_in_output_list_ready),
                        tuple(_param_list_ready),
                        *_func_args, **_func_kwargs
                    )
                return custom_func(
                    _input_list_ready,
                    _in_output_list_ready,
                    _param_list_ready,
                    *_func_args, **_func_kwargs
                )
            return custom_func(
                *_input_list_ready,
                *_in_output_list_ready,
                *_param_list_ready,
                *_func_args, **_func_kwargs
            )

        if per_column:
            output = []
            for col in range(orig_input_shape_2d[1]):
                # Select the column of each input and in-place output, and the respective parameter combination
                _input_list_ready = []
                for _inputs in input_list_ready:
                    # Each input array is now one column wide
                    _input_list_ready.append(_inputs[col])

                _in_output_list_ready = []
                for _in_outputs in in_output_list_ready:
                    # Each in-output array is now one column wide
                    if isinstance(_in_outputs, List):
                        __in_outputs = List()
                    else:
                        __in_outputs = []
                    __in_outputs.append(_in_outputs[col])
                    _in_output_list_ready.append(__in_outputs)

                _param_list_ready = []
                for _params in param_list_ready:
                    # Each parameter list is now one element long
                    if isinstance(_params, List):
                        __params = List()
                    else:
                        __params = []
                    __params.append(_params[col])
                    _param_list_ready.append(__params)

                _func_args = func_args
                _func_kwargs = func_kwargs.copy()
                if 'use_cache' in func_kwargs:
                    use_cache = func_kwargs['use_cache']
                    if isinstance(use_cache, list) and len(use_cache) == orig_input_shape_2d[1]:
                        # Pass cache for this column
                        _func_kwargs['use_cache'] = func_kwargs['use_cache'][col]
                if pass_col:
                    _func_kwargs['col'] = col
                col_output = _call_custom_func(
                    _input_list_ready,
                    _in_output_list_ready,
                    _param_list_ready,
                    *_func_args,
                    **_func_kwargs
                )
                output.append(col_output)
        else:
            output = _call_custom_func(
                input_list_ready,
                in_output_list_ready,
                param_list_ready,
                *func_args,
                **func_kwargs
            )

        # Return cache
        if kwargs.get('return_cache', False):
            if use_run_unique and not silence_warnings:
                warnings.warn("Cache is produced by unique parameter "
                              "combinations when run_unique=True", stacklevel=2)
            return output

        def _split_output(output):
            # Post-process results
            if output is None:
                _output_list = []
                _other_list = []
            else:
                if isinstance(output, (tuple, list, List)):
                    _output_list = list(output)
                else:
                    _output_list = [output]
                # 其他输出应该在不进行后处理的情况下返回（例如cache_dict）
                if len(_output_list) > num_ret_outputs:
                    _other_list = _output_list[num_ret_outputs:]
                    if use_run_unique and not silence_warnings:
                        warnings.warn("Additional output objects are produced by unique parameter "
                                      "combinations when run_unique=True", stacklevel=2)
                else:
                    _other_list = []
                # Process only the num_ret_outputs outputs
                _output_list = _output_list[:num_ret_outputs]
            if len(_output_list) != num_ret_outputs:
                raise ValueError("Number of returned outputs other than expected")
            _output_list = list(map(lambda x: reshape_fns.to_2d_array(x), _output_list))
            return _output_list, _other_list

        if per_column:
            output_list = []
            other_list = []
            for _output in output:
                __output_list, __other_list = _split_output(_output)
                output_list.append(__output_list)
                if len(__other_list) > 0:
                    other_list.append(__other_list)
            # Concatenate each output (must be one column wide)
            output_list = [np.hstack(input_group) for input_group in zip(*output_list)]
        else:
            output_list, other_list = _split_output(output)

        # In-place outputs are treated as outputs from here
        output_list = in_output_list + output_list

        # Prepare raw
        param_map = list(zip(*param_list_unique))  # account for use_run_unique
        output_shape = output_list[0].shape
        for output in output_list:
            if output.shape != output_shape:
                raise ValueError("All outputs must have the same shape")
        if per_column:
            n_input_cols = 1
        else:
            n_input_cols = output_shape[1] // n_unique_param_values
        if input_shape_2d is not None:
            if n_input_cols != input_shape_2d[1]:
                if per_column:
                    raise ValueError("All outputs must have one column when per_column=True")
                else:
                    raise ValueError("All outputs must have the number of columns = #input columns x #parameters")
        raw = output_list, param_map, n_input_cols, other_list
        if return_raw:
            if use_run_unique and not silence_warnings:
                warnings.warn("Raw output is produced by unique parameter "
                              "combinations when run_unique=True", stacklevel=2)
            return raw
        if use_run_unique:
            output_list, param_map, n_input_cols, other_list = _use_raw(raw)

    # Update shape and other meta if no inputs
    if input_shape is None:
        if n_input_cols == 1:
            input_shape = (output_list[0].shape[0],)
        else:
            input_shape = (output_list[0].shape[0], n_input_cols)
    else:
        input_shape = orig_input_shape
    if input_index is None:
        input_index = pd.RangeIndex(start=0, step=1, stop=input_shape[0])
    if input_columns is None:
        input_columns = pd.RangeIndex(start=0, step=1, stop=input_shape[1] if len(input_shape) > 1 else 1)

    # 构建列层次结构和创建映射器
    if len(param_list) > 0:
        # 在输入级别之上构建新的列级别
        param_indexes, new_columns = build_columns(
            param_list,
            input_columns,
            level_names=level_names,
            hide_levels=hide_levels,
            param_settings=param_settings,
            per_column=per_column,
            **stacking_kwargs
        )
        # 构建映射器，将输入中的旧列映射到新列
        # 不将所有输入平铺到输出的形状并浪费内存，
        # 我们只保留一个映射器，并在需要时执行平铺
        input_mapper = None
        if len(input_list) > 0:
            if per_column:
                input_mapper = np.arange(len(input_columns))
            else:
                input_mapper = np.tile(np.arange(len(input_columns)), n_param_values)
        # 构建映射器以便在参数和列之间轻松映射
        mapper_list = [param_indexes[i] for i in range(len(param_list))]
    else:
        # 某些指标没有任何参数
        new_columns = input_columns
        input_mapper = None
        mapper_list = []

    # 返回构件：没有pandas对象，只有包装器和NumPy数组
    new_ndim = len(input_shape) if output_list[0].shape[1] == 1 else output_list[0].ndim
    wrapper = ArrayWrapper(input_index, new_columns, new_ndim, **wrapper_kwargs)

    return wrapper, \
           input_list, \
           input_mapper, \
           output_list[:len(in_output_list)], \
           output_list[len(in_output_list):], \
           param_list, \
           mapper_list, \
           other_list


def combine_objs(obj: tp.SeriesFrame,
                 other: tp.MaybeTupleList[tp.Union[tp.ArrayLike, BaseAccessor]],
                 *args, level_name: tp.Optional[str] = None,
                 keys: tp.Optional[tp.IndexLike] = None,
                 allow_multiple: bool = True,
                 **kwargs) -> tp.SeriesFrame:
    """
    组合/比较对象以生成信号或其他结果
    
    功能说明：
    - 将obj与other进行组合或比较，例如生成交易信号
    - 支持与单个对象或多个对象进行比较
    - 自动处理数据的广播和对齐
    - 用于指标之间的交叉比较和信号生成
    
    参数：
        obj (tp.SeriesFrame): 主要的数据对象（Series或DataFrame）
        other (tp.MaybeTupleList[tp.Union[tp.ArrayLike, BaseAccessor]]): 要比较的对象或对象列表
        *args: 传递给组合函数的位置参数
        level_name (tp.Optional[str]): 当处理多个对象时，新创建的列级别名称
        keys (tp.Optional[tp.IndexLike]): 用于标识不同对象的键
        allow_multiple (bool): 是否允许多个对象的比较
        **kwargs: 传递给组合函数的关键字参数
    
    返回值：
        tp.SeriesFrame: 组合后的结果，通常是布尔值表示的信号
    
    使用示例：
        >>> # 单个对象比较
        >>> signals = combine_objs(ma_short, ma_long, vbt.Rep('crossed_above'))
        >>> # 多个对象比较
        >>> signals = combine_objs(price, [ma_10, ma_20, ma_50], vbt.Rep('crossed_above'))
    
    注意：
        两个对象将一起广播。
        将other作为元组或列表传递以与多个参数进行比较。
        在这种情况下，将创建一个名为level_name的新列级别。
    
    参考：
        详见 `vectorbt.base.accessors.BaseAccessor.combine`
    """
    # 如果允许多个对象比较且other是元组或列表
    if allow_multiple and isinstance(other, (tuple, list)):
        if keys is None:
            # 从other的值创建索引键
            keys = index_fns.index_from_values(other, name=level_name)
    
    # 调用基础访问器的combine方法
    return obj.vbt.combine(other, *args, keys=keys, concat=True, allow_multiple=allow_multiple, **kwargs)


# ================================================================================
# 指标基础类型定义
# ================================================================================

# 指标基础类型变量：用于类型提示，绑定到IndicatorBase类
IndicatorBaseT = tp.TypeVar("IndicatorBaseT", bound="IndicatorBase")

# 运行输出类型：指标运行的输出类型，可以是指标实例、元组、原始输出或缓存输出
RunOutputT = tp.Union[IndicatorBaseT, tp.Tuple[tp.Any, ...], RawOutputT, CacheOutputT]

# 运行组合输出类型：多个指标实例的元组，用于参数组合运行
RunCombsOutputT = tp.Tuple[IndicatorBaseT, ...]


class MetaIndicatorBase(type(StatsBuilderMixin), type(PlotsBuilderMixin)):
    """
    指标基础元类
    
    功能说明：
    - 继承自StatsBuilderMixin和PlotsBuilderMixin的元类
    - 用于创建指标基础类的元类
    - 提供统计和绘图功能的元类级别支持
    
    这个元类为指标类提供了额外的功能，包括：
    - 统计构建器的元类功能
    - 绘图构建器的元类功能
    - 为指标类提供统一的元类基础
    """
    pass


class IndicatorBase(Wrapping, StatsBuilderMixin, PlotsBuilderMixin, metaclass=MetaIndicatorBase):
    """
    指标基础类
    
    功能说明：
    - 所有技术指标的基础类
    - 提供指标的基本功能和属性
    - 继承包装、统计构建和绘图构建功能
    - 定义指标的标准接口和行为
    
    主要特性：
    - 数据包装功能（来自Wrapping）
    - 统计计算功能（来自StatsBuilderMixin）
    - 绘图功能（来自PlotsBuilderMixin）
    - 统一的指标接口
    
    属性说明：
    - _short_name: 指标的简短名称
    - _level_names: 参数级别名称
    - _input_names: 输入数据名称
    - _param_names: 参数名称
    - _in_output_names: 就地输出名称
    - _output_names: 输出名称
    - _output_flags: 输出标志
    
    注意：
    属性应该在实例化之前设置。
    
    使用示例：
        >>> # 这是一个基础类，通常不直接使用
        >>> # 而是通过IndicatorFactory创建具体的指标类
        >>> MyIndicator = IndicatorFactory(...).from_apply_func(...)
        >>> indicator = MyIndicator.run(data, params)
    """
    # 类属性定义
    _short_name: str  # 指标的简短名称
    _level_names: tp.Tuple[str, ...]  # 参数级别名称元组
    _input_names: tp.Tuple[str, ...]  # 输入数据名称元组
    _param_names: tp.Tuple[str, ...]  # 参数名称元组
    _in_output_names: tp.Tuple[str, ...]  # 就地输出名称元组
    _output_names: tp.Tuple[str, ...]  # 输出名称元组
    _output_flags: tp.Kwargs  # 输出标志字典

    @property
    def short_name(self) -> str:
        """
        获取指标的简短名称
        
        返回值：
            str: 指标的简短名称，用于标识和显示
        
        使用示例：
            >>> indicator = MyIndicator.run(data, params)
            >>> print(indicator.short_name)  # 输出：'my_indicator'
        """
        return self._short_name

    @property
    def level_names(self) -> tp.Tuple[str, ...]:
        """
        获取参数对应的列级别名称
        
        返回值：
            tp.Tuple[str, ...]: 参数级别名称元组，用于多级列索引
        
        使用示例：
            >>> indicator = MyIndicator.run(data, window=[10, 20])
            >>> print(indicator.level_names)  # 输出：('window',)
        """
        return self._level_names

    @classproperty
    def input_names(cls_or_self) -> tp.Tuple[str, ...]:
        """
        获取输入数组的名称
        
        返回值：
            tp.Tuple[str, ...]: 输入数组名称元组
        
        使用示例：
            >>> print(MyIndicator.input_names)  # 输出：('close', 'volume')
        """
        return cls_or_self._input_names

    @classproperty
    def param_names(cls_or_self) -> tp.Tuple[str, ...]:
        """
        获取参数的名称
        
        返回值：
            tp.Tuple[str, ...]: 参数名称元组
        
        使用示例：
            >>> print(MyIndicator.param_names)  # 输出：('window', 'alpha')
        """
        return cls_or_self._param_names

    @classproperty
    def in_output_names(cls_or_self) -> tp.Tuple[str, ...]:
        """
        获取就地输出数组的名称
        
        返回值：
            tp.Tuple[str, ...]: 就地输出数组名称元组
        
        使用示例：
            >>> print(MyIndicator.in_output_names)  # 输出：('cache',)
        """
        return cls_or_self._in_output_names

    @classproperty
    def output_names(cls_or_self) -> tp.Tuple[str, ...]:
        """
        获取常规输出数组的名称
        
        返回值：
            tp.Tuple[str, ...]: 常规输出数组名称元组
        
        使用示例：
            >>> print(MyIndicator.output_names)  # 输出：('ma', 'signal')
        """
        return cls_or_self._output_names

    @classproperty
    def output_flags(cls_or_self) -> tp.Kwargs:
        """
        获取输出标志字典
        
        返回值：
            tp.Kwargs: 输出标志字典，包含各种输出配置
        
        使用示例：
            >>> print(MyIndicator.output_flags)  # 输出：{'ma': {...}, 'signal': {...}}
        """
        return cls_or_self._output_flags

    def __init__(self,
                 wrapper: ArrayWrapper,
                 input_list: InputListT,
                 input_mapper: InputMapperT,
                 in_output_list: InOutputListT,
                 output_list: OutputListT,
                 param_list: ParamListT,
                 mapper_list: MapperListT,
                 short_name: str,
                 level_names: tp.Tuple[str, ...]) -> None:
        """
        初始化指标基础类实例
        
        功能说明：
        - 初始化指标实例的所有核心组件
        - 设置数据包装器、输入输出列表、参数列表等
        - 执行必要的验证检查确保数据一致性
        - 继承并初始化混合类的功能
        
        参数：
            wrapper (ArrayWrapper): 数组包装器，用于处理索引和列信息
            input_list (InputListT): 输入数组列表
            input_mapper (InputMapperT): 输入映射器，可选
            in_output_list (InOutputListT): 就地输出数组列表
            output_list (OutputListT): 输出数组列表
            param_list (ParamListT): 参数列表
            mapper_list (MapperListT): 映射器列表
            short_name (str): 指标的简短名称
            level_names (tp.Tuple[str, ...]): 级别名称元组
        
        验证检查：
        - 输入映射器形状与包装器形状一致
        - 输入数组行数与包装器行数一致
        - 输出数组形状与包装器形状一致
        - 参数列表长度一致
        - 映射器长度与包装器列数一致
        - 短名称为字符串类型
        - 级别名称数量与参数列表数量一致
        
        使用示例：
            >>> # 通常不直接调用，而是通过run方法创建
            >>> indicator = MyIndicator.run(data, params)
        """
        # 初始化包装功能
        Wrapping.__init__(
            self,
            wrapper,
            input_list=input_list,
            input_mapper=input_mapper,
            in_output_list=in_output_list,
            output_list=output_list,
            param_list=param_list,
            mapper_list=mapper_list,
            short_name=short_name,
            level_names=level_names
        )
        
        # 初始化统计构建器功能
        StatsBuilderMixin.__init__(self)
        
        # 初始化绘图构建器功能
        PlotsBuilderMixin.__init__(self)

        # 验证输入映射器形状
        if input_mapper is not None:
            checks.assert_equal(input_mapper.shape[0], wrapper.shape_2d[1])
        
        # 验证输入数组形状
        for ts in input_list:
            checks.assert_equal(ts.shape[0], wrapper.shape_2d[0])
        
        # 验证输出数组形状
        for ts in in_output_list + output_list:
            checks.assert_equal(ts.shape, wrapper.shape_2d)
        
        # 验证参数列表长度
        for params in param_list:
            checks.assert_len_equal(param_list[0], params)
        
        # 验证映射器长度
        for mapper in mapper_list:
            checks.assert_equal(len(mapper), wrapper.shape_2d[1])
        
        # 验证短名称类型
        checks.assert_instance_of(short_name, str)
        
        # 验证级别名称数量
        checks.assert_len_equal(level_names, param_list)

        # 设置短名称和级别名称属性
        setattr(self, '_short_name', short_name)
        setattr(self, '_level_names', level_names)

        # 设置输入数组属性
        for i, ts_name in enumerate(self.input_names):
            setattr(self, f'_{ts_name}', input_list[i])
        
        # 设置输入映射器属性
        setattr(self, '_input_mapper', input_mapper)
        
        # 设置就地输出数组属性
        for i, in_output_name in enumerate(self.in_output_names):
            setattr(self, f'_{in_output_name}', in_output_list[i])
        
        # 设置输出数组属性
        for i, output_name in enumerate(self.output_names):
            setattr(self, f'_{output_name}', output_list[i])
        
        # 设置参数列表和映射器属性
        for i, param_name in enumerate(self.param_names):
            setattr(self, f'_{param_name}_list', param_list[i])
            setattr(self, f'_{param_name}_mapper', mapper_list[i])
        
        # 如果有多个参数，创建元组映射器
        if len(self.param_names) > 1:
            tuple_mapper = list(zip(*list(mapper_list)))
            setattr(self, '_tuple_mapper', tuple_mapper)

    def indexing_func(self: IndicatorBaseT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> IndicatorBaseT:
        """
        对指标基础类执行索引操作
        
        功能说明：
        - 对指标实例执行pandas风格的索引操作
        - 支持行索引和列索引
        - 保持指标的内部结构和一致性
        - 返回索引后的新指标实例
        
        参数：
            pd_indexing_func (tp.PandasIndexingFunc): pandas索引函数
            **kwargs: 传递给索引函数的关键字参数
        
        返回值：
            IndicatorBaseT: 索引后的新指标实例
        
        使用示例：
            >>> # 选择特定的行和列
            >>> sub_indicator = indicator.indexing_func(
            ...     lambda x: x.iloc[10:20, ['A', 'B']]
            ... )
            >>> # 或者使用更简单的方式
            >>> sub_indicator = indicator.iloc[10:20, ['A', 'B']]
        """
        # 获取索引元数据
        new_wrapper, idx_idxs, _, col_idxs = self.wrapper.indexing_func_meta(pd_indexing_func, **kwargs)
        
        # 转换索引为一维数组
        idx_idxs_arr = reshape_fns.to_1d_array(idx_idxs)
        col_idxs_arr = reshape_fns.to_1d_array(col_idxs)
        
        # 如果索引覆盖所有行，使用切片
        if np.array_equal(idx_idxs_arr, np.arange(self.wrapper.shape_2d[0])):
            idx_idxs_arr = slice(None, None, None)
        
        # 如果索引覆盖所有列，使用切片
        if np.array_equal(col_idxs_arr, np.arange(self.wrapper.shape_2d[1])):
            col_idxs_arr = slice(None, None, None)

        # 处理输入映射器
        input_mapper = getattr(self, '_input_mapper', None)
        if input_mapper is not None:
            input_mapper = input_mapper[col_idxs_arr]
        input_list = []
        for input_name in self.input_names:
            input_list.append(getattr(self, f'_{input_name}')[idx_idxs_arr])
        in_output_list = []
        for in_output_name in self.in_output_names:
            in_output_list.append(getattr(self, f'_{in_output_name}')[idx_idxs_arr, :][:, col_idxs_arr])
        output_list = []
        for output_name in self.output_names:
            output_list.append(getattr(self, f'_{output_name}')[idx_idxs_arr, :][:, col_idxs_arr])
        param_list = []
        for param_name in self.param_names:
            param_list.append(getattr(self, f'_{param_name}_list'))
        mapper_list = []
        for param_name in self.param_names:
            # Tuple mapper is a list because of its complex data type
            mapper_list.append(getattr(self, f'_{param_name}_mapper')[col_idxs_arr])

        return self.replace(
            wrapper=new_wrapper,
            input_list=input_list,
            input_mapper=input_mapper,
            in_output_list=in_output_list,
            output_list=output_list,
            param_list=param_list,
            mapper_list=mapper_list
        )

    @classmethod
    def _run(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunOutputT:
        """
        私有运行方法
        
        功能说明：
        - 实际的指标计算逻辑实现
        - 由子类重写以提供具体的计算功能
        - 被公共run方法调用
        
        注意：
        这是一个抽象方法，必须由子类实现。
        
        抛出异常：
            NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError

    @classmethod
    def run(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunOutputT:
        """
        公共运行方法
        
        功能说明：
        - 指标计算的主要入口点
        - 接受输入数据和参数，返回计算结果
        - 调用私有的_run方法执行实际计算
        
        参数：
            *args: 位置参数，通常包含输入数据
            **kwargs: 关键字参数，包含计算参数和配置选项
        
        返回值：
            RunOutputT: 运行结果，可以是指标实例、元组或其他类型
        
        使用示例：
            >>> # 运行移动平均指标
            >>> ma_indicator = MyMA.run(price_data, window=20)
            >>> # 运行RSI指标
            >>> rsi_indicator = MyRSI.run(price_data, window=14)
        """
        return cls._run(*args, **kwargs)

    @classmethod
    def _run_combs(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunCombsOutputT:
        """
        私有运行组合方法
        
        功能说明：
        - 实际的参数组合计算逻辑
        - 用于生成参数的所有组合并分别计算
        - 由子类重写以提供具体的组合计算功能
        
        注意：
        这是一个抽象方法，必须由子类实现。
        
        抛出异常：
            NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError

    @classmethod
    def run_combs(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunCombsOutputT:
        """
        公共运行组合方法
        
        功能说明：
        - 参数组合计算的主要入口点
        - 计算所有参数组合的结果
        - 返回多个指标实例用于比较
        - 调用私有的_run_combs方法执行实际计算
        
        参数：
            *args: 位置参数，通常包含输入数据
            **kwargs: 关键字参数，包含计算参数和配置选项
        
        返回值：
            RunCombsOutputT: 多个指标实例的元组
        
        使用示例：
            >>> # 运行多个窗口的移动平均
            >>> ma1, ma2 = MyMA.run_combs(price_data, window=[10, 20, 30])
            >>> # 生成交叉信号
            >>> signals = ma1.crossed_above(ma2)
        """
        return cls._run_combs(*args, **kwargs)


class IndicatorFactory:
    """
    指标工厂类 - 用于创建新的技术指标类
    
    功能说明：
    - 提供便捷的方式创建自定义技术指标类
    - 支持多种输入、参数和输出配置
    - 自动生成指标类的方法和属性
    - 支持与第三方库的集成
    
    主要特性：
    - 灵活的参数配置
    - 自动生成属性访问器
    - 支持缓存和性能优化
    - 统计和绘图功能集成
    - 第三方库集成支持
    
    使用流程：
    1. 初始化IndicatorFactory创建骨架
    2. 使用from_custom_func等方法绑定计算函数
    3. 获得完整的指标类用于计算
    
    使用示例：
        >>> # 创建简单的移动平均指标
        >>> MyMA = IndicatorFactory(
        ...     class_name='MyMA',
        ...     input_names=['close'],
        ...     param_names=['window'],
        ...     output_names=['ma']
        ... ).from_apply_func(lambda close, window: close.rolling(window).mean())
        >>> 
        >>> # 运行指标
        >>> ma_result = MyMA.run(price_data, window=20)
    """
    
    def __init__(self,
                 class_name: str = 'Indicator',
                 class_docstring: str = '',
                 module_name: tp.Optional[str] = __name__,
                 short_name: tp.Optional[str] = None,
                 prepend_name: bool = True,
                 input_names: tp.Optional[tp.Sequence[str]] = None,
                 param_names: tp.Optional[tp.Sequence[str]] = None,
                 in_output_names: tp.Optional[tp.Sequence[str]] = None,
                 output_names: tp.Optional[tp.Sequence[str]] = None,
                 output_flags: tp.KwargsLike = None,
                 custom_output_props: tp.KwargsLike = None,
                 attr_settings: tp.KwargsLike = None,
                 metrics: tp.Optional[tp.Kwargs] = None,
                 stats_defaults: tp.Union[None, tp.Callable, tp.Kwargs] = None,
                 subplots: tp.Optional[tp.Kwargs] = None,
                 plots_defaults: tp.Union[None, tp.Callable, tp.Kwargs] = None) -> None:
        """
        初始化指标工厂类
        
        功能说明：
        - 创建指标类的骨架结构
        - 设置输入、参数和输出的名称和配置
        - 配置统计和绘图功能
        - 准备后续绑定计算函数
        
        参数：
            class_name (str): 创建的指标类名称，默认为'Indicator'
            class_docstring (str): 创建的指标类的文档字符串
            module_name (tp.Optional[str]): 指定类所属的模块名称
            short_name (tp.Optional[str]): 指标的简短名称
                默认为小写的class_name
            prepend_name (bool): 是否在每个参数级别前添加short_name
            input_names (tp.Optional[tp.Sequence[str]]): 输入数组名称列表
                例如：['close', 'volume']
            param_names (tp.Optional[tp.Sequence[str]]): 参数名称列表
                例如：['window', 'alpha']
            in_output_names (tp.Optional[tp.Sequence[str]]): 就地输出数组名称列表
                就地输出是不返回但就地修改的输出。优点包括：
                1) 不需要返回
                2) 可以像输入一样在函数间传递
                3) 可以提供已分配的数据来节省内存
                4) 如果未提供数据或默认值，则创建为空以不占用内存
            output_names (tp.Optional[tp.Sequence[str]]): 输出数组名称列表
                例如：['ma', 'signal']
            output_flags (tp.KwargsLike): 就地输出和常规输出标志字典
            custom_output_props (tp.KwargsLike): 用户自定义函数字典
                将绑定到指标类并用@cached_property包装
            attr_settings (tp.KwargsLike): 按属性名称的设置字典
                属性可以是input_names、in_output_names、output_names和custom_output_props
                接受的键：
                - dtype: 用于确定围绕此属性生成哪些方法的数据类型
                  设置为None以禁用。默认为np.float64
                  可以设置为collections.namedtuple实例作为枚举类型或其他映射
            metrics (tp.Optional[tp.Kwargs]): 统计构建器支持的指标
                如果是dict，将转换为vectorbt.utils.config.Config
            stats_defaults (tp.Union[None, tp.Callable, tp.Kwargs]): 统计默认值
            subplots (tp.Optional[tp.Kwargs]): 子图配置
            plots_defaults (tp.Union[None, tp.Callable, tp.Kwargs]): 绘图默认值
        
        使用示例：
            >>> # 创建基本的指标工厂
            >>> factory = IndicatorFactory(
            ...     class_name='MyIndicator',
            ...     input_names=['close'],
            ...     param_names=['window'],
            ...     output_names=['result']
            ... )
            >>> 
            >>> # 创建带有多个输入和参数的复杂指标
            >>> complex_factory = IndicatorFactory(
            ...     class_name='ComplexIndicator',
            ...     input_names=['open', 'high', 'low', 'close', 'volume'],
            ...     param_names=['short_window', 'long_window', 'alpha'],
            ...     output_names=['signal', 'strength'],
            ...     metrics={'sharpe_ratio': dict(calc_func=lambda x: x.sharpe_ratio())},
            ...     attr_settings={'signal': {'dtype': bool}}
            ... )
        """
        # 验证和初始化参数
        if input_names is None:
            input_names = []
        if param_names is None:
            param_names = []
        if in_output_names is None:
            in_output_names = []
        if output_names is None:
            output_names = []
        if output_flags is None:
            output_flags = {}
        if custom_output_props is None:
            custom_output_props = {}
        if attr_settings is None:
            attr_settings = {}
        if metrics is None:
            metrics = {}
        if subplots is None:
            subplots = {}
        if short_name is None:
            short_name = class_name.lower()
        
        # 检查并保存参数
        self.class_name = class_name
        checks.assert_instance_of(class_name, str)

        self.class_docstring = class_docstring
        checks.assert_instance_of(class_docstring, str)

        self.module_name = module_name
        if module_name is not None:
            checks.assert_instance_of(module_name, str)

        if short_name is None:
            if class_name == 'Indicator':
                short_name = 'custom'
            else:
                short_name = class_name.lower()
        self.short_name = short_name
        checks.assert_instance_of(short_name, str)

        self.prepend_name = prepend_name
        checks.assert_instance_of(prepend_name, bool)

        if input_names is None:
            input_names = []
        else:
            checks.assert_sequence(input_names)
            input_names = list(input_names)
        self.input_names = input_names

        if param_names is None:
            param_names = []
        else:
            checks.assert_sequence(param_names)
            param_names = list(param_names)
        self.param_names = param_names

        if in_output_names is None:
            in_output_names = []
        else:
            checks.assert_sequence(in_output_names)
            in_output_names = list(in_output_names)
        self.in_output_names = in_output_names

        if output_names is None:
            output_names = []
        else:
            checks.assert_sequence(output_names)
            output_names = list(output_names)
        self.output_names = output_names

        all_output_names = in_output_names + output_names
        if len(all_output_names) == 0:
            raise ValueError("Must have at least one in-place or regular output")

        if output_flags is None:
            output_flags = {}
        checks.assert_instance_of(output_flags, dict)
        if len(output_flags) > 0:
            checks.assert_dict_valid(output_flags, all_output_names)
        self.output_flags = output_flags

        if custom_output_props is None:
            custom_output_props = {}
        checks.assert_instance_of(custom_output_props, dict)
        self.custom_output_props = custom_output_props

        if attr_settings is None:
            attr_settings = {}
        checks.assert_instance_of(attr_settings, dict)
        all_attr_names = input_names + all_output_names + list(custom_output_props.keys())
        if len(attr_settings) > 0:
            checks.assert_dict_valid(attr_settings, all_attr_names)
        self.attr_settings = attr_settings

        # Set up class
        ParamIndexer = build_param_indexer(
            param_names + (['tuple'] if len(param_names) > 1 else []),
            module_name=module_name
        )
        Indicator = type(self.class_name, (IndicatorBase, ParamIndexer), {})
        Indicator.__doc__ = self.class_docstring
        if module_name is not None:
            Indicator.__module__ = self.module_name

        # Create read-only properties
        setattr(Indicator, "_input_names", tuple(input_names))
        setattr(Indicator, "_param_names", tuple(param_names))
        setattr(Indicator, "_in_output_names", tuple(in_output_names))
        setattr(Indicator, "_output_names", tuple(output_names))
        setattr(Indicator, "_output_flags", output_flags)

        for param_name in param_names:
            def param_list_prop(self, _param_name=param_name) -> tp.List[tp.Param]:
                return getattr(self, f'_{_param_name}_list')

            param_list_prop.__doc__ = f"List of `{param_name}` values."
            setattr(Indicator, f'{param_name}_list', property(param_list_prop))

        for input_name in input_names:
            def input_prop(self, _input_name: str = input_name) -> tp.SeriesFrame:
                """Input array."""
                old_input = reshape_fns.to_2d_array(getattr(self, '_' + _input_name))
                input_mapper = getattr(self, '_input_mapper')
                if input_mapper is None:
                    return self.wrapper.wrap(old_input)
                return self.wrapper.wrap(old_input[:, input_mapper])

            input_prop.__name__ = input_name
            setattr(Indicator, input_name, cached_property(input_prop))

        for output_name in all_output_names:
            def output_prop(self, _output_name: str = output_name) -> tp.SeriesFrame:
                return self.wrapper.wrap(getattr(self, '_' + _output_name))

            if output_name in in_output_names:
                output_prop.__doc__ = """In-place output array."""
            else:
                output_prop.__doc__ = """Output array."""

            output_prop.__name__ = output_name
            if output_name in output_flags:
                _output_flags = output_flags[output_name]
                if isinstance(_output_flags, (tuple, list)):
                    _output_flags = ', '.join(_output_flags)
                output_prop.__doc__ += "\n\n" + _output_flags
            setattr(Indicator, output_name, property(output_prop))

        # Add __init__ method
        def __init__(self,
                     wrapper: ArrayWrapper,
                     input_list: InputListT,
                     input_mapper: InputMapperT,
                     in_output_list: InOutputListT,
                     output_list: OutputListT,
                     param_list: ParamListT,
                     mapper_list: MapperListT,
                     short_name: str,
                     level_names: tp.Tuple[str, ...]) -> None:
            IndicatorBase.__init__(
                self,
                wrapper,
                input_list,
                input_mapper,
                in_output_list,
                output_list,
                param_list,
                mapper_list,
                short_name,
                level_names
            )
            if len(param_names) > 1:
                tuple_mapper = list(zip(*list(mapper_list)))
            else:
                tuple_mapper = None

            # Initialize indexers
            mapper_sr_list = []
            for i, m in enumerate(mapper_list):
                mapper_sr_list.append(pd.Series(m, index=wrapper.columns))
            if tuple_mapper is not None:
                mapper_sr_list.append(pd.Series(tuple_mapper, index=wrapper.columns))
            ParamIndexer.__init__(self, mapper_sr_list, level_names=[*level_names, level_names])

        setattr(Indicator, '__init__', __init__)

        # Add user-defined outputs
        for prop_name, prop in custom_output_props.items():
            if prop.__doc__ is None:
                prop.__doc__ = f"""Custom property."""
            prop.__name__ = prop_name
            prop = cached_property(prop)
            setattr(Indicator, prop_name, prop)

        # Add comparison & combination methods for all inputs, outputs, and user-defined properties
        def assign_combine_method(func_name: str,
                                  combine_func: tp.Callable,
                                  def_kwargs: tp.Kwargs,
                                  attr_name: str,
                                  docstring: str) -> None:
            def combine_method(self: IndicatorBaseT,
                               other: tp.MaybeTupleList[tp.Union[IndicatorBaseT, tp.ArrayLike, BaseAccessor]],
                               level_name: tp.Optional[str] = None,
                               allow_multiple: bool = True,
                               _prepend_name: bool = prepend_name,
                               **kwargs) -> tp.SeriesFrame:
                if allow_multiple and isinstance(other, (tuple, list)):
                    other = list(other)
                    for i in range(len(other)):
                        if isinstance(other[i], IndicatorBase):
                            other[i] = getattr(other[i], attr_name)
                else:
                    if isinstance(other, IndicatorBase):
                        other = getattr(other, attr_name)
                if level_name is None:
                    if _prepend_name:
                        if attr_name == self.short_name:
                            level_name = f'{self.short_name}_{func_name}'
                        else:
                            level_name = f'{self.short_name}_{attr_name}_{func_name}'
                    else:
                        level_name = f'{attr_name}_{func_name}'
                out = combine_objs(
                    getattr(self, attr_name),
                    other,
                    combine_func=combine_func,
                    level_name=level_name,
                    allow_multiple=allow_multiple,
                    **merge_dicts(def_kwargs, kwargs)
                )
                return out

            combine_method.__qualname__ = f'{Indicator.__name__}.{attr_name}_{func_name}'
            combine_method.__doc__ = docstring
            setattr(Indicator, f'{attr_name}_{func_name}', combine_method)

        for attr_name in all_attr_names:
            _attr_settings = attr_settings.get(attr_name, {})
            checks.assert_dict_valid(_attr_settings, ['dtype'])
            dtype = _attr_settings.get('dtype', np.float64)

            if checks.is_mapping_like(dtype):
                def attr_readable(self,
                                  _attr_name: str = attr_name,
                                  _mapping: tp.MappingLike = dtype) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt(mapping=_mapping).apply_mapping()

                attr_readable.__qualname__ = f'{Indicator.__name__}.{attr_name}_readable'
                attr_readable.__doc__ = inspect.cleandoc(
                    """`{attr_name}` in readable format based on the following mapping: 
                                
                    ```json
                    {dtype}
                    ```"""
                ).format(
                    attr_name=attr_name,
                    dtype=to_doc(to_mapping(dtype))
                )
                setattr(Indicator, f'{attr_name}_readable', property(attr_readable))

                def attr_stats(self, *args,
                               _attr_name: str = attr_name,
                               _mapping: tp.MappingLike = dtype,
                               **kwargs) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt(mapping=_mapping).stats(*args, **kwargs)

                attr_stats.__qualname__ = f'{Indicator.__name__}.{attr_name}_stats'
                attr_stats.__doc__ = inspect.cleandoc(
                    """Stats of `{attr_name}` based on the following mapping: 

                    ```json
                    {dtype}
                    ```"""
                ).format(
                    attr_name=attr_name,
                    dtype=to_doc(to_mapping(dtype))
                )
                setattr(Indicator, f'{attr_name}_stats', attr_stats)

            elif np.issubdtype(dtype, np.number):
                func_info = [
                    ('above', np.greater, dict()),
                    ('below', np.less, dict()),
                    ('equal', np.equal, dict()),
                    ('crossed_above', lambda x, y, wait=0: generic_nb.crossed_above_nb(x, y, wait), dict(to_2d=True)),
                    ('crossed_below', lambda x, y, wait=0: generic_nb.crossed_above_nb(y, x, wait), dict(to_2d=True))
                ]
                for func_name, np_func, def_kwargs in func_info:
                    method_docstring = f"""Return True for each element where `{attr_name}` is {func_name} `other`. 
                
                    See `vectorbt.indicators.factory.combine_objs`."""
                    assign_combine_method(func_name, np_func, def_kwargs, attr_name, method_docstring)

                def attr_stats(self, *args, _attr_name: str = attr_name, **kwargs) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt.stats(*args, **kwargs)

                attr_stats.__qualname__ = f'{Indicator.__name__}.{attr_name}_stats'
                attr_stats.__doc__ = f"""Stats of `{attr_name}` as generic."""
                setattr(Indicator, f'{attr_name}_stats', attr_stats)

            elif np.issubdtype(dtype, np.bool_):
                func_info = [
                    ('and', np.logical_and, dict()),
                    ('or', np.logical_or, dict()),
                    ('xor', np.logical_xor, dict())
                ]
                for func_name, np_func, def_kwargs in func_info:
                    method_docstring = f"""Return `{attr_name} {func_name.upper()} other`. 

                    See `vectorbt.indicators.factory.combine_objs`."""
                    assign_combine_method(func_name, np_func, def_kwargs, attr_name, method_docstring)

                def attr_stats(self, *args, _attr_name: str = attr_name, **kwargs) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt.signals.stats(*args, **kwargs)

                attr_stats.__qualname__ = f'{Indicator.__name__}.{attr_name}_stats'
                attr_stats.__doc__ = f"""Stats of `{attr_name}` as signals."""
                setattr(Indicator, f'{attr_name}_stats', attr_stats)

        # Prepare stats
        if metrics is not None:
            if not isinstance(metrics, Config):
                metrics = Config(metrics, copy_kwargs=dict(copy_mode='deep'))
            setattr(Indicator, "_metrics", metrics.copy())

        if stats_defaults is not None:
            if isinstance(stats_defaults, dict):
                def stats_defaults_prop(self, _stats_defaults: tp.Kwargs = stats_defaults) -> tp.Kwargs:
                    return _stats_defaults
            else:
                def stats_defaults_prop(self, _stats_defaults: tp.Kwargs = stats_defaults) -> tp.Kwargs:
                    return stats_defaults(self)
            stats_defaults_prop.__name__ = "stats_defaults"
            setattr(Indicator, "stats_defaults", property(stats_defaults_prop))

        # Prepare plots
        if subplots is not None:
            if not isinstance(subplots, Config):
                subplots = Config(subplots, copy_kwargs=dict(copy_mode='deep'))
            setattr(Indicator, "_subplots", subplots.copy())

        if plots_defaults is not None:
            if isinstance(plots_defaults, dict):
                def plots_defaults_prop(self, _plots_defaults: tp.Kwargs = plots_defaults) -> tp.Kwargs:
                    return _plots_defaults
            else:
                def plots_defaults_prop(self, _plots_defaults: tp.Kwargs = plots_defaults) -> tp.Kwargs:
                    return plots_defaults(self)
            plots_defaults_prop.__name__ = "plots_defaults"
            setattr(Indicator, "plots_defaults", property(plots_defaults_prop))

        # Save indicator
        self.Indicator = Indicator

    def from_custom_func(self,
                         custom_func: tp.Callable,
                         require_input_shape: bool = False,
                         param_settings: tp.KwargsLike = None,
                         in_output_settings: tp.KwargsLike = None,
                         hide_params: tp.Optional[tp.Sequence[str]] = None,
                         hide_default: bool = True,
                         var_args: bool = False,
                         keyword_only_args: bool = False,
                         **pipeline_kwargs) -> tp.Type[IndicatorBase]:
        """
        基于自定义计算函数构建指标类
        
        功能说明：
        - 相比于 `IndicatorFactory.from_apply_func`，此方法提供完全的灵活性
        - 需要开发者自己处理缓存和为每个参数组合连接列
        - 确保每个输出数组具有适当的列数（输入数组列数 × 参数组合数）
        
        参数说明：
        - custom_func: 自定义计算函数
          接收广播后的输入数组（对应input_names）
          接收广播后的就地输出数组（对应in_output_names）
          接收广播后的参数数组（对应param_names）
          返回对应output_names的输出和其他对象
        - require_input_shape: 是否需要输入形状
        - param_settings: 参数设置字典（按名称索引）
        - in_output_settings: 就地输出设置字典（按名称索引）
        - hide_params: 要隐藏列级别的参数名称列表
        - hide_default: 是否隐藏具有默认值的参数的列级别
        - var_args: 运行方法是否接受可变参数(*args)
        - keyword_only_args: 运行方法是否接受仅关键字参数
        - **pipeline_kwargs: 传递给run_pipeline的关键字参数
        
        返回值：
        - 返回指标类和可能的其他对象
        
        注意事项：
        - 自定义函数可以进行Numba编译
        - 每个输出的形状应该相同，匹配输入形状沿列轴堆叠n次
        - n为参数值的数量
        
        使用示例：
        ```python
        @njit
        def custom_func(ts1, ts2, p1, p2, arg1, arg2):
            return vbt.base.combine_fns.apply_and_concat_multiple_nb(
                len(p1), apply_func_nb, ts1, ts2, p1, p2, arg1, arg2)
        
        MyInd = vbt.IndicatorFactory(
            input_names=['ts1', 'ts2'],
            param_names=['p1', 'p2'],
            output_names=['o1', 'o2']
        ).from_custom_func(custom_func, var_args=True, arg2=200)
        ```
        """
        # 获取指标类的引用
        Indicator = self.Indicator

        # 获取工厂的配置信息
        short_name = self.short_name
        prepend_name = self.prepend_name
        input_names = self.input_names
        param_names = self.param_names
        in_output_names = self.in_output_names
        output_names = self.output_names

        # 创建所有输入名称的组合列表
        all_input_names = input_names + param_names + in_output_names

        # 将自定义函数设置为指标类的属性
        setattr(Indicator, 'custom_func', custom_func)

        def _merge_settings(old_settings: tp.KwargsLike,
                            new_settings: tp.KwargsLike,
                            allowed_keys: tp.Optional[tp.Sequence[tp.MaybeSequence[str]]] = None) -> tp.Kwargs:
            """
            合并设置字典的内部函数
            
            参数：
            - old_settings: 旧设置字典
            - new_settings: 新设置字典
            - allowed_keys: 允许的键列表（可选）
            
            返回：
            - 合并后的设置字典
            """
            new_settings = merge_dicts(old_settings, new_settings)  # 合并字典
            if len(new_settings) > 0 and allowed_keys is not None:
                checks.assert_dict_valid(new_settings, allowed_keys)  # 验证字典键的有效性
            return new_settings

        def _resolve_refs(input_list: tp.Sequence[tp.ArrayLike],
                          param_list: tp.Sequence[tp.Param],
                          in_output_list: tp.Sequence[tp.ArrayLike]) \
                -> tp.Tuple[tp.List[tp.ArrayLike], tp.List[tp.Param], tp.List[tp.ArrayLike]]:
            """
            解析输入、参数和就地输出之间的引用关系
            
            功能：
            - 可以在输入、参数和就地输出之间引用任何内容
            - 甚至可以将参数引用到输入（通过广播实现）
            
            参数：
            - input_list: 输入数组列表
            - param_list: 参数列表
            - in_output_list: 就地输出数组列表
            
            返回：
            - 解析后的输入列表、参数列表和就地输出列表
            """
            # 创建包含所有输入的列表
            all_inputs = list(input_list) + list(param_list) + list(in_output_list)
            
            # 遍历所有输入，解析字符串引用
            for i in range(len(all_inputs)):
                input = all_inputs[i]
                is_default = False
                
                # 检查是否为默认值
                if isinstance(input, Default):
                    input = input.value
                    is_default = True
                
                # 如果是字符串引用，解析为实际对象
                if isinstance(input, str):
                    if input in all_input_names:
                        new_input = all_inputs[all_input_names.index(input)]
                        if is_default:
                            new_input = Default(new_input)
                        all_inputs[i] = new_input
            
            # 重新分割列表
            input_list = all_inputs[:len(input_list)]
            all_inputs = all_inputs[len(input_list):]
            param_list = all_inputs[:len(param_list)]
            in_output_list = all_inputs[len(param_list):]
            
            return input_list, param_list, in_output_list

        def _extract_inputs(args: tp.Sequence) \
                -> tp.Tuple[tp.List[tp.ArrayLike], tp.List[tp.Param], tp.List[tp.ArrayLike], tuple]:
            """
            从参数序列中提取输入数组的内部函数
            
            参数：
            - args: 传入的参数序列
            
            返回：
            - 输入数组列表、参数列表、就地输出列表和剩余参数
            """
            # 提取输入数组
            input_list = args[:len(input_names)]
            checks.assert_len_equal(input_list, input_names)  # 验证输入数组数量
            args = args[len(input_names):]  # 移除已提取的输入数组

            param_list = args[:len(param_names)]
            checks.assert_len_equal(param_list, param_names)
            args = args[len(param_names):]

            in_output_list = args[:len(in_output_names)]
            checks.assert_len_equal(in_output_list, in_output_names)
            args = args[len(in_output_names):]
            if not var_args and len(args) > 0:
                raise TypeError("Variable length arguments are not supported by this function "
                                "(var_args is set to False)")

            input_list, param_list, in_output_list = _resolve_refs(input_list, param_list, in_output_list)
            return input_list, param_list, in_output_list, args

        for k, v in pipeline_kwargs.items():
            if k in param_names and not isinstance(v, Default):
                pipeline_kwargs[k] = Default(v)  # track default params
        pipeline_kwargs = merge_dicts({k: None for k in in_output_names}, pipeline_kwargs)

        # Display default parameters and in-place outputs in the signature
        default_kwargs = {}
        for k in list(pipeline_kwargs.keys()):
            if k in input_names or k in param_names or k in in_output_names:
                default_kwargs[k] = pipeline_kwargs.pop(k)

        if var_args and keyword_only_args:
            raise ValueError("var_args and keyword_only_args cannot be used together")

        # Add private run method
        def_run_kwargs = dict(
            short_name=short_name,
            hide_params=hide_params,
            hide_default=hide_default,
            **default_kwargs
        )

        def _run(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunOutputT:
            _short_name = kwargs.pop('short_name', def_run_kwargs['short_name'])
            _hide_params = kwargs.pop('hide_params', def_run_kwargs['hide_params'])
            _hide_default = kwargs.pop('hide_default', def_run_kwargs['hide_default'])
            _param_settings = _merge_settings(
                param_settings,
                kwargs.pop('param_settings', {}),
                [param_names]
            )
            _in_output_settings = _merge_settings(
                in_output_settings,
                kwargs.pop('in_output_settings', {}),
                [in_output_names]
            )

            if _hide_params is None:
                _hide_params = []

            args = list(args)

            # Extract inputs
            input_list, param_list, in_output_list, args = _extract_inputs(args)

            # Prepare column levels
            level_names = []
            hide_levels = []
            for i, pname in enumerate(param_names):
                level_name = _short_name + '_' + pname if prepend_name else pname
                level_names.append(level_name)
                if pname in _hide_params or (_hide_default and isinstance(param_list[i], Default)):
                    hide_levels.append(i)
            param_list = [params.value if isinstance(params, Default) else params for params in param_list]

            # Run the pipeline
            results = run_pipeline(
                len(output_names),  # number of returned outputs
                custom_func,
                *args,
                require_input_shape=require_input_shape,
                input_list=input_list,
                in_output_list=in_output_list,
                param_list=param_list,
                level_names=level_names,
                hide_levels=hide_levels,
                param_settings=[_param_settings.get(n, {}) for n in param_names],
                in_output_settings=[_in_output_settings.get(n, {}) for n in in_output_names],
                **merge_dicts(pipeline_kwargs, kwargs)
            )

            # Return the raw result if any of the flags are set
            if kwargs.get('return_raw', False) or kwargs.get('return_cache', False):
                return results

            # Unpack the result
            wrapper, \
            new_input_list, \
            input_mapper, \
            in_output_list, \
            output_list, \
            new_param_list, \
            mapper_list, \
            other_list = results

            # Create a new instance
            obj = cls(
                wrapper,
                new_input_list,
                input_mapper,
                in_output_list,
                output_list,
                new_param_list,
                mapper_list,
                short_name,
                tuple(level_names)
            )
            if len(other_list) > 0:
                return (obj, *tuple(other_list))
            return obj

        setattr(Indicator, '_run', classmethod(_run))

        # Add public run method
        # Create function dynamically to provide user with a proper signature
        def compile_run_function(func_name: str, docstring: str, _default_kwargs: tp.KwargsLike = None) -> tp.Callable:
            pos_names = []
            main_kw_names = []
            other_kw_names = []
            if _default_kwargs is None:
                _default_kwargs = {}
            for k in input_names + param_names:
                if k in _default_kwargs:
                    main_kw_names.append(k)
                else:
                    pos_names.append(k)
            main_kw_names.extend(in_output_names)  # in_output_names are keyword-only
            for k, v in _default_kwargs.items():
                if k not in pos_names and k not in main_kw_names:
                    other_kw_names.append(k)

            _0 = func_name
            _1 = '*, ' if keyword_only_args else ''
            _2 = []
            if require_input_shape:
                _2.append('input_shape')
            _2.extend(pos_names)
            _2 = ', '.join(_2) + ', ' if len(_2) > 0 else ''
            _3 = '*args, ' if var_args else ''
            _4 = ['{}={}'.format(k, k) for k in main_kw_names + other_kw_names]
            _4 = ', '.join(_4) + ', ' if len(_4) > 0 else ''
            _5 = docstring
            _6 = all_input_names
            _6 = ', '.join(_6) + ', ' if len(_6) > 0 else ''
            _7 = []
            if require_input_shape:
                _7.append('input_shape')
            _7.extend(other_kw_names)
            _7 = ['{}={}'.format(k, k) for k in _7]
            _7 = ', '.join(_7) + ', ' if len(_7) > 0 else ''
            func_str = "@classmethod\n" \
                       "def {0}(cls, {1}{2}{3}{4}**kwargs):\n" \
                       "    \"\"\"{5}\"\"\"\n" \
                       "    return cls._{0}({6}{3}{7}**kwargs)".format(
                _0, _1, _2, _3, _4, _5, _6, _7
            )
            scope = {**dict(Default=Default), **_default_kwargs}
            filename = inspect.getfile(lambda: None)
            code = compile(func_str, filename, 'single')
            exec(code, scope)
            return scope[func_name]

        _0 = self.class_name
        _1 = ''
        if len(self.input_names) > 0:
            _1 += '\n* Inputs: ' + ', '.join(map(lambda x: f'`{x}`', self.input_names))
        if len(self.in_output_names) > 0:
            _1 += '\n* In-place outputs: ' + ', '.join(map(lambda x: f'`{x}`', self.in_output_names))
        if len(self.param_names) > 0:
            _1 += '\n* Parameters: ' + ', '.join(map(lambda x: f'`{x}`', self.param_names))
        if len(self.output_names) > 0:
            _1 += '\n* Outputs: ' + ', '.join(map(lambda x: f'`{x}`', self.output_names))
        run_docstring = """运行 `{0}` 指标。
{1}

传递参数名称列表作为 `hide_params` 来隐藏它们的列级别。
设置 `hide_default` 为 False 以显示具有默认值的参数的列级别。

其他关键字参数将传递给 `vectorbt.indicators.factory.run_pipeline`。""".format(_0, _1)
        run = compile_run_function('run', run_docstring, def_run_kwargs)
        setattr(Indicator, 'run', run)

        if len(param_names) > 0:
            # Add private run_combs method
            def_run_combs_kwargs = dict(
                r=2,
                param_product=False,
                comb_func=itertools.combinations,
                run_unique=True,
                short_names=None,
                hide_params=hide_params,
                hide_default=hide_default,
                **default_kwargs
            )

            def _run_combs(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunCombsOutputT:
                _r = kwargs.pop('r', def_run_combs_kwargs['r'])
                _param_product = kwargs.pop('param_product', def_run_combs_kwargs['param_product'])
                _comb_func = kwargs.pop('comb_func', def_run_combs_kwargs['comb_func'])
                _run_unique = kwargs.pop('run_unique', def_run_combs_kwargs['run_unique'])
                _short_names = kwargs.pop('short_names', def_run_combs_kwargs['short_names'])
                _hide_params = kwargs.pop('hide_params', def_run_kwargs['hide_params'])
                _hide_default = kwargs.pop('hide_default', def_run_kwargs['hide_default'])
                _param_settings = _merge_settings(
                    param_settings,
                    kwargs.get('param_settings', {}),  # get, not pop
                    [param_names]
                )

                if _hide_params is None:
                    _hide_params = []
                if _short_names is None:
                    _short_names = [f'{short_name}_{str(i + 1)}' for i in range(_r)]

                args = list(args)

                # Extract inputs
                input_list, param_list, in_output_list, args = _extract_inputs(args)

                # Hide params
                for i, pname in enumerate(param_names):
                    if _hide_default and isinstance(param_list[i], Default):
                        if pname not in _hide_params:
                            _hide_params.append(pname)
                        param_list[i] = param_list[i].value
                checks.assert_len_equal(param_list, param_names)

                # Prepare params
                param_settings_list = [_param_settings.get(n, {}) for n in param_names]
                for i in range(len(param_list)):
                    is_tuple = param_settings_list[i].get('is_tuple', False)
                    is_array_like = param_settings_list[i].get('is_array_like', False)
                    param_list[i] = params_to_list(param_list[i], is_tuple, is_array_like)
                if _param_product:
                    param_list = create_param_product(param_list)
                else:
                    param_list = broadcast_params(param_list)
                if not isinstance(param_list, (tuple, list)):
                    param_list = [param_list]

                # Speed up by pre-calculating raw outputs
                if _run_unique:
                    raw_results = cls._run(
                        *input_list,
                        *param_list,
                        *in_output_list,
                        *args,
                        return_raw=True,
                        run_unique=False,
                        **kwargs
                    )
                    kwargs['use_raw'] = raw_results  # use them next time

                # Generate indicator instances
                instances = []
                if _comb_func == itertools.product:
                    param_lists = zip(*_comb_func(zip(*param_list), repeat=_r))
                else:
                    param_lists = zip(*_comb_func(zip(*param_list), _r))
                for i, param_list in enumerate(param_lists):
                    instances.append(cls._run(
                        *input_list,
                        *zip(*param_list),
                        *in_output_list,
                        *args,
                        short_name=_short_names[i],
                        hide_params=_hide_params,
                        hide_default=_hide_default,
                        run_unique=False,
                        **kwargs
                    ))
                return tuple(instances)

            setattr(Indicator, '_run_combs', classmethod(_run_combs))

            # Add public run_combs method
            _0 = self.class_name
            _1 = ''
            if len(self.input_names) > 0:
                _1 += '\n* Inputs: ' + ', '.join(map(lambda x: f'`{x}`', self.input_names))
            if len(self.in_output_names) > 0:
                _1 += '\n* In-place outputs: ' + ', '.join(map(lambda x: f'`{x}`', self.in_output_names))
            if len(self.param_names) > 0:
                _1 += '\n* Parameters: ' + ', '.join(map(lambda x: f'`{x}`', self.param_names))
            if len(self.output_names) > 0:
                _1 += '\n* Outputs: ' + ', '.join(map(lambda x: f'`{x}`', self.output_names))
            run_combs_docstring = """使用函数 `comb_func` 创建多个 `{0}` 指标的组合。
{1}

`comb_func` 必须接受参数元组的可迭代对象和 `r` 参数。
也接受来自 itertools 的所有组合迭代器，如 `itertools.combinations`。
传递 `r` 来指定要运行的指标数量。
传递 `short_names` 来指定每个指标的简短名称。
设置 `run_unique` 为 True 以首先计算所有参数的原始输出，
然后使用它们来构建每个指标（更快）。

其他关键字参数将传递给 `{0}.run`。""".format(_0, _1)
            run_combs = compile_run_function('run_combs', run_combs_docstring, def_run_combs_kwargs)
            setattr(Indicator, 'run_combs', run_combs)

        return Indicator

    def from_apply_func(self,
                        apply_func: tp.Callable,
                        cache_func: tp.Optional[tp.Callable] = None,
                        pass_packed: bool = False,
                        kwargs_to_args: tp.Optional[tp.Sequence[str]] = None,
                        numba_loop: bool = False,
                        **kwargs) -> tp.Type[IndicatorBase]:
        """
        基于应用函数构建指标类
        
        功能说明：
        - 相比于 `IndicatorFactory.from_custom_func`，此方法会为你处理很多事情
        - 包括缓存、参数选择和连接等
        - 你只需要编写一个接受参数选择的`apply_func`函数
        - 然后自动将结果数组连接成每个输出的单个数组
        
        参数说明：
        - apply_func: 应用函数
          接收输入、参数选择和其他参数，进行计算产生输出
          参数传递顺序：输入数组 → 就地输出数组 → 单个参数选择 → 变长参数
        - cache_func: 缓存函数，用于预处理数据
          接收与apply_func相同的参数，返回对象传递给apply_func
        - pass_packed: 是否为输入、就地输出和参数传递打包的元组
        - kwargs_to_args: 从kwargs字典中作为位置参数传递的关键字参数列表
        - numba_loop: 是否使用Numba循环
          适用于大量小输入的迭代，但不支持变长关键字参数
        - **kwargs: 传递给`IndicatorFactory.from_custom_func`的关键字参数
        
        返回值：
        - 返回指标类
        
        注意事项：
        - 如果apply_func是Numba编译的函数，所有输入自动转换为NumPy数组
        - 每个输出的形状应该相同，匹配每个输入的形状
        - 支持use_ray参数进行并行计算（仅当numba_loop=False时）
        
        使用示例：
        ```python
        @njit
        def apply_func_nb(ts1, ts2, p1, p2, arg1, arg2):
            return ts1 * p1 + arg1, ts2 * p2 + arg2
        
        MyInd = vbt.IndicatorFactory(
            input_names=['ts1', 'ts2'],
            param_names=['p1', 'p2'],
            output_names=['o1', 'o2']
        ).from_apply_func(
            apply_func_nb, var_args=True,
            kwargs_to_args=['arg2'], arg2=200)
        ```
        """
        # 获取指标类的引用
        Indicator = self.Indicator

        # 设置应用函数为指标类的属性
        setattr(Indicator, 'apply_func', apply_func)

        # 处理kwargs_to_args参数
        if kwargs_to_args is None:
            kwargs_to_args = []

        # 获取必要的配置信息
        module_name = self.module_name
        output_names = self.output_names
        in_output_names = self.in_output_names
        param_names = self.param_names

        # 计算返回输出的数量
        num_ret_outputs = len(output_names)

        # 构建一个选择参数元组的函数
        # 在这里完成，避免每次运行custom_func时都进行Numba编译
        
        # 构建函数参数字符串
        _0 = "i"  # 参数索引
        _0 += ", args_before"  # 前置参数
        _0 += ", input_tuple"  # 输入元组
        if len(in_output_names) > 0:
            _0 += ", in_output_tuples"  # 就地输出元组
        if len(param_names) > 0:
            _0 += ", param_tuples"  # 参数元组
        _0 += ", *args"  # 变长参数
        if not numba_loop:
            _0 += ", **_kwargs"  # 关键字参数（非Numba模式）
        
        # 构建函数调用字符串
        _1 = "*args_before"  # 展开前置参数
        if pass_packed:
            # 传递打包的元组
            _1 += ", input_tuple"
            if len(in_output_names) > 0:
                _1 += ", in_output_tuples[i]"
            else:
                _1 += ", ()"
            if len(param_names) > 0:
                _1 += ", param_tuples[i]"
            else:
                _1 += ", ()"
        else:
            # 展开元组中的各个元素
            _1 += ", *input_tuple"
            if len(in_output_names) > 0:
                _1 += ", *in_output_tuples[i]"
            if len(param_names) > 0:
                _1 += ", *param_tuples[i]"
        _1 += ", *args"  # 展开变长参数
        if not numba_loop:
            _1 += ", **_kwargs"  # 展开关键字参数
        
        # 动态生成参数选择函数
        func_str = "def select_params_func({0}):\n   return apply_func({1})".format(_0, _1)
        scope = {'apply_func': apply_func}
        filename = inspect.getfile(lambda: None)
        code = compile(func_str, filename, 'single')
        exec(code, scope)
        select_params_func = scope['select_params_func']
        
        # 设置模块名称
        if module_name is not None:
            select_params_func.__module__ = module_name
        
        # 如果启用Numba循环，编译函数
        if numba_loop:
            select_params_func = njit(select_params_func)

        def custom_func(input_list: tp.List[tp.AnyArray],
                        in_output_list: tp.List[tp.List[tp.AnyArray]],
                        param_list: tp.List[tp.List[tp.Param]],
                        *args,
                        input_shape: tp.Optional[tp.Shape] = None,
                        col: tp.Optional[int] = None,
                        flex_2d: tp.Optional[bool] = None,
                        return_cache: bool = False,
                        use_cache: tp.Optional[CacheOutputT] = None,
                        use_ray: bool = False,
                        **_kwargs) -> tp.Union[None, CacheOutputT, tp.Array2d, tp.List[tp.Array2d]]:
            """
            自定义函数，将输入和参数转发给apply_func
            
            参数：
            - input_list: 输入数组列表
            - in_output_list: 就地输出数组列表的列表
            - param_list: 参数列表的列表
            - *args: 变长参数
            - input_shape: 输入形状（可选）
            - col: 列索引（可选）
            - flex_2d: 是否灵活2D（可选）
            - return_cache: 是否返回缓存
            - use_cache: 使用的缓存
            - use_ray: 是否使用Ray进行并行计算
            - **_kwargs: 其他关键字参数
            
            返回：
            - 无、缓存输出、2D数组或2D数组列表
            """

            # 检查Ray和就地输出的兼容性
            if use_ray:
                if len(in_output_names) > 0:
                    raise ValueError("Ray doesn't support in-place outputs")
            
            # 根据是否使用numba_loop选择合适的应用和连接函数
            if numba_loop:
                if use_ray:
                    raise ValueError("Ray cannot be used within Numba")
                # 选择Numba版本的应用和连接函数
                if num_ret_outputs > 1:
                    apply_and_concat_func = combine_fns.apply_and_concat_multiple_nb
                elif num_ret_outputs == 1:
                    apply_and_concat_func = combine_fns.apply_and_concat_one_nb
                else:
                    apply_and_concat_func = combine_fns.apply_and_concat_none_nb
            else:
                # 选择Python版本的应用和连接函数
                if num_ret_outputs > 1:
                    if use_ray:
                        apply_and_concat_func = combine_fns.apply_and_concat_multiple_ray
                    else:
                        apply_and_concat_func = combine_fns.apply_and_concat_multiple
                elif num_ret_outputs == 1:
                    if use_ray:
                        apply_and_concat_func = combine_fns.apply_and_concat_one_ray
                    else:
                        apply_and_concat_func = combine_fns.apply_and_concat_one
                else:
                    if use_ray:
                        raise ValueError("Ray requires regular outputs")
                    apply_and_concat_func = combine_fns.apply_and_concat_none

            # 计算参数组合的数量
            n_params = len(param_list[0]) if len(param_list) > 0 else 1
            
            # 准备输入、就地输出和参数的元组
            input_tuple = tuple(input_list)
            in_output_tuples = list(zip(*in_output_list))
            param_tuples = list(zip(*param_list))
            
            # 准备前置参数
            args_before = ()
            if input_shape is not None and 'input_shape' not in kwargs_to_args:
                args_before += (input_shape,)
            if col is not None and 'col' not in kwargs_to_args:
                args_before += (col,)

            # 将一些关键字参数作为位置参数传递（Numba要求）
            more_args = ()
            for key in kwargs_to_args:
                value = _kwargs.pop(key)  # important: remove from kwargs
                more_args += (value,)
            if flex_2d is not None and 'flex_2d' not in kwargs_to_args:
                more_args += (flex_2d,)

            # Caching
            cache = use_cache
            if cache is None and cache_func is not None:
                _in_output_list = in_output_list
                _param_list = param_list
                if checks.is_numba_func(cache_func):
                    if len(in_output_list) > 0:
                        _in_output_list = [to_typed_list(in_outputs) for in_outputs in in_output_list]
                    if len(param_list) > 0:
                        _param_list = [to_typed_list(params) for params in param_list]
                cache = cache_func(
                    *args_before,
                    *input_tuple,
                    *_in_output_list,
                    *_param_list,
                    *args,
                    *more_args,
                    **_kwargs
                )
            if return_cache:
                return cache
            if cache is None:
                cache = ()
            if not isinstance(cache, tuple):
                cache = (cache,)

            if len(in_output_names) > 0:
                _in_output_tuples = in_output_tuples
                if numba_loop:
                    _in_output_tuples = to_typed_list(_in_output_tuples)
                _in_output_tuples = (_in_output_tuples,)
            else:
                _in_output_tuples = ()
            if len(param_names) > 0:
                _param_tuples = param_tuples
                if numba_loop:
                    _param_tuples = to_typed_list(_param_tuples)
                _param_tuples = (_param_tuples,)
            else:
                _param_tuples = ()

            return apply_and_concat_func(
                n_params,
                select_params_func,
                args_before,
                input_tuple,
                *_in_output_tuples,
                *_param_tuples,
                *args,
                *more_args,
                *cache,
                **_kwargs
            )

        # 调用from_custom_func方法，传递自定义函数和参数
        return self.from_custom_func(custom_func, as_lists=True, **kwargs)

    @classmethod
    def get_talib_indicators(cls) -> tp.Set[str]:
        """
        获取所有TA-Lib指标
        
        返回值：
        - 包含所有TA-Lib指标函数名称的集合
        """
        import talib

        return set(talib.get_functions())

    @classmethod
    def from_talib(cls, func_name: str, init_kwargs: tp.KwargsLike = None, **kwargs) -> tp.Type[IndicatorBase]:
        """
        基于TA-Lib函数构建指标类
        
        功能说明：
        - 围绕TA-Lib函数构建指标类
        - 需要安装TA-Lib库（https://github.com/mrjbq7/ta-lib）
        - 自动解析输入、参数和输出名称
        
        参数说明：
        - func_name: TA-Lib函数名称
        - init_kwargs: 传递给IndicatorFactory的关键字参数
        - **kwargs: 传递给IndicatorFactory.from_custom_func的关键字参数
        
        返回值：
        - 返回指标类
        
        说明：
        - 输入、参数和输出名称请参考TA-Lib文档
        - 文档地址：https://github.com/mrjbq7/ta-lib/blob/master/docs/index.md
        
        使用示例：
            ```pycon
            >>> SMA = vbt.IndicatorFactory.from_talib('SMA')

            >>> sma = SMA.run(price, timeperiod=[2, 3])
            >>> sma.real
            sma_timeperiod         2         3
                              a    b    a    b
            2020-01-01      NaN  NaN  NaN  NaN
            2020-01-02      1.5  4.5  NaN  NaN
            2020-01-03      2.5  3.5  2.0  4.0
            2020-01-04      3.5  2.5  3.0  3.0
            2020-01-05      4.5  1.5  4.0  2.0
            ```

            * To get help on running the indicator, use the `help` command:

            ```pycon
            >>> help(SMA.run)
            Help on method run:

            run(close, timeperiod=30, short_name='sma', hide_params=None, hide_default=True, **kwargs) method of builtins.type instance
                运行 `SMA` 指标。

                * 输入: `close`
                * 参数: `timeperiod`
                * 输出: `real`

                传递参数名称列表作为 `hide_params` 来隐藏它们的列级别。
                设置 `hide_default` 为 False 以显示具有默认值的参数的列级别。

                其他关键字参数将传递给 `vectorbt.indicators.factory.run_pipeline`。
            ```
        """
        # 导入必要的TA-Lib模块
        import talib
        from talib import abstract

        # 将函数名转换为大写（TA-Lib约定）
        func_name = func_name.upper()
        talib_func = getattr(talib, func_name)
        
        # 获取TA-Lib函数的信息
        info = abstract.Function(func_name).info
        
        # 提取输入名称
        input_names = []
        for in_names in info['input_names'].values():
            if isinstance(in_names, (list, tuple)):
                input_names.extend(list(in_names))
            else:
                input_names.append(in_names)
        
        # 提取类相关信息
        class_name = info['name']
        class_docstring = "{}, {}".format(info['display_name'], info['group'])
        param_names = list(info['parameters'].keys())
        output_names = info['output_names']
        output_flags = info['output_flags']

        def apply_func(input_list: tp.List[tp.AnyArray],
                       in_output_tuple: tp.Tuple[tp.AnyArray, ...],
                       param_tuple: tp.Tuple[tp.Param, ...],
                       **kwargs) -> tp.Union[tp.Array2d, tp.List[tp.Array2d]]:
            """
            应用TA-Lib函数的内部函数
            
            参数：
            - input_list: 输入数组列表
            - in_output_tuple: 就地输出元组
            - param_tuple: 参数元组
            - **kwargs: 其他关键字参数
            
            返回：
            - 2D数组或2D数组列表
            """
            # TA-Lib函数只能处理1维数组
            n_input_cols = input_list[0].shape[1]
            outputs = []
            
            # 逐列处理数据
            for col in range(n_input_cols):
                output = talib_func(
                    *map(lambda x: x[:, col], input_list),  # 提取每列数据
                    *param_tuple,  # 传递参数
                    **kwargs  # 传递其他关键字参数
                )
                outputs.append(output)
            
            # 处理多输出情况
            if isinstance(outputs[0], tuple):  # 多输出
                outputs = list(zip(*outputs))
                return list(map(np.column_stack, outputs))
            # 单输出情况：将各列结果堆叠成2D数组
            return np.column_stack(outputs)

        # 创建TA-Lib指标类
        TALibIndicator = cls(
            **merge_dicts(
                dict(
                    class_name=class_name,
                    class_docstring=class_docstring,
                    input_names=input_names,
                    param_names=param_names,
                    output_names=output_names,
                    output_flags=output_flags
                ),
                init_kwargs  # 用户提供的初始化参数
            )
        ).from_apply_func(
            apply_func,
            pass_packed=True,  # 传递打包的参数
            **info['parameters'],  # 传递TA-Lib函数的参数默认值
            **kwargs  # 传递用户提供的其他参数
        )
        return TALibIndicator

    @classmethod
    def parse_pandas_ta_config(cls,
                               func: tp.Callable,
                               test_input_names: tp.Optional[tp.Sequence[str]] = None,
                               test_index_len: int = 100) -> tp.Kwargs:
        """
        解析pandas-ta指标的配置
        
        功能说明：
        - 通过分析函数签名和运行测试来获取指标配置
        - 自动识别输入参数、参数名称和输出名称
        - 标准化输出名称格式
        
        参数说明：
        - func: pandas-ta指标函数
        - test_input_names: 测试输入名称集合（可选）
        - test_index_len: 测试索引长度
        
        返回值：
        - 包含指标配置的字典
        """
        # 设置默认的测试输入名称
        if test_input_names is None:
            test_input_names = {'open_', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividends', 'split'}

        # 初始化配置信息
        input_names = []
        param_names = []
        defaults = {}
        output_names = []

        # 解析函数签名以获取输入名称
        sig = inspect.signature(func)
        for k, v in sig.parameters.items():
            # 跳过可变参数
            if v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD):
                # 检查参数类型和名称
                if v.annotation != inspect.Parameter.empty and v.annotation == pd.Series:
                    input_names.append(k)  # 注解为pd.Series的参数
                elif k in test_input_names:
                    input_names.append(k)  # 在测试输入名称中的参数
                elif v.default == inspect.Parameter.empty:
                    # 任何位置参数都被视为输入
                    input_names.append(k)
                else:
                    # 有默认值的参数被视为指标参数
                    param_names.append(k)
                    defaults[k] = v.default

        # 为了获取输出名称，需要运行指标
        test_df = pd.DataFrame(
            {c: np.random.uniform(1, 10, size=(test_index_len,)) for c in input_names},
            index=[datetime(2020, 1, 1) + timedelta(days=i) for i in range(test_index_len)]
        )
        new_args = {c: test_df[c] for c in input_names}
        try:
            result = func(**new_args)
        except Exception as e:
            raise ValueError("Couldn't parse the indicator: " + str(e))

        # 如果结果是元组，则连接Series/DataFrames
        if isinstance(result, tuple):
            results = []
            for i, r in enumerate(result):
                if not pd.Index.equals(r.index, test_df.index):
                    warnings.warn(f"Couldn't parse the output at index {i}: mismatching index", stacklevel=2)
                else:
                    results.append(r)
            if len(results) > 1:
                result = pd.concat(results, axis=1)
            elif len(results) == 1:
                result = results[0]
            else:
                raise ValueError("Couldn't parse the output")

        # 测试产生的数组是否具有相同的索引长度
        if not pd.Index.equals(result.index, test_df.index):
            raise ValueError("Couldn't parse the output: mismatching index")

        # 标准化输出名称：移除数字、移除连字符并转换为小写
        output_cols = result.columns.tolist() if isinstance(result, pd.DataFrame) else [result.name]
        new_output_cols = []
        for i in range(len(output_cols)):
            name_parts = []
            for name_part in output_cols[i].split('_'):
                try:
                    float(name_part)  # 尝试转换为浮点数
                    continue  # 跳过数字部分
                except:
                    name_parts.append(name_part.replace('-', '_').lower())
            output_col = '_'.join(name_parts)
            new_output_cols.append(output_col)

        # 为重复项添加数字后缀
        for k, v in Counter(new_output_cols).items():
            if v == 1:
                output_names.append(k)
            else:
                for i in range(v):
                    output_names.append(k + str(i))

        # 返回配置字典
        return dict(
            class_name=func.__name__.upper(),
            class_docstring=func.__doc__,
            input_names=input_names,
            param_names=param_names,
            output_names=output_names,
            defaults=defaults
        )

    @classmethod
    def get_pandas_ta_indicators(cls, silence_warnings: bool = True) -> tp.Set[str]:
        """
        获取所有pandas-ta指标
        
        功能说明：
        - 返回所有成功解析的pandas-ta指标
        - 对于解析失败的指标，可以选择是否显示警告
        
        参数说明：
        - silence_warnings: 是否静默警告
        
        返回值：
        - 包含所有成功解析的指标名称的集合
        
        注意：
        - 只返回成功解析的指标
        """
        import pandas_ta

        indicators = set()
        # 遍历所有pandas-ta类别中的函数
        for func_name in [_k for k, v in pandas_ta.Category.items() for _k in v]:
            try:
                # 尝试解析指标配置
                cls.parse_pandas_ta_config(getattr(pandas_ta, func_name))
                indicators.add(func_name.upper())
            except Exception as e:
                # 如果解析失败，根据设置决定是否显示警告
                if not silence_warnings:
                    warnings.warn(f"Function {func_name}: " + str(e), stacklevel=2)
        return indicators

    @classmethod
    def from_pandas_ta(cls, func_name: str, parse_kwargs: tp.KwargsLike = None,
                       init_kwargs: tp.KwargsLike = None, **kwargs) -> tp.Type[IndicatorBase]:
        """
        基于pandas-ta函数构建指标类
        
        功能说明：
        - 围绕pandas-ta函数构建指标类
        - 需要安装pandas-ta库（https://github.com/twopirllc/pandas-ta）
        - 自动解析函数配置并创建指标
        
        参数说明：
        - func_name: pandas-ta函数名称
        - parse_kwargs: 传递给parse_pandas_ta_config的关键字参数
        - init_kwargs: 传递给IndicatorFactory的关键字参数
        - **kwargs: 传递给IndicatorFactory.from_custom_func的关键字参数
        
        返回值：
        - 返回指标类
        
        使用示例：
            ```pycon
            >>> SMA = vbt.IndicatorFactory.from_pandas_ta('SMA')

            >>> sma = SMA.run(price, length=[2, 3])
            >>> sma.sma
            sma_length         2         3
                          a    b    a    b
            2020-01-01  NaN  NaN  NaN  NaN
            2020-01-02  1.5  4.5  NaN  NaN
            2020-01-03  2.5  3.5  2.0  4.0
            2020-01-04  3.5  2.5  3.0  3.0
            2020-01-05  4.5  1.5  4.0  2.0
            ```

            * To get help on running the indicator, use the `help` command:

            ```pycon
            >>> help(SMA.run)
            Help on method run:

            run(close, length=None, offset=None, short_name='sma', hide_params=None, hide_default=True, **kwargs) method of builtins.type instance
                Run `SMA` indicator.

                * Inputs: `close`
                * 参数: `length`, `offset`
                * 输出: `sma`

                传递参数名称列表作为 `hide_params` 来隐藏它们的列级别。
                设置 `hide_default` 为 False 以显示具有默认值的参数的列级别。

                其他关键字参数将传递给 `vectorbt.indicators.factory.run_pipeline`。
            ```

            * To get the indicator docstring, use the `help` command or print the `__doc__` attribute:

            ```pycon
            >>> print(SMA.__doc__)
            Simple Moving Average (SMA)

            The Simple Moving Average is the classic moving average that is the equally
            weighted average over n periods.

            Sources:
                https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/simple-moving-average-sma/

            Calculation:
                Default Inputs:
                    length=10
                SMA = SUM(close, length) / length

            Args:
                close (pd.Series): Series of 'close's
                length (int): It's period. Default: 10
                offset (int): How many periods to offset the result. Default: 0

            Kwargs:
                adjust (bool): Default: True
                presma (bool, optional): If True, uses SMA for initial value.
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method

            Returns:
                pd.Series: New feature generated.
            ```
        """
        # 导入pandas-ta库
        import pandas_ta

        # 转换函数名为小写（pandas-ta约定）
        func_name = func_name.lower()
        pandas_ta_func = getattr(pandas_ta, func_name)

        # 设置默认的解析参数
        if parse_kwargs is None:
            parse_kwargs = {}
        
        # 解析pandas-ta函数配置
        config = cls.parse_pandas_ta_config(pandas_ta_func, **parse_kwargs)

        def apply_func(input_list: tp.List[tp.SeriesFrame],
                       in_output_tuple: tp.Tuple[tp.SeriesFrame, ...],
                       param_tuple: tp.Tuple[tp.Param, ...],
                       **kwargs) -> tp.Union[tp.Array2d, tp.List[tp.Array2d]]:
            """
            应用pandas-ta函数的内部函数
            
            参数：
            - input_list: 输入数据列表
            - in_output_tuple: 就地输出元组
            - param_tuple: 参数元组
            - **kwargs: 其他关键字参数
            
            返回：
            - 2D数组或2D数组列表
            """
            # 判断输入是否为Series
            is_series = isinstance(input_list[0], pd.Series)
            n_input_cols = 1 if is_series else len(input_list[0].columns)
            outputs = []
            
            # 逐列处理数据
            for col in range(n_input_cols):
                # 调用pandas-ta函数
                output = pandas_ta_func(
                    **{
                        name: input_list[i] if is_series else input_list[i].iloc[:, col]
                        for i, name in enumerate(config['input_names'])
                    },
                    **{
                        name: param_tuple[i]
                        for i, name in enumerate(config['param_names'])
                    },
                    **kwargs
                )
                
                # 处理元组输出
                if isinstance(output, tuple):
                    _outputs = []
                    for o in output:
                        if pd.Index.equals(input_list[0].index, o.index):
                            _outputs.append(o)
                    if len(_outputs) > 1:
                        output = pd.concat(_outputs, axis=1)
                    elif len(_outputs) == 1:
                        output = _outputs[0]
                    else:
                        raise ValueError("No valid outputs were returned")
                
                # 处理DataFrame输出
                if isinstance(output, pd.DataFrame):
                    output = tuple([output.iloc[:, i] for i in range(len(output.columns))])
                outputs.append(output)
            
            # 处理多输出情况
            if isinstance(outputs[0], tuple):  # 多输出
                outputs = list(zip(*outputs))
                return list(map(np.column_stack, outputs))
            # 单输出：堆叠列
            return np.column_stack(outputs)

        # 提取默认参数
        defaults = config.pop('defaults')
        
        # 创建pandas-ta指标类
        PTAIndicator = cls(
            **merge_dicts(
                config,  # 指标配置
                init_kwargs  # 用户提供的初始化参数
            )
        ).from_apply_func(
            apply_func,
            pass_packed=True,  # 传递打包的参数
            keep_pd=True,  # 保持pandas格式
            to_2d=False,  # 不转换为2D
            **defaults,  # 默认参数
            **kwargs  # 其他参数
        )
        return PTAIndicator

    @classmethod
    def get_ta_indicators(cls) -> tp.Set[str]:
        """
        获取所有ta库指标
        
        返回值：
        - 包含所有ta库指标类名称的集合
        """
        import ta

        # 获取ta库中所有模块名称
        ta_module_names = [k for k in dir(ta) if isinstance(getattr(ta, k), ModuleType)]
        indicators = set()
        
        # 遍历所有模块
        for module_name in ta_module_names:
            module = getattr(ta, module_name)
            # 遍历模块中的所有对象
            for name in dir(module):
                obj = getattr(module, name)
                # 检查是否为指标类
                if isinstance(obj, type) \
                        and obj != ta.utils.IndicatorMixin \
                        and issubclass(obj, ta.utils.IndicatorMixin):
                    indicators.add(obj.__name__)
        return indicators

    @classmethod
    def find_ta_indicator(cls, cls_name: str) -> IndicatorMixinT:
        """
        根据名称查找ta指标类
        
        参数：
        - cls_name: 指标类名称
        
        返回值：
        - ta指标类对象
        
        异常：
        - ValueError: 如果找不到指定的指标类
        """
        import ta

        # 获取ta库中所有模块名称
        ta_module_names = [k for k in dir(ta) if isinstance(getattr(ta, k), ModuleType)]
        
        # 在所有模块中查找指标类
        for module_name in ta_module_names:
            module = getattr(ta, module_name)
            if cls_name in dir(module):
                return getattr(module, cls_name)
        
        # 如果没有找到，抛出异常
        raise ValueError(f"Indicator \"{cls_name}\" not found")

    @classmethod
    def parse_ta_config(cls, ind_cls: IndicatorMixinT) -> tp.Kwargs:
        """
        解析ta指标的配置
        
        参数：
        - ind_cls: ta指标类
        
        返回值：
        - 包含指标配置的字典
        
        功能：
        - 解析指标的输入参数、参数名称和输出名称
        - 从指标类的__init__方法签名中提取信息
        """
        # 初始化配置信息
        input_names = []
        param_names = []
        defaults = {}
        output_names = []

        # 解析指标类的__init__方法签名以获取输入名称
        sig = inspect.signature(ind_cls)
        # 遍历所有参数
        for k, v in sig.parameters.items():
            # 跳过可变参数
            if v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD):
                # 检查参数是否有注解
                if v.annotation == inspect.Parameter.empty:
                    raise ValueError(f"Argument \"{k}\" has no annotation")
                
                # 根据注解类型分类参数
                if v.annotation == pd.Series:
                    input_names.append(k)  # pd.Series注解的参数为输入
                else:
                    param_names.append(k)  # 其他注解的参数为指标参数
                    if v.default != inspect.Parameter.empty:
                        defaults[k] = v.default

        # 通过检查实例方法获取输出名称
        for attr in dir(ind_cls):
            if not attr.startswith('_'):  # 跳过私有属性
                # 检查方法返回类型注解
                if inspect.signature(getattr(ind_cls, attr)).return_annotation == pd.Series:
                    output_names.append(attr)
                # 检查方法文档字符串
                elif 'Returns:\n            pandas.Series' in getattr(ind_cls, attr).__doc__:
                    output_names.append(attr)

        # 返回配置字典
        return dict(
            class_name=ind_cls.__name__,
            class_docstring=ind_cls.__doc__,
            input_names=input_names,
            param_names=param_names,
            output_names=output_names,
            defaults=defaults
        )

    @classmethod
    def from_ta(cls, cls_name: str, init_kwargs: tp.KwargsLike = None, **kwargs) -> tp.Type[IndicatorBase]:
        """
        基于ta类构建指标类
        
        功能说明：
        - 围绕ta类构建指标类
        - 需要安装ta库（https://github.com/bukosabino/ta）
        - 自动解析类配置并创建指标
        
        参数说明：
        - cls_name: ta类名称
        - init_kwargs: 传递给IndicatorFactory的关键字参数
        - **kwargs: 传递给IndicatorFactory.from_custom_func的关键字参数
        
        返回值：
        - 返回指标类
        
        使用示例：
            ```pycon
            >>> SMAIndicator = vbt.IndicatorFactory.from_ta('SMAIndicator')

            >>> sma = SMAIndicator.run(price, window=[2, 3])
            >>> sma.sma_indicator
            smaindicator_window    2         3
                                   a    b    a    b
            2020-01-01           NaN  NaN  NaN  NaN
            2020-01-02           1.5  4.5  NaN  NaN
            2020-01-03           2.5  3.5  2.0  4.0
            2020-01-04           3.5  2.5  3.0  3.0
            2020-01-05           4.5  1.5  4.0  2.0
            ```

            * To get help on running the indicator, use the `help` command:

            ```pycon
            >>> help(SMAIndicator.run)
            Help on method run:

            run(close, window, fillna=False, short_name='smaindicator', hide_params=None, hide_default=True, **kwargs) method of builtins.type instance
                Run `SMAIndicator` indicator.

                * Inputs: `close`
                * 参数: `window`, `fillna`
                * 输出: `sma_indicator`

                传递参数名称列表作为 `hide_params` 来隐藏它们的列级别。
                设置 `hide_default` 为 False 以显示具有默认值的参数的列级别。

                其他关键字参数将传递给 `vectorbt.indicators.factory.run_pipeline`。
            ```

            * To get the indicator docstring, use the `help` command or print the `__doc__` attribute:

            ```pycon
            >>> print(SMAIndicator.__doc__)
            SMA - Simple Moving Average

                Args:
                    close(pandas.Series): dataset 'Close' column.
                    window(int): n period.
                    fillna(bool): if True, fill nan values.
            ```
        """
        # 查找ta指标类
        ind_cls = cls.find_ta_indicator(cls_name)
        
        # 解析ta指标配置
        config = cls.parse_ta_config(ind_cls)

        def apply_func(input_list: tp.List[tp.SeriesFrame],
                       in_output_tuple: tp.Tuple[tp.SeriesFrame, ...],
                       param_tuple: tp.Tuple[tp.Param, ...],
                       **kwargs) -> tp.Union[tp.Array2d, tp.List[tp.Array2d]]:
            """
            应用ta指标的内部函数
            
            参数：
            - input_list: 输入数据列表
            - in_output_tuple: 就地输出元组
            - param_tuple: 参数元组
            - **kwargs: 其他关键字参数
            
            返回：
            - 2D数组或2D数组列表
            """
            # 判断输入是否为Series
            is_series = isinstance(input_list[0], pd.Series)
            n_input_cols = 1 if is_series else len(input_list[0].columns)
            outputs = []
            
            # 逐列处理数据
            for col in range(n_input_cols):
                # 创建ta指标实例
                ind = ind_cls(
                    **{
                        name: input_list[i] if is_series else input_list[i].iloc[:, col]
                        for i, name in enumerate(config['input_names'])
                    },
                    **{
                        name: param_tuple[i]
                        for i, name in enumerate(config['param_names'])
                    },
                    **kwargs
                )
                
                # 获取所有输出
                output = []
                for output_name in config['output_names']:
                    output.append(getattr(ind, output_name)())
                
                # 处理单/多输出
                if len(output) == 1:
                    output = output[0]
                else:
                    output = tuple(output)
                outputs.append(output)
            
            # 处理多输出情况
            if isinstance(outputs[0], tuple):  # 多输出
                outputs = list(zip(*outputs))
                return list(map(np.column_stack, outputs))
            # 单输出：堆叠列
            return np.column_stack(outputs)

        # 提取默认参数
        defaults = config.pop('defaults')
        
        # 创建ta指标类
        TAIndicator = cls(
            **merge_dicts(
                config,  # 指标配置
                init_kwargs  # 用户提供的初始化参数
            )
        ).from_apply_func(
            apply_func,
            pass_packed=True,  # 传递打包的参数
            keep_pd=True,  # 保持pandas格式
            to_2d=False,  # 不转换为2D
            **defaults,  # 默认参数
            **kwargs  # 其他参数
        )
        return TAIndicator