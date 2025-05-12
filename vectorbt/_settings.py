# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""全局设置模块

本模块定义了vectorbt库的全局配置系统，提供了一个层次化的配置结构，允许用户自定义库的行为。
配置系统采用嵌套字典的形式，支持针对不同子包、模块和类的个性化设置，并提供了灵活的读取和修改机制。
所有vectorbt内部组件都会读取这里定义的默认参数来控制其行为，用户可以通过修改这些设置来自定义库的行为而无需修改代码。

`settings`配置也可以通过`vectorbt.settings`访问.

以下是`settings`配置的主要特性:

* 它是一个嵌套配置，即一个由多个子配置组成的配置，
    每个子包（如'data'）、模块（如'array_wrapper'）甚至类（如'configured'）各有一个。
    每个子配置可能由其他子配置组成。
* 它具有冻结的键 - 你不能添加其他子配置或删除现有的子配置，但你可以修改它们。
* 每个子配置可以通过使用`dict`继承父配置的属性，或者
    通过使用自己的`vectorbt.utils.config.Config`覆盖它们。定义自己的配置的主要原因是允许
    添加新的键（例如，'plotting.layout'）。

例如，你可以更改每个图表的默认宽度和高度:

```pycon
>>> import vectorbt as vbt

>>> vbt.settings['plotting']['layout']['width'] = 800
>>> vbt.settings['plotting']['layout']['height'] = 400
```
    
主子配置（如绘图）也可以使用点符号访问/修改：

```
>>> vbt.settings.plotting['layout']['width'] = 800
```

一些子配置也允许使用点符号，但这取决于它们是否继承了根配置的规则。

```plaintext
>>> vbt.settings.data - ok
>>> vbt.settings.data.binance - ok
>>> vbt.settings.data.binance.api_key - error
>>> vbt.settings.data.binance['api_key'] - ok
```

由于这只在查看源代码时可见，建议始终使用括号符号。

!!! 注意
    任何更改都会立即生效。但它是否立即反映取决于访问设置的地方。例如，更改'array_wrapper.freq'会立即生效，因为
    每次调用`vectorbt.base.array_wrapper.ArrayWrapper.freq`时都会解析该值。
    另一方面，更改'portfolio.fillna_close'只对将来创建的`vectorbt.portfolio.base.Portfolio`
    实例有效，而不是现有的实例，因为该值是在构造时解析的。
    但大多数情况下，你仍然可以通过使用
    `vectorbt.portfolio.base.Portfolio.replace`替换实例来强制更新默认值。

    vectorbt中的所有地方都从`vectorbt._settings.settings`导入`settings`，而不是从`vectorbt`导入。
    覆盖`vectorbt.settings`只覆盖为用户创建的引用。
    考虑更新设置配置而不是替换它。

## 保存

像任何其他继承`vectorbt.utils.config.Config`的类一样，我们可以将设置保存到磁盘，
加载回来，并就地更新：

```pycon
>>> vbt.settings.save('my_settings')
>>> vbt.settings['caching']['enabled'] = False
>>> vbt.settings['caching']['enabled']
False

>>> vbt.settings.load_update('my_settings')  # load() would return a new object!
>>> vbt.settings['caching']['enabled']
True
```

附加功能：你可以对`settings`内的任何子配置执行相同的操作！
"""

# 导入必要的库
import json  # 用于JSON数据处理
import pkgutil  # 用于访问包中的资源文件

# 导入第三方库
import numpy as np  # 数值计算库
import plotly.graph_objects as go  # Plotly图形对象
import plotly.io as pio  # Plotly输入/输出操作

# 导入vectorbt内部模块
from vectorbt.base.array_wrapper import ArrayWrapper  # 数组包装器
from vectorbt.base.column_grouper import ColumnGrouper  # 列分组器
from vectorbt.records.col_mapper import ColumnMapper  # 列映射器
from vectorbt.utils.config import Config  # 配置基类
from vectorbt.utils.datetime_ import get_local_tz, get_utc_tz  # 时区工具函数
from vectorbt.utils.decorators import CacheCondition  # 缓存条件装饰器
from vectorbt.utils.template import Sub, RepEval  # 模板工具

__pdoc__ = {}  # 用于存储文档字符串


class SettingsConfig(Config):
    """扩展Config类，专用于全局设置的配置类。
    
    该类继承自vectorbt.utils.config.Config，添加了特定于主题和模板管理的方法，
    用于管理vectorbt的全局设置和可视化主题。
    """

    def register_template(self, theme: str) -> None:
        """注册指定主题的模板。
        
        参数:
            theme: str - 主题名称
        """
        pio.templates['vbt_' + theme] = go.layout.Template(self['plotting']['themes'][theme]['template'])

    def register_templates(self) -> None:
        """注册所有主题的模板。
        
        遍历所有已定义的主题并注册它们的模板。
        """
        for theme in self['plotting']['themes']:
            self.register_template(theme)

    def set_theme(self, theme: str) -> None:
        """设置默认主题。
        
        参数:
            theme: str - 要设置为默认的主题名称
        """
        self.register_template(theme)
        self['plotting']['color_schema'].update(self['plotting']['themes'][theme]['color_schema'])
        self['plotting']['layout']['template'] = 'vbt_' + theme

    def reset_theme(self) -> None:
        """重置为默认主题('light')。"""
        self.set_theme('light')


# 创建全局设置实例
settings = SettingsConfig(
    dict(
        # Numba相关设置
        numba=dict(
            check_func_type=True,  # 是否检查函数类型
            check_func_suffix=False  # 是否检查函数后缀
        ),
        # 配置模块设置
        config=Config(),  # 灵活配置
        # 已配置对象的设置
        configured=dict(
            config=Config(  # 灵活配置
                dict(
                    readonly=True  # 默认设置为只读模式
                )
            ),
        ),
        # 缓存相关设置
        caching=dict(
            enabled=True,  # 启用缓存
            whitelist=[  # 缓存白名单
                CacheCondition(base_cls=ArrayWrapper),  # 缓存ArrayWrapper类
                CacheCondition(base_cls=ColumnGrouper),  # 缓存ColumnGrouper类
                CacheCondition(base_cls=ColumnMapper)  # 缓存ColumnMapper类
            ],
            blacklist=[]  # 缓存黑名单，默认为空
        ),
        # 数据广播设置
        broadcasting=dict(
            align_index=False,  # 不对齐索引
            align_columns=True,  # 对齐列
            index_from='strict',  # 索引来源策略为严格模式
            columns_from='stack',  # 列来源策略为堆叠模式
            ignore_sr_names=True,  # 忽略Series名称
            drop_duplicates=True,  # 删除重复项
            keep='last',  # 保留最后一个重复项
            drop_redundant=True,  # 删除冗余列
            ignore_default=True  # 忽略默认值
        ),
        # 数组包装器设置
        array_wrapper=dict(
            column_only_select=False,  # 不仅选择列
            group_select=True,  # 启用组选择
            freq=None,  # 频率默认为None
            silence_warnings=False  # 不静默警告
        ),
        # 日期时间设置
        datetime=dict(
            naive_tz=get_local_tz(),  # 朴素时区使用本地时区
            to_py_timezone=True  # 转换为Python时区对象
        ),
        # 数据获取设置
        data=dict(
            tz_localize=get_utc_tz(),  # 时区本地化为UTC
            tz_convert=get_utc_tz(),  # 时区转换为UTC
            missing_index='nan',  # 缺失索引处理策略为'nan'
            missing_columns='raise',  # 缺失列处理策略为抛出异常
            alpaca=Config(  # Alpaca API设置
                dict(
                    api_key=None,  # API密钥
                    secret_key=None  # 密钥
                )
            ),
            binance=Config(  # Binance API设置
                dict(
                    api_key=None,  # API密钥
                    api_secret=None  # API密钥
                )
            ),
            ccxt=Config(  # CCXT设置
                dict(
                    enableRateLimit=True  # 启用速率限制
                )
            ),
            stats=Config(),  # 数据统计设置
            plots=Config()  # 数据绘图设置
        ),
        # 绘图设置
        plotting=dict(
            use_widgets=True,  # 使用交互式小部件
            show_kwargs=Config(),  # 显示参数
            color_schema=Config(  # 颜色方案
                dict(
                    increasing="#1b9e76",  # 上升颜色
                    decreasing="#d95f02"  # 下降颜色
                )
            ),
            contrast_color_schema=Config(  # 对比颜色方案
                dict(
                    blue="#4285F4",  # 蓝色
                    orange="#FFAA00",  # 橙色
                    green="#37B13F",  # 绿色
                    red="#EA4335",  # 红色
                    gray="#E2E2E2"  # 灰色
                )
            ),
            themes=dict(  # 主题设置
                light=dict(  # 亮色主题
                    color_schema=Config(  # 颜色方案
                        dict(
                            blue="#1f77b4",  # 蓝色
                            orange="#ff7f0e",  # 橙色
                            green="#2ca02c",  # 绿色
                            red="#dc3912",  # 红色
                            purple="#9467bd",  # 紫色
                            brown="#8c564b",  # 棕色
                            pink="#e377c2",  # 粉色
                            gray="#7f7f7f",  # 灰色
                            yellow="#bcbd22",  # 黄色
                            cyan="#17becf"  # 青色
                        )
                    ),
                    template=Config(json.loads(pkgutil.get_data(__name__, "templates/light.json"))),  # 亮色模板
                ),
                dark=dict(  # 暗色主题
                    color_schema=Config(  # 颜色方案
                        dict(
                            blue="#1f77b4",  # 蓝色
                            orange="#ff7f0e",  # 橙色
                            green="#2ca02c",  # 绿色
                            red="#dc3912",  # 红色
                            purple="#9467bd",  # 紫色
                            brown="#8c564b",  # 棕色
                            pink="#e377c2",  # 粉色
                            gray="#7f7f7f",  # 灰色
                            yellow="#bcbd22",  # 黄色
                            cyan="#17becf"  # 青色
                        )
                    ),
                    template=Config(json.loads(pkgutil.get_data(__name__, "templates/dark.json"))),  # 暗色模板
                ),
                seaborn=dict(  # Seaborn风格主题
                    color_schema=Config(  # 颜色方案
                        dict(
                            blue="#1f77b4",  # 蓝色
                            orange="#ff7f0e",  # 橙色
                            green="#2ca02c",  # 绿色
                            red="#dc3912",  # 红色
                            purple="#9467bd",  # 紫色
                            brown="#8c564b",  # 棕色
                            pink="#e377c2",  # 粉色
                            gray="#7f7f7f",  # 灰色
                            yellow="#bcbd22",  # 黄色
                            cyan="#17becf"  # 青色
                        )
                    ),
                    template=Config(json.loads(pkgutil.get_data(__name__, "templates/seaborn.json"))),  # Seaborn模板
                ),
            ),
            layout=Config(  # 布局设置
                dict(
                    width=700,  # 宽度
                    height=350,  # 高度
                    margin=dict(  # 边距
                        t=30, b=30, l=30, r=30  # 上下左右边距
                    ),
                    legend=dict(  # 图例设置
                        orientation="h",  # 水平方向
                        yanchor="bottom",  # y锚点位置
                        y=1.02,  # y位置
                        xanchor="right",  # x锚点位置
                        x=1,  # x位置
                        traceorder='normal'  # 轨迹顺序
                    )
                )
            ),
        ),
        # 统计构建器设置
        stats_builder=dict(
            metrics='all',  # 包含所有指标
            tags='all',  # 包含所有标签
            silence_warnings=False,  # 不静默警告
            template_mapping=Config(),  # 模板映射
            filters=Config(  # 过滤器设置
                dict(
                    is_not_grouped=dict(  # 未分组数据过滤器
                        filter_func=lambda self, metric_settings:
                        not self.wrapper.grouper.is_grouped(group_by=metric_settings['group_by']),
                        warning_message=Sub("Metric '$metric_name' does not support grouped data")  # 警告消息
                    ),
                    has_freq=dict(  # 有频率过滤器
                        filter_func=lambda self, metric_settings:
                        self.wrapper.freq is not None,
                        warning_message=Sub("Metric '$metric_name' requires frequency to be set")  # 警告消息
                    )
                )
            ),
            settings=Config(  # 基本设置
                dict(
                    to_timedelta=None,  # 转换为时间差
                    use_caching=True  # 使用缓存
                )
            ),
            metric_settings=Config(),  # 指标设置
        ),
        # 绘图构建器设置
        plots_builder=dict(
            subplots='all',  # 包含所有子图
            tags='all',  # 包含所有标签
            silence_warnings=False,  # 不静默警告
            template_mapping=Config(),  # 模板映射
            filters=Config(  # 过滤器设置
                dict(
                    is_not_grouped=dict(  # 未分组数据过滤器
                        filter_func=lambda self, subplot_settings:
                        not self.wrapper.grouper.is_grouped(group_by=subplot_settings['group_by']),
                        warning_message=Sub("Subplot '$subplot_name' does not support grouped data")  # 警告消息
                    ),
                    has_freq=dict(  # 有频率过滤器
                        filter_func=lambda self, subplot_settings:
                        self.wrapper.freq is not None,
                        warning_message=Sub("Subplot '$subplot_name' requires frequency to be set")  # 警告消息
                    )
                )
            ),
            settings=Config(  # 基本设置
                dict(
                    use_caching=True,  # 使用缓存
                    hline_shape_kwargs=dict(  # 水平线形状参数
                        type='line',  # 类型为线
                        line=dict(  # 线设置
                            color='gray',  # 颜色为灰色
                            dash="dash",  # 虚线样式
                        )
                    )
                )
            ),
            subplot_settings=Config(),  # 子图设置
            show_titles=True,  # 显示标题
            hide_id_labels=True,  # 隐藏ID标签
            group_id_labels=True,  # 分组ID标签
            make_subplots_kwargs=Config(),  # 创建子图参数
            layout_kwargs=Config(),  # 布局参数
        ),
        # 通用模块设置
        generic=dict(
            stats=Config(  # 统计设置
                dict(
                    filters=dict(  # 过滤器
                        has_mapping=dict(  # 有映射过滤器
                            filter_func=lambda self, metric_settings:
                            metric_settings.get('mapping', self.mapping) is not None
                        )
                    ),
                    settings=dict(  # 基本设置
                        incl_all_keys=False  # 不包含所有键
                    )
                )
            ),
            plots=Config()  # 绘图设置
        ),
        # 范围相关设置
        ranges=dict(
            stats=Config(),  # 统计设置
            plots=Config()  # 绘图设置
        ),
        # 回撤相关设置
        drawdowns=dict(
            stats=Config(  # 统计设置
                dict(
                    settings=dict(  # 基本设置
                        incl_active=False  # 不包含活跃回撤
                    )
                )
            ),
            plots=Config()  # 绘图设置
        ),
        # OHLCV数据设置
        ohlcv=dict(
            plot_type='OHLC',  # 绘图类型为OHLC
            column_names=dict(  # 列名映射
                open='Open',  # 开盘价列名
                high='High',  # 最高价列名
                low='Low',  # 最低价列名
                close='Close',  # 收盘价列名
                volume='Volume'  # 成交量列名
            ),
            stats=Config(),  # 统计设置
            plots=Config()  # 绘图设置
        ),
        # 信号相关设置
        signals=dict(
            stats=Config(  # 统计设置
                dict(
                    filters=dict(  # 过滤器
                        silent_has_other=dict(  # 静默有其他过滤器
                            filter_func=lambda self, metric_settings:
                            metric_settings.get('other', None) is not None
                        ),
                    ),
                    settings=dict(  # 基本设置
                        other=None,  # 其他信号
                        other_name='Other',  # 其他信号名称
                        from_other=False  # 不从其他信号计算
                    )
                )
            ),
            plots=Config()  # 绘图设置
        ),
        # 收益率相关设置
        returns=dict(
            year_freq='365 days',  # 年频率为365天
            defaults=Config(  # 默认值
                dict(
                    start_value=0.,  # 起始值
                    window=10,  # 窗口大小
                    minp=None,  # 最小周期数
                    ddof=1,  # 自由度调整
                    risk_free=0.,  # 无风险利率
                    levy_alpha=2.,  # 莱维alpha参数
                    required_return=0.,  # 要求收益率
                    cutoff=0.05  # 截断值
                )
            ),
            stats=Config(  # 统计设置
                dict(
                    filters=dict(  # 过滤器
                        has_year_freq=dict(  # 有年频率过滤器
                            filter_func=lambda self, metric_settings:
                            self.year_freq is not None,
                            warning_message=Sub("Metric '$metric_name' requires year frequency to be set")  # 警告消息
                        ),
                        has_benchmark_rets=dict(  # 有基准收益率过滤器
                            filter_func=lambda self, metric_settings:
                            metric_settings.get('benchmark_rets', self.benchmark_rets) is not None,
                            warning_message=Sub("Metric '$metric_name' requires benchmark_rets to be set")  # 警告消息
                        )
                    ),
                    settings=dict(  # 基本设置
                        check_is_not_grouped=True  # 检查未分组
                    )
                )
            ),
            plots=Config()  # 绘图设置
        ),
        # QuantStats适配器设置
        qs_adapter=dict(
            defaults=Config(),  # 默认值
        ),
        # 记录相关设置
        records=dict(
            stats=Config(),  # 统计设置
            plots=Config()  # 绘图设置
        ),
        # 映射数组设置
        mapped_array=dict(
            stats=Config(  # 统计设置
                dict(
                    filters=dict(  # 过滤器
                        has_mapping=dict(  # 有映射过滤器
                            filter_func=lambda self, metric_settings:
                            metric_settings.get('mapping', self.mapping) is not None
                        )
                    ),
                    settings=dict(  # 基本设置
                        incl_all_keys=False  # 不包含所有键
                    )
                )
            ),
            plots=Config()  # 绘图设置
        ),
        # 订单相关设置
        orders=dict(
            stats=Config(),  # 统计设置
            plots=Config()  # 绘图设置
        ),
        # 交易相关设置
        trades=dict(
            stats=Config(  # 统计设置
                dict(
                    settings=dict(  # 基本设置
                        incl_open=False  # 不包含未平仓交易
                    ),
                    template_mapping=dict(  # 模板映射
                        incl_open_tags=RepEval("['open', 'closed'] if incl_open else ['closed']")  # 包含开放标签
                    )
                )
            ),
            plots=Config()  # 绘图设置
        ),
        # 日志相关设置
        logs=dict(
            stats=Config()  # 统计设置
        ),
        # 投资组合设置
        portfolio=dict(
            call_seq='default',  # 调用顺序为默认
            init_cash=100.,  # 初始资金
            size=np.inf,  # 交易大小
            size_type='amount',  # 大小类型为金额
            fees=0.,  # 费用
            fixed_fees=0.,  # 固定费用
            slippage=0.,  # 滑点
            reject_prob=0.,  # 拒绝概率
            min_size=1e-8,  # 最小交易大小
            max_size=np.inf,  # 最大交易大小
            size_granularity=np.nan,  # 大小粒度
            lock_cash=False,  # 不锁定资金
            allow_partial=True,  # 允许部分执行
            raise_reject=False,  # 不抛出拒绝异常
            val_price=np.inf,  # 估值价格
            accumulate=False,  # 不累积
            sl_stop=np.nan,  # 止损点
            sl_trail=False,  # 不使用追踪止损
            tp_stop=np.nan,  # 止盈点
            stop_entry_price='close',  # 止损入场价为收盘价
            stop_exit_price='stoplimit',  # 止损出场价为止损限价
            stop_conflict_mode='exit',  # 止损冲突模式为出场
            upon_stop_exit='close',  # 止损触发后操作为关闭
            upon_stop_update='override',  # 止损更新模式为覆盖
            use_stops=None,  # 使用止损设置
            log=False,  # 不记录日志
            upon_long_conflict='ignore',  # 多头冲突处理为忽略
            upon_short_conflict='ignore',  # 空头冲突处理为忽略
            upon_dir_conflict='ignore',  # 方向冲突处理为忽略
            upon_opposite_entry='reversereduce',  # 反向入场处理为反转减少
            signal_direction='longonly',  # 信号方向为仅多头
            order_direction='both',  # 订单方向为双向
            cash_sharing=False,  # 不共享资金
            call_pre_segment=False,  # 不调用前置段
            call_post_segment=False,  # 不调用后置段
            ffill_val_price=True,  # 前向填充估值价格
            update_value=False,  # 不更新价值
            fill_pos_record=True,  # 填充持仓记录
            row_wise=False,  # 不按行计算
            flexible=False,  # 不使用灵活模式
            use_numba=True,  # 使用Numba加速
            seed=None,  # 随机种子
            freq=None,  # 频率
            attach_call_seq=False,  # 不附加调用序列
            fillna_close=True,  # 填充缺失收盘价
            trades_type='exittrades',  # 交易类型为出场交易
            stats=Config(  # 统计设置
                dict(
                    filters=dict(  # 过滤器
                        has_year_freq=dict(  # 有年频率过滤器
                            filter_func=lambda self, metric_settings:
                            metric_settings['year_freq'] is not None,
                            warning_message=Sub("Metric '$metric_name' requires year frequency to be set")  # 警告消息
                        )
                    ),
                    settings=dict(  # 基本设置
                        use_asset_returns=False,  # 不使用资产收益率
                        incl_open=False  # 不包含未平仓交易
                    ),
                    template_mapping=dict(  # 模板映射
                        incl_open_tags=RepEval("['open', 'closed'] if incl_open else ['closed']")  # 包含开放标签
                    )
                )
            ),
            plots=Config(  # 绘图设置
                dict(
                    subplots=['orders', 'trade_pnl', 'cum_returns'],  # 子图列表
                    settings=dict(  # 基本设置
                        use_asset_returns=False  # 不使用资产收益率
                    )
                )
            )
        ),
        # 消息传递设置
        messaging=dict(
            telegram=Config(  # Telegram设置
                dict(
                    token=None,  # 令牌
                    use_context=True,  # 使用上下文
                    persistence='telegram_bot.pickle',  # 持久化文件
                    defaults=Config(),  # 默认值
                    drop_pending_updates=True  # 丢弃挂起的更新
                )
            ),
            giphy=dict(  # GIPHY设置
                api_key=None,  # API密钥
                weirdness=5  # 怪异度
            ),
        ),
    ),
    copy_kwargs=dict(  # 复制参数
        copy_mode='deep'  # 深拷贝模式
    ),
    frozen_keys=True,  # 冻结键
    nested=True,  # 嵌套模式
    convert_dicts=Config  # 转换字典为Config对象
)
"""全局设置对象"""

# 初始化设置
settings.reset_theme()  # 重置为默认主题
settings.make_checkpoint()  # 创建检查点
settings.register_templates()  # 注册所有模板

# 文档字符串设置
__pdoc__['settings'] = f"""全局设置配置对象。

__numba__

应用于Numba的设置。

```json
{settings['numba'].to_doc()}
```

__config__

应用于vectorbt.utils.config.Config的设置。

```json
{settings['config'].to_doc()}
```

__configured__

应用于vectorbt.utils.config.Configured的设置。

```json
{settings['configured'].to_doc()}
```

__caching__

应用于vectorbt.utils.decorators的设置。

参见vectorbt.utils.decorators.should_cache。

```json
{settings['caching'].to_doc()}
```

__broadcasting__

应用于vectorbt.base.reshape_fns的设置。

```json
{settings['broadcasting'].to_doc()}
```

__array_wrapper__

应用于vectorbt.base.array_wrapper.ArrayWrapper的设置。

```json
{settings['array_wrapper'].to_doc()}
```

__datetime__

应用于vectorbt.utils.datetime_的设置。

```json
{settings['datetime'].to_doc()}
```

__data__

应用于vectorbt.data的设置。

```json
{settings['data'].to_doc()}
```
    
* binance:
    参见binance.client.Client。

* ccxt:
    参见[配置API密钥](https://ccxt.readthedocs.io/en/latest/manual.html#configuring-api-keys)。
    可以为每个交易所定义密钥。如果在根级别定义了密钥，则适用于所有交易所。

__plotting__

应用于绘制Plotly图形的设置。

```json
{settings['plotting'].to_doc(replace={
    'settings.plotting.themes.light.template': "{ ... templates/light.json ... }",
    'settings.plotting.themes.dark.template': "{ ... templates/dark.json ... }",
    'settings.plotting.themes.seaborn.template': "{ ... templates/seaborn.json ... }"
}, path='settings.plotting')}
```

__stats_builder__

应用于vectorbt.generic.stats_builder.StatsBuilderMixin的设置。

```json
{settings['stats_builder'].to_doc()}
```

__plots_builder__

应用于vectorbt.generic.plots_builder.PlotsBuilderMixin的设置。

```json
{settings['plots_builder'].to_doc()}
```

__generic__

应用于vectorbt.generic的设置。

```json
{settings['generic'].to_doc()}
```

__ranges__

应用于vectorbt.generic.ranges的设置。

```json
{settings['ranges'].to_doc()}
```

__drawdowns__

应用于vectorbt.generic.drawdowns的设置。

```json
{settings['drawdowns'].to_doc()}
```

__ohlcv__

应用于vectorbt.ohlcv_accessors的设置。

```json
{settings['ohlcv'].to_doc()}
```

__signals__

应用于vectorbt.signals的设置。

```json
{settings['signals'].to_doc()}
```

__returns__

应用于vectorbt.returns的设置。

```json
{settings['returns'].to_doc()}
```

__qs_adapter__

应用于vectorbt.returns.qs_adapter的设置。

```json
{settings['qs_adapter'].to_doc()}
```

__records__

应用于vectorbt.records.base的设置。

```json
{settings['records'].to_doc()}
```

__mapped_array__

应用于vectorbt.records.mapped_array的设置。

```json
{settings['mapped_array'].to_doc()}
```

__orders__

应用于vectorbt.portfolio.orders的设置。

```json
{settings['orders'].to_doc()}
```

__trades__

应用于vectorbt.portfolio.trades的设置。

```json
{settings['trades'].to_doc()}
```

__logs__

应用于vectorbt.portfolio.logs的设置。

```json
{settings['logs'].to_doc()}
```

__portfolio__

应用于vectorbt.portfolio.base.Portfolio的设置。

```json
{settings['portfolio'].to_doc()}
```

__messaging__
    
应用于vectorbt.messaging的设置。

```json
{settings['messaging'].to_doc()}
```

* telegram:
    应用于[python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)的设置。
    
    将`persistence`设置为字符串，以在telegram.ext.PicklePersistence中用作`filename`。
    对于`defaults`，参见telegram.ext.Defaults。其他设置将分布在
    telegram.ext.Updater和telegram.ext.updater.Updater.start_polling中。

* giphy:
    应用于[GIPHY Translate Endpoint](https://developers.giphy.com/docs/api/endpoint#translate)的设置。
"""
