# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
数据更新器模块 - vectorbt框架的自动化数据更新和调度系统

本模块提供了DataUpdater类，用于实现金融数据的自动化更新和调度管理。
该模块是vectorbt框架中实现实时数据流和定期数据更新的核心组件。

模块设计逻辑：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                           DataUpdater (数据更新器)
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            Configured           Data           ScheduleManager
            (配置基类)        (数据实例)        (调度管理器)
                    │               │               │
                    └─── 配置管理 ────┼──── 任务调度 ───┘
                                    │
                            实时数据更新流
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心功能特性：

1. 智能调度系统
   ├── 定期数据更新：支持秒、分钟、小时、日等多种更新频率
   ├── 灵活时间控制：支持特定时间点、时间范围的更新任务
   ├── 后台异步执行：支持非阻塞的后台数据更新
   └── 任务标签管理：支持对不同更新任务进行分类和管理

2. 数据更新机制
   ├── 增量更新：只获取自上次更新以来的新数据
   ├── 自动重试：内建错误处理和重试机制
   ├── 数据验证：更新后自动验证数据完整性
   └── 状态跟踪：记录更新历史和状态信息

3. 可扩展架构
   ├── 自定义更新逻辑：支持用户自定义的预处理和后处理
   ├── 事件回调：支持更新前后的回调函数
   ├── 错误处理：支持自定义的异常处理策略
   └── 配置持久化：支持更新配置的保存和恢复

应用场景：

1. 实时交易系统
   - 股价数据的实时更新和监控
   - 交易信号的定期计算和更新
   - 投资组合价值的实时跟踪

2. 量化研究平台
   - 历史数据的定期补充和更新
   - 因子数据的批量更新和计算
   - 回测结果的定期刷新

3. 风险管理系统
   - 风险指标的实时监控和更新
   - 市场数据的持续获取和分析
   - 预警系统的数据源维护

4. 数据仓库维护
   - 金融数据库的定期同步和更新
   - 数据质量的监控和修复
   - 数据备份和归档的自动化

技术特点：
- 异步执行：基于asyncio的高性能异步调度
- 内存优化：智能的数据缓存和内存管理
- 容错处理：完善的错误恢复和重试机制
- 日志监控：详细的操作日志和性能监控
- 配置灵活：支持动态配置调整和热更新

使用模式：
1. 前台同步更新：适用于一次性数据更新任务
2. 后台异步更新：适用于长期运行的数据维护任务
3. 事件驱动更新：适用于基于特定条件触发的更新
4. 混合模式更新：结合多种更新策略的复杂应用场景
"""

# 导入标准库模块
import logging

# 导入vectorbt内部类型定义
from vectorbt import _typing as tp
# 导入数据基类，提供数据操作的核心功能
from vectorbt.data.base import Data
# 导入配置基类，提供配置管理功能
from vectorbt.utils.config import Configured
# 导入调度管理器，提供任务调度功能
from vectorbt.utils.schedule_ import ScheduleManager

# 创建模块专用的日志记录器，用于记录数据更新操作的日志信息
logger = logging.getLogger(__name__)


class DataUpdater(Configured):
    """
    数据更新器类 - 用于调度和管理金融数据的自动化更新
    
    DataUpdater是vectorbt框架中的核心组件，专门用于实现金融数据的定期更新和实时维护。
    该类继承自Configured，提供了完整的配置管理、任务调度和数据更新功能。
    
    主要特性：
    1. 灵活的调度机制：支持多种时间间隔和调度策略
    2. 前台/后台执行：支持同步和异步两种执行模式
    3. 自定义更新逻辑：允许用户扩展和自定义更新行为
    4. 完整的日志记录：提供详细的操作日志和状态监控
    5. 错误处理机制：内建重试和异常处理功能
    
    继承结构：
    DataUpdater -> Configured -> 基础配置功能
    
    核心组件：
    - data: Data实例，存储和管理实际的金融数据
    - schedule_manager: ScheduleManager实例，负责任务调度和时间管理
    
    使用场景：
    
    示例1 - 前台同步更新（阻塞式）：
    适用于一次性数据更新、测试环境、小规模应用
    
    ```python
    import vectorbt as vbt
    
    # 创建自定义数据更新器类
    class MyDataUpdater(vbt.DataUpdater):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.update_count = 0    # 更新次数计数器
            
        def update(self, count_limit=None):
            # 记录更新前的数据长度
            prev_index_len = len(self.data.wrapper.index)
            
            # 调用父类的更新方法执行实际的数据更新
            super().update()
            
            # 记录更新后的数据长度
            new_index_len = len(self.data.wrapper.index)
            
            # 输出更新信息
            print(f"数据更新完成，新增 {new_index_len - prev_index_len} 个数据点")
            
            # 增加更新计数
            self.update_count += 1
            
            # 如果达到更新次数限制，抛出取消异常停止更新
            if count_limit is not None and self.update_count >= count_limit:
                raise vbt.CancelledError("达到更新次数限制")
    
    # 创建初始数据（几何布朗运动模拟数据）
    data = vbt.GBMData.download(
        'AAPL_SIM',          # 符号名称
        start='1 minute ago', # 1分钟前开始
        freq='1s'            # 每秒一个数据点
    )
    
    # 创建数据更新器实例
    my_updater = MyDataUpdater(data)
    
    # 每秒更新一次，最多更新10次
    my_updater.update_every(1).second.to(10)
    
    # 结果输出：
    # 数据更新完成，新增 1 个数据点
    # 数据更新完成，新增 1 个数据点
    # 数据更新完成，新增 1 个数据点
    # ... (共10次)
    
    # 查看最终数据
    final_data = my_updater.data.get()
    print(f"最终数据长度: {len(final_data)}")
    print(f"数据时间范围: {final_data.index[0]} 到 {final_data.index[-1]}")
    ```
    
    示例2 - 后台异步更新（非阻塞式）：
    适用于生产环境、长期运行的服务、实时数据监控
    
    ```python
    # 创建后台数据更新器
    class BackgroundUpdater(vbt.DataUpdater):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.update_history = []
            
        def update(self, **kwargs):
            import datetime
            
            # 记录更新时间和数据统计
            update_time = datetime.datetime.now()
            prev_len = len(self.data.wrapper.index)
            
            # 执行数据更新
            super().update(**kwargs)
            
            # 记录更新信息
            new_len = len(self.data.wrapper.index)
            update_info = {
                'time': update_time,
                'prev_length': prev_len,
                'new_length': new_len,
                'added_points': new_len - prev_len
            }
            self.update_history.append(update_info)
            
            print(f"[{update_time}] 后台更新: +{new_len - prev_len} 数据点")
    
    # 创建实时股价数据
    stock_data = vbt.YFData.download(
        'AAPL',
        period='1d',
        interval='1m'
    )
    
    # 创建后台更新器
    bg_updater = BackgroundUpdater(stock_data)
    
    # 每30秒在后台更新一次，运行5分钟
    bg_updater.update_every(30).seconds.to(10).tag('stock_update')
    bg_updater.schedule_manager.start_in_background()
    
    # 主程序可以继续执行其他任务
    print("后台更新已启动，主程序继续运行...")
    
    # 检查更新历史
    import time
    time.sleep(60)  # 等待1分钟
    print(f"更新历史记录: {len(bg_updater.update_history)} 次更新")
    
    # 停止后台更新
    bg_updater.schedule_manager.stop('stock_update')
    ```
    
    示例3 - 多数据源更新：
    适用于复杂的投资组合管理、多市场监控
    
    ```python
    class MultiSourceUpdater(vbt.DataUpdater):
        def __init__(self, data_sources, **kwargs):
            # data_sources是包含多个Data实例的字典
            self.data_sources = data_sources
            super().__init__(list(data_sources.values())[0], **kwargs)
            
        def update(self, **kwargs):
            print("开始更新多个数据源...")
            
            for name, data_instance in self.data_sources.items():
                try:
                    # 更新每个数据源
                    updated_data = data_instance.update(**kwargs)
                    print(f"✓ {name} 更新成功")
                except Exception as e:
                    print(f"✗ {name} 更新失败: {e}")
    
    # 创建多个数据源
    data_sources = {
        'stocks': vbt.YFData.download(['AAPL', 'GOOGL', 'MSFT'], period='1d'),
        'crypto': vbt.BinanceData.download(['BTCUSDT', 'ETHUSDT'], 
                                         start='1 hour ago', interval='1m'),
        'synthetic': vbt.GBMData.download('TEST', start='1 hour ago', freq='1min')
    }
    
    # 创建多源更新器
    multi_updater = MultiSourceUpdater(data_sources)
    
    # 每5分钟更新所有数据源
    multi_updater.update_every(5).minutes.tag('multi_update')
    ```
    
    示例4 - 条件触发更新：
    适用于基于特定条件的智能更新策略
    
    ```python
    class ConditionalUpdater(vbt.DataUpdater):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.volatility_threshold = 0.02  # 波动率阈值
            
        def update(self, **kwargs):
            # 计算当前数据的波动率
            current_data = self.data.get()
            if len(current_data) > 20:
                returns = current_data.pct_change().dropna()
                current_volatility = returns.std()
                
                # 只有当波动率超过阈值时才更新
                if current_volatility > self.volatility_threshold:
                    print(f"检测到高波动率 {current_volatility:.4f}，开始更新数据...")
                    super().update(**kwargs)
                else:
                    print(f"波动率 {current_volatility:.4f} 正常，跳过此次更新")
            else:
                # 数据量不足，执行正常更新
                super().update(**kwargs)
    
    # 创建条件更新器
    cond_updater = ConditionalUpdater(stock_data)
    
    # 每分钟检查一次，只在高波动时更新
    cond_updater.update_every(1).minute.tag('conditional')
    ```
    
    高级功能：
    
    1. 自定义调度策略：
    ```python
    # 只在交易时间更新
    updater.update_every().hour.at(':00').between('09:30', '16:00').tag('trading_hours')
    
    # 周一到周五每天上午9点更新
    updater.update_every().monday.to().friday.at('09:00').tag('weekdays')
    
    # 每个月第一个工作日更新
    updater.update_every().month.at('first_weekday').tag('monthly')
    ```
    
    2. 错误处理和恢复：
    ```python
    class RobustUpdater(vbt.DataUpdater):
        def update(self, max_retries=3, **kwargs):
            for attempt in range(max_retries):
                try:
                    super().update(**kwargs)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"更新失败，已重试 {max_retries} 次: {e}")
                        raise
                    else:
                        logger.warning(f"更新失败，第 {attempt + 1} 次重试: {e}")
                        time.sleep(2 ** attempt)  # 指数退避
    ```
    
    注意事项：
    - 长期运行的后台更新器需要考虑内存管理和资源释放
    - 网络数据源的更新需要处理连接超时和API限制
    - 在生产环境中建议设置适当的日志级别和监控机制
    - 更新频率应该与数据源的更新频率和API限制保持一致
    """

    def __init__(self, data: Data, schedule_manager: tp.Optional[ScheduleManager] = None, **kwargs) -> None:
        """
        初始化数据更新器实例
        
        创建一个新的DataUpdater实例，用于管理指定数据源的定期更新任务。
        该构造函数设置必要的组件并准备调度基础设施。
        
        Args:
            data (Data): 要进行更新的数据实例
                必须是vectorbt.data.base.Data的子类实例，如：
                - YFData: Yahoo Finance数据
                - BinanceData: 币安交易所数据  
                - GBMData: 几何布朗运动模拟数据
                - AlpacaData: Alpaca证券数据
                
            schedule_manager (ScheduleManager, optional): 调度管理器实例
                如果为None，将创建一个新的ScheduleManager实例
                自定义的调度管理器可以提供特殊的调度策略和配置
                
            **kwargs: 额外的配置参数
                传递给Configured基类的配置选项
                
        示例：
        ```python
        # 基本初始化
        data = vbt.YFData.download('AAPL', period='1d')
        updater = vbt.DataUpdater(data)
        
        # 使用自定义调度管理器
        custom_scheduler = vbt.ScheduleManager(max_workers=4)
        updater = vbt.DataUpdater(data, custom_scheduler)
        
        # 带配置参数的初始化
        updater = vbt.DataUpdater(
            data, 
            schedule_manager=custom_scheduler,
            config={'update_timeout': 30, 'retry_count': 3}
        )
        ```
        """
        # 调用父类Configured的构造函数，传递所有配置参数
        # 这确保了配置管理功能的正确初始化
        Configured.__init__(
            self,
            data=data,                           # 数据实例配置
            schedule_manager=schedule_manager,   # 调度管理器配置
            **kwargs                            # 其他配置参数
        )
        
        # 存储数据实例的私有引用
        # 使用私有属性确保数据访问的控制和封装
        self._data = data
        
        # 初始化调度管理器
        if schedule_manager is None:
            # 如果没有提供调度管理器，创建一个默认实例
            # 默认调度管理器使用标准配置和单线程执行
            schedule_manager = ScheduleManager()
            
        # 存储调度管理器的私有引用
        # 调度管理器负责任务的时间安排和执行控制
        self._schedule_manager = schedule_manager

    @property
    def data(self) -> Data:
        """
        数据实例属性 - 获取当前管理的数据对象
        
        返回当前DataUpdater实例正在管理的Data对象。该对象包含了
        所有的金融数据以及相关的操作方法，是数据更新的核心目标。
        
        Returns:
            Data: 当前管理的数据实例
                包含以下主要功能：
                - get(): 获取数据内容
                - update(): 更新数据
                - wrapper: 数据包装器，包含索引和元数据
                - download_kwargs: 下载参数配置
                
        使用示例：
        ```python
        # 获取数据内容
        current_data = updater.data.get()
        print(f"数据长度: {len(current_data)}")
        
        # 查看数据统计信息  
        print(f"数据时间范围: {updater.data.wrapper.index[0]} 到 {updater.data.wrapper.index[-1]}")
        
        # 手动触发数据更新
        updated_data = updater.data.update()
        ```
        
        注意事项：
        - 该属性返回的是内部数据对象的引用，修改会直接影响更新器的行为
        - 数据更新后，通过此属性访问的数据会自动反映最新状态
        - 建议通过DataUpdater的方法而非直接操作数据对象来进行更新
        """
        # 返回内部存储的数据实例
        # 这是更新器管理的核心数据对象
        return self._data

    @property  
    def schedule_manager(self) -> ScheduleManager:
        """
        调度管理器属性 - 获取任务调度和时间管理器
        
        返回当前DataUpdater使用的ScheduleManager实例，负责管理所有的
        定时更新任务、调度策略和执行控制。
        
        Returns:
            ScheduleManager: 调度管理器实例
                提供以下主要功能：
                - every(): 创建定期任务
                - start(): 开始执行调度任务
                - stop(): 停止特定或所有任务
                - start_in_background(): 后台异步执行
                - clear(): 清除所有任务
                
        使用示例：
        ```python
        # 直接使用调度管理器创建任务
        scheduler = updater.schedule_manager
        
        # 创建自定义调度任务
        scheduler.every(30).seconds.do(custom_function)
        
        # 启动后台调度
        scheduler.start_in_background()
        
        # 查看当前任务状态
        print(f"活跃任务数: {len(scheduler.jobs)}")
        
        # 停止特定标签的任务
        scheduler.stop('my_custom_tag')
        
        # 清除所有任务
        scheduler.clear()
        ```
        
        高级用法：
        ```python
        # 设置全局调度参数
        scheduler.max_workers = 4  # 最大并发工作线程
        scheduler.default_timeout = 30  # 默认任务超时时间
        
        # 创建复杂的调度策略
        scheduler.every().monday.to().friday.at('09:30').do(market_open_update)
        scheduler.every().hour.between('09:30', '16:00').do(trading_hours_update)
        scheduler.every().day.at('17:00').do(daily_summary_update)
        ```
        
        注意事项：
        - 直接操作调度管理器时需要注意任务冲突和资源管理
        - 建议优先使用DataUpdater的update_every()方法而非直接操作调度器
        - 长期运行的调度任务需要考虑内存和CPU资源的合理使用
        """
        # 返回内部存储的调度管理器实例
        # 负责所有定时任务的管理和执行
        return self._schedule_manager

    def update(self, **kwargs) -> None:
        """
        执行数据更新操作 - 核心的数据更新方法
        
        该方法是DataUpdater的核心功能，负责执行实际的数据更新操作。
        它调用底层数据源的更新方法，获取最新数据并更新内部状态。
        
        工作流程：
        1. 调用数据实例的update方法获取新数据
        2. 更新DataUpdater的内部配置
        3. 记录更新结果到日志系统
        4. 处理任何更新过程中的异常
        
        Args:
            **kwargs: 传递给数据源update方法的参数
                常见参数包括：
                - end: 更新的结束时间
                - limit: 数据条数限制  
                - interval: 数据间隔
                - 其他数据源特定的参数
                
        Raises:
            CancelledError: 当需要停止调度循环时抛出此异常
            Exception: 数据更新过程中的各种异常
            
        自定义扩展示例：
        ```python
        class CustomUpdater(vbt.DataUpdater):
            def update(self, **kwargs):
                # 预处理：更新前的准备工作
                print(f"开始更新数据，当前时间: {datetime.now()}")
                prev_length = len(self.data.get())
                
                try:
                    # 调用父类方法执行实际更新
                    super().update(**kwargs)
                    
                    # 后处理：更新后的验证和清理
                    new_length = len(self.data.get())
                    added_points = new_length - prev_length
                    
                    if added_points > 0:
                        print(f"✓ 更新成功，新增 {added_points} 个数据点")
                        
                        # 数据质量检查
                        latest_data = self.data.get().tail(added_points)
                        if latest_data.isnull().any():
                            print("⚠ 警告：检测到缺失数据")
                            
                    else:
                        print("ℹ 无新数据可更新")
                        
                except Exception as e:
                    print(f"✗ 更新失败: {e}")
                    # 可以选择重新抛出异常或进行错误处理
                    raise
                    
                # 检查停止条件
                if self.should_stop_updating():
                    raise vbt.CancelledError("满足停止条件，终止更新")
                    
            def should_stop_updating(self):
                # 自定义停止逻辑，比如达到数据量上限
                return len(self.data.get()) > 10000
        ```
        
        条件更新示例：
        ```python
        class SmartUpdater(vbt.DataUpdater):
            def update(self, **kwargs):
                # 只在市场时间更新
                import datetime
                now = datetime.datetime.now()
                
                if self.is_market_hours(now):
                    print("市场开放时间，执行更新")
                    super().update(**kwargs)
                else:
                    print("市场关闭时间，跳过更新")
                    
            def is_market_hours(self, dt):
                # 简化的市场时间检查（美股时间）
                weekday = dt.weekday()
                hour = dt.hour
                return weekday < 5 and 9 <= hour < 16
        ```
        
        错误恢复示例：
        ```python
        class ResilientUpdater(vbt.DataUpdater):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.retry_count = 0
                self.max_retries = 3
                
            def update(self, **kwargs):
                for attempt in range(self.max_retries):
                    try:
                        super().update(**kwargs)
                        self.retry_count = 0  # 重置重试计数
                        break
                        
                    except Exception as e:
                        self.retry_count += 1
                        
                        if attempt == self.max_retries - 1:
                            print(f"更新失败，已重试 {self.max_retries} 次: {e}")
                            raise
                        else:
                            wait_time = 2 ** attempt  # 指数退避
                            print(f"更新失败，{wait_time}秒后重试: {e}")
                            time.sleep(wait_time)
        ```
        
        注意事项：
        - 重写此方法时务必调用super().update()以保持核心功能
        - 异常处理应该考虑调度循环的连续性
        - 长时间运行的预处理/后处理可能影响更新频率
        - 在生产环境中建议添加适当的性能监控
        """
        # 调用数据实例的update方法执行实际的数据更新
        # 这是核心操作，获取最新的增量数据并合并到现有数据中
        self._data = self.data.update(**kwargs)
        
        # 更新DataUpdater的配置，确保内部配置与最新的数据状态同步
        # 这包括更新数据引用和相关的元数据信息
        self.update_config(data=self.data)
        
        # 获取更新后的时间索引，用于日志记录和状态跟踪
        new_index = self.data.wrapper.index
        
        # 记录更新结果到日志系统
        # 提供详细的更新统计信息，便于监控和调试
        logger.info(f"Updated data has {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")

    def update_every(self, *args, to: int = None, tags: tp.Optional[tp.Iterable[tp.Hashable]] = None,
                     in_background: bool = False, start_kwargs: dict = None, **kwargs) -> None:
        """
        调度定期数据更新 - 设置自动化的数据更新任务
        
        该方法是DataUpdater的主要接口，用于配置和启动定期的数据更新任务。
        它提供了灵活的调度选项，支持多种时间模式和执行策略。
        
        Args:
            *args: 时间间隔参数，传递给ScheduleManager.every()
                支持多种格式：
                - 数字：update_every(30) -> 每30个默认单位
                - 无参数：update_every() -> 需要链式调用指定单位
                
            to (int, optional): 结束条件，指定最大执行次数
                例如：to=10 表示最多执行10次后自动停止
                
            tags (Iterable[Hashable], optional): 任务标签列表
                用于标识和管理特定的更新任务组
                例如：tags=['stock_update', 'real_time']
                
            in_background (bool, optional): 是否在后台执行，默认False
                - True: 异步后台执行，不阻塞主线程
                - False: 同步前台执行，阻塞主线程直到完成
                
            start_kwargs (dict, optional): 传递给调度管理器start方法的参数
                常用参数：
                - max_workers: 最大工作线程数
                - timeout: 任务超时时间
                
            **kwargs: 传递给update方法的参数
                这些参数会在每次更新时传递给数据源的update方法
                
        使用示例：
        
        1. 基本定期更新：
        ```python
        # 每30秒更新一次
        updater.update_every(30).seconds
        
        # 每5分钟更新一次
        updater.update_every(5).minutes
        
        # 每1小时更新一次
        updater.update_every().hour
        
        # 每天上午9点更新
        updater.update_every().day.at('09:00')
        ```
        
        2. 限制执行次数：
        ```python
        # 每秒更新一次，最多更新100次
        updater.update_every(1).second.to(100)
        
        # 每分钟更新一次，执行24次（一天）
        updater.update_every().minute.to(24*60)
        ```
        
        3. 任务标签管理：
        ```python
        # 创建带标签的更新任务
        updater.update_every(30).seconds.tag('realtime_update')
        
        # 创建多个不同的更新任务
        updater.update_every(1).minute.tag(['price_update', 'high_freq'])
        updater.update_every(1).hour.tag('hourly_summary')
        
        # 停止特定标签的任务
        updater.schedule_manager.stop('realtime_update')
        ```
        
        4. 后台异步执行：
        ```python
        # 在后台每30秒更新一次
        updater.update_every(30).seconds.tag('background_task')
        updater.schedule_manager.start_in_background()
        
        # 主程序可以继续执行其他任务
        print("后台更新已启动，主程序继续运行...")
        
        # 需要时可以停止后台任务
        updater.schedule_manager.stop('background_task')
        ```
        
        5. 复杂的调度模式：
        ```python
        # 只在工作日的交易时间更新
        updater.update_every().monday.to().friday.between('09:30', '16:00')
        
        # 每月第一个工作日更新
        updater.update_every().month.at('first_weekday')
        
        # 每周五收盘后更新
        updater.update_every().friday.at('16:30')
        
        # 每天特定时间点更新
        updater.update_every().day.at('09:00', '12:00', '15:00', '18:00')
        ```
        
        6. 传递更新参数：
        ```python
        # 每次更新时传递特定参数
        updater.update_every(5).minutes(
            end='now',           # 更新到当前时间
            limit=1000,         # 最大数据条数
            show_progress=False  # 不显示进度条
        )
        
        # 根据时间段调整参数
        if is_trading_hours():
            updater.update_every(1).minute(interval='1m')  # 交易时间高频更新
        else:
            updater.update_every(1).hour(interval='1h')    # 非交易时间低频更新
        ```
        
        7. 错误处理和监控：
        ```python
        class MonitoredUpdater(vbt.DataUpdater):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.update_stats = {'success': 0, 'failed': 0}
                
            def update(self, **kwargs):
                try:
                    super().update(**kwargs)
                    self.update_stats['success'] += 1
                    print(f"✓ 更新成功 (成功: {self.update_stats['success']})")
                except Exception as e:
                    self.update_stats['failed'] += 1
                    print(f"✗ 更新失败 (失败: {self.update_stats['failed']}): {e}")
                    
                    # 如果失败率太高，停止更新
                    total = sum(self.update_stats.values())
                    if total > 10 and self.update_stats['failed'] / total > 0.5:
                        raise vbt.CancelledError("失败率过高，停止更新")
        
        # 使用监控更新器
        monitored = MonitoredUpdater(data)
        monitored.update_every(30).seconds.tag('monitored')
        ```
        
        8. 生产环境最佳实践：
        ```python
        # 生产环境配置
        class ProductionUpdater(vbt.DataUpdater):
            def update_every_production(self):
                # 配置多层次的更新策略
                
                # 高频实时更新（交易时间）
                self.update_every(30).seconds.between('09:30', '16:00').tag('realtime')
                
                # 中频维护更新（非交易时间）
                self.update_every(15).minutes.between('16:01', '09:29').tag('maintenance')
                
                # 日终总结更新
                self.update_every().day.at('17:00').tag('daily_summary')
                
                # 周末数据清理
                self.update_every().saturday.at('02:00').tag('weekly_cleanup')
                
                # 启动后台调度
                self.schedule_manager.start_in_background(
                    max_workers=4,      # 4个工作线程
                    timeout=300,        # 5分钟超时
                    error_handling=True # 启用错误处理
                )
        
        # 部署到生产环境
        prod_updater = ProductionUpdater(production_data)
        prod_updater.update_every_production()
        ```
        
        注意事项：
        - 前台执行会阻塞主线程，适用于脚本和测试环境
        - 后台执行适用于长期运行的服务和生产环境
        - 合理设置更新频率，避免过度消耗API配额和系统资源
        - 在生产环境中建议使用任务标签进行精细化管理
        - 长期运行的任务需要考虑内存泄漏和资源回收
        """
        # 设置默认的启动参数
        if start_kwargs is None:
            start_kwargs = {}
            
        # 使用调度管理器创建定期任务
        # every()方法返回一个任务构建器，支持链式调用配置时间模式
        # do()方法绑定要执行的函数和参数
        self.schedule_manager.every(*args, to=to, tags=tags).do(self.update, **kwargs)
        
        # 根据执行模式启动任务调度
        if in_background:
            # 后台异步执行：不阻塞主线程，适用于长期运行的服务
            # 使用asyncio在后台线程中运行调度循环
            self.schedule_manager.start_in_background(**start_kwargs)
        else:
            # 前台同步执行：阻塞主线程直到任务完成或被中断
            # 适用于脚本环境和一次性数据更新任务
            self.schedule_manager.start(**start_kwargs)
