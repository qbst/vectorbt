# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
异步任务调度管理器模块 - vectorbt框架的定时任务调度核心组件

本模块为vectorbt量化交易框架提供了强大的异步任务调度能力，主要用于：
1. 定期数据获取和更新（如股价数据、新闻数据等）
2. 策略定时执行（如每日交易信号生成）
3. 报告和监控任务（如风险监控、绩效报告）
4. 系统维护任务（如数据清理、缓存更新）
"""

# 导入异步编程核心模块
import asyncio  # Python异步编程标准库，用于实现协程和异步任务管理
import inspect  # 代码内省模块，用于检查函数和对象的属性
import logging  # 日志记录模块，用于记录调度器的运行状态和错误信息
import time     # 时间处理模块，用于线程阻塞等待
from datetime import datetime, timedelta, time as dt_time  # 日期时间处理模块

# 导入第三方调度库的核心组件
from schedule import Scheduler, Job, CancelJob  # schedule包：Python任务调度库的核心类

# 导入vectorbt框架的类型定义和工具函数
from vectorbt import _typing as tp  # vectorbt框架的类型注解定义
from vectorbt.utils import checks   # vectorbt框架的参数检查工具
from vectorbt.utils.datetime_ import tzaware_to_naive_time  # 时区转换工具函数

# 创建模块级别的日志记录器，用于记录调度相关的日志信息
logger = logging.getLogger(__name__)


class CancelledError(asyncio.CancelledError):
    """
    任务取消异常类 - 用于标识调度任务被主动取消的异常情况
    
    继承自asyncio.CancelledError，专门用于标识调度管理器中的任务取消事件。
    这个自定义异常类提供了更明确的语义，区分了调度器层面的取消和asyncio层面的取消。
    
    使用场景：
    1. 手动取消长时间运行的任务
    2. 系统关闭时批量取消所有任务
    3. 任务执行异常时的清理操作
    4. 用户界面中的任务停止操作
    
    异常传播路径：
    任务函数 → AsyncJob.async_run() → AsyncScheduler._async_run_job() → ScheduleManager
    
    示例用法：
    ```python
    import vectorbt as vbt
    
    async def risky_task():
        try:
            # 执行可能需要取消的任务
            await some_long_running_operation()
        except vbt.CancelledError:
            print("任务被用户取消")
            # 清理资源
            await cleanup_resources()
            raise  # 重新抛出异常
    
    # 在调度管理器中使用
    manager = vbt.ScheduleManager()
    job = manager.every(5, 'minutes').do(risky_task)
    
    # 稍后取消任务
    manager.scheduler.cancel_job(job)
    ```
    
    注意事项：
    - 继承自asyncio.CancelledError，保持与asyncio生态的兼容性
    - 应该在任务函数中妥善处理此异常
    - 取消后的清理工作应该在异常处理中完成
    """
    pass


class AsyncJob(Job):
    """
    异步任务类 - 扩展了schedule.Job以支持异步任务执行
    
    AsyncJob是对标准schedule.Job的异步增强版本，专门用于处理需要异步执行的任务。
    它继承了schedule.Job的所有调度功能，同时添加了对Python asyncio协程的原生支持。
    
    核心特性：
    1. 异步执行：支持async/await语法的协程任务
    2. 非阻塞：任务执行不会阻塞主线程或其他任务
    3. 错误处理：完善的异步异常处理机制
    4. 状态管理：维护任务的执行状态和时间戳
    5. 自动调度：任务完成后自动计算下次执行时间
    
    与标准Job的区别：
    - 标准Job：同步执行，会阻塞调度器线程
    - AsyncJob：异步执行，支持并发任务处理
    
    适用场景：
    - 数据获取：异步从API获取金融数据
    - 网络请求：异步发送交易指令或通知
    - 数据处理：异步处理大量数据文件
    - 数据库操作：异步数据库查询和更新
    
    使用示例：
    ```python
    import asyncio
    import vectorbt as vbt
    
    # 定义异步任务函数
    async def fetch_stock_data(symbol):
        print(f"开始获取{symbol}的数据...")
        # 模拟异步API调用
        await asyncio.sleep(2)  # 模拟网络延迟
        print(f"完成获取{symbol}的数据")
        return f"{symbol}数据"
    
    # 创建调度管理器
    manager = vbt.ScheduleManager()
    
    # 添加异步任务 - 每分钟获取苹果股票数据
    job = manager.every(1, 'minute').do(fetch_stock_data, 'AAPL')
    print(f"任务类型: {type(job)}")  # <class 'AsyncJob'>
    
    # 启动异步调度
    await manager.async_start()
    ```
    
    生命周期：
    1. 创建：通过AsyncScheduler.every()创建
    2. 配置：设置执行间隔、时间等参数
    3. 绑定：通过.do()方法绑定具体的任务函数
    4. 调度：AsyncScheduler检查是否到达执行时间
    5. 执行：调用async_run()方法异步执行任务
    6. 完成：更新执行状态，计算下次执行时间
    
    注意事项：
    - 任务函数可以是普通函数或async函数
    - 如果任务函数返回CancelJob，任务将被取消
    - 异常会被记录但不会停止调度器
    """
    
    async def async_run(self) -> tp.Any:
        """
        异步执行任务的核心方法 - 处理异步任务的执行逻辑
        
        这是AsyncJob的核心方法，负责异步执行绑定的任务函数。它能够智能地处理
        同步和异步两种类型的任务函数，提供统一的异步执行接口。
        
        执行流程：
        1. 记录任务开始执行的日志
        2. 调用绑定的任务函数(self.job_func)
        3. 检查返回值是否为协程对象
        4. 如果是协程，使用await等待执行完成
        5. 更新任务的最后执行时间
        6. 计算并设置下次执行时间
        7. 返回任务函数的执行结果
        
        Returns:
            tp.Any: 任务函数的返回值，可能是任何类型
                   - 如果返回CancelJob，任务将被取消
                   - 如果返回其他值，任务继续按计划执行
        
        异常处理：
        - 任务函数抛出的异常会向上传播
        - 调度器会捕获异常并记录到日志
        - 单个任务的异常不会影响其他任务的执行
        
        支持的任务函数类型：
        
        1. 普通同步函数：
        ```python
        def sync_task():
            print("执行同步任务")
            return "完成"
        ```
        
        2. 异步协程函数：
        ```python
        async def async_task():
            await asyncio.sleep(1)
            print("执行异步任务")
            return "完成"
        ```
        
        3. 带参数的任务函数：
        ```python
        def task_with_args(symbol, amount):
            print(f"买入{amount}股{symbol}")
            return f"订单已提交"
        
        # 使用.do()绑定参数
        job.do(task_with_args, 'AAPL', 100)
        ```
        
        4. 返回取消指令的任务：
        ```python
        def conditional_task():
            if should_cancel():
                return CancelJob  # 任务将被取消
            return "继续执行"
        ```
        
        时间管理：
        - self.last_run：记录任务最后执行的时间
        - self._schedule_next_run()：计算下次执行时间
        - 支持时区感知的时间处理
        
        使用示例：
        ```python
        import asyncio
        import vectorbt as vbt
        
        # 创建混合任务示例
        async def portfolio_update():
            # 异步获取最新价格
            prices = await fetch_latest_prices()
            
            # 同步计算信号
            signals = calculate_signals(prices)
            
            # 异步提交订单
            if signals:
                await submit_orders(signals)
                
            return f"更新完成，处理了{len(signals)}个信号"
        
        # 创建和执行任务
        manager = vbt.ScheduleManager()
        job = manager.every(15, 'minutes').do(portfolio_update)
        
        # 手动测试任务执行
        result = await job.async_run()
        print(f"任务结果: {result}")
        ```
        
        注意事项：
        - 此方法不应该直接调用，而是由AsyncScheduler调用
        - 任务函数的执行时间会影响调度的精确性
        - 长时间运行的任务应该考虑使用后台任务队列
        """
        # 记录任务开始执行的日志，包含任务的字符串表示
        logger.info('Running job %s', self)
        
        # 调用绑定的任务函数，获取返回值
        # self.job_func是在.do()方法中绑定的实际任务函数
        ret = self.job_func()
        
        # 检查返回值是否为协程对象（awaitable）
        # 如果是异步函数，返回值会是协程对象，需要用await等待
        if inspect.isawaitable(ret):
            ret = await ret  # 等待协程执行完成
        
        # 更新任务的最后执行时间为当前时间
        # 这个时间戳用于计算下次执行时间和任务统计
        self.last_run = datetime.now()
        
        # 根据任务配置计算下次执行时间
        # 这会更新self.next_run属性
        self._schedule_next_run()
        
        # 返回任务函数的执行结果
        # 如果返回CancelJob，调度器会取消这个任务
        return ret


class AsyncScheduler(Scheduler):
    """
    异步调度器类 - 扩展了schedule.Scheduler以支持异步任务调度
    
    AsyncScheduler是对标准schedule.Scheduler的异步增强版本，专门用于管理和执行
    异步任务。它继承了schedule.Scheduler的所有调度功能，同时添加了对asyncio协程的原生支持。
    
    核心特性：
    1. 并发执行：多个任务可以同时异步执行，不会相互阻塞
    2. 异步调度：使用asyncio.gather()实现并发任务执行
    3. 错误隔离：单个任务的错误不会影响其他任务的执行
    4. 资源效率：避免了线程切换的开销，提高了系统性能
    5. 兼容性：完全兼容标准Scheduler的API
    
    与标准Scheduler的区别：
    - 标准Scheduler：同步执行，一次只能执行一个任务
    - AsyncScheduler：异步执行，支持多个任务并发执行
    
    适用场景：
    - 高频数据获取：同时从多个数据源获取数据
    - 并发交易：同时处理多个交易信号
    - 系统监控：并发监控多个系统指标
    - 报告生成：同时生成多种类型的报告
    
    使用示例：
    ```python
    import asyncio
    import vectorbt as vbt
    
    # 创建异步调度器
    scheduler = vbt.AsyncScheduler()
    
    # 添加多个异步任务
    scheduler.every(1, 'minute').do(fetch_stock_data, 'AAPL')
    scheduler.every(2, 'minutes').do(fetch_stock_data, 'GOOGL')
    scheduler.every(5, 'minutes').do(generate_report)
    
    # 异步执行所有到期任务
    await scheduler.async_run_pending()
    
    # 或者执行所有任务（不管是否到期）
    await scheduler.async_run_all(delay_seconds=1)
    ```
    
    调度模式：
    1. 按需调度：只执行到期的任务（async_run_pending）
    2. 全量调度：执行所有任务（async_run_all）
    3. 混合调度：结合两种模式的复杂调度策略
    
    性能优势：
    - 并发性：多个任务可以同时执行
    - 非阻塞：不会因为一个任务而阻塞整个调度器
    - 资源利用：更好地利用CPU和I/O资源
    - 可扩展性：支持大量并发任务
    
    注意事项：
    - 所有任务必须是AsyncJob类型
    - 任务函数可以是同步或异步函数
    - 错误处理需要在任务函数中实现
    - 资源竞争需要适当的同步机制
    """
    
    async def async_run_pending(self) -> None:
        """
        异步执行所有到期任务 - 并发执行所有应该运行的任务
        
        这是AsyncScheduler的核心方法，负责检查所有已注册的任务，找出应该执行的任务，
        并使用asyncio.gather()并发执行它们。这种方式比串行执行效率更高。
        
        执行流程：
        1. 遍历所有注册的任务(self.jobs)
        2. 筛选出应该运行的任务(job.should_run为True)
        3. 为每个任务创建异步执行协程
        4. 使用asyncio.gather()并发执行所有任务
        5. 等待所有任务完成
        
        并发执行的优势：
        - 时间效率：多个任务同时执行，总时间更短
        - 资源利用：更好地利用CPU和I/O资源
        - 响应性：不会因为一个慢任务而阻塞其他任务
        - 可扩展性：支持大量任务的并发执行
        
        任务调度条件：
        - 任务的next_run时间已到达或已过期
        - 任务没有被暂停或取消
        - 任务的job_func已正确绑定
        
        错误处理：
        - 每个任务的错误都会被独立处理
        - 一个任务的异常不会影响其他任务
        - 错误信息会记录到日志中
        
        使用示例：
        ```python
        import asyncio
        import vectorbt as vbt
        
        # 创建调度器并添加任务
        scheduler = vbt.AsyncScheduler()
        
        # 添加每分钟执行的数据获取任务
        scheduler.every(1, 'minute').do(fetch_market_data)
        scheduler.every(1, 'minute').do(update_portfolio)
        scheduler.every(5, 'minutes').do(check_risk_metrics)
        
        # 主调度循环
        async def main_loop():
            while True:
                # 执行所有到期任务
                await scheduler.async_run_pending()
                
                # 等待1秒后再次检查
                await asyncio.sleep(1)
        
        # 启动调度循环
        await main_loop()
        ```
        
        性能考虑：
        - 任务数量：过多的并发任务可能导致资源竞争
        - 任务类型：I/O密集型任务更适合并发执行
        - 依赖关系：有依赖的任务需要额外的同步机制
        - 错误传播：需要适当的错误处理和恢复机制
        
        注意事项：
        - 此方法应该在asyncio事件循环中调用
        - 任务函数应该是异步友好的
        - 资源竞争需要适当的锁机制
        - 内存使用需要监控和控制
        """
        # 使用生成器表达式筛选出应该运行的任务
        # job.should_run会检查任务的next_run时间是否已到达
        runnable_jobs = (job for job in self.jobs if job.should_run)
        
        # 使用asyncio.gather()并发执行所有应该运行的任务
        # 为每个任务创建异步执行协程，并等待所有任务完成
        await asyncio.gather(*[self._async_run_job(job) for job in runnable_jobs])

    async def async_run_all(self, delay_seconds: int = 0) -> None:
        """
        异步执行所有任务 - 强制执行所有已注册任务，不管是否到期
        
        这个方法会强制执行所有已注册的任务，不考虑它们的调度时间。主要用于：
        1. 系统初始化时的任务预热
        2. 测试环境中的批量任务执行
        3. 维护模式下的全量任务执行
        4. 紧急情况下的强制任务执行
        
        Args:
            delay_seconds (int, optional): 任务之间的延迟时间，默认为0
                - 0：任务立即连续执行
                - >0：任务之间等待指定秒数
                - 用于控制系统负载和资源使用
        
        执行流程：
        1. 记录执行日志，包含任务总数和延迟时间
        2. 遍历所有任务(self.jobs[:])
        3. 逐个异步执行任务
        4. 在任务之间等待指定的延迟时间
        5. 继续执行下一个任务
        
        延迟机制的作用：
        - 资源控制：避免同时执行过多任务导致系统过载
        - 错误恢复：给系统时间处理前一个任务的错误
        - 调试友好：便于观察和调试任务执行过程
        - 外部依赖：为外部系统提供响应时间
        
        使用场景：
        
        1. 系统初始化：
        ```python
        # 系统启动时预热所有任务
        await scheduler.async_run_all(delay_seconds=2)
        ```
        
        2. 测试环境：
        ```python
        # 测试所有任务的执行情况
        await scheduler.async_run_all(delay_seconds=0)
        ```
        
        3. 维护模式：
        ```python
        # 维护时间窗口内执行所有任务
        await scheduler.async_run_all(delay_seconds=5)
        ```
        
        4. 紧急处理：
        ```python
        # 市场异常时强制执行所有风险检查
        risk_scheduler = get_risk_scheduler()
        await risk_scheduler.async_run_all(delay_seconds=1)
        ```
        
        与async_run_pending的区别：
        - async_run_pending：只执行到期的任务，并发执行
        - async_run_all：执行所有任务，串行执行，可设置延迟
        
        注意事项：
        - 使用self.jobs[:]创建副本，避免执行过程中jobs列表被修改
        - 延迟时间会显著影响总执行时间
        - 所有任务都会被执行，无论调度时间如何
        - 适合用于批量处理和系统维护
        """
        # 记录执行日志，显示任务总数和延迟设置
        logger.info('Running *all* %i jobs with %is delay in-between',
                    len(self.jobs), delay_seconds)
        
        # 遍历所有任务（使用副本避免执行过程中列表被修改）
        for job in self.jobs[:]:
            # 异步执行当前任务
            await self._async_run_job(job)
            
            # 在任务之间等待指定的延迟时间
            # 这有助于控制系统负载和资源使用
            await asyncio.sleep(delay_seconds)

    async def _async_run_job(self, job: AsyncJob) -> None:
        """
        异步执行单个任务 - 执行指定任务并处理结果
        
        这是AsyncScheduler的内部方法，负责执行单个AsyncJob任务并处理其返回值。
        它是async_run_pending和async_run_all的核心执行单元。
        
        Args:
            job (AsyncJob): 要执行的异步任务对象
        
        执行流程：
        1. 调用job.async_run()异步执行任务
        2. 获取任务的返回值
        3. 检查返回值是否为取消指令
        4. 如果是取消指令，从调度器中移除该任务
        
        返回值处理：
        - 普通返回值：任务继续按计划执行
        - CancelJob实例：任务被取消并从调度器中移除
        - CancelJob类：任务被取消并从调度器中移除
        - 异常：错误会向上传播到调度器层面
        
        取消机制：
        任务函数可以通过返回CancelJob来请求取消自己：
        ```python
        from schedule import CancelJob
        
        def conditional_task():
            if some_condition():
                return CancelJob  # 任务将被取消
            return "继续执行"
        ```
        
        错误处理：
        - 任务执行中的异常会向上传播
        - 调度器级别会捕获并记录异常
        - 单个任务的错误不会影响其他任务
        
        使用示例：
        ```python
        # 这个方法通常不直接调用，而是由调度器内部使用
        # 但可以用于测试单个任务的执行
        
        scheduler = vbt.AsyncScheduler()
        job = scheduler.every(1, 'minute').do(some_task)
        
        # 手动执行单个任务（用于测试）
        await scheduler._async_run_job(job)
        ```
        
        生命周期管理：
        - 任务执行成功：更新next_run时间，继续调度
        - 任务返回取消：从jobs列表中移除任务
        - 任务执行失败：记录错误，根据策略决定是否继续
        
        注意事项：
        - 这是私有方法，不应该直接调用
        - 任务的生命周期由返回值决定
        - 错误处理需要在上层实现
        - 任务取消是不可逆的操作
        """
        # 异步执行任务，获取返回值
        ret = await job.async_run()
        
        # 检查返回值是否为取消指令
        # 支持两种取消方式：CancelJob实例或CancelJob类本身
        if isinstance(ret, CancelJob) or ret is CancelJob:
            # 如果任务请求取消，从调度器中移除该任务
            self.cancel_job(job)

    def every(self, *args, to: tp.Optional[int] = None,
              tags: tp.Optional[tp.Iterable[tp.Hashable]] = None) -> AsyncJob:
        """
        创建新的定期任务 - 智能解析参数并创建AsyncJob实例
        
        这是ScheduleManager的核心方法，提供了灵活而强大的任务调度接口。它能够智能解析
        多种参数格式，自动构建合适的调度规则，支持从简单的时间间隔到复杂的时区调度。
        
        参数解析规则：
        *args最多可以包含四个不同的参数，按严格顺序：interval, unit, start_day, at
        
        参数类型识别：
        - interval: 整数或datetime.timedelta对象
        - unit: 字符串，必须在ScheduleManager.units中
        - start_day: 字符串，必须在ScheduleManager.weekdays中
        - at: 字符串（包含:）或datetime.time对象
        
        Args:
            *args: 变长参数，按顺序可包含：
                - interval (int/timedelta): 执行间隔
                - unit (str): 时间单位
                - start_day (str): 起始日期
                - at (str/time): 执行时间
            to (int, optional): 随机间隔的上限，创建随机间隔任务
            tags (Iterable[Hashable], optional): 任务标签，用于分类和批量管理
            
        Returns:
            AsyncJob: 配置完成的异步任务对象
            
        参数解析示例：
        
        1. 基本间隔调度：
        ```python
        manager.every().do(task)                    # 每秒执行
        manager.every(5).do(task)                   # 每5秒执行
        manager.every(timedelta(minutes=30)).do(task) # 每30分钟执行
        ```
        
        2. 时间单位调度：
        ```python
        manager.every(10, 'minutes').do(task)       # 每10分钟执行
        manager.every('hour').do(task)              # 每小时执行
        manager.every(2, 'days').do(task)           # 每2天执行
        ```
        
        3. 特定时间调度：
        ```python
        manager.every('09:30').do(task)             # 每天9:30执行
        manager.every('day', '14:15').do(task)      # 每天14:15执行
        manager.every('hour', ':30').do(task)       # 每小时30分执行
        ```
        
        4. 工作日调度：
        ```python
        manager.every('monday').do(task)            # 每周一执行
        manager.every('friday', '17:00').do(task)   # 每周五17:00执行
        ```
        
        5. 时区支持：
        ```python
        import pytz
        eastern = pytz.timezone('US/Eastern')
        utc_time = datetime.time(9, 30, tzinfo=pytz.utc)
        manager.every('day', utc_time).do(task)     # 每天UTC时间9:30执行
        ```
        
        6. 随机间隔：
        ```python
        manager.every(10, to=30).do(task)           # 每10-30秒随机执行
        manager.every(1, 'hour', to=2).do(task)     # 每1-2小时随机执行
        ```
        
        7. 任务标签：
        ```python
        manager.every(5, 'minutes', tags=['monitoring']).do(health_check)
        manager.every(1, 'day', tags=['reports', 'daily']).do(daily_report)
        ```
        
        复杂示例：
        ```python
        import datetime
        import pytz
        import vectorbt as vbt
        
        manager = vbt.ScheduleManager()
        
        # 股票数据获取 - 工作日每分钟
        manager.every('minute', tags=['data']).do(fetch_stock_data)
        
        # 交易信号生成 - 每5分钟
        manager.every(5, 'minutes', tags=['trading']).do(generate_signals)
        
        # 风险检查 - 每15分钟，带随机延迟
        manager.every(15, 'minutes', to=20, tags=['risk']).do(risk_check)
        
        # 日终处理 - 每天16:00，东部时间
        eastern = pytz.timezone('US/Eastern')
        eod_time = datetime.time(16, 0, tzinfo=eastern)
        manager.every('day', eod_time, tags=['eod']).do(end_of_day_process)
        
        # 周报生成 - 每周五17:30
        manager.every('friday', '17:30', tags=['reports']).do(weekly_report)
        
        # 系统健康检查 - 每30秒到2分钟随机
        manager.every(30, 'seconds', to=120, tags=['health']).do(health_check)
        ```
        
        时间格式智能处理：
        1. 字符串时间：'HH:MM', 'HH:MM:SS', ':MM', ':MM:SS'
        2. 时间对象：datetime.time(hour, minute, second, tzinfo)
        3. 时区转换：自动处理时区感知的时间对象
        4. 相对时间：基于不同时间单位的相对时间设置
        
        参数验证和错误处理：
        - 参数类型检查：确保参数符合预期类型
        - 参数顺序验证：严格按照interval, unit, start_day, at的顺序
        - 时间格式验证：检查时间字符串的格式正确性
        - 时区处理：自动转换时区感知的时间对象
        
        高级特性：
        - 链式调用：支持schedule包的链式语法
        - 参数默认值：智能设置默认参数
        - 时区转换：自动处理时区转换
        - 标签管理：支持任务分类和批量操作
        
        注意事项：
        - 参数顺序必须严格按照interval, unit, start_day, at
        - 时间字符串必须包含冒号(:)
        - 时区时间对象会自动转换为本地时间
        - 标签可以是任何可哈希的对象
        - 创建任务后需要调用.do()方法绑定任务函数
        """
        # 初始化参数解析变量
        interval = 1      # 默认间隔为1
        unit = None       # 时间单位
        start_day = None  # 起始日期
        at = None        # 执行时间

        # 参数类型识别函数
        def _is_arg_interval(arg):
            """检查参数是否为间隔类型（整数或timedelta）"""
            return isinstance(arg, (int, timedelta))

        def _is_arg_unit(arg):
            """检查参数是否为时间单位类型"""
            return isinstance(arg, str) and arg in self.units

        def _is_arg_start_day(arg):
            """检查参数是否为工作日类型"""
            return isinstance(arg, str) and arg in self.weekdays

        def _is_arg_at(arg):
            """检查参数是否为执行时间类型"""
            return (isinstance(arg, str) and ':' in arg) or isinstance(arg, dt_time)

        # 期望参数列表，用于控制解析顺序
        expected_args = ['interval', 'unit', 'start_day', 'at']
        
        # 遍历所有传入的参数进行解析
        for i, arg in enumerate(args):
            # 检查是否为间隔参数
            if 'interval' in expected_args and _is_arg_interval(arg):
                interval = arg
                # 移除已解析的参数类型，确保顺序正确
                expected_args = expected_args[expected_args.index('interval') + 1:]
                continue
            
            # 检查是否为时间单位参数
            if 'unit' in expected_args and _is_arg_unit(arg):
                unit = arg
                expected_args = expected_args[expected_args.index('unit') + 1:]
                continue
            
            # 检查是否为起始日期参数
            if 'start_day' in expected_args and _is_arg_start_day(arg):
                start_day = arg
                expected_args = expected_args[expected_args.index('start_day') + 1:]
                continue
            
            # 检查是否为执行时间参数
            if 'at' in expected_args and _is_arg_at(arg):
                at = arg
                expected_args = expected_args[expected_args.index('at') + 1:]
                continue
            
            # 如果参数不符合任何预期类型，抛出错误
            raise ValueError(f"Arg at index {i} is unexpected")

        # 智能默认值设置
        # 如果指定了执行时间但没有指定单位和起始日期，默认为每天
        if at is not None:
            if unit is None and start_day is None:
                unit = 'days'
        
        # 如果没有指定单位和起始日期，默认为秒
        if unit is None and start_day is None:
            unit = 'seconds'

        # 创建基础任务对象
        job = self.scheduler.every(interval)
        
        # 应用时间单位
        if unit is not None:
            job = getattr(job, unit)
        
        # 应用起始日期
        if start_day is not None:
            job = getattr(job, start_day)
        
        # 处理执行时间
        if at is not None:
            # 如果是时间对象，需要进行格式转换
            if isinstance(at, dt_time):
                # 对于按天或按工作日的任务，需要处理时区
                if job.unit == "days" or job.start_day:
                    # 如果时间对象包含时区信息，转换为本地时间
                    if at.tzinfo is not None:
                        at = tzaware_to_naive_time(at, None)
                
                # 转换为ISO格式字符串
                at = at.isoformat()
                
                # 根据时间单位调整时间格式
                if job.unit == "hours":
                    # 小时任务只需要分钟和秒
                    at = ':'.join(at.split(':')[1:])
                if job.unit == "minutes":
                    # 分钟任务只需要秒
                    at = ':' + at.split(':')[2]
            
            # 应用执行时间
            job = job.at(at)
        
        # 应用随机间隔上限
        if to is not None:
            job = job.to(to)
        
        # 应用任务标签
        if tags is not None:
            # 确保标签是元组格式
            if not isinstance(tags, tuple):
                tags = (tags,)
            job = job.tag(*tags)

        # 返回配置完成的任务对象
        return job


class ScheduleManager:
    """
    调度管理器类 - vectorbt框架的高级任务调度管理器
    
    ScheduleManager是vectorbt框架中的核心调度管理组件，它为AsyncScheduler提供了更高级的
    管理接口和功能。这个类专门设计用于量化交易系统中的任务调度管理，提供了丰富的API
    和强大的任务管理能力。
    
    core features:
    1. Scheduler management: Encapsulates and manages the AsyncScheduler instance
    2. Task lifecycle: Complete task creation, execution, monitoring, and stop process
    3. Time parsing: Smartly parses various time formats and scheduling rules
    4. Asynchronous support: Complete asynchronous task execution and management
    5. Background running: Supports both background tasks and foreground tasks
    
    Design philosophy:
    - Ease of use: Provides a concise and intuitive API interface
    - Flexibility: Supports multiple time formats and scheduling modes
    - Reliability: Comprehensive error handling and exception recovery mechanism
    - Extensibility: Supports custom schedulers and task types
    - Performance: Efficient asynchronous task execution and resource management
    
    Use cases:
    1. Quantitative trading system: Manages trading strategies, data acquisition, risk control, etc.
    2. Data processing pipeline: Manages batch tasks such as data collection, cleaning, analysis
    3. Monitoring system: Manages system health checks, performance monitoring, etc.
    4. Report system: Manages report generation, sending, etc.
    
    Complete example:
    ```python
    import asyncio
    import datetime
    import pytz
    import vectorbt as vbt
    
    # Create a scheduler manager
    manager = vbt.ScheduleManager()
    
    # Define task functions
    async def fetch_stock_data(symbol):
        print(f"Fetching data for {symbol}...")
        # Simulate API call
        await asyncio.sleep(1)
        return f"{symbol} data fetched"
    
    def generate_report():
        print("Generating daily report...")
        return "Report generated"
    
    # Add various types of tasks
    # 1. Basic interval tasks
    manager.every(30).do(fetch_stock_data, 'AAPL')
    
    # 2. Specify time units
    manager.every(5, 'minutes').do(fetch_stock_data, 'GOOGL')
    
    # 3. Execute at specific times
    manager.every('09:30').do(fetch_stock_data, 'MSFT')
    
    # 4. Weekday tasks
    manager.every('monday', '17:00').do(generate_report)
    
    # 5. Use timezones
    eastern = pytz.timezone('US/Eastern')
    manager.every('day', datetime.time(9, 30, tzinfo=eastern)).do(fetch_stock_data, 'TSLA')
    
    # 6. Tasks with tags
    manager.every(1, 'hour', tags=['monitoring']).do(health_check)
    
    # Start the scheduler
    # Method 1: Synchronous run (blocks the main thread)
    manager.start()
    
    # Method 2: Asynchronous run
    await manager.async_start()
    
    # Method 3: Background run (non-blocking)
    manager.start_in_background()
    
    # Check background task status
    if manager.async_task_running:
        print("Scheduler is running in the background")
    
    # Stop background task
    manager.stop()
    ```
    
    Time format support:
    - Numbers: 1, 5, 30 (combined with units)
    - Time strings: '09:30', '14:15:30'
    - Time objects: datetime.time(9, 30)
    - Time delta: datetime.timedelta(minutes=30)
    - Weekdays: 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
    - Time units: 'second', 'minute', 'hour', 'day', 'week'
    
    Advanced features:
    - Task tags: Support task classification and batch management
    - Timezone support: Automatically handles task scheduling across different timezones
    - Error handling: Task-level exception handling and recovery
    - Lifecycle management: Task creation, execution, monitoring, and stop
    - Dynamic scheduling: Add and delete tasks at runtime
    """

    # Class variables: List of supported time units
    # These units can be used in the every() method, supporting singular and plural forms
    units: tp.ClassVar[tp.Tuple[str, ...]] = (
        "second",    # Second (singular)
        "seconds",   # Second (plural)
        "minute",    # Minute (singular)
        "minutes",   # Minute (plural)
        "hour",      # Hour (singular)
        "hours",     # Hour (plural)
        "day",       # Day (singular)
        "days",      # Day (plural)
        "week",      # Week (singular)
        "weeks"      # Week (plural)
    )

    # Class variables: List of supported weekdays
    # These weekdays can be used in the every() method, for tasks on specific days
    weekdays: tp.ClassVar[tp.Tuple[str, ...]] = (
        "monday",     # Monday
        "tuesday",    # Tuesday
        "wednesday",  # Wednesday
        "thursday",   # Thursday
        "friday",     # Friday
        "saturday",   # Saturday
        "sunday",     # Sunday
    )

    def __init__(self, scheduler: tp.Optional[AsyncScheduler] = None) -> None:
        """
        Initialize the scheduler manager
        
        Creates a new scheduler manager instance, which can use a default AsyncScheduler or provide a custom scheduler.
        
        Args:
            scheduler (AsyncScheduler, optional): An optional asynchronous scheduler instance
                - If None, a new AsyncScheduler instance will be automatically created
                - If provided, it must be an instance of AsyncScheduler
                - Used to reuse an existing scheduler or use a custom scheduler with specific configuration
        
        Initialization process:
        1. Parameter validation: Ensure the provided scheduler is of type AsyncScheduler
        2. Scheduler setting: Set or create an AsyncScheduler instance
        3. Task state initialization: Initialize the state variables related to the background task
        
        Usage example:
        ```python
        # Use default scheduler
        manager = vbt.ScheduleManager()
        
        # Use custom scheduler
        custom_scheduler = vbt.AsyncScheduler()
        manager = vbt.ScheduleManager(scheduler=custom_scheduler)
        
        # Reuse existing scheduler
        existing_scheduler = get_existing_scheduler()
        manager = vbt.ScheduleManager(scheduler=existing_scheduler)
        ```
        
        Notes:
        - Be careful when sharing the scheduler instance among multiple ScheduleManager instances to avoid task conflicts
        - Custom schedulers should inherit from AsyncScheduler
        - Once initialized, the scheduler cannot be changed, only create new manager instances
        """
        # If no scheduler is provided, create a new AsyncScheduler instance
        if scheduler is None:
            scheduler = AsyncScheduler()
        
        # Validate scheduler type: must be an instance of AsyncScheduler
        checks.assert_instance_of(scheduler, AsyncScheduler)

        # Set private attribute: store the scheduler instance
        self._scheduler = scheduler
        
        # Initialize asynchronous task state: for managing background running tasks
        self._async_task = None

    @property
    def scheduler(self) -> AsyncScheduler:
        """
        Get the underlying asynchronous scheduler instance
        
        This property provides read-only access to the underlying AsyncScheduler instance, allowing users to directly
        operate on the scheduler for more advanced task management and configuration.
        
        Returns:
            AsyncScheduler: The underlying asynchronous scheduler instance
        
        Usage scenarios:
        1. Direct scheduler operation: Access more low-level scheduler features
        2. Task querying: Get detailed information about all current tasks
        3. Advanced configuration: Perform scheduler-level configuration modifications
        4. Debugging and monitoring: Check the internal state of the scheduler
        
        Usage example:
        ```python
        manager = vbt.ScheduleManager()
        
        # Get scheduler instance
        scheduler = manager.scheduler
        
        # View all tasks
        print(f"Current number of tasks: {len(scheduler.jobs)}")
        for job in scheduler.jobs:
            print(f"Task: {job}, Next run: {job.next_run}")
        
        # Cancel specific tasks
        for job in scheduler.jobs:
            if 'AAPL' in str(job):
                scheduler.cancel_job(job)
        
        # Clear all tasks
        scheduler.clear()
        ```
        
        Notes:
        - This is a read-only property, do not assign directly
        - Be cautious when directly operating on the scheduler to avoid breaking the manager's state
        - Modifications to the scheduler will affect all manager instances that use it
        """
        return self._scheduler

    @property
    def async_task(self) -> tp.Optional[asyncio.Task]:
        """
        Get the current asynchronous task instance
        
        This property returns the current asynchronous task instance that is running in the background,
        or None if no background task is running. It is primarily used for monitoring and managing
        background running scheduling tasks.
        
        Returns:
            asyncio.Task or None: The current asynchronous task instance, or None if no background task is running
        
        Task status:
        - None: No background task is running
        - asyncio.Task: A background task is running
        - Completed Task: The background task has completed or been cancelled
        
        Usage scenarios:
        1. Status monitoring: Check if the background task is running
        2. Task management: Get the task instance for operations
        3. Error handling: Check if the task exited abnormally
        4. Performance monitoring: Get the execution status and statistics of the task
        
        Usage example:
        ```python
        manager = vbt.ScheduleManager()
        
        # Start background task
        manager.start_in_background()
        
        # Check task status
        task = manager.async_task
        if task is not None:
            print(f"Task status: {task.get_name()}")
            print(f"Task completed: {task.done()}")
            print(f"Task cancelled: {task.cancelled()}")
            
            # Wait for task to complete
            try:
                result = await task
                print(f"Task result: {result}")
            except asyncio.CancelledError:
                print("Task was cancelled")
            except Exception as e:
                print(f"Task error: {e}")
        else:
            print("No background task is running")
        
        # Manually cancel task
        if task and not task.done():
            task.cancel()
        ```
        
        Notes:
        - The returned Task object is read-only, do not modify directly
        - The task may have already completed or been cancelled when checked
        - After cancelling the task, it should wait for it to fully stop
        - Calling start_in_background() multiple times will override previous tasks
        """
        return self._async_task

    def start(self, sleep: int = 1) -> None:
        """
        同步启动调度管理器 - 在主线程中阻塞运行调度循环
        
        这个方法会启动一个无限循环来持续检查和执行到期的任务。它会阻塞当前线程，
        直到收到中断信号（如Ctrl+C）或程序异常退出。适用于调度器是程序主要功能的场景。
        
        Args:
            sleep (int, optional): 检查间隔时间（秒），默认为1秒
                - 较小的值：更精确的调度，但会消耗更多CPU资源
                - 较大的值：节省CPU资源，但调度精度降低
                - 建议范围：1-10秒，根据任务精度要求调整
        
        执行流程：
        1. 记录启动日志，显示当前所有任务
        2. 进入无限循环
        3. 检查并执行到期任务（调用scheduler.run_pending）
        4. 休眠指定时间
        5. 重复步骤3-4，直到收到中断信号
        
        中断处理：
        - KeyboardInterrupt：用户按Ctrl+C
        - asyncio.CancelledError：程序取消信号
        - 其他异常：会导致调度器停止
        
        使用场景：
        1. 独立的调度程序：调度器是程序的主要功能
        2. 简单的定时任务：不需要复杂的异步处理
        3. 脚本式应用：一次性执行的定时任务脚本
        4. 开发测试：测试调度器的基本功能
        
        使用示例：
        ```python
        import vectorbt as vbt
        
        # 创建调度管理器
        manager = vbt.ScheduleManager()
        
        # 添加任务
        manager.every(30, 'seconds').do(fetch_stock_data)
        manager.every(5, 'minutes').do(generate_signals)
        manager.every('09:30').do(market_open_task)
        
        # 启动调度器（阻塞运行）
        try:
            print("启动调度器...")
            manager.start(sleep=2)  # 每2秒检查一次
        except KeyboardInterrupt:
            print("用户中断，调度器停止")
        except Exception as e:
            print(f"调度器异常退出: {e}")
        ```
        
        性能考虑：
        - sleep参数影响调度精度和CPU使用率
        - 任务执行时间会影响整体调度精度
        - 长时间运行的任务会阻塞其他任务
        - 建议将耗时任务放在异步函数中执行
        
        与其他启动方式的比较：
        - start()：同步阻塞，简单直接
        - async_start()：异步非阻塞，需要在asyncio环境中
        - start_in_background()：后台运行，不阻塞主线程
        
        注意事项：
        - 此方法会阻塞当前线程
        - 适合用作程序的主循环
        - 任务函数应该避免长时间阻塞
        - 异常会导致整个调度器停止
        """
        # 记录启动日志，显示当前所有已注册的任务
        logger.info("Starting schedule manager with jobs %s", str(self.scheduler.jobs))
        
        try:
            # 进入无限循环，持续检查和执行任务
            while True:
                # 检查并执行所有到期的任务
                # 这里使用同步的run_pending()方法
                self.scheduler.run_pending()
                
                # 休眠指定时间，控制检查频率
                time.sleep(sleep)
        except (KeyboardInterrupt, asyncio.CancelledError):
            # 捕获中断信号，优雅地停止调度器
            logger.info("Stopping schedule manager")

    async def async_start(self, sleep: int = 1) -> None:
        """
        异步启动调度管理器 - 在asyncio事件循环中非阻塞运行
        
        这个方法在asyncio事件循环中启动调度器，支持异步任务的并发执行。
        它不会阻塞事件循环，允许其他协程并发运行。适用于需要异步处理的复杂应用。
        
        Args:
            sleep (int, optional): 检查间隔时间（秒），默认为1秒
                - 与start()方法的sleep参数作用相同
                - 但这里使用asyncio.sleep()实现非阻塞等待
        
        执行流程：
        1. 记录启动日志和任务信息
        2. 进入异步无限循环
        3. 异步检查并执行到期任务（调用async_run_pending）
        4. 异步休眠指定时间
        5. 重复步骤3-4，直到收到取消信号
        
        异步特性：
        - 非阻塞执行：不会阻塞事件循环
        - 并发任务：多个任务可以同时执行
        - 协程友好：与其他asyncio代码良好集成
        - 异常隔离：单个任务异常不影响整体调度
        
        使用场景：
        1. 异步应用：需要同时处理多种异步任务
        2. Web应用：在Web服务器中运行定时任务
        3. 实时系统：需要高并发和低延迟的系统
        4. 复杂应用：包含多个异步组件的大型应用
        
        使用示例：
        ```python
        import asyncio
        import vectorbt as vbt
        
        async def main():
            # 创建调度管理器
            manager = vbt.ScheduleManager()
            
            # 添加异步任务
            manager.every(1, 'minute').do(async_fetch_data)
            manager.every(5, 'minutes').do(async_process_data)
            
            # 创建多个并发任务
            tasks = [
                asyncio.create_task(manager.async_start(sleep=1)),
                asyncio.create_task(web_server_task()),
                asyncio.create_task(monitoring_task())
            ]
            
            # 等待所有任务完成
            await asyncio.gather(*tasks)
        
        # 运行主协程
        asyncio.run(main())
        ```
        
        错误处理：
        ```python
        async def safe_async_start():
            manager = vbt.ScheduleManager()
            
            try:
                await manager.async_start()
            except asyncio.CancelledError:
                print("调度器被取消")
            except Exception as e:
                print(f"调度器异常: {e}")
            finally:
                print("调度器清理完成")
        ```
        
        性能优势：
        - 并发执行：多个任务同时运行
        - 资源效率：更好的CPU和I/O利用率
        - 响应性：不会因为单个任务而阻塞整个系统
        - 可扩展性：支持大量并发任务
        
        注意事项：
        - 必须在asyncio事件循环中运行
        - 任务函数应该是异步友好的
        - 需要妥善处理异常和取消信号
        - 适合用于异步应用架构
        """
        # 记录启动日志，显示后台运行状态
        logger.info("Starting schedule manager in the background with jobs %s", str(self.scheduler.jobs))
        # 详细记录所有任务信息，用于调试
        logger.info("Jobs: %s", str(self.scheduler.jobs))
        
        try:
            # 进入异步无限循环
            while True:
                # 异步检查并执行所有到期的任务
                # 使用async_run_pending()支持并发任务执行
                await self.scheduler.async_run_pending()
                
                # 异步休眠，不阻塞事件循环
                await asyncio.sleep(sleep)
        except asyncio.CancelledError:
            # 捕获取消信号，优雅地停止调度器
            logger.info("Stopping schedule manager")

    def done_callback(self, async_task: asyncio.Task) -> None:
        """
        异步任务完成回调函数 - 处理后台任务的完成事件
        
        这个回调函数在后台异步任务完成、失败或被取消时自动调用。
        它用于记录任务状态、清理资源和处理异常情况。
        
        Args:
            async_task (asyncio.Task): 已完成的异步任务实例
                - 可能是正常完成、异常退出或被取消
                - 包含任务的结果或异常信息
        
        回调触发条件：
        1. 任务正常完成：调度器正常停止
        2. 任务异常退出：调度器因异常而停止
        3. 任务被取消：用户或系统取消任务
        4. 资源清理：系统关闭时的清理过程
        
        处理逻辑：
        - 记录任务状态到日志
        - 可以扩展为错误报告、重启逻辑等
        - 清理相关资源
        - 通知其他系统组件
        
        使用场景：
        1. 状态监控：记录调度器的运行状态
        2. 错误处理：处理调度器异常退出
        3. 资源清理：清理相关资源和连接
        4. 通知机制：通知其他组件调度器状态变化
        
        扩展示例：
        ```python
        def custom_done_callback(self, async_task: asyncio.Task) -> None:
            try:
                # 获取任务结果
                result = async_task.result()
                logger.info(f"调度器正常完成: {result}")
            except asyncio.CancelledError:
                logger.info("调度器被用户取消")
            except Exception as e:
                logger.error(f"调度器异常退出: {e}")
                # 可以在这里添加重启逻辑
                self.restart_scheduler()
            finally:
                # 清理资源
                self.cleanup_resources()
                # 通知其他组件
                self.notify_scheduler_stopped()
        
        # 自定义回调
        manager.done_callback = custom_done_callback
        ```
        
        任务状态检查：
        ```python
        def detailed_done_callback(self, async_task: asyncio.Task) -> None:
            logger.info(f"任务对象: {async_task}")
            logger.info(f"任务名称: {async_task.get_name()}")
            logger.info(f"任务完成: {async_task.done()}")
            logger.info(f"任务取消: {async_task.cancelled()}")
            
            if async_task.exception():
                logger.error(f"任务异常: {async_task.exception()}")
            elif not async_task.cancelled():
                logger.info(f"任务结果: {async_task.result()}")
        ```
        
        注意事项：
        - 回调函数应该快速执行，避免阻塞
        - 异常处理要充分，防止回调函数本身出错
        - 可以通过子类化或赋值来自定义回调逻辑
        - 回调函数在任务线程中执行
        """
        # 记录任务完成信息到日志
        # 包含任务对象的字符串表示，便于调试
        logger.info(async_task)

    def start_in_background(self, **kwargs) -> None:
        """
        在后台启动调度管理器 - 创建后台任务，不阻塞主线程
        
        这个方法会创建一个asyncio.Task来在后台运行调度器，允许主线程继续执行其他代码。
        适用于需要调度器和其他功能并行运行的应用场景。
        
        Args:
            **kwargs: 传递给async_start()方法的关键字参数
                - sleep: 检查间隔时间
                - 其他async_start()支持的参数
        
        执行流程：
        1. 创建async_start()的异步任务
        2. 设置任务完成回调函数
        3. 记录任务创建信息
        4. 保存任务引用以供管理
        5. 立即返回，不等待任务完成
        
        后台任务管理：
        - 任务创建：使用asyncio.create_task()
        - 回调设置：自动调用done_callback()
        - 状态跟踪：通过async_task属性访问
        - 生命周期：可通过stop()方法停止
        
        使用场景：
        1. GUI应用：在图形界面中运行定时任务
        2. Web应用：在Web框架中运行后台任务
        3. 混合应用：需要多种功能并行运行
        4. 服务程序：作为系统服务的一部分运行
        
        基本使用示例：
        ```python
        import time
        import vectorbt as vbt
        
        # 创建调度管理器
        manager = vbt.ScheduleManager()
        
        # 添加任务
        manager.every(30, 'seconds').do(background_task)
        manager.every(5, 'minutes').do(periodic_check)
        
        # 启动后台调度器
        manager.start_in_background(sleep=2)
        
        # 主线程继续执行其他工作
        print("调度器已启动，主程序继续运行...")
        for i in range(60):
            print(f"主程序工作中... {i}")
            time.sleep(1)
        
        # 停止后台调度器
        manager.stop()
        print("调度器已停止")
        ```
        
        Web应用示例：
        ```python
        from flask import Flask
        import vectorbt as vbt
        
        app = Flask(__name__)
        manager = vbt.ScheduleManager()
        
        # 添加后台数据更新任务
        manager.every(1, 'minute').do(update_cache)
        manager.every(10, 'minutes').do(cleanup_temp_files)
        
        @app.before_first_request
        def start_scheduler():
            # 在Web应用启动时启动调度器
            manager.start_in_background()
        
        @app.teardown_appcontext
        def stop_scheduler(exception):
            # 在应用关闭时停止调度器
            manager.stop()
        
        if __name__ == '__main__':
            app.run()
        ```
        
        状态监控示例：
        ```python
        manager = vbt.ScheduleManager()
        manager.every(1, 'minute').do(monitoring_task)
        
        # 启动后台调度器
        manager.start_in_background()
        
        # 监控调度器状态
        while True:
            if manager.async_task_running:
                print("调度器正在运行")
            else:
                print("调度器已停止")
                break
            time.sleep(5)
        ```
        
        错误处理：
        ```python
        def safe_background_start():
            try:
                manager.start_in_background()
                print("后台调度器启动成功")
            except Exception as e:
                print(f"启动失败: {e}")
        
        # 检查启动结果
        safe_background_start()
        time.sleep(1)  # 等待一点时间
        if not manager.async_task_running:
            print("调度器启动失败或立即退出")
        ```
        
        注意事项：
        - 必须有活跃的asyncio事件循环
        - 多次调用会覆盖之前的后台任务
        - 应用关闭前记得调用stop()方法
        - 后台任务的异常会记录到日志中
        """
        # 创建异步任务，运行async_start()方法
        # 传递所有关键字参数给async_start()
        async_task = asyncio.create_task(self.async_start(**kwargs))
        
        # 设置任务完成回调函数
        # 当任务完成、异常或被取消时会自动调用
        async_task.add_done_callback(self.done_callback)
        
        # 记录任务创建信息到日志
        logger.info(async_task)
        
        # 保存任务引用，用于后续管理和状态检查
        self._async_task = async_task

    @property
    def async_task_running(self) -> bool:
        """
        检查异步任务是否正在运行 - 返回后台任务的运行状态
        
        这个属性提供了一个便捷的方式来检查后台调度任务是否仍在运行。
        它结合了任务存在性和完成状态的检查。
        
        Returns:
            bool: 如果后台任务正在运行返回True，否则返回False
                - True: 有后台任务且正在运行
                - False: 没有后台任务或任务已完成/取消
        
        状态判断逻辑：
        1. 检查是否存在异步任务实例（self.async_task is not None）
        2. 检查任务是否未完成（not self.async_task.done()）
        3. 两个条件都满足时返回True
        
        可能的状态：
        - 未启动：async_task为None
        - 运行中：async_task存在且未完成
        - 已完成：async_task存在但已完成
        - 已取消：async_task存在但被取消
        - 异常退出：async_task存在但因异常退出
        
        使用场景：
        1. 状态检查：在执行操作前检查调度器状态
        2. 条件控制：根据调度器状态决定后续操作
        3. 用户界面：在UI中显示调度器状态
        4. 健康监控：监控系统各组件的运行状态
        
        基本使用示例：
        ```python
        import time
        import vectorbt as vbt
        
        manager = vbt.ScheduleManager()
        manager.every(1, 'minute').do(background_task)
        
        # 检查初始状态
        print(f"启动前状态: {manager.async_task_running}")  # False
        
        # 启动后台任务
        manager.start_in_background()
        
        # 检查启动后状态
        print(f"启动后状态: {manager.async_task_running}")  # True
        
        # 停止任务
        manager.stop()
        time.sleep(0.1)  # 等待任务完全停止
        
        # 检查停止后状态
        print(f"停止后状态: {manager.async_task_running}")  # False
        ```
        
        循环监控示例：
        ```python
        manager = vbt.ScheduleManager()
        manager.every(30, 'seconds').do(monitoring_task)
        manager.start_in_background()
        
        # 监控循环
        while True:
            if manager.async_task_running:
                print("✓ 调度器运行正常")
            else:
                print("✗ 调度器已停止")
                # 尝试重启
                manager.start_in_background()
            
            time.sleep(10)  # 每10秒检查一次
        ```
        
        条件控制示例：
        ```python
        def safe_operation():
            if not manager.async_task_running:
                print("调度器未运行，启动中...")
                manager.start_in_background()
                time.sleep(1)  # 等待启动
            
            if manager.async_task_running:
                print("调度器运行中，执行操作...")
                # 执行需要调度器运行的操作
            else:
                print("调度器启动失败")
        ```
        
        用户界面示例：
        ```python
        class SchedulerUI:
            def __init__(self):
                self.manager = vbt.ScheduleManager()
            
            def get_status_text(self):
                if self.manager.async_task_running:
                    return "🟢 调度器运行中"
                else:
                    return "🔴 调度器已停止"
            
            def toggle_scheduler(self):
                if self.manager.async_task_running:
                    self.manager.stop()
                else:
                    self.manager.start_in_background()
        ```
        
        注意事项：
        - 状态检查是即时的，可能在检查后立即变化
        - 任务可能在done()检查时刚好完成
        - 建议在关键操作前多次检查状态
        - 异常退出的任务也会被认为是"不运行"状态
        """
        # 检查异步任务是否存在且未完成
        # 两个条件必须同时满足：任务存在 AND 任务未完成
        return self.async_task is not None and not self.async_task.done()

    def stop(self) -> None:
        """
        停止异步任务 - 取消后台运行的调度任务
        
        这个方法用于优雅地停止后台运行的调度器任务。它会检查任务状态，
        如果任务正在运行则发送取消信号，让任务自然结束。
        
        停止流程：
        1. 检查是否有后台任务在运行
        2. 如果有，发送取消信号给任务
        3. 任务收到信号后会优雅地停止
        4. 触发done_callback()进行清理
        
        取消机制：
        - 使用asyncio.Task.cancel()发送取消信号
        - 任务会在下次await点收到CancelledError
        - async_start()中的try-except会捕获异常
        - 调度器会记录停止日志并清理资源
        
        使用场景：
        1. 程序关闭：在应用关闭时停止调度器
        2. 功能切换：切换到不同的调度配置
        3. 资源清理：释放调度器占用的资源
        4. 错误恢复：在检测到问题时重启调度器
        
        基本使用示例：
        ```python
        import time
        import vectorbt as vbt
        
        manager = vbt.ScheduleManager()
        manager.every(1, 'minute').do(background_task)
        
        # 启动后台调度器
        print("启动调度器...")
        manager.start_in_background()
        
        # 运行一段时间
        print("调度器运行中...")
        time.sleep(60)  # 运行1分钟
        
        # 停止调度器
        print("停止调度器...")
        manager.stop()
        
        # 等待完全停止
        while manager.async_task_running:
            time.sleep(0.1)
        print("调度器已完全停止")
        ```
        
        安全停止示例：
        ```python
        def safe_stop_scheduler(manager, timeout=5):
            if not manager.async_task_running:
                print("调度器未运行，无需停止")
                return True
            
            print("发送停止信号...")
            manager.stop()
            
            # 等待停止，带超时
            start_time = time.time()
            while manager.async_task_running:
                if time.time() - start_time > timeout:
                    print("停止超时，调度器可能无响应")
                    return False
                time.sleep(0.1)
            
            print("调度器已安全停止")
            return True
        ```
        
        重启调度器示例：
        ```python
        def restart_scheduler(manager):
            print("重启调度器...")
            
            # 停止当前调度器
            if manager.async_task_running:
                manager.stop()
                # 等待完全停止
                while manager.async_task_running:
                    time.sleep(0.1)
            
            # 重新启动
            manager.start_in_background()
            print("调度器重启完成")
        ```
        
        应用关闭时的清理：
        ```python
        import atexit
        import signal
        
        # 创建调度管理器
        manager = vbt.ScheduleManager()
        manager.every(1, 'minute').do(cleanup_task)
        manager.start_in_background()
        
        # 注册清理函数
        def cleanup():
            print("应用关闭，停止调度器...")
            manager.stop()
        
        # 正常退出时的清理
        atexit.register(cleanup)
        
        # 信号处理（Unix系统）
        def signal_handler(signum, frame):
            print(f"收到信号 {signum}，执行清理...")
            cleanup()
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        ```
        
        异常恢复示例：
        ```python
        def monitor_and_restart():
            manager = vbt.ScheduleManager()
            manager.every(1, 'minute').do(critical_task)
            
            while True:
                # 启动调度器
                manager.start_in_background()
                
                # 监控运行状态
                last_check = time.time()
                while manager.async_task_running:
                    time.sleep(10)
                    current_time = time.time()
                    
                    # 检查是否正常运行
                    if current_time - last_check > 300:  # 5分钟无响应
                        print("调度器可能无响应，重启...")
                        manager.stop()
                        break
                    last_check = current_time
                
                print("调度器停止，等待重启...")
                time.sleep(5)  # 等待5秒后重启
        ```
        
        注意事项：
        - stop()是非阻塞的，任务可能需要时间完全停止
        - 如果需要确保完全停止，应该检查async_task_running状态
        - 重复调用stop()是安全的，不会产生错误
        - 停止操作不会影响已注册的任务配置
        - 可以在停止后重新启动调度器
        """
        # 检查是否有后台任务正在运行
        if self.async_task_running:
            # 如果有，则取消该任务
            # cancel()方法会向任务发送CancelledError异常
            self.async_task.cancel()
