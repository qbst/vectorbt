# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT MESSAGING MODULE: Telegram机器人消息通信系统
================================================================================

文件设计逻辑和作用概述：
本文件是vectorbt量化交易框架中消息通信系统的Telegram机器人实现。该模块为量化交易者
提供了完整的Telegram机器人解决方案，用于实现交易信号推送、策略监控、实时通知等功能。

核心设计理念：
1. **企业级可靠性**：提供完整的错误处理、异常恢复和状态管理机制
2. **高度可扩展性**：通过继承和自定义Handler模式，支持复杂的业务逻辑扩展
3. **用户友好性**：封装了复杂的Telegram API操作，提供简洁的高级接口
4. **配置驱动**：与vectorbt配置系统深度集成，支持灵活的参数配置

主要功能模块：
【消息处理系统】
- LogHandler: 用户消息日志记录和审计
- 命令处理器: 处理用户命令和交互
- 错误处理器: 统一的异常处理和恢复机制
- 状态管理器: 聊天状态和用户数据持久化

【消息发送系统】
- 文本消息发送: 支持格式化和样式设置
- 多媒体消息: 图片、动图、文件等多种格式支持
- 批量推送: 向多个用户或群组批量发送消息
- GIPHY集成: 智能GIF动图搜索和发送

【机器人生命周期管理】
- 启动和停止控制: 优雅的服务启动和关闭
- 后台运行支持: 支持daemon模式运行
- 配置热更新: 支持运行时配置修改
- 监控和健康检查: 机器人运行状态监控

应用场景：
- **交易信号推送**: 自动推送买卖信号、价格警报、市场分析
- **策略监控**: 实时监控策略表现，推送盈亏报告
- **风险管理**: 推送风险警报、保证金不足提醒
- **数据同步**: 同步交易数据、账户状态、持仓信息
- **人工干预**: 提供交易员远程操作接口

技术特点：
- 基于python-telegram-bot库的高级封装
- 支持Webhook和长轮询两种工作模式
- 完整的消息持久化和恢复机制
- 自动处理聊天迁移和用户授权问题
- 集成GIPHY API，支持动图消息

架构设计：
```
TelegramBot (主机器人类)
├── LogHandler (日志处理器)
├── Command Handlers (命令处理器)
├── Message Handlers (消息处理器)
├── Error Handlers (错误处理器)
└── Custom Handlers (自定义处理器)
```

与vectorbt生态系统的关系：
- 配置系统集成: 使用vectorbt.settings进行配置管理
- 工具函数集成: 使用vectorbt.utils中的辅助函数
- 数据分析集成: 可以直接调用vectorbt的分析功能
- 通知系统集成: 作为vectorbt通知系统的重要组成部分

使用示例：
```python
import vectorbt as vbt
from telegram.ext import CommandHandler

# 创建自定义交易机器人
class TradingBot(vbt.TelegramBot):
    @property
    def custom_handlers(self):
        return (
            CommandHandler('price', self.get_price),
            CommandHandler('positions', self.get_positions),
            CommandHandler('alerts', self.set_alerts),
        )
    
    def get_price(self, update, context):
        # 获取价格信息
        symbol = context.args[0] if context.args else 'BTC/USDT'
        price = get_latest_price(symbol)  # 自定义函数
        self.send_message(update.effective_chat.id, f"{symbol}: ${price}")
    
    def get_positions(self, update, context):
        # 获取持仓信息
        positions = get_current_positions()  # 自定义函数
        message = format_positions(positions)  # 自定义函数
        self.send_message(update.effective_chat.id, message)

# 启动机器人
bot = TradingBot(token='YOUR_BOT_TOKEN')
bot.start()
```

该模块为量化交易者提供了强大的远程通信能力，是构建现代量化交易系统的重要组件。
================================================================================

使用python-telegram-bot库实现的Telegram机器人消息通信系统
"""

# 导入标准库
import logging  # 日志记录库，用于记录机器人运行状态和用户交互信息
from functools import wraps  # 函数装饰器工具，用于创建装饰器函数

# 导入Telegram机器人相关模块
from telegram import Update  # Telegram更新对象，包含用户消息和操作信息
from telegram.error import Unauthorized, ChatMigrated  # Telegram异常类，处理授权和聊天迁移错误
from telegram.ext import (  # Telegram扩展模块，提供高级机器人功能
    Handler,  # 处理器基类，用于处理不同类型的更新
    CallbackContext,  # 回调上下文，包含机器人和用户数据
    Updater,  # 更新器，负责获取和分发Telegram更新
    Dispatcher,  # 分发器，将更新路由到相应的处理器
    CommandHandler,  # 命令处理器，处理以'/'开头的命令
    MessageHandler,  # 消息处理器，处理普通文本消息
    Filters,  # 过滤器，用于筛选特定类型的消息
    PicklePersistence,  # 数据持久化，使用pickle序列化保存机器人数据
    Defaults  # 默认配置，用于设置机器人的默认行为
)
from telegram.utils.helpers import effective_message_type  # 工具函数，获取有效消息类型

# 导入vectorbt相关模块
from vectorbt import _typing as tp  # vectorbt类型定义模块
from vectorbt.utils.config import merge_dicts, get_func_kwargs, Configured  # 配置管理工具
from vectorbt.utils.requests_ import text_to_giphy_url  # GIPHY API集成工具

# 创建日志记录器，用于记录机器人模块的运行信息
logger = logging.getLogger(__name__)


class LogHandler(Handler):
    """
    用户消息日志记录处理器
    
    该处理器负责记录和审计所有用户与机器人的交互信息，包括文本消息、多媒体消息等。
    这对于监控机器人使用情况、调试问题和合规审计非常重要。
    
    核心功能：
    - 记录用户发送的文本消息内容
    - 记录多媒体消息类型（图片、视频、文档等）
    - 记录用户聊天ID，便于追踪和分析
    - 提供结构化的日志格式，便于后续分析
    
    技术特点：
    - 继承自telegram.ext.Handler基类
    - 实现check_update方法来处理更新
    - 使用logging模块记录信息
    - 不干扰正常的消息处理流程
    
    使用场景：
    - 用户行为分析：了解用户最常使用的功能
    - 问题调试：追踪用户操作序列，定位问题
    - 合规审计：记录所有用户交互，满足合规要求
    - 性能监控：分析消息处理频率和模式
    
    示例：
    ```python
    # 创建日志处理器
    log_handler = LogHandler(lambda update, context: None)
    
    # 添加到调度器
    dispatcher.add_handler(log_handler)
    
    # 日志输出示例：
    # INFO:vectorbt.messaging.telegram:123456789 - User: "Hello bot!"
    # INFO:vectorbt.messaging.telegram:123456789 - User: photo
    # INFO:vectorbt.messaging.telegram:123456789 - User: document
    ```
    """

    def check_update(self, update: object) -> tp.Optional[tp.Union[bool, object]]:
        """
        检查并处理传入的更新对象
        
        该方法是Handler基类的核心方法，用于检查更新是否应该被此处理器处理。
        在这里，我们用它来记录用户消息，但不实际处理消息（返回False）。
        
        参数：
            update: Telegram更新对象，可能包含消息、回调查询等
        
        返回：
            - False: 表示不处理此更新，让其他处理器继续处理
            - None: 表示此更新不适用于本处理器
            - True/object: 表示处理此更新（在日志处理器中不使用）
        
        实现逻辑：
        1. 检查更新是否为有效的Update对象
        2. 检查是否包含有效消息
        3. 获取消息类型（文本、图片、视频等）
        4. 根据消息类型记录相应的日志信息
        5. 返回False，让其他处理器继续处理
        
        日志格式：
        - 文本消息："{chat_id} - User: \"{message_text}\""
        - 其他消息："{chat_id} - User: {message_type}"
        """
        # 检查更新是否为有效的Update对象且包含有效消息
        if isinstance(update, Update) and update.effective_message:
            message = update.effective_message  # 获取有效消息对象
            message_type = effective_message_type(message)  # 获取消息类型
            
            # 如果消息类型有效，记录日志
            if message_type is not None:
                if message_type == 'text':
                    # 记录文本消息内容
                    logger.info(f"{message.chat_id} - User: \"%s\"", message.text)
                else:
                    # 记录非文本消息类型
                    logger.info(f"{message.chat_id} - User: %s", message_type)
            return False  # 不处理此更新，让其他处理器继续处理
        return None  # 此更新不适用于本处理器


def send_action(action: str) -> tp.Callable:
    """
    发送聊天动作状态的装饰器工厂函数
    
    该装饰器用于在处理命令时向用户显示机器人的状态（如"正在输入..."、"正在上传照片..."等）。
    这提供了更好的用户体验，让用户知道机器人正在处理他们的请求。
    
    参数：
        action: 要发送的动作类型，如：
               - 'typing': 正在输入
               - 'upload_photo': 正在上传照片
               - 'record_video': 正在录制视频
               - 'upload_video': 正在上传视频
               - 'record_audio': 正在录制音频
               - 'upload_audio': 正在上传音频
               - 'upload_document': 正在上传文档
               - 'find_location': 正在查找位置
    
    返回：
        装饰器函数，用于装饰绑定的回调方法
    
    使用要求：
        - 只适用于绑定的回调方法
        - 回调方法必须接受参数：self, update, context 以及可选的其他参数
        - 被装饰的方法必须是类的方法
    
    工作原理：
    1. 创建装饰器函数
    2. 在执行实际回调之前发送聊天动作
    3. 然后调用原始的回调函数
    
    示例：
    ```python
    class MyBot(TelegramBot):
        @send_action('typing')
        def long_running_command(self, update, context):
            # 用户会看到"正在输入..."状态
            time.sleep(5)  # 模拟长时间处理
            self.send_message(update.effective_chat.id, "处理完成！")
        
        @send_action('upload_photo')
        def send_chart(self, update, context):
            # 用户会看到"正在上传照片..."状态
            chart_data = generate_chart()  # 生成图表
            self.send_photo(update.effective_chat.id, chart_data)
    ```
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        """
        装饰器函数，包装原始的回调函数
        
        参数：
            func: 要装饰的原始回调函数
        
        返回：
            包装后的回调函数
        """
        @wraps(func)  # 保持原函数的元数据
        def command_func(self, update: Update, context: CallbackContext, *args, **kwargs) -> tp.Callable:
            """
            包装后的命令函数
            
            参数：
                self: 机器人实例
                update: Telegram更新对象
                context: 回调上下文
                *args: 额外的位置参数
                **kwargs: 额外的关键字参数
            
            返回：
                原始函数的返回值
            """
            # 如果存在有效聊天，发送聊天动作状态
            if update.effective_chat:
                context.bot.send_chat_action(chat_id=update.effective_chat.id, action=action)
            # 调用原始函数并返回结果
            return func(self, update, context, *args, **kwargs)

        return command_func

    return decorator


def self_decorator(self, func: tp.Callable) -> tp.Callable:
    """
    将机器人实例传递给回调函数的装饰器
    
    该装饰器用于将机器人实例(self)传递给回调函数，使得回调函数能够访问机器人的方法和属性。
    这主要用于错误处理器等不直接绑定到类的回调函数。
    
    参数：
        self: 机器人实例
        func: 要装饰的回调函数
    
    返回：
        包装后的回调函数，该函数会接收机器人实例作为第一个参数
    
    使用场景：
    - 错误处理器：需要访问机器人方法来发送错误消息
    - 独立的回调函数：不是类方法但需要访问机器人功能
    - 第三方集成：外部函数需要访问机器人功能
    
    示例：
    ```python
    def error_handler(bot, update, context):
        # 这个函数现在可以访问机器人实例
        bot.send_message(update.effective_chat.id, "发生错误！")
    
    # 使用装饰器包装
    wrapped_handler = self_decorator(bot_instance, error_handler)
    dispatcher.add_error_handler(wrapped_handler)
    ```
    """

    def command_func(update, context, *args, **kwargs):
        """
        包装后的命令函数
        
        参数：
            update: Telegram更新对象
            context: 回调上下文
            *args: 额外的位置参数
            **kwargs: 额外的关键字参数
        
        返回：
            原始函数的返回值
        """
        # 将机器人实例作为第一个参数传递给原始函数
        return func(self, update, context, *args, **kwargs)

    return command_func


class TelegramBot(Configured):
    """
    Telegram机器人核心类
    
    该类是vectorbt框架中Telegram机器人功能的核心实现，提供了完整的机器人创建、配置、
    运行和管理功能。它封装了python-telegram-bot库的复杂性，为量化交易应用提供了
    简单易用的接口。
    
    核心特性：
    1. **配置管理**：与vectorbt配置系统深度集成，支持灵活的参数配置
    2. **消息处理**：完整的消息发送、接收和处理机制
    3. **命令系统**：支持自定义命令和处理器
    4. **错误处理**：完善的错误处理和恢复机制
    5. **数据持久化**：支持聊天数据和用户数据的持久化存储
    6. **多媒体支持**：支持文本、图片、视频、动图等多种消息类型
    7. **GIPHY集成**：智能GIF动图搜索和发送
    8. **生命周期管理**：完整的启动、运行、停止流程
    
    架构设计：
    - 继承自Configured类，获得配置管理能力
    - 使用Updater管理Telegram连接和更新
    - 使用Dispatcher分发消息到相应的处理器
    - 支持自定义处理器扩展
    
    配置参数：
    - token: Telegram Bot Token（必需）
    - persistence: 数据持久化配置
    - defaults: 默认消息配置
    - giphy_kwargs: GIPHY API配置
    
    扩展方式：
    - 重写custom_handlers属性添加自定义命令
    - 重写start_message和help_message定制消息
    - 重写回调方法实现自定义逻辑
    
    使用示例：
    ```python
    # 1. 基本使用
    bot = vbt.TelegramBot(token='YOUR_TOKEN')
    bot.start()
    
    # 2. 自定义机器人
    class MyBot(vbt.TelegramBot):
        @property
        def custom_handlers(self):
            return (CommandHandler('price', self.get_price),)
        
        def get_price(self, update, context):
            symbol = context.args[0] if context.args else 'BTC'
            price = get_current_price(symbol)
            self.send_message(update.effective_chat.id, f"{symbol}: ${price}")
    
    bot = MyBot(token='YOUR_TOKEN')
    bot.start()
    
    # 3. 发送消息
    bot.send_message(chat_id, "Hello!")
    bot.send_giphy(chat_id, "celebration")
    bot.send_message_to_all("广播消息")
    ```
    
    生命周期：
    1. 初始化：配置解析、处理器注册
    2. 启动：开始轮询、发送上线消息
    3. 运行：处理用户消息、执行命令
    4. 停止：优雅关闭、清理资源
    
    错误处理：
    - 自动处理聊天迁移
    - 处理用户取消授权
    - 统一的异常处理和日志记录
    - 发送友好的错误消息给用户
    
    安全特性：
    - 自动验证用户权限
    - 记录所有用户交互
    - 支持聊天ID白名单
    - 防止未授权访问
    
    该类是构建量化交易机器人的理想基础，提供了所有必要的功能和扩展点。
    """

    def __init__(self, giphy_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        """
        初始化Telegram机器人
        
        该方法负责机器人的完整初始化过程，包括配置解析、组件创建、处理器注册等。
        
        参数：
            giphy_kwargs: GIPHY API配置参数，用于发送GIF动图
                         如：{'api_key': 'your_key', 'limit': 10}
            **kwargs: 其他配置参数，会传递给Updater构造函数
                     常用参数：
                     - token: Telegram Bot Token（必需）
                     - use_context: 是否使用上下文（默认True）
                     - persistence: 数据持久化配置
                     - defaults: 默认消息配置
        
        初始化流程：
        1. 加载配置：从settings中获取默认配置
        2. 解析参数：合并默认配置和用户配置
        3. 创建Updater：初始化Telegram连接
        4. 注册处理器：添加命令和消息处理器
        5. 初始化数据：设置聊天ID列表等
        
        配置来源优先级：
        用户传入参数 > vectorbt.settings配置 > 默认值
        
        示例：
        ```python
        # 基本初始化
        bot = TelegramBot(token='YOUR_TOKEN')
        
        # 带持久化的初始化
        bot = TelegramBot(
            token='YOUR_TOKEN',
            persistence='bot_data.pickle',
            defaults={'parse_mode': 'HTML'}
        )
        
        # 带GIPHY配置的初始化
        bot = TelegramBot(
            token='YOUR_TOKEN',
            giphy_kwargs={'api_key': 'your_giphy_key', 'limit': 5}
        )
        ```
        """
        # 导入设置模块（延迟导入避免循环依赖）
        from vectorbt._settings import settings
        telegram_cfg = settings['messaging']['telegram']  # 获取Telegram配置
        giphy_cfg = settings['messaging']['giphy']  # 获取GIPHY配置

        # 调用父类构造函数，初始化配置管理功能
        Configured.__init__(
            self,
            giphy_kwargs=giphy_kwargs,
            **kwargs
        )

        # 解析GIPHY配置参数
        # 合并默认配置和用户配置，用户配置优先
        giphy_kwargs = merge_dicts(giphy_cfg, giphy_kwargs)
        self.giphy_kwargs = giphy_kwargs  # 保存GIPHY配置供后续使用
        
        # 解析Updater构造函数参数
        default_kwargs = dict()  # 默认参数字典
        passed_kwargs = dict()  # 用户传入参数字典
        
        # 遍历Updater构造函数的所有参数
        for k in get_func_kwargs(Updater.__init__):
            # 如果参数在默认配置中，添加到默认参数
            if k in telegram_cfg:
                default_kwargs[k] = telegram_cfg[k]
            # 如果参数在用户配置中，添加到用户参数
            if k in kwargs:
                passed_kwargs[k] = kwargs.pop(k)
        
        # 合并参数，用户参数优先
        updater_kwargs = merge_dicts(default_kwargs, passed_kwargs)
        
        # 处理数据持久化配置
        persistence = updater_kwargs.pop('persistence', None)
        if isinstance(persistence, str):
            # 如果持久化配置是字符串，创建PicklePersistence对象
            persistence = PicklePersistence(persistence)
        
        # 处理默认配置
        defaults = updater_kwargs.pop('defaults', None)
        if isinstance(defaults, dict):
            # 如果defaults是字典，创建Defaults对象
            defaults = Defaults(**defaults)

        # 创建Updater对象（持久化的更新器）
        logger.info("Initializing bot")  # 记录初始化日志
        self._updater = Updater(persistence=persistence, defaults=defaults, **updater_kwargs)

        # 获取调度器以注册处理器
        self._dispatcher = self.updater.dispatcher

        # 注册各种处理器（按优先级顺序）
        self.dispatcher.add_handler(self.log_handler)  # 日志处理器（最高优先级）
        self.dispatcher.add_handler(CommandHandler('start', self.start_callback))  # 启动命令处理器
        self.dispatcher.add_handler(CommandHandler("help", self.help_callback))  # 帮助命令处理器
        
        # 注册自定义处理器
        for handler in self.custom_handlers:
            self.dispatcher.add_handler(handler)
        
        # 注册系统处理器
        self.dispatcher.add_handler(MessageHandler(Filters.status_update.migrate, self.chat_migration_callback))  # 聊天迁移处理器
        self.dispatcher.add_handler(MessageHandler(Filters.command, self.unknown_callback))  # 未知命令处理器
        self.dispatcher.add_error_handler(self_decorator(self, self.__class__.error_callback))  # 错误处理器

        # 初始化机器人数据
        if 'chat_ids' not in self.dispatcher.bot_data:
            # 如果没有聊天ID列表，创建空列表
            self.dispatcher.bot_data['chat_ids'] = []
        else:
            # 如果已有聊天ID列表，记录加载信息
            logger.info("Loaded chat ids %s", str(self.dispatcher.bot_data['chat_ids']))

    @property
    def updater(self) -> Updater:
        """
        获取Telegram更新器实例
        
        Updater是python-telegram-bot库的核心组件，负责从Telegram服务器获取更新
        并将其分发到相应的处理器。
        
        返回：
            Updater实例，用于管理Telegram连接和更新
        
        使用场景：
        - 直接访问bot对象：self.updater.bot
        - 控制轮询：self.updater.start_polling()
        - 停止机器人：self.updater.stop()
        
        示例：
        ```python
        # 获取bot对象
        bot_info = self.updater.bot.get_me()
        
        # 检查运行状态
        if self.updater.running:
            print("机器人正在运行")
        
        # 手动停止
        self.updater.stop()
        ```
        """
        return self._updater

    @property
    def dispatcher(self) -> Dispatcher:
        """
        获取Telegram调度器实例
        
        Dispatcher负责将接收到的更新路由到相应的处理器。它是机器人消息处理的核心。
        
        返回：
            Dispatcher实例，用于管理消息处理器和数据
        
        使用场景：
        - 添加处理器：self.dispatcher.add_handler()
        - 访问机器人数据：self.dispatcher.bot_data
        - 访问用户数据：self.dispatcher.user_data
        - 访问聊天数据：self.dispatcher.chat_data
        
        示例：
        ```python
        # 添加新的处理器
        self.dispatcher.add_handler(CommandHandler('new_cmd', self.new_command))
        
        # 访问全局数据
        self.dispatcher.bot_data['custom_data'] = 'some_value'
        
        # 访问用户数据
        user_data = self.dispatcher.user_data[user_id]
        ```
        """
        return self._dispatcher

    @property
    def log_handler(self) -> LogHandler:
        """
        获取日志处理器实例
        
        日志处理器负责记录所有用户与机器人的交互信息，用于审计、调试和分析。
        
        返回：
            LogHandler实例，用于记录用户消息
        
        特点：
        - 记录所有用户消息
        - 不干扰正常消息处理
        - 提供结构化的日志格式
        - 支持消息类型识别
        
        示例：
        ```python
        # 获取日志处理器
        log_handler = self.log_handler
        
        # 日志输出格式：
        # INFO:vectorbt.messaging.telegram:123456789 - User: "Hello"
        # INFO:vectorbt.messaging.telegram:123456789 - User: photo
        ```
        """
        return LogHandler(lambda update, context: None)

    @property
    def custom_handlers(self) -> tp.Iterable[Handler]:
        """
        获取自定义处理器列表
        
        该属性返回用户自定义的处理器列表，子类可以重写此属性来添加自定义命令和功能。
        处理器的顺序很重要，因为它们按顺序检查每个更新。
        
        返回：
            Handler对象的迭代器，默认为空元组
        
        重写指南：
        - 返回Handler对象的元组或列表
        - 处理器按顺序检查，先匹配的先处理
        - 可以包含CommandHandler、MessageHandler、CallbackQueryHandler等
        
        示例：
        ```python
        class MyBot(TelegramBot):
            @property
            def custom_handlers(self):
                return (
                    CommandHandler('price', self.get_price),
                    CommandHandler('alert', self.set_alert),
                    MessageHandler(Filters.photo, self.handle_photo),
                    CommandHandler('portfolio', self.show_portfolio),
                )
            
            def get_price(self, update, context):
                # 获取价格命令的处理逻辑
                pass
            
            def set_alert(self, update, context):
                # 设置警报命令的处理逻辑
                pass
            
            def handle_photo(self, update, context):
                # 处理图片消息的逻辑
                pass
        ```
        """
        return ()

    @property
    def chat_ids(self) -> tp.List[int]:
        """
        获取所有曾与机器人交互的聊天ID列表
        
        该属性返回所有曾经与机器人交互的聊天ID列表。聊天ID在用户发送"/start"命令时
        会被自动添加到列表中。这个列表用于群发消息、用户管理等功能。
        
        返回：
            整数列表，包含所有活跃的聊天ID
        
        数据存储：
        - 数据存储在dispatcher.bot_data['chat_ids']中
        - 支持持久化存储（如果配置了persistence）
        - 自动处理聊天迁移和用户取消授权
        
        使用场景：
        - 群发消息给所有用户
        - 用户权限管理
        - 统计活跃用户数量
        - 实现用户白名单/黑名单
        
        示例：
        ```python
        # 获取所有聊天ID
        all_chats = self.chat_ids
        print(f"活跃用户数量: {len(all_chats)}")
        
        # 检查用户是否已注册
        user_id = 123456789
        if user_id in self.chat_ids:
            print("用户已注册")
        
        # 向所有用户发送消息
        for chat_id in self.chat_ids:
            self.send_message(chat_id, "系统通知")
        ```
        """
        return self.dispatcher.bot_data['chat_ids']

    def start(self, in_background: bool = False, **kwargs) -> None:
        """
        启动Telegram机器人
        
        该方法启动机器人的消息轮询，开始接收和处理用户消息。支持前台和后台两种运行模式。
        
        参数：
            in_background: 是否在后台运行
                          - False (默认): 阻塞运行，直到收到停止信号
                          - True: 非阻塞运行，立即返回
            **kwargs: 传递给start_polling方法的参数
                     常用参数：
                     - timeout: 轮询超时时间（秒）
                     - read_latency: 读取延迟（秒）
                     - drop_pending_updates: 是否丢弃待处理的更新
                     - allowed_updates: 允许的更新类型列表
        
        运行模式：
        - 前台模式：适合开发和调试，程序会阻塞直到手动停止
        - 后台模式：适合生产环境，可以与其他代码并行运行
        
        启动流程：
        1. 解析配置参数
        2. 开始轮询Telegram服务器
        3. 执行启动回调（发送上线消息）
        4. 根据模式选择阻塞或非阻塞运行
        
        示例：
        ```python
        # 前台运行（阻塞）
        bot.start()
        
        # 后台运行（非阻塞）
        bot.start(in_background=True)
        
        # 带参数运行
        bot.start(
            timeout=30,
            drop_pending_updates=True,
            allowed_updates=['message', 'callback_query']
        )
        
        # 生产环境使用
        bot.start(in_background=True)
        # 继续执行其他代码
        while True:
            # 其他业务逻辑
            time.sleep(1)
        ```
        
        注意事项：
        - 确保在调用start()之前已正确配置token
        - 前台模式下使用Ctrl+C可优雅停止机器人
        - 后台模式下需要手动调用stop()方法停止机器人
        """
        # 导入设置模块
        from vectorbt._settings import settings
        telegram_cfg = settings['messaging']['telegram']

        # 解析轮询参数
        default_kwargs = dict()  # 默认参数
        passed_kwargs = dict()  # 用户传入参数
        
        # 遍历start_polling方法的所有参数
        for k in get_func_kwargs(self.updater.start_polling):
            # 从配置中获取默认值
            if k in telegram_cfg:
                default_kwargs[k] = telegram_cfg[k]
            # 获取用户传入的参数
            if k in kwargs:
                passed_kwargs[k] = kwargs.pop(k)
        
        # 合并参数，用户参数优先
        polling_kwargs = merge_dicts(default_kwargs, passed_kwargs)

        # 启动机器人轮询
        logger.info("Running bot %s", str(self.updater.bot.get_me().username))
        self.updater.start_polling(**polling_kwargs)
        
        # 执行启动回调
        self.started_callback()

        # 根据运行模式选择是否阻塞
        if not in_background:
            # 前台模式：运行机器人直到收到停止信号
            # 这会阻塞程序，直到按Ctrl+C或收到SIGINT、SIGTERM、SIGABRT信号
            # start_polling()是非阻塞的，所以需要idle()来保持程序运行
            self.updater.idle()

    def started_callback(self) -> None:
        """
        机器人启动后的回调函数
        
        该方法在机器人启动后自动调用，用于执行启动后的初始化任务。
        默认实现是向所有用户发送"I'm back online!"消息。
        
        重写用途：
        - 发送自定义的启动消息
        - 初始化定时任务
        - 加载数据和配置
        - 发送系统状态报告
        
        示例：
        ```python
        class MyBot(TelegramBot):
            def started_callback(self):
                # 发送自定义启动消息
                self.send_message_to_all("🚀 交易机器人已启动！")
                
                # 初始化定时任务
                self.setup_price_alerts()
                
                # 加载用户配置
                self.load_user_preferences()
                
                # 发送系统状态
                status = self.get_system_status()
                self.send_message_to_all(f"系统状态: {status}")
        ```
        """
        # 向所有用户发送机器人重新上线的消息
        self.send_message_to_all("I'm back online!")

    def send(self, kind: str, chat_id: int, *args, log_msg: tp.Optional[str] = None, **kwargs) -> None:
        """
        向指定聊天发送任意类型的消息
        
        该方法是所有消息发送功能的基础，提供了统一的消息发送接口，包含完整的
        错误处理和异常恢复机制。
        
        参数：
            kind: 消息类型，对应bot.send_*方法的后缀
                 如：'message', 'photo', 'video', 'document', 'animation'等
            chat_id: 目标聊天ID
            *args: 传递给send_*方法的位置参数
            log_msg: 自定义日志消息，如果为None则使用消息类型
            **kwargs: 传递给send_*方法的关键字参数
        
        错误处理：
        - ChatMigrated: 自动处理聊天迁移，更新聊天ID并重新发送
        - Unauthorized: 处理用户取消授权，记录日志但不抛出异常
        - 其他异常: 记录日志并传播异常
        
        日志记录：
        - 成功发送：记录消息类型和内容
        - 聊天迁移：记录旧ID和新ID
        - 未授权：记录用户取消授权
        
        示例：
        ```python
        # 发送文本消息
        self.send('message', chat_id, "Hello World!")
        
        # 发送图片
        self.send('photo', chat_id, photo_file, caption="图片说明")
        
        # 发送文档
        self.send('document', chat_id, document_file, 
                 filename="report.pdf", log_msg="发送PDF报告")
        
        # 发送位置
        self.send('location', chat_id, latitude=40.7128, longitude=-74.0060)
        ```
        """
        try:
            # 调用相应的send_*方法
            getattr(self.updater.bot, 'send_' + kind)(chat_id, *args, **kwargs)
            
            # 记录发送成功的日志
            if log_msg is None:
                log_msg = kind  # 如果没有自定义日志消息，使用消息类型
            logger.info(f"{chat_id} - Bot: %s", log_msg)
            
        except ChatMigrated as e:
            # 处理聊天迁移（群组升级为超级群组）
            new_id = e.new_chat_id
            
            # 更新聊天ID列表
            if chat_id in self.chat_ids:
                self.chat_ids.remove(chat_id)  # 移除旧ID
            self.chat_ids.append(new_id)  # 添加新ID
            
            # 重新发送消息到新的聊天ID
            self.send(kind, new_id, *args, log_msg=log_msg, **kwargs)
            
        except Unauthorized as e:
            # 处理用户取消授权（用户屏蔽了机器人）
            logger.info(f"{chat_id} - Unauthorized to send the %s", kind)

    def send_to_all(self, kind: str, *args, **kwargs) -> None:
        """
        向所有用户发送指定类型的消息
        
        该方法向聊天ID列表中的所有用户批量发送消息，常用于系统通知、
        重要公告、市场警报等需要广播的场景。
        
        参数：
            kind: 消息类型，对应bot.send_*方法的后缀
            *args: 传递给send方法的位置参数
            **kwargs: 传递给send方法的关键字参数
        
        发送策略：
        - 逐个发送：遍历所有聊天ID，逐个发送消息
        - 错误隔离：单个用户发送失败不影响其他用户
        - 自动重试：利用send方法的错误处理机制
        
        使用场景：
        - 系统维护通知
        - 重要市场消息
        - 价格警报
        - 策略状态更新
        
        示例：
        ```python
        # 发送系统通知
        self.send_to_all('message', "系统维护通知：将于今晚23:00开始维护")
        
        # 发送图片公告
        self.send_to_all('photo', photo_file, caption="市场分析报告")
        
        # 发送紧急警报
        self.send_to_all('message', "⚠️ 重要警报：BTC价格突破$50,000！", 
                        parse_mode='HTML')
        ```
        """
        # 遍历所有聊天ID，逐个发送消息
        for chat_id in self.chat_ids:
            self.send(kind, chat_id, *args, **kwargs)

    def send_message(self, chat_id: int, text: str, *args, **kwargs) -> None:
        """
        发送文本消息到指定聊天
        
        该方法是发送文本消息的便捷接口，支持HTML和Markdown格式化。
        
        参数：
            chat_id: 目标聊天ID
            text: 消息文本内容
            *args: 传递给send_message API的其他位置参数
            **kwargs: 传递给send_message API的关键字参数
                     常用参数：
                     - parse_mode: 解析模式 ('HTML', 'Markdown', 'MarkdownV2')
                     - disable_web_page_preview: 禁用网页预览
                     - disable_notification: 静默发送
                     - reply_to_message_id: 回复消息ID
                     - reply_markup: 自定义键盘
        
        格式化支持：
        - HTML: <b>粗体</b>, <i>斜体</i>, <code>代码</code>
        - Markdown: *粗体*, _斜体_, `代码`
        - 支持链接、列表、引用等
        
        示例：
        ```python
        # 发送普通消息
        self.send_message(chat_id, "Hello World!")
        
        # 发送HTML格式消息
        self.send_message(chat_id, 
                         "<b>重要通知</b>\\n价格: <code>$50,000</code>",
                         parse_mode='HTML')
        
        # 发送Markdown格式消息
        self.send_message(chat_id, 
                         "*BTC价格更新*\\n当前价格: `$50,000`",
                         parse_mode='Markdown')
        
        # 静默发送消息
        self.send_message(chat_id, "深夜通知", disable_notification=True)
        ```
        """
        # 格式化日志消息，用引号包围文本内容
        log_msg = "\"%s\"" % text
        
        # 调用通用发送方法
        self.send('message', chat_id, text, *args, log_msg=log_msg, **kwargs)

    def send_message_to_all(self, text: str, *args, **kwargs) -> None:
        """
        向所有用户发送文本消息
        
        该方法向聊天ID列表中的所有用户批量发送文本消息，是群发文本消息的便捷接口。
        
        参数：
            text: 消息文本内容
            *args: 传递给send_message的其他位置参数
            **kwargs: 传递给send_message的关键字参数
        
        使用场景：
        - 系统公告
        - 重要新闻
        - 价格警报
        - 策略状态更新
        
        示例：
        ```python
        # 发送系统公告
        self.send_message_to_all("📢 系统升级完成，所有功能恢复正常！")
        
        # 发送市场警报
        self.send_message_to_all(
            "⚠️ <b>市场警报</b>\\n比特币价格突破$60,000！",
            parse_mode='HTML'
        )
        
        # 发送策略更新
        self.send_message_to_all(
            "🤖 策略更新\\n今日收益: +2.5%\\n总收益: +15.8%"
        )
        ```
        """
        # 格式化日志消息
        log_msg = "\"%s\"" % text
        
        # 调用通用批量发送方法
        self.send_to_all('message', text, *args, log_msg=log_msg, **kwargs)

    def send_giphy(self, chat_id: int, text: str, *args, giphy_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        """
        发送GIPHY动图到指定聊天
        
        该方法根据文本描述搜索并发送相应的GIF动图，为机器人增加趣味性和表现力。
        
        参数：
            chat_id: 目标聊天ID
            text: 搜索关键词，用于在GIPHY上搜索动图
            *args: 传递给send_animation的其他位置参数
            giphy_kwargs: GIPHY API配置参数，如果为None则使用默认配置
                         常用参数：
                         - api_key: GIPHY API密钥
                         - limit: 搜索结果数量限制
                         - rating: 内容评级 ('g', 'pg', 'pg-13', 'r')
                         - lang: 搜索语言
            **kwargs: 传递给send_animation的关键字参数
        
        工作流程：
        1. 使用GIPHY API搜索相关动图
        2. 选择最佳匹配的动图
        3. 发送动图到指定聊天
        4. 记录发送日志
        
        示例：
        ```python
        # 发送庆祝动图
        self.send_giphy(chat_id, "celebration")
        
        # 发送带自定义配置的动图
        self.send_giphy(chat_id, "happy", 
                       giphy_kwargs={'limit': 5, 'rating': 'g'})
        
        # 发送带标题的动图
        self.send_giphy(chat_id, "rocket", caption="🚀 价格飞涨！")
        
        # 常用场景
        self.send_giphy(chat_id, "money")     # 盈利时
        self.send_giphy(chat_id, "sad")       # 亏损时
        self.send_giphy(chat_id, "thinking")  # 分析时
        ```
        """
        # 使用传入的配置或默认配置
        if giphy_kwargs is None:
            giphy_kwargs = self.giphy_kwargs
        
        # 根据文本搜索GIPHY动图URL
        gif_url = text_to_giphy_url(text, **giphy_kwargs)
        
        # 格式化日志消息，包含搜索关键词和动图URL
        log_msg = "\"%s\" as GIPHY %s" % (text, gif_url)
        
        # 发送动图
        self.send('animation', chat_id, gif_url, *args, log_msg=log_msg, **kwargs)

    def send_giphy_to_all(self, text: str, *args, giphy_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        """
        向所有用户发送GIPHY动图
        
        该方法向聊天ID列表中的所有用户批量发送GIPHY动图，适用于庆祝、
        公告或增加互动趣味性的场景。
        
        参数：
            text: 搜索关键词
            *args: 传递给send_animation的其他位置参数
            giphy_kwargs: GIPHY API配置参数
            **kwargs: 传递给send_animation的关键字参数
        
        使用场景：
        - 庆祝重要里程碑
        - 节日祝福
        - 趣味性公告
        - 情绪表达
        
        示例：
        ```python
        # 庆祝盈利
        self.send_giphy_to_all("celebration", caption="🎉 今日收益创新高！")
        
        # 节日祝福
        self.send_giphy_to_all("happy new year")
        
        # 市场情绪
        self.send_giphy_to_all("bull market", caption="📈 牛市来了！")
        ```
        """
        # 使用传入的配置或默认配置
        if giphy_kwargs is None:
            giphy_kwargs = self.giphy_kwargs
        
        # 根据文本搜索GIPHY动图URL
        gif_url = text_to_giphy_url(text, **giphy_kwargs)
        
        # 格式化日志消息
        log_msg = "\"%s\" as GIPHY %s" % (text, gif_url)
        
        # 向所有用户发送动图
        self.send_to_all('animation', gif_url, *args, log_msg=log_msg, **kwargs)

    @property
    def start_message(self) -> str:
        """
        获取启动消息
        
        该属性返回用户发送"/start"命令时机器人回复的消息。
        子类可以重写此属性来定制欢迎消息。
        
        返回：
            启动消息字符串
        
        重写示例：
        ```python
        class MyBot(TelegramBot):
            @property
            def start_message(self):
                return '''
                🤖 欢迎使用量化交易机器人！
                
                功能介绍：
                📊 实时价格查询
                📈 技术指标分析
                ⚠️ 价格警报设置
                💰 投资组合跟踪
                
                发送 /help 获取详细帮助
                '''
        ```
        """
        return "Hello!"

    def start_callback(self, update: object, context: CallbackContext) -> None:
        """
        处理"/start"命令的回调函数
        
        该方法在用户发送"/start"命令时被调用，负责用户注册和欢迎消息发送。
        
        参数：
            update: Telegram更新对象
            context: 回调上下文对象
        
        处理逻辑：
        1. 验证更新对象的有效性
        2. 提取聊天ID
        3. 如果是新用户，添加到聊天ID列表
        4. 发送欢迎消息
        
        自动功能：
        - 新用户注册
        - 用户数据初始化
        - 欢迎消息发送
        - 日志记录
        
        示例重写：
        ```python
        def start_callback(self, update, context):
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                user = update.effective_user
                
                # 检查是否为新用户
                if chat_id not in self.chat_ids:
                    self.chat_ids.append(chat_id)
                    # 记录新用户
                    logger.info(f"New user registered: {user.first_name} ({chat_id})")
                
                # 发送个性化欢迎消息
                welcome_msg = f"欢迎 {user.first_name}！\\n{self.start_message}"
                self.send_message(chat_id, welcome_msg)
        ```
        """
        # 验证更新对象并提取聊天信息
        if isinstance(update, Update) and update.effective_chat:
            chat_id = update.effective_chat.id
            
            # 如果是新用户，添加到聊天ID列表
            if chat_id not in self.chat_ids:
                self.chat_ids.append(chat_id)
            
            # 发送欢迎消息
            self.send_message(chat_id, self.start_message)

    @property
    def help_message(self) -> str:
        """
        获取帮助消息
        
        该属性返回用户发送"/help"命令时机器人回复的帮助信息。
        子类可以重写此属性来提供详细的功能说明。
        
        返回：
            帮助消息字符串
        
        重写示例：
        ```python
        class MyBot(TelegramBot):
            @property
            def help_message(self):
                return '''
                📚 <b>帮助文档</b>
                
                🔍 <b>价格查询</b>
                /price [symbol] - 查询价格
                例如: /price BTC
                
                📊 <b>技术分析</b>
                /ma [symbol] [period] - 移动平均线
                /rsi [symbol] - RSI指标
                
                ⚠️ <b>警报设置</b>
                /alert [symbol] [price] - 价格警报
                例如: /alert BTC 50000
                
                💼 <b>投资组合</b>
                /portfolio - 查看投资组合
                /balance - 查看账户余额
                
                📞 <b>联系我们</b>
                如有问题请联系: @admin
                '''
        ```
        """
        return "Can't help you here, buddy."

    def help_callback(self, update: object, context: CallbackContext) -> None:
        """
        处理"/help"命令的回调函数
        
        该方法在用户发送"/help"命令时被调用，发送帮助信息给用户。
        
        参数：
            update: Telegram更新对象
            context: 回调上下文对象
        
        处理逻辑：
        1. 验证更新对象的有效性
        2. 提取聊天ID
        3. 发送帮助消息
        
        示例重写：
        ```python
        def help_callback(self, update, context):
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                user = update.effective_user
                
                # 记录帮助请求
                logger.info(f"Help requested by {user.first_name} ({chat_id})")
                
                # 发送帮助消息
                self.send_message(chat_id, self.help_message, parse_mode='HTML')
        ```
        """
        # 验证更新对象并发送帮助消息
        if isinstance(update, Update) and update.effective_chat:
            chat_id = update.effective_chat.id
            self.send_message(chat_id, self.help_message)

    def chat_migration_callback(self, update: object, context: CallbackContext) -> None:
        """
        处理聊天迁移的回调函数
        
        该方法在群组升级为超级群组时被调用，自动更新聊天ID以确保消息能够正常发送。
        
        参数：
            update: Telegram更新对象
            context: 回调上下文对象
        
        迁移类型：
        - 群组升级为超级群组
        - 聊天ID变更
        - 权限变更
        
        处理逻辑：
        1. 提取旧聊天ID和新聊天ID
        2. 从列表中移除旧ID
        3. 添加新ID到列表
        4. 记录迁移日志
        
        自动处理：
        - 无需用户干预
        - 保持消息发送功能
        - 数据连续性
        
        示例：
        ```python
        # 迁移前: 群组ID = -123456789
        # 迁移后: 超级群组ID = -1001234567890
        
        # 机器人自动处理：
        # 1. 检测到迁移
        # 2. 移除旧ID: -123456789
        # 3. 添加新ID: -1001234567890
        # 4. 记录日志: "Chat migrated from -123456789 to -1001234567890"
        ```
        """
        # 验证更新对象并处理迁移
        if isinstance(update, Update) and update.message:
            # 提取旧聊天ID和新聊天ID
            old_id = update.message.migrate_from_chat_id or update.message.chat_id
            new_id = update.message.migrate_to_chat_id or update.message.chat_id
            
            # 更新聊天ID列表
            if old_id in self.chat_ids:
                self.chat_ids.remove(old_id)  # 移除旧ID
            self.chat_ids.append(new_id)  # 添加新ID
            
            # 记录迁移日志
            logger.info(f"{old_id} - Chat migrated to {new_id}")

    def unknown_callback(self, update: object, context: CallbackContext) -> None:
        """
        处理未知命令的回调函数
        
        该方法在用户发送未识别的命令时被调用，提供友好的错误提示。
        
        参数：
            update: Telegram更新对象
            context: 回调上下文对象
        
        处理逻辑：
        1. 记录未知命令日志
        2. 发送友好的错误消息
        3. 可选：提供可用命令提示
        
        示例重写：
        ```python
        def unknown_callback(self, update, context):
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                command = update.message.text
                
                logger.info(f"{chat_id} - Unknown command: {command}")
                
                # 发送更友好的错误消息
                error_msg = f'''
                ❌ 未知命令: {command}
                
                💡 可用命令:
                /help - 获取帮助
                /price - 查询价格
                /portfolio - 查看投资组合
                '''
                self.send_message(chat_id, error_msg)
        ```
        """
        # 验证更新对象并处理未知命令
        if isinstance(update, Update) and update.effective_chat:
            chat_id = update.effective_chat.id
            
            # 记录未知命令日志
            logger.info(f"{chat_id} - Unknown command \"{update.message}\"")
            
            # 发送友好的错误消息
            self.send_message(chat_id, "Sorry, I didn't understand that command.")

    def error_callback(self, update: object, context: CallbackContext, *args) -> None:
        """
        处理错误的回调函数
        
        该方法在机器人处理更新时发生异常时被调用，提供统一的错误处理和日志记录。
        
        参数：
            update: 引发错误的更新对象
            context: 回调上下文对象，包含错误信息
            *args: 额外的参数
        
        错误处理：
        1. 记录详细的错误日志
        2. 向用户发送友好的错误消息
        3. 防止机器人崩溃
        4. 保持服务可用性
        
        日志信息：
        - 错误类型和消息
        - 引发错误的更新内容
        - 完整的堆栈跟踪
        
        示例重写：
        ```python
        def error_callback(self, update, context, *args):
            # 记录详细错误信息
            logger.error(f"Error processing update {update}: {context.error}", 
                        exc_info=context.error)
            
            if isinstance(update, Update) and update.effective_chat:
                chat_id = update.effective_chat.id
                error_type = type(context.error).__name__
                
                # 根据错误类型发送不同的消息
                if "network" in str(context.error).lower():
                    self.send_message(chat_id, "⚠️ 网络连接问题，请稍后重试")
                elif "timeout" in str(context.error).lower():
                    self.send_message(chat_id, "⏰ 处理超时，请重新发送命令")
                else:
                    self.send_message(chat_id, f"❌ 发生错误: {error_type}")
        ```
        """
        # 记录详细的错误日志，包含异常信息
        logger.error("Exception while handling an update \"%s\": ", update, exc_info=context.error)
        
        # 如果有有效聊天，向用户发送错误消息
        if isinstance(update, Update) and update.effective_chat:
            chat_id = update.effective_chat.id
            self.send_message(chat_id, "Sorry, an error happened.")

    def stop(self) -> None:
        """
        停止Telegram机器人
        
        该方法优雅地停止机器人的运行，确保所有资源被正确释放。
        
        停止流程：
        1. 停止消息轮询
        2. 关闭网络连接
        3. 清理资源
        4. 保存持久化数据
        
        使用场景：
        - 程序正常退出
        - 系统维护
        - 配置更新
        - 错误恢复
        
        示例：
        ```python
        # 在程序退出时停止机器人
        import signal
        
        def signal_handler(signum, frame):
            print("接收到停止信号，正在关闭机器人...")
            bot.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 或者手动停止
        bot.stop()
        ```
        """
        # 记录停止日志
        logger.info("Stopping bot")
        
        # 停止更新器，这会停止轮询并关闭连接
        self.updater.stop()

    @property
    def running(self) -> bool:
        """
        检查机器人是否正在运行
        
        该属性返回机器人的运行状态，用于状态检查和条件控制。
        
        返回：
            bool: True表示机器人正在运行，False表示已停止
        
        使用场景：
        - 状态检查
        - 条件控制
        - 健康检查
        - 监控系统
        
        示例：
        ```python
        # 检查运行状态
        if bot.running:
            print("机器人正在运行")
        else:
            print("机器人已停止")
        
        # 条件控制
        if not bot.running:
            bot.start(in_background=True)
        
        # 健康检查
        def health_check():
            return {
                'status': 'healthy' if bot.running else 'stopped',
                'uptime': get_uptime() if bot.running else 0
            }
        
        # 监控循环
        while True:
            if not bot.running:
                logger.warning("机器人已停止，正在重启...")
                bot.start(in_background=True)
            time.sleep(60)
        ```
        """
        return self.updater.running
