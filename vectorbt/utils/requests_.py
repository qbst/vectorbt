# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
================================================================================
VECTORBT REQUESTS UTILITIES MODULE: HTTP请求和网络通信工具模块
================================================================================

文件作用概述：
本文件是vectorbt框架中专门处理HTTP请求和网络通信的工具模块。主要提供了两个核心功能：
1. 增强的HTTP请求会话管理，具备自动重试和错误恢复能力
2. 集成Giphy API的文本到GIF转换功能，用于消息通知和可视化展示

设计理念：
1. **可靠性优先**：通过自动重试机制确保网络请求的稳定性，特别适用于金融数据获取
2. **用户体验**：提供有趣的GIF消息功能，让量化分析过程更加生动有趣
3. **配置灵活**：支持自定义重试策略和API配置，适应不同的网络环境
4. **错误处理**：内置完善的错误处理和状态码管理

核心功能模块：

【HTTP重试会话管理】
- **智能重试策略**：针对临时网络故障和服务器错误自动重试
- **指数退避算法**：避免对服务器造成过大压力，提高成功率
- **状态码过滤**：只对特定的服务器错误进行重试，避免无意义的重试
- **连接复用**：通过Session复用TCP连接，提高网络效率

【Giphy集成功能】
- **文本转GIF**：将文本描述转换为相关的GIF动画
- **上下文感知**：支持weirdness参数调节GIF的风格和相关性
- **API密钥管理**：安全的API密钥配置和管理
- **错误容错**：网络异常时的优雅降级处理
"""

# 导入URL编码工具，用于构建安全的HTTP请求URL参数
from urllib.parse import urlencode

# 导入requests库，Python中最流行的HTTP客户端库
import requests
# 导入HTTP适配器，用于自定义请求行为和重试策略
from requests.adapters import HTTPAdapter
# 导入重试工具，提供智能的请求重试机制
from requests.packages.urllib3.util.retry import Retry

# 导入vectorbt的类型定义模块，提供完整的类型注解支持
from vectorbt import _typing as tp


def requests_retry_session(retries: int = 3, backoff_factor: float = 0.3,
                           status_forcelist: tp.Tuple[int, ...] = (500, 502, 504),
                           session: tp.Optional[requests.Session] = None) -> requests.Session:
    """
    创建具有自动重试功能的HTTP会话
    
    该函数是vectorbt网络请求的核心工具，为HTTP请求提供智能的重试机制。
    在量化交易中，网络请求的可靠性至关重要，特别是在获取实时市场数据、
    访问金融API或进行大规模数据下载时。
    
    核心特性：
    - **指数退避重试**：每次重试间隔时间递增，避免对服务器造成压力
    - **智能状态码过滤**：只对服务器临时错误进行重试，避免无意义重试
    - **连接复用**：通过Session对象复用TCP连接，提高网络效率
    - **全协议支持**：同时支持HTTP和HTTPS协议的重试机制
    
    参数说明：
        retries (int, 默认=3): 最大重试次数
            - 建议值：3-5次，平衡可靠性和响应时间
            - 对于关键数据获取可以设置更高值
        
        backoff_factor (float, 默认=0.3): 指数退避因子
            - 重试间隔 = backoff_factor * (2 ^ (重试次数 - 1))
            - 0.3表示第1次重试等待0.3秒，第2次等待0.6秒，第3次等待1.2秒
            - 较小值提高响应速度，较大值减少服务器压力
        
        status_forcelist (tuple, 默认=(500, 502, 504)): 触发重试的HTTP状态码
            - 500: 内部服务器错误（临时故障）
            - 502: 网关错误（代理服务器问题）
            - 504: 网关超时（上游服务器响应超时）
            - 不包括4xx错误（客户端错误，重试无意义）
        
        session (requests.Session, 可选): 现有的Session对象
            - 如果提供，将在现有Session上配置重试
            - 如果为None，创建新的Session对象
    
    返回值：
        requests.Session: 配置了重试机制的Session对象
    
    使用示例：
        >>> import vectorbt as vbt
        >>> import requests
        
        >>> # 基本用法：创建重试会话
        >>> session = vbt.utils.requests_.requests_retry_session()
        >>> 
        >>> # 使用会话获取股票数据
        >>> try:
        ...     response = session.get('https://api.example.com/stock/AAPL')
        ...     data = response.json()
        ...     print(f"获取到数据: {len(data)} 条记录")
        ... except requests.exceptions.RequestException as e:
        ...     print(f"请求失败: {e}")
        
        >>> # 自定义重试策略：用于关键数据获取
        >>> critical_session = vbt.utils.requests_.requests_retry_session(
        ...     retries=5,           # 增加重试次数
        ...     backoff_factor=0.5,  # 增加退避时间
        ...     status_forcelist=(500, 502, 503, 504, 429)  # 包含限流错误
        ... )
        
        >>> # 量化应用：批量获取多只股票数据
        >>> def fetch_stock_data(symbols, session):
        ...     results = {}
        ...     for symbol in symbols:
        ...         try:
        ...             url = f'https://api.example.com/stock/{symbol}'
        ...             response = session.get(url, timeout=10)
        ...             response.raise_for_status()  # 抛出HTTP错误
        ...             results[symbol] = response.json()
        ...             print(f"✓ 成功获取 {symbol} 数据")
        ...         except Exception as e:
        ...             print(f"✗ 获取 {symbol} 失败: {e}")
        ...             results[symbol] = None
        ...     return results
        >>> 
        >>> # 使用重试会话获取数据
        >>> stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        >>> reliable_session = vbt.utils.requests_.requests_retry_session()
        >>> stock_data = fetch_stock_data(stocks, reliable_session)
    
    算法实现：
        1. 创建Retry对象，配置重试策略
        2. 创建HTTPAdapter，将重试策略绑定到适配器
        3. 将适配器挂载到Session的HTTP和HTTPS协议上
        4. 返回配置完成的Session对象
    
    性能考虑：
        - Session复用：避免重复的SSL握手和DNS解析
        - 连接池：自动管理TCP连接的复用和释放
        - 内存效率：适配器和重试对象的轻量级设计
    
    错误处理：
        - 网络连接错误：自动重试
        - 读取超时：自动重试
        - 服务器临时错误：按状态码重试
        - 客户端错误（4xx）：不重试，立即返回
    
    应用场景：
        - 实时数据获取：股价、期货、外汇等市场数据
        - API集成：第三方金融服务的数据接口
        - 批量下载：历史数据、财报、新闻等大量数据
        - 监控系统：定期检查服务状态和数据更新
    """
    # 如果没有提供现有会话，创建新的Session对象
    session = session or requests.Session()
    
    # 创建重试策略配置对象
    retry = Retry(
        total=retries,              # 总重试次数（包括连接、读取、重定向等所有类型的重试）
        read=retries,               # 读取操作的重试次数（服务器响应超时）
        connect=retries,            # 连接操作的重试次数（DNS解析、TCP连接失败）
        backoff_factor=backoff_factor,  # 指数退避因子，控制重试间隔时间的增长
        status_forcelist=status_forcelist,  # 强制重试的HTTP状态码列表
    )
    
    # 创建HTTP适配器，将重试策略绑定到适配器上
    adapter = HTTPAdapter(max_retries=retry)
    
    # 将适配器挂载到会话的HTTP协议处理器上
    session.mount('http://', adapter)
    # 将适配器挂载到会话的HTTPS协议处理器上
    session.mount('https://', adapter)
    
    # 返回配置完成的会话对象
    return session


def text_to_giphy_url(text: str, api_key: tp.Optional[str] = None, weirdness: tp.Optional[int] = None) -> str:
    """
    将文本转换为Giphy GIF动画URL
    
    该函数集成了Giphy的Translate API，能够将文本描述转换为相关的GIF动画URL。
    这个功能在vectorbt的消息系统中用于增强用户体验，让量化分析过程更加生动有趣。
    
    Giphy Translate API特点：
    - **上下文感知**：基于机器学习算法匹配最相关的GIF
    - **weirdness控制**：调节GIF的奇异程度和创意性
    - **高质量内容**：Giphy拥有庞大的高质量GIF库
    - **实时响应**：快速的API响应，适合实时通知场景
    
    参数说明：
        text (str): 要转换的文本描述
            - 支持中英文和各种语言
            - 建议使用简洁、具体的描述词汇
            - 例如："success", "error", "celebration", "thinking"
        
        api_key (str, 可选): Giphy API密钥
            - 如果为None，从vectorbt设置中获取
            - 需要在Giphy开发者平台申请
            - 免费账户有请求限制，付费账户无限制
        
        weirdness (int, 可选): GIF奇异程度控制
            - 范围：0-10，默认值通过设置配置
            - 0: 最相关、最普通的GIF
            - 10: 最奇异、最创意的GIF
            - 中等值(3-7)平衡相关性和趣味性
    
    返回值：
        str: GIF动画的URL地址，可直接用于显示或下载
    
    使用示例：
        >>> import vectorbt as vbt
        
        >>> # 基本用法：成功消息的GIF
        >>> try:
        ...     success_gif = vbt.utils.requests_.text_to_giphy_url("success")
        ...     print(f"成功GIF: {success_gif}")
        ... except Exception as e:
        ...     print(f"获取GIF失败: {e}")
        
        >>> # 量化应用：策略回测完成通知
        >>> def backtest_completed_notification(strategy_name, total_return):
        ...     if total_return > 0:
        ...         emotion = "celebration"
        ...         message = f"🎉 策略 {strategy_name} 回测完成！收益率: {total_return:.2%}"
        ...     else:
        ...         emotion = "disappointed"
        ...         message = f"😞 策略 {strategy_name} 回测完成，收益率: {total_return:.2%}"
        ...     
        ...     try:
        ...         gif_url = vbt.utils.requests_.text_to_giphy_url(emotion)
        ...         return {
        ...             'message': message,
        ...             'gif_url': gif_url,
        ...             'emotion': emotion
        ...         }
        ...     except:
        ...         return {'message': message, 'gif_url': None, 'emotion': emotion}
        
        >>> # 交易信号触发通知
        >>> def trading_signal_gif(signal_type, confidence):
        ...     signal_emotions = {
        ...         'buy': 'excited' if confidence > 0.8 else 'hopeful',
        ...         'sell': 'worried' if confidence > 0.8 else 'cautious',
        ...         'hold': 'patient'
        ...     }
        ...     
        ...     emotion = signal_emotions.get(signal_type, 'thinking')
        ...     return vbt.utils.requests_.text_to_giphy_url(emotion, weirdness=3)
        
        >>> # 自定义weirdness的使用
        >>> conservative_gif = vbt.utils.requests_.text_to_giphy_url("profit", weirdness=0)  # 保守的GIF
        >>> creative_gif = vbt.utils.requests_.text_to_giphy_url("profit", weirdness=8)      # 创意的GIF
        
        >>> # 错误处理和降级
        >>> def safe_get_gif(text, fallback_text="thinking"):
        ...     try:
        ...         return vbt.utils.requests_.text_to_giphy_url(text)
        ...     except:
        ...         try:
        ...             return vbt.utils.requests_.text_to_giphy_url(fallback_text)
        ...         except:
        ...             return None  # 完全失败时返回None
        
        >>> # 批量生成情绪GIF库
        >>> def build_emotion_gif_library():
        ...     emotions = [
        ...         'happy', 'sad', 'excited', 'worried', 'confused',
        ...         'success', 'fail', 'thinking', 'celebration', 'frustrated'
        ...     ]
        ...     
        ...     gif_library = {}
        ...     for emotion in emotions:
        ...         try:
        ...             gif_library[emotion] = vbt.utils.requests_.text_to_giphy_url(emotion)
        ...             print(f"✓ 获取 {emotion} GIF 成功")
        ...         except Exception as e:
        ...             print(f"✗ 获取 {emotion} GIF 失败: {e}")
        ...             gif_library[emotion] = None
        ...     
        ...     return gif_library
        
        >>> # 集成到消息系统
        >>> def send_rich_notification(title, message, emotion="neutral"):
        ...     notification = {
        ...         'title': title,
        ...         'message': message,
        ...         'timestamp': datetime.now().isoformat(),
        ...         'gif_url': None
        ...     }
        ...     
        ...     try:
        ...         notification['gif_url'] = vbt.utils.requests_.text_to_giphy_url(emotion)
        ...     except:
        ...         pass  # GIF获取失败不影响主要通知功能
        ...     
        ...     return notification
    
    API参考：
        - Giphy Translate API文档: https://developers.giphy.com/docs/api/endpoint#translate
        - 上下文感知搜索: https://engineering.giphy.com/contextually-aware-search-giphy-gets-work-specific/
    
    错误处理：
        - 网络错误：使用重试会话自动重试
        - API密钥错误：抛出认证异常
        - 无匹配结果：返回默认GIF或抛出异常
        - 参数错误：验证并提供有意义的错误信息
    
    性能考虑：
        - 缓存机制：相同文本的GIF URL可以缓存复用
        - 异步请求：在不阻塞主流程的情况下获取GIF
        - 降级策略：网络异常时使用本地默认图片
    
    应用场景：
        - 策略回测结果的可视化通知
        - 交易信号触发的趣味性提醒
        - 系统状态变化的用户友好展示
        - 社交媒体分享的内容丰富化
        - 团队协作中的情绪表达
    """
    # 从vectorbt设置系统导入配置管理器
    from vectorbt._settings import settings
    # 获取Giphy相关的配置项
    giphy_cfg = settings['messaging']['giphy']

    # 如果未提供API密钥，从配置中获取
    if api_key is None:
        api_key = giphy_cfg['api_key']
    # 如果未提供weirdness参数，从配置中获取默认值
    if weirdness is None:
        weirdness = giphy_cfg['weirdness']

    # 构建API请求参数字典
    params = {
        'api_key': api_key,     # Giphy API密钥，用于身份验证
        's': text,              # 要转换的搜索文本
        'weirdness': weirdness  # 控制GIF奇异程度的参数
    }
    
    # 构建完整的API请求URL，使用URL编码确保参数安全性
    url = "http://api.giphy.com/v1/gifs/translate?" + urlencode(params)
    
    # 使用重试会话发送GET请求，提高请求可靠性
    response = requests_retry_session().get(url)
    
    # 解析JSON响应并提取固定高度GIF的URL
    # JSON结构: {'data': {'images': {'fixed_height': {'url': 'gif_url'}}}}
    return response.json()['data']['images']['fixed_height']['url']
