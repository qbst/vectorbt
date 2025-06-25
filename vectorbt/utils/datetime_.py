# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
日期时间工具模块

该模块为vectorbt量化交易框架提供全面的日期时间处理功能，主要包括：

1. 时区管理：UTC和本地时区的获取与转换
2. 时间类型转换：在时区感知(timezone-aware)和朴素时间(naive time)之间转换
3. 日期时间解析：支持多种格式的日期时间字符串解析和标准化
4. 时间戳处理：毫秒级时间戳转换和时间间隔计算
5. 频率处理：时间序列频率到时间增量的转换
"""

import copy  # 导入copy模块，用于对象深拷贝操作
from datetime import datetime, timezone, timedelta, tzinfo, time  # 导入Python标准库的日期时间相关类

import dateparser  # 导入第三方日期解析库，支持自然语言日期解析
import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import pandas as pd  # 导入Pandas库，用于时间序列数据处理
import pytz  # 导入pytz库，提供世界时区数据库支持

from vectorbt import _typing as tp  # 导入vectorbt内部类型定义模块

# DatetimeIndex：表示具体的时间点，即某个确切的时间
# TimedeltaIndex：表示时间间隔，即两个时间点之间的差值
# PeriodIndex：表示时间段，有固定的开始和结束时间
DatetimeIndexes = (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)


def freq_to_timedelta(arg: tp.FrequencyLike) -> pd.Timedelta:
    """
    将频率规格转换为pandas时间增量对象
    
    该函数是`pd.to_timedelta`的增强版本，可以处理不带数字的单位缩写。
    当输入如'D'、'H'、'M'等单位缩写时，自动添加数字1。
    
    Args:
        arg: 频率规格，可以是字符串（如'1D'、'H'）或其他频率类型
        
    Returns:
        pd.Timedelta: 转换后的时间增量对象
        
    Examples:
        >>> import pandas as pd
        >>> from vectorbt.utils.datetime_ import freq_to_timedelta
        
        # 不带数字的单位缩写
        >>> freq_to_timedelta('D')
        Timedelta('1 days 00:00:00')
        
        >>> freq_to_timedelta('H')
        Timedelta('0 days 01:00:00')
        
        # 带数字的频率字符串
        >>> freq_to_timedelta('2H')
        Timedelta('0 days 02:00:00')
        
        >>> freq_to_timedelta('5min')
        Timedelta('0 days 00:05:00')
        
        # 其他pandas时间增量兼容格式
        >>> freq_to_timedelta('1.5H')
        Timedelta('0 days 01:30:00')
    """
    if isinstance(arg, str) and not arg[0].isdigit():  # 检查字符串是否以数字开头
        # 如果不是数字开头，添加数字1避免"ValueError: unit abbreviation w/o a number"错误
        return pd.Timedelta(1, unit=arg)  # 创建1个单位的时间增量
    return pd.Timedelta(arg)  # 直接转换为时间增量


def get_utc_tz() -> timezone:
    """
    获取UTC时区对象
    
    Returns:
        timezone: UTC时区对象
        
    Examples:
        >>> from vectorbt.utils.datetime_ import get_utc_tz
        >>> from datetime import datetime
        
        # 获取UTC时区
        >>> utc_tz = get_utc_tz()
        >>> print(utc_tz)
        datetime.timezone.utc
        
        # 创建UTC时间
        >>> utc_time = datetime.now(utc_tz)
        >>> print(utc_time)
        2024-01-15 10:30:00+00:00
        
        # 检查时区偏移量
        >>> print(utc_tz.utcoffset(datetime.now()))
        0:00:00
        
    Note:
        这是一个便捷函数，统一获取UTC时区，确保项目中时区处理的一致性
    """
    return timezone.utc  # 返回标准库中的UTC时区对象


def get_local_tz() -> timezone:
    """
    获取本地时区对象
    
    该函数通过获取当前UTC时间转换到本地时间的偏移量来确定本地时区。
    这种方法确保了跨平台的兼容性。
    
    Returns:
        timezone: 本地时区对象
        
    Examples:
        >>> from vectorbt.utils.datetime_ import get_local_tz
        >>> from datetime import datetime
        
        # 获取本地时区（假设在北京时间环境下）
        >>> local_tz = get_local_tz()
        >>> print(local_tz)
        datetime.timezone(datetime.timedelta(seconds=28800))
        
        # 创建本地时间
        >>> local_time = datetime.now(local_tz)
        >>> print(local_time)
        2024-01-15 18:30:00+08:00
        
        # 检查本地时区偏移量
        >>> print(local_tz.utcoffset(datetime.now()))
        8:00:00  # 北京时间UTC+8
        
        # 不同系统环境下的结果会不同
        # 在纽约: UTC-5 (冬季) 或 UTC-4 (夏季)
        # 在伦敦: UTC+0 (冬季) 或 UTC+1 (夏季)
        
    Note:
        返回的时区对象基于当前系统设置，考虑了夏令时等因素
    """
    # 获取当前UTC时间，转换为本地时间，提取偏移量，创建时区对象
    return timezone(datetime.now(timezone.utc).astimezone().utcoffset())


def convert_tzaware_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """
    转换时区感知的时间到指定时区
    
    该函数处理已包含时区信息的time对象，将其转换到目标时区。
    
    Args:
        t: 包含时区信息的time对象（tzinfo不为None）
        tz_out: 目标时区，可选参数
        
    Returns:
        time: 转换后的时区感知time对象
        
    Examples:
        >>> from vectorbt.utils.datetime_ import convert_tzaware_time, get_utc_tz, get_local_tz
        >>> from datetime import time, timezone, timedelta
        
        # 创建一个UTC时间
        >>> utc_time = time(14, 30, 0, tzinfo=get_utc_tz())  # 14:30:00 UTC
        >>> print(utc_time)
        14:30:00+00:00
        
        # 转换到北京时间 (UTC+8)
        >>> beijing_tz = timezone(timedelta(hours=8))
        >>> beijing_time = convert_tzaware_time(utc_time, beijing_tz)
        >>> print(beijing_time)
        22:30:00+08:00  # 14:30 UTC = 22:30 Beijing
        
        # 转换到纽约时间 (UTC-5)
        >>> ny_tz = timezone(timedelta(hours=-5))
        >>> ny_time = convert_tzaware_time(utc_time, ny_tz)
        >>> print(ny_time)
        09:30:00-05:00  # 14:30 UTC = 09:30 New York
        
        # 转换到本地时区
        >>> local_time = convert_tzaware_time(utc_time, get_local_tz())
        >>> print(local_time)
        # 输出取决于系统本地时区
        
    Note:
        输入的time对象必须包含tzinfo信息
    """
    # 将time对象与今天的日期组合，转换时区，然后提取带时区的时间部分
    return datetime.combine(datetime.today(), t).astimezone(tz_out).timetz()


def tzaware_to_naive_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """
    将时区感知的时间转换为朴素时间
    
    该函数将包含时区信息的time对象转换为不包含时区信息的朴素时间，
    但时间值会根据指定时区进行调整。
    
    Args:
        t: 包含时区信息的time对象（tzinfo不为None）
        tz_out: 目标时区，用于时间值转换
        
    Returns:
        time: 转换后的朴素time对象（不包含tzinfo）
        
    Examples:
        >>> from vectorbt.utils.datetime_ import tzaware_to_naive_time, get_utc_tz
        >>> from datetime import time, timezone, timedelta
        
        # 创建一个UTC时间
        >>> utc_time = time(14, 30, 0, tzinfo=get_utc_tz())  # 14:30:00 UTC
        >>> print(utc_time)
        14:30:00+00:00
        
        # 转换为北京时区的朴素时间
        >>> beijing_tz = timezone(timedelta(hours=8))
        >>> naive_beijing = tzaware_to_naive_time(utc_time, beijing_tz)
        >>> print(naive_beijing)
        22:30:00  # 没有时区信息，但时间值已调整
        >>> print(naive_beijing.tzinfo)
        None
        
        # 转换为纽约时区的朴素时间
        >>> ny_tz = timezone(timedelta(hours=-5))
        >>> naive_ny = tzaware_to_naive_time(utc_time, ny_tz)
        >>> print(naive_ny)
        09:30:00  # 没有时区信息
        
        # 应用场景：当你需要显示本地时间但不需要时区信息时
        >>> display_time = tzaware_to_naive_time(utc_time, get_local_tz())
        >>> print(f"当前时间: {display_time}")
        当前时间: 22:30:00  # 假设本地是北京时间
        
    Note:
        输入的time对象必须包含tzinfo信息
    """
    # 组合日期时间，转换时区，提取朴素时间部分
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time()


def naive_to_tzaware_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """
    将朴素时间转换为时区感知时间
    
    该函数将不包含时区信息的time对象转换为包含指定时区信息的时间。
    
    Args:
        t: 朴素time对象（tzinfo为None）
        tz_out: 目标时区信息
        
    Returns:
        time: 转换后的时区感知time对象
        
    Examples:
        >>> from vectorbt.utils.datetime_ import naive_to_tzaware_time, get_utc_tz
        >>> from datetime import time, timezone, timedelta
        
        # 创建一个朴素时间
        >>> naive_time = time(14, 30, 0)  # 14:30:00，无时区信息
        >>> print(naive_time)
        14:30:00
        >>> print(naive_time.tzinfo)
        None
        
        # 转换为UTC时区感知时间
        >>> utc_aware = naive_to_tzaware_time(naive_time, get_utc_tz())
        >>> print(utc_aware)
        14:30:00+00:00
        
        # 转换为北京时区感知时间
        >>> beijing_tz = timezone(timedelta(hours=8))
        >>> beijing_aware = naive_to_tzaware_time(naive_time, beijing_tz)
        >>> print(beijing_aware)
        14:30:00+08:00
        
        # 应用场景：处理用户输入的本地时间
        >>> user_input_time = time(9, 0, 0)  # 用户输入 09:00
        >>> market_open_utc = naive_to_tzaware_time(user_input_time, get_utc_tz())
        >>> print(f"市场开盘时间 (UTC): {market_open_utc}")
        市场开盘时间 (UTC): 09:00:00+00:00
        
    Note:
        输入的time对象不应包含tzinfo信息
    """
    # 组合日期时间，转换时区，提取时间部分，然后添加时区信息
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time().replace(tzinfo=tz_out)


def convert_naive_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """
    转换朴素时间到指定时区的朴素时间
    
    该函数处理不包含时区信息的time对象，根据指定时区调整时间值，
    但返回的仍是朴素时间对象。
    
    Args:
        t: 朴素time对象（tzinfo为None）
        tz_out: 用于时间转换的时区
        
    Returns:
        time: 调整后的朴素time对象
        
    Examples:
        >>> from vectorbt.utils.datetime_ import convert_naive_time, get_utc_tz
        >>> from datetime import time, timezone, timedelta
        
        # 创建朴素时间（假设是本地时间）
        >>> local_naive = time(14, 30, 0)  # 14:30:00 本地时间
        >>> print(local_naive)
        14:30:00
        
        # 转换为UTC对应的朴素时间（假设本地是北京时间UTC+8）
        >>> utc_naive = convert_naive_time(local_naive, get_utc_tz())
        >>> print(utc_naive)
        06:30:00  # 14:30 Beijing = 06:30 UTC，但没有时区信息
        
        # 从纽约时间转为伦敦时间（都是朴素时间）
        >>> ny_time = time(9, 0, 0)  # 09:00 纽约时间
        >>> london_tz = timezone(timedelta(hours=0))  # UTC+0 (伦敦冬季)
        >>> london_naive = convert_naive_time(ny_time, london_tz)
        >>> print(london_naive)
        14:00:00  # 09:00 NY = 14:00 London，但都是朴素时间
        
        # 应用场景：时区转换但保持朴素时间格式
        >>> trading_time = time(9, 30, 0)  # 交易开盘时间
        >>> utc_trading = convert_naive_time(trading_time, get_utc_tz())
        >>> print(f"UTC交易时间: {utc_trading}")
        UTC交易时间: 01:30:00  # 假设从北京时间转换
        
    Note:
        输入的time对象不应包含tzinfo信息
    """
    # 组合日期时间，按时区转换，提取朴素时间部分
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time()


def is_tz_aware(dt: tp.SupportsTZInfo) -> bool:
    """
    检查日期时间对象是否包含时区信息
    
    该函数判断给定的日期时间对象是否为时区感知的。
    
    Args:
        dt: 支持时区信息的日期时间对象
        
    Returns:
        bool: True表示包含有效时区信息，False表示为朴素时间
        
    Examples:
        >>> from vectorbt.utils.datetime_ import is_tz_aware, get_utc_tz
        >>> from datetime import datetime, time, timezone, timedelta
        >>> import pandas as pd
        
        # 测试朴素datetime
        >>> naive_dt = datetime(2024, 1, 15, 14, 30)
        >>> print(is_tz_aware(naive_dt))
        False
        
        # 测试时区感知datetime
        >>> aware_dt = datetime(2024, 1, 15, 14, 30, tzinfo=get_utc_tz())
        >>> print(is_tz_aware(aware_dt))
        True
        
        # 测试朴素time
        >>> naive_time = time(14, 30, 0)
        >>> print(is_tz_aware(naive_time))
        False
        
        # 测试时区感知time
        >>> aware_time = time(14, 30, 0, tzinfo=timezone(timedelta(hours=8)))
        >>> print(is_tz_aware(aware_time))
        True
        
        # 测试pandas Timestamp
        >>> naive_ts = pd.Timestamp('2024-01-15 14:30:00')
        >>> print(is_tz_aware(naive_ts))
        False
        
        >>> aware_ts = pd.Timestamp('2024-01-15 14:30:00', tz='UTC')
        >>> print(is_tz_aware(aware_ts))
        True
        
        # 应用场景：数据验证
        >>> def process_datetime(dt):
        ...     if not is_tz_aware(dt):
        ...         print("警告：输入的时间缺少时区信息")
        ...         return dt.replace(tzinfo=get_utc_tz())
        ...     return dt
        
    Note:
        仅仅有tzinfo属性还不够，还需要确保该时区能提供有效的UTC偏移量
    """
    tz = dt.tzinfo  # 获取时区信息
    if tz is None:  # 如果时区信息为空
        return False  # 返回False，表示为朴素时间
    # 检查时区是否能提供有效的UTC偏移量
    return tz.utcoffset(datetime.now()) is not None


def to_timezone(tz: tp.TimezoneLike, to_py_timezone: tp.Optional[bool] = None, **kwargs) -> tzinfo:
    """
    解析时区规格并返回标准时区对象
    
    该函数是时区处理的核心函数，支持多种时区输入格式：
    - 字符串：通过pytz和dateparser解析
    - 数字：作为小时偏移量处理
    - timedelta：作为偏移量处理
    - tzinfo对象：直接处理或转换
    
    Args:
        tz: 时区规格，支持多种类型
        to_py_timezone: 是否强制转换为datetime.timezone，默认从配置读取
        **kwargs: 传递给dateparser.parse的额外参数
        
    Returns:
        tzinfo: 标准化的时区对象
        
    Examples:
        >>> from vectorbt.utils.datetime_ import to_timezone
        >>> from datetime import timezone, timedelta
        
        # 字符串时区名称
        >>> tz1 = to_timezone('UTC')
        >>> print(tz1)
        <DstTzInfo 'UTC' LMT+0:00:00 STD>
        
        >>> tz2 = to_timezone('Asia/Shanghai')
        >>> print(tz2)
        <DstTzInfo 'Asia/Shanghai' LMT+8:05:00 STD>
        
        # 数字偏移量（小时）
        >>> tz3 = to_timezone(8)  # UTC+8
        >>> print(tz3)
        datetime.timezone(datetime.timedelta(seconds=28800))
        
        >>> tz4 = to_timezone(-5)  # UTC-5
        >>> print(tz4)
        datetime.timezone(datetime.timedelta(days=-1, seconds=68400))
        
        # timedelta对象
        >>> tz5 = to_timezone(timedelta(hours=9))  # UTC+9 (东京)
        >>> print(tz5)
        datetime.timezone(datetime.timedelta(seconds=32400))
        
        # 自然语言时区描述
        >>> tz6 = to_timezone('+0800')  # 北京时间
        >>> print(tz6)
        datetime.timezone(datetime.timedelta(seconds=28800))
        
        # None 返回本地时区
        >>> local_tz = to_timezone(None)
        >>> print(local_tz)
        # 根据系统本地时区返回相应对象
        
        # 强制转换为Python标准时区
        >>> py_tz = to_timezone('Asia/Shanghai', to_py_timezone=True)
        >>> print(type(py_tz))
        <class 'datetime.timezone'>
        
        # 应用场景：统一时区处理
        >>> user_inputs = ['UTC', 8, timedelta(hours=-5), 'Asia/Tokyo']
        >>> normalized_timezones = [to_timezone(tz) for tz in user_inputs]
        
    Raises:
        TypeError: 无法解析时区规格时抛出
        
    Note:
        该函数优先使用pytz解析字符串时区，失败时尝试dateparser
    """
    from vectorbt._settings import settings  # 导入vectorbt配置模块
    datetime_cfg = settings['datetime']  # 获取日期时间配置

    if tz is None:  # 如果时区为空
        return get_local_tz()  # 返回本地时区
    if to_py_timezone is None:  # 如果未指定转换选项
        to_py_timezone = datetime_cfg['to_py_timezone']  # 从配置中读取
    if isinstance(tz, str):  # 如果是字符串类型
        try:
            tz = pytz.timezone(tz)  # 尝试使用pytz解析时区名称
        except pytz.UnknownTimeZoneError:  # 如果pytz无法识别
            dt = dateparser.parse('now %s' % tz, **kwargs)  # 使用dateparser解析
            if dt is not None:  # 如果解析成功
                tz = dt.tzinfo  # 提取时区信息
    if isinstance(tz, (int, float)):  # 如果是数字类型
        tz = timezone(timedelta(hours=tz))  # 转换为小时偏移的时区对象
    if isinstance(tz, timedelta):  # 如果是时间增量类型
        tz = timezone(tz)  # 转换为时区对象
    if isinstance(tz, tzinfo):  # 如果已经是时区信息对象
        if to_py_timezone:  # 如果需要转换为Python标准时区
            return timezone(tz.utcoffset(datetime.now()))  # 创建标准时区对象
        return tz  # 直接返回时区对象
    raise TypeError("Couldn't parse the timezone")  # 无法解析时抛出类型错误


def to_tzaware_datetime(dt_like: tp.DatetimeLike,
                        naive_tz: tp.Optional[tp.TimezoneLike] = None,
                        tz: tp.Optional[tp.TimezoneLike] = None,
                        **kwargs) -> datetime:
    """
    解析各种格式的日期时间并返回时区感知的datetime对象
    
    该函数是日期时间处理的核心函数，支持多种输入格式：
    - 浮点数：作为时间戳处理（秒）
    - 整数：作为时间戳处理（自动识别秒/毫秒/微秒）
    - 字符串：通过dateparser解析
    - pd.Timestamp：转换为datetime
    - np.datetime64：转换为datetime
    - datetime：直接处理
    
    Args:
        dt_like: 类似日期时间的对象，支持多种类型
        naive_tz: 朴素时间的默认时区，默认从配置读取
        tz: 最终转换到的目标时区
        **kwargs: 传递给dateparser.parse的额外参数
        
    Returns:
        datetime: 时区感知的datetime对象
        
    Examples:
        >>> from vectorbt.utils.datetime_ import to_tzaware_datetime, get_utc_tz
        >>> import pandas as pd
        >>> import numpy as np
        
        # 时间戳转换（秒级）
        >>> timestamp = 1705334400  # 2024-01-15 14:00:00 UTC
        >>> dt1 = to_tzaware_datetime(timestamp)
        >>> print(dt1)
        2024-01-15 14:00:00+00:00
        
        # 毫秒级时间戳
        >>> ms_timestamp = 1705334400000
        >>> dt2 = to_tzaware_datetime(ms_timestamp)
        >>> print(dt2)
        2024-01-15 14:00:00+00:00
        
        # 字符串解析
        >>> dt3 = to_tzaware_datetime('2024-01-15 14:30:00')
        >>> print(dt3)
        2024-01-15 14:30:00+08:00  # 假设本地是北京时间
        
        >>> dt4 = to_tzaware_datetime('Jan 15, 2024 2:30 PM')
        >>> print(dt4)
        2024-01-15 14:30:00+08:00
        
        # pandas Timestamp
        >>> ts = pd.Timestamp('2024-01-15 14:30:00')
        >>> dt5 = to_tzaware_datetime(ts)
        >>> print(dt5)
        2024-01-15 14:30:00+08:00
        
        # numpy datetime64
        >>> np_dt = np.datetime64('2024-01-15T14:30:00')
        >>> dt6 = to_tzaware_datetime(np_dt)
        >>> print(dt6)
        2024-01-15 14:30:00+08:00
        
        # 指定朴素时间的时区
        >>> naive_str = '2024-01-15 14:30:00'
        >>> dt7 = to_tzaware_datetime(naive_str, naive_tz='UTC')
        >>> print(dt7)
        2024-01-15 14:30:00+00:00
        
        # 转换到指定时区
        >>> dt8 = to_tzaware_datetime('2024-01-15 14:30:00 UTC', tz='Asia/Shanghai')
        >>> print(dt8)
        2024-01-15 22:30:00+08:00
        
        # 应用场景：处理多种数据源的时间
        >>> data_sources = [
        ...     1705334400,  # API返回的时间戳
        ...     '2024-01-15 14:30:00',  # CSV文件中的字符串
        ...     pd.Timestamp.now(),  # pandas处理的时间
        ... ]
        >>> unified_times = [to_tzaware_datetime(dt) for dt in data_sources]
        
    Raises:
        ValueError: 无法解析日期时间时抛出
        
    Note:
        - 原始时间戳会被本地化为UTC
        - 朴素datetime会被本地化为naive_tz
        - 最终可通过tz参数转换到指定时区
    """
    from vectorbt._settings import settings  # 导入配置模块
    datetime_cfg = settings['datetime']  # 获取日期时间配置

    if naive_tz is None:  # 如果未指定朴素时区
        naive_tz = datetime_cfg['naive_tz']  # 从配置中读取默认值
    if isinstance(dt_like, float):  # 如果是浮点数时间戳
        dt = datetime.fromtimestamp(dt_like, timezone.utc)  # 从UTC时间戳创建datetime
    elif isinstance(dt_like, int):  # 如果是整数时间戳
        if len(str(dt_like)) > 10:  # 如果数字长度大于10位（毫秒或微秒级）
            # 自动缩放到秒级时间戳（除以10的适当次幂）
            dt = datetime.fromtimestamp(dt_like / 10 ** (len(str(dt_like)) - 10), timezone.utc)
        else:  # 如果是标准10位秒级时间戳
            dt = datetime.fromtimestamp(dt_like, timezone.utc)  # 直接创建UTC datetime
    elif isinstance(dt_like, str):  # 如果是字符串格式
        dt = dateparser.parse(dt_like, **kwargs)  # 使用dateparser解析字符串
    elif isinstance(dt_like, pd.Timestamp):  # 如果是pandas时间戳
        dt = dt_like.to_pydatetime()  # 转换为Python datetime对象
    elif isinstance(dt_like, np.datetime64):  # 如果是numpy datetime64
        dt = datetime.combine(dt_like.astype(datetime), time())  # 转换为datetime对象
    else:  # 其他情况
        dt = dt_like  # 直接使用输入值

    if dt is None:  # 如果解析失败
        raise ValueError("Couldn't parse the datetime")  # 抛出值错误

    if not is_tz_aware(dt):  # 如果是朴素时间
        dt = dt.replace(tzinfo=to_timezone(naive_tz))  # 添加朴素时区信息
    else:  # 如果已有时区信息
        dt = dt.replace(tzinfo=to_timezone(dt.tzinfo))  # 标准化时区对象
    if tz is not None:  # 如果指定了目标时区
        dt = dt.astimezone(to_timezone(tz))  # 转换到目标时区
    return dt  # 返回处理后的datetime对象


def datetime_to_ms(dt: datetime) -> int:
    """
    将datetime对象转换为毫秒级时间戳
    
    该函数将时区感知的datetime对象转换为自epoch以来的毫秒数。
    主要用于高频交易数据处理和API接口。
    
    Args:
        dt: 时区感知的datetime对象
        
    Returns:
        int: 毫秒级时间戳
        
    Examples:
        >>> from vectorbt.utils.datetime_ import datetime_to_ms, get_utc_tz
        >>> from datetime import datetime, timezone, timedelta
        
        # UTC时间转毫秒时间戳
        >>> utc_dt = datetime(2024, 1, 15, 14, 30, 0, tzinfo=get_utc_tz())
        >>> ms_timestamp = datetime_to_ms(utc_dt)
        >>> print(ms_timestamp)
        1705334200000  # 毫秒时间戳
        
        # 本地时间转毫秒时间戳
        >>> beijing_tz = timezone(timedelta(hours=8))
        >>> beijing_dt = datetime(2024, 1, 15, 22, 30, 0, tzinfo=beijing_tz)
        >>> ms_timestamp = datetime_to_ms(beijing_dt)
        >>> print(ms_timestamp)
        1705334200000  # 相同的毫秒时间戳（等价的UTC时间）
        
        # 当前时间转毫秒时间戳
        >>> now_utc = datetime.now(get_utc_tz())
        >>> current_ms = datetime_to_ms(now_utc)
        >>> print(f"当前毫秒时间戳: {current_ms}")
        
        # 应用场景：API接口调用
        >>> # 币安API需要毫秒时间戳
        >>> api_params = {
        ...     'symbol': 'BTCUSDT',
        ...     'startTime': datetime_to_ms(start_time),
        ...     'endTime': datetime_to_ms(end_time),
        ... }
        
        # 验证转换结果
        >>> original_dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=get_utc_tz())
        >>> ms = datetime_to_ms(original_dt)
        >>> back_to_dt = datetime.fromtimestamp(ms/1000, get_utc_tz())
        >>> print(f"原始: {original_dt}")
        >>> print(f"毫秒: {ms}")
        >>> print(f"还原: {back_to_dt}")
        
    Note:
        返回的时间戳考虑了时区信息，确保时间一致性
    """
    epoch = datetime.fromtimestamp(0, dt.tzinfo)  # 创建同一时区的epoch时间点
    return int((dt - epoch).total_seconds() * 1000.0)  # 计算时间差并转换为毫秒


def interval_to_ms(interval: str) -> tp.Optional[int]:
    """
    将时间间隔字符串转换为毫秒数
    
    该函数解析常见的时间间隔字符串格式（如'1m', '5h', '1d', '2w'），
    并转换为对应的毫秒数。主要用于API参数处理和数据聚合。
    
    Args:
        interval: 时间间隔字符串，格式为数字+单位（m/h/d/w）
        
    Returns:
        int | None: 对应的毫秒数，解析失败时返回None
        
    Examples:
        >>> from vectorbt.utils.datetime_ import interval_to_ms
        
        # 分钟
        >>> print(interval_to_ms('1m'))
        60000  # 1分钟 = 60秒 = 60,000毫秒
        
        >>> print(interval_to_ms('5m'))
        300000  # 5分钟 = 300秒 = 300,000毫秒
        
        # 小时
        >>> print(interval_to_ms('1h'))
        3600000  # 1小时 = 3600秒 = 3,600,000毫秒
        
        >>> print(interval_to_ms('4h'))
        14400000  # 4小时 = 14,400,000毫秒
        
        # 天
        >>> print(interval_to_ms('1d'))
        86400000  # 1天 = 86,400秒 = 86,400,000毫秒
        
        >>> print(interval_to_ms('7d'))
        604800000  # 7天 = 604,800,000毫秒
        
        # 周
        >>> print(interval_to_ms('1w'))
        604800000  # 1周 = 7天 = 604,800,000毫秒
        
        >>> print(interval_to_ms('2w'))
        1209600000  # 2周 = 1,209,600,000毫秒
        
        # 无效输入
        >>> print(interval_to_ms('invalid'))
        None
        
        >>> print(interval_to_ms('1x'))  # 不支持的单位
        None
        
        >>> print(interval_to_ms('m'))  # 缺少数字
        None
        
        # 应用场景：K线数据聚合
        >>> intervals = ['1m', '5m', '15m', '1h', '4h', '1d']
        >>> ms_intervals = {iv: interval_to_ms(iv) for iv in intervals}
        >>> print(ms_intervals)
        {'1m': 60000, '5m': 300000, '15m': 900000, 
         '1h': 3600000, '4h': 14400000, '1d': 86400000}
        
        # API参数设置
        >>> def get_kline_data(symbol, interval):
        ...     interval_ms = interval_to_ms(interval)
        ...     if interval_ms is None:
        ...         raise ValueError(f"不支持的时间间隔: {interval}")
        ...     # 调用API获取数据
        ...     pass
        
        # 时间范围计算
        >>> start_time_ms = 1705334400000  # 某个起始时间
        >>> interval_ms = interval_to_ms('1h')
        >>> end_time_ms = start_time_ms + interval_ms
        >>> print(f"1小时后: {end_time_ms}")
        
    Note:
        支持的单位：m(分钟)、h(小时)、d(天)、w(周)
    """
    # 定义各时间单位对应的秒数映射表
    seconds_per_unit = {
        "m": 60,  # 分钟：60秒
        "h": 60 * 60,  # 小时：3600秒
        "d": 24 * 60 * 60,  # 天：86400秒
        "w": 7 * 24 * 60 * 60,  # 周：604800秒
    }
    try:
        # 解析数字部分（除最后一个字符）和单位部分（最后一个字符）
        # 计算：数字 * 单位秒数 * 1000（转换为毫秒）
        return int(interval[:-1]) * seconds_per_unit[interval[-1]] * 1000
    except (ValueError, KeyError):  # 如果解析失败或单位不支持
        return None  # 返回None表示解析失败
