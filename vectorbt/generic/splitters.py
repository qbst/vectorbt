# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""
数据分割器模块 - 用于金融时间序列交叉验证和数据分割

1. RangeSplitter - 范围分割器：将数据分成指定数量或长度的连续区间
2. RollingSplitter - 滚动分割器：使用滑动窗口进行前向验证分割
3. ExpandingSplitter - 扩展分割器：使用递增窗口进行扩展验证分割

与scikit-learn兼容，可以与vectorbt的其他模块无缝集成。
"""

import math

import numpy as np
import pandas as pd

from vectorbt import _typing as tp
from vectorbt.base.index_fns import find_first_occurrence
from vectorbt.base.reshape_fns import to_any_array
from vectorbt.utils import checks

# 定义范围生成器类型别名，用于生成数据分割的索引范围
RangesT = tp.Generator[tp.Sequence[tp.ArrayLikeSequence], None, None]


def split_ranges_into_sets(start_idxs: tp.ArrayLike,
                           end_idxs: tp.ArrayLike,
                           set_lens: tp.MaybeSequence[tp.Sequence[float]] = (),
                           left_to_right: tp.MaybeSequence[bool] = True) -> RangesT:
    """
    将索引范围分割成多个数据集的核心函数
    
    根据起始和结束索引生成范围，并可选择性地将每个范围分割为一个或多个数据集。
    这是所有分割器的基础函数，用于实现训练集、验证集和测试集的分割。
    
    参数说明：
        start_idxs (array_like): 起始索引数组，每个元素表示一个分割范围的开始位置
        end_idxs (array_like): 结束索引数组，每个元素表示一个分割范围的结束位置（包含该位置）
        set_lens (list of float): 每个范围内数据集的长度列表
            - 返回的数据集数量 = len(set_lens) + 1，最后一个数据集包含剩余元素
            - 可以是绝对数量（>= 1）或相对比例（0-1之间）
            - 可以为每个范围单独指定，也可以全局指定
        left_to_right (bool or list of bool): 是否从左到右解析set_lens
            - True: 最后一个数据集长度可变（常用于测试集）
            - False: 第一个数据集长度可变（常用于训练集）
            - 可以为每个范围单独指定
    
    返回值：
        RangesT: 生成器，每次迭代返回一个元组，包含该范围内所有数据集的索引数组
    """
    # 将输入转换为NumPy数组以便处理
    start_idxs = np.asarray(start_idxs)
    end_idxs = np.asarray(end_idxs)
    # 检查起始索引和结束索引数组长度是否一致
    checks.assert_len_equal(start_idxs, end_idxs)

    # 遍历每个范围进行分割
    for i in range(len(start_idxs)):
        start_idx = start_idxs[i]  # 当前范围的起始索引
        end_idx = end_idxs[i]      # 当前范围的结束索引

        range_len = end_idx - start_idx + 1  # 计算当前范围的总长度（包含结束索引）
        new_set_lens = []  # 存储计算后的实际数据集长度
        
        # 如果没有指定分割长度，直接返回整个范围
        if len(set_lens) == 0:
            yield (np.arange(start_idx, end_idx + 1),)
        else:
            # 解析当前范围的分割长度配置
            if checks.is_sequence(set_lens[0]):
                # 如果set_lens是二维序列，每个范围有独立的配置
                _set_lens = set_lens[i]
            else:
                # 如果set_lens是一维序列，所有范围使用相同配置
                _set_lens = set_lens
            
            # 解析当前范围的分割方向配置
            if checks.is_sequence(left_to_right):
                # 如果left_to_right是序列，每个范围有独立的配置
                _left_to_right = left_to_right[i]
            else:
                # 如果left_to_right是布尔值，所有范围使用相同配置
                _left_to_right = left_to_right
            
            # 计算每个数据集的实际长度
            for j, set_len in enumerate(_set_lens):
                # 如果是比例（0-1之间），转换为绝对数量
                if 0 < set_len < 1:
                    set_len = math.floor(set_len * range_len)
                
                # 检查数据集长度是否有效
                if set_len == 0:
                    raise ValueError(f"Set {j} in the range {i} is empty")
                
                new_set_lens.append(set_len)
            
            # 处理剩余数据，分配给可变长度的数据集
            if sum(new_set_lens) < range_len:
                # 计算剩余长度
                remaining_len = range_len - sum(new_set_lens)
                
                if _left_to_right:
                    # 从左到右：剩余数据分配给最后一个数据集（通常是测试集）
                    new_set_lens = new_set_lens + [remaining_len]
                else:
                    # 从右到左：剩余数据分配给第一个数据集（通常是训练集）
                    new_set_lens = [remaining_len] + new_set_lens
            else:
                # 如果指定的长度总和大于等于范围长度，抛出错误
                raise ValueError(f"Range of length {range_len} too short to split into {len(_set_lens) + 1} sets")

            # 将每个范围分割为多个数据集
            idx_offset = 0  # 当前偏移量
            set_ranges = []  # 存储各个数据集的索引范围
            
            for set_len in new_set_lens:
                # 计算当前数据集的结束偏移量
                new_idx_offset = idx_offset + set_len
                # 生成当前数据集的索引范围
                set_ranges.append(np.arange(start_idx + idx_offset, start_idx + new_idx_offset))
                # 更新偏移量
                idx_offset = new_idx_offset

            # 生成当前范围的所有数据集索引
            yield tuple(set_ranges)


class SplitterT(tp.Protocol):
    """
    分割器协议类型定义
    
    定义了所有分割器必须实现的接口，确保类型安全和一致性。
    任何实现了split方法的类都可以作为分割器使用。
    """
    def split(self, X: tp.ArrayLike, **kwargs) -> RangesT:
        """
        分割方法接口
        
        参数：
            X: 要分割的数据，通常是时间序列数据
            **kwargs: 额外的分割参数
        
        返回：
            RangesT: 分割结果的生成器
        """
        ...


class BaseSplitter:
    """
    抽象分割器基类
    
    所有具体分割器的基类，定义了分割器的基本接口。
    提供了统一的分割方法签名，确保所有子类都实现split方法。
    """

    def split(self, X: tp.ArrayLike, **kwargs) -> RangesT:
        """
        抽象分割方法
        
        子类必须实现这个方法来提供具体的分割逻辑。
        
        参数：
            X: 要分割的数据
            **kwargs: 额外参数
        
        返回：
            RangesT: 分割结果生成器
        
        抛出：
            NotImplementedError: 子类未实现该方法时抛出
        """
        raise NotImplementedError


class RangeSplitter(BaseSplitter):
    """
    范围分割器
    
    将时间序列数据分割成指定数量或长度的连续区间。适用于以下场景：
    - 简单的训练/测试分割
    - 基于时间段的数据划分
    - 固定窗口大小的数据分割
    
    特点：
    - 支持按数量分割（n个区间）
    - 支持按长度分割（每个区间固定长度）
    - 支持自定义起始和结束索引
    - 支持多种分割配置组合
    """

    def split(self,
              X: tp.ArrayLike,
              n: tp.Optional[int] = None,
              range_len: tp.Optional[float] = None,
              min_len: int = 1,
              start_idxs: tp.Optional[tp.ArrayLike] = None,
              end_idxs: tp.Optional[tp.ArrayLike] = None, **kwargs) -> RangesT:
        """
        执行范围分割
        
        根据指定参数将数据分割成多个连续区间。可以通过三种方式指定分割：
        1. 指定区间数量（n）和/或区间长度（range_len）
        2. 直接指定起始和结束索引数组
        3. 组合使用以上参数
        
        参数说明：
            X (array-like): 要分割的数据，可以是pandas Series/DataFrame或numpy数组
            n (int, optional): 分割的区间数量
                - 如果range_len未指定，数据将被均匀分成n个区间
                - 如果同时指定，将从可能的区间中均匀选择n个
            range_len (float, optional): 每个区间的长度
                - 如果是0-1之间的小数，表示占总长度的比例
                - 如果是>=1的整数，表示绝对长度
                - 如果n未指定，将生成尽可能多的区间
            min_len (int): 区间的最小长度，短于此长度的区间将被过滤
            start_idxs (array-like, optional): 自定义起始索引数组
                - 可以是numpy数组（绝对位置）或pandas Index（标签）
            end_idxs (array-like, optional): 自定义结束索引数组
                - 可以是numpy数组（绝对位置）或pandas Index（标签）
                - 结束索引是包含的（inclusive）
            **kwargs: 传递给split_ranges_into_sets的额外参数
        
        返回值：
            RangesT: 分割结果生成器，每次迭代返回该分割的所有数据集索引
        
        使用示例：
            >>> import pandas as pd
            >>> import numpy as np
            >>> from vectorbt.generic.splitters import RangeSplitter
            
            >>> # 创建示例数据
            >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
            >>> prices = pd.Series(np.random.randn(100).cumsum(), index=dates)
            
            >>> splitter = RangeSplitter()
            
            >>> # 示例1：分成3个等长区间
            >>> for ranges in splitter.split(prices, n=3):
            ...     print(f"区间包含 {len(ranges)} 个数据集")
            ...     for i, range_idx in enumerate(ranges):
            ...         print(f"  数据集{i}: {len(range_idx)} 个样本")
            
            >>> # 示例2：每个区间30天，尽可能多的区间
            >>> for ranges in splitter.split(prices, range_len=30):
            ...     print(f"区间长度: {len(ranges[0])} 天")
            
            >>> # 示例3：每个区间占30%的数据
            >>> for ranges in splitter.split(prices, range_len=0.3):
            ...     print(f"区间长度: {len(ranges[0])} 天")
            
            >>> # 示例4：自定义起始和结束日期
            >>> start_dates = pd.Index(['2020-01-01', '2020-02-01', '2020-03-01'])
            >>> end_dates = pd.Index(['2020-01-31', '2020-02-29', '2020-03-31'])
            >>> for ranges in splitter.split(prices, start_idxs=start_dates, end_idxs=end_dates):
            ...     print(f"自定义区间: {len(ranges[0])} 天")
            
            >>> # 示例5：分割成训练集和测试集（70%训练，30%测试）
            >>> for train_idx, test_idx in splitter.split(prices, n=1, set_lens=(0.7,)):
            ...     print(f"训练集: {len(train_idx)} 样本, 测试集: {len(test_idx)} 样本")
        
        异常处理：
            ValueError: 当参数不合法时抛出，例如：
                - n、range_len、start_idxs和end_idxs都未指定
                - start_idxs和end_idxs只指定了一个
                - n大于可能的区间数量
                - 没有区间满足min_len要求
        """
        # 将输入数据转换为标准格式
        X = to_any_array(X)
        
        # 提取或创建索引
        if isinstance(X, (pd.Series, pd.DataFrame)):
            index = X.index  # 使用原始索引
        else:
            index = pd.Index(np.arange(X.shape[0]))  # 创建数字索引

        # 解析起始和结束索引
        if start_idxs is None and end_idxs is None:
            # 情况1：根据n和range_len自动生成索引
            if range_len is None and n is None:
                raise ValueError("At least n, range_len, or start_idxs and end_idxs must be set")
            
            # 计算区间长度
            if range_len is None:
                range_len = len(index) // n  # 根据区间数量计算长度
            
            # 处理比例形式的长度
            if 0 < range_len < 1:
                range_len = math.floor(range_len * len(index))
            
            # 生成所有可能的起始和结束索引
            start_idxs = np.arange(len(index) - range_len + 1)
            end_idxs = np.arange(range_len - 1, len(index))
            
        elif start_idxs is None or end_idxs is None:
            # 情况2：起始和结束索引必须同时指定
            raise ValueError("Both start_idxs and end_idxs must be set")
        else:
            # 情况3：使用自定义的起始和结束索引
            if isinstance(start_idxs, pd.Index):
                # 如果是pandas Index（标签），转换为位置索引
                start_idxs = np.asarray([find_first_occurrence(idx, index) for idx in start_idxs])
            else:
                start_idxs = np.asarray(start_idxs)
            
            if isinstance(end_idxs, pd.Index):
                # 如果是pandas Index（标签），转换为位置索引
                end_idxs = np.asarray([find_first_occurrence(idx, index) for idx in end_idxs])
            else:
                end_idxs = np.asarray(end_idxs)

        # 过滤掉过短的区间
        start_idxs, end_idxs = np.broadcast_arrays(start_idxs, end_idxs)  # 确保数组形状一致
        range_lens = end_idxs - start_idxs + 1  # 计算每个区间的长度
        min_len_mask = range_lens >= min_len    # 创建长度过滤掩码
        
        if not np.any(min_len_mask):
            raise ValueError(f"There are no ranges that meet range_len>={min_len}")
        
        # 应用过滤掩码
        start_idxs = start_idxs[min_len_mask]
        end_idxs = end_idxs[min_len_mask]

        # 如果指定了n，均匀选择n个区间
        if n is not None:
            if n > len(start_idxs):
                raise ValueError(f"n cannot be bigger than the maximum number of ranges {len(start_idxs)}")
            
            # 使用线性空间均匀选择索引
            idxs = np.round(np.linspace(0, len(start_idxs) - 1, n)).astype(int)
            start_idxs = start_idxs[idxs]
            end_idxs = end_idxs[idxs]

        # 调用核心分割函数
        return split_ranges_into_sets(start_idxs, end_idxs, **kwargs)


class RollingSplitter(BaseSplitter):
    """
    滚动分割器（前向验证分割器）
    
    使用滑动窗口技术进行时间序列数据分割，实现前向验证（walk-forward validation）。
    这是金融时间序列分析中最重要的验证方法之一，确保模型训练时不会使用未来数据。
    
    核心概念：
    - 滑动窗口：固定大小的时间窗口在时间序列上滑动
    - 前向验证：每个窗口只使用历史数据进行训练，用未来数据进行测试
    - 时间一致性：严格保持时间顺序，避免前瞻偏差
    
    应用场景：
    - 量化交易策略的回测验证
    - 时间序列预测模型的性能评估
    - 风险模型的稳健性测试
    - 机器学习模型的时间序列交叉验证
    
    优势：
    - 模拟真实交易环境
    - 提供多个独立的验证结果
    - 可以观察模型性能的时间变化
    - 更好地评估模型的泛化能力
    """

    def split(self,
              X: tp.ArrayLike,
              n: tp.Optional[int] = None,
              window_len: tp.Optional[float] = None,
              min_len: int = 1,
              **kwargs) -> RangesT:
        """
        执行滚动窗口分割
        
        使用固定大小的滑动窗口对时间序列数据进行分割。每个窗口生成一个训练/测试组合，
        所有窗口按时间顺序排列，确保不会产生前瞻偏差。
        
        参数说明：
            X (array-like): 要分割的时间序列数据
            n (int, optional): 要生成的窗口数量
                - 如果指定，将从所有可能的窗口中均匀选择n个
                - 如果未指定，将生成所有可能的窗口
            window_len (float, optional): 窗口长度
                - 如果是0-1之间的小数，表示占总长度的比例
                - 如果是>=1的整数，表示绝对长度
                - 如果未指定，将根据n计算窗口长度
            min_len (int): 窗口的最小长度，短于此长度的窗口将被过滤
            **kwargs: 传递给split_ranges_into_sets的额外参数，如：
                - set_lens: 指定训练集、验证集、测试集的长度比例
                - left_to_right: 指定分割方向
        
        返回值：
            RangesT: 分割结果生成器，每次迭代返回该窗口的所有数据集索引
        
        使用示例：
            >>> import pandas as pd
            >>> import numpy as np
            >>> from vectorbt.generic.splitters import RollingSplitter
            
            >>> # 创建示例数据（1年的日度数据）
            >>> dates = pd.date_range('2020-01-01', periods=365, freq='D')
            >>> prices = pd.Series(np.random.randn(365).cumsum(), index=dates)
            
            >>> splitter = RollingSplitter()
            
            >>> # 示例1：30天窗口，生成所有可能的窗口
            >>> windows = list(splitter.split(prices, window_len=30))
            >>> print(f"生成了 {len(windows)} 个窗口")
            >>> for i, (window_data,) in enumerate(windows[:3]):  # 显示前3个窗口
            ...     print(f"窗口{i+1}: {len(window_data)} 天")
            
            >>> # 示例2：选择10个窗口，每个窗口60天
            >>> windows = list(splitter.split(prices, n=10, window_len=60))
            >>> print(f"选择了 {len(windows)} 个窗口")
            
            >>> # 示例3：滚动窗口分割为训练集和测试集
            >>> # 80%用于训练，20%用于测试
            >>> for train_idx, test_idx in splitter.split(prices, window_len=50, set_lens=(0.8,)):
            ...     print(f"训练集: {len(train_idx)} 天, 测试集: {len(test_idx)} 天")
            ...     break  # 只显示第一个窗口
            
            >>> # 示例4：三分割 - 60%训练，20%验证，20%测试
            >>> for train_idx, valid_idx, test_idx in splitter.split(
            ...     prices, window_len=100, set_lens=(0.6, 0.2)
            ... ):
            ...     print(f"训练: {len(train_idx)}, 验证: {len(valid_idx)}, 测试: {len(test_idx)}")
            ...     break  # 只显示第一个窗口
            
            >>> # 示例5：前向验证示例
            >>> # 每个窗口前80%训练，后20%测试，模拟真实交易
            >>> results = []
            >>> for train_idx, test_idx in splitter.split(
            ...     prices, window_len=50, set_lens=(0.8,), n=5
            ... ):
            ...     # 模拟训练和测试过程
            ...     train_data = prices.iloc[train_idx]
            ...     test_data = prices.iloc[test_idx]
            ...     
            ...     # 计算简单移动平均策略的收益
            ...     train_mean = train_data.mean()
            ...     test_return = (test_data.iloc[-1] - test_data.iloc[0]) / test_data.iloc[0]
            ...     results.append(test_return)
            ...     
            ...     print(f"窗口期间收益: {test_return:.2%}")
            >>> print(f"平均收益: {np.mean(results):.2%}")
        
        异常处理：
            ValueError: 当参数不合法时抛出，例如：
                - n和window_len都未指定
                - n大于可能的窗口数量
                - 没有窗口满足min_len要求
        """
        # 将输入数据转换为标准格式
        X = to_any_array(X)
        
        # 提取或创建索引
        if isinstance(X, (pd.Series, pd.DataFrame)):
            index = X.index  # 使用原始索引
        else:
            index = pd.Index(np.arange(X.shape[0]))  # 创建数字索引

        # 解析窗口参数
        if window_len is None and n is None:
            raise ValueError("At least n or window_len must be set")
        
        # 计算窗口长度
        if window_len is None:
            window_len = len(index) // n  # 根据窗口数量计算长度
        
        # 处理比例形式的长度
        if 0 < window_len < 1:
            window_len = math.floor(window_len * len(index))
        
        # 生成所有可能的窗口起始和结束索引
        # 滑动窗口的关键：每个窗口固定长度，起始位置递增
        start_idxs = np.arange(len(index) - window_len + 1)  # 从0开始，到最后一个可能的起始位置
        end_idxs = np.arange(window_len - 1, len(index))     # 对应的结束位置

        # 过滤掉过短的窗口
        window_lens = end_idxs - start_idxs + 1  # 计算每个窗口的长度
        min_len_mask = window_lens >= min_len    # 创建长度过滤掩码
        
        if not np.any(min_len_mask):
            raise ValueError(f"There are no ranges that meet window_len>={min_len}")
        
        # 应用过滤掩码
        start_idxs = start_idxs[min_len_mask]
        end_idxs = end_idxs[min_len_mask]

        # 如果指定了n，均匀选择n个窗口
        if n is not None:
            if n > len(start_idxs):
                raise ValueError(f"n cannot be bigger than the maximum number of windows {len(start_idxs)}")
            
            # 使用线性空间均匀选择索引
            idxs = np.round(np.linspace(0, len(start_idxs) - 1, n)).astype(int)
            start_idxs = start_idxs[idxs]
            end_idxs = end_idxs[idxs]

        # 调用核心分割函数
        return split_ranges_into_sets(start_idxs, end_idxs, **kwargs)


class ExpandingSplitter(BaseSplitter):
    """
    扩展分割器（递增窗口分割器）
    
    使用递增窗口技术进行时间序列数据分割，实现扩展验证（expanding validation）。
    与滚动分割器不同，扩展分割器的窗口起始位置固定，窗口大小逐步增加。
    
    核心概念：
    - 锚定起点：所有窗口都从同一个起始点开始
    - 递增窗口：窗口的结束位置逐步向后移动，窗口大小不断增加
    - 累积学习：每个窗口都包含之前所有的历史数据
    
    与滚动分割器的区别：
    - 滚动分割器：窗口大小固定，起始位置移动
    - 扩展分割器：起始位置固定，窗口大小增加
    
    应用场景：
    - 评估模型在数据量增加时的性能变化
    - 研究更多历史数据对模型性能的影响
    - 模拟真实环境中数据逐步积累的情况
    - 长期投资策略的性能评估
    
    优势：
    - 更好地利用历史数据
    - 可以观察模型性能随数据量的变化
    - 适合需要大量历史数据的模型
    - 更稳定的验证结果
    """

    def split(self,
              X: tp.ArrayLike,
              n: tp.Optional[int] = None,
              min_len: int = 1,
              **kwargs) -> RangesT:
        """
        执行扩展窗口分割
        
        从固定起始点开始，创建大小递增的窗口序列。每个窗口都包含从起始点到当前位置的所有数据。
        
        参数说明：
            X (array-like): 要分割的时间序列数据
            n (int, optional): 要生成的窗口数量
                - 如果指定，将从所有可能的窗口中均匀选择n个
                - 如果未指定，将生成所有可能的窗口
            min_len (int): 窗口的最小长度，短于此长度的窗口将被过滤
            **kwargs: 传递给split_ranges_into_sets的额外参数，如：
                - set_lens: 指定训练集、验证集、测试集的长度比例
                - left_to_right: 指定分割方向
        
        返回值：
            RangesT: 分割结果生成器，每次迭代返回该窗口的所有数据集索引
        
        使用示例：
            >>> import pandas as pd
            >>> import numpy as np
            >>> from vectorbt.generic.splitters import ExpandingSplitter
            
            >>> # 创建示例数据（1年的日度数据）
            >>> dates = pd.date_range('2020-01-01', periods=365, freq='D')
            >>> prices = pd.Series(np.random.randn(365).cumsum(), index=dates)
            
            >>> splitter = ExpandingSplitter()
            
            >>> # 示例1：生成所有可能的扩展窗口
            >>> windows = list(splitter.split(prices, min_len=30))
            >>> print(f"生成了 {len(windows)} 个窗口")
            >>> for i, (window_data,) in enumerate(windows[:3]):  # 显示前3个窗口
            ...     print(f"窗口{i+1}: {len(window_data)} 天")
            
            >>> # 示例2：选择10个扩展窗口
            >>> windows = list(splitter.split(prices, n=10, min_len=30))
            >>> print(f"选择了 {len(windows)} 个窗口")
            >>> for i, (window_data,) in enumerate(windows):
            ...     print(f"窗口{i+1}: {len(window_data)} 天")
            
            >>> # 示例3：扩展窗口分割为训练集和测试集
            >>> # 前80%用于训练，后20%用于测试
            >>> for train_idx, test_idx in splitter.split(prices, n=5, set_lens=(0.8,), min_len=50):
            ...     print(f"训练集: {len(train_idx)} 天, 测试集: {len(test_idx)} 天")
            
            >>> # 示例4：反向分割 - 固定测试集大小，训练集递增
            >>> # 最后30天作为测试集，训练集大小递增
            >>> for train_idx, test_idx in splitter.split(
            ...     prices, n=5, set_lens=(30,), left_to_right=False, min_len=60
            ... ):
            ...     print(f"训练集: {len(train_idx)} 天, 测试集: {len(test_idx)} 天")
            
            >>> # 示例5：性能随数据量变化的分析
            >>> # 观察模型性能如何随着训练数据的增加而变化
            >>> performances = []
            >>> for train_idx, test_idx in splitter.split(
            ...     prices, n=10, set_lens=(0.9,), min_len=100
            ... ):
            ...     # 模拟训练和测试过程
            ...     train_data = prices.iloc[train_idx]
            ...     test_data = prices.iloc[test_idx]
            ...     
            ...     # 计算简单策略的性能
            ...     train_volatility = train_data.std()
            ...     test_return = (test_data.iloc[-1] - test_data.iloc[0]) / test_data.iloc[0]
            ...     sharpe_ratio = test_return / train_volatility if train_volatility > 0 else 0
            ...     
            ...     performances.append({
            ...         'train_size': len(train_idx),
            ...         'test_return': test_return,
            ...         'sharpe_ratio': sharpe_ratio
            ...     })
            ...     
            ...     print(f"训练集大小: {len(train_idx)}, 测试收益: {test_return:.2%}, 夏普比率: {sharpe_ratio:.2f}")
            
            >>> # 示例6：与滚动分割器的对比
            >>> # 扩展分割器的窗口大小递增，滚动分割器的窗口大小固定
            >>> from vectorbt.generic.splitters import RollingSplitter
            >>> 
            >>> rolling_splitter = RollingSplitter()
            >>> expanding_windows = list(splitter.split(prices, n=5, min_len=50))
            >>> rolling_windows = list(rolling_splitter.split(prices, n=5, window_len=100))
            >>> 
            >>> print("扩展分割器窗口大小:", [len(w[0]) for w in expanding_windows])
            >>> print("滚动分割器窗口大小:", [len(w[0]) for w in rolling_windows])
        
        异常处理：
            ValueError: 当参数不合法时抛出，例如：
                - n大于可能的窗口数量
                - 没有窗口满足min_len要求
        """
        # 将输入数据转换为标准格式
        X = to_any_array(X)
        
        # 提取或创建索引
        if isinstance(X, (pd.Series, pd.DataFrame)):
            index = X.index  # 使用原始索引
        else:
            index = pd.Index(np.arange(X.shape[0]))  # 创建数字索引

        # 生成扩展窗口的起始和结束索引
        # 扩展窗口的关键：起始位置固定为0，结束位置递增
        start_idxs = np.full(len(index), 0)        # 所有窗口都从索引0开始
        end_idxs = np.arange(len(index))           # 结束位置从0到最后一个索引

        # 过滤掉过短的窗口
        window_lens = end_idxs - start_idxs + 1  # 计算每个窗口的长度
        min_len_mask = window_lens >= min_len    # 创建长度过滤掩码
        
        if not np.any(min_len_mask):
            raise ValueError(f"There are no ranges that meet window_len>={min_len}")
        
        # 应用过滤掩码
        start_idxs = start_idxs[min_len_mask]
        end_idxs = end_idxs[min_len_mask]

        # 如果指定了n，均匀选择n个窗口
        if n is not None:
            if n > len(start_idxs):
                raise ValueError(f"n cannot be bigger than the maximum number of windows {len(start_idxs)}")
            
            # 使用线性空间均匀选择索引
            idxs = np.round(np.linspace(0, len(start_idxs) - 1, n)).astype(int)
            start_idxs = start_idxs[idxs]
            end_idxs = end_idxs[idxs]

        # 调用核心分割函数
        return split_ranges_into_sets(start_idxs, end_idxs, **kwargs)
