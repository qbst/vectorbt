import numpy as np
import pandas as pd
from vectorbt.base import reshape_fns

def indexing_on_mapper(mapper: pd.Series, ref_obj: pd.DataFrame,
                       pd_indexing_func) -> pd.Series:
    """
    当对数据进行索引时，同步更新参数映射
    
    Args:
        mapper: 参数映射Series，索引为原始位置，值为参数值
        ref_obj: 参考DataFrame，用于确定索引操作的目标形状
        pd_indexing_func: 索引函数，定义如何对数据进行索引
    
    Returns:
        新的参数映射Series，反映索引操作后的参数关系
    """
    # 1. 创建位置索引的广播版本
    df_range_mapper = reshape_fns.broadcast_to(np.arange(len(mapper.index)), ref_obj)
    print(df_range_mapper)
    
    # 2. 对位置索引应用相同的索引操作
    loced_range_mapper = pd_indexing_func(df_range_mapper)
    print(loced_range_mapper)
    
    # 3. 根据索引结果获取对应的参数值
    new_mapper = mapper.iloc[loced_range_mapper.values[0]]
    print(new_mapper)
    
    # 4. 构造新的参数映射
    if isinstance(loced_range_mapper, pd.DataFrame):
        return pd.Series(new_mapper.values, 
                        index=loced_range_mapper.columns, 
                        name=mapper.name)
    elif isinstance(loced_range_mapper, pd.Series):
        return pd.Series([new_mapper], 
                        index=[loced_range_mapper.name], 
                        name=mapper.name)
    
    return None

# ===== 例子1: DataFrame列索引 =====
print("=== 例子1: DataFrame列索引 ===")

# 创建原始数据 - 包含3个不同参数的回测结果
data = pd.DataFrame({
    'strategy_A_period_10': [100, 105, 110, 108, 112],
    'strategy_A_period_20': [100, 102, 105, 107, 109], 
    'strategy_B_period_10': [100, 98, 95, 97, 99],
    'strategy_B_period_20': [100, 101, 103, 102, 105],
    'strategy_C_period_10': [100, 103, 106, 109, 112],
    'strategy_C_period_20': [100, 99, 98, 101, 104]
})

# 创建参数映射 - 每列对应的策略类型
param_mapper = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'], 
                        index=data.columns,
                        name='strategy')

print("原始数据:")
print(data)
print("\n原始参数映射:")
print(param_mapper)

# 定义索引操作 - 选择前3列
def select_first_three_cols(df):
    return df.iloc[:, :3]

# 应用indexing_on_mapper
new_mapper = indexing_on_mapper(param_mapper, data, select_first_three_cols)

print("\n索引操作后的数据:")
print(select_first_three_cols(data))
print("\n索引操作后的参数映射:")
print(new_mapper)