import numpy as np
import pandas as pd
from vectorbt.base.array_wrapper import ArrayWrapper

wrapper = ArrayWrapper(
    index=pd.Index(['A', 'B']),
    columns=pd.Index(['X1', 'X2', 'Y1', 'Y2']),
    ndim=2,
    group_by=['G1', 'G1', 'G2', 'G2'])

print(wrapper.grouped_ndim)