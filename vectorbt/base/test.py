import pandas as pd
import numpy as np

def to_any_array(arg):

    return np.asarray(arg)

def broadcast_to_array_of(arg1, arg2):

    arg1 = np.asarray(arg1)
    arg2 = np.asarray(arg2)

    if arg1.ndim == arg2.ndim + 1:
        if arg1.shape[1:] == arg2.shape:
            return arg1
        
    if arg2.ndim == 0:
        return arg1
    
    for i in range(arg2.ndim):
        arg1 = np.expand_dims(arg1, axis=-1)
        
    return np.tile(arg1, (1, *arg2.shape))

def broadcast_to_axis_of(arg1, arg2, axis, require_kwargs):
    if require_kwargs is None:
        require_kwargs = {}
    
    # 确保 arg2 是 numpy 数组或 pandas 对象
    arg2 = to_any_array(arg2)
    
    if arg2.ndim < axis + 1:
        return np.broadcast_to(arg1, (1,))[0]  
    
    arg1 = np.broadcast_to(arg1, (arg2.shape[axis],))
    
    arg1 = np.require(arg1, **require_kwargs)
    
    return arg1
if __name__ == "__main__":
    arg1 = [[1, 2, 3], [4, 5, 6]]
    arg2 = np.array([10, 20])    

    print(broadcast_to_axis_of(arg1, arg2, 0, None))


