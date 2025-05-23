from tqdm import tqdm
import time

# 简单包装可迭代对象
# 自定义显示格式
for i in tqdm(range(100), 
              desc="处理中", 
              unit="文件", 
              unit_scale=True):
    time.sleep(0.01)