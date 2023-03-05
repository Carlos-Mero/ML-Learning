import os
import torch
import pandas as pd

os.makedirs(os.path.join('data'), exist_ok=True)
data_file = os.path.join('data', 'data.csv')
# 这里我们采用标准的 CSV 格式来存储数据

#with open(data_file, 'w') as f:
#    f.write('NumRooms, Alley, Price\n')
#    f.write('NaN, Pave, 127500\n')
#    f.write('2, NaN, 106000\n')
#    f.write('4, NaN, 178100\n')
#    f.write('NaN, NaN, 140000\n')

data = pd.read_csv(data_file)
print(data)
# pandas 会将缺失值自动识别为 NaN

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
# 使用平均值替换缺失值
# 而对于巷子类型这种离散数值，我们就分别将其转换为 0 和 1
inputs = pd.get_dummies(inputs, dummy_na=True)

print(inputs)

# 接着尝试将两个数值类型转变为pytorch当中的tensor

x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x, y)
# 其实直接使用强制类型转换就可以了
