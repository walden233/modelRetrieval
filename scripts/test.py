# import numpy as np

# arr_2d = np.array([[1, 2, 8],
#                    [5, 9, 3]])

# # 查找所有值大于 4 的元素的索引
# indices_2d = np.where(arr_2d > 4)

# print(f"返回内容: {indices_2d}")
# # 返回内容: (array([0, 1, 1]), array([2, 0, 1]))

# # 分别获取行和列的索引
# row_indices = indices_2d[0]
# col_indices = indices_2d[1]

# print(f"满足条件的行索引: {row_indices}") # [0 1 1]
# print(f"满足条件的列索引: {col_indices}") # [2 0 1]

import torch
print(torch.randn(1, 1, 5))