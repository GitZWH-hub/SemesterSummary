import numpy as np
from numpy import random as rd
if __name__ == "__main__":
    arr = rd.randint(0, 10, size=(4, 9))
    print('******************** 随机生成4行9列的原arr:\n {}'.format(arr))
    print('******************** 数组arr的形状shape:\n', arr.shape)
    print('******************** 数组arr第一维长度（数组行数）:\n', arr.shape[0])

    arrC = arr[0, :]
    print('******************** 取第i行数据:\n {}'.format(arrC))

    arrC = arr[0:2, :]  # 这里不包含下标2行即第三行数据
    print('******************** 取第i行到第j行数据:\n {}'.format(arrC))

    arrC = arr[:, :]
    print('******************** 原数据不变:\n {}'.format(arrC))

    arrC = arr[:, 3:6]
    print('******************** 行不变，列从第i列取到第j列:\n {}'.format(arrC))

    arrMax = np.max(arr, axis=1)
    print('******************** 数组的每行最大值:\n {}'.format(arrC))
    arrMin = np.min(arr, axis=1)
    # arrC = np.max(arr, axis=0)
    # print('**** 数组的每列最大值:\n {}'.format(arrC))
    
    arrMax = np.reshape(a=arrMax, newshape=(arrMax.shape[0], 1))
    print('******************** 不改变数据内容，改变数组格式reshape:\n {}'.format(arrC))
    arrMin = np.reshape(a=arrMin, newshape=(arrMin.shape[0], 1))
    
    arr = (2 * arr - (arrMax + arrMin)) / (arrMax - arrMin)
    print(arr)

