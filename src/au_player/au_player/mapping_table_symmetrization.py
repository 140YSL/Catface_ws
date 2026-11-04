#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取指定路径的 npy 文件（三维数组），打印，对特定行处理后重新写回。"""

import numpy as np
"""
    /mujoco_ctrl 各位含义
    0: neck_pitch
    1: neck_yaw
    2: Head
    3: Jaw
    4: left_upper_palate
    5: right_upper_palate
    6: left_eye_bottom_skin
    7: left_eye_upper_skin
    8: right_eye_bottom_skin
    9: right_eye_upper_skin
    10: left_ear_rotate
    11: left_ear
    12: right_ear_rotate
    13: right_ear
"""

# ==================== 参数区 ====================
NPY_PATH = "../config/au2joint.npy"   # 改成你的 npy 文件路径
# ===============================================

AU_NAMES = [
    "LAU143",
    "RAU143",
    "AU109+110",
    "AU118",
    "AU26",
    "LEAD101",
    "LEAD104",
    "LEAD106",
    "READ101",
    "READ104",
    "READ106",
]

def process_symmertrization(arr_row: np.ndarray) -> np.ndarray:
    """
    对称处理
    """
    arr_row_sym = arr_row.copy()  # 避免修改原数组之外的副作用

    arr_row_sym[:, 5] = -arr_row_sym[:, 4]

    arr_row_sym[:, 8] = -arr_row_sym[:, 6]
    arr_row_sym[:, 9] = -arr_row_sym[:, 7]
    arr_row_sym[:, 12] = -arr_row_sym[:, 10]
    arr_row_sym[:, 13] = -arr_row_sym[:, 11]

    arr_row_sym[:, 6] = np.zeros(5)
    arr_row_sym[:, 7] = np.zeros(5)
    arr_row_sym[:, 10] = np.zeros(5)
    arr_row_sym[:, 11] = np.zeros(5)
    # print("arr_row_sym", arr_row_sym)
    return arr_row_sym

def process_limit(arr_row: np.ndarray, same_index: int) -> np.ndarray:
    """
    限幅处理
    """
    arr_row_lim = arr_row.copy()  # 避免修改原数组之外的副作用

    if same_index>4:
        print("错误的开始相同位")
        pass

    for i in range(same_index+1, 5):
        arr_row_lim[i, :] = arr_row_lim[same_index, :]

    # print("arr_row_lim", arr_row_lim)
    return arr_row_lim

def process_total(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()  # 避免修改原数组之外的副作用

    threshold = 1.5e-3
    arr[abs(arr) < threshold] = 0.0   # 所有小于 1.5e-3 的元素置零

    for i in range(len(AU_NAMES)):
        print(f"{AU_NAMES[i]} : {arr[i, :, :]} \n")

    # 上限截止
    arr[0, :, : ] = process_limit(arr[0, :, : ], 3)
    arr[3, :, : ] = process_limit(arr[3, :, : ], 0)
    arr[5, :, : ] = process_limit(arr[5, :, : ], 2)
    arr[7, :, : ] = process_limit(arr[7, :, : ], 2)

    # 左右对称
    arr[2, :, : ] = process_symmertrization(arr[2, :, : ])
    arr[3, :, : ] = process_symmertrization(arr[3, :, : ])
    arr[1, :, : ] = process_symmertrization(arr[0, :, : ])
    arr[8, :, : ] = process_symmertrization(arr[5, :, : ])
    arr[9, :, : ] = process_symmertrization(arr[6, :, : ])
    arr[10, :, : ] = process_symmertrization(arr[7, :, : ])

    # process_symmertrization(arr[0, :, : ])

    return arr


def main():
    # 1. 读取
    arr = np.load(NPY_PATH)

    for i in range(len(AU_NAMES)):
        print(f"{AU_NAMES[i]} : {arr[i, :, :]} \n")


    # 2. 处理特定行
    arr_new = process_total(arr)
    print("处理后的文件" ,arr_new)

    # # # 3. 写回同一文件
    np.save(NPY_PATH, arr_new)


if __name__ == "__main__":
    main()