"""
@author: <sanghao_nwpu>
@email: <sanghao_nwpu@163.com>
@date: 2021/11/18
"""

import os
from curvepy.core import Polyline

def read_polyline_from_file(file_path):
    """
    从 txt 文件读取多段线的点，并创建 Polyline 对象。

    Args:
        file_path (str): txt 文件路径。

    Returns:
        Polyline: 由文件数据创建的 Polyline 对象。
    """
    polylines = {}  # 用于存储多段线ID及其对应的坐标点

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')  # 假设ID和坐标用空格分隔
            polyline_id = parts[0]  # 第一列为多段线ID
            polyline_time = parts[1]  # 第二列为时间戳
            point = tuple(map(float, parts[2:]))  # 剩下的为点坐标
            
            if polyline_id not in polylines:
                polylines[polyline_id] = []  # 如果ID不存在，初始化一个空列表

            polylines[polyline_id].append(point)  # 添加点到对应的多段线

    # 将字典转换为Polyline对象集合
    polyline_list = [Polyline(points) for _, points in polylines.items()]
    
    return polyline_list  # 返回Polyline对象的列表


def example_polyline_main(script_dir: str):
    # 从文件中读取多段线
    polylines1 = read_polyline_from_file(os.path.join(script_dir, '../example_data/polyline_1.txt'))
    polylines2 = read_polyline_from_file(os.path.join(script_dir, '../example_data/polyline_2.txt'))


    # 计算点到多段线的最短法向距离
    point = (2.5, 0.5)
    for polyline in polylines1:
        distance = polyline.calculate_distance_to_point(point, use_extension=True)
        print(f"Shortest distance from {point} to Polyline{polyline.points}: {distance}")

    # 计算多段线到多段线的误差
    for polyline1 in polylines1:
        for polyline2 in polylines2:
            distances, avg_distance = polyline1.calculate_distance_to_polyline(polyline2, use_extension=True)
            print(f"Distance between Polyline{polyline1.points} and Polyline{polyline2.points}: {avg_distance}")

    # 计算多段线 A 到 B 集合中最接近的多段线
    for polyline1 in polylines1:
        closest_polyline, min_distance = polyline1.find_closest_polyline(polylines2)
        # print(f"Closest polyline to Polyline{polyline1.id}: {closest_polyline.points}")
        print(f"Distance: {min_distance}")

if __name__ == "__main__":
    # 获取脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_polyline_main(script_dir)
