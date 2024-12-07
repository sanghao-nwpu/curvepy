"""
@author: <sanghao_nwpu>
@email: <sanghao_nwpu@163.com>
@date: 2021/11/18
"""

from curvepy.core import Polyline, calculate_distance_between_polylines

def read_polyline_from_file(file_path):
    """
    从 txt 文件读取多段线的点，并创建 Polyline 对象。

    Args:
        file_path (str): txt 文件路径。

    Returns:
        Polyline: 由文件数据创建的 Polyline 对象。
    """
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            # 每行是一个点的坐标，按逗号分割
            point = tuple(map(float, line.strip().split(',')))
            points.append(point)
    return Polyline(points)


def example_polyline_main():
    # 从文件中读取多段线
    polyline1 = read_polyline_from_file('./example_data/polyline_1.txt')
    polyline2 = read_polyline_from_file('./example_data/polyline_2.txt')

    # 调用 Polyline 类中的方法进行操作
    print("Polyline1:")
    print(f"Points: {polyline1.points}")

    # 计算点到多段线的最短法向距离
    point = (2.5, 0.5)
    distance = polyline1.calculate_shortest_distance(point)
    print(f"Shortest distance from {point} to Polyline1: {distance}")

    # 计算多段线到多段线的误差
    distances_polyline, avg_distance = polyline1.calculate_distance_to_another_polyline(polyline2)
    print(f"Distance between Polyline1 and Polyline2: {avg_distance}")

    # 计算多段线 A 到 B 集合中最接近的多段线
    polyline_set = [polyline1, polyline2]  # 示例集合
    closest_polyline, min_distance = polyline1.find_closest_polyline(polyline_set)
    print(f"Closest polyline to Polyline1: {closest_polyline.points}")
    print(f"Distance: {min_distance}")

    # 计算多段线集合中的所有多段线与 Polyline1 的误差
    polyline1_set = [polyline1, polyline2]
    polyline2_set = [polyline1, polyline2]
    errors = calculate_distance_between_polylines(polyline1_set, polyline2_set)
    print("Errors between Polyline1 and all polylines in the set:")
    for i, error in enumerate(errors):
        print(f"Error with Polyline {i + 1}: {error}")


if __name__ == "__main__":
    example_polyline_main()
