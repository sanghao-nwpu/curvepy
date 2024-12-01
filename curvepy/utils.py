# curvepy/utils.py

import numpy as np


def generate_random_control_points(num_points=5, x_range=(0, 10), y_range=(0, 10)):
    """
    生成随机的控制点。

    :param num_points: 控制点的数量。
    :param x_range: 控制点x坐标的范围，默认(0, 10)。
    :param y_range: 控制点y坐标的范围，默认(0, 10)。
    :return: 随机控制点，二维数组。
    """
    x_coords = np.random.uniform(x_range[0], x_range[1], num_points)
    y_coords = np.random.uniform(y_range[0], y_range[1], num_points)
    return np.vstack((x_coords, y_coords)).T


def calculate_fitting_error(control_points, target_points, curve_type="cardinal", tension=0.5):
    """
    计算拟合误差。

    :param control_points: 控制点，二维数组或列表。
    :param target_points: 目标点，二维数组或列表。
    :param curve_type: 使用的曲线类型，"cardinal" 或 "spline"。
    :param tension: 张力，默认值为0.5，仅对Cardinal曲线有效。
    :return: 拟合误差。
    """
    if curve_type == "cardinal":
        from curvepy.curves import CardinalCurve
        curve = CardinalCurve(control_points, tension)
    elif curve_type == "spline":
        from curvepy.curves import CubicSplineCurve
        curve = CubicSplineCurve(control_points)
    else:
        raise ValueError(f"Unsupported curve type: {curve_type}")

    # 计算曲线点与目标点之间的误差
    curve_points = np.array([curve.evaluate(t) for t in np.linspace(0, 1, len(target_points))])
    error = np.linalg.norm(curve_points[:, 1] - target_points[:, 1])  # y轴的误差
    return error
