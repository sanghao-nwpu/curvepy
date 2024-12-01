# curvepy/fitting.py

import numpy as np
from .curves import CardinalCurve
from scipy.optimize import least_squares

def least_squares_fit(control_points, target_points):
    """
    使用最小二乘法拟合给定的目标点，返回拟合后的控制点。

    :param control_points: 初始控制点，二维数组或列表。
    :param target_points: 目标点，二维数组或列表。
    :return: 拟合后的控制点，二维数组或列表。
    """
    # 目标点的x和y
    target_points = np.array(target_points)
    x_target, y_target = target_points[:, 0], target_points[:, 1]

    # 假设我们使用三次样条拟合
    def residuals(control_points_flat):
        # 将控制点展平并进行重构
        control_points = control_points_flat.reshape(-1, 2)
        spline = CubicSplineCurve(control_points)
        fitted_points = np.array([spline.evaluate(t)[1] for t in np.linspace(0, 1, len(target_points))])
        return fitted_points - y_target

    # 将控制点展平并进行优化
    control_points_flat = control_points.flatten()
    result = least_squares(residuals, control_points_flat)

    # 将结果重新构造为二维数组
    fitted_control_points = result.x.reshape(-1, 2)
    return fitted_control_points


def generate_cardinal_curve(control_points, tension=0.5):
    """
    生成Cardinal曲线对象。

    :param control_points: 控制点，二维数组或列表，形如 [[x1, y1], [x2, y2], ...]
    :param tension: 张力，默认值为0.5。
    :return: CardinalCurve对象
    """
    return CardinalCurve(control_points, tension)
