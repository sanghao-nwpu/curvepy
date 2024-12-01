# curvepy/visualization.py

import matplotlib.pyplot as plt
from core import Curve
from curves import CardinalCurve


def plot_curve(curve: Curve, num_points=100):
    """
    绘制给定的曲线。

    :param curve: 曲线对象，必须继承自Curve。
    :param num_points: 曲线上的点的数量，默认为100。
    """
    t_values = np.linspace(0, 1, num_points)
    curve_points = np.array([curve.evaluate(t) for t in t_values])

    plt.plot(curve_points[:, 0], curve_points[:, 1], label=curve.__class__.__name__)
    plt.scatter(curve.control_points[:, 0], curve.control_points[:, 1], color='red', label="Control Points")
    plt.title(f"{curve.__class__.__name__} - Curve")
    plt.legend()
    plt.show()


def plot_fitted_curve(control_points, target_points, curve_type="cardinal", tension=0.5):
    """
    根据拟合结果绘制曲线，支持Cardinal曲线和Cubic Spline曲线。

    :param control_points: 控制点，二维数组或列表。
    :param target_points: 目标点，二维数组或列表。
    :param curve_type: 使用的曲线类型，"cardinal" 或 "spline"。
    :param tension: 张力，默认值为0.5，仅对Cardinal曲线有效。
    """
    if curve_type == "cardinal":
        curve = CardinalCurve(control_points, tension)
    elif curve_type == "spline":
        curve = CubicSplineCurve(control_points)
    else:
        raise ValueError(f"Unsupported curve type: {curve_type}")

    # 绘制拟合曲线和目标点
    plt.scatter(target_points[:, 0], target_points[:, 1], color="green", label="Target Points")
    plot_curve(curve)
