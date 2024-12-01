# curvepy/core.py

import numpy as np
import matplotlib.pyplot as plt


class Curve:
    def __init__(self, control_points):
        """
        初始化曲线对象。

        :param control_points: 控制点，二维数组或列表，形如 [[x1, y1], [x2, y2], ...]
        """
        self.control_points = np.array(control_points)

    def evaluate(self, t):
        """
        基础的曲线评估函数，根据参数t评估曲线点。
        这是一个抽象方法，在子类中实现具体的评估方式。

        :param t: 参数值，通常是0到1之间的数。
        :return: 曲线上的点 (x, y)
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def plot(self, num_points=100):
        """
        绘制曲线。

        :param num_points: 曲线上点的数量，默认是100个点。
        """
        t_values = np.linspace(0, 1, num_points)
        curve_points = np.array([self.evaluate(t) for t in t_values])

        plt.plot(curve_points[:, 0], curve_points[:, 1], label=self.__class__.__name__)
        plt.scatter(self.control_points[:, 0], self.control_points[:, 1], color='red', label="Control Points")
        plt.title(f"{self.__class__.__name__} - Curve")
        plt.legend()
        plt.show()

    def fit(self, points, method="least_squares"):
        """
        用给定的点拟合曲线。这里实现一个简单的最小二乘法拟合。

        :param points: 目标点，形如 [[x1, y1], [x2, y2], ...]
        :param method: 拟合方法，当前支持 "least_squares"。
        :return: 拟合后的控制点
        """
        points = np.array(points)
        if method == "least_squares":
            # 使用最小二乘法拟合
            # 这里只是一个简化的例子，具体方法根据具体曲线类型不同而不同
            # 假设我们是通过线性插值拟合
            x_points, y_points = points[:, 0], points[:, 1]
            control_points = np.vstack([x_points, y_points]).T
            self.control_points = control_points
            return control_points
        else:
            raise ValueError(f"Method '{method}' not supported.")
