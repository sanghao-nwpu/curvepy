import numpy as np
from scipy.interpolate import CubicSpline as SciPyCubicSpline


class CubicSpline:
    def __init__(self, control_points):
        """
        初始化三次样条曲线。

        :param control_points: 控制点的列表，例如 [(x1, y1), (x2, y2), ...]
        """
        self.control_points = np.array(control_points)
        self.x = self.control_points[:, 0]
        self.y = self.control_points[:, 1]
        self.cs = SciPyCubicSpline(self.x, self.y)

    def generate_curve(self, num_points=100):
        """
        生成三次样条曲线。

        :param num_points: 生成的曲线点的数量
        :return: 包含生成曲线点的列表
        """
        x_new = np.linspace(self.x.min(), self.x.max(), num_points)
        y_new = self.cs(x_new)
        return np.column_stack((x_new, y_new))  # 返回(x, y)坐标点
