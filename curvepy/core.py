# curvepy/core.py

import numpy as np
import matplotlib.pyplot as plt
from .utils import cubic_spline_fit

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

class Polyline:
    """
    Represents a polyline, which is a sequence of connected line segments defined by a set of points.
    """

    def __init__(self, points=None):
        """
        Initialize the polyline with an optional set of points.

        Args:
            points (list of tuple): A list of (x, y) tuples representing the points of the polyline.
                                    If None, initializes an empty polyline.
        """
        self.points = points if points else []

    def add_point(self, point):
        """
        Add a point to the polyline.

        Args:
            point (tuple): A tuple (x, y) representing the new point to add.
        """
        self.points.append(point)

    def length(self):
        """
        Calculate the total length of the polyline.

        Returns:
            float: The total length of the polyline.
        """
        if len(self.points) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(self.points)):
            total_length += self._distance(self.points[i - 1], self.points[i])
        return total_length

    def get_segments(self):
        """
        Get the list of line segments that make up the polyline.

        Returns:
            list of tuple: A list of line segments, where each segment is a tuple ((x1, y1), (x2, y2)).
        """
        return [(self.points[i - 1], self.points[i]) for i in range(1, len(self.points))]

    @staticmethod
    def _distance(point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Args:
            point1 (tuple): The first point as (x, y).
            point2 (tuple): The second point as (x, y).

        Returns:
            float: The Euclidean distance between the two points.
        """
        x1, y1 = point1
        x2, y2 = point2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def discretize(self, num_samples, method="direct"):
        """
        对多段线进行等间距离散化。

        Args:
            num_samples (int): 需要生成的采样点数量。
            method (str): 离散化方法，可选 "direct" 或 "spline"。

        Returns:
            list[tuple]: 等间距离散化后的点集。
        """
        if method == "direct":
            # 简单的等间距采样，直接取点
            return self._direct_discretize(num_samples)
        elif method == "spline":
            # 使用三次样条拟合后再离散化
            return cubic_spline_fit(self.points, num_samples)
        else:
            raise ValueError(f"Unknown method '{method}' for discretization.")

    def _direct_discretize(self, num_samples):
        """
        使用线性插值进行简单的等间距离散化。

        Args:
            num_samples (int): 需要生成的采样点数量。

        Returns:
            list[tuple]: 等间距采样后的点集。
        """
        total_length = self.length()
        segment_lengths = [((self.points[i][0] - self.points[i - 1][0]) ** 2 +
                            (self.points[i][1] - self.points[i - 1][1]) ** 2) ** 0.5
                           for i in range(1, len(self.points))]

        # 计算累计长度
        cumulative_lengths = [0] + np.cumsum(segment_lengths).tolist()

        # 等间距点的目标位置
        target_distances = np.linspace(0, total_length, num_samples)

        # 在累计长度中找到对应的插值点
        sampled_points = []
        for d in target_distances:
            for i in range(1, len(cumulative_lengths)):
                if cumulative_lengths[i - 1] <= d <= cumulative_lengths[i]:
                    # 插值计算点坐标
                    t = (d - cumulative_lengths[i - 1]) / (cumulative_lengths[i] - cumulative_lengths[i - 1])
                    x = self.points[i - 1][0] + t * (self.points[i][0] - self.points[i - 1][0])
                    y = self.points[i - 1][1] + t * (self.points[i][1] - self.points[i - 1][1])
                    sampled_points.append((x, y))
                    break

        return sampled_points

    def calculate_shortest_distance(self, point: tuple, use_extension=True):
        """
        计算点到多段线的最短距离。可以选择是否考虑线段延长线。

        Args:
            point (tuple): 输入点的坐标 (x, y)。
            use_extension (bool): 是否考虑延长线，默认为 True。

        Returns:
            float: 点到多段线的最短法向距离。
        """
        px, py = point
        projections_outside_start = True
        projections_outside_end = True
        min_distance = float('inf')

        for i in range(1, len(self.points)):
            x1, y1 = self.points[i - 1]
            x2, y2 = self.points[i]

            # 计算点到线段的投影参数 t
            segment_vec = np.array([x2 - x1, y2 - y1])
            point_vec = np.array([px - x1, py - y1])
            segment_length_squared = np.dot(segment_vec, segment_vec)

            if segment_length_squared > 0:
                t = np.dot(point_vec, segment_vec) / segment_length_squared
            else:
                t = 0

            # 投影点
            if t < 0: # 投影点在起点外
                projection_point = np.array([x1, y1])
                projections_outside_end = False
            elif t > 1: # 投影点在终点外
                projection_point = np.array([x2, y2])
                projections_outside_start = False
            else:
                projection_point = np.array([x1, y1]) + t * segment_vec
                projections_outside_start = False
                projections_outside_end = False

            # 计算点到投影点的欧式距离
            distance = np.linalg.norm(np.array([px, py]) - projection_point)
            # 更新最小距离
            min_distance = min(min_distance, distance)

        # 如果选择考虑延长线，且所有投影点都在起点外，计算点到第一条线段延长线的法向距离
        if use_extension and projections_outside_start:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            segment_vec = np.array([x2 - x1, y2 - y1])
            point_vec = np.array([px - x1, py - y1])
            segment_length_squared = np.dot(segment_vec, segment_vec)

            if segment_length_squared > 0:
                t = np.dot(point_vec, segment_vec) / segment_length_squared
                projection_point = np.array([x1, y1]) + t * segment_vec  # 延长线起点
                min_distance = np.linalg.norm(np.array([px, py]) - projection_point)

        # 如果选择考虑延长线，且所有投影点都在终点外，计算点到最后一条线段延长线的法向距离
        if use_extension and projections_outside_end:
            x1, y1 = self.points[-2]
            x2, y2 = self.points[-1]
            segment_vec = np.array([x2 - x1, y2 - y1])
            point_vec = np.array([px - x1, py - y1])
            segment_length_squared = np.dot(segment_vec, segment_vec)

            if segment_length_squared > 0:
                t = np.dot(point_vec, segment_vec) / segment_length_squared
                projection_point = np.array([x1, y1]) + t * segment_vec  # 延长线起点
                min_distance = np.linalg.norm(np.array([px, py]) - projection_point)

        return min_distance

    def calculate_normal_distance(self, point):
        """
        计算点到多段线中最近线段的最短法向距离。

        Args:
            point (tuple): 输入点的坐标 (x, y)。
            polyline_points (list[tuple]): 多段线的离散点集 [(x1, y1), (x2, y2), ...]。

        Returns:
            float: 点到最近线段的最短法向距离。
        """

        min_distance = float('inf')  # 初始化为无穷大
        px, py = point

        for i in range(1, len(polyline_points)):
            # 线段起点和终点
            x1, y1 = polyline_points[i - 1]
            x2, y2 = polyline_points[i]

            # 将线段向量和点到起点的向量表示为 numpy 数组
            segment_vec = np.array([x2 - x1, y2 - y1])
            point_vec = np.array([px - x1, py - y1])

            # 线段长度平方（避免多次计算平方根提高性能）
            segment_length_squared = np.dot(segment_vec, segment_vec)

            # 点投影到线段的参数 t（0 <= t <= 1 表示投影点在线段上）
            if segment_length_squared > 0:
                t = np.dot(point_vec, segment_vec) / segment_length_squared
            else:
                t = 0  # 零长度线段的特殊情况

            # 投影点
            if t < 0:  # 投影点在起点外
                projection_point = np.array([x1, y1])
            elif t > 1:  # 投影点在终点外
                projection_point = np.array([x2, y2])
            else:  # 投影点在线段上
                projection_point = np.array([x1, y1]) + t * segment_vec

            # 计算点到投影点的欧式距离
            distance = np.linalg.norm(np.array([px, py]) - projection_point)

            # 更新最小距离
            min_distance = min(min_distance, distance)

        return min_distance
