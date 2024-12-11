# curvepy/core.py

import numpy as np
import matplotlib.pyplot as plt
# from .utils import cubic_spline_fit

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
            id (int): The unique identifier of the polyline.
            points (list of tuple): A list of (x, y) tuples representing the points of the polyline.
                                    If None, initializes an empty polyline.
        """
        # self.id = id
        self.points = points if points else []

    def add_point(self, point):
        """
        Add a point to the polyline.

        Args:
            point (tuple): A tuple (x, y) representing the new point to add.
        """
        self.points.append(point)

    # def setID(self, id):
    #     """
    #     Set the unique identifier of the polyline.

    #     Args:
    #         id (int): The new unique identifier of the polyline.
    #     """
    #     self.id = id

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
        discretize_points = None
        if method == "direct":
            # 简单的等间距采样，直接取点
            discretize_points = self._direct_discretize(num_samples)
        elif method == "spline":
            # 使用三次样条拟合后再离散化
            # discretize_points = cubic_spline_fit(self.points, num_samples)
            raise ValueError(f"Unknown method '{method}' for discretization.")
        else:
            raise ValueError(f"Unknown method '{method}' for discretization.")
        return Polyline(discretize_points)

    def _direct_discretize(self, num_samples):
        """
        使用线性插值进行简单的等间距离散化。

        Args:
            num_samples (int): 需要生成的采样点数量。

        Returns:
            list[tuple]: 等间距采样后的点集。
        """
        total_length = self.length()
        segment_lengths = [
            ((self.points[i][0] - self.points[i - 1][0]) ** 2 +
             (self.points[i][1] - self.points[i - 1][1]) ** 2) ** 0.5
            for i in range(1, len(self.points))
        ]

        # 计算累计长度
        cumulative_lengths = [0] + np.cumsum(segment_lengths).tolist()
        # 等间距点的目标位置(+1是为了包括终点)
        target_distances = np.linspace(0, total_length, num_samples + 1)

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

    def _calculate_projection(self, point: tuple, segment: tuple):
        """
        计算点到线段的投影点，若投影点在线段的端点外，则返回端点。

        Args:
            point (tuple): 输入点的坐标 (px, py)。
            segment (tuple): 线段的端点坐标 ((x1, y1), (x2, y2))。

        Returns:
            np.array: 投影点坐标。
            bool: 投影是否在线段的端点外。
        """
        px, py = point
        x1, y1 = segment[0]
        x2, y2 = segment[1]

        segment_vec = np.array([x2 - x1, y2 - y1])
        point_vec = np.array([px - x1, py - y1])
        segment_length_squared = np.dot(segment_vec, segment_vec)

        if segment_length_squared > 0:
            t = np.dot(point_vec, segment_vec) / segment_length_squared
        else:
            t = 0

        if t < 0:  # 投影点在起点外
            projection_point = np.array([x1, y1])
        elif t > 1:  # 投影点在终点外
            projection_point = np.array([x2, y2])
        else:  # 投影点在线段上
            projection_point = np.array([x1, y1]) + t * segment_vec
        return projection_point, t

    def calculate_distance_to_point(self, point: tuple, use_extension=False):
        """
        计算点到多段线的最短距离。可以选择是否考虑线段延长线。

        Args:
            point (tuple): 输入点的坐标 (x, y)。
            use_extension (bool): 是否考虑延长线，默认为 True。

        Returns:
            float: 点到多段线的最短法向距离。
        """
        px, py = point
        exist_projections_inside = False
        min_distance = float('inf')

        for i in range(1, len(self.points)):
            segment = (self.points[i - 1], self.points[i])

            # 计算点到当前线段的投影点
            projection_point, t = self._calculate_projection(point, segment)

            # 更新最小距离
            distance = np.linalg.norm(np.array([px, py]) - projection_point)
            min_distance = min(min_distance, distance)

            # 判断是否存在投影点在线段的端点内
            if t > 0 and t < 1:
                exist_projections_inside = True
           
            # 计算点到投影点的欧式距离
            distance = np.linalg.norm(np.array([px, py]) - projection_point)
            # 更新最小距离
            min_distance = min(min_distance, distance)
        
        # 考虑延长线
        if use_extension and not exist_projections_inside:
            segment = (self.points[0], self.points[1])
            t = self._calculate_projection(point, segment)[1]
            if t > 1:  # 投影点在终点外
                segment = (self.points[-2], self.points[-1])
                t = self._calculate_projection(point, segment)[1]
            projection_point_extension = np.array(segment[0]) + t * np.array(np.array(segment[1]) - np.array(segment[0]))
            min_distance = np.linalg.norm(np.array([px, py]) - projection_point_extension)
        
        return min_distance
    
    def calculate_distance_to_polyline(self, polyline: 'Polyline', use_extension=False):
        """
        计算当前多段线到另一个多段线的距离。

        Args:
            polyline (Polyline): 另一个多段线对象。

        Returns:
            list: 当前多段线每个点到多段线 B 的最短距离列表。
        """
        distances = []
        for point in self.points:
            # 计算当前多段线的每个点到多段线B的最短距离
            distance = polyline.calculate_distance_to_point(point, use_extension)
            distances.append(distance)

        avg_distance = sum(distances) / len(distances)

        return distances, avg_distance
    
    def evaluate_metric(self, polyline_base: 'Polyline', precision_thre=1.5):
        """
        评测当前多段线的性能指标。

        Args:
            polyline_base (Polyline): 多段线基准。

        Returns:
            float: 返回的指标值。
        """
        num = round(self.length() / 0.2)
        eva_polyline = self.discretize(num)
        num = round(polyline_base.length() / 0.2)
        eva_polyline_base = polyline_base.discretize(num)

        distances, _ = eva_polyline.calculate_distance_to_polyline(eva_polyline_base)

        precision = (len([x for x in distances if x < precision_thre])) / len(distances)
        recall = (len([x for x in distances if x < precision_thre])) / len(eva_polyline_base.points)
        mae = sum([abs(x) for x in distances]) / len(distances)
        rmse = (sum([x ** 2 for x in distances]) / len(distances)) ** 0.5

        return precision, recall, mae, rmse

    def find_closest_polyline(self, candidate_polylines: list):
        """
        从候选多段线集合中找到与当前多段线最近的多段线。

        Args:
            candidate_polylines (list[Polyline]): 候选多段线对象的列表。

        Returns:
            Polyline: 距离当前多段线最近的多段线对象。
            float: 距离值。
        """
        min_distance = float('inf')
        closest_polyline = None

        for candidate_polyline in candidate_polylines:
            # 计算当前多段线与候选多段线的距离
            distances, avg_distance = self.calculate_distance_to_polyline(candidate_polyline)

            # 如果找到一个更小的距离，则更新最小距离和最接近的多段线
            if avg_distance < min_distance:
                min_distance = avg_distance
                closest_polyline = candidate_polyline

        return closest_polyline, min_distance

def discretize_polyline(polyline: 'Polyline', num_samples: int, method: str) -> 'Polyline':
    """
    对多段线进行等间距散化。

    Args:
        polyline (Polyline): 多段线对象。
        num_samples (int): 需要生成的采样点数量。
        method (str): 离散化方法，可选 "direct" 或 "spline"。

    Returns:
        Polyline: 等间距散化后的多段线对象。
    """
    return polyline.discretize(num_samples, method)

def calculate_distance_between_polylines(polylines_estimation: list, polylines_groundtruth: list) -> list:
    """
    计算估计值集合中每个多段线与真值集合中最近多段线之间的误差，并将所有误差值记录在列表中。

    Args:
        polylines_estimation (list[Polyline]): 多段线集合A。
        polylines_groundtruth (list[Polyline]): 多段线集合B。

    Returns:
        list: 存储每个多段线与集合B中最近多段线之间的误差的列表。
    """
    error_list = []

    # 遍历集合A中的每个多段线
    for polyline_estimation in polylines_estimation:
        # 找到与当前多段线最接近的多段线，并计算误差
        closest_polyline, min_distance = polyline_estimation.find_closest_polyline(polylines_groundtruth)
        
        # 将误差值添加到列表中（这里使用最小距离或者平均距离，具体依需求而定）
        error_list.append(min_distance)

    return error_list

