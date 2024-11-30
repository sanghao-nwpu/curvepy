import numpy as np


class CardinalCurve:
    def __init__(self, control_points, tension=0.5):
        """
        初始化Cardinal曲线。

        :param control_points: 控制点的列表，例如 [(x1, y1), (x2, y2), ...]
        :param tension: 张力值，控制曲线形状
        """
        self.control_points = np.array(control_points)
        self.tension = tension

    @staticmethod
    def calculate(t: object, p0: object, p1: object, p2: object, p3: object) -> object:
        """
        根据参数t和控制点计算曲线上的点。

        :param t: 参数值
        :param p0: 控制点0
        :param p1: 控制点1
        :param p2: 控制点2
        :param p3: 控制点3
        :return: 曲线上的点
        """
        # Cardinal曲线的计算公式
        # 注意这里的具体实现可能需要根据具体公式调整
        t2 = t * t
        t3 = t2 * t
        return (2 * p1 +
                (-p0 + p2) * t +
                (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                (-p0 + 3 * p1 - 3 * p2 + p3) * t3) * 0.5

    def generate_curve(self, num_points=100):
        """
        生成Cardinal曲线。

        :param num_points: 生成的曲线点的数量
        :return: 包含生成曲线点的列表
        """
        points = []
        n = len(self.control_points)

        for i in range(1, n - 2):  # 确保有足够的控制点
            for j in range(num_points):
                t = j / (num_points - 1)  # 归一化t
                point = self.calculate(t,
                                       self.control_points[i - 1],
                                       self.control_points[i],
                                       self.control_points[i + 1],
                                       self.control_points[i + 2])
                points.append(point)

        return np.array(points)

