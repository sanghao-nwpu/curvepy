# curvepy/curves.py

import numpy as np
from curvepy.core import Curve


class CardinalCurve(Curve):
    def __init__(self, control_points, tension=0.5):
        """
        Cardinal曲线初始化，支持张力参数调整。

        :param control_points: 控制点，二维数组或列表，形如 [[x1, y1], [x2, y2], ...]
        :param tension: 张力参数，控制曲线的弯曲程度，默认0.5。
        """
        super().__init__(control_points)
        self.tension = tension

    def evaluate(self, t):
        """
        评估Cardinal曲线上的点。

        :param t: 参数t，0 <= t <= 1
        :return: 曲线上的点 (x, y)
        """
        p = self.control_points
        n = len(p)

        # 对于一个给定的t，计算出对应的插值点
        i = int(t * (n - 1))
        t0 = (t - i) * (n - 1)

        p0, p1, p2, p3 = p[i - 1 if i > 0 else 0], p[i], p[i + 1 if i + 1 < n else i], p[i + 2 if i + 2 < n else i + 1]

        # 采用Catmull-Rom样条公式，考虑张力参数
        t_squared = t0 * t0
        t_cubed = t_squared * t0
        m0 = (p2 - p0) * self.tension
        m1 = (p3 - p1) * self.tension
        p0 = p1 + (0.5 * (p2 - p0) + m0) * t0 + (0.5 * (p3 - p1) + m1) * t_squared
        p1 = p0 + (t0 - 1) * t_cubed
        return np.array(p1)

