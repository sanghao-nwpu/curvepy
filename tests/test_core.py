import unittest
import numpy as np
from curvepy.core import Polyline

class TestPolyline(unittest.TestCase):
    
    def setUp(self):
        """设置测试数据，初始化 Polyline 实例"""
        self.points_a = [(0, 0), (1, 1), (2, 2)]
        self.points_b = [(0, 1), (1, 2), (2, 3)]
        self.polyline_a = Polyline(self.points_a)
        self.polyline_b = Polyline(self.points_b)

    def test_calculate_shortest_distance(self):
        """测试点到多段线的最短距离"""
        point = (1, 0)
        distance = self.polyline_a.calculate_distance_to_points(point)
        print(distance)
        self.assertAlmostEqual(distance, 1 / np.sqrt(2), places=5)

    def test_calculate_distance_to_polyline(self):
        """测试多段线到多段线的距离"""
        distances, avg_distance = self.polyline_a.calculate_distance_to_another_polyline(self.polyline_b)
        print(distances)
        self.assertAlmostEqual(avg_distance, 1.0 / np.sqrt(2), places=5)

    def test_find_nearest_polyline(self):
        """测试找到最邻近的多段线"""
        polyline_list = [self.polyline_b]  # 假设有一个多段线列表
        nearest_polyline, min_distance = self.polyline_a.find_closest_polyline(polyline_list)
        print(nearest_polyline)
        self.assertEqual(nearest_polyline, self.polyline_b)

    def test_calculate_projection(self):
        """测试计算点到线段的投影"""
        point = (1, 0)
        segment = ((0, 0), (2, 2))
        projection, t = self.polyline_a._calculate_projection(point, segment)
        print(projection, t)
        print(projection[0], projection[1])
        self.assertAlmostEqual(np.linalg.norm(projection - np.array([0.5, 0.5])), 0, places=5)


if __name__ == "__main__":
    unittest.main()
