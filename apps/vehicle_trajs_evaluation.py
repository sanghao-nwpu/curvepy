import os
from curvepy.core import *


def calculate_overlap_duration(duration_a: list, duration_b: list):
    """
        计算两个范围的交集长度。

    Args:
        duration_a (list): 第一个范围的开始和结束时间。
        duration_b (list): 第二个范围的开始和结束时间。

    Returns:
        vehicle_trajs (dict): 由文件数据创建的 Polyline 对象。
    """
    # 获取两个时间范围的开始和结束时间
    start_a, end_a = duration_a
    start_b, end_b = duration_b
    
    # 计算交叉的起始和结束时间
    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)

    # 返回交集的长度，负值表示无交集
    return overlap_end - overlap_start

def cut_vehicle_traj_by_duration(vehicle_traj: list, duration: list):
    """
        根据时间范围裁剪车辆轨迹。

    Args:
        vehicle_traj (list): 车辆轨迹(时间戳和坐标点）。
        duration (list): 时间范围的开始和结束时间。

    Returns:
        cut_vehicle_traj (list): 裁剪后的车辆轨迹。
    """
    start_time, end_time = duration
    cut_vehicle_traj = [
        (time_stamp, point) for time_stamp, point in vehicle_traj 
        if start_time <= time_stamp <= end_time
    ]
    return cut_vehicle_traj

def read_vehicle_trajs_from_file(file_path):
    """
    从 txt 文件读取车辆轨迹数据，并创建 Polyline 对象。

    Args:
        file_path (str): txt 文件路径。

    Returns:
        vehicle_trajs (dict): 由文件数据创建的 Polyline 对象。
    """
    vehicle_trajs = {}  # 用于存储多段线ID及其对应的坐标点

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')  # 假设ID和坐标用空格分隔
            traj_id = (int)(parts[0])  # 第一列为多段线ID
            time_stamp = float(parts[1])  # 第二列为时间戳
            point = tuple(map(float, parts[2:]))  # 剩下的为点坐标
            
            if traj_id not in vehicle_trajs:
                vehicle_trajs[traj_id] = []  # 如果ID不存在，初始化一个空列表

            vehicle_trajs[traj_id].append((time_stamp, point))  # 添加时间戳和点坐标
    
    return vehicle_trajs

def find_associated_polyline(target_vehicle_traj: tuple, candidate_vehicle_trajs: dict):
    """
        查找与目标车辆轨迹最接近的候选车辆轨迹。

    Args:
        target_vehicle_traj (tuple): 目标车辆轨迹的 ID 和坐标点。
        candidate_vehicle_trajs (dict): 候选车辆轨迹的 ID 和坐标点。

    Returns:
        associated_traj_id (str): 与目标车辆轨迹最接近的候选车辆轨迹的 ID。
        min_dist (float): 与目标车辆轨迹的距离。
    """
    target_points = [point for _, point in target_vehicle_traj]
    target_polyline = Polyline(target_points)
    target_duration = [target_vehicle_traj[0][0], target_vehicle_traj[-1][0]]
    min_dist = float('inf')
    dist = 0
    associated_traj_id = None
    for traj_id, candidata_vehicle_traj in candidate_vehicle_trajs.items():
        points = [point for _, point in candidata_vehicle_traj]
        duration = [candidata_vehicle_traj[0][0], candidata_vehicle_traj[-1][0]]
        if calculate_overlap_duration(target_duration, duration) <= 0:
            continue
        polyline = Polyline(points)
        _, dist = target_polyline.calculate_distance_to_polyline(polyline)
        if dist < min_dist:
            min_dist = dist
            associated_traj_id = traj_id
    return associated_traj_id, min_dist

def evaluate_vehicle_trajs(trajs_est, trajs_gt, threshold=10):
    assoociated_num = 0
    unassoociated_num = 0
    errors = []
    for traj_est_id, traj_est in trajs_est.items():
        traj_est_points = [point for _, point in traj_est]
        traj_est_polyline = Polyline(traj_est_points)
        
        assoociated_traj_id, min_dist = find_associated_polyline(traj_est, trajs_gt)
        if min_dist < threshold:
            assoociated_num += 1
        else:
            unassoociated_num += 1
            continue
        assoociated_traj = trajs_gt[assoociated_traj_id]
        assoociated_traj_points = [point for _, point in assoociated_traj]
        assoociated_traj_polyline = Polyline(assoociated_traj_points)
        
        precision, recall, mae, rmse = \
            traj_est_polyline.evaluate_metric(assoociated_traj_polyline)
        errors.append([traj_est_id, assoociated_traj_id, precision, recall, mae, rmse])
    return errors


def main() -> None:
    # 获取脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 从文件中读取车辆轨迹
    trajs_est = read_vehicle_trajs_from_file(os.path.join(script_dir, '../example_data/vehicle_trajs_est.txt'))
    trajs_gt = read_vehicle_trajs_from_file(os.path.join(script_dir, '../example_data/vehicle_trajs_gt.txt'))
    trajs_obs = read_vehicle_trajs_from_file(os.path.join(script_dir, '../example_data/vehicle_trajs_obs.txt'))
    errors_est_vs_gt = evaluate_vehicle_trajs(trajs_est, trajs_gt)
    errors_est_vs_obs = evaluate_vehicle_trajs(trajs_est, trajs_obs)
    
    # 和真值评测准确率和精度
    precision = sum([x[2] for x in errors_est_vs_gt]) / len(errors_est_vs_gt)
    mae = sum([abs(x[4]) for x in errors_est_vs_gt]) / len(errors_est_vs_gt)
    # 和观测值评测召回率
    recall = sum([x[3] for x in errors_est_vs_obs]) / len(errors_est_vs_obs)

    # 打印结果
    print('Estimated vs Ground Truth:')
    print('Number of associated trajectories:', len(errors_est_vs_gt))
    print('Number of unassociated trajectories:', len(trajs_est) - len(errors_est_vs_gt))
    print('Error:', errors_est_vs_gt)
    print('Estimated vs Observed:')
    print('Number of associated trajectories:', len(errors_est_vs_obs))
    print('Number of unassociated trajectories:', len(trajs_est) - len(errors_est_vs_obs))
    print('Error:', errors_est_vs_obs)
    print('Overall:')
    print('Precision:', precision)
    print('Recall:', recall)
    print('MAE:', mae)

    return None

if __name__ == '__main__':
    main()
