import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

def read_tum_trajectory(file_path):
    """
    读取 TUM 格式轨迹文件（timestamp tx ty tz qx qy qz qw）
    返回时间戳（秒），平移向量（N×3），四元数（N×4，顺序 [qx,qy,qz,qw]）
    """
    data = np.loadtxt(file_path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]          # qx, qy, qz, qw
    return timestamps, positions, quaternions

def normalize_quaternions(q):
    """确保四元数符号一致（相邻帧点积为正），消除符号跳变"""
    q_norm = np.copy(q)
    for i in range(1, len(q_norm)):
        if np.dot(q_norm[i-1], q_norm[i]) < 0:
            q_norm[i] = -q_norm[i]
    return q_norm

def slerp_quat(q0, q1, t):
    """
    球面线性插值 (SLERP) 两个四元数
    q0, q1: 四元数数组 (x,y,z,w)
    t: 插值因子 [0,1]
    返回插值后的四元数 (x,y,z,w)
    """
    # 点积
    dot = np.dot(q0, q1)
    # 取最短路径
    if dot < 0:
        q1 = -q1
        dot = -dot
    # 数值稳定处理：当夹角很小时用线性插值
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        q /= np.linalg.norm(q)
        return q
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    coeff0 = np.sin((1 - t) * theta) / sin_theta
    coeff1 = np.sin(t * theta) / sin_theta
    q = coeff0 * q0 + coeff1 * q1
    q /= np.linalg.norm(q)
    return q

def interpolate_quaternions(t_src, q_src, t_target):
    """
    将源四元数序列插值到目标时间戳上
    使用 SLERP（球面线性插值）保证旋转插值的正确性
    """
    # 确保源四元数符号一致
    q_src = normalize_quaternions(q_src)
    
    q_interp = []
    for t in t_target:
        if t <= t_src[0]:
            q_interp.append(q_src[0])
        elif t >= t_src[-1]:
            q_interp.append(q_src[-1])
        else:
            idx = np.searchsorted(t_src, t) - 1
            t0, t1 = t_src[idx], t_src[idx+1]
            alpha = (t - t0) / (t1 - t0)
            # SLERP 插值
            q_slerp = slerp_quat(q_src[idx], q_src[idx+1], alpha)
            q_interp.append(q_slerp)
    return np.array(q_interp)

def compute_angular_velocity(timestamps, quaternions, min_dt=1e-6):
    """
    计算角速度（度/秒），使用中心差分提高精度
    """
    quaternions = normalize_quaternions(quaternions)
    
    n = len(timestamps)
    time_mids = []
    ang_vels = []
    
    # 中心差分：从索引 1 开始，到 n-2 结束
    for i in range(1, n - 1):
        dt = timestamps[i+1] - timestamps[i-1]
        if dt <= 2 * min_dt:
            continue
        
        q_prev = R.from_quat(quaternions[i-1])
        q_curr = R.from_quat(quaternions[i+1])
        rel = q_prev.inv() * q_curr
        angle_rad = rel.magnitude()
        ang_vel_deg = np.degrees(angle_rad) / dt
        time_mids.append(timestamps[i])  # 使用中间时刻
        ang_vels.append(ang_vel_deg)
    
    # 处理首尾点（使用前向/后向差分）
    if n >= 2:
        # 首点：前向差分
        dt = timestamps[1] - timestamps[0]
        if dt > min_dt:
            q0 = R.from_quat(quaternions[0])
            q1 = R.from_quat(quaternions[1])
            rel = q0.inv() * q1
            angle_rad = rel.magnitude()
            ang_vel_deg = np.degrees(angle_rad) / dt
            time_mids.insert(0, (timestamps[0] + timestamps[1]) / 2.0)
            ang_vels.insert(0, ang_vel_deg)
        
        # 尾点：后向差分
        dt = timestamps[-1] - timestamps[-2]
        if dt > min_dt:
            q_prev = R.from_quat(quaternions[-2])
            q_curr = R.from_quat(quaternions[-1])
            rel = q_prev.inv() * q_curr
            angle_rad = rel.magnitude()
            ang_vel_deg = np.degrees(angle_rad) / dt
            time_mids.append((timestamps[-2] + timestamps[-1]) / 2.0)
            ang_vels.append(ang_vel_deg)
    
    return np.array(time_mids), np.array(ang_vels)

def plot_angular_velocity_comparison(t_common, v1, v2, mean_diff, rmse, name1='Traj 1', name2='Traj 2'):
    """绘制角速度曲线和散点图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(t_common, v1, 'b-', label=name1, linewidth=1.5)
    ax1.plot(t_common, v2, 'r-', label=name2, linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angular Velocity (deg/s)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Angular Velocity over Time')
    
    ax2.scatter(v1, v2, alpha=0.6, s=10)
    ax2.plot([min(v1.min(), v2.min()), max(v1.max(), v2.max())],
             [min(v1.min(), v2.min()), max(v1.max(), v2.max())],
             'k--', alpha=0.5, label='Identity')
    ax2.set_xlabel(f'{name1} Angular Velocity (deg/s)')
    ax2.set_ylabel(f'{name2} Angular Velocity (deg/s)')
    ax2.set_title(f'Scatter Plot (Mean Diff = {mean_diff:.2f}, RMSE = {rmse:.2f} deg/s)')
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.show()

def main(file1, file2):
    # 读取轨迹
    t1, _, q1 = read_tum_trajectory(file1)
    t2, _, q2 = read_tum_trajectory(file2)

    
    # 确定共同时间范围
    t_start = max(t1[0], t2[0])
    t_end = min(t1[-1], t2[-1])
    
    if t_start >= t_end:
        print("两条轨迹在时间上没有重叠区域，无法比较。")
        return
    
    # 裁剪到共同时间范围
    mask1 = (t1 >= t_start) & (t1 <= t_end)
    mask2 = (t2 >= t_start) & (t2 <= t_end)
    t1_crop, q1_crop = t1[mask1], q1[mask1]
    t2_crop, q2_crop = t2[mask2], q2[mask2]
    
    # 以轨迹 1 的时间戳为基准，将轨迹 2 插值到轨迹 1 的时间轴
    q2_interp = interpolate_quaternions(t2_crop, q2_crop, t1_crop)
    
    # 在相同时间戳上计算角速度（频率一致）
    t_mid, v1 = compute_angular_velocity(t1_crop, q1_crop)
    _, v2 = compute_angular_velocity(t1_crop, q2_interp)  # 使用插值后的四元数
    
    # 确保两组角速度序列长度一致
    min_len = min(len(v1), len(v2))
    t_mid = t_mid[:min_len]
    v1 = v1[:min_len]
    v2 = v2[:min_len]
    
    # 计算差异指标
    diff = np.abs(v1 - v2)
    mean_diff = np.mean(diff)
    rmse = np.sqrt(np.mean((v1 - v2)**2))
    
    print(f"重叠时间区间：{t_start:.2f} ~ {t_end:.2f} s")
    print(f"有效数据点数：{len(t_mid)}")
    print(f"角速度差异均值：{mean_diff:.2f} deg/s")
    print(f"均方根误差 (RMSE): {rmse:.2f} deg/s")
    print(f"最大差异：{np.max(diff):.2f} deg/s")
    
    # 提取文件名用于图例
    label1 = os.path.basename(file1)
    label2 = os.path.basename(file2)
    
    plot_angular_velocity_comparison(t_mid, v1, v2, mean_diff, rmse, label1, label2)

if __name__ == "__main__":
    main("./bell412_gt/bell412_3_gt.txt", "./rovo/rovo_3.txt")
    main("./bell412_gt/bell412_4_gt.txt", "./rovo/rovo_4.txt")
    main("./bell412_gt/bell412_5_gt.txt", "./rovo/rovo_5.txt")
    main("./bell412_gt/bell412_3_gt.txt", "./rovo/rovo_3_scale.txt")
    main("./bell412_gt/bell412_4_gt.txt", "./rovo/rovo_4_scale.txt")
    main("./bell412_gt/bell412_5_gt.txt", "./rovo/rovo_5_scale.txt")
    
    main("./bell412_gt/bell412_3_gt.txt", "./dpvo/dpvo_3.txt")
    main("./bell412_gt/bell412_4_gt.txt", "./dpvo/dpvo_4.txt")
    main("./bell412_gt/bell412_5_gt.txt", "./dpvo/dpvo_5.txt")
    main("./bell412_gt/bell412_3_gt.txt", "./dpvo/dpvo_3_scale.txt")
    main("./bell412_gt/bell412_4_gt.txt", "./dpvo/dpvo_4_scale.txt")
    main("./bell412_gt/bell412_5_gt.txt", "./dpvo/dpvo_5_scale.txt")

    main("./bell412_gt/bell412_3_gt.txt", "./orbslam3/orbslam_3.txt")
    main("./bell412_gt/bell412_4_gt.txt", "./orbslam3/orbslam_4.txt")
    main("./bell412_gt/bell412_5_gt.txt", "./orbslam3/orbslam_5.txt")
    main("./bell412_gt/bell412_3_gt.txt", "./orbslam3/orbslam_3_scale.txt")
    main("./bell412_gt/bell412_4_gt.txt", "./orbslam3/orbslam_4_scale.txt")
    main("./bell412_gt/bell412_5_gt.txt", "./orbslam3/orbslam_5_scale.txt")


    
   
    