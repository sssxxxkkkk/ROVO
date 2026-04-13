import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import matplotlib.lines as mlines

# ---------------------- 基础工具函数（保留核心逻辑，无破坏性修改） ----------------------
def read_tum_trajectory(file_path):
    """读取 TUM 格式轨迹文件（timestamp tx ty tz qx qy qz qw）"""
    if not os.path.exists(file_path):
        return None, None, None
    data = np.loadtxt(file_path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]  # qx, qy, qz, qw
    return timestamps, positions, quaternions

def normalize_quaternions(q):
    """确保四元数符号一致，消除符号跳变"""
    q_norm = np.copy(q)
    for i in range(1, len(q_norm)):
        if np.dot(q_norm[i-1], q_norm[i]) < 0:
            q_norm[i] = -q_norm[i]
    return q_norm

def slerp_quat(q0, q1, t):
    """球面线性插值 (SLERP) 两个四元数"""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
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
    """将源四元数序列插值到目标时间戳上"""
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
            q_slerp = slerp_quat(q_src[idx], q_src[idx+1], alpha)
            q_interp.append(q_slerp)
    return np.array(q_interp)

def compute_angular_velocity(timestamps, quaternions, min_dt=1e-6):
    """计算角速度（度/秒），中心差分保证精度"""
    quaternions = normalize_quaternions(quaternions)
    n = len(timestamps)
    time_mids = []
    ang_vels = []
    
    # 中心差分（主体数据）
    for i in range(1, n - 1):
        dt = timestamps[i+1] - timestamps[i-1]
        if dt <= 2 * min_dt:
            continue
        q_prev = R.from_quat(quaternions[i-1])
        q_curr = R.from_quat(quaternions[i+1])
        rel = q_prev.inv() * q_curr
        angle_rad = rel.magnitude()
        ang_vel_deg = np.degrees(angle_rad) / dt
        time_mids.append(timestamps[i])
        ang_vels.append(ang_vel_deg)
    
    # 首尾点前向/后向差分补全
    if n >= 2:
        # 首点
        dt = timestamps[1] - timestamps[0]
        if dt > min_dt:
            q0 = R.from_quat(quaternions[0])
            q1 = R.from_quat(quaternions[1])
            rel = q0.inv() * q1
            angle_rad = rel.magnitude()
            ang_vel_deg = np.degrees(angle_rad) / dt
            time_mids.insert(0, (timestamps[0] + timestamps[1]) / 2.0)
            ang_vels.insert(0, ang_vel_deg)
        # 尾点
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

# ---------------------- 新增核心函数：统一时间轴处理 ----------------------
def get_trajectory_time_range(file_path):
    """获取单个轨迹文件的时间起止范围，文件不存在返回None"""
    t, _, _ = read_tum_trajectory(file_path)
    if t is None or len(t) < 2:
        return None
    return t[0], t[-1]

def get_angular_velocity_error_on_unified_t(gt_file, est_file, t_unified):
    """
    计算角速度误差，并插值到统一时间轴上
    输入：真值文件、估计文件、统一时间戳数组
    输出：插值到统一时间轴的误差数组（est - gt），失败返回None
    """
    # 读取轨迹
    t_gt, _, q_gt = read_tum_trajectory(gt_file)
    t_est, _, q_est = read_tum_trajectory(est_file)
    if t_gt is None or t_est is None:
        return None
    
    # 裁剪到统一时间范围（避免外插）
    t_min, t_max = t_unified[0], t_unified[-1]
    mask_gt = (t_gt >= t_min) & (t_gt <= t_max)
    mask_est = (t_est >= t_min) & (t_est <= t_max)
    if np.sum(mask_gt) < 2 or np.sum(mask_est) < 2:
        return None
    
    t_gt_crop, q_gt_crop = t_gt[mask_gt], q_gt[mask_gt]
    t_est_crop, q_est_crop = t_est[mask_est], q_est[mask_est]
    
    # 插值估计四元数到真值裁剪后的时间轴
    q_est_interp = interpolate_quaternions(t_est_crop, q_est_crop, t_gt_crop)
    
    # 计算角速度
    t_mid_gt, v_gt = compute_angular_velocity(t_gt_crop, q_gt_crop)
    t_mid_est, v_est = compute_angular_velocity(t_gt_crop, q_est_interp)
    
    # 对齐角速度的时间轴
    min_len = min(len(v_gt), len(v_est))
    t_mid = t_mid_gt[:min_len]
    v_gt = v_gt[:min_len]
    v_est = v_est[:min_len]
    
    # 计算原始误差
    error_raw = v_est - v_gt
    
    # 将误差插值到全局统一时间轴
    if len(t_mid) < 2:
        return None
    interp_func = interp1d(
        t_mid, error_raw, kind='linear', 
        bounds_error=False, fill_value=(error_raw[0], error_raw[-1])
    )
    error_unified = interp_func(t_unified)
    
    return error_unified

# ---------------------- 主函数：按序列统一时间轴绘图 ----------------------
def main():
    # ====================== 字体与样式配置 ======================
    # 设置全局字体为 Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    # 解决负号显示问题（可选，视系统而定，通常 Times New Roman 支持良好）
    plt.rcParams['axes.unicode_minus'] = False 

    # 基础配置
    gt_base_path = "./bell412_gt/bell412_{}_gt.txt"
    sequences = ['3', '4', '5']  # 待处理序列
    
    # 算法配置：(算法名, 路径模板, 颜色, 线型) 固定配色/线型，保证全图统一
    alg_configs = [
        ("ORB-SLAM3", "./orbslam3/orbslam_{}{}.txt", "#2ca02c", "-"),    # 绿色实线
        ("DPVO",      "./dpvo/dpvo_{}{}.txt",       "#1f77b4", "--"),   # 蓝色虚线
        ("ROVO",      "./rovo/rovo_{}{}.txt",       "#ff7f0e", "-."),   # 橙色点划线
    ]
    
    # 分组配置：(分组名, 文件后缀)
    group_configs = [
        ("Bell412", ""),
        ("3x Scaled Bell412", "_scale")
    ]
    
    # 预存储每个序列、每个分组的统一时间轴 + 所有算法误差数据
    # 结构: seq_data[seq][group_name] = {"t_unified": ..., "error_data": {...}}
    seq_data = {}

    # ====================== 第一步：按序列和分组分别计算统一时间轴 ======================
    for seq in sequences:
        print(f"\n===== 处理序列 {seq} =====")
        seq_data[seq] = {} # 初始化当前序列的数据容器
        
        gt_file = gt_base_path.format(seq)
        
        # 遍历每个分组，独立计算时间轴和误差
        for group_name, group_suffix in group_configs:
            print(f"  --- 处理分组: {group_name} ---")
            
            # 1. 收集该序列、该分组所有相关文件路径
            seq_group_files = []
            # 加入真值文件
            seq_group_files.append(gt_file)
            # 加入当前分组下所有算法的文件
            for _, alg_path_template, _, _ in alg_configs:
                est_file = alg_path_template.format(seq, group_suffix)
                # 兼容dpvo命名问题
                seq_group_files.append(est_file)
            
            # 2. 读取所有有效文件的时间范围
            time_ranges = []
            for file in seq_group_files:
                tr = get_trajectory_time_range(file)
                if tr is not None:
                    time_ranges.append(tr)
                    # print(f"    有效文件: {os.path.basename(file)} | 时间范围: {tr[0]:.2f} ~ {tr[1]:.2f} s")
                else:
                    print(f"    无效文件: {os.path.basename(file)} | 跳过")
            
            if len(time_ranges) == 0:
                print(f"  序列 {seq} - 分组 {group_name} 无有效文件，跳过")
                continue
            
            # 3. 计算该分组全局统一时间范围
            t_unified_start = max([tr[0] for tr in time_ranges])
            t_unified_end = min([tr[1] for tr in time_ranges])
            
            if t_unified_start >= t_unified_end:
                print(f"  序列 {seq} - 分组 {group_name} 无公共时间区间，跳过")
                continue
            
            print(f"  序列 {seq} - 分组 {group_name} 统一时间范围: {t_unified_start:.2f} ~ {t_unified_end:.2f} s")
            
            # 4. 生成统一时间轴：以真值的时间戳为基准，裁剪到统一时间范围
            t_gt_full, _, _ = read_tum_trajectory(gt_file)
            if t_gt_full is None:
                print(f"  序列 {seq} - 分组 {group_name} 真值文件读取失败，跳过")
                continue
                
            mask_unified = (t_gt_full >= t_unified_start) & (t_gt_full <= t_unified_end)
            t_unified = t_gt_full[mask_unified]
            
            if len(t_unified) < 2:
                print(f"  序列 {seq} - 分组 {group_name} 统一时间轴点数不足，跳过")
                continue
            print(f"  序列 {seq} - 分组 {group_name} 统一时间戳点数: {len(t_unified)}")
            
            # 5. 计算该分组下所有算法在统一时间轴上的误差
            group_error_data = {}
            for alg_name, alg_path_template, color, linestyle in alg_configs:
                est_file = alg_path_template.format(seq, group_suffix)
                
                error = get_angular_velocity_error_on_unified_t(gt_file, est_file, t_unified)
                if error is not None:
                    group_error_data[alg_name] = error
                    print(f"    成功计算: {alg_name}")
                else:
                    print(f"    计算失败: {alg_name} | 跳过")
            
            # 保存该序列、该分组的数据
            seq_data[seq][group_name] = {
                "t_unified": t_unified,
                "error_data": group_error_data
            }

    # ====================== 第二步：统一绘图 ======================
    if len(seq_data) == 0:
        print("无有效序列数据，无法绘图")
        return
    
    # 创建画布：2行（分组） x 3列（序列）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=100)
    
    # 遍历分组和序列绘图
    for row_idx, (group_name, _) in enumerate(group_configs):
        for col_idx, seq in enumerate(sequences):
            ax = axes[row_idx, col_idx]
            
            # 检查该序列该分组是否有数据
            if seq not in seq_data or group_name not in seq_data[seq]:
                ax.text(0.5, 0.5, "No Valid Data", ha='center', va='center', transform=ax.transAxes, fontname='Times New Roman')
                ax.set_title(f"{group_name}_{seq}", fontname='Times New Roman', fontweight='bold')
                ax.grid(True, alpha=0.3)
                continue
            
            # 获取该分组特有的统一时间轴和误差数据
            t_unified = seq_data[seq][group_name]["t_unified"]
            error_data = seq_data[seq][group_name]["error_data"]
            
            # 用于存放当前子图的图例句柄和标签
            current_legend_handles = []
            current_legend_labels = []
            
            # 绘制每个算法的误差曲线
            for alg_name, _, color, linestyle in alg_configs:
                if alg_name not in error_data:
                    # 如果数据不存在，添加一个空的占位符到图例中显示 failed
                    fake_line = mlines.Line2D([], [], color='gray', linestyle='-', label=f"{alg_name}: failed")
                    current_legend_handles.append(fake_line)
                    current_legend_labels.append(f"{alg_name}: failed")
                    continue
                
                error = error_data[alg_name]
                line, = ax.plot(
                    t_unified, error, 
                    color=color, linestyle=linestyle, 
                    linewidth=1.2, alpha=0.85
                )
                
                # 计算均值
                mean_err = np.mean(np.abs(error))
                
                # 将均值添加到图例标签中
                legend_label = f"{alg_name}: {mean_err:.2f}"
                current_legend_handles.append(line)
                current_legend_labels.append(legend_label)

            # 子图格式设置
            ax.set_title(f"{group_name}_{seq}", fontsize=12, fontweight='bold', fontname='Times New Roman')
            ax.set_xlabel("Time (s)", fontsize=10, fontname='Times New Roman', fontweight='bold')
            ax.set_ylabel("Angular Velocity Error (deg/s)", fontsize=10, fontname='Times New Roman',fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 在每个子图中添加独立图例
            if current_legend_handles:
                leg = ax.legend(
                    current_legend_handles, 
                    current_legend_labels,
                    loc='upper right',
                    fontsize=9,
                    frameon=True,
                    shadow=False,
                    borderpad=0.5,
                    labelspacing=0.3,
                    handlelength=1.5,
                    prop={'family': 'Times New Roman'} # 确保图例字体也是 Times New Roman
                )

    # 全局格式调整
    plt.tight_layout(h_pad=2.0) 

    # 修改点2: 保存高质量PNG图像
    output_filename = "angular_velocity_error_comparison.png"
    plt.savefig(
        output_filename, 
        dpi=300,          # 高分辨率 (300 DPI 是出版级标准，也可设为 600)
        bbox_inches='tight', # 自动调整边界，防止标签被切掉
        pad_inches=0.5    # 增加一点内边距
    )
    print(f"图像已保存至: {output_filename}")

    plt.show()

if __name__ == "__main__":
    main()