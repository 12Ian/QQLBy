import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ================= 物理参数设置 =================
GRID_SIZE = 150  # 网格分辨率，越大图像越细腻但计算越慢
SPACE_LIMIT = 10.0  # 空间边界 [-10, 10]

# 速度设置 (你可以修改这里测试不同的 lambda)
V_P = 1.0  # 追击者速度 (Pursuer)
V_E = 0.9  # 逃逸者速度 (Evader) - 这里设为逃逸者略慢，确保有限空间内必定收敛

# 初始位置：3架追击机形成等边三角形包围网，逃逸者在中间偏上的位置
P_pos = np.array([
    [-6.0, -5.0],  # 追击者 1
    [6.0, -5.0],  # 追击者 2
    [0.0, 7.0]  # 追击者 3
])
E_pos = np.array([[0.0, 2.0]])  # 逃逸者

DT = 0.1  # 仿真步长

# ================= 初始化绘图环境 =================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-SPACE_LIMIT, SPACE_LIMIT)
ax.set_ylim(-SPACE_LIMIT, SPACE_LIMIT)
ax.set_aspect('equal')
ax.set_title("Dynamic Voronoi Enclosure in Pursuit-Evasion", fontsize=14)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

# 生成空间网格
x = np.linspace(-SPACE_LIMIT, SPACE_LIMIT, GRID_SIZE)
y = np.linspace(-SPACE_LIMIT, SPACE_LIMIT, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# 图像句柄
contour_plot = None
p_scat = ax.scatter(P_pos[:, 0], P_pos[:, 1], c='blue', s=100, label='Pursuers (P)', zorder=5)
e_scat = ax.scatter(E_pos[:, 0], E_pos[:, 1], c='red', s=100, label='Evader (E)', zorder=5)
text_area = ax.text(-9, 8.5, '', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
ax.legend(loc="upper right")


# ================= 核心演化逻辑 =================
def update(frame):
    global P_pos, E_pos, contour_plot

    # --- 1. 策略计算 (Movement Logic) ---
    # 逃逸者策略：根据人工势场法，背离所有追击者逃跑
    escape_vec = np.zeros(2)
    for p in P_pos:
        vec = E_pos[0] - p
        dist = np.linalg.norm(vec)
        if dist > 0:
            # 距离越近，排斥力越强 (平方反比)
            escape_vec += (vec / dist) * (1.0 / (dist ** 2))

            # 归一化并更新逃逸者位置
    if np.linalg.norm(escape_vec) > 0:
        escape_vec = escape_vec / np.linalg.norm(escape_vec)
    E_pos[0] += escape_vec * V_E * DT

    # 追击者策略：纯追踪 (Pure Pursuit)，直接飞向逃逸者
    for i in range(3):
        pursuit_vec = E_pos[0] - P_pos[i]
        dist = np.linalg.norm(pursuit_vec)
        if dist > 0:
            # 归一化并更新追击者位置
            P_pos[i] += (pursuit_vec / dist) * V_P * DT

    # 边界限制 (防止跑出画框)
    E_pos = np.clip(E_pos, -SPACE_LIMIT, SPACE_LIMIT)
    P_pos = np.clip(P_pos, -SPACE_LIMIT, SPACE_LIMIT)

    # --- 2. 实时计算 Voronoi 绝对安全区 ---
    # 计算网格上每个点到逃逸者的到达时间
    T_E = np.sqrt((X - E_pos[0, 0]) ** 2 + (Y - E_pos[0, 1]) ** 2) / V_E

    # 计算网格上每个点到最近追击者的到达时间
    T_P_min = np.full_like(T_E, np.inf)
    for p in P_pos:
        T_P = np.sqrt((X - p[0]) ** 2 + (Y - p[1]) ** 2) / V_P
        T_P_min = np.minimum(T_P_min, T_P)

    # 判定安全区：逃逸者到达时间 < 最近追击者到达时间
    # 结果是一个布尔矩阵，True 代表属于逃逸者的 Voronoi 细胞
    safe_zone = T_E < T_P_min

    # 计算安全区面积占比
    safe_area_ratio = np.sum(safe_zone) / (GRID_SIZE * GRID_SIZE) * 100

    # 清除旧的安全区色块
    if contour_plot is not None:
        if hasattr(contour_plot, 'collections'):
            # 兼容旧版本 Matplotlib (<3.8)
            for c in contour_plot.collections:
                c.remove()
        else:
            # 兼容新版本 Matplotlib (>=3.8)
            contour_plot.remove()

    # 绘制新的安全区 (红色半透明区域)
    contour_plot = ax.contourf(X, Y, safe_zone, levels=[0.5, 1.5], colors=['#ff9999'], alpha=0.5)

    # 更新散点图位置
    p_scat.set_offsets(P_pos)
    e_scat.set_offsets(E_pos)

    # 更新文字信息
    text_area.set_text(f"Time Step: {frame}\nEvader Safe Area: {safe_area_ratio:.1f}%")

    # 当安全区被彻底压扁时，停止动画
    if safe_area_ratio < 0.1:
        text_area.set_text("CAPTURE COMPLETE!\nSafe Area: 0.0%")
        ani.event_source.stop()

    return p_scat, e_scat, text_area


# 创建动画 (50 毫秒刷新一次)
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)

plt.show()