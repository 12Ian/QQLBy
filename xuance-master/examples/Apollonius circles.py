import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 初始参数设定
d = 10.0  # 追击者 P 和逃逸者 E 之间的初始距离
initial_lambda = 1.5  # 初始速度比 lambda (vp / ve)

# 创建画布和子图
fig, ax = plt.subplots(figsize=(9, 8))
plt.subplots_adjust(bottom=0.25)  # 给底部的滑块留出空间

# 绘制逃逸者 E (原点) 和追击者 P
E_point, = ax.plot(0, 0, 'ro', markersize=8, label='Evader (E)')
P_point, = ax.plot(d, 0, 'bo', markersize=8, label='Pursuer (P)')

# 初始化阿波罗尼乌斯圆和中垂线 (用于 lambda = 1 的特殊情况)
circle_patch = plt.Circle((0, 0), 1, alpha=0.3, label="Apollonius Boundary")
ax.add_patch(circle_patch)
line_patch = ax.axvline(x=d / 2, color='green', linestyle='--', alpha=0.5, visible=False)

# 设置坐标轴范围和网格
ax.set_xlim(-15, 25)
ax.set_ylim(-20, 20)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_title("Interactive Apollonius Circle in Pursuit-Evasion")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.legend(loc="upper right")

# 添加文本框用于实时显示状态
text_info = ax.text(-14, 16, '', fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

# 创建交互式滑块
ax_lambda = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
s_lambda = Slider(ax_lambda, 'Speed Ratio\n(vp/ve)', 0.2, 3.0, valinit=initial_lambda, valstep=0.05)


# 滑块更新函数
def update(val):
    lam = s_lambda.val

    # 临界情况：速度相等
    if abs(lam - 1.0) < 1e-5:
        circle_patch.set_visible(False)
        line_patch.set_visible(True)
        text_info.set_text("Speed Ratio = 1.0\nBoundary is a straight line.\nEqual reachability for P and E.")
    else:
        circle_patch.set_visible(True)
        line_patch.set_visible(False)

        # 计算圆心坐标 (xc, 0) 和半径 R
        xc = d / (1 - lam ** 2)
        R = abs(lam * d / (1 - lam ** 2))

        circle_patch.set_center((xc, 0))
        circle_patch.set_radius(R)

        # 根据 lambda 的大小改变圆的颜色和物理意义
        if lam > 1:
            # 追击者更快：圆包围逃逸者
            circle_patch.set_facecolor('red')
            text_info.set_text(f"Speed Ratio = {lam:.2f} (P is faster)\n"
                               f"Circle encloses E (Red Zone).\n"
                               f"E can only safely reach inside.\n"
                               f"P can intercept E anywhere OUTSIDE.")
        else:
            # 逃逸者更快：圆包围追击者
            circle_patch.set_facecolor('blue')
            text_info.set_text(f"Speed Ratio = {lam:.2f} (E is faster)\n"
                               f"Circle encloses P (Blue Zone).\n"
                               f"P can ONLY intercept E INSIDE.\n"
                               f"E is completely safe OUTSIDE.")

    fig.canvas.draw_idle()


# 初始化调用一次以绘制初始状态
update(initial_lambda)

# 绑定滑块事件并展示
s_lambda.on_changed(update)
plt.show()