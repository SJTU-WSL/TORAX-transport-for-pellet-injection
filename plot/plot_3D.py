import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def visualize_real_data(data, file_name='output.png'):
    # 1. 你的真实数据
    # data = np.array([
    #     30.04476017, 29.76968922, 28.82062044, 26.89013904, 24.0056778,
    #     20.54030882, 16.9819226, 13.66743719, 11.76182147, 10.16142334,
    #     8.81121413, 7.6535457, 6.6464252, 5.76770589, 5.01092725,
    #     4.36792797, 3.82025964, 3.33836815, 2.89298845, 2.45655087,
    #     2.00688735, 1.53049234, 1.02052512, 0.75682721, 0.3934382
    # ])

    # 2. 重建径向坐标 rho (25个网格中心)
    n_rho = len(data)
    # rho 分布在 0.02, 0.06, ..., 0.98
    rho_norm = (np.arange(n_rho) + 0.5) / n_rho

    # 3. 几何参数 (ITER Hybrid)
    # 如果你有 Torax 的 geometry 对象，这里可以用 geometry.R_major 等替换
    geo_params = {
        'R0': 6.2,  # 大半径 [m]
        'a': 2.0,  # 小半径 [m]
        'kappa': 1.8,  # 拉长比
        'delta': 0.4  # 三角形变
    }

    # 4. 绘图参数
    n_theta = 180  # 极向角度分辨率
    data_label = "Electron Temp. $T_e$ (keV)"

    # --- 数据处理 ---
    # 创建 (Rho, Theta) 网格
    theta_1d = np.linspace(0, 2 * np.pi, n_theta, endpoint=True)
    Rho_grid, Theta_grid = np.meshgrid(rho_norm, theta_1d, indexing='ij')

    # 将 1D 温度扩展为 2D 场 (Z轴高度)
    Data_Z_grid = np.tile(data[:, np.newaxis], (1, n_theta))

    # --- 几何重建 (Miller 公式) ---
    # 计算局部物理半径 r = a * rho
    r_grid = geo_params['a'] * Rho_grid

    # 限制 delta 防止数值误差
    delta_safe = np.clip(geo_params['delta'], -0.99, 0.99)

    # 计算空间坐标 (X=R, Y=Z)
    # X轴: 大半径方向
    R_X_grid = geo_params['R0'] + r_grid * np.cos(Theta_grid + np.arcsin(delta_safe) * np.sin(Theta_grid))

    # Y轴: 垂直方向 (Z)
    Z_Y_grid = geo_params['kappa'] * r_grid * np.sin(Theta_grid)

    # --- 3D 绘图 ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面
    surf = ax.plot_surface(
        R_X_grid, Z_Y_grid, Data_Z_grid,
        cmap=cm.jet,  # 使用类似 MATLAB logo 的彩虹色谱
        linewidth=0,  # 去掉网格线让曲面更平滑
        antialiased=False,
        rstride=1, cstride=1,
        alpha=0.9  # 轻微透明
    )

    # 标签与装饰
    ax.set_xlabel('Major Radius R [m]', fontsize=12, labelpad=10)
    ax.set_ylabel('Vertical Z [m]', fontsize=12, labelpad=10)
    ax.set_zlabel(data_label, fontsize=12, labelpad=10)
    ax.set_title('3D Reconstruction of Plasma Profile', fontsize=16)

    # 调整视角
    ax.view_init(elev=35, azim=-60)

    # 添加 Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.1)
    cbar.set_label(data_label, fontsize=12)

    scaling = np.array([getattr(ax, 'get_xlim')(), getattr(ax, 'get_ylim')(), getattr(ax, 'get_zlim')()])
    ax.set_box_aspect((np.ptp(scaling[0]), np.ptp(scaling[1]), np.ptp(scaling[2]) * 0.1))

    plt.tight_layout()
    plt.savefig('frames/' + file_name + '.png')
    # plt.show()
    # plt.show()


# if __name__ == "__main__":
#     visualize_real_data()
