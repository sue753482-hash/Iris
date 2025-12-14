from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# 只添加这两行解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("任务三：可视化3D Probability Map")
print("两分类 / 三个特征")

# 加载数据
iris = load_iris()
X = iris.data[:100, :3]
y = iris.target[:100]

print(f"特征: 花萼长度, 花萼宽度, 花瓣长度")

# 训练模型
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 创建网格（从负值开始）
x_min, x_max = X[:, 0].min()-1.5, X[:, 0].max()+1.5
y_min, y_max = X[:, 1].min()-1.5, X[:, 1].max()+1.5

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                     np.linspace(y_min, y_max, 30))

# 计算决策曲面（找到概率=0.5的z值）
z_surface = np.zeros_like(xx)

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        z_samples = np.linspace(0, 8, 100)  # 花瓣长度范围
        test_points = np.column_stack([
            np.full(100, xx[i, j]),
            np.full(100, yy[i, j]),
            z_samples
        ])
        probs = model.predict_proba(test_points)[:, 1]
        idx = np.argmin(np.abs(probs - 0.5))
        z_surface[i, j] = z_samples[idx]

# 创建图形
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# ================= 1. 3D曲面：只画网格线框架（wireframe） =================
wire = ax.plot_wireframe(xx, yy, z_surface, 
                        color='black',      # 网格线颜色
                        linewidth=0.8,      # 线宽
                        alpha=0.7,          # 透明度
                        rstride=2,          # 减少网格密度
                        cstride=2)          # 让图更清晰

# ================= 2. 三个平面的投影：上色显示概率 =================

# (1) XY平面投影（底部）：显示概率分布
z_proj = np.full_like(xx, 0)  # Z=0平面
# 计算XY平面上的概率（固定Z=中间值）
z_fixed = 4.0  # 固定的花瓣长度
xy_points = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, z_fixed)]
xy_probs = model.predict_proba(xy_points)[:, 1].reshape(xx.shape)

# 在XY平面上色
xy_proj = ax.contourf(xx, yy, xy_probs, 
                     zdir='z', offset=0,     # 投影到Z=0平面
                     levels=20, cmap='RdYlBu', alpha=0.6)

# (2) XZ平面投影（左侧）
y_fixed = 3.0  # 固定的花萼宽度
xz_points = np.c_[xx.ravel(), np.full(xx.ravel().shape, y_fixed), 
                  z_surface.ravel()]
xz_probs = model.predict_proba(xz_points)[:, 1].reshape(xx.shape)

# 在XZ平面上色
xz_proj = ax.contourf(xx, xy_probs, z_surface,  # 注意参数顺序
                     zdir='y', offset=y_min,    # 投影到Y=y_min平面
                     levels=20, cmap='RdYlBu', alpha=0.6)

# (3) YZ平面投影（后侧）
x_fixed = 5.0  # 固定的花萼长度
yz_points = np.c_[np.full(yy.ravel().shape, x_fixed), 
                  yy.ravel(), z_surface.ravel()]
yz_probs = model.predict_proba(yz_points)[:, 1].reshape(yy.shape)

# 在YZ平面上色
yz_proj = ax.contourf(yz_probs, yy, z_surface,  # 注意参数顺序
                     zdir='x', offset=x_min,    # 投影到X=x_min平面
                     levels=20, cmap='RdYlBu', alpha=0.6)

# ================= 设置坐标轴和视角 =================
ax.set_xlabel('sepal_length (cm)', labelpad=10)
ax.set_ylabel('sepal_width (cm)', labelpad=10)
ax.set_zlabel('petal_length (cm)', labelpad=10)
ax.set_title('3D Probability Map', pad=20)

# 设置坐标范围
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_zlim([0, 8])

# 设置视角
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig('task3_result.png', dpi=200, bbox_inches='tight')
plt.show()

print("任务三完成：3D概率图可视化")