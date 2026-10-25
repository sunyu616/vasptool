import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==========================================
# 1. 顶刊绘图风格 (Nature/Science/Cell)
# ==========================================
def apply_publication_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'lines.linewidth': 2.5,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.frameon': False,
        'figure.dpi': 150
    })

apply_publication_style()

# ==========================================
# 2. 物理常数与数据
# ==========================================
# 物理常数 (SI单位)
F = 96485.332   # 法拉第常数 C/mol
R = 8.314       # 气体常数 J/(mol·K)
T = 298.15      # 温度 K
k_B_eV = 8.617e-5 # 玻尔兹曼常数 eV/K (用于计算 j0)

# 动力学参数
alpha = 0.5     # 传递系数 (对称因子)
# j0_factor: 用于调节整体电流数量级，模拟活性位点密度
# 调整此值以确保曲线在 -0.5V 内能达到 -10 mA/cm2
j0_scaling_factor = 2.0 

# 经典配色
colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']

# VASP 数据: Delta G_H* (eV)
my_data = {
    '1': -0.153,
    '2': 0.123,
    '3': 0.001,
    '4': 0.032
}

# ==========================================
# 3. 核心计算 (Butler-Volmer)
# ==========================================
def calculate_bv_current(delta_G, potential_array):
    """
    使用 Butler-Volmer 方程计算电流
    j = j0 * ( exp(-alpha*f*eta) - exp((1-alpha)*f*eta) )
    """
    # 1. 计算交换电流密度 j0 (基于火山图原理)
    # j0 与 exp(-|Delta G| / kT) 成正比
    # 注意: Delta G 是 eV, 所以分母用 k_B_eV * T
    j0 = j0_scaling_factor * np.exp( -np.abs(delta_G) / (k_B_eV * T) )
    
    # 2. 定义过电势 eta
    # 在 LSV 扫描中 (0 -> -0.5V), 阴极过电势通常取绝对值计算
    eta = np.abs(potential_array)
    
    # 3. Butler-Volmer 公式 (用户指定形式)
    # 第一项 exp(-...) 在 eta 增大时趋向 0
    # 第二项 exp((1-a)...) 在 eta 增大时趋向无穷大
    # 结果为负值，符合还原电流定义
    term_constant = F * eta / (R * T)
    
    term1 = np.exp(-alpha * term_constant)
    term2 = np.exp((1 - alpha) * term_constant)
    
    j = j0 * (term1 - term2)
    
    return j

# ==========================================
# 4. 绘图与导出逻辑
# ==========================================
# 定义严格的坐标范围
x_min, x_max = -0.5, 0.0
y_min, y_max = -10.0, 0.0

# 生成数据点
potential_axis = np.linspace(x_min, x_max, 1000)

fig, ax = plt.subplots(figsize=(6, 5))

# 导出数据容器
export_header = "Potential(V)"
export_data = [potential_axis]

for i, (name, dG) in enumerate(my_data.items()):
    # 计算
    j_raw = calculate_bv_current(dG, potential_axis)
    
    # 保存原始数据
    export_header += f"\t{name}"
    export_data.append(j_raw)
    
    # 截断处理 (Masking)
    # 小于 -10 的部分不画 (NaN)
    j_plot = np.where(j_raw < y_min, np.nan, j_raw)
    # 大于 0 的部分不画 (NaN)
    j_plot = np.where(j_plot > y_max, np.nan, j_plot)
    
    # 绘图
    ax.plot(potential_axis, j_plot, label=name, color=colors[i % len(colors)])

# ==========================================
# 5. 样式精细调整 (Nature Style)
# ==========================================

# 1. 严格锁定坐标轴 (Zero Gap)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.margins(0) # 移除所有留白

# 2. 标签
ax.set_xlabel('Potential (V vs. RHE)', fontweight='bold')
ax.set_ylabel('Current density (mA cm$^{-2}$)', fontweight='bold')

# 3. 刻度控制
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1)) # X轴步长 0.1
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))   # Y轴步长 2

# 4. 图例位置
ax.legend(loc='lower left', fontsize=11)

plt.tight_layout()
plt.show()

# ==========================================
# 6. 数据保存
# ==========================================
try:
    save_data = np.column_stack(export_data)
    np.savetxt('lsv.dat', save_data, delimiter='\t', header=export_header, comments='', fmt='%.5f')
    print("成功: 已生成符合 Butler-Volmer 模型的 lsv.dat 文件")
except Exception as e:
    print(f"保存失败: {e}")




