import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.interpolate import PchipInterpolator
import os
import sys

# 必须依赖 ASE
try:
    from ase.io import read
    from ase.geometry import find_mic
    print("--- 已检测到 ASE 库，准备开始处理 ---")
except ImportError:
    print("[错误] 未检测到 ase 库。请先安装: pip install ase")
    sys.exit(1)

# ================= 配置区域 =================
DATA_FILE = 'neb.dat'
OUTPUT_IMG = 'neb_ase_final_line.png'

STYLE = {
    'line_color': "#3C5488",          # Nature 深蓝
    'marker_color': "#E64B35",        # Nature 朱红
    'marker_edge_color': "white",     # 白边
    'marker_size': 9,
    'line_width': 2.0,
    'font_family': 'DejaVu Sans',
    'padding_x': 0.08,
    'padding_y': 0.15
}
# ===========================================

def get_neb_folders():
    """自动寻找 00, 01, 02... 文件夹"""
    dirs = []
    i = 0
    while True:
        d = f"{i:02d}"
        if os.path.isdir(d):
            if os.path.exists(os.path.join(d, 'OUTCAR')) or os.path.exists(os.path.join(d, 'vasprun.xml')):
                dirs.append(d)
            else:
                if i > 0: 
                    print(f"  [提示] 文件夹 {d} 为空或无输出文件，停止搜索后续目录。")
                    break
            i += 1
        else:
            break
    return dirs

def extract_data_with_ase():
    """使用 ASE 提取数据并生成 neb.dat"""
    dirs = get_neb_folders()
    if not dirs:
        print("[错误] 当前目录下未找到含 OUTCAR 的 00-XX 文件夹。")
        return False

    print(f"  找到计算文件夹: {dirs}")
    print("  正在读取 OUTCAR (这可能需要几秒钟)...")

    images = []
    for d in dirs:
        try:
            # 读取最后一步收敛结构
            atoms = read(os.path.join(d, 'OUTCAR'), format='vasp-out', index=-1)
            images.append(atoms)
            print(f"    [OK] 读取 {d}")
        except Exception as e:
            print(f"    [失败] 读取 {d} 出错: {e}")
            try:
                atoms = read(os.path.join(d, 'vasprun.xml'), index=-1)
                images.append(atoms)
                print(f"    [OK] 读取 {d} (from vasprun.xml)")
            except:
                return False

    energies = []
    for img in images:
        try:
            e = img.get_potential_energy()
            energies.append(e)
        except RuntimeError:
            print("[错误] 无法获取势能。")
            return False
    
    # 能量归零
    base_e = energies[0]
    rel_energies = np.array(energies) - base_e

    # 计算反应坐标 (处理周期性边界)
    coords = [0.0]
    for i in range(len(images) - 1):
        d_vec = images[i+1].get_positions() - images[i].get_positions()
        cell = images[i].get_cell()
        pbc = images[i].get_pbc()
        
        if cell.any() and pbc.any():
            d_vec_mic, _ = find_mic(d_vec, cell, pbc)
            dist = np.linalg.norm(d_vec_mic)
        else:
            dist = np.linalg.norm(d_vec)
        coords.append(coords[-1] + dist)

    with open(DATA_FILE, 'w') as f:
        f.write(f"# NEB Data Extracted using ASE\n")
        f.write(f"# Reaction_Coordinate(A)    Energy(eV)\n")
        for r, e in zip(coords, rel_energies):
            f.write(f"{r:<25.8f} {e:<15.8f}\n")
    
    print(f"  [成功] 数据已保存至 {DATA_FILE}")
    return True

def plot_smart_neb():
    """读取 neb.dat 并绘制带末态辅助线的图"""
    if not os.path.exists(DATA_FILE): return

    # 读取数据
    rc_list, e_list = [], []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            if line.strip().startswith('#') or not line.strip(): continue
            parts = line.split()
            rc_list.append(float(parts[0]))
            e_list.append(float(parts[1]))
    
    rc = np.array(rc_list)
    E = np.array(e_list)
    idx = np.argsort(rc)
    rc = rc[idx]
    E = E[idx]

    # 设置样式
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = [STYLE['font_family'], 'Arial', 'sans-serif']
    mpl.rcParams['mathtext.fontset'] = 'stixsans'
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['xtick.major.size'] = 0
    mpl.rcParams['ytick.major.size'] = 0
    mpl.rcParams['axes.spines.top'] = True
    mpl.rcParams['axes.spines.right'] = True
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['figure.dpi'] = 300

    # 平滑插值
    try:
        interpolator = PchipInterpolator(rc, E)
        x_new = np.linspace(rc[0], rc[-1], 300)
        y_smooth = interpolator(x_new)
    except:
        x_new, y_smooth = rc, E

    # 关键数据点
    E_max_idx = np.argmax(y_smooth)
    E_barrier = y_smooth[E_max_idx]
    max_x = x_new[E_max_idx]
    E_initial = E[0]
    E_final = E[-1]  # 末态能量
    Ea_value = E_barrier - E_initial

    # === 绘图 ===
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # 1. 曲线
    ax.plot(x_new, y_smooth, '-', color=STYLE['line_color'], 
            linewidth=STYLE['line_width'], alpha=0.9, zorder=1)

    # 2. 数据点
    ax.plot(rc, E, 'o', color=STYLE['marker_color'], 
            markeredgecolor=STYLE['marker_edge_color'], 
            markeredgewidth=1.5, markersize=STYLE['marker_size'], zorder=2)

    # ================= 辅助线区域 =================
    
    # 3. 初态辅助线 (y=0)
    # 样式: 灰色, 细线, 点状虚线(:), 半透明
    ax.axhline(E_initial, color='gray', linewidth=0.8, linestyle=':', alpha=0.5, zorder=0)

    # 4. [新增] 末态辅助线 (y=E_final)
    # 只有当初态和末态能量差异明显时(>0.01eV)，才画这条线，避免重合变粗不好看
    if abs(E_final - E_initial) > 0.01:
        ax.axhline(E_final, color='gray', linewidth=0.8, linestyle=':', alpha=0.5, zorder=0)
        
    # ============================================

    # 5. 智能标注 Ea
    if Ea_value > 0.02:
        # 箭头
        ax.annotate('', xy=(max_x, E_initial), xytext=(max_x, E_barrier),
                    arrowprops=dict(arrowstyle='<->', color='#444444', lw=1.2))
        
        # 计算位置
        x_len = rc[-1] - rc[0]
        mid_x = (rc[-1] + rc[0]) / 2
        
        if max_x < mid_x: # 峰在左侧，字在右侧
            text_x = max_x + x_len * 0.05
            ha = 'left'
        else: # 峰在右侧，字在左侧
            text_x = max_x - x_len * 0.05
            ha = 'right'
            
        text_y = (E_barrier + E_initial) / 2
        
        ax.text(text_x, text_y, f'$E_a = {Ea_value:.2f}$ eV',
                verticalalignment='center', horizontalalignment=ha,
                fontsize=10, color='#333333',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))

    # 6. 装饰
    ax.set_xlabel(r"Reaction Coordinate ($\AA$)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("NEB Energy Profile", pad=15, fontweight='bold')

    # 7. 留白
    x_span = rc[-1] - rc[0]
    ax.set_xlim(rc[0] - x_span * STYLE['padding_x'], rc[-1] + x_span * STYLE['padding_x'])
    
    y_range = np.max(y_smooth) - np.min(y_smooth)
    if y_range == 0: y_range = 1
    ax.set_ylim(np.min(y_smooth) - y_range * STYLE['padding_y'], 
                np.max(y_smooth) + y_range * STYLE['padding_y'])

    # 保存
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_IMG.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"--- 处理完成 ---")
    print(f"  图片保存至: {OUTPUT_IMG}")
    
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    if extract_data_with_ase():
        plot_smart_neb()

