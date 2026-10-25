import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==============================================================================
# 配置区域
# ==============================================================================
# 文件夹命名格式 (支持 0.95, 0.96... 1.00 ... 1.05)
FOLDER_PATTERN = re.compile(r"^(\d+\.\d+)$")
# 基准文件夹名字 (脚本会去这个文件夹读取原始a,b,c)
REF_FOLDER_NAMES = ["1.00", "1.0"] 
# ==============================================================================

def get_vasp_data(folder):
    """从文件夹中提取体积和能量，若是基准文件夹则额外提取晶格常数"""
    outcar = os.path.join(folder, "OUTCAR")
    oszicar = os.path.join(folder, "OSZICAR")
    
    if not (os.path.exists(outcar) and os.path.exists(oszicar)):
        return None

    vol = None
    ene = None
    lattice_abc = None

    # 1. 读取 OUTCAR (体积 & 晶格常数)
    with open(outcar, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # 提取体积 (取最后一个出现的)
            if "volume of cell :" in line:
                vol = float(line.split()[4])
            
            # 提取晶格常数 (找 "length of vectors")
            # 格式: length of vectors
            #       3.183000000    3.183000000   12.500000000
            if "length of vectors" in line:
                try:
                    vec_line = lines[i+1].split()
                    lattice_abc = [float(x) for x in vec_line[:3]]
                except:
                    pass

    # 2. 读取 OSZICAR (能量)
    # 提取 E0 (energy for sigma->0)
    try:
        # 读取最后一行包含 E0= 的数据
        with open(oszicar, 'r') as f:
            content = f.read().strip().split('\n')
            for line in reversed(content):
                if "E0=" in line:
                    # line format example: ... E0= -.123456E+02 ...
                    # split 找 E0= 后面的数
                    parts = line.split("E0=")[1].split()
                    ene = float(parts[0])
                    break
    except:
        pass

    if vol is not None and ene is not None:
        return {"vol": vol, "ene": ene, "abc": lattice_abc}
    return None

def birch_murnaghan(V, E0, V0, B0, B0_prime):
    """三阶 Birch-Murnaghan 状态方程"""
    eta = (V0 / V) ** (2 / 3)
    return E0 + 9 * V0 * B0 / 16 * (
        (eta - 1) ** 3 * B0_prime + 
        (eta - 1) ** 2 * (6 - 4 * eta)
    )

def main():
    print("--- 开始处理 VASP EOS 数据 ---")
    
    data_points = []
    ref_data = None
    ref_folder_found = None

    # 1. 遍历文件夹
    current_dirs = sorted([d for d in os.listdir('.') if os.path.isdir(d)])
    
    for folder in current_dirs:
        # 检查是否是数字命名的文件夹
        if FOLDER_PATTERN.match(folder):
            res = get_vasp_data(folder)
            if res:
                print(f"读取文件夹 {folder}: V={res['vol']:.3f}, E={res['ene']:.5f}")
                data_points.append([res['vol'], res['ene']])
                
                # 检查是否是基准文件夹 (1.00)
                if folder in REF_FOLDER_NAMES:
                    ref_data = res
                    ref_folder_found = folder
            else:
                print(f"跳过文件夹 {folder}: 数据不完整")

    if not data_points:
        print("错误：未提取到有效数据。")
        return

    if ref_data is None:
        print("警告：未找到 '1.00' 或 '1.0' 文件夹作为基准。无法计算精确晶格常数，只能输出体积。")
    else:
        print(f"\n基准参考 ({ref_folder_found}):")
        print(f"  Ref Vol: {ref_data['vol']:.4f}")
        print(f"  Ref a,b,c: {ref_data['abc']}")

    # 2. 拟合 EOS
    data = np.array(data_points)
    # 按体积排序
    data = data[data[:, 0].argsort()]
    volumes = data[:, 0]
    energies = data[:, 1]

    # 初始猜测
    p0 = [
        np.min(energies),          # E0
        np.mean(volumes),          # V0
        160.0 / 160.2,             # B0 (approx)
        4.0                        # B0'
    ]

    try:
        popt, pcov = curve_fit(birch_murnaghan, volumes, energies, p0=p0, maxfev=5000)
        E0_fit, V0_fit, B0_fit, B0_prime_fit = popt
        B0_GPa = B0_fit * 160.21766
    except Exception as e:
        print(f"拟合失败: {e}")
        return

    # 3. 计算最终晶格常数
    print("\n" + "="*40)
    print("       拟合结果 (Results)")
    print("="*40)
    print(f"平衡体积 V0: {V0_fit:.6f} A^3")
    print(f"平衡能量 E0: {E0_fit:.6f} eV")
    print(f"体模量   B0: {B0_GPa:.3f} GPa")
    
    if ref_data and ref_data['abc']:
        ref_v = ref_data['vol']
        ref_abc = ref_data['abc']
        
        # 计算缩放比例 f = (V0 / V_ref)^(1/3)
        scale_factor = (V0_fit / ref_v) ** (1/3)
        
        final_abc = [x * scale_factor for x in ref_abc]
        
        print("-" * 40)
        print("★ 能量最低点晶胞参数 (Final Lattice Constants):")
        print(f"   a = {final_abc[0]:.6f} A")
        print(f"   b = {final_abc[1]:.6f} A")
        print(f"   c = {final_abc[2]:.6f} A")
        print(f"   (相对于 1.00 结构的缩放系数: {scale_factor:.6f})")
        print("-" * 40)
    else:
        print("\n未能计算晶格常数 (缺少 1.00 文件夹数据)")

    # 4. 绘图
    plt.figure(figsize=(8, 6))
    plt.scatter(volumes, energies, c='r', label='VASP Data')
    
    v_smooth = np.linspace(min(volumes)*0.99, max(volumes)*1.01, 200)
    e_smooth = birch_murnaghan(v_smooth, *popt)
    
    # 标记最低点
    plt.plot(v_smooth, e_smooth, label='Birch-Murnaghan Fit')
    plt.axvline(V0_fit, color='g', linestyle='--', alpha=0.5, label=f'V0={V0_fit:.2f}')
    
    plt.xlabel(r'Volume ($\AA^3$)')
    plt.ylabel('Energy (eV)')
    plt.title(f'EOS Fit: V0={V0_fit:.3f}, B0={B0_GPa:.1f} GPa')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('EOS_Fit.png', dpi=300)
    print("拟合曲线已保存为: EOS_Fit.png")

if __name__ == "__main__":
    main()
