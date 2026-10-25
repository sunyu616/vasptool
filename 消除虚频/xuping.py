import sys
import os
import numpy as np

# ================= 默认文件名 =================
POSCAR_FILE = "POSCAR"          # 原胞文件
SPOSCAR_FILE = "SPOSCAR"        # 超胞文件 (或任意一个包含超胞晶格信息的文件)
FORCE_SETS_FILE = "FORCE_SETS"  # 力数据文件
OUTPUT_FILE = "FORCE_CONSTANTS" # 输出文件名
# ============================================

def get_supercell_matrix():
    """
    自动读取 POSCAR 和 SPOSCAR，计算扩胞矩阵 (supercell_matrix)
    """
    print(f"正在自动检测扩胞倍数 ({POSCAR_FILE} vs {SPOSCAR_FILE})...")
    
    try:
        from phonopy.interface.vasp import read_vasp
    except ImportError:
        print("错误: 无法导入 phonopy。请检查环境。")
        sys.exit(1)

    if not os.path.exists(POSCAR_FILE) or not os.path.exists(SPOSCAR_FILE):
        print(f"错误: 找不到 {POSCAR_FILE} 或 {SPOSCAR_FILE}。无法自动计算扩胞倍数。")
        sys.exit(1)

    # 读取结构
    unitcell = read_vasp(POSCAR_FILE)
    supercell = read_vasp(SPOSCAR_FILE)
    
    # 获取晶格矩阵 (行向量形式: [a, b, c])
    P = unitcell.get_cell()
    S = supercell.get_cell()
    
    # 计算变换矩阵 M: S = M * P  =>  M = S * P^-1
    # 注意: Phonopy 中 supercell_matrix 通常定义为基矢量的线性组合系数
    # 这里我们用 numpy 求解线性方程组 S = M.dot(P) => M = S.dot(inv(P))
    try:
        P_inv = np.linalg.inv(P)
        M = np.dot(S, P_inv) # 注意点乘顺序
        
        # 四舍五入为整数
        M_int = np.round(M).astype(int)
        
        # 验证误差
        if not np.allclose(M, M_int, atol=1e-3):
            print("警告: 计算出的扩胞矩阵非整数，可能 POSCAR 和 SPOSCAR 坐标系未对齐。")
            print(f"计算矩阵:\n{M}")
            print("将强制使用四舍五入后的整数矩阵。")
        
        # 检查是否为对角矩阵 (便于显示)
        is_diag = np.count_nonzero(M_int - np.diag(np.diagonal(M_int))) == 0
        if is_diag:
            diag = np.diagonal(M_int)
            print(f"  -> 检测到对角扩胞: {diag[0]} x {diag[1]} x {diag[2]}")
        else:
            print(f"  -> 检测到非对角扩胞矩阵:\n{M_int}")
            
        return M_int

    except Exception as e:
        print(f"计算扩胞矩阵失败: {e}")
        sys.exit(1)

def get_phonopy_instance(supercell_matrix):
    """
    初始化 Phonopy 对象
    """
    try:
        from phonopy import Phonopy
        from phonopy.interface.vasp import read_vasp
        from phonopy.file_IO import parse_FORCE_SETS
    except ImportError:
        sys.exit(1)

    unitcell = read_vasp(POSCAR_FILE)
    
    # 初始化
    # primitive_matrix=np.identity(3) 表示我们假设输入的 POSCAR 就是原胞
    phonon = Phonopy(unitcell,
                     supercell_matrix,
                     primitive_matrix=np.identity(3))

    if not os.path.exists(FORCE_SETS_FILE):
        print(f"错误: 找不到 {FORCE_SETS_FILE}")
        sys.exit(1)

    print(f"正在读取 {FORCE_SETS_FILE}...")
    force_sets = parse_FORCE_SETS(filename=FORCE_SETS_FILE)
    phonon.set_displacement_dataset(force_sets)

    return phonon

def apply_asr_and_symmetry(phonon):
    """
    计算力常数并应用修正 (核心算法)
    """
    print("正在计算原始力常数...")
    phonon.produce_force_constants()
    
    # 获取 Phonopy 内部计算好的 FC
    # 此时 Phonopy 内部可能已经做过简单的对称化
    fc = phonon.get_force_constants()
    
    print("应用额外的晶体对称性修正 (Symmetrization)...")
    try:
        phonon.symmetrize_force_constants()
        fc = phonon.get_force_constants()
    except:
        print("  -> 跳过内部对称化，将在后续步骤处理。")

    print("应用声学支求和规则 (ASR - Translational Invariance)...")
    # 算法：将每一行 (i, a) 的残差 sum_j Phi(i, j, a, b) 平均分配扣除
    # 这种方法比单纯修改对角项更平滑，能保留原本的矩阵结构特征
    
    fc_new = fc.copy()
    n_s = fc.shape[0] # 超胞原子数
    
    for i in range(n_s):
        for a in range(3):      # Cartesian index a
            for b in range(3):  # Cartesian index b
                # 1. 计算这一行的总和 (理想情况应为0)
                total_sum = 0.0
                for j in range(n_s):
                    total_sum += fc[i, j, a, b]
                
                # 2. 计算平均修正量
                correction = total_sum / n_s
                
                # 3. 从每个元素中扣除
                for j in range(n_s):
                    fc_new[i, j, a, b] -= correction

    # Double Check: 强制对角项完全满足 sum = 0
    # 这是消除 Gamma 虚频的最后一道保险
    for i in range(n_s):
        sum_off_diag = np.zeros((3, 3))
        for j in range(n_s):
            if i == j: continue
            sum_off_diag += fc_new[i, j]
        
        # Phi_ii = - Sum_{j!=i} Phi_ij
        fc_new[i, i] = -sum_off_diag

    return fc_new

def write_force_constants(fc, filename):
    """写入 Phonopy 格式"""
    print(f"正在写入修正后的文件: {filename}")
    n_s = fc.shape[0]
    with open(filename, 'w') as f:
        f.write(f"{n_s} {n_s}\n")
        for i in range(n_s):
            for j in range(n_s):
                f.write(f"{i+1} {j+1}\n")
                # 写入 3x3 矩阵
                for row in fc[i, j]:
                    f.write(f" {row[0]:20.15f} {row[1]:20.15f} {row[2]:20.15f}\n")

def main():
    # 1. 自动计算 dim
    matrix = get_supercell_matrix()
    
    # 2. 初始化 Phonopy
    phonon = get_phonopy_instance(matrix)
    
    # 3. 计算并修正
    fc_fixed = apply_asr_and_symmetry(phonon)
    
    # 4. 输出
    write_force_constants(fc_fixed, OUTPUT_FILE)
    
    # 5. 生成对应的运行命令建议
    # 将矩阵转换为 phonopy 的 --dim 参数格式
    if np.count_nonzero(matrix - np.diag(np.diagonal(matrix))) == 0:
        d = np.diagonal(matrix)
        dim_str = f"{d[0]} {d[1]} {d[2]}"
    else:
        # 非对角矩阵展开显示
        dim_str = " ".join(map(str, matrix.flatten()))

    print("\n" + "="*50)
    print("修正完成！")
    print(f"1. 已生成文件: {OUTPUT_FILE}")
    print(f"2. 自动检测到的扩胞参数: --dim='{dim_str}'")
    print("-" * 50)
    print("请使用以下命令绘图 (注意加上 --readfc):")
    print(f"phonopy -p -c {POSCAR_FILE} --dim='{dim_str}' --readfc band.conf")
    print("="*50)

if __name__ == "__main__":
    main()




