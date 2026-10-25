import os
import subprocess
import shutil

# ================= 配置区域 =================

# 1. 结构设置
BOX_SIZE = 20.0       # 真空盒子大小 (Å)

# 2. 提交脚本内容 (完全按照您的要求)
PBS_CONTENT = """#!/bin/bash
#PBS -N sy
#PBS -l nodes=master:ppn=24
#PBS -j n
#PBS -e ${PBS_JOBNAME}.err
#PBS -o ${PBS_JOBNAME}.out
#PBS -q batch

# --- 公共设置 ---
ulimit -s unlimited
export OMP_NUM_THREADS=1

# --- 自动判断服务器环境 ---
# 定义两个服务器上不同的 VASP 路径
VASP_PATH_1="/public/apps/vasp/vasp_std"
VASP_PATH_2="/public/apps/vasp/bin/vasp_std"

if [ -f "$VASP_PATH_1" ]; then
    # === 情况 1: 对应你的第一个脚本 ===
    EXEC=$VASP_PATH_1
    # 如果第一个服务器不需要特殊环境变量，这里留空即可

elif [ -f "$VASP_PATH_2" ]; then
    # === 情况 2: 对应你的第二个脚本 ===
    EXEC=$VASP_PATH_2
    # 第二个脚本特有的环境变量
    export I_MPI_ADJUST_REDUCE=3

else
    # === 错误处理 ===
    echo "Error: Cannot find VASP executable on this server."
    exit 1
fi

# --- 执行计算 ---
cd $PBS_O_WORKDIR
NP=`cat $PBS_NODEFILE|wc -l`

# 打印一下当前使用的程序路径，方便在 output 文件中排查问题
echo "Running on host: $(hostname)"
echo "Using VASP executable: $EXEC"
echo "Total processes (NP): $NP"

mpirun -np $NP $EXEC > ${PBS_JOBNAME}.out
"""

# 3. 稳健型 INCAR (解决孤立分子不收敛问题)
INCAR_ROBUST = """SYSTEM = Molecule Opt
ISTART = 0
ICHARG = 2
NPAR   = 6

# --- 电子步收敛 (关键修复) ---
ENCUT  = 450
NELM   = 200     # 增加步数
EDIFF  = 1E-4
ISYM   = 0       # 关闭对称性
ALGO   = All
PREC   =A     

# --- 混合参数 (防止电荷震荡) ---
AMIX   = 0.1
BMIX   = 0.0001
AMIX_MAG = 0.1
BMIX_MAG = 0.0001

# --- Smearing ---
ISMEAR = 0
SIGMA  = 0.05

# --- 离子弛豫 ---
IBRION = 2
NSW    = 200
EDIFFG = -0.05
ISIF   = 0       # 固定晶胞

# --- 输出 ---
LREAL  = .FALSE.
LCHARG = .FALSE.
LWAVE  = .FALSE.
"""

KPOINTS_CONTENT = """Gamma Point Only
0
Gamma
1  1  1
0  0  0
"""

# ================= 功能函数 =================

def parse_mol_v2000(filepath):
    """解析 .mol 文件获取原子坐标"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    atoms = []
    try:
        counts_line = lines[3].strip().split()
        num_atoms = int(counts_line[0])
        for i in range(num_atoms):
            line = lines[4 + i]
            parts = line.split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            sym = parts[3]
            atoms.append({'sym': sym, 'x': x, 'y': y, 'z': z})
    except Exception as e:
        print(f"❌ 解析错误 {filepath}: {e}")
        return None
    return atoms

def get_center(atoms):
    xs = [a['x'] for a in atoms]
    ys = [a['y'] for a in atoms]
    zs = [a['z'] for a in atoms]
    return (min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2

def write_poscar(atoms, folder_path, title):
    """生成 POSCAR"""
    atoms.sort(key=lambda x: x['sym']) # 按元素排序
    cx, cy, cz = get_center(atoms)
    
    # 移动向量 (从几何中心移到盒子中心)
    shift_x = (BOX_SIZE / 2.0) - cx
    shift_y = (BOX_SIZE / 2.0) - cy
    shift_z = (BOX_SIZE / 2.0) - cz

    # 统计元素
    element_counts = {}
    elements_order = []
    current_sym = None
    for atom in atoms:
        sym = atom['sym']
        if sym != current_sym:
            elements_order.append(sym)
            current_sym = sym
            element_counts[sym] = 0
        element_counts[sym] += 1

    # 写入文件
    with open(os.path.join(folder_path, "POSCAR"), 'w') as f:
        f.write(f"{title}\n1.0\n")
        f.write(f"   {BOX_SIZE:.6f}   0.000000   0.000000\n")
        f.write(f"   0.000000   {BOX_SIZE:.6f}   0.000000\n")
        f.write(f"   0.000000   0.000000   {BOX_SIZE:.6f}\n")
        f.write("   " + "   ".join(elements_order) + "\n")
        f.write("   " + "   ".join([str(element_counts[e]) for e in elements_order]) + "\n")
        f.write("Cartesian\n")
        for atom in atoms:
            f.write(f"   {atom['x']+shift_x:.8f}   {atom['y']+shift_y:.8f}   {atom['z']+shift_z:.8f}\n")
    return elements_order

def generate_potcar_vaspkit(folder_path):
    """调用 vaspkit 生成 POTCAR"""
    original_cwd = os.getcwd()
    try:
        # 切换到任务文件夹
        os.chdir(folder_path)
        
        # 检查是否安装了 vaspkit
        if shutil.which("vaspkit") is None:
            print(f"⚠️  警告: 未找到 'vaspkit' 命令，跳过 POTCAR 生成。")
            return False

        # 调用 vaspkit 103
        # stdout=subprocess.DEVNULL 可以隐藏 vaspkit 的刷屏输出，如果想看可以去掉
        subprocess.run("vaspkit -task 103", shell=True, check=True, stdout=subprocess.DEVNULL)
        
        if os.path.exists("POTCAR"):
            print(f"   ✅ POTCAR 生成成功 (Vaspkit)")
            return True
        else:
            print(f"   ❌ POTCAR 生成失败 (Vaspkit 运行了但未生成文件)")
            return False

    except subprocess.CalledProcessError:
        print(f"   ❌ Vaspkit 运行出错")
        return False
    finally:
        # 必须切回原目录
        os.chdir(original_cwd)

def main():
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.lower().endswith('.mol')]
    
    if not files:
        print("❌ 当前目录下没有 .mol 文件")
        return

    print(f"🚀 开始处理 {len(files)} 个分子任务...\n")

    for mol_file in files:
        base_name = os.path.splitext(mol_file)[0]
        folder_name = base_name.replace(" ", "_")
        
        # 1. 创建文件夹
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        print(f"📂 正在处理: {folder_name}")

        # 2. 生成 POSCAR
        atoms = parse_mol_v2000(mol_file)
        if not atoms: continue
        write_poscar(atoms, folder_name, base_name)
        
        # 3. 生成 INCAR & KPOINTS
        with open(os.path.join(folder_name, "INCAR"), 'w') as f:
            f.write(INCAR_ROBUST)
        with open(os.path.join(folder_name, "KPOINTS"), 'w') as f:
            f.write(KPOINTS_CONTENT)
            
        # 4. 生成 wk.pbs
        with open(os.path.join(folder_name, "wk.pbs"), 'w') as f:
            f.write(PBS_CONTENT)
            
        # 5. 调用 Vaspkit 生成 POTCAR
        generate_potcar_vaspkit(folder_name)

        print("-" * 50)

    print("\n🎉 所有任务处理完成！")

if __name__ == "__main__":
    main()

