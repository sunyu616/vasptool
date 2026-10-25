import os
import subprocess
import shutil

# ================= 配置区域 =================
BOX_SIZE = 15.0       # 真空盒子大小 (Å)，需与分子计算保持一致或足够大

# --- 单原子专用 INCAR ---
# 关键区别：
# 1. ISPIN = 2 (开启自旋，原子通常有磁矩)
# 2. NSW = 0 / IBRION = -1 (单个原子不需要移动，只算电子步)
# 3. ISYM = 0 (破坏对称性，允许电子云自由分布)
INCAR_ATOM = """SYSTEM = Single Atom Energy
ISTART = 0
ICHARG = 2

# --- 电子结构 (必须开启自旋) ---
ISPIN  = 2       # [CRITICAL] 开启自旋极化
ENCUT  = 450     # 必须与分子计算的 ENCUT 保持一致！
NELM   = 100
EDIFF  = 1E-4    # 能量计算需要高精度
ALGO   = Normal  # 稳健算法
ISYM   = 0       # 关闭对称性

# --- 离子步 (不需要优化) ---
IBRION = -1      # 静态计算 (不移动离子)
NSW    = 0       # 0步
ISIF   = 0

# --- 展宽与投影 ---
ISMEAR = 0       # Gaussian
SIGMA  = 0.05
LREAL  = .FALSE. # 倒空间投影，单原子计算最准
PREC   = Accurate

# --- 输出 ---
LCHARG = .FALSE.
LWAVE  = .FALSE.
"""

KPOINTS_CONTENT = """Gamma Point Only
0
Gamma
1  1  1
0  0  0
"""

# 您指定的提交脚本
PBS_CONTENT = """#!/bin/bash
#PBS -N atom_calc
#PBS -l nodes=master:ppn=24
#PBS -j n
#PBS -e ${PBS_JOBNAME}.err
#PBS -o ${PBS_JOBNAME}.out
#PBS -q batch

ulimit -s unlimited
export OMP_NUM_THREADS=1

VASP_PATH_1="/public/apps/vasp/vasp_std"
VASP_PATH_2="/public/apps/vasp/bin/vasp_std"

if [ -f "$VASP_PATH_1" ]; then
    EXEC=$VASP_PATH_1
elif [ -f "$VASP_PATH_2" ]; then
    EXEC=$VASP_PATH_2
    export I_MPI_ADJUST_REDUCE=3
else
    echo "Error: Cannot find VASP executable."
    exit 1
fi

cd $PBS_O_WORKDIR
NP=`cat $PBS_NODEFILE|wc -l`
mpirun -np $NP $EXEC > ${PBS_JOBNAME}.out
"""

# ================= 功能函数 =================

def write_atom_poscar(element, folder_path):
    """生成单原子 POSCAR"""
    center = BOX_SIZE / 2.0
    filepath = os.path.join(folder_path, "POSCAR")
    with open(filepath, 'w') as f:
        f.write(f"Atom {element}\n")
        f.write("1.0\n")
        f.write(f"   {BOX_SIZE:.6f}   0.000000   0.000000\n")
        f.write(f"   0.000000   {BOX_SIZE:.6f}   0.000000\n")
        f.write(f"   0.000000   0.000000   {BOX_SIZE:.6f}\n")
        f.write(f"   {element}\n")
        f.write("   1\n")
        f.write("Cartesian\n")
        f.write(f"   {center:.6f}   {center:.6f}   {center:.6f}\n")

def generate_potcar(folder_path):
    """调用 vaspkit 生成 POTCAR"""
    original_cwd = os.getcwd()
    try:
        os.chdir(folder_path)
        if shutil.which("vaspkit") is None:
            print(f"   ⚠️  Vaspkit 未找到，跳过 POTCAR")
            return
        # 屏蔽输出
        subprocess.run("vaspkit -task 103", shell=True, check=True, stdout=subprocess.DEVNULL)
        if os.path.exists("POTCAR"):
            print(f"   ✅ POTCAR 生成成功")
        else:
            print(f"   ❌ POTCAR 生成失败")
    except Exception:
        print(f"   ❌ Vaspkit 运行出错")
    finally:
        os.chdir(original_cwd)

def main():
    print("="*50)
    print("   单原子能量计算任务生成器")
    print("="*50)
    
    # 1. 获取输入
    user_input = input("请输入要计算的元素符号 (用空格分隔，例如: C H O Fe): ").strip()
    if not user_input:
        print("未输入元素，退出。")
        return
        
    elements = user_input.split()
    
    print(f"\n🚀 准备生成以下原子的计算任务: {elements}\n")
    
    for el in elements:
        # 规范化元素名 (首字母大写)
        el = el.capitalize()
        folder_name = f"atom_{el}"
        
        # 2. 创建文件夹
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"📂 创建目录: {folder_name}")
        else:
            print(f"📂 目录已存在: {folder_name} (覆盖配置)")
            
        # 3. 写入文件
        write_atom_poscar(el, folder_name)
        
        with open(os.path.join(folder_name, "INCAR"), 'w') as f:
            f.write(INCAR_ATOM)
            
        with open(os.path.join(folder_name, "KPOINTS"), 'w') as f:
            f.write(KPOINTS_CONTENT)
            
        # 自动修改 PBS 脚本中的任务名为 atom_元素名
        pbs_final = PBS_CONTENT.replace("#PBS -N atom_calc", f"#PBS -N atom_{el}")
        with open(os.path.join(folder_name, "wk.pbs"), 'w') as f:
            f.write(pbs_final)
            
        # 4. 生成 POTCAR
        generate_potcar(folder_name)
        
        print("-" * 40)

    print("\n🎉 生成完毕！请进入各文件夹提交任务 (qsub wk.pbs)。")
    print("💡 提示：计算完成后，OUTCAR 中 'energy  without entropy' 的值即为单原子能量。")

if __name__ == "__main__":
    main()

