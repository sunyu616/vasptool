import os
import glob
import shutil
import warnings
from ase.io import read
from ase.geometry.dimensionality import analyze_dimensionality
from ase.spacegroup import get_spacegroup

# ================= ⚙️ 脚本配置中心 =================
INPUT_EXTENSION = "*.cif"             # 待处理文件后缀
REPORT_FILE = "final_classification_report.txt"
METHOD = 'RDA'                        # 维度判定算法 (推荐 'RDA', 可选 'TSA')

# --- 结构判别精度设置 ---
# 1. 空间群对称性精度 (Symprec)
#    用于判断空间群。0.1 是个宽松值，能忽略轻微的原子坐标弛豫差异。
SYMPREC = 0.1                         

# 2. 晶格常数容差 (Lattice Tolerance, 单位: Å)
#    如果两个结构的晶格边长差异小于此值，且其他条件相同，视为同一结构。
LATTICE_TOL = 0.1                     

# 3. 角度容差 (Angle Tolerance, 单位: 度)
#    晶格角度的容许误差。
ANGLE_TOL = 1.0                       
# ====================================================

def get_prefix_number(filename):
    """
    提取文件名中第一个 '_' 前的数字。
    用于在同一组文件中找到“序号最小”的文件，以此来命名文件夹。
    例如: '12_MoS2.cif' -> 12
    """
    try:
        basename = os.path.basename(filename)
        if "_" in basename:
            prefix = basename.split("_")[0]
            if prefix.isdigit():
                return int(prefix)
    except:
        pass
    # 如果找不到数字，返回无穷大，让其排在最后
    return float('inf')

def get_high_precision_fingerprint(atoms):
    """
    生成结构的【高精度指纹】。
    只有当两个结构的指纹完全一致时，才会被归入同一个文件夹。
    指纹包含: (化学式, 空间群, 原子数, 排序后的晶格边长, 排序后的角度, 平均体积)
    """
    # 1. 化学式 (Hill Notation)
    formula = atoms.get_chemical_formula(mode='hill')
    
    # 2. 空间群编号 (1-230)
    try:
        sg_info = get_spacegroup(atoms, symprec=SYMPREC)
        sg_num = sg_info.no
    except:
        sg_num = 0 # 无法识别

    # 3. 原子总数 (区分超胞)
    n_atoms = len(atoms)

    # 4. 晶胞参数 (Lengths & Angles)
    #    关键步骤: 使用 sorted() 排序。
    #    目的: 解决轴定义不同带来的误判 (例如 a=3,b=5 和 a=5,b=3 应视为相同)
    lengths = sorted(atoms.cell.lengths())
    angles = sorted(atoms.cell.angles())

    #    离散化处理: 根据容差 LATTICE_TOL 进行四舍五入
    #    如果不这样做，3.14159 和 3.14158 会被判为不同结构
    fingerprint_lengths = tuple([round(x / LATTICE_TOL) * LATTICE_TOL for x in lengths])
    fingerprint_lengths = tuple([round(x, 3) for x in fingerprint_lengths]) # 保留3位小数供存储

    fingerprint_angles = tuple([round(x / ANGLE_TOL) * ANGLE_TOL for x in angles])
    fingerprint_angles = tuple([round(x, 2) for x in fingerprint_angles])

    # 5. 平均原子体积 (Vol/atom)
    vol = atoms.get_volume() / n_atoms
    fingerprint_vol = round(vol, 2)

    return (formula, sg_num, n_atoms, fingerprint_lengths, fingerprint_angles, fingerprint_vol)

def main():
    # 屏蔽 ASE 的部分非关键警告
    warnings.filterwarnings('ignore')

    # 1. 扫描文件
    cif_files = glob.glob(INPUT_EXTENSION)
    if not cif_files:
        print(f"当前目录下未找到 {INPUT_EXTENSION} 文件。")
        return

    print(f"找到 {len(cif_files)} 个文件。")
    print(f"开始执行高精度分类 (Dim: {METHOD} | LatTol: {LATTICE_TOL}A)...\n")

    # 数据存储: data[维度][指纹] = [文件路径列表]
    classified_data = {}
    
    stats = {'success': 0, 'fail': 0}

    # --- 第一阶段：分析文件 ---
    for i, filename in enumerate(cif_files):
        print(f"[{i+1}/{len(cif_files)}] 分析: {filename} ... ", end="")
        
        try:
            atoms = read(filename)
            
            # --- A. 维度判断 (带重试机制) ---
            try:
                intervals = analyze_dimensionality(atoms, method=METHOD)
            except Exception:
                fallback = 'TSA' if METHOD == 'RDA' else 'RDA'
                intervals = analyze_dimensionality(atoms, method=fallback)

            if intervals:
                dim_type = intervals[0].dimtype # '0D', '1D', '2D', '3D'
            else:
                dim_type = "Unknown"

            # --- B. 计算结构指纹 ---
            fp = get_high_precision_fingerprint(atoms)
            # 解包指纹用于打印显示
            formula, sg, _, _, _, _ = fp

            # --- C. 存入字典 ---
            if dim_type not in classified_data:
                classified_data[dim_type] = {}
            
            if fp not in classified_data[dim_type]:
                classified_data[dim_type][fp] = []
            
            classified_data[dim_type][fp].append(filename)

            print(f"-> {dim_type} | {formula} (SG:{sg})")
            stats['success'] += 1

        except Exception as e:
            print(f"-> [错误] {str(e)}")
            stats['fail'] += 1

    print(f"\n分析阶段完成。开始建立目录结构并移动文件...\n")

    # --- 第二阶段：创建文件夹并移动 ---
    # 遍历维度 (例如 2D, 3D...)
    for dim_type, groups in classified_data.items():
        # 1. 创建维度主目录
        if not os.path.exists(dim_type):
            os.makedirs(dim_type)
        
        # 遍历该维度下的每一个独特结构组
        for fp, files in groups.items():
            
            # 2. 寻找“组长”文件 (命名基准)
            sorted_files = sorted(files, key=get_prefix_number)
            leader_file = sorted_files[0]
            
            # 3. 确定文件夹名
            folder_name = os.path.splitext(os.path.basename(leader_file))[0]
            target_dir = os.path.join(dim_type, folder_name)
            
            # 创建组文件夹
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            # 4. 移动文件
            for f in files:
                src = f
                dst = os.path.join(target_dir, os.path.basename(f))
                
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                else:
                    print(f"警告: 文件 {dst} 已存在，跳过移动。")
            
            print(f"归档: {dim_type}/{folder_name}/ (共 {len(files)} 个结构)")

    # --- 第三阶段：生成报告 ---
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("=== 高精度结构分类总结报告 ===\n")
        f.write(f"总计处理: {stats['success'] + stats['fail']} (成功: {stats['success']})\n")
        f.write(f"分类逻辑: 维度 -> 化学式 -> 空间群 -> 晶格参数(±{LATTICE_TOL}Å)\n")
        f.write("======================================\n\n")

        # 为了报告清晰，对维度和组进行排序
        for dim in sorted(classified_data.keys()):
            f.write(f"### {dim} 结构区域 ###\n")
            
            # 组排序：先按化学式字母序
            sorted_groups = sorted(classified_data[dim].items(), key=lambda x: x[0][0])
            
            for fp, files in sorted_groups:
                formula, sg, n_atoms, lengths, angles, vol = fp
                
                # 获取文件夹名用于展示
                sorted_files = sorted(files, key=get_prefix_number)
                folder_name = os.path.splitext(os.path.basename(sorted_files[0]))[0]

                f.write(f"  [文件夹]: {folder_name}/\n")
                f.write(f"    - 包含文件: {len(files)} 个\n")
                f.write(f"    - 基本信息: {formula} | 空间群 No.{sg} | 原子数 {n_atoms}\n")
                f.write(f"    - 晶胞参数: {lengths} Å\n")
                f.write(f"    - 晶胞角度: {angles}°\n")
                f.write(f"    - 单原子体积: {vol} Å³\n")
                f.write("\n")
            f.write("-" * 40 + "\n\n")

    print(f"\n全部完成！详细报告已保存至: {REPORT_FILE}")

if __name__ == "__main__":
    main()
