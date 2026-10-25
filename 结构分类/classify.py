import os
import glob
import shutil
import warnings
import spglib
from ase import Atoms
from ase.io import read, write
from ase.geometry.dimensionality import analyze_dimensionality
from ase.spacegroup import get_spacegroup

# ================= 脚本配置区域 =================
INPUT_PATTERNS = ["*.cif", "POSCAR*", "*.vasp"] 
OUTPUT_DIR = "Classified_Structures_Supercell"   # 输出归档目录
REPORT_FILE = "classification_report.md"         # 报告文件名

METHOD = 'TSA'                                   # 维度分析算法 (RDA 或 TSA)
TOLERANCES = [0.001, 0.01, 0.1, 0.2, 0.5]        # 容差扫描梯度

# --- ✨ 新增：扩胞 (Supercell) 设置 ✨ ---
ENABLE_SUPERCELL = True     # 是否开启连续方向自动扩胞
SUPERCELL_SIZE = 3          # 在连续方向上扩增的倍数 (例如 3 代表 3x3 平面或 1x3 链)
EXPAND_3D = False           # 是否对 3D 体相结构也强制扩胞 (通常设为 False，避免文件过大)
# ===============================================

def is_subpath(child_path, parent_path):
    """安全判断 child_path 是否在 parent_path 内"""
    try:
        child_abs = os.path.abspath(child_path)
        parent_abs = os.path.abspath(parent_path)
        return os.path.commonpath([child_abs, parent_abs]) == parent_abs
    except ValueError:
        return False

def get_files_by_patterns(patterns, exclude_dir):
    """获取匹配文件，安全过滤输出目录"""
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    valid_files = []
    for f in files:
        if os.path.isfile(f) and not is_subpath(f, exclude_dir):
            valid_files.append(f)
            
    return valid_files

def apply_continuous_supercell(atoms, dim_type, size):
    """根据维度启发式地在连续方向上进行扩胞"""
    if dim_type in ['0D', 'Unknown'] or size <= 1:
        return atoms
        
    if dim_type == '3D' and not EXPAND_3D:
        return atoms
        
    lengths = atoms.cell.lengths()
    # 根据晶格长度从小到大排序的轴索引 (真空层方向往往最长)
    sorted_indices = sorted(range(3), key=lambda i: lengths[i])
    
    mult = [1, 1, 1]
    
    if dim_type == '1D' or '1D' in dim_type:
        # 1D: 最短的 1 个方向是连续轴，其余是真空
        mult[sorted_indices[0]] = size
    if dim_type == '2D' or '2D' in dim_type:
        # 2D: 最短的 2 个方向是连续平面，最长的是真空
        mult[sorted_indices[0]] = size
        mult[sorted_indices[1]] = size
    elif dim_type == '3D' and EXPAND_3D:
        # 3D: 三个方向全部扩胞
        mult = [size, size, size]
        
    # 利用 ASE 的乘法重载快速构建超胞
    super_atoms = atoms * mult
    # 归位所有原子到边界内，使可视化更漂亮
    super_atoms.wrap()
    
    return super_atoms

def get_symmetrized_atoms(atoms, symprec):
    """利用 spglib 重置为标准化的最高对称性晶胞"""
    cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    std_cell = spglib.standardize_cell(cell, symprec=symprec, to_primitive=False)
    
    if std_cell is not None:
        lattice, scaled_positions, numbers = std_cell
        return Atoms(numbers=numbers, cell=lattice, scaled_positions=scaled_positions, pbc=True)
    return atoms

def safe_copy_and_symmetrize(src, dst_folder, symprec, dim_type):
    """安全提取、对称性重写、按需扩胞并拷贝"""
    filename = os.path.basename(src)
    dst_path = os.path.join(dst_folder, filename)
    
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(dst_path):
        dst_path = os.path.join(dst_folder, f"{base}_{counter}{ext}")
        counter += 1
        
    try:
        atoms = read(src)
        
        # 1. 寻找最高对称性并标准化原胞
        sym_atoms = get_symmetrized_atoms(atoms, symprec)
        
        # 2. 在标准原胞的基础上，进行连续方向扩胞
        if ENABLE_SUPERCELL:
            sym_atoms = apply_continuous_supercell(sym_atoms, dim_type, SUPERCELL_SIZE)
        
        # 3. 智能判断输出格式写入
        is_vasp = 'POSCAR' in filename.upper() or ext.lower() == '.vasp'
        if is_vasp:
            write(dst_path, sym_atoms, format='vasp', vasp5=True)
        else:
            write(dst_path, sym_atoms)
            
    except Exception:
        shutil.copy2(src, dst_path)

def find_highest_symmetry(atoms):
    """遍历容差列表，寻找最高空间群并返回触发容差"""
    max_sg_num = 0
    best_symbol = "N_A"
    best_tol = TOLERANCES[0]

    for tol in TOLERANCES:
        try:
            sg_info = get_spacegroup(atoms, symprec=tol)
            if sg_info.no > max_sg_num:
                max_sg_num = sg_info.no
                best_symbol = sg_info.symbol
                best_tol = tol
        except Exception:
            continue  

    best_symbol_clean = best_symbol.replace("/", "_").replace(" ", "")
    return max_sg_num, best_symbol_clean, best_tol

def main():
    warnings.filterwarnings('ignore')

    structure_files = get_files_by_patterns(INPUT_PATTERNS, OUTPUT_DIR)
    total_files = len(structure_files)
    
    if not total_files:
        print(f"[-] 未找到符合 {INPUT_PATTERNS} 的未处理结构。")
        return

    supercell_msg = f"已开启定向扩胞 ({SUPERCELL_SIZE}x)" if ENABLE_SUPERCELL else "未开启扩胞"
    print(f"[+] 发现 {total_files} 个文件，开始扫描 [维度 + 对称性 | {supercell_msg}]...\n")
    
    header = f"{'进度':<10} | {'源文件名称':<25} | {'维度':<6} | {'最高空间群':<15} | {'容差':<6} | {'原胞体积(Å³)'}"
    print(header)
    print("-" * len(header))

    classified_data = {}
    failed_files = []

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 阶段一：扫描分析 (基于原胞) ---
    for i, filename in enumerate(structure_files):
        display_name = (filename[:22] + '...') if len(filename) > 25 else filename
        progress = f"[{i+1}/{total_files}]"
        
        try:
            atoms = read(filename)
            
            try:
                intervals = analyze_dimensionality(atoms, method=METHOD)
            except:
                fallback = 'TSA' if METHOD == 'RDA' else 'RDA'
                try:
                    intervals = analyze_dimensionality(atoms, method=fallback)
                except:
                    intervals = None

            dim_type = "Unknown"
            if intervals and len(intervals) > 0:
                dim_type = intervals[0].dimtype

            formula = atoms.get_chemical_formula(mode='hill')
            sg_num, sg_symbol, matched_tol = find_highest_symmetry(atoms)
            vol_per_atom = atoms.get_volume() / len(atoms)

            fingerprint = (formula, sg_num, sg_symbol)
            sg_display = f"SG:{sg_num} ({sg_symbol})"
            print(f"{progress:<10} | {display_name:<25} | {dim_type:<6} | {sg_display:<15} | {matched_tol:<6} | {vol_per_atom:.2f}")

            if dim_type not in classified_data:
                classified_data[dim_type] = {}
            if fingerprint not in classified_data[dim_type]:
                classified_data[dim_type][fingerprint] = []
                
            classified_data[dim_type][fingerprint].append((filename, vol_per_atom, matched_tol))

        except Exception as e:
            err_msg = str(e).split('\n')[0][:20]
            print(f"{progress:<10} | {display_name:<25} | {'ERROR':<6} | {'-':<15} | {'-':<6} | {err_msg}")
            failed_files.append((filename, str(e)))

    print("\n[+] 扫描结束。正在执行【无损提取 -> 对称性标准化 -> 定向扩胞 -> 写入】...\n")

    # --- 阶段二：物理写入与拷贝 (在此阶段扩胞) ---
    success_count = total_files - len(failed_files)
    
    for dim_type, groups in classified_data.items():
        dim_dir = os.path.join(OUTPUT_DIR, dim_type)
        os.makedirs(dim_dir, exist_ok=True)
        
        for fingerprint, file_info_list in groups.items():
            formula, sg_num, sg_symbol = fingerprint
            folder_name = f"{formula}_SG{sg_num}_{sg_symbol}"
            target_dir = os.path.join(dim_dir, folder_name)
            
            os.makedirs(target_dir, exist_ok=True)
            
            for f, _, best_tol in file_info_list:
                # 传入 dim_type 以决定在哪些方向上扩胞
                safe_copy_and_symmetrize(f, target_dir, best_tol, dim_type)

    # --- 阶段三：生成报告 ---
    report_path = os.path.join(OUTPUT_DIR, REPORT_FILE)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# CALYPSO 结构分类与对称性报告\n\n")
        f.write("> **核心处理流**：无损读取 -> 晶胞高对称性重置 -> **自动定向扩胞** -> 独立导出。\n\n")
        
        f.write("## 1. 扫描与处理汇总\n")
        f.write(f"- **总扫描文件数**: `{total_files}`\n")
        f.write(f"- **成功解析分类**: `{success_count}`\n")
        f.write(f"- **解析失败/崩溃**: `{len(failed_files)}`\n")
        f.write(f"- **容差扫描梯度**: `{TOLERANCES}`\n")
        if ENABLE_SUPERCELL:
            f.write(f"- **连续方向扩胞**: ✅ `已开启` (超胞倍数: **{SUPERCELL_SIZE}**)\n")
        else:
            f.write("- **连续方向扩胞**: ❌ `未开启`\n")
        f.write("---\n\n")

        f.write("## 2. 分类明细\n\n")
        for dim in sorted(classified_data.keys()):
            f.write(f"### 💠 维度类别: `{dim}`\n\n")
            groups = classified_data[dim]
            
            sorted_fps = sorted(groups.keys(), key=lambda x: x[1], reverse=True)
            
            for fp in sorted_fps:
                formula, sg, symbol = fp
                file_info_list = groups[fp]
                folder_name = f"{formula}_SG{sg}_{symbol}"
                
                f.write(f"#### 📁 {folder_name}\n")
                f.write(f"- **空间群**: No.{sg} ({symbol})\n")
                f.write(f"- **文件数量**: {len(file_info_list)} 个\n")
                
                f.write("- **包含文件详情**:\n")
                for df, vol, tol in file_info_list:
                    base_df = os.path.basename(df)
                    f.write(f"  - `{base_df}` (覆盖容差: {tol}, 原胞单原子体积: {vol:.2f} Å³)\n")
                f.write("\n")
                
        if failed_files:
            f.write("---\n\n## 3. 解析失败清单 ⚠️\n\n")
            f.write("| 原始文件名 | 报错摘要 |\n")
            f.write("|---|---|\n")
            for fname, err in failed_files:
                clean_err = err.replace('\n', ' ').replace('|', '/')
                f.write(f"| `{os.path.basename(fname)}` | {clean_err} |\n")

    print(f"[+] 运行完毕！\n    结果归档目录: {os.path.abspath(OUTPUT_DIR)}")
    print(f"    分类统计报告: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    main()

