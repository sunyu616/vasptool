import os
import shutil
import numpy as np
from ase.io import read, write

# ================= 用户配置区域 =================

# 磁性元素符号
MAG_ELEMENT = 'Fe'

# 磁矩大小
MAG_VALUE = 1.0

# 提交脚本文件名
JOB_SCRIPT = 'job.sh'

# 坐标聚类的容差 (埃/分数坐标)
# 用于判断原子是否在同一层，通常 0.1 足够
TOLERANCE = 0.05 

# ===============================================

SOURCE_DIR = os.getcwd()

def get_smart_folder_name(full_mag_list, prefix="afm"):
    """
    生成简洁的文件夹名
    """
    signs = ['+' if x > 0 else '-' for x in full_mag_list]
    
    # 如果是全正或全负，标记为FM
    if all(x > 0 for x in full_mag_list): return "FM"
    
    # 尝试压缩名字
    if len(signs) <= 4:
        suffix = "".join(signs)
    else:
        name_parts = []
        current_sign = signs[0]
        current_count = 0
        for s in signs:
            if s == current_sign:
                current_count += 1
            else:
                name_parts.append(f"{current_count}{current_sign}")
                current_sign = s
                current_count = 1
        name_parts.append(f"{current_count}{current_sign}")
        suffix = "_".join(name_parts)
        
    return f"{prefix}_{suffix}"

def get_vasp_magmom_str(atoms):
    """生成 VASP MAGMOM 字符串"""
    mags = atoms.get_initial_magnetic_moments()
    if len(mags) == 0: return ""
    
    mag_parts = []
    current_val = mags[0]
    count = 1
    
    for i in range(1, len(mags)):
        if np.isclose(mags[i], current_val, atol=0.01):
            count += 1
        else:
            mag_parts.append(f"{count}*{current_val:.3f}")
            current_val = mags[i]
            count = 1
    mag_parts.append(f"{count}*{current_val:.3f}")
    return " ".join(mag_parts)

def modify_incar(src_incar, dst_incar, magmom_str):
    """写入 INCAR"""
    if not os.path.exists(src_incar): return

    with open(src_incar, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        upper = line.strip().upper()
        if any(x in upper for x in ['MAGMOM', 'NUPDOWN', 'M_CONSTR']):
            continue
        if 'ICHARG' in upper:
            new_lines.append("ICHARG = 2\n")
            continue
        if 'ISTART' in upper:
            new_lines.append("ISTART = 0\n")
            continue
        new_lines.append(line)
    
    new_lines.append("\n# --- AFM Configuration ---\n")
    new_lines.append(f"MAGMOM = {magmom_str}\n")
    if not any('ICHARG' in line for line in new_lines):
        new_lines.append("ICHARG = 2\n")
    
    with open(dst_incar, 'w') as f:
        f.writelines(new_lines)

def create_afm_task(folder_name, original_atoms, full_mag_list):
    """
    创建任务 (保持元素顺序)
    """
    target_path = os.path.join(SOURCE_DIR, folder_name)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # 1. 设置磁矩
    atoms = original_atoms.copy()
    mag_indices = [atom.index for atom in atoms if atom.symbol == MAG_ELEMENT]
    
    init_mags = np.zeros(len(atoms))
    for i, atom_idx in enumerate(mag_indices):
        init_mags[atom_idx] = full_mag_list[i] * MAG_VALUE
    atoms.set_initial_magnetic_moments(init_mags)
    
    # 2. 保持 POTCAR 元素顺序排序
    element_order = []
    for a in atoms:
        if a.symbol not in element_order:
            element_order.append(a.symbol)
            
    sorted_indices = []
    for el in element_order:
        indices = [i for i, a in enumerate(atoms) if a.symbol == el]
        sorted_indices.extend(indices)
        
    sorted_atoms = atoms[sorted_indices]
    
    # 3. 写入文件
    write(os.path.join(target_path, 'POSCAR'), sorted_atoms, format='vasp', sort=False)
    
    mag_str = get_vasp_magmom_str(sorted_atoms)
    
    modify_incar(os.path.join(SOURCE_DIR, 'INCAR'), 
                 os.path.join(target_path, 'INCAR'), 
                 mag_str)
    
    for fname in ['POTCAR', 'KPOINTS', JOB_SCRIPT]:
        src = os.path.join(SOURCE_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(target_path, fname))
            
    print(f" -> 生成: {folder_name:<20}")

# ================= 空间逻辑核心 =================

def get_layer_indices(positions, axis=2, tolerance=0.05):
    """
    将原子按指定轴的坐标分组（分层）。
    返回一个列表，列表长度等于原子数，值为该原子所在的层索引 (0, 1, 2...)
    """
    coords = positions[:, axis]
    # 简单的聚类：排序后，如果相邻距离大于 tolerance，则认为是新的一层
    indices = np.argsort(coords)
    sorted_coords = coords[indices]
    
    layer_map = np.zeros(len(coords), dtype=int)
    current_layer = 0
    
    # 记录原始索引到层号的映射
    # 第一层
    layer_map[indices[0]] = 0
    
    for i in range(1, len(coords)):
        if sorted_coords[i] - sorted_coords[i-1] > tolerance:
            current_layer += 1
        layer_map[indices[i]] = current_layer
        
    return layer_map, current_layer + 1

def main():
    print(f"=== AFM 空间构建工具 (支持长方形/层状结构) ===")
    
    struct_file = "POSCAR" if os.path.exists("POSCAR") else "CONTCAR"
    if not os.path.exists(struct_file):
        print("错误: 找不到 POSCAR")
        exit()
        
    raw_atoms = read(struct_file)
    # 获取所有磁性原子及其分数坐标
    mag_atoms = [a for a in raw_atoms if a.symbol == MAG_ELEMENT]
    mag_indices = [a.index for a in raw_atoms if a.symbol == MAG_ELEMENT]
    mag_frac_positions = raw_atoms.get_scaled_positions()[mag_indices]
    
    count = len(mag_atoms)
    print(f"磁性原子: {count} 个 {MAG_ELEMENT}")

    if count < 2:
        print("原子数不足。")
        exit()

    generated_configs = set()

    def submit_config(config_list, name_hint):
        config_tuple = tuple(config_list)
        if config_tuple in generated_configs:
            # print(f" [跳过] 配置重复 ({name_hint})")
            return
        generated_configs.add(config_tuple)
        
        # 使用智能命名，或者结合空间提示命名
        final_name = get_smart_folder_name(config_list, prefix=name_hint)
        create_afm_task(final_name, raw_atoms, config_list)

    print("\n--- 1. 基于索引的模式 (适合简单结构) ---")
    # 基础交替
    pattern_alt = []
    for i in range(count): pattern_alt.append(1 if i%2==0 else -1)
    submit_config(pattern_alt, "afm_idx")

    print("\n--- 2. 基于空间坐标的模式 (适合长方形/层状结构) ---")
    
    # 定义三个方向
    axes = {'X': 0, 'Y': 1, 'Z': 2}
    
    for axis_name, axis_idx in axes.items():
        # 获取分层信息
        layer_ids, num_layers = get_layer_indices(mag_frac_positions, axis=axis_idx, tolerance=TOLERANCE)
        
        # 只有当层数 >= 2 时，层间反铁磁才有意义
        if num_layers >= 2:
            # 策略A: 层间交替 (Layer AFM: + - + -)
            # 这一层全是+，下一层全是-
            mag_config = []
            for i in range(count):
                layer_id = layer_ids[i]
                sign = 1 if layer_id % 2 == 0 else -1
                mag_config.append(sign)
            
            submit_config(mag_config, f"AFM_Layer_{axis_name}")
            
    print("\n--- 3. 棋盘格模式 (G-Type / Checkerboard) ---")
    # 简单的 3D 棋盘格逻辑: sign = (-1)^(x_layer + y_layer + z_layer)
    # 这对于长方形晶格往往能生成 G-type 或 C-type
    
    lx, _ = get_layer_indices(mag_frac_positions, 0)
    ly, _ = get_layer_indices(mag_frac_positions, 1)
    lz, _ = get_layer_indices(mag_frac_positions, 2)
    
    g_config = []
    for i in range(count):
        # 索引之和为偶数则+，奇数则-
        grid_sum = lx[i] + ly[i] + lz[i]
        sign = 1 if grid_sum % 2 == 0 else -1
        g_config.append(sign)
        
    submit_config(g_config, "AFM_G_Type")

    print("\n完成！建议优先检查 AFM_Layer_X/Y/Z 系列文件夹。")

if __name__ == "__main__":
    main()

