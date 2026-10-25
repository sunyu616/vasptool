import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from itertools import combinations

# ================= 配置区域 =================
# 设置吸附方向： "top" (上表面, z向上) 或 "bottom" (下表面, z向下)
SEARCH_DIRECTION = "bottom"

# 设置吸附物种类型: "O", "OH", "OOH"
ADSORPTION_TYPE = "OOH"

# 搜索半径 (只保留距离几何中心一定范围内的位点)
RADIUS_LIMIT = 3.0

# 跨类型去重阈值 (Å)
DUPLICATE_THRESHOLD = 0.8

# 输入文件名
INPUT_FILE = "POSCAR"
# 输出根目录名称
OUTPUT_ROOT_DIR = f"adsorption_{ADSORPTION_TYPE}_{SEARCH_DIRECTION}_smart_bond"
# ===========================================

def get_geometric_center(structure):
    """计算晶体结构的几何中心"""
    positions = structure.get_positions()
    return np.mean(positions, axis=0)

def get_surface_atoms(structure, direction="top", z_threshold_ratio=0.8):
    """筛选表面原子"""
    z_coords = structure.positions[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    z_span = z_max - z_min
    
    if direction == "top":
        z_threshold = z_min + z_span * z_threshold_ratio
        return [atom for atom in structure if atom.position[2] >= z_threshold]
    else: # bottom
        z_threshold = z_min + z_span * (1.0 - z_threshold_ratio)
        return [atom for atom in structure if atom.position[2] <= z_threshold]

def get_circumcenter_2d(p1, p2, p3):
    """计算外接圆中心"""
    x1, y1, x2, y2, x3, y3 = p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6: return None, None
    Ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    Uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    center = np.array([Ux, Uy])
    radius = np.linalg.norm(center - np.array([x1, y1]))
    return center, radius

def get_site_fingerprint(structure, site_pos, num_neighbors=6, tolerance=1):
    """生成位点指纹"""
    distances = np.linalg.norm(structure.positions - site_pos, axis=1)
    sorted_indices = np.argsort(distances)[:num_neighbors]
    fingerprint_list = []
    for idx in sorted_indices:
        symbol = structure[idx].symbol
        dist = distances[idx]
        fingerprint_list.append((symbol, round(dist, tolerance)))
    fingerprint_list.sort(key=lambda x: (x[1], x[0]))
    return tuple(fingerprint_list)

def merge_sites_internal(structure, sites, center):
    """同类型内部合并"""
    grouped = {}
    for site in sites:
        fp = get_site_fingerprint(structure, site['pos'])
        dist = np.linalg.norm(site['pos'] - center)
        if fp not in grouped:
            grouped[fp] = []
        grouped[fp].append({'site': site, 'dist': dist})
    unique = []
    for fp, candidates in grouped.items():
        best = sorted(candidates, key=lambda x: x['dist'])[0]
        unique.append(best['site'])
    return unique

def cross_category_filter(sites_dict, threshold=0.8):
    """跨类型去重: Top > Bridge > Hollow"""
    final_top = sites_dict.get('top', [])
    
    raw_bridge = sites_dict.get('bridge', [])
    final_bridge = []
    for b_site in raw_bridge:
        is_duplicate = False
        for t_site in final_top:
            if np.linalg.norm(b_site['pos'] - t_site['pos']) < threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            final_bridge.append(b_site)
            
    raw_hollow = sites_dict.get('hollow', [])
    final_hollow = []
    existing_sites = final_top + final_bridge
    for h_site in raw_hollow:
        is_duplicate = False
        for ex_site in existing_sites:
            if np.linalg.norm(h_site['pos'] - ex_site['pos']) < threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            final_hollow.append(h_site)
            
    return {'top': final_top, 'bridge': final_bridge, 'hollow': final_hollow}

# =========================================================================
# 新增核心功能：智能高度计算
# =========================================================================

def get_covalent_radius(symbol):
    """获取元素的共价半径"""
    return covalent_radii[atomic_numbers[symbol]]

def calculate_optimal_height(site_info, ads_connecting_atom="O"):
    """
    根据位点类型和构成的原子，利用几何关系计算最佳垂直高度。
    
    原理 (勾股定理): 
    Bond_Length^2 = Vertical_Height^2 + Horizontal_Dist^2
    Target_Bond_Length = r_cov(Surface) + r_cov(Adsorbate)
    """
    # 1. 获取吸附连接原子的半径
    r_ads = get_covalent_radius(ads_connecting_atom)
    
    # 2. 获取表面构成原子的半径
    # site_info['constituent_atoms'] 存储了构成该位点的原子符号列表
    surf_symbols = site_info['constituent_atoms']
    
    # 简单起见，取构成位点的原子的平均半径
    r_surf_list = [get_covalent_radius(s) for s in surf_symbols]
    r_surf_avg = np.mean(r_surf_list)
    
    # 3. 计算目标键长 (可稍微加一点经验值，如 0.05Å，因为吸附键通常略长于共价键)
    target_bond_length = r_surf_avg + r_ads 
    
    # 4. 获取位点中心到表面原子的水平距离 (Horizontal Distance)
    # 这个值在 generate_sites_minimal 中被预先计算并存储为 'dist_to_atom_xy'
    # Top: 0, Bridge: bond/2, Hollow: circumradius
    dist_xy = site_info.get('dist_to_atom_xy', 0.0)
    
    # 5. 计算垂直高度 (h = sqrt(d^2 - r^2))
    if target_bond_length > dist_xy:
        height = np.sqrt(target_bond_length**2 - dist_xy**2)
    else:
        # 如果孔太小（dist_xy很大），吸附物进不去，设定一个最小高度防止重叠
        height = 1.0 
        
    return height

def get_adsorbate_geometry(ads_type, height, direction="top"):
    """
    构建吸附物几何结构
    参数:
      height: 动态计算出的垂直高度 (第一个原子到底面的距离)
    """
    # 定义分子内部几何 (基于 O 在原点)
    atoms_data = []
    
    if ads_type == "O":
        atoms_data = [("O", [0.0, 0.0, 0.0])]
        
    elif ads_type == "OH":
        # O-H 键长 ~0.97, 倾斜
        tilt_x = 0.97 * np.sin(np.radians(30)) 
        tilt_z = 0.97 * np.cos(np.radians(30))
        atoms_data = [("O", [0.0, 0.0, 0.0]), ("H", [tilt_x, 0.0, tilt_z])]
        
    elif ads_type == "OOH":
        # O-O ~1.45, O-H ~0.98, O-O-H ~101度
        atoms_data = [
            ("O", [0.0, 0.0, 0.0]),          
            ("O", [1.2, 0.0, 0.9]),          # 第二个O
            ("H", [1.0, 0.0, 1.9])           # H
        ]
    
    symbols = [d[0] for d in atoms_data]
    positions = [d[1] for d in atoms_data]
    ads = Atoms(symbols=symbols, positions=positions)
    
    z_factor = 1.0 if direction == "top" else -1.0
    
    # 如果是 Bottom，翻转分子自身
    if direction == "bottom":
        pos = ads.get_positions()
        pos[:, 2] = -pos[:, 2]
        ads.set_positions(pos)
        
    # 应用计算出的最佳高度
    final_z_offset = height * z_factor
    ads.translate([0, 0, final_z_offset])
    
    return ads

def generate_sites_minimal(structure, center, radius=3.0, direction="top"):
    """生成位点并记录构成原子信息，用于后续键长计算"""
    surface_atoms = get_surface_atoms(structure, direction=direction)
    surf_coords = np.array([atom.position for atom in surface_atoms])
    
    raw_sites = {'top': [], 'bridge': [], 'hollow': []}

    # 1. Top 位点
    for atom in surface_atoms:
        pos = atom.position
        if np.linalg.norm(pos - center) <= radius:
            anchor = f"{atom.symbol}{atom.index + 1}"
            raw_sites['top'].append({
                'pos': pos, 
                'anchor': anchor,
                'constituent_atoms': [atom.symbol], # 记录原子种类
                'dist_to_atom_xy': 0.0 # Top位点水平距离为0
            })

    # 2. Bridge 位点
    for i, j in combinations(range(len(surface_atoms)), 2):
        atom1, atom2 = surface_atoms[i], surface_atoms[j]
        dist_atoms = np.linalg.norm(atom1.position - atom2.position)
        
        if 2.0 < dist_atoms < 3.2:
            mid_pos = (atom1.position + atom2.position) / 2
            if np.linalg.norm(mid_pos - center) <= radius:
                anchor = f"{atom1.symbol}{atom1.index + 1}-{atom2.symbol}{atom2.index + 1}"
                
                # 计算 Bridge中心 到 原子的水平距离 (即键长的一半)
                # 这里我们假设表面大致平行于XY平面，简化计算
                dist_xy = dist_atoms / 2.0
                
                raw_sites['bridge'].append({
                    'pos': mid_pos, 
                    'anchor': anchor,
                    'constituent_atoms': [atom1.symbol, atom2.symbol],
                    'dist_to_atom_xy': dist_xy
                })

    # 3. Hollow 位点
    indices_near_center = [i for i, atom in enumerate(surface_atoms) 
                           if np.linalg.norm(atom.position - center) < radius + 4.0]
    
    for combo in combinations(indices_near_center, 3):
        p1, p2, p3 = surf_coords[list(combo)]
        dists = [np.linalg.norm(p1-p2), np.linalg.norm(p2-p3), np.linalg.norm(p1-p3)]
        if max(dists) > 5.0 or min(dists) < 2.0: continue
            
        xy_center, xy_radius = get_circumcenter_2d(p1[:2], p2[:2], p3[:2])
        if xy_center is None or xy_radius > 2.5: continue
        
        combo_atoms = [surface_atoms[k] for k in combo]
        avg_z = np.mean([a.position[2] for a in combo_atoms])
        base_pos = np.array([xy_center[0], xy_center[1], avg_z])
        
        if np.linalg.norm(base_pos - center) <= radius:
            anchor = f"Hollow_{combo_atoms[0].symbol}" 
            raw_sites['hollow'].append({
                'pos': base_pos, 
                'anchor': anchor, 
                'coordination': 3,
                'constituent_atoms': [a.symbol for a in combo_atoms],
                'dist_to_atom_xy': xy_radius # 外接圆半径即为到原子的水平距离
            })

    # 4. 合并与去重
    print(f"正在搜索 [{direction}] 表面，自动识别键长...")
    merged_sites = {}
    for cat, sites in raw_sites.items():
        if sites:
            merged = merge_sites_internal(structure, sites, center)
            merged_sites[cat] = merged
        else:
            merged_sites[cat] = []
            
    final_sites = cross_category_filter(merged_sites, threshold=DUPLICATE_THRESHOLD)
    
    # 打印统计
    for cat, sites in final_sites.items():
        print(f"  - {cat.ljust(6)}: {len(sites)} 个")
        
    return final_sites

def generate_files(input_poscar, sites_dict, output_root, ads_type="OOH", direction="top"):
    try:
        structure = read(input_poscar)
    except Exception as e:
        print(f"读取POSCAR失败: {e}")
        return
        
    # 确定吸附物连接的原子 (默认第一个原子是连接原子)
    ads_connecting_atom = ads_type[0] # "O" from "OOH" or "OH"
    if ads_type == "H": ads_connecting_atom = "H" # 特殊情况
    
    for stype, sites in sites_dict.items():
        if not sites: continue
        
        base_dir = os.path.join(output_root, stype)
        vasp_dir = os.path.join(base_dir, "vasp")
        cif_dir = os.path.join(base_dir, "cif")
        os.makedirs(vasp_dir, exist_ok=True)
        os.makedirs(cif_dir, exist_ok=True)
        
        for idx, sinfo in enumerate(sites):
            # ================= 核心调用 =================
            # 1. 自动计算该具体位点的最佳吸附高度
            optimal_height = calculate_optimal_height(sinfo, ads_connecting_atom)
            
            # 2. 生成对应高度的吸附物几何
            adsorbate = get_adsorbate_geometry(ads_type, optimal_height, direction)
            # ===========================================
            
            # 3. 移动到位点
            ads_copy = adsorbate.copy()
            ads_copy.translate(sinfo['pos'])
            
            # 4. 合并
            final_struct = structure.copy()
            final_struct += ads_copy
            
            # 5. 输出
            constituent_str = "".join(set(sinfo['constituent_atoms'])) # 简化的元素标记
            coord = f"_{sinfo['coordination']}fold" if 'coordination' in sinfo else ""
            fname_base = f"{stype}{coord}_{idx+1}_{ads_type}_on_{constituent_str}"
            
            # 限制文件名长度，避免太长
            if len(fname_base) > 50: fname_base = f"{stype}{coord}_{idx+1}_{ads_type}"

            write(os.path.join(vasp_dir, f"{fname_base}.vasp"), final_struct, format="vasp", sort=True)
            write(os.path.join(cif_dir, f"{fname_base}.cif"), final_struct, format="cif")
            
            # 打印调试信息（可选，查看计算出的高度是否合理）
            # print(f"Site {idx}: {sinfo['constituent_atoms']} -> Height: {optimal_height:.2f} A")

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        struct = read(INPUT_FILE)
        center = get_geometric_center(struct)
        
        sites = generate_sites_minimal(struct, center, RADIUS_LIMIT, SEARCH_DIRECTION)
        
        total = sum(len(v) for v in sites.values())
        if total > 0:
            generate_files(INPUT_FILE, sites, OUTPUT_ROOT_DIR, ADSORPTION_TYPE, SEARCH_DIRECTION)
            print(f"\n生成完成！共 {total} 个结构。")
            print(f"输出目录: {OUTPUT_ROOT_DIR}")
        else:
            print(f"在 {SEARCH_DIRECTION} 表面未找到位点。")
    else:
        print("当前目录下找不到POSCAR文件。")

