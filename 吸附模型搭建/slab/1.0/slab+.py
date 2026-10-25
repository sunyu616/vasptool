import os
from ase.io import read, write
import numpy as np
from itertools import combinations

# ================= 配置区域 =================
# 设置吸附方向： "top" (上表面, z向上) 或 "bottom" (下表面, z向下)
SEARCH_DIRECTION = "bottom"

# 设置吸附物种类型
ADSORPTION_TYPE = "OOH"
# 【关键修改】搜索半径缩小到 3.0 Å，只看中心！
RADIUS_LIMIT = 3.0
# 输入文件名
INPUT_FILE = "POSCAR"
# 输出根目录名称
OUTPUT_ROOT_DIR = f"adsorption_{ADSORPTION_TYPE}_{SEARCH_DIRECTION}_minimal"
# ===========================================

def get_geometric_center(structure):
    """计算晶体结构的几何中心"""
    positions = structure.get_positions()
    return np.mean(positions, axis=0)

def get_surface_atoms(structure, direction="top", z_threshold_ratio=0.8):
    """
    筛选表面原子
    direction="top": 筛选z坐标最大的部分 (z >= min + span * 0.8)
    direction="bottom": 筛选z坐标最小的部分 (z <= min + span * 0.2)
    """
    z_coords = structure.positions[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    z_span = z_max - z_min
    
    if direction == "top":
        z_threshold = z_min + z_span * z_threshold_ratio
        return [atom for atom in structure if atom.position[2] >= z_threshold]
    else: # bottom
        # 对于下表面，阈值是反过来的，例如取底部 20%
        z_threshold = z_min + z_span * (1.0 - z_threshold_ratio)
        return [atom for atom in structure if atom.position[2] <= z_threshold]

def get_circumcenter_2d(p1, p2, p3):
    """计算外接圆"""
    x1, y1, x2, y2, x3, y3 = p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6: return None, None
    Ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    Uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    center = np.array([Ux, Uy])
    radius = np.linalg.norm(center - np.array([x1, y1]))
    return center, radius

def get_site_fingerprint(structure, site_pos, num_neighbors=8, tolerance=1):
    """
    【核心去重逻辑】
    生成位点的“粗糙”化学环境指纹。
    tolerance=1 表示距离四舍五入到 0.1 Å。
    """
    distances = np.linalg.norm(structure.positions - site_pos, axis=1)
    # 获取最近的N个邻居
    sorted_indices = np.argsort(distances)[:num_neighbors]
    
    fingerprint_list = []
    for idx in sorted_indices:
        symbol = structure[idx].symbol
        dist = distances[idx]
        # 激进的四舍五入：保留1位小数
        fingerprint_list.append((symbol, round(dist, tolerance)))
    
    # 再次排序：确保 [(Pt, 2.5), (Pt, 2.6)] 和 [(Pt, 2.6), (Pt, 2.5)] 被视为相同
    fingerprint_list.sort(key=lambda x: (x[1], x[0]))
    
    return tuple(fingerprint_list)

def filter_and_merge_sites(structure, raw_sites, center):
    """
    根据指纹合并位点，每组只保留离几何中心最近的一个
    """
    grouped_sites = {}
    
    for site in raw_sites:
        fp = get_site_fingerprint(structure, site['pos'])
        dist_to_center = np.linalg.norm(site['pos'] - center)
        
        if fp not in grouped_sites:
            grouped_sites[fp] = []
        grouped_sites[fp].append({'site': site, 'dist': dist_to_center})
    
    unique_sites = []
    
    for fp, candidates in grouped_sites.items():
        # 取距离中心最近的一个
        best_candidate = sorted(candidates, key=lambda x: x['dist'])[0]
        unique_sites.append(best_candidate['site'])
        
    return unique_sites

def generate_sites_minimal(structure, center, radius=3.0, direction="top"):
    """生成并极简合并位点，支持方向选择"""
    surface_atoms = get_surface_atoms(structure, direction=direction)
    surf_coords = np.array([atom.position for atom in surface_atoms])
    
    # 定义Z轴方向系数：Top为1，Bottom为-1
    z_factor = 1.0 if direction == "top" else -1.0
    
    raw_sites = {'top': [], 'bridge': [], 'hollow': []}

    # 1. 扫描所有潜在位点 (Top)
    for atom in surface_atoms:
        # 位点高度：表面原子位置 + 2.0 * 方向
        top_pos = atom.position + np.array([0, 0, 2.0 * z_factor])
        if np.linalg.norm(top_pos - center) <= radius:
            anchor = f"{atom.symbol}{atom.index + 1}"
            raw_sites['top'].append({'pos': top_pos, 'anchor': anchor})

    # 2. 扫描所有潜在位点 (Bridge)
    for i, j in combinations(range(len(surface_atoms)), 2):
        atom1, atom2 = surface_atoms[i], surface_atoms[j]
        dist = np.linalg.norm(atom1.position - atom2.position)
        if 2.0 < dist < 3.0: # 典型键长
            # 位点高度：中点位置 + 1.5 * 方向
            bridge_pos = (atom1.position + atom2.position) / 2 + np.array([0, 0, 1.5 * z_factor])
            if np.linalg.norm(bridge_pos - center) <= radius:
                anchor = f"{atom1.symbol}{atom1.index + 1}"
                raw_sites['bridge'].append({'pos': bridge_pos, 'anchor': anchor})

    # 3. 扫描所有潜在位点 (Hollow - Circumcircle)
    # 缩小搜索圈，只对中心附近的原子进行组合计算
    indices_near_center = [i for i, atom in enumerate(surface_atoms) 
                           if np.linalg.norm(atom.position - center) < radius + 3.0]
    
    for combo in combinations(indices_near_center, 3):
        p1, p2, p3 = surf_coords[list(combo)]
        dists = [np.linalg.norm(p1-p2), np.linalg.norm(p2-p3), np.linalg.norm(p1-p3)]
        if max(dists) > 5.0 or min(dists) < 2.0: continue
            
        xy_center, xy_radius = get_circumcenter_2d(p1[:2], p2[:2], p3[:2])
        if xy_center is None or xy_radius > 3.0: continue
        
        # 统计圆周原子
        coordination_atoms = []
        avg_z = 0.0
        for k in indices_near_center:
            pos = surf_coords[k]
            if abs(np.linalg.norm(pos[:2] - xy_center) - xy_radius) < 0.3:
                coordination_atoms.append(surface_atoms[k])
                avg_z += pos[2]
        
        coord = len(coordination_atoms)
        if coord >= 3:
            avg_z /= coord
            # 根据配位数调整高度，并乘以方向系数
            base_offset = 1.8 if coord == 3 else (1.4 if coord == 4 else 1.0)
            site_pos = np.array([xy_center[0], xy_center[1], avg_z + base_offset * z_factor])
            
            if np.linalg.norm(site_pos - center) <= radius:
                anchor = f"{coordination_atoms[0].symbol}{coordination_atoms[0].index + 1}"
                raw_sites['hollow'].append({
                    'pos': site_pos, 'anchor': anchor, 'coordination': coord
                })

    # 4. 执行激进合并
    print(f"正在搜索 [{direction}] 表面，半径 {radius}Å，并进行合并...")
    final_sites = {}
    for cat, sites in raw_sites.items():
        if sites:
            merged = filter_and_merge_sites(structure, sites, center)
            final_sites[cat] = merged
            print(f"  - {cat.ljust(6)}: 扫描到 {len(sites)} 个 -> 合并保留 {len(merged)} 个")
        else:
            final_sites[cat] = []
    
    return final_sites

def generate_files(input_poscar, sites_dict, output_root, ads_type="OOH", direction="top"):
    try:
        structure = read(input_poscar)
    except:
        return
        
    z_factor = 1.0 if direction == "top" else -1.0
    
    # 基础吸附构型 (相对位置，基于Top面)
    configs = {
        "O": [("O", [0,0,0])],
        "OH": [("O", [0,0,0]), ("H", [0.96,0,0])],
        "OOH": [("O", [0,0,0]), ("O", [0,0,1.4]), ("H", [0.9,0,1.8])]
    }
    
    for stype, sites in sites_dict.items():
        if not sites: continue
        sdir = os.path.join(output_root, stype)
        os.makedirs(sdir, exist_ok=True)
        
        for idx, sinfo in enumerate(sites):
            new_s = structure.copy()
            for elem, rel in configs.get(ads_type, configs["OOH"]):
                new_s.append(elem)
                # 调整Z轴相对位置：乘以方向系数
                # 例如OOH在下表面：O(0), O(-1.4), H(-1.8)
                adj_rel = np.array(rel)
                adj_rel[2] *= z_factor
                
                new_s.positions[-1] = sinfo['pos'] + adj_rel
            
            coord = f"-{sinfo['coordination']}fold" if 'coordination' in sinfo else ""
            fname = f"{sinfo['anchor']}_{stype}{coord}_{idx+1}_{ads_type}.vasp"
            write(os.path.join(sdir, fname), new_s, format="vasp")

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        struct = read(INPUT_FILE)
        center = get_geometric_center(struct)
        
        sites = generate_sites_minimal(struct, center, RADIUS_LIMIT, SEARCH_DIRECTION)
        
        total = sum(len(v) for v in sites.values())
        if total > 0:
            generate_files(INPUT_FILE, sites, OUTPUT_ROOT_DIR, ADSORPTION_TYPE, SEARCH_DIRECTION)
            print(f"\n生成完成！共 {total} 个结构，保存在: {OUTPUT_ROOT_DIR}")
        else:
            print(f"在 {SEARCH_DIRECTION} 表面未找到位点，请检查结构或尝试稍微增大半径。")
    else:
        print("找不到POSCAR文件。")


