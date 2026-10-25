import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from ase.build import molecule
from itertools import combinations
import shutil
import random

# ================= 用户配置区域 =================

INPUT_FILE = "POSCAR"
SEARCH_DIRECTION = "top"
RADIUS_LIMIT = 3.0

# 吸附模式选择
SELECT_MODE = "OH"   

ADSORPTION_SCHEMES = {
    "H":   {"molecule": "H",   "binding": "H", "scale": 0.9,  "tilt": 0,  "rotate": False},
    "O":   {"molecule": "O",   "binding": "O", "scale": 1.0,  "tilt": 0,  "rotate": False},
    "OH":  {"molecule": "OH",  "binding": "O", "scale": 1.05, "tilt": 15, "rotate": True},
    "OOH": {"molecule": "OOH", "binding": "O", "scale": 1.10, "tilt": 45, "rotate": True}
}

# [关键参数：桥位判定]
MIN_BRIDGE_LENGTH = 2.0
MAX_BRIDGE_LENGTH = 3.8 
# [新增] 桥位两端原子的最大高度差 (Å)
# 防止连接顶层原子和次层原子，确保桥是“平”的
MAX_BRIDGE_Z_DIFF = 0.5 

# [Hollow 判定参数]
MAX_HOLLOW_RADIUS = 2.3 
COORDINATION_TOLERANCE = 0.2

# 合并阈值
HOLLOW_TO_TOP_MERGE = 1.0     
HOLLOW_TO_BRIDGE_MERGE = 0.6    
HOLLOW_DUPLICATE_DIST = 0.5     

# --- 自动加载配置 ---
if SELECT_MODE not in ADSORPTION_SCHEMES:
    raise ValueError(f"Unknown mode: {SELECT_MODE}")

CURRENT_CONFIG = ADSORPTION_SCHEMES[SELECT_MODE]
ADSORBATE_INPUT = CURRENT_CONFIG["molecule"]
BINDING_ELEMENT = CURRENT_CONFIG["binding"]
BOND_SCALE_FACTOR = CURRENT_CONFIG["scale"]
TILT_ANGLE = CURRENT_CONFIG["tilt"]
ENABLE_RANDOM_ROTATION = CURRENT_CONFIG["rotate"]

OUTPUT_ROOT_DIR = f"adsorption_{ADSORBATE_INPUT}_{SEARCH_DIRECTION}"

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    from scipy.spatial import Voronoi, Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# =========================================================================
# 核心模块
# =========================================================================

def optimize_molecule_geometry(input_str):
    if len(input_str) <= 2 and input_str in ["H", "O", "N", "C", "S"]:
        return Atoms(input_str, positions=[(0, 0, 0)]), "Single Atom"

    mol_atoms = None
    method = "Unknown"

    try:
        if input_str == "OOH":
             mol_atoms = Atoms('OOH', positions=[(0, 0, 0), (0, 0, 1.45), (0.9, 0, 1.7)])
             method = "Manual OOH"
        else:
            mol_atoms = molecule(input_str)
            method = "ASE Database"
    except: pass

    if mol_atoms is None and HAS_RDKIT:
        try:
            rd_mol = Chem.MolFromSmiles(input_str)
            if rd_mol:
                rd_mol = Chem.AddHs(rd_mol)
                if AllChem.EmbedMolecule(rd_mol, AllChem.ETKDG()) == 0:
                    AllChem.MMFFOptimizeMolecule(rd_mol)
                    conf = rd_mol.GetConformer()
                    mol_atoms = Atoms(symbols=[a.GetSymbol() for a in rd_mol.GetAtoms()], positions=conf.GetPositions())
                    method = "RDKit"
        except: pass

    if mol_atoms is None:
        if input_str == "OH": 
            mol_atoms = Atoms('OH', positions=[(0, 0, 0), (0, 0, 0.97)])
            method = "Manual"
            
    return mol_atoms, method

def calculate_smart_height(site_info, binding_symbol):
    surf_radii = [covalent_radii[atomic_numbers[s]] for s in site_info['constituent_atoms']]
    r_surf = np.mean(surf_radii)
    r_ads = covalent_radii[atomic_numbers[binding_symbol]]
    target_bond_length = (r_surf + r_ads) * BOND_SCALE_FACTOR
    dist_xy = site_info['dist_xy']
    
    if target_bond_length > dist_xy:
        height = np.sqrt(target_bond_length**2 - dist_xy**2)
    else:
        height = 1.0 
    return height

def prepare_adsorbate_ultimate(input_str, target_element, direction, tilt_angle, random_rot):
    ads, method = optimize_molecule_geometry(input_str)
    if ads is None: return None, None, None
    
    target_indices = [atom.index for atom in ads if atom.symbol == target_element]
    if not target_indices: return None, None, None
    binding_idx = target_indices[0]
    binding_sym = ads[binding_idx].symbol

    ads.translate(-ads.positions[binding_idx])
    
    if len(ads) > 1:
        other_indices = [i for i in range(len(ads)) if i != binding_idx]
        geom_center = np.mean(ads.get_positions()[other_indices], axis=0)
        if np.linalg.norm(geom_center) > 0.01:
            ads.rotate(geom_center, 'z') 

    if abs(tilt_angle) > 0.1 and len(ads) > 1:
        ads.rotate(tilt_angle, 'x')

    if random_rot and len(ads) > 1:
        rand_angle = random.uniform(0, 360)
        ads.rotate(rand_angle, 'z')

    if direction == "bottom":
        pos = ads.get_positions(); pos[:, 2] = -pos[:, 2]; ads.set_positions(pos)
        
    return ads, binding_sym

def get_surface_atoms(structure, direction="top", z_threshold_ratio=0.8):
    z_coords = structure.positions[:, 2]
    z_span = z_coords.max() - z_coords.min()
    if direction == "top":
        threshold = z_coords.min() + z_span * z_threshold_ratio
        return [atom for atom in structure if atom.position[2] >= threshold]
    else:
        threshold = z_coords.min() + z_span * (1.0 - z_threshold_ratio)
        return [atom for atom in structure if atom.position[2] <= threshold]

# =========================================================================
# 优化的 Delaunay 桥位搜索 (基于图论的最近邻)
# =========================================================================
def get_bridge_sites_delaunay(surface_atoms, center, radius_limit):
    """
    使用 Delaunay 三角剖分的边来寻找桥位。
    Delaunay 的边天然连接了最近邻原子，避免了 O(N^2) 遍历和误判。
    """
    if not HAS_SCIPY or len(surface_atoms) < 3:
        return []

    points_2d = np.array([a.position[:2] for a in surface_atoms])
    tri = Delaunay(points_2d)
    
    # 提取所有唯一的边
    edges = set()
    for simplex in tri.simplices:
        # simplex 是三角形三个顶点的索引 [i, j, k]
        # 排序以保证唯一性 (0,1) == (1,0)
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[0], simplex[2]))))
    
    candidates = []
    
    for (i, j) in edges:
        a1 = surface_atoms[i]
        a2 = surface_atoms[j]
        
        # 1. 距离检查
        dist_3d = np.linalg.norm(a1.position - a2.position)
        if not (MIN_BRIDGE_LENGTH < dist_3d < MAX_BRIDGE_LENGTH):
            continue
            
        # 2. [新增] 高度差检查
        # 避免连接两个高度差异很大的原子 (例如连接了台阶上和台阶下的)
        if abs(a1.position[2] - a2.position[2]) > MAX_BRIDGE_Z_DIFF:
            continue
            
        # 3. 计算中点
        mid = (a1.position + a2.position) / 2
        
        # 4. 搜索半径检查
        if np.linalg.norm(mid - center) <= radius_limit:
            candidates.append({
                'pos': mid, 
                'anchor': f"{a1.symbol}-{a2.symbol}", 
                'constituent_atoms': [a1.symbol, a2.symbol], 
                'dist_xy': dist_3d/2
            })
            
    return candidates

def get_hollow_sites_voronoi(surface_atoms, center, radius_limit):
    if not HAS_SCIPY or len(surface_atoms) < 3: return []
    points_2d = np.array([a.position[:2] for a in surface_atoms])
    vor = Voronoi(points_2d)
    candidates = []
    for v_idx, vertex in enumerate(vor.vertices):
        if np.linalg.norm(vertex - center[:2]) > radius_limit: continue
        dists = np.linalg.norm(points_2d - vertex, axis=1)
        min_dist = np.min(dists)
        if min_dist > MAX_HOLLOW_RADIUS: continue
        neighbor_indices = np.where(dists < min_dist + COORDINATION_TOLERANCE)[0]
        if len(neighbor_indices) < 3: continue 
        neighbors = [surface_atoms[i] for i in neighbor_indices]
        avg_z = np.mean([a.position[2] for a in neighbors])
        full_pos = np.array([vertex[0], vertex[1], avg_z])
        if np.linalg.norm(full_pos - center) <= radius_limit:
            candidates.append({
                'pos': full_pos, 'anchor': "Voronoi", 
                'constituent_atoms': [a.symbol for a in neighbors], 'dist_xy': min_dist 
            })
    return candidates

# Fallback functions
def get_circumcenter_2d(p1, p2, p3):
    x1, y1, x2, y2, x3, y3 = p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6: return None, None
    Ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    Uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    return np.array([Ux, Uy]), np.linalg.norm(np.array([Ux, Uy]) - np.array([x1, y1]))

def check_triangle_connectivity(p1, p2, p3, max_edge):
    d1 = np.linalg.norm(p1 - p2); d2 = np.linalg.norm(p2 - p3); d3 = np.linalg.norm(p3 - p1)
    return max(d1, d2, d3) <= max_edge

def merge_sites_geometric(structure, sites, center):
    if not sites: return []
    grouped = {}
    for site in sites:
        dists = np.linalg.norm(structure.positions - site['pos'], axis=1)
        fp_list = sorted([(structure[i].symbol, round(d, 1)) for i, d in enumerate(dists) if d < 4.0])
        fp = tuple(fp_list[:4]) 
        if fp not in grouped: grouped[fp] = []
        grouped[fp].append({'site': site, 'dist': np.linalg.norm(site['pos'] - center)})
    return [sorted(g, key=lambda x: x['dist'])[0]['site'] for g in grouped.values()]

# =========================================================================
# 整合后的位点生成逻辑
# =========================================================================
def generate_sites_ultimate(structure, center, radius=3.0, direction="top"):
    surface_atoms = get_surface_atoms(structure, direction=direction)
    
    raw_top = []
    raw_bridge = []
    raw_hollow = []
    
    # --- 1. Top ---
    for a in surface_atoms:
        if np.linalg.norm(a.position - center) <= radius:
            raw_top.append({
                'pos': a.position, 'anchor': f"{a.symbol}", 
                'constituent_atoms': [a.symbol], 'dist_xy': 0.0
            })

    # --- 2. Bridge (Delaunay 优化版 / 组合法备用) ---
    if HAS_SCIPY:
        # 使用 Delaunay Edges 寻找最近邻桥位
        raw_bridge = get_bridge_sites_delaunay(surface_atoms, center, radius)
    else:
        # Fallback 到暴力组合 (增加高度差检查)
        for i, j in combinations(range(len(surface_atoms)), 2):
            a1, a2 = surface_atoms[i], surface_atoms[j]
            dist = np.linalg.norm(a1.position - a2.position)
            z_diff = abs(a1.position[2] - a2.position[2])
            
            # 严格检查: 距离合适 且 高度差小
            if (MIN_BRIDGE_LENGTH < dist < MAX_BRIDGE_LENGTH) and (z_diff < MAX_BRIDGE_Z_DIFF): 
                mid = (a1.position + a2.position) / 2
                if np.linalg.norm(mid - center) <= radius:
                    raw_bridge.append({
                        'pos': mid, 'anchor': f"{a1.symbol}-{a2.symbol}", 
                        'constituent_atoms': [a1.symbol, a2.symbol], 'dist_xy': dist/2
                    })

    # --- 3. Hollow (Voronoi 优先 / 组合法备用) ---
    if HAS_SCIPY:
        raw_hollow = get_hollow_sites_voronoi(surface_atoms, center, radius)
    else:
        near_atoms = [a for a in surface_atoms if np.linalg.norm(a.position-center) < radius+4.0]
        for combo in combinations(near_atoms, 3):
            p = np.array([c.position for c in combo])
            # 连通性检查使用 MAX_BRIDGE_LENGTH，保证构成三角形的边都是有效的桥位
            if not check_triangle_connectivity(p[0], p[1], p[2], MAX_BRIDGE_LENGTH): continue 
            xy_c, xy_r = get_circumcenter_2d(p[0], p[1], p[2])
            if xy_c is not None and xy_r < MAX_HOLLOW_RADIUS: 
                base_pos = np.array([xy_c[0], xy_c[1], np.mean(p[:,2])])
                if np.linalg.norm(base_pos - center) <= radius:
                    raw_hollow.append({
                        'pos': base_pos, 'anchor': "Hollow", 
                        'constituent_atoms': [c.symbol for c in combo], 'dist_xy': xy_r
                    })

    # ---------------- 物理竞争与筛选 ----------------
    
    physically_valid_hollows = []
    raw_hollow.sort(key=lambda x: x['dist_xy']) 
    
    for h_cand in raw_hollow:
        keep = True
        # Top 覆盖检查
        for t_site in raw_top:
            if np.linalg.norm(h_cand['pos'][:2] - t_site['pos'][:2]) < HOLLOW_TO_TOP_MERGE:
                keep = False; break
        if not keep: continue

        # Bridge 覆盖检查
        for b_site in raw_bridge:
            if np.linalg.norm(h_cand['pos'][:2] - b_site['pos'][:2]) < HOLLOW_TO_BRIDGE_MERGE:
                keep = False; break
        if not keep: continue
        
        # Hollow 内部竞争
        for h_exist in physically_valid_hollows:
            if np.linalg.norm(h_cand['pos'][:2] - h_exist['pos'][:2]) < HOLLOW_DUPLICATE_DIST:
                keep = False; break 
        
        if keep:
            physically_valid_hollows.append(h_cand)

    # ---------------- 对称性去重 ----------------
    
    final_top = merge_sites_geometric(structure, raw_top, center)
    final_bridge = merge_sites_geometric(structure, raw_bridge, center)
    final_hollow = merge_sites_geometric(structure, physically_valid_hollows, center)
    
    return {'top': final_top, 'bridge': final_bridge, 'hollow': final_hollow}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Missing {INPUT_FILE}"); return
    
    print(f"Loading {INPUT_FILE}...")
    structure = read(INPUT_FILE)
    center = np.mean(structure.get_positions(), axis=0)
    
    base_ads, binding_sym = prepare_adsorbate_ultimate(ADSORBATE_INPUT, BINDING_ELEMENT, SEARCH_DIRECTION, TILT_ANGLE, ENABLE_RANDOM_ROTATION)
    if base_ads is None: return
    
    method_str = "Delaunay/Voronoi" if HAS_SCIPY else "Combinatorial"
    print(f"Generating sites using {method_str} method (Radius < {RADIUS_LIMIT} A)...")
    
    sites_dict = generate_sites_ultimate(structure, center, RADIUS_LIMIT, SEARCH_DIRECTION)
    
    if os.path.exists(OUTPUT_ROOT_DIR): shutil.rmtree(OUTPUT_ROOT_DIR)
    z_factor = 1.0 if SEARCH_DIRECTION == "top" else -1.0
    
    seen_configs = set()
    final_results = []
    name_counters = {} 

    for stype in ['top', 'bridge', 'hollow']:
        sites = sites_dict.get(stype, [])
        if not sites: continue
        
        sites.sort(key=lambda s: np.linalg.norm(s['pos'] - center))
        target_dir = os.path.join(OUTPUT_ROOT_DIR, stype)
        
        for sinfo in sites:
            h = calculate_smart_height(sinfo, binding_sym)
            atoms_list = sorted(sinfo['constituent_atoms'])
            comp_str = "-".join(atoms_list)
            
            pos_hash = (int(sinfo['pos'][0]*5), int(sinfo['pos'][1]*5))
            merge_key = (stype, comp_str, round(h, 2), pos_hash)
            if merge_key in seen_configs: continue 
            seen_configs.add(merge_key)
            
            if not os.path.exists(target_dir): os.makedirs(target_dir)
            
            type_suffix_map = {'top': 'top', 'bridge': 'bri', 'hollow': 'hol'}
            type_suffix = type_suffix_map.get(stype, stype)
            
            if len(atoms_list) > 3:
                unique_elements = list(set(atoms_list))
                if len(unique_elements) == 1:
                    short_comp = f"{unique_elements[0]}{len(atoms_list)}"
                else:
                    short_comp = "Multi" 
                base_name = f"{short_comp}_{type_suffix}"
            else:
                base_name = f"{comp_str}_{type_suffix}"
            
            if base_name not in name_counters:
                name_counters[base_name] = 1
                fname = f"{base_name}.cif"
            else:
                count = name_counters[base_name] + 1
                name_counters[base_name] = count
                fname = f"{base_name}_{count}.cif"

            final = structure.copy()
            ads = base_ads.copy()
            ads.translate([0,0, h*z_factor])
            ads.translate(sinfo['pos'])
            final += ads
            
            write(os.path.join(target_dir, fname), final, format="cif")
            
            final_results.append({
                'type': stype, 'composition': comp_str,
                'height': h, 'binding_atom': binding_sym, 'filename': fname
            })
            
    headers = ["ID", "Type", "Comp.", "H(A)", "Bind", "File"]
    row_fmt = "{:<2} | {:<7} | {:<30} | {:<5} | {:<4} | {:<30}"
    print("\n" + "="*85)
    print(f"{'Ultimate Results (Mode: ' + SELECT_MODE + ')':^85}")
    print("="*85)
    print(row_fmt.format(*headers))
    print("-" * 85)
    for i, res in enumerate(final_results):
        print(row_fmt.format(
            i + 1, res['type'].capitalize(), res['composition'],
            f"{res['height']:.2f}", res['binding_atom'], 
            res['filename']
        ))
    print("="*85 + "\n")

if __name__ == "__main__":
    main()

