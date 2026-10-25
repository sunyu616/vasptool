import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from ase.build import molecule
from itertools import combinations
import shutil
import random

# ================= 用户配置区域 (User Configuration) =================

INPUT_FILE = "POSCAR"          
SEARCH_DIRECTION = "top"       
RADIUS_LIMIT = 4.0             
SURFACE_THICKNESS = 1.5        # 表面层厚度判定阈值(Å)

# 吸附模式选择
SELECT_MODE = "H"             

# 吸附物配置库
ADSORPTION_SCHEMES = {
    "H":   {"molecule": "H",   "binding": "H", "scale": 0.9,  "tilt": 0,  "rotate": False},
    "O":   {"molecule": "O",   "binding": "O", "scale": 1.0,  "tilt": 0,  "rotate": False},
    "OH":  {"molecule": "OH",  "binding": "O", "scale": 1.05, "tilt": 15, "rotate": True},
    "OOH": {"molecule": "OOH", "binding": "O", "scale": 1.10, "tilt": 45, "rotate": True}
}

# 桥位与空位判定参数
MIN_BRIDGE_LENGTH = 1.5        
MAX_BRIDGE_HARD_LIMIT = 5.0    # 桥长的绝对物理硬上限，防止算法爆炸
DYNAMIC_BRIDGE_TOLERANCE = 1.15 # ★ 智能动态桥长容差系数 (吸附物-A键长 + 吸附物-B键长) * 1.15
MAX_BRIDGE_Z_DIFF = 1.0        

MAX_HOLLOW_RADIUS = 3.5 
MAX_HOLLOW_EDGE = 4.5          # 判定空位时的三角形最大边长限制       
COORDINATION_TOLERANCE = 0.5   

# 物理空间竞争阈值 (★全新优化的合并策略★)
CROSS_MERGE_TOL = 0.2          # 跨类型重叠阈值(Å)。优先级：Top > Hollow > Bridge

# 同类型核心去重算法参数 (支持 PBC 最小镜像)
SPATIAL_MERGE_TOL = 0.4        # 同类型空间去重阈值
ENV_MERGE_TOL = 0.15           # 指纹环境容差
ENV_CUTOFF = 4.5               

# ================= 配置区域结束 =================

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
# 自动化学直觉引擎 (Auto Chemical Intuition Engine)
# =========================================================================
def get_ideal_bond_length(sym1, sym2):
    non_metals = {'H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I'}
    
    nm, m = None, None
    if sym1 in non_metals and sym2 not in non_metals:
        nm, m = sym1, sym2
    elif sym2 in non_metals and sym1 not in non_metals:
        nm, m = sym2, sym1

    if nm and m:
        z = atomic_numbers.get(m, 0)
        is_metal = (21 <= z <= 30) or (39 <= z <= 48) or (57 <= z <= 80) or (89 <= z <= 112) or z in [13, 31, 49, 50, 81, 82, 83]
        
        if is_metal:
            empirical_bonds = {
                'O': 1.95, 'N': 1.90, 'C': 1.95, 'S': 2.30, 'H': 1.65, 
                'F': 1.90, 'Cl': 2.35, 'P': 2.25, 'Si': 2.30
            }
            base_bond = empirical_bonds.get(nm, 2.0)
            
            if 39 <= z <= 48:
                base_bond += 0.08
            elif 57 <= z <= 80:
                base_bond += 0.12
            elif z in [13, 31, 49, 50, 81, 82, 83]:
                base_bond += 0.15
            
            if z in [21, 22, 23, 39, 40, 41, 71, 72, 73]:
                base_bond -= 0.06
                
            return base_bond

    # 安全地从 numpy array 获取半径，修复 getattr 错误
    z1 = atomic_numbers.get(sym1, 0)
    z2 = atomic_numbers.get(sym2, 0)
    r1 = covalent_radii[z1] if z1 < len(covalent_radii) else 1.0
    r2 = covalent_radii[z2] if z2 < len(covalent_radii) else 1.0
    
    electronegativities = {
        'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
        'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
        'Y': 1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 2.16, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93,
        'Lu': 1.27, 'Hf': 1.30, 'Ta': 1.50, 'W': 2.36, 'Re': 1.90, 'Os': 2.20, 'Ir': 2.20, 'Pt': 2.28, 'Au': 2.54
    }
    en1 = electronegativities.get(sym1, 2.0)
    en2 = electronegativities.get(sym2, 2.0)
    
    correction = 0.09 * abs(en1 - en2)
    bond = r1 + r2 - correction
    return bond

def calculate_smart_height(site_info, binding_symbol):
    if 'recommended_height' in site_info:
        return site_info['recommended_height']

    bonds = [get_ideal_bond_length(s, binding_symbol) for s in site_info['constituent_atoms']]
    target_bond_length = np.mean(bonds) * BOND_SCALE_FACTOR
    dist_xy = site_info['dist_xy']
    
    if target_bond_length > dist_xy:
        height = np.sqrt(target_bond_length**2 - dist_xy**2)
    else:
        height = 0.5 
    return max(height, 0.5)

# =========================================================================
# 分子构造与表面几何模块
# =========================================================================
def calculate_bridge_position_and_height(a1, a2, binding_sym):
    sym1, sym2 = a1.symbol, a2.symbol
    p1, p2 = a1.position, a2.position
    
    L_vec = p2 - p1
    L = np.linalg.norm(L_vec)
    
    d1 = get_ideal_bond_length(sym1, binding_sym) * BOND_SCALE_FACTOR
    d2 = get_ideal_bond_length(sym2, binding_sym) * BOND_SCALE_FACTOR
    
    if sym1 == sym2 or L < 1e-3:
        bridge_pos = (p1 + p2) / 2.0
        dist_xy = L / 2.0
        actual_h2 = d1**2 - dist_xy**2
        height = np.sqrt(actual_h2) if actual_h2 > 0 else 0.5
        return bridge_pos, dist_xy, max(height, 0.5)
        
    x = (L**2 + d1**2 - d2**2) / (2 * L)
    ratio = x / L
    ratio = np.clip(ratio, 0.15, 0.85)  
    
    bridge_pos = p1 + ratio * L_vec
    actual_x = ratio * L
    actual_h2 = d1**2 - actual_x**2
    height = np.sqrt(actual_h2) if actual_h2 > 0 else 0.5
    
    dist_xy = max(actual_x, L - actual_x)
    return bridge_pos, dist_xy, max(height, 0.5)

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

def get_surface_atoms(structure, center, search_radius, direction="top"):
    z_coords = structure.positions[:, 2]
    
    if direction == "top":
        threshold = z_coords.max() - SURFACE_THICKNESS
        base_atoms = [atom for atom in structure if atom.position[2] >= threshold]
    else:
        threshold = z_coords.min() + SURFACE_THICKNESS
        base_atoms = [atom for atom in structure if atom.position[2] <= threshold]

    has_pbc = any(structure.pbc) if hasattr(structure, 'pbc') else bool(np.any(structure.cell))
    if not has_pbc or not np.any(structure.cell):
        return base_atoms
        
    expanded_atoms = []
    xy_seen = set()
    buffer_radius = search_radius + MAX_HOLLOW_RADIUS + 2.0
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            shift = dx * structure.cell[0] + dy * structure.cell[1]
            for a in base_atoms:
                new_pos = a.position + shift
                if np.linalg.norm(new_pos[:2] - center[:2]) <= buffer_radius:
                    xy = (round(new_pos[0], 3), round(new_pos[1], 3))
                    if xy not in xy_seen:
                        xy_seen.add(xy)
                        dummy_atom = Atoms(a.symbol, positions=[new_pos])[0]
                        expanded_atoms.append(dummy_atom)
                    
    return expanded_atoms

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

def get_safe_2d_atoms(surface_atoms, direction="top"):
    xy_seen = set()
    valid_atoms = []
    points_2d = []
    
    if direction == "top":
        sorted_atoms = sorted(surface_atoms, key=lambda a: -a.position[2])
    else:
        sorted_atoms = sorted(surface_atoms, key=lambda a: a.position[2])
        
    for a in sorted_atoms:
        xy = (round(a.position[0], 4), round(a.position[1], 4))
        if xy not in xy_seen:
            xy_seen.add(xy)
            valid_atoms.append(a)
            points_2d.append(a.position[:2])
    return valid_atoms, np.array(points_2d)

def get_bridge_sites_delaunay(surface_atoms, center, radius_limit, binding_sym, direction="top"):
    if not HAS_SCIPY or len(surface_atoms) < 3: return []
    valid_atoms, points_2d = get_safe_2d_atoms(surface_atoms, direction)
    if len(points_2d) < 3: return []
    
    try:
        tri = Delaunay(points_2d)
    except Exception:
        return []
        
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[0], simplex[2]))))
    
    candidates = []
    for (i, j) in edges:
        a1, a2 = valid_atoms[i], valid_atoms[j]
        dist_3d = np.linalg.norm(a1.position - a2.position)
        
        # ★ 动态桥长判定：针对当前 A 和 B 元素，实时计算物理允许的极限跨度
        d1 = get_ideal_bond_length(a1.symbol, binding_sym) * BOND_SCALE_FACTOR
        d2 = get_ideal_bond_length(a2.symbol, binding_sym) * BOND_SCALE_FACTOR
        dynamic_max_bridge = min((d1 + d2) * DYNAMIC_BRIDGE_TOLERANCE, MAX_BRIDGE_HARD_LIMIT)
        
        if not (MIN_BRIDGE_LENGTH < dist_3d < dynamic_max_bridge): continue
        if abs(a1.position[2] - a2.position[2]) > MAX_BRIDGE_Z_DIFF: continue
            
        bridge_pos, dist_xy, rec_h = calculate_bridge_position_and_height(a1, a2, binding_sym)
        if np.linalg.norm(bridge_pos[:2] - center[:2]) <= radius_limit:
            candidates.append({
                'pos': bridge_pos, 
                'anchor': f"{a1.symbol}-{a2.symbol}", 
                'constituent_atoms': [a1.symbol, a2.symbol], 
                'dist_xy': dist_xy,
                'recommended_height': rec_h
            })
    return candidates

def get_hollow_sites_voronoi(surface_atoms, center, radius_limit, direction="top"):
    if not HAS_SCIPY or len(surface_atoms) < 3: return []
    valid_atoms, points_2d = get_safe_2d_atoms(surface_atoms, direction)
    if len(points_2d) < 3: return []
    
    try:
        vor = Voronoi(points_2d)
    except Exception:
        return []
        
    candidates = []
    for v_idx, vertex in enumerate(vor.vertices):
        if np.linalg.norm(vertex - center[:2]) > radius_limit: continue
        dists = np.linalg.norm(points_2d - vertex, axis=1)
        min_dist = np.min(dists)
        
        if min_dist > MAX_HOLLOW_RADIUS: continue
        
        neighbor_indices = np.where(dists < min_dist + COORDINATION_TOLERANCE)[0]
        if len(neighbor_indices) < 3: continue 
        
        neighbors = [valid_atoms[i] for i in neighbor_indices]
        avg_z = np.mean([a.position[2] for a in neighbors])
        full_pos = np.array([vertex[0], vertex[1], avg_z])
        
        if np.linalg.norm(full_pos[:2] - center[:2]) <= radius_limit:
            candidates.append({'pos': full_pos, 'anchor': "Voronoi", 'constituent_atoms': [a.symbol for a in neighbors], 'dist_xy': min_dist})
    return candidates

def merge_duplicate_sites(structure, sites, center, spatial_tol=0.5, env_tol=0.15):
    if not sites: return []
    has_pbc = any(structure.pbc) if hasattr(structure, 'pbc') else bool(np.any(structure.cell))
        
    def get_site_fingerprint(pos):
        dummy = structure.copy()
        dummy += Atoms('He', positions=[pos])
        he_index = len(dummy) - 1
        dists = dummy.get_distances(he_index, range(len(structure)), mic=has_pbc)
        mask = dists < ENV_CUTOFF
        near_atoms = [(structure[i].symbol, dists[i]) for i in np.where(mask)[0]]
        fp = {}
        for sym, d in near_atoms:
            if sym not in fp: fp[sym] = []
            fp[sym].append(d)
        for sym in fp:
            fp[sym] = np.sort(fp[sym])
        return fp
        
    def is_same_environment(fp1, fp2):
        if fp1.keys() != fp2.keys(): return False
        for sym in fp1:
            d1, d2 = fp1[sym], fp2[sym]
            min_len = min(len(d1), len(d2))
            if min_len == 0:
                if len(d1) > 0 or len(d2) > 0: return False
                continue
            if np.any(np.abs(d1[:min_len] - d2[:min_len]) > env_tol): return False
        return True

    unique_sites = []
    sorted_sites = sorted(sites, key=lambda x: np.linalg.norm(x['pos'][:2] - center[:2]))
    
    for site in sorted_sites:
        pos = site['pos']
        fp = get_site_fingerprint(pos)
        is_duplicate = False
        for u_site in unique_sites:
            if np.linalg.norm(pos - u_site['pos']) < spatial_tol:
                is_duplicate = True; break
            if is_same_environment(fp, u_site['fp']):
                is_duplicate = True; break
        if not is_duplicate:
            site['fp'] = fp
            unique_sites.append(site)
            
    for s in unique_sites: s.pop('fp', None)
    return unique_sites

def generate_sites_ultimate(structure, center, radius=3.0, direction="top", binding_sym="H"):
    surface_atoms = get_surface_atoms(structure, center, radius, direction=direction)
    raw_top, raw_bridge, raw_hollow = [], [], []
    
    for a in surface_atoms:
        if np.linalg.norm(a.position[:2] - center[:2]) <= radius:
            raw_top.append({'pos': a.position, 'anchor': f"{a.symbol}", 'constituent_atoms': [a.symbol], 'dist_xy': 0.0})

    bridge_done = False
    hollow_done = False
    if HAS_SCIPY:
        scipy_bridge = get_bridge_sites_delaunay(surface_atoms, center, radius, binding_sym, direction)
        scipy_hollow = get_hollow_sites_voronoi(surface_atoms, center, radius, direction)
        if scipy_bridge:
            raw_bridge = scipy_bridge
            bridge_done = True
        if scipy_hollow:
            raw_hollow = scipy_hollow
            hollow_done = True

    # 兜底：组合学算法
    if not bridge_done:
        for i, j in combinations(range(len(surface_atoms)), 2):
            a1, a2 = surface_atoms[i], surface_atoms[j]
            dist_3d = np.linalg.norm(a1.position - a2.position)
            z_diff = abs(a1.position[2] - a2.position[2])
            
            # ★ 动态桥长判定 (兜底算法中同样生效)
            d1 = get_ideal_bond_length(a1.symbol, binding_sym) * BOND_SCALE_FACTOR
            d2 = get_ideal_bond_length(a2.symbol, binding_sym) * BOND_SCALE_FACTOR
            dynamic_max_bridge = min((d1 + d2) * DYNAMIC_BRIDGE_TOLERANCE, MAX_BRIDGE_HARD_LIMIT)
            
            if (MIN_BRIDGE_LENGTH < dist_3d < dynamic_max_bridge) and (z_diff < MAX_BRIDGE_Z_DIFF): 
                bridge_pos, dist_xy, rec_h = calculate_bridge_position_and_height(a1, a2, binding_sym)
                if np.linalg.norm(bridge_pos[:2] - center[:2]) <= radius:
                    raw_bridge.append({
                        'pos': bridge_pos, 
                        'anchor': f"{a1.symbol}-{a2.symbol}", 
                        'constituent_atoms': [a1.symbol, a2.symbol], 
                        'dist_xy': dist_xy,
                        'recommended_height': rec_h
                    })

    if not hollow_done:
        near_atoms = [a for a in surface_atoms if np.linalg.norm(a.position[:2]-center[:2]) < radius+4.0]
        for combo in combinations(near_atoms, 3):
            p = np.array([c.position for c in combo])
            if not check_triangle_connectivity(p[0], p[1], p[2], MAX_HOLLOW_EDGE): continue 
            xy_c, xy_r = get_circumcenter_2d(p[0], p[1], p[2])
            if xy_c is not None and xy_r < MAX_HOLLOW_RADIUS: 
                base_pos = np.array([xy_c[0], xy_c[1], np.mean(p[:,2])])
                if np.linalg.norm(base_pos[:2] - center[:2]) <= radius:
                    raw_hollow.append({'pos': base_pos, 'anchor': "Hollow", 'constituent_atoms': [c.symbol for c in combo], 'dist_xy': xy_r})

    # ================= 核心跨类型裁决引擎优化 =================
    valid_bridges = []
    for b in raw_bridge:
        if not any(np.linalg.norm(b['pos'][:2] - t['pos'][:2]) < CROSS_MERGE_TOL for t in raw_top):
            valid_bridges.append(b)

    valid_hollows = []
    for h in raw_hollow:
        if not any(np.linalg.norm(h['pos'][:2] - t['pos'][:2]) < CROSS_MERGE_TOL for t in raw_top):
            valid_hollows.append(h)
            
    valid_hollows.sort(key=lambda x: x['dist_xy'])

    surviving_bridges = []
    for b in valid_bridges:
        if not any(np.linalg.norm(b['pos'][:2] - h['pos'][:2]) < CROSS_MERGE_TOL for h in valid_hollows):
            surviving_bridges.append(b)

    final_top = merge_duplicate_sites(structure, raw_top, center, SPATIAL_MERGE_TOL, ENV_MERGE_TOL)
    final_bridge = merge_duplicate_sites(structure, surviving_bridges, center, SPATIAL_MERGE_TOL, ENV_MERGE_TOL)
    final_hollow = merge_duplicate_sites(structure, valid_hollows, center, SPATIAL_MERGE_TOL, ENV_MERGE_TOL)
    
    return {'top': final_top, 'bridge': final_bridge, 'hollow': final_hollow}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Missing {INPUT_FILE}"); return
    
    print(f"Loading {INPUT_FILE}...")
    structure = read(INPUT_FILE)
    
    if np.any(structure.cell):
        cell_center_xy = np.dot([0.5, 0.5, 0], structure.cell)[:2]
        center_z = np.mean(structure.get_positions()[:, 2]) 
        center = np.array([cell_center_xy[0], cell_center_xy[1], center_z])
    else:
        center = np.mean(structure.get_positions(), axis=0)
    
    base_ads, binding_sym = prepare_adsorbate_ultimate(ADSORBATE_INPUT, BINDING_ELEMENT, SEARCH_DIRECTION, TILT_ANGLE, ENABLE_RANDOM_ROTATION)
    if base_ads is None: return
    
    method_str = "Delaunay/Voronoi Auto-Fallback" if HAS_SCIPY else "Combinatorial (No Scipy)"
    print(f"Generating sites using {method_str} method (Radius < {RADIUS_LIMIT} A)...")
    print(f"[*] Dynamic Bridge Engine ENABLED: Auto-adjusting limits for {binding_sym} adsorbate.")
    
    sites_dict = generate_sites_ultimate(structure, center, RADIUS_LIMIT, SEARCH_DIRECTION, binding_sym)
    
    if os.path.exists(OUTPUT_ROOT_DIR): shutil.rmtree(OUTPUT_ROOT_DIR)
    z_factor = 1.0 if SEARCH_DIRECTION == "top" else -1.0
    
    final_results = []
    name_counters = {} 

    for stype in ['top', 'bridge', 'hollow']:
        sites = sites_dict.get(stype, [])
        if not sites: continue
        
        target_dir = os.path.join(OUTPUT_ROOT_DIR, stype)
        if not os.path.exists(target_dir): os.makedirs(target_dir)
            
        for sinfo in sites:
            h = calculate_smart_height(sinfo, binding_sym)
            atoms_list = sorted(sinfo['constituent_atoms'])
            comp_str = "-".join(atoms_list)
            
            type_suffix_map = {'top': 'top', 'bridge': 'bri', 'hollow': 'hol'}
            type_suffix = type_suffix_map.get(stype, stype)
            
            if len(atoms_list) > 3:
                unique_elements = list(set(atoms_list))
                short_comp = f"{unique_elements[0]}{len(atoms_list)}" if len(unique_elements) == 1 else "Multi" 
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
            
            mark = "*" if stype == 'bridge' and len(atoms_list) == 2 and atoms_list[0] != atoms_list[1] else ""
            
            final_results.append({
                'type': stype, 'composition': comp_str + mark,
                'height': h, 'binding_atom': binding_sym, 'filename': fname
            })
            
    final_results.sort(key=lambda x: (x['type'], x['composition'], x['height']))

    headers = ["ID", "Type", "Comp.", "H(A)", "Bind", "File"]
    row_fmt = "{:<3} | {:<8} | {:<25} | {:<5} | {:<4} | {:<30}"
    print("\n" + "="*85)
    print(f"{'Ultimate Results (Auto Engine) | Mode: ' + SELECT_MODE:^85}")
    print("="*85)
    print(row_fmt.format(*headers))
    print("-" * 85)
    for i, res in enumerate(final_results):
        print(row_fmt.format(
            i + 1, res['type'].capitalize(), res['composition'],
            f"{res['height']:.2f}", res['binding_atom'], 
            res['filename']
        ))
    print("="*85)
    print("Note: '*' in Comp. indicates an optimized Hetero-nuclear bridge site.")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()



