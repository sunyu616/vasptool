import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from ase.geometry import find_mic
from ase.neighborlist import NeighborList
import shutil

# =================================================================
# 🛠️ 极简控制面板 (User Settings)
# =================================================================
INPUT_FILE = "POSCAR"          # 表面结构文件 (POSCAR/CONTCAR/cif 等)
ADSORBATE  = "H"               # 吸附原子元素符号
MAX_ATOMS  = 5                 # 最大吸附原子数
SITES      = ["top", "bri", "hol"] # 目标吸附位点 (支持: "top", "bri", "hol")

ADSORB_MODE = "symmetric"      # 吸附模式: 
                               #  - "symmetric" : 双面交替/镜像吸附 (抵消偶极，推荐)
                               #  - "top"       : 仅在顶层面 (Top) 吸附
                               #  - "bottom"    : 仅在底层面 (Bottom) 吸附

MIN_DIST   = 2.0               # 物理安全红线(Å)，低于此距离强制停止生长
REPULSION_POWER = 3            # 物理斥力衰减指数 (3: 偶极排斥, 推荐)

# ============ 高级：上H吸附优化参数 ============
H_OPTIMIZATION = {
    "use_effective_coordination": True,   # 使用有效配位数修正高度
    "coordination_cutoff": 1.15,          # 配位数计算的截断因子
    "height_scaling_factor": 0.95,        # 基础键长缩放因子 (源自slab.py的经验值)
    "prefer_hollow_adjacent": True,       # 优先选择邻近hollow位的top位点
    "hollow_proximity_weight": 0.3,       # hollow邻近度在评分中的权重
    "surface_normal_relaxation": True,    # 考虑表面法向弛豫修正
    "layer_threshold": 0.5,               # 层间距阈值(Å)，用于识别表面层
}

# ============ 底层：几何与物理容差参数 (来自 slab.py) ============
MIN_BRIDGE_LENGTH = 1.5        
MAX_BRIDGE_HARD_LIMIT = 5.0    
DYNAMIC_BRIDGE_TOLERANCE = 1.15 
MAX_BRIDGE_Z_DIFF = 1.0        
MAX_HOLLOW_RADIUS = 3.5 
COORDINATION_TOLERANCE = 0.5   
CROSS_MERGE_TOL = 0.2          

OUTPUT_DIR = f"Coverage_{ADSORBATE}_{ADSORB_MODE.capitalize()}_Ultimate"

try:
    from scipy.spatial import Voronoi, Delaunay
    HAS_SCIPY = True
except ImportError:
    print("⚠️ 警告: 未安装 scipy，将无法寻找 bridge 和 hollow 位点。建议: pip install scipy")
    HAS_SCIPY = False

# =========================================================================
# 🧬 核心组件 1: 自动化学直觉引擎 (来自 slab.py)
# =========================================================================
def get_ideal_bond_length(sym1, sym2):
    non_metals = {'H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I'}
    nm, m = None, None
    if sym1 in non_metals and sym2 not in non_metals: nm, m = sym1, sym2
    elif sym2 in non_metals and sym1 not in non_metals: nm, m = sym2, sym1

    if nm and m:
        z = atomic_numbers.get(m, 0)
        is_metal = (21 <= z <= 30) or (39 <= z <= 48) or (57 <= z <= 80) or (89 <= z <= 112) or z in [13, 31, 49, 50, 81, 82, 83]
        if is_metal:
            empirical_bonds = {'O': 1.95, 'N': 1.90, 'C': 1.95, 'S': 2.30, 'H': 1.65, 'F': 1.90, 'Cl': 2.35, 'P': 2.25, 'Si': 2.30}
            base_bond = empirical_bonds.get(nm, 2.0)
            if 39 <= z <= 48: base_bond += 0.08
            elif 57 <= z <= 80: base_bond += 0.12
            elif z in [13, 31, 49, 50, 81, 82, 83]: base_bond += 0.15
            if z in [21, 22, 23, 39, 40, 41, 71, 72, 73]: base_bond -= 0.06
            return base_bond

    z1 = atomic_numbers.get(sym1, 0)
    z2 = atomic_numbers.get(sym2, 0)
    r1 = covalent_radii[z1] if z1 < len(covalent_radii) else 1.0
    r2 = covalent_radii[z2] if z2 < len(covalent_radii) else 1.0
    
    electronegativities = {
        'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
        'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
        'Y': 1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 2.16, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93,
        'Pt': 2.28, 'Au': 2.54 
    }
    en1 = electronegativities.get(sym1, 2.0)
    en2 = electronegativities.get(sym2, 2.0)
    return r1 + r2 - 0.09 * abs(en1 - en2)

def mic_dist(p1, p2, cell, pbc):
    v = np.array(p1) - np.array(p2)
    if v.shape[-1] == 2: v = np.append(v, 0.0) 
    if not any(pbc): return np.linalg.norm(v)
    _, d_len = find_mic(np.atleast_2d(v), cell, pbc)
    return d_len[0]

# =========================================================================
# 🧬 核心组件 2: 动态高度与修饰系统 (融合版)
# =========================================================================
def get_effective_coordination(atom_pos, neighbor_positions, atom_symbol, neighbor_symbols, cutoff_factor=1.15):
    if len(neighbor_positions) == 0: return 0.0
    r_atom = covalent_radii[atomic_numbers[atom_symbol]]
    effective_coord = 0.0
    for i, npos in enumerate(neighbor_positions):
        r_neigh = covalent_radii[atomic_numbers[neighbor_symbols[i]]]
        r_ideal = (r_atom + r_neigh) * cutoff_factor
        dist = np.linalg.norm(np.array(atom_pos) - np.array(npos))
        weight = 1.0 / (1.0 + np.exp((dist - r_ideal) / 0.1)) # Fermi smoothing
        effective_coord += weight
    return effective_coord

def calculate_optimized_height(site_info, binding_symbol, structure, h_params):
    """基于化学直觉计算基础高度，再施加覆盖度脚本的配位数与弛豫修正"""
    if 'recommended_height' in site_info:
        base_height = site_info['recommended_height']
    else:
        # 使用 slab.py 的精确理想键长
        bonds = [get_ideal_bond_length(s, binding_symbol) for s in site_info['atoms']]
        target_bond_length = np.mean(bonds) * h_params["height_scaling_factor"]
        dist_xy = site_info['dist_xy']
        base_height = np.sqrt(target_bond_length**2 - dist_xy**2) if target_bond_length > dist_xy else 1.0

    # 优化1: 有效配位数修正
    if h_params["use_effective_coordination"] and 'effective_coordination' in site_info:
        coord = site_info['effective_coordination']
        if len(site_info['atoms']) == 1:   coord_factor = 1.0 - 0.05 * (9 - coord) # Top
        elif len(site_info['atoms']) == 2: coord_factor = 1.02 # Bridge
        else:                              coord_factor = 0.98 # Hollow
        base_height *= coord_factor
    
    # 优化2: 表面法向弛豫修正
    if h_params["surface_normal_relaxation"]:
        z_coords = structure.positions[:, 2]
        is_top_surf = abs(site_info['pos'][2] - z_coords.max()) < h_params["layer_threshold"]
        is_bot_surf = abs(site_info['pos'][2] - z_coords.min()) < h_params["layer_threshold"]
        if is_top_surf or is_bot_surf:
            base_height += 0.05 
            
    return max(base_height, 0.5)

# =========================================================================
# 🧬 核心组件 3: 极致几何位点测绘 (Delaunay/Voronoi + 动态过滤)
# =========================================================================
def calculate_bridge_position(a1, a2, binding_sym, scale_factor):
    p1, p2 = a1.position, a2.position
    L_vec = p2 - p1
    L = np.linalg.norm(L_vec)
    d1 = get_ideal_bond_length(a1.symbol, binding_sym) * scale_factor
    d2 = get_ideal_bond_length(a2.symbol, binding_sym) * scale_factor
    
    if a1.symbol == a2.symbol or L < 1e-3:
        actual_h2 = d1**2 - (L/2.0)**2
        return (p1 + p2) / 2.0, L / 2.0, max(np.sqrt(actual_h2) if actual_h2 > 0 else 0.5, 0.5)
        
    x = (L**2 + d1**2 - d2**2) / (2 * L)
    ratio = np.clip(x / L, 0.15, 0.85)  
    actual_x = ratio * L
    actual_h2 = d1**2 - actual_x**2
    return p1 + ratio * L_vec, max(actual_x, L - actual_x), max(np.sqrt(actual_h2) if actual_h2 > 0 else 0.5, 0.5)

def get_sites_ultimate(structure, h_params):
    z_coords = structure.positions[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    
    # 提取顶底最外层 (考虑层间距阈值)
    top_atoms = [a for a in structure if a.position[2] >= z_max - 1.5]
    bot_atoms = [a for a in structure if a.position[2] <= z_min + 1.5]
    
    cutoffs = [covalent_radii[atomic_numbers[a.symbol]] * 2.5 for a in structure]
    nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(structure)
    
    def process_layer(layer_atoms, side_label):
        raw_top, raw_bridge, raw_hollow = [], [], []
        if not layer_atoms: return {'top': [], 'bridge': [], 'hollow': []}
        
        layer_indices = [a.index for a in layer_atoms]
        
        for a in layer_atoms:
            indices, offsets = nl.get_neighbors(a.index)
            n_pos, n_sym = [], []
            for idx, offset in zip(indices, offsets):
                if idx in layer_indices:  
                    n_pos.append(structure.positions[idx] + np.dot(offset, structure.cell))
                    n_sym.append(structure[idx].symbol)
            
            eff_coord = get_effective_coordination(a.position, n_pos, a.symbol, n_sym, h_params["coordination_cutoff"])
            raw_top.append({'pos': a.position, 'atoms': [a.symbol], 'dist_xy': 0.0, 'side': side_label, 'effective_coordination': eff_coord})

        if HAS_SCIPY and len(layer_atoms) >= 3:
            # 建立基于当前层中心的周期性扩展边界 (防止边界丢失位点)
            points_2d = np.array([a.position[:2] for a in layer_atoms])
            
            # --- Bridge via Delaunay ---
            try:
                tri = Delaunay(points_2d)
                edges = set()
                for s in tri.simplices:
                    edges.update([tuple(sorted((s[0], s[1]))), tuple(sorted((s[1], s[2]))), tuple(sorted((s[0], s[2])))])
                
                for (i, j) in edges:
                    a1, a2 = layer_atoms[i], layer_atoms[j]
                    dist_3d = np.linalg.norm(a1.position - a2.position)
                    d1 = get_ideal_bond_length(a1.symbol, ADSORBATE) * h_params["height_scaling_factor"]
                    d2 = get_ideal_bond_length(a2.symbol, ADSORBATE) * h_params["height_scaling_factor"]
                    dynamic_max = min((d1 + d2) * DYNAMIC_BRIDGE_TOLERANCE, MAX_BRIDGE_HARD_LIMIT)
                    
                    if (MIN_BRIDGE_LENGTH < dist_3d < dynamic_max) and abs(a1.position[2] - a2.position[2]) < MAX_BRIDGE_Z_DIFF:
                        b_pos, d_xy, rec_h = calculate_bridge_position(a1, a2, ADSORBATE, h_params["height_scaling_factor"])
                        raw_bridge.append({'pos': b_pos, 'atoms': [a1.symbol, a2.symbol], 'dist_xy': d_xy, 'side': side_label, 'recommended_height': rec_h})
            except Exception as e: print(f"Bridge calc error: {e}")

            # --- Hollow via Voronoi ---
            try:
                vor = Voronoi(points_2d)
                for vertex in vor.vertices:
                    dists = np.linalg.norm(points_2d - vertex, axis=1)
                    min_dist = np.min(dists)
                    if min_dist > MAX_HOLLOW_RADIUS: continue 
                    neighbors = [layer_atoms[i] for i in np.where(dists < min_dist + COORDINATION_TOLERANCE)[0]]
                    if len(neighbors) >= 3:
                        avg_z = np.mean([a.position[2] for a in neighbors])
                        raw_hollow.append({'pos': np.array([vertex[0], vertex[1], avg_z]), 'atoms': [a.symbol for a in neighbors], 'dist_xy': min_dist, 'side': side_label})
            except: pass

        # 空间排他性过滤 (去除伪位点)
        valid_top = raw_top
        valid_bridge = [b for b in raw_bridge if not any(mic_dist(b['pos'][:2], t['pos'][:2], structure.cell, structure.pbc) < CROSS_MERGE_TOL for t in valid_top)]
        valid_hollow = [h for h in raw_hollow if not any(mic_dist(h['pos'][:2], t['pos'][:2], structure.cell, structure.pbc) < CROSS_MERGE_TOL for t in valid_top)]
        
        # 桥位与空位互斥
        surviving_bridges = [b for b in valid_bridge if not any(mic_dist(b['pos'][:2], h['pos'][:2], structure.cell, structure.pbc) < CROSS_MERGE_TOL for h in valid_hollow)]

        return {'top': valid_top, 'bridge': surviving_bridges, 'hollow': valid_hollow}

    top_sites = process_layer(top_atoms, 'top')
    bot_sites = process_layer(bot_atoms, 'bot')
    
    # 增加 Top 位点的 Hollow 邻近属性
    if h_params["prefer_hollow_adjacent"]:
        all_hollows = top_sites['hollow'] + bot_sites['hollow']
        for top_list in [top_sites['top'], bot_sites['top']]:
            for t in top_list:
                adjacent = []
                for h in all_hollows:
                    d = mic_dist(t['pos'][:2], h['pos'][:2], structure.cell, structure.pbc)
                    if d < 2.0: adjacent.append(d)
                t['hollow_proximity_score'] = 1.0 / (1.0 + min(adjacent)) if adjacent else 0.0
    
    return {
        'top': top_sites['top'] + bot_sites['top'],
        'bridge': top_sites['bridge'] + bot_sites['bridge'],
        'hollow': top_sites['hollow'] + bot_sites['hollow']
    }

# =========================================================================
# 🧬 核心组件 4: 势场全局演化算法 (来自 coverage.py，不变且极强)
# =========================================================================
def generate_global_best_sequential(sites, max_steps, min_dist, cell, pbc, mode, rep_power, h_params):
    history, occupied = [], []
    top_sites = [i for i, s in enumerate(sites) if s['side'] == 'top']
    bot_sites = [i for i, s in enumerate(sites) if s['side'] == 'bot']
    
    if mode == "top": bot_sites = []
    elif mode == "bottom": top_sites = []
    elif mode == "symmetric":
        if top_sites and not bot_sites: print("     ⚠️ 降级为【仅Top】")
        elif bot_sites and not top_sites: print("     ⚠️ 降级为【仅Bottom】")
            
    if not top_sites and not bot_sites: return []
        
    valid_starts = top_sites if top_sites else bot_sites
    center_xy = np.mean([sites[i]['pos'][:2] for i in valid_starts], axis=0)

    for step in range(1, max_steps + 1):
        curr_top = sum(1 for i in occupied if sites[i]['side'] == 'top')
        curr_bot = sum(1 for i in occupied if sites[i]['side'] == 'bot')
        
        if mode == "symmetric" and top_sites and bot_sites: target_side = 'bot' if curr_top > curr_bot else 'top'
        else: target_side = 'top' if top_sites else 'bot'
            
        candidate_pool = top_sites if target_side == 'top' else bot_sites
        same_side_occ = [occ for occ in occupied if sites[occ]['side'] == target_side]
        other_side_occ = [occ for occ in occupied if sites[occ]['side'] == ('top' if target_side == 'bot' else 'bot')]
        
        is_mirroring = (mode == "symmetric" and len(other_side_occ) > len(same_side_occ))
        target_mirror_idx = other_side_occ[len(same_side_occ)] if is_mirroring else -1
            
        best_idx, best_eval_score = -1, -float('inf')
        
        for i in candidate_pool:
            if i in occupied: continue
            s = sites[i]
            
            if is_mirroring:
                d = mic_dist(s['pos'][:2], sites[target_mirror_idx]['pos'][:2], cell, pbc)
                eval_score = -d
                if h_params["prefer_hollow_adjacent"] and 'hollow_proximity_score' in s:
                    eval_score += h_params["hollow_proximity_weight"] * s['hollow_proximity_score']
            elif not same_side_occ:
                d = mic_dist(s['pos'][:2], center_xy, cell, pbc)
                eval_score = -d
                if 'effective_coordination' in s:
                    eval_score += 0.2 * (1.0 - abs(s['effective_coordination'] - 9.0) / 9.0)
                if h_params["prefer_hollow_adjacent"] and 'hollow_proximity_score' in s:
                    eval_score += h_params["hollow_proximity_weight"] * s['hollow_proximity_score']
            else:
                distances = [mic_dist(s['pos'][:2], sites[occ_i]['pos'][:2], cell, pbc) for occ_i in same_side_occ]
                eval_score = -sum([1.0 / ((d + 1e-4) ** rep_power) for d in distances])
                
                if 'effective_coordination' in s:
                    eval_score += 0.15 * (1.0 - s['effective_coordination'] / 12.0)
                if h_params["prefer_hollow_adjacent"] and 'hollow_proximity_score' in s:
                    eval_score += h_params["hollow_proximity_weight"] * s['hollow_proximity_score']
                if len(same_side_occ) >= 2:
                    local_density = sum([1.0 / (mic_dist(s['pos'][:2], sites[occ_i]['pos'][:2], cell, pbc) + 1e-4) for occ_i in same_side_occ])
                    eval_score -= 0.1 * local_density 
                
            if eval_score > best_eval_score: best_eval_score, best_idx = eval_score, i
                
        if best_idx == -1: break
        if same_side_occ:
            real_min_dist = min([mic_dist(sites[best_idx]['pos'][:2], sites[occ_i]['pos'][:2], cell, pbc) for occ_i in same_side_occ])
            if real_min_dist < min_dist: break
            
        occupied.append(best_idx)
        history.append(list(occupied))
        
        side_str = "Top" if target_side == 'top' else "Bottom"
        extra = f" | 配位数: {sites[best_idx].get('effective_coordination', 0):.1f}" if 'effective_coordination' in sites[best_idx] else ""
        if is_mirroring: print(f"   ✔️ 步 {step:02d}: {ADSORBATE} -> {side_str:6} [镜像对齐]{extra}")
        elif not same_side_occ: print(f"   ✔️ 步 {step:02d}: {ADSORBATE} -> {side_str:6} [结构中心]{extra}")
        else: print(f"   ✔️ 步 {step:02d}: {ADSORBATE} -> {side_str:6} [势能极小位]{extra}")
            
    return history

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}"); return
        
    structure = read(INPUT_FILE)
    print(f"\n==================================================")
    print(f"💎 终极版覆盖度演化器 (Chemical Intuition + Field Evolution)")
    print(f"==================================================")
    
    sites_dict = get_sites_ultimate(structure, H_OPTIMIZATION)
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    
    type_map = {"top": "top", "bri": "bridge", "hol": "hollow"}
    
    for req_type in SITES:
        actual_type = type_map.get(req_type.lower()[:3])
        if not actual_type: continue
        base_sites = sites_dict.get(actual_type, [])
        if not base_sites: continue
        
        grouped = {}
        for s in base_sites:
            comp = "-".join(sorted(s['atoms']))
            if comp not in grouped: grouped[comp] = []
            grouped[comp].append(s)
            
        for comp_str, specific_sites in grouped.items():
            print(f"\n🎯 演化通道: [{actual_type.upper()}] - {comp_str} (发现可用位点 {len(specific_sites)} 个)")
            target_dir = os.path.join(OUTPUT_DIR, f"{actual_type}_{comp_str}")
            os.makedirs(target_dir, exist_ok=True)
            
            history = generate_global_best_sequential(
                specific_sites, MAX_ATOMS, MIN_DIST, 
                structure.cell, structure.pbc, mode=ADSORB_MODE.lower(), 
                rep_power=REPULSION_POWER, h_params=H_OPTIMIZATION
            )
            
            for step_idx, combo in enumerate(history):
                step = step_idx + 1
                final = structure.copy()
                base_ads = Atoms(ADSORBATE, positions=[(0, 0, 0)])
                top_count, bot_count = 0, 0
                
                for site_idx in combo:
                    sinfo = specific_sites[site_idx]
                    h = calculate_optimized_height(sinfo, ADSORBATE, structure, H_OPTIMIZATION)
                    ads = base_ads.copy()
                    
                    if sinfo['side'] == 'top': ads.translate([0, 0, h]); top_count += 1
                    else: ads.translate([0, 0, -h]); bot_count += 1
                        
                    ads.translate(sinfo['pos'])
                    final += ads
                
                fname = f"cov_{step:02d}{ADSORBATE}_{top_count}T_{bot_count}B_opt.cif"
                write(os.path.join(target_dir, fname), final, format="cif")
                
                if step == 1: print(f"   📏 智能基准高度: {h:.3f} Å")

    print(f"\n🎉 演化完毕！高精度构型已存储于: ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    main()

