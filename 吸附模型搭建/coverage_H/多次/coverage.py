import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from ase.geometry import find_mic
from ase.neighborlist import NeighborList
from itertools import combinations
import shutil
import math

# =================================================================
# 🛠️ 实验级控制面板 (User Settings)
# =================================================================
INPUT_FILE = "POSCAR"          # 你的输入文件 (干净表面, 或已含有H的优化后结构)
ADSORBATE  = "H"               # 吸附物
N_ADD      = 1                 # 本次想【新增】几个H原子 
TOP_K      = 5                 # 输出排名前K个【能量评分最高且结构不同】的候选构型

SITES      = ["top"]  # 允许吸附的位点池

# --- 🎯 核心新增：吸附面控制 ---
ADSORB_MODE = "both"            # 支持模式:
                               #  - "top"    : 仅顶面吸附 (常规单面研究)
                               #  - "bottom" : 仅底面吸附
                               #  - "both"   : 双面同时吸附 (自动寻找最优的双面排布)

# --- 物理约束与评分参数 ---
MIN_H_H_DIST = 2.0             # H-H之间的绝对安全距离下限(Å)
REPULSION_POWER = 3            # 势场排斥衰减 (3: 偶极排斥)
DIVERSITY_TOL = 0.5            # 构型多样性过滤阈值(Å)

OUTPUT_DIR = f"Candidates_{ADSORBATE}_Add{N_ADD}_{ADSORB_MODE.capitalize()}"

# 屏蔽Scipy警告，保持输出整洁
try:
    from scipy.spatial import Voronoi, Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ 建议安装 scipy 以启用高级几何位点分析 (pip install scipy)")

# ==================== 底层物理引擎 ====================
def mic_dist(p1, p2, cell, pbc):
    v = np.array(p1) - np.array(p2)
    if v.shape[-1] == 2: v = np.append(v, 0.0) 
    if not any(pbc): return np.linalg.norm(v)
    _, d_len = find_mic(np.atleast_2d(v), cell, pbc)
    return d_len[0]

def get_ideal_bond_length(sym1, sym2):
    r1 = covalent_radii[atomic_numbers.get(sym1, 0)]
    r2 = covalent_radii[atomic_numbers.get(sym2, 0)]
    en1 = 2.20 if sym1 == 'H' else 2.0
    en2 = 2.20 if sym2 == 'H' else 2.0
    return r1 + r2 - 0.09 * abs(en1 - en2)

def get_effective_coordination(atom_pos, neighbor_positions, atom_symbol, neighbor_symbols):
    if len(neighbor_positions) == 0: return 0.0
    r_atom = covalent_radii[atomic_numbers[atom_symbol]]
    eff_coord = 0.0
    for i, npos in enumerate(neighbor_positions):
        r_neigh = covalent_radii[atomic_numbers[neighbor_symbols[i]]]
        r_ideal = (r_atom + r_neigh) * 1.15
        dist = np.linalg.norm(np.array(atom_pos) - np.array(npos))
        eff_coord += 1.0 / (1.0 + np.exp((dist - r_ideal) / 0.1))
    return eff_coord

def calculate_height(site_info, binding_sym, structure):
    bonds = [get_ideal_bond_length(s, binding_sym) for s in site_info['atoms']]
    target_bond = np.mean(bonds) * 0.95
    dist_xy = site_info.get('dist_xy', 0)
    base_h = np.sqrt(target_bond**2 - dist_xy**2) if target_bond > dist_xy else 1.0
    
    coord = site_info.get('eff_coord', 9.0)
    if len(site_info['atoms']) == 1:   base_h *= (1.0 - 0.05 * (9 - coord))
    elif len(site_info['atoms']) == 2: base_h *= 1.02
    else:                              base_h *= 0.98
    return max(base_h, 0.5)

# ==================== 位点扫描与分类 (支持双面) ====================
def scan_surface_sites(structure, existing_h_positions, mode):
    """扫描表面并剔除已被现存H占据的位点"""
    z_coords = structure.positions[:, 2]
    z_max, z_min = z_coords.max(), z_coords.min()
    
    # 根据用户模式，决定扫描哪些层
    layers_to_scan = []
    if mode.lower() in ["top", "both", "symmetric"]:
        layers_to_scan.append(("top", [a for a in structure if a.position[2] >= z_max - 1.5]))
    if mode.lower() in ["bottom", "both", "symmetric"]:
        layers_to_scan.append(("bot", [a for a in structure if a.position[2] <= z_min + 1.5]))

    cutoffs = [covalent_radii[atomic_numbers[a.symbol]] * 2.5 for a in structure]
    nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(structure)
    
    available_sites = []
    
    for side, layer_atoms in layers_to_scan:
        raw_sites = []
        layer_indices = [a.index for a in layer_atoms]
        
        # Top位点
        for a in layer_atoms:
            indices, offsets = nl.get_neighbors(a.index)
            n_pos, n_sym = [], []
            for idx, offset in zip(indices, offsets):
                if idx in layer_indices:
                    n_pos.append(structure.positions[idx] + np.dot(offset, structure.cell))
                    n_sym.append(structure[idx].symbol)
            
            coord = get_effective_coordination(a.position, n_pos, a.symbol, n_sym)
            raw_sites.append({'type': 'top', 'pos': a.position, 'atoms': [a.symbol], 'eff_coord': coord, 'dist_xy': 0.0, 'side': side})

        if HAS_SCIPY and len(layer_atoms) >= 3:
            points_2d = np.array([a.position[:2] for a in layer_atoms])
            
            # Bridge
            try:
                tri = Delaunay(points_2d)
                for s in tri.simplices:
                    for (i, j) in [(s[0], s[1]), (s[1], s[2]), (s[0], s[2])]:
                        a1, a2 = layer_atoms[i], layer_atoms[j]
                        d_3d = np.linalg.norm(a1.position - a2.position)
                        if 1.5 < d_3d < 3.5:
                            pos = (a1.position + a2.position) / 2.0
                            raw_sites.append({'type': 'bri', 'pos': pos, 'atoms': [a1.symbol, a2.symbol], 'dist_xy': d_3d/2, 'side': side})
            except: pass

            # Hollow
            try:
                vor = Voronoi(points_2d)
                for v in vor.vertices:
                    dists = np.linalg.norm(points_2d - v, axis=1)
                    min_dist = np.min(dists)
                    if min_dist <= 3.5:
                        neighbors = [layer_atoms[i] for i in np.where(dists < min_dist + 0.5)[0]]
                        if len(neighbors) >= 3:
                            avg_z = np.mean([n.position[2] for n in neighbors])
                            raw_sites.append({'type': 'hol', 'pos': np.array([v[0], v[1], avg_z]), 'atoms': [n.symbol for n in neighbors], 'dist_xy': min_dist, 'side': side})
            except: pass

        filtered_types = [s for s in raw_sites if s['type'] in SITES or s['type'][:3] in SITES]
        
        # 当前层去重与生成 3D 高度坐标
        unique_sites = []
        for s in filtered_types:
            if not any(mic_dist(s['pos'][:2], u['pos'][:2], structure.cell, structure.pbc) < 0.5 for u in unique_sites):
                h = calculate_height(s, ADSORBATE, structure)
                # ★ 核心逻辑：顶面向上加高度，底面向下减高度
                z_shift = h if side == 'top' else -h
                s['final_3d_pos'] = s['pos'] + np.array([0, 0, z_shift])
                unique_sites.append(s)

        # 检查是否与现有的H碰撞
        for s in unique_sites:
            conflict = False
            for exh in existing_h_positions:
                if mic_dist(s['final_3d_pos'], exh, structure.cell, structure.pbc) < MIN_H_H_DIST:
                    conflict = True
                    break
            if not conflict:
                available_sites.append(s)

    return available_sites

# ==================== 智能候选池生成引擎 ====================
def evaluate_configuration(combo, existing_h, cell, pbc):
    score = 0.0
    all_h_positions = existing_h + [s['final_3d_pos'] for s in combo]
    
    n_total = len(all_h_positions)
    for i in range(n_total):
        for j in range(i + 1, n_total):
            # 这里的 3D 距离计算能完美处理顶底面的自然排斥
            d = mic_dist(all_h_positions[i], all_h_positions[j], cell, pbc)
            if d < MIN_H_H_DIST:
                return -float('inf') 
            score -= 1.0 / ((d + 1e-4) ** REPULSION_POWER)
            
    for s in combo:
        if 'eff_coord' in s:
            score += 0.2 * (1.0 - s['eff_coord'] / 12.0)
            
    return score

def is_different_enough(combo_a, combo_b, cell, pbc):
    pos_a = np.array([s['final_3d_pos'] for s in combo_a])
    pos_b = np.array([s['final_3d_pos'] for s in combo_b])
    
    total_diff = 0
    for pa in pos_a:
        min_d = min([mic_dist(pa, pb, cell, pbc) for pb in pos_b])
        total_diff += min_d
        
    return (total_diff / len(pos_a)) > DIVERSITY_TOL

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return
        
    structure = read(INPUT_FILE)
    substrate = structure.copy()
    existing_h = []
    del_indices = []
    
    for i, atom in enumerate(substrate):
        if atom.symbol == ADSORBATE:
            existing_h.append(atom.position)
            del_indices.append(i)
            
    if del_indices:
        del substrate[del_indices]
        
    current_coverage = len(existing_h)
    
    print(f"\n==================================================")
    print(f"🧬 高通量覆盖度候选池生成器 (Ensemble Generator)")
    print(f"==================================================")
    print(f"模式: [{ADSORB_MODE.upper()}] | 基础存在 {current_coverage} 个 | 目标新增 {N_ADD} 个")
    
    sites = scan_surface_sites(substrate, existing_h, mode=ADSORB_MODE)
    print(f"✅ 扫描完成：寻找到 {len(sites)} 个可用空闲位点。")
    
    if len(sites) < N_ADD:
        print(f"⛔ 错误：表面空间不足，无法再添加 {N_ADD} 个原子。")
        return
        
    print(f"⚙️ 正在评估构型势场，计算组合空间...")
    scored_combos = []
    total_combinations = math.comb(len(sites), N_ADD)
    
    if total_combinations > 100000:
        print(f"⚠️ 组合数过大 ({total_combinations})，可能需要较长时间...")

    for combo in combinations(sites, N_ADD):
        score = evaluate_configuration(combo, existing_h, substrate.cell, substrate.pbc)
        if score > -float('inf'):
            scored_combos.append((score, combo))
            
    scored_combos.sort(key=lambda x: x[0], reverse=True)
    
    if not scored_combos:
        print(f"⛔ 找不到符合安全距离限制 (>{MIN_H_H_DIST}Å) 的构型。")
        return
        
    final_candidates = []
    for score, combo in scored_combos:
        is_novel = True
        for _, existing_combo in final_candidates:
            if not is_different_enough(combo, existing_combo, substrate.cell, substrate.pbc):
                is_novel = False
                break
        
        if is_novel:
            final_candidates.append((score, combo))
            if len(final_candidates) >= TOP_K:
                break
                
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    print(f"\n🏆 筛选完毕！生成了 {len(final_candidates)} 个高质量候选构型：")
    for i, (score, combo) in enumerate(final_candidates):
        rank = i + 1
        
        final_struct = structure.copy()
        
        # 统计顶底数量用于命名
        top_c = sum(1 for s in combo if s['side'] == 'top')
        bot_c = sum(1 for s in combo if s['side'] == 'bot')
        site_types = [s['type'][:3] for s in combo] # 取前三个字母，例如 top, bri, hol
        type_str = "_".join(site_types)
        
        for s in combo:
            final_struct += Atoms(ADSORBATE, positions=[s['final_3d_pos']])
            
        cand_dir = os.path.join(OUTPUT_DIR, str(rank))
        os.makedirs(cand_dir, exist_ok=True)
        
        # VASP POSCAR
        poscar_path = os.path.join(cand_dir, "POSCAR")
        write(poscar_path, final_struct, format="vasp", direct=True, vasp5=True)
        
        # 详尽命名的 CIF 文件
        cif_filename = f"Cov{current_coverage + N_ADD}_{top_c}T_{bot_c}B_{type_str}.cif"
        cif_path = os.path.join(cand_dir, cif_filename)
        write(cif_path, final_struct, format="cif")
        
        print(f"   [{rank}] 评分: {score:8.3f} | {top_c}上_{bot_c}下 | {type_str:15} -> 文件夹: {cand_dir}/")

    print(f"\n💡 提示：文件已备好！将 {OUTPUT_DIR}/ 下的 1, 2, 3... 目录提交VASP即可。")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()


