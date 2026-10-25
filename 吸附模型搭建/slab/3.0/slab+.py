import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from ase.build import molecule
from itertools import combinations

# ================= 配置区域 =================

# 1. 晶体结构设置
INPUT_FILE = "POSCAR"
SEARCH_DIRECTION = "top"  # "top" 或 "bottom"
RADIUS_LIMIT = 3.0

# 2. 吸附物设置 (支持 分子式 / SMILES / 文件名)
# 示例: "CO", "H2O", "C6H6" (分子式)
# 示例: "CCO" (乙醇 SMILES), "C=O" (甲醛 SMILES)
ADSORBATE_INPUT = "CO2" 

# 3. 指定连接原子 (Binding Atom)
# 指定吸附物中哪个原子与表面成键。填写索引 (0, 1, ...)。
BINDING_ATOM_INDEX = 0 

# 4. 输出设置
OUTPUT_ROOT_DIR = f"adsorption_{ADSORBATE_INPUT}_{SEARCH_DIRECTION}_optimized"
DUPLICATE_THRESHOLD = 0.8 

# ===========================================

# --- 尝试导入 RDKit 用于高级几何优化 ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("提示: 未检测到 RDKit 库。将使用 ASE 标准数据库进行分子构建。")
    print("      (如需支持 SMILES 输入和力场优化，请安装: pip install rdkit)")

# =========================================================================
# 核心模块：分子几何优化器
# =========================================================================

def optimize_molecule_geometry(input_str):
    """
    智能构建并优化分子几何结构。
    1. 尝试从文件读取。
    2. 尝试使用 RDKit 解析 SMILES 并进行 MMFF94 力场优化 (最推荐)。
    3. 回退使用 ASE 数据库 (G2 Dataset, 实验值)。
    """
    mol_atoms = None
    method = "Unknown"

    # 1. 尝试从文件读取
    if os.path.exists(input_str):
        try:
            mol_atoms = read(input_str)
            method = "File (Raw)"
            # 这里可以加简单的去重叠处理，但为了不破坏用户意图，暂时保持原样
        except:
            pass
    
    # 2. 尝试使用 RDKit (处理 SMILES 或 优化普通分子)
    if mol_atoms is None and HAS_RDKIT:
        try:
            # 尝试将输入视为 SMILES (例如 "CCO", "c1ccccc1")
            rd_mol = Chem.MolFromSmiles(input_str)
            if rd_mol:
                # 添加氢原子 (SMILES通常不含氢)
                rd_mol = Chem.AddHs(rd_mol)
                # 生成 3D 构象 (Embed)
                res = AllChem.EmbedMolecule(rd_mol, AllChem.ETKDG())
                if res == 0:
                    # 使用 MMFF94 力场进行几何优化 (调整键长键角)
                    AllChem.MMFFOptimizeMolecule(rd_mol)
                    
                    # 转换回 ASE Atoms 对象
                    conf = rd_mol.GetConformer()
                    positions = conf.GetPositions()
                    symbols = [atom.GetSymbol() for atom in rd_mol.GetAtoms()]
                    mol_atoms = Atoms(symbols=symbols, positions=positions)
                    method = "RDKit (MMFF94 Optimized)"
        except Exception as e:
            # print(f"RDKit 解析失败 (非SMILES): {e}")
            pass

    # 3. 尝试使用 ASE 数据库 (分子式)
    if mol_atoms is None:
        try:
            # ASE 的 molecule 数据库已经是优化好的实验结构 (G2)
            mol_atoms = molecule(input_str)
            method = "ASE Database (G2 Exp)"
        except:
            pass

    # 4. 手动后备 (针对 ASE 不包含的简单基团)
    if mol_atoms is None:
        if input_str == "OOH":
            mol_atoms = Atoms("OOH", positions=[[0,0,0], [1.3,0,0.8], [1.1,0,1.8]])
            method = "Manual Fallback"
            
    if mol_atoms:
        print(f"分子构建成功: {input_str} | 方法: {method} | 原子数: {len(mol_atoms)}")
        return mol_atoms
    else:
        print(f"错误: 无法构建分子 '{input_str}'。请检查拼写或安装 RDKit 使用 SMILES。")
        return None

# =========================================================================
# 辅助几何函数
# =========================================================================

def get_geometric_center(structure):
    positions = structure.get_positions()
    return np.mean(positions, axis=0)

def get_surface_atoms(structure, direction="top", z_threshold_ratio=0.8):
    z_coords = structure.positions[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    z_span = z_max - z_min
    if direction == "top":
        z_threshold = z_min + z_span * z_threshold_ratio
        return [atom for atom in structure if atom.position[2] >= z_threshold]
    else:
        z_threshold = z_min + z_span * (1.0 - z_threshold_ratio)
        return [atom for atom in structure if atom.position[2] <= z_threshold]

def get_circumcenter_2d(p1, p2, p3):
    x1, y1, x2, y2, x3, y3 = p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6: return None, None
    Ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    Uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    center = np.array([Ux, Uy])
    radius = np.linalg.norm(center - np.array([x1, y1]))
    return center, radius

def get_site_fingerprint(structure, site_pos, num_neighbors=6, tolerance=1):
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
    grouped = {}
    for site in sites:
        fp = get_site_fingerprint(structure, site['pos'])
        dist = np.linalg.norm(site['pos'] - center)
        if fp not in grouped: grouped[fp] = []
        grouped[fp].append({'site': site, 'dist': dist})
    unique = []
    for fp, candidates in grouped.items():
        best = sorted(candidates, key=lambda x: x['dist'])[0]
        unique.append(best['site'])
    return unique

def cross_category_filter(sites_dict, threshold=0.8):
    final_top = sites_dict.get('top', [])
    raw_bridge = sites_dict.get('bridge', [])
    final_bridge = []
    for b_site in raw_bridge:
        is_duplicate = False
        for t_site in final_top:
            if np.linalg.norm(b_site['pos'] - t_site['pos']) < threshold:
                is_duplicate = True; break
        if not is_duplicate: final_bridge.append(b_site)
            
    raw_hollow = sites_dict.get('hollow', [])
    final_hollow = []
    existing_sites = final_top + final_bridge
    for h_site in raw_hollow:
        is_duplicate = False
        for ex_site in existing_sites:
            if np.linalg.norm(h_site['pos'] - ex_site['pos']) < threshold:
                is_duplicate = True; break
        if not is_duplicate: final_hollow.append(h_site)
    return {'top': final_top, 'bridge': final_bridge, 'hollow': final_hollow}

# =========================================================================
# 键长计算与结构对齐模块
# =========================================================================

def get_covalent_radius(symbol):
    return covalent_radii[atomic_numbers[symbol]]

def calculate_optimal_height(site_info, binding_symbol):
    """
    根据 吸附原子 和 表面原子 的共价半径自动计算吸附高度。
    """
    # 1. 表面有效半径
    surf_symbols = site_info['constituent_atoms']
    r_surf_avg = np.mean([get_covalent_radius(s) for s in surf_symbols])
    
    # 2. 吸附原子半径
    r_ads = get_covalent_radius(binding_symbol)
    
    # 3. 理想键长 (Sum of covalent radii)
    # 经验修正: 吸附键通常比共价单键略长 (约+5%)，这里保持原始值作为紧密接触的下限
    target_bond_length = r_surf_avg + r_ads
    
    # 4. 几何高度计算
    dist_xy = site_info.get('dist_to_atom_xy', 0.0)
    
    if target_bond_length > dist_xy:
        height = np.sqrt(target_bond_length**2 - dist_xy**2)
    else:
        # 物理限制: 如果孔太小，吸附物无法进入平面，给予一个最小高度
        height = 1.5 
        
    return height, target_bond_length

def prepare_adsorbate(input_str, binding_idx, direction):
    """
    获取优化后的分子，并进行旋转对齐，使其适合吸附。
    """
    # 1. 获取几何优化后的分子
    ads = optimize_molecule_geometry(input_str)
    if ads is None: return None, None
    
    # 2. 确定连接原子
    if binding_idx >= len(ads):
        print(f"警告: 索引 {binding_idx} 超出范围，重置为 0")
        binding_idx = 0
    binding_symbol = ads[binding_idx].symbol
    
    # 3. 平移: 将连接原子置于原点 (0,0,0)
    ads.translate(-ads.positions[binding_idx])
    
    # 4. 旋转: 尝试将分子“竖直”立起来
    # 计算分子质心 (相对于连接原子)
    if len(ads) > 1:
        com = ads.get_center_of_mass()
        # 目标向量: Z轴 (0,0,1)
        # 将 COM 向量旋转对齐到 Z 轴
        # 这样分子的主体就会位于连接原子的上方
        if np.linalg.norm(com) > 0.1:
            ads.rotate(com, 'z')
            
    # 5. 方向处理: 如果是 bottom 面，翻转 Z 轴
    if direction == "bottom":
        pos = ads.get_positions()
        pos[:, 2] = -pos[:, 2]
        ads.set_positions(pos)
        
    return ads, binding_symbol

def generate_sites_minimal(structure, center, radius=3.0, direction="top"):
    # ... (保持原有的位点搜索逻辑不变，因为它已经很完善) ...
    surface_atoms = get_surface_atoms(structure, direction=direction)
    surf_coords = np.array([atom.position for atom in surface_atoms])
    
    raw_sites = {'top': [], 'bridge': [], 'hollow': []}

    # Top
    for atom in surface_atoms:
        if np.linalg.norm(atom.position - center) <= radius:
            anchor = f"{atom.symbol}{atom.index + 1}"
            raw_sites['top'].append({
                'pos': atom.position, 'anchor': anchor,
                'constituent_atoms': [atom.symbol], 'dist_to_atom_xy': 0.0
            })

    # Bridge
    for i, j in combinations(range(len(surface_atoms)), 2):
        atom1, atom2 = surface_atoms[i], surface_atoms[j]
        dist = np.linalg.norm(atom1.position - atom2.position)
        if 2.0 < dist < 3.5:
            mid_pos = (atom1.position + atom2.position) / 2
            if np.linalg.norm(mid_pos - center) <= radius:
                anchor = f"{atom1.symbol}{atom1.index + 1}-{atom2.symbol}{atom2.index + 1}"
                raw_sites['bridge'].append({
                    'pos': mid_pos, 'anchor': anchor,
                    'constituent_atoms': [atom1.symbol, atom2.symbol], 'dist_to_atom_xy': dist/2.0
                })

    # Hollow
    indices_near = [i for i, a in enumerate(surface_atoms) if np.linalg.norm(a.position-center) < radius+4.0]
    for combo in combinations(indices_near, 3):
        p1, p2, p3 = surf_coords[list(combo)]
        dists = [np.linalg.norm(p1-p2), np.linalg.norm(p2-p3), np.linalg.norm(p1-p3)]
        if max(dists) > 5.5 or min(dists) < 2.0: continue
        xy_c, xy_r = get_circumcenter_2d(p1[:2], p2[:2], p3[:2])
        if xy_c is None or xy_r > 2.8: continue
        
        combo_atoms = [surface_atoms[k] for k in combo]
        avg_z = np.mean([a.position[2] for a in combo_atoms])
        base_pos = np.array([xy_c[0], xy_c[1], avg_z])
        
        if np.linalg.norm(base_pos - center) <= radius:
            anchor = f"Hollow_{combo_atoms[0].symbol}" 
            raw_sites['hollow'].append({
                'pos': base_pos, 'anchor': anchor, 'coordination': 3,
                'constituent_atoms': [a.symbol for a in combo_atoms], 'dist_to_atom_xy': xy_r
            })

    merged_sites = {}
    for cat, sites in raw_sites.items():
        if sites:
            merged = merge_sites_internal(structure, sites, center)
            merged_sites[cat] = merged
        else:
            merged_sites[cat] = []
    
    return cross_category_filter(merged_sites, threshold=DUPLICATE_THRESHOLD)

def generate_files(input_poscar, sites_dict, output_root, ads_input, binding_idx, direction):
    try:
        structure = read(input_poscar)
    except: return

    # 1. 准备吸附物 (经过 几何优化 和 旋转对齐)
    base_adsorbate, binding_symbol = prepare_adsorbate(ads_input, binding_idx, direction)
    if base_adsorbate is None: return
    
    print(f"吸附物优化完毕。连接原子: {binding_symbol}")
    
    z_factor = 1.0 if direction == "top" else -1.0
    
    for stype, sites in sites_dict.items():
        if not sites: continue
        
        vasp_dir = os.path.join(output_root, stype, "vasp")
        cif_dir = os.path.join(output_root, stype, "cif")
        os.makedirs(vasp_dir, exist_ok=True)
        os.makedirs(cif_dir, exist_ok=True)
        
        for idx, sinfo in enumerate(sites):
            # 2. 计算吸附高度 (表面-吸附物 键长优化)
            optimal_height, _ = calculate_optimal_height(sinfo, binding_symbol)
            
            # 3. 组合
            this_ads = base_adsorbate.copy()
            # 垂直位移
            this_ads.translate([0, 0, optimal_height * z_factor])
            # 水平位移
            this_ads.translate(sinfo['pos'])
            
            final_struct = structure.copy()
            final_struct += this_ads
            
            atoms_str = "".join(set(sinfo['constituent_atoms']))
            fname_base = f"{stype}_{idx+1}_{ads_input}_on_{atoms_str}"
            if len(fname_base) > 60: fname_base = f"{stype}_{idx+1}_{ads_input}"

            write(os.path.join(vasp_dir, f"{fname_base}.vasp"), final_struct, format="vasp", sort=True)
            write(os.path.join(cif_dir, f"{fname_base}.cif"), final_struct, format="cif")

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        struct = read(INPUT_FILE)
        center = get_geometric_center(struct)
        
        print(f"=== 分子吸附生成器 v3.0 ===")
        print(f"吸附物输入: {ADSORBATE_INPUT}")
        
        sites = generate_sites_minimal(struct, center, RADIUS_LIMIT, SEARCH_DIRECTION)
        
        total = sum(len(v) for v in sites.values())
        if total > 0:
            generate_files(INPUT_FILE, sites, OUTPUT_ROOT_DIR, ADSORBATE_INPUT, BINDING_ATOM_INDEX, SEARCH_DIRECTION)
            print(f"\n生成完成! 结果保存在: {OUTPUT_ROOT_DIR}")
        else:
            print("未找到位点。")
    else:
        print("未找到 POSCAR。")

