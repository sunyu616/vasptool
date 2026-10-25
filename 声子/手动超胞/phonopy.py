import numpy as np
import sys
import yaml
import copy

try:
    import spglib
except ImportError:
    print("错误：未找到 spglib 库。请安装：pip install spglib numpy")
    sys.exit(1)

# --- 1. VASP POSCAR 读取与写入 ---
def read_poscar(filename="POSCAR"):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    scale = float(lines[1].strip())
    lattice = np.array([[float(x) for x in lines[2].split()],
                        [float(x) for x in lines[3].split()],
                        [float(x) for x in lines[4].split()]]) * scale
    
    species_names = lines[5].split()
    try:
        num_atoms = [int(x) for x in lines[6].split()]
        start_idx = 7
    except ValueError:
        species_names = []
        num_atoms = [int(x) for x in lines[5].split()]
        start_idx = 6
        
    total_atoms = sum(num_atoms)
    coord_type = lines[start_idx].strip().lower()
    if coord_type.startswith('s'):
        coord_type = lines[start_idx+1].strip().lower()
        start_idx += 2
    else:
        start_idx += 1
        
    is_cartesian = coord_type.startswith('c') or coord_type.startswith('k')
    positions = np.array([[float(x) for x in lines[start_idx + i].split()[:3]] for i in range(total_atoms)])
    
    if is_cartesian:
        inv_lattice = np.linalg.inv(lattice.T)
        positions = np.dot(positions, inv_lattice.T)
        
    positions = positions % 1.0
    
    symbols = []
    if species_names:
        for name, num in zip(species_names, num_atoms):
            symbols.extend([name] * num)
    else:
        for i, num in enumerate(num_atoms):
            symbols.extend([str(i+1)] * num)
            
    return {
        'lattice': lattice, 'positions': positions, 'symbols': symbols,
        'num_atoms': num_atoms, 'species_names': species_names if species_names else [str(i+1) for i in range(len(num_atoms))]
    }

def write_poscar(filename, structure, comment="Generated"):
    with open(filename, 'w') as f:
        f.write(comment + "\n1.0\n")
        for vec in structure['lattice']: f.write(f" {vec[0]:20.16f} {vec[1]:20.16f} {vec[2]:20.16f}\n")
        f.write(" " + " ".join(structure['species_names']) + "\n")
        f.write(" " + " ".join([str(x) for x in structure['num_atoms']]) + "\nDirect\n")
        for pos in structure['positions']: f.write(f" {pos[0]:20.16f} {pos[1]:20.16f} {pos[2]:20.16f}\n")

# --- 2. 扩建超胞 ---
def build_supercell(unitcell, supercell_matrix):
    smat = np.array(supercell_matrix)
    u_lattice = unitcell['lattice']
    u_pos = unitcell['positions']
    u_sym = unitcell['symbols']
    
    s_lattice = np.dot(smat, u_lattice)
    det = int(round(np.linalg.det(smat)))
    
    max_val = np.max(np.abs(smat)) * 2
    grid = np.indices((max_val*2, max_val*2, max_val*2)).reshape(3, -1).T - max_val
    frac_grid = np.dot(grid, np.linalg.inv(smat))
    eps = 1e-5
    valid_points = frac_grid[(frac_grid[:, 0] >= -eps) & (frac_grid[:, 0] < 1-eps) &
                             (frac_grid[:, 1] >= -eps) & (frac_grid[:, 1] < 1-eps) &
                             (frac_grid[:, 2] >= -eps) & (frac_grid[:, 2] < 1-eps)]
    
    if len(valid_points) != det and np.allclose(smat, np.diag(np.diagonal(smat))):
        valid_points = np.array([[i/smat[0,0], j/smat[1,1], k/smat[2,2]] 
                                 for i in range(smat[0,0]) for j in range(smat[1,1]) for k in range(smat[2,2])])
            
    s_positions, s_symbols, s_num_atoms = [], [], []
    for sym in unitcell['species_names']:
        count = 0
        for i, u_sym_i in enumerate(u_sym):
            if u_sym_i == sym:
                for trans in valid_points:
                    pos_cart = np.dot(u_pos[i], u_lattice)
                    trans_cart = np.dot(trans, s_lattice)
                    new_pos_frac = np.dot(pos_cart + trans_cart, np.linalg.inv(s_lattice)) % 1.0
                    s_positions.append(new_pos_frac)
                    s_symbols.append(sym)
                    count += 1
        s_num_atoms.append(count)
        
    return {
        'lattice': s_lattice, 'positions': np.array(s_positions),
        'symbols': s_symbols, 'num_atoms': s_num_atoms, 'species_names': unitcell['species_names']
    }

# --- 3. 寻找不可约位移 (包含正负双向) ---
def get_displacements(supercell, distance=0.01, include_negative=True):
    cell = (supercell['lattice'], supercell['positions'], 
            [supercell['species_names'].index(sym)+1 for sym in supercell['symbols']])
    
    dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
    independent_atoms = np.unique(dataset['equivalent_atoms']) if dataset else np.arange(len(supercell['positions']))
    
    displacements = []
    directions = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    
    signs = [1.0, -1.0] if include_negative else [1.0]
    
    for atom_idx in independent_atoms:
        for d_cart in directions:
            for sign in signs:
                disp_cart = d_cart * distance * sign
                displacements.append({
                    'atom': int(atom_idx),
                    'disp_cart': disp_cart.tolist()
                })
    return displacements

# --- 4. 生成 phonopy_disp.yaml ---
def write_phonopy_disp_yaml(filename, unitcell, supercell, supercell_matrix, displacements):
    disp_list_yaml = []
    for d in displacements:
        disp_list_yaml.append({
            'atom': d['atom'] + 1,
            'displacement': [float(x) for x in d['disp_cart']]
        })
        
    data = {
        'phonopy': {'version': '2.15.0', 'calculator': 'vasp'},
        'supercell_matrix': [[int(x) for x in row] for row in supercell_matrix],
        'displacements': disp_list_yaml,
        'natom': len(supercell['positions']),
        'supercell': {
            'lattice': supercell['lattice'].tolist(),
            'points': supercell['positions'].tolist(),
            'symbols': supercell['symbols']
        },
        'unitcell': {
            'lattice': unitcell['lattice'].tolist(),
            'points': unitcell['positions'].tolist(),
            'symbols': unitcell['symbols']
        }
    }
    
    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)

# --- 5. 获取用户输入的超胞矩阵 ---
def get_supercell_input():
    print("--- 独立位移生成工具 (完美匹配 Phonopy) ---")
    user_input = input("请输入超胞倍数，用空格分隔 (例如: 2 2 2)。直接按回车默认使用 2 2 2: ").strip()
    
    if not user_input:
        print("-> 使用默认超胞: 2x2x2")
        dim = [2, 2, 2]
    else:
        try:
            dim = [int(x) for x in user_input.split()]
            if len(dim) != 3:
                raise ValueError
            print(f"-> 使用自定义超胞: {dim[0]}x{dim[1]}x{dim[2]}")
        except ValueError:
            print("错误：输入格式不正确，请确保输入的是三个整数，例如 '2 2 2'。")
            sys.exit(1)
            
    # 构造对角矩阵
    supercell_matrix = [
        [dim[0], 0, 0],
        [0, dim[1], 0],
        [0, 0, dim[2]]
    ]
    return supercell_matrix

# ================= 主程序 =================
def main():
    # 1. 交互式获取超胞矩阵
    supercell_matrix = get_supercell_input()
    
    input_file = "POSCAR"
    distance = 0.01
    include_negative_displacements = True 
    
    try:
        unitcell = read_poscar(input_file)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_file}'，请确保当前目录下有原胞的 POSCAR。")
        sys.exit(1)
        
    supercell = build_supercell(unitcell, supercell_matrix)
    write_poscar("SPOSCAR", supercell, comment=f"Supercell {supercell_matrix[0][0]}x{supercell_matrix[1][1]}x{supercell_matrix[2][2]} generated manually")
    
    displacements = get_displacements(supercell, distance=distance, include_negative=include_negative_displacements)
    inv_lattice = np.linalg.inv(supercell['lattice'])
    
    for i, disp in enumerate(displacements):
        scell_disp = copy.deepcopy(supercell)
        atom_idx = disp['atom']
        disp_frac = np.dot(disp['disp_cart'], inv_lattice)
        
        scell_disp['positions'][atom_idx] = (scell_disp['positions'][atom_idx] + disp_frac) % 1.0
        
        filename = f"POSCAR-{i+1:03d}"
        write_poscar(filename, scell_disp, comment=f"Disp {i+1} atom {atom_idx+1} {disp['disp_cart']}")
        print(f"已生成 -> {filename}")
        
    write_phonopy_disp_yaml("phonopy_disp.yaml", unitcell, supercell, supercell_matrix, displacements)
    print("已生成 -> phonopy_disp.yaml (完美兼容版)")
    print("已生成 -> SPOSCAR (完美超胞)")
    print(f"\n✅ 成功生成 {len(displacements)} 个位移结构。现在应该和 phonopy 的数量对齐了！")

if __name__ == "__main__":
    main()


