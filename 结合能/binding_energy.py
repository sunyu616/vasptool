import os
import re

def get_energy(folder_path, filename='OUTCAR'):
    """
    从 OUTCAR 中读取最终能量 (energy without entropy)。
    如果使用的是其他软件（如 Gaussian），请修改此处的正则匹配逻辑。
    """
    file_path = os.path.join(folder_path, filename)
    if not os.path.exists(file_path):
        return None

    energy = None
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # 倒序读取，找到最后一次出现的能量
            for line in reversed(lines):
                # VASP 输出示例: energy  without entropy=     -14.23423423  energy(sigma->0) = ...
                if "energy  without entropy" in line:
                    parts = line.split()
                    #通常 without entropy 后的数字是我们要的，或者 sigma->0
                    #这里取 energy(sigma->0) 更准确用于结合能，通常在行尾
                    #格式: energy  without entropy=   -11.231231   energy(sigma->0) =   -11.232323
                    if "energy(sigma->0)" in line:
                        energy = float(parts[-1])
                    else:
                        # 某些旧版本可能格式不同，取 without entropy 后面的值
                        idx = parts.index("entropy=")
                        energy = float(parts[idx+1])
                    break
    except Exception as e:
        print(f"读取能量出错 {folder_path}: {e}")
    return energy

def get_composition_from_poscar(folder_path, filename='POSCAR'):
    """
    从 POSCAR 读取原子种类和数量 (VASP 5.x 格式)。
    返回字典: {'C': 1, 'O': 2}
    """
    file_path = os.path.join(folder_path, filename)
    if not os.path.exists(file_path):
        # 尝试从 CONTCAR 读取
        file_path = os.path.join(folder_path, 'CONTCAR')
        if not os.path.exists(file_path):
            return None

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 7:
                return None
            
            # 第6行通常是元素名称 (VASP 5+), 第7行是数量
            # 注意：Python索引从0开始，所以是 lines[5] 和 lines[6]
            elements_line = lines[5].strip().split()
            counts_line = lines[6].strip().split()

            # 简单的检查，防止 VASP 4 格式（第6行直接是数字）
            if elements_line[0].isdigit():
                print(f"警告: {folder_path} 的 POSCAR 可能是旧格式(VASP 4)，无法自动识别元素种类，跳过。")
                return None
            
            composition = {}
            for elem, count in zip(elements_line, counts_line):
                composition[elem] = int(count)
            return composition
    except Exception as e:
        print(f"读取结构出错 {folder_path}: {e}")
        return None

def main():
    root_dir = os.getcwd() # 当前文件夹
    atom_energies = {}     # 存储单原子能量: {'C': -1.23, 'O': -4.56}
    materials = []         # 存储物质信息

    print("正在扫描文件夹...")

    # 第一步：扫描所有 atom_ 开头的文件夹，建立数据库
    for folder in os.listdir(root_dir):
        if os.path.isdir(folder) and folder.startswith('atom_'):
            # 提取元素名称，例如 atom_C -> C
            element_name = folder.replace('atom_', '')
            energy = get_energy(folder)
            
            if energy is not None:
                atom_energies[element_name] = energy
                print(f"找到单原子: {element_name}, 能量: {energy:.6f} eV")
            else:
                print(f"警告: 无法读取 {folder} 中的能量")

    if not atom_energies:
        print("错误: 未找到任何有效的 atom_ 文件夹或能量读取失败。程序终止。")
        return

    # 第二步：扫描物质文件夹
    for folder in os.listdir(root_dir):
        # 排除 atom_ 文件夹，排除隐藏文件夹，排除非文件夹
        if os.path.isdir(folder) and not folder.startswith('atom_') and not folder.startswith('.'):
            total_energy = get_energy(folder)
            composition = get_composition_from_poscar(folder)

            if total_energy is not None and composition is not None:
                # 计算参考能量总和 (Sum of single atom energies)
                ref_energy_sum = 0.0
                missing_ref = False
                
                atom_details = [] # 用于调试或打印
                
                for elem, count in composition.items():
                    if elem in atom_energies:
                        ref_energy_sum += atom_energies[elem] * count
                        atom_details.append(f"{elem}:{count}")
                    else:
                        print(f"警告: 物质 {folder} 包含元素 {elem}，但未找到对应的 atom_{elem} 文件夹。")
                        missing_ref = True
                        break
                
                if not missing_ref:
                    # 计算结合能: E_binding = E_total - sum(n_i * E_atom_i)
                    binding_energy = total_energy - ref_energy_sum
                    materials.append({
                        'name': folder,
                        'total_E': total_energy,
                        'binding_E': binding_energy
                    })
                    print(f"处理物质: {folder}, 结合能: {binding_energy:.6f} eV")
            else:
                # 只是静默跳过无法解析的文件夹，避免刷屏
                pass

    # 第三步：写入 dat 文件
    output_file = 'binding_energy_results.dat'
    sorted_atoms = sorted(atom_energies.items()) # 转换为列表以便索引
    
    with open(output_file, 'w') as f:
        # 写入表头
        header = f"{'Material':<20} {'Total_Energy(eV)':<20} {'Binding_Energy(eV)':<20} {'':<5} {'Atom_Ref':<10} {'Atom_Energy(eV)':<20}\n"
        f.write(header)
        
        # 获取最大行数（物质数量 vs 原子种类数量，取大的那个以便打印）
        max_rows = max(len(materials), len(sorted_atoms))
        
        for i in range(max_rows):
            # 构建物质数据部分
            if i < len(materials):
                m = materials[i]
                line_part1 = f"{m['name']:<20} {m['total_E']:<20.6f} {m['binding_E']:<20.6f} {'':<5}"
            else:
                line_part1 = f"{'':<20} {'':<20} {'':<20} {'':<5}"
            
            # 构建单原子参考数据部分 (放在同一行的右侧)
            if i < len(sorted_atoms):
                atom_name, atom_e = sorted_atoms[i]
                line_part2 = f" {atom_name:<10} {atom_e:<20.6f}"
            else:
                line_part2 = ""
            
            f.write(line_part1 + line_part2 + "\n")

    print(f"\n计算完成！结果已保存至 {output_file}")

if __name__ == "__main__":
    main()

