from ase.io import read
from ase.neighborlist import neighbor_list, natural_cutoffs

# 1. 读取结构文件 (例如 POSCAR, xyz, cif 等)
atoms = read('POSCAR')

# 2. 生成自然截断半径 (Natural Cutoffs)
# mult=1.2 表示在共价半径之和的基础上放大 1.2 倍作为容差 (类似 MS 里的 Tolerance)
cutoffs = natural_cutoffs(atoms, mult=1.2)

# 3. 计算邻居列表
# 'i' 和 'j' 分别是成键原子的索引列表
# 'd' 是它们之间的距离 (可选)
i_indices, j_indices, distances = neighbor_list('ijd', atoms, cutoffs)

# 4. 打印结果
print("成键列表:")
for i, j, d in zip(i_indices, j_indices, distances):
    # 避免重复打印 (比如 A-B 和 B-A 只打一次)
    if i < j:
        print(f"原子 {i} ({atoms[i].symbol}) - 原子 {j} ({atoms[j].symbol}) : 距离 {d:.3f} Å")

# 5. 判断特定两个原子是否成键
# 假设你想看索引为 0 和 1 的原子是否在列表里
target_pair = {0, 1}
is_bonded = False
for i, j in zip(i_indices, j_indices):
    if {i, j} == target_pair:
        is_bonded = True
        break

print(f"\n原子 0 和 1 是否成键? {is_bonded}")
