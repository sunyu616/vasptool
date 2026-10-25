#!/usr/bin/env python3
from ase.io import read
from ase.geometry import get_distances
import numpy as np

atoms = read('structure.cif')

min_distance = 0.8

dist_matrix = atoms.get_all_distances(mic=True)

np.fill_diagonal(dist_matrix, np.inf)

if np.any(dist_matrix < min_distance):
    print("警告：检测到原子重叠！")
    indices = np.where(dist_matrix < min_distance)
    print(f"涉及的原子索引: {indices}")
else:
    print("结构合理，无严重重叠。")
