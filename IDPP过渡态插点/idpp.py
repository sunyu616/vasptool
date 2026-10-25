import os
import numpy as np
from ase.io import read, write
from ase.mep import NEB

# ================= 用户配置区域 =================
# 1. 输入文件名
initial_file = 'initial.poscar'  # 初态文件名
final_file = 'final.poscar'      # 末态文件名

# 2. 中间插值点的数量 (不包括头尾)
# 例如：如果你想要 00, 01, 02, 03, 04, 05, 06 (共7个文件夹)
# 其中 00 是初态，06 是末态，中间有 5 个点，这里就填 5
n_images = 5 
# ==============================================

def main():
    print(f"--- 开始读取结构 ---")
    try:
        initial = read(initial_file)
        final = read(final_file)
        print(f"成功读取: {initial_file} 和 {final_file}")
    except FileNotFoundError:
        print("错误：找不到输入文件，请检查文件名是否正确。")
        return

    # ---------------------------------------------------------
    # 关键修复步骤：解决 "Variable cell" 报错
    # ---------------------------------------------------------
    print(f"--- 正在检查并修复晶胞差异 ---")
    
    # 获取初态的晶胞信息
    cell_init = initial.get_cell()
    cell_final = final.get_cell()
    
    # 强制将末态的晶胞设置为与初态完全一致
    # scale_atoms=True 保证原子的相对分数坐标不变，随晶胞微调
    final.set_cell(cell_init, scale_atoms=True)
    
    # 再次确认原子数一致
    if len(initial) != len(final):
        raise ValueError("错误：初态和末态的原子数量不一致！")

    print("晶胞修复完成：末态晶胞已强制匹配初态。")

    # ---------------------------------------------------------
    # 生成插值路径
    # ---------------------------------------------------------
    print(f"--- 正在生成 IDPP 路径 (中间点数: {n_images}) ---")
    
    # 创建列表：[初态, 复制品..., 末态]
    images = [initial]
    for i in range(n_images):
        images.append(initial.copy())
    images.append(final)

    # 初始化 NEB 对象
    neb = NEB(images)

    # 运行 IDPP 插值
    # mic=True (Minimum Image Convention) 用于处理周期性边界条件
    print("正在运行 IDPP 算法优化坐标 (请稍候)...")
    neb.interpolate(method='idpp', mic=True)

    # ---------------------------------------------------------
    # 保存结果
    # ---------------------------------------------------------
    print(f"--- 正在输出文件 ---")
    
    # 1. 保存总轨迹文件 (用于 ase gui 可视化)
    traj_filename = 'idpp_path_fixed.traj'
    write(traj_filename, images)
    print(f"可视化轨迹已保存为: {traj_filename}")
    print(f"  -> 请运行 'ase gui {traj_filename}' 检查路径是否合理")

    # 2. 生成 VASP 文件夹 (00, 01, ...)
    for i, image in enumerate(images):
        dirname = f'{i:02d}' # 生成 00, 01, 02 格式的目录名
        
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        
        # 写入 POSCAR
        output_path = os.path.join(dirname, 'POSCAR')
        write(output_path, image, format='vasp', direct=True) # direct=True 使用分数坐标
        print(f"  -> 已生成: {output_path}")

    print("\n=== 全部完成！ ===")
    print("下一步建议：")
    print("1. 检查生成的 POSCAR 是否合理。")
    print("2. 将 POTCAR, KPOINTS, INCAR 复制到主目录或各子目录。")
    print(f"3. 确保 INCAR 中设置 IMAGES = {n_images}")

if __name__ == "__main__":
    main()
