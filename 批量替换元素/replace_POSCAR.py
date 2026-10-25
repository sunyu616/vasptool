import os
import shutil

# ================= 用户配置区域 =================

# 1. 定义替换规则 (格式: '旧元素': '新元素')
# 示例: {'Fe': 'Co'} -> 将 Fe 换成 Co
# 示例: {'Fe': 'Ni', 'Co': 'Ni'} -> 将 Fe 和 Co 都换成 Ni (多换一)
REPLACE_RULES = {
    'Ta': 'Co', 
    
    # 你可以在这里继续添加规则
}

# 2. 指定要处理的文件名
TARGET_FILENAME = 'POSCAR'

# ==============================================

def process_poscar(file_path):
    """读取 POSCAR，检查第6行，替换元素，备份并写入"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # VASP 5+ 格式：元素符号位于第 6 行 (索引 5)
        # 如果文件行数少于6行，肯定不是合法的 POSCAR
        if len(lines) < 6:
            return

        element_line_index = 5
        original_line = lines[element_line_index]
        
        # 安全检查：如果第6行包含数字，说明可能是 VASP 4 旧格式或 Selective Dynamics 开启位置不同
        # 我们只处理第6行是字母的情况
        if any(char.isdigit() for char in original_line):
            print(f"[跳过] {file_path} (第6行包含数字，可能格式不兼容)")
            return

        elements = original_line.split()
        new_elements = []
        is_changed = False

        # 遍历当前行中的每一个元素
        for el in elements:
            clean_el = el.strip()
            if clean_el in REPLACE_RULES:
                # 如果在规则字典里，就替换
                new_elements.append(REPLACE_RULES[clean_el])
                is_changed = True
            else:
                # 否则保持原样
                new_elements.append(clean_el)

        # 只有当发生改变时才写入文件
        if is_changed:
            # 1. 创建备份
            backup_path = file_path + ".bak"
            shutil.copy2(file_path, backup_path)

            # 2. 修改内存中的数据
            # 使用 3 个空格连接，保持一定的格式美观
            lines[element_line_index] = "   " + "   ".join(new_elements) + " \n"

            # 3. 写入原文件
            with open(file_path, 'w') as f_out:
                f_out.writelines(lines)
            
            print(f"[修改成功] {file_path}")
            print(f"    └─ 变化: {elements}  --->  {new_elements}")

    except Exception as e:
        print(f"[错误] 处理文件 {file_path} 时发生异常: {e}")

def main():
    root_dir = os.getcwd() # 获取当前运行脚本的目录
    print("="*60)
    print(f"开始任务: 遍历并替换 POSCAR 元素")
    print(f"工作目录: {root_dir}")
    print(f"替换规则: {REPLACE_RULES}")
    print("="*60)

    count = 0
    # os.walk 递归遍历所有子目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if TARGET_FILENAME in filenames:
            full_path = os.path.join(dirpath, TARGET_FILENAME)
            process_poscar(full_path)
            count += 1
    
    print("="*60)
    print("处理完成。")

if __name__ == "__main__":
    main()




