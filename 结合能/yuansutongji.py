import os
import sys

def is_vasp5(line_parts):
    """
    判断是否为 VASP 5 格式（第6行是元素符号而不是数字）
    """
    if not line_parts:
        return False
    # 尝试将第一个部分转换为整数
    try:
        int(line_parts[0])
        return False # 如果成功转为数字，说明是原子数量，属于 VASP 4
    except ValueError:
        return True # 无法转为数字，说明是元素符号，属于 VASP 5

def parse_poscar(filepath):
    """
    读取 POSCAR 文件并提取元素信息
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        if len(lines) < 6:
            return {"status": "error", "msg": "文件行数不足"}

        # 去除首尾空白字符
        line6 = lines[5].strip()
        parts = line6.split()

        if is_vasp5(parts):
            # VASP 5 格式：第6行是元素名称，第7行是数量
            elements = parts
            counts = lines[6].strip().split()
            # 简单的校验
            if len(elements) != len(counts):
                return {"status": "warning", "elements": elements, "msg": "元素符号与数量不匹配"}
            return {"status": "success", "elements": elements, "counts": counts, "version": "VASP 5"}
        else:
            # VASP 4 格式：第6行是数量，元素信息通常不在 POSCAR 内（或在注释里）
            # 这种情况下无法准确获取元素，除非读取 POTCAR
            counts = parts
            return {"status": "vasp4", "counts": counts, "version": "VASP 4 (无元素标签)"}

    except Exception as e:
        return {"status": "error", "msg": str(e)}

def main():
    root_dir = os.getcwd() # 获取当前工作目录
    print(f"正在遍历目录: {root_dir}\n")
    print(f"{'路径':<60} | {'格式':<15} | {'元素组成'}")
    print("-" * 100)

    total_poscars = 0
    global_elements = set()
    vasp4_count = 0

    # 遍历目录
    for root, dirs, files in os.walk(root_dir):
        if "POSCAR" in files:
            full_path = os.path.join(root, "POSCAR")
            rel_path = os.path.relpath(full_path, root_dir)
            
            result = parse_poscar(full_path)
            total_poscars += 1

            if result["status"] == "success":
                # 格式化输出：例如 Li:2, O:1
                comp_str = ", ".join([f"{e}:{c}" for e, c in zip(result['elements'], result['counts'])])
                print(f"{rel_path:<60} | {result['version']:<15} | {comp_str}")
                
                # 收集所有出现的元素
                for e in result['elements']:
                    global_elements.add(e)

            elif result["status"] == "vasp4":
                vasp4_count += 1
                counts_str = ", ".join(result['counts'])
                print(f"{rel_path:<60} | {result['version']:<15} | 原子数: {counts_str} (未知元素)")

            elif result["status"] == "warning":
                 print(f"{rel_path:<60} | {result['msg']:<15} | {' '.join(result['elements'])}")
            
            else:
                print(f"{rel_path:<60} | 读取错误 | {result['msg']}")

    print("-" * 100)
    print("\n总结:")
    print(f"共找到 POSCAR 文件: {total_poscars} 个")
    
    if global_elements:
        sorted_elements = sorted(list(global_elements))
        print(f"包含的所有唯一元素 ({len(sorted_elements)}种): {', '.join(sorted_elements)}")
    else:
        print("未检测到明确的元素信息。")

    if vasp4_count > 0:
        print(f"注意: 有 {vasp4_count} 个文件是 VASP 4 旧格式，POSCAR 内部不包含元素符号，仅包含原子数量。")

if __name__ == "__main__":
    main()

