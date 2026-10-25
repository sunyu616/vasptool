import os
import re
import glob

# ================= 配置区域 =================

# 指定需要处理的文件夹路径 ('.' 代表当前脚本所在目录)
INPUT_FOLDER = '.'

# 指定输出文件夹名称 (自动创建)
OUTPUT_FOLDER = 'modified_cifs'

# 定义替换规则字典
# 格式: '原元素': '新元素'
# 例如: 将 Zr 和 Mo 都替换为 Ir
REPLACEMENT_RULES = {
    'Zr': 'Ir',
    'Mo': 'Ir'
}

# ===========================================

def apply_replacements(text, rules):
    """
    对文本行应用替换规则。
    使用正则表达式确保匹配的是单词边界或元素标签（如 Mo1, Zr2a），
    防止误伤其他单词（如 Monoclinic）。
    """
    sorted_keys = sorted(rules.keys(), key=len, reverse=True)
    
    for old_el in sorted_keys:
        new_el = rules[old_el]
        
        # 正则解释:
        # \b : 单词开头边界
        # old_el : 原元素符号 (如 Mo)
        # (?=\d|\b|_) : 后面紧跟数字(如Mo1)、单词结束边界、或下划线(有时出现在标签中)
        # 这样可以匹配 "Mo", "Mo1", "Mo_2b" 但不匹配 "Monoclinic" (因为o后面是n)
        pattern = r'\b' + re.escape(old_el) + r'(?=\d|\s|$|_)'
        
        text = re.sub(pattern, new_el, text)
        
    return text

def process_cif_files():
    # 检查并创建输出目录
    output_path = os.path.join(INPUT_FOLDER, OUTPUT_FOLDER)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"已创建输出目录: {output_path}")

    # 获取所有 .cif 文件
    cif_files = glob.glob(os.path.join(INPUT_FOLDER, "*.cif"))
    
    if not cif_files:
        print("未找到 .cif 文件。请确认脚本位置或 INPUT_FOLDER 设置。")
        return

    print(f"找到 {len(cif_files)} 个 CIF 文件，准备开始处理...")
    print(f"替换规则: {REPLACEMENT_RULES}")

    count = 0
    for file_path in cif_files:
        file_name = os.path.basename(file_path)
        
        # 跳过输出目录中的文件（如果脚本重复运行）
        if OUTPUT_FOLDER in file_path:
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            new_lines = []
            
            # 简单的状态机，判断是否处于原子位点数据块中
            in_atom_loop = False
            
            for line in lines:
                original_line = line
                stripped = line.strip()
                
                # 判断是否是元数据行 (以 _ 开头)
                is_metadata = stripped.startswith('_')
                
                # 特殊处理：只允许修改特定的元数据行（如化学式），防止误改 _symmetry_cell_setting Monoclinic
                allow_metadata_change = False
                if is_metadata:
                    if stripped.startswith('_chemical_formula') or stripped.startswith('_atom_site'):
                        allow_metadata_change = True
                
                # 如果是 loop_ 或者是数据行，或者是允许修改的元数据行，则尝试替换
                if not is_metadata or allow_metadata_change:
                     # 执行替换
                    modified_line = apply_replacements(original_line, REPLACEMENT_RULES)
                    new_lines.append(modified_line)
                else:
                    # 对于其他的元数据行（如 _cell_length, _symmetry...），保持原样，防止破坏文件结构
                    new_lines.append(original_line)

            # 写入新文件
            output_file_path = os.path.join(output_path, file_name)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            count += 1
            print(f"[成功] {file_name} -> {OUTPUT_FOLDER}/{file_name}")

        except Exception as e:
            print(f"[错误] 处理文件 {file_name} 时出错: {e}")

    print(f"\n处理完成！共修改了 {count} 个文件。")
    print(f"请检查 '{OUTPUT_FOLDER}' 文件夹。")

if __name__ == "__main__":
    process_cif_files()

