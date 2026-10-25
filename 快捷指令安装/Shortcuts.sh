#!/bin/bash

# 检查是否在bash环境中
if [ -z "$BASH_VERSION" ]; then
    echo "警告: 此脚本需要在bash环境中运行"
fi

# 备份原有的 .bashrc
if [ -f ~/.bashrc ]; then
    BACKUP_FILE=~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
    cp ~/.bashrc "$BACKUP_FILE"
    echo "✅ 已备份原有的 .bashrc 到: $BACKUP_FILE"
fi

# 添加分隔标记和内容
cat >> ~/.bashrc << 'EOF'

# ==========================================
# 自定义快捷命令 (添加于 $(date))
# ==========================================

# PBS Job Directory Jumper
# Usage: qj <job_id>
function qj() {
    if [ -z "$1" ]; then
        echo "Error: Please provide a Job ID."
        return 1
    fi

    local job_id=$1
    
    # 1. 获取 qstat 信息
    # 2. sed ':a;N;$!ba;s/\n\s//g' -> 核心逻辑：将由于过长而被折行的输出合并。
    #    PBS通常在折行处使用 "换行符+空格" 或 "换行符+制表符"。
    # 3. grep 提取 PBS_O_WORKDIR 行
    # 4. sed 清理变量名，提取等号后的路径
    # 5. cut 去除可能存在的后续逗号（部分PBS版本会在变量间加逗号）
    
    local work_dir=$(qstat -f "$job_id" 2>/dev/null | \
        sed ':a;N;$!ba;s/\n\t//g;s/\n //g' | \
        grep "PBS_O_WORKDIR" | \
        sed 's/.*PBS_O_WORKDIR=//' | \
        cut -d',' -f1)

    # 检查是否成功获取路径
    if [ -z "$work_dir" ]; then
        echo "Error: Could not find run directory for job $job_id."
        echo "Note: The job might have finished and cleared from history."
        return 1
    fi

    # 检查路径是否存在
    if [ -d "$work_dir" ]; then
        echo "Jumping to: $work_dir"
        cd "$work_dir"
    else
        echo "Error: Directory does not exist: $work_dir"
        return 1
    fi
}

# ==========================================
# 快速路径记录与跳转 (sp & lp)
# ==========================================

# sp: Save Path - 保存当前路径
sp() {
    pwd > ~/.saved_path
    echo "📌 路径已保存: $(pwd)"
}

# lp: Load Path - 跳转到保存的路径
lp() {
    if [ -f ~/.saved_path ]; then
        local target_dir=$(cat ~/.saved_path)
        if [ -d "$target_dir" ]; then
            cd "$target_dir" || return
            echo "🚀 已跳转至: $(pwd)"
        else
            echo "❌ 错误: 记录的目录现在已不存在 ($target_dir)"
        fi
    else
        echo "⚠️ 尚未记录任何路径，请先在某个目录下执行 'sp' 命令。"
    fi
}

# ==========================================
# 快捷指令别名
# ==========================================
alias gr="grep 'reached required accuracy' OUTCAR"
alias gt="grep Total OUTCAR"
alias gfe="grep 'free  energy' OUTCAR"
alias gwe="grep 'energy  without entropy' OUTCAR"
alias gtot="grep 'tot' OUTCAR"
alias cp1="cp INCAR POSCAR POTCAR wk.pbs KPOINTS"
alias cp2="cp INCAR CONTCAR POTCAR wk.pbs KPOINTS"
alias mcp="mv CONTCAR POSCAR"
alias poc="pos2cif.pl CONTCAR"
alias rm="rm -r"
alias ccp="cp CONTCAR POSCAR"
alias mag="awk '/magnetization \(x\)/,/^ *tot/' OUTCAR"
alias ta="tail -f auto_opt.log"
alias subautovasp="nohup python3 -u vasp_infinite_wait.py > stdout.log 2>&1 &"
alias findpid="pgrep -u \$USER -f vasp_infinite_wait.py | xargs pwdx | grep \"\$PWD\""
alias gn="grep NBANDS OUTCAR"
alias cpwk="ls -d */ | xargs -I {} cp wk.pbs {}"
alias subworkflow="nohup python3 -u workflow_master.py > workflow.log 2>&1 &"
alias tw="tail -f workflow.log"

EOF

echo ""
echo "✅ 所有快捷命令已成功写入 ~/.bashrc"
echo ""

# 执行 source
if [ -n "$BASH_VERSION" ]; then
    source ~/.bashrc
    echo "✅ 已执行 source ~/.bashrc，所有命令现已生效！"
else
    echo "⚠️  请手动执行: source ~/.bashrc"
fi

echo ""
echo "📋 已添加的功能："
echo "  • qj <job_id>  - 跳转到PBS作业目录"
echo "  • sp           - 保存当前路径"
echo "  • lp           - 跳转到保存的路径"
echo "  • gr, gt, gfe, gwe, gtot, gn - VASP输出分析"
echo "  • cp1, cp2, ccp, mcp - 文件管理"
echo "  • mag, ta, poc - 其他实用功能"
echo "  • subautovasp, findpid - 任务管理"