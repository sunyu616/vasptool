import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from ase.build import molecule
from itertools import combinations
import shutil
import random
import subprocess
import re

# ================= 用户配置区域 (User Configuration) =================

INPUT_FILE = "POSCAR"          
SEARCH_DIRECTION = "top"       
SURFACE_THICKNESS = 1.5        # 表面层厚度判定阈值(Å)

# 吸附模式选择
SELECT_MODE = "H"             

# 吸附物配置库
ADSORPTION_SCHEMES = {
    "H":   {"molecule": "H",   "binding": "H", "scale": 0.9,  "tilt": 0,  "rotate": False},
    "O":   {"molecule": "O",   "binding": "O", "scale": 1.0,  "tilt": 0,  "rotate": False},
    "OH":  {"molecule": "OH",  "binding": "O", "scale": 1.05, "tilt": 15, "rotate": True},
    "OOH": {"molecule": "OOH", "binding": "O", "scale": 1.10, "tilt": 45, "rotate": True}
}

# 桥位与空位判定参数
MIN_BRIDGE_LENGTH = 1.5        
MAX_BRIDGE_HARD_LIMIT = 5.0    
DYNAMIC_BRIDGE_TOLERANCE = 1.15 
MAX_BRIDGE_Z_DIFF = 1.0        

MAX_HOLLOW_RADIUS = 3.5 
MAX_HOLLOW_EDGE = 4.5                 
COORDINATION_TOLERANCE = 0.5   

# 物理空间竞争阈值
CROSS_MERGE_TOL = 0.2          

# 同类型核心去重算法参数
SPATIAL_MERGE_TOL = 0.4        
ENV_MERGE_TOL = 0.15           
ENV_CUTOFF = 4.5               

# ================= 内置 VASP 计算模板文件 =================

INCAR_CONTENT = r'''   ISTART = 0
   ICHARG = 2
   NPAR   =6

   IBRION = 2
   ISIF   = 2
   NSW    = 300

   GGA    = PE
   ISPIN  = 1
 # MAGMOM = 
 
   PREC   = N
   ADDGRID= T
   EDIFF  = 1.0e-5
   EDIFFG = -0.02
   
   ENCUT  = 450
   NELM   = 100
   NELMIN = 2
   ISMEAR = 0
   SIGMA  = 0.01
   LWAVE  = F
   LCHARG = F
   IALGO  = 48
   LORBIT = 11
   LREAL  = Auto
   ISYM   = 0

   IVDW=11
   VDW_RADIUS=50.2
   VDW_CNRADIUS=20.0
'''

KPOINTS_CONTENT = r'''K-Spacing Value to Generate K-Mesh: 0.000
0
Gamma
   5   5   1
0.0  0.0  0.0
'''

PBS_CONTENT = r'''#!/bin/bash
#PBS -N sy
#PBS -l nodes=master:ppn=24
#PBS -j n
#PBS -e ${PBS_JOBNAME}.err
#PBS -o ${PBS_JOBNAME}.out
#PBS -q batch

# --- 公共设置 ---
ulimit -s unlimited
export OMP_NUM_THREADS=1

# --- 自动判断服务器环境 ---
# 定义两个服务器上不同的 VASP 路径
VASP_PATH_1="/public/apps/vasp/vasp_std"
VASP_PATH_2="/public/apps/vasp/bin/vasp_std"

if [ -f "$VASP_PATH_1" ]; then
    # === 情况 1: 对应你的第一个脚本 ===
    EXEC=$VASP_PATH_1
    # 如果第一个服务器不需要特殊环境变量，这里留空即可

elif [ -f "$VASP_PATH_2" ]; then
    # === 情况 2: 对应你的第二个脚本 ===
    EXEC=$VASP_PATH_2
    # 第二个脚本特有的环境变量
    export I_MPI_ADJUST_REDUCE=3

else
    # === 错误处理 ===
    echo "Error: Cannot find VASP executable on this server."
    exit 1
fi

# --- 执行计算 ---
cd $PBS_O_WORKDIR
NP=`cat $PBS_NODEFILE|wc -l`

# 打印一下当前使用的程序路径，方便在 output 文件中排查问题
echo "Running on host: $(hostname)"
echo "Using VASP executable: $EXEC"
echo "Total processes (NP): $NP"

mpirun -np $NP $EXEC > ${PBS_JOBNAME}.out
'''

WAIT_PY_CONTENT = r'''import os
import time
import subprocess
import shutil
import numpy as np
import sys
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.header import Header

# --- Optional Libraries ---
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

HAS_SYMMETRY_LIB = False
try:
    import ase
    from ase.io import read, write
    import spglib
    from ase import Atoms
    from ase.build import make_supercell
    HAS_SYMMETRY_LIB = True
except ImportError:
    HAS_SYMMETRY_LIB = False

# ================== 1. USER CONFIGURATION ==================

SCRIPT_MODE = 'MONITOR'  # 脚本运行模式: 'OPT'为晶胞自动优化, 'MONITOR'为仅监控单次任务

# --- Symmetry Protection (Full Sweep) / 对称性保护设置 ---
ENABLE_SYMMETRY = True         # 是否开启对称性保护 (强烈建议开启以维持晶系)
TOLERANCE_SWEEP = [0.001, 0.01, 0.1, 0.2, 0.5] # 寻找最高对称性时遍历的精度容差列表

# --- Sync Modes / 晶轴同步模式 ---
SYNC_MODE = 'None'   # 晶轴形变同步: 'None'(独立变化), 'AB'(强制a=b, 如四方/六方晶系), 'ABC'(强制a=b=c, 如立方晶系)

# --- Opt Switches / 优化维度开关 ---
OPT_A = True  # 是否优化 a 轴
OPT_B = True  # 是否优化 b 轴
OPT_C = False # 是否优化 c 轴 (2D材料通常设为False)

# --- Email Config / 邮件提醒配置 ---
ENABLE_EMAIL = True                  # 是否开启邮件通知
SMTP_SERVER = "smtp.qq.com"          # SMTP 服务器地址
SMTP_PORT = 465                      # SMTP 端口 (SSL通常为465)
SENDER_EMAIL = "your_qq_number@qq.com"    # 发件人邮箱
SENDER_PASSWORD = "在此填入QQ邮箱授权码" # 发件人邮箱授权码/密码
RECEIVER_EMAIL = "your_qq_number@qq.com"  # 收件人邮箱

# --- Task Management / 任务提交流程配置 ---
SUBMIT_CMD = "qsub wk.pbs"      # 提交作业的命令 (如 sbatch run.sh)
CHECK_CMD = "qstat"             # 检查作业运行状态的命令 (如 squeue)
JOB_ID_REGEX = r"(\d+)\.master" # 从提交命令中提取作业ID的正则表达式 (按服务器输出格式修改)
MAX_RETRIES = 3                 # 电子步不收敛或任务意外中断时的最大重启尝试次数

# --- Waiting Strategy / 监控等待策略 ---
MIN_CHECK_INTERVAL = 5     # 检查任务状态的初始/最小间隔时间 (秒)
MAX_CHECK_INTERVAL = 60    # 检查任务状态的最大间隔时间 (秒), 随等待时间递增降低查询频率

# --- Convergence Criteria / 收敛标准 ---
TARGET_TOLERANCE = 1.0      # 晶格应力的目标收敛阈值 (单位: kB)
MAX_ITER = 20               # 允许的最大自动优化迭代次数
MAX_STRAIN_PER_STEP = 0.04  # 每次迭代允许的最大形变比例限制 (防止结构突变或崩溃)

# --- Learning Alpha / 自适应学习率参数 (控制形变步长) ---
INITIAL_ALPHA = 0.0005 # 初始学习率 (应变/应力的比例)
MIN_ALPHA = 0.0001     # 自适应衰减的最小学习率限制
MAX_ALPHA = 0.0050     # 自适应放大的最大学习率限制

# ================== 2. LOGGING ENGINE (Optimized) ==================

class LogColors:
    RESET = "\033[0m"
    SUCCESS = "\033[92m"   # 绿色 (计算成功/邮件发送)
    ERROR = "\033[91m"     # 红色 (错误报告)
    ITER = "\033[1;35m"    # 紫色加粗 (Iteration)
    DATA = "\033[94m"      # 蓝色 (应力数据)

def c_log(msg, level="INFO"):
    """Force output to both stdout and file with English timestamps"""
    # 处理空行逻辑
    if not msg:
        print("", flush=True)
        try:
            with open("auto_opt.log", "a", encoding="utf-8") as f:
                f.write("\n")
        except: pass
        return

    t_str = time.strftime('%m-%d %H:%M:%S')
    
    # 仅对指定级别赋予颜色，INFO/WARN保持终端默认色
    color = ""
    if level == "ERROR": color = LogColors.ERROR
    elif level == "SUCCESS": color = LogColors.SUCCESS
    elif level == "ITER": color = LogColors.ITER
    elif level == "DATA": color = LogColors.DATA

    color_suffix = LogColors.RESET if color else ""
    colored_line = f"{color}{t_str} | [{level:^7}] {msg}{color_suffix}"
    
    print(colored_line, flush=True) 
    try:
        with open("auto_opt.log", "a", encoding="utf-8") as f:
            f.write(colored_line + "\n")
    except: pass

history_db = []
current_alphas = np.array([INITIAL_ALPHA, INITIAL_ALPHA, INITIAL_ALPHA])

# ================== 3. CORE SYMMETRY LOGIC ==================

def enforce_symmetry(file_path="POSCAR", is_final=False):
    if not ENABLE_SYMMETRY: return False
    if not HAS_SYMMETRY_LIB:
        c_log("ASE or SPGLIB not installed. Skipping symmetry.", "WARN")
        return False
        
    try:
        if not os.path.exists(file_path):
            c_log(f"{file_path} not found.", "WARN")
            return False
            
        atoms = read(file_path)
        
        orig_symbols = atoms.get_chemical_symbols()
        orig_order = []
        for sym in orig_symbols:
            if sym not in orig_order: orig_order.append(sym)
                
        cell_tuple = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
        
        prefix = "Final" if is_final else "Intermediate"
        c_log(f"Symmetry sweep for {prefix} {file_path}...", "INFO")
        
        best_dataset = None
        best_symprec = None
        max_sg_num = -1
        sweep_summaries = []
        
        for tol in TOLERANCE_SWEEP:
            ang_tol = -1.0 if tol < 0.1 else 5.0 
            dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=tol, angle_tolerance=ang_tol)
            
            if dataset is not None:
                sg_num = dataset['number']
                sweep_summaries.append(f"{tol}:{dataset['international']}")
                if sg_num > max_sg_num:
                    max_sg_num = sg_num
                    best_dataset = dataset
                    best_symprec = tol
            else:
                sweep_summaries.append(f"{tol}:FAIL")
                
        c_log(f"Sweep results: {', '.join(sweep_summaries)}", "INFO")

        if best_dataset is None:
            c_log("All tolerances failed to identify symmetry. Skipping.", "ERROR")
            return False

        c_log(f"Selected Best: Tol {best_symprec} -> SG {best_dataset['international']} ({best_dataset['number']})", "INFO")
        
        std_cell = spglib.standardize_cell(cell_tuple, to_primitive=False, no_idealize=False, symprec=best_symprec)
        
        if std_cell is not None:
            new_lattice, new_positions, new_numbers = std_cell
            
            if len(new_numbers) < len(atoms):
                c_log(f"Supercell detected ({len(atoms)} -> {len(new_numbers)} atoms). Reconstructing...", "WARN")
                C_orig = atoms.get_cell()
                C_std = new_lattice
                P = np.dot(C_orig, np.linalg.inv(C_std))
                P_int = np.round(P).astype(int)
                
                if np.max(np.abs(P - P_int)) > 0.05:
                    c_log("Transformation matrix deviation too high. Skipping reconstruction.", "WARN")
                    return False
                    
                std_atoms = Atoms(numbers=new_numbers, cell=new_lattice, scaled_positions=new_positions, pbc=True)
                final_atoms = make_supercell(std_atoms, P_int)
                
                if len(final_atoms) != len(atoms):
                    c_log("Atom count mismatch after reconstruction. Skipping.", "WARN")
                    return False
            else:
                final_atoms = Atoms(numbers=new_numbers, cell=new_lattice, scaled_positions=new_positions, pbc=True)
            
            sorted_indices = []
            for sym in orig_order:
                for i, atom in enumerate(final_atoms):
                    if atom.symbol == sym:
                        sorted_indices.append(i)
            final_atoms = final_atoms[sorted_indices]
            
            write(file_path, final_atoms, format='vasp', vasp5=True)
            c_log(f"{prefix} structure standardized & overwritten.", "INFO")
            return True
        else:
            c_log("Standardization returned None. Skipping.", "WARN")
            return False
            
    except Exception as e:
        c_log(f"CRITICAL ERROR in enforce_symmetry: {e}", "ERROR")
        return False

# ================== 4. COMMAND & MONITORING ==================

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('utf-8', errors='ignore').strip()
    except: return ""

def get_nelm_from_incar():
    nelm = 60
    if os.path.exists("INCAR"):
        try:
            with open("INCAR", 'r') as f:
                for line in f:
                    if "NELM" in line and "=" in line:
                        val = re.search(r"NELM\s*=\s*(\d+)", line)
                        if val: nelm = int(val.group(1))
        except: pass
    return nelm

def check_electronic_convergence():
    if not os.path.exists("OSZICAR"): return True 
    nelm = get_nelm_from_incar()
    last_steps = 0
    try:
        with open("OSZICAR", 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "DAV:" in line or "RMM:" in line:
                    parts = line.split()
                    if len(parts) > 1: last_steps = int(parts[1])
                    break
    except: pass

    if last_steps >= nelm:
        c_log(f"Electronic steps did NOT converge (reached NELM={nelm}).", "WARN")
        return False
    return True

def check_job_success():
    if not os.path.exists("OUTCAR"): return False
    try:
        fsize = os.path.getsize("OUTCAR")
        if fsize == 0: return False
        with open("OUTCAR", 'rb') as f:
            f.seek(max(0, fsize - 51200))
            if "General timing" in f.read().decode('utf-8', errors='ignore'):
                return True
    except: pass
    return False

def submit_job():
    out = run_cmd(SUBMIT_CMD)
    match = re.search(JOB_ID_REGEX, out)
    if match: return match.group(1)
    alt = re.search(r"(\d+)", out)
    return alt.group(1) if alt else None

def is_job_running(job_id):
    out = run_cmd(CHECK_CMD)
    if job_id not in out: return False
    for line in out.splitlines():
        if job_id in line and re.search(r"\s[CE]\s|\s[CE]$", line): return False
    return True

def wait_job(job_id):
    c_log(f"Job [{job_id}] queued/running. Monitoring...", "INFO")
    time.sleep(MIN_CHECK_INTERVAL)
    curr = MIN_CHECK_INTERVAL
    while is_job_running(job_id):
        time.sleep(curr)
        curr = min(curr * 1.2, MAX_CHECK_INTERVAL)
    c_log(f"Job [{job_id}] finished execution.", "SUCCESS")
    time.sleep(5)

# ================== 5. VISUALIZATION & NOTIFICATION ==================

def generate_optimization_plot():
    if not HAS_PLOT or not history_db: return None
    try:
        iters = [d['iter'] for d in history_db]
        s_xx = [d['stress'][0] for d in history_db]
        s_yy = [d['stress'][1] for d in history_db]
        s_zz = [d['stress'][2] for d in history_db]
        
        plt.figure(figsize=(8, 5))
        plt.plot(iters, s_xx, 'r-o', label='Stress XX')
        plt.plot(iters, s_yy, 'g-s', label='Stress YY')
        plt.plot(iters, s_zz, 'b-^', label='Stress ZZ')
        
        plt.axhline(y=TARGET_TOLERANCE, color='k', linestyle='--', alpha=0.3)
        plt.axhline(y=-TARGET_TOLERANCE, color='k', linestyle='--', alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        plt.title('VASP Optimization History')
        plt.xlabel('Iteration')
        plt.ylabel('Stress (kB)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        img_name = "opt_progress.png"
        plt.savefig(img_name)
        plt.close()
        return img_name
    except:
        return None

def send_email(status, start_time, iter_count=0, stress=None):
    if not ENABLE_EMAIL: return
    duration = time.time() - start_time
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    
    msg = MIMEMultipart()
    
    if SCRIPT_MODE == 'OPT':
        subject = f"[VASP OPT] {status} (Iter {iter_count})"
        s_str = f"XX={stress[0]:.2f}, YY={stress[1]:.2f}, ZZ={stress[2]:.2f}" if stress is not None else "N/A"
        body = (f"Status: {status}\n"
                f"WorkDir: {os.getcwd()}\n"
                f"Duration: {int(h)}h {int(m)}m {int(s)}s\n"
                f"Iterations: {iter_count}\n"
                f"Stress (kB): {s_str}\n")
        
        img_path = generate_optimization_plot()
        if img_path and os.path.exists(img_path):
            try:
                with open(img_path, 'rb') as f: img_data = f.read()
                msg.attach(MIMEImage(img_data, name=os.path.basename(img_path)))
            except: pass
    else:
        # 仅监控模式下的精简邮件内容 (无应力和曲线图)
        subject = f"[VASP Monitor] {status}"
        body = (f"Status: {status}\n"
                f"WorkDir: {os.getcwd()}\n"
                f"Duration: {int(h)}h {int(m)}m {int(s)}s\n")

    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, [RECEIVER_EMAIL], msg.as_string())
        server.quit()
        c_log("Email notification sent.", "SUCCESS")
    except Exception as e:
        c_log(f"Email failed: {e}", "ERROR")

# ================== 6. OPTIMIZATION LOGIC ==================

def read_stress_kb():
    if not os.path.exists("OUTCAR"): return None
    try:
        with open("OUTCAR", 'r') as f:
            for line in reversed(f.readlines()):
                if "in kB" in line:
                    parts = line.split()
                    try: return np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                    except: continue
    except: pass
    return None

def calculate_next_strain(current_stress):
    global current_alphas, history_db
    strains = np.zeros(3)

    for i in range(3):
        if len(history_db) >= 1:
            prev = history_db[-1]
            d_stress = current_stress[i] - prev['stress'][i]
            d_strain = prev['strain'][i]
            
            # 高阶策略: 震荡惩罚与信噪比过滤
            if current_stress[i] * prev['stress'][i] < 0:
                current_alphas[i] *= 0.5 
            elif abs(d_stress) > 1.0 and abs(d_strain) > 1e-5:
                local_alpha = abs(d_strain / d_stress)
                local_alpha = max(MIN_ALPHA, min(local_alpha, MAX_ALPHA))
                current_alphas[i] = 0.6 * current_alphas[i] + 0.4 * local_alpha

        mag = abs(current_stress[i]) * current_alphas[i]
        val = mag if current_stress[i] > 0 else -mag

        # 高阶策略: 动态收紧应变上限
        current_max = MAX_STRAIN_PER_STEP
        if abs(current_stress[i]) < 15.0:
            dynamic_limit = max(0.005, abs(current_stress[i]) * MAX_ALPHA)
            current_max = min(MAX_STRAIN_PER_STEP, dynamic_limit)

        if abs(val) > current_max:
            val = np.sign(val) * current_max

        # 高阶策略: 死区截断
        if abs(val) < 1e-4:
            val = 0.0

        strains[i] = val

    return strains

def update_poscar(current_stress):
    if not os.path.exists("CONTCAR") or os.path.getsize("CONTCAR") < 10: return False
    shutil.copy2("CONTCAR", "POSCAR")
    try:
        with open("POSCAR", 'r') as f: lines = f.readlines()
        vecs = [np.array([float(x) for x in lines[i].split()]) for i in range(2, 5)]
    except: return False

    eff_stress = np.copy(current_stress)
    if SYNC_MODE == 'AB':
        mean_v = np.mean(eff_stress[:2]); eff_stress[0] = eff_stress[1] = mean_v
    elif SYNC_MODE == 'ABC':
        mean_v = np.mean(eff_stress); eff_stress[:] = mean_v

    strains = calculate_next_strain(eff_stress)
    
    if SYNC_MODE == 'AB': strains[0] = strains[1] = np.mean(strains[:2])
    elif SYNC_MODE == 'ABC': strains[:] = np.mean(strains)

    history_db.append({'iter': len(history_db)+1, 'stress': np.copy(eff_stress), 'strain': np.copy(strains)})

    # 在所有轴计算并同步完成后，最终验证哪些轴允许被写入形变
    for i in range(3):
        if [OPT_A, OPT_B, OPT_C][i]: vecs[i] = vecs[i] * (1.0 + strains[i])
        lines[2+i] = f" {vecs[i][0]:22.16f} {vecs[i][1]:22.16f} {vecs[i][2]:22.16f}\n"

    with open("POSCAR", 'w') as f: f.writelines(lines)
    return True

# ================== 7. MAIN EXECUTION ==================

def main():
    start_t = time.time()
    open("auto_opt.log", "w").close() 
    c_log("=== VASP ENHANCED OPTIMIZER STARTED ===", "ITER")
    
    if SCRIPT_MODE == 'OPT':
        f_status = "Failed"; stress = None
        for i in range(1, MAX_ITER + 1):
            c_log("") # 插入空行
            c_log(f"--- Iteration {i} / {MAX_ITER} ---", "ITER")
            
            success = False
            for retry in range(MAX_RETRIES + 1):
                enforce_symmetry("POSCAR") 
                job_id = submit_job()
                if not job_id: time.sleep(30); continue
                wait_job(job_id)
                if not check_job_success():
                    c_log(f"Retry {retry}: Job incomplete.", "WARN")
                    if retry < MAX_RETRIES and os.path.exists("CONTCAR"):
                        shutil.copy2("CONTCAR", "POSCAR"); continue
                    else: break
                if not check_electronic_convergence(): break 
                success = True; break
            
            if not success: f_status = "Aborted"; break
            stress = read_stress_kb()
            if stress is None: break
            
            # 高亮输出应力信息
            s_str = f"XX={stress[0]:.2f}, YY={stress[1]:.2f}, ZZ={stress[2]:.2f}"
            c_log(f"Current Stress (kB): {s_str}", "DATA")
            
            converged = True
            if OPT_A and abs(stress[0]) > TARGET_TOLERANCE: converged = False
            if OPT_B and abs(stress[1]) > TARGET_TOLERANCE: converged = False
            if OPT_C and abs(stress[2]) > TARGET_TOLERANCE: converged = False
            
            if converged:
                f_status = "Converged"; c_log("Target reached! Optimization finished.", "SUCCESS"); break
            if not update_poscar(stress): break
        
        if os.path.exists("CONTCAR"):
            c_log("") # 插入空行
            c_log("Processing final structure symmetry...", "INFO")
            shutil.copy2("CONTCAR", "POSCAR")
            enforce_symmetry("POSCAR", is_final=True)
            shutil.copy2("POSCAR", "CONTCAR")
            
        send_email(f_status, start_t, i, stress)
    else:
        # MONITOR MODE
        jid = submit_job()
        if jid: 
            wait_job(jid)
            # 监控模式完成，直接读取 OUTCAR 尾部判断成功与否并发送精简邮件
            if check_job_success():
                m_status = "Success"
            else:
                m_status = "Failed/Incomplete"
            send_email(m_status, start_t)

if __name__ == "__main__":
    main()
'''

# ================= 配置区域结束 =================

# --- 自动加载配置 ---
if SELECT_MODE not in ADSORPTION_SCHEMES:
    raise ValueError(f"Unknown mode: {SELECT_MODE}")

CURRENT_CONFIG = ADSORPTION_SCHEMES[SELECT_MODE]
ADSORBATE_INPUT = CURRENT_CONFIG["molecule"]
BINDING_ELEMENT = CURRENT_CONFIG["binding"]
BOND_SCALE_FACTOR = CURRENT_CONFIG["scale"]
TILT_ANGLE = CURRENT_CONFIG["tilt"]
ENABLE_RANDOM_ROTATION = CURRENT_CONFIG["rotate"]

OUTPUT_ROOT_DIR = f"adsorption_{ADSORBATE_INPUT}_{SEARCH_DIRECTION}"

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    from scipy.spatial import Voronoi, Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# =========================================================================
# ★ 全新模块: 真实模拟手工输入的 Vaspkit 交互引擎 (破除 EOF 闪退) ★
# =========================================================================
def generate_potcar_with_vaspkit_interactive(target_dir, symbols):
    vaspkit_config = os.path.expanduser("~/.vaspkit")
    pp_path = None
    
    # 解析配置文件获取路径，主要是为了确认“无后缀/有后缀”到底存在与否
    if os.path.exists(vaspkit_config):
        with open(vaspkit_config, 'r') as f:
            for line in f:
                if line.strip().startswith("VASP_PP_PATH"):
                    parts = line.split("=")
                    if len(parts) == 2:
                        pp_path = parts[1].strip().strip("'\"")
                        break
                        
    candidate_bases = []
    if pp_path:
        candidate_bases.extend([
            pp_path,
            os.path.join(pp_path, "potpaw_PBE"),
            os.path.join(pp_path, "potpaw_PBE.54"),
            os.path.join(pp_path, "potpaw_PBE.52")
        ])
        
    preferred_map = {}
    for el in symbols:
        best_choice = el 
        found = False
        for base in candidate_bases:
            if found: break
            if not os.path.isdir(base): continue
            
            # 严格遵从你的法则：优先无后缀，若无则使用 _sv 版本
            if os.path.exists(os.path.join(base, el, "POTCAR")):
                best_choice = el
                found = True
            elif os.path.exists(os.path.join(base, f"{el}_sv", "POTCAR")):
                best_choice = f"{el}_sv"
                found = True
        preferred_map[el] = best_choice
        
    if not shutil.which("vaspkit"):
        print(f"Warning: 未找到 vaspkit 环境变量，跳过 {target_dir} 的 POTCAR 生成。")
        return
        
    try:
        # 使用 subprocess.Popen 并采用二进制通信，建立不中断的底层交互隧道
        p = subprocess.Popen(['vaspkit'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=target_dir)
        
        # 发起请求生成 POTCAR (功能指令 104)
        p.stdin.write(b"104\n")
        p.stdin.flush()
        
        buffer = ""
        while True:
            char_bytes = p.stdout.read(1)
            if not char_bytes: # 如果 vaspkit 结束运行
                break
                
            try:
                buffer += char_bytes.decode('utf-8')
            except: pass
            
            # 【抓包监控】：一旦终端弹出这句话，意味着 Vaspkit 停下来等待键盘输入了！
            if "please type the one you want" in buffer:
                # 用正则提取出它当前正在询问哪个元素
                match = re.search(r"is\s+([A-Za-z0-9_]+)\s*,", buffer)
                if match:
                    detected = match.group(1)            # 如 'Mo_sv'
                    elem = detected.split('_')[0]        # 提取 'Mo'
                    
                    # 查找我们在上一步钦定的完美答案
                    ans = preferred_map.get(elem, elem)
                    
                    # 真正模拟真人键盘敲击并回车
                    p.stdin.write(f"{ans}\n".encode('utf-8'))
                    p.stdin.flush()
                    
                    # 清空当前终端行缓存，为回答下一个元素做准备
                    buffer = ""
                    
        p.wait() # 确保生成完毕
    except Exception as e:
        print(f"VASPKIT Interactive Error: {e}")

# =========================================================================
# 分子构造与全表面边界扫描及中心折叠算法 (MIC Merging)
# =========================================================================
def get_ideal_bond_length(sym1, sym2):
    non_metals = {'H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I'}
    nm, m = None, None
    if sym1 in non_metals and sym2 not in non_metals:
        nm, m = sym1, sym2
    elif sym2 in non_metals and sym1 not in non_metals:
        nm, m = sym2, sym1

    if nm and m:
        z = atomic_numbers.get(m, 0)
        is_metal = (21 <= z <= 30) or (39 <= z <= 48) or (57 <= z <= 80) or (89 <= z <= 112) or z in [13, 31, 49, 50, 81, 82, 83]
        if is_metal:
            empirical_bonds = {
                'O': 1.95, 'N': 1.90, 'C': 1.95, 'S': 2.30, 'H': 1.65, 
                'F': 1.90, 'Cl': 2.35, 'P': 2.25, 'Si': 2.30
            }
            base_bond = empirical_bonds.get(nm, 2.0)
            if 39 <= z <= 48: base_bond += 0.08
            elif 57 <= z <= 80: base_bond += 0.12
            elif z in [13, 31, 49, 50, 81, 82, 83]: base_bond += 0.15
            if z in [21, 22, 23, 39, 40, 41, 71, 72, 73]: base_bond -= 0.06
            return base_bond

    z1 = atomic_numbers.get(sym1, 0)
    z2 = atomic_numbers.get(sym2, 0)
    r1 = covalent_radii[z1] if z1 < len(covalent_radii) else 1.0
    r2 = covalent_radii[z2] if z2 < len(covalent_radii) else 1.0
    
    electronegativities = {
        'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
        'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
        'Y': 1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 2.16, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93,
        'Lu': 1.27, 'Hf': 1.30, 'Ta': 1.50, 'W': 2.36, 'Re': 1.90, 'Os': 2.20, 'Ir': 2.20, 'Pt': 2.28, 'Au': 2.54
    }
    en1 = electronegativities.get(sym1, 2.0)
    en2 = electronegativities.get(sym2, 2.0)
    
    correction = 0.09 * abs(en1 - en2)
    bond = r1 + r2 - correction
    return bond

def calculate_smart_height(site_info, binding_symbol):
    if 'recommended_height' in site_info:
        return site_info['recommended_height']

    bonds = [get_ideal_bond_length(s, binding_symbol) for s in site_info['constituent_atoms']]
    target_bond_length = np.mean(bonds) * BOND_SCALE_FACTOR
    dist_xy = site_info['dist_xy']
    
    if target_bond_length > dist_xy:
        height = np.sqrt(target_bond_length**2 - dist_xy**2)
    else:
        height = 0.5 
    return max(height, 0.5)

def calculate_bridge_position_and_height(a1, a2, binding_sym):
    sym1, sym2 = a1.symbol, a2.symbol
    p1, p2 = a1.position, a2.position
    L_vec = p2 - p1
    L = np.linalg.norm(L_vec)
    
    d1 = get_ideal_bond_length(sym1, binding_sym) * BOND_SCALE_FACTOR
    d2 = get_ideal_bond_length(sym2, binding_sym) * BOND_SCALE_FACTOR
    
    if sym1 == sym2 or L < 1e-3:
        bridge_pos = (p1 + p2) / 2.0
        dist_xy = L / 2.0
        actual_h2 = d1**2 - dist_xy**2
        height = np.sqrt(actual_h2) if actual_h2 > 0 else 0.5
        return bridge_pos, dist_xy, max(height, 0.5)
        
    x = (L**2 + d1**2 - d2**2) / (2 * L)
    ratio = x / L
    ratio = np.clip(ratio, 0.15, 0.85)  
    
    bridge_pos = p1 + ratio * L_vec
    actual_x = ratio * L
    actual_h2 = d1**2 - actual_x**2
    height = np.sqrt(actual_h2) if actual_h2 > 0 else 0.5
    dist_xy = max(actual_x, L - actual_x)
    return bridge_pos, dist_xy, max(height, 0.5)

def optimize_molecule_geometry(input_str):
    if len(input_str) <= 2 and input_str in ["H", "O", "N", "C", "S"]:
        return Atoms(input_str, positions=[(0, 0, 0)]), "Single Atom"

    mol_atoms = None
    method = "Unknown"
    try:
        if input_str == "OOH":
             mol_atoms = Atoms('OOH', positions=[(0, 0, 0), (0, 0, 1.45), (0.9, 0, 1.7)])
             method = "Manual OOH"
        else:
            mol_atoms = molecule(input_str)
            method = "ASE Database"
    except: pass

    if mol_atoms is None and HAS_RDKIT:
        try:
            rd_mol = Chem.MolFromSmiles(input_str)
            if rd_mol:
                rd_mol = Chem.AddHs(rd_mol)
                if AllChem.EmbedMolecule(rd_mol, AllChem.ETKDG()) == 0:
                    AllChem.MMFFOptimizeMolecule(rd_mol)
                    conf = rd_mol.GetConformer()
                    mol_atoms = Atoms(symbols=[a.GetSymbol() for a in rd_mol.GetAtoms()], positions=conf.GetPositions())
                    method = "RDKit"
        except: pass

    if mol_atoms is None:
        if input_str == "OH": 
            mol_atoms = Atoms('OH', positions=[(0, 0, 0), (0, 0, 0.97)])
            method = "Manual"
    return mol_atoms, method

def prepare_adsorbate_ultimate(input_str, target_element, direction, tilt_angle, random_rot):
    ads, method = optimize_molecule_geometry(input_str)
    if ads is None: return None, None, None
    
    target_indices = [atom.index for atom in ads if atom.symbol == target_element]
    if not target_indices: return None, None, None
    binding_idx = target_indices[0]
    binding_sym = ads[binding_idx].symbol

    ads.translate(-ads.positions[binding_idx])
    if len(ads) > 1:
        other_indices = [i for i in range(len(ads)) if i != binding_idx]
        geom_center = np.mean(ads.get_positions()[other_indices], axis=0)
        if np.linalg.norm(geom_center) > 0.01:
            ads.rotate(geom_center, 'z') 

    if abs(tilt_angle) > 0.1 and len(ads) > 1:
        ads.rotate(tilt_angle, 'x')
    if random_rot and len(ads) > 1:
        rand_angle = random.uniform(0, 360)
        ads.rotate(rand_angle, 'z')
    if direction == "bottom":
        pos = ads.get_positions(); pos[:, 2] = -pos[:, 2]; ads.set_positions(pos)
        
    return ads, binding_sym

def get_surface_atoms(structure, center, auto_radius, direction="top"):
    """★ 全新重构：使用自适应半径智能拓展，保证包含完整表面的前提下防止算法爆炸 ★"""
    z_coords = structure.positions[:, 2]
    if direction == "top":
        threshold = z_coords.max() - SURFACE_THICKNESS
        base_atoms = [atom for atom in structure if atom.position[2] >= threshold]
    else:
        threshold = z_coords.min() + SURFACE_THICKNESS
        base_atoms = [atom for atom in structure if atom.position[2] <= threshold]

    has_pbc = any(structure.pbc) if hasattr(structure, 'pbc') else bool(np.any(structure.cell))
    if not has_pbc or not np.any(structure.cell):
        return base_atoms, base_atoms
        
    expanded_atoms = []
    xy_seen = set()
    buffer_radius = auto_radius + MAX_HOLLOW_RADIUS + 2.0
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            shift = dx * structure.cell[0] + dy * structure.cell[1]
            for a in base_atoms:
                new_pos = a.position + shift
                # 控制扩展阵列规模，仅仅获取覆盖所需区域内的原子，极大提升计算效率
                if np.linalg.norm(new_pos[:2] - center[:2]) <= buffer_radius:
                    xy = (round(new_pos[0], 3), round(new_pos[1], 3))
                    if xy not in xy_seen:
                        xy_seen.add(xy)
                        dummy_atom = Atoms(a.symbol, positions=[new_pos])[0]
                        expanded_atoms.append(dummy_atom)
                    
    return base_atoms, expanded_atoms

def get_circumcenter_2d(p1, p2, p3):
    x1, y1, x2, y2, x3, y3 = p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6: return None, None
    Ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    Uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    return np.array([Ux, Uy]), np.linalg.norm(np.array([Ux, Uy]) - np.array([x1, y1]))

def check_triangle_connectivity(p1, p2, p3, max_edge):
    d1 = np.linalg.norm(p1 - p2); d2 = np.linalg.norm(p2 - p3); d3 = np.linalg.norm(p3 - p1)
    return max(d1, d2, d3) <= max_edge

def get_safe_2d_atoms(surface_atoms, direction="top"):
    xy_seen = set()
    valid_atoms = []
    points_2d = []
    sorted_atoms = sorted(surface_atoms, key=lambda a: -a.position[2] if direction=="top" else a.position[2])
    for a in sorted_atoms:
        xy = (round(a.position[0], 4), round(a.position[1], 4))
        if xy not in xy_seen:
            xy_seen.add(xy)
            valid_atoms.append(a)
            points_2d.append(a.position[:2])
    return valid_atoms, np.array(points_2d)

def get_bridge_sites_delaunay(expanded_atoms, center, auto_radius, binding_sym, direction="top"):
    if not HAS_SCIPY or len(expanded_atoms) < 3: return []
    valid_atoms, points_2d = get_safe_2d_atoms(expanded_atoms, direction)
    if len(points_2d) < 3: return []
    
    try:
        tri = Delaunay(points_2d)
    except Exception:
        return []
        
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[0], simplex[2]))))
    
    candidates = []
    for (i, j) in edges:
        a1, a2 = valid_atoms[i], valid_atoms[j]
        dist_3d = np.linalg.norm(a1.position - a2.position)
        
        d1 = get_ideal_bond_length(a1.symbol, binding_sym) * BOND_SCALE_FACTOR
        d2 = get_ideal_bond_length(a2.symbol, binding_sym) * BOND_SCALE_FACTOR
        dynamic_max_bridge = min((d1 + d2) * DYNAMIC_BRIDGE_TOLERANCE, MAX_BRIDGE_HARD_LIMIT)
        
        if not (MIN_BRIDGE_LENGTH < dist_3d < dynamic_max_bridge): continue
        if abs(a1.position[2] - a2.position[2]) > MAX_BRIDGE_Z_DIFF: continue
            
        bridge_pos, dist_xy, rec_h = calculate_bridge_position_and_height(a1, a2, binding_sym)
        # 用自适应覆盖半径截断，代替死板的 [0,1) 切分
        if np.linalg.norm(bridge_pos[:2] - center[:2]) <= auto_radius:
            candidates.append({
                'pos': bridge_pos, 
                'anchor': f"{a1.symbol}-{a2.symbol}", 
                'constituent_atoms': [a1.symbol, a2.symbol], 
                'dist_xy': dist_xy,
                'recommended_height': rec_h
            })
    return candidates

def get_hollow_sites_voronoi(expanded_atoms, center, auto_radius, direction="top"):
    if not HAS_SCIPY or len(expanded_atoms) < 3: return []
    valid_atoms, points_2d = get_safe_2d_atoms(expanded_atoms, direction)
    if len(points_2d) < 3: return []
    
    try:
        vor = Voronoi(points_2d)
    except Exception:
        return []
        
    candidates = []
    for v_idx, vertex in enumerate(vor.vertices):
        dists = np.linalg.norm(points_2d - vertex, axis=1)
        min_dist = np.min(dists)
        
        if min_dist > MAX_HOLLOW_RADIUS: continue
        
        neighbor_indices = np.where(dists < min_dist + COORDINATION_TOLERANCE)[0]
        if len(neighbor_indices) < 3: continue 
        
        neighbors = [valid_atoms[i] for i in neighbor_indices]
        avg_z = np.mean([a.position[2] for a in neighbors])
        full_pos = np.array([vertex[0], vertex[1], avg_z])
        
        if np.linalg.norm(full_pos[:2] - center[:2]) <= auto_radius:
            candidates.append({'pos': full_pos, 'anchor': "Voronoi", 'constituent_atoms': [a.symbol for a in neighbors], 'dist_xy': min_dist})
    return candidates

def get_mic_distance(pos1, pos2, cell):
    """计算考虑周期性边界条件的最小镜像距离 (MIC)"""
    if cell is None or not np.any(cell):
        return np.linalg.norm(pos1 - pos2)
    inv_cell = np.linalg.inv(cell.T)
    diff = pos1 - pos2
    frac_diff = np.dot(inv_cell, diff)
    frac_diff -= np.round(frac_diff)  # 移位到 [-0.5, 0.5] 从而折叠进最近的镜像
    mic_diff = np.dot(cell.T, frac_diff)
    return np.linalg.norm(mic_diff)

def merge_duplicate_sites(structure, sites, center, env_tol=0.15):
    if not sites: return []
    has_pbc = any(structure.pbc) if hasattr(structure, 'pbc') else bool(np.any(structure.cell))
        
    def get_site_fingerprint(pos):
        dummy = structure.copy()
        dummy += Atoms('He', positions=[pos])
        he_index = len(dummy) - 1
        dists = dummy.get_distances(he_index, range(len(structure)), mic=has_pbc)
        mask = dists < ENV_CUTOFF
        near_atoms = [(structure[i].symbol, dists[i]) for i in np.where(mask)[0]]
        fp = {}
        for sym, d in near_atoms:
            if sym not in fp: fp[sym] = []
            fp[sym].append(d)
        for sym in fp:
            fp[sym] = np.sort(fp[sym])
        return fp
        
    def is_same_environment(fp1, fp2):
        if fp1.keys() != fp2.keys(): return False
        for sym in fp1:
            d1, d2 = fp1[sym], fp2[sym]
            min_len = min(len(d1), len(d2))
            if min_len == 0:
                if len(d1) > 0 or len(d2) > 0: return False
                continue
            if np.any(np.abs(d1[:min_len] - d2[:min_len]) > env_tol): return False
        return True

    # ★ 完美恢复“趋于中心”逻辑 ★
    # 在所有备选位点中，首先按与物理几何中心的距离排序
    sorted_sites = sorted(sites, key=lambda x: np.linalg.norm(x['pos'][:2] - center[:2]))
    
    unique_sites = []
    cell_matrix = structure.cell.array if has_pbc else None
    
    for site in sorted_sites:
        pos = site['pos']
        fp = get_site_fingerprint(pos)
        is_duplicate = False
        
        for u_site in unique_sites:
            # 引入真正的 MIC 空间重合判定：
            # 如果两个位点在物理上因为周期性折叠而贴在一起，它们就是同一个位点。
            # 而由于我们上面对 sorted_sites 进行了按中心距离排序，
            # 这里永远会保留距离晶格几何中心更近的那一个！
            mic_dist = get_mic_distance(pos, u_site['pos'], cell_matrix)
            if mic_dist < SPATIAL_MERGE_TOL:
                is_duplicate = True; break
                
            if is_same_environment(fp, u_site['fp']):
                is_duplicate = True; break
                
        if not is_duplicate:
            site['fp'] = fp
            unique_sites.append(site)
            
    for s in unique_sites: s.pop('fp', None)
    return unique_sites

def generate_sites_ultimate(structure, center, auto_radius, direction="top", binding_sym="H"):
    base_atoms, expanded_atoms = get_surface_atoms(structure, center, auto_radius, direction=direction)
    raw_top, raw_bridge, raw_hollow = [], [], []
    
    for a in base_atoms:
        raw_top.append({'pos': a.position, 'anchor': f"{a.symbol}", 'constituent_atoms': [a.symbol], 'dist_xy': 0.0})

    bridge_done = False
    hollow_done = False
    if HAS_SCIPY:
        scipy_bridge = get_bridge_sites_delaunay(expanded_atoms, center, auto_radius, binding_sym, direction)
        scipy_hollow = get_hollow_sites_voronoi(expanded_atoms, center, auto_radius, direction)
        if scipy_bridge:
            raw_bridge = scipy_bridge
            bridge_done = True
        if scipy_hollow:
            raw_hollow = scipy_hollow
            hollow_done = True

    if not bridge_done:
        for i, j in combinations(range(len(expanded_atoms)), 2):
            a1, a2 = expanded_atoms[i], expanded_atoms[j]
            dist_3d = np.linalg.norm(a1.position - a2.position)
            z_diff = abs(a1.position[2] - a2.position[2])
            
            d1 = get_ideal_bond_length(a1.symbol, binding_sym) * BOND_SCALE_FACTOR
            d2 = get_ideal_bond_length(a2.symbol, binding_sym) * BOND_SCALE_FACTOR
            dynamic_max_bridge = min((d1 + d2) * DYNAMIC_BRIDGE_TOLERANCE, MAX_BRIDGE_HARD_LIMIT)
            
            if (MIN_BRIDGE_LENGTH < dist_3d < dynamic_max_bridge) and (z_diff < MAX_BRIDGE_Z_DIFF): 
                bridge_pos, dist_xy, rec_h = calculate_bridge_position_and_height(a1, a2, binding_sym)
                if np.linalg.norm(bridge_pos[:2] - center[:2]) <= auto_radius:
                    raw_bridge.append({
                        'pos': bridge_pos, 'anchor': f"{a1.symbol}-{a2.symbol}", 
                        'constituent_atoms': [a1.symbol, a2.symbol], 'dist_xy': dist_xy, 'recommended_height': rec_h
                    })

    if not hollow_done:
        for combo in combinations(expanded_atoms, 3):
            p = np.array([c.position for c in combo])
            if not check_triangle_connectivity(p[0], p[1], p[2], MAX_HOLLOW_EDGE): continue 
            xy_c, xy_r = get_circumcenter_2d(p[0], p[1], p[2])
            if xy_c is not None and xy_r < MAX_HOLLOW_RADIUS: 
                base_pos = np.array([xy_c[0], xy_c[1], np.mean(p[:,2])])
                if np.linalg.norm(base_pos[:2] - center[:2]) <= auto_radius:
                    raw_hollow.append({'pos': base_pos, 'anchor': "Hollow", 'constituent_atoms': [c.symbol for c in combo], 'dist_xy': xy_r})

    valid_bridges = []
    for b in raw_bridge:
        if not any(np.linalg.norm(b['pos'][:2] - t['pos'][:2]) < CROSS_MERGE_TOL for t in raw_top):
            valid_bridges.append(b)

    valid_hollows = []
    for h in raw_hollow:
        if not any(np.linalg.norm(h['pos'][:2] - t['pos'][:2]) < CROSS_MERGE_TOL for t in raw_top):
            valid_hollows.append(h)
    valid_hollows.sort(key=lambda x: x['dist_xy'])

    surviving_bridges = []
    for b in valid_bridges:
        if not any(np.linalg.norm(b['pos'][:2] - h['pos'][:2]) < CROSS_MERGE_TOL for h in valid_hollows):
            surviving_bridges.append(b)

    # 传入中心点，交由底层的 MIC 算法进行折叠去重
    final_top = merge_duplicate_sites(structure, raw_top, center, ENV_MERGE_TOL)
    final_bridge = merge_duplicate_sites(structure, surviving_bridges, center, ENV_MERGE_TOL)
    final_hollow = merge_duplicate_sites(structure, valid_hollows, center, ENV_MERGE_TOL)
    
    return {'top': final_top, 'bridge': final_bridge, 'hollow': final_hollow}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Missing {INPUT_FILE}"); return
    
    print(f"Loading {INPUT_FILE}...")
    structure = read(INPUT_FILE)
    
    base_ads, binding_sym = prepare_adsorbate_ultimate(ADSORBATE_INPUT, BINDING_ELEMENT, SEARCH_DIRECTION, TILT_ANGLE, ENABLE_RANDOM_ROTATION)
    if base_ads is None: return
    
    # ★ 智能计算中心点以及可以包揽全表面的扫描半径 ★
    if np.any(structure.cell):
        cell = structure.cell.array
        corners = [
            np.array([0, 0, 0]),
            cell[0],
            cell[1],
            cell[0] + cell[1]
        ]
        cell_center_xy = np.dot([0.5, 0.5, 0], cell)[:2]
        center_z = np.mean(structure.get_positions()[:, 2]) 
        center = np.array([cell_center_xy[0], cell_center_xy[1], center_z])
        
        # 计算晶胞四角距离中心的最远距离，加一点 buffer，即可完美囊括原胞的表面区域！
        max_dist = max(np.linalg.norm(c[:2] - center[:2]) for c in corners)
        auto_radius = max_dist + 1.5 
    else:
        center = np.mean(structure.get_positions(), axis=0)
        auto_radius = 12.0
    
    method_str = "Delaunay/Voronoi Auto-Fallback" if HAS_SCIPY else "Combinatorial (No Scipy)"
    print(f"[*] Auto Bounds: Center-Seeking Enabled, Scanning radius = {auto_radius:.2f} Å")
    print(f"Generating sites using {method_str} method...")
    
    sites_dict = generate_sites_ultimate(structure, center, auto_radius, SEARCH_DIRECTION, binding_sym)
    
    if os.path.exists(OUTPUT_ROOT_DIR): shutil.rmtree(OUTPUT_ROOT_DIR)
    z_factor = 1.0 if SEARCH_DIRECTION == "top" else -1.0
    
    CIF_ROOT = os.path.join(OUTPUT_ROOT_DIR, "cif")
    VASP_ROOT = os.path.join(OUTPUT_ROOT_DIR, SELECT_MODE)
    os.makedirs(CIF_ROOT, exist_ok=True)
    os.makedirs(VASP_ROOT, exist_ok=True)
    
    final_results = []
    name_counters = {} 

    for stype in ['top', 'bridge', 'hollow']:
        sites = sites_dict.get(stype, [])
        if not sites: continue
        
        target_cif_dir = os.path.join(CIF_ROOT, stype)
        os.makedirs(target_cif_dir, exist_ok=True)
            
        for sinfo in sites:
            h = calculate_smart_height(sinfo, binding_sym)
            atoms_list = sorted(sinfo['constituent_atoms'])
            comp_str = "-".join(atoms_list)
            
            type_suffix_map = {'top': 'top', 'bridge': 'bri', 'hollow': 'hol'}
            type_suffix = type_suffix_map.get(stype, stype)
            
            if len(atoms_list) > 3:
                unique_elements = list(set(atoms_list))
                short_comp = f"{unique_elements[0]}{len(atoms_list)}" if len(unique_elements) == 1 else "Multi" 
                base_name = f"{short_comp}_{type_suffix}"
            else:
                base_name = f"{comp_str}_{type_suffix}"
            
            if base_name not in name_counters:
                name_counters[base_name] = 1
                site_name = base_name
            else:
                count = name_counters[base_name] + 1
                name_counters[base_name] = count
                site_name = f"{base_name}_{count}"

            cif_fname = f"{site_name}.cif"

            final = structure.copy()
            ads = base_ads.copy()
            ads.translate([0,0, h*z_factor])
            ads.translate(sinfo['pos'])
            final += ads
            
            write(os.path.join(target_cif_dir, cif_fname), final, format="cif")
            
            NON_METALS = {'H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Si', 'P', 'S', 'Cl', 'Ar', 'As', 'Se', 'Br', 'Kr', 'Te', 'I', 'Xe', 'Rn'}
            
            base_symbols = list(dict.fromkeys(structure.get_chemical_symbols()))
            ads_symbols = list(dict.fromkeys(ads.get_chemical_symbols()))
            
            metals = []
            non_metals = []
            for sym in base_symbols:
                if sym in NON_METALS:
                    non_metals.append(sym)
                else:
                    metals.append(sym)
                    
            adsorbates = []
            for sym in ads_symbols:
                if sym not in base_symbols:
                    adsorbates.append(sym)
                    
            if 'H' in adsorbates:
                adsorbates.remove('H')
                adsorbates.append('H')
            elif 'H' in non_metals: 
                non_metals.remove('H')
                non_metals.append('H')

            unique_symbols = metals + non_metals + adsorbates
            
            symbols_array = np.array(final.get_chemical_symbols())
            sort_indices = []
            for sym in unique_symbols:
                sort_indices.extend(np.where(symbols_array == sym)[0].tolist())
                
            final_vasp = final[sort_indices]
            
            target_vasp_dir = os.path.join(VASP_ROOT, site_name)
            os.makedirs(target_vasp_dir, exist_ok=True)
            
            # ★ 强制输出 Direct (分数坐标)，完全满足你的习惯
            write(os.path.join(target_vasp_dir, "POSCAR"), final_vasp, format="vasp", vasp5=True, sort=False, direct=True)
            
            # ★ 彻底告别管道报错，使用真实的后台键盘交互接口敲入答案！
            generate_potcar_with_vaspkit_interactive(target_vasp_dir, unique_symbols)
            
            with open(os.path.join(target_vasp_dir, "INCAR"), "w") as f:
                f.write(INCAR_CONTENT)
            
            with open(os.path.join(target_vasp_dir, "KPOINTS"), "w") as f:
                f.write(KPOINTS_CONTENT)
                
            with open(os.path.join(target_vasp_dir, "wk.pbs"), "w") as f:
                f.write(PBS_CONTENT)
                
            with open(os.path.join(target_vasp_dir, "vasp_infinite_wait.py"), "w") as f:
                f.write(WAIT_PY_CONTENT)
            
            mark = "*" if stype == 'bridge' and len(atoms_list) == 2 and atoms_list[0] != atoms_list[1] else ""
            
            final_results.append({
                'type': stype, 'composition': comp_str + mark,
                'height': h, 'binding_atom': binding_sym, 'site_name': site_name
            })
            
    final_results.sort(key=lambda x: (x['type'], x['composition'], x['height']))

    headers = ["ID", "Type", "Comp.", "H(A)", "Bind", "Site Name"]
    row_fmt = "{:<3} | {:<8} | {:<25} | {:<5} | {:<4} | {:<30}"
    print("\n" + "="*85)
    print(f"{'Ultimate Results (Auto Engine) | Mode: ' + SELECT_MODE:^85}")
    print("="*85)
    print(row_fmt.format(*headers))
    print("-" * 85)
    for i, res in enumerate(final_results):
        print(row_fmt.format(
            i + 1, res['type'].capitalize(), res['composition'],
            f"{res['height']:.2f}", res['binding_atom'], 
            res['site_name']
        ))
    print("="*85)
    print("Note: '*' in Comp. indicates an optimized Hetero-nuclear bridge site.")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()



