# VASP/ASE 实用脚本集

面向 DFT 计算科研人的 VASP + ASE + Phonopy 辅助脚本工具箱，覆盖**建模 → 提交计算 → 后处理 → 绘图 → 全流程自动化**的完整链路。在 Linux 集群（PBS）环境下开箱即用。

📖 **每个脚本的详细使用方法见 [docs/使用说明.md](docs/使用说明.md)**

## 功能总览

| 模块 | 脚本 | 一句话功能 |
|------|------|-----------|
| 结构优化与拟合 | `autovasp`、`setup_eos`、`nihe.py` | 晶胞全自动迭代优化（对称性保护+自适应学习率）、EOS 批量建模与 Birch-Murnaghan 拟合 |
| K 点与绘图 | `KPATHS.py`、`bandcal`、`doscal`、`mdcal` | 高对称路径自动寻找、能带/PDOS/MD 能量轨迹论文级绘图 |
| NEB 过渡态 | `idpp.py`、`huitu.py` | IDPP 插点生成 NEB 初猜、能垒曲线与活化能自动标注 |
| 声子计算 | `mkphono`、`phonopy.py`、`phonocal`、`xuping.py`、`band_conf` | 超胞一键生成分发、全自动声子谱+虚频三档诊断、ASR+对称性修正消虚频 |
| 磁性 | `afm.py`、`jtpd` | AFM 磁构型自动枚举、磁基态（NM/FM/AFM/FiM）智能判定 |
| 吸附建模 | `slab.py`、`slabcover.py`、`coverage.py`、`check_ads` | 顶/桥/空位自动识别+计算文件夹一键生成、H 覆盖度演化、底物形变评估 |
| 结合能 | `mol_to_vasp.py`、`single_atom.py`、`binding_energy.py`、`yuansutongji.py` | 分子/单原子参考计算自动部署、结合能一键汇总 |
| 吸附计算 | `frozen`、`tijiao` | 批量冻结原子、批量放开表面层并提交 |
| 电化学 | `lsv.py` | Butler-Volmer 模型 HER 极化曲线 |
| 结构分析 | `center`、`structure_inspect.py`、`chenjian.py`、`classify.py`、`classify_high.py` | PBC 居中、重叠检查、成键判断、CALYPSO 结构按维度/空间群归档 |
| 元素替换 | `replace.py`、`replace_POSCAR.py` | CIF / POSCAR（递归）批量元素替换 |
| 快捷工具 | `ISPIN`、`suball`、`check_force`、`check_poscar`、`clear_*` 系列、`Shortcuts.sh` | 参数快改、批量提交、受力查询、输入检查、一键清理 |
| 电荷分析 | `bader` | Henkelman Bader 电荷分析可执行程序 |
| 热稳定性 | `check_md` | AIMD 前后结构多维评估（能量热力学主导判定） |
| 工作流 | `workflow`、`workfm`、`workzpe`、`workbd` | 单体六步高通量、FM 快速补算、ZPE 频率部署、能带+DOS 全自动出图 |

## 环境要求

- **OS**：Linux 集群（bash 脚本、PBS/qsub 提交、Linux 二进制）
- **Python**：python3 + 以下库

  ```bash
  pip install ase seekpath numpy scipy matplotlib spglib phonopy pyyaml
  ```

- **外部工具**：vaspkit（POTCAR/K 路径）、phonopy（声子）、PBS 集群（qsub）

## 快速开始

```bash
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>

# 方式一：安装到 PATH（推荐）
mkdir -p ~/Script && find . -maxdepth 3 -type f \( -name "*.py" -o ! -name "*.*" \) -exec cp {} ~/Script/ \;
echo 'export PATH=~/Script:$PATH' >> ~/.bashrc && source ~/.bashrc
chmod u+x ~/Script/*
# 之后在任意计算目录直接输入脚本名使用，如: jtpd、suball、bandcal

# 方式二：直接调用
bash 工作流/fm优化工作流/workfm
python 结合能/binding_energy.py
```

（可选）一键安装常用终端快捷命令（`gr` 查收敛、`mag` 查磁矩、`ccp` 复制 CONTCAR 等）：

```bash
bash 快捷指令安装/Shortcuts.sh
```

## ⚠️ 使用前必读

1. **邮件通知需自行配置**：带邮件功能的脚本中，真实邮箱/授权码已替换为占位符
   `your_qq_number@qq.com` 和 `在此填入QQ邮箱授权码`，用前请按
   [docs/使用说明.md · 附录 B](docs/使用说明.md#附录-b邮件通知配置) 填入自己的信息，或设 `ENABLE_EMAIL = False` 关闭。
2. **清理脚本不可逆**：`clear_all` / `clear_cal` / `clear_fold` 会直接删除计算文件，
   使用前请确认（详见使用说明第十二节）。
3. **多版本脚本请用最新版**：slab 用 8.0、autovasp 用 7.0、phonocal 和 check_md 用 3.0，
   旧版本仅保留供追溯。

## 目录结构

```
.
├── README.md
├── .gitignore
├── docs/
│   ├── 使用说明.md              # ← 全部脚本的详细使用方法（必读）
│   └── autovasp使用说明书.docx   # autovasp 专项手册
├── python托管自动优化/           # autovasp v1~v7（晶胞自动优化引擎）
├── 优化拟合/                    # EOS 建模与拟合
├── K点寻找/  band+dos绘制/  md绘制/
├── IDPP过渡态插点/
├── 声子/                        # 含手动超胞 + 自动后处理 v1~v3
├── 消除虚频/
├── afm寻找和建立/  基态判断/
├── 吸附模型搭建/                # slab v1~v8 + coverage_H 系列
├── 吸附计算/                    # frozen / tijiao
├── 结合能/
├── lsv曲线/
├── 结构居中/  结构检查/  成键判断/  结构分类/
├── 批量替换元素/
├── 快捷指令/  快捷指令安装/      # 含批量提交 suball
├── vasp文件夹清理/               # clear_* 系列
├── 应力判断/
├── bader电荷/
├── 热稳定性判断/                 # check_md v1~v3 + 原理图
└── 工作流/                      # workflow / workfm / workzpe / workbd
```

## 说明

- 脚本在 PBS 集群（qsub + wk.pbs）环境下开发，使用 Slurm 的用户需修改脚本中的
  `SUBMIT_CMD`/`CHECK_CMD`（如 `sbatch`/`squeue`）。
- `bader` 二进制来自 [Henkelman 课题组](http://theory.cm.utexas.edu/henkelman/code/bader/)，版权归原作者。
- 欢迎 issue / PR 交流改进。
