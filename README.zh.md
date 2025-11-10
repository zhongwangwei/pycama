# PyCaMa - CaMa-Flood 的 Python 实现

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

CaMa-Flood（Catchment-based Macro-scale Floodplain）水动力模型的 Python 实现，为大尺度河道汇流和洪水淹没模拟提供高效灵活的框架。

[English](README.md) | 简体中文

## 目录

- [概述](#概述)
- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [工作流程](#工作流程)
- [配置说明](#配置说明)
- [性能表现](#性能表现)
- [已知限制](#已知限制)
- [项目结构](#项目结构)
- [测试](#测试)
- [贡献](#贡献)
- [引用](#引用)
- [许可证](#许可证)

## 概述

PyCaMa 是 CaMa-Flood 模型的 Python 重新实现，原始模型使用 Fortran 编写。该模型使用局地惯性方程方法，在大陆至全球尺度模拟河流流量、水位和洪水淹没。模型采用单元集水区离散化方案，高效表示河网和洪泛区动力学。

**与原始 CaMa-Flood 的主要区别：**

- **编程语言**：纯 Python 实现（原版为 Fortran 90）
- **精度**：全程使用 Float64 以保证数值稳定性
- **性能**：比 Fortran 慢约 50%（持续优化中）
- **准确性**：与 Fortran 版本相比误差约 1%

## 特性

### 核心功能

- ✅ **河网生成**：从全球地图提取区域河网
- ✅ **NetCDF 初始化**：将二进制河网数据转换为 CF 规范的 NetCDF
- ✅ **洪水演算模拟**：局地惯性方程与自适应时间步长
- ✅ **分汊支持**：模拟河流分汊和分流
- ✅ **洪泛区动力学**：可选的洪泛区淹没和蓄水
- ✅ **灵活重启**：支持小时/天/月/年频率的重启
- ✅ **多格式 I/O**：支持 NetCDF 和二进制文件
- ✅ **径流插值**：从粗分辨率输入网格保守降尺度

### 物理选项

- 河流流动的局地惯性近似
- 可选运动波方案（计算更快）
- 自适应子时间步进以保证数值稳定性
- Manning 糙率参数化
- 河道-洪泛区相互作用
- 水面高程计算

### 实验/未完成功能

- ⚠️ **水库运行**：代码已实现但未充分测试
- ⚠️ **调试追踪器**：运行时单元格追踪（非生产环境）
- ⚠️ **地下水延迟**：功能存在但需要验证

## 安装

### 前置要求

- Python 3.8 或更高版本
- NumPy
- SciPy
- netCDF4-python
- （可选）xarray 用于数据分析

### 安装步骤

1. **克隆仓库：**

```bash
git clone https://github.com/yourusername/pycama.git
cd pycama
```

2. **安装依赖：**

```bash
pip install numpy scipy netCDF4
```

或使用 conda：

```bash
conda install numpy scipy netcdf4
```

3. **验证安装：**

```bash
python src/main.py --help
```

### 可选：解压测试数据

**仅当您想要运行包含的测试用例时才需要：**

```bash
# 解压测试模拟的初始化文件
cd output/Global15min/Initialization
unzip grid_routing_data.nc.zip
cd ../../..
```

**注意：** 如果您正在生成自己的河网（使用 `--grid` 和 `--init` 选项），则不需要解压此文件。

## 快速开始

### 运行测试用例

仓库包含一个完整的测试用例，进行 3 天模拟（1980-01-01 至 1980-01-03）。

**首先，解压测试数据**（参见上面的[可选：解压测试数据](#可选解压测试数据)）：

```bash
cd output/Global15min/Initialization
unzip grid_routing_data.nc.zip
cd ../../..
```

**然后运行模拟：**

```bash
# 仅运行模型模拟（使用预生成的数据）
python src/main.py nml/namelist-15min.input --run-only
```

**预期输出：**
- 河流流量、水深和蓄水量保存在 `output/Global15min/model_output/`
- 输出文件按月组织：`Global15min_198001.nc`

### 基本使用模式

```bash
# 仅生成区域河网
python src/main.py nml/namelist.input --grid-only

# 转换为 NetCDF（初始化）
python src/main.py nml/namelist.input --init-only

# 运行洪水模拟
python src/main.py nml/namelist.input --run-only

# 运行特定组合
python src/main.py nml/namelist.input --grid --init
```

## 工作流程

PyCaMa 包含三个顺序工作流：

```
┌─────────────────────────────────────────────────────────────┐
│  1. 河网生成                                                 │
│  输入：全球河网地图（二进制）                                 │
│  输出：区域河网 → output/{case}/rivermap/                    │
│        - nextxy.bin, params.txt, diminfo.txt 等              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 初始化                                                   │
│  输入：二进制河网文件                                        │
│  输出：NetCDF 文件 → output/{case}/Initialization/           │
│        - grid_routing_data.nc (CF 规范)                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 模型模拟                                                 │
│  输入：NetCDF 初始化文件 + 强迫数据（径流）                  │
│  输出：时间序列 → output/{case}/model_output/                │
│        - 流量、水深、洪泛范围                                │
└─────────────────────────────────────────────────────────────┘
```

## 配置说明

所有配置通过 Fortran 风格的 namelist 文件完成（例如 `nml/namelist-15min.input`）。

### 主要配置部分

#### 1. 选项（控制运行哪些工作流）

```fortran
&OPTIONS
  run_grid = .false.    ! 河网生成
  run_init = .false.    ! 初始化为 NetCDF
  run_sim  = .true.     ! 模型模拟
/
```

#### 2. 输出设置

```fortran
&OUTPUT
  output_base_dir = 'output/'
  case_name = 'Global15min'  ! 所有输出保存到 output/Global15min/
/
```

#### 3. 河网生成

```fortran
&RiverMap_Gen
  global_map_dir = '/path/to/global/map/'
  west = -180.0    ! 区域边界
  east = 180.0
  south = -90.0
  north = 90.0

  run_inpmat = .true.        ! 生成输入矩阵
  run_params = .true.        ! 计算参数
  run_bifurcation = .true.   ! 处理分汊
  run_dam = .false.          ! 水库分配（实验性）
/
```

#### 4. 模型物理

```fortran
&MODEL_RUN
  ! 时间设置
  syear = 1980
  smon  = 1
  sday  = 1
  eyear = 1980
  emon  = 1
  eday  = 3

  dt = 3600                  ! 时间步长 [秒]
  ifrq_inp = 24              ! 强迫频率 [小时]
  ifrq_out = 24              ! 输出频率 [小时]

  ! 物理选项
  ladpstp  = .true.          ! 自适应时间步长
  lpthout  = .true.          ! 分汊方案
  ldamout  = .false.         ! 水库运行
  lfplain  = .true.          ! 洪泛区方案
  lkine    = .false.         ! 运动波（vs 局地惯性）

  ! 强迫数据
  linpcdf  = .true.
  crofdir_nc = './data/GRFR_0p25/'
  crofpre_nc = 'RUNOFF_remap_sel_'
  crofsuf_nc = '.nc'
  forcing_file_freq = 'yearly'  ! 'single', 'yearly', 'monthly', 'daily'
/
```

#### 5. 重启配置

```fortran
&MODEL_RUN
  lrestart = .false.         ! 从重启文件开始
  creststo = ''              ! 输入重启文件

  ! 重启频率
  ifrq_rst = 1               ! 每 1 个单位重启
  cfrq_rst_unit = 'month'    ! 单位：'hour', 'day', 'month', 'year'

  lrestcdf = .true.          ! NetCDF 格式（vs 二进制）

  ! 默认重启目录：output/{case_name}/restart/
/
```

## 性能表现

### 计算性能

| 指标 | PyCaMa | Fortran CaMa-Flood | 比率 |
|--------|--------|-------------------|-------|
| **速度** | 慢约 50% | 基准 | 0.5x |
| **内存** | Float64 精度 | Float32/64 混合 | ~1.3x |
| **准确性** | 误差约 1% | 基准 | 99% |

**性能说明：**

- Python 实现全程使用 float64 以保证数值稳定性
- Python 中的自适应时间步长开销高于 Fortran
- 计划的未来优化：
  - 关键循环的 Numba JIT 编译
  - 大区域的稀疏矩阵操作
  - 独立河段的并行处理

### 基准测试（全球 15 分分辨率，3 天模拟）

```
区域：1440 x 720 网格单元
活动单元：~250,000
时间步数：72（1 小时时间步长）
挂钟时间：~1-2 分钟（单核）
```

## 已知限制

### 未完成功能

1. **水库运行**（`ldamout = .true.`）
   - 实现完成但未验证
   - 水库分配可用，但释放计算需要测试
   - **状态**：使用风险自负

2. **调试追踪器**（`src/model_run/trace_debug.py`）
   - 用于逐单元格追踪的运行时调试工具
   - 不适用于生产运行
   - **状态**：仅开发工具

3. **地下水延迟**（`lgdwdly`）
   - 代码存在但需要验证
   - **状态**：实验性

### 已知问题

1. **数值差异**
   - 与 Fortran 版本偏差约 1%
   - 主要原因：
     - Float64 vs 混合精度
     - 不同的编译器优化
     - 中间计算的舍入

2. **性能**
   - 目前比 Fortran 慢约 50%
   - 主要开销在自适应时间步长和 I/O
   - 持续优化中

3. **内存使用**
   - 由于 float64 导致更高的内存占用
   - 基于序列的数组尚未完全优化

### 平台特定说明

- **macOS**：已测试并工作正常
- **Linux**：应该可以工作（未广泛测试）
- **Windows**：可能需要调整路径

## 项目结构

```
pycama/
├── src/
│   ├── main.py                      # 统一入口
│   ├── river_network/               # 河网生成
│   │   ├── workflow.py              # 流程编排
│   │   ├── region_tools.py          # 区域切割
│   │   ├── param_tools.py           # 参数计算
│   │   ├── dam_param_tools.py       # 水库处理
│   │   └── namelist.py              # 配置解析
│   ├── initialization/              # NetCDF 转换
│   │   └── grid_routing_init.py
│   └── model_run/                   # 模拟引擎
│       ├── runner.py                # 主编排器
│       ├── physics.py               # 水动力方程
│       ├── forcing.py               # 输入数据处理
│       ├── output.py                # NetCDF 输出
│       ├── time_control.py          # 时间步进
│       ├── restart.py               # 重启 I/O
│       └── dam_operation.py         # 水库释放（实验性）
├── nml/
│   └── namelist-15min.input         # 测试配置
├── output/
│   └── Global15min/                 # 测试用例输出
│       ├── rivermap/                # 生成的河网
│       ├── Initialization/          # NetCDF 初始化
│       │   └── grid_routing_data.nc.zip  # 需要解压！
│       └── model_output/            # 模拟结果
├── data/                            # 输入数据（不在仓库中）
├── CLAUDE.md                        # AI 助手指南
└── README.md                        # 本文件
```

## 测试

### 运行测试用例

仓库包含一个全球 15 分分辨率的完整测试用例：

```bash
# 1. 解压初始化文件（必需！）
cd output/Global15min/Initialization
unzip grid_routing_data.nc.zip
cd ../../..

# 2. 运行 3 天模拟
python src/main.py nml/namelist-15min.input --run-only

# 3. 检查结果
ls output/Global15min/model_output/
# 预期：Global15min_198001.nc
```

### 验证结果

```python
import netCDF4 as nc

# 打开输出文件
ds = nc.Dataset('output/Global15min/model_output/Global15min_198001.nc')

# 检查变量
print(ds.variables.keys())
# 预期：'rivout', 'rivsto', 'rivdph', 'outflw' 等

# 查看流量时间序列
discharge = ds.variables['rivout'][:]  # 形状：(time, y, x)
print(f"流量范围：{discharge.min():.2f} - {discharge.max():.2f} m³/s")
```

## 贡献

欢迎贡献！改进方向：

- **性能优化**：Numba、Cython 或并行处理
- **验证**：更多测试用例和基准测试
- **水库运行**：测试和验证
- **文档**：更多示例和教程
- **测试**：单元测试和持续集成

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发指南。

## 引用

如果您在研究中使用 PyCaMa，请引用：

**原始 CaMa-Flood 模型：**
```
Yamazaki, D., Kanae, S., Kim, H., & Oki, T. (2011).
A physically based description of floodplain inundation dynamics in a global river routing model.
Water Resources Research, 47(4), W04501. doi:10.1029/2010WR009726
```

**PyCaMa（本工作）：**
```
[您的引用 - 发表后请更新]
```

## 致谢

- 原始 CaMa-Flood 模型由 Prof. Dai Yamazaki（东京大学）开发
- Fortran 代码库：http://hydro.iis.u-tokyo.ac.jp/~yamadai/cama-flood/

## 许可证

本项目采用 Apache License 2.0 许可 - 详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题和反馈：
- **Issues**：[GitHub Issues](https://github.com/yourusername/pycama/issues)
- **讨论**：[GitHub Discussions](https://github.com/yourusername/pycama/discussions)

## 版本历史

- **v0.1.0**（当前）- 初始 Python 实现
  - 核心功能可用
  - 相对 Fortran 准确度 ~1%
  - 性能慢约 50%
  - 水库和追踪器功能未完成

---

**注意**：这是研究级实现。在用于生产或发表前，请针对您的特定应用验证结果。
