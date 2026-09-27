# 多尺度全波形反演（geowave.fwi）

在现有射线层析（`geowave.ray`）基础上增加的声波全波形反演作业包，用于近地表速度
由低频到高频逐级更新。2D 常密度声波方程，时间二阶 / 空间四阶有限差分，顶部自由
表面，其余边界海绵吸收层。

## 功能与对应实现

| 需求 | 实现 |
| --- | --- |
| 生成合成炮集 | `MultiscaleFWI.generate_synthetic_shots()`（每炮 `(nt, nrec)`） |
| 伴随梯度 | `AcousticModeler2D.adjoint_gradient()`，伴随源为检波点残差；对离散正演算子的精确转置（已经暴力 Jacobian 与有限差分验证到 1e-9 量级） |
| 约束速度范围 | 盒约束 `[vmin, vmax]`，每次试步后 `np.clip` 投影 |
| 可回退线搜索 | `armijo_backtracking()`：Armijo 充分下降 + 回溯；模型只有在被接受时才更新，失败自动回退到上一快照并进入下一频段 |
| 每轮目标函数与模型快照 | `objective_log.csv`（每轮一行）、`linesearch_trials.jsonl`（每次试步）、`snapshots/band*_iter*_model.npy`（含梯度快照）、`model_latest.npy`、`model_final.npy`、`run_manifest.json` |
| 非有限值中止 | 观测数据、每炮合成数据、每炮伴随梯度、堆叠梯度均做 `require_finite` 检查；发现 NaN/Inf 即中止，原因写入 `run.log` 与 `run_manifest.json`，并保存 `model_aborted.npy`，CLI 退出码 3 |

## 快速开始

```bash
# 1) 伴随梯度正确性验证（有限差分对比，应 < 1e-2）
python -m geowave.fwi.run --gradient-test

# 2) 自测演示：内置真实模型生成观测数据，从 1D 渐变初始模型反演
python -m geowave.fwi.run --self-test --workdir runs/demo --bands 2.5:4,5:4,10:4

# 3) 生产作业：射线层析模型作初始模型 + 观测炮集目录
python -m geowave.fwi.run --workdir runs/prod \
    --obs-dir observed_npy --init-model raytomo.npy \
    --nx 401 --nz 121 --dx 5 --tmax 1.5 --shots 21 \
    --bands 3:6,5:6,8:8,12:8 --vmin 700 --vmax 2600
```

- `--bands cutoff_hz:迭代数,...`：必须按低频到高频排列（配置校验强制）。
- 观测数据格式：`obs-dir` 下 `shot_*.npy`，每炮 `(nt, nrec)`，与配置的时间轴 /
  检波器数量一致（不一致会中止并说明期望形状）。
- 初始模型：`--init-model model.npy`，形状 `(nz, nx)`；可直接给射线层析结果，
  也可用上次中断的 `snapshots/model_latest.npy` 续算。
- 子波：反演使用 `--wavelet-peak` 的 Ricker 子波，每个频段与观测数据做相同的
  零相位低通（`scipy.signal.sosfiltfilt`），保证尺度一致。
- 照明预条件（`illumination_preconditioner`，默认开）：用各炮正演波场能量近似
  Hessian 对角，补偿深部弱照明；`illumination_epsilon`（默认 0.1）限制最大放大
  倍数，越小放大越激进（过小会在照明阴影区放大伪影，建议 0.03~0.1）。

## 输出目录结构

```
workdir/
  config.json              # 全部反演参数
  objective_log.csv        # 每轮: band, iter, objective, step, |g|max, vmin/max, accepted, snapshot
  linesearch_trials.jsonl  # 线搜索每次试步的目标函数
  run.log                  # 运行日志（含中止原因）
  run_manifest.json        # 状态(completed/aborted)、原因、目标函数序列、快照清单
  snapshots/
    model_init.npy
    band00_iter000_model.npy / band00_iter000_gradient.npy ...
    model_latest.npy       # 最近被接受的模型（回退点）
    model_final.npy
    model_aborted.npy      # 仅中止时
```

## 中止策略

出现以下任一情况立即中止并说明原因（`run.log` 与 manifest 中 `status=aborted`）：
观测/滤波后数据含非有限值；某炮合成数据含非有限值（通常意味着 CFL 不稳定或模型
非法）；某炮或堆叠梯度含非有限值；观测数据形状与配置不符；dt 违反 CFL 限制。
色散风险提示（每最短波长 < 5 个网格点）只警告不中止。

## 当前局限

- 2D 常密度声波，无弹性/衰减/自由表面多次波专门处理。
- 正演波场全量存内存（每炮 `nt × nz × nx × 8` 字节）；大模型需换检查点或边界存储策略。
- 最速下降方向（梯度平滑 + 归一化）；如需 L-BFGS 可在 `inversion.py` 的方向计算处替换。
- 纯 NumPy 引擎，演示规模足够；生产规模建议替换为 numba/C 内核（接口不变）。
