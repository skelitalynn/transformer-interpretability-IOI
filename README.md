# IOI 项目 - 混合云机制可解释性实验

## 📖 项目简介

本项目实现 **IOI (Indirect Object Identification, 间接宾语识别)** 任务的机制可解释性研究，通过对 GPT-2 Small 模型进行**激活值修补（Activation Patching）**分析，识别并量化各注意力头在 IOI 任务中的因果贡献。

### 核心特性

- ✅ **灵活的混合云执行**：每个 GPU 密集环节可独立选择本地或云端执行
- ✅ **详细的性能分析**：记录纯计算时间、通信时间、墙上时间
- ✅ **与原始 Notebook 完全一致**：计时结构对标 `IOI.ipynb`
- ✅ **自动化编排**：一键运行本地/云端/混合模式
- ✅ **可视化对比**：生成三种模式的性能对比报告

## 🚀 快速开始

### 1. 安装依赖

**本地环境**：

```bash
pip install torch transformer-lens matplotlib seaborn tqdm paramiko jaxtyping
```

### 2. 下载模型到本地（首次必需）

**在本地**运行以下命令下载 GPT-2 模型到缓存：

```bash
python -c "from transformer_lens import HookedTransformer; HookedTransformer.from_pretrained('gpt2-small')"
```

这会将模型下载到 `~/.cache/huggingface/`（首次约 1-2 分钟）。

### 3. 准备远端环境（如需远端执行）

登录云服务器并安装依赖（使用 conda 或系统 Python 均可）：

```bash
ssh -p 端口 用户名@主机地址

# 安装依赖
pip install torch transformer-lens matplotlib seaborn tqdm jaxtyping

# 验证
python3 -c "import transformer_lens; print('OK')"
exit
```

**然后在本地上传模型缓存**：

```bash
python upload_model_cache.py
```

这会将本地的 GPT-2 缓存上传到远端（约 500MB，需 2-5 分钟）。

### 4. 运行实验

#### 全本地模式（推荐首次运行）

```bash
python ioi_orchestrator.py --config configs/local_all.json
```

#### 混合模式（筛选和绘图本地，重计算云端）

```bash
python ioi_orchestrator.py --config configs/hybrid.json
```

#### 全云端模式

```bash
python ioi_orchestrator.py --config configs/remote_all.json
```

### 3. 对比三种模式

依次运行三种配置后：

```bash
python compare_reports.py timing_local_all.json timing_remote_all.json timing_hybrid.json
```

## 📂 项目结构

```
IOI/
├── IOI.ipynb                  # 原始实验笔记本
├── ioi_orchestrator.py        # 核心编排器（支持混合云执行）
├── ioi_modules.py             # GPU密集模块（filter/collect/patch/plot）
├── ioi_local_pre.py           # 本地数据准备（generate/check）
├── compare_reports.py         # 三种模式性能对比工具
├── upload_model_cache.py      # 模型缓存上传工具
├── configs/                   # 配置文件目录
│   ├── local_all.json        # 全本地配置
│   ├── remote_all.json       # 全云端配置
│   └── hybrid.json           # 混合模式配置
├── names.json                # 姓名数据（A/B对）
├── sentences.json            # 句子模板
├── QUICKSTART.md             # 详细使用指南
└── README.md                 # 本文件
```

## 🔧 配置说明

### 执行位置配置

编辑 `configs/hybrid.json` 自定义执行策略：

```json
{
  "execution": {
    "filter_gpt2": "local",           // 可选: local 或 remote
    "collect_activations": "remote",  // 可选: local 或 remote
    "patch_activations": "remote",    // 可选: local 或 remote
    "plot_heatmap": "local"           // 可选: local 或 remote
  },
  "ssh": {
    "host": "your.cloud.server",
    "port": 22,
    "username": "root",
    "password": "your_password"
  }
}
```

### SSH 配置（远程执行必需）

根据远端环境选择配置方式：

**使用 conda 环境**：

```json
{
  "ssh": {
    "host": "connect.westc.gpuhub.com",
    "port": 19337,
    "username": "root",
    "password": "your_password",
    "python_bin": "python",
    "conda_env": "ioi",
    "setup_cmd": "eval \"$(conda shell.bash hook)\""
  }
}
```

**使用系统 Python**：

```json
{
  "ssh": {
    "host": "your.server.com",
    "port": 22,
    "username": "root",
    "password": "your_password",
    "python_bin": "python3",
    "conda_env": "",
    "setup_cmd": ""
  }
}
```

## 📊 计时字段说明

### 字段结构（完整）

每个环节记录 **4 种时间维度**：

| 字段后缀         | 含义       | 示例值       | 说明                            |
| ---------------- | ---------- | ------------ | ------------------------------- |
| `_time`          | 纯计算时间 | 10.18s       | 与 IOI.ipynb 一致的模块内部耗时 |
| `_wall_time`     | 墙上时间   | 10.25s       | 实际耗时（含进程/SSH开销）      |
| `_upload_time`   | 上传时间   | 0.82s        | 仅远端执行时有                  |
| `_download_time` | 下载时间   | 3.35s        | 仅远端执行时有                  |
| `_total_time`    | 总时间     | 8.94s        | 计算+上传+下载（远端）          |
| `_location`      | 执行位置   | local/remote | 标记                            |

### 示例：混合模式计时报告

```json
{
  "generate_data_time": 0.001,
  "generate_data_wall_time": 0.073,
  "check_structure_time": 0.001,
  "check_structure_wall_time": 0.072,
  
  "filter_gpt2_time": 11.191,
  "filter_gpt2_wall_time": 11.264,
  "filter_gpt2_location": "local",
  
  "collect_activations_time": 4.762,
  "collect_activations_upload_time": 0.821,
  "collect_activations_download_time": 3.353,
  "collect_activations_total_time": 8.936,
  "collect_activations_wall_time": 8.995,
  "collect_activations_location": "remote",
  
  "patch_activations_time": 33.495,
  "patch_activations_upload_time": 13.730,
  "patch_activations_download_time": 1.231,
  "patch_activations_total_time": 48.455,
  "patch_activations_wall_time": 48.521,
  "patch_activations_location": "remote",
  
  "plot_heatmap_time": 0.467,
  "plot_heatmap_wall_time": 0.540,
  "plot_heatmap_location": "local"
}
```

## 🎯 实验流程

### 数据流向图

```
本地准备（固定本地）
  ├─ 生成数据 (generate)
  └─ 结构校验 (check)
        ↓
GPU密集环节（可配置本地/云端）
  ├─ GPT-2样本筛选 (filter)      ← 可配置
  ├─ 缓存激活值 (collect)         ← 可配置
  ├─ 修补激活值 (patch)           ← 可配置
  └─ 绘制热力图 (plot)            ← 可配置
        ↓
产物输出
  ├─ saved_data.pt
  ├─ results.pt
  ├─ HeatMap.png
  └─ timing_*.json
```

### 与 IOI.ipynb 的对应关系

| Notebook Cell                | 模块函数                           | 计时字段                   |
| ---------------------------- | ---------------------------------- | -------------------------- |
| `generate_data()`            | `ioi_local_pre.py --step generate` | `generate_data_time`       |
| `check_sentence_structure()` | `ioi_local_pre.py --step check`    | `check_structure_time`     |
| `filter_with_gpt2()`         | `ioi_modules.py --task filter`     | `filter_gpt2_time`         |
| `get_clean_activations()`    | `ioi_modules.py --task collect`    | `collect_activations_time` |
| `activation_patching()`      | `ioi_modules.py --task patch`      | `patch_activations_time`   |
| `plot_attention_heatmap()`   | `ioi_modules.py --task plot`       | `plot_heatmap_time`        |

## 🔬 GPU 敏感度分析

根据实验数据（全本地模式）：

| 环节       | 耗时   | GPU敏感度  | 推荐执行位置    |
| ---------- | ------ | ---------- | --------------- |
| 生成数据   | 0.001s | ⚪️ 无       | 本地            |
| 结构校验   | 0.001s | ⚪️ 无       | 本地            |
| GPT-2筛选  | 11.2s  | 🟡 中等     | 本地/云端均可   |
| 缓存激活值 | 10.2s  | 🟡 中等     | 取决于网络      |
| 修补激活值 | 214.9s | 🔴 **极高** | **云端GPU推荐** |
| 绘制热力图 | 0.6s   | ⚪️ 无       | 本地            |

**结论**：

- `patch_activations` 是最耗时的环节（占总时间 90%+），强烈建议放云端GPU执行
- `collect_activations` 次之，但输出文件较大（saved_data.pt），需权衡通信开销
- `filter_gpt2` 和 `plot` 相对轻量，建议本地执行

## 📈 性能对比示例

基于实际测试数据：

```
【计算环节耗时对比】
环节                    全本地         全云端       混合模式        最优
--------------------------------------------------------------------------
生成数据                0.001s        0.001s       0.001s         -
结构校验                0.001s        0.001s       0.001s         -
GPT-2筛选              11.19s        12.34s       11.19s         全本地
缓存激活值             10.18s         4.76s        4.76s         全云端
修补激活值            214.87s        33.50s       33.50s         全云端
绘图                    0.61s         0.82s        0.47s         混合

【通信开销】（混合模式）
缓存-上传：0.82s  | 缓存-下载：3.35s
修补-上传：13.73s | 修补-下载：1.23s

【总耗时】
全本地：237.0s  |  全云端：51.4s  |  混合：60.8s  ⭐最优：全云端
```

## 🛠️ 高级用法

### 单独执行某个模块

```bash
# 仅筛选
python ioi_modules.py --task filter --input data_check1.json --output data_check2.json

# 仅缓存激活值
python ioi_modules.py --task collect --input data_check2.json --output saved_data.pt

# 仅修补
python ioi_modules.py --task patch --input saved_data.pt --output results.pt

# 仅绘图
python ioi_modules.py --task plot --input results.pt --output HeatMap.png
```

### 自定义混合策略

根据你的网络带宽和GPU性能，调整 `configs/hybrid.json`：

**场景1：网络快，GPU慢** → 全云端

```json
{"execution": {"filter_gpt2": "remote", "collect_activations": "remote", 
               "patch_activations": "remote", "plot_heatmap": "remote"}}
```

**场景2：网络慢，GPU快** → 全本地

```json
{"execution": {"filter_gpt2": "local", "collect_activations": "local",
               "patch_activations": "local", "plot_heatmap": "local"}}
```

**场景3：网络中等，仅重计算上云** → 混合

```json
{"execution": {"filter_gpt2": "local", "collect_activations": "local",
               "patch_activations": "remote", "plot_heatmap": "local"}}
```

## 🔍 常见问题

### Q1: 为什么 `_time` 和 `_wall_time` 不一样？

- `_time`：模块内部纯计算时间（与 IOI.ipynb 一致）
- `_wall_time`：外部实际墙上时间（含进程启动、SSH开销等）

**示例**：

```
generate_data_time: 0.001s       ← 纯逻辑
generate_data_wall_time: 0.073s  ← 含Python启动 (0.072s开销)
```

### Q2: 远端执行失败怎么办？

**检查清单**：

1. SSH连接是否正常：`ssh -p 端口 用户@主机`
2. 远端 conda 环境是否创建：`conda env list`
3. 远端依赖是否安装：`pip list | grep transformer-lens`
4. 模型缓存是否上传：运行 `python upload_model_cache.py`

### Q3: `upload_time` 和 `download_time` 包含什么？

- **upload**：本地文件通过 SFTP 上传到远端服务器的时间
- **download**：远端结果文件通过 SFTP 下载到本地的时间
- 不包括 SSH 握手（一次性开销）

**文件大小参考**：

- `data_check2.json`: ~15KB
- `saved_data.pt`: ~50MB（较大！）
- `results.pt`: ~5KB
- `HeatMap.png`: ~200KB

### Q4: 如何优化混合模式性能？

**策略**：

1. 将**输出文件小、计算重**的环节放云端（如 `patch`）
2. 将**输出文件大、计算轻**的环节放本地（避免传输 `saved_data.pt`）
3. 根据 `wall_time` 实际测试调优

**示例优化**：

```json
{
  "filter_gpt2": "local",           // 轻量，本地即可
  "collect_activations": "local",   // 避免传输大文件saved_data.pt
  "patch_activations": "remote",    // 最耗时，云端GPU加速
  "plot_heatmap": "local"           // 轻量，本地即可
}
```

### Q5: 离线模式如何工作？

`ioi_modules.py` 在开头设置：

```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

这会强制 Hugging Face 只使用本地缓存，不联网下载。

## 📦 产物文件

运行后生成：

| 文件               | 大小   | 说明              |
| ------------------ | ------ | ----------------- |
| `data.json`        | ~15KB  | 生成的50个样本    |
| `data_check1.json` | ~15KB  | 结构校验后的样本  |
| `data_check2.json` | ~15KB  | GPT-2筛选后的样本 |
| `saved_data.pt`    | ~50MB  | 激活值缓存        |
| `results.pt`       | ~5KB   | Patch结果矩阵     |
| `HeatMap.png`      | ~200KB | 注意力头热力图    |
| `timing_*.json`    | ~2KB   | 详细计时报告      |

## 🎓 技术细节

### IOI 任务定义

给定句子：

- **Clean**: "After A and B went to the store, A gave a bottle of milk to"
- **Corrupted**: "After A and B went to the store, B gave a bottle of milk to"

模型预测：

- Clean 应预测 → B
- Corrupted 应预测 → A

### Activation Patching 原理

对每个注意力头 `(layer, head)`：

1. 在 corrupted 句子前向时
2. 将该头的激活值 `z` 替换为 clean 句子的对应值
3. 观察 logits 变化，计算恢复程度

指标：`(patched - corrupted) / (clean - corrupted)`

### 热力图解读

- **红色（正值）**：该头有助于恢复正确预测
- **蓝色（负值）**：该头阻碍正确预测
- **数值大小**：因果贡献强度

## 🌐 云服务器配置示例

### AutoDL / SeetaCloud

```json
{
  "ssh": {
    "host": "connect.westb.seetacloud.com",
    "port": 21946,
    "username": "root",
    "password": "your_password",
    "python_bin": "python",
    "conda_env": "ioi",
    "setup_cmd": "eval \"$(conda shell.bash hook)\""
  }
}
```

### 阿里云 / 腾讯云

```json
{
  "ssh": {
    "host": "your.ip.address",
    "port": 22,
    "username": "ubuntu",
    "pkey_path": "~/.ssh/id_rsa",
    "password": null,
    "conda_env": "base"
  }
}
```

## 📚 扩展阅读

- **[QUICKSTART.md](QUICKSTART.md)** - 详细使用教程
- **[IOI.ipynb](IOI.ipynb)** - 原始实验笔记本
- **[Mechanistic Interpretability 论文](https://arxiv.org/abs/2211.00593)** - IOI 任务出处

## 🤝 贡献

本项目基于以下开源工具：

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - 机制可解释性工具库
- [GPT-2](https://github.com/openai/gpt-2) - OpenAI 的预训练语言模型
- [Paramiko](https://www.paramiko.org/) - Python SSH 库

## 📝 更新日志

**v2.0** (2025-11-11)

- ✨ 新增混合云执行支持
- ✨ 详细计时系统（纯计算/墙上/通信时间）
- ✨ 模块化架构重构
- ✨ 三种模式自动对比工具

**v1.0** (初始版本)

- ✅ 基于 IOI.ipynb 的单机实验


**最后更新**: 2025-11-11
