# GPT-2 间接宾语识别（IOI）项目

## 一、项目概述

本项目旨在研究 GPT-2 模型在执行 **间接宾语识别（Indirect Object Identification, IOI）** 任务时的注意力机制，并通过 **激活修补（Activation Patching）** 技术分析各层与注意力头（Attention Head）的贡献。同时，从体系结构角度评估系统在 **本地（Local）**、**云端（Remote）** 和 **混合（Hybrid）** 部署下的性能差异。

---

## 二、项目目标

1. 分析 GPT-2 模型如何通过注意力机制解决间接宾语识别任务。
2. 通过激活修补实验定量测量各层与注意力头的重要性。
3. 设计可在本地与云端灵活切换的模块化 Pipeline 系统。
4. 比较不同架构部署模式下的系统性能与计算效率。

---

## 三、系统架构

系统包含四个核心模块：

* **筛选（Filter）**：筛除预测错误的样本。
* **缓存（Collect）**：提取并缓存模型的中间激活值。
* **修补（Patch）**：替换激活值并计算 logits diff。
* **可视化（Plot）**：生成层与注意力头重要性的热力图。

### 部署模式

| 模式     | 描述                     |
| ------ | ---------------------- |
| Local  | 所有模块在本地执行。             |
| Remote | 所有模块在云端 GPU 上执行。       |
| Hybrid | 计算密集模块在云端运行，其余模块在本地执行。 |

---

## 四、实验设置

* **本地环境**：i7 CPU，16GB 内存，无 GPU。
* **云端环境**：NVIDIA 3080Ti，CUDA 12.1，Ubuntu 20.04。
* **依赖库**：Python 3.10、PyTorch 2.1、Transformers 4.36、Matplotlib、Seaborn。
* **数据规模**：100 对正常与损坏的 IOI 句子。

### 示例输入

```json
{
  "normal": "After John and Mary went to the store, John gave the bag to",
  "corrupted": "After John and Mary went to the store, Mary gave the bag to",
  "normal_target": "the coach",
  "corrupted_target": "the captain"
}
```

### 测试指标

* 上传时间（Upload Time）
* 下载时间（Download Time）
* 纯计算时间（Computation Time）
* 总执行时间（Wall Time）
* Logits Diff（修补效果指标）

---

## 五、实验结果

### 5.1 三种模式性能对比

实验结果来自 `timing_local_all.json`、`timing_remote_all.json` 和 `timing_hybrid.json`。

* **Local 模式**：计算时间长但无通信延迟。
* **Remote 模式**：GPU 加速显著，但存在网络传输开销。
* **Hybrid 模式**：计算与通信性能平衡，整体最优。

| 模块           | Local 平均耗时 (s) | Remote 平均耗时 (s) | Hybrid 平均耗时 (s) | 最优执行位置 |
| ------------ | -------------- | --------------- | --------------- | ------ |
| 筛选 (Filter)  | 0.8            | 0.9             | 0.8             | 本地     |
| 缓存 (Collect) | 2.1            | 0.7             | 0.7             | 云端     |
| 修补 (Patch)   | 3.2            | 1.1             | 1.1             | 云端     |
| 绘图 (Plot)    | 0.6            | 0.8             | 0.6             | 本地     |

### 5.2 激活修补分析结果

* 使用激活修补（Activation Patching）计算各层与注意力头对预测结果的影响。
* **Layer 9–10** 的注意力头对任务贡献最大。
* 中层（Layer 7–8）负责信息传递与整合。
* 低层（Layer 1–3）影响相对较弱。

### 5.3 可视化结果

`HeatMap.png` 展示了各层与注意力头的重要性分布，颜色深浅代表对 logits diff 的贡献强度。

---

## 六、讨论与结论

### 6.1 性能分析与架构优化

* 云端 GPU 显著提升计算密集任务（Collect、Patch）的执行速度。
* 网络 I/O 是 Remote 模式的主要瓶颈，占总耗时约 20%。
* Hybrid 模式综合性能最佳，总体时间比 Local 模式减少约 40%。

### 6.2 模块划分策略的合理性

* 计算密集模块放置于云端、I/O 模块保留在本地可最大化性能收益。
* 模块粒度划分实现了体系结构层面的性能可控与灵活调度。

### 6.3 模型机制与可解释性

* GPT-2 在 IOI 任务中表现出分层的注意力机制：

  * 低层：识别句子结构与实体；
  * 中层：传递信息；
  * 高层：整合语义并输出预测。
* 模型关键机制集中于 Layer 9–10，对预测正确性影响最大。

### 6.4 综合结论

1. 模块化体系结构显著提升系统运行效率与可扩展性。
2. Hybrid 部署模式在计算与通信之间实现最优平衡。
3. 激活修补实验验证了 GPT-2 模型在语言关系任务中的层级化机制。
4. 本研究为后续大模型可解释性与分布式体系结构优化提供了参考。

---

## 七、项目结构

```
IOI-GPT2-ActivationPatching/
│
├── data/
│   ├── names.json
│   ├── sentences.json
│   ├── data.json
│   ├── data_check1.json
│   ├── data_check2.json
│
├── src/
│   ├── generate_data.py
│   ├── check_model_output.py
│   ├── patching_experiment.py
│   └── visualize.py
│
├── results/
│   └── HeatMap.png
│
├── report/
│   └── final_report.pdf
│
├── requirements.txt
└── README.md
```

---

## 八、运行方式

```bash
pip install -r requirements.txt
python src/run_experiment.py
```

在 `configs/hybrid.json` 中可修改模块运行位置（local / remote / hybrid）。
