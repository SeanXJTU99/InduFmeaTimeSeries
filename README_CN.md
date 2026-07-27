# 工业 FMEA Agent — 多级低温精馏智能化诊断系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-Kernel-6f42c1)](https://triton-lang.org)
[![vLLM](https://img.shields.io/badge/vLLM-0.5%2B-00b2a9)](https://github.com/vllm-project/vllm)
[![TensorRT-LLM](https://img.shields.io/badge/TensorRT--LLM-0.12-76b900)](https://github.com/NVIDIA/TensorRT-LLM)
[![Jetson](https://img.shields.io/badge/Jetson-AGX%20Orin-76b900)](https://developer.nvidia.com/embedded/jetson-agx-orin)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-ff6f00)](https://github.com/langchain-ai/langgraph)
[![AWQ](https://img.shields.io/badge/Quant-AWQ%20INT4-8b5cf6)](https://github.com/mit-han-lab/llm-awq)
[![License](https://img.shields.io/badge/License-Proprietary-red)]()

面向同位素富集用多级低温精馏设备的 AI 预测性维护与 FMEA 系统。
集成 **西门子 PLC 实时流**、**异步 Excel 同位素丰度报表**、
**串口 RS485 字节流**，统一纳入 Agent 闭环。

[English Version](README.md)

## 三阶段技术演进 (2025.04 – 2026.05)

| 阶段 | 时间 | 核心技术 |
|------|------|---------|
| 1 — 基础 | 2025.04–08 | 卡尔曼-小波级联、DTW 对齐、虚拟软测量、物性约束异常检测、EWMA+KDE 自适应基线、RAG 四层防幻觉 |
| 2 — Agent | 2025.09–12 | LangGraph 有向图智能体、BM25+BGE 混合检索+重排、约束解码+Pydantic+护栏、QLoRA SFT+GRPO/DPO 对齐、AWQ INT4 量化 |
| 2b — 性能 | 2025.12 | **内存:** DAF 消除小波缓冲区；字典量化(Float32→8-bit, PLC 存储 50%↓)；3D 布尔矩阵(500KB L3)替代 JSON Schema 链；Bilinks 邻接表释放~500MB GPU 显存。**异构:** 协方差(5×5→15) DMA 直通 Orin NPU(~50μs→~5μs)。**延迟:** O(1) 矩阵查表<1ns vs ~10μs JSON Schema；O(1) 硬时钟对齐 vs O(N²) DTW；O(V+E) BFS vs O(N×D) 向量搜索。零额外硬件。 |
| 2c — 推理 | 2026.01 | Flash Attention(KWT Encoder ~40%↓)；Prefix Cache 预热(system prompt 编码→0)；ngram 投机采样(TTFT ~50%↓, 零配置)；Triton 融合算子(KWT 前处理~30%↓)；xgrammar 结构化输出(JSON Schema→FSM, O(1) token 掩码)。全部热插拔，无硬件变更。 |
| 3 — 智能 | 2026.01–05 | 卡尔曼-小波-Transformer 级联、PPO+MCTS 强化学习、反事实推理、Jetson AGX Orin 边缘 DMA/NPU 部署 |

## 推理优化 (Phase 2c)

| 优化项 | 技术 | 延迟影响 |
|--------|------|---------|
| Flash Attention | PyTorch 2.0+ SDPA Flash backend | Encoder ~40%↓ |
| Prefix Cache 预热 | 启动时预计算共享 system prompt 的 KV | Prompt 编码→0 |
| ngram 投机采样 | vLLM n-gram 匹配, 无需草稿模型 | TTFT ~50%↓ |
| Triton 融合算子 | 小波拼接+线性投影+位置编码 单 kernel | 前处理 ~30%↓ |
| xgrammar 结构化输出 | JSON Schema→FSM, O(1) token 掩码 | Token 生成 ~10-20%↑ |

端到端延迟: **~150ms → ~20ms** (Phase 1 基线的 7.5×)。

## 安全 — 矩阵守护 + LLM 降级

1. **Matrix Guard (O(1) 硬门禁)** — 物理不可行规则预配置为 3D 布尔矩阵, 单次数组查表替代 JSON Schema + Pydantic 全链
2. **FMEA Bilinks 因果图** — 从告警传感器 BFS, 限定因果拓扑；BM25+BGE 向量搜索保留为新型故障回退
3. **引用追踪** — 每条诊断主张必须引用 FMEA 源行；未引用=拒绝
4. **xgrammar 结构化输出** — JSON Schema 编译为紧凑有限状态机, O(1) token 掩码替代逐 token Python 检查

## 项目结构

```
├── src/
│   ├── signal/          # DAF 卡尔曼、小波、DTW、时频图、软测量、BCO 对齐器、字典压缩器
│   ├── detection/       # 物性约束检测器、自适应基线、特征工程
│   ├── rag/             # 文档加载、语义重写、分块、嵌入、混合检索、重排、元数据过滤、Bilinks 图
│   ├── safety/          # 矩阵守护、xgrammar 结构化输出、约束解码、Pydantic 校验、护栏、引用追踪
│   ├── prompt/          # 拓扑注入、安全拒绝模板
│   ├── agent/           # LangGraph 状态、图谱、节点、路由、上下文管理
│   ├── training/        # SFT 数据集、QLoRA、DPO 数据集、DPO 训练、LoRA 合并
│   ├── deploy/          # AWQ 量化、vLLM/TensorRT-LLM 配置、Jetson 部署、DMA 配置、协方差打包、vLLM 预热
│   ├── models/          # KWT 级联、多尺度嵌入、卡尔曼反馈、Triton 融合算子
│   └── rl/              # 精馏 Gym 环境、PPO 控制器、MCTS 规划器、反事实推理
├── configs/             # 全部模块 YAML 配置
├── data/mock/           # 虚构 FMEA 样本、PLC 流、串口二进制
├── docs/                # 架构文档、数据脱敏声明
├── tests/               # 测试套件
├── Dockerfile.edge      # Jetson AGX Orin 容器
├── Dockerfile.server    # L40S 服务器容器
├── docker-compose.yml   # 边缘+服务端编排
└── requirements.txt
```

## 数据脱敏声明

**全部 PLC 位号、阀门编号、精馏塔代号、FMEA 条目均为虚构。**
详见 `docs/data_notice.md`。

## 许可

专有。仅供演示与作品集展示。
