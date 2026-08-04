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
| 2c — 推理 | 2026.01 | Flash Attention(KWT Encoder ~40%↓)；Prefix Cache 预热(system prompt 编码→0)；ngram 投机采样(TTFT ~50%↓, 零配置)；Triton 融合算子(KWT 前处理~30%↓)；xgrammar 结构化输出(JSON Schema→FSM, O(1) token 掩码)；KV Cache 量化(L40S FP8 / Orin INT8, KV 显存 50%↓)。全部热插拔，无硬件变更。 |
| 3 — 智能 | 2026.01–05 | 卡尔曼-小波-Transformer 级联、PPO+MCTS 强化学习、反事实推理、Jetson AGX Orin 边缘 DMA/NPU 部署 |

## 技术选型 — 理由与实现细节

### 1. 信号处理（DAF 卡尔曼-小波级联）

- **两级卡尔曼** — *为什么:* 第一级 (Q=1e-5, R=1e-3) 吸收 EMI/测量噪声，第二级 (Q=1e-6, R=5e-3) 跟踪物理过程；按位号独立调 Q/R（TE-301 一级 R=1e-2 vs FT-301 R=2e-3），而非全局单一协方差。*怎么实现:* 标量状态预测/更新 + innovation 剪裁；DAF（Deterministic Annealing Filter, Fruehwirth & Strandlie 1999）在 32 样本滑窗（步长 16）内做批量卡尔曼+联合权重 — beta 对数退火 100→0.1 共 5 步，权重 <0.1 判为离群。
- **小波降噪 (db4, 3 级)** — *为什么:* db4 紧支撑贴合工业瞬态；软阈值保留趋势，硬阈值会削掉。*怎么实现:* universal 阈值 σ·√(2 ln n)，σ 用 MAD（median/0.6745）；按信号类型覆盖（压力 2 级、流量 db6 2 级、温度 4 级、阀位 1 级+SURE）。
- **BCO 硬时钟对齐 + DTW 回退** — *为什么:* DTW O(N²) 无法实时；PLC/串口/Excel 流自带 NTP 时间戳，对齐可降到 O(1)。*怎么实现:* 每数据源注册 delay_ns 偏移，事件按 ntp_ns+delay 分桶；全部源在 max_spread_ns=10 ms 内到齐才算对齐。DTW（Sakoe-Chiba 带、平方欧氏）保留为 NTP 不可用时的回退。
- **虚拟软测量** — *为什么:* 同位素丰度来自异步化验报表，无在线仪表。*怎么实现:* XGBoost（可选 LSTM）5 特征、60 样本回看、预测步长 1；物理先验回退钳位 [0,100]%。
- **字典压缩** — *为什么:* 8-bit 字典量化使 PLC 流存储减半，RMS 误差 <0.5% 量程。*怎么实现:* 唯一值贪心凝聚聚类压缩到 ≤2^nbits 个质心；流式分块 1024。

### 2. RAG 知识工程

- **语义重写** — *为什么:* 原始 FMEA 表分块后因果链断裂。*怎么实现:* 行级重写为固定 S/O/D 句式，保留 RPN=S×O×D 关联。
- **混合检索 (BM25 0.4 + dense 0.6, RRF k=60)** — *为什么:* 稠密向量匹配不到设备位号 (TE-301)；BM25 精确命中 tag，bge-large-en-v1.5 (1024 维) 兜语义；RRF 基于排名融合，无需分数归一化。*怎么实现:* 两路各 top-50 → RRF Σ w/(k+rank+1) → top-10；cross-encoder (ms-marco-MiniLM-L-6-v2) 重排 20→5。
- **FMEA Bilinks 因果图** — *为什么:* 向量检索给的是主题相关而非因果相关的命中，用于诊断是危险的。*怎么实现:* 4 类节点（传感器/故障模式/根因/缓解）、3 类双向边（observes/caused_by/mitigated_by）；从告警传感器 BFS 限深 3；BM25+BGE 保留为新型故障回退。

### 3. 安全与结构化输出

- **Matrix Guard (3D 布尔矩阵)** — *为什么:* 物理不可能（丰度>100%）必须硬拦截而非"纠正"；50×2000×5 (500KB) 布尔矩阵 O(1) 查表 <1ns，JSON Schema 链则 ~10µs。*怎么实现:* numpy 切片批量 block/allow；路由三态：block / pass（severity≤INFO 免 LLM 模板应答）/ llm。
- **约束解码 + Pydantic + Guardrails** — *为什么:* 三层防御应对不同失效点。*怎么实现:* outlines logit 掩码约束 JSON Schema（enum: 21 位号、10 故障模式；model_validator 强制 rpn=S×O×D）；guardrails 钳位物理边界（丰度 [0,100]、阀位 [0,100]%）并拒收 "100% sure" 类措辞。
- **xgrammar 结构化输出** — *为什么:* 逐 token Python logit 检查 ~10µs/token；编译为 FSM 后掩码 O(1)。*怎么实现:* Grammar.from_json_schema → FSM，每解码步生成 token 掩码；vLLM `--guided-decoding-backend xgrammar`；缺库回退 Pydantic+Guardrails。
- **引用追踪** — *为什么:* 无引用的诊断主张不可核验，工业审计要求可溯源。*怎么实现:* 4 种引用模式；得分=有效引用/总 claims；claim 用启发式识别（位号 `[A-Z]{2,}-\d{3}` 或含数字句）。

### 4. LangGraph Agent

- **为什么选 StateGraph:** 有向图+条件边是确定性、可审计的，优于自由 agent 循环 — 高危工艺需要可证明的路由。*怎么实现:* ContextResolver → FMEAReasoner → ReportGenerator；置信门 (<0.60) → SystemFallback；Reflection 节点批判草稿，feedback_gate 允许一轮重试；checkpointer 支持人在回路。
- **意图分类** — 关键词法（每类 15/10/8 个关键词），置信度=命中占比，可选 LLM few-shot。不训分类器：零训练成本、行为确定。
- **上下文管理** — 4000 token 预算、CJK 感知估算（中文÷2 + ASCII÷4）、FIFO 裁剪 + 抽取式摘要（10 个领域关键词、800 字摘要），先裁剪再进 LLM。

### 5. 后训练（SFT + DPO/GRPO + 量化）

- **QLoRA 微调 Qwen2.5-7B-Instruct** — *为什么:* 单卡 L40S 可跑 4-bit NF4 微调；r=64/α=128 保留工业领域容量。*怎么实现:* NF4+双重量化、bf16 计算、全 7 个线性层、3 epochs、有效 batch 32、lr 2e-4 cosine、paged AdamW-8bit、flash-attn；SFT 数据 ≤5000 条（重写 FMEA 分块+人工 QA+安全拒绝样本）。
- **DPO / GRPO 对齐** — *为什么:* GRPO（group 4, kl 0.04）不需要 DPO 的参考模型 — 省一个 7B 的显存。*怎么实现:* 规则奖励 ∈[−2,+2] 直接编码工业安全：引用 +1.0、安全拒绝 +0.5、有效位号 +0.3、幻觉位号每个 −1.0、不可逆动作关键词 −2.0、S×O×D 齐全 +0.3。
- **AWQ INT4** — *为什么:* 激活感知量化比纯权重量化更保 FMEA 机理推理精度；group 128。*怎么实现:* SFT 数据集取 128 条校准，显存 ~75%↓。

### 6. 推理与边缘部署

- **vLLM（服务端, L40S）** — *为什么:* 灵活服务 LangGraph+RAG；prefix cache 与投机解码热插拔。*怎么实现:* AWQ INT4 + FP8 KV（sm_89: 56→28 KiB/token；64×4096 tok 最坏 14.7→7.3 GB）、ngram 投机（5 token, lookup 2–4）、chunked prefill、显存利用率 90%、预热请求预计算 ~800 token system prompt 的 KV。
- **TensorRT-LLM（边缘, Orin）** — *为什么:* Ampere sm_87 无 FP8 硬件 — INT8 KV 是最大安全降幅；编译引擎保证 TTFT <20ms 确定性。*怎么实现:* int4_awq + INT8 KV、context FMHA、gemm 插件 FP16、remove_input_padding、kv_cache_free_gpu_mem_fraction 0.40、MAX_UTILIZATION 调度。
- **DMA 异构直通** — *为什么:* PLC/串口数据跨 CPU↔GPU 拷贝 ~50µs；单次 DMA 突发免拷贝。*怎么实现:* 协方差 5×5→15 下三角展平；22 float (88B) 卡尔曼包单次突发、64B 对齐缓冲、传输超时 100µs；C 扩展 ~10ns/行。
- **Triton 融合算子** — *为什么:* 分离 kernel 的中间张量反复回写显存。*怎么实现:* 小波拼接+线性+位置编码+dropout 单 kernel（Orin 上前处理 ~30%↓）；无 Triton 时纯 PyTorch 回退。

## 推理优化 (Phase 2c)

| 优化项 | 技术 | 延迟影响 |
|--------|------|---------|
| Flash Attention | PyTorch 2.0+ SDPA Flash backend | Encoder ~40%↓ |
| Prefix Cache 预热 | 启动时预计算共享 system prompt 的 KV | Prompt 编码→0 |
| ngram 投机采样 | vLLM n-gram 匹配, 无需草稿模型 | TTFT ~50%↓ |
| Triton 融合算子 | 小波拼接+线性投影+位置编码 单 kernel | 前处理 ~30%↓ |
| xgrammar 结构化输出 | JSON Schema→FSM, O(1) token 掩码 | Token 生成 ~10-20%↑ |
| KV Cache 量化 | FP8(L40S) / INT8(Orin) KV cache | KV 显存 50%↓ |

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
