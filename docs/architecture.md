# Architecture

## System Overview

The Industrial FMEA Agent is a three-phase evolutionary system for intelligent
predictive maintenance of multi-stage cryogenic distillation equipment.

### Phase 1 (2025.04–08): Signal Processing Foundation

Kalman-Wavelet cascade → DTW alignment → virtual soft sensor → physics-informed
anomaly detection with EWMA+KDE adaptive baseline.

### Phase 2 (2025.09–12): RAG + Agent Orchestration

MarkItDown ingestion → semantic FMEA rewriting → BM25+BGE hybrid retrieval →
cross-encoder reranking → four-layer anti-hallucination safety stack →
LangGraph StateGraph agent with human-in-the-loop reflection.

### Phase 3 (2026.01–05): Transformer + RL + Edge Deployment

Kalman-Wavelet-Transformer cascade with Kalman feedback loop → Model-based RL
(PPO + MCTS) → counterfactual advisory → AWQ INT4 quantization → Jetson AGX
Orin edge deployment with DMA.

## Data Flow

```
PLC (S7/OPC UA) ──→ Kalman Filter ──→ Wavelet Decomposition ──→ KWT Encoder
                                                                      │
Serial (RS485) ────→ Protocol Adapter ──→ Feature Engineering ────────┤
                                                                      │
Excel (Async) ─────→ MarkItDown ──→ Semantic Rewrite ──→ FAISS ──────┤
                                                                      │
                                                                    Agent
                                                                   (LangGraph)
                                                                      │
                                                              ┌───────┴────────┐
                                                              │                  │
                                                        Report Generator    Fallback
                                                              │                  │
                                                              └────────┬─────────┘
                                                                       │
                                                                  MES / SCADA
```

## Safety Architecture

Four-layer anti-hallucination defense:

1. **Constrained Decoding** — JSON Schema + asset dictionary token rejection
2. **Pydantic Validation** — runtime type checking, S×O×D=RPN consistency
3. **Guardrails Gateway** — physical boundary checks (abundance ≤ 100%, temp > −273°C)
4. **Citation Tracking** — every claim must cite a source; uncited = rejected

## Deployment Topology

```
┌──────────────────────────────────────────────────────┐
│  Edge (Jetson AGX Orin)                              │
│  • Kalman filter (streaming, < 100 µs)               │
│  • DMA buffer (PLC + Serial → NPU)                   │
│  • TensorRT-LLM (AWQ INT4, TTFT < 20 ms)             │
│  • Local RAG (FAISS in-memory)                       │
└──────────────────────┬───────────────────────────────┘
                       │ MQTT / OPC UA
┌──────────────────────┴───────────────────────────────┐
│  Server (L40S)                                       │
│  • LangGraph agent orchestration                     │
│  • RAG hybrid search + reranking                     │
│  • RL training sandbox (digital twin)                │
│  • vLLM inference server                             │
└──────────────────────────────────────────────────────┘
```
