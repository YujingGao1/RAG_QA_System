# RAG 多路召回 + CrossEncoder 重排序智能问答系统

基于 LangChain + Chroma + BM25 + bge-reranker 的混合检索 RAG 系统，含 LLM 查询改写与自动化量化评测框架。

## 项目特点

- **四路检索策略可切换**：纯向量 / 纯 BM25 / 向量+BM25 / 混合+CrossEncoder 重排序
- **中文 CrossEncoder 重排序**：bge-reranker-base 对 (query, doc) 联合编码精细排序
- **LLM 查询改写**：通义千问自动生成 3 种同义变体，弥补口语化输入与文档措辞差异
- **双层自动化评测**：gold_facts 检索评测 + LLM-as-Judge 端到端答案评测

## 架构

```
用户问题
  │
  ├─ [查询改写]  qwen-turbo → 生成 3 个同义变体
  │
  ├─ [多路召回]  每个变体:
  │     ├─ 向量检索 (Chroma + paraphrase-multilingual)
  │     └─ BM25 关键词检索 (jieba)
  │
  ├─ [重排序]    bge-reranker-base CrossEncoder 打分 → Top5
  │
  └─ [生成答案]  qwen-turbo → 最终回答
```

## 实验结果

| 指标 | 纯向量 | 混合+CrossEncoder | 提升 |
|------|--------|-------------------|------|
| Recall@5 | 65.0% | **86.0%** | +32.3% |
| Hit@5 | 76.0% | **100.0%** | +31.6% |
| MRR | 0.637 | **0.835** | +31.1% |

**困难场景**（含语义干扰文档）：Recall@5 从 50.0% → 92.9%（接近翻倍）

**端到端答案质量**：LLM-as-Judge 平均 **4.7/5.0**，25 题中 21 题满分（84%）

完整实验数据见 [benchmark_full_report.json](benchmark_full_report.json)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置 API Key

```bash
# Linux / macOS
export DASHSCOPE_API_KEY="your-dashscope-api-key"

# Windows PowerShell
$env:DASHSCOPE_API_KEY="your-dashscope-api-key"
```

首次运行会自动通过 ModelScope 下载 bge-reranker-base 模型（约 1.1GB），请确保网络畅通。

### 3. 交互问答

```bash
python main.py
```

放入你的 TXT/PDF 文档到 `data/` 目录，启动后输入问题即可获得回答。

```bash
mkdir data
cp /path/to/your/企业文档.pdf data/
python main.py
```

### 4. 量化评测（可选）

```bash
python benchmark_final.py
```

运行 15 篇文档 + 25 道分层测试题的完整评测，输出检索指标对比和端到端答案评分。

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | **交互问答入口**：加载知识库 → 命令行交互问答 |
| `rag_core.py` | RAG 核心模块：检索策略、重排序、查询改写、答案生成 |
| `benchmark_final.py` | 量化对比实验：5种策略对比、端到端评测 |
| `benchmark_full_report.json` | 实验结果数据 |

## 技术栈

- **检索**：LangChain · Chroma · BM25 (rank-bm25) · jieba
- **模型**：paraphrase-multilingual-MiniLM-L12-v2 · bge-reranker-base
- **LLM**：通义千问 (qwen-turbo) via DashScope
