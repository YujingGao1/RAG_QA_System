# RAG_QA_System

基于大语言模型的智能问答与 Agent 系统，包含两个子项目：

| 项目 | 目录 | 关键词 |
|------|------|--------|
| RAG 混合检索问答系统 | [RAG_QA/](RAG_QA/) | 多路召回 · CrossEncoder 重排序 · LLM 查询改写 · 自动化评测 |
| 智能电商客服 Agent | [Ecommerce_Agent/](Ecommerce_Agent/) | ReAct Agent · ChromaDB 长期记忆 · MySQL 订单管理 · 情绪检测 |

## 快速导航

- [RAG 项目 README](RAG_QA/README.md) — 架构、实验结果、快速开始
- [Agent 项目 README](Ecommerce_Agent/README.md) — 多智能体协作、双记忆机制、数据库建表

## 环境要求

```bash
pip install -r requirements.txt
```

- Python 3.10+
- 通义千问 API Key（[DashScope 控制台](https://dashscope.console.aliyun.com/)）
- Agent 项目需要本地 MySQL 服务（5.7+）

## 技术栈

**RAG**：LangChain · Chroma · BM25 · bge-reranker · jieba  
**Agent**：LangChain ReAct Agent · ChromaDB · MySQL · SummaryBuffer  
**LLM**：通义千问 qwen-plus / qwen-turbo via DashScope
