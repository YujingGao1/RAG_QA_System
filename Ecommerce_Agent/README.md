# 智能电商客服 Agent 系统

基于 LangChain ReAct Agent + ChromaDB + MySQL 的智能客服系统，具备双记忆机制、情绪感知和多轮对话能力。

## 项目特点

- **ReAct Agent 自主决策**：LLM 根据用户意图自主选择工具（订单查询 / FAQ 检索 / 情绪检测 / 记忆召回），非硬编码流程
- **双层记忆系统**：短期记忆（SummaryBuffer 滑动窗口 + LLM 压缩）+ 长期记忆（ChromaDB 向量库，含重要度评分与 30 天过期）
- **多用户隔离**：ChromaDB filter 按用户维度隔离长期记忆，支持会话重启后记忆恢复
- **情绪感知**：关键词匹配 + LLM 工具调用双重检测，负面情绪自动安抚

## 架构

```
用户输入
  │
  ├─ [SummaryBuffer] 注入短期记忆（最近 2 轮原文 + 历史摘要）
  │
  ├─ [ReAct Agent] qwen-plus 驱动
  │     ├─ Thought: 分析用户意图
  │     ├─ Action: 调用工具
  │     │     ├─ query_order      → MySQL 订单状态
  │     │     ├─ search_faq        → FAQ 政策查询
  │     │     ├─ detect_emotion    → 用户情绪检测
  │     │     └─ recall_memory     → ChromaDB 长期记忆
  │     └─ Observation: 工具返回结果 → 生成回复
  │
  ├─ [记忆更新]
  │     ├─ SummaryBuffer.add()  → 溢出自动 LLM 压缩
  │     └─ save_to_long_term()  → 重要度评分 → 去重 → 入库
  │
  └─ 客服回复
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r ../requirements.txt
pip install pymysql
```

### 2. 创建 MySQL 数据库

```sql
-- 执行 sql/init.sql
mysql -u root -p < sql/init.sql
```

或手动创建：

```sql
CREATE DATABASE ecommerce;
USE ecommerce;

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer VARCHAR(50),
    status VARCHAR(20),
    items VARCHAR(200),
    total DECIMAL(10,2),
    created_at DATETIME
);

INSERT INTO orders VALUES
(10001, '张三', '已发货', 'iPhone 15 Pro', 8999.00, '2026-07-01'),
(10002, '张三', '待付款', 'MacBook Air', 9499.00, '2026-07-03'),
(10003, '李四', '已完成', 'AirPods Pro', 1899.00, '2026-06-28');
```

### 3. 设置 API Key

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

### 4. 运行

```bash
python ecommerce_agent.py
```

```
用户名: 张三

小智：您好亲！我是小智，有什么可以帮您的？(q 退出)
==================================================

【用户】我想查一下订单10001的状态
【小智】亲，您的订单10001已发货，商品是iPhone 15 Pro...

【用户】对了，退货政策是什么？
【小智】亲，我们支持7天无理由退货...

【用户】我之前问过订单和退货的事情，你还记得吗？
[长期记忆召回] → 小智：亲，根据记录您之前查询过订单10001...
```

## 记忆机制详解

### 短期记忆（SummaryBuffer）

| 参数 | 值 | 说明 |
|------|-----|------|
| max_turns | 2 | 保留最近 2 轮原文 |
| 压缩模型 | qwen-turbo | 溢出对话 LLM 压缩为摘要 |
| 压缩策略 | 一次性全量总结 | 不增量合并，避免信息丢失 |

### 长期记忆（ChromaDB）

| 机制 | 实现 |
|------|------|
| 重要度评分 | qwen-turbo 对每轮对话打分（1-10），< 3 分跳过 |
| 去重 | 向量相似度检索，内容重复则跳过 |
| 过期 | 30 天自动过期，过期后不再召回 |
| 隔离 | 按 `user` 字段 filter，多用户数据不交叉 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `ecommerce_agent.py` | 客服 Agent 主程序：工具定义、记忆模块、交互入口 |
| `sql/init.sql` | MySQL 建库建表 + 示例数据 |

## 技术栈

- **Agent**：LangChain ReAct Agent (create_agent) · 通义千问 (qwen-plus)
- **记忆**：SummaryBuffer（短期） · ChromaDB + HuggingFace Embeddings（长期）
- **数据库**：MySQL (pymysql)
- **模型**：qwen-plus（主控） · qwen-turbo（压缩/评分） · paraphrase-multilingual-MiniLM-L12-v2（向量化）
