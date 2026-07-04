"""
RAG 问答系统交互入口
支持 TXT/PDF 文档知识库，命令行交互问答
"""
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from rag_core import RAGRetrievalBenchmark
from langchain_core.documents import Document

# ================================================================
# 加载文档知识库
# ================================================================
def load_documents(data_dir: str = "./data"):
    """加载 data 目录下的所有 TXT 和 PDF 文件"""
    from pypdf import PdfReader

    docs = []
    if not os.path.exists(data_dir):
        print(f"⚠️ data 目录不存在，使用内置测试文档")
        return _default_docs()

    files = [f for f in os.listdir(data_dir) if f.endswith(('.txt', '.pdf'))]
    if not files:
        print(f"⚠️ data 目录为空，使用内置测试文档")
        return _default_docs()

    for fname in files:
        fpath = os.path.join(data_dir, fname)
        print(f"  📄 加载: {fname}")
        if fname.endswith('.pdf'):
            reader = PdfReader(fpath)
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text))
        else:
            with open(fpath, 'r', encoding='utf-8') as f:
                docs.append(Document(page_content=f.read()))

    return docs


def _default_docs():
    """内置示例文档"""
    return [
        Document(page_content="""
2023年度公司财务报告

2023年全年营业收入2.5亿元，同比增长35%。第四季度营收5000万元。
AI助手产品线销售额1.2亿元，占比48%，是最大收入来源。
2024年目标营收突破3亿元。
        """),
        Document(page_content="""
智能AI助手产品手册 v2.0

核心功能：自然语言对话、知识库问答、任务自动化、数据分析、多语言翻译。
技术指标：响应延迟<500ms，并发吞吐量1000QPS。
        """),
        Document(page_content="""
公司考勤管理制度

年假：入职满1年5天，上限15天。病假全年最多30天。事假全年最多10天。
加班：工作日1.5倍、休息日2倍、节假日3倍，每月上限36小时。
连续旷工3天按自动离职处理。
        """),
    ]


# ================================================================
# 主程序
# ================================================================
def main():
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("❌ 请设置 DASHSCOPE_API_KEY 环境变量")
        print("   export DASHSCOPE_API_KEY='your-key'")
        return

    print("\n" + "=" * 55)
    print("  RAG 智能问答系统")
    print("  向量检索 + BM25 + CrossEncoder 重排序 + 查询改写")
    print("=" * 55)

    # 初始化
    rag = RAGRetrievalBenchmark(dashscope_api_key=api_key)

    # 加载文档
    print("\n📂 加载知识库...")
    documents = load_documents()
    rag.build_knowledge_base(documents)
    print(f"   共加载 {len(documents)} 篇文档，切分为 {len(rag.all_chunks)} 个文本块\n")

    # 交互问答
    print("💬 开始问答（输入 quit 退出）\n")
    while True:
        try:
            question = input("🧑 你问: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not question:
            continue
        if question.lower() in ('quit', 'exit', 'q', '退出'):
            print("👋 再见！")
            break

        # 查询改写 + 多路检索 + 重排序 + 生成答案
        result = rag.retrieve_with_rewrite(question, strategy="hybrid_rerank", k=5)
        answer = rag.generate_answer(question, result["docs"])

        print(f"🤖 回答: {answer}")
        print(f"    (检索改写: {' | '.join(result['rewritten_queries'][1:4])})")
        print()


if __name__ == "__main__":
    main()
