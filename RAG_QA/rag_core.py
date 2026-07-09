"""
RAG 核心模块：向量检索 + BM25关键词检索 + CrossEncoder重排序 + LLM问答
用于量化对比实验
"""
import os
import re
import warnings
from typing import List, Dict, Optional

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dashscope import Generation

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# 中文 CrossEncoder 模型路径（本地优先，不存在则自动下载）
def _get_reranker_path():
    local = os.path.join(os.path.dirname(__file__), "models", "BAAI", "bge-reranker-base")
    if os.path.exists(local):
        return local
    print("📥 首次运行，正在通过 ModelScope 下载 bge-reranker-base ...")
    try:
        from modelscope import snapshot_download
        return snapshot_download("BAAI/bge-reranker-base", cache_dir=os.path.join(os.path.dirname(__file__), "models"))
    except ImportError:
        raise RuntimeError(
            "请安装 modelscope 后重试: pip install modelscope\n"
            "或手动下载模型到 models/BAAI/bge-reranker-base/"
        )

RERANKER_MODEL_PATH = _get_reranker_path()


class RAGRetrievalBenchmark:
    """
    RAG 检索器 —— 支持四种检索策略 + LLM查询改写 + LLM答案生成
    检索模式：
      - 'vector'         : 纯向量语义检索（基线）
      - 'bm25'           : 纯 BM25 关键词检索（基线）
      - 'hybrid'         : 向量 + BM25 混合检索（去重，无重排序）
      - 'hybrid_rerank'  : 向量 + BM25 + CrossEncoder 重排序（完整版）
    """

    def __init__(
        self,
        dashscope_api_key: str,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        llm_model: str = "qwen-turbo",
        chunk_size: int = 150,
        chunk_overlap: int = 20,
    ):
        print("=" * 60)
        print("🚀 初始化 RAG 系统（含 LLM + CrossEncoder 重排序）")
        print("=" * 60)

        # ---------- LLM 配置 ----------
        self.llm_model = llm_model
        self.api_key = dashscope_api_key
        print(f"🤖 LLM 模型: {llm_model}")

        # ---------- 文本分割 ----------
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

        # ---------- 向量化模型 ----------
        print(f"📦 加载 Embedding 模型: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        self._st_model = self.embeddings._client
        print(f"   ✅ 模型维度: {self._st_model.get_sentence_embedding_dimension()}")

        # ---------- CrossEncoder 重排序模型 ----------
        print(f"📦 加载 CrossEncoder 重排序模型: bge-reranker-base")
        self.reranker = CrossEncoder(RERANKER_MODEL_PATH, device="cpu")
        print(f"   ✅ CrossEncoder 就绪")

        # ---------- 提示词模板 ----------
        self.qa_prompt = PromptTemplate(
            template="""你是一个专业的AI助手。请基于以下检索到的上下文信息来回答问题。
【重要规则】
1. 只使用提供的上下文信息回答
2. 如果上下文中没有相关信息，明确说"根据提供的信息无法回答"
3. 回答要准确、简洁，不超过3句话

上下文信息：
{context}

问题: {question}

答案:""",
            input_variables=["context", "question"]
        )

        # ---------- 内部状态 ----------
        self.vectorstore: Optional[Chroma] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.all_chunks: List[Document] = []
        self.chunk_texts: List[str] = []

        print("✅ 初始化完成\n")

    # ================================================================
    #  LLM 调用
    # ================================================================
    def _call_llm(self, prompt: str, temperature: float = 0.1) -> str:
        """调用通义千问"""
        response = Generation.call(
            api_key=self.api_key,
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
            temperature=temperature,
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            return f"[LLM调用失败: {response.code} - {response.message}]"

    # ================================================================
    #  查询改写（Query Rewriting）
    # ================================================================
    def query_rewrite(self, original_query: str) -> List[str]:
        """通过 LLM 生成3个语义等价但表达不同的查询变体"""
        prompt = f"""原始查询: {original_query}

请生成3个语义相同但表达不同的查询变体，用于提高检索召回率。
仅输出3行查询文本，不要额外解释、序号、标点。"""

        response = self._call_llm(prompt, temperature=0.3)
        variants = []
        for line in response.split('\n'):
            line = line.strip()
            if line and len(line) > 2:
                variants.append(line)
        variants = list(dict.fromkeys(variants))[:3]
        return [original_query] + variants

    # ================================================================
    #  生成答案
    # ================================================================
    def generate_answer(self, question: str, retrieved_docs: List[Document]) -> str:
        """基于检索到的文档，调用 LLM 生成答案"""
        if not retrieved_docs:
            return "根据提供的信息无法回答"
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = self.qa_prompt.format(context=context, question=question)
        return self._call_llm(prompt, temperature=0.1)

    # ================================================================
    #  CrossEncoder 重排序
    # ================================================================
    def _cross_encoder_rerank(
        self, query: str, candidates: List[Document], top_k: int
    ) -> List[Document]:
        """使用 bge-reranker-base 做 CrossEncoder 精细重排序"""
        if not candidates:
            return []
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked_idx = np.argsort(scores)[::-1]
        return [candidates[i] for i in ranked_idx[:top_k]]

    # ================================================================
    #  多查询融合检索（查询改写后的增强检索）
    # ================================================================
    def retrieve_with_rewrite(
        self, query: str, strategy: str = "hybrid_rerank", k: int = 5
    ) -> Dict:
        """查询改写 + 多路检索 + 融合 + CrossEncoder 重排序"""
        queries = self.query_rewrite(query)

        all_docs = []
        per_query = {}
        for q in queries:
            docs = self.retrieve(q, strategy=strategy, k=k)
            per_query[q] = docs
            all_docs.extend(docs)

        # 全局去重
        seen, merged = set(), []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)

        # CrossEncoder 重排序——用原始问题做重排（改写变体只负责扩大召回，不替代用户意图）
        if len(merged) > k:
            merged = self._cross_encoder_rerank(query, merged, k)

        return {
            "docs": merged[:k],
            "rewritten_queries": queries,
            "per_query_docs": per_query,
        }

    # ================================================================
    #  知识库构建
    # ================================================================
    def build_knowledge_base(self, documents: List[Document]):
        """切片 + 向量库 + BM25 索引"""
        print("🔨 构建知识库...")

        self.all_chunks = self.text_splitter.split_documents(documents)
        self.chunk_texts = [doc.page_content for doc in self.all_chunks]
        print(f"   切分为 {len(self.all_chunks)} 个文本块")

        self.vectorstore = Chroma.from_texts(
            texts=self.chunk_texts,
            embedding=self.embeddings,
            persist_directory="./chroma_benchmark_db"
        )

        tokenized_corpus = [list(jieba.cut(t)) for t in self.chunk_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print(f"   ✅ 向量库 + BM25 索引构建完成\n")

    # ================================================================
    #  四种检索策略
    # ================================================================
    def retrieve_vector(self, query: str, k: int = 5) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def retrieve_bm25(self, query: str, k: int = 5) -> List[Document]:
        tokenized_q = list(jieba.cut(query))
        scores = self.bm25_index.get_scores(tokenized_q)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.all_chunks[i] for i in top_idx]

    def retrieve_hybrid(self, query: str, k: int = 5) -> List[Document]:
        vec_docs = self.vectorstore.similarity_search(query, k=k)
        tokenized_q = list(jieba.cut(query))
        scores = self.bm25_index.get_scores(tokenized_q)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        bm25_docs = [self.all_chunks[i] for i in top_idx]
        seen, merged = set(), []
        for doc in vec_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
        return merged[:k]

    def retrieve_hybrid_rerank(self, query: str, k: int = 5) -> List[Document]:
        """宽召回(向量+BM25各15条) → 去重 → CrossEncoder 重排序 → Top5"""
        recall_n = max(k * 3, 12)
        vec_docs = self.vectorstore.similarity_search(query, k=recall_n)
        tokenized_q = list(jieba.cut(query))
        scores = self.bm25_index.get_scores(tokenized_q)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:recall_n]
        bm25_docs = [self.all_chunks[i] for i in top_idx]
        seen, merged = set(), []
        for doc in vec_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
        if not merged:
            return []
        return self._cross_encoder_rerank(query, merged, k)

    def retrieve(self, query: str, strategy: str = "hybrid_rerank", k: int = 5) -> List[Document]:
        if strategy == "vector":
            return self.retrieve_vector(query, k=k)
        elif strategy == "bm25":
            return self.retrieve_bm25(query, k=k)
        elif strategy == "hybrid":
            return self.retrieve_hybrid(query, k=k)
        elif strategy == "hybrid_rerank":
            return self.retrieve_hybrid_rerank(query, k=k)
        else:
            raise ValueError(f"未知策略: {strategy}")
