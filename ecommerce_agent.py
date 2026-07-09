"""
电商客服 Agent —— 千问版

功能：FAQ问答 / 订单查询 / 情绪检测 / 多轮对话记忆
"""
import os
import pymysql

# MySQL 配置（改成你的密码）
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "ecommerce",
    "charset": "utf8mb4",
}
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

os.environ["DASHSCOPE_API_KEY"] = "sk-ws-H.RPMRDLM.BKfB.MEYCIQD0Hl6-yQhWU03iY5MeA5GmJ6NfiqDDm21LlrBREpG8kgIhAJ0x-gYZ6R2TaXk5Mm4IT7_Dyp6Ybnl02vhaLvOALLmy"


# ================================================================
# 1. 数据库连接（表结构见文件底部 SQL，在 PyCharm Console 执行）
# ================================================================
def get_conn():
    return pymysql.connect(**MYSQL_CONFIG)

current_user = None  # 当前登录用户（模拟登录态）


# ================================================================
# 2. 长期记忆层 — ChromaDB
# ================================================================
from datetime import datetime, timedelta
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore")

embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
)
long_memory = Chroma(
    collection_name="cs_agent_memory",
    embedding_function=embeddings,
    persist_directory="./chroma_agent_db",
)


# ================================================================
# 3. 工具定义
# ================================================================
@tool
def query_order(order_id: str) -> str:
    """查询订单状态。输入订单号(如10001)。只能查到当前登录用户的订单。"""
    conn = get_conn()
    with conn.cursor() as cursor:
        if current_user:
            cursor.execute(
                "SELECT status, items, total, created_at FROM orders WHERE order_id = %s AND customer = %s",
                (order_id, current_user)
            )
        else:
            cursor.execute(
                "SELECT status, items, total, created_at FROM orders WHERE order_id = %s", (order_id,)
            )
        row = cursor.fetchone()
    conn.close()
    if row:
        status, items, total, created = row
        return f"订单{order_id} | 商品:{items} | 金额:{total}元 | 状态:{status} | 下单:{created}"
    return f"未找到您的订单{order_id}，请确认订单号是否正确"


@tool
def search_faq(question: str) -> str:
    """搜索常见问题(FAQ)。输入问题关键词，如'退货''发货''支付'等。"""
    faq = {
        "退货": "7天无理由退货，商品需保持完好。退货运费由平台承担。退款将在收到退货后3个工作日内原路返回。",
        "发货": "工作日16:00前下单当天发货，节假日顺延。默认发顺丰，预计1-3天送达。",
        "支付": "支持微信支付、支付宝、信用卡、花呗分期。大额订单可联系客服走对公转账。",
        "换货": "收到商品7天内可申请换货。如因质量问题换货，来回运费由平台承担。",
        "物流": "可在订单详情页查看实时物流。如超过预计时间未收到，请联系客服查询。",
        "发票": "下单时可选择开具电子发票。电子发票将在确认收货后自动发送至注册邮箱。",
    }
    for key, answer in faq.items():
        if key in question:
            return f"【{key}】{answer}"
    return f"未找到关于'{question}'的FAQ，建议转人工客服处理"


@tool
def detect_emotion(text: str) -> str:
    """检测用户情绪。输入用户消息文本，返回'negative'(负面)或'neutral'(中性)。"""
    negative_words = ["生气", "不满意", "糟糕", "差", "垃圾", "投诉", "退款", "坑", "骗", "慢", "烂"]
    count = sum(1 for w in negative_words if w in text)
    if count >= 2:
        return "strong_negative"  # 强烈负面
    elif count == 1:
        return "negative"          # 轻微负面
    return "neutral"               # 中性


@tool
def recall_memory(query: str) -> str:
    """检索该用户的历史对话记忆。查询用户的偏好、历史问题、重要信息时使用。"""
    docs = long_memory.similarity_search(
        query, k=5,
        filter={"user": current_user or "default"},
    )
    results = []
    now = datetime.now()
    for doc in docs:
        meta = doc.metadata
        expiry = datetime.fromisoformat(meta.get("expiry", "2099-01-01T00:00:00"))
        if expiry > now:
            results.append(doc.page_content)
    return "\n".join(f"· {r}" for r in results) if results else "该用户没有历史记录"


def save_to_long_term(user_msg: str, agent_msg: str):
    """重要对话存入 ChromaDB（带重要度判断 + 去重）"""
    # 重要性判断
    prompt = f"""判断以下客服对话是否包含值得长期记忆的信息（用户偏好、投诉、重要需求、个人信息等）。
仅输出 1-10 的数字：
用户: {user_msg}
客服: {agent_msg}"""
    resp = Generation.call(
        api_key=os.environ["DASHSCOPE_API_KEY"], model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        result_format="message", temperature=0.0, seed=42,
    )
    raw = resp.output.choices[0].message.get("content", "0").strip() if resp.status_code == 200 else "0"
    try:
        score = int(''.join(c for c in raw if c.isdigit()))
    except ValueError:
        score = 0

    if score < 3:
        return  # 不重要，跳过

    # 去重
    existing = long_memory.similarity_search(
        user_msg, k=1, filter={"user": current_user or "default"}
    )
    if existing and len(user_msg) > 3 and user_msg in existing[0].page_content:
        return

    text = f"用户: {user_msg}\n客服回复: {agent_msg}"
    meta = {
        "user": current_user or "default",
        "timestamp": datetime.now().isoformat(),
        "expiry": (datetime.now() + timedelta(days=30)).isoformat(),
    }
    long_memory.add_texts([text], metadatas=[meta])
    print(f"  [长期] 已存入（重要度{score}，30天过期）")


# ================================================================
# 3. 客服 Agent
# ================================================================
llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.3,
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SYSTEM_PROMPT = """你是一个友好、专业的电商客服AI助手，名字叫"小智"。

严格规则（必须遵守）：
1. 涉及订单信息 → 先看消息中可能包含「本轮对话摘要」和最近的对话原文，这是短期记忆，若没有相关信息，则用 query_order 查询，拿到真实数据后再回答
2. 涉及退货/发货/支付等政策 → 必须先用 search_faq 查询
3. 用户情绪激动 → 先用 detect_emotion 检测
4. 用户提到之前聊过的内容 → 用 recall_memory 检索该用户的历史记录
5. 绝对禁止编造订单状态、金额或政策内容
6. 工具查不到 → 如实告知并建议转人工

风格：
- 称呼用户为"亲"，保持礼貌耐心
- 情绪不好的用户，先安抚再处理
- 回答简洁，2-3句话"""


# ================================================================
# 4. 滑动窗口记忆
# ================================================================
class SummaryBuffer:
    """保留最近 2 轮原文，超出的 LLM 压缩为摘要"""
    def __init__(self, max_turns: int = 2):
        self.max_turns = max_turns
        self.turns = []
        self.summary = ""

    def add(self, user_msg: str, agent_msg: str):
        self.turns.append({"user": user_msg, "agent": agent_msg})
        if len(self.turns) > self.max_turns:
            self._compress()

    def _compress(self):
        overflow = self.turns[:-self.max_turns]
        self.turns = self.turns[-self.max_turns:]
        # 把所有溢出对话拼一起，LLM 一次性总结（不依赖增量合并）
        all_overflow = self.summary + "\n" if self.summary else ""
        all_overflow += "\n".join(f"用户: {t['user']}\n客服: {t['agent']}" for t in overflow)
        prompt = f"""将以下客服对话总结为一句摘要（不超过100字），覆盖所有关键信息：

{all_overflow[:1500]}

摘要:"""
        resp = Generation.call(
            api_key=os.environ["DASHSCOPE_API_KEY"], model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            result_format="message", temperature=0.0, seed=42,
        )
        if resp.status_code == 200:
            self.summary = resp.output.choices[0].message.get("content", "").strip()

    def build_messages(self, user_input: str) -> list:
        msgs = [SystemMessage(content=SYSTEM_PROMPT)]
        if self.summary:
            msgs.append(SystemMessage(content=f"[历史摘要] {self.summary}"))
        for t in self.turns:
            msgs.append(HumanMessage(content=t["user"]))
            msgs.append(AIMessage(content=t["agent"]))
        msgs.append(HumanMessage(content=user_input))
        return msgs


# ================================================================
# 5. 对话
# ================================================================
from dashscope import Generation

buffer = SummaryBuffer(max_turns=2)


def chat(user_input: str) -> str:
    """一轮对话"""
    messages = buffer.build_messages(user_input)

    agent = create_agent(
        model=llm,
        tools=[query_order, search_faq, detect_emotion, recall_memory],
        system_prompt=SYSTEM_PROMPT,
    )

    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 100},
    )

    # # 打印完整过程
    # for msg in result["messages"]:
    #     role = msg.type if hasattr(msg, 'type') else msg.get("role", "?")
    #     if role == "human":
    #         print(f"    👤 用户: {msg.content[:80]}...")
    #     elif role == "ai":
    #         if hasattr(msg, "tool_calls") and msg.tool_calls:
    #             for tc in msg.tool_calls:
    #                 print(f"    🔧 调工具: {tc['name']}({tc['args']})")
    #         elif msg.content:
    #             print(f"    🤖 回答: {msg.content[:200]}...")
    #     elif role == "tool":
    #         print(f"    📋 工具返回: {str(msg.content)[:200]}...")
    #     print()

    reply = result["messages"][-1].content
    buffer.add(user_input, reply)
    save_to_long_term(user_input, reply)
    return reply


# ================================================================
# 4. 演示
# ================================================================
if __name__ == "__main__":
    current_user = input("用户名: ").strip() or "张三"
    print(f"\n🔐 {current_user}，您好亲！我是小智，有什么可以帮您的？(q 退出)")
    print("=" * 50)

    while True:
        user_input = input("\n【用户】").strip()
        if user_input.lower() == 'q':
            print("小智：感谢您的咨询，再见亲～")
            break
        if not user_input:
            continue
        reply = chat(user_input)
        buffer.summary and print(f"  [记忆摘要] {buffer.summary}")
        print(f"【小智】{reply}")


# if __name__ == "__main__":
#     # ── 模拟登录 ──
#     # 修改模块级变量
#     current_user = "张三"
#     print("=" * 50)
#     print(f"🔐 已登录: {current_user}")
#     print("小智：您好亲！我是AI客服小助手，有什么可以帮您的？")
#     print("=" * 50)
#
#     # 第1轮：查自己的订单
#     print("\n【用户】我想查一下订单10001的状态")
#     reply = chat("我想查一下订单10001的状态")
#     print(f"【小智】{reply}")
#
#     # 第2轮：继续查
#     print("\n【用户】我还有个订单10002，也帮我看看")
#     reply = chat("我还有个订单10002，也帮我看看")
#     print(f"【小智】{reply}")
#
#     # 第3轮：FAQ（此时前两轮溢出，触发摘要）
#     print("\n【用户】对了，退货政策是什么？")
#     reply = chat("对了，退货政策是什么？")
#     print(f"【小智】{reply}")
#     buffer.summary and print(f"  [记忆摘要] {buffer.summary}")
#
#     # 第4轮：凭短期记忆回答
#     print("\n【用户】刚才查的那个10001，状态是什么？")
#     reply = chat("刚才查的那个10001，状态是什么？")
#     print(f"【小智】{reply}")
#     buffer.summary and print(f"  [短期摘要] {buffer.summary}")
#
#     # ── 模拟重启：短期清空，靠长期 ──
#     print("\n\n" + "=" * 50)
#     print("🔄 模拟重启（短期记忆清空）")
#     print("=" * 50)
#     buffer = SummaryBuffer(max_turns=2)  # 新 buffer，短期全空
#
#     print("\n【用户】我之前问过订单和退货的事情，你还记得吗？")
#     reply = chat("我之前问过订单和退货的事情，你还记得吗？")
#     print(f"【小智】{reply}")
