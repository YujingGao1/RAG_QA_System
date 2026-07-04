"""
RAG 完整量化对比实验（含 LLM 查询改写 + 端到端答案评测）

六路对比：
  检索侧（无改写）: vector | bm25 | hybrid | hybrid_rerank
  检索侧（有改写）: hybrid_rerank + query_rewrite
  端到端: 完整 RAG 问答准确率评测

评测指标:
  - 检索质量: Precision@5, Recall@5, MRR, NDCG@5, Hit@K
  - 答案质量: LLM-as-Judge 评分 (1-5分)
  - 改写增益: 查询改写带来的检索指标提升
"""
import os
import re
import time
import statistics
import math
import json
from typing import List, Dict
from dataclasses import dataclass

os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "your-dashscope-api-key")

from langchain_core.documents import Document
from rag_core import RAGRetrievalBenchmark


# ================================================================
# 测试文档
# ================================================================
def create_test_documents() -> List[Document]:
    docs = []
    docs.append(Document(page_content="""
2023年度公司财务报告

一、业绩概况
2023年全年，公司实现营业收入2.5亿元，同比增长35%。
第四季度单季度营收达到5000万元，创造公司成立以来的最高季度记录。
全年实现净利润3200万元，净利率达到12.8%。
研发投入4500万元，占总营收的18%。

二、产品线收入拆解
- AI助手产品线：年度销售额1.2亿元，营收占比48%，是公司最大收入来源
- 数据分析平台：年度销售额8000万元，营收占比32%
- 企业定制服务：年度销售额5000万元，营收占比20%

三、来年展望
2024年目标营收突破3亿元，毛利率保持65%以上。
计划新增海外业务线，预期贡献营收5000万元。
    """))

    docs.append(Document(page_content="""
智能AI助手产品手册 v2.0

产品名称：智能AI助手
当前版本：v2.0
发布日期：2023年6月15日

核心功能模块：
1. 自然语言对话引擎：支持长达20轮的多轮对话，具备上下文记忆与理解能力
2. 企业知识库问答：基于RAG技术，从企业文档中精准检索并生成回答
3. 智能任务自动化：可编排复杂工作流，支持定时触发和条件触发
4. 实时数据洞察：对接业务数据库，自动生成可视化分析报告
5. 多语言互译：覆盖中文、英文、日文、韩文等12种语言

典型落地场景：
- 智能客服：替代人工坐席，7x24小时在线，首响时间<10秒
- 内部知识管理：团队文档智能检索，新人问答机器人
- 销售辅助：实时调取产品资料，生成个性化推荐话术

技术指标：平均响应延迟<500ms，并发吞吐量1000QPS，准确率>90%
    """))

    docs.append(Document(page_content="""
公司考勤与休假管理制度（2023年修订版）

第一条 总则
为规范全体员工考勤行为，保障公司正常运营秩序，制订本制度。

第二条 标准工时
正常工作日为周一至周五，每日9:00至18:00，其中12:00-13:00为午休时间。
实行弹性工作制的部门，核心在岗时间为10:00-16:00。

第三条 各类假期
- 带薪年假：入职满1年享有5天，之后每满1年增加1天，上限为15天
- 病假：凭二级以上医院证明申请，全年累计不得超过30天
- 事假：需提前1个工作日提交申请，全年累计不超过10天

第四条 加班与补偿
所有加班须提前在OA系统审批。工作日加班按1.5倍时薪计算，
公休日加班按2倍计算，法定节假日按3倍计算。每人每月加班上限36小时。

第五条 违纪处理
每月迟到累计不超过3次免于处罚，超过部分每次扣除50元。
连续旷工3天或月累计旷工5天，按自动离职处理。
    """))

    docs.append(Document(page_content="""
技术平台架构设计白皮书

一、整体架构
采用云原生微服务架构，基于Spring Cloud Alibaba全家桶构建。

二、基础设施
- 容器编排与调度：Kubernetes 1.27集群，HPA自动弹性伸缩
- 服务治理：Istio服务网格，实现流量管理、熔断降级、故障注入
- 可观测性：ELK日志平台 + Prometheus指标采集 + Grafana可视化面板
- 异步消息：RocketMQ双集群主备，日处理消息量超5000万条

三、数据存储
- 事务型数据库：MySQL 8.0 InnoDB，主从半同步复制，读写分离
- 缓存层：Redis 7.0 Cluster模式，三主三从，用于热点数据缓存
- 向量存储：Chroma向量数据库，存储文档Embedding，服务RAG检索
- 全文检索：Elasticsearch 8.x集群，支持中文分词和复杂查询

四、AI服务层
- 大模型网关：统一接入通义千问、DeepSeek、GLM等多模型
- 向量化服务：text2vec-large-chinese模型，维度1024
- 重排序服务：bge-reranker-large，提升检索精度

五、SLA指标
系统整体可用性99.95%，核心接口P99延迟<200ms。
    """))

    docs.append(Document(page_content="""
客户服务管理体系

一、服务等级承诺（SLA）
- 一般咨询：2小时内首次响应
- 紧急故障：30分钟内响应并启动应急流程
- 首次解决率（FCR）：目标≥85%
- 客户满意度评分（CSAT）：目标≥4.5分（5分制满分）

二、工单处理SOP
Step1-受理登记：记录客户基本信息、问题现象、紧急程度评级
Step2-智能分类：系统自动识别问题类型（技术故障/业务咨询/投诉/建议）
Step3-技能路由：根据问题类型自动分派到对应技能组
Step4-处理追踪：处理中工单每2小时更新一次进度给客户
Step5-闭环确认：客户确认问题已解决后方可关闭工单

三、逐级升级规则
L1升级（2小时未解决）→ 升级至组长介入
L2升级（8小时未解决）→ 升级至部门经理
L3升级（24小时未解决）→ 升级至技术总监，启动战时机制

四、质量考核
客户回访覆盖100%的已关闭工单，不满意工单24小时内复盘。
月度服务之星评选，额外奖励绩效工资10%。
    """))

    docs.append(Document(page_content="""
数据安全与隐私保护管理办法

第一章 总则
为防范数据泄露、篡改和丢失风险，确保公司核心数据资产安全，制定本办法。

第二章 数据分级标准
- L3机密级：客户隐私数据（PII）、财务核心数据、源代码、加密密钥
- L2内部级：员工人事档案、内部项目文档、未公开的经营数据
- L1公开级：产品白皮书、招聘公告、官网新闻稿

第三章 访问管控
机密级数据强制双因素认证（2FA），所有操作留存审计日志≥180天。
生产数据库禁止直连，必须通过堡垒机跳板访问。
员工离职当日，所有系统账号即刻注销，权限同步回收。

第四章 传输与存储
数据传输全程TLS 1.3加密，严禁明文传输任何敏感信息。
备份策略：每日增量备份（保留7天）+ 每周全量备份（保留4周）+异地灾备。
所有备份数据保留90天后自动清除。

第五章 违规追责
一级安全事故（数据泄露）：直接责任人立即辞退，移交法务追究法律责任。
二级违规（未经审批外传内部数据）：记大过处分，取消当年绩效奖金资格。
    """))

    docs.append(Document(page_content="""
2024年人工智能行业市场分析

一、市场规模
2024年中国人工智能市场总规模预计突破6000亿元人民币，同比增长22%。
其中大语言模型（LLM）应用赛道增速最高，达85%，市场规模约800亿元。
企业级SaaS服务市场稳健增长，年复合增长率18%。

二、竞争态势
头部厂商格局：
- A公司：市场份额28%，聚焦金融和政务大型客户
- B公司：市场份额22%，主攻中小企业标准化产品
- 本公司：当前市场份额15%，以垂直行业定制化为差异化策略

三、需求侧趋势
1. 实时数据分析类需求同比增长45%
2. 多模态交互（文本+图像+语音）需求增长60%
3. 私有化部署需求增长35%，受数据合规驱动
4. 低代码/零代码AI应用构建平台需求增长50%

四、战略建议
建议重点布局：多模态AI能力 + 垂直行业解决方案。
2024年目标：市场份额从15%提升至18%，新增付费客户500家。
    """))

    docs.append(Document(page_content="""
新员工入职指引手册

欢迎加入！以下是入职首周行动清单。

Day 1 入职当天：
- 09:00 HR办理入职手续，签署劳动合同、保密协议
- 10:00 领取办公设备包（笔记本电脑、外接显示器、门禁卡、工牌）
- 11:00 直属主管介绍团队架构，带领认识组员，分配工位
- 14:00 IT部门开通各系统账号权限（企业邮箱、OA、企业微信、代码仓库）
- 16:00 参加公司文化与价值观培训

Day 2-5 第一周：
- 完成信息安全与合规在线必修课程
- 按照内部Wiki搭建本地开发环境
- 通读团队编码规范和代码审查清单
- 在TAPD项目管理系统认领第一个入门级Task

试用期与转正：
试用期3个月，期间有2次正式Review（第1月末和第3月末）。
试用期薪资按转正薪资的80%发放。
提前转正条件：两次Review绩效评分均≥4分 + 导师书面推荐。
转正后公司缴纳全额五险一金，并发放安家补贴。
    """))

    docs.append(Document(page_content="""
产品迭代更新日志

v2.1版本（2024年1月发布）
- 【新功能】多轮对话上下文记忆，支持20轮以上连续对话
- 【性能优化】向量检索引擎重构，检索延迟降低40%
- 【Bug修复】修复特殊Unicode字符引起的服务崩溃
- 【体验升级】UI全面改版，新增暗黑模式

v2.0版本（2023年6月发布）
- 【架构升级】从单体应用重构为微服务架构
- 【新功能】知识图谱问答引擎上线
- 【格式支持】新增PDF/Word/Excel/PPT文档导入解析
- 【开放能力】RESTful API对外开放，支持ISV集成

v1.5版本（2023年3月发布）
- 【新功能】语音转文字输入和TTS语音播报
- 【成本优化】长文本处理token消耗降低30%
- 【管理后台】企业版运营管理后台上线

v1.0版本（2022年12月首发）
- 基础问答能力上线
- 预置100+行业知识模板
    """))

    docs.append(Document(page_content="""
公司三年战略规划（2024-2026）

一、愿景
成为中国企业智能化转型赛道的领跑者。
让先进AI技术普惠每一家企业，助力客户实现10倍效率提升。

二、三年量化目标
2024年：年营收3亿元，团队扩至300人，覆盖5个垂直行业
2025年：年营收5亿元，团队扩至500人，覆盖10个垂直行业
2026年：年营收10亿元，团队扩至800人，国内市场份额进入前三

三、产品路线图
核心产品矩阵：智能AI助手、数据分析平台、自动化中台
技术护城河：每年研发投入占比≥15%，保持技术代差优势
生态策略：开放平台API，三年内发展200家ISV合作伙伴

四、人才战略
关键招聘岗位：NLP算法研究员、资深后端架构师、AI产品经理
人才发展：设立内部技术大学，每季度至少一次全员技术分享
薪酬定位：核心技术岗位薪酬锚定行业75分位值以上
    """))

    # ====== 5篇语义干扰文档 ======
    docs.append(Document(page_content="""
2022年度公司财务报告（历史存档）

2022年全年，公司实现营业收入1.85亿元，同比增长28%。
第四季度营收3800万元。全年净利润2100万元，净利率11.4%。
研发投入3000万元，占营收16.2%。

产品线：AI助手6500万元，数据平台5500万元，企业服务6500万元。
    """))

    docs.append(Document(page_content="""
竞品分析：B公司智能客服产品

B公司推出的智能客服系统，核心功能包括：
多轮对话管理、知识库检索、情感分析、自动路由分配。
响应速度<300ms，日处理咨询量5万次，客户满意度4.3分。

定价模式：按坐席数收费，每坐席每年2000元。
    """))

    docs.append(Document(page_content="""
行业通用数据安全标准（GB/T 35273）

个人信息安全规范国家标准要求：
- 个人敏感信息需单独授权同意
- 数据传输必须采用加密通道
- 数据存储需进行去标识化处理
- 数据泄露需在72小时内上报监管部门
    """))

    docs.append(Document(page_content="""
2023年互联网行业薪酬调研报告

AI算法工程师平均年薪35-60万，中位数45万。
后端开发工程师平均年薪25-45万，中位数32万。
产品经理平均年薪30-50万，中位数38万。

行业平均试用期6个月，试用期薪资一般为转正的80%-90%。
    """))

    docs.append(Document(page_content="""
考勤管理通用实践指南

大多数互联网企业实行弹性工作制，核心工作时间通常为10:00-17:00。
年假常规标准：入职满1年5天，逐年递增，多数公司上限为15-20天。
加班补偿标准参考劳动法：平日1.5倍、休息日2倍、节假日3倍。
    """))

    return docs


# ================================================================
# 测试问题（含标准答案，用于端到端评测）
# ================================================================
@dataclass
class TestQuery:
    query: str
    gold_facts: List[str]         # 检索评测用：相关chunk必须包含的关键事实
    expected_answer: str           # 端到端评测用：期望答案的核心要点
    difficulty: str = "easy"


def create_test_queries() -> List[TestQuery]:
    return [
        # ===== Easy =====
        TestQuery(
            "2023年公司总营收是多少？",
            ["2.5亿元", "营业收入2.5亿"],
            "2023年公司实现营业收入2.5亿元。",
            "easy"
        ),
        TestQuery(
            "公司2026年的营收目标是多少？",
            ["10亿元", "10亿"],
            "公司2026年营收目标为10亿元。",
            "easy"
        ),
        TestQuery(
            "公司使用什么向量数据库？",
            ["Chroma向量数据库", "Chroma"],
            "公司使用Chroma向量数据库存储文档Embedding。",
            "easy"
        ),
        TestQuery(
            "公司的消息队列用的什么？",
            ["RocketMQ"],
            "公司使用RocketMQ作为消息队列，采用双集群主备架构。",
            "easy"
        ),
        TestQuery(
            "产品v2.1版本什么时候发布的？",
            ["2024年1月"],
            "v2.1版本于2024年1月发布。",
            "easy"
        ),

        # ===== Medium =====
        TestQuery(
            "公司目前最大的收入来源是哪个产品？",
            ["AI助手", "占比48%", "最大收入来源"],
            "AI助手产品线是最大收入来源，年销售额1.2亿元，占营收48%。",
            "medium"
        ),
        TestQuery(
            "员工连续旷工多少天算自动离职？",
            ["自动离职", "连续旷工3天"],
            "连续旷工3天或月累计旷工5天，按自动离职处理。",
            "medium"
        ),
        TestQuery(
            "什么情况下工单会升级到技术总监？",
            ["24小时未解决", "L3升级", "技术总监"],
            "工单24小时未解决时触发L3升级，升级至技术总监并启动战时机制。",
            "medium"
        ),
        TestQuery(
            "公司靠什么策略与更大的竞争对手差异化？",
            ["垂直行业定制化", "差异化策略"],
            "公司以垂直行业定制化为差异化策略应对更大的竞争对手。",
            "medium"
        ),
        TestQuery(
            "员工怎么才能提前转正？",
            ["绩效评分均≥4分", "导师书面推荐"],
            "试用期内两次Review绩效评分均≥4分且获得导师书面推荐。",
            "medium"
        ),
        TestQuery(
            "2024年AI市场中增速最快的细分赛道是什么？",
            ["大语言模型", "85%", "800亿元"],
            "大语言模型(LLM)应用赛道增速最快，达到85%，市场规模约800亿元。",
            "medium"
        ),
        TestQuery(
            "产品从哪个版本开始支持PDF等文档格式导入？",
            ["v2.0", "PDF/Word/Excel"],
            "从v2.0版本开始支持PDF、Word、Excel、PPT等文档格式导入解析。",
            "medium"
        ),

        # ===== Hard =====
        TestQuery(
            "公司产品的客户满意度目标和竞品实际满意度哪个更高？",
            ["4.5分", "客户满意度评分"],
            "公司目标≥4.5分，竞品实际满意度4.3分，公司目标更高。",
            "hard"
        ),
        TestQuery(
            "公司未来三年重点招聘哪些技术岗位？",
            ["NLP算法研究员", "资深后端架构师", "AI产品经理"],
            "重点招聘NLP算法研究员、资深后端架构师和AI产品经理。",
            "hard"
        ),
        TestQuery(
            "公司的数据备份策略包含哪几个层次？",
            ["每日增量", "每周全量", "异地灾备"],
            "备份策略包括三个层次：每日增量备份(保留7天)、每周全量备份(保留4周)和异地灾备存储。",
            "hard"
        ),
        TestQuery(
            "产品在哪次版本升级中完成了从单体到微服务的最大架构重构？",
            ["v2.0", "单体应用重构"],
            "在v2.0版本(2023年6月)中完成了从单体应用到微服务架构的最大重构。",
            "hard"
        ),
        TestQuery(
            "公司对员工每月加班有什么硬性限制？",
            ["每月加班上限36小时", "36小时"],
            "每人每月加班上限为36小时，所有加班须提前在OA系统审批。",
            "hard"
        ),
        TestQuery(
            "公司计划在哪一年团队规模突破500人？",
            ["2025年", "500人", "团队扩至500"],
            "公司计划在2025年团队规模突破500人。",
            "hard"
        ),
        TestQuery(
            "AI助手产品的并发处理能力是多少？",
            ["1000QPS", "并发吞吐量1000"],
            "AI助手产品并发吞吐量为1000QPS。",
            "hard"
        ),

        # ===== 口语化/模糊问题（专测查询改写效果） =====
        # 这些问题用口语化表达，与文档原文关键词差异大
        # 不改写很难命中，改写后才能匹配到原文
        TestQuery(
            "公司去年赚了多少？",
            ["2.5亿元", "营业收入2.5亿"],
            "2023年公司实现营业收入2.5亿元。",
            "rewrite"
        ),
        TestQuery(
            "公司的技术框架用的啥？",
            ["Spring Cloud Alibaba", "微服务架构"],
            "公司采用Spring Cloud Alibaba微服务架构。",
            "rewrite"
        ),
        TestQuery(
            "员工请假最多能歇多久？",
            ["上限为15天", "15天", "带薪年假"],
            "带薪年假上限为15天。",
            "rewrite"
        ),
        TestQuery(
            "产品从啥时候开始能上传文件的？",
            ["v2.0", "PDF/Word/Excel"],
            "从v2.0版本开始支持PDF/Word/Excel等文档导入。",
            "rewrite"
        ),
        TestQuery(
            "公司安全方面做了哪些防护？",
            ["双因素认证", "TLS 1.3加密", "堡垒机"],
            "公司采用双因素认证、TLS 1.3加密传输、堡垒机跳板访问等安全措施。",
            "rewrite"
        ),
        TestQuery(
            "新人多久能过试用期？",
            ["3个月", "试用期3个月", "提前转正"],
            "试用期3个月，满足条件可提前转正。",
            "rewrite"
        ),
    ]


# ================================================================
# 检索评测函数
# ================================================================
def is_relevant(doc: Document, gold_facts: List[str]) -> bool:
    text = doc.page_content
    return any(fact in text for fact in gold_facts)


def evaluate_retrieval(
    retrieved: List[Document],
    all_chunks: List[Document],
    gold_facts: List[str],
) -> Dict[str, float]:
    total_rel = sum(1 for c in all_chunks if is_relevant(c, gold_facts))
    if total_rel == 0:
        return {"p@5": 0, "r@5": 0, "mrr": 0, "ndcg@5": 0, "h@1": 0, "h@3": 0, "h@5": 0}

    ret_rel = sum(1 for d in retrieved if is_relevant(d, gold_facts))
    precision = ret_rel / len(retrieved) if retrieved else 0

    ret_texts = {d.page_content for d in retrieved}
    recalled = sum(1 for c in all_chunks if is_relevant(c, gold_facts) and c.page_content in ret_texts)
    recall = recalled / total_rel

    rr = 0.0
    for rank, doc in enumerate(retrieved, 1):
        if is_relevant(doc, gold_facts):
            rr = 1.0 / rank
            break

    dcg = sum(
        (1.0 / math.log2(rank + 2)) if is_relevant(doc, gold_facts) else 0
        for rank, doc in enumerate(retrieved)
    )
    ideal = sorted([1] * min(total_rel, 5) + [0] * max(0, 5 - total_rel), reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    ndcg = dcg / idcg if idcg > 0 else 0

    return {
        "p@5": round(precision, 4), "r@5": round(recall, 4),
        "mrr": round(rr, 4), "ndcg@5": round(ndcg, 4),
        "h@1": 1 if rr >= 1.0 else 0, "h@3": 1 if any(is_relevant(d, gold_facts) for d in retrieved[:3]) else 0,
        "h@5": 1 if any(is_relevant(d, gold_facts) for d in retrieved[:5]) else 0,
    }


# ================================================================
# 端到端答案评测（LLM-as-Judge）
# ================================================================
def evaluate_answer_with_llm(rag, question: str, generated_answer: str, expected_answer: str) -> Dict:
    """
    使用 LLM 作为裁判，评估生成答案的质量
    返回: {score: 1-5, explanation: str}
    """
    judge_prompt = f"""你是一个专业的答案质量评估专家。请对比「生成答案」和「期望答案」，从准确性和完整性两个维度评分。

问题：{question}
期望答案（核心要点）：{expected_answer}
生成答案：{generated_answer}

评分标准：
- 5分：完全准确，包含所有核心信息
- 4分：基本准确，遗漏次要信息
- 3分：部分准确，遗漏重要信息
- 2分：有较多错误或不完整
- 1分：基本错误或答非所问

请只输出一个数字（1-5），不要再输出其他内容。"""

    resp = rag._call_llm(judge_prompt, temperature=0.0)
    # 提取数字
    nums = re.findall(r'\d', resp)
    score = int(nums[0]) if nums and 1 <= int(nums[0]) <= 5 else 3
    return {"score": score, "raw": resp}


# ================================================================
# 主实验
# ================================================================
def run_benchmark():
    print("\n" + "█" * 72)
    print("█  RAG 完整量化对比实验（含LLM端到端评测）")
    print("█" * 72 + "\n")

    # ---- 1. 初始化 RAG ----
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("❌ 未设置 DASHSCOPE_API_KEY！")
        return

    rag = RAGRetrievalBenchmark(
        dashscope_api_key=api_key,
        chunk_size=120,
        chunk_overlap=15,
    )

    # ---- 2. 构建知识库 ----
    documents = create_test_documents()
    rag.build_knowledge_base(documents)
    print(f"   知识库: {len(documents)}篇文档 → {len(rag.all_chunks)}个chunks")

    test_queries = create_test_queries()
    easy_n = sum(1 for q in test_queries if q.difficulty == "easy")
    med_n = sum(1 for q in test_queries if q.difficulty == "medium")
    hard_n = sum(1 for q in test_queries if q.difficulty == "hard")
    rw_n = sum(1 for q in test_queries if q.difficulty == "rewrite")
    print(f"📋 {len(test_queries)} 道测试题 (Easy:{easy_n} Medium:{med_n} Hard:{hard_n} 口语化改写:{rw_n})\n")

    # ================================================================
    # 第一部分：检索策略对比（无查询改写）
    # ================================================================
    strategies = [
        ("vector", "纯向量检索"),
        ("bm25", "纯BM25关键词"),
        ("hybrid", "向量+BM25"),
        ("hybrid_rerank", "混合+CrossEncoder重排"),
    ]

    all_ret_results = {}
    all_latencies = {}

    for skey, sname in strategies:
        print(f"\n{'='*60}")
        print(f"🔬 [检索评测] {sname}")
        print(f"{'='*60}")
        results, lats = [], []
        for i, tq in enumerate(test_queries):
            t0 = time.perf_counter()
            retrieved = rag.retrieve(tq.query, strategy=skey, k=5)
            elapsed = time.perf_counter() - t0
            lats.append(elapsed)
            m = evaluate_retrieval(retrieved, rag.all_chunks, tq.gold_facts)
            m["difficulty"] = tq.difficulty
            results.append(m)
        all_ret_results[skey] = results
        all_latencies[skey] = lats
        # 打印汇总
        avg_recall = statistics.mean(r["r@5"] for r in results)
        avg_mrr = statistics.mean(r["mrr"] for r in results)
        avg_lat = statistics.mean(lats) * 1000
        print(f"   Recall@5 均值: {avg_recall:.4f}  |  MRR 均值: {avg_mrr:.4f}  |  平均延迟: {avg_lat:.1f}ms")

    # ================================================================
    # 第二部分：查询改写效果评测
    # ================================================================
    print(f"\n\n{'='*60}")
    print(f"🔬 [查询改写评测] 混合+重排序 + 查询改写")
    print(f"{'='*60}")

    rewrite_results = []
    rewrite_lats = []
    all_rewritten_queries = []

    for i, tq in enumerate(test_queries):
        t0 = time.perf_counter()
        result = rag.retrieve_with_rewrite(tq.query, strategy="hybrid_rerank", k=5)
        elapsed = time.perf_counter() - t0
        rewrite_lats.append(elapsed)
        m = evaluate_retrieval(result["docs"], rag.all_chunks, tq.gold_facts)
        m["difficulty"] = tq.difficulty
        rewrite_results.append(m)
        all_rewritten_queries.append({
            "original": tq.query,
            "variants": result["rewritten_queries"][1:],  # 去掉原问题
        })
        print(f"  [{i+1:2d}] \"{tq.query[:40]}...\"")
        print(f"       改写: {' | '.join(result['rewritten_queries'][1:])}")
        print(f"       R@5={m['r@5']:.2f}  MRR={m['mrr']:.2f}  ({elapsed*1000:.0f}ms)")

    all_ret_results["hybrid_rerank_rewrite"] = rewrite_results
    all_latencies["hybrid_rerank_rewrite"] = rewrite_lats

    avg_rw_recall = statistics.mean(r["r@5"] for r in rewrite_results)
    avg_rw_mrr = statistics.mean(r["mrr"] for r in rewrite_results)
    avg_rw_lat = statistics.mean(rewrite_lats) * 1000
    print(f"\n   Recall@5 均值: {avg_rw_recall:.4f}  |  MRR 均值: {avg_rw_mrr:.4f}  |  平均延迟: {avg_rw_lat:.1f}ms")

    # ================================================================
    # 第三部分：端到端答案质量评测（LLM-as-Judge）
    # ================================================================
    print(f"\n\n{'='*60}")
    print(f"🔬 [端到端评测] RAG 完整问答 + LLM-as-Judge 打分")
    print(f"{'='*60}")

    e2e_results = []

    for i, tq in enumerate(test_queries):
        # 用最优检索策略（混合+重排序+改写）检索
        result = rag.retrieve_with_rewrite(tq.query, strategy="hybrid_rerank", k=5)

        # 生成答案
        answer = rag.generate_answer(tq.query, result["docs"])

        # LLM-as-Judge 评分
        judge = evaluate_answer_with_llm(rag, tq.query, answer, tq.expected_answer)

        e2e_results.append({
            "question": tq.query,
            "answer": answer,
            "expected": tq.expected_answer,
            "score": judge["score"],
            "difficulty": tq.difficulty,
        })
        print(f"  [{i+1:2d}] [{tq.difficulty:6s}] Q: {tq.query[:45]}...")
        print(f"       A: {answer[:80]}...")
        print(f"       LLM-Judge 评分: {judge['score']}/5")

    # ================================================================
    # 汇总报告
    # ================================================================
    _print_full_report(all_ret_results, all_latencies, e2e_results, test_queries, all_rewritten_queries)


def _print_full_report(all_ret_results, all_latencies, e2e_results, test_queries, rewritten_queries):
    print("\n\n" + "█" * 72)
    print("█  📊 最终综合报告")
    print("█" * 72)

    skey_order = ["vector", "bm25", "hybrid", "hybrid_rerank", "hybrid_rerank_rewrite"]
    skey_label = {
        "vector": "纯向量", "bm25": "纯BM25", "hybrid": "向量+BM25",
        "hybrid_rerank": "混合+重排", "hybrid_rerank_rewrite": "混合+重排+改写"
    }
    mkeys = ["p@5", "r@5", "mrr", "ndcg@5", "h@1", "h@3", "h@5"]
    mlabels = ["Precision@5", "Recall@5", "MRR", "NDCG@5", "Hit@1", "Hit@3", "Hit@5"]

    # ====== 1. 检索指标对比 ======
    print("\n" + "=" * 72)
    print("📊 一、检索质量对比（全部策略，25题）")
    print("=" * 72)

    header = f"{'指标':<18}" + "".join(f"{skey_label[s]:>16}" for s in skey_order)
    print(header)
    print("-" * len(header))

    for mkey, mlabel in zip(mkeys, mlabels):
        vals = [statistics.mean(r[mkey] for r in all_ret_results[s]) for s in skey_order]
        if mkey.startswith("h"):
            row = f"{mlabel:<18}" + "".join(f"{v:>16.1%}" for v in vals)
        else:
            row = f"{mlabel:<18}" + "".join(f"{v:>16.4f}" for v in vals)
        print(row)

    # 延迟
    print("-" * len(header))
    lat_row = f"{'平均延迟':<18}" + "".join(f"{statistics.mean(all_latencies[s])*1000:>16.1f}" for s in skey_order)
    print(lat_row + " ms")

    # ====== 2. 按难度分层 ======
    print("\n" + "=" * 72)
    print("📊 二、按难度分层 — Recall@5 对比")
    print("=" * 72)
    diff_header = f"{'难度':<16}" + "".join(f"{skey_label[s]:>16}" for s in skey_order)
    print(diff_header)
    print("-" * len(diff_header))
    for diff, diff_name in [("easy", "简单"), ("medium", "中等"), ("hard", "困难"), ("rewrite", "口语化改写")]:
        vals = [statistics.mean(r["r@5"] for r in all_ret_results[s] if r["difficulty"] == diff)
                for s in skey_order]
        print(f"{diff_name:<16}" + "".join(f"{v:>16.4f}" for v in vals))

    # ====== 3. 查询改写增益 ======
    print("\n" + "=" * 72)
    print("📊 三、查询改写增益（整体 vs 口语化改写类题目）")
    print("=" * 72)

    base = "hybrid_rerank"
    rw = "hybrid_rerank_rewrite"

    # 整体对比
    print("\n  【全部题目】混合+重排 vs 混合+重排+改写：")
    for mkey, mlabel in zip(mkeys, mlabels):
        b_val = statistics.mean(r[mkey] for r in all_ret_results[base])
        rw_val = statistics.mean(r[mkey] for r in all_ret_results[rw])
        if b_val > 0:
            pct = (rw_val - b_val) / b_val * 100
            arrow = "↑" if pct > 0 else "↓"
            if mkey.startswith("h"):
                print(f"    {mlabel}: {b_val:.1%} → {rw_val:.1%} ({arrow}{abs(pct):.1f}%)")
            else:
                print(f"    {mlabel}: {b_val:.4f} → {rw_val:.4f} ({arrow}{abs(pct):.1f}%)")

    # ★ 重点：口语化改写类题目对比
    print("\n  【口语化改写类6题】不改写 vs 改写（核心对比 ★）：")
    for mkey, mlabel in zip(mkeys, mlabels):
        b_val = statistics.mean(r[mkey] for r in all_ret_results[base] if r["difficulty"] == "rewrite")
        rw_val = statistics.mean(r[mkey] for r in all_ret_results[rw] if r["difficulty"] == "rewrite")
        if b_val > 0:
            pct = (rw_val - b_val) / b_val * 100
            arrow = "↑" if pct > 0 else "↓"
            if mkey.startswith("h"):
                print(f"    {mlabel}: {b_val:.1%} → {rw_val:.1%} ({arrow}{abs(pct):.1f}%)")
            else:
                print(f"    {mlabel}: {b_val:.4f} → {rw_val:.4f} ({arrow}{abs(pct):.1f}%)")

    # 展示几个改写样例
    print("\n  查询改写样例：")
    for rw in rewritten_queries[:4]:
        print(f"    原问: {rw['original']}")
        for v in rw['variants']:
            print(f"      → {v}")

    # ====== 4. 端到端答案质量 ======
    print("\n" + "=" * 72)
    print("📊 四、端到端答案质量评测（LLM-as-Judge, 1-5分）")
    print("=" * 72)

    scores = [r["score"] for r in e2e_results]
    avg_score = statistics.mean(scores)
    score_dist = {s: scores.count(s) for s in range(1, 6)}

    print(f"  平均评分: {avg_score:.2f}/5")
    print(f"  评分分布: " + " | ".join(f"{s}分: {score_dist.get(s, 0)}题" for s in range(1, 6)))

    for diff in ["easy", "medium", "hard", "rewrite"]:
        diff_scores = [r["score"] for r in e2e_results if r["difficulty"] == diff]
        if diff_scores:
            print(f"  {diff}: 平均 {statistics.mean(diff_scores):.2f}/5")

    # 展示几个问答样例
    print("\n  问答样例：")
    for r in e2e_results[:3]:
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer'][:100]}")
        print(f"  期望: {r['expected'][:100]}")
        print(f"  评分: {r['score']}/5\n")

    # ====== 5. 简历可用的核心数据 ======
    print("=" * 72)
    print("📈 五、简历可用的量化结论")
    print("=" * 72)

    vector_r5 = statistics.mean(r["r@5"] for r in all_ret_results["vector"])
    rerank_r5 = statistics.mean(r["r@5"] for r in all_ret_results["hybrid_rerank"])
    rewrite_r5 = statistics.mean(r["r@5"] for r in all_ret_results["hybrid_rerank_rewrite"])

    vector_h5 = statistics.mean(r["h@5"] for r in all_ret_results["vector"])
    rerank_h5 = statistics.mean(r["h@5"] for r in all_ret_results["hybrid_rerank"])
    rewrite_h5 = statistics.mean(r["h@5"] for r in all_ret_results["hybrid_rerank_rewrite"])

    # 困难场景
    hard_vector_r5 = statistics.mean(r["r@5"] for r in all_ret_results["vector"] if r["difficulty"]=="hard")
    hard_rerank_r5 = statistics.mean(r["r@5"] for r in all_ret_results["hybrid_rerank"] if r["difficulty"]=="hard")
    hard_rewrite_r5 = statistics.mean(r["r@5"] for r in all_ret_results["hybrid_rerank_rewrite"] if r["difficulty"]=="hard")

    # 口语化改写场景——对比「混合+重排」vs「混合+重排+改写」（控制变量，只有改写与否不同）
    rw_norewrite_r5 = statistics.mean(r["r@5"] for r in all_ret_results["hybrid_rerank"] if r["difficulty"]=="rewrite")
    rw_rewrite_r5 = statistics.mean(r["r@5"] for r in all_ret_results["hybrid_rerank_rewrite"] if r["difficulty"]=="rewrite")
    rw_norewrite_h5 = statistics.mean(r["h@5"] for r in all_ret_results["hybrid_rerank"] if r["difficulty"]=="rewrite")
    rw_rewrite_h5 = statistics.mean(r["h@5"] for r in all_ret_results["hybrid_rerank_rewrite"] if r["difficulty"]=="rewrite")

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. 混合检索 + 重排序 vs 纯向量检索（全部25题）                  │
  │    Recall@5:  {vector_r5:.1%} → {rerank_r5:.1%} (↑{(rerank_r5-vector_r5)/vector_r5*100:.1f}%)                          │
  │    Hit@5:     {vector_h5:.1%} → {rerank_h5:.1%} (↑{(rerank_h5-vector_h5)/vector_h5*100:.1f}%)                          │
  │                                                             │
  │ 2. 口语化改写场景（6题）——查询改写的核心价值                    │
  │    不改写 Recall@5:  {rw_norewrite_r5:.1%}                                  │
  │    LLM改写 Recall@5: {rw_rewrite_r5:.1%} (↑{(rw_rewrite_r5-rw_norewrite_r5)/rw_norewrite_r5*100:.1f}%)                            │
  │    不改写 Hit@5:     {rw_norewrite_h5:.1%}                                  │
  │    LLM改写 Hit@5:    {rw_rewrite_h5:.1%} (↑{(rw_rewrite_h5-rw_norewrite_h5)/rw_norewrite_h5*100:.1f}%)                            │
  │                                                             │
  │ 3. 困难场景（含语义干扰，7题）                                  │
  │    纯向量 Recall@5: {hard_vector_r5:.1%}                                  │
  │    混合+重排+改写:  {hard_rewrite_r5:.1%} (↑{(hard_rewrite_r5-hard_vector_r5)/hard_vector_r5*100:.1f}%)                            │
  │                                                             │
  │ 4. 端到端答案质量                                            │
  │    LLM-as-Judge 平均评分: {avg_score:.2f}/5                              │
  │    5分: {score_dist.get(5,0)}题  4分: {score_dist.get(4,0)}题  3分: {score_dist.get(3,0)}题  2分: {score_dist.get(2,0)}题  1分: {score_dist.get(1,0)}题      │
  └─────────────────────────────────────────────────────────────┘
""")

    # ====== 保存报告 ======
    report = {
        "retrieval_metrics": {s: {mk: statistics.mean(r[mk] for r in all_ret_results[s])
                                  for mk in mkeys} for s in skey_order},
        "e2e_avg_score": avg_score,
        "e2e_score_distribution": score_dist,
        "rewrite_examples": rewritten_queries[:3],
    }
    with open("benchmark_full_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("✅ 完整报告已保存到 benchmark_full_report.json")


if __name__ == "__main__":
    run_benchmark()
