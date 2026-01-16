"""
RAG智能问答系统 - 基于LlamaIndex + DeepSeek + 本地Embedding
功能：读取本地文档，构建向量索引，支持智能问答和成本追踪
"""

import os
import shutil
from pathlib import Path
#os.environ["HF_HUB_OFFLINE"] = "1"
# 2. 关键：改用国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = r"F:\AI_Models\huggingface"
from urllib import response
import tiktoken
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    PromptTemplate
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.readers.file import PyMuPDFReader
#from llama_index.core import Settings
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer



# ==================== 配置区 ====================

class Config:
    """系统配置类 - 所有可调参数集中管理"""

    # DeepSeek API配置
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL = "deepseek-chat"
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    TEMPERATURE = 0.3  # 控制回答的创造性(0-1，越高越随机)

    # 本地嵌入模型配置
    EMBED_MODEL = "BAAI/bge-base-zh-v1.5"  # 中文向量模型
    RERANKER_MODEL = "BAAI/bge-reranker-base"  # 重排序模型

    # 文本分块参数
    CHUNK_SIZE = 512  # 每个文本块的大小
    CHUNK_OVERLAP = 50  # 文本块之间的重叠字符数

    # 检索参数
    TOP_K = 10  # 初步检索的文档数量
    RERANK_TOP_N = 5  # 重排序后保留的文档数量

    # 成本计算(DeepSeek官方价格 USD/百万tokens)
    PRICE_INPUT = 0.14 / 1_000_000  # 输入价格
    PRICE_OUTPUT = 0.28 / 1_000_000  # 输出价格


# ==================== 提示词模板 ====================

QA_PROMPT_TEMPLATE = """你是一位深思熟虑的系统架构分析专家。下面是相关的文本片段：

---------------------
{context_str}
---------------------

请基于提供的参考内容回答问题：{query_str}

要求：
1. 不要简单复读原文，要根据上下文进行合理解读，并给出专业建议。
2. 如果原文没有直说，请结合语境推断。
3. 如果参考内容完全不相关，请诚实说明
4. 回答要条理清晰，易于理解

回答："""


# ==================== 系统初始化 ====================

def init_settings():
    """初始化全局设置：LLM、Embedding、Token计数器"""

    print("🔧 正在初始化系统...")

    # 1. 配置Token计数器(用于追踪成本)
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.get_encoding("cl100k_base").encode
    )
    Settings.callback_manager = CallbackManager([token_counter])

    # 2. 配置云端大语言模型(DeepSeek)
    if not Config.DEEPSEEK_API_KEY:
        raise ValueError("❌ 请设置环境变量 DEEPSEEK_API_KEY")

    Settings.llm = OpenAILike(
        model=Config.DEEPSEEK_MODEL,
        api_key=Config.DEEPSEEK_API_KEY,
        api_base=Config.DEEPSEEK_API_BASE,
        temperature=Config.TEMPERATURE,
        is_chat_model=True,
        timeout=120.0 # 显式设置超时时间为 120 秒
    )

    # 3. 配置本地嵌入模型(用于向量化文本)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBED_MODEL,
        embed_batch_size=40,  # CPU 建议 32-64，提高向量化速度
        device="cpu"
    )
    Settings.chunk_size = Config.CHUNK_SIZE
    Settings.chunk_overlap = Config.CHUNK_OVERLAP

    print("✅ 系统初始化完成\n")
    return token_counter


# ==================== 索引管理 ====================

def get_or_create_index(data_path, storage_path):
    """
    获取或创建向量索引
    - 如果索引已存在，直接加载
    - 如果不存在，从文档创建新索引
    """

    if os.path.exists(storage_path):
        print(f"📂 发现已有索引，从 {storage_path} 加载...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
        print("✅ 索引加载成功")
    else:
        print(f"📚 未找到索引，开始构建新索引...")
        print(f"📖 读取文档：{data_path}")

        # 1. 高速读取
        reader = SimpleDirectoryReader(
            input_dir=data_path,
            file_extractor={".pdf": PyMuPDFReader()} 
        )
        documents = reader.load_data()

        # # 2. 诊断文本量
        # char_count = sum(len(d.text) for d in documents)
        # print(f"✅ 解析完成！共 {len(documents)} 页，总字符数: {char_count:,}")

        # # 3. 并行构建（增加进度条）
        # print("🔨 正在并行构建向量索引（请看进度条）...")
        # index = VectorStoreIndex.from_documents(
        #     documents,
        #     show_progress=True, # 强烈建议开启，避免焦虑
        #     num_workers=8       # 充分利用 i7 的多核
        # )

        print(f"🔨 正在构建索引，开启多核加速...")
        # show_progress=True 会显示进度条，你就知道它没卡死
        index = VectorStoreIndex.from_documents(
            documents, 
            show_progress=True, 
            num_workers=4  # i7-13700H 可以设为 4-8，显著提升 CPU 向量化效率
        )

        index.storage_context.persist(persist_dir=storage_path)
        print("✅ 索引构建完成")

        # if not os.path.exists(data_path):
        #     raise FileNotFoundError(f"❌ 数据目录不存在：{data_path}")

        # documents = SimpleDirectoryReader(data_path).load_data()
        # print(f"✅ 共读取 {len(documents)} 个文档")

        # print("🔨 构建向量索引中...")
        # index = VectorStoreIndex.from_documents(documents)

        # print(f"💾 保存索引到：{storage_path}")
        # index.storage_context.persist(persist_dir=storage_path)
        # print("✅ 索引构建完成")

    return index


# ==================== 查询引擎 ====================

def create_query_engine(index):
    """创建配置好的查询引擎"""

    print("⚙️  配置查询引擎...")

    # 初始化重排序器(提高检索精度)
    reranker = FlagEmbeddingReranker(
        model=Config.RERANKER_MODEL,
        top_n=Config.RERANK_TOP_N
    )

    # 创建查询引擎
    query_engine = index.as_query_engine(
        similarity_top_k=Config.TOP_K,
        node_postprocessors=[reranker],
        text_qa_template=PromptTemplate(QA_PROMPT_TEMPLATE)
    )

    print("✅ 查询引擎就绪\n")
    return query_engine

def create_chat_engine(index):
    print("⚙️  配置对话式查询引擎 (带记忆功能)...")
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    # 1. 初始化重排序器
    reranker = FlagEmbeddingReranker(
        model=Config.RERANKER_MODEL,
        top_n=Config.RERANK_TOP_N
    )

    # 2. 将 Index 转换为 Chat Engine
    # chat_mode="condense_plus_context" 是最强大的模式：
    # 它会先压缩问题，再检索上下文，最后回答
    # chat_engine = index.as_chat_engine(
    #     chat_mode="condense_plus_context",
    #     similarity_top_k=Config.TOP_K,
    #     node_postprocessors=[reranker],
    #     system_prompt=(
    #         "你是一位深思熟虑的系统架构分析专家。"
    #         "你会结合对话历史和提供的参考内容来回答问题。"
    #         "如果用户要求解释之前的回答，请务必结合之前的上下文。"
    #     ),
    #     # 保持之前的提示词风格
    #     context_prompt=QA_PROMPT_TEMPLATE 
    # )

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        streaming=True,  # 必须开启流式输出才能边答边显示
        memory=memory, # 关键：引入有上限的记忆
        similarity_top_k=3, # 建议将 10 改为 5，减少 context 负担
        #node_postprocessors=[reranker], # 之前的重排序器
        system_prompt="你是一位金融/政务集成架构专家...",
        context_prompt=QA_PROMPT_TEMPLATE
    )

    print("✅ 对话引擎就绪\n")
    return chat_engine


# ==================== 成本统计 ====================

def print_token_stats(token_counter):
    """打印Token消耗和成本统计"""

    prompt_tokens = token_counter.prompt_llm_token_count
    completion_tokens = token_counter.completion_llm_token_count
    total_cost = (
            prompt_tokens * Config.PRICE_INPUT +
            completion_tokens * Config.PRICE_OUTPUT
    )

    print("\n" + "=" * 50)
    print("💰 本次对话Token消耗统计:")
    print(f"   📥 输入Token:  {prompt_tokens:,}")
    print(f"   📤 输出Token:  {completion_tokens:,}")
    print(f"   💵 总计费用:   ${total_cost:.6f} USD")
    print("=" * 50)


def print_sources(source_nodes):
    """打印检索到的来源文档"""

    print("\n📚 来源依据:")
    print("-" * 50)
    for i, node in enumerate(source_nodes, 1):
        score = node.score
        content = node.node.get_content()[:100].replace('\n', ' ')
        print(f"{i}. [相关度: {score:.4f}]")
        print(f"   {content}...")
        print()


# ==================== 主程序 ====================

def start_rag():
    """RAG系统主入口"""

    # 获取路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data")
    storage_path = os.path.join(current_dir, "..", "storage")

    # 初始化系统
    token_counter = init_settings()

    # 加载或创建索引
    index = get_or_create_index(data_path, storage_path)

    # 创建查询引擎
    # query_engine = create_query_engine(index)

    # 创建对话式查询引擎
    chat_engine = create_chat_engine(index)

    # 交互式问答循环
    print("=" * 50)
    print("🤖 RAG智能问答系统已启动")
    print("💡 提示：输入 'exit' 或 'quit' 退出系统")
    print("=" * 50)

    while True:
        # 获取用户输入
        user_input = input("\n❓ 请输入问题: ").strip()

        # 退出条件
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("\n👋 感谢使用，再见！")
            break

        # 跳过空输入
        if not user_input:
            continue

        # 重置计数器
        token_counter.reset_counts()

        # 执行查询
        try:
            # print("\n🔍 正在检索相关文档...")
            # response = query_engine.query(user_input)

            print("\n🔍 正在分析问题并检索...")

            response = chat_engine.stream_chat(user_input)

            print("\n✨ 回答:")
            # 迭代打印流式输出
            for token in response.response_gen:
                print(token, end="", flush=True)
            print("\n") # 换行
            
            # 注意这里改用 chat() 而不是 query()
            # response = chat_engine.chat(user_input)
            # 输出答案
            # print(f"\n✨ 回答:\n{response}\n")

            # 显示Token统计
            # print_token_stats(token_counter)

            # 显示来源文档
            # if hasattr(response, 'source_nodes'):
            #     print_sources(response.source_nodes)

        except Exception as e:
            print(f"\n❌ 出错了: {str(e)}")


# ==================== 程序入口 ====================

if __name__ == "__main__":
    try:
        start_rag()
    except KeyboardInterrupt:
        print("\n\n👋 程序已中断，再见！")
    except Exception as e:
        print(f"\n❌ 系统错误: {str(e)}")
        raise