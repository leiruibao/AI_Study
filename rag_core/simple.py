"""
简化版 RAG 智能问答系统
功能：基于文档的问答 API（支持流式输出）
技术：FastAPI + LlamaIndex + DeepSeek
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

from llama_index.core.llms import ChatMessage

# ==================== 配置参数 ====================
# 这里集中管理所有配置，方便修改

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")  # 从环境变量获取
DEEPSEEK_MODEL = "deepseek-chat"
EMBED_MODEL = "BAAI/bge-base-zh-v1.5"  # 中文向量模型

# 使用绝对路径，避免路径问题
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")       # 放文档的文件夹
STORAGE_PATH = os.path.join(BASE_DIR, "storage") # 索引保存位置

DATA_BASE_PATH = os.path.join(BASE_DIR, "data")
STORAGE_BASE_PATH = os.path.join(BASE_DIR, "storage")


# ==================== 请求和响应的数据结构 ====================

class QueryRequest(BaseModel):
    """用户提问的请求格式"""
    query: str           # 用户的问题
    stream: bool = False # 是否使用流式输出（默认否）


class QueryResponse(BaseModel):
    """系统回答的响应格式（非流式）"""
    answer: str  # AI的回答


# ==================== 全局变量：改为字典映射 ====================
# 用来存储已经加载好的索引，避免重复加载

index = None  # 向量索引（用于检索文档）


# ==================== 核心函数 ====================

def ensure_directories():
    """
    确保必要的目录存在
    如果不存在就自动创建
    """
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(STORAGE_PATH, exist_ok=True)
    print(f"📂 数据目录：{DATA_PATH}")
    print(f"💾 索引目录：{STORAGE_PATH}")


def init_system():
    """
    初始化系统：配置大语言模型和向量模型
    这个函数在服务启动时只运行一次
    """
    print("🔧 正在初始化系统...")

    # 检查 API Key 是否配置
    if not DEEPSEEK_API_KEY:
        raise ValueError("❌ 请设置环境变量 DEEPSEEK_API_KEY")

    # 1. 配置大语言模型（用于生成回答）
    Settings.llm = OpenAILike(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        api_base="https://api.deepseek.com/v1",
        is_chat_model=True,  # 关键：告诉 LlamaIndex 这是一个对话模型
        temperature=0.3,
        # 强制指定上下文窗口大小，防止它因为不认识 deepseek 而报错
        context_window=64000
    )

    # 2. 配置向量模型（用于理解文档内容）
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device="cpu"  # 使用 CPU，如果有 GPU 可以改成 "cuda"
    )

    # 3. 设置文本分块大小（把长文档切成小块，方便检索）
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50  # 块之间有重叠，避免信息丢失

    print("✅ 系统初始化完成\n")


def load_or_create_index():
    """
    加载或创建索引
    - 如果已有索引：直接加载
    - 如果没有：读取文档并创建新索引
    """
    global index

    # 情况1：已经有索引了，直接加载
    if os.path.exists(STORAGE_PATH) and os.listdir(STORAGE_PATH):
        print(f"📂 从 {STORAGE_PATH} 加载已有索引...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_PATH)
        index = load_index_from_storage(storage_context)
        print("✅ 索引加载成功\n")
        return

    # 情况2：没有索引，需要创建
    print(f"📚 未找到索引，开始构建...")

    # 检查 data 目录是否有文件
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print("⚠️  data 目录为空，请先添加文档文件")
        print(f"   请将文档（txt, pdf, docx 等）放到：{DATA_PATH}")
        # 创建空索引，避免报错
        index = VectorStoreIndex.from_documents([])
        return

    print(f"📖 读取文档目录：{DATA_PATH}")

    # 读取所有文档（支持 txt, pdf, docx 等格式）
    reader = SimpleDirectoryReader(input_dir=DATA_PATH)
    documents = reader.load_data()

    print(f"📄 共读取 {len(documents)} 个文档")
    print("🔨 正在构建索引...")

    # 创建向量索引（这一步会比较慢，需要处理所有文档）
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True  # 显示进度条
    )

    # 保存索引到磁盘，下次直接加载
    index.storage_context.persist(persist_dir=STORAGE_PATH)
    print("✅ 索引构建并保存成功\n")


def get_answer(question: str) -> str:
    """
    根据问题返回答案（非流式）

    工作流程：
    1. 用向量检索找到相关文档片段
    2. 把片段和问题一起发给大模型
    3. 大模型基于文档内容生成回答
    """
    if index is None:
        raise ValueError("索引未加载")

    print(f"🔍 正在处理问题：{question[:50]}...")

    # 创建查询引擎（负责检索+生成回答）
    query_engine = index.as_query_engine(
        similarity_top_k=3  # 检索最相关的 3 个文档片段
    )

    # 执行查询
    response = query_engine.query(question)

    print("✅ 回答生成完成")
    return str(response)


def get_answer_stream(question: str):
    """
    根据问题返回答案（流式输出）

    流式输出的好处：
    - 不用等待所有内容生成完才看到结果
    - 逐字输出，体验更好
    - 适合长回答
    """
    if index is None:
        raise ValueError("索引未加载")

    print(f"🔍 正在处理流式问题：{question[:50]}...")

    # 创建查询引擎，开启流式模式
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        streaming=True  # 关键：开启流式输出
    )

    # 执行查询，返回流式响应对象
    streaming_response = query_engine.query(question)

    # 返回生成器，逐个输出 token
    return streaming_response.response_gen


# ==================== FastAPI 应用 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理（新版 FastAPI 推荐写法）
    - 启动时执行初始化
    - 关闭时执行清理
    """
    # 启动时执行
    print("🚀 RAG API 服务启动中...")
    ensure_directories()  # 确保目录存在
    init_system()
    load_or_create_index()
    print("✅ 服务启动完成，可以开始提问了！\n")

    yield  # 这里是服务运行期间

    # 关闭时执行
    print("🛑 服务关闭中...")


app = FastAPI(
    title="RAG 智能问答 API",
    description="基于文档的智能问答系统（支持流式输出）",
    version="1.0.0",
    lifespan=lifespan  # 绑定生命周期管理
)


@app.get("/")
async def root():
    """首页，显示 API 信息"""
    return {
        "service": "RAG 智能问答 API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "问答接口（支持流式和非流式）",
            "GET /": "API 信息"
        },
        "usage": {
            "非流式": {"query": "你的问题", "stream": False},
            "流式": {"query": "你的问题", "stream": True}
        }
    }


@app.post("/query")
async def query(request: QueryRequest):
    """
    问答接口（支持流式和非流式）

    使用方法：
    POST http://localhost:8000/query

    非流式：
    Body: {"query": "你的问题", "stream": false}

    流式：
    Body: {"query": "你的问题", "stream": true}
    """
    try:
        # 根据 stream 参数决定使用哪种模式
        if request.stream:
            # 流式输出：逐字返回
            def generate():
                """生成器函数，用于流式输出"""
                for token in get_answer_stream(request.query):
                    yield token  # 每次返回一个字符或词

            # 返回流式响应
            return StreamingResponse(
                generate(),
                media_type="text/plain"  # 纯文本格式
            )
        else:
            # 非流式输出：一次性返回完整答案
            answer = get_answer(request.query)
            return QueryResponse(answer=answer)

    except Exception as e:
        # 如果出错，返回错误信息
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.post("/mentor-chat")
async def mentor_chat(request: Request):
    data = await request.json()
    q_id = data.get("qId")
    user_query = data.get("query")
    history_data = data.get("history", [])  # 前端传来的对话数组

    # 1. 将前端传来的 history 转换为 LlamaIndex 需要的 ChatMessage 对象
    chat_history = []
    for msg in history_data:
        role = "user" if msg["role"] == "user" else "assistant"
        chat_history.append(ChatMessage(role=role, content=msg["content"]))

    # 2. 创建一个上下文对话引擎
    # context_template 可以告诉 AI 它的身份
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        system_prompt=(
            "你是一名资深的法考导师。你现在的任务是根据提供的法律案例背景，"
            "回答学生针对该案例的追问。你的回答要专业、严谨，并多引用案例中的事实。"
        ),
    )

    # 3. 发起对话（传入历史，AI 就能记住之前聊过什么）
    response = chat_engine.chat(user_query, chat_history=chat_history)

    return {"answer": response.response}


# ==================== 启动服务 ====================

if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("🚀 启动 RAG API 服务")
    print("=" * 50)

    # 启动 Web 服务
    uvicorn.run(
        app,
        host="0.0.0.0",  # 允许外部访问
        port=8000,       # 端口号
        reload=False     # 生产环境建议关闭自动重载
    )