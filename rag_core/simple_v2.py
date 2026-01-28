"""
多学科强化版 RAG 智能问答系统
功能：支持学科路由（crim, civ, java等）、自动预热加载、流式输出
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

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = "deepseek-chat"
EMBED_MODEL = "BAAI/bge-base-zh-v1.5"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BASE_PATH = os.path.join(BASE_DIR, "data")
STORAGE_BASE_PATH = os.path.join(BASE_DIR, "storage")

# ==================== 全局状态：学科索引映射 ====================
# key: 学科标识 (如 crim), value: 对应的 VectorStoreIndex 对象
index_map = {}


# ==================== 请求和响应的数据结构 ====================
class QueryRequest(BaseModel):
    query: str
    stream: bool = False
    subject: str = "default"  # 接收来自 Java 端的学科标识 (subject_code)


class QueryResponse(BaseModel):
    answer: str


# ==================== 核心逻辑函数 ====================

def init_system():
    """初始化 LLM 和 Embedding 配置"""
    print("🔧 正在初始化全局 AI 配置...")
    if not DEEPSEEK_API_KEY:
        raise ValueError("❌ 请设置环境变量 DEEPSEEK_API_KEY")

    Settings.llm = OpenAILike(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        api_base="https://api.deepseek.com/v1",
        is_chat_model=True,
        temperature=1,
        context_window=64000
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device="cpu"
    )

    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    print("✅ 全局配置初始化完成")


def load_index_for_subject(subject_name: str):
    """根据学科名加载或创建索引 (核心路由逻辑)"""
    global index_map

    # 路径准备
    subject_data_path = os.path.join(DATA_BASE_PATH, subject_name)
    subject_storage_path = os.path.join(STORAGE_BASE_PATH, subject_name)

    # 确保文件夹存在
    os.makedirs(subject_data_path, exist_ok=True) # 如果目录链中有不存在的文件夹则会自动创建，True代表存在的话不会报错
    os.makedirs(subject_storage_path, exist_ok=True)

    # 尝试加载
    if os.path.exists(subject_storage_path) and os.listdir(subject_storage_path):
        print(f"📂 正在从磁盘加载【{subject_name}】索引...")
        storage_context = StorageContext.from_defaults(persist_dir=subject_storage_path) # StorageContext 是 LlamaIndex 库里的一个“配置管家”
        idx = load_index_from_storage(storage_context) # 
    else:
        # 如果没有索引则构建
        print(f"📚 正在为【{subject_name}】构建新索引...")
        if not os.listdir(subject_data_path):
            print(f"⚠️  警告: 【{subject_name}】数据目录为空，创建空索引")
            idx = VectorStoreIndex.from_documents([])
        else:
            # 本地文档变成 AI 能懂的数据库
            reader = SimpleDirectoryReader(input_dir=subject_data_path) # 找搬运工。实例化一个扫描器，瞄准存放文档的文件夹
            documents = reader.load_data() # 搬货上车。把 PDF/Word/TXT 等原始文件读进内存，变成代码能处理的通用格式。
            idx = VectorStoreIndex.from_documents(documents, show_progress=True) # 切碎并索引（核心步骤）。把文档切成小块（Chunk），计算特征值（Embedding），做成类似字典的“索引书架”。
            idx.storage_context.persist(persist_dir=subject_storage_path) # 存入仓库。把内存里做好的索引保存到硬盘，下次启动直接读，不用再重复前三步。

    index_map[subject_name] = idx
    return idx


def warmup_indexes():
    """启动预热：遍历 data 文件夹加载所有学科索引"""
    print("🔥 正在启动预热，预加载所有学科索引...")
    if not os.path.exists(DATA_BASE_PATH):
        os.makedirs(DATA_BASE_PATH)
        return

    # 获取所有子目录
    subjects = [d for d in os.listdir(DATA_BASE_PATH)
                if os.path.isdir(os.path.join(DATA_BASE_PATH, d))]

    for sub in subjects:
        try:
            load_index_for_subject(sub)
        except Exception as e:
            print(f"❌ 加载学科【{sub}】失败: {str(e)}")

    print(f"✅ 预热完成，已加载学科: {list(index_map.keys())}")


# ==================== FastAPI 生命周期 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 服务启动中...")
    init_system()
    warmup_indexes()  # 执行预热
    yield
    print("🛑 服务关闭中...")


app = FastAPI(title="RAG 多学科 API", lifespan=lifespan) # lifespan：函数名。你可以起名叫 startup_and_shutdown，但在 FastAPI 里约定俗成叫 lifespan（生命周期）。


# ==================== API 接口 ====================

@app.get("/")
async def root():
    return {"loaded_subjects": list(index_map.keys()), "status": "running"}


@app.post("/query")
async def query(request: QueryRequest):
    """问答接口：支持学科路由"""
    # 1. 获取学科索引（如果预热没加载到，这里会动态尝试加载）
    subject = request.subject if request.subject in index_map else "default"
    if subject not in index_map:
        # 尝试动态加载（比如运行期间新加了文件夹）
        try:
            current_index = load_index_for_subject(request.subject)
        except:
            raise HTTPException(status_code=404, detail=f"学科库 {request.subject} 不存在")
    else:
        current_index = index_map[subject]

    try:
        if request.stream:
            def generate():
                query_engine = current_index.as_query_engine(streaming=True, similarity_top_k=3)
                response = query_engine.query(request.query)
                for token in response.response_gen:
                    yield token

            return StreamingResponse(generate(), media_type="text/plain")
        else:
            query_engine = current_index.as_query_engine(similarity_top_k=3)
            response = query_engine.query(request.query)
            return QueryResponse(answer=str(response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mentor-chat")
async def mentor_chat(request: Request):
    """导师对话接口：带历史记忆和学科背景"""
    data = await request.json()
    subject_code = data.get("subject", "default")
    user_query = data.get("query")
    history_data = data.get("history", [])

    # 获取索引
    if subject_code not in index_map:
        # 如果没加载过则尝试加载
        current_index = load_index_for_subject(subject_code)
    else:
        current_index = index_map[subject_code]

    # 转换历史格式
    chat_history = []
    for msg in history_data:
        role = "user" if msg["role"] == "user" else "assistant"
        chat_history.append(ChatMessage(role=role, content=msg["content"]))

    # 创建对话引擎
    chat_engine = current_index.as_chat_engine(
        chat_mode="context",
        system_prompt=(
            f"你是一名资深的【{subject_code}】专家导师。请根据提供的文档背景回答学生的问题。"
            "回答要专业严谨，多引用背景材料中的事实。"
        )
    )

    response = chat_engine.chat(user_query, chat_history=chat_history)
    return {"answer": response.response}


@app.post("/mentor-chat-stream")
async def mentor_chat_stream(request: Request):
    data = await request.json()
    # 这个 query 现在是 Java 传过来的“超级 Prompt”，里面已经包含了专家身份和灌水后的模板
    query = data.get("query")
    history_data = data.get("history", [])
    subject = data.get("subject", "default")

    index = index_map.get(subject, index_map.get("default"))

    # 1. 转换历史格式
    chat_history = []
    # 排除掉最后一条（因为最后一条通常就是当前的 query，ChatEngine 会自动处理）
    for msg in history_data[:-1]:
        role = "user" if msg["role"] == "user" else "assistant"
        chat_history.append(ChatMessage(role=role, content=msg["content"]))

    # 2. 【核心改动】极简系统提示词
    # “请严格执行用户 Prompt 中设定的专家角色和批改逻辑。”
    minimal_system_prompt = (
        "你是一个高度专业的 AI 助手。请根据下方提供的【批改标准与身份设定】，"
        "结合参考资料，以该专家的口吻与学生进行深度复盘。"
    )

    # 3. 创建对话引擎
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        system_prompt=minimal_system_prompt
    )

    def generate():
        # 注意：这里的 query 包含了 Java 侧 String.format 后的所有信息
        response = chat_engine.stream_chat(query, chat_history=chat_history)
        for token in response.response_gen:
            # 这里的 \n\n 是 SSE 协议要求的格式
            yield f"data: {token}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000) # host="0.0.0.0"：全网监听。不仅仅本机 127.0.0.1 能访问，局域网里的 Java 后端或其他机器也能通过 IP 找到它。