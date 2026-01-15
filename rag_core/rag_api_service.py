"""
RAG智能问答系统 - FastAPI 微服务版本
基于 LlamaIndex + DeepSeek + 本地 Embedding
提供 HTTP API 接口，支持普通问答、流式输出和文档上传
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import tiktoken
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
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
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from sse_starlette.sse import EventSourceResponse

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

    # 路径配置
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    STORAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage")


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


# ==================== 单例管理器 ====================

class RAGServiceManager:
    """RAG服务管理器 - 单例模式管理索引和引擎"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGServiceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.token_counter = None
            self.index = None
            self.query_engine = None
            self.chat_engines = {}  # 按会话ID存储聊天引擎
            self._initialized = True
    
    def init_settings(self):
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
            timeout=120.0
        )
        
        # 3. 配置本地嵌入模型(用于向量化文本)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=Config.EMBED_MODEL,
            embed_batch_size=40,
            device="cpu"
        )
        Settings.chunk_size = Config.CHUNK_SIZE
        Settings.chunk_overlap = Config.CHUNK_OVERLAP
        
        self.token_counter = token_counter
        print("✅ 系统初始化完成\n")
    
    def get_or_create_index(self):
        """获取或创建向量索引"""
        
        if self.index is not None:
            return self.index
        
        if os.path.exists(Config.STORAGE_PATH):
            print(f"📂 发现已有索引，从 {Config.STORAGE_PATH} 加载...")
            storage_context = StorageContext.from_defaults(persist_dir=Config.STORAGE_PATH)
            index = load_index_from_storage(storage_context)
            print("✅ 索引加载成功")
        else:
            print(f"📚 未找到索引，开始构建新索引...")
            print(f"📖 读取文档：{Config.DATA_PATH}")
            
            # 高速读取
            reader = SimpleDirectoryReader(
                input_dir=Config.DATA_PATH,
                file_extractor={".pdf": PyMuPDFReader()}
            )
            documents = reader.load_data()
            
            print(f"🔨 正在构建索引，开启多核加速...")
            index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True,
                num_workers=4
            )
            
            index.storage_context.persist(persist_dir=Config.STORAGE_PATH)
            print("✅ 索引构建完成")
        
        self.index = index
        return index
    
    def create_query_engine(self):
        """创建配置好的查询引擎"""
        
        if self.query_engine is not None:
            return self.query_engine
        
        print("⚙️  配置查询引擎...")
        
        # 初始化重排序器(提高检索精度)
        reranker = FlagEmbeddingReranker(
            model=Config.RERANKER_MODEL,
            top_n=Config.RERANK_TOP_N
        )
        
        # 创建查询引擎
        query_engine = self.index.as_query_engine(
            similarity_top_k=Config.TOP_K,
            node_postprocessors=[reranker],
            text_qa_template=PromptTemplate(QA_PROMPT_TEMPLATE)
        )
        
        self.query_engine = query_engine
        print("✅ 查询引擎就绪\n")
        return query_engine
    
    def get_chat_engine(self, conversation_id: str = "default"):
        """获取或创建聊天引擎（带记忆功能）"""
        
        if conversation_id not in self.chat_engines:
            print(f"⚙️  为会话 {conversation_id} 配置对话式查询引擎...")
            memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
            
            reranker = FlagEmbeddingReranker(
                model=Config.RERANKER_MODEL,
                top_n=Config.RERANK_TOP_N
            )
            
            chat_engine = self.index.as_chat_engine(
                chat_mode="condense_plus_context",
                streaming=True,
                memory=memory,
                similarity_top_k=3,
                system_prompt="你是一位金融/政务集成架构专家...",
                context_prompt=QA_PROMPT_TEMPLATE
            )
            
            self.chat_engines[conversation_id] = chat_engine
            print(f"✅ 会话 {conversation_id} 的对话引擎就绪\n")
        
        return self.chat_engines[conversation_id]
    
    def reset_token_counter(self):
        """重置Token计数器"""
        if self.token_counter:
            self.token_counter.reset_counts()
    
    def get_token_stats(self) -> Dict[str, Any]:
        """获取Token消耗统计"""
        if not self.token_counter:
            return {}
        
        prompt_tokens = self.token_counter.prompt_llm_token_count
        completion_tokens = self.token_counter.completion_llm_token_count
        total_cost = (
            prompt_tokens * Config.PRICE_INPUT +
            completion_tokens * Config.PRICE_OUTPUT
        )
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "total_cost_usd": total_cost,
            "input_price_per_token": Config.PRICE_INPUT,
            "output_price_per_token": Config.PRICE_OUTPUT
        }
    
    def add_document(self, file_path: str):
        """添加新文档到索引"""
        print(f"📄 正在添加文档: {file_path}")
        
        # 读取新文档
        reader = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor={".pdf": PyMuPDFReader()}
        )
        documents = reader.load_data()
        
        # 将新文档插入索引
        for doc in documents:
            self.index.insert(doc)
        
        # 保存更新后的索引
        self.index.storage_context.persist(persist_dir=Config.STORAGE_PATH)
        print(f"✅ 文档已成功添加到索引并保存")


# ==================== Pydantic 模型 ====================

class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="用户查询问题")
    conversation_id: Optional[str] = Field("default", description="会话ID，用于多轮对话")
    user_id: Optional[str] = Field(None, description="用户ID，用于统计和个性化")

class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str = Field(..., description="AI回答内容")
    conversation_id: str = Field(..., description="会话ID")
    token_stats: Optional[Dict[str, Any]] = Field(None, description="Token消耗统计")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="来源文档信息")

class UploadResponse(BaseModel):
    """上传响应模型"""
    success: bool = Field(..., description="上传是否成功")
    message: str = Field(..., description="响应消息")
    file_path: Optional[str] = Field(None, description="保存的文件路径")
    document_count: Optional[int] = Field(None, description="处理的文档数量")

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    index_loaded: bool = Field(..., description="索引是否加载")
    model_ready: bool = Field(..., description="模型是否就绪")
    storage_path: str = Field(..., description="存储路径")


# ==================== FastAPI 应用 ====================

# 创建单例管理器
rag_manager = RAGServiceManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    print("🚀 RAG API 服务启动中...")
    try:
        rag_manager.init_settings()
        rag_manager.get_or_create_index()
        rag_manager.create_query_engine()
        print("✅ RAG API 服务启动完成")
    except Exception as e:
        print(f"❌ 服务启动失败: {str(e)}")
        raise
    
    yield
    
    # 关闭时清理
    print("🛑 RAG API 服务关闭中...")

# 创建 FastAPI 应用
app = FastAPI(
    title="RAG智能问答API",
    description="基于LlamaIndex和DeepSeek的RAG智能问答系统",
    version="1.0.0",
    lifespan=lifespan
)


# ==================== API 接口 ====================

@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "RAG智能问答API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "普通问答接口",
            "POST /query_stream": "流式输出接口",
            "POST /upload_doc": "上传文档接口",
            "GET /health": "健康检查接口"
        }
    }

@app.get("/health")
async def health_check() -> HealthResponse:
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        index_loaded=rag_manager.index is not None,
        model_ready=rag_manager.query_engine is not None,
        storage_path=Config.STORAGE_PATH
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """普通问答接口"""
    try:
        # 重置Token计数器
        rag_manager.reset_token_counter()
        
        # 获取聊天引擎
        chat_engine = rag_manager.get_chat_engine(request.conversation_id)
        
        print(f"🔍 正在处理查询: {request.query[:50]}...")
        
        # 执行查询
        response = await asyncio.to_thread(
            chat_engine.chat,
            request.query
        )
        
        # 获取Token统计
        token_stats = rag_manager.get_token_stats()
        
        # 提取来源信息
        sources = []
        if hasattr(response, 'source_nodes'):
            for i, node in enumerate(response.source_nodes, 1):
                sources.append({
                    "index": i,
                    "score": float(node.score) if node.score else 0.0,
                    "content_preview": node.node.get_content()[:100].replace('\n', ' ')
                })
        
        return QueryResponse(
            answer=str(response),
            conversation_id=request.conversation_id,
            token_stats=token_stats,
            sources=sources if sources else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@app.post("/query_stream")
async def query_stream(request: QueryRequest):
    """流式输出接口"""
    
    async def event_generator():
        """事件生成器，用于流式输出"""
        try:
            # 获取聊天引擎
            chat_engine = rag_manager.get_chat_engine(request.conversation_id)
            
            print(f"🔍 正在处理流式查询: {request.query[:50]}...")
            
            # 执行流式查询
            streaming_response = chat_engine.stream_chat(request.query)
            
            # 流式输出回答
            for token in streaming_response.response_gen:
                yield {
                    "event": "message",
                    "data": token
                }
                await asyncio.sleep(0.01)  # 小延迟，避免发送过快
            
            # 发送完成事件
            yield {
                "event": "complete",
                "data": "stream_completed"
            }
            
        except Exception as e:
            yield {
                "event": "error",
                "data": f"流式查询失败: {str(e)}"
            }
    
    return EventSourceResponse(event_generator())

@app.post("/upload_doc", response_model=UploadResponse)
async def upload_doc(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """上传文档接口"""
    try:
        # 检查文件类型
        if not file.filename.endswith('.pdf'):
            return UploadResponse(
                success=False,
                message="仅支持PDF文件格式"
            )
        
        # 创建临时保存路径
        upload_dir = os.path.join(Config.DATA_PATH, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        # 保存文件
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # 在后台任务中添加文档到索引
        if background_tasks:
            background_tasks.add_task(rag_manager.add_document, file_path)
            return UploadResponse(
                success=True,
                message="文件已上传，正在后台处理添加到索引",
                file_path=file_path,
                document_count=1
            )
        else:
            # 如果没有后台任务，直接处理
            rag_manager.add_document(file_path)
            return UploadResponse(
                success=True,
                message="文件已上传并成功添加到索引",
                file_path=file_path,
                document_count=1
            )
        
    except Exception as e:
        return UploadResponse(
            success=False,
            message=f"文件上传失败: {str(e)}"
        )


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 启动 RAG API 服务...")
    print(f"📂 数据路径: {Config.DATA_PATH}")
    print(f"💾 存储路径: {Config.STORAGE_PATH}")
    
    uvicorn.run(
        "rag_api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
