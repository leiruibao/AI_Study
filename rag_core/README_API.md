# RAG 智能问答 API 服务

基于 FastAPI 的 RAG（检索增强生成）智能问答微服务，将原有的 `rag_demo_claude.py` 脚本改造为可通过 HTTP 访问的 AI 微服务。

## 🚀 功能特性

- **单例模式管理**: 索引和模型只在服务启动时加载一次
- **异步支持**: 基于 FastAPI 的 async/await，完美支持 AI 长耗时操作
- **多接口支持**:
  - `POST /query`: 普通问答接口
  - `POST /query_stream`: 流式输出接口（SSE）
  - `POST /upload_doc`: 文档上传接口，动态更新索引
- **保持原有逻辑**: 完整保留了原有的重排序和记忆逻辑
- **成本追踪**: 实时统计 Token 消耗和费用

## 📁 文件结构

```
AI_Study/rag_core/
├── rag_api_service.py      # FastAPI 主服务文件
├── rag_demo_claude.py      # 原始 CLI 版本（保留）
├── requirements.txt        # Python 依赖
├── start_service.bat      # Windows 启动脚本
├── test_api.py           # API 测试脚本
└── README_API.md         # 本文档
```

## 🔧 快速开始

### 1. 环境准备

```bash
# 设置 DeepSeek API Key（必需）
set DEEPSEEK_API_KEY=your_api_key_here

# 或永久设置（Windows）
setx DEEPSEEK_API_KEY "your_api_key_here"
```

### 2. 安装依赖

```bash
cd AI_Study/rag_core
pip install -r requirements.txt
```

### 3. 启动服务

**方法一：使用启动脚本（推荐）**
```bash
start_service.bat
```

**方法二：手动启动**
```bash
python rag_api_service.py
```

服务将在 `http://localhost:8000` 启动。

## 📚 API 接口文档

### 根路径
- **GET** `/`
- 返回服务信息和可用接口

### 健康检查
- **GET** `/health`
- 返回服务状态、索引加载情况等

### 普通问答接口
- **POST** `/query`
- **请求体**:
```json
{
  "query": "你的问题",
  "conversation_id": "session_123",
  "user_id": "user_001"
}
```
- **响应**:
```json
{
  "answer": "AI回答内容",
  "conversation_id": "session_123",
  "token_stats": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_cost_usd": 0.000028
  },
  "sources": [...]
}
```

### 流式输出接口
- **POST** `/query_stream`
- 使用 Server-Sent Events (SSE) 流式返回回答
- 适合需要实时显示的场景

### 文档上传接口
- **POST** `/upload_doc`
- **Content-Type**: `multipart/form-data`
- **参数**: `file` (PDF 文件)
- 支持动态添加文档到索引

## 🧪 测试服务

### 运行测试脚本
```bash
python test_api.py
```

测试脚本会验证所有接口的功能。

### 手动测试

1. 访问 API 文档: http://localhost:8000/docs
2. 使用 curl 测试:
```bash
# 测试普通问答
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是系统架构？", "conversation_id": "test"}'

# 测试流式输出
curl -X POST "http://localhost:8000/query_stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "请介绍系统架构", "conversation_id": "test"}' \
  -N
```

## ⚙️ 配置说明

所有配置参数都在 `Config` 类中集中管理：

```python
class Config:
    # DeepSeek API 配置
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL = "deepseek-chat"
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    
    # 本地嵌入模型
    EMBED_MODEL = "BAAI/bge-base-zh-v1.5"
    RERANKER_MODEL = "BAAI/bge-reranker-base"
    
    # 文本分块参数
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    # 检索参数
    TOP_K = 10
    RERANK_TOP_N = 5
    
    # 路径配置
    DATA_PATH = "../data"      # 文档存储路径
    STORAGE_PATH = "../storage" # 索引存储路径
```

## 🔄 与原版对比

| 特性 | 原版 (rag_demo_claude.py) | API 版 (rag_api_service.py) |
|------|--------------------------|----------------------------|
| 交互方式 | 命令行交互 | HTTP API |
| 部署方式 | 本地运行 | 可部署为微服务 |
| 并发支持 | 单用户 | 多用户并发 |
| 扩展性 | 有限 | 易于集成到其他系统 |
| 文档管理 | 静态 | 支持动态上传更新 |
| 输出方式 | 同步输出 | 支持流式输出 |

## 🐛 常见问题

### Q1: 服务启动失败，提示缺少 API Key
**A**: 确保已设置 `DEEPSEEK_API_KEY` 环境变量。

### Q2: 索引构建失败
**A**: 检查 `data` 目录下是否有文档文件（PDF格式）。

### Q3: 流式接口不工作
**A**: 确保客户端支持 SSE（Server-Sent Events），或使用 `/query` 普通接口。

### Q4: 上传文档后索引未更新
**A**: 上传后需要重新加载索引，服务会自动处理。

## 📈 性能优化建议

1. **GPU 加速**: 如有 GPU，可将嵌入模型设置为 `device="cuda"`
2. **批量处理**: 上传多个文档时，建议批量处理
3. **缓存策略**: 频繁查询的问题可加入缓存
4. **索引优化**: 定期清理和优化向量索引

## 📄 许可证

本项目基于原有 RAG 系统改造，遵循原有许可证。

---

**🎯 改造完成**: 已成功将 CLI 工具转换为生产可用的微服务架构！
