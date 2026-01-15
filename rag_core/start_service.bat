@echo off
echo ========================================
echo 🚀 RAG API 服务启动脚本
echo ========================================
echo.

REM 检查是否设置了 DEEPSEEK_API_KEY 环境变量
if "%DEEPSEEK_API_KEY%"=="" (
    echo ❌ 错误: 未设置 DEEPSEEK_API_KEY 环境变量
    echo.
    echo 请先设置环境变量:
    echo set DEEPSEEK_API_KEY=your_api_key_here
    echo.
    echo 或者临时设置:
    echo set DEEPSEEK_API_KEY=your_key && python rag_api_service.py
    echo.
    pause
    exit /b 1
)

echo ✅ 环境变量检查通过
echo 📂 当前目录: %CD%
echo.

REM 检查 requirements.txt 是否存在
if not exist "requirements.txt" (
    echo ❌ 错误: 未找到 requirements.txt 文件
    pause
    exit /b 1
)

echo 🔧 检查 Python 依赖...
python -m pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo ❌ 依赖安装失败
    pause
    exit /b 1
)

echo ✅ 依赖安装完成
echo.

echo 🚀 启动 RAG API 服务...
echo 📍 服务地址: http://localhost:8000
echo 📍 API 文档: http://localhost:8000/docs
echo.
echo ⚠️  按 Ctrl+C 停止服务
echo ========================================
echo.

REM 启动服务
python rag_api_service.py

if %ERRORLEVEL% neq 0 (
    echo ❌ 服务启动失败
    pause
    exit /b 1
)
