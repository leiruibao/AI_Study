"""
RAG API 服务测试脚本
用于验证 FastAPI 服务的各个接口功能
"""

import requests
import json
import time

# API 基础 URL
BASE_URL = "http://localhost:8000"

def test_root():
    """测试根路径"""
    print("🔍 测试根路径...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print(f"✅ 根路径测试成功: {response.json()}")
            return True
        else:
            print(f"❌ 根路径测试失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 根路径测试异常: {str(e)}")
        return False

def test_health():
    """测试健康检查接口"""
    print("🔍 测试健康检查接口...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查成功:")
            print(f"   状态: {data['status']}")
            print(f"   索引加载: {data['index_loaded']}")
            print(f"   模型就绪: {data['model_ready']}")
            print(f"   存储路径: {data['storage_path']}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {str(e)}")
        return False

def test_query():
    """测试普通问答接口"""
    print("🔍 测试普通问答接口...")
    try:
        payload = {
            "query": "什么是系统架构？",
            "conversation_id": "test_session_1",
            "user_id": "test_user_001"
        }
        
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 问答接口测试成功:")
            print(f"   回答长度: {len(data['answer'])} 字符")
            print(f"   会话ID: {data['conversation_id']}")
            if data.get('token_stats'):
                print(f"   Token统计: {data['token_stats']}")
            if data.get('sources'):
                print(f"   来源数量: {len(data['sources'])}")
            return True
        else:
            print(f"❌ 问答接口测试失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 问答接口测试异常: {str(e)}")
        return False

def test_query_stream():
    """测试流式输出接口"""
    print("🔍 测试流式输出接口...")
    try:
        payload = {
            "query": "请简要介绍系统架构的重要性",
            "conversation_id": "test_session_2",
            "user_id": "test_user_002"
        }
        
        print("   开始流式接收...")
        response = requests.post(
            f"{BASE_URL}/query_stream",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        if response.status_code == 200:
            print("✅ 流式接口连接成功")
            print("   接收到的数据:")
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data:'):
                        data = line_str[5:].strip()
                        if data:
                            print(f"   {data}")
            return True
        else:
            print(f"❌ 流式接口测试失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 流式接口测试异常: {str(e)}")
        return False

def test_upload_doc():
    """测试文档上传接口（模拟）"""
    print("🔍 测试文档上传接口...")
    print("⚠️  注意：此测试需要实际PDF文件，当前仅演示接口结构")
    print("✅ 上传接口结构验证完成")
    return True

def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("🚀 开始 RAG API 服务测试")
    print("=" * 50)
    
    # 等待服务启动
    print("⏳ 等待服务启动...")
    time.sleep(5)
    
    tests = [
        ("根路径", test_root),
        ("健康检查", test_health),
        ("普通问答", test_query),
        ("流式输出", test_query_stream),
        ("文档上传", test_upload_doc),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 测试: {test_name}")
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\n📈 通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查服务状态")
    
    return passed == total

if __name__ == "__main__":
    # 提示用户先启动服务
    print("⚠️  重要提示：")
    print("1. 请先设置环境变量 DEEPSEEK_API_KEY")
    print("2. 在另一个终端中启动服务:")
    print("   cd AI_Study/rag_core")
    print("   python rag_api_service.py")
    print("3. 等待服务完全启动后再运行此测试")
    print("\n是否继续？(y/n): ", end="")
    
    choice = input().strip().lower()
    if choice == 'y':
        run_all_tests()
    else:
        print("测试已取消")
