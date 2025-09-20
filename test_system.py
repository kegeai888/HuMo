#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuMo系统测试脚本
用于验证app.py的基本功能
"""

import sys
import os
import time
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "humo"))
sys.path.insert(0, str(project_root))

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")

    try:
        import torch
        print(f"✅ PyTorch导入成功 (版本: {torch.__version__})")

        import gradio as gr
        print(f"✅ Gradio导入成功 (版本: {gr.__version__})")

        from app import HuMoGradioApp
        print("✅ HuMoGradioApp导入成功")

        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {str(e)}")
        return False

def test_gpu_detection():
    """测试GPU检测"""
    print("\n🖥️ 测试GPU检测...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ CUDA可用")
            print(f"   设备数量: {torch.cuda.device_count()}")
            print(f"   当前设备: {torch.cuda.get_device_name()}")
            print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            return True
        else:
            print("⚠️ CUDA不可用，将使用CPU模式")
            return True
    except Exception as e:
        print(f"❌ GPU检测失败: {str(e)}")
        return False

def test_app_initialization():
    """测试应用初始化"""
    print("\n🚀 测试应用初始化...")

    try:
        from app import HuMoGradioApp

        app = HuMoGradioApp()
        print("✅ HuMoGradioApp初始化成功")

        # 测试GPU状态获取
        gpu_status = app.get_gpu_status()
        if gpu_status["available"]:
            print(f"✅ GPU状态检测成功: {gpu_status['device_name']}")
        else:
            print(f"⚠️ GPU不可用: {gpu_status['message']}")

        # 测试内存优化
        result = app.optimize_memory()
        print(f"✅ 内存优化测试: {result}")

        # 清理
        app.cleanup()
        print("✅ 应用清理成功")

        return True
    except Exception as e:
        print(f"❌ 应用初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_interface_creation():
    """测试界面创建"""
    print("\n🎨 测试界面创建...")

    try:
        from app import create_interface

        # 创建界面（不启动）
        interface = create_interface()
        print("✅ Gradio界面创建成功")

        return True
    except Exception as e:
        print(f"❌ 界面创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_config_files():
    """测试配置文件"""
    print("\n📁 测试配置文件...")

    config_files = [
        "humo/configs/inference/generate_1_7B.yaml",
        "humo/configs/inference/generate.yaml"
    ]

    all_exist = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ 配置文件存在: {config_file}")
        else:
            print(f"❌ 配置文件缺失: {config_file}")
            all_exist = False

    return all_exist

def test_model_paths():
    """测试模型路径"""
    print("\n🤖 测试模型路径...")

    model_paths = [
        "weights/HuMo/HuMo-1.7B",
        "weights/HuMo/HuMo-17B",
        "weights/Wan2.1-T2V-1.3B"
    ]

    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✅ 模型路径存在: {model_path}")
        else:
            print(f"⚠️ 模型路径不存在: {model_path}")

def test_dependencies():
    """测试依赖包"""
    print("\n📦 测试依赖包...")

    dependencies = [
        "torch",
        "gradio",
        "PIL",
        "numpy",
        "mediapy",
        "pathlib"
    ]

    all_available = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} 可用")
        except ImportError:
            print(f"❌ {dep} 不可用")
            all_available = False

    return all_available

def test_startup_script():
    """测试启动脚本"""
    print("\n📜 测试启动脚本...")

    if os.path.exists("start_app.sh"):
        print("✅ start_app.sh 存在")

        # 检查脚本权限
        if os.access("start_app.sh", os.X_OK):
            print("✅ start_app.sh 有执行权限")
        else:
            print("⚠️ start_app.sh 没有执行权限")

        return True
    else:
        print("❌ start_app.sh 不存在")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🎬 HuMo 系统测试开始")
    print("=" * 50)

    test_results = []

    # 运行各项测试
    test_results.append(("模块导入", test_imports()))
    test_results.append(("GPU检测", test_gpu_detection()))
    test_results.append(("应用初始化", test_app_initialization()))
    test_results.append(("界面创建", test_interface_creation()))
    test_results.append(("配置文件", test_config_files()))
    test_results.append(("依赖包", test_dependencies()))
    test_results.append(("启动脚本", test_startup_script()))

    # 模型路径检查（不影响通过/失败状态）
    test_model_paths()

    # 总结测试结果
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n📈 通过率: {passed}/{total} ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 所有测试通过！系统准备就绪。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关配置。")
        return False

if __name__ == "__main__":
    success = run_all_tests()

    if success:
        print("\n🚀 您可以运行以下命令启动应用:")
        print("   ./start_app.sh")
        print("   或者直接运行: python3 app.py")
    else:
        print("\n🔧 请解决上述问题后重新测试。")

    sys.exit(0 if success else 1)