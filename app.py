#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuMo视频生成器 - 单页面优化版
基于原gradio_app.py的功能，单页面设计，界面更美观直观
"""

import gradio as gr
import os
import json
import tempfile
import shutil
from pathlib import Path
import torch
import sys
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import mediapy
import time
import traceback
import gc

# 添加项目路径
path_to_insert = "humo"
if path_to_insert not in sys.path:
    sys.path.insert(0, path_to_insert)

from common.config import load_config, create_object
from common.distributed import get_device, get_global_rank, init_torch
from common.logger import get_logger
from humo.models.utils.utils import tensor_to_video

# 视频尺寸配置
SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '1024*1024': (1024, 1024),
}

class HuMoGradioApp:
    def __init__(self):
        self.generator = None
        self.config = None
        self.temp_dir = tempfile.mkdtemp()
        self.progress = 0
        self.is_generating = False
        self.model_type = None  # 记录当前模型类型
        self.logger = get_logger(self.__class__.__name__)

        # GPU优化设置
        self._setup_gpu_optimization()

    def _setup_gpu_optimization(self):
        """设置GPU优化"""
        if torch.cuda.is_available():
            # 设置CUDA内存分配策略
            torch.cuda.set_per_process_memory_fraction(0.95)

            # 启用CuDNN基准测试以优化性能
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # 设置CUDA内存分配器
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

            self.logger.info(f"GPU优化已启用，设备: {torch.cuda.get_device_name()}")
            self.logger.info(f"CUDA版本: {torch.version.cuda}")
            self.logger.info(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            self.logger.warning("CUDA不可用，将使用CPU模式")

    def get_gpu_status(self) -> dict:
        """获取GPU状态信息"""
        if not torch.cuda.is_available():
            return {
                "available": False,
                "message": "❌ CUDA不可用",
                "details": "请检查CUDA安装和GPU驱动"
            }

        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)

            total_memory = props.total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            reserved_memory = torch.cuda.memory_reserved(device)
            free_memory = total_memory - allocated_memory

            return {
                "available": True,
                "device_name": props.name,
                "device_count": torch.cuda.device_count(),
                "current_device": device,
                "total_memory_gb": total_memory / 1024**3,
                "allocated_memory_gb": allocated_memory / 1024**3,
                "reserved_memory_gb": reserved_memory / 1024**3,
                "free_memory_gb": free_memory / 1024**3,
                "utilization": (allocated_memory / total_memory) * 100,
                "cuda_version": torch.version.cuda,
                "message": f"✅ GPU可用: {props.name}"
            }
        except Exception as e:
            return {
                "available": False,
                "message": f"❌ GPU状态检查失败: {str(e)}",
                "details": str(e)
            }

    def optimize_memory(self):
        """优化内存使用"""
        try:
            # Python垃圾回收
            gc.collect()

            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 清理临时文件
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                # 只清理旧文件，保留当前会话的文件
                for root, dirs, files in os.walk(self.temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            # 删除超过1小时的临时文件
                            if time.time() - os.path.getctime(file_path) > 3600:
                                os.remove(file_path)
                        except:
                            pass

            self.logger.info("内存优化完成")
            return "✅ 内存优化完成"
        except Exception as e:
            self.logger.error(f"内存优化失败: {str(e)}")
            return f"❌ 内存优化失败: {str(e)}"

    def load_model(self, model_type: str = "1.7B"):
        """加载HuMo模型"""
        try:
            # 先进行内存优化
            self.optimize_memory()

            # 先释放显存
            gc.collect()
            torch.cuda.empty_cache()

            # 根据模型类型设置路径和配置
            if model_type == "1.7B":
                model_path = "./weights/HuMo/HuMo-1.7B"
                config_path = "humo/configs/inference/generate_1_7B.yaml"
                self.model_type = "1.7B"
            elif model_type == "17B":
                model_path = "./weights/HuMo/HuMo-17B"
                config_path = "humo/configs/inference/generate.yaml"
                self.model_type = "17B"
            else:
                return f"❌ 不支持的模型类型: {model_type}"

            if not os.path.exists(model_path):
                return f"❌ 模型路径不存在: {model_path}"

            if not os.path.exists(config_path):
                return f"❌ 配置文件不存在: {config_path}"

            # 检查GPU内存是否足够
            gpu_status = self.get_gpu_status()
            if gpu_status["available"]:
                required_memory = 8.0 if model_type == "1.7B" else 24.0
                if gpu_status["free_memory_gb"] < required_memory:
                    return f"❌ GPU显存不足，需要 {required_memory}GB，可用 {gpu_status['free_memory_gb']:.1f}GB"

            # 加载配置
            self.config = load_config(config_path, [])
            self.config.dit.checkpoint_dir = model_path

            # 在单机模式下禁用序列并行
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_world_size() == 1:
                self.config.dit.sp_size = 1
                self.config.generation.sequence_parallel = 1
                print("🔧 单机模式：已禁用序列并行")

            # 初始化torch
            init_torch(cudnn_benchmark=False)

            # 创建生成器
            self.generator = create_object(self.config)

            # 配置模型组件（包括vae）
            self.generator.configure_models()

            # 再次检查内存状态
            gpu_status = self.get_gpu_status()
            memory_info = ""
            if gpu_status["available"]:
                memory_info = f"\n🖥️ GPU: {gpu_status['device_name']}\n📊 显存使用: {gpu_status['allocated_memory_gb']:.1f}GB / {gpu_status['total_memory_gb']:.1f}GB"

            return f"✅ {model_type}模型加载成功！{memory_info}"
        except Exception as e:
            traceback.print_exc()
            return f"❌ 模型加载失败: {str(e)}"

    def update_progress(self) -> float:
        """更新进度条"""
        if not self.is_generating:
            return 0
        self.progress += 1
        # 模拟进度，最大值为99%，留1%给最终处理
        progress_value = min(99, self.progress)
        return progress_value

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        ref_image: Optional[Image.Image] = None,
        audio_file: Optional[str] = None,
        mode: str = "TA",
        frames: int = 97,
        height: int = 720,
        width: int = 1280,
        scale_i: float = 5.0,
        scale_a: float = 5.5,
        scale_t: float = 5.0,
        sampling_steps: int = 50,
        seed: int = 666666,
        fps: int = 25
    ) -> Tuple[str, str, float]:
        """生成视频"""
        # 记录开始时间
        start_time = time.time()

        try:
            # 重置进度
            self.progress = 0
            self.is_generating = True

            # 详细的输入验证
            if self.generator is None:
                self.is_generating = False
                return "", "❌ 请先加载模型\n💡 提示：在模型配置区域选择模型类型并点击'加载模型'按钮", 0

            if not prompt.strip():
                self.is_generating = False
                return "", "❌ 请输入文本提示词\n💡 提示：详细描述您想要生成的视频内容，例如'一位年轻女性在音乐节上热情地跳舞'", 0

            if mode == "TIA" and ref_image is None:
                self.is_generating = False
                return "", "❌ TIA模式需要提供参考图像\n💡 提示：请上传一张参考图像，或切换到TA模式", 0

            # GPU内存检查
            gpu_status = self.get_gpu_status()
            if gpu_status["available"]:
                if gpu_status["free_memory_gb"] < 4.0:
                    self.is_generating = False
                    return "", f"❌ GPU显存不足 ({gpu_status['free_memory_gb']:.1f}GB可用)\n💡 提示：点击'优化内存'按钮释放显存，或降低分辨率/帧数", 0

            # 参数合理性检查
            if frames > 200:
                return "", "❌ 帧数过多，建议不超过200帧\n💡 提示：过多帧数会显著增加生成时间和显存占用", 0

            if sampling_steps > 100:
                return "", "❌ 采样步数过多，建议不超过100步\n💡 提示：过多采样步数会显著增加生成时间，通常30-50步已足够", 0

            # 准备输出目录
            output_dir = os.path.join(self.temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            self.logger.info("开始创建测试用例文件")
            # 创建临时测试用例文件（使用正确的格式）
            test_case = {
                "case_1": {
                    "img_paths": [],
                    "audio_path": None,
                    "prompt": prompt
                }
            }

            # 处理参考图像
            if ref_image is not None and mode == "TIA":
                try:
                    img_path = os.path.join(self.temp_dir, "ref_image.png")
                    ref_image.save(img_path)
                    test_case["case_1"]["img_paths"] = [img_path]
                    self.logger.info(f"参考图像已保存: {img_path}")
                except Exception as e:
                    self.is_generating = False
                    return "", f"❌ 参考图像处理失败: {str(e)}\n💡 提示：请检查图像格式是否正确", 0

            # 处理音频文件
            if audio_file is not None:
                self.logger.info(f"收到音频文件: {audio_file}")

                try:
                    if os.path.isfile(audio_file):
                        # 检查音频文件大小
                        file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
                        if file_size > 100:  # 100MB限制
                            self.is_generating = False
                            return "", f"❌ 音频文件过大 ({file_size:.1f}MB)\n💡 提示：请使用小于100MB的音频文件", 0

                        audio_path = os.path.join(self.temp_dir, "audio.wav")
                        shutil.copy2(audio_file, audio_path)
                        test_case["case_1"]["audio_path"] = audio_path
                        self.logger.info(f"音频文件已复制到: {audio_path}")
                    else:
                        self.logger.warning(f"音频文件路径无效: {audio_file}")
                        test_case["case_1"]["audio_path"] = None
                except Exception as e:
                    self.logger.warning(f"音频文件处理失败: {str(e)}")
                    test_case["case_1"]["audio_path"] = None

            # 保存测试用例
            test_case_path = os.path.join(self.temp_dir, "test_case.json")
            with open(test_case_path, 'w', encoding='utf-8') as f:
                json.dump(test_case, f, ensure_ascii=False, indent=2)

            self.logger.info(f"测试用例文件已创建: {test_case_path}")

            # 使用统一的配置更新方法
            config_updates = {
                'mode': mode,
                'frames': frames,
                'height': height,
                'width': width,
                'scale_i': scale_i,
                'scale_a': scale_a,
                'scale_t': scale_t,
                'seed': seed,
                'fps': fps,
                'positive_prompt': test_case_path,
                'output': {'dir': output_dir}
            }

            # 更新反向提示词配置
            if negative_prompt.strip():
                config_updates['sample_neg_prompt'] = negative_prompt

            # 更新generation配置
            self.generator.update_generation_config(**config_updates)

            # 更新其他配置（如diffusion配置）
            self.generator.update_config(
                diffusion_timesteps_sampling_steps=sampling_steps
            )

            self.logger.info("配置更新完成，开始生成视频...")

            # 记录生成开始时间
            generation_start_time = time.time()

            # 生成视频 - 使用inference_loop方法
            self.generator.inference_loop()

            # 记录生成结束时间并计算生成耗时
            generation_end_time = time.time()
            generation_duration = generation_end_time - generation_start_time

            self.is_generating = False
            self.progress = 100  # 设置为100%完成

            # inference_loop方法会保存视频文件，查找生成的文件
            output_files = list(Path(output_dir).glob("*.mp4"))

            self.logger.info(f"查找输出目录: {output_dir}")
            self.logger.info(f"找到的视频文件: {output_files}")

            if output_files:
                video_path = str(output_files[0])
                # 将相对路径转换为绝对路径
                video_path = os.path.abspath(video_path)

                # 计算总耗时
                total_duration = time.time() - start_time

                self.logger.info(f"生成完成，视频路径: {video_path}")

                if os.path.exists(video_path):
                    # 获取视频文件大小
                    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB

                    success_message = (
                        f"✅ 视频生成成功！\n"
                        f"⏱️ 生成耗时: {generation_duration:.1f}秒\n"
                        f"📊 总耗时: {total_duration:.1f}秒\n"
                        f"🎬 视频大小: {file_size:.1f}MB\n"
                        f"📐 分辨率: {width}x{height}\n"
                        f"🎞️ 帧数: {frames} | 帧率: {fps}FPS\n"
                        f"🤖 模型: {self.model_type}"
                    )
                    return video_path, success_message, 100
                else:
                    self.logger.warning(f"视频文件不存在: {video_path}")
                    return "", "❌ 视频文件生成后丢失\n💡 提示：可能是磁盘空间不足或权限问题", 0
            else:
                self.logger.warning(f"未找到视频文件，输出目录内容: {list(Path(output_dir).iterdir())}")
                return "", "❌ 未找到生成的视频文件\n💡 提示：生成可能未完成或保存失败，请重试", 0

        except torch.cuda.OutOfMemoryError as e:
            total_duration = time.time() - start_time
            self.logger.error(f"GPU显存不足，总耗时: {total_duration:.1f}秒")
            self.is_generating = False
            return "", "❌ GPU显存不足\n💡 解决方案：\n1. 点击'优化内存'按钮\n2. 降低分辨率或帧数\n3. 使用1.7B模型而非17B模型", 0

        except FileNotFoundError as e:
            total_duration = time.time() - start_time
            self.logger.error(f"文件未找到，总耗时: {total_duration:.1f}秒")
            self.is_generating = False
            return "", f"❌ 文件未找到: {str(e)}\n💡 提示：请检查模型文件是否正确下载", 0

        except PermissionError as e:
            total_duration = time.time() - start_time
            self.logger.error(f"权限错误，总耗时: {total_duration:.1f}秒")
            self.is_generating = False
            return "", f"❌ 权限错误\n💡 提示：请检查文件夹权限或使用管理员权限运行", 0

        except Exception as e:
            # 计算总耗时（即使失败也记录）
            total_duration = time.time() - start_time
            self.logger.error(f"视频生成失败，总耗时: {total_duration:.1f}秒")
            traceback.print_exc()
            self.is_generating = False

            # 提供更具体的错误信息
            error_message = f"❌ 视频生成失败: {str(e)}\n💡 常见解决方案：\n"
            if "CUDA" in str(e):
                error_message += "• 重启应用并重新加载模型\n• 检查CUDA驱动和PyTorch版本"
            elif "memory" in str(e).lower():
                error_message += "• 降低分辨率、帧数或采样步数\n• 点击'优化内存'按钮"
            elif "model" in str(e).lower():
                error_message += "• 重新下载模型文件\n• 检查模型文件完整性"
            else:
                error_message += "• 检查输入参数是否合理\n• 重启应用重试"

            return "", error_message, 0

    def cleanup(self):
        """清理临时文件"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

# 创建应用实例
app = HuMoGradioApp()

def create_interface():
    """创建单页面Gradio界面"""

    # 精美的CSS样式
    custom_css = """
    .gradio-container {
        font-family: 'Inter', 'Microsoft YaHei', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    .main-container {
        background: white;
        border-radius: 20px;
        padding: 25px;
        margin: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }

    .header-section {
        text-align: center;
        padding: 30px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        color: white;
    }

    .section-title {
        font-size: 1.3em;
        font-weight: 600;
        color: #374151;
        margin: 20px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e5e7eb;
    }

    .gr-button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        font-size: 1em;
    }

    .gr-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .gr-form {
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        background: #fafafa;
    }

    .parameter-box {
        background: #f8fafc;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        margin: 10px 0;
    }

    .status-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 12px;
        padding: 15px;
        border-left: 4px solid #0ea5e9;
        margin: 10px 0;
    }

    .video-output {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .example-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
        margin: 15px 0;
    }

    .example-item {
        background: #f1f5f9;
        padding: 12px;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid #e2e8f0;
    }

    .example-item:hover {
        background: #e2e8f0;
        transform: scale(1.02);
    }
    """

    with gr.Blocks(
        title="HuMo 视频生成器",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="gray",
            font=gr.themes.GoogleFont("Inter")
        ),
        css=custom_css
    ) as interface:

        # 主容器
        with gr.Column(elem_classes="main-container"):

            # 精美的头部
            gr.HTML("""
            <div class="header-section">
                <h1 style="font-size: 3.5em; margin-bottom: 15px; font-weight: 700;">🎬 HuMo 视频生成器</h1>
                <p style="font-size: 1.3em; opacity: 0.95; margin: 5px 0;">基于协作多模态条件的人体中心视频生成</p>
                <p style="font-size: 1.1em; opacity: 0.85; margin-top: 15px;">✨ 支持文本、图像、音频多种输入模式的智能视频生成</p>
                <p style="font-size: 1.1em; opacity: 0.85; margin-top: 15px; color: white;">
    ✨ 二次开发构建 By：科哥 
    <a href="https://github.com/kegeai888/HuMo" style="color: white;">欢迎 start</a>
</p>

            </div>
            """)

            # 模型配置区域
            gr.HTML('<div class="section-title">🤖 模型配置</div>')
            with gr.Row():
                with gr.Column(scale=3):
                    model_type = gr.Dropdown(
                        choices=["1.7B", "17B"],
                        value="1.7B",
                        label="选择模型类型",
                        info="💡 1.7B: 轻量级，速度快，显存需求低 | 17B: 高质量，效果佳，显存需求高",
                        elem_classes="parameter-box"
                    )
                with gr.Column(scale=1):
                    load_btn = gr.Button(
                        "🔄 加载模型",
                        variant="primary",
                        size="lg"
                    )

            model_status = gr.Textbox(
                label="📊 模型状态",
                interactive=False,
                elem_classes="status-box"
            )

            # GPU状态监控
            with gr.Row():
                with gr.Column(scale=2):
                    gpu_status_display = gr.Textbox(
                        label="🖥️ GPU状态",
                        interactive=False,
                        lines=2,
                        elem_classes="status-box"
                    )
                with gr.Column(scale=1):
                    with gr.Row():
                        gpu_info_btn = gr.Button("🔍 检查GPU", size="sm")
                        memory_optimize_btn = gr.Button("🧹 优化内存", size="sm")

            # 主要输入区域
            gr.HTML('<div class="section-title">✨ 内容创作</div>')
            with gr.Row():
                with gr.Column(scale=2):
                    # 文本输入
                    prompt = gr.Textbox(
                        label="🎯 文本提示词 (必填)",
                        placeholder="详细描述你想要生成的视频内容，例如：一位年轻女性在音乐节上热情地跳舞，伴随着节拍挥舞双手...",
                        lines=4,
                        max_lines=8
                    )

                    negative_prompt = gr.Textbox(
                        label="🚫 负面提示词 (可选)",
                        placeholder="描述你不想要的内容...",
                        lines=2,
                        value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
                    )

                    # 生成模式
                    mode = gr.Radio(
                        choices=[("文本+音频", "TA"), ("文本+图像+音频", "TIA")],
                        value="TA",
                        label="🎭 生成模式",
                        info="选择输入模式：TA模式无需参考图像，TIA模式可控制人物外观"
                    )

                with gr.Column(scale=1):
                    # 文件上传
                    ref_image = gr.Image(
                        label="🖼️ 参考图像 (TIA模式需要)",
                        type="pil",
                        visible=False
                    )

                    audio_file = gr.Audio(
                        label="🎵 音频文件 (可选)",
                        type="filepath"
                    )

                    # 示例提示
                    gr.Markdown("### 💡 示例提示")
                    example_prompts = gr.Dataset(
                        components=[prompt],
                        samples=[
                            ["一位年轻女性在音乐节上热情地跳舞，伴随着节拍挥舞双手，脸上洋溢着快乐的笑容"],
                            ["一个男人在办公室里自信地进行商务演讲，手势自然流畅，表情专业认真"],
                            ["一位音乐家在昏暗的房间里充满激情地弹奏吉他，眼神专注，完全沉浸在音乐中"],
                            ["一个孩子在花园里快乐地奔跑玩耍，阳光洒在脸上，动作活泼自然"]
                        ],
                        label="点击加载示例",
                        elem_classes="example-grid"
                    )

            # 参数配置区域
            gr.HTML('<div class="section-title">⚙️ 参数配置</div>')
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        gr.Markdown("#### 📐 视频参数")
                        with gr.Row():
                            frames = gr.Slider(
                                minimum=-1, maximum=200, value=-1, step=1,
                                label="视频帧数",
                                info="-1 表示根据音频长度自动计算"
                            )
                            fps = gr.Slider(
                                minimum=15, maximum=60, value=25, step=1,
                                label="帧率 (FPS)"
                            )

                        resolution = gr.Dropdown(
                            choices=[
                                "720P 竖屏 (720x1280)",
                                "720P 横屏 (1280x720)",
                                "480P 竖屏 (480x832)",
                                "480P 横屏 (832x480)",
                                "1024P 方形 (1024x1024)"
                            ],
                            value="720P 竖屏 (720x1280)",
                            label="视频分辨率",
                            info="720P 质量更好，480P 速度更快"
                        )

                with gr.Column():
                    with gr.Group():
                        gr.Markdown("#### 🎛️ 生成参数")
                        with gr.Row():
                            scale_t = gr.Slider(
                                minimum=1.0, maximum=15.0, value=5.0, step=0.1,
                                label="文本引导强度",
                                info="控制对文本描述的遵循程度"
                            )
                            scale_a = gr.Slider(
                                minimum=1.0, maximum=15.0, value=5.5, step=0.1,
                                label="音频引导强度",
                                info="控制动作与音频的同步程度"
                            )
                            scale_i = gr.Slider(
                                minimum=1.0, maximum=15.0, value=5.0, step=0.1,
                                label="图像引导强度",
                                info="控制对参考图像的相似程度"
                            )

                        with gr.Row():
                            sampling_steps = gr.Slider(
                                minimum=20, maximum=100, value=50, step=5,
                                label="采样步数",
                                info="更多步数质量更好但时间更长"
                            )
                            seed = gr.Number(
                                label="随机种子",
                                value=-1,
                                precision=0,
                                info="-1 为随机种子"
                            )

            # 生成按钮
            generate_btn = gr.Button(
                "🎬 开始生成视频",
                variant="primary",
                size="lg",
                elem_id="generate-btn"
            )

            # 输出区域
            gr.HTML('<div class="section-title">🎥 生成结果</div>')
            with gr.Row():
                with gr.Column():
                    output_video = gr.Video(
                        label="生成的视频",
                        height=450,
                        elem_classes="video-output"
                    )
                    status_text = gr.Textbox(
                        label="📊 生成状态",
                        interactive=False,
                        lines=3,
                        elem_classes="status-box"
                    )

            # 使用帮助
            with gr.Accordion("💡 使用指南", open=False):
                gr.Markdown("""
                ## 🚀 快速开始

                1. **加载模型**: 选择1.7B(快速)或17B(高质量)模型，点击"加载模型"
                2. **选择模式**: TA模式无需参考图像，TIA模式可控制人物外观
                3. **输入描述**: 详细描述想要生成的视频内容
                4. **上传文件**: 根据需要上传音频文件和参考图像
                5. **调整参数**: 根据需要调整视频和生成参数
                6. **开始生成**: 点击生成按钮，等待视频完成

                ## 💡 优化建议

                **文本提示技巧**:
                - 使用具体、详细的描述
                - 包含动作、表情、环境等要素
                - 示例："一位年轻女性在音乐节上热情地跳舞，伴随着节拍挥舞双手，脸上洋溢着快乐的笑容"

                **参数调整**:
                - 文本引导强度: 控制对文本描述的遵循度
                - 音频引导强度: 控制动作与音频的同步度
                - 采样步数: 30-50步通常足够，更多步数质量更好但更慢

                **性能优化**:
                - 1.7B模型适合快速测试
                - 720P质量更好，480P速度更快
                - 调整帧数控制视频长度和生成时间
                """)

        # 事件处理
        def load_model_wrapper(model_type):
            return app.load_model(model_type)

        def get_gpu_status_display():
            """获取GPU状态显示"""
            gpu_status = app.get_gpu_status()
            if gpu_status["available"]:
                return (f"GPU: {gpu_status['device_name']}\n"
                       f"显存: {gpu_status['allocated_memory_gb']:.1f}GB / {gpu_status['total_memory_gb']:.1f}GB "
                       f"({gpu_status['utilization']:.1f}% 使用率)\n"
                       f"可用: {gpu_status['free_memory_gb']:.1f}GB")
            else:
                return gpu_status["message"]

        def optimize_memory_wrapper():
            """内存优化包装函数"""
            result = app.optimize_memory()
            # 同时返回优化结果和更新的GPU状态
            gpu_status = get_gpu_status_display()
            return result, gpu_status

        def generate_video_wrapper(prompt, negative_prompt, ref_image, audio_file, mode, frames, resolution,
                                 scale_i, scale_a, scale_t, sampling_steps, seed, fps, progress=gr.Progress()):
            progress(0, desc="准备生成...")

            # 解析分辨率
            if "720P 横屏" in resolution:
                height, width = 720, 1280
            elif "720P 竖屏" in resolution:
                height, width = 1280, 720
            elif "480P 横屏" in resolution:
                height, width = 480, 832
            elif "480P 竖屏" in resolution:
                height, width = 832, 480
            elif "1024P 方形" in resolution:
                height, width = 1024, 1024
            else:
                height, width = 720, 1280  # 默认值

            # 创建一个生成器函数来更新进度
            def progress_tracker():
                while app.is_generating:
                    progress_value = app.update_progress()
                    progress(progress_value/100)
                    time.sleep(0.5)
                    yield progress_value

            # 启动进度更新
            progress_gen = progress_tracker()

            # 生成视频
            video_path, status, final_progress = app.generate_video(
                prompt, negative_prompt, ref_image, audio_file, mode, frames, height, width,
                scale_i, scale_a, scale_t, sampling_steps, seed, fps
            )

            # 设置最终进度
            progress(final_progress/100)

            return video_path, status

        # 模式切换时显示/隐藏参考图像输入
        def update_ref_image_visibility(mode):
            return gr.update(visible=(mode == "TIA"))

        # 处理示例提示选择
        def load_example_prompt(evt: gr.SelectData):
            return evt.value[0]  # 返回选中的示例文本

        # 绑定事件
        load_btn.click(
            fn=load_model_wrapper,
            inputs=[model_type],
            outputs=[model_status]
        )

        # GPU状态检查
        gpu_info_btn.click(
            fn=get_gpu_status_display,
            outputs=[gpu_status_display]
        )

        # 内存优化
        memory_optimize_btn.click(
            fn=optimize_memory_wrapper,
            outputs=[model_status, gpu_status_display]
        )

        mode.change(
            fn=update_ref_image_visibility,
            inputs=[mode],
            outputs=[ref_image]
        )

        example_prompts.select(
            fn=load_example_prompt,
            inputs=[],
            outputs=[prompt]
        )

        generate_btn.click(
            fn=generate_video_wrapper,
            inputs=[prompt, negative_prompt, ref_image, audio_file, mode, frames, resolution,
                   scale_i, scale_a, scale_t, sampling_steps, seed, fps],
            outputs=[output_video, status_text]
        )

    return interface

if __name__ == "__main__":
    # 创建界面
    interface = create_interface()

    # 启动应用
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )
