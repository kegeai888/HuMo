#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import time
import traceback

# 添加项目路径
path_to_insert = "humo"
if path_to_insert not in sys.path:
    sys.path.insert(0, path_to_insert)

from common.config import load_config, create_object
from common.distributed import get_device, get_global_rank, init_torch

class HuMoGradioApp:
    def __init__(self):
        self.generator = None
        self.config = None
        self.temp_dir = tempfile.mkdtemp()
        self.progress = 0
        self.is_generating = False
        self.progress = 0
        self.is_generating = False
        
    def load_model(self, model_path: str = "./weights/HuMo/HuMo-1.7B"):
        """加载HuMo模型"""
        try:
            if not os.path.exists(model_path):
                return f"❌ 模型路径不存在: {model_path}"
            
            # 初始化配置 generate.yaml是17B
            config_path = "humo/configs/inference/generate_1_7B.yaml"
            if not os.path.exists(config_path):
                return f"❌ 配置文件不存在: {config_path}"
            
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
            
            return "✅ 模型加载成功！"
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
        ref_image: Optional[Image.Image] = None,
        audio_file: Optional[str] = None,
        mode: str = "TA",
        frames: int = 97,
        height: int = 720,
        width: int = 1280,
        scale_a: float = 5.5,
        scale_t: float = 5.0,
        sampling_steps: int = 50,
        seed: int = 666666,
        fps: int = 25
    ) -> Tuple[str, str, float]:
        """生成视频"""
        try:
            # 重置进度
            self.progress = 0
            self.is_generating = True
            
            if self.generator is None:
                self.is_generating = False
                return None, "❌ 请先加载模型", 0
            
            if not prompt.strip():
                self.is_generating = False
                return None, "❌ 请输入文本提示", 0
            
            # 准备输出目录
            output_dir = os.path.join(self.temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建临时测试用例文件
            test_case = {
                "case_1": {
                    "text": prompt,
                    "ref_img": None,
                    "audio": None,
                    "itemname": "gradio_generated"
                }
            }
            
            # 处理参考图像
            if ref_image is not None and mode == "TIA":
                img_path = os.path.join(self.temp_dir, "ref_image.png")
                ref_image.save(img_path)
                test_case["case_1"]["ref_img"] = img_path
            elif mode == "TIA" and ref_image is None:
                self.is_generating = False
                return None, "❌ TIA模式需要提供参考图像", 0
            
            # 处理音频文件
            if audio_file is not None:
                audio_path = os.path.join(self.temp_dir, "audio.wav")
                shutil.copy2(audio_file, audio_path)
                test_case["case_1"]["audio"] = audio_path
            
            # 保存测试用例
            test_case_path = os.path.join(self.temp_dir, "test_case.json")
            with open(test_case_path, 'w', encoding='utf-8') as f:
                json.dump(test_case, f, ensure_ascii=False, indent=2)
            
            # 更新配置
            self.config.generation.mode = mode
            self.config.generation.frames = frames
            self.config.generation.height = height
            self.config.generation.width = width
            self.config.generation.scale_a = scale_a
            self.config.generation.scale_t = scale_t
            self.config.diffusion.timesteps.sampling.steps = sampling_steps
            self.config.generation.seed = seed
            self.config.generation.fps = fps
            self.config.generation.positive_prompt = test_case_path
            self.config.generation.output.dir = output_dir
            
            # 生成视频
            self.generator.inference_loop()
            
            # 查找生成的视频文件
            output_files = list(Path(output_dir).glob("*.mp4"))
            self.is_generating = False
            self.progress = 100  # 设置为100%完成
            
            if output_files:
                video_path = str(output_files[0])
                return video_path, "✅ 视频生成成功！", 100
            else:
                return None, "❌ 未找到生成的视频文件", 0
                
        except Exception as e:
            traceback.print_exc()
            self.is_generating = False
            return None, f"❌ 视频生成失败: {str(e)}", 0
    
    def cleanup(self):
        """清理临时文件"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

# 创建应用实例
app = HuMoGradioApp()

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="HuMo 视频生成器", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎬 HuMo 多模态视频生成器
        
        基于文本、图像和音频输入生成高质量的人类视频
        """)
        
        # 模型配置部分 - 简化布局
        with gr.Row():
            with gr.Column(scale=4):
                model_path = gr.Textbox(
                    label="模型路径",
                    # 17B带不动
                    value="./weights/HuMo/HuMo-1.7B",
                    placeholder="请输入HuMo模型路径",
                    show_label=True
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("🔄 加载模型", variant="primary")
        
        model_status = gr.Textbox(label="模型状态", interactive=False)
        
        # 主要输入部分 - 使用标签页区分基本设置和高级参数
        with gr.Tabs() as tabs:
            # 基本设置标签页 - 只显示核心参数
            with gr.TabItem("基本设置"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # 核心参数：文本提示
                        prompt = gr.Textbox(
                            label="文本提示 (必填)",
                            placeholder="描述你想要生成的视频内容...",
                            lines=3
                        )
                        
                        # 核心参数：生成模式
                        mode = gr.Radio(
                            choices=["TA", "TIA"],
                            value="TA",  # 默认使用TA模式，不需要参考图像
                            label="生成模式",
                            info="TA: 文本+音频, TIA: 文本+图像+音频"
                        )
                        
                        # 核心参数：参考图像和音频
                        with gr.Row():
                            with gr.Column(scale=1):
                                ref_image = gr.Image(
                                    label="参考图像 (TIA模式需要)",
                                    type="pil",
                                    visible=False  # 默认隐藏，只在TIA模式下显示
                                )
                            
                            with gr.Column(scale=1):
                                audio_file = gr.Audio(
                                    label="音频文件 (可选)",
                                    type="filepath"
                                )
                    
                    # 示例和生成按钮
                    with gr.Column(scale=1):
                        # 添加一些示例提示
                        gr.Markdown("### 示例提示")
                        example_prompts = gr.Dataset(
                            components=[prompt],
                            samples=[
                                ["A person dancing energetically to upbeat music"],
                                ["Someone speaking confidently in a business meeting"],
                                ["A person playing guitar with passion"]
                            ],
                            label="点击加载示例提示"
                        )
                        
                        # 生成按钮
                        generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
            
            # 高级参数标签页 - 包含所有可调整的参数
            with gr.TabItem("高级参数"):
                with gr.Row():
                    with gr.Column():
                        # 视频参数折叠面板
                        with gr.Accordion("视频参数", open=False):
                            with gr.Row():
                                frames = gr.Slider(
                                    minimum=25, maximum=97, value=97, step=4,
                                    label="视频帧数",
                                    info="帧数越多视频越长，但生成时间也越长"
                                )
                                fps = gr.Slider(
                                    minimum=15, maximum=30, value=25, step=1,
                                    label="帧率 (FPS)",
                                    info="每秒显示的帧数，影响视频流畅度"
                                )
                            
                            with gr.Row():
                                height = gr.Slider(
                                    minimum=480, maximum=720, value=720, step=8,
                                    label="视频高度",
                                    info="720p质量更好，480p速度更快"
                                )
                                width = gr.Slider(
                                    minimum=832, maximum=1280, value=1280, step=8,
                                    label="视频宽度",
                                    info="与高度对应的宽度设置"
                                )
                        
                        # 生成参数折叠面板
                        with gr.Accordion("生成参数", open=False):
                            with gr.Row():
                                scale_a = gr.Slider(
                                    minimum=1.0, maximum=10.0, value=5.5, step=0.1,
                                    label="音频引导强度", 
                                    info="越高音频同步越好，推荐值：5.0-6.0"
                                )
                                scale_t = gr.Slider(
                                    minimum=1.0, maximum=10.0, value=5.0, step=0.1,
                                    label="文本引导强度", 
                                    info="越高越遵循文本提示，推荐值：4.0-6.0"
                                )
                            
                            with gr.Row():
                                sampling_steps = gr.Slider(
                                    minimum=20, maximum=100, value=50, step=5,
                                    label="采样步数", 
                                    info="更多步数质量更好但更慢，推荐值：30-50"
                                )
                                seed = gr.Number(
                                    label="随机种子",
                                    value=-1,
                                    precision=0,
                                    info="-1为随机种子，固定种子可重现结果"
                                )
        
        # 进度显示
        progress_bar = gr.Progress()
        
        # 输出部分
        with gr.Row():
            with gr.Column():
                output_video = gr.Video(
                    label="生成的视频",
                    height=400
                )
                status_text = gr.Textbox(
                    label="生成状态",
                    interactive=False,
                    lines=2
                )
        
        # 事件处理
        def load_model_wrapper(model_path):
            return app.load_model(model_path)
        
        def generate_video_wrapper(prompt, ref_image, audio_file, mode, frames, height, width, 
                                 scale_a, scale_t, sampling_steps, seed, fps, progress=gr.Progress()):
            progress(0, desc="准备生成...")
            
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
                prompt, ref_image, audio_file, mode, frames, height, width,
                scale_a, scale_t, sampling_steps, seed, fps
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
            inputs=[model_path],
            outputs=[model_status]
        )
        
        mode.change(
            fn=update_ref_image_visibility,
            inputs=[mode],
            outputs=[ref_image]
        )

        # 添加示例提示选择事件
        example_prompts.select(
            fn=load_example_prompt,
            inputs=[],
            outputs=[prompt]
        )
        
        generate_btn.click(
            fn=generate_video_wrapper,
            inputs=[prompt, ref_image, audio_file, mode, frames, height, width,
                   scale_a, scale_t, sampling_steps, seed, fps],
            outputs=[output_video, status_text]
        )
        
        # 添加示例和帮助信息
        with gr.Accordion("💡 使用帮助", open=False):
            gr.Markdown("""
            ## 快速入门指南
            
            1. **加载模型**: 首先点击"加载模型"按钮，确保模型路径正确
            2. **选择模式**: 
               - **TA模式**: 仅需文本和音频（可选）
               - **TIA模式**: 需要文本、参考图像和音频（可选）
            3. **输入提示**: 详细描述您想要生成的视频内容
            4. **上传文件**: 根据需要上传参考图像和/或音频文件
            5. **点击生成**: 等待视频生成完成
            
            ## 提示技巧
            
            ### 高效文本提示：
            - 使用具体、详细的描述（如"A young woman with blonde hair dancing energetically to pop music"）
            - 包含动作、表情和场景细节（如"A man smiling and speaking confidently in a bright office setting"）
            - 描述情感和氛围（如"A person playing guitar passionately with eyes closed in a dimly lit room"）
            
            ### 参数调整建议：
            - **音频引导强度 (scale_a)**: 5.0-6.0 适合大多数情况，更高值使动作更紧密跟随音频
            - **文本引导强度 (scale_t)**: 4.0-6.0 平衡创意和准确性，更高值使生成更符合文本描述
            - **采样步数**: 30-50 平衡质量和速度，更多步数生成质量更高但耗时更长
            - **视频分辨率**: 720p 质量更好，480p 速度更快
            - **帧数**: 97帧适合大多数场景，减少帧数可加快生成速度
            
            ### 常见问题解答：
            - **生成失败**: 检查模型路径是否正确，确保TIA模式下提供了参考图像
            - **视频质量不佳**: 尝试增加采样步数，调整文本和音频引导强度
            - **动作不自然**: 调整音频引导强度，或使用更清晰的音频文件
            - **生成缓慢**: 降低分辨率、减少帧数或采样步数可加快生成速度
            
            ### 注意事项：
            - 确保模型文件已正确下载到指定路径
            - TIA模式需要提供参考图像
            - 音频文件支持WAV格式
            - 生成时间取决于硬件配置和参数设置
            """)
    
    return interface

if __name__ == "__main__":
    # 创建界面
    interface = create_interface()
    
    # 启动应用
    interface.launch(
        server_name="0.0.0.0",
        share=False,
        debug=True,
        show_error=True
    )