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
        
    def load_model(self, model_type: str = "1.7B"):
        """加载HuMo模型"""
        try:
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
            
            return f"✅ {model_type}模型加载成功！"
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
            
            if self.generator is None:
                self.is_generating = False
                return "", "❌ 请先加载模型", 0
            
            if not prompt.strip():
                self.is_generating = False
                return "", "❌ 请输入文本提示", 0
            
            # 准备输出目录
            output_dir = os.path.join(self.temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
        
            self.logger.info("创建新的测试用例文件")
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
                img_path = os.path.join(self.temp_dir, "ref_image.png")
                ref_image.save(img_path)
                test_case["case_1"]["img_paths"] = [img_path]
            elif mode == "TIA" and ref_image is None:
                self.is_generating = False
                return "", "❌ TIA模式需要提供参考图像", 0
            
            # 处理音频文件
            if audio_file is not None:
                self.logger.info(f"收到音频文件: {audio_file}")
                self.logger.info(f"音频文件类型: {type(audio_file)}")
                
                # 检查audio_file是否是有效的文件路径
                if os.path.isfile(audio_file):
                    self.logger.info(f"音频文件验证通过: {audio_file}")
                    audio_path = os.path.join(self.temp_dir, "audio.wav")
                    shutil.copy2(audio_file, audio_path)
                    test_case["case_1"]["audio_path"] = audio_path
                    self.logger.info(f"音频文件已复制到: {audio_path}")
                elif os.path.isdir(audio_file):
                    self.logger.warning(f"音频文件路径是目录而不是文件: {audio_file}")
                    test_case["case_1"]["audio_path"] = None
                else:
                    self.logger.warning(f"音频文件路径无效或不存在: {audio_file}")
                    test_case["case_1"]["audio_path"] = None
            
            # 保存测试用例
            test_case_path = os.path.join(self.temp_dir, "test_case.json")
            with open(test_case_path, 'w', encoding='utf-8') as f:
                json.dump(test_case, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"创建的测试用例文件: {test_case_path}")
            self.logger.info(f"测试用例内容: {test_case}")
            
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
            
            self.logger.info(f"配置更新完成，开始生成视频...")

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
                self.logger.info(f"生成耗时: {generation_duration:.2f}秒")
                self.logger.info(f"合成总耗时: {total_duration:.2f}秒")
                self.logger.debug(f"视频文件存在: {os.path.exists(video_path)}")
                self.logger.debug(f"视频文件是文件: {os.path.isfile(video_path)}")
                
                if os.path.exists(video_path):
                    success_message = f"✅ 视频生成成功！生成耗时: {generation_duration:.2f}秒，总耗时: {total_duration:.2f}秒"
                    return video_path, success_message, 100
                else:
                    self.logger.warning(f"视频文件不存在: {video_path}")
                    return "", "❌ 视频文件不存在", 0
            else:
                self.logger.warning(f"未找到视频文件，输出目录内容: {list(Path(output_dir).iterdir())}")
                return "", "❌ 未找到生成的视频文件", 0
                
        except Exception as e:
            # 计算总耗时（即使失败也记录）
            total_duration = time.time() - start_time
            self.logger.error(f"视频生成失败，总耗时: {total_duration:.2f}秒")
            traceback.print_exc()
            self.is_generating = False
            return "", f"❌ 视频生成失败: {str(e)}", 0
    
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
                model_type = gr.Dropdown(
                    choices=["1.7B", "17B"],
                    value="1.7B",
                    label="模型类型",
                    info="1.7B: 轻量级模型，速度快 | 17B: 高质量模型，需要更多显存"
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
                        
                        # 反向提示词
                        negative_prompt = gr.Textbox(
                            label="反向提示词 (可选)",
                            placeholder="描述你不想要的内容...",
                            lines=2,
                            value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
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
                                    label="音频文件",
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
                        with gr.Accordion("视频参数", open=True):
                            with gr.Row():
                                frames = gr.Slider(
                                    minimum=-1, maximum=200, value=-1, step=1,
                                    label="视频帧数",
                                    info="设置为-1时根据音频长度自动计算帧数，其他值手动指定帧数"
                                )
                                fps = gr.Slider(
                                    minimum=15, maximum=60, value=25, step=1,
                                    label="帧率 (FPS)",
                                    info="每秒显示的帧数，影响视频流畅度"
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
                                info="720P质量更好，480P速度更快；竖屏适合手机观看，横屏适合电脑观看，方形适合社交媒体"
                            )
                        
                        # 生成参数折叠面板
                        with gr.Accordion("生成参数", open=True):
                            with gr.Row():
                                scale_i = gr.Slider(
                                    minimum=1.0, maximum=10.0, value=5.0, step=0.1,
                                    label="图像引导强度", 
                                    info="越高越遵循参考图像，推荐值：4.0-6.0"
                                )
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
        def load_model_wrapper(model_type):
            return app.load_model(model_type)
        
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
            inputs=[prompt, negative_prompt, ref_image, audio_file, mode, frames, resolution,
                   scale_i, scale_a, scale_t, sampling_steps, seed, fps],
            outputs=[output_video, status_text]
        )
        
        # 添加示例和帮助信息
        with gr.Accordion("💡 使用帮助", open=False):
            gr.Markdown("""
            ## 快速入门指南
            
            1. **选择模型**: 从下拉菜单选择模型类型
               - **1.7B模型**: 轻量级，速度快，显存需求低，适合快速测试
               - **17B模型**: 高质量，效果好，显存需求高，适合高质量生成
            2. **加载模型**: 点击"加载模型"按钮加载选定的模型
            3. **选择模式**: 
               - **TA模式**: 仅需文本和音频（可选）
               - **TIA模式**: 需要文本、参考图像和音频（可选）
            4. **输入提示**: 详细描述您想要生成的视频内容
            5. **上传文件**: 根据需要上传参考图像和/或音频文件
            6. **点击生成**: 等待视频生成完成
            
            ## 提示技巧
            
            ### 高效文本提示：
            - **正向提示词**：使用具体、详细的描述（如"A young woman with blonde hair dancing energetically to pop music"）
            - **反向提示词**：描述你不想要的内容（如"模糊，低质量，静态画面"）
            - 包含动作、表情和场景细节（如"A man smiling and speaking confidently in a bright office setting"）
            - 描述情感和氛围（如"A person playing guitar passionately with eyes closed in a dimly lit room"）
            
            ### 参数调整建议：
            - **音频引导强度 (scale_a)**: 5.0-6.0 适合大多数情况，更高值使动作更紧密跟随音频
            - **文本引导强度 (scale_t)**: 4.0-6.0 平衡创意和准确性，更高值使生成更符合文本描述
            - **采样步数**: 30-50 平衡质量和速度，更多步数生成质量更高但耗时更长
            - **视频分辨率**: 720P 质量更好，480P 速度更快
            - **帧数**: 97帧适合大多数场景，减少帧数可加快生成速度
            
            ### 常见问题解答：
            - **生成失败**: 检查模型路径是否正确，确保TIA模式下提供了参考图像
            - **视频质量不佳**: 尝试增加采样步数，调整文本和音频引导强度
            - **动作不自然**: 调整音频引导强度，或使用更清晰的音频文件
            - **生成缓慢**: 降低分辨率、减少帧数或采样步数可加快生成速度
            
            ### 注意事项：
            - 确保模型文件已正确下载到对应路径（1.7B: ./weights/HuMo/HuMo-1.7B, 17B: ./weights/HuMo/HuMo-17B）
            - 1.7B模型使用.pth格式，17B模型使用safetensors格式
            - TIA模式需要提供参考图像
            - 音频文件支持WAV格式
            - 生成时间取决于硬件配置和参数设置
            - 17B模型需要更多显存，建议至少520GB显存
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