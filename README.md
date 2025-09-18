<div align="center">
<h1> HuMo: 基于协作多模态条件的人体中心视频生成 </h1>

<a href="https://arxiv.org/abs/2509.08519"><img src="https://img.shields.io/badge/arXiv%20paper-2509.08519-b31b1b.svg"></a>
<a href="https://phantom-video.github.io/HuMo/"><img src="https://img.shields.io/badge/Project_page-More_visualizations-green"></a>
<a href="https://huggingface.co/bytedance-research/HuMo"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=orange"></a>

[Liyang Chen](https://scholar.google.com/citations?user=jk6jWXgAAAAJ&hl)<sup> * </sup>, [Tianxiang Ma](https://tianxiangma.github.io/)<sup> * </sup>, [Jiawei Liu](https://scholar.google.com/citations?user=X21Fz-EAAAAJ), [Bingchuan Li](https://scholar.google.com/citations?user=ac5Se6QAAAAJ)<sup> &dagger; </sup>, <br>[Zhuowei Chen](https://scholar.google.com/citations?user=ow1jGJkAAAAJ), [Lijie Liu](https://liulj13.github.io/), [Xu He](https://scholar.google.com/citations?user=KMrFk2MAAAAJ&hl), [Gen Li](https://scholar.google.com/citations?user=wqA7EIoAAAAJ), [Qian He](https://scholar.google.com/citations?user=9rWWCgUAAAAJ), [Zhiyong Wu](https://scholar.google.com/citations?user=7Xl6KdkAAAAJ)<sup> § </sup><br>
<sup> * </sup>共同第一作者, <sup> &dagger; </sup>项目负责人, <sup> § </sup>通讯作者  
清华大学 | 字节跳动智能创作团队

</div>

<p align="center">
<img src="assets/teaser.png" width=95%>
<p>

## 🔥 最新动态

* HuMo最佳实践指南即将发布，敬请期待。
* 2025年9月16日：🔥🔥 我们发布了[1.7B权重](https://huggingface.co/bytedance-research/HuMo/tree/main/HuMo-1.7B)，可在32G GPU上8分钟内生成480P视频。虽然视觉质量比17B模型略低，但音视频同步效果几乎不受影响。
* 2025年9月13日：🔥🔥 17B模型已集成到[ComfyUI-Wan](https://github.com/kijai/ComfyUI-WanVideoWrapper)，可在NVIDIA 3090 GPU上运行。感谢[kijai](https://github.com/kijai)的更新！
* 2025年9月10日：🔥🔥 我们发布了[17B权重](https://huggingface.co/bytedance-research/HuMo/tree/main/HuMo-17B)和推理代码。
* 2025年9月9日：我们发布了**HuMo**的[项目主页](https://phantom-video.github.io/HuMo/)和[技术报告](https://arxiv.org/abs/2509.08519/)。

## ✨ 核心特性
HuMo是一个统一的、以人为中心的视频生成框架，旨在从多模态输入（包括文本、图像和音频）生成高质量、细粒度且可控的人体视频。它支持强大的文本提示跟随、一致的主体保持和同步的音频驱动动作。

> - **文本-图像视频生成** - 使用文本提示结合参考图像自定义角色外观、服装、妆容、道具和场景。
> - **文本-音频视频生成** - 仅从文本和音频输入生成音频同步视频，无需图像参考，实现更大的创作自由度。
> - **文本-图像-音频视频生成** - 通过结合文本、图像和音频指导实现更高级别的定制和控制。

## 📑 待办事项
- [x] 发布论文
- [x] HuMo-17B检查点
- [x] HuMo-1.7B检查点
- [x] 推理代码
  - [ ] 文本-图像输入
  - [x] 文本-音频输入
  - [x] 文本-图像-音频输入
- [x] 多GPU推理
- [ ] HuMo电影级生成最佳实践指南
- [ ] 生成***无面权游***演示的提示词
- [ ] 训练数据

## ⚡️ 快速开始

### 安装环境
```
conda create -n humo python=3.11
conda activate humo
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install flash_attn==2.6.3
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

### 模型准备
| 模型       | 下载链接                                                                                                                                           |    说明                      |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| HuMo-17B      | 🤗 [Huggingface](https://huggingface.co/bytedance-research/HuMo/tree/main/HuMo-17B)   | 支持480P和720P 
| HuMo-1.7B | 🤗 [Huggingface](https://huggingface.co/bytedance-research/HuMo/tree/main/HuMo-1.7B) | 32G GPU轻量版
| Wan-2.1 | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | VAE和文本编码器
| Whisper-large-v3 |      🤗 [Huggingface](https://huggingface.co/openai/whisper-large-v3)          | 音频编码器
| Audio separator |      🤗 [Huggingface](https://huggingface.co/huangjackson/Kim_Vocal_2)          | 去除背景噪音（可选）

使用huggingface-cli下载模型：
``` sh
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./weights/Wan2.1-T2V-1.3B
huggingface-cli download bytedance-research/HuMo --local-dir ./weights/HuMo
huggingface-cli download openai/whisper-large-v3 --local-dir ./weights/whisper-large-v3
huggingface-cli download huangjackson/Kim_Vocal_2 --local-dir ./weights/audio_separator
```

### 运行多模态条件视频生成

我们的模型兼容480P和720P分辨率。720P推理将获得更好的质量。
> 一些建议
> - 请按照[test_case.json](./examples/test_case.json)中的描述准备您的文本、参考图像和音频。
> - 我们使用FSDP + 序列并行支持多GPU推理。
> - 模型在25 FPS的97帧视频上训练。生成超过97帧的视频可能会降低性能。我们将提供用于更长生成的新检查点。

#### 配置HuMo

可以通过修改[generate.yaml](humo/configs/inference/generate.yaml)配置文件来自定义HuMo的行为和输出。
以下参数控制生成长度、视频分辨率以及文本、图像和音频输入的平衡：

```yaml
generation:
  frames: <int>                 # 生成视频的帧数
  scale_a: <float>              # 音频指导强度。越高 = 音频运动同步越好
  scale_t: <float>              # 文本指导强度。越高 = 更好地遵循文本提示
  mode: "TA"                    # 输入模式："TA"表示文本+音频；"TIA"表示文本+图像+音频
  height: 720                   # 视频高度（例如720或480）
  width: 1280                   # 视频宽度（例如1280或832）

dit:
  sp_size: <int>                # 序列并行大小。设置为使用的GPU数量

diffusion:
  timesteps:
    sampling:
      steps: 50                 # 去噪步数。较低（30-40） = 更快生成
```

#### 1. 文本-音频输入

``` sh
bash scripts/infer_ta.sh  # 使用17B模型推理
bash scripts/infer_ta_1_7B.sh  # 使用1.7B模型推理
```

#### 2. 文本-图像-音频输入

``` sh
bash scripts/infer_tia.sh  # 使用17B模型推理
bash scripts/infer_tia_1_7B.sh  # 使用1.7B模型推理
```

## 致谢
我们的工作建立在并受到几个优秀开源项目的极大启发，包括[Phantom](https://github.com/Phantom-video/Phantom)、[SeedVR](https://github.com/IceClear/SeedVR?tab=readme-ov-file)、[MEMO](https://github.com/memoavatar/memo)、[Hallo3](https://github.com/fudan-generative-vision/hallo3)、[OpenHumanVid](https://github.com/fudan-generative-vision/OpenHumanVid)、[OpenS2V-Nexus](https://github.com/PKU-YuanGroup/OpenS2V-Nexus)、[ConsisID](https://github.com/PKU-YuanGroup/ConsisID)和[Whisper](https://github.com/openai/whisper)。我们衷心感谢这些项目的作者和贡献者慷慨分享他们优秀的代码和想法。

## ⭐ 引用

如果HuMo对您有帮助，请帮助为仓库点⭐。

如果您认为这个项目对您的研究有用，请考虑引用我们的[论文](https://arxiv.org/abs/2509.08519)。

### BibTeX
```bibtex
@misc{chen2025humo,
      title={HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning}, 
      author={Liyang Chen and Tianxiang Ma and Jiawei Liu and Bingchuan Li and Zhuowei Chen and Lijie Liu and Xu He and Gen Li and Qian He and Zhiyong Wu},
      year={2025},
      eprint={2509.08519},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.08519}, 
}
```

## 📧 联系方式
如果您对这个开源项目有任何意见或问题，请提出新的issue或联系[Liyang Chen](https://leoniuschen.github.io/)和[Tianxiang Ma](https://tianxiangma.github.io/)。