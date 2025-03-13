# ComfyUI-IF_VideoDatasetMaker

<img src="https://count.getloli.com/get/@IFAIDATASET_comfy?theme=moebooru" alt=":IFAIDATASET_comfy" />

An advanced ComfyUI node that converts video files or YouTube links into well-structured training datasets for AI image generation models.
This tool is based on the work done by [zsxkib](https://github.com/zsxkib) in their repository [cog-create-video-dataset](https://github.com/zsxkib/cog-create-video-dataset).

## Overview

The **IF_VideoDatasetMaker** node allows you to:

- Download YouTube videos or use local video files
- Automatically segment videos into high-quality clips
- Generate intelligent captions using Qwen-VL models that describe the visual content
- Create a training-ready dataset with the proper file structure
- Customize trigger words and captions to fit your specific needs

Perfect for creating training datasets for HyperNetworks, LoRAs, Dreambooth or other fine-tuning approaches.

## Features

- **Multi-source Input**: Process YouTube links or local video files
- **Intelligent Scene Detection**: Automatically extract the best quality clips based on content changes
- **AI-powered Captioning**: Generate detailed, useful captions with multimodal AI
- **Customizable Output**: Control caption prefixes, suffixes, and triggers
- **Structured Organization**: Creates proper dataset structure for immediate training use
- **Automatic Compression**: Generates a ready-to-share ZIP file of your dataset
- **Debugging Options**: Additional debug information to help troubleshoot issues

## Installation

### Prerequisites

- ComfyUI installed and working
- Python 3.8+ environment
- FFmpeg installed on your system (required for video processing)

### Installing the Node

1. Clone this repository to your ComfyUI custom_nodes directory:

```bash
cd ./custom_nodes
git clone https://github.com/if-ai/ComfyUI-IF_DatasetMkr.git
```

2. Install the required dependencies:

```bash
cd ./ComfyUI-IF_DatasetMkr.
pip install -r requirements.txt
```

If you want to use AWQ to save VRAM and up to 3x faster inference
you need to install triton and autoawq

```
pip install triton
pip install --no-deps --no-build-isolation autoawq
```



3. Restart ComfyUI

## Dependencies

This node requires several packages for video processing, AI captioning, and file management:

- yt-dlp - For YouTube video downloading
- scenedetect - For intelligent clip extraction
- python-slugify - For filename normalization
- moviepy - For video editing and frame extraction
- opencv-python - For image processing
- qwen-vl-utils - For multimodal AI caption generation
- ffmpeg-python - For advanced video processing
- Hugging Face Transformers - For running the Qwen-VL models

## Usage

1. Add the **IF_VideoDatasetMkr** node to your ComfyUI workflow
2. Configure the node with your desired settings:
   - Provide a YouTube URL or local video file path
   - Choose your captioning model
   - Set custom trigger words if needed
   - Select output directory and options
3. Run the workflow to generate your dataset
4. Use the resulting dataset ZIP for training your models

## Node Parameters

| Parameter | Description |
|-----------|-------------|
| video_url | YouTube/video URL to process |
| video_file | Local video file path to process |
| trigger_word | Custom trigger word (optional) |
| autocaption | Enable/disable AI captioning |
| custom_caption | Static caption to use for all frames |
| autocaption_prefix | Text to add before all generated captions |
| autocaption_suffix | Text to add after all generated captions |
| output_dir | Custom output directory (defaults to ComfyUI output folder) |
| model_variant | Qwen-VL model to use for captioning |
| model_offload | Toggle CPU offloading to save VRAM |
| hf_token | Hugging Face token for downloading models |
| profile | Captioning profile/persona to use |
| image_size | Resolution for processing frames |
| debug_mode | Enable additional debugging information |

## Output

The node generates a structured dataset with:

- A `videos` folder containing all extracted clips
- A `captions` folder with text files matching each clip's name
- Caption files following the naming convention: `{trigger_word}_{clip_number}.txt`
- A compressed ZIP archive for easy sharing

## Customization

### Caption Profiles

The node comes with built-in caption profiles that control the style and content of generated descriptions. You can edit these profiles or create your own by modifying the `profiles.json` file.

### Trigger Words

For training specialized models, you can set custom trigger words that will be included in the dataset. These can be used later to activate your trained model.

## Troubleshooting

- **Video Download Issues**: Ensure yt-dlp is up to date and the URL is valid
- **FFmpeg Errors**: Make sure FFmpeg is installed on your system and in your PATH
- **Caption Generation Errors**: Check that you have enough VRAM or try a smaller model
- **Missing Clips**: Try enabling debug mode to see detailed processing information

## Requirements

This is a comprehensive list of requirements for the node:

```
# Core dependencies
torch>=2.0.0
pillow>=10.0.0
numpy>=1.24.0
huggingface_hub>=0.26.0

# AutoAWQ with specific version - MUST be installed before transformers
autoawq==0.2.8
flash-attn>=2.0.0;platform_system!="Darwin"  # Optional for performance, excluded on MacOS

# Transformers - MUST be installed after AutoAWQ
transformers>=4.49.0
accelerate>=0.21.0

# Qwen model dependencies
tokenizers>=0.15.0
safetensors>=0.3.1
qwen-vl-utils[decord]>=0.0.8

# Video processing
opencv-python>=4.8.0
decord>=0.6.0
ffmpeg-python>=0.2.0
imageio_ffmpeg>=0.6.0
moviepy>=2.1.2
scenedetect>=0.6.2

# Downloading
yt-dlp>=2023.3.4

# Utilities
tqdm>=4.66.1
requests>=2.31.0
python-slugify>=8.0.1
psutil>=5.9.0
packaging>=23.1
aiohttp>=3.8.5
dotenv-python>=0.0.1
```

## Support

If you find this tool useful, please consider supporting my work by:
* Starring the repository on GitHub: [ComfyUI-IF_VideoPrompts](https://github.com/yourusername/ComfyUI-IF_VideoPrompts)
* Subscribing to my YouTube channel: [Impact Frames](https://youtube.com/c/impactframes)
* Follow me on X: [Impact Frames X](https://x.com/impactframes)

Thank You!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
