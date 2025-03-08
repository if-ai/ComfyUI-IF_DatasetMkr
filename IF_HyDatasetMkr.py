import os
import re
import json
import shutil
import logging
import asyncio
import math
from pathlib import Path

import yt_dlp
from scenedetect import detect, ContentDetector
from slugify import slugify
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image

import torch
import numpy as np
from typing import Optional, Union, List, Dict, Tuple, Any


# Import the Qwen-VL utilities and transformer API manager.
from .transformers_api import _transformers_manager
from folder_paths import models_dir, get_output_directory


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ----------------------------
# Helper Functions
# ----------------------------

def ensure_model_downloaded(model_name: str, token: Optional[str] = None) -> str:
    """Ensures the model is downloaded to correct location with AWQ support"""
    from huggingface_hub import snapshot_download
    import os
    
    # Get ComfyUI models directory
    models_dir_path = os.path.join(models_dir, "LLM")
    os.makedirs(models_dir_path, exist_ok=True)
    
    # Handle local models
    if model_name.startswith("local:"):
        local_name = model_name.split(":", 1)[1]
        model_dir = os.path.join(models_dir_path, local_name)
        if not os.path.exists(model_dir):
            raise ValueError(f"Local model {local_name} not found in {model_dir}")
        return model_dir
    
    # Extract base model name (without AWQ)
    model_base = model_name.split('/')[-1].replace("-AWQ", "")
    model_dir = os.path.join(models_dir_path, model_base)
    
    # Special handling for AWQ models
    if "AWQ" in model_name:
        awq_dir = os.path.join(models_dir_path, f"{model_base}-AWQ")
        
        # Check if AWQ model exists
        if not os.path.exists(os.path.join(awq_dir, "quant_config.json")):
            logger.info(f"Downloading AWQ model {model_name} to {awq_dir}")
            try:
                snapshot_download(
                    repo_id=model_name,
                    local_dir=awq_dir,
                    local_dir_use_symlinks=False,
                    token=token,
                    allow_patterns=["*.json", "*.bin", "*.model", "*.py", "*.txt", "*.safetensors"]
                )
                logger.info(f"Successfully downloaded AWQ model to {awq_dir}")
                return awq_dir
            except Exception as e:
                logger.error(f"Failed to download AWQ model {model_name}: {e}")
                raise RuntimeError(f"Failed to download AWQ model {model_name}: {e}")
        else:
            logger.info(f"Using existing AWQ model at {awq_dir}")
            return awq_dir
    
    # Standard model check
    if not os.path.exists(model_dir):
        logger.info(f"Downloading {model_name} to {model_dir}...")
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                token=token
            )
            logger.info(f"Successfully downloaded {model_name} to {model_dir}")
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            raise RuntimeError(f"Failed to download model {model_name}: {e}")
    else:
        logger.info(f"Using existing model at {model_dir}")
    
    return model_dir

def get_optimal_device():
    """Determine the best device for model loading"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def extract_title_and_trigger(title: str, username: str = "deepfates", video_prefix: str = "The Beauty of") -> tuple[str, str]:
    # Remove prefix and clean title
    title = re.sub(f'^{re.escape(video_prefix)}\\s+', '', title, flags=re.IGNORECASE).strip()
    logging.info(f"Cleaned title: {title}")
    model_name = f"{username}/hunyuan-{slugify(title)}"
    # Generate trigger word from first 5 consonants; fallback if none found
    consonants = ''.join(re.findall(r'[BCDFGHJKLMNPQRSTVWXYZ]', title.upper()))
    trigger_word = consonants[:5] if consonants else slugify(title).upper()[:5]
    logging.info(f"Generated model_name: {model_name}, trigger_word: {trigger_word}")
    return model_name, trigger_word

def process_video_frame(frame) -> Optional[Image.Image]:
    """
    Safely process a video frame into a PIL Image with proper dimension handling
    
    Args:
        frame: numpy array or torch tensor of video frame
        
    Returns:
        PIL Image or None if processing fails
    """
    try:
        # Handle None input
        if frame is None:
            logger.error("Received None frame, cannot process")
            return None
            
        # Handle tensor input
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
            
        # Ensure numpy array
        if not isinstance(frame, np.ndarray):
            logger.error(f"Unsupported frame type: {type(frame)}")
            return None
            
        # Check for empty arrays
        if frame.size == 0:
            logger.error("Empty frame array, cannot process")
            return None
            
        # Normalize values to [0,255] range
        if frame.dtype in [np.float32, np.float64]:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
                
        # Handle different dimension arrangements
        shape = frame.shape
        if len(shape) == 4:  # (B,H,W,C) or (B,C,H,W)
            if shape[-1] not in [1, 3, 4]:  # Probably (B,C,H,W)
                frame = frame.transpose(0, 2, 3, 1)
            frame = frame[0]  # Take first batch
        elif len(shape) == 3:
            if shape[0] in [1, 3, 4]:  # (C,H,W) format
                frame = frame.transpose(1, 2, 0)
                
        # Handle color channels
        if frame.shape[-1] == 4:  # RGBA
            rgb = frame[..., :3]
            alpha = frame[..., 3:]
            # Create white background
            white_bg = np.ones_like(rgb) * 255
            # Alpha blend
            frame = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        elif frame.shape[-1] == 1:  # Grayscale
            frame = np.repeat(frame, 3, axis=-1)
            
        # Ensure valid dimensions and values
        if len(frame.shape) != 3 or frame.shape[-1] != 3:
            logger.error(f"Invalid frame shape after processing: {frame.shape}")
            return None
            
        if frame.min() < 0 or frame.max() > 255:
            logger.warning("Frame values out of valid range, clipping to [0,255]")
            frame = np.clip(frame, 0, 255)
            
        return Image.fromarray(frame)
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return None

def process_clips_batch(video_path: Path, scenes: list, clips_dir: Path, trigger_word: str, end_time: float, debug_dir: Optional[Path] = None) -> list:
    """
    Process video clips with improved error handling and frame processing
    """
    clips_metadata = []
    valid_clips = 0
    
    try:
        import subprocess
        import tempfile
        
        # Try different approaches to extract clips
        video_file = str(video_path)
        
        # Save scene info if debug mode is enabled
        if debug_dir is not None:
            try:
                with open(debug_dir / "scenes_info.txt", "w") as f:
                    f.write(f"Total scenes detected: {len(scenes)}\n")
                    for i, scene in enumerate(scenes):
                        start_time = scene[0].get_seconds()
                        end_time_scene = scene[1].get_seconds()
                        f.write(f"Scene {i}: start={start_time:.2f}s, end={end_time_scene:.2f}s, duration={end_time_scene-start_time:.2f}s\n")
            except Exception as e:
                logger.error(f"Error saving scene debug info: {e}")
        
        for i, scene in enumerate(scenes):
            start_time = scene[0].get_seconds()
            if start_time >= end_time:
                continue
                
            end_time_scene = min(scene[1].get_seconds(), end_time)
            duration = end_time_scene - start_time
            
            if 1 <= duration <= 5:
                clip_path = clips_dir / f"clip_{i:04d}.mp4"
                
                try:
                    # Try using ffmpeg directly to extract the clip
                    # Add a small buffer to avoid potential frame issues
                    adjusted_start = max(0, start_time - 0.04)
                    adjusted_duration = min(duration + 0.08, end_time_scene - adjusted_start)
                    
                    ffmpeg_cmd = [
                        get_ffmpeg_path(),
                        "-ss", str(adjusted_start),
                        "-t", str(adjusted_duration),
                        "-i", video_file,
                        "-c:v", "libx264",
                        "-preset", "fast",  # Use faster preset for better compatibility
                        "-crf", "23",       # Reasonable quality
                        "-c:a", "aac",
                        "-strict", "experimental",
                        "-b:a", "128k",
                        "-y",
                        str(clip_path)
                    ]
                    
                    # Save command if debug mode is enabled
                    if debug_dir is not None:
                        with open(debug_dir / f"clip_{i:04d}_cmd.txt", "w") as f:
                            f.write(" ".join(ffmpeg_cmd) + "\n")
                    
                    # Run the ffmpeg command
                    try:
                        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                        
                        # Save output if debug mode is enabled
                        if debug_dir is not None:
                            with open(debug_dir / f"clip_{i:04d}_output.txt", "w") as f:
                                f.write(f"STDOUT:\n{result.stdout.decode() if result.stdout else ''}\n")
                                f.write(f"STDERR:\n{result.stderr.decode() if result.stderr else ''}\n")
                        
                        # Verify the clip was created and is valid
                        if clip_path.exists() and os.path.getsize(str(clip_path)) > 0:
                            # Extract a test frame to validate the clip
                            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp:
                                test_frame_path = tmp.name
                            
                            # Extract first frame to verify clip is valid
                            frame_cmd = [
                                get_ffmpeg_path(),
                                "-i", str(clip_path),
                                "-vframes", "1",
                                "-f", "image2",
                                "-y",
                                test_frame_path
                            ]
                            
                            frame_result = subprocess.run(frame_cmd, capture_output=True)
                            
                            if os.path.exists(test_frame_path) and os.path.getsize(test_frame_path) > 0:
                                # Successfully created and validated clip
                                clips_metadata.append({
                                    "file_name": clip_path.name,
                                    "id": f"{trigger_word}_{i:04d}",
                                    "duration": duration,
                                    "start_time": start_time,
                                    "end_time": end_time_scene
                                })
                                valid_clips += 1
                                logger.info(f"Successfully created clip {i} using ffmpeg")
                                
                                # Save test frame if debug mode is enabled
                                if debug_dir is not None:
                                    try:
                                        debug_frame_path = debug_dir / f"clip_{i:04d}_frame.jpg"
                                        shutil.copy(test_frame_path, debug_frame_path)
                                    except Exception as e:
                                        logger.error(f"Error saving debug frame: {e}")
                                
                                # Clean up test frame
                                try:
                                    os.remove(test_frame_path)
                                except:
                                    pass
                            else:
                                logger.error(f"Clip {i} created but failed validation")
                                # Try to clean up invalid clip
                                try:
                                    os.remove(str(clip_path))
                                except:
                                    pass
                        else:
                            logger.error(f"Failed to create clip {i}: ffmpeg ran but no valid output file was created")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Error running ffmpeg for clip {i}: {e}")
                        logger.error(f"ffmpeg stderr: {e.stderr.decode() if e.stderr else 'None'}")
                        
                        # Save error if debug mode is enabled
                        if debug_dir is not None:
                            with open(debug_dir / f"clip_{i:04d}_error.txt", "w") as f:
                                f.write(f"Error: {e}\n")
                                f.write(f"STDERR:\n{e.stderr.decode() if e.stderr else 'None'}\n")
                        
                except Exception as e:
                    logger.error(f"Error processing clip {i}: {e}")
                    continue
                        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return clips_metadata

    logger.info(f"Successfully processed {valid_clips} clips")
    return clips_metadata

def get_clip_frame(clip_path: Path) -> Optional[Image.Image]:
    """
    Safely extract the first frame from a video clip
    """
    try:
        # Try using MoviePy to get the frame
        try:
            with VideoFileClip(str(clip_path)) as clip:
                frame = clip.get_frame(0)
                return process_video_frame(frame)
        except Exception as e:
            logger.error(f"MoviePy error extracting frame from {clip_path}: {e}")
        
        # If MoviePy fails, try using ffmpeg directly
        import subprocess
        import tempfile
        
        # Create a temporary file for the frame
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_filename = tmp.name
        
        try:
            # Extract a single frame using ffmpeg
            ffmpeg_cmd = [
                get_ffmpeg_path(),
                "-i", str(clip_path),
                "-ss", "0",
                "-vframes", "1",
                "-f", "image2",
                tmp_filename
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            
            # Open the frame if it was successfully extracted
            if os.path.exists(tmp_filename):
                pil_img = Image.open(tmp_filename)
                pil_img = pil_img.convert("RGB")  # Ensure RGB mode
                os.unlink(tmp_filename)  # Clean up
                return pil_img
        except Exception as e:
            logger.error(f"ffmpeg error extracting frame from {clip_path}: {e}")
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)  # Clean up temporary file
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting frame from {clip_path}: {e}")
        return None

def get_ffmpeg_path() -> str:
    """
    Find ffmpeg executable across different platforms.
    Returns the path to ffmpeg or raises an exception if not found.
    """
    import platform
    import shutil
    import os
    system = platform.system().lower()

    # First, try to find ffmpeg in the system PATH
    ffmpeg_in_path = shutil.which('ffmpeg')
    if ffmpeg_in_path:
        return ffmpeg_in_path

    # Define common installation paths for each platform
    possible_paths = []
    if system == 'darwin':  # macOS
        possible_paths = [
            "/opt/homebrew/bin/ffmpeg",  # Apple Silicon Homebrew
            "/usr/local/bin/ffmpeg",     # Intel Homebrew
            "/usr/bin/ffmpeg",           # System install
        ]
    elif system == 'windows':
        possible_paths = [
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            os.path.join(os.getenv('APPDATA', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
            os.path.join(os.getenv('LOCALAPPDATA', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
        ]
    elif system == 'linux':
        possible_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/ffmpeg/bin/ffmpeg",
            os.path.expanduser("~/.local/bin/ffmpeg"),
        ]

    # Check if any of the possible paths actually exist
    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Provide platform-specific installation instructions if ffmpeg isn't found
    install_instructions = {
        'darwin': "Install ffmpeg using: brew install ffmpeg",
        'windows': "Download ffmpeg from https://www.ffmpeg.org/download.html#build-windows or install using: winget install ffmpeg",
        'linux': "Install ffmpeg using your package manager, e.g.: sudo apt install ffmpeg (Ubuntu/Debian) or sudo dnf install ffmpeg (Fedora)"
    }
    
    raise RuntimeError(
        f"ffmpeg not found. {install_instructions.get(system, 'Please install ffmpeg for your operating system.')}"
    )

# ----------------------------
# Custom Node Definition
# ----------------------------

class IF_HyDatasetMkr:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.presets_dir = os.path.join(self.current_dir, "presets")
        self.profiles_path = os.path.join(self.presets_dir, "profiles.json")
        
        # Create presets directory if needed
        os.makedirs(self.presets_dir, exist_ok=True)
        
        # Initialize profiles from JSON file
        self.profiles = self.load_presets(self.profiles_path)
        
        # Remove the example profile creation block
        # Simply load existing profiles without defaults

    def load_presets(self, file_path: str) -> Dict[str, Any]:
        """Load profiles from external JSON file"""
        if not os.path.exists(file_path):
            # Create empty JSON file instead of default profiles
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f, indent=2)
                logger.info(f"Created empty profiles file at {file_path}")
                return {}
            except Exception as e:
                logger.error(f"Failed to create profiles file: {e}")
                return {}

        # Load existing profiles with encoding fallback
        encodings = ['utf-8', 'utf-8-sig', 'latin1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load profiles with {encoding}: {e}")
                continue
            
        logger.error("Failed to load profiles with any supported encoding")
        return {}

    def get_system_prompt(self, profile: str, custom_prompt: str = None) -> str:
        """Get system prompt from profile or custom input"""
        if custom_prompt and custom_prompt.strip():
            return custom_prompt.strip()
        
        if profile != "None" and profile in self.profiles:
            content = self.profiles[profile]
            if isinstance(content, dict):
                instruction = content.get("instruction", "")
                rules = "\n".join([f"- {rule}" for rule in content.get("rules", [])])
                
                # Format the system prompt with instruction and rules
                system_prompt = f"{instruction}\n\n{rules}" if rules else instruction
                
                # Add examples if they exist
                examples = content.get("examples", [])
                if examples:
                    examples_text = "\n\nExamples:"
                    for i, example in enumerate(examples):
                        examples_text += f"\n\nExample {i+1}:"
                        examples_text += f"\nInput: {example.get('input', '')}"
                        examples_text += f"\nOutput: {example.get('output', '')}"
                    
                    system_prompt += examples_text
                
                return system_prompt
            return str(content)
        
        # Fallback default
        return "You are a helpful assistant that generates cinematic captions for video frames, highlighting composition, lighting, and mood."

    @classmethod
    def INPUT_TYPES(cls):
        instance = cls()
        profile_choices = ["None"] + list(instance.profiles.keys())
        
        return {
            "required": {
                "video_url": ("STRING", {"default": "", "placeholder": "YouTube/video URL"}),
                "video_file": ("STRING", {"default": "", "placeholder": "Local video file path"}),
                "trigger_word": ("STRING", {"default": "", "placeholder": "Custom trigger word (optional)"}),
                "autocaption": ("BOOLEAN", {"default": True}),
                "custom_caption": ("STRING", {"default": "", "placeholder": "Custom caption for all frames"}),
                "autocaption_prefix": ("STRING", {"default": "", "placeholder": "Prefix for auto-generated captions"}),
                "autocaption_suffix": ("STRING", {"default": "", "placeholder": "Suffix for auto-generated captions"}),
                "output_dir": ("STRING", {"default": "", "placeholder": "Optional custom output dir (default: ComfyUI output)"}),
                "model_variant": ([
                    "Qwen/Qwen2.5-VL-3B-Instruct", 
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    "Qwen/Qwen2.5-VL-72B-Instruct",
                    "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
                    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                    "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
                ], {"default": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"}),
                "model_offload": (["Yes", "No"], {"default": "Yes", "tooltip": "Offload model to CPU when not in use to save VRAM"}),
                "hf_token": ("STRING", {"default": "", "placeholder": "Hugging Face token (optional)"}),
                "profile": (profile_choices, {"default": "VideoDatasetAnalyzer"}),
                "image_size": ([
                    448,   # Minimum size
                    512,   # SD 1.5 base
                    640,   # Mid-range 
                    768,   # Standard SDXL
                    896,   # Large
                    1024   # XL size
                ], {"default": 768}),
                "debug_mode": (["No", "Yes"], {"default": "No", "tooltip": "Save debug information to help troubleshoot issues"}),
            },
            "optional": {
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_info",)
    FUNCTION = "create_video_dataset"
    CATEGORY = "ImpactFrames💥🎞️/dataset"

    def create_video_dataset(
        self,
        video_url: str,
        video_file: str,
        trigger_word: str,
        autocaption: bool,
        custom_caption: str,
        autocaption_prefix: str,
        autocaption_suffix: str,
        output_dir: str,
        model_variant: str,
        model_offload: str,
        hf_token: str,
        profile: str,
        image_size: int,
        custom_system_prompt: str = "",
        debug_mode: str = "No"
    ) -> tuple:
        try:
            # If hf_token is provided, override environment variable
            if hf_token.strip():
                import os
                os.environ["HF_TOKEN"] = hf_token.strip()
                logger.info("Using user-supplied HF_TOKEN from node input.")

            # Get ComfyUI's output directory as base
            base_output_dir = Path(get_output_directory())
            
            # If output_dir is provided, make it a subdirectory of ComfyUI's output
            if output_dir.strip():
                # Remove any leading/trailing slashes and create path
                clean_subdir = output_dir.strip().strip('/')
                out_dir = base_output_dir / clean_subdir
            else:
                # Default to video_datasets subdirectory
                out_dir = base_output_dir / "video_datasets"
                
            # Create directory if it doesn't exist
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Get system message
            system_message = self.get_system_prompt(profile, custom_system_prompt)
            
            metadata = self.process_video_dataset(
                video_url=video_url, 
                video_file=video_file, 
                output_dir=out_dir,
                trigger_word=trigger_word, 
                autocaption=autocaption, 
                custom_caption=custom_caption,
                autocaption_prefix=autocaption_prefix, 
                autocaption_suffix=autocaption_suffix,
                model_variant=model_variant, 
                hf_token=hf_token, 
                model_offload=model_offload,
                system_message=system_message,
                image_size=image_size,
                debug_mode=debug_mode
            )
            
            # Update paths in metadata to be relative to ComfyUI output directory
            if isinstance(metadata, dict) and 'error' not in metadata:
                try:
                    # Convert paths to be relative to base output directory
                    for key in ['zip_path', 'videos_dir', 'captions_dir']:
                        if key in metadata:
                            abs_path = Path(metadata[key])
                            rel_path = abs_path.relative_to(base_output_dir)
                            metadata[key] = str(rel_path)
                except Exception as e:
                    logger.warning(f"Error converting paths to relative: {e}")

            return (json.dumps(metadata, indent=2),)
        except Exception as e:
            error_message = f"Error: {e}"
            logger.error(error_message)
            import traceback
            logger.error(traceback.format_exc())
            return (error_message,)

    def process_video_dataset(
        self,
        video_url: str,
        video_file: str,
        output_dir: Path,
        trigger_word: str,
        autocaption: bool,
        custom_caption: str,
        autocaption_prefix: str,
        autocaption_suffix: str,
        model_variant: str,
        hf_token: str,
        model_offload: str = "Yes",
        system_message: str = "",
        image_size: int = 768,
        debug_mode: str = "No"
    ) -> dict:
        """
        Processes a video (downloaded from a URL or taken from a local file),
        extracts clips, auto-generates (or uses a custom) caption for each clip via Qwen-VL,
        organizes the output into a training-ready dataset structure, and zips it.
        
        Returns a metadata dictionary.
        """
        try:
            # Get ffmpeg path first
            ffmpeg_path = get_ffmpeg_path()
            logger.info(f"Using ffmpeg from: {ffmpeg_path}")
            
            temp_dir = output_dir / "temp"
            os.makedirs(temp_dir, exist_ok=True)

            # Ensure model is available
            if hf_token and hf_token.strip():
                model_path = ensure_model_downloaded(model_variant, token=hf_token.strip())
            else:
                model_path = ensure_model_downloaded(model_variant)
            
            # Determine source: local file takes precedence over URL
            if video_file.strip():
                source_path = Path(video_file.strip())
                if not source_path.exists():
                    raise ValueError(f"Local video file not found: {video_file}")
                video_path = temp_dir / source_path.name
                shutil.copy(str(source_path), str(video_path))
                title = source_path.stem
            elif video_url.strip():
                ydl_opts = {
                    'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]',
                    'quiet': True,
                    'outtmpl': str(temp_dir / '%(title)s.%(ext)s'),
                    'socket_timeout': 30,
                    'retries': 3,
                    'ffmpeg_location': ffmpeg_path,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    logger.info(f"Downloading video from URL: {video_url}")
                    info = ydl.extract_info(video_url, download=True)
                    title = info.get('title', 'video')
                video_files = list(temp_dir.glob("*.mp4"))
                if not video_files:
                    raise ValueError("Failed to download video from URL.")
                video_path = video_files[0]
            else:
                raise ValueError("Either a video URL or a local video file must be provided.")

            # Use the provided trigger word if given; otherwise auto-generate.
            tw = trigger_word.strip() if trigger_word.strip() else extract_title_and_trigger(title)[1]
            
            # Create a directory for clip extraction.
            clips_dir = temp_dir / "clips"
            os.makedirs(clips_dir, exist_ok=True)
            
            # Create debug directory if debug mode is enabled
            debug_dir = None
            if debug_mode == "Yes":
                debug_dir = temp_dir / "debug"
                os.makedirs(debug_dir, exist_ok=True)
                logger.info(f"Debug mode enabled, saving debug info to {debug_dir}")
                
                # Save a copy of the command line used
                with open(debug_dir / "ffmpeg_path.txt", "w") as f:
                    f.write(f"FFmpeg path: {ffmpeg_path}\n")
                    
                # Save video info
                try:
                    video_info_cmd = [
                        ffmpeg_path,
                        "-i", str(video_path),
                        "-hide_banner"
                    ]
                    video_info = subprocess.run(video_info_cmd, capture_output=True, text=True)
                    with open(debug_dir / "video_info.txt", "w") as f:
                        f.write(f"Video path: {video_path}\n")
                        f.write(f"Video info stderr:\n{video_info.stderr}\n")
                        f.write(f"Video info stdout:\n{video_info.stdout}\n")
                except Exception as e:
                    logger.error(f"Error getting video info: {e}")

            # Open the video and compute effective duration (exclude last 20s as outro).
            with VideoFileClip(str(video_path)) as video:
                end_time = video.duration - 20
                
                if debug_mode == "Yes":
                    with open(debug_dir / "video_details.txt", "w") as f:
                        f.write(f"Video duration: {video.duration}\n")
                        f.write(f"Video size: {video.size}\n")
                        f.write(f"Video fps: {video.fps}\n")
                        f.write(f"End time for processing: {end_time}\n")

            # Detect scenes using content detection.
            logger.info("Detecting scenes...")
            scenes = detect(str(video_path), ContentDetector())
            clips_metadata = process_clips_batch(video_path, scenes, clips_dir, tw, end_time, debug_dir)

            # Select optimal device based on hardware
            device = get_optimal_device()
            logger.info(f"Using device: {device} for model inference")
            
            # Configure model parameters based on variant
            is_awq = "AWQ" in model_variant
            precision = "int8" if is_awq else "fp16" 
            
            # Choose optimal attention implementation and precision
            if torch.cuda.is_available():
                attn_impl = "flash_attention_2" if not is_awq else "eager"
                # FlashAttention only supports fp16 or bf16
                if not is_awq:
                    precision = "fp16"
            elif device == "mps":
                attn_impl = "sdpa"
                # MPS acceleration works better with fp32 for non-quantized models
                if not is_awq:
                    precision = "fp32"
            else:
                attn_impl = "eager" if is_awq else "sdpa"
                
            logger.info(f"Using attention implementation: {attn_impl} with precision: {precision}")
            
            # Generate captions for each clip.
            captions = []
            for clip_meta in clips_metadata:
                clip_file = clips_dir / clip_meta["file_name"]
                try:
                    # Extract the first frame
                    pil_image = get_clip_frame(clip_file)
                    
                    # Default caption if extraction fails
                    caption_text = f"A scene from {title}"
                    
                    if pil_image is not None and autocaption:
                        # Format image for the model
                        # Resize while maintaining aspect ratio
                        width, height = pil_image.size
                        max_dim = max(width, height)
                        scale = image_size / max_dim
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                        
                        # Create a square canvas
                        square_img = Image.new('RGB', (image_size, image_size), (0, 0, 0))
                        paste_x = (image_size - new_width) // 2
                        paste_y = (image_size - new_height) // 2
                        square_img.paste(pil_image, (paste_x, paste_y))
                        
                        # Save debug image if debug mode is enabled
                        if debug_dir is not None:
                            try:
                                debug_img_path = debug_dir / f"{clip_meta['id']}_processed.jpg"
                                square_img.save(debug_img_path)
                                logger.info(f"Saved debug image to {debug_img_path}")
                            except Exception as e:
                                logger.error(f"Error saving debug image: {e}")
                        
                        # Verify image before sending to the model
                        images_to_send = []
                        if square_img is not None and square_img.width >= 10 and square_img.height >= 10:
                            images_to_send = [square_img]
                        else:
                            logger.warning(f"Invalid image for {clip_meta['file_name']}, using text-only caption")
                        
                        # Use model for image captioning
                        user_prompt = "Generate a detailed caption for this video frame. Describe the visual composition, lighting, colors, and mood."
                        
                        try:
                            # Save debug info before model call
                            if debug_dir is not None:
                                with open(debug_dir / f"{clip_meta['id']}_request.txt", "w") as f:
                                    f.write(f"Model: {model_variant}\n")
                                    f.write(f"System message: {system_message}\n")
                                    f.write(f"User prompt: {user_prompt}\n")
                                    f.write(f"Precision: {precision}\n")
                                    f.write(f"Attention: {attn_impl}\n")
                                    f.write(f"Image count: {len(images_to_send)}\n")
                            
                            # Send request to the transformers manager
                            response = run_async(
                                _transformers_manager.send_transformers_request(
                                    model_name=model_variant,
                                    system_message=system_message,
                                    user_prompt=user_prompt,
                                    messages=[],
                                    max_tokens=100,
                                    images=images_to_send,
                                    temperature=0.7,
                                    top_p=0.9,
                                    top_k=50,
                                    stop_string="\n",
                                    repetition_penalty=1.1,
                                    seed=42,
                                    keep_alive=True,
                                    precision=precision,
                                    attention=attn_impl
                                )
                            )
                            
                            # Save debug response
                            if debug_dir is not None:
                                with open(debug_dir / f"{clip_meta['id']}_response.txt", "w") as f:
                                    f.write(f"Raw response type: {type(response)}\n")
                                    f.write(f"Raw response: {response}\n")
                            
                            # Parse the response - handle both string and dict responses
                            if isinstance(response, dict):
                                if "response" in response:
                                    caption_text = response["response"]
                                elif "content" in response:
                                    caption_text = response["content"]
                                elif "text" in response:
                                    caption_text = response["text"]
                                elif "generated_text" in response:
                                    caption_text = response["generated_text"]
                                elif "choices" in response and len(response["choices"]) > 0:
                                    # Handle OpenAI-like response format
                                    choice = response["choices"][0]
                                    if isinstance(choice, dict):
                                        if "message" in choice and "content" in choice["message"]:
                                            caption_text = choice["message"]["content"]
                                        elif "text" in choice:
                                            caption_text = choice["text"]
                                        else:
                                            caption_text = str(choice)
                                    else:
                                        caption_text = str(choice)
                                elif "error" in response:
                                    logger.error(f"Error from model: {response['error']}")
                                    caption_text = f"A scene from {title}"
                                else:
                                    # Try to extract any string value from the dict
                                    for key, value in response.items():
                                        if isinstance(value, str) and len(value) > 10:
                                            caption_text = value
                                            break
                                    else:
                                        caption_text = str(response)
                            elif isinstance(response, str):
                                caption_text = response
                            else:
                                caption_text = str(response)
                            
                            # Clean up the caption
                            caption_text = caption_text.strip()
                            caption_text = re.sub(r'^(###\s*)?(?:Assistant|AI):\s*', '', caption_text, flags=re.IGNORECASE)
                            caption_text = caption_text.replace("<|im_end|>", "").strip()
                            caption_text = caption_text.replace("[Note: Generated without image due to processing error]", "").strip()
                            
                            # Save processed caption
                            if debug_dir is not None:
                                with open(debug_dir / f"{clip_meta['id']}_caption.txt", "w") as f:
                                    f.write(f"Processed caption: {caption_text}\n")
                            
                            # If the caption is empty or too short, use a fallback
                            if len(caption_text) < 10:
                                logger.warning(f"Caption too short for {clip_meta['file_name']}, using fallback")
                                caption_text = f"A scene from {title}"
                                
                            logger.info(f"Generated caption for {clip_meta['file_name']}: {caption_text[:50]}...")
                        except Exception as e:
                            logger.error(f"Error during model inference for {clip_meta['file_name']}: {e}")
                            caption_text = f"A scene from {title}"
                            
                            # Save error info
                            if debug_dir is not None:
                                with open(debug_dir / f"{clip_meta['id']}_error.txt", "w") as f:
                                    f.write(f"Error: {e}\n")
                                    import traceback
                                    f.write(f"Traceback: {traceback.format_exc()}\n")
                    else:
                        # Use custom caption if autocaption is disabled or frame extraction failed
                        if custom_caption.strip():
                            caption_text = custom_caption.strip()
                        else:
                            caption_text = f"A scene from {title}"
                    
                    # Add prefix/suffix if provided
                    if autocaption_prefix.strip():
                        caption_text = autocaption_prefix.strip() + " " + caption_text
                    if autocaption_suffix.strip():
                        caption_text = caption_text + " " + autocaption_suffix.strip()
                    
                    captions.append((clip_meta["id"], caption_text))
                
                except Exception as e:
                    logger.error(f"Error processing caption for {clip_meta['file_name']}: {e}")
                    # Add a fallback caption
                    captions.append((clip_meta["id"], f"A scene from {title}"))

            # Save captions to text files.
            captions_dir = temp_dir / "captions"
            os.makedirs(captions_dir, exist_ok=True)
            for clip_id, text in captions:
                caption_path = captions_dir / f"{clip_id}.txt"
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(text)

            # Organize final dataset: create a folder with subdirectories "videos" and "captions".
            dataset_name = f"{slugify(title)}_{tw}_dataset"
            dataset_dir = output_dir / dataset_name
            videos_dir = dataset_dir / "videos"
            final_captions_dir = dataset_dir / "captions"
            os.makedirs(videos_dir, exist_ok=True)
            os.makedirs(final_captions_dir, exist_ok=True)
            
            # Move all clip files.
            for clip_file in clips_dir.glob("*.mp4"):
                shutil.copy2(str(clip_file), str(videos_dir / clip_file.name))
                
            # Move caption files.
            for txt_file in captions_dir.glob("*.txt"):
                shutil.copy2(str(txt_file), str(final_captions_dir / txt_file.name))
                
            # Optionally include the original processed video.
            shutil.copy2(str(video_path), str(videos_dir / video_path.name))

            # Create a zip archive of the dataset.
            zip_path = output_dir / f"{dataset_name}.zip"
            shutil.make_archive(str(zip_path.with_suffix('')), 'zip', dataset_dir)

            # Cleanup temporary directory.
            shutil.rmtree(temp_dir)

            metadata = {
                "dataset_name": dataset_name,
                "zip_path": str(zip_path),
                "videos_dir": str(videos_dir),
                "captions_dir": str(final_captions_dir),
                "number_of_clips": len(clips_metadata),
                "model_used": model_variant,
                "device_used": device
            }
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing video dataset: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Error processing video dataset: {e}"}

NODE_CLASS_MAPPINGS = {
    "IF_HyDatasetMkr": IF_HyDatasetMkr,
}

__all__ = ['NODE_CLASS_MAPPINGS']