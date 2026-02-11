"""
Replicate API Helper
Direct HTTP requests for image generation (Python 3.7+ compatible)
"""

import os
import time
import requests
from typing import Dict, Any

REPLICATE_API_BASE = "https://api.replicate.com/v1"

# Available models with costs (per image from Replicate pricing)
MODELS = {
    "flux-schnell": {
        "id": "black-forest-labs/flux-schnell",
        "name": "FLUX Schnell",
        "cost": 0.003,
        "description": "Fast & cheap, good quality"
    },
    "flux-dev": {
        "id": "black-forest-labs/flux-dev",
        "name": "FLUX Dev",
        "cost": 0.025,
        "description": "Highest FLUX quality, slower"
    },
    "seedream-4": {
        "id": "bytedance/seedream-4",
        "name": "Seedream 4",
        "cost": 0.03,
        "description": "High quality, photorealistic"
    },
    "nano-banana": {
        "id": "google/nano-banana",
        "name": "Nano Banana",
        "cost": 0.039,
        "description": "Google Gemini, great quality"
    },
    "seedream-4.5": {
        "id": "bytedance/seedream-4.5",
        "name": "Seedream 4.5",
        "cost": 0.04,
        "description": "Best quality, latest model"
    }
}

DEFAULT_MODEL = "flux-schnell"

# Image-to-Video model
I2V_MODEL = "wan-video/wan-2.2-i2v-fast"
I2V_COST_480P = 0.05
I2V_COST_720P = 0.11

# Legacy constants for compatibility
NANO_BANANA_MODEL = "google/nano-banana"
FLUX_SCHNELL_MODEL = "black-forest-labs/flux-schnell"
FLUX_DEV_MODEL = "black-forest-labs/flux-dev"


def get_available_models():
    """Return list of available models with metadata."""
    return [
        {"key": key, **info}
        for key, info in MODELS.items()
    ]


def get_replicate_key():
    """Get Replicate API key from environment or .env file."""
    key = os.environ.get("REPLICATE_KEY")
    if not key:
        # Try loading from .env file (check parent directory too)
        try:
            from pathlib import Path
            for env_path in [Path(__file__).parent / '.env', Path(__file__).parent.parent / '.env']:
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        line = line.strip()
                        if line.startswith('REPLICATE_KEY='):
                            key = line.split('=', 1)[1].strip().strip('"').strip("'")
                            break
                if key:
                    break
        except Exception:
            pass
    return key


def get_headers():
    """Get authentication headers for Replicate API."""
    key = get_replicate_key()
    if not key:
        raise ValueError("REPLICATE_KEY not configured. Set REPLICATE_KEY in .env file")
    return {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
        "Prefer": "wait"  # Use sync mode when possible
    }


def create_prediction(model_version: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a prediction on Replicate.
    Returns the prediction object with id for polling.
    """
    url = f"{REPLICATE_API_BASE}/models/{model_version}/predictions"

    payload = {"input": input_data}

    response = requests.post(url, json=payload, headers=get_headers())
    response.raise_for_status()
    return response.json()


def get_prediction(prediction_id: str) -> Dict[str, Any]:
    """Get the status of a prediction."""
    url = f"{REPLICATE_API_BASE}/predictions/{prediction_id}"
    response = requests.get(url, headers=get_headers())
    response.raise_for_status()
    return response.json()


def poll_until_complete(prediction_id: str, timeout: int = 300, interval: int = 2, callback=None) -> Dict[str, Any]:
    """
    Poll for prediction completion with timeout.
    Returns the final prediction result.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        prediction = get_prediction(prediction_id)
        status = prediction.get("status")

        if callback:
            callback(status, prediction)

        if status == "succeeded":
            return prediction
        elif status == "failed":
            error = prediction.get("error", "Unknown error")
            raise Exception("Prediction failed: " + str(error))
        elif status == "canceled":
            raise Exception("Prediction was canceled")

        time.sleep(interval)

    raise TimeoutError(f"Prediction {prediction_id} timed out after {timeout}s")


def generate_image(prompt: str, aspect_ratio: str = "9:16", num_outputs: int = 1, model: str = None) -> Dict[str, Any]:
    """
    Generate an image using Replicate.

    Args:
        prompt: Text prompt for image generation
        aspect_ratio: Aspect ratio (e.g., "9:16" for vertical, "16:9" for horizontal, "1:1" for square)
        num_outputs: Number of images to generate (1-4)
        model: Model key from MODELS dict (default: flux-schnell)

    Returns:
        dict with 'url' key for the generated image
    """
    if model is None:
        model = DEFAULT_MODEL

    # Handle legacy model names
    if model == "schnell":
        model = "flux-schnell"
    elif model == "dev":
        model = "flux-dev"

    # Get model config
    model_config = MODELS.get(model, MODELS[DEFAULT_MODEL])
    model_version = model_config["id"]

    # Build input based on model type
    if model == "nano-banana":
        input_data = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
        }
    elif model in ["seedream-4", "seedream-4.5"]:
        input_data = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_outputs": num_outputs,
        }
    elif model == "flux-schnell":
        input_data = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_outputs": num_outputs,
            "output_format": "png",
            "output_quality": 90,
        }
    else:  # flux-dev or unknown
        input_data = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_outputs": num_outputs,
            "output_format": "png",
            "output_quality": 90,
            "num_inference_steps": 28,
            "guidance": 3.5,
        }

    try:
        prediction = create_prediction(model_version, input_data)

        if prediction.get("status") == "succeeded":
            output = prediction.get("output", [])
            if output:
                return {"url": output[0] if isinstance(output, list) else output}
            return {}

        result = poll_until_complete(prediction["id"], timeout=120, interval=2)
        output = result.get("output", [])
        if output:
            return {"url": output[0] if isinstance(output, list) else output}
        return {}

    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")


def generate_image_flux(prompt: str, aspect_ratio: str = "9:16", model: str = None) -> Dict[str, Any]:
    """
    Generate an image using Replicate.

    Args:
        prompt: Text prompt (detailed prompts work best)
        aspect_ratio: e.g., "9:16" for vertical shorts
        model: Model key (default: flux-schnell)

    Returns:
        dict with 'url', 'width', 'height' keys
    """
    if model is None:
        model = DEFAULT_MODEL
    result = generate_image(prompt, aspect_ratio, 1, model)

    # Parse dimensions from aspect ratio
    if aspect_ratio == "9:16":
        width, height = 1080, 1920
    elif aspect_ratio == "16:9":
        width, height = 1920, 1080
    elif aspect_ratio == "1:1":
        width, height = 1024, 1024
    elif aspect_ratio == "4:3":
        width, height = 1024, 768
    elif aspect_ratio == "3:4":
        width, height = 768, 1024
    else:
        width, height = 1080, 1920  # Default vertical

    if result.get("url"):
        return {
            "url": result["url"],
            "width": width,
            "height": height
        }
    return {}


def generate_video_from_image(image_url: str, prompt: str, resolution: str = "720p", num_frames: int = 81) -> Dict[str, Any]:
    """
    Convert an image to video using wan-2.2-i2v-fast.

    Args:
        image_url: URL of the source image
        prompt: Motion/action prompt for the video
        resolution: "480p" ($0.05) or "720p" ($0.11)
        num_frames: Number of frames (81 recommended, 81-121 range)

    Returns:
        dict with 'url' key for the generated video
    """
    input_data = {
        "image": image_url,
        "prompt": prompt,
        "num_frames": num_frames,
        "resolution": resolution,
        "frames_per_second": 16,
        "go_fast": True,
        "sample_shift": 12
    }

    try:
        prediction = create_prediction(I2V_MODEL, input_data)

        if prediction.get("status") == "succeeded":
            output = prediction.get("output")
            if output:
                return {"url": output if isinstance(output, str) else output[0]}
            return {}

        # Video generation takes longer, increase timeout
        result = poll_until_complete(prediction["id"], timeout=300, interval=5)
        output = result.get("output")
        if output:
            return {"url": output if isinstance(output, str) else output[0]}
        return {}

    except Exception as e:
        raise Exception(f"Video generation failed: {str(e)}")
