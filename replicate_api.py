"""
Replicate API Helper
Direct HTTP requests for image generation (Python 3.7+ compatible)
"""

import os
import time
import requests
from typing import Dict, Any

REPLICATE_API_BASE = "https://api.replicate.com/v1"

# Available models
NANO_BANANA_MODEL = "google/nano-banana"
FLUX_SCHNELL_MODEL = "black-forest-labs/flux-schnell"
FLUX_DEV_MODEL = "black-forest-labs/flux-dev"


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


def generate_image(prompt: str, aspect_ratio: str = "9:16", num_outputs: int = 1, model: str = "nano-banana") -> Dict[str, Any]:
    """
    Generate an image using Replicate.

    Args:
        prompt: Text prompt for image generation
        aspect_ratio: Aspect ratio (e.g., "9:16" for vertical, "16:9" for horizontal, "1:1" for square)
        num_outputs: Number of images to generate (1-4)
        model: Model to use - "nano-banana" (default), "schnell" (fast FLUX), or "dev" (high quality FLUX)

    Returns:
        dict with 'url' key for the generated image
    """
    if model == "nano-banana":
        model_version = NANO_BANANA_MODEL
        input_data = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
        }
    elif model == "schnell":
        model_version = FLUX_SCHNELL_MODEL
        input_data = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_outputs": num_outputs,
            "output_format": "png",
            "output_quality": 90,
        }
    else:  # dev
        model_version = FLUX_DEV_MODEL
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


def generate_image_flux(prompt: str, aspect_ratio: str = "9:16", model: str = "nano-banana") -> Dict[str, Any]:
    """
    Generate an image using Replicate.

    Args:
        prompt: Text prompt (detailed prompts work best)
        aspect_ratio: e.g., "9:16" for vertical shorts
        model: "nano-banana" (default), "schnell" (fast FLUX), or "dev" (high quality FLUX)

    Returns:
        dict with 'url', 'width', 'height' keys
    """
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
