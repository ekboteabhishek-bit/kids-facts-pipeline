"""
Fal AI REST API Helper
Direct HTTP requests without fal-client (Python 3.7 compatible)
"""

import os
import time
import requests
from typing import Dict, Any, Optional

FAL_API_BASE = "https://queue.fal.run"


def get_fal_key():
    """Get Fal API key from environment, .env file, or config."""
    key = os.environ.get("FAL_KEY")
    if not key:
        # Try loading from .env file
        try:
            from pathlib import Path
            env_path = Path(__file__).parent / '.env'
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line.startswith('FAL_KEY='):
                        key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass
    if not key:
        # Fallback: try config.json
        try:
            import json
            from pathlib import Path
            config_path = Path(__file__).parent / 'config.json'
            with open(config_path) as f:
                config = json.load(f)
                key = config.get('fal_api_key')
        except Exception:
            pass
    return key


def get_headers():
    """Get authentication headers for Fal API."""
    key = get_fal_key()
    if not key or key == "YOUR_FAL_API_KEY_HERE":
        raise ValueError("FAL_KEY not configured. Set FAL_KEY environment variable or update config.json")
    return {
        "Authorization": "Key " + key,
        "Content-Type": "application/json"
    }


def submit_request(model, inputs):
    """
    Submit a request to Fal AI queue.
    Returns request_id for polling.
    """
    url = FAL_API_BASE + "/" + model
    response = requests.post(url, json=inputs, headers=get_headers())
    response.raise_for_status()
    data = response.json()
    return data.get("request_id")


def get_status(model, request_id):
    """Check status of a queued request."""
    url = FAL_API_BASE + "/" + model + "/requests/" + request_id + "/status"
    response = requests.get(url, headers=get_headers())
    response.raise_for_status()
    return response.json()


def get_result(model, request_id):
    """Get result of a completed request."""
    url = FAL_API_BASE + "/" + model + "/requests/" + request_id
    response = requests.get(url, headers=get_headers())
    response.raise_for_status()
    return response.json()


def poll_until_complete(model, request_id, timeout=300, interval=5, callback=None):
    """
    Poll for request completion with timeout.
    Returns the final result.

    Args:
        model: The Fal AI model endpoint
        request_id: The request ID to poll
        timeout: Maximum seconds to wait (default 300)
        interval: Seconds between polls (default 5)
        callback: Optional function to call with status updates
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        status_data = get_status(model, request_id)
        status = status_data.get("status")

        if callback:
            callback(status, status_data)

        if status == "COMPLETED":
            return get_result(model, request_id)
        elif status == "FAILED":
            error = status_data.get("error", "Unknown error")
            raise Exception("Request failed: " + str(error))

        time.sleep(interval)

    raise TimeoutError("Request " + request_id + " timed out after " + str(timeout) + "s")


def generate_image(prompt, reference_url=None, size="1080x1920"):
    """
    Generate an image using FLUX.

    Args:
        prompt: Text prompt for image generation
        reference_url: Optional URL of reference image for style guidance
        size: Image size as "WxH" string (default 9:16 vertical)

    Returns:
        dict with 'url' key for the generated image
    """
    model = "fal-ai/flux/dev"

    width, height = size.split("x")
    inputs = {
        "prompt": prompt,
        "image_size": {"width": int(width), "height": int(height)},
        "num_inference_steps": 28,
        "guidance_scale": 7.5
    }

    if reference_url:
        # Use image-to-image for style reference
        inputs["image_url"] = reference_url
        inputs["strength"] = 0.65  # Balance between reference and prompt

    request_id = submit_request(model, inputs)
    result = poll_until_complete(model, request_id)

    images = result.get("images", [])
    if images:
        return images[0]
    return {}


def generate_image_grok(prompt, aspect_ratio="9:16", output_format="png", num_images=1):
    """
    Generate an image using Grok Imagine (xAI) via Fal queue API.

    Args:
        prompt: Text prompt (max 8000 chars)
        aspect_ratio: e.g., "9:16" for vertical (default)
        output_format: "jpeg", "png", or "webp"
        num_images: 1-4

    Returns:
        dict with 'url', 'width', 'height' keys (first image)
    """
    model = "xai/grok-imagine-image"
    inputs = {
        "prompt": prompt[:8000],
        "aspect_ratio": aspect_ratio,
        "output_format": output_format,
        "num_images": num_images
    }

    request_id = submit_request(model, inputs)
    result = poll_until_complete(model, request_id, timeout=120, interval=3)

    images = result.get("images", [])
    if images:
        return {
            'url': images[0].get('url'),
            'width': images[0].get('width'),
            'height': images[0].get('height'),
            'revised_prompt': result.get('revised_prompt')
        }
    return {}


def generate_video_from_image(image_url, prompt, duration="5"):
    """
    Generate video from image using Kling image-to-video.

    Args:
        image_url: URL of the source image
        prompt: Motion/action prompt for the video
        duration: Video duration in seconds (default "5")

    Returns:
        dict with 'url' key for the generated video
    """
    model = "fal-ai/kling-video/v1.5/pro/image-to-video"
    inputs = {
        "prompt": prompt,
        "image_url": image_url,
        "duration": duration,
        "aspect_ratio": "9:16"
    }

    request_id = submit_request(model, inputs)
    # Video generation takes longer, increase timeout
    result = poll_until_complete(model, request_id, timeout=600, interval=10)

    return result.get("video", {})


def generate_video_from_text(prompt, duration="5"):
    """
    Generate video directly from text using Kling text-to-video.

    Args:
        prompt: Text prompt for video generation
        duration: Video duration in seconds (default "5")

    Returns:
        dict with 'url' key for the generated video
    """
    model = "fal-ai/kling-video/v1.5/pro/text-to-video"
    inputs = {
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": "9:16"
    }

    request_id = submit_request(model, inputs)
    result = poll_until_complete(model, request_id, timeout=600, interval=10)

    return result.get("video", {})


def upload_image_to_fal(image_path):
    """
    Upload a local image to Fal's CDN for use as reference.

    Args:
        image_path: Local path to the image file

    Returns:
        URL of the uploaded image
    """
    # Use Fal's upload endpoint
    upload_url = "https://fal.run/fal-ai/flux/dev/upload"

    with open(image_path, 'rb') as f:
        files = {'file': f}
        headers = {"Authorization": "Key " + get_fal_key()}
        response = requests.post(upload_url, files=files, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data.get("url")

    # Fallback: use base64 encoding if upload fails
    import base64
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Determine mime type
    ext = image_path.lower().split('.')[-1]
    mime_types = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp'}
    mime = mime_types.get(ext, 'image/jpeg')

    return "data:" + mime + ";base64," + image_data
