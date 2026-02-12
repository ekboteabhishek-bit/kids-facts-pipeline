"""
Kids Facts Video Pipeline - Backend Server
Handles: Upload, Script Analysis, Kling Generation, Review, Assembly
Enhanced with: Auto-transcription, Image-first workflow, Reference images, Effects editor
"""

import os

# Add ~/bin to PATH for ffmpeg before any imports that might need it
_home_bin = os.path.expanduser('~/bin')
if os.path.exists(os.path.join(_home_bin, 'ffmpeg')):
    os.environ['PATH'] = _home_bin + ':' + os.environ.get('PATH', '')

import json
import uuid
import subprocess
import re
import tempfile
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import time
import requests as http_requests

# Try importing Whisper for local transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
    WHISPER_MODEL = None  # Lazy-loaded on first use
except ImportError:
    whisper = None
    WHISPER_AVAILABLE = False
    print("Warning: openai-whisper not installed. Auto-transcription disabled.")

# Try importing pydub for audio conversion (optional)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
    # Configure pydub to look in ~/bin for ffmpeg
    home_bin = os.path.expanduser('~/bin')
    if os.path.exists(os.path.join(home_bin, 'ffmpeg')):
        AudioSegment.converter = os.path.join(home_bin, 'ffmpeg')
        AudioSegment.ffprobe = os.path.join(home_bin, 'ffprobe')
except ImportError:
    AudioSegment = None
    PYDUB_AVAILABLE = False
    print("Warning: pydub not installed. Audio format conversion disabled.")

# Import our Replicate API helper (HTTP-based)
try:
    import replicate_api
    REPLICATE_API_AVAILABLE = True
except ImportError:
    replicate_api = None
    REPLICATE_API_AVAILABLE = False
    print("Warning: replicate_api module not found.")

# Legacy fal_api import (kept for compatibility but not used)
try:
    import fal_api
    FAL_API_AVAILABLE = True
except ImportError:
    fal_api = None
    FAL_API_AVAILABLE = False

# OpenAI for LLM-based prompt generation
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. LLM-based prompt generation disabled.")

app = Flask(__name__, static_folder='static')
CORS(app, supports_credentials=True)

# Secret key for sessions
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Simple password auth
APP_PASSWORD = os.environ.get('APP_PASSWORD', 'kidsfacts2024')

from functools import wraps
from flask import session, redirect, url_for

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        password = data.get('password', '')
        if password == APP_PASSWORD:
            session['authenticated'] = True
            if request.is_json:
                return jsonify({'success': True})
            return redirect('/')
        if request.is_json:
            return jsonify({'error': 'Invalid password'}), 401
        return '''
        <!DOCTYPE html>
        <html><head><title>Login</title>
        <style>body{font-family:system-ui;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#1a1a2e;}
        .login{background:#16213e;padding:40px;border-radius:12px;text-align:center;color:#fff;}
        input{padding:12px;font-size:16px;border:none;border-radius:6px;margin:10px 0;width:200px;}
        button{padding:12px 24px;font-size:16px;background:#4a90a4;color:#fff;border:none;border-radius:6px;cursor:pointer;}
        button:hover{background:#3a7a94;}.error{color:#ff6b6b;}</style></head>
        <body><div class="login"><h2>Kids Facts Pipeline</h2><p class="error">Invalid password</p>
        <form method="post"><input type="password" name="password" placeholder="Password" autofocus>
        <br><button type="submit">Login</button></form></div></body></html>
        '''
    return '''
    <!DOCTYPE html>
    <html><head><title>Login</title>
    <style>body{font-family:system-ui;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#1a1a2e;}
    .login{background:#16213e;padding:40px;border-radius:12px;text-align:center;color:#fff;}
    input{padding:12px;font-size:16px;border:none;border-radius:6px;margin:10px 0;width:200px;}
    button{padding:12px 24px;font-size:16px;background:#4a90a4;color:#fff;border:none;border-radius:6px;cursor:pointer;}
    button:hover{background:#3a7a94;}</style></head>
    <body><div class="login"><h2>Kids Facts Pipeline</h2>
    <form method="post"><input type="password" name="password" placeholder="Password" autofocus>
    <br><button type="submit">Login</button></form></div></body></html>
    '''

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect('/login')

@app.before_request
def check_auth():
    # Allow login/logout routes without auth
    if request.path in ['/login', '/logout']:
        return None
    # Allow static files needed for login page
    if request.path.startswith('/static/'):
        return None
    # Check authentication
    if not session.get('authenticated'):
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({'error': 'Unauthorized'}), 401
        return redirect('/login')
    return None

# Load config
CONFIG_PATH = Path(__file__).parent / 'config.json'
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

# Use persistent storage on cloud platforms (Render or Railway)
PERSISTENT_DATA_DIR = None

# Check for Render
if os.environ.get('RENDER') and Path('/var/data').exists():
    PERSISTENT_DATA_DIR = Path('/var/data')
    print(f"[STORAGE] Running on Render - using {PERSISTENT_DATA_DIR}")

# Check for Railway - try multiple possible volume mount paths
elif os.environ.get('RAILWAY_ENVIRONMENT'):
    # Railway volumes can be mounted at various paths - check common ones
    possible_paths = ['/data', '/var/data', '/app/data', '/mnt/data']
    for vol_path in possible_paths:
        if Path(vol_path).exists():
            PERSISTENT_DATA_DIR = Path(vol_path)
            print(f"[STORAGE] Railway volume found at {PERSISTENT_DATA_DIR}")
            break

    if not PERSISTENT_DATA_DIR:
        # No volume found - create local storage (will be lost on redeploy but works for testing)
        PERSISTENT_DATA_DIR = Path('/app/data')
        PERSISTENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[STORAGE] WARNING: No Railway volume found! Using ephemeral storage at {PERSISTENT_DATA_DIR}")
        print(f"[STORAGE] Data WILL BE LOST on redeploy. Please attach a volume in Railway dashboard.")

if PERSISTENT_DATA_DIR:
    print(f"[STORAGE] Using persistent directory: {PERSISTENT_DATA_DIR}")
    # Override config paths to use persistent storage
    for path_key in ['uploads', 'clips', 'outputs', 'temp']:
        CONFIG['paths'][path_key] = str(PERSISTENT_DATA_DIR / path_key)
        Path(CONFIG['paths'][path_key]).mkdir(parents=True, exist_ok=True)
    PROJECTS_DB_PATH = PERSISTENT_DATA_DIR / 'projects.json'
    print(f"[STORAGE] Projects DB path: {PROJECTS_DB_PATH}")
else:
    PROJECTS_DB_PATH = Path(CONFIG['paths'].get('projects_db', 'projects.json'))
    print(f"[STORAGE] Local mode - Projects DB: {PROJECTS_DB_PATH}")

# Ensure directories exist
for path_key in ['uploads', 'clips', 'outputs', 'temp']:
    Path(CONFIG['paths'][path_key]).mkdir(parents=True, exist_ok=True)

def load_projects(verbose=True):
    """Load projects from JSON file."""
    if PROJECTS_DB_PATH.exists():
        try:
            with open(PROJECTS_DB_PATH, 'r') as f:
                projects = json.load(f)
                return projects
        except Exception as e:
            print(f"[STORAGE] ERROR loading projects: {e}")
    return {}

def save_projects():
    """Save projects to JSON file."""
    try:
        # Ensure parent directory exists
        PROJECTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROJECTS_DB_PATH, 'w') as f:
            json.dump(PROJECTS, f, indent=2, default=str)
    except Exception as e:
        print(f"[STORAGE] ERROR saving projects: {e}")

def sync_projects():
    """Reload projects from disk to sync across workers."""
    global PROJECTS
    disk_projects = load_projects()
    # Always use disk version as source of truth for multi-worker consistency
    for pid, proj in disk_projects.items():
        PROJECTS[pid] = proj
    return len(disk_projects)

def fetch_project(project_id):
    """Get a project by ID, syncing from disk if needed."""
    if project_id not in PROJECTS:
        sync_projects()
    return PROJECTS.get(project_id)

# Load existing projects on startup
PROJECTS = load_projects()

# Sync projects from disk on every API request (for multi-worker consistency)
@app.before_request
def before_request_sync():
    if request.path.startswith('/api/projects'):
        sync_projects()  # Silently sync

# In-memory transcription store
TRANSCRIPTIONS = {}


# ============== TRANSCRIPTION SERVICE ==============

def get_ffmpeg_path():
    """Get the path to ffmpeg/ffprobe, checking common locations."""
    home_bin = os.path.expanduser('~/bin')
    # Add ~/bin to PATH if ffmpeg exists there
    if os.path.exists(os.path.join(home_bin, 'ffprobe')):
        os.environ['PATH'] = home_bin + ':' + os.environ.get('PATH', '')
        return home_bin
    return None

def check_ffmpeg_available():
    """Check if ffmpeg/ffprobe is available."""
    # First try to find it in ~/bin
    get_ffmpeg_path()
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

FFMPEG_AVAILABLE = check_ffmpeg_available()


def convert_audio_to_wav(audio_path):
    """Convert audio file to WAV format for SpeechRecognition."""
    if not PYDUB_AVAILABLE:
        raise Exception("pydub not installed - cannot convert audio format")

    if not FFMPEG_AVAILABLE:
        raise Exception("FFmpeg not installed - cannot convert audio format. Please upload a WAV file instead.")

    audio = AudioSegment.from_file(audio_path)
    # Convert to mono, 16kHz for speech recognition
    audio = audio.set_channels(1).set_frame_rate(16000)

    wav_path = tempfile.mktemp(suffix=".wav")
    audio.export(wav_path, format="wav")
    return wav_path


def get_whisper_model():
    """Lazy-load Whisper model on first use. Uses 'base' model for good speed/accuracy balance."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        model_name = CONFIG.get('transcription', {}).get('whisper_model', 'base')
        print(f"Loading Whisper model '{model_name}'... (this may take a moment on first run)")
        WHISPER_MODEL = whisper.load_model(model_name)
        print(f"Whisper model '{model_name}' loaded successfully")
    return WHISPER_MODEL


def transcribe_audio_openai(audio_path, language="en"):
    """
    Transcribe audio using OpenAI's Whisper API (cloud-based).
    Returns (full_text, confidence) tuple.
    """
    if not OPENAI_AVAILABLE:
        raise Exception("OpenAI not available for cloud transcription")

    api_key = CONFIG.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise Exception("OpenAI API key not configured")

    client = OpenAI(api_key=api_key)

    print(f"Transcribing with OpenAI Whisper API...")
    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language.split('-')[0] if language else "en"
        )

    text = result.text.strip() if result.text else ""
    confidence = 0.95  # OpenAI API doesn't return confidence, assume high

    print(f"OpenAI transcription complete: {len(text)} chars")
    return text, confidence


def transcribe_audio_file(audio_path, chunk_duration=30, language="en-US"):
    """
    Transcribe an audio file using OpenAI Whisper API.
    Returns (full_text, average_confidence) tuple.
    """
    # Debug logging
    print(f"[TRANSCRIPTION DEBUG] OPENAI_AVAILABLE: {OPENAI_AVAILABLE}")
    print(f"[TRANSCRIPTION DEBUG] WHISPER_AVAILABLE: {WHISPER_AVAILABLE}")
    api_key = CONFIG.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
    print(f"[TRANSCRIPTION DEBUG] API key configured: {bool(api_key)}")
    print(f"[TRANSCRIPTION DEBUG] API key from config: {bool(CONFIG.get('openai_api_key'))}")
    print(f"[TRANSCRIPTION DEBUG] API key from env: {bool(os.environ.get('OPENAI_API_KEY'))}")

    # Validate file format
    file_ext = audio_path.lower().split('.')[-1]
    if file_ext not in ['wav', 'mp3', 'm4a', 'ogg', 'flac', 'aac', 'wma', 'webm']:
        raise Exception("Unsupported audio format: " + file_ext)

    # ALWAYS use OpenAI Whisper API - no fallback to slow local Whisper
    if not OPENAI_AVAILABLE:
        raise Exception("OpenAI package not installed. Cannot transcribe.")

    if not api_key:
        raise Exception("OPENAI_API_KEY environment variable not set. Cannot transcribe.")

    print("[TRANSCRIPTION] Using OpenAI Whisper API (cloud)")
    return transcribe_audio_openai(audio_path, language)


def run_transcription(transcription_id, audio_path):
    """Background task to run transcription."""
    try:
        TRANSCRIPTIONS[transcription_id]['status'] = 'processing'

        language = CONFIG.get('transcription', {}).get('language', 'en-US')
        chunk_duration = CONFIG.get('transcription', {}).get('chunk_duration_seconds', 30)

        text, confidence = transcribe_audio_file(audio_path, chunk_duration, language)

        TRANSCRIPTIONS[transcription_id]['status'] = 'completed'
        TRANSCRIPTIONS[transcription_id]['text'] = text
        TRANSCRIPTIONS[transcription_id]['confidence'] = confidence

    except Exception as e:
        TRANSCRIPTIONS[transcription_id]['status'] = 'failed'
        TRANSCRIPTIONS[transcription_id]['error'] = str(e)


# ============== EFFECTS MANAGEMENT ==============

def generate_effect_id():
    """Generate unique effect ID."""
    return "eff_" + uuid.uuid4().hex[:8]


def detect_effects_for_shot(shot, start_frame, fps=30):
    """
    Auto-detect Remotion effects for a shot based on content analysis.
    Assigns: kinetic_text, emoji pops, particles, screen_shake, flash, zoom_pulse.
    Multiple effects can stack per shot for rich motion graphics.
    """
    effects = []
    phrase = shot['phrase']
    phrase_lower = phrase.lower()
    shot_frames = int(shot['duration'] * fps)

    effect_config = CONFIG.get('effects', {}).get('types', {})

    # ---- KINETIC TEXT: Add to every shot for word-by-word text reveal ----
    # Extract 1-3 highlight words (most impactful words in the phrase)
    highlight_words = extract_highlight_words(phrase)
    effects.append({
        'id': generate_effect_id(),
        'type': 'kinetic_text',
        'enabled': True,
        'trigger_phrase': phrase[:30],
        'frame': start_frame,
        'parameters': {
            'text': phrase,
            'highlight_words': highlight_words,
            'words_per_second': max(3, len(phrase.split()) / max(shot['duration'], 1)),
            'duration': shot_frames
        }
    })

    # ---- EXPLOSION / IMPACT: screen shake + flash + particles + emoji ----
    if any(word in phrase_lower for word in ['boom', 'explod', 'burst', 'pop', 'crash', 'smash', 'blast']):
        trigger = next((w for w in ['boom', 'explod', 'burst', 'pop', 'crash', 'smash'] if w in phrase_lower), 'impact')

        effects.append({
            'id': generate_effect_id(),
            'type': 'screen_shake',
            'enabled': True,
            'trigger_phrase': trigger,
            'frame': start_frame,
            'parameters': {
                'intensity': effect_config.get('screen_shake', {}).get('default_intensity', 12),
                'duration': effect_config.get('screen_shake', {}).get('default_duration', 15)
            }
        })

        effects.append({
            'id': generate_effect_id(),
            'type': 'flash',
            'enabled': True,
            'trigger_phrase': trigger,
            'frame': start_frame,
            'parameters': {
                'duration': effect_config.get('flash', {}).get('default_duration', 6),
                'color': 'white'
            }
        })

        effects.append({
            'id': generate_effect_id(),
            'type': 'particles',
            'enabled': True,
            'trigger_phrase': trigger,
            'frame': start_frame,
            'parameters': {
                'x': 50, 'y': 50,
                'count': effect_config.get('particles', {}).get('default_count', 15),
                'duration': effect_config.get('particles', {}).get('default_duration', 25)
            }
        })

    # ---- NUMBERS / STATISTICS: zoom pulse + number counter ----
    numbers = re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?%?)\b', phrase)
    if numbers:
        effects.append({
            'id': generate_effect_id(),
            'type': 'zoom_pulse',
            'enabled': True,
            'trigger_phrase': numbers[0],
            'frame': start_frame + 5,
            'parameters': {
                'intensity': 1.08,
                'duration': 18
            }
        })

        # Add number counter for the first number found
        num_str = numbers[0].replace(',', '').replace('%', '')
        try:
            num_val = float(num_str)
            suffix = '%' if '%' in numbers[0] else ''
            effects.append({
                'id': generate_effect_id(),
                'type': 'number_counter',
                'enabled': True,
                'trigger_phrase': numbers[0],
                'frame': start_frame,
                'parameters': {
                    'start': 0,
                    'end': num_val,
                    'suffix': suffix,
                    'duration': min(shot_frames, 30)
                }
            })
        except ValueError:
            pass

    # ---- QUESTION / CURIOSITY: emoji + subtle zoom ----
    elif '?' in phrase or any(w in phrase_lower for w in ['why', 'how', 'what if']):
        effects.append({
            'id': generate_effect_id(),
            'type': 'emoji',
            'enabled': True,
            'trigger_phrase': '?',
            'frame': start_frame + 5,
            'parameters': {
                'emoji': '🤔',
                'x': 75, 'y': 20,
                'duration': min(shot_frames, 25)
            }
        })

    # ---- POSITIVE / EXCITING: emoji + particles ----
    elif any(word in phrase_lower for word in ['amazing', 'incredible', 'awesome', 'cool', 'wow', 'love', 'beautiful']):
        trigger = next((w for w in ['amazing', 'incredible', 'awesome'] if w in phrase_lower), 'wow')

        emoji_map = {
            'love': '❤️', 'beautiful': '✨', 'cool': '🔥',
            'wow': '🤯', 'amazing': '⭐', 'incredible': '🌟', 'awesome': '💪'
        }
        emoji = emoji_map.get(trigger, '✨')

        effects.append({
            'id': generate_effect_id(),
            'type': 'emoji',
            'enabled': True,
            'trigger_phrase': trigger,
            'frame': start_frame + 8,
            'parameters': {
                'emoji': emoji,
                'x': 70, 'y': 25,
                'duration': min(shot_frames, 25)
            }
        })

        effects.append({
            'id': generate_effect_id(),
            'type': 'particles',
            'enabled': True,
            'trigger_phrase': trigger,
            'frame': start_frame + 3,
            'parameters': {
                'x': 50, 'y': 50,
                'count': 10,
                'duration': min(shot_frames, 20)
            }
        })

    # ---- SCARY / DANGER: flash + shake ----
    elif any(word in phrase_lower for word in ['danger', 'scary', 'deadly', 'toxic', 'poison', 'die', 'kill']):
        trigger = next((w for w in ['danger', 'scary', 'deadly', 'toxic'] if w in phrase_lower), 'danger')
        effects.append({
            'id': generate_effect_id(),
            'type': 'flash',
            'enabled': True,
            'trigger_phrase': trigger,
            'frame': start_frame,
            'parameters': {'duration': 4, 'color': 'red'}
        })
        effects.append({
            'id': generate_effect_id(),
            'type': 'screen_shake',
            'enabled': True,
            'trigger_phrase': trigger,
            'frame': start_frame + 2,
            'parameters': {'intensity': 8, 'duration': 10}
        })

    # ---- REVEAL / SURPRISE: zoom pulse ----
    elif any(word in phrase_lower for word in ['actually', 'turns out', 'secret', 'hidden', 'reveal']):
        trigger = next((w for w in ['actually', 'turns out', 'secret'] if w in phrase_lower), 'reveal')
        effects.append({
            'id': generate_effect_id(),
            'type': 'zoom_pulse',
            'enabled': True,
            'trigger_phrase': trigger,
            'frame': start_frame,
            'parameters': {'intensity': 1.1, 'duration': 20}
        })

    return effects


def extract_highlight_words(phrase):
    """Extract 1-3 most impactful words from a phrase for kinetic text highlighting."""
    words = phrase.split()
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                  'could', 'should', 'can', 'it', 'its', 'this', 'that', 'they',
                  'them', 'their', 'and', 'or', 'but', 'so', 'to', 'of', 'in',
                  'on', 'at', 'by', 'for', 'with', 'not', 'just', 'very', 'really'}

    # Prioritize: numbers, all-caps words, long words, emotional words
    impact_words = ['boom', 'explode', 'million', 'billion', 'incredible', 'amazing',
                    'never', 'always', 'only', 'biggest', 'smallest', 'fastest',
                    'hottest', 'coldest', 'deadly', 'dangerous', 'secret', 'hidden',
                    'actually', 'impossible', 'massive', 'tiny', 'ancient', 'first', 'last']

    scored = []
    for w in words:
        clean = w.strip('.,!?;:').lower()
        if clean in stop_words or len(clean) <= 2:
            continue
        score = 0
        if re.match(r'\d', clean):
            score += 5
        if clean in impact_words:
            score += 4
        if len(clean) >= 7:
            score += 2
        score += 1  # base score for content words
        scored.append((w.strip('.,!?;:'), score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w, s in scored[:3]]


def initialize_shot_effects(project):
    """Initialize effects for all shots in a project based on auto-detection."""
    fps = 30
    # Get actual hook duration if available
    hook_duration = 10  # default fallback
    if project.get('hook_path') and os.path.exists(project['hook_path']):
        try:
            hook_duration = get_video_duration(project['hook_path'])
        except Exception:
            pass
    current_frame = int(hook_duration * fps)

    for shot in project['shots']:
        start_frame = current_frame

        # Detect and assign effects
        shot['effects'] = detect_effects_for_shot(shot, start_frame, fps)

        current_frame += int(shot['duration'] * fps)


# ============== IMAGE GENERATION ==============

def generate_image_for_shot(project_id, shot_id, custom_prompt=None, retry_count=0):
    """
    Generate an image for a shot using FLUX via Replicate.
    Updates shot state as generation progresses.
    Retries with modified prompt on NSFW filter errors.
    """
    if not REPLICATE_API_AVAILABLE:
        raise Exception("replicate_api module not available")

    print(f"[IMAGE GEN] Starting image generation for project {project_id}, shot {shot_id}" + (f" (retry {retry_count})" if retry_count > 0 else ""))

    # Always use PROJECTS[project_id] directly to avoid reference orphaning
    def get_shot():
        return PROJECTS[project_id]['shots'][shot_id - 1]

    try:
        get_shot()['image_status'] = 'generating'
        save_projects()

        # Use custom prompt or full prompt
        shot = get_shot()
        base_prompt = custom_prompt if custom_prompt else shot['full_prompt']

        # Add project's image style to the prompt
        image_style = PROJECTS[project_id].get('image_style', '')
        if image_style:
            prompt = f"{image_style}. {base_prompt}"
        else:
            prompt = base_prompt

        # On retry, add child-friendly prefix to avoid NSFW filter
        if retry_count > 0:
            safe_prefix = "Child-friendly, educational illustration, safe for all ages, colorful and wholesome: "
            prompt = safe_prefix + prompt
            print(f"[IMAGE GEN] Shot {shot_id}: Using safe-prefix prompt")

        # Get model from project (default: flux-schnell)
        image_model = PROJECTS[project_id].get('image_model', 'flux-schnell')

        print(f"[IMAGE GEN] Shot {shot_id}: Using model {image_model}")

        # Generate image using selected model on Replicate
        result = replicate_api.generate_image_flux(prompt, aspect_ratio="9:16", model=image_model)

        if result and result.get('url'):
            image_url = result['url']
            print(f"[IMAGE GEN] Shot {shot_id}: Got image URL: {image_url[:50]}...")

            # Download and save image locally
            image_filename = project_id + "_shot_" + str(shot_id).zfill(2) + "_image.png"
            image_path = Path(CONFIG['paths']['clips']) / image_filename

            response = http_requests.get(image_url)
            with open(image_path, 'wb') as f:
                f.write(response.content)

            get_shot()['image_status'] = 'generated'
            get_shot()['image_path'] = str(image_path)
            get_shot()['image_url'] = image_url
            save_projects()

            print(f"[IMAGE GEN] Shot {shot_id}: Image saved to {image_path}")
            return True
        else:
            get_shot()['image_status'] = 'failed'
            get_shot()['image_error'] = 'No image URL in response'
            save_projects()
            print(f"[IMAGE GEN] Shot {shot_id}: FAILED - No URL in response")
            return False

    except Exception as e:
        error_str = str(e)
        is_rate_limited = '429' in error_str or 'Too Many Requests' in error_str
        is_nsfw_error = 'NSFW' in error_str or 'nsfw' in error_str.lower()
        print(f"[IMAGE GEN] Shot {shot_id}: ERROR - {error_str}")

        # Retry with safe prompt prefix on NSFW errors (max 2 retries)
        if is_nsfw_error and retry_count < 2:
            print(f"[IMAGE GEN] Shot {shot_id}: NSFW filter triggered, retrying with safe prefix...")
            time.sleep(2)  # Brief delay before retry
            return generate_image_for_shot(project_id, shot_id, custom_prompt, retry_count + 1)

        if project_id in PROJECTS:
            get_shot()['image_status'] = 'failed'
            get_shot()['image_error'] = error_str
            get_shot()['rate_limited'] = is_rate_limited
            save_projects()
        return 'rate_limited' if is_rate_limited else False


def generate_video_from_shot_image(project_id, shot_id):
    """
    Generate video from an approved shot image using wan-2.2-i2v-fast on Replicate.
    Uses simple motion prompts for consistent, subtle animations.
    """
    if not REPLICATE_API_AVAILABLE:
        raise Exception("replicate_api module not available")

    project = PROJECTS[project_id]
    shot = project['shots'][shot_id - 1]

    if not shot.get('image_url'):
        raise Exception("Shot has no image URL")

    try:
        shot['video_status'] = 'generating'
        save_projects()

        # Generate motion prompt (simple defaults)
        motion_prompt = generate_motion_prompt(
            shot.get('text', ''),
            shot.get('full_prompt', '')
        )
        shot['motion_prompt'] = motion_prompt

        print(f"Generating video for shot {shot_id} with motion: {motion_prompt}")

        # Generate video using wan-2.2-i2v-fast on Replicate (720p, $0.11/video)
        result = replicate_api.generate_video_from_image(
            shot['image_url'],
            motion_prompt,
            resolution="720p",
            num_frames=81
        )

        if result and result.get('url'):
            video_url = result['url']

            # Download and save video locally
            clip_filename = project_id + "_shot_" + str(shot_id).zfill(2) + "_raw.mp4"
            clip_path = Path(CONFIG['paths']['clips']) / clip_filename

            response = http_requests.get(video_url)
            with open(clip_path, 'wb') as f:
                f.write(response.content)

            shot['video_status'] = 'generated'
            shot['video_url'] = video_url
            shot['clip_path'] = str(clip_path)
            save_projects()

            # Auto-trim to required duration
            trim_clip(project_id, shot_id)

            return True
        else:
            shot['video_status'] = 'failed'
            shot['video_error'] = 'No video URL in response'
            save_projects()
            return False

    except Exception as e:
        error_str = str(e)
        is_rate_limited = '429' in error_str or 'Too Many Requests' in error_str
        print(f"Error generating video: {error_str}")
        shot['video_status'] = 'failed'
        shot['video_error'] = error_str
        save_projects()
        return 'rate_limited' if is_rate_limited else False


def analyze_script(script_text, audio_duration=None, image_style=None):
    """
    Break script into timed segments and generate shot prompts.
    Returns list of shots with timing and prompts.
    image_style: Optional style string to apply to all generated prompts.
    """
    # Split into sentences/phrases
    lines = [l.strip() for l in script_text.strip().split('\n') if l.strip()]
    
    # Flatten into phrases (split on periods, exclamations, ellipsis)
    phrases = []
    for line in lines:
        # Split on sentence endings but keep the punctuation
        parts = re.split(r'([.!?…]+)', line)
        current = ""
        for i, part in enumerate(parts):
            if re.match(r'^[.!?…]+$', part):
                current += part
                if current.strip():
                    phrases.append(current.strip())
                current = ""
            else:
                if current.strip():
                    phrases.append(current.strip())
                current = part
        if current.strip():
            phrases.append(current.strip())
    
    # Filter empty and very short phrases
    phrases = [p for p in phrases if len(p) > 5]
    
    # Estimate duration per phrase (roughly 2-4 seconds each)
    num_phrases = len(phrases)
    if audio_duration:
        avg_duration = audio_duration / num_phrases
    else:
        # Estimate: ~3 words per second for narration
        avg_duration = 3.0
    
    # Clamp durations
    min_dur, max_dur = CONFIG['video']['clip_duration_range']
    
    shots = []
    current_time = 0.0
    
    for i, phrase in enumerate(phrases):
        # Calculate duration based on word count
        word_count = len(phrase.split())
        duration = max(min_dur, min(max_dur, word_count * 0.4))
        
        # Generate visual prompt based on phrase content
        prompt = generate_visual_prompt(phrase, style=image_style)
        
        # Detect SFX triggers
        sfx = detect_sfx_triggers(phrase)
        
        shots.append({
            'id': i + 1,
            'phrase': phrase,
            'start_time': round(current_time, 2),
            'duration': round(duration, 2),
            'end_time': round(current_time + duration, 2),
            'prompt': prompt,
            'full_prompt': prompt + CONFIG['style']['prompt_suffix'],
            'sfx': sfx,
            # Image generation (step 1)
            'image_status': 'pending',  # pending, generating, generated, approved, rejected
            'image_path': None,
            'image_url': None,
            'image_error': None,
            # Video generation (step 2)
            'status': 'pending',  # pending, generating, generated, approved, rejected
            'clip_path': None,
            'trimmed_path': None,
            'error': None,
            # Effects (will be populated by initialize_shot_effects)
            'effects': []
        })
        
        current_time += duration
    
    return shots


def generate_visual_prompt(phrase, style=None):
    """
    Convert a transcript phrase into a detailed, descriptive image generation prompt.
    Uses the provided style or falls back to a default aesthetic.
    """
    phrase_lower = phrase.lower()
    subject = extract_subject(phrase)

    # Use provided style or fall back to default
    if style:
        STYLE = f"{style}. Vertical 9:16 composition, single focused subject centered, clean background. No text, no watermarks, no UI elements."
    else:
        STYLE = (
            "Stylized digital art, semi-realistic digital painting. "
            "Rich saturated colors, dramatic cinematic lighting with volumetric rays, "
            "deep atmospheric depth, concept art quality. "
            "Vertical 9:16 composition, single focused subject centered, "
            "clean background with subtle gradient bokeh. "
            "No text, no watermarks, no UI elements."
        )

    # Scene descriptor based on content analysis
    scene = ""

    # Explosion / Impact
    if any(w in phrase_lower for w in ['boom', 'explode', 'burst', 'blast', 'crash', 'smash', 'destroy']):
        scene = (
            f"Dramatic explosion scene showing {subject} bursting apart in slow motion, "
            f"golden debris particles and shockwave rings expanding outward, "
            f"intense orange-red backlighting, motion blur on flying fragments"
        )

    # Inside / Cross-section
    elif any(w in phrase_lower for w in ['inside', 'interior', 'cross-section', 'within', 'core']):
        scene = (
            f"Cutaway cross-section illustration of {subject}, revealing detailed internal structure, "
            f"glowing highlighted layers with labeled depth, scientific diagram aesthetic, "
            f"cool blue-teal lighting on exposed internals"
        )

    # Tiny / Microscopic
    elif any(w in phrase_lower for w in ['tiny', 'microscopic', 'small', 'miniature', 'atom']):
        scene = (
            f"Extreme macro close-up of {subject} at microscopic scale, "
            f"intricate surface detail and texture visible, shallow depth of field, "
            f"dewdrop lens effect, warm backlight creating rim glow"
        )

    # Hot / Temperature / Fire
    elif any(w in phrase_lower for w in ['hot', 'heat', 'fire', 'burn', 'lava', 'molten', 'degrees']):
        scene = (
            f"{subject.capitalize()} surrounded by intense heat and fire, "
            f"visible heat distortion waves rising, molten orange-red glow, "
            f"embers floating upward, dramatic thermal contrast"
        )

    # Cold / Ice / Frozen
    elif any(w in phrase_lower for w in ['cold', 'frozen', 'ice', 'freeze', 'arctic', 'snow']):
        scene = (
            f"{subject.capitalize()} encased in crystalline ice formations, "
            f"frost patterns spreading outward, cool blue-white color palette, "
            f"sparkling ice particles catching light, misty cold atmosphere"
        )

    # Water / Ocean
    elif any(w in phrase_lower for w in ['water', 'ocean', 'sea', 'underwater', 'swim', 'wave', 'splash']):
        scene = (
            f"{subject.capitalize()} in a deep underwater environment, "
            f"shafts of sunlight filtering through turquoise water, "
            f"air bubbles rising, bioluminescent particles floating, caustic light patterns"
        )

    # Space / Stars
    elif any(w in phrase_lower for w in ['space', 'star', 'planet', 'galaxy', 'sun', 'moon', 'orbit', 'universe']):
        scene = (
            f"Cosmic scene showing {subject} in deep space, "
            f"surrounded by stars and nebula clouds in purple and blue, "
            f"distant galaxies in background, rim-lit by a nearby star, sense of vast scale"
        )

    # Speed / Fast
    elif any(w in phrase_lower for w in ['fast', 'speed', 'quick', 'rapid', 'race', 'zoom', 'lightning']):
        scene = (
            f"{subject.capitalize()} in high-speed motion, "
            f"intense motion blur streaks trailing behind, "
            f"dynamic speed lines, neon energy trail, wind effect on surroundings"
        )

    # Grow / Transform / Change
    elif any(w in phrase_lower for w in ['grow', 'transform', 'change', 'evolve', 'become', 'turn into', 'morph']):
        scene = (
            f"{subject.capitalize()} mid-transformation, morphing between forms, "
            f"magical particles swirling around the transition point, "
            f"ethereal glow at the edges, before-and-after visible in same frame"
        )

    # Animal
    elif any(w in phrase_lower for w in ['animal', 'creature', 'dog', 'cat', 'bird', 'fish', 'insect', 'dinosaur', 'whale', 'lion', 'elephant']):
        scene = (
            f"Majestic portrait of {subject}, "
            f"dramatic side lighting highlighting fur/scales/feathers texture, "
            f"piercing expressive eyes, natural habitat blurred in background, "
            f"golden hour warmth"
        )

    # Food
    elif any(w in phrase_lower for w in ['eat', 'food', 'cook', 'taste', 'delicious', 'recipe', 'popcorn', 'candy', 'chocolate']):
        scene = (
            f"Appetizing close-up of {subject}, "
            f"warm golden overhead lighting, steam or freshness visible, "
            f"rich textures and colors, shallow depth of field, "
            f"droplets of condensation or sauce glistening"
        )

    # Big / Massive
    elif any(w in phrase_lower for w in ['big', 'huge', 'massive', 'giant', 'enormous', 'colossal', 'tall']):
        scene = (
            f"Dramatic low-angle shot looking up at {subject} towering overhead, "
            f"a tiny human silhouette at the base for scale comparison, "
            f"dramatic clouds and golden light behind, awe-inspiring sense of size"
        )

    # Question / Mystery
    elif '?' in phrase or any(w in phrase_lower for w in ['why', 'how', 'what if', 'mystery', 'wonder', 'secret']):
        scene = (
            f"Mysterious atmospheric scene depicting {subject}, "
            f"moody volumetric fog with shafts of light breaking through, "
            f"question-mark-shaped light beam subtly visible, intrigue and curiosity"
        )

    # Number / Statistic
    elif re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?%?\b', phrase):
        scene = (
            f"Bold infographic-style illustration of {subject}, "
            f"the key quantity visualized as a dramatic physical scale comparison, "
            f"clean geometric composition, vibrant accent colors on dark background, "
            f"data-visualization inspired aesthetic"
        )

    # Default: descriptive scene from the phrase content
    else:
        scene = (
            f"Vivid illustration depicting the concept: {phrase.rstrip('.')}. "
            f"Central subject clearly visible and well-lit, "
            f"atmospheric depth with soft background gradient, "
            f"dynamic composition with slight diagonal energy"
        )

    return f"{scene}. {STYLE}"


def extract_subject(phrase):
    """Extract the main subject/noun from a phrase."""
    words = phrase.split()

    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'into', 'onto', 'to', 'from', 'with', 'without', 'and', 'or',
                  'but', 'so', 'yet', 'for', 'nor', 'every', 'each', 'there',
                  'it', 'its', 'this', 'that', 'these', 'those', 'they', 'them',
                  'their', 'can', 'not', 'very', 'really', 'just', 'even', 'about',
                  'when', 'where', 'which', 'who', 'whom', 'what', 'if', 'of', 'in',
                  'on', 'at', 'by', 'up', 'out', 'off', 'over', 'under', 'again',
                  'then', 'once', 'here', 'all', 'any', 'both', 'more', 'most',
                  'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than', 'too'}

    content_words = [w.strip('.,!?;:') for w in words if w.lower() not in stop_words and len(w) > 1]

    if len(content_words) >= 3:
        return ' '.join(content_words[:4])
    elif content_words:
        return ' '.join(content_words)
    return phrase[:40]


def detect_sfx_triggers(phrase):
    """Detect sound effect triggers in a phrase."""
    sfx_config = CONFIG.get('sfx', {})
    if not sfx_config.get('enabled', True):
        return None

    phrase_lower = phrase.lower()
    triggers = sfx_config.get('triggers', sfx_config)
    
    for keyword, sfx_file in triggers.items():
        if keyword in phrase_lower:
            return {
                'trigger': keyword,
                'file': sfx_file
            }
    return None


# ============== AUDIO-DRIVEN PIPELINE ==============

def get_audio_duration(audio_path):
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def detect_animation_type(phrase):
    """
    Detect the best animation type for a shot based on transcript content.
    Returns (animation_type, animation_params) tuple.
    10 animation types available for visual variety.
    """
    phrase_lower = phrase.lower()

    # Category 1: Exciting/Impact -> zoom pulse
    impact_words = ['boom', 'explode', 'burst', 'crash', 'smash', 'incredible',
                    'amazing', 'unbelievable', 'insane', 'mind-blowing', 'shocking',
                    'destroy', 'massive', 'enormous', 'blast', 'pow', 'bang']
    if any(w in phrase_lower for w in impact_words):
        return 'zoom_pulse', {
            'start_zoom': 1.0, 'end_zoom': 1.4,
            'pan_x': 0, 'pan_y': 0,
            'speed': 'fast',
            'easing': 'ease_out'
        }

    # Category 2: Reveal/Big picture -> pull back (zoom out)
    reveal_words = ['actually', 'turns out', 'the truth', 'revealed', 'discover',
                    'in reality', 'the whole', 'big picture', 'entire', 'all of',
                    'secret', 'hidden', 'underneath', 'behind']
    if any(w in phrase_lower for w in reveal_words):
        return 'pull_back', {
            'start_zoom': 1.5, 'end_zoom': 1.0,
            'pan_x': 0, 'pan_y': 0,
            'easing': 'ease_in_out'
        }

    # Category 3: Numbers/Facts -> slow drift
    has_numbers = bool(re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?%?\b', phrase))
    fact_words = ['percent', 'million', 'billion', 'trillion', 'times',
                  'degrees', 'miles', 'kilometers', 'pounds', 'kilograms',
                  'meters', 'feet', 'hours', 'years']
    if has_numbers or any(w in phrase_lower for w in fact_words):
        return 'slow_drift', {
            'start_zoom': 1.05, 'end_zoom': 1.05,
            'pan_x': 20, 'pan_y': 0,
            'easing': 'linear'
        }

    # Category 4: Questions/Curiosity -> tilt up (rising motion, wonder)
    if '?' in phrase or any(w in phrase_lower for w in ['why', 'how', 'what if', 'imagine', 'wonder', 'curious']):
        return 'tilt_up', {
            'start_zoom': 1.1, 'end_zoom': 1.1,
            'pan_x': 0, 'pan_y': -40,
            'easing': 'ease_in'
        }

    # Category 5: Direction/Movement -> pan left or right
    left_words = ['back', 'before', 'past', 'ago', 'history', 'remember', 'old', 'ancient']
    if any(w in phrase_lower for w in left_words):
        return 'pan_left', {
            'start_zoom': 1.1, 'end_zoom': 1.1,
            'pan_x': -40, 'pan_y': 0,
            'easing': 'ease_in_out'
        }

    right_words = ['next', 'forward', 'future', 'then', 'after', 'becomes', 'turns']
    if any(w in phrase_lower for w in right_words):
        return 'pan_right', {
            'start_zoom': 1.1, 'end_zoom': 1.1,
            'pan_x': 40, 'pan_y': 0,
            'easing': 'ease_in_out'
        }

    # Category 6: Downward reveal -> tilt down
    down_words = ['deep', 'below', 'underneath', 'ground', 'ocean', 'bottom', 'buried', 'underground']
    if any(w in phrase_lower for w in down_words):
        return 'tilt_down', {
            'start_zoom': 1.1, 'end_zoom': 1.1,
            'pan_x': 0, 'pan_y': 40,
            'easing': 'ease_in_out'
        }

    # Category 7: Focus/Close-up -> dolly push (fast zoom in)
    focus_words = ['look', 'see', 'watch', 'notice', 'inside', 'tiny', 'small',
                   'detail', 'close', 'zoom', 'focus', 'specific']
    if any(w in phrase_lower for w in focus_words):
        return 'dolly_push', {
            'start_zoom': 1.0, 'end_zoom': 1.35,
            'pan_x': 0, 'pan_y': 0,
            'easing': 'ease_in'
        }

    # Alternate between remaining types for variety
    # Use phrase length as a simple alternator
    word_count = len(phrase.split())
    alternates = [
        ('ken_burns_zoom_in', {
            'start_zoom': 1.0, 'end_zoom': 1.15,
            'pan_x': 25, 'pan_y': -15,
            'easing': 'ease_in_out'
        }),
        ('ken_burns_zoom_out', {
            'start_zoom': 1.25, 'end_zoom': 1.0,
            'pan_x': -15, 'pan_y': 10,
            'easing': 'ease_in_out'
        }),
        ('pan_right', {
            'start_zoom': 1.08, 'end_zoom': 1.08,
            'pan_x': 30, 'pan_y': -5,
            'easing': 'linear'
        }),
    ]
    return alternates[word_count % len(alternates)]


def generate_motion_prompt(shot_text: str, image_prompt: str) -> str:
    """
    Generate a contextual motion prompt by analyzing the image content.
    Uses OpenAI to understand the scene and create appropriate camera movement.
    """
    if not OPENAI_AVAILABLE:
        return _simple_motion_fallback(shot_text)

    api_key = CONFIG.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return _simple_motion_fallback(shot_text)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """You are a cinematographer creating motion prompts for image-to-video AI.

Analyze the image description and create a motion prompt that:
1. Matches the content (zoom into faces, pan across landscapes, follow action)
2. Enhances the storytelling of the narration
3. Uses cinematic camera movements appropriate for the scene

Motion types to use:
- PEOPLE/FACES: "slow push in towards the subject's face, intimate close-up feel"
- LANDSCAPES/WIDE SHOTS: "slow pan across the scene revealing details, epic scope"
- ACTION/MOVEMENT: "dynamic camera follow, energetic tracking shot"
- OBJECTS/PRODUCTS: "smooth orbit around the subject, showcase rotation"
- EMOTIONAL MOMENTS: "gentle drift with subtle zoom, contemplative mood"
- DRAMATIC REVEALS: "slow zoom out revealing the full scene, dramatic unveil"

Output ONLY the motion prompt (15-25 words), no explanations."""
            }, {
                "role": "user",
                "content": f"IMAGE DESCRIPTION: {image_prompt}\n\nNARRATION: {shot_text}\n\nGenerate the perfect camera motion for this shot:"
            }],
            max_tokens=60,
            temperature=0.7
        )
        motion = response.choices[0].message.content.strip()
        print(f"Generated motion prompt: {motion}")
        return motion
    except Exception as e:
        print(f"Motion prompt generation failed: {e}, using fallback")
        return _simple_motion_fallback(shot_text)


def _simple_motion_fallback(shot_text: str) -> str:
    """Fallback simple motion prompts when OpenAI is unavailable."""
    simple_motions = [
        "slow cinematic push in, professional documentary feel",
        "gentle zoom out revealing the full scene",
        "smooth pan from left to right, natural movement",
        "subtle camera drift with soft parallax",
        "slow Ken Burns effect, elegant and timeless"
    ]
    return simple_motions[hash(shot_text) % len(simple_motions)]


def generate_shots_with_llm(transcript_text, audio_duration, image_style=None):
    """
    Use GPT-4 to intelligently segment transcript and generate specific visual prompts.
    Each shot will be under 2 seconds with a unique, context-aware image prompt.
    """
    if not OPENAI_AVAILABLE:
        print("OpenAI not available, falling back to rule-based generation")
        return None

    api_key = CONFIG.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("OpenAI API key not configured, falling back to rule-based generation")
        return None

    try:
        client = OpenAI(api_key=api_key)

        # Calculate target number of shots based on max 2 seconds each
        max_duration = CONFIG['video']['clip_duration_range'][1]
        min_shots = int(audio_duration / max_duration) + 1

        style_description = image_style if image_style else 'cinematic, high quality, dramatic lighting'

        system_prompt = f"""You are an expert at creating visual storyboards for short-form educational videos.

Your task is to break a transcript into shots and generate specific image prompts for each shot.

RULES:
1. Each shot should cover 3-6 words of narration (roughly 1-2 seconds when spoken)
2. Split sentences at natural meaning boundaries - a single sentence often needs 2-3 shots
3. Generate SPECIFIC visual descriptions, not literal restatements of the text
4. Think about what IMAGE would best represent the concept being discussed

CRITICAL - EACH PROMPT MUST BE COMPLETELY SELF-CONTAINED:
- Each prompt will be sent to an image generation AI INDEPENDENTLY
- The image AI has NO knowledge of previous prompts or the overall video
- NEVER use phrases like "same style as before", "continuing from previous", "matching the earlier scene"
- NEVER reference other shots (e.g., "similar to shot 1", "in the same environment")
- Each prompt must include ALL necessary details: subject, setting, lighting, colors, camera angle, and style
- A prompt should make complete sense if read in isolation

STYLE TO INCLUDE IN EVERY PROMPT:
{style_description}

PROMPT WRITING GUIDELINES:
- Describe the SCENE, not the words being spoken
- Include specific visual details: lighting, camera angle, colors, composition
- For abstract concepts, create concrete visual metaphors
- Always specify "vertical 9:16 composition" for portrait format
- Include the visual style in each prompt (e.g., "{style_description}")
- End each prompt with "No text, no watermarks, no UI elements."

OUTPUT FORMAT (JSON):
{{
  "shots": [
    {{
      "phrase": "the exact transcript text for this shot",
      "prompt": "detailed visual description for image generation"
    }}
  ]
}}

The transcript is approximately {audio_duration:.1f} seconds long, so aim for {min_shots}-{min_shots + 5} shots."""

        user_prompt = f"""Break this transcript into shots and generate image prompts:

TRANSCRIPT:
{transcript_text}

Remember:
- Each shot = 3-6 words max
- Prompts describe VISUALS, not text
- Each prompt must be FULLY SELF-CONTAINED (no references to other prompts)
- Include the style "{style_description}" in every single prompt
- Include all details (subject, lighting, colors, camera angle) in each prompt"""

        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        result = json.loads(response.choices[0].message.content)
        shots_data = result.get('shots', [])

        if not shots_data:
            print("LLM returned no shots, falling back to rule-based generation")
            return None

        # Calculate durations proportionally
        word_counts = [len(s['phrase'].split()) for s in shots_data]
        total_words = sum(word_counts)
        min_dur, max_dur = CONFIG['video']['clip_duration_range']

        # Allocate time proportionally
        raw_durations = []
        for wc in word_counts:
            raw_dur = (wc / total_words) * audio_duration if total_words > 0 else 1.5
            clamped = max(float(min_dur), min(float(max_dur), raw_dur))
            raw_durations.append(clamped)

        # Scale to match audio duration
        total_raw = sum(raw_durations)
        if total_raw > 0:
            scale = audio_duration / total_raw
            durations = [round(d * scale, 2) for d in raw_durations]
        else:
            durations = raw_durations

        # Build shot objects
        transition_duration = CONFIG['animation'].get('default_transition_duration', 0.5)
        shots = []
        current_time = 0.0

        for i, shot_data in enumerate(shots_data):
            duration = durations[i]
            phrase = shot_data['phrase']
            prompt = shot_data['prompt']

            sfx = detect_sfx_triggers(phrase)
            animation_type, animation_params = detect_animation_type(phrase)

            shots.append({
                'id': i + 1,
                'phrase': phrase,
                'start_time': round(current_time, 2),
                'duration': round(duration, 2),
                'end_time': round(current_time + duration, 2),
                'prompt': prompt,
                'full_prompt': prompt,  # LLM prompt is already complete
                'sfx': sfx,
                'image_status': 'pending',
                'image_path': None,
                'image_url': None,
                'image_error': None,
                'animation_type': animation_type,
                'animation_params': animation_params,
                'transition_in': CONFIG['animation'].get('default_transition', 'crossfade'),
                'transition_duration': transition_duration,
                'animated_clip_path': None,
                'status': 'pending',
                'clip_path': None,
                'trimmed_path': None,
                'error': None,
                'effects': []
            })

            current_time += duration

        print(f"LLM generated {len(shots)} shots for {audio_duration:.1f}s audio")
        return shots

    except Exception as e:
        print(f"LLM shot generation failed: {e}")
        return None


def smart_segment_transcript(transcript_text, audio_duration, image_style=None):
    """
    Smart segmentation of transcript into shots with varying durations.
    First tries LLM-based generation for high-quality context-aware prompts.
    Falls back to rule-based segmentation if LLM is unavailable.
    image_style: Optional style string to apply to all generated prompts.
    """
    # Try LLM-based generation first (higher quality)
    llm_shots = generate_shots_with_llm(transcript_text, audio_duration, image_style)
    if llm_shots:
        return llm_shots

    print("Using rule-based segmentation (LLM not available)")

    # Split into sentences
    lines = [l.strip() for l in transcript_text.strip().split('\n') if l.strip()]

    phrases = []
    for line in lines:
        parts = re.split(r'([.!?…]+)', line)
        current = ""
        for i, part in enumerate(parts):
            if re.match(r'^[.!?…]+$', part):
                current += part
                if current.strip():
                    phrases.append(current.strip())
                current = ""
            else:
                if current.strip():
                    phrases.append(current.strip())
                current = part
        if current.strip():
            phrases.append(current.strip())

    # Filter empty and very short phrases
    phrases = [p for p in phrases if len(p) > 5]

    if not phrases:
        return []

    # Group phrases into shots using smart breakpoints
    transition_words = {'but', 'however', 'now', 'next', 'another', 'meanwhile',
                        'speaking of', 'what about', 'so basically', 'also',
                        'in fact', 'actually', 'the thing is', 'here\'s'}

    shots_phrases = []
    current_group = [phrases[0]]

    for i in range(1, len(phrases)):
        phrase = phrases[i]
        phrase_lower = phrase.lower()

        # Start new shot on:
        # 1. Topic transition words
        starts_with_transition = any(phrase_lower.startswith(tw) for tw in transition_words)
        # 2. Questions
        is_question = '?' in phrase
        # 3. Number/statistic (standalone for emphasis)
        has_numbers = bool(re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?%?\b', phrase))
        # 4. Previous phrase was a question
        prev_was_question = '?' in current_group[-1]

        if starts_with_transition or is_question or has_numbers or prev_was_question:
            shots_phrases.append(' '.join(current_group))
            current_group = [phrase]
        else:
            # Check combined word count - split if too long for punchy cuts
            combined = ' '.join(current_group + [phrase])
            combined_words = len(combined.split())
            if combined_words > 6:  # ~2 seconds at 3 words/sec for fast punchy cuts
                shots_phrases.append(' '.join(current_group))
                current_group = [phrase]
            else:
                current_group.append(phrase)

    if current_group:
        shots_phrases.append(' '.join(current_group))

    # Allocate durations proportionally by word count, scaled to audio_duration
    word_counts = [len(p.split()) for p in shots_phrases]
    total_words = sum(word_counts)

    min_dur, max_dur = CONFIG['video']['clip_duration_range']

    raw_durations = []
    for wc in word_counts:
        raw_dur = (wc / total_words) * audio_duration if total_words > 0 else 3.0
        clamped = max(float(min_dur), min(float(max_dur), raw_dur))
        raw_durations.append(clamped)

    # Scale to match audio_duration exactly
    total_raw = sum(raw_durations)
    if total_raw > 0:
        scale = audio_duration / total_raw
        durations = [round(d * scale, 2) for d in raw_durations]
    else:
        durations = raw_durations

    # Build shot objects
    transition_duration = CONFIG['animation'].get('default_transition_duration', 0.5)
    shots = []
    current_time = 0.0

    for i, phrase in enumerate(shots_phrases):
        duration = durations[i]
        prompt = generate_visual_prompt(phrase, style=image_style)
        sfx = detect_sfx_triggers(phrase)
        animation_type, animation_params = detect_animation_type(phrase)

        shots.append({
            'id': i + 1,
            'phrase': phrase,
            'start_time': round(current_time, 2),
            'duration': round(duration, 2),
            'end_time': round(current_time + duration, 2),
            'prompt': prompt,
            'full_prompt': prompt + ', ' + CONFIG['style']['prompt_suffix'],
            'sfx': sfx,
            # Image generation
            'image_status': 'pending',
            'image_path': None,
            'image_url': None,
            'image_error': None,
            # Animation (replaces video generation)
            'animation_type': animation_type,
            'animation_params': animation_params,
            'transition_in': CONFIG['animation'].get('default_transition', 'crossfade'),
            'transition_duration': transition_duration,
            'animated_clip_path': None,
            'status': 'pending',
            'error': None,
            # Effects (populated by initialize_shot_effects)
            'effects': []
        })

        current_time += duration

    return shots


def animate_image_to_clip(image_path, output_path, duration, animation_type, animation_params, resolution="1080x1920"):
    """
    Create an animated video clip from a static image using FFmpeg zoompan.
    Supports Ken Burns zoom/pan, zoom pulse, and slow drift animations.
    """
    fps = 30
    total_frames = int(duration * fps)
    w, h = resolution.split('x')

    start_zoom = animation_params.get('start_zoom', 1.0)
    end_zoom = animation_params.get('end_zoom', 1.2)
    pan_x = animation_params.get('pan_x', 0)
    pan_y = animation_params.get('pan_y', 0)
    easing = animation_params.get('easing', 'linear')

    # Build zoom expression based on animation type and easing
    if animation_type == 'zoom_pulse':
        # Fast zoom in first 40%, then hold
        peak_frame = int(total_frames * 0.4)
        z_expr = f"if(lt(on\\,{peak_frame})\\,{start_zoom}+({end_zoom}-{start_zoom})*on/{peak_frame}\\,{end_zoom})"
    elif animation_type == 'dolly_push':
        # Accelerating zoom in
        z_expr = f"{start_zoom}+({end_zoom}-{start_zoom})*(on/{total_frames})*(on/{total_frames})"
    elif animation_type == 'pull_back':
        # Smooth zoom out
        z_expr = f"{start_zoom}+({end_zoom}-{start_zoom})*(1-cos(on/{total_frames}*PI))/2"
    elif easing == 'ease_in_out':
        z_expr = f"{start_zoom}+({end_zoom}-{start_zoom})*(1-cos(on/{total_frames}*PI))/2"
    elif easing == 'ease_out':
        z_expr = f"{start_zoom}+({end_zoom}-{start_zoom})*(1-(1-on/{total_frames})*(1-on/{total_frames}))"
    elif easing == 'ease_in':
        z_expr = f"{start_zoom}+({end_zoom}-{start_zoom})*(on/{total_frames})*(on/{total_frames})"
    else:  # linear
        z_expr = f"{start_zoom}+({end_zoom}-{start_zoom})*on/{total_frames}"

    # Pan expressions: center with progressive offset
    pan_x_expr = f"(iw-iw/zoom)/2+{pan_x}*on/{total_frames}"
    pan_y_expr = f"(ih-ih/zoom)/2+{pan_y}*on/{total_frames}"

    # Memory optimization: process zoompan at half resolution, then scale up
    # This dramatically reduces memory usage (4x less pixels to process)
    half_w = int(int(w) // 2)
    half_h = int(int(h) // 2)

    # Scale down -> zoompan at half res -> scale back up
    filter_str = (
        f"scale={half_w}:{half_h}:force_original_aspect_ratio=increase,"
        f"crop={half_w}:{half_h},"
        f"zoompan=z='{z_expr}'"
        f":x='{pan_x_expr}'"
        f":y='{pan_y_expr}'"
        f":d={total_frames}"
        f":s={half_w}x{half_h}"
        f":fps={fps},"
        f"scale={w}:{h}:flags=lanczos"  # Scale back up with high quality
    )

    cmd = [
        'ffmpeg', '-y',
        '-nostdin',  # Don't wait for input
        '-threads', '1',  # Limit threads to reduce memory usage
        '-loop', '1',
        '-i', image_path,
        '-vf', filter_str,
        '-t', str(duration),
        '-c:v', 'libx264',
        '-crf', '23',  # Slightly higher CRF for less memory usage
        '-preset', 'fast',  # Faster preset uses less memory
        '-pix_fmt', 'yuv420p',
        '-max_muxing_queue_size', '1024',  # Limit muxing queue
        '-an',
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise Exception(f"Animation timed out after 120 seconds")
    except subprocess.CalledProcessError as e:
        # Get last 500 chars of stderr (actual error is at the end, not the banner)
        stderr_text = e.stderr.decode() if e.stderr else ""
        stdout_text = e.stdout.decode() if e.stdout else ""
        # Extract just the error lines (skip the banner)
        error_lines = [l for l in stderr_text.split('\n') if l.strip() and not l.startswith('  ')]
        error_msg = '\n'.join(error_lines[-10:]) if error_lines else stderr_text[-500:]
        if not error_msg:
            error_msg = stdout_text[-500:] or str(e)
        raise Exception(f"FFmpeg error: {error_msg}")


def animate_shot_image(project_id, shot_id):
    """Create animated clip from approved shot image using FFmpeg zoompan."""
    print(f"[ANIMATE] Starting animation for project {project_id}, shot {shot_id}")

    # Always use PROJECTS[project_id] directly to avoid reference orphaning
    def get_shot():
        return PROJECTS[project_id]['shots'][shot_id - 1]

    shot = get_shot()
    if not shot.get('image_path'):
        raise Exception("Shot has no image")

    # Validate image file exists on disk
    if not os.path.exists(shot['image_path']):
        raise Exception(f"Image file not found: {shot['image_path']}")

    try:
        get_shot()['status'] = 'animating'
        save_projects()

        clip_filename = f"{project_id}_shot_{shot_id:02d}_animated.mp4"
        clip_path = str(Path(CONFIG['paths']['clips']) / clip_filename)

        animate_image_to_clip(
            shot['image_path'],
            clip_path,
            shot['duration'],
            shot['animation_type'],
            shot['animation_params'],
            CONFIG['video'].get('resolution', '1080x1920')
        )

        get_shot()['status'] = 'generated'
        get_shot()['animated_clip_path'] = clip_path
        save_projects()
        print(f"[ANIMATE] Shot {shot_id}: Animation complete")
        return True

    except Exception as e:
        print(f"[ANIMATE] Shot {shot_id}: ERROR - {e}")
        if project_id in PROJECTS:
            get_shot()['status'] = 'failed'
            get_shot()['error'] = str(e)
            save_projects()
        return False


def concatenate_with_transitions(clip_paths, transition_durations, output_path):
    """
    Concatenate video clips with crossfade transitions using FFmpeg xfade.
    """
    if len(clip_paths) == 0:
        return
    if len(clip_paths) == 1:
        import shutil
        shutil.copy(clip_paths[0], output_path)
        return

    # Get actual durations of each clip via ffprobe
    durations = [get_video_duration(clip) for clip in clip_paths]

    inputs = []
    for clip in clip_paths:
        inputs.extend(['-i', clip])

    n = len(clip_paths)

    if n == 2:
        td = transition_durations[0] if transition_durations else 0.5
        offset = max(0, durations[0] - td)
        filter_str = f"[0:v][1:v]xfade=transition=fade:duration={td}:offset={offset}[outv]"
    else:
        cumulative_dur = durations[0]
        cumulative_trans = 0
        prev_label = "0:v"
        filter_parts = []

        for i in range(1, n):
            td = transition_durations[i - 1] if (i - 1) < len(transition_durations) else 0.5
            offset = max(0, cumulative_dur - cumulative_trans - td)

            if i < n - 1:
                out_label = f"v{i}"
                filter_parts.append(
                    f"[{prev_label}][{i}:v]xfade=transition=fade:duration={td}:offset={offset}[{out_label}]"
                )
                prev_label = out_label
            else:
                filter_parts.append(
                    f"[{prev_label}][{i}:v]xfade=transition=fade:duration={td}:offset={offset}[outv]"
                )

            cumulative_dur += durations[i]
            cumulative_trans += td

        filter_str = ";".join(filter_parts)

    cmd = [
        'ffmpeg', '-y',
        *inputs,
        '-filter_complex', filter_str,
        '-map', '[outv]',
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'medium',
        '-pix_fmt', 'yuv420p',
        '-an',
        output_path
    ]

    subprocess.run(cmd, check=True, capture_output=True, timeout=300)


def generate_clip_with_kling(prompt, project_id, shot_id):
    """
    Call Fal AI / Kling to generate a video clip.
    Updates project state as generation progresses.
    """
    try:
        # Update status
        PROJECTS[project_id]['shots'][shot_id - 1]['status'] = 'generating'

        # Check if fal_client is available
        if not FAL_AVAILABLE:
            PROJECTS[project_id]['shots'][shot_id - 1]['status'] = 'failed'
            PROJECTS[project_id]['shots'][shot_id - 1]['error'] = 'fal-client not installed (requires Python 3.8+)'
            return False

        # Call Kling via Fal
        result = fal_client.subscribe(
            CONFIG['kling_text_to_video'],
            arguments={
                "prompt": prompt,
                "duration": "5",  # Kling generates 5-second clips
                "aspect_ratio": "9:16"
            }
        )
        
        # Download the video
        if result and 'video' in result and 'url' in result['video']:
            video_url = result['video']['url']
            
            # Download video
            clip_filename = f"{project_id}_shot_{shot_id:02d}_raw.mp4"
            clip_path = Path(CONFIG['paths']['clips']) / clip_filename
            
            # Use requests or wget to download
            import requests
            response = requests.get(video_url)
            with open(clip_path, 'wb') as f:
                f.write(response.content)
            
            # Update project state
            PROJECTS[project_id]['shots'][shot_id - 1]['status'] = 'generated'
            PROJECTS[project_id]['shots'][shot_id - 1]['clip_path'] = str(clip_path)
            
            # Auto-trim to required duration
            trim_clip(project_id, shot_id)
            
            return True
        else:
            PROJECTS[project_id]['shots'][shot_id - 1]['status'] = 'failed'
            return False
            
    except Exception as e:
        print(f"Error generating clip: {e}")
        PROJECTS[project_id]['shots'][shot_id - 1]['status'] = 'failed'
        PROJECTS[project_id]['shots'][shot_id - 1]['error'] = str(e)
        return False


def trim_clip(project_id, shot_id):
    """
    Trim a 5-second Kling clip to the required duration.
    Uses motion analysis to find the best segment.
    """
    shot = PROJECTS[project_id]['shots'][shot_id - 1]
    clip_path = shot['clip_path']
    duration = shot['duration']
    
    if not clip_path or not os.path.exists(clip_path):
        return
    
    # For now, take from the start (can add motion analysis later)
    trimmed_filename = clip_path.replace('_raw.mp4', '_trimmed.mp4')
    
    cmd = [
        'ffmpeg', '-y',
        '-i', clip_path,
        '-t', str(duration),
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'fast',
        '-an',  # No audio for b-roll
        trimmed_filename
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        shot['trimmed_path'] = trimmed_filename
    except subprocess.CalledProcessError as e:
        print(f"Error trimming clip: {e}")
    except subprocess.TimeoutExpired:
        print(f"Timeout trimming clip: {clip_path}")


def apply_color_grade(input_path, output_path):
    """
    Apply consistent color grading to a clip.
    """
    if not os.path.exists(input_path):
        raise Exception(f"Input clip not found: {input_path}")

    grade = CONFIG['style']['color_grade']

    # Build FFmpeg filter for color grading
    filter_str = (
        f"eq=brightness={grade['brightness']}:contrast={grade['contrast']}:saturation={grade['saturation']},"
        f"colorbalance=rs={grade['warmth']}:gs={grade['warmth']/2}:bs=-{grade['warmth']}"
    )

    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vf', filter_str,
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'fast',
        output_path
    ]

    subprocess.run(cmd, check=True, capture_output=True, timeout=120)


def generate_effects_config(project_id, platform='tiktok'):
    """
    Generate Remotion effects configuration from per-shot effects.
    Uses user-customized effects stored in each shot.
    """
    project = PROJECTS[project_id]
    platform_config = CONFIG['platforms'].get(platform, CONFIG['platforms']['tiktok'])

    effects = []
    captions = []

    # Calculate frame positions (30fps)
    fps = 30
    hook_duration = 10  # default fallback
    if project.get('hook_path') and os.path.exists(project['hook_path']):
        try:
            hook_duration = get_video_duration(project['hook_path'])
        except Exception:
            pass
    current_frame = int(hook_duration * fps)  # Start after hook

    for shot in project['shots']:
        if shot['status'] not in ['approved', 'generated']:
            continue

        start_frame = current_frame
        end_frame = current_frame + int(shot['duration'] * fps)

        # Add caption
        captions.append({
            'text': shot['phrase'],
            'startFrame': start_frame,
            'endFrame': end_frame,
            'highlights': detect_highlight_words(shot['phrase'])
        })

        # Add enabled effects from shot's effects list
        for effect in shot.get('effects', []):
            if not effect.get('enabled', True):
                continue

            params = effect.get('parameters', {})
            effect_data = {
                'type': effect['type'],
                'frame': effect.get('frame', start_frame)
            }

            # Add type-specific parameters
            if effect['type'] == 'zoom_pulse':
                effect_data['intensity'] = params.get('intensity', 1.08)
                effect_data['duration'] = params.get('duration', 20)
            elif effect['type'] == 'screen_shake':
                effect_data['intensity'] = params.get('intensity', 12)
                effect_data['duration'] = params.get('duration', 15)
            elif effect['type'] == 'flash':
                effect_data['duration'] = params.get('duration', 6)
                effect_data['color'] = params.get('color', 'white')
            elif effect['type'] == 'particles':
                effect_data['x'] = params.get('x', 50)
                effect_data['y'] = params.get('y', 50)
                effect_data['count'] = params.get('count', 15)
                effect_data['duration'] = params.get('duration', 25)
            elif effect['type'] == 'emoji':
                effect_data['emoji'] = params.get('emoji', '✨')
                effect_data['x'] = params.get('x', 50)
                effect_data['y'] = params.get('y', 50)
                effect_data['duration'] = params.get('duration', 30)

            effects.append(effect_data)

        current_frame = end_frame

    # Use user-selected caption settings if available, else fall back to platform config
    caption_settings = project.get('caption_settings', {})

    # Collect SFX from shots
    sfx_list = []
    sfx_dir = Path(__file__).parent / 'sfx'
    current_sfx_frame = int(hook_duration * fps)

    for shot in project['shots']:
        if shot['status'] not in ['approved', 'generated']:
            continue
        shot_sfx = shot.get('sfx', [])
        for sfx_item in shot_sfx:
            sfx_file = sfx_dir / sfx_item.get('file', '')
            if sfx_file.exists():
                sfx_list.append({
                    'src': str(sfx_file.absolute()),
                    'frame': current_sfx_frame + int(sfx_item.get('offset', 0) * fps),
                    'duration': int(sfx_item.get('duration', 1) * fps),
                    'volume': sfx_item.get('volume', 0.7)
                })
        current_sfx_frame += int(shot['duration'] * fps)

    return {
        'effects': effects,
        'captions': captions,
        'sfx': sfx_list,
        'captionStyle': {
            'font': caption_settings.get('font', platform_config.get('font', 'Nunito')),
            'fontSize': caption_settings.get('size', platform_config.get('font_size', 64)),
            'bottom': caption_settings.get('bottom', platform_config.get('position_bottom', 180)),
            'highlightColor': caption_settings.get('highlight', platform_config.get('highlight_color', '#8b5cf6')),
            'captionMode': caption_settings.get('style', 'karaoke'),
            'bgStyle': caption_settings.get('bg', 'dark'),
        },
        'durationInFrames': current_frame + (10 * fps)  # Add end duration
    }


def detect_highlight_words(phrase):
    """Detect words that should be highlighted in captions."""
    highlights = []
    
    # Numbers
    import re
    numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?%?\b', phrase)
    highlights.extend(numbers)
    
    # Key action words
    action_words = ['boom', 'explode', 'pop', 'burst', 'transform', 'love', 'amazing']
    for word in action_words:
        if word in phrase.lower():
            # Find the actual word in the phrase (with proper casing)
            for w in phrase.split():
                if word in w.lower():
                    highlights.append(w.strip('.,!?'))
    
    return highlights


def render_with_remotion(project_id, video_path, platform='tiktok'):
    """
    Apply Remotion effects to the assembled video.
    Returns path to the final video with effects.
    """
    project = PROJECTS[project_id]
    
    # Generate effects config
    effects_config = generate_effects_config(project_id, platform)
    effects_config['videoSrc'] = video_path
    effects_config['audioSrc'] = None  # Audio already in video
    
    # Write config to temp file
    temp_dir = Path(CONFIG['paths']['temp']) / project_id
    config_path = temp_dir / 'effects-config.json'
    
    with open(config_path, 'w') as f:
        json.dump(effects_config, f, indent=2)
    
    # Output path
    output_path = Path(CONFIG['paths']['outputs']) / f"{project['name']}_{platform}_with_effects.mp4"
    
    # Check if Remotion is available
    remotion_dir = Path(__file__).parent / 'remotion'
    if not (remotion_dir / 'node_modules').exists():
        print("Remotion not installed. Skipping effects layer.")
        return video_path
    
    # Run Remotion render
    cmd = [
        'node',
        str(remotion_dir / 'render-effects.js'),
        '--config', str(config_path),
        '--output', str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(remotion_dir))
        print(result.stdout)
        return str(output_path)
    except subprocess.CalledProcessError as e:
        print(f"Remotion render failed: {e.stderr}")
        return video_path  # Return video without effects
    except FileNotFoundError:
        print("Node.js not found. Skipping effects layer.")
        return video_path


def assemble_final_video(project_id, platform='tiktok', apply_effects=True):
    """
    Assemble all components into final video:
    Hook + Animated image clips with transitions + Audio
    Optionally applies Remotion effects layer.
    """
    project = PROJECTS[project_id]
    platform_config = CONFIG['platforms'].get(platform, CONFIG['platforms']['tiktok'])

    # Paths
    hook_path = project['hook_path']
    audio_path = project['audio_path']

    # Get all approved/generated animated clips and their transition durations
    clips = []
    transitions = []
    for shot in project['shots']:
        if shot.get('skipped'):
            continue
        clip_path = shot.get('animated_clip_path') or shot.get('trimmed_path')
        if shot['status'] in ['approved', 'generated'] and clip_path:
            clips.append(clip_path)
            transitions.append(shot.get('transition_duration', 0.5))

    if not clips:
        return None, "No clips available for assembly"

    # Transitions list should be len(clips) - 1
    transitions = transitions[:-1] if len(transitions) > 1 else []

    # Create temp directory for this assembly
    temp_dir = Path(CONFIG['paths']['temp']) / project_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Color grade all b-roll clips
    graded_clips = []
    for i, clip in enumerate(clips):
        graded_path = temp_dir / f"graded_{i:02d}.mp4"
        apply_color_grade(clip, str(graded_path))
        graded_clips.append(str(graded_path))

    # Step 2: Concatenate b-roll with crossfade transitions
    broll_combined = temp_dir / "broll_combined.mp4"
    concatenate_with_transitions(graded_clips, transitions, str(broll_combined))

    # Step 3: Add audio to b-roll
    broll_with_audio = temp_dir / "broll_with_audio.mp4"
    cmd = [
        'ffmpeg', '-y',
        '-i', str(broll_combined),
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        str(broll_with_audio)
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)

    # Step 4: Create final concat (hook + broll with audio)
    final_concat_file = temp_dir / "final_concat.txt"
    with open(final_concat_file, 'w') as f:
        f.write(f"file '{hook_path}'\n")
        f.write(f"file '{broll_with_audio}'\n")

    # Step 5: Final concatenation
    final_no_captions = temp_dir / "final_no_captions.mp4"
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(final_concat_file),
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'fast',
        '-c:a', 'aac',
        str(final_no_captions)
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)

    # Step 6: Apply Remotion effects layer (if enabled)
    if apply_effects:
        final_output = render_with_remotion(project_id, str(final_no_captions), platform)
    else:
        final_output = Path(CONFIG['paths']['outputs']) / f"{project['name']}_{platform}_final.mp4"
        import shutil
        shutil.copy(str(final_no_captions), str(final_output))
        final_output = str(final_output)

    project['final_video'] = final_output
    project['status'] = 'complete'
    save_projects()

    return final_output, None


# ============== API ROUTES ==============

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/projects', methods=['POST'])
def create_project():
    """Create a new video project. Requires hook video and audio file."""
    data = request.form
    files = request.files

    project_id = str(uuid.uuid4())[:8]
    project_name = secure_filename(data.get('name', f'project_{project_id}'))

    # Save uploaded files
    uploads_dir = Path(CONFIG['paths']['uploads']) / project_id
    print(f"[PROJECT {project_id}] Creating project...")
    print(f"[PROJECT {project_id}] PERSISTENT_DATA_DIR: {PERSISTENT_DATA_DIR}")
    print(f"[PROJECT {project_id}] uploads_dir: {uploads_dir}")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PROJECT {project_id}] uploads_dir created: {uploads_dir.exists()}")

    hook_path = None
    audio_path = None

    if 'hook' in files:
        hook = files['hook']
        hook_path = str(uploads_dir / secure_filename(hook.filename))
        hook.save(hook_path)
        print(f"[PROJECT {project_id}] Saved hook to: {hook_path}, exists: {os.path.exists(hook_path)}")

    if 'audio' in files:
        audio = files['audio']
        audio_path = str(uploads_dir / secure_filename(audio.filename))
        audio.save(audio_path)
        print(f"[PROJECT {project_id}] Saved audio to: {audio_path}, exists: {os.path.exists(audio_path)}, size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 0}")

    if not hook_path or not audio_path:
        return jsonify({'error': 'Hook video and audio file are required'}), 400

    # Get image style - either preset key or custom text
    image_style_preset = data.get('image_style_preset', CONFIG.get('image_styles', {}).get('default', 'pixar_3d'))
    image_style_custom = data.get('image_style_custom', '')

    # Get image model (default to flux-schnell)
    image_model = data.get('image_model', 'flux-schnell')

    # Resolve style: custom overrides preset
    if image_style_custom.strip():
        image_style = image_style_custom.strip()
    else:
        presets = CONFIG.get('image_styles', {}).get('presets', {})
        image_style = presets.get(image_style_preset, presets.get('pixar_3d', ''))

    # Create project - transcription will run in background
    project = {
        'id': project_id,
        'name': project_name,
        'created_at': datetime.now().isoformat(),
        'status': 'transcribing',
        'script': None,
        'shots': [],
        'hook_path': hook_path,
        'audio_path': audio_path,
        'audio_duration': None,
        'platform': data.get('platform', 'tiktok'),
        'image_style': image_style,
        'image_style_preset': image_style_preset,
        'image_model': image_model,
        'final_video': None,
        'transcription': {
            'status': 'processing',
            'text': None,
            'confidence': None
        }
    }

    PROJECTS[project_id] = project

    # Background: transcribe audio, then smart-segment into shots
    def transcribe_and_segment():
        import traceback
        try:
            # IMPORTANT: Always reference PROJECTS[project_id] directly, not local 'project' var
            # because sync_projects() may replace the dict reference during polling
            print(f"[PROJECT {project_id}] Starting transcription pipeline...")
            print(f"[PROJECT {project_id}] Audio path: {audio_path}")
            print(f"[PROJECT {project_id}] Audio file exists: {os.path.exists(audio_path)}")

            # Get audio duration
            print(f"[PROJECT {project_id}] Getting audio duration...")
            audio_dur = get_audio_duration(audio_path)
            PROJECTS[project_id]['audio_duration'] = audio_dur
            print(f"[PROJECT {project_id}] Audio duration: {audio_dur}s")

            # Transcribe
            print(f"[PROJECT {project_id}] Starting OpenAI Whisper transcription...")
            text, confidence = transcribe_audio_file(audio_path)
            print(f"[PROJECT {project_id}] Transcription complete: {len(text)} chars, confidence: {confidence}")

            PROJECTS[project_id]['transcription']['text'] = text
            PROJECTS[project_id]['transcription']['confidence'] = confidence
            PROJECTS[project_id]['transcription']['status'] = 'completed'
            PROJECTS[project_id]['script'] = text

            # Smart segment with project's image style
            print(f"[PROJECT {project_id}] Segmenting into shots...")
            shots = smart_segment_transcript(text, audio_dur, image_style=PROJECTS[project_id].get('image_style'))
            PROJECTS[project_id]['shots'] = shots
            print(f"[PROJECT {project_id}] Created {len(shots)} shots")

            # Initialize effects
            initialize_shot_effects(PROJECTS[project_id])

            PROJECTS[project_id]['status'] = 'created'
            save_projects()
            print(f"[PROJECT {project_id}] Pipeline complete! Status: created, Shots: {len(shots)}")

        except Exception as e:
            print(f"[PROJECT {project_id}] TRANSCRIPTION FAILED!")
            print(f"[PROJECT {project_id}] Error: {e}")
            print(f"[PROJECT {project_id}] Traceback: {traceback.format_exc()}")
            if project_id in PROJECTS:
                PROJECTS[project_id]['transcription']['status'] = 'failed'
                PROJECTS[project_id]['transcription']['error'] = str(e)
                PROJECTS[project_id]['status'] = 'transcription_failed'
            save_projects()

    thread = threading.Thread(target=transcribe_and_segment)
    thread.start()

    save_projects()
    return jsonify({
        'success': True,
        'project': project
    })


@app.route('/api/projects', methods=['GET'])
def list_projects():
    """List all projects."""
    projects_list = []
    for pid, proj in PROJECTS.items():
        projects_list.append({
            'id': pid,
            'name': proj.get('name', 'Untitled'),
            'status': proj.get('status', 'unknown'),
            'created_at': proj.get('created_at'),
            'shots_count': len(proj.get('shots', [])),
            'platform': proj.get('platform', 'tiktok'),
            'has_final_video': bool(proj.get('final_video'))
        })
    # Sort by creation date, newest first
    projects_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return jsonify({'projects': projects_list})


@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404
    del PROJECTS[project_id]
    save_projects()
    return jsonify({'success': True})


@app.route('/api/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get project details."""
    project = fetch_project(project_id)
    if not project:
        print(f"[API] GET /api/projects/{project_id} - NOT FOUND")
        return jsonify({'error': 'Project not found'}), 404
    print(f"[API] GET /api/projects/{project_id} - status: {project.get('status')}, shots: {len(project.get('shots', []))}")
    return jsonify(project)


@app.route('/api/projects/<project_id>/generate', methods=['POST'])
def generate_clips(project_id):
    """Start generating all b-roll clips."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404
    
    project = PROJECTS[project_id]
    project['status'] = 'generating'
    
    # Generate clips in background threads
    def generate_all():
        for shot in project['shots']:
            if shot['status'] == 'pending':
                generate_clip_with_kling(
                    shot['full_prompt'],
                    project_id,
                    shot['id']
                )
    
    thread = threading.Thread(target=generate_all)
    thread.start()
    
    return jsonify({'success': True, 'message': 'Generation started'})


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/regenerate', methods=['POST'])
def regenerate_shot(project_id, shot_id):
    """Regenerate a specific shot."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404
    
    project = PROJECTS[project_id]
    
    # Find the shot
    shot = None
    for s in project['shots']:
        if s['id'] == shot_id:
            shot = s
            break
    
    if not shot:
        return jsonify({'error': 'Shot not found'}), 404
    
    # Check if custom prompt provided
    data = request.get_json() or {}
    if 'prompt' in data:
        shot['prompt'] = data['prompt']
        shot['full_prompt'] = data['prompt'] + CONFIG['style']['prompt_suffix']
    
    # Reset status and regenerate
    shot['status'] = 'pending'
    
    def regen():
        generate_clip_with_kling(shot['full_prompt'], project_id, shot_id)
    
    thread = threading.Thread(target=regen)
    thread.start()
    
    return jsonify({'success': True, 'message': 'Regeneration started'})


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/approve', methods=['POST'])
def approve_shot(project_id, shot_id):
    """Approve a generated shot."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404
    
    for shot in PROJECTS[project_id]['shots']:
        if shot['id'] == shot_id:
            shot['status'] = 'approved'
            return jsonify({'success': True})
    
    return jsonify({'error': 'Shot not found'}), 404


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/reject', methods=['POST'])
def reject_shot(project_id, shot_id):
    """Reject a generated shot."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404
    
    for shot in PROJECTS[project_id]['shots']:
        if shot['id'] == shot_id:
            shot['status'] = 'rejected'
            return jsonify({'success': True})
    
    return jsonify({'error': 'Shot not found'}), 404


@app.route('/api/projects/<project_id>/assemble', methods=['POST'])
def assemble_video(project_id):
    """Assemble final video from approved clips."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json() or {}
    platform = data.get('platform', PROJECTS[project_id].get('platform', 'tiktok'))
    apply_effects = data.get('apply_effects', True)

    # Store caption style settings in project for Remotion
    PROJECTS[project_id]['caption_settings'] = {
        'style': data.get('caption_style', 'karaoke'),
        'bg': data.get('caption_bg', 'dark'),
        'highlight': data.get('caption_highlight', '#8b5cf6'),
        'font': data.get('caption_font', 'Nunito-Bold'),
        'size': data.get('caption_size', 64),
        'bottom': data.get('caption_bottom', 180),
    }

    try:
        output_path, error = assemble_final_video(project_id, platform, apply_effects)

        if error:
            return jsonify({'error': error}), 400

        return jsonify({
            'success': True,
            'output_path': output_path,
            'has_effects': apply_effects
        })
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        print(f"[ASSEMBLE] FFmpeg error: {error_msg}")
        return jsonify({'error': 'Video assembly failed', 'details': error_msg}), 500
    except Exception as e:
        print(f"[ASSEMBLE] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<project_id>/download', methods=['GET'])
def download_video(project_id):
    """Download the final video."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    project = PROJECTS[project_id]
    if not project.get('final_video'):
        return jsonify({'error': 'Video not yet assembled'}), 400

    return send_file(project['final_video'], as_attachment=True)


@app.route('/api/projects/<project_id>/preview', methods=['GET'])
def preview_video(project_id):
    """Preview the final assembled video (inline playback)."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    project = PROJECTS[project_id]
    if not project.get('final_video'):
        return jsonify({'error': 'Video not yet assembled'}), 400

    return send_file(project['final_video'], mimetype='video/mp4')


@app.route('/api/projects/<project_id>/clips/<int:shot_id>/preview', methods=['GET'])
def preview_clip(project_id, shot_id):
    """Preview a generated clip."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404
    
    for shot in PROJECTS[project_id]['shots']:
        if shot['id'] == shot_id:
            clip_path = shot.get('animated_clip_path') or shot.get('trimmed_path') or shot.get('clip_path')
            if clip_path and os.path.exists(clip_path):
                return send_file(clip_path, mimetype='video/mp4')

    return jsonify({'error': 'Clip not found'}), 404


# ============== TRANSCRIPTION ENDPOINTS ==============

@app.route('/api/transcribe', methods=['POST'])
def start_transcription():
    """Upload audio and start transcription."""
    if not WHISPER_AVAILABLE and not OPENAI_AVAILABLE:
        return jsonify({'error': 'No transcription method available. Configure OpenAI API key.'}), 500

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    transcription_id = str(uuid.uuid4())[:8]

    # Save audio file temporarily
    temp_dir = Path(CONFIG['paths']['temp']) / 'transcriptions'
    temp_dir.mkdir(parents=True, exist_ok=True)
    audio_path = str(temp_dir / secure_filename(audio_file.filename))
    audio_file.save(audio_path)

    # Initialize transcription state
    TRANSCRIPTIONS[transcription_id] = {
        'id': transcription_id,
        'status': 'pending',
        'audio_path': audio_path,
        'text': None,
        'confidence': None,
        'error': None
    }

    # Start transcription in background
    thread = threading.Thread(target=run_transcription, args=(transcription_id, audio_path))
    thread.start()

    return jsonify({
        'success': True,
        'transcription_id': transcription_id,
        'status': 'processing'
    })


@app.route('/api/transcription/<transcription_id>', methods=['GET'])
def get_transcription(transcription_id):
    """Get transcription status and result."""
    if transcription_id not in TRANSCRIPTIONS:
        return jsonify({'error': 'Transcription not found'}), 404

    trans = TRANSCRIPTIONS[transcription_id]
    return jsonify({
        'id': trans['id'],
        'status': trans['status'],
        'text': trans['text'],
        'confidence': trans['confidence'],
        'error': trans['error']
    })


# ============== IMAGE GENERATION ENDPOINTS ==============

@app.route('/api/projects/<project_id>/shots/<int:shot_id>/generate-image', methods=['POST'])
def generate_shot_image(project_id, shot_id):
    """Generate an image for a specific shot."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    project = PROJECTS[project_id]
    shot = None
    for s in project['shots']:
        if s['id'] == shot_id:
            shot = s
            break

    if not shot:
        return jsonify({'error': 'Shot not found'}), 404

    # Check for custom prompt
    data = request.get_json() or {}
    custom_prompt = data.get('prompt')

    if custom_prompt:
        shot['prompt'] = custom_prompt
        shot['full_prompt'] = custom_prompt + CONFIG['style']['prompt_suffix']

    # Generate in background
    def generate():
        generate_image_for_shot(project_id, shot_id, custom_prompt)

    thread = threading.Thread(target=generate)
    thread.start()

    return jsonify({'success': True, 'status': 'generating'})


@app.route('/api/projects/<project_id>/generate-all-images', methods=['POST'])
def generate_all_images(project_id):
    """Generate images for all pending shots."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    def generate_all():
        # Always get fresh reference to avoid orphaning
        shots = PROJECTS[project_id]['shots']
        # Include 'failed' shots for retry
        to_generate = [s for s in shots if s['image_status'] in ['pending', 'failed'] and not s.get('skipped')]
        print(f"[IMAGE GEN] Starting batch generation for {len(to_generate)} shots (pending + failed)")

        processed = 0
        successful = 0
        for shot in to_generate:
            result = generate_image_for_shot(project_id, shot['id'])
            processed += 1

            if result == True:
                successful += 1
                # Standard delay between successful requests
                if processed < len(to_generate):
                    print(f"[IMAGE GEN] Waiting 5 seconds before next request...")
                    time.sleep(5)
            elif result == 'rate_limited':
                # Got rate limited - wait longer before continuing
                if processed < len(to_generate):
                    print(f"[IMAGE GEN] Rate limited! Waiting 15 seconds before continuing...")
                    time.sleep(15)
            else:
                # Other failure - still wait a bit
                if processed < len(to_generate):
                    print(f"[IMAGE GEN] Failed, waiting 5 seconds before next request...")
                    time.sleep(5)

        print(f"[IMAGE GEN] Batch generation complete - {successful}/{processed} shots successful")

    thread = threading.Thread(target=generate_all)
    thread.start()

    return jsonify({'success': True, 'message': 'Image generation started'})


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/image/preview', methods=['GET'])
def preview_shot_image(project_id, shot_id):
    """Preview a generated shot image."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    for shot in PROJECTS[project_id]['shots']:
        if shot['id'] == shot_id:
            image_path = shot.get('image_path')
            if image_path and os.path.exists(image_path):
                return send_file(image_path, mimetype='image/png')

    return jsonify({'error': 'Image not found'}), 404


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/approve-image', methods=['POST'])
def approve_shot_image(project_id, shot_id):
    """Approve an image and optionally start video generation."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    shot = None
    for s in PROJECTS[project_id]['shots']:
        if s['id'] == shot_id:
            shot = s
            break

    if not shot:
        return jsonify({'error': 'Shot not found'}), 404

    shot['image_status'] = 'approved'
    save_projects()

    # Check if auto-generate video is requested
    data = request.get_json() or {}
    if data.get('generate_video', False):
        def generate():
            generate_video_from_shot_image(project_id, shot_id)
        thread = threading.Thread(target=generate)
        thread.start()
        return jsonify({'success': True, 'video_generation': 'started'})

    return jsonify({'success': True})


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/reject-image', methods=['POST'])
def reject_shot_image(project_id, shot_id):
    """Reject an image."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    for shot in PROJECTS[project_id]['shots']:
        if shot['id'] == shot_id:
            shot['image_status'] = 'rejected'
            save_projects()
            return jsonify({'success': True})

    return jsonify({'error': 'Shot not found'}), 404


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/regenerate-image', methods=['POST'])
def regenerate_shot_image(project_id, shot_id):
    """Regenerate image for a shot with optional new prompt."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    shot = None
    for s in PROJECTS[project_id]['shots']:
        if s['id'] == shot_id:
            shot = s
            break

    if not shot:
        return jsonify({'error': 'Shot not found'}), 404

    # Check for custom prompt
    data = request.get_json() or {}
    custom_prompt = data.get('prompt')

    if custom_prompt:
        shot['prompt'] = custom_prompt
        shot['full_prompt'] = custom_prompt + CONFIG['style']['prompt_suffix']

    shot['image_status'] = 'pending'

    def generate():
        generate_image_for_shot(project_id, shot_id, custom_prompt)

    thread = threading.Thread(target=generate)
    thread.start()

    return jsonify({'success': True, 'status': 'generating'})


@app.route('/api/projects/<project_id>/generate-videos-from-images', methods=['POST'])
@app.route('/api/projects/<project_id>/animate-all', methods=['POST'])
def animate_all_shots(project_id):
    """Create animated clips from all approved images using FFmpeg zoompan."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    def animate_all():
        # Always get fresh reference to avoid orphaning
        shots = PROJECTS[project_id]['shots']
        # Include 'failed' shots for retry
        to_animate = [s for s in shots if s['image_status'] == 'approved' and s['status'] in ['pending', 'failed'] and not s.get('skipped')]
        print(f"[ANIMATE] Starting batch animation for {len(to_animate)} shots (pending + failed)")

        processed = 0
        successful = 0
        for shot in to_animate:
            result = animate_shot_image(project_id, shot['id'])
            processed += 1
            if result:
                successful += 1
            # Small delay between FFmpeg processes to avoid overwhelming the system
            if processed < len(to_animate):
                time.sleep(0.5)

        print(f"[ANIMATE] Batch animation complete - {successful}/{processed} shots successful")

    thread = threading.Thread(target=animate_all)
    thread.start()

    return jsonify({'success': True, 'message': 'Animation generation started'})


# ============== EFFECTS MANAGEMENT ENDPOINTS ==============

@app.route('/api/projects/<project_id>/effects', methods=['GET'])
def get_project_effects(project_id):
    """Get all effects configuration for a project."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    project = PROJECTS[project_id]

    # Build effects by shot
    shots_effects = {}
    for shot in project['shots']:
        shots_effects[str(shot['id'])] = shot.get('effects', [])

    # Get available effect types from config
    effect_types = list(CONFIG.get('effects', {}).get('types', {}).keys())
    if not effect_types:
        effect_types = ['zoom_pulse', 'screen_shake', 'emoji', 'flash', 'particles']

    available_emojis = CONFIG.get('effects', {}).get('types', {}).get('emoji', {}).get('available_emojis',
        ['💥', '✨', '❤️', '🍿', '😮', '🔥', '⭐'])

    return jsonify({
        'shots': shots_effects,
        'available_types': effect_types,
        'available_emojis': available_emojis,
        'effect_defaults': CONFIG.get('effects', {}).get('types', {})
    })


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/effects', methods=['GET'])
def get_shot_effects(project_id, shot_id):
    """Get effects for a specific shot."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    for shot in PROJECTS[project_id]['shots']:
        if shot['id'] == shot_id:
            return jsonify({
                'shot_id': shot_id,
                'effects': shot.get('effects', [])
            })

    return jsonify({'error': 'Shot not found'}), 404


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/effects', methods=['PUT'])
def update_shot_effects(project_id, shot_id):
    """Replace all effects for a shot."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json() or {}
    new_effects = data.get('effects', [])

    for shot in PROJECTS[project_id]['shots']:
        if shot['id'] == shot_id:
            shot['effects'] = new_effects
            return jsonify({'success': True, 'effects': shot['effects']})

    return jsonify({'error': 'Shot not found'}), 404


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/effects', methods=['POST'])
def add_shot_effect(project_id, shot_id):
    """Add a new effect to a shot."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json() or {}
    effect_type = data.get('type')
    parameters = data.get('parameters', {})

    if not effect_type:
        return jsonify({'error': 'Effect type required'}), 400

    for shot in PROJECTS[project_id]['shots']:
        if shot['id'] == shot_id:
            # Calculate frame based on shot timing
            fps = 30
            hook_duration = 10
            start_frame = int((hook_duration + shot['start_time']) * fps)

            new_effect = {
                'id': generate_effect_id(),
                'type': effect_type,
                'enabled': True,
                'trigger_phrase': 'manual',
                'frame': start_frame,
                'parameters': parameters
            }

            if 'effects' not in shot:
                shot['effects'] = []
            shot['effects'].append(new_effect)

            return jsonify({'success': True, 'effect': new_effect})

    return jsonify({'error': 'Shot not found'}), 404


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/effects/<effect_id>', methods=['PATCH'])
def update_effect(project_id, shot_id, effect_id):
    """Update a specific effect's properties."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json() or {}

    for shot in PROJECTS[project_id]['shots']:
        if shot['id'] == shot_id:
            for effect in shot.get('effects', []):
                if effect['id'] == effect_id:
                    if 'enabled' in data:
                        effect['enabled'] = data['enabled']
                    if 'parameters' in data:
                        effect['parameters'].update(data['parameters'])
                    return jsonify({'success': True, 'effect': effect})
            return jsonify({'error': 'Effect not found'}), 404

    return jsonify({'error': 'Shot not found'}), 404


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/effects/<effect_id>', methods=['DELETE'])
def delete_effect(project_id, shot_id, effect_id):
    """Delete an effect from a shot."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    for shot in PROJECTS[project_id]['shots']:
        if shot['id'] == shot_id:
            effects = shot.get('effects', [])
            shot['effects'] = [e for e in effects if e['id'] != effect_id]
            return jsonify({'success': True})

    return jsonify({'error': 'Shot not found'}), 404


@app.route('/api/projects/<project_id>/reset-effects', methods=['POST'])
def reset_project_effects(project_id):
    """Reset all effects to auto-detected defaults."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    project = PROJECTS[project_id]
    initialize_shot_effects(project)

    return jsonify({'success': True, 'message': 'Effects reset to auto-detected'})


# ============== ANIMATION ENDPOINTS ==============

@app.route('/api/projects/<project_id>/shots/<int:shot_id>/animate', methods=['POST'])
def animate_single_shot(project_id, shot_id):
    """Create animated clip from a single approved shot image."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    shot = None
    for s in PROJECTS[project_id]['shots']:
        if s['id'] == shot_id:
            shot = s
            break

    if not shot:
        return jsonify({'error': 'Shot not found'}), 404

    if shot.get('image_status') != 'approved':
        return jsonify({'error': 'Image must be approved before animating'}), 400

    def animate():
        animate_shot_image(project_id, shot_id)

    thread = threading.Thread(target=animate)
    thread.start()

    return jsonify({'success': True, 'status': 'animating'})


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/animation', methods=['PUT'])
def update_shot_animation(project_id, shot_id):
    """Update animation type and parameters for a shot."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json() or {}

    for shot in PROJECTS[project_id]['shots']:
        if shot['id'] == shot_id:
            if 'animation_type' in data:
                shot['animation_type'] = data['animation_type']
                # Load default params for the new type from config
                type_defaults = CONFIG['animation'].get('types', {}).get(data['animation_type'], {})
                shot['animation_params'] = {
                    'start_zoom': type_defaults.get('start_zoom', 1.0),
                    'end_zoom': type_defaults.get('end_zoom', 1.2),
                    'pan_x': type_defaults.get('pan_x', 0),
                    'pan_y': type_defaults.get('pan_y', 0),
                    'easing': 'ease_in_out'
                }
            if 'animation_params' in data:
                shot['animation_params'].update(data['animation_params'])
            if 'transition_in' in data:
                shot['transition_in'] = data['transition_in']
            if 'transition_duration' in data:
                shot['transition_duration'] = data['transition_duration']

            return jsonify({'success': True, 'shot': shot})

    return jsonify({'error': 'Shot not found'}), 404


@app.route('/api/projects/<project_id>/re-segment', methods=['POST'])
def re_segment_project(project_id):
    """Re-segment transcript after manual edits."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    project = PROJECTS[project_id]
    data = request.get_json() or {}

    transcript = data.get('transcript', project.get('script', ''))
    if not transcript:
        return jsonify({'error': 'No transcript available'}), 400

    audio_duration = project.get('audio_duration')
    if not audio_duration:
        try:
            audio_duration = get_audio_duration(project['audio_path'])
            project['audio_duration'] = audio_duration
        except Exception as e:
            return jsonify({'error': f'Cannot get audio duration: {e}'}), 500

    shots = smart_segment_transcript(transcript, audio_duration, image_style=project.get('image_style'))
    project['script'] = transcript
    project['shots'] = shots
    initialize_shot_effects(project)

    return jsonify({'success': True, 'project': project})


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/skip', methods=['POST'])
def skip_shot(project_id, shot_id):
    """Skip a shot - exclude from image generation and assembly."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    for shot in PROJECTS[project_id]['shots']:
        if shot['id'] == shot_id:
            shot['skipped'] = True
            shot['image_status'] = 'skipped'
            return jsonify({'success': True, 'shot': shot})

    return jsonify({'error': 'Shot not found'}), 404


@app.route('/api/projects/<project_id>/shots/<int:shot_id>/unskip', methods=['POST'])
def unskip_shot(project_id, shot_id):
    """Unskip a shot - re-include in pipeline."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    for shot in PROJECTS[project_id]['shots']:
        if shot['id'] == shot_id:
            shot['skipped'] = False
            shot['image_status'] = 'pending'
            return jsonify({'success': True, 'shot': shot})

    return jsonify({'error': 'Shot not found'}), 404


@app.route('/api/config/replicate-key', methods=['POST'])
def set_replicate_api_key():
    """Set the Replicate API token. Persists to .env file for future sessions."""
    data = request.get_json() or {}
    key = data.get('key', '').strip()
    if not key:
        return jsonify({'error': 'API key is required'}), 400

    # Save to .env file (persists across restarts)
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    env_lines = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip().startswith('REPLICATE_API_TOKEN='):
                    env_lines.append(f'REPLICATE_API_TOKEN={key}\n')
                    found = True
                else:
                    env_lines.append(line)
    if not found:
        env_lines.append(f'REPLICATE_API_TOKEN={key}\n')

    with open(env_path, 'w') as f:
        f.writelines(env_lines)

    # Also set in current process environment so it takes effect immediately
    os.environ['REPLICATE_API_TOKEN'] = key

    return jsonify({'success': True, 'message': 'API key saved to .env'})


@app.route('/api/config/replicate-key', methods=['GET'])
def get_replicate_api_key_status():
    """Check if Replicate API token is configured."""
    import replicate_api
    key = replicate_api.get_replicate_key()
    is_set = bool(key) and key != 'YOUR_REPLICATE_API_TOKEN_HERE'
    return jsonify({'configured': is_set, 'masked': f"...{key[-4:]}" if is_set else None})


@app.route('/api/projects/<project_id>/generate-caption', methods=['POST'])
def generate_post_caption(project_id):
    """Generate a social media post caption with hashtags from the transcript."""
    if project_id not in PROJECTS:
        return jsonify({'error': 'Project not found'}), 404

    project = PROJECTS[project_id]
    transcript = project.get('script', '')
    platform = project.get('platform', 'instagram')

    if not transcript:
        return jsonify({'error': 'No transcript available'}), 400

    caption = build_post_caption(transcript, platform)
    project['post_caption'] = caption
    return jsonify({'success': True, 'caption': caption})


def build_post_caption(transcript, platform='instagram'):
    """Build a social media caption from transcript content."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip() and len(s.strip()) > 5]

    if not sentences:
        return "Check out this video! #facts #viral"

    # Hook: first sentence or first question
    hook = ""
    for s in sentences:
        if '?' in s or any(w in s.lower() for w in ['did you know', 'have you ever', 'what if', 'imagine']):
            hook = s.strip() + '?'
            break
    if not hook:
        hook = sentences[0].strip()
        if not hook.endswith(('?', '!')):
            hook = hook + '!'

    # Key fact: find the most interesting sentence (has numbers or superlatives)
    key_fact = ""
    for s in sentences[1:]:
        if re.search(r'\d', s) or any(w in s.lower() for w in ['most', 'biggest', 'fastest', 'only', 'never', 'always', 'million', 'billion']):
            key_fact = s.strip()
            break
    if not key_fact and len(sentences) > 1:
        key_fact = sentences[1].strip()

    # Extract topic words for hashtags
    stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'it',
            'its', 'this', 'that', 'they', 'them', 'their', 'and', 'or', 'but', 'so',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'not', 'just', 'very',
            'really', 'you', 'your', 'we', 'our', 'my', 'me', 'i', 'if', 'when', 'where',
            'what', 'how', 'why', 'who', 'which', 'there', 'here', 'all', 'every', 'some',
            'into', 'from', 'about', 'than', 'more', 'also', 'like', 'even', 'get', 'got'}

    all_words = re.findall(r'\b[a-zA-Z]{3,}\b', transcript.lower())
    word_freq = {}
    for w in all_words:
        if w not in stop:
            word_freq[w] = word_freq.get(w, 0) + 1
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:6]
    topic_tags = ['#' + w for w, _ in top_words]

    # Platform-specific generic tags
    generic_tags = {
        'tiktok': ['#fyp', '#foryou', '#facts', '#didyouknow', '#learning', '#viral'],
        'instagram': ['#reels', '#facts', '#didyouknow', '#explore', '#viral', '#educational'],
        'youtube_shorts': ['#shorts', '#facts', '#didyouknow', '#viral', '#education']
    }
    platform_tags = generic_tags.get(platform, generic_tags['instagram'])

    # CTA
    ctas = [
        "Follow for more mind-blowing facts!",
        "Save this for later!",
        "Share with someone who needs to know this!",
        "Drop a comment if you knew this!",
        "Which fact surprised you the most?"
    ]
    import random
    cta = random.choice(ctas)

    # Build caption
    caption_parts = []
    caption_parts.append(hook)
    if key_fact:
        caption_parts.append("")
        caption_parts.append(key_fact)
    caption_parts.append("")
    caption_parts.append(cta)
    caption_parts.append("")
    all_tags = topic_tags[:4] + platform_tags[:5]
    caption_parts.append(' '.join(all_tags))

    return '\n'.join(caption_parts)


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration (without API keys)."""
    # Get available models from replicate_api
    models = []
    if REPLICATE_API_AVAILABLE:
        try:
            models = replicate_api.get_available_models()
        except:
            pass

    safe_config = {
        'platforms': CONFIG['platforms'],
        'sfx': {'enabled': CONFIG['sfx'].get('enabled', True), 'triggers': list(CONFIG['sfx'].get('triggers', CONFIG['sfx']).keys())},
        'video': CONFIG['video'],
        'effects': CONFIG.get('effects', {}),
        'animation': CONFIG.get('animation', {}),
        'image_models': models,
        'default_image_model': replicate_api.DEFAULT_MODEL if REPLICATE_API_AVAILABLE else 'flux-schnell'
    }
    return jsonify(safe_config)


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available image generation models."""
    if not REPLICATE_API_AVAILABLE:
        return jsonify({'error': 'Replicate API not available'}), 500

    models = replicate_api.get_available_models()
    return jsonify({
        'models': models,
        'default': replicate_api.DEFAULT_MODEL
    })


@app.route('/api/config/platform', methods=['POST'])
def update_platform_config():
    """Update platform-specific caption settings."""
    data = request.get_json()
    platform = data.get('platform')
    settings = data.get('settings', {})
    
    if platform not in CONFIG['platforms']:
        return jsonify({'error': 'Unknown platform'}), 400
    
    # Update allowed settings
    allowed = ['font', 'font_size', 'font_color', 'outline_color', 'outline_width', 
               'position_bottom', 'max_width_percent', 'highlight_color']
    
    for key in allowed:
        if key in settings:
            CONFIG['platforms'][platform][key] = settings[key]
    
    # Save config
    with open(CONFIG_PATH, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    return jsonify({'success': True, 'config': CONFIG['platforms'][platform]})


if __name__ == '__main__':
    app.run(debug=True, port=5001, host='127.0.0.1')
