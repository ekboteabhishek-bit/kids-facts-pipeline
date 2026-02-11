# Kids Facts Video Pipeline

Automated video production pipeline for educational short-form content. Upload your hook, end, and audio → get AI-generated b-roll → assemble final video with effects.

## Features

- 📤 **Upload Interface** — Drop hook video, end video, and voiceover audio
- 📝 **Script Analysis** — Automatically breaks script into timed shots
- 🎬 **AI B-Roll Generation** — Uses Kling via Fal AI to generate visuals
- 🎨 **Consistent Style** — Auto color grading for visual continuity
- ✅ **Review System** — Approve, reject, or regenerate individual shots
- ✂️ **Smart Trimming** — 5-second Kling clips auto-trimmed to 2-4 seconds
- ✨ **Remotion Effects** — Auto-detected zoom pulses, screen shake, emoji pops, kinetic captions
- 💬 **Platform Captions** — Configurable for TikTok, Instagram, YouTube
- 🔊 **Auto SFX** — Detects trigger words and adds sound effects

## Quick Start

### 1. Install Python Dependencies

```bash
cd kids-facts-pipeline
pip install -r requirements.txt
```

### 2. Install Remotion (Optional but Recommended)

For the effects layer, you need Node.js and Remotion:

```bash
cd remotion
npm install
cd ..
```

### 3. Configure API Key

Edit `config.json` and add your Fal AI API key:

```json
{
  "fal_api_key": "YOUR_FAL_API_KEY_HERE",
  ...
}
```

Or set as environment variable:
```bash
export FAL_KEY=your_api_key_here
```

### 4. Install FFmpeg

The pipeline requires FFmpeg for video processing:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 5. Run the Server

```bash
./start.sh
# or
python server.py
```

Open http://localhost:5000 in your browser.

## Remotion Effects

The pipeline automatically detects trigger words in your script and applies effects:

| Trigger Words | Effects Applied |
|---------------|-----------------|
| boom, explode, burst, pop | Screen shake + Flash + Particles + 💥 emoji |
| tiny, small, little, inside | Zoom pulse (reveal) |
| push, pressure, build | Subtle zoom (tension) |
| love, fluffy, amazing | Emoji pop (✨ ❤️ 🍿) |
| Numbers (10 million, 50%) | Zoom pulse + highlight |

### Effect Types

- **Screen Shake** — Camera shake for impact moments
- **Zoom Pulse** — Subtle scale animation for emphasis
- **Flash** — Quick white flash for explosions
- **Emoji Pop** — Animated emoji that pops in and floats
- **Particle Burst** — Colorful particles exploding outward
- **Kinetic Captions** — Word-by-word animated captions with highlights

### Disabling Effects

In the Assembly step, uncheck "Apply Remotion Effects" to render without effects.

## Usage

### Step 1: Upload

1. Drop your **hook video** (your on-camera intro, ~10 sec)
2. Drop your **end video** (your on-camera outro, ~10 sec)
3. Drop your **audio file** (voiceover for the content section)
4. Paste your **script** (what you say in the voiceover)
5. Select target **platform** (TikTok, Instagram, YouTube)
6. Click "Analyze Script & Create Project"

### Step 2: Review

1. Click "Generate All Clips" to start AI generation
2. Wait for clips to generate (watch the progress bar)
3. Preview each clip
4. **Approve** good clips ✓
5. **Reject** bad clips ✗
6. **Regenerate** rejected clips with same or edited prompt
7. Once all clips are approved/generated, proceed to assembly

### Step 3: Assemble

1. Review the timeline
2. Adjust caption settings if needed
3. Click "Assemble Final Video"
4. Download your video!

## Project Structure

```
kids-facts-pipeline/
├── server.py           # Flask backend
├── config.json         # Configuration
├── requirements.txt    # Python dependencies
├── static/
│   └── index.html      # Web interface
├── uploads/            # Uploaded files (per project)
├── clips/              # Generated AI clips
├── outputs/            # Final assembled videos
├── sfx/                # Sound effect files
└── temp/               # Temporary processing files
```

## Configuration

### Platform Caption Settings

Each platform has customizable caption settings:

```json
"platforms": {
  "tiktok": {
    "font": "Nunito-Bold",
    "font_size": 64,
    "position_bottom": 180,
    "highlight_color": "#4A90A4"
  }
}
```

Adjust `position_bottom` to keep captions in the safe zone for each platform's UI.

### Style Settings

Control the AI b-roll visual style:

```json
"style": {
  "prompt_suffix": "warm color temperature, soft diffused lighting...",
  "color_grade": {
    "brightness": 0.02,
    "contrast": 1.05,
    "saturation": 0.92,
    "warmth": 0.08
  }
}
```

### Sound Effects

Auto-detected trigger words:

| Word | Sound |
|------|-------|
| boom, explode | impact.wav |
| pop, burst | pop.wav |
| whoosh, fast | whoosh.wav |
| million, billion | ding.wav |

Add your own `.wav` files to the `sfx/` folder and update `config.json`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/projects` | POST | Create new project |
| `/api/projects/<id>` | GET | Get project details |
| `/api/projects/<id>/generate` | POST | Start clip generation |
| `/api/projects/<id>/shots/<n>/approve` | POST | Approve a shot |
| `/api/projects/<id>/shots/<n>/reject` | POST | Reject a shot |
| `/api/projects/<id>/shots/<n>/regenerate` | POST | Regenerate a shot |
| `/api/projects/<id>/assemble` | POST | Assemble final video |
| `/api/projects/<id>/download` | GET | Download final video |

## Troubleshooting

### "Clip generation failed"
- Check your Fal API key is valid
- Ensure you have API credits
- Try a simpler prompt

### "FFmpeg not found"
- Install FFmpeg and ensure it's in your PATH
- On Windows, add FFmpeg bin folder to system PATH

### "Video assembly failed"
- Check all required files exist (hook, end, audio)
- Ensure at least one clip is generated
- Check disk space

## License

MIT
