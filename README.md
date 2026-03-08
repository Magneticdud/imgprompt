# Multi-Model Image Editor

A simple tool to edit or create images using various models via API (OpenAI, Google, Black Forest Labs, Bytedance, Sourceful, etc.) via a terminal user interface (TUI).

## Features
- Direct image path input via CLI.
- Interactive file selection if no path is provided.
- **Batch processing**: Process multiple images with the same model and prompt.
- Dynamic cost calculation and display.
- Selection of Resolution and Quality.
- Pre-made and custom prompts.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file with your API Keys:
   ```env
   OPENAI_API_KEY=your_openai_key
   GOOGLE_API_KEY=your_google_key
   OPENROUTER_API_KEY=your_openrouter_key
   ```

## Usage
Run the script:
```bash
python imgedit.py [options] [image_paths...]
```

### Basic Usage
```bash
# Single image (interactive selection if no path provided)
python imgedit.py
python imgedit.py photo.jpg

# Text-to-Image mode (no input image)
python imgedit.py --free
```

### Batch Processing
Pass multiple images to process them all with the same model, resolution, quality, and prompt:

```bash
# Multiple specific images
python imgedit.py photo1.jpg photo2.jpg photo3.jpg

# Glob pattern (all JPG files)
python imgedit.py "*.jpg"

# Glob pattern (all PNG files in a folder)
python imgedit.py ./folder/*.png

# All images in a directory
python imgedit.py ./my_folder

# Combine multiple inputs
python imgedit.py img1.jpg "*.png" ./other_folder
```

When you provide 2+ images, the script automatically enters **batch mode**:
- Shows total cost estimate (per image × number of images)
- Processes each image sequentially
- Reports success/failure count at the end

### Supported Image Formats
- PCX, PNG, JPG/JPEG, BMP, TIFF, WEBP

## Supported Models & Costs
- **OpenAI**: `gpt-image-1.5`, `gpt-image-1-mini`. Official documentation lists prices that don't match reality (like 2-3x more expensive than what's supposed to be), so I'm doing trial&error to find the actual prices.
- **Google (Nano Banana)** (direct or via OpenRouter): 
  - `gemini-2.5-flash-image`: $0.04 per image.
  - `gemini-3.1-flash-image-preview`: $0.07 (1K), $0.10 (2K), $0.15 (4K).
  - `gemini-3-pro-image-preview`: $0.15 (1K/2K) or $0.25 (4K).
- **OpenRouter**:
  - `bytedance-seed/seedream-4.5`: $0.04 per image (any size).
  - `black-forest-labs/flux.2-klein-4b`: $0.014 (1K), $0.017 (2K).
  - `black-forest-labs/flux.2-flex`: Output $0.06 (1K), $0.24 (2K); Input $0.06/MP.
  - `black-forest-labs/flux.2-pro`: Output $0.03 (1K), $0.075 (2K); Input $0.015/MP.
  - `black-forest-labs/flux.2-max`: Output $0.07 (1K), $0.16 (2K); Input $0.03/MP.
  - `sourceful/riverflow-v2-fast`: $0.02 (1K), $0.04 (2K).
  - `sourceful/riverflow-v2-pro`: $0.15 (1K/2K), $0.33 (4K).

Note: Black Forest Labs models cap output at 4MP, so requesting higher resolutions is useless. Input images >4MP are automatically downscaled.

## License
This is licensed under GPL-3.0-or-later
