# Multi-Model Image Editor

A simple tool to edit or create images using various models via API (OpenAI, Google, Black Forest Labs, Bytedance, Sourceful, Stable Diffusion XL, etc.) via a terminal user interface (TUI).

## Features
- Direct image path input via CLI.
- Interactive file selection if no path is provided.
- **PDF input**: pass a `.pdf` and it is rasterized to a bitmap before upload (a PNG is saved next to the source); for multi-page PDFs you choose which page to use.
- **Batch processing**: Process multiple images with the same model and prompt.
- Dynamic cost calculation and display.
- Selection of Resolution and Quality.
- Pre-made and custom prompts.
- **Multi-line prompts**: write custom prompts spanning several lines (handy for typographic prompts), or load a prompt from a text file.
- **Replay**: re-run the last generation with identical parameters and prompt.

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
   OVH_AI_ENDPOINTS_ACCESS_TOKEN=your_ovh_token
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

### Multi-line Prompts
When you pick **"Custom Prompt"** (or **"Edit Prompt"** from the summary), the input is multi-line: press **Enter** to submit and **Ctrl+J** to insert a newline. This lets you lay out prompts across several lines, which is useful for typographic generations, e.g.:

```
Create a monochrome typographic illustration with the following text,
on three separate lines:
Hello
World
2026
```

### Prompt from a Text File
Instead of typing the prompt, you can load it from a `.txt` file (newlines preserved). This skips the interactive prompt step:

```bash
# Explicit flag
python imgedit.py photo.jpg --prompt-file prompt.txt
python imgedit.py photo.jpg -p prompt.txt

# Auto-detected: a .txt among the positional arguments is treated as the
# prompt file, in any order. An image + a .txt can be passed together.
python imgedit.py photo.jpg prompt.txt
python imgedit.py prompt.txt photo.jpg

# Works in Text-to-Image mode too
python imgedit.py --free prompt.txt
```

### Replay the Last Generation
Every completed run saves its full request (provider, model, resolution, quality, prompt and input images) to a gitignored `.last_generation.json`. Re-run it verbatim without walking the wizard again:

```bash
# Repeat the exact same generation
python imgedit.py --replay

# Repeat it N times (e.g. for more variations)
python imgedit.py --replay -n 3

# Image arguments are ignored, so you can just press arrow-up to recall
# your previous command and append --replay
python imgedit.py photo.jpg --replay
```

Whenever a previous run is saved, a **"🔁 Replay last generation"** entry also appears in the first interactive menu — the image-selection menu when launched with no arguments, or the provider menu when an image was passed as an argument. This lets you simply press arrow-up to recall your last command and pick replay, without typing `--replay`.

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
- **OpenAI**: `gpt-image-2`. (Discontinued: `gpt-image-1.5`, `gpt-image-1-mini`).
  - ⚠️ **Note**: `gpt-image-2` is also available on OpenRouter, but requests are sent server-side with quality set to `high`, making it significantly more expensive than using OpenAI directly.
- **Google (Nano Banana)** (direct or via OpenRouter):
  - `gemini-3.1-flash-lite-image`: $0.034 per image. **1K only**, supports all 14 aspect ratios of the Gemini 3.x family (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9 + 1:4, 4:1, 1:8, 8:1). Cheapest 1K option — recommended when 2K/4K isn't needed.
  - `gemini-3.1-flash-image`: $0.07 (1K), $0.10 (2K), $0.15 (4K).
  - `gemini-3-pro-image`: $0.15 (1K/2K) or $0.25 (4K).
  - ~~`gemini-2.5-flash-image`~~: removed in this release. Google retires it on 2 Oct 2026; existing `.last_generation.json` entries that still point to it will refuse to replay with an explicit error.

  > ⚠️ **Google direct API is currently untested.** Only the OpenRouter path is verified end-to-end in this project — the `Google` provider in `imgprompt/providers/google_provider.py` prints a banner on first use and may drift from Google's API without warning. Prefer the OpenRouter route for Gemini unless you have a specific reason to hit Google's API directly.
- **OpenRouter**:
  - `bytedance-seed/seedream-4.5`: $0.04 per image (any size).
  - `black-forest-labs/flux.2-klein-4b`: $0.014 (1K), $0.017 (2K).
  - `black-forest-labs/flux.2-flex`: Output $0.06 (1K), $0.24 (2K); Input $0.06/MP.
  - `black-forest-labs/flux.2-pro`: Output $0.03 (1K), $0.075 (2K); Input $0.015/MP.
  - `black-forest-labs/flux.2-max`: Output $0.07 (1K), $0.16 (2K); Input $0.03/MP.
  - `microsoft/mai-image-2.5`: token-billed (output $47/Mtok, input image $8/Mtok, input text $5/Mtok — ≈$0.19 for a typical image; the real charge is reported after each call). No resolution tiers: the model picks the output size from the aspect ratio (1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9). Single image per call (`n` capped at 1 upstream).
  - `sourceful/riverflow-v2.5-fast`: $0.02 (1K), $0.04 (2K).
  - `sourceful/riverflow-v2.5-pro`: $0.15 (1K/2K), $0.33 (4K).
- **OVH AI Endpoints**:
  - `stabilityai/stable-diffusion-xl-base-1.0`: Free (Rate limited: 2 per minute without API key, 400 per minute with API key). Fixed 1024x1024. With such generous rate limits it does not need API keys, but if needed you can [read how to get one](https://help.ovhcloud.com/csm/en-gb-public-cloud-ai-endpoints-getting-started?id=kb_article_view&sysparm_article=KB0065401)

Note: Black Forest Labs models cap output at 4MP, so requesting higher resolutions is useless. Input images >4MP are automatically downscaled.

## Testing
The pure logic (dimension/DPI math, pricing-token estimates, image helpers, and
the replay-history round-trip) is covered by a `pytest` suite under `tests/`.
Network calls to the providers are out of scope and are not tested.

Install the test dependency (once) and run the suite from the project root:

```bash
pip install -r requirements-dev.txt   # installs runtime deps + pytest
pytest
```

Useful variations:

```bash
pytest -v                       # verbose, one line per test
pytest tests/test_presets.py    # run a single file
pytest -k physical_to_pixels    # run tests matching a name
```

`tests/test_presets.py` covers the gpt-image-2 dimension logic, including a grid
test that asserts `auto_adjust_gpt_image2_dims` always returns valid dimensions —
add new cases there when you touch that math.

## OpenRouter Usage Stats
You can track this app's usage on OpenRouter here: [OpenRouter app usage](https://openrouter.ai/apps?url=https%3A%2F%2Fgithub.com/Magneticdud/imgprompt).

## License
This is licensed under GPL-3.0-or-later
