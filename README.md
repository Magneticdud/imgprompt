# Multi-Model POC Image Editor

A Proof of Concept tool to edit images using OpenAI's `gpt-image-1.5` / `gpt-image-1-mini` or Google's `gemini-2.5-flash-image` (Nano Banana) / `gemini-3-pro-image-preview` models via a terminal user interface (TUI).

## Features
- Direct image path input via CLI.
- Interactive file selection if no path is provided.
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
   ```

## Usage
Run the script:
```bash
python imgedit.py [path_to_image]
```
If no image path is provided, the tool will list images in the current directory for selection.

## Supported Models & Costs
- **OpenAI**: `gpt-image-1.5`, `gpt-image-1-mini`. Official documentation lists prices that don't match reality (like 2-3x more expensive than what's supposed to be), so I'm doing trial&error to find the actual prices.
- **Google (Nano Banana)**: 
  - `gemini-2.5-flash-image`: $0.04 per image.
  - `gemini-3-pro-image-preview`: $0.15 (1K/2K) or $0.25 (4K).

## License
This is licensed under GPL-3.0-or-later
