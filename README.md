# GPT Image 1.5 POC - Image Editor

A Proof of Concept tool to edit images using OpenAI's `gpt-image-1.5` or `gpt-image-1-mini` models via a terminal user interface (TUI).

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
2. Create a `.env` file with your OpenAI API Key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage
Run the script:
```bash
python imgedit.py [path_to_image]
```
If no image path is provided, the tool will list images in the current directory for selection.

## License

This is licensed under GPL-3.0-or-later
