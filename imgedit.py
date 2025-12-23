import os
import sys
import argparse
from typing import Optional, List
import base64
import questionary
import requests
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants for pricing
COSTS = {
    "Low": {
        "1024x1024": 0.009,
        "1024x1536": 0.013,
        "1536x1024": 0.013
    },
    "Medium": {
        "1024x1024": 0.034,
        "1024x1536": 0.05,
        "1536x1024": 0.05
    },
    "High": {
        "1024x1024": 0.133,
        "1024x1536": 0.20,
        "1536x1024": 0.20
    }
}

PRESET_PROMPTS = [
    "Outpaint the provided image, maintain all existing details.",
    "The quality of this logo is poor, recreate it faithfully as if it were vector-based, with sharp edges and limited colors.",
    "Transform this scene into a cyberpunk style with neon lights and futuristic elements.",
    "Convert this photo into a classic oil painting style.",
    "Add a realistic sunset lighting to this landscape.",
    "Remove the background and replace it with a clean minimalist studio setting.",
    "Enhance the details and sharpness of this image while keeping it natural.",
    "Change the season of this photo to winter, adding snow and frost.",
    "Give this portrait a 1950s vintage film look.",
    "Modify the colors to follow a warm autumnal palette.",
    "Custom Prompt"
]

def get_images_in_cwd() -> List[str]:
    """Returns a list of image files in the current working directory."""
    extensions = ('.pcx', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    return [f for f in os.listdir('.') if f.lower().endswith(extensions)]

def select_image(provided_path: Optional[str]) -> str:
    """Selects an image either from arguments or from a list of files."""
    if provided_path:
        if os.path.isfile(provided_path):
            return provided_path
        else:
            print(f"Error: {provided_path} is not a valid file.")
            sys.exit(1)
    
    images = get_images_in_cwd()
    if not images:
        print("No image files found in the current directory.")
        sys.exit(1)
    
    selected = questionary.select(
        "Select an image to edit:",
        choices=images
    ).ask()
    
    if not selected:
        sys.exit(0)
    return selected

def main():
    parser = argparse.ArgumentParser(description="GPT-Image-1.5 POC Image Editor")
    parser.add_argument("image", nargs="?", help="Path to the image to edit")
    args = parser.parse_args()

    # 1. Select Image
    image_path = select_image(args.image)
    print(f"\nSelected Image: {image_path}")

    # 2. Select Resolution
    resolution = questionary.select(
        "Select resolution:",
        choices=[
            "1024x1024 (Square)",
            "1024x1536 (Vertical)",
            "1536x1024 (Horizontal)"
        ]
    ).ask()
    if not resolution: sys.exit(0)
    res_key = resolution.split(" ")[0]

    # 3. Select Quality and show costs
    quality_choices = []
    for q in ["Low", "Medium", "High"]:
        cost = COSTS[q][res_key]
        quality_choices.append(f"{q} (${cost:.3f})")

    quality_selected = questionary.select(
        "Select quality:",
        choices=quality_choices
    ).ask()
    if not quality_selected: sys.exit(0)
    quality_key = quality_selected.split(" ")[0]

    # 4. Select Prompt
    prompt_selection = questionary.select(
        "Select a prompt or enter a custom one:",
        choices=PRESET_PROMPTS
    ).ask()
    if not prompt_selection: sys.exit(0)

    final_prompt = prompt_selection
    if prompt_selection == "Custom Prompt":
        final_prompt = questionary.text("Enter your custom prompt:").ask()
        if not final_prompt: sys.exit(0)

    # Summary
    final_cost = COSTS[quality_key][res_key]
    print("\n--- Summary ---")
    print(f"Image:      {image_path}")
    print(f"Resolution: {res_key}")
    print(f"Quality:    {quality_key}")
    print(f"Prompt:     {final_prompt}")
    print(f"Total Cost: ${final_cost:.3f}")
    
    confirm = questionary.confirm("Proceed with API call?").ask()
    if not confirm:
        print("Cancelled.")
        return

    # 5. API Call
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    
    print("\nSending request to OpenAI (gpt-image-1.5)...")
    try:      
        with open(image_path, "rb") as image_file:
            # Using the edit endpoint (or generation with image input if applicable)
            response = client.images.edit(
                model="gpt-image-1.5",
                image=image_file,
                prompt=final_prompt,
                n=1,
                size=res_key,
                quality=quality_key.lower()
            )

        # Support both direct URL and base64 response structures
        image_url = None
        image_b64 = None
        
        if hasattr(response, 'data') and len(response.data) > 0:
            image_url = getattr(response.data[0], 'url', None)
            image_b64 = getattr(response.data[0], 'b64_json', None)
        elif isinstance(response, dict) and 'data' in response:
            image_url = response['data'][0].get('url')
            image_b64 = response['data'][0].get('b64_json')

        if image_url or image_b64:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"edited_{timestamp}_{os.path.basename(image_path)}"
            
            if image_url:
                print(f"\nSuccess! Edited image available at:\n{image_url}")
                print(f"Downloading and saving to {filename}...")
                img_data = requests.get(image_url).content
            else:
                print(f"\nSuccess! Received base64 image data.")
                print(f"Decoding and saving to {filename}...")
                img_data = base64.b64decode(image_b64)
                
            with open(filename, 'wb') as handler:
                handler.write(img_data)
            print(f"File saved successfully as {filename}")
        else:
            print("\nError: Could not retrieve image data from the API response.")
            print(f"Debug Response: {response}")
        
    except Exception as e:
        print(f"\nAn error occurred during the API call: {e}")

if __name__ == "__main__":
    main()
