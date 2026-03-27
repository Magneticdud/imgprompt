BACK_OPTION = "← Go back"

# Pricing constants (approximate, for reference only)
COSTS = {
    "gpt-image-1.5": {
        "Low": {"1024x1024": 0.06, "1024x1536": 0.07, "1536x1024": 0.07},
        "Medium": {"1024x1024": 0.07, "1024x1536": 0.11, "1536x1024": 0.11},
        "High": {"1024x1024": 0.133, "1024x1536": 0.26, "1536x1024": 0.26},
    },
    "gpt-image-1-mini": {
        "Low": {"1024x1024": 0.01, "1024x1536": 0.007, "1536x1024": 0.007},
        "Medium": {"1024x1024": 0.011, "1024x1536": 0.016, "1536x1024": 0.016},
        "High": {"1024x1024": 0.036, "1024x1536": 0.054, "1536x1024": 0.054},
    },
    "gemini-2.5-flash-image": {
        "1K": {"fixed": 0.04},
    },
    "gemini-3.1-flash-image-preview": {
        "1K": {"fixed": 0.07},
        "2K": {"fixed": 0.10},
        "4K": {"fixed": 0.15},
    },
    "gemini-3-pro-image-preview": {
        "1K": {"fixed": 0.14},
        "2K": {"fixed": 0.14},
        "4K": {"fixed": 0.25},
    },
    "sourceful/riverflow-v2-fast": {
        "1K": {"fixed": 0.02},
        "2K": {"fixed": 0.04},
    },
    "sourceful/riverflow-v2-pro": {
        "1K": {"fixed": 0.15},
        "2K": {"fixed": 0.15},
        "4K": {"fixed": 0.33},
    },
    "bytedance-seed/seedream-4.5": {
        "1K": {"fixed": 0.04},
        "2K": {"fixed": 0.04},
        "4K": {"fixed": 0.04},
    },
    "black-forest-labs/flux.2-klein-4b": {
        "1K": {"fixed": 0.014},
        "2K": {"fixed": 0.017},
    },
    "black-forest-labs/flux.2-flex": {
        "1K": {"fixed": 0.06},
        "2K": {"fixed": 0.24},
        "input_mp_rate": 0.06,
    },
    "black-forest-labs/flux.2-pro": {
        "1K": {"fixed": 0.03},
        "2K": {"fixed": 0.075},
        "input_mp_rate": 0.015,
    },
    "black-forest-labs/flux.2-max": {
        "1K": {"fixed": 0.07},
        "2K": {"fixed": 0.16},
        "input_mp_rate": 0.03,
    },
}

# OpenRouter prefixed versions for Gemini models
COSTS["google/gemini-2.5-flash-image"] = COSTS["gemini-2.5-flash-image"]
COSTS["google/gemini-3.1-flash-image-preview"] = COSTS["gemini-3.1-flash-image-preview"]
COSTS["google/gemini-3-pro-image-preview"] = COSTS["gemini-3-pro-image-preview"]

GEMINI_RESOLUTIONS = {
    "1:1": "1024x1024",
    "2:3": "832x1248",
    "3:2": "1248x832",
    "3:4": "864x1184",
    "4:3": "1184x864",
    "4:5": "896x1152",
    "5:4": "1152x896",
    "9:16": "768x1344",
    "16:9": "1344x768",
    "21:9": "1536x672",
    "1:4": "512x2048",
    "4:1": "2048x512",
    "1:8": "512x4096",
    "8:1": "4096x512",
}

OPENROUTER_RESOLUTIONS = {
    "1:1": "1024x1024",
    "2:3": "832x1248",
    "3:2": "1248x832",
    "3:4": "864x1184",
    "4:3": "1184x864",
    "4:5": "896x1152",
    "5:4": "1152x896",
    "9:16": "768x1344",
    "16:9": "1344x768",
    "21:9": "1536x672",
    "1:4": "512x2048",
    "4:1": "2048x512",
    "1:8": "512x4096",
    "8:1": "4096x512",
}

OPENROUTER_STANDARD_RATIOS = [
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
]

ASPECT_RATIO_VALUES = {
    "1:1": 1.0,
    "2:3": 2 / 3,
    "3:2": 3 / 2,
    "3:4": 3 / 4,
    "4:3": 4 / 3,
    "4:5": 4 / 5,
    "5:4": 5 / 4,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
    "1:4": 1 / 4,
    "4:1": 4.0,
    "1:8": 1 / 8,
    "8:1": 8.0,
    "1024x1024 (Square)": 1.0,
    "1024x1536 (Vertical)": 1024 / 1536,
    "1536x1024 (Horizontal)": 1536 / 1024,
}

PRESET_PROMPTS_EDIT = [
    "Outpaint the provided image, maintain all existing details. Preserve the exact composition and identity.",
    "The quality of this logo is poor, recreate it faithfully as if it were vector-based, with sharp edges and limited colors.",
    "Upscale this photo 4x. Preserve the exact composition and identity. Remove JPEG artifacts and noise, enhance real details only. Do not add or remove objects. Do not change facial features. Do not hallucinate text or logos; if unreadable, keep it unreadable. High-resolution output.",
    "Object Removal (High Quality)",
    "Convert this photo into a classic oil painting style.",
    'Restore this photograph using **strict conservation restoration**. Remove only physical damage: **tears, scratches, scuffs, dust spots, stains, crease lines, and fold marks**. **Do not change anything else.** Keep **exactly** the original composition, framing, geometry, perspective, colors, white balance, exposure, contrast, saturation, grain, sharpness, and texture. Do **not** add, remove, or alter any objects, people, faces, hair, clothing, background details, text, logos, or patterns. Do **not** beautify, retouch skin, or "improve" lighting. Reconstruct missing areas by copying/repairing from the **nearest surrounding pixels** so the result matches the original. Output a **1:1 faithful restoration** at the same resolution.',
    "Transform the input photo into a Japanese manga illustration. Preserve the person identity, pose, clothing, and background composition. Clean black ink lineart, confident contours, simplified shapes, screentone shading, high-contrast black and white, crisp lines, minimal gray tones, manga panel style, detailed eyes and hair with ink strokes, no photorealistic texture.",
    "Convert the input photo into a high-quality anime illustration. Preserve identity and facial features. Cel shading, clean linework, smooth gradient highlights, stylized but realistic proportions, vibrant but controlled colors, sharp eyes, defined hair shapes, studio anime lighting.",
    "Give this portrait a 1950s vintage film look.",
    "Turn the photo into a shoujo manga style illustration. Delicate lineart, soft screentones, elegant facial features, sparkly eyes, light blush, airy hair highlights, romantic composition, clean black-and-white manga look.",
    "Transform the uploaded photo into a black-and-white graphite pencil drawing. Use clean line art with cross-hatching for shadows and volume, visible paper texture, and no solid black fills. Keep the exact composition, subject identity, pose, proportions, and camera framing from the original photo. Simplify the background slightly but keep it consistent. No color. Avoid: color, watercolor, oil paint, digital painting, CGI/3D, cartoon/anime, vector-clean outlines, automatic sketch filter look, blur, noisy artifacts, soft airbrushed shading, heavy solid blacks.",
    "Convert the photo into a classic pencil illustration style: precise ink-like pencil outlines, diagonal and cross-hatching for skies/shadows, graphite-only shading (no smooth airbrush gradients), detailed textures on hair/clothing, and a sketchbook look. Maintain the original photo composition and subject identity exactly. Monochrome only. Avoid: color, watercolor, oil paint, digital painting, CGI/3D, cartoon/anime, vector-clean outlines, automatic sketch filter look, blur, noisy artifacts, soft airbrushed shading, heavy solid blacks.",
    "Restore this scanned page with maximum fidelity. Only perform non-destructive cleanup: remove dust/specks, scan noise, paper texture and stains; normalize the halftone/screen pattern to be uniform; correct slight skew. Do NOT redraw, reinterpret, or invent any content. Preserve all original linework, shapes, proportions, fonts, and text exactly. No style changes. Output a clean, flat, high-resolution image that matches the original as closely as possible.",
    "Rebuild the business card as a flat print file. Canvas size: 91×61 mm including 3 mm bleed on all sides (final trim 85×55 mm). Keep all text inside a 4 mm safe margin from the trim edge. Match the original layout from the reference photo. Output: 300 DPI.",
    "Perform conservative color restoration only on the provided 1970s photo. Correct color cast (yellow/magenta/green), restore faded colors, and rebalance white balance to a natural analog-photo look. Do not change any details: keep identical geometry, composition, crop, perspective, faces, skin texture, hair, edges, background, text, film grain, dust, scratches, stains, and any imperfections. No enhancement: no denoise, no sharpening, no deblur, no upscaling, no HDR, no relighting, no beautification. Output must match the original framing and resolution; only chroma/tonal color values may change.",
    "Convert this image into a clean, black and white line art. Use sharp black outlines on a pure white background. Remove all shading, colors, and gradients. It must look like a high-quality adult coloring book page, staying faithful to the original subject and background details.",
    """Perform a strictly conservative photo restoration on the provided image. Goal: improve readability while maintaining absolute faithfulness to the original photo. Allowed adjustments only:
1. neutralize the strong blue/purple color cast with a realistic daylight white balance
2. exposure and contrast correction (no HDR, no dramatic changes)
3. gentle noise/grain reduction while preserving natural film grain
Hard constraints: do not add, remove, move, or alter any real objects or people; do not change faces, bodies, clothing, background, geometry, perspective, cropping, or composition. Do not invent missing details. No style transfer. Output must look like the same photograph, only corrected.""",
    "Custom Prompt",
]

PRESET_PROMPTS_GENERATE = [
    "A futuristic cyberpunk cityscape at night, neon lights, rain, high detail.",
    "A cute minimalist vector logo of a fox.",
    "A photorealistic portrait of an astronaut on Mars, cinematic lighting.",
    "Abstract geometric patterns, vibrant colors, 3d render style.",
    "A serene japanese garden with cherry blossoms, watercolor style.",
    "Isometric view of a cozy coffee shop interior.",
    "A retro-style BW lettering with thick outline",
    "1990s Memphis Style Logo",
    "Business Card",
    "APPROVED Stamp",
    "Generic Logotype",
    "Comic Book Style Text",
    "Custom Prompt",
]

PRESET_PROMPTS_DUAL = [
    "Combine the contents of IMG_1 and IMG_2 into a coherent scene.",
    "Use the composition of IMG_1 and the style of IMG_2.",
    "IMG_1 is the subject, IMG_2 is the background.",
    "Create a vintage etching / engraved illustration double exposure using two input photos. Use IMG_1 as the main subject silhouette and keep its pose, proportions, and outline faithful. Use IMG_2 as the internal scene, visible only inside the silhouette of IMG_1 (no spill outside the outline). Convert everything to black-and-white ink linework with cross-hatching and etched shading, consistent line weight, high detail. Fit and scale IMG_2 to the silhouette while preserving its aspect ratio; adjust position for a pleasing composition. Clean white background, no text, no frame, no extra objects.The outer area must remain blank white; all texture must be inside the silhouette only.",
    "Custom Prompt",
]
