# Preset imgprompt

Copia dei quattro cataloghi del repository. Le etichette interattive sono espanse in `templates.md`.

## PRESET_PROMPTS_EDIT

### 1

Outpaint the provided image, maintain all existing details. Preserve the exact composition and identity.

### 2

The quality of this logo is poor, recreate it faithfully as if it were vector-based, with sharp edges and limited colors.

### 3

Upscale this photo 4x. Preserve the exact composition and identity. Remove JPEG artifacts and noise, enhance real details only. Do not add or remove objects. Do not change facial features. Do not hallucinate text or logos; if unreadable, keep it unreadable. High-resolution output.

### 4

Remove JPEG compression artifacts from this image: blocking, ringing/mosquito noise around edges, color banding, and chroma bleeding. Restore smooth gradients and clean, sharp edges. Keep exactly the same composition, framing, resolution, colors, and content. Do not add, remove, or reinterpret any objects, faces, text, or logos; if text is unreadable, keep it unreadable. No style changes, no beautification, no over-smoothing.

### 5

Object Removal (High Quality)

### 6

Convert this photo into a classic oil painting style.

### 7

Restore this photograph using **strict conservation restoration**. Remove only physical damage: **tears, scratches, scuffs, dust spots, stains, crease lines, and fold marks**. **Do not change anything else.** Keep **exactly** the original composition, framing, geometry, perspective, colors, white balance, exposure, contrast, saturation, grain, sharpness, and texture. Do **not** add, remove, or alter any objects, people, faces, hair, clothing, background details, text, logos, or patterns. Do **not** beautify, retouch skin, or "improve" lighting. Reconstruct missing areas by copying/repairing from the **nearest surrounding pixels** so the result matches the original. Output a **1:1 faithful restoration** at the same resolution.

### 8

Transform the input photo into a Japanese manga illustration. Preserve the person identity, pose, clothing, and background composition. Clean black ink lineart, confident contours, simplified shapes, screentone shading, high-contrast black and white, crisp lines, minimal gray tones, manga panel style, detailed eyes and hair with ink strokes, no photorealistic texture.

### 9

Convert the input photo into a high-quality anime illustration. Preserve identity and facial features. Cel shading, clean linework, smooth gradient highlights, stylized but realistic proportions, vibrant but controlled colors, sharp eyes, defined hair shapes, studio anime lighting.

### 10

Give this portrait a 1950s vintage film look.

### 11

Turn the photo into a shoujo manga style illustration. Delicate lineart, soft screentones, elegant facial features, sparkly eyes, light blush, airy hair highlights, romantic composition, clean black-and-white manga look.

### 12

Transform the uploaded photo into a black-and-white graphite pencil drawing. Use clean line art with cross-hatching for shadows and volume, visible paper texture, and no solid black fills. Keep the exact composition, subject identity, pose, proportions, and camera framing from the original photo. Simplify the background slightly but keep it consistent. No color. Avoid: color, watercolor, oil paint, digital painting, CGI/3D, cartoon/anime, vector-clean outlines, automatic sketch filter look, blur, noisy artifacts, soft airbrushed shading, heavy solid blacks.

### 13

Convert the photo into a classic pencil illustration style: precise ink-like pencil outlines, diagonal and cross-hatching for skies/shadows, graphite-only shading (no smooth airbrush gradients), detailed textures on hair/clothing, and a sketchbook look. Maintain the original photo composition and subject identity exactly. Monochrome only. Avoid: color, watercolor, oil paint, digital painting, CGI/3D, cartoon/anime, vector-clean outlines, automatic sketch filter look, blur, noisy artifacts, soft airbrushed shading, heavy solid blacks.

### 14

Restore this scanned page with maximum fidelity. Only perform non-destructive cleanup: remove dust/specks, scan noise, paper texture and stains; normalize the halftone/screen pattern to be uniform; correct slight skew. Do NOT redraw, reinterpret, or invent any content. Preserve all original linework, shapes, proportions, fonts, and text exactly. No style changes. Output a clean, flat, high-resolution image that matches the original as closely as possible.

### 15

Clean this scan, removing halftones and making everything more uniform.

### 16

Translate Text

### 17

Rebuild the business card as a flat print file. Canvas size: 91×61 mm including 3 mm bleed on all sides (final trim 85×55 mm). Keep all text inside a 4 mm safe margin from the trim edge. Match the original layout from the reference photo. Output: 300 DPI. Do NOT draw any crop marks, trim marks, registration marks, cut lines, corner marks, rulers, or measurement annotations; extend the artwork to the full canvas edges as a single flat design.

### 18

Perform conservative color restoration only on the provided 1970s photo. Correct color cast (yellow/magenta/green), restore faded colors, and rebalance white balance to a natural analog-photo look. Do not change any details: keep identical geometry, composition, crop, perspective, faces, skin texture, hair, edges, background, text, film grain, dust, scratches, stains, and any imperfections. No enhancement: no denoise, no sharpening, no deblur, no upscaling, no HDR, no relighting, no beautification. Output must match the original framing and resolution; only chroma/tonal color values may change.

### 19

Convert this image into a clean, black and white line art. Use sharp black outlines on a pure white background. Remove all shading, colors, and gradients. It must look like a high-quality adult coloring book page, staying faithful to the original subject and background details.

### 20

Perform a strictly conservative photo restoration on the provided image. Goal: improve readability while maintaining absolute faithfulness to the original photo. Allowed adjustments only:
1. neutralize the strong blue/purple color cast with a realistic daylight white balance
2. exposure and contrast correction (no HDR, no dramatic changes)
3. gentle noise/grain reduction while preserving natural film grain
Hard constraints: do not add, remove, move, or alter any real objects or people; do not change faces, bodies, clothing, background, geometry, perspective, cropping, or composition. Do not invent missing details. No style transfer. Output must look like the same photograph, only corrected.

### 21

Upscale without any deformation, and add an empty 10% outpainting margin around the edges, keeping the same background style.

### 22

Upscale this image and adapt it to {target_ratio}. Do not stretch, squash, or distort the original content, and do not crop away any of it. Fill the extra area by outpainting, seamlessly continuing the existing background, lighting, and style. Preserve the exact subject, composition, identity, colors, and text. Remove compression artifacts and enhance real details only. High-resolution output.

### 23

Custom Prompt

## PRESET_PROMPTS_GENERATE

### 1

A futuristic cyberpunk cityscape at night, neon lights, rain, high detail.

### 2

A cute minimalist vector logo of a fox.

### 3

A photorealistic portrait of an astronaut on Mars, cinematic lighting.

### 4

Abstract geometric patterns, vibrant colors, 3d render style.

### 5

A serene japanese garden with cherry blossoms, watercolor style.

### 6

Isometric view of a cozy coffee shop interior.

### 7

A retro-style BW lettering with thick outline

### 8

1990s Memphis Style Logo

### 9

Business Card

### 10

APPROVED Stamp

### 11

Generic Logotype

### 12

Comic Book Style Text

### 13

Pokémon Style Lettering

### 14

Minecraft Style Lettering

### 15

Custom Prompt

## PRESET_PROMPTS_MULTI

### 1

Combine the contents of every input image (IMG_1..IMG_N) into a coherent scene.

### 2

Use the composition of IMG_1 and the style of the remaining input images.

### 3

Blend all input images into a single balanced composition, preserving each subject's identity.

### 4

Compose a group scene from every input image, keeping each subject's identity and pose faithful.

### 5

Use IMG_1 as the background and place the subjects from IMG_2 onward into it naturally.

### 6

Create a seamless collage that combines all input images while preserving their content.

### 7

Custom Prompt

## PRESET_PROMPTS_DUAL

### 1

Combine the contents of IMG_1 and IMG_2 into a coherent scene.

### 2

Use the composition of IMG_1 and the style of IMG_2.

### 3

IMG_1 is the subject, IMG_2 is the background.

### 4

Create a vintage etching / engraved illustration double exposure using two input photos. Use IMG_1 as the main subject silhouette and keep its pose, proportions, and outline faithful. Use IMG_2 as the internal scene, visible only inside the silhouette of IMG_1 (no spill outside the outline). Convert everything to black-and-white ink linework with cross-hatching and etched shading, consistent line weight, high detail. Fit and scale IMG_2 to the silhouette while preserving its aspect ratio; adjust position for a pleasing composition. Clean white background, no text, no frame, no extra objects.The outer area must remain blank white; all texture must be inside the silhouette only.

### 5

Custom Prompt
