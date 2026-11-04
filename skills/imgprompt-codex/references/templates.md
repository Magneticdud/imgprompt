# Template parametrizzati imgprompt

Fonte: ramo finale di `step_prompt` in `imgedit.py`. Leggere il codice come template, senza eseguirlo. Sostituire gli input interattivi con i dati della conversazione.

```python
    final_prompt = prompt_selection
    if prompt_selection == "Custom Prompt":
        final_prompt = multiline_prompt("Enter your custom prompt:")
        if not final_prompt:
            return BACK_OPTION, prompt_selection
    elif prompt_selection == "Object Removal (High Quality)":
        base_prompt = "Preserve the exact composition and identity. Remove JPEG artifacts and noise, enhance real details only. Do not change facial features. Do not hallucinate text or logos; if unreadable, keep it unreadable. High-resolution output."
        remove_input = questionary.text("What to remove?").ask()
        if not remove_input:
            final_prompt = base_prompt
        else:
            final_prompt = f"{base_prompt} Remove {remove_input}."
    elif prompt_selection == "Translate Text":
        language = (
            questionary.text("Translate into which language?", default="Italian").ask()
            or ""
        ).strip()
        if not language:
            return BACK_OPTION, prompt_selection
        clean_scan = questionary.confirm("Also clean up the scan?", default=False).ask()
        if clean_scan is None:
            return BACK_OPTION, prompt_selection
        cleanup = (
            "Also clean up the scan, non-destructively only: remove dust, specks, scan noise, paper texture, stains, and show-through from the back of the page; "
            "make the paper background uniform; correct slight skew. Do not redraw or reinterpret any non-text content. "
            if clean_scan
            else ""
        )
        final_prompt = (
            f"Translate all visible text in this image into {language}. "
            f"Replace the original text in place with the translation, matching the original layout, position, alignment, font style, weight, size hierarchy, color, and line spacing as closely as possible; "
            f"if a translation is longer, reduce its font size slightly so it fits its original area, never overflow or overlap other elements. "
            f"Translate accurately and completely: headings, body text, captions, labels and callouts inside diagrams, table cells, and footnotes. "
            f"Keep proper nouns, brand names, product and model codes, part numbers, numbers, units, and measurements unchanged. "
            f"Do not alter anything that is not text: keep photos, illustrations, diagrams, icons, logos, background, margins, and composition exactly as they are. "
            f"Do not summarize, omit, or add content. If a word is unreadable, leave it as it is rather than guessing. "
            f"{cleanup}"
            f"Output the same page at the same resolution and aspect ratio."
        )
    elif prompt_selection == "A retro-style BW lettering with thick outline":
        text_input = _ask_required("What text to write?")
        if not text_input:
            return BACK_OPTION, prompt_selection
        final_prompt = (
            f"Create a clean vector-style black and white typographic logo. "
            f'Text: "{text_input}" on two lines, centered and slightly slanted upward to the right. '
            f"Use bold retro script lettering (smooth connected cursive, thick strokes), white fill with a thick black outline. "
            f"Add a large black drop shadow offset down-right to create a strong 3D sticker effect. "
            f"Add a swoosh underline under the second word, also white with black outline and black shadow. "
            f"High contrast, crisp edges, no textures, no gradients, bright green background (it will be keyed out), no extra elements. "
            f"Export as a logo/wordmark."
        )
    elif prompt_selection == "1990s Memphis Style Logo":
        text_input = _ask_required("What text to write?")
        if not text_input:
            return BACK_OPTION, prompt_selection
        final_prompt = (
            f'Create a 1990s Memphis-inspired typographic logo that reads: "{text_input}". '
            f"Style: 90s TV commercial / snack packaging lettering, playful and energetic, slightly italic script with uneven hand-drawn feel, two-tone gradients and airbrushed shading, subtle halftone/print vibe, thin white highlight, no shadow at all, not a thick sticker outline). "
            f"Add a simple zig-zag / squiggle underline in Memphis style. "
            f"Background: solid flat chroma key green (#00FF00), perfectly uniform. "
            f"Strictly avoid: 3D chrome, rainbow neon, glossy metallic look, thick black outline sticker effect, modern esports logo style, glow effects, bevel/emboss, photorealism, extra objects, patterns, textures, collage, food images, shadow. "
            f"Centered composition, clean edges, high resolution."
        )
    elif prompt_selection == "Business Card":
        print("Enter ONLY the main title (e.g. 'DJ Set'):")
        main_title = _ask_required("Main Title:", "Main title is required.")
        if not main_title:
            return BACK_OPTION, prompt_selection

        print(
            "Enter the rest of the details (multiline). Press Alt+Enter or Esc+Enter to submit:"
        )
        details = _ask_required("Details:", "Details are required.", multiline=True)
        if not details:
            return BACK_OPTION, prompt_selection

        final_prompt = (
            f"Create a 2D graphic design for a business card without mockup: no 3D rendering, no scene, no photography, no perspective.\n"
            f"Canvas: 85x55 mm (aspect ratio 1.545:1), horizontal, equivalent to 300 dpi, 3 mm safety margins, text strictly within the safe area.\n"
            f"Text Layout: Centered in a single column. Visual hierarchy: '{main_title}' must be very large and bold. The rest smaller but clearly legible.\n"
            f"Write EXACTLY the following text:\n\n"
            f"{main_title}\n"
            f"{details}\n\n"
            f"Negative constraints: No other text. No QR code. No social media icons. No watermark. No invented logo. Output: flat 2D graphic only."
        )
    elif prompt_selection == "APPROVED Stamp":
        custom_string = _ask_required(
            "Enter the custom text (will be followed by 'APPROVED'):",
            "Custom text is required for this preset.",
        )
        if not custom_string:
            return BACK_OPTION, prompt_selection

        final_prompt = (
            f"Create an APPROVED-style rubber stamp graphic, rectangular with slightly rounded corners. "
            f"Text: '{custom_string} APPROVED' on two lines, red, bold, all caps, with a light distressed texture. "
            f"White background. No extra elements, no gradients, no shadows. Vector/flat style, high resolution."
        )
    elif prompt_selection == "Generic Logotype":
        print(
            "Enter the text for the logotype (multiline). Press Alt+Enter or Esc+Enter to submit:"
        )
        logo_text = _ask_required(
            "Logotype Text:", "Text is required for this preset.", multiline=True
        )
        if not logo_text:
            return BACK_OPTION, prompt_selection

        final_prompt = f"A typographic logo, with centered text, with the following text: {logo_text}"
    elif prompt_selection == "Comic Book Style Text":
        text_input = _ask_required("What text to write?")
        if not text_input:
            return BACK_OPTION, prompt_selection
        final_prompt = (
            f'Comic book style bold text that reads "{text_input}", '
            f"retro superhero font, thick black outline, yellow to red gradient fill, "
            f"halftone dot pattern texture, 3D extruded shadow effect, pop art style, "
            f"white background, high contrast, vintage Marvel/DC comics typography."
        )
    elif prompt_selection == "Pokémon Style Lettering":
        text_input = _ask_required("What name to write?")
        if not text_input:
            return BACK_OPTION, prompt_selection
        final_prompt = (
            f'Write "{text_input}" using a Pokémon-style font, '
            f"white background, yellow text with a blue outline."
        )
    elif prompt_selection == "Minecraft Style Lettering":
        text_input = _ask_required("What name to write?")
        if not text_input:
            return BACK_OPTION, prompt_selection
        final_prompt = (
            f'Write "{text_input}" in Minecraft style, '
            f"with a Minecraft-style background."
        )

```
