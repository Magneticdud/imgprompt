# imgprompt GUI (Tauri prototype)

A desktop front-end that will eventually replace the `imgedit.py` wizard.
**Prototype scope: OVH only**, text-to-image, one image per call.

```
gui/
├── core/   imgprompt-core — Rust port of the provider layer, no GUI deps
│           (request.rs ↔ base.py GenerationRequest, provider.rs ↔ ImageProvider,
│            ovh.rs ↔ ovh_provider.py, images.rs, history.rs)
└── app/    imgprompt-gui — Tauri 2 shell; ui/ is plain HTML/CSS/JS, no bundler
```

The menus are driven by the same provider hooks as the CLI
(`resolution_choices` → `resolve_resolution` → `quality_choices` →
`resolve_quality`), and the cost estimate updates live as you pick options.
`.env` and `.last_generation.json` are read from the repo root, so a GUI run
can be replayed with `python imgedit.py --replay`.

## Prerequisites (Fedora)

```bash
sudo dnf install webkit2gtk4.1-devel libsoup3-devel
```

Rust stable is enough; the Tauri CLI is not needed to run the prototype.

## Run

```bash
cd gui
cargo test -p imgprompt-core     # logic tests, no WebKit needed
cargo run -p imgprompt-gui       # opens the window
```

To work on the layout alone, open `app/ui/index.html` in a browser: without
Tauri, `main.js` falls back to canned answers.
