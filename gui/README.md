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

## Windows x64 build (cross-compiled from Fedora)

Needs `mingw64-gcc` and the Rust std for `x86_64-pc-windows-gnu`
(`rustup target add x86_64-pc-windows-gnu`).

```bash
cd gui
CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc \
CC_x86_64_pc_windows_gnu=x86_64-w64-mingw32-gcc \
AR_x86_64_pc_windows_gnu=x86_64-w64-mingw32-ar \
cargo build --release -p imgprompt-gui --features tauri/custom-protocol \
  --target x86_64-pc-windows-gnu
```

`tauri/custom-protocol` is what `tauri build` enables for production: the UI
is embedded in the exe. With the GNU target the exe loads
`WebView2Loader.dll` dynamically, so ship it alongside (cargo copies it to
`target/x86_64-pc-windows-gnu/release/`). Outside the repo, `.env` is read
from the exe's folder.

To work on the layout alone, open `app/ui/index.html` in a browser: without
Tauri, `main.js` falls back to canned answers.
