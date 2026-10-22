#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! Tauri shell around `imgprompt-core`. Every command takes the provider and
//! model by name and asks the provider for its choices, mirroring how
//! `imgedit.py` drives its wizard through the `ImageProvider` hooks.

use std::path::{Path, PathBuf};

use base64::Engine;
use imgprompt_core::history::{save_last_generation, LAST_GENERATION_FILE};
use imgprompt_core::images::image_extension;
use imgprompt_core::{all_providers, provider_by_name, Choices, GenerationRequest, ImageProvider};
use serde::Serialize;

/// Prototype shortcut: the repo root is located from the source tree, so
/// `.env` and `.last_generation.json` are shared with the Python CLI.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn provider(name: &str) -> Result<Box<dyn ImageProvider>, String> {
    provider_by_name(name).ok_or_else(|| format!("Unknown provider: {name}"))
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    max_input_images: Option<u32>,
    min_input_images: u32,
}

#[derive(Serialize)]
struct ProviderInfo {
    name: &'static str,
    default_model: String,
    models: Vec<ModelInfo>,
    status_note: Option<String>,
}

#[derive(Serialize)]
struct Estimate {
    res_key: String,
    quality_key: String,
    cost: f64,
}

#[derive(Serialize)]
struct GeneratedImage {
    path: String,
    data_url: String,
}

#[tauri::command]
fn providers() -> Vec<ProviderInfo> {
    all_providers()
        .into_iter()
        .map(|p| ProviderInfo {
            name: p.provider_name(),
            default_model: p.default_model(),
            models: p
                .supported_models()
                .into_iter()
                .map(|m| ModelInfo {
                    max_input_images: p.max_input_images(&m),
                    min_input_images: p.min_input_images(&m),
                    name: m,
                })
                .collect(),
            status_note: p.status_note(),
        })
        .collect()
}

#[tauri::command]
fn resolution_choices(provider_name: String, model: String) -> Result<Choices, String> {
    Ok(provider(&provider_name)?.resolution_choices(&model, None))
}

#[tauri::command]
fn quality_choices(provider_name: String, model: String, resolution: String) -> Result<Choices, String> {
    let p = provider(&provider_name)?;
    let res = p.resolve_resolution(&model, &resolution);
    Ok(p.quality_choices(&model, &res, None))
}

#[tauri::command]
fn estimate(provider_name: String, model: String, resolution: String, quality: String) -> Result<Estimate, String> {
    let p = provider(&provider_name)?;
    let res = p.resolve_resolution(&model, &resolution);
    let q = p.resolve_quality(&model, &res, &quality);
    Ok(Estimate { res_key: res.res_key, quality_key: q.quality_key, cost: q.cost })
}

#[tauri::command]
fn default_output_dir() -> String {
    std::env::current_dir().unwrap_or_else(|_| repo_root()).display().to_string()
}

#[tauri::command]
async fn generate(
    provider_name: String,
    model: String,
    resolution: String,
    quality: String,
    prompt: String,
    output_dir: String,
) -> Result<Vec<GeneratedImage>, String> {
    let prompt = prompt.trim().to_string();
    if prompt.is_empty() {
        return Err("Write a prompt first.".into());
    }
    let output_dir = PathBuf::from(output_dir);
    if !output_dir.is_dir() {
        return Err(format!("Output folder does not exist: {}", output_dir.display()));
    }

    // The provider call blocks for up to minutes; keep it off the async
    // runtime so the window stays responsive.
    tauri::async_runtime::spawn_blocking(move || {
        let p = provider(&provider_name)?;
        let res = p.resolve_resolution(&model, &resolution);
        let q = p.resolve_quality(&model, &res, &quality);
        let mut request = GenerationRequest::text_to_image(prompt, model, resolution, res.res_key, q.quality_key);
        request.width = res.width;
        request.height = res.height;

        // Saved before the call, like the CLI, so a failed run can be replayed.
        if let Err(e) = save_last_generation(&repo_root().join(LAST_GENERATION_FILE), p.provider_name(), &request) {
            eprintln!("Warning: could not save last generation: {e}");
        }

        let paths = p.run(&request, &output_dir).map_err(|e| e.to_string())?;
        paths
            .into_iter()
            .map(|path| {
                let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
                let mime = match image_extension(&bytes) {
                    ".jpg" => "image/jpeg",
                    ".webp" => "image/webp",
                    _ => "image/png",
                };
                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                Ok(GeneratedImage { path: path.display().to_string(), data_url: format!("data:{mime};base64,{b64}") })
            })
            .collect()
    })
    .await
    .map_err(|e| e.to_string())?
}

fn main() {
    // Same lookup as the CLI's load_dotenv(), plus the repo root so launching
    // from gui/ still finds the keys.
    let _ = dotenvy::from_path(repo_root().join(".env"));
    let _ = dotenvy::dotenv();

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            providers,
            resolution_choices,
            quality_choices,
            estimate,
            default_output_dir,
            generate
        ])
        .run(tauri::generate_context!())
        .expect("error while running imgprompt");
}
