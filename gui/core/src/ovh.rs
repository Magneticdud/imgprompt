use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine;
use serde_json::{json, Value};

use crate::images::save_image_bytes;
use crate::provider::{Choices, ImageProvider, ProviderError, Quality, Resolution};
use crate::request::GenerationRequest;

pub const OVH_ENDPOINT: &str = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/images/generations";
pub const TOKEN_ENV: &str = "OVH_AI_ENDPOINTS_ACCESS_TOKEN";

pub struct OvhProvider {
    token: Option<String>,
}

impl OvhProvider {
    pub fn from_env() -> Self {
        Self::new(std::env::var(TOKEN_ENV).ok())
    }

    /// Empty tokens and the `your_...` placeholder from `.env.example` count
    /// as anonymous access, as in the Python provider.
    pub fn new(token: Option<String>) -> Self {
        let token = token.filter(|t| !t.is_empty() && !t.starts_with("your_"));
        Self { token }
    }

    pub fn is_authenticated(&self) -> bool {
        self.token.is_some()
    }
}

fn payload(request: &GenerationRequest) -> Value {
    json!({
        "model": request.model,
        "prompt": request.prompt,
        "size": request.res_key,
        "response_format": "b64_json",
    })
}

fn extract_b64(response: &Value) -> Option<&str> {
    response.get("data")?.get(0)?.get("b64_json")?.as_str()
}

impl ImageProvider for OvhProvider {
    fn provider_name(&self) -> &'static str {
        "OVH"
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["stabilityai/stable-diffusion-xl-base-1.0".into()]
    }

    fn supports_batch(&self) -> bool {
        false
    }

    fn max_input_images(&self, _model: &str) -> Option<u32> {
        // OVH's endpoint is text-to-image only: it accepts no input images.
        Some(0)
    }

    fn status_note(&self) -> Option<String> {
        Some(if self.is_authenticated() {
            "Authenticated access (400 rpm)".into()
        } else {
            format!("Anonymous access (2 rpm) — set {TOKEN_ENV} in .env for more")
        })
    }

    fn resolution_choices(&self, _model: &str, _image_path: Option<&str>) -> Choices {
        Choices::single("1:1")
    }

    fn resolve_resolution(&self, _model: &str, _selection: &str) -> Resolution {
        Resolution { res_key: "1024x1024".into(), width: None, height: None }
    }

    fn quality_choices(&self, _model: &str, _resolution: &Resolution, _image_path: Option<&str>) -> Choices {
        Choices::single("1K (Free)")
    }

    fn resolve_quality(&self, _model: &str, _resolution: &Resolution, _selection: &str) -> Quality {
        Quality { quality_key: "1K".into(), cost: 0.0 }
    }

    fn run(&self, request: &GenerationRequest, output_dir: &Path) -> Result<Vec<PathBuf>, ProviderError> {
        // The blocking client is built here rather than stored on the
        // provider: constructing or dropping it inside an async runtime
        // panics, and providers are created from Tauri's async commands.
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;
        let mut call = client
            .post(OVH_ENDPOINT)
            .header("Accept", "application/json")
            .json(&payload(request));
        if let Some(token) = &self.token {
            call = call.bearer_auth(token);
        }

        let response = call.send()?;
        let status = response.status();
        if !status.is_success() {
            return Err(ProviderError::Api {
                provider: "OVH",
                status: status.as_u16(),
                body: response.text().unwrap_or_default(),
            });
        }

        let body: Value = response.json()?;
        let b64 = extract_b64(&body).ok_or_else(|| {
            ProviderError::BadResponse("Could not retrieve image data from the OVH API response.".into())
        })?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| ProviderError::BadResponse(format!("OVH returned invalid base64: {e}")))?;

        Ok(vec![save_image_bytes(&bytes, output_dir, None)?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_and_empty_tokens_are_anonymous() {
        assert!(!OvhProvider::new(None).is_authenticated());
        assert!(!OvhProvider::new(Some(String::new())).is_authenticated());
        assert!(!OvhProvider::new(Some("your_token_here".into())).is_authenticated());
        assert!(OvhProvider::new(Some("abc123".into())).is_authenticated());
    }

    #[test]
    fn payload_matches_python_provider() {
        let req = GenerationRequest::text_to_image("a fox", "stabilityai/stable-diffusion-xl-base-1.0", "1:1", "1024x1024", "1K");
        assert_eq!(
            payload(&req),
            json!({
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "prompt": "a fox",
                "size": "1024x1024",
                "response_format": "b64_json",
            })
        );
    }

    #[test]
    fn extracts_first_b64_image_or_none() {
        assert_eq!(extract_b64(&json!({"data": [{"b64_json": "QUJD"}]})), Some("QUJD"));
        assert_eq!(extract_b64(&json!({"data": []})), None);
        assert_eq!(extract_b64(&json!({"error": "x"})), None);
    }

    #[test]
    fn free_and_text_only() {
        let p = OvhProvider::new(None);
        let model = p.default_model();
        let res = p.resolve_resolution(&model, "1:1");
        assert_eq!(res.res_key, "1024x1024");
        assert_eq!(p.resolve_quality(&model, &res, "1K (Free)").cost, 0.0);
        assert_eq!(p.max_input_images(&model), Some(0));
    }
}
