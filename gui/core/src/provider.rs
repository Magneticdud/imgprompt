use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::ovh::OvhProvider;
use crate::request::GenerationRequest;

/// Options a provider offers for one wizard step, plus the preselected one.
/// Python returns this as a `(choices, default)` tuple.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Choices {
    pub options: Vec<String>,
    pub default: String,
}

impl Choices {
    pub fn single(option: impl Into<String>) -> Self {
        let option = option.into();
        Self { options: vec![option.clone()], default: option }
    }
}

/// Result of `resolve_resolution`: Python's `(res_key, width, height)`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Resolution {
    pub res_key: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

/// Result of `resolve_quality`: Python's `(quality_key, cost)`, cost in USD
/// for one API call.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Quality {
    pub quality_key: String,
    pub cost: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("network error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("{provider} returned HTTP {status}: {body}")]
    Api { provider: &'static str, status: u16, body: String },
    #[error("{0}")]
    BadResponse(String),
    #[error("could not save image: {0}")]
    Io(#[from] std::io::Error),
}

/// Port of Python's `ImageProvider` ABC. The GUI drives its menus through
/// the same four hooks the terminal wizard uses, so provider-specific
/// capability logic stays out of the frontend.
pub trait ImageProvider: Send + Sync {
    fn provider_name(&self) -> &'static str;

    fn supported_models(&self) -> Vec<String>;

    fn default_model(&self) -> String {
        self.supported_models().into_iter().next().unwrap_or_default()
    }

    fn supports_batch(&self) -> bool {
        true
    }

    fn supports_dual(&self) -> bool {
        false
    }

    /// Reference images accepted in one combined call; None = no advertised bound.
    fn max_input_images(&self, _model: &str) -> Option<u32> {
        None
    }

    fn min_input_images(&self, _model: &str) -> u32 {
        0
    }

    /// One-line, user-facing note about how the provider will be reached
    /// (e.g. authenticated vs anonymous rate limits). None when there is
    /// nothing worth showing.
    fn status_note(&self) -> Option<String> {
        None
    }

    fn resolution_choices(&self, model: &str, image_path: Option<&str>) -> Choices;

    fn resolve_resolution(&self, model: &str, selection: &str) -> Resolution;

    fn quality_choices(&self, model: &str, resolution: &Resolution, image_path: Option<&str>) -> Choices;

    fn resolve_quality(&self, model: &str, resolution: &Resolution, selection: &str) -> Quality;

    /// Execute the request and save the result(s) into `output_dir`. Unlike
    /// the Python `run`, which prints its outcome, this returns the saved
    /// paths so a GUI can show them.
    fn run(&self, request: &GenerationRequest, output_dir: &Path) -> Result<Vec<PathBuf>, ProviderError>;
}

/// Port of `PROVIDER_MAP` in `imgedit.py`. Only OVH exists in this prototype.
pub fn all_providers() -> Vec<Box<dyn ImageProvider>> {
    vec![Box::new(OvhProvider::from_env())]
}

pub fn provider_by_name(name: &str) -> Option<Box<dyn ImageProvider>> {
    all_providers().into_iter().find(|p| p.provider_name() == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_provider_offers_its_own_defaults() {
        // The GUI preselects the defaults, so they must be valid options.
        for provider in all_providers() {
            for model in provider.supported_models() {
                let res = provider.resolution_choices(&model, None);
                assert!(res.options.contains(&res.default), "{model}");
                let resolved = provider.resolve_resolution(&model, &res.default);
                let quality = provider.quality_choices(&model, &resolved, None);
                assert!(quality.options.contains(&quality.default), "{model}");
            }
        }
    }

    #[test]
    fn lookup_by_name() {
        assert!(provider_by_name("OVH").is_some());
        assert!(provider_by_name("Nope").is_none());
    }
}
