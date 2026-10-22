use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Mirror of Python's `GenerationRequest`.
///
/// This is the same persisted schema as `.last_generation.json`, so a request
/// written by one side must load on the other. Every field after the first
/// five keeps a serde default for the same reason the Python dataclass gives
/// them one: history files written before the field existed still load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub prompt: String,
    pub model: String,
    pub aspect_ratio: String,
    pub res_key: String,
    pub quality_key: String,
    #[serde(default)]
    pub images: Vec<String>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default = "default_n")]
    pub n: u32,
    #[serde(default)]
    pub extras: Map<String, Value>,
    #[serde(default)]
    pub is_dual: bool,
    #[serde(default)]
    pub is_multi: bool,
    #[serde(default)]
    pub estimated_cost: Option<f64>,
}

fn default_n() -> u32 {
    1
}

impl GenerationRequest {
    /// Text-to-image request with every optional field at its default.
    pub fn text_to_image(
        prompt: impl Into<String>,
        model: impl Into<String>,
        aspect_ratio: impl Into<String>,
        res_key: impl Into<String>,
        quality_key: impl Into<String>,
    ) -> Self {
        Self {
            prompt: prompt.into(),
            model: model.into(),
            aspect_ratio: aspect_ratio.into(),
            res_key: res_key.into(),
            quality_key: quality_key.into(),
            images: Vec::new(),
            width: None,
            height: None,
            n: default_n(),
            extras: Map::new(),
            is_dual: false,
            is_multi: false,
            estimated_cost: None,
        }
    }

    pub fn is_combined(&self) -> bool {
        self.is_dual || self.is_multi
    }

    pub fn is_batch(&self) -> bool {
        self.images.len() > 1 && !self.is_combined()
    }

    pub fn is_text_to_image(&self) -> bool {
        self.images.is_empty()
    }

    pub fn primary_image(&self) -> Option<&str> {
        self.images.first().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_minimal_python_history_without_optional_fields() {
        // Shape of a .last_generation.json request written before `n`,
        // `extras`, `is_dual`, `is_multi` and `estimated_cost` existed.
        let json = r#"{
            "prompt": "a cat", "model": "m", "aspect_ratio": "1:1",
            "res_key": "1024x1024", "quality_key": "1K",
            "images": [], "width": null, "height": null
        }"#;
        let req: GenerationRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.n, 1);
        assert!(req.extras.is_empty());
        assert!(!req.is_combined());
        assert_eq!(req.estimated_cost, None);
    }

    #[test]
    fn batch_and_combined_are_exclusive() {
        let mut req = GenerationRequest::text_to_image("p", "m", "1:1", "1K", "1K");
        assert!(req.is_text_to_image());
        req.images = vec!["a.png".into(), "b.png".into()];
        assert!(req.is_batch());
        req.is_dual = true;
        assert!(!req.is_batch());
        assert_eq!(req.primary_image(), Some("a.png"));
    }
}
