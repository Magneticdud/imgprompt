//! Port of `imgprompt/history.py`. The file format is identical, so a request
//! saved by the GUI can be replayed with `python imgedit.py --replay` and
//! vice versa.

use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::request::GenerationRequest;

pub const LAST_GENERATION_FILE: &str = ".last_generation.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LastGeneration {
    pub provider: String,
    pub request: GenerationRequest,
}

pub fn save_last_generation(path: &Path, provider: &str, request: &GenerationRequest) -> io::Result<()> {
    let data = LastGeneration { provider: provider.to_string(), request: request.clone() };
    // Pretty-printed with 2-space indent and raw UTF-8, matching
    // json.dump(indent=2, ensure_ascii=False).
    fs::write(path, serde_json::to_string_pretty(&data)?)
}

/// None when the file is missing or unreadable, as in Python.
pub fn load_last_generation(path: &Path) -> Option<LastGeneration> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let path = std::env::temp_dir().join(format!("imgprompt-history-{}.json", std::process::id()));
        let mut req = GenerationRequest::text_to_image("città di notte", "m", "1:1", "1024x1024", "1K");
        req.estimated_cost = Some(0.04);
        save_last_generation(&path, "OVH", &req).unwrap();
        assert!(fs::read_to_string(&path).unwrap().contains("città"));
        let loaded = load_last_generation(&path).unwrap();
        assert_eq!(loaded.provider, "OVH");
        assert_eq!(loaded.request, req);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn missing_file_is_none() {
        assert!(load_last_generation(Path::new("/nonexistent/.last_generation.json")).is_none());
    }
}
