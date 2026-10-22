//! Rust port of the `imgprompt` Python package, kept free of any GUI code so
//! it can be unit-tested without the Tauri/WebKit stack.
//!
//! Module map (Python → Rust):
//! - `imgprompt/providers/base.py`  → [`request`] + [`provider`]
//! - `imgprompt/providers/ovh_provider.py` → [`ovh`]
//! - `imgprompt/images.py` (saving only) → [`images`]
//! - `imgprompt/history.py` → [`history`]

pub mod history;
pub mod images;
pub mod ovh;
pub mod provider;
pub mod request;

pub use provider::{all_providers, provider_by_name, Choices, ImageProvider, ProviderError, Quality, Resolution};
pub use request::GenerationRequest;
