use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Port of `get_image_extension`. Python asks Pillow for the format; magic
/// bytes give the same answer for the three formats it maps, and PNG is the
/// same fallback.
pub fn image_extension(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        ".jpg"
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        ".webp"
    } else {
        ".png"
    }
}

/// Timestamped output path, following `save_image_bytes`: edits are saved
/// next to their source as `edited_<ts>_<name><ext>`, generations as
/// `generated_<ts><ext>`. Timestamps are second-granular, so a `_2`, `_3`...
/// suffix keeps results from the same second from overwriting each other.
pub fn output_path(output_dir: &Path, original: Option<&Path>, ext: &str, timestamp: &str) -> PathBuf {
    let candidate = match original {
        Some(orig) => {
            let stem = orig.file_stem().map(|s| s.to_string_lossy()).unwrap_or_default();
            let dir = orig.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(output_dir);
            dir.join(format!("edited_{timestamp}_{stem}{ext}"))
        }
        None => output_dir.join(format!("generated_{timestamp}{ext}")),
    };
    if !candidate.exists() {
        return candidate;
    }
    let root = candidate.with_extension("");
    (2..)
        .map(|i| PathBuf::from(format!("{}_{i}{ext}", root.display())))
        .find(|p| !p.exists())
        .expect("unbounded counter always finds a free path")
}

pub fn save_image_bytes(bytes: &[u8], output_dir: &Path, original: Option<&Path>) -> io::Result<PathBuf> {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let path = output_path(output_dir, original, image_extension(bytes), &timestamp);
    fs::write(&path, bytes)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("imgprompt-core-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn detects_formats_by_magic_bytes() {
        assert_eq!(image_extension(&[0xFF, 0xD8, 0xFF, 0xE0]), ".jpg");
        assert_eq!(image_extension(b"RIFF\0\0\0\0WEBPVP8 "), ".webp");
        assert_eq!(image_extension(b"\x89PNG\r\n\x1a\n"), ".png");
        assert_eq!(image_extension(b"garbage"), ".png");
    }

    #[test]
    fn names_generated_and_edited_outputs() {
        let dir = Path::new("/out");
        assert_eq!(output_path(dir, None, ".png", "20260101_120000"), dir.join("generated_20260101_120000.png"));
        assert_eq!(
            output_path(dir, Some(Path::new("/photos/cat.jpg")), ".webp", "20260101_120000"),
            Path::new("/photos/edited_20260101_120000_cat.webp")
        );
    }

    #[test]
    fn same_second_results_get_a_counter() {
        let dir = scratch_dir("counter");
        let first = output_path(&dir, None, ".png", "ts");
        fs::write(&first, b"x").unwrap();
        let second = output_path(&dir, None, ".png", "ts");
        assert_eq!(second, dir.join("generated_ts_2.png"));
        fs::write(&second, b"x").unwrap();
        assert_eq!(output_path(&dir, None, ".png", "ts"), dir.join("generated_ts_3.png"));
        fs::remove_dir_all(dir).unwrap();
    }
}
