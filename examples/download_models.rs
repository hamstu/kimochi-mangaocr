//! Optional Rust-only provisioning tool. Inference itself never accesses the network.
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::Path,
};
fn hash(path: &Path) -> std::io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("../scripts/model-manifest.json"))?;
    let dir = std::env::args().nth(1).unwrap_or_else(|| "models".into());
    let dir = Path::new(&dir);
    fs::create_dir_all(dir)?;
    let repo = manifest["repository"].as_str().unwrap();
    let revision = manifest["revision"].as_str().unwrap();
    let mut files = manifest["sha256"].as_object().unwrap().clone();
    let detector_name = manifest["detector"]["filename"].as_str().unwrap();
    files.insert(
        detector_name.to_string(),
        manifest["detector"]["sha256"].clone(),
    );
    for (name, digest) in &files {
        let expected = digest.as_str().unwrap();
        let dest = dir.join(name);
        if dest.is_file() && hash(&dest)? == expected {
            eprintln!("Verified {name}");
            continue;
        }
        let part = dir.join(format!("{name}.part"));
        // Exclusive creation prevents competing downloads from corrupting one another.
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&part)?;
        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            eprintln!("Downloading {name} at {revision}");
            let url = if name == detector_name {
                manifest["detector"]["url"].as_str().unwrap().to_string()
            } else {
                format!("https://huggingface.co/{repo}/resolve/{revision}/{name}")
            };
            let mut response = ureq::get(url).call()?;
            std::io::copy(&mut response.body_mut().as_reader(), &mut file)?;
            file.flush()?;
            file.sync_all()?;
            if hash(&part)? != expected {
                return Err(format!("SHA-256 mismatch for {name}").into());
            }
            Ok(())
        })();
        drop(file);
        if let Err(e) = result {
            let _ = fs::remove_file(&part);
            return Err(e);
        }
        fs::rename(&part, &dest)?;
        eprintln!("Installed {name}");
    }
    Ok(())
}
