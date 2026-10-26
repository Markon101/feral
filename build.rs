//! Compile the optional CUDA kernels at build time with a toolkit that still supports the
//! requested SM architecture. Pascal (sm_61) was removed from CUDA 13, so the search prefers
//! any nvcc whose release is 11.x/12.x. FERAL_NVCC and FERAL_CUDA_ARCH override the choice.
use std::{env, path::PathBuf, process::Command};

fn nvcc_version(nvcc: &str) -> Option<(u32, u32)> {
    let out = Command::new(nvcc).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let release = text.split("release ").nth(1)?;
    let version = release.split(',').next()?.trim();
    let mut parts = version.split('.');
    Some((
        parts.next()?.parse().ok()?,
        parts.next().unwrap_or("0").parse().ok()?,
    ))
}

fn candidates() -> Vec<String> {
    let mut list = Vec::new();
    if let Ok(explicit) = env::var("FERAL_NVCC") {
        if !explicit.is_empty() {
            list.push(explicit);
        }
    }
    for path in [
        "/usr/local/cuda-12.9/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
        "/usr/local/cuda-12.6/bin/nvcc",
        "/usr/local/cuda-12.4/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11.8/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
    ] {
        list.push(path.to_string());
    }
    if let Ok(home) = env::var("HOME") {
        for env_name in ["llama-turbo-61", "agent_env", "hermes"] {
            list.push(format!("{home}/micromamba/envs/{env_name}/bin/nvcc"));
        }
    }
    list.push("nvcc".to_string());
    list
}

fn main() {
    // Any change under src/cuda (including included .cuh files) must recompile the cubin;
    // a stale cubin silently ignores kernel edits.
    println!("cargo:rerun-if-changed=src/cuda");
    for entry in std::fs::read_dir("src/cuda")
        .into_iter()
        .flatten()
        .flatten()
    {
        println!("cargo:rerun-if-changed={}", entry.path().display());
    }
    println!("cargo:rerun-if-env-changed=FERAL_NVCC");
    println!("cargo:rerun-if-env-changed=FERAL_CUDA_ARCH");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let arch = env::var("FERAL_CUDA_ARCH").unwrap_or_else(|_| "61".into());
    let arch: u32 = arch
        .trim()
        .parse()
        .expect("FERAL_CUDA_ARCH must be a numeric SM version, e.g. 61 for Pascal");
    let sm = format!("sm_{arch}");
    let compute = format!("compute_{arch}");
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let source = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("src/cuda/kernels.cu");

    let mut errors = Vec::new();
    for nvcc in candidates() {
        let Some((major, _minor)) = nvcc_version(&nvcc) else {
            continue;
        };
        // CUDA 13 dropped offline compilation for Maxwell/Pascal/Volta.
        if major >= 13 && arch < 70 && env::var("FERAL_NVCC").is_err() {
            errors.push(format!("{nvcc}: CUDA {major}.x dropped sm_{arch}"));
            continue;
        }
        let cubin = out.join("feral.cubin");
        let ptx = out.join("feral.ptx");
        if cubin.exists() {
            let _ = std::fs::remove_file(&cubin);
        }
        if ptx.exists() {
            let _ = std::fs::remove_file(&ptx);
        }
        // fmad=false and precise division/sqrt keep device math close to the CPU oracle.
        let common = [
            "-O3",
            "--std=c++14",
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "-Xptxas",
            "-O3",
            "-I",
            out.to_str().unwrap_or("."),
        ];
        let cubin_build = Command::new(&nvcc)
            .args(["-cubin", &format!("-arch={sm}")])
            .args(common)
            .arg("-o")
            .arg(&cubin)
            .arg(&source)
            .output();
        let ptx_build = Command::new(&nvcc)
            .args(["-ptx", &format!("-arch={compute}")])
            .args(common)
            .arg("-o")
            .arg(&ptx)
            .arg(&source)
            .output();
        let ok = |r: &std::io::Result<std::process::Output>| {
            r.as_ref().map(|o| o.status.success()).unwrap_or(false)
        };
        if ok(&cubin_build) && ok(&ptx_build) && cubin.exists() && ptx.exists() {
            println!("cargo:rustc-env=FERAL_CUDA_NVCC={nvcc}");
            println!("cargo:rustc-env=FERAL_CUDA_ARCH={arch}");
            println!("cargo:warning=FERAL CUDA kernels compiled with {nvcc} for {sm}");
            return;
        }
        let detail = match (&cubin_build, &ptx_build) {
            (Ok(o), _) if !o.status.success() => String::from_utf8_lossy(&o.stderr).to_string(),
            (Err(e), _) => e.to_string(),
            (_, Ok(o)) if !o.status.success() => String::from_utf8_lossy(&o.stderr).to_string(),
            (_, Err(e)) => e.to_string(),
            _ => "unknown compile failure".into(),
        };
        errors.push(format!("{nvcc} (CUDA {major}.x): {detail}"));
    }
    panic!(
        "FERAL CUDA build requires an nvcc that supports {sm}. Tried:\n{}\n\
         Set FERAL_NVCC=/path/to/nvcc (CUDA 11.8 or 12.x for sm_61).",
        errors.join("\n")
    );
}
