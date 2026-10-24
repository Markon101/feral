use super::OpenClTrainer;
use crate::{
    audio, conditioning,
    data::Dataset,
    flow::{self, Sampling},
    model::{Architecture, Model, Velocity},
    parameterization::{self, FlowConfig},
    rng::Rng,
    runtime::{self, Adam, Checkpoint},
};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{fs, path::Path, time::Instant};
fn max_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max)
}
fn close(a: &[f32], b: &[f32], abs: f32, rel: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= abs + rel * b.abs())
}
fn parity_case(
    model: &Model,
    adam: &Adam,
    flow: &FlowConfig,
    n: usize,
    t: f32,
    amplitude: f32,
) -> Result<Value> {
    let mut rng = Rng::new(77);
    let y: Vec<f32> = (0..n)
        .map(|i| amplitude * (i as f32 * 0.13).sin())
        .collect();
    let noise: Vec<f32> = (0..n).map(|_| rng.normal() * flow.prior_std()).collect();
    let mut mask = vec![0.; n];
    mask[..n / 3].fill(1.);
    let (loss, g) = parameterization::loss_gradient(model, flow, &y, &mask, &noise, t);
    let mut gpu = OpenClTrainer::new(model, adam, n, flow.clone())?;
    let result = gpu.gradients(&y, &mask, &noise, t)?;
    let gg = gpu.read_gradients()?;
    let prediction = gpu.read_prediction()?;
    let x: Vec<f32> = y
        .iter()
        .zip(&noise)
        .zip(&mask)
        .map(|((&y, &z), &m)| if m == 1. { y } else { (1. - t) * z + t * y })
        .collect();
    let v = parameterization::ConfiguredVelocity {
        model,
        config: flow,
    }
    .velocity(&x, &y, &mask, t);
    let (skip, out, _) = flow.coefficients(t);
    let gv: Vec<f32> = prediction
        .iter()
        .zip(&x)
        .map(|(&f, &x)| skip * x + out * f)
        .collect();
    ensure!(
        (loss - result.loss).abs() <= 1e-5 + 1e-4 * loss.abs(),
        "GPU loss parity failed: {loss} vs {}",
        result.loss
    );
    ensure!(
        close(&gg, &g, 1e-5, 1e-3),
        "GPU gradient parity failed: max {}",
        max_delta(&gg, &g)
    );
    ensure!(
        close(&gv, &v, 1e-5, 1e-3),
        "GPU forward parity failed: max {}",
        max_delta(&gv, &v)
    );
    let mut cpu = model.clone();
    let mut ca = adam.clone();
    let norm = ca.update(&mut cpu, &g)?;
    gpu.train_step(&y, &mask, &noise, t)?;
    let mut gm = model.clone();
    let mut ga = adam.clone();
    gpu.sync(&mut gm, &mut ga)?;
    ensure!(
        close(&gm.parameters(), &cpu.parameters(), 5e-6, 1e-4),
        "GPU Adam parameter parity failed"
    );
    ensure!(
        close(&ga.m, &ca.m, 1e-6, 1e-3) && close(&ga.v, &ca.v, 1e-6, 1e-3),
        "GPU Adam moment parity failed"
    );
    Ok(
        json!({"architecture":model.architecture,"flow":flow,"samples":n,"t":t,"loss_cpu":loss,"loss_gpu":result.loss,"forward_max_abs":max_delta(&gv,&v),"gradient_max_abs":max_delta(&gg,&g),"parameter_max_abs":max_delta(&gm.parameters(),&cpu.parameters()),"moment_m_max_abs":max_delta(&ga.m,&ca.m),"moment_v_max_abs":max_delta(&ga.v,&ca.v),"gradient_norm_cpu":norm,"gradient_norm_gpu":result.norm,"clip_active":norm>1.}),
    )
}
pub fn temperatures() -> Value {
    let mut readings = serde_json::Map::new();
    if let Ok(entries) = fs::read_dir("/sys/class/thermal") {
        for e in entries.flatten() {
            let p = e.path();
            if let (Ok(kind), Ok(value)) = (
                fs::read_to_string(p.join("type")),
                fs::read_to_string(p.join("temp")),
            ) {
                let name = kind.trim();
                if name.contains("gpu") || name == "battery" {
                    if let Ok(v) = value.trim().parse::<f64>() {
                        if (0.0..200000.).contains(&v) {
                            readings.insert(name.into(), json!(v / 1000.));
                        }
                    }
                }
            }
        }
    }
    Value::Object(readings)
}
pub fn audit(checkpoint: &Path, output: &Path, steps: usize) -> Result<()> {
    ensure!((2..=200).contains(&steps), "audit steps must be 2..200");
    ensure!(!output.exists(), "audit output exists");
    if let Some(p) = output.parent() {
        fs::create_dir_all(p)?;
    }
    let ck = Checkpoint::load(checkpoint)?;
    let start_temp = temperatures();
    let mut cases = vec![];
    let tiny = Model::new(
        Architecture {
            block: 4,
            width: 5,
            dilations: vec![1, 2],
            scales: vec![1, 2],
        },
        7,
    );
    for flow in [
        FlowConfig::Velocity,
        FlowConfig::Preconditioned {
            data_scale: 0.1,
            prior_std: 1.,
        },
        FlowConfig::Preconditioned {
            data_scale: 0.1,
            prior_std: 0.1,
        },
    ] {
        for t in [0., 0.37, 1.] {
            cases.push(parity_case(
                &tiny,
                &Adam::new(tiny.count(), 0.0003),
                &flow,
                68,
                t,
                0.7,
            )?);
        }
    }
    let baseline = Model::new(Architecture::default(), 7);
    cases.push(parity_case(
        &baseline,
        &Adam::new(baseline.count(), 0.0003),
        &FlowConfig::Velocity,
        2048,
        0.37,
        0.2,
    )?);
    cases.push(parity_case(
        &ck.model,
        &ck.adam,
        &ck.flow,
        ck.crop_samples,
        0.37,
        0.15,
    )?);
    let input = &ck
        .dataset
        .sources
        .first()
        .ok_or_else(|| anyhow::anyhow!("checkpoint has no source"))?
        .path;
    let dataset = Dataset::load(input, ck.dataset.preprocess.clone(), ck.crop_samples)?;
    let mut gpu = OpenClTrainer::new(&ck.model, &ck.adam, ck.crop_samples, ck.flow.clone())?;
    let mut model = ck.model.clone();
    let mut adam = ck.adam.clone();
    let mut rng = Rng::new(771);
    let mut cpu_ms = vec![];
    let mut gpu_ms = vec![];
    let mut loss_max = 0f32;
    for i in 0..steps + 2 {
        let clean = dataset.crop(ck.crop_samples, &mut rng);
        ensure!(!clean.is_empty(), "silent audit crop");
        let (mask, _) = conditioning::training_mask(clean.len(), &mut rng);
        let noise: Vec<f32> = (0..clean.len())
            .map(|_| rng.normal() * ck.flow.prior_std())
            .collect();
        let t = rng.uniform();
        let cpu_step = |model: &mut Model, adam: &mut Adam| -> Result<(f32, f64)> {
            let now = Instant::now();
            let (loss, g) =
                parameterization::loss_gradient(model, &ck.flow, &clean, &mask, &noise, t);
            adam.update(model, &g)?;
            Ok((loss, now.elapsed().as_secs_f64() * 1000.))
        };
        let (closs, ctime, gstep, gtime) = if i % 2 == 0 {
            let (c, ct) = cpu_step(&mut model, &mut adam)?;
            let now = Instant::now();
            let g = gpu.train_step(&clean, &mask, &noise, t)?;
            (c, ct, g, now.elapsed().as_secs_f64() * 1000.)
        } else {
            let now = Instant::now();
            let g = gpu.train_step(&clean, &mask, &noise, t)?;
            let gt = now.elapsed().as_secs_f64() * 1000.;
            let (c, ct) = cpu_step(&mut model, &mut adam)?;
            (c, ct, g, gt)
        };
        loss_max = loss_max.max((closs - gstep.loss).abs());
        if i >= 2 {
            cpu_ms.push(ctime);
            gpu_ms.push(gtime);
        }
    }
    let mut gm = ck.model.clone();
    let mut ga = ck.adam.clone();
    gpu.sync(&mut gm, &mut ga)?;
    ensure!(
        close(&gm.parameters(), &model.parameters(), 2e-4, 2e-3),
        "multi-step CPU/GPU parameter drift too large: {}",
        max_delta(&gm.parameters(), &model.parameters())
    );
    let mut snapshot = ck.clone();
    snapshot.model = gm.clone();
    snapshot.adam = ga.clone();
    snapshot.completed_steps = ga.step;
    snapshot.rng = rng.clone();
    let state_path = output.with_extension("ckpt");
    snapshot.save(&state_path)?;
    let loaded = Checkpoint::load(&state_path)?;
    let mut resumed = OpenClTrainer::new(
        &loaded.model,
        &loaded.adam,
        ck.crop_samples,
        ck.flow.clone(),
    )?;
    let clean = dataset.crop(ck.crop_samples, &mut rng);
    let (mask, _) = conditioning::training_mask(clean.len(), &mut rng);
    let noise: Vec<f32> = (0..clean.len())
        .map(|_| rng.normal() * ck.flow.prior_std())
        .collect();
    gpu.train_step(&clean, &mask, &noise, 0.47)?;
    resumed.train_step(&clean, &mask, &noise, 0.47)?;
    let mut rm = gm.clone();
    let mut ra = ga.clone();
    gpu.sync(&mut gm, &mut ga)?;
    resumed.sync(&mut rm, &mut ra)?;
    ensure!(
        gm.parameters() == rm.parameters() && ga.m == ra.m && ga.v == ra.v,
        "GPU checkpoint resume is not deterministic"
    );
    let mut portable = loaded.clone();
    let (_, g) = parameterization::loss_gradient(
        &portable.model,
        &portable.flow,
        &clean,
        &mask,
        &noise,
        0.47,
    );
    portable.adam.update(&mut portable.model, &g)?;
    ensure!(
        close(&portable.model.parameters(), &gm.parameters(), 5e-6, 1e-4),
        "GPU checkpoint cannot continue with CPU within tolerance"
    );
    let zeros = vec![0.; ck.crop_samples];
    let sample = flow::sample(
        &loaded.velocity(),
        &zeros,
        &zeros,
        &Sampling {
            steps: 4,
            ..Default::default()
        },
    )?;
    audio::write_wav(&output.with_extension("wav"), &sample, ck.sample_rate)?;
    let cpu_mean = cpu_ms.iter().sum::<f64>() / steps as f64;
    let gpu_mean = gpu_ms.iter().sum::<f64>() / steps as f64;
    runtime::write_json(
        output,
        &json!({"checkpoint":checkpoint,"cases":cases,"all_parity_gates_passed":true,"gpu_resume_exact":true,"cpu_portable_resume":true,"steps_timed":steps,"warmup_updates":2,"cpu_step_ms":cpu_ms,"gpu_step_ms":gpu_ms,"cpu_mean_ms":cpu_mean,"gpu_mean_ms":gpu_mean,"observed_speedup":cpu_mean/gpu_mean,"multi_step_parameter_max_abs":max_delta(&snapshot.model.parameters(),&model.parameters()),"multi_step_loss_max_abs":loss_max,"device":gpu.info(),"temperatures_start_c":start_temp,"temperatures_end_c":temperatures(),"affinity":fs::read_to_string("/proc/self/status").unwrap_or_default().lines().find(|s|s.starts_with("Cpus_allowed_list:")).unwrap_or("unavailable"),"note":"foreground confirmed by owner; phone already warm; short timing, not thermally controlled peak or sustained guarantee"}),
    )?;
    eprintln!(
        "GPU audit passed | CPU {cpu_mean:.3} ms | GPU {gpu_mean:.3} ms | observed {:.2}x | {}",
        cpu_mean / gpu_mean,
        output.display()
    );
    Ok(())
}
