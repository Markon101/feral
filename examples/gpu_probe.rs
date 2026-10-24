#[cfg(not(feature = "opencl"))]
fn main() {
    eprintln!("Build with --features opencl");
    std::process::exit(2);
}
#[cfg(feature = "opencl")]
fn main() -> anyhow::Result<()> {
    use feral::{
        gpu::OpenClTrainer, model::Velocity, parameterization, rng::Rng, runtime::Checkpoint,
    };
    let ck = Checkpoint::load(std::path::Path::new(
        "runs/phase3-calibrated-v1/calibrated-42/checkpoint.json",
    ))?;
    let n = ck.crop_samples;
    let mut rng = Rng::new(81);
    let y: Vec<f32> = (0..n).map(|i| 0.15 * (i as f32 * 0.12).sin()).collect();
    let z: Vec<f32> = (0..n).map(|_| rng.normal() * ck.flow.prior_std()).collect();
    let mut mask = vec![0.; n];
    mask[..n / 3].fill(1.);
    let t = 0.37;
    let mut gpu = OpenClTrainer::new(&ck.model, &ck.adam, n, ck.flow.clone())?;
    println!("{}", serde_json::to_string_pretty(&gpu.info())?);
    let (cpu_loss, cpu_g) = parameterization::loss_gradient(&ck.model, &ck.flow, &y, &mask, &z, t);
    let gs = gpu.gradients(&y, &mask, &z, t)?;
    let g = gpu.read_gradients()?;
    let f = gpu.read_prediction()?;
    let x: Vec<f32> = y
        .iter()
        .zip(&z)
        .zip(&mask)
        .map(|((&y, &z), &m)| if m == 1. { y } else { (1. - t) * z + t * y })
        .collect();
    let cpu_v = ck.velocity().velocity(&x, &y, &mask, t);
    let (skip, out, _) = ck.flow.coefficients(t);
    let forward = f
        .iter()
        .zip(&x)
        .zip(cpu_v)
        .map(|((&f, &x), v)| (skip * x + out * f - v).abs())
        .fold(0f32, f32::max);
    let grad = g
        .iter()
        .zip(&cpu_g)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "loss cpu {cpu_loss} gpu {} | forward max {forward} | grad max {grad} | GPU wall {} ms",
        gs.loss, gs.wall_ms
    );
    let mut model = ck.model.clone();
    let mut adam = ck.adam.clone();
    let norm = adam.update(&mut model, &cpu_g)?;
    gpu.train_step(&y, &mask, &z, t)?;
    let mut gm = ck.model.clone();
    let mut ga = ck.adam.clone();
    gpu.sync(&mut gm, &mut ga)?;
    println!(
        "norm CPU {norm} GPU {} | parameter delta {} | moments delta {}",
        gs.norm,
        gm.parameters()
            .iter()
            .zip(model.parameters())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max),
        ga.m.iter()
            .zip(adam.m)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    );
    println!("{}", serde_json::to_string_pretty(&gpu.info())?);
    Ok(())
}
