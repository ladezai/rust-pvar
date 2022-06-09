use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pvar::pvar::p_var_backbone;

fn generate_bm(n: usize) -> Vec<f64> {
    use rand::{
        distributions::{Distribution, Standard},
        prelude::*,
    };

    let mut out = vec![0.; n + 1];
    let sigma = 1. / f64::sqrt(n as f64);
    let mut rng = StdRng::from_entropy();
    out[1..].copy_from_slice(
        &Distribution::<bool>::map(Standard, |b| if b { sigma } else { -sigma })
            .sample_iter(&mut rng)
            .take(n)
            .scan(0., |a, x| {
                *a += x;
                Some(*a)
            })
            .collect::<Vec<_>>(),
    );
    out
}

fn criterion_benchmark(c: &mut Criterion) {
    for n in 1..=6 {
        let v = generate_bm(10_usize.pow(n));
        c.bench_function(&format!("backbone 2.5-var BM {}", 10_usize.pow(n)), |b| {
            b.iter(|| p_var_backbone(&v, black_box(2.5), |a, b| f64::abs(b - a)))
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
