use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pvar::pvar::{p_var_backbone, p_var_backbone_ref};

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

fn backbone_bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("p-var");
    for n in 1..=4 {
        let v = generate_bm(10_usize.pow(n));
        g.bench_with_input(
            BenchmarkId::new("Backbone 2.5-var BM", v.len() - 1),
            &v,
            |b, u| b.iter(|| p_var_backbone(u, black_box(2.5), black_box(|a, b| f64::abs(b - a)))),
        );
        g.bench_with_input(
            BenchmarkId::new("Backbone ref 2.5-var BM", v.len() - 1),
            &v,
            |b, u| {
                b.iter(|| p_var_backbone_ref(u, black_box(2.5), black_box(|a, b| f64::abs(b - a))))
            },
        );
    }
    g.finish();
}

criterion_group!(benches, backbone_bench);
criterion_main!(benches);
