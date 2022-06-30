use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pvar::p_var::{p_var_backbone, p_var_backbone_ref};
use rand::{
    distributions::{Distribution, Standard},
    prelude::*,
};
use rayon::prelude::*;

fn generate_bm(n: usize) -> Vec<f64> {
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

fn ndim_bm(n: usize, d: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.; d]];
    let sigma = 1. / f64::sqrt(n as f64);
    let mut rng = StdRng::from_entropy();
    out.extend_from_slice(
        &Standard
            .map(|b| if b { vec![sigma; d] } else { vec![-sigma; d] })
            .sample_iter(&mut rng)
            .take(n - 1)
            .scan(vec![0.; d], |acc, x| {
                for (k, a) in acc.iter_mut().enumerate() {
                    *a += x[k];
                }
                Some(acc.clone())
            })
            .collect::<Vec<Vec<f64>>>(),
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

fn l2_norm(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn l2_norm_parallel(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.par_iter()
        .zip(b)
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn high_dimension(c: &mut Criterion) {
    let mut g = c.benchmark_group("high-dim");
    let v = ndim_bm(2000, 64 * 64);
    g.bench_with_input(BenchmarkId::new("N-dim 2.5-var BM", 64 * 64), &v, |b, u| {
        b.iter(|| p_var_backbone(u, black_box(2.5), black_box(l2_norm)))
    });
    g.bench_with_input(
        BenchmarkId::new("N-dim parallel 2.5-var BM", 64 * 64),
        &v,
        |b, u| b.iter(|| p_var_backbone(u, black_box(2.5), black_box(l2_norm_parallel))),
    );
}

criterion_group!(benches, backbone_bench);
criterion_group!(high_dim, high_dimension);
criterion_main!(high_dim);
