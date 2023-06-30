mod tests;

pub mod p_var {
    #![allow(non_snake_case)]
    use std::fmt::{self, Display, Formatter};

    #[derive(Debug, PartialEq)]
    pub enum PVarError {
        EmptyArray,
        PRange,
    }

    impl Display for PVarError {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            match self {
                Self::EmptyArray => write!(f, "input array is empty"),
                Self::PRange => write!(f, "exponent must be greater or equal than 1.0"),
            }
        }
    }

    pub fn p_var_backbone<'a, T, F>(v: &'a [T], p: f64, dist: F) -> Result<f64, PVarError>
    where
        F: Fn(&'a T, &'a T) -> f64,
    {
        if p < 1. {
            return Err(PVarError::PRange);
        }
        if v.is_empty() {
            return Err(PVarError::EmptyArray);
        }

        if v.len() == 1 {
            return Ok(0.);
        }

        let mut run_pvar = vec![0f64; v.len()];
        let mut N = 1;
        let s = v.len() - 1;

        while (s >> N) > 0 {
            N += 1;
        }

        let mut radius = vec![0f64; s];
        let ind_n = |j, n| (s >> n) + (j >> n);
        let center = |j, n| ((j >> n) << n) + (1usize << (n - 1));
        let center_outside_range = |j, n| (j >> n == s >> n && (s >> (n - 1)) % 2usize == 0usize);

        let mut point_links = vec![0usize; v.len()];
        let mut max_p_var = 0f64;

        for (j, u) in v.iter().enumerate() {
            for n in 1..=N {
                if !center_outside_range(j, n) {
                    let r = &mut radius[ind_n(j, n)];
                    *r = f64::max(*r, dist(&v[center(j, n)], u));
                }
            }
            if j == 0 {
                continue;
            }

            let mut m = j - 1;
            point_links[j] = m;

            let mut delta = dist(&v[m], u);

            max_p_var = run_pvar[m] + delta.powf(p);

            let mut n = 0;

            while m > 0 {
                while n < N && (m >> n) % 2 == 0 {
                    n += 1;
                }
                m -= 1;

                let mut delta_needs_update = true;
                while n > 0 {
                    if !center_outside_range(m, n) {
                        let id = radius[ind_n(m, n)] + dist(&v[center(m, n)], u);
                        if delta >= id {
                            break;
                        } else if delta_needs_update {
                            delta = (max_p_var - run_pvar[m]).powf(1f64 / p);
                            delta_needs_update = false;
                            if delta >= id {
                                break;
                            }
                        }
                    }
                    n -= 1;
                }
                if n > 0 {
                    m = (m >> n) << n;
                } else {
                    let d = dist(&v[m], u);
                    if d >= delta {
                        let new_p_var = run_pvar[m] + d.powf(p);
                        if new_p_var >= max_p_var {
                            max_p_var = new_p_var;
                            point_links[j] = m;
                        }
                    }
                }
            }

            run_pvar[j] = max_p_var;
        }
        Ok(max_p_var)
    }

    pub fn p_var_backbone_ref<'a, T, F>(v: &'a [T], p: f64, dist: F) -> Result<f64, PVarError>
    where
        F: Fn(&'a T, &'a T) -> f64,
        T: Copy,
    {
        if v.len() == 0 {
            return Err(PVarError::EmptyArray);
        }
        if v.len() == 1 {
            return Ok(0.);
        }

        let mut cum_p_var = vec![0f64; v.len()];

        for j in 1..v.len() {
            for m in 0..j {
                cum_p_var[j] = f64::max(cum_p_var[j], cum_p_var[m] + dist(&v[m], &v[j]).powf(p));
            }
        }

        Ok(*cum_p_var.last().unwrap())
    }
}

use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "p_var")]
fn pvar_wrapper(path: Vec<Vec<f64>>, p: f64, dist: f64) -> f64 {
    let dist_fn = |a: &Vec<f64>, b: &Vec<f64>| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x-y).powf(dist))
                .sum::<f64>()
                .powf(1.0/dist)
    };

    p_var::p_var_backbone(&path, p, dist_fn).unwrap()
}

#[pymodule]
fn pvar(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pvar_wrapper, m)?)?;
    Ok(())
}
