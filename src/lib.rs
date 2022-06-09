#[cfg(test)]
mod tests {
    use super::pvar::p_var_backbone;
    #[test]
    fn backbone() {
        let v = (0..=10).map(f64::from).collect::<Vec<f64>>();
        assert_eq!(p_var_backbone(&v, 1., |a, b| f64::abs(a - b)), Ok(10.));
    }
}

pub mod pvar {
    #[derive(Debug, PartialEq)]
    pub enum PVarError {
        EmptyArray,
        PRange
    }

    pub fn p_var_backbone<T, F>(v: &[T], p: f64, dist: F) -> Result<f64, PVarError>
        where F: Fn(&T, &T) -> f64 {
            if p < 1. {
                return Err(PVarError::PRange);
            }
            if v.is_empty() {
                return Err(PVarError::EmptyArray);
            }

            if v.len() == 1 {
                return Ok(0.);
            }

            let mut run_pvar: Vec<f64> = vec![0f64; v.len()];
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

            for j in 0..v.len() {
                for n in 1..=N {
                    if !center_outside_range(j, n) {
                        let r = &mut radius[ind_n(j, n)];
                        *r = f64::max(*r, dist(&v[center(j, n)], &v[j]));
                    }
                }
                if j == 0 {
                    continue;
                }

                let mut m = j - 1;
                point_links[j] = m;

                let mut delta = dist(&v[m], &v[j]);

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
                            let id = radius[ind_n(m, n)] + dist(&v[center(m, n)], &v[j]);
                            if delta >= id { break; }
                            else if delta_needs_update {
                                delta = (max_p_var - run_pvar[m]).powf(1f64 / p);
                                delta_needs_update = false;
                                if delta >= id { break; }
                            }
                        }
                        n -= 1;
                    }
                    if n > 0 {
                        m = (m >> n) << n;
                    }
                    else {
                        let d = dist(&v[m], &v[j]);
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
}
