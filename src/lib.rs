#[cfg(test)]

mod tests {
    use super::pvar::*;
    use rand::prelude::*;
    use rand::distributions::Standard;

    fn dist<const N: usize>(a: [f64; N], b: [f64; N]) -> f64 {
        f64::sqrt(a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powf(2.)).sum())
    }

    fn dist_1d(a: f64, b: f64) -> f64 {
        f64::abs(a - b)
    }

    #[test]
    fn square_path() {
        let v: [[f64; 2]; 8] = [[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]];
        let mut p = 1.;
        while p <= 4. {
            assert_eq!(p_var_backbone(&v, p, dist), p_var_backbone_ref(&v, p, dist));
            p += 0.5;
        }
    }

    #[test]
    fn bm() {
        const N: usize = 2500;
        let mut path = [0.; N + 1];
        let sigma = 1. / f64::sqrt(N as f64);
        path[1..].copy_from_slice(&StdRng::from_entropy().sample::<[bool; N], Standard>(Standard).map(|x| if x { sigma } else { -sigma }).iter().scan(0., |acc, x| { *acc += x; Some(*acc) }).collect::<Vec<_>>());
        for p in [1., f64::sqrt(2.), 2., f64::exp(1.)] {
            assert_eq!(p_var_backbone(&path, p, dist_1d), p_var_backbone_ref(&path, p, dist_1d));
        }
    }

    #[test]
    fn errors() {
        assert_eq!(format!("{}", p_var_backbone(&[], 1., dist_1d).err().unwrap()), "input array is empty");
        assert_eq!(format!("{}", p_var_backbone(&[0., 1.], 0., dist_1d).err().unwrap()), "exponent must be greater or equal than 1.0");
    }
}

pub mod pvar {
    #![allow(non_snake_case)]
    use std::fmt::{Display, Formatter, self};


    #[derive(Debug, PartialEq)]
    pub enum PVarError {
        EmptyArray,
        PRange
    }

    impl Display for PVarError {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            match self {
                Self::EmptyArray => write!(f, "input array is empty"),
                Self::PRange => write!(f, "exponent must be greater or equal than 1.0")
            }
        }
    }

    pub fn p_var_backbone<T, F>(v: &[T], p: f64, dist: F) -> Result<f64, PVarError>
        where F: Fn(T, T) -> f64, T: Copy {
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
                        *r = f64::max(*r, dist(v[center(j, n)], v[j]));
                    }
                }
                if j == 0 {
                    continue;
                }

                let mut m = j - 1;
                point_links[j] = m;

                let mut delta = dist(v[m], v[j]);

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
                            let id = radius[ind_n(m, n)] + dist(v[center(m, n)], v[j]);
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
                        let d = dist(v[m], v[j]);
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

    pub fn p_var_backbone_ref<T, F>(v: &[T], p: f64, dist: F) -> Result<f64, PVarError>
        where F: Fn(T, T) -> f64, T: Copy {
            if v.len() == 0 {
                return Err(PVarError::EmptyArray);
            }
            if v.len() == 1 {
                return Ok(0.);
            }
            
            let mut cum_p_var = vec![0f64; v.len()];

            for j in 1..v.len() {
                for m in 0..j {
                    cum_p_var[j] = f64::max(cum_p_var[j], cum_p_var[m] + dist(v[m], v[j]).powf(p));
                }
            }

            Ok(*cum_p_var.last().unwrap())
    }
}
