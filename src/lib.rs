#[cfg(test)]

mod tests {
    use super::pvar::*;
    use rand::distributions::Standard;
    use rand::prelude::*;

    fn dist<const N: usize>(a: [f64; N], b: [f64; N]) -> f64 {
        f64::sqrt(
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).powf(2.))
                .sum(),
        )
    }

    fn dist_1d(a: f64, b: f64) -> f64 {
        f64::abs(a - b)
    }

    #[test]
    fn square_path() {
        let v: [[f64; 2]; 8] = [
            [0., 0.],
            [0., 1.],
            [1., 1.],
            [1., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
        ];
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
        path[1..].copy_from_slice(
            &StdRng::from_entropy()
                .sample::<[bool; N], Standard>(Standard)
                .map(|x| if x { sigma } else { -sigma })
                .iter()
                .scan(0., |acc, x| {
                    *acc += x;
                    Some(*acc)
                })
                .collect::<Vec<_>>(),
        );
        for p in [1., f64::sqrt(2.), 2., f64::exp(1.)] {
            assert_eq!(
                p_var_backbone(&path, p, dist_1d),
                p_var_backbone_ref(&path, p, dist_1d)
            );
        }
    }

    #[test]
    fn errors() {
        assert_eq!(
            format!("{}", p_var_backbone(&[], 1., dist_1d).err().unwrap()),
            "input array is empty"
        );
        assert_eq!(
            format!("{}", p_var_backbone(&[0., 1.], 0., dist_1d).err().unwrap()),
            "exponent must be greater or equal than 1.0"
        );
    }
}

pub mod pvar {
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

    pub fn p_var_backbone<T, F>(v: &[T], p: f64, dist: F) -> Result<f64, PVarError>
    where
        F: Fn(T, T) -> f64,
        T: Copy,
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
    where
        F: Fn(T, T) -> f64,
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
                cum_p_var[j] = f64::max(cum_p_var[j], cum_p_var[m] + dist(v[m], v[j]).powf(p));
            }
        }

        Ok(*cum_p_var.last().unwrap())
    }
}

pub mod ndarray {
    use std::error::Error;
    use std::fmt::{self, Display, Formatter};
    use std::fs::File;
    use std::io::{self, BufRead, BufReader, Read, Write};
    use std::ops::{Index, IndexMut};
    use std::path::Path;

    #[derive(Debug)]
    pub enum ArrayError {
        ReshapeError,
        NotEnoughData,
    }

    impl Display for ArrayError {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
            match self {
                Self::ReshapeError => write!(f, "shape missmatch"),
                Self::NotEnoughData => write!(f, "not enough bytes to read"),
            }
        }
    }

    impl Error for ArrayError {}

    #[derive(Clone, Debug)]
    pub struct Array<const N: usize> {
        shape: [usize; N],
        data: Vec<f64>,
    }

    impl<const N: usize> Index<[usize; N]> for Array<N> {
        type Output = f64;
        fn index(&self, idx: [usize; N]) -> &Self::Output {
            let offset: usize = idx
                .into_iter()
                .enumerate()
                .map(|(k, n)| n * self.shape[k + 1..].iter().product::<usize>())
                .sum();
            &self.data[offset]
        }
    }

    impl<const N: usize> IndexMut<[usize; N]> for Array<N> {
        fn index_mut(&mut self, idx: [usize; N]) -> &mut Self::Output {
            let offset: usize = idx
                .into_iter()
                .enumerate()
                .map(|(k, n)| n * self.shape[k + 1..].iter().product::<usize>())
                .sum();
            &mut self.data[offset]
        }
    }

    impl Display for Array<2> {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            let reshaped: Vec<Vec<f64>> = self
                .data
                .chunks(self.shape[1])
                .map(|v| v.to_vec())
                .collect::<Vec<Vec<f64>>>();
            writeln!(f, "{:#.6?}", reshaped)
        }
    }

    impl<const N: usize> Array<N> {
        pub fn new(shape: [usize; N]) -> Self {
            Array {
                shape: shape,
                data: Vec::with_capacity(shape.iter().product::<usize>()),
            }
        }

        pub fn zeros(shape: [usize; N]) -> Self {
            Array {
                shape: shape,
                data: vec![0f64; shape.iter().product::<usize>()],
            }
        }

        pub fn reshape<const M: usize>(&self, shape: [usize; M]) -> Result<Array<M>, ArrayError> {
            if self.data.len() != shape.iter().product::<usize>() {
                return Err(ArrayError::ReshapeError);
            }

            Ok(Array {
                shape: shape,
                data: self.data.clone(),
            })
        }

        pub fn from_bytes(v: Vec<u8>, shape: [usize; N]) -> Result<Self, ArrayError> {
            if v.len() / 8_usize != shape.iter().product::<usize>() {
                return Err(ArrayError::NotEnoughData);
            }
            Ok(Array {
                shape: shape,
                data: v
                    .chunks(8)
                    .map(|a| f64::from_le_bytes(a.try_into().unwrap()))
                    .collect(),
            })
        }
    }
}
