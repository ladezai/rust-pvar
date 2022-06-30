#[cfg(test)]
mod tests {
    use crate::p_var::*;
    use rand::distributions::Standard;
    use rand::prelude::*;

    fn dist<'z, const N: usize>(a: &'z [f64; N], b: &'z [f64; N]) -> f64 {
        f64::sqrt(
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).powf(2.))
                .sum(),
        )
    }

    fn dist_1d<'z>(a: &'z f64, b: &'z f64) -> f64 {
        f64::abs(a - b)
    }

    #[test]
    fn square_path() {
        let v = [
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
            assert_eq!(
                p_var_backbone(&v, p, dist).ok(),
                p_var_backbone_ref(&v, p, dist).ok()
            );
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
