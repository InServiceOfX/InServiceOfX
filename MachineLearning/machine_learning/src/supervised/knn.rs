use num_traits::Float;
use std::iter::Sum;
use std::ops::Sub;

pub fn euclidean_distance<I, O>(x1: I, x2: I) -> O
// We need to constrain the types.
where
    I: IntoIterator<Item = O>,
    //T: Sub<Output = T> + Sum<T> + Into<O>,
    O: Sum<O> + Sub<Output = O> + Copy + Float,
{
    x1.into_iter()
        .zip(x2.into_iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<O>()
        .sqrt()     
}

/*
pub fn euclidean_distance(x1: &[f64], x2: &[f64]) -> f64
{
    x1.iter()
        .zip(x2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()    
}
*/

#[cfg(test)]
mod test
{
    use super::*;

    #[test]
    fn test_euclidean_distance()
    {
        let x : &[f64] = &[5.25, 2.5, 3.0, 1.125];
        let y : &[f64] = &[6.125, 3.0, 4.75, 1.5];
        let result : f64 = *euclidean_distance(x, y);



    }
}