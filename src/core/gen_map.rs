use rand::rngs::StdRng;
use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;

pub fn get_rand_matrix(shared_rng: Rc<RefCell<StdRng>>) -> Vec<Vec<bool>> {
    let vec_of_bools = vec![
        vec![true, true, false],
        vec![true, false, true],
        vec![false, true, true],
        vec![true, false, false],
        vec![false, true, false],
        vec![false, false, true],
    ];

    let mut rng_ref = shared_rng.borrow_mut();
    let mut index: usize = rng_ref.gen_range(0..6);
    let mut random_vecs: Vec<Vec<bool>> = Vec::new();
    // Randomly select two distinct elements from the vector
    random_vecs.push(vec_of_bools[index].clone());

    index = rng_ref.gen_range(0..6);
    random_vecs.push(vec_of_bools[index].clone());

    index = rng_ref.gen_range(0..6);
    random_vecs.push(vec_of_bools[index].clone());

    // Assign the two random elements to variables
    let random_vec1 = random_vecs[0].clone();
    let random_vec2 = random_vecs[1].clone();
    let mut random_vec3 = random_vecs[2].clone();

    // Column-wise check
    for i in 0..random_vec1.len() {
        if random_vec1[i] == random_vec2[i] {
            random_vec3[i] = !random_vec1[i];
        }
    }
    // final row check
    if random_vec3[0] == random_vec3[1] && random_vec3[0] == random_vec3[2] {
        for i in 0..random_vec1.len() {
            if random_vec1[i] != random_vec2[i] {
                random_vec3[i] = !random_vec3[i];
                break;
            }
        }
    }

    let matrix = vec![random_vec1, random_vec2, random_vec3];

    // Randomly transform the matrix
    drop(rng_ref);
    random_transform(matrix, shared_rng.clone())
}

// Rotate the matrix clockwise by 90 degrees
fn rotate_matrix(matrix: &Vec<Vec<bool>>) -> Vec<Vec<bool>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut rotated = vec![vec![false; rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            rotated[j][rows - 1 - i] = matrix[i][j];
        }
    }
    rotated
}

// Randomly transform the matrix by rotating it or mirroring it horizontally/vertically
fn random_transform(matrix1: Vec<Vec<bool>>, shared_rng: Rc<RefCell<StdRng>>) -> Vec<Vec<bool>> {
    let mut rng = shared_rng.borrow_mut();
    let operation1 = rng.gen_range(0..3);
    let matrix2 = match operation1 {
        0 => matrix1,
        1 => matrix1.into_iter().rev().collect(), // Vertical mirror
        2 => matrix1
            .into_iter()
            .map(|row| row.into_iter().rev().collect())
            .collect(), // Horizontal mirror
        _ => unreachable!(),
    };

    let operation2 = rng.gen_range(0..4);

    match operation2 {
        0 => matrix2,                                                 // No rotation
        1 => rotate_matrix(&matrix2),                                 // Rotate 90 degrees
        2 => rotate_matrix(&rotate_matrix(&matrix2)),                 // Rotate 180 degrees
        3 => rotate_matrix(&rotate_matrix(&rotate_matrix(&matrix2))), // Rotate 270 degrees
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_env() {
        let seed: [u8; 32] = [0; 32];
        let mut rng = StdRng::from_seed(seed);
        let rm = get_rand_matrix(&mut rng);
        assert_eq!(
            rm[2]
                .get(0)
                .map_or(true, |first| rm[2].iter().all(|item| item == first)),
            false
        )
    }
}
