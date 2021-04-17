#![feature(destructuring_assignment)]
#![feature(map_first_last)]
extern crate nalgebra as na;
use std::env;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::collections::BTreeSet;
use ndarray::prelude::*;
use na::{DMatrix, DVector};


fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn read_to_grid(path: &String) -> Array3<i32> {
    let mut n: usize = 0;
    if let Ok(lines) = read_lines(path) {
        // Consumes the iterator, count line numbers
        n = lines.count();
    }
    println!("{} strategies", n);
    let mut grid: Array3<i32> = Array::zeros((n, n, 2));
    if let Ok(lines) = read_lines(path) {
        // Consumes the iterator, returns an (Optional) String
        let mut i = 0;
        for line in lines {
            if let Ok(mut line_content) = line {
                line_content.retain(|c| !c.is_whitespace());
                let mut line_iter = line_content.split(",");
                for j in 0..n {
                    grid[[i,j,0]] = line_iter.next().unwrap().parse().unwrap();
                    grid[[i,j,1]] = line_iter.next().unwrap().parse().unwrap();
                }
            }
            i += 1;
        }
    }
    return grid;
}

fn column_argmax(col: ArrayView1<i32>) -> Vec<usize> {
    // println!("{:?}", col);
    let col_iter = col.indexed_iter();
    let mut best_po = i32::MIN;
    let mut br = Vec::new();
    for (index, po) in col_iter {
        if index == 0 {
            best_po = po.to_owned();
            br.push(index);
        } else {
            if po > &best_po {
                best_po = po.to_owned();
                br = Vec::new();
                br.push(index);
            } else if po == &best_po {
                br.push(index);
            }
        }
    }
    return br;
}

fn row_argmax(row: ArrayView1<i32>) -> Vec<usize> {
    // println!("{:?}", row);
    let col_iter = row.indexed_iter();
    let mut best_po = i32::MIN;
    let mut br = Vec::new();
    for (index, po) in col_iter {
        if index == 0 {
            best_po = po.to_owned();
            br.push(index);
        } else {
            if po > &best_po {
                best_po = po.to_owned();
                br = Vec::new();
                br.push(index);
            } else if po == &best_po {
                br.push(index);
            }
        }
    }
    return br;
}

fn solve_pure(grid: &Array3<i32>) -> Vec<(usize, usize)> {
    // let n = grid.shape()[0];

    // calculate p0's br
    // println!("{:?}", grid.index_axis(Axis(2), 0));
    let br_p0_array: Array1<Vec<usize>> = grid.index_axis(Axis(2), 0).map_axis(Axis(0), column_argmax);
    // println!("{:?}", br_p0_array);
    let br_p1_array: Array1<Vec<usize>> = grid.index_axis(Axis(2), 1).map_axis(Axis(1), row_argmax);
    // println!("{:?}", br_p1_array);

    let mut nash_vec = Vec::new();

    for (p1_s, p0_response_set) in br_p0_array.indexed_iter() {
        for p0_s in p0_response_set {
            if br_p1_array[p0_s.to_owned()].iter().any(|&v| v == p1_s) {
                nash_vec.push((p0_s.to_owned(), p1_s));
            }
        }
    }
    return nash_vec;
}

fn rm_p0(grid: &Array3<i32>, rows: Vec<usize>, cols: &Vec<usize>) -> (bool, Vec<usize>) {
    let row_num = rows.len();
    // let col_num = cols.len();
    let mut max_values = Vec::new();
    let po_p0 = grid.index_axis(Axis(2), 0);
    for col in cols {
        let this_col = po_p0.column(*col);
        let col_iter = this_col.indexed_iter();
        let mut best_po = i32::MIN;
        let mut br = BTreeSet::new();
        for (index, po) in col_iter {
            if rows.iter().any(|&i| i == index) {
                if index == rows[0] {
                    best_po = po.to_owned();
                    br.insert(index);
                } else {
                    if po > &best_po {
                        best_po = po.to_owned();
                        br = BTreeSet::new();
                        br.insert(index);
                    } else if po == &best_po {
                        br.insert(index);
                    }
                }
            }
        }
        max_values.push(br);
    }

    let mut rows_to_keep = Vec::new();
    while !max_values.is_empty() {
        let mut max_intersection = max_values[0].clone();
        for c in 1..max_values.len() {
            let test_intersection: BTreeSet<usize> = max_values[c].intersection(&max_intersection).cloned().collect();
            if !test_intersection.is_empty() {
                max_intersection = test_intersection;
            }
        }
        let max_index = max_intersection.pop_first().unwrap();
        rows_to_keep.push(max_index);
        max_values.retain(|row|{!row.contains(&max_index)});
    }

    rows_to_keep.sort();
    return (row_num == rows_to_keep.len(), rows_to_keep);
}

fn rm_p1(grid: &Array3<i32>, rows: &Vec<usize>, cols: Vec<usize>) -> (bool, Vec<usize>) {
    // let row_num = rows.len();
    let col_num = cols.len();
    let mut max_values = Vec::new();
    let po_p1 = grid.index_axis(Axis(2), 1);
    for row in rows {
        let this_row = po_p1.row(*row);
        let row_iter = this_row.indexed_iter();
        let mut best_po = i32::MIN;
        let mut br = BTreeSet::new();
        for (index, po) in row_iter {
            if cols.iter().any(|&i| i == index) {
                if index == cols[0] {
                    best_po = po.to_owned();
                    br.insert(index);
                } else {
                    if po > &best_po {
                        best_po = po.to_owned();
                        br = BTreeSet::new();
                        br.insert(index);
                    } else if po == &best_po {
                        br.insert(index);
                    }
                }
            }
        }
        max_values.push(br);
    }

    let mut cols_to_keep = Vec::new();
    while !max_values.is_empty() {
        let mut max_intersection = max_values[0].clone();
        for c in 1..max_values.len() {
            let test_intersection: BTreeSet<usize> = max_values[c].intersection(&max_intersection).cloned().collect();
            if !test_intersection.is_empty() {
                max_intersection = test_intersection;
            }
        }
        let max_index = max_intersection.pop_first().unwrap();
        cols_to_keep.push(max_index);
        max_values.retain(|col|{!col.contains(&max_index)});
    }

    cols_to_keep.sort();
    return (col_num == cols_to_keep.len(), cols_to_keep);
}

fn iesds(grid: &Array3<i32>) -> (Vec<usize>, Vec<usize>) {
    let n = grid.shape()[0];
    let mut rows: Vec<usize> = (0..n).collect();
    let mut cols: Vec<usize> = (0..n).collect();
    let mut end_p0;
    let mut end_p1;
    loop {
        (end_p0, rows) = rm_p0(&grid, rows, &cols);
        (end_p1, cols) = rm_p1(&grid, &rows, cols);
        if end_p0 && end_p1 {
            return (rows, cols);
        }
    }
}

fn solve_mixed(grid: &Array3<i32>) {
    // let n = grid.shape()[0];
    let (rows, cols) = iesds(grid);
    // println!("{:?}", rows);
    // println!("{:?}", cols);
    if rows.len() == 1 && cols.len() == 1 {
        println!("No mixed Nash equilibrium.");
        return;
    }
    if rows.len() == 1 || cols.len() == 1 {
        println!("Infinite many mixed Nash equilibrium.");
        return;
    }
    // solve for p1's move
    let n = cols.len();
    let mut solve_q_a = DMatrix::repeat(n, n, 1.);
    let mut solve_q_b = DVector::zeros(n);
    solve_q_b[n-1] = 1.;
    for i in 0..n-1 {
        for j in 0..n {
            solve_q_a[(i,j)] = (grid[[rows[i], cols[j], 0]] - grid[[rows[i+1], cols[j], 0]]) as f64;
        }
    }
    let decomp_q = solve_q_a.lu();
    let sol_q = decomp_q.solve(&solve_q_b).expect("Solving linear system failed.");

    // solve for p0's move
    let m = rows.len();
    let mut solve_p_a = DMatrix::repeat(m, m, 1.);
    let mut solve_p_b = DVector::zeros(m);
    solve_p_b[m-1] = 1.;
    for i in 0..m-1 {
        for j in 0..m {
            solve_p_a[(i, j)] = (grid[[rows[j], cols[i], 1]] - grid[[rows[j], cols[i+1], 1]]) as f64;
        }
    }
    let decomp_p = solve_p_a.lu();
    let sol_p = decomp_p.solve(&solve_p_b).expect("Solving linear system failed.");

    if sol_p.iter().any(|&v| v < 0.) || sol_q.iter().any(|&v| v < 0.) {
        println!("No mixed Nash or unable to find one.");
        return;
    } else {
        println!("Mixed Nash found: ");
        println!("Player 1");
        for (index, row) in rows.iter().enumerate() {
            println!("Strategy {} with probability {}", row, sol_p[index]);
        }
        println!("Player 2");
        for (index, col) in cols.iter().enumerate() {
            println!("Strategy {} with probability {}", col, sol_q[index]);
        }
        return;
    }

}

fn main() {
    let args: Vec<String> = env::args().collect();
    let filepath = &args[1];
    println!("Read {}", filepath);
    let payoff_grid = read_to_grid(filepath);
    let pure_nash = solve_pure(&payoff_grid);
    println!("Pure strategy Nash {:?}", pure_nash);
    solve_mixed(&payoff_grid);
}
