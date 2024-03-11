use clap::Parser;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cell::RefCell;
use std::fs::{create_dir, File};
use std::io::{self, Read, Write};
use std::rc::Rc;

use pathogen_engine::core::action::Action;
use pathogen_engine::core::action::ActionPhase;
use pathogen_engine::core::tree::TreeNode;
use pathogen_engine::core::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// SGF file to be loaded from
    #[arg(short, long)]
    load: String,

    /// SGF file to be loaded from
    #[arg(short, long)]
    save: Option<String>,

    /// Random seed
    #[arg(long)]
    seed: Option<String>,

    /// Iterations, round down to the nearest thousand
    #[arg(short, long)]
    iter: Option<usize>,
}

struct ActionMonitor {
    action: Action,
}

impl ActionMonitor {
    fn new() -> ActionMonitor {
        let action = Action::new();
        ActionMonitor { action }
    }
}

impl Drop for ActionMonitor {
    fn drop(&mut self) {
        if self.action.action_phase != ActionPhase::Done {
            println!("{:?}", self.action);
        }
    }
}

fn next(g: &mut Game, a: &Action) {
    g.append_history_with_new_tree(&a.to_sgf_string(g));
    g.commit_action(&a);
    g.next();
}

fn main() -> Result<(), std::io::Error> {
    let args = Args::parse();

    let mut output_path = if let Some(ref op) = args.save {
        String::from(op)
    } else {
        String::from("dummy")
    };

    if args.save != None {
        output_path = output_path + "/" + &get_setup_name(args.load.as_str());
        if let Err(e) = create_dir(output_path.as_str()) {
            match e.kind() {
                std::io::ErrorKind::AlreadyExists => {}
                _ => {
                    panic!("Error creating directory: {:?}", e);
                }
            }
        }
    }

    let mut contents = String::new();
    let mut file = File::open(args.load.as_str())?;
    file.read_to_string(&mut contents)
        .expect("Failed to read file");
    let origin_iter = contents.trim().chars().peekable();
    drop(file);
    if args.save != None {
        to_misc(&(output_path.clone() + "/setup.sgf"), &contents)?;
    }

    let n = match args.iter {
        Some(n) => n,
        _ => 1,
    };

    let unit = 1000;

    let mut seed: [u8; 32] = [0; 32];
    if let Some(s) = args.seed {
        let bytes = s.as_bytes();
        seed[..bytes.len()].copy_from_slice(bytes);
    }
    let mut rng = StdRng::from_seed(seed);
    let ranges: Vec<usize> = (0..n / unit).map(|_| rng.gen::<usize>()).collect();

    let (plague_wins, plague_total_moves, total_moves) = ranges
        .iter()
        .map(|&index| {
            let index_range: Vec<usize> = (index..index + unit).collect();
            let (plague_wins, plague_total_moves, total_moves) = index_range
                .par_iter()
                .map(|&num| {
                    let mut plague_wins = 0;
                    let mut plague_total_moves = 0;
                    let mut total_moves = 0;
                    let rng = from_seed(num);
                    let shared_rng = Rc::new(RefCell::new(rng));
                    let mut iter = origin_iter.clone();
                    let t = TreeNode::new(&mut iter, None);
                    let mut g = Game::init_with_rng(Some(t), shared_rng.clone());
                    if !g.is_setup() {
                        panic!("The game is either not ready or finished");
                    }

                    while let Phase::Main(x) = g.phase {
                        let mut ram = ActionMonitor::new();
                        if ram.action.random_move(&mut g) {
                            next(&mut g, &ram.action);
                        } else {
                            g.next();
                        }
                        let turn: usize = x.try_into().unwrap();
                        if g.is_ended() {
                            total_moves = total_moves + x;
                            if turn % 2 == 1 {
                                plague_total_moves = plague_total_moves + x;
                                plague_wins = plague_wins + 1;
                            }
                            let is = num.to_string();
                            if args.save != None {
                                match to_sgf(&(output_path.clone() + "/" + &is + ".sgf"), &g) {
                                    Ok(()) => {}
                                    _ => {
                                        panic!("to_sgf failed");
                                    }
                                }
                            }
                            break;
                        }
                    }
                    (plague_wins, plague_total_moves, total_moves)
                })
                .reduce(
                    || (0, 0, 0),
                    |(a0, a1, a2), (b0, b1, b2)| (a0 + b0, a1 + b1, a2 + b2),
                );
            (plague_wins, plague_total_moves, total_moves)
        })
        .fold((0, 0, 0), |(a0, a1, a2), (b0, b1, b2)| {
            (a0 + b0, a1 + b1, a2 + b2)
        });

    let result = format!(
        "{}/{}; steps: {}/{}",
        plague_wins, n, plague_total_moves, total_moves
    );
    let win_rate = (plague_wins as f32) / (n as f32);
    let bytes = win_rate.to_le_bytes();
    io::stdout().write_all(&bytes)?;
    io::stdout().flush()?;

    if args.save != None {
        to_misc(&(output_path.clone() + "/result.txt"), result.as_str())
    } else {
        Ok(())
    }
}

fn get_setup_name(filename: &str) -> String {
    if let Some(index1) = filename.rfind('.') {
        let (name1, _) = filename.split_at(index1);
        if let Some(index2) = name1.rfind('/') {
            let (_, name2) = name1.split_at(index2 + 1);
            name2.to_string()
        } else {
            name1.to_string()
        }
    } else {
        filename.to_string() // No suffix found
    }
}

fn from_seed(es: usize) -> StdRng {
    let mut seed: [u8; 32] = [0; 32]; // Initialize with zeros
    let s = format!("=!={}===", es);
    let bytes = s.as_bytes();
    seed[..bytes.len()].copy_from_slice(bytes);
    StdRng::from_seed(seed)
}

fn to_misc(filename: &str, buffer: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    write!(file, "{}", buffer)?;
    Ok(())
}

fn to_sgf(filename: &str, g: &Game) -> std::io::Result<()> {
    let mut buffer = String::new();
    g.history.borrow().to_root().borrow().to_string(&mut buffer);
    let mut file = File::create(filename)?;
    write!(file, "{}", buffer)?;
    Ok(())
}
