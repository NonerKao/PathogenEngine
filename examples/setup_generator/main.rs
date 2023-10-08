use clap::Parser;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::fs::File;
use std::io::Read;
use std::io::Write;

use pathogen_engine::core::status_code::str_to_full_msg;
use pathogen_engine::core::tree::TreeNode;
use pathogen_engine::core::{Game, Phase};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// SGF file to be loaded from
    #[arg(short, long)]
    load: Option<String>,

    /// SGF file to be saved to
    #[arg(short, long)]
    save: Option<String>,

    /// random seed in ASCII string, at most the starting 32 bytes are used
    #[arg(long)]
    seed: Option<String>,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let e =
        "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen])".to_string();
    let mut iter = e.trim().chars().peekable();
    let mut contents = String::new();
    match args.load {
        Some(filename) => {
            let mut file = File::open(filename.as_str())?;
            file.read_to_string(&mut contents)
                .expect("Failed to read file");
            iter = contents.trim().chars().peekable();
        }
        None => {}
    }

    // main
    let tn = TreeNode::new(&mut iter, None);
    let mut g = Game::init(Some(tn.clone()));
    if g.is_setup() {
        to_file(&g)?;
        return Ok(());
    }
    let mut map_coord_pool: Vec<String> = Vec::new();
    let map_base: i32 = 'i' as i32;
    // Because we only need to generate the Setup3 steps now
    // Doctors remain in the inner 3x3
    for i in -1..=1 {
        for j in -1..=1 {
            let mut s = String::new();
            s.push(std::char::from_u32((j + map_base) as u32).unwrap());
            s.push(std::char::from_u32((i + map_base) as u32).unwrap());
            map_coord_pool.push(s);
        }
    }
    let mut env_coord_pool: Vec<String> = Vec::new();
    let env_base: i32 = 'a' as i32;
    for i in 0..=5 {
        for j in 0..=5 {
            let mut s = String::new();
            s.push(std::char::from_u32((j + env_base) as u32).unwrap());
            s.push(std::char::from_u32((i + env_base) as u32).unwrap());
            env_coord_pool.push(s);
        }
    }

    // setup seed
    let mut rng = from_seed(args.seed);
    let env_full = env_coord_pool.clone();
    loop {
        let c = env_coord_pool.choose(&mut rng).unwrap().clone();
        let r = if g.phase != Phase::Setup3 {
            g.setup_with_alpha(&c)
        } else {
            g.setup_with_alpha(map_coord_pool.choose(&mut rng).unwrap())
        };
        match r {
            Ok(Phase::Main(1)) => {
                break;
            }
            Ok(x) => {
                println!("{:?} done one step", x);
                env_coord_pool = env_full.clone();
            }
            Err(e) => {
                println!("{} from attempting {}", str_to_full_msg(e), c);
                env_coord_pool.retain(|x| *x != *c);
            }
        }
    }

    to_file(&g)
}

fn to_file(g: &Game) -> std::io::Result<()> {
    let args = Args::parse();
    let mut buffer = String::new();
    g.history.borrow().to_root().borrow().to_string(&mut buffer);
    match args.save {
        Some(filename) => {
            let mut file = File::create(filename.as_str())?;
            write!(file, "{}", buffer)?;
        }
        None => {}
    }
    Ok(())
}

fn from_seed(es: Option<String>) -> StdRng {
    let mut seed: [u8; 32] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32,
    ];
    match es {
        Some(s) => {
            for (index, c) in s.chars().enumerate() {
                if index as i32 >= 32 {
                    break;
                }
                seed[index] = c as u8;
            }
        }
        None => {}
    }
    let rng = StdRng::from_seed(seed);
    rng
}
