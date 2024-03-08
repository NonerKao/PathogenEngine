use clap::{Parser, ValueEnum};

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::io::Write;
use std::rc::Rc;

use pathogen_engine::core::grid_coord::{Coord, MAP_OFFSET};
use pathogen_engine::core::status_code::str_to_full_msg;
use pathogen_engine::core::tree::TreeNode;
use pathogen_engine::core::*;

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

    /// Execution mode
    #[arg(long)]
    mode: Mode,
}

#[derive(Clone, Debug, PartialEq, ValueEnum)]
enum Mode {
    SGF,
    DATASET,
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
    let mut rng = from_seed(args.seed.clone());
    let shared_rng = Rc::new(RefCell::new(rng.clone()));
    let mut g = Game::init_with_rng(Some(tn.clone()), shared_rng.clone());
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

    // Generate
    let env_full = env_coord_pool.clone();
    loop {
        let e_index = rng.gen_range(0..env_coord_pool.len());
        let m_index = rng.gen_range(0..map_coord_pool.len());
        let c = if g.phase != Phase::Setup3 {
            env_coord_pool[e_index].clone() //env_coord_pool.choose(&mut rng).unwrap().clone();
        } else {
            map_coord_pool[m_index].clone() //env_coord_pool.choose(&mut rng).unwrap().clone();
        };
        let r = g.setup_with_alpha(&c); //map_coord_pool.choose(&mut rng).unwrap())
        match r {
            Ok(Phase::Main(1)) => {
                break;
            }
            Ok(x) => {
                #[cfg(debug_assertions)]
                {
                    println!("{:?} done one step", x);
                }
                env_coord_pool = env_full.clone();
            }
            Err(e) => {
                #[cfg(debug_assertions)]
                {
                    println!("{} from attempting {}", str_to_full_msg(e), c);
                }
                env_coord_pool.retain(|x| *x != *c);
            }
        }
    }

    to_file(&g)
}

fn encode(g: &Game, buffer: &mut String) {
    for i in 0..SIZE as usize {
        for j in 0..SIZE as usize {
            let c = Coord::new(i.try_into().unwrap(), j.try_into().unwrap());
            if g.env.get(&c).unwrap() == &World::Humanity {
                buffer.push('H');
            } else {
                buffer.push(' ');
            }
        }
    }
    for i in 0..SIZE as usize {
        for j in 0..SIZE as usize {
            let c = Coord::new(i.try_into().unwrap(), j.try_into().unwrap());
            if g.env.get(&c).unwrap() == &World::Underworld {
                buffer.push('U');
            } else {
                buffer.push(' ');
            }
        }
    }
    for i in 0..SIZE as usize {
        for j in 0..SIZE as usize {
            let c = Coord::new(i.try_into().unwrap(), j.try_into().unwrap());
            match g.stuff.get(&c) {
                Some(_) => {
                    buffer.push('S');
                }
                _ => {
                    buffer.push(' ');
                }
            }
        }
    }
    let d0 = g.character.get(&(World::Humanity, Camp::Doctor)).unwrap();
    let d1 = g.character.get(&(World::Underworld, Camp::Doctor)).unwrap();
    for i in 0..SIZE as usize {
        for j in 0..SIZE as usize {
            let c = Coord::new(i.try_into().unwrap(), j.try_into().unwrap());
            if c == *d0 || c == *d1 {
                buffer.push('D');
            } else {
                buffer.push(' ');
            }
        }
    }
    let p0 = g.character.get(&(World::Humanity, Camp::Plague)).unwrap();
    let p1 = g.character.get(&(World::Underworld, Camp::Plague)).unwrap();
    for i in 0..SIZE as usize {
        for j in 0..SIZE as usize {
            let c = Coord::new(i.try_into().unwrap(), j.try_into().unwrap());
            if c == *p0 || c == *p1 {
                buffer.push('P');
            } else {
                buffer.push(' ');
            }
        }
    }
    let m = g.map.get(&Camp::Doctor).unwrap();
    for i in -MAP_OFFSET.x..=MAP_OFFSET.x {
        for j in -MAP_OFFSET.y..=MAP_OFFSET.y {
            let c = Coord::new(i.try_into().unwrap(), j.try_into().unwrap());
            if *m == c {
                buffer.push('M');
            } else {
                buffer.push(' ');
            }
        }
    }
}

fn to_file(g: &Game) -> std::io::Result<()> {
    let args = Args::parse();
    let mut buffer = String::new();
    if args.mode == Mode::DATASET {
        encode(&g, &mut buffer);
    } else {
        g.history.borrow().to_root().borrow().to_string(&mut buffer);
    }
    match args.save {
        Some(filename) => {
            if args.mode == Mode::SGF {
                let mut file = File::create(filename.as_str())?;
                write!(file, "{}", buffer)?;
            } else {
                let mut file = OpenOptions::new()
                    .append(true) // Open the file in append mode.
                    .create(true) // Create the file if it does not exist.
                    .open(filename)?;
                write!(file, "{}", buffer)?;
            }
        }
        None => {}
    }
    Ok(())
}

fn from_seed(es: Option<String>) -> StdRng {
    let mut seed: [u8; 32] = [0; 32]; // Initialize with zeros

    if let Some(s) = es {
        let bytes = s.as_bytes();
        seed[..bytes.len()].copy_from_slice(bytes);
    }
    StdRng::from_seed(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_encode1() {
        let s0 = "(
            ;C[Setup0]
            AW[fa][eb][be][fd][ee][ef][ab][ad][cf][ca][bb][dc][ec][de][ac]
            AB[aa][ae][bf][ba][db][af][dd][fe][bc][ea][fb][cd][df][ed][cc][bd][ff][da][cb][ce][fc]
            ;C[Setup1]AB[ba][ee][af][fb]
            ;C[Setup2]AW[ec]
            ;C[Setup2]AB[bc]
            ;C[Setup2]AW[db]
            ;C[Setup2]AB[de]
            ;C[Setup3]AW[jj]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));

        let mut buffer = String::new();
        let mut ans = String::from(" HHH   H  H H    H  H H  HH HHH  H  ");
        ans = ans + "U   UUU UU U UUUU UU U UU  U   UU UU";
        ans = ans + "     SS                     S  S    ";
        ans = ans + "                   D      D         ";
        ans = ans + "        P             P             ";
        ans = ans + "                  M      ";
        encode(&g, &mut buffer);
        assert_eq!(buffer, ans);
    }
}
