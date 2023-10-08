use clap::Parser;

use std::fs::File;
use std::io::Read;
use std::io::Write;

use pathogen_engine::core::action::Action;
use pathogen_engine::core::grid_coord::Coord;
use pathogen_engine::core::tree::TreeNode;
use pathogen_engine::core::Game;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// SGF file to be loaded from
    #[arg(short, long)]
    load: Option<String>,

    /// SGF file to be saved to
    #[arg(short, long)]
    save: Option<String>,
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
    let g = Game::init(Some(tn.clone()));
    let mut map_coord_pool: Vec<String> = Vec::new();
    let map_base: i32 = 'i' as i32;
    for i in -2..=2 {
        for j in -2..=2 {
            if let c = i * j {
                if c >= 4 || c <= -4 {
                    continue;
                }
            }
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
    println!("{:?}", map_coord_pool);
    println!("{:?}", env_coord_pool);

    let mut buffer = String::new();
    tn.borrow().to_string(&mut buffer);
    match args.save {
        Some(filename) => {
            let mut file = File::create(filename.as_str())?;
            write!(file, "{}", buffer)?;
        }
        None => {}
    }

    Ok(())
}
