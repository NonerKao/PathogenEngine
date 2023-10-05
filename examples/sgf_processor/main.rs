use clap::Parser;

use std::fs::File;
use std::io::Read;
use std::io::Write;

use pathogen_engine::core::action::Action;
use pathogen_engine::core::grid_coord::Coord;
use pathogen_engine::core::status_code::str_to_full_msg;
use pathogen_engine::core::tree::TreeNode;
use pathogen_engine::core::Game;

/// Simple program to greet a person
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

    let e = "3333".to_string();
    let mut iter = e.trim().chars().peekable();
    let mut contents = String::new();
    match args.load {
        Some(filename) => {
            // Ideally, this should be the pure "view mode", where we read the games(s).
            // In reality, I need this to bridge the gap before all phases are treated equal.
            let mut file = File::open(filename.as_str())?;
            file.read_to_string(&mut contents)
                .expect("Failed to read file");
            iter = contents.trim().chars().peekable();
        }
        None => {}
    }

    // main
    let tn = TreeNode::new(&mut iter, None);
    let mut buffer = String::new();
    tn.borrow().to_string(&mut buffer);

    match args.save {
        Some(filename) => {
            let mut file = File::create(filename.as_str())?;
            write!(file, "{}", buffer)?;
        }
        None => {}
    }

    let mut g = Game::init(Some(tn));
    let mut a = Action::new();
    match a.add_hero(&g, Coord::new(0, 0)) {
        Ok(_) => {}
        Err(x) => {
            println!("{}", str_to_full_msg(x));
        }
    }
    g.commit_action(&a);

    Ok(())
}
