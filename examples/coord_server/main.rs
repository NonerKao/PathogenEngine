use clap::Parser;
use ndarray::{Array, Array1};
use std::fs::File;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use pathogen_engine::core::action::Action;
use pathogen_engine::core::grid_coord::Coord;
use pathogen_engine::core::grid_coord::MAP_OFFSET;
use pathogen_engine::core::tree::TreeNode;
use pathogen_engine::core::*;

const MAX_STEPS: usize = 5;

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

fn encode(g: &Game, _a: &Action) -> Array1<u8> {
    // Check the README/HACKING for why it is 9
    let mut e = Array::from_shape_fn((SIZE as usize, SIZE as usize, 9 as usize), |(_, _, _)| {
        0 as u8
    });
    for i in 0..SIZE as usize {
        for j in 0..SIZE as usize {
            let c = Coord::new(i.try_into().unwrap(), j.try_into().unwrap());
            if g.env.get(&c).unwrap() == &World::Underworld {
                e[[i, j, 0 /*Underworld*/]] = 1;
            } else {
                e[[i, j, 1 /*Humanity*/]] = 1;
            }
            match g.stuff.get(&c) {
                None => {}
                Some((Camp::Doctor, Stuff::Colony)) => {
                    e[[i, j, 4 /*Doctor Colony*/]] = 1;
                }
                Some((Camp::Plague, Stuff::Colony)) => {
                    e[[i, j, 5 /*Plague Colony*/]] = 1;
                }
                Some((Camp::Doctor, Stuff::Marker(x))) => {
                    e[[i, j, 6 /*Doctor Marker*/]] = *x;
                }
                Some((Camp::Plague, Stuff::Marker(x))) => {
                    e[[i, j, 7 /*Plague Marker*/]] = *x;
                }
            }
        }
    }
    for ((_, camp), c) in g.character.iter() {
        if *camp == Camp::Doctor {
            e[[c.y as usize, c.x as usize, 2 /*Doctor Hero*/]] = 1;
        } else {
            e[[c.y as usize, c.x as usize, 3 /*Plague Hero*/]] = 1;
        }
    }
    // 2 for the two sides
    let mut m = Array::from_shape_fn(
        (MAP_SIZE as usize, MAP_SIZE as usize, 2 as usize),
        |(_, _, _)| 0 as u8,
    );

    for (camp, c) in g.map.iter() {
        if *camp == Camp::Doctor {
            m[[
                (c.y + MAP_OFFSET.y) as usize,
                (c.x + MAP_OFFSET.x) as usize,
                0, /* Doctor Marker */
            ]] = 1;
        } else {
            m[[
                (c.y + MAP_OFFSET.y) as usize,
                (c.x + MAP_OFFSET.x) as usize,
                1, /* Plague Marker */
            ]] = 1;
        }
    }

    // for flow control
    let fc = Array::from_shape_fn(2 /* map moves */ + MAX_STEPS + MAX_MARKER as usize, |_| {
        0 as u8
    });

    let ret = e
        .into_shape(((SIZE as usize) * (SIZE as usize) * 9,))
        .unwrap()
        .into_iter()
        .chain(
            m.into_shape((MAP_SIZE * MAP_SIZE * 2,))
                .unwrap()
                .into_iter(),
        )
        .chain(
            fc.into_shape((2 + MAX_STEPS + MAX_MARKER as usize,))
                .unwrap()
                .into_iter(),
        )
        .collect::<Array1<_>>();
    ret
}

fn handle_client(mut stream: &TcpStream, _g: &mut Game) -> bool {
    let mut buffer = [0; 1]; // to read the 1-byte action from agent
    let _a = Action::new();
    loop {
        match stream.peek(&mut buffer) {
            Ok(0) => {
                println!("Client disconnected.");
                return false;
            }
            Ok(_) => {
                let _bytes_read = stream.read(&mut buffer).unwrap();
                println!("{:?}", buffer);

                let _action = buffer[0] as usize;

                // Here, interact with the IIG environment using the action
                // and get the resulting game state and status code.
                let game_state: [u8; 150] = [0; 150]; // This should be obtained from IIG environment
                let status_code: [u8; 4] = [0, 0, 0, 0]; // This too, from IIG

                let response = [&game_state[..], &status_code[..]].concat();
                match stream.write(&response) {
                    Err(_) => {
                        println!("Client disconnected.");
                        return false;
                    }
                    _ => {}
                }
            }
            Err(e) => {
                println!("Error occurred: {:?}", e);
                return false;
            }
        }
    }
    return true;
}

fn main() -> Result<(), std::io::Error> {
    let args = Args::parse();

    let e = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen];C[Setup0]AW[df][da][db][ab][ba][ea][cc][fd][fc][af][ce][ee][cb][ef][bd][bc][fe];C[Setup0]AB[aa][ec][cd][bb][ae][be][ed][ad][ff][eb][fb][bf][ac][fa][dd][dc][cf][de][ca];C[Setup1]AB[fc];C[Setup1]AB[ed];C[Setup1]AB[ab];C[Setup1]AB[cf];C[Setup2]AW[cc];C[Setup2]AB[ec];C[Setup2]AW[ca];C[Setup2]AB[ce];C[Setup3]AW[jj])".to_string();
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

    let t = TreeNode::new(&mut iter, None);
    let mut g = Game::init(Some(t));
    if !g.is_setup() {
        panic!("The game is not ready");
    }

    let (mut w, mut b) = network_setup()?;
    let mut w_live = true;
    let mut b_live = true;

    while w_live || b_live {
        if w_live {
            w_live = handle_client(&mut w, &mut g);
        }
        if b_live {
            b_live = handle_client(&mut b, &mut g);
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

fn network_setup() -> Result<(TcpStream, TcpStream), std::io::Error> {
    let white_listener = TcpListener::bind("127.0.0.1:6241").unwrap();
    let black_listener = TcpListener::bind("127.0.0.1:3698").unwrap();

    let hello: [u8; 154] = [0; 154];
    let mut w = white_listener.accept().unwrap().0;
    w.write(&hello)?;
    let mut b = black_listener.accept().unwrap().0;
    b.write(&hello)?;

    Ok((w, b))
}
