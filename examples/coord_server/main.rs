use clap::Parser;
use ndarray::{Array, Array1};
use std::fs::File;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use pathogen_engine::core::action::Action;
use pathogen_engine::core::action::ActionPhase;
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

fn encode(g: &Game, a: &Action) -> Array1<u8> {
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

    // apply the effect of a partial Action to the Game

    // wrap it up
    let ret = e
        .into_shape(((SIZE as usize) * (SIZE as usize) * 9,))
        .unwrap()
        .into_iter()
        .chain(
            m.into_shape((MAP_SIZE * MAP_SIZE * 2,))
                .unwrap()
                .into_iter(),
        )
        .collect::<Array1<_>>();

    assert_eq!(ret.len(), 374);
    ret
}

trait ActionCoord {
    fn to_coord(&self) -> Coord;
}

impl ActionCoord for u8 {
    fn to_coord(&self) -> Coord {
        match *self {
            // env
            0 => Coord::new(0, 0),
            1 => Coord::new(1, 0),
            2 => Coord::new(2, 0),
            3 => Coord::new(3, 0),
            4 => Coord::new(4, 0),
            5 => Coord::new(5, 0),
            6 => Coord::new(0, 1),
            7 => Coord::new(1, 1),
            8 => Coord::new(2, 1),
            9 => Coord::new(3, 1),
            10 => Coord::new(4, 1),
            11 => Coord::new(5, 1),
            12 => Coord::new(0, 2),
            13 => Coord::new(1, 2),
            14 => Coord::new(2, 2),
            15 => Coord::new(3, 2),
            16 => Coord::new(4, 2),
            17 => Coord::new(5, 2),
            18 => Coord::new(0, 3),
            19 => Coord::new(1, 3),
            20 => Coord::new(2, 3),
            21 => Coord::new(3, 3),
            22 => Coord::new(4, 3),
            23 => Coord::new(5, 3),
            24 => Coord::new(0, 4),
            25 => Coord::new(1, 4),
            26 => Coord::new(2, 4),
            27 => Coord::new(3, 4),
            28 => Coord::new(4, 4),
            29 => Coord::new(5, 4),
            30 => Coord::new(0, 5),
            31 => Coord::new(1, 5),
            32 => Coord::new(2, 5),
            33 => Coord::new(3, 5),
            34 => Coord::new(4, 5),
            35 => Coord::new(5, 5),
            // map
            36 => Coord::new(-1, -2),
            37 => Coord::new(0, -2),
            38 => Coord::new(1, -2),
            39 => Coord::new(-2, -1),
            40 => Coord::new(-1, -1),
            41 => Coord::new(0, -1),
            42 => Coord::new(1, -1),
            43 => Coord::new(2, -1),
            44 => Coord::new(-2, 0),
            45 => Coord::new(-1, 0),
            46 => Coord::new(0, 0),
            47 => Coord::new(1, 0),
            48 => Coord::new(2, 0),
            49 => Coord::new(-2, 1),
            50 => Coord::new(-1, 1),
            51 => Coord::new(0, 1),
            52 => Coord::new(1, 1),
            53 => Coord::new(2, 1),
            54 => Coord::new(-1, 2),
            55 => Coord::new(0, 2),
            56 => Coord::new(1, 2),
            // The rest are not supposed to happen
            _ => Coord::new(-2, -2),
        }
    }
}

const FC_LEN: usize = 2 /* map move */ + 1 /* set character*/ + MAX_STEPS + DOCTOR_MARKER as usize;

fn handle_client(stream: &mut TcpStream, g: &mut Game) -> bool {
    let mut buffer = [0; 1]; // to read the 1-byte action from agent
    'restart: loop {
        match stream.peek(&mut buffer) {
            Ok(0) => {
                println!("Client disconnected.");
                return false;
            }
            Ok(_) => {
                let mut a = Action::new();
                let mut fc: [u8; FC_LEN] = [0; FC_LEN];
                let _bytes_read = stream.read(&mut buffer).unwrap();
                println!("{:?}", buffer);

                let action = buffer[0] as u8;
                let c = action.to_coord();

                // Check tree.rs:to_action() function for the following
                // big block. s for status code in the spec.
                let mut s = "Ix01";

                // Add the map move first
                assert_eq!(a.action_phase, ActionPhase::SetMap);
                fc[0 /* set map */] = 1;
                loop {
                    match a.add_map_step(g, c) {
                        Err(e) => {
                            s = e;
                        }
                        Ok(()) => {}
                    }
                    if stream.update_agent(g, &a, &fc, &s) == false {
                        return false;
                    } else {
                        if s.as_bytes()[0] == b'E' {
                            continue 'restart;
                        }
                        break;
                    }
                }

                // Optional for Doctor: lockdown?
                if a.action_phase == ActionPhase::Lockdown {
                    fc[0 /* set map */] = 0;
                    fc[1 /* lockdown */] = 1;
                    loop {
                        match a.add_lockdown_by_coord(g, c) {
                            Err(e) => {
                                s = e;
                            }
                            Ok(()) => {}
                        }
                        if stream.update_agent(g, &a, &fc, &s) == false {
                            return false;
                        } else {
                            if s.as_bytes()[0] == b'E' {
                                continue 'restart;
                            }
                            break;
                        }
                    }
                }

                // Set the character
                assert_eq!(a.action_phase, ActionPhase::SetCharacter);
                fc[0 /* set map */] = 0;
                fc[1 /* lockdown */] = 0;
                for i in 0..=a.steps {
                    fc[2 /* set character */ + i] = 1;
                }
                loop {
                    match a.add_character(g, c) {
                        Err(e) => {
                            s = e;
                        }
                        Ok(()) => {}
                    }
                    if stream.update_agent(g, &a, &fc, &s) == false {
                        return false;
                    } else {
                        if s.as_bytes()[0] == b'E' {
                            continue 'restart;
                        }
                        break;
                    }
                }

                // Move the character on the board
                assert_eq!(a.action_phase, ActionPhase::BoardMove);
                fc[2 /* set character */] = 0;
                for i in 0..a.steps {
                    loop {
                        match a.add_board_single_step(g, c) {
                            Err(e) => {
                                s = e;
                            }
                            Ok(()) => {}
                        }
                        if stream.update_agent(g, &a, &fc, &s) == false {
                            return false;
                        } else {
                            if s.as_bytes()[0] == b'E' {
                                continue 'restart;
                            }
                            break;
                        }
                    }
                    fc[3 /* board step */ + i] = 0;
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

trait StreamInform {
    fn update_agent(&mut self, g: &Game, a: &Action, fc: &[u8; FC_LEN], s: &'static str) -> bool;
}

impl StreamInform for TcpStream {
    fn update_agent(&mut self, g: &Game, a: &Action, fc: &[u8; FC_LEN], s: &'static str) -> bool {
        let encoded = encode(g, &a);
        let sb = s.as_bytes();
        let enc = encoded.as_slice().unwrap();

        let response = [&enc[..], &fc[..], &sb].concat();
        match self.write(&response) {
            Err(_) => {
                println!("Client disconnected.");
                return false;
            }
            _ => {
                return true;
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
}
