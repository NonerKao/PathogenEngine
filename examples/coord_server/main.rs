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

fn handle_client<T: Read + ReaderExtra + Write + WriterExtra>(
    stream: &mut T,
    g: &mut Game,
) -> bool {
    let mut buffer = [0; 1]; // to read the 1-byte action from agent

    let ea = Action::new();
    let ec: [u8; FC_LEN] = [0; FC_LEN];
    let mut s = "Ix03";
    if stream.update_agent(g, &ea, &ec, &s) == false {
        return false;
    }
    loop {
        match stream.peek(&mut buffer) {
            Ok(0) => {
                println!("Client disconnected.");
                return false;
            }
            Ok(_) => {
                let mut a = Action::new();
                let mut fc: [u8; FC_LEN] = [0; FC_LEN];

                // Check tree.rs:to_action() function for the following
                // big block of state machine. s for status code in the spec.

                // Add the map move first
                assert_eq!(a.action_phase, ActionPhase::SetMap);
                fc[0 /* set map */] = 1;
                'set_map: loop {
                    let bytes_read = stream.read(&mut buffer).unwrap();
                    assert_eq!(bytes_read, 1);
                    println!("{:?}", buffer);
                    let c = (buffer[0] as u8).to_coord();
                    match a.add_map_step(g, c) {
                        Err(e) => {
                            s = e;
                        }
                        Ok(o) => {
                            s = o;
                        }
                    }
                    if stream.update_agent(g, &a, &fc, &s) == false {
                        return false;
                    } else {
                        if s.as_bytes()[0] == b'E' {
                            continue 'set_map;
                        } else if s == "Ix00"
                        /* Skip!? */
                        {
                            g.switch();
                            return true;
                        }
                        break;
                    }
                }

                // Optional for Doctor: lockdown?
                if a.action_phase == ActionPhase::Lockdown {
                    fc[0 /* set map */] = 0;
                    fc[1 /* lockdown */] = 1;
                    'lockdown: loop {
                        let bytes_read = stream.read(&mut buffer).unwrap();
                        assert_eq!(bytes_read, 1);
                        println!("{:?}", buffer);
                        let c = (buffer[0] as u8).to_coord();
                        match a.add_lockdown_by_coord(g, c) {
                            Err(e) => {
                                println!("{}", e);
                                s = e;
                            }
                            Ok(o) => {
                                s = o;
                            }
                        }
                        if stream.update_agent(g, &a, &fc, &s) == false {
                            return false;
                        } else {
                            if s.as_bytes()[0] == b'E' {
                                continue 'lockdown;
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
                'set_character: loop {
                    let bytes_read = stream.read(&mut buffer).unwrap();
                    assert_eq!(bytes_read, 1);
                    println!("sc {:?}", buffer);
                    let c = (buffer[0] as u8).to_coord();
                    match a.add_character(g, c) {
                        Err(e) => {
                            s = e;
                        }
                        Ok(o) => {
                            s = o;
                        }
                    }
                    if stream.update_agent(g, &a, &fc, &s) == false {
                        return false;
                    } else {
                        if s.as_bytes()[0] == b'E' {
                            continue 'set_character;
                        }
                        break;
                    }
                }

                // Move the character on the board
                assert_eq!(a.action_phase, ActionPhase::BoardMove);
                fc[2 /* set character */] = 0;
                for i in 0..a.steps {
                    'board_move: loop {
                        let bytes_read = stream.read(&mut buffer).unwrap();
                        assert_eq!(bytes_read, 1);
                        println!("{:?}", buffer);
                        let c = (buffer[0] as u8).to_coord();
                        match a.add_board_single_step(g, c) {
                            Err(e) => {
                                s = e;
                            }
                            Ok(o) => {
                                s = o;
                            }
                        }
                        if stream.update_agent(g, &a, &fc, &s) == false {
                            return false;
                        } else {
                            if s.as_bytes()[0] == b'E' {
                                continue 'board_move;
                            }
                            break;
                        }
                    }
                    fc[3 /* board step */ + i] = 0;
                }

                // Set the markers
                assert!(fc.iter().all(|&x| x == 0));
                'set_marker: loop {
                    assert_eq!(a.action_phase, ActionPhase::SetMarkers);
                    for i in 0..DOCTOR_MARKER as usize {
                        fc[8 /* set character */ + i] = 1;
                    }
                    if g.turn == Camp::Plague {
                        fc[12 /* final one: Plague has only 4 markers */] = 0;
                    }
                    let mut i = 8;
                    while i < FC_LEN {
                        if fc[i] == 0 {
                            continue;
                        }
                        let bytes_read = stream.read(&mut buffer).unwrap();
                        assert_eq!(bytes_read, 1);
                        println!("{:?}", buffer);
                        let c = (buffer[0] as u8).to_coord();
                        match a.add_single_marker(g, c) {
                            Err(e) => {
                                s = e;
                            }
                            Ok(o) => {
                                s = o;
                                fc[i] = 0;
                                i = i + 1;
                            }
                        }
                        if stream.update_agent(g, &a, &fc, &s) == false {
                            return false;
                        } else {
                            if s.as_bytes()[0] == b'E' {
                                continue 'set_marker;
                            } else if s == "Ix02" {
                                break 'set_marker;
                            }
                        }
                    }
                }
                // commit the action to the game
                assert_eq!(a.action_phase, ActionPhase::Done);
                assert!(fc.iter().all(|&x| x == 0));
                // Holy shit, the order of the following two lines
                // is tricky. You shouldn't commit it first and then
                // try to interpret the Action with the new Game
                // status.
                g.append_history_with_new_tree(&a.to_sgf_string(g));
                g.commit_action(&a);
                g.next();
                break;
            }
            Err(e) => {
                println!("Error occurred: {:?}", e);
                return false;
            }
        }
    }
    return true;
}

trait WriterExtra {
    fn update_agent(&mut self, g: &Game, a: &Action, fc: &[u8; FC_LEN], s: &'static str) -> bool;
}

impl WriterExtra for TcpStream {
    fn update_agent(&mut self, g: &Game, a: &Action, fc: &[u8; FC_LEN], s: &'static str) -> bool {
        let encoded = encode(g, &a);
        let sb = s.as_bytes();
        let enc = encoded.as_slice().unwrap();

        let response = [&enc[..], &fc[..], &sb].concat();
        assert!(response.len() == 391);
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

trait ReaderExtra {
    fn peek(&mut self, buffer: &mut [u8]) -> std::io::Result<usize>;
}

impl ReaderExtra for TcpStream {
    fn peek(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        TcpStream::peek(self, buffer)
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

    let w = white_listener.accept().unwrap().0;
    let b = black_listener.accept().unwrap().0;

    Ok((w, b))
}

// For the test only
impl ReaderExtra for std::io::Cursor<&mut [u8]> {
    fn peek(&mut self, _buffer: &mut [u8]) -> std::io::Result<usize> {
        return Ok(1);
    }
}

// For the test only
impl WriterExtra for std::io::Cursor<&mut [u8]> {
    fn update_agent(&mut self, g: &Game, a: &Action, fc: &[u8; FC_LEN], s: &'static str) -> bool {
        let encoded = encode(g, &a);
        let sb = s.as_bytes();
        let enc = encoded.as_slice().unwrap();

        let response = [&enc[..], &fc[..], &sb].concat();
        assert!(response.len() == 391);
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
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_handle_client_with_cursor() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            ;C[Setup3]AW[ij]
            ;B[jj][ad][cd][ad][ad][ad][ad]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));

        const LEN: usize = 391 + (1 + 391) * 10;
        let mut buf_origin: [u8; LEN] = [0; LEN];
        let buf = &mut buf_origin[..];
        let s1 = ";W[ii][hh][aa][ab][bb][ab][ab][ab][aa][aa]";
        buf[391] = 46;
        buf[392 * 2 - 1] = 40;
        buf[392 * 3 - 1] = 0;
        buf[392 * 4 - 1] = 6;
        buf[392 * 5 - 1] = 7;
        buf[392 * 6 - 1] = 6;
        buf[392 * 7 - 1] = 0;
        buf[392 * 8 - 1] = 6;
        buf[392 * 9 - 1] = 0;
        buf[392 * 10 - 1] = 6;
        let mut fake_stream = Cursor::new(buf);
        assert!(handle_client(&mut fake_stream, &mut g) == true);
        let mut buffer = String::new();
        g.history.borrow().to_string(&mut buffer);
        assert_eq!(buffer, s1);
    }

    #[test]
    fn test_handle_client_with_cursor2() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            ;C[Setup3]AW[ij]
            ;B[jj][ad][cd][ad][ad][ad][ad]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));

        const LEN: usize = 391 + (1 + 391) * 11;
        let mut buf_origin: [u8; LEN] = [0; LEN];
        let buf = &mut buf_origin[..];
        // in real correct SGF file, of course we cannot assign "hi" as the
        // lockdown position, but this is for a demo
        let s1 = ";W[ii][hi][hh][aa][ab][bb][ab][ab][ab][aa][aa]";
        buf[391] = 46;
        buf[392 * 2 - 1] = 45;
        buf[392 * 3 - 1] = 40;
        buf[392 * 4 - 1] = 0;
        buf[392 * 5 - 1] = 6;
        buf[392 * 6 - 1] = 7;
        buf[392 * 7 - 1] = 6;
        buf[392 * 8 - 1] = 0;
        buf[392 * 9 - 1] = 6;
        buf[392 * 10 - 1] = 0;
        buf[392 * 11 - 1] = 6;
        let mut fake_stream = Cursor::new(buf);
        assert!(handle_client(&mut fake_stream, &mut g) == true);
        let mut buffer = String::new();
        g.history.borrow().to_string(&mut buffer);
        assert_eq!(buffer, s1.replace("[hi]", ""));
    }

    #[test]
    fn test_handle_client_with_cursor3() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            ;C[Setup3]AW[ij]
            ;B[jj][ad][cd][ad][ad][ad][ad]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));

        const LEN: usize = 391 + (1 + 391) * 11;
        let mut buf_origin: [u8; LEN] = [0; LEN];
        let buf = &mut buf_origin[..];
        // It's difficult to reason why it was ([bf] was [af]) working.
        // [af] happended to be the other character. Before the candidate
        // mechanism, there is no check that [hh] lockdown with [af]
        // character cannot work together. BUT! Even now, there is no
        // way to do the check, so theoretically it should still fail.
        //
        let s1 = ";W[ii][hh][bf][aa][ab][bb][ab][ab][ab][aa][aa]";
        buf[391] = 46;
        buf[392 * 2 - 1] = 40;
        buf[392 * 3 - 1] = 31;
        buf[392 * 4 - 1] = 0;
        buf[392 * 5 - 1] = 6;
        buf[392 * 6 - 1] = 7;
        buf[392 * 7 - 1] = 6;
        buf[392 * 8 - 1] = 0;
        buf[392 * 9 - 1] = 6;
        buf[392 * 10 - 1] = 0;
        buf[392 * 11 - 1] = 6;
        let mut fake_stream = Cursor::new(buf);
        assert!(handle_client(&mut fake_stream, &mut g) == true);
        let mut buffer = String::new();
        g.history.borrow().to_string(&mut buffer);
        assert_eq!(buffer, s1.replace("[bf]", ""));
    }

    #[test]
    fn test_handle_client_with_cursor3_2() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            ;C[Setup3]AW[ij]
            ;B[jj][ad][cd][ad][ad][ad][ad]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));

        const LEN: usize = 391 + (1 + 391) * 11;
        let s2 = ";W[ii][hh][af][aa][ab][bb][ab][ab][ab][aa][aa]";
        let mut buf_origin2: [u8; LEN] = [0; LEN];
        let buf2 = &mut buf_origin2[..];
        buf2[391] = 46;
        buf2[392 * 2 - 1] = 40;
        buf2[392 * 3 - 1] = 30;
        buf2[392 * 4 - 1] = 0;
        buf2[392 * 5 - 1] = 6;
        buf2[392 * 6 - 1] = 7;
        buf2[392 * 7 - 1] = 6;
        buf2[392 * 8 - 1] = 0;
        buf2[392 * 9 - 1] = 6;
        buf2[392 * 10 - 1] = 0;
        buf2[392 * 11 - 1] = 6;
        let mut fake_stream = Cursor::new(buf2);
        assert!(handle_client(&mut fake_stream, &mut g) == true);
        let mut buffer = String::new();
        g.history.borrow().to_string(&mut buffer);
        assert_eq!(buffer, s2.replace("[af]", ""));
    }

    #[test]
    fn test_handle_client_with_cursor4() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            ;C[Setup3]AW[ij]
            ;B[jj][ad][cd][ad][ad][ad][ad]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));

        const LEN: usize = 391 + (1 + 391) * 12;
        let mut buf_origin: [u8; LEN] = [0; LEN];
        let buf = &mut buf_origin[..];
        let s1 = ";W[ii][hh][aa][dd][ab][ac][bb][ab][ab][ab][aa][aa]";
        buf[391] = 46;
        buf[392 * 2 - 1] = 40;
        buf[392 * 3 - 1] = 0;
        buf[392 * 4 - 1] = 21;
        buf[392 * 5 - 1] = 6;
        buf[392 * 6 - 1] = 12;
        buf[392 * 7 - 1] = 7;
        buf[392 * 8 - 1] = 6;
        buf[392 * 9 - 1] = 0;
        buf[392 * 10 - 1] = 6;
        buf[392 * 11 - 1] = 0;
        buf[392 * 12 - 1] = 6;
        let mut fake_stream = Cursor::new(buf);
        assert!(handle_client(&mut fake_stream, &mut g) == true);
        let mut buffer = String::new();
        g.history.borrow().to_string(&mut buffer);
        assert_eq!(buffer, s1.replace("[dd]", "").replace("[ac]", ""));
    }

    #[test]
    fn test_handle_client_with_cursor5() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            ;C[Setup3]AW[ij]
            ;B[jj][ad][cd][ad][ad][ad][ad]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));

        const LEN: usize = 391 + (1 + 391) * 15;
        let mut buf_origin: [u8; LEN] = [0; LEN];
        let buf = &mut buf_origin[..];
        // in real correct SGF file, of course we cannot assign "hi" as the
        // lockdown position, but this is for a demo
        let s1 = ";W[ii][hh][aa][ab][bb][aa][aa][aa][aa][aa][ab][aa][aa][aa][aa]";
        buf[391] = 46;
        buf[392 * 2 - 1] = 40;
        buf[392 * 3 - 1] = 0;
        buf[392 * 4 - 1] = 6;
        buf[392 * 5 - 1] = 7;
        buf[392 * 6 - 1] = 0;
        buf[392 * 7 - 1] = 0;
        buf[392 * 8 - 1] = 0;
        buf[392 * 9 - 1] = 0;
        buf[392 * 10 - 1] = 0;
        buf[392 * 11 - 1] = 0;
        buf[392 * 12 - 1] = 0;
        buf[392 * 13 - 1] = 0;
        buf[392 * 14 - 1] = 0;
        buf[392 * 15 - 1] = 6;
        let mut fake_stream = Cursor::new(buf);
        assert!(handle_client(&mut fake_stream, &mut g) == true);
        let mut buffer = String::new();
        g.history.borrow().to_string(&mut buffer);
        assert_eq!(buffer, s1.replace("[aa][aa][aa][aa][aa]", ""));
    }
}
