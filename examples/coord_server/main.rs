use clap::Parser;
use ndarray::{Array, Array1};
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use pathogen_engine::core::action::Action;
use pathogen_engine::core::action::ActionPhase;
use pathogen_engine::core::action::Candidate;
use pathogen_engine::core::grid_coord::{Coord, MAP_OFFSET};
use pathogen_engine::core::tree::TreeNode;
use pathogen_engine::core::*;

const MAX_STEPS: usize = 5;

const BOARD_DATA: usize = 288; /*8x6x6*/
const MAP_DATA: usize = 50; /*2x5x5*/
const TURN_DATA: usize = 25; /*1x5x5*/
const FLOW_MAP_DATA: usize = 50; /*2x5x5*/
const FLOW_ENV_DATA: usize = 396; /*11x6x6*/
const FLOW_DATA: usize = FLOW_MAP_DATA + FLOW_ENV_DATA; /*2x5x5+11x6x6*/

const CODE_DATA: usize = 4;

const DATA_UNIT: usize = BOARD_DATA + MAP_DATA + TURN_DATA + FLOW_DATA + CODE_DATA;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// SGF file to be loaded from
    #[arg(short, long)]
    load: Option<String>,

    /// SGF file to be saved to
    #[arg(short, long)]
    save: Option<String>,

    /// Random seed
    #[arg(long)]
    seed: Option<String>,
}

const FC_LEN: usize = 2 /* map move */ + 1 /* set character*/ + MAX_STEPS + DOCTOR_MARKER as usize;

fn encode(g: &Game, a: &Action) -> Array1<u8> {
    // 1. BOARD_DATA
    // Check the README/HACKING for why it is 8
    let mut e = Array::from_shape_fn((SIZE as usize, SIZE as usize, 8 as usize), |(_, _, _)| {
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
            e[[c.x as usize, c.y as usize, 2 /*Doctor Hero*/]] = 1;
        } else {
            e[[c.x as usize, c.y as usize, 3 /*Plague Hero*/]] = 1;
        }
    }

    // 2. MAP_DATA
    // 2 for the two sides
    let mut m = Array::from_shape_fn(
        (MAP_SIZE as usize, MAP_SIZE as usize, 2 as usize),
        |(_, _, _)| 0 as u8,
    );

    for (camp, mc) in g.map.iter() {
        let c = *mc;
        if *camp == Camp::Doctor {
            m[[
                (c.x + MAP_OFFSET.x) as usize,
                (c.y + MAP_OFFSET.y) as usize,
                0, /* Doctor Marker */
            ]] = 1;
        } else {
            m[[
                (c.x + MAP_OFFSET.x) as usize,
                (c.y + MAP_OFFSET.y) as usize,
                1, /* Plague Marker */
            ]] = 1;
        }
    }

    // 3. TURN_DATA
    //
    let mut t = Array::from_shape_fn(
        (MAP_SIZE as usize, MAP_SIZE as usize, 1 as usize),
        |(_, _, _)| 0 as u8,
    );
    for (camp, mc) in g.map.iter() {
        let c = *mc;
        if *camp == g.turn {
            t[[
                (c.x + MAP_OFFSET.x) as usize,
                (c.y + MAP_OFFSET.y) as usize,
                0,
            ]] = 1;
        }
    }

    // 4. FLOW_DATA
    //
    let mut fm = Array::from_shape_fn(
        (MAP_SIZE as usize, MAP_SIZE as usize, 2 as usize),
        |(_, _, _)| 0 as u8,
    );
    let mut fe = Array::from_shape_fn((SIZE as usize, SIZE as usize, 11 as usize), |(_, _, _)| {
        0 as u8
    });

    if a.action_phase > ActionPhase::SetMap {
        let c = match a.map {
            None => {
                // Ix00?
                *g.map.get(&g.turn).unwrap()
            }
            Some(x) => {
                x
            }
        };
        fm[[
            (c.x + MAP_OFFSET.x) as usize,
            (c.y + MAP_OFFSET.y) as usize,
            0,
        ]] = 1;
    }
    if a.action_phase > ActionPhase::Lockdown {
        // Thanks to the fact that Coord::lockdown() doesn't require the
        // operation a doctor-limited one, we can always duplicate the
        // opponent's map position here.
        let mut c = *g.map.get(&g.opposite(g.turn)).unwrap();
        c = c.lockdown(a.lockdown);
        fm[[
            (c.x + MAP_OFFSET.x) as usize,
            (c.y + MAP_OFFSET.y) as usize,
            1,
        ]] = 1;
    }
    if a.action_phase > ActionPhase::SetCharacter {
        // a.trajectory contains at most 6 coordinates.
        // 1 for the character's position before the action,
        // 5 for the possible maximum steps.
        for i in 0..6 {
            let index = if i < a.trajectory.len() {
                i
            } else if i >= a.trajectory.len() && i <= a.steps {
                // During the whole ActionPhase::BoardMove phase,
                // this break will leave the sub-step empty and thus (theoretically)
                // deliver the message "do this sub-step".
                // For example, if a.steps == 2, then a.trajectory will
                // eventually grow to contain 3 elements, so when
                // 0 <= i < a.trajectory.len() <= 3, the first block takes;
                // as i goes to 3, it can no longer take this block.
                break;
            } else {
                a.trajectory.len() - 1
            };
            fe[[
                a.trajectory[index].x as usize,
                a.trajectory[index].y as usize,
                i, /* fe starts fresh */
            ]] = 1;
        }
    }
    if a.action_phase >= ActionPhase::SetMarkers {
        // a.markers contains at most 5 coordinates.
        for i in 0..5 {
            if i >= a.markers.len() {
                break;
            };
            fe[[
                a.markers[i].x as usize,
                a.markers[i].y as usize,
                i + 6, /* offset by the trajectory: 1 + 5 */
            ]] += 1;
        }
    }

    // Some simple checks
    #[cfg(debug_assertions)]
    {
        let mm = m.clone();
        let mut count = 0;
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..2 {
                    if mm[[i, j, k]] > 0 {
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(2, count);
    }

    // wrap it up
    let ret = e
        .into_shape((BOARD_DATA,))
        .unwrap()
        .into_iter()
        .chain(m.into_shape((MAP_DATA,)).unwrap().into_iter())
        .chain(t.into_shape((TURN_DATA,)).unwrap().into_iter())
        .chain(fm.into_shape((FLOW_MAP_DATA,)).unwrap().into_iter())
        .chain(fe.into_shape((FLOW_ENV_DATA,)).unwrap().into_iter())
        .collect::<Array1<_>>();

    assert_eq!(ret.len(), DATA_UNIT - CODE_DATA);
    ret
}

const MIN_MAP_CODE: u8 = 100;
const MAX_ENV_CODE: u8 = 36;
const QUERY_CODE: u8 = 255;

trait ActionCoord {
    fn to_coord(&self) -> Coord;
}

impl ActionCoord for u8 {
    fn to_coord(&self) -> Coord {
        if *self >= MIN_MAP_CODE {
            let cv = *self - MIN_MAP_CODE;
            let cvx = cv / MAP_SIZE as u8;
            let cvy = cv % MAP_SIZE as u8;
            Coord::new(cvx as i32 - 2, cvy as i32 - 2)
        } else {
            let cv = *self;
            let cvx = cv / SIZE as u8;
            let cvy = cv % SIZE as u8;
            Coord::new(cvx as i32, cvy as i32)
        }
    }
}

trait EncodeCoord {
    fn to_map_encode(&self) -> u8;
    fn to_env_encode(&self) -> u8;
}

impl EncodeCoord for Coord {
    fn to_map_encode(&self) -> u8 {
        let ms: i32 = (MAP_SIZE as i32).try_into().unwrap();
        let base: i32 = (MIN_MAP_CODE as i32).try_into().unwrap();
        (base + ((self.x + 2) * ms + (self.y + 2)))
            .try_into()
            .unwrap()
    }
    fn to_env_encode(&self) -> u8 {
        ((self.x) * SIZE + (self.y)).try_into().unwrap()
    }
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

fn handle_client<T: Read + ReaderExtra + Write + WriterExtra>(
    stream: &mut T,
    g: &mut Game,
) -> bool {
    let mut buffer = [0; 1]; // to read the 1-byte action from agent
    fn get_action<T>(stream: &mut T, buffer: &mut [u8]) -> bool
    where
        T: std::io::Read,
    {
        match stream.read(buffer) {
            Ok(0) => {
                return false;
            }
            Err(x) => {
                #[cfg(debug_assertions)]
                {
                    println!("{}", x);
                }
                return false;
            }
            _ => {
                #[cfg(debug_assertions)]
                {
                    println!("{:?}", buffer[0]);
                }
                return true;
            }
        }
    }

    let ea = Action::new();
    let mut ec: [u8; FC_LEN] = [0; FC_LEN];
    ec[0 /* set map */] = 1;
    let mut s = "Ix03";
    let ss = "Wx00";
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
                let mut am = ActionMonitor::new();
                let mut fc: [u8; FC_LEN] = [0; FC_LEN];

                // Check tree.rs:to_action() function for the following
                // big block of state machine. s for status code in the spec.

                // Add the map move first
                assert_eq!(am.action.action_phase, ActionPhase::SetMap);
                fc[0 /* set map */] = 1;
                'set_map: loop {
                    if !get_action(stream, &mut buffer) {
                        return false;
                    }
                    if buffer[0] < MIN_MAP_CODE {
                        s = "Ex26";
                    } else if buffer[0] == QUERY_CODE {
                        if false == stream.return_query(g, &am.action) {
                            return false;
                        }
                        continue 'set_map;
                    } else {
                        let c = (buffer[0] as u8).to_coord();
                        match am.action.add_map_step(g, c) {
                            Err(e) => {
                                s = e;
                            }
                            Ok(o) => {
                                s = o;
                            }
                        }
                    }
                    if stream.update_agent(g, &am.action, &fc, &s) == false {
                        return false;
                    } else {
                        if s.as_bytes()[0] == b'E' {
                            continue 'set_map;
                        } else if s == "Ix00" {
                            /* Skip!? */
                            next(g, &am.action);
                            return true;
                        }
                        break;
                    }
                }

                // Optional for Doctor: lockdown?
                if am.action.action_phase == ActionPhase::Lockdown {
                    fc[0 /* set map */] = 0;
                    fc[1 /* lockdown */] = 1;
                    'lockdown: loop {
                        if !get_action(stream, &mut buffer) {
                            return false;
                        }
                        if buffer[0] < MIN_MAP_CODE {
                            s = "Ex26";
                        } else if buffer[0] == QUERY_CODE {
                            if false == stream.return_query(g, &am.action) {
                                return false;
                            }
                            continue 'lockdown;
                        } else {
                            let c = (buffer[0] as u8).to_coord();
                            match am.action.add_lockdown_by_coord(g, c) {
                                Err(e) => {
                                    s = e;
                                }
                                Ok(o) => {
                                    s = o;
                                }
                            }
                        }
                        if stream.update_agent(g, &am.action, &fc, &s) == false {
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
                assert_eq!(am.action.action_phase, ActionPhase::SetCharacter);
                fc[0 /* set map */] = 0;
                fc[1 /* lockdown */] = 0;
                for i in 0..=am.action.steps {
                    fc[2 /* set character */ + i] = 1;
                }
                'set_character: loop {
                    if !get_action(stream, &mut buffer) {
                        return false;
                    }
                    if buffer[0] == QUERY_CODE {
                        if false == stream.return_query(g, &am.action) {
                            return false;
                        }
                        continue 'set_character;
                    } else if buffer[0] > MAX_ENV_CODE {
                        s = "Ex27";
                    } else {
                        let c = (buffer[0] as u8).to_coord();
                        match am.action.add_character(g, c) {
                            Err(e) => {
                                s = e;
                            }
                            Ok(o) => {
                                s = o;
                            }
                        }
                    }
                    if stream.update_agent(g, &am.action, &fc, &s) == false {
                        return false;
                    } else {
                        if s.as_bytes()[0] == b'E' {
                            continue 'set_character;
                        }
                        break;
                    }
                }

                // Move the character on the board
                assert_eq!(am.action.action_phase, ActionPhase::BoardMove);
                fc[2 /* set character */] = 0;
                for i in 0..am.action.steps {
                    'board_move: loop {
                        if !get_action(stream, &mut buffer) {
                            return false;
                        }
                        if buffer[0] == QUERY_CODE {
                            if false == stream.return_query(g, &am.action) {
                                return false;
                            }
                            continue 'board_move;
                        } else if buffer[0] > MAX_ENV_CODE {
                            s = "Ex27";
                        } else {
                            let c = (buffer[0] as u8).to_coord();
                            match am.action.add_board_single_step(g, c) {
                                Err(e) => {
                                    s = e;
                                }
                                Ok(o) => {
                                    s = o;
                                }
                            }
                        }
                        if stream.update_agent(g, &am.action, &fc, &s) == false {
                            return false;
                        } else {
                            if s.as_bytes()[0] == b'E' {
                                continue 'board_move;
                            } else if s == "Ix02" {
                                next(g, &am.action);
                                return true;
                            }
                            break;
                        }
                    }
                    fc[3 /* board step */ + i] = 0;
                }

                // Set the markers
                assert!(fc.iter().all(|&x| x == 0));
                let mut i = 8;
                for i in 0..DOCTOR_MARKER as usize {
                    fc[8 /* set marker */ + i] = 1;
                }
                if g.turn == Camp::Plague {
                    fc[12 /* final one: Plague has only 4 markers */] = 0;
                }
                'set_marker: loop {
                    assert_eq!(am.action.action_phase, ActionPhase::SetMarkers);
                    while i < FC_LEN {
                        if fc[i] == 0 {
                            continue;
                        }
                        if !get_action(stream, &mut buffer) {
                            return false;
                        }
                        if buffer[0] == QUERY_CODE {
                            if false == stream.return_query(g, &am.action) {
                                return false;
                            }
                            continue 'set_marker;
                        } else if buffer[0] > MAX_ENV_CODE {
                            s = "Ex27";
                        } else {
                            let c = (buffer[0] as u8).to_coord();
                            match am.action.add_single_marker(g, c) {
                                Err(e) => {
                                    s = e;
                                }
                                Ok(o) => {
                                    s = o;
                                }
                            }
                        }
                        if stream.update_agent(g, &am.action, &fc, &s) == false {
                            return false;
                        } else {
                            if s.as_bytes()[0] == b'E' {
                                continue 'set_marker;
                            } else if s == "Ix02" {
                                break 'set_marker;
                            }
                            break;
                        }
                    }
                    fc[i] = 0;
                    i = i + 1;
                }
                // commit the action to the game
                assert_eq!(am.action.action_phase, ActionPhase::Done);
                //assert!(fc.iter().all(|&x| x == 0));
                // Holy shit, the order of the following two lines
                // is tricky. You shouldn't commit it first and then
                // try to interpret the Action with the new Game
                // status.
                next(g, &am.action);
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

fn next(g: &mut Game, a: &Action) {
    g.append_history_with_new_tree(&a.to_sgf_string(g));
    g.commit_action(&a);
    g.next();
}

trait WriterExtra {
    fn update_agent(&mut self, g: &Game, a: &Action, fc: &[u8; FC_LEN], s: &'static str) -> bool;
    fn return_query(&mut self, g: &Game, a: &Action) -> bool;
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
        panic!("The game is either not ready or finished");
    }

    let (w, b) = network_setup()?;
    let mut s: [TcpStream; 2] = [w, b];

    let ea = Action::new();
    let ec: [u8; FC_LEN] = [0; FC_LEN];
    while let Phase::Main(x) = g.phase {
        let turn: usize = x.try_into().unwrap();
        if !handle_client(&mut s[turn % 2], &mut g) {
            s[turn % 2].update_agent(&g, &ea, &ec, &"Ix06");
            s[1 - turn % 2].update_agent(&g, &ea, &ec, &"Ix06");
            drop(s);
            break;
        }
        if g.is_ended() {
            s[turn % 2].update_agent(&g, &ea, &ec, &"Ix04");
            s[1 - turn % 2].update_agent(&g, &ea, &ec, &"Ix05");
            break;
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

impl<T: Write> WriterExtra for T {
    fn update_agent(&mut self, g: &Game, a: &Action, _fc: &[u8; FC_LEN], s: &'static str) -> bool {
        let encoded = encode(g, &a);
        let enc = encoded.as_slice().unwrap();
        let sb = s.as_bytes();

        let response = [&sb, &enc[..]].concat();
        assert!(response.len() == DATA_UNIT);
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
    fn return_query(&mut self, g: &Game, a: &Action) -> bool {
        let mut response: Vec<u8> = Vec::new();
        match a.action_phase {
            ActionPhase::SetMap => {
                let mut coord_candidate: Vec<Coord> = Vec::new();
                for i in -MAP_OFFSET.x + 1..MAP_OFFSET.x {
                    for j in -MAP_OFFSET.y..=MAP_OFFSET.y {
                        coord_candidate.push(Coord::new(i, j));
                    }
                }
                for j in -MAP_OFFSET.y + 1..MAP_OFFSET.y {
                    coord_candidate.push(Coord::new(-MAP_OFFSET.x, j));
                    coord_candidate.push(Coord::new(MAP_OFFSET.x, j));
                }
                let candidate = coord_candidate.clone();

                for &cc in candidate.iter() {
                    let mut ea = Action::new();
                    let s = ea.add_map_step(g, cc);
                    match s {
                        Ok(_) => {}
                        Err(_) => {
                            coord_candidate.retain(|&e| e != cc);
                            continue;
                        }
                    }
                }

                let len = coord_candidate.len() + 5;
                response = vec![0; len];
                response[0] = coord_candidate.len() as u8;
                for j in 1..coord_candidate.len() + 1 {
                    response[j] = coord_candidate[j - 1].to_map_encode();
                }
            }
            ActionPhase::Lockdown => {
                let h = <Vec<Candidate> as Clone>::clone(&a.candidate)
                    .into_iter()
                    .map(|c| c.lockdown)
                    .collect::<HashSet<_>>();

                let len = h.len() + 5;
                response = vec![0; len];
                response[0] = h.len() as u8;

                let cp = *g.map.get(&Camp::Plague).unwrap();
                let mut j = 1;
                for ld in h.iter() {
                    response[j] = cp.lockdown(*ld).to_map_encode();
                    j = j + 1;
                }
            }
            ActionPhase::SetCharacter => {
                let h = <Vec<Candidate> as Clone>::clone(&a.candidate)
                    .into_iter()
                    .map(|c| c.character)
                    .collect::<HashSet<_>>();

                let len = h.len() + 5;
                response = vec![0; len];
                response[0] = h.len() as u8;

                let mut j = 1;
                for c in h.iter() {
                    response[j] = c.to_env_encode();
                    j = j + 1;
                }
            }
            ActionPhase::BoardMove => {
                let trajectory_index = a.trajectory.len();
                let h = <Vec<Candidate> as Clone>::clone(&a.candidate)
                    .into_iter()
                    .map(|c| c.trajectory[trajectory_index])
                    .collect::<HashSet<_>>();

                let len = h.len() + 5;
                response = vec![0; len];
                response[0] = h.len() as u8;

                let mut j = 1;
                for c in h.iter() {
                    response[j] = c.to_env_encode();
                    j = j + 1;
                }
            }
            ActionPhase::SetMarkers => {
                let mut marker_candidate: Vec<Coord> = Vec::new();
                for ms in a.marker_slot.iter() {
                    let mut dummy_action = a.clone();
                    match dummy_action.add_single_marker_trial(g, ms.0, true) {
                        Ok(_) => {
                            marker_candidate.push(ms.0);
                        }
                        _ => {}
                    }
                }
                let len = marker_candidate.len() + 5;
                response = vec![0; len];
                response[0] = marker_candidate.len() as u8;
                let mut j = 1;
                for c in marker_candidate.iter() {
                    response[j] = c.to_env_encode();
                    j = j + 1;
                }
            }
            _ => {
                return false;
            }
        }
        let len = response.len();
        response[len - 4] = 'W' as u8;
        response[len - 3] = 'x' as u8;
        response[len - 2] = '0' as u8;
        response[len - 1] = '0' as u8;
        let res: &[u8] = &response;
        match self.write(&res) {
            Err(x) => {
                println!("Write failed: {}", x);
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
    fn test_handle_client_with_cursor1() {
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

        const LEN: usize = DATA_UNIT + (1 + DATA_UNIT) * 10;
        let mut buf_origin: [u8; LEN] = [0; LEN];
        let buf = &mut buf_origin[..];
        let s1 = ";W[ii][hh][aa][ab][bb][ab][aa][ab][aa][ab]";
        buf[DATA_UNIT] = "ii".to_map().to_map_encode();
        buf[(DATA_UNIT + 1) * 2 - 1] = "hh".to_map().to_map_encode();
        buf[(DATA_UNIT + 1) * 3 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 4 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 5 - 1] = "bb".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 6 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 7 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 8 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 9 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 10 - 1] = "ab".to_env().to_env_encode();
        let mut fake_stream = Cursor::new(buf);
        assert!(handle_client(&mut fake_stream, &mut g) == true);
        let mut buffer = String::new();
        g.history.borrow().to_string(&mut buffer);
        assert_eq!(buffer, s1);
        let buf_after = fake_stream.get_ref();
        let env_offset = CODE_DATA + BOARD_DATA;
        let turn_offset = CODE_DATA + BOARD_DATA + MAP_DATA;
        let flow_map_offset = CODE_DATA + BOARD_DATA + MAP_DATA + TURN_DATA;
        let flow_env_offset = CODE_DATA + BOARD_DATA + MAP_DATA + TURN_DATA + FLOW_MAP_DATA;

        // when it is SetMap...
        // Human
        assert_eq!(
            buf_after[CODE_DATA + 5 * 6 * 8 + 5 * 8 + 1], /* "ff" */
            1
        );
        // Underworld
        assert_eq!(
            buf_after[CODE_DATA + 5 * 6 * 8 + 3 * 8 + 0], /* "fd" */
            1
        );
        // Doctor's token
        assert_eq!(buf_after[env_offset + 2 * 5 * 2 + 3 * 2] /* "ij" */, 1);
        // Plague's token
        assert_eq!(
            buf_after[env_offset + 3 * 5 * 2 + 3 * 2 + 1], /* "jj" */
            1
        );
        // Doctor's turn: duplicate the map for the doctor only
        assert_eq!(
            buf_after[turn_offset + 3 * 5 * 1 + 3 * 1], /* "jj" */
            0
        );
        assert_eq!(
            buf_after[turn_offset + 2 * 5 * 1 + 3 * 1], /* "ij" */
            1
        );

        // when it is Lockdown...
        let mut base_offset = DATA_UNIT + 1;
        // Yes, after 0.6, the game status remain the same
        // So this should be 1
        assert_eq!(
            buf_after[base_offset + env_offset + 2 * 5 * 2 + 3 * 2], /* "ij" */
            1
        );
        // it moves to "ii" in SetMap
        assert_eq!(
            buf_after[base_offset + flow_map_offset + 2 * 5 * 2 + 2 * 2], /* "ii" */
            1
        );

        // when it is SetCharacter...
        base_offset = base_offset + (DATA_UNIT + 1);
        assert_eq!(
            buf_after[base_offset + flow_map_offset + 3 * 5 * 2 + 3 * 2 + 1], /* "jj" */
            0
        );
        assert_eq!(
            buf_after[base_offset + flow_map_offset + 1 * 5 * 2 + 1 * 2 + 1], /* "hh" */
            1
        );
        assert_eq!(
            buf_after[base_offset + CODE_DATA + 0 * 6 * 11 + 0 * 11 + 0], /* "aa" */
            0
        );

        // when it is BoardMove...
        base_offset = base_offset + (DATA_UNIT + 1);
        assert_eq!(
            buf_after[base_offset + flow_map_offset + 1 * 5 * 2 + 1 * 2 + 1], /* "hh" */
            1
        );
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 0 * 11 + 0], /* "aa" */
            1
        );

        // when it is BoardMove... after the first move
        base_offset = base_offset + (DATA_UNIT + 1);
        assert_eq!(
            buf_after[base_offset + flow_map_offset + 1 * 5 * 2 + 1 * 2 + 1], /* "hh" */
            1
        );
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 0 * 11 + 0], /* "aa" */
            1
        );
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 1 * 11 + 1], /* "ab" */
            1
        );
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 1 * 6 * 11 + 1 * 11 + 2], /* "bb" */
            0
        );

        // when it is SetMarkers... after the second BoardMove
        base_offset = base_offset + (DATA_UNIT + 1);
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 0 * 11 + 0], /* "aa" */
            1
        );
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 1 * 11 + 1], /* "ab" */
            1
        );
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 1 * 6 * 11 + 1 * 11 + 2], /* "bb" */
            1
        );
        // the rest repeat the destination
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 1 * 6 * 11 + 1 * 11 + 3], /* "bb" */
            1
        );
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 1 * 6 * 11 + 1 * 11 + 5], /* "bb" */
            1
        );
        // the first marker position is yet to be filled ("ab")
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 1 * 11 + 6], /* "ab" */
            0
        );

        // when it is SetMarkers... the 2nd
        base_offset = base_offset + (DATA_UNIT + 1);
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 1 * 11 + 6], /* "ab" */
            1
        );
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 0 * 11 + 7], /* "aa" */
            0
        );

        // when it is SetMarkers... the 3rd
        base_offset = base_offset + (DATA_UNIT + 1);
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 0 * 11 + 7], /* "aa" */
            1
        );
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 1 * 11 + 8], /* "ab" */
            0
        );

        // when it is SetMarkers... the 4th
        base_offset = base_offset + (DATA_UNIT + 1);
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 1 * 11 + 8], /* "ab" */
            1
        );
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 0 * 11 + 9], /* "aa" */
            0
        );

        // when it is SetMarkers... the 5th
        base_offset = base_offset + (DATA_UNIT + 1);
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 0 * 11 + 9], /* "aa" */
            1
        );
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 1 * 11 + 10], /* "ab" */
            0
        );

        // Done
        base_offset = base_offset + (DATA_UNIT + 1);
        assert_eq!(
            buf_after[base_offset + flow_env_offset + 0 * 6 * 11 + 1 * 11 + 10], /* "ab" */
            1
        );
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

        const LEN: usize = DATA_UNIT + (1 + DATA_UNIT) * 11;
        let mut buf_origin: [u8; LEN] = [0; LEN];
        let buf = &mut buf_origin[..];
        // in real correct SGF file, of course we cannot assign "hi" as the
        // lockdown position, but this is for a demo
        let s1 = ";W[ii][hi][hh][aa][ab][bb][ab][aa][ab][aa][ab]";
        buf[DATA_UNIT] = "ii".to_map().to_map_encode();
        buf[(DATA_UNIT + 1) * 2 - 1] = "hi".to_map().to_map_encode();
        buf[(DATA_UNIT + 1) * 3 - 1] = "hh".to_map().to_map_encode();
        buf[(DATA_UNIT + 1) * 4 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 5 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 6 - 1] = "bb".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 7 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 8 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 9 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 10 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 11 - 1] = "ab".to_env().to_env_encode();
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

        const LEN: usize = DATA_UNIT + (1 + DATA_UNIT) * 11;
        let mut buf_origin: [u8; LEN] = [0; LEN];
        let buf = &mut buf_origin[..];
        // It's difficult to reason why it was ([bf] was [af]) working.
        // [af] happended to be the other character. Before the candidate
        // mechanism, there is no check that [hh] lockdown with [af]
        // character cannot work together. BUT! Even now, there is no
        // way to do the check, so theoretically it should still fail.
        //
        let s1 = ";W[ii][hh][bf][aa][ab][bb][ab][aa][ab][aa][ab]";
        buf[DATA_UNIT] = "ii".to_map().to_map_encode();
        buf[(DATA_UNIT + 1) * 2 - 1] = "hh".to_map().to_map_encode();
        buf[(DATA_UNIT + 1) * 3 - 1] = "bf".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 4 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 5 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 6 - 1] = "bb".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 7 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 8 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 9 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 10 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 11 - 1] = "ab".to_env().to_env_encode();
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

        const LEN: usize = DATA_UNIT + (1 + DATA_UNIT) * 11;
        let s2 = ";W[ii][hh][af][aa][ab][bb][ab][aa][ab][aa][ab]";
        let mut buf_origin2: [u8; LEN] = [0; LEN];
        let buf2 = &mut buf_origin2[..];
        buf2[DATA_UNIT] = "ii".to_map().to_map_encode();
        buf2[(DATA_UNIT + 1) * 2 - 1] = "hh".to_map().to_map_encode();
        buf2[(DATA_UNIT + 1) * 3 - 1] = "af".to_env().to_env_encode();
        buf2[(DATA_UNIT + 1) * 4 - 1] = "aa".to_env().to_env_encode();
        buf2[(DATA_UNIT + 1) * 5 - 1] = "ab".to_env().to_env_encode();
        buf2[(DATA_UNIT + 1) * 6 - 1] = "bb".to_env().to_env_encode();
        buf2[(DATA_UNIT + 1) * 7 - 1] = "ab".to_env().to_env_encode();
        buf2[(DATA_UNIT + 1) * 8 - 1] = "aa".to_env().to_env_encode();
        buf2[(DATA_UNIT + 1) * 9 - 1] = "ab".to_env().to_env_encode();
        buf2[(DATA_UNIT + 1) * 10 - 1] = "aa".to_env().to_env_encode();
        buf2[(DATA_UNIT + 1) * 11 - 1] = "ab".to_env().to_env_encode();
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

        const LEN: usize = DATA_UNIT + (1 + DATA_UNIT) * 12;
        let mut buf_origin: [u8; LEN] = [0; LEN];
        let buf = &mut buf_origin[..];
        let s1 = ";W[ii][hh][aa][dd][ab][ac][bb][ab][aa][ab][aa][ab]";
        buf[DATA_UNIT] = "ii".to_map().to_map_encode();
        buf[(DATA_UNIT + 1) * 2 - 1] = "hh".to_map().to_map_encode();
        buf[(DATA_UNIT + 1) * 3 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 4 - 1] = "dd".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 5 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 6 - 1] = "ac".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 7 - 1] = "bb".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 8 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 9 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 10 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 11 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 12 - 1] = "ab".to_env().to_env_encode();
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

        const LEN: usize = DATA_UNIT + (1 + DATA_UNIT) * 15;
        let mut buf_origin: [u8; LEN] = [0; LEN];
        let buf = &mut buf_origin[..];
        // in real correct SGF file, of course we cannot assign "hi" as the
        // lockdown position, but this is for a demo
        let s1 = ";W[ii][hh][aa][ab][bb][aa][aa][aa][aa][aa][ab][aa][aa][aa][aa]";
        buf[DATA_UNIT] = "ii".to_map().to_map_encode();
        buf[(DATA_UNIT + 1) * 2 - 1] = "hh".to_map().to_map_encode();
        buf[(DATA_UNIT + 1) * 3 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 4 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 5 - 1] = "bb".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 6 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 7 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 8 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 9 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 10 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 11 - 1] = "ab".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 12 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 13 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 14 - 1] = "aa".to_env().to_env_encode();
        buf[(DATA_UNIT + 1) * 15 - 1] = "aa".to_env().to_env_encode();
        let mut fake_stream = Cursor::new(buf);
        assert!(handle_client(&mut fake_stream, &mut g) == true);
        let mut buffer = String::new();
        g.history.borrow().to_string(&mut buffer);
        assert_eq!(buffer, s1.replace("[aa][aa][aa][aa][aa]", ""));
    }

    #[test]
    fn test_query1() {
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
        let g = Game::init(Some(t));
        let mut ea = Action::new();
        ea.action_phase = ActionPhase::Done;

        const LEN: usize = DATA_UNIT + (1 + DATA_UNIT) * 15;
        let mut buf_origin: [u8; LEN] = [0; LEN];
        let buf = &mut buf_origin[..];
        let mut fake_stream = Cursor::new(buf);
        assert_eq!(false, fake_stream.return_query(&g, &ea));
    }

    #[test]
    fn test_query2() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cd][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][ce][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            ;C[Setup3]AW[ij]
            ;B[jj][ac][cc][ac][ac][ac][ac]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));
        let mut a = Action::new();

        // SetMap
        const LEN: usize = 1 + 2 + 4;
        let mut buf_origin: [u8; LEN] = [0; LEN];
        let buf = &mut buf_origin[..];
        let mut fake_stream = Cursor::new(buf);
        assert_eq!(true, fake_stream.return_query(&g, &a));
        let buf_after = fake_stream.get_ref();
        assert_eq!(3 as u8, buf_after[0]);
        // In this case, it will be either 112(ii), 117(ji) or 113(ij, skip).
        let _ = a.add_map_step(&g, "ii".to_map());

        // Lockdown
        const LEN2: usize = 1 + 2 + 4;
        let mut buf_origin2: [u8; LEN2] = [0; LEN2];
        let buf2 = &mut buf_origin2[..];
        fake_stream = Cursor::new(buf2);
        assert_eq!(true, fake_stream.return_query(&g, &a));
        let buf_after2 = fake_stream.get_ref();
        assert_eq!(2 as u8, buf_after2[0]);
        // it will be either 106(hh) or 108(hj). we choose 106 here.
        assert_eq!(Ok("Ix01"), a.add_lockdown_by_coord(&g, "hh".to_map()));

        // SetCharacter
        const LEN3: usize = 1 + 1 + 4;
        let mut buf_origin3: [u8; LEN3] = [0; LEN3];
        let buf3 = &mut buf_origin3[..];
        fake_stream = Cursor::new(buf3);
        assert_eq!(true, fake_stream.return_query(&g, &a));
        let buf_after3 = fake_stream.get_ref();
        assert_eq!(1 as u8, buf_after3[0]);
        // it will be 0(aa)
        assert_eq!(Ok("Ix01"), a.add_character(&g, "aa".to_env()));

        // BoardMove1
        const LEN4: usize = 1 + 2 + 4;
        let mut buf_origin4: [u8; LEN4] = [0; LEN4];
        let buf4 = &mut buf_origin4[..];
        fake_stream = Cursor::new(buf4);
        assert_eq!(true, fake_stream.return_query(&g, &a));
        let buf_after4 = fake_stream.get_ref();
        assert_eq!(2 as u8, buf_after4[0]);
        // it will be 12(ca) and 1(ab)
        assert_eq!(Ok("Ix01"), a.add_board_single_step(&g, "ca".to_env()));
        // BoardMove2
        const LEN5: usize = 1 + 1 + 4;
        let mut buf_origin5: [u8; LEN5] = [0; LEN5];
        let buf5 = &mut buf_origin5[..];
        fake_stream = Cursor::new(buf5);
        assert_eq!(true, fake_stream.return_query(&g, &a));
        let buf_after5 = fake_stream.get_ref();
        assert_eq!(1 as u8, buf_after5[0]);
        // it will be 16(ce)
        assert_eq!(Ok("Ix01"), a.add_board_single_step(&g, "ce".to_env()));

        // SetMarker
        const LEN6: usize = 1 + 2 + 4;
        let mut buf_origin6: [u8; LEN6] = [0; LEN6];
        let buf6 = &mut buf_origin6[..];
        fake_stream = Cursor::new(buf6);
        assert_eq!(true, fake_stream.return_query(&g, &a));
        let buf_after6 = fake_stream.get_ref();
        assert_eq!(2 as u8, buf_after6[0]);
        // it will be 12(ca) and 0(aa)
        g.stuff
            .insert("aa".to_env(), (Camp::Plague, Stuff::Marker(1)));
        assert_eq!(Err("Ex24"), a.add_single_marker(&g, "ca".to_env()));
        assert_eq!(Ok("Ix01"), a.add_single_marker(&g, "aa".to_env()));
        assert_eq!(Ok("Ix01"), a.add_single_marker(&g, "aa".to_env()));
        assert_eq!(Ok("Ix01"), a.add_single_marker(&g, "ca".to_env()));
        assert_eq!(Ok("Ix01"), a.add_single_marker(&g, "ca".to_env()));
        assert_eq!(Ok("Ix02"), a.add_single_marker(&g, "aa".to_env()));
    }
}
