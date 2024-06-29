use clap::Parser;
use ndarray::{Array, Array1};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use pathogen_engine::core::action::Action;
use pathogen_engine::core::action::ActionPhase;
use pathogen_engine::core::action::Candidate;
use pathogen_engine::core::grid_coord::{Coord, MAP_OFFSET};
use pathogen_engine::core::tree::TreeNode;
use pathogen_engine::core::*;

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
            Some(x) => x,
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
const SAVE_CODE: u8 = 254;
const RETURN_CODE: u8 = 253;
const CLEAR_CODE: u8 = 252;
const MIN_SPECIAL_CODE: u8 = CLEAR_CODE;

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

fn do_action<F>(g: &Game, a: &mut Action, func: F, input: &mut [u8], is_map: bool) -> &'static str
where
    F: Fn(&mut Action, &Game, Coord) -> Result<&'static str, &'static str>,
{
    if !is_map && input[0] > MAX_ENV_CODE {
        "Ex27"
    } else if is_map && input[0] < MIN_MAP_CODE {
        "Ex26"
    } else {
        let c = (input[0] as u8).to_coord();
        match func(a, g, c) {
            Err(e) => e,
            Ok(o) => o,
        }
    }
}

const MAX_STEPS: usize = 5;
fn common_loop<F, T: Read + ReaderExtra + Write + WriterExtra>(
    stream: &mut T,
    g: &mut Game,
    a: &mut Action,
    input: &mut [u8],
    func: F,
    outer_bound: usize,
    ap: ActionPhase,
) -> bool
where
    // Check action.rs add_* calls
    F: Fn(&mut Action, &Game, Coord) -> Result<&'static str, &'static str>,
{
    // Argurably, this is only necessary for the BoardMove phase.
    // Others are either redundent (SetMarkers) or one-shot.
    // But this helps that I don't have to consider the indirection among
    // different phases.
    for _ in 0..outer_bound {
        // Unconditioned loop that was mainly designed to eliminate the
        // trial-and-error behavior from the agents.
        loop {
            assert_eq!(a.action_phase, ap);
            if !get_action(stream, input) {
                return false;
            }
            if input[0] >= MIN_SPECIAL_CODE {
                match input[0] {
                    SAVE_CODE => {
                        g.save();
                    }
                    RETURN_CODE => {
                        g.reset(false);
                    }
                    CLEAR_CODE => {
                        g.reset(true);
                    }
                    QUERY_CODE => {
                        // nothing special to be done because the `return_query` is
                        // shared by all special codes now.
                    }
                    _ => {}
                }
                if false == stream.return_query(g, a) {
                    return false;
                }
                continue;
            }

            let s = do_action(g, a, &func, input, ap <= ActionPhase::Lockdown);
            if stream.update_agent(g, a, &s) == false {
                return false;
            } else {
                if s.as_bytes()[0] == b'E' {
                    continue;
                } else {
                    if s == "Ix02" || s == "Ix00" {
                        return true;
                    }
                    break;
                }
            }
        }
    }
    return true;
}

fn set_marker_loop<T: Read + ReaderExtra + Write + WriterExtra>(
    stream: &mut T,
    g: &mut Game,
    a: &mut Action,
    input: &mut [u8],
) -> bool {
    common_loop(
        stream,
        g,
        a,
        input,
        Action::add_single_marker,
        MAX_STEPS,
        ActionPhase::SetMarkers,
    )
}

fn set_board_move_loop<T: Read + ReaderExtra + Write + WriterExtra>(
    stream: &mut T,
    g: &mut Game,
    a: &mut Action,
    input: &mut [u8],
) -> bool {
    common_loop(
        stream,
        g,
        a,
        input,
        Action::add_board_single_step,
        a.steps,
        ActionPhase::BoardMove,
    )
}

fn set_character_loop<T: Read + ReaderExtra + Write + WriterExtra>(
    stream: &mut T,
    g: &mut Game,
    a: &mut Action,
    input: &mut [u8],
) -> bool {
    common_loop(
        stream,
        g,
        a,
        input,
        Action::add_character,
        1,
        ActionPhase::SetCharacter,
    )
}

fn set_lockdown_loop<T: Read + ReaderExtra + Write + WriterExtra>(
    stream: &mut T,
    g: &mut Game,
    a: &mut Action,
    input: &mut [u8],
) -> bool {
    if a.action_phase != ActionPhase::Lockdown {
        return true;
    }
    common_loop(
        stream,
        g,
        a,
        input,
        Action::add_lockdown_by_coord,
        1,
        ActionPhase::Lockdown,
    )
}

fn set_map_loop<T: Read + ReaderExtra + Write + WriterExtra>(
    stream: &mut T,
    g: &mut Game,
    a: &mut Action,
    input: &mut [u8],
) -> bool {
    // Add the map move first
    common_loop(
        stream,
        g,
        a,
        input,
        Action::add_map_step,
        1,
        ActionPhase::SetMap,
    )
}

fn get_action<T: Read>(stream: &mut T, buffer: &mut [u8]) -> bool {
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

// With true, it means all communications are happy;
// with false, there was something wrong, and not about game logic.
fn handle_client<T: Read + ReaderExtra + Write + WriterExtra>(
    stream: &mut T,
    g: &mut Game,
) -> bool {
    let mut buffer = [0; 1]; // to read the 1-byte action from agent

    let ea = Action::new();
    let mut es = "Ix03";
    loop {
        // A state machine for RL support mode
        // This way, when an "Ix02"/"Ix00" is encounted,
        // this agent keeps occupying the server by staying
        // in this loop, but use a different initial code to
        // indicate the situation.
        if g.savepoint {
            if es == "Ix03" || es == "Ix08" {
                es = "Ix07";
            } else if es == "Ix07" {
                es = "Ix08";
            }
        }

        if stream.update_agent(g, &ea, &es) == false {
            return false;
        }
        match stream.peek(&mut buffer) {
            Ok(0) => {
                println!("Client disconnected.");
                return false;
            }
            Ok(_) => {
                let mut am = ActionMonitor::new();

                // Check tree.rs:to_action() function for the state machine
                // the suffix "_loop" is meant to be a reminder that these
                // are themselves states that accept unlimited failed trials,
                // well, mostly only for accumodating trial-and-error random agent.
                // XXX: but actually we can improve this...... all the pieces are
                //      ready now.
                loop {
                    if !match am.action.action_phase {
                        ActionPhase::SetMap => set_map_loop(stream, g, &mut am.action, &mut buffer),
                        ActionPhase::Lockdown => {
                            set_lockdown_loop(stream, g, &mut am.action, &mut buffer)
                        }
                        ActionPhase::SetCharacter => {
                            set_character_loop(stream, g, &mut am.action, &mut buffer)
                        }
                        ActionPhase::BoardMove => {
                            set_board_move_loop(stream, g, &mut am.action, &mut buffer)
                        }
                        ActionPhase::SetMarkers => {
                            set_marker_loop(stream, g, &mut am.action, &mut buffer)
                        }
                        ActionPhase::Done => {
                            break;
                        }
                    } {
                        return false;
                    }
                }

                // commit the action to the game
                next(g, &am.action);

                if g.savepoint {
                    continue;
                } else {
                    break;
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

fn next(g: &mut Game, a: &Action) {
    g.append_history_with_new_tree(&a.to_sgf_string(g));
    g.commit_action(&a);
    g.next();
}

trait WriterExtra {
    fn update_agent(&mut self, g: &Game, a: &Action, s: &'static str) -> bool;
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
    while let Phase::Main(x) = g.phase {
        let turn: usize = x.try_into().unwrap();
        if !handle_client(&mut s[turn % 2], &mut g) {
            s[turn % 2].update_agent(&g, &ea, &"Ix06");
            s[1 - turn % 2].update_agent(&g, &ea, &"Ix06");
            drop(s);
            break;
        }
        if g.is_ended() {
            s[turn % 2].update_agent(&g, &ea, &"Ix04");
            s[1 - turn % 2].update_agent(&g, &ea, &"Ix05");
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
    fn update_agent(&mut self, g: &Game, a: &Action, s: &'static str) -> bool {
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
        let mut response = match a.action_phase {
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
                let mut response = vec![0; len];
                response[0] = coord_candidate.len() as u8;
                for j in 1..coord_candidate.len() + 1 {
                    response[j] = coord_candidate[j - 1].to_map_encode();
                }
                response
            }
            ActionPhase::Lockdown => {
                let h = <Vec<Candidate> as Clone>::clone(&a.candidate)
                    .into_iter()
                    .map(|c| c.lockdown)
                    .collect::<BTreeSet<_>>();

                let len = h.len() + 5;
                let mut response = vec![0; len];
                response[0] = h.len() as u8;

                let cp = *g.map.get(&Camp::Plague).unwrap();
                let mut j = 1;
                for ld in h.iter() {
                    response[j] = cp.lockdown(*ld).to_map_encode();
                    j = j + 1;
                }
                response
            }
            ActionPhase::SetCharacter => {
                let h = <Vec<Candidate> as Clone>::clone(&a.candidate)
                    .into_iter()
                    .map(|c| c.character)
                    .collect::<BTreeSet<_>>();

                let len = h.len() + 5;
                let mut response = vec![0; len];
                response[0] = h.len() as u8;

                let mut j = 1;
                for c in h.iter() {
                    response[j] = c.to_env_encode();
                    j = j + 1;
                }
                response
            }
            ActionPhase::BoardMove => {
                let trajectory_index = a.trajectory.len();
                let h = <Vec<Candidate> as Clone>::clone(&a.candidate)
                    .into_iter()
                    .map(|c| c.trajectory[trajectory_index])
                    .collect::<BTreeSet<_>>();

                let len = h.len() + 5;
                let mut response = vec![0; len];
                response[0] = h.len() as u8;

                let mut j = 1;
                for c in h.iter() {
                    response[j] = c.to_env_encode();
                    j = j + 1;
                }
                response
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
                let mut response = vec![0; len];
                response[0] = marker_candidate.len() as u8;
                let mut j = 1;
                for c in marker_candidate.iter() {
                    response[j] = c.to_env_encode();
                    j = j + 1;
                }
                response
            }
            _ => {
                println!("Not an anticipated action phase");
                return false;
            }
        };
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

    #[test]
    fn test_save() {
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

        g.save();

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

        g.reset(true);
        assert_eq!(g.savepoint, false);
        assert_eq!(g.phase, Phase::Main(2));
    }

    #[test]
    fn test_simulated_turn() {
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

        const SAVE: usize = 1 + 1 + 2 + 4;
        const SIM: usize = DATA_UNIT;
        const LEN: usize = DATA_UNIT + SAVE + (1 + DATA_UNIT) * 10 + SIM + SAVE + (1 + DATA_UNIT);
        let mut buf_origin: [u8; LEN] = [0; LEN];
        let buf = &mut buf_origin[..];
        let _s1 = ";W[ii][hh][aa][ab][bb][ab][aa][ab][aa][ab]";
        // First we do a save here, right at the beginning of the Main(2)
        buf[DATA_UNIT] = SAVE_CODE;
        buf[SAVE + DATA_UNIT] = "ii".to_map().to_map_encode();
        buf[SAVE + (DATA_UNIT + 1) * 2 - 1] = "hh".to_map().to_map_encode();
        buf[SAVE + (DATA_UNIT + 1) * 3 - 1] = "aa".to_env().to_env_encode();
        buf[SAVE + (DATA_UNIT + 1) * 4 - 1] = "ab".to_env().to_env_encode();
        buf[SAVE + (DATA_UNIT + 1) * 5 - 1] = "bb".to_env().to_env_encode();
        buf[SAVE + (DATA_UNIT + 1) * 6 - 1] = "ab".to_env().to_env_encode();
        buf[SAVE + (DATA_UNIT + 1) * 7 - 1] = "aa".to_env().to_env_encode();
        buf[SAVE + (DATA_UNIT + 1) * 8 - 1] = "ab".to_env().to_env_encode();
        buf[SAVE + (DATA_UNIT + 1) * 9 - 1] = "aa".to_env().to_env_encode();
        buf[SAVE + (DATA_UNIT + 1) * 10 - 1] = "ab".to_env().to_env_encode();
        buf[SAVE + (DATA_UNIT + 1) * 11 - 1 + SIM] = CLEAR_CODE;
        buf[SAVE + (DATA_UNIT + 1) * 11 - 1 + SIM + SAVE] = "ij".to_map().to_map_encode();
        let mut fake_stream = Cursor::new(buf);
        assert!(handle_client(&mut fake_stream, &mut g) == true);

        let buf_after = fake_stream.get_ref();

        // Ix03: The start, sad that I have to check this
        assert_eq!(buf_after[0], 73);
        assert_eq!(buf_after[1], 120);
        assert_eq!(buf_after[2], 48);
        assert_eq!(buf_after[3], 51);

        // Wx00: returned by an save request
        assert_eq!(buf_after[DATA_UNIT], 254);
        assert_eq!(buf_after[DATA_UNIT + 1], 2);
        assert_eq!(buf_after[DATA_UNIT + SAVE - 4], 87);
        assert_eq!(buf_after[DATA_UNIT + SAVE - 3], 120);
        assert_eq!(buf_after[DATA_UNIT + SAVE - 2], 48);
        assert_eq!(buf_after[DATA_UNIT + SAVE - 1], 48);

        // Ix02: After the doctor's turn is over
        assert_eq!(buf_after[SAVE + (DATA_UNIT + 1) * 10], 73);
        assert_eq!(buf_after[SAVE + (DATA_UNIT + 1) * 10 + 1], 120);
        assert_eq!(buf_after[SAVE + (DATA_UNIT + 1) * 10 + 2], 48);
        assert_eq!(buf_after[SAVE + (DATA_UNIT + 1) * 10 + 3], 50);

        // Ix07: Begin a simulated Plague's turn
        assert_eq!(buf_after[SAVE + (DATA_UNIT + 1) * 11 - 1], 73);
        assert_eq!(buf_after[SAVE + (DATA_UNIT + 1) * 11], 120);
        assert_eq!(buf_after[SAVE + (DATA_UNIT + 1) * 11 + 1], 48);
        assert_eq!(buf_after[SAVE + (DATA_UNIT + 1) * 11 + 2], 55);

        // Wx00: returned by an clear request
        assert_eq!(
            buf_after[DATA_UNIT + SAVE + (DATA_UNIT + 1) * 10 + SIM],
            252
        );
        assert_eq!(
            buf_after[DATA_UNIT + SAVE + (DATA_UNIT + 1) * 10 + SIM + 1],
            2
        );
        assert_eq!(
            buf_after[DATA_UNIT + SAVE + (DATA_UNIT + 1) * 10 + SIM + SAVE - 4],
            87
        );
        assert_eq!(
            buf_after[DATA_UNIT + SAVE + (DATA_UNIT + 1) * 10 + SIM + SAVE - 3],
            120
        );
        assert_eq!(
            buf_after[DATA_UNIT + SAVE + (DATA_UNIT + 1) * 10 + SIM + SAVE - 2],
            48
        );
        assert_eq!(
            buf_after[DATA_UNIT + SAVE + (DATA_UNIT + 1) * 10 + SIM + SAVE - 1],
            48
        );

        // Ix00: return to Doctor's play and close
        assert_eq!(
            buf_after[DATA_UNIT + SAVE + (DATA_UNIT + 1) * 10 + SIM + SAVE + 1],
            73
        );
        assert_eq!(
            buf_after[DATA_UNIT + SAVE + (DATA_UNIT + 1) * 10 + SIM + SAVE + 2],
            120
        );
        assert_eq!(
            buf_after[DATA_UNIT + SAVE + (DATA_UNIT + 1) * 10 + SIM + SAVE + 3],
            48
        );
        assert_eq!(
            buf_after[DATA_UNIT + SAVE + (DATA_UNIT + 1) * 10 + SIM + SAVE + 4],
            48
        );
    }
}
