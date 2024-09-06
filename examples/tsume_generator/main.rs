use bytemuck::cast_slice;
use clap::Parser;
use ndarray::{Array, Array1};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fs::OpenOptions;
use std::fs::{create_dir_all, read_dir, DirEntry, File};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::rc::Rc;
use std::thread;
use std::time::Duration;

use pathogen_engine::core::action::Action;
use pathogen_engine::core::action::ActionPhase;
use pathogen_engine::core::action::Candidate;
use pathogen_engine::core::grid_coord::{Coord, MAP_OFFSET};
use pathogen_engine::core::tree::{Property, TreeNode};
use pathogen_engine::core::*;

const BOARD_DATA: usize = 288; /*8x6x6*/
const MAP_DATA: usize = 50; /*2x5x5*/
const TURN_DATA: usize = 25; /*1x5x5*/
const FLOW_MAP_DATA: usize = 50; /*2x5x5*/
const FLOW_ENV_DATA: usize = 396; /*11x6x6*/
const FLOW_DATA: usize = FLOW_MAP_DATA + FLOW_ENV_DATA; /*2x5x5+11x6x6*/
const TOTAL_POS: usize = 61; /*36+25*/
const DATASET_UNIT: usize = 1024;

const CODE_DATA: usize = 4;

const DATA_UNIT: usize = BOARD_DATA + MAP_DATA + TURN_DATA + FLOW_DATA + CODE_DATA;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The directory to a bunch of SGF files to be loaded from
    #[arg(short, long)]
    load_dir: Option<String>,

    /// The file name of which to be saved to
    #[arg(long)]
    dataset: Option<String>,

    /// The file that describes each dataset entry in the format:
    /// offset, winner, #moves
    #[arg(long)]
    desc: Option<String>,

    /// Random seed
    #[arg(long)]
    seed: Option<String>,

    /// Batch number: How many games are we exploiting?
    #[arg(long, default_value_t = 1)]
    batch: u32,
}

fn write_entry(args: &Args) -> Result<(), std::io::Error> {
    // write to the da
    let buffer: [u8; 4] = [0; 4];
    match &args.dataset {
        Some(dataset) => {
            let mut save_file = OpenOptions::new()
                .append(true) // Set the option to append data to the file
                .create(true) // Create the file if it doesn't exist
                .open(dataset)?; // Open the file
            save_file.write_all(&buffer)?;
        }
        None => {}
    }
    Ok(())
}

fn encode(g: &Game, a: &Action, vec: &mut Vec<f32>) {
    // 1. BOARD_DATA
    // Check the README/HACKING for why it is 8
    let mut e = Array::from_shape_fn((8 as usize, SIZE as usize, SIZE as usize), |(_, _, _)| {
        0 as f32
    });
    for i in 0..SIZE as usize {
        for j in 0..SIZE as usize {
            let c = Coord::new(i.try_into().unwrap(), j.try_into().unwrap());
            if g.env.get(&c).unwrap() == &World::Underworld {
                e[[0 /*Underworld*/, i, j]] = 1.0;
            } else {
                e[[1 /*Humanity*/, i, j]] = 1.0;
            }
            match g.stuff.get(&c) {
                None => {}
                Some((Camp::Doctor, Stuff::Colony)) => {
                    e[[4 /*Doctor Colony*/, i, j]] = 1.0;
                }
                Some((Camp::Plague, Stuff::Colony)) => {
                    e[[5 /*Plague Colony*/, i, j]] = 1.0;
                }
                Some((Camp::Doctor, Stuff::Marker(x))) => {
                    e[[6 /*Doctor Marker*/, i, j]] = *x as f32;
                }
                Some((Camp::Plague, Stuff::Marker(x))) => {
                    e[[7 /*Plague Marker*/, i, j]] = *x as f32;
                }
            }
        }
    }
    for ((_, camp), c) in g.character.iter() {
        if *camp == Camp::Doctor {
            e[[2 /*Doctor Hero*/, c.x as usize, c.y as usize]] = 1.0;
        } else {
            e[[3 /*Plague Hero*/, c.x as usize, c.y as usize]] = 1.0;
        }
    }

    // 2. MAP_DATA
    // 2 for the two sides
    let mut m = Array::from_shape_fn(
        (2 as usize, MAP_SIZE as usize, MAP_SIZE as usize),
        |(_, _, _)| 0.0 as f32,
    );

    for (camp, mc) in g.map.iter() {
        let c = *mc;
        if *camp == Camp::Doctor {
            m[[
                0, /* Doctor Marker */
                (c.x + MAP_OFFSET.x) as usize,
                (c.y + MAP_OFFSET.y) as usize,
            ]] = 1.0;
        } else {
            m[[
                1, /* Plague Marker */
                (c.x + MAP_OFFSET.x) as usize,
                (c.y + MAP_OFFSET.y) as usize,
            ]] = 1.0;
        }
    }

    // 3. TURN_DATA
    //
    let mut t = Array::from_shape_fn(
        (1 as usize, MAP_SIZE as usize, MAP_SIZE as usize),
        |(_, _, _)| 0.0 as f32,
    );
    for (camp, mc) in g.map.iter() {
        let c = *mc;
        if *camp == g.turn {
            t[[
                0,
                (c.x + MAP_OFFSET.x) as usize,
                (c.y + MAP_OFFSET.y) as usize,
            ]] = 1.0;
        }
    }

    // 4. FLOW_DATA
    //
    let mut fm = Array::from_shape_fn(
        (2 as usize, MAP_SIZE as usize, MAP_SIZE as usize),
        |(_, _, _)| 0.0 as f32,
    );
    let mut fe = Array::from_shape_fn((11 as usize, SIZE as usize, SIZE as usize), |(_, _, _)| {
        0.0 as f32
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
            0,
            (c.x + MAP_OFFSET.x) as usize,
            (c.y + MAP_OFFSET.y) as usize,
        ]] = 1.0;
    }
    if a.action_phase > ActionPhase::Lockdown {
        // Thanks to the fact that Coord::lockdown() doesn't require the
        // operation a doctor-limited one, we can always duplicate the
        // opponent's map position here.
        let mut c = *g.map.get(&g.opposite(g.turn)).unwrap();
        c = c.lockdown(a.lockdown);
        fm[[
            1,
            (c.x + MAP_OFFSET.x) as usize,
            (c.y + MAP_OFFSET.y) as usize,
        ]] = 1.0;
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
                i, /* fe starts fresh */
                a.trajectory[index].x as usize,
                a.trajectory[index].y as usize,
            ]] = 1.0;
        }
    }
    if a.action_phase >= ActionPhase::SetMarkers {
        // a.markers contains at most 5 coordinates.
        for i in 0..5 {
            if i >= a.markers.len() {
                break;
            };
            fe[[
                i + 6, /* offset by the trajectory: 1 + 5 */
                a.markers[i].x as usize,
                a.markers[i].y as usize,
            ]] += 1.0;
        }
    }

    // Some simple checks
    #[cfg(debug_assertions)]
    {
        let mm = m.clone();
        let mut count = 0;
        for i in 0..2 {
            for j in 0..5 {
                for k in 0..5 {
                    if mm[[i, j, k]] > 0.0 {
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(2, count);
    }

    // XXX: finish these three heads.
    // policy
    let mut policy = Array::from_shape_fn((TOTAL_POS as usize), |(_)| 0.0 as f32);
    let c = a.map.unwrap();

    // valid
    let mut valid = Array::from_shape_fn((TOTAL_POS as usize), |(_)| 0.0 as f32);

    // value: always 1, because with the original final move,
    // this player is winning.
    let mut value = Array::from_shape_fn((1 as usize), |(_)| 1.0 as f32);

    // dummy
    let mut dummy = Array::from_shape_fn(
        (DATASET_UNIT - (DATA_UNIT - CODE_DATA) - 2 * TOTAL_POS - 1 as usize),
        |(_)| 0.0 as f32,
    );

    // wrap it up
    let vtemp = e
        .into_shape((BOARD_DATA,))
        .unwrap()
        .into_iter()
        .chain(m.into_shape((MAP_DATA,)).unwrap().into_iter())
        .chain(t.into_shape((TURN_DATA,)).unwrap().into_iter())
        .chain(fm.into_shape((FLOW_MAP_DATA,)).unwrap().into_iter())
        .chain(fe.into_shape((FLOW_ENV_DATA,)).unwrap().into_iter())
        .chain(policy.into_shape((TOTAL_POS,)).unwrap().into_iter())
        .chain(valid.into_shape((TOTAL_POS,)).unwrap().into_iter())
        .chain(value.into_shape((1,)).unwrap().into_iter())
        .chain(
            dummy
                .into_shape((DATASET_UNIT - (DATA_UNIT - CODE_DATA) - 2 * TOTAL_POS - 1,))
                .unwrap()
                .into_iter(),
        )
        .collect::<Vec<f32>>();

    assert_eq!(vtemp.len(), DATASET_UNIT);
    (*vec).extend(vtemp);
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

fn next(g: &mut Game, a: &Action) {
    g.append_history_with_new_tree(&a.to_sgf_string(g));
    g.commit_action(&a);
    g.next();
}

fn main() -> Result<(), std::io::Error> {
    let args = Args::parse();

    match args.load_dir {
        Some(ref dirname) => {
            let mut num_replay: u32 = 0;
            let mut total_play = 0;
            'total: while total_play < args.batch {
                for filename in read_dir(dirname)? {
                    mine_tsume(&filename?, &args, num_replay)?;

                    total_play = total_play + 1;
                    if total_play >= args.batch {
                        break 'total;
                    }
                }
                num_replay = num_replay + 1;
            }
        }
        None => {
            // We are not ready for this route
            // "".to_string().trim().chars().peekable()
            panic!("Not ready for a fresh play.");
        }
    }

    Ok(())
}

fn mine_tsume(filename: &DirEntry, args: &Args, suffix: u32) -> Result<(), std::io::Error> {
    let mut contents = String::new();
    let mut file = File::open(filename.path())?;
    file.read_to_string(&mut contents)
        .expect("Failed to read file");
    let mut iter = contents.trim().chars().peekable();

    let mut seed: [u8; 32] = [0; 32];
    if let Some(s) = &args.seed {
        let ss = format!("=!={:?}==={}", s.as_bytes(), suffix);
        let bytes = ss.as_bytes();
        seed[..bytes.len()].copy_from_slice(bytes);
    }
    let rng = StdRng::from_seed(seed);
    let shared_rng = Rc::new(RefCell::new(rng));

    let t = TreeNode::new(&mut iter, None);
    let mut g = Game::init_with_rng(Some(t), shared_rng.clone());

    if !g.is_ended() {
        panic!("The game hasn't finished.");
    }

    Ok(())

    /*let ea = Action::new();
    let result: String = loop {
        if let Phase::Main(x) = g.phase {
            let turn: usize = x.try_into().unwrap();
            if g.is_ended() {
                if max_dataset_entries <= num_dataset_entries {
                }
            }
        } else {
            panic!("Not in the main game!");
        }
    };*/
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_something() {
        let s0 = "(
;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen]RE[B+55]
;C[Setup0]AW[ea][af][ce][cc][ad][dc][df][be][de][fa][cb][ba][fb][ed][cf][fd][eb][ab]
;C[Setup0]AB[bb][ef][fc][bd][ac][cd][ae][dd][fe][ec][db][ee][aa][ff][bc][bf][ca][da]
;C[Setup1]AB[fd]
;C[Setup1]AB[ae]
;C[Setup1]AB[cc]
;C[Setup1]AB[eb]
;C[Setup2]AW[ff]
;C[Setup2]AB[bc]
;C[Setup2]AW[cb]
;C[Setup2]AB[dc]
;C[Setup3]AW[ji]
;B[kj][bc][ec][ee][bc][ec][bc][ec]
;W[ih][ff][fe][ee][ec][bc][ec][ec][ff][ff][ff]
;B[gi][dc][de][ce][be][dc][de][ce][de]
;W[hi][cb][eb][cb][cb][cb][cb][cb]
;B[gh][ee][ae][ac][ae][ee][ae][ee]
;W[ih][bc][ec][fc][bc][bc][bc][ec][ec]
;B[jj][ac][bc][bd][bf][ac][bc][bd][ac]
;W[ji][eb][ea][eb][eb][eb][eb][eb]
;B[kh][be][ce][cc][ce][be][be][ce]
;W[hh][fc][ec][bc][ac][fc][ec][bc][bc][ec]
;B[jg][cc][cb][eb][fb][cb][cc][eb][cc]
;W[ih][ea][ba][be][ea][ea][ea][ea][ea]
;B[hi][fb][eb][ed][eb][fb][eb][fb]
;W[ij][be][ce][cf][ce][ce][ce][be][be]
;B[hg][bf][bd][bc][ac][aa][bf][bd][bc][bd]
;W[ih][ac][ae][ee][ae][ae][ae][ac][ac]
;B[jj][aa][ac][bc][bd][ac][aa][bc][ac]
;W[ii][jj][cf][ce][be][cf][ce][ce][cf][ce]
;B[ij][bd][bf][bd][bd][bd]
;W[hi][ee][ae][ac][ee][ee][ee][ae][ae]
;B[hg][ed][eb][ea][ed][eb][eb][ed]
;W[ih][ac][ae][ee][ac][ac][ac][ac][ac]
;B[ij][ea][eb][ed][eb][ea][ea][eb]
;W[ji][be][ce][cc][ce][ce][ce][be][be]
;B[jg][ed][eb][ea][ed][eb][ed][eb]
;W[ih][cc][ce][be][cc][cc][cc][cc][cc]
;B[jj][ea][eb][ed][fd][ed][eb][ea][ed]
;W[ii][hj][ee][fe][fc][ee][ee][ee][ee][ee]
;B[ih][fd][fb][fd][fd][fd][fd]
;W[hi][fc][fe][ee][fe][fe][fe][fe][fe]
;B[ii][bf][ef][bf][bf][bf][bf]
;W[jh][be][ba][ea][ba][ba][ba][ba][ba]
;B[hh][fb][eb][cb][fb][fb][fb]
;W[ih][ea][fa][ea][ea][ea][ea]
;B[ii][cb][cc][cb][cb][cb][cb]
;W[jh][ee][ec][fc][ec]
;B[ki][cc][dc][de][cc][dc][cc][dc]
;W[hj][fc][ec][bc][ac][ae][fc][ac][fc][fc][bc]
;B[gi][ef][bf][bd][ef][ef][ef][ef]
;W[hi][ae][ee][ae][ae][ae]
;B[ii][bd][cd]
;W[ih][ee][ec]
;B[hi][cd][bd][bf][cd][cd][cd][cd]
;W[hj][fa][fb][fa][fa][fa][fa][fa]
;B[gi][de][dc][cc][dc][de][dc][de]
;W[ii][ig][ec][ee][ef]
;B[hh][cc][cb][ab][cc][cb][cc][cb]
;W[hi][fb][fd][fb][fb][fb][fb][fb]
;B[ki][ab][cb][eb][fb][cb][ab][ab][cb]
;W[jh][ef][ee][ae][ef][ef][ef][ef][ef]
;B[hj][fb][eb][ed][ad][af][fb][ad][fb][ad]
;W[ij][ae][ee]
;B[jg][bf][bd][bc][bb][db][bc][bb][bb][bc]
;W[jh][ee][ef];B[ii][db][dd][cd][db][dd][db][dd])"
            .to_string();
        let _final_step = "";
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));

        let t2 = g.history.clone();
        g.undo();
        let mut vec: Vec<f32> = Vec::new();
        if let Ok(a) = t2.borrow().to_action_do_func(&g, encode, &mut vec) {
            assert_eq!(Coord::new(0 as i32, 0 as i32), a.map.unwrap());
            assert_eq!(Coord::new(2 as i32, 3 as i32), a.character.unwrap());
            assert_eq!(4 * DATASET_UNIT, vec.len());
            let offset = BOARD_DATA + MAP_DATA + TURN_DATA;
            assert_eq!(vec[DATASET_UNIT * 0 + offset + 0 * 5 * 5 + 2 * 5 + 2], 1.0);
            assert_eq!(vec[DATASET_UNIT * 0 + offset + 1 * 5 * 5 + 2 * 5 + 2], 0.0);
            assert_eq!(
                vec[DATASET_UNIT * 0 + offset + FLOW_MAP_DATA + 0 * 6 * 6 + 3 * 6 + 1],
                0.0
            );
            assert_eq!(vec[DATASET_UNIT * 1 + offset + 0 * 5 * 5 + 2 * 5 + 2], 1.0);
            assert_eq!(vec[DATASET_UNIT * 1 + offset + 1 * 5 * 5 + 2 * 5 + 2], 0.0);
            assert_eq!(
                vec[DATASET_UNIT * 1 + offset + FLOW_MAP_DATA + 0 * 6 * 6 + 3 * 6 + 1],
                1.0
            );
            assert_eq!(
                vec[DATASET_UNIT * 1 + offset + FLOW_MAP_DATA + 1 * 6 * 6 + 3 * 6 + 3],
                0.0
            );
            assert_eq!(
                vec[DATASET_UNIT * 2 + offset + FLOW_MAP_DATA + 1 * 6 * 6 + 3 * 6 + 3],
                1.0
            );
            let bytes: &[u8] = cast_slice(&vec);
        } else {
            panic!("??");
        };
    }
}
