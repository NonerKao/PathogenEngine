use actix_web::{web, Responder};
use serde_derive::Serialize;
use std::cell::RefCell;
use std::env;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

// use pathogen_engine::core::action::Action;
use pathogen_engine::core::tree::TreeNode;
use pathogen_engine::core::*;

mod web_server;
use web_server::start_web_server;
use web_server::StatusCode;

const MAX_SGF_LEN: usize = 500;
const MAX_ACT_LEN: usize = 100;
fn read_exact_bytes(stream: &mut TcpStream, n: usize, buffer: &mut [u8]) -> io::Result<()> {
    let mut total_read = 0; // Track the total number of bytes read

    while total_read < n {
        match stream.read(&mut buffer[total_read..]) {
            Ok(0) => {
                // Connection was closed, but we haven't read all the expected bytes
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Connection closed early",
                ));
            }
            Ok(bytes_read) => {
                total_read += bytes_read;
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                // Non-blocking socket, no data available yet, continue
                continue;
            }
            Err(e) => {
                // Other errors
                return Err(e);
            }
        }
    }

    Ok(())
}

fn main() -> io::Result<()> {
    // Accepting host and port from command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <host> <port>", args[0]);
        return Ok(());
    }

    let host = &args[1];
    let port = &args[2];
    let address = format!("{}:{}", host, port);

    // Connect to the server
    let mut stream = TcpStream::connect(address)?;

    // Send the initial byte 0xc8: SET_ACTION_CLIENT
    stream.write(&[0xc8])?;

    // Receive up to 40 bytes from the server
    let mut status = vec![0u8; 4];

    'set: loop {
        read_exact_bytes(&mut stream, 4, &mut status)?;
        let mut sgf_buf: [u8; MAX_SGF_LEN] = [0; MAX_SGF_LEN];
        if status == "Ix0c".as_bytes() {
            // Get sgf
            read_exact_bytes(&mut stream, MAX_SGF_LEN, &mut sgf_buf)?;
        } else if status == "Ix0e".as_bytes() {
            // No more games to play
            break 'set;
        } else {
            panic!("unexpected status: {:?}", status);
        }

        let sgf = match std::str::from_utf8(&sgf_buf[..]) {
            Ok(s) => {
                // Remove trailing null bytes (optional, depending on your needs)
                s.trim_end_matches('\0').to_string()
            }
            Err(_) => {
                panic!("The byte array contains invalid UTF-8 data.");
            }
        };
        let mut iter = sgf.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));

        let mut gs = GameState {
            white_positions: vec![],
            black_positions: vec![],
            steps: vec![],
        };
        setup_to_web_ui(&g, &mut gs);
        let shared_amgs = Arc::new(Mutex::new(gs));
        let setup_state = web::Data::new(AppState {
            amgs: shared_amgs.clone(),
        });

        let ns = StatusString {
            status: String::new(),
        };
        let shared_amss = Arc::new(Mutex::new(ns));
        let next_status = web::Data::new(NextStatus {
            amss: shared_amss.clone(),
        });

        let (click_tx, click_rx): (
            Sender<(u8, Sender<StatusCode>)>,
            Receiver<(u8, Sender<StatusCode>)>,
        ) = mpsc::channel();
        let _web_server_handle = thread::spawn(move || {
            start_web_server(click_tx, setup_state.clone(), next_status.clone());
        });

        'game: loop {
            read_exact_bytes(&mut stream, 4, &mut status)?;
            if status == "Ix0d".as_bytes() {
                let mut action_buf: [u8; MAX_ACT_LEN] = [0; MAX_ACT_LEN];
                read_exact_bytes(&mut stream, MAX_ACT_LEN, &mut action_buf)?;
                let action_string = match std::str::from_utf8(&action_buf[..]) {
                    Ok(a) => {
                        // Remove trailing null bytes (optional, depending on your needs)
                        a.trim_end_matches('\0').to_string()
                    }
                    Err(_) => {
                        panic!("The byte array contains invalid UTF-8 data.");
                    }
                };
                g.append_history_with_new_tree(&action_string);
                one_standard_move_to_web_ui(&g, shared_amgs.clone());
                let temp_t = g.history.clone();
                if let Ok(a) = temp_t.borrow().to_action(&g) {
                    g.commit_action(&a);
                    g.next();
                } else {
                    panic!("A wrong action was sent?");
                };
                continue 'game;
            }

            'turn: loop {
                if status == "Ix02".as_bytes() || status == "Ix00".as_bytes() {
                    continue 'game;
                } else if status == "Ix04".as_bytes() {
                    update_next_status(shared_amss.clone(), "Win!");
                    break 'game;
                } else if status == "Ix05".as_bytes() {
                    update_next_status(shared_amss.clone(), "Lose!");
                    break 'game;
                } else {
                    if status == "Ix01".as_bytes() || status == "Ix0b".as_bytes() {
                        // the normal middle moves
                    } else if status == "Ix03".as_bytes() {
                        // the main play of this round
                        update_next_status(shared_amss.clone(), "Ready");
                    } else {
                        // Error?
                    }

                    if let Ok((c, status_tx)) = click_rx.recv() {
                        stream.write(&[c])?;
                        read_exact_bytes(&mut stream, 4, &mut status)?;
                        let s = format!("{:?}", status);
                        let _ = status_tx.send(StatusCode { status: s });
                    }
                    continue 'turn;
                }
            }
        } // 'game
        std::thread::sleep(Duration::from_millis(1000));
    } // 'set, but actually not supported for web

    Ok(())
}

#[derive(Serialize, Clone)]
struct GameState {
    white_positions: Vec<String>,
    black_positions: Vec<String>,
    steps: Vec<Step>,
}

#[derive(Serialize, Clone)]
struct Step {
    id: u32,
    pos: String,
    is_marker: bool,
    char1: char,
    marker: i32,
}

async fn update_state(data: web::Data<AppState>) -> impl Responder {
    let gs = data.amgs.lock().unwrap();
    web::Json(gs.clone())
}

async fn game_state(data: web::Data<AppState>) -> impl Responder {
    let gs = data.amgs.lock().unwrap();
    web::Json(gs.clone())
}

async fn get_next_status(data: web::Data<NextStatus>) -> impl Responder {
    let ss = data.amss.lock().unwrap();
    web::Json(ss.clone())
}

#[derive(Serialize, Clone)]
struct StatusString {
    status: String,
}

struct NextStatus {
    amss: Arc<Mutex<StatusString>>,
}

fn update_next_status(amss: Arc<Mutex<StatusString>>, s: &'static str) {
    let mut ns = amss.lock().unwrap();
    ns.status = s.to_string();
}

struct AppState {
    amgs: Arc<Mutex<GameState>>,
}

fn setup_to_web_ui(g: &Game, gs: &mut GameState) {
    let setup0_h = g.history.borrow().to_root().borrow().children[0]
        .borrow()
        .children[0]
        .clone();
    let setup0_u = g.history.borrow().to_root().borrow().children[0]
        .borrow()
        .children[0]
        .borrow()
        .children[0]
        .clone();
    gs.white_positions = setup0_h.borrow().properties[1].value.clone();
    gs.black_positions = setup0_u.borrow().properties[1].value.clone();

    let mut id = 0;
    let setup1_0 = setup0_u.borrow().children[0].clone();
    let setup1_1 = setup1_0.borrow().children[0].clone();
    let setup1_2 = setup1_1.borrow().children[0].clone();
    let setup1_3 = setup1_2.borrow().children[0].clone();
    gs.steps.push(Step {
        id: id,
        pos: setup1_0.borrow().properties[1].value[0].clone(),
        is_marker: true,
        char1: 'P',
        marker: -1,
    });
    id = id + 1;
    gs.steps.push(Step {
        id: id,
        pos: setup1_1.borrow().properties[1].value[0].clone(),
        is_marker: true,
        char1: 'P',
        marker: -1,
    });
    id = id + 1;
    gs.steps.push(Step {
        id: id,
        pos: setup1_2.borrow().properties[1].value[0].clone(),
        is_marker: true,
        char1: 'P',
        marker: -1,
    });
    id = id + 1;
    gs.steps.push(Step {
        id: id,
        pos: setup1_3.borrow().properties[1].value[0].clone(),
        is_marker: true,
        char1: 'P',
        marker: -1,
    });
    id = id + 1;

    let setup2_0 = setup1_3.borrow().children[0].clone();
    let setup2_1 = setup2_0.borrow().children[0].clone();
    let setup2_2 = setup2_1.borrow().children[0].clone();
    let setup2_3 = setup2_2.borrow().children[0].clone();
    gs.steps.push(Step {
        id: id,
        pos: setup2_0.borrow().properties[1].value[0].clone(),
        is_marker: false,
        char1: 'D',
        marker: 0,
    });
    id = id + 1;
    gs.steps.push(Step {
        id: id,
        pos: setup2_1.borrow().properties[1].value[0].clone(),
        is_marker: false,
        char1: 'P',
        marker: 0,
    });
    id = id + 1;
    gs.steps.push(Step {
        id: id,
        pos: setup2_2.borrow().properties[1].value[0].clone(),
        is_marker: false,
        char1: 'D',
        marker: 0,
    });
    id = id + 1;
    gs.steps.push(Step {
        id: id,
        pos: setup2_3.borrow().properties[1].value[0].clone(),
        is_marker: false,
        char1: 'P',
        marker: 0,
    });
    id = id + 1;

    let setup3_0 = setup2_3.borrow().children[0].clone();
    gs.steps.push(Step {
        id: id,
        pos: setup3_0.borrow().properties[1].value[0].clone(),
        is_marker: false,
        char1: 'D',
        marker: 0,
    });

    if setup3_0.borrow().children.len() > 0 {
        let curr_node = setup3_0.borrow().children[0].clone();
        decode_standard_move(gs, curr_node.clone());
    }
}

fn one_standard_move_to_web_ui(g: &Game, amgs: Arc<Mutex<GameState>>) {
    let mut gs = amgs.lock().unwrap();
    let curr_node = g.history.clone();
    decode_standard_move(&mut gs, curr_node.clone());
}

fn decode_standard_move(gs: &mut GameState, curr_node: Rc<RefCell<TreeNode>>) {
    // opening
    let mut id: u32 = gs.steps.len().try_into().unwrap();
    let side = if curr_node.borrow().properties[0].ident == "W" {
        'D'
    } else {
        'P'
    };

    // core
    let mut route = vec![];
    for s in curr_node.borrow().properties[0].value.iter() {
        let is_marker = route.contains(s);
        gs.steps.push(Step {
            id: id,
            pos: s.clone(),
            is_marker: is_marker,
            char1: side,
            marker: if !is_marker {
                0
            } else if side == 'D' {
                1
            } else {
                -1
            },
        });

        id = id + 1;
        if !is_marker {
            route.push(s.clone());
        }
    }

    // closing
    assert_eq!(curr_node.borrow().children.len(), 0);
}
