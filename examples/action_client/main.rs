use actix_web::{post, web, App, HttpResponse, HttpServer, Responder, Result};
use serde_derive::Serialize;
use std::env;
use std::io::{self, Read, Write};
use std::net::TcpStream;
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
    println!("Connected to the server.");

    // Send the initial byte 0xc8: SET_ACTION_CLIENT
    stream.write(&[0xc8])?;
    println!("Sent byte 0xc8 to the server.");

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
        history_to_web_ui(&g, &mut gs);
        let setup_state = web::Data::new(AppState {
            amgs: Arc::new(Mutex::new(gs)),
        });

        let (click_tx, click_rx): (Sender<u8>, Receiver<u8>) = mpsc::channel();
        let web_server_handle = thread::spawn(move || {
            start_web_server(click_tx, setup_state.clone());
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
                let temp_t = g.history.clone();
                if let Ok(a) = temp_t.borrow().to_action(&g) {
                    g.commit_action(&a);
                    g.next();
                } else {
                    panic!("A wrong action was sent?");
                };
                continue 'game;
            }
            if status == "Ix02".as_bytes() || status == "Ix00".as_bytes() {
                continue 'game;
            } else if status == "Ix04".as_bytes() {
                println!("win!");
                break 'game;
            } else if status == "Ix05".as_bytes() {
                println!("lose!");
                break 'game;
            } else {
                if status == "Ix01".as_bytes() || status == "Ix0b".as_bytes() {
                    // the normal middle moves
                } else if status == "Ix03".as_bytes() {
                    // the main play of this round
                } else {
                    // Error?
                }

                if let Ok(c) = click_rx.recv() {
                    println!("Received {} from web interface.", c);
                    stream.write(&[c])?;
                }
            }
        }
    }

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

async fn game_state(data: web::Data<AppState>) -> impl Responder {
    let gs = data.amgs.lock().unwrap();
    web::Json(gs.clone())
}

struct AppState {
    amgs: Arc<Mutex<GameState>>,
}

fn history_to_web_ui(g: &Game, gs: &mut GameState) {
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
    id = id + 1;

    /*let mut curr_node = setup3_0.borrow().children[0].clone();
    loop {
        // opening
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
        if curr_node.borrow().children.len() == 0 {
            break;
        } else {
            let temp = curr_node.borrow().children[0].clone();
            curr_node = temp;
        }
    }*/
}
