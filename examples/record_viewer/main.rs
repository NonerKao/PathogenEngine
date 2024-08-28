use actix_files as fs;
use actix_web::{web, App, HttpServer, Responder, Result};
use clap::Parser;
use serde_derive::Serialize;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use std::sync::Mutex;

use pathogen_engine::core::tree::*;
use pathogen_engine::core::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// SGF files to be loaded from
    #[arg(short, long)]
    load: String,
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

async fn index() -> Result<fs::NamedFile> {
    Ok(fs::NamedFile::open("static/index.html")?)
}

struct AppState {
    amgs: Arc<Mutex<GameState>>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Load the record
    let args = Args::parse();
    let mut contents = String::new();
    print!("Loading ... ");
    let mut file = File::open(args.load)?;
    file.read_to_string(&mut contents)
        .expect("Failed to read file");
    let mut iter = contents.trim().chars().peekable();
    let t = TreeNode::new(&mut iter, None);
    let g = Game::init(Some(t));

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

    let mut gs = GameState {
        white_positions: setup0_h.borrow().properties[1].value.clone(),
        black_positions: setup0_u.borrow().properties[1].value.clone(),
        steps: vec![],
    };

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

    let mut curr_node = setup3_0.borrow().children[0].clone();
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
    }

    println!("... Done");

    let setup_state = web::Data::new(AppState {
        amgs: Arc::new(Mutex::new(gs)),
    });

    // Start the service
    HttpServer::new(move || {
        App::new()
            .app_data(setup_state.clone())
            .route("/", web::get().to(index))
            .route("/game_state", web::get().to(game_state))
            .service(fs::Files::new("/static", "static").show_files_listing())
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
