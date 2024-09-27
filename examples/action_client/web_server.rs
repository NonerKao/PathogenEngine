use crate::{game_state, update_state, AppState};
use actix_files as fs;
use actix_web::{post, web, App, HttpResponse, HttpServer, Responder, Result};
use clap::Parser;
use serde_derive::{Deserialize, Serialize};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

#[derive(Deserialize)]
struct ClickData {
    coord: u8,
}

#[derive(Serialize)]
pub struct StatusCode {
    pub status: String,
}

// Handler for receiving click coordinates
#[post("/click")]
async fn receive_click(
    data: web::Json<ClickData>,
    click_tx: web::Data<Sender<(u8, Sender<StatusCode>)>>,
) -> impl Responder {
    let c = data.coord;
    let (status_tx, status_rx): (Sender<StatusCode>, Receiver<StatusCode>) = mpsc::channel();
    // Send the coordinates back to main.rs
    if let Err(e) = click_tx.send((c, status_tx)) {
        eprintln!("Failed to send click data to main: {}", e);
        return HttpResponse::InternalServerError().body("Failed to process click");
    }

    if let Ok(game_response) = status_rx.recv() {
        return HttpResponse::Ok().json(game_response);
    }

    HttpResponse::InternalServerError().body("Error processing the move")
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// SGF files to be loaded from
    #[arg(short, long)]
    load: String,
}

async fn index() -> Result<fs::NamedFile> {
    Ok(fs::NamedFile::open("static/index.html")?)
}

pub fn start_web_server(
    click_tx: Sender<(u8, Sender<StatusCode>)>,
    setup_state: web::Data<AppState>,
) {
    // Clone the click_tx to move into the Actix-web data
    let click_tx_data = web::Data::new(click_tx);

    // Start the Actix system
    let server = HttpServer::new(move || {
        App::new()
            .app_data(click_tx_data.clone())
            .app_data(setup_state.clone())
            .service(receive_click)
            .route("/", web::get().to(index))
            .route("/update_state", web::get().to(update_state))
            .route("/game_state", web::get().to(game_state))
            .service(fs::Files::new("/static", "static").show_files_listing())
    })
    .bind(("127.0.0.1", 8080))
    .expect("Can not bind to port 8080")
    .run();

    // Run the server (this blocks until the server is stopped)
    let server_handle = thread::spawn(move || {
        let sys = actix_web::rt::System::new();
        sys.block_on(server)
    });

    // Wait for both threads to finish
    let _ = server_handle.join().expect("Server thread panicked");
}
