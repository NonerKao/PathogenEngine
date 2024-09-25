// src/web_server.rs

use actix_files as fs;
use actix_web::{post, web, App, HttpResponse, HttpServer, Responder};
use serde_derive::Deserialize;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;

#[derive(Deserialize)]
struct ClickData {
    x: i32,
    y: i32,
}

// Handler for receiving click coordinates
#[post("/click")]
async fn receive_click(
    data: web::Json<ClickData>,
    click_tx: web::Data<Sender<(i32, i32)>>,
) -> impl Responder {
    let x = data.x;
    let y = data.y;
    // Send the coordinates back to main.rs
    if let Err(e) = click_tx.send((x, y)) {
        eprintln!("Failed to send click data to main: {}", e);
        return HttpResponse::InternalServerError().body("Failed to process click");
    }
    HttpResponse::Ok().body("Click received")
}

pub fn start_web_server(click_tx: Sender<(i32, i32)>) {
    // Clone the click_tx to move into the Actix-web data
    let click_tx_data = web::Data::new(click_tx);

    // Start the Actix system
    let server = HttpServer::new(move || {
        App::new()
            .app_data(click_tx_data.clone())
            .service(receive_click)
            .service(fs::Files::new("/", "./static").index_file("index.html"))
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
