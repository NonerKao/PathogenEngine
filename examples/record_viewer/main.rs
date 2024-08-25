use actix_files as fs;
use actix_web::{web, App, HttpServer, Responder, Result};
use serde_derive::Serialize;

#[derive(Serialize)]
struct GameState {
    white_positions: Vec<String>,
    black_positions: Vec<String>,
}

async fn game_state() -> impl Responder {
    let state = GameState {
        white_positions: vec!["db", "ef", "fe"].into_iter().map(String::from).collect(),
        black_positions: vec!["bd", "af", "fd"].into_iter().map(String::from).collect(),
    };
    web::Json(state)
}

async fn index() -> Result<fs::NamedFile> {
    Ok(fs::NamedFile::open("static/index.html")?)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(index))
            .route("/game_state", web::get().to(game_state))
            .service(fs::Files::new("/static", "static").show_files_listing())
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

