use clap::Parser;
use std::boxed::Box;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::net::TcpListener;
use std::thread;
use std::time::Duration;

use pathogen_engine::core::Camp;
use pathogen_engine::core::Game;
// XXX: no, this should be called server

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// SGF file to be loaded from
    #[arg(short, long)]
    load: Option<String>,

    /// SGF file to be saved to
    #[arg(short, long, default_value_t = String::from("output.sgf"))]
    save: String,

    /// IP to be exported
    #[arg(short, long)]
    ip: Option<String>,

    /// Port to be exported
    #[arg(short, long)]
    port: Option<String>,

    /// player interface: None, hjkl-bot, move-bot
    #[arg(long)]
    doctor: Option<String>,
    #[arg(long)]
    plague: Option<String>,
}

fn main() {
    let mut args = Args::parse();
    let mut p = HashMap::new(); //<Camp, TcpStream>,

    let listener: Option<TcpListener> = if args.ip == None || args.port == None {
        if args.doctor != None || args.plague != None {
            panic!("No server provided");
        }
        None
    } else {
        let mut s = args.ip.unwrap().clone();
        s = s + ":" + &args.port.unwrap();
        println!("{}", s);
        Some(TcpListener::bind(s.as_str()).unwrap())
    };

    match listener {
        None => {}
        Some(listener) => {
            if args.doctor != None || args.plague != None {
                listener.set_nonblocking(true).unwrap();

                // accept two players within 60 secs
                let timeout = Duration::from_secs(60);
                let start_time = std::time::Instant::now();

                while start_time.elapsed() < timeout {
                    match listener.accept() {
                        Ok((mut stream, addr)) => {
                            print!("Client connected from {}, playing...", addr);
                            if args.doctor != None && args.plague != None {
                                // XXX: we later need to specify more prompts for the other types of bot
                                loop {
                                    stream
                                        .write("Are you play as Doctor? (y/n)".as_bytes())
                                        .unwrap();

                                    let mut b: [u8; 1] = [0];
                                    stream.read(&mut b).unwrap();
                                    if b[0] == 'y' as u8 {
                                        p.insert(Camp::Doctor, stream);
                                        args.doctor = None;
                                        break;
                                    } else if b[0] == 'n' as u8 {
                                        p.insert(Camp::Plague, stream);
                                        args.plague = None;
                                        break;
                                    }
                                }
                            } else if args.doctor != None && args.plague == None {
                                p.insert(Camp::Doctor, stream);
                                args.doctor = None;
                                break;
                            } else if args.doctor == None && args.plague != None {
                                p.insert(Camp::Plague, stream);
                                args.plague = None;
                                break;
                            }
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            // No new client yet, wait for a short while before checking again
                            thread::sleep(Duration::from_millis(100));
                        }
                        Err(e) => {
                            eprintln!("Error: {}", e);
                            break;
                        }
                    }
                }
            }
        }
    }

    if args.doctor != None && p.get(&Camp::Doctor).is_none() {
        panic!("No player connected");
    };
    if args.plague != None && p.get(&Camp::Plague).is_none() {
        panic!("No player connected");
    };

    let g = Box::new(Game::init());
    let e = "".to_string();
    let mut iter = e.trim().chars().peekable();
    match args.load {
        Some(filename) => {
            // Ideally, this should be the pure "view mode", where we read the games(s).
            // In reality, I need this to bridge the gap before all phases are treated equal.
            let mut file = File::open(filename.as_str()).expect("Failed to open file");
            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .expect("Failed to read file");
            let mut iter = contents.trim().chars().peekable();
        }
        None => {}
    }
}
