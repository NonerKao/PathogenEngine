use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

fn handle_client(mut stream: &TcpStream) -> bool {
    let mut buffer = [0; 1]; // to read the 1-byte action from agent
    loop {
        match stream.peek(&mut buffer) {
            Ok(0) => {
                println!("Client disconnected.");
                return false;
            }
            Ok(_) => {
                let bytes_read = stream.read(&mut buffer).unwrap();
                println!("{:?}", buffer);

                let action = buffer[0] as usize;

                // Here, interact with the IIG environment using the action
                // and get the resulting game state and status code.
                let game_state: [u8; 150] = [0; 150]; // This should be obtained from IIG environment
                let status_code: [u8; 4] = [0, 0, 0, 0]; // This too, from IIG

                let response = [&game_state[..], &status_code[..]].concat();
                match stream.write(&response) {
                    Err(_) => {
                        println!("Client disconnected.");
                        return false;
                    }
                    _ => {}
                }
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

fn main() -> Result<(), std::io::Error> {
    let white_listener = TcpListener::bind("127.0.0.1:6241").unwrap();
    let black_listener = TcpListener::bind("127.0.0.1:3698").unwrap();

    let hello: [u8; 154] = [0; 154];
    let mut w = white_listener.accept().unwrap().0;
    w.write(&hello)?;
    let mut b = black_listener.accept().unwrap().0;
    b.write(&hello)?;

    let mut w_live = true;
    let mut b_live = true;

    while w_live || b_live {
        if w_live {
            w_live = handle_client(&mut w);
        }
        if b_live {
            b_live = handle_client(&mut b);
        }
    }
    Ok(())
}
