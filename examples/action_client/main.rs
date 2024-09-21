use std::env;
use std::io::{self, Read, Write};
use std::net::TcpStream;

use pathogen_engine::core::action::Action;
use pathogen_engine::core::tree::TreeNode;
use pathogen_engine::core::*;

const MAX_SGF_LEN: usize = 400;
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

    loop {
        read_exact_bytes(&mut stream, 4, &mut status)?;
        let mut sgf_buf: [u8; MAX_SGF_LEN] = [0; MAX_SGF_LEN];
        if status == "Ix0c".as_bytes() {
            // Get sgf
            read_exact_bytes(&mut stream, MAX_SGF_LEN, &mut sgf_buf)?;
            println!("{:?}", sgf_buf);
        } else if status == "Ix0e".as_bytes() {
            // No more games to play
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
        let g = Game::init(Some(t));

        // Get input from the user in 4-byte format like 0xXX
        println!("Enter a hexadecimal value (e.g., 0xA1) or 'exit' to quit:");
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        // Exit condition
        if input.trim().eq_ignore_ascii_case("exit") {
            println!("Exiting the client.");
            break;
        }

        // Parse input as hexadecimal (expecting format like 0xXX)
        if let Ok(byte) = u8::from_str_radix(input.trim().trim_start_matches("0x"), 16) {
            // Send the parsed byte to the server
            stream.write(&[byte])?;
            println!("Sent byte {:#04x} to the server.", byte);
        } else {
            println!("Invalid input. Please enter a valid hexadecimal value.");
        }
    }

    Ok(())
}
