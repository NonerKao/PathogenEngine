use std::env;
use std::io::{self, Write, Read};
use std::net::TcpStream;
use std::str::FromStr;

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
    let mut buffer = [0u8; 40];
    let bytes_read = stream.read(&mut buffer)?;
    if bytes_read > 0 {
        let received_str = String::from_utf8_lossy(&buffer[..bytes_read]);
        println!("Received: {}", received_str);
    } else {
        println!("No data received from the server.");
    }

    // Enter interaction mode
    loop {
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

