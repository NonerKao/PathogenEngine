pub fn str_to_full_msg(s: &'static str) -> &'static str {
    let r = u32::from_str_radix(&s[1..], 16);
    let mut index = 0;

    match r {
        Ok(n) => index = n,
        Err(_) => panic!("Invalid status code encountered. No number."),
    }

    match s.chars().next() {
        Some('E') => index = index + 512,
        Some('W') => index = index + 256,
        Some('I') => index = index + 0,
        _ => panic!("Invalid status code encountered. Wrong Type."),
    }

    match index {
        514 => return "Invalid hero position",
        513 => return "Invalid compass area",
        512 => return "Invalid compass position: Collide with opponent",
        0 => return "Skip",
        _ => return "??",
    }
}
