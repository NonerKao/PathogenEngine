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
        527 => return "Invalid lockdown resolution",
        526 => return "Shouldn't trigger lockdown action, check application",
        525 => return "Invalid marker action: (Doctor) wrong order to cure",
        524 => return "Invalid marker action: (Plague) not evenly distributed",
        523 => return "Invalid marker action: more than a colony",
        522 => return "Invalid marker action: over the quota",
        521 => return "Invalid marker action: not in the trajectory",
        520 => return "Invalid marker action: opponent is here",
        519 => return "Invalid move: stopping at the opponent's hero",
        518 => return "Invalid move: going through or stopping at the opponent's colony",
        517 => return "Invalid move along the direction",
        516 => return "Invalid compass move",
        515 => return "Invalid coordinate as the next step",
        514 => return "Invalid hero position",
        513 => return "Invalid compass area",
        512 => return "Invalid compass position: Collide with opponent",
        0 => return "Skip",
        _ => return "??",
    }
}
