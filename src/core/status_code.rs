pub fn str_to_full_msg(s: &'static str) -> &'static str {
    // The indexing is tricky. It should skip the 'x', so from 2.
    let r = u32::from_str_radix(&s[2..], 16);
    let mut index = u32::default();

    match r {
        Ok(n) => index += n,
        Err(e) => {
            panic!("{} {}", e, "Invalid status code encountered. No number.")
        }
    }

    match s.chars().next() {
        Some('E') => index = index + 512,
        Some('W') => index = index + 256,
        Some('I') => index = index + 0,
        _ => panic!("Invalid status code encountered. Wrong Type."),
    }

    match index {
        // Ex20
        544 => return "Invalid map position: No possible route",
        543 => return "[Setup2] Invalid position: other characters",
        542 => return "[Setup2] Character already here",
        541 => return "[Setup1] Invalid stacking",
        540 => return "Setup not done",
        539 => return "[Setup3] Invalid position",
        538 => return "No need to setup",
        537 => return "[Setup2] Invalid position: marker here",
        // Ex18
        536 => return "[Setup2] Invalid order",
        535 => return "[Setup2] Not finished",
        534 => return "[Setup0] out-of-board setting",
        533 => return "[Setup1] Too many markers",
        532 => return "[Setup1] Not finished",
        531 => return "[Setup0] Not finished",
        530 => return "Invalid checkpoint: Setup0",
        529 => return "[Setup1] Shouldn't share any row, column, and quandrant",
        // Ex10
        528 => return "Invalid checkpoint: Setup1",
        527 => return "Invalid lockdown resolution",
        526 => return "Shouldn't trigger lockdown action, check application",
        525 => return "Invalid marker action: (Doctor) wrong order to cure",
        524 => return "Invalid marker action: (Plague) not evenly distributed",
        523 => return "Invalid marker action: more than a colony",
        522 => return "Invalid marker action: over the quota",
        521 => return "Invalid marker action: not in the trajectory",
        // Ex08
        520 => return "Invalid marker action: opponent is here",
        519 => return "Invalid move: stopping at the opponent's character",
        518 => return "Invalid move: going through or stopping at the opponent's colony",
        517 => return "Invalid move along the direction",
        516 => return "Invalid board move",
        515 => return "Invalid coordinate as the next step",
        514 => return "Invalid character position",
        513 => return "Invalid map position: Lockdown now",
        512 => return "Invalid map position: Collide with opponent",
        // Ix01 and beyond: Currently used only in coord_server
        5 => return "LOSE",
        4 => return "WIN",
        3 => return "TURN",
        2 => return "DONE",
        1 => return "OK",
        0 => return "Skip",
        _ => return "??",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_msg() {
        assert_eq!(str_to_full_msg("Ix00"), "Skip");
    }
}
