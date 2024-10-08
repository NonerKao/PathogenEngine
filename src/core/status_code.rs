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
        551 => return "Invalid coordinates: should be board",
        550 => return "Invalid coordinates: should be map",
        549 => return "Invalid map position: Out of the possible 21 grids",
        548 => return "Invalid set-marker choice: Doctor should prioritize",
        547 => return "Invalid move choice: No possible route",
        546 => return "Invalid lockdown choice: No possible route",
        545 => return "Invalid character choice: No possible route",
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
        // Wx00: always success side channel commands instead of move status
        256 => return "Query for valid moves",
        // Ix01 and beyond: Currently used only in coord_server
        13 => return "TURN_ACTION_CLIENT",
        12 => return "OK_ACTION_CLIENT",
        11 => return "MARKERS",
        10 => return "SIM_MYSELF_WIN",
        9 => return "SIM_OPPONENT_WIN",
        8 => return "SIM_MYSELF",
        7 => return "SIM_OPPONENT",
        6 => return "DROP",
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
