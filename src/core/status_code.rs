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
        539 => return "[Setup3] Invalid position",
        538 => return "No need to setup",
        537 => return "[Setup2] Invalid position",
        536 => return "[Setup2] Invalid order",
        535 => return "[Setup2] Not finished",
        534 => return "[Setup0] out-of-board setting",
        533 => return "[Setup1] Too many markers",
        532 => return "[Setup1] Not finished",
        531 => return "[Setup0] Not finished",
        530 => return "Invalid checkpoint: Setup0",
        529 => return "[Setup1] Shouldn't share any row, column, and quandrant",
        528 => return "Invalid checkpoint: Setup1",
        527 => return "Invalid lockdown resolution",
        526 => return "Shouldn't trigger lockdown action, check application",
        525 => return "Invalid marker action: (Doctor) wrong order to cure",
        524 => return "Invalid marker action: (Plague) not evenly distributed",
        523 => return "Invalid marker action: more than a colony",
        522 => return "Invalid marker action: over the quota",
        521 => return "Invalid marker action: not in the trajectory",
        520 => return "Invalid marker action: opponent is here",
        519 => return "Invalid move: stopping at the opponent's character",
        518 => return "Invalid move: going through or stopping at the opponent's colony",
        517 => return "Invalid move along the direction",
        516 => return "Invalid map move",
        515 => return "Invalid coordinate as the next step",
        514 => return "Invalid character position",
        513 => return "Invalid map area",
        512 => return "Invalid map position: Collide with opponent",
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
