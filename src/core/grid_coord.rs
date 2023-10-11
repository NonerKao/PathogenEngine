use std::collections::HashMap;
use std::ops::{Add, Sub};

pub const BOARD_MIN: i32 = 0;
pub const BOARD_MAX: i32 = crate::core::SIZE;
pub const ORIGIN: Coord = Coord { x: 0, y: 0 };
pub const MAP_OFFSET: Coord = Coord { x: 2, y: 2 };

#[derive(Hash, PartialEq, Eq, Debug, Copy, Clone)]
pub struct Coord {
    pub x: i32,
    pub y: i32,
}

impl Coord {
    pub fn new(x: i32, y: i32) -> Coord {
        Coord { x: x, y: y }
    }

    pub fn env_to_sgf(&self) -> String {
        let mut s = String::new();
        let base: i32 = 'a' as i32;
        let x: u32 = (base + self.x) as u32;
        let y: u32 = (base + self.y) as u32;
        s.push(std::char::from_u32(x).unwrap());
        s.push(std::char::from_u32(y).unwrap());
        return s;
    }

    pub fn map_to_sgf(&self) -> String {
        let mut s = String::new();
        let base: i32 = 'i' as i32;
        let x: u32 = (base + self.x) as u32;
        let y: u32 = (base + self.y) as u32;
        s.push(std::char::from_u32(x).unwrap());
        s.push(std::char::from_u32(y).unwrap());
        return s;
    }

    pub fn lockdown(&self, ld: Lockdown) -> Coord {
        match ld {
            Lockdown::Normal => *self,
            Lockdown::CC90 => Coord {
                x: self.y,
                y: -self.x,
            },
            Lockdown::CC180 => Coord {
                x: -self.x,
                y: -self.y,
            },
            Lockdown::CC270 => Coord {
                x: -self.y,
                y: self.x,
            },
        }
    }

    pub fn to_quad(&self) -> i32 {
        if self.x < crate::core::SIZE / 2 && self.y < crate::core::SIZE / 2 {
            0
        } else if self.x >= crate::core::SIZE / 2 && self.y < crate::core::SIZE / 2 {
            1
        } else if self.x < crate::core::SIZE / 2 && self.y >= crate::core::SIZE / 2 {
            2
        } else if self.x >= crate::core::SIZE / 2 && self.y >= crate::core::SIZE / 2 {
            3
        } else {
            -1
        }
    }

    pub fn in_boundary(&self) -> bool {
        if self.x >= BOARD_MIN && self.y >= BOARD_MIN && self.x < BOARD_MAX && self.y < BOARD_MAX {
            return true;
        }
        return false;
    }

    pub fn to_direction(&self) -> Direction {
        if *self == Coord::new(1, 0) {
            Direction::Right
        } else if *self == Coord::new(0, 1) {
            Direction::Up
        } else if *self == Coord::new(-1, 0) {
            Direction::Left
        } else if *self == Coord::new(0, -1) {
            Direction::Down
        } else {
            panic!("not meant for this usage")
        }
    }
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum Lockdown {
    Normal,
    CC90,
    CC180,
    CC270,
}

impl Direction {
    pub fn to_coord(&self) -> Coord {
        match *self {
            Direction::Right => Coord::new(1, 0),
            Direction::Up => Coord::new(0, 1),
            Direction::Left => Coord::new(-1, 0),
            Direction::Down => Coord::new(0, -1),
        }
    }
}

impl Add<&Direction> for Coord {
    type Output = Coord;

    fn add(self, d: &Direction) -> Coord {
        match *d {
            Direction::Up => Coord {
                x: self.x,
                y: self.y - 1,
            },
            Direction::Down => Coord {
                x: self.x,
                y: self.y + 1,
            },
            Direction::Left => Coord {
                x: self.x - 1,
                y: self.y,
            },
            Direction::Right => Coord {
                x: self.x + 1,
                y: self.y,
            },
        }
    }
}

impl Sub<&Coord> for Coord {
    type Output = HashMap<Direction, i32>;

    fn sub(self, c: &Coord) -> HashMap<Direction, i32> {
        let x_diff = self.x - c.x;
        let y_diff = self.y - c.y;
        let mut ret = HashMap::new();

        if x_diff > 0 {
            ret.insert(Direction::Right, x_diff);
        } else if x_diff < 0 {
            ret.insert(Direction::Left, -x_diff);
        }
        if y_diff < 0 {
            ret.insert(Direction::Up, -y_diff);
        } else if y_diff > 0 {
            ret.insert(Direction::Down, y_diff);
        }
        ret
    }
}

#[derive(Hash, PartialEq, Eq, Debug, Copy, Clone)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    pub fn new(b: u8) -> Direction {
        if b == b'h' {
            Direction::Left
        } else if b == b'l' {
            Direction::Right
        } else if b == b'k' {
            Direction::Up
        } else {
            Direction::Down
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord() {
        let c = Coord::new(1, 3);
        assert_eq!(c.x, 1);
        assert_eq!(c.y, 3);
    }

    #[test]
    fn test_boundary1() {
        let c = Coord::new(1, 3);
        assert_eq!(c.in_boundary(), true);
    }

    #[test]
    fn test_boundary2() {
        let c = Coord::new(7, 3);
        assert_eq!(c.in_boundary(), false);
    }

    #[test]
    fn test_sub() {
        let c1 = Coord::new(2, -1);
        let c2 = Coord::new(-1, 0);
        let res = c1 - &c2;
        let mut iter = res.iter();
        let r1 = iter.next().unwrap();
        assert_eq!(&Direction::Up == r1.0, &Direction::Right != r1.0);
        let r2 = iter.next().unwrap();
        assert_eq!(&3 == r2.1, &1 != r2.1);
    }
}
