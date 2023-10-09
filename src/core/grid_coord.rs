use std::collections::HashMap;
use std::ops::{Add, Sub};

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
        let y: u32 = (base + self.y) as u32;
        let x: u32 = (base + self.x) as u32;
        s.push(std::char::from_u32(y).unwrap());
        s.push(std::char::from_u32(x).unwrap());
        return s;
    }

    pub fn map_to_sgf(&self) -> String {
        let mut s = String::new();
        let base: i32 = 'i' as i32;
        let y: u32 = (base + self.y) as u32;
        let x: u32 = (base + self.x) as u32;
        s.push(std::char::from_u32(y).unwrap());
        s.push(std::char::from_u32(x).unwrap());
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
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum Lockdown {
    Normal,
    CC90,
    CC180,
    CC270,
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
}
