mod action;
mod gen_map;
mod grid_coord;
mod status_code;

use action::Action;
use gen_map::get_rand_matrix;
use grid_coord::*;
use ndarray::{Array, Array3};
use std::collections::HashMap;

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub enum World {
    Humanity,
    Underworld,
}

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub enum Camp {
    Doctor,
    Plague,
}

#[derive(Hash, PartialEq, Eq, Debug, Copy, Clone)]
pub enum Stuff {
    Colony,
    Marker(u8),
}

#[derive(Hash, PartialEq, Eq, Debug)]
pub enum Phase {
    Setup0, // Humanity and Underworld
    Setup1, // Plague to put the 4 markers
    Setup2, // Put heroes on the board
    Setup3, // Doctor to put the marker on the compass
    Main(u32),
}

#[derive(Debug)]
pub struct Game {
    pub env: HashMap<Coord, World>,
    pub compass: HashMap<Camp, Coord>,
    pub hero: HashMap<(World, Camp), Coord>,
    pub stuff: HashMap<Coord, (Camp, Stuff)>,
    pub turn: Camp,
    pub phase: Phase,
}

pub const SIZE: i32 = 6;
const COMPASS_SIZE: usize = 5;
pub const DOCTOR_MARKER: i32 = 5;
pub const PLAGUE_MARKER: i32 = 4;

pub const MAX_MARKER: u8 = 5;

impl Game {
    pub fn init() -> Game {
        let mut g = Game {
            env: HashMap::new(),
            compass: HashMap::new(),
            hero: HashMap::new(),
            stuff: HashMap::new(),
            turn: Camp::Plague,
            phase: Phase::Setup0,
        };

        // Setup the env board for
        //    1. an empty game
        //    2. the SGF file provides no setup0 info
        for k in 0..=1 {
            for l in 0..=1 {
                let m = get_rand_matrix();
                for i in 0..SIZE / 2 {
                    for j in 0..SIZE / 2 {
                        let mut w = World::Humanity;
                        if m[i as usize][j as usize] == false {
                            w = World::Underworld;
                        }
                        g.env
                            .insert(Coord::new(k * (SIZE / 2) + i, l * (SIZE / 2) + j), w);
                    }
                }
            }
        }
        g
    }

    pub fn end(&self) -> bool {
        return self.end1() || self.end2();
    }

    fn end1(&self) -> bool {
        let t = self.turn;
        let dir = vec![
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ];

        fn check_path(
            stuff: &HashMap<Coord, (Camp, Stuff)>,
            t: Camp,
            dir: &Vec<Direction>,
            start_coord_fn: impl Fn(i32) -> Coord,
            end_check_fn: impl Fn(&Coord) -> bool,
        ) -> bool {
            let mut v = Vec::new();
            let mut visit: Vec<Vec<bool>> = vec![vec![false; SIZE as usize]; SIZE as usize];

            for i in 0..SIZE {
                let c = start_coord_fn(i);
                if let Some((cc, _)) = stuff.get(&c) {
                    if *cc == t {
                        v.push(c);
                        visit[c.x as usize][c.y as usize] = true;
                    }
                }
            }

            while let Some(c) = v.pop() {
                if c.x < 0 || c.y < 0 || c.x > SIZE || c.y > SIZE {
                    continue;
                }
                assert_eq!(visit[c.x as usize][c.y as usize], true);

                if end_check_fn(&c) {
                    return true;
                }

                for i in 0..4 {
                    let cc = c + &dir[i];
                    if let Some((camp, _)) = stuff.get(&cc) {
                        if *camp == t && !visit[cc.x as usize][cc.y as usize] {
                            v.push(cc);
                            visit[cc.x as usize][cc.y as usize] = true;
                        }
                    }
                }
            }
            false
        }

        if check_path(
            &self.stuff,
            t,
            &dir,
            |i| Coord::new(i, 0),
            |c| c.x == SIZE - 1,
        ) {
            return true;
        }

        if check_path(
            &self.stuff,
            t,
            &dir,
            |i| Coord::new(0, i),
            |c| c.y == SIZE - 1,
        ) {
            return true;
        }

        false
    }

    fn end2(&self) -> bool {
        return false;
    }

    pub fn encode(&self) -> (Array3<u8>, Array3<u8>) {
        // 9 entries: cursor, underworld, humanity, doctor hero, plague hero, doctor colony, plague colony, doctor marker, plague marker
        let mut a =
            Array::from_shape_fn((SIZE as usize, SIZE as usize, 9 as usize), |(_, _, _)| {
                0 as u8
            });

        // In the core game engine, we don't maintain cursor information. The concept belongs to UI, not here. We just reserve a space for the UI to return the value.
        for i in 0..SIZE as usize {
            for j in 0..SIZE as usize {
                let c = Coord::new(i.try_into().unwrap(), j.try_into().unwrap());
                if self.env.get(&c).unwrap() == &World::Underworld {
                    a[[i, j, 1 /*Underworld*/]] = 1;
                } else {
                    a[[i, j, 2 /*Humanity*/]] = 1;
                }
                match self.stuff.get(&c) {
                    None => {}
                    Some((Camp::Doctor, Stuff::Colony)) => {
                        a[[i, j, 5 /*Doctor Colony*/]] = 1;
                    }
                    Some((Camp::Plague, Stuff::Colony)) => {
                        a[[i, j, 6 /*Plague Colony*/]] = 1;
                    }
                    Some((Camp::Doctor, Stuff::Marker(x))) => {
                        a[[i, j, 7 /*Doctor Marker*/]] = *x;
                    }
                    Some((Camp::Plague, Stuff::Marker(x))) => {
                        a[[i, j, 8 /*Plague Marker*/]] = *x;
                    }
                }
            }
        }
        for ((_, camp), c) in self.hero.iter() {
            if *camp == Camp::Doctor {
                a[[c.y as usize, c.x as usize, 3 /*Doctor Hero*/]] = 1;
            } else {
                a[[c.y as usize, c.x as usize, 4 /*Plague Hero*/]] = 1;
            }
        }

        // 3 entries: cursor, doctor, plague
        let mut b = Array::from_shape_fn((COMPASS_SIZE, COMPASS_SIZE, 3 as usize), |(_, _, _)| {
            0 as u8
        });
        for (camp, c) in self.compass.iter() {
            if *camp == Camp::Doctor {
                b[[
                    (c.y + COMPASS_OFFSET.y) as usize,
                    (c.x + COMPASS_OFFSET.x) as usize,
                    1, /*Doctor Marker*/
                ]] = 1;
            } else {
                b[[
                    (c.y + COMPASS_OFFSET.y) as usize,
                    (c.x + COMPASS_OFFSET.x) as usize,
                    2, /*Plague Marker*/
                ]] = 1;
            }
        }
        (a, b)
    }

    pub fn add_marker(&mut self, c: &Coord, camp: &Camp) {
        match self.stuff.get(c) {
            None => {
                self.stuff.insert(*c, (*camp, Stuff::Marker(1)));
            }
            Some((cc, Stuff::Marker(x))) => {
                if cc == camp {
                    if *x < MAX_MARKER {
                        self.stuff.insert(*c, (*camp, Stuff::Marker(*x + 1)));
                    } else {
                        self.stuff.insert(*c, (*camp, Stuff::Colony));
                    }
                } else {
                    if *x > 1 {
                        self.stuff
                            .insert(*c, (self.opposite(*camp), Stuff::Marker(*x - 1)));
                    } else {
                        self.stuff.remove(c);
                    }
                }
            }
            Some((_, Stuff::Colony)) => panic!("Cannot add marker to Colony"),
        }
    }

    // This is a stronger interpretation than the rule book.
    // The rule doesn't state if it counts as lockdown state when
    // the Doctor occupies the center of compass at the beginning.
    pub fn lockdown(&self) -> bool {
        let c = self.compass.get(&Camp::Doctor).unwrap();
        if c.x == ORIGIN.x && c.y == ORIGIN.y {
            return true;
        }
        return false;
    }

    pub fn set_compass(&mut self, t: Camp, c: Coord) {
        self.compass.insert(t, c);
    }

    /// Check if the setup in setup3 is legal
    pub fn apply_move(&mut self, s: &Vec<String>) {
        let c_start = *self.compass.get(&self.opposite(self.turn)).unwrap();
        let c_end = self.sgf_to_compass(&s[0]);
        let mut ls = Lockdown::Normal;
        if c_end.x == 0 && c_end.y == 0 && self.turn == Camp::Doctor {
            let l = s.len();
            match s[l - 1].as_str() {
                "2" => {
                    ls = Lockdown::CC90;
                }
                "3" => {
                    ls = Lockdown::CC180;
                }
                "4" => {
                    ls = Lockdown::CC270;
                }
                _ => {}
            }
        }
        self.set_compass(self.turn, c_end.lockdown(ls));
        self.set_compass(self.opposite(self.turn), c_start.lockdown(ls));

        let mov = c_end - &c_start;
        let mut index: usize = 1;
        for (_, i) in mov.iter() {
            index = index + *i as usize;
        }

        // update hero
        let c_to = self.sgf_to_env(&s[index]);
        let w = self.env.get(&c_start).unwrap();
        let tuple = (*w, self.turn);
        self.hero.insert(tuple, c_to);

        // update marker
        let t = self.turn;
        // note that the final string is lockdown indicator for Doctor
        let end = if t == Camp::Doctor {
            s.len() - 1
        } else {
            s.len()
        };
        for i in index + 1..end {
            let c = self.sgf_to_env(&s[i]);
            self.add_marker(&c, &t);
        }
    }

    pub fn next(&mut self) {
        // Next?
        self.switch();
        if let Phase::Main(x) = self.phase {
            self.phase = Phase::Main(x + 1);
        }
        return;
    }

    pub fn commit_action(&mut self, a: &Action) {
        if a.lockdown != Lockdown::Normal {
            let c_start = *self.compass.get(&Camp::Plague).unwrap();
            self.set_compass(Camp::Plague, c_start.lockdown(a.lockdown));
        }
        self.set_compass(self.turn, a.compass.unwrap());
        self.hero
            .insert((a.world.unwrap(), self.turn), a.hero.unwrap());
        let m = a.markers.clone();
        let t = self.turn;
        for c in m.iter() {
            self.add_marker(c, &t);
        }
    }

    /// Check if the setup in setup3 is legal
    pub fn is_illegal_setup3(&self) -> bool {
        for (_, coord) in self.compass.iter() {
            if coord.x > 1 || coord.x < -1 || coord.y > 1 || coord.y < -1 {
                return true;
            }
        }
        return false;
    }

    /// Check if the setup in setup2 is legal
    pub fn is_illegal_setup2(&self) -> bool {
        if self.hero.len() != 4 && self.hero.len() != 2 {
            return true;
        }
        match self.hero.len() {
            4 => {
                // 4 heroes are placed, it must be legal because there are exactly 4 combinations of camp and world
                return false;
            }
            2 => {
                // temporary
                return false;
            }
            _ => {
                // weird state
                return true;
            }
        }
    }

    /// Check if the setup in setup1 is legal
    pub fn is_illegal_setup1(&self) -> bool {
        let mut row: [bool; crate::core::SIZE as usize] = [false; crate::core::SIZE as usize];
        let mut col: [bool; crate::core::SIZE as usize] = [false; crate::core::SIZE as usize];
        if self.stuff.len() != 4 {
            return true;
        }
        for c in self.stuff.iter() {
            let coord = c.0;
            if row[coord.x as usize] || col[coord.y as usize] {
                return true;
            }
            row[coord.x as usize] = true;
            col[coord.y as usize] = true;
        }
        return false;
    }

    pub fn sgf_to_compass(&self, s: &String) -> Coord {
        // row major, x as row index and y as column index
        // ghijk
        let x: i32 = s.chars().nth(1).unwrap() as i32 - 'i' as i32;
        let y: i32 = s.chars().nth(0).unwrap() as i32 - 'i' as i32;
        Coord::new(x, y)
    }

    pub fn sgf_to_env(&self, s: &String) -> Coord {
        // row major, x as row index and y as column index
        // abcdef
        let x: i32 = s.chars().nth(1).unwrap() as i32 - 'a' as i32;
        let y: i32 = s.chars().nth(0).unwrap() as i32 - 'a' as i32;
        Coord::new(x, y)
    }

    pub fn opposite(&self, c: Camp) -> Camp {
        match c {
            Camp::Doctor => Camp::Plague,
            Camp::Plague => Camp::Doctor,
        }
    }

    pub fn switch(&mut self) {
        self.turn = self.opposite(self.turn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_end1_1() {
        let mut g = Game::init();
        g.turn = Camp::Doctor;
        g.add_marker(&Coord::new(0, 0), &Camp::Doctor);
        g.add_marker(&Coord::new(1, 0), &Camp::Doctor);
        g.add_marker(&Coord::new(2, 0), &Camp::Doctor);
        g.add_marker(&Coord::new(3, 0), &Camp::Doctor);
        g.add_marker(&Coord::new(4, 0), &Camp::Doctor);
        g.add_marker(&Coord::new(5, 0), &Camp::Doctor);
        assert_eq!(g.end1(), true);
    }

    #[test]
    fn test_end1_2() {
        let mut g = Game::init();
        g.turn = Camp::Doctor;
        g.add_marker(&Coord::new(0, 0), &Camp::Doctor);
        g.add_marker(&Coord::new(1, 0), &Camp::Doctor);
        g.add_marker(&Coord::new(1, 1), &Camp::Doctor);
        g.add_marker(&Coord::new(1, 2), &Camp::Doctor);
        g.add_marker(&Coord::new(2, 2), &Camp::Doctor);
        for i in 0..MAX_MARKER {
            g.add_marker(&Coord::new(3, 2), &Camp::Doctor);
        }
        g.add_marker(&Coord::new(3, 3), &Camp::Doctor);
        g.add_marker(&Coord::new(3, 4), &Camp::Doctor);
        g.add_marker(&Coord::new(3, 5), &Camp::Doctor);
        assert_eq!(g.end1(), true);
    }

    #[test]
    fn test_end1_3() {
        let mut g = Game::init();
        g.turn = Camp::Doctor;
        g.add_marker(&Coord::new(0, 0), &Camp::Doctor);
        g.add_marker(&Coord::new(1, 0), &Camp::Doctor);
        g.add_marker(&Coord::new(1, 1), &Camp::Doctor);
        g.add_marker(&Coord::new(1, 2), &Camp::Doctor);
        g.add_marker(&Coord::new(2, 2), &Camp::Doctor);
        g.add_marker(&Coord::new(3, 3), &Camp::Doctor);
        g.add_marker(&Coord::new(3, 4), &Camp::Doctor);
        g.add_marker(&Coord::new(3, 5), &Camp::Doctor);
        assert_eq!(g.end1(), false);
    }
}
