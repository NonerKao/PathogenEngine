use ndarray::{Array, Array3};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::collections::HashMap;
use std::ops::{Add, Sub};

const ORIGIN: Coord = Coord { x: 0, y: 0 };
const COMPASS_OFFSET: Coord = Coord { x: 2, y: 2 };

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

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum Lockdown {
    Normal,
    CC90,
    CC180,
    CC270,
}

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

    pub fn compass_to_sgf(&self) -> String {
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

#[derive(Debug)]
pub struct Game {
    pub env: HashMap<Coord, World>,
    pub compass: HashMap<Camp, Coord>,
    pub hero: HashMap<(World, Camp), Coord>,
    pub stuff: HashMap<Coord, (Camp, Stuff)>,
    pub turn: Camp,
    pub phase: Phase,
    pub action: Action,
}

#[derive(Debug)]
pub struct Action {
    compass: Option<Coord>,
    pub lockdown: Lockdown,
    hero: Option<Coord>,
    world: Option<World>,
    pub restriction: HashMap<Direction, i32>,
    pub steps: i32,
    trajectory: Vec<Coord>,
    markers: Vec<Coord>,
}

impl Action {
    pub fn new() -> Action {
        return Action {
            compass: None,
            lockdown: Lockdown::Normal,
            hero: None,
            world: None,
            restriction: HashMap::new(),
            trajectory: Vec::new(),
            markers: Vec::new(),
            steps: 0,
        };
    }
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
            action: Action::new(),
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

    pub fn flush_action(&mut self) {
        self.action = Action::new();
    }

    pub fn commit_action(&mut self) {
        if self.action.lockdown != Lockdown::Normal {
            let c_start = *self.compass.get(&Camp::Plague).unwrap();
            self.set_compass(Camp::Plague, c_start.lockdown(self.action.lockdown));
            self.action.lockdown = Lockdown::Normal;
        }
        self.set_compass(self.turn, self.action.compass.unwrap());
        self.hero.insert(
            (self.action.world.unwrap(), self.turn),
            self.action.hero.unwrap(),
        );
        let m = self.action.markers.clone();
        let t = self.turn;
        for c in m.iter() {
            self.add_marker(c, &t);
        }

        self.flush_action();
    }

    pub fn check_action_compass(&mut self, c: Coord) -> Result<String, String> {
        let mut ret = Ok("".to_string());
        if *self.compass.get(&self.opposite(self.turn)).unwrap() == c {
            return Err("Collide with opponent".to_string());
        }
        if *self.compass.get(&self.turn).unwrap() == c {
            self.next();
            return Err("Skip".to_string());
        }
        if (self.lockdown() && self.turn == Camp::Plague) || self.turn == Camp::Doctor {
            // Plague cannot outbreak when lockdown
            if c.x < -1 || c.x > 1 || c.y < -1 || c.y > 1 {
                return Err("Exceed valid compass area".to_string());
            }
        }
        if *self.compass.get(&self.turn).unwrap() == c {
            match self.phase {
                Phase::Main(n) => {
                    if n != 1 {
                        ret = Ok("Skip this move".to_string());
                    } else {
                        // XXX: What if it cannot? Should we handle this here?
                        return Err("Plague must start".to_string());
                    }
                }
                _ => {}
            }
        }
        self.action.compass = Some(c);
        let op = self.compass.get(&self.opposite(self.turn)).unwrap();
        self.action.restriction = c - op;
        if self.action.steps != 0 {
            panic!("{:?}:{:?}==", self.action.steps, self.action.restriction);
        }
        for (_, i) in self.action.restriction.iter() {
            self.action.steps += *i;
        }
        return ret;
    }

    pub fn check_action_hero(&mut self, c: Coord) -> Result<String, String> {
        let hh = *self.hero.get(&(World::Humanity, self.turn)).unwrap();
        let hu = *self.hero.get(&(World::Underworld, self.turn)).unwrap();
        if c != hh && c != hu {
            return Err("Invalid character".to_string());
        } else if c == hh {
            self.action.world = Some(World::Humanity);
        } else {
            self.action.world = Some(World::Underworld);
        }
        self.action.hero = Some(c);
        self.action.trajectory.push(c);
        return Ok("".to_string());
    }

    pub fn check_action_lockdown(&mut self, ld: Lockdown) -> Result<String, String> {
        let mut nr: HashMap<Direction, i32> = HashMap::new();
        let u = self.action.restriction.get(&Direction::Up);
        let r = self.action.restriction.get(&Direction::Right);
        let d = self.action.restriction.get(&Direction::Down);
        let l = self.action.restriction.get(&Direction::Left);

        match ld {
            Lockdown::CC90 => {
                if r != None {
                    nr.insert(Direction::Up, *r.unwrap());
                }
                if d != None {
                    nr.insert(Direction::Right, *d.unwrap());
                }
                if l != None {
                    nr.insert(Direction::Down, *l.unwrap());
                }
                if u != None {
                    nr.insert(Direction::Left, *u.unwrap());
                }
                self.action.restriction = nr;
            }
            Lockdown::CC180 => {
                if d != None {
                    nr.insert(Direction::Up, *d.unwrap());
                }
                if l != None {
                    nr.insert(Direction::Right, *l.unwrap());
                }
                if u != None {
                    nr.insert(Direction::Down, *u.unwrap());
                }
                if r != None {
                    nr.insert(Direction::Left, *r.unwrap());
                }
                self.action.restriction = nr;
            }
            Lockdown::CC270 => {
                if l != None {
                    nr.insert(Direction::Up, *l.unwrap());
                }
                if u != None {
                    nr.insert(Direction::Right, *u.unwrap());
                }
                if r != None {
                    nr.insert(Direction::Down, *r.unwrap());
                }
                if d != None {
                    nr.insert(Direction::Left, *d.unwrap());
                }
                self.action.restriction = nr;
            }
            _ => {}
        }
        self.action.lockdown = ld;
        return Ok("".to_string());
    }

    pub fn check_action_step(&mut self, to: Coord) -> Result<String, String> {
        // impossible number as a implicit assertion
        let mut from = Coord::new(-999, -999);
        if let Some(x) = self.action.trajectory.last() {
            from = *x;
        }
        let dd = to - &from;
        let mov = self.action.restriction.clone();
        let w = self.action.world.unwrap();

        // more than 1 directions?
        // this can happen when loading a broken SGF file
        if dd.len() != 1 {
            return Err("Invalid move".to_string());
        }

        for (d, _) in dd.iter() {
            if mov.get(d) == None {
                return Err("Invalid direction of the move".to_string());
            }

            let mut c = from;
            loop {
                c = c + d;
                match self.env.get(&c) {
                    Some(x) => {
                        if *x == w {
                            if c != to {
                                return Err("Not consecutive move".to_string());
                            } else {
                                break;
                            }
                        } else if c == to {
                            return Err("Cannot cross world".to_string());
                        }
                    }
                    None => {
                        return Err("Invalid move along the direction".to_string());
                    }
                }
            }
        }

        let camp = self.opposite(self.turn);
        match self.stuff.get(&to) {
            Some((c, Stuff::Colony)) => {
                if *c == camp {
                    // XXX: this is not tested yet
                    return Err("Cannot walk through or stop at the opponent's colony".to_string());
                }
            }
            _ => {}
        }

        self.action.trajectory.push(to);
        if self.action.trajectory.len() > self.action.steps.try_into().unwrap() {
            // Final step: No collision?
            let op = *self.hero.get(&(w, camp)).unwrap();
            if op == to {
                return Err("Collide with opponent's hero".to_string());
            }
        }

        self.action.hero = Some(to);
        return Ok("".to_string());
    }

    pub fn check_action_marker(&mut self, t: Coord) -> Result<String, String> {
        let quota = if self.turn == Camp::Doctor {
            DOCTOR_MARKER
        } else {
            PLAGUE_MARKER
        };

        let op = self.opposite(self.turn);
        if let Some(oph) = self.hero.get(&(self.action.world.unwrap(), op)) {
            if *oph == t {
                return Err("Opponent is here".to_string());
            }
        }

        if !self.action.trajectory.contains(&t) {
            return Err("Not in the trajectory".to_string());
        }

        self.action.markers.push(t);
        if self.action.markers.len() > quota.try_into().unwrap() {
            return Err("Cannot over-place markers".to_string());
        } else if self.action.markers.len() == quota.try_into().unwrap() {
            // when all markers are given ... most checks are here
            // 1. if there is overflow
            // 2. (Doctor) if any plagues are ignored
            // 3. (Plague) if distributed evenly
            let last = self.action.trajectory.len() - 1;
            self.action.trajectory.remove(last);

            // This sort-and-traverse was for Plague only because it would be easier to calculate max/min,
            // but now we need to check if any marker overflows to Colony. Move Plague check here as well
            self.action.markers.sort_by(|a, b| {
                let na = a.x + SIZE * a.y;
                let nb = b.x + SIZE * b.y;
                nb.cmp(&na)
            });
            let mut cur = 1;
            let m = &self.action.markers;
            let t = &mut self.action.trajectory;

            // both side shouldn't count the grid occupied by an opponent
            t.retain(|&y| {
                let hh = *self.hero.get(&(World::Humanity, op)).unwrap();
                let hu = *self.hero.get(&(World::Underworld, op)).unwrap();
                y != hh && y != hu
            });
            let max = (PLAGUE_MARKER as f64 / t.len() as f64).ceil() as u8;
            let min = (PLAGUE_MARKER as f64 / t.len() as f64).floor() as u8;
            for i in 1..=m.len() {
                // check the total markers under two conditions
                // 1. all markers reside in the same position, or
                // 2. there are different markers in this move
                if i == m.len() || m[i - 1] != m[i] {
                    if let Some((c, Stuff::Marker(x))) = self.stuff.get(&m[i - 1]) {
                        if *c == self.turn && MAX_MARKER < x + cur {
                            return Err("Overflow markers".to_string());
                        } else if *c == op && MAX_MARKER < cur - x {
                            // yes, this is not possible in the official rule,
                            // but for future variant, it may be more flexible to add this here
                            return Err("Overflow markers".to_string());
                        }
                    }
                    if self.turn == Camp::Plague && cur != max && cur != min {
                        return Err("Not even distributed".to_string());
                    }
                    cur = 1;
                } else {
                    cur += 1;
                }
            }

            if self.turn == Camp::Doctor {
                // need to clean the plague first
                // priorities the marker placement
                t.sort_by(|a, b| {
                    let mut a_sick = false;
                    let mut b_sick = false;
                    if let Some((Camp::Plague, Stuff::Marker(_))) = self.stuff.get(a) {
                        a_sick = true;
                    }
                    if let Some((Camp::Plague, Stuff::Marker(_))) = self.stuff.get(b) {
                        b_sick = true;
                    }
                    b_sick.cmp(&a_sick)
                });
                // on encountering plagues, cure them
                // so, are they cured?
                let mut m = self.action.markers.clone();
                for c in t.iter() {
                    let before = m.len();
                    if before == 0 {
                        break;
                    }
                    if let Some((Camp::Plague, Stuff::Marker(x))) = self.stuff.get(c) {
                        m.retain(|&y| y != *c);
                        let after = m.len();
                        if before - after < *x as usize && after != 0 {
                            return Err("Mustn't ignore encountered plagues".to_string());
                        }
                    }
                }
            }
        }
        return Ok("".to_string());
    }
    /// Check if the move is legal
    pub fn check_move(&mut self, s: &Vec<String>) -> Result<String, String> {
        // compass move OK?
        let ccac = self.sgf_to_compass(&s[0]);
        self.check_action_compass(ccac)?;

        // start position OK?
        let ccah = self.sgf_to_env(&s[1]);
        self.check_action_hero(ccah)?;

        // check lockdown state
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
        self.check_action_lockdown(ls)?;

        for i in 2..(self.action.steps as usize) + 2 {
            let c = self.sgf_to_env(&s[i]);
            self.check_action_step(c)?;
        }

        let end = if self.turn == Camp::Doctor {
            s.len() - 1
        } else {
            s.len()
        };
        for i in (self.action.steps as usize) + 2..end {
            self.check_action_marker(self.sgf_to_env(&s[i]))?;
        }

        return Ok("".to_string());
    }

    // eventually this will be for loading SGF only
    pub fn check_and_apply_move(&mut self, s: &Vec<String>) -> Result<String, String> {
        self.check_move(s)?;
        // the move looks OK! Apply it...

        self.apply_move(s);
        self.flush_action();
        self.next();
        return Ok("".to_string());
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

fn get_rand_matrix() -> Vec<Vec<bool>> {
    let vec_of_bools = vec![
        vec![true, true, false],
        vec![true, false, true],
        vec![false, true, true],
        vec![true, false, false],
        vec![false, true, false],
        vec![false, false, true],
    ];

    let mut rng = thread_rng();

    // Randomly select two distinct elements from the vector
    let random_vecs: Vec<_> = vec_of_bools.choose_multiple(&mut rng, 3).collect();

    // Assign the two random elements to variables
    let random_vec1 = random_vecs[0].clone();
    let random_vec2 = random_vecs[1].clone();
    let mut random_vec3 = random_vecs[2].clone();

    for i in 0..random_vec1.len() {
        if random_vec1[i] == random_vec2[i] {
            random_vec3[i] = !random_vec1[i];
        }
    }

    let matrix = vec![random_vec1, random_vec2, random_vec3];

    // Randomly transform the matrix
    random_transform(matrix)
}

// Rotate the matrix clockwise by 90 degrees
fn rotate_matrix(matrix: &Vec<Vec<bool>>) -> Vec<Vec<bool>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut rotated = vec![vec![false; rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            rotated[j][rows - 1 - i] = matrix[i][j];
        }
    }
    rotated
}

// Randomly transform the matrix by rotating it or mirroring it horizontally/vertically
fn random_transform(matrix1: Vec<Vec<bool>>) -> Vec<Vec<bool>> {
    let mut rng = thread_rng();

    let operation1 = rng.gen_range(0..3);
    let matrix2 = match operation1 {
        0 => matrix1,
        1 => matrix1.into_iter().rev().collect(), // Vertical mirror
        2 => matrix1
            .into_iter()
            .map(|row| row.into_iter().rev().collect())
            .collect(), // Horizontal mirror
        _ => unreachable!(),
    };

    let operation2 = rng.gen_range(0..4);

    match operation2 {
        0 => matrix2,                                                 // No rotation
        1 => rotate_matrix(&matrix2),                                 // Rotate 90 degrees
        2 => rotate_matrix(&rotate_matrix(&matrix2)),                 // Rotate 180 degrees
        3 => rotate_matrix(&rotate_matrix(&rotate_matrix(&matrix2))), // Rotate 270 degrees
        _ => unreachable!(),
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

    /*#[test]
    fn test_colony() {
        let mut g = Game::init(None);
        let p00 = Coord::new(0, 0);
        let p01 = Coord::new(1, 0);
        g.stuff.insert(p00, (Camp::Doctor, Stuff::Colony));
        g.hero.insert((World::Humanity, Camp::Plague), p01);
        g.env.insert(p00, World::Humanity);
        g.env.insert(p01, World::Humanity);

        let mut s = Vec::new();
        let mov = p00 - &p01;
        s.push("ii".to_string());
        s.push("ab".to_string());
        s.push("aa".to_string());
        assert_eq!(
            g.check_trajectory(1, &s, &mov),
            Err("Cannot walk through or stop at the opponent's colony".to_string())
        );
    }*/

    #[test]
    fn test_env() {
        let g = Game::init();
        assert_eq!(g.env.get(&Coord::new(3, 4)).unwrap(), &World::Humanity);
    }
}
