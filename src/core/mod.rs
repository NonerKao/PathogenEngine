pub mod action;
mod gen_map;
pub mod grid_coord;
mod setup;
pub mod status_code;
pub mod tree;

use action::Action;
use gen_map::get_rand_matrix;
use grid_coord::*;
use ndarray::{Array, Array3};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use tree::TreeNode;

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub enum World {
    Humanity,
    Underworld,
}

impl World {
    const COUNT: usize = 2;
}

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub enum Camp {
    Doctor,
    Plague,
}

impl Camp {
    const COUNT: usize = 2;
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
    Setup2, // Put characteres on the board
    Setup3, // Doctor to put the marker on the map
    Main(u32),
}

trait SGFCoord {
    fn to_map(&self) -> Coord;
    fn to_env(&self) -> Coord;
}

impl SGFCoord for str {
    fn to_map(&self) -> Coord {
        // row major, x as row index and y as column index
        // ghijk
        let x: i32 = self.chars().nth(1).unwrap() as i32 - 'i' as i32;
        let y: i32 = self.chars().nth(0).unwrap() as i32 - 'i' as i32;
        Coord::new(x, y)
    }

    fn to_env(&self) -> Coord {
        // row major, x as row index and y as column index
        // abcdef
        let x: i32 = self.chars().nth(1).unwrap() as i32 - 'a' as i32;
        let y: i32 = self.chars().nth(0).unwrap() as i32 - 'a' as i32;
        Coord::new(x, y)
    }
}

#[derive(Debug)]
pub struct Game {
    pub env: HashMap<Coord, World>,
    pub map: HashMap<Camp, Coord>,
    pub character: HashMap<(World, Camp), Coord>,
    pub stuff: HashMap<Coord, (Camp, Stuff)>,
    pub turn: Camp,
    pub phase: Phase,
    // Given history, given empty tree for recording, or None
    pub history: Rc<RefCell<TreeNode>>,
}

pub const SIZE: i32 = 6;
const COMPASS_SIZE: usize = 5;
pub const QUAD_SIZE: usize = 4;
pub const DOCTOR_MARKER: i32 = 5;
pub const PLAGUE_MARKER: i32 = 4;

pub const MAX_MARKER: u8 = 5;
const SETUP1_MARKER: usize = 4;

impl Game {
    pub fn init(file: Option<Rc<RefCell<TreeNode>>>) -> Game {
        let s = String::from(
            "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen]",
        );
        let mut iter = s.trim().chars().peekable();
        // The default SGF history starts with a common game-info node
        let mut g = Game {
            env: HashMap::new(),
            map: HashMap::new(),
            character: HashMap::new(),
            stuff: HashMap::new(),
            turn: Camp::Plague,
            phase: Phase::Setup0,
            history: TreeNode::new(&mut iter, None),
        };

        fn load_history(t: &TreeNode, is_front: bool, g: &mut Game) {
            if !is_front {
                return;
            }
            match t.checkpoint() {
                "Setup0" => {
                    if g.phase == Phase::Setup0 {
                        let mut h: Vec<String> = Vec::new();
                        let mut u: Vec<String> = Vec::new();
                        t.get_general("AW".to_string(), &mut h);
                        t.get_general("AB".to_string(), &mut u);
                        for c in h.iter() {
                            g.env.insert(c.to_env(), World::Humanity);
                        }
                        for c in u.iter() {
                            g.env.insert(c.to_env(), World::Underworld);
                        }
                        if g.is_setup0_done() {
                            g.phase = Phase::Setup1;
                        }
                    } else {
                        panic!("Ex12");
                    }
                }
                "Setup1" => {
                    if g.phase == Phase::Setup0 {
                        panic!("Ex13");
                    } else if g.phase == Phase::Setup1 {
                        let mut m: Vec<String> = Vec::new();
                        t.get_general("AB".to_string(), &mut m);
                        for c in m.iter() {
                            g.add_marker(&c.as_str().to_env(), &Camp::Plague);
                        }
                        if g.is_setup1_done() {
                            if g.is_illegal_setup1() {
                                panic!("Ex11");
                            }
                            g.phase = Phase::Setup2;
                        }
                    } else {
                        panic!("Ex10");
                    }
                }
                "Setup2" => {
                    if g.phase == Phase::Setup0 {
                        panic!("Ex13");
                    } else if g.phase == Phase::Setup1 {
                        panic!("Ex14");
                    } else if g.phase == Phase::Setup2 {
                        let mut s = String::new();
                        g.switch();
                        match g.turn {
                            Camp::Plague => {
                                t.get_value("AB".to_string(), &mut s);
                            }
                            Camp::Doctor => {
                                t.get_value("AW".to_string(), &mut s);
                            }
                        }
                        let c1 = s.as_str().to_env();
                        g.character.insert((*g.env.get(&c1).unwrap(), g.turn), c1);
                        if g.is_illegal_order_setup2() {
                            // Since the `g.switch` above ensures the execution order
                            // in this phase, this error is not possible to be triggered.
                            panic!("Ex18");
                        }
                        if g.is_illegal_position_setup2() {
                            panic!("Ex19");
                        }
                        if g.is_setup2_done() {
                            g.phase = Phase::Setup3;
                        }
                    } else {
                        panic!("Ex17");
                    }
                }
                "Setup3" => {
                    if g.phase == Phase::Setup0 {
                        panic!("Ex13");
                    } else if g.phase == Phase::Setup1 {
                        panic!("Ex14");
                    } else if g.phase == Phase::Setup2 {
                        panic!("Ex17");
                    } else if g.phase == Phase::Setup3 {
                        let mut s = String::new();
                        // In theory this doesn't matter, but the existence of
                        // this turn-aware setting may benefit future clients
                        g.switch();
                        t.get_value("AW".to_string(), &mut s);
                        let c = s.as_str().to_map();
                        g.map.insert(Camp::Doctor, c);
                        // check the start()::Setup3 for why this is done here
                        g.map.insert(Camp::Plague, c);
                        g.phase = Phase::Main(1);
                        if g.is_illegal_setup3() {
                            panic!("Ex1b");
                        }
                    } else {
                        panic!("Ex1a");
                    }
                }
                _ => {}
            }
            /*
            match g.phase {
                Phase::Setup3 => {
                }
                Phase::Main(_) => {
                    let mut m: Vec<String> = Vec::new();
                    t.get_general("W".to_string(), &mut m);
                    t.get_general("B".to_string(), &mut m);
                    t.get_general("C".to_string(), &mut m);
                    t.get_general("IT".to_string(), &mut m);
                    match g.check_and_apply_move(&m) {
                        Ok(_) => {}
                        Err(x) => {
                            panic!("{}", x);
                        }
                    }
                }
                _ => {}
            }
            */
        }
        match file {
            Some(x) => {
                x.borrow().traverse(&load_history, &mut g);
            }
            _ => {}
        }
        if g.phase == Phase::Setup0 {
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
                            let c = Coord::new(k * (SIZE / 2) + i, l * (SIZE / 2) + j);
                            g.env.insert(c, w);
                        }
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
        let t = self.turn;
        if QUAD_SIZE
            == self
                .stuff
                .iter()
                .filter(|&(_, (camp, stuff))| *camp == t && *stuff == Stuff::Colony)
                .count()
        {
            return true;
        }
        false
    }

    pub fn encode(&self) -> (Array3<u8>, Array3<u8>) {
        // 9 entries: cursor, underworld, humanity, doctor character, plague character, doctor colony, plague colony, doctor marker, plague marker
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
        for ((_, camp), c) in self.character.iter() {
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
        for (camp, c) in self.map.iter() {
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

    // XXX: Need to add some restriction that each qudrant can
    //      have one colony at most
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
    // the Doctor occupies the center of map at the beginning.
    pub fn lockdown(&self) -> bool {
        let c = self.map.get(&Camp::Doctor).unwrap();
        if c.x == ORIGIN.x && c.y == ORIGIN.y {
            return true;
        }
        return false;
    }

    pub fn set_map(&mut self, t: Camp, c: Coord) {
        self.map.insert(t, c);
    }

    /// Check if the setup in setup3 is legal
    pub fn apply_move(&mut self, s: &Vec<String>) {
        let c_start = *self.map.get(&self.opposite(self.turn)).unwrap();
        let c_end = s[0].as_str().to_map();
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
        self.set_map(self.turn, c_end.lockdown(ls));
        self.set_map(self.opposite(self.turn), c_start.lockdown(ls));

        let mov = c_end - &c_start;
        let mut index: usize = 1;
        for (_, i) in mov.iter() {
            index = index + *i as usize;
        }

        // update character
        let c_to = s[index].as_str().to_env();
        let w = self.env.get(&c_start).unwrap();
        let tuple = (*w, self.turn);
        self.character.insert(tuple, c_to);

        // update marker
        let t = self.turn;
        // note that the final string is lockdown indicator for Doctor
        let end = if t == Camp::Doctor {
            s.len() - 1
        } else {
            s.len()
        };
        for i in index + 1..end {
            let c = s[i].as_str().to_env();
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
            let c_start = *self.map.get(&Camp::Plague).unwrap();
            self.set_map(Camp::Plague, c_start.lockdown(a.lockdown));
        }
        self.set_map(self.turn, a.map.unwrap());
        self.character
            .insert((a.world.unwrap(), self.turn), a.character.unwrap());
        let m = a.markers.clone();
        let t = self.turn;
        for c in m.iter() {
            self.add_marker(c, &t);
        }
    }

    pub fn is_setup0_done(&self) -> bool {
        let s2 = SIZE * SIZE;
        if self.env.len() > s2 as usize {
            panic!("Ex16");
        }
        return self.env.len() == s2 as usize;
    }

    pub fn is_setup1_done(&self) -> bool {
        if self.stuff.len() > SETUP1_MARKER {
            panic!("Ex15");
        }
        return self.stuff.len() == SETUP1_MARKER;
    }

    /// Check if the setup in setup1 is legal
    pub fn is_illegal_setup1(&self) -> bool {
        let mut row: [bool; crate::core::SIZE as usize] = [false; crate::core::SIZE as usize];
        let mut col: [bool; crate::core::SIZE as usize] = [false; crate::core::SIZE as usize];
        let mut quad: [bool; crate::core::QUAD_SIZE] = [false; crate::core::QUAD_SIZE];
        for c in self.stuff.iter() {
            let coord = c.0;
            let q = coord.to_quad();
            if row[coord.x as usize] || col[coord.y as usize] || quad[q as usize] {
                return true;
            }
            row[coord.x as usize] = true;
            col[coord.y as usize] = true;
            quad[q as usize] = true;
        }
        return false;
    }

    pub fn is_setup2_done(&self) -> bool {
        return self.character.len() == Camp::COUNT * World::COUNT;
    }

    /// Check if the setup in setup2 is legal
    pub fn is_illegal_order_setup2(&self) -> bool {
        let dc = self
            .character
            .iter()
            .filter(|&((_, camp), _)| *camp == Camp::Doctor)
            .count();
        let pc = self
            .character
            .iter()
            .filter(|&((_, camp), _)| *camp == Camp::Plague)
            .count();
        match self.turn {
            Camp::Plague => {
                if dc == 1 && pc == 1 {
                    return false;
                } else if dc == 2 && pc == 2 {
                    return false;
                } else {
                    return true;
                }
            }
            Camp::Doctor => {
                if dc == 1 && pc == 0 {
                    return false;
                } else if dc == 2 && pc == 1 {
                    return false;
                } else {
                    return true;
                }
            }
        }
    }
    pub fn is_illegal_position_setup2(&self) -> bool {
        for ((_, _), c) in self.character.iter() {
            if self.stuff.get(c) != None {
                return true;
            }
        }
        return false;
    }

    /// Check if the setup in setup3 is legal
    pub fn is_illegal_setup3(&self) -> bool {
        for (_, coord) in self.map.iter() {
            if coord.x > 1 || coord.x < -1 || coord.y > 1 || coord.y < -1 {
                return true;
            }
        }
        return false;
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
    fn test_start1() {
        let g = Game::init(None);
        assert_eq!(g.history.borrow().children[0].borrow().properties.len(), 4);
        assert_eq!(
            g.history.borrow().children[0].borrow().properties[3]
                .value
                .len(),
            1
        );
    }

    #[test]
    fn test_start2() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen]
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd])
            "
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        assert_eq!(g.phase, Phase::Setup1);
        // Note that the "a" in "ab" SGF notation stands for y-axis.
        assert_eq!(*g.env.get(&Coord::new(0, 5)).unwrap(), World::Humanity);
        assert_eq!(*g.env.get(&Coord::new(2, 5)).unwrap(), World::Underworld);
    }

    #[test]
    fn test_setup0() {
        let s0 = "(;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            ;C[Setup0]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd])
            "
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        assert_eq!(g.phase, Phase::Setup1);
    }

    #[test]
    #[should_panic(expected = "Ex14")]
    fn test_setup1_1() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen]
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd])
            ;C[Setup2]
            "
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let _g = Game::init(Some(t));
    }

    #[test]
    #[should_panic(expected = "Ex11")]
    fn test_setup1_2() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen]
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd])
            ;C[Setup1]
            AB[ab][cd][ef][bc]
            "
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        assert_eq!(g.phase, Phase::Setup2);
    }

    #[test]
    fn test_setup1_3() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen]
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd])
            ;C[Setup1]AB[ab]
            ;C[Setup1]AB[dc][bd]
            "
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        assert_eq!(g.phase, Phase::Setup1);
        assert_eq!(g.stuff.len(), 3);
    }

    #[test]
    #[should_panic(expected = "Ex15")]
    fn test_setup1_4() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen]
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd])
            ;C[Setup1]AB[ab]
            ;C[Setup1]AB[dc][bd]
            ;C[Setup1]AB[dd][be]
            "
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let _g = Game::init(Some(t));
    }

    #[test]
    #[should_panic(expected = "Ex19")]
    fn test_setup2_position() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd])
            ;C[Setup1]AB[ab][dc][bd][ef]
            ;C[Setup2]AW[ab]
            "
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let _g = Game::init(Some(t));
    }

    #[test]
    #[should_panic]
    fn test_setup2_order() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd])
            ;C[Setup1]AB[ab][dc][bd][ef]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AW[ac]
            "
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let _g = Game::init(Some(t));
    }

    #[test]
    #[should_panic(expected = "Ex1b")]
    fn test_setup3() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd])
            ;C[Setup1]AB[ab][dc][bd][ef]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ad]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ac]
            ;C[Setup3]AW[gg]
            "
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let _g = Game::init(Some(t));
    }

    #[test]
    fn test_end1_1() {
        let mut g = Game::init(None);
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
        let mut g = Game::init(None);
        g.turn = Camp::Doctor;
        g.add_marker(&Coord::new(0, 0), &Camp::Doctor);
        g.add_marker(&Coord::new(1, 0), &Camp::Doctor);
        g.add_marker(&Coord::new(1, 1), &Camp::Doctor);
        g.add_marker(&Coord::new(1, 2), &Camp::Doctor);
        g.add_marker(&Coord::new(2, 2), &Camp::Doctor);
        for _ in 0..MAX_MARKER {
            g.add_marker(&Coord::new(3, 2), &Camp::Doctor);
        }
        g.add_marker(&Coord::new(3, 3), &Camp::Doctor);
        g.add_marker(&Coord::new(3, 4), &Camp::Doctor);
        g.add_marker(&Coord::new(3, 5), &Camp::Doctor);
        assert_eq!(g.end1(), true);
    }

    #[test]
    fn test_end1_3() {
        let mut g = Game::init(None);
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

    #[test]
    fn test_end2_1() {
        let mut g = Game::init(None);
        g.turn = Camp::Plague;
        for _ in 0..=MAX_MARKER {
            g.add_marker(&Coord::new(3, 2), &Camp::Plague);
            g.add_marker(&Coord::new(3, 3), &Camp::Plague);
            g.add_marker(&Coord::new(3, 4), &Camp::Plague);
            g.add_marker(&Coord::new(3, 5), &Camp::Plague);
        }
        assert_eq!(g.end2(), true);
    }

    #[test]
    fn test_add_marker() {
        let mut g = Game::init(None);
        let mut g2 = Game::init(None);
        g2.turn = Camp::Doctor;
        g.add_marker(&Coord::new(3, 2), &g2.turn);
        g2.turn = Camp::Plague;
        assert_eq!(
            g.stuff.get(&Coord::new(3, 2)),
            Some(&(Camp::Doctor, Stuff::Marker(1)))
        );
    }
}
