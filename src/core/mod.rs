pub mod action;
mod gen_map;
pub mod grid_coord;
mod setup;
pub mod status_code;
pub mod tree;

use action::Action;
use gen_map::get_rand_matrix;
use grid_coord::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
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
    End(u32),
}

pub trait SGFCoord {
    fn to_map(&self) -> Coord;
    fn to_env(&self) -> Coord;
}

impl SGFCoord for str {
    fn to_map(&self) -> Coord {
        // row major, x as row index and y as column index
        // ghijk
        let x: i32 = self.chars().nth(0).unwrap() as i32 - 'i' as i32;
        let y: i32 = self.chars().nth(1).unwrap() as i32 - 'i' as i32;
        Coord::new(x, y)
    }

    fn to_env(&self) -> Coord {
        // row major, x as row index and y as column index
        // abcdef
        let x: i32 = self.chars().nth(0).unwrap() as i32 - 'a' as i32;
        let y: i32 = self.chars().nth(1).unwrap() as i32 - 'a' as i32;
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
    pub savepoint: bool,

    rng: Rc<RefCell<StdRng>>,
}

pub const SIZE: i32 = 6;
pub const MAP_SIZE: usize = 5;
pub const QUAD_SIZE: usize = 4;
pub const DOCTOR_MARKER: i32 = 5;
pub const PLAGUE_MARKER: i32 = 4;
pub const MAX_MARKER: u8 = 5;
const SETUP1_MARKER: usize = 4;

impl Game {
    pub fn init(file: Option<Rc<RefCell<TreeNode>>>) -> Self {
        let seed: [u8; 32] = [0; 32];
        let rng = StdRng::from_seed(seed);
        let shared_rng = Rc::new(RefCell::new(rng));
        Self::init_with_rng(file, shared_rng)
    }
    pub fn init_with_rng(file: Option<Rc<RefCell<TreeNode>>>, rng: Rc<RefCell<StdRng>>) -> Self {
        let mut iter = "".trim().chars().peekable();
        // The default SGF history starts with a common game-info node
        let mut g = Game {
            env: HashMap::new(),
            map: HashMap::new(),
            character: HashMap::new(),
            stuff: HashMap::new(),
            turn: Camp::Plague,
            phase: Phase::Setup0,
            history: TreeNode::new(&mut iter, None),
            savepoint: false,
            rng: rng.clone(),
        };
        // OK, I admit, this is a over-engineering mistake...
        g.history.borrow_mut().divergent = true;

        fn load_history(t: &TreeNode, is_front: bool, g: &mut Game) {
            if !is_front {
                return;
            }
            if let Some(p) = t.to_sgf_node() {
                g.append_history(p.clone());
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
                            match g.is_illegal_setup1() {
                                Err(x) => {
                                    panic!("{}", x);
                                }
                                _ => {}
                            }
                        }
                        if g.is_setup1_done() {
                            g.phase = Phase::Setup2;
                            g.switch();
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
                        match g.turn {
                            Camp::Plague => {
                                t.get_value("AB".to_string(), &mut s);
                            }
                            Camp::Doctor => {
                                t.get_value("AW".to_string(), &mut s);
                            }
                        }
                        let c1 = s.as_str().to_env();
                        let w = g.env.get(&c1).unwrap();
                        match g.character.get(&(*w, g.turn)) {
                            Some(_) => {
                                panic!("Ex1e");
                            }
                            None => {
                                g.character.insert((*w, g.turn), c1);
                            }
                        }
                        match g.is_illegal_order_setup2() {
                            Err(x) => {
                                panic!("{}", x);
                            }
                            _ => {}
                        }
                        match g.is_illegal_position_setup2() {
                            Err(x) => {
                                panic!("{}", x);
                            }
                            _ => {}
                        }
                        if g.is_setup2_done() {
                            g.phase = Phase::Setup3;
                        }
                        g.switch();
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
                        t.get_value("AW".to_string(), &mut s);
                        let c = s.as_str().to_map();
                        g.map.insert(Camp::Doctor, c);
                        // check the start()::Setup3 for why this is done here
                        g.map.insert(Camp::Plague, c);
                        g.next();
                        g.phase = Phase::Main(1);
                        if g.is_illegal_setup3() {
                            panic!("Ex1b");
                        }
                    } else {
                        panic!("Ex1a");
                    }
                }
                _ => match g.phase {
                    Phase::Main(_) => {
                        match t.to_action(&g) {
                            Ok(a) => g.commit_action(&a),
                            Err(x) => {
                                panic!("{}", x);
                            }
                        }
                        g.next();
                    }
                    Phase::End(_) => {
                        // do nothing more
                    }
                    Phase::Setup0 => { //game-info?
                    }
                    _ => {
                        panic!("Ex1c");
                    }
                },
            }
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
                    let m = get_rand_matrix(g.rng.clone());
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
            // Update history
            let mut h: Vec<String> = Vec::new();
            let mut u: Vec<String> = Vec::new();
            for (c, w) in g.env.iter() {
                if *w == World::Humanity {
                    h.push(c.env_to_sgf())
                } else if *w == World::Underworld {
                    u.push(c.env_to_sgf())
                }
            }
            let hs = String::from("(;C[Setup0]AW") + "[" + &h.join("][") + "])";
            g.append_history_with_new_tree(&hs);
            let us = String::from("(;C[Setup0]AB") + "[" + &u.join("][") + "])";
            g.append_history_with_new_tree(&us);
            g.phase = Phase::Setup1;
        }
        g
    }

    pub fn is_ended(&self) -> bool {
        if let Phase::End(_x) = self.phase {
            return true;
        }
        // XXX: Did I handle this? draw or both lose?
        if let Phase::Main(100) = self.phase {
            return true;
        }
        false
    }
    fn check_and_set_end(&mut self) {
        if self.is_ended() {
            return;
        }
        #[cfg(debug_assertions)]
        {
            println!("{:?} by {:?}", self.phase, self.turn);
            for i in 0..SIZE {
                for j in 0..SIZE {
                    let c = Coord::new(j, i);
                    let mut ch = "0";
                    match self.stuff.get(&c) {
                        Some((Camp::Plague, Stuff::Marker(_))) => {
                            ch = "p";
                        }
                        Some((Camp::Doctor, Stuff::Marker(_))) => {
                            ch = "d";
                        }
                        Some((Camp::Plague, Stuff::Colony)) => {
                            ch = "P";
                        }
                        Some((Camp::Doctor, Stuff::Colony)) => {
                            ch = "D";
                        }
                        _ => {}
                    }
                    print!("{}", ch);
                }
                println!("");
            }
        }
        if self.end1() || self.end2() {
            if let Phase::Main(x) = self.phase {
                self.phase = Phase::End(x + 1);
            }
        }
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
            |c| c.y == SIZE - 1,
        ) {
            return true;
        }

        if check_path(
            &self.stuff,
            t,
            &dir,
            |i| Coord::new(0, i),
            |c| c.x == SIZE - 1,
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
            Some((cc, Stuff::Colony)) => {
                if self.savepoint {
                    // since we are in simulation, this is allowed to revert the effect.
                    assert_ne!(cc, camp);
                    self.stuff.insert(*c, (*cc, Stuff::Marker(MAX_MARKER)));
                } else {
                    panic!("Cannot add marker to Colony");
                }
            }
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
        if let Phase::End(_x) = self.phase {
            panic!("The game is finished. Not expected here.");
        }
        if a.lockdown != Lockdown::Normal {
            let c_start = *self.map.get(&Camp::Plague).unwrap();
            self.set_map(Camp::Plague, c_start.lockdown(a.lockdown));
        }
        if a.map != None {
            self.set_map(self.turn, a.map.unwrap());
        } else {
            // been skipped
            self.check_and_set_end();
            return;
        }
        self.character
            .insert((a.world.unwrap(), self.turn), a.character.unwrap());
        let m = a.markers.clone();
        let t = self.turn;
        for c in m.iter() {
            self.add_marker(c, &t);
        }
        self.check_and_set_end();
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
    pub fn is_illegal_setup1(&self) -> Result<(), &'static str> {
        let mut row: [bool; crate::core::SIZE as usize] = [false; crate::core::SIZE as usize];
        let mut col: [bool; crate::core::SIZE as usize] = [false; crate::core::SIZE as usize];
        let mut quad: [bool; crate::core::QUAD_SIZE] = [false; crate::core::QUAD_SIZE];
        for c in self.stuff.iter() {
            if let Stuff::Marker(x) = c.1 .1 {
                if x > 1 {
                    return Err("Ex1d");
                }
            }
            let coord = c.0;
            let q = coord.to_quad();
            if row[coord.x as usize] || col[coord.y as usize] || quad[q as usize] {
                return Err("Ex11");
            }
            row[coord.x as usize] = true;
            col[coord.y as usize] = true;
            quad[q as usize] = true;
        }
        return Ok(());
    }

    pub fn is_setup2_done(&self) -> bool {
        return self.character.len() == Camp::COUNT * World::COUNT;
    }

    /// Check if the setup in setup2 is legal
    pub fn is_illegal_order_setup2(&self) -> Result<(), &'static str> {
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
                    return Ok(());
                } else if dc == 2 && pc == 2 {
                    return Ok(());
                } else if dc - pc == 1 {
                    return Err("Ex1e");
                } else {
                    return Err("Ex18");
                }
            }
            Camp::Doctor => {
                if dc == 1 && pc == 0 {
                    return Ok(());
                } else if dc == 1 && pc == 1 {
                    return Err("Ex1e");
                } else if dc == 2 && pc == 1 {
                    return Ok(());
                } else {
                    return Err("Ex18");
                }
            }
        }
    }
    pub fn is_illegal_position_setup2(&self) -> Result<(), &'static str> {
        let mut cl: Vec<Coord> = Vec::new();
        for ((_, _), c) in self.character.iter() {
            if self.stuff.get(c) != None {
                return Err("Ex19");
            }
            if !cl.contains(c) {
                cl.push(*c);
            } else {
                return Err("Ex1f");
            }
        }
        Ok(())
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

    pub fn setup_with_coord(&mut self, c: Coord) -> Result<Phase, &'static str> {
        match self.phase {
            Phase::Setup3 => {
                self.map.insert(Camp::Doctor, c);
                self.map.insert(Camp::Plague, c);
                self.phase = Phase::Main(1);
                Ok(Phase::Main(1))
            }
            Phase::Setup1 => {
                self.add_marker(&c, &Camp::Plague);
                match self.is_illegal_setup1() {
                    Err(x) => {
                        // This is funny. I was seriously thinking,
                        // "Shit, I have to add a sub_marker..."
                        self.add_marker(&c, &Camp::Doctor);
                        return Err(x);
                    }
                    _ => {}
                }
                if self.is_setup1_done() {
                    self.phase = Phase::Setup2;
                    self.switch();
                    return Ok(Phase::Setup1);
                }
                Ok(Phase::Setup1)
            }
            Phase::Setup2 => {
                let w = *self.env.get(&c).unwrap();
                match self.character.get(&(w, self.turn)) {
                    Some(_) => {
                        return Err("Ex1e");
                    }
                    None => {
                        self.character.insert((w, self.turn), c);
                    }
                }
                match self.is_illegal_order_setup2() {
                    Err("Ex18") => {
                        self.character
                            .remove(&(*self.env.get(&c).unwrap(), self.turn));
                        return Err("Ex18");
                    }
                    Err("Ex1e") => {
                        return Err("Ex1e");
                    }
                    _ => {}
                }
                match self.is_illegal_position_setup2() {
                    Err(x) => {
                        self.character
                            .remove(&(*self.env.get(&c).unwrap(), self.turn));
                        return Err(x);
                    }
                    _ => {}
                }
                if self.is_setup2_done() {
                    self.phase = Phase::Setup3;
                    self.switch();
                    return Ok(Phase::Setup2);
                }
                self.switch();
                Ok(Phase::Setup2)
            }
            _ => {
                panic!("Not possible");
            }
        }
    }

    pub fn setup_with_alpha(&mut self, s: &String) -> Result<Phase, &'static str> {
        let mut c = s.as_str().to_env();
        let mut s0 = if self.phase == Phase::Setup1 {
            String::from("(;C[Setup1]")
        } else if self.phase == Phase::Setup2 {
            String::from("(;C[Setup2]")
        } else {
            String::from("(;C[Setup3]")
        };
        let t = self.turn;
        if self.phase == Phase::Setup3 {
            c = s.as_str().to_map();
        }
        match self.setup_with_coord(c) {
            Err(x) => {
                return Err(x);
            }
            Ok(x) => {
                // the self.turn is deeply binding with the state transition
                // so error-prune. Here we check the camp on enter.
                s0 = s0 + if t == Camp::Doctor { "AW[" } else { "AB[" };
                s0 = s0 + s + "])";
                self.append_history_with_new_tree(&s0);
                return Ok(x);
            }
        }
    }

    pub fn append_history_with_new_tree(&mut self, s0: &String) {
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut buffer = String::new();
        t.borrow().to_string(&mut buffer);
        if let Some(p) = t.borrow().children[0].borrow().to_sgf_node() {
            self.append_history(p.clone());
        };
    }

    fn append_history(&mut self, t: Rc<RefCell<TreeNode>>) {
        self.history
            .borrow_mut()
            .children
            .push(t.borrow().children[0].clone());
        t.borrow().children[0].borrow_mut().parent = Some(self.history.clone());
        self.history = t.borrow().children[0].clone();
    }

    pub fn is_setup(&self) -> bool {
        if let Phase::Main(_) = self.phase {
            return true;
        } else {
            return false;
        }
    }

    fn near_but_not_colony(&self, c: Coord, oa: Option<&Action>) -> bool {
        // if a is not None, we are in the middle of SetMarker,
        // do the complicated additions here.
        let iter = vec![0..(SIZE / 2), (SIZE / 2)..(SIZE)];
        let q: usize = c.to_quad().try_into().unwrap();
        for xi in iter[q % 2].clone() {
            for yi in iter[q / 2].clone() {
                let ci = Coord::new(xi, yi);
                if let Some(&(t, si)) = self.stuff.get(&ci) {
                    if t != self.turn {
                        continue;
                    }
                    if si == Stuff::Colony {
                        return true;
                    } else if let Some(a) = oa {
                        let Stuff::Marker(m) = si else {
                            panic!("Not possible");
                        };
                        let added = a
                            .markers
                            .iter()
                            .map(|&ctemp| if ctemp == ci { 1 } else { 0 })
                            .sum::<u8>();
                        // Everytime a.markers is updated, it must have already
                        // checked this function. So this should be OK.
                        if m + added > MAX_MARKER && c != ci {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    pub fn get_marker_capacity(&self, c: Coord, oa: Option<&Action>) -> u8 {
        // is a Colony near within the same quadrant?
        let near_but_not_colony = self.near_but_not_colony(c, oa);
        let mut modifier_from_action = 0;
        if let Some(a) = oa {
            modifier_from_action = a
                .markers
                .iter()
                .map(|&y| if y == c { 1 } else { 0 })
                .sum::<u8>();
        }

        let mut max = MAX_MARKER + if near_but_not_colony { 0 } else { 1 };
        if self.stuff.get(&c) == None {
            return max;
        }
        let &(camp, s) = self.stuff.get(&c).unwrap();
        if s == Stuff::Colony {
            return 0;
        }

        let Stuff::Marker(m) = s else {
            panic!("not possible");
        };
        if camp == self.turn {
            max = max - m - modifier_from_action;
        } else {
            max = max + m - modifier_from_action;
        }
        return max;
    }

    pub fn undo(&mut self) {
        if let Phase::End(x) = self.phase {
            self.phase = Phase::Main(x);
            // A common usage is like coord_server::next, in which taking turns
            // is a must, so here we shouldn't do this extra switch.
            // self.switch();
        }
        match self.phase {
            Phase::Main(1) => {
                // Effectively, the game state is right after setup is done.
                // You cannot undo more, because setup is not undo-able now.
                return;
            }
            Phase::Main(x) => {
                // Do the switch first.
                // For instance (check test_undo1), Turn 7 is Plague's turn.
                // When undoing, change the perspective to the Doctor, and reverse
                // the effect the Doctor has done in Turn 6, which is recorded in
                // self.history.
                self.switch();

                // If there was an Ix00, then we shouldn't need to do anything,
                // except for reverting the history.
                if self.history.borrow().properties[0].value.len() != 1 {
                    let op_map = if let Some(p) = &self.history.borrow().parent {
                        if x != 2 {
                            p.as_ref().borrow().properties[0].value[0].as_str().to_map()
                        } else {
                            p.as_ref()
                                .borrow()
                                .properties
                                .iter()
                                .find(|prop| prop.ident == "AW")
                                .unwrap()
                                .value[0]
                                .as_str()
                                .to_map()
                        }
                    } else {
                        panic!("No parent");
                    };

                    let my_map = self.history.borrow().properties[0].value[0]
                        .as_str()
                        .to_map();
                    let restriction = my_map - &op_map;
                    let steps = restriction.iter().map(|(_, i)| *i as usize).sum::<usize>();

                    // undo the map
                    // Normally, this will need the reference from 2 moves ago.
                    let my_map_prev = if let Some(p) = &self.history.borrow().parent {
                        // BUT! A special case is when undoing a Plague's move,
                        // and the previous Doctor's move happens to be a locked-down!
                        // AND Doctor doesn't just stay there!
                        if self.turn == Camp::Plague
                            && op_map == "ii".to_map()
                            && p.as_ref().borrow().properties[0].value.len() > 1
                        {
                            p.as_ref().borrow().properties[0].value[1]
                                .as_str()
                                .to_map()
                                .clone()
                        } else if let Some(pp) = &p.as_ref().borrow().parent {
                            if x > 3 {
                                pp.as_ref().borrow().properties[0].value[0]
                                    .as_str()
                                    .to_map()
                                    .clone()
                            } else if x == 3 {
                                pp.as_ref()
                                    .borrow()
                                    .properties
                                    .iter()
                                    .find(|prop| prop.ident == "AW")
                                    .unwrap()
                                    .value[0]
                                    .as_str()
                                    .to_map()
                                    .clone()
                            } else {
                                // x == 2
                                // Plauge undo-es its first move, which goes back to some
                                // value that will be overwirtten soon anyway.
                                // Previously we use Coord::new(-998, -997) for this purpose,
                                // but as we use the information for encoding,
                                // the coordinate is obviuosly not OK.
                                // Similar to the way we initialize a game, we set the
                                // map position of the Plague to that of the Doctor.
                                p.as_ref()
                                    .borrow()
                                    .properties
                                    .iter()
                                    .find(|prop| prop.ident == "AW")
                                    .unwrap()
                                    .value[0]
                                    .as_str()
                                    .to_map()
                                    .clone()
                            }
                        } else {
                            panic!("No grandparent");
                        }
                    } else {
                        panic!("No parent");
                    };
                    self.set_map(self.turn, my_map_prev);

                    // if this is lockdown, set this for the Plague as well
                    let op_camp = self.opposite(self.turn);
                    if self.turn == Camp::Doctor && my_map == "ii".to_map() {
                        let op_map_prev = if let Some(p) = &self.history.borrow().parent {
                            p.borrow().properties[0].value[0].as_str().to_map()
                        } else {
                            panic!("No parent");
                        };
                        self.set_map(op_camp, op_map_prev);
                    }

                    // undo the character
                    let index = if self.turn == Camp::Doctor && my_map == "ii".to_map() {
                        2
                    } else {
                        1
                    };
                    let my_character_prev = self.history.borrow().properties[0].value[index]
                        .as_str()
                        .to_env();
                    let my_character_now = self.history.borrow().properties[0].value[index + steps]
                        .as_str()
                        .to_env();
                    for ((_w, _t), c) in self.character.iter_mut() {
                        if *c == my_character_now {
                            *c = my_character_prev;
                        }
                    }

                    // undo markers, this can be done by adding opponent's markers
                    let len = self.history.borrow().properties[0].value.len();
                    for i in (index + steps + 1)..len {
                        let c_str = self.history.borrow().properties[0].value[i].clone();
                        let c = c_str.to_env().clone();
                        self.add_marker(&c, &op_camp);
                    }
                }
                self.phase = Phase::Main(x - 1);
                let temp = if let Some(p) = &self.history.borrow().parent {
                    p.clone()
                } else {
                    panic!("No parent");
                };
                self.history = temp.clone();
            }
            _ => {}
        }
    }

    // RL support function
    // When we set some state (Main(x)) and its sub-state (Action) as
    // the root node to be explored, the agent will issue a SAVE_CODE.
    pub fn save(&mut self) {
        self.history.borrow_mut().savepoint = true;
        self.savepoint = true;
    }

    // RL support function
    // This is called by agent **actively** when one evaluation trial ends,
    // or **passively** when the end of a game is reached.
    // General usage for this call:
    // * active/passive reset for EVAL_TIMES, then the agent decides the move
    // * in the next state+action state, issue save
    //
    // assigning clear as true will erase the savepoint flags, and mark the
    // game a **normal** game; otherwise, it is a game being evaluating and
    // backtracking.
    pub fn reset(&mut self, clear: bool) {
        loop {
            if self.history.borrow().savepoint {
                break;
            }
            self.undo();
            if self.phase == Phase::Main(1) {
                // Otherwise, why are we reseting history???
                // If in the future we can undo setup, it would be useful, but...
                assert_eq!(self.history.borrow().savepoint, true);
                break;
            }
        }
        if clear {
            self.history.borrow_mut().savepoint = false;
            self.savepoint = false;
        }
        let _ = self.history.borrow_mut().children.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start1() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen])
            "
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        let temp = g.history.borrow().to_root().borrow().children[0].clone();
        assert_eq!(temp.borrow().properties.len(), 4);
        assert_eq!(temp.borrow().properties[3].value.len(), 1);
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
        assert_eq!(*g.env.get(&Coord::new(0, 5)).unwrap(), World::Underworld);
        assert_eq!(*g.env.get(&Coord::new(2, 0)).unwrap(), World::Humanity);
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
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup2]
            )"
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
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]
            AB[ab][cd][ef][bc]
            )"
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
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab]
            ;C[Setup1]AB[dc][bd]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        assert_eq!(g.phase, Phase::Setup1);
        assert_eq!(g.stuff.len(), 3);
    }

    #[test]
    #[should_panic(expected = "Ex11")]
    fn test_setup1_4() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen]
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab]
            ;C[Setup1]AB[dc][bd]
            ;C[Setup1]AB[dd][be]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let _g = Game::init(Some(t));
    }

    #[test]
    #[should_panic(expected = "Ex1d")]
    fn test_setup1_5() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen]
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab]
            ;C[Setup1]AB[dc][bd]
            ;C[Setup1]AB[ab]
            )"
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
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][dc][bd][ef]
            ;C[Setup2]AW[ab]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let _g = Game::init(Some(t));
    }

    #[test]
    #[should_panic(expected = "Ex1f")]
    fn test_setup2_position2() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][dc][bd][ef]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[aa]
            )"
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
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][dc][bd][ef]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AW[ac]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let _g = Game::init(Some(t));
    }

    #[test]
    #[should_panic]
    fn test_setup2_order1() {
        // It looks like a definite Ex1e case, but through g.init.traverse.load_history,
        // this will just become a panic due to empty AW.
        // Only through other way around can trigger a Ex1e.
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][dc][bd][ef]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AB[af]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let _g = Game::init(Some(t));
    }

    #[test]
    #[should_panic(expected = "Ex1b")]
    fn test_setup3_1() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][dc][bd][ef]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ad]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ac]
            ;C[Setup3]AW[gg]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let _g = Game::init(Some(t));
    }

    #[test]
    fn test_setup3_2() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][dc][bd][ef]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ad]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ac]
            ;C[Setup3]AW[hh]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        assert_eq!(g.phase, Phase::Main(1));
        assert_eq!(g.turn, Camp::Plague);
    }

    #[test]
    #[should_panic(expected = "Ex1c")]
    fn test_setup_misc() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][dc][bd][ef]
            ;W[ii][hh][aa][ab][bb][ab][aa][aa][aa][aa]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let _g = Game::init(Some(t));
    }

    #[test]
    fn test_two_steps() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            ;C[Setup3]AW[ij]
            ;B[jj][ad][cd][ad][ad][ad][ad]
            ;W[ii][hh][aa][ab][bb][ab][aa][aa][aa][aa]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        assert_eq!(
            *g.character.get(&(World::Humanity, Camp::Doctor)).unwrap(),
            Coord::new(1, 1)
        );
        assert_eq!(
            *g.character.get(&(World::Humanity, Camp::Plague)).unwrap(),
            Coord::new(2, 3)
        );
        assert_eq!(
            *g.character.get(&(World::Underworld, Camp::Doctor)).unwrap(),
            Coord::new(0, 5)
        );
        assert_eq!(
            *g.character.get(&(World::Underworld, Camp::Plague)).unwrap(),
            Coord::new(0, 2)
        );
        assert_eq!(g.stuff.len(), 5);
        assert_eq!(
            *g.stuff.get(&Coord::new(0, 3)).unwrap(),
            (Camp::Plague, Stuff::Marker(4))
        );
        assert_eq!(
            *g.stuff.get(&Coord::new(0, 0)).unwrap(),
            (Camp::Doctor, Stuff::Marker(4))
        );
        assert_eq!(g.stuff.get(&Coord::new(1, 0)), None);
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

    #[test]
    fn test_action_to_sgf() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            ;C[Setup3]AW[ij]
            ;B[jj][ad][cd][ad][ad][ad][ad]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        let s1 = "(;W[ii][hh][aa][ab][bb][ab][aa][aa][aa][aa])";
        iter = s1.trim().chars().peekable();
        let t2 = TreeNode::new(&mut iter, None);
        match t2.borrow().children[0].borrow().to_action(&g) {
            Ok(a) => assert_eq!(a.to_sgf_string(&g), s1),
            Err(x) => panic!("{}", x),
        };
    }

    #[test]
    fn test_action_to_sgf2() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            ;C[Setup3]AW[ij]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        let s1 = "(;B[jj][ad][cd][ad][ad][ad][ad])";
        iter = s1.trim().chars().peekable();
        let t2 = TreeNode::new(&mut iter, None);
        match t2.borrow().children[0].borrow().to_action(&g) {
            Ok(a) => assert_eq!(a.to_sgf_string(&g), s1),
            Err(x) => panic!("{}", x),
        };
    }

    #[test]
    fn test_setup_with() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));
        assert_eq!(g.setup_with_alpha(&String::from("ii")), Ok(Phase::Main(1)));
    }

    #[test]
    fn test_setup_with2() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));
        assert_eq!(g.setup_with_alpha(&String::from("af")), Ok(Phase::Setup2));
    }

    #[test]
    fn test_setup_state_machine1() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        assert_eq!(g.turn, Camp::Plague);
    }

    #[test]
    fn test_setup_state_machine2() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        assert_eq!(g.turn, Camp::Doctor);
    }

    #[test]
    fn test_setup_state_machine3() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        assert_eq!(g.turn, Camp::Doctor);
    }

    #[test]
    fn test_setup_state_machine4() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            ;C[Setup3]AW[ii]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        assert_eq!(g.turn, Camp::Plague);
    }

    #[test]
    fn test_load() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            ;C[Setup3]AW[ij]
            ;B[jj][ad][cd][ad][ad][ad][ad]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        let mut buffer = String::new();
        g.history.borrow().to_root().borrow().to_string(&mut buffer);
        assert_eq!(
            s0.chars()
                .filter(|&c| !c.is_whitespace())
                .collect::<String>(),
            buffer
        );
    }

    #[test]
    fn test_marker_capacity() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[aa]
            ;C[Setup2]AB[ac]
            ;C[Setup2]AW[af]
            ;C[Setup2]AB[ad]
            ;C[Setup3]AW[ij]
            ;B[jj][ad][cd][ad][ad][ad][ad]
            )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));
        // The condition looks a bit weird.
        // The MAX_MARKER was defined as the real amount that a
        // grid can have markers. Not really the capacity. So
        // to count the one that upgrades the grid to a colony,
        // an +1 here.
        assert_eq!(
            g.get_marker_capacity(Coord::new(/* [cf] */ 2, 5), None),
            MAX_MARKER + 1
        );
        g.stuff
            .insert(Coord::new(2, 4), (Camp::Doctor, Stuff::Colony));
        assert_eq!(
            g.get_marker_capacity(Coord::new(/* [cf] */ 2, 5), None),
            MAX_MARKER
        );
        g.stuff
            .insert(Coord::new(0, 5), (Camp::Doctor, Stuff::Marker(3)));
        assert_eq!(
            g.get_marker_capacity(Coord::new(/* [af] */ 0, 5), None),
            MAX_MARKER - 3
        );
    }

    #[test]
    fn test_precise_route() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen];C[Setup0]AW[fa][ef][ed][eb][cf][cc][dc][ca][ad][fe][ab][db][bb][be][fb][ae][ac][df];C[Setup0]AB[af][ba][dd][da][ff][bf][ee][bc][de][ec][cb][aa][ea][bd][ce][fc][cd][fd];C[Setup1]AB[ee];C[Setup1]AB[cf];C[Setup1]AB[fa];C[Setup1]AB[bc];C[Setup2]AW[bf];C[Setup2]AB[ce];C[Setup2]AW[db];C[Setup2]AB[eb];C[Setup3]AW[ih])"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));
        let _s1 = "(;B[jg][ce][de][dd][ce][ce][de][de])";
        let _s2 = "(;B[jg][ce][cd][dd][cd][cd][de][de])";
        let _s3 = "(;B[jg][eb][fb][fa][eb][eb][fb][fb])";
        let mut a = Action::new();

        g.stuff.insert("fb".to_env(), (Camp::Doctor, Stuff::Colony));
        g.character
            .insert((World::Underworld, Camp::Doctor), "dd".to_env());
        assert_eq!(Err("Ex20"), a.add_map_step(&g, "jg".to_map()));
    }

    #[test]
    fn test_clolonized() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen];C[Setup0]AW[fa][ef][ed][eb][cf][cc][dc][ca][ad][fe][ab][db][bb][be][fb][ae][ac][df];C[Setup0]AB[af][ba][dd][da][ff][bf][ee][bc][de][ec][cb][aa][ea][bd][ce][fc][cd][fd];C[Setup1]AB[ee];C[Setup1]AB[cf];C[Setup1]AB[fa];C[Setup1]AB[bc];C[Setup2]AW[bf];C[Setup2]AB[ce];C[Setup2]AW[db];C[Setup2]AB[eb];C[Setup3]AW[ih])"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));
        let mut a = Action::new();

        g.stuff.insert("ff".to_env(), (Camp::Plague, Stuff::Colony));
        assert_eq!(g.near_but_not_colony("dd".to_env(), None), true);
        assert_eq!(g.near_but_not_colony("da".to_env(), None), false);
        g.stuff
            .insert("af".to_env(), (Camp::Plague, Stuff::Marker(4)));
        a.markers.push("af".to_env());
        assert_eq!(g.near_but_not_colony("ad".to_env(), Some(&a)), false);
        a.markers.push("af".to_env());
        assert_eq!(g.near_but_not_colony("ad".to_env(), Some(&a)), true);
    }

    #[test]
    fn test_capacity() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen];C[Setup0]AW[fa][ef][ed][eb][cf][cc][dc][ca][ad][fe][ab][db][bb][be][fb][ae][ac][df];C[Setup0]AB[af][ba][dd][da][ff][bf][ee][bc][de][ec][cb][aa][ea][bd][ce][fc][cd][fd];C[Setup1]AB[ee];C[Setup1]AB[cf];C[Setup1]AB[fa];C[Setup1]AB[bc];C[Setup2]AW[bf];C[Setup2]AB[ce];C[Setup2]AW[db];C[Setup2]AB[eb];C[Setup3]AW[ih])"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));
        let mut a = Action::new();

        g.stuff
            .insert("bf".to_env(), (Camp::Plague, Stuff::Marker(3)));
        g.stuff
            .insert("af".to_env(), (Camp::Plague, Stuff::Marker(4)));
        assert_eq!(g.get_marker_capacity("af".to_env(), None), 2);
        a.markers.push("af".to_env());
        assert_eq!(g.get_marker_capacity("af".to_env(), Some(&a)), 1);
        a.markers.push("af".to_env());
        assert_eq!(g.get_marker_capacity("af".to_env(), Some(&a)), 0);
        a.markers.push("bf".to_env());
        assert_eq!(g.get_marker_capacity("bf".to_env(), Some(&a)), 1);
    }

    #[test]
    fn test_undo1() {
        let s0 = "(
            ;C[Setup0]
            AW[df][da][db][ab][ba][ea][cc][fd][fc][af][ce][ee][cb][ef][bd][bc][fe]
            AB[aa][ec][cd][bb][ae][be][ed][ad][ff][eb][fb][bf][ac][fa][dd][dc][cf][de][ca]
            ;C[Setup1]AB[fc][ed][ab][cf]
            ;C[Setup2]AW[cc]
            ;C[Setup2]AB[ec]
            ;C[Setup2]AW[ca]
            ;C[Setup2]AB[ce]
            ;C[Setup3]AW[jj]
            ;B[ki][ce][cc][fc][ce][ce][ce][ce]             C[1]
            ;W[ji][cc][bc][cc][cc][cc][cc][cc]             C[2]
            ;B[hh][ec][dc][ac][aa][ac][ec][dc][ac]         C[3]
            ;W[jj][bc][cc][ce][ee][ef][ce][ce][ce][ce][bc] C[4]
            ;B[ik][fc][cc][ce][cc][fc][fc][cc]             C[5]
            ;W[ii][ig][ca][cd][cf][ca][cd][ca][ca][ca]     C[6][XXX Accidentlly find that C cannot go before B or W]
        )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));

        // Turn 7 has yet to play any sub-moves;
        // The state should be identical to what it was when Turn 6 ends except
        // for the fact that it became the Plague's turn for Turn 7.
        assert_eq!(g.phase, Phase::Main(7));
        assert_eq!(g.turn, Camp::Plague);
        assert_eq!(
            *g.stuff.get(&Coord::new(2, 0)).unwrap(),
            (Camp::Doctor, Stuff::Marker(4))
        );

        g.undo();
        assert_eq!(g.stuff.get(&Coord::new(2, 0)), None);
        assert_eq!(g.turn, Camp::Doctor);
        assert_eq!(*g.map.get(&Camp::Plague).unwrap(), Coord::new(0, 2));
        assert_eq!(g.phase, Phase::Main(6));
        g.undo();
        assert_eq!(g.phase, Phase::Main(5));
        g.undo();
        assert_eq!(g.phase, Phase::Main(4));

        g.undo();
        assert_eq!(g.phase, Phase::Main(3));

        g.undo();
        assert_eq!(g.phase, Phase::Main(2));
        g.undo();
        assert_eq!(g.phase, Phase::Main(1));
        assert_eq!(
            *g.character.get(&(World::Underworld, Camp::Plague)).unwrap(),
            Coord::new(4, 2)
        );
        assert_eq!(g.turn, Camp::Plague);
        g.undo();
        assert_eq!(g.phase, Phase::Main(1));
        assert_eq!(g.turn, Camp::Plague);
        assert_eq!(
            *g.character.get(&(World::Underworld, Camp::Doctor)).unwrap(),
            Coord::new(2, 0)
        );
        assert_eq!(
            *g.character.get(&(World::Underworld, Camp::Plague)).unwrap(),
            Coord::new(4, 2)
        );
        assert_eq!(g.turn, Camp::Plague);
    }

    #[test]
    fn test_undo2() {
        // Test if undo() works for Phase::End(x)
        let s0 = "(
            ;C[Setup0]
            AW[df][da][db][ab][ba][ea][cc][fd][fc][af][ce][ee][cb][ef][bd][bc][fe]
            AB[aa][ec][cd][bb][ae][be][ed][ad][ff][eb][fb][bf][ac][fa][dd][dc][cf][de][ca]
            ;C[Setup1]AB[fc][ed][ab][cf]
            ;C[Setup2]AW[cc]
            ;C[Setup2]AB[ec]
            ;C[Setup2]AW[ca]
            ;C[Setup2]AB[ce]
            ;C[Setup3]AW[jj]
        )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));

        // Some extra setup first
        g.stuff
            .insert("ca".to_env(), (Camp::Plague, Stuff::Marker(4)));
        g.stuff
            .insert("cb".to_env(), (Camp::Plague, Stuff::Marker(4)));
        g.stuff
            .insert("cc".to_env(), (Camp::Plague, Stuff::Marker(4)));
        g.stuff
            .insert("cd".to_env(), (Camp::Plague, Stuff::Marker(4)));

        let s1 = "(;B[ki][ce][cc][fc][ce][ce][ce][ce]             C[1])";
        iter = s1.trim().chars().peekable();
        let t2 = TreeNode::new(&mut iter, None);
        if let Ok(a) = t2.borrow().children[0].borrow().to_action(&g) {
            g.append_history_with_new_tree(&a.to_sgf_string(&g));
            g.commit_action(&a);
            g.next();
        };

        assert_eq!(g.phase, Phase::End(2));
        assert_eq!(g.turn, Camp::Doctor);
        g.undo();
        assert_eq!(g.phase, Phase::Main(1));
        assert_eq!(g.turn, Camp::Plague);
    }

    #[test]
    fn test_save_reset1() {
        let s0 = "(
            ;C[Setup0]
            AW[df][da][db][ab][ba][ea][cc][fd][fc][af][ce][ee][cb][ef][bd][bc][fe]
            AB[aa][ec][cd][bb][ae][be][ed][ad][ff][eb][fb][bf][ac][fa][dd][dc][cf][de][ca]
            ;C[Setup1]AB[fc][ed][ab][cf]
            ;C[Setup2]AW[cc]
            ;C[Setup2]AB[ec]
            ;C[Setup2]AW[ca]
            ;C[Setup2]AB[ce]
            ;C[Setup3]AW[jj]
        )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));

        g.save();

        let s1 = "(;B[ki][ce][cc][fc][ce][ce][ce][ce]             C[1])";
        iter = s1.trim().chars().peekable();
        let t2 = TreeNode::new(&mut iter, None);
        if let Ok(a) = t2.borrow().children[0].borrow().to_action(&g) {
            g.append_history_with_new_tree(&a.to_sgf_string(&g));
            g.commit_action(&a);
            g.next();
        };

        g.reset(false);

        let s1_5 = "(;B[ki][ce][cc][fc][ce][ce][ce][ce]             C[1])";
        iter = s1_5.trim().chars().peekable();
        let t2_5 = TreeNode::new(&mut iter, None);
        if let Ok(a) = t2_5.borrow().children[0].borrow().to_action(&g) {
            g.append_history_with_new_tree(&a.to_sgf_string(&g));
            g.commit_action(&a);
            g.next();
        } else {
            panic!("!!!");
        };

        let s2 = "(;W[ji][cc][bc][cc][cc][cc][cc][cc]             C[2])";
        iter = s2.trim().chars().peekable();
        let t3 = TreeNode::new(&mut iter, None);
        if let Ok(a) = t3.borrow().children[0].borrow().to_action(&g) {
            g.append_history_with_new_tree(&a.to_sgf_string(&g));
            g.commit_action(&a);
            g.next();
        };

        let s3 = "(;B[hh][ec][dc][ac][aa][ac][ec][dc][ac]         C[3])";
        iter = s3.trim().chars().peekable();
        let t4 = TreeNode::new(&mut iter, None);
        if let Ok(a) = t4.borrow().children[0].borrow().to_action(&g) {
            g.append_history_with_new_tree(&a.to_sgf_string(&g));
            g.commit_action(&a);
            g.next();
        };

        assert_eq!(g.phase, Phase::Main(4));
        assert_eq!(g.turn, Camp::Doctor);

        g.reset(true);

        assert_eq!(g.phase, Phase::Main(1));
        assert_eq!(g.turn, Camp::Plague);
        assert_eq!(g.history.borrow().children.len(), 0);
        assert_eq!(g.savepoint, false);
    }

    #[test]
    fn test_save_reset2() {
        let s0 = "(
            ;C[Setup0]
            AW[df][da][db][ab][ba][ea][cc][fd][fc][af][ce][ee][cb][ef][bd][bc][fe]
            AB[aa][ec][cd][bb][ae][be][ed][ad][ff][eb][fb][bf][ac][fa][dd][dc][cf][de][ca]
            ;C[Setup1]AB[fc][ed][ab][cf]
            ;C[Setup2]AW[cc]
            ;C[Setup2]AB[ec]
            ;C[Setup2]AW[ca]
            ;C[Setup2]AB[ce]
            ;C[Setup3]AW[jj]
        )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));

        g.save();

        let s1 = "(;B[ki][ce][cc][fc][ce][ce][ce][ce]             C[1])";
        iter = s1.trim().chars().peekable();
        let t2 = TreeNode::new(&mut iter, None);
        if let Ok(a) = t2.borrow().children[0].borrow().to_action(&g) {
            g.append_history_with_new_tree(&a.to_sgf_string(&g));
            g.commit_action(&a);
            g.next();
        };

        g.reset(false);

        let s1_5 = "(;B[ki][ce][cc][fc][ce][ce][ce][ce]             C[1])";
        iter = s1_5.trim().chars().peekable();
        let t2_5 = TreeNode::new(&mut iter, None);
        if let Ok(a) = t2_5.borrow().children[0].borrow().to_action(&g) {
            g.append_history_with_new_tree(&a.to_sgf_string(&g));
            g.commit_action(&a);
            g.next();
        } else {
            panic!("!!!");
        };

        let s2 = "(;W[jj]             C[2])";
        iter = s2.trim().chars().peekable();
        let t3 = TreeNode::new(&mut iter, None);
        if let Ok(a) = t3.borrow().children[0].borrow().to_action(&g) {
            g.append_history_with_new_tree(&a.to_sgf_string(&g));
            g.commit_action(&a);
            g.next();
        };

        g.reset(true);

        assert_eq!(g.phase, Phase::Main(1));
        assert_eq!(g.turn, Camp::Plague);
        assert_eq!(g.history.borrow().children.len(), 0);
        assert_eq!(g.savepoint, false);
    }
}
