use ndarray::Axis;
use std::boxed::Box;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::rc::Rc;
use std::thread;
use std::time;
use termion::raw::IntoRawMode;
use termion::{clear, color, cursor, style};

use crate::core::{Camp, Coord, Direction, Game, Lockdown, Phase, Stuff, World};
use crate::tree::TreeNode;

#[derive(Clone)]
struct Cursor {
    // Env
    env: Coord,
    // Compass
    comp: Coord,
}

#[derive(PartialEq)]
enum Type {
    Env,
    Compass,
}

const ORIGIN: Coord = Coord { x: 0, y: 0 };
const XUNIT: i32 = 6;
const XBORDER: i32 = 2;
const YUNIT: i32 = 3;
const YBORDER: i32 = 1;

const ORIGIN_ENV: Coord = Coord { x: 0, y: 0 };

const ORIGIN_COMPASS: Coord = Coord { x: 50, y: 8 };
const ORIGIN_COMPASS_STR: &str = "ii";
const XUNIT_COMPASS: i32 = 4;
const YUNIT_COMPASS: i32 = 2;
const XOFF: i32 = 2;
const YOFF: i32 = 1;

const SETUP1_MARKER: i32 = 4;
const SETUP2_HERO: i32 = 4;

const STATUS: Coord = Coord { x: 50, y: 18 };
impl Cursor {
    fn new() -> Cursor {
        Cursor {
            env: ORIGIN,
            comp: ORIGIN,
        }
    }

    // [Env] The left-top and the right-bottom coordinates
    fn block_range(&self) -> [Coord; 2] {
        let (x, y) = (self.env.x * XUNIT, self.env.y * YUNIT);
        [
            Coord::new(x + XBORDER, y + YBORDER + 1),
            Coord::new(x + XUNIT, y + YUNIT + 1),
        ]
    }

    fn to_hint(&self) -> Coord {
        Coord::new(STATUS.x, STATUS.y)
    }

    // [Env] The position to output hero info in each grid
    fn to_hero(&self) -> Coord {
        let (x, y) = (self.env.x * XUNIT, self.env.y * YUNIT);
        Coord::new(x + XUNIT - XBORDER, y + YUNIT - YBORDER)
    }

    // [Env] The position to output marker info in each grid
    fn to_marker(&self, t: Type) -> Coord {
        if t == Type::Env {
            let (x, y) = (self.env.x * XUNIT, self.env.y * YUNIT);
            Coord::new(x + XUNIT - XBORDER, y + YUNIT - YBORDER + 1)
        } else {
            let (x, y) = (
                ORIGIN_COMPASS.x + self.comp.x * XUNIT_COMPASS,
                ORIGIN_COMPASS.y + self.comp.y * YUNIT_COMPASS,
            );
            Coord::new(x + XOFF, y + YOFF)
        }
    }

    // Never Err. Possible results:
    //    Ok: Done
    //    Ok: Warning: Walls
    fn go(&mut self, dir: Direction, t: Type) -> Result<String, String> {
        if t == Type::Env {
            match dir {
                Direction::Right => {
                    if self.env.x < crate::core::SIZE - 1 {
                        self.env.x += 1;
                        return Ok("Done".to_string());
                    }
                }
                Direction::Left => {
                    if self.env.x > ORIGIN.x {
                        self.env.x -= 1;
                        return Ok("Done".to_string());
                    }
                }
                Direction::Up => {
                    if self.env.y > ORIGIN.y {
                        self.env.y -= 1;
                        return Ok("Done".to_string());
                    }
                }
                Direction::Down => {
                    if self.env.y < crate::core::SIZE - 1 {
                        self.env.y += 1;
                        return Ok("Done".to_string());
                    }
                }
            }
        } else {
            /*Type::Compass*/
            // XXX: remove the magics
            match dir {
                Direction::Right => {
                    if self.comp.x < 2
                        && !(self.comp.x == 1 && (self.comp.y == -2 || self.comp.y == 2))
                    {
                        self.comp.x += 1;
                        return Ok("Done".to_string());
                    }
                }
                Direction::Left => {
                    if self.comp.x > -2
                        && !(self.comp.x == -1 && (self.comp.y == -2 || self.comp.y == 2))
                    {
                        self.comp.x -= 1;
                        return Ok("Done".to_string());
                    }
                }
                Direction::Up => {
                    if self.comp.y > -2
                        && !(self.comp.y == -1 && (self.comp.x == -2 || self.comp.x == 2))
                    {
                        self.comp.y -= 1;
                        return Ok("Done".to_string());
                    }
                }
                Direction::Down => {
                    if self.comp.y < 2
                        && !(self.comp.y == 1 && (self.comp.x == -2 || self.comp.x == 2))
                    {
                        self.comp.y += 1;
                        return Ok("Done".to_string());
                    }
                }
            }
        }
        Ok("Warning: Wall".to_string())
    }
}

/// The game state.
pub struct Board {
    /// Standard output.
    stdout: Box<dyn Write>,
    /// Standard input.
    stdin: Box<dyn Read>,
    /// Cursor on the grid
    cursor: Cursor,
    pub game: Box<Game>,
    pub tree: Rc<RefCell<TreeNode>>,
    pub player: HashMap<Camp, TcpStream>,
}

const WHITE: color::Rgb = color::Rgb(255, 255, 255);
const BLACK: color::Rgb = color::Rgb(0, 0, 0);
const GRAY: color::Rgb = color::Rgb(96, 96, 96);
const ASH: color::Rgb = color::Rgb(196, 196, 196);
const BLUE: color::Rgb = color::Rgb(0, 127, 255);
const RED: color::Rgb = color::Rgb(255, 65, 54);

const ARENA: Coord = Coord { x: 100, y: 25 };

impl Board {
    /// Construct a game state.
    // if the given tree has no children, this is a new game,
    // otherwise, resume the game history
    pub fn new(g: Box<Game>, t: Rc<RefCell<TreeNode>>, p: HashMap<Camp, TcpStream>) -> Board {
        let mut b = Board {
            stdout: Box::new(io::stdout().lock().into_raw_mode().unwrap()),
            stdin: Box::new(io::stdin().lock()),
            cursor: Cursor::new(),
            game: g,
            tree: t,
            player: p,
        };

        if b.tree.borrow().children.len() > 0 {
            b.resume();
        }
        b.init();
        b
    }

    fn report(&mut self, x: &String) -> Result<String, String> {
        if let Some(mut s) = self.player.get(&self.game.turn) {
            let (mut e, mut c) = self.game.encode();
            e[[self.cursor.env.y as usize, self.cursor.env.x as usize, 0]] = 1;
            c[[
                (self.cursor.comp.y + 2) as usize,
                (self.cursor.comp.x + 2) as usize,
                0,
            ]] = 1;

            let mut formatted_string = String::new();
            for element in e
                .axis_iter(Axis(0))
                .into_iter()
                .flatten()
                .chain(c.axis_iter(Axis(0)).into_iter().flatten())
            {
                formatted_string += &format!("{} ", element);
            }
            if let Err(z) = s.write(formatted_string.as_bytes()) {
                return Err(z.to_string());
            }
            if let Err(z) = s.write(x.as_bytes()) {
                return Err(z.to_string());
            }
            if let Err(z) = s.flush() {
                return Err(z.to_string());
            }
        }
        Ok("".to_string())
    }

    /// If provided with a game history, load it. Note the phases.
    pub fn resume(&mut self) {
        fn load_history(t: &TreeNode, is_front: bool, g: &mut crate::core::Game) {
            if !is_front {
                return;
            }
            match g.phase {
                Phase::Setup0 => {
                    if t.checkpoint("Setup0".to_string()) {
                        let mut h: Vec<String> = Vec::new();
                        let mut u: Vec<String> = Vec::new();
                        t.get_general("AW".to_string(), &mut h);
                        t.get_general("AB".to_string(), &mut u);
                        for c in h.iter() {
                            g.env.insert(g.sgf_to_env(c), World::Humanity);
                        }
                        for c in u.iter() {
                            g.env.insert(g.sgf_to_env(c), World::Underworld);
                        }
                        g.phase = Phase::Setup1;
                    }
                }
                Phase::Setup1 => {
                    if t.checkpoint("Setup1".to_string()) {
                        let mut m: Vec<String> = Vec::new();
                        t.get_general("AB".to_string(), &mut m);
                        for c in m.iter() {
                            g.stuff
                                .insert(g.sgf_to_env(c), (Camp::Plague, Stuff::Marker(1)));
                        }
                        if g.is_illegal_setup1() {
                            panic!("Rule violation: The initial 4 markers can not share any rows or columns.");
                        }
                        g.phase = Phase::Setup2;
                    } else {
                        panic!("Unexpected SGF content: Setup1 not done.");
                    }
                }
                Phase::Setup2 => {
                    // with the assumption that the sequence is always D->P->D->P
                    let mut s = String::new();
                    if t.checkpoint("Setup2Begin".to_string())
                        || t.checkpoint("Setup2End".to_string())
                    {
                        t.get_value("W".to_string(), &mut s);
                        let c1 = g.sgf_to_env(&s);
                        g.hero.insert((*g.env.get(&c1).unwrap(), Camp::Doctor), c1);
                        s = String::from("");
                        t.get_value("B".to_string(), &mut s);
                        let c2 = g.sgf_to_env(&s);
                        g.hero.insert((*g.env.get(&c2).unwrap(), Camp::Plague), c2);
                    }
                    if t.checkpoint("Setup2End".to_string()) {
                        g.phase = Phase::Setup3;
                    }
                    if g.is_illegal_setup2() {
                        panic!("Rule violation: Should be exactly one hero in Humanity and exactly one in Underworld for each side.");
                    }
                }
                Phase::Setup3 => {
                    let mut s = String::new();
                    if t.checkpoint("Setup3".to_string()) {
                        t.get_value("W".to_string(), &mut s);
                        let c = g.sgf_to_compass(&s);
                        g.compass.insert(Camp::Doctor, c);
                        // check the start()::Setup3 for why this is done here
                        g.compass.insert(Camp::Plague, c);
                        g.phase = Phase::Main(1);
                    }
                    if g.is_illegal_setup3() {
                        panic!("Rule violation: Expect Doctor's marker within centor 3x3 on the compass");
                    }
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
            }
        }
        self.tree.borrow().traverse(&load_history, &mut self.game);
    }

    fn read_single_action(&mut self, b: &mut [u8]) {
        // mainly for hjkl-bot, now
        // XXX
        //    Later, change the Map into (PlayerType, TcpStream) so that we can...???
        //    No!!! half-/move-bot cannot share this interface...
        match self.player.get(&self.game.turn) {
            None => {
                self.stdin.read(b).unwrap();
            }
            Some(mut x) => {
                x.read(b).unwrap();
            }
        }
    }

    /// Start the game loop.
    ///
    /// This will listen to events and do the appropriate actions.
    pub fn start(&mut self) -> Result<String, String> {
        loop {
            match self.game.phase {
                Phase::Setup0 => {
                    // already initialized by Game::init randomly, so nothing to be done here
                    self.game.phase = Phase::Setup1;
                }
                Phase::Setup1 => {
                    let mut i = SETUP1_MARKER;
                    while i > 0 {
                        let mut b = [0];
                        self.highlight(true);
                        self.update();
                        self.read_single_action(&mut b);
                        self.highlight(false);
                        match b[0] {
                            b'h' | b'j' | b'k' | b'l' => {
                                let d = Direction::new(b[0]);
                                match self.roam(d) {
                                    Ok(_) => {}
                                    Err(_) => {
                                        panic!("Impossible!!!");
                                    }
                                }
                            }
                            b' ' => {
                                let t = self.game.turn;
                                self.game.add_marker(&self.cursor.env, &t);
                                i = i - 1;
                            }
                            b'q' => return Ok("".to_string()),
                            _ => {}
                        }
                    }
                    if self.game.is_illegal_setup1() {
                        panic!("Rule violation: The initial 4 markers can not share any rows or columns.");
                    }
                    self.game.phase = Phase::Setup2;
                }
                Phase::Setup2 => {
                    self.game.turn = Camp::Doctor;
                    let mut i = SETUP2_HERO;
                    while i > 0 {
                        let mut b = [0];
                        self.highlight(true);
                        self.update();
                        self.read_single_action(&mut b);
                        self.highlight(false);
                        match b[0] {
                            b'h' | b'j' | b'k' | b'l' => {
                                let d = Direction::new(b[0]);
                                match self.roam(d) {
                                    Ok(_) => {}
                                    Err(_) => {
                                        panic!("Impossible!!!");
                                    }
                                }
                            }
                            b' ' => {
                                self.put_hero();
                                i = i - 1;
                            }
                            b'q' => return Ok("".to_string()),
                            _ => {}
                        }
                    }
                    if self.game.is_illegal_setup2() {
                        panic!("Rule violation: Should be exactly one hero in Humanity and exactly one in Underworld for each side.");
                    }
                    self.game.phase = Phase::Setup3;
                }
                Phase::Setup3 => {
                    print!("{}", cursor::Show);
                    self.highlight(false);
                    self.game.turn = Camp::Doctor;
                    loop {
                        let mut b = [0];
                        self.update();
                        self.read_single_action(&mut b);
                        match b[0] {
                            b'h' | b'j' | b'k' | b'l' => {
                                let d = Direction::new(b[0]);
                                match self.jump(d) {
                                    Ok(_) => {}
                                    Err(_) => {
                                        panic!("Impossible!!!");
                                    }
                                }
                            }
                            b' ' => {
                                // XXX: core crate should provide interface for all setup phases.
                                self.game
                                    .compass
                                    .insert(Camp::Doctor, self.cursor.comp.clone());
                                // Why are we setting Plague's coordinates here?
                                // It is because, soon when we start Main game loop,
                                // we will check compass positions for both camps,
                                // so if Plague's marker on the compass does not exist,
                                // the get().unwrap() will not be available.
                                // To prevent the shit conditions, just setup this at the
                                // same position. This will soon change anyway.
                                self.game
                                    .compass
                                    .insert(Camp::Plague, self.cursor.comp.clone());
                                break;
                            }
                            b'q' => return Ok("".to_string()),
                            _ => {}
                        }
                    }
                    self.game.phase = Phase::Main(1);
                }
                Phase::Main(n) => {
                    // In this stage, many actions will be carried out,
                    // but they shouldn't be commited directly, because
                    //    1. actions can be inreversable
                    //    2. some fault can only be captured with later conditions
                    //
                    // Therefore, make this stage roughly like
                    //    1. game.action1
                    //    2. game.action2
                    //    ...
                    //    n. game.action_n
                    //    n+1. together with all actions above, do the final check
                    //         if all are good, then do game.commit
                    //         otherwise, do game.flush
                    print!("{}", cursor::Show);
                    self.print_hint(n.to_string());
                    if n % 2 == 0 {
                        self.game.turn = Camp::Doctor;
                    } else {
                        self.game.turn = Camp::Plague;
                    }

                    'main_loop: loop {
                        let mut head: Vec<String> = Vec::new();
                        let mut tail: String = String::new();
                        self.game.flush_action();
                        loop {
                            let mut b = [0];
                            self.update();
                            //thread::sleep(time::Duration::from_millis(5000));
                            self.read_single_action(&mut b);
                            match b[0] {
                                b'h' | b'j' | b'k' | b'l' => {
                                    let d = Direction::new(b[0]);
                                    match self.jump(d) {
                                        Ok(x) => {
                                            // Only for bot players: we need to send the status update
                                            if let Err(z) = self.report(&x) {
                                                return Err(z);
                                            }
                                        }
                                        Err(_) => {
                                            panic!("Impossible!!!");
                                        }
                                    }
                                }
                                b'q' => return Ok("".to_string()),
                                b' ' => {
                                    if let Err(x) = self.game.check_action_compass(self.cursor.comp)
                                    {
                                        self.print_hint(x);
                                        continue 'main_loop;
                                    }
                                    let next_compass = self.cursor.comp.compass_to_sgf();
                                    // lockdown?
                                    // if this is Doctor, do you lockdown? if so, degree?
                                    // set a temprary flag for display only,
                                    // if the move is valid, setting it normal
                                    // otherwise, this move has to be restarted anyway
                                    let mut lockdown = Lockdown::Normal;
                                    while self.game.turn == Camp::Doctor
                                        && next_compass == ORIGIN_COMPASS_STR
                                    {
                                        self.print_hint(
                                            "Rotation: (h) No (j) 90 (k) 180 (l) 270".to_string(),
                                        );
                                        let mut bi = [0];
                                        self.read_single_action(&mut bi);
                                        match bi[0] {
                                            b'h' => {
                                                lockdown = Lockdown::Normal;
                                                tail.push(bi[0] as char);
                                                break;
                                            }
                                            b'j' => {
                                                lockdown = Lockdown::CC90;
                                                tail.push(bi[0] as char);
                                                break;
                                            }
                                            b'k' => {
                                                lockdown = Lockdown::CC180;
                                                tail.push(bi[0] as char);
                                                break;
                                            }
                                            b'l' => {
                                                lockdown = Lockdown::CC270;
                                                tail.push(bi[0] as char);
                                                break;
                                            }
                                            b'q' => return Ok("".to_string()),
                                            _ => {}
                                        }
                                    }
                                    self.clear_compass();
                                    if let Err(x) = self.game.check_action_lockdown(lockdown) {
                                        panic!("{}", x);
                                    }
                                    self.update_compass();
                                    head.push(next_compass);
                                    break;
                                }
                                _ => {}
                            }
                        }

                        // set hero
                        // ideally, directly determing if this hero is playable given the position is good,
                        // but it need more complicated control...
                        let mut w = 0;
                        loop {
                            let mut b = [0];
                            let vw = vec![World::Humanity, World::Underworld];
                            let camp = self.game.turn;
                            self.cursor.env = *self.game.hero.get(&(vw[w], camp)).unwrap();
                            self.highlight(true);
                            self.read_single_action(&mut b);
                            self.highlight(false);
                            match b[0] {
                                b'\t' => {
                                    w = 1 - w;
                                }
                                b' ' => {
                                    // index 1..n is the trajectory, this is 1
                                    head.push(self.cursor.env.env_to_sgf());
                                    if let Err(x) = self.game.check_action_hero(self.cursor.env) {
                                        self.print_hint(x);
                                        continue 'main_loop;
                                    }
                                    break;
                                }
                                b'q' => return Ok("".to_string()),
                                _ => {}
                            }
                        }

                        // set moves
                        // collect legal directions first
                        // this paragraph references to core:check_move a lot
                        let mut mov = self.game.action.restriction.clone();
                        let n_steps: usize = self.game.action.steps.try_into().unwrap();
                        let mut steps = n_steps;
                        while steps > 0 {
                            let mut b = [0];
                            self.highlight(true);
                            self.read_single_action(&mut b);
                            self.highlight(false);
                            match b[0] {
                                b'h' | b'j' | b'k' | b'l' => {
                                    let d = Direction::new(b[0]);
                                    match mov.get(&d) {
                                        None => {
                                            let x = "Warning: Wrong Direction".to_string();
                                            if let Err(z) = self.report(&x) {
                                                return Err(z);
                                            }
                                            continue;
                                        }
                                        Some(x) => {
                                            if *x > 1 {
                                                mov.insert(d, *x - 1);
                                            } else {
                                                mov.remove(&d);
                                            }
                                            steps -= 1;
                                        }
                                    }
                                    match self.walk(d) {
                                        Some(x) => {
                                            head.push((*x).to_string());
                                            if let Err(x) =
                                                self.game.check_action_step(self.cursor.env)
                                            {
                                                self.print_hint(x);
                                                continue 'main_loop;
                                            }
                                        }
                                        None => {
                                            let x = "Error: Wrong Direction".to_string();
                                            if let Err(z) = self.report(&x) {
                                                return Err(z);
                                            }
                                            continue 'main_loop;
                                        }
                                    }
                                }
                                b' ' => {}
                                b'q' => return Ok("".to_string()),
                                _ => {}
                            }
                        }

                        // set markers
                        let mut t = 0; // valid value: 0..n_steps-1, because the stop position doesn't count
                        let mut quota = if self.game.turn == Camp::Doctor {
                            crate::core::DOCTOR_MARKER
                        } else {
                            crate::core::PLAGUE_MARKER
                        };
                        while quota > 0 {
                            let mut b = [0];
                            self.cursor.env = self.game.sgf_to_env(&head[t + 1]);
                            self.highlight(true);
                            self.read_single_action(&mut b);
                            self.highlight(false);
                            match b[0] {
                                b'\t' => {
                                    t = if t < n_steps - 1 { t + 1 } else { 0 };
                                }
                                b' ' => {
                                    // index 1..n is the trajectory, this is 1
                                    quota -= 1;
                                    if let Err(x) = self.game.check_action_marker(self.cursor.env) {
                                        self.print_hint(x);
                                        continue 'main_loop;
                                    }
                                    //head.push(self.cursor.env.env_to_sgf());
                                    self.update();
                                }
                                b'q' => return Ok("".to_string()),
                                _ => {}
                            }
                        }

                        //if self.game.turn == Camp::Doctor {
                        //    head.push(tail);
                        //}
                        self.clear_compass();
                        //self.game.apply_move(&head);
                        self.game.commit_action();
                        self.game.next();
                    }
                }
            }
        }
    }

    fn print_hint_off(&mut self, s: String, off: i32) {
        let sc = self.cursor.to_hint();
        write!(
            self.stdout,
            "{}{}",
            cursor::Goto((sc.x + off).try_into().unwrap(), (sc.y).try_into().unwrap(),),
            s,
        )
        .unwrap();
        self.stdout.flush().unwrap();
    }

    fn print_hint(&mut self, s: String) {
        self.print_hint_off("                                        ".to_string(), 0);
        self.print_hint_off(s, 0);
    }

    // We do Env part dynamically here because they are backgrounds,
    // so we cannot just based on some static map and draw them.
    // On the other hand, we can just draw a grid as the Compass,
    // but the overlapping between it and the Env causes me headache,
    // so still dynamically generate it here.
    fn init(&mut self) {
        print!("{}", cursor::Hide);
        self.draw_rect(
            Coord::new(ORIGIN.x + 1, ORIGIN.y + 1),
            Coord::new(ARENA.x, ARENA.y),
            GRAY,
        );
        self.init_env();
        self.init_compass();
    }
    fn init_env(&mut self) {
        for i in 0..crate::core::SIZE {
            for j in 0..crate::core::SIZE {
                let c = Coord::new(i, j);
                self.cursor.env = c.clone();
                match self.game.env.get(&c).unwrap() {
                    World::Humanity => {
                        self.draw_rect(
                            Coord::new(i * XUNIT + XBORDER, j * YUNIT + YBORDER + 1),
                            Coord::new((i + 1) * XUNIT, (j + 1) * YUNIT + 1),
                            WHITE,
                        );
                    }
                    World::Underworld => {
                        self.draw_rect(
                            Coord::new(i * XUNIT + XBORDER, j * YUNIT + YBORDER + 1),
                            Coord::new((i + 1) * XUNIT, (j + 1) * YUNIT + 1),
                            BLACK,
                        );
                    }
                }
                self.update();
            }
        }
        self.cursor.env = ORIGIN_ENV;
    }
    fn init_compass(&mut self) {
        let cx = ORIGIN_COMPASS.x - 2 * XUNIT_COMPASS;
        let mut cy = ORIGIN_COMPASS.y - 2 * YUNIT_COMPASS;
        write!(self.stdout, "{}{}", color::Fg(ASH), color::Bg(GRAY),).unwrap();
        write!(
            self.stdout,
            "{}",
            cursor::Goto(cx.try_into().unwrap(), cy.try_into().unwrap()),
        )
        .unwrap();
        write!(self.stdout, "    ┏┅┅┅┳┅┅┅┳┅┅┅┓").unwrap();
        cy = cy + 1;
        write!(
            self.stdout,
            "{}",
            cursor::Goto(cx.try_into().unwrap(), cy.try_into().unwrap()),
        )
        .unwrap();
        write!(self.stdout, "    ┇   ┇   ┇   ┇").unwrap();
        cy = cy + 1;
        write!(
            self.stdout,
            "{}",
            cursor::Goto(cx.try_into().unwrap(), cy.try_into().unwrap()),
        )
        .unwrap();
        write!(self.stdout, "┏┅┅┅╋━━━╋━━━╋━━━╋┅┅┅┓").unwrap();
        cy = cy + 1;
        write!(
            self.stdout,
            "{}",
            cursor::Goto(cx.try_into().unwrap(), cy.try_into().unwrap()),
        )
        .unwrap();
        write!(self.stdout, "┇   ┃   ┃   ┃   ┃   ┇").unwrap();
        cy = cy + 1;
        write!(
            self.stdout,
            "{}",
            cursor::Goto(cx.try_into().unwrap(), cy.try_into().unwrap()),
        )
        .unwrap();
        write!(self.stdout, "┣┅┅┅╋━━━╋━━━╋━━━╋┅┅┅┫").unwrap();
        cy = cy + 1;
        write!(
            self.stdout,
            "{}",
            cursor::Goto(cx.try_into().unwrap(), cy.try_into().unwrap()),
        )
        .unwrap();
        write!(self.stdout, "┇   ┃   ┃   ┃   ┃   ┇").unwrap();
        cy = cy + 1;
        write!(
            self.stdout,
            "{}",
            cursor::Goto(cx.try_into().unwrap(), cy.try_into().unwrap()),
        )
        .unwrap();
        write!(self.stdout, "┣┅┅┅╋━━━╋━━━╋━━━╋┅┅┅┫").unwrap();
        cy = cy + 1;
        write!(
            self.stdout,
            "{}",
            cursor::Goto(cx.try_into().unwrap(), cy.try_into().unwrap()),
        )
        .unwrap();
        write!(self.stdout, "┇   ┃   ┃   ┃   ┃   ┇").unwrap();
        cy = cy + 1;
        write!(
            self.stdout,
            "{}",
            cursor::Goto(cx.try_into().unwrap(), cy.try_into().unwrap()),
        )
        .unwrap();
        write!(self.stdout, "┗┅┅┅╋━━━╋━━━╋━━━╋┅┅┅┛").unwrap();
        cy = cy + 1;
        write!(
            self.stdout,
            "{}",
            cursor::Goto(cx.try_into().unwrap(), cy.try_into().unwrap()),
        )
        .unwrap();
        write!(self.stdout, "    ┇   ┇   ┇   ┇").unwrap();
        cy = cy + 1;
        write!(
            self.stdout,
            "{}",
            cursor::Goto(cx.try_into().unwrap(), cy.try_into().unwrap()),
        )
        .unwrap();
        write!(self.stdout, "    ┗┅┅┅┻┅┅┅┻┅┅┅┛").unwrap();
        self.stdout.flush().unwrap();
    }

    fn put_hero(&mut self) {
        if self.game.phase != Phase::Setup2 {
            panic!("Don't touch the heroes!");
        }
        match self.game.env.get(&self.cursor.env) {
            Some(x) => {
                self.game
                    .hero
                    .insert((x.clone(), self.game.turn.clone()), self.cursor.env.clone());
            }
            None => {
                panic!("Why is board not set?");
            }
        }
        self.game.switch();
    }

    fn highlight(&mut self, on: bool) {
        let mut fg = RED;
        if self.game.turn == Camp::Doctor {
            fg = BLUE;
        }
        write!(
            self.stdout,
            "{}{}{}",
            color::Fg(fg),
            style::Bold,
            color::Bg(GRAY),
        )
        .unwrap();
        let ca = self.cursor.block_range();
        let b = vec!["┏", "┓", "┗", "┛", "━", "┃", "┅", "┇"];
        let lt = if on == true { b[0] } else { " " };
        let rt = if on == true { b[1] } else { " " };
        let lb = if on == true { b[2] } else { " " };
        let rb = if on == true { b[3] } else { " " };
        write!(
            self.stdout,
            "{}{}{}{}{}{}{}{}",
            cursor::Goto(
                (ca[0].x - 1).try_into().unwrap(),
                (ca[0].y - 1).try_into().unwrap()
            ),
            lt,
            cursor::Goto(
                ca[1].x.try_into().unwrap(),
                (ca[0].y - 1).try_into().unwrap()
            ),
            rt,
            cursor::Goto(
                (ca[0].x - 1).try_into().unwrap(),
                (ca[1].y).try_into().unwrap()
            ),
            lb,
            cursor::Goto(ca[1].x.try_into().unwrap(), (ca[1].y).try_into().unwrap()),
            rb,
        )
        .unwrap();

        let h = if on == true { b[4] } else { " " };
        let v = if on == true { b[5] } else { " " };
        for i in ca[0].x..ca[1].x {
            write!(
                self.stdout,
                "{}{}{}{}",
                cursor::Goto(i.try_into().unwrap(), (ca[0].y - 1).try_into().unwrap()),
                h,
                cursor::Goto(i.try_into().unwrap(), (ca[1].y).try_into().unwrap()),
                h,
            )
            .unwrap();
        }
        for j in ca[0].y..ca[1].y {
            write!(
                self.stdout,
                "{}{}{}{}",
                cursor::Goto((ca[0].x - 1).try_into().unwrap(), j.try_into().unwrap()),
                v,
                cursor::Goto((ca[1].x).try_into().unwrap(), j.try_into().unwrap()),
                v,
            )
            .unwrap();
        }
        let temp = self.cursor.to_marker(Type::Compass);
        // always show the cursor at the compass side
        write!(
            self.stdout,
            "{}",
            cursor::Goto(temp.x.try_into().unwrap(), temp.y.try_into().unwrap()),
        )
        .unwrap();
        self.stdout.flush().unwrap();
    }

    fn draw_rect(&mut self, lt: Coord, rb: Coord, c: color::Rgb) {
        for i in lt.x..rb.x {
            for j in lt.y..rb.y {
                write!(
                    self.stdout,
                    "{}{} ",
                    cursor::Goto(i.try_into().unwrap(), j.try_into().unwrap()),
                    color::Bg(c),
                )
                .unwrap();
            }
        }
        self.stdout.flush().unwrap();
    }

    /// Move the cursor to the player position.
    fn update(&mut self) {
        let c = self.cursor.to_marker(Type::Env);
        let mut bg = BLACK;
        if self.game.env.get(&self.cursor.env).unwrap() == &World::Humanity {
            bg = WHITE;
        }
        let mut m = 0;
        match self.game.stuff.get(&self.cursor.env) {
            Some((camp, x)) => {
                if *camp == Camp::Doctor {
                    write!(self.stdout, "{}{}", color::Fg(BLUE), color::Bg(bg)).unwrap();
                } else {
                    write!(self.stdout, "{}{}", color::Fg(RED), color::Bg(bg)).unwrap();
                }
                match x {
                    Stuff::Marker(y) => {
                        m = *y;
                    }
                    Stuff::Colony => {
                        m = 6;
                    }
                }
            }
            None => {}
        }
        if m != 0 {
            write!(
                self.stdout,
                "{}{}",
                cursor::Goto(c.x.try_into().unwrap(), c.y.try_into().unwrap()),
                m
            )
            .unwrap();
        } else {
            write!(
                self.stdout,
                "{}{} ",
                cursor::Goto(c.x.try_into().unwrap(), c.y.try_into().unwrap()),
                color::Bg(bg),
            )
            .unwrap();
        }

        // Env mark heros
        let hc = self.cursor.to_hero();
        for ((_, camp), value) in self.game.hero.iter() {
            if *value == self.cursor.env {
                let fg = if *camp == Camp::Doctor {
                    color::Fg(BLUE)
                } else {
                    color::Fg(RED)
                };
                write!(
                    self.stdout,
                    "{}{}X",
                    cursor::Goto(hc.x.try_into().unwrap(), hc.y.try_into().unwrap(),),
                    fg,
                )
                .unwrap();
            }
        }
        self.update_compass();
        self.stdout.flush().unwrap();
    }

    fn clear_compass(&mut self) {
        for (_, coord) in self.game.compass.iter() {
            let temp = self.cursor.comp.clone();
            self.cursor.comp = coord.lockdown(self.game.action.lockdown);
            let c = self.cursor.to_marker(Type::Compass);
            let bg = color::Bg(GRAY);
            write!(
                self.stdout,
                "{}{} ",
                cursor::Goto(c.x.try_into().unwrap(), c.y.try_into().unwrap()),
                bg,
            )
            .unwrap();
            self.cursor.comp = temp.clone();
        }
    }

    fn update_compass(&mut self) {
        // Compass
        // XXX: need to fix
        //     Setup3: should display Plague first so that it is overwritten
        //     Main: mis-applying one more
        let temp = self.cursor.comp.clone();
        for (camp, coord) in self.game.compass.iter() {
            let ldc = coord.lockdown(self.game.action.lockdown);
            self.cursor.comp = ldc.clone();
            let c2 = self.cursor.to_marker(Type::Compass);
            let fg2 = if *camp == Camp::Doctor {
                color::Fg(BLUE)
            } else {
                color::Fg(RED)
            };
            let bg2 = color::Bg(GRAY);
            write!(
                self.stdout,
                "{}{}{}X",
                cursor::Goto(c2.x.try_into().unwrap(), c2.y.try_into().unwrap()),
                fg2,
                bg2,
            )
            .unwrap();
        }
        // resume the cursor
        self.cursor.comp = temp.clone();
        let c3 = self.cursor.to_marker(Type::Compass);
        write!(
            self.stdout,
            "{}",
            cursor::Goto(c3.x.try_into().unwrap(), c3.y.try_into().unwrap()),
        )
        .unwrap();
        self.stdout.flush().unwrap();
    }

    /// [Env] walk the grids with highlight
    fn walk(&mut self, dir: Direction) -> Option<String> {
        let e = self.game.env.get(&self.cursor.env).unwrap();
        let mut ret: Option<String> = None;
        let mut walked = false;
        match dir {
            Direction::Right => {
                while self.cursor.env.x < crate::core::SIZE - 1 {
                    self.cursor.env.x += 1;
                    let e2 = self.game.env.get(&self.cursor.env).unwrap();
                    if e == e2 {
                        walked = true;
                        ret = Some(self.cursor.env.env_to_sgf());
                        break;
                    }
                }
            }
            Direction::Left => {
                while self.cursor.env.x > ORIGIN.x {
                    self.cursor.env.x -= 1;
                    let e2 = self.game.env.get(&self.cursor.env).unwrap();
                    if e == e2 {
                        walked = true;
                        ret = Some(self.cursor.env.env_to_sgf());
                        break;
                    }
                }
            }
            Direction::Up => {
                while self.cursor.env.y > ORIGIN.y {
                    self.cursor.env.y -= 1;
                    let e2 = self.game.env.get(&self.cursor.env).unwrap();
                    if e == e2 {
                        walked = true;
                        ret = Some(self.cursor.env.env_to_sgf());
                        break;
                    }
                }
            }
            Direction::Down => {
                while self.cursor.env.y < crate::core::SIZE - 1 {
                    self.cursor.env.y += 1;
                    let e2 = self.game.env.get(&self.cursor.env).unwrap();
                    if e == e2 {
                        walked = true;
                        ret = Some(self.cursor.env.env_to_sgf());
                        break;
                    }
                }
            }
        }
        if walked {
            return ret;
        } else {
            return None;
        }
    }

    // roam is for the env part and jump is for the compass part
    // No Err is possible here, and we ignore the Ok content for Setups for now
    fn roam(&mut self, dir: Direction) -> Result<String, String> {
        let r = self.cursor.go(dir, Type::Env);
        self.stdout.flush().unwrap();
        thread::sleep(time::Duration::from_millis(10));
        r
    }

    fn jump(&mut self, dir: Direction) -> Result<String, String> {
        let r = self.cursor.go(dir, Type::Compass);
        self.stdout.flush().unwrap();
        thread::sleep(time::Duration::from_millis(10));
        r
    }
}

impl Drop for Board {
    fn drop(&mut self) {
        // When done, restore the defaults to avoid messing with the terminal.
        print!("{}", cursor::Show);
        write!(
            self.stdout,
            "{}{}{}",
            clear::All,
            style::Reset,
            cursor::Goto(1, 1)
        )
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cursor_new() {
        let c = Cursor::new();
        assert_eq!(c.comp.x, 0);
        assert_eq!(c.env.x, 1);
    }
}
