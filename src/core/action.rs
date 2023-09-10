use super::grid_coord::*;
use super::*;
//use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug)]
pub struct Action {
    game: Option<Rc<RefCell<Game>>>,
    pub compass: Option<Coord>,
    pub lockdown: Lockdown,
    pub hero: Option<Coord>,
    pub world: Option<World>,
    pub restriction: HashMap<Direction, i32>,
    pub steps: i32,
    pub trajectory: Vec<Coord>,
    pub markers: Vec<Coord>,
}

impl Action {
    pub fn new() -> Action {
        return Action {
            game: None,
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

    pub fn set_game(&mut self, g: &Rc<RefCell<Game>>) {
        self.game = Some(g.clone());
    }

    pub fn check_action_compass(&mut self, c: Coord) -> Result<String, String> {
        let mut ret = Ok("".to_string());
        if *self
            .game
            .as_ref()
            .unwrap()
            .borrow()
            .compass
            .get(
                &self
                    .game
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .opposite(self.game.as_ref().unwrap().borrow().turn),
            )
            .unwrap()
            == c
        {
            return Err("Collide with opponent".to_string());
        }
        if *self
            .game
            .as_ref()
            .unwrap()
            .borrow()
            .compass
            .get(&self.game.as_ref().unwrap().borrow().turn)
            .unwrap()
            == c
        {
            // self.game.unwrap().next();
            // Previously this changes the state of Game.
            // This is not clean because it implies that a check in the action
            // formation causes the change of the state of Game. We should
            // avoid such an implicit behavior.
            // XXX: But then, who will do the transition?
            return Err("Skip".to_string());
        }
        if (self.game.as_ref().unwrap().borrow().lockdown()
            && self.game.as_ref().unwrap().borrow().turn == Camp::Plague)
            || self.game.as_ref().unwrap().borrow().turn == Camp::Doctor
        {
            // Plague cannot outbreak when lockdown
            if c.x < -1 || c.x > 1 || c.y < -1 || c.y > 1 {
                return Err("Exceed valid compass area".to_string());
            }
        }
        if *self
            .game
            .as_ref()
            .unwrap()
            .borrow()
            .compass
            .get(&self.game.as_ref().unwrap().borrow().turn)
            .unwrap()
            == c
        {
            match self.game.as_ref().unwrap().borrow().phase {
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
        self.compass = Some(c);
        self.restriction = c - self
            .game
            .as_ref()
            .unwrap()
            .borrow()
            .compass
            .get(
                &self
                    .game
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .opposite(self.game.as_ref().unwrap().borrow().turn),
            )
            .unwrap();
        if self.steps != 0 {
            panic!("{:?}:{:?}==", self.steps, self.restriction);
        }
        for (_, i) in self.restriction.iter() {
            self.steps += *i;
        }
        return ret;
    }
}
