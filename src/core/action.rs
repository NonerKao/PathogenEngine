use super::grid_coord::*;
use super::*;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Action {
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

    pub fn add_compass_step(&mut self, g: &Game, c: Coord) -> Result<(), &'static str> {
        if *g.compass.get(&g.opposite(g.turn)).unwrap() == c {
            return Err("Ex00");
        }
        if *g.compass.get(&g.turn).unwrap() == c {
            // self.game.unwrap().next();
            // Previously this changes the state of Game.
            // This is not clean because it implies that a check in the action
            // formation causes the change of the state of Game. We should
            // avoid such an implicit behavior.
            // XXX: But then, who will do the transition?
            // Also remove the match g.phase block below because it doesn't make
            // sense that the Plague must not skip at the 1st round.
            return Err("Ix00");
        }
        if (g.lockdown() && g.turn == Camp::Plague) || g.turn == Camp::Doctor {
            // Plague cannot outbreak when lockdown
            if c.x < -1 || c.x > 1 || c.y < -1 || c.y > 1 {
                return Err("Ex01");
            }
        }

        // Update action
        self.compass = Some(c);
        self.restriction = c - g.compass.get(&g.opposite(g.turn)).unwrap();
        if self.steps != 0 {
            panic!("{:?}:{:?}==", self.steps, self.restriction);
        }
        for (_, i) in self.restriction.iter() {
            self.steps += *i;
        }
        return Ok(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compass1() {
        let mut g = Game::init();
        let c1 = Coord::new(2, 4);
        let c2 = Coord::new(2, 3);
        let mut a = Action::new();
        g.set_compass(Camp::Doctor, c2);
        g.set_compass(Camp::Plague, c1);
        if let Err(e) = a.add_compass_step(&g, c2) {
            assert_eq!(e, "Ex00");
        }
    }
}
