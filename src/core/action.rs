use super::grid_coord::*;
use super::*;
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub enum ActionPhase {
    SetMap,
    Lockdown,
    SetCharacter,
    BoardMove,
    SetMarkers,
    Done,
}

#[derive(Debug)]
pub struct Action {
    pub map: Option<Coord>,
    pub lockdown: Lockdown,
    pub character: Option<Coord>,
    pub world: Option<World>,
    pub restriction: HashMap<Direction, i32>,
    pub steps: i32,
    pub trajectory: Vec<Coord>,
    pub markers: Vec<Coord>,
    pub action_phase: ActionPhase,
}

impl Action {
    pub fn new() -> Action {
        return Action {
            map: None,
            lockdown: Lockdown::Normal,
            character: None,
            world: None,
            restriction: HashMap::new(),
            steps: 0,
            trajectory: Vec::new(),
            markers: Vec::new(),
            action_phase: ActionPhase::SetMap,
        };
    }

    // There are various combos for the following add* functions.

    pub fn add_map_step(&mut self, g: &Game, c: Coord) -> Result<(), &'static str> {
        if *g.map.get(&g.opposite(g.turn)).unwrap() == c {
            return Err("Ex00");
        }
        if *g.map.get(&g.turn).unwrap() == c {
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
        self.map = Some(c);
        self.restriction = c - g.map.get(&g.opposite(g.turn)).unwrap();
        if self.steps != 0 {
            panic!("{:?}:{:?}==", self.steps, self.restriction);
        }
        for (_, i) in self.restriction.iter() {
            self.steps += *i;
        }

        self.transit(g);
        return Ok(());
    }

    pub fn add_lockdown_by_rotation(&mut self, g: &Game, ld: Lockdown) -> Result<(), &'static str> {
        let o = Coord::new(0, 0);
        if g.turn != Camp::Doctor || self.map.unwrap() != o {
            return Err("Ex0E");
        }
        let mut cp = *g.map.get(&Camp::Plague).unwrap();
        cp = cp.lockdown(ld);

        // Update action
        self.lockdown = ld;
        self.restriction = o - &cp;

        self.transit(g);
        return Ok(());
    }

    pub fn add_lockdown_by_coord(&mut self, g: &Game, c: Coord) -> Result<(), &'static str> {
        let o = Coord::new(0, 0);
        if g.turn != Camp::Doctor || self.map.unwrap() != o {
            return Err("Ex0E");
        }
        let cp = *g.map.get(&Camp::Plague).unwrap();
        let lda = vec![
            Lockdown::Normal,
            Lockdown::CC90,
            Lockdown::CC180,
            Lockdown::CC270,
        ];
        let cc: Vec<_> = lda.iter().map(|&x| cp.lockdown(x)).collect();
        if let Some(ld) = cc.iter().position(|&x| x == c) {
            self.lockdown = lda[ld];
            self.restriction = o - &c;
            self.transit(g);
            return Ok(());
        }
        return Err("Ex0F");
    }

    pub fn add_character(&mut self, g: &Game, c: Coord) -> Result<(), &'static str> {
        let hh = *g.character.get(&(World::Humanity, g.turn)).unwrap();
        let hu = *g.character.get(&(World::Underworld, g.turn)).unwrap();

        // Update action
        if c != hh && c != hu {
            return Err("Ex02");
        } else if c == hh {
            self.world = Some(World::Humanity);
        } else {
            self.world = Some(World::Underworld);
        }
        self.character = Some(c);
        self.trajectory.push(c);

        self.transit(g);
        return Ok(());
    }

    // This is intended to be used multiple times, and only when every single step
    // is valid the action is complete. UIs that does not generate valid candidate
    // routes can use this directly.
    pub fn add_board_single_step(&mut self, g: &Game, to: Coord) -> Result<(), &'static str> {
        // impossible number as a implicit assertion
        let mut from = Coord::new(-999, -999);
        if let Some(x) = self.trajectory.last() {
            from = *x;
        }

        // more than 1 steps given the destination being "to"
        let dd = to - &from;
        if dd.len() != 1 {
            return Err("Ex03");
        }

        let mov = self.restriction.clone();
        let w = self.world.unwrap();
        // This is expected to run only once since dd.len() == 1.
        for (d, _) in dd.iter() {
            if mov.get(d) == None {
                return Err("Ex04");
            }

            let mut c = from;
            // This is non-deterministic because of the nature of Pathogen
            // Game board has two Worlds.
            loop {
                // try to move along direction d
                c = c + d;
                match g.env.get(&c) {
                    Some(x) => {
                        // the same world, a valid move
                        if *x == w {
                            // single step and valid, but not the destination?
                            // return error.
                            if c != to {
                                return Err("Ex03");
                            // make sense, get out of this loop
                            } else {
                                break;
                            }
                        // not the same world, but equal to the destination?
                        // return error.
                        } else if c == to {
                            return Err("Ex03");
                        }
                    }
                    None => {
                        return Err("Ex05");
                    }
                }
            }
        }

        let enemy_camp = g.opposite(g.turn);
        match g.stuff.get(&to) {
            Some((c, Stuff::Colony)) => {
                if *c == enemy_camp {
                    // XXX: this is not tested yet
                    return Err("Ex06");
                }
            }
            _ => {}
        }

        // Update action
        self.trajectory.push(to);
        // Finish taking the action steps
        if self.trajectory.len() > self.steps.try_into().unwrap() {
            // Final step: No collision?
            let op = *g.character.get(&(w, enemy_camp)).unwrap();
            if op == to {
                return Err("Ex07");
            }
            self.transit(g);
        }

        self.character = Some(to);
        return Ok(());
    }

    pub fn add_single_marker(&mut self, g: &Game, t: Coord) -> Result<(), &'static str> {
        let quota = if g.turn == Camp::Doctor {
            DOCTOR_MARKER
        } else {
            PLAGUE_MARKER
        };

        let op = g.opposite(g.turn);
        if let Some(oph) = g.character.get(&(self.world.unwrap(), op)) {
            if *oph == t {
                return Err("Ex08");
            }
        }

        if !self.trajectory.contains(&t) {
            return Err("Ex09");
        }

        // Update action
        self.markers.push(t);
        if self.markers.len() > quota.try_into().unwrap() {
            return Err("Ex0A");
        } else if self.markers.len() == quota.try_into().unwrap() {
            // when all markers are given ... most checks are here
            // 1. if there is overflow
            // 2. (Doctor) if any plagues are ignored
            // 3. (Plague) if distributed evenly
            let last = self.trajectory.len() - 1;
            self.trajectory.remove(last);

            // This sort-and-traverse was for Plague only because it would be easier to calculate max/min,
            // but now we need to check if any marker overflows to Colony. Move Plague check here as well
            self.markers.sort_by(|a, b| {
                let na = a.x + SIZE * a.y;
                let nb = b.x + SIZE * b.y;
                nb.cmp(&na)
            });
            let mut cur = 1;
            let m = &self.markers;
            let t = &mut self.trajectory;

            // both side shouldn't count the grid occupied by an opponent
            t.retain(|&y| {
                let hh = *g.character.get(&(World::Humanity, op)).unwrap();
                let hu = *g.character.get(&(World::Underworld, op)).unwrap();
                y != hh && y != hu
            });
            let max = (PLAGUE_MARKER as f64 / t.len() as f64).ceil() as u8;
            let min = (PLAGUE_MARKER as f64 / t.len() as f64).floor() as u8;
            for i in 1..=m.len() {
                // check the total markers under two conditions
                // 1. all markers reside in the same position, or
                // 2. there are different markers in this move
                if i == m.len() || m[i - 1] != m[i] {
                    if let Some((c, Stuff::Marker(x))) = g.stuff.get(&m[i - 1]) {
                        if *c == g.turn && MAX_MARKER < x + cur {
                            return Err("Ex0B");
                        }
                    }
                    if g.turn == Camp::Plague && cur != max && cur != min {
                        return Err("Ex0C");
                    }
                    cur = 1;
                } else {
                    cur += 1;
                }
            }

            if g.turn == Camp::Doctor {
                // need to clean the plague first
                // priorities the marker placement
                t.sort_by(|a, b| {
                    let mut a_sick = false;
                    let mut b_sick = false;
                    if let Some((Camp::Plague, Stuff::Marker(_))) = g.stuff.get(a) {
                        a_sick = true;
                    }
                    if let Some((Camp::Plague, Stuff::Marker(_))) = g.stuff.get(b) {
                        b_sick = true;
                    }
                    b_sick.cmp(&a_sick)
                });
                // on encountering plagues, cure them
                // so, are they cured?
                let mut m = self.markers.clone();
                for c in t.iter() {
                    let before = m.len();
                    if before == 0 {
                        break;
                    }
                    if let Some((Camp::Plague, Stuff::Marker(x))) = g.stuff.get(c) {
                        m.retain(|&y| y != *c);
                        let after = m.len();
                        if before - after < *x as usize && after != 0 {
                            return Err("Ex0D");
                        }
                    }
                }
            }
            self.transit(g);
        }
        return Ok(());
    }

    pub fn to_sgf_string(&self, g: &Game) -> String {
        let mut v = Vec::<String>::new();
        let m = if g.turn == Camp::Doctor { "(;W" } else { "(;B" };
        // map
        let temp = self.map.unwrap();
        v.push(temp.map_to_sgf());

        // lockdown
        if self.lockdown != Lockdown::Normal {
            let mut cp = *g.map.get(&Camp::Plague).unwrap();
            cp = cp.lockdown(self.lockdown);
            v.push(cp.map_to_sgf());
        }

        // trajectory: in reverse order
        let mut i = self.steps - 1;
        while i >= 0 {
            v.push(self.trajectory[i as usize].env_to_sgf());
            i = i - 1;
        }
        // character
        let ch = self.character.unwrap();
        v.push(ch.env_to_sgf());

        // markers
        for c in self.markers.iter() {
            v.push(c.env_to_sgf());
        }

        let s = String::from(m) + "[" + &v.join("][") + "])";
        return s;
    }

    fn transit(&mut self, g: &Game) {
        match &self.action_phase {
            ActionPhase::SetMap => {
                if let Some(c) = self.map {
                    if c == ORIGIN && g.turn == Camp::Doctor {
                        self.action_phase = ActionPhase::Lockdown;
                        return;
                    }
                }
                self.action_phase = ActionPhase::SetCharacter;
            }
            ActionPhase::Lockdown => {
                self.action_phase = ActionPhase::SetCharacter;
            }
            ActionPhase::SetCharacter => {
                self.action_phase = ActionPhase::BoardMove;
            }
            ActionPhase::BoardMove => {
                self.action_phase = ActionPhase::SetMarkers;
            }
            ActionPhase::SetMarkers => {
                self.action_phase = ActionPhase::Done;
            }
            _ => {
                panic!("transit after done?");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map1() {
        let mut g = Game::init(None);
        let c1 = Coord::new(2, 4);
        let c2 = Coord::new(2, 3);
        let mut a = Action::new();
        g.set_map(Camp::Doctor, c2);
        g.set_map(Camp::Plague, c1);
        if let Err(e) = a.add_map_step(&g, c2) {
            assert_eq!(e, "Ex00");
        }
    }

    #[test]
    fn test_lockdown() {
        let mut g = Game::init(None);
        g.turn = Camp::Doctor;
        let cp = Coord::new(-2, -1);
        g.map.insert(Camp::Doctor, cp);
        g.map.insert(Camp::Plague, cp);
        let mut a = Action::new();
        let cd = Coord::new(0, 0);
        assert!(a.add_map_step(&g, cd).is_ok());
        assert_eq!(*a.restriction.get(&Direction::Right).unwrap(), 2);
        assert_eq!(*a.restriction.get(&Direction::Down).unwrap(), 1);
        assert!(a.add_lockdown_by_coord(&g, Coord::new(-1, 2)).is_ok());
        assert_eq!(*a.restriction.get(&Direction::Up).unwrap(), 2);
        assert_eq!(*a.restriction.get(&Direction::Right).unwrap(), 1);
    }

    #[test]
    fn test_character() {
        let mut g = Game::init(None);
        let ch = Coord::new(4, 4);
        let cu = Coord::new(2, 3);
        g.character.insert((World::Humanity, Camp::Plague), ch);
        g.character.insert((World::Underworld, Camp::Plague), cu);
        let mut a = Action::new();
        let c = Coord::new(1, 5);
        if let Err(e) = a.add_character(&g, c) {
            assert_eq!(e, "Ex02");
        }
        let r1 = a.add_character(&g, ch);
        assert!(r1.is_ok());
        assert_eq!(a.trajectory.len(), 1);
        let r2 = a.add_character(&g, cu);
        assert!(r2.is_ok());
        assert_eq!(a.trajectory.len(), 2);
        assert_eq!(a.character.unwrap(), cu);
    }

    #[test]
    fn test_integrate1() {
        let mut g = Game::init(None);
        // For not panic the functions
        let ch2 = Coord::new(-2, -2);
        g.env.insert(ch2, World::Underworld);
        g.map.insert(Camp::Doctor, ch2);
        g.character.insert((World::Underworld, Camp::Doctor), ch2);
        // what really necessary
        g.turn = Camp::Doctor;
        g.map.insert(Camp::Plague, Coord::new(-1, -2));
        let ch = Coord::new(3, 4);
        g.character.insert((World::Humanity, Camp::Doctor), ch);
        g.env.insert(ch, World::Humanity);
        // A failed route
        let cf1 = Coord::new(3, 5);
        g.env.insert(cf1, World::Underworld);
        // A longer, failed route
        let clf1 = Coord::new(4, 4);
        let clf2 = Coord::new(5, 4);
        let clf3 = Coord::new(5, 5);
        g.env.insert(clf1, World::Underworld);
        g.env.insert(clf2, World::Humanity);
        g.env.insert(clf3, World::Underworld);

        let mut a = Action::new();
        assert_eq!(a.action_phase, ActionPhase::SetMap);
        let r1 = a.add_map_step(&g, Coord::new(0, -1));
        assert!(r1.is_ok());
        assert_eq!(a.action_phase, ActionPhase::SetCharacter);
        let r2 = a.add_character(&g, ch);
        assert!(r2.is_ok());
        assert_eq!(a.action_phase, ActionPhase::BoardMove);
        if let Err(e) = a.add_board_single_step(&g, cf1) {
            assert_eq!(e, "Ex03");
        }
        let r4 = a.add_board_single_step(&g, clf2);
        assert!(r4.is_ok());
        if let Err(e) = a.add_board_single_step(&g, clf3) {
            assert_eq!(e, "Ex03");
        }
    }
}
