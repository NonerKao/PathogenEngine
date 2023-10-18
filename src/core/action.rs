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

#[derive(Debug, Clone)]
pub struct Candidate {
    lockdown: Lockdown,
    character: Coord,
    trajectory: Vec<Coord>,
}

impl Candidate {
    pub fn new() -> Candidate {
        return Candidate {
            lockdown: Lockdown::Normal,
            character: Coord::new(-999, -999),
            trajectory: Vec::new(),
        };
    }
}

#[derive(Debug)]
pub struct Action {
    pub map: Option<Coord>,
    pub lockdown: Lockdown,
    pub character: Option<Coord>,
    pub world: Option<World>,
    pub restriction: HashMap<Direction, i32>,
    pub steps: usize,
    pub trajectory: Vec<Coord>,
    pub markers: Vec<Coord>,
    pub action_phase: ActionPhase,
    pub marker_slot: Vec<(Coord, u8)>,
    pub candidate: Vec<Candidate>,
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
            marker_slot: Vec::new(),
            candidate: Vec::new(),
        };
    }

    // There are various combos for the following add* functions.

    pub fn add_map_step(&mut self, g: &Game, c: Coord) -> Result<&'static str, &'static str> {
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
            return Ok("Ix00");
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
            self.steps += *i as usize;
        }

        // Check if the choice is valid immediately.
        // It is possible that the chosen map position does not work
        // for both characters. Check it now.
        let mut possible_route: Vec<Vec<Direction>> = Vec::new();
        let is_lockdown = g.turn == Camp::Doctor && c == ORIGIN;
        self.find_route(&mut possible_route, is_lockdown);
        let total = possible_route.len();
        for (i, pr) in possible_route.iter().enumerate() {
            // Previously, the design was aiming at finding one viable route,
            // but now I want to travese the whole possible trajectories and
            // record them as candidates for later verification.
            let mut lockdown_type = Lockdown::Normal;
            if is_lockdown {
                if total / 4 <= i && i < total / 2 {
                    lockdown_type = Lockdown::CC90;
                } else if total / 2 <= i && i < 3 * total / 4 {
                    lockdown_type = Lockdown::CC180;
                } else if 3 * total / 4 <= i && i < total {
                    lockdown_type = Lockdown::CC270;
                }
            }
            self.traverse(g, pr, lockdown_type, None);
        }

        if self.candidate.len() == 0 {
            // Cleanup self. It is recommanded that the application should
            // clean up the state after seeing this error. Anyway we also
            // wipe out the three attributes we have set above.
            self.map = None;
            self.restriction = HashMap::new();
            self.steps = 0;
            return Err("Ex20");
        } else {
            self.transit(g);
            return Ok("Ix01");
        }
    }

    pub fn add_lockdown_by_rotation(
        &mut self,
        g: &Game,
        ld: Lockdown,
    ) -> Result<&'static str, &'static str> {
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
        return Ok("Ix01");
    }

    pub fn add_lockdown_by_coord(
        &mut self,
        g: &Game,
        c: Coord,
    ) -> Result<&'static str, &'static str> {
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
            let mut temp_candidate = self.candidate.clone();
            temp_candidate.retain(|candidate| candidate.lockdown == lda[ld]);
            if temp_candidate.len() == 0 {
                return Err("Ex22");
            }
            self.candidate
                .retain(|candidate| candidate.lockdown == lda[ld]);
            self.lockdown = lda[ld];
            self.restriction = o - &c;
            self.transit(g);
            return Ok("Ix01");
        }
        return Err("Ex0F");
    }

    pub fn add_character(&mut self, g: &Game, c: Coord) -> Result<&'static str, &'static str> {
        let mut temp_candidate = self.candidate.clone();
        temp_candidate.retain(|candidate| candidate.character == c);
        if temp_candidate.len() == 0 {
            return Err("Ex21");
        }
        self.candidate.retain(|candidate| candidate.character == c);
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
        return Ok("Ix01");
    }

    // This is intended to be used multiple times, and only when every single step
    // is valid the action is complete. UIs that does not generate valid candidate
    // routes can use this directly.
    pub fn add_board_single_step(
        &mut self,
        g: &Game,
        to: Coord,
    ) -> Result<&'static str, &'static str> {
        let target_index = self.trajectory.len();
        let mut temp_candidate = self.candidate.clone();
        temp_candidate.retain(|candidate| candidate.trajectory[target_index] == to);
        if temp_candidate.len() == 0 {
            return Err("Ex23");
        }
        self.candidate
            .retain(|candidate| candidate.trajectory[target_index] == to);
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
        if self.trajectory.len() <= self.steps {
            return Ok("Ix01");
        }
        // Final step: No collision?
        let op = *g.character.get(&(w, enemy_camp)).unwrap();
        if op == to {
            let _ = self.trajectory.pop().unwrap();
            return Err("Ex07");
        }

        self.transit(g);
        // Based on the established trajectory, make the list
        // of available grids: excluding opponent's characters,
        // own's colony, and already full due to neighboring
        // colony ones.
        self.prepare_for_marker(g);

        self.character = Some(to);
        if self.marker_slot.len() == 0 {
            // Also, after excluding all, we should juist return DONE?
            return Ok("Ix02");
        } else {
            return Ok("Ix01");
        }
    }

    fn prepare_for_marker(&mut self, g: &Game) {
        let mut t = self.trajectory.clone();
        let _ = t.pop();
        t.retain(|&y| {
            let op = g.opposite(g.turn);
            let hh = *g.character.get(&(World::Humanity, op)).unwrap();
            let hu = *g.character.get(&(World::Underworld, op)).unwrap();
            y != hh && y != hu
        });
        for c in t.iter() {
            let n = g.get_marker_capacity(*c, None);
            if n > 0 && !self.marker_slot.contains(&(*c, n)) {
                self.marker_slot.push((*c, n));
            }
        }
        self.marker_slot.sort_by(|a, b| {
            let mut a_sick = false;
            let mut b_sick = false;
            if let Some((Camp::Plague, Stuff::Marker(_))) = g.stuff.get(&a.0) {
                a_sick = true;
            }
            if let Some((Camp::Plague, Stuff::Marker(_))) = g.stuff.get(&b.0) {
                b_sick = true;
            }
            b_sick.cmp(&a_sick)
        });
    }

    pub fn add_single_marker_plague(
        &mut self,
        g: &Game,
        c: Coord,
    ) -> Result<&'static str, &'static str> {
        let max = (PLAGUE_MARKER as f64 / self.marker_slot.len() as f64).ceil() as u8;
        let min = (PLAGUE_MARKER as f64 / self.marker_slot.len() as f64).floor() as u8;
        return Ok("Ix01");
    }

    fn doctor_is_free(&self, g: &Game, c: Coord) -> bool {
        let mut is_free = false;
        let cure_count = self
            .markers
            .iter()
            .map(|mc| if *mc == c { 1 } else { 0 })
            .sum::<u8>();
        if let Some((Camp::Plague, Stuff::Marker(x))) = g.stuff.get(&c) {
            if *x - cure_count <= 0 {
                is_free = true;
            }
        } else {
            is_free = true;
        }
        return is_free;
    }

    pub fn add_single_marker_doctor(
        &mut self,
        g: &Game,
        c: Coord,
    ) -> Result<&'static str, &'static str> {
        // Can we freely distribute vaccines now?
        // With the assumption that the sickness is sorted.
        // We should be safe to do so because marker_slot.len() == 0
        // implies DONE in previous SetMarker.
        match g.stuff.get(&self.marker_slot[0].0) {
            Some((Camp::Plague, Stuff::Marker(_))) => {
                if c != self.marker_slot[0].0 && !self.doctor_is_free(g, self.marker_slot[0].0) {
                    return Err("Ex24");
                }
            }
            _ => {}
        }

        self.update_marker_slot(g, c);

        if self.is_done(g, DOCTOR_MARKER) {
            // This action is done
            return Ok("Ix02");
        }

        return Ok("Ix01");
    }

    fn is_done(&self, g: &Game, q: i32) -> bool {
        if self.marker_slot.iter().map(|(_, n)| n).sum::<u8>() == 0 {
            return true;
        }
        if self.markers.len() as i32 - q >= 0 {
            return true;
        }
        return false;
    }

    fn update_marker_slot(&mut self, g: &Game, c: Coord) {
        let mut temp_marker_slot = self.marker_slot.clone();
        temp_marker_slot.retain(|&(mc, n)| mc != c || (mc == c && n > 1));
        // for get_marker_slot
        for (c, n) in temp_marker_slot.iter_mut() {
            *n = g.get_marker_capacity(*c, Some(&self));
        }
        temp_marker_slot.retain(|&(_, n)| n != 0);
        temp_marker_slot.sort_by(|a, b| {
            let a_sick = self.doctor_is_free(g, a.0);
            let b_sick = self.doctor_is_free(g, b.0);
            b_sick.cmp(&a_sick)
        });

        self.marker_slot = temp_marker_slot;
    }

    pub fn add_single_marker(&mut self, g: &Game, c: Coord) -> Result<&'static str, &'static str> {
        let mut is_qualified = false;
        for m in self.marker_slot.iter() {
            if m.0 == c {
                is_qualified = true;
            }
        }
        if !is_qualified {
            return Err("Ex22");
        }

        let mut res = Ok("Ix01");
        // Update action
        self.markers.push(c);
        if g.turn == Camp::Doctor {
            return self.add_single_marker_doctor(g, c);
        } else {
            res = self.add_single_marker_plague(g, c);
        };

        if res == Ok("Ix02") {
            self.transit(g);
            return res;
        }

        let quota = if g.turn == Camp::Doctor {
            DOCTOR_MARKER
        } else {
            PLAGUE_MARKER
        };

        let op = g.opposite(g.turn);
        if let Some(oph) = g.character.get(&(self.world.unwrap(), op)) {
            if *oph == c {
                return Err("Ex08");
            }
        }

        if !self.trajectory.contains(&c) {
            return Err("Ex09");
        }

        if self.markers.len() > quota.try_into().unwrap() {
            // technically, this is not possible for all servers we have now.
            return Err("Ex0A");
        } else if self.markers.len() == quota.try_into().unwrap() {
            // when all markers are given ... most checks are here
            // 1. if there is overflow
            // 2. (Doctor) if any plagues are ignored
            // 3. (Plague) if distributed evenly
            let last = self.trajectory.len() - 1;
            let recover = self.trajectory.remove(last);

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
                            self.markers = Vec::new();
                            self.trajectory.push(recover);
                            return Err("Ex0B");
                        }
                    }
                    if g.turn == Camp::Plague && cur != max && cur != min {
                        self.markers = Vec::new();
                        self.trajectory.push(recover);
                        return Err("Ex0C");
                    }
                    cur = 1;
                } else {
                    cur += 1;
                }
            }

            self.transit(g);
            return Ok("Ix02");
        }
        return res;
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

        if g.turn == Camp::Doctor {
            for i in 0..=self.steps {
                v.push(self.trajectory[i].env_to_sgf());
            }
        } else {
            // trajectory: in reverse order
            let mut i = self.steps;
            while i > 0 {
                v.push(self.trajectory[i - 1].env_to_sgf());
                i = i - 1;
            }
            // character
            let ch = self.character.unwrap();
            v.push(ch.env_to_sgf());
        }

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

    fn find_route(&self, ret: &mut Vec<Vec<Direction>>, ld: bool) {
        if self.restriction.len() == 1 {
            let mut inner: Vec<Direction> = Vec::new();
            let (d, n) = self.restriction.iter().next().unwrap();
            let nn = *n as usize;
            for _ in 0..nn {
                inner.push(*d);
            }
            ret.push(inner);
        } else if self.restriction.len() == 2 {
            let mut iter = self.restriction.iter();
            let mut r1 = iter.next().unwrap();
            let mut r2 = iter.next().unwrap();
            // r1 has less steps along its direction, r2 is another
            let d1 = r1.0;
            let d2 = r2.0;
            let n1 = r1.1;
            let n2 = r2.1;
            if n1 <= n2 {
                r1 = (d1, n1);
                r2 = (d2, n2);
            } else {
                r1 = (d2, n2);
                r2 = (d1, n1);
            };

            // recursively find all permutations in this situation
            // there will be (n1+n2)!/n1!n2! routes
            let mut mask = vec![false; (n1 + n2).try_into().unwrap()];
            fn find_single_route(
                n: i32,
                since: usize,
                ret: &mut Vec<Vec<Direction>>,
                mask: &mut Vec<bool>,
                major: Direction,
                minor: Direction,
            ) {
                if n == 0 {
                    let mut inner: Vec<Direction> = Vec::new();
                    for b in mask.iter() {
                        if *b {
                            inner.push(major);
                        } else {
                            inner.push(minor);
                        }
                    }
                    ret.push(inner);
                } else {
                    for i in (since as usize)..mask.len() {
                        if mask[i] {
                            continue;
                        }
                        mask[i] = true;
                        find_single_route(n - 1, i, ret, mask, major, minor);
                        mask[i] = false;
                    }
                }
            }
            find_single_route(*r1.1, 0, ret, &mut mask, *r1.0, *r2.0);
        } else {
            panic!("On a 2D space, this is just not possible.");
        }
        if ld {
            let base = ret.clone();
            let lda = vec![Lockdown::CC90, Lockdown::CC180, Lockdown::CC270];
            for &m in lda.iter() {
                for b in base.iter() {
                    let inner = b
                        .iter()
                        .map(|&x| x.to_coord().lockdown(m).to_direction())
                        .collect::<Vec<_>>();
                    ret.push(inner);
                }
            }
        }
    }

    // Upon this choice of map position, we can build the candidates
    // of following choices. For this specific route (`r`), two possible
    // trajectories can bee kept.
    pub fn traverse(
        &mut self,
        g: &Game,
        r: &Vec<Direction>,
        ld: Lockdown,
        w: Option<World>,
    ) -> bool {
        let ch = *g.character.get(&(World::Humanity, g.turn)).unwrap();
        let cu = *g.character.get(&(World::Underworld, g.turn)).unwrap();
        let mut ca: Vec<Coord> = Vec::new();
        if w == None {
            ca.push(ch);
            ca.push(cu);
        } else if w.unwrap() == World::Humanity {
            ca.push(ch);
        } else if w.unwrap() == World::Underworld {
            ca.push(cu);
        }

        'next_character: for c in ca.iter() {
            let w = g.env.get(&c).unwrap();
            let mut ctemp = c.clone();
            let mut r_clone = r.clone();
            let mut temp_trajectory: Vec<Coord> = Vec::new();
            temp_trajectory.push(ctemp);
            r_clone.reverse();
            'new_dir: while let Some(d) = r_clone.pop() {
                ctemp = ctemp + &d;
                while ctemp.in_boundary() {
                    match g.env.get(&ctemp) {
                        Some(x) => {
                            if *x == *w {
                                match g.stuff.get(&ctemp) {
                                    Some((camp, Stuff::Colony)) => {
                                        if *camp != g.turn {
                                            // Cannot go through opponent's colony
                                            continue 'next_character;
                                        }
                                    }
                                    _ => {}
                                }
                                temp_trajectory.push(ctemp);
                                continue 'new_dir;
                            } else {
                                ctemp = ctemp + &d;
                                continue;
                            }
                        }
                        None => {
                            // shouldn't be here?
                            break 'new_dir;
                        }
                    }
                }
                // not in the boundary
                continue 'next_character;
            }
            if r_clone.is_empty() {
                match g.character.get(&(*w, g.opposite(g.turn))) {
                    Some(&c) => {
                        if ctemp == c {
                            // Cannot stop at the opponent
                            continue 'next_character;
                        }
                    }
                    _ => {}
                }
                // Yes, we find a real viable route here
                let mut candidate = Candidate::new();
                candidate.character = *c;
                candidate.lockdown = ld;
                candidate.trajectory = temp_trajectory;
                self.candidate.push(candidate);
            }
        }
        // For characters in the both worlds, this `r` just doesn't work.
        return false;
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
    fn test_lockdown_and_character() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen];C[Setup0]AW[fa][ef][ed][eb][cf][cc][dc][ca][ad][fe][ab][db][bb][be][fd][ae][ac][df];C[Setup0]AB[af][ba][dd][da][ff][bf][ee][bc][de][ec][cb][aa][ea][bd][ce][fc][cd][fb];C[Setup1]AB[ee];C[Setup1]AB[cf];C[Setup1]AB[fa];C[Setup1]AB[bc];C[Setup2]AW[bf];C[Setup2]AB[ce];C[Setup2]AW[db];C[Setup2]AB[eb];C[Setup3]AW[ih]
        ;B[hj][eb][db][dc][df][eb][eb][dc][dc]
        )"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        let mut a = Action::new();
        let c1 = Coord::new(0, 0);
        assert!(a.add_map_step(&g, c1).is_ok());
        assert_eq!(*a.restriction.get(&Direction::Up).unwrap(), 1);
        assert_eq!(*a.restriction.get(&Direction::Right).unwrap(), 1);
        assert!(a.add_lockdown_by_coord(&g, Coord::new(-1, 1)).is_ok());
        assert_eq!(*a.restriction.get(&Direction::Up).unwrap(), 1);
        assert_eq!(*a.restriction.get(&Direction::Right).unwrap(), 1);
        let c2 = Coord::new(2, 5);
        if let Err(e) = a.add_character(&g, c2) {
            assert_eq!(e, "Ex21");
            // This will be shadowed by action::candidate.
            // assert_eq!(e, "Ex02");
        }
        let c3 = Coord::new(1, 5);
        let r1 = a.add_character(&g, c3);
        assert!(r1.is_ok());
        assert_eq!(a.trajectory.len(), 1);
    }

    #[test]
    fn test_integrate1() {
        let s0 = "(
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd]
            ;C[Setup1]AB[ab][cd][ef][da]
            ;C[Setup2]AW[df]
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

        let mut a = Action::new();
        assert_eq!(a.action_phase, ActionPhase::SetMap);
        let r1 = a.add_map_step(&g, Coord::new(0, -1));
        assert!(r1.is_ok());
        assert_eq!(a.action_phase, ActionPhase::SetCharacter);
        let r2 = a.add_character(&g, Coord::new(3, 5));
        assert_eq!(r2, Ok("Ix01"));
        assert_eq!(a.action_phase, ActionPhase::BoardMove);
        if let Err(e) = a.add_board_single_step(&g, Coord::new(0, 5)) {
            assert_eq!(e, "Ex23");
        }
        if let Err(e) = a.add_board_single_step(&g, Coord::new(2, 5)) {
            assert_eq!(e, "Ex23");
        }
        let r4 = a.add_board_single_step(&g, Coord::new(1, 5));
        assert!(r4.is_ok());
    }

    #[test]
    fn test_find_route() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen];C[Setup0]AW[fa][ef][ed][eb][cf][cc][dc][ca][ad][fe][ab][db][bb][be][fd][ae][ac][df];C[Setup0]AB[af][ba][dd][da][ff][bf][ee][bc][de][ec][cb][aa][ea][bd][ce][fc][cd][fb];C[Setup1]AB[ee];C[Setup1]AB[cf];C[Setup1]AB[fa];C[Setup1]AB[bc];C[Setup2]AW[bf];C[Setup2]AB[ce];C[Setup2]AW[db];C[Setup2]AB[eb];C[Setup3]AW[ih])"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        let c1 = Coord::new(0, 2); /* AB[ik] */
        let mut a = Action::new();
        // effectively a.add_map_step(&g, c1), but less because we are testing find_route
        a.restriction = c1 - g.map.get(&g.opposite(g.turn)).unwrap();
        let mut ret: Vec<Vec<Direction>> = Vec::new();
        a.find_route(&mut ret, false);
        assert_eq!(ret.len(), 1);
    }

    #[test]
    fn test_find_route2() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen];C[Setup0]AW[fa][ef][ed][eb][cf][cc][dc][ca][ad][fe][ab][db][bb][be][fd][ae][ac][df];C[Setup0]AB[af][ba][dd][da][ff][bf][ee][bc][de][ec][cb][aa][ea][bd][ce][fc][cd][fb];C[Setup1]AB[ee];C[Setup1]AB[cf];C[Setup1]AB[fa];C[Setup1]AB[bc];C[Setup2]AW[bf];C[Setup2]AB[ce];C[Setup2]AW[db];C[Setup2]AB[eb];C[Setup3]AW[ih])"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        let c1 = Coord::new(-1, 2); /* AB[hk] */
        let mut a = Action::new();
        a.restriction = c1 - g.map.get(&g.opposite(g.turn)).unwrap();
        let mut ret: Vec<Vec<Direction>> = Vec::new();
        a.find_route(&mut ret, false);
        assert_eq!(ret.len(), 4);
    }

    #[test]
    fn test_find_route3() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen];C[Setup0]AW[fa][ef][ed][eb][cf][cc][dc][ca][ad][fe][ab][db][bb][be][fd][ae][ac][df];C[Setup0]AB[af][ba][dd][da][ff][bf][ee][bc][de][ec][cb][aa][ea][bd][ce][fc][cd][fb];C[Setup1]AB[ee];C[Setup1]AB[cf];C[Setup1]AB[fa];C[Setup1]AB[bc];C[Setup2]AW[bf];C[Setup2]AB[ce];C[Setup2]AW[db];C[Setup2]AB[eb];C[Setup3]AW[ih])"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t));
        let c1 = Coord::new(2, 1); /* AB[kj] */
        let mut a = Action::new();
        a.restriction = c1 - g.map.get(&g.opposite(g.turn)).unwrap();
        let mut ret: Vec<Vec<Direction>> = Vec::new();
        a.find_route(&mut ret, false);
        assert_eq!(ret.len(), 6);
    }

    #[test]
    fn test_find_route4() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen];C[Setup0]AW[fa][ef][ed][eb][cf][cc][dc][ca][ad][fe][ab][db][bb][be][fb][ae][ac][df];C[Setup0]AB[af][ba][dd][da][ff][bf][ee][bc][de][ec][cb][aa][ea][bd][ce][fc][cd][fd];C[Setup1]AB[ee];C[Setup1]AB[cf];C[Setup1]AB[fa];C[Setup1]AB[bc];C[Setup2]AW[bf];C[Setup2]AB[ce];C[Setup2]AW[db];C[Setup2]AB[eb];C[Setup3]AW[ih])"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));
        let mut buffer = String::new();
        g.history.borrow().to_root().borrow().to_string(&mut buffer);
        assert_eq!(s0, buffer);
        let _s1 = "(;B[jg][ce][de][dd][ce][ce][de][de])";
        let _s2 = "(;B[jg][ce][cd][dd][cd][cd][de][de])";
        let _s3 = "(;B[jg][eb][fb][fa][eb][eb][fb][fb])";
        let mut a = Action::new();
        // Two roadblocks to shutdown the possiblity
        g.stuff.insert("fb".to_env(), (Camp::Doctor, Stuff::Colony));
        g.character
            .insert((World::Underworld, Camp::Doctor), "dd".to_env());
        let r1 = a.add_map_step(&g, "jg".to_map());
        assert_eq!(Err("Ex20"), r1);
    }

    #[test]
    fn test_fail_to_lockdown() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen];C[Setup0]AW[fa][ef][ed][eb][cf][cc][dc][ca][ad][fe][ab][db][bb][be][fb][ae][ac][df];C[Setup0]AB[af][ba][dd][da][ff][bf][ee][bc][de][ec][cb][aa][ea][bd][ce][fc][cd][fd];C[Setup1]AB[ee];C[Setup1]AB[cf];C[Setup1]AB[fa];C[Setup1]AB[bc];C[Setup2]AW[bf];C[Setup2]AB[ce];C[Setup2]AW[db];C[Setup2]AB[eb];C[Setup3]AW[ih];B[jg][ce][de][dd][ce][ce][de][de])"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));
        // On choosing [hk] to lockdown, [eb] is not an option.
        // from [bf], two intermediates are [bd] here and [ff]. Lock them.
        let _s1 = "(;W[ii][hk][bf][bd][bc][ec][bc][bf][bf][bf][bd])";
        let mut a = Action::new();

        // Comment this out to keep the r2 assertion, otherwise it will
        // be shadowed by the implementation of action::candidate.
        // g.stuff.insert("bd".to_env(), (Camp::Plague, Stuff::Colony));
        g.stuff.insert("ff".to_env(), (Camp::Plague, Stuff::Colony));
        let r1 = a.add_map_step(&g, "ii".to_map());
        assert_eq!(Ok("Ix01"), r1);
        let r2 = a.add_lockdown_by_coord(&g, "hk".to_map());
        assert_eq!(Ok("Ix01"), r2);
        let r3 = a.add_character(&g, "bf".to_env());
        assert_eq!(Ok("Ix01"), r3);
        // Since we have comment the [bd] colony out, this should be
        // fixed accordingly.
        // assert_eq!(Err("Ex21"), r3);
    }

    #[test]
    fn test_doctor_marker() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen];C[Setup0]AW[fa][ef][ed][eb][cf][cc][dc][ca][ad][fe][ab][db][bb][be][fb][ae][ac][df];C[Setup0]AB[af][ba][dd][da][ff][bf][ee][bc][de][ec][cb][aa][ea][bd][ce][fc][cd][fd];C[Setup1]AB[ee];C[Setup1]AB[cf];C[Setup1]AB[fa];C[Setup1]AB[bc];C[Setup2]AW[bf];C[Setup2]AB[ce];C[Setup2]AW[db];C[Setup2]AB[eb];C[Setup3]AW[ih];B[jg][ce][de][dd][ce][ce][de][de])"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));
        let _s1 = "(;W[ii][hk][bf][bd][bc][ec][bc][bf][bf][bf][bd])";
        let mut a = Action::new();
        // check test_fail_to_lockdown
        let r1 = a.add_map_step(&g, "ii".to_map());
        assert_eq!(Ok("Ix01"), r1);
        let r2 = a.add_lockdown_by_coord(&g, "hk".to_map());
        assert_eq!(Ok("Ix01"), r2);
        let r3 = a.add_character(&g, "bf".to_env());
        assert_eq!(Ok("Ix01"), r3);
        let r4 = a.add_board_single_step(&g, "bd".to_env());
        assert_eq!(Ok("Ix01"), r4);
        let r5 = a.add_board_single_step(&g, "bc".to_env());
        assert_eq!(Ok("Ix01"), r5);
        g.stuff.insert("aa".to_env(), (Camp::Doctor, Stuff::Colony));
        g.stuff.insert("ad".to_env(), (Camp::Doctor, Stuff::Colony));
        g.stuff
            .insert("bf".to_env(), (Camp::Doctor, Stuff::Marker(5)));
        g.stuff
            .insert("bd".to_env(), (Camp::Doctor, Stuff::Marker(5)));
        g.stuff
            .insert("bc".to_env(), (Camp::Doctor, Stuff::Marker(5)));
        let r6 = a.add_board_single_step(&g, "ec".to_env());
        assert_eq!(Ok("Ix02"), r6);
        assert_eq!(0, a.marker_slot.len());
    }

    #[test]
    fn test_doctor_marker2() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen];C[Setup0]AW[fa][ef][ed][eb][cf][cc][dc][ca][ad][fe][ab][db][bb][be][fb][ae][ac][df];C[Setup0]AB[af][ba][dd][da][ff][bf][ee][bc][de][ec][cb][aa][ea][bd][ce][fc][cd][fd];C[Setup1]AB[ee];C[Setup1]AB[cf];C[Setup1]AB[fa];C[Setup1]AB[bc];C[Setup2]AW[bf];C[Setup2]AB[ce];C[Setup2]AW[db];C[Setup2]AB[eb];C[Setup3]AW[ih];B[jg][ce][de][dd][ce][ce][de][de])"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));
        let _s1 = "(;W[ii][hk][bf][bd][bc][ec][bc][bf][bf][bf][bd])";
        let mut a = Action::new();
        let _ = a.add_map_step(&g, "ii".to_map());
        let _ = a.add_lockdown_by_coord(&g, "hk".to_map());
        let _ = a.add_character(&g, "bf".to_env());
        let _ = a.add_board_single_step(&g, "bd".to_env());
        let _ = a.add_board_single_step(&g, "bc".to_env());
        g.stuff.insert("aa".to_env(), (Camp::Doctor, Stuff::Colony));
        g.stuff
            .insert("bf".to_env(), (Camp::Doctor, Stuff::Marker(5)));
        g.stuff
            .insert("bd".to_env(), (Camp::Doctor, Stuff::Marker(5)));
        g.stuff
            .insert("bc".to_env(), (Camp::Doctor, Stuff::Marker(5)));
        let r6 = a.add_board_single_step(&g, "ec".to_env());
        assert_eq!(Ok("Ix01"), r6);
        assert_eq!(2, a.marker_slot.len());
        let r7 = a.add_single_marker(&g, "bf".to_env());
        assert_eq!(Ok("Ix02"), r7);
        assert_eq!(g.near_but_not_colony("bd".to_env(), None), false);
        assert_eq!(g.near_but_not_colony("bd".to_env(), Some(&a)), true);
    }

    #[test]
    fn test_doctor_marker3() {
        let s0 = "(;FF[4]GM[41]SZ[6]GN[https://boardgamegeek.com/boardgame/369862/pathogen];C[Setup0]AW[fa][ef][ed][eb][cf][cc][dc][ca][ad][fe][ab][db][bb][be][fb][ae][ac][df];C[Setup0]AB[af][ba][dd][da][ff][bf][ee][bc][de][ec][cb][aa][ea][bd][ce][fc][cd][fd];C[Setup1]AB[ee];C[Setup1]AB[cf];C[Setup1]AB[fa];C[Setup1]AB[bc];C[Setup2]AW[bf];C[Setup2]AB[ce];C[Setup2]AW[db];C[Setup2]AB[eb];C[Setup3]AW[ih];B[jg][ce][de][dd][ce][ce][de][de])"
        .to_string();
        let mut iter = s0.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let mut g = Game::init(Some(t));
        let _s1 = "(;W[ii][hk][bf][bd][bc][ec][bc][bf][bf][bf][bd])";
        let mut a = Action::new();
        let _ = a.add_map_step(&g, "ii".to_map());
        let _ = a.add_lockdown_by_coord(&g, "hk".to_map());
        let _ = a.add_character(&g, "bf".to_env());
        let _ = a.add_board_single_step(&g, "bd".to_env());
        let _ = a.add_board_single_step(&g, "bc".to_env());
        g.stuff
            .insert("bf".to_env(), (Camp::Plague, Stuff::Marker(5)));
        let r6 = a.add_board_single_step(&g, "ec".to_env());
        assert_eq!(Ok("Ix01"), r6);
        assert_eq!(3, a.marker_slot.len());
        let r7 = a.add_single_marker(&g, "bc".to_env());
        assert_eq!(Err("Ex24"), r7);
        let r8 = a.add_single_marker(&g, "bf".to_env());
        assert_eq!(Ok("Ix01"), r8);
        let _ = a.add_single_marker(&g, "bf".to_env());
        let _ = a.add_single_marker(&g, "bf".to_env());
        let _ = a.add_single_marker(&g, "bf".to_env());
        let r9 = a.add_single_marker(&g, "bf".to_env());
        assert_eq!(Ok("Ix02"), r9);
    }
}
