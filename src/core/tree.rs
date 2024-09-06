use super::action::*;
use super::*;
use std::cell::RefCell;
use std::iter::Peekable;
use std::rc::Rc;
use std::str::Chars;

#[derive(Debug, Default, PartialEq)]
pub struct Property {
    pub ident: String,
    pub value: Vec<String>,
}

impl Property {
    pub fn new(iter: &mut Peekable<Chars<'_>>) -> Property {
        let mut p = Property {
            ident: String::new(),
            value: Vec::new(),
        };
        while let Some(c) = iter.peek() {
            match c {
                '[' => {
                    break;
                }
                _ => {
                    p.ident.push(iter.next().unwrap());
                }
            }
        }
        let mut temp = String::new();
        while let Some(c) = iter.peek() {
            match c {
                ')' | ';' => {
                    return p;
                }
                '[' => {
                    iter.next();
                }
                ']' => {
                    iter.next();
                    eliminate_whitespace(iter);
                    p.value.push(temp.clone());
                    let c2 = iter.peek();
                    if c2 == Some(&'[') {
                        temp = String::new();
                    } else {
                        return p;
                    }
                }
                _ => {
                    temp.push(iter.next().unwrap());
                }
            }
        }
        return p;
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct TreeNode {
    pub divergent: bool,
    pub savepoint: bool,
    pub properties: Vec<Property>,
    pub children: Vec<Rc<RefCell<TreeNode>>>,
    pub parent: Option<Rc<RefCell<TreeNode>>>,
}

fn eliminate_whitespace(iter: &mut Peekable<Chars<'_>>) {
    loop {
        let c2 = iter.peek();
        match c2 {
            Some(x) if x.is_ascii_whitespace() => {
                iter.next();
            }
            _ => {
                return;
            }
        }
    }
}

impl TreeNode {
    pub fn new(
        iter: &mut Peekable<Chars<'_>>,
        p: Option<Rc<RefCell<TreeNode>>>,
    ) -> Rc<RefCell<TreeNode>> {
        let root = Rc::new(RefCell::new(TreeNode {
            properties: Vec::new(),
            divergent: false,
            savepoint: false,
            children: Vec::new(),
            parent: p,
        }));
        if iter.peek() == None {
            return root;
        }
        loop {
            let c = iter.peek();
            match c {
                Some('(') => {
                    iter.next();
                    eliminate_whitespace(iter);
                    if iter.peek() != Some(&';') {
                        panic!("';' is mandated by the the spec for a node");
                    }
                    iter.next();
                    eliminate_whitespace(iter);
                    root.borrow_mut().divergent = true;
                    let new_node = Self::new(iter, Some(root.clone()));
                    root.borrow_mut().children.push(new_node.clone());
                }
                Some(')') => {
                    iter.next();
                    eliminate_whitespace(iter);
                    return root;
                }
                Some(';') => {
                    iter.next();
                    eliminate_whitespace(iter);
                    let new_node = Self::new(iter, Some(root.clone()));
                    root.borrow_mut().children.push(new_node.clone());
                    return root;
                }
                Some(_) => {
                    root.borrow_mut().properties.push(Property::new(iter));
                }
                None => {
                    return root;
                }
            }
        }
    }

    pub fn traverse<F, BT>(&self, f: &F, state: &mut BT)
    where
        F: Fn(&TreeNode, bool, &mut BT),
    {
        // deal with the contents in this node
        f(self, true, state);

        for c in self.children.iter() {
            // for each node, do f...
            c.borrow().traverse(f, state);
        }

        f(self, false, state);
    }

    pub fn checkpoint(&self) -> &str {
        for p in self.properties.iter() {
            if p.ident == "C".to_string() {
                return p.value[0].as_str();
            }
        }
        return "";
    }

    pub fn get_value(&self, key: String, v: &mut String) {
        for p in self.properties.iter() {
            if p.ident == key {
                for s in p.value.iter() {
                    *v = s.to_string().clone();
                    return;
                }
            }
        }
    }

    pub fn get_general(&self, key: String, v: &mut Vec<String>) {
        for p in self.properties.iter() {
            if p.ident == key {
                for s in p.value.iter() {
                    v.push(s.to_string());
                }
            }
        }
    }

    pub fn to_string(&self, buf: &mut String) {
        self.traverse(&print_node, buf);
    }

    pub fn to_action(&self, g: &Game) -> Result<Action, &'static str> {
        fn nil_func(_g: &Game, _a: &Action, _vec: &mut Vec<f32>, _coord: Coord) {}
        let mut vec = Vec::new();
        self.to_action_do_func(g, nil_func, &mut vec)
    }

    // Check action.rs add_* calls
    pub fn to_action_do_func<F>(
        &self,
        g: &Game,
        func: F,
        vec: &mut Vec<f32>,
    ) -> Result<Action, &'static str>
    where
        F: Fn(&Game, &Action, &mut Vec<f32>, Coord),
    {
        let mut a = Action::new();
        let mut vi = 0;
        let c = self.properties[0].value[vi].as_str().to_map();
        vi = vi + 1;
        func(g, &a, vec, c);
        match a.add_map_step(g, c) {
            Err(e) => {
                return Err(e);
            }
            Ok("Ix00") => {
                return Ok(a);
            }
            _ => {}
        }

        if c == Coord::new(0, 0) && g.turn == Camp::Doctor {
            let c = self.properties[0].value[vi].as_str().to_map();
            func(g, &a, vec, c);
            a.add_lockdown_by_coord(g, c)?;
            vi = vi + 1;
        }

        let c = self.properties[0].value[vi].as_str().to_env();
        func(g, &a, vec, c);
        a.add_character(g, c)?;
        vi = vi + 1;
        for _ in 0..a.steps {
            let c = self.properties[0].value[vi].as_str().to_env();
            func(g, &a, vec, c);
            a.add_board_single_step(g, c)?;
            vi = vi + 1;
        }
        for _ in vi..self.properties[0].value.len() {
            a.add_single_marker(g, self.properties[0].value[vi].as_str().to_env())?;
            vi = vi + 1;
        }
        Ok(a)
    }

    pub fn to_sgf_node(&self) -> Option<Rc<RefCell<TreeNode>>> {
        let mut buffer = String::new();
        if self.properties.len() > 0 {
            buffer.push_str("(;");
            for i in self.properties.iter() {
                buffer.push_str(&i.ident);
                for v in i.value.iter() {
                    buffer.push('[');
                    buffer.push_str(v);
                    buffer.push(']');
                }
            }
            buffer.push(')');
        } else {
            return None;
        }
        let mut iter = buffer.trim().chars().peekable();
        Some(TreeNode::new(&mut iter, None))
    }

    pub fn to_root(&self) -> Rc<RefCell<TreeNode>> {
        match &self.parent {
            None => {
                panic!("self cannot be root!");
            }
            Some(p) => match p.borrow().parent {
                None => {
                    assert_eq!(p.borrow().properties.len(), 0);
                    return p.clone();
                }
                Some(_) => {
                    return p.borrow().to_root().clone();
                }
            },
        }
    }
}

pub fn print_node(t: &TreeNode, is_front: bool, buffer: &mut String) {
    if is_front {
        match &t.parent {
            Some(x) => {
                if x.as_ref().borrow().divergent {
                    buffer.push('(');
                }
            }
            None => {}
        }
        if t.properties.len() > 0 {
            buffer.push(';');
        }
        for i in t.properties.iter() {
            buffer.push_str(&i.ident);
            for v in i.value.iter() {
                buffer.push('[');
                buffer.push_str(v);
                buffer.push(']');
            }
        }
    } else {
        match &t.parent {
            Some(x) => {
                if x.as_ref().borrow().divergent {
                    buffer.push(')');
                }
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let s = String::from("(;FF[4];rt[123])");
        let mut iter = s.trim().chars().peekable();
        let tree = TreeNode::new(&mut iter, None);
        let mut buffer = String::new();
        tree.borrow().traverse(&print_node, &mut buffer);
        assert_eq!(buffer, s);
    }

    #[test]
    fn test_whitespace() {
        let s = String::from(
            "(;FF[4]
        [try][this]
    G[ll]  ;FF[3])",
        );
        let mut iter = s.trim().chars().peekable();
        let tree = TreeNode::new(&mut iter, None);
        let mut buffer = String::new();
        tree.borrow().traverse(&print_node, &mut buffer);
        assert_eq!(
            buffer,
            s.chars().filter(|c| !c.is_whitespace()).collect::<String>()
        );
    }

    #[test]
    fn test_new3() {
        let s = String::from("(;FF[4][try][this]G[ll];FF[3](;B[12];W[34])(;B[13];W[24]))");
        let mut iter = s.trim().chars().peekable();
        let tree = TreeNode::new(&mut iter, None);
        let mut buffer = String::new();
        tree.borrow().traverse(&print_node, &mut buffer);
        assert_eq!(buffer, s);
    }

    #[test]
    fn test_new4() {
        let s = String::from("(;FF[4])");
        let mut iter = s.trim().chars().peekable();
        let tree = TreeNode::new(&mut iter, None);
        assert_eq!(tree.borrow().children.len(), 1);
    }

    #[test]
    fn test_new5() {
        let s = String::from("(;FF[4][5][6])");
        let mut iter = s.trim().chars().peekable();
        let tree = TreeNode::new(&mut iter, None);
        let tree2 = tree.borrow().children[0].clone();
        assert_eq!(tree.borrow().properties.len(), 0);
        assert_eq!(tree2.borrow().properties.len(), 1);
        assert_eq!(tree2.borrow().properties[0].value.len(), 3);
    }

    #[test]
    fn test_new6() {
        let s = String::from("(;FF[4][5][6];EE[1][2])");
        let mut iter = s.trim().chars().peekable();
        let tree = TreeNode::new(&mut iter, None);
        assert_eq!(tree.borrow().properties.len(), 0);
        assert_eq!(
            tree.borrow().children[0].borrow().properties[0].value.len(),
            3
        );
        assert_eq!(
            tree.borrow().children[0].borrow().children[0]
                .borrow()
                .properties[0]
                .value
                .len(),
            2
        );
    }

    #[test]
    fn test_sgf_node() {
        let s = String::from("(;FF[4][5][6];EE[1][2];DD[0][uuu];CCC[A])");
        let mut iter = s.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let n = t.borrow().children[0].borrow().children[0]
            .borrow()
            .children[0]
            .borrow()
            .to_sgf_node()
            .unwrap();
        let mut buffer = String::new();
        n.borrow().traverse(&print_node, &mut buffer);
        let ns = String::from("(;DD[0][uuu])");
        assert_eq!(buffer, ns);
    }

    #[test]
    fn test_to_root() {
        let s = String::from("(;FF[4][5][6];EE[1][2];DD[0][uuu];CCC[A])");
        let mut iter = s.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let n = t.borrow().children[0].borrow().children[0]
            .borrow()
            .children[0]
            .borrow()
            .children[0]
            .clone();
        let root = n.borrow().to_root();
        let mut buffer = String::new();
        root.borrow().traverse(&print_node, &mut buffer);
        assert_eq!(buffer, s);
    }

    #[test]
    fn test_to_root_add_history_check() {
        let s = String::from(
            "(;FF[4][5][6];EE[1][2];DD[0][uuu];CCC[A]
            ;C[Setup0]
            AW[aa][ab][ad][ae][bb][bc][bf][ca][cd][ce][dc][dd][df][ea][ec][ee][fa][fb][fe][ff]
            AB[ac][af][ba][bd][be][cb][cc][cf][da][db][de][eb][ed][ef][fc][fd])",
        );
        let mut iter = s.trim().chars().peekable();
        let t = TreeNode::new(&mut iter, None);
        let g = Game::init(Some(t.clone()));
        let root = g.history.borrow().to_root();
        let mut buffer = String::new();
        t.borrow().traverse(&print_node, &mut buffer);
        assert_eq!(t.borrow().divergent, root.borrow().divergent);
        let mut buffer2 = String::new();
        root.borrow().traverse(&print_node, &mut buffer2);
        assert_eq!(buffer, buffer2);
    }
}
