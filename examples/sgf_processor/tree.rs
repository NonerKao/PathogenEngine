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
    divergent: bool,
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

    pub fn checkpoint(&self, key: String) -> bool {
        for p in self.properties.iter() {
            if p.ident == "C".to_string() && p.value[0] == key {
                return true;
            }
        }
        return false;
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

    pub fn toString(&self, buf: &mut String) {
        self.traverse(&print_node, buf);
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
}
