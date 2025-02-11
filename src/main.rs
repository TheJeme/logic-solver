use clap::Parser;
use std::collections::{HashMap, HashSet};

//
// TOKENIZATION SECTION
//

#[derive(Debug, Clone)]
enum Token {
    ComplexVariable(String),
    ConstTrue,
    ConstFalse,
    LParen,
    RParen,
    Not,
    And,
    Or,
    Variable(String),
}

/// Tokenizes the input string and returns a list of tokens.
/// Returns an error if an unexpected token is encountered.
fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut s = input;
    while !s.is_empty() {
        // Skip leading whitespace
        s = s.trim_start();
        if s.is_empty() {
            break;
        }
        // COMPLEX_VARIABLE: starts with a double quote
        if s.starts_with("\"") {
            if let Some(end_quote_pos) = s[1..].find("\"") {
                let content = &s[1..1 + end_quote_pos];
                tokens.push(Token::ComplexVariable(content.to_string()));
                s = &s[1 + end_quote_pos + 1..];
                continue;
            } else {
                return Err("Unterminated quoted string".to_string());
            }
        }
        // CONST_TRUE: "T" (ensure next character is not part of the identifier)
        if s.starts_with("T") {
            if s.len() == 1 || {
                let c = s.chars().nth(1).unwrap();
                !c.is_alphanumeric() && c != '_'
            } {
                tokens.push(Token::ConstTrue);
                s = &s[1..];
                continue;
            }
        }
        // CONST_FALSE: "F"
        if s.starts_with("F") {
            if s.len() == 1 || {
                let c = s.chars().nth(1).unwrap();
                !c.is_alphanumeric() && c != '_'
            } {
                tokens.push(Token::ConstFalse);
                s = &s[1..];
                continue;
            }
        }
        // NOT operator: "not" (case-insensitive), "!" or "¬"
        if s.len() >= 3 && s[..3].eq_ignore_ascii_case("not") {
            if s.len() == 3 || {
                let c = s.chars().nth(3).unwrap();
                !c.is_alphanumeric() && c != '_'
            } {
                tokens.push(Token::Not);
                s = &s[3..];
                continue;
            }
        }
        if s.starts_with("!") {
            tokens.push(Token::Not);
            s = &s[1..];
            continue;
        }
        if s.starts_with("¬") {
            tokens.push(Token::Not);
            s = &s['¬'.len_utf8()..];
            continue;
        }
        // AND operator: "and" (case-insensitive), "&&" or "∧"
        if s.len() >= 3 && s[..3].eq_ignore_ascii_case("and") {
            if s.len() == 3 || {
                let c = s.chars().nth(3).unwrap();
                !c.is_alphanumeric() && c != '_'
            } {
                tokens.push(Token::And);
                s = &s[3..];
                continue;
            }
        }
        if s.starts_with("&&") {
            tokens.push(Token::And);
            s = &s[2..];
            continue;
        }
        if s.starts_with("∧") {
            tokens.push(Token::And);
            s = &s['∧'.len_utf8()..];
            continue;
        }
        // OR operator: "or" (case-insensitive), "||" or "∨"
        if s.len() >= 2 && s[..2].eq_ignore_ascii_case("or") {
            if s.len() == 2 || {
                let c = s.chars().nth(2).unwrap();
                !c.is_alphanumeric() && c != '_'
            } {
                tokens.push(Token::Or);
                s = &s[2..];
                continue;
            }
        }
        if s.starts_with("||") {
            tokens.push(Token::Or);
            s = &s[2..];
            continue;
        }
        if s.starts_with("∨") {
            tokens.push(Token::Or);
            s = &s['∨'.len_utf8()..];
            continue;
        }
        // Parentheses
        if s.starts_with("(") {
            tokens.push(Token::LParen);
            s = &s[1..];
            continue;
        }
        if s.starts_with(")") {
            tokens.push(Token::RParen);
            s = &s[1..];
            continue;
        }
        // VARIABLE: starts with a letter or underscore and continues with alphanumerics/underscores
        if let Some(ch) = s.chars().next() {
            if ch.is_alphabetic() || ch == '_' {
                let mut len = 0;
                for c in s.chars() {
                    if c.is_alphanumeric() || c == '_' {
                        len += c.len_utf8();
                    } else {
                        break;
                    }
                }
                let var_name = &s[..len];
                tokens.push(Token::Variable(var_name.to_string()));
                s = &s[len..];
                continue;
            }
        }
        return Err(format!("Unexpected token starting at: {}", s));
    }
    Ok(tokens)
}

//
// PARSER SECTION (Recursive, renamed to ExprParser)
//

#[derive(Debug, Clone)]
enum AST {
    Variable(String),
    Const(bool),
    Not(Box<AST>),
    And(Box<AST>, Box<AST>),
    Or(Box<AST>, Box<AST>),
}

struct ExprParser {
    tokens: Vec<Token>,
    pos: usize,
}

impl ExprParser {
    fn new(tokens: Vec<Token>) -> Self {
        ExprParser { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn consume_token(&mut self) -> Option<Token> {
        let token = self.peek()?.clone();
        self.pos += 1;
        Some(token)
    }

    // Expression -> OrExp
    fn parse_expression(&mut self) -> Result<AST, String> {
        self.parse_or()
    }

    // OrExp -> AndExp ( OR AndExp )*
    fn parse_or(&mut self) -> Result<AST, String> {
        let mut node = self.parse_and()?;
        while let Some(Token::Or) = self.peek() {
            self.consume_token(); // consume OR
            let right = self.parse_and()?;
            node = AST::Or(Box::new(node), Box::new(right));
        }
        Ok(node)
    }

    // AndExp -> NotExp ( AND NotExp )*
    fn parse_and(&mut self) -> Result<AST, String> {
        let mut node = self.parse_not()?;
        while let Some(Token::And) = self.peek() {
            self.consume_token(); // consume AND
            let right = self.parse_not()?;
            node = AST::And(Box::new(node), Box::new(right));
        }
        Ok(node)
    }

    // NotExp -> (NOT)* Primary
    fn parse_not(&mut self) -> Result<AST, String> {
        if let Some(Token::Not) = self.peek() {
            self.consume_token(); // consume NOT
            let operand = self.parse_not()?;
            return Ok(AST::Not(Box::new(operand)));
        }
        self.parse_primary()
    }

    // Primary -> ( Expression ) | VARIABLE | COMPLEX_VARIABLE | CONST_TRUE | CONST_FALSE
    fn parse_primary(&mut self) -> Result<AST, String> {
        match self.peek() {
            Some(Token::LParen) => {
                self.consume_token(); // consume '('
                let node = self.parse_expression()?;
                match self.peek() {
                    Some(Token::RParen) => {
                        self.consume_token(); // consume ')'
                        Ok(node)
                    }
                    other => Err(format!(
                        "Expected closing parenthesis, but got: {:?}",
                        other
                    )),
                }
            }
            Some(Token::Variable(_)) => {
                let token = self.consume_token().unwrap();
                if let Token::Variable(n) = token {
                    Ok(AST::Variable(n))
                } else {
                    unreachable!()
                }
            }
            Some(Token::ComplexVariable(_)) => {
                let token = self.consume_token().unwrap();
                if let Token::ComplexVariable(n) = token {
                    // Convert complex variable to a simple variable
                    Ok(AST::Variable(n))
                } else {
                    unreachable!()
                }
            }
            Some(Token::ConstTrue) => {
                self.consume_token();
                Ok(AST::Const(true))
            }
            Some(Token::ConstFalse) => {
                self.consume_token();
                Ok(AST::Const(false))
            }
            other => Err(format!("Unexpected token: {:?}", other)),
        }
    }
}

//
// AST Evaluation and Variable Extraction
//

/// Evaluates the AST using the given environment (mapping variable names to booleans).
fn eval_ast(ast: &AST, env: &HashMap<String, bool>) -> Result<bool, String> {
    match ast {
        AST::Variable(name) => env
            .get(name)
            .cloned()
            .ok_or_else(|| format!("Variable {} is not defined", name)),
        AST::Const(b) => Ok(*b),
        AST::Not(expr) => Ok(!eval_ast(expr, env)?),
        AST::And(left, right) => Ok(eval_ast(left, env)? && eval_ast(right, env)?),
        AST::Or(left, right) => Ok(eval_ast(left, env)? || eval_ast(right, env)?),
    }
}

/// Extracts all variables from the AST (each only once) in the order of appearance.
fn get_variables(ast: &AST) -> Vec<String> {
    let mut vars = Vec::new();
    fn traverse(ast: &AST, vars: &mut Vec<String>) {
        match ast {
            AST::Variable(name) => {
                if !vars.contains(name) {
                    vars.push(name.clone());
                }
            }
            AST::Const(_) => {}
            AST::Not(expr) => traverse(expr, vars),
            AST::And(left, right) | AST::Or(left, right) => {
                traverse(left, vars);
                traverse(right, vars);
            }
        }
    }
    traverse(ast, &mut vars);
    vars
}

//
// QUINE–MCCLUSKEY ALGORITHM
//

/// Converts a number to a binary string with the specified width.
fn to_binary_string(num: usize, width: usize) -> String {
    format!("{:0width$b}", num, width = width)
}

/// Attempts to combine two terms; if they differ by exactly one bit, returns the combined term.
fn combine_terms(t1: &str, t2: &str) -> Option<String> {
    let mut diff_count = 0;
    let mut combined = String::new();
    for (c1, c2) in t1.chars().zip(t2.chars()) {
        if c1 == c2 {
            combined.push(c1);
        } else if c1 != '-' && c2 != '-' {
            diff_count += 1;
            combined.push('-');
        } else {
            return None;
        }
    }
    if diff_count == 1 {
        Some(combined)
    } else {
        None
    }
}

/// Struct representing a prime implicant.
#[derive(Debug, Clone)]
struct PrimeImplicant {
    term: String,
    minterms: Vec<usize>,
}

/// Quine–McCluskey algorithm: returns the prime implicants for the given minterm values.
fn quine_mccluskey(minterms: &[usize], num_vars: usize) -> Vec<PrimeImplicant> {
    // Initial terms: each minterm as a binary string
    let mut current_terms: Vec<PrimeImplicant> = minterms
        .iter()
        .map(|&m| PrimeImplicant {
            term: to_binary_string(m, num_vars),
            minterms: vec![m],
        })
        .collect();

    let mut prime_implicants = Vec::new();

    loop {
        // Group current terms by the number of '1's in the term
        let mut groups: HashMap<usize, Vec<PrimeImplicant>> = HashMap::new();
        for term in &current_terms {
            let ones = term.term.chars().filter(|&c| c == '1').count();
            groups
                .entry(ones)
                .or_insert_with(Vec::new)
                .push(term.clone());
        }
        let mut new_terms = Vec::new();
        // Use sorted group keys
        let mut group_keys: Vec<usize> = groups.keys().cloned().collect();
        group_keys.sort();

        // Mark terms that are combined using a separate vector
        let mut used = vec![false; current_terms.len()];
        for i in 0..group_keys.len().saturating_sub(1) {
            let group1 = groups.get(&group_keys[i]).unwrap();
            let group2 = groups.get(&group_keys[i + 1]).unwrap();
            for term1 in group1 {
                for term2 in group2 {
                    if let Some(combined) = combine_terms(&term1.term, &term2.term) {
                        // Mark terms as used
                        for (j, t) in current_terms.iter().enumerate() {
                            if t.term == term1.term && t.minterms == term1.minterms {
                                used[j] = true;
                            }
                            if t.term == term2.term && t.minterms == term2.minterms {
                                used[j] = true;
                            }
                        }
                        // Combine the minterm lists
                        let mut new_minterms = term1.minterms.clone();
                        new_minterms.extend(&term2.minterms);
                        new_minterms.sort();
                        new_minterms.dedup();
                        let new_implicant = PrimeImplicant {
                            term: combined,
                            minterms: new_minterms,
                        };
                        if !new_terms
                            .iter()
                            .any(|x: &PrimeImplicant| x.term == new_implicant.term)
                        {
                            new_terms.push(new_implicant);
                        }
                    }
                }
            }
        }
        // Terms that were not combined become prime implicants
        for (i, term) in current_terms.iter().enumerate() {
            if !used[i]
                && !prime_implicants
                    .iter()
                    .any(|x: &PrimeImplicant| x.term == term.term && x.minterms == term.minterms)
            {
                prime_implicants.push(term.clone());
            }
        }
        if new_terms.is_empty() {
            break;
        }
        current_terms = new_terms;
    }
    prime_implicants
}

/// Returns whether a term covers the given minterm (by comparing its binary representation).
fn term_covers(term: &str, minterm: usize, num_vars: usize) -> bool {
    let m_bin = to_binary_string(minterm, num_vars);
    for (c_term, c_bin) in term.chars().zip(m_bin.chars()) {
        if c_term != '-' && c_term != c_bin {
            return false;
        }
    }
    true
}

/// Selects the essential prime implicants that cover all minterms.
/// A brute-force search is performed to find a minimal cover for any remaining minterms.
fn find_essential_prime_implicants(
    prime_implicants: &[PrimeImplicant],
    minterms: &[usize],
    num_vars: usize,
) -> Vec<PrimeImplicant> {
    let mut chart: HashMap<usize, Vec<usize>> = HashMap::new();
    for &m in minterms {
        let mut indices = Vec::new();
        for (i, imp) in prime_implicants.iter().enumerate() {
            if term_covers(&imp.term, m, num_vars) {
                indices.push(i);
            }
        }
        chart.insert(m, indices);
    }
    let mut essential_indices = HashSet::new();
    for (_m, indices) in &chart {
        if indices.len() == 1 {
            essential_indices.insert(indices[0]);
        }
    }
    let remaining_minterms: Vec<usize> = minterms
        .iter()
        .filter(|&&m| {
            !prime_implicants.iter().enumerate().any(|(i, imp)| {
                essential_indices.contains(&i) && term_covers(&imp.term, m, num_vars)
            })
        })
        .cloned()
        .collect();

    let non_essential: Vec<usize> = (0..prime_implicants.len())
        .filter(|i| !essential_indices.contains(i))
        .collect();

    let total = non_essential.len();
    let mut best_cover: Option<HashSet<usize>> = None;
    for i in 0..(1 << total) {
        let mut cover_set = essential_indices.clone();
        for j in 0..total {
            if (i & (1 << j)) != 0 {
                cover_set.insert(non_essential[j]);
            }
        }
        let covers_all = remaining_minterms.iter().all(|&m| {
            cover_set
                .iter()
                .any(|&idx| term_covers(&prime_implicants[idx].term, m, num_vars))
        });
        if covers_all {
            if best_cover.is_none() || cover_set.len() < best_cover.as_ref().unwrap().len() {
                best_cover = Some(cover_set);
            }
        }
    }
    let mut result = Vec::new();
    if let Some(best) = best_cover {
        for idx in best {
            result.push(prime_implicants[idx].clone());
        }
    }
    result
}

/// Sorts the prime implicants so that the final expression preserves the order of variables.
fn sort_implicants_by_var_order(implicants: Vec<PrimeImplicant>) -> Vec<PrimeImplicant> {
    let mut sorted = implicants;
    sorted.sort_by_key(|imp| {
        imp.term
            .find(|c: char| c == '0' || c == '1')
            .unwrap_or(usize::MAX)
    });
    sorted
}

/// Converts the prime implicants to a string expression using the list of variables.
fn implicants_to_expression(implicants: Vec<PrimeImplicant>, var_list: &[String]) -> String {
    let sorted = sort_implicants_by_var_order(implicants);
    let mut parts = Vec::new();
    for imp in sorted {
        let mut term_parts = Vec::new();
        for (i, ch) in imp.term.chars().enumerate() {
            if i >= var_list.len() {
                break;
            }
            match ch {
                '1' => term_parts.push(var_list[i].clone()),
                '0' => term_parts.push(format!("!{}", var_list[i])),
                '-' => {}
                _ => {}
            }
        }
        if term_parts.is_empty() {
            parts.push("true".to_string());
        } else {
            parts.push(term_parts.join(" && "));
        }
    }
    parts.join(" || ")
}

//
// SIMPLIFICATION AND TRUTH TABLE GENERATION
//

struct SimplificationResult {
    simplified: String,
    truth_table: Vec<(HashMap<String, bool>, bool)>,
    var_list: Vec<String>,
}

/// Simplifies the Boolean expression by building a truth table and performing Quine–McCluskey minimization.
fn simplify_boolean_expression(ast: &AST) -> Result<SimplificationResult, String> {
    let var_list = get_variables(ast);
    let num_vars = var_list.len();
    let total = 1 << num_vars;
    let mut truth_table = Vec::new();
    let mut minterms = Vec::new();
    for i in 0..total {
        let mut env = HashMap::new();
        // The first variable corresponds to the MSB
        for (j, var) in var_list.iter().enumerate() {
            let value = (i & (1 << (num_vars - j - 1))) != 0;
            env.insert(var.clone(), value);
        }
        let result = eval_ast(ast, &env)?;
        truth_table.push((env.clone(), result));
        if result {
            minterms.push(i);
        }
    }
    let simplified = if minterms.is_empty() {
        "F".to_string()
    } else if minterms.len() == total {
        "T".to_string()
    } else {
        let prime_implicants = quine_mccluskey(&minterms, num_vars);
        let essential = find_essential_prime_implicants(&prime_implicants, &minterms, num_vars);
        implicants_to_expression(essential, &var_list)
    };

    Ok(SimplificationResult {
        simplified,
        truth_table,
        var_list,
    })
}

//
// IMPROVED TRUTH TABLE PRINTING
//

/// Prints the truth table with improved spacing by dynamically computing column widths.
fn print_truth_table(var_list: &[String], truth_table: &[(HashMap<String, bool>, bool)]) {
    // Extra padding for each column.
    let padding = 2;
    // Compute widths for each variable column.
    let widths: Vec<usize> = var_list
        .iter()
        .map(|var| var.len().max(1) + padding * 2)
        .collect();
    // Compute width for the "Result" column.
    let result_header = "Result";
    let result_width = result_header.len().max(1) + padding * 2;

    // Print the top border row.
    print!("┌");
    for (i, width) in widths.iter().enumerate() {
        if i == 0 {
            print!("{:─^w$}", "", w = *width);
        } else {
            print!("┬{:─^w$}", "", w = *width);
        }
    }
    print!("┬{:─^w$}┐", "", w = result_width);
    println!();

    // Print the header row.
    for (i, var) in var_list.iter().enumerate() {
        print!("│{:^width$}", var, width = widths[i]);
    }
    print!("│{:^w$}│", result_header, w = result_width);
    println!();

    // Print the separator row.
    for (i, width) in widths.iter().enumerate() {
        if i == 0 {
            print!("├{:─^w$}", "", w = *width);
        } else {
            print!("┼{:─^w$}", "", w = *width);
        }
    }
    print!("┼{:─^w$}┤", "", w = result_width);
    println!();

    // Print each data row.
    for (env, res) in truth_table {
        for (i, var) in var_list.iter().enumerate() {
            let val = if *env.get(var).unwrap() { "T" } else { "F" };
            print!("│{:^width$}", val, width = widths[i]);
        }
        let res_str = if *res { "T" } else { "F" };
        print!("│{:^width$}│", res_str, width = result_width);
        println!();
    }

    // Print the bottom border row.
    print!("└");
    for (i, width) in widths.iter().enumerate() {
        if i == 0 {
            print!("{:─^w$}", "", w = *width);
        } else {
            print!("┴{:─^w$}", "", w = *width);
        }
    }
    print!("┴{:─^w$}┘", "", w = result_width);
}

//
// COMMAND-LINE INTERFACE (Clap)
//

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Display the simplified expression.
    #[clap(short = 's', long = "simplify")]
    simplify: bool,

    /// Display the truth table.
    #[clap(short = 't', long = "table")]
    table: bool,

    /// Boolean expression (e.g., a or b && c). If the expression contains spaces, use quotes.
    expression: Vec<String>,
}

//
// MAIN FUNCTION
//

fn main() {
    // Clap automatically provides help and version info.
    let args = Args::parse();

    if args.expression.is_empty() {
        eprintln!("Error: Please provide a Boolean expression.");
        std::process::exit(1);
    }
    let input_str = args.expression.join(" ");
    match tokenize(&input_str) {
        Ok(tokens) => {
            let mut parser = ExprParser::new(tokens);
            match parser.parse_expression() {
                Ok(ast) => {
                    match simplify_boolean_expression(&ast) {
                        Ok(result) => {
                            // If neither flag is set, display both.
                            if !args.simplify && !args.table {
                                println!("Simplified expression:");
                                println!("{}", result.simplified);
                                println!("\nTruth Table:");
                                print_truth_table(&result.var_list, &result.truth_table);
                            } else {
                                if args.simplify {
                                    println!("Simplified expression:");
                                    println!("{}", result.simplified);
                                }
                                if args.table {
                                    println!("Truth Table:");
                                    print_truth_table(&result.var_list, &result.truth_table);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Simplification error: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Parsing error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("Tokenization error: {}", e);
            std::process::exit(1);
        }
    }
}
