# Logic Solver

**Logic Solver** is a command-line tool written in Rust that parses, simplifies, and generates truth tables for Boolean expressions. It supports both simple and complex variables (enclosed in quotes) and utilizes the Quine–McCluskey algorithm for expression minimization.

## Features

- **Boolean Expression Parsing:**  
  Supports logical operators:
  - **AND:** `and`, `&&`, `∧`
  - **OR:** `or`, `||`, `∨`
  - **NOT:** `not`, `!`, `¬`

- **Constants:**  
  Supports `T` and `F` as constants.

- **Simple Variables:**
  Variables without spaces or special characters are parsed as-is.

- **Complex Variables:**  
  Variables with spaces or special characters can be enclosed in quotes.

- **Expression Simplification:**  
  Uses the Quine–McCluskey algorithm to simplify Boolean expressions.

- **Truth Table Generation:**  
  Generates a complete truth table for the provided expression.

- **Customizable Output:**  
  Option to display only the simplified expression, only the truth table, or both.

## Installation

Ensure you have [Rust](https://www.rust-lang.org/tools/install) installed.

Clone the repository:

```bash
git clone https://github.com/TheJeme/logic-solver.git
cd logic-solver
```

Build the project:

```bash
cargo build --release
```

The compiled binary will be available at `target/release/logic-solver`.

## Usage

```bash
logic-solver [FLAGS] <expression>
```

### Flags

- `-s`, `--simplify`: Display the simplified expression.
- `-t`, `--table`: Display the truth table.
- `-h`, `--help`: Prints help information.
- `-v`, `--version`: Prints version information.  
