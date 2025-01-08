pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

// TODO documation like code on top
use rusqlite::Connection;
//pub struct // stores and intilizes which database your using for tensor

/*
pub fn tensor_test() {
    let mut test = vec![1, 3, 4];
    test.push(1);
    println!("{test:?}");
}
*/

/*
pub fn open_create_memory_database(path: &str) -> Connection {
    let conn = Connection::open(path).unwrap();
    conn
}
*/

/*
use rusqlite::Connection;

fn main() -> rusqlite::Result<()> {
    // Attempt to connect to the database file
    let conn = Connection::open("my_database.db")?;

    // If "my_database.db" does not exist, it will be created automatically.
    println!("Database connected or created successfully!");

    // Optional: You can create a table as well if it doesn't exist
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER
        )",
        [],
    )?;

    println!("Table ensured to exist!");

    Ok(())
}
*/

pub struct MemoryDatabase {
    pub database: Connection,
}

impl MemoryDatabase {
    pub fn open_create(path: &str) -> Self {
        let conn = Connection::open(path).unwrap(); // example "my_database.db"
        MemoryDatabase { database: conn }
    }
}

/*
pub struct Shape {
    pub values: Vec<i32>,
}

/*
impl Shape {
    fn new
}
*/

pub fn create_tensor(shape: Shape) {
    println!("sdjdsf");
}
*/

/*
pub struct Shape<'a> {
    pub values: &'a [i32],
}

pub fn create_tensors(shape: Shape) {
    let shape_values = shape.values;
    for &shape_value in shape_values {
        println!("tensor: {:?}", shape_values);
    }
}
*/
