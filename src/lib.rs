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

/* DELETE BELOW COMMENTS IN NEXT VERISON */
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
/* DELETE ABOVE COMMENTS IN NEXT VERISON */

use rusqlite::Connection;

pub struct MemoryDatabase {
    pub conn: Connection,
    pub current_layers: i128,
}

impl MemoryDatabase {
    pub fn open_create(path: &str) -> Self {
        let conn = Connection::open(path).unwrap(); // example "my_database.db"
        MemoryDatabase {
            conn: conn,
            current_layers: 0,
        }
    }
}

pub struct Shape {
    pub x: i128,
    pub y: i128,
}

fn save_layer(/*conn: Connection, layer: i128*/ memory: &mut MemoryDatabase) {
    // Attempt to connect to the database file
    let conn = Connection::open("my_database.db").unwrap();

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
    )
    .unwrap();

    println!("Table ensured to exist!");
}

pub fn create_tensors(memory: &mut MemoryDatabase, shape: Shape) {
    memory.current_layers += 1;

    //let conn_static: Connection = memory.conn.clone();
    //let layer = memory.current_layers;

    //save_layer(conn_static, layer);
    save_layer(memory);

    for row in 0..shape.x {
        for column in 0..shape.y {
            println!(":T:");
        }
    }
}
