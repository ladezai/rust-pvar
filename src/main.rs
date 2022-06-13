use pvar::ndarray::Array;
use std::{fs::File, io::{self, Read, BufReader}};

fn main() -> io::Result<()> {
    let file = File::open("test.bin")?;
    let mut reader = BufReader::new(file);
    let mut bytes: Vec<u8> = vec![];

    let nbytes = reader.read_to_end(&mut bytes)?;
    println!("Read {} bytes", nbytes);

    let a = Array::from_bytes(bytes, [10, 10]);

    println!("{}", a.unwrap());

    Ok(())
}
