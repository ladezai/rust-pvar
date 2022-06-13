use std::{fs::File, io::{self, Read, BufReader, BufRead}, env};

fn read_n<R: Read, const N: usize>(reader: &mut R) -> [u8; N] {
    let mut buf = vec![];
    reader.by_ref().take(N as u64).read_to_end(&mut buf).expect("not enough bytes");
    buf.try_into().unwrap()
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut reader: Box<dyn BufRead> = if args.len() == 1 {
        Box::new(io::stdin().lock())
    } else {
        let file = File::open(&args[1])?;
        Box::new(BufReader::new(file))
    };
    let mut bytes: Vec<u8> = vec![];
    let shape = [usize::from_le_bytes(read_n(&mut reader)), usize::from_le_bytes(read_n(&mut reader))];

    let _nbytes = reader.read_to_end(&mut bytes)?;

    let a: Vec<Vec<f64>> = bytes.chunks(shape[1] * 8).map(|v| v.chunks(8).map(|u| f64::from_le_bytes(u.try_into().unwrap())).collect()).collect();

    println!("{:#.6?}", a);

    Ok(())
}
