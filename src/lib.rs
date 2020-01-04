pub mod regression;


#[cfg(test)]
mod tests {

    use crate::regression::linear::add;

    #[test]
    fn it_works() {
        assert_eq!(add(2, 2), 4);
    }
}
