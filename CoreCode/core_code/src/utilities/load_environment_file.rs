use super::configure_paths::setup_paths;
use dotenv;
use std::path::{Path};

pub fn load_environment_file(path: &str)
{
    dotenv::from_filename(Path::new(path)).ok();
}

pub fn load_environment_file_from_default_path()
{
    dotenv::from_filename(Path::new(&setup_paths()
        .expect("setup_paths should have file directories")
        .project_path))
        .ok();
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn test_load_environment_file_loads_example()
    {
        let example_path = setup_paths()
            .expect("setup_paths should have file directories")
            .project_path.join(".envExample");

        load_environment_file(example_path.to_str().expect("Path expected to exist"));

        let key = "OPENAI_API_KEY";
        let value = dotenv::var(key).unwrap();

        assert_eq!(value, "your_api_key_here");
    }
}
