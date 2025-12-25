use super::configure_paths::{setup_paths, get_default_path_to_env_file};
use dotenv;
use std::path::{Path};

pub fn load_environment_file(path: &str)
{
    dotenv::from_filename(Path::new(path)).ok();
}

pub fn load_environment_file_from_default_path()
{
    let env_path = get_default_path_to_env_file()
        .expect("Failed to get default .env path");
    dotenv::from_filename(&env_path).ok();
}

use std::env;

pub fn get_environment_variable(name: &str) -> Result<String, env::VarError>
{
    env::var(name)
}

/// Panics if variable not set (matches Python's os.environ[name] behavior).
pub fn get_environment_variable_unwrap(name: &str) -> String
{
    env::var(name).expect(&format!("Environment variable '{}' not set", name))
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

        load_environment_file(example_path.to_str().expect(
            "Path expected to exist"));

        let key = "OPENAI_API_KEY";
        let value = dotenv::var(key).unwrap();

        assert_eq!(value, "your_api_key_here");
    }

    #[test]
    fn test_get_environment_variable()
    {
        env::set_var("TEST_KEY", "test_value");
        assert_eq!(get_environment_variable("TEST_KEY").unwrap(), "test_value");

        // Test missing var.
        assert!(get_environment_variable("MISSING").is_err());
    }

    #[test]
    #[should_panic(expected = "Environment variable 'PANIC_KEY' not set")]
    fn test_get_environment_variable_unwrap_panics() {
        get_environment_variable_unwrap("PANIC_KEY");
    }
}
