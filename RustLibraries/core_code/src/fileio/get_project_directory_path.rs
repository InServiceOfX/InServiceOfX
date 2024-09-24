use std::env;
use std::path::{PathBuf};

pub fn get_current_filepath() -> PathBuf
{
  let current_dir = env::current_dir().unwrap();

  // Get relative path of current file.
  let file_path = file!();

  let absolute_path = current_dir.join(file_path);

  return absolute_path.to_path_buf();
}

//------------------------------------------------------------------------------
/// \brief Gets the project directory path by traversing up a specific number of
/// levels.
//------------------------------------------------------------------------------
pub fn get_project_directory_path() -> PathBuf
{
    let current_filepath = env::current_exe().unwrap().canonicalize().unwrap();
    let number_of_parents_to_project_path = 6;
    let mut project_path = current_filepath.clone();

    for _ in 0..number_of_parents_to_project_path {
        project_path = project_path.parent().unwrap().to_path_buf();
    }

    project_path
}

//------------------------------------------------------------------------------
/// \brief Gets the project directory path by finding a `.git` directory
/// recursively.
//------------------------------------------------------------------------------
pub fn get_project_directory_path_recursive() -> Result<PathBuf, &'static str> {
    let current_filepath = env::current_exe().unwrap().canonicalize().unwrap();

    for parent in current_filepath.ancestors() {
        if parent.join(".git").exists() {
            return Ok(parent.to_path_buf());
        }
    }

    Err("Repository main directory not found or .git wasn't.")
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn test_get_project_directory_path() {
        let project_path = get_project_directory_path();
        // Assuming the project directory contains Cargo.toml
        //assert!(project_path.join("Cargo.toml").exists());

        println!("Project directory path1: {:?}", project_path);
    }

    #[test]
    fn test_get_project_directory_path_contains_subdirectories()
    {
        let project_path = get_current_filepath();

        println!("Project directory path2: {:?}", project_path);

        let project_path = get_project_directory_path();
        //let project_path_str = project_path.to_str().unwrap();

        println!("Project directory path3: {:?}", project_path);

        let project_path = get_project_directory_path_recursive();

        println!("Project directory path4: {:?}", project_path)
    }

    #[test]
    fn test_get_project_directory_path_recursive()
    {
        match get_project_directory_path_recursive()
        {
            Ok(path) =>
            {
                // Assuming the project directory contains .git
                assert!(path.join(".git").exists());
            }
            Err(e) => panic!("{}", e),
        }
    }
}
