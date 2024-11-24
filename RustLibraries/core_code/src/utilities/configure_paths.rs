// https://doc.rust-lang.org/std/error/trait.Error.html
// Error is a trait representing basic expectations for error values, i.e. values of type E in
// Result<T, E>.
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct BasicProjectPaths
{
    // PathBuf is an owned, mutable path (akin to String)
    pub configure_paths_path: PathBuf,
    pub core_code_rust_path: PathBuf,
    pub core_code_path: PathBuf,
    pub project_path: PathBuf    
}

// Instead of unwrapping paths directly (which could cause a panic if path isn't valid), it's
// better to return a Result which can be handled gracefully
// https://doc.rust-lang.org/std/boxed/struct.Box.html
// Struct std::boxed::Box
// pub struct Box<T, A = Global>
// Box is a pointer type that uniquely owns a heap allocation of type T.
// dyn is a keyword used to highlight that calls to methods on associated Trait are dynamically
// dispatched.
pub fn setup_paths() -> Result<BasicProjectPaths, Box<dyn Error>>
{
    // ? Operator used to propagate errors in Rust.
    // If value of Result is Ok, value inside Ok will get returned from this expression, and
    // program will continue. If value is an Err, Err will be returned from whole function as if we
    // had used return keyword so error value gets propagated to calling code.
    let core_code_rust_path_buf = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();

    Ok(BasicProjectPaths {
        configure_paths_path: fs::canonicalize(&core_code_rust_path_buf.join("src/utilities"))?,
        core_code_rust_path: core_code_rust_path_buf.clone(),
        core_code_path: fs::canonicalize(&core_code_rust_path_buf.join("../"))?,
        project_path: fs::canonicalize(&core_code_rust_path_buf.join("../../"))?,
    })
}

pub fn get_default_path_to_env_file() -> Result<PathBuf, Box<dyn Error>>
{
    Ok(setup_paths()?.project_path)
}

#[cfg(test)]
mod tests
{
    use super::*;

    fn is_path_contains_substring(path: &PathBuf, substring: &str) -> bool
    {
        // Convert PathBuf to a string.
        // If conversion fails (which can happen if path isn't valid Unicode), return false.
        if let Some(path_str) = path.to_str()
        {
            path_str.contains(substring)
        }
        else
        {
            false
        }
    }

    #[test]
    fn test_setup_paths() -> Result<(), Box<dyn Error>>
    {
        let paths = setup_paths()?;

        assert!(paths.configure_paths_path.exists());
        assert!(paths.core_code_rust_path.exists());
        assert!(paths.core_code_path.exists());
        assert!(paths.project_path.exists());

        assert!(is_path_contains_substring(&paths.configure_paths_path, "InServiceOfX"));
        assert!(is_path_contains_substring(&paths.configure_paths_path, "RustLibraries"));
        assert!(is_path_contains_substring(&paths.configure_paths_path, "core_code"));
        assert!(is_path_contains_substring(&paths.configure_paths_path, "src"));
        assert!(is_path_contains_substring(&paths.configure_paths_path, "utilities"));

        assert!(is_path_contains_substring(&paths.core_code_rust_path, "InServiceOfX"));
        assert!(is_path_contains_substring(&paths.core_code_rust_path, "RustLibraries"));
        assert!(is_path_contains_substring(&paths.core_code_rust_path, "core_code"));
        assert!(!is_path_contains_substring(&paths.core_code_rust_path, "src"));
        assert!(!is_path_contains_substring(&paths.core_code_rust_path, "utilities"));

        assert!(is_path_contains_substring(&paths.core_code_path, "InServiceOfX"));
        assert!(is_path_contains_substring(&paths.core_code_path, "RustLibraries"));
        assert!(!is_path_contains_substring(&paths.core_code_path, "core_code"));
        assert!(!is_path_contains_substring(&paths.core_code_path, "src"));
        assert!(!is_path_contains_substring(&paths.core_code_path, "utilities"));

        assert!(is_path_contains_substring(&paths.project_path, "InServiceOfX"));
        assert!(!is_path_contains_substring(&paths.project_path, "RustLibraries"));
        assert!(!is_path_contains_substring(&paths.project_path, "core_code"));
        assert!(!is_path_contains_substring(&paths.project_path, "src"));
        assert!(!is_path_contains_substring(&paths.project_path, "utilities"));
 
        Ok(())
    }
}