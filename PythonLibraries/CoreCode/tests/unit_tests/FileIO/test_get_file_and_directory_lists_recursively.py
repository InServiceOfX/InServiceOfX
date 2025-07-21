from corecode.FileIO import (
    get_file_and_directory_lists_recursively,
    get_project_directory_path,
)

project_directory_path = get_project_directory_path()
corecode_path = project_directory_path / "PythonLibraries" / "CoreCode"

def test_get_file_and_directory_lists_recursively():
    gitignore_path = project_directory_path / ".gitignore"
    parsers_path = corecode_path / "corecode" / "Parsers"

    files, directories = get_file_and_directory_lists_recursively(
        parsers_path,
        gitignore_path)

    expected_file_substrings = [
        "InServiceOfX/PythonLibraries/CoreCode/corecode/Parsers/__init__.py   ",
        "InServiceOfX/PythonLibraries/CoreCode/corecode/Parsers/pdf/to_images.py",
        "InServiceOfX/PythonLibraries/CoreCode/corecode/Parsers/pdf/__init__.py",
        "InServiceOfX/PythonLibraries/CoreCode/corecode/Parsers/parse_tables.py",
        "InServiceOfX/PythonLibraries/CoreCode/corecode/Parsers/jinja/parse_jinja_template.py",
        "InServiceOfX/PythonLibraries/CoreCode/corecode/Parsers/jinja/__init__.py",
    ]

    expected_subdirectory_substrings = [
        "InServiceOfX/PythonLibraries/CoreCode/corecode/Parsers/pdf",
        "InServiceOfX/PythonLibraries/CoreCode/corecode/Parsers/jinja",
    ]

    assert len(expected_file_substrings) == len(files)
    assert len(expected_subdirectory_substrings) == len(directories)

    assert any(
        any(substring in str(file) for substring in expected_file_substrings)
        for file in files
    ), f"No files found containing any of the expected substrings: {expected_file_substrings}"

    assert any(
        any(substring in str(directory) \
            for substring in expected_subdirectory_substrings)
        for directory in directories
    ), f"No directories found containing any of the expected substrings: {expected_subdirectory_substrings}"

    # Uncomment to print the files and directories
    # for file in files:
    #     print(file)

    # for directory in directories:
    #     print(directory)