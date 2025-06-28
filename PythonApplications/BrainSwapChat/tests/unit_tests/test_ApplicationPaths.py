import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import os

from brainswapchat.ApplicationPaths import ApplicationPaths

class TestApplicationPaths:

    TEST_ROOT = Path(__file__).resolve().parents[1] / "TestSetups" / "TestData"

    def test_create_path_names_with_new_project_root(self):
        
        result = ApplicationPaths.create_path_names(
            new_project_root_path=self.TEST_ROOT)
        
        assert isinstance(result, ApplicationPaths)
        expected_path = self.TEST_ROOT / "Configurations" / \
            "system_messages.json"
        assert result.system_messages_file_path == expected_path

    def test_create_path_names_with_default_parameters(self):

        result = ApplicationPaths.create_path_names()

        assert isinstance(result, ApplicationPaths)
        expected_path = Path.home() / ".config" / "brainswapchat" / \
            "system_messages.json"
        assert result.system_messages_file_path == expected_path

    def test_create_missing_files(self):
        """Create file when they didn't exist before."""
        test_file_path = self.TEST_ROOT / "TestApplicationPaths" / \
            "Configurations" / "system_messages.json"

        assert not test_file_path.exists()

        app_paths = ApplicationPaths.create_path_names(
            new_project_root_path=self.TEST_ROOT / "TestApplicationPaths")

        result = app_paths.create_missing_files("system_messages_file_path")

        assert test_file_path.exists()
        assert result == {"system_messages_file_path": True}

        # Clean up
        shutil.rmtree(self.TEST_ROOT / "TestApplicationPaths")

    def test_create_missing_files_already_exists(self):
        test_file_path = self.TEST_ROOT / "TestApplicationPaths" / \
            "Configurations" / "system_messages.json"

        Path(self.TEST_ROOT / "TestApplicationPaths" / "Configurations").mkdir(
            parents=True, exist_ok=True)
        test_file_path.touch()
        
        app_paths = ApplicationPaths.create_path_names(
            new_project_root_path=self.TEST_ROOT / "TestApplicationPaths")
        
        result = app_paths.create_missing_files("system_messages_file_path")
        
        assert result == {"system_messages_file_path": True}
        assert test_file_path.exists()

        # Clean up
        shutil.rmtree(self.TEST_ROOT / "TestApplicationPaths")


    # TODO: Check if following tests are worth adding.
    #def test_check_paths_exist_all_exist(self, tmp_path):
    #     """Test check_paths_exist when all paths exist."""
    #     # Create a temporary file
    #     test_file = tmp_path / "test_system_messages.json"
    #     test_file.touch()
        
    #     app_paths = ApplicationPaths(system_messages_file_path=test_file)
    #     result = app_paths.check_paths_exist()
        
    #     assert result == {"system_messages_file_path": True}

    # def test_check_paths_exist_none_exist(self, tmp_path):
    #     """Test check_paths_exist when no paths exist."""
    #     non_existent_file = tmp_path / "non_existent.json"
        
    #     app_paths = ApplicationPaths(system_messages_file_path=non_existent_file)
    #     result = app_paths.check_paths_exist()
        
    #     assert result == {"system_messages_file_path": False}

    # def test_check_paths_exist_mixed(self, tmp_path):
    #     """Test check_paths_exist with some paths existing and others not."""
    #     # Create one file
    #     existing_file = tmp_path / "existing.json"
    #     existing_file.touch()
        
    #     # Use non-existent file for the other
    #     non_existent_file = tmp_path / "non_existent.json"
        
    #     app_paths = ApplicationPaths(system_messages_file_path=existing_file)
    #     result = app_paths.check_paths_exist()
        
    #     assert result == {"system_messages_file_path": True}

    # def test_create_missing_files_invalid_path_name(self):
    #     """Test create_missing_files with invalid path name."""
    #     app_paths = ApplicationPaths(system_messages_file_path=Path("/test/path"))
        
    #     result = app_paths.create_missing_files("invalid_path_name")
        
    #     assert result == {"invalid_path_name": False}

    # def test_create_missing_files_multiple_paths(self, tmp_path):
    #     """Test create_missing_files with multiple path names."""
    #     test_file_path = tmp_path / "system_messages.json"
        
    #     app_paths = ApplicationPaths(system_messages_file_path=test_file_path)
        
    #     result = app_paths.create_missing_files("system_messages_file_path")
        
    #     assert result == {"system_messages_file_path": True}
    #     assert test_file_path.exists()

    # def test_create_all_missing_paths(self, tmp_path):
    #     """Test create_all_missing_paths creates all files."""
    #     test_file_path = tmp_path / "Configurations" / "system_messages.json"
        
    #     app_paths = ApplicationPaths(system_messages_file_path=test_file_path)
        
    #     # Verify file doesn't exist initially
    #     assert not test_file_path.exists()
        
    #     # Create all missing files
    #     result = app_paths.create_all_missing_paths()
        
    #     # Verify file was created
    #     assert test_file_path.exists()
    #     assert result == {"system_messages_file_path": True}
        
    #     # Clean up
    #     shutil.rmtree(tmp_path / "Configurations")

    # def test_integration_with_testdata_directory(self):
    #     """Integration test using TestData directory path."""
    #     # Get the TestData directory path
    #     test_data_path = Path(__file__).parent.parent / "TestSetups" / "TestData"
        
    #     # Create ApplicationPaths with TestData as project root
    #     app_paths = ApplicationPaths.create_path_names(new_project_root_path=test_data_path)
        
    #     # Check if paths exist (should be False initially)
    #     existence_check = app_paths.check_paths_exist()
    #     assert existence_check == {"system_messages_file_path": False}
        
    #     # Create the missing file
    #     creation_result = app_paths.create_missing_files("system_messages_file_path")
    #     assert creation_result == {"system_messages_file_path": True}
        
    #     # Verify file was created in the correct location
    #     expected_file = test_data_path / "Configurations" / "system_messages.json"
    #     assert expected_file.exists()
        
    #     # Clean up - remove the created file and directory
    #     if expected_file.exists():
    #         expected_file.unlink()
    #     if expected_file.parent.exists():
    #         expected_file.parent.rmdir()

    # def test_create_missing_files_with_parent_directories(self, tmp_path):
    #     """Test that create_missing_files creates parent directories."""
    #     # Create a path that requires parent directories
    #     deep_path = tmp_path / "deep" / "nested" / "path" / "system_messages.json"
        
    #     app_paths = ApplicationPaths(system_messages_file_path=deep_path)
        
    #     # Verify parent directories don't exist
    #     assert not deep_path.parent.exists()
        
    #     # Create the file
    #     result = app_paths.create_missing_files("system_messages_file_path")
        
    #     # Verify file and parent directories were created
    #     assert deep_path.exists()
    #     assert deep_path.parent.exists()
    #     assert result == {"system_messages_file_path": True}
        
    #     # Clean up
    #     shutil.rmtree(tmp_path / "deep")

    # def test_error_handling_in_create_missing_files(self):
    #     """Test error handling in create_missing_files."""
    #     # Create a path that would cause permission issues (if possible)
    #     # This is a basic test - in practice, you'd need to test with actual permission issues
        
    #     app_paths = ApplicationPaths(system_messages_file_path=Path("/root/system_messages.json"))
        
    #     # This should handle the error gracefully
    #     result = app_paths.create_missing_files("system_messages_file_path")
        
    #     # Should return False for failed creation
    #     assert result == {"system_messages_file_path": False}
