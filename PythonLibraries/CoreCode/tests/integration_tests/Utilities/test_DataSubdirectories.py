from corecode.Utilities import DataSubdirectories

def test_DataSubdirectories_Data_fields_are_equal():
    data_subdirectories = DataSubdirectories()
    assert data_subdirectories.Data == \
        data_subdirectories.get_data_path(0)
