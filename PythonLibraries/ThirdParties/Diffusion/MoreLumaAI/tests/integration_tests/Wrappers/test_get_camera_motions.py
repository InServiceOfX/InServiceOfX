from corecode.Utilities import (load_environment_file, get_environment_variable)
from morelumaai.Wrappers import get_camera_motions

load_environment_file()

def test_get_camera_motions():
    camera_motions = get_camera_motions(
        get_environment_variable("LUMAAI_API_KEY"))
    assert len(camera_motions) > 0
    assert set(camera_motions) == set(
        ['Static', 'Move Left', 'Move Right', 'Move Up', 'Move Down', 'Push In', 'Pull Out', 'Zoom In', 'Zoom Out', 'Pan Left', 'Pan Right', 'Orbit Left', 'Orbit Right', 'Crane Up', 'Crane Down'])

