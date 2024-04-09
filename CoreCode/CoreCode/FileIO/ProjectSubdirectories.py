from .get_main_directory_path import get_main_directory_path

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ProjectSubdirectories:
    ThirdParty__unit_tests__SampleData: Path = field(init=False)

    """
    We do not define a __init__(..) function because then the fields would be
    positional arguments required when creating a class instance.
    """

    def __post_init__(self):
        self.ThirdParty__unit_tests__SampleData = \
            get_main_directory_path() / "ThirdParty" / "unit_tests" / "SampleData/"