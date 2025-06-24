from pathlib import Path
import sys

class SetupInternalModules:
    def __init__(self):
        self._project_path = Path(__file__).parents[3]

        self._commonapi_path = self._project_path / "PythonLibraries" / \
            "ThirdParties" / "APIs" / "CommonAPI"

        self._corecode_path = self._project_path / "PythonLibraries" / \
            "CoreCode"

        self._moregroq_path = self._project_path / "PythonLibraries" / \
            "ThirdParties" / "APIs" / "MoreGroq"

        if self._commonapi_path.exists():
            commonapi_path_str = str(self._commonapi_path)
            if commonapi_path_str not in sys.path:
                sys.path.append(commonapi_path_str)

        if self._corecode_path.exists():
            corecode_path_str = str(self._corecode_path)
            if corecode_path_str not in sys.path:
                sys.path.append(corecode_path_str)

        if self._moregroq_path.exists():
            moregroq_path_str = str(self._moregroq_path)
            if moregroq_path_str not in sys.path:
                sys.path.append(moregroq_path_str)
