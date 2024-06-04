from pathlib import Path

class VerifyDataSubdirectories:

    data_subdirectories = ["Models",
        "Models/Diffusion",
        "Models/Diffusion/LoRAs"
        "Models/LLM"]

    def __init__(self, base_data_directory: Path):

        self.base_data_dir = base_data_directory

    def verify_or_add_subdirectories(self) -> None:
        """
        @brief Ensure that each predefined subdirectory exists; creating them
        if necessary.
        """
        for subdirectory in VerifyDataSubdirectories.data_subdirectories:
            full_path = self.base_data_dir / subdirectory

            full_path.mkdir(parents=True, exist_ok=True)
            print(f"Ensured directory exists: {full_path}")
