from pathlib import Path
from corecode.Utilities import (
    DataSubdirectories,
    )

from morediffusers.Configurations import IPAdapterConfiguration

from morediffusers.Wrappers import (
    create_stable_diffusion_xl_pipeline,
    load_ip_adapter)

data_sub_dirs = DataSubdirectories()
test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_load_ip_adapter_loads():

    pipe = create_stable_diffusion_xl_pipeline(
        data_sub_dirs.ModelsDiffusion / "stabilityai" / "stable-diffusion-xl-base-1.0",
        None,
        is_enable_cpu_offload=False,
        is_enable_sequential_cpu=False
        )

    test_file_path = test_data_directory / "ip_adapter_configuration_single.yml"
    ip_adapter_configuration = IPAdapterConfiguration(test_file_path)

    load_ip_adapter(pipe, ip_adapter_configuration)

    assert True