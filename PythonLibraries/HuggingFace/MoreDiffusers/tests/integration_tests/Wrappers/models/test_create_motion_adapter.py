from corecode.Utilities import (
    DataSubdirectories,
    )

from morediffusers.Wrappers.models import create_motion_adapter

import os
import torch.nn as nn

data_sub_dirs = DataSubdirectories()

def test_create_motion_adapter_creates_model():

    pretrained_model_name_or_path = \
        data_sub_dirs.ModelsDiffusion / "guoyww" / "animatediff-motion-adapter-sdxl-beta"

    adapter = create_motion_adapter(
        pretrained_model_name_or_path,
        )

    assert os.path.isdir(pretrained_model_name_or_path)

    assert adapter.conv_in == None
    assert adapter.mid_block == None
    assert adapter.down_blocks != None
    assert adapter.up_blocks != None
    assert isinstance(adapter.down_blocks, nn.ModuleList)
    assert isinstance(adapter.up_blocks, nn.ModuleList)
