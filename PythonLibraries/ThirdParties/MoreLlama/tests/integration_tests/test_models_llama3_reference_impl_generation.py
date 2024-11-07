from corecode.Utilities import DataSubdirectories
from corecode.FileIO import is_zipfile

import io
import json
import pickle
from pathlib import Path
import torch

from torch.serialization import (_open_zipfile_reader, _is_torchscript_zip)

data_sub_dirs = DataSubdirectories()

def test_Llama_build():
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    ckpt_dir = pretrained_model_path / "original"    
    
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

    assert isinstance(checkpoints, list)
    assert len(checkpoints) == 1
    assert isinstance(checkpoints[0], Path) 
    assert str(checkpoints[0]).endswith(".pth")
    assert "original" in str(checkpoints[0])
    assert "consolidated.00.pth" in str(checkpoints[0])

    checkpoint = torch.load(
        checkpoints[0],
        map_location="cpu",
        weights_only=True)

    assert isinstance(checkpoint, dict)
    assert "layers.0.attention.wk.weight" in checkpoint
    expected_keys = [
        'layers.0.attention.wk.weight',
        'layers.0.attention.wo.weight',
        'layers.0.attention.wq.weight',
        'layers.0.attention.wv.weight',
        'layers.0.attention_norm.weight',
        'layers.0.feed_forward.w1.weight',
        'layers.0.feed_forward.w2.weight',
        'layers.0.feed_forward.w3.weight',
        'layers.0.ffn_norm.weight',
        'layers.1.attention.wk.weight',
        'layers.1.attention.wo.weight',
        'layers.1.attention.wq.weight',
        'layers.1.attention.wv.weight',
        'layers.1.attention_norm.weight',
        'layers.1.feed_forward.w1.weight',
        'layers.1.feed_forward.w2.weight',
        'layers.1.feed_forward.w3.weight',
        'layers.1.ffn_norm.weight',
        'layers.10.attention.wk.weight',
        'layers.10.attention.wo.weight',
        'layers.10.attention.wq.weight',
        'layers.10.attention.wv.weight',
        'layers.10.attention_norm.weight',
        'layers.10.feed_forward.w1.weight',
        'layers.10.feed_forward.w2.weight',
        'layers.10.feed_forward.w3.weight',
        'layers.10.ffn_norm.weight',
        'layers.11.attention.wk.weight',
        'layers.11.attention.wo.weight',
        'layers.11.attention.wq.weight',
        'layers.11.attention.wv.weight',
        'layers.11.attention_norm.weight',
        'layers.11.feed_forward.w1.weight',
        'layers.11.feed_forward.w2.weight',
        'layers.11.feed_forward.w3.weight',
        'layers.11.ffn_norm.weight',
        'layers.12.attention.wk.weight',
        'layers.12.attention.wo.weight',
        'layers.12.attention.wq.weight',
        'layers.12.attention.wv.weight',
        'layers.12.attention_norm.weight',
        'layers.12.feed_forward.w1.weight',
        'layers.12.feed_forward.w2.weight',
        'layers.12.feed_forward.w3.weight',
        'layers.12.ffn_norm.weight',
        'layers.13.attention.wk.weight',
        'layers.13.attention.wo.weight',
        'layers.13.attention.wq.weight',
        'layers.13.attention.wv.weight',
        'layers.13.attention_norm.weight',
        'layers.13.feed_forward.w1.weight',
        'layers.13.feed_forward.w2.weight',
        'layers.13.feed_forward.w3.weight',
        'layers.13.ffn_norm.weight',
        'layers.14.attention.wk.weight',
        'layers.14.attention.wo.weight',
        'layers.14.attention.wq.weight',
        'layers.14.attention.wv.weight',
        'layers.14.attention_norm.weight',
        'layers.14.feed_forward.w1.weight',
        'layers.14.feed_forward.w2.weight',
        'layers.14.feed_forward.w3.weight',
        'layers.14.ffn_norm.weight',
        'layers.15.attention.wk.weight',
        'layers.15.attention.wo.weight',
        'layers.15.attention.wq.weight',
        'layers.15.attention.wv.weight',
        'layers.15.attention_norm.weight',
        'layers.15.feed_forward.w1.weight',
        'layers.15.feed_forward.w2.weight',
        'layers.15.feed_forward.w3.weight',
        'layers.15.ffn_norm.weight',
        'layers.2.attention.wk.weight',
        'layers.2.attention.wo.weight',
        'layers.2.attention.wq.weight',
        'layers.2.attention.wv.weight',
        'layers.2.attention_norm.weight',
        'layers.2.feed_forward.w1.weight',
        'layers.2.feed_forward.w2.weight',
        'layers.2.feed_forward.w3.weight',
        'layers.2.ffn_norm.weight',
        'layers.3.attention.wk.weight',
        'layers.3.attention.wo.weight',
        'layers.3.attention.wq.weight',
        'layers.3.attention.wv.weight',
        'layers.3.attention_norm.weight',
        'layers.3.feed_forward.w1.weight',
        'layers.3.feed_forward.w2.weight',
        'layers.3.feed_forward.w3.weight',
        'layers.3.ffn_norm.weight',
        'layers.4.attention.wk.weight',
        'layers.4.attention.wo.weight',
        'layers.4.attention.wq.weight',
        'layers.4.attention.wv.weight',
        'layers.4.attention_norm.weight',
        'layers.4.feed_forward.w1.weight',
        'layers.4.feed_forward.w2.weight',
        'layers.4.feed_forward.w3.weight',
        'layers.4.ffn_norm.weight',
        'layers.5.attention.wk.weight',
        'layers.5.attention.wo.weight',
        'layers.5.attention.wq.weight',
        'layers.5.attention.wv.weight',
        'layers.5.attention_norm.weight',
        'layers.5.feed_forward.w1.weight',
        'layers.5.feed_forward.w2.weight',
        'layers.5.feed_forward.w3.weight',
        'layers.5.ffn_norm.weight',
        'layers.6.attention.wk.weight',
        'layers.6.attention.wo.weight',
        'layers.6.attention.wq.weight',
        'layers.6.attention.wv.weight',
        'layers.6.attention_norm.weight',
        'layers.6.feed_forward.w1.weight',
        'layers.6.feed_forward.w2.weight',
        'layers.6.feed_forward.w3.weight',
        'layers.6.ffn_norm.weight',
        'layers.7.attention.wk.weight',
        'layers.7.attention.wo.weight',
        'layers.7.attention.wq.weight',
        'layers.7.attention.wv.weight',
        'layers.7.attention_norm.weight',
        'layers.7.feed_forward.w1.weight',
        'layers.7.feed_forward.w2.weight',
        'layers.7.feed_forward.w3.weight',
        'layers.7.ffn_norm.weight',
        'layers.8.attention.wk.weight',
        'layers.8.attention.wo.weight',
        'layers.8.attention.wq.weight',
        'layers.8.attention.wv.weight',
        'layers.8.attention_norm.weight',
        'layers.8.feed_forward.w1.weight',
        'layers.8.feed_forward.w2.weight',
        'layers.8.feed_forward.w3.weight',
        'layers.8.ffn_norm.weight',
        'layers.9.attention.wk.weight',
        'layers.9.attention.wo.weight',
        'layers.9.attention.wq.weight',
        'layers.9.attention.wv.weight',
        'layers.9.attention_norm.weight',
        'layers.9.feed_forward.w1.weight',
        'layers.9.feed_forward.w2.weight',
        'layers.9.feed_forward.w3.weight',
        'layers.9.ffn_norm.weight',
        'norm.weight',
        'output.weight',
        'tok_embeddings.weight']

    assert set(checkpoint.keys()) == set(expected_keys)
    assert isinstance(checkpoint['layers.0.attention.wk.weight'], torch.Tensor)
    assert checkpoint['layers.0.attention.wk.weight'].shape == (512, 2048)
    assert checkpoint['layers.0.attention.wo.weight'].shape == (2048, 2048)
    assert checkpoint['layers.0.attention.wq.weight'].shape == (2048, 2048)
    assert checkpoint['layers.0.attention.wv.weight'].shape == (512, 2048)
    assert checkpoint['layers.0.attention_norm.weight'].shape == torch.Size(
        [2048])
    
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    assert isinstance(params, dict)

    expected_params_keys = [
        'dim',
        'ffn_dim_multiplier',
        'multiple_of',
        'n_heads',
        'n_kv_heads',
        'n_layers',
        'norm_eps',
        'rope_theta',
        'use_scaled_rope',
        'vocab_size']

    assert set(params.keys()) == set(expected_params_keys)
    assert params["dim"] == 2048
    assert params["ffn_dim_multiplier"] == 1.5
    assert params["multiple_of"] == 256
    assert params["n_heads"] == 32
    assert params["n_kv_heads"] == 8
    assert params["n_layers"] == 16
    assert params["norm_eps"] == 1e-05
    assert params["rope_theta"] == 500000.0
    assert params["use_scaled_rope"] == True
    assert params["vocab_size"] == 128256

def test_torch_load():
    pretrained_model_path = data_sub_dirs.ModelsLLM / "meta-llama" / \
        "Llama-3.2-1B-Instruct"

    ckpt_dir = pretrained_model_path / "original"    
    
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

    # Instead of doing torch.load(..), let's show the steps taken by the load
    # implementation. See torch/serialization.py in pytorch.

    # in def load(..) in torch/serialization.py, since
    # "encoding" not in pickle_load_args.keys(), then implementation does
    # pickle_load_args["encoding"] = "utf-8"
    pickle_load_args = {}
    pickle_load_args["encoding"] = "utf-8"

    with open(checkpoints[0], "rb") as f:
        assert is_zipfile(f)
        original_position = f.tell()
        overall_storage=None
        # In our case, overall_storage is None, so we assert this as we're
        # tracing through the implementation.
        assert overall_storage is None
        with _open_zipfile_reader(f) as opened_zipfile:
            assert not _is_torchscript_zip(opened_zipfile)

        # See the nested if statements in def load(..) in
        # torch/serialization.py where
        # with _open_zipfile_reader(f) as opened_zipfile:
        #     if _is_zipfile(opened_file):
        #         ...
        #         with _open_zipfile_reader(opened_zipfile) as opened_zipfile:
        # ...
        # For the immediate, above case, we enter if weights_only: case.

            pickle_module = None
            # In our case, pickle_module is None, so we assert this as we're tracing
            # through the implementation.
            assert pickle_module is None

            # Jump as look at def _load(..) in torch/serialization.py.

            byteordername = "byteorder"
            byteorderdata = None
            assert opened_zipfile.has_record(byteordername)
            assert opened_zipfile.get_record(byteordername) == b'little'
            byteorderdata = opened_zipfile.get_record(byteordername)

            pickle_file="data.pkl"

            data_file = io.BytesIO(opened_zipfile.get_record(pickle_file))
            assert isinstance(data_file, io.BytesIO)

            binary_data = data_file.getvalue()
            assert isinstance(binary_data, bytes)
            assert len(binary_data) == 18792

            # Test first few bytes (pickle protocol marker and version)
            assert binary_data[0] == 0x80
            assert binary_data[1] == 0x02

            assert binary_data[10:14] == b'\x00tok'

            # Because pickle_module is None from load(..), then pickle_module
            # gets default value pickle, which goes into _load(..). All of this
            # is in torch/serialization.py.
            unpickler = pickle.Unpickler(data_file, **pickle_load_args)