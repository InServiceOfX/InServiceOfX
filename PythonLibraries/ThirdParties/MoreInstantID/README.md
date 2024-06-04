## Integration tests, `tests/integration_tests`

To run the integration tests on a setup, it's advised to run them individually as pytest, by default, will attempt to run them in parallel. On a setup with relatively low amounts of VRAM, or a single GPU, this causes problems as there's only one device to access! (or not enough VRAM!)

```
InServiceOfX/PythonLibraries/ThirdParties/MoreInstantID/tests/integration_tests# pytest Wrappers/test_create_stable_diffusion_xl_pipeline.py::test_create_stable_diffusion_xl_pipeline_can_change_scheduler

```

From which subdirectory you run the integration test does matter because pytest at least has to recognize a file `conftest.py` in order to "make available" in `sys.path` the modules needed to be imported.