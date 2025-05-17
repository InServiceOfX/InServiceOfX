## Integration Tests

To run them, one can run `pytest` from the following directory, and specify the subdirectory `integration_tests` in the following manner:

```
InServiceOfX/PythonLibraries/ThirdParties/MoreInsightFace/tests# pytest ./integration_tests/
```

Because the integration test is using substantial or almost all CPU or GPU resources, you'll want to run each integration test individually. I also found that you'll want to run it from a parent directory that contains a `conftest.py` file that'll add relative paths recognized by the system (Python's `sys.path`).

For example,

```
/InServiceOfX/PythonLibraries/ThirdParties/MoreInsightFace/tests# pytest ./integration_tests/Wrappers/test_FaceAnalysis.py::test_FaceAnalysisWrapper_inits
```

To print out `print` statements onto the console, you can add the `--capture=no` option:

```
/InServiceOfX/PythonLibraries/ThirdParties/MoreInsightFace/tests# pytest --capture=no ./integration_tests/Wrappers/test_FaceAnalysis.py::test_FaceAnalysisWrapper_inits
```
or
```
/InServiceOfX/PythonLibraries/ThirdParties/MoreInsightFace/tests# pytest --capture=no ./integration_tests/Wrappers/test_FaceAnalysis.py::test_FaceAnalysisWrapper_gets_face_embedding
```