# InServiceOfX
Monorepo (single or "mono" repository) for deep learning.

## Creating and starting a virtual environment for Python 3

Create a directory for a virtual environment:

```
/InServiceOfX$ python3 -m venv ./venv/
```

Activate it:
```
/InServiceOfX$ source ./venv/bin/activate
```
You should see the prompt have a prefix `(venv)`.

Deactivate it:
```
deactivate
```

## Running Python tests

From the "base directory" of this repository, you may run the Python unit tests and integration tests as follows:

```
$ pytest ./ThirdParty/NeuralOperators
```
This is a specific example (you may change it) illustrating running the integration tests for `neuraloperators`, and it seems pytest will recursively run all the tests it detects in all the subdirectories.