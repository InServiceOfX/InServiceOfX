Created with 
```
poetry new MoreDiffusers
```

See https://python-poetry.org/docs/basic-usage/

## `git clone` HuggingFace Models

```
GIT_LFS_SKIP_SMUDGE=1 git clone
```
Also, try with the `--progress` option.

See https://gist.github.com/iamalbert/ee4b4c89da02e2f9a12b6d700eec7c84

Otherwise LFS files are large.

To pull, i.e. download, the large LFS file, do something like this:

```
$ git lfs pull --include="v1-5-pruned.safetensors"
```

You're going to want to indicate the file you want to pull by its relative path to the original repository structure, not relative from your local copy.

### Running integration tests