See https://github.com/karpathy/llm.c/blob/master/README.md and https://github.com/karpathy/llm.c/blob/master/dev/cuda/README.md

It says in https://github.com/karpathy/llm.c/blob/master/README.md

"our `dev/cuda` folder is a place for a library of kernels for all the layers that are manually hand-written and very well documented, starting from very simple kernels all the way to more complex /faster kernels."

"In comparison, the `dev/` folder is a bit more of a scratch space for us to develop a library of kernels or classes and share useful or related or educationals code, and some of this code could be ok to be (locally) complex."

As with `dev/cuda` "this directory is scratch space for developing various versions of the needed CUDA kernels. Each file develops a kernel, and usually multiple versions of that kernel that could have different running times and of different code or time complexity."

Similarly, but more to the point, this directory contains only scratch *drafts*. Those selected to be the "best" are used in the other directories.
