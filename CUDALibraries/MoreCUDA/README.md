## Building the C++ project

Make a new Build directory from the "top" directory of `MoreCUDA`. You can name it anything but typically I like to name it "BuildGcc", or `Build<compiler type>` where compiler type is the compiler I'm using, such as GCC.
```
MoreCUDA$ mkdir BuildGcc
```
Then
```
cd BuildGcc
cmake ../Source
make
```
Then running `./Check` will run the unit tests suite. For example,

```
MoreCUDA/BuildGcc$ ./Check
```
