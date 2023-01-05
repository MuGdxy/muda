# How to import the library to your workspace

## xmake user

```shell
$ xmake create --language=cuda hellomuda
```

Your `hellomuda` directory may looks like this
```shell
- hellomuda
    - src
        - main.cu
    xmake.lua
```
And configure xmake.lua as the following:

```lua

includes("D:/repos/cent/external/muda") -- CHANGE CONTENT HERE WITH THE_PATH_TO_MUDA_REPO
set_languages("cxx17") -- RECOMMAND TO USE C++17 FEATURES
add_rules("mode.debug", "mode.release")

target("hellocuda")
    set_kind("binary")
    add_deps("muda-full") -- ADD MUDA FULL DEPENDENCIES
    add_files("src/*.cu")
    -- SET ACCORDING TO YOUR OWN ARCHITECTURE
    add_cugencodes("native")
    add_cugencodes("compute_75")
```

And edit `main.cu` as a hello world file

```c++
#include <muda/muda.h>

int main()
{
    muda::launch(1,1)
        .apply(
            [] __device__()
            {
                printf("hello muda!\n");
            }
        ).wait();
    return 0;
}

```


Then we can compile your own project

```shell
$ xmake
```

Your are supposed to output a "hello muda!" string to console

```shell
$ xmake run hellomuda
hello muda!
```