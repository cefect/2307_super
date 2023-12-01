# Super-resolution of flood inundation grids



## install
build conda environment from ./environment.yml

create a ./definitions.py file similar to that shown below

add submodules

ensure  your system path has the following:
- Graphviz (C:\Program Files\Graphviz\bin)

 

## submodules

### flood grid performance
2023-08-07: started incorporating this (trying to migrate off of coms) and gave up
git submodule add -b 2307_super https://github.com/cefect/fperf

 




### definitions.py
```
import os, sys

# default working directory
wrk_dir = r'l:\10_IO\2307_super'

 
```