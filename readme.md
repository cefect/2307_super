# Super-resolution of flood inundation grids



## install
build conda environment from ./environment.yml

create a ./definitions.py file similar to that shown below

add submodules

add ./coms to PYTHONPATH

## submodules

### flood grid performance
2023-08-07: started incorporating this (trying to migrate off of coms) and gave up
git submodule add -b 2307_super https://github.com/cefect/fperf

 




### definitions.py
```
import os, sys

src_dir = os.path.dirname(os.path.abspath(__file__))
src_name = os.path.basename(src_dir)

# default working directory
wrk_dir = r'l:\10_IO\2307_super'

# logging configuration file
logcfg_file = os.path.join(src_dir, 'logger.conf')

# add graphifviz to system paths (used by DASK for profiling)
os.environ['PATH'] += R";C:\Program Files\Graphviz\bin"
```