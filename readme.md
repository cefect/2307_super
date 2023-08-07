# Super-resolution of flood inundation grids



## install
build conda environment from ./environment.yml

create a ./definitions.py file similar to that shown below





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