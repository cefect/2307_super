'''
Created on Dec. 11, 2023

@author: cef
'''
import os, logging, pprint, webbrowser, sys
import numpy as np
import pandas as pd
from datetime import datetime

today_str = datetime.now().strftime('%Y%m%d')


 

    
#===============================================================================
# MISC-----------
#===============================================================================
def dstr(d,
         width=100, indent=0.3, compact=True, sort_dicts =False,
         ):
    return pprint.pformat(d, width=width, indent=indent, compact=compact, sort_dicts =sort_dicts)

def view(df):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    import webbrowser
    #import pandas as pd
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        #type(f)
        df.to_html(buf=f)
        
    webbrowser.open(f.name)
    
#===============================================================================
# files/folders---------
#===============================================================================

def get_temp_dir(temp_dir_sfx = r'py\temp'):
    
    from pathlib import Path
    homedir = str(Path.home())
    temp_dir = os.path.join(homedir, temp_dir_sfx)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    return temp_dir


def get_directory_size(directory):
    total_size = 0
    for path, dirs, files in os.walk(directory):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    return total_size / 1024 / 1024 / 1024

def get_filepaths(search_dir, ext='.nc', count=None, nested=False):
    """search a directory for files with the provided extension. return the first result"""
    assert os.path.exists(search_dir), search_dir
    find_l = list()
    
    #===========================================================================
    # get a flat list
    #===========================================================================
    if not nested:
    
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith(ext):
                    find_l.append(os.path.join(root, file))
        
        if not count is None:
            assert len(find_l)==count,f'failed to get uinique match {len(find_l)} from \n    {search_dir}'
            if count ==1:
                return find_l[0]
            else:
                raise NotImplementedError(count)
        else:
            return find_l
        
    #===========================================================================
    # get a nested listed (needed by open_mfdataset)
    #===========================================================================
    else:
        nested_l = list()
        #sub_dirs_l = [os.path.join(search_dir, e) for e in os.listdir(search_dir)]
        sub_dirs_l = [os.path.join(search_dir, e) for e in next(os.walk(search_dir))[1]]
        for sdir1 in sub_dirs_l:
            res_l = get_filepaths(sdir1, ext=ext, count=count, nested=False)
            if len(res_l)>0:
                nested_l.append(res_l)
            
        print(f'get_filepaths w/ {len(nested_l)} levels')
        
        return nested_l
        
    
    


 