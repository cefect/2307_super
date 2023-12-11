'''
Created on Dec. 10, 2023

@author: cef

test for cnnf.post_plots
'''

import pytest, os


from superd.cnnf.post_plots import Plot_inun_peformance
#===============================================================================
# module variables
#===============================================================================

from definitions import test_data_dir
tdata = lambda x:os.path.join(test_data_dir, x)


#===============================================================================
# tests
#===============================================================================

@pytest.mark.parametrize('pick_fp',[tdata('meta_20231205.pkl')]) 
def test_plot_metric_v_backprops(pick_fp, tmp_path):
    Plot_inun_peformance().plot()