'''
Created on Aug. 22, 2023

@author: cefect

skinny implementation of fperf
'''
import logging, datetime, copy, math

import numpy as np
import pandas as pd
from hp.logr import get_log_stream

from parameters import confusion_codes

from sklearn.metrics import confusion_matrix
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal

class ErrorCalcs(object):
    def __init__(self,
            pred_ser=None,
            true_ser=None,
            logger=None,
            normed=False,
            ):
        #attach
        
        self.pred_ser=pred_ser.rename('pred')
        self.true_ser=true_ser.rename('true')
        self.normed=normed
        
        self.check_match()
        
        self.df_raw = pred_ser.rename('pred').to_frame().join(true_ser.rename('true'))
        
        
        self.logger=logger
        
        self.res_d = dict()
        
        
        self.data_retrieve_hndls = {
            'bias':         lambda **kwargs:self.get_bias(**kwargs),
            'bias_shift':       lambda **kwargs:self.get_bias1(**kwargs),
            'meanError':    lambda **kwargs:self.get_meanError(**kwargs),
            'meanErrorAbs': lambda **kwargs:self.get_meanErrorAbs(**kwargs),
            'RMSE':         lambda **kwargs:self.get_RMSE(**kwargs),
            'pearson':      lambda **kwargs:self.get_pearson(**kwargs),
            'confusion':    lambda **kwargs:self.get_confusion(**kwargs),
            'stats':        lambda **kwargs:self.get_stats(**kwargs),
            }
        
    def retrieve(self, #skinny retrival
                 dkey,
 
                 logger=None,
                 **kwargs
                 ):
        """based on oop.Session.retrieve"""
 
        if logger is None: logger=self.logger
        log = logger.getChild('ret')
        
        drh_d = self.data_retrieve_hndls

        start = datetime.datetime.now()
        
        assert dkey in drh_d, dkey
        
        f = drh_d[dkey]
        
        return f(dkey=dkey, logger=log, **kwargs)
    
    def get_bias1(self, #shift bias to be zero centered
                  dkey='bias_1',
                  **kwargs):
        return self.get_bias(**kwargs)-1
        
    def get_bias(self,
                 per_element=False,
                 dkey='bias', logger=None,
                 ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        assert dkey=='bias'
        log = logger.getChild('bias')
        pred_ser=self.pred_ser
        true_ser=self.true_ser
        df = self.df_raw.copy()

        
        if not per_element:
            s1 = df.sum()
            return s1['pred']/s1['true']
        else:
            res1 = pred_ser/true_ser
            res1 = res1.rename('bias')
            #=======================================================================
            # both zeros
            #=======================================================================
            #true_bx = true_ser==0
            
            bx = np.logical_and(
                true_ser==0,
                pred_ser==0)
            
            if bx.any():
                log.info('replacing %i/%i zero matches w/ 1.0'%(bx.sum(), len(bx)))
                res1.loc[bx] = 1.0
                
            #=======================================================================
            # pred zeros
            #=======================================================================
            bx = np.logical_and(
                true_ser!=0,
                pred_ser==0)
            
            if bx.any():
                log.info('replacing %i/%i zero mismatches w/ null'%(bx.sum(), len(bx)))
                res1.loc[bx] = np.nan
                
            #=======================================================================
            # true zeros
            #=======================================================================
            bx = np.logical_and(
                true_ser==0,
                pred_ser!=0)
            if bx.any():
                log.info('replacing %i/%i zero mismatches w/ null'%(bx.sum(), len(bx)))
                res1.loc[bx] = np.nan
                
            #=======================================================================
            # wrap
            #=======================================================================
            log.info('finished w/ mean bias = %.2f (%i/%i nulls)'%(
                res1.dropna().mean(), res1.isna().sum(), len(res1)))
            
            """
            bx = res1==np.inf
            
            view(df.join(res1).loc[bx, :])
            res1[bx]
            pred_ser.to_frame().join(true_ser).loc[bx, :]
            """
            
            return res1
    
    def get_meanError(self,
                      dkey='meanError',
                      logger=None
                      ):
        assert dkey=='meanError'
 

        df = self.df_raw.copy()
        
        return (df['pred'] - df['true']).sum()/len(df)
        
    def get_meanErrorAbs(self,
                       dkey='meanErrorAbs',
                      logger=None
                      ):
 
        assert dkey=='meanErrorAbs'
        df = self.df_raw.copy()
        
        return (df['pred'] - df['true']).abs().sum()/len(df)
    
    def get_RMSE(self,
                 dkey='RMSE',
                 logger=None):
        assert dkey=='RMSE'
        df = self.df_raw.copy()
        
        
        return math.sqrt(np.square(df['pred'] - df['true']).mean())
    
    
    def get_all(self, #load all the stats in the retrieve handles 
                dkeys_l = None,
                logger=None):
        
        if dkeys_l is None:
            dkeys_l = self.data_retrieve_hndls.keys()
        
        
        res_d = dict()
        for dkey in dkeys_l:
            res_d[dkey] = self.retrieve(dkey, logger=logger)
            
        return res_d
    
    def get_pearson(self,
                    dkey='pearson',
                    logger=None):
        assert dkey=='pearson'
        df = self.df_raw.copy()
        pearson, pval = scipy.stats.pearsonr(df['true'], df['pred'])
        return pearson
    
    def get_confusion(self,
                      dkey='confusion',
                     wetdry=False,
                     normed=None, #normalize confusion values by total count
                     logger=None):
        """get a confusion matrix with nice labels
        
        Returns
        -------------
        pd.DataFrame
            classic confusion matrix
            
        pd.DataFrame
            unstacked confusion matrix
            
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_confusion')
        assert dkey=='confusion'
        if normed is None: normed=self.normed
        df_raw = self.df_raw.copy()
        
        #=======================================================================
        # prep data
        #=======================================================================
        if wetdry:
            assert np.array([['float' in e for e in [d.name for d in df_raw.dtypes]]]).all()
            
            df1 = pd.DataFrame('dry', index=df_raw.index, columns=df_raw.columns)
            
            df1[df_raw>0.0] = 'wet'
            
            labels = ['wet', 'dry']
            
        else:
            raise IOError('not impelemented')
            df1 = df_raw.copy()
            
            labels=['pred', 'true']
            
 
        #build matrix
        cm_ar = confusion_matrix(df1['true'], df1['pred'], labels=labels)
        
        cm_df = pd.DataFrame(cm_ar, index=labels, columns=labels)
        
        #=======================================================================
        # normalize
        #=======================================================================
        if normed:
            cm_df = cm_df/len(df_raw)
        
        #convert and label
        
        cm_df2 = cm_df.unstack().rename('counts').to_frame()
        
        cm_df2.index.set_names(['true', 'pred'], inplace=True)
        
        cm_df2['codes'] = ['TP', 'FP', 'FN', 'TN']
        
        cm_df2 = cm_df2.set_index('codes', append=True)
        
        return cm_df, cm_df2.swaplevel(i=0, j=2)
    
    def get_stats(self, #get baskc stats on one series
                  ser=None,
                  logger=None,
                  dkey='stats',
                  stats_l = ['min', 'mean', 'max'],
                  ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('get_stats')
        assert dkey=='stats'
        if ser is None: ser = self.pred_ser
        
        return {stat:getattr(ser, stat)() for stat in stats_l}
        
    
    def check_match(self,
                    pred_ser=None,
                    true_ser=None,
                    ):
        if pred_ser is None:
            pred_ser=self.pred_ser
        if true_ser is None:
            true_ser=self.true_ser
            
        assert isinstance(pred_ser, pd.Series)
        assert isinstance(true_ser, pd.Series)
        
        assert_index_equal(pred_ser.index, true_ser.index)
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args,**kwargs):
        for k in copy.copy(list(self.__dict__.keys())):
            del self.__dict__[k]
 

def get_samp_errs(gdf_raw, log=None, ):
    """calc errors between pred and true"""
 
    
    #=======================================================================
    # clean
    #=======================================================================
    gdf = gdf_raw.drop('geometry', axis=1)  # .dropna(how='any', subset=['true'])
    
    assert gdf.notna().all().all()
    
    #=======================================================================
    # calc
    #=======================================================================
    
    with ErrorCalcs(pred_ser=gdf['pred'], true_ser=gdf['true'], logger=log) as wrkr:
        err_d = wrkr.get_all(
            dkeys_l=['bias', 'meanError', 'meanErrorAbs', 'RMSE', 'pearson'])
        
        # get confusion
        _, cm_dx = wrkr.get_confusion(wetdry=True, normed=False)            
        err_d.update(cm_dx.droplevel([1, 2])['counts'].to_dict())
        
    return err_d

def get_confusion_cat(true_arB, pred_arB, 
                      confusion_codes={'TP':'TP', 'TN':'TN', 'FP':'FP', 'FN':'FN'},
                      ):
    """compute the confusion code for each element
    
    Parameters
    -----------
    confusion_codes: dict
        optional mapping for naming the 4 confusion categories
    """
    #start with dummy
    res_ar = np.full(true_arB.shape, np.nan)
    #true positives
    res_ar = np.where(
        np.logical_and(true_arB, pred_arB), 
        confusion_codes['TP'], res_ar)
    #true negatives
    res_ar = np.where(
        np.logical_and(np.invert(true_arB), np.invert(pred_arB)), 
        confusion_codes['TN'], res_ar)
    #false positives
    res_ar = np.where(
        np.logical_and(np.invert(true_arB), pred_arB), 
        confusion_codes['FP'], res_ar)
    #false negatives
    res_ar = np.where(
        np.logical_and(true_arB, np.invert(pred_arB)), 
        confusion_codes['FN'], res_ar)
    
    
    return res_ar 

class ValidateMask(object):
    """compute validation metrics for a inundation mask
    
    
    WARNING: use context management for multiple runs
    """ 
    
    confusion_ser=None
    
 
 
    
    def __init__(self,
                 confu_ar=None,
 
                 logger=None,
                  ):
        """
        
        Pars
        ------------
 
            
        true_inun_fp: str
            filepath to inundation (rlay or poly) to compare extents against
            
        pred_inun_fp: str
            filepath to predicted/modelled inundation (rlay) to evaluate
        """
        self.confu_ar=confu_ar
 
        #=======================================================================
        # pre init
        #=======================================================================
        if logger is None:
            logger = get_log_stream(level=logging.DEBUG)
            
        self.logger=logger
        
 
 
 
    #===========================================================================
    # grid inundation metrics------
    #===========================================================================
    def get_hitRate(self, **kwargs):
        """proportion of wet benchmark data that was replicated by the model"""
        # log, true_ar, pred_ar = self._func_setup_local('hitRate', **kwargs)
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['TP'] / (cf_ser['TP'] + cf_ser['FN'])
    
    def get_falseAlarms(self, **kwargs):
        # log, true_ar, pred_ar = self._func_setup_local('hitRate', **kwargs)
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['FP'] / (cf_ser['TP'] + cf_ser['FP'])
    
    def get_criticalSuccessIndex(self, **kwargs):
        """critical success index. accounts for both overprediction and underprediction"""
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['TP'] / (cf_ser['TP'] + cf_ser['FP'] + cf_ser['FN'])
    
    def get_errorBias(self, **kwargs):
        """indicates whether the model has a tendency toward overprediction or underprediction"""
        
        cf_ser = self._confusion(**kwargs)
        return cf_ser['FP'] / cf_ser['FN']
    
    def get_inundation_all(self, **kwargs):
        """convenience for getting all the inundation metrics
        NOT using notation from Wing (2017)
        """
        
        d = {
            'hitRate':self.get_hitRate(**kwargs),
            'falseAlarms':self.get_falseAlarms(**kwargs),
            'criticalSuccessIndex':self.get_criticalSuccessIndex(**kwargs),
            'errorBias':self.get_errorBias(**kwargs),
            }
        
        # add confusion codes
        d.update(self.confusion_ser.to_dict())
        
        #=======================================================================
        # #checks
        #=======================================================================
        assert set(d.keys()).symmetric_difference(
            ['hitRate', 'falseAlarms', 'criticalSuccessIndex', 'errorBias', 'TN', 'FP', 'FN', 'TP']
            ) == set()
            
        assert pd.Series(d).notna().all(), d
        
        self.logger.debug('computed all inundation metrics:\n    %s' % d)
        return d

    def get_confusion_grid(self,
                           confusion_codes=None, **kwargs):
        """generate confusion grid
        
        Parameters
        ----------
        confusion_codes: dict
            integer codes for confusion labels
        """
        #(true=wet=nonnull)
        log, true_arB, pred_arB = self._func_setup_local('confuGrid', **kwargs)
        
        if confusion_codes is None: confusion_codes = self.confusion_codes
 
        #build a confusion map (using the int codes)
        res_ar = get_confusion_cat(true_arB, pred_arB, confusion_codes=confusion_codes)
        
        #=======================================================================
        # check
        #=======================================================================
        if __debug__:
            """compare against our aspatial confusion generator"""
            cf_ser = self._confusion(true_mar=true_arB, pred_mar=pred_arB, **kwargs)
            
            # build a frame with the codes
            df1 = pd.Series(res_ar.ravel(), name='grid_counts').value_counts().to_frame().reset_index()            
            df1['index'] = df1['index'].astype(int) 
            df2 = df1.join(pd.Series({v:k for k, v in confusion_codes.items()}, name='codes'), on='index'
                           ).set_index('index')
                           
            # join the values from sklearn calcs
            df3 = df2.join(cf_ser.rename('sklearn_counts').reset_index().rename(columns={'index':'codes'}).set_index('codes'),
                     on='codes')
            
            #compare
            if not df3['grid_counts'].sum()==df3['sklearn_counts'].sum():
                raise AssertionError('mismatch in confusion counts')
            
            compare_bx = df3['grid_counts'] == df3['sklearn_counts']
            if not compare_bx.all():
                raise AssertionError('confusion count mismatch\n    %s' % compare_bx.to_dict())
            
        log.info('finished on %s' % str(res_ar.shape))
        
        return res_ar
 
    #===========================================================================
    # private helpers------
    #===========================================================================
    def _confusion(self, **kwargs):
        """retrieve or construct the wet/dry confusion series"""
        if self.confusion_ser is None:
 
            cser = pd.Series(self.confu_ar.ravel()).value_counts()
            
            #re-label index 
            cser.index = cser.index.map(lambda x: {v:k for k,v in confusion_codes.items()}[x])
            
            #add empties
            for k,v in confusion_codes.items():
                if not k in cser.index:
                    cser[k] = 0

            self.confusion_ser = cser.copy()
            
        return self.confusion_ser.copy()

 
        
        
    def __enter__(self):
        return self
    
    def __exit__(self,  *args,**kwargs):
        self.confusion_ser=None
        pass
 
 
        
   