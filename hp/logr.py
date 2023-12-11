'''
Created on Dec. 11, 2023

@author: cef
'''

import os, logging, pprint, sys 

import logging.config

from definitions import wrk_dir
from parameters import logcfg_file

#===============================================================================
# loggers-----------
#===============================================================================
log_format_str = '%(levelname)s.%(name)s.%(asctime)s:  %(message)s'
def init_root_logger(
 
        log_dir = wrk_dir,
        ):
    logger = logging.getLogger() #get the root logger
    
    logging.config.fileConfig(logcfg_file,
                              defaults={'logdir':str(log_dir).replace('\\','/')},
                              disable_existing_loggers=True,
                              ) #load the configuration file 
    
    logger.info(f'root logger initiated and configured from file: {logcfg_file}\n    logdir={log_dir}')
    
    return logger

def get_new_file_logger(
        name='r',
        level=logging.DEBUG,
        fp=None, #file location to log to
        logger=None,
        ):
    
    #===========================================================================
    # configure the logger
    #===========================================================================
    if logger is None:
        logger = logging.getLogger(name)
        
    logger.setLevel(level)
    
    #===========================================================================
    # configure the handler
    #===========================================================================
    assert fp.endswith('.log')
    
    formatter = logging.Formatter(log_format_str)        
    handler = logging.FileHandler(fp, mode='w') #Create a file handler at the passed filename 
    handler.setFormatter(formatter) #attach teh formater object
    handler.setLevel(level) #set the level of the handler
    
    logger.addHandler(handler) #attach teh handler to the logger
    
    logger.info('built new file logger  here \n    %s'%(fp))
    
    return logger

def init_log(
 
        log_dir=wrk_dir,
        **kwargs):
    """wrapper to setup the root loger and create a file logger"""
    
    root_logger = init_root_logger(log_dir=log_dir) 
    
    #set up the file logger
    return get_new_file_logger(**kwargs)


def get_log_stream(name=None, level=None):
    """get a logger with stream handler"""
    if name is None: name=str(os.getpid())
    if level is None:
        if __debug__:
            level=logging.DEBUG
        else:
            level=logging.INFO
    
    logger = logging.getLogger(name)
    
    #see if it has been configured
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(
            stream=sys.stdout, #send to stdout (supports colors)
            ) #Create a file handler at the passed filename 
        formatter = logging.Formatter(log_format_str) 
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger