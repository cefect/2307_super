import os, sys

src_dir = os.path.dirname(os.path.abspath(__file__))
src_name = os.path.basename(src_dir)


# logging configuration file
logcfg_file = os.path.join(src_dir, 'logger.conf')

confusion_codes={'TP':11, 'TN':12, 'FP':21, 'FN':22}

log_format_str = '%(levelname)s.%(name)s:  %(message)s'