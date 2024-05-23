
import os
import time
import logging
import colorlog
from config import PROJECT_ROOT

class Logger(object):
    def __init__(self, logger_name: str,
                 level=logging.DEBUG,
                 std_out: bool=True,
                 save2file: bool=False,
                 file_name: str=None)->None:
        super(Logger, self).__init__()
        if std_out is False and save2file is False:
            raise ValueError('[std_out, save2file] must one of them be True !')

        datefmt = '%Y-%m-%d %H:%M:%S'

        if std_out:
            std_logfmt = "[%(asctime)s.%(msecs)03d] [%(levelname)s]: %(log_color)s%(message)s"
            self.stdout_logger = logging.getLogger(name='{}_std'.format(logger_name))
            self.stdout_logger.setLevel(level)
            log_colors_config = {
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red'
            }

            formatter = colorlog.ColoredFormatter(
                fmt=std_logfmt,
                datefmt=datefmt,
                log_colors=log_colors_config,
            )
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            sh.setLevel(level)
            self.stdout_logger.addHandler(sh)

        if save2file:
            file_logfmt = "[%(asctime)s.%(msecs)03d] [%(levelname)s]: %(message)s"
            self.file_logger = logging.getLogger(name='{}_file'.format(logger_name))
            self.file_logger.setLevel(level)

            base_dir = PROJECT_ROOT + '/logs'
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            if file_name is not None:
                log_file = file_name
            else:
                log_file = base_dir + '/' + logger_name + '-' + str(time.strftime('%Y%m%d', time.localtime())) + '.log'
            fh = logging.FileHandler(filename=log_file, mode='a', encoding='utf-8')
            save_fmt = logging.Formatter(fmt=file_logfmt, datefmt=datefmt)
            fh.setFormatter(save_fmt)
            fh.setLevel(level)
            self.file_logger.addHandler(fh)

    def info(self, message: str, std_out: bool=True, save_to_file: bool=False)->None:
        if std_out:
            self.stdout_logger.info(msg=message)
        if save_to_file:
            self.file_logger.info(msg=message)

if __name__ == '__main__':
    logging = Logger('test', level=logging.INFO, save2file=True)
    logging.info('test log')
    logging.info('hello logging', save_to_file=True)

