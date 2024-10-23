import logging
import os


def log_print(status: str, content:str)-> None:
    assert  status == 'INFO' or status == 'WARNING' or status == 'CRITICAL' or status == 'ERROR',\
    log_print("ERROR", "Status must in ['INFO','WARNING','CRITICAL','ERROR'], "
                       "current status: {0}".format(status))
    if status == 'INFO':
        print("\033[0;32;1mINFO:\033[0m", content)
    elif status == 'WARNING':
        print("\033[0;33;1mWARNING:\033[0m", content)
    elif status == 'ERROR':
        print("\033[0;31;1mERROR:\033[0m", content)
    elif status == 'CRITICAL':
        print("\033[0;35;1mCRITICAL:\033[0m", content)
    return

# def test_log():
#     # 配置logging
#     logging.basicConfig(filename='example.log',
#                         level=logging.DEBUG,  # 设置日志级别
#                         format='%(asctime)s %(levelname)s: %(message)s',
#                         datefmt='%Y-%m-%d %H:%M'
#                         )  # 日志格式
#     # 写入日志
#     logging.debug('这是一个调试信息')
#     logging.info('这是一个普通信息')
#     logging.warning('这是一个警告信息')
#     logging.error('这是一个错误信息')
#     logging.critical('这是一个严重错误信息')

def write2log(log_file_path, log_status, content):
    # Set Log file
    logging.basicConfig(filename=log_file_path,
                        level=logging.DEBUG,  # 设置日志级别
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M'
                        )  # 日志格式
    if log_status == 'INFO':
        logging.info(content)
    elif log_status == 'WARNING':
        logging.warning(content)


def make_model_saving_dir(model_config, hyper_para_config, model_para_save_path):
    dir_name = None
    if model_config['model_name'] == 'dual_MBConv_VAE':
        dir_name = "{0} in_c={1} num_c={2} c_list={3} rec_w={4} kl_w={5} con_w={6} lr={7}".format(
            model_config['model_name'],
            model_config['in_channel'],
            model_config['num_class'],
            model_config['channel_list'],
            hyper_para_config['recon_weight'],
            hyper_para_config['kl_weight'],
            hyper_para_config['contras_weight'],
            hyper_para_config['lr']
        )
    else:
        dir_name = "{0} in_c={1} num_c={2} c_list={3} lr={4}".format(
            model_config['model_name'],
            model_config['in_channel'],
            model_config['num_class'],
            model_config['channel_list'],
            hyper_para_config['lr']
        )
    assert dir_name is not None, log_print("ERROR", "Model saving direction str is None!!!")
    if not os.path.exists(os.path.join(model_para_save_path, dir_name)):
        os.makedirs(os.path.join(model_para_save_path, dir_name))
    log_print("INFO", "Direction {0} is made!!!".format(dir_name))
    return os.path.join(model_para_save_path, dir_name)


