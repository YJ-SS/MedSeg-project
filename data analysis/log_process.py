import re
from typing import Union, List, Any

from matplotlib import pyplot as plt, ticker


def extract_log_by_time_stamp(log_data, logstamp)-> Union[list[Any], list[str]]:
    '''
    Use logstamp to extract target logs
    :param log_data:
    :param logstamp:
    :return:
    '''
    pattern = re.compile(rf".*{logstamp}.*$", re.MULTILINE)
    matches = pattern.findall(log_data)
    return matches

def extract_fields(log, fields)->dict[str, float]:
    '''
    Extract field's value from log
    :param log: "epoch=2 total_loss=0.32"
    :param fields: ['epoch','total_loss']
    :return: {'epoch':2, 'total_loss':0.32}
    '''
    data_dict = {}
    for field in fields:
        pattern = rf"{field}=(\S+)"  # 正则表达式，匹配字段名和值
        match = re.search(pattern, log)
        if match:
            data_dict[field] = match.group(1)
    return data_dict



def extract_data_from_log(log_path, logstamp, fields)->dict[str,list[float]]:
    '''
    Get information during training, return as a dict
    :param log_path:
    :param logstamp:
    :param fields:
    :return:{'epoch':[0,1,2,3,4,5], 'total_loss':[1.2,1.1,0.8,0.5,0.3]}
    '''
    matches_data = None
    train_info_dict = {}
    for field in fields:
        train_info_dict[field] = []

    with open(log_path, 'r') as f:
        log_data = f.read()
        matches_data = extract_log_by_time_stamp(log_data, logstamp)
        f.close()

    for line in matches_data:
        data_dict = extract_fields(line, fields)
        for field in fields:
            train_info_dict[field].append(float(data_dict[field]))
    return train_info_dict


def draw_line_graph(x, y_list, label_list, color_list, title, loc='upper right', ylabel='LOSS', graph_name='line_graph'):
    '''

    :param x:
    :param y_list:
    :param label_list:
    :param color_list:
    :param title:
    :param loc:
    :param ylabel:
    :param graph_name:
    :return:
    '''
    plt.figure(figsize=(10, 5), dpi=80)
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))

    for i in range(len(y_list)):
        plt.plot(x, y_list[i], color_list[i], alpha=0.7, linewidth=1.3, label=label_list[i])
    plt.rcParams.update({'font.size': 10})
    plt.legend(loc=loc)
    plt.xlabel('EPOCH', fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(title)
    plt.tight_layout()
    # plt.gca().invert_yaxis()
    plt.savefig(graph_name + ".png")
    plt.show()