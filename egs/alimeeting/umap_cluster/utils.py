#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn
import os
def validate_path(dir_name):
    """ Create the directory if it doesn't exist
    :param dir_name
    :return: None
    """
    dir_name = os.path.dirname(dir_name)  # get the path
    if not os.path.exists(dir_name) and (dir_name != ''):
        os.makedirs(dir_name)


def read_scp(scp_file):
    """read scp file (also support PIPE format)

    Args:
        scp_file (str): path to the scp file

    Returns:
        list: key_value_list
    """
    key_value_list = []
    with open(scp_file, "r", encoding='utf8') as fin:
        for line in fin:
            tokens = line.strip().split()
            key = tokens[0]
            value = " ".join(tokens[1:])
            key_value_list.append((key, value))
    return key_value_list

