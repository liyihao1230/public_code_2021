import os
import sys
import time
import datetime
import glob
import pandas as pd
import io
from io import BytesIO

import paramiko
from tools.ssh_connector import SSHConnection_pwd, SSHConnection_pkey

if __name__ == '__main__':

    host = '192.168.200.111'
    port = 22
    username = 'root'
    pkey = 'configs/bigdata.pem'
    # 创建类
    ssh = SSHConnection_pkey(host='192.168.200.111', port=22, username='root', pkey='configs/bigdata.pem')
    # 开启连接
    ssh.connect()
    # 操作 upload, download, cmd
    ssh.upload('/home/administrator/Dataset/file_sync/configs/bigdata.pem', '/root/lyh/bigdata.pem')
    res = ssh.cmd('pwd')
    print(res)
    # 关闭连接
    ssh.close()

    pass