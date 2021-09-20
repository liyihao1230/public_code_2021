#!/usr/bin/env python
# encoding: utf-8
"""
@AUTHOR: cuizhengliang
@LICENCE: (C)Copyright 2013-2020, JNBY+ Corporation Limited
@CONTACT: czlwork@qq.com
@FILE: loader.py
@TIME: 2020/12/17 17:26
@DESC:
"""
import os
import logging

SPARK_HOME = '/opt/spark-2.4.7-bin-hadoop2.7'

# ++++ bigdata server ++++
PYTHON_PATH = "/root/anaconda3/envs/rec/bin/python3.7"
PYSPARK_LIB_PATH = "/root/anaconda3/envs/rec/lib/python3.7/site-packages"

# ++++ GPU server ++++
# PYTHON_PATH = "/home/administrator/anaconda3/envs/rec_1.1/bin/python3.7"
# PYSPARK_LIB_PATH = "/home/administrator/anaconda3/envs/rec_1.1/lib/python3.7/site-packages"

os.environ["PYSPARK_PYTHON"] = PYTHON_PATH
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_PATH

from pyspark import SparkConf
from pyspark.sql import SparkSession


conf = SparkConf()
conf.set("spark.pyspark.python", PYTHON_PATH) \
    .set("spark.pyspark.driver.python", PYTHON_PATH) \
    .set("spark.executorEnv.PYSPARK_PYTHON", PYTHON_PATH) \
    .set("spark.executorEnv.PYSPARK_DRIVER_PYTHON", PYTHON_PATH) \
    .set("spark.yarn.appMasterEnv.PYSPARK_PYTHON", PYTHON_PATH) \
    .set("spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON", PYTHON_PATH) \
    .set("spark.sql.execution.arrow.enabled", "true") \
    .set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation", "true") \
    .set("spark.sql.broadcastTimeout", "36000") \
    .set("spark.debug.maxToStringFields", "100") \
    .set("spark.driver.memory", "10g") \
    .set("spark.executor.memory", "8g") \
    .set("spark.driver.maxResultSize", "10g") \
    .set("spark.scheduler.listenerbus.eventqueue.capacity", "200000")

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class DataLoader(object):
    def __init__(self):
        self.spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

    def load_data(self, table, cols, condition=None):
        assert type(table) == str and type(cols) == list
        cols = ",".join(cols)
        if condition:
            return self.spark.sql("select {} from {} where {}".format(cols, table, condition))
        else:
            return self.spark.sql("select {} from {}".format(cols, table))

