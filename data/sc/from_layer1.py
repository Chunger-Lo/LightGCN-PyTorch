import numpy as np
import pandas as pd
from mlaas_tools2.db_tool import DatabaseConnections
from mlaas_tools2.config_info import ConfigPass
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DateType, StructType, StructField, StringType, LongType, DoubleType
from pyspark.sql.functions import create_map, lit
from pyspark.sql.functions import expr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime,  timedelta
import os
import argparse

# # Spark 連線
default = {"spark.driver.memory":'16g',
           "fs.s3a.access.key": "smartchannel",
           "fs.s3a.secret.key": "smartchannel",
           "fs.s3a.endpoint": "http://10.240.205.23:9000",
           "fs.s3a.connection.ssl.enabled": False,
           "fs.s3a.path.style.access": True,} 

spark = SparkSession.builder.config(
    conf = (SparkConf().setAppName("T").setAll(default.items()))).getOrCreate()
# # 從feature group 以及 layer1 讀資料
#train_start_date = '20211217'
#train_end_date = '20211226'
#test_date = '20211227'
# test_date = '20220131'
parser = argparse.ArgumentParser()
parser.add_argument("-train_start_date",help="test date",type=str)
parser.add_argument("-train_end_date",help="test date",type=str)
parser.add_argument("-test_date",help="test date",type=str)
args = parser.parse_args()
# ## 1. item-subtag (sc_item)
item_sdf = (
    spark.read.parquet(
        "s3a://df-smart-channel/recsys-dataset/beta_v2/layer1/sc_item"
    ).where(col('service')=='smart_channel').
    where(
        (col("date")>=args.train_start_date) & (col("date")<=args.test_date)
    )
    # .select(['item_id', 'date','subtag_list'])
)
item_subtag_exploded = item_sdf.select(item_sdf.item_id, item_sdf.date,  F.explode(item_sdf.subtag_list))

item_subtag_details = item_subtag_exploded.select(['item_id', 'date', 'col.subtag' ,'col.subtag_chinese_desc','col.subtag_eng_desc']).filter(item_subtag_exploded['col.subtag_chinese_desc'].startswith('03_'))
item_subtag_net = item_subtag_details.select(['item_id', 'date', 'subtag_eng_desc'])

item_subtag_net_train = item_subtag_net.filter(item_subtag_net.date == args.train_end_date)
item_subtag_net_test = item_subtag_net.filter(item_subtag_net.date == args.test_date)

item_subtag_df_train = item_subtag_net_train.toPandas()
item_subtag_df_test = item_subtag_net_train.toPandas()

def export_file(df, test_date, file_name):
    dir_name = f'date={test_date}'
    dir_path = os.path.join(f'..//..//..//data//preprocessed//{dir_name}')
    # print(dir_path)
    if not os.path.isdir(dir_path):
    #if not os.path.isdir(dir_path):
        # print( os.path.isdir(dir_path))
        os.mkdir(dir_name)
        #os.makedirs(dir_name , exist_ok=False)
    file_path = os.path.join(dir_path, file_name)
    print(file_path)
    df.to_csv(file_path, index = False)

export_file(item_subtag_df_test, args.test_date, 'item_subtag.csv')


# ### 2.user-item (context)

context_sdf = (
    spark.read.parquet(
        "s3a://df-smart-channel/recsys-dataset/beta_v2/layer1/context"
    ).where(
        (col("date")>=args.train_start_date) & (col("date")<=args.test_date)
    )
)
# 去除沒有點擊紀錄的
clean_context_sdf = context_sdf.where(col('item_click_list').isNotNull())
print('去除沒有點擊紀錄後context的數目:', clean_context_sdf.count())
# 將item_click_list展開
clean_context_sdf = clean_context_sdf.select(clean_context_sdf.cust_no, 
                                             clean_context_sdf.date,
                                             F.explode(clean_context_sdf.item_click_list))
# 選出欄位
flatten_clean_context_sdf = clean_context_sdf.select(F.col('cust_no'), F.col('date'), F.col('col.*'))
print('展開item後context的數目:', flatten_clean_context_sdf.count())
# 選 df_smart_channel
flatten_clean_context_sdf = flatten_clean_context_sdf.where(col('hits_eventinfo_eventcategory')=='smart_channel')
print('選 df_smart_channel後context的數目:', flatten_clean_context_sdf.count())
#和item sdf合併
flatten_clean_context_sdf = flatten_clean_context_sdf.join(item_sdf.drop('click', 'show'), on=['item_id', 'date'], how='left')
# 過濾掉公版
# flatten_clean_context_sdf = flatten_clean_context_sdf.where(col('service')!='public_content')
print('過濾掉公版後context的數目:', flatten_clean_context_sdf.count())
flatten_clean_context_sdf = flatten_clean_context_sdf.select(F.col('item_id'), F.col('cust_no'), F.col('date'),
                               F.col('visitdatetime'), F.col('eventdatetime'), F.col('click'), F.col('show')
                              )
# 找出最先點擊或最先曝光(一天可能點同一個item多次)
flatten_clean_context_sdf = flatten_clean_context_sdf.withColumn("day_order", F.row_number().over(Window.partitionBy(['cust_no', 'date', 'item_id']).orderBy(flatten_clean_context_sdf['eventdatetime'])))
# 計算該顧客在當天點同一個item的點擊和曝光總數
cust_click_bydate = flatten_clean_context_sdf.groupby(['cust_no', 'date', 'item_id']).sum('click').withColumnRenamed('sum(click)', 'click')
cust_show_bydate = flatten_clean_context_sdf.groupby(['cust_no', 'date', 'item_id']).sum('show').withColumnRenamed('sum(show)', 'show')
# 選最先點擊的item
flatten_clean_context_sdf = flatten_clean_context_sdf.where(col('day_order')==1).drop('click', 'show', 'day_order')
flatten_clean_context_sdf = flatten_clean_context_sdf.join(cust_click_bydate, on=['cust_no', 'date', 'item_id'], how='left')
flatten_clean_context_sdf = flatten_clean_context_sdf.join(cust_show_bydate, on=['cust_no', 'date', 'item_id'], how='left')
# 計算點擊排名。挑出有點擊的人做排名 15674人有點擊 rank最多為2
click_sdf = flatten_clean_context_sdf.where(col('click')>0)
click_sdf = click_sdf.withColumn("rank", F.row_number().over(Window.partitionBy(['cust_no', 'date']).orderBy(click_sdf['eventdatetime'])))
flatten_clean_context_sdf = flatten_clean_context_sdf.join(click_sdf.select('click', 'cust_no', 'date', 'item_id', 'rank'), on=['click', 'cust_no', 'date', 'item_id'], how='left')
print('Total Context number:', flatten_clean_context_sdf.count())


context_sdf_train = flatten_clean_context_sdf.filter(col('date') <= args.train_end_date)
context_sdf_test = flatten_clean_context_sdf.filter(col('date') == args.test_date)


uniqueUsersObserved = context_sdf_train.select('cust_no').distinct().collect()
unique_users_list = [each_user.__getitem__('cust_no') for each_user in uniqueUsersObserved]

# unique_users_list
## filter unobserved in test
context_sdf_test = context_sdf_test.filter(context_sdf_test.cust_no.isin(unique_users_list))
context_df_train = context_sdf_train.toPandas()
context_df_test = context_sdf_test.toPandas()

export_file(context_df_train, args.test_date, 'context_train.csv')
export_file(context_df_test, args.test_date, 'context_test.csv')


# ## 3. user-subtag (user)
user_sdf = (
    spark.read.parquet(
        "s3a://df-smart-channel/recsys-dataset/beta_v2/layer1/user"
    ).where(
        (col("date")>=args.train_end_date) & (col("date")<=args.test_date)
        #col("date") == train_end_date | col("date") == test_date
    )
)
user_sdf = user_sdf.drop('click', 'show')
user_sdf = user_sdf.select([
    'cust_no', 'date', 'mobile_login_90', 'dd_my', 'dd_md', 'onlymd_ind', 'onlycc_ind', 'efingo_card_ind',  'cl_cpa_amt', 'fc_ind'
])
# .filter(user_sdf.date.isin(uniqueUsersInContext))
print('User Sdf數目:', user_sdf.count())

unique_users_list = [each_user.__getitem__('cust_no') for each_user in uniqueUsersObserved]
user_sdf_sub = user_sdf.filter(user_sdf.cust_no.isin(unique_users_list))
# user_df_train = user_sdf.filter(user_sdf.date == train_end_date).toPandas()
user_df_test = user_sdf_sub.filter(user_sdf_sub.date == args.test_date).toPandas()

export_file(user_df_test, args.test_date, 'user_subtag.csv')

