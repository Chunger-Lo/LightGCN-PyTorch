import numpy as np
import pandas as pd
import math
import os
from sklearn.preprocessing import LabelEncoder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-test_date",
                    help="test date",
                    default = '20211226',
                    type=str)
args = parser.parse_args()

def export_file(date, pairs_dict, file_name):
    dir_name = f'date={date}'
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    file_path = os.path.join(dir_name, file_name)
    file = open(file_path,"w")
    print(f'write to file path: {file}')
    counts = 0
    for key,value in pairs_dict.items():
        file.write('%s' % key)
        # for item in value:
        #     file.write('%s ' % item)
        for i, item in enumerate(value):
            if math.isnan(item):
                continue
            else:
                if i == (len(value)-1):
                    file.write(' %s' % int(item))
                else:
                    file.write(' %s' % int(item))
        file.write('\n')
        counts += 1
    print(counts)
    file.close()
context_train = pd.read_csv(f'/home/jovyan/df-smart-channel/graph/data/preprocessed/date={args.test_date}/context_train.csv')
context_test = pd.read_csv(f'/home/jovyan/df-smart-channel/graph/data/preprocessed/date={args.test_date}/context_test.csv')
user_subtag = pd.read_csv(f'/home/jovyan/df-smart-channel/graph/data/preprocessed/date={args.test_date}/user_subtag.csv')
item_subtag = pd.read_csv(f'/home/jovyan/df-smart-channel/graph/data/preprocessed/date={args.test_date}/item_subtag.csv')


le_user = LabelEncoder()
context_train.cust_no = le_user.fit_transform(context_train.cust_no)
context_test.cust_no = le_user.transform(context_test.cust_no)
user_subtag['cust_no'] = le_user.transform(user_subtag['cust_no'])

le_item = LabelEncoder()
context_train.item_id = le_item.fit_transform(context_train.item_id)
context_test.item_id = le_item.transform(context_test.item_id)
item_subtag.item_id = le_item.transform(item_subtag.item_id)


# ## get item-subtag
mapping_df = pd.read_excel('subtag_map.xlsx', sheet_name = 'mapping')
available_items = mapping_df.subtag_03.unique()
item_subtag.subtag_eng_desc = item_subtag.subtag_eng_desc.str[3:]
item_subtag = item_subtag[item_subtag.subtag_eng_desc.isin(available_items)]

item_subtags = pd.merge(item_subtag, mapping_df, how = 'left', left_on = 'subtag_eng_desc', right_on = 'subtag_03', copy = False)

subtags_by_item = item_subtags.groupby('item_id')['code'].unique()
item_subtag_dict = dict(subtags_by_item.apply(list))

export_file(args.test_date, item_subtag_dict, 'item_subtag.txt')
# ### get user-subtag
# 行銀活躍用戶
mobile_active = user_subtag.mobile_login_90
# 有外幣帳戶
forex_digital_account = user_subtag.dd_my
# 存戶
account = user_subtag.dd_md
# 純存戶
account_only = user_subtag.onlymd_ind
# 純卡戶
credit_card_only = user_subtag.onlycc_ind
# 卡存戶有e.Fingo指定卡
card_pi_only_ubear = user_subtag.efingo_card_ind
# 有信貸者
personal_loan_account_cust = user_subtag.cl_cpa_amt.notna()
# 無理專顧客
no_fc_cust = user_subtag.fc_ind == 0

user_subtag = user_subtag.sort_values('cust_no')

user_subtag_dict = {}
for i, ids in enumerate(user_subtag.cust_no):
    subtags_list = []
    if mobile_active[i]: subtags_list.append(0);
    if forex_digital_account[i]: subtags_list.append(1);
    if account[i]: subtags_list.append(2);
    if account_only[i]: subtags_list.append(3);        
    if credit_card_only[i]: subtags_list.append(4);
    if card_pi_only_ubear[i]: subtags_list.append(5);
    if personal_loan_account_cust[i]: subtags_list.append(6);
    if no_fc_cust[i]: subtags_list.append(7);
    user_subtag_dict[ids] = subtags_list

export_file(args.test_date, user_subtag_dict, 'user_subtag.txt')

# ### get user-item pairs
## version1: retain no click users
cust_item_pair_train = context_train.groupby(['cust_no', 'item_id'])['click'].sum().reset_index()
cust_item_pair_test = context_test.groupby(['cust_no', 'item_id'])['click'].sum().reset_index()
cust_item_pair_train = cust_item_pair_train[cust_item_pair_train.click > 0]
cust_item_pair_test = cust_item_pair_test[cust_item_pair_test.click > 0]
cust_no = context_train.cust_no.unique() ##observed
cust_no_test = context_test.cust_no.unique() 
init_df = pd.DataFrame({'cust_no': cust_no})
init_df_test = pd.DataFrame({'cust_no': cust_no_test})

combined_train = pd.merge(init_df, cust_item_pair_train, on = 'cust_no', how = 'left')[['cust_no', 'item_id']]
# [['cust_no', 'item_id', 'click']]
# combined_train.item_id = combined_train.item_id.astype(int)

combined_test = pd.merge(init_df_test, cust_item_pair_test, on = 'cust_no', how = 'left')[['cust_no', 'item_id']]
# combined_test.item_id = combined_test.item_id.replace(np.nan, -1)
# combined_test = combined_test[combined_test.item_id != -1]
# combined_test.item_id = combined_test.item_id.astype(int)

items_by_cust_train = combined_train.groupby('cust_no')['item_id'].unique()
items_by_cust_test = combined_test.groupby('cust_no')['item_id'].unique()

cust_items_dict_train = dict(items_by_cust_train.apply(list))
cust_items_dict_test = dict(items_by_cust_test.apply(list))
export_file(args.test_date, cust_items_dict_train, 'train_user_item_retain.txt')
export_file(args.test_date, cust_items_dict_test, 'test_user_item_retain.txt')




