import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy

# ===== Example 1: Load Built-In Dataset ===== #
# Easy tutorial: https://reintech.io/blog/how-to-create-a-recommendation-system-with-surprise
data = Dataset.load_builtin('ml-100k')
all_raw_ratings = data.__dict__['raw_ratings']
id_list = [i[0] for i in all_raw_ratings] # 943 users
movie_list = [i[1] for i in all_raw_ratings] #1682 items
rating_list = [i[2] for i in all_raw_ratings]# 5 ratings
unique_list = [i[3] for i in all_raw_ratings]# timestamps
trainset, testset = train_test_split(data, test_size=0.25)

algo=SVD()
algo.fit(trainset)
predictions = algo.test(testset)
mse = accuracy.mse(predictions)
rmse = accuracy.rmse(predictions)



# ===== Example 2: Using Custom Dataset (Caravan insurance) ===== #
# Dataset can be obtained from: https://www.kaggle.com/datasets/uciml/caravan-insurance-challenge
# How to load a custom dataset: https://surprise.readthedocs.io/en/v1.0.4/getting_started.html#:~:text=To%20load%20a%20dataset%20from%20a%20pandas%20dataframe%2C%20you%20will,the%20ratings%20in%20this%20order.
file_name = "/home/ubuntu/research-ops/cf_vae_recommender/data/caravan-insurance-challenge.csv"
df = pd.read_csv(file_name)
policy_list = ["PWAPART","PWABEDR","PWALAND","PPERSAUT","PBESAUT","PMOTSCO","PVRAAUT","PAANHANG","PTRACTOR","PWERKT","PBROM","PLEVEN","PPERSONG","PGEZONG","PWAOREG","PBRAND","PZEILPL","PPLEZIER","PFIETS","PINBOED","PBYSTAND"]
policy_list_values = dict(zip(policy_list, [str(x+1) for x in list(range(len(policy_list)))]))
nb_policy = ["AWAPART","AWABEDR","AWALAND","APERSAUT","ABESAUT","AMOTSCO","AVRAAUT","AAANHANG","ATRACTOR","AWERKT","ABROM","ALEVEN","APERSONG","AGEZONG","AWAOREG","ABRAND","AZEILPL","APLEZIER","AFIETS","AINBOED","ABYSTAND","CARAVAN"]
reader = Reader(rating_scale = (1,9))
# convert data into the format of user_ids, item_ids, ratings columns
new_df = pd.DataFrame(columns=['user_id','item_id','rating'])
users_overall = dict()
users_included = dict()
users_removed = dict()
for user in range(len(df)):
    i_list = df.iloc[user][policy_list]
    i_list = i_list[i_list>0]
    users_overall[str(user+1)]=user
    if len(i_list)== 0:
        users_removed[str(user+1)] = user
        continue
    else:
        users_included[str(user+1)] = user
    for ii, vv in zip(i_list.index,i_list.values):
        app_data = pd.DataFrame({'user_id':str(user+1),'item_id':policy_list_values[ii],'rating':[vv]})
        new_df = pd.concat([new_df,app_data],ignore_index=True)

caravan_data = Dataset.load_from_df(new_df[['user_id','item_id','rating']], reader)
trainset, testset = train_test_split(caravan_data, test_size=0.25)
algo=SVD()
algo.fit(trainset)
predictions = algo.test(testset)
mse = accuracy.mse(predictions)
rmse = accuracy.rmse(predictions)
# ========== #
