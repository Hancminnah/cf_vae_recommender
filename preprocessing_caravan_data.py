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

# ========== Find best algorithm for recommender ========== #
# https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise.model_selection.validation import cross_validate
from surprise.model_selection import GridSearchCV

benchmark = []
# Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
    # Perform cross validation
    results = cross_validate(algorithm, caravan_data, measures=['RMSE'], cv=3, verbose=False)
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = pd.concat([tmp,pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm'])])
    benchmark.append(tmp)
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse') 

# Original
algo = KNNBaseline()
cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False)

# Using grid search cv doesn't quite gives the lowest rmse
params = {
    'bsl_options': {
        'method': ['als', 'sgd'],
        'reg': [1, 2],
    },
    'k': [2, 3],
    'sim_options': {
        'name': ['msd', 'cosine'],
        'min_support': [1, 5],
        'user_based': [False],
    },
}
gs = GridSearchCV(KNNBaseline, param_grid=params, measures=["rmse", "mae"], cv=3, n_jobs=-1)
gs.fit(caravan_data)
print(gs.best_score["rmse"])
print(gs.best_params["rmse"])

# Heck just use original
trainset, testset = train_test_split(caravan_data, test_size=0.25)
algo = KNNBaseline()
predictions = algo.fit(trainset).test(testset)
accuracy.rmse(predictions)

def get_Iu(uid):
    """ return the number of items rated by given user
    args: 
      uid: the id of the user
    returns: 
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """ return number of users that have rated given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0
    
df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.iid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)
best_predictions = df.sort_values(by='err')[:10]
worst_predictions = df.sort_values(by='err')[-10:]
