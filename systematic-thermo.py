import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from nupack import *
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqUtils import GC, molecular_weight

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, \
                             AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import sklearn.gaussian_process as gp
from sklearn.neural_network import MLPRegressor

from itertools import product

from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from scipy.stats import pearsonr

### CONFIGURATION ###
CONFIG = {
    'path': 'datasets/',
    'remove_control': True,
    'remove_shared_sequence': True,
    'logscale': True,
    'use_median': True, #otherwise use mean
    'num_rna5': 36,
    'num_rna3': 36
}
THERMO_CONFIG = {
    'model_nostacking': False,
    'encode_non_numeric': True
}

### MODELS ###
regressors = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', KernelRidge()),
    ('Elastic Net Regression', ElasticNet()),
    ('Bayesian Regression', BayesianRidge()),
    ('k-Nearest Neighbors', KNeighborsRegressor()),
    ('Support Vector Machine', SVR()),
    ('Multi-layer Perceptrons (Neural Network)', MLPRegressor()),
    ('Ada Boost', AdaBoostRegressor()),
    ('Random Forest', RandomForestRegressor()),
    #('Gradient Boosting', GradientBoostingRegressor()),
    #('Histogram Gradient Boosting', HistGradientBoostingRegressor()),
    ('Extreme Gradient Boosting', XGBRegressor()),
    ('Light GBM', LGBMRegressor()),
    ('Category Boosting', CatBoostRegressor()),
    #('Stochastic Gradient Descent Regressor', SGDRegressor()),
]

def preprocess_data(raw_data, config=CONFIG):

    data = raw_data.copy()
    if config['remove_shared_sequence']:
        data['rna5'] = data['rna5'].apply(lambda x: x.replace('T', 'U')[577:-49]) #hardcoded position for the non-shared region of 5-egs
        data['rna3'] = data['rna3'].apply(lambda x: x.replace('T', 'U')[68:-732]) #hardcoded position for the non-shared region of 3-egs
    if config['use_median']:
        data['fluo'] = data[['fluo1', 'fluo2', 'fluo3']].median(axis=1)
    else:
        data['fluo'] = data[['fluo1', 'fluo2', 'fluo3']].mean(axis=1)
    if config['logscale']:
        data['fluo'] = np.log10(data['fluo'])
    if config['remove_control']:
        data['len5'] = data['rna5'].apply(lambda x: len(x))
        data = data[data['len5']>0].reset_index(drop=True) #remove control because controls do not have non-shared 5-egs
        data.drop('len5', axis=1, inplace=True)

    return data.drop(['fluo1', 'fluo2', 'fluo3'], axis=1)

def create_systematic_split(data, config=CONFIG):

    rna5_list = data['rna5'].unique().tolist()
    rna3_list = data['rna3'].unique().tolist()
    systematic_split = []
    # 4*4 cross validation
    kf = KFold(n_splits=4, shuffle=True)

    for k in list(product([x[1] for x in kf.split(np.arange(config['num_rna5']))], [x[1] for x in kf.split(np.arange(config['num_rna3']))])):
        
        out_rna5 = [x for i, x in enumerate(rna5_list) if i in k[0]]
        out_rna3 = [y for j, y in enumerate(rna3_list) if j in k[1]]
        data_a = data[(~data['rna5'].isin(out_rna5)) & (~data['rna3'].isin(out_rna3))]
        data_b = data[(~data['rna5'].isin(out_rna5)) & (data['rna3'].isin(out_rna3))]
        data_c = data[(data['rna5'].isin(out_rna5)) & (~data['rna3'].isin(out_rna3))]
        data_d = data[(data['rna5'].isin(out_rna5)) & (data['rna3'].isin(out_rna3))]
        systematic_split.append((data_a, data_b, data_c, data_d))
        
    return systematic_split

def generate_thermo_features(raw_data, config=THERMO_CONFIG):
    
    data = raw_data.copy()
    
    models = [Model(material='RNA', ensemble='stacking')]
    if config['model_nostacking']:
        models.append(Model(material='RNA', ensemble='nostacking'))
    
    data['rna5-rna3'] = data['rna5'] + data['rna3']
    data['len'] = data['rna5-rna3'].apply(lambda x: len(x))
    data['len5'] = data['rna5'].apply(lambda x: len(x))
    data['len3'] = data['rna3'].apply(lambda x: len(x))
    data['weight'] = [molecular_weight(x, 'RNA') for x in data['rna5-rna3']]
    data['weight5'] = [molecular_weight(x, 'RNA') for x in data['rna5']]
    data['weight3'] = [molecular_weight(x, 'RNA') for x in data['rna3']]
    data['gc'] = [GC(x) for x in data['rna5-rna3']]
    data['gc5'] = [GC(x) for x in data['rna5']]
    data['gc3'] = [GC(x) for x in data['rna3']]
    data['C'] = [Seq(x).count('C') for x in data['rna5-rna3']]
    data['C5'] = [Seq(x).count('C') for x in data['rna5']]
    data['C3'] = [Seq(x).count('C') for x in data['rna3']]
    data['G'] = [Seq(x).count('G') for x in data['rna5-rna3']]
    data['G5'] = [Seq(x).count('G') for x in data['rna5']]
    data['G3'] = [Seq(x).count('G') for x in data['rna3']]
    data['U'] = [Seq(x).count('U') for x in data['rna5-rna3']]
    data['U5'] = [Seq(x).count('U') for x in data['rna5']]
    data['U3'] = [Seq(x).count('U') for x in data['rna3']]
    data['A'] = [Seq(x).count('A') for x in data['rna5-rna3']]
    data['A5'] = [Seq(x).count('A') for x in data['rna5']]
    data['A3'] = [Seq(x).count('A') for x in data['rna3']]
    
    data['distance'] = [seq_distance(str1, str2)
                        for str1, str2 in zip(data['rna5'].tolist(), data['rna3'].tolist())]
    data['global_alg'] = [pairwise2.align.globalxx(str1, str2, score_only=True)
                          for str1, str2 in zip(data['rna5'].tolist(), data['rna3'].tolist())]
    data['local_alg'] = [pairwise2.align.localxx(str1, str2, score_only=True)
                         for str1, str2 in zip(data['rna5'].tolist(), data['rna3'].tolist())]
    
    #partition function
    for i, model in tqdm(enumerate(models)):
        arr = [pfunc(strands=[str1, str2], model=model) \
               for str1, str2 in zip(data['rna5'].tolist(), data['rna3'].tolist())]
        data['pfunc-0_{}'.format(i)] = ([x[0] for x in arr])
        data['pfunc-0_{}'.format(i)] = data['pfunc-0_{}'.format(i)].astype(float)
        data['pfunc-1_{}'.format(i)] = [x[1] for x in arr]
    
    #equilibrium base-pairing probabilities
    for i, model in tqdm(enumerate(models)):
        arr = [pairs(strands=[str1, str2], model=model) \
               for str1, str2 in zip(data['rna5'].tolist(), data['rna3'].tolist())]
        data['prob_matrix_{}'.format(i)] = [x.to_array().ravel() for x in arr]
    
    #minimum free energy
    for i, model in tqdm(enumerate(models)):
        arr = [mfe(strands=[str1, str2], model=model) \
               for str1, str2 in zip(data['rna5'].tolist(), data['rna3'].tolist())]
        data['mfe_energy_{}'.format(i)] = [x[0].energy for x in arr]
        data['mfe_stack_energy_{}'.format(i)] = [x[0].stack_energy for x in arr]
        data['mfe_structure_{}'.format(i)] = [str(x[0].structure) for x in arr]
        data['mfe_matrix_{}'.format(i)] = [x[0].structure.matrix().ravel() for x in arr]
        data['mfe_pairlist_{}'.format(i)] = [x[0].structure.pairlist() for x in arr]
        data['mfe_nicks_{}'.format(i)] = [x[0].structure.nicks() for x in arr]
        
    #structure free energy
    for i, model in tqdm(enumerate(models)):
        arr = [structure_energy(strands=[str1, str2], structure=structure, model=model) \
               for str1, str2, structure in zip(data['rna5'].tolist(), data['rna3'].tolist(), \
               data['mfe_structure_{}'.format(i)])]
        data['struct_dg_{}'.format(i)] = arr
    
    #equilibrium structure probability
    for i, model in tqdm(enumerate(models)):
        arr = [structure_probability(strands=[str1, str2], structure=structure, model=model) \
               for str1, str2, structure in zip(data['rna5'].tolist(), data['rna3'].tolist(), \
               data['mfe_structure_{}'.format(i)])]
        data['struct_proba_{}'.format(i)] = arr

    #suboptimal proxy structure
    for i, model in tqdm(enumerate(models)):
        arr = [subopt(strands=[str1, str2], energy_gap=1.5, model=model) \
               for str1, str2 in zip(data['rna5'].tolist(), data['rna3'].tolist())]
        data['subopt_energy_{}'.format(i)] = [x[0].energy for x in arr]
        data['subopt_stack_energy_{}'.format(i)] = [x[0].stack_energy for x in arr]
    
    #ensemble size
    for i, model in tqdm(enumerate(models)):
        arr = [ensemble_size(strands=[str1, str2], model=model) \
               for str1, str2 in zip(data['rna5'].tolist(), data['rna3'].tolist())]
        data['ensemble_{}'.format(i)] = arr
    
    '''
    try:
        for i, model in tqdm(enumerate(models)):
            arr = [defect(strands=[str1, str2], structure=structure, model=model) \
                   for str1, str2, structure in zip(data['rna5'].tolist(), data['rna3'].tolist(), \
                   data['mfe_structure_{}'.format(i)])]
            data['defect_{}'.format(i)] = arr
    except:
        print('***DEFECT does not work***')
    '''
    
    if config['encode_non_numeric']:
        #dealing with non numeric features
        for i, model in tqdm(enumerate(models)):
            temp = pd.DataFrame(data['prob_matrix_{}'.format(i)].tolist())
            temp.columns = ['pme_{}_{}'.format(i, j) for j in np.arange(temp.shape[1])]
            temp.drop([col for col, val in temp.sum().iteritems() if val==0], axis=1, inplace=True)
            data.drop('prob_matrix_{}'.format(i), axis=1, inplace=True)
            data = pd.concat([data, temp], axis=1)

        for i, model in tqdm(enumerate(models)):
            temp = pd.DataFrame(data['mfe_matrix_{}'.format(i)].tolist())
            temp.columns = ['mme_{}_{}'.format(i, j) for j in np.arange(temp.shape[1])]
            temp.drop([col for col, val in temp.sum().iteritems() if val==0 or val==1296], axis=1, inplace=True)
            data.drop('mfe_matrix_{}'.format(i), axis=1, inplace=True)
            data = pd.concat([data, temp], axis=1)

        for i, model in tqdm(enumerate(models)):
            temp = pd.DataFrame(data['mfe_pairlist_{}'.format(i)].tolist())
            temp.columns = ['mpe_{}_{}'.format(i, j) for j in np.arange(temp.shape[1])]
            data.drop('mfe_pairlist_{}'.format(i), axis=1, inplace=True)
            data = pd.concat([data, temp], axis=1)

        for i, model in tqdm(enumerate(models)):
            temp = pd.DataFrame(data['mfe_nicks_{}'.format(i)].tolist())
            temp.columns = ['mne_{}_{}'.format(i, j) for j in np.arange(temp.shape[1])]
            data.drop('mfe_nicks_{}'.format(i), axis=1, inplace=True)
            data = pd.concat([data, temp], axis=1)

        for i, model in tqdm(enumerate(models)):
            data['mfe_structure_{}_dot'.format(i)] = data['mfe_structure_{}'.format(i)].apply(lambda x: x.count('.'))
            data['mfe_structure_{}_left_bracket'.format(i)] = data['mfe_structure_{}'.format(i)].apply(lambda x: x.count('('))
            data['mfe_structure_{}_plus'.format(i)] = data['mfe_structure_{}'.format(i)].apply(lambda x: x.count('+'))
            data['mfe_structure_{}_right_bracket'.format(i)] = data['mfe_structure_{}'.format(i)].apply(lambda x: x.count(')'))
            data.drop('mfe_structure_{}'.format(i), axis=1, inplace=True)
    else:
        for i, model in tqdm(enumerate(models)):
            data.drop(['prob_matrix_{}'.format(i), 'mfe_matrix_{}'.format(i), 'mfe_pairlist_{}'.format(i),
                       'mfe_nicks_{}'.format(i), 'mfe_structure_{}'.format(i)], axis=1, inplace=True)
      
    return data.drop(['rna5', 'rna3', 'rna5-rna3', 'fluo'], axis=1)

### READ DATASET ###
raw_data = pd.read_csv('egs.csv')
data = preprocess_data(raw_data)

print('### SYSTEMATIC SPLIT - THERMODYNAMICAL MODEL ###')
scenario = 'systematic-split-thermo'
performances = []
predictions = []

#generate data split
systematic_split = create_systematic_split(data)
#feature engineering
transformed_data = generate_thermo_features(data)
#16-fold cross validation
for k, (data_a, data_b, data_c, data_d) in tqdm(enumerate(systematic_split)):

    training_data = data_a.copy()
    test_data_collection = [data_b.copy(), data_c.copy(), data_d.copy()]

    X_train = transformed_data.iloc[training_data.index.tolist()].values
    y_train = training_data['fluo'].values

    for i, test_data in enumerate(test_data_collection):

        X_test = transformed_data.iloc[test_data.index.tolist()].values
        y_test = test_data['fluo'].values
        
        for name, estimator in tqdm(regressors):

            if name=='Category Boosting':
                estimator.fit(X_train, y_train, logging_level='Silent')
            else:
                estimator.fit(X_train, y_train)
            y_pred_train = estimator.predict(X_train)
            y_pred_test = estimator.predict(X_test)

            performances.append((scenario, k, i, name, \
                                    np.sqrt(mean_squared_error(y_train, y_pred_train)), np.sqrt(mean_squared_error(y_test, y_pred_test)), \
                                    1-mean_absolute_percentage_error(y_train, y_pred_train), 1-mean_absolute_percentage_error(y_test, y_pred_test), \
                                    r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test), \
                                    pearsonr(y_train, y_pred_train)[0], pearsonr(y_test, y_pred_test)[0]))
            
            predictions.append((scenario, k, i, name, \
                                    y_test, y_pred_train, y_pred_test))
            
#generating reports
report = pd.DataFrame(performances)
report.columns = columns=['scenario', 'fold', 'group', 'name', 'train_rmse', 'test_rmse', 'train_mape', 'test_mape', 'train_r2', 'test_r2', 'train_pearson', 'test_pearson']
report.to_csv('report_{}.csv'.format(scenario), index=False)

pred_train_data = pd.DataFrame()
for prediction in predictions:
    temp = pd.DataFrame()
    temp['y_train'] = prediction[4]
    temp['y_pred_train'] = prediction[6]
    temp['scenario'] = prediction[0]
    temp['fold'] = prediction[1]
    temp['group'] = prediction[2]
    temp['name'] = prediction[3]
    pred_train_data = pd.concat([pred_train_data, temp])
pred_train_data.to_csv('pred-train-data_{}.csv'.format(scenario), index=False)

pred_test_data = pd.DataFrame()
for prediction in predictions:
    temp = pd.DataFrame()
    temp['y_test'] = prediction[5]
    temp['y_pred_test'] = prediction[7]
    temp['scenario'] = prediction[0]
    temp['fold'] = prediction[1]
    temp['test_index'] = prediction[2]
    temp['name'] = prediction[3]
    pred_test_data = pd.concat([pred_test_data, temp])
pred_test_data.to_csv('pred-test-data_{}.csv'.format(scenario), index=False)













