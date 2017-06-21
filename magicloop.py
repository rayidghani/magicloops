
# Import Statements
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
from datetime import timedelta
from datetime import datetime
from __future__ import division
import random
from scipy import optimize
import time
import seaborn as sns
import csv

from mlfunctions import *


def main():

    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)

    # parse input parameters

    # csv data file to be used as input
    infile = sys.argv[1]

    # the filename we want to write results to
    outfile = sys.argv[2]

    # which model(s) to run
    model = sys.argv[3]

    # which parameter grid do we want to use (test, small, large)
    grid_size = sys.argv[4]

    #read the csv data
    data = pd.read_csv(infile)

    # which variable to use for prediction_time
    prediction_time = 'dis_date'

    # outcome variables we want to loop over
    outcomes = ['30_day_readmits', '60_day_readmits','180_day_readmits']
    
    # validation dates we want to loop over
    validation_dates = ['2012-04-01', '2012-10-01', '2013-04-01']

    # define feature groups
    demographic_predictors = ['age', 'gender', 'race']
    admission_predictors = ['num_visits_so_far','avg_los_so_far','min_los_so_far','max_los_so_far','std_los_so_far']
    sensor_predictors = ['reading1', 'reading2', 'reading3']
    survey_predictors=['response1', 'response2', 'response3']

   
    # models_to_run=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', 'NB']
    if (model == 'all'):
        models_to_run=['RF','LR','DT','ET','AB']
    else:
        models_to_run = []
        models_to_run.append(model)

    clfs, grid = define_clfs_params(grid_size)

    # which feature/predictor sets do we want to use in our analysis
    predictor_sets = [demographic_predictors, admission_predictors,sensor_predictors,survey_predictors]
    
    # generate all possible subsets of the feature/predictor groups
    predictor_subsets = get_subsets(predictor_sets)

    all_predictors=[]
    for p in predictor_subsets:
        merged = list(itertools.chain.from_iterable(p))
        all_predictors.append(merged)

    # write header for the csv
    with open(outfile, "w") as myfile:
        myfile.write("model_type ,clf, parameters, outcome, validation_date, group,train_set_size, validation_set_size,predictors,baseline,precision_at_5,precision_at_10,precision_at_20,precision_at_30,precision_at_40,precision_at_50,recall_at_5,recall_at_10,recall_at_20,recall_at_30,recall_at_40, ecall_at_50,auc-roc")

    # define dataframe to write results to
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'outcome', 'validation_date', 'group',
                                        'train_set_size', 'validation_set_size','predictors',
                                        'baseline','precision_at_5','precision_at_10','precision_at_20','precision_at_30','precision_at_40',
                                        'precision_at_50','recall_at_5','recall_at_10','recall_at_20','recall_at_30','recall_at_40',
                                        'recall_at_50','auc-roc'))

    # the magic loop starts here
    # we will loop over models, parameters, outcomes, validation_Dates
    # and store several evaluation metrics

    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            for current_outcome in outcomes:
                for predictor in all_predictors:
                    for validation_date in validation_dates:
                        try:
                            print models_to_run[index]
                            clf.set_params(**p)
                            if (outcome == '30_day_readmits'):
                                delta = 30
                            elif (outcome == '60_day_readmits'):
                                delta = 60
                            elif (outcome == '180_day_readmits'):
                                delta = 180
                            else:
                                raise ValueError('value of outcome is unknown')                 
                        
                            train_set = data[data[prediction_time] <= datetime.strptime(validation_date, '%Y-%m-%d') - timedelta(days=delta)]
                            # fill in missing values for train set using just the train set
                            # we'll do it a very naive way here but you should think more carefully about this first
                            train_set.fillna(train_set.mean(), inplace=True)
                            train_set.dropna(axis=1, how='any', inplace=True)
                            
                            validation_set = data[data[prediction_time] > datetime.strptime(validation_date, '%Y-%m-%d') - timedelta(days=0)]
                            # fill in missing values for validation set using all the data
                            # we'll do it a very naive way here but you should think more carefully about this first
                            validation_set.fillna(data.mean(), inplace=True)
                            validation_set.dropna(axis=1, how='any', inplace=True)

                            print predictor
                            # get predictors by removing those dropped by dropna
                            predictors_to_use = list(set(predictor).intersection(train_set.columns))

                            model = clf.fit(train_set[predictor], train_set[current_outcome]) 
                            pred_probs = clf.predict_proba(validation_set[predictor])[::,1]
                            print len(train_set)
                            print len(validation_set)
                            #pred_probs_sorted, true_outcome_sorted = zip(*sorted(zip(pred_probs, validation_set[current_outcome]), reverse=True))
                            results_df.loc[len(results_df)] = [models_to_run[index],clf, p, current_outcome, validation_date, group,
                                                               len(train_set),len(validation_set), 
                                                               predictor, 
                                                                precision_at_k(validation_set[current_outcome],pred_probs, 100),
                                                                precision_at_k(validation_set[current_outcome],pred_probs, 5),
                                                                precision_at_k(validation_set[current_outcome],pred_probs, 10),
                                                                precision_at_k(validation_set[current_outcome],pred_probs, 20),
                                                                precision_at_k(validation_set[current_outcome],pred_probs, 30),
                                                                precision_at_k(validation_set[current_outcome],pred_probs, 40),
                                                                precision_at_k(validation_set[current_outcome],pred_probs, 50),
                                                                recall_at_k(validation_set[current_outcome],pred_probs, 5),
                                                                recall_at_k(validation_set[current_outcome],pred_probs, 10),
                                                                recall_at_k(validation_set[current_outcome],pred_probs, 20),
                                                                recall_at_k(validation_set[current_outcome],pred_probs, 30),
                                                                recall_at_k(validation_set[current_outcome],pred_probs, 40),
                                                                recall_at_k(validation_set[current_outcome],pred_probs, 50),
                                                                roc_auc_score(validation_set[current_outcome], pred_probs)]

                            # plot precision recall graph
                            # we'll show them here but you can also save them to disk
                            plot_precision_recall_n(validation_set[current_outcome], pred_probs, clf, 'show')
                            # write results to csv as they come in so we always have something to see even if models runs for days
                            with open(outfile, "a") as myfile:
                                csvwriter = csv.writer(myfile, dialect='excel', quoting=csv.QUOTE_ALL)
                                strp = str(p)
                                strp.replace('\n', '')
                                strclf = str(clf)
                                strclf.replace('\n', '')
                                csvwriter.writerow([models_to_run[index],strclf, strp, current_outcome, validation_date, group,len(train_set),len(validation_set), predictor,  precision_at_k(validation_set[current_outcome],pred_probs, 100), precision_at_k(validation_set[current_outcome],pred_probs, 5), precision_at_k(validation_set[current_outcome],pred_probs, 10), precision_at_k(validation_set[current_outcome],pred_probs, 20), precision_at_k(validation_set[current_outcome],pred_probs, 30), precision_at_k(validation_set[current_outcome],pred_probs, 40), precision_at_k(validation_set[current_outcome],pred_probs, 50), recall_at_k(validation_set[current_outcome],pred_probs, 5), recall_at_k(validation_set[current_outcome],pred_probs, 10), recall_at_k(validation_set[current_outcome],pred_probs, 20), recall_at_k(validation_set[current_outcome],pred_probs, 30), recall_at_k(validation_set[current_outcome],pred_probs, 40), recall_at_k(validation_set[current_outcome],pred_probs, 50),roc_auc_score(validation_set[current_outcome], pred_probs)])
                        except IndexError, e:
                            print 'Error:',e
                            continue
    
    # write final dataframe to csv
    dfoutfile = 'df_' + outfile
    results_df.to_csv(dfoutfile, index=False)


if __name__ == '__main__':
    main()

