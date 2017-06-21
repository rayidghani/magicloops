# magicloops
This is an example for loop that takes a number of machine learning classifiers implemented in sklearn, goes over a range of hyper-parameters for each classifier, and stores a set of evaluation metrics in a dataframe (or csv). 

There are three grid sizes:
* test: to test if things are working
* small: if you've got less than an hour
* large: if you've got time or cores

You can add more classifiers to it, add more hyperparameters, metrics, adapt it for regression, or clustering. You can also add another level of for loops to loop over different features, to see the effect of leaving a feature out or just using that feature.en

There are three files here that may be of interest:
1. mlfunctions.py contains several helpfer functions you might want to use for doing machine learning
2. simpleloops.py simple for loop over models
3. magicloop.py uses mlfunctions and buids a simple machine learning pipelie that loops over
  * various models
  * hyperparameters for each model
  * different outcomes
  * different dates for validation
  * different predictor/feature subsets
 
and then stores results with several evaluation metrics including
  * precision at k%
  * recall at k%
  * area under ROC curve
  * baselines
  
