# ---------------------------------------------------------------------------------------------------------------------#
# import modules
import pandas as pd
import shutil
import os
import glob
import numpy as np
import logging
from sklearn.model_selection import KFold


# import biogeme modules
import biogeme
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import (
    Beta,
    Variable,
)

# configure logging
logging.basicConfig(
    filename='../outputs/1-2-mnl-I-val.log',  
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s'  
)
logging.info("Script execution started.")

# ---------------------------------------------------------------------------------------------------------------------#
# import prepared data
df = pd.read_pickle("../data/prepared_data.pkl")
df.describe

# define the data as biogeme global database
database = db.Database("mydata", df)

# define the variables
hv_tt = Variable("hv_tt")
hv_tc = Variable("hv_tc")
av_tt = Variable("av_tt")
av_tc = Variable("av_tc")
avwl_tt = Variable("avwl_tt")
avwl_tc = Variable("avwl_tc")
hv_av = Variable("hv_av")
av_av = Variable("av_av")
avwl_av = Variable("avwl_av")
chosen = Variable("chosen")

# ---------------------------------------------------------------------------------------------------------------------#
## 1-2-mnl-I-val
# 5-fold cross-validation of 1-1-mnl-I 

# function to get utility equations
def get_biogeme_model():
    
    # parameters to be estimated - alternative specific constants
    asc_av = Beta("asc_av", 0, None, None, 0)
    asc_avwl = Beta("asc_avwl", 0, None, None, 0)

    # parameters to be estimated - travel time and cost parameters
    b_vot_hv = Beta("b_vot_hv", 0, None, None, 0)
    b_vot_av = Beta("b_vot_av", 0, None, None, 0)
    b_vot_avwl = Beta("b_vot_avwl", 0, None, None, 0)
    b_cost = Beta("b_cost", 0, None, None, 0)

    # define utility functions
    v1 = b_cost * b_vot_hv * hv_tt + b_cost * hv_tc
    v2 = asc_av + b_cost * b_vot_av * av_tt + b_cost * av_tc
    v3 = asc_avwl + b_cost * b_vot_avwl * avwl_tt + b_cost * avwl_tc

    # link utility functions with the numbering of alternatives
    v = {1: v1, 2: v2, 3: v3}

    # availability of each alternatives
    av = {1: hv_av, 2: av_av, 3: avwl_av}

    # define the model
    logprob = models.loglogit(v, av, chosen)

    # return model
    return logprob

# split data for five-fold cross-validaiton
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# fit model for each splits
for train_index, test_index in kf.split(df, groups=df["id"]):
    
    # train and test splits
    train_data, test_data = df.iloc[train_index], df.iloc[test_index]

    # set train split as the database
    database = db.Database("mydata", train_data)

    # define loglikelihood
    logprob = get_biogeme_model()

    # estimate the model on train data
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = "1-2-mnl-I-val"
    results = biogeme.estimate()
    betas = results.get_beta_values()
    train_loglik = results.data.logLike
    logging.info(f"Log likelihood for {database.get_number_of_observations()} training data: {train_loglik}")

    # set test split as the database
    database = db.Database("mydata", test_data)

    # simulate log likelihood for test data
    simulate = {"Loglikelihood": logprob}
    sim_biogeme = bio.BIOGEME(database, simulate)
    sim_result = sim_biogeme.simulate(the_beta_values=betas)
    test_loglik = sim_result["Loglikelihood"].sum()
    logging.info(f"Log likelihood for {database.get_number_of_observations()} validation data: {test_loglik}")

    # remove intermediate outputs
    os.remove("1-2-mnl-I-val.pickle")
    os.remove("1-2-mnl-I-val.html")
    os.remove("__1-2-mnl-I-val.iter")

# ---------------------------------------------------------------------------------------------------------------------#
