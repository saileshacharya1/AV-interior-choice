# ---------------------------------------------------------------------------------------------------------------------#
# import modules
import pandas as pd
import shutil
import os
import numpy as np
from sklearn.model_selection import GroupKFold
import logging

# import biogeme modules
import biogeme
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import (
    Beta,
    log,
    Variable,
    PanelLikelihoodTrajectory,
    MonteCarlo,
    bioDraws,
    exp,
)
    
# configure logging
logging.basicConfig(
    filename='../outputs/3-2-mxl-III-val.log',  
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

# specify number of draws
number_of_draws = 10000

# ---------------------------------------------------------------------------------------------------------------------#
## 3-2-mxl-III-val
# 5-fold cross-validation of 3-1-mxl-III

# define triangular distribution generator
def the_triangular_generator(sample_size, number_of_draws):
    """
    Provide my own random number generator to the database.
    """
    return np.random.triangular(-1, 0, 1, (sample_size, number_of_draws))


myRandomNumberGenerators = {
    "TRIANGULAR": (
        the_triangular_generator,
        "Draws from a triangular distribution",
    )
}

# function to get utility equations
def get_biogeme_model():
    
    # parameters to be estimated
    # alternative specific constants
    asc_av = Beta("asc_av", -0.789, None, None, 0)
    s_asc_av = Beta("s_asc_av", 0.229, None, None, 0)
    asc_av_rnd = asc_av + s_asc_av * bioDraws("asv_av_rnd", "NORMAL")

    asc_avwl = Beta("asc_avwl", -0.584, None, None, 0)
    s_asc_avwl = Beta("s_asc_avwl", 1.54, None, None, 0)
    asc_avwl_rnd = asc_avwl + s_asc_avwl * bioDraws("asv_avwl_rnd", "NORMAL")

    # travel time and cost parameters
    b_cost = Beta("b_cost", -0.00584, None, None, 0)

    b_vot_hv = Beta("b_vot_hv", 48.9, None, None, 0)
    sigma_vot_hv = Beta("sigma_vot_hv", 171, None, None, 0)
    b_vot_hv_rnd = b_vot_hv + sigma_vot_hv * bioDraws("b_vot_hv_rnd", "TRIANGULAR")

    b_vot_av = Beta("b_vot_av", 34, None, None, 0)
    sigma_vot_av = Beta("sigma_vot_av", -21.6, None, None, 0)
    b_vot_av_rnd = b_vot_av + sigma_vot_av * bioDraws("b_vot_av_rnd", "TRIANGULAR")

    b_vot_avwl = Beta("b_vot_avwl", 32.1, None, None, 0)
    sigma_vot_avwl = Beta("sigma_vot_avwl", -13.9, None, None, 0)
    b_vot_avwl_rnd = b_vot_avwl + sigma_vot_avwl * bioDraws("b_vot_avwl_rnd", "TRIANGULAR")

    # define utility functions
    v1 = b_cost * b_vot_hv_rnd * hv_tt + b_cost * hv_tc
    v2 = asc_av_rnd + b_cost * b_vot_av_rnd * av_tt + b_cost * av_tc
    v3 = asc_avwl_rnd + b_cost * b_vot_avwl_rnd * avwl_tt + b_cost * avwl_tc

    # link utility functions with the numbering of alternatives
    v = {1: v1, 2: v2, 3: v3}

    # availability of each alternatives
    av = {1: hv_av, 2: av_av, 3: avwl_av}

    # define the model
    obsprob = models.logit(v, av, chosen)

    # return model
    return obsprob

# split data for five-fold cross-validaiton
kf = GroupKFold(n_splits=5)

# fit model for each splits
for train_index, test_index in kf.split(df, groups=df["id"]):
    
    # train and test splits
    train_data, test_data = df.iloc[train_index], df.iloc[test_index]

    # set train split as the database
    database = db.Database("mydata", train_data)
    database.panel("id")
    database.set_random_number_generators(myRandomNumberGenerators)

    # define loglikelihood
    obsprob = get_biogeme_model()
    condprobIndiv = PanelLikelihoodTrajectory(obsprob)
    logprob = log(MonteCarlo(condprobIndiv))
    
    # estimate the model on train data
    biogeme = bio.BIOGEME(database, logprob, number_of_draws=number_of_draws)
    biogeme.modelName = "3-2-mxl-III-val"
    results = biogeme.estimate()
    betas = results.get_beta_values()
    train_loglik = results.data.logLike
    logging.info(f"Log likelihood for {database.get_number_of_observations()} training data: {train_loglik}")


    # set test split as the database
    database = db.Database("mydata", test_data)
    database.panel("id")
    database.set_random_number_generators(myRandomNumberGenerators)

    # simulate log likelihood for test data
    simulate = {"Loglikelihood": logprob}
    sim_biogeme = bio.BIOGEME(database,simulate)
    sim_result = sim_biogeme.simulate(the_beta_values=betas)
    test_loglik = sim_result["Loglikelihood"].sum()
    logging.info(f"Log likelihood for {database.get_number_of_observations()} validation data: {test_loglik}")

    # remove intermediate outputs
    os.remove("3-2-mxl-III-val.pickle")
    os.remove("3-2-mxl-III-val.html")

# ---------------------------------------------------------------------------------------------------------------------#
