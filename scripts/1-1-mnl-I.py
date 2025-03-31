# ---------------------------------------------------------------------------------------------------------------------#
# import modules
import pandas as pd
import shutil
import os
import glob
import numpy as np
import logging

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
    filename='../outputs/1-1-mnl-I.log',  
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
## 1-1-mnl-I
# simple multinomial logit model with time and cost parameters only
# time parameter fixed but different across modes
# cost parameter fixed and same across modes
# vot space specification

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

# create biogeme object
biogeme = bio.BIOGEME(database, logprob)

# model name
biogeme.modelName = "1-1-mnl-I"

# calculate null loglikelihood
biogeme.calculateNullLoglikelihood(avail=av)

# get the results in a pandas table
print(biogeme.estimate().getEstimatedParameters())

# move outpts to outputs folder
shutil.move("1-1-mnl-I.html", "../outputs/1-1-mnl-I.html")

# remove intermediate outputs
os.remove("1-1-mnl-I.pickle")
os.remove("__1-1-mnl-I.iter")

# ---------------------------------------------------------------------------------------------------------------------#
