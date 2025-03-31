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
    log,
    Variable,
    PanelLikelihoodTrajectory,
    MonteCarlo,
    bioDraws,
    exp,
)

# configure logging
logging.basicConfig(
    filename='../outputs/5-1-mxl-V-initial.log',  
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

tba_g1_hv = Variable("tba_g1_hv_cat")
tba_g2_hv = Variable("tba_g2_hv_cat")
tba_g3_hv = Variable("tba_g3_hv_cat")
tba_g4_hv = Variable("tba_g4_hv_cat")
tba_g5_hv = Variable("tba_g5_hv_cat")
tba_g6_hv = Variable("tba_g6_hv_cat")
tba_g7_hv = Variable("tba_g7_hv_cat")
tba_g1_av = Variable("tba_g1_av_cat")
tba_g2_av = Variable("tba_g2_av_cat")
tba_g3_av = Variable("tba_g3_av_cat")
tba_g4_av = Variable("tba_g4_av_cat")
tba_g5_av = Variable("tba_g5_av_cat")
tba_g6_av = Variable("tba_g6_av_cat")
tba_g7_av = Variable("tba_g7_av_cat")

ttu_hv = Variable("ttu_hv_std")
ttu_av = Variable("ttu_av_std")
tba_tot_hv = Variable("tba_tot_hv")
tba_tot_av = Variable("tba_tot_av")

# specify number of draws
number_of_draws = 5000

# ---------------------------------------------------------------------------------------------------------------------#
## 5-1-mxl-V-initial
# mixed logit model considering heterogeniety in VOT parameters
# time parameter: different across modes, triangular distribution
# cost parameter: fixed and same across modes
# asc parameter: normal distribution
# VOT space specification
# heterogeneity in VOT


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
database.set_random_number_generators(myRandomNumberGenerators)

# parameters to be estimated
# alternative specific constants
asc_av = Beta("asc_av", 0, None, None, 0)
s_asc_av = Beta("s_asc_av", 1, None, None, 0)
asc_av_rnd = asc_av + s_asc_av * bioDraws("asv_av_rnd", "NORMAL")
asc_avwl = Beta("asc_avwl", 0, None, None, 0)
s_asc_avwl = Beta("s_asc_avwl", 1, None, None, 0)
asc_avwl_rnd = asc_avwl + s_asc_avwl * bioDraws("asv_av_rnd", "NORMAL")

# travel time and cost parameters
b_cost = Beta("b_cost", -0.00324, None, None, 0)
b_vot_hv = Beta("b_vot_hv", 30, None, None, 0)
s_vot_hv = Beta("s_vot_hv", 1, None, None, 0)
b_vot_av = Beta("b_vot_av", 30, None, None, 0)
s_vot_av = Beta("s_vot_av", 1, None, None, 0)
b_vot_avwl = Beta("b_vot_avwl", 30, None, None, 0)
s_vot_avwl = Beta("s_vot_avwl", 1, None, None, 0)

# tba parameters
b_tba_g1_hv = Beta("b_tba_g1_hv", 0, None, None, 0)
b_tba_g2_hv = Beta("b_tba_g2_hv", 0, None, None, 0)
b_tba_g3_hv = Beta("b_tba_g3_hv", 0, None, None, 0)
b_tba_g4_hv = Beta("b_tba_g4_hv", 0, None, None, 0)
b_tba_g5_hv = Beta("b_tba_g5_hv", 0, None, None, 0)
b_tba_g6_hv = Beta("b_tba_g6_hv", 0, None, None, 0)
b_tba_g7_hv = Beta("b_tba_g7_hv", 0, None, None, 0)

b_tba_g1_av = Beta("b_tba_g1_av", 0, None, None, 0)
b_tba_g2_av = Beta("b_tba_g2_av", 0, None, None, 0)
b_tba_g3_av = Beta("b_tba_g3_av", 0, None, None, 0)
b_tba_g4_av = Beta("b_tba_g4_av", 0, None, None, 0)
b_tba_g5_av = Beta("b_tba_g5_av", 0, None, None, 0)
b_tba_g6_av = Beta("b_tba_g6_av", 0, None, None, 0)
b_tba_g7_av = Beta("b_tba_g7_av", 0, None, None, 0)

b_tba_g1_avwl = Beta("b_tba_g1_avwl", 0, None, None, 0)
b_tba_g2_avwl = Beta("b_tba_g2_avwl", 0, None, None, 0)
b_tba_g3_avwl = Beta("b_tba_g3_avwl", 0, None, None, 0)
b_tba_g4_avwl = Beta("b_tba_g4_avwl", 0, None, None, 0)
b_tba_g5_avwl = Beta("b_tba_g5_avwl", 0, None, None, 0)
b_tba_g6_avwl = Beta("b_tba_g6_avwl", 0, None, None, 0)
b_tba_g7_avwl = Beta("b_tba_g7_avwl", 0, None, None, 0)

b_ttu_hv = Beta("b_ttu_hv", 0, None, None, 0)
b_ttu_av = Beta("b_ttu_av", 0, None, None, 0)
b_ttu_avwl = Beta("b_ttu_avwl", 0, None, None, 0)

b_tba_tot_hv = Beta("b_tba_tot_hv", 0, None, None, 0)
b_tba_tot_av = Beta("b_tba_tot_av", 0, None, None, 0)
b_tba_tot_avwl = Beta("b_tba_tot_avwl", 0, None, None, 0)

# random VOT parameters
b_vot_hv_rnd = (
    b_vot_hv
    + s_vot_hv * bioDraws("b_vot_hv_rnd", "TRIANGULAR")
    + b_tba_g1_hv * tba_g1_hv
    + b_tba_g2_hv * tba_g2_hv
    + b_tba_g3_hv * tba_g3_hv
    + b_tba_g4_hv * tba_g4_hv
    + b_tba_g5_hv * tba_g5_hv
    + b_tba_g6_hv * tba_g6_hv
    + b_tba_g7_hv * tba_g7_hv
    + b_ttu_hv * ttu_hv
)
b_vot_av_rnd = (
    b_vot_av
    + s_vot_av * bioDraws("b_vot_av_rnd", "TRIANGULAR")
    + b_tba_g1_av * tba_g1_av
    + b_tba_g2_av * tba_g2_av
    + b_tba_g3_av * tba_g3_av
    + b_tba_g4_av * tba_g4_av
    + b_tba_g5_av * tba_g5_av
    + b_tba_g6_av * tba_g6_av
    + b_tba_g7_av * tba_g7_av
    + b_ttu_av * ttu_av
)
b_vot_avwl_rnd = (
    b_vot_avwl
    + s_vot_avwl * bioDraws("b_vot_avwl_rnd", "TRIANGULAR")
    + b_tba_g1_avwl * tba_g1_av
    + b_tba_g2_avwl * tba_g2_av
    + b_tba_g3_avwl * tba_g3_av
    + b_tba_g4_avwl * tba_g4_av
    + b_tba_g5_avwl * tba_g5_av
    + b_tba_g6_avwl * tba_g6_av
    + b_tba_g7_avwl * tba_g7_av
    + b_ttu_avwl * ttu_av
)

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

# panel data
database.panel("id")
condprobIndiv = PanelLikelihoodTrajectory(obsprob)
logprob = log(MonteCarlo(condprobIndiv))

# estimate the model
biogeme = bio.BIOGEME(database, logprob, number_of_draws=number_of_draws)
# biogeme.loadSavedIteration()
biogeme.modelName = "5-1-mxl-V-initial"

# get the results in a pandas table
print(biogeme.estimate().getEstimatedParameters())

# move outpts to outputs folder
shutil.move("5-1-mxl-V-initial.html", "../outputs/5-1-mxl-V-initial.html")

# remove intermediate outputs
os.remove("5-1-mxl-V-initial.pickle")
os.remove("__5-1-mxl-V-initial.iter")

# ---------------------------------------------------------------------------------------------------------------------#
