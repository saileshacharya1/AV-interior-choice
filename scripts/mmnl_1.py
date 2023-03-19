# ---------------------------------------------------------------------------------------------------------------------#
# import modules
import pandas as pd
import shutil

# import biogeme modules
import biogeme
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models

import biogeme.messaging as msg
from biogeme import vns
from biogeme import assisted
from biogeme.expressions import (
    Beta,
    log,
    Elem,
    Numeric,
    Variable,
    PanelLikelihoodTrajectory,
    MonteCarlo,
    bioDraws,
)

from biogeme.assisted import (
    DiscreteSegmentationTuple,
    TermTuple,
    SegmentedParameterTuple,
)


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

age_grp_2 = Variable("age_grp_2")
age_grp_3 = Variable("age_grp_3")
hh_vehs = Variable("hh_vehs")

tu_diff = Variable("tu_diff")

# ---------------------------------------------------------------------------------------------------------------------#
## mnl

# parameters to be estimated - alternative specific constants
asc_av = Beta("asc_av", 0, None, None, 0)
asc_avwl = Beta("asc_avwl", 0, None, None, 0)

# parameters to be estimated - travel time and cost
b_tt_hv = Beta("b_tt_hv", 0, None, None, 0)
b_tc_hv = Beta("b_tc_hv", 0, None, None, 0)
b_tt_av = Beta("b_tt_av", 0, None, None, 0)
b_tc_av = Beta("b_tc_av", 0, None, None, 0)
b_tt_avwl = Beta("b_tt_avwl", 0, None, None, 0)
b_tc_avwl = Beta("b_tc_avwl", 0, None, None, 0)

# sociodemographic parameters to be estimated
b_age_grp_2_av = Beta("b_age_grp_2_av", 0, None, None, 0)
b_age_grp_2_avwl = Beta("b_age_grp_2_avwl", 0, None, None, 0)
b_age_grp_3_av = Beta("b_age_grp_3_av", 0, None, None, 0)
b_age_grp_3_avwl = Beta("b_age_grp_3_avwl", 0, None, None, 0)

b_hh_vehs_av = Beta("b_hh_vehs_av", 0, None, None, 0)
b_hh_vehs_avwl = Beta("b_hh_vehs_avwl", 0, None, None, 0)

b_tu_diff_av = Beta("b_tu_diff_av", 0, None, None, 0)
b_tu_diff_avwl = Beta("b_tu_diff_avwl", 0, None, None, 0)

# define utility functions
v1 = b_tt_hv * hv_tt + b_tc_hv * hv_tc
v2 = (
    asc_av
    + b_tt_av * av_tt
    + b_tc_av * av_tc
    + b_hh_vehs_av * hh_vehs
    + b_age_grp_2_av * age_grp_2
    + b_age_grp_3_av * age_grp_3
    + b_tu_diff_av * tu_diff
)
v3 = (
    asc_avwl
    + b_tt_avwl * avwl_tt
    + b_tc_avwl * avwl_tc
    + b_hh_vehs_avwl * hh_vehs
    + b_age_grp_2_avwl * age_grp_2
    + b_age_grp_3_avwl * age_grp_3
    + b_tu_diff_avwl * tu_diff
)

# link utility functions with the numbering of alternatives
v = {1: v1, 2: v2, 3: v3}

# availability of each alternatives
av = {1: hv_av, 2: av_av, 3: avwl_av}

# define the model
logprog = models.loglogit(v, av, chosen)

# create biogeme object
biogeme = bio.BIOGEME(database, logprog, suggestScales=False)

# model name
biogeme.modelName = "mnl"

# calculate null loglikelihood
biogeme.calculateNullLoglikelihood(avail=av)

# get the results in a pandas table
print(biogeme.estimate().getEstimatedParameters())

# move outpts to outputs folder
shutil.move("mnl.pickle", "../outputs/mnl.pickle")
shutil.move("mnl.html", "../outputs/mnl.html")

# ---------------------------------------------------------------------------------------------------------------------#
## mmnl

# parameters to be estimated
# alternative specific constants
# asc_hv = Beta('asc_hv', 0, None, None, 0)
asc_av = Beta("asc_av", 0, None, None, 0)
asc_avwl = Beta("asc_avwl", 0, None, None, 0)

# travel time and cost parameters
b_tt_hv = Beta("b_tt_hv", 0, None, None, 0)
b_tc_hv = Beta("b_tc_hv", 0, None, None, 0)
sigma_tt_hv = Beta("sigma_tt_hv", 0, None, None, 0)
sigma_tc_hv = Beta("sigma_tc_hv", 0, None, None, 0)
b_tt_hv_rnd = b_tt_hv + sigma_tt_hv * bioDraws("b_tt_hv_rnd", "NORMAL")
b_tc_hv_rnd = b_tc_hv + sigma_tc_hv * bioDraws("b_tc_hv_rnd", "NORMAL")

b_tt_av = Beta("b_tt_av", 0, None, None, 0)
b_tc_av = Beta("b_tc_av", 0, None, None, 0)
sigma_tt_av = Beta("sigma_tt_av", 0, None, None, 0)
sigma_tc_av = Beta("sigma_tc_av", 0, None, None, 0)
b_tt_av_rnd = b_tt_av + sigma_tt_av * bioDraws("b_tt_av_rnd", "NORMAL")
b_tc_av_rnd = b_tc_av + sigma_tc_av * bioDraws("b_tc_av_rnd", "NORMAL")

b_tt_avwl = Beta("b_tt_avwl", 0, None, None, 0)
b_tc_avwl = Beta("b_tc_avwl", 0, None, None, 0)
sigma_tt_avwl = Beta("sigma_tt_avwl", 0, None, None, 0)
sigma_tc_avwl = Beta("sigma_tc_avwl", 0, None, None, 0)
b_tt_avwl_rnd = b_tt_avwl + sigma_tt_avwl * bioDraws("b_tt_avwl_rnd", "NORMAL")
b_tc_avwl_rnd = b_tc_avwl + sigma_tc_avwl * bioDraws("b_tc_avwl_rnd", "NORMAL")

# define utility functions
v1 = b_tt_hv_rnd * hv_tt + b_tc_hv_rnd * hv_tc
v2 = asc_av + b_tt_av_rnd * av_tt + b_tc_av_rnd * av_tc
v3 = asc_avwl + b_tt_avwl_rnd * avwl_tt + b_tc_avwl_rnd * avwl_tc


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
biogeme = bio.BIOGEME(database, logprob, suggestScales=False)
# biogeme.loadSavedIteration()
biogeme.modelName = "mmnl"

# get the results in a pandas table
print(biogeme.estimate().getEstimatedParameters())

# move outpts to outputs folder
shutil.move("mnnl.pickle", "../outputs/mnnl_1.pickle")
shutil.move("mnnl.html", "../outputs/mnnl_1.html")

# ---------------------------------------------------------------------------------------------------------------------#
