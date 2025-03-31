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
    filename='../outputs/4-6-mxl-IV-final-elast.log',  
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

age_grp_2 = Variable("age_grp_2")
age_grp_3 = Variable("age_grp_3")
race_1 = Variable("race_1")
gender_1 = Variable("gender_1")
education_2 = Variable("education_2")
education_3 = Variable("education_3")
school_2 = Variable("school_2")
school_3 = Variable("school_3")
hh_adult = Variable("hh_adult")
hh_child = Variable("hh_child")
income_grp_2 = Variable("income_grp_2")
income_grp_3 = Variable("income_grp_3")
income_grp_4 = Variable("income_grp_4")
income_grp_5 = Variable("income_grp_5")
employment_2 = Variable("employment_2")
employment_3 = Variable("employment_3")
driving_exp = Variable("driving_exp")
citation = Variable("citation")
crash_exp = Variable("crash_exp")
hh_vehs = Variable("hh_vehs")
mode_commute_3 = Variable("mode_commute_3")
mode_shopping_3 = Variable("mode_shopping_3")
mode_personal_3 = Variable("mode_personal_3")
mode_social_3 = Variable("mode_social_3")
rec_trips = Variable("rec_trips")
av_usefulness = Variable("av_usefulness_std")
av_concern = Variable("av_concern_std")
tech_savviness = Variable("tech_savviness_std")
driving_enjoyment = Variable("driving_enjoyment_std")
polychronicity = Variable("polychronicity_std")
envt_concern = Variable("envt_concern")
av_fam = Variable("av_fam_std")
tba_tot_diff = Variable("tba_tot_diff")
ttu_diff = Variable("ttu_diff_std")

# specify number of draws
number_of_draws = 5000

# ---------------------------------------------------------------------------------------------------------------------#
## 4-6-mxl-IV-final-elast
# calculate elasticity of parameters of 4-2-mxl-IV-final

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
asc_av = Beta("asc_av", -0.668, None, None, 0)
s_asc_av = Beta("s_asc_av", 0.142, None, None, 0)
asc_avwl = Beta("asc_avwl", -0.554, None, None, 0)
s_asc_avwl = Beta("s_asc_avwl", 0.108, None, None, 0)

# travel time and cost parameters
b_cost = Beta("b_cost", -0.00584, None, None, 0)
b_vot_hv = Beta("b_vot_hv", 0.731, None, None, 0)
s_vot_hv = Beta("s_vot_hv", 2.55, None, None, 0)
b_vot_hv_rnd = b_vot_hv + s_vot_hv * bioDraws("b_vot_hv_rnd", "TRIANGULAR")
b_vot_av = Beta("b_vot_av", 0.532, None, None, 0)
s_vot_av = Beta("s_vot_av", 0.276, None, None, 0)
b_vot_av_rnd = b_vot_av + s_vot_av * bioDraws("b_vot_av_rnd", "TRIANGULAR")
b_vot_avwl = Beta("b_vot_avwl", 0.492, None, None, 0)
s_vot_avwl = Beta("s_vot_avwl", 0.108, None, None, 0)
b_vot_avwl_rnd = b_vot_avwl + s_vot_avwl * bioDraws("b_vot_avwl_rnd", "TRIANGULAR")

# socio-demographic parameters
b_age_grp_2_av = Beta("b_age_grp_2_av", -0.559, None, None, 0)
b_age_grp_3_av = Beta("b_age_grp_3_av", -1.46, None, None, 0)
b_race_1_av = Beta("b_race_1_av", -0.02, None, None, 0)
b_gender_1_av = Beta("b_gender_1_av", -0.19, None, None, 0)
b_education_2_av = Beta("b_education_2_av", 0.304, None, None, 0)
b_education_3_av = Beta("b_education_3_av", 0.755, None, None, 0)
b_school_2_av = Beta("b_school_2_av", 1.13, None, None, 0)
b_school_3_av = Beta("b_school_3_av", 0.226, None, None, 0)
b_hh_adult_av = Beta("b_hh_adult_av", -0.009, None, None, 0)
b_hh_child_av = Beta("b_hh_child_av", 0.226, None, None, 0)
b_income_grp_2_av = Beta("b_income_grp_2_av", 0, None, None, 0)
b_income_grp_3_av = Beta("b_income_grp_3_av", 0, None, None, 0)
b_income_grp_4_av = Beta("b_income_grp_4_av", 0, None, None, 0)
b_income_grp_5_av = Beta("b_income_grp_5_av", 0, None, None, 0)
b_employment_2_av = Beta("b_employment_2_av", 0, None, None, 0)
b_employment_3_av = Beta("b_employment_3_av", 0, None, None, 0)
b_driving_exp_av = Beta("b_driving_exp_av", 0, None, None, 0)
b_citation_av = Beta("b_citation_av", 0, None, None, 0)
b_crash_exp_av = Beta("b_crash_exp_av", 0, None, None, 0)
b_hh_vehs_av = Beta("b_hh_vehs_av", -0.0668, None, None, 0)
b_mode_commute_3_av = Beta("b_mode_commute_3_av", 0, None, None, 0)
b_mode_shopping_3_av = Beta("b_mode_shopping_3_av", 0, None, None, 0)
b_mode_personal_3_av = Beta("b_mode_personal_3_av", 0, None, None, 0)
b_mode_social_3_av = Beta("b_mode_social_3_av", 0, None, None, 0)
b_rec_trips_av = Beta("b_rec_trips_av", 0, None, None, 0)
b_av_usefulness_av = Beta("b_av_usefulness_av", 0, None, None, 0)
b_av_concern_av = Beta("b_av_concern_av", 0, None, None, 0)
b_tech_savviness_av = Beta("b_tech_savviness_av", 0, None, None, 0)
b_driving_enjoyment_av = Beta("b_driving_enjoyment_av", 0, None, None, 0)
b_polychronicity_av = Beta("b_polychronicity_av", 0, None, None, 0)
b_envt_concern_av = Beta("b_envt_concern_av", 0, None, None, 0)
b_av_fam_av = Beta("b_av_fam_av", 0, None, None, 0)
b_ttu_diff_av = Beta("b_ttu_diff_av", 0, None, None, 0)
b_tba_tot_diff_av = Beta("b_tba_tot_diff_av", 0, None, None, 0)

b_age_grp_2_avwl = Beta("b_age_grp_2_avwl", -0.663, None, None, 0)
b_age_grp_3_avwl = Beta("b_age_grp_3_avwl", -2.08, None, None, 0)
b_race_1_avwl = Beta("b_race_1_avwl", 0.493, None, None, 0)
b_gender_1_avwl = Beta("b_gender_1_avwl", -0.231, None, None, 0)
b_education_2_avwl = Beta("b_education_2_avwl", 0.352, None, None, 0)
b_education_3_avwl = Beta("b_education_3_avwl", 0.17, None, None, 0)
b_school_2_avwl = Beta("b_school_2_avwl", 0.281, None, None, 0)
b_school_3_avwl = Beta("b_school_3_avwl", -0.0875, None, None, 0)
b_hh_adult_avwl = Beta("b_hh_adult_avwl", -0.0334, None, None, 0)
b_hh_child_avwl = Beta("b_hh_child_avwl", 0.099, None, None, 0)
b_income_grp_2_avwl = Beta("b_income_grp_2_avwl", 0, None, None, 0)
b_income_grp_3_avwl = Beta("b_income_grp_3_avwl", 0, None, None, 0)
b_income_grp_4_avwl = Beta("b_income_grp_4_avwl", 0, None, None, 0)
b_income_grp_5_avwl = Beta("b_income_grp_5_avwl", 0, None, None, 0)
b_employment_2_avwl = Beta("b_employment_2_avwl", 0, None, None, 0)
b_employment_3_avwl = Beta("b_employment_3_avwl", 0, None, None, 0)
b_driving_exp_avwl = Beta("b_driving_exp_avwl", 0, None, None, 0)
b_citation_avwl = Beta("b_citation_avwl", 0, None, None, 0)
b_crash_exp_avwl = Beta("b_crash_exp_avwl", 0, None, None, 0)
b_hh_vehs_avwl = Beta("b_hh_vehs_avwl", -0.0152, None, None, 0)
b_mode_commute_3_avwl = Beta("b_mode_commute_3_avwl", 0, None, None, 0)
b_mode_shopping_3_avwl = Beta("b_mode_shopping_3_avwl", 0, None, None, 0)
b_mode_personal_3_avwl = Beta("b_mode_personal_3_avwl", 0, None, None, 0)
b_mode_social_3_avwl = Beta("b_mode_social_3_avwl", 0, None, None, 0)
b_rec_trips_avwl = Beta("b_rec_trips_avwl", 0, None, None, 0)
b_av_usefulness_avwl = Beta("b_av_usefulness_avwl", 0, None, None, 0)
b_av_concern_avwl = Beta("b_av_concern_avwl", 0, None, None, 0)
b_tech_savviness_avwl = Beta("b_tech_savviness_avwl", 0, None, None, 0)
b_driving_enjoyment_avwl = Beta("b_driving_enjoyment_avwl", 0, None, None, 0)
b_polychronicity_avwl = Beta("b_polychronicity_avwl", 0, None, None, 0)
b_envt_concern_avwl = Beta("b_envt_concern_avwl", 0, None, None, 0)
b_av_fam_avwl = Beta("b_av_fam_avwl", 0, None, None, 0)
b_ttu_diff_avwl = Beta("b_ttu_diff_avwl", 0, None, None, 0)
b_tba_tot_diff_avwl = Beta("b_tba_tot_diff_avwl", 0, None, None, 0)

# function to get utility equations
def get_utility():
    # random asc parameters
    asc_av_rnd = (
        asc_av
        + s_asc_av * bioDraws("asv_av_rnd", "NORMAL")
        + b_school_2_av * school_2
        + b_hh_child_av * hh_child
        + b_income_grp_2_av * income_grp_2
        + b_income_grp_3_av * income_grp_3
        + b_income_grp_5_av * income_grp_5
        + b_mode_social_3_av * mode_social_3
        + b_av_usefulness_av * av_usefulness
        + b_polychronicity_av * polychronicity
        + b_envt_concern_av * envt_concern
        + b_tba_tot_diff_av * tba_tot_diff
        + b_ttu_diff_av * ttu_diff
    )
    asc_avwl_rnd = (
        asc_avwl
        + s_asc_avwl * bioDraws("asv_avwl_rnd", "NORMAL")
        + b_age_grp_3_avwl * age_grp_3
        + b_school_3_avwl * school_3
        + b_citation_avwl * citation
        + b_av_usefulness_avwl * av_usefulness
        + b_envt_concern_avwl * envt_concern
        + b_av_fam_avwl * av_fam
        + b_tba_tot_diff_avwl * tba_tot_diff
        + b_ttu_diff_avwl * ttu_diff
    )

    # define utility functions
    v1 = b_cost * b_vot_hv_rnd * hv_tt + b_cost * hv_tc
    v2 = asc_av_rnd + b_cost * b_vot_av_rnd * av_tt + b_cost * av_tc
    v3 = asc_avwl_rnd + b_cost * b_vot_avwl_rnd * avwl_tt + b_cost * avwl_tc

    # link utility functions with the numbering of alternatives
    v = {1: v1, 2: v2, 3: v3}
    
    return v

# utility
v = get_utility()

# availability of each alternatives
av = {1: hv_av, 2: av_av, 3: avwl_av}

# define the model
obsprob = models.logit(v, av, chosen)

# panel data
database.panel("id")
condprobIndiv = PanelLikelihoodTrajectory(obsprob)
logprob = log(MonteCarlo(condprobIndiv))

# estimate the model
biogeme = bio.BIOGEME(database, logprob, suggestScales=False, number_of_draws=number_of_draws)
# biogeme.loadSavedIteration()
biogeme.modelName = "4-6-mxl-IV-final-elast"

# get betas from the estimated model
betas = biogeme.estimate().get_beta_values()

# base choice probabilities 
obsprob_hv = models.logit(v, av, 1)
condprobIndiv_hv = PanelLikelihoodTrajectory(obsprob_hv)
logprob_hv = log(MonteCarlo(condprobIndiv_hv))
obsprob_av = models.logit(v, av, 2)
condprobIndiv_av = PanelLikelihoodTrajectory(obsprob_av)
logprob_av = log(MonteCarlo(condprobIndiv_av))
obsprob_avwl = models.logit(v, av, 3)
condprobIndiv_avwl = PanelLikelihoodTrajectory(obsprob_avwl)
logprob_avwl = log(MonteCarlo(condprobIndiv_avwl))


# function to get mean changes in probabilities between a scenario and the base
def get_probability_changes(scenario):
    
    # utility
    v = get_utility()

    # availability of each alternatives
    av = {1: hv_av, 2: av_av, 3: avwl_av}

    # after choice probabilities
    obsprob_hv_scen = models.logit(v, av, 1)
    condprobIndiv_hv_scen = PanelLikelihoodTrajectory(obsprob_hv_scen)
    logprob_hv_scen = log(MonteCarlo(condprobIndiv_hv_scen))
    obsprob_av_scen = models.logit(v, av, 2)
    condprobIndiv_av_scen = PanelLikelihoodTrajectory(obsprob_av_scen)
    logprob_av_scen = log(MonteCarlo(condprobIndiv_av_scen))
    obsprob_avwl_scen = models.logit(v, av, 3)
    condprobIndiv_avwl_scen = PanelLikelihoodTrajectory(obsprob_avwl_scen)
    logprob_avwl_scen = log(MonteCarlo(condprobIndiv_avwl_scen))

    # set up to get the simulated outputs
    simulate = {
        'prob_hv': exp(logprob_hv),
        'prob_hv_scen': exp(logprob_hv_scen),
        'prob_av': exp(logprob_av),
        'prob_av_scen': exp(logprob_av_scen),
        'prob_avwl': exp(logprob_avwl),
        'prob_avwl_scen': exp(logprob_avwl_scen),
    }

    # simulate the model
    biosim = bio.BIOGEME(database, simulate)
    biosim.modelName = '4-6-mxl-IV-final-elast-sim'
    sim_result = biosim.simulate(the_beta_values=betas)

    # calculate % change in probabilites
    sim_result["per_change_hv"] = (sim_result["prob_hv_scen"] - sim_result["prob_hv"])/sim_result["prob_hv"]
    sim_result["per_change_av"] = (sim_result["prob_av_scen"] - sim_result["prob_av"])/sim_result["prob_av"]
    sim_result["per_change_avwl"] = (sim_result["prob_avwl_scen"] - sim_result["prob_avwl"])/sim_result["prob_avwl"]

    # mean changes
    change_hv = round(sim_result["per_change_hv"].mean() * 100, 2)
    change_av = round(sim_result["per_change_av"].mean() * 100, 2)
    change_avwl = round(sim_result["per_change_avwl"].mean() * 100, 2)

    # print
    logging.info(f"\nScenario: {scenario}...")
    logging.info(f"    Average increase in HV choice prob = {change_hv}%")
    logging.info(f"    Average increase in AV choice prob = {change_av}%")
    logging.info(f"    Average increase in AVWL choice prob = {change_avwl}%")

# when TBA difference increased by 1
scenario = "tba_tot_diff increased by 1"
change = +1
tba_tot_diff = tba_tot_diff + change
database.variables["tba_tot_diff"] = tba_tot_diff
get_probability_changes(scenario)
tba_tot_diff = tba_tot_diff - change
database.variables["tba_tot_diff"] = tba_tot_diff

# when TTU difference increased by 1 
# (1 SD, as the variable is standarized)
scenario = "ttu_diff increased by 1"
change = +1
ttu_diff = ttu_diff + change
database.variables["ttu_diff"] = ttu_diff
get_probability_changes(scenario)
ttu_diff = ttu_diff - change
database.variables["ttu_diff"] = ttu_diff

# when av_fam increased by 1 
# (1 SD, as the variable is standarized)
scenario = "av_fam increased by 1"
change = +1
av_fam = av_fam + change
database.variables["av_fam"] = av_fam
get_probability_changes(scenario)
av_fam = av_fam - change
database.variables["av_fam"] = av_fam

# when all invidiuals were considered 65+ years old
scenario = "all invidiuals were considered 65+ years old"
age_grp_2_org = age_grp_2
age_grp_3_org = age_grp_3
age_grp_2 = 0
age_grp_3 = 1
database.variables["age_grp_2"] = age_grp_2
database.variables["age_grp_3"] = age_grp_3
get_probability_changes(scenario)
database.variables["age_grp_2"] = age_grp_2_org
database.variables["age_grp_3"] = age_grp_3_org

# when av_usefulness increased by 1 
# (1 SD, as the variable is standarized)
scenario = "av_usefulness increased by 1"
change = +1
av_usefulness = av_usefulness + change
database.variables["av_usefulness"] = av_usefulness
get_probability_changes(scenario)
av_usefulness = av_usefulness - change
database.variables["av_usefulness"] = av_usefulness

# when polychronicity increased by 1 
# (1 SD, as the variable is standarized)
scenario = "polychronicity increased by 1"
change = +1
polychronicity = polychronicity + change
database.variables["polychronicity"] = polychronicity
get_probability_changes(scenario)
polychronicity = polychronicity - change
database.variables["polychronicity"] = polychronicity

# when envt_concern increased by 1 
# (1 SD, as the variable is standarized)
scenario = "envt_concern increased by 1"
change = +1
envt_concern = envt_concern + change
database.variables["envt_concern"] = envt_concern
get_probability_changes(scenario)
envt_concern = envt_concern - change
database.variables["envt_concern"] = envt_concern

# when avwl_tc decreased by 20%
scenario = "avwl_tc decreased by 20%"
change = 0.8
avwl_tc = avwl_tc * change
database.variables["avwl_tc"] = avwl_tc
get_probability_changes(scenario)
avwl_tc = avwl_tc / change
database.variables["avwl_tc"] = avwl_tc

# when envt_concern set to sample Q3 
scenario = "envt_concern set to sample Q3"
envt_concern_org = envt_concern
envt_concern = 0.81
database.variables["envt_concern"] = envt_concern
get_probability_changes(scenario)
database.variables["envt_concern"] = envt_concern_org

# when envt_concern set to sample Q3 and above
scenario = "envt_concern set to sample Q3 and above"
envt_concern_org = envt_concern
envt_concern = Variable("envt_concern_scen")
database.variables["envt_concern"] = envt_concern
get_probability_changes(scenario)
database.variables["envt_concern"] = envt_concern_org

# when av_usefulness set to sample Q3 
scenario = "av_usefulness set to sample Q3"
av_usefulness_org = av_usefulness
av_usefulness = 0.77
database.variables["av_usefulness"] = av_usefulness
get_probability_changes(scenario)
database.variables["av_usefulness"] = av_usefulness_org

# when av_usefulness set to sample Q3 and above
scenario = "av_usefulness set to sample Q3 and above"
av_usefulness_org = av_usefulness
av_usefulness = Variable("av_usefulness_scen")
database.variables["av_usefulness"] = av_usefulness
get_probability_changes(scenario)
database.variables["av_usefulness"] = av_usefulness_org

# when polychronicity set to sample Q3 
scenario = "polychronicity set to sample Q3"
polychronicity_org = polychronicity
polychronicity = 0.70
database.variables["polychronicity"] = polychronicity
get_probability_changes(scenario)
database.variables["polychronicity"] = polychronicity_org

# when polychronicity set to sample Q3 and above
scenario = "polychronicity set to sample Q3 and above"
polychronicity_org = polychronicity
polychronicity = Variable("polychronicity_scen")
database.variables["polychronicity"] = polychronicity
get_probability_changes(scenario)
database.variables["polychronicity"] = polychronicity_org

# remove intermediate outputs
os.remove("4-6-mxl-IV-final-elast.html")
os.remove("4-6-mxl-IV-final-elast.pickle")

# ---------------------------------------------------------------------------------------------------------------------#
