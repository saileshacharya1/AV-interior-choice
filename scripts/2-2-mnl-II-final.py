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
    filename='../outputs/2-2-mnl-II-final.log',  
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
envt_concern = Variable("envt_concern_std")
av_fam = Variable("av_fam_std")
tba_tot_diff = Variable("tba_tot_diff")
ttu_diff = Variable("ttu_diff_std")

# ---------------------------------------------------------------------------------------------------------------------#
## 2-2-mnl-II-final
# 2-1-mnl-II-initial with significant parameters only 

# parameters to be estimated
# alternative specific constants
asc_av = Beta("asc_av", 0, None, None, 0)
asc_avwl = Beta("asc_avwl", 0, None, None, 0)

# travel time and cost parameters
b_vot_hv = Beta("b_vot_hv", 0, None, None, 0)
b_vot_av = Beta("b_vot_av", 0, None, None, 0)
b_vot_avwl = Beta("b_vot_avwl", 0, None, None, 0)
b_cost = Beta("b_cost", 0, None, None, 0)

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

# asc parameters
asc_av_par = (
    asc_av
    + b_age_grp_2_av * age_grp_2
    + b_age_grp_3_av * age_grp_3
    + b_gender_1_av * gender_1
    + b_school_2_av * school_2
    + b_school_3_av * school_3
    + b_hh_child_av * hh_child
    + b_crash_exp_av * crash_exp
    + b_mode_commute_3_av * mode_commute_3
    + b_mode_social_3_av * mode_social_3
    + b_av_usefulness_av * av_usefulness
    + b_av_concern_av * av_concern
    + b_polychronicity_av * polychronicity
    + b_envt_concern_av * envt_concern
    + b_tba_tot_diff_av * tba_tot_diff
    + b_ttu_diff_av * ttu_diff
)
asc_avwl_par = (
    asc_avwl
    + b_age_grp_2_avwl * age_grp_2
    + b_age_grp_3_avwl * age_grp_3
    + b_gender_1_avwl * gender_1
    + b_school_3_avwl * school_3
    + b_citation_avwl * citation
    + b_crash_exp_avwl * crash_exp
    + b_mode_shopping_3_avwl * mode_shopping_3
    + b_mode_social_3_avwl * mode_social_3
    + b_av_usefulness_avwl * av_usefulness
    + b_av_concern_avwl * av_concern
    + b_envt_concern_avwl * envt_concern
    + b_av_fam_avwl * av_fam
    + b_tba_tot_diff_avwl * tba_tot_diff
    + b_ttu_diff_avwl * ttu_diff
)

# define utility functions
v1 = b_cost * b_vot_hv * hv_tt + b_cost * hv_tc
v2 = asc_av_par + b_cost * b_vot_av * av_tt + b_cost * av_tc
v3 = asc_avwl_par + b_cost * b_vot_avwl * avwl_tt + b_cost * avwl_tc

# link utility functions with the numbering of alternatives
v = {1: v1, 2: v2, 3: v3}

# availability of each alternatives
av = {1: hv_av, 2: av_av, 3: avwl_av}

# define the model
logprog = models.loglogit(v, av, chosen)

# create biogeme object
biogeme = bio.BIOGEME(database, logprog, suggestScales=False)

# model name
biogeme.modelName = "2-2-mnl-II-final"

# calculate null loglikelihood
biogeme.calculateNullLoglikelihood(avail=av)

# get the results in a pandas table
print(biogeme.estimate().getEstimatedParameters())

# move outpts to outputs folder
shutil.move("2-2-mnl-II-final.html", "../outputs/2-2-mnl-II-final.html")

# 5-fold cross-validaiton
validation_data = database.split(slices=5)
validation_results = biogeme.validate(biogeme.estimate(), validation_data)
for slide in validation_results:
    log_message = (
        f'Log likelihood for {slide.shape[0]} validation data: '
        f'{slide["Loglikelihood"].sum()}'
    )
    logging.info(log_message)  

# delete intermediate and unnecessary files
extensions = ["*.html", "*.pickle", "*.iter"]
for ext in extensions:
    for file_path in glob.glob(os.path.join(ext)):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# ---------------------------------------------------------------------------------------------------------------------#
