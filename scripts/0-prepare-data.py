# ---------------------------------------------------------------------------------------------------------------------#
# Author: Sailesh Acharya
# Data: 2025-01-01
# Project name: Choice analysis of vehicle interior design in case of conventional and autonomous vehicles.

# This script prepares the data for choice analysis using Biogeme.
# The data is collected from a survey of US national park visitors.
# ---------------------------------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------------------------------#
# import modules
import pandas as pd
import numpy as np
import semopy
from sklearn.preprocessing import StandardScaler


# import data
df = pd.read_csv("../data/data.csv")
df.describe

# ---------------------------------------------------------------------------------------------------------------------#
## latent variables definition and score estimate

# model specification
mod = """
av_usefulness     =~ av_benefit_1 + av_benefit_2 + av_benefit_3 + \
                     av_benefit_4 + av_benefit_5 + av_benefit_6 + \
                     av_concern_1 + av_concern_4 + av_concern_5
av_concern        =~ av_concern_2 + av_concern_3 + av_concern_6 + \
                     av_concern_7
tech_savviness    =~ tech_savvy_1  + tech_savvy_3
driving_enjoyment =~ enjoy_driving_1 + enjoy_driving_3 + enjoy_driving_4
polychronicity    =~ polychronicity_1 + polychronicity_2 + polychronicity_3
envt_concern      =~ envt_concern_1 + envt_concern_2 + envt_concern_3
"""

# model fit
model = semopy.Model(mod)
model.fit(df, obj="DWLS")
pd.set_option("display.max_rows", 500)
model.inspect(std_est=True)
stats = semopy.calc_stats(model)
print(stats.T)

# estimate factor scores
factors = model.predict_factors(df)
df = df.join(factors)

# standarized latent variable scores
columns = [
    "av_concern",
    "av_usefulness",
    "driving_enjoyment",
    "envt_concern",
    "polychronicity",
    "tech_savviness",
]
standardized_values = StandardScaler().fit_transform(df[columns])
for i, col in enumerate(columns):
    df[f"{col}_std"] = standardized_values[:, i]


# add several columns of low, medium, and high latent factors' scores
df["av_usefulness_cat"] = pd.qcut(
    df["av_usefulness"], [0, 0.333, 0.666, 1], labels=["low", "med", "high"]
)
df["av_concern_cat"] = pd.qcut(
    df["av_concern"], [0, 0.333, 0.666, 1], labels=["low", "med", "high"]
)
df["tech_savviness_cat"] = pd.qcut(
    df["tech_savviness"], [0, 0.333, 0.666, 1], labels=["low", "med", "high"]
)
df["driving_enjoyment_cat"] = pd.qcut(
    df["driving_enjoyment"], [0, 0.333, 0.666, 1], labels=["low", "med", "high"]
)
df["polychronicity_cat"] = pd.qcut(
    df["polychronicity"], [0, 0.333, 0.666, 1], labels=["low", "med", "high"]
)
df["envt_concern_cat"] = pd.qcut(
    df["envt_concern"], [0, 0.333, 0.666, 1], labels=["low", "med", "high"]
)

# latent variables for scenarios
q3 = df["envt_concern_std"].quantile(0.75)
df["envt_concern_scen"] = np.where(df["envt_concern_std"] < q3, q3, df["envt_concern_std"])
q3 = df["av_usefulness_std"].quantile(0.75)
df["av_usefulness_scen"] = np.where(df["av_usefulness_std"] < q3, q3, df["av_usefulness_std"])
q3 = df["polychronicity_std"].quantile(0.75)
df["polychronicity_scen"] = np.where(df["polychronicity_std"] < q3, q3, df["polychronicity_std"])

# ---------------------------------------------------------------------------------------------------------------------#
## calculate tba and ttu scores for hv and av

# hv-tba scores
df["tba_g1_hv"] = df["tba_hv_7"]
df["tba_g2_hv"] = df["tba_hv_5"] + df["tba_hv_6"] + df["tba_hv_10"]
df["tba_g3_hv"] = df["tba_hv_3"] + df["tba_hv_4"]
df["tba_g4_hv"] = df["tba_hv_2"] + df["tba_hv_8"] + df["tba_hv_9"]
df["tba_g5_hv"] = df["tba_hv_11"] + df["tba_hv_12"]
df["tba_g6_hv"] = df["tba_hv_13"] + df["tba_hv_14"] + df["tba_hv_15"]
df["tba_g7_hv"] = df["tba_hv_16"]
df["tba_tot_hv"] = (
    df["tba_hv_1"]
    + df["tba_hv_2"]
    + df["tba_hv_3"]
    + df["tba_hv_4"]
    + df["tba_hv_5"]
    + df["tba_hv_6"]
    + df["tba_hv_7"]
    + df["tba_hv_8"]
    + df["tba_hv_9"]
    + df["tba_hv_10"]
    + df["tba_hv_11"]
    + df["tba_hv_12"]
    + df["tba_hv_13"]
    + df["tba_hv_14"]
    + df["tba_hv_15"]
    + df["tba_hv_16"]
    + df["tba_hv_17"]
)


# av-tba scores
df["tba_g1_av"] = df["tba_av_7"]
df["tba_g2_av"] = df["tba_av_5"] + df["tba_av_6"] + df["tba_av_10"]
df["tba_g3_av"] = df["tba_av_3"] + df["tba_av_4"]
df["tba_g4_av"] = df["tba_av_2"] + df["tba_av_8"] + df["tba_av_9"]
df["tba_g5_av"] = df["tba_av_11"] + df["tba_av_12"]
df["tba_g6_av"] = df["tba_av_13"] + df["tba_av_14"] + df["tba_av_15"]
df["tba_g7_av"] = df["tba_av_16"]
df["tba_tot_av"] = (
    df["tba_av_1"]
    + df["tba_av_2"]
    + df["tba_av_3"]
    + df["tba_av_4"]
    + df["tba_av_5"]
    + df["tba_av_6"]
    + df["tba_av_7"]
    + df["tba_av_8"]
    + df["tba_av_9"]
    + df["tba_av_10"]
    + df["tba_av_11"]
    + df["tba_av_12"]
    + df["tba_av_13"]
    + df["tba_av_14"]
    + df["tba_av_15"]
    + df["tba_av_16"]
    + df["tba_av_17"]
)
# categorical grouped tba scores
df["tba_g1_hv_cat"] = (df["tba_g1_hv"] > 0).astype(int)
df["tba_g2_hv_cat"] = (df["tba_g2_hv"] > 0).astype(int)
df["tba_g3_hv_cat"] = (df["tba_g3_hv"] > 0).astype(int)
df["tba_g4_hv_cat"] = (df["tba_g4_hv"] > 0).astype(int)
df["tba_g5_hv_cat"] = (df["tba_g5_hv"] > 0).astype(int)
df["tba_g6_hv_cat"] = (df["tba_g6_hv"] > 0).astype(int)
df["tba_g7_hv_cat"] = (df["tba_g7_hv"] > 0).astype(int)
df["tba_g1_av_cat"] = (df["tba_g1_av"] > 0).astype(int)
df["tba_g2_av_cat"] = (df["tba_g2_av"] > 0).astype(int)
df["tba_g3_av_cat"] = (df["tba_g3_av"] > 0).astype(int)
df["tba_g4_av_cat"] = (df["tba_g4_av"] > 0).astype(int)
df["tba_g5_av_cat"] = (df["tba_g5_av"] > 0).astype(int)
df["tba_g6_av_cat"] = (df["tba_g6_av"] > 0).astype(int)
df["tba_g7_av_cat"] = (df["tba_g7_av"] > 0).astype(int)


# difference in av-tba and hv-tba scores
for i in range(1, 8):
    df[f"tba_g{i}_diff"] = df[f"tba_g{i}_av"] - df[f"tba_g{i}_hv"]
del i
df["tba_tot_diff"] = df["tba_tot_av"] - df["tba_tot_hv"]

# difference in travel usefulness between AV and HV
df["ttu_diff"] = df["tu_av"] - df["tu_hv"]

# standarize ttu variables
df["ttu_hv_std"] = StandardScaler().fit_transform(df[["tu_hv"]])
df["ttu_av_std"] = StandardScaler().fit_transform(df[["tu_av"]])
df["ttu_diff_std"] = StandardScaler().fit_transform(df[["ttu_diff"]])

# standarize av familiarity
df["av_fam_std"] = StandardScaler().fit_transform(df[["av_fam"]])


# ---------------------------------------------------------------------------------------------------------------------#
## data preparation for choice models

# convert wide to long format
value_vars = list(df.filter(like="stated_pref").columns)
id_vars = list(df.columns.drop(value_vars))
df = pd.melt(
    df, id_vars=id_vars, value_vars=value_vars, var_name="scenario", value_name="chosen"
)

# drop scenarios without chosen values
df = df[df["chosen"].notnull()]

# convert travel time to minutes from hours
# df["time"] = df["time"] * 60

# add travel time and travel cost columns by translating choice scenarios
# choices: current vehicle (hv-1); autonomous vehicle with current vehilce interior(av-2);
# autonomous vehicle with work and leisure interior (avwl-3)

# conditons
conditions = [
    df["scenario"].eq("stated_pref_1"),
    df["scenario"].eq("stated_pref_2"),
    df["scenario"].eq("stated_pref_3"),
    df["scenario"].eq("stated_pref_4"),
    df["scenario"].eq("stated_pref_5"),
    df["scenario"].eq("stated_pref_6"),
    df["scenario"].eq("stated_pref_7"),
    df["scenario"].eq("stated_pref_8"),
    df["scenario"].eq("stated_pref_9"),
    df["scenario"].eq("stated_pref_10"),
    df["scenario"].eq("stated_pref_11"),
    df["scenario"].eq("stated_pref_12"),
]

# travel time options for hv
choices_hv_tt = [
    df["time"],
    df["time"] * 1.2,
    df["time"],
    df["time"] * 0.8,
    df["time"],
    df["time"],
    df["time"] * 1.2,
    df["time"] * 0.8,
    df["time"] * 1.2,
    df["time"] * 0.8,
    df["time"] * 1.2,
    df["time"] * 0.8,
]

# travel cost options for hv
choices_hv_tc = [
    df["cost"],
    df["cost"],
    df["cost"] * 1.2,
    df["cost"] * 0.8,
    df["cost"],
    df["cost"] * 1.2,
    df["cost"] * 0.8,
    df["cost"],
    df["cost"] * 0.8,
    df["cost"] * 0.8,
    df["cost"] * 1.2,
    df["cost"] * 1.2,
]

# travel time options for av
choices_av_tt = [
    df["time"] * 1.2,
    df["time"],
    df["time"] * 0.8,
    df["time"],
    df["time"] * 1.2,
    df["time"] * 0.8,
    df["time"] * 0.8,
    df["time"] * 1.2,
    df["time"],
    df["time"],
    df["time"],
    df["time"],
]

# travel cost options for av
choices_av_tc = [
    df["cost"] * 1.2,
    df["cost"],
    df["cost"] * 0.8,
    df["cost"],
    df["cost"] * 0.8,
    df["cost"] * 1.2,
    df["cost"],
    df["cost"],
    df["cost"] * 0.8,
    df["cost"] * 1.2,
    df["cost"] * 1.2,
    df["cost"] * 0.8,
]

# travel time options for avwl
choices_avwl_tt = [
    df["time"],
    df["time"] * 1.2,
    df["time"],
    df["time"] * 0.8,
    df["time"],
    df["time"],
    df["time"] * 1.2,
    df["time"] * 0.8,
    df["time"] * 0.8,
    df["time"] * 1.2,
    df["time"] * 0.8,
    df["time"] * 1.2,
]

# travel cost options for avwl
choices_avwl_tc = [
    df["cost"] * 1.2,
    df["cost"] * 0.8,
    df["cost"] * 1.2,
    df["cost"] * 0.8,
    df["cost"] * 0.8,
    df["cost"] * 0.8,
    df["cost"] * 1.2,
    df["cost"] * 1.2,
    df["cost"],
    df["cost"],
    df["cost"],
    df["cost"],
]

# travel time and cost for choices
df["hv_tt"] = np.select(conditions, choices_hv_tt)
df["hv_tc"] = np.select(conditions, choices_hv_tc)
df["av_tt"] = np.select(conditions, choices_av_tt)
df["av_tc"] = np.select(conditions, choices_av_tc)
df["avwl_tt"] = np.select(conditions, choices_avwl_tt)
df["avwl_tc"] = np.select(conditions, choices_avwl_tc)

# availabilities of hv, av, and avwl
df["hv_av"] = 1
df["av_av"] = 1
df["avwl_av"] = 1

# some summary statistics
# df['chosen'] = df['chosen'].astype('category')
# df['chosen'].value_counts()

# dummy coding of some columns having categorical options

df = pd.get_dummies(
    data=df,
    columns=[
        "age_grp",
        "gender",
        "education",
        "school",
        "income_grp",
        "employment",
        "mode_commute",
        "mode_shopping",
        "mode_personal",
        "mode_social",
        "av_usefulness_cat",
        "av_concern_cat",
        "tech_savviness_cat",
        "driving_enjoyment_cat",
        "polychronicity_cat",
        "envt_concern_cat",
    ],
)

# ensure all dummy columns are numeric (convert True/False to 0/1)
df = df.astype({col: int for col in df.columns if df[col].dtype == bool})
print(df.dtypes)


# pepare data for biogeme with no null values and sorted by id
df = df.drop(df.columns[df.isnull().any()], axis=1)
df = df.select_dtypes(exclude=["object"])
df = df.sort_values("id")

# export the prepared dataset
df.to_pickle("../data/prepared_data.pkl")
df.to_csv("../data/prepared_data.csv")


# ---------------------------------------------------------------------------------------------------------------------#
