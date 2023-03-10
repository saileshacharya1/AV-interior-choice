# ---------------------------------------------------------------------------------------------------------------------#
# Author: Sailesh Acharya
# Data: 2023-02-24
# Project name: Choice analysis of vehicle interior design in case of conventional and autonomous vehicles.

# This script prepares the data for choice analysis using Biogeme. The data is collected from a survey
# of US national park visitors.
# ---------------------------------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------------------------------#
# import modules
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------------------------------#
# import data
df = pd.read_csv("../data/data.csv")
df.describe

# convert wide to long format
value_vars = list(df.filter(like="stated_pref").columns)
id_vars = list(df.columns.drop(value_vars))
df = pd.melt(
    df, id_vars=id_vars, value_vars=value_vars, var_name="scenario", value_name="chosen"
)

# drop scenarios without chosen values
df = df[df["chosen"].notnull()]

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
        "av_fam",
    ],
)

# pepare data for biogeme with no null values and sorted by id
df = df.drop(df.columns[df.isnull().any()], axis=1)
df = df.select_dtypes(exclude=["object"])
df = df.sort_values("id")

# add column of difference in travel usefulness between AV and HV
df["tu_diff"] = df["tu_av"] - df["tu_hv"]

# export the prepared dataset
df.to_pickle("../data/prepared_data.pkl")
# ---------------------------------------------------------------------------------------------------------------------#
