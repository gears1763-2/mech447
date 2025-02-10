"""
    Anthony Truelove MASc, P.Eng.  
    Python Certified Professional Programmer (PCPP1)  
    2025

    This is an example of using the `mech447` package to solve the questions
    of assignment 3.
"""

# ==== IMPORTS ============================================================== #

import os
import sys
sys.path.append("../")  # <-- just point to where the `mech447` package folder
                        #     is (either relative or absolute path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mech447.mixtureplanner as mp    # <-- this is importing from the
                                       #     `mech447` package


# ==== CONSTANTS ============================================================ #

PATH_2_EXAMPLE_DATA = "example_data/mixture_example_data.csv"


# ==== SCALAR INPUTS ======================================================== #

installed_solar_capacity_MW = 200
installed_wind_capacity_MW = 200


# ==== ARRAY INPUTS ========================================================= #

#   1. load time series data into pandas DataFrame
time_series_dataframe = pd.read_csv(PATH_2_EXAMPLE_DATA)
feature_list = list(time_series_dataframe)
print(feature_list)

#   2. extract time series arrays
time_array_hrs = time_series_dataframe[feature_list[0]].values
demand_array_MW = time_series_dataframe[feature_list[1]].values
solar_production_per_MWc_MW = time_series_dataframe[feature_list[2]].values
wind_production_per_MWc_MW = time_series_dataframe[feature_list[3]].values

#   3. construct screening curve dict (approximated from given plot)
screening_curve_dict_CAD_MWc_yr = {
    "Coal": 1000 * (105 * np.linspace(0, 1, 1000) + 140),
    "Gas": 1000 * (325 * np.linspace(0, 1, 1000) + 50),
    "Combined Cycle": 1000 * (95 * np.linspace(0, 1, 1000) + 155)
}

#   4. init renewable production dict
renewable_production_dict_MW = {}


# ==== QUESTION 1 =========================================================== #

#   4. init MixturePlanner
mixture_planner_q1 = mp.MixturePlanner(
    time_array_hrs,
    demand_array_MW,
    renewable_production_dict_MW,
    screening_curve_dict_CAD_MWc_yr,
    power_units_str="MW"
)

#   5. run mixture planner, print, plot
mixture_planner_q1.run()
print(mixture_planner_q1)

mixture_planner_q1.plot(
    show_flag=False,
    save_flag=False
)


# ==== QUESTION 2 =========================================================== #

#   6. populate renewable production dict
renewable_production_dict_MW["Solar"] = (
    installed_solar_capacity_MW
    * solar_production_per_MWc_MW
)

renewable_production_dict_MW["Wind"] = (
    installed_wind_capacity_MW
    * wind_production_per_MWc_MW
)

#   7. init MixturePlanner
mixture_planner_q2 = mp.MixturePlanner(
    time_array_hrs,
    demand_array_MW,
    renewable_production_dict_MW,
    screening_curve_dict_CAD_MWc_yr,
    power_units_str="MW"
)

#   8. run mixture planner, print, plot
mixture_planner_q2.run()
print(mixture_planner_q2)

mixture_planner_q2.plot(
    show_flag=False,
    save_flag=False
)


# ==== QUESTION 3 =========================================================== #

#   9. reset renewable production dict
renewable_production_dict_MW = {
    "Wind": 3 * installed_wind_capacity_MW * wind_production_per_MWc_MW
}

#   10. init MixturePlanner
mixture_planner_q3 = mp.MixturePlanner(
    time_array_hrs,
    demand_array_MW,
    renewable_production_dict_MW,
    screening_curve_dict_CAD_MWc_yr,
    power_units_str="MW"
)

#   11. run mixture planner, print, plot
mixture_planner_q3.run()
print(mixture_planner_q3)

mixture_planner_q3.plot(
    show_flag=False,
    save_flag=True
)


# ==== CLEAN UP ============================================================= #

#   8. clean up
input("Press [enter] to clean up figures ...")

for _, _, filename_list in os.walk(os.getcwd()):
    for filename in filename_list:
        print(filename)
        if ".png" in filename:
            os.remove(filename)
