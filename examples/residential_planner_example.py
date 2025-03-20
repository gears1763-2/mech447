"""
    Anthony Truelove MASc, P.Eng.  
    Python Certified Professional Programmer (PCPP1)  
    2025

    This is an example of using the `mech447` package to solve question 4 of
    assignment 5.
"""

# ==== IMPORTS ============================================================== #

import os
import sys
sys.path.append("../")  # <-- just point to where the `mech447` package folder
                        #     is (either relative or absolute path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mech447.residentialplanner as rp    # <-- this is importing from the
                                           #     `mech447` package


# ==== CONSTANTS ============================================================ #

PATH_2_EXAMPLE_DATA = "example_data/PS5_Hypothetical_Residence.csv"


# ==== SCALAR INPUTS ======================================================== #

peak_load_kW = 8
solar_capacity_kW = 6

storage_energy_capacity_kWh = 10
storage_initial_SOC = 0
storage_charging_efficiency = 0.9
storage_discharging_efficiency = 0.9



# ==== ARRAY INPUTS ========================================================= #

#   1. load time series data into pandas DataFrame
time_series_dataframe = pd.read_csv(PATH_2_EXAMPLE_DATA)
feature_list = list(time_series_dataframe)
print(feature_list)

#   2. extract time series arrays
time_array_hours = time_series_dataframe[feature_list[0]].values

solar_production_array_kW = (
    solar_capacity_kW * time_series_dataframe[feature_list[1]].values
)

demand_array_kW = peak_load_kW * time_series_dataframe[feature_list[2]].values

#   3. construct energy tariffs dict [kW]: [$/kWh]
energy_tariffs_dict = {
    -1 * np.inf: -0.07,
    0: 0.07,
    5: 0.11
}


# ==== MAIN ================================================================= #

#   4. construct renewable list
solar_6kW = rp.Renewable(
    solar_production_array_kW,
    type_str="6 kW Solar"
)

renewable_list = [solar_6kW]

#   5. construct storage
storage_10kWh = rp.Storage(
    storage_energy_capacity_kWh,
    power_capacity=np.inf,
    initial_state_of_charge=storage_initial_SOC,
    charging_efficiency=storage_charging_efficiency,
    discharging_efficiency=storage_discharging_efficiency
)

#   6. init ResidentialPlanner
residential_planner = rp.ResidentialPlanner(
    time_array_hours,
    demand_array_kW,
    energy_tariffs_dict,
    renewable_list=renewable_list,
    storage=storage_10kWh,
    power_units_str="kW",
    currency_units_str="CAD"
)

#   7. run residential planner, print, plot
residential_planner.run()
print(residential_planner)

residential_planner.plot(
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
