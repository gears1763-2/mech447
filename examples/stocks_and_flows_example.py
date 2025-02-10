"""
    Anthony Truelove MASc, P.Eng.  
    Python Certified Professional Programmer (PCPP1)  
    2025

    This is an example of using the `mech447` package to solve question 3 of
    assignment 1.
"""

# ==== IMPORTS ============================================================== #

import os
import sys
sys.path.append("../")  # <-- just point to where the `mech447` package folder
                        #     is (either relative or absolute path)

import numpy as np

import mech447.stockflow as sf  # <-- this is importing from the
                                #     `mech447` package


# ==== MODELLING INPUTS ===================================================== #

#   1. define modelling inputs

#   1.1. model over a 100-year horizon, in steps of 1/1000 of a year
n_years = 100
modelling_density = 1000
time_array_years = np.linspace(0, n_years, modelling_density * n_years)

#   1.2. define constants
alpha_per_year = 0.03
mu_per_year = -0.02

P_0_Mt_per_year = 5
total_resource_Mt = 1000


# ==== PART A =============================================================== #

#   2. set up, run, and plot results for part a
F_0_Mt_per_year = 100

unconstrained_addition_array_Mt_per_year = (
    F_0_Mt_per_year * np.exp(mu_per_year * time_array_years)
)

unconstrained_source_array_Mt_per_year = np.zeros(len(time_array_years))

unconstrained_production_array_Mt_per_year = (
    P_0_Mt_per_year * np.exp(alpha_per_year * time_array_years)
)

reserve_initial_Mt = 0
maximum_reserve_Mt = np.inf

stocks_and_flows_q3a = sf.StocksAndFlows(
    time_array_years,
    unconstrained_addition_array_Mt_per_year,
    unconstrained_source_array_Mt_per_year,
    unconstrained_production_array_Mt_per_year,
    total_resource_units=total_resource_Mt,
    reserve_initial_units=reserve_initial_Mt,
    maximum_reserve_units=maximum_reserve_Mt,
    units_str="Mt"
)

stocks_and_flows_q3a.run()
stocks_and_flows_q3a.plot(
    show_flag=True,
    save_flag=True,
    save_path=""
)
print()


# ==== PART B =============================================================== #

#   2. set up, run, and plot results for part b
F_0_Mt_per_year = 20

unconstrained_addition_array_Mt_per_year = (
    F_0_Mt_per_year * np.exp(mu_per_year * time_array_years)
)

unconstrained_source_array_Mt_per_year = np.zeros(len(time_array_years))

unconstrained_production_array_Mt_per_year = (
    P_0_Mt_per_year * np.exp(alpha_per_year * time_array_years)
)

reserve_initial_Mt = 0
maximum_reserve_Mt = np.inf

stocks_and_flows_q3b = sf.StocksAndFlows(
    time_array_years,
    unconstrained_addition_array_Mt_per_year,
    unconstrained_source_array_Mt_per_year,
    unconstrained_production_array_Mt_per_year,
    total_resource_units=total_resource_Mt,
    reserve_initial_units=reserve_initial_Mt,
    maximum_reserve_units=maximum_reserve_Mt,
    units_str="Mt"
)

stocks_and_flows_q3b.run()
stocks_and_flows_q3b.plot(
    show_flag=True,
    save_flag=True,
    save_path=""
)
print()


# ==== CLEAN UP ============================================================= #

#   3. clean up
input("Press [enter] to clean up figures ...")

for _, _, filename_list in os.walk(os.getcwd()):
    for filename in filename_list:
        print(filename)
        if ".png" in filename:
            os.remove(filename)
