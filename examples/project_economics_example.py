"""
    Anthony Truelove MASc, P.Eng.  
    Python Certified Professional Programmer (PCPP1)  
    2025

    This is an example of using the `mech447` package to solve question 4 of
    assignment 2.
"""

# ==== IMPORTS ============================================================== #

import os
import sys
sys.path.append("../")  # <-- just point to where the `mech447` package folder
                        #     is (either relative or absolute path)

import matplotlib.pyplot as plt
import numpy as np

import mech447.projecteconomics as pe  # <-- this is importing from the
                                       #     `mech447` package


# ==== CONSTANTS ============================================================ #

GJ_PER_MMBTU = 1.0551
KWH_PER_MWH = 1000
HOURS_PER_YEAR = 8760


# ==== SCALAR INPUTS ======================================================== #

#   1. define inputs
plant_life_years = 30
discount_rate_annual = 0.08

capacity_kW = 1
print("capacity_kW:", capacity_kW)
print()

#   1.1. build cost
overnight_build_cost = 1092 * capacity_kW
build_time_years = 2
print("overnight_build_cost:", overnight_build_cost, "CAD")
print()

#   1.2. annual fixed operation and maintenance
fixed_OnM_annual = 17.39 * capacity_kW
print("fixed_OnM_annual:", round(fixed_OnM_annual, 2), "CAD/yr")
print()

#   1.3. annual variable operation and maintenance
annual_capacity_factor = 0.25
variable_OnM_MWh = 3.48

annual_generation_MWh = (
    annual_capacity_factor
    * HOURS_PER_YEAR
    * capacity_kW
    * (1 / KWH_PER_MWH)
)
print("annual_generation_MWh:", annual_generation_MWh)
print()

variable_OnM_annual = annual_generation_MWh * variable_OnM_MWh
print("variable_OnM_annual:", round(variable_OnM_annual, 2), "CAD/yr")
print()

#   1.4. annual fuel cost
heat_rate_MMBtu_MWh = 9.6
fuel_cost_GJ = 4

annual_fuel_consumption_GJ = (
    annual_generation_MWh
    * heat_rate_MMBtu_MWh
    * GJ_PER_MMBTU
)

print(
    "annual_fuel_consumption_GJ:",
    round(annual_fuel_consumption_GJ, 2),
    "GJ/yr"
)
print()

fuel_cost_annual = annual_fuel_consumption_GJ * fuel_cost_GJ
print("fuel_cost_annual:", round(fuel_cost_annual, 2), "CAD/yr")
print()


# ==== ARRAY INPUTS ========================================================= #

#   2. prep input arrays

#   2.1. period array
period_array_years = np.array(
    [
        -1 * build_time_years + i for
        i in range(0, plant_life_years + build_time_years + 1)
    ]
)
print("period_array_years:", period_array_years)
print()

#   2.2. nominal expense arrays

#   2.2.1. nominal build expenses
nominal_build_expense_array = np.zeros(len(period_array_years))

for i in range(0, build_time_years):
    nominal_build_expense_array[i] = overnight_build_cost / build_time_years

print("nominal_build_expense_array:", nominal_build_expense_array)
print()

#   2.2.2. nominal fixed operation and maintenance expenses
nominal_fixed_OnM_expense_array = np.zeros(len(period_array_years))

for i in range(build_time_years + 1, len(period_array_years)):
    nominal_fixed_OnM_expense_array[i] = fixed_OnM_annual

print("nominal_fixed_OnM_expense_array:", nominal_fixed_OnM_expense_array)
print()

#   2.2.3. nominal variable operation and maintenance expenses
nominal_variable_OnM_expense_array = np.zeros(len(period_array_years))

for i in range(build_time_years + 1, len(period_array_years)):
    nominal_variable_OnM_expense_array[i] = variable_OnM_annual

print("nominal_variable_OnM_expense_array:", nominal_variable_OnM_expense_array)
print()

#   2.2.4. nominal fuel expenses
nominal_fuel_expense_array = np.zeros(len(period_array_years))

for i in range(build_time_years + 1, len(period_array_years)):
    nominal_fuel_expense_array[i] = fuel_cost_annual

print("nominal_fuel_expense_array:", nominal_fuel_expense_array)
print()

#   2.2. nominal income arrays
nominal_income_array = np.zeros(len(period_array_years))

print("nominal_income_array:", nominal_income_array)
print()


# ==== PART A =============================================================== #
print()
print("# ==== PART A ====================================================== #")
print()

#   3. set up ProjectEconomics instance
nominal_expense_array_dict_a = {
    "Build Expenses": nominal_build_expense_array,
    "Fixed O&M Expenses": nominal_fixed_OnM_expense_array,
    "Variable O&M Expenses": nominal_variable_OnM_expense_array,
    "Fuel Expenses": nominal_fuel_expense_array
}

nominal_income_array_dict = {
    "Example Income": nominal_income_array
}

project_economics_a = pe.ProjectEconomics(
    period_array_years,
    nominal_expense_array_dict_a,
    nominal_income_array_dict,
    discount_rate_annual,
    period_str="year",
    currency_str="CAD"
)

#   4. run, print, and plot
project_economics_a.run()
project_economics_a.printKeyMetrics()
project_economics_a.plot(
    show_flag=True,
    save_flag=True,
    save_path=""
)


# ==== PART B =============================================================== #
print()
print()
print("# ==== PART B ====================================================== #")
print()

#   5. construct nominal carbon expense array
carbon_intensity_tCO2_GJ = 0.050

carbon_tax_array_tCO2 = np.linspace(50, 100, 30)
print("carbon_tax_array_tCO2:", carbon_tax_array_tCO2)
print()

nominal_carbon_expense_array = np.zeros(len(period_array_years))

for i in range(build_time_years + 1, len(period_array_years)):
    nominal_carbon_expense_array[i] = (
        carbon_tax_array_tCO2[i - (build_time_years + 1)]
        * annual_fuel_consumption_GJ
        * carbon_intensity_tCO2_GJ
    )

print("nominal_carbon_expense_array:", nominal_carbon_expense_array)
print()


#   6. set up ProjectEconomics instance
nominal_expense_array_dict_b = {
    "Build Expenses": nominal_build_expense_array,
    "Fixed O&M Expenses": nominal_fixed_OnM_expense_array,
    "Variable O&M Expenses": nominal_variable_OnM_expense_array,
    "Fuel Expenses": nominal_fuel_expense_array,
    "Carbon Tax Expenses": nominal_carbon_expense_array
}

nominal_income_array_dict = {
    "Example Income": nominal_income_array
}

project_economics_b = pe.ProjectEconomics(
    period_array_years,
    nominal_expense_array_dict_b,
    nominal_income_array_dict,
    discount_rate_annual,
    period_str="year",
    currency_str="CAD"
)

#   7. run, print, and plot
project_economics_b.run()
project_economics_b.printKeyMetrics()
project_economics_b.plot(
    show_flag=True,
    save_flag=True,
    save_path=""
)


# ==== CLEAN UP ============================================================= #

#   8. clean up
input("Press [enter] to clean up figures ...")

for _, _, filename_list in os.walk(os.getcwd()):
    pass

for filename in filename_list:
    if ".png" in filename:
        os.remove(filename)

