"""
    Anthony Truelove MASc, P.Eng.  
    Python Certified Professional Programmer (PCPP1)  
    2025

    This is an example of using the `mech447` package to do unit price
    forecasting given a deployment time series and unit price learning data.
"""


import sys
sys.path.append("../")  # <-- just point to where the `mech447` package folder is (either relative or absolute path)

import numpy as np

import mech447.unitpriceforecaster as upf  # <-- this is importing from the `mech447` package


#   1. construct cumulative deployment time series
time_array_years = np.linspace(0, 100, 1000 * 100)

deployment_array_units = (
    2500 * (np.tanh(0.1 * (time_array_years - 30)) + 1)
)

#   2. construct unit price learning data
initial_price_per_unit = 2000
learning_exponent = -0.25
learning_array_units = np.array([1, 10, 20, 50, 100, 125, 185, 350, 500])

learning_array_price_per_unit = (
    initial_price_per_unit
    * np.power(
        (1 / learning_array_units[0])
        * learning_array_units, learning_exponent
    )
)

random_array = (
    0.4 * np.random.rand(len(learning_array_price_per_unit))
    + 0.8
)

learning_array_price_per_unit = np.multiply(
    random_array,
    learning_array_price_per_unit
)

#   3. construct unit price forecaster
unit_price_forecaster = upf.UnitPriceForecaster(
    time_array_years,
    deployment_array_units,
    learning_array_units,
    learning_array_price_per_unit,
    currency_str = "CAD",
    units_str = "MW"
)

#   4. run and plot results
unit_price_forecaster.run()
unit_price_forecaster.plot()
