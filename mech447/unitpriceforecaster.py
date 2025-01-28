"""
    Anthony Truelove MASc, P.Eng.  
    Python Certified Professional Programmer (PCPP1)  
    2025

    A unit price forecaster class, as part of the `mech447` package.
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi
import sklearn.linear_model as skllm


class UnitPriceForecaster:
    """
    A class which contains modelled unit price values (under a learning model)
    over a modelling horizon.
    """
    
    def __init__(
        self,
        time_array_years: np.array,
        deployment_array_units: np.array,
        learning_array_units: np.array,
        learning_array_price_per_unit: np.array,
        initial_price_per_unit: float = 100,
        currency_str: str = "CAD",
        units_str: str = "MW",
        regression_flag: bool = True
    ) -> None:
        """
        UnitPriceForecaster class constructor.
        
        Parameters
        ----------
        time_array_years: np.array
            This is an array of points in time [years]. This defines both
            the modelling horizon and the modelling resolution.

        deployment_array_units: np.array
            This is an array of cumulative deployment [units] at every point
            in time.

        learning_array_units: np.array
            This is an array of cumulative deployment [units] at every point for
            which price per unit data is available.

        learning_array_price_per_unit: np.array
            This is an array of price per unit [currency/unit] data.

        initial_price_per_unit: float = 100
            This is the price per unit [currency/unit] for the first unit
            deployed. 
            ***NOT USED***

        currency_str: str, optional, default "CAD"
            This is a string defining what the currency is, for plotting.

        units_str: str, optional, default "MW"
            This is a string defining what the units are, for plotting.
        
        regression_flag: bool, optional, default True
            A flag which indicates whether linear regression should be applied
            to the learning data (`True`), or if simple linear 
            interpolation/extrapolation should be applied instead (`False`).
            
        
        Returns
        -------
        None
        """
        
        #   1. cast inputs to arrays
        self.time_array_years = np.array(time_array_years)
        """
        An array of points in time [years].
        """
        
        deployment_array_units = np.array(deployment_array_units)
        learning_array_units = np.array(learning_array_units)
        learning_array_price_per_unit = np.array(learning_array_price_per_unit)
        
        #   2. check inputs
        self.__checkInputs(
            deployment_array_units,
            learning_array_units,
            learning_array_price_per_unit,
            initial_price_per_unit
        )
        
        #   3. init attributes
        self.deployment_array_units = deployment_array_units
        """
        An array of cumulative deployment [units] at every point in time.
        """
        
        self.learning_array_units = learning_array_units
        """
        An array of cumulative deployment [units] at every point for which
        price per unit data is available.
        """
        
        self.learning_array_price_per_unit = learning_array_price_per_unit
        """
        An array of price per unit [currency/unit] data.
        """
        
        self.price_per_unit_array = np.zeros(len(self.time_array_years))
        """
        An array of forecasted price per unit [currency/data] data,
        corresponding to the given deployment data.
        """
        
        self.initial_price_per_unit = initial_price_per_unit
        """
        The price per unit [currency/unit] for the first unit deployed.
        ***NOT USED***
        """
        
        self.currency_str = currency_str
        """
        A string defining what the currency is, for plotting.
        """
        
        self.units_str = units_str
        """
        A string defining what the units are, for plotting.
        """
        
        self.regression_flag = bool(regression_flag)
        """
        A flag which indicates whether linear regression should be applied
        to the learning data (`True`), or if simple linear 
        interpolation/extrapolation should be applied instead (`False`).
        """
        
        if self.regression_flag:
            self.learning_regressor = skllm.LinearRegression().fit(
                np.log10(self.learning_array_units).reshape(-1, 1),
                np.log10(self.learning_array_price_per_unit)
            )
            """
            A linear regressor (`sklearn.linear_model.LinearRegression`)
            trained on the given learning arrays. Is only used if
            linear_regression is `True`.
            """
        
        else:
            self.learning_interpolator = spi.interp1d(
                np.log10(self.learning_array_units),
                np.log10(self.learning_array_price_per_unit),
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate"
            )
            """
            A linear interpolator/extrapolator (`scipy.interpolate.interp1d`)
            trained on the given learning arrays. Is only used if
            linear_regression is `False`.
            """
        
        return

    
    def __checkInputs(
        self,
        deployment_array_units: np.array,
        learning_array_units: np.array,
        learning_array_price_per_unit: np.array,
        initial_price_per_unit: float
    ) -> None:
        """
        Helper method to check __init__ inputs.
        
        Parameters
        ----------
        deployment_array_units: np.array
            This is an array of cumulative deployment [units] at every point
            in time.

        learning_array_units: np.array
            This is an array of cumulative deployment [units] at every point for
            which price per unit data is available.

        learning_array_price_per_unit: np.array
            This is an array of price per unit [currency/unit] data.

        initial_price_per_unit: float
            This is the price per unit [currency/unit] for the first unit
            deployed. 
            ***NOT USED***
        
        Returns
        -------
        None
        """
        
        #   1. time array must be strictly increasing
        boolean_mask = np.diff(self.time_array_years) <= 0
        
        if boolean_mask.any():
            error_string = "ERROR: UnitPriceForecaster.__checkInputs():\t"
            error_string += "time array must be strictly increasing"
            error_string += " (t[i + 1] > t[i] for all i)."

            raise RuntimeError(error_string)

        #   2. deployment array must be strictly increasing
        boolean_mask = np.diff(deployment_array_units) <= 0
        
        if boolean_mask.any():
            error_string = "ERROR: UnitPriceForecaster.__checkInputs():\t"
            error_string += "deployment array must be strictly increasing"
            error_string += " (D[i + 1] > D[i] for all i)."

            raise RuntimeError(error_string)

        #   3. learning array (units) must be strictly increasing
        boolean_mask = np.diff(learning_array_units) <= 0
        
        if boolean_mask.any():
            error_string = "ERROR: UnitPriceForecaster.__checkInputs():\t"
            error_string += "learning array (units) must be strictly increasing"
            error_string += " (U[i + 1] > U[i] for all i)."

            raise RuntimeError(error_string)

        #   4. time and deployment arrays must be same size
        if len(self.time_array_years) != len(deployment_array_units):
            error_string = "ERROR: UnitPriceForecaster.__checkInputs():\t"
            error_string += "time array and deployment array must be the"
            error_string += " same length"
            
            raise RuntimeError(error_string)

        #   5. learning arrays must be same size
        if len(learning_array_units) != len(learning_array_price_per_unit):
            error_string = "ERROR: UnitPriceForecaster.__checkInputs():\t"
            error_string += "learning arrays must be the same length"
            
            raise RuntimeError(error_string)

        #   6. learning array (units) must be strictly positive
        boolean_mask = learning_array_units <= 0
        
        if boolean_mask.any():
            error_string = "ERROR: UnitPriceForecaster.__checkInputs():\t"
            error_string += "learning array (units) must be strictly positive"
            error_string += " (x[i] > 0 for all i)"
            
            raise RuntimeError(error_string)

        #   7. learning array (price per unit) must be strictly positive
        boolean_mask = learning_array_price_per_unit <= 0
        
        if boolean_mask.any():
            error_string = "ERROR: UnitPriceForecaster.__checkInputs():\t"
            error_string += "learning array (price per unit) must be strictly"
            error_string += " positive (x[i] > 0 for all i)"
            
            raise RuntimeError(error_string)

        #   8. deployment array must be strictly positive
        boolean_mask = deployment_array_units <= 0
        
        if boolean_mask.any():
            error_string = "ERROR: UnitPriceForecaster.__checkInputs():\t"
            error_string += "deployment array must be strictly"
            error_string += " positive (x[i] > 0 for all i)"
            
            raise RuntimeError(error_string)

        #   9. initial price per unit must be strictly positive
        if initial_price_per_unit <= 0:
            error_string = "ERROR: UnitPriceForecaster.__checkInputs():\t"
            error_string += "initial price per unit must be strictly"
            error_string += " positive (x > 0)"
            
            raise RuntimeError(error_string)
        
        return


    def run(self) -> None:
        """
        Method to run and populate state arrays.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        if self.regression_flag:
            self.price_per_unit_array = np.power(
                10,
                self.learning_regressor.predict(
                    np.log10(self.deployment_array_units).reshape(-1, 1)
                )
            )
        
        else:
            self.price_per_unit_array = np.power(
                10,
                self.learning_interpolator(
                    np.log10(self.deployment_array_units)
                )
            )
        
        return


    def plot(self) -> None:
        """
        Method to plot state arrays.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        #   1. plot deployment
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.plot(
            self.time_array_years,
            self.deployment_array_units,
            zorder=2
        )
        plt.xlim(
            self.time_array_years[0],
            self.time_array_years[-1]
        )
        plt.xlabel(r"Time $t$ [years]")
        plt.ylim(
            0.98 * self.deployment_array_units[0],
            1.02 * self.deployment_array_units[-1]
        )
        plt.ylabel(r"Deployment $D$ [{}]".format(self.units_str))
        plt.tight_layout()

        #   2. plot learning curve
        learning_model_input_array = np.sort(
            np.append(
                self.learning_array_units,
                self.deployment_array_units
            )
        )
        
        if self.regression_flag:
            learning_model_output_array = np.power(
                10, 
                self.learning_regressor.predict(
                    np.log10(learning_model_input_array).reshape(-1, 1)
                )
            )
            
            learning_model_string = "learning model (linear regression)"
        
        else:
            learning_model_output_array = np.power(
                10, 
                self.learning_interpolator(
                    np.log10(learning_model_input_array)
                )
            )
            
            learning_model_string = (
                "learning model (linear interpolation/extrapolation)"
            )
        
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.scatter(
            self.learning_array_units,
            self.learning_array_price_per_unit,
            zorder=2,
            label="learning data"
        )
        plt.plot(
            learning_model_input_array,
            learning_model_output_array,
            color="black",
            linestyle="--",
            alpha=0.5,
            label=learning_model_string
        )
        plt.xscale("log")
        plt.xlim(
            0.98 * learning_model_input_array[0],
            1.02 * learning_model_input_array[-1]
        )
        plt.xlabel(r"Deployment $D$ [{}]".format(self.units_str))
        plt.yscale("log")
        plt.ylabel(
            r"Unit Price $p$ [{}/{}]".format(
                self.currency_str,
                self.units_str
            )
        )
        plt.legend()
        plt.tight_layout()
        
        #   3. forecasted unit price
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.plot(
            self.time_array_years,
            self.price_per_unit_array,
            zorder=2
        )
        plt.xlim(
            self.time_array_years[0],
            self.time_array_years[-1]
        )
        plt.xlabel(r"Time $t$ [years]")
        plt.ylabel(
            r"Unit Price $p$ [{}/{}]".format(
                self.currency_str,
                self.units_str
            )
        )
        plt.tight_layout()
        
        #   4. show
        plt.show()
        
        return


if __name__ == "__main__":
    print("TESTING:\tUnitPriceForecaster")
    print()

    good_time_array = [1, 2, 3, 4]
    good_deployment_array = [1, 2, 3, 4]
    good_learning_array_u = [1, 2, 3, 4]
    good_learning_array_ppu = [1, 2, 3, 4]
    
    #   1. passing a bad time array (not strictly increasing)
    try:
        bad_time_array = [0, 1, 0, 1]

        UnitPriceForecaster(
            bad_time_array,
            good_deployment_array,
            good_learning_array_u,
            good_learning_array_ppu
        )    

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   2. passing a bad deployment array (not strictly increasing)
    try:
        bad_deployment_array = [0, 1, 0, 1]

        UnitPriceForecaster(
            good_time_array,
            bad_deployment_array,
            good_learning_array_u,
            good_learning_array_ppu
        )    

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   3. passing a bad learning array (units) (not strictly increasing)
    try:
        bad_learning_array_u = [0, 1, 0, 1]

        UnitPriceForecaster(
            good_time_array,
            good_deployment_array,
            bad_learning_array_u,
            good_learning_array_ppu
        )    

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   4. passing a bad deployment array (not same length as time)
    try:
        bad_deployment_array = [1, 2, 3]

        UnitPriceForecaster(
            good_time_array,
            bad_deployment_array,
            good_learning_array_u,
            good_learning_array_ppu
        )    

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   5. passing a bad learning array (units) (learning arrays of different size)
    try:
        bad_learning_array_u = [1, 2, 3]

        UnitPriceForecaster(
            good_time_array,
            good_deployment_array,
            bad_learning_array_u,
            good_learning_array_ppu
        )    

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   6. passing a bad learning array (units) (not strictly positive)
    try:
        bad_learning_array_u = [-1, 0, 1, 2]

        UnitPriceForecaster(
            good_time_array,
            good_deployment_array,
            bad_learning_array_u,
            good_learning_array_ppu
        )    

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   7. passing a bad learning array (price per unit) (not strictly positive)
    try:
        bad_learning_array_ppu = [-1, 0, 1, 2]

        UnitPriceForecaster(
            good_time_array,
            good_deployment_array,
            good_learning_array_u,
            bad_learning_array_ppu
        )    

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   8. passing a bad deployment array (not strictly positive)
    try:
        bad_deployment_array = [-1, 0, 1, 2]

        UnitPriceForecaster(
            good_time_array,
            bad_deployment_array,
            good_learning_array_u,
            good_learning_array_ppu
        )    

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   9. passing a bad initial unit price (not strictly positive)
    try:
        bad_initial_unit_price = 0

        UnitPriceForecaster(
            good_time_array,
            good_deployment_array,
            good_learning_array_u,
            good_learning_array_ppu,
            initial_price_per_unit=bad_initial_unit_price
        )    

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   10. good construction
    time_array_years = np.linspace(0, 100, 1000 * 100)
    deployment_array_units = (
        2500 * (
            np.tanh(0.1 * (time_array_years - 30)) + 1
        )
    )
    
    initial_price_per_unit = 2000
    learning_array_units = np.array([1, 10, 20, 50, 100, 125, 185, 350, 500])
    learning_exponent = -0.25
    
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
    
    unit_price_forecaster = UnitPriceForecaster(
        time_array_years,
        deployment_array_units,
        learning_array_units,
        learning_array_price_per_unit,
        initial_price_per_unit=initial_price_per_unit
    )

    unit_price_forecaster.run()
    unit_price_forecaster.plot()
    
    print()
    print("TESTING:\tUnitPriceForecaster\tPASS")
    print()
    
    
    
