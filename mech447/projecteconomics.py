"""
Anthony Truelove MASc, P.Eng.  
Python Certified Professional Programmer (PCPP1)

Copyright 2025 - Anthony Truelove  
--> ***SEE LICENSE TERMS [HERE](../../LICENSE)*** <--

A project economics class, as part of the `mech447` package.
"""


import math

import matplotlib.pyplot as plt
import numpy as np


FLOAT_TOLERANCE = 1e-6


class ProjectEconomics:
    """
    A class which takes in a number of nominal economic metrics (expenses,
    income, rates, etc.) and computes various real (discounted) and
    levellized economic metrics.
    """

    def __init__(
        self,
        period_array: np.array,
        nominal_expense_array_dict: dict[str, np.array],
        nominal_income_array_dict: dict[str, np.array],
        discount_rate_per_period: float,
        period_str: str = "year",
        currency_str: str = "CAD"
    ) -> None:
        """
        ProjectEconomics class constructor.

        Parameters
        ----------
        period_array: np.array
            An array of the periods at which the economics (discrete) are
            assessed.

        nominal_expense_array_dict: dict[str, np.array]
            An dict of nominal expense arrays corresponding to each period.

        nominal_income_array_dict: dict[str, np.array]
            An dict of nominal income arrays corredponding to each period.

        discount_rate_per_period: float
            The discount rate per period.

        period_str: str, optional, default "year"
            This is a string defining what the period units are, for printing
            and plotting.

        currency_str: str, optional, default "CAD"
            This is a string defining what the currency units are, for printing
            and plotting.

        Returns
        -------
        None
        """

        #   1. cast inputs to arrays
        self.period_array = np.array(period_array)
        """
        An array of the periods at which the economics (discrete) are
        assessed.
        """

        for key in nominal_expense_array_dict.keys():
            nominal_expense_array_dict[key] = np.array(
                nominal_expense_array_dict[key]
            )

        for key in nominal_income_array_dict.keys():
            nominal_income_array_dict[key] = np.array(
                nominal_income_array_dict[key]
            )

        #   2. check inputs
        self.__checkInputs(
            nominal_expense_array_dict,
            nominal_income_array_dict,
            discount_rate_per_period
        )

        #   3. init attributes
        self.nominal_expense_array_dict = nominal_expense_array_dict
        """
        An array of nominal expenses corresponding to each period.
        """

        if len(self.nominal_expense_array_dict) == 0:
            self.nominal_expense_array_dict["DUMMY"] = np.zeros(
                len(self.period_array)
            )

        self.discounted_expense_array_dict = {}
        """
        An array of discounted expenses corresponding to each period.
        """

        for key in self.nominal_expense_array_dict.keys():
            self.discounted_expense_array_dict[key] = np.zeros(
                len(self.period_array)
            )

        self.nominal_income_array_dict = nominal_income_array_dict
        """
        An array of nominal incomes corredponding to each period.
        """

        if len(self.nominal_income_array_dict) == 0:
            self.nominal_income_array_dict["DUMMY"] = np.zeros(
                len(self.period_array)
            )

        self.discounted_income_array_dict = {}
        """
        An array of discounted incomes corredponding to each period.
        """

        for key in self.nominal_income_array_dict.keys():
            self.discounted_income_array_dict[key] = np.zeros(
                len(self.period_array)
            )

        self.discount_rate_per_period = discount_rate_per_period
        """
        The discount rate per period.
        """

        self.discount_scalar_array = np.zeros(len(self.period_array))
        """
        An array of discounting scalars, one for each period in the period
        array.
        """

        self.period_str = period_str
        """
        This is a string defining what the period units are, for printing
        and plotting.
        """

        self.currency_str = currency_str
        """
        This is a string defining what the currency units are, for printing
        and plotting.
        """

        self.net_present_cost = 0
        """
        The net present cost of the project (i.e., the sum of discounted
        expenses).
        """

        self.net_present_revenue = 0
        """
        The net present revenue of the project (i.e., the sum of discounted
        incomes).
        """

        self.net_present_value = 0
        """
        The net present value of the project (i.e., the net present revenue
        minus the net present cost).
        """

        self.capital_recovery_factor = 0
        """
        The capital recovery factor for the project.
        """

        self.levellized_cost = 0
        """
        The levellized cost of the project (i.e., the product of the capital
        recovery factor and the net present cost).
        """

        self.levellized_revenue = 0
        """
        The levellized revenue of the project (i.e., the product of the capital
        recovery factor and the net present revenue).
        """

        self.levellized_value = 0
        """
        The levellized value of the project (i.e., the product of the capital
        recovery factor and the net present value).
        """

        return


    def __checkInputs(
        self,
        nominal_expense_array_dict: dict[str, np.array],
        nominal_income_array_dict: dict[str, np.array],
        discount_rate_per_period: float
    ) -> None:
        """
        Helper method to check __init__ inputs.

        Parameters
        ----------
        nominal_expense_array: np.array
            An array of nominal expenses corresponding to each period.

        nominal_income_array: np.array
            An array of nominal incomes corredponding to each period.

        discount_rate_per_period: float
            The discount rate per period.

        Returns
        -------
        None
        """

        #   1. period array must be strictly increasing
        boolean_mask = np.diff(self.period_array) <= 0
        
        if boolean_mask.any():
            error_string = "ERROR: ProjectEconomics.__checkInputs():\t"
            error_string += "period array must be strictly increasing"
            error_string += " (t[i + 1] > t[i] for all i)."

            raise RuntimeError(error_string)

        #   2. all arrays must be same size
        for key in nominal_expense_array_dict.keys():
            array = nominal_expense_array_dict[key]

            if len(array) != len(self.period_array):
                error_string = "ERROR: ProjectEconomics.__checkInputs():\t"
                error_string += "period array and expense arrays must be the"
                error_string += " same length, expense array '"
                error_string += key
                error_string += "' is the wrong size"

                raise RuntimeError(error_string)

        for key in nominal_income_array_dict.keys():
            array = nominal_income_array_dict[key]

            if len(array) != len(self.period_array):
                error_string = "ERROR: ProjectEconomics.__checkInputs():\t"
                error_string += "period array and income arrays must be the"
                error_string += " same length, income array '"
                error_string += key
                error_string += "' is the wrong size"

                raise RuntimeError(error_string)

        #   3. expense array must be non-negative
        for key in nominal_expense_array_dict.keys():
            array = nominal_expense_array_dict[key]

            boolean_mask = array < 0

            if boolean_mask.any():
                error_string = "ERROR: ProjectEconomics.__checkInputs():\t"
                error_string += "expense arrays must be strictly non-negative"
                error_string += " (x[i] >= 0 for all i), expense array '"
                error_string += key
                error_string += "' is not"
                
                raise RuntimeError(error_string)

        #   4. income array must be non-negative
        for key in nominal_income_array_dict.keys():
            array = nominal_income_array_dict[key]

            boolean_mask = array < 0

            if boolean_mask.any():
                error_string = "ERROR: ProjectEconomics.__checkInputs():\t"
                error_string += "income arrays must be strictly non-negative"
                error_string += " (x[i] >= 0 for all i), income array '"
                error_string += key
                error_string += "' is not"
                
                raise RuntimeError(error_string)

        return


    def run(self) -> None:
        """
        Method to run and populate economic metrics and arrays.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        #   1. populate discount scalar array
        self.discount_scalar_array = np.divide(
            1,
            np.power(
                (1 + self.discount_rate_per_period),
                self.period_array
            )
        )

        #   2. populate discounted expense array dict
        for key in self.discounted_expense_array_dict.keys():
            self.discounted_expense_array_dict[key] = np.multiply(
                self.discount_scalar_array,
                self.nominal_expense_array_dict[key]
            )

        #   3. populate discounted income array dict
        for key in self.discounted_income_array_dict.keys():
            self.discounted_income_array_dict[key] = np.multiply(
                self.discount_scalar_array,
                self.nominal_income_array_dict[key]
            )

        #   4. compute net present cost, revenue, and value
        self.net_present_cost = 0

        for key in self.discounted_expense_array_dict.keys():
            self.net_present_cost += np.sum(
                self.discounted_expense_array_dict[key]
            )

        self.net_present_revenue = 0

        for key in self.discounted_income_array_dict.keys():
            self.net_present_revenue += np.sum(
                self.discounted_income_array_dict[key]
            )

        self.net_present_value = (
            self.net_present_revenue
            - self.net_present_cost
        )
        
        #   5. compute levellized cost, revenue, and value
        N_periods = self.period_array[-1]

        self.capital_recovery_factor = (
            self.discount_rate_per_period
            / (1 - math.pow(1 + self.discount_rate_per_period, -1 * N_periods))
        )

        self.levellized_cost = (
            self.capital_recovery_factor * self.net_present_cost
        )

        self.levellized_revenue = (
            self.capital_recovery_factor * self.net_present_revenue
        )

        self.levellized_value = (
            self.capital_recovery_factor * self.net_present_value
        )

        return


    def plot(
        self,
        show_flag: bool = True,
        save_flag: bool = False,
        save_path: str = ""
    ) -> None:
        """
        Method to plot economics arrays.

        Parameters
        ----------
        show_flag: bool, optional, default True
            Flag which indicates whether or not to show the generated plots.

        save_flag: bool, optional, default False
            Flag which indicates whether or not to save the generated plots.

        save_path: str, optional, default ""
            Path (either relative or absolute) where the generated plots
            should be saved. Defaults to the empty string, in which case the
            plots will be saved to the current working directory.

        Returns
        -------
        None
        """

        #   1. plot nominal expenses and income
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)

        bar_heights = np.zeros(len(self.period_array))
        bar_order = 999

        for key in self.nominal_expense_array_dict:
            if key.upper() == "DUMMY":
                continue

            elif (
                np.sum(self.nominal_expense_array_dict[key])
                <= FLOAT_TOLERANCE
            ):
                continue

            bar_heights += self.nominal_expense_array_dict[key]

            plt.bar(
                self.period_array,
                -1 * bar_heights,
                zorder=bar_order,
                label=key
            )

            bar_order -= 1

        bar_heights = np.zeros(len(self.period_array))
        bar_order = 999

        for key in self.nominal_income_array_dict:
            if key.upper() == "DUMMY":
                continue

            elif (
                np.sum(self.nominal_income_array_dict[key])
                <= FLOAT_TOLERANCE
            ):
                continue

            bar_heights += self.nominal_income_array_dict[key]

            plt.bar(
                self.period_array,
                bar_heights,
                zorder=bar_order,
                label=key
            )

            bar_order -= 1

        plt.xlabel("Period [{}]".format(self.period_str))
        plt.ylabel("Value (nominal) [{}]".format(self.currency_str))
        plt.legend()
        plt.tight_layout()

        if save_flag:
            fig_path = save_path + "nominal_income_expenses.png"

            plt.savefig(
                fig_path,
                format="png",
                dpi=128
            )

            print(
                "ProjectEconomics.plot():\tnominal income expenses plot saved to",
                fig_path
            )

        #   2. plot discounted expenses and income
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)

        bar_heights = np.zeros(len(self.period_array))
        bar_order = 999

        for key in self.discounted_expense_array_dict:
            if key.upper() == "DUMMY":
                continue

            elif (
                np.sum(self.discounted_expense_array_dict[key])
                <= FLOAT_TOLERANCE
            ):
                continue

            bar_heights += self.discounted_expense_array_dict[key]

            plt.bar(
                self.period_array,
                -1 * bar_heights,
                zorder=bar_order,
                label=key
            )

            bar_order -= 1

        bar_heights = np.zeros(len(self.period_array))
        bar_order = 999

        for key in self.discounted_income_array_dict:
            if key.upper() == "DUMMY":
                continue

            elif (
                np.sum(self.discounted_income_array_dict[key])
                <= FLOAT_TOLERANCE
            ):
                continue

            bar_heights += self.discounted_income_array_dict[key]

            plt.bar(
                self.period_array,
                bar_heights,
                zorder=bar_order,
                label=key
            )

            bar_order -= 1

        plt.xlabel("Period [{}]".format(self.period_str))
        plt.ylabel("Value (discounted) [{}]".format(self.currency_str))
        plt.legend()
        plt.tight_layout()

        if save_flag:
            fig_path = save_path + "discounted_income_expenses.png"

            plt.savefig(
                fig_path,
                format="png",
                dpi=128
            )

            print(
                "ProjectEconomics.plot():\tdiscounted income expenses plot saved to",
                fig_path
            )

        #   3. show
        if show_flag:
            plt.show()

        return


    def printKeyMetrics(self) -> None:
        """
        Method to print key economic metrics.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        print(
            "net_present_cost:",
            round(self.net_present_cost, 2),
            self.currency_str
        )

        print(
            "net_present_revenue:",
            round(self.net_present_revenue, 2),
            self.currency_str
        )

        print(
            "net_present_value:",
            round(self.net_present_value, 2),
            self.currency_str
        )

        print()

        print(
            "capital_recovery_factor:",
            round(self.capital_recovery_factor, 5)
        )

        print(
            "levellized_cost:",
            round(self.levellized_cost, 2),
            self.currency_str + "/" + self.period_str
        )

        print(
            "levellized_revenue:",
            round(self.levellized_revenue, 2),
            self.currency_str + "/" + self.period_str
        )

        print(
            "levellized_value:",
            round(self.levellized_value, 2),
            self.currency_str + "/" + self.period_str
        )

        print()


if __name__ == "__main__":
    print("TESTING:\tProjectEconomics")
    print()

    good_period_array = [-2, -1, 0, 1, 2]

    good_expense_array_dict = {
        "first expense": [1, 0, 1, 0, 1],
        "second expense": [1, 0, 1, 0, 1],
        "third expense": [1, 0, 1, 0, 1]
    }

    good_income_array_dict = {
        "first income": [1, 0, 1, 0, 1],
        "second income": [1, 0, 1, 0, 1],
        "third income": [1, 0, 1, 0, 1]
    }

    discount_rate_per_period = 0.08

    #   1. passing a bad period array (not strictly increasing)
    try:
        bad_period_array = [2, 1, 0, -1, -2]

        ProjectEconomics(
            bad_period_array,
            good_expense_array_dict,
            good_income_array_dict,
            discount_rate_per_period
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   2. passing a bad expense array dict (arrays of different size)
    try:
        bad_expense_array_dict = {
            "first expense": [1, 0, 1, 0, 1],
            "second expense": [1, 0, 1, 0],
            "third expense": [1, 0, 1, 0, 1]
        }

        ProjectEconomics(
            good_period_array,
            bad_expense_array_dict,
            good_income_array_dict,
            discount_rate_per_period
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   3. passing a bad income array dict (arrays of different size)
    try:
        bad_income_array_dict = {
            "first income": [1, 0, 1, 0, 1],
            "second income": [1, 0, 1, 0, 1],
            "third income": [1, 0, 1]
        }

        ProjectEconomics(
            good_period_array,
            good_expense_array_dict,
            bad_income_array_dict,
            discount_rate_per_period
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   4. passing a bad expense array dict (not strictly non-negative)
    try:
        bad_expense_array_dict = {
            "first expense": [-1, 0, -1, 0, 1],
            "second expense": [1, 0, 1, 0, 1],
            "third expense": [1, 0, 1, 0, 1]
        }

        ProjectEconomics(
            good_period_array,
            bad_expense_array_dict,
            good_income_array_dict,
            discount_rate_per_period
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   5. passing a bad income array dict (not strictly non-negative)
    try:
        bad_income_array_dict = {
            "first income": [1, 0, 1, 0, 1],
            "second income": [1, 0, -1, 0, -1],
            "third income": [1, 0, 1, 0, 1]
        }

        ProjectEconomics(
            good_period_array,
            good_expense_array_dict,
            bad_income_array_dict,
            discount_rate_per_period
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    print()

    #   6. good construction
    overnight_build_cost = 1092
    fixed_annual_OnM = 17.39
    variable_annual_OnM = 7.62
    fuel_annual = 88.73

    period_array = np.array([-2 + i for i in range(0, 33)])

    build_cost_array = np.zeros(len(period_array))

    for i in range(0, 2):
        build_cost_array[i] = overnight_build_cost / 2

    fixed_OnM_array = np.zeros(len(period_array))

    for i in range(3, 33):
        fixed_OnM_array[i] = fixed_annual_OnM

    variable_OnM_array = np.zeros(len(period_array))

    for i in range(3, 33):
        variable_OnM_array[i] = variable_annual_OnM

    fuel_array = np.zeros(len(period_array))

    for i in range(3, 33):
        fuel_array[i] = fuel_annual

    nominal_expense_array_dict = {
        "Build Costs": build_cost_array,
        "Fixed O&M Costs": fixed_OnM_array,
        "Variable O&M Costs": variable_OnM_array,
        "Fuel Costs": fuel_array
    }

    income_array = 60 * np.ones(len(period_array))

    for i in range(0, 3):
        income_array[i] = 0

    nominal_income_array_dict = {
        "Test Income": income_array
    }

    project_economics = ProjectEconomics(
        period_array,
        nominal_expense_array_dict,
        nominal_income_array_dict,
        discount_rate_per_period
    )

    project_economics.run()
    project_economics.printKeyMetrics()
    project_economics.plot()

    print()
    print("TESTING:\tProjectEconomics\tPASS")
    print()
