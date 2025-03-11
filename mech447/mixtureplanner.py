"""
Anthony Truelove MASc, P.Eng.  
Python Certified Professional Programmer (PCPP1)

Copyright 2025 - Anthony Truelove  
--> ***SEE LICENSE TERMS [HERE](../../LICENSE)*** <--

A production mixture planning class, as part of the `mech447` package.
"""


import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as spi


HOURS_PER_YEAR = 8760


class MixturePlanner:
    """
    A class which takes in a demand time series, an abitrary number of 
    renewable production time series, and an arbitrary number of dispatchable
    tech screening curves, and then returns an optimal production mix.
    """

    def __init__(
        self,
        time_array_hrs: np.array,
        demand_array: np.array,
        renewable_production_dict: dict[str, np.array],
        screening_curve_dict: dict[str, np.array],
        sizing_override_array: np.array = np.array([]),
        power_units_str: str = "MW",
        currency_units_str: str = "CAD"
    ) -> None:
        """
        MixturePlanner class constructor

        Parameters
        ----------
        time_array_hrs: np.array
            This is an array of points in time [hours]. This defines the time
            series inputs of demand_array and the elements of
            renewable_production_dict. SHOULD BE EXACTLY ONE YEAR (i.e., 
            8760 hours).

        demand_array: np.array
            This is an array of average demand (power) values corresponding to
            each point of time_array_hrs.

        renewable_production_dict: dict[str, np.array]
            This is a dictionary of renewable production time series, where
            each time series contains the average production (power) of each
            renewable correponding to each point of time_array_hrs.

        screening_curve_dict: dict[str, np.array]
            This is a dictionary of screening curves for each dispatchable
            tech considered in the production mix. Note that all elements of
            the dict must be the same length, and it is assumed that each 
            element spans the capacity factor interval [0, 1].

        sizing_override_array: np.array, optional, default []
            This is an array of sizing overrides (e.g. reserve capacity). The
            order of the size overrides corresponds to the order of elements
            in the screening curve dictionary (and so both inputs must have
            the same number of elements).

        power_units_str: str, optional, default "MW"
            This is a string defining what the power units are, for printing
            and plotting.

        currency_units_str: str, optional, default "CAD"
            This is a string defining what the currency units are, for printing
            and plotting.

        Returns
        -------
        None
        """

        #   1. cast inputs to arrays, get delta time array
        self.time_array_hrs = np.array(time_array_hrs)
        """
        This is an array of points in time [hours]. This defines the time
        series inputs of demand_array and the elements of
        renewable_production_dict. SHOULD BE EXACTLY ONE YEAR (i.e., 
        8760 hours).
        """

        self.delta_time_array_hrs = np.diff(self.time_array_hrs)
        self.delta_time_array_hrs = np.append(
            self.delta_time_array_hrs[0],
            self.delta_time_array_hrs
        )
        """
        This is an array of time deltas, for use in integrating power time
        series to get energy amounts.
        """

        #   2. check inputs
        self.__checkInputs(
            demand_array,
            renewable_production_dict,
            screening_curve_dict,
            sizing_override_array
        )

        #   3. init attributes
        self.demand_array = np.array(demand_array)
        """
        This is an array of average demand (power) values corresponding to
        each point of time_array_hrs.
        """

        self.renewable_production_dict = renewable_production_dict
        """
        This is a dictionary of renewable production time series, where
        each time series contains the average production (power) of each
        renewable correponding to each point of time_array_hrs.
        """

        for key in self.renewable_production_dict.keys():
            self.renewable_production_dict[key] = np.array(
                self.renewable_production_dict[key]
            )

        self.screening_curve_dict = screening_curve_dict
        """
        This is a dictionary of screening curves for each dispatchable
        tech considered in the production mix. Note that all elements of
        the dict must be the same length, and it is assumed that each 
        element spans the capacity factor interval [0, 1].
        """

        for key in self.screening_curve_dict.keys():
            self.screening_curve_dict[key] = np.array(
                self.screening_curve_dict[key]
            )

        self.sizing_override_array = np.array(sizing_override_array)
        """
        This is an array of sizing overrides (e.g. reserve capacity). The
        order of the size overrides corresponds to the order of elements
        in the screening curve dictionary (and so both inputs must have
        the same number of elements).
        """

        self.power_units_str = power_units_str
        """
        This is a string defining what the power units are, for printing
        and plotting.
        """

        self.currency_units_str = currency_units_str
        """
        This is a string defining what the currency units are, for printing
        and plotting.
        """

        self.residual_demand_array = 0
        """
        This is an array of the average residual demand (power) after the 
        average renewable productions have all been deducted from the 
        corresonding average demand.
        """

        self.duration_x_array = 0
        """
        Base array (x-axis) for load duration plots.
        """

        self.load_duration_array = 0
        """
        This is an array which contains the points of a load duration curve
        (corresponds to the given demand array).
        """

        self.residual_load_duration_array = 0
        """
        This is an array which contains the points of a residual load duration
        curve (corresponds to the computed residual demand array).
        """

        self.capacity_factor_array = 0
        """
        Base array (x-axis) for technology screening curves. Is assumed to be
        the interval [0, 1] and of the same length as the screening curve
        arrays.
        """

        self.minimum_cost_frontier = 0
        """
        An array containing the minimum screened technology costs for each
        capacity factor value.
        """

        self.changeover_ref_key_array = 0
        """
        An array of screening dict keys corresponding to the new minimum cost
        technology at each changeover point.
        """

        self.changeover_capacity_factor_array = 0
        """
        An array of the capacity factor at which a technology changeover
        happens.
        """

        self.changeover_cost_array = 0
        """
        An array of costs at which a technology changeover happens.
        """

        self.cf_2_residual_load_interp = 0
        """
        A `scipy.interpolate.interp1d` that is used to map from arbitrary
        capacity factors to the residual load at the corresponding 
        exceedance proportion.
        """

        self.residual_load_2_cf_interp = 0
        """
        A `scipy.interpolate.interp1d` that is used to map from arbitrary
        residual loads (at some exceedance proportion) to the corresponding
        capacity factor.
        """

        self.sizing_dict = 0
        """
        A dictionary of the minimum cost sizing of each screened technology.
        """

        self.total_demand = 0
        """
        The total demand (energy) on the system over the entire modelling 
        horizon.
        """

        self.production_dict = 0
        """
        A dictionary of total energy production from all sources.
        """

        self.tech_capacity_factor_dict = 0
        """
        A dictionary of the capacity factors of the screened technologies at
        their selected sizes.
        """

        self.tech_cost_dict = 0
        """
        A dictionary of the costs of the screened technologies at their
        selected sizes.
        """

        self.renewable_production_duration_dict = 0
        """
        A dictionary of production duration curves for each renewable tech.
        """

        self.supply_stack_dict = 0
        """
        A supply stack dictionary consisting of four parallel arrays:
        technologies, installed capacities, cumulative installed capacities,
        and pool price. All arrays are in merit order.
        """

        self.pool_price_array = 0
        """
        An array of the pool price at each point in time.
        """

        self.pool_price_duration_array = 0
        """
        This is an array which contains the points of a pool price duration
        curve (corresponds to the computed pool price array).
        """

        self.wholesale_price = 0
        """
        The wholesale price of energy. In other words, this is the energy
        weighted average pool price over the modelling period.
        """

        self.capacity_price = 0
        """
        The fixed cost of the cheapest dispatchable technology considered.
        """

        self.LACE_dict = 0
        """
        A dictionary of the capacity credit and computed LACE values for each
        renewable generation asset.
        """

        return


    def __str__(self,) -> None:
        """
        __str__ magic method, to handle print(self).

        Parameters
        ----------
        None

        Returns
        -------
        str
            Just returns an empty string. This is just a wrapper of
            printKeyMetrics().
        """

        self.printKeyMetrics()
        return ""


    def __checkInputs(
        self,
        demand_array: np.array,
        renewable_production_dict: dict[str, np.array],
        screening_curve_dict: dict[str, np.array],
        sizing_override_array: np.array
    ) -> None:
        """
        Helper method to check __init__ inputs.

        Parameters
        ----------
        demand_array: np.array
            This is an array of average demand (power) values corresponding to
            each point of time_array_hrs.

        renewable_production_dict: dict[str, np.array]
            This is a dictionary of renewable production time series, where
            each time series contains the average production (power) of each
            renewable correponding to each point of time_array_hrs.

        screening_curve_dict: dict[str, np.array]
            This is a dictionary of screening curves for each dispatchable
            tech considered in the production mix. Note that all elements of
            the dict must be the same length, and it is assumed that each 
            element spans the capacity factor interval [0, 1].

        sizing_override_array: np.array
            This is an array of sizing overrides (e.g. reserve capacity). The
            order of the size overrides corresponds to the order of elements
            in the screening curve dictionary (and so both inputs must have
            the same number of elements).

        Returns
        -------
        None
        """

        #   1. time array must be exactly one year (i.e., 8760 hours)
        if len(self.time_array_hrs) != 8760:
            error_string = "ERROR: MixturePlanner.__checkInputs():\t"
            error_string += "time array [hours] must be of length 8760 ("
            error_string += "assuming 8760 hours/year)."

            raise RuntimeError(error_string)

        #   2. time array must be strictly increasing
        boolean_mask = np.diff(self.time_array_hrs) <= 0
        
        if boolean_mask.any():
            error_string = "ERROR: MixturePlanner.__checkInputs():\t"
            error_string += "time array [hours] must be strictly increasing"
            error_string += " (t[i + 1] > t[i] for all i)."

            raise RuntimeError(error_string)

        #   3. demand array must be same length as time array
        if len(demand_array) != len(self.time_array_hrs):
            error_string = "ERROR: MixturePlanner.__checkInputs():\t"
            error_string += "demand array and time array [hours] must "
            error_string += "be the same length."

            raise RuntimeError(error_string)

        #   4. demand array must be non-negative
        boolean_mask = demand_array < 0

        if boolean_mask.any():
            error_string = "ERROR: MixturePlanner.__checkInputs():\t"
            error_string += "demand array must be strictly non-negative"
            error_string += " (x[i] >= 0 for all i)."

            raise RuntimeError(error_string)

        #   5. renewable production arrays must all be same length as time array
        for key in renewable_production_dict.keys():
            if len(renewable_production_dict[key]) != len(self.time_array_hrs):
                error_string = "ERROR: MixturePlanner.__checkInputs():\t"
                error_string += "renewable production arrays must be same "
                error_string += "length as time array [hours], renewable "
                error_string += "production array '"
                error_string += key
                error_string += "' is not"

                raise RuntimeError(error_string)

        #   6. renewable production arrays must all be non-negative
        for key in renewable_production_dict.keys():
            boolean_mask = renewable_production_dict[key] < 0

            if boolean_mask.any():
                error_string = "ERROR: MixturePlanner.__checkInputs():\t"
                error_string += "renewable production arrays must be strictly "
                error_string += "non-negative (x[i] >= 0 for all i), "
                error_string += "renewable production array '"
                error_string += key
                error_string += "' is not"

                raise RuntimeError(error_string)

        #   7. screening curve dict must be non-empty
        if len(screening_curve_dict) <= 0:
            error_string = "ERROR: MixturePlanner.__checkInputs():\t"
            error_string += "screening curve dictionary must be non-empty"

            raise RuntimeError(error_string)

        #   8. screening curve arrays must all be the same length
        ref_length = 0
        ref_key = ""
        set_ref_length = True

        for key in screening_curve_dict.keys():
            if set_ref_length:
                ref_length = len(screening_curve_dict[key])
                ref_key = key
                set_ref_length = False
                continue

            if len(screening_curve_dict[key]) != ref_length:
                error_string = "ERROR: MixturePlanner.__checkInputs():\t"
                error_string += "screening curve arrays must all be the same "
                error_string += "length, but screening curve array '"
                error_string += key
                error_string += "' is of a different length than screening "
                error_string += "curve array '"
                error_string += ref_key
                error_string += "'"

                raise RuntimeError(error_string)

        #   9. screening curve arrays must all be strictly positive
        for key in screening_curve_dict.keys():
            boolean_mask = screening_curve_dict[key] <= 0

            if boolean_mask.any():
                error_string = "ERROR: MixturePlanner.__checkInputs():\t"
                error_string += "screening curve arrays must be strictly "
                error_string += "positive (x[i] > 0 for all i), "
                error_string += "screening curve array '"
                error_string += key
                error_string += "' is not"

                raise RuntimeError(error_string)

        #   10. non-empty sizing override array should have as many elements as
        #       screening curve dictionary has keys
        if len(sizing_override_array) > 0:
            N = len(screening_curve_dict)

            if len(sizing_override_array) != N:
                error_string = "ERROR: MixturePlanner.__checkInputs():\t"
                error_string += "sizing override array must have as many "
                error_string += "elements as the screening curve dictionary "
                error_string += "has keys"

                raise RuntimeError(error_string)

        #   11. sizing override array must be strictly non-negative
        for i in range(0, len(sizing_override_array)):
            sizing = sizing_override_array[i]

            if sizing < 0:
                error_string = "ERROR: MixturePlanner.__checkInputs():\t"
                error_string += "sizing override array must be strictly "
                error_string += "non-negative (x[i] >= 0 for all i), "
                error_string += "sizing override array element "
                error_string += str(i)
                error_string += " is not"

                raise RuntimeError(error_string)

        #   12. screening curve arrays should be sufficiency dense
        for key in screening_curve_dict.keys():
            if len(screening_curve_dict[key]) < 100:
                warning_string = "WARNING: MixturePlanner.__checkInputs():\t"
                warning_string += "the given screening curve array for '"
                warning_string += key
                warning_string += "' seems sparse (which can lead to "
                warning_string += "computation errors), try increasing array "
                warning_string += "density (like, span the capacity factor "
                warning_string += "interval [0, 1] in 100 steps or more)"

                print(warning_string)
                break

        return


    def computeResidualDemand(self) -> None:
        """
        Helper method to compute the residual demand at every time step.
        Residual demand is defined as

        $$ \\widehat{L}(t) = L(t) - \\sum_i R_i(t) $$

        where $\\widehat{L}$ is residual demand (residual load), $L$ is demand
        (load), and $R_i$ is production from the $i^\\textrm{th}$ renewable
        generation asset.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.residual_demand_array = copy.deepcopy(self.demand_array)

        for key in self.renewable_production_dict.keys():
            self.residual_demand_array = np.subtract(
                self.residual_demand_array,
                self.renewable_production_dict[key]
            )

        return


    def constructLoadDurationCurves(self) -> None:
        """
        Helper method to construct load duration curves (load and residual
        load, as appropriate). Again, 'load' and 'demand' are synonymous
        here.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        #   1. extract load duration curve
        N = len(self.demand_array)
        self.duration_x_array = (1 / N) * np.linspace(0, N - 1, N)
        self.load_duration_array = np.flip(np.sort(self.demand_array))

        #   2. extract residual load duration curve
        self.residual_load_duration_array = np.flip(
            np.sort(self.residual_demand_array)
        )

        return


    def constructRenewableProductionDurationCurves(self) -> None:
        """
        Helper method to construct renewable production duration curves.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if len(self.renewable_production_dict) > 0:
            self.renewable_production_duration_dict = {}

            for key in self.renewable_production_dict.keys():
                self.renewable_production_duration_dict[key] = np.flip(
                    np.sort(self.renewable_production_dict[key])
                )

        return


    def extractMinimumCostFrontier(self) -> None:
        """
        Helper method to extract minimum cost frontier and track technology
        changeover points.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        for key in self.screening_curve_dict.keys():
            self.capacity_factor_array = np.linspace(
                0, 1, len(self.screening_curve_dict[key])
            )
            break

        self.minimum_cost_frontier = np.zeros(len(self.capacity_factor_array))
        ref_key = ""

        self.changeover_ref_key_array = []
        self.changeover_capacity_factor_array = []
        self.changeover_cost_array = []

        for i in range(0, len(self.capacity_factor_array)):
            min_cost = np.inf

            for key in self.screening_curve_dict.keys():
                if self.screening_curve_dict[key][i] < min_cost:
                    min_cost = self.screening_curve_dict[key][i]
                    ref_key = key

            self.minimum_cost_frontier[i] = min_cost

            if (
                len(self.changeover_ref_key_array) == 0
                or ref_key != self.changeover_ref_key_array[-1]
            ):
                self.changeover_ref_key_array.append(ref_key)

                self.changeover_capacity_factor_array.append(
                    self.capacity_factor_array[i]
                )

                self.changeover_cost_array.append(min_cost)

        self.changeover_ref_key_array = np.flip(
            np.array(self.changeover_ref_key_array)
        )

        self.changeover_capacity_factor_array = np.flip(
            np.array(self.changeover_capacity_factor_array)
        )

        self.changeover_cost_array = np.flip(
            np.array(self.changeover_cost_array)
        )

        return


    def getMinimumCostSizing(self) -> None:
        """
        Helper method to compute minimum cost sizing (i.e., cost-optimal
        generation mix).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        #   1. init sizing dictionary
        self.sizing_dict = {}

        for key in self.screening_curve_dict.keys():
            self.sizing_dict[key] = 0

        #   2. handle no override
        if len(self.sizing_override_array) == 0:
            total_installed = 0

            for i in range(0, len(self.changeover_ref_key_array)):
                key = self.changeover_ref_key_array[i]
                capacity_factor = self.changeover_capacity_factor_array[i]

                installed_capacity = self.cf_2_residual_load_interp(
                    capacity_factor
                ) - total_installed

                self.sizing_dict[key] = installed_capacity
                total_installed += installed_capacity

        #   3. handle override
        else:
            idx = 0

            for key in self.sizing_dict.keys():
                self.sizing_dict[key] = self.sizing_override_array[idx]
                idx += 1

                if key not in self.changeover_ref_key_array:
                    self.changeover_ref_key_array = np.append(
                        self.changeover_ref_key_array,
                        key
                    )

        return

    def getTotalProduction(self) -> None:
        """
        Helper method to compute total production from all generation assets.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        #   1. renewable generation assets
        self.production_dict = {}

        for key in self.renewable_production_dict.keys():
            self.production_dict[key] = np.dot(
                self.renewable_production_dict[key],
                self.delta_time_array_hrs
            )

        #   2. dispatchable generation assets
        for key in self.sizing_dict.keys():
            self.production_dict[key] = 0

        N = len(self.time_array_hrs)

        supply_stack_keys = [key for key in self.supply_stack_dict.keys()]
        M = len(self.supply_stack_dict[supply_stack_keys[0]])

        for i in range(0, N):
            residual_demand_remaining = self.residual_demand_array[i]
            delta_time_hrs = self.delta_time_array_hrs[i]

            for j in range(0, M):
                tech = self.supply_stack_dict[supply_stack_keys[0]][j]
                capacity = self.supply_stack_dict[supply_stack_keys[1]][j]

                if residual_demand_remaining > capacity:
                    self.production_dict[tech] += capacity * delta_time_hrs
                    residual_demand_remaining -= capacity

                else:
                    self.production_dict[tech] += (
                        residual_demand_remaining * delta_time_hrs
                    )
                    residual_demand_remaining = 0

        return


    def computeTechnologyCapacityFactors(self) -> None:
        """
        Helper method to compute capacity factors for each technology in the
        generation mix.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.tech_capacity_factor_dict = {}

        for key in self.sizing_dict.keys():
            if self.sizing_dict[key] <= 0:
                capacity_factor = 0

            else:
                capacity_factor = (
                    self.production_dict[key]
                    / (HOURS_PER_YEAR * self.sizing_dict[key])
                )

            self.tech_capacity_factor_dict[key] = capacity_factor

        return


    def computeTechnologyCosts(self) -> None:
        """
        Helper method to compute the costs (annual revenue requirement) for
        each technology in the generation mix.
        """

        self.tech_cost_dict = {}

        for key in self.tech_capacity_factor_dict.keys():
            capacity_factor = self.tech_capacity_factor_dict[key]

            if capacity_factor > 0:
                idx_cf = np.where(
                    self.capacity_factor_array >= capacity_factor
                )[0][0]

                self.tech_cost_dict[key] = (
                    self.screening_curve_dict[key][idx_cf]
                )

            else:
                self.tech_cost_dict[key] = (
                    self.screening_curve_dict[key][0]
                )

        return


    def constructSupplyStack(self) -> None:
        """
        Helper method to construct supply stack for each technology in the
        generation mix.

        Note that the pool price associated with each tech is computed as the
        slope of the corresponding screening curve divided by 8760 hours per
        year.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        key_array = np.array([key for key in self.sizing_dict.keys()])

        N = len(key_array)

        capacity_array = np.zeros(N)
        cumulative_capacity_array = np.zeros(N)
        pool_price_array = np.zeros(N)

        for i in range(0, N):
            capacity_array[i] = self.sizing_dict[key_array[i]]

            pool_price_array[i] = (
                self.screening_curve_dict[key_array[i]][-1]
                - self.screening_curve_dict[key_array[i]][0]
            ) / HOURS_PER_YEAR

        idx_sort = np.argsort(pool_price_array)

        key_array = key_array[idx_sort]
        capacity_array = capacity_array[idx_sort]
        pool_price_array = pool_price_array[idx_sort]

        for i in range(0, N):
            if i == 0:
                cumulative_capacity_array[i] = capacity_array[i]

            else:
                cumulative_capacity_array[i] = (
                    cumulative_capacity_array[i - 1]
                    + capacity_array[i]
                )

        self.supply_stack_dict = {
            "Technologies": key_array,
            "Capacities [{}]".format(self.power_units_str):
                capacity_array,
            "Cumulative Capacities [{}]".format(self.power_units_str):
                cumulative_capacity_array,
            "Pool Price [{}/{}h]".format(
                self.currency_units_str,
                self.power_units_str
            ):
                pool_price_array
        }

        return


    def generatePoolPriceArrays(self) -> None:
        """
        Helper method to generate pool price array and pool price duration
        curve array.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        #   1. init pool price array
        N = len(self.time_array_hrs)
        self.pool_price_array = np.zeros(N)

        #   2. populate pool price array
        key_list = [key for key in self.supply_stack_dict.keys()]
        M = len(self.supply_stack_dict[key_list[0]])

        for i in range(0, N):
            if self.residual_demand_array[i] <= 0:
                continue

            for j in range(0, M):
                if (
                    self.residual_demand_array[i]
                    <= self.supply_stack_dict[key_list[2]][j]
                ):
                    break

            self.pool_price_array[i] = (
                self.supply_stack_dict[key_list[3]][j]
            )

        #   3. extract pool price duration array
        self.pool_price_duration_array = np.flip(
            np.sort(self.pool_price_array)
        )

        return


    def computeWholesalePrice(self) -> None:
        """
        Helper method to compute wholesale price. Wholesale price is defined
        as

        $$ P_\\textrm{wholesale} = \\frac{\\int_{0}^{T}L(t)\\textrm{PP}(t) dt}{\\int_{0}^{T}L(t) dt} \\cong \\frac{\\sum_{i}L_i\\textrm{PP}_i(\\Delta t)_i}{\\sum_{i}L_i(\\Delta t)_i} $$

        where $P_\\textrm{wholesale}$ is wholesale price (i.e., energy weighted
        average pool price), $T$ is the modelling period, $L$ is demand (load),
        and $\\textrm{PP}$ is pool price.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        numerator = np.sum(
            np.multiply(
                np.multiply(
                    self.demand_array,
                    self.pool_price_array
                ),
                self.delta_time_array_hrs
            )
        )

        self.wholesale_price = numerator / self.total_demand

        return


    def computeLACE(self) -> None:
        """
        Helper method to compute levellized avoided cost of energy (LACE) for
        each renewable generation asset. LACE is defined for each renewable
        generation asset as

        $$ \\textrm{LACE} = \\frac{\\textrm{energy revenue} + \\textrm{capacity payment}}{\\textrm{energy sold}} = \\frac{\\sum_i\\left[\\textrm{PP}_iR_i(\\Delta t)_i\\right] + P_\\textrm{capacity}\\textrm{CC}}{\\sum_i\\left[R_i(\\Delta t)_i\\right]} $$

        where $\\textrm{PP}$ is pool price, $R$ is renewable generation,
        $P_\\textrm{capacity}$ is the capacity price (taken to be the fixed
        cost of the cheapest dispatchable technology considered), and
        $\\textrm{CC}$ is the capacity credit. Note that capacity credit is
        computed here as

        $$ \\textrm{CC} = L_\\textrm{max} - \\textrm{max}\\left(L_i - R_i\\right) $$

        where $L$ is demand (load), and $\\left(L_i - R_i\\right)$ is the
        set of partial residuals.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        #   1. compute capacity price (for each time step)
        self.capacity_price = np.inf

        for key in self.screening_curve_dict.keys():
            if self.screening_curve_dict[key][0] < self.capacity_price:
                self.capacity_price = self.screening_curve_dict[key][0]

        #   2. init LACE dict
        self.LACE_dict = {}

        for key in self.renewable_production_dict.keys():
            self.LACE_dict[key] = [0, 0]

        #   3. populate LACE dict
        for key in self.LACE_dict.keys():
            annual_energy_sales = np.sum(
                np.multiply(
                    np.multiply(
                        self.pool_price_array,
                        self.renewable_production_dict[key]
                    ),
                    self.delta_time_array_hrs
                )
            )

            partial_residual_load_array = (
                self.demand_array - self.renewable_production_dict[key]
            )

            capacity_credit = (
                np.max(self.demand_array)
                - np.max(partial_residual_load_array)
            )

            self.LACE_dict[key][0] = capacity_credit

            annual_capacity_payment = self.capacity_price * capacity_credit

            self.LACE_dict[key][1] = (
                (annual_energy_sales + annual_capacity_payment)
                / self.production_dict[key]
            )

        return


    def __DEPRECATED_computeLACE(self) -> None:
        """
        Helper method to compute levellized avoided cost of energy (LACE) for
        each renewable generation asset. LACE is defined for each renewable
        generation asset as

        $$ \\textrm{LACE} = \\frac{\\sum_i\\left[\\textrm{PP}_iR_i(\\Delta t)_i + P_\\textrm{capacity}R_i\\right]}{\\sum_iR_i(\\Delta t)_i} $$

        where $\\textrm{PP}$ is pool price, $R$ is renewable generation,
        and $P_\\textrm{capacity}$ is the capacity price (taken to be the fixed
        cost of the cheapest dispatchable technology available).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        #   1. compute capacity price (for each time step)
        lowest_fixed_cost = np.inf

        for key in self.screening_curve_dict.keys():
            if self.screening_curve_dict[key][0] < lowest_fixed_cost:
                lowest_fixed_cost = self.screening_curve_dict[key][0]

        self.capacity_price = lowest_fixed_cost / HOURS_PER_YEAR

        #   2. init LACE dict
        self.LACE_dict = {}

        for key in self.renewable_production_dict.keys():
            self.LACE_dict[key] = 0

        #   3. populate LACE dict
        for key in self.LACE_dict.keys():
            annual_energy_sales = np.sum(
                np.multiply(
                    np.multiply(
                        self.pool_price_array,
                        self.renewable_production_dict[key]
                    ),
                    self.delta_time_array_hrs
                )
            )

            annual_capacity_payment = self.capacity_price * np.sum(
                self.renewable_production_dict[key]
            )

            self.LACE_dict[key] = (
                (annual_energy_sales + annual_capacity_payment)
                / self.production_dict[key]
            )

        return


    def run(self,) -> None:
        """
        Method to run the mixture planner and generate results.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        #   1. compute residual demand array
        self.computeResidualDemand()

        #   2. construct load duration curves
        self.constructLoadDurationCurves()

        #   3. construct renewable production duration curves
        self.constructRenewableProductionDurationCurves()

        #   4. extract minimum cost frontier, track changeover points
        self.extractMinimumCostFrontier()

        #   5. construct capacity factor to residual load interpolator
        self.cf_2_residual_load_interp = spi.interp1d(
            self.duration_x_array,
            self.residual_load_duration_array,
            kind="linear",
            fill_value="extrapolate"
        )

        #   6. construct residual load to capacity factor interpolator
        self.residual_load_2_cf_interp = spi.interp1d(
            self.residual_load_duration_array,
            self.duration_x_array,
            kind="linear",
            fill_value="extrapolate"
        )

        #   7. get minimum cost sizing (based on changeover points),
        #      override sizing where appropriate
        self.getMinimumCostSizing()

        #   8. compute total demand
        self.total_demand = np.dot(
            self.demand_array,
            self.delta_time_array_hrs
        )

        #   9. construct supply stack
        self.constructSupplyStack()

        #   10. get total production from all generation assets in the supply
        #       stack
        self.getTotalProduction()

        #   11. compute technology capacity factors
        self.computeTechnologyCapacityFactors()

        #   12. compute technology costs
        self.computeTechnologyCosts()

        #   13. generate pool price array and pool price duration curve array
        self.generatePoolPriceArrays()

        #   14. compute wholesale price
        self.computeWholesalePrice()

        #   15. compute levellized avoided cost of electricity (LACE)
        self.computeLACE()

        return


    def plot(
        self,
        show_flag: bool = True,
        save_flag: bool = False,
        save_path: str = ""
    ) -> None:
        """
        Method to plot mixture planning results.

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

        #   1. plot load time series
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.plot(
            self.time_array_hrs,
            self.demand_array,
            zorder=2,
            label="Demand"
        )

        if len(self.renewable_production_dict) > 0:
            plt.plot(
                self.time_array_hrs,
                self.residual_demand_array,
                zorder=2,
                alpha=0.8,
                label="Residual Demand"
            )

        plt.xlim(self.time_array_hrs[0], self.time_array_hrs[-1])
        plt.xlabel(r"Time Elapsed [hours]")
        plt.ylabel("Load [" + self.power_units_str + "]")
        plt.legend()
        plt.tight_layout()

        if save_flag:
            fig_path = save_path + "demand_time_series.png"

            plt.savefig(
                fig_path,
                format="png",
                dpi=128
            )

            print(
                "MixturePlanner.plot():\tdemand time series plot saved to",
                fig_path
            )

        #   2. plot load duration curves
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.plot(
            self.duration_x_array,
            self.load_duration_array,
            zorder=2,
            label="Load Duration Curve"
        )

        if len(self.renewable_production_dict) > 0:
            plt.fill_between(
                self.duration_x_array,
                self.load_duration_array,
                y2=self.residual_load_duration_array,
                color="green",
                hatch="|",
                alpha=0.25,
                zorder=1,
                label="Renewable Production"
            )

            plt.plot(
                self.duration_x_array,
                self.residual_load_duration_array,
                linestyle="--",
                zorder=3,
                label="Residual Load Duration Curve"
            )

        base_height = 0
        max_height = 0

        key_list = [key for key in self.supply_stack_dict]
        N = len(self.supply_stack_dict[key_list[0]])

        for i in range(0, N):
            if self.supply_stack_dict[key_list[1]][i] <= 0:
                continue

            key = self.supply_stack_dict[key_list[0]][i]
            max_height += self.supply_stack_dict[key_list[1]][i]

            height_array = self.cf_2_residual_load_interp(
                self.capacity_factor_array
            )

            for j in range(0, len(height_array)):
                if height_array[j] > max_height:
                    height_array[j] = max_height

            base_array = base_height * np.ones(len(height_array))

            boolean_mask = height_array > base_height

            plt.fill_between(
                self.capacity_factor_array[boolean_mask],
                height_array[boolean_mask],
                y2=base_array[boolean_mask],
                alpha=0.25,
                zorder=1,
                label=key + " Energy"
            )

            base_height = max_height

        plt.scatter(
            self.changeover_capacity_factor_array,
            self.cf_2_residual_load_interp(
                self.changeover_capacity_factor_array
            ),
            edgecolor="black",
            facecolors="none",
            marker="o",
            zorder=5,
            label="Screening Changeover Points"
        )

        plt.xlim(0, 1)
        plt.xlabel("Proportion of Year [  ]")
        plt.ylim(0, 1.02 * np.max(self.load_duration_array))
        plt.ylabel("Power [" + self.power_units_str + "]")
        plt.legend()
        plt.tight_layout()

        if save_flag:
            fig_path = save_path + "load_duration_curves.png"

            plt.savefig(
                fig_path,
                format="png",
                dpi=128
            )

            print(
                "MixturePlanner.plot():\tload duration curves plot saved to",
                fig_path
            )

        #   3. plot screening curves and minimum cost frontier
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)

        for key in self.screening_curve_dict.keys():
            plt.plot(
                self.capacity_factor_array,
                self.screening_curve_dict[key],
                zorder=2,
                label=key
            )

        plt.plot(
            self.capacity_factor_array,
            self.minimum_cost_frontier,
            color="black",
            alpha=0.5,
            linestyle="--",
            zorder=3,
            label="Minimum Cost Frontier"
        )
        plt.scatter(
            self.changeover_capacity_factor_array,
            self.changeover_cost_array,
            edgecolor="black",
            facecolors="none",
            marker="o",
            zorder=4,
            label="Screening Changeover Points"
        )
        plt.xlim(0, 1)
        plt.xlabel("Capacity Factor [  ]")
        plt.ylabel(
            "Cost ["
            + self.currency_units_str
            + "/"
            + self.power_units_str
            + r"c-yr]"
        )
        plt.legend()
        plt.tight_layout()

        if save_flag:
            fig_path = save_path + "screening_curves.png"

            plt.savefig(
                fig_path,
                format="png",
                dpi=128
            )

            print(
                "MixturePlanner.plot():\tscreening curves plot saved to",
                fig_path
            )

        #   4. renewable production duration curves
        if len(self.renewable_production_dict) > 0:
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, which="both", zorder=1)

            y_max = 0

            for key in self.renewable_production_duration_dict.keys():
                plt.plot(
                    self.duration_x_array,
                    self.renewable_production_duration_dict[key],
                    zorder=2,
                    label=key
                )

                if self.renewable_production_duration_dict[key][0] > y_max:
                    y_max = self.renewable_production_duration_dict[key][0]

            plt.xlim(0, 1)
            plt.xlabel("Proportion of Year [  ]")
            plt.ylim(0, 1.02 * y_max)
            plt.ylabel("Production [" + self.power_units_str + "]")
            plt.legend()
            plt.tight_layout()

            if save_flag:
                fig_path = save_path + "renewable_production_duration_curves.png"

                plt.savefig(
                    fig_path,
                    format="png",
                    dpi=128
                )

                print(
                    "MixturePlanner.plot():\trenewable production duration"
                    "curves plot saved to",
                    fig_path
                )

        #   5. plot supply stack
        key_list = [key for key in self.supply_stack_dict.keys()]

        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)

        for i in range(0, len(self.supply_stack_dict[key_list[0]])):
            if self.supply_stack_dict[key_list[1]][i] <= 0:
                continue

            plt.bar(
                self.supply_stack_dict[key_list[2]][i]
                - self.supply_stack_dict[key_list[1]][i],
                self.supply_stack_dict[key_list[3]][i],
                width=self.supply_stack_dict[key_list[1]][i],
                align="edge",
                zorder=2,
                label=self.supply_stack_dict[key_list[0]][i]
            )

        plt.xlim(0, self.supply_stack_dict[key_list[2]][-1])
        plt.xlabel(key_list[2])
        plt.ylabel(key_list[3])
        plt.legend()
        plt.tight_layout()

        if save_flag:
            fig_path = save_path + "supply_stack.png"

            plt.savefig(
                fig_path,
                format="png",
                dpi=128
            )

            print(
                "MixturePlanner.plot():\tsupply stack plot saved to",
                fig_path
            )

        #   6. plot pool price time series
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.plot(
            self.time_array_hrs,
            self.pool_price_array,
            zorder=2
        )

        plt.xlim(self.time_array_hrs[0], self.time_array_hrs[-1])
        plt.xlabel(r"Time Elapsed [hours]")
        plt.ylabel(
            "Pool Price [{}/{}h]".format(
                self.currency_units_str,
                self.power_units_str
            )
        )
        plt.tight_layout()

        if save_flag:
            fig_path = save_path + "pool_price_time_series.png"

            plt.savefig(
                fig_path,
                format="png",
                dpi=128
            )

            print(
                "MixturePlanner.plot():\tpool price time series plot saved to",
                fig_path
            )

        #   7. plot pool price duration curve
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.plot(
            self.duration_x_array,
            self.pool_price_duration_array,
            zorder=2
        )

        plt.xlim(0, 1)
        plt.xlabel("Proportion of Year [  ]")
        plt.ylim(0, 1.02 * np.max(self.pool_price_duration_array))
        plt.ylabel(
            "Pool Price [{}/{}h]".format(
                self.currency_units_str,
                self.power_units_str
            )
        )
        plt.tight_layout()

        if save_flag:
            fig_path = save_path + "pool_price_duration_curves.png"

            plt.savefig(
                fig_path,
                format="png",
                dpi=128
            )

            print(
                "MixturePlanner.plot():\tpool price duration curve plot saved to",
                fig_path
            )

        #   8. show
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

        print()

        #   1. technology changeover points
        print("Technology Changeover Points (for cost-optimal sizing):")

        for i in range(0, len(self.changeover_ref_key_array) - 1):
            print(
                "\t",
                self.changeover_ref_key_array[i]
                + " to "
                + self.changeover_ref_key_array[i + 1],
                ": Capacity Factor ",
                round(self.changeover_capacity_factor_array[i], 3)
            )

        #   2. build override string
        if len(self.sizing_override_array) > 0:
            override_str = "(sizing override):"

        else:
            override_str = "(for cost-optimal sizing)"

        #   2. system sizing
        key_list = [key for key in self.supply_stack_dict.keys()]

        print()
        print("System Sizing " + override_str)

        for key in self.supply_stack_dict[key_list[0]]:
            print(
                "\t",
                key,
                ":",
                round(self.sizing_dict[key], 3),
                self.power_units_str
            )

        #   3. system total demand and production
        print()
        print(
            "System Total Demand:",
            round(self.total_demand, 3),
            self.power_units_str + "h"
        )

        print()
        print("System Production " + override_str)

        for key in self.production_dict.keys():
            print(
                "\t",
                key,
                ":",
                round(self.production_dict[key], 3),
                self.power_units_str + "h"
            )

        #   4. system capacity factors
        print()
        print("System Capacity Factors " + override_str)

        for key in self.supply_stack_dict[key_list[0]]:
            print(
                "\t",
                key,
                ":",
                round(self.tech_capacity_factor_dict[key], 3)
            )

        #   5. system costs
        print()
        print("System Costs " + override_str)

        for key in self.tech_cost_dict.keys():
            print(
                "\t",
                key,
                ":",
                round(self.tech_cost_dict[key], 3),
                self.currency_units_str
                + "/"
                + self.power_units_str
                + "c-yr"
            )

        #   6. supply stack
        key_list = [key for key in self.supply_stack_dict.keys()]

        print()
        print("Supply Stack:")

        for i in range(0, len(self.supply_stack_dict[key_list[0]])):
            print(
                "\t",
                " + ".join(self.supply_stack_dict[key_list[0]][0 : i + 1]),
                ":",
                "up to {} {}".format(
                    round(self.supply_stack_dict[key_list[2]][i], 3),
                    self.power_units_str
                ),
                "at {} {}/{}h".format(
                    round(self.supply_stack_dict[key_list[3]][i], 3),
                    self.currency_units_str,
                    self.power_units_str
                )
            )

        #   7. wholesale price
        print()
        print(
            "Wholesale Price:",
            str(round(self.wholesale_price, 3)),
            "{}/{}h".format(
                self.currency_units_str,
                self.power_units_str
            )
        )

        #   8. capacity price
        print()
        print(
            "Capacity Price:",
            str(round(self.capacity_price, 3)),
            "{}/{}".format(
                self.currency_units_str,
                self.power_units_str
            )
        )

        #   9. LACE
        if len(self.renewable_production_dict) > 0:
            print()
            print("LACE:")

            for key in self.LACE_dict.keys():
                print(
                    "\t",
                    key,
                    "(capacity credit = {} {}):".format(
                        round(self.LACE_dict[key][0], 3),
                        self.power_units_str
                    ),
                    str(round(self.LACE_dict[key][1], 3)),
                    "{}/{}h".format(
                        self.currency_units_str,
                        self.power_units_str
                    )
                )

        return


if __name__ == "__main__":
    print("TESTING:\tMixturePlanner")
    print()

    good_time_array_hrs = np.linspace(0, 8759, 8760)
    good_demand_array = np.random.rand(8760)

    good_renewable_production_dict = {
        "Solar": np.random.rand(8760),
        "Wind": np.random.rand(8760),
    }

    good_screening_curve_dict = {
        "Coal": 1000 * (105 * np.linspace(0, 1, 1000) + 140),
        "Gas": 1000 * (325 * np.linspace(0, 1, 1000) + 50),
        "Combined Cycle": 1000 * (95 * np.linspace(0, 1, 1000) + 155)
    }

    #   1. passing a bad time array (not 8760 hours)
    try:
        bad_time_array_hrs = np.linspace(0, 729, 730)

        MixturePlanner(
            bad_time_array_hrs,
            good_demand_array,
            good_renewable_production_dict,
            good_screening_curve_dict
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   2. passing a bad time array (not strictly increasing)
    try:
        bad_time_array_hrs = np.random.rand(8760)

        MixturePlanner(
            bad_time_array_hrs,
            good_demand_array,
            good_renewable_production_dict,
            good_screening_curve_dict
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   3. passing a bad demand array (not same length as time array)
    try:
        bad_demand_array = np.random.rand(730)

        MixturePlanner(
            good_time_array_hrs,
            bad_demand_array,
            good_renewable_production_dict,
            good_screening_curve_dict
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   4. passing a bad demand array (not strictly non-negative)
    try:
        bad_demand_array = -1 * good_demand_array

        MixturePlanner(
            good_time_array_hrs,
            bad_demand_array,
            good_renewable_production_dict,
            good_screening_curve_dict
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   5. passing a bad renewable production dict (not same length as time array)
    try:
        bad_renewable_production_dict = {
            "Solar": np.random.rand(8760),
            "Wind": np.random.rand(730),
        }

        MixturePlanner(
            good_time_array_hrs,
            good_demand_array,
            bad_renewable_production_dict,
            good_screening_curve_dict
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   6. passing a bad renewable production dict (not strictly non-negative)
    try:
        bad_renewable_production_dict = {
            "Solar": -1 * np.random.rand(8760),
            "Wind": np.random.rand(8760),
        }

        MixturePlanner(
            good_time_array_hrs,
            good_demand_array,
            bad_renewable_production_dict,
            good_screening_curve_dict
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   7. passing a bad screening curve dict (empty)
    try:
        bad_screening_curve_dict = {}

        MixturePlanner(
            good_time_array_hrs,
            good_demand_array,
            good_renewable_production_dict,
            bad_screening_curve_dict
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   8. passing a bad screening curve dict (arrays of different size)
    try:
        bad_screening_curve_dict = {
            "Coal": 105 * np.linspace(0, 1, 1000) + 140,
            "Gas": 325 * np.linspace(0, 1, 500) + 50,
            "Combined Cycle": 95 * np.linspace(0, 1, 1000) + 155
        }

        MixturePlanner(
            good_time_array_hrs,
            good_demand_array,
            good_renewable_production_dict,
            bad_screening_curve_dict
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   9. passing a bad screening curve dict (not strictly positive)
    try:
        bad_screening_curve_dict = {
            "Coal": 105 * np.linspace(0, 1, 1000) + 140,
            "Gas": 325 * np.linspace(0, 1, 1000) + 50,
            "Combined Cycle": 95 * np.linspace(0, 1, 1000) - 155
        }

        MixturePlanner(
            good_time_array_hrs,
            good_demand_array,
            good_renewable_production_dict,
            bad_screening_curve_dict
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   9. passing a bad sizing override array (wrong size)
    try:
        bad_sizing_override_array = [100, 100]

        MixturePlanner(
            good_time_array_hrs,
            good_demand_array,
            good_renewable_production_dict,
            good_screening_curve_dict,
            sizing_override_array=bad_sizing_override_array
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   10. passing a bad sizing override array (negative sizing)
    try:
        bad_sizing_override_array = [100, 100, -100]

        MixturePlanner(
            good_time_array_hrs,
            good_demand_array,
            good_renewable_production_dict,
            good_screening_curve_dict,
            sizing_override_array=bad_sizing_override_array
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   11. good construction
    test_load_dataframe = pd.read_csv("test_data/test_demand_data.csv")
    feature_list = list(test_load_dataframe)

    test_time_array_hours = test_load_dataframe[feature_list[0]].values
    test_demand_array_MW = test_load_dataframe[feature_list[1]].values

    test_renewable_production_dict_MW = {
        "Solar": 150 * np.random.rand(8760),
        "Wind": 150 * np.random.rand(8760)
    }

    test_screening_curve_dict_CAD_MWc_yr = {
        "Coal": 1000 * (105 * np.linspace(0, 1, 1000) + 140),
        "Gas": 1000 * (325 * np.linspace(0, 1, 1000) + 50),
        #"Combined Cycle": 1000 * (60 * np.linspace(0, 1, 1000) + 170),
        "Combined Cycle": 1000 * (95 * np.linspace(0, 1, 1000) + 155),
        #"Combined Cycle": 1000 * (95 * np.linspace(0, 1, 1000) + 145)
    }

    test_mixture_planner = MixturePlanner(
        test_time_array_hours,
        test_demand_array_MW,
        test_renewable_production_dict_MW,
        test_screening_curve_dict_CAD_MWc_yr,
        sizing_override_array=[],
        power_units_str="MW",
        currency_units_str="CAD"
    )

    test_mixture_planner.run()
    print(test_mixture_planner)

    #   12. assert that sum of system productions is matching total demand,
    #       plot results
    total_production = 0

    for key in test_mixture_planner.production_dict:
        total_production += test_mixture_planner.production_dict[key]

    assert(
        abs(test_mixture_planner.total_demand - total_production)
        / test_mixture_planner.total_demand < 0.001
    )

    test_mixture_planner.plot()
    print()
    print("==================================================================")
    print()

    #   13. test sizing override
    test_mixture_planner = MixturePlanner(
        test_time_array_hours,
        test_demand_array_MW,
        test_renewable_production_dict_MW,
        test_screening_curve_dict_CAD_MWc_yr,
        sizing_override_array=[1200, 850, 250],
        power_units_str="MW",
        currency_units_str="CAD"
    )

    test_mixture_planner.run()
    print(test_mixture_planner)

    #   14. assert that sum of system productions is matching total demand,
    #       plot results
    total_production = 0

    for key in test_mixture_planner.production_dict:
        total_production += test_mixture_planner.production_dict[key]

    assert(
        abs(test_mixture_planner.total_demand - total_production)
        / test_mixture_planner.total_demand < 0.001
    )

    #   15. assert that capacity price reflects cheapest dispatchable tech
    assert(
        test_mixture_planner.capacity_price == (
            test_screening_curve_dict_CAD_MWc_yr["Gas"][0]
        )
    )

    test_mixture_planner.plot()

    print()
    print("TESTING:\tMixturePlanner\tPASS")
    print()
