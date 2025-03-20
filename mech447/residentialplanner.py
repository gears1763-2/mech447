"""
Anthony Truelove MASc, P.Eng.  
Python Certified Professional Programmer (PCPP1)

Copyright 2025 - Anthony Truelove  
--> ***SEE LICENSE TERMS [HERE](../../LICENSE)*** <--

A residential planning class, as part of the `mech447` package.
"""


import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HOURS_PER_YEAR = 8760


class Renewable:
    """
    A helper class which serves as a renewable generation asset.
    """

    def __init__(
        self,
        production_array: np.array,
        type_str: str
    ) -> None:
        """
        Renewable class constructor.

        Parameters
        ----------
        production_array: np.array
            An array of production from the renewable asset in each time step.
            SHOULD BE EXACTLY ONE YEAR (i.e., 8760 hours). IS ASSUMED TO HAVE
            THE SAME UNITS AS THE DEMAND ARRAY.

        type_str: str
            This is a string defining what the renewable asset is, for printing
            and plotting.

        Returns
        -------
        None
        """

        self.production_array = np.array(production_array)
        """
        This is an array of how much power the renewable asset is producing in
        each time step.
        """

        self.type_str = type_str
        """
        This is a string defining what the renewable asset is.
        """

        self.dispatch_array = np.zeros(len(production_array))
        """
        This is an array of how much of the production is dispatched directly
        to meet demand.
        """

        self.storage_array = np.zeros(len(production_array))
        """
        This is an array of how much of the production is stored for later use.
        """

        self.curtailment_array = np.zeros(len(production_array))
        """
        This is an array of how much of the production is curtailed. In the
        context of this module, curtailment is sent to the grid.
        """

        return


    def getProduction(self, i: int) -> float:
        """
        Getter method to retrieve the production of this asset for the current
        time step.

        Parameters
        ----------
        i: int
            The index of the current time step.

        Returns
        -------
        float
            The production for the current time step.
        """

        return self.production_array[i]


class Storage:
    """
    A helper class which serves as a storage asset.
    """

    def __init__(
        self,
        energy_capacity: float,
        power_capacity: float = np.inf,
        initial_state_of_charge: float = 0.5,
        charging_efficiency: float = 0.9,
        discharging_efficiency: float = 0.9
    ) -> None:
        """
        Storage class constructor.

        Parameters
        ----------
        energy_capacity: float
            The energy capacity of the storage asset. IS ASSUMED TO HAVE UNITS
            OF DEMAND ARRAY UNITS X HOURS.

        power_capacity: float, optional, default np.inf
            The power capacity of the storage asset. IS ASSUMED TO HAVE THE
            SAME UNITS AS THE DEMAND ARRAY.

        initial_state_of_charge: float, optional, default 0.5
            The initial state of charge of the storage asset.

        charging_efficiency: float, optional, default 0.9
            The charging efficiency of the storage asset.

        discharging_efficiency: float, optional, default 0.9
            The discharging efficiency of the storage asset.

        Returns
        -------
        None
        """

        #   1. init attributes
        self.energy_capacity = energy_capacity
        """
        The energy capacity of the storage asset.
        """

        self.power_capacity = power_capacity
        """
        The power capacity of the storage asset.
        """

        self.initial_state_of_charge = initial_state_of_charge
        """
        The initial state of charge of the storage asset.
        """

        self.charging_efficiency = charging_efficiency
        """
        The charging efficiency of the storage asset.
        """

        self.discharging_efficiency = discharging_efficiency
        """
        The discharging efficiency of the storage asset.
        """

        self.charge = self.initial_state_of_charge * self.energy_capacity
        """
        The current charge of the storage asset.
        """

        self.charge_array = 0
        """
        This is an array that holds the storage asset's charge at the end of
        every time step.
        """

        self.power_array = 0
        """
        This is an array that holds the charging/discharging power of the
        storage asset in each time step. Note that discharging is positive, and
        charging is negative.
        """

        self.total_discharge = 0
        """
        The total amount of energy discharged from the storage asset over the
        modelling horizon.
        """

        #   2. check inputs
        self.__checkInputs()

        return


    def __checkInputs(self) -> None:
        """
        Helper method to check Storage inputs.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        #   1. energy capacity must be strictly positive
        if self.energy_capacity <= 0:
            error_string = "ERROR: Storage.__checkInputs():\t"
            error_string += "energy capacity must be strictly positive (> 0)."

            raise RuntimeError(error_string)

        #   2. power capacity must be strictly positive
        if self.power_capacity <= 0:
            error_string = "ERROR: Storage.__checkInputs():\t"
            error_string += "power capacity must be strictly positive (> 0)."

            raise RuntimeError(error_string)

        #   3. initial state of charge must be between 0 and 1
        if (
            self.initial_state_of_charge < 0
            or self.initial_state_of_charge > 1
        ):
            error_string = "ERROR: Storage.__checkInputs():\t"
            error_string += "initial state of charge must be in [0, 1]."

            raise RuntimeError(error_string)

        #   4. charge efficiency must be between 0 and 1
        if (
            self.charging_efficiency < 0
            or self.charging_efficiency > 1
        ):
            error_string = "ERROR: Storage.__checkInputs():\t"
            error_string += "charging efficiency must be in [0, 1]."

            raise RuntimeError(error_string)

        #   5. discharging effieciency must be between 0 and 1
        if (
            self.discharging_efficiency < 0
            or self.discharging_efficiency > 1
        ):
            error_string = "ERROR: Storage.__checkInputs():\t"
            error_string += "discharging efficiency must be in [0, 1]."

            raise RuntimeError(error_string)


    def getAvailableDischargePower(self, delta_time_hours: float) -> float:
        """
        Method to compute the discharging power available from the storage
        asset.

        Parameters
        ----------
        delta_time_hours: float
            The time delta of the current time step.

        Returns
        -------
        float
            The discharging power available from the storage asset.
        """

        #   1. get available from charge
        available_discharge = self.charge / delta_time_hours
        available_discharge *= self.discharging_efficiency

        #   2. enforce power capacity
        if available_discharge > self.power_capacity:
            available_discharge = self.power_capacity

        return available_discharge


    def commitDischarge(
        self,
        i: int,
        delta_time_hours: float,
        discharge: float
    ) -> None:
        """
        Method to commit a discharging action for the current time step. Note
        that discharging powers are logged as positive.

        Parameters
        ----------
        i: int
            The index of the current time step.

        delta_time_hours: float
            The time delta of the current time step.

        discharge: float
            The discharging power for the current time step.

        Returns
        -------
        None
        """

        #   1. log discharging power
        self.power_array[i] = discharge

        #   2. decrement charge
        self.charge -= (
            (discharge * delta_time_hours)
            / self.discharging_efficiency
        )

        #   3. log end of time step charge
        self.charge_array[i] = self.charge

        #   4. log total discharge
        self.total_discharge += discharge * delta_time_hours

        return


    def getAcceptableChargePower(self, delta_time_hours: float) -> float:
        """
        Method to compute the charging power that can be accepted by the
        storage asset.

        Parameters
        ----------
        delta_time_hours: float
            The time delta of the current time step.

        Returns
        -------
        float
            The charging power that can be accepted by the storage asset.
        """

        #   1. get acceptable from charge
        acceptable_charge = (
            (self.energy_capacity - self.charge)
            / delta_time_hours
        )

        acceptable_charge /= self.charging_efficiency

        #   2. enforce power capacity
        if acceptable_charge > self.power_capacity:
            acceptable_charge = self.power_capacity

        return acceptable_charge


    def commitCharge(
        self,
        i: int,
        delta_time_hours: float,
        charge: float
    ) -> None:
        """
        Method to commit a charging action for the current time step. Note that
        charging powers are logged as negative.

        Parameters
        ----------
        i: int
            The index of the current time step.

        delta_time_hours: float
            The time delta of the current time step.

        charge: float
            The charging power for the current time step.

        Returns
        -------
        None
        """

        #   1. log charging power
        self.power_array[i] = -1 * charge

        #   2. increment charge
        self.charge += (
            (charge * delta_time_hours)
            * self.charging_efficiency
        )

        #   3. log end of time step charge
        self.charge_array[i] = self.charge

        return


class ResidentialPlanner:
    """
    A class which takes in a demand time series, an energy tariffs dict, a list
    of renewable generation assets (optional), and a storage asset (optional),
    and then models the operation of a residential unit with the given system.
    """

    def __init__(
        self,
        time_array_hours: np.array,
        demand_array: np.array,
        energy_tariffs_dict: dict[float, float],
        renewable_list: list[Renewable] = [],
        storage: Storage = None,
        power_units_str: str = "MW",
        currency_units_str: str = "CAD"
    ) -> None:
        """
        ResidentialPlanner class construtor.

        Parameters
        ----------
        time_array_hours: np.array
            This is an array of points in time [hours]. This defines all time
            series. SHOULD BE EXACTLY ONE YEAR (i.e., 8760 hours).

        demand_array: np.array
            This is an array of average demand (power) values corresponding to
            each point of time_array_hours.

        energy_tariffs_dict: dict[float, float]
            A dictionary of the energy tariffs being charged.

        renewable_list: list[Renewable], optional, default []
            A list of renewable generation assets included in the residential
            system. Note that the order of the assets in the list defines
            their merit order.

        storage: Storage, optional, default None
            A storage asset included in the residential system.

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
        self.time_array_hours = np.array(time_array_hours)
        """
        This is an array of points in time [hours]. This defines the time
        series inputs of demand_array and the elements of
        renewable_production_dict. SHOULD BE EXACTLY ONE YEAR (i.e., 
        8760 hours).
        """

        self.delta_time_array_hours = np.diff(self.time_array_hours)
        self.delta_time_array_hours = np.append(
            self.delta_time_array_hours[0],
            self.delta_time_array_hours
        )
        """
        This is an array of time deltas, for use in integrating power time
        series to get energy amounts.
        """

        self.storage = storage
        """
        A storage asset (i.e., `Storage`) included in the residential system.
        """

        if self.storage is not None:
            self.storage.charge_array = np.zeros(len(self.time_array_hours))
            self.storage.power_array = np.zeros(len(self.time_array_hours))

        #   2. check inputs
        self.__checkInputs(
            demand_array,
            energy_tariffs_dict,
            renewable_list,
            storage
        )

        #   3. init attributes
        self.demand_array = np.array(demand_array)
        """
        This is an array of average demand (power) values corresponding to
        each point of time_array_hours.
        """

        self.energy_tariffs_dict = energy_tariffs_dict
        """
        A dictionary of the energy tariffs being charged. Keys are power lower
        bounds, and values are cost per unit energy.
        """

        self.renewable_list = renewable_list
        """
        A list of renewable generation assets (i.e. `Renewable`) included in
        the residential system.
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

        self.residual_demand_array = np.zeros(len(self.demand_array))
        """
        This is an array of the residual demand (load) after the renewable
        productions have all been deducted from the corresponding demand.
        """

        self.net_demand_array = np.zeros(len(self.demand_array))
        """
        This is an array of the net demand (load) after the storage discharge
        (if any) has been deducted from the corresponding residual demand.
        """

        self.grid_power_array = np.zeros(len(self.demand_array))
        """
        This is an array of the power flow from the grid to the residence.
        Note that positive values are flows from the grid to the residence, and
        negative values are flows from the residence to the grid.
        """

        self.grid_energy_price_array = np.zeros(len(self.demand_array))
        """
        This is an array of the grid energy price in each time step.
        """

        self.grid_energy_cost_array = np.zeros(len(self.demand_array))
        """
        This is an array of the grid energy cost in each time step. Note that
        positive costs are amounts paid by the residence, whereas negative costs
        are amounts paid to the residence.
        """

        self.total_energy_demand = 0
        """
        The total energy demanded by the residence over the entire modelling
        horizon.
        """

        self.total_energy_cost = 0
        """
        The total energy cost to the residence over the entire modelling
        horizon.
        """

        self.average_cost_of_energy = 0
        """
        The average cost of energy over the entire modelling horizon. Is taken
        to be simply the total energy cost over the total energy demand.
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
        energy_tariffs_dict: dict[float, float],
        renewable_list: list[Renewable],
        storage: Storage
    ) -> None:
        """
        Helper method to check __init__ inputs.

        Parameters
        ----------
        demand_array: np.array
            This is an array of average demand (power) values corresponding to
            each point of time_array_hours.

        energy_tariffs_dict: dict[float, float]
            A dictionary of the energy tariffs being charged.

        renewable_list: list[Renewable], optional, default []
            A list of renewable generation assets included in the residential
            system. Note that the order of the assets in the list defines
            their merit order.

        storage: Storage, optional, default None
            A storage asset included in the residential system.

        Returns
        -------
        None
        """

        #   1. time array must be exactly one year (i.e., 8760 hours)
        if len(self.time_array_hours) != 8760:
            error_string = "ERROR: ResidentialPlanner.__checkInputs():\t"
            error_string += "time array [hours] must be of length 8760 ("
            error_string += "assuming 8760 hours/year)."

            raise RuntimeError(error_string)

        #   2. time array must be strictly increasing
        boolean_mask = np.diff(self.time_array_hours) <= 0

        if boolean_mask.any():
            error_string = "ERROR: ResidentialPlanner.__checkInputs():\t"
            error_string += "time array [hours] must be strictly increasing"
            error_string += " (t[i + 1] > t[i] for all i)."

            raise RuntimeError(error_string)

        #   3. demand array must be same length as time array
        if len(demand_array) != len(self.time_array_hours):
            error_string = "ERROR: ResidentialPlanner.__checkInputs():\t"
            error_string += "demand array and time array [hours] must "
            error_string += "be the same length."

            raise RuntimeError(error_string)

        #   4. demand array must be non-negative
        boolean_mask = demand_array < 0

        if boolean_mask.any():
            error_string = "ERROR: ResidentialPlanner.__checkInputs():\t"
            error_string += "demand array must be strictly non-negative"
            error_string += " (x[i] >= 0 for all i)."

            raise RuntimeError(error_string)

        #   5. energy tariffs dict must have strictly increasing keys
        key_list = [key for key in energy_tariffs_dict.keys()]
        boolean_mask = np.diff(key_list) <= 0

        if boolean_mask.any():
            error_string = "ERROR: ResidentialPlanner.__checkInputs():\t"
            error_string += "energy tariffs dict keys must be strictly"
            error_string += " increasing (key[i + 1] > key[i] for all i)."

            raise RuntimeError(error_string)

        #   6. check renewable production arrays
        for renewable in renewable_list:
            #   6.1. production array must be same length as time array
            if len(renewable.production_array) != len(self.time_array_hours):
                error_string = "ERROR: ResidentialPlanner.__checkInputs():\t"
                error_string += "renewable production array and time array "
                error_string += "[hours] must be the same length. Production "
                error_string += "array of `"
                error_string += renewable.type_str
                error_string += "` is not."

                raise RuntimeError(error_string)

            #   6.2. production array must be non-negative
            boolean_mask = renewable.production_array < 0

            if boolean_mask.any():
                error_string = "ERROR: ResidentialPlanner.__checkInputs():\t"
                error_string += "renewable production array must be strictly "
                error_string += "non-negative (x[i] >= 0 for all i). Production "
                error_string += "array of `"
                error_string += renewable.type_str
                error_string += "` is not."

                raise RuntimeError(error_string)

        return


    def getResidualDemand(self, i: int) -> None:
        """
        Helper method to compute and return the residual demand (load) for the
        given time step. Residual demand is defined as

        $$ \\widehat{L} = L - \\sum_kR_k $$

        where $\\widehat{L}$ is residual demand, $L$ is demand, and $R_k$ is 
        the production from the $k^\\textrm{th}$ renewable asset.

        Parameters
        ----------
        i: int
            The index of the current time step.

        Returns
        -------
        None
        """

        #   1. init residual demand to demand
        residual_demand = self.demand_array[i]

        #   2. iterate over renewables, deduct production
        for renewable in self.renewable_list:
            residual_demand -= renewable.getProduction(i)

        #   3. log
        self.residual_demand_array[i] = residual_demand

        return


    def computeDispatchCurtailment(self, i: int) -> None:
        """
        Helper method to compute the dispatch and curtailment for each
        renewable asset.

        Parameters
        ----------
        i: int
            The index of the current time step.

        Returns
        -------
        None
        """

        #   1. compute total production
        total_production = 0

        for renewable in self.renewable_list:
            total_production += renewable.production_array[i]

        #   2. compute dispatch and curtailment
        if total_production <= self.demand_array[i]:
            for renewable in self.renewable_list:
                renewable.dispatch_array[i] = renewable.production_array[i]

        else:
            for renewable in self.renewable_list:
                renewable.dispatch_array[i] = (
                    (self.demand_array[i] / total_production)
                    * renewable.production_array[i]
                )

                renewable.curtailment_array[i] = (
                    renewable.production_array[i]
                    - renewable.dispatch_array[i]
                )

        return


    def handleStorageChargeDischarge(self, i) -> None:
        """
        Helper method to handle the charging and discharging of the storage asset
        (if any). The logic here is simple: if residual load is positive, then
        the storage asset attempts to discharge, otherwise it attempts to
        charge using excess renewable production.

        If the storage is discharging, then net demand (load) is defined as

        $$ L_\\textrm{net} = \\widehat{L} - P_\\textrm{D} $$

        where $L_\\textrm{net}$ is net demand, $\\widehat{L}$ is residual
        demand, and $P_\\textrm{D}$ is storage discharge power.

        If the storage is charging, then net demand (load) is equal to the
        residual demand.

        Parameters
        ----------
        i: int
            The index of the current time step.

        Returns
        -------
        None
        """

        #   1. init net demand to residual demand
        net_demand = self.residual_demand_array[i]

        #   2. if storage == None, log and return
        if self.storage is None:
            self.net_demand_array[i] = net_demand
            return

        #   3. handle discharging, update net demand
        if self.residual_demand_array[i] >= 0:
            available_discharge = self.storage.getAvailableDischargePower(
                 self.delta_time_array_hours[i]
            )

            discharge = available_discharge

            if discharge > self.residual_demand_array[i]:
                discharge = self.residual_demand_array[i]

            self.storage.commitDischarge(
                i,
                self.delta_time_array_hours[i],
                discharge
            )

            net_demand -= discharge

        #   4. handle charging
        else:
            acceptable_charge = self.storage.getAcceptableChargePower(
                 self.delta_time_array_hours[i]
            )

            total_curtailment = 0

            for renewable in self.renewable_list:
                total_curtailment += renewable.curtailment_array[i]

            if total_curtailment <= acceptable_charge:
                charge = total_curtailment

                for renewable in self.renewable_list:
                    renewable.storage_array[i] = renewable.curtailment_array[i]
                    renewable.curtailment_array[i] = 0

            else:
                charge = acceptable_charge

                for renewable in self.renewable_list:
                    renewable.storage_array[i] = (
                        (acceptable_charge / total_curtailment)
                        * renewable.curtailment_array[i]
                    )

                    renewable.curtailment_array[i] -= renewable.storage_array[i]

            self.storage.commitCharge(
                i,
                self.delta_time_array_hours[i],
                charge
            )

        #   5. log
        self.net_demand_array[i] = net_demand

        return


    def handleGrid(self, i: int) -> None:
        """
        Helper method to handle the drawing of power from the grid. Note that
        positive values are flows from the grid to the residence, and negative
        values are flows from the residence to the grid.

        Parameters
        ----------
        i: int
            The index of the current time step.

        Returns
        -------
        None
        """

        #   1. log grid power
        self.grid_power_array[i] = self.net_demand_array[i]

        #   2. log energy price
        key_array = np.flip(
            np.array([key for key in self.energy_tariffs_dict.keys()])
        )

        for key in key_array:
            if self.grid_power_array[i] >= key:
                break

        energy = self.grid_power_array[i] * self.delta_time_array_hours[i]
        self.grid_energy_price_array[i] = self.energy_tariffs_dict[key]

        self.grid_energy_cost_array[i] = (
            abs(energy) * self.grid_energy_price_array[i]
        )

        return


    def getTotalEnergyDemand(self) -> None:
        """
        Helper method to compute the total energy demanded by the residence.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.total_energy_demand = np.dot(
            self.demand_array,
            self.delta_time_array_hours
        )

        return

    def getTotalEnergyCost(self) -> None:
        """
        Helper method to compute the total energy cost to the residence.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.total_energy_cost += np.sum(self.grid_energy_cost_array)

        return


    def getAverageCostOfEnergy(self) -> None:
        """
        Helper method to compute the average cost of energy, which is taken to
        be simply the total energy cost over the total energy demand.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.average_cost_of_energy = (
            self.total_energy_cost / self.total_energy_demand
        )


    def run(self,) -> None:
        """
        Method to run the residential planner and generate results.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        for i in range(0, len(self.time_array_hours)):
            #   1. get residual demand, log
            self.getResidualDemand(i)

            #   2. compute dispatch and curtailment, log
            self.computeDispatchCurtailment(i)

            #   3. handle storage charge/discharge, log net load
            self.handleStorageChargeDischarge(i)

            #   4. handle grid, log flow and price
            self.handleGrid(i)

        #   5. compute total energy demand
        self.getTotalEnergyDemand()

        #   6. get total energy cost
        self.getTotalEnergyCost()

        #   7. get average cost of energy
        self.getAverageCostOfEnergy()

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

        #   1. plot demand time series
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.plot(
            self.time_array_hours,
            self.demand_array,
            zorder=2,
            label="Demand"
        )

        if len(self.renewable_list) > 0:
            plt.plot(
                self.time_array_hours,
                self.residual_demand_array,
                zorder=2,
                alpha=0.8,
                label="Residual Demand"
            )

        if self.storage is not None:
            plt.plot(
                self.time_array_hours,
                self.net_demand_array,
                zorder=2,
                alpha=0.8,
                label="Net Demand"
            )

        plt.xlim(self.time_array_hours[0], self.time_array_hours[-1])
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
                "ResidentialPlanner.plot():\tdemand time series plot saved to",
                fig_path
            )

        #   2. plot grid power time series
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.plot(
            self.time_array_hours,
            self.grid_power_array,
            zorder=2
        )
        plt.xlim(self.time_array_hours[0], self.time_array_hours[-1])
        plt.xlabel(r"Time Elapsed [hours]")
        plt.ylabel("Grid Power [" + self.power_units_str + "]")
        plt.tight_layout()

        if save_flag:
            fig_path = save_path + "grid_power_time_series.png"

            plt.savefig(
                fig_path,
                format="png",
                dpi=128
            )

            print(
                "ResidentialPlanner.plot():\tgrid power time series plot saved to",
                fig_path
            )

        #   3. plot grid energy cost time series
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.plot(
            self.time_array_hours,
            self.grid_energy_cost_array,
            zorder=2
        )
        plt.xlim(self.time_array_hours[0], self.time_array_hours[-1])
        plt.xlabel(r"Time Elapsed [hours]")
        plt.ylabel("Grid Energy Cost [" + self.currency_units_str + "]")
        plt.tight_layout()

        if save_flag:
            fig_path = save_path + "grid_energy_cost_time_series.png"

            plt.savefig(
                fig_path,
                format="png",
                dpi=128
            )

            print(
                "ResidentialPlanner.plot():\tgrid energy cost time series plot saved to",
                fig_path
            )

        #   4. plot grid energy price time series
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
        plt.plot(
            self.time_array_hours,
            self.grid_energy_price_array,
            zorder=2
        )
        plt.xlim(self.time_array_hours[0], self.time_array_hours[-1])
        plt.xlabel(r"Time Elapsed [hours]")
        plt.ylabel(
            "Grid Energy Price [{}/{}h]".format(
                self.currency_units_str,
                self.power_units_str
            )
        )
        plt.tight_layout()

        if save_flag:
            fig_path = save_path + "grid_energy_price_time_series.png"

            plt.savefig(
                fig_path,
                format="png",
                dpi=128
            )

            print(
                "ResidentialPlanner.plot():\tgrid energy price time series plot saved to",
                fig_path
            )

        if len(self.renewable_list) > 0:
            #   5. plot renewable production time series
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, which="both", zorder=1)

            for renewable in self.renewable_list:
                    plt.plot(
                        self.time_array_hours,
                        renewable.production_array,
                        zorder=2,
                        label="{} Production".format(renewable.type_str)
                    )

            plt.xlim(self.time_array_hours[0], self.time_array_hours[-1])
            plt.xlabel(r"Time Elapsed [hours]")
            plt.ylabel("Power [" + self.power_units_str + "]")
            plt.legend()
            plt.tight_layout()

            if save_flag:
                fig_path = save_path + "renewable_production_time_series.png"

                plt.savefig(
                    fig_path,
                    format="png",
                    dpi=128
                )

                print(
                    "ResidentialPlanner.plot():\trenewable production time series",
                    "plot saved to",
                    fig_path
                )

            #   6. plot renewable dispatch time series
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, which="both", zorder=1)

            for renewable in self.renewable_list:
                    plt.plot(
                        self.time_array_hours,
                        renewable.dispatch_array,
                        zorder=2,
                        label="{} Dispatch".format(renewable.type_str)
                    )

            plt.xlim(self.time_array_hours[0], self.time_array_hours[-1])
            plt.xlabel(r"Time Elapsed [hours]")
            plt.ylabel("Power [" + self.power_units_str + "]")
            plt.legend()
            plt.tight_layout()

            if save_flag:
                fig_path = save_path + "renewable_dispatch_time_series.png"

                plt.savefig(
                    fig_path,
                    format="png",
                    dpi=128
                )

                print(
                    "ResidentialPlanner.plot():\trenewable dispatch time series",
                    "plot saved to",
                    fig_path
                )

            #   7. plot renewable curtailment time series
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, which="both", zorder=1)

            for renewable in self.renewable_list:
                    plt.plot(
                        self.time_array_hours,
                        renewable.curtailment_array,
                        zorder=2,
                        label="{} Curtailment".format(renewable.type_str)
                    )

            plt.xlim(self.time_array_hours[0], self.time_array_hours[-1])
            plt.xlabel(r"Time Elapsed [hours]")
            plt.ylabel("Power [" + self.power_units_str + "]")
            plt.legend()
            plt.tight_layout()

            if save_flag:
                fig_path = save_path + "renewable_curtailment_time_series.png"

                plt.savefig(
                    fig_path,
                    format="png",
                    dpi=128
                )

                print(
                    "ResidentialPlanner.plot():\trenewable curtailment time series",
                    "plot saved to",
                    fig_path
                )

            #   8. plot renewable storage time series
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, which="both", zorder=1)

            for renewable in self.renewable_list:
                    plt.plot(
                        self.time_array_hours,
                        renewable.storage_array,
                        zorder=2,
                        label="{} Storage".format(renewable.type_str)
                    )

            plt.xlim(self.time_array_hours[0], self.time_array_hours[-1])
            plt.xlabel(r"Time Elapsed [hours]")
            plt.ylabel("Power [" + self.power_units_str + "]")
            plt.legend()
            plt.tight_layout()

            if save_flag:
                fig_path = save_path + "renewable_storage_time_series.png"

                plt.savefig(
                    fig_path,
                    format="png",
                    dpi=128
                )

                print(
                    "ResidentialPlanner.plot():\trenewable storage time series",
                    "plot saved to",
                    fig_path
                )

        if self.storage is not None:
            #   9. plot storage charge time series
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
            plt.plot(
                self.time_array_hours,
                self.storage.charge_array,
                zorder=2
            )
            plt.xlim(self.time_array_hours[0], self.time_array_hours[-1])
            plt.xlabel(r"Time Elapsed [hours]")
            plt.ylabel("Storage Charge [" + self.power_units_str + "h]")
            plt.tight_layout()

            if save_flag:
                fig_path = save_path + "storage_charge_time_series.png"

                plt.savefig(
                    fig_path,
                    format="png",
                    dpi=128
                )

                print(
                    "ResidentialPlanner.plot():\tstorage charge time series",
                    "plot saved to",
                    fig_path
                )

            #   10. plot storage power time series
            plt.figure(figsize=(8, 6))
            plt.grid(color="C7", alpha=0.5, which="both", zorder=1)
            plt.plot(
                self.time_array_hours,
                self.storage.power_array,
                zorder=2
            )
            plt.xlim(self.time_array_hours[0], self.time_array_hours[-1])
            plt.xlabel(r"Time Elapsed [hours]")
            plt.ylabel("Storage Power [" + self.power_units_str + "]")
            plt.tight_layout()

            if save_flag:
                fig_path = save_path + "storage_power_time_series.png"

                plt.savefig(
                    fig_path,
                    format="png",
                    dpi=128
                )

                print(
                    "ResidentialPlanner.plot():\tstorage power time series",
                    "plot saved to",
                    fig_path
                )

        #   11. show
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

        #   1. total energy demand
        print(
            "Total Energy Demand:",
            round(self.total_energy_demand, 3),
            "{}h".format(self.power_units_str)
        )

        #   2. total energy cost
        print()
        print(
            "Total Energy Cost:",
            round(self.total_energy_cost, 2),
            self.currency_units_str
        )

        #   3. wholesale energy cost
        print()
        print(
            "Average Cost of Energy (wholesale):",
            round(self.average_cost_of_energy, 3),
            "{}/{}h".format(
                self.currency_units_str,
                self.power_units_str
            )
        )

        #   4. renewable performance metrics
        print()
        print("Renewable Performance:")

        for renewable in self.renewable_list:
            print("\t{}:".format(renewable.type_str))

            print(
                "\t\tTotal Energy Produced:",
                round(
                    np.dot(
                        renewable.production_array,
                        self.delta_time_array_hours
                    )
                ),
                "{}h".format(self.power_units_str)
            )

            print(
                "\t\tTotal Energy Dispatched:",
                round(
                    np.dot(
                        renewable.dispatch_array,
                        self.delta_time_array_hours
                    )
                ),
                "{}h".format(self.power_units_str)
            )

            print(
                "\t\tTotal Energy Stored:",
                round(
                    np.dot(
                        renewable.storage_array,
                        self.delta_time_array_hours
                    )
                ),
                "{}h".format(self.power_units_str)
            )

            print(
                "\t\tTotal Energy Curtailed:",
                round(
                    np.dot(
                        renewable.curtailment_array,
                        self.delta_time_array_hours
                    )
                ),
                "{}h".format(self.power_units_str)
            )

            print()

        # 5. storage performance metrics
        print()
        print("Storage Performance:")

        print(
            "\t",
            "Total Energy Discharged:",
            round(self.storage.total_discharge),
            "{}h".format(self.power_units_str)
        )

        print(
            "\t",
            "Approximate Number of Cycles:",
            round(
                self.storage.total_discharge / self.storage.energy_capacity
            )
        )

        return


if __name__ == "__main__":
    print("TESTING:\tResidentialPlanner")
    print()

    good_time_array_hours = np.linspace(0, 8759, 8760)
    good_demand_array = np.random.rand(8760)

    good_energy_tariffs_dict = {
        -1 * np.inf: -0.07,
        0: 0.07,
        5: 0.11
    }

    good_renewable_list = []

    good_energy_capacity = 10
    good_power_capacity = 100
    good_initial_SOC = 0
    good_charging_efficiency = 0.9
    good_discharging_efficiency = 0.9

    good_storage = None

    #   1. passing a bad time array (not 8760 hours)
    try:
        bad_time_array_hours = np.linspace(0, 729, 730)

        ResidentialPlanner(
            bad_time_array_hours,
            good_demand_array,
            good_energy_tariffs_dict,
            renewable_list=good_renewable_list,
            storage=good_storage
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   2. passing a bad time array (not strictly increasing)
    try:
        bad_time_array_hours = np.random.rand(8760)

        ResidentialPlanner(
            bad_time_array_hours,
            good_demand_array,
            good_energy_tariffs_dict,
            renewable_list=good_renewable_list,
            storage=good_storage
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

        ResidentialPlanner(
            good_time_array_hours,
            bad_demand_array,
            good_energy_tariffs_dict,
            renewable_list=good_renewable_list,
            storage=good_storage
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

        ResidentialPlanner(
            good_time_array_hours,
            bad_demand_array,
            good_energy_tariffs_dict,
            renewable_list=good_renewable_list,
            storage=good_storage
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   5. passing a bad energy tariffs dict (keys not strictly increasing)
    try:
        bad_energy_tariffs_dict = {
            -1: 0,
            1: 0,
            0: 0
        }

        ResidentialPlanner(
            good_time_array_hours,
            good_demand_array,
            bad_energy_tariffs_dict,
            renewable_list=good_renewable_list,
            storage=good_storage
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   6. passing a bad renewable list (production array of wrong length)
    try:
        bad_solar = Renewable(
            np.random.rand(730),
            "solar"
        )

        good_wind = Renewable(
            np.random.rand(8760),
            "wind"
        )

        bad_renewable_list = [bad_solar, good_wind]

        ResidentialPlanner(
            good_time_array_hours,
            good_demand_array,
            good_energy_tariffs_dict,
            renewable_list=bad_renewable_list,
            storage=good_storage
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   7. passing a bad renewable list (negative production values)
    try:
        good_solar = Renewable(
            np.random.rand(8760),
            "solar"
        )

        bad_wind = Renewable(
            -1 * np.random.rand(8760),
            "wind"
        )

        bad_renewable_list = [good_solar, bad_wind]

        ResidentialPlanner(
            good_time_array_hours,
            good_demand_array,
            good_energy_tariffs_dict,
            renewable_list=bad_renewable_list,
            storage=good_storage
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   8. passing a bad storage energy capacity (not strictly positive)
    try:
        bad_energy_capacity = -1

        Storage(
            bad_energy_capacity,
            power_capacity=good_power_capacity,
            initial_state_of_charge=good_initial_SOC,
            charging_efficiency=good_charging_efficiency,
            discharging_efficiency=good_discharging_efficiency
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   9. passing a bad storage power capacity (not strictly positive)
    try:
        bad_power_capacity = -1

        Storage(
            good_energy_capacity,
            power_capacity=bad_power_capacity,
            initial_state_of_charge=good_initial_SOC,
            charging_efficiency=good_charging_efficiency,
            discharging_efficiency=good_discharging_efficiency
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   10. passing a bad initial SOC (not in [0, 1])
    try:
        bad_initial_SOC = -1

        Storage(
            good_energy_capacity,
            power_capacity=good_power_capacity,
            initial_state_of_charge=bad_initial_SOC,
            charging_efficiency=good_charging_efficiency,
            discharging_efficiency=good_discharging_efficiency
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   11. passing a charging efficiency (not in [0, 1])
    try:
        bad_charging_efficiency = 999

        Storage(
            good_energy_capacity,
            power_capacity=good_power_capacity,
            initial_state_of_charge=good_initial_SOC,
            charging_efficiency=bad_charging_efficiency,
            discharging_efficiency=good_discharging_efficiency
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   12. passing a discharging efficiency (not in [0, 1])
    try:
        bad_discharging_efficiency = -999

        Storage(
            good_energy_capacity,
            power_capacity=good_power_capacity,
            initial_state_of_charge=good_initial_SOC,
            charging_efficiency=good_charging_efficiency,
            discharging_efficiency=bad_discharging_efficiency
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   13. good construction
    test_load_dataframe = pd.read_csv("test_data/test_demand_data.csv")
    feature_list = list(test_load_dataframe)

    test_time_array_hours = test_load_dataframe[feature_list[0]].values
    test_demand_array_MW = test_load_dataframe[feature_list[1]].values

    max_demand_MW = np.max(test_demand_array_MW)
    min_demand_MW = np.min(test_demand_array_MW)

    TARGET_MAX_KW = 8
    TARGET_MIN_KW = 0.2

    m = (
        (TARGET_MAX_KW - TARGET_MIN_KW)
        / (max_demand_MW - min_demand_MW)
    )

    b = TARGET_MIN_KW - m * min_demand_MW

    test_demand_array_kW = m * test_demand_array_MW + b

    test_energy_tariffs_dict = {
        -1 * np.inf: -0.07,
        0: 0.07,
        5: 0.11
    }

    solar_capacity_kW = 1
    test_solar = Renewable(
        solar_capacity_kW * np.random.rand(8760),
        "{} kW Solar".format(solar_capacity_kW)
    )

    wind_capacity_kW = 1
    test_wind = Renewable(
        wind_capacity_kW * np.random.rand(8760),
        "{} kW Wind".format(wind_capacity_kW)
    )

    test_renewable_list = [test_solar, test_wind]

    test_energy_capacity_kWh = 10
    test_power_capacity_kW = 100
    test_initial_SOC = 0
    test_charging_efficiency = 0.9
    test_discharging_efficiency = 0.9

    test_storage = Storage(
        test_energy_capacity_kWh,
        power_capacity=test_power_capacity_kW,
        initial_state_of_charge=test_initial_SOC,
        charging_efficiency=test_charging_efficiency,
        discharging_efficiency=test_discharging_efficiency
    )

    test_residential_planner = ResidentialPlanner(
        test_time_array_hours,
        test_demand_array_kW,
        test_energy_tariffs_dict,
        renewable_list=test_renewable_list,
        storage=test_storage,
        power_units_str="kW",
        currency_units_str="CAD"
    )

    test_residential_planner.run()
    print(test_residential_planner)

    test_residential_planner.plot()

    print()
    print("TESTING:\tResidentialPlanner\tPASS")
    print()
