"""
    Anthony Truelove MASc, P.Eng.  
    Python Certified Professional Programmer (PCPP1)  
    2025

    A stocks and flows modelling class, as part of the `mech447` package.
"""


import matplotlib.pyplot as plt
import numpy as np


class StocksAndFlows:
    """
    A class which contains modelled stock and flow data for a given system over
    a given modelling horizon.
    """

    def __init__(
        self,
        time_array_years: np.array,
        unconstrained_addition_array_units_per_year: np.array,
        unconstrained_source_array_units_per_year: np.array,
        unconstrained_production_array_units_per_year: np.array,
        total_resource_units: float = 1000,
        reserve_initial_units: float = 0,
        maximum_reserve_units: float = -1,
        units_str : str = "units"
    ) -> None:
        """
        StockAndFlow class constructor.

        Parameters
        ----------
        time_array_years: np.array
            This is an array of points in time [years]. This defines both
            the modelling horizon and the modelling resolution.

        unconstrained_addition_array_units_per_year: np.array
            This is an array of unconstrained (or target) addition rates
            [units/yr] at every point in time.

        unconstrained_source_array_units_per_year: np.array
            This is an array of unconstrained (or target) source rates
            [units/yr] at every point in time.

        unconstrained_production_array_units_per_year: np.array
            This is an array of unconstrained (or target) production rates
            [units/yr] at every point in time.

        total_resource_units: float, optional, default 1000
            This is the total available resource [units] that can be added to
            the reserve.

        reserve_initial_units: float, optional, default 0
            This is the initial state of the reserve [units].

        maximum_reserve_units: float, optional, default -1
            This is the maximum capacity of the reserve [units]. -1 is a
            sentinel value, and if passed in it indicates that the reserve
            has unlimited capacity.

        units_str : str, optional, default "units"
            This is a string defining what the units are, for plotting.

        Returns
        -------
        None
        """

        #   1. cast inputs to arrays
        self.time_array_years = np.array(time_array_years)
        """
        An array of points in time [years].
        """

        unconstrained_addition_array_units_per_year = np.array(
            unconstrained_addition_array_units_per_year
        )

        unconstrained_source_array_units_per_year = np.array(
            unconstrained_source_array_units_per_year
        )

        unconstrained_production_array_units_per_year = np.array(
            unconstrained_production_array_units_per_year
        )

        #   2. construct time delta array
        self.time_delta_array_years = np.diff(time_array_years)
        """
        An array of time delta (dt) values [years].
        """
        self.time_delta_array_years = np.append(
            self.time_delta_array_years,
            self.time_delta_array_years[-1]
        )

        #   3. check inputs, set dt array
        self.__checkInputs(
            unconstrained_addition_array_units_per_year,
            unconstrained_source_array_units_per_year,
            unconstrained_production_array_units_per_year,
            total_resource_units,
            reserve_initial_units,
            maximum_reserve_units
        )

        #   4. init attributes
        self.total_resource_units = total_resource_units
        """
        The total available resource that can be added to the reserve [units].
        """

        self.reserve_initial_units = reserve_initial_units
        """
        The initial state of the reserve [units].
        """

        self.maximum_reserve_units = maximum_reserve_units
        """
        The maximum capacity of the reserve [units].
        """

        self.units_str = units_str
        """
        This is a string defining what the units are, for plotting.
        """

        if self.maximum_reserve_units < 0:
            self.maximum_reserve_units = np.inf

        self.N = len(time_array_years)
        """
        This is the number of points (i.e. number of states) in the model.
        """

        self.unconstrained_addition_array_units_per_year = (
            unconstrained_addition_array_units_per_year
        )
        """
        An array of unconstrained (or target) addition rates [units/yr] at
        every point in time.
        """

        self.unconstrained_source_array_units_per_year = (
            unconstrained_source_array_units_per_year
        )
        """
        An array of unconstrained (or target) source rates [units/yr] at every
        point in time.
        """

        self.unconstrained_production_array_units_per_year = (
            unconstrained_production_array_units_per_year
        )
        """
        An array of unconstrained (or target) production rates [units/yr] at
        every point in time.
        """

        self.constrained_addition_array_units_per_year = np.zeros(self.N)
        """
        An array of constrained addition rates [units/yr] at every point in
        time.
        """

        self.constrained_source_array_units_per_year = np.zeros(self.N)
        """
        An array of constrained source rates [units/yr] at every point in
        time.
        """

        self.constrained_production_array_units_per_year = np.zeros(self.N)
        """
        An array of constrained production rates [units/yr] at every point in
        time.
        """

        self.cumulative_addition_array_units = np.zeros(self.N)
        """
        An array of the cumulative addition [units] up to every point in time.
        """

        self.cumulative_source_array_units = np.zeros(self.N)
        """
        An array of the cumulative source [units] up to every point in time.
        """

        self.cumulative_production_array_units = np.zeros(self.N)
        """
        An array of the cumulative production [units] up to every point in time.
        """

        self.reserve_array_units = np.zeros(self.N)
        """
        An array of the cumulative reserve [units] up to every point in time.
        """

        self.reserve_array_units[0] = reserve_initial_units

        return


    def __checkInputs(
        self,
        unconstrained_addition_array_units_per_year: np.array,
        unconstrained_source_array_units_per_year: np.array,
        unconstrained_production_array_units_per_year: np.array,
        total_resource_units: float,
        reserve_initial_units: float,
        maximum_reserve_units: float
    ) -> None:
        """
        Helper method to check __init__ inputs.

        Parameters
        ----------
        unconstrained_addition_array_units_per_year: np.array
            This is an array of unconstrained (or target) addition rates
            [units/yr] at every point in time.

        unconstrained_source_array_units_per_year: np.array
            This is an array of unconstrained (or target) source rates
            [units/yr] at every point in time.

        unconstrained_production_array_units_per_year: np.array
            This is an array of unconstrained (or target) production rates
            [units/yr] at every point in time.

        total_resource_units: float, optional, default 1000
            This is the total available resource [units] that can be added to
            the reserve.

        reserve_initial_units: float
            This is the initial state of the reserve [units].

        maximum_reserve_units: float
            This is the maximum capacity of the reserve [units]. -1 is a
            sentinel value, and if passed in it indicates that the reserve
            has unlimited capacity.

        Returns
        -------
        None
        """

        #   1. time delta array must be strictly positive
        boolean_mask = self.time_delta_array_years <= 0

        if boolean_mask.any():
            error_string = "ERROR: StocksAndFlows.__checkInputs():\t"
            error_string += "time array must be strictly increasing"
            error_string += " (t[i + 1] > t[i] for all i)."

            raise RuntimeError(error_string)

        #   2. input arrays must all be same size
        expected_array_size = len(self.time_array_years)

        if (
            len(unconstrained_addition_array_units_per_year)
            != expected_array_size
        ):
            error_string = "ERROR: StocksAndFlows.__checkInputs():\t"
            error_string += "unconstrained addition array must be same size as"
            error_string += " time array."

            raise RuntimeError(error_string)

        elif (
            len(unconstrained_source_array_units_per_year)
            != expected_array_size
        ):
            error_string = "ERROR: StocksAndFlows.__checkInputs():\t"
            error_string += "unconstrained source array must be same size as"
            error_string += " time array."

            raise RuntimeError(error_string)

        elif (
            len(unconstrained_production_array_units_per_year)
            != expected_array_size
        ):
            error_string = "ERROR: StocksAndFlows.__checkInputs():\t"
            error_string += "unconstrained production array must be same size"
            error_string += " as time array."

            raise RuntimeError(error_string)

        #   3. input arrays must all be non-negative
        boolean_mask = unconstrained_addition_array_units_per_year < 0

        if boolean_mask.any():
            error_string = "ERROR: StocksAndFlows.__checkInputs():\t"
            error_string += "unconstrained addition array must be non-negative"
            error_string += " (x[i] >= 0 for all i)."

            raise RuntimeError(error_string)

        boolean_mask = unconstrained_source_array_units_per_year < 0

        if boolean_mask.any():
            error_string = "ERROR: StocksAndFlows.__checkInputs():\t"
            error_string += "unconstrained source array must be non-negative"
            error_string += " (x[i] >= 0 for all i)."

            raise RuntimeError(error_string)

        boolean_mask = unconstrained_production_array_units_per_year < 0

        if boolean_mask.any():
            error_string = "ERROR: StocksAndFlows.__checkInputs():\t"
            error_string += "unconstrained production array must be"
            error_string += " non-negative (x[i] >= 0 for all i)."

            raise RuntimeError(error_string)

        #   4. reserve initial condition must be non-negative
        if reserve_initial_units < 0:
            error_string = "ERROR: StocksAndFlows.__checkInputs():\t"
            error_string += "reserve initial condition must be non-negative"
            error_string += " (R(0) >= 0)."

            raise RuntimeError(error_string)

        #   5. reserve initial condition must not exceed reserve maximum
        if maximum_reserve_units >= 0:
            if reserve_initial_units > maximum_reserve_units:
                error_string = "ERROR: StocksAndFlows.__checkInputs():\t"
                error_string += "reserve initial condition cannot exceed"
                error_string += " reserve maximum (R(0) <= R_max)."

                raise RuntimeError(error_string)

        #   6. total resource should be positive
        if total_resource_units < 0:
            error_string = "ERROR: StocksAndFlows.__checkInputs():\t"
            error_string += "total available resource must be non-negative"
            error_string += " (T >= 0)."

            raise RuntimeError(error_string)

        return

    def getDerivative(
        self,
        i : int
    ) -> None:
        """
        Method to get current system derivative (constrained). Logic is as
        follows:

        First, constrain addition $\\dot{F}(t)$ such that
        $$ \\dot{F}(t) \\leq \\frac{T - F(t)}{\\Delta t} $$
        where $\\Delta t$ is the time step, $T$ is the total resource available,
        and $F(t)$ is cumulative addition. This ensures that cumulative
        addition never exceeds the resource availability.

        Second, constrain addition $\\dot{F}(t)$ and source $\\dot{\\sigma}(t)$
        such that
        $$ R(t) + \\dot{F}(t)\\Delta t + \\dot{\\sigma}(t)\\Delta t - \\dot{P}(t)\\Delta t \\leq R_\\textrm{max} $$
        where $R(t)$ is reserve state, $\\dot{P}(t)$ is production, and $R_\\textrm{max}$
        is the reserve maximum. This constraint is enforced by first reducing
        $\\dot{F}(t)$ to zero, and then reducing $\\dot{\\sigma}(t)$ to zero
        (so throttle addition first, then curtail source if necessary). This
        ensures that the reserve state never exceeds the reserve max (i.e.
        reserve cannot overflow).

        Third, constrain production $\\dot{P}(t)$ such that
        $$ \\dot{P}(t) \\leq \\frac{R(t)}{\\Delta t} + \\dot{F}(t) + \\dot{\\sigma}(t) $$
        This ensures that the reserve state is everywhere non-negative (i.e.
        reserve cannot be overdrawn).

        Parameters
        ----------
        i : int
            The current time step index.

        Returns
        -------
        None
        """

        #   1. get current values
        time_delta_years = self.time_delta_array_years[i]
        addition = self.unconstrained_addition_array_units_per_year[i]
        cumulative_addition = self.cumulative_addition_array_units[i]
        source = self.unconstrained_source_array_units_per_year[i]
        production = self.unconstrained_production_array_units_per_year[i]
        reserve = self.reserve_array_units[i]

        #   2. enforce constraint 1
        right_hand_side = (
            (self.total_resource_units - cumulative_addition)
            / time_delta_years
        )

        if addition > right_hand_side:
            addition = max([0, right_hand_side])

        #   3. enforce constraint 2
        #      (reduce addition first, then source if necessary)
        left_hand_side = (
            reserve / time_delta_years
            + addition
            + source
            - production
        )

        right_hand_side = self.maximum_reserve_units / time_delta_years

        constraint_violation = max([0, left_hand_side - right_hand_side])

        reduction = min([addition, constraint_violation])
        addition -= reduction
        constraint_violation -= reduction

        reduction = min([source, constraint_violation])
        source -= reduction
        constraint_violation -= reduction

        #   4. enforce constraint 3
        right_hand_side = (
            reserve / time_delta_years
            + addition
            + source
        )

        if production > right_hand_side:
            production = max([0, right_hand_side])

        #   5. log results
        self.constrained_addition_array_units_per_year[i] = addition
        self.constrained_source_array_units_per_year[i] = source
        self.constrained_production_array_units_per_year[i] = production

        return

    def run(self) -> None:
        """
        Method to run model and populate state arrays.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        for i in range(0, self.N):
            #   1. get derivative
            self.getDerivative(i)

            if i < self.N - 1:
                #   2. integrate addition forward
                time_delta_years = self.time_delta_array_years[i]

                self.cumulative_addition_array_units[i + 1] = (
                    self.cumulative_addition_array_units[i]
                    + (
                        self.constrained_addition_array_units_per_year[i]
                        * time_delta_years
                    )
                )

                #   3. integrate source forward
                self.cumulative_source_array_units[i + 1] = (
                    self.cumulative_source_array_units[i]
                    + (
                        self.constrained_source_array_units_per_year[i]
                        * time_delta_years
                    )
                )

                #   4. integrate production forward
                self.cumulative_production_array_units[i + 1] = (
                    self.cumulative_production_array_units[i]
                    + (
                        self.constrained_production_array_units_per_year[i]
                        * time_delta_years
                    )
                )

                #   5. integrate reserve forward
                self.reserve_array_units[i + 1] = (
                    self.reserve_initial_units
                    + self.cumulative_addition_array_units[i + 1]
                    + self.cumulative_source_array_units[i + 1]
                    - self.cumulative_production_array_units[i + 1]
                )

            pass

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

        #   1. make flows plot
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)

        plt.plot(
            self.time_array_years,
            self.unconstrained_addition_array_units_per_year,
            alpha=0.5,
            linestyle="--",
            color="C0",
            label=r"unconstrained addition $\dot{F}(t)$",
            zorder=2
        )
        plt.plot(
            self.time_array_years,
            self.constrained_addition_array_units_per_year,
            color="C0",
            label=r"constrained addition $\dot{F}(t)$",
            zorder=2
        )

        plt.plot(
            self.time_array_years,
            self.unconstrained_source_array_units_per_year,
            alpha=0.5,
            linestyle="--",
            color="C2",
            label=r"unconstrained source $\dot{\sigma}(t)$",
            zorder=2
        )
        plt.plot(
            self.time_array_years,
            self.constrained_source_array_units_per_year,
            color="C2",
            label=r"constrained source $\dot{\sigma}(t)$",
            zorder=2
        )

        plt.plot(
            self.time_array_years,
            self.unconstrained_production_array_units_per_year,
            alpha=0.5,
            linestyle="--",
            color="C3",
            label=r"unconstrained production $\dot{P}(t)$",
            zorder=2
        )
        plt.plot(
            self.time_array_years,
            self.constrained_production_array_units_per_year,
            color="C3",
            label=r"constrained production $\dot{P}(t)$",
            zorder=2
        )

        plt.xlim(self.time_array_years[0], self.time_array_years[-1])
        plt.xlabel(r"Time $t$ [years]")
        plt.ylabel("Flows [" + self.units_str + "/yr]")
        plt.legend()
        plt.tight_layout()

        #   2. make stocks plot
        plt.figure(figsize=(8, 6))
        plt.grid(color="C7", alpha=0.5, which="both", zorder=1)

        plt.axhline(
            y=self.total_resource_units,
            alpha=0.5,
            linestyle="--",
            color="C0",
            label="total available resource $T$",
            zorder=2
        )

        if self.maximum_reserve_units < np.inf:
            plt.axhline(
                y=self.maximum_reserve_units,
                alpha=0.5,
                linestyle="--",
                color="black",
                label="maximum reserve capacity $R_{max}$",
                zorder=2
            )

        plt.plot(
            self.time_array_years,
            self.cumulative_addition_array_units,
            color="C0",
            label=r"cumulative addition $F(t)$",
            zorder=3
        )

        plt.plot(
            self.time_array_years,
            self.cumulative_source_array_units,
            color="C2",
            label=r"cumulative source $\sigma(t)$",
            zorder=3
        )

        plt.plot(
            self.time_array_years,
            self.reserve_array_units,
            color="black",
            label=r"reserve $R(t)$",
            zorder=3
        )

        plt.plot(
            self.time_array_years,
            self.cumulative_production_array_units,
            color="C3",
            label=r"cumulative production $P(t)$",
            zorder=3
        )

        plt.xlim(self.time_array_years[0], self.time_array_years[-1])
        plt.xlabel(r"Time $t$ [years]")
        plt.ylabel("Stocks [" + self.units_str + "]")
        plt.legend()
        plt.tight_layout()

        #   3. show
        plt.show()

        return


if __name__ == "__main__":
    print("TESTING:\tStocksAndFlows")
    print()

    good_time_array = [0, 1, 2, 3]
    good_addition_array = [1, 1, 1, 1]
    good_source_array = [1, 1, 1, 1]
    good_production_array = [1, 1, 1, 1]

    #   1. passing a bad time array (not strictly increasing)
    try:
        bad_time_array = [0, 1, 0, 1]

        StocksAndFlows(
            bad_time_array,
            good_addition_array,
            good_source_array,
            good_production_array
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   2. passing a bad addition array (wrong size)
    try:
        bad_addition_array = [-1, -1, -1]

        StocksAndFlows(
            good_time_array,
            bad_addition_array,
            good_source_array,
            good_production_array
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   3. passing a bad source array (wrong size)
    try:
        bad_source_array = [-1, -1, -1]

        StocksAndFlows(
            good_time_array,
            good_addition_array,
            bad_source_array,
            good_production_array
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   4. passing a bad production array (wrong size)
    try:
        bad_production_array = [-1, -1, -1]

        StocksAndFlows(
            good_time_array,
            good_addition_array,
            good_source_array,
            bad_production_array
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   5. passing a bad addition array (negatives)
    try:
        bad_addition_array = [-1, -1, -1, -1]

        StocksAndFlows(
            good_time_array,
            bad_addition_array,
            good_source_array,
            good_production_array
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   6. passing a bad source array (negatives)
    try:
        bad_source_array = [-1, -1, -1, -1]

        StocksAndFlows(
            good_time_array,
            good_addition_array,
            bad_source_array,
            good_production_array
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   7. passing a bad production array (negatives)
    try:
        bad_production_array = [-1, -1, -1, -1]

        StocksAndFlows(
            good_time_array,
            good_addition_array,
            good_source_array,
            bad_production_array
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   8. passing a bad total resource (zero)
    try:
        bad_total_resource = -1

        StocksAndFlows(
            good_time_array,
            good_addition_array,
            good_source_array,
            good_production_array,
            total_resource_units=bad_total_resource
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   9. passing a bad reserve initial condition (negative)
    try:
        bad_initial_reserve = -1

        StocksAndFlows(
            good_time_array,
            good_addition_array,
            good_source_array,
            good_production_array,
            reserve_initial_units=bad_initial_reserve
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   10. passing a bad reserve initial condition (greater than reserve max)
    try:
        reserve_max = 10
        bad_initial_reserve = 2 * reserve_max

        StocksAndFlows(
            good_time_array,
            good_addition_array,
            good_source_array,
            good_production_array,
            reserve_initial_units=bad_initial_reserve,
            maximum_reserve_units=reserve_max
        )

    except RuntimeError as e:
        print(e)

    except Exception as e:
        raise(e)

    else:
        raise RuntimeError("Expected a RuntimeError here.")

    #   11. good construction
    StocksAndFlows(
        good_time_array,
        good_addition_array,
        good_source_array,
        good_production_array
    )

    #   12. solve MECH 447 assignment 1, question 3a
    time_array_years = np.linspace(0, 100, 1000 * 100)

    alpha_per_year = 0.03
    mu_per_year = -0.02

    F_0_Mt_per_year = 100
    P_0_Mt_per_year = 5

    unconstrained_addition_array_Mt_per_year = (
        F_0_Mt_per_year * np.exp(mu_per_year * time_array_years)
    )

    unconstrained_source_array_Mt_per_year = (
        0.02 * F_0_Mt_per_year * np.ones(len(time_array_years))
    )

    unconstrained_production_array_Mt_per_year = (
        P_0_Mt_per_year * np.exp(alpha_per_year * time_array_years)
    )

    total_resource_Mt = 1000
    reserve_initial_Mt = 100
    maximum_reserve_Mt = 500

    stocks_and_flows_q3a = StocksAndFlows(
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
    stocks_and_flows_q3a.plot()

    #   13. solve MECH 447 assignment 1, question 3b
    F_0_Mt_per_year = 20

    unconstrained_addition_array_Mt_per_year = (
        F_0_Mt_per_year * np.exp(mu_per_year * time_array_years)
    )

    unconstrained_source_array_Mt_per_year = (
        0.02 * F_0_Mt_per_year * np.ones(len(time_array_years))
    )

    reserve_initial_Mt = 25
    maximum_reserve_Mt = 100

    stocks_and_flows_q3b = StocksAndFlows(
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
    stocks_and_flows_q3b.plot()

    print()
    print("TESTING:\tStocksAndFlows\tPASS")
    print()
