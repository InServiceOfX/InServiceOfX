from moregroq.ToolApplications.StateMachine.ExtendedFiniteStateMachine import (
    ExtendedFiniteStateMachine,
    ExtendedFiniteStateMachineRunner
)

from typing import Callable, Tuple, Any

def temperature_sensor_components_1():
    # S, set of symbolic states
    states = {"normal", "high"}
    # new temperature reading received.
    input_alphabet = {"update"}
    # O, set of output symbols
    output_alphabet = {None,}

    def update_function_0(temperature: float) -> float:
        return temperature

    update_functions = {"identity": update_function_0,}

    def enabling_function_0(temperature: float) -> bool:
        return temperature >= 50

    enabling_functions = {"guard_high_temperature": enabling_function_0,}

    def transition(
            state: str,
            input_symbol: str,
            enable_function: Callable,
            ) -> Callable[float, Tuple[str, Any, Callable]]:
        if input_symbol == "update":

            def transition_function(temperature: float) -> \
                Tuple[str, Any, Callable]:
                if enable_function(temperature):
                    return ("high", None, update_function_0)
                else:
                    return ("normal", None, update_function_0)

            return transition_function
        else:
            raise ValueError(
                f"Invalid transition: {state} {input_symbol} {enable_function}")

    return {
        "states": states,
        "input_alphabet": input_alphabet,
        "output_alphabet": output_alphabet,
        "update_functions": update_functions,
        "enabling_functions": enabling_functions,
        "transition": transition}

def test_extended_finite_state_machine():

    temperature_sensor_components = temperature_sensor_components_1()

    efsm = ExtendedFiniteStateMachine(
        temperature_sensor_components["input_alphabet"],
        temperature_sensor_components["output_alphabet"],
        temperature_sensor_components["states"],
        temperature_sensor_components["enabling_functions"],
        temperature_sensor_components["update_functions"],
        temperature_sensor_components["transition"]
    )
    runner = ExtendedFiniteStateMachineRunner(efsm)

    assert runner.current_state is None
    runner.reset("normal")
    assert runner.current_state == "normal"

    runner.get_transition_function(
        "update",
        temperature_sensor_components["enabling_functions"]["guard_high_temperature"])

    runner.transition(30)
    assert runner.current_state == "normal"

    assert runner.update(30) == 30

    runner.transition(51)
    assert runner.current_state == "high"

    assert runner.update(51) == 51

    runner.transition(50)
    assert runner.current_state == "high"

    assert runner.update(50) == 50

    runner.transition(49)
    assert runner.current_state == "normal"

    assert runner.update(49) == 49

    runner.transition(50)
    assert runner.current_state == "high"

    assert runner.update(50) == 50

    history = runner.transition_history
    assert len(history) == 5
    assert history[0][0] == 30
    assert history[0][1] == "normal"
    assert history[0][2] == "update"
    assert history[0][3] == \
        temperature_sensor_components["enabling_functions"]["guard_high_temperature"]

    assert history[1][0] == 51
    assert history[1][1] == "normal"
    assert history[1][2] == "update"
    assert history[1][3] == \
        temperature_sensor_components["enabling_functions"]["guard_high_temperature"]
    assert history[1][4] == "high"
    assert history[1][5] == None
    assert history[1][6] == \
        temperature_sensor_components["update_functions"]["identity"]

    assert history[2][0] == 50
    assert history[2][1] == "high"
    assert history[2][2] == "update"
    assert history[2][3] == \
        temperature_sensor_components["enabling_functions"]["guard_high_temperature"]
    assert history[2][4] == "high"
    assert history[2][5] == None
    assert history[2][6] == \
        temperature_sensor_components["update_functions"]["identity"]

    assert history[3][0] == 49
    assert history[3][1] == "high"
    assert history[3][2] == "update"
    assert history[3][3] == \
        temperature_sensor_components["enabling_functions"]["guard_high_temperature"]
    assert history[3][4] == "normal"
    assert history[3][5] == None
    assert history[3][6] == \
        temperature_sensor_components["update_functions"]["identity"]

    assert history[4][0] == 50
    assert history[4][1] == "normal"
    assert history[4][2] == "update"
    assert history[4][3] == \
        temperature_sensor_components["enabling_functions"]["guard_high_temperature"]
