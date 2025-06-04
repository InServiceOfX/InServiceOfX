from moregroq.ToolApplications.StateMachine.ExtendedFiniteStateMachine import (
    ExtendedFiniteStateMachine,
    ExtendedFiniteStateMachineRunner
)

from typing import Callable, Tuple

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

    def transition_function(
            state: str,
            input_symbol: str,
            enable_function: Callable) -> Tuple[str, Any, Callable]:
        if state == "normal" and input_symbol == "update" and enable_function == "guard_high_temperature":


            return ("high", None, "identity")
        else:
            raise ValueError(f"Invalid transition: {state} {input_symbol} {enable_function}")

def test_extended_finite_state_machine():
    efsm = ExtendedFiniteStateMachine()
    runner = ExtendedFiniteStateMachineRunner(efsm)
    runner.run()
