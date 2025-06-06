from dataclasses import dataclass

from typing import Any, Set, Dict, Tuple, List, Callable, Optional

@dataclass
class ExtendedFiniteStateMachine:
    """A finite state machine where transition can be expressed by an 'if
    statement' consisting of a set of trigger conditions.

    M=(I,O,S,D,F,U,T)
    - I: Set of input symbols
    - O: Set of output symbols
    - S: Set of symbolic states
    - D: n-dim linear space D_1 x D_2 x ... x D_n
    - F: Set of enabling functions f_i: D -> {0,1}
    - U: Set of update functions u_i: D -> D
    - T: transition relation T:SxIxF -> SxOxU
    """
    input_alphabet: Set[Any]
    output_alphabet: Set[Any]
    states: Set[Any]
    enabling_functions: Dict[Any, Any]
    update_functions: Dict[Any, Any]
    transition_relation: Callable

    def __post_init__(self):
        if not self.states:
            raise ValueError("States must be non-empty.")

@dataclass
class ExtendedFiniteStateMachineRunner:
    efsm: ExtendedFiniteStateMachine
    current_state: Optional[Any] = None

    # (
    #   variables,
    #   previous_state,
    #   input_symbol,
    #   enable_function,
    #   new_state,
    #   output_symbol,
    #   update_function)
    transition_history: \
        List[Tuple[Any, Any, Any, Any, Any, Any, Any]] = None

    current_transition: Optional[Callable] = None

    current_input_symbol: Optional[Any] = None
    current_enable_function: Optional[Callable] = None
    current_output_symbol: Optional[Any] = None
    current_update_function: Optional[Callable] = None

    def reset(self, initial_state: Any):
        self.current_state = initial_state
        self.transition_history = []

        self.current_transition_function = None
        self.current_input_symbol = None
        self.current_enable_function = None
        self.current_output_symbol = None
        self.current_update_function = None

    def get_transition_function(self, input_symbol: Any, enable_function: Any):
        self.current_transition_function = self.efsm.transition_relation(
            self.current_state,
            input_symbol,
            enable_function)

        self.current_input_symbol = input_symbol
        self.current_enable_function = enable_function

        return self.current_transition_function

    def transition(self, variables):

        previous_state = self.current_state

        self.current_state, self.current_output_symbol, self.current_update_function = \
            self.current_transition_function(variables)

        self.transition_history.append(
            (
                variables,
                previous_state,
                self.current_input_symbol,
                self.current_enable_function,
                self.current_state,
                self.current_output_symbol,
                self.current_update_function))

        return self.current_state, self.current_output_symbol

    def update(self, variables):
        return self.current_update_function(variables)