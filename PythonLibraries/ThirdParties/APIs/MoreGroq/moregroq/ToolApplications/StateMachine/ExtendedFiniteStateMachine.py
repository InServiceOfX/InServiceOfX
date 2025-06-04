from dataclasses import dataclass

from typing import Any, Set, Dict, Tuple, List, Callable

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
    transition_relation: Dict[Tuple[Any, Any, Any], Tuple[Any, Any, Any]] | \
        Callable[[Any, Any, Any], Tuple[Any, Any, Any]]

    def __post_init__(self):
        if not self.states:
            raise ValueError("States must be non-empty.")
    
    def get_variable(self, variable_name: Any) -> Any:
        return self.variables.get(variable_name, None)

    def set_variable(self, variable_name: Any, value: Any):
        self.variables[variable_name] = value

    def get_transition_relation(self, state: Any, input_symbol: Any) -> Tuple[Any, Any, Any]:
        return self.transition_relation.get((state, input_symbol), None)

@dataclass
class ExtendedFiniteStateMachineRunner:
    efsm: ExtendedFiniteStateMachine
    current_state: Any = None

    # Variables in D, where a variable belongs to a linear space D_i
    variables: Dict[Any, Any]

    transition_history: List[Tuple[Any, Any, Any, Any, Any, Any]] = None

    def reset(self, initial_state: Any, initial_variables: Dict[Any, Any]):
        self.current_state = initial_state
        self.current_output_symbol = None
        self.current_update_function = None
        self.variables = initial_variables
        self.transition_history = []

    def transition(self, input_symbol: Any, enable_function: Any, update_function: Any):
        previous_state = self.current_state
        self.current_state, self.current_output_symbol, self.current_update_function = \
            self.efsm.transition_relation[
                (previous_state, input_symbol, enable_function)]

        self.transition_history.append(
            (
                previous_state,
                input_symbol,
                enable_function,
                self.current_state,
                self.current_output_symbol,
                self.current_update_function))

    def update(self)

        self.variables = self.current_update_function(self.variables)
        return self.current_output_symbol
        