from dataclasses import dataclass

from typing import Any, Set, Dict, Tuple, List

@dataclass
class FiniteStateMachine:
    """A deterministic finite-state machine (DFSM) as a 5-tuple:
    M = (Σ, S, s₀, δ, F)
    - Σ: Input alphabet (set of symbols)
    - S: Set of states
    - s₀: Initial state
    - δ: Transition function (dict mapping (state, symbol) to next state)
    - F: Set of final states (must be subset of S)
    """
    # S: Set of states (must be hashable, e.g. strings)
    states: Set[Any]
    # Σ: Input alphabet (must be hashable, e.g., strings)
    alphabet: Set[Any]
    # s₀: Initial state (must be in S)
    initial_state: Any
    # δ: Transition function, S x Σ → S
    transition_function: Dict[Tuple[Any, Any], Any]
    #F: Set of final states (must be subset of S)
    final_states: Set[Any]

    def __post_init__(self):
        # Validate non-emptiness of states and alphabet.
        if not self.states:
            raise ValueError("States must be non-empty.")
        if not self.alphabet:
            raise ValueError("Alphabet must be non-empty.")

        # Validate initial state.
        if self.initial_state not in self.states:
            raise ValueError("Initial state must be in states.")
    
        # Validate final states.
        if not self.final_states.issubset(self.states):
            raise ValueError("Final states must be a subset of states.")

    def is_valid_transition_function(self):
        """Check that all keys in transitions are valid (state, symbol) pairs.
        - Check that all values are valid states.        
        """
        for (state, symbol), next_state in self.transition_function.items():
            if state not in self.states:
                return False
            if symbol not in self.alphabet:
                return False
            if next_state not in self.states:
                return False
        return True

    def is_transition_function_total(self):
        """Check if for every state in S and every symbol in Σ, there must be a
        transition defined.
        """
        for state in self.states:
            for symbol in self.alphabet:
                if (state, symbol) not in self.transition_function:
                    return False, state, symbol
        return True, None, None

    def accepts(self, input_string: str) -> bool:
        """Check if the given input string is accepted by the FSM.
        - Start from the initial state.
        - For each symbol in the input, follow the transition function.
        - If the final state is in F, return True; otherwise, False.
        """
        current_state = self.initial_state
        for symbol in input_string:
            if symbol not in self.alphabet:
                raise ValueError(f"Symbol {symbol} is not in the alphabet.")
            if (current_state, symbol) not in self.transition_function:
                # This should never happen due to totality, but included for
                # safety.
                return False
            current_state = self.transition_function[(current_state, symbol)]
        return current_state in self.final_states

    def __repr__(self):
        return f"FiniteStateMachine(states={self.states}, alphabet={self.alphabet}, initial_state={self.initial_state}, transition_function={self.transition_function}, final_states={self.final_states})"

@dataclass
class FiniteStateMachineRunner:
    """Tracks runtime state transitions and state for a specific finite state
    machine execution.
    """
    fsm: FiniteStateMachine
    current_state: Any = None
    transition_history: List[Tuple[Any, Any, Any]] = None

    def reset(self):
        """Reset to initial state (s_0)."""
        self.current_state = self.fsm.initial_state
        self.transition_history = []

    def step(self, symbol: Any):
        """Transition to the next state based on the current state and input
        symbol.
        """
        if symbol not in self.fsm.alphabet:
            raise ValueError(f"Symbol {symbol} is not in the alphabet.")

        previous_state = self.current_state
        self.current_state = self.fsm.transition_function[
            (self.current_state, symbol)]
        self.transition_history.append((previous_state, symbol, self.current_state))
        return self.current_state

    def process_sequence(self, symbols: list[Any]) -> bool:
        """Process a sequence of symbols and return whether the FSM accepts the
        sequence.
        """
        for symbol in symbols:
            self.step(symbol)
        return self.current_state in self.fsm.final_states

    @property
    def accepted(self) -> bool:
        """Is current state in final states?"""
        return self.current_state in self.fsm.final_states

def create_fsm_functions(fsm_runner):
    """
    Function factory that creates functions bound to a specific
    FiniteStateMachineRunner instance.
    
    Args:
        fsm_runner: An instance of FiniteStateMachineRunner
        
    Returns:
        tuple: (get_current_state, run_step) functions that operate on the
        provided fsm_runner
    """
    def get_current_state():
        """
        Get the current state of the FSM.
        
        Returns:
            Any: The current state of the FSM
        """
        return fsm_runner.current_state
    
    def run_step(symbol):
        """
        Run a step in the FSM with the given symbol.
        
        Args:
            symbol: Any - The input symbol to process
            
        Returns:
            Any: The new current state after processing the symbol
        """
        fsm_runner.step(symbol)
        return fsm_runner.current_state
    
    return get_current_state, run_step

def create_fsm_tools_with_strings(input_get_current_state, input_run_step):
    def get_current_state():
        """Get the current state of the state machine.

        Returns:
            str: The current state of the state machine.
        """
        return str(input_get_current_state())
    
    def run_step(symbol: str):
        """Run a step in the state machine with the given symbol.

        Args:
            symbol: The input symbol to process

        Returns:
            str: The new current state after processing the symbol
        """
        return str(input_run_step(symbol))

    return get_current_state, run_step

def get_user_input():
    """Get user input. The reason behind this function is to handle the
    following possible examples: the LLM model needs user input to continue to
    run some tool or agent or to complete a task. The LLM model needs
    clarification from the user.

    Returns:
        str: The user input
    """
    return input("Enter input: ")
