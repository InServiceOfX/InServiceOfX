import pytest
from typing import List, Any, Set, Dict, Tuple
from moregroq.ToolApplications.StateMachine.FiniteStateMachine import (
    FiniteStateMachine,
    FiniteStateMachineRunner
)


@pytest.fixture
def fsm_components():
    """
    Create components for a simple FSM that accepts strings ending with 'ab'
    """
    states = {"q0", "q1", "q2"}
    alphabet = {"a", "b"}
    initial_state = "q0"
    transition_function = {
        ("q0", "a"): "q1",
        ("q0", "b"): "q0",
        ("q1", "a"): "q1",
        ("q1", "b"): "q2",
        ("q2", "a"): "q1",
        ("q2", "b"): "q0"
    }
    final_states = {"q2"}
    
    return {
        "states": states,
        "alphabet": alphabet,
        "initial_state": initial_state,
        "transition_function": transition_function,
        "final_states": final_states
    }


@pytest.fixture
def fsm(fsm_components):
    """Create a finite state machine that accepts strings ending with 'ab'"""
    return FiniteStateMachine(
        states=fsm_components["states"],
        alphabet=fsm_components["alphabet"],
        initial_state=fsm_components["initial_state"],
        transition_function=fsm_components["transition_function"],
        final_states=fsm_components["final_states"]
    )


@pytest.fixture
def runner(fsm):
    """Create a runner for the FSM"""
    runner = FiniteStateMachineRunner(fsm=fsm)
    runner.reset()  # Initialize to initial state
    return runner


def test_fsm_initialization(fsm, fsm_components):
    """Test that the FSM is initialized correctly"""
    assert fsm.states == fsm_components["states"]
    assert fsm.alphabet == fsm_components["alphabet"]
    assert fsm.initial_state == fsm_components["initial_state"]
    assert fsm.transition_function == fsm_components["transition_function"]
    assert fsm.final_states == fsm_components["final_states"]


def test_transition_function_validation(fsm, fsm_components):
    """Test that the transition function validation works"""
    assert fsm.is_valid_transition_function()
    
    # Test with an invalid transition function
    invalid_fsm = FiniteStateMachine(
        states=fsm_components["states"],
        alphabet=fsm_components["alphabet"],
        initial_state=fsm_components["initial_state"],
        transition_function={("q0", "a"): "q3"},  # q3 is not in states
        final_states=fsm_components["final_states"]
    )
    assert not invalid_fsm.is_valid_transition_function()


def test_transition_function_totality(fsm, fsm_components):
    """Test that the transition function totality check works"""
    is_total, _, _ = fsm.is_transition_function_total()
    assert is_total
    
    # Test with a non-total transition function
    incomplete_fsm = FiniteStateMachine(
        states=fsm_components["states"],
        alphabet=fsm_components["alphabet"],
        initial_state=fsm_components["initial_state"],
        transition_function={("q0", "a"): "q1"},  # Missing transitions
        final_states=fsm_components["final_states"]
    )
    is_total, missing_state, missing_symbol = \
        incomplete_fsm.is_transition_function_total()
    assert not is_total
    assert missing_state in fsm_components["states"]
    assert missing_symbol in fsm_components["alphabet"]


def test_accepts_method(fsm):
    """Test the accepts method with various inputs"""
    # Strings that should be accepted (ending with 'ab')
    assert fsm.accepts("ab")
    assert fsm.accepts("aab")
    assert fsm.accepts("bab")
    assert fsm.accepts("aaab")
    assert fsm.accepts("abab")
    
    # Strings that should be rejected
    assert not fsm.accepts("")
    assert not fsm.accepts("a")
    assert not fsm.accepts("b")
    assert not fsm.accepts("ba")
    assert not fsm.accepts("aba")
    assert not fsm.accepts("abb")
    
    # Test with invalid symbol
    with pytest.raises(ValueError):
        fsm.accepts("abc")  # 'c' is not in the alphabet


def test_runner_step(runner):
    """Test the runner's step method"""
    # Start at q0
    assert runner.current_state == "q0"
    
    # Step with 'a' -> q1
    assert runner.step("a") == "q1"
    assert runner.current_state == "q1"
    
    # Step with 'b' -> q2 (accepting state)
    assert runner.step("b") == "q2"
    assert runner.current_state == "q2"
    assert runner.accepted
    
    # Step with 'b' -> q0 (non-accepting state)
    assert runner.step("b") == "q0"
    assert runner.current_state == "q0"
    assert not runner.accepted
    
    # Test with invalid symbol
    with pytest.raises(ValueError):
        runner.step("c")  # 'c' is not in the alphabet


def test_runner_process_sequence(runner):
    """Test the runner's process_sequence method"""
    # Reset to initial state
    runner.reset()
    
    # Process "ab" -> should end in accepting state q2
    assert runner.process_sequence(["a", "b"])
    assert runner.current_state == "q2"
    assert runner.accepted
    
    # Reset and process "aba" -> should end in non-accepting state q1
    runner.reset()
    assert not runner.process_sequence(["a", "b", "a"])
    assert runner.current_state == "q1"
    assert not runner.accepted


def test_transition_history(runner):
    """Test that the transition history is recorded correctly"""
    runner.reset()
    
    # Process "abab"
    runner.process_sequence(["a", "b", "a", "b"])
    
    # Expected transitions: (q0,a,q1), (q1,b,q2), (q2,a,q1), (q1,b,q2)
    expected_history = [
        ("q0", "a", "q1"),
        ("q1", "b", "q2"),
        ("q2", "a", "q1"),
        ("q1", "b", "q2")
    ]
    
    assert runner.transition_history == expected_history
