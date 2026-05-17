"""
Steady-state physics analysis for P&ID fluid systems.

Supports known systems registered via _register():
  - psas_pid-20: PSAS LFETS Pressure Panel (N2 pressurization)

Each analysis() call returns a dict with subsystems, each containing
operating_conditions (measurables) and derived (computed quantities).
All SI units unless noted.  Uncertainties flagged with assumptions[].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Physical constants ─────────────────────────────────────────────────────────
R_UNIVERSAL = 8.314          # J/(mol·K)
G_STD       = 9.80665        # m/s²

# Molar masses (kg/mol)
M_N2 = 0.028014

def _psia_to_pa(psia: float) -> float:
    return psia * 6894.757


def _ideal_gas_density(P_pa: float, M_kg_mol: float, T_K: float) -> float:
    """kg/m³ from ideal gas."""
    return P_pa * M_kg_mol / (R_UNIVERSAL * T_K)


def _ideal_gas_mass(P_pa: float, V_m3: float, M_kg_mol: float, T_K: float) -> float:
    return _ideal_gas_density(P_pa, M_kg_mol, T_K) * V_m3


# ── PSAS LFETS Pressure Panel ──────────────────────────────────────────────────

PSAS_PARAMS = {
    "bottle_pressure_psia": 3000,        # operating pressure of N2 bottle
    "bottle_volume_L": 49.0,             # typical 49 L DOT bottle
    "bottle_temperature_K": 293,         # ambient storage
    "regulator_set_lox_psia": 350,       # LOX pressurant set point (LFETS spec ~350 psia)
    "regulator_set_fuel_psia": 350,      # Fuel pressurant set point
    "lox_tank_volume_L": 23,             # LFD LOX tank (approx)
    "fuel_tank_volume_L": 18,            # LFD fuel tank (approx)
    "lox_tank_ullage_start_fraction": 0.10,
    "fuel_tank_ullage_start_fraction": 0.10,
    "burn_duration_s": 5,                # short LFETS burn
}


def analyse_psas(params: dict = PSAS_PARAMS) -> Dict[str, Any]:
    P_bottle = _psia_to_pa(params["bottle_pressure_psia"])
    V_bottle  = params["bottle_volume_L"] * 1e-3
    T_bottle  = params["bottle_temperature_K"]

    m_N2_bottle = _ideal_gas_mass(P_bottle, V_bottle, M_N2, T_bottle)

    P_reg_lox  = _psia_to_pa(params["regulator_set_lox_psia"])
    P_reg_fuel = _psia_to_pa(params["regulator_set_fuel_psia"])

    V_lox_tank  = params["lox_tank_volume_L"] * 1e-3
    V_fuel_tank = params["fuel_tank_volume_L"] * 1e-3
    ull_lox  = params["lox_tank_ullage_start_fraction"]
    ull_fuel = params["fuel_tank_ullage_start_fraction"]

    # N2 mass required to fill ullage to regulator pressure at ambient T
    m_N2_lox_ullage  = _ideal_gas_mass(P_reg_lox,  V_lox_tank  * ull_lox,  M_N2, T_bottle)
    m_N2_fuel_ullage = _ideal_gas_mass(P_reg_fuel, V_fuel_tank * ull_fuel, M_N2, T_bottle)

    # Burn-time pressurization: as propellant is consumed the ullage grows.
    # Assume LOX density 1141 kg/m³, flow rate TBD (very rough for PSAS scale)
    # Use simplified: delta_P/dt ≈ 0 (regulated system; regulator maintains set point)

    return {
        "system": "PSAS LFETS Pressure Panel",
        "document": "psas_pid-20",
        "assumptions": [
            "Bottle volume 49 L (standard DOT), assumed ambient temperature 293 K",
            "Regulator set points 350 psia (typical LFETS spec)",
            "LOX/fuel tank volumes estimated from LFETS design report",
            "Ideal gas for N2 (Z ≈ 1.0 at these conditions)",
        ],
        "subsystems": {
            "N2_bottle": {
                "description": "High-pressure N2 supply",
                "operating_conditions": {
                    "pressure_psia": params["bottle_pressure_psia"],
                    "pressure_MPa": round(P_bottle / 1e6, 3),
                    "temperature_K": T_bottle,
                    "volume_L": params["bottle_volume_L"],
                },
                "derived": {
                    "N2_mass_stored_kg": round(m_N2_bottle, 3),
                    "N2_mass_stored_lbm": round(m_N2_bottle / 0.4536, 3),
                    "density_kg_per_m3": round(_ideal_gas_density(P_bottle, M_N2, T_bottle), 2),
                },
            },
            "LOX_pressurant": {
                "description": "Regulated N2 → LOX tank pressurant",
                "operating_conditions": {
                    "regulator_set_psia": params["regulator_set_lox_psia"],
                    "regulator_set_MPa": round(P_reg_lox / 1e6, 3),
                },
                "derived": {
                    "N2_mass_for_initial_ullage_g": round(m_N2_lox_ullage * 1e3, 2),
                    "regulated_density_kg_per_m3": round(
                        _ideal_gas_density(P_reg_lox, M_N2, T_bottle), 4),
                    "bottle_depletion_fraction_for_ullage": round(
                        m_N2_lox_ullage / m_N2_bottle, 4),
                },
            },
            "Fuel_pressurant": {
                "description": "Regulated N2 → Fuel tank pressurant",
                "operating_conditions": {
                    "regulator_set_psia": params["regulator_set_fuel_psia"],
                    "regulator_set_MPa": round(P_reg_fuel / 1e6, 3),
                },
                "derived": {
                    "N2_mass_for_initial_ullage_g": round(m_N2_fuel_ullage * 1e3, 2),
                    "regulated_density_kg_per_m3": round(
                        _ideal_gas_density(P_reg_fuel, M_N2, T_bottle), 4),
                    "bottle_depletion_fraction_for_ullage": round(
                        m_N2_fuel_ullage / m_N2_bottle, 4),
                },
            },
            "total_N2_budget": {
                "description": "Overall N2 mass budget",
                "derived": {
                    "m_bottle_kg": round(m_N2_bottle, 3),
                    "m_pressurant_needed_kg": round(m_N2_lox_ullage + m_N2_fuel_ullage, 4),
                    "margin_fraction": round(
                        1.0 - (m_N2_lox_ullage + m_N2_fuel_ullage) / m_N2_bottle, 3),
                    "note": "Margin covers burn-time ullage growth — actual need is higher",
                },
            },
        },
    }


# ── Dispatcher ─────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Any] = {}


def _register(key_fragment: str, fn):
    _REGISTRY[key_fragment] = fn


_register("psas_pid-20",        analyse_psas)


def analyse(doc_id: str) -> Optional[Dict[str, Any]]:
    """Return physics analysis for a known document ID, or None."""
    for fragment, fn in _REGISTRY.items():
        if fragment in doc_id:
            return fn()
    return None


def list_supported_documents() -> List[str]:
    return list(_REGISTRY.keys())
