
from __future__ import annotations
def bps_to_frac(bps: float) -> float:
    return float(bps) / 10000.0

def expected_value_bps(p_up: float, rr: float, fee_bps: float, slip_bps: float) -> float:
    """EV в б.п. при симметричных входах: RR = TPdist/SLdist (например 1.8).
    Учитываем комиссии обеих сторон и слippage на вход/выход.
    Формула: EV = p_up*(+RR) + (1-p_up)*(-1) - cost, где cost = 2*fee + 2*slip (в б.п.).
    Возвращает EV в б.п. (basis points).
    """
    cost = 2*fee_bps + 2*slip_bps
    ev = p_up*rr*10000.0 + (1.0 - p_up)*(-10000.0) - cost
    return ev

def pass_ev_gate(p_up: float, rr: float, fee_bps: float, slip_bps: float, min_ev_bps: float) -> bool:
    return expected_value_bps(p_up, rr, fee_bps, slip_bps) >= float(min_ev_bps)


def expected_value_dynamic(p_up: float, rr: float, fee_bps_entry: float, fee_bps_exit: float, impact_bps_entry: float, impact_bps_exit: float) -> float:
    """EV в б.п. с динамическими издержками: комиссии и импакт на вход и выход.
    Консервативно считаем импакт для выхода таким же, как для входа (или задаём отдельно).
    EV = p_up*(+RR*10000 - (fee_e+fee_x) - (imp_e+imp_x)) + (1-p_up)*(-10000 - (fee_e+fee_x) - (imp_e+imp_x))
    """
    cost = (fee_bps_entry + fee_bps_exit + impact_bps_entry + impact_bps_exit)
    ev = p_up*(rr*10000.0 - cost) + (1.0 - p_up)*(-10000.0 - cost)
    return ev
