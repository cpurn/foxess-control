"""Tests for FoxESS WebSocket real-time data mapping and password hashing."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from .conftest import SimulatorHandle

from custom_components.foxess_control.foxess.realtime_ws import (
    FoxESSRealtimeWS,
    GridDirectionTracker,
    _is_plausible,
    map_ws_to_coordinator,
)
from custom_components.foxess_control.foxess.web_session import (
    ensure_password_hash,
)

# ---------------------------------------------------------------------------
# ensure_password_hash
# ---------------------------------------------------------------------------


class TestEnsurePasswordHash:
    def test_raw_password_is_hashed(self) -> None:
        result = ensure_password_hash("mypassword")
        assert result == hashlib.md5(b"mypassword").hexdigest()

    def test_md5_hex_passed_through(self) -> None:
        md5 = hashlib.md5(b"test").hexdigest()
        assert ensure_password_hash(md5) == md5

    def test_md5_uppercase_lowered(self) -> None:
        md5 = hashlib.md5(b"test").hexdigest().upper()
        assert ensure_password_hash(md5) == md5.lower()

    def test_31_char_string_is_hashed(self) -> None:
        short = "a" * 31
        result = ensure_password_hash(short)
        assert result == hashlib.md5(short.encode()).hexdigest()

    def test_33_char_string_is_hashed(self) -> None:
        long = "a" * 33
        result = ensure_password_hash(long)
        assert result == hashlib.md5(long.encode()).hexdigest()

    def test_non_hex_32_chars_is_hashed(self) -> None:
        non_hex = "g" * 32
        result = ensure_password_hash(non_hex)
        assert result == hashlib.md5(non_hex.encode()).hexdigest()

    def test_empty_string_is_hashed(self) -> None:
        result = ensure_password_hash("")
        assert result == hashlib.md5(b"").hexdigest()


# ---------------------------------------------------------------------------
# generate_signature — WASM request signing
# ---------------------------------------------------------------------------


class TestGenerateSignature:
    def test_known_signature(self) -> None:
        import custom_components.foxess_control.foxess.signature as sig_mod

        # Reset the WASM singleton — prior tests' calls leave residual
        # heap state that changes the output suffix.
        sig_mod._engine = None
        sig = sig_mod.generate_signature(
            "/basic/v0/user/login", "", "en", "1776124242356"
        )
        assert sig == "02ed69731394e020c1a7e28d56a51013.5245784"

    def test_different_timestamp_gives_different_signature(self) -> None:
        from custom_components.foxess_control.foxess.signature import (
            generate_signature,
        )

        sig1 = generate_signature("/basic/v0/user/login", "", "en", "1776124242356")
        sig2 = generate_signature("/basic/v0/user/login", "", "en", "1776124300000")
        assert sig1 != sig2

    def test_signature_format(self) -> None:
        from custom_components.foxess_control.foxess.signature import (
            generate_signature,
        )

        sig = generate_signature("/basic/v0/user/login", "", "en", "1776124242356")
        parts = sig.split(".")
        assert len(parts) == 2
        assert len(parts[0]) == 32  # MD5 hex


# ---------------------------------------------------------------------------
# map_ws_to_coordinator — pure data mapping
# ---------------------------------------------------------------------------


class TestMapWsToCoordinator:
    def _make_msg(self, **node_overrides: object) -> dict[str, object]:
        """Build a minimal WebSocket message with node data.

        Default: solar 3500 W, load 2000 W, bat discharging 1500 W.
        Power balance (kW): 2.0 + 0 - 1.5 - 3.5 = -3.0 → exporting.
        """
        node = {
            "solar": {"power": {"value": "3500"}},
            "grid": {
                "power": {"value": "2000"},
                "gridStatus": 2,
            },
            "bat": {
                "power": {"value": "1500"},
                "soc": 65,
                "charge": 0,
            },
            "load": {"power": {"value": "2000"}},
        }
        node.update(node_overrides)
        return {"errno": 0, "result": {"node": node, "timeDiff": 5}}

    def test_basic_mapping_export(self) -> None:
        """Default scenario: solar excess → grid export."""
        data = map_ws_to_coordinator(self._make_msg())
        assert data["SoC"] == 65.0
        assert data["pvPower"] == pytest.approx(3.5)
        assert data["loadsPower"] == pytest.approx(2.0)
        assert data["batDischargePower"] == pytest.approx(
            1.5
        )  # charge=0 -> discharging
        assert data["batChargePower"] == 0.0
        # Power balance negative → exporting
        assert data["feedinPower"] == pytest.approx(2.0)
        assert data["gridConsumptionPower"] == 0.0

    def test_battery_charging(self) -> None:
        msg = self._make_msg(bat={"power": {"value": "2000"}, "soc": 45, "charge": 1})
        data = map_ws_to_coordinator(msg)
        assert data["SoC"] == 45.0
        assert data["batChargePower"] == pytest.approx(2.0)
        assert data["batDischargePower"] == 0.0

    def test_battery_charge_flag_as_string(self) -> None:
        """WS may send charge flag as string or int."""
        msg = self._make_msg(bat={"power": {"value": "2000"}, "soc": 45, "charge": "1"})
        data = map_ws_to_coordinator(msg)
        assert data["batChargePower"] == pytest.approx(2.0)
        assert data["batDischargePower"] == 0.0

    def test_grid_importing_from_balance(self) -> None:
        """Grid direction derived from power balance: load > solar → import."""
        msg = self._make_msg(
            solar={"power": {"value": "500"}},
            load={"power": {"value": "2000"}},
            bat={"power": {"value": "10000"}, "soc": 45, "charge": 1},
            grid={"power": {"value": "11500"}, "gridStatus": 99},
        )
        data = map_ws_to_coordinator(msg)
        # balance (kW): 2.0 + 10.0 - 0 - 0.5 = 11.5 > 0 → importing
        assert data["gridConsumptionPower"] == pytest.approx(11.5)
        assert data["feedinPower"] == 0.0

    def test_grid_exporting_from_balance(self) -> None:
        """Grid direction derived from power balance: solar > load → export."""
        msg = self._make_msg(
            solar={"power": {"value": "5000"}},
            load={"power": {"value": "1000"}},
            bat={"power": {"value": "1000"}, "soc": 80, "charge": 1},
            grid={"power": {"value": "3000"}, "gridStatus": 99},
        )
        data = map_ws_to_coordinator(msg)
        # balance (kW): 1.0 + 1.0 - 0 - 5.0 = -3.0 < 0 → exporting
        assert data["gridConsumptionPower"] == 0.0
        assert data["feedinPower"] == pytest.approx(3.0)

    def test_grid_balance_unreliable_unmeasured_generation(self) -> None:
        """External generation not visible to FoxESS skews the balance.

        Balance predicts import 0.28 kW but actual grid is 1.31 kW —
        the 3x+ divergence triggers fallback to gridStatus which correctly
        reports export (gridStatus=2).  Reproduces GitHub issue #3.
        """
        msg = self._make_msg(
            solar={"power": {"value": "0"}},
            load={"power": {"value": "280"}},
            bat={"power": {"value": "5"}, "soc": 100, "charge": 1},
            grid={"power": {"value": "1310"}, "gridStatus": 2},
        )
        data = map_ws_to_coordinator(msg)
        # Balance: 0.28 + 0.005 - 0 - 0 = 0.285 (predicts import)
        # Actual grid: 1.31 kW — ratio 4.6x → balance unreliable
        # gridStatus=2 → exporting
        assert data["gridConsumptionPower"] == 0.0
        assert data["feedinPower"] == pytest.approx(1.31)

    def test_grid_balance_unreliable_importing(self) -> None:
        """Divergent balance falls back to gridStatus=3 → importing."""
        msg = self._make_msg(
            solar={"power": {"value": "5000"}},
            load={"power": {"value": "200"}},
            bat={"power": {"value": "0"}, "soc": 100, "charge": 0},
            grid={"power": {"value": "800"}, "gridStatus": "3"},
        )
        data = map_ws_to_coordinator(msg)
        # Balance: 0.2 + 0 - 0 - 5.0 = -4.8 (predicts export)
        # Actual grid: 0.8 kW — ratio 6x → balance unreliable
        # gridStatus=3 → importing
        assert data["gridConsumptionPower"] == pytest.approx(0.8)
        assert data["feedinPower"] == 0.0

    def test_grid_fallback_to_gridstatus(self) -> None:
        """When solar/load missing, fall back to gridStatus."""
        msg = {
            "errno": 0,
            "result": {
                "node": {
                    "grid": {"power": {"value": "1500"}, "gridStatus": "3"},
                },
                "timeDiff": 5,
            },
        }
        data = map_ws_to_coordinator(msg)
        assert data["gridConsumptionPower"] == pytest.approx(1.5)
        assert data["feedinPower"] == 0.0

    def test_empty_message(self) -> None:
        assert map_ws_to_coordinator({}) == {}

    def test_empty_node(self) -> None:
        assert map_ws_to_coordinator({"result": {"node": {}}}) == {}

    def test_missing_power_value(self) -> None:
        msg = self._make_msg(solar={"power": None})
        data = map_ws_to_coordinator(msg)
        assert "pvPower" not in data

    def test_non_numeric_value_skipped(self) -> None:
        msg = self._make_msg(solar={"power": {"value": "N/A", "unit": "W"}})
        data = map_ws_to_coordinator(msg)
        assert "pvPower" not in data

    def test_zero_power(self) -> None:
        msg = self._make_msg(solar={"power": {"value": "0"}})
        data = map_ws_to_coordinator(msg)
        assert data["pvPower"] == 0.0

    def test_battery_soc_type_conversion(self) -> None:
        msg = self._make_msg(bat={"power": {"value": "0"}, "soc": "75", "charge": 0})
        data = map_ws_to_coordinator(msg)
        assert data["SoC"] == 75.0

    def test_error_message_returns_empty(self) -> None:
        msg = {"errno": 1, "result": {"node": {}}}
        # map_ws_to_coordinator only maps data, caller checks errno
        data = map_ws_to_coordinator(msg)
        assert data == {}

    def test_kw_unit_field_skips_division(self) -> None:
        """When WS sends unit='kW' on a field, use value as-is."""
        msg = {
            "errno": 0,
            "result": {
                "node": {
                    "solar": {"power": {"value": "0", "unit": "W"}},
                    "grid": {"power": {"value": "4700", "unit": "W"}, "gridStatus": 2},
                    "bat": {
                        "power": {"value": "5.29", "unit": "kW"},
                        "soc": 46,
                        "charge": 0,
                    },
                    "load": {"power": {"value": "427", "unit": "W"}},
                },
                "timeDiff": 5,
            },
        }
        data = map_ws_to_coordinator(msg)
        # bat unit=kW → used directly; others unit=W → /1000
        assert data["batDischargePower"] == pytest.approx(5.29)
        assert data["loadsPower"] == pytest.approx(0.427)

    def test_mixed_units_handled_per_field(self) -> None:
        """WS can send kW for battery but W for everything else."""
        msg = {
            "errno": 0,
            "result": {
                "node": {
                    "solar": {"power": {"value": "0", "unit": "W"}},
                    "grid": {
                        "power": {"value": "5000", "unit": "W"},
                        "gridStatus": 2,
                    },
                    "bat": {
                        "power": {"value": "5.46", "unit": "kW"},
                        "soc": 44,
                        "charge": 0,
                    },
                    "load": {"power": {"value": "427", "unit": "W"}},
                },
                "timeDiff": 5,
            },
        }
        data = map_ws_to_coordinator(msg)
        assert data["batDischargePower"] == pytest.approx(5.46)
        assert data["loadsPower"] == pytest.approx(0.427)
        # balance: 0.427 + 0 - 5.46 - 0 = -5.033 → exporting
        assert data["feedinPower"] == pytest.approx(5.0)

    def test_real_world_sample(self) -> None:
        """Test with actual FoxESS WebSocket message structure.

        Values are in watts (as strings), despite the unit field
        sometimes reading "W".  The coordinator expects kW.
        """
        msg = {
            "errno": 0,
            "msg": "",
            "result": {
                "node": {
                    "solar": {"power": {"value": "809", "unit": "W"}},
                    "grid": {
                        "power": {"value": "19", "unit": "W"},
                        "gridStatus": 3,
                        "gridToHidden": -1,
                    },
                    "bat": {
                        "power": {"value": "607", "unit": "W"},
                        "soc": 34,
                        "charge": 1,
                        "batToDevice": -1,
                    },
                    "load": {
                        "power": {"value": "183", "unit": "W"},
                        "normalLoad": {"power": {"value": "183", "unit": "W"}},
                        "backupLoad": {"power": {"value": "0", "unit": "W"}},
                    },
                    "device": {"power": {"value": "202", "unit": "W"}},
                    "charger": {"display": False},
                    "heatpump": {"display": False},
                },
                "timeDiff": 5,
                "lastUpdateDate": "Updated  within 1 minute",
                "plantId": "8d3f1896-19a6-40b0-86a1-d892185f5366",
            },
        }
        data = map_ws_to_coordinator(msg)
        assert data["SoC"] == 34.0
        assert data["pvPower"] == pytest.approx(0.809)
        assert data["batChargePower"] == pytest.approx(0.607)
        assert data["batDischargePower"] == 0.0
        assert data["loadsPower"] == pytest.approx(0.183)
        # balance (kW): 0.183 + 0.607 - 0 - 0.809 = -0.019 → slight export
        assert data["gridConsumptionPower"] == 0.0
        assert data["feedinPower"] == pytest.approx(0.019)


# ---------------------------------------------------------------------------
# Grid direction with an unmeasured / separately-reported EV charger
# (live capture 2026-10-27, EV on the house side of the inverter, gridStatus
# reads 2 for IMPORT on this firmware but 2 means EXPORT in GH issue #3)
# ---------------------------------------------------------------------------


def _ev_msg(
    *,
    solar: float,
    load: float,
    bat_charge: float,
    grid: float,
    charger: float | None = 0.0,
    grid_status: object = 2,
    grid_flag: object = 1,
    device: float | None = None,
) -> dict[str, object]:
    """Build a WS frame shaped like the live capture (all values in kW)."""

    def p(kw: float) -> dict[str, str]:
        return {"value": f"{kw}", "unit": "kW"}

    grid_node: dict[str, object] = {"power": p(grid), "gridStatus": grid_status}
    if grid_flag is not None:
        grid_node["gridToHidden"] = grid_flag
    node: dict[str, object] = {
        "solar": {"power": p(solar)},
        "grid": grid_node,
        "bat": {
            "power": p(bat_charge),
            "soc": 50,
            "charge": 1,
            "batteryId": "00000000-0000-0000-0000-000000000000",
            "multipleBatterySoc": [{"soc": 50, "charge": 1, "batSn": "SN-PLACEHOLDER"}],
        },
        "load": {"power": p(load)},
        "homeLoad": {"power": p(load)},
        "heatpump": {"display": False, "power": {"value": "--", "unit": "kW"}},
        "aux": {"display": False, "power": {"value": "--", "unit": "kW"}},
    }
    if charger is not None:
        node["charger"] = {"power": p(charger), "state": 0, "chargerToHidden": 1}
    if device is not None:
        node["device"] = {"power": p(device)}
    return {"errno": 0, "result": {"node": node, "timeDiff": 5}}


# (label, solar, load, charger, bat_charge, device, grid) — verbatim from the log
_LIVE_IMPORT_SAMPLES = [
    ("ev_inside_load", 7.76, 2.99, 0.0, 8.51, -0.744, 3.73),
    ("switchover_charger_2.6_load_4.7", 7.71, 4.72, 2.60, 8.52, -0.811, 8.13),
    ("settled_ev_6.9", 7.73, 0.464, 6.90, 8.52, -0.787, 8.15),
    ("settled_ratio_6.5x", 7.69, 0.386, 6.90, 8.52, -0.822, 8.11),
    ("solar_dip", 2.71, 0.309, 6.90, 7.69, -4.98, 12.2),
    ("solar_recovers", 7.26, 0.454, 6.90, 8.52, -1.26, 8.61),
]


class TestGridDirectionEvCharger:
    @pytest.mark.parametrize(
        ("label", "solar", "load", "charger", "bat", "device", "grid"),
        _LIVE_IMPORT_SAMPLES,
        ids=[s[0] for s in _LIVE_IMPORT_SAMPLES],
    )
    def test_live_import_samples_never_report_export(
        self,
        label: str,
        solar: float,
        load: float,
        charger: float,
        bat: float,
        device: float,
        grid: float,
    ) -> None:
        msg = _ev_msg(
            solar=solar,
            load=load,
            charger=charger,
            bat_charge=bat,
            device=device,
            grid=grid,
        )
        data = map_ws_to_coordinator(msg)
        assert data["gridConsumptionPower"] == pytest.approx(grid)
        assert data["feedinPower"] == 0.0

    def test_signed_flag_beats_wrong_gridstatus(self) -> None:
        """11:07:56 — balance without EV is 6.5x off; gridStatus=2 is import here."""
        msg = _ev_msg(solar=7.73, load=0.464, charger=None, bat_charge=8.52, grid=8.15)
        data = map_ws_to_coordinator(msg)
        assert data["gridConsumptionPower"] == pytest.approx(8.15)
        assert data["feedinPower"] == 0.0

    def test_live_export_sample_still_exports(self) -> None:
        """11:01:01 — before Force Charge: gridStatus=1, gridToHidden=-1."""
        msg = _ev_msg(
            solar=7.67,
            load=0.476,
            charger=0.0,
            bat_charge=2.19,
            grid=5.01,
            grid_status=1,
            grid_flag=-1,
            device=5.48,
        )
        data = map_ws_to_coordinator(msg)
        assert data["gridConsumptionPower"] == 0.0
        assert data["feedinPower"] == pytest.approx(5.01)

    def test_flag_accepts_string_values(self) -> None:
        msg = _ev_msg(
            solar=1.0, load=0.5, bat_charge=0.0, grid=5.0, charger=0.0, grid_flag="1"
        )
        assert map_ws_to_coordinator(msg)["gridConsumptionPower"] == pytest.approx(5.0)
        msg = _ev_msg(
            solar=9.0, load=0.5, bat_charge=0.0, grid=5.0, charger=0.0, grid_flag="-1"
        )
        assert map_ws_to_coordinator(msg)["feedinPower"] == pytest.approx(5.0)

    def test_foxess_app_48x_divergence(self) -> None:
        """App sample: solar 8.82, home 0.455, charge 8.51, EV 6.9, grid 7.05.

        Balance without the EV predicts 0.15 kW vs 7.05 actual (48x).  No
        signed flag in the frame: the charger node must close the balance.
        """
        msg = _ev_msg(
            solar=8.82,
            load=0.455,
            charger=6.9,
            bat_charge=8.51,
            grid=7.05,
            grid_flag=None,
        )
        data = map_ws_to_coordinator(msg)
        assert data["gridConsumptionPower"] == pytest.approx(7.05)
        assert data["feedinPower"] == 0.0

    def test_charger_node_does_not_change_reported_house_load(self) -> None:
        msg = _ev_msg(solar=7.73, load=0.464, charger=6.9, bat_charge=8.52, grid=8.15)
        data = map_ws_to_coordinator(msg)
        assert data["loadsPower"] == pytest.approx(0.464)

    def test_charger_placeholder_values_ignored(self) -> None:
        """display=False nodes carry '--' — must not break the balance."""
        msg = _ev_msg(
            solar=2.0, load=1.0, charger=None, bat_charge=0.0, grid=1.0, grid_flag=None
        )
        msg["result"]["node"]["charger"] = {  # type: ignore[index]
            "display": False,
            "power": {"value": "--", "unit": "kW"},
        }
        data = map_ws_to_coordinator(msg)
        assert data["feedinPower"] == pytest.approx(1.0)  # 1.0 - 2.0 → export
        assert data["gridConsumptionPower"] == 0.0


def _charger_placeholder(msg: dict[str, object]) -> dict[str, object]:
    """Replace the charger node by what the WS sends while it has no reading."""
    msg["result"]["node"]["charger"] = {  # type: ignore[index]
        "power": {"value": "--", "unit": "kW"},
        "display": True,
        "state": 0,
        "chargerToHidden": 1,
    }
    return msg


class TestGridDirectionChargerPlaceholder:
    """Live capture 2026-10-28 (Force Charge, EV at 32 A)."""

    def test_ev_inside_load_charger_placeholder(self) -> None:
        """charger.power '--' (EV still counted in load): 5.15+8.50-2.03=11.62."""
        msg = _charger_placeholder(
            _ev_msg(solar=2.03, load=5.15, bat_charge=8.50, grid=11.6, charger=None)
        )
        data = map_ws_to_coordinator(msg)
        assert data["gridConsumptionPower"] == pytest.approx(11.6)
        assert data["feedinPower"] == 0.0

    def test_settled_ev_in_charger_node(self) -> None:
        """EV split out: 0.905+6.14+6.90-1.75=12.195 vs grid 12.2."""
        msg = _ev_msg(solar=1.75, load=0.905, bat_charge=6.14, grid=12.2, charger=6.90)
        data = map_ws_to_coordinator(msg)
        assert data["gridConsumptionPower"] == pytest.approx(12.2)
        assert data["feedinPower"] == 0.0

    @pytest.mark.parametrize("grid_status", [2, 3])
    def test_no_flag_and_placeholder_decided_by_balance(self, grid_status: int) -> None:
        """No gridToHidden, charger '--' (= 0): the balance must decide.

        gridStatus is deliberately 2 (import on this firmware) and 3 (what the
        legacy fallback treated as export) — neither may influence the result.
        """
        msg = _charger_placeholder(
            _ev_msg(
                solar=2.03,
                load=5.15,
                bat_charge=8.50,
                grid=11.6,
                charger=None,
                grid_status=grid_status,
                grid_flag=None,
            )
        )
        tracker = GridDirectionTracker()
        data = map_ws_to_coordinator(msg, tracker)
        assert data["gridConsumptionPower"] == pytest.approx(11.6)
        assert data["feedinPower"] == 0.0
        # the decision came from the balance, not from the status code
        assert tracker._status_meaning == {str(grid_status): True}


class TestGridDirectionTracker:
    """Direction must survive unmeasured load/generation (no flag in frame)."""

    @staticmethod
    def _bare(
        *, solar: float, load: float, bat: float, grid: float, status: object
    ) -> dict[str, object]:
        return _ev_msg(
            solar=solar,
            load=load,
            charger=None,
            bat_charge=bat,
            grid=grid,
            grid_status=status,
            grid_flag=None,
        )

    @pytest.mark.parametrize("ev_kw", [4.0, 5.0, 6.0, 7.0, 8.0])
    def test_variable_unmeasured_load_holds_import(self, ev_kw: float) -> None:
        """5-min averages 11:15: solar 7.7, load 2.7, charge 8.5, real grid 10.3.

        The EV is invisible (no charger node, no flag).  Whatever the
        4-8 kW EV draw is, the ratio to the balance hovers around the 3x
        cutoff — direction must stay import.
        """
        tracker = GridDirectionTracker()
        # A trustworthy frame first (balance matches): learns import / status 2.
        first = map_ws_to_coordinator(
            self._bare(solar=2.0, load=1.0, bat=8.5, grid=7.5, status=2), tracker
        )
        assert first["gridConsumptionPower"] == pytest.approx(7.5)
        grid = 3.5 + ev_kw  # balance (3.5) + unmeasured EV
        data = map_ws_to_coordinator(
            self._bare(solar=7.7, load=2.7, bat=8.5, grid=grid, status=2), tracker
        )
        assert data["gridConsumptionPower"] == pytest.approx(grid)
        assert data["feedinPower"] == 0.0

    def test_ratio_hovering_around_3x_does_not_flip(self) -> None:
        tracker = GridDirectionTracker()
        map_ws_to_coordinator(
            self._bare(solar=2.0, load=1.0, bat=8.5, grid=7.5, status=2), tracker
        )
        # predicted 3.5 vs actual 10.3 (2.9x), 11.1 vs 3.3 (3.4x), 8.9 vs 4.8
        for solar, load, bat, grid in [
            (7.7, 2.7, 8.5, 10.3),
            (5.2, 0.9, 7.6, 11.1),
            (4.4, 0.7, 8.5, 8.9),
            (7.7, 2.7, 8.5, 11.8),
            (7.7, 2.7, 8.5, 10.4),
        ]:
            data = map_ws_to_coordinator(
                self._bare(solar=solar, load=load, bat=bat, grid=grid, status=2),
                tracker,
            )
            assert data["gridConsumptionPower"] == pytest.approx(grid)
            assert data["feedinPower"] == 0.0

    def test_48x_divergence_holds_previous_direction(self) -> None:
        tracker = GridDirectionTracker()
        map_ws_to_coordinator(
            self._bare(solar=2.0, load=1.0, bat=8.5, grid=7.5, status=2), tracker
        )
        data = map_ws_to_coordinator(
            self._bare(solar=8.82, load=0.455, bat=8.51, grid=7.05, status=2), tracker
        )
        assert data["gridConsumptionPower"] == pytest.approx(7.05)
        assert data["feedinPower"] == 0.0

    def test_unknown_gridstatus_holds_previous_direction(self) -> None:
        tracker = GridDirectionTracker()
        map_ws_to_coordinator(
            self._bare(solar=2.0, load=1.0, bat=8.5, grid=7.5, status=2), tracker
        )
        data = map_ws_to_coordinator(
            self._bare(solar=7.7, load=0.5, bat=8.5, grid=8.0, status=99), tracker
        )
        assert data["gridConsumptionPower"] == pytest.approx(8.0)

    def test_gridstatus_meaning_is_learned_per_firmware(self) -> None:
        """Same gridStatus=2 means export on the issue #3 firmware."""
        tracker = GridDirectionTracker()
        # Reliable export frame teaches status 2 == export.
        map_ws_to_coordinator(
            self._bare(solar=5.0, load=1.0, bat=0.0, grid=4.0, status=2), tracker
        )
        # Then an import frame with unmeasured load: status 2 -> export would
        # be wrong only if learned wrongly; here the learned meaning is used.
        data = map_ws_to_coordinator(
            self._bare(solar=0.0, load=0.28, bat=0.005, grid=1.31, status=2), tracker
        )
        assert data["gridConsumptionPower"] == 0.0
        assert data["feedinPower"] == pytest.approx(1.31)

    def test_issue_3_hold_after_export(self) -> None:
        """GH #3: unmeasured generation; status 99 must not flip direction."""
        tracker = GridDirectionTracker()
        map_ws_to_coordinator(
            self._bare(solar=5.0, load=1.0, bat=0.0, grid=4.0, status=1), tracker
        )
        data = map_ws_to_coordinator(
            self._bare(solar=0.0, load=0.28, bat=0.005, grid=1.31, status=99),
            tracker,
        )
        assert data["gridConsumptionPower"] == 0.0
        assert data["feedinPower"] == pytest.approx(1.31)

    def test_ambiguous_gridstatus_is_not_trusted(self) -> None:
        tracker = GridDirectionTracker()
        map_ws_to_coordinator(
            self._bare(solar=5.0, load=1.0, bat=0.0, grid=4.0, status=2), tracker
        )  # status 2 -> export
        map_ws_to_coordinator(
            self._bare(solar=1.0, load=5.0, bat=0.0, grid=4.0, status=2), tracker
        )  # status 2 -> import (now ambiguous; last direction = import)
        data = map_ws_to_coordinator(
            self._bare(solar=7.7, load=0.5, bat=8.5, grid=8.0, status=2), tracker
        )
        assert data["gridConsumptionPower"] == pytest.approx(8.0)  # held import

    def test_near_zero_grid_does_not_overwrite_direction(self) -> None:
        tracker = GridDirectionTracker()
        map_ws_to_coordinator(
            self._bare(solar=2.0, load=1.0, bat=8.5, grid=7.5, status=2), tracker
        )
        map_ws_to_coordinator(
            self._bare(solar=3.0, load=1.0, bat=2.0, grid=0.01, status=2), tracker
        )
        data = map_ws_to_coordinator(
            self._bare(solar=7.7, load=0.5, bat=8.5, grid=8.0, status=99), tracker
        )
        assert data["gridConsumptionPower"] == pytest.approx(8.0)

    def test_signed_flag_wins_over_reliable_balance_disagreement(self) -> None:
        """Flag is the primary signal: a stale balance cannot override it."""
        msg = _ev_msg(
            solar=0.0, load=1.0, charger=0.0, bat_charge=0.0, grid=1.0, grid_flag=-1
        )
        data = map_ws_to_coordinator(msg, GridDirectionTracker())
        assert data["feedinPower"] == pytest.approx(1.0)

    def test_no_tracker_cold_start_keeps_legacy_gridstatus_fallback(self) -> None:
        msg = self._bare(solar=5.0, load=0.2, bat=0.0, grid=0.8, status="3")
        data = map_ws_to_coordinator(msg)
        assert data["gridConsumptionPower"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# WS warmup — first N messages after connect are skipped
# ---------------------------------------------------------------------------


class TestStaleness:
    """Verify the WebSocket skips stale messages based on timeDiff."""

    @staticmethod
    def _make_ws_msg(time_diff: int = 5, soc: int = 65) -> aiohttp.WSMessage:
        import json

        return aiohttp.WSMessage(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(
                {
                    "errno": 0,
                    "result": {
                        "node": {
                            "solar": {"power": {"value": "3500"}},
                            "grid": {"power": {"value": "2000"}, "gridStatus": 2},
                            "bat": {
                                "power": {"value": "1500"},
                                "soc": soc,
                                "charge": 0,
                            },
                            "load": {"power": {"value": "2000"}},
                        },
                        "timeDiff": time_diff,
                    },
                }
            ),
            extra=None,
        )

    @pytest.mark.asyncio
    async def test_stale_messages_skipped(self) -> None:
        """Messages with timeDiff > MAX_TIME_DIFF are not forwarded."""
        on_data = AsyncMock()
        on_disconnect = MagicMock()
        web_session = AsyncMock()
        web_session.async_ensure_token = AsyncMock(return_value="tok")

        ws = FoxESSRealtimeWS("plant1", web_session, on_data, on_disconnect)

        messages = [
            self._make_ws_msg(time_diff=215),  # stale — skip
            self._make_ws_msg(time_diff=60),  # stale — skip
            self._make_ws_msg(time_diff=5),  # fresh — forward
            self._make_ws_msg(time_diff=5),  # fresh — forward
            aiohttp.WSMessage(type=aiohttp.WSMsgType.CLOSED, data=None, extra=None),
        ]

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(side_effect=messages)
        mock_ws.closed = True

        ws._ws = mock_ws
        ws._connected = True
        ws._stop_event.clear()

        with patch.object(
            ws, "_try_reconnect", new_callable=AsyncMock
        ) as mock_reconnect:

            async def _fail_reconnect() -> None:
                ws._connected = False

            mock_reconnect.side_effect = _fail_reconnect
            await ws._listen_loop()

        # Only the 2 fresh messages (timeDiff=5) should be forwarded
        assert on_data.call_count == 2


class TestIsPlausible:
    """Plausibility filter: reject WS messages where any power key diverges >10x."""

    NORMAL = {
        "SoC": 83.0,
        "batChargePower": 0.0,
        "batDischargePower": 5.5,
        "pvPower": 0.0,
        "loadsPower": 0.48,
        "gridConsumptionPower": 0.0,
        "feedinPower": 5.02,
    }

    def test_similar_values_accepted(self) -> None:
        candidate = {**self.NORMAL, "batDischargePower": 5.49, "feedinPower": 5.01}
        assert _is_plausible(candidate, self.NORMAL) is True

    def test_aberrant_battery_rejected(self) -> None:
        candidate = {**self.NORMAL, "batDischargePower": 0.53, "feedinPower": 0.07}
        assert _is_plausible(candidate, self.NORMAL) is False

    def test_aberrant_feedin_only_rejected(self) -> None:
        candidate = {**self.NORMAL, "feedinPower": 0.05}
        assert _is_plausible(candidate, self.NORMAL) is False

    def test_near_zero_reference_accepts_any(self) -> None:
        ref = {**self.NORMAL, "batDischargePower": 0.05}
        candidate = {**self.NORMAL, "batDischargePower": 5.5}
        assert _is_plausible(candidate, ref) is True

    def test_zero_candidate_always_accepted(self) -> None:
        candidate = {**self.NORMAL, "batDischargePower": 0.0, "feedinPower": 0.0}
        assert _is_plausible(candidate, self.NORMAL) is True

    def test_no_reference_always_accepted(self) -> None:
        assert _is_plausible(self.NORMAL, None) is True

    def test_empty_reference_always_accepted(self) -> None:
        assert _is_plausible(self.NORMAL, {"SoC": 80.0}) is True

    def test_missing_candidate_key_accepted(self) -> None:
        candidate = {"SoC": 83.0, "batDischargePower": 5.5, "loadsPower": 0.48}
        assert _is_plausible(candidate, self.NORMAL) is True

    def test_charge_anomaly_rejected(self) -> None:
        ref = {**self.NORMAL, "batChargePower": 3.8, "batDischargePower": 0.0}
        candidate = {**ref, "batChargePower": 0.35}
        assert _is_plausible(candidate, ref) is False

    def test_solar_anomaly_rejected(self) -> None:
        ref = {**self.NORMAL, "pvPower": 4.0}
        candidate = {**ref, "pvPower": 0.3}
        assert _is_plausible(candidate, ref) is False

    def test_load_anomaly_rejected(self) -> None:
        ref = {**self.NORMAL, "loadsPower": 5.0}
        candidate = {**ref, "loadsPower": 0.4}
        assert _is_plausible(candidate, ref) is False


class TestWsPlausibilityFilter:
    """FoxESSRealtimeWS drops aberrant messages before calling on_data."""

    @staticmethod
    def _make_ws_msg(
        discharge: float = 5500.0,
        feedin: float = 5000.0,
        load: float = 480.0,
        soc: int = 83,
        grid_status: int = 1,
    ) -> aiohttp.WSMessage:
        import json

        return aiohttp.WSMessage(
            type=aiohttp.WSMsgType.TEXT,
            data=json.dumps(
                {
                    "errno": 0,
                    "result": {
                        "node": {
                            "solar": {"power": {"value": "0"}},
                            "grid": {
                                "power": {"value": str(feedin + load)},
                                "gridStatus": grid_status,
                            },
                            "bat": {
                                "power": {"value": str(discharge)},
                                "soc": soc,
                                "charge": 0,
                            },
                            "load": {"power": {"value": str(load)}},
                        },
                        "timeDiff": 5,
                    },
                }
            ),
            extra=None,
        )

    @pytest.mark.asyncio
    async def test_aberrant_message_not_forwarded(self) -> None:
        """Aberrant WS message (10x lower power) must not reach on_data."""
        on_data = AsyncMock()
        on_disconnect = MagicMock()
        web_session = AsyncMock()
        web_session.async_ensure_token = AsyncMock(return_value="tok")

        ws = FoxESSRealtimeWS("plant1", web_session, on_data, on_disconnect)

        messages = [
            self._make_ws_msg(discharge=5500, feedin=5000),  # normal — accepted
            self._make_ws_msg(discharge=530, feedin=70),  # aberrant — dropped
            self._make_ws_msg(discharge=5490, feedin=5010),  # normal — accepted
            aiohttp.WSMessage(type=aiohttp.WSMsgType.CLOSED, data=None, extra=None),
        ]

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(side_effect=messages)
        mock_ws.closed = True

        ws._ws = mock_ws
        ws._connected = True
        ws._stop_event.clear()

        with patch.object(
            ws, "_try_reconnect", new_callable=AsyncMock
        ) as mock_reconnect:

            async def _fail_reconnect() -> None:
                ws._connected = False

            mock_reconnect.side_effect = _fail_reconnect
            await ws._listen_loop()

        assert on_data.call_count == 2, (
            f"Expected 2 calls (aberrant dropped), got {on_data.call_count}"
        )

    @pytest.mark.asyncio
    async def test_first_message_always_accepted(self) -> None:
        """First message after connect has no reference — must be accepted."""
        on_data = AsyncMock()
        on_disconnect = MagicMock()
        web_session = AsyncMock()
        web_session.async_ensure_token = AsyncMock(return_value="tok")

        ws = FoxESSRealtimeWS("plant1", web_session, on_data, on_disconnect)

        messages = [
            self._make_ws_msg(
                discharge=530, feedin=70
            ),  # would be aberrant, but first msg
            aiohttp.WSMessage(type=aiohttp.WSMsgType.CLOSED, data=None, extra=None),
        ]

        mock_ws = AsyncMock()
        mock_ws.receive = AsyncMock(side_effect=messages)
        mock_ws.closed = True

        ws._ws = mock_ws
        ws._connected = True
        ws._stop_event.clear()

        with patch.object(
            ws, "_try_reconnect", new_callable=AsyncMock
        ) as mock_reconnect:

            async def _fail_reconnect() -> None:
                ws._connected = False

            mock_reconnect.side_effect = _fail_reconnect
            await ws._listen_loop()

        assert on_data.call_count == 1

    @pytest.mark.asyncio
    async def test_reconnect_resets_reference(self) -> None:
        """After reconnect, _last_accepted is reset so first message is accepted."""
        on_data = AsyncMock()
        on_disconnect = MagicMock()
        web_session = AsyncMock()
        web_session.async_ensure_token = AsyncMock(return_value="tok")

        ws = FoxESSRealtimeWS("plant1", web_session, on_data, on_disconnect)
        # Simulate a prior accepted message
        ws._last_accepted = {
            "batDischargePower": 5.5,
            "feedinPower": 5.02,
            "loadsPower": 0.48,
        }

        # After reconnect, _last_accepted should be None
        ws2 = FoxESSRealtimeWS("plant1", web_session, on_data, on_disconnect)
        assert ws2._last_accepted is None


class TestReconnectRespectsGate:
    """Regression (FOURTH, distinct from the three start-gate fixes): the
    autonomous reconnect loop must NOT resurrect the connection when the
    WS should be DOWN (no active session).

    Live evidence (2026-06-07, UTC; ``ws_mode=smart_sessions``, confirmed
    idle — ``intelligente_steuerung=idle``, both laden/entladen off,
    ``betriebsmodus=SelfUse``), straight from the rolling debug log::

        10:20:30  WebSocket connected
        10:20:30  WebSocket: skipping stale message (timeDiff=61)
        10:25:59  WebSocket stale (no data in 30s)
        10:25:59  WebSocket disconnected, falling back to REST polling
        10:25:59  WebSocket reconnecting in 6.0s (attempt 1/5)
        10:26:06  WebSocket connected
        10:26:06  WebSocket: skipping stale message (timeDiff=61)

    A self-perpetuating connect → stale(30s) → disconnect → reconnect(6s)
    cycle running with NO active session, every ~5.5 min — the
    ``data_freshness`` ws↔api sawtooth.  Every frame is ``timeDiff=61`` →
    discarded per C-005, so the WS delivers NOTHING useful: pure churn
    (C-020 leak).

    Root cause: ``_try_reconnect`` is fully autonomous — it gates only on
    the instance-local ``_no_reconnect`` / ``_stop_event`` flags.  It never
    consults ``_should_start_realtime_ws``.  The three prior fixes all live
    in the start chokepoint (``_should_start_realtime_ws`` /
    ``_maybe_start_realtime_ws``), which the reconnect loop never calls —
    so the gate's correct "WS should be DOWN" answer never reaches the
    reconnect decision, and the loop revives the connection 6s later,
    below the coordinator's/start-gate's visibility.

    The unified reconcile: ``FoxESSRealtimeWS`` takes a
    ``should_reconnect`` predicate (wired to ``_should_start_realtime_ws``).
    ``_try_reconnect`` consults it before scheduling any network I/O.
    C-020 (UI reflects true state), C-025 (boundary cleanliness),
    C-005 (stale-discard context), D-008/D-009 (reconnect/linger contract).
    """

    @staticmethod
    def _make_ws(
        should_reconnect: Callable[[], bool] | None,
    ) -> FoxESSRealtimeWS:
        on_data = AsyncMock()
        on_disconnect = MagicMock()
        web_session = AsyncMock()
        web_session.async_ensure_token = AsyncMock(return_value="tok")
        # The reconcile predicate wired to the (mocked) start-gate.  The
        # test exercises the *behaviour* of ``_try_reconnect`` — does it
        # consult the gate before scheduling reconnect I/O?  On
        # pre-fix code the reconnect loop ignores the predicate and
        # reconnects unconditionally, so the gate-False behavioural
        # assertions fail, matching the live idle-reconnect symptom.
        return FoxESSRealtimeWS(
            "plant1",
            web_session,
            on_data,
            on_disconnect,
            should_reconnect=should_reconnect,
        )

    @pytest.mark.asyncio
    async def test_no_reconnect_when_gate_false(self) -> None:
        """Gate False (idle) → _try_reconnect must NOT reconnect.

        Drives the real ``_try_reconnect`` (not a mock) with the
        predicate returning False, exactly as it would during confirmed
        idle.  No ``_do_connect`` may be scheduled, no backoff sleep, and
        the connection must end up DOWN.  On current ``develop`` the
        reconnect loop has no gate awareness and reconnects anyway — this
        test FAILS, matching the live symptom.
        """
        gate_calls = {"n": 0}

        def gate_false() -> bool:
            gate_calls["n"] += 1
            return False

        ws = self._make_ws(gate_false)
        ws._connected = True
        ws._stop_event.clear()

        connect_attempts = {"n": 0}

        async def _count_connect(_token: str) -> None:
            connect_attempts["n"] += 1
            ws._connected = True

        with (
            patch.object(ws, "_do_connect", side_effect=_count_connect),
            patch.object(ws, "_close_ws", new_callable=AsyncMock),
            patch(
                "custom_components.foxess_control.foxess.realtime_ws.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            await ws._try_reconnect()

        assert gate_calls["n"] > 0, (
            "_try_reconnect never consulted the should_reconnect gate — "
            "the reconnect loop is still autonomous (the leak)."
        )
        assert connect_attempts["n"] == 0, (
            "_try_reconnect re-established the WS while the gate said it "
            "should be DOWN (no active session) — the self-perpetuating "
            "idle reconnect cycle (C-020 leak)."
        )
        assert mock_sleep.await_count == 0, (
            "_try_reconnect scheduled a backoff sleep despite the gate "
            "refusing — it should short-circuit before any backoff."
        )
        assert ws._connected is False, "WS must be DOWN when the gate is False."

    @pytest.mark.asyncio
    async def test_reconnect_when_gate_true(self) -> None:
        """Inverse (must NOT regress): active session → reconnect DOES happen.

        During a genuine active session the gate returns True; a
        stale/dropped WS must still reconnect (D-008/D-009 — the
        legitimate reconnect case).
        """
        ws = self._make_ws(lambda: True)
        ws._connected = True
        ws._stop_event.clear()

        connect_attempts = {"n": 0}

        async def _count_connect(_token: str) -> None:
            connect_attempts["n"] += 1
            ws._connected = True

        with (
            patch.object(ws, "_do_connect", side_effect=_count_connect),
            patch.object(ws, "_close_ws", new_callable=AsyncMock),
            patch(
                "custom_components.foxess_control.foxess.realtime_ws.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            await ws._try_reconnect()

        assert connect_attempts["n"] == 1, (
            "Active session (gate True): a stale/dropped WS must reconnect "
            "— the legitimate reconnect case was broken."
        )
        assert ws._connected is True

    @pytest.mark.asyncio
    async def test_no_gate_defaults_to_reconnect(self) -> None:
        """Back-compat: no predicate supplied → reconnect as before.

        ``always`` mode and existing call sites that pass no predicate
        must retain the legacy autonomous-reconnect behaviour.
        """
        ws = self._make_ws(None)
        ws._connected = True
        ws._stop_event.clear()

        connect_attempts = {"n": 0}

        async def _count_connect(_token: str) -> None:
            connect_attempts["n"] += 1
            ws._connected = True

        with (
            patch.object(ws, "_do_connect", side_effect=_count_connect),
            patch.object(ws, "_close_ws", new_callable=AsyncMock),
            patch(
                "custom_components.foxess_control.foxess.realtime_ws.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            await ws._try_reconnect()

        assert connect_attempts["n"] == 1


# ---------------------------------------------------------------------------
# Overlap reconnect: hand over to a second socket before the cloud cuts the
# stream (~300 s after connect, socket stays open but silent)
# ---------------------------------------------------------------------------


_CLOSED_MSG = aiohttp.WSMessage(type=aiohttp.WSMsgType.CLOSED, data=None, extra=None)


def _text_frame(time_diff: int = 5, soc: int = 65) -> aiohttp.WSMessage:
    return TestStaleness._make_ws_msg(time_diff=time_diff, soc=soc)


class _QueueSocket:
    """Fake ``ClientWebSocketResponse``: ``receive()`` pops a queue.

    Like aiohttp, ``close()`` wakes a pending ``receive()`` with CLOSED.
    """

    def __init__(self, name: str = "sock") -> None:
        self.name = name
        self.closed = False
        self.sent: list[str] = []
        self._queue: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self) -> aiohttp.WSMessage:
        return await self._queue.get()

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._queue.put_nowait(_CLOSED_MSG)

    def feed(self, msg: aiohttp.WSMessage) -> None:
        self._queue.put_nowait(msg)


async def _until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    """Poll *predicate* until true (bounded)."""
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


class _OverlapHarness:
    """A ``FoxESSRealtimeWS`` wired to fake sockets, constants shrunk."""

    def __init__(
        self,
        *,
        proactive: float = 0.2,
        handover_timeout: float = 1.0,
        stale: float = 0.6,
        backoff: float = 0.05,
        sockets: list[Any] | None = None,
    ) -> None:
        self.loop = asyncio.get_event_loop()
        self.delivered: list[dict[str, object]] = []
        self.delivered_at: list[float] = []
        self.connected_at_delivery: list[bool] = []
        self.on_disconnect = MagicMock()
        self.disconnects_before_stop = 0
        self.deliveries_before_stop = 0
        web_session = AsyncMock()
        web_session.async_ensure_token = AsyncMock(return_value="tok")
        self.ws = FoxESSRealtimeWS(
            "plant1", web_session, self._on_data, self.on_disconnect
        )
        self.ws.PROACTIVE_RECONNECT_AFTER = proactive
        self.ws.HANDOVER_TIMEOUT = handover_timeout
        self.ws.STALE_TIMEOUT = stale
        self.ws.HANDOVER_RETRY_BACKOFF = backoff
        self.ws.RECONNECT_BASE_DELAY = 0.02
        self.opened: list[Any] = list(sockets or [])
        self.open_times: list[float] = []
        self.ws._open_socket = self._open  # type: ignore[method-assign,assignment]

    @property
    def pending(self) -> object:
        """The half-open second socket during a handover."""
        return self.ws._pending_ws

    @property
    def live(self) -> object:
        """The socket the listen loop currently reads."""
        return self.ws._ws

    async def _on_data(self, mapped: dict[str, object]) -> None:
        self.delivered.append(mapped)
        self.delivered_at.append(self.loop.time())
        self.connected_at_delivery.append(self.ws.is_connected)

    async def _open(self, token: str) -> _QueueSocket:
        self.open_times.append(self.loop.time())
        nxt = self.opened.pop(0) if self.opened else _QueueSocket("auto")
        if isinstance(nxt, BaseException):
            raise nxt
        await nxt.send_str("getdata")  # as the real _open_socket does
        return nxt

    async def start(self, live: _QueueSocket) -> None:
        """Connect with *live* as the first socket and start listening."""
        self.opened.insert(0, live)
        self.t_connect = self.loop.time()
        await self.ws.async_connect()

    async def stop(self) -> None:
        # the listen loop reports a disconnect when it ends on teardown;
        # what matters is whether one was reported before that
        self.disconnects_before_stop = self.on_disconnect.call_count
        self.deliveries_before_stop = len(self.delivered)
        await self.ws.async_disconnect()

    @property
    def tasks_left(self) -> list[asyncio.Task[object]]:
        me = asyncio.current_task()
        return [
            t
            for t in asyncio.all_tasks()
            if t is not me
            and not t.done()
            and any(
                n in repr(t.get_coro())
                for n in ("_handover_loop", "_listen_loop", "_first_accepted_frame")
            )
        ]


@pytest.mark.asyncio
class TestOverlapReconnect:
    async def test_defaults(self) -> None:
        assert FoxESSRealtimeWS.PROACTIVE_RECONNECT_AFTER == 270.0
        assert FoxESSRealtimeWS.HANDOVER_TIMEOUT == 15.0

    async def test_handover_fires_at_interval_and_not_before(self) -> None:
        old, new = _QueueSocket("old"), _QueueSocket("new")
        h = _OverlapHarness(proactive=0.3, sockets=[new])
        await h.start(old)
        try:
            for _ in range(5):  # keep the old socket healthy, ~0.1 s total
                old.feed(_text_frame())
                await asyncio.sleep(0.02)
            # the only open so far is the initial connect
            assert len(h.open_times) == 1
            await _until(lambda: len(h.open_times) >= 2)
            assert h.open_times[1] - h.t_connect >= 0.3 - 0.01
            new.feed(_text_frame())
            await _until(lambda: h.live is new)
        finally:
            await h.stop()

    async def test_successful_swap_is_seamless(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level("INFO", logger="custom_components.foxess_control.foxess")
        old, new = _QueueSocket("old"), _QueueSocket("new")
        h = _OverlapHarness(proactive=0.2, sockets=[new])
        await h.start(old)
        connected_at_before = h.ws._connected_at
        try:
            await _until(lambda: len(h.open_times) >= 2)
            # old keeps delivering while the second socket is connecting
            old.feed(_text_frame(soc=61))
            await _until(lambda: any(d.get("SoC") == 61.0 for d in h.delivered))
            # first message on the new socket is the usual old snapshot
            new.feed(_text_frame(time_diff=120, soc=1))
            new.feed(_text_frame(soc=62))
            await _until(lambda: h.live is new)
            await _until(lambda: old.closed)
        finally:
            await h.stop()
        assert h.disconnects_before_stop == 0
        assert all(h.connected_at_delivery[: h.deliveries_before_stop])
        socs = [d.get("SoC") for d in h.delivered]
        assert 61.0 in socs and 62.0 in socs  # nothing dropped
        assert 1.0 not in socs  # the stale snapshot was skipped
        assert h.ws._connected_at > connected_at_before
        assert new.sent == ["getdata"]
        msgs = [(r.levelname, r.getMessage()) for r in caplog.records]
        assert not [m for m in msgs if m[0] == "WARNING"]
        assert any("handover started" in m[1] for m in msgs)
        assert any("handover complete" in m[1] for m in msgs)

    async def test_old_socket_going_quiet_does_not_trigger_reconnect(self) -> None:
        """Newest client wins: the old socket may fall silent at once."""
        old, new = _QueueSocket("old"), _QueueSocket("new")
        h = _OverlapHarness(proactive=0.2, stale=0.4, handover_timeout=2.0)
        h.opened = [new]
        reconnect = AsyncMock()
        h.ws._try_reconnect = reconnect  # type: ignore[method-assign]
        await h.start(old)
        try:
            old.feed(_text_frame())
            await _until(lambda: len(h.open_times) >= 2)
            # old stays silent past STALE_TIMEOUT; new answers late
            await asyncio.sleep(0.6)
            new.feed(_text_frame(soc=70))
            await _until(lambda: h.live is new)
        finally:
            await h.stop()
        reconnect.assert_not_called()
        assert h.disconnects_before_stop == 0

    async def test_old_socket_stale_frames_do_not_trigger_reconnect(self) -> None:
        old, new = _QueueSocket("old"), _QueueSocket("new")
        h = _OverlapHarness(proactive=0.2, stale=0.4, handover_timeout=2.0)
        h.opened = [new]
        reconnect = AsyncMock()
        h.ws._try_reconnect = reconnect  # type: ignore[method-assign]
        await h.start(old)
        try:
            old.feed(_text_frame())
            await _until(lambda: len(h.open_times) >= 2)
            for _ in range(10):  # cloud keepalives for the superseded socket
                old.feed(_text_frame(time_diff=999))
                await asyncio.sleep(0.05)
            new.feed(_text_frame(soc=71))
            await _until(lambda: h.live is new)
        finally:
            await h.stop()
        reconnect.assert_not_called()

    async def test_new_socket_delivering_nothing_is_dropped_and_retried(self) -> None:
        old = _QueueSocket("old")
        dud, good = _QueueSocket("dud"), _QueueSocket("good")
        h = _OverlapHarness(proactive=0.15, handover_timeout=0.2, sockets=[dud, good])
        await h.start(old)
        try:
            await _until(lambda: dud.closed)
            assert h.live is old and not old.closed
            assert h.ws._pending_ws is None
            old.feed(_text_frame())
            await _until(lambda: len(h.open_times) >= 3)  # retried after backoff
            good.feed(_text_frame(soc=72))
            await _until(lambda: h.live is good)
        finally:
            await h.stop()
        assert h.disconnects_before_stop == 0

    async def test_new_socket_connect_failure_keeps_old(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level("INFO", logger="custom_components.foxess_control.foxess")
        old = _QueueSocket("old")
        h = _OverlapHarness(proactive=0.15, sockets=[ConnectionRefusedError("no")])
        await h.start(old)
        try:
            await _until(lambda: len(h.open_times) >= 3)  # failed once, retried
            assert h.live is old and not old.closed
            assert h.ws.is_connected
        finally:
            await h.stop()
        assert h.disconnects_before_stop == 0
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    async def test_old_socket_dying_during_handover_uses_normal_reconnect(
        self,
    ) -> None:
        old = _QueueSocket("old")
        dud = _QueueSocket("dud")
        h = _OverlapHarness(proactive=0.15, handover_timeout=0.3, sockets=[dud])
        reconnect = AsyncMock()

        async def _fake_reconnect() -> None:
            h.ws._connected = False

        reconnect.side_effect = _fake_reconnect
        h.ws._try_reconnect = reconnect  # type: ignore[method-assign]
        await h.start(old)
        try:
            await _until(lambda: len(h.open_times) >= 2)
            await old.close()  # server closes the old socket; new gives nothing
            await _until(lambda: reconnect.await_count == 1)
        finally:
            await h.stop()
        assert dud.closed

    async def test_old_socket_closing_during_handover_swaps_instead(self) -> None:
        old, new = _QueueSocket("old"), _QueueSocket("new")
        h = _OverlapHarness(proactive=0.15, handover_timeout=2.0, sockets=[new])
        reconnect = AsyncMock()
        h.ws._try_reconnect = reconnect  # type: ignore[method-assign]
        await h.start(old)
        try:
            await _until(lambda: len(h.open_times) >= 2)
            await old.close()
            new.feed(_text_frame(soc=73))
            await _until(lambda: h.live is new)
        finally:
            await h.stop()
        reconnect.assert_not_called()
        assert h.disconnects_before_stop == 0

    async def test_async_disconnect_during_handover_leaves_nothing(self) -> None:
        old, half_open = _QueueSocket("old"), _QueueSocket("half")
        h = _OverlapHarness(proactive=0.1, handover_timeout=5.0, sockets=[half_open])
        await h.start(old)
        await _until(lambda: h.pending is half_open)
        await h.ws.async_disconnect()
        assert half_open.closed and old.closed
        assert h.ws._pending_ws is None
        assert h.ws._handover_task is None
        assert h.ws._handover_active is False
        await asyncio.sleep(0.05)
        assert h.tasks_left == []
        assert len(h.open_times) == 2  # no further attempts

    async def test_request_stop_during_handover_aborts_without_swap(self) -> None:
        old, new = _QueueSocket("old"), _QueueSocket("new")
        h = _OverlapHarness(proactive=0.1, handover_timeout=5.0, sockets=[new])
        await h.start(old)
        try:
            await _until(lambda: h.pending is new)
            h.ws.request_stop()  # linger phase: no revival, no handover
            new.feed(_text_frame(soc=74))
            await _until(lambda: new.closed)
            assert h.live is old
            assert 74.0 not in [d.get("SoC") for d in h.delivered]
            await _until(
                lambda: h.ws._handover_task is not None and h.ws._handover_task.done()
            )
        finally:
            await h.stop()
        assert len(h.open_times) == 2

    async def test_stop_event_prevents_handover(self) -> None:
        old = _QueueSocket("old")
        h = _OverlapHarness(proactive=0.1)
        await h.start(old)
        h.ws._stop_event.set()
        await asyncio.sleep(0.25)
        await h.stop()
        assert len(h.open_times) == 1

    async def test_disabled_when_interval_is_zero(self) -> None:
        h = _OverlapHarness(proactive=0)
        await h.start(_QueueSocket("old"))
        try:
            assert h.ws._handover_task is None
        finally:
            await h.stop()

    async def test_tracker_and_reference_survive_the_swap(self) -> None:
        old, new = _QueueSocket("old"), _QueueSocket("new")
        h = _OverlapHarness(proactive=0.15, sockets=[new])
        tracker = h.ws._grid_direction

        def frame(
            *, solar: float = 2.0, grid: float = 5.0, grid_flag: int | None = 1
        ) -> aiohttp.WSMessage:
            msg = _ev_msg(
                solar=solar,
                load=0.5,
                bat_charge=0.0,
                grid=grid,
                charger=0.0,
                grid_status=2,
                grid_flag=grid_flag,
            )
            return aiohttp.WSMessage(
                type=aiohttp.WSMsgType.TEXT, data=json.dumps(msg), extra=None
            )

        await h.start(old)
        old.feed(frame())
        await _until(lambda: len(h.delivered) == 1)
        assert tracker.importing is True
        learned = dict(tracker._status_meaning)
        assert learned  # gridStatus 2 learned as import from the signed flag
        last_useful = h.ws._last_useful_data
        try:
            await _until(lambda: len(h.open_times) >= 2)
            # no signed flag and an unusable balance (0.0 kW predicted): only
            # the learned gridStatus=2 -> import meaning can decide.  A fresh
            # tracker would fall back to the legacy rule and say export.
            new.feed(frame(grid_flag=None, solar=0.5, grid=0.6))
            await _until(lambda: h.live is new)
        finally:
            await h.stop()
        assert h.ws._grid_direction is tracker
        assert tracker._status_meaning == learned
        assert h.ws._last_useful_data >= last_useful
        assert h.delivered[-1]["gridConsumptionPower"] == pytest.approx(0.6)


class _CutoffCloud:
    """Fake cloud: each socket delivers until ``cut_after`` s, then goes silent.

    Newest socket gets real frames; older ones get ``timeDiff: 999`` keepalives
    (the simulator's newest-wins rule).
    """

    def __init__(self, cut_after: float, interval: float = 0.02) -> None:
        self.cut_after = cut_after
        self.interval = interval
        self.sockets: list[_QueueSocket] = []
        self._tasks: list[asyncio.Task[None]] = []

    async def open(self, token: str) -> _QueueSocket:
        sock = _QueueSocket(f"cloud{len(self.sockets)}")
        self.sockets.append(sock)
        self._tasks.append(asyncio.ensure_future(self._feed(sock)))
        return sock

    async def _feed(self, sock: _QueueSocket) -> None:
        loop = asyncio.get_event_loop()
        born = loop.time()
        # first frame after connect is an old snapshot, as on the real cloud
        sock.feed(_text_frame(time_diff=61, soc=1))
        while not sock.closed and loop.time() - born < self.cut_after:
            await asyncio.sleep(self.interval)
            newest = sock is self.sockets[-1]
            sock.feed(_text_frame(soc=65) if newest else _text_frame(time_diff=999))

    async def aclose(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


def _max_gap(times: list[float], start: float, end: float) -> float:
    pts = [start, *[t for t in times if start <= t <= end], end]
    return max(b - a for a, b in itertools.pairwise(pts))


@pytest.mark.asyncio
class TestStreamCutAtFixedAge:
    """The cloud cuts every connection at a fixed age; the gap must vanish."""

    RUN_FOR = 2.0
    CUT = 0.6

    async def _run(self, proactive: float) -> tuple[_OverlapHarness, float, float]:
        cloud = _CutoffCloud(cut_after=self.CUT)
        h = _OverlapHarness(proactive=proactive, stale=0.5, handover_timeout=1.0)
        h.ws._open_socket = cloud.open  # type: ignore[method-assign,assignment]
        t0 = h.loop.time()
        try:
            await h.ws.async_connect()
            # first frame is a stale snapshot; wait for real data, then run
            await _until(lambda: len(h.delivered) >= 1)
            start = h.loop.time()
            await asyncio.sleep(self.RUN_FOR)
            end = h.loop.time()
        finally:
            await h.stop()
            await cloud.aclose()
        assert t0 < start
        return h, start, end

    async def test_no_gap_with_overlap_reconnect(self) -> None:
        h, start, end = await self._run(proactive=0.35)
        assert h.disconnects_before_stop == 0
        assert all(h.connected_at_delivery[: h.deliveries_before_stop])
        assert _max_gap(h.delivered_at, start, end) < 0.25

    async def test_baseline_without_overlap_has_the_old_gap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "custom_components.foxess_control.foxess.realtime_ws.random.uniform",
            lambda a, b: 0.0,
        )
        h, start, end = await self._run(proactive=0)
        assert h.disconnects_before_stop >= 1  # badge flipped to api
        assert _max_gap(h.delivered_at, start, end) >= 0.5  # >= STALE_TIMEOUT


@pytest.mark.asyncio
async def test_overlap_reconnect_against_the_simulator(
    foxess_sim: SimulatorHandle,
) -> None:
    """Real sockets, real aiohttp close semantics, newest-wins + stream cutoff."""
    foxess_sim.set(
        fuzzing=False, ws_push_interval_s=0.05, ws_stream_cutoff_s=1.5, soc=65
    )
    loop = asyncio.get_event_loop()
    times: list[float] = []

    async def on_data(mapped: dict[str, object]) -> None:
        times.append(loop.time())

    on_disconnect = MagicMock()
    web_session = AsyncMock()
    web_session.async_ensure_token = AsyncMock(return_value="tok")
    ws = FoxESSRealtimeWS(
        "plant1",
        web_session,
        on_data,
        on_disconnect,
        ws_url=foxess_sim.url.replace("http://", "ws://") + "/dew/v0/wsmaitian",
    )
    ws.PROACTIVE_RECONNECT_AFTER = 0.6
    ws.HANDOVER_TIMEOUT = 3.0
    ws.STALE_TIMEOUT = 1.0
    swaps = 0
    original = ws._handover_once

    async def counting() -> bool:
        nonlocal swaps
        ok = await original()
        swaps += ok
        return ok

    ws._handover_once = counting  # type: ignore[method-assign]
    try:
        await ws.async_connect()
        await _until(lambda: len(times) >= 1)
        start = loop.time()
        await _until(lambda: swaps >= 3, timeout=30.0)
        end = loop.time()
    finally:
        disconnects_before_stop = on_disconnect.call_count
        await ws.async_disconnect()
    assert disconnects_before_stop == 0
    assert _max_gap(times, start, end) < 0.9
