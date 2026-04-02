from __future__ import annotations

import json
from typing import Any

import streamlit as st

from UI.backend_adapter import resolve_ui_config, run_from_ui
from UI.preset_capabilities import PRESET_CAPABILITIES
from UI.state import PRESET_ORDER, REAL_PRESETS, get_capability_preset, reset_selections_for_preset


def _set_active_preset(preset: str) -> None:
    st.session_state.selected_preset = preset
    if preset == "other":
        cap_preset = st.session_state.other_base_preset
    else:
        cap_preset = preset
    st.session_state.selections = reset_selections_for_preset(cap_preset)


def _render_preset_row() -> None:
    st.subheader("Preset")
    cols = st.columns(len(PRESET_ORDER))
    for col, preset in zip(cols, PRESET_ORDER):
        label = preset.replace("_", " ").title()
        if col.button(label, use_container_width=True):
            _set_active_preset(preset)


def _render_dynamic_options(cap_preset: str) -> dict[str, Any]:
    cap = PRESET_CAPABILITIES[cap_preset]
    allowed = cap.get("allowed", {})
    defaults = cap.get("defaults", {})
    selections = st.session_state.selections

    st.subheader("Options")
    for key in ["experiment", "actuator", "controller", "estimator"]:
        if key not in allowed:
            if key in defaults:
                st.caption(f"{key}: `{defaults[key]}` (preset default)")
            continue

        options = allowed[key]
        current = selections.get(key, defaults.get(key))
        if current not in options:
            current = options[0]

        if len(options) == 1:
            selections[key] = options[0]
            st.write(f"**{key}**: `{options[0]}`")
            continue

        idx = options.index(current)
        picked = st.radio(
            key.replace("_", " ").title(),
            options=options,
            index=idx,
            horizontal=True,
            key=f"field_{key}",
        )
        selections[key] = picked

    controller = selections.get("controller", defaults.get("controller"))
    if controller == "circle":
        radius = float(selections.get("radius", 0.1))
        selections["radius"] = st.number_input(
            "Circle Radius (m)",
            min_value=0.01,
            max_value=0.5,
            value=radius,
            step=0.005,
            format="%.3f",
        )
    else:
        selections.pop("radius", None)

    return selections


def _render_other_overrides(selected_preset: str) -> tuple[dict[str, Any], str | None]:
    if selected_preset != "other":
        return {}, None

    st.subheader("Other: Dev Overrides")
    raw = st.text_area(
        "JSON overrides (dot paths)",
        value=st.session_state.override_text,
        height=180,
        help='Example: {"params.run.dt": 0.002, "params.hardware.dvs_hough.mixing_factor": 0.03}',
    )
    st.session_state.override_text = raw

    if not raw.strip():
        return {}, None
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}, "Overrides JSON must be an object/dict."
        return parsed, None
    except json.JSONDecodeError as exc:
        return {}, f"Invalid JSON: {exc}"


def main() -> None:
    st.set_page_config(page_title="Experiment Launcher", layout="wide")
    st.title("Local Experiment Launcher")
    st.caption("Preset-driven controls with optional dev overrides.")

    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = "sim"
    if "other_base_preset" not in st.session_state:
        st.session_state.other_base_preset = "sim"
    if "selections" not in st.session_state:
        st.session_state.selections = reset_selections_for_preset("sim")
    if "override_text" not in st.session_state:
        st.session_state.override_text = "{}"

    _render_preset_row()

    selected_preset = st.session_state.selected_preset
    if selected_preset == "other":
        st.subheader("Other Base Preset")
        picked_base = st.selectbox(
            "Choose base preset",
            REAL_PRESETS,
            index=REAL_PRESETS.index(st.session_state.other_base_preset),
        )
        if picked_base != st.session_state.other_base_preset:
            st.session_state.other_base_preset = picked_base
            st.session_state.selections = reset_selections_for_preset(picked_base)

    cap_preset = get_capability_preset(selected_preset, st.session_state.other_base_preset)
    selections = _render_dynamic_options(cap_preset)

    resolved = resolve_ui_config(cap_preset, selections)
    dev_overrides, dev_error = _render_other_overrides(selected_preset)

    st.subheader("Resolved Config Preview")
    st.json(
        {
            "selected_preset": selected_preset,
            "effective_preset": cap_preset,
            "resolved": resolved,
            "dev_overrides": dev_overrides if selected_preset == "other" else {},
        }
    )

    if dev_error:
        st.error(dev_error)

    st.divider()
    if st.button("Launch", type="primary", use_container_width=True):
        if dev_error:
            st.error("Fix override JSON before launching.")
            return
        with st.spinner("Running experiment..."):
            outcome = run_from_ui(
                preset=cap_preset,
                resolved=resolved,
                dev_overrides=dev_overrides if selected_preset == "other" else None,
            )
        if outcome.get("ok"):
            st.success("Run completed.")
        else:
            st.error(outcome.get("error", "Unknown error"))
        if outcome.get("logs"):
            st.subheader("Run Logs")
            st.code(outcome["logs"])


if __name__ == "__main__":
    main()
