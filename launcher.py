import streamlit as st
import subprocess
import shlex

from src.experiment.experiment import EXPERIMENT_PRESETS
from src.system import SYSTEM_REGISTRY


def resolve_system_field(preset_name):
    """Walk base chain to find the system field."""
    seen = set()
    name = preset_name
    while name and name not in seen:
        seen.add(name)
        p = EXPERIMENT_PRESETS.get(name, {})
        if "system" in p:
            return p["system"]
        name = p.get("base")
    return None


st.set_page_config(page_title="Pencil Balancer Launcher")
st.title("Pencil Balancer Launcher")

exp_preset = st.selectbox("Experiment preset", sorted(EXPERIMENT_PRESETS.keys()))

# Derive compatible system presets from the experiment preset's system type
default_system = resolve_system_field(exp_preset) or ""
sys_type, _, sys_preset_default = default_system.partition(":")

system_options = [""] + sorted(
    f"{sys_type}:{name}"
    for name in SYSTEM_REGISTRY[sys_type].Presets
) if sys_type in SYSTEM_REGISTRY else [""]

system = st.selectbox("System preset", system_options)

cmd = [".venv/bin/python", "-m", "main", "-p", exp_preset]
if system:
    cmd += ["--system", system]

st.code(" ".join(shlex.quote(x) for x in cmd), language="bash")

if st.button("Run"):
    with st.spinner("Running..."):
        result = subprocess.run(cmd, cwd=".", text=True, capture_output=True)
    st.subheader("stdout")
    st.text(result.stdout or "[no output]")
    st.subheader("stderr")
    st.text(result.stderr or "[no errors]")
