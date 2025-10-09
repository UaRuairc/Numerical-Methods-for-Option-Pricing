import streamlit as st

from src.config.config_management import ConfigManager
from collections import defaultdict


def set_defaults():
    """set session state variables defaults"""

    default_parameters = {}

    for parameter, default in default_parameters.items():
        st.session_state.setdefault(parameter, default)

    set_default_config()

    return

def update_duration_box_on_change():
    choice_ = st.session_state["config"]["segmented_control"]["duration"]["value"]
    if choice_ is not None and choice_ != 3:
        st.session_state["config"]["number_input_box"]["duration"]["value"] = 30 * (2 ** choice_)

    if choice_ is None:
        st.session_state["config"]["number_input_box"]["duration"]["value"] = None


def set_default_config(suppress=True):
    if "suppress" not in st.session_state:
        st.session_state["suppress"] = suppress

    if "config" in st.session_state:
        #print("default config already set.")
        return

    def tree():
        return defaultdict(tree)

    config = tree()
    st.session_state["config"] = config

    operators = ["add", "subtract", "mult", "div"]
    widget_labels = {
        "checkbox": {
        },
        "slider": {
        },
        "number_input_box": {
        },
        "custom_input_box": {
        },
        "segmented_control": {
        }
    }

    segmented_control_options = {
    }

    default_duration = 2

    widget_overrides = {
        "checkbox": {
        },


        "segmented_control": {
        },

        "slider": {
            },

        "number_input_box": {
        },
    }

    for widget_category, widgets in widget_labels.items():

        for widget_name, widget_label in widgets.items():
            overrides = (widget_overrides.get(widget_category, {}).get(widget_name, {}))

            ConfigManager.add_widget(
                name = widget_name,
                widget_category=widget_category,
                **overrides
            )
    if not st.session_state["suppress"]:
        print("Initialisation complete. To suppresses these startup messages, call set_default_config(suppress=True) in `config_management.py` instead.")