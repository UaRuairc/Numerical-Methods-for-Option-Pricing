import streamlit as st
import inspect
from src.ui.widgets import MakeWidget

CALLABLES = {
    "streamlit":{
        "checkbox": st.checkbox,
        "slider": st.slider,
        "number_input_box": st.number_input,
        "segmented_control": st.segmented_control,
        "text_input_boxes": st.text_input,
        "buttons": st.button,
        },
    "custom": {
    },
}

class ConfigManager:

    @staticmethod
    def generate_config(name: str, widget_category: str, **overrides):

        base_config = ConfigManager.base_config(widget_category=widget_category, framework="streamlit")

        base_config_defaults = {
            "label": f"{name}",
            "key": f"{name}_{widget_category}",
        }

        base_config.update(base_config_defaults)


        additional_config = {
            "widget_category": widget_category,
            "name": name,
            "on_change": None,
            "extra_callback": None
        }

        config = {}
        config.update(base_config)
        config.update(additional_config)




        # some streamlit widgets do not have a `value` parameter
        #
        # MakeWidget generically uses the value param to store info on how to rebuild the widget, whether the widget has
        # a `value` parameter or not. MakeWidget then updates the config so "value" is renamed to whatever key that widget uses
        if "value" not in config:
            config["value"] = None
        if widget_category == "number_input_box":
            config["value"] = 0
        # Apply overrides

        config.update(overrides)
        return config

    @staticmethod
    def base_config(widget_category: str, framework: str="streamlit"):

        if st.session_state["callables"].get(framework, None) is None:
            raise NotImplementedError(
                f"Framework {framework} is not supported. Currently, only streamlit is supported.")

        if st.session_state["callables"][framework].get(widget_category, None) is None:
            raise NotImplementedError(
                f"Widget category {widget_category} is not supported for framework {framework}.")

        widget_callable = CALLABLES[framework][widget_category]
        sig = inspect.signature(widget_callable)

        base_config = {}
        for param_name, param in sig.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if param.default is not inspect.Parameter.empty:
                base_config[param_name] = param.default
                continue

            # doesn't have a default, needs to be set, assume user provides overrides.
            base_config[param_name] = None

        return base_config




    @staticmethod
    def add_to_session_state(config):
        config_copy = config.copy()

        name = config["name"]
        config_copy.pop("name")

        category = config["widget_category"]
        st.session_state["config"][category][name] = config_copy
        if st.session_state["suppress"] == False:
            print("\n Added config to session state: {}\n".format(st.session_state["config"][category][name] ))

    @staticmethod
    def validate_config(widget_key, widget_category):
        MakeWidget(widget_key, widget_category)
        # work in progress could just use MakeWidget give an error when invalid

    @staticmethod
    def get_widget_config(widget_key, widget_category):
        if ConfigManager.is_widget_configured(widget_key, widget_category):
            return st.session_state["config"][widget_category][widget_key]
        else:
            print(f"The widget {widget_key} of category {widget_category} does not exist. Please add it before trying to get it.")
            return None

    @staticmethod
    def add_widget(name: str, widget_category: str, **overrides):

        if ConfigManager.is_widget_configured(name, widget_category):
            print(f"A {widget_category} widget with this name already exists. Choose a different name.")
            return

        config = ConfigManager.generate_config(name, widget_category, **overrides)

        ConfigManager.add_to_session_state(config)


    @staticmethod
    def remove_widget(name: str, widget_category: str):
        if widget_category not in st.session_state["config"]:
            print(f"The widget category {widget_category} is already empty. Is {name} in a different category?")
            return

        if name not in st.session_state["config"][widget_category]:
            st.session_state["config"][widget_category].pop(name)

    @staticmethod
    def get_widget_value(widget_name, widget_category, arg="value"):
        """ Get the value of a widget argument from the session state.
            By default, it returns the value arg, but can return any argument of the widget config."""
        return st.session_state["config"][widget_category][widget_name][arg]

    @staticmethod
    def add_widget_arg(widget_name, widget_category, new_arg, new_arg_value):
        """
        Add a new argument to the widget config in session state.
        This is used to add new arguments to existing widgets.
        """
        if not ConfigManager.is_widget_configured(widget_name, widget_category):
            print(f"The widget {widget_name} of category {widget_category} does not exist. Please add it before trying to add an argument.")
            return
        else:
            widget_config = ConfigManager.get_widget_config(widget_name, widget_category)

        if new_arg in widget_config:
            print("That argument already exists. If you want to update it, use the update_widget_arg method.")
            return

        widget_config[new_arg] = new_arg_value

        return

    @staticmethod
    def update_widget_arg(widget_name: str,
                          widget_category: str,
                          updated_arg_value,
                          arg: str="value",
                          ):
        """
        Update arguments to existing widget configs.
        """
        if ConfigManager.is_widget_configured(widget_name, widget_category):
            widget_config = st.session_state["config"][widget_category][widget_name]
        else:
            print(f"The widget {widget_name} of category {widget_category} does not exist. Please add it before trying to update it.")
            return False

        if arg not in widget_config:
            print("That argument does not exist. If you want to add it, use the add_new_widget_arg method.")
            return False

        if type(updated_arg_value) != type(widget_config[arg]) and widget_config[arg] is not None:
            print(f"WARNING WHEN UPDATING {widget_category}|{widget_name}: {arg} is currently of type {type(widget_config[arg])}, but the updated value is of type {type(updated_arg_value)}.")
            print("updated the value anyway, but this may cause issues.")


        if arg == "extra_callback":
            if updated_arg_value is not None and not callable(updated_arg_value):
                print(f"extra_callback must be a callable, got {type(updated_arg_value)} instead.")
                return False


        widget_config[arg] = updated_arg_value
        print(f"Updated {widget_category}|{widget_name}:"
              f"set arg {arg} to: {updated_arg_value}")



        return True

    @staticmethod
    def is_widget_configured(widget_name, widget_category):
        """
        Check if a widget exists in the session state.
        """
        if widget_category not in st.session_state["config"] or widget_name not in st.session_state["config"][
            widget_category]:
            #print(f"The widget {widget_name} of category {widget_category} does not exist.")
            return False
        else:
            return True
