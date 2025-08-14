#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
import mujoco as mj
import numpy as np
from piper_control import piper_connect
from piper_control import piper_init
from piper_control import piper_interface
from piper_control import piper_control

from nicegui import ui

SCRIPT_DIR = Path(__file__).parent.resolve()
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
MJ_MODEL_PATH = (
    SCRIPT_DIR
    / "src"
    / "piper_control_ros2"
    / "piper_control_ros2"
    / "teach_mode"
    / "piper_grav_comp.xml"
)


class JointSliderGUI:
    def __init__(self):
        self.model: mj.MjModel = mj.MjModel.from_xml_path(MJ_MODEL_PATH.as_posix())
        self.data = mj.MjData(self.model)
        self.joint_indices = np.array(
            [self.model.joint(name).id for name in JOINT_NAMES]
        )
        self.joint_limits = self.model.jnt_range[self.joint_indices]
        self.joint_positions = {name: 0.0 for name in JOINT_NAMES}
        self.home_positions = np.sum(self.joint_limits, axis=1) / 2.0
        self.robot = piper_interface.PiperInterface(can_port="can0")

        # Store slider/input references for programmatic updates
        self.sliders = {}
        self.inputs = {}

        # Create the GUI
        self.create_gui()

    def on_input_change(self, joint_name, value_str):
        """Handle input box value changes."""
        try:
            value = float(value_str)
            limits = self.joint_limits[joint_name]
            # Clamp value to joint limits
            value = max(limits["lower"], min(limits["upper"], value))

            self.joint_positions[joint_name] = value
            self.sliders[joint_name].value = value
            self.publish_joint_positions()
        except ValueError:
            pass  # Ignore invalid input

    def on_slider_change(self, joint_name, value):
        """Handle slider value changes."""
        self.inputs[joint_name].value = f"{value:.2f}"  # Update input box
        self.joint_positions[joint_name] = value
        # self.publish_joint_positions()

    def create_gui(self):
        with ui.column():
            with ui.row():
                with ui.column(), ui.card():
                    # Joints for this controller
                    for joint_index, joint_name in enumerate(JOINT_NAMES):
                        limits = self.joint_limits[joint_index]
                        with ui.card():
                            # Joint name
                            ui.label(joint_name)

                            # Slider with min/max labels
                            with ui.row().style(
                                "display: flex; align-items: center; width: 100%;",
                            ):
                                joint_limits = self.joint_limits[joint_index]
                                lower, upper = joint_limits[0], joint_limits[1]
                                ui.label(f"{lower:.2f}").style(
                                    "width: 25px; text-align: right; font-size: 12px; color: #888; flex-shrink: 0;",
                                )
                                slider = ui.slider(
                                    min=lower,
                                    max=upper,
                                    value=0.0,
                                    step=0.01,
                                    on_change=lambda e, name=joint_name: self.on_slider_change(
                                        name,
                                        e.value,
                                    ),
                                ).style(
                                    "flex: 1; margin: 0 10px; min-width: 100px;",
                                )
                                ui.label(f"{upper:.2f}").style(
                                    "width: 25px; text-align: left; font-size: 12px; color: #888; flex-shrink: 0;",
                                )
                                ui_input = ui.input(
                                    value="0.00",
                                    on_change=lambda e, name=joint_name: self.on_input_change(
                                        name,
                                        e.value,
                                    ),
                                ).style("width: 40px; font-size: 12px;")

                            # Store slider reference
                            self.sliders[joint_name] = slider
                            self.inputs[joint_name] = ui_input


def main():
    try:
        gui_node = JointSliderGUI()

        ui.run(
            title="Robot Joint Control",
            port=8080,
            host="0.0.0.0",  # noqa: S104
            reload=False,
            uvicorn_reload_dirs=str(SCRIPT_DIR),
            show=True,
            favicon="🤖",
        )

    except KeyboardInterrupt:
        LOGGER.info("Shutting down...")
    except Exception:
        LOGGER.exception("Error while running GUI")
    finally:
        LOGGER.info("Exiting...")


if __name__ == "__main__":
    main()
