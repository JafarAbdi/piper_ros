import mujoco
import copy
import loop_rate_limiters
import mujoco.viewer
import numpy as np
import time
from robot_descriptions import piper_mj_description


def main() -> None:
    model = mujoco.MjModel.from_xml_path(piper_mj_description.PACKAGE_PATH + "/scene.xml")
    data = mujoco.MjData(model)
    gravity_compenstation_data = copy.deepcopy(data)

    rate = loop_rate_limiters.RateLimiter(200)
    # Override the simulation timestep.
    model.opt.timestep = rate.dt

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Reset the simulation to the initial keyframe.
        mujoco.mj_resetData(model, data)
        mujoco.mj_resetData(model, gravity_compenstation_data)

        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while viewer.is_running():
            # data.ctrl[actuator_ids] = q[dof_ids]
            gravity_compenstation_data.qpos = data.qpos
            mujoco.mj_forward(model, gravity_compenstation_data)
            data.qfrc_applied = gravity_compenstation_data.qfrc_bias

            # Step the simulation.
            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
