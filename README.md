# piper_ros

ROS2 driver for AgileX Piper arms using the [R2
piper_control](https://github.com/Reimagine-Robotics/piper_control) interface

## Installation

### pixi Install (dev and testing)

Make sure you have [pixi](https://pixi.sh/latest/#installation) installed.

```bash
pixi install
```

### Manual Install

  1.  Install `can-utils` and `ethtool`:

      ```bash
      sudo apt install -y can-utils ethtool
      ```

  2.  Install `piper_control`:

      ```bash
        pip install "piper_control @ git+https://github.com/Reimagine-Robotics/piper_control.git@main"
      ```

  3.  Install `piper_ros` to your system or active virtual/conda environment:

      ```bash
      pip install .
      ```

### CAN udev Rule Generator

Automatically rename CAN interfaces and set bitrate on plug-in.

#### Usage

1. Plug in your CAN adapter
2. Run the script:
   ```bash
   sudo ./scripts/generate_udev_rule.bash -i can0 -b 1000000
   # Or
   sudo ./scripts/generate_udev_rule.bash -i can0 -n myrobot -b 1000000
   ```
3. Unplug and replug the adapter to test

#### Test

```bash
ip link show myrobot
```

That's it!
