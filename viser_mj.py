# Based on https://github.com/bdaiinstitute/judo/tree/8d97cf588a8beab9ad9132ab4f6c0bf51e065fc5/judo/visualizers
import more_itertools
from scipy import optimize
import collections
import numpy as np
import loop_rate_limiters
from piper_control import piper_connect
from piper_control import piper_init
from piper_control import piper_interface
from piper_control import piper_control
from piper_control_ros2.teach_mode.run_gather_data import GravityTorqueSampler
from piper_control_ros2.teach_mode.pid_controller import PIDController

import time
import mujoco as mj
from pathlib import Path
from typing import Any, List, Tuple

import mujoco
import numpy as np
import trimesh
from mujoco import MjData, MjsMaterial, MjSpec
from trimesh.creation import box, capsule, cylinder, icosphere
from trimesh.transformations import scale_and_translate
from trimesh.visual import ColorVisuals, TextureVisuals
from trimesh.visual.material import PBRMaterial
import viser
import viser.uplot
from viser import (
    ClientHandle,
    LineSegmentsHandle,
    SceneNodeHandle,
    SplineCatmullRomHandle,
    ViserServer,
)

import warnings
from pathlib import Path
from typing import List

import mujoco
import numpy as np
import trimesh
from mujoco import MjModel, MjsGeom, MjsMaterial, MjSpec
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial


def get_sensor_name(model: MjModel, sensorid: int) -> str:
    """Return name of the sensor with given ID from MjModel."""
    index = model.name_sensoradr[sensorid]
    end = model.names.find(b"\x00", index)
    name = model.names[index:end].decode("utf-8")
    if len(name) == 0:
        name = f"sensor{sensorid}"
    return name


def get_mesh_data(model: MjModel, meshid: int) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve the vertices and faces of a specified mesh from a MuJoCo model.

    Args:
        model : MjModel The MuJoCo model containing the mesh data.
        meshid : int The index of the mesh to retrieve.

    Result:
        tuple[np.ndarray, np.ndarray]
        Vertices (N, 3) and faces (M, 3) of the mesh.
    """
    vertadr = model.mesh_vertadr[meshid]
    vertnum = model.mesh_vertnum[meshid]
    vertices = model.mesh_vert[vertadr : vertadr + vertnum, :]

    faceadr = model.mesh_faceadr[meshid]
    facenum = model.mesh_facenum[meshid]
    faces = model.mesh_face[faceadr : faceadr + facenum]
    return vertices, faces


def get_mesh_file(spec: MjSpec, geom: MjsGeom) -> Path:
    """Extracts the mesh filepath for a particular geom from an MjSpec."""
    assert (
        geom.type == mujoco.mjtGeom.mjGEOM_MESH
    ), f"Can only get mesh files for meshes, got type {geom.type}"

    meshname = geom.meshname
    mesh = spec.mesh(meshname)

    mesh_path = Path(spec.modelfiledir) / spec.meshdir / mesh.file
    return mesh_path


def get_mesh_scale(spec: MjSpec, geom: MjsGeom) -> np.ndarray:
    """Extracts the relevant scale parameters for a given geom in the MjSpec."""
    assert (
        geom.type == mujoco.mjtGeom.mjGEOM_MESH
    ), f"Can only get mesh scale for mesh-type geoms, got type {geom.type}."

    meshname = geom.meshname
    mesh = spec.mesh(meshname)

    return mesh.scale  # type: ignore


def apply_mujoco_material(
    mesh: trimesh.Trimesh,
    material: MjsMaterial,
) -> None:
    """Applies a MuJoCo material to a trimesh mesh.

    This sets up PBR parameters and handles RGBA conversion.

    Args:
        mesh: the trimesh.Trimesh to modify
        model: the Mujoco MjModel to read textures (spec.texturedir if available)
        material: an object matching the mjsMaterial struct
        texture_dir: optional override of the directory for texture files
    """
    # prepare PBR material
    pbr = PBRMaterial()

    # get RGBA, convert if needed
    rgba = np.array(material.rgba)
    if np.issubdtype(rgba.dtype, np.floating):
        rgba = rgba_float_to_int(rgba)
    color = tuple(int(x) for x in rgba.tolist())
    pbr.alphaMode = "BLEND" if rgba[3] < 255 else "OPAQUE"

    # set PBR values
    pbr.metallicFactor = float(material.metallic)
    pbr.roughnessFactor = float(material.roughness)
    pbr.emissiveFactor = [material.emission] * 3
    if material.roughness == 0.0 and getattr(material, "shininess", 0) > 0:
        pbr.roughnessFactor = np.sqrt(2.0 / (material.shininess + 2.0)).item()

    dummy = Image.new("RGBA", (1, 1), color)
    pbr.baseColorTexture = dummy
    uv = getattr(mesh.visual, "uv", None)
    mesh.visual = TextureVisuals(material=pbr, uv=uv)

    if getattr(material, "textures", None):
        warnings.warn(
            "Textured meshes are currently unsupported. Loading with RGBA color instead.",
            stacklevel=2,
        )

    mesh.visual = TextureVisuals(material=pbr, uv=None)


def is_trace_sensor(model: MjModel, sensorid: int) -> bool:
    """Check if a sensor is a trace sensor."""
    sensor_name = get_sensor_name(model, sensorid)
    return (
        model.sensor_type[sensorid] == mujoco.mjtSensor.mjSENS_FRAMEPOS
        and model.sensor_datatype[sensorid] == mujoco.mjtDataType.mjDATATYPE_REAL
        and model.sensor_dim[sensorid] == 3
        and "trace" in sensor_name
    )


def count_trace_sensors(model: MjModel) -> int:
    """Count the number of trace sensors of a given mujoco model."""
    num_traces = 0
    for id in range(model.nsensor):
        num_traces += is_trace_sensor(model, id)
    return num_traces


def get_trace_sensors(model: MjModel) -> List[int]:
    """Get the IDs of all trace sensors in a given mujoco model."""
    return [id for id in range(model.nsensor) if is_trace_sensor(model, id)]


def rgba_float_to_int(rgba_float: np.ndarray) -> np.ndarray:
    """Convert RGBA float values in [0, 1] to int values in [0, 255]."""
    return (255 * rgba_float).astype("int")


def rgba_int_to_float(rgba_int: np.ndarray) -> np.ndarray:
    """Convert RGBA int values in [0, 255] to float values in [0, 1]."""
    return rgba_int / 255.0


DEFAULT_GRID_SECTION_COLOR = (0.02, 0.14, 0.44)
DEFAULT_GRID_CELL_COLOR = (0.27, 0.55, 1)
DEFAULT_SPHERE_SUBDIVISIONS = 3
DEFAULT_SPLINE_COLOR = (0.8, 0.1, 0.8)
DEFAULT_BEST_SPLINE_COLOR = (0.96, 0.7, 0.0)


class ViserMjModel:
    """Helper for rendering MJCF models in viser.

    Args:
        target: ViserServer or ClientHandle to add MjModel to.
        spec: MjSpec of the model to be visualized.
        show_ground_plane: optional flag to show the default ground plane.
        geom_exclude_substring: optional string to exclude a geom from visualization.
    """

    def __init__(
        self,
        target: ViserServer | ClientHandle,
        spec: MjSpec,
        show_ground_plane: bool = True,
        geom_exclude_substring: str = "",
    ) -> None:
        """Constructor for ViserMjModel."""
        self._target = target
        self._spec = spec

        # give default names to any unnamed geoms and bodies
        _geom_placeholder_idx = 0
        _body_placeholder_idx = 0
        for body in self._spec.bodies[1:]:
            # Sharp edge: not using the tree structure of the kinematics ...
            body_name = body.name
            if not body_name:
                body_name = f"JUDO_BODY_{_body_placeholder_idx}"
                body.name = body_name
                _body_placeholder_idx += 1

        for geom in self._spec.geoms:
            geom_name = geom.name
            if not geom_name:  # if geom has no name, use a placeholder.
                geom_name = f"JUDO_GEOM_{_geom_placeholder_idx}"
                geom.name = geom_name
                _geom_placeholder_idx += 1

        self._model = spec.compile()

        # Assume first body is root of kinematic tree.
        self._bodies = [
            self._target.scene.add_frame(self._spec.bodies[0].name, show_axes=False)
        ]
        self._geoms: List = []

        # Show world plane if desired.
        if show_ground_plane:
            self._geoms.append(add_plane(self._target, "ground_plane"))

        # Add coordinate frame for each non-world body in model.
        for body in self._spec.bodies[1:]:
            # Sharp edge: not using the tree structure of the kinematics ...
            body_name = body.name
            self._bodies.append(
                self._target.scene.add_frame(body_name, show_axes=False)
            )

            for geom in body.geoms:
                # classname is mj.MjsDefault
                if geom.classname.name == "collision":
                    # Skip collision geoms.
                    continue
                geom_name = f"{body_name}/geom_{geom.name}"
                if geom_exclude_substring and geom_exclude_substring in geom_name:
                    continue
                self.add_geom(geom_name, geom)

        # Add traces
        self._num_trace_sensors = count_trace_sensors(self._model)
        self.all_traces_rollout_size = 0
        self.add_traces()

    def add_geom(self, geom_name: str, geom: Any) -> None:
        """Helper function for adding geoms to scene tree."""
        # Store compiled model geom info (handles things like fromto).
        model_geom = self._model.geom(geom.name)
        match geom.type:
            case mujoco.mjtGeom.mjGEOM_PLANE:
                # TODO(pculbert): support more color options.
                self._geoms.append(
                    add_plane(
                        self._target,
                        geom_name,
                        pos=geom.pos,
                        quat=geom.quat,
                    )
                )
            case mujoco.mjtGeom.mjGEOM_HFIELD:
                # TODO(pculbert): Implement HField viz as collection of boxes (?).
                raise NotImplementedError("HField is not implemented.")
            case mujoco.mjtGeom.mjGEOM_SPHERE:
                self._geoms.append(
                    add_sphere(
                        self._target,
                        geom_name,
                        radius=model_geom.size[0],
                        pos=geom.pos,
                        quat=geom.quat,
                        rgba=model_geom.rgba,
                    )
                )
            case mujoco.mjtGeom.mjGEOM_CAPSULE:
                self._geoms.append(
                    add_capsule(
                        self._target,
                        geom_name,
                        radius=model_geom.size[0],
                        length=2 * model_geom.size[1],  # MJC has capsule half-lengths.
                        pos=model_geom.pos,
                        quat=model_geom.quat,
                        rgba=model_geom.rgba,
                    )
                )
            case mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                raise NotImplementedError
            case mujoco.mjtGeom.mjGEOM_CYLINDER:
                self._geoms.append(
                    add_cylinder(
                        self._target,
                        geom_name,
                        radius=model_geom.size[0],
                        height=2 * model_geom.size[1],
                        pos=model_geom.pos,
                        quat=model_geom.quat,
                        rgba=model_geom.rgba,
                    )
                )
            case mujoco.mjtGeom.mjGEOM_BOX:
                self._geoms.append(
                    add_box(
                        self._target,
                        geom_name,
                        size=2 * model_geom.size,  # MJC has box half-lengths.
                        pos=model_geom.pos,
                        quat=model_geom.quat,
                        rgba=model_geom.rgba,
                    )
                )
            case mujoco.mjtGeom.mjGEOM_MESH:
                # Get necessary mesh properties.
                mesh_file = get_mesh_file(self._spec, geom)
                mesh_scale = get_mesh_scale(self._spec, geom)

                # Introspect on texture.
                mjs_material = self._spec.material(geom.material)

                # Call the new, robust function to add the mesh.
                handle = add_mesh_from_file(
                    target=self._target,
                    name=geom_name,
                    mesh_file=mesh_file,
                    pos=geom.pos,
                    quat=geom.quat,
                    mesh_scale=mesh_scale,
                    mjs_material=mjs_material,
                )
                self._geoms.append(handle)
            case mujoco.mjtGeom.mjGEOM_SDF:
                raise NotImplementedError("")
            case _:
                raise NotImplementedError(
                    f"Geom type {geom.type} is not supported for visualization."
                )

    def add_traces(
        self,
        num_traces: int = 0,
        all_traces_rollout_size: int = 0,
        trace_name: str = "trace",
    ) -> None:
        """Add a collection of all traces to the visualizer, done in one go to avoid having too many things.

        We have two sets of traces to care about: the "elite" reward traces and the regular ones. Due to how the line
        segments work, we only need one handle per type.
        """
        # Size is num_traces * size of rollout per trace
        self._traces = []
        self._traces.append(
            add_segments(
                self._target,
                f"best_{trace_name}",
                1e-4
                * np.random.rand(4, 2, 3),  # non zero initialization to avoid errors
                rgb=DEFAULT_BEST_SPLINE_COLOR,
            )
        )
        self._traces[0].colors = np.tile(
            self._traces[0].colors[0, :, :], (all_traces_rollout_size, 1, 1)
        )
        if (rest_trace_size := num_traces - all_traces_rollout_size) > 0:
            self._traces.append(
                add_segments(
                    self._target,
                    f"other_{trace_name}",
                    1e-4
                    * np.random.rand(
                        4, 2, 3
                    ),  # non zero initialization to avoid errors
                    rgb=DEFAULT_SPLINE_COLOR,
                )
            )
            self._traces[1].colors = np.tile(
                self._traces[1].colors[0, :, :], (rest_trace_size, 1, 1)
            )

    def remove_traces(self) -> None:
        """Remove traces."""
        for trace in self._traces:
            trace.remove()
        self._traces = []

    def set_data(self, data: MjData) -> None:
        """Write updated configuration from mujoco data to viser viewer."""
        # Loop over all bodies, just reading the FK results from data.
        for i in range(1, len(self._bodies)):
            # Use atomic to update both position/orientation synchronously.
            with self._target.atomic():
                # Line up order of bodies in spec with order of bodies in model.
                data_idx = self._spec.bodies[i].id
                self._bodies[i].position = tuple(data.xpos[data_idx])
                self._bodies[i].wxyz = tuple(data.xquat[data_idx])

    def set_traces(
        self, traces: np.ndarray | None, all_traces_rollout_size: int
    ) -> None:
        """Write updated traces to viser viewer.

        Args:
            traces: trace sensors readings of size (self.num_elite * all_traces_rollout_size, 2, 3).
            all_traces_rollout_size: num_trace_sensors * single_rollout, size of all grouped trace sensor rollouts.
        """
        # Erase all traces if None is received
        if traces is None or self._num_trace_sensors == 0:
            self.remove_traces()
        else:
            num_traces, num_points, trace_dim = traces.shape
            assert trace_dim == 3
            assert num_points == 2, "Number of points in line segment must be 2"

            # check if the number of traces has updated
            if (
                len(self._traces) <= 1
                or self.all_traces_rollout_size != all_traces_rollout_size
                or num_traces != self.num_traces
            ):
                self.remove_traces()
                self.add_traces(
                    num_traces=num_traces,
                    all_traces_rollout_size=all_traces_rollout_size,
                )
                self.all_traces_rollout_size = all_traces_rollout_size
                self.num_traces = num_traces

            # check if there is only the elite trace
            if num_traces == all_traces_rollout_size:
                for trace in self._traces[1:]:
                    trace.remove()
                self._traces = [self._traces[0]]

            # Use atomic to update all traces synchronously
            with self._target.atomic():
                set_segment_points(
                    self._traces[0], traces[:all_traces_rollout_size, :, :]
                )
                if num_traces > all_traces_rollout_size and len(self._traces) > 1:
                    set_segment_points(
                        self._traces[1], traces[all_traces_rollout_size:, :, :]
                    )

    def remove(self) -> None:
        """Wrapper function to remove all geometries from Viser."""
        for geom in self._geoms:
            geom.remove()
        self.remove_traces()


def add_plane(
    target: ViserServer | ClientHandle,
    name: str,
    pos: Tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    quat: Tuple[float, float, float, float] | np.ndarray = (1.0, 0.0, 0.0, 0.0),
) -> SceneNodeHandle:
    """Add a plane geometry to the visualizer with optional position, quaternion, material, and name."""
    return target.scene.add_grid(
        name,
        position=pos,
        wxyz=quat,
        section_color=DEFAULT_GRID_SECTION_COLOR,
        cell_color=DEFAULT_GRID_CELL_COLOR,
    )


def add_sphere(
    target: ViserServer | ClientHandle,
    name: str,
    radius: float,
    pos: np.ndarray,
    quat: np.ndarray,
    rgba: np.ndarray,
) -> SceneNodeHandle:
    """Add a sphere geometry to the visualizer with optional position, quaternion, material, and name."""
    sphere_mesh = icosphere(DEFAULT_SPHERE_SUBDIVISIONS, radius)
    set_mesh_color(sphere_mesh, rgba)
    return target.scene.add_mesh_trimesh(name, sphere_mesh, position=pos, wxyz=quat)


def add_cylinder(
    target: ViserServer | ClientHandle,
    name: str,
    radius: float,
    height: float,
    pos: np.ndarray,
    quat: np.ndarray,
    rgba: np.ndarray,
) -> SceneNodeHandle:
    """Add a cylinder geometry to the visualizer with optional position, quaternion, material, and name.

    The cylinder is aligned with the z-axis
    """
    cylinder_mesh = cylinder(radius, height)
    set_mesh_color(cylinder_mesh, rgba)
    return target.scene.add_mesh_trimesh(name, cylinder_mesh, position=pos, wxyz=quat)


def add_box(
    target: ViserServer | ClientHandle,
    name: str,
    size: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    rgba: np.ndarray,
) -> SceneNodeHandle:
    """Add a box geometry to the visualizer with optional position, quaternion, material, and name."""
    box_mesh = box(size)
    set_mesh_color(box_mesh, rgba)
    return target.scene.add_mesh_trimesh(name, box_mesh, position=pos, wxyz=quat)


def add_capsule(
    target: ViserServer | ClientHandle,
    name: str,
    radius: float,
    length: float,
    pos: np.ndarray,
    quat: np.ndarray,
    rgba: np.ndarray,
) -> SceneNodeHandle:
    """Add a capsule geometry to the visualizer with optional position, quaternion, material, and name.

    The capsule is aligned with the z-axis
    """
    # First create a capsule mesh with trimesh since viser doesn't implement it.
    capsule_mesh = capsule(length, radius)
    set_mesh_color(capsule_mesh, rgba)
    return target.scene.add_mesh_trimesh(
        name,
        capsule_mesh,
        wxyz=quat,
        position=pos,
    )


def add_ellipsoid(
    target: ViserServer | ClientHandle,
    name: str,
    scaling: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    rgba: np.ndarray,
) -> SceneNodeHandle:
    """Add an ellipsoid geometry to the visualizer."""
    assert len(scaling) == 3, "Must provide exactly three scalings for ellipsoid."

    # Create sphere mesh.
    ellipsoid_mesh = icosphere(DEFAULT_SPHERE_SUBDIVISIONS, 1.0)

    # Scale this to get an ellipsoid.
    ellipsoid_mesh.apply_transform(scale_and_translate(scaling))

    set_mesh_color(ellipsoid_mesh, rgba)
    return target.scene.add_mesh_trimesh(name, ellipsoid_mesh, position=pos, wxyz=quat)


def add_mesh(
    target: ViserServer | ClientHandle,
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    pos: np.ndarray,
    quat: np.ndarray,
    rgba: np.ndarray,
) -> SceneNodeHandle:
    """Add a triangular mesh geometry to the visualizer.

    Add a triangular mesh geometry to the visualizer with specified vertices and faces,
    with optional position, quaternion, material, and name.

    Vertices: float (N, 3) and faces: int (M, 3).
    """
    mesh = trimesh.Trimesh(vertices, faces)
    set_mesh_color(mesh, rgba)
    return target.scene.add_mesh_trimesh(name, mesh, position=pos, wxyz=quat)


def add_mesh_from_file(
    target: ViserServer | ClientHandle,
    name: str,
    mesh_file: Path,
    pos: np.ndarray,
    quat: np.ndarray,
    mesh_scale: np.ndarray | None = None,
    mjs_material: MjsMaterial | None = None,
) -> SceneNodeHandle:
    """Add a triangle mesh from file, via trimesh."""
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file {mesh_file} does not exist.")
    mesh = trimesh.load(mesh_file, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh), "Loaded geometry is not a mesh type."
    if mesh_scale is not None:
        mesh.apply_scale(mesh_scale)

    # If mesh does not have a good texture, apply MuJoCo one.
    if isinstance(mesh.visual, ColorVisuals) and mjs_material is not None:
        apply_mujoco_material(mesh, mjs_material)

    return target.scene.add_mesh_trimesh(name, mesh, position=pos, wxyz=quat)


def add_spline(
    target: ViserServer | ClientHandle,
    name: str,
    positions: tuple[tuple[float, float, float], ...] | np.ndarray,
    pos: Tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    quat: Tuple[float, float, float, float] | np.ndarray = (1.0, 0.0, 0.0, 0.0),
    rgb: Tuple[float, float, float] = DEFAULT_SPLINE_COLOR,
    line_width: float = 4.0,
    segments: int | None = None,
    visible: bool = True,
) -> SplineCatmullRomHandle:
    """Add a spline to the visualizer with optional position, quaternion."""
    return target.scene.add_spline_catmull_rom(
        name,
        positions,
        position=pos,
        wxyz=quat,
        color=rgb,
        line_width=line_width,
        segments=segments,
        visible=visible,
    )


def add_segments(
    target: ViserServer | ClientHandle,
    name: str,
    points: np.ndarray,
    pos: Tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    quat: Tuple[float, float, float, float] | np.ndarray = (1.0, 0.0, 0.0, 0.0),
    rgb: Tuple[float, float, float] = DEFAULT_SPLINE_COLOR,
    line_width: float = 4.0,
    visible: bool = True,
) -> LineSegmentsHandle:
    """Add line segments to the visualizer with an optional position and orientation.

    TODO(@bhung) Potentially add support for different kinds of segments

    Args:
        target: ViserServer or handle to attach the segments to
        name: name of the segments
        points: size (N x 2 x 3) where index 0 is point, 1 is start vs end, and 2 is 3D coord
        pos: position that the points are defined with respect to. Defaults to origin
        quat: orientation that the points are defined with respect to. Defaults to identity
        rgb: colors of the points. Can be sized (N x 2 x 3) or a broadcastable shape
        line_width: width of the line, in pixels
        visible: whether or not the lines are initially visible
    """
    return target.scene.add_line_segments(
        name,
        points,
        rgb,
        line_width=line_width,
        wxyz=quat,
        position=pos,
        visible=visible,
    )


def set_mesh_color(mesh: trimesh.Trimesh, rgba: np.ndarray) -> None:
    """Set the color of a trimesh mesh."""
    if np.issubdtype(rgba.dtype, np.floating):
        rgba = rgba_float_to_int(rgba)

    mesh.visual = TextureVisuals(
        material=PBRMaterial(
            baseColorFactor=rgba,
            main_color=rgba,
            metallicFactor=0.5,
            roughnessFactor=1.0,
            alphaMode="BLEND" if rgba[-1] < 255 else "OPAQUE",
        )
    )


def set_spline_points(
    handle: SplineCatmullRomHandle,
    points: tuple[tuple[float, float, float], ...] | np.ndarray,
) -> None:
    """Set the spline waypoints."""
    points = np.asarray(points)
    assert len(points[0]) == 3 and len(points.shape) == 2
    handle.points = points


def set_segment_points(handle: LineSegmentsHandle, points: np.ndarray) -> None:
    """Set the line waypoints.

    Args:
        handle: handle for the line segments
        points: start and end points of the line segments (N x 2 x 3)
    """
    if isinstance(points, np.ndarray):
        assert len(points.shape) == 3 and points.shape[1] == 2 and points.shape[2] == 3
    handle.points = points


SCRIPT_DIR = Path(__file__).parent.resolve()

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
MJ_MODEL_PATH = (
    SCRIPT_DIR
    / "src"
    / "piper_control_ros2"
    / "piper_control_ros2"
    / "teach_mode"
    / "piper_grav_comp.xml"
)

from robot_descriptions import piper_mj_description

spec = MjSpec.from_file(piper_mj_description.MJCF_PATH)
model = spec.compile()
data = MjData(model)
server = ViserServer()
visualizer = ViserMjModel(server, spec)
robot = None  # piper_interface.PiperInterface(can_port="can0")
desired_joint_positions = np.zeros(len(JOINT_NAMES))
current_joint_positions = np.zeros(len(JOINT_NAMES))
joint_indices = [model.joint(name).id for name in JOINT_NAMES]
qpos_indices = model.jnt_qposadr[joint_indices]
qvel_indices = model.jnt_dofadr[joint_indices]
qpos_range = model.jnt_range[joint_indices]

kp = 0.2
kd = 2 * kp**0.5
ki = 0.1
joint1_pid_controller = PIDController(kp=kp, kd=kd, ki=ki, max_output=5.0)


def set_joint_positions(
    joint_positions: np.ndarray,
) -> None:
    """Set the joint positions in the mujoco data."""
    for joint_name, joint_pos in zip(JOINT_NAMES, joint_positions):
        joint_idx = model.joint(joint_name).id
        joint_qpos_idx = model.jnt_qposadr[joint_idx]
        data.qpos[joint_qpos_idx] = joint_pos
    mj.mj_kinematics(model, data)
    visualizer.set_data(data)


def get_joint_positions() -> np.ndarray:
    """Get the joint positions from the mujoco data."""
    joint_positions = np.zeros(len(JOINT_NAMES))
    for joint_name in JOINT_NAMES:
        joint_idx = model.joint(joint_name).id
        joint_qpos_idx = model.jnt_qposadr[joint_idx]
        joint_positions[joint_idx] = data.qpos[joint_qpos_idx]
    return joint_positions


kp_slider = server.gui.add_slider(
    label="Kp",
    min=0.0,
    max=10.0,
    step=0.1,
    initial_value=kp,
)
kd_slider = server.gui.add_slider(
    label="Kd",
    min=0.0,
    max=10.0,
    step=0.1,
    initial_value=kd,
)
ki_slider = server.gui.add_slider(
    label="Ki",
    min=0.0,
    max=1.0,
    step=0.05,
    initial_value=ki,
)


def kp_update_callback(event: viser.GuiEvent[viser.GuiSliderHandle]) -> None:
    joint1_pid_controller.kp = event.target.value


def kd_update_callback(event: viser.GuiEvent[viser.GuiSliderHandle]) -> None:
    joint1_pid_controller.kd = event.target.value


def ki_update_callback(event: viser.GuiEvent[viser.GuiSliderHandle]) -> None:
    joint1_pid_controller.ki = event.target.value


kp_slider.on_update(kp_update_callback)
kd_slider.on_update(kd_update_callback)
ki_slider.on_update(ki_update_callback)


def slider_update_callback(event: viser.GuiEvent[viser.GuiSliderHandle]) -> None:
    global desired_joint_positions
    joint_positions = get_joint_positions()
    joint_idx = model.joint(event.target.label).id
    joint_qpos_idx = model.jnt_qposadr[joint_idx]
    joint_positions[joint_qpos_idx] = event.target.value
    desired_joint_positions = joint_positions
    # if robot is None:
    # robot.command_joint_positions(joint_positions)
    # set_joint_positions(robot.get_joint_positions())
    # grav_torque_sampler = GravityTorqueSampler(
    #     robot,
    #     controller,
    #     sample_pose,
    #     p_gains=np.array([3.0, 15.0, 12.0, 3.0, 3.0, 2.0]),
    #     d_gains=np.array([3.0, 3.0, 3.0, 2.0, 2.0, 2.0]),
    # )
    # grav_torque_sampler.sample()


def create_robot_control_sliders(
    server: viser.ViserServer,
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    slider_handles: list[viser.GuiInputHandle[float]] = []
    for joint_name in JOINT_NAMES:
        joint_idx = model.joint(joint_name).id
        joint_qpos_idx = model.jnt_qposadr[joint_idx]
        limits = model.jnt_range[joint_idx]
        slider = server.gui.add_slider(
            label=joint_name,
            min=limits[0],
            max=limits[1],
            step=1e-3,
            initial_value=data.qpos[joint_qpos_idx],
        )
        slider.on_update(slider_update_callback)
        slider_handles.append(slider)
    return slider_handles


with server.gui.add_folder("Joint position control"):
    slider_handles = create_robot_control_sliders(server)


output_uplot = server.gui.add_uplot(
    data=(np.zeros(1000), np.zeros(1000)),
    series=(
        viser.uplot.Series(label="Time"),
        viser.uplot.Series(
            label="Output",
            stroke=["green"],
            width=1,
        ),
    ),
    legend=viser.uplot.Legend(show=True),
    scales={
        "x": viser.uplot.Scale(
            time=False,
            auto=True,
        ),
    },
    aspect=1.0,
)
uplot = server.gui.add_uplot(
    data=(np.zeros(1000), np.zeros(1000), np.zeros(1000)),
    series=(
        viser.uplot.Series(label="Time"),
        viser.uplot.Series(
            label="Joint",
            stroke=["red"],
            width=1,
        ),
        viser.uplot.Series(
            label="Target",
            stroke=["green"],
            width=1,
        ),
    ),
    legend=viser.uplot.Legend(show=True),
    scales={
        "x": viser.uplot.Scale(
            time=False,
            auto=True,
        ),
        "y": viser.uplot.Scale(range=(-3.14, 3.14)),
    },
    aspect=1.0,
)

robot = piper_interface.PiperInterface(can_port="can0")
robot.set_installation_pos(piper_interface.ArmInstallationPos.UPRIGHT)
piper_init.reset_arm(
    robot,
    arm_controller=piper_interface.ArmController.MIT,
    move_mode=piper_interface.MoveMode.MIT,
)
robot.show_status()
current_joint_positions = robot.get_joint_positions()
desired_joint_positions = current_joint_positions.copy()
set_joint_positions(current_joint_positions)

rate = loop_rate_limiters.RateLimiter(200)
# Move the arm joints using Mit mode controller.
# while True:
#     current_joint_positions = robot.get_joint_positions()
#     set_joint_positions(current_joint_positions)
#     rate.sleep()

JOINT_INDEX = 4


def tau_gravity(joint_positions):
    assert len(joint_positions) == len(JOINT_NAMES)
    data.qpos[qpos_indices] = joint_positions
    mj.mj_forward(model, data)
    return data.qfrc_bias[qvel_indices]


import json

with (SCRIPT_DIR / "left_grav_comp_samples.json").open("r") as f:
    json_data = json.load(f)

gravity_compenstation_torques = []
target_joint_angles = []
mj_gravity_compenstation_torques = []
for chunks in more_itertools.chunked(json_data, len(JOINT_NAMES)):
    gravity_compenstation_torque = np.zeros(len(JOINT_NAMES))
    for chunk in chunks:
        gravity_compenstation_torque[chunk["joint_idx"]] = chunk["grav_comp_torque"]
    target_joint_angles.append(
        chunks[-1]["target_joint_angles"]
    )  # Assume they are the same
    gravity_compenstation_torques.append(gravity_compenstation_torque)
    mj_gravity_compenstation_torques.append(tau_gravity(target_joint_angles[-1]))
target_joint_angles = np.asarray(target_joint_angles)
gravity_compenstation_torques = np.asarray(gravity_compenstation_torques)
mj_gravity_compenstation_torques = np.asarray(mj_gravity_compenstation_torques)


def cubic_gravity_tau(sim_torque, a, b, c, d):
    """A cubic adjustment of sim-predicted gravity torques."""
    return (
        a * sim_torque * sim_torque * sim_torque
        + b * sim_torque * sim_torque
        + c * sim_torque
        + d
    )


cubic_polynomial = []
for joint_index in range(len(JOINT_NAMES)):
    opt_params = optimize.curve_fit(
        cubic_gravity_tau,
        mj_gravity_compenstation_torques[:, joint_index],
        gravity_compenstation_torques[:, joint_index],
        p0=[0.0, 0.0, 1.0, 0.0],
    )
    cubic_polynomial.append(opt_params[0])

output_plot_data = collections.deque(maxlen=1000)
plot_data = collections.deque(maxlen=1000)
plot_time = np.arange(0, 1000)
with piper_control.MitJointPositionController(
    robot,
    kp_gains=10.0,
    kd_gains=0.8,
    rest_position=np.zeros(len(JOINT_NAMES)),
) as controller:
    while True:
        # joint1_pid_controller.set_target(desired_joint_positions[JOINT_INDEX])
        current_joint_positions = robot.get_joint_positions()
        # plot_data.append(current_joint_positions[JOINT_INDEX])
        set_joint_positions(current_joint_positions)
        # output = joint1_pid_controller.compute(current_joint_positions[JOINT_INDEX])
        # commanded_joint_torques = np.zeros(len(JOINT_NAMES))
        # commanded_joint_torques[JOINT_INDEX] = output
        mj_gravity_compenstation_torques = tau_gravity(current_joint_positions)
        commanded_joint_torques = [
            cubic_gravity_tau(
                mj_gravity_compenstation_torques[JOINT_INDEX],
                *cubic_polynomial[JOINT_INDEX],
            )
            for i in range(len(mj_gravity_compenstation_torques))
        ]
        controller.command_torques(commanded_joint_torques)
        # output_plot_data.append(output)
        # output_uplot.data = (
        #     plot_time,
        #     np.asarray(output_plot_data),
        # )
        # uplot.data = (
        #     plot_time,
        #     np.asarray(plot_data),
        #     np.full(len(plot_data), desired_joint_positions[JOINT_INDEX]),
        # )
        # grav_torque_sampler = GravityTorqueSampler(
        #     robot,
        #     controller,
        #     desired_joint_positions,
        #     p_gains=np.array([3.0, 15.0, 12.0, 3.0, 3.0, 2.0]),
        #     d_gains=np.array([3.0, 3.0, 3.0, 2.0, 2.0, 2.0]),
        # )
        #
        # grav_torque_sampler.sample()
        # current_joint_positions = robot.get_joint_positions()
        rate.sleep()
