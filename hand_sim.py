import argparse
import time
import xml.etree.ElementTree as ET
from multiprocessing import shared_memory
from pathlib import Path
from typing import Literal

import mujoco
import mujoco.viewer
import numpy as np

TARGET_SHM_NAME = "svh_hand_targets"

def get_hand_urdf_path(side: Literal["left", "right"] = "right") -> Path:
    if side == "left":
        return Path(__file__).parent / 'schunk_hand/schunk_svh_hand_left.urdf'
    elif side == "right":
        return Path(__file__).parent / 'schunk_hand/schunk_svh_hand_right.urdf'
    else:
        raise ValueError(f"Invalid side: {side}")

def get_joint_names(side: Literal["left", "right"] = "right") -> list[str]:
    """Ordered joint names straight from the URDF.

    This is the single source of truth for shared-memory target indices:
    We create one actuator per joint in this exact order,
    so index i here == data.ctrl[i] in the compiled model.
    """
    path = str(get_hand_urdf_path(side))
    hand_spec = mujoco.MjSpec.from_file(path)
    return [joint.name for joint in hand_spec.joints]

def get_mimic_joints(side: Literal["left", "right"] = "right") -> dict[str, tuple[str, float, float]]:
    """Map mimic joint name -> (driver joint name, multiplier, offset), parsed from <mimic> in the URDF.

    MuJoCo's URDF importer silently drops <mimic> (same as <transmission>),
    so this never reaches MjSpec/MjModel -- these 11 joints are mechanically
    slaved to one of the 9 independently-actuated joints on the real
    hardware (mimic_pos = multiplier * driver_pos + offset), so we apply the
    same ratios in software after retargeting.
    """
    root = ET.parse(get_hand_urdf_path(side)).getroot()
    mimics = {}
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            mimics[joint.get("name")] = (
                mimic.get("joint"),
                float(mimic.get("multiplier", 1.0)),
                float(mimic.get("offset", 0.0)),
            )
    return mimics


def apply_mimic_joints(targets: np.ndarray, joint_names: list[str], side: Literal["left", "right"] = "right") -> None:
    """Fill in the 11 mimic joint targets from the 9 real ones, in place.

    targets/joint_names must be in the same order (e.g. from get_joint_names).
    Only the 9 independently-driven joints need to be set beforehand --
    this derives the rest.
    """
    index_of = {name: i for i, name in enumerate(joint_names)}
    for mimic_name, (driver_name, multiplier, offset) in get_mimic_joints(side).items():
        targets[index_of[mimic_name]] = multiplier * targets[index_of[driver_name]] + offset


def add_position_actuators(hand_spec: mujoco.MjSpec, kp: float = 10.0, kv: float = 5.0) -> None:
    """Add one position (PD) actuator per joint, mirroring MJCF's <position> shorthand.

    force = kp * (ctrl - joint_pos) - kv * joint_vel
    URDF has no notion of actuators, so mujoco.MjModel.from_xml_path(urdf)
    always compiles with nu == 0.
    """
    for joint in hand_spec.joints:
        gainprm = [kp] + [0.0] * 9
        biasprm = [0.0, -kp, -kv] + [0.0] * 7
        hand_spec.add_actuator(
            name="act_" + joint.name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=joint.name,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=gainprm,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=biasprm,
            ctrllimited=True,
            ctrlrange=list(joint.range),
        )

def get_actuator_ids(model: mujoco.MjModel) -> dict[str, int]:
    """Map actuator name -> data.ctrl index, e.g. 'hand_act_left_hand_Ring_Finger'."""
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(model.nu) # model.nu is the number of actuators. here we're adding one for every joint
    }

def build_scene(side: Literal["left", "right"] = "right") -> mujoco.MjModel:
    path = str(get_hand_urdf_path(side))
    hand_spec = mujoco.MjSpec.from_file(path)
    add_position_actuators(hand_spec)

    # Disable finger self-collision: at rest pose the fingers already touch
    # the palm/each other, and once mj_step resolves contacts for real
    # those forces fight the position actuators instead of letting them
    # converge. contype=2/conaffinity=1 still collides with the ground
    # (contype=1/conaffinity=1) since 1&1 != 0, but skips hand-vs-hand
    # pairs since 2&1 == 0 both ways.
    for geom in hand_spec.geoms:
        geom.contype = 2
        geom.conaffinity = 1

    scene_spec = mujoco.MjSpec()
    # implicitfast handles actuator stiffness/damping implicitly -- needed
    # here because these finger links are extremely light (~1e-6 kg*m^2
    # inertias), so a stiff PD gain blows up under the default Euler
    # integrator (NaN in qacc) but is stable under implicitfast.
    scene_spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene_spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512, height=512,
    )
    scene_spec.worldbody.add_light(
        name="key",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[1, 1, 2], dir=[-1, -1, -2],
        diffuse=[0.8, 0.8, 0.8],
    )
    scene_spec.worldbody.add_light(
        name="fill",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[-1, -1, 2], dir=[1, 1, -2],
        diffuse=[0.3, 0.3, 0.35],
    )
#     <worldbody>
#     <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
#     <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.9 0.8 1"/>
#   </worldbody>
    scene_spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.1],
        rgba=[0.8, 0.9, 0.8, 1]
    )

    frame = scene_spec.worldbody.add_frame(pos=[0, 0, 0])
    scene_spec.attach(hand_spec, prefix="hand_", frame=frame)

    return scene_spec.compile()


def open_target_shm(nu: int, name: str = TARGET_SHM_NAME) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    """Create (or re-create) the shared joint-target buffer.

    This process owns the segment: any external writer (e.g. the CV
    process) attaches with create=False using the same name/dtype/shape
    and just overwrites values -- there's no locking, the reader always
    gets whatever the latest write happened to be, which is exactly the
    "never block, always use latest" behavior we want for real-time control.
    """
    nbytes = nu * np.dtype(np.float64).itemsize
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
    except FileExistsError:
        # Leftover from a previous crashed/force-killed run -- clear and
        # recreate. Loud on purpose: if a writer (e.g. main.py) already
        # attached to the stale segment before this runs, it'll keep writing
        # into the orphaned old one while we read from this new one --
        # symptom is main.py's values changing but hand_sim.py's frozen.
        stale = shared_memory.SharedMemory(name=name)
        stale.close()
        stale.unlink()
        shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
    targets = np.ndarray((nu,), dtype=np.float64, buffer=shm.buf)
    return shm, targets

def simulate_hand(
    side: Literal["left", "right"] = "right"
):
    print(f"hand_sim.py: simulating {side.upper()} hand -- main.py must use --side {side} too")
    model = build_scene(side)
    data = mujoco.MjData(model)

    ctrl_lower = model.actuator_ctrlrange[:, 0]
    ctrl_upper = model.actuator_ctrlrange[:, 1]

    shm, targets = open_target_shm(model.nu)
    targets[:] = data.qpos[model.jnt_qposadr]  # start at the resting pose

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            viewer.sync()

            while viewer.is_running():
                data.ctrl[:] = np.clip(targets, ctrl_lower, ctrl_upper)
                mujoco.mj_step(model, data)
                viewer.sync()

                time.sleep(1 / 60)
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for Schunk SVH Hand")
    parser.add_argument("--side", choices=["left", "right"], default="right")
    args = parser.parse_args()
    simulate_hand(side=args.side)
