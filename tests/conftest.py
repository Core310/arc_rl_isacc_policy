import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from agent.intersection_graph import (
    IntersectionGraph,
    IntersectionNode,
    ApproachInfo,
    ExitOption,
    EdgeGeometry,
    TurnCommand,
)
from agent.intersection_geometry import IntersectionLayout


def _make_cross_graph() -> IntersectionGraph:
    """
    Build a calibrated 4-way cross at origin, 2m approaches along each
    cardinal axis. Road IDs match the intersection_topology.json
    convention:
        road_A enters from +Y (heading = 270deg / -pi/2, going south)
        road_B enters from +X (heading = 180deg /  pi,  going west)
        road_C enters from -Y (heading =  90deg /  pi/2, going north)
        road_D enters from -X (heading =   0deg / 0.0,  going east)
    """
    # Build ApproachInfo for each road with left/straight/right exits
    # matching the star topology from the topology JSON.
    approaches = {
        "road_A": ApproachInfo(
            road_id="road_A",
            heading_rad=math.radians(270),
            exits={
                TurnCommand.LEFT:     ExitOption(TurnCommand.LEFT, "road_D"),
                TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_C"),
                TurnCommand.RIGHT:    ExitOption(TurnCommand.RIGHT, "road_B"),
            },
        ),
        "road_B": ApproachInfo(
            road_id="road_B",
            heading_rad=math.radians(180),
            exits={
                TurnCommand.LEFT:     ExitOption(TurnCommand.LEFT, "road_A"),
                TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_D"),
                TurnCommand.RIGHT:    ExitOption(TurnCommand.RIGHT, "road_C"),
            },
        ),
        "road_C": ApproachInfo(
            road_id="road_C",
            heading_rad=math.radians(90),
            exits={
                TurnCommand.LEFT:     ExitOption(TurnCommand.LEFT, "road_B"),
                TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_A"),
                TurnCommand.RIGHT:    ExitOption(TurnCommand.RIGHT, "road_D"),
            },
        ),
        "road_D": ApproachInfo(
            road_id="road_D",
            heading_rad=math.radians(0),
            exits={
                TurnCommand.LEFT:     ExitOption(TurnCommand.LEFT, "road_C"),
                TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_B"),
                TurnCommand.RIGHT:    ExitOption(TurnCommand.RIGHT, "road_A"),
            },
        ),
    }

    node = IntersectionNode(
        node_id="int_main",
        approaches=approaches,
        position=(0.0, 0.0),
        radius=1.5,
    )

    # Edge geometries: 2.5m straight approaches along each axis
    edges = {
        "road_A": EdgeGeometry(
            edge_id="road_A", length=2.5, heading=math.radians(270),
            from_node=None, to_node="int_main",
            start_position=(0.0, 2.5), end_position=(0.0, 0.0),
        ),
        "road_B": EdgeGeometry(
            edge_id="road_B", length=2.5, heading=math.radians(180),
            from_node=None, to_node="int_main",
            start_position=(2.5, 0.0), end_position=(0.0, 0.0),
        ),
        "road_C": EdgeGeometry(
            edge_id="road_C", length=2.5, heading=math.radians(90),
            from_node=None, to_node="int_main",
            start_position=(0.0, -2.5), end_position=(0.0, 0.0),
        ),
        "road_D": EdgeGeometry(
            edge_id="road_D", length=2.5, heading=math.radians(0),
            from_node=None, to_node="int_main",
            start_position=(-2.5, 0.0), end_position=(0.0, 0.0),
        ),
    }

    graph = IntersectionGraph(
        intersections={"int_main": node},
        edge_geometry=edges,
    )
    return graph


@pytest.fixture
def cross_graph() -> IntersectionGraph:
    return _make_cross_graph()


@pytest.fixture
def default_layout() -> IntersectionLayout:
    return IntersectionLayout(
        intersection_half_width=0.5,
        lane_half_width=0.25,
        pre_gate_distance=1.5,
        stop_line_tolerance=0.08,
        exit_detection_radius=0.8,
    )
