import math
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.intersection_graph import TurnCommand
from agent.agent_node import WorkerConfig, WorkerNode


class TestWorkerSubstateMachine:
    """
    Scripted-position tests that exercise WorkerNode.step() directly.
    Uses detector_kind='geometric' so we don't need camera images.
    """

    def test_cruising_holds_when_far_from_intersection(self, cross_graph, default_layout):
        """Far from the intersection, Worker stays CRUISING, go=1."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=True, detector_kind="geometric",
                layout=default_layout,
            ),
        )
        token, go = worker.step(
            position=(-2.3, 0.0), heading=0.0, speed=1.0, dt=0.05,
        )
        assert worker.state == WorkerNode.CRUISING
        assert worker.substate == WorkerNode.SUB_NONE
        assert go == 1.0

    def test_pre_gate_promotes_to_deciding_approaching(self, cross_graph, default_layout):
        """Within pre-gate distance, Worker promotes to DECIDING/APPROACHING."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=True, detector_kind="geometric",
                layout=default_layout,
                mode="route", route=[TurnCommand.STRAIGHT],
            ),
        )
        token, go = worker.step(
            position=(-1.2, 0.0), heading=0.0, speed=1.0, dt=0.05,
        )
        assert worker.state == WorkerNode.DECIDING
        assert worker.substate == WorkerNode.SUB_APPROACHING
        assert go == 0.0
        assert worker.current_approach_road_id == "road_D"

    def test_approaching_to_stopping_when_speed_drops(self, cross_graph, default_layout):
        """APPROACHING -> STOPPING when speed < threshold."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=True, detector_kind="geometric",
                layout=default_layout,
                mode="route", route=[TurnCommand.STRAIGHT],
                stopped_speed_threshold=0.1,
            ),
        )
        worker.step(position=(-1.2, 0.0), heading=0.0, speed=1.0, dt=0.05)
        worker.step(position=(-0.7, 0.0), heading=0.0, speed=0.5, dt=0.05)
        assert worker.substate == WorkerNode.SUB_APPROACHING
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.05)
        assert worker.substate == WorkerNode.SUB_STOPPING

    def test_stopping_to_cleared_respects_dwell_and_scheduler(self, cross_graph, default_layout):
        """STOPPING -> CLEARED requires dwell_time elapsed AND scheduler go."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=True, detector_kind="geometric",
                layout=default_layout,
                mode="route", route=[TurnCommand.STRAIGHT],
                stopped_speed_threshold=0.1,
                stop_dwell_time=0.3,
            ),
        )
        worker.step(position=(-1.2, 0.0), heading=0.0, speed=1.0, dt=0.05)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.05)
        assert worker.substate == WorkerNode.SUB_STOPPING
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        assert worker.substate == WorkerNode.SUB_STOPPING
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        assert worker.state in (WorkerNode.DECIDING, WorkerNode.COMMITTED)

    def test_committed_traversing_to_exited_correct_road(self, cross_graph, default_layout):
        """TRAVERSING -> EXITED fires on correct road exit, exit_correct=True."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=True, detector_kind="geometric",
                layout=default_layout,
                mode="route", route=[TurnCommand.STRAIGHT],
                stopped_speed_threshold=0.1,
                stop_dwell_time=0.1,
                moving_speed_threshold=0.2,
            ),
        )
        worker.step(position=(-1.2, 0.0), heading=0.0, speed=1.0, dt=0.05)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.05)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        assert worker.state == WorkerNode.COMMITTED
        assert worker.substate == WorkerNode.SUB_TRAVERSING
        assert worker.committed_exit_road_id == "road_B"
        worker.step(position=(1.0, 0.0), heading=0.0, speed=1.0, dt=0.05)
        assert worker.exited_road_id == "road_B"
        assert worker.exit_correct is True

    def test_committed_traversing_to_exited_wrong_road(self, cross_graph, default_layout):
        """Exiting on a road != committed road sets exit_correct=False."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=True, detector_kind="geometric",
                layout=default_layout,
                mode="route", route=[TurnCommand.STRAIGHT],
                stopped_speed_threshold=0.1,
                stop_dwell_time=0.1,
                moving_speed_threshold=0.2,
            ),
        )
        worker.step(position=(-1.2, 0.0), heading=0.0, speed=1.0, dt=0.05)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.05)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        worker.step(position=(-0.6, 0.0), heading=0.0, speed=0.05, dt=0.1)
        assert worker.committed_exit_road_id == "road_B"
        worker.step(
            position=(0.0, -1.0), heading=-math.pi / 2, speed=1.0, dt=0.05,
        )
        assert worker.exited_road_id == "road_C"
        assert worker.exit_correct is False

    def test_legacy_mode_preserves_original_behavior(self, cross_graph):
        """use_stop_line=False => DECIDING flips to COMMITTED in one tick."""
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=False,
                mode="route", route=[TurnCommand.STRAIGHT],
            ),
        )
        token, go = worker.step(
            position=(-1.0, 0.0), heading=0.0, speed=1.0, dt=0.05,
        )
        assert worker.state == WorkerNode.COMMITTED
        assert worker.substate == WorkerNode.SUB_NONE
