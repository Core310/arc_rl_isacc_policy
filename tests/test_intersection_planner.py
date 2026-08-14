import math
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.intersection_graph import TurnCommand, ApproachInfo, ExitOption, IntersectionNode
from agent.agent_node import AgentNode, AgentConfig, WorkerConfig, WorkerNode


class TestPlanarPathPlanner:
    """
    Geometric unit tests for the Worker-side 2D trajectory planner.
    """

    def test_straight_plan_stays_in_right_lane(self, cross_graph, default_layout):
        from agent.planar_planner import PlanarPathPlanner
        planner = PlanarPathPlanner()
        node = cross_graph.get_intersection("int_main")
        plan = planner.plan(
            current_xy=(-1.2, 0.0), current_heading=0.0,
            intersection=node,
            entry_road_id="road_D", exit_road_id="road_B",
            turn_command=TurnCommand.STRAIGHT, layout=default_layout,
        )
        assert plan is not None
        assert plan.num_waypoints == 5
        lane_y = -default_layout.lane_half_width
        for wp in plan.waypoints[1:]:
            assert abs(wp.y - lane_y) < 1e-6
        for wp in plan.waypoints[1:]:
            assert abs(wp.heading - 0.0) < 1e-6

    def test_right_turn_plan_ends_on_correct_road(self, cross_graph, default_layout):
        from agent.planar_planner import PlanarPathPlanner
        planner = PlanarPathPlanner()
        node = cross_graph.get_intersection("int_main")
        plan = planner.plan(
            current_xy=(-1.2, 0.0), current_heading=0.0,
            intersection=node,
            entry_road_id="road_D", exit_road_id="road_A",
            turn_command=TurnCommand.RIGHT, layout=default_layout,
        )
        assert plan is not None
        import math as _m
        final = plan.waypoints[-1]
        assert abs(_m.degrees(final.heading) - 90.0) < 1e-3

    def test_left_turn_plan_ends_on_correct_road(self, cross_graph, default_layout):
        from agent.planar_planner import PlanarPathPlanner
        planner = PlanarPathPlanner()
        node = cross_graph.get_intersection("int_main")
        plan = planner.plan(
            current_xy=(-1.2, 0.0), current_heading=0.0,
            intersection=node,
            entry_road_id="road_D", exit_road_id="road_C",
            turn_command=TurnCommand.LEFT, layout=default_layout,
        )
        assert plan is not None
        import math as _m
        final = plan.waypoints[-1]
        assert abs(_m.degrees(final.heading) - (-90.0)) < 1e-3

    def test_plan_rejects_past_center_vehicle(self, cross_graph, default_layout):
        from agent.planar_planner import PlanarPathPlanner
        planner = PlanarPathPlanner()
        node = cross_graph.get_intersection("int_main")
        plan = planner.plan(
            current_xy=(+1.0, 0.0), current_heading=0.0,
            intersection=node,
            entry_road_id="road_D", exit_road_id="road_B",
            turn_command=TurnCommand.STRAIGHT, layout=default_layout,
        )
        assert plan is None

    def test_plan_rejects_missing_exit_road(self, cross_graph, default_layout):
        from agent.planar_planner import PlanarPathPlanner
        planner = PlanarPathPlanner()
        node = cross_graph.get_intersection("int_main")
        plan = planner.plan(
            current_xy=(-1.2, 0.0), current_heading=0.0,
            intersection=node,
            entry_road_id="road_D", exit_road_id=None,
            turn_command=TurnCommand.STRAIGHT, layout=default_layout,
        )
        assert plan is None

    def test_plan_rejects_uncalibrated_node(self, default_layout):
        from agent.planar_planner import PlanarPathPlanner
        approaches = {
            "road_X": ApproachInfo(
                road_id="road_X", heading_rad=0.0,
                exits={TurnCommand.STRAIGHT: ExitOption(TurnCommand.STRAIGHT, "road_Y")},
            ),
            "road_Y": ApproachInfo(
                road_id="road_Y", heading_rad=math.pi,
                exits={},
            ),
        }
        node = IntersectionNode(node_id="uncal", approaches=approaches, position=None)
        plan = PlanarPathPlanner().plan(
            current_xy=(-1.0, 0.0), current_heading=0.0,
            intersection=node,
            entry_road_id="road_X", exit_road_id="road_Y",
            turn_command=TurnCommand.STRAIGHT, layout=default_layout,
        )
        assert plan is None

    def test_cross_track_distance_zero_on_path(self, cross_graph, default_layout):
        from agent.planar_planner import PlanarPathPlanner
        planner = PlanarPathPlanner()
        node = cross_graph.get_intersection("int_main")
        plan = planner.plan(
            current_xy=(-1.2, 0.0), current_heading=0.0,
            intersection=node,
            entry_road_id="road_D", exit_road_id="road_B",
            turn_command=TurnCommand.STRAIGHT, layout=default_layout,
        )
        assert plan.cross_track_distance((0.0, -0.25)) < 1e-6
        assert abs(plan.cross_track_distance((0.0, -0.15)) - 0.1) < 1e-6

    def test_plan_arc_length_monotone(self, cross_graph, default_layout):
        from agent.planar_planner import PlanarPathPlanner
        planner = PlanarPathPlanner()
        node = cross_graph.get_intersection("int_main")
        plan = planner.plan(
            current_xy=(-1.2, 0.0), current_heading=0.0,
            intersection=node,
            entry_road_id="road_D", exit_road_id="road_B",
            turn_command=TurnCommand.STRAIGHT, layout=default_layout,
        )
        assert plan.waypoints[0].s == 0.0
        for i in range(1, plan.num_waypoints):
            assert plan.waypoints[i].s >= plan.waypoints[i - 1].s


class TestPlanViaWorker:
    def test_worker_produces_plan_on_pre_gate(self, cross_graph, default_layout):
        worker = WorkerNode(
            agent_id="a0",
            graph=cross_graph,
            config=WorkerConfig(
                use_stop_line=True, detector_kind="geometric",
                layout=default_layout,
                mode="route", route=[TurnCommand.STRAIGHT],
            ),
        )
        assert worker.current_plan is None
        worker.step(position=(-1.2, 0.0), heading=0.0, speed=1.0, dt=0.05)
        assert worker.state == WorkerNode.DECIDING
        plan = worker.current_plan
        assert plan is not None
        assert plan.entry_road_id == "road_D"
        assert plan.exit_road_id == "road_B"
        assert plan.num_waypoints == 5

    def test_plan_visible_in_agent_info_dict(self, cross_graph, default_layout):
        agent = AgentNode(
            graph=cross_graph,
            config=AgentConfig(
                agent_id="a0",
                worker=WorkerConfig(
                    use_stop_line=True, detector_kind="geometric",
                    layout=default_layout,
                    mode="route", route=[TurnCommand.STRAIGHT],
                ),
            ),
        )
        agent.worker_step(position=(-1.2, 0.0), heading=0.0, speed=1.0, dt=0.05)
        info = agent.info
        assert info["plan_present"] is True
        assert info["plan_num_waypoints"] == 5
        assert info["plan_length_m"] > 0.0
