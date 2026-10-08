from reachy_retarget.report import markdown, reason_kind, summarize


def test_reason_kind_strips_numbers_sides_and_objects():
    assert reason_kind("left TCP rotation residual 0.115 rad > 0.05 rad at frame 12 (3 frames)") == "TCP rotation residual"
    assert reason_kind("right TCP position residual 7.2 mm > 5 mm at frame 3 (1 frames)") == "TCP position residual"
    assert reason_kind("right hand-SquareNut relative pose drifts 18 mm / 0.3 rad during frames 3-9") \
        == "hand-object relative pose drifts"
    assert reason_kind("joint limit exceeded by 0.01 rad at frame 3") == "joint limit exceeded"
    assert reason_kind("speed limit exceeded: joint 5 at 1.2 x limit (frame 3)") == "speed limit exceeded"
    assert reason_kind("self-clearance 5 mm < 9 mm at frame 3") == "self-clearance"
    assert reason_kind("base footprint overlaps static scene geometry by 3 mm at frame 5") \
        == "base footprint overlaps static scene geometry"
    assert reason_kind("grasp_drift: right/cubeA: 0.0148 m, 0.0490 rad") == "grasp_drift"
    assert reason_kind("object_environment_penetration: 0.0051 m > source-relative threshold 0.0039 m") \
        == "object_environment_penetration"
    assert reason_kind("scene: missing collision assets: a.stl") == "scene"
    assert reason_kind("IK produced non-finite joint values") == "IK produced non-finite joint values"


def test_summary_buckets_reasons_per_tier():
    recs = [
        {"family": "f", "dataset": "d", "status": "ok",
         "K": {"passed": False, "reasons": ["left TCP rotation residual 0.115 rad > 0.05 rad at frame 1 (1 frames)",
                                            "right TCP rotation residual 0.2 rad > 0.05 rad at frame 2 (4 frames)"]},
         "P": {"passed": False, "reasons": ["grasp_drift: right/cubeA: 0.01 m, 0.04 rad", "tcp_tracking: 0.04 m, 0.1 rad"]}},
        {"family": "f", "dataset": "d", "status": "ok",
         "K": {"passed": False, "reasons": ["right TCP rotation residual 0.3 rad > 0.05 rad at frame 9 (2 frames)"]},
         "P": {"passed": False, "reasons": ["tcp_tracking: 0.05 m, 0.1 rad"]}},
    ]
    s = summarize(recs)
    assert s["top_failure_reasons"]["K"] == {"TCP rotation residual": 2}
    assert s["top_failure_reasons"]["P"] == {"grasp_drift": 1, "tcp_tracking": 1}
    assert s["failure_reasons_any"]["P"] == {"tcp_tracking": 2, "grasp_drift": 1}
    assert "TCP rotation residual: 2 / 2" in markdown(s)
