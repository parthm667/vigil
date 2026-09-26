"""ReachGlass laptop stack: Tello scout perception, exploration and mission control.

Layout (each module sits behind a small interface so it can be swapped):
    types.py      shared data types and the frame/angle conventions
    geometry.py   camera model: pixels <-> bearings, ranges from known sizes
    config.py     nested dataclass config, YAML overrides
    sources/      frame sources (Tello UDP via OpenCV, webcam, video file, sim)
    detect/       detectors (dummy colour blob, YOLO, sim oracle)
    track/        multi-object tracker + target lock
    person/       person geometry: range, facing
    query/        text query -> target class
    drone/        drone interface (Tello, dry run, sim) + safety governor
    sim/          simulator used for closed-loop verification
    mapping/      odometry, occupancy grid, free space, semantic memory
    behaviors/    follow, scan, explore, approach
    mission/      mission state machine + guidance info
    app.py        main loop
"""

__version__ = "0.1.0"
