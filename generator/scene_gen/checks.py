"""Check connections, supports and valve access in a room."""

import math


EPS = 1e-8


def result(status, detail, **measurements):
    return {"status": status, "detail": detail, **measurements}


def contains(box, point, tolerance):
    return all(box[i] - tolerance <= point[i] <= box[i + 3] + tolerance for i in range(3))


def check_connections(scene, entities, tolerance):
    used = {valve["id"]: set() for valve in scene["valves"]}
    for link in scene["connections"]:
        if math.dist(link["point_a"], link["point_b"]) > tolerance:
            return result("fail", 'Connection points do not coincide', link=link)
        for side, other in (("a", "b"), ("b", "a")):
            name, point = link[side], link["point_" + side]
            entity = entities.get(name)
            if entity is None or link[side] == link[other]:
                return result("fail", 'Connection references a missing component or itself', link=link)
            if name in used and entities.get(link[other], {}).get("kind") != "pipe":
                return result("fail", 'A valve end must connect to a pipe', link=link)
            endpoints = entity.get("endpoints")
            if endpoints is None:
                if not contains(entity["box"], point, tolerance):
                    return result("fail", 'Pickup point is outside the component', link=link)
                continue
            matching = [index for index, end in enumerate(endpoints) if math.dist(end, point) <= tolerance]
            if len(matching) != 1:
                return result("fail", 'Pickup point is not on the declared port', link=link)
            if name in used:
                if matching[0] in used[name]:
                    return result("fail", 'The same valve end is connected more than once', link=link)
                used[name].add(matching[0])
    for name, endpoints in used.items():
        if len(endpoints) != 2:
            return result("fail", 'Both valve ends must connect to pipe', component=name)
    return result("pass", 'Both valve ends connect to pipe with aligned connection points', connections=len(scene["connections"]))


def check_supports(components, tolerance):
    contacts = 0
    for name, item in components.items():
        if item["kind"] == "floor":
            continue
        for parent in (item.get("supported_by"), *item.get("extra_supports", [])):
            if parent not in components or parent == name:
                return result("fail", 'Component is missing its designated support', component=name, support=parent)
            a, b = item["box"], components[parent]["box"]
            overlap = [min(a[i + 3], b[i + 3]) - max(a[i], b[i]) for i in range(3)]
            if min(overlap) < -tolerance or sum(value > EPS for value in overlap) < 2:
                return result("fail", 'Component and designated support do not have face contact', component=name, support=parent)
            contacts += 1
    for name in components:
        current, visited = name, set()
        while components[current]["kind"] != "floor":
            if current in visited:
                return result("fail", 'The support does not reach the ground', component=current)
            visited.add(current)
            current = components[current]["supported_by"]
    return result("pass", 'Designated support envelopes touch and the main support chain reaches the ground', contacts=contacts)


def check_install_clearance(components, policy):
    """Check R3 post clearances and R4 parallel bare-pipe spacing in this bay."""
    keys = ("pipe_post_clearance_m", "parallel_pipe_clearance_m")
    if not all(key in policy for key in keys):
        return result("pending", 'Missing bare-pipe clearances to columns or parallel pipes')
    post_limit, parallel_limit = (policy[key] for key in keys)
    pipes = [item for item in components.values() if item["kind"] == "pipe"]
    posts = [item for item in components.values() if item["kind"] == "post"]
    axes = {}
    for pipe in pipes:
        a, b = pipe["endpoints"]
        changed = [i for i in range(3) if abs(a[i] - b[i]) > EPS]
        if len(changed) != 1:
            return result("pending", 'Clearance checks cover straight pipes aligned with the coordinate axes', component=pipe["id"])
        axes[pipe["id"]] = changed[0]
    min_post = min_parallel = math.inf
    for index, pipe in enumerate(pipes):
        a = pipe["box"]
        exempt = (pipe.get("supported_by"), *pipe.get("extra_supports", []))
        for post in posts:
            if post["id"] in exempt:
                continue
            b = post["box"]
            gap = math.hypot(*(max(a[i] - b[i + 3], b[i] - a[i + 3], 0) for i in range(3)))
            min_post = min(min_post, gap)
            if gap + EPS < post_limit:
                return result("fail", 'Insufficient bare-pipe clearance to a column', pair=[pipe["id"], post["id"]], measured_m=gap, limit_m=post_limit)
        axis = axes[pipe["id"]]
        for other in pipes[index + 1:]:
            b = other["box"]
            overlap = min(a[axis + 3], b[axis + 3]) - max(a[axis], b[axis])
            if axes[other["id"]] != axis or overlap <= EPS:
                continue
            gap = math.hypot(*(
                pipe["endpoints"][0][i] - other["endpoints"][0][i] for i in range(3) if i != axis
            )) - pipe["radius_m"] - other["radius_m"]
            min_parallel = min(min_parallel, gap)
            if gap + EPS < parallel_limit:
                return result("fail", 'Insufficient clearance between parallel bare pipes', pair=[pipe["id"], other["id"]], measured_m=gap, limit_m=parallel_limit)
    return result(
        "pass", 'Bare-pipe clearances to columns and parallel lines meet the parameters',
        min_pipe_post_m=min_post if math.isfinite(min_post) else None,
        min_parallel_pipe_m=min_parallel if math.isfinite(min_parallel) else None,
        pipe_post_limit_m=post_limit, parallel_pipe_limit_m=parallel_limit,
    )


def handle_boxes(valve, handle, angle_deg):
    x, y, z = valve["center"]
    angle = math.radians(valve["yaw_deg"] + angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    for box in handle["boxes"]:
        px, py, pz = box["center"]
        hx, hy, hz = box["half_size"]
        heading = angle + math.radians(box.get("yaw_deg", 0))
        yield (x + c * px - s * py, y + s * px + c * py, z + pz), (hx, hy, hz), heading


def obstacles_for(components, valves, handle):
    obstacles = {name: item["box"] for name, item in components.items()}
    for valve in valves:
        if "body_box" not in valve or "initial_angle_deg" not in valve:
            return None
        boxes = [valve["body_box"]]
        for (x, y, z), (hx, hy, hz), heading in handle_boxes(valve, handle, valve["initial_angle_deg"]):
            c, s = abs(math.cos(heading)), abs(math.sin(heading))
            dx, dy = c * hx + s * hy, s * hx + c * hy
            boxes.append([x - dx, y - dy, z - hz, x + dx, y + dy, z + hz])
        obstacles[valve["id"]] = [
            *(min(box[i] for box in boxes) for i in range(3)),
            *(max(box[i] for box in boxes) for i in range(3, 6)),
        ]
    return obstacles


def box_overlaps(a, b):
    return all(min(a[i + 3], b[i + 3]) - max(a[i], b[i]) > EPS for i in range(3))


def surface_covers(box, point, half_width=0, half_depth=0):
    x, y, z = point
    return (
        abs(box[5] - z) <= EPS
        and box[0] - EPS <= x - half_width <= x + half_width <= box[3] + EPS
        and box[1] - EPS <= y - half_depth <= y + half_depth <= box[4] + EPS
    )


def check_station(station, surfaces, obstacles, policy):
    x, y, z = station
    dx, dy = policy["station_width_m"] / 2, policy["station_depth_m"] / 2
    height = policy["station_height_m"]
    if not any(surface_covers(box, station, dx, dy) for box in surfaces.values()):
        return result("fail", 'The standing area does not rest fully on one plane', point=station)
    for name, box in obstacles.items():
        if box_overlaps((x - dx, y - dy, z, x + dx, y + dy, z + height), box):
            return result("fail", 'Standing space is occupied', component=name, point=station)
    return result("pass", 'The standing area is fully supported and its tested volume is clear', size_m=[2 * dx, 2 * dy, height])


def clip_segment(a, b, lower, upper):
    """Return the portion of the segment inside a box of the given dimension."""
    first, last = 0.0, 1.0
    for i, (low, high) in enumerate(zip(lower, upper)):
        delta = b[i] - a[i]
        if abs(delta) < EPS:
            if not low <= a[i] <= high:
                return None
            continue
        enter, leave = sorted(((low - a[i]) / delta, (high - a[i]) / delta))
        first, last = max(first, enter), min(last, leave)
        if first > last:
            return None
    return first, last


def surface_intervals(a, b, boxes):
    intervals = []
    for box in boxes:
        interval = clip_segment(a, b, [box[0] - EPS, box[1] - EPS], [box[3] + EPS, box[4] + EPS])
        if interval is not None:
            intervals.append(interval)
    extent = 0.0
    for first, last in sorted(intervals):
        if first > extent + EPS:
            return False
        extent = max(extent, last)
    return extent >= 1 - EPS


def check_route(route, station, surfaces, obstacles, policy):
    if len(route) < 2 or math.dist(route[-1], station) > EPS:
        return result("fail", 'Route does not connect the entrance to the standing area')
    for point in route:
        if not any(surface_covers(box, point) for box in surfaces.values()):
            return result("fail", 'Route foothold is off the floor or tread', point=point)

    width, depth, height = (policy[key] for key in ("walk_width_m", "walk_depth_m", "walk_height_m"))
    max_rise = policy["max_step_height_m"]
    max_span = policy["max_step_span_m"]
    headroom, max_step = math.inf, 0.0
    for index, (a, b) in enumerate(zip(route, route[1:])):
        low, high = sorted((a[2], b[2]))
        rise, span = high - low, math.dist(a[:2], b[:2])
        max_step = max(max_step, rise)
        if rise > max_rise + EPS or (rise > EPS and not EPS < span <= max_span + EPS):
            return result("fail", 'Stair tread exceeds the test envelope', segment=index, rise_m=rise, span_m=span)
        if span <= EPS:
            continue
        levels = [box for box in surfaces.values() if abs(box[5] - low) <= EPS or abs(box[5] - high) <= EPS]
        if not surface_intervals(a, b, levels):
            return result("fail", 'Walking surfaces are discontinuous under the route', segment=index)

        dx, dy = (abs(b[axis] - a[axis]) / span for axis in (0, 1))
        x_radius = (depth * dx + width * dy) / 2
        y_radius = (depth * dy + width * dx) / 2
        for name, box in obstacles.items():
            xy_low = [box[0] - x_radius + EPS, box[1] - y_radius + EPS]
            xy_high = [box[3] + x_radius - EPS, box[4] + y_radius - EPS]
            xy = clip_segment(a, b, xy_low, xy_high)
            if xy is None or box[5] <= low + EPS:
                continue
            if name in surfaces and box[5] <= low + max_rise + EPS:
                # Permit the walking envelope to extend over the edge of a low
                # tread. The center still cannot cross its riser on a flat leg.
                if rise > EPS and abs(box[5] - high) <= EPS:
                    continue
                xy_low, xy_high = [box[0] + EPS, box[1] + EPS], [box[3] - EPS, box[4] - EPS]
                if clip_segment(a, b, xy_low, xy_high) is None:
                    continue
            top_foot = max(a[2] + fraction * (b[2] - a[2]) for fraction in xy)
            if box[2] >= top_foot - EPS:
                headroom = min(headroom, box[2] - top_foot)
            hit = clip_segment(a, b, [*xy_low, box[2] - height + EPS], [*xy_high, box[5] - EPS])
            if hit is not None:
                return result("fail", 'Test envelope intersects an obstacle along the route', segment=index, component=name)
    return result(
        "pass", 'The foothold centerline has continuous support. The test envelope is clear from the entrance to the stance and back.',
        checked_width_m=width, checked_depth_m=depth, checked_height_m=height, max_rise_m=max_step,
        min_headroom_m=headroom if math.isfinite(headroom) else None,
    )


def rotated_box_hits(center, half_size, heading, obstacle, margin):
    """Use all four separating axes in plan plus the vertical interval."""
    delta = [center[i] - (obstacle[i] + obstacle[i + 3]) / 2 for i in range(3)]
    a, b, h = [(obstacle[i + 3] - obstacle[i]) / 2 for i in range(3)]
    x, y, z = [width + margin for width in half_size]
    c, s = math.cos(heading), math.sin(heading)
    dx, dy, dz = delta
    return (
        abs(dz) < h + z - EPS
        and abs(dx) < a + abs(c) * x + abs(s) * y - EPS
        and abs(dy) < b + abs(s) * x + abs(c) * y - EPS
        and abs(c * dx + s * dy) < x + abs(c) * a + abs(s) * b - EPS
        and abs(-s * dx + c * dy) < y + abs(s) * a + abs(c) * b - EPS
    )


def check_handle_sweep(scene, target, obstacles, policy):
    handle = scene["handle"]
    opened, closed = handle["open_deg"], handle["closed_deg"]
    start, end = (closed, opened) if scene["task"] == "open" else (opened, closed)
    samples = max(1, math.ceil(abs(end - start) / 2))
    radial = max(math.hypot(*box["center"][:2]) + math.hypot(*box["half_size"][:2]) for box in handle["boxes"])
    # Every instant is at most half a sample step away. Pad by the maximum
    # movement of any lever point between that instant and its nearest sample.
    pad = 2 * radial * math.sin(math.radians(abs(end - start) / samples) / 4)
    margin = policy["sweep_margin_m"]
    for step in range(samples + 1):
        angle = start + (end - start) * step / samples
        for center, half_size, heading in handle_boxes(target, handle, angle):
            for name, box in obstacles.items():
                if name != target["id"] and rotated_box_hits(center, half_size, heading, box, margin + pad):
                    return result("fail", 'Operator sweep overlaps an obstacle', component=name, sample_angle_deg=angle)
    return result("pass", 'The full operator sweep is unobstructed', angle_range_deg=[start, end], margin_m=margin, interpolation_pad_m=pad)


def check_scene(scene, policy):
    """Return per-check evidence. Policy values describe draft test envelopes."""
    components = {item["id"]: item for item in scene["components"]}
    valves = {item["id"]: item for item in scene["valves"]}
    surfaces = {name: item["box"] for name, item in components.items() if item["kind"] == "floor" or item.get("walkable")}
    tolerance = policy.get("join_tolerance_m", 1e-6)
    checks = {
        "connections": check_connections(scene, components | valves, tolerance),
        "supports": check_supports(components, tolerance),
        "install_clearance": check_install_clearance(components, policy),
    }
    handle = scene.get("handle")
    obstacles = obstacles_for(components, valves.values(), handle) if handle and handle.get("boxes") else None
    if obstacles is None:
        for name in ("station", "route", "handle_sweep"):
            checks[name] = result("pending", 'Missing valve body, handle or initial-angle collision geometry')
        return checks
    checks["station"] = check_station(scene["station"], surfaces, obstacles, policy)
    checks["route"] = check_route(scene["route"], scene["station"], surfaces, obstacles, policy)
    checks["handle_sweep"] = check_handle_sweep(scene, valves[scene["target"]], obstacles, policy)
    return checks
