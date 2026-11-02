# My changes to the pvtrace library in 2025

These are my working notes. The patched library is not in this repo.

```text
First change made is using the following script:
Makes the library compatible with the latest numpy version.

Made the following changes inside photon tracer algorithm (photon_tracer.py):
Original code:
def find_container(intersections):
    """ Returns the container node.
    ...
    """
    if len(intersections) == 1:
        return intersections[0].hit
    count = collections.Counter([x.hit for x in intersections]).most_common()
    candidates = [x[0] for x in count if x[1] == 1]
    pairs = []
    for intersection in intersections:
        node = intersection.hit
        if node in candidates:
            pairs.append((node, intersection.distance))
    # [(node, dist), (node, dist)... ]
    pairs = sorted(pairs, key=lambda tup: tup[1])
    containers, _ = zip(*pairs)
    container = containers[0]
    return container
My modified code:
def find_container(intersections):
    """ Returns the container node of a ray's origin. """
    if not intersections: return None
    if len(intersections) == 1: return intersections[0].hit
    nodes = [x.hit for x in intersections]
    counts = collections.Counter(nodes)
    for node, count in counts.items():
        if count % 2 == 1: return node
    return min(intersections, key=lambda x: x.distance).hit


Original code: 
def next_hit(scene, ray):
    """ Returns information about the next interface ... """
    # Intersections are in the local node's coordinate system
    intersections = scene.intersections(ray.position, ray.direction)

    # Remove on surface intersections
    intersections = [x for x in intersections if not close_to_zero(x.distance)]

    # Convert intersection points to world node
    intersections = [x.to(scene.root) for x in intersections]

    # Node which owns the surface
    if len(intersections) == 0:
        return None

    # The surface being hit
    hit = intersections[0]
    if len(intersections) == 1:
        hit_node = hit.hit
        return hit_node, (hit_node, None), hit.point, hit.distance

    container = find_container(intersections)
    hit = intersections[0]
    # Intersection point and distance from ray
    point = hit.point
    hit_node = hit.hit
    distance = distance_between(ray.position, point)
    if container == hit_node:
        adjacent = intersections[1].hit
    else:
        adjacent = hit_node
    return hit_node, (container, adjacent), point, distance
Modified code:
def next_hit(scene, ray):
    """ Returns information about the next physical surface the ray will hit. """
    intersections = scene.intersections(ray.position, ray.direction)
    min_distance = 1e-9
    intersections = [x for x in intersections if x.distance > min_distance]
    
    if not intersections: return None
    world_intersections = sorted([x.to(scene.root) for x in intersections], key=lambda x: x.distance)
    
    for hit in world_intersections:
        if hit.hit is scene.root: continue
        container = find_container(world_intersections)
        adjacent = scene.root
        for other in world_intersections:
            if other.distance > hit.distance + min_distance:
                adjacent = other.hit
                break
        return hit.hit, (container, adjacent), hit.point, hit.distance
    return None


Next_hit ja find_container made simpler. Now they use the even-odd-rule to determine whether a light ray is inside an object or not. 
After each surface interaction, a ray is propagated over a small distance (nudge). This ensures the ray doesn’t get stuck in the surface interacting with it infinitely. 
The main ‘follow’ function is completely rewritten. The original used a while loop that found the next intersection and then decided what to do. My version implements a more event-racing algorithm that independently calculates the distance to the next absorption event and the distance to the next surface, then chooses whichever is closer.

The code in simulation_physics.py holds the Hemisphere geometry.
```
