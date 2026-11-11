EXCEL_CLASS_COLUMNS = (
    (2, 'Car'),
    (3, 'Motorcycle'),
    (5, 'Bus'),
    (7, 'Truck'),
)
TRACK_HOLD_SECONDS = 4.0
TRACK_IOU_THRESHOLD = 0.2
STOPPED_VEHICLE_SECONDS = 60
MOVEMENT_TOLERANCE_RATIO = 0.12
MIN_MOVEMENT_TOLERANCE_PX = 5
SPARSE_TRAFFIC_MAX = 10
CROWDED_TRAFFIC_FROM = 25
STABILITY_WINDOW_SECONDS = 10
STABILITY_MAX_RANGE = 3


def update_vehicle_tracks(detections, tracks, now):
    matched_tracks = set()

    for detection in sorted(detections, key=lambda item: item[4], reverse=True):
        x1, y1, x2, y2, confidence, class_id = detection
        best_track = None
        best_iou = TRACK_IOU_THRESHOLD

        for index, track in enumerate(tracks):
            if (
                index in matched_tracks
                or track['class_id'] != class_id
                or now - track['last_seen'] > TRACK_HOLD_SECONDS
            ):
                continue

            width = max(0, min(x2, track['x2']) - max(x1, track['x1']))
            height = max(0, min(y2, track['y2']) - max(y1, track['y1']))
            intersection = width * height
            detection_area = max(0, x2 - x1) * max(0, y2 - y1)
            track_area = max(0, track['x2'] - track['x1']) * max(0, track['y2'] - track['y1'])
            union = detection_area + track_area - intersection
            iou = intersection / union if union else 0

            if iou > best_iou:
                best_iou = iou
                best_track = index

        updated = {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'confidence': confidence, 'class_id': class_id, 'last_seen': now,
        }
        if best_track is None:
            updated['stationary_since'] = now
            updated['stationary_center'] = ((x1 + x2) / 2, (y1 + y2) / 2)
            tracks.append(updated)
            matched_tracks.add(len(tracks) - 1)
        else:
            previous = tracks[best_track]
            new_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            box_size = min(x2 - x1, y2 - y1)
            tolerance = max(MIN_MOVEMENT_TOLERANCE_PX, box_size * MOVEMENT_TOLERANCE_RATIO)
            stationary_center = previous.get('stationary_center', new_center)
            movement = (
                (new_center[0] - stationary_center[0]) ** 2
                + (new_center[1] - stationary_center[1]) ** 2
            ) ** 0.5
            if movement <= tolerance:
                updated['stationary_since'] = previous.get('stationary_since', now)
                updated['stationary_center'] = stationary_center
            else:
                updated['stationary_since'] = now
                updated['stationary_center'] = new_center
            tracks[best_track] = updated
            matched_tracks.add(best_track)

    active_tracks = [track for track in tracks if now - track['last_seen'] <= TRACK_HOLD_SECONDS]
    for track in active_tracks:
        track['stopped_seconds'] = max(0, now - track.get('stationary_since', now))
    return active_tracks


def assess_traffic(count, samples, now):
    if not samples:
        density = None
    elif count <= SPARSE_TRAFFIC_MAX:
        density = 'sparse'
    elif count >= CROWDED_TRAFFIC_FROM:
        density = 'crowded'
    else:
        density = 'normal'

    window_samples = [
        sample for sample in samples
        if now - sample['timestamp'] <= STABILITY_WINDOW_SECONDS
    ]
    enough_history = (
        len(window_samples) >= 2
        and window_samples[-1]['timestamp'] - window_samples[0]['timestamp']
        >= STABILITY_WINDOW_SECONDS - 1
    )
    if not enough_history:
        stability, count_range = 'collecting', None
    else:
        counts = [sample['count'] for sample in window_samples]
        count_range = max(counts) - min(counts)
        stability = 'stable' if count_range <= STABILITY_MAX_RANGE else 'changing'

    return {
        'density': density,
        'stability': stability,
        'stability_range': count_range,
        'sample_count': len(window_samples),
    }
