TRAFFIC_CLASSES = [2, 3, 5, 7]
INFERENCE_SIZE = 640
DETECTION_CONFIDENCE = 0.20
EDGE_SCAN_INTERVAL = 30


def detect_vehicles(model, frame, frame_counter):
    results = model.predict(
        frame,
        verbose=False,
        imgsz=INFERENCE_SIZE,
        conf=DETECTION_CONFIDENCE,
        classes=TRAFFIC_CLASSES,
    )
    detections = []
    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((
                x1, y1, x2, y2,
                float(box.conf[0].item()),
                int(box.cls[0].item()),
            ))

    if frame_counter % EDGE_SCAN_INTERVAL:
        return detections

    frame_height, frame_width = frame.shape[:2]
    tile_width = round(frame_width * 0.6)
    tile_height = round(frame_height * 0.6)
    edge_regions = (
        (0, 0),
        (frame_width - tile_width, 0),
        (0, frame_height - tile_height),
        (frame_width - tile_width, frame_height - tile_height),
    )

    for x_start, y_start in edge_regions:
        crop = frame[y_start:y_start + tile_height, x_start:x_start + tile_width]
        result = model.predict(
            crop,
            imgsz=INFERENCE_SIZE,
            conf=DETECTION_CONFIDENCE,
            classes=TRAFFIC_CLASSES,
            verbose=False,
        )[0]
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            candidate = (
                x1 + x_start,
                y1 + y_start,
                x2 + x_start,
                y2 + y_start,
                float(box.conf[0].item()),
                int(box.cls[0].item()),
            )
            if not any(_overlaps_duplicate(candidate, existing) for existing in detections):
                detections.append(candidate)

    return detections


def _overlaps_duplicate(candidate, existing):
    if existing[5] != candidate[5]:
        return False

    candidate_area = max(0, candidate[2] - candidate[0]) * max(0, candidate[3] - candidate[1])
    existing_area = max(0, existing[2] - existing[0]) * max(0, existing[3] - existing[1])
    intersection_width = max(0, min(existing[2], candidate[2]) - max(existing[0], candidate[0]))
    intersection_height = max(0, min(existing[3], candidate[3]) - max(existing[1], candidate[1]))
    intersection = intersection_width * intersection_height
    union_area = existing_area + candidate_area - intersection
    return bool(union_area and intersection / union_area >= 0.5)
