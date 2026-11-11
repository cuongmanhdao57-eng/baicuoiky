from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import time
from ultralytics import YOLO
import threading
import numpy as np
import os
import tempfile
from datetime import datetime
import queue

from traffic_detection import (
    DETECTION_CONFIDENCE,
    INFERENCE_SIZE,
    TRAFFIC_CLASSES,
    detect_vehicles,
)
from traffic_events import add_verified_incident, get_incident_summary
from traffic_logic import (
    EXCEL_CLASS_COLUMNS,
    STOPPED_VEHICLE_SECONDS,
    assess_traffic,
    update_vehicle_tracks,
)
from traffic_reports import init_excel_file, thread_excel_writer

print(">> Đang khởi tạo hệ thống (Camera IP & Video Offline)...")

# --- CẤU HÌNH TỐI ƯU CHO CAMERA ---
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

app = Flask(__name__)
model = YOLO('yolov8n.pt')

# --- THIẾT LẬP ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_FILE_NAME = os.path.join(APP_DIR, "baocao_giao_thong.xlsx")
INCIDENT_LOG_FILE = os.path.join(APP_DIR, "incident_log.csv")
EXCEL_SAVE_INTERVAL = 1

# --- BIẾN TOÀN CỤC ---
cap = None
latest_frame = None
last_extra_boxes = []
vehicle_tracks = []
last_count = 0
last_stopped_count = 0
last_class_counts = {class_id: 0 for class_id, _ in EXCEL_CLASS_COLUMNS}
is_running = False
system_state = "stopped"
current_source = None       
current_video_path = None   

chart_data = [] 
chart_sequence = 0
data_lock = threading.Lock()
frame_lock = threading.Lock()
results_lock = threading.Lock()

capture_thread = None
ai_thread = None
excel_thread = None
excel_queue = queue.Queue()
stop_event = threading.Event()

# --- XỬ LÝ FILE ---
def cleanup_temp_file():
    global current_video_path
    if current_video_path and os.path.exists(current_video_path):
        try:
            os.remove(current_video_path)
            current_video_path = None
        except: pass


def build_excel_assessments(count, stopped_count, now):
    with data_lock:
        samples = list(chart_data)
    assessment = assess_traffic(count, samples, now)
    density_labels = {
        'sparse': 'Vắng',
        'normal': 'Bình thường',
        'crowded': 'Đông',
    }
    stability_labels = {
        'stable': 'Ổn định',
        'changing': 'Đang biến động',
        'collecting': 'Chưa đủ dữ liệu',
    }
    incidents = get_incident_summary(INCIDENT_LOG_FILE)
    return {
        'density': density_labels.get(assessment['density'], 'Chưa đủ dữ liệu'),
        'stability': stability_labels[assessment['stability']],
        'stopped_vehicles': stopped_count,
        'verified_incidents_30_days': incidents['count_30_days'],
    }


def queue_excel_sample(class_counts, stopped_count, sampled_at):
    count = sum(class_counts.values())
    now = datetime.now()
    excel_queue.put({
        'date': now.strftime("%d/%m/%Y"),
        'time': now.strftime("%H:%M:%S"),
        'count': count,
        'class_counts': class_counts,
        **build_excel_assessments(count, stopped_count, sampled_at),
    })


def clear_detection_state(reset_chart_sequence=False):
    global latest_frame, last_extra_boxes, last_count, last_stopped_count
    global last_class_counts, vehicle_tracks, chart_sequence

    with frame_lock:
        latest_frame = None
    with results_lock:
        last_extra_boxes = []
        last_count = last_stopped_count = 0
        last_class_counts = {class_id: 0 for class_id, _ in EXCEL_CLASS_COLUMNS}
        vehicle_tracks = []
    with data_lock:
        chart_data.clear()
        if reset_chart_sequence:
            chart_sequence = 0


def api_error(message, status):
    return jsonify({'status': 'error', 'message': message}), status


# --- XỬ LÝ VIDEO & AUTO RECONNECT ---

def thread_capture_video():
    global latest_frame, cap, current_source, is_running, system_state, last_extra_boxes, last_count, last_stopped_count, last_class_counts, vehicle_tracks, chart_data
    print(f">> Luồng Capture bắt đầu với nguồn: {current_source}")
    is_stream = isinstance(current_source, str) and ("rtsp" in current_source or "http" in current_source)
    source_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None and not is_stream else 0
    frame_interval = 1.0 / min(source_fps, 30) if source_fps and source_fps > 0 else 1.0 / 25
    next_frame_time = time.monotonic()
    
    while not stop_event.is_set():
        if cap is None: time.sleep(1); continue

        ret, frame = cap.read()
        
        if ret:
            system_state = "running"
            with frame_lock:
                if frame.shape[1] > 1280:
                    scale = 1280 / frame.shape[1]
                    frame = cv2.resize(
                        frame,
                        (1280, round(frame.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA
                    )
                latest_frame = frame.copy()
            if not is_stream:
                next_frame_time += frame_interval
                wait_time = next_frame_time - time.monotonic()
                if wait_time > 0:
                    stop_event.wait(wait_time)
                else:
                    next_frame_time = time.monotonic()
            else:
                time.sleep(0.01)
        else:
            if is_stream:
                system_state = "reconnecting"
                print("⚠️ Mất kết nối Camera! Đang thử lại sau 3s...")
                if stop_event.wait(3):
                    break
                try:
                    cap.release()
                    cap = cv2.VideoCapture(current_source)
                    if cap.isOpened():
                        system_state = "connecting"
                        print("✅ Đã kết nối lại!")
                except: pass
            else:
                source_is_camera = isinstance(current_source, int)
                system_state = "error" if source_is_camera else "ended"
                print("⚠️ Không nhận được hình ảnh từ camera." if source_is_camera else ">> File video đã kết thúc.")
                stop_event.set()
                is_running = False
                clear_detection_state()
                break

def thread_process_ai():
    global latest_frame, last_extra_boxes, last_count, last_stopped_count, last_class_counts, chart_data, chart_sequence, vehicle_tracks
    frame_counter = 0
    last_excel_time = time.monotonic()
    last_chart_time = time.time()
    latest_class_counts = None
    current_stopped_count = 0

    while not stop_event.is_set():
        with frame_lock:
            if latest_frame is None: time.sleep(0.05); continue
            frame_to_process = latest_frame.copy()

        frame_counter += 1

        try:
            detections = detect_vehicles(model, frame_to_process, frame_counter)

            with results_lock:
                vehicle_tracks = update_vehicle_tracks(detections, vehicle_tracks, time.monotonic())
                last_extra_boxes = [track.copy() for track in vehicle_tracks]
                current_count = len(vehicle_tracks)
                last_count = current_count
                current_class_counts = {
                    class_id: sum(track['class_id'] == class_id for track in vehicle_tracks)
                    for class_id, _ in EXCEL_CLASS_COLUMNS
                }
                last_class_counts = current_class_counts.copy()
                current_stopped_count = sum(
                    track['stopped_seconds'] >= STOPPED_VEHICLE_SECONDS
                    for track in vehicle_tracks
                )
                last_stopped_count = current_stopped_count

            latest_class_counts = current_class_counts

            if time.time() - last_chart_time >= 1.0:
                time_str = datetime.now().strftime("%H:%M:%S")
                with data_lock:
                    if len(chart_data) >= 50:
                        chart_data.pop(0)
                    chart_sequence += 1
                    chart_data.append({
                        'id': chart_sequence,
                        'time': time_str,
                        'timestamp': time.monotonic(),
                        'count': current_count,
                    })
                last_chart_time = time.time()

            now_monotonic = time.monotonic()
            if now_monotonic - last_excel_time >= EXCEL_SAVE_INTERVAL:
                queue_excel_sample(latest_class_counts, current_stopped_count, now_monotonic)
                latest_class_counts = None
                last_excel_time = now_monotonic

        except Exception as e: print(f"Lỗi AI: {e}")

    if latest_class_counts is not None:
        queue_excel_sample(latest_class_counts, current_stopped_count, time.monotonic())

def stop_system():
    global cap, is_running, system_state
    is_running = False
    system_state = "stopped"
    stop_event.set()
    if cap is not None:
        cap.release()
        cap = None

    if capture_thread and capture_thread.is_alive():
        capture_thread.join(timeout=10)
        if capture_thread.is_alive():
            app.logger.error("Luồng đọc video không dừng sau khi giải phóng nguồn video")
            return False
    if ai_thread and ai_thread.is_alive():
        ai_thread.join(timeout=10)
        if ai_thread.is_alive():
            app.logger.error("Luồng nhận diện không dừng trong thời gian chờ")
            return False
    if excel_thread and excel_thread.is_alive():
        excel_queue.put(None)
        excel_thread.join(timeout=10)
        if excel_thread.is_alive():
            app.logger.error("Luồng ghi Excel không dừng trong thời gian chờ")
            return False
    
    cleanup_temp_file()

    clear_detection_state(reset_chart_sequence=True)
    
    is_running = False
    system_state = "stopped"
    stop_event.clear()
    return True

def start_processing_threads():
    global capture_thread, ai_thread, excel_thread, is_running, system_state
    is_running = True
    system_state = "connecting"
    capture_thread = threading.Thread(target=thread_capture_video, daemon=True)
    ai_thread = threading.Thread(target=thread_process_ai, daemon=True)
    excel_thread = threading.Thread(
        target=thread_excel_writer,
        args=(excel_queue, EXCEL_FILE_NAME, EXCEL_CLASS_COLUMNS),
        daemon=True,
    )
    capture_thread.start()
    ai_thread.start()
    excel_thread.start()

# --- ROUTES ---

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if not is_running:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                _, b = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + b.tobytes() + b'\r\n')
                time.sleep(0.5); continue

            annotated_frame = None
            with frame_lock:
                if latest_frame is not None: annotated_frame = latest_frame.copy()
            
            with results_lock:
                if annotated_frame is not None:
                    for track in last_extra_boxes:
                        x1, y1 = int(track['x1']), int(track['y1'])
                        x2, y2 = int(track['x2']), int(track['y2'])
                        label = f"{model.names[track['class_id']]} {track['confidence']:.2f}"
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
                        cv2.putText(annotated_frame, label, (x1, max(20, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    cv2.rectangle(annotated_frame, (0, 0), (280, 70), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, f"MAT DO: {last_count}", (10, 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            if annotated_frame is not None:
                _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0'}
    )

@app.route('/start', methods=['POST'])
def start():
    global cap, current_source
    if not stop_system():
        return api_error('Nguồn video trước đó chưa dừng được. Vui lòng thử lại.', 503)
    
    src = request.form.get('source', '0')
    src = 0 if src == '0' or src == '' else src
    current_source = src

    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): 
        return jsonify({'status': 'error', 'message': 'Không mở được Camera/Video'})
    
    init_excel_file(EXCEL_FILE_NAME, reset=True)
    start_processing_threads()
    return jsonify({'status': 'ok'})

@app.route('/stop', methods=['POST'])
def stop():
    if not stop_system():
        return api_error('Các luồng xử lý chưa dừng được. Vui lòng thử lại.', 503)
    return jsonify({'status': 'ok'})

@app.route('/upload', methods=['POST'])
def upload():
    global cap, current_source, current_video_path
    f = request.files.get('file')
    if not f or not f.filename:
        return api_error('Vui lòng chọn file video.', 400)

    suffix = os.path.splitext(f.filename)[1].lower()
    if suffix not in {'.mp4', '.avi', '.mov'}:
        return api_error('Định dạng video không được hỗ trợ. Hãy chọn MP4, AVI hoặc MOV.', 400)

    if not stop_system():
        return api_error('Nguồn video trước đó chưa dừng được. Vui lòng thử lại.', 503)

    temp_path = None
    try:
        file_descriptor, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(file_descriptor)
        f.save(temp_path)
    except OSError:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        app.logger.exception("Không thể lưu video tải lên")
        return api_error('Không thể lưu video trên máy chủ. Kiểm tra dung lượng đĩa rồi thử lại.', 500)

    current_video_path = temp_path
    current_source = temp_path

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        cap.release()
        cleanup_temp_file()
        cap = None
        return api_error(
            'Không mở được video. Hãy kiểm tra file có bị lỗi hoặc dùng codec MP4, AVI, MOV được hỗ trợ.',
            422,
        )

    init_excel_file(EXCEL_FILE_NAME, reset=True)
    start_processing_threads()
    return jsonify({'status': 'ok'})

@app.route('/data')
def data():
    with data_lock:
        chart_snapshot = list(chart_data)
    with results_lock:
        count = last_count
        stopped_count = last_stopped_count
        class_counts = last_class_counts.copy()
    assessment = assess_traffic(count, chart_snapshot, time.monotonic())
    return jsonify({
        'state': system_state,
        'count': count,
        'stopped_vehicles': {
            'count': stopped_count,
            'threshold_seconds': STOPPED_VEHICLE_SECONDS,
            'category': 'none' if stopped_count == 0 else 'few' if stopped_count < 3 else 'many',
            'review_recommended': stopped_count > 0,
        },
        'class_counts': class_counts,
        'assessment': assessment,
        'chart': chart_snapshot,
    })

@app.route('/incidents', methods=['GET', 'POST'])
def incidents():
    if request.method == 'GET':
        return jsonify(get_incident_summary(INCIDENT_LOG_FILE))

    payload = request.get_json(silent=True) or {}
    timestamp = payload.get('time', '')
    description = payload.get('description', '')
    try:
        occurred_at = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'message': 'Vui lòng nhập ngày giờ sự cố hợp lệ.'}), 400
    if occurred_at.tzinfo is not None:
        occurred_at = occurred_at.astimezone().replace(tzinfo=None)

    if occurred_at > datetime.now():
        return jsonify({'status': 'error', 'message': 'Thời gian sự cố không được ở tương lai.'}), 400
    if not isinstance(description, str) or not description.strip() or len(description.strip()) > 200:
        return jsonify({'status': 'error', 'message': 'Mô tả cần từ 1 đến 200 ký tự.'}), 400

    try:
        add_verified_incident(INCIDENT_LOG_FILE, occurred_at, description.strip())
    except OSError:
        app.logger.exception("Không thể lưu nhật ký sự cố giao thông")
        return jsonify({'status': 'error', 'message': 'Không thể lưu nhật ký sự cố.'}), 500
    return jsonify({'status': 'ok', 'summary': get_incident_summary(INCIDENT_LOG_FILE)})

@app.route('/download')
def download():
    if os.path.exists(EXCEL_FILE_NAME):
        excel_queue.join()
        response = send_file(EXCEL_FILE_NAME, as_attachment=True, download_name="baocao_giao_thong.xlsx")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response
    return "No Data", 404

if __name__ == '__main__':
    init_excel_file(EXCEL_FILE_NAME)
    print(">> Đang khởi động trước bộ nhận diện...")
    model.predict(
        np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8),
        imgsz=INFERENCE_SIZE,
        conf=DETECTION_CONFIDENCE,
        classes=TRAFFIC_CLASSES,
        verbose=False,
    )
    print(">> Bộ nhận diện đã sẵn sàng.")
    app.run(debug=True, threaded=True, use_reloader=False)