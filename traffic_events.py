import csv
import os
import threading
from datetime import datetime, timedelta


_EVENT_LOG_LOCK = threading.Lock()
EVENT_HISTORY_DAYS = 30
FREQUENT_INCIDENTS_FROM = 3


def add_verified_incident(file_path, occurred_at, description):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with _EVENT_LOG_LOCK:
        file_exists = os.path.exists(file_path)
        with open(file_path, 'a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Thời gian', 'Mô tả'])
            writer.writerow([occurred_at.isoformat(timespec='minutes'), description])


def get_incident_summary(file_path, now=None):
    now = now or datetime.now()
    cutoff = now - timedelta(days=EVENT_HISTORY_DAYS)
    incidents = []

    if os.path.exists(file_path):
        with _EVENT_LOG_LOCK, open(file_path, newline='', encoding='utf-8-sig') as file:
            for row in csv.DictReader(file):
                try:
                    occurred_at = datetime.fromisoformat(row['Thời gian'])
                except (KeyError, TypeError, ValueError):
                    continue
                if cutoff <= occurred_at <= now:
                    incidents.append({
                        'time': occurred_at.isoformat(timespec='minutes'),
                        'description': row.get('Mô tả', ''),
                    })

    incidents.sort(key=lambda incident: incident['time'], reverse=True)
    count = len(incidents)
    category = (
        'no_data' if count == 0
        else 'frequent' if count >= FREQUENT_INCIDENTS_FROM
        else 'few'
    )
    return {
        'count_30_days': count,
        'category': category,
        'history_days': EVENT_HISTORY_DAYS,
        'frequent_from': FREQUENT_INCIDENTS_FROM,
        'recent': incidents[:5],
    }
