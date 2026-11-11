import os
import queue

from openpyxl import Workbook, load_workbook


REPORT_HEADERS = [
    'Ngày',
    'Giờ',
    'Car',
    'Motorcycle',
    'Bus',
    'Truck',
    'Tổng Số Xe',
    'Mật Độ Giao Thông',
    'Độ Ổn Định',
    'Số Xe Dừng Lâu',
    'Sự Cố Đã Xác Minh (30 Ngày)',
]


def init_excel_file(file_path, reset=False):
    if not reset and os.path.exists(file_path):
        workbook = load_workbook(file_path)
        try:
            sheet = workbook.active
            for column, header in enumerate(REPORT_HEADERS, start=1):
                sheet.cell(row=1, column=column, value=header)
            workbook.save(file_path)
        finally:
            workbook.close()
        return

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Du Lieu Xe"
    sheet.append(REPORT_HEADERS)
    workbook.save(file_path)
    workbook.close()


def thread_excel_writer(data_queue, file_path, class_columns):
    while True:
        try:
            data_item = data_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:
            if data_item is None:
                return

            workbook = load_workbook(file_path)
            try:
                workbook.active.append([
                    data_item['date'],
                    data_item['time'],
                    *(data_item['class_counts'][class_id] for class_id, _ in class_columns),
                    data_item['count'],
                    data_item['density'],
                    data_item['stability'],
                    data_item['stopped_vehicles'],
                    data_item['verified_incidents_30_days'],
                ])
                workbook.save(file_path)
            finally:
                workbook.close()

            print(f"[Excel] Saved: {data_item['time']} - Total: {data_item['count']} vehicles")
        except Exception as error:
            print(f"[Excel] Failed to save report: {error}")
        finally:
            data_queue.task_done()
