# main.py - Script gộp phân tích mật độ giao thông và kiểm tra ảnh mới (Local Version - File Dialog)

# --- HƯỚNG DẪN CÀI ĐẶT THƯ VIỆN TRÊN MÁY TÍNH CÁ NHÂN ---
# Mở Terminal/Command Prompt tại thư mục dự án và chạy:
# python -m venv .venv              # Tạo môi trường ảo (nếu chưa có)
# (Windows) .venv\Scripts\activate    # Kích hoạt môi trường ảo (Windows)
# (macOS/Linux) source .venv/bin/activate # Kích hoạt môi trường ảo (macOS/Linux)
# pip install opencv-python numpy pandas matplotlib scikit-learn ultralytics pillow tqdm seaborn tkinter # Cài đặt tất cả thư viện cần thiết

# --- IMPORT THƯ VIỆN ---
print("--- Đang Import Thư viện ---")
import os
import glob
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shutil
import random

# Import cho tkinter để dùng file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = filedialog = None # Đặt None nếu không import được )
    print("Cảnh báo: Thư viện tkinter không khả dụng. Bước 8 (Chọn file ảnh) sẽ không hoạt động.")


# import tensorflow as tf # Không cần cho script này nếu không huấn luyện CNN
print("Đã Import các thư viện.")

# --- KHAI BÁO BIẾN TOÀN CỤC VÀ CẤU HÌNH (CẦN CHỈNH SỬA) ---

# Đường dẫn đến thư mục dataset gốc trên máy tính
LOCAL_DATASET_PATH = 'D:/DeepLearning/DoAn_DeepLearning/Vehicle_Detection_Image_Dataset' # <--- THAY THẾ DÒNG NÀY

# Thư mục output trên máy tính để lưu kết quả phân tích và dữ liệu đã tổ chức
OUTPUT_BASE_DIR_LOCAL = './traffic_analysis_output' # Lưu trong thư mục dự án hiện tại

# Đường dẫn đầy đủ đến các thư mục con output
OUTPUT_ANALYSIS_DIR = os.path.join(OUTPUT_BASE_DIR_LOCAL, 'analysis_results') # Cho kết quả phân tích CSV, KDE examples
PROCESSED_DATA_FOR_CNN_DIR = os.path.join(OUTPUT_BASE_DIR_LOCAL, 'processed_for_cnn') # Cho dữ liệu đã tổ chức theo nhãn

# Cấu hình YOLO
YOLO_MODEL_NAME = 'yolov8n.pt' # Hoặc 'yolov8s.pt', ... (sẽ tải về nếu chưa có)
VEHICLE_CLASS_IDS_TO_COUNT = [2, 3, 5, 7] # COCO: 2:car, 3:motorcycle, 5:bus, 7:truck
YOLO_CONFIDENCE_THRESHOLD = 0.3 # Ngưỡng tin cậy cho phát hiện

# Cấu hình KDE
KDE_BANDWIDTH = 30 # Thử nghiệm giá trị này
DOWNSCALE_FACTOR_FOR_KDE = 4 # Giảm kích thước KDE map để nhanh hơn, ví dụ: 1 (không giảm), 2, 4
MAX_KDE_EXAMPLES_TO_SAVE = 10 # Số lượng ảnh ví dụ KDE để lưu trong Bước 4

# Cấu hình Trực quan hóa Mẫu Dataset
NUM_SAMPLE_IMAGES_TO_VISUALIZE_DATASET = 8 # Số lượng ảnh mẫu từ dataset để trực quan hóa kết quả

# --- CẤU HÌNH NGƯỠNG MẬT ĐỘ (QUAN TRỌNG! CHỈNH SỬA SAU KHI PHÂN TÍCH Ở BƯỚC 5) ---
# Logic: Low <= 5, Medium 6-15, High > 15
COUNT_THRESHOLD_LOW_TO_MEDIUM = 5  # Ngưỡng trên của Low (inclusive)
COUNT_THRESHOLD_MEDIUM_TO_HIGH = 15 # Ngưỡng trên của Medium (inclusive)
# Các giá trị này sẽ được sử dụng để gán nhãn mật độ.
# HÃY ĐIỀU CHỈNH CHÚNG SAU KHI CHẠY XONG BƯỚC 5 VÀ XEM XÉT BIỂU ĐỒ/THỐNG KÊ.


# --- KHAI BÁO CÁC HÀM XỬ LÝ (TỪ BƯỚC 3) ---

# Hàm đọc ảnh (chỉ dùng đường dẫn file trong phiên bản local này)
def simple_preprocess_image(image_path):
    """Đọc ảnh từ đường dẫn file."""
    if isinstance(image_path, str): # Chỉ chấp nhận đường dẫn file
        img = cv2.imread(image_path)
    else:
        print(f"Cảnh báo: simple_preprocess_image chỉ nhận đường dẫn file. Nhận được kiểu {type(image_path)}")
        return None
    if img is None: print(f"Cảnh báo: Không thể đọc ảnh từ nguồn {image_path}")
    return img


def count_vehicles_and_get_results(image_np, model, target_classes, confidence_thresh):
    """Đếm phương tiện bằng YOLO, trả về số lượng, tọa độ tâm, và results của YOLO."""
    vehicle_count = 0
    vehicle_centers = []
    if image_np is None or image_np.size == 0: return 0, [], None
    try:
        results = model(image_np, verbose=False, conf=confidence_thresh)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in target_classes:
                    vehicle_count += 1
                    x1, y1, x2, y2 = box.xyxy[0]
                    vehicle_centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        return vehicle_count, vehicle_centers, results
    except Exception as e:
        print(f"Lỗi nhỏ khi chạy YOLO: {e}")
        return 0, [], None


def generate_kde_map(image_shape, centers, bandwidth, downscale_factor):
    """Tạo bản đồ mật độ từ tọa độ tâm bằng KDE."""
    if not centers or len(centers) < 2 :
        output_shape = (image_shape[0] // downscale_factor, image_shape[1] // downscale_factor)
        if output_shape[0] <=0 or output_shape[1] <= 0: output_shape = (max(1, image_shape[0]//downscale_factor), max(1, image_shape[1]//downscale_factor))
        return np.zeros(output_shape, dtype=np.float32)
    centers_array = np.array(centers)
    x_coords = centers_array[:, 0] / downscale_factor
    y_coords = centers_array[:, 1] / downscale_factor
    grid_y_max = image_shape[0] // downscale_factor
    grid_x_max = image_shape[1] // downscale_factor
    if grid_x_max <= 0 or grid_y_max <=0: return np.zeros((max(1,grid_y_max), max(1,grid_x_max)), dtype=np.float32)
    xx, yy = np.mgrid[0:grid_x_max:1, 0:grid_y_max:1]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    try:
        if len(np.unique(x_coords)) < 2 or len(np.unique(y_coords)) < 2 or centers_array.shape[0] < min(centers_array.shape[1]+1, 2):
             return np.zeros((grid_y_max, grid_x_max), dtype=np.float32)
        effective_bandwidth = bandwidth / downscale_factor
        kernel = stats.gaussian_kde(np.vstack([x_coords, y_coords]), bw_method=effective_bandwidth / np.mean([grid_x_max, grid_x_max]))
        density_map_flat = kernel(positions)
        density_map = np.reshape(density_map_flat.T, xx.shape)
        return density_map.T # (height, width)
    except Exception as e: return np.zeros((grid_y_max, grid_x_max), dtype=np.float32)

# Hàm gán nhãn mật độ (sử dụng ngưỡng toàn cục đã định nghĩa ở đầu script)
def assign_density_label(vehicle_count):
    """Gán nhãn mật độ dựa trên số lượng phương tiện và ngưỡng toàn cục."""
    count = int(vehicle_count)
    # Sử dụng các biến ngưỡng toàn cục đã định nghĩa ở đầu script
    if count <= COUNT_THRESHOLD_LOW_TO_MEDIUM:
        return 'Low'
    elif count > COUNT_THRESHOLD_LOW_TO_MEDIUM and count <= COUNT_THRESHOLD_MEDIUM_TO_HIGH: # Logic: từ 6 đến 15
        return 'Medium'
    else: # count > COUNT_THRESHOLD_MEDIUM_TO_HIGH
        return 'High'

# Hàm trực quan hóa kết quả (tách ra để dùng lại, sử dụng matplotlib cho phiên bản local)
def visualize_traffic_results(image_np, vehicle_count, density_label, kde_max_density, yolo_results, title):
    """Tạo ảnh trực quan với box, đếm xe, và thông tin mật độ, hiển thị bằng Matplotlib."""
    if image_np is None or image_np.size == 0:
        print(f"Không thể trực quan hóa: Ảnh không hợp lệ cho '{title}'.")
        return

    img_display = image_np.copy()
    # Đảm bảo ảnh là BGR trước khi chuyển đổi (OpenCV đọc BGR)
    if len(img_display.shape) == 3 and img_display.shape[2] == 3:
        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    elif len(img_display.shape) == 2: # Xử lý ảnh grayscale nếu có
         img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
    else: # Trường hợp khác
         print(f"Cảnh báo: Định dạng ảnh không chuẩn BGR/Gray: {img_display.shape}")
         img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB) # Thử chuyển đổi mặc định


    # Vẽ box và số thứ tự xe
    count_on_image = 0
    if yolo_results:
        for result in yolo_results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in VEHICLE_CLASS_IDS_TO_COUNT:
                    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2) # Màu xanh lá cây
                    text_label = str(count_on_image + 1)
                    (text_width, text_height), baseline = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_origin = (max(x1, 0), max(y1 - 5, text_height + 5))
                    cv2.putText(img_display, text_label, text_origin , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # Text trắng
                    count_on_image += 1

    plt.figure(figsize=(10, 8))
    plt.imshow(img_display)
    # Thêm thông tin đếm và nhãn mật độ lên tiêu đề
    title_text = f"{title}\nVehicle Count: {vehicle_count} (Boxed: {count_on_image}) | Density: {density_label}"
    # Chỉ thêm KDE max nếu nó có giá trị và không phải NaN
    if kde_max_density is not None and not np.isnan(kde_max_density) and kde_max_density > 0:
        title_text += f"\nKDE Max Density: {kde_max_density:.4f}"

    plt.title(title_text)
    plt.axis('off')
    plt.show() # Hiển thị ảnh trong cửa sổ Matplotlib


# --- MAIN EXECUTION BLOCK (Chạy khi script được gọi trực tiếp) ---
if __name__ == "__main__":
    print("--- Bắt đầu thực thi script main.py ---")

    # --- BƯỚC 1: KIỂM TRA ĐƯỜNG DẪN DATASET VÀ LOAD MÔ HÌNH YOLO ---
    print("\n--- 1. Kiểm tra Đường dẫn Dataset và Load mô hình YOLO ---")
    # Biến cờ để kiểm soát việc bỏ qua các bước xử lý dataset nếu không tìm thấy dataset
    skip_dataset_processing = False

    # Kiểm tra sự tồn tại của thư mục dataset gốc
    if not os.path.exists(LOCAL_DATASET_PATH):
        print(f"LỖI: Đường dẫn dataset gốc '{LOCAL_DATASET_PATH}' không tồn tại.")
        print("Vui lòng chỉnh sửa biến LOCAL_DATASET_PATH ở đầu script.")
        print("Script sẽ bỏ qua các bước xử lý dataset gốc (4-7).")
        skip_dataset_processing = True
        all_image_paths = [] # Đặt rỗng để các bước xử lý dataset không chạy
    else:
        try:
            # Load mô hình YOLO đã pre-trained
            yolo_model = YOLO(YOLO_MODEL_NAME)
            print(f"Tải mô hình YOLO '{YOLO_MODEL_NAME}' thành công.")

            # --- BƯỚC 2: THU THẬP DANH SÁCH ẢNH TỪ DATASET ---
            print("\n--- 2. Thu thập Danh sách Ảnh từ Dataset ---")
            train_images_dir = os.path.join(LOCAL_DATASET_PATH, 'train', 'images')
            valid_images_dir = os.path.join(LOCAL_DATASET_PATH, 'valid', 'images')

            if not os.path.exists(train_images_dir) or not os.path.exists(valid_images_dir):
                 print(f"LỖI: Không tìm thấy thư mục '{train_images_dir}' hoặc '{valid_images_dir}'.")
                 print(f"Kiểm tra lại cấu trúc dataset tại '{LOCAL_DATASET_PATH}'.")
                 skip_dataset_processing = True
                 all_image_paths = []
            else:
                train_image_paths = glob.glob(os.path.join(train_images_dir, '*.*'))
                valid_image_paths = glob.glob(os.path.join(valid_images_dir, '*.*'))
                all_image_paths = train_image_paths + valid_image_paths

                if not all_image_paths:
                    print("LỖI: Không tìm thấy file ảnh nào trong dataset. Kiểm tra lại thư mục 'images'.")
                    skip_dataset_processing = True
                    all_image_paths = []

            if skip_dataset_processing:
                 print("Bỏ qua các bước xử lý dataset gốc (4-7) do không tìm thấy ảnh hoặc dataset.")
                 # yolo_model sẽ vẫn được load nếu Bước 1 thành công, cần cho Bước 8
            else:
                 print(f"Tìm thấy tổng cộng {len(all_image_paths)} ảnh.")


        except Exception as e_step1_fail:
             print(f"LỖI nghiêm trọng ở Bước 1 hoặc Bước 2: {e_step1_fail}")
             print("Bỏ qua các bước xử lý dataset.")
             skip_dataset_processing = True
             all_image_paths = [] # Đảm bảo all_image_paths rỗng

    # --- BƯỚC 3: HÀM XỬ LÝ ĐÃ ĐỊNH NGHĨA Ở TRÊN ---
    print("\n--- 3. Các hàm xử lý đã được định nghĩa ở đầu script ---")


    # --- BƯỚC 4-7: CHỈ THỰC HIỆN NẾU CÓ DATASET GỐC VÀ LOAD THÀNH CÔNG MODEL YOLO ---
    if not skip_dataset_processing and 'yolo_model' in locals() and yolo_model is not None:

        # --- BƯỚC 4: THỰC HIỆN ĐẾM XE, TẠO KDE VÀ LƯU KẾT QUẢ PHÂN TÍCH ---
        print("\n--- 4. Thực hiện đếm xe, tạo KDE và lưu kết quả phân tích ---")
        os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)
        kde_examples_dir = os.path.join(OUTPUT_ANALYSIS_DIR, 'kde_examples')
        os.makedirs(kde_examples_dir, exist_ok=True)

        processed_data_info = []
        kde_saved_count = 0

        print(f"Bắt đầu xử lý {len(all_image_paths)} ảnh...")
        for i, image_path in enumerate(tqdm(all_image_paths, desc="Bước 4: Đang xử lý ảnh")):
            filename = os.path.basename(image_path)
            split = 'train' if image_path in train_image_paths else 'valid'

            img_np = simple_preprocess_image(image_path)
            if img_np is None: continue

            vehicle_count, vehicle_centers, _ = count_vehicles_and_get_results(
                img_np, yolo_model, VEHICLE_CLASS_IDS_TO_COUNT, YOLO_CONFIDENCE_THRESHOLD
            )

            kde_map = None; max_density_kde = 0.0; mean_density_kde = 0.0
            if vehicle_centers:
                kde_map = generate_kde_map(img_np.shape, vehicle_centers, KDE_BANDWIDTH, DOWNSCALE_FACTOR_FOR_KDE)
                if kde_map is not None and kde_map.size > 0 and kde_map.max() > 0 :
                     max_density_kde = np.max(kde_map); mean_density_kde = np.mean(kde_map)

            processed_data_info.append({
                'filename': filename, 'split': split, 'original_path': image_path,
                'image_height': img_np.shape[0], 'image_width': img_np.shape[1],
                'vehicle_count': vehicle_count, 'vehicle_centers': vehicle_centers, # Lưu cả vehicle_centers
                'kde_max_density': max_density_kde, 'kde_mean_density': mean_density_kde,
            })

            # Lưu một vài ví dụ KDE map
            if kde_map is not None and kde_map.size > 0 and np.any(kde_map) and kde_saved_count < MAX_KDE_EXAMPLES_TO_SAVE:
                try:
                    plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1)
                    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB); plt.imshow(img_rgb); plt.title(f"Original: {filename[:20]}... C:{vehicle_count}");
                    if vehicle_centers: centers_np = np.array(vehicle_centers); plt.scatter(centers_np[:, 0], centers_np[:, 1], s=10, c='r', marker='.')
                    plt.axis('off'); plt.subplot(1, 2, 2)
                    if kde_map.max() > kde_map.min(): plt.imshow(kde_map, cmap='jet', origin='upper')
                    else: plt.imshow(np.zeros_like(kde_map), cmap='gray', origin='upper', vmin=0, vmax=1)
                    plt.title(f"KDE (DS {DOWNSCALE_FACTOR_FOR_KDE}x) MaxD:{max_density_kde:.4f}");
                    if max_density_kde > 0: plt.colorbar(label='Density')
                    plt.axis('off')

                    safe_filename = "".join([c if (c.isalnum() or c in '._-') else "_" for c in filename])
                    kde_example_path = os.path.join(kde_examples_dir, f"kde_{kde_saved_count}_{safe_filename}.png")
                    plt.savefig(kde_example_path); plt.close(); kde_saved_count += 1
                except Exception as e_plot: print(f"Lỗi vẽ/lưu KDE cho {filename}: {e_plot}"); plt.close(); # kde_saved_count +=1 // Không tăng count nếu lỗi vẽ


        df_results = pd.DataFrame(processed_data_info)
        csv_output_path = os.path.join(OUTPUT_ANALYSIS_DIR, 'traffic_analysis_summary.csv')

        try:
            df_results.to_csv(csv_output_path, index=False)
            print(f"\nHoàn tất Bước 4. {len(df_results)} ảnh được xử lý. CSV lưu tại: {csv_output_path}")
            print(f"Ví dụ KDE map (nếu có) được lưu tại: {kde_examples_dir}")
        except Exception as e_csv:
            print(f"LỖI khi lưu file CSV: {e_csv}")

        print("\n--- Kết thúc Bước 4 ---")

        # --- BƯỚC 5: PHÂN TÍCH KẾT QUẢ VÀ XÁC ĐỊNH NGƯỠNG GÁN NHÃN MẬT ĐỘ ---
        print("\n--- 5. Phân tích Kết quả và Xác định Ngưỡng Gán Nhãn Mật độ ---")

        df_analysis = df_results.copy()

        # --- ÉP KIỂU CỘT 'vehicle_count' SANG SỐ NGUYÊN ---
        try:
            df_analysis['vehicle_count'] = pd.to_numeric(df_analysis['vehicle_count'], errors='coerce')
            df_analysis['vehicle_count'] = df_analysis['vehicle_count'].fillna(0).astype(int)
            print("Đã ép kiểu cột 'vehicle_count' sang số nguyên.")
        except Exception as e_coerce:
            print(f"CẢNH BÁO: Lỗi khi ép kiểu cột 'vehicle_count': {e_coerce}")
            print("Kiểm tra lại dữ liệu gốc trong DataFrame.")

        print("Phân tích thống kê các cột số:")
        if not df_analysis.empty:
            print(df_analysis[['vehicle_count', 'kde_max_density', 'kde_mean_density']].describe())

            print("\nBiểu đồ phân bố:")
            if not df_analysis.empty and df_analysis['vehicle_count'].sum() > 0:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                max_count_for_bins = df_analysis['vehicle_count'].max()
                bins_count = max(max_count_for_bins // 2, 1) if max_count_for_bins > 0 else 1
                sns.histplot(df_analysis['vehicle_count'], kde=True, bins=bins_count); plt.title('Phân bố Số lượng Phương tiện'); plt.xlabel('Số lượng Phương tiện'); plt.ylabel('Tần suất')
                plt.subplot(1, 2, 2)
                kde_data_for_plot = df_analysis[df_analysis['kde_max_density'] > 0]['kde_max_density']
                if not kde_data_for_plot.empty:
                   sns.histplot(kde_data_for_plot, kde=True, bins=30); plt.title('Phân bố Mật độ KDE Tối đa (>0)'); plt.xlabel('Mật độ KDE Tối đa'); plt.ylabel('Tần suất')
                else:
                   plt.title('Phân bố Mật độ KDE Tối đa (Không có xe)'); plt.xlabel('Mật độ KDE Tối đa'); plt.ylabel('Tần suất')
                plt.tight_layout(); plt.show()
            else:
                print("Không có dữ liệu phương tiện để vẽ biểu đồ phân bố.")


            # Gán nhãn mật độ vào DataFrame
            if not df_analysis.empty:
                # Sử dụng hàm assign_density_label với ngưỡng toàn cục
                df_analysis['density_label'] = df_analysis['vehicle_count'].apply(assign_density_label)
                print("\nThống kê số lượng ảnh theo nhãn mật độ mới:")
                print(df_analysis['density_label'].value_counts())
            else:
                print("Không có dữ liệu để gán nhãn mật độ.")

            print("\n--- Kết thúc Bước 5 ---")
            print("LƯU Ý: Hãy xem xét kết quả value_counts. Nếu các lớp quá mất cân bằng, bạn có thể cần điều chỉnh ngưỡng ở đầu script (COUNT_THRESHOLD_...).")

            # --- BƯỚC 6: TỔ CHỨC LẠI ẢNH VÀO CÁC THƯ MỤC THEO NHÃN MẬT ĐỘ (CHO BÁO CÁO/KIỂM TRA) ---
            print("\n--- 6. Tổ chức lại Ảnh vào các Thư mục theo Nhãn Mật độ ---")
            print("Thư mục này được tạo ra chủ yếu để kiểm tra và báo cáo.")

            if not df_analysis.empty:
                if os.path.exists(PROCESSED_DATA_FOR_CNN_DIR):
                    print(f"Xóa thư mục cũ '{PROCESSED_DATA_FOR_CNN_DIR}'...")
                    shutil.rmtree(PROCESSED_DATA_FOR_CNN_DIR)
                    print("Đã xóa.")

                for split_type in ['train', 'valid']:
                    split_path = os.path.join(PROCESSED_DATA_FOR_CNN_DIR, split_type)
                    os.makedirs(split_path, exist_ok=True)
                    for label_name in ['Low', 'Medium', 'High']:
                        label_path = os.path.join(split_path, label_name)
                        os.makedirs(label_path, exist_ok=True)

                print("Đã tạo cấu trúc thư mục cho dữ liệu đã tổ chức.")

                print("\nBắt đầu copy ảnh vào các thư mục theo nhãn mật độ...")
                for index, row in tqdm(df_analysis.iterrows(), total=df_analysis.shape[0], desc="Bước 6: Copy ảnh"):
                    original_image_path = row['original_path']
                    density_label = row['density_label']
                    split_type = row['split']

                    if density_label in ['Low', 'Medium', 'High']:
                         destination_dir = os.path.join(PROCESSED_DATA_FOR_CNN_DIR, split_type, density_label)
                         destination_path = os.path.join(destination_dir, row['filename'])

                         if os.path.exists(original_image_path):
                             try: shutil.copyfile(original_image_path, destination_path)
                             except Exception as e_copy: print(f"Lỗi khi copy file {original_image_path}: {e_copy}")
                         # else: print(f"CẢNH BÁO: Không tìm thấy file ảnh gốc {original_image_path}")
                    else:
                         print(f"CẢNH BÁO: Bỏ qua ảnh {row['filename']} với nhãn mật độ không hợp lệ: {density_label}")


                print("\nHoàn tất việc tổ chức lại ảnh.")

                print("\nKiểm tra số lượng file trong các thư mục đã tổ chức:")
                for split_type in ['train', 'valid']:
                    print(f"\nTập {split_type}:")
                    for label_name in ['Low', 'Medium', 'High']:
                        count = len(os.listdir(os.path.join(PROCESSED_DATA_FOR_CNN_DIR, split_type, label_name)))
                        print(f"  Thư mục {label_name}: {count} ảnh")
            else:
                print("Không có dữ liệu để tổ chức lại ảnh.")

            print("\n--- Kết thúc Bước 6 ---")


            # --- BƯỚC 7: TRỰC QUAN HÓA KẾT QUẢ ĐẾM XE VÀ MẬT ĐỘ TRÊN ẢNH MẪU TỪ DATASET ---
            print("\n--- 7. Trực quan hóa Kết quả Đếm xe và Mật độ trên Ảnh mẫu ---")
            print(f"Chọn ngẫu nhiên {NUM_SAMPLE_IMAGES_TO_VISUALIZE_DATASET} ảnh từ dataset để trực quan hóa.")

            if NUM_SAMPLE_IMAGES_TO_VISUALIZE_DATASET > 0 and not df_analysis.empty:
                sample_images_df = df_analysis.sample(n=min(NUM_SAMPLE_IMAGES_TO_VISUALIZE_DATASET, len(df_analysis)), random_state=42)

                print("\nĐang tạo trực quan hóa cho các ảnh mẫu...")
                for index, row in tqdm(sample_images_df.iterrows(), total=len(sample_images_df), desc="Bước 7: Vẽ ảnh mẫu"):
                    image_path = row['original_path']
                    vehicle_count_analysis = row['vehicle_count']
                    density_label_analysis = row['density_label']
                    kde_max_density_analysis = row['kde_max_density']

                    img_np = simple_preprocess_image(image_path)
                    if img_np is None: continue

                    # Chạy YOLO lại chỉ để lấy 'results' để vẽ bounding box
                    try:
                        _, _, results_yolo_rerun = count_vehicles_and_get_results(
                            img_np, yolo_model, VEHICLE_CLASS_IDS_TO_COUNT, YOLO_CONFIDENCE_THRESHOLD
                        )
                        # Visualize
                        visualize_traffic_results(
                            img_np,
                            vehicle_count_analysis,
                            density_label_analysis,
                            kde_max_density_analysis,
                            results_yolo_rerun,
                            title=f"Dataset Sample: {row['filename'][:20]}..."
                        )

                    except Exception as e_viz:
                         print(f"Lỗi khi trực quan hóa ảnh mẫu {row['filename']}: {e_viz}")

            else:
                print("\nKhông có đủ ảnh để trực quan hóa mẫu.")

            print("\n--- Kết thúc Bước 7 ---")

    else: # End of if not skip_dataset_processing
         print("\nBỏ qua các bước xử lý và trực quan hóa dataset gốc (4-7) do không tìm thấy dataset hoặc lỗi load YOLO.")


    # --- BƯỚC 8: KIỂM TRA MẬT ĐỘ VÀ ĐẾM XE TRÊN ẢNH TẢI LÊN (Sử dụng file dialog) ---
    # Chỉ chạy trong môi trường local có tkinter (không phải Colab) VÀ yolo_model đã load thành công (ở Bước 1)
    print("\n--- 8. Kiểm tra Mật độ và Đếm xe trên Ảnh được Chọn ---")
    print("Để kiểm tra ảnh mới, một cửa sổ chọn file sẽ hiện ra.")

    # Kiểm tra xem ngưỡng đã được định nghĩa chưa (từ Bước 5) VÀ có tkinter không VÀ yolo_model đã load thành công
    # Nếu Bước 1 thất bại, yolo_model có thể là None hoặc gây lỗi
    try:
        # Kiểm tra sự tồn tại của các biến ngưỡng từ Bước 5. Sẽ báo NameError nếu chưa chạy Bước 5.
        COUNT_THRESHOLD_LOW_TO_MEDIUM
        COUNT_THRESHOLD_MEDIUM_TO_HIGH

        # Kiểm tra xem tkinter.filedialog có sẵn không
        if tk is None or filedialog is None:
            raise ImportError("Thư viện tkinter không khả dụng hoặc không được cài đặt đúng.")

        # Kiểm tra xem yolo_model đã load thành công chưa (kiểm tra biến yolo_model từ Bước 1)
        if 'yolo_model' not in locals() or yolo_model is None:
             # yolo_model có thể là None nếu Bước 1 thành công nhưng load model thất bại
             raise NameError("Mô hình YOLO chưa được load thành công (Kiểm tra lại Bước 1).")


        print(f"Sử dụng ngưỡng mật độ: Low <= {COUNT_THRESHOLD_LOW_TO_MEDIUM}, Medium > {COUNT_THRESHOLD_LOW_TO_MEDIUM} và <= {COUNT_THRESHOLD_MEDIUM_TO_HIGH}, High > {COUNT_THRESHOLD_MEDIUM_TO_HIGH}")

        # --- BẮT ĐẦU KHỐI TRY LỚN CHO BƯỚC 8 ---
        # Khối try này bao quanh toàn bộ logic xử lý chọn file và ảnh
        try:
            # Tạo một cửa sổ Tkinter gốc (ẩn đi)
            root = tk.Tk()
            root.withdraw() # Ẩn cửa sổ gốc

            # Mở cửa sổ chọn file
            file_paths_to_check = filedialog.askopenfilenames(
                title="Chọn file ảnh để kiểm tra mật độ",
                filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*"))
            )

            if not file_paths_to_check:
                print("Không có file nào được chọn. Bỏ qua phần xử lý ảnh đã chọn.")
            else:
                print(f"Đã chọn {len(file_paths_to_check)} file.")
                # Sử dụng tqdm để hiển thị tiến trình xử lý các file đã chọn
                for image_path_to_check in tqdm(file_paths_to_check, desc="Bước 8: Đang xử lý ảnh chọn"):

                    print(f"\nĐang xử lý ảnh: {image_path_to_check}")

                    # Đọc ảnh từ đường dẫn
                    img_np_check = simple_preprocess_image(image_path_to_check)
                    if img_np_check is None:
                        print(f"LỖI: Không thể đọc ảnh từ đường dẫn '{image_path_to_check}'. Bỏ qua.")
                        continue

                    # Chạy YOLO để đếm xe và lấy results
                    vehicle_count_check, vehicle_centers_check, results_yolo_check = count_vehicles_and_get_results(
                         img_np_check, yolo_model, VEHICLE_CLASS_IDS_TO_COUNT, YOLO_CONFIDENCE_THRESHOLD
                     )

                    # Gán nhãn mật độ dựa trên số lượng đếm được và ngưỡng toàn cục
                    density_label_check = assign_density_label(vehicle_count_check)

                    # Tính KDE max density cho ảnh này (tùy chọn)
                    kde_max_density_check = 0.0
                    if img_np_check.shape[0] > DOWNSCALE_FACTOR_FOR_KDE and img_np_check.shape[1] > DOWNSCALE_FACTOR_FOR_KDE:
                         kde_map_check = generate_kde_map(img_np_check.shape, vehicle_centers_check, KDE_BANDWIDTH, DOWNSCALE_FACTOR_FOR_KDE)
                         if kde_map_check is not None and kde_map_check.size > 0:
                              kde_max_density_check = np.max(kde_map_check) if kde_map_check.max() > 0 else 0.0
                    else:
                         # print("Cảnh báo: Ảnh quá nhỏ để tính KDE.")
                         pass # Bỏ qua tính KDE nếu ảnh nhỏ


                    # --- Hiển thị ảnh với kết quả ---
                    visualize_traffic_results(
                         img_np_check,
                         vehicle_count_check,
                         density_label_check,
                         kde_max_density_check,
                         results_yolo_check,
                         title=f"Checked: {os.path.basename(image_path_to_check)}"
                     )

                    # In kết quả nhận xét
                    print(f"\n--- Kết quả phân tích ảnh '{os.path.basename(image_path_to_check)}' ---")
                    print(f"  Số lượng phương tiện đếm được: {vehicle_count_check}")
                    print(f"  Mật độ ước lượng: {density_label_check}")
                    if kde_max_density_check is not None and not np.isnan(kde_max_density_check) and kde_max_density_check > 0:
                         print(f"  Mật độ KDE tối đa: {kde_max_density_check:.4f}")
                    print("-------------------------------------------------")

        # --- KHỐI FINALLY BÊN TRONG ĐỂ ĐÓNG CỬA SỔ TKINTER ---
        finally: # Luôn chạy khối này sau khối try bên trong
             # Kiểm tra xem biến 'root' có tồn tại trong phạm vi local và khác None không
             if 'root' in locals() and root:
                root.destroy() # Hủy bỏ cửa sổ gốc Tkinter

    # --- BẮT CÁC LỖI TIÊN QUYẾT CHO BƯỚC 8 (KHỐI EXCEPT CỦA TRY LỚN) ---
    # Bắt lỗi NameError (thiếu ngưỡng/model YOLO) hoặc ImportError (thiếu tkinter)
    except (NameError, ImportError) as e_prerequisite:
        if isinstance(e_prerequisite, NameError):
             print("Bỏ qua Bước 8 (Chế độ kiểm tra ảnh được chọn) do thiếu ngưỡng mật độ (Chưa chạy Bước 5 thành công) HOẶC mô hình YOLO chưa load thành công (Bước 1 bị lỗi).")
        elif isinstance(e_prerequisite, ImportError):
             print("Bỏ qua Bước 8 (Chế độ kiểm tra ảnh được chọn) do thư viện tkinter không khả dụng.")
    except Exception as e_step8_fail:
         # Bắt các lỗi không xác định khác xảy ra trong khối try lớn
         print(f"LỖI không xác định khi chạy Bước 8: {e_step8_fail}")
         # Đảm bảo cửa sổ tkinter được đóng ngay cả khi lỗi xảy ra trước finally
         if 'root' in locals() and root:
            root.destroy()


    print("\n--- Script main.py hoàn thành các bước phân tích, trực quan hóa và kiểm tra ảnh mới ---")