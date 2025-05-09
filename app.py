# app.py - Ứng dụng Streamlit phân tích và kiểm tra mật độ giao thông (Web Version)

# --- IMPORT THƯ VIỆN ---
import streamlit as st
import os
import glob # Chỉ dùng cho các hàm không cache, không dùng trong main flow
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import shutil # Không dùng trong main flow của app
import random # Không dùng trong main flow của app
import tempfile # Có thể dùng để lưu file tạm từ upload nếu cần, nhưng Streamlit có thể đọc trực tiếp
import torch
import torchvision.transforms as transforms
from model_utils import ResNetWith4Channels  # Giả định file model_utils.py có sẵn

# --- KHAI BÁO BIẾN TOÀN CỤC VÀ CẤU HÌNH (CẦN CHỈNH SỬA) ---

# --- CẤU HÌNH NGƯỠNG MẬT ĐỘ (QUAN TRỌNG!) ---
# !!! ĐIỀU CHỈNH CÁC NGƯỠNG NÀY DỰA TRÊN KẾT QUẢ PHÂN TÍCH TỪ BƯỚC 5 CỦA SCRIPT main.py !!!
# Logic: Low <= 5, Medium 6-15, High > 15
COUNT_THRESHOLD_LOW_TO_MEDIUM = 5  # <--- THAY ĐỔI GIÁ TRỊ NÀY NẾU CẦN
COUNT_THRESHOLD_MEDIUM_TO_HIGH = 15 # <--- THAY ĐỔI GIÁ TRỊ NÀY NẾU CẦN
# Các ngưỡng này được cố định trong ứng dụng Streamlit này.

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đường dẫn tương đối đến file CSV kết quả phân tích (đã tạo từ script main.py)
# Giả định file này nằm trong cùng thư mục với app.py hoặc trong thư mục con
ANALYSIS_CSV_PATH = './traffic_analysis_output/analysis_results/traffic_analysis_summary.csv' # <-- CHỈNH SỬA ĐƯỜNG DẪN NÀY CHO ĐÚNG VỚI VỊ TRÍ FILE CSV

# Cấu hình YOLO
YOLO_MODEL_NAME = 'yolov8n.pt' # Hoặc 'yolov8s.pt', ... (sẽ tải về lần đầu)
VEHICLE_CLASS_IDS_TO_COUNT = [2, 3, 5, 7] # COCO: 2:car, 3:motorcycle, 5:bus, 7:truck
YOLO_CONFIDENCE_THRESHOLD = 0.3 # Ngưỡng tin cậy cho phát hiện

# Cấu hình CNN
CNN_MODEL_PATH = 'traffic_density_cnn.pth'
CLASS_NAMES = ['Low', 'Medium', 'High']

# Cấu hình KDE (Không tính KDE trong ảnh tải lên cho đơn giản)
# Các biến này chỉ dùng để hiển thị thông tin từ file CSV
KDE_BANDWIDTH = 30
DOWNSCALE_FACTOR_FOR_KDE = 4

# --- HÀM XỬ LÝ DỮ LIỆU ĐƠN LẺ (ĐẾM XE, GÁN NHÃN) ---

# @st.cache_resource giúp load mô hình YOLO một lần duy nhất khi ứng dụng khởi động
@st.cache_resource
def load_yolo_model(model_name):
    """Load mô hình YOLO và cache nó."""
    try:
        model = YOLO(model_name)
        # Không cần st.success ở đây vì nó sẽ hiển thị mỗi lần load, chỉ cần trả về model
        # st.success(f"Đã load mô hình YOLO '{model_name}'.")
        return model
    except Exception as e:
        st.error(f"LỖI khi load mô hình YOLO '{model_name}': {e}")
        st.warning("Vui lòng kiểm tra cài đặt thư viện ultralytics và kết nối internet.")
        st.stop() # Dừng ứng dụng nếu load model lỗi

# Load mô hình CNN
@st.cache_resource
def load_cnn_model():
    """Load mô hình CNN và cache nó."""
    try:
        model = ResNetWith4Channels(num_classes=3)
        model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"LỖI khi load mô hình CNN '{CNN_MODEL_PATH}': {e}")
        st.warning("Vui lòng kiểm tra đường dẫn file mô hình CNN.")
        st.stop()

# --- ĐỊNH NGHĨA HÀM BỊ THIẾU ---
def simple_preprocess_image_streamlit(uploaded_file):
    """Đọc ảnh từ Streamlit UploadedFile."""
    if uploaded_file is not None:
        # Đọc file byte
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Decode ảnh bằng OpenCV
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.warning("Không thể đọc file ảnh. Vui lòng kiểm tra định dạng.")
            return None
        # YOLO và OpenCV làm việc với BGR. cv2.imdecode trả về BGR.
        return img # Trả về ảnh BGR

    return None
# --- KẾT THÚC ĐỊNH NGHĨA HÀM BỊ THIẾU ---

def count_vehicles_and_get_results(image_np, model, target_classes, confidence_thresh):
    """Đếm phương tiện bằng YOLO và trả về số lượng và results."""
    vehicle_count = 0
    if image_np is None or image_np.size == 0:
         return 0, None
    try:
        results = model(image_np, verbose=False, conf=confidence_thresh)
        # Trích xuất box và class_id để đếm
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) in target_classes:
                    vehicle_count += 1
        return vehicle_count, results
    except Exception as e:
        st.error(f"Lỗi khi chạy YOLO trên ảnh: {e}")
        return 0, None

# Sử dụng các biến ngưỡng toàn cục đã khai báo ở đầu script
def assign_density_label(vehicle_count):
    """Gán nhãn mật độ dựa trên số lượng phương tiện và ngưỡng toàn cục."""
    count = int(vehicle_count) # Ép kiểu sang int
    # Sử dụng các biến ngưỡng toàn cục
    if count <= COUNT_THRESHOLD_LOW_TO_MEDIUM:
        return 'Low'
    elif count > COUNT_THRESHOLD_LOW_TO_MEDIUM and count <= COUNT_THRESHOLD_MEDIUM_TO_HIGH: # Logic: từ 6 đến 15
        return 'Medium'
    else: # count > COUNT_THRESHOLD_MEDIUM_TO_HIGH
        return 'High'

def draw_boxes_on_image(image_np, yolo_results, target_classes):
    """Vẽ bounding box lên ảnh từ kết quả YOLO."""
    if image_np is None or yolo_results is None:
        return None

    img_display = image_np.copy()
    # Chuyển đổi sang RGB cho hiển thị trong Streamlit
    if len(img_display.shape) == 3 and img_display.shape[2] == 3:
        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    elif len(img_display.shape) == 2: # Xử lý ảnh grayscale nếu có
         img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
    else: # Trường hợp khác
         st.warning(f"Định dạng ảnh đầu vào không chuẩn BGR/Gray: {img_display.shape}. Thử chuyển đổi mặc định.")
         img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB) # Thử chuyển đổi mặc định

    count_on_image = 0
    if yolo_results:
        for result in yolo_results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in target_classes:
                    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2) # Màu xanh lá cây
                    text_label = str(count_on_image + 1)
                    (text_width, text_height), baseline = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_origin = (max(x1, 0), max(y1 - 5, text_height + 5))
                    # Vẽ text màu trắng
                    cv2.putText(img_display, text_label, text_origin , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    count_on_image += 1

    return img_display # Trả về ảnh đã vẽ box

def predict_density_cnn(image_np_rgb, model, yolo_model):
    """Dự đoán mật độ giao thông bằng mô hình CNN (dùng KDE thật từ YOLO)."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Chuẩn hóa ResNet
    ])

    # Áp dụng transform cho ảnh RGB
    rgb_tensor = transform(image_np_rgb)

    # === Tính KDE thực tế từ YOLO ===
    image_h, image_w = image_np_rgb.shape[:2]
    results = yolo_model(image_np_rgb, verbose=False, conf=0.3)
    centroids = []

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) in VEHICLE_CLASS_IDS_TO_COUNT:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2 / image_w
                cy = (y1 + y2) / 2 / image_h
                centroids.append([cx, cy])

    # Nếu centroids không đủ → dùng tensor 0
    if len(centroids) < 2:
        kde_tensor = torch.zeros(1, 224, 224)
    else:
        try:
            x, y = zip(*centroids)
            x = np.array(x) + np.random.normal(0, 1e-4, len(x))
            y = np.array(y) + np.random.normal(0, 1e-4, len(y))
            kde = gaussian_kde([x, y], bw_method=0.2)
            X, Y = np.meshgrid(np.linspace(0, 1, 224), np.linspace(0, 1, 224))
            positions = np.vstack([X.ravel(), Y.ravel()])
            kde_values = kde(positions).reshape(224, 224)
            kde_tensor = torch.tensor(kde_values, dtype=torch.float32).unsqueeze(0)
        except np.linalg.LinAlgError:
            kde_tensor = torch.zeros(1, 224, 224)

    # Tạo input 4 channel
    input_tensor = torch.cat([rgb_tensor, kde_tensor], dim=0).unsqueeze(0)

    # Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        prob = torch.softmax(output, dim=1)[0][pred].item()

    return CLASS_NAMES[pred], prob

# --- GIAO DIỆN STREAMLIT ---

def main():
    st.set_page_config(
        page_title="Phân loại & Đánh giá Mật độ Giao thông",
        page_icon=":car:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Ứng dụng Phân loại và Đánh giá Mật độ Giao thông")
    st.write("Sử dụng Học sâu (YOLOv8 và CNN) để đếm phương tiện và ước lượng mật độ từ hình ảnh.")

    # Load mô hình YOLO và CNN
    yolo_model = load_yolo_model(YOLO_MODEL_NAME)
    cnn_model = load_cnn_model()

    # Lấy tên lớp từ model YOLO (sau khi model đã load)
    try:
         model_names = yolo_model.names
    except Exception:
         model_names = {cid: f'Class_{cid}' for cid in VEHICLE_CLASS_IDS_TO_COUNT} # Fallback nếu không lấy được tên lớp
    # -- Sidebar ---
    st.sidebar.header("Cấu hình & Thông tin")
    st.sidebar.write(f"Mô hình YOLO: {YOLO_MODEL_NAME}")
    st.sidebar.write(f"Ngưỡng tin cậy YOLO: {YOLO_CONFIDENCE_THRESHOLD}")
    st.sidebar.write(f"Các lớp phương tiện đếm: {', '.join([model_names.get(cid, f'Class_{cid}') for cid in VEHICLE_CLASS_IDS_TO_COUNT])}")
    st.sidebar.write(f"Mô hình CNN: {CNN_MODEL_PATH}")

    st.sidebar.header("Ngưỡng Mật độ")
    # Hiển thị mô tả ngưỡng chính xác
    st.sidebar.write(f"**Low:** <= {COUNT_THRESHOLD_LOW_TO_MEDIUM} phương tiện")
    st.sidebar.write(f"**Medium:** > {COUNT_THRESHOLD_LOW_TO_MEDIUM} và <= {COUNT_THRESHOLD_MEDIUM_TO_HIGH} phương tiện")
    st.sidebar.write(f"**High:** > {COUNT_THRESHOLD_MEDIUM_TO_HIGH} phương tiện")

    st.sidebar.write("*(Các ngưỡng này được cố định trong code ứng dụng, dựa trên phân tích dataset)*")

    # --- Section: Phân tích Tổng quan Dataset ---
    st.header("Phân tích Tổng quan Dataset")
    st.write("Kết quả phân tích trên toàn bộ dataset ban đầu.")

    if os.path.exists(ANALYSIS_CSV_PATH):
        try:
            df_analysis = pd.read_csv(ANALYSIS_CSV_PATH)
            st.write("<h6>Thông tin Thống kê Số lượng Phương tiện và Mật độ KDE</h6>", unsafe_allow_html=True)
            # Ép kiểu lại cột vehicle_count và kde_max_density nếu cần
            try:
                 df_analysis['vehicle_count'] = pd.to_numeric(df_analysis['vehicle_count'], errors='coerce').fillna(0).astype(int)
                 df_analysis['kde_max_density'] = pd.to_numeric(df_analysis['kde_max_density'], errors='coerce').fillna(0.0)
            except Exception as e_csv_coerce:
                 st.warning(f"Lỗi ép kiểu cột trong CSV: {e_csv_coerce}")

            st.dataframe(df_analysis[['vehicle_count', 'kde_max_density', 'kde_mean_density']].describe())

            st.write("<h6>Phân bố Số lượng Phương tiện và Mật độ KDE</h6>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                 fig_count, ax_count = plt.subplots()
                 # Tránh lỗi bins nếu max count quá nhỏ
                 if not df_analysis.empty and df_analysis['vehicle_count'].max() > 0:
                    max_count_for_bins = df_analysis['vehicle_count'].max()
                    bins_count = max(max_count_for_bins // 2, 1) # Đảm bảo bins > 0
                    sns.histplot(df_analysis['vehicle_count'], kde=True, bins=bins_count, ax=ax_count)
                 else:
                    ax_count.text(0.5, 0.5, 'Không có dữ liệu\nphương tiện để vẽ', ha='center', va='center')

                 ax_count.set_title('Phân bố Số lượng Phương tiện')
                 ax_count.set_xlabel('Số lượng Phương tiện')
                 ax_count.set_ylabel('Tần suất')
                 st.pyplot(fig_count)
                 plt.close(fig_count) # Đóng figure để giải phóng bộ nhớ

            with col2:
                 fig_kde, ax_kde = plt.subplots()
                 # Lọc giá trị 0 cho biểu đồ KDE nếu cần
                 kde_data_for_plot = df_analysis[df_analysis['kde_max_density'] > 0]['kde_max_density']
                 if not kde_data_for_plot.empty:
                    sns.histplot(kde_data_for_plot, kde=True, bins=30, ax=ax_kde)
                    ax_kde.set_title('Phân bố Mật độ KDE Tối đa (>0)')
                 else:
                    ax_kde.text(0.5, 0.5, 'Không có dữ liệu KDE\nđể vẽ biểu đồ', ha='center', va='center')
                    ax_kde.set_title('Phân bố Mật độ KDE Tối đa')
                 ax_kde.set_xlabel('Mật độ KDE Tối đa')
                 ax_kde.set_ylabel('Tần suất')
                 st.pyplot(fig_kde)
                 plt.close(fig_kde) # Đóng figure

            # Hiển thị phân phối theo nhãn (tính toán lại dựa trên ngưỡng cố định trong app)
            if 'vehicle_count' in df_analysis.columns:
                # Áp dụng hàm gán nhãn với ngưỡng cố định
                df_analysis['density_label_app'] = df_analysis['vehicle_count'].apply(assign_density_label)
                st.write("<h6>Phân bố Số lượng ảnh theo Nhãn Mật độ (Dựa trên ngưỡng ứng dụng)</h6>", unsafe_allow_html=True)
                st.dataframe(df_analysis['density_label_app'].value_counts())
            else:
                 st.warning("Không tìm thấy cột 'vehicle_count' trong file CSV để tính nhãn.")

        except FileNotFoundError:
             st.error(f"Không tìm thấy file phân tích '{ANALYSIS_CSV_PATH}'.")
             st.info("Vui lòng chạy script phân tích ban đầu (main.py) để tạo file này.")
        except Exception as e:
            st.error(f"Lỗi khi đọc hoặc hiển thị file phân tích '{ANALYSIS_CSV_PATH}': {e}")
            st.warning("Hãy đảm bảo file CSV tồn tại và đúng định dạng.")
    else:
        st.info(f"Không tìm thấy file phân tích '{ANALYSIS_CSV_PATH}'. Vui lòng chạy script phân tích ban đầu (main.py) để tạo file này.")

    # --- Section: Kiểm tra Mật độ trên Ảnh mới ---
    st.header("Kiểm tra Mật độ trên Ảnh mới")
    st.write("Tải lên một file ảnh giao thông để kiểm tra số lượng phương tiện và ước lượng mật độ.")

    uploaded_file = st.file_uploader("Chọn file ảnh (.jpg, .png, etc.)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

    if uploaded_file is not None:
        # Đọc ảnh từ file tải lên (Streamlit UploadedFile)
        image_np_bgr = simple_preprocess_image_streamlit(uploaded_file)

        if image_np_bgr is not None:
            # Streamlit expect RGB, simple_preprocess_image_streamlit trả về BGR
            image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
            st.image(image_np_rgb, caption="Ảnh gốc được tải lên", use_column_width=True)

            # Chạy xử lý khi có ảnh
            st.write("Đang xử lý ảnh...")
            # Pass BGR image_np_bgr to YOLO functions
            vehicle_count, yolo_results = count_vehicles_and_get_results(
                image_np_bgr, yolo_model, VEHICLE_CLASS_IDS_TO_COUNT, YOLO_CONFIDENCE_THRESHOLD
            )

            # Gán nhãn mật độ dựa trên số lượng đếm được và ngưỡng cố định trong app
            density_label = assign_density_label(vehicle_count)

            st.subheader("🔍 Kết quả từ YOLO (Đếm xe)")
            st.write(f"**Số lượng Phương tiện Đếm được:** {vehicle_count}")
            st.write(f"**Mật độ Ước lượng:** {density_label}")

            # Hiển thị ảnh với bounding box
            if yolo_results is not None:
                # Pass BGR image_np_bgr to draw function
                image_with_boxes = draw_boxes_on_image(image_np_bgr, yolo_results, VEHICLE_CLASS_IDS_TO_COUNT)
                if image_with_boxes is not None:
                     # draw_boxes_on_image trả về RGB, phù hợp cho st.image
                     st.image(image_with_boxes, caption=f"Kết quả đếm xe (Tổng: {vehicle_count})", use_column_width=True)

            # Chạy CNN phân tích
            st.subheader("🧠 Kết quả từ CNN (Phân loại ảnh)")
            label, conf = predict_density_cnn(image_np_rgb, cnn_model, yolo_model)
            st.write(f"**Mật độ dự đoán bởi CNN:** {label}")
            st.write(f"**Độ tin cậy:** {conf:.2f}")

            # So sánh kết quả YOLO và CNN
            if label != density_label:
                st.warning(f"⚠️ Mô hình YOLO và CNN cho kết quả khác nhau: YOLO → {density_label}, CNN → {label}")

            st.write("---")
            st.write("*(Lưu ý: Việc đếm xe và ước lượng mật độ dựa trên mô hình YOLOv8n và CNN với các ngưỡng đã xác định trước đó. Độ chính xác có thể thay đổi tùy thuộc vào chất lượng ảnh và điều kiện giao thông.)*")

    else:
        st.info("Vui lòng tải lên một file ảnh để bắt đầu kiểm tra.")

    # Footer tùy chọn
    st.markdown("---")
    st.markdown("Đồ án Deep Learning") # Thay đổi thông tin này

# --- MAIN EXECUTION BLOCK (Chỉ chạy khi script được gọi trực tiếp, không phải khi Streamlit import) ---
if __name__ == "__main__":
    main()