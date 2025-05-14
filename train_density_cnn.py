import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.stats import gaussian_kde
from ultralytics import YOLO
import cv2
import glob
import shutil
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----- CẤU HÌNH -----
DATA_DIR = './traffic_analysis_output/processed_for_cnn'
NUM_CLASSES = 3  # Low, Medium, High
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
YOLO_MODEL_NAME = 'yolov8n.pt'
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
YOLO_CONFIDENCE_THRESHOLD = 0.3  # Tăng để cải thiện phát hiện
COUNT_THRESHOLD_LOW_TO_MEDIUM = 5
COUNT_THRESHOLD_MEDIUM_TO_HIGH = 15

# Thiết lập logging
logging.basicConfig(filename='kde_warnings.log', level=logging.WARNING, 
                    format='%(asctime)s - %(message)s')

# ----- CUSTOM DATASET: KDE THỰC TẾ -----
class TrafficDatasetWithKDE(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, yolo_model=None):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)

        # Ép class mapping đúng thứ tự mong muốn
        self.class_to_idx = {'Low': 0, 'Medium': 1, 'High': 2}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.dataset.classes = ['Low', 'Medium', 'High']
        self.dataset.class_to_idx = self.class_to_idx

        # ✅ Cập nhật lại nhãn trong danh sách samples
        self.dataset.samples = [
            (path, self.class_to_idx[os.path.basename(os.path.dirname(path))])
            for path, _ in self.dataset.samples
        ]

        self.transform = transform
        self.yolo_model = yolo_model
        self.root_dir = root_dir


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, _ = self.dataset.samples[idx]
        img_rgb, _ = self.dataset[idx]  # img_rgb: 3 x H x W
        label_name = os.path.basename(os.path.dirname(img_path))
        try:
            label = self.class_to_idx[label_name]
        except KeyError:
            raise KeyError(f"Tên thư mục '{label_name}' không khớp với class_to_idx {self.class_to_idx}. Kiểm tra tên thư mục trong {self.root_dir}.")
        
        centroids = self.get_centroids(img_path, retry_with_lower_conf=False)
        if len(centroids) < 2:
            logging.warning(f"Không đủ centroids ({len(centroids)}) cho {img_path}. Sử dụng KDE mặc định.")
            kde_channel = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        else:
            try:
                x, y = zip(*centroids)
                x, y = np.array(x), np.array(y)
                # Kiểm tra độ lệch chuẩn để đảm bảo dữ liệu đủ đa dạng
                if np.std(x) < 1e-4 or np.std(y) < 1e-4:
                    logging.warning(f"Dữ liệu quá đồng nhất ({len(centroids)} centroids) cho {img_path}. Sử dụng KDE mặc định.")
                    kde_channel = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
                else:
                    # Thêm nhiễu lớn hơn
                    x += np.random.normal(0, 1e-4, len(x))
                    y += np.random.normal(0, 1e-4, len(y))
                    kde = gaussian_kde([x, y], bw_method=0.2)
                    X, Y = np.meshgrid(np.linspace(0, 1, IMAGE_SIZE), np.linspace(0, 1, IMAGE_SIZE))
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    kde_values = kde(positions).reshape(IMAGE_SIZE, IMAGE_SIZE)
                    kde_channel = torch.tensor(kde_values, dtype=torch.float32).unsqueeze(0)
            except np.linalg.LinAlgError as e:
                logging.warning(f"Không thể tính KDE cho {img_path} ({len(centroids)} centroids): {e}. Sử dụng KDE mặc định.")
                kde_channel = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        
        img_4ch = torch.cat([img_rgb, kde_channel], dim=0)  # 4 x H x W
        return img_4ch, label

    def get_centroids(self, img_path, retry_with_lower_conf=False):
        """Tính tọa độ tâm của các hộp giới hạn bằng YOLO."""
        if self.yolo_model is None:
            return []
        img = cv2.imread(img_path)
        if img is None:
            return []
        conf = YOLO_CONFIDENCE_THRESHOLD if not retry_with_lower_conf else 0.2
        results = self.yolo_model(img, verbose=False, conf=conf)
        centroids = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) in VEHICLE_CLASS_IDS:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    centroid_x = (x1 + x2) / 2 / img.shape[1]
                    centroid_y = (y1 + y2) / 2 / img.shape[0]
                    centroids.append([centroid_x, centroid_y])
        # Thử lại với ngưỡng thấp hơn nếu không có centroids
        if len(centroids) == 0 and not retry_with_lower_conf:
            return self.get_centroids(img_path, retry_with_lower_conf=True)
        return centroids

# ----- CUSTOM RESNET50 (4 CHANNELS INPUT) -----
class ResNetWith4Channels(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# ----- HÀM LOAD DỮ LIỆU -----
def load_data():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    yolo_model = YOLO(YOLO_MODEL_NAME)
    train_dataset = TrafficDatasetWithKDE(os.path.join(DATA_DIR, 'train'), transform, yolo_model)
    valid_dataset = TrafficDatasetWithKDE(os.path.join(DATA_DIR, 'valid'), transform, yolo_model)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader

# ----- HÀM KIỂM TRA DỮ LIỆU -----
def audit_dataset():
    """Quét tập dữ liệu để xác định hình ảnh có ít hoặc không có centroids."""
    model = YOLO(YOLO_MODEL_NAME)
    splits = ['train', 'valid']
    audit_results = {'Low': [], 'Medium': [], 'High': []}
    
    for split in splits:
        split_dir = os.path.join(DATA_DIR, split)
        for label in ['Low', 'Medium', 'High']:
            img_paths = glob.glob(os.path.join(split_dir, label, '*.jpg'))
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                results = model(img, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                vehicle_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) in VEHICLE_CLASS_IDS)
                if vehicle_count < 2:
                    audit_results[label].append((img_path, vehicle_count))
    
    # In kết quả kiểm tra
    print("Kết quả kiểm tra tập dữ liệu:")
    for label, issues in audit_results.items():
        if issues:
            print(f"Thư mục {label}:")
            for img_path, count in issues:
                print(f"  - {img_path}: {count} centroids")
    return audit_results

# ----- HÀM KIỂM TRA VÀ SỬA NHÃN DỮ LIỆU -----
def check_and_fix_labels():
    """Kiểm tra và sửa nhãn dữ liệu dựa trên số đếm phương tiện từ YOLO."""
    model = YOLO(YOLO_MODEL_NAME)
    splits = ['train', 'valid']
    for split in splits:
        split_dir = os.path.join(DATA_DIR, split)
        for label in ['Low', 'Medium', 'High']:
            os.makedirs(os.path.join(split_dir, label), exist_ok=True)

        for img_path in glob.glob(os.path.join(split_dir, '*', '*.jpg')):
            img = cv2.imread(img_path)
            if img is None:
                continue
            results = model(img, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
            vehicle_count = sum(1 for result in results for box in result.boxes if int(box.cls[0]) in VEHICLE_CLASS_IDS)
            if vehicle_count <= COUNT_THRESHOLD_LOW_TO_MEDIUM:
                new_label = 'Low'
            elif vehicle_count <= COUNT_THRESHOLD_MEDIUM_TO_HIGH:
                new_label = 'Medium'
            else:
                new_label = 'High'
            new_path = os.path.join(split_dir, new_label, os.path.basename(img_path))
            if img_path != new_path:
                shutil.move(img_path, new_path)
                print(f"Đã di chuyển {img_path} → {new_path}")

# ----- TRAINING -----
def train_model():
    train_loader, valid_loader = load_data()
    model = ResNetWith4Channels(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Train Loss: {total_loss:.4f} | Train Acc: {acc*100:.2f}%")

        # Đánh giá trên tập validation
        # Đánh giá trên tập validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            all_preds = []
            all_labels = []

            for images, labels in valid_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            val_acc = correct / total
            mse = mean_squared_error(all_labels, all_preds)
            mae = mean_absolute_error(all_labels, all_preds)
            r2 = r2_score(all_labels, all_preds)

            print(f"Validation Accuracy: {val_acc*100:.2f}% | MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")


        scheduler.step()

    # Lưu model
    torch.save(model.state_dict(), "traffic_density_cnn.pth")
    print("✅ Đã lưu model vào 'traffic_density_cnn.pth'")

if __name__ == "__main__":
    # Kiểm tra tập dữ liệu
    print("Kiểm tra tập dữ liệu...")
    audit_results = audit_dataset()
    # Kiểm tra và sửa nhãn
    print("Kiểm tra và sửa nhãn dữ liệu...")
    check_and_fix_labels()
    print("Bắt đầu huấn luyện...")
    train_model()