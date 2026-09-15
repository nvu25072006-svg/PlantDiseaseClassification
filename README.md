# Plant Disease Classification from Leaf Images

Hệ thống phân loại bệnh cây trồng từ ảnh chụp lá cây trên bộ dữ liệu **PlantVillage** sử dụng Học máy cổ điển (`HSV Color Histogram + Logistic Regression`) và Học chuyển giao (`MobileNetV3-Small`, `ResNet18`).

## Cấu trúc thư mục

```text
plant_disease_project/
├── data/
│   └── download_data.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_train.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── demo/
│   └── app.py
├── results/
│   ├── experiment_log.csv
│   ├── training_history.csv
│   ├── misclassified_samples.csv
│   ├── fig3_1_class_distribution.png
│   ├── fig3_2_feature_histograms.png
│   ├── fig3_3_sample_images_grid.png
│   ├── fig4_1_confusion_matrix.png
│   └── fig4_2_learning_curves.png
├── report/
├── requirements.txt
└── README.md
```

## Cài đặt và Chạy thực nghiệm

1. Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

2. Nạp dữ liệu PlantVillage:
```bash
python data/download_data.py --local_path "C:/Users/<User>/Downloads/PlantVillage-Dataset-master.zip"
```

3. Chia tập dữ liệu (`70% Train / 15% Val / 15% Test`, `SEED = 42`) và huấn luyện mô hình:
```bash
python src/train.py --epochs 3 --max_per_class 120
```

4. Chạy ứng dụng Demo (Gradio):
```bash
python demo/app.py
```

## Kết quả thực nghiệm

| Model | Config | Val Acc | Val Macro F1 | Test Acc | Test Precision | Test Recall | Test Macro F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Dummy Classifier | `strategy='most_frequent'` | 12.50% | 0.0278 | 12.50% | 1.56% | 12.50% | 0.0278 |
| HSV Color Hist + LogReg | `HSV 8-bins + StandardScaler + LogReg` | 84.72% | 0.8449 | 79.86% | 80.01% | 79.86% | 0.7966 |
| MobileNetV3-Small | `freeze backbone, lr=1e-3, 3 epochs` | 25.69% | 0.2021 | 22.92% | 24.80% | 22.92% | 0.1664 |
| **ResNet18** | `unfreeze all, lr=3e-4, Aug=True, 3 epochs` | **89.58%** | **0.8904** | **91.67%** | **92.16%** | **91.67%** | **0.9145** |
