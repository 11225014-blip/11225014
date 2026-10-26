# 如何使用Google Colab免費算力訓練自己的模型_google colab算力

## 一、專案介紹

本專案使用 Google Colab 雲端運算環境，搭配 YOLO26 物件偵測模型，實作 AI 影像辨識與模型訓練。

透過 Google Colab 提供的 Python 執行環境與可用的 GPU 資源，即使沒有高效能顯示卡，也能學習 AI 模型的基本操作。

本專案以現成的預訓練模型與 COCO8 範例資料集進行練習，包含圖片物件辨識、模型訓練、結果檢視及模型保存。

## 二、專案目標

1. 學習使用 Google Colab 執行 Python 程式。
2. 了解 YOLO 物件偵測模型的基本原理與使用方式。
3. 使用預訓練模型辨識圖片中的物件。
4. 使用 COCO8 範例資料集進行模型訓練。
5. 觀察訓練結果，並保存模型權重。

## 三、使用工具

| 工具 | 用途 |
|---|---|
| Google Colab | 雲端 Python 執行環境 |
| Python | 撰寫及執行程式 |
| Ultralytics | YOLO 模型載入、訓練與推論 |
| YOLO26 | 影像物件偵測 |
| PyTorch | 深度學習運算框架 |
| COCO8 | 小型範例訓練資料集 |
| Google Drive | 保存訓練後的模型 |
| GitHub | 儲存及分享程式碼與專案文件 |

## 四、實作環境

- 執行平台：Google Colab
- 模型：YOLO26n
- 資料集：COCO8
- 訓練回合數：3 Epochs
- 圖片尺寸：640
- Batch Size：8
- 硬體加速：依 Google Colab 當下提供的資源而定

注意：Google Colab 免費資源的供應情況及可使用時間可能有所不同，不保證每次都能取得相同型號的 GPU。

## 五、實作流程

### 步驟 1：建立 Google Colab 筆記本

開啟 Google Colab，建立新的 Python 筆記本，並依需求啟用 GPU 硬體加速。

Google Colab：https://colab.research.google.com/

### 步驟 2：安裝 YOLO 套件

```python
!pip install -q ultralytics
```

### 步驟 3：載入預訓練模型

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
```

第一次執行時，程式會在需要時下載模型權重。

### 步驟 4：進行圖片物件辨識

使用公開的公車範例圖片進行物件偵測。

```python
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.25,
    save=True
)
```

模型會辨識圖片中的物件，並產生包含偵測框的結果。

### 步驟 5：訓練模型

使用 COCO8 範例資料集進行 3 個 Epoch 的訓練。

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

train_results = model.train(
    data="coco8.yaml",
    epochs=3,
    imgsz=640,
    batch=8,
    project="/content/yolo_project",
    name="first_train",
    exist_ok=True
)
```

COCO8 是用來學習訓練流程的小型資料集，適合初學者測試程式是否能正常執行。

### 步驟 6：查看訓練結果

訓練結果預設儲存在：

```text
/content/yolo_project/first_train/
```

主要輸出包括：

- `weights/best.pt`：訓練過程中選出的最佳模型權重。
- `weights/last.pt`：最後一個訓練回合的模型權重。
- `results.csv`：訓練指標紀錄。
- `results.png`：訓練結果圖表。

實際產生的檔案會依執行狀況而有所不同。

### 步驟 7：保存模型

將訓練完成的模型保存到 Google Drive，避免 Colab 執行環境中斷後遺失。

```python
from google.colab import drive

drive.mount("/content/drive")
```

接著可將 `best.pt` 複製到自己的 Google Drive 資料夾。

## 六、專案檔案

```text
google-colab-yolo-training/
├── README.md
├── yolo_colab_training.ipynb
└── results/
    └── results.png
```

說明：

- `README.md`：專案介紹及操作說明。
- `yolo_colab_training.ipynb`：完整 Colab 實作筆記本。
- `results/results.png`：選擇性保存的訓練結果圖表。

若尚未下載訓練結果圖表，可以先不建立 `results` 資料夾。

## 七、實作成果

本專案完成以下練習：

1. 成功在 Google Colab 執行 YOLO 模型。
2. 使用預訓練模型進行圖片物件偵測。
3. 使用 COCO8 範例資料集執行模型訓練。
4. 產生並查看模型訓練結果。
5. 將訓練完成的模型保存至 Google Drive。

本次使用少量範例資料進行流程驗證，主要目的是學習模型訓練方法，不代表模型已達到正式應用所需的準確度。

## 八、後續改進方向

- 使用自己的圖片建立資料集。
- 學習物件標註及 YOLO 資料集格式。
- 增加訓練資料與 Epoch 數量。
- 評估 Precision、Recall 及 mAP 等指標。
- 將訓練後的模型應用於圖片或影片辨識。

## 九、參考資源

- Google Colab：https://colab.research.google.com/
- Ultralytics 官方文件：https://docs.ultralytics.com/
- YOLO 訓練文件：https://docs.ultralytics.com/modes/train/
- Ultralytics GitHub：https://github.com/ultralytics/ultralytics

## 十、備註

本專案為 AI 影像辨識與模型訓練的學習實作。模型權重及資料集未必會包含在 GitHub 儲存庫中，使用者可依照程式執行時的提示下載必要資源。

