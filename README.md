# 人流辨識系統
利用監視器進行人流辨識，以避免入住人數超過房型限制。

- 人物辨識與追蹤
- 低光度影像增強
- 監視器干擾偵測

## 前置作業

### 開發環境

> [!WARNING]  
> 若您的裝置並未安裝 MSVC 14.0 或以上的版本，請先參考 [此連結](https://blog.csdn.net/bluishglc/article/details/128889935) 的教學進行安裝。

本專案使用 [Python 3.10](https://www.python.org/downloads/release/python-3100/) 與 pip 21.1.1，並建議以虛擬環境安裝其餘套件：

```powershell
# 虛擬環境
python -m venv .venv
.venv/Scripts/activate.ps1
# 套件安裝
pip install -r requirements.txt
```

若裝置有 GPU 可供使用，可再另行安裝支援 CUDA 的 PyTorch 版本：

```powershell
# 請根據 nvidia-smi 的狀態自行設定 CUDA 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu[version]
pip install cupy-cuda[version]
```

### 必要檔案與環境變數

- **中文字型**
  
  本專案預設使用源流明體，請自 [官方出處](https://github.com/ButTaiwan/genryu-font/tree/master) 下載並置於根目錄。
  
## 操作方法

> [!NOTE]  
> 步驟壹、貳屬前置作業，只需操作一遍即可執行主程式。

### 壹、相機角度調整

將物理平面校正為實際的拍攝角度：

```powershell
python util/estimate_cam_para.py -s [影片所在的資料夾路徑] -v [影片的完整名稱]
```

詳細的後續操作請參照原作者 [UCMCTrack](https://github.com/corfyi/UCMCTrack/tree/master?tab=readme-ov-file#-camera-parameter-estimation-tool) 的教學步驟。

### 貳、出入口標記

手動標示畫面上的出入口以便人流追蹤：

```powershell
python util/mark_entrance.py -s [影片所在的資料夾路徑] -v [影片的完整名稱]
```

逐次分別點選出入口所在的**左上角**與**右下角**即可完成註冊，若要重新選取請按 <kbd>c</kbd> 以清除記錄。

待完成所有標記後，請按下 <kbd>q</kbd> 以儲存並退出。

### 參、系統主程式

輸入以下指令後即可開始進行人流追蹤：

```powershell
python demo.py -y [使用的 YOLO 版本] -s [影片所在的資料夾路徑] -v [影片的完整名稱]
```

其餘的參數定義與使用方法請參考

```powershell
python demo.py --help
```
