# FanFormation
> 為一日球迷所設計的AI工具

FanFormation通過串接大型語言模型與網頁爬蟲，讓語言模型可以檢索當前最新的體育新聞，透過對話回覆使用者的提問。

## 安裝
> 請務必依據你的專案來調整內容。

以下是將此專案安裝到您的電腦上的步驟。建議使用 Python 3.9.13 版本與Docker 27.2.0 版本。

### 取得專案
至`https://github.com/yu-working/FanFormation.git`下載專案檔案

## 設定

### 環境變數設定
將環境變數儲存於 `.env` 檔案中，並將該檔案放置於專案的根目錄下。

```env
# .env file
OPENAI_API_KEY={your OpenAI api key}
```

> [!IMPORTANT] 
> 注意：請確保`.env`欄位名稱與上述格式一致。

### 資料夾說明

- `.env` - 環境變數設定檔
- `app.py` - 主程式
- `Dockerfile` - 測試設定檔

### 建立映像檔
```
# 建立映像檔
docker build -t fanformation -f /path/to/Dockerfile .
```
### 根據映像檔`fanformation`啟動容器
```
# 啟動容器
docker run -d -p 8501:8501 --name fanformation fanformation
```
### 測試
開啟瀏覽器，查看`localhost:8501`，確認網頁是否正確運行
![空白頁](https://github.com/user-attachments/assets/3247eb6e-61ef-40f0-abc2-5c954018de8d)

### 提問
在下方的對話框中輸入你的提問，語言模型將結合新聞時事，彙整回覆
![image](https://github.com/user-attachments/assets/7d063354-627c-4122-9edd-4b3629660bc5)

## 聯絡作者

你可以透過以下方式與我聯絡

- [E-mail : tsaiyuforwork@gmail.com]
