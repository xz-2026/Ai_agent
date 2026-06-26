#1.写FastAPI服务:加载模型,定义请求/响应格式。实现/predict端点
import torch
import io
import uvicorn
import nest_asyncio
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

# 解决Jupyter Notebook事件循环冲突
nest_asyncio.apply()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 和训练完全一致的网络结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型权重
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()
print("模型加载成功")

app = FastAPI(title="MNIST 手写数字识别API")

# 完整预处理：裁剪居中 + 黑白反转 + 归一化标准化（和训练完全对齐）
def backend_preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    gray_img = np.array(img, dtype=np.uint8)

    # 1.二值化过滤杂点
    threshold = 30
    binary = np.where(gray_img > threshold, 255, 0).astype(np.uint8)

    # 2.裁剪有效数字区域
    y_coords, x_coords = np.where(binary == 255)
    if len(y_coords) == 0:
        raise ValueError("图片未检测到数字")
    min_y, max_y = y_coords.min(), y_coords.max()
    min_x, max_x = x_coords.min(), x_coords.max()
    crop = binary[min_y:max_y+1, min_x:max_x+1]

    # 3.填充正方形、数字居中
    h, w = crop.shape
    max_side = max(h, w)
    pad_top = (max_side - h) // 2
    pad_bottom = max_side - h - pad_top
    pad_left = (max_side - w) // 2
    pad_right = max_side - w - pad_left
    square = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)

    # 4.缩放至28×28
    pil_square = Image.fromarray(square).resize((28, 28), Image.LANCZOS)
    final_arr = np.array(pil_square, dtype=np.float32)

    # 关键1：黑白反转，匹配MNIST白底黑字
    final_arr = 255.0 - final_arr
    # 关键2：和训练完全一致归一化+标准化
    final_arr = final_arr / 255.0
    mean = 0.1307
    std = 0.3081
    final_arr = (final_arr - mean) / std

    # 构造张量 [batch, channel, H, W]
    img_tensor = torch.tensor(final_arr).unsqueeze(0).unsqueeze(0)
    return img_tensor.to(device)

@app.get("/")
def root():
    return {"message": "MNIST数字识别API", "usage": "POST /predict 上传手写数字图片"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="上传文件为空")

        img_tensor = backend_preprocess(image_bytes)

        with torch.no_grad():
            outputs = model(img_tensor)
            predicted = torch.argmax(outputs, dim=1).item()

        return JSONResponse({
            "prediction": predicted,
            "status": "success"
        })

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理异常：{str(e)}")

# 启动服务
# 本地直接运行脚本时启动服务
if __name__ == "__main__":
    import uvicorn
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
