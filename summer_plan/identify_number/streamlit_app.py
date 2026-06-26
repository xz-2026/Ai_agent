import streamlit as st#界面框架
import requests       #网络请求,给后端传图片
from PIL import Image #图像处理
import io             #二进制内存缓存
import numpy as np    #数组像素计算
from streamlit_drawable_canvas import st_canvas
st.set_page_config(page_title="手写数字识别", page_icon="✏️")
st.title("✏️ 手写数字识别")

# 初始化画布key
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None

# 手写画板
canvas_result = st_canvas(
    fill_color="#000000",      #填充色黑色
    stroke_width=20,             #stroke画笔
    stroke_color="#FFFFFF",    #画笔白色
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",     #自由手绘模式
    key=str(st.session_state.canvas_key), #动态key,清空靠这个 str()转换为字符串
)

# 展示上次识别结果
if st.session_state.last_pred is not None:  #只要last_pred不为空则显示
    st.success(f"**上次预测结果：{st.session_state.last_pred}**")

# 纯numpy图像预处理，裁剪+居中，无cv2依赖
def preprocess_mnist_style(gray_img: np.ndarray, target_size=28, threshold=30):
    binary = np.where(gray_img > threshold, 255, 0).astype(np.uint8)#大于30灰度=白色笔画,设置为255
    y_coords, x_coords = np.where(binary == 255)  #提取所有白色笔画像素坐标
    if len(y_coords) == 0:  #画布空白,无笔画返回0
        return None
    #找到数字最小/最大边界,戒掉大量空白黑边
    min_y, max_y = y_coords.min(), y_coords.max()
    min_x, max_x = x_coords.min(), x_coords.max()
    digit_crop = binary[min_y:max_y+1, min_x:max_x+1]#取白边矩阵范围。多余河边去除
    #将数字统一变成正方形
    h, w = digit_crop.shape   
    max_side = max(h, w)                #比如(20,13)最后补成(20,20)
    pad_top = (max_side - h) // 2       #//整数除法
    pad_bottom = max_side - h - pad_top
    pad_left = (max_side - w) // 2      #7/2=3
    pad_right = max_side - w - pad_left #补4
    square_img = np.pad(
        digit_crop,
        ((pad_top, pad_bottom), (pad_left, pad_right)),#上下,左右固定形式
        mode="constant",#填充固定不变的纯色像素。
        constant_values=0 #填充黑色
    )
    pil_img = Image.fromarray(square_img).resize((target_size, target_size), Image.LANCZOS)
    return np.array(pil_img)
col1,col2=st.columns(2)

# 识别按钮
if col1.button("🔍 识别", type="primary", use_container_width=True):
    if canvas_result.image_data is None: 
        # canvas_result.image：画布原始 PIL 图像 canvas_result.json_data：画布上所有线条、图形的矢量坐标信息
        st.warning("请先书写数字！")
    else:
        #取出画布RGBA像素分组
        img_array = canvas_result.image_data.astype(np.uint8)  #image_data 画布完整像素的RGBA 数组。(画布高, 画布宽, 通道数)通道(R,G,B,A)
        # RGBA转灰度
        if img_array.shape[-1] == 4:   #若最后一列通道数为4,则是RGBA图片
            img_gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114]) #img_array[..., :3] 取R,G,B三像素。
        else:
            img_gray = img_array[..., 0]
        img_gray = img_gray.astype(np.uint8) #unsigned int 8  0-255
        img_gray = 255 - img_gray  # 转为白底黑字

        processed = preprocess_mnist_style(img_gray)#上面预处理
        if processed is None:
            st.warning("未检测到数字，请重新书写！")
        else:
            #将numpy数组转PIL图片
            img_pil = Image.fromarray(processed)
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            try:
                url = "http://localhost:8000/predict"
                files = {"file": ("digit.png", img_bytes, "image/png")}
                res = requests.post(url, files=files, timeout=10)
                if res.status_code == 200:
                    pred = res.json()["prediction"]
                    st.session_state.last_pred = pred
                    st.success(f"预测数字：{pred}")
                    st.balloons()
                    st.image(img_pil, caption="模型实际输入图像(28×28)", width=120)
                else:
                    st.error(f"后端错误 {res.status_code}：{res.text}")
            except requests.exceptions.ConnectionError:
                st.error("后端未启动，执行 python api.py")
            except Exception as e:
                st.error(f"请求失败：{str(e)}")
# 清空画布按钮
if col2.button("🗑️ 清除", use_container_width=True):
    st.session_state.canvas_key += 1
    st.session_state.last_pred = None

#cd Desktop\ai notebook\summer_plan\identify_number
#streamlit run streamlit_app.py