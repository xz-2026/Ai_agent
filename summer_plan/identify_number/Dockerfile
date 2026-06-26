FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# pip清华源加速安装
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# 这里改成 app:app 第一个app是文件名app.py，第二个app是代码里的app实例