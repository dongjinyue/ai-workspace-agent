FROM python:3.12-slim

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# RapidOCR 声明依赖桌面版 OpenCV；服务器容器改用 API 兼容的 headless 版本，
# 避免引入 X11、OpenGL 等 OCR 不需要的桌面系统库。
RUN pip uninstall --yes opencv-python \
    && pip install --no-cache-dir opencv-python-headless==5.0.0.93

COPY --chown=app:app app ./app
COPY --chown=app:app mcp_servers ./mcp_servers

RUN mkdir -p /app/data && chown -R app:app /app/data

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
