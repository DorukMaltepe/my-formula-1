FROM python:3.13-slim
# libgomp1: OpenMP runtime that LightGBM's wheel needs
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py tuned_params.json ./
# .cache holds downloaded seasons; owned by the app user so the mounted volume is writable
RUN useradd --create-home app && mkdir .cache && chown app .cache
USER app
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
# 0.0.0.0 is inside the container only; docker-compose.yml publishes it to 127.0.0.1 on the host.
# viewer toolbar hides developer options, no file watcher since code is baked in, XSRF/CORS checks stay on
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false", "--client.toolbarMode=viewer", "--server.fileWatcherType=none", \
     "--server.enableXsrfProtection=true", "--server.enableCORS=true"]
