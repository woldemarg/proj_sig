# Built from the repo root (compose.yaml): the same layout as the host, so app.py finds ../data/demo in both
FROM python:3.12-slim
RUN pip install --no-cache-dir fastapi uvicorn httpx pandas shapely jinja2 "h3>=4,<5"
WORKDIR /app/aoi-demo
COPY aoi-demo/app.py aoi-demo/geo.py ./
COPY aoi-demo/static ./static
COPY aoi-demo/prompts ./prompts
COPY data/demo/chicago_crimes_2024.csv data/demo/chicago_bars.csv /app/data/demo/
CMD ["python", "app.py"]
