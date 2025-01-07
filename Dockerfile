
FROM python:3.9-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY language_configs.json level_details.json interface_languages.json ./
RUN [ -f "language_configs.json" ] && [ -f "level_details.json" ] && [ -f "interface_languages.json" ]

COPY . .


EXPOSE 8080


CMD ["uvicorn", "speech_correction_server:app", "--host", "0.0.0.0", "--port", "8080"]
