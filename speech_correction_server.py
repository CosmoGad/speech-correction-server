from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os

# Инициализация FastAPI
app = FastAPI()

# Разрешение CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели данных
class CorrectionRequest(BaseModel):
    text: str
    language: str
    level: str

class CorrectionResult(BaseModel):
    corrected_text: str
    error_analysis: str

@app.get("/")
async def root():
    """Эндпоинт для проверки работоспособности сервера и переменных окружения"""
    api_key = os.getenv("OPENAI_API_KEY")
    return {
        "status": "ok",
        "message": "Speech correction server is running",
        "api_key_status": "present" if api_key else "missing",
        "api_key_length": len(api_key) if api_key else 0
    }

@app.post("/process-text/")
async def process_text(request: CorrectionRequest):
    # Проверка API ключа при каждом запросе
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse(
            status_code=500,
            content=CorrectionResult(
                corrected_text=request.text,
                error_analysis="OpenAI API key is missing. Please check server environment variables."
            ).dict()
        )

    try:
        # Инициализация клиента OpenAI для каждого запроса
        client = OpenAI(api_key=api_key)

        # Определение системного сообщения на основе языка и уровня
        if language := request.language.lower():
            system_messages = {
                "русский": f"Ты помощник, который исправляет текст на русском языке для уровня {request.level}.",
                "немецкий": f"Du bist ein Helfer, der den Text auf Deutsch für das Niveau {request.level} korrigiert.",
                "английский": f"You are an assistant who corrects text in English for level {request.level}.",
                "украинский": f"Ти помічник, який виправляє текст українською мовою для рівня {request.level}."
            }
            system_message = system_messages.get(
                language,
                f"Ты помощник, который исправляет текст для уровня {request.level}. Язык {request.language} не распознан."
            )

        # Вызов OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Исправь текст: {request.text}"}
            ],
            max_tokens=500,
            temperature=0.7
        )

        corrected_text = response.choices[0].message.content.strip()
        error_analysis = f"The corrections were made for {request.language} text at level {request.level}."

    except Exception as e:
        error_msg = str(e)
        print(f"Error details: {error_msg}")  # Добавляем вывод ошибки в логи

        if "api_key" in error_msg.lower():
            error_msg = "OpenAI API key error. Please check server configuration."
        elif "rate limit" in error_msg.lower():
            error_msg = "Rate limit exceeded. Please try again later."

        return JSONResponse(
            status_code=500,
            content=CorrectionResult(
                corrected_text=request.text,
                error_analysis=f"Error during processing: {error_msg}"
            ).dict()
        )

    return JSONResponse(
        content=CorrectionResult(
            corrected_text=corrected_text,
            error_analysis=error_analysis
        ).dict(),
        media_type="application/json"
    )
