from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os

# Проверка наличия API ключа
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not set. "
        "Please set it in your environment variables."
    )

# Инициализация клиента OpenAI
client = OpenAI(api_key=api_key)

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
    """Простой эндпоинт для проверки работоспособности сервера"""
    return {"status": "ok", "message": "Speech correction server is running"}

@app.post("/process-text/")
async def process_text(request: CorrectionRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text field cannot be empty")

    try:
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
        # Более детальная обработка ошибок
        error_msg = str(e)
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

# Обработчик ошибок для некорректных запросов
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=CorrectionResult(
            corrected_text="",
            error_analysis=str(exc.detail)
        ).dict()
    )
