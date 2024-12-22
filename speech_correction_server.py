from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
import os

# Установка API-ключа OpenAI из переменных окружения
openai.api_key = os.getenv("OPENAI_API_KEY")

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

@app.post("/process-text/")
async def process_text(request: CorrectionRequest):
    text = request.text
    language = request.language
    level = request.level

    try:
        # Определение системного сообщения на основе языка и уровня
        if language == "Русский":
            system_message = f"Ты помощник, который исправляет текст на русском языке для уровня {level}."
        elif language == "Немецкий":
            system_message = f"Du bist ein Helfer, der den Text auf Deutsch für das Niveau {level} korrigiert."
        elif language == "Английский":
            system_message = f"You are an assistant who corrects text in English for level {level}."
        elif language == "Украинский":
            system_message = f"Ти помічник, який виправляє текст українською мовою для рівня {level}."
        else:
            system_message = f"Ты помощник, который исправляет текст для уровня {level}. Язык {language} не распознан, используй общий стиль."

        # Вызов OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Исправь текст: {text}"}
            ],
            max_tokens=500,
            temperature=0.7
        )

        corrected_text = response.choices[0].message['content'].strip()
        error_analysis = f"The corrections were made for {language} text at level {level}."

    except Exception as e:
        # В случае ошибки, вернуть исходный текст и сообщение об ошибке
        corrected_text = text
        error_analysis = f"Error during processing: {str(e)}"

    return JSONResponse(
        content=CorrectionResult(
            corrected_text=corrected_text,
            error_analysis=error_analysis
        ).dict(),
        media_type="application/json"
    )
