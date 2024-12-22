from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai

import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Замените "*" на список допустимых доменов, если нужно
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        # Определение системного сообщения для языка и уровня
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

        # Обращение к GPT для обработки текста
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Исправь текст: {text}"}
            ],
            max_tokens=500,
            temperature=0.7
        )

        corrected_text = response['choices'][0]['message']['content'].strip()
        error_analysis = f"Исправления выполнены GPT для {language} языка."

    except Exception as e:
        corrected_text = text
        error_analysis = f"Ошибка обработки: {str(e)}"

    return JSONResponse(
        content=CorrectionResult(
            corrected_text=corrected_text,
            error_analysis=error_analysis
        ).dict(),
        media_type="application/json"
    )
@app.get("/")
async def root():
    return {"message": "Speech Correction Server is running!"}
@app.get("/")
def read_root():
    return {"message": "Server is live"}
