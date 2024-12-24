from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from openai import OpenAI
from datetime import datetime
import os
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CorrectionRequest(BaseModel):
    text: str
    language: str
    level: str
    interface_language: str = "Русский"
    recognition_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class CorrectionResponse(BaseModel):
    corrected_text: str
    explanation: str
    grammar_notes: str
    pronunciation_tips: str
    level_appropriate_suggestions: str
    original_text: str

# Детальная конфигурация уровней языка
LEVEL_DETAILS = {
    "A1": {
        "description": {
            "Русский": "Начальный уровень. Базовые фразы и простые предложения.",
            "English": "Beginner level. Basic phrases and simple sentences.",
            "Deutsch": "Anfängerniveau. Grundlegende Phrasen und einfache Sätze."
        },
        "complexity": "basic",
        "explanation_style": "очень подробный, с примерами",
        "grammar_focus": ["простое настоящее время", "личные местоимения", "базовые предлоги"]
    },
    "A2": {
        "description": {
            "Русский": "Элементарный уровень. Простые диалоги и описания.",
            "English": "Elementary level. Simple dialogues and descriptions.",
            "Deutsch": "Grundlegendes Niveau. Einfache Dialoge und Beschreibungen."
        },
        "complexity": "elementary",
        "explanation_style": "подробный, с аналогиями",
        "grammar_focus": ["прошедшее время", "простое будущее", "базовые союзы"]
    },
    "B1": {
        "description": {
            "Русский": "Средний уровень. Свободное общение на знакомые темы.",
            "English": "Intermediate level. Free communication on familiar topics.",
            "Deutsch": "Mittelstufe. Freie Kommunikation über vertraute Themen."
        },
        "complexity": "intermediate",
        "explanation_style": "детальный, с контекстом",
        "grammar_focus": ["все времена", "условные предложения", "модальные глаголы"]
    },
    "B2": {
        "description": {
            "Русский": "Средне-продвинутый уровень. Сложные темы и абстрактные понятия.",
            "English": "Upper-intermediate level. Complex topics and abstract concepts.",
            "Deutsch": "Höhere Mittelstufe. Komplexe Themen und abstrakte Konzepte."
        },
        "complexity": "upper-intermediate",
        "explanation_style": "академический, с примерами из жизни",
        "grammar_focus": ["сложные времена", "пассивный залог", "косвенная речь"]
    },
    "C1": {
        "description": {
            "Русский": "Продвинутый уровень. Свободное владение языком.",
            "English": "Advanced level. Fluent language proficiency.",
            "Deutsch": "Fortgeschrittenes Niveau. Fließende Sprachkenntnisse."
        },
        "complexity": "advanced",
        "explanation_style": "профессиональный, с литературными примерами",
        "grammar_focus": ["все аспекты грамматики", "стилистические приемы", "идиомы"]
    },
    "C2": {
        "description": {
            "Русский": "Уровень носителя языка. Совершенное владение.",
            "English": "Native-like level. Perfect mastery.",
            "Deutsch": "Muttersprachliches Niveau. Perfekte Beherrschung."
        },
        "complexity": "native-like",
        "explanation_style": "экспертный, с культурным контекстом",
        "grammar_focus": ["все аспекты языка", "диалекты", "профессиональная лексика"]
    }
}

LANGUAGE_CONFIGS = {
    "Русский": {
        "code": "ru",
        "script": "cyrillic",
        "common_errors": ["падежи", "глагольные приставки", "ударения"],
        "pronunciation_focus": ["ь", "ъ", "щ", "ж", "ш"]
    },
    "Немецкий": {
        "code": "de",
        "script": "latin",
        "common_errors": ["артикли", "порядок слов", "составные глаголы"],
        "pronunciation_focus": ["ä", "ö", "ü", "ß"]
    },
    "Английский": {
        "code": "en",
        "script": "latin",
        "common_errors": ["артикли", "времена", "предлоги"],
        "pronunciation_focus": ["th", "w", "r", "schwa"]
    },
    "Украинский": {
        "code": "uk",
        "script": "cyrillic",
        "common_errors": ["наголос", "чергування", "м'який знак"],
        "pronunciation_focus": ["ї", "є", "і", "г"]
    }
}


def generate_teacher_prompt(request: CorrectionRequest) -> str:
    """Генерирует детальный промпт для GPT в стиле преподавателя"""
    level_info = LEVEL_DETAILS[request.level]

    return f"""Ты - опытный преподаватель {request.language} языка, специализирующийся на работе со студентами уровня {request.level}.
Твоя задача - помочь студенту улучшить его языковые навыки через подробный анализ и понятные объяснения.

Уровень студента: {request.level}
Описание уровня: {level_info['description'][request.interface_language]}
Стиль объяснения: {level_info['explanation_style']}
Фокус на грамматике: {', '.join(level_info['grammar_focus'])}

Проанализируй следующий текст и предоставь:

1. ИСПРАВЛЕНО:
Предоставь исправленную версию на {request.language}, сохраняя стиль автора.

2. ОБЪЯСНЕНИЕ:
Дай подробное объяснение на {request.interface_language}, учитывая:
- Основные ошибки и их исправление
- Почему возникли эти ошибки
- Как избежать подобных ошибок в будущем

3. ГРАММАТИКА:
- Разбор использованных конструкций
- Соответствие уровню {request.level}
- Альтернативные варианты выражения мысли

4. ПРОИЗНОШЕНИЕ:
- Сложные звуки и их артикуляция
- Ритм и интонация
- Типичные ошибки произношения для носителей {request.interface_language}

5. РЕКОМЕНДАЦИИ ПО УРОВНЮ:
- Оценка соответствия уровню {request.level}
- Конкретные упражнения для практики
- Следующие шаги в обучении

Важно:
- Пиши доброжелательно и поддерживающе
- Используй примеры из повседневной жизни
- Объясняй ошибки как возможности для улучшения
- Отмечай успешные моменты
- Давай конкретные советы для практики

Все объяснения должны быть на {request.interface_language}, кроме исправленного текста.
"""

@app.post("/process-text/")
async def process_text(request: CorrectionRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")

    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text provided")

        if request.language not in LANGUAGE_CONFIGS:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {request.language}")

        start_time = datetime.now()
        client = OpenAI(api_key=api_key)
        prompt = generate_teacher_prompt(request)

        try:
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": request.text}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
            )
            logger.info(f"Request processed in {(datetime.now() - start_time).total_seconds()}s")
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise HTTPException(status_code=502, detail=str(e))

        sections = parse_correction_response(response.choices[0].message.content)
        return CorrectionResponse(**sections, original_text=request.text)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_correction_response(response: str) -> dict:
    sections = {
        "corrected_text": "",
        "explanation": "",
        "grammar_notes": "",
        "pronunciation_tips": "",
        "level_suggestions": ""
    }

    current_section = None
    lines = []

    for line in response.split('\n'):
        if "ИСПРАВЛЕНО:" in line:
            current_section = "corrected_text"
            lines = []
        elif "ОБЪЯСНЕНИЕ:" in line:
            if current_section:
                sections[current_section] = '\n'.join(lines).strip()
            current_section = "explanation"
            lines = []
        elif "ГРАММАТИКА:" in line:
            if current_section:
                sections[current_section] = '\n'.join(lines).strip()
            current_section = "grammar_notes"
            lines = []
        elif "ПРОИЗНОШЕНИЕ:" in line:
            if current_section:
                sections[current_section] = '\n'.join(lines).strip()
            current_section = "pronunciation_tips"
            lines = []
        elif "РЕКОМЕНДАЦИИ ПО УРОВНЮ:" in line:
            if current_section:
                sections[current_section] = '\n'.join(lines).strip()
            current_section = "level_suggestions"
            lines = []
        elif current_section and line.strip():
            lines.append(line.strip())

    if current_section and lines:
        sections[current_section] = '\n'.join(lines).strip()

    return sections

# Добавляем эндпоинт для проверки здоровья сервера
@app.get("/")
async def root():
    return JSONResponse(content={"message": "API is running. Use /process-text/ for requests."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
