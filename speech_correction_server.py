from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
from openai import OpenAI
from datetime import datetime
import os
import logging
import asyncio
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enhanced logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler(
    'app.log',
    maxBytes=10000000,
    backupCount=5,
    encoding='utf-8'
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI(
    title="Speech Correction API",
    description="Advanced API for language learning and speech correction",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurations
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

class CorrectionRequest(BaseModel):
    text: str
    language: str
    level: str
    interface_language: str = "Русский"
    recognition_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @validator('language')
    def validate_language(cls, v):
        if v not in LANGUAGE_CONFIGS:
            raise ValueError(f"Unsupported language: {v}")
        return v

    @validator('level')
    def validate_level(cls, v):
        if v not in LEVEL_DETAILS:
            raise ValueError(f"Unsupported level: {v}")
        return v

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

class CorrectionResponse(BaseModel):
    corrected_text: str
    explanation: str
    grammar_notes: str
    pronunciation_tips: str
    level_appropriate_suggestions: str
    original_text: str

    class Config:
        json_encoders = {
            str: lambda v: v
        }

def generate_teacher_prompt(request: CorrectionRequest) -> str:
    """Generates a structured prompt for GPT based on the request parameters"""
    level_info = LEVEL_DETAILS[request.level]
    lang_config = LANGUAGE_CONFIGS[request.language]

    return f"""You are an experienced {request.language} language teacher specializing in {request.level} level.
Analyze the following text considering:
- Level: {request.level}
- Level Description: {level_info['description'][request.interface_language]}
- Common Errors: {', '.join(lang_config['common_errors'])}
- Pronunciation Focus: {', '.join(lang_config['pronunciation_focus'])}
- Grammar Focus: {', '.join(level_info['grammar_focus'])}

Provide the analysis in the following EXACT format (these headers are critical for parsing):

CORRECTED_TEXT:
[Corrected version in {request.language}]

EXPLANATION:
[Detailed error explanation in {request.interface_language}]

GRAMMAR_NOTES:
[Grammar analysis in {request.interface_language}]

PRONUNCIATION_TIPS:
[Pronunciation advice in {request.interface_language}]

LEVEL_APPROPRIATE_SUGGESTIONS:
[Level-specific suggestions in {request.interface_language}]

Requirements:
1. Use exactly these headers
2. No empty sections
3. All explanations in {request.interface_language}
4. Be encouraging and supportive
5. Provide practical examples
6. Focus on improvement opportunities"""

def parse_correction_response(response: str) -> Dict[str, str]:
    """Parses GPT response into structured sections"""
    sections = {
        "corrected_text": "",
        "explanation": "",
        "grammar_notes": "",
        "pronunciation_tips": "",
        "level_appropriate_suggestions": ""
    }

    headers_mapping = {
        "CORRECTED_TEXT:": "corrected_text",
        "EXPLANATION:": "explanation",
        "GRAMMAR_NOTES:": "grammar_notes",
        "PRONUNCIATION_TIPS:": "pronunciation_tips",
        "LEVEL_APPROPRIATE_SUGGESTIONS:": "level_appropriate_suggestions"
    }

    current_section = None
    content_lines = []

    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue

        is_header = False
        for header, section_key in headers_mapping.items():
            if line.startswith(header):
                if current_section and content_lines:
                    sections[current_section] = '\n'.join(content_lines).strip()
                current_section = section_key
                content_lines = []
                is_header = True
                break

        if not is_header and current_section:
            content_lines.append(line)

    if current_section and content_lines:
        sections[current_section] = '\n'.join(content_lines).strip()

    # Ensure all sections have content
    for key, value in sections.items():
        if not value.strip():
            sections[key] = f"No {key.replace('_', ' ')} available."

    return sections

@app.post("/process-text/")
async def process_text(request: CorrectionRequest) -> JSONResponse:
    """Processes the text correction request"""
    start_time = datetime.now()
    logger.info(f"Starting text processing. Language: {request.language}, Level: {request.level}")

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured")

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

            response_text = response.choices[0].message.content
            sections = parse_correction_response(response_text)

            response_data = {
                "corrected_text": sections["corrected_text"],
                "explanation": sections["explanation"],
                "grammar_notes": sections["grammar_notes"],
                "pronunciation_tips": sections["pronunciation_tips"],
                "level_appropriate_suggestions": sections["level_appropriate_suggestions"],
                "original_text": request.text
            }

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Text processing completed in {processing_time:.2f} seconds")

            return JSONResponse(
                content=response_data,
                media_type="application/json; charset=utf-8"
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.1.0"
        },
        media_type="application/json; charset=utf-8"
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse(
        content={
            "message": "Speech Correction API is running",
            "version": "2.1.0",
            "documentation": "/docs"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
