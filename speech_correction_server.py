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
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json

# Load environment variables
load_dotenv()

# Загрузка языковых конфигураций
with open("language_configs.json", "r", encoding="utf-8") as f:
    LANGUAGE_CONFIGS = json.load(f)

# Загрузка уровней
with open("level_details.json", "r", encoding="utf-8") as f:
    LEVEL_DETAILS = json.load(f)

with open("interface_languages.json", "r", encoding="utf-8") as f:
    INTERFACE_LANGUAGES = json.load(f)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler(
    "app.log", maxBytes=10000000, backupCount=5, encoding="utf-8"
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI(
    title="Speech Correction API",
    description="Advanced API for language learning and speech correction",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
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
    interface_language: str
    recognition_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @validator("interface_language")
    def validate_interface_language(cls, v):
        if v not in INTERFACE_LANGUAGES:
            raise ValueError(f"Unsupported interface language: {v}")
        return v

    @validator("language")
    def validate_language(cls, v):
        if v not in LANGUAGE_CONFIGS:
            raise ValueError(
                f"Unsupported language: {v}. Supported languages: {list(LANGUAGE_CONFIGS.keys())}"
            )
        return v

    @validator("level")
    def validate_level(cls, v):
        if v not in LEVEL_DETAILS:
            raise ValueError(f"Unsupported level: {v}")
        return v

    @validator("text")
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
        json_encoders = {str: lambda v: v}


def generate_teacher_prompt(request: CorrectionRequest) -> str:
    """Generates a structured prompt for GPT based on the request parameters with strict language control"""
    level_info = LEVEL_DETAILS[request.level]
    lang_config = LANGUAGE_CONFIGS[request.language]
    interface_lang_config = INTERFACE_LANGUAGES[request.interface_language]

    # Строгие инструкции по языку для GPT
    language_instruction = f"""You are an experienced {request.language} language teacher specializing in {request.level} level.

CRITICAL LANGUAGE REQUIREMENTS:
1. You MUST provide ALL explanations and analysis in {interface_lang_config['name']} (ISO: {interface_lang_config['language_code']})
2. DO NOT use English or any other language for explanations
3. Only the original mistakes and corrected examples should be in {request.language}
4. Even if you see "Deutsch" or any other language name, stick to {interface_lang_config['name']} for explanations

This is a hard requirement - never switch to English or any other language for explanations."""

    prompt = f"""{language_instruction}

Please analyze the provided text according to these rules:

VERY IMPORTANT FORMATTING RULES:
1. The CORRECTED_TEXT section should ONLY contain the corrected text in {request.language} with no translations or explanations.
2. For all explanation sections, use STRICTLY {interface_lang_config['name']} as the main language.
3. When referring to specific words or phrases from the original or corrected text in the explanations, ALWAYS keep them in {request.language} and put them in quotes.
4. Do NOT translate the original text's words/phrases when discussing them - keep them in {request.language}.
5. In the ERROR_STATISTICS section, provide a brief count of errors by category in {interface_lang_config['name']}.
6. In the ALTERNATIVES section, suggest other ways to express the same meaning at the current level, with explanations in {interface_lang_config['name']}.

Current context:
- Level: {request.level}
- Level Description: {level_info['description'].get(interface_lang_config['language_code'], level_info['description']['English'])}
- Common Errors: {', '.join(lang_config['common_errors'])}
- Pronunciation Focus: {', '.join(lang_config['pronunciation_focus'])}
- Grammar Focus: {', '.join(level_info['grammar_focus'])}

Use this EXACT format in your response:

CORRECTED_TEXT:
[Only the corrected version in {request.language}]

ERROR_STATISTICS:
[Statistics in {interface_lang_config['name']}]
- Grammar: [number] errors
- Vocabulary: [number] errors
- Pronunciation: [number] errors
- Other: [number] errors

EXPLANATION:
[Detailed error analysis in {interface_lang_config['name']}, keeping original {request.language} phrases in quotes]

GRAMMAR_NOTES:
[Grammar explanations in {interface_lang_config['name']}, keeping {request.language} examples in quotes]

PRONUNCIATION_TIPS:
[Pronunciation advice in {interface_lang_config['name']}, keeping {request.language} examples in quotes]

ALTERNATIVES:
[2-3 alternative ways to express the same meaning in {request.language}, with brief explanations in {interface_lang_config['name']}]

LEVEL_APPROPRIATE_SUGGESTIONS:
[Level-specific suggestions in {interface_lang_config['name']}, keeping {request.language} examples in quotes]"""

    return prompt


def parse_correction_response(response: str) -> Dict[str, str]:
    """Parses GPT response into structured sections"""
    sections = {
        "corrected_text": "",
        "explanation": "",
        "grammar_notes": "",
        "pronunciation_tips": "",
        "level_appropriate_suggestions": "",
    }

    current_section = None
    content_lines = []

    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("CORRECTED_TEXT:"):
            current_section = "corrected_text"
            content_lines = []
        elif line.startswith("EXPLANATION:"):
            if current_section:
                sections[current_section] = "\n".join(content_lines).strip()
            current_section = "explanation"
            content_lines = []
        elif line.startswith("GRAMMAR_NOTES:"):
            if current_section:
                sections[current_section] = "\n".join(content_lines).strip()
            current_section = "grammar_notes"
            content_lines = []
        elif line.startswith("PRONUNCIATION_TIPS:"):
            if current_section:
                sections[current_section] = "\n".join(content_lines).strip()
            current_section = "pronunciation_tips"
            content_lines = []
        elif line.startswith("LEVEL_APPROPRIATE_SUGGESTIONS:"):
            if current_section:
                sections[current_section] = "\n".join(content_lines).strip()
            current_section = "level_appropriate_suggestions"
            content_lines = []
        else:
            content_lines.append(line)

    if current_section and content_lines:
        sections[current_section] = "\n".join(content_lines).strip()

    return sections


@app.post("/process-text/")
async def process_text(request: CorrectionRequest) -> JSONResponse:
    """Processes the text correction request"""
    start_time = datetime.now()
    logger.info(
        f"Starting text processing. Language: {request.language}, Level: {request.level}"
    )

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured")

        client = OpenAI(api_key=api_key)

        # Generate the prompt
        prompt = generate_teacher_prompt(request)

        try:
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": request.text},
                    ],
                    temperature=0.7,
                    max_tokens=1500,
                )
            )

            response_text = response.choices[0].message.content
            sections = parse_correction_response(response_text)

            response_data = {
                "corrected_text": sections["corrected_text"],
                "explanation": sections["explanation"],
                "grammar_notes": sections["grammar_notes"],
                "pronunciation_tips": sections["pronunciation_tips"],
                "level_appropriate_suggestions": sections[
                    "level_appropriate_suggestions"
                ],
                "original_text": request.text,
            }

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Processing completed in {processing_time:.2f} seconds")

            return JSONResponse(
                content=response_data, media_type="application/json; charset=utf-8"
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
    try:
        # Проверяем наличие API ключа
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "reason": "OpenAI API key not configured",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        # Проверяем наличие конфигурационных файлов
        required_files = [
            "language_configs.json",
            "level_details.json",
            "interface_languages.json",
        ]
        for file in required_files:
            if not os.path.exists(file):
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "reason": f"Configuration file {file} not found",
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        return JSONResponse(
            content={
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.1.0",
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": str(e),
                "timestamp": datetime.now().isoformat(),
            },
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse(
        content={
            "message": "Speech Correction API is running",
            "version": "2.1.0",
            "documentation": "/docs",
        }
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server...")
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Server will run on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")
