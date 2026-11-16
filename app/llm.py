import os
import json
import logging

from copy import deepcopy

from google import genai
from google.genai import types
from groq import Groq

from dotenv import load_dotenv
from types import SimpleNamespace

load_dotenv()

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-3.5-flash-lite"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

MAX_OUTPUT_TOKENS = 4000
REQUEST_TIMEOUT_MS = 60000

_client = None
_groq_client = None


# --------------------------------------------------
# Gemini client
# --------------------------------------------------

def get_client():
    """
    Create the Gemini client only when needed.
    """

    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured"
        )

    _client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS
        ),
    )

    return _client


# --------------------------------------------------
# Groq client
# --------------------------------------------------

def get_groq_client():
    """
    Create the Groq client only when needed.
    """

    global _groq_client

    if _groq_client is not None:
        return _groq_client

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured"
        )

    _groq_client = Groq(
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_MS / 1000,
        max_retries=0,
    )

    return _groq_client


# --------------------------------------------------
# Response schema preparation
# --------------------------------------------------

def _prepare_response_schema(response_schema):
    """
    Prepare a Pydantic model or JSON schema.

    Resolve internal references and remove
    additionalProperties recursively.
    """

    if hasattr(response_schema, "model_json_schema"):
        schema = response_schema.model_json_schema()

    elif isinstance(response_schema, dict):
        schema = deepcopy(response_schema)

    else:
        raise TypeError(
            "response_schema must be a Pydantic model "
            "or a JSON schema dictionary."
        )

    definitions = schema.get("$defs", {})

    def resolve_references(value):

        if isinstance(value, dict):

            if "$ref" in value:
                reference = value["$ref"]

                prefix = "#/$defs/"

                if not reference.startswith(prefix):
                    raise ValueError(
                        f"Unsupported schema reference: {reference}"
                    )

                definition_name = reference[len(prefix):]

                if definition_name not in definitions:
                    raise ValueError(
                        f"Schema definition not found: {definition_name}"
                    )

                resolved = deepcopy(
                    definitions[definition_name]
                )

                for key, nested_value in value.items():
                    if key != "$ref":
                        resolved[key] = nested_value

                return resolve_references(resolved)

            return {
                key: resolve_references(nested_value)
                for key, nested_value in value.items()
                if key != "additionalProperties"
                and key not in ("$defs", "$schema")
            }

        if isinstance(value, list):
            return [
                resolve_references(item)
                for item in value
            ]

        return value

    return resolve_references(schema)


# --------------------------------------------------
# Groq schema preparation
# --------------------------------------------------

def _prepare_groq_schema(response_schema):
    """
    Convert the existing schema into standard
    JSON Schema format accepted by Groq.
    """

    schema = _prepare_response_schema(response_schema)

    def normalize(value):

        if isinstance(value, dict):
            normalized = {}

            for key, nested_value in value.items():

                if key in ("title", "default", "examples"):
                    continue

                if key == "type" and isinstance(nested_value, str):
                    normalized[key] = nested_value.lower()

                elif key == "type" and isinstance(nested_value, list):
                    normalized[key] = [
                        item.lower() if isinstance(item, str) else item
                        for item in nested_value
                    ]

                else:
                    normalized[key] = normalize(nested_value)

            return normalized

        if isinstance(value, list):
            return [
                normalize(item)
                for item in value
            ]

        return value

    return normalize(schema)


# --------------------------------------------------
# Skill extraction JSON schemas
# --------------------------------------------------

SKILL_CATEGORIES = [
    "languages",
    "frameworks",
    "databases",
    "cloud_devops",
    "ml_data",
    "tools",
    "other",
]


def _build_skills_schema(score_field):
    """
    Build a simple JSON schema for skill extraction.
    """

    return {
        "type": "OBJECT",
        "properties": {
            "skills": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {
                            "type": "STRING",
                        },
                        "category": {
                            "type": "STRING",
                            "enum": SKILL_CATEGORIES,
                        },
                        score_field: {
                            "type": "NUMBER",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": [
                        "name",
                        "category",
                        score_field,
                    ],
                },
            },
        },
        "required": ["skills"],
    }


def build_resume_skills_schema():
    return _build_skills_schema("confidence_score")


def build_job_skills_schema():
    return _build_skills_schema("importance")


# --------------------------------------------------
# Temporary error detection
# --------------------------------------------------

def _is_temporary_error(error):
    """
    Detect temporary provider errors that may justify
    switching from Gemini to Groq.

    Do not switch for schema, prompt, or validation errors.
    """

    status_code = (
        getattr(error, "status_code", None)
        or getattr(error, "code", None)
    )

    if status_code in (408, 429, 500, 502, 503, 504):
        return True

    error_name = type(error).__name__.lower()
    error_message = str(error).lower()

    temporary_names = (
        "timeout",
        "connectionerror",
        "servererror",
    )

    if any(name in error_name for name in temporary_names):
        return True

    temporary_messages = (
        "resource_exhausted",
        "service unavailable",
        "temporarily unavailable",
        "deadline exceeded",
        "timed out",
        "connection reset",
    )

    return any(
        message in error_message
        for message in temporary_messages
    )


# --------------------------------------------------
# Gemini request
# --------------------------------------------------

def _generate_with_gemini(
    system_prompt,
    user_message,
    json_mode=False,
    response_schema=None,
):
    """
    Generate a response using Gemini.
    """

    config_options = {
        "system_instruction": system_prompt,
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }

    if json_mode:
        config_options["response_mime_type"] = (
            "application/json"
        )

        if isinstance(response_schema, dict):
            config_options["response_json_schema"] = (
                response_schema
            )

        elif response_schema is not None:
            config_options["response_schema"] = (
                _prepare_response_schema(response_schema)
            )

    elif response_schema is not None:
        raise ValueError(
            "response_schema requires json_mode=True"
        )

    response = get_client().models.generate_content(
        model=MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            **config_options
        ),
    )

    return response


# --------------------------------------------------
# Groq request
# --------------------------------------------------

def _generate_with_groq(
    system_prompt,
    user_message,
    json_mode=False,
    response_schema=None,
):
    """
    Generate a response using Groq as a fallback.
    """

    request_options = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
    }

    if json_mode:

        if response_schema is not None:
            groq_schema = _prepare_groq_schema(
                response_schema
            )

            request_options["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ai_response",
                    "strict": False,
                    "schema": groq_schema,
                },
            }

        else:
            request_options["response_format"] = {
                "type": "json_object",
            }

    elif response_schema is not None:
        raise ValueError(
            "response_schema requires json_mode=True"
        )

    response = get_groq_client().chat.completions.create(
        **request_options
    )

    return SimpleNamespace(
    text=response.choices[0].message.content
    )


# --------------------------------------------------
# Main generation with fallback
# --------------------------------------------------

def _generate_response(
    system_prompt,
    user_message,
    json_mode=False,
    response_schema=None,
):
    """
    Try Gemini first.

    If Gemini fails due to a temporary error,
    switch to Groq.
    """

    try:
        return _generate_with_gemini(
            system_prompt,
            user_message,
            json_mode=json_mode,
            response_schema=response_schema,
        )

    except Exception as error:

        if not _is_temporary_error(error):
            raise

        logger.warning(
            "Gemini temporary failure (%s). "
            "Switching to Groq model '%s'.",
            type(error).__name__,
            GROQ_MODEL,
        )

        try:
            return _generate_with_groq(
                system_prompt,
                user_message,
                json_mode=json_mode,
                response_schema=response_schema,
            )

        except Exception:
            logger.exception(
                "Groq fallback failed."
            )
            raise


# --------------------------------------------------
# Text response
# --------------------------------------------------

def ask_llm(system_prompt, user_message):
    """
    Generate a regular text response.
    """

    response = _generate_response(
        system_prompt,
        user_message,
    )

    response_text = response.text

    if not response_text:
        raise ValueError(
            "AI provider returned an empty response"
        )

    return response_text


# --------------------------------------------------
# JSON response
# --------------------------------------------------

def ask_llm_json(
    system_prompt,
    user_message,
    response_schema=None,
):
    """
    Generate and parse a JSON response.

    Gemini is the primary provider.
    Groq is used only after a temporary Gemini failure.
    """

    response = _generate_response(
        system_prompt,
        user_message,
        json_mode=True,
        response_schema=response_schema,
    )

    response_text = response.text

    if not response_text:
        raise ValueError(
            "AI provider returned an empty response"
        )

    try:
        result = json.loads(response_text)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"AI provider returned invalid JSON: {error}"
        ) from error

    if not isinstance(result, dict):
        raise ValueError(
            "AI JSON response must be an object"
        )

    return result