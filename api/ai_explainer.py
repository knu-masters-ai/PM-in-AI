import os, json, logging, time
from typing import Dict, Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

log = logging.getLogger(__name__)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXPLAINER_DEBUG = os.getenv("EXPLAINER_DEBUG", "0") == "1"

_client = None


def _get_client():
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK is not installed. Add `openai` to requirements.txt")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)
    return _client


def explain_annotated_image_as_json(data_url_png: str, extra_hint: str | None = None) -> dict:
    prompt = (
                 (extra_hint + "\n\n") if extra_hint else ""
             ) + (
                 "Ти — медичний асистент, що пояснює результати виявлення каменів у нирках простою українською. "
                 "На зображенні виділені підозрілі ділянки. "
                 "Сформуй стислий підсумок для користувача, findings[{region,evidence}], next_steps[], "
                 "та коротку clinical_note для лікаря. "
                 "Не став діагноз і не давай лікування. Відповідай СУВОРО у JSON."
             )

    client = _get_client()

    # Прості ретраї
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,  # gpt-4o / gpt-4o-mini
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Ти уважний медичний асистент. Відповідай лише JSON."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url_png}}
                        ],
                    },
                ],
            )
            txt = resp.choices[0].message.content or "{}"
            return json.loads(txt)
        except Exception as e:
            if EXPLAINER_DEBUG:
                log.exception("OpenAI (chat.completions) attempt %d failed", attempt + 1)
            if attempt == 2:
                raise
            time.sleep(1.2 * (attempt + 1))
