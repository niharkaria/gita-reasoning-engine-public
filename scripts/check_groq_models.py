"""Throwaway script -- NOT part of the real codebase, do not commit.

Confirms the REAL current list of models available to this specific
Groq account/API key, rather than trusting search results about Groq's
catalog in general -- model availability can differ by account/tier.
"""

import httpx
from gita_engine.core.config import get_settings

settings = get_settings()
response = httpx.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {settings.groq_api_key}"},
    timeout=30.0,
)
response.raise_for_status()
data = response.json()

print(f"Real models available to this account ({len(data['data'])} total):")
for model in data["data"]:
    print(f"  - {model['id']}")
