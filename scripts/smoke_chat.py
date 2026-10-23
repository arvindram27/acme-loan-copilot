"""One-shot chat smoke test. Run: uv run python scripts/smoke_chat.py"""

import sys

from agentmem.providers.chat import ChatClient
from agentmem.settings import get_settings

prompt = sys.argv[1] if len(sys.argv) > 1 else "Reply with the single word: ready."

if not get_settings().has_api_key:
    print("MEM_API_KEY not set — put your key in agent-memory/.env first.")
    sys.exit(1)

out = ChatClient().complete([{"role": "user", "content": prompt}])
print(f"model: {out.model}")
print(f"usage: prompt={out.prompt_tokens} completion={out.completion_tokens}")
print(f"reply: {out.text}")
