"""Native first-party cloud adapters. Implemented: Anthropic Messages API (`anthropic_messages`), OpenAI
Responses API (`openai_responses`), Gemini generateContent (`gemini_generate`), Mistral chat completions
(`mistral_chat`), xAI Responses API (`xai_responses`, reusing the OpenAI adapter), DeepSeek chat completions
(`deepseek_chat`), GLM chat completions on Z.ai (`glm_chat`), TypeSafe Jev System One (`typesafe_jev`).

Each adapter must capture the serialized request at the transport boundary after SDK serialization,
redact authentication fields before persistence, disable hidden SDK retries (or instrument every wire
attempt), and never substitute a model or change a generation parameter.
"""
