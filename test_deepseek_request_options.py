"""Offline checks for the DeepSeek request configuration.

Run directly with `python test_deepseek_request_options.py`.
"""

import asyncio
from types import SimpleNamespace

import speech_correction_server as server


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            async def chunks():
                yield SimpleNamespace(
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(content='{"corrected_text":"ok"}')
                    )],
                    usage=None,
                )
                yield SimpleNamespace(choices=[], usage=SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=6,
                ))

            return chunks()
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content='{"corrected_text":"ok"}'),
            )],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=6),
        )


class _FakeClient:
    def __init__(self):
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_non_streaming_disables_thinking():
    client = _FakeClient()
    response = asyncio.run(server._call_deepseek(client, "prompt", "text"))
    assert response == '{"corrected_text":"ok"}'
    call = client.completions.calls[0]
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert call.get("stream") is not True


def test_streaming_disables_thinking_and_collects_usage():
    client = _FakeClient()

    async def collect():
        return [part async for part in server._call_deepseek_stream(client, "prompt", "text")]

    assert asyncio.run(collect()) == ['{"corrected_text":"ok"}']
    call = client.completions.calls[0]
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert call["stream_options"] == {"include_usage": True}


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
