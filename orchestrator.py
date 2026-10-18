import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from uagents import Agent, Context
from uagents_core.contrib.protocols.chat import (
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    chat_protocol_spec,
)
from uagents.protocol import Protocol

import litellm
from litellm import acompletion

litellm.suppress_debug_info = True

ASI1_API_KEY = os.getenv("ASI1_API_KEY") or os.getenv("ASI_ONE_API_KEY")
ASI1_BASE_URL = os.getenv("ASI1_BASE_URL", "https://api.asi1.ai/v1")
ASI1_MODEL = os.getenv("ASI1_MODEL", "asi1-mini")
OTHER_AGENT_ADDRESS = os.getenv("OTHER_AGENT_ADDRESS")

PERSONA = (
    "You are a warm, friendly coordinator named Orchestrator. "
    "Speak like a real person, not like an AI or a bot. "
    "Be conversational, natural, and helpful. "
    "You help users and coordinate with other AI agents. "
    "If an agent or feature is not available yet, gently say it is still getting "
    "ready and offer what you can help with instead. "
    "Never mention internal errors, providers, models, or technical details. "
    "Never claim to be an AI model. "
    "Other agent configured: {other}."
).format(other="yes" if OTHER_AGENT_ADDRESS else "no")


def build_agent() -> Agent:
    seed = os.getenv("ORCHESTRATOR_SEED", "orchestrator-default-seed")
    import asyncio
    loop = asyncio.new_event_loop()
    return Agent(name="orchestrator", seed=seed, loop=loop)


agent = build_agent()
chat_proto = Protocol(spec=chat_protocol_spec)


async def call_llm(messages):
    if not ASI1_API_KEY:
        raise RuntimeError("not_ready")
    kwargs = {
        "model": f"openai/{ASI1_MODEL}",
        "messages": messages,
        "api_base": ASI1_BASE_URL,
        "api_key": ASI1_API_KEY,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    resp = await acompletion(**kwargs)
    return (resp.choices[0].message.content or "").strip()


@chat_proto.on_message(ChatAcknowledgement)
async def _ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.debug(f"ack from {sender} for {msg.acknowledged_msg_id}")


@chat_proto.on_message(ChatMessage)
async def _chat(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id
        ),
    )

    text = ""
    try:
        text = (msg.text() or "").strip()
    except Exception:
        for c in msg.content:
            if isinstance(c, TextContent):
                text = (c.text or "").strip()
                break
    if not text:
        return

    history = []
    try:
        for entry in ctx.session_history() or []:
            payload = entry.payload
            if not payload:
                continue
            try:
                hm = ChatMessage.parse_raw(payload)
            except Exception:
                continue
            t = ""
            try:
                t = (hm.text() or "").strip()
            except Exception:
                for c in hm.content:
                    if isinstance(c, TextContent):
                        t = (c.text or "").strip()
                        break
            if not t:
                continue
            role = "assistant" if entry.sender == ctx.agent.address else "user"
            history.append({"role": role, "content": t})
    except Exception:
        pass

    messages = [{"role": "system", "content": PERSONA}]
    messages.extend(history)
    if not history or history[-1].get("content") != text:
        messages.append({"role": "user", "content": text})

    try:
        reply = await call_llm(messages)
        if not reply:
            reply = (
                "Hey, I'm here. Things are still warming up on my end, "
                "but I'm happy to chat about what I can help with so far."
            )
    except Exception:
        reply = (
            "I'm still getting things set up over here, so I'm not quite ready "
            "to dive deep yet. But I'm around—what's on your mind? "
            "I'll do my best with what I have."
        )

    await ctx.send(sender, ChatMessage(content=[TextContent(type="text", text=reply)]))


agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
