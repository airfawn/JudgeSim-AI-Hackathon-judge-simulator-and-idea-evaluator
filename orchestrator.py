import os
import asyncio
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

IDEA_GENERATOR_ADDRESS = os.getenv("IDEA_GENERATOR_ADDRESS") or os.getenv("GENERATOR_AGENT_ADDRESS")

_judges_env = os.getenv("JUDGE_ADDRESSES") or os.getenv("JUDGES") or ""
JUDGE_ADDRESSES = [a.strip() for a in _judges_env.split(",") if a.strip()]
for i in range(1, 6):
    addr = os.getenv(f"JUDGE{i}_ADDRESS")
    if addr:
        JUDGE_ADDRESSES.append(addr.strip())
# de-dup preserving order
_seen = set()
JUDGE_ADDRESSES = [a for a in JUDGE_ADDRESSES if not (a in _seen or _seen.add(a))]

PERSONA = (
    "You are a warm, friendly coordinator named Orchestrator. "
    "Speak like a real person, not like an AI or a bot. "
    "Be conversational, natural, and helpful. "
    "You help users generate ideas and analyse ideas using multiple judges. "
    "If an agent or feature is not available yet, gently say it is still getting "
    "ready and offer what you can help with instead. "
    "Never mention internal errors, providers, models, or technical details. "
    "Never claim to be an AI model."
)


def build_agent() -> Agent:
    seed = os.getenv("ORCHESTRATOR_SEED", "orchestrator-default-seed")
    loop = asyncio.new_event_loop()
    return Agent(name="orchestrator", seed=seed, loop=loop)


agent = build_agent()
chat_proto = Protocol(spec=chat_protocol_spec)


async def llm_call(messages, temperature=0.7, max_tokens=1024):
    if not ASI1_API_KEY:
        raise RuntimeError("not_ready")
    kwargs = {
        "model": f"openai/{ASI1_MODEL}",
        "messages": messages,
        "api_base": ASI1_BASE_URL,
        "api_key": ASI1_API_KEY,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = await acompletion(**kwargs)
    return (resp.choices[0].message.content or "").strip()


async def classify_intent(text: str) -> str:
    try:
        out = await llm_call(
            [
                {
                    "role": "system",
                    "content": (
                        "Classify the user request into exactly one label: "
                        "GENERATE_IDEA, ANALYSE_IDEAS, or CHAT. "
                        "Reply with only the label."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=16,
        )
        label = (out or "").strip().upper().split()[0] if out else "CHAT"
        if label not in {"GENERATE_IDEA", "ANALYSE_IDEAS", "CHAT"}:
            label = "CHAT"
        return label
    except Exception:
        low = text.lower()
        if any(k in low for k in ["generate idea", "idea for", "come up with", "brainstorm"]):
            return "GENERATE_IDEA"
        if any(k in low for k in ["analyse", "analyze", "judge", "evaluate", "review idea"]):
            return "ANALYSE_IDEAS"
        return "CHAT"


def extract_text(msg: ChatMessage) -> str:
    try:
        t = (msg.text() or "").strip()
        if t:
            return t
    except Exception:
        pass
    for c in msg.content:
        if isinstance(c, TextContent):
            t = (c.text or "").strip()
            if t:
                return t
    return ""


async def send_progress(ctx, sender, text):
    await ctx.send(sender, ChatMessage(content=[TextContent(type="text", text=text)]))


async def handle_generate(ctx, sender, user_text):
    if not IDEA_GENERATOR_ADDRESS:
        await send_progress(
            ctx, sender,
            "The idea generator is still getting ready, so I can't spin up a new idea just yet. "
            "But I'm here—want to talk through what you're thinking?",
        )
        return
    await send_progress(ctx, sender, "Sending your prompt to the idea generator...")
    try:
        resp, status = await ctx.send_and_receive(
            IDEA_GENERATOR_ADDRESS,
            ChatMessage(content=[TextContent(type="text", text=user_text)]),
            response_type=ChatMessage,
            timeout=120,
        )
    except Exception:
        resp, status = None, None
    if not resp:
        await send_progress(
            ctx, sender,
            "The idea generator didn't respond in time. It might still be warming up—try again in a moment?",
        )
        return
    idea = extract_text(resp)
    if not idea:
        idea = "I got a response, but it came back empty. Want me to try again?"
    await send_progress(ctx, sender, "Here's what came back:\n\n" + idea)


async def handle_analyse(ctx, sender, user_text):
    if not JUDGE_ADDRESSES:
        await send_progress(
            ctx, sender,
            "No judges are connected yet, so I can't run an analysis right now. "
            "I'll let you know once they're ready.",
        )
        return
    idea = user_text
    outputs = []
    for idx, addr in enumerate(JUDGE_ADDRESSES, start=1):
        await send_progress(ctx, sender, f"Sending data to judge {idx}....")
        try:
            resp, status = await ctx.send_and_receive(
                addr,
                ChatMessage(content=[TextContent(type="text", text=idea)]),
                response_type=ChatMessage,
                timeout=120,
            )
        except Exception:
            resp, status = None, None
        if not resp:
            await send_progress(ctx, sender, f"Judge {idx} didn't respond—might not be ready yet.")
            outputs.append(f"Judge {idx}: (no response)")
            continue
        out = extract_text(resp)
        outputs.append(f"Judge {idx}: {out if out else '(empty response)'}")
        await send_progress(ctx, sender, f"Collected output from judge {idx}...")
    # synthesize
    combined = "\n\n".join(outputs)
    try:
        final = await llm_call(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a friendly coordinator. Synthesize the following judge outputs "
                        "into a clear, natural analysis for the user. Speak like a person."
                    ),
                },
                {"role": "user", "content": f"Idea:\n{idea}\n\nJudge outputs:\n{combined}"},
            ]
        )
        if not final:
            final = combined
    except Exception:
        final = combined
    await send_progress(ctx, sender, "Here's the consolidated analysis:\n\n" + final)


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

    text = extract_text(msg)
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
            t = extract_text(hm)
            if not t:
                continue
            role = "assistant" if entry.sender == ctx.agent.address else "user"
            history.append({"role": role, "content": t})
    except Exception:
        pass

    intent = await classify_intent(text)

    if intent == "GENERATE_IDEA":
        await handle_generate(ctx, sender, text)
        return
    if intent == "ANALYSE_IDEAS":
        await handle_analyse(ctx, sender, text)
        return

    # CHAT fallback
    messages = [{"role": "system", "content": PERSONA}]
    messages.extend(history)
    if not history or history[-1].get("content") != text:
        messages.append({"role": "user", "content": text})
    try:
        reply = await llm_call(messages)
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
