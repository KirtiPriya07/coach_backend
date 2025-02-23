from __future__ import annotations
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai
from dotenv import load_dotenv
from api import AssistantFnc
from prompts import WELCOME_MESSAGE, INSTRUCTIONS, LOOKUP_PROFILE_MESSAGE
import os

load_dotenv()

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    await ctx.wait_for_participant()
    
    model = openai.realtime.RealtimeModel(
        instructions=INSTRUCTIONS,
        voice="shimmer",
        temperature=0.8,
        modalities=["audio", "text"]
    )
    
    # Instantiate your assistant function context.
    # Ensure AssistantFnc now supports profile lookups for interview context.
    assistant_fnc = AssistantFnc()
    
    assistant = MultimodalAgent(model=model, fnc_ctx=assistant_fnc)
    assistant.start(ctx.room)
    
    session = model.sessions[0]
    session.conversation.item.create(
        llm.ChatMessage(
            role="assistant",
            content=WELCOME_MESSAGE
        )
    )
    session.response.create()
    
    @session.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        # Convert list content to string if necessary.
        if isinstance(msg.content, list):
            msg.content = "\n".join("[image]" if isinstance(x, llm.ChatImage) else x for x in msg.content)
        
        # Check if we have an interview profile (e.g., job role, interview type) for the user.
        if assistant_fnc.has_profile():
            handle_query(msg)
        else:
            find_profile(msg)
    
    def find_profile(msg: llm.ChatMessage):
        session.conversation.item.create(
            llm.ChatMessage(
                role="system",
                content=LOOKUP_PROFILE_MESSAGE(msg)
            )
        )
        session.response.create()
    
    def handle_query(msg: llm.ChatMessage):
        session.conversation.item.create(
            llm.ChatMessage(
                role="user",
                content=msg.content
            )
        )
        session.response.create()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
