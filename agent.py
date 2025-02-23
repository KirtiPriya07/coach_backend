from __future__ import annotations
import asyncio
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
    # Connect and wait for a participant to join the room.
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    await ctx.wait_for_participant()
    
    # Initialize the realtime model, but do not send a welcome message yet.
    model = openai.realtime.RealtimeModel(
        instructions=INSTRUCTIONS,
        voice="shimmer",
        temperature=0.8,
        modalities=["audio", "text"]
    )
    
    # Instantiate your assistant function context.
    assistant_fnc = AssistantFnc()
    
    # Create and start the multimodal agent.
    assistant = MultimodalAgent(model=model, fnc_ctx=assistant_fnc)
    assistant.start(ctx.room)
    
    # Grab the session.
    session = model.sessions[0]
    
    # Create an asyncio Event to wait for the trigger.
    trigger_event = asyncio.Event()
    
    # Register a temporary handler that listens for the trigger message.
    @session.on("user_speech_committed")
    def wait_for_trigger(msg: llm.ChatMessage):
        # If the message content is a list, convert it to a string.
        if isinstance(msg.content, list):
            msg.content = "\n".join("[image]" if isinstance(x, llm.ChatImage) else x for x in msg.content)
        print(f"Trigger handler received message: {msg.content}")
        if msg.content.strip().lower() == "start a conversation":
            print("Trigger received; proceeding to send welcome message.")
            trigger_event.set()
        else:
            print("Received message is not the trigger; ignoring.")

    # Wait until the trigger is received.
    await trigger_event.wait()
    
    # Once triggered, send the welcome message.
    session.conversation.item.create(
        llm.ChatMessage(
            role="assistant",
            content=WELCOME_MESSAGE
        )
    )
    session.response.create()
    
    # Now register the normal message handler for subsequent messages.
    @session.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        # If message content is a list, convert it to a string.
        if isinstance(msg.content, list):
            msg.content = "\n".join("[image]" if isinstance(x, llm.ChatImage) else x for x in msg.content)
        print(f"Session handler received message: {msg.content}")
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
