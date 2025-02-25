from __future__ import annotations
import asyncio
import logging
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    VoicePipelineAgent
)
from livekit.agents.llm import ChatMessage, ChatImage
from livekit.plugins import openai
from dotenv import load_dotenv
from api import AssistantFnc
from prompts import WELCOME_MESSAGE, INSTRUCTIONS, LOOKUP_PROFILE_MESSAGE
import os

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

async def get_video_track(room: rtc.Room):
    for participant_id, participant in room.remote_participants.items():
        for track_id, track_publication in participant.track_publications.items():
            if track_publication.track and isinstance(track_publication.track, rtc.RemoteVideoTrack):
                logger.info(f"Found video track {track_publication.track.sid} from participant {participant_id}")
                return track_publication.track
    raise ValueError("No remote video track found in the room")

async def get_latest_image(room: rtc.Room):
    video_stream = None
    try:
        video_track = await get_video_track(room)
        video_stream = rtc.VideoStream(video_track)
        async for event in video_stream:
            logger.debug("Captured latest video frame")
            return event.frame
    except Exception as e:
        logger.error(f"Failed to get latest image: {e}")
        return None
    finally:
        if video_stream:
            await video_stream.aclose()

async def before_llm_cb(assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
    latest_image = await get_latest_image(assistant.room)
    if latest_image:
        image_content = [ChatImage(image=latest_image)]
        chat_ctx.messages.append(ChatMessage(role="user", content=image_content))
        logger.debug("Added latest frame to conversation context")

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    await ctx.wait_for_participant()
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit that can both see and hear. "
            "You should use short and concise responses, avoiding unpronounceable punctuation. "
            "When you see an image in our conversation, naturally incorporate what you see into your response. "
            "Keep visual descriptions brief but informative."
        ),
    )
    
    assistant_fnc = AssistantFnc()
    
    assistant = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
        before_llm_cb=before_llm_cb
    )
    assistant.start(ctx.room)
    
    session = assistant.session
    session.conversation.item.create(
        llm.ChatMessage(
            role="assistant",
            content=WELCOME_MESSAGE
        )
    )
    session.response.create()
    
    @session.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        if isinstance(msg.content, list):
            msg.content = "\n".join("[image]" if isinstance(x, llm.ChatImage) else x for x in msg.content)
        print(f"Received user speech: {msg.content}")
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
