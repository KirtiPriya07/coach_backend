from __future__ import annotations
import asyncio
import logging
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, openai, deepgram, silero  # Removed turn_detector import
from livekit import rtc
from livekit.agents.llm import ChatMessage, ChatImage
from api import AssistantFnc
from prompts import WELCOME_MESSAGE, INSTRUCTIONS, LOOKUP_PROFILE_MESSAGE
import os

# Load environment variables
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.DEBUG)

async def get_video_track(room: rtc.Room):
    """Find and return the first available remote video track in the room."""
    for participant_id, participant in room.remote_participants.items():
        for track_id, track_publication in participant.track_publications.items():
            if track_publication.track and isinstance(track_publication.track, rtc.RemoteVideoTrack):
                logger.info(
                    f"Found video track {track_publication.track.sid} from participant {participant_id}"
                )
                return track_publication.track
    raise ValueError("No remote video track found in the room")

async def get_latest_image(room: rtc.Room):
    """Capture and return a single frame from the video track."""
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

def prewarm(proc: JobProcess):
    # Preload the voice activity detector (VAD) using Silero.
    proc.userdata["vad"] = silero.VAD.load()

async def before_llm_cb(assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
    """
    Callback that runs right before the LLM generates a response.
    Captures the current video frame and adds it to the conversation context.
    """
    latest_image = await get_latest_image(assistant._room)
    if latest_image:
        image_content = [ChatImage(image=latest_image)]
        chat_ctx.messages.append(ChatMessage(role="user", content=image_content))
        logger.debug("Added latest frame to conversation context")

async def entrypoint(ctx: JobContext):
    # Set up the initial system prompt.
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit that can both see and hear. "
            "You should use short and concise responses, avoiding unpronounceable punctuation. "
            "When you see an image in our conversation, naturally incorporate what you see "
            "into your response. Keep visual descriptions brief but informative."
        ),
    )

    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    # Wait for the first participant to connect.
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant {participant.identity}")

    # Instantiate your assistant function context (for profile lookups, etc.)
    assistant_fnc = AssistantFnc()

    # Create the VoicePipelineAgent without the turn detector.
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
        chat_ctx=initial_ctx,
        before_llm_cb=before_llm_cb
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    @agent.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        # If message content is a list, convert it to a string.
        if isinstance(msg.content, list):
            msg.content = "\n".join("[image]" if isinstance(x, ChatImage) else x for x in msg.content)
        # Use your assistant function context to check for a user profile.
        if assistant_fnc.has_profile():
            handle_query(msg)
        else:
            find_profile(msg)

    def find_profile(msg: llm.ChatMessage):
        agent.session.conversation.item.create(
            ChatMessage(
                role="system",
                content=LOOKUP_PROFILE_MESSAGE(msg)
            )
        )
        agent.session.response.create()

    def handle_query(msg: llm.ChatMessage):
        agent.session.conversation.item.create(
            ChatMessage(
                role="user",
                content=msg.content
            )
        )
        agent.session.response.create()

    # Start the agent in the room.
    agent.start(ctx.room, participant)

    # Greet the user using the welcome message from prompts.
    await agent.say(WELCOME_MESSAGE, allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
