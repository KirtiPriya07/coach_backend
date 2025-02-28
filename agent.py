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
from livekit.plugins import cartesia, openai, deepgram, silero
from livekit import rtc
from livekit.agents.llm import ChatMessage, ChatImage
from api import AssistantFnc
from prompts import WELCOME_MESSAGE, LOOKUP_PROFILE_MESSAGE
import os

# Load environment variables from .env.local
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.DEBUG)

async def get_video_track(room: rtc.Room):
    """
    Attempt to find a remote video track published by the frontend webcam.
    This function loops (up to a timeout) and logs details about available tracks.
    """
    logger.debug("Searching for remote video tracks in the room...")
    timeout = 10  # seconds to wait for a video track
    start_time = asyncio.get_event_loop().time()
    while True:
        for participant_id, participant in room.remote_participants.items():
            logger.debug(f"Participant {participant_id} has {len(participant.track_publications)} track publications")
            for track_id, track_publication in participant.track_publications.items():
                track = track_publication.track
                # Log track details. Some tracks might not have a 'kind' attribute.
                track_kind = getattr(track, "kind", None) if track else None
                logger.debug(f"Examining track {track_id}: track={track}, kind={track_kind}")
                if track and (track_kind == "video" or isinstance(track, rtc.RemoteVideoTrack)):
                    logger.info(f"Found video track {track.sid} from participant {participant_id}")
                    return track
        if asyncio.get_event_loop().time() - start_time > timeout:
            break
        await asyncio.sleep(0.5)
    raise ValueError("No remote video track found in the room")

async def get_latest_image(room: rtc.Room):
    """
    Capture and return a single frame from the remote video track.
    If no track is found, log the error and return None.
    """
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
    """
    Preload the voice activity detector (VAD) from Silero.
    """
    proc.userdata["vad"] = silero.VAD.load()

async def before_llm_cb(assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
    """
    Before the LLM generates a response, capture the current frame from the webcam.
    If a frame is captured, add it to the conversation context as a ChatImage.
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
            "Keep responses short and use clear language. When you see an image, incorporate it into your answer briefly."
        ),
    )

    logger.info(f"Connecting to room {ctx.room.name}")
    # Connect to the room with auto-subscription enabled so all tracks (audio and video) are received.
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    # Wait for a participant (e.g. the frontend publishing its webcam stream) to connect.
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant {participant.identity}")

    assistant_fnc = AssistantFnc()

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
        # Convert message content list (if any, e.g., images) to a string.
        if isinstance(msg.content, list):
            msg.content = "\n".join("[image]" if isinstance(x, ChatImage) else x for x in msg.content)
        asyncio.create_task(process_user_speech(msg))

    async def process_user_speech(msg: llm.ChatMessage):
        if hasattr(assistant_fnc, "has_profile"):
            try:
                if await assistant_fnc.has_profile():
                    await handle_query(msg)
                else:
                    await find_profile(msg)
            except Exception as e:
                logger.error(f"Error in has_profile: {e}")
                await handle_query(msg)
        else:
            await handle_query(msg)

    async def find_profile(msg: llm.ChatMessage):
        agent.session.conversation.item.create(
            ChatMessage(
                role="system",
                content=LOOKUP_PROFILE_MESSAGE(msg)
            )
        )
        agent.session.response.create()

    async def handle_query(msg: llm.ChatMessage):
        agent.session.conversation.item.create(
            ChatMessage(
                role="user",
                content=msg.content
            )
        )
        agent.session.response.create()

    agent.start(ctx.room, participant)
    await agent.say(WELCOME_MESSAGE, allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
