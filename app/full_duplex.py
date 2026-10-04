import asyncio
import math
import os
import struct

import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# Configuration
# ============================================================

load_dotenv()

MODEL = "gemini-3.8-live-extended-thinking"

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000

CHANNELS = 1
CHUNK_SIZE = 1024

API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# Gemini configuration
# ============================================================

CONFIG = {
    "response_modalities": ["AUDIO"],

    # Gemini -> text transcription
    "output_audio_transcription": {},

    # Microphone -> Gemini transcription
    # Very useful for diagnosing false interruptions.
    "input_audio_transcription": {},

    "thinking_config": {
        "thinking_level": "low",
    },

    "realtime_input_config": {
        "automatic_activity_detection": {
            "disabled": False,

            # Less aggressive speech detection.
            "start_of_speech_sensitivity": (
                types.StartSensitivity.START_SENSITIVITY_LOW
            ),

            "end_of_speech_sensitivity": (
                types.EndSensitivity.END_SENSITIVITY_LOW
            ),

            "prefix_padding_ms": 100,

            # Give Gemini a little more time before deciding
            # that the user stopped speaking.
            "silence_duration_ms": 700,
        }
    },
}

gemini_is_speaking = False
# ============================================================
# PCM RMS calculation
# ============================================================

def pcm16_rms(data: bytes) -> float:
    """
    Calculate RMS level of signed 16-bit PCM audio.

    This replaces audioop, which is unavailable in Python 3.13+.
    """

    if not data:
        return 0.0

    sample_count = len(data) // 2

    if sample_count == 0:
        return 0.0

    samples = struct.unpack(
        f"<{sample_count}h",
        data[: sample_count * 2],
    )

    mean_square = sum(
        sample * sample
        for sample in samples
    ) / sample_count

    return math.sqrt(mean_square)


# ============================================================
# Microphone statistics
# ============================================================

class MicStats:

    def __init__(self):
        self.chunks = 0
        self.sum_squares = 0.0
        self.samples = 0

    def add(self, data: bytes):
        if not data:
            return

        sample_count = len(data) // 2

        if sample_count == 0:
            return

        samples = struct.unpack(
            f"<{sample_count}h",
            data[: sample_count * 2],
        )

        self.sum_squares += sum(
            sample * sample
            for sample in samples
        )

        self.samples += sample_count
        self.chunks += 1

    def report(self):
        if self.samples == 0:
            return

        rms = math.sqrt(
            self.sum_squares / self.samples
        )

        normalized = rms / 32768.0

        print(
            f"\n[Mic RMS] "
            f"{rms:7.1f} / 32768 "
            f"({normalized:.4f})"
        )

        self.chunks = 0
        self.sum_squares = 0.0
        self.samples = 0


# ============================================================
# Audio player
# ============================================================

async def audio_player(
    audio_queue: asyncio.Queue,
    stop_event: asyncio.Event,
):
    print("[Audio] Starting speaker...")

    stream = sd.RawOutputStream(
        samplerate=OUTPUT_SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK_SIZE,
    )

    stream.start()

    try:
        while not stop_event.is_set():

            try:
                audio_data = await asyncio.wait_for(
                    audio_queue.get(),
                    timeout=0.1,
                )
            except asyncio.TimeoutError:
                continue

            if audio_data is None:
                break

            loop = asyncio.get_running_loop()

            await loop.run_in_executor(
                None,
                stream.write,
                audio_data,
            )

            audio_queue.task_done()

    finally:
        stream.stop()
        stream.close()

        print("[Audio] Speaker stopped.")


# ============================================================
# Microphone recorder
# ============================================================

async def audio_recorder(
    input_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    mic_stats: MicStats,
):
    loop = asyncio.get_running_loop()

    def callback(indata, frames, time_info, status):

        if status:
            print(f"\n[Mic Status] {status}")

        audio_bytes = bytes(indata)

        # -----------------------------------------------
        # Diagnostics
        # -----------------------------------------------

        mic_stats.add(audio_bytes)

        # Approximately once per second.
        if mic_stats.chunks >= 16:
            mic_stats.report()

        # -----------------------------------------------
        # Send microphone data to asyncio queue
        # -----------------------------------------------

        try:
            loop.call_soon_threadsafe(
                input_queue.put_nowait,
                audio_bytes,
            )

        except asyncio.QueueFull:
            print("[Mic] Input queue full - dropping chunk.")

    stream = sd.RawInputStream(
        samplerate=INPUT_SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK_SIZE,
        callback=callback,
    )

    stream.start()

    print("[Audio] Microphone started.")

    try:
        while not stop_event.is_set():
            await asyncio.sleep(0.1)

    finally:
        stream.stop()
        stream.close()

        print("[Audio] Microphone stopped.")


# ============================================================
# Send microphone audio to Gemini
# ============================================================

async def send_audio_loop(session, input_queue, stop_event):
    print("[Gemini] Audio sender started.")

    try:
        while not stop_event.is_set():
            try:
                audio_chunk = await asyncio.wait_for(
                    input_queue.get(),
                    timeout=0.1,
                )
            except asyncio.TimeoutError:
                continue

            input_queue.task_done()

            # DIAGNOSTIC:
            # Capture microphone, but DO NOT send it to Gemini.
            continue

    except asyncio.CancelledError:
        raise
    finally:
        print("[Gemini] Audio sender stopped.")
           

# ============================================================
# Flush audio playback queue
# ============================================================

def flush_audio_queue(audio_queue: asyncio.Queue):

    flushed = 0

    while True:

        try:
            audio_queue.get_nowait()
            audio_queue.task_done()
            flushed += 1

        except asyncio.QueueEmpty:
            break

    if flushed:
        print(
            f"[Audio] Flushed "
            f"{flushed} queued chunks."
        )


# ============================================================
# Flush microphone queue
# ============================================================

def flush_input_queue(input_queue: asyncio.Queue):

    flushed = 0

    while True:

        try:
            input_queue.get_nowait()
            input_queue.task_done()
            flushed += 1

        except asyncio.QueueEmpty:
            break

    if flushed:
        print(
            f"[Mic] Flushed "
            f"{flushed} queued chunks."
        )


# ============================================================
# Gemini receiver
# ============================================================

async def receive_loop(
    session,
    audio_queue: asyncio.Queue,
    stop_event: asyncio.Event,
):

    print("[Gemini] Receiver started.")

    try:

        # IMPORTANT:
        #
        # session.receive() completes after one Gemini
        # interaction.
        #
        # Therefore we call it repeatedly for a continuous
        # voice assistant.

        while not stop_event.is_set():

            async for response in session.receive():

                if stop_event.is_set():
                    break

                server_content = response.server_content

                if not server_content:
                    continue

                # ====================================================
                # Gemini detected user speech / interruption
                # ====================================================

                if server_content.interrupted:

                    print(
                        "\n[Gemini interrupted]"
                    )

                    flush_audio_queue(
                        audio_queue
                    )

                    continue

                # ====================================================
                # What Gemini thinks USER said
                # ====================================================

                if server_content.input_transcription:

                    text = (
                        server_content
                        .input_transcription
                        .text
                    )

                    if text:
                        print(
                            f"\n[User heard by Gemini]: "
                            f"{text}",
                            end="",
                            flush=True,
                        )

                # ====================================================
                # What Gemini generated
                # ====================================================

                if server_content.output_transcription:

                    text = (
                        server_content
                        .output_transcription
                        .text
                    )

                    if text:
                        print(
                            f"\n[Gemini]: "
                            f"{text}",
                            end="",
                            flush=True,
                        )

                # ====================================================
                # Audio response
                # ====================================================

                if server_content.model_turn:
                    gemini_is_speaking = True

                    for part in server_content.model_turn.parts:

                        if part.inline_data:

                            audio_data = (
                                part.inline_data.data
                            )

                            if audio_data:

                                await audio_queue.put(
                                    audio_data
                                )

                # ====================================================
                # Turn complete
                # ====================================================

                if server_content.turn_complete:
                    gemini_is_speaking = False

                    print(
                        "\n[Gemini] Turn complete."
                    )

            # session.receive() returned because the
            # interaction completed.
            #
            # Do NOT exit the receiver.
            #
            # Start waiting for the next interaction.

            if not stop_event.is_set():

                await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        raise

    except Exception as exc:

        print(
            f"\n[Gemini Receiver Error] "
            f"{type(exc).__name__}: {exc}"
        )

        stop_event.set()

    finally:

        print(
            "[Gemini] Receiver stopped."
        )


# ============================================================
# Main voice assistant
# ============================================================

async def run_voice_assistant():

    if not API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    print("=" * 60)
    print("Gemini Live Voice Agent")
    print("=" * 60)

    print(f"Model: {MODEL}")
    print("Thinking level: low")

    print()
    print("[Audio] Full-duplex microphone mode enabled.")
    print("[Diagnostics] Microphone RMS monitoring enabled.")
    print("[Diagnostics] Input transcription enabled.")
    print("[Tip] Use headphones for this test.")
    print()

    client = genai.Client(
        api_key=API_KEY
    )

    # Bounded queues prevent unbounded latency growth.
    input_queue = asyncio.Queue(
        maxsize=50
    )

    audio_queue = asyncio.Queue(
        maxsize=100
    )

    stop_event = asyncio.Event()

    mic_stats = MicStats()

    player_task = None
    recorder_task = None
    sender_task = None
    receiver_task = None

    try:

        # ========================================================
        # Start speaker
        # ========================================================

        player_task = asyncio.create_task(
            audio_player(
                audio_queue,
                stop_event,
            )
        )

        # ========================================================
        # Connect Gemini Live
        # ========================================================

        async with client.aio.live.connect(
            model=MODEL,
            config=CONFIG,
        ) as session:

            print(
                "[Connected to Gemini Live]"
            )

            print(
                "[Listening... Speak now]"
            )

            print(
                "[Press Ctrl+C to stop]"
            )

            print()

            # ====================================================
            # Start receiver
            # ====================================================

            receiver_task = asyncio.create_task(
                receive_loop(
                    session,
                    audio_queue,
                    stop_event,
                )
            )

            # ====================================================
            # Start microphone
            # ====================================================

            recorder_task = asyncio.create_task(
                audio_recorder(
                    input_queue,
                    stop_event,
                    mic_stats,
                )
            )

            # ====================================================
            # Start sender
            # ====================================================

            sender_task = asyncio.create_task(
                send_audio_loop(
                    session,
                    input_queue,
                    stop_event,
                )
            )

            # ====================================================
            # Keep session alive
            # ====================================================

            while not stop_event.is_set():

                await asyncio.sleep(0.25)

    except KeyboardInterrupt:

        print(
            "\n\n[Shutdown] Ctrl+C received."
        )

    except Exception as exc:

        print(
            f"\n[Session Error] "
            f"{type(exc).__name__}: {exc}"
        )

    finally:

        stop_event.set()

        # ========================================================
        # Stop microphone/sender/receiver
        # ========================================================

        for task in [
            recorder_task,
            sender_task,
            receiver_task,
        ]:

            if task:

                task.cancel()

        await asyncio.gather(
            recorder_task,
            sender_task,
            receiver_task,
            return_exceptions=True,
        )

        # ========================================================
        # Stop speaker
        # ========================================================

        if player_task:

            player_task.cancel()

            await asyncio.gather(
                player_task,
                return_exceptions=True,
            )

        print(
            "[Session] Finished cleanly."
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    try:
        asyncio.run(
            run_voice_assistant()
        )

    except KeyboardInterrupt:

        print(
            "\nStopped."
        )