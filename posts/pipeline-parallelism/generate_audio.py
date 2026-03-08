"""Generate TTS audio for each section of the Pipeline Parallelism blog post using Kokoro-82M."""

import os
import soundfile as sf
import torch
from kokoro import KPipeline

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

SECTIONS = [
    (
        "what_is_pipeline_parallelism",
        "In Part 1, we saw how Data Parallelism replicates the entire model across "
        "devices and splits the data. But what happens when a model is too large to "
        "fit on a single device, even with FSDP? Pipeline Parallelism takes a "
        "different approach. Instead of replicating the model, it partitions the "
        "model itself across multiple devices. Each device holds a subset of the "
        "model's layers, called a stage, and data flows through the stages "
        "sequentially, much like an assembly line in a factory.",
    ),
    (
        "how_pipeline_parallelism_works",
        "In pipeline parallelism, the model is split into consecutive groups of "
        "layers, and each group is assigned to a different device. During the "
        "forward pass, each device processes its layers and sends the activations "
        "to the next device. During the backward pass, gradients flow in the "
        "reverse direction. This allows us to train models that are too large for "
        "a single device's memory, since each device only needs to store a fraction "
        "of the total parameters.",
    ),
    (
        "naive_pipeline_parallelism",
        "The simplest form of pipeline parallelism processes one mini-batch at a "
        "time through all stages sequentially. While straightforward, this approach "
        "has a major drawback: at any given time, only one device is actively "
        "computing while all others sit idle. This means device utilization is "
        "roughly 1 over N, where N is the number of stages. The idle time wasted "
        "across devices is known as the pipeline bubble.",
    ),
    (
        "bubble_problem",
        "The pipeline bubble is the key inefficiency of naive pipeline parallelism. "
        "If we have 4 stages and it takes time t for each stage to process a "
        "mini-batch, then during the forward pass, stage 1 finishes at t but stage "
        "4 doesn't start until 3t. The total idle time across all devices grows "
        "linearly with the number of stages. Reducing this bubble is the primary "
        "goal of more advanced pipeline scheduling strategies.",
    ),
    (
        "gpipe_microbatching",
        "GPipe addresses the bubble problem by splitting each mini-batch into "
        "smaller micro-batches. Instead of waiting for an entire mini-batch to pass "
        "through all stages, GPipe injects multiple micro-batches into the pipeline "
        "in quick succession. This way, while stage 2 is processing micro-batch 1, "
        "stage 1 can already start on micro-batch 2. The more micro-batches we use, "
        "the smaller the bubble becomes relative to the total computation. Gradients "
        "are accumulated across all micro-batches and synchronized at the end.",
    ),
    (
        "one_f_one_b_schedule",
        "The 1F1B, or One Forward One Backward, schedule, introduced by PipeDream, "
        "further improves pipeline efficiency. After an initial warm-up phase where "
        "forward passes fill the pipeline, each device alternates between one "
        "forward pass and one backward pass. This interleaved scheduling reduces "
        "peak memory usage compared to GPipe, since devices don't need to store "
        "activations for all micro-batches simultaneously, while maintaining "
        "similar pipeline utilization.",
    ),
    (
        "pipeline_vs_data_parallelism",
        "Pipeline parallelism and data parallelism solve different problems and are "
        "often used together. Data parallelism replicates the model and splits the "
        "data, which is ideal when the model fits on a single device but you want "
        "faster training. Pipeline parallelism splits the model across devices, "
        "which is necessary when the model is too large for one device. In practice, "
        "large-scale training combines both: the model is partitioned across "
        "pipeline stages, and each stage is replicated across multiple devices "
        "using data parallelism.",
    ),
    (
        "summary",
        "Pipeline Parallelism enables training of models too large for a single "
        "device by partitioning layers across multiple devices. The naive approach "
        "suffers from the pipeline bubble, where devices sit idle waiting for data "
        "to flow through the pipeline. GPipe reduces this bubble through "
        "micro-batching, and PipeDream's 1F1B schedule further optimizes memory "
        "usage with interleaved forward and backward passes. Combined with data "
        "parallelism, pipeline parallelism is a core building block for training "
        "today's largest AI models.",
    ),
]

VOICE = "am_michael"
SAMPLE_RATE = 24000


def main():
    print("Initializing Kokoro TTS pipeline...")
    pipeline = KPipeline(lang_code="a")

    for name, text in SECTIONS:
        out_path = os.path.join(AUDIO_DIR, f"{name}.wav")
        print(f"Generating: {name}...")

        # Collect all audio chunks from the generator
        audio_chunks = []
        for result in pipeline(text, voice=VOICE):
            if result.output is not None and result.output.audio is not None:
                audio_chunks.append(result.output.audio)

        if audio_chunks:
            full_audio = torch.cat(audio_chunks)
            sf.write(out_path, full_audio.numpy(), SAMPLE_RATE)
            print(f"  Saved: {out_path} ({len(full_audio) / SAMPLE_RATE:.1f}s)")
        else:
            print(f"  WARNING: No audio generated for {name}")

    print("Done! All audio files generated.")


if __name__ == "__main__":
    main()
