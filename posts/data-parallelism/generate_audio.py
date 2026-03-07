"""Generate TTS audio for each section of the Data Parallelism blog post using Kokoro-82M."""

import os
import soundfile as sf
import torch
from kokoro import KPipeline

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

SECTIONS = [
    (
        "what_is_parallelism",
        "One of the core problems with current neural architecture based AI is that "
        "it needs a massive amount of computation during training and inference. To "
        "perform these computations, a.k.a. FLOPS (Floating Point Operations), we "
        "have to utilize specialized hardware like GPUs, TPUs, NPUs, etc. These "
        "hardware devices can take a single instruction and run multiple processes "
        "simultaneously. However, there is a limitation of time and memory on a "
        "single device that makes training of AI models infeasible. Hence, we "
        "leverage multiple devices to speed up our training process and handle "
        "large models.",
    ),
    (
        "intro_data_parallelism",
        "There are many parallelism techniques that exist for training and inference "
        "of AI models. In this section, we will focus on Data Parallelism. This "
        "technique is specifically useful during model training. As the name suggests, "
        "we shard (or divide) our data into smaller batches and each batch is "
        "processed in parallel across multiple devices. Compared to single device "
        "batching where we process the entire data one by one in batches, here we "
        "utilize multiple devices to speed up the model training process.",
    ),
    (
        "data_parallelism_detail",
        "In data parallelism, each device maintains a copy of the model parameters. "
        "During training, each device processes a different subset of the training "
        "data and computes the gradients independently. After computing the gradients, "
        "an AllReduce operation is performed to aggregate the gradients across all "
        "devices. This ensures that all devices have the same updated model parameters "
        "for the next iteration.",
    ),
    (
        "allreduce_explained",
        "We can think of AllReduce as a communication operation that takes the "
        "gradients computed by each device and combines them, for example by summing, "
        "across all devices. This allows each device to have the same updated "
        "gradients, which are then used to update the model parameters. AllReduce is "
        "a critical component of data parallelism, as it ensures that all devices "
        "stay in sync during training.",
    ),
    (
        "what_is_fsdp",
        "Fully Sharded Data Parallel, or FSDP, is an advanced parallelism strategy "
        "that goes beyond traditional data parallelism. In FSDP, the model parameters "
        "are sharded, that is, divided across multiple devices, rather than each "
        "device maintaining a full copy of the model. This allows for training larger "
        "models that may not fit into the memory of a single device. FSDP also "
        "incorporates techniques to efficiently manage communication and "
        "synchronization between devices, making it a powerful tool for training "
        "large-scale AI models.",
    ),
    (
        "fsdp_workflow",
        "In the FSDP workflow, the model parameters are sharded across multiple "
        "devices. During training, each device computes gradients for its shard of "
        "the model parameters. The gradients are then communicated between devices to "
        "ensure that all devices have the necessary information to update their "
        "respective shards of the model. This allows for efficient training of large "
        "models while managing memory constraints effectively.",
    ),
    (
        "fsdp_vs_data_parallel",
        "The difference between FSDP and traditional data parallelism lies in how the "
        "model parameters are managed. In data parallelism, each device maintains a "
        "full copy of the model parameters, which can lead to memory constraints when "
        "training large models. In contrast, FSDP shards the model parameters across "
        "multiple devices, allowing for larger models to be trained without running "
        "into memory issues. Additionally, FSDP incorporates more efficient "
        "communication strategies to manage synchronization between devices, making "
        "it a more scalable solution for training large-scale AI models. "
        "For example, suppose we have a 7 Billion parameter model which we want to "
        "train on a 4 GPU node. In DP, we will place the entire 7B model copy on "
        "each GPU. However, in FSDP each device will only hold one fourth of the "
        "model parameters.",
    ),
    (
        "fsdp_summary",
        "In summary, Fully Sharded Data Parallel, or FSDP, is an advanced parallelism "
        "strategy that allows for training larger AI models by sharding the model "
        "parameters across multiple devices. It incorporates efficient communication "
        "and synchronization techniques to manage the training process effectively. "
        "FSDP is a powerful tool for training large-scale AI models while managing "
        "memory constraints and ensuring efficient communication between devices.",
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
