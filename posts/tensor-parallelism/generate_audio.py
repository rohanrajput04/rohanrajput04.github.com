"""Generate TTS audio for each section of the Tensor Parallelism blog post using Kokoro-82M."""

import os
import soundfile as sf
import torch
from kokoro import KPipeline

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

SECTIONS = [
    (
        "what_is_tensor_parallelism",
        "In the previous parts, we saw how Data Parallelism replicates the model "
        "and splits data, and how Pipeline Parallelism partitions the model by "
        "layers. Tensor Parallelism takes this a step further: instead of splitting "
        "at the layer level, it splits individual tensors, that is, weight matrices, "
        "within a single layer across multiple devices. Each device computes a "
        "portion of a layer's operation simultaneously, enabling us to parallelize "
        "even within a single transformer block. This is especially powerful for "
        "very large layers that are too memory-intensive for a single GPU.",
    ),
    (
        "column_parallel_linear",
        "The first building block of tensor parallelism is the Column Parallel "
        "Linear layer. Given a weight matrix W, we split it along the column "
        "dimension into N partitions, one per device. Each device holds a slice "
        "and computes its portion independently using the full input X. Since the "
        "columns are independent, no communication is needed during the forward "
        "computation itself. Each device produces a partial output that corresponds "
        "to a subset of the output features. The partial outputs are then "
        "concatenated or used directly by the next layer to form the complete result.",
    ),
    (
        "row_parallel_linear",
        "The complement to column parallelism is the Row Parallel Linear layer. "
        "Here, the weight matrix is split along the row dimension. Each device "
        "holds a horizontal slice and receives a corresponding partition of the "
        "input. Each device computes a partial matrix multiplication, producing a "
        "partial result. These partial results are then summed across devices using "
        "an AllReduce operation to produce the final output. Row parallel layers are "
        "typically paired with column parallel layers so that the output partitioning "
        "of one naturally feeds into the input partitioning of the other, minimizing "
        "communication.",
    ),
    (
        "mlp_tensor_parallel",
        "In a Transformer's MLP, or Feed-Forward block, there are typically two "
        "linear layers with a non-linearity, such as GeLU, in between. Tensor "
        "parallelism applies column parallelism to the first linear layer and row "
        "parallelism to the second. The first layer splits its output features "
        "across devices, and the GeLU activation is applied locally on each device. "
        "The second layer then takes these partitioned activations as input and "
        "performs a row-parallel computation, finishing with an AllReduce to "
        "synchronize the output. This design requires only one AllReduce per MLP "
        "block in the forward pass, keeping communication overhead minimal.",
    ),
    (
        "attention_tensor_parallel",
        "The multi-head attention mechanism is naturally suited for tensor "
        "parallelism because attention heads are independent computations. Each "
        "device is assigned a subset of the attention heads. The Query, Key, and "
        "Value projection matrices are split column-wise so that each device "
        "computes projections for its assigned heads. After computing attention "
        "independently, the output projection is applied as a row-parallel linear "
        "layer, with an AllReduce to combine the results. Just like the MLP block, "
        "this requires only one AllReduce per attention block in the forward pass.",
    ),
    (
        "allreduce_cost",
        "Tensor parallelism relies on AllReduce operations to synchronize partial "
        "results across devices. In each transformer layer, there are two AllReduce "
        "operations in the forward pass, one for the attention block and one for "
        "the MLP block, and two in the backward pass. Unlike pipeline parallelism, "
        "which only communicates between adjacent stages, tensor parallelism "
        "requires all-to-all communication within each layer. This makes tensor "
        "parallelism most effective when devices are connected via high-bandwidth "
        "interconnects, such as NVLink within a single node, as the communication "
        "latency directly impacts throughput.",
    ),
    (
        "tensor_vs_pipeline_vs_data",
        "Each parallelism strategy operates at a different granularity and serves a "
        "different purpose. Data parallelism splits data across devices, pipeline "
        "parallelism splits layers across devices, and tensor parallelism splits "
        "individual tensors within layers across devices. In practice, state-of-the-art "
        "training systems like Megatron-LM combine all three in a 3D parallelism "
        "configuration: tensor parallelism within a node leveraging fast NVLink, "
        "pipeline parallelism across nodes, and data parallelism across pipeline "
        "replicas.",
    ),
    (
        "summary",
        "Tensor Parallelism splits individual weight matrices across devices, "
        "enabling parallelism at the finest granularity. Column parallel layers "
        "split output features, row parallel layers split input features, and the "
        "two are paired together within Transformer MLP and attention blocks to "
        "minimize communication. Each transformer block requires only two AllReduce "
        "operations in the forward pass. Because of its high communication "
        "requirements, tensor parallelism works best within a single node with fast "
        "interconnects. Combined with pipeline and data parallelism in a 3D "
        "parallelism setup, it is essential for training today's largest language "
        "models.",
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
