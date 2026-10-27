# Third-party notices

The root [MIT License](LICENSE) covers Tadpole's original code and documentation.
It does not replace the licenses of the material listed below or of installed dependencies.

## Adapted nanobot tools — MIT

Parts of the shell, web, filesystem and document tools are adapted from
[HKUDS/nanobot](https://github.com/HKUDS/nanobot), revision
`0b1fa0c3e44510e3d34d9bed4d491884cb19de7e`.
The attribution, adaptation details and complete upstream MIT notice are retained in
[agent_from_scratch/tools/THIRD_PARTY.md](agent_from_scratch/tools/THIRD_PARTY.md).

## Qwen chat template — Apache-2.0

[agent_from_scratch/prompts/qwen_chat.jinja](agent_from_scratch/prompts/qwen_chat.jinja)
is adapted from the Qwen chat template published by the Qwen team, Alibaba Cloud,
in [Qwen2.5-7B-Instruct's tokenizer configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json).
The upstream model repository distributes this material under
[Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE);
a full copy is included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

Tadpole modifies the tool-call instructions, limits batches to eight calls, handles serialized
argument strings, and escapes `<` in arguments and tool results. The template is used for
both local inference and training/evaluation with Qwen3. These adaptations do not change
the upstream license. The template's adjacent
[licensing and modification notice](agent_from_scratch/prompts/qwen_chat.jinja.license)
keeps its frozen evaluation and training-data hashes unchanged.

## Separately obtained models and datasets

The following resources are used by the documented workflows and retain their own terms.
Their full model weights and dataset distributions are not bundled with this repository.

| Resource | Upstream license / terms |
|---|---|
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) and its [GGUF conversion](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) | Apache-2.0; retain the notices supplied with the downloaded model |
| [NVIDIA When2Call](https://huggingface.co/datasets/nvidia/When2Call) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/); converted rows retain source/revision/license metadata in the export |
| [Agent-Ark Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) | Apache-2.0; converted rows retain source/revision/license metadata in the export |
| [Berkeley Function Calling Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | Follow the code and dataset licenses in the separately obtained upstream checkout |

The public-data converter changes conversation formatting and selects assistant turns as
training targets; it does not relicense the source rows. Redistributing datasets, trained
adapters or model weights requires retaining the applicable upstream terms and attribution.
Python dependencies are installed separately and remain under their respective licenses.
