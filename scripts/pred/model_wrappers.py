# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import torch
from typing import Dict, List, Optional


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=name_or_path,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                model_kwargs=model_kwargs,
            )
        except:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,)
            
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.pipeline is None:
            if self.tokenizer.chat_template is not None:
                prompts = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in prompts
                ]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs, )
            assert len(output) == len(prompts)
            # output in the form of a list of list of dictionaries
            # outer list len = batch size
            # inner list len = 1
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            # This is a workaround for the llama3 tokenizer not being able to reproduce the same prompt after tokenization
            # see Issue https://github.com/NVIDIA/RULER/issues/54 for explaination
            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
                prompt = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]

            # Strip Qwen3.5 thinking blocks if present.
            if "<think>" in text and "</think>" in text:
                text = text.split("</think>")[-1]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]


class QwenGDN2ModelWrapper:
    """RULER wrapper for the custom GDN2-swapped Qwen3.5 model."""

    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        import json
        import sys
        from pathlib import Path

        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        sys.path.insert(0, "/workspace/gdn2-experiment/src")
        from qwen_gdn2 import (
            QWEN3_5_CONFIG,
            Qwen3_5GDN2Model,
            load_weights_into_qwen3_5_gdn2,
        )

        base_model_dir = Path("/workspace/gdn2-experiment/models/Qwen3.5-0.8B")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_dir, trust_remote_code=True
        )

        self.device = torch.device("cuda")
        self.model = Qwen3_5GDN2Model(QWEN3_5_CONFIG)

        with open(base_model_dir / "model.safetensors.index.json") as f:
            index = json.load(f)
        weights = {}
        for fn in sorted(set(index["weight_map"].values())):
            weights.update(load_file(base_model_dir / fn))

        ckpt_path = Path(name_or_path)
        has_ckpt = (ckpt_path / "model.pt").exists()

        load_weights_into_qwen3_5_gdn2(self.model, weights)

        if has_ckpt:
            ckpt_weights = torch.load(
                ckpt_path / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
            # SFT checkpoints are saved in the model's native state-dict format
            # (keys like "trf_blocks.*" / "tok_emb.*"). These do not overlap with
            # the HF-style keys consumed by load_weights_into_qwen3_5_gdn2, so we
            # load them directly. If the checkpoint instead uses HF keys, fall
            # back to the merge path for backward compatibility.
            sample_key = next(iter(ckpt_weights.keys()))
            if sample_key.startswith(("trf_blocks.", "tok_emb.")):
                missing, unexpected = self.model.load_state_dict(
                    ckpt_weights, strict=False
                )
                if missing:
                    print(f"Warning: missing keys when loading checkpoint: {missing}")
                if unexpected:
                    print(
                        f"Warning: unexpected keys when loading checkpoint: "
                        f"{unexpected[:10]}"
                    )
            else:
                weights.update(ckpt_weights)
                load_weights_into_qwen3_5_gdn2(self.model, weights)
        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.model.eval()

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")
        self.max_new_tokens = self.generation_kwargs.pop("max_new_tokens")
        self.use_chat_template = self.generation_kwargs.pop("use_chat_template", True)
        self.use_cache = self.generation_kwargs.pop("use_cache", False)

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        results = []
        for prompt in prompts:
            if self.tokenizer.chat_template is not None and getattr(self, "use_chat_template", True):
                messages = [{"role": "user", "content": prompt}]
                templated = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                templated = prompt
            inputs = self.tokenizer(templated, return_tensors="pt", padding=False)
            input_ids = inputs["input_ids"].to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=0.0,
                    use_cache=self.use_cache,
                )
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            tokenized_prompt = self.tokenizer(templated, return_tensors="pt")
            prompt_decoded = self.tokenizer.decode(
                tokenized_prompt.input_ids[0], skip_special_tokens=True
            )
            if text.startswith(prompt_decoded):
                text = text[len(prompt_decoded):]

            # Strip Qwen3.5 thinking blocks and keep only the final answer.
            if "<think>" in text and "</think>" in text:
                text = text.split("</think>")[-1]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({"text": [text]})
        return results
