from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

@dataclass
class LLMBackend:
      name: str

    def generate(self, *, prompt: str, text_inputs: Optional[str] = None,
                 images: Optional[List[Any]] = None,
                 audio: Optional[Any] = None,
                 video: Optional[Any] = None,
                 json_only: bool = False) -> str:
        raise NotImplementedError("Implement your model API call here.")


@dataclass
class PerceptionAgent:
    backend: LLMBackend
    prompt: str

    def run(self, *, eeg_graph_img=None, audio_clip=None, video_clip=None, extra_text: str = "") -> str:
        return self.backend.generate(
            prompt=self.prompt,
            text_inputs=extra_text if extra_text else None,
            images=[eeg_graph_img] if eeg_graph_img is not None else None,
            audio=audio_clip,
            video=video_clip,
        )


@dataclass
class CentralReasoner:
    backend: LLMBackend
    cot_prompt: str

    def run(self, *, eeg_cue: str, audio_cue: str, visual_cue: str,
            eeg_graph_img=None, audio_clip=None, video_clip=None) -> str:
        packed_text = f"EEG Cue:\n{eeg_cue}\n\nAudio Cue:\n{audio_cue}\n\nVisual Cue:\n{visual_cue}\n"
        return self.backend.generate(
            prompt=self.cot_prompt,
            text_inputs=packed_text,
            images=[eeg_graph_img] if eeg_graph_img is not None else None,
            audio=audio_clip,
            video=video_clip,
        )


@dataclass
class EmotionMappingAgent:
    backend: LLMBackend
    prompt: str
    label_set: List[str]

    def run(self, *, stage_text: str) -> Dict[str, Any]:
        text = f"Label set: {self.label_set}\n\nReasoning:\n{stage_text}\n"
        out = self.backend.generate(prompt=self.prompt, text_inputs=text, json_only=True)
        return {"raw": out}


@dataclass
class JudgeAgent:
    backend: LLMBackend
    prompt: str

    def score(self, *, stage_text: str, evidence_text: str = "",
              eeg_cue: str = "", audio_cue: str = "", visual_cue: str = "",
              audio_clip=None, video_clip=None) -> Dict[str, Any]:
        packed = f"Stage text:\n{stage_text}\n\nEvidence text:\n{evidence_text}\n\nEEG Cue:\n{eeg_cue}\n\nAudio Cue:\n{audio_cue}\n\nVisual Cue:\n{visual_cue}\n"
        out = self.backend.generate(prompt=self.prompt, text_inputs=packed, audio=audio_clip, video=video_clip, json_only=True)
        return {"raw": out}
