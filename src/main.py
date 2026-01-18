from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from .agents import PerceptionAgent, CentralReasoner, EmotionMappingAgent, JudgeAgent

def split_stages(stage_text: str) -> Dict[str, str]:
    out = {"stage1": "", "stage2": "", "stage3": ""}
    cur = None
    for line in stage_text.splitlines():
        l = line.strip()
        if l.lower().startswith("stage1"):
            cur = "stage1"; out[cur] += line + "\n"; continue
        if l.lower().startswith("stage2"):
            cur = "stage2"; out[cur] += line + "\n"; continue
        if l.lower().startswith("stage3"):
            cur = "stage3"; out[cur] += line + "\n"; continue
        if cur:
            out[cur] += line + "\n"
    return {k: v.strip() for k, v in out.items()}

@dataclass
class PHEMOPipeline:
    eeg_agent: PerceptionAgent
    audio_agent: PerceptionAgent
    visual_agent: PerceptionAgent
    reasoner: CentralReasoner
    mapper: EmotionMappingAgent
    judge: JudgeAgent

    def run_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        eeg_graph = sample.get("eeg_graph_img", None)
        audio = sample.get("audio_clip", None)
        video = sample.get("video_clip", None)

        S_E = self.eeg_agent.run(eeg_graph_img=eeg_graph)
        S_A = self.audio_agent.run(audio_clip=audio)
        S_V = self.visual_agent.run(video_clip=video)

      
        Y = self.reasoner.run(eeg_cue=S_E, audio_cue=S_A, visual_cue=S_V,
                              eeg_graph_img=eeg_graph, audio_clip=audio, video_clip=video)
        stages = split_stages(Y)

        
        mapping = self.mapper.run(stage_text=Y)

       
        j1 = self.judge.score(stage_text=stages["stage1"], evidence_text="Use audio/video evidence.", audio_clip=audio, video_clip=video)
        j2 = self.judge.score(stage_text=stages["stage2"], evidence_text="Use EEG cue evidence.", eeg_cue=S_E)
        j3 = self.judge.score(stage_text=stages["stage3"], evidence_text="Use Audio/Visual cues.", audio_cue=S_A, visual_cue=S_V)

        return {
            "S_E": S_E, "S_A": S_A, "S_V": S_V,
            "Y": Y, "stages": stages,
            "mapping_raw": mapping["raw"],
            "judge_raw": [j1["raw"], j2["raw"], j3["raw"]],
            "y_true": sample.get("y_true"),
        }
