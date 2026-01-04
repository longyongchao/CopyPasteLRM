from .FaithEval import FaithEval
from .HotpotQA import HotpotQA
from .Qasper import Qasper
from .MultiRC import MultiRC
from .PopQA import PopQA
from .PubMedQA import PubMedQA
from .TwoWikiMultiHopQA import TwoWikiMultihopQA
from .MuSiQue import MuSiQue
from .ConFiQA import ConFiQA

from typing import Literal
from enum import Enum
from enum import Enum

class AvailableDataset(Enum):
    FAITHEVAL = "FaithEval"
    HOTPOTQA = "HotpotQA"
    MULTIRC = "MultiRC"
    MUSIQUE = "MuSiQue"
    POPQA = "PopQA"
    PUBMEDQA = "PubMedQA"
    QASPER = "Qasper"
    TWO_WIKI_MULTI_HOP_QA = "2WikiMultiHopQA"
    CONFIQA_QA = "ConFiQA-QA"
    CONFIQA_MC = "ConFiQA-MC"
    CONFIQA_MR = "ConFiQA-MR"
    CONFIQA_QA_ORIGINAL = "ConFiQA-QA-Original"
    CONFIQA_MC_ORIGINAL = "ConFiQA-MC-Original"
    CONFIQA_MR_ORIGINAL = "ConFiQA-MR-Original"


def load(
    name: AvailableDataset,
    split: Literal["train", "validation", "test"],
    max_samples: int = -1,
    distractor_docs: int = 8,
    unanswerable: bool = False,
    reload: bool = False,
):
    if name == AvailableDataset.FAITHEVAL:
        return FaithEval(
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
        )
    elif name == AvailableDataset.HOTPOTQA:
        return HotpotQA(
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
            split="train" if split == "train" else "validation",
        )
    elif name == AvailableDataset.MULTIRC:
        return MultiRC(
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
            split="train" if split == "train" else "dev",
        )
    elif name == AvailableDataset.MUSIQUE:
        return MuSiQue(
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
            split="train" if split == "train" else "validation",
        )
    elif name == AvailableDataset.POPQA:
        return PopQA(
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
            split="train" if split == "train" else "test",
        )
    elif name == AvailableDataset.PUBMEDQA:
        return PubMedQA(
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
            dataset_name="pqa_artificial" if split == "train" else "pqa_labeled",
        )
    elif name == AvailableDataset.QASPER:
        return Qasper(
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
            split="train" if split == "train" else "test",
        )
    elif name == AvailableDataset.TWO_WIKI_MULTI_HOP_QA:
        return TwoWikiMultihopQA(
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
            split="dev",
        )
    elif name == AvailableDataset.CONFIQA_QA:
        return ConFiQA(
            subset="QA",
            split="counterfactual",
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
        )
    elif name == AvailableDataset.CONFIQA_MC:
        return ConFiQA(
            subset="MC",
            split="counterfactual",
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
        )
    elif name == AvailableDataset.CONFIQA_MR:
        return ConFiQA(
            subset="MR",
            split="counterfactual",
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
        )
    elif name == AvailableDataset.CONFIQA_QA_ORIGINAL:
        return ConFiQA(
            subset="QA",
            split="original",
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
        )
    elif name == AvailableDataset.CONFIQA_MC_ORIGINAL:
        return ConFiQA(
            subset="MC",
            split="original",
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
        )
    elif name == AvailableDataset.CONFIQA_MR_ORIGINAL:
        return ConFiQA(
            subset="MR",
            split="original",
            reload=reload,
            max_samples=max_samples,
            distractor_docs=distractor_docs,
            unanswerable=unanswerable,
        )
    else:
        raise f"不支持 {name} 数据集"
