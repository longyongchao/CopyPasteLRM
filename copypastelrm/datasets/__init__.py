from .BaseDatasetLoader import BaseDatasetLoader
from .FaithEval import FaithEval
from .HotpotQA import HotpotQA
from .Qasper import Qasper
from .MultiRC import MultiRC
from .PopQA import PopQA
from .PubMedQA import PubMedQA
from .TwoWikiMultiHopQA import TwoWikiMultihopQA
from .MuSiQue import MuSiQue
from .CopyPaste import CopyPaste

from enum import Enum

class AvailebleDatasets(Enum):
    FAITHEVAL = 'faitheval'
    HOTPOTQA = 'hotpotqa'
    MULTIRC = 'multirc'
    MUSIQUE = ' musique'
    POPQA = 'popqa'
    PUBMEDQA = 'pubmedqa'
    QASPER = 'qasper'
    TWOWIKI = '2wikimultihopqa'

def load(name: AvailebleDatasets, reload: bool =False):
    if name == AvailebleDatasets.FAITHEVAL:
        return FaithEval(reload=reload)
    elif name == AvailebleDatasets.HOTPOTQA:
        return HotpotQA(reload=reload)
    elif name == AvailebleDatasets.MULTIRC:
        return MultiRC(reload=reload)
    elif name == AvailebleDatasets.MUSIQUE:
        return MuSiQue(reload=reload)
    elif name == AvailebleDatasets.POPQA:
        return PopQA(reload=reload)
    elif name == AvailebleDatasets.PUBMEDQA:
        return PubMedQA(reload=reload)
    elif name == AvailebleDatasets.QASPER:
        return Qasper(reload=reload)
    elif name == AvailebleDatasets.TWOWIKI:
        return TwoWikiMultihopQA(reload=reload)
    else:
        raise f"不支持 {name} 数据集"
