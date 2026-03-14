from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoadedDocumentPage:
    image_bgr: np.ndarray
    source_type: str
    filename: str
    extension: str
