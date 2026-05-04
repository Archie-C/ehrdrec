
from ehrdrec.models.data_loading import LoadedData
from ehrdrec.models.data_processing import ProcessedDataMultiHot
from ehrdrec.processing.base import BaseProcessor

class MultiHotProcessor(BaseProcessor):
    
    def process(self, data: LoadedData) -> ProcessedDataMultiHot:
        pass 