from pathlib import Path
from buffer_jc import RolloutBuffer



class DataModule():

    def __init__(self, buffer_path: Path):
        super().__init__()
        self.buffer = RolloutBuffer(buffer_path, )
        # self.hparams_rl = hparams_rl

    def prepare_data(self, ):
        self.buffer.load_data()

    def setup(self, stage: str):
        self.buffer.load_data()

    def train_dataloader(self):
        return self.buffer.create_dataloader(2, 2)


