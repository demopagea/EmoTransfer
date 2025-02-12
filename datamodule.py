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
        # return self.buffer.create_dataloader(int(len(self.buffer)/16), 1)  #batch size, num_worker
        print('call_train_dataloader')
        return self.buffer.create_dataloader(2, 2)

    # def val_dataloader(self):

    #     return dataloader.create_dataloader(self.hparams_rl, 0)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    # def teardown(self, stage: str):
    #     # Used to clean-up when the run is finished
    #     ...
