import pytorch_lightning as pl
from models.run_models import run_shallow_model, run_GRU_atten
from models.combination import CombinedModel
from pytorch_lightning.loggers import TensorBoardLogger
from datasets.elec_dataset import ElectricityDataModule
from config import config

pl.seed_everything(config.seed, workers=True)

# preprare base models
# atten_error = run_GRU_atten()
# lgb_error = run_shallow_model('lightgbm')
# print(atten_error, lgb_error)

#
net = CombinedModel()
dm = ElectricityDataModule()

trainer = pl.Trainer(max_epochs=200,
                     min_epochs=config.min_epochs,
                     logger=[TensorBoardLogger("lightning_logs", name="CombinedNet")],
                     gpus=1)

trainer.fit(net, datamodule=dm)
trainer.test(datamodule=dm, ckpt_path='best', verbose=True)


