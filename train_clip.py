import os
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.CLIP import CLIPDualEncoderModel
from lightning.pytorch import seed_everything
import lightning.pytorch as pl
import warnings
warnings.filterwarnings("ignore")
#set gpu available
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
        # devices=args.devices,
        devices=[2,3,4,5], 
        #devices=[0,1],
        #devices=[0],
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.pretrain_max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )
    model = CLIPDualEncoderModel(args.vision_model,args.text_model)

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)


if __name__ == '__main__':
    main()