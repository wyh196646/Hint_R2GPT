from transformers import SwinModel
#from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb
from transformers import AutoModel
import torch.nn as nn
import torch
import itertools
import torch.nn.functional as F
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    AutoTokenizer,
)
import warnings
warnings.filterwarnings("ignore")
import lightning as  pl
import transformers
#import optim
from torch import optim

class A:  
    def __init__(self,xx,yy):  
        self.x=xx  
        self.y=yy  
    def add(self):  
        print("x和y的和为：%d"%(self.x+self.y))  
        
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class ImageEncoder(nn.Module):
    def __init__(
        self, model_name: str, trainable: bool = True
    ) -> None:
        super().__init__()
        self.model = SwinModel.from_pretrained(model_name,add_pooling_layer=True)
        for param in self.model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, x):
        return self.model(x)['pooler_output']

class TextEncoder(nn.Module):
    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()

        self.model = transformers.AutoModel.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
    
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)

        x += projected

        return self.layer_norm(x)    
    
class CLIPDualEncoderModel(pl.LightningModule):
    def __init__(
        self,
        image_encoder_alias: str,
        text_encoder_alias: str,
        image_encoder_pretrained: bool = True,
        image_encoder_trainable: bool = True,
        text_encoder_trainable: bool = True,
        image_embedding_dims: int = 1024,
        text_embedding_dims: int = 768,
        projection_dims: int = 1024,
        dropout: float = 0.0,
        temperature: float = 1.0,
        weight_decay: float = 0.0,
        head_lr: float = 1e-3,
        image_encoder_lr: float = 1e-4,
        text_encoder_lr: float = 1e-5,
        lr_scheduler_patience: float = 1.0,
        lr_scheduler_factor: float = 0.8,
        logit_scale_init_value=0.07,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_encoder = ImageEncoder(
            model_name=image_encoder_alias,
            #pretrained=image_encoder_pretrained,
            trainable=image_encoder_trainable,
        )
        self.text_encoder = TextEncoder(
            model_name=text_encoder_alias, trainable=text_encoder_trainable
        )
        self.image_projection = ProjectionHead(
            embedding_dim=image_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_alias)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.head_lr = head_lr
        self.image_encoder_lr = image_encoder_lr
        self.text_encoder_lr = text_encoder_lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
        self.save_hyperparameters()
    
    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def _compute_losses(self, image_embeddings, text_embeddings):
        logits_per_image = self.compute_logits(image_embeddings, text_embeddings)
        logits_per_text = logits_per_image.t()

     
        loss = self.clip_loss(logits_per_text)
        # logits = (text_embeddings @ image_embeddings.T) / self.temperature
        # images_similarity = image_embeddings @ image_embeddings.T
        # texts_similarity = text_embeddings @ text_embeddings.T
        # targets = F.softmax(
        #     (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        # )
        # texts_loss = cross_entropy(logits, targets, reduction='none')
        # images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        # loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        # return loss.mean()
        # logits = (text_embeddings @ image_embeddings.T) / self.temperature
        # images_similarity = image_embeddings @ image_embeddings.T
        # texts_similarity = text_embeddings @ text_embeddings.T
        # targets = F.softmax(
        #     (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        # )
        # images_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        # texts_loss = (-targets * self.log_softmax(logits)).sum(1)
        # return (images_loss + texts_loss) / 2.0
        return loss
    
    def forward(self, inputs):
        image_features = self.image_encoder(inputs["image"][0])
        text = inputs["input_text"]
        input_ids,attention_mask=self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.text_encoder.model.device),self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)['attention_mask'].to(self.text_encoder.model.device)

        text_features = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        return image_embeddings, text_embeddings

    def configure_optimizers(self):
        parameters = [
            {"params": self.image_encoder.parameters(), "lr": self.image_encoder_lr},
            {"params": self.text_encoder.parameters(), "lr": self.text_encoder_lr},
            {
                "params": itertools.chain(
                    self.image_projection.parameters(),
                    self.text_projection.parameters(),
                ),
                "lr": self.head_lr,
                "weight_decay": self.weight_decay,
            },
        ]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_scheduler_patience,
            factor=self.lr_scheduler_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }

    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(image_embeddings, text_embeddings).mean()
        train_loss = self.all_gather(loss)
        self.log("train/loss", train_loss.mean())
        return loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(image_embeddings, text_embeddings).mean()
        val_loss = self.all_gather(loss)
        self.log("val/loss", val_loss.mean())
        return loss


