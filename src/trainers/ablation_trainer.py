# src/trainers/ablation_trainer.py

"""
NTP (Next Token Prediction) Trainer with support for LoRA layer ablation experiments.
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.utils import log_to_wandb
from src.models.model_loader import apply_chat_template
from src.models.selective_lora import SelectiveLoRAConfig, apply_selective_lora
# Fallback import for standard LoRA
from src.models import apply_lora_and_get_counts


class SimpleDictDataset(Dataset):
    """Simple Dataset wrapper for a list of dictionaries."""
    def __init__(self, data_list: List[Dict[str, Any]]):
        self.data = data_list
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


class AblationNTPTrainer:
    """
    Trainer class for Next Token Prediction tasks, featuring selective LoRA support 
    specifically designed for ablation studies.
    """
    def __init__(self, config: Dict[str, Any], base_model: nn.Module, tokenizer, dataset: List[Dict[str, Any]], optimizer: Optimizer):
        self.config = config
        self.training_config = config['training']
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.output_dir = config['output_dir']
        
        self._initialize_model(base_model)
        
        self.device = next(self.model.parameters()).device
        
        train_dataset = SimpleDictDataset(dataset)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        self.scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.training_config['num_train_steps'],
            eta_min=0
        )
        
        logging.info(f"AblationNTPTrainer initialized (Batch size: {self.training_config['batch_size']})")

    def _initialize_model(self, base_model: nn.Module):
        """Applies either Selective LoRA or Standard LoRA based on configuration."""
        ablation_config = self.config.get('lora_ablation', {})
        
        if ablation_config:
            selective_config = SelectiveLoRAConfig(
                projection_type=ablation_config['projection_type'],
                layer_range=ablation_config['layer_range'],
                matrix_type=ablation_config['matrix_type'],
                rank=self.config['model']['lora_config']['r'],
                alpha=self.config['model']['lora_config']['lora_alpha'],
                dropout=self.config['model']['lora_config'].get('lora_dropout', 0.1)
            )
            self.model = apply_selective_lora(base_model, selective_config)
            
            # Persist ablation configuration for reproducibility
            ablation_info_path = os.path.join(self.output_dir, 'ablation_config.json')
            os.makedirs(self.output_dir, exist_ok=True)
            with open(ablation_info_path, 'w') as f:
                json.dump(selective_config.get_config_dict(), f, indent=2)
        else:
            self.model, _, _ = apply_lora_and_get_counts(base_model, self.config)

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Tokenizes batch data and creates attention masks and labels.
        Labels are set to -100 for padding tokens to be ignored by the loss function.
        """
        use_template = self.config.get('dataset', {}).get('use_chat_template', False)
        
        texts = []
        for item in batch:
            raw_text = item['text']
            if use_template:
                texts.append(apply_chat_template(self.tokenizer, raw_text))
            else:
                texts.append(raw_text + self.tokenizer.eos_token)
        
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        labels = encodings['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }
    
    def _calculate_loss_per_sample(self, outputs, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the average loss per sample, accounting for masking.
        """
        batch_size = labels.shape[0]
        
        # Shift tokens for next-token prediction
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(batch_size, -1)
        
        # Apply masking and calculate average loss per non-padded token
        masked_loss = loss * shift_attention_mask
        per_sample_loss = masked_loss.sum(dim=1) / shift_attention_mask.sum(dim=1)
        
        return per_sample_loss.mean()
    
    def _save_checkpoint(self, step_or_name: str):
        checkpoint_dir = os.path.join(self.output_dir, str(step_or_name))
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        logging.info(f"Checkpoint saved: {checkpoint_dir}")
    
    def train(self):
        """Executes the main training loop."""
        self.model.train()
        num_steps = self.training_config['num_train_steps']
        
        logging.info("Saving initial state at step_0...")
        self._save_checkpoint("step_0")
        
        data_iterator = iter(self.train_dataloader)
        progress_bar = tqdm(range(num_steps), desc="Training")
        
        for step in progress_bar:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader)
                batch = next(data_iterator)
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            loss = self._calculate_loss_per_sample(
                outputs,
                batch['labels'],
                batch['attention_mask']
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            wandb_metrics = {
                "train/loss": loss.item(),
                "train/learning_rate": self.scheduler.get_last_lr()[0],
                "train/step": step + 1
            }
            log_to_wandb(wandb_metrics, step=step + 1)
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{self.scheduler.get_last_lr()[0]:.7f}"
            })
            
            # Checkpoint strategy
            strategy_cfg = self.training_config.get('checkpoint_strategy', {})
            if strategy_cfg.get('policy') == "steps" and (step + 1) % strategy_cfg.get('value', 0) == 0:
                self._save_checkpoint(f"step_{step+1}")
        
        self._save_checkpoint("final")
        logging.info("Training completed.")