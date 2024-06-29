# model_utils.py
import torch
import torch.nn as nn
from loguru import logger
from torch.nn import functional as F
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig, AutoModelForPreTraining

from src.engine.config import BenchmarkConfig


def setup_model_and_tokenizer(config: BenchmarkConfig):
    logger.info("Setting up model and tokenizer...")

    model = create_transformer_model(config)
    logger.info("Created Transformer-based model")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    logger.info(f"Loaded tokenizer from {config.tokenizer_path}")

    if config.data_path:
        logger.info(f"Loading pre-trained model from {config.data_path}")
        state_dict = torch.load(config.data_path, map_location='cpu')
        model.load_state_dict(state_dict)

    return model, tokenizer


def create_transformer_model(config: BenchmarkConfig) -> PreTrainedModel:
    model_config = PretrainedConfig(
        vocab_size=config.model_params.vocab_size,
        d_model=config.model_params.d_model,
        n_heads=config.model_params.n_heads,
        d_ff=config.model_params.d_ff,
        n_layers=config.model_params.n_layers,
        dropout=config.model_params.dropout,
        max_position_embeddings=config.model_params.max_position_embeddings,
        pad_token_id=config.model_params.pad_token_id,
    )

    return AutoModelForPreTraining.from_pretrained(model_config)


def apply_knowledge_distillation(teacher_model: nn.Module, student_model: nn.Module, alpha: float = 0.5,
                                 temperature: float = 2.0):
    def knowledge_distillation_loss(student_logits, teacher_logits, labels, _alpha, _temperature):
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / _temperature, dim=1),
            F.softmax(teacher_logits / _temperature, dim=1)
        ) * (_temperature ** 2)
        student_loss = F.cross_entropy(student_logits, labels)
        return _alpha * distillation_loss + (1 - _alpha) * student_loss

    student_model.knowledge_distillation_loss = knowledge_distillation_loss
    student_model.teacher_model = teacher_model
    student_model.kd_alpha = alpha
    student_model.kd_temperature = temperature

    logger.info(f"Applied knowledge distillation with _alpha={alpha} and _temperature={temperature}")


def optimize_model_for_inference(model: PreTrainedModel):
    logger.info("Optimizing model for inference...")
    model.eval()

    model = torch.jit.script(model)

    logger.info('Model optimized with TorchScript')
    return model


def analyze_model_complexity(model: PreTrainedModel):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
