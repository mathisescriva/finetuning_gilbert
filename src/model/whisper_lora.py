"""
Configuration LoRA pour Whisper.
Permet adaptation légère du modèle pour spécialisation domaine.
"""

from peft import LoraConfig, get_peft_model, TaskType
from transformers import WhisperForConditionalGeneration
from typing import Optional, Dict


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    bias: str = "none",
) -> LoraConfig:
    """
    Crée une configuration LoRA pour Whisper.
    
    Args:
        r: Rank des matrices de décomposition
        lora_alpha: Scaling factor
        lora_dropout: Dropout pour LoRA
        target_modules: Modules à adapter (décodeur par défaut)
        bias: Type de biais à entraîner
    
    Returns:
        LoraConfig configurée
    """
    if target_modules is None:
        # Modules du décodeur Whisper
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc1",
            "fc2",
        ]
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.FEATURE_EXTRACTION,
        # Note: Whisper utilise FEATURE_EXTRACTION pour ASR
    )
    
    return config


def apply_lora_to_whisper(
    model: WhisperForConditionalGeneration,
    lora_config: LoraConfig,
) -> WhisperForConditionalGeneration:
    """
    Applique LoRA à un modèle Whisper.
    
    Args:
        model: Modèle Whisper
        lora_config: Configuration LoRA
    
    Returns:
        Modèle avec LoRA appliqué
    """
    model = get_peft_model(model, lora_config)
    return model


def create_lora_whisper_from_config(
    base_model_name: str,
    lora_config: Optional[LoraConfig] = None,
    lora_kwargs: Optional[Dict] = None,
) -> WhisperForConditionalGeneration:
    """
    Crée un modèle Whisper avec LoRA depuis une config.
    
    Args:
        base_model_name: Nom du modèle de base (HuggingFace)
        lora_config: Config LoRA (si None, créée depuis lora_kwargs)
        lora_kwargs: Arguments pour créer config LoRA
    
    Returns:
        Modèle Whisper avec LoRA
    """
    # Charger modèle de base
    model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    
    # Créer config LoRA
    if lora_config is None:
        if lora_kwargs is None:
            lora_kwargs = {}
        lora_config = create_lora_config(**lora_kwargs)
    
    # Appliquer LoRA
    model = apply_lora_to_whisper(model, lora_config)
    
    return model

