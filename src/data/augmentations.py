"""
Augmentations audio spécifiques aux réunions.
Simule les conditions réelles : bruit de fond, écho, qualité variable, etc.
"""

import numpy as np
import librosa
from typing import Optional, Tuple
import random


class AudioAugmentationPipeline:
    """Pipeline d'augmentations audio pour l'entraînement."""
    
    def __init__(
        self,
        enable_noise: bool = True,
        enable_echo: bool = True,
        enable_volume: bool = True,
        enable_codec: bool = True,
        noise_snr_range: Tuple[float, float] = (5, 15),
        echo_delay_range: Tuple[float, float] = (0.1, 0.3),
        volume_gain_range: Tuple[float, float] = (-6, 6),
    ):
        """
        Args:
            enable_noise: Activer ajout de bruit
            enable_echo: Activer simulation écho/réverbération
            enable_volume: Activer variations de volume
            enable_codec: Activer compression codec (via librosa)
            noise_snr_range: Range SNR pour bruit (dB)
            echo_delay_range: Range délai pour écho (seconds)
            volume_gain_range: Range gain volume (dB)
        """
        self.enable_noise = enable_noise
        self.enable_echo = enable_echo
        self.enable_volume = enable_volume
        self.enable_codec = enable_codec
        self.noise_snr_range = noise_snr_range
        self.echo_delay_range = echo_delay_range
        self.volume_gain_range = volume_gain_range
    
    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applique les augmentations à l'audio.
        
        Args:
            audio: Signal audio (1D array)
            sample_rate: Sample rate
        
        Returns:
            Audio augmenté
        """
        augmented = audio.copy()
        
        # Volume (appliqué en premier)
        if self.enable_volume and random.random() > 0.5:
            augmented = self._apply_volume(augmented)
        
        # Bruit
        if self.enable_noise and random.random() > 0.5:
            augmented = self._apply_noise(augmented, sample_rate)
        
        # Écho/réverbération
        if self.enable_echo and random.random() > 0.5:
            augmented = self._apply_echo(augmented, sample_rate)
        
        # Compression codec (simulation via filtrage)
        if self.enable_codec and random.random() > 0.3:
            augmented = self._apply_codec_simulation(augmented, sample_rate)
        
        # Re-normalisation finale
        if np.max(np.abs(augmented)) > 0:
            augmented = augmented / np.max(np.abs(augmented)) * 0.95
        
        return augmented
    
    def _apply_volume(self, audio: np.ndarray) -> np.ndarray:
        """Applique une variation de volume."""
        gain_db = random.uniform(*self.volume_gain_range)
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def _apply_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Ajoute du bruit de fond de type bureau/ambiant.
        Simule bruit de clavier, ventilation, fond sonore léger.
        """
        snr_db = random.uniform(*self.noise_snr_range)
        
        # Générer bruit (mélange gaussien + basses fréquences)
        noise_length = len(audio)
        noise = np.random.normal(0, 1, noise_length).astype(np.float32)
        
        # Ajouter composante basse fréquence (ventilation)
        if noise_length > sample_rate * 0.5:  # Au moins 0.5s
            low_freq_noise = np.sin(
                2 * np.pi * np.arange(noise_length) * 60 / sample_rate
            )  # 60 Hz
            noise = noise * 0.7 + low_freq_noise * 0.3
        
        # Calculer puissance
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Ajuster SNR
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(target_noise_power / (noise_power + 1e-10))
        
        return audio + noise
    
    def _apply_echo(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Simule un écho/réverbération de salle.
        Simple delay avec atténuation.
        """
        delay = random.uniform(*self.echo_delay_range)
        decay = random.uniform(0.3, 0.7)
        
        delay_samples = int(delay * sample_rate)
        echo = np.zeros_like(audio)
        echo[delay_samples:] = audio[:-delay_samples] * decay
        
        # Mix original + écho
        return audio * 0.8 + echo * 0.2
    
    def _apply_codec_simulation(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Simule la compression codec (mp3, opus, etc.).
        Via filtrage passe-bas pour simuler perte haute fréquence.
        """
        # Fréquence de coupure variable (simule qualité codec)
        cutoff_freq = random.uniform(4000, 7000)  # Hz
        
        # Filtre passe-bas simple (Butterworth via scipy si disponible)
        try:
            from scipy import signal
            b, a = signal.butter(4, cutoff_freq / (sample_rate / 2), 'low')
            filtered = signal.filtfilt(b, a, audio)
        except ImportError:
            # Fallback: simple moyenne glissante (moins précis)
            window_size = int(sample_rate / cutoff_freq)
            if window_size > 1:
                filtered = np.convolve(
                    audio,
                    np.ones(window_size) / window_size,
                    mode='same'
                )
            else:
                filtered = audio
        
        # Mix original + filtré pour simuler compression
        return audio * 0.7 + filtered * 0.3


def create_augmentation_pipeline(config: dict) -> Optional[AudioAugmentationPipeline]:
    """
    Crée un pipeline d'augmentations depuis une config YAML.
    
    Args:
        config: Dict avec clés 'enabled', 'noise', 'echo', etc.
    
    Returns:
        AudioAugmentationPipeline ou None si disabled
    """
    if not config.get("enabled", False):
        return None
    
    aug_config = config.get("augmentations", {})
    
    return AudioAugmentationPipeline(
        enable_noise=aug_config.get("noise", {}).get("enabled", False),
        enable_echo=aug_config.get("echo", {}).get("enabled", False),
        enable_volume=aug_config.get("volume", {}).get("enabled", False),
        enable_codec=aug_config.get("codec_compression", {}).get("enabled", False),
        noise_snr_range=(
            aug_config.get("noise", {}).get("min_snr_db", 5),
            aug_config.get("noise", {}).get("max_snr_db", 15),
        ),
        echo_delay_range=(
            aug_config.get("echo", {}).get("min_delay", 0.1),
            aug_config.get("echo", {}).get("max_delay", 0.3),
        ),
        volume_gain_range=(
            aug_config.get("volume", {}).get("min_gain_db", -6),
            aug_config.get("volume", {}).get("max_gain_db", 6),
        ),
    )

