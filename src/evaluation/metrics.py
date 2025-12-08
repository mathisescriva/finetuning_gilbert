"""
Métriques d'évaluation pour ASR : WER, CER, et métriques spécialisées.
"""

import jiwer
import re
import numpy as np
from typing import Dict, List, Optional, Set
from collections import defaultdict


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Calcule le Word Error Rate (WER).
    
    Args:
        reference: Texte de référence
        hypothesis: Texte prédit
    
    Returns:
        WER en pourcentage (0-100)
    """
    return jiwer.wer(reference, hypothesis) * 100


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Calcule le Character Error Rate (CER).
    
    Args:
        reference: Texte de référence
        hypothesis: Texte prédit
    
    Returns:
        CER en pourcentage (0-100)
    """
    return jiwer.cer(reference, hypothesis) * 100


def normalize_text(text: str) -> str:
    """
    Normalise le texte pour évaluation (minuscules, ponctuation, etc.).
    """
    # Minuscules
    text = text.lower()
    
    # Normaliser espaces
    text = re.sub(r'\s+', ' ', text)
    
    # Garder accents français (pas de suppression)
    # Juste normaliser ponctuation
    text = re.sub(r'[^\w\sàâäéèêëïîôùûüÿç]', '', text)
    
    return text.strip()


def extract_named_entities(text: str, entities: Optional[Set[str]] = None) -> List[str]:
    """
    Extrait les entités nommées potentielles du texte.
    Basique : mots en majuscules ou mots dans une liste fournie.
    
    Args:
        text: Texte à analyser
        entities: Set d'entités connues (noms propres, acronymes)
    
    Returns:
        Liste d'entités trouvées
    """
    found = []
    
    # Mots en majuscules (acronymes, noms propres)
    words = text.split()
    for word in words:
        # Acronymes (tout en majuscules, 2+ caractères)
        if word.isupper() and len(word) >= 2:
            found.append(word)
        # Noms propres (première lettre majuscule, pas en début de phrase)
        elif word and word[0].isupper() and word.lower() != word.lower().capitalize():
            found.append(word)
    
    # Entités fournies
    if entities:
        for entity in entities:
            if entity.lower() in text.lower():
                found.append(entity)
    
    return list(set(found))


def compute_specialized_wer(
    references: List[str],
    hypotheses: List[str],
    entity_list: Optional[Set[str]] = None,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Calcule WER spécialisé pour différents types de contenu.
    
    Args:
        references: Liste de textes de référence
        hypotheses: Liste de textes prédits
        entity_list: Set d'entités connues (participants, entreprises, etc.)
        normalize: Normaliser les textes avant calcul
    
    Returns:
        Dict avec métriques :
        - 'overall_wer': WER global
        - 'overall_cer': CER global
        - 'named_entities_wer': WER sur entités nommées uniquement
        - 'acronyms_wer': WER sur acronymes
    """
    overall_errors = []
    overall_chars_ref = 0
    overall_chars_hyp = 0
    entity_errors = []
    acronym_errors = []
    
    all_entities_ref = []
    all_entities_hyp = []
    all_acronyms_ref = []
    all_acronyms_hyp = []
    
    for ref, hyp in zip(references, hypotheses):
        if normalize:
            ref_norm = normalize_text(ref)
            hyp_norm = normalize_text(hyp)
        else:
            ref_norm = ref
            hyp_norm = hyp
        
        # WER/CER global
        overall_errors.append(jiwer.wer(ref_norm, hyp_norm))
        overall_chars_ref += len(ref_norm.replace(' ', ''))
        overall_chars_hyp += len(hyp_norm.replace(' ', ''))
        
        # Extraire entités
        entities_ref = extract_named_entities(ref, entity_list)
        entities_hyp = extract_named_entities(hyp, entity_list)
        all_entities_ref.extend(entities_ref)
        all_entities_hyp.extend(entities_hyp)
        
        # Extraire acronymes (tout en majuscules)
        acronyms_ref = [e for e in entities_ref if e.isupper() and len(e) >= 2]
        acronyms_hyp = [e for e in entities_hyp if e.isupper() and len(e) >= 2]
        all_acronyms_ref.extend(acronyms_ref)
        all_acronyms_hyp.extend(acronyms_hyp)
    
    # Métriques globales
    overall_wer = np.mean(overall_errors) * 100 if overall_errors else 0.0
    
    # CER approximatif
    overall_cer = (
        abs(overall_chars_hyp - overall_chars_ref) / overall_chars_ref * 100
        if overall_chars_ref > 0
        else 0.0
    )
    
    # WER entités (simple comparaison de sets)
    if all_entities_ref or all_entities_hyp:
        # Compter entités correctes
        entities_ref_set = set([e.lower() for e in all_entities_ref])
        entities_hyp_set = set([e.lower() for e in all_entities_hyp])
        correct = len(entities_ref_set & entities_hyp_set)
        total = len(entities_ref_set)
        entity_wer = (1 - correct / total) * 100 if total > 0 else 0.0
    else:
        entity_wer = 0.0
    
    # WER acronymes
    if all_acronyms_ref or all_acronyms_hyp:
        acronyms_ref_set = set([e.lower() for e in all_acronyms_ref])
        acronyms_hyp_set = set([e.lower() for e in all_acronyms_hyp])
        correct = len(acronyms_ref_set & acronyms_hyp_set)
        total = len(acronyms_ref_set)
        acronym_wer = (1 - correct / total) * 100 if total > 0 else 0.0
    else:
        acronym_wer = 0.0
    
    return {
        "overall_wer": overall_wer,
        "overall_cer": overall_cer,
        "named_entities_wer": entity_wer,
        "acronyms_wer": acronym_wer,
    }


def compute_detailed_metrics(
    references: List[str],
    hypotheses: List[str],
    entity_list: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """
    Calcule un ensemble complet de métriques détaillées.
    
    Returns:
        Dict avec toutes les métriques
    """
    # Normaliser
    refs_norm = [normalize_text(r) for r in references]
    hyps_norm = [normalize_text(h) for h in hypotheses]
    
    # Métriques spécialisées
    specialized = compute_specialized_wer(refs_norm, hyps_norm, entity_list, normalize=False)
    
    # Métriques jiwer additionnelles
    measures = jiwer.compute_measures(
        truth=refs_norm,
        hypothesis=hyps_norm,
    )
    
    return {
        **specialized,
        "wer": measures.get("wer", 0.0) * 100,
        "cer": measures.get("mer", 0.0) * 100,  # MER ≈ CER
        "substitutions": measures.get("substitutions", 0),
        "deletions": measures.get("deletions", 0),
        "insertions": measures.get("insertions", 0),
        "hits": measures.get("hits", 0),
    }

