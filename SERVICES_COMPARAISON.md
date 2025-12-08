# Comparaison Services de Transcription

## 🏆 Services Disponibles

### 1. AssemblyAI ⭐ RECOMMANDÉ

**Pourquoi c'est le meilleur choix :**
- ✅ Excellent rapport qualité/prix
- ✅ Très bon pour le français
- ✅ API simple et rapide
- ✅ 50$ de crédit gratuit pour commencer
- ✅ Diarisation, timestamps inclus

**Prix :**
- $0.0001 par minute audio (~$0.006/heure)
- Crédit gratuit de $50 = ~833 heures gratuites

**Installation :**
```bash
pip install assemblyai
```

**Utilisation :**
```bash
export ASSEMBLYAI_API_KEY="votre_cle"
python scripts/generate_transcripts_commercial.py \
  --service assemblyai \
  --dataset_name MEscriva/french-education-speech
```

**Note :** Créer un compte sur https://www.assemblyai.com pour obtenir la clé API gratuite.

---

### 2. Deepgram

**Avantages :**
- ✅ Très haute précision (modèle Nova-2)
- ✅ Bon support français
- ✅ API moderne

**Prix :**
- $0.0043 par minute (~$0.26/heure)
- Plus cher qu'AssemblyAI

**Installation :**
```bash
pip install deepgram-sdk
```

---

### 3. Azure Speech Services

**Avantages :**
- ✅ Très bon support multilingue
- ✅ Intégration Azure si déjà utilisateur
- ✅ Modèles personnalisables

**Prix :**
- $1.00 par heure (prix standard)
- Plus cher mais très fiable

**Installation :**
```bash
pip install azure-cognitiveservices-speech
```

---

### 4. Google Cloud Speech-to-Text

**Avantages :**
- ✅ Excellence qualité (surtout Google)
- ✅ Support très large langues

**Prix :**
- $0.006 par 15 secondes (~$1.44/heure)
- Le plus cher mais souvent meilleure qualité

**Installation :**
```bash
pip install google-cloud-speech
# Nécessite fichier credentials JSON
```

---

### 5. Whisper (Open Source) - Alternative Gratuite

**Avantages :**
- ✅ **100% gratuit**
- ✅ Fonctionne offline
- ✅ Pas de limites

**Inconvénients :**
- ⚠️ Qualité généralement inférieure aux services commerciaux
- ⚠️ Plus lent (pas d'API optimisée)
- ⚠️ Nécessite GPU pour bonne performance

**Quand l'utiliser :**
- Budget très limité
- Données sensibles (offline nécessaire)
- Test rapide avant d'investir

---

## 📊 Comparaison Rapide

| Service | Qualité | Prix/heure | Gratuit | Recommandation |
|---------|---------|------------|---------|----------------|
| **AssemblyAI** | ⭐⭐⭐⭐ | ~$0.006 | $50 crédit | ✅ **Meilleur choix** |
| **Deepgram** | ⭐⭐⭐⭐⭐ | ~$0.26 | Non | Bon si budget OK |
| **Azure** | ⭐⭐⭐⭐ | ~$1.00 | Non | Si déjà Azure |
| **Google** | ⭐⭐⭐⭐⭐ | ~$1.44 | Non | Meilleure qualité mais cher |
| **Whisper** | ⭐⭐⭐ | Gratuit | Oui | Alternative économique |

## 💰 Estimation de Coût

Pour votre dataset `french-education-speech`, estimons :

**Exemple :** 100 heures d'audio
- AssemblyAI : ~$0.60 (dans les crédits gratuits !)
- Deepgram : ~$26
- Azure : ~$100
- Google : ~$144
- Whisper : Gratuit (mais qualité moindre)

**Recommandation :** Commencez par AssemblyAI avec le crédit gratuit, puis comparez la qualité avec Whisper.

## 🎯 Quelle Approche Choisir ?

### Pour Qualité Maximale (Pseudo-labels)

```bash
# 1. Service commercial (AssemblyAI recommandé)
export ASSEMBLYAI_API_KEY="votre_cle"
python scripts/generate_transcripts_commercial.py \
  --service assemblyai \
  --dataset_name MEscriva/french-education-speech
```

### Pour Économie Maximale

```bash
# Whisper gratuit
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech
```

### Approche Hybride (Recommandée)

1. **Générer 10-20% avec service commercial** (AssemblyAI) pour validation
2. **Comparer qualité** avec Whisper sur même échantillon
3. **Décider** :
   - Si différence notable → Service commercial pour tout
   - Si Whisper suffit → Utiliser Whisper (économique)
   - Mix des deux selon budget

## 📝 Exemple d'Approche Pragmatique

```bash
# 1. Test avec 10 échantillons sur AssemblyAI (gratuit)
python scripts/generate_transcripts_commercial.py \
  --dataset_name MEscriva/french-education-speech \
  --service assemblyai \
  --max_samples 10

# 2. Test avec 10 échantillons sur Whisper
python scripts/generate_transcripts.py \
  --dataset_name MEscriva/french-education-speech \
  --max_samples 10

# 3. Comparer les résultats
# Si AssemblyAI nettement meilleur → utiliser pour tout
# Si comparable → Whisper pour économiser

# 4. Générer tout le dataset avec service choisi
```

## ✅ Recommandation Finale

**Pour votre cas (french-education-speech) :**

1. **Commencez par AssemblyAI** : 
   - Crédit gratuit $50 = ~833 heures
   - Qualité très bonne
   - Si votre dataset < 833h, c'est gratuit !

2. **Comparez avec Whisper** :
   - Test sur échantillon
   - Si différence mineure → Whisper pour le reste

3. **Approche itérative** :
   - Générer transcripts avec AssemblyAI
   - Fine-tuner Whisper
   - Régénérer avec Whisper fine-tuné (moins cher)

