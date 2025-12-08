# Instructions pour pousser le projet sur GitHub

## 📋 Étapes

### 1. Créer un repo sur GitHub

1. Aller sur https://github.com
2. Cliquer sur "+" → "New repository"
3. Nom : `finetuning_gilbert` (ou autre)
4. **Ne pas** cocher "Initialize with README" (projet existe déjà)
5. Cliquer "Create repository"

### 2. Pousser le projet

Depuis votre terminal Mac, dans le répertoire du projet :

```bash
cd /Users/mathisescriva/CascadeProjects/finetuning_gilbert

# Vérifier les fichiers à commiter
git status

# Ajouter tous les fichiers (sauf ceux dans .gitignore)
git add .

# Créer un commit
git commit -m "Initial commit: Whisper QAT fine-tuning project"

# Ajouter le remote GitHub (remplacez par votre URL)
git remote add origin https://github.com/VOTRE-USERNAME/finetuning_gilbert.git

# Pousser
git branch -M main
git push -u origin main
```

### 3. Si vous devez vous authentifier

Si GitHub demande une authentification :
- Utiliser un **Personal Access Token** (pas le mot de passe)
- Créer un token : GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
- Permissions : `repo` (toutes)
- Utiliser le token comme mot de passe

### 4. Alternative : SSH

Si vous avez configuré SSH sur GitHub :

```bash
git remote add origin git@github.com:VOTRE-USERNAME/finetuning_gilbert.git
git push -u origin main
```

## ✅ Après le push

Une fois sur GitHub, sur Vast.ai :

```bash
cd /workspace
git clone https://github.com/VOTRE-USERNAME/finetuning_gilbert.git
cd finetuning_gilbert
bash setup_and_train.sh
```

---

**Remplacez `VOTRE-USERNAME` par votre nom d'utilisateur GitHub !**

