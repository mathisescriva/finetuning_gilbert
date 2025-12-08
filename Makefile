# Makefile pour faciliter l'exécution des scripts

.PHONY: install download-datasets generate-transcripts eval-baseline train-phase1 train-phase2 train-lora quantize benchmark help

help:
	@echo "Commandes disponibles:"
	@echo "  make install          - Installer les dépendances"
	@echo "  make download-datasets - Télécharger datasets publics français"
	@echo "  make estimate-cost - Estimer le coût de transcription (AssemblyAI, etc.)"
	@echo "  make generate-transcripts - Générer transcripts automatiques pour dataset audio"
	@echo "  make generate-transcripts-assemblyai - Générer avec AssemblyAI (service commercial)"
	@echo "  make eval-baseline    - Évaluer le modèle baseline"
	@echo "  make train-phase1     - Fine-tuning phase 1 (encoder frozen)"
	@echo "  make train-phase2     - Fine-tuning phase 2 (full)"
	@echo "  make train-lora       - Fine-tuning LoRA"
	@echo "  make train-qat-int8   - Entraînement QAT (Quantization-Aware Training)"
	@echo "  make evaluate-qat     - Évaluer modèle QAT/quantifié"
	@echo "  make quantize         - Quantifier le modèle"
	@echo "  make benchmark        - Benchmark comparatif"

download-datasets:
	python scripts/download_datasets.py \
		--datasets common_voice \
		--output_dir data/processed

generate-transcripts:
	python scripts/generate_transcripts.py \
		--dataset_name MEscriva/french-education-speech \
		--output_dir data/processed

estimate-cost:
	python scripts/estimate_cost.py \
		--dataset_name MEscriva/french-education-speech

generate-transcripts-assemblyai:
	python scripts/generate_transcripts_commercial.py \
		--dataset_name MEscriva/french-education-speech \
		--service assemblyai \
		--output_dir data/processed

install:
	pip install -r requirements.txt

eval-baseline:
	python scripts/evaluate_baseline.py \
		--model_name bofenghuang/whisper-large-v3-distil-fr-v0.2 \
		--test_data data/test_sets/example_test_data.json \
		--output_dir outputs/evaluations

train-phase1:
	python scripts/fine_tune_meetings.py \
		--train_data data/raw/train_data.json \
		--eval_data data/raw/eval_data.json \
		--output_dir outputs/models/whisper-meetings-phase1 \
		--phase 1

train-phase2:
	python scripts/fine_tune_meetings.py \
		--train_data data/raw/train_data.json \
		--eval_data data/raw/eval_data.json \
		--output_dir outputs/models/whisper-meetings-phase2 \
		--phase 2 \
		--model_name outputs/models/whisper-meetings-phase1/final

train-qat-int8:
	python scripts/train_qat.py \
		--base_model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
		--train_data data/processed/common_voice_fr \
		--eval_data data/processed/common_voice_fr \
		--quantization_type int8 \
		--num_epochs 5 \
		--max_samples 60000 \
		--per_device_batch_size 8 \
		--output_dir outputs/models/whisper-qat-int8

evaluate-qat:
	python scripts/evaluate_qat.py \
		--model_path outputs/models/whisper-qat-int8-quantized \
		--baseline_model bofenghuang/whisper-large-v3-distil-fr-v0.2 \
		--test_data data/test_sets/eval_data.json

train-lora:
	python scripts/fine_tune_meetings.py \
		--train_data data/raw/train_data.json \
		--eval_data data/raw/eval_data.json \
		--output_dir outputs/models/whisper-meetings-lora \
		--phase 3 \
		--use_lora

quantize:
	python scripts/distill_quantize.py \
		--model_path outputs/models/whisper-meetings-phase2/final \
		--output_dir outputs/models/whisper-meetings-int8 \
		--quantization_type int8

benchmark:
	python scripts/benchmark.py \
		--test_data data/test_sets/example_test_data.json \
		--models \
			openai/whisper-large-v3 \
			bofenghuang/whisper-large-v3-distil-fr-v0.2 \
		--output_dir outputs/evaluations

