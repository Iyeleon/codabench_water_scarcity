.PHONY: prep_dataset
prep_dataset: src/preprocessing.py ./data/dataset/
	@python -m src.preprocessing

.PHONY: prep_mini_dataset
prep_mini_dataset: src/preprocessing.py ./data/dataset ./data/dataset_mini_challenge/
	@python -m src.preprocessing --is-mini	
