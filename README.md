# PEARL
Official repository for [_PEARL: A Review-driven Persona-Knowledge Grounded Conversational Recommendation Dataset_](https://aclanthology.org/2024.findings-acl.65/) accepted at ACL Findings 2024.

## Setup
1. Set environment
```
cd PEARL
conda create -n [YOUR_CONDA_ENV] python=3.10
conda activate [YOUR_CONDA_ENV]
pip install -r requirements.txt
```
2. Create an OpenAI API key [here](https://openai.com/).

## Dialogue Generation
1. Download `resources.zip` from [gdrive](https://drive.google.com/file/d/1-rKk7FCGMUtFLGTmUYEqvij6gw-S9sOh/view?usp=sharing).
2. Unzip `resources.zip`.
3. Extract `vector_store_per_movie.tar.gz` file.
4. `mkdir -p data`
5. Place `vector_store_per_movie`, `abstracts_per_user.json`, and `dialogue_input_data.json` under `data`.
6. Modify `datagen/config/personal_info.json`
- Replace `[YOUR_API_KEY_NAME]` and `[YOUR_OPENAI_API_KEY]` with your OpenAI API credentials.
- You may add more API keys to run the code more efficiently (make sure to set the API key name correctly in the script).
7. Modify `scripts/generate_turn_by_turn.sh`
- Replace `[YOUR_API_KEY_NAME]` with one of the keys from `datagen/config/personal_info.json`.
8. Run script
```
sh scripts/generate_turn_by_turn.sh
```