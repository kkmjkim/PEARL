# PEARL
Official repository for “PEARL: A Review-driven Persona-Knowledge Grounded Conversational Recommendation Dataset” accepted at ACL Findings 2024.

## Setup
1. Set environment
```
cd PEARL
conda create -n [YOUR_CONDA_ENV] python=3.10
conda activate [YOUR_CONDA_ENV]
pip install -r requirements.txt
```
2. Make an OpenAI API key [here](https://openai.com/).

## Dialogue Generation
1. Download `resources.zip` from [gdrive](https://drive.google.com/file/d/1-rKk7FCGMUtFLGTmUYEqvij6gw-S9sOh/view?usp=sharing).
2. Unzip `resources.zip`.
3. Extract `vector_store_per_movie.tar.gz` file.
4. Place `vector_store_per_movie`, `abstracts_per_user.json`, and `dialogue_input_data.json` under `data`.
5. Modify `config/personal_info.json`
- Replace `[YOUR_API_NAME]`, `[YOUR_OPENAI_API_KEY]`, and `[YOUR_OPENAI_ORGANIZATION_ID]` with your OpenAI API information.
- You may add more API keys.
6. Modify `scripts/generate_turn_by_turn.sh`
- Replace `[YOUR_API_KEY_NAME]` with your `[YOUR_API_NAME]` in `config/personal_info.json`.
7. Run script
```
sh scripts/generate_turn_by_turn.sh
```