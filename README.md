# PEARL

Official repository for "PEARL: A Review-driven Persona-Knowledge Grounded Conversational Recommendation Dataset" accepted at ACL Findings 2024.

## Setup
1. Set environment
```
cd PEARL
conda create -n crs python=3.10
conda activate crs
pip install -r requirements.txt
```

2. Make an OpenAI API key [here](https://openai.com/).

## Dialogue Generation
1. Download vector store for movie knowledge from [gdrive](https://drive.google.com/file/d/1bbDctgmERNv8IOjdHlQOzztEb0V_IWur/view?usp=sharing).

2. Extract and place it under `data/`

3. Modify `config/personal_info.json`
- Replace `[YOUR_API_NAME]`, `[YOUR_OPENAI_API_KEY]`, `[YOUR_OPENAI_ORGANIZATION_ID]` with your OpenAI information.
- You may add more API keys.

4. Modify `scripts/generate_turn_by_turn.sh`
- Replace `[YOUR_API_KEY_NAME]` with your `[YOUR_API_NAME]` in `config/personal_info.json`.

5. Run script
```
sh scripts/generate_turn_by_turn.sh
```