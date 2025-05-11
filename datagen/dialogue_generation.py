import os
import json
import asyncio
from tqdm import tqdm

from config.args import parse_args
from config.set_api_key import set_openai_api
from data_preparation import *
from models import *


async def main(args, vector_db):
    all_model_data = load_and_prepare_utterance_data(args)
    all_model_data = filter_data(all_model_data, args.num_sample)

    args.save_dir = args.save_dir + f"_temp{args.temperature}_0"
    # if exist, make a new dir
    while os.path.exists(args.save_dir):
        file_name, file_num = "_".join(args.save_dir.split("_")[:-1]), int(args.save_dir.split("_")[-1])
        args.save_dir = file_name + "_" + str(file_num + 1)
    os.makedirs(args.save_dir, exist_ok=False)

    # load databases
    with open(args.path_to_abstract_per_user, 'r') as json_file:
        abstracts_per_user = json.load(json_file)
    all_results = []
    chunk = 500
    if len(all_model_data) > chunk:
        for start_idx in tqdm(range(0, len(all_model_data), chunk)):
            cur_model_data = all_model_data[start_idx:start_idx+chunk]
            all_results.extend(await generate_concurrently(cur_model_data, abstracts_per_user, start_idx, vector_db, args))
    else:
        all_results = await generate_concurrently(all_model_data, abstracts_per_user, 0, vector_db, args)

    # save all results
    total_result_path = args.save_dir + "_total_results.json"
    with open(os.path.join(total_result_path), "w", encoding='UTF-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print("Saved at: ", total_result_path)
    
if __name__ == "__main__":
    args = parse_args()
    set_openai_api(args.api_key_name)
    vector_db = save_vector_store(args.candidates_vector_store_dir, args.path_to_abstract_per_user, args.embedding_model_name)
    asyncio.run(main(args, vector_db))