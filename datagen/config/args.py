import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key_name", type=str, help="The name of the OpenAI api key. Make sure config/personal_info.json exists and both org and api info in it.")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--path_to_abstract_per_user", type=str)
    parser.add_argument("--candidates_vector_store_dir", type=str, help="Directory path to the vector store for candidate documents of general abstracts.")
    parser.add_argument("--seeker_prompt_path", type=str, default=None)
    parser.add_argument("--recommender_prompt_path", type=str, default=None)
    parser.add_argument("--seed_dialogue_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True, help="It should be a NEW DIRECTORY. Please do not use an existing one.")
    parser.add_argument("--num_sample", type=int, default=None, help="If you want to test your code by sampling a small number of data, you can set this argument.")
    parser.add_argument("--model_name", choices=["gpt-3.5-turbo-1106", "gpt-4-1106-preview"], type=str, default="gpt-3.5-turbo-1106", help="OpenAI model name. Options: gpt-3.5-turbo-1106 or gpt-4-1106-preview")
    parser.add_argument("--embedding_model_name", type=str, help="OpenAI embedding model name.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=500)
    args = parser.parse_args()
    if args.num_sample:
        args.save_dir = args.save_dir + f"_sample{args.num_sample}"
    return args 