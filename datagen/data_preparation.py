import json
import random
from tqdm import tqdm

def load_dialogue_prompt(args):
    """
    Load .txt file as a prompt.
    """
    seeker_prompt = None
    recommender_prompt = None
    if args.seeker_prompt_path:
        with open(args.seeker_prompt_path, 'r') as f1:
            seeker_prompt = f1.read()
    if args.recommender_prompt_path:
        with open(args.recommender_prompt_path, 'r') as f2:
            recommender_prompt = f2.read()
    return seeker_prompt, recommender_prompt

def prepare_model_input_utterance(seeker_prompt:str, recommender_prompt:str, data_path:str, seed_dialogue_path:str):
    '''Prepare data for utterance generation from simulators.'''
    print("Loading data for generating utterances...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    with open(seed_dialogue_path, 'r') as f:
        seed_dialogue = json.load(f)
    all_model_data = []
    for d in tqdm(data):
        input_temp = dict()
        input_temp["data_id"] = d["data_id"]
        input_temp["user_id"] = d["user_id"]
        input_temp["user_persona"] = '\n\n'.join(abstract['abstract'].strip() for abstract in d['seen_abstracts'])
        input_temp["seen_movie_titles"] = [abstract['title'] for abstract in d['seen_abstracts']]
        input_temp['gt_abstract'] = d["gt_abstract"]['abstract'].strip()
        input_temp["gt_movie_title"] = d["gt_abstract"]['title']
        input_temp["gt_genre"] = ", ".join(d["gt_abstract"]['genres'])
        input_temp["gt_director"] = ", ".join(d["gt_abstract"]['director'])
        input_temp["gt_cast"] = ", ".join(d["gt_abstract"]['cast'][:3])
        input_temp["gt_abstract"] = f"Title: {input_temp['gt_movie_title']}\nGenre: {input_temp['gt_genre']}\nDirector: {input_temp['gt_director']}\nCast: {input_temp['gt_cast']}\nReview: {input_temp['gt_abstract']}"
        input_temp['user_sim_input'] = seeker_prompt.format(**{
            "gt_movie_title": d["gt_abstract"]['title'],
            "user_persona": input_temp["user_persona"],   
            "gt_abstract": input_temp['gt_abstract'],    
            "rec_movie_abstract": "",  # fill if seeker is recommended a movie.
            "dialogue_context": seed_dialogue
        })
        input_temp["rec_sim_input"] = recommender_prompt.format(**{
            "k_movies_info": "",  # fill if the topk are retrieved.
            "dialogue_context": seed_dialogue,
        })
        all_model_data.append(input_temp)
    return all_model_data

def load_and_prepare_utterance_data(args):
    seeker_prompt, recommender_prompt = load_dialogue_prompt(args)
    print("Preparing model inputs...")
    all_model_data = prepare_model_input_utterance(
        seeker_prompt, recommender_prompt, args.input_path, args.seed_dialogue_path)
    return all_model_data

def sample_indices(all_model_inputs, num_sample):
    random.seed(0)
    cand_indices = list(range(len(all_model_inputs)))
    sampled_indices = random.sample(cand_indices, num_sample)
    return sampled_indices

def filter_data(all_model_data, num_sample):
    if num_sample:
        sampled_indices = sample_indices(all_model_data, num_sample)
        all_model_data = [all_model_data[i] for i in sampled_indices]
    return all_model_data