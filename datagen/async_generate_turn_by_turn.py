import argparse
import asyncio
import json
import os
import random
from copy import deepcopy
import pandas as pd
import re
from collections import defaultdict
import time
import tiktoken

from tqdm import tqdm
# from nltk import word_tokenize
from tqdm.asyncio import tqdm_asyncio

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from rank_bm25 import BM25Okapi
import pickle
# from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed

TOTAL_COST = 0
retrieve_call_time = 0

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

def set_openai_api(api_key_name):
    personal_info = json.load(open("config/personal_info.json", "r"))
    assert api_key_name in personal_info
    os.environ["OPENAI_API_KEY"] = personal_info[api_key_name]["api_key"]
    os.environ["OPENAI_ORGANIZATION"] = personal_info[api_key_name]["org_id"]
    print(f"Set OpenAI API Key and Organization of {api_key_name}.")


## TODO : change this function to load the prompt from your own file ##
def load_dialogue_prompt(args):
    """
    Load .txt file as a prompt.
    """
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

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def retrieve_candidate_with_vector_db_multi(dialogue_context, k, vector_store_for_user, embedding_model):
    """
    Returns top k documents from the vectorstore of a user, based on the similarity with the dialogue context(query).
    """
    global retrieve_call_time
    start_time = time.time()

    query = "\n".join(dialogue_context)
    query_embedding_vector = embedding_model.embed_query(query)
    docs_and_scores = []
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
        future_to_vector_store = {executor.submit(vector_store['db'].similarity_search_with_score_by_vector, query_embedding_vector): vector_store for vector_store in vector_store_for_user}
        for future in as_completed(future_to_vector_store):
            docs_scores = future.result()
            docs_and_scores.extend(docs_scores)
    docs_and_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in docs_and_scores[:k]]
    end_time = time.time()
    retrieve_call_time += (end_time - start_time)

    # calculate cost
    global TOTAL_COST
    n_embedded_tokens = num_tokens_from_string(query, "cl100k_base")
    TOTAL_COST += n_embedded_tokens / 1000000 * 0.1
    
    return top_docs

def save_vector_store(candidates_vector_store_dir, path_to_abstract_per_user, embedding_model_name):
    """
    Construct movie knowledge vectorstore (retrieval pool) per user ID.
    A db of a user contains knowledge embeddings of movies for which a user had written a review.
    """
    with open(path_to_abstract_per_user, 'r') as json_file:
        user_seen_movie = json.load(json_file)
        
    embeddings = OpenAIEmbeddings(model=embedding_model_name)
    vector_store_dict = {}
    print("Constructing vectorstore for embedded movie knowledge per user...")
    for user_id, seen_movie_info in tqdm(user_seen_movie.items()):
        seen_candidates = []
        for movie in seen_movie_info:
            imdb_id = movie['imdb_id']
            title = movie['title']
            try:
                # get general abstract (candidate documents)
                with open(os.path.join(candidates_vector_store_dir, f"{imdb_id}.pkl"), "rb") as f:
                    serialized_data = pickle.load(f)
                db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=serialized_data)
                seen_candidates.append({'imdb_id': imdb_id, 'title': title, 'db': db})
            except:
                print(f"Error occurred while loading {imdb_id}.pkl")
                pass
        vector_store_dict[user_id] = seen_candidates
    return vector_store_dict


async def async_generate(seeker, recommender, seed_dialogue_path, model_data, abstracts_per_user, idx, save_dir, vector_db, embedding_model, args):
    global TOTAL_COST
    try:
        seeker_prompt, recommender_prompt = load_dialogue_prompt(args)
        user_persona, gt_abstract = model_data["user_persona"], model_data["gt_abstract"]
        gt_movie_title = model_data["gt_movie_title"]
        user_id = model_data['user_id']

        # lists for record
        retriever_input = []
        retrieved_abstract = []
        all_rec_movie_user_abstracts = []
        record_dialogue_context = []
        rec_raw_responses = []
        all_retrieved_movie_titles = []

        # set seed dialogue
        dialogue_context = json.load(open(seed_dialogue_path, "r"))
        
        # prepare candidate movies
        user_abstracts = abstracts_per_user[model_data['user_id']]

        # make it as a dictionary for easy access; processing done in advance
        user_abstract_dict = dict()
        for abstract in user_abstracts:
            processed_abstract_title = abstract['title'].replace("\t", " ").replace("  ", " ").strip().lower()
            user_abstract_dict[processed_abstract_title] = abstract['abstract']
        
        vector_store_for_user = vector_db[user_id]
        MAX_VALID_TURN = 6
        MAX_TURN = MAX_VALID_TURN + 1  # to get the seeker's reaction to the last recommender response
        MAX_POOL_SIZE = 3
        for turn in range(1, MAX_TURN):
            seeker_model_input = model_data['user_sim_input'] 
            human_message1 = HumanMessage(content=seeker_model_input)
            response1 = await seeker.agenerate([[human_message1]], stop=["Recommender:"])
            processed_response1 = "Seeker: " + response1.generations[0][0].text.split("Seeker:")[-1].strip()  # force name tag
            dialogue_context.append(processed_response1)

            # retrieve max. TOP_K items among the movies the user had watched
            if turn == MAX_VALID_TURN:
                TOP_K = 1
            else:
                TOP_K = min(MAX_POOL_SIZE, MAX_VALID_TURN-turn)

            topk_movies_idx = retrieve_candidate_with_vector_db_multi(dialogue_context[2:], TOP_K, vector_store_for_user, embedding_model)
            retriever_input.append("\n".join(dialogue_context))   # to record

            retrieved_movie_titles = [i.page_content.split('\n')[0].split('Title: ')[1] for i in topk_movies_idx]
            all_retrieved_movie_titles.append(retrieved_movie_titles)
            if (turn >= (MAX_VALID_TURN-MAX_POOL_SIZE)) and (gt_movie_title not in retrieved_movie_titles):  # if start narrowing down, and if gt not included
                # print(f"Turn {turn}. GT not included in top {TOP_K} retrieved movies. Replacing lowest rank with gt.")
                topk_movie_info_list = [i.page_content for i in topk_movies_idx[:TOP_K-1]] + [gt_abstract]
            else:  # if not yet, or already included
                topk_movie_info_list = [i.page_content for i in topk_movies_idx]
            
            topk_movie_info = '\n\n'.join(topk_movie_info_list)
            retrieved_abstract.append(topk_movie_info_list)

            # update recommender model input 
            model_data["rec_sim_input"] = recommender_prompt.format(**{
                "k_movies_info": topk_movie_info,
                "dialogue_context":"\n".join(dialogue_context),
            })

            # get recommender response
            recommender_model_input = model_data['rec_sim_input']
            human_message2 = HumanMessage(content=recommender_model_input)
            response2 = await recommender.agenerate([[human_message2]], stop=["Seeker:"])
            raw_response = response2.generations[0][0].text
            rec_raw_responses.append(raw_response)

            recommended_movie = ""
            if "Movie:" in raw_response:
                recommended_movie = raw_response.split("Movie:")[-1].strip().split("\n")[0].replace("Title:", "").replace('"', '').strip()
                # remove movie from the vector store if already recommended
                for store in vector_store_for_user:
                    store_title = store['title'].replace("\t", " ").replace("  ", " ")
                    if store_title.strip().lower() == recommended_movie.strip().lower():
                        vector_store_for_user.remove(store)
                        break
                    
            processed_response2 = "Recommender: " + raw_response.split("Recommender:")[-1].split("\n")[0].strip()
            dialogue_context.append(processed_response2)

            # get the user review abstract of the recommendation
            rec_movie_user_abstract = None
            try:
                rec_movie_user_abstract = "Here are your thoughts about the recommendation:\n" + user_abstract_dict[recommended_movie.strip().lower()]
            except:
                rec_movie_user_abstract = ""

            all_rec_movie_user_abstracts.append(rec_movie_user_abstract)
            # update user_sim_input
            model_data['user_sim_input'] = seeker_prompt.format(**{
                "gt_movie_title": gt_movie_title,
                "user_persona": user_persona,
                "gt_abstract": gt_abstract,
                "rec_movie_abstract": rec_movie_user_abstract,
                "dialogue_context":"\n".join(dialogue_context)
            })
                
            record_dialogue_context.append("\n".join(dialogue_context))

            # calculate cost
            input_tokens = response1.llm_output['token_usage']['prompt_tokens'] + response2.llm_output['token_usage']['prompt_tokens']
            output_tokens = response1.llm_output['token_usage']['completion_tokens'] + response2.llm_output['token_usage']['completion_tokens']
            if seeker.model_name == "gpt-3.5-turbo-1106":
                TOTAL_COST += input_tokens / 1000 * 0.001
                TOTAL_COST += output_tokens / 1000 * 0.002
            elif seeker.model_name == "gpt-4-1106-preview":
                TOTAL_COST += input_tokens / 1000 * 0.01
                TOTAL_COST += output_tokens / 1000 * 0.03
            print(f"{model_data['data_id']}-{turn}", TOTAL_COST)
                
        # Change this code if you want to save it in a different way
        result = deepcopy(model_data)
        result['prediction'] = dialogue_context
        for i in range(len(rec_raw_responses)):
            top_3 = [abstract for abstract in retrieved_abstract[i]]
            result[f"turn_{i+2}"] = {
                "retriever_input": retriever_input[i],
                "retrieved_top3_titles": all_retrieved_movie_titles[i],
                "top_3": "\n\n".join(top_3),
                "rec_user_abstract": all_rec_movie_user_abstracts[i],
                "rec_raw_response": rec_raw_responses[i],
            }
        result['user_sim'] = result.pop('user_sim_input')
        result['rec_sim'] = result.pop('rec_sim_input')
        result["prediction_str"] = "\n".join(dialogue_context)
        with open(os.path.join(save_dir, f"{idx}.json"), "w", encoding='UTF-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Exception occurred: {e}")
        result = deepcopy(model_data)
        result["failed"] = True
        with open(os.path.join(save_dir, f"{idx}.json"), "w", encoding='UTF-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    return result


async def generate_concurrently(all_model_data, user_abstracts, start_idx, vector_db, args):
    seeker = ChatOpenAI(model_name=args.model_name,
                     temperature=args.temperature,
                     max_tokens=args.max_tokens,
                     max_retries=100)
    
    recommender = ChatOpenAI(model_name=args.model_name,
                     temperature=args.temperature,
                     max_tokens=args.max_tokens,
                     max_retries=100)
    embedding_model = OpenAIEmbeddings(model=args.embedding_model_name)
    
    tasks = [async_generate(seeker, recommender, args.seed_dialogue_path, model_data,
                            user_abstracts, i+start_idx, args.save_dir, vector_db, embedding_model, args)
             for i, model_data in enumerate(all_model_data)]
    return await tqdm_asyncio.gather(*tasks)


async def main(args, vector_db):
    global retrieve_call_time
    start_time = time.time() 
    
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
    print(f"Total cost: {TOTAL_COST}")
    print("Saved at: ", total_result_path)

    end_time = time.time() 

    total_duration = end_time - start_time 
    total_retrieve = retrieve_call_time 

    print(f"Total process took {total_duration:.2f} seconds.")  
    print(f"Total retrieval took {total_retrieve:.2f} seconds.") 
    
if __name__ == "__main__":
    args = parse_args()
    set_openai_api(args.api_key_name)
    vector_db = save_vector_store(args.candidates_vector_store_dir, args.path_to_abstract_per_user, args.embedding_model_name)
    asyncio.run(main(args, vector_db))
    print(f'Total cost for generating conversations(generation + embedding): ', TOTAL_COST)