import json
import os
import pickle
from copy import deepcopy
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage

from data_preparation import load_dialogue_prompt

def retrieve_candidate_with_vector_db_multi(dialogue_context, k, vector_store_for_user, embedding_model):
    """
    Returns top k documents from the vectorstore of a user, based on the similarity with the dialogue context(query).
    """
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
        vector_store_dict[user_id] = seen_candidates
    return vector_store_dict

async def async_generate(seeker, recommender, seed_dialogue_path, model_data, abstracts_per_user, idx, save_dir, vector_db, embedding_model, args):
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