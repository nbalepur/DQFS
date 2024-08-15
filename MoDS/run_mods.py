import argparse
import tqdm
from memory import Memory
from data_loader import ConflictDataset
from llm import LLM
from moderator import Moderator
from speaker import Speaker
from retriever import Retriever
import pickle
import copy
import traceback

from dotenv import load_dotenv, find_dotenv
env_path = ''
load_dotenv(env_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Run the round robin discussion')
    parser.add_argument('--run_name', type=str, default="default_run", help='Run name to identify the inference type.')
    parser.add_argument('--num_to_run', type=int, default=20, help='Number of data instances to run')
    parser.add_argument('--top_k', type=int, default=3, help='Number to retrieve')
    parser.add_argument('--num_topics', type=int, default=3, help='Number of topics to generate')
    parser.add_argument('--res_dir', type=str, default="./", help='Directory for the results')
    parser.add_argument('--use_cot', nargs='+', type=str, default=["False"], help='Use CoT? Enter multiple values separated by spaces.')
    parser.add_argument('--use_rationale', nargs='+', type=str, default=["False"], help='Use the CoT rationale? Enter multiple values separated by spaces.')
    parser.add_argument('--use_subtopic_retrieval', type=str, default="True", help='Use the subtopic for retrieval in round two?')
    parser.add_argument('--select_agents', type=str, default="True", help='Should the moderator select a subset of agents?')
    args = parser.parse_args()
    return args

def mods(idx, ds, llm, num_topics, top_k, use_cot_list, use_rationale_list, use_point_for_retrieval, select_agents, num_tries=0, max_tries=5):
    
    try:
        query, docs = ds.get_item(idx)
        print(f"{idx}) Query (try number {num_tries}):", query)
        base_memory = Memory(query)
        retriever = Retriever(docs, 300, 64, 8, 'colbert-ir/colbertv2.0')
        moderator = Moderator(retriever, llm)
        speakers = [Speaker(retriever, llm, docs_, doc_num) for doc_num, docs_ in enumerate(docs)]

        discussion_points = moderator.plan_discussion_points(query, num_topics, top_k)
        base_memory.set_topics(discussion_points)

        memory_out = dict()

        # all CoT variations
        for use_cot, use_rationale in zip(use_cot_list, use_rationale_list):

            memory = copy.deepcopy(base_memory)
            
            # discussion points
            for point in tqdm.tqdm(memory.get_topics()):
                
                # initialize the memory
                memory.initialize_topic()
                
                # classify relevant agents
                if select_agents:
                    mod_selection = moderator.select_speakers_for_point_question(query, point, top_k, use_cot, use_point_for_retrieval)
                    memory.add_selected_speaker_info(mod_selection)
                
                # speakers
                speaker_list = memory.get_speaker_question_pairs() if select_agents else zip(list(range(len(docs))), [None for _ in docs])
                for speaker_num, gen_question in speaker_list:
                    speaker_out = speakers[speaker_num].speak_rag(query, top_k, point, gen_question if use_rationale else point)
                    memory.add_facts(speaker_out, speaker_num)

            memory_out[(use_cot, use_rationale)] = memory

        return memory_out

    except Exception as e:
        excep = traceback.format_exc()
        print("Overall Exception:", excep)
        if num_tries == max_tries - 1:
            return {k: str(excep) for k in zip(use_cot_list, use_rationale_list)}
        return mods(idx, ds, llm, num_topics, top_k, use_cot_list, use_rationale_list, use_point_for_retrieval, select_agents, num_tries=num_tries+1, max_tries=max_tries)

def save_checkpoint(outputs_dict, out_dict, ds_name, args, USE_POINT_RETRIEVAL, SELECT_AGENTS):

    curr_outputs_dict = copy.deepcopy(outputs_dict)
    curr_out_dict = copy.deepcopy(out_dict)

    for k, v in curr_outputs_dict.items():
        curr_out_dict[k][ds_name] = v

    for k, v in curr_out_dict.items():
        use_cot_, use_rationale_ = k
        with open(f'{args.res_dir}/{args.run_name}/mods_{use_cot_}-CoT_{use_rationale_}-Rationale_{USE_POINT_RETRIEVAL}-PointRetrieval_{SELECT_AGENTS}-Select_TEMP.pkl', 'wb') as handle:
            pickle.dump(v, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(args):

    llm = LLM('GPT4', 0.0, 127000)
    TOP_K = args.top_k
    NUM_TOPICS = args.num_topics
    NUM_TO_RUN = args.num_to_run
    USE_COT = [b == 'True' for b in args.use_cot]
    USE_RATIONALE = [b == 'True' for b in args.use_rationale]
    USE_POINT_RETRIEVAL = args.use_subtopic_retrieval == 'True'
    SELECT_AGENTS = args.select_agents == 'True'
    
    out_dict = {(a, b): dict() for a,b in zip(USE_COT, USE_RATIONALE)}
    for ds_name in ['Debatepedia', 'ConflictingQA']:
        ds = ConflictDataset(ds_name, 1)
        outputs_dict = {(a, b): [] for a,b in zip(USE_COT, USE_RATIONALE)}

        for idx in range(0, NUM_TO_RUN if NUM_TO_RUN != 0 else ds.length()):
            mem_out = mods(idx, ds, llm, NUM_TOPICS, TOP_K, USE_COT, USE_RATIONALE, USE_POINT_RETRIEVAL, SELECT_AGENTS)
            for k, v in mem_out.items():
                outputs_dict[k].append(v)

            if idx % 10 == 0:
                save_checkpoint(outputs_dict, out_dict, ds_name, args, USE_POINT_RETRIEVAL, SELECT_AGENTS)
        
        for k, v in outputs_dict.items():
            out_dict[k][ds_name] = v

    for k, v in out_dict.items():
        use_cot_, use_rationale_ = k
        with open(f'{args.res_dir}/{args.run_name}/mods_{use_cot_}-CoT_{use_rationale_}-Rationale_{USE_POINT_RETRIEVAL}-PointRetrieval_{SELECT_AGENTS}-Select.pkl', 'wb') as handle:
            pickle.dump(v, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    args = parse_args()
    main(args)
