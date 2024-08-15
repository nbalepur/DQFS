import argparse
from summarizer import Summarizer
from llm import LLM
import pickle
import tqdm
import traceback

from dotenv import load_dotenv, find_dotenv
env_path = ''
load_dotenv(env_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize the outline from the agentic framework.')
    parser.add_argument('--run_name', type=str, default="default_run", help='Run name to identify the inference type.')
    parser.add_argument('--res_dir', type=str, default="./", help='Directory for the results')
    parser.add_argument('--use_cot', type=str, default="False", help='Use CoT?')
    parser.add_argument('--use_rationale', type=str, default="False", help='Use the Rationale?')
    parser.add_argument('--num_points', type=int, default=3, help='Number of points to generate')
    parser.add_argument('--use_subtopic_retrieval', type=str, default="True", help='Use the subtopic for retrieval in round two?')
    parser.add_argument('--select_agents', type=str, default="True", help='Should the moderator select a subset of agents?')
    args = parser.parse_args()
    return args

def summarize_outline(curr_outline, summarizer, num_tries=0, max_tries=5):

    if type(curr_outline) == type(''):
        return curr_outline

    try:
        summ = summarizer.summarize_outline_full(curr_outline)
        return summ
    except Exception as e:
        excep = traceback.format_exc()
        print("Overall Exception:", excep)
        if num_tries == max_tries - 1:
            return str(excep)
        return summarize_outline(curr_outline, summarizer, num_tries=num_tries+1, max_tries=max_tries)


def main(args):

    use_cot = (args.use_cot == 'True')
    use_rationale = (args.use_rationale == 'True')
    USE_POINT_RETRIEVAL = (args.use_subtopic_retrieval == 'True')
    SELECT_AGENTS = (args.select_agents == 'True')

    with open(f'{args.res_dir}/{args.run_name}/mods_{use_cot}-CoT_{use_rationale}-Rationale_{USE_POINT_RETRIEVAL}-PointRetrieval_{SELECT_AGENTS}-Select.pkl', 'rb') as handle:
        out = pickle.load(handle)

    llm = LLM('GPT4', 0.0, 127000)
    summarizer = Summarizer(llm, args.num_points)

    out_summaries = dict()
    for k, v in out.items():
        out_summaries[k] = []
        for v_ in tqdm.tqdm(v):
            summ = summarize_outline(v_, summarizer)
            out_summaries[k].append(summ)

    with open(f'{args.res_dir}/{args.run_name}/mods_{use_cot}-CoT_{use_rationale}-Rationale_{USE_POINT_RETRIEVAL}-PointRetrieval_{SELECT_AGENTS}-Select_summary_full.pkl', 'wb') as handle:
        pickle.dump(out_summaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    args = parse_args()
    main(args)
