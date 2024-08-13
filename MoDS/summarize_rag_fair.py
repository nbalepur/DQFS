import argparse
from summarizer import Summarizer
from llm import LLM
from data_loader import ConflictDataset
from rag import RAG
import pickle
import tqdm
import traceback

from dotenv import load_dotenv, find_dotenv
env_path = '/sensei-fs-3/users/nbalepur/keys.env'
load_dotenv(env_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Summarize the documents with RAG.')
    parser.add_argument('--run_name', type=str, default="default_run", help='Run name to identify the inference type.')
    parser.add_argument('--num_to_run', type=int, default=20, help='Number of data instances to run')
    parser.add_argument('--top_k', type=int, default=5, help='Number of items to retrieve')
    parser.add_argument('--res_dir', type=str, default="./", help='Directory for the results')
    parser.add_argument('--num_points', type=int, default=3, help='Number of points to generate')
    args = parser.parse_args()
    return args

def summarize(idx, summarizer, TOP_K, ds, num_tries=0, max_tries=5):

    try:
        query, docs = ds.get_item(idx)
        rag = RAG(docs, 300, 64, 8, 'colbert-ir/colbertv2.0', summarizer)
        summ = rag.generate_fair(query, summarizer.num_points * TOP_K)
        return summ
    except Exception as e:
        excep = traceback.format_exc()
        print("Overall Exception:", excep)
        if num_tries == max_tries - 1:
            return str(excep)
        return summarize(idx, summarizer, TOP_K, ds, num_tries=num_tries+1, max_tries=max_tries)

def main(args):

    NUM_TO_RUN = args.num_to_run
    TOP_K = args.top_k

    llm = LLM('GPT4', 0.0, 127000)
    summarizer = Summarizer(llm, args.num_points)
    out_summaries = dict()

    for ds_name in ['Debatepedia', 'ConflictingQA']:

        ds = ConflictDataset(ds_name, 1)

        out_summaries[ds_name] = []
        for idx in tqdm.tqdm(range(NUM_TO_RUN if NUM_TO_RUN != 0 else ds.length())):
            summ = summarize(idx, summarizer, TOP_K, ds)
            out_summaries[ds_name].append(summ)
                
    with open(f'{args.res_dir}/{args.run_name}/rag_fair_summary.pkl', 'wb') as handle:
        pickle.dump(out_summaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    args = parse_args()
    main(args)