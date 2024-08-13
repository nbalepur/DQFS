""" Decides which agent speaks next and plans topics """
class Moderator:

    def __init__(self, retriever, llm):

        self.llm = llm
        self.retriever = retriever

        self.discussion_point_definition = f"The discussions point should be short, around 5 words. "
        self.discussion_point_definition += f"They should be much more specific than the query, capturing high-level themes or arguments. "
        self.discussion_point_definition += f"Avoid overly broad terms like 'Impacts' or 'Benefits'; instead, provide focused terms that are still high-level themes. "

    def answer_qa_normal(self, query, contexts):

        prompt = f"The following are documents related to the query: {query}"
        prompt += f"\n{contexts}"
        prompt += f"\n\nGive a summary that briefly answers the query in a maximum of three sentences. "
        prompt += "Each sentence in the paragraph should include a citation, in the form of a number inside square brackets, indicating the source documents from which the information was derived. "
        prompt += "Use only the documents that are relevant for answering the query. "
        prompt += f"Your final output must be a JSON dictionary with keys for \"summary\"."

        parsed_out = self.llm.generate(prompt, ["summary"])
        return parsed_out["summary"]

    def abstain_answer(self, query, contexts):

        prompt = f"The following are documents related to the query: {query}"
        prompt += f"\n{contexts}"
        prompt += f"\n\nBriefly summarize in less than three sentences why this query cannot be answered by the documents. "
        prompt += f"Also recommend up to three questions that are very related to the original question: {query}. These questions must be answerable by at least one of the documents. Each question should include a citation, in the form of a number inside square brackets, indicating the source documents from which the information was derived and the question can be answered. "
        prompt += f"Your final output must be a JSON dictionary with keys for \"summary\" explaining why the question is unanswerable, and \"questions\" which should be a list containing the generated questions with citations."

        parsed_out = self.llm.generate(prompt, ["summary", "questions"])
        return parsed_out

    def route_query_answer(self, query, top_k):

        paras, _ = self.retriever.get_doc_candidates(query, top_k)
        contexts = '\n'.join([f'Document {idx+1}: ' + ' '.join(p) for idx, p in enumerate(paras)])

        prompt = f"The following are documents related to the query: {query}"
        prompt += f"\n{contexts}"
        prompt += f"\n\nClassify whether the documents discuss conflicting or opposing perspectives in relation to the query "
        prompt += "Your output should be \"opposing\" if the documents give opposing answers to the query. Your output should be \"not opposing\" if there is only one main, non-opposing answer to the query from the documents. Your output should be \"unanswerable\" if there is no answer to the query in the documents. "
        prompt += f"Your final output must only be a JSON dictionary with keys for \"label\", denoting the label for the query's answer type contained in the documents (\"opposing\" for diverse answers, \"not opposing\" for non-diverse answers, and \"unanswerable\" when no answer is present), and \"reasoning\", denoting your reasoning which should be briefly explained in one sentence"

        parsed_out = self.llm.generate(prompt, ["label", "reasoning"])
        return parsed_out, contexts

    def plan_discussion_points_demo(self, query, num_points, contexts):

        #keys = [f'"discussion point {point_num+1}"' for point_num in range(num_points)]
        #key_text = ", ".join(keys)

        prompt = f"The following are documents related to the query: {query}"
        prompt += f"\n{contexts}"
        prompt += f"\n\nBased on the documents, propose three fine-grained discussion points that encompass almost all of the information in these documents. The discussion points should not be biased towards any side. "
        prompt += f"Along with these points, propose up to {num_points-3} other discussion points that a user may also be interested in. You can propose less than {num_points-3} other points if you believe all points have already been covered. "
        prompt += self.discussion_point_definition
        prompt += "All discussion points must be unique. "
        prompt += f"Your final output must be a JSON dictionary with the key \"important points\", containing a list of the three important discussion points, and the key \"other points\", containing a list of the other discussion points "

        parsed_out = self.llm.generate(prompt, "important points", "other points")
        return parsed_out

    def plan_discussion_points(self, query, num_points, top_k):

        paras, _ = self.retriever.get_doc_candidates(query, top_k)
        contexts = '\n'.join([f'Document {idx+1}: ' + ' '.join(p) for idx, p in enumerate(paras)])

        keys = [f'"discussion point {point_num+1}"' for point_num in range(num_points)]
        key_text = ", ".join(keys)

        prompt = f"The following are documents related to the query: {query}"
        prompt += f"\n{contexts}"
        prompt += f"\n\nBased on the documents, produce {num_points} fine-grained discussion points discussed in these documents. "
        prompt += self.discussion_point_definition
        prompt += f"Your final output must be a JSON dictionary with keys for {key_text}"

        parsed_out = self.llm.generate(prompt, keys)
        return parsed_out

    def extract_rationales(self, prompt, num_tries=0, max_tries=5):
        if num_tries == max_tries:
            return None

        try:
            parsed_out = self.llm.generate(prompt, ['relevant documents'])
            rel_docs = parsed_out['relevant documents']
            needed_rationales = [f"Document {idx} Rationale" for idx in rel_docs]
            has_all_rationales = True
            for r in needed_rationales:
                if r.lower() not in parsed_out:
                    print(r.lower(), type(parsed_out))
                    has_all_rationales = False
                    break
            if has_all_rationales:
                return parsed_out
            print("Retrying parsing!", parsed_out)
            return self.extract_rationales(prompt, num_tries + 1, max_tries)
        except Exception as e:
            print("Retrying generation!", e)
            return self.extract_rationales(prompt, num_tries + 1, max_tries)

    def extract_questions(self, prompt, num_tries=0, max_tries=5):
        
        if num_tries == max_tries:
            return None

        try:
            parsed_out = self.llm.generate(prompt, ['relevant documents'])
            rel_docs = parsed_out['relevant documents']
            needed_rationales = [f"Document {idx} Question" for idx in rel_docs]
            has_all_rationales = True
            for r in needed_rationales:
                if r.lower() not in parsed_out:
                    print(r.lower(), type(parsed_out))
                    has_all_rationales = False
                    break
            if has_all_rationales:
                return parsed_out
            print("Retrying parsing!", parsed_out)
            return self.extract_rationales(prompt, num_tries + 1, max_tries)
        except Exception as e:
            print("Retrying generation!", e)
            return self.extract_rationales(prompt, num_tries + 1, max_tries)

    def select_speakers_for_point(self, point, top_k, use_cot):

        paras, _ = self.retriever.get_doc_candidates(point, top_k)
        contexts = '\n'.join([f'Document {idx+1}: ' + ' '.join(p) for idx, p in enumerate(paras)])
        
        prompt = f"The following are documents related to the discussion point: {point}"
        prompt += f"\n{contexts}"
        prompt += f"\n\nBased on the documents, classify which documents may contain useful information and diverse perspectives related to \"{point}\". "
        prompt += f"Your final output must be a JSON dictionary with a key for \"relevant documents\", containing a list of integers corresponding to the documents relevant to the point. "

        if not use_cot:
            prompt += f"Do not use any reasoning."
            parsed_out = self.llm.generate(prompt, ['relevant documents'])
        else:
            prompt += f"Think step by step before answering, and include a rationale or reasoning for each document in the form \"Document N Rationale:\" as a key in the JSON file, where N is the number of one of the documents."
            parsed_out = self.extract_rationales(prompt)
        
        return parsed_out
    
    def select_speakers_for_point_question(self, query, point, top_k, use_cot, use_point_for_retrieval):

        paras, _ = self.retriever.get_doc_candidates(point if use_point_for_retrieval else query, top_k)
        contexts = '\n'.join([f'Document {idx+1}: ' + ' '.join(p) for idx, p in enumerate(paras)])
        
        prompt = f"The following are documents related to the discussion point: {point}"
        prompt += f"\n{contexts}"
        prompt += f"\n\nBased on the documents, classify which documents contain relevant information or diverse perspectives related to \"{point}\". "
        prompt += f"Your final output must be a JSON dictionary with a key for \"relevant documents\", containing a list of integers corresponding to the documents relevant to the point. "

        if not use_cot:
            prompt += f"Do not use any reasoning."
            parsed_out = self.llm.generate(prompt, ['relevant documents'])
        else:
            prompt += f"For each relevant document, generate a very short question that you think the document is an expert in and captures the perspective of the document related to \"{point}\". "
            prompt += f"Each question should be in the form \"Document N Question:\" as a key in the JSON file, where N is the number of one of the documents."
            parsed_out = self.extract_questions(prompt)
        
        return parsed_out