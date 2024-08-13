import tiktoken

""" Agent responsible for a single document """
class Speaker:

    def __init__(self, retriever, llm, docs, doc_num):
        self.llm = llm
        self.retriever = retriever
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        self.document = '\n'.join(self.prune_sources(docs, encoding))
        self.doc_num = doc_num

        self.discussion_point_definition = f"The discussion point should be short, around 5 words. "
        self.discussion_point_definition += f"It should be much more specific than the query, capturing a high-level theme or argument. "
        self.discussion_point_definition += f"Avoid overly broad terms like 'Impacts' or 'Benefits'; instead, provide focused terms that are still high-level themes. "

    def prune_sources(self, docs, encoding):
        pruned_docs = []
        total_tokens = 0
        for idx, doc in enumerate(docs):
            if total_tokens > self.llm.token_limit:
                print('pruned!')
                return pruned_docs, pruned_idxs
            pruned_docs.append(doc)
            total_tokens += len(encoding.encode(doc))
        return pruned_docs

    def speak(self, query, discussion_point, rationale):

        prompt = f"The following is a document related to the query: {query}"
        prompt += f"\nDocument:\n{self.document}"
        prompt += f"\n\nUsing the document, generate two lists of factual sentences, a \"yes\" list and \"no\" list, related to the query. "
        prompt += f"The \"yes\" list should only contain facts for why the answer to the query under the discussion point is yes, and the \"no\" list should only contain facts for why the answer to the query under the discussion point is no. "
        prompt += f"The lists for yes facts or no facts should be empty if no fact for that answer exists. "
        prompt += f"Only produce facts that are directly related to discussion point of \"{discussion_point}\". "
        prompt += f"Your output must be a JSON dictionary with keys \"discussion point\" for the discussion point, \"yes facts\" for the list of yes facts, and \"no facts\" for the list of no facts"

        if rationale != None:
            prompt += f"\n\nAn external moderator gave the following rationale to justify why you are appropriate to produce facts for this discussion point: {rationale} "
        
        parsed_out = self.llm.generate(prompt, ['Discussion point', 'Yes facts', 'No facts'])
        return parsed_out

    def speak_rag(self, query, top_k, discussion_point, search_query):

        #print(query, top_k, discussion_point, search_query)

        context = ' '.join(self.retriever.retrieve(search_query, top_k, self.doc_num))

        prompt = f"The following is a document related to the query: {query}"
        prompt += f"\nDocument:\n{context}"
        prompt += f"\n\nUsing the document, generate two lists of factual sentences, a \"yes\" list and \"no\" list, related to the query. "
        prompt += f"The \"yes\" list should only contain facts for why the answer to the query under the discussion point is yes, and the \"no\" list should only contain facts for why the answer to the query under the discussion point is no. "
        prompt += f"The lists for yes facts or no facts should be empty if no fact for that answer exists. "
        search_query_text = '' if discussion_point == search_query else f"and the subquestion of \"{search_query}\""
        prompt += f"Only produce facts that are directly related to discussion point of \"{discussion_point}\" {search_query_text}. "
        prompt += f"Your output must be a JSON dictionary with keys \"discussion point\" for the discussion point, \"yes facts\" for the list of yes facts, and \"no facts\" for the list of no facts"

        #print(prompt)
        
        parsed_out = self.llm.generate(prompt, ['Discussion point', 'Yes facts', 'No facts'])
        return parsed_out

    def speak_retrieve_all(self, query, retr_docs, retr_idxs, discussion_point):

        context = '\n'.join([f'Document {retr_idxs[idx] + 1}: {d}' for idx, d in enumerate(retr_docs)])

        prompt = f"The following is a document related to the query: {query}"
        prompt += f"\n{context}"
        prompt += f"\n\nUsing the document, generate two lists of factual sentences, a \"yes\" list and \"no\" list, related to the query. "
        prompt += f"The \"yes\" list should only contain facts for why the answer to the query under the discussion point is yes, and the \"no\" list should only contain facts for why the answer to the query under the discussion point is no. "
        prompt += f"The lists for yes facts or no facts should be empty if no fact for that answer exists. "
        prompt += f"For each fact, cite the document from which the information was derived, in the form of a number surrounded by square brackets. "
        prompt += f"Only produce facts that are directly related to discussion point of \"{discussion_point}\". "
        prompt += f"Your output must be a JSON dictionary with keys \"discussion point\" for the discussion point, \"yes facts\" for the list of yes facts, and \"no facts\" for the list of no facts"
        
        parsed_out = self.llm.generate(prompt, ['Discussion point', 'Yes facts', 'No facts'])
        return parsed_out