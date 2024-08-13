import tiktoken
import random

class Summarizer:

    def __init__(self, llm, num_points):
        self.llm = llm
        self.encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

        self.discussion_point_definition = f"The discussion point should be short, around 5 words. "
        self.discussion_point_definition += f"It should be much more specific than the query, capturing a high-level theme or argument. "
        self.discussion_point_definition += f"Avoid overly broad terms like 'Impacts' or 'Benefits'; instead, provide focused terms that are still high-level themes. "

        json_keys = [[f'discussion point {idx + 1}', f'summary {idx + 1}'] for idx in range(num_points)]
        self.json_keys = [x for xs in json_keys for x in xs]
        self.json_keys_quoted = ", ".join(['"' + x + '"' for x in self.json_keys])
        self.num_points = num_points

    def update_init(self, num_points):
        json_keys = [[f'discussion point {idx + 1}', f'summary {idx + 1}'] for idx in range(num_points)]
        self.json_keys = [x for xs in json_keys for x in xs]
        self.json_keys_quoted = ", ".join(['"' + x + '"' for x in self.json_keys])
        self.num_points = num_points
    
    def prune_sources(self, docs, doc_idxs):
        pruned_docs = []
        pruned_idxs = []
        total_tokens = 0
        for idx, doc in enumerate(docs):
            if total_tokens > self.llm.token_limit:
                print('pruned!')
                return pruned_docs, pruned_idxs
            pruned_docs.append(doc)
            pruned_idxs.append(doc_idxs[idx])
            total_tokens += len(self.encoding.encode(doc))
        return pruned_docs, pruned_idxs

    def prune_sources_collection(self, docs, doc_idxs):
        pruned_docs = []
        pruned_idxs = []
        total_tokens = 0
        for idx, docs_ in enumerate(docs):
            pruned_docs.append([])
            pruned_idxs.append(None)
            for doc in docs_:
                if total_tokens > self.llm.token_limit:
                    print('pruned!')
                    return pruned_docs, pruned_idxs
                pruned_docs[-1].append(doc)
                pruned_idxs[-1] = idx

        if pruned_idxs[-1] == None:
            pruned_docs.pop(-1)
            pruned_idxs.pop(-1)

        return pruned_docs, pruned_idxs
    
    def summarize_outline_ind(self, outline):

        final_out = dict()
        
        for idx in range(self.num_points):
            
            prompt = f"The following is an outline for the query {outline.query} and a single discussion point. Under the discussion point, there is a list of documents and a subquestion explaining the document's expertise on the discussion point. Under each document and question, there will be a bullet point list of facts preceded by either \"Yes Fact:\" and \"No Fact:\", denoting whether the fact gives evidence for why the answer to the query is yes or no:\n"
            prompt += self.print_outline_ind(outline, idx)
            prompt += f"\n\nSynthesize the Yes Facts and No Facts from the outline and produce a brief summary that answers the query under the same discussion point. The summary should be one brief, three-sentence paragraph. "
            prompt += "Each sentence in the paragraph should include a citation, in the form of a number inside square brackets, indicating the source documents from which the information was derived. "
            prompt += "Use as many documents as possible. "
            prompt += "Your final output must be a JSON dictionary with keys for \"discussion point\" and \"summary\". "
    
            parsed_out = self.llm.generate(prompt, ["Discussion point", "Summary"])
            
            final_out[f'discussion point {idx+1}'] = parsed_out['discussion point']
            final_out[f'summary {idx+1}'] = parsed_out['summary']
            
        return final_out

    def summarize_outline_ind_no_q(self, outline):

        final_out = dict()
        
        for idx in range(self.num_points):
            
            prompt = f"The following is an outline for the query {outline.query} and a single discussion point. Under the discussion point, there is a list of documents. Under each document, there will be a bullet point list of facts preceded by either \"Yes Fact:\" and \"No Fact:\", denoting whether the fact gives evidence for why the answer to the query is yes or no:\n"
            prompt += self.print_outline_ind_no_q(outline, idx)
            prompt += f"\n\nSynthesize the Yes Facts and No Facts from the outline and produce a brief summary that answers the query under the same discussion point. The summary should be one brief, three-sentence paragraph. "
            prompt += "Each sentence in the paragraph should include a citation, in the form of a number inside square brackets, indicating the source documents from which the information was derived. "
            prompt += "Use as many documents as possible. "
            prompt += "Your final output must be a JSON dictionary with keys for \"discussion point\" and \"summary\". "
    
            parsed_out = self.llm.generate(prompt, ["Discussion point", "Summary"])
            
            final_out[f'discussion point {idx+1}'] = parsed_out['discussion point']
            final_out[f'summary {idx+1}'] = parsed_out['summary']
            
        return final_out
    
    def summarize_outline_full(self, outline):
        
        prompt = f"The following is an outline for the query {outline.query}, broken down into {self.num_points} fine-grained discussion points. Under the discussion point, there is a list of documents and a subquestion explaining the document's expertise on the discussion point. Under each document and question, there will be a bullet point list of facts preceded by either \"Yes Fact:\" and \"No Fact:\", denoting whether the fact gives evidence for why the answer to the query is yes or no:\n"
        prompt += self.print_outline_full(outline)
        prompt += f"\n\nSynthesize the Yes Facts and No Facts from the outline and produce a brief summary that answers the query under the same {self.num_points} discussion points. The summary should be one brief, three-sentence paragraph per point. "
        prompt += "Each sentence in the paragraph should include a citation, in the form of a number inside square brackets, indicating the source documents from which the information was derived. "
        prompt += "Use as many documents as possible. "
        prompt += f"Your final output must be a JSON dictionary with keys for {self.json_keys_quoted}. "

        parsed_out = self.llm.generate(prompt, self.json_keys)
        return parsed_out

    def summarize_outline_ind_nomod(self, outline):

        final_out = dict()
        
        for idx in range(self.num_points):
            
            prompt = f"The following is an outline for the query {outline.query} and a single discussion point. Under the discussion point, there is a list of documents and a subquestion explaining the document's expertise on the discussion point. Under each document and question, there will be a bullet point list of facts preceded by either \"Yes Fact:\" and \"No Fact:\", denoting whether the fact gives evidence for why the answer to the query is yes or no:\n"
            prompt += self.print_outline_ind_nomod(outline, idx)
            prompt += f"\n\nSynthesize the Yes Facts and No Facts from the outline and produce a brief summary that answers the query under the same discussion point. The summary should be one brief, three-sentence paragraph. "
            prompt += "Each sentence in the paragraph should include a citation, in the form of a number inside square brackets, indicating the source documents from which the information was derived. "
            prompt += "Use as many documents as possible. "
            prompt += "Your final output must be a JSON dictionary with keys for \"discussion point\" and \"summary\". "
    
            parsed_out = self.llm.generate(prompt, ["Discussion point", "Summary"])
            
            final_out[f'discussion point {idx+1}'] = parsed_out['discussion point']
            final_out[f'summary {idx+1}'] = parsed_out['summary']
            
        return final_out
    
    def summarize_outline_full_nomod(self, outline):
        
        prompt = f"The following is an outline for the query {outline.query}, broken down into {self.num_points} fine-grained discussion points. Under the discussion point, there is a list of documents and a subquestion explaining the document's expertise on the discussion point. Under each document and question, there will be a bullet point list of facts preceded by either \"Yes Fact:\" and \"No Fact:\", denoting whether the fact gives evidence for why the answer to the query is yes or no:\n"
        prompt += self.print_outline_full_nomod(outline)
        prompt += f"\n\nSynthesize the Yes Facts and No Facts from the outline and produce a brief summary that answers the query under the same {self.num_points} discussion points. The summary should be one brief, three-sentence paragraph per point. "
        prompt += "Each sentence in the paragraph should include a citation, in the form of a number inside square brackets, indicating the source documents from which the information was derived. "
        prompt += "Use as many documents as possible. "
        prompt += f"Your final output must be a JSON dictionary with keys for {self.json_keys_quoted}. "

        parsed_out = self.llm.generate(prompt, self.json_keys)
        return parsed_out

    def summarize_docs(self, query, docs, doc_idxs):

        pruned_docs, pruned_idxs = self.prune_sources_collection(docs, doc_idxs)

        doc_texts = [f'Document {pruned_idxs[idx] + 1}: {" ".join(d)}' for idx, d in enumerate(pruned_docs)]
        doc_text = '\n'.join(doc_texts)

        prompt = f"The following are documents for the query: {query}\n"
        prompt += doc_text
        prompt += f"\n\nSynthesize the facts from the document and produce a brief summary that answers the query under {self.num_points} fine-grained discussion points. The summary should have one brief, three-sentence paragraph per point. "
        prompt += "Each sentence in the paragraph should include a citation, in the form of a number inside square brackets, indicating the source documents from which the information was derived. "
        prompt += "Use as many documents as possible. "
        prompt += self.discussion_point_definition
        prompt += f"Your final output must be a JSON dictionary with keys for {self.json_keys_quoted}. "

        parsed_out = self.llm.generate(prompt, self.json_keys)
        return parsed_out

    def summarize_docs_flat(self, query, docs, doc_idxs):

        pruned_docs, pruned_idxs = self.prune_sources(docs, doc_idxs)

        doc_texts = [f'Document {pruned_idxs[idx] + 1}: {d}' for idx, d in enumerate(pruned_docs)]
        doc_text = '\n'.join(doc_texts)

        prompt = f"The following are documents for the query: {query}\n"
        prompt += doc_text
        prompt += f"\n\nSynthesize the facts from the document and produce a summary that answers the query under {self.num_points} fine-grained discussion points. The summary should be one brief, three-sentence paragraph per point. "
        prompt += "Each sentence in the paragraph should include a citation, in the form of a number inside square brackets, indicating the source documents from which the information was derived. "
        prompt += "Use as many documents as possible. "
        prompt += self.discussion_point_definition
        prompt += f"Your final output must be a JSON dictionary with keys for {self.json_keys_quoted}. "

        parsed_out = self.llm.generate(prompt, self.json_keys)
        return parsed_out

    def summarize_docs_flat_point(self, query, point, docs, doc_idxs):

        pruned_docs, pruned_idxs = self.prune_sources(docs, doc_idxs)

        doc_texts = [f'Document {pruned_idxs[idx] + 1}: {d}' for idx, d in enumerate(pruned_docs)]
        doc_text = '\n'.join(doc_texts)

        prompt = f"The following are documents for the query: {query} and discussion point: {point}\n"
        prompt += doc_text
        prompt += f"\n\nSynthesize the facts from the document and produce a summary that answers the query related to the discussion point. The summary should be one brief, three-sentence paragraph. "
        prompt += "Each sentence in the paragraph should include a citation, in the form of a number inside square brackets, indicating the source documents from which the information was derived. "
        prompt += "Use as many documents as possible. "
        prompt += self.discussion_point_definition
        prompt += f"Your final output must be a JSON dictionary with keys for \"summary\". "

        parsed_out = self.llm.generate(prompt, "summary")
        return parsed_out
        
    def summarize_docs_single(self, query, docs, doc_idxs):

        pruned_docs, pruned_idxs = self.prune_sources(docs, doc_idxs)

        doc_texts = [f'Document {pruned_idxs[idx] + 1}: {d}' for idx, d in enumerate(pruned_docs)]
        doc_text = '\n'.join(doc_texts)

        prompt = f"The following are documents for the query: {query}\n"
        prompt += doc_text
        prompt += "\n\nSynthesize the facts from the document and produce a summary that answers the query under a single fine-grained discussion point. The summary should be one brief, three-sentence paragraph per point. "
        prompt += "Each sentence in the paragraph should include a citation, in the form of a number inside square brackets, indicating the source documents from which the information was derived. "
        prompt += "Use as many documents as possible. "
        prompt += self.discussion_point_definition
        prompt += "Your final output must be a JSON dictionary with keys for \"discussion point\" and \"summary\". "

        parsed_out = self.llm.generate(prompt, ['Discussion point', 'Summary'])
        return parsed_out

    def summarize_one_doc(self, query, doc):

        prompt = f"The following is a document for the query: {query}\n"
        prompt += ' '.join(doc)
        prompt += "\n\nSynthesize the facts from the document and produce a summary that answers the query."
        prompt += "Your final output must be a JSON dictionary with a key for \"summary\". "

        parsed_out = self.llm.generate(prompt, ['Summary'])
        return parsed_out

    def summarize_one_doc_point(self, query, doc, point):

        prompt = f"The following is a document for the query: {query}\n"
        prompt += ' '.join(doc)
        prompt += f"\n\nSynthesize the facts from the document and produce a summary that answers the query and is related to the point {point}. "
        prompt += f"If there is no relevant information respond with \"N/A\" as your summary. "
        prompt += "Your final output must be a JSON dictionary with a key for \"summary\", and the value can be \"N/A\" if there is no relevant information. "

        parsed_out = self.llm.generate(prompt, ['Summary'])
        return parsed_out

    def refine_summary_ind(self, query, topic, summary):

        prompt = f"The following is a summary for the query {query} under the discussion point of {topic}:"
        prompt += summary
        prompt += "\n\nRefine the summary so it is only one brief, three-sentence paragraph. "
        prompt += "Each sentence in the paragraph should include a citation, in the form of a number inside square brackets, indicating the source documents from which the information was derived. "
        prompt += "Use as many documents as possible. "
        prompt += "Your final output must be a JSON dictionary with keys for \"discussion point\" and \"summary\". "

        parsed_out = self.llm.generate(prompt, ['Discussion point', 'Summary'])
        return parsed_out

    def refine_summary_full(self, query, summary):

        prompt = f"The following is a summary for the query: {query}, broken down into {self.num_points} fine-grained discussion points. Under each discussion point is a paragraph related to the query and point:"
        prompt += summary
        prompt += "\n\nRefine the summary so that each paragraph is only three sentences, using the same discussion points. "
        prompt += "Each sentence in the paragraph should include a citation, in the form of a number inside square brackets, indicating the source documents from which the information was derived. "
        prompt += "Use as many documents as possible. "
        prompt += f"Your final output must be a JSON dictionary with keys for {self.json_keys_quoted}. "

        parsed_out = self.llm.generate(prompt, self.json_keys)
        return parsed_out

    def parse_outline_nostance(self, outline):

        sections = []
        out = f'Query: {outline.query}'
        sections.append(out)
        
        for idx in range(self.num_points):
            topic = outline.topics[idx]
            facts = outline.facts[idx]
            select_info = outline.topic_speakers[idx]
        
            curr_facts = outline.facts[idx]['facts']
            curr_docs = outline.facts[idx]['doc_nums']
            curr_labels = outline.facts[idx]['labels']

            out = f'# Discussion Point {idx+1}: {topic}'

            for doc_num in set(curr_docs):
                out += f'\n## Document {doc_num + 1}: {select_info["document " + str(doc_num + 1) + " question"]}'
                yes_facts = [f'- Fact: ' + curr_facts[idx] for idx in range(len(curr_docs)) if curr_labels[idx] == 'yes' and curr_docs[idx] == doc_num]
                no_facts = [f'- Fact: ' + curr_facts[idx] for idx in range(len(curr_docs)) if curr_labels[idx] == 'no' and curr_docs[idx] == doc_num]
                all_facts = yes_facts + no_facts
                random.shuffle(all_facts)
                fact_text = "\n".join(all_facts)
                out += f'\n{fact_text}'
            sections.append(out)

        return sections

    def parse_outline(self, outline):

        sections = []
        out = f'Query: {outline.query}'
        sections.append(out)
        
        for idx in range(self.num_points):
            topic = outline.topics[idx]
            facts = outline.facts[idx]
            select_info = outline.topic_speakers[idx]
        
            curr_facts = outline.facts[idx]['facts']
            curr_docs = outline.facts[idx]['doc_nums']
            curr_labels = outline.facts[idx]['labels']

            out = f'# Discussion Point {idx+1}: {topic}'

            for doc_num in set(curr_docs):
                out += f'\n## Document {doc_num + 1}: {select_info["document " + str(doc_num + 1) + " question"]}'
                yes_facts = [f'- Yes Fact: ' + curr_facts[idx] for idx in range(len(curr_docs)) if curr_labels[idx] == 'yes' and curr_docs[idx] == doc_num]
                no_facts = [f'- No Fact: ' + curr_facts[idx] for idx in range(len(curr_docs)) if curr_labels[idx] == 'no' and curr_docs[idx] == doc_num]
                fact_text = "\n".join(yes_facts + no_facts)
                out += f'\n{fact_text}'
            sections.append(out)

        return sections

    def parse_outline_no_q(self, outline):

        sections = []
        out = f'Query: {outline.query}'
        sections.append(out)
        
        for idx in range(self.num_points):
            topic = outline.topics[idx]
            facts = outline.facts[idx]
            select_info = outline.topic_speakers[idx]
        
            curr_facts = outline.facts[idx]['facts']
            curr_docs = outline.facts[idx]['doc_nums']
            curr_labels = outline.facts[idx]['labels']

            out = f'# Discussion Point {idx+1}: {topic}'

            for doc_num in set(curr_docs):
                out += f'\n## Document {doc_num + 1}:'
                yes_facts = [f'- Yes Fact: ' + curr_facts[idx] for idx in range(len(curr_docs)) if curr_labels[idx] == 'yes' and curr_docs[idx] == doc_num]
                no_facts = [f'- No Fact: ' + curr_facts[idx] for idx in range(len(curr_docs)) if curr_labels[idx] == 'no' and curr_docs[idx] == doc_num]
                fact_text = "\n".join(yes_facts + no_facts)
                out += f'\n{fact_text}'
            sections.append(out)

        return sections

    def parse_outline_nomod(self, outline):

        sections = []
        out = f'Query: {outline.query}'
        sections.append(out)
        
        for idx in range(self.num_points):
            topic = outline.topics[idx]
            facts = outline.facts[idx]
        
            curr_facts = outline.facts[idx]['facts']
            curr_docs = outline.facts[idx]['doc_nums']
            curr_labels = outline.facts[idx]['labels']

            out = f'# Discussion Point {idx+1}: {topic}'

            for doc_num in set(curr_docs):
                out += f'\n## Document {doc_num + 1}:'
                yes_facts = [f'- Yes Fact: ' + curr_facts[idx] for idx in range(len(curr_docs)) if curr_labels[idx] == 'yes' and curr_docs[idx] == doc_num]
                no_facts = [f'- No Fact: ' + curr_facts[idx] for idx in range(len(curr_docs)) if curr_labels[idx] == 'no' and curr_docs[idx] == doc_num]
                fact_text = "\n".join(yes_facts + no_facts)
                out += f'\n{fact_text}'
            sections.append(out)

        return sections

    def parse_outline_markdown(self, outline):

        if outline == None:
            return ""

        sections = []
        out = f'# Outline: {outline.query}'
        sections.append(out)
        
        for idx in range(self.num_points):
            
            out = ''

            if idx < len(outline.topics):
                topic = outline.topics[idx]
                if self.see_points:
                    out += f'## Discussion Point {idx+1}: {topic}'

            if idx < len(outline.topic_speakers) and idx < len(outline.facts):
                facts = outline.facts[idx]
                select_info = outline.topic_speakers[idx]
            
                curr_facts = outline.facts[idx]['facts']
                curr_docs = outline.facts[idx]['doc_nums']
                curr_labels = outline.facts[idx]['labels']

                for doc_num in set(curr_docs):
                    if self.see_doc_num:
                        out += f'\n#### Document {doc_num + 1}:'

                    if self.see_questions:
                        out += ('' if self.see_doc_num else '\n####') + f' {select_info["document " + str(doc_num + 1) + " question"]}'

                    yes_facts = [f'- ' + (' <b><span style="color:green">Support</span>:</b> ') + curr_facts[idx] for idx in range(len(curr_docs)) if curr_labels[idx] == 'yes' and curr_docs[idx] == doc_num]
                    no_facts = [f'- ' + (' <b><span style="color:red">Refute</span>:</b> ') + curr_facts[idx] for idx in range(len(curr_docs)) if curr_labels[idx] == 'no' and curr_docs[idx] == doc_num]

                    if self.see_support:
                        yes_fact_text = '\n'.join(yes_facts)
                        out += f'\n{yes_fact_text}'

                    if self.see_refute:
                        no_fact_text = '\n'.join(no_facts)
                        out += f'\n{no_fact_text}'

                    if (self.see_doc_num or self.see_questions) and (int(self.see_refute) * len(no_facts) + int(self.see_support) * len(yes_facts) == 0):
                        out += f'\n\n- Desired facts not found'

            sections.append(out)

        return sections

    def set_visible_info(self, see_points=True, see_support=True, see_refute=True, see_doc_num=True, see_questions=True):
        self.see_points = see_points
        self.see_refute = see_refute
        self.see_support = see_support
        self.see_doc_num = see_doc_num
        self.see_questions = see_questions

    def print_outline_full_nomod(self, outline):
        sections = self.parse_outline_nomod(outline)
        return '\n\n'.join(sections)

    def print_outline_ind_nomod(self, outline, idx):
        sections = self.parse_outline_nomod(outline)
        return '\n\n'.join([sections[0]] + [sections[idx + 1]])

    def print_outline_full(self, outline):
        sections = self.parse_outline(outline)
        return '\n\n'.join(sections)

    def print_outline_full_markdown(self, outline):
        sections = self.parse_outline_markdown(outline)
        return '\n\n'.join(sections)

    def print_outline_ind(self, outline, idx):
        sections = self.parse_outline(outline)
        return '\n\n'.join([sections[0]] + [sections[idx + 1]])

    def print_outline_full_nostance(self, outline):
        sections = self.parse_outline_nostance(outline)
        return '\n\n'.join(sections)

    def print_outline_ind_nostance(self, outline, idx):
        sections = self.parse_outline_nostance(outline)
        return '\n\n'.join([sections[0]] + [sections[idx + 1]])

    def print_outline_ind_no_q(self, outline, idx):
        sections = self.parse_outline_no_q(outline)
        return '\n\n'.join([sections[0]] + [sections[idx + 1]])