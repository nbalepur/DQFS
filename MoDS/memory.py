import re

class Memory:

    """ A class that manages the discussion """
    def __init__(self, query):
        self.query = query
        self.topic_speakers = []
        self.topics = []
        self.facts = []
        self.rounds = []

    def set_topics(self, topic_data):
        topic_data = dict(sorted(topic_data.items()))
        for _, v in topic_data.items():
            self.topics.append(v)

    def update_topics(self, topics):
        self.topics = topics
        self.topic_speakers = []
        self.facts = []
        self.rounds = []

    def get_topics(self):
        return self.topics
        
    def initialize_topic(self):
        self.facts.append({'facts': [], 'labels': [], 'doc_nums': []})

    def add_selected_speaker_info(self, speaker_info):
        self.topic_speakers.append(speaker_info)

    def get_speaker_rationale_pairs(self):
        speaker_info = self.topic_speakers[-1]
        rel_docs = speaker_info['relevant documents']
        return [(doc_num - 1, speaker_info.get(f'document {doc_num} rationale', None)) for doc_num in rel_docs]

    def get_speaker_question_pairs(self):
        speaker_info = self.topic_speakers[-1]
        rel_docs = speaker_info['relevant documents']
        return [(doc_num - 1, speaker_info.get(f'document {doc_num} question', None)) for doc_num in rel_docs]

    def add_facts(self, fact_info, speaker_num):
        yes_facts, no_facts = fact_info['yes facts'], fact_info['no facts']
        num_facts = len(yes_facts) + len(no_facts)
        
        self.facts[-1]['facts'].extend(yes_facts + no_facts)
        self.facts[-1]['labels'].extend(['yes' for _ in yes_facts] + ['no' for _ in no_facts])
        self.facts[-1]['doc_nums'].extend([speaker_num for _ in range(num_facts)])

    def add_facts_citation(self, fact_info):
        yes_facts, no_facts = fact_info['yes facts'], fact_info['no facts']
        num_facts = len(yes_facts) + len(no_facts)

        cites = []
        clean_facts = []
        pattern = r'\[\D*(\d+)\]'
        for fact in yes_facts + no_facts:
            cites.append(re.findall(pattern, fact)[0])
            clean_facts.append(re.sub(pattern, '', fact).replace(' .', '.'))

        self.facts[-1]['facts'].extend(clean_facts)
        self.facts[-1]['labels'].extend(['yes' for _ in yes_facts] + ['no' for _ in no_facts])
        self.facts[-1]['doc_nums'].extend(cites)

    def print(self):

        out = f'Query: {self.query}'
        for idx in range(len(self.topics)):
            topic = self.topics[idx]
            facts = self.facts[idx]
        
            curr_facts = self.facts[idx]['facts']
            curr_docs = self.facts[idx]['doc_nums']
            curr_labels = self.facts[idx]['labels']
            yes_facts = ['- Yes: ' + curr_facts[idx] + f' [{doc_num+1}]' for idx, doc_num in enumerate(curr_docs) if curr_labels[idx] == 'yes']
            no_facts = ['- No: ' + curr_facts[idx] + f' [{doc_num+1}]' for idx, doc_num in enumerate(curr_docs) if curr_labels[idx] == 'no']
            fact_text = "\n".join(yes_facts + no_facts)
            out += f'\n\nTopic: {topic}'
            out += f'\nFacts:\n{fact_text}'

        return out