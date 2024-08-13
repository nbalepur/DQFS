import json
from openai import AzureOpenAI, OpenAI
import os
import re
import time

""" Basic LLM for all experiments """
class LLM():
    def __init__(self, model, temp, token_limit):
        self.model = model
        self.temp = temp
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv(f'{model}_ENDPOINT'),  
            api_version='2024-05-01-preview',
            api_key=os.getenv(f'{model}_API_KEY')
        )
        self.deployment_name = os.getenv(f'{model}_DEPLOYMENT_NAME')
        self.token_limit = token_limit

    def prompt_model(self, prompt, num_tries=0, max_tries=3):

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.deployment_name, 
                messages=messages,
                temperature=self.temp
            )
            return response.choices[0].message.content
        except Exception as e:
            if num_tries == max_tries - 1:
                raise e 
            return self.prompt_model(prompt, num_tries+1, max_tries)

    def generate(self, prompt, labels=[], num_tries=0, max_tries=5):
        if num_tries == max_tries:
            return None

        output = self.prompt_model(prompt)
        #print(output)
        if labels == None or len(labels) == 0:
            return output
        parse_out = self.parse_json_out(output, labels)
        
        if parse_out == None:
            print("Retrying:", output)
            parse_out = self.generate(prompt, labels, num_tries+1, max_tries)
        return parse_out

    def parse_json_out(self, output, labels):
    
        for k in labels:
            if k.lower() not in output.lower().replace('_', ' '):
                print('key doesnt exist:', k)
                return None
        try:
            # try loading the JSON directly
            parse_output = output.replace('json', '').replace('JSON', '').replace('```', '').strip()
            result = json.loads(parse_output)
            result = {k.lower().replace('_', ' ').replace(':', ''): v for k, v in result.items()}
            return result
        except Exception as e:
            try:
                print(e)
                # try extracting the JSON part (CoT)
                pattern = r'```json(.*?)```'
                matches = list(re.findall(pattern, output, re.DOTALL))
                parse_output = matches[-1].strip()
                result = json.loads(parse_output)
                result = {k.lower().replace('_', ' ').replace(':', ''): v for k, v in result.items()}
                return result
            except Exception as e:
                print(e)
                # load the string naively
                print('loading naively')
                pattern = r'"?(' + '|'.join(re.escape(label) for label in labels) + ')"?: (.+)'
                matches = re.findall(pattern, output, flags=re.IGNORECASE)
                result = {match[0].lower().replace('_', ' '): match[1] for match in matches}
                return result

        return None