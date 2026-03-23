import json
from collections import Counter
from pydoc import doc
from openai import OpenAI
import requests
import re
import spacy
nlp = spacy.load("en_core_web_sm")


# TODO
def refine_declarations(declarations, llm, args):
    template_path = "/home/chenzhb/Workspaces/verl/mcts_utils/prompts/refine_declaration.txt"
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    prompt = template + "\n" + declarations
    response = llm.generate(prompt, args=args)
    return response
# TODO
def knowledge_injection(declarations, llm, args):
    template_path = "/home/chenzhb/Workspaces/verl/mcts_utils/prompts/knowledge_injection.txt"
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    prompt = template + "\n" + declarations
    response = llm.generate(prompt, args=args)
    return response

def check_declarations(declarations):
    z3_code = "from z3 import *\n\n" + declarations
    try:
        exec(z3_code)
        return True, None
    except Exception as e:
        print(f"Error occurred while checking declarations: {e}")
        return False, str(e)

def split_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    # print(sentences)

def extract_python_code_block(text):
    pattern = r"```python\s+([\s\S]*?)```"
    # 搜索匹配项
    match = re.search(pattern, text)
    if match:
        # group(1) 返回第一个括号内匹配的内容，即我们需要的部分
        return match.group(1).strip()
    else:
        return False
    

def extract_plain_text_block(text):
    pattern = r"```plain_text\s+([\s\S]*?)```"
    # 搜索匹配项
    match = re.search(pattern, text)
    if match:
        # group(1) 返回第一个括号内匹配的内容，即我们需要的部分
        return match.group(1).strip()
    else:
        return False
    
def rephrase(response, llm, args):
    template_path = "/home/chenzhb/Workspaces/verl/mcts_utils/prompts/rephrase.txt"
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    prompt = template + "\n" +response
    response = llm.generate(prompt, args=args)
    return response

def extract_declarations(question, response, llm, args):
    template_path = "/home/chenzhb/Workspaces/verl/mcts_utils/prompts/extract_declaration.txt"
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    prompt = template  + question   + "\n\n ### Declaration"
    response = llm.generate(prompt, args=args)
    return response

def split_steps(response):
    # 使用正则表达式提取 steps
    steps = re.findall(r'<step>(.*?)</step>', response, re.DOTALL)
    return steps

def translate_to_fol(response, declarations, llm, args):
    template_path = "/home/chenzhb/Workspaces/verl/mcts_utils/prompts/translate2fol_new.txt"
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    # 先分 step 
    previous_exp = [] # 存储之前的步骤的 FOL 表示，防止变量的重复和冲突
    step_list = split_steps(response)
    for i, step in enumerate(step_list):
        premises = re.findall(r'<premise>(.*?)</premise>', step)
        conclusion = re.search(r'<conclusion>(.*?)</conclusion>', step)
        sentence = "The administrative service area is southwest of the cultural area."
        prompt = template.format(sentence=sentence, previous_exp="\n".join(previous_exp), declarations=declarations)
        import pdb;pdb.set_trace()
        response = llm.generate(prompt, args=args)
        
    return extract_plain_text_block(response.choices[0].message.content.strip())

class LLM:
    def __init__(self, base_url="http://localhost:4869/v1", api_key="EMPTY"):
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def generate(self, data, model='qwen2.5-3b', args=None):
        chat_response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": data},
            ],
            max_tokens=args['max_tokens'],
            temperature=args['temperature'],
            top_p=args['top_p'],
        )
        return chat_response.choices[0].message.content
    
    def health_check(self):
        health_url = self.base_url.replace("/v1", "") + "/health"
        response = requests.get(health_url)
        if response.status_code == 200:
            print("LLM API is available.")
        else:
            print(f"LLM API health check failed with status code: {response.status_code}")




if __name__ == "__main__":
    question_file = "/home/chenzhb/Workspaces/verl/mcts_utils/data/logiqa.jsonl"
    response_file = "/home/chenzhb/Workspaces/verl/mcts_utils/data/qwen2.5-3b_mcts_responses.jsonl"
    base_url = "http://localhost:4869/v1"
    api_key = "EMPTY"
    args = {
        "max_tokens": 4096,
        "temperature": 0.1,
        "top_p": 0.8,
    }
    # Initialize the LLM and test the API call
    llm = LLM(base_url=base_url, api_key=api_key)
    llm.health_check()
    f_w = open("/home/chenzhb/Workspaces/verl/mcts_utils/data/translated_responses.jsonl", "w", encoding="utf-8")
    with open(response_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            question = data["question"]
            response = data["response"]
            
            # Step 1: Rephrase the response
            rephrase_response = rephrase(response, llm, args)
            response_sentences = split_sentences(question+rephrase_response)

            response_steps = split_steps(response)

            # Step 2: Extract declarations
            # res = False
            # count = 0
            # while not res:
            #     count += 1
            #     print(count)
            #     declarations = extract_declarations(question, rephrase_response, llm, args)
            #     declarations_code = extract_python_code_block(declarations)
            #     if not declarations_code:
            #         continue
            #     res, e  = check_declarations(declarations_code)

            # declarations = knowledge_injection(declarations, llm, args)# common knowledge injection
            declarations = """Communities, (cultural_area, leisure_area, commercial_area, administrative_service_area) = EnumSort('Communities',['cultural_area', 'leisure_area', 'commercial_area', 'administrative_service_area'])\n\nDirection, (southwest, southeast, northwest, northeast, south, east, north, west) = EnumSort('Direction',['southwest','southeast','northwest','northeast','south','east','north','west'])\n\nLocation = Function('location', Communities, Communities, Direction)"""
            
            
            # Step 3: Translate context and response into FOL
            fol_exp = translate_to_fol(response, declarations,llm, args)

            # Step 4: convert to Z3 code and execute
            z3_code = "from z3 import *\n\ns = Solver()\n\n" + declarations + "\n\n" + fol_exp + "\n\nif s.check() == sat:\n    print(1)\nelse:\n    print(0)"
