from tools import *

from openai import OpenAI
import requests

import re
import subprocess
import sys
from string import Template
from pydantic import BaseModel
from typing import Dict, List, Set, Any

class OutputSchema(BaseModel):
    data: Dict[str, List[Any]]

class LLM:
    def __init__(self, base_url="http://localhost:4869/v1", api_key="EMPTY"):
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.health_check()

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

    def constrain_generate(self, data, model='qwen2.5-3b', format=None, args=None):
        chat_response = self.client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "user", "content": data},
            ],
            response_format=format,
            max_tokens=args['max_tokens'],
            temperature=args['temperature'],
            top_p=args['top_p'],
        )
        return chat_response.choices[0].message.parsed
    
    def health_check(self):
        health_url = self.base_url.replace("/v1", "") + "/health"
        response = requests.get(health_url)
        if response.status_code == 200:
            print("LLM API is available.")
        else:
            print(f"LLM API health check failed with status code: {response.status_code}")


def rephrase(llm, args, context, question, options):
    with open("/home/chenzhb/Workspaces/verl/mcts_utils/prompts/rephrase.txt", "r", encoding='utf-8') as f:
        template = f.read()
    prompt = Template(template).safe_substitute(context=context, question=question, options=options)
    rephrase_res = llm.generate(data=prompt, args=args)
    return rephrase_res

def object_extract(llm, args, context, question, options):
    with open("/home/chenzhb/Workspaces/verl/mcts_utils/prompts/object_exctract.txt", "r", encoding='utf-8') as f:
        template = f.read()
    prompt = Template(template).safe_substitute(context=context, question=question, options=options)
    obj_ext_res = llm.constrain_generate(data=prompt, format=OutputSchema, args=args)
    return obj_ext_res.data

def predicate_extract(llm, args, context, question, options, obj_list):
    with open("/home/chenzhb/Workspaces/verl/mcts_utils/prompts/predicate_extraction.txt", "r", encoding='utf-8') as f:
        template = f.read()
    prompt = Template(template).safe_substitute(context=context, question=question, options=options, obj_list=obj_list)
    pred_ext_res = llm.constrain_generate(data=prompt, format=OutputSchema, args=args)
    return pred_ext_res.data

import string

def generate_z3_declarations(entities):
    code_lines = []
    
    code_lines.append("# Z3 Type Declaration")
    for entity_type in entities.keys():
        line = f"{entity_type} = DeclareSort('{entity_type}')"
        code_lines.append(line)
    
    code_lines.append("\n# Constants Definition")
    for entity_type, names in entities.items():
        for name in names:
            # 处理空格，将实体名中的空格替换为下划线
            formatted_name = name.replace(" ", "_")
            line = f"{formatted_name} = Const('{formatted_name}', {entity_type})"
            code_lines.append(line)
            
    code_lines.append("\n# Variable Declarations")
    # string.ascii_lowercase 提供 'abcdefghijklmnopqrstuvwxyz'
    alphabet = string.ascii_lowercase
    
    for i, entity_type in enumerate(entities.keys()):
        if i < len(alphabet):
            var_name = alphabet[i]
            line = f"{var_name} = Const('{var_name}', {entity_type})"
            code_lines.append(line)
        else:
            # 如果类型数量超过26个，可以在此处扩展逻辑（如 aa, ab...）
            break
            
    return "\n".join(code_lines)

def generate_z3_functions(predicates):
    code_lines = []
    code_lines.append("# Z3 Function/Predicate Declaration")
    
    for func_name, types in predicates.items():
        types_str = ", ".join(types)
        line = f"{func_name} = Function('{func_name}', {types_str})"
        code_lines.append(line)
    return "\n".join(code_lines)

def generate_rollout(llm, args, context, question, options):
    with open("/home/chenzhb/Workspaces/verl/mcts_utils/prompts/generate.txt", "r", encoding='utf-8') as f:
        template = f.read()
    prompt = Template(template).safe_substitute(context=context, question=question, options=options)
    gen_res = llm.generate(data=prompt, args=args)
    return gen_res

def get_step_list(text_content):
    Steps = []
    pattern = r"<step.*?>(.*?)</step>"
    matches = re.findall(pattern, text_content, flags=re.DOTALL)
    for i, content in enumerate(matches, 1):
        clean_content = content.strip()
        Steps.append(clean_content)
    return Steps

def get_premise_conclusion(step_content):
    premise_list = []
    pattern = r"<premise>(.*?)</premise>"
    matches = re.findall(pattern, step_content, flags=re.DOTALL)
    if matches:
        for i, content in enumerate(matches, 1):
            clean_content = content.strip()
            premise_list.append(clean_content)

    pattern = r"<conclusion>(.*?)</conclusion>"
    matches = re.findall(pattern, step_content, flags=re.DOTALL)
    conclusion = matches[-1] if matches else None
    return premise_list, conclusion

def extract_python_block(code):
    py_pattern = r"```python\s+(.*?)```"
    clean_code = re.findall(py_pattern, code, re.DOTALL)[-1]
    return clean_code


def translate_step_to_z3(rephrase_res, declaration_code, step_content):
    with open("/home/chenzhb/Workspaces/verl/mcts_utils/prompts/translate_step.txt", "r", encoding='utf-8') as f:
        template = f.read()
    prompt = Template(template).safe_substitute(context=rephrase_res, declaration=declaration_code, step=step_content)
    trans_output = llm.generate(data=prompt, args=args)
    trans_code = extract_python_block(trans_output)
    z3_code_exe = wrap_z3_code(declaration_code, trans_code)
    return z3_code_exe

def wrap_z3_code(declaration, expression):
    z3_code = ""
    z3_code += "from z3 import *\n\n"
    z3_code += "s = Solver()\n\n"
    z3_code += "s.reset()"
    z3_code += "# --- Declarations ---\n\n"
    z3_code += declaration + "\n\n"
    z3_code += "# --- Expressions ---\n\n"
    z3_code += expression + "\n\n"
    z3_code += "s.add(premise_fol)\n\n"
    z3_code += "s.add(conclusion_fol)\n\n"
    z3_code += "result = s.check()\n"
    z3_code += "print(f'Result: {result}')\n"
    z3_code += "if result == sat:\n"
    z3_code += "    print('Model:', s.model())\n"
    return z3_code

def run_code(code_string):
    try:
        # 使用当前 python 解释器 (sys.executable) 执行代码
        # capture_output=True 会同时捕获 stdout 和 stderr
        # text=True 会让结果以字符串形式返回，而不是字节
        result = subprocess.run(
            [sys.executable, "-c", code_string],
            capture_output=True,
            text=True,
            timeout=10 # 设置超时防止死循环
        )
        
        if result.returncode == 0:
            return {"success": True, "output": result.stdout, "error": None}
        else:
            # 如果 returncode != 0，说明出错了
            # stderr 通常包含 Traceback
            return {"success": False, "output": result.stdout, "error": result.stderr}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "RunTimeError: 代码执行超时！"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def correct_z3_code(llm, args, code, error):
    with open("/home/chenzhb/Workspaces/verl/mcts_utils/prompts/correct_code.txt", "r", encoding='utf-8') as f:
        z3_code_correction_template = f.read()
    fix_bug_prompt = z3_code_correction_template
    fix_prompt = Template(fix_bug_prompt).safe_substitute(code=code, error=error)
    fix_out = llm.generate(data=fix_prompt, args=args)
    print(fix_out)
    pattern = r"```python\s+(.*?)```"
    matches = re.findall(pattern, fix_out, re.DOTALL)
    return matches[-1]

def correct_loop(llm, args, code):
    res = run_code(code)
    print(res)
    max_tries = 8
    tries = 0
    while not res["success"] and tries < max_tries:
        code = correct_z3_code(llm, args, code, res["error"])
        res = run_code(code)
        print(res)
        tries += 1
        args["temperature"] += 0.1  # 每次迭代增加温度，鼓励模型生成更多样化的修正方案
    return res["output"]


def main(llm, args):
    context = "Zhao Ming, Qian Hong and Sun Jie were admitted to Peking University, Tsinghua University and Beijing Normal University. Which schools were they admitted to? The students made the following guesses? Classmate A guessed? Zhao Ming was admitted to Tsinghua University and Sun Jie was admitted to Beijing Normal University. Student B guess? Zhao Ming was admitted to Beijing Normal University, Qian Hong was admitted to Tsinghua University. Student C guess? Zhao Ming was admitted to Peking University, Sun Jie was admitted to Tsinghua University. As a result, the students' guesses were half correct."
    question="Well, their admission status is."

    options="""
    Option (A):Zhao Ming, Qian Hong and Sun Jie were accepted by Peking University, Tsinghua University and Beijing Normal University respectively.

    Option (B):Zhao Ming, Qian Hong and Sun Jie were admitted to Tsinghua University, Beijing Normal University and Peking University respectively.

    Option (C):Zhao Ming, Qian Hong and Sun Jie were admitted to Beijing Normal University, Tsinghua University and Peking University respectively.

    Option (D):Zhao Ming, Qian Hong and Sun Jie were accepted by Peking University, Beijing Normal University and Tsinghua University respectively.
    """
    rephrased_context = rephrase(llm, args, context, question, options)
    objectives = object_extract(llm, args, context, question, options)
    predicates = predicate_extract(llm, args, context, question, options, objectives)
    z3_code = generate_z3_declarations(objectives)
    z3_function_code = generate_z3_functions(predicates)
    generations = generate_rollout(llm, args, context, question, options)
    declaration_code = z3_code + z3_function_code
    step_list = get_step_list(generations)
    # for i, step in enumerate(step_list, 1):
    #     print(f"### Step {i}: ### ")
    #     premise_list, conclusion = get_premise_conclusion(step)
    #     print(f"  Premises: {premise_list}")
    #     print(f"  Conclusion: {conclusion}")
    exe_code = translate_step_to_z3(rephrased_context, declaration_code, step_list[0])
    result = correct_loop(llm, args, exe_code)
    print(f"Final Result: {result}")





if __name__ == '__main__':
    args = {
        "max_tokens": 4096,
        "temperature": 0.1,
        "top_p": 0.8,
    }
    llm = LLM()
    main(llm, args)
